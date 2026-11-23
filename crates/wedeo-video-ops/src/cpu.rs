//! Data-Oriented CPU `SwsOpList` backend with `256×16` RGBA `f32` L1/L2 tile-fused
//! scratch execution and zero per-frame scratch heap allocations.

use std::sync::Arc;

use wedeo_core::{
    AlphaMode, BorderPadding, BorderedPlane, BufferPool, ChromaLocation, ColorModel, CompOrder,
    CropRect, Endian, Error, MemoryDomain, PixelLayout, PlanePacking, PositionedPlane, Result,
    SampleDepth, SideData, Subsampling, VideoFrame,
};

use crate::color;
use crate::ops::{DisplayTransform, DitherMode, SwsOp, SwsOpList, SwsScaler, layout_shift};
use crate::optimizer;

/// Tile width in pixels (`256 × 16 × 16 B = 64 KiB` L1/L2 scratch slab).
pub const TILE_W: usize = 256;
/// Tile height in rows (`256 × 16 × 16 B = 64 KiB` L1/L2 scratch slab).
pub const TILE_H: usize = 16;
const SLAB_PIXELS: usize = TILE_W * TILE_H;

/// Execution backend compiling an `SwsOpList` into a reusable `CompiledSws` pipeline.
pub trait SwsBackend: Send + Sync {
    fn compile(&self, ops: &SwsOpList) -> Result<Box<dyn CompiledSws>>;
}

/// Compiled, zero-steady-state-scratch-alloc video scaling/color/overlay pipeline.
pub trait CompiledSws: Send {
    fn execute(&mut self, src: &VideoFrame, dst_pool: &Arc<BufferPool>) -> Result<VideoFrame>;

    /// Composite positioned overlays (e.g., bitmap subtitles) onto `dst` in-place.
    fn blend_overlay(
        &mut self,
        dst: &mut VideoFrame,
        overlays: &[PositionedPlane],
        dst_pool: &Arc<BufferPool>,
    ) -> Result<()>;
}

/// Pure-Rust CPU scalar reference backend with `256×16` tile-fused execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuSwsBackend;

impl CpuSwsBackend {
    pub const fn new() -> Self {
        Self
    }
}

impl SwsBackend for CpuSwsBackend {
    fn compile(&self, ops: &SwsOpList) -> Result<Box<dyn CompiledSws>> {
        let compiled = CpuCompiledSws::compile(ops)?;
        Ok(Box::new(compiled))
    }
}

/// Pre-computed 1D separable resampling filter kernel.
#[derive(Debug, Clone)]
struct FilterKernel {
    taps: usize,
    offsets: Vec<usize>,
    weights: Vec<f32>,
}

impl FilterKernel {
    fn generate(src_size: u32, dst_size: u32, scaler: SwsScaler) -> Self {
        let src_n = src_size as usize;
        let dst_n = dst_size as usize;
        let ratio = dst_n as f64 / src_n as f64;
        let stretch = if ratio < 1.0 && scaler != SwsScaler::Point {
            1.0 / ratio
        } else {
            1.0
        };

        let base_radius = match scaler {
            SwsScaler::Point => 0.5,
            SwsScaler::Bilinear => 1.0 - 1e-6,
            SwsScaler::Bicubic => 2.0 - 1e-6,
            SwsScaler::Lanczos => 3.0 - 1e-6,
        };
        let radius = if scaler == SwsScaler::Point {
            0.5
        } else {
            base_radius * stretch
        };

        let taps = if scaler == SwsScaler::Point {
            1
        } else {
            ((radius * 2.0).ceil() as usize).clamp(1, src_n)
        };

        let mut offsets = vec![0usize; dst_n];
        let mut weights = vec![0.0f32; dst_n * taps];

        for dst_pos in 0..dst_n {
            let src_pos = (dst_pos as f64 + 0.5) / ratio - 0.5;
            if taps == 1 {
                let idx = src_pos.round().clamp(0.0, (src_n - 1) as f64) as usize;
                offsets[dst_pos] = idx;
                weights[dst_pos] = 1.0;
                continue;
            }

            let start_pos = src_pos - radius;
            let start_idx = (start_pos.ceil() as isize).clamp(0, (src_n - taps) as isize) as usize;
            offsets[dst_pos] = start_idx;

            let row_w = &mut weights[dst_pos * taps..(dst_pos + 1) * taps];
            let mut sum = 0.0f64;
            for (i, w_out) in row_w.iter_mut().enumerate() {
                let x = ((start_idx + i) as f64 - src_pos).abs() / stretch;
                let w = sample_kernel(scaler, x);
                sum += w;
                *w_out = w as f32;
            }
            if sum.abs() > 1e-12 {
                let inv = (1.0 / sum) as f32;
                for w_out in row_w.iter_mut() {
                    *w_out *= inv;
                }
            } else {
                row_w[0] = 1.0;
            }
        }

        Self {
            taps,
            offsets,
            weights,
        }
    }
}

#[inline]
fn sample_kernel(scaler: SwsScaler, x: f64) -> f64 {
    match scaler {
        SwsScaler::Point => {
            if x <= 0.5 {
                1.0
            } else {
                0.0
            }
        }
        SwsScaler::Bilinear => (1.0 - x).max(0.0),
        SwsScaler::Bicubic => {
            // Exact FFmpeg n9.1 filters.c cubic(x, B=0.0, C=0.6)
            const B: f64 = 0.0;
            const C: f64 = 0.6;
            if x >= 2.0 {
                return 0.0;
            }
            let p0 = 6.0 - 2.0 * B;
            if x < 1.0 {
                let p2 = -18.0 + 12.0 * B + 6.0 * C;
                let p3 = 12.0 - 9.0 * B - 6.0 * C;
                (p0 + x * x * (p2 + x * p3)) / p0
            } else {
                let q0 = 8.0 * B + 24.0 * C;
                let q1 = -12.0 * B - 48.0 * C;
                let q2 = 6.0 * B + 30.0 * C;
                let q3 = -B - 6.0 * C;
                (q0 + x * (q1 + x * (q2 + x * q3))) / p0
            }
        }
        SwsScaler::Lanczos => {
            if x >= 3.0 {
                return 0.0;
            }
            sinc(x) * sinc(x / 3.0)
        }
    }
}

#[inline]
fn sinc(x: f64) -> f64 {
    if x < 1e-8 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Generate the 16×16 Bayer matrix from FFmpeg `libswscale/format.c` `generate_bayer_matrix(4)`,
/// centered around `0.0` (values in `(-0.5, +0.5)`).
fn build_bayer_16x16() -> [[f32; 16]; 16] {
    let mut m = [[0i32; 16]; 16];
    let mut sz = 1usize;
    while sz < 16 {
        let mult = 4;
        for y in 0..sz {
            for x in 0..sz {
                let v = m[y][x] * mult;
                m[y][x] = v;
                m[y + sz][x + sz] = v + 1;
                m[y][x + sz] = v + 2;
                m[y + sz][x] = v + 3;
            }
        }
        sz <<= 1;
    }
    let mut out = [[0.0f32; 16]; 16];
    for (y, row) in m.iter().enumerate() {
        for (x, &v) in row.iter().enumerate() {
            // (v + 0.5) / 256.0 - 0.5 -> zero-mean symmetric threshold offset
            out[y][x] = (v as f32 + 0.5) / 256.0 - 0.5;
        }
    }
    out
}

///Compiled CPU pipeline executing a validated, optimized `SwsOpList`.
pub struct CpuCompiledSws {
    ops: SwsOpList,
    is_noop: bool,
    read_layout: PixelLayout,
    read_crop: CropRect,
    display_tx: DisplayTransform,
    oriented_w: u32,
    oriented_h: u32,
    write_layout: PixelLayout,
    pre_scale_ops: Vec<SwsOp>,
    between_scale_ops: Vec<SwsOp>,
    post_scale_ops: Vec<SwsOp>,
    h_kernel: Option<FilterKernel>,
    v_kernel: Option<FilterKernel>,
    /// Reusable 64 KB (`256×16` RGBA `f32`) L1/L2 scratch slab (zero per-frame allocs).
    slab: Vec<[f32; 4]>,
    /// Reusable single-row buffer for horizontal filtering (`oriented_w` pixels).
    h_row_buf: Vec<[f32; 4]>,
    /// Reusable ring buffer for vertical filtering (`v_ring_rows * dst_w` pixels).
    v_ring: Vec<[f32; 4]>,
    v_ring_rows: usize,
    v_ring_tags: Vec<i64>,
    /// Pre-allocated Floyd-Steinberg error-diffusion carry rows (`dst_w + 2` pixels each).
    err_band: Vec<[f32; 4]>,
    bayer: [[f32; 16]; 16],
    dst_quant_max: f32,
    src_peak_nits: f32,
    dst_peak_nits: f32,
}

impl CpuCompiledSws {
    pub fn compile(op_list: &SwsOpList) -> Result<Self> {
        if op_list.src.domain != MemoryDomain::Cpu || op_list.dst.domain != MemoryDomain::Cpu {
            return Err(Error::unsupported(
                "wedeo-video-ops",
                "CpuSwsBackend supports MemoryDomain::Cpu frames only",
            ));
        }

        let mut ops = op_list.clone();
        optimizer::optimize(&mut ops);

        let is_noop = optimizer::is_noop(&ops);

        let Some(SwsOp::Read {
            layout: read_layout,
            crop: read_crop,
            display_matrix,
        }) = ops.ops.first().cloned()
        else {
            return Err(Error::invalid_argument(
                "wedeo-video-ops",
                "SwsOpList must begin with SwsOp::Read",
            ));
        };
        let Some(SwsOp::Write(write_layout)) = ops.ops.last().cloned() else {
            return Err(Error::invalid_argument(
                "wedeo-video-ops",
                "SwsOpList must end with SwsOp::Write",
            ));
        };

        validate_layout(read_layout)?;
        validate_layout(write_layout)?;

        let display_tx = DisplayTransform::from_matrix(&display_matrix)?;
        let (cw, ch) = read_crop.display_size(ops.src.width, ops.src.height);
        let (oriented_w, oriented_h) = display_tx.output_dims(cw, ch);

        let mut pre_scale_ops = Vec::new();
        let mut between_scale_ops = Vec::new();
        let mut post_scale_ops = Vec::new();
        let mut h_kernel = None;
        let mut v_kernel = None;
        let mut seen_h = false;
        let mut seen_v = false;
        let mut has_error_diffusion = false;

        let middle = &ops.ops[1..ops.ops.len() - 1];
        for op in middle {
            match op {
                SwsOp::ScaleH { dst_w, kernel, .. } => {
                    h_kernel = Some(FilterKernel::generate(oriented_w, *dst_w, *kernel));
                    seen_h = true;
                }
                SwsOp::ScaleV { dst_h, kernel, .. } => {
                    v_kernel = Some(FilterKernel::generate(oriented_h, *dst_h, *kernel));
                    seen_v = true;
                }
                other => {
                    if matches!(other, SwsOp::Dither(DitherMode::ErrorDiffusion)) {
                        has_error_diffusion = true;
                    }
                    if seen_v {
                        post_scale_ops.push(other.clone());
                    } else if seen_h {
                        between_scale_ops.push(other.clone());
                    } else {
                        pre_scale_ops.push(other.clone());
                    }
                }
            }
        }

        // If only ScaleH is present (no ScaleV), treat between_scale_ops as post_scale_ops
        if seen_h && !seen_v {
            post_scale_ops = between_scale_ops;
            between_scale_ops = Vec::new();
        }

        let dst_w = ops.dst.width as usize;
        let h_row_buf = if h_kernel.is_some() {
            vec![[0.0; 4]; oriented_w as usize]
        } else {
            Vec::new()
        };

        let (v_ring, v_ring_rows, v_ring_tags) = if let Some(ref vk) = v_kernel {
            let rows = (vk.taps + TILE_H).max(vk.taps * 2);
            (vec![[0.0; 4]; rows * dst_w], rows, vec![-1i64; rows])
        } else {
            (Vec::new(), 0, Vec::new())
        };

        let err_band = if has_error_diffusion {
            vec![[0.0; 4]; (TILE_H + 1) * (dst_w + 2)]
        } else {
            Vec::new()
        };

        let dst_quant_max = match write_layout.depth {
            SampleDepth::F32 | SampleDepth::F16 => 1.0,
            d => ((1u32 << d.bits()) - 1) as f32,
        };
        let src_peak_nits =
            color::effective_src_peak_nits(ops.src.color.transfer, ops.options.src_peak_nits);
        let dst_peak_nits = ops.options.dst_peak_nits.max(1.0);

        Ok(Self {
            ops,
            is_noop,
            read_layout,
            read_crop,
            display_tx,
            oriented_w,
            oriented_h,
            write_layout,
            pre_scale_ops,
            between_scale_ops,
            post_scale_ops,
            h_kernel,
            v_kernel,
            slab: vec![[0.0; 4]; SLAB_PIXELS],
            h_row_buf,
            v_ring,
            v_ring_rows,
            v_ring_tags,
            err_band,
            bayer: build_bayer_16x16(),
            dst_quant_max,
            src_peak_nits,
            dst_peak_nits,
        })
    }

    fn alloc_dst_planes(&self, dst_pool: &Arc<BufferPool>) -> Result<Vec<BorderedPlane>> {
        let count = self.write_layout.plane_count();
        let mut planes = Vec::with_capacity(count);
        for p in 0..count {
            let Some((wb, hr)) =
                self.write_layout
                    .plane_dims(self.ops.dst.width, self.ops.dst.height, p)
            else {
                return Err(Error::internal(
                    "wedeo-video-ops",
                    "invalid destination plane index",
                ));
            };
            planes.push(BorderedPlane::new_reused(
                dst_pool,
                wb,
                hr,
                BorderPadding::NONE,
            ));
        }
        Ok(planes)
    }
}

/// Immutable execution parameters passed to `apply_pointwise_slice`.
#[derive(Clone, Copy)]
struct PointwiseCtx {
    tile_x: usize,
    tile_y: usize,
    tile_w: usize,
    dst_w: usize,
    dst_quant_max: f32,
    src_peak_nits: f32,
    dst_peak_nits: f32,
    has_alpha: bool,
    is_gray: bool,
}

impl CpuCompiledSws {
    fn pointwise_ctx(
        &self,
        tile_x: usize,
        tile_y: usize,
        tile_w: usize,
        dst_w: usize,
    ) -> PointwiseCtx {
        PointwiseCtx {
            tile_x,
            tile_y,
            tile_w,
            dst_w,
            dst_quant_max: self.dst_quant_max,
            src_peak_nits: self.src_peak_nits,
            dst_peak_nits: self.dst_peak_nits,
            has_alpha: !matches!(self.read_layout.alpha, AlphaMode::None),
            is_gray: self.read_layout.model == ColorModel::Gray,
        }
    }

    fn apply_pointwise_slice(
        ops: &[SwsOp],
        pixels: &mut [[f32; 4]],
        ctx: PointwiseCtx,
        bayer: &[[f32; 16]; 16],
        err_band: &mut [[f32; 4]],
    ) {
        let tile_x = ctx.tile_x;
        let tile_y = ctx.tile_y;
        let tile_w = ctx.tile_w;
        let dst_w = ctx.dst_w;
        let dst_quant_max = ctx.dst_quant_max;
        let src_peak_nits = ctx.src_peak_nits;
        let dst_peak_nits = ctx.dst_peak_nits;
        let has_alpha = ctx.has_alpha;
        let is_gray = ctx.is_gray;
        for op in ops {
            match *op {
                SwsOp::Unpack { depth, shift } => {
                    let max_val = match depth {
                        SampleDepth::F32 | SampleDepth::F16 => 1.0,
                        d => ((1u32 << d.bits()) - 1) as f32,
                    };
                    let inv = 1.0 / max_val;
                    if shift == 0 {
                        for px in pixels.iter_mut() {
                            px[0] *= inv;
                            px[1] *= inv;
                            px[2] *= inv;
                            px[3] = if has_alpha { px[3] * inv } else { 1.0 };
                        }
                    } else {
                        for px in pixels.iter_mut() {
                            px[0] = ((px[0] as u32 >> shift) as f32) * inv;
                            px[1] = ((px[1] as u32 >> shift) as f32) * inv;
                            px[2] = ((px[2] as u32 >> shift) as f32) * inv;
                            px[3] = if has_alpha {
                                ((px[3] as u32 >> shift) as f32) * inv
                            } else {
                                1.0
                            };
                        }
                    }
                }
                SwsOp::Swizzle(s) => {
                    for px in pixels.iter_mut() {
                        let orig = *px;
                        px[0] = orig[s[0] as usize];
                        px[1] = orig[s[1] as usize];
                        px[2] = orig[s[2] as usize];
                        px[3] = orig[s[3] as usize];
                    }
                }
                SwsOp::ColorMatrix(m) => {
                    for px in pixels.iter_mut() {
                        let x0 = px[0];
                        let x1 = px[1];
                        let x2 = px[2];
                        px[0] = m[0][0] * x0 + m[0][1] * x1 + m[0][2] * x2 + m[0][3];
                        px[1] = m[1][0] * x0 + m[1][1] * x1 + m[1][2] * x2 + m[1][3];
                        px[2] = m[2][0] * x0 + m[2][1] * x1 + m[2][2] * x2 + m[2][3];
                    }
                }
                SwsOp::Linearize(trc) => {
                    for px in pixels.iter_mut() {
                        let lin = color::linearize_rgb(trc, [px[0], px[1], px[2]]);
                        px[0] = lin[0];
                        px[1] = lin[1];
                        px[2] = lin[2];
                    }
                }
                SwsOp::ToneMap(tm) => {
                    for px in pixels.iter_mut() {
                        let mapped = color::tone_map_rgb(
                            tm,
                            [px[0], px[1], px[2]],
                            src_peak_nits,
                            dst_peak_nits,
                        );
                        px[0] = mapped[0];
                        px[1] = mapped[1];
                        px[2] = mapped[2];
                    }
                }
                SwsOp::BlendAlpha => {
                    for px in pixels.iter_mut() {
                        let a = px[3].clamp(0.0, 1.0);
                        px[0] *= a;
                        px[1] *= a;
                        px[2] *= a;
                        px[3] = 1.0;
                    }
                }
                SwsOp::Delinearize(trc) => {
                    for px in pixels.iter_mut() {
                        let nl = color::delinearize_rgb(trc, [px[0], px[1], px[2]]);
                        px[0] = nl[0];
                        px[1] = nl[1];
                        px[2] = nl[2];
                    }
                }
                SwsOp::Dither(DitherMode::Ordered) => {
                    // Brute-forced 16×16 offsets from FFmpeg format.c:1445
                    const OFF: [usize; 4] = [0, 3, 2, 5];
                    let inv_q = 1.0 / dst_quant_max;
                    let th = pixels.len() / tile_w;
                    for ly in 0..th {
                        let gy = tile_y + ly;
                        let row = &mut pixels[ly * tile_w..(ly + 1) * tile_w];
                        for (lx, px) in row.iter_mut().enumerate() {
                            let gx = (tile_x + lx) & 15;
                            if is_gray {
                                let d = bayer[gy & 15][gx] * inv_q;
                                px[0] += d;
                                px[1] += d;
                                px[2] += d;
                            } else {
                                px[0] += bayer[(gy + OFF[0]) & 15][gx] * inv_q;
                                px[1] += bayer[(gy + OFF[1]) & 15][gx] * inv_q;
                                px[2] += bayer[(gy + OFF[2]) & 15][gx] * inv_q;
                            }
                            px[3] += bayer[(gy + OFF[3]) & 15][gx] * inv_q;
                        }
                    }
                }
                SwsOp::Dither(DitherMode::ErrorDiffusion) => {
                    let stride = dst_w + 2;
                    let th = pixels.len() / tile_w;
                    let ch_limit = if is_gray { 1 } else { 3 };
                    for ly in 0..th {
                        let row = &mut pixels[ly * tile_w..(ly + 1) * tile_w];
                        let (cur_slice, nxt_slice) = err_band.split_at_mut((ly + 1) * stride);
                        let cur_row = &mut cur_slice[ly * stride..(ly + 1) * stride];
                        let nxt_row = &mut nxt_slice[..stride];
                        for (lx, px) in row.iter_mut().enumerate() {
                            let ex = tile_x + lx + 1;
                            for c in 0..ch_limit {
                                let v = (px[c] + cur_row[ex][c]).clamp(0.0, 1.0);
                                let q = (v * dst_quant_max).round().clamp(0.0, dst_quant_max)
                                    / dst_quant_max;
                                let err = v - q;
                                px[c] = q;
                                cur_row[ex + 1][c] += err * (7.0 / 16.0);
                                nxt_row[ex - 1][c] += err * (3.0 / 16.0);
                                nxt_row[ex][c] += err * (5.0 / 16.0);
                                nxt_row[ex + 1][c] += err * (1.0 / 16.0);
                            }
                        }
                    }
                }
                SwsOp::Pack(dst_layout) => {
                    let shift = layout_shift(dst_layout);
                    let max_val = match dst_layout.depth {
                        SampleDepth::F32 | SampleDepth::F16 => 1.0,
                        d => ((1u32 << d.bits()) - 1) as f32,
                    };
                    if matches!(dst_layout.depth, SampleDepth::F32) {
                        for px in pixels.iter_mut() {
                            px[0] = px[0].clamp(0.0, 1.0);
                            px[1] = px[1].clamp(0.0, 1.0);
                            px[2] = px[2].clamp(0.0, 1.0);
                            px[3] = px[3].clamp(0.0, 1.0);
                        }
                    } else if shift == 0 {
                        for px in pixels.iter_mut() {
                            px[0] = (px[0] * max_val).round().clamp(0.0, max_val);
                            px[1] = (px[1] * max_val).round().clamp(0.0, max_val);
                            px[2] = (px[2] * max_val).round().clamp(0.0, max_val);
                            px[3] = (px[3] * max_val).round().clamp(0.0, max_val);
                        }
                    } else {
                        for px in pixels.iter_mut() {
                            px[0] = (((px[0] * max_val).round().clamp(0.0, max_val) as u32)
                                << shift) as f32;
                            px[1] = (((px[1] * max_val).round().clamp(0.0, max_val) as u32)
                                << shift) as f32;
                            px[2] = (((px[2] * max_val).round().clamp(0.0, max_val) as u32)
                                << shift) as f32;
                            px[3] = (((px[3] * max_val).round().clamp(0.0, max_val) as u32)
                                << shift) as f32;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn execute_no_scale(
        &mut self,
        src: &VideoFrame,
        dst_planes: &mut [BorderedPlane],
    ) -> Result<()> {
        let src_planes = src
            .planes()
            .ok_or_else(|| Error::unsupported("wedeo-video-ops", "expected CPU source frame"))?;
        let dst_w = self.ops.dst.width as usize;
        let dst_h = self.ops.dst.height as usize;
        let has_err = !self.err_band.is_empty();
        if has_err {
            self.err_band.fill([0.0; 4]);
        }
        let has_alpha = !matches!(self.read_layout.alpha, AlphaMode::None);
        let is_gray = self.read_layout.model == ColorModel::Gray;

        for ty in (0..dst_h).step_by(TILE_H) {
            let th = (dst_h - ty).min(TILE_H);
            if has_err && ty > 0 {
                let stride = dst_w + 2;
                self.err_band
                    .copy_within(TILE_H * stride..(TILE_H + 1) * stride, 0);
                self.err_band[stride..].fill([0.0; 4]);
            }
            for tx in (0..dst_w).step_by(TILE_W) {
                let tw = (dst_w - tx).min(TILE_W);
                let ctx = self.pointwise_ctx(tx, ty, tw, dst_w);
                let tile = &mut self.slab[..tw * th];
                read_tile(
                    src_planes,
                    src.width,
                    src.height,
                    self.read_layout,
                    self.ops.src.color.chroma_location,
                    self.read_crop,
                    self.display_tx,
                    self.oriented_w,
                    self.oriented_h,
                    tx,
                    ty,
                    tw,
                    th,
                    tile,
                )?;
                Self::apply_pointwise_slice(
                    &self.pre_scale_ops,
                    tile,
                    ctx,
                    &self.bayer,
                    &mut self.err_band,
                );
                write_tile(
                    dst_planes,
                    dst_w as u32,
                    dst_h as u32,
                    self.write_layout,
                    tx,
                    ty,
                    tw,
                    th,
                    tile,
                )?;
            }
        }
        let _ = (has_alpha, is_gray);
        Ok(())
    }

    fn execute_scale_h_only(
        &mut self,
        src: &VideoFrame,
        dst_planes: &mut [BorderedPlane],
    ) -> Result<()> {
        let src_planes = src
            .planes()
            .ok_or_else(|| Error::unsupported("wedeo-video-ops", "expected CPU source frame"))?;
        let dst_w = self.ops.dst.width as usize;
        let dst_h = self.ops.dst.height as usize;
        let ow = self.oriented_w as usize;
        let hk = self.h_kernel.as_ref().expect("h_kernel checked");
        let has_err = !self.err_band.is_empty();
        if has_err {
            self.err_band.fill([0.0; 4]);
        }
        let has_alpha = !matches!(self.read_layout.alpha, AlphaMode::None);
        let is_gray = self.read_layout.model == ColorModel::Gray;

        for ty in (0..dst_h).step_by(TILE_H) {
            let th = (dst_h - ty).min(TILE_H);
            if has_err && ty > 0 {
                let stride = dst_w + 2;
                self.err_band
                    .copy_within(TILE_H * stride..(TILE_H + 1) * stride, 0);
                self.err_band[stride..].fill([0.0; 4]);
            }
            for tx in (0..dst_w).step_by(TILE_W) {
                let tw = (dst_w - tx).min(TILE_W);
                let sx_min = hk.offsets[tx];
                let sx_max = (hk.offsets[tx + tw - 1] + hk.taps).min(ow);
                let span = sx_max - sx_min;

                for r in 0..th {
                    let y = ty + r;
                    let pre_ctx = self.pointwise_ctx(sx_min, y, span, ow);
                    let src_span = &mut self.h_row_buf[sx_min..sx_max];
                    read_tile(
                        src_planes,
                        src.width,
                        src.height,
                        self.read_layout,
                        self.ops.src.color.chroma_location,
                        self.read_crop,
                        self.display_tx,
                        self.oriented_w,
                        self.oriented_h,
                        sx_min,
                        y,
                        span,
                        1,
                        src_span,
                    )?;
                    Self::apply_pointwise_slice(
                        &self.pre_scale_ops,
                        src_span,
                        pre_ctx,
                        &self.bayer,
                        &mut [],
                    );
                    let dst_row = &mut self.slab[r * tw..(r + 1) * tw];
                    convolve_h(hk, &self.h_row_buf, tx, tw, dst_row);
                }

                let post_ctx = self.pointwise_ctx(tx, ty, tw, dst_w);
                let tile = &mut self.slab[..tw * th];
                Self::apply_pointwise_slice(
                    &self.post_scale_ops,
                    tile,
                    post_ctx,
                    &self.bayer,
                    &mut self.err_band,
                );
                write_tile(
                    dst_planes,
                    dst_w as u32,
                    dst_h as u32,
                    self.write_layout,
                    tx,
                    ty,
                    tw,
                    th,
                    tile,
                )?;
            }
        }
        let _ = (has_alpha, is_gray);
        Ok(())
    }

    fn execute_scale_v(
        &mut self,
        src: &VideoFrame,
        dst_planes: &mut [BorderedPlane],
    ) -> Result<()> {
        let src_planes = src
            .planes()
            .ok_or_else(|| Error::unsupported("wedeo-video-ops", "expected CPU source frame"))?;
        let dst_w = self.ops.dst.width as usize;
        let dst_h = self.ops.dst.height as usize;
        let ow = self.oriented_w as usize;
        let vk = self.v_kernel.as_ref().expect("v_kernel checked").clone();
        self.v_ring_tags.fill(-1);
        let has_err = !self.err_band.is_empty();
        if has_err {
            self.err_band.fill([0.0; 4]);
        }
        let has_alpha = !matches!(self.read_layout.alpha, AlphaMode::None);
        let is_gray = self.read_layout.model == ColorModel::Gray;

        for ty in (0..dst_h).step_by(TILE_H) {
            let th = (dst_h - ty).min(TILE_H);
            if has_err && ty > 0 {
                let stride = dst_w + 2;
                self.err_band
                    .copy_within(TILE_H * stride..(TILE_H + 1) * stride, 0);
                self.err_band[stride..].fill([0.0; 4]);
            }

            // Populate all required source rows for this 16-row band in v_ring
            let sy_min = vk.offsets[ty];
            let sy_max = (vk.offsets[ty + th - 1] + vk.taps).min(self.oriented_h as usize);
            for sy in sy_min..sy_max {
                let slot = sy % self.v_ring_rows;
                if self.v_ring_tags[slot] == sy as i64 {
                    continue;
                }
                let pre_ctx = self.pointwise_ctx(0, sy, ow, ow);
                let bet_ctx = self.pointwise_ctx(0, sy, dst_w, dst_w);
                let ring_row = &mut self.v_ring[slot * dst_w..(slot + 1) * dst_w];
                if let Some(ref hk) = self.h_kernel {
                    // Read full source row into h_row_buf, run pre_scale_ops, then ScaleH -> ring_row
                    read_tile(
                        src_planes,
                        src.width,
                        src.height,
                        self.read_layout,
                        self.ops.src.color.chroma_location,
                        self.read_crop,
                        self.display_tx,
                        self.oriented_w,
                        self.oriented_h,
                        0,
                        sy,
                        ow,
                        1,
                        &mut self.h_row_buf[..ow],
                    )?;
                    Self::apply_pointwise_slice(
                        &self.pre_scale_ops,
                        &mut self.h_row_buf[..ow],
                        pre_ctx,
                        &self.bayer,
                        &mut [],
                    );
                    convolve_h(hk, &self.h_row_buf, 0, dst_w, ring_row);
                    Self::apply_pointwise_slice(
                        &self.between_scale_ops,
                        ring_row,
                        bet_ctx,
                        &self.bayer,
                        &mut [],
                    );
                } else {
                    // No ScaleH: ow == dst_w, read straight into ring_row
                    read_tile(
                        src_planes,
                        src.width,
                        src.height,
                        self.read_layout,
                        self.ops.src.color.chroma_location,
                        self.read_crop,
                        self.display_tx,
                        self.oriented_w,
                        self.oriented_h,
                        0,
                        sy,
                        dst_w,
                        1,
                        ring_row,
                    )?;
                    Self::apply_pointwise_slice(
                        &self.pre_scale_ops,
                        ring_row,
                        bet_ctx,
                        &self.bayer,
                        &mut [],
                    );
                }
                self.v_ring_tags[slot] = sy as i64;
            }

            // Convolve vertically across 256×16 output tiles + post_scale_ops + Write
            for tx in (0..dst_w).step_by(TILE_W) {
                let tw = (dst_w - tx).min(TILE_W);
                for r in 0..th {
                    let dy = ty + r;
                    let sy0 = vk.offsets[dy];
                    let kw = &vk.weights[dy * vk.taps..(dy + 1) * vk.taps];
                    let dst_row = &mut self.slab[r * tw..(r + 1) * tw];
                    dst_row.fill([0.0; 4]);
                    for (t, &w) in kw.iter().enumerate() {
                        let slot = (sy0 + t) % self.v_ring_rows;
                        let src_row = &self.v_ring[slot * dst_w + tx..slot * dst_w + tx + tw];
                        for (d, s) in dst_row.iter_mut().zip(src_row.iter()) {
                            d[0] += s[0] * w;
                            d[1] += s[1] * w;
                            d[2] += s[2] * w;
                            d[3] += s[3] * w;
                        }
                    }
                }
                let post_ctx = self.pointwise_ctx(tx, ty, tw, dst_w);
                let tile = &mut self.slab[..tw * th];
                Self::apply_pointwise_slice(
                    &self.post_scale_ops,
                    tile,
                    post_ctx,
                    &self.bayer,
                    &mut self.err_band,
                );
                write_tile(
                    dst_planes,
                    dst_w as u32,
                    dst_h as u32,
                    self.write_layout,
                    tx,
                    ty,
                    tw,
                    th,
                    tile,
                )?;
            }
        }
        let _ = (has_alpha, is_gray);
        Ok(())
    }
}

impl CompiledSws for CpuCompiledSws {
    fn execute(&mut self, src: &VideoFrame, dst_pool: &Arc<BufferPool>) -> Result<VideoFrame> {
        if src.domain() != MemoryDomain::Cpu {
            return Err(Error::unsupported(
                "wedeo-video-ops",
                "CpuCompiledSws expects MemoryDomain::Cpu input",
            ));
        }
        if src.width != self.ops.src.width
            || src.height != self.ops.src.height
            || src.layout != self.ops.src.layout
        {
            return Err(Error::invalid_argument(
                "wedeo-video-ops",
                "source VideoFrame dimensions/layout do not match compiled VideoContract",
            ));
        }

        if self.is_noop {
            return Ok(src.clone());
        }

        let mut dst_planes = self.alloc_dst_planes(dst_pool)?;
        if self.v_kernel.is_some() {
            self.execute_scale_v(src, &mut dst_planes)?;
        } else if self.h_kernel.is_some() {
            self.execute_scale_h_only(src, &mut dst_planes)?;
        } else {
            self.execute_no_scale(src, &mut dst_planes)?;
        }

        let mut out = VideoFrame::from_planes(
            self.ops.dst.width,
            self.ops.dst.height,
            self.write_layout,
            dst_planes,
        );
        out.pts = src.pts;
        out.pkt_dts = src.pkt_dts;
        out.duration = src.duration;
        out.time_base = src.time_base;
        out.picture_type = src.picture_type;
        out.flags = src.flags;
        out.repeat_pict = src.repeat_pict;
        out.sar = if self.display_tx.swaps_axes() && src.sar.num > 0 && src.sar.den > 0 {
            wedeo_core::Rational::new(src.sar.den, src.sar.num)
        } else {
            src.sar
        };
        out.crop = CropRect::ZERO;
        out.color = color::sanitize_output_color(self.write_layout, src.color, self.ops.dst.color);

        let tone_mapped = self.ops.src.color.transfer.is_hdr()
            && !out.color.transfer.is_hdr()
            && self.ops.options.tone_map != crate::ops::ToneMapOp::None;
        for sd in &src.side_data {
            match sd {
                SideData::DisplayMatrix(_) => {}
                SideData::MasteringDisplay(_)
                | SideData::ContentLightLevel { .. }
                | SideData::Hdr10Plus(_)
                    if tone_mapped => {}
                other => out.side_data.push(other.clone()),
            }
        }

        Ok(out)
    }

    fn blend_overlay(
        &mut self,
        dst: &mut VideoFrame,
        overlays: &[PositionedPlane],
        dst_pool: &Arc<BufferPool>,
    ) -> Result<()> {
        blend_overlays_in_place(dst, overlays, dst_pool)
    }
}

#[inline]
fn convolve_h(
    hk: &FilterKernel,
    src_row: &[[f32; 4]],
    tx: usize,
    tw: usize,
    dst_row: &mut [[f32; 4]],
) {
    let taps = hk.taps;
    for (lx, dst_px) in dst_row[..tw].iter_mut().enumerate() {
        let dx = tx + lx;
        let sx0 = hk.offsets[dx];
        let kw = &hk.weights[dx * taps..(dx + 1) * taps];
        let mut acc = [0.0f32; 4];
        for (i, &w) in kw.iter().enumerate() {
            let s = src_row[sx0 + i];
            acc[0] += s[0] * w;
            acc[1] += s[1] * w;
            acc[2] += s[2] * w;
            acc[3] += s[3] * w;
        }
        *dst_px = acc;
    }
}

/// Composite a slice of `PositionedPlane` overlays onto `dst` in-place (with CoW via `dst_pool`).
pub fn blend_overlays_in_place(
    dst: &mut VideoFrame,
    overlays: &[PositionedPlane],
    dst_pool: &Arc<BufferPool>,
) -> Result<()> {
    if overlays.is_empty() {
        return Ok(());
    }
    if dst.domain() != MemoryDomain::Cpu {
        return Err(Error::unsupported(
            "wedeo-video-ops",
            "blend_overlay supports MemoryDomain::Cpu frames only",
        ));
    }
    let dst_w = dst.width;
    let dst_h = dst.height;
    let dst_layout = dst.layout;
    let dst_color = color::resolve_color(dst_layout, dst.color, None);
    validate_layout(dst_layout)?;

    // Ensure all destination planes are uniquely writable (CoW from pool if shared).
    {
        let planes = dst.planes_mut().ok_or_else(|| {
            Error::unsupported("wedeo-video-ops", "expected CPU destination frame")
        })?;
        for plane in planes.iter_mut() {
            if plane.buffer.make_writable(dst_pool).is_none() {
                return Err(Error::internal(
                    "wedeo-video-ops",
                    "failed to acquire writable CPU plane",
                ));
            }
        }
    }

    let dst_max_val = match dst_layout.depth {
        SampleDepth::F32 | SampleDepth::F16 => 1.0,
        d => ((1u32 << d.bits()) - 1) as f32,
    };
    let dst_shift = layout_shift(dst_layout);
    let dst_to_write_swiz = crate::ops::canonical_to_write_swizzle(dst_layout);
    let dst_from_read_swiz = crate::ops::read_to_canonical_swizzle(dst_layout);
    let rgb_to_dst_mat = color::encode_from_rgb_matrix(dst_layout, dst_color)?;

    for ov in overlays {
        validate_layout(ov.layout)?;
        let x0 = ov.x.min(dst_w);
        let y0 = ov.y.min(dst_h);
        let x1 = ov.x.saturating_add(ov.width).min(dst_w);
        let y1 = ov.y.saturating_add(ov.height).min(dst_h);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        let ov_planes = std::slice::from_ref(&ov.plane);
        let ov_premul = matches!(ov.layout.alpha, AlphaMode::Premultiplied);
        let dst_has_alpha = !matches!(dst_layout.alpha, AlphaMode::None);

        let dst_planes = dst.planes_mut().expect("verified CPU");

        if matches!(dst_layout.model, ColorModel::Rgb | ColorModel::Gray) {
            for dy in y0..y1 {
                let oy = (dy - ov.y) as usize;
                for dx in x0..x1 {
                    let ox = (dx - ov.x) as usize;
                    let raw_ov = read_pixel_direct(
                        ov_planes,
                        ov.width,
                        ov.height,
                        ov.layout,
                        ChromaLocation::Left,
                        ox,
                        oy,
                    )?;
                    let rgba = color::unpack_alpha_overlay_pixel(
                        ov.layout,
                        raw_ov,
                        color::resolve_color(ov.layout, wedeo_core::ColorDesc::SRGB, None),
                    );
                    let a = rgba[3];
                    if a <= 0.0 {
                        continue;
                    }
                    let inv_a = 1.0 - a;
                    let ov_canon = [
                        rgb_to_dst_mat[0][0] * rgba[0]
                            + rgb_to_dst_mat[0][1] * rgba[1]
                            + rgb_to_dst_mat[0][2] * rgba[2]
                            + rgb_to_dst_mat[0][3],
                        rgb_to_dst_mat[1][0] * rgba[0]
                            + rgb_to_dst_mat[1][1] * rgba[1]
                            + rgb_to_dst_mat[1][2] * rgba[2]
                            + rgb_to_dst_mat[1][3],
                        rgb_to_dst_mat[2][0] * rgba[0]
                            + rgb_to_dst_mat[2][1] * rgba[1]
                            + rgb_to_dst_mat[2][2] * rgba[2]
                            + rgb_to_dst_mat[2][3],
                        a,
                    ];
                    let raw_dst = read_pixel_direct(
                        dst_planes,
                        dst_w,
                        dst_h,
                        dst_layout,
                        dst_color.chroma_location,
                        dx as usize,
                        dy as usize,
                    )?;
                    let dst_norm = [
                        ((raw_dst[0] as u32 >> dst_shift) as f32) / dst_max_val,
                        ((raw_dst[1] as u32 >> dst_shift) as f32) / dst_max_val,
                        ((raw_dst[2] as u32 >> dst_shift) as f32) / dst_max_val,
                        if dst_has_alpha {
                            ((raw_dst[3] as u32 >> dst_shift) as f32) / dst_max_val
                        } else {
                            1.0
                        },
                    ];
                    let dst_canon = [
                        dst_norm[dst_from_read_swiz[0] as usize],
                        dst_norm[dst_from_read_swiz[1] as usize],
                        dst_norm[dst_from_read_swiz[2] as usize],
                        dst_norm[dst_from_read_swiz[3] as usize],
                    ];
                    let mut blended = [0.0f32; 4];
                    for c in 0..3 {
                        let fg = if ov_premul {
                            ov_canon[c]
                        } else {
                            ov_canon[c] * a
                        };
                        blended[c] = (fg + dst_canon[c] * inv_a).clamp(0.0, 1.0);
                    }
                    blended[3] = (a + dst_canon[3] * inv_a).clamp(0.0, 1.0);
                    let out_mem = [
                        blended[dst_to_write_swiz[0] as usize],
                        blended[dst_to_write_swiz[1] as usize],
                        blended[dst_to_write_swiz[2] as usize],
                        blended[dst_to_write_swiz[3] as usize],
                    ];
                    let out_raw = if matches!(dst_layout.depth, SampleDepth::F32) {
                        out_mem
                    } else {
                        [
                            (((out_mem[0] * dst_max_val).round() as u32) << dst_shift) as f32,
                            (((out_mem[1] * dst_max_val).round() as u32) << dst_shift) as f32,
                            (((out_mem[2] * dst_max_val).round() as u32) << dst_shift) as f32,
                            (((out_mem[3] * dst_max_val).round() as u32) << dst_shift) as f32,
                        ]
                    };
                    write_luma_or_full_pixel(
                        dst_planes,
                        dst_layout,
                        dx as usize,
                        dy as usize,
                        out_raw,
                    );
                }
            }
        } else {
            // YCbCr destination: blend Y per-pixel and Cb/Cr per chroma sample
            for dy in y0..y1 {
                let oy = (dy - ov.y) as usize;
                for dx in x0..x1 {
                    let ox = (dx - ov.x) as usize;
                    let raw_ov = read_pixel_direct(
                        ov_planes,
                        ov.width,
                        ov.height,
                        ov.layout,
                        ChromaLocation::Left,
                        ox,
                        oy,
                    )?;
                    let rgba = color::unpack_alpha_overlay_pixel(
                        ov.layout,
                        raw_ov,
                        color::resolve_color(ov.layout, wedeo_core::ColorDesc::SRGB, None),
                    );
                    let a = rgba[3];
                    if a <= 0.0 {
                        continue;
                    }
                    let y_fg = rgb_to_dst_mat[0][0] * rgba[0]
                        + rgb_to_dst_mat[0][1] * rgba[1]
                        + rgb_to_dst_mat[0][2] * rgba[2]
                        + rgb_to_dst_mat[0][3];
                    let y_dst_raw = read_plane_sample(
                        &dst_planes[0],
                        dst_layout.depth,
                        dst_layout.endian,
                        dx as usize,
                        dy as usize,
                    );
                    let y_dst = ((y_dst_raw as u32 >> dst_shift) as f32) / dst_max_val;
                    let y_blend = (y_fg * a + y_dst * (1.0 - a)).clamp(0.0, 1.0);
                    let y_out = (((y_blend * dst_max_val).round() as u32) << dst_shift) as f32;
                    write_plane_sample(
                        &mut dst_planes[0],
                        dst_layout.depth,
                        dst_layout.endian,
                        dx as usize,
                        dy as usize,
                        y_out,
                    );
                }
            }
            let (sub_x, sub_y) = match dst_layout.subsampling {
                Subsampling::Cs444 => (1u32, 1u32),
                Subsampling::Cs422 => (2, 1),
                Subsampling::Cs420 => (2, 2),
                _ => (2, 2),
            };
            let cx0 = x0 / sub_x;
            let cy0 = y0 / sub_y;
            let (cw_max, ch_max) = dst_layout.subsampling.chroma_dims(dst_w, dst_h);
            let cx1 = x1.div_ceil(sub_x).min(cw_max);
            let cy1 = y1.div_ceil(sub_y).min(ch_max);
            for cy in cy0..cy1 {
                for cx in cx0..cx1 {
                    let mut sum_a = 0.0f32;
                    let mut sum_u = 0.0f32;
                    let mut sum_v = 0.0f32;
                    let mut cnt = 0.0f32;
                    for sy in (cy * sub_y)..((cy + 1) * sub_y).min(dst_h) {
                        for sx in (cx * sub_x)..((cx + 1) * sub_x).min(dst_w) {
                            if sx >= x0 && sx < x1 && sy >= y0 && sy < y1 {
                                let raw_ov = read_pixel_direct(
                                    ov_planes,
                                    ov.width,
                                    ov.height,
                                    ov.layout,
                                    ChromaLocation::Left,
                                    (sx - ov.x) as usize,
                                    (sy - ov.y) as usize,
                                )?;
                                let rgba = color::unpack_alpha_overlay_pixel(
                                    ov.layout,
                                    raw_ov,
                                    color::resolve_color(
                                        ov.layout,
                                        wedeo_core::ColorDesc::SRGB,
                                        None,
                                    ),
                                );
                                let a = rgba[3];
                                let u_fg = rgb_to_dst_mat[1][0] * rgba[0]
                                    + rgb_to_dst_mat[1][1] * rgba[1]
                                    + rgb_to_dst_mat[1][2] * rgba[2]
                                    + rgb_to_dst_mat[1][3];
                                let v_fg = rgb_to_dst_mat[2][0] * rgba[0]
                                    + rgb_to_dst_mat[2][1] * rgba[1]
                                    + rgb_to_dst_mat[2][2] * rgba[2]
                                    + rgb_to_dst_mat[2][3];
                                sum_a += a;
                                sum_u += u_fg * a;
                                sum_v += v_fg * a;
                            }
                            cnt += 1.0;
                        }
                    }
                    if sum_a <= 0.0 || cnt <= 0.0 {
                        continue;
                    }
                    let avg_a = sum_a / cnt;
                    let avg_u = sum_u / cnt;
                    let avg_v = sum_v / cnt;
                    let (du_raw, dv_raw) =
                        read_chroma_sample(dst_planes, dst_layout, cx as usize, cy as usize);
                    let du = ((du_raw as u32 >> dst_shift) as f32) / dst_max_val;
                    let dv = ((dv_raw as u32 >> dst_shift) as f32) / dst_max_val;
                    let nu = (avg_u + du * (1.0 - avg_a)).clamp(0.0, 1.0);
                    let nv = (avg_v + dv * (1.0 - avg_a)).clamp(0.0, 1.0);
                    let u_out = (((nu * dst_max_val).round() as u32) << dst_shift) as f32;
                    let v_out = (((nv * dst_max_val).round() as u32) << dst_shift) as f32;
                    write_chroma_sample(
                        dst_planes,
                        dst_layout,
                        cx as usize,
                        cy as usize,
                        u_out,
                        v_out,
                    );
                }
            }
        }
    }

    Ok(())
}

fn validate_layout(layout: PixelLayout) -> Result<()> {
    match (layout.model, layout.packing, layout.subsampling) {
        (ColorModel::Xyz, _, _) => Err(Error::unsupported(
            "wedeo-video-ops",
            "XYZ pixel layout is not supported",
        )),
        (_, _, Subsampling::Cs411 | Subsampling::Cs410) => Err(Error::unsupported(
            "wedeo-video-ops",
            "4:1:1 and 4:1:0 subsampling are not supported",
        )),
        _ if matches!(layout.depth, SampleDepth::F16) => Err(Error::unsupported(
            "wedeo-video-ops",
            "F16 sample depth is not supported on CPU",
        )),
        _ => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
fn read_tile(
    planes: &[BorderedPlane],
    src_w: u32,
    src_h: u32,
    layout: PixelLayout,
    chroma_loc: ChromaLocation,
    crop: CropRect,
    display_tx: DisplayTransform,
    oriented_w: u32,
    oriented_h: u32,
    tx: usize,
    ty: usize,
    tw: usize,
    th: usize,
    out: &mut [[f32; 4]],
) -> Result<()> {
    let cw = src_w - crop.left - crop.right;
    let ch = src_h - crop.top - crop.bottom;

    // Fast path: identity display_matrix + 0 crop + non-subsampled packed/planar can read row-wise
    // We still keep the unified path super clean and branch-predicted!
    let _ = (oriented_w, oriented_h);
    for ly in 0..th {
        let y_out = (ty + ly) as u32;
        let row_out = &mut out[ly * tw..(ly + 1) * tw];
        for (lx, dst_px) in row_out.iter_mut().enumerate() {
            let x_out = (tx + lx) as u32;
            let (xc, yc) = if display_tx.is_identity() {
                (x_out, y_out)
            } else {
                display_tx.map_to_src(x_out, y_out, cw, ch)
            };
            let sx = (crop.left + xc) as usize;
            let sy = (crop.top + yc) as usize;
            *dst_px = read_pixel_direct(planes, src_w, src_h, layout, chroma_loc, sx, sy)?;
        }
    }
    Ok(())
}

#[inline]
fn chroma_siting_offsets(loc: ChromaLocation, sub_x: bool, sub_y: bool) -> (f32, f32) {
    let (xo, yo) = match loc {
        ChromaLocation::Unspecified | ChromaLocation::Left => (0.0, 0.5),
        ChromaLocation::Center => (0.5, 0.5),
        ChromaLocation::TopLeft => (0.0, 0.0),
        ChromaLocation::Top => (0.5, 0.0),
        ChromaLocation::BottomLeft => (0.0, 1.0),
        ChromaLocation::Bottom => (0.5, 1.0),
    };
    (if sub_x { xo } else { 0.0 }, if sub_y { yo } else { 0.0 })
}

#[inline]
fn read_pixel_direct(
    planes: &[BorderedPlane],
    src_w: u32,
    src_h: u32,
    layout: PixelLayout,
    chroma_loc: ChromaLocation,
    sx: usize,
    sy: usize,
) -> Result<[f32; 4]> {
    match layout.packing {
        PlanePacking::Packed => {
            let row = planes[0].row(sy).ok_or_else(|| {
                Error::invalid_data("wedeo-video-ops", "source row out of bounds")
            })?;
            match layout.order {
                CompOrder::Rgb | CompOrder::Bgr => {
                    let bps = layout.depth.bytes_per_sample();
                    let off = sx * 3 * bps;
                    Ok([
                        read_sample_from_slice(&row[off..], layout.depth, layout.endian),
                        read_sample_from_slice(&row[off + bps..], layout.depth, layout.endian),
                        read_sample_from_slice(&row[off + 2 * bps..], layout.depth, layout.endian),
                        1.0,
                    ])
                }
                CompOrder::Rgba | CompOrder::Bgra | CompOrder::Argb | CompOrder::Abgr => {
                    let bps = layout.depth.bytes_per_sample();
                    let off = sx * 4 * bps;
                    Ok([
                        read_sample_from_slice(&row[off..], layout.depth, layout.endian),
                        read_sample_from_slice(&row[off + bps..], layout.depth, layout.endian),
                        read_sample_from_slice(&row[off + 2 * bps..], layout.depth, layout.endian),
                        read_sample_from_slice(&row[off + 3 * bps..], layout.depth, layout.endian),
                    ])
                }
                CompOrder::Gray => {
                    let bps = layout.depth.bytes_per_sample();
                    let y = read_sample_from_slice(&row[sx * bps..], layout.depth, layout.endian);
                    Ok([y, 0.0, 0.0, 1.0])
                }
                CompOrder::Yuyv | CompOrder::Uyvy => {
                    // 4:2:2 packed: [Y0, U, Y1, V] or [U, Y0, V, Y1]
                    let pair = sx / 2;
                    let is_odd = sx & 1;
                    let base = pair * 4;
                    let y = if matches!(layout.order, CompOrder::Yuyv) {
                        row[base + is_odd * 2] as f32
                    } else {
                        row[base + 1 + is_odd * 2] as f32
                    };
                    let (u_off, v_off) = if matches!(layout.order, CompOrder::Yuyv) {
                        (1, 3)
                    } else {
                        (0, 2)
                    };
                    let cw = src_w.div_ceil(2) as usize;
                    let fx = (sx as f32) * 0.5;
                    let cx0 = (fx.floor() as usize).min(cw - 1);
                    let cx1 = (cx0 + 1).min(cw - 1);
                    let tx = fx - cx0 as f32;
                    let u =
                        row[cx0 * 4 + u_off] as f32 * (1.0 - tx) + row[cx1 * 4 + u_off] as f32 * tx;
                    let v =
                        row[cx0 * 4 + v_off] as f32 * (1.0 - tx) + row[cx1 * 4 + v_off] as f32 * tx;
                    Ok([y, u, v, 1.0])
                }
                _ => Err(Error::unsupported(
                    "wedeo-video-ops",
                    "unsupported packed component order",
                )),
            }
        }
        PlanePacking::SemiPlanar => {
            let y = read_plane_sample(&planes[0], layout.depth, layout.endian, sx, sy);
            let (cw, ch) = layout.subsampling.chroma_dims(src_w, src_h);
            let sub_x = matches!(layout.subsampling, Subsampling::Cs420 | Subsampling::Cs422);
            let sub_y = matches!(layout.subsampling, Subsampling::Cs420);
            let (xo, yo) = chroma_siting_offsets(chroma_loc, sub_x, sub_y);
            let fx = if sub_x {
                ((sx as f32 - xo) * 0.5).clamp(0.0, (cw.max(1) - 1) as f32)
            } else {
                sx as f32
            };
            let fy = if sub_y {
                ((sy as f32 - yo) * 0.5).clamp(0.0, (ch.max(1) - 1) as f32)
            } else {
                sy as f32
            };
            let cx0 = (fx.floor() as usize).min(cw as usize - 1);
            let cx1 = (cx0 + 1).min(cw as usize - 1);
            let cy0 = (fy.floor() as usize).min(ch as usize - 1);
            let cy1 = (cy0 + 1).min(ch as usize - 1);
            let wx = fx - cx0 as f32;
            let wy = fy - cy0 as f32;

            let (u00, v00) = read_semiplanar_uv(&planes[1], layout.depth, layout.endian, cx0, cy0);
            let (u10, v10) = read_semiplanar_uv(&planes[1], layout.depth, layout.endian, cx1, cy0);
            let (u01, v01) = read_semiplanar_uv(&planes[1], layout.depth, layout.endian, cx0, cy1);
            let (u11, v11) = read_semiplanar_uv(&planes[1], layout.depth, layout.endian, cx1, cy1);
            let u = bilerp(u00, u10, u01, u11, wx, wy);
            let v = bilerp(v00, v10, v01, v11, wx, wy);
            let a = if !matches!(layout.alpha, AlphaMode::None) {
                read_plane_sample(&planes[2], layout.depth, layout.endian, sx, sy)
            } else {
                1.0
            };
            Ok([y, u, v, a])
        }
        PlanePacking::Planar => {
            if layout.model == ColorModel::Gray {
                let y = read_plane_sample(&planes[0], layout.depth, layout.endian, sx, sy);
                let a = if !matches!(layout.alpha, AlphaMode::None) {
                    read_plane_sample(&planes[1], layout.depth, layout.endian, sx, sy)
                } else {
                    1.0
                };
                return Ok([y, 0.0, 0.0, a]);
            }
            let c0 = read_plane_sample(&planes[0], layout.depth, layout.endian, sx, sy);
            let (c1, c2) = match layout.subsampling {
                Subsampling::Cs444 => (
                    read_plane_sample(&planes[1], layout.depth, layout.endian, sx, sy),
                    read_plane_sample(&planes[2], layout.depth, layout.endian, sx, sy),
                ),
                Subsampling::Cs422 | Subsampling::Cs420 => {
                    let (cw, ch) = layout.subsampling.chroma_dims(src_w, src_h);
                    let sub_y = matches!(layout.subsampling, Subsampling::Cs420);
                    let (xo, yo) = chroma_siting_offsets(chroma_loc, true, sub_y);
                    let fx = ((sx as f32 - xo) * 0.5).clamp(0.0, (cw.max(1) - 1) as f32);
                    let fy = if sub_y {
                        ((sy as f32 - yo) * 0.5).clamp(0.0, (ch.max(1) - 1) as f32)
                    } else {
                        sy as f32
                    };
                    let cx0 = (fx.floor() as usize).min(cw as usize - 1);
                    let cx1 = (cx0 + 1).min(cw as usize - 1);
                    let cy0 = (fy.floor() as usize).min(ch as usize - 1);
                    let cy1 = (cy0 + 1).min(ch as usize - 1);
                    let wx = fx - cx0 as f32;
                    let wy = fy - cy0 as f32;
                    let u = bilerp(
                        read_plane_sample(&planes[1], layout.depth, layout.endian, cx0, cy0),
                        read_plane_sample(&planes[1], layout.depth, layout.endian, cx1, cy0),
                        read_plane_sample(&planes[1], layout.depth, layout.endian, cx0, cy1),
                        read_plane_sample(&planes[1], layout.depth, layout.endian, cx1, cy1),
                        wx,
                        wy,
                    );
                    let v = bilerp(
                        read_plane_sample(&planes[2], layout.depth, layout.endian, cx0, cy0),
                        read_plane_sample(&planes[2], layout.depth, layout.endian, cx1, cy0),
                        read_plane_sample(&planes[2], layout.depth, layout.endian, cx0, cy1),
                        read_plane_sample(&planes[2], layout.depth, layout.endian, cx1, cy1),
                        wx,
                        wy,
                    );
                    (u, v)
                }
                _ => (0.0, 0.0),
            };
            let a = if !matches!(layout.alpha, AlphaMode::None) {
                read_plane_sample(&planes[3], layout.depth, layout.endian, sx, sy)
            } else {
                1.0
            };
            Ok([c0, c1, c2, a])
        }
    }
}

#[inline]
fn bilerp(v00: f32, v10: f32, v01: f32, v11: f32, wx: f32, wy: f32) -> f32 {
    let top = v00 * (1.0 - wx) + v10 * wx;
    let bot = v01 * (1.0 - wx) + v11 * wx;
    top * (1.0 - wy) + bot * wy
}

#[inline]
fn read_sample_from_slice(slice: &[u8], depth: SampleDepth, endian: Endian) -> f32 {
    match depth {
        SampleDepth::U8 => slice[0] as f32,
        SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16 => {
            let bytes = [slice[0], slice[1]];
            let v = match endian {
                Endian::Little => u16::from_le_bytes(bytes),
                Endian::Big => u16::from_be_bytes(bytes),
            };
            v as f32
        }
        SampleDepth::F32 => {
            let bytes = [slice[0], slice[1], slice[2], slice[3]];
            match endian {
                Endian::Little => f32::from_le_bytes(bytes),
                Endian::Big => f32::from_be_bytes(bytes),
            }
        }
        SampleDepth::F16 => 0.0,
    }
}

#[inline]
fn write_sample_to_slice(slice: &mut [u8], depth: SampleDepth, endian: Endian, val: f32) {
    match depth {
        SampleDepth::U8 => {
            slice[0] = val.round().clamp(0.0, 255.0) as u8;
        }
        SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16 => {
            let v = val.round().clamp(0.0, 65535.0) as u16;
            let bytes = match endian {
                Endian::Little => v.to_le_bytes(),
                Endian::Big => v.to_be_bytes(),
            };
            slice[0] = bytes[0];
            slice[1] = bytes[1];
        }
        SampleDepth::F32 => {
            let bytes = match endian {
                Endian::Little => val.to_le_bytes(),
                Endian::Big => val.to_be_bytes(),
            };
            slice[0..4].copy_from_slice(&bytes);
        }
        SampleDepth::F16 => {}
    }
}

#[inline]
fn read_plane_sample(
    plane: &BorderedPlane,
    depth: SampleDepth,
    endian: Endian,
    x: usize,
    y: usize,
) -> f32 {
    let bps = depth.bytes_per_sample();
    let row = plane.row(y).expect("plane row in bounds");
    read_sample_from_slice(&row[x * bps..], depth, endian)
}

#[inline]
fn write_plane_sample(
    plane: &mut BorderedPlane,
    depth: SampleDepth,
    endian: Endian,
    x: usize,
    y: usize,
    val: f32,
) {
    let bps = depth.bytes_per_sample();
    let row = plane.row_mut(y).expect("plane row in bounds");
    write_sample_to_slice(&mut row[x * bps..], depth, endian, val);
}

#[inline]
fn read_semiplanar_uv(
    plane: &BorderedPlane,
    depth: SampleDepth,
    endian: Endian,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    let bps = depth.bytes_per_sample();
    let row = plane.row(cy).expect("chroma row in bounds");
    let off = cx * 2 * bps;
    (
        read_sample_from_slice(&row[off..], depth, endian),
        read_sample_from_slice(&row[off + bps..], depth, endian),
    )
}

#[inline]
fn read_chroma_sample(
    planes: &[BorderedPlane],
    layout: PixelLayout,
    cx: usize,
    cy: usize,
) -> (f32, f32) {
    match layout.packing {
        PlanePacking::SemiPlanar => {
            let (c1, c2) = read_semiplanar_uv(&planes[1], layout.depth, layout.endian, cx, cy);
            if layout.order == CompOrder::Yvu {
                (c2, c1)
            } else {
                (c1, c2)
            }
        }
        PlanePacking::Planar => (
            read_plane_sample(&planes[1], layout.depth, layout.endian, cx, cy),
            read_plane_sample(&planes[2], layout.depth, layout.endian, cx, cy),
        ),
        PlanePacking::Packed => (0.0, 0.0),
    }
}

#[inline]
fn write_chroma_sample(
    planes: &mut [BorderedPlane],
    layout: PixelLayout,
    cx: usize,
    cy: usize,
    u: f32,
    v: f32,
) {
    match layout.packing {
        PlanePacking::SemiPlanar => {
            let bps = layout.depth.bytes_per_sample();
            let row = planes[1].row_mut(cy).expect("chroma row in bounds");
            let off = cx * 2 * bps;
            if layout.order == CompOrder::Yvu {
                write_sample_to_slice(&mut row[off..], layout.depth, layout.endian, v);
                write_sample_to_slice(&mut row[off + bps..], layout.depth, layout.endian, u);
            } else {
                write_sample_to_slice(&mut row[off..], layout.depth, layout.endian, u);
                write_sample_to_slice(&mut row[off + bps..], layout.depth, layout.endian, v);
            }
        }
        PlanePacking::Planar => {
            write_plane_sample(&mut planes[1], layout.depth, layout.endian, cx, cy, u);
            write_plane_sample(&mut planes[2], layout.depth, layout.endian, cx, cy, v);
        }
        PlanePacking::Packed => {}
    }
}

#[inline]
fn write_luma_or_full_pixel(
    planes: &mut [BorderedPlane],
    layout: PixelLayout,
    x: usize,
    y: usize,
    px: [f32; 4],
) {
    let bps = layout.depth.bytes_per_sample();
    match layout.packing {
        PlanePacking::Packed => {
            let row = planes[0].row_mut(y).expect("row in bounds");
            match layout.order {
                CompOrder::Rgb | CompOrder::Bgr => {
                    let off = x * 3 * bps;
                    write_sample_to_slice(&mut row[off..], layout.depth, layout.endian, px[0]);
                    write_sample_to_slice(
                        &mut row[off + bps..],
                        layout.depth,
                        layout.endian,
                        px[1],
                    );
                    write_sample_to_slice(
                        &mut row[off + 2 * bps..],
                        layout.depth,
                        layout.endian,
                        px[2],
                    );
                }
                CompOrder::Rgba | CompOrder::Bgra | CompOrder::Argb | CompOrder::Abgr => {
                    let off = x * 4 * bps;
                    write_sample_to_slice(&mut row[off..], layout.depth, layout.endian, px[0]);
                    write_sample_to_slice(
                        &mut row[off + bps..],
                        layout.depth,
                        layout.endian,
                        px[1],
                    );
                    write_sample_to_slice(
                        &mut row[off + 2 * bps..],
                        layout.depth,
                        layout.endian,
                        px[2],
                    );
                    write_sample_to_slice(
                        &mut row[off + 3 * bps..],
                        layout.depth,
                        layout.endian,
                        px[3],
                    );
                }
                CompOrder::Gray => {
                    write_sample_to_slice(&mut row[x * bps..], layout.depth, layout.endian, px[0]);
                }
                _ => {}
            }
        }
        PlanePacking::Planar => {
            write_plane_sample(&mut planes[0], layout.depth, layout.endian, x, y, px[0]);
            if layout.model != ColorModel::Gray {
                write_plane_sample(&mut planes[1], layout.depth, layout.endian, x, y, px[1]);
                write_plane_sample(&mut planes[2], layout.depth, layout.endian, x, y, px[2]);
                if !matches!(layout.alpha, AlphaMode::None) {
                    write_plane_sample(&mut planes[3], layout.depth, layout.endian, x, y, px[3]);
                }
            } else if !matches!(layout.alpha, AlphaMode::None) {
                write_plane_sample(&mut planes[1], layout.depth, layout.endian, x, y, px[3]);
            }
        }
        PlanePacking::SemiPlanar => {
            write_plane_sample(&mut planes[0], layout.depth, layout.endian, x, y, px[0]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_tile(
    planes: &mut [BorderedPlane],
    dst_w: u32,
    dst_h: u32,
    layout: PixelLayout,
    tx: usize,
    ty: usize,
    tw: usize,
    th: usize,
    tile: &[[f32; 4]],
) -> Result<()> {
    let bps = layout.depth.bytes_per_sample();
    match layout.packing {
        PlanePacking::Packed => match layout.order {
            CompOrder::Rgb | CompOrder::Bgr => {
                for ly in 0..th {
                    let row = planes[0].row_mut(ty + ly).expect("row in bounds");
                    let src_row = &tile[ly * tw..(ly + 1) * tw];
                    for (lx, px) in src_row.iter().enumerate() {
                        let off = (tx + lx) * 3 * bps;
                        write_sample_to_slice(&mut row[off..], layout.depth, layout.endian, px[0]);
                        write_sample_to_slice(
                            &mut row[off + bps..],
                            layout.depth,
                            layout.endian,
                            px[1],
                        );
                        write_sample_to_slice(
                            &mut row[off + 2 * bps..],
                            layout.depth,
                            layout.endian,
                            px[2],
                        );
                    }
                }
            }
            CompOrder::Rgba | CompOrder::Bgra | CompOrder::Argb | CompOrder::Abgr => {
                for ly in 0..th {
                    let row = planes[0].row_mut(ty + ly).expect("row in bounds");
                    let src_row = &tile[ly * tw..(ly + 1) * tw];
                    for (lx, px) in src_row.iter().enumerate() {
                        let off = (tx + lx) * 4 * bps;
                        write_sample_to_slice(&mut row[off..], layout.depth, layout.endian, px[0]);
                        write_sample_to_slice(
                            &mut row[off + bps..],
                            layout.depth,
                            layout.endian,
                            px[1],
                        );
                        write_sample_to_slice(
                            &mut row[off + 2 * bps..],
                            layout.depth,
                            layout.endian,
                            px[2],
                        );
                        write_sample_to_slice(
                            &mut row[off + 3 * bps..],
                            layout.depth,
                            layout.endian,
                            px[3],
                        );
                    }
                }
            }
            CompOrder::Gray => {
                for ly in 0..th {
                    let row = planes[0].row_mut(ty + ly).expect("row in bounds");
                    let src_row = &tile[ly * tw..(ly + 1) * tw];
                    for (lx, px) in src_row.iter().enumerate() {
                        write_sample_to_slice(
                            &mut row[(tx + lx) * bps..],
                            layout.depth,
                            layout.endian,
                            px[0],
                        );
                    }
                }
            }
            CompOrder::Yuyv | CompOrder::Uyvy => {
                let is_yuyv = layout.order == CompOrder::Yuyv;
                for ly in 0..th {
                    let row = planes[0].row_mut(ty + ly).expect("row in bounds");
                    let src_row = &tile[ly * tw..(ly + 1) * tw];
                    let pairs = tw.div_ceil(2);
                    for p in 0..pairs {
                        let x0 = p * 2;
                        let x1 = (x0 + 1).min(tw - 1);
                        let p0 = src_row[x0];
                        let p1 = src_row[x1];
                        let u = ((p0[1] + p1[1]) * 0.5).round().clamp(0.0, 255.0) as u8;
                        let v = ((p0[2] + p1[2]) * 0.5).round().clamp(0.0, 255.0) as u8;
                        let y0 = p0[0].round().clamp(0.0, 255.0) as u8;
                        let y1 = p1[0].round().clamp(0.0, 255.0) as u8;
                        let base = ((tx / 2) + p) * 4;
                        if is_yuyv {
                            row[base] = y0;
                            row[base + 1] = u;
                            row[base + 2] = y1;
                            row[base + 3] = v;
                        } else {
                            row[base] = u;
                            row[base + 1] = y0;
                            row[base + 2] = v;
                            row[base + 3] = y1;
                        }
                    }
                }
            }
            _ => {
                return Err(Error::unsupported(
                    "wedeo-video-ops",
                    "unsupported packed write order",
                ));
            }
        },
        PlanePacking::SemiPlanar | PlanePacking::Planar => {
            // Plane 0 (luma or G or Gray)
            for ly in 0..th {
                let row = planes[0].row_mut(ty + ly).expect("row in bounds");
                let src_row = &tile[ly * tw..(ly + 1) * tw];
                for (lx, px) in src_row.iter().enumerate() {
                    write_sample_to_slice(
                        &mut row[(tx + lx) * bps..],
                        layout.depth,
                        layout.endian,
                        px[0],
                    );
                }
            }

            if layout.model == ColorModel::Gray {
                if !matches!(layout.alpha, AlphaMode::None) {
                    for ly in 0..th {
                        let row = planes[1].row_mut(ty + ly).expect("alpha row in bounds");
                        let src_row = &tile[ly * tw..(ly + 1) * tw];
                        for (lx, px) in src_row.iter().enumerate() {
                            write_sample_to_slice(
                                &mut row[(tx + lx) * bps..],
                                layout.depth,
                                layout.endian,
                                px[3],
                            );
                        }
                    }
                }
                return Ok(());
            }

            match layout.subsampling {
                Subsampling::Cs444 => {
                    for ly in 0..th {
                        let src_row = &tile[ly * tw..(ly + 1) * tw];
                        for (lx, px) in src_row.iter().enumerate() {
                            write_chroma_sample(planes, layout, tx + lx, ty + ly, px[1], px[2]);
                        }
                    }
                }
                Subsampling::Cs422 => {
                    let cw = tw.div_ceil(2);
                    let cx_base = tx / 2;
                    for ly in 0..th {
                        let src_row = &tile[ly * tw..(ly + 1) * tw];
                        for lx_c in 0..cw {
                            let x0 = lx_c * 2;
                            let x1 = (x0 + 1).min(tw - 1);
                            let u = (src_row[x0][1] + src_row[x1][1]) * 0.5;
                            let v = (src_row[x0][2] + src_row[x1][2]) * 0.5;
                            write_chroma_sample(planes, layout, cx_base + lx_c, ty + ly, u, v);
                        }
                    }
                }
                Subsampling::Cs420 => {
                    let _ = (dst_w, dst_h);
                    let cw = tw.div_ceil(2);
                    let ch = th.div_ceil(2);
                    let cx_base = tx / 2;
                    let cy_base = ty / 2;
                    for ly_c in 0..ch {
                        let y0 = ly_c * 2;
                        let y1 = (y0 + 1).min(th - 1);
                        let r0 = &tile[y0 * tw..(y0 + 1) * tw];
                        let r1 = &tile[y1 * tw..(y1 + 1) * tw];
                        for lx_c in 0..cw {
                            let x0 = lx_c * 2;
                            let x1 = (x0 + 1).min(tw - 1);
                            let u = (r0[x0][1] + r0[x1][1] + r1[x0][1] + r1[x1][1]) * 0.25;
                            let v = (r0[x0][2] + r0[x1][2] + r1[x0][2] + r1[x1][2]) * 0.25;
                            write_chroma_sample(
                                planes,
                                layout,
                                cx_base + lx_c,
                                cy_base + ly_c,
                                u,
                                v,
                            );
                        }
                    }
                }
                _ => {}
            }

            if !matches!(layout.alpha, AlphaMode::None) {
                let a_idx = if matches!(layout.packing, PlanePacking::SemiPlanar) {
                    2
                } else {
                    3
                };
                for ly in 0..th {
                    let row = planes[a_idx].row_mut(ty + ly).expect("alpha row in bounds");
                    let src_row = &tile[ly * tw..(ly + 1) * tw];
                    for (lx, px) in src_row.iter().enumerate() {
                        write_sample_to_slice(
                            &mut row[(tx + lx) * bps..],
                            layout.depth,
                            layout.endian,
                            px[3],
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
