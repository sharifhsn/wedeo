//! Structural WGSL compute shader generator, uniform buffer packer, `naga` validator,
//! pipeline cache, and custom [`WgslComputePass`] helper for [`SwsOpList`] and `WgslFilter`.

use std::fmt::Write as _;
use std::sync::Arc;

use wedeo_core::{
    AlphaMode, BufferPool, ChromaLocation, ColorModel, CompOrder, Error, MemoryDomain, PixelLayout,
    PlanePacking, Result, SampleDepth, Subsampling, TransferCharacteristic, VideoFrame,
};
use wedeo_video_ops::ops::layout_shift;
use wedeo_video_ops::{
    DisplayTransform, DitherMode, SwsOp, SwsOpList, SwsScaler, ToneMapOp, color, optimizer,
};

use crate::context::{WgpuContext, plane_texture_format};

/// Workgroup dimensions emitted for `SwsOpList` WGSL compute shaders (`16 × 16 × 1`).
pub const WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

/// Maximum number of `SwsOp::ColorMatrix` operations supported in a single shader pass.
/// (After `SwsOpList::optimize()`, at most 2–3 matrices remain: decode, gamut, encode.)
pub const MAX_COLOR_MATRICES: usize = 4;

/// Byte size of [`SwsUniforms`] (`80 + 12 * 16 = 272` bytes, 16-byte aligned).
pub const SWS_UNIFORMS_SIZE: usize = 272;

/// Uniform buffer payload corresponding to WGSL `struct SwsUniforms`.
///
/// Storing all numeric parameters (dimensions, crop margins, display rotation matrix,
/// HDR peak luminance, chroma siting offsets, and 3×4 color matrices) in this uniform
/// buffer ensures that generated WGSL shaders and pipeline cache keys depend *only*
/// on the structural sequence of [`SwsOp`]s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwsUniforms {
    pub src_w: u32,
    pub src_h: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub crop_left: u32,
    pub crop_top: u32,
    pub crop_w: u32,
    pub crop_h: u32,
    pub oriented_w: u32,
    pub oriented_h: u32,
    pub rot_a: i32,
    pub rot_b: i32,
    pub rot_c: i32,
    pub rot_d: i32,
    pub src_peak_nits: f32,
    pub dst_peak_nits: f32,
    pub chroma_offset_x: f32,
    pub chroma_offset_y: f32,
    pub matrices: [[[f32; 4]; 3]; MAX_COLOR_MATRICES],
}

impl SwsUniforms {
    /// Extract runtime uniform parameters from an [`SwsOpList`].
    pub fn from_op_list(ops: &SwsOpList) -> Result<Self> {
        let Some(SwsOp::Read {
            layout: read_layout,
            crop,
            display_matrix,
        }) = ops.ops.first().cloned()
        else {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "SwsOpList must begin with SwsOp::Read",
            ));
        };

        let (crop_w, crop_h) = crop.display_size(ops.src.width, ops.src.height);
        if crop_w == 0 || crop_h == 0 {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "crop rectangle leaves zero active pixels",
            ));
        }
        let dt = DisplayTransform::from_matrix(&display_matrix)?;
        let (oriented_w, oriented_h) = dt.output_dims(crop_w, crop_h);

        let sub_x = matches!(
            read_layout.subsampling,
            Subsampling::Cs420 | Subsampling::Cs422
        );
        let sub_y = matches!(read_layout.subsampling, Subsampling::Cs420);
        let (chroma_offset_x, chroma_offset_y) =
            chroma_siting_offsets(ops.src.color.chroma_location, sub_x, sub_y);

        let src_peak_nits =
            color::effective_src_peak_nits(ops.src.color.transfer, ops.options.src_peak_nits);
        let dst_peak_nits = ops.options.dst_peak_nits.max(1.0);

        let mut matrices = [[
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]; MAX_COLOR_MATRICES];

        let mut slot = 0usize;
        for op in &ops.ops {
            if let SwsOp::ColorMatrix(m) = op {
                if slot >= MAX_COLOR_MATRICES {
                    return Err(Error::unsupported(
                        "wedeo-gpu-wgpu",
                        "SwsOpList contains more than MAX_COLOR_MATRICES (4) un-fused ColorMatrix ops",
                    ));
                }
                matrices[slot] = *m;
                slot += 1;
            }
        }

        Ok(Self {
            src_w: ops.src.width,
            src_h: ops.src.height,
            dst_w: ops.dst.width,
            dst_h: ops.dst.height,
            crop_left: crop.left,
            crop_top: crop.top,
            crop_w,
            crop_h,
            oriented_w,
            oriented_h,
            rot_a: dt.a,
            rot_b: dt.b,
            rot_c: dt.c,
            rot_d: dt.d,
            src_peak_nits,
            dst_peak_nits,
            chroma_offset_x,
            chroma_offset_y,
            matrices,
        })
    }

    /// Serialize into a 16-byte-aligned 272-byte std140 uniform buffer array using plain `to_le_bytes()`.
    pub fn to_bytes(&self) -> [u8; SWS_UNIFORMS_SIZE] {
        let mut buf = [0u8; SWS_UNIFORMS_SIZE];
        let mut pos = 0usize;

        let mut write_u32 = |v: u32| {
            buf[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
            pos += 4;
        };
        write_u32(self.src_w);
        write_u32(self.src_h);
        write_u32(self.dst_w);
        write_u32(self.dst_h);

        write_u32(self.crop_left);
        write_u32(self.crop_top);
        write_u32(self.crop_w);
        write_u32(self.crop_h);

        write_u32(self.oriented_w);
        write_u32(self.oriented_h);
        write_u32(0);
        write_u32(0);

        let mut write_i32 = |v: i32| {
            buf[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
            pos += 4;
        };
        write_i32(self.rot_a);
        write_i32(self.rot_b);
        write_i32(self.rot_c);
        write_i32(self.rot_d);

        let mut write_f32 = |v: f32| {
            buf[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
            pos += 4;
        };
        write_f32(self.src_peak_nits);
        write_f32(self.dst_peak_nits);
        write_f32(self.chroma_offset_x);
        write_f32(self.chroma_offset_y);

        for mat in &self.matrices {
            for row in mat {
                for &val in row {
                    write_f32(val);
                }
            }
        }
        debug_assert_eq!(pos, SWS_UNIFORMS_SIZE);
        buf
    }
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

/// How the generated WGSL compute shader writes its output pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WgslOutputBinding {
    /// Writes directly to a `texture_storage_2d<rgba8unorm, write>`,
    /// `texture_storage_2d<bgra8unorm, write>`, or `texture_storage_2d<rgba32float, write>`.
    StorageTexture(wgpu::TextureFormat),
    /// Writes quantized sample values to 1..4 `var<storage, read_write> dst_plane_i: array<u32>` buffers.
    PlanarStorageBuffers { plane_count: usize },
}

/// Generated structural WGSL compute shader and binding metadata for an [`SwsOpList`].
#[derive(Debug, Clone)]
pub struct GeneratedWgsl {
    pub structural_hash: u64,
    pub source: String,
    pub entry_point: &'static str,
    pub workgroup_size: [u32; 3],
    pub read_layout: PixelLayout,
    pub write_layout: PixelLayout,
    pub src_plane_count: usize,
    pub output_binding: WgslOutputBinding,
    pub num_color_matrices: usize,
}

/// Cached compiled `wgpu::ComputePipeline` and `wgpu::BindGroupLayout` keyed by [`structural_op_list_hash`].
#[derive(Debug, Clone)]
pub struct CachedPipeline {
    pub structural_hash: u64,
    pub wgsl_source: Arc<str>,
    pub output_binding: WgslOutputBinding,
    pub src_plane_count: usize,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

/// Deterministic 64-bit FNV-1a hash helper.
struct Fnv1a64(u64);

impl Fnv1a64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }

    #[inline]
    fn write_u8(&mut self, byte: u8) {
        self.0 ^= byte as u64;
        self.0 = self.0.wrapping_mul(Self::PRIME);
    }

    #[inline]
    fn write_layout(&mut self, layout: PixelLayout) {
        self.write_u8(match layout.model {
            ColorModel::Ycbcr => 1,
            ColorModel::Rgb => 2,
            ColorModel::Gray => 3,
            ColorModel::Xyz => 4,
        });
        self.write_u8(match layout.packing {
            PlanePacking::Planar => 1,
            PlanePacking::SemiPlanar => 2,
            PlanePacking::Packed => 3,
        });
        self.write_u8(match layout.subsampling {
            Subsampling::Cs444 => 1,
            Subsampling::Cs422 => 2,
            Subsampling::Cs420 => 3,
            Subsampling::Cs411 => 4,
            Subsampling::Cs410 => 5,
            Subsampling::Cs400 => 6,
        });
        self.write_u8(match layout.depth {
            SampleDepth::U8 => 1,
            SampleDepth::U10 => 2,
            SampleDepth::U12 => 3,
            SampleDepth::U14 => 4,
            SampleDepth::U16 => 5,
            SampleDepth::F16 => 6,
            SampleDepth::F32 => 7,
        });
        self.write_u8(match layout.endian {
            wedeo_core::Endian::Little => 1,
            wedeo_core::Endian::Big => 2,
        });
        self.write_u8(match layout.alpha {
            AlphaMode::None => 1,
            AlphaMode::Straight => 2,
            AlphaMode::Premultiplied => 3,
        });
        self.write_u8(match layout.order {
            CompOrder::Yuv => 1,
            CompOrder::Yvu => 2,
            CompOrder::Rgb => 3,
            CompOrder::Bgr => 4,
            CompOrder::Gbr => 5,
            CompOrder::Rgba => 6,
            CompOrder::Bgra => 7,
            CompOrder::Argb => 8,
            CompOrder::Abgr => 9,
            CompOrder::Yuyv => 10,
            CompOrder::Uyvy => 11,
            CompOrder::Gray => 12,
        });
    }

    #[inline]
    const fn finish(self) -> u64 {
        self.0
    }
}

/// Compute the structural hash of an [`SwsOpList`] for pipeline caching.
///
/// Crucially, `ColorMatrix` float coefficients, `Read` crop rectangles, `display_matrix`
/// values, frame dimensions, and HDR peak nits are **excluded** from the structural hash
/// because they are passed via [`SwsUniforms`]. Different color matrices or frame dimensions
/// with the same structural op list produce the exact same hash and reuse the cached pipeline.
pub fn structural_op_list_hash(ops: &SwsOpList) -> u64 {
    let mut h = Fnv1a64::new();
    let mut mat_idx = 0u8;
    for op in &ops.ops {
        match *op {
            SwsOp::Read { layout, .. } => {
                h.write_u8(1);
                h.write_layout(layout);
            }
            SwsOp::Unpack { depth, shift } => {
                h.write_u8(2);
                h.write_u8(depth.bits());
                h.write_u8(shift);
            }
            SwsOp::Swizzle(s) => {
                h.write_u8(3);
                for b in s {
                    h.write_u8(b);
                }
            }
            SwsOp::Linearize(trc) => {
                h.write_u8(4);
                h.write_u8(color::canonical_transfer(trc) as u8);
            }
            SwsOp::ColorMatrix(_) => {
                h.write_u8(5);
                h.write_u8(mat_idx);
                mat_idx = mat_idx.wrapping_add(1);
            }
            SwsOp::ToneMap(tm) => {
                h.write_u8(6);
                h.write_u8(match tm {
                    ToneMapOp::None => 0,
                    ToneMapOp::Clip => 1,
                    ToneMapOp::Hable => 2,
                    ToneMapOp::Bt2390 => 3,
                });
            }
            SwsOp::ScaleH {
                kernel, chroma_loc, ..
            } => {
                h.write_u8(7);
                h.write_u8(scaler_tag(kernel));
                h.write_u8(chroma_loc as u8);
            }
            SwsOp::ScaleV {
                kernel, chroma_loc, ..
            } => {
                h.write_u8(8);
                h.write_u8(scaler_tag(kernel));
                h.write_u8(chroma_loc as u8);
            }
            SwsOp::BlendAlpha => {
                h.write_u8(9);
            }
            SwsOp::Delinearize(trc) => {
                h.write_u8(10);
                h.write_u8(color::canonical_transfer(trc) as u8);
            }
            SwsOp::Dither(mode) => {
                h.write_u8(11);
                h.write_u8(match mode {
                    DitherMode::None => 0,
                    DitherMode::Ordered => 1,
                    DitherMode::ErrorDiffusion => 2,
                });
            }
            SwsOp::Pack(layout) => {
                h.write_u8(12);
                h.write_layout(layout);
            }
            SwsOp::Write(layout) => {
                h.write_u8(13);
                h.write_layout(layout);
            }
        }
    }
    h.finish()
}

#[inline]
const fn scaler_tag(scaler: SwsScaler) -> u8 {
    match scaler {
        SwsScaler::Point => 1,
        SwsScaler::Bilinear => 2,
        SwsScaler::Bicubic => 3,
        SwsScaler::Lanczos => 4,
    }
}

/// Validate a WGSL shader source string using `wgpu::naga`'s WGSL frontend and IR validator.
///
/// Works in all environments, including headless CI containers without GPU drivers.
pub fn validate_wgsl(source: &str) -> Result<()> {
    let module = wgpu::naga::front::wgsl::parse_str(source).map_err(|e| {
        Error::invalid_data(
            "wedeo-gpu-wgpu",
            format!("WGSL parse error: {}", e.emit_to_string(source)),
        )
    })?;

    let mut validator = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::default(),
    );
    validator.validate(&module).map_err(|e| {
        Error::invalid_data("wedeo-gpu-wgpu", format!("WGSL validation error: {e}"))
    })?;
    Ok(())
}

/// Generate a validated structural WGSL compute shader for `ops`.
pub fn generate_wgsl(ops: &SwsOpList) -> Result<GeneratedWgsl> {
    generate_wgsl_internal(ops, true)
}

pub(crate) fn generate_wgsl_internal(
    ops: &SwsOpList,
    allow_bgra_storage_texture: bool,
) -> Result<GeneratedWgsl> {
    let Some(SwsOp::Read {
        layout: read_layout,
        ..
    }) = ops.ops.first().cloned()
    else {
        return Err(Error::invalid_argument(
            "wedeo-gpu-wgpu",
            "SwsOpList must begin with SwsOp::Read",
        ));
    };
    let Some(SwsOp::Write(write_layout)) = ops.ops.last().cloned() else {
        return Err(Error::invalid_argument(
            "wedeo-gpu-wgpu",
            "SwsOpList must end with SwsOp::Write",
        ));
    };

    // Validate layouts are supported by the GPU texture/buffer model.
    let src_plane_count = read_layout.plane_count();
    for p in 0..src_plane_count {
        let _ = plane_texture_format(read_layout, p)?;
    }

    let output_binding = select_output_binding(write_layout, allow_bgra_storage_texture)?;
    let structural_hash = structural_op_list_hash(ops);

    let mut pre_scale_ops = Vec::new();
    let mut between_scale_ops = Vec::new();
    let mut post_scale_ops = Vec::new();
    let mut scale_h: Option<SwsScaler> = None;
    let mut scale_v: Option<SwsScaler> = None;
    let mut seen_h = false;
    let mut seen_v = false;
    let mut num_color_matrices = 0usize;

    let middle = &ops.ops[1..ops.ops.len() - 1];
    for op in middle {
        if matches!(op, SwsOp::ColorMatrix(_)) {
            num_color_matrices += 1;
            if num_color_matrices > MAX_COLOR_MATRICES {
                return Err(Error::unsupported(
                    "wedeo-gpu-wgpu",
                    "SwsOpList exceeds MAX_COLOR_MATRICES (4)",
                ));
            }
        }
        match *op {
            SwsOp::ScaleH { kernel, .. } => {
                scale_h = Some(kernel);
                seen_h = true;
            }
            SwsOp::ScaleV { kernel, .. } => {
                scale_v = Some(kernel);
                seen_v = true;
            }
            ref other => {
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
    if seen_h && !seen_v {
        post_scale_ops = between_scale_ops;
        between_scale_ops = Vec::new();
    }

    let mut wgsl = String::with_capacity(8192);

    // 1. Uniform struct and group(0) resource bindings
    emit_uniforms_and_bindings(&mut wgsl, read_layout, src_plane_count, output_binding)?;

    // 2. Helper functions: math, transfer curves, tone-mapping, scaling kernels, Bayer dither
    emit_math_and_color_helpers(&mut wgsl);

    // 3. Coordinate mapping (crop + DisplayTransform 0/90/180/270 + flip) & raw plane reader
    emit_read_helpers(&mut wgsl, read_layout);

    // 4. Pre-scale pointwise evaluator: `fn eval_pre_scale(ox: u32, oy: u32) -> vec4<f32>`
    let mut mat_cursor = 0usize;
    emit_pre_scale_fn(
        &mut wgsl,
        read_layout,
        write_layout,
        &pre_scale_ops,
        &mut mat_cursor,
    );

    // 5. Full destination pixel evaluator: `fn eval_dst_pixel(dx: u32, dy: u32) -> vec4<f32>`
    emit_dst_pixel_fn(
        &mut wgsl,
        read_layout,
        write_layout,
        scale_h,
        scale_v,
        &between_scale_ops,
        &post_scale_ops,
        &mut mat_cursor,
    );

    // 6. `@compute @workgroup_size(16, 16, 1) fn main(@builtin(global_invocation_id) gid: vec3<u32>)`
    emit_main_entry(&mut wgsl, write_layout, output_binding);

    // Validate generated WGSL via naga before returning.
    validate_wgsl(&wgsl)?;

    Ok(GeneratedWgsl {
        structural_hash,
        source: wgsl,
        entry_point: "main",
        workgroup_size: WORKGROUP_SIZE,
        read_layout,
        write_layout,
        src_plane_count,
        output_binding,
        num_color_matrices,
    })
}

pub(crate) fn select_output_binding(
    write_layout: PixelLayout,
    allow_bgra_storage_texture: bool,
) -> Result<WgslOutputBinding> {
    let plane_count = write_layout.plane_count();
    for p in 0..plane_count {
        let _ = plane_texture_format(write_layout, p)?;
    }

    if matches!(write_layout.packing, PlanePacking::Packed) {
        match (write_layout.order, write_layout.depth) {
            (CompOrder::Rgba | CompOrder::Argb | CompOrder::Abgr, SampleDepth::U8) => {
                return Ok(WgslOutputBinding::StorageTexture(
                    wgpu::TextureFormat::Rgba8Unorm,
                ));
            }
            (CompOrder::Bgra, SampleDepth::U8) if allow_bgra_storage_texture => {
                return Ok(WgslOutputBinding::StorageTexture(
                    wgpu::TextureFormat::Bgra8Unorm,
                ));
            }
            (CompOrder::Rgba, SampleDepth::F32) => {
                return Ok(WgslOutputBinding::StorageTexture(
                    wgpu::TextureFormat::Rgba32Float,
                ));
            }
            _ => {}
        }
    }

    Ok(WgslOutputBinding::PlanarStorageBuffers { plane_count })
}

fn emit_uniforms_and_bindings(
    wgsl: &mut String,
    read_layout: PixelLayout,
    src_plane_count: usize,
    output_binding: WgslOutputBinding,
) -> Result<()> {
    wgsl.push_str(
        "struct SwsUniforms {\n\
         \x20   src_w: u32,\n\
         \x20   src_h: u32,\n\
         \x20   dst_w: u32,\n\
         \x20   dst_h: u32,\n\
         \x20   crop_left: u32,\n\
         \x20   crop_top: u32,\n\
         \x20   crop_w: u32,\n\
         \x20   crop_h: u32,\n\
         \x20   oriented_w: u32,\n\
         \x20   oriented_h: u32,\n\
         \x20   pad0: u32,\n\
         \x20   pad1: u32,\n\
         \x20   rot_a: i32,\n\
         \x20   rot_b: i32,\n\
         \x20   rot_c: i32,\n\
         \x20   rot_d: i32,\n\
         \x20   src_peak_nits: f32,\n\
         \x20   dst_peak_nits: f32,\n\
         \x20   chroma_offset_x: f32,\n\
         \x20   chroma_offset_y: f32,\n\
         \x20   matrix_rows: array<vec4<f32>, 12>,\n\
         }\n\n\
         @group(0) @binding(0) var<uniform> params: SwsUniforms;\n",
    );

    let tex_scalar = match read_layout.depth {
        SampleDepth::U8 | SampleDepth::F32 | SampleDepth::F16 => "f32",
        SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16 => "u32",
    };

    for p in 0..src_plane_count {
        let binding = p + 1;
        let _ = writeln!(
            wgsl,
            "@group(0) @binding({binding}) var src_tex_{p}: texture_2d<{tex_scalar}>;"
        );
    }

    let dst_base = src_plane_count + 1;
    match output_binding {
        WgslOutputBinding::StorageTexture(fmt) => {
            let wgsl_fmt = match fmt {
                wgpu::TextureFormat::Rgba8Unorm => "rgba8unorm",
                wgpu::TextureFormat::Bgra8Unorm => "bgra8unorm",
                wgpu::TextureFormat::Rgba32Float => "rgba32float",
                other => {
                    return Err(Error::unsupported(
                        "wedeo-gpu-wgpu",
                        format!("unsupported WGSL storage texture format {other:?}"),
                    ));
                }
            };
            let _ = writeln!(
                wgsl,
                "@group(0) @binding({dst_base}) var dst_tex_0: texture_storage_2d<{wgsl_fmt}, write>;"
            );
        }
        WgslOutputBinding::PlanarStorageBuffers { plane_count } => {
            for p in 0..plane_count {
                let binding = dst_base + p;
                let _ = writeln!(
                    wgsl,
                    "@group(0) @binding({binding}) var<storage, read_write> dst_plane_{p}: array<u32>;"
                );
            }
        }
    }
    wgsl.push('\n');
    Ok(())
}

fn emit_math_and_color_helpers(wgsl: &mut String) {
    wgsl.push_str(
        "const PI: f32 = 3.141592653589793;\n\
         const SDR_WHITE_NITS: f32 = 203.0;\n\
         const PQ_PEAK_NITS: f32 = 10000.0;\n\
         const HLG_PEAK_NITS: f32 = 1000.0;\n\n\
         fn pq_eotf_nits(e: f32) -> f32 {\n\
         \x20   if (e <= 0.0) { return 0.0; }\n\
         \x20   let ec = min(e, 1.0);\n\
         \x20   let em = pow(ec, 1.0 / 78.84375);\n\
         \x20   if (em <= 0.8359375) { return 0.0; }\n\
         \x20   let num = em - 0.8359375;\n\
         \x20   let den = max(18.8515625 - 18.6875 * em, 1e-12);\n\
         \x20   return PQ_PEAK_NITS * pow(num / den, 1.0 / 0.1593017578125);\n\
         }\n\n\
         fn pq_oetf_nits(nits: f32) -> f32 {\n\
         \x20   if (nits <= 0.0) { return 0.0; }\n\
         \x20   let l = clamp(nits / PQ_PEAK_NITS, 0.0, 1.0);\n\
         \x20   let ln = pow(l, 0.1593017578125);\n\
         \x20   return pow((0.8359375 + 18.8515625 * ln) / (1.0 + 18.6875 * ln), 78.84375);\n\
         }\n\n\
         fn eotf_bt709_ch(e: f32) -> f32 {\n\
         \x20   if (e <= 0.0) { return 0.0; }\n\
         \x20   if (e < 4.5 * 0.018053968) { return e / 4.5; }\n\
         \x20   return pow((e + 0.099296827) / 1.099296827, 1.0 / 0.45);\n\
         }\n\n\
         fn oetf_bt709_ch(l: f32) -> f32 {\n\
         \x20   if (l <= 0.0) { return 0.0; }\n\
         \x20   if (l < 0.018053968) { return 4.5 * l; }\n\
         \x20   return 1.099296827 * pow(l, 0.45) - 0.099296827;\n\
         }\n\n\
         fn eotf_srgb_ch(e: f32) -> f32 {\n\
         \x20   if (e <= 0.0) { return 0.0; }\n\
         \x20   if (e < 12.92 * 0.0031308) { return e / 12.92; }\n\
         \x20   return pow((e + 0.055) / 1.055, 2.4);\n\
         }\n\n\
         fn oetf_srgb_ch(l: f32) -> f32 {\n\
         \x20   if (l <= 0.0) { return 0.0; }\n\
         \x20   if (l < 0.0031308) { return 12.92 * l; }\n\
         \x20   return 1.055 * pow(l, 1.0 / 2.4) - 0.055;\n\
         }\n\n\
         fn eotf_smpte240m_ch(e: f32) -> f32 {\n\
         \x20   if (e <= 0.0) { return 0.0; }\n\
         \x20   if (e < 4.0 * 0.0228) { return e / 4.0; }\n\
         \x20   return pow((e + 0.1115) / 1.1115, 1.0 / 0.45);\n\
         }\n\n\
         fn oetf_smpte240m_ch(l: f32) -> f32 {\n\
         \x20   if (l <= 0.0) { return 0.0; }\n\
         \x20   if (l < 0.0228) { return 4.0 * l; }\n\
         \x20   return 1.1115 * pow(l, 0.45) - 0.1115;\n\
         }\n\n\
         fn hlg_inv_oetf_ch(e: f32) -> f32 {\n\
         \x20   if (e <= 0.0) { return 0.0; }\n\
         \x20   if (e <= 0.5) { return (e * e) / 3.0; }\n\
         \x20   return (exp((e - 0.55991073) / 0.17883277) + 0.28466892) / 12.0;\n\
         }\n\n\
         fn hlg_oetf_ch(s_in: f32) -> f32 {\n\
         \x20   let s = max(s_in, 0.0);\n\
         \x20   if (s <= 1.0 / 12.0) { return sqrt(3.0 * s); }\n\
         \x20   return 0.17883277 * log(12.0 * s - 0.28466892) + 0.55991073;\n\
         }\n\n\
         fn linearize_hlg(rgb: vec3<f32>) -> vec3<f32> {\n\
         \x20   let rs = hlg_inv_oetf_ch(rgb.x);\n\
         \x20   let gs = hlg_inv_oetf_ch(rgb.y);\n\
         \x20   let bs = hlg_inv_oetf_ch(rgb.z);\n\
         \x20   let ys = max(0.2627 * rs + 0.6780 * gs + 0.0593 * bs, 0.0);\n\
         \x20   var scale = 0.0;\n\
         \x20   if (ys > 0.0) {\n\
         \x20       scale = (HLG_PEAK_NITS / SDR_WHITE_NITS) * pow(ys, 0.2);\n\
         \x20   }\n\
         \x20   return vec3<f32>(rs * scale, gs * scale, bs * scale);\n\
         }\n\n\
         fn delinearize_hlg(rgb: vec3<f32>) -> vec3<f32> {\n\
         \x20   let rd = max(rgb.x * SDR_WHITE_NITS, 0.0);\n\
         \x20   let gd = max(rgb.y * SDR_WHITE_NITS, 0.0);\n\
         \x20   let bd = max(rgb.z * SDR_WHITE_NITS, 0.0);\n\
         \x20   let yd = 0.2627 * rd + 0.6780 * gd + 0.0593 * bd;\n\
         \x20   if (yd <= 0.0) { return vec3<f32>(0.0); }\n\
         \x20   let gain = pow(yd / HLG_PEAK_NITS, -1.0 / 6.0) / HLG_PEAK_NITS;\n\
         \x20   return vec3<f32>(hlg_oetf_ch(rd * gain), hlg_oetf_ch(gd * gain), hlg_oetf_ch(bd * gain));\n\
         }\n\n\
         fn tone_map_clip(rgb: vec3<f32>, src_peak_nits: f32, dst_peak_nits: f32) -> vec3<f32> {\n\
         \x20   let _unused = src_peak_nits;\n\
         \x20   let max_rel = max(dst_peak_nits / SDR_WHITE_NITS, 1e-6);\n\
         \x20   let c = max(rgb, vec3<f32>(0.0));\n\
         \x20   let sig = max(c.x, max(c.y, c.z));\n\
         \x20   if (sig <= max_rel) { return c; }\n\
         \x20   return c * (max_rel / sig);\n\
         }\n\n\
         fn hable_curve(x: f32) -> f32 {\n\
         \x20   let a = 0.15;\n\
         \x20   let b = 0.50;\n\
         \x20   let c = 0.10;\n\
         \x20   let d = 0.20;\n\
         \x20   let e = 0.02;\n\
         \x20   let f = 0.30;\n\
         \x20   return ((x * (x * a + b * c) + d * e) / (x * (x * a + b) + d * f)) - (e / f);\n\
         }\n\n\
         fn tone_map_hable(rgb: vec3<f32>, src_peak_nits: f32, dst_peak_nits: f32) -> vec3<f32> {\n\
         \x20   let c = max(rgb, vec3<f32>(0.0));\n\
         \x20   let sig = max(c.x, max(c.y, c.z));\n\
         \x20   if (sig <= 1e-6) { return vec3<f32>(0.0); }\n\
         \x20   let src_peak_rel = max(src_peak_nits / SDR_WHITE_NITS, 1.0);\n\
         \x20   let dst_peak_rel = max(dst_peak_nits / SDR_WHITE_NITS, 1e-6);\n\
         \x20   let mapped = (hable_curve(sig) / hable_curve(src_peak_rel)) * dst_peak_rel;\n\
         \x20   return c * (mapped / sig);\n\
         }\n\n\
         fn tone_map_bt2390(rgb: vec3<f32>, src_peak_nits: f32, dst_peak_nits: f32) -> vec3<f32> {\n\
         \x20   let c = max(rgb, vec3<f32>(0.0));\n\
         \x20   let sig = max(c.x, max(c.y, c.z));\n\
         \x20   if (sig <= 1e-6) { return vec3<f32>(0.0); }\n\
         \x20   let dst_p = max(dst_peak_nits, 1.0);\n\
         \x20   let src_p = max(src_peak_nits, dst_p);\n\
         \x20   let dst_max_rel = dst_p / SDR_WHITE_NITS;\n\
         \x20   if (abs(src_p - dst_p) < 1e-3) {\n\
         \x20       let new_sig = min(sig, dst_max_rel);\n\
         \x20       return c * (new_sig / sig);\n\
         \x20   }\n\
         \x20   let sig_nits = sig * SDR_WHITE_NITS;\n\
         \x20   let lw = pq_oetf_nits(src_p);\n\
         \x20   let lmax = pq_oetf_nits(dst_p);\n\
         \x20   let e_in = pq_oetf_nits(sig_nits);\n\
         \x20   let e1 = clamp(e_in / lw, 0.0, 1.0);\n\
         \x20   let max_lum = clamp(lmax / lw, 0.0, 1.0);\n\
         \x20   let ks = 1.5 * max_lum - 0.5;\n\
         \x20   var e2 = e1;\n\
         \x20   if (ks <= 0.0 || e1 >= ks) {\n\
         \x20       let ks_c = max(ks, 0.0);\n\
         \x20       let denom = max(1.0 - ks_c, 1e-6);\n\
         \x20       let t = clamp((e1 - ks_c) / denom, 0.0, 1.0);\n\
         \x20       let t2 = t * t;\n\
         \x20       let t3 = t2 * t;\n\
         \x20       e2 = (2.0 * t3 - 3.0 * t2 + 1.0) * ks_c\n\
         \x20          + (t3 - 2.0 * t2 + t) * (1.0 - ks_c)\n\
         \x20          + (-2.0 * t3 + 3.0 * t2) * max_lum;\n\
         \x20   }\n\
         \x20   let out_nits = pq_eotf_nits(clamp(e2 * lw, 0.0, 1.0));\n\
         \x20   let new_sig = clamp(out_nits / SDR_WHITE_NITS, 0.0, dst_max_rel);\n\
         \x20   return c * (new_sig / sig);\n\
         }\n\n\
         fn sinc_fn(x: f32) -> f32 {\n\
         \x20   if (x < 1e-6) { return 1.0; }\n\
         \x20   let px = PI * x;\n\
         \x20   return sin(px) / px;\n\
         }\n\n\
         fn kernel_weight(kind: u32, x: f32) -> f32 {\n\
         \x20   if (kind == 1u) {\n\
         \x20       return select(0.0, 1.0, x <= 0.5);\n\
         \x20   }\n\
         \x20   if (kind == 2u) {\n\
         \x20       return max(1.0 - x, 0.0);\n\
         \x20   }\n\
         \x20   if (kind == 3u) {\n\
         \x20       if (x >= 2.0) { return 0.0; }\n\
         \x20       let c = 0.6;\n\
         \x20       let p0 = 6.0;\n\
         \x20       if (x < 1.0) {\n\
         \x20           let p2 = -18.0 + 6.0 * c;\n\
         \x20           let p3 = 12.0 - 6.0 * c;\n\
         \x20           return (p0 + x * x * (p2 + x * p3)) / p0;\n\
         \x20       }\n\
         \x20       let q0 = 24.0 * c;\n\
         \x20       let q1 = -48.0 * c;\n\
         \x20       let q2 = 30.0 * c;\n\
         \x20       let q3 = -6.0 * c;\n\
         \x20       return (q0 + x * (q1 + x * (q2 + x * q3))) / p0;\n\
         \x20   }\n\
         \x20   if (x >= 3.0) { return 0.0; }\n\
         \x20   return sinc_fn(x) * sinc_fn(x / 3.0);\n\
         }\n\n\
         fn bayer_threshold_16x16(x: u32, y: u32) -> f32 {\n\
         \x20   var val: u32 = 0u;\n\
         \x20   var bit: u32 = 0u;\n\
         \x20   for (var s: u32 = 0u; s < 4u; s = s + 1u) {\n\
         \x20       let shift = 3u - s;\n\
         \x20       let xb = (x >> shift) & 1u;\n\
         \x20       let yb = (y >> shift) & 1u;\n\
         \x20       let d = ((xb ^ yb) << 1u) | yb;\n\
         \x20       val = (val << 2u) | d;\n\
         \x20       bit = bit + 1u;\n\
         \x20   }\n\
         \x20   return (f32(val) + 0.5) / 256.0 - 0.5;\n\
         }\n\n",
    );
}

fn emit_read_helpers(wgsl: &mut String, read_layout: PixelLayout) {
    wgsl.push_str(
        "fn map_oriented_to_src(ox: u32, oy: u32) -> vec2<u32> {\n\
         \x20   let cw_i = i32(params.crop_w);\n\
         \x20   let ch_i = i32(params.crop_h);\n\
         \x20   let min_p = select(0, params.rot_a * (cw_i - 1), params.rot_a < 0)\n\
         \x20             + select(0, params.rot_c * (ch_i - 1), params.rot_c < 0);\n\
         \x20   let min_q = select(0, params.rot_b * (cw_i - 1), params.rot_b < 0)\n\
         \x20             + select(0, params.rot_d * (ch_i - 1), params.rot_d < 0);\n\
         \x20   let p_raw = i32(ox) + min_p;\n\
         \x20   let q_raw = i32(oy) + min_q;\n\
         \x20   let x_crop = u32(clamp(params.rot_a * p_raw + params.rot_b * q_raw, 0, max(cw_i - 1, 0)));\n\
         \x20   let y_crop = u32(clamp(params.rot_c * p_raw + params.rot_d * q_raw, 0, max(ch_i - 1, 0)));\n\
         \x20   return vec2<u32>(params.crop_left + x_crop, params.crop_top + y_crop);\n\
         }\n\n\
         fn bilerp_f32(v00: f32, v10: f32, v01: f32, v11: f32, wx: f32, wy: f32) -> f32 {\n\
         \x20   let top = v00 * (1.0 - wx) + v10 * wx;\n\
         \x20   let bot = v01 * (1.0 - wx) + v11 * wx;\n\
         \x20   return top * (1.0 - wy) + bot * wy;\n\
         }\n\n",
    );

    let is_u8 = matches!(read_layout.depth, SampleDepth::U8);
    let scale_expr = if is_u8 { " * 255.0" } else { "" };

    wgsl.push_str("fn read_raw_pixel(sx: u32, sy: u32) -> vec4<f32> {\n");
    wgsl.push_str("    let ix = i32(min(sx, params.src_w - 1u));\n");
    wgsl.push_str("    let iy = i32(min(sy, params.src_h - 1u));\n");

    match read_layout.packing {
        PlanePacking::Packed => match read_layout.order {
            CompOrder::Bgra => {
                // wgpu Bgra8Unorm textureLoad returns logical (R, G, B, A); swizzle back to memory
                // order (B, G, R, A) so SwsOp::Swizzle([2, 1, 0, 3]) operates identically to CPU!
                let _ = writeln!(
                    wgsl,
                    "    let t = vec4<f32>(textureLoad(src_tex_0, vec2<i32>(ix, iy), 0)){scale_expr};\n\
                     \x20   return vec4<f32>(t.z, t.y, t.x, t.w);"
                );
            }
            CompOrder::Rgba | CompOrder::Argb | CompOrder::Abgr => {
                let _ = writeln!(
                    wgsl,
                    "    return vec4<f32>(textureLoad(src_tex_0, vec2<i32>(ix, iy), 0)){scale_expr};"
                );
            }
            CompOrder::Gray => {
                let _ = writeln!(
                    wgsl,
                    "    let y = f32(textureLoad(src_tex_0, vec2<i32>(ix, iy), 0).x){scale_expr};\n\
                     \x20   return vec4<f32>(y, 0.0, 0.0, 1.0);"
                );
            }
            CompOrder::Yuyv | CompOrder::Uyvy => {
                let is_yuyv = matches!(read_layout.order, CompOrder::Yuyv);
                let (y0_comp, u_comp, y1_comp, v_comp) = if is_yuyv {
                    ("x", "y", "z", "w")
                } else {
                    ("y", "x", "w", "z")
                };
                let _ = writeln!(
                    wgsl,
                    "    let cw = i32((params.src_w + 1u) / 2u);\n\
                     \x20   let pair = min(ix / 2, cw - 1);\n\
                     \x20   let is_odd = (ix & 1) != 0;\n\
                     \x20   let m0 = vec4<f32>(textureLoad(src_tex_0, vec2<i32>(pair, iy), 0)){scale_expr};\n\
                     \x20   let y_val = select(m0.{y0_comp}, m0.{y1_comp}, is_odd);\n\
                     \x20   let fx = f32(sx) * 0.5;\n\
                     \x20   let cx0 = min(i32(floor(fx)), cw - 1);\n\
                     \x20   let cx1 = min(cx0 + 1, cw - 1);\n\
                     \x20   let tx = fx - f32(cx0);\n\
                     \x20   let c0 = vec4<f32>(textureLoad(src_tex_0, vec2<i32>(cx0, iy), 0)){scale_expr};\n\
                     \x20   let c1 = vec4<f32>(textureLoad(src_tex_0, vec2<i32>(cx1, iy), 0)){scale_expr};\n\
                     \x20   let u_val = c0.{u_comp} * (1.0 - tx) + c1.{u_comp} * tx;\n\
                     \x20   let v_val = c0.{v_comp} * (1.0 - tx) + c1.{v_comp} * tx;\n\
                     \x20   return vec4<f32>(y_val, u_val, v_val, 1.0);"
                );
            }
            _ => {
                let _ = writeln!(
                    wgsl,
                    "    return vec4<f32>(textureLoad(src_tex_0, vec2<i32>(ix, iy), 0)){scale_expr};"
                );
            }
        },
        PlanePacking::SemiPlanar => {
            let sub_x = matches!(
                read_layout.subsampling,
                Subsampling::Cs420 | Subsampling::Cs422
            );
            let sub_y = matches!(read_layout.subsampling, Subsampling::Cs420);
            let fx_expr = if sub_x {
                "clamp((f32(sx) - params.chroma_offset_x) * 0.5, 0.0, f32(max(cw - 1, 0)))"
            } else {
                "f32(sx)"
            };
            let fy_expr = if sub_y {
                "clamp((f32(sy) - params.chroma_offset_y) * 0.5, 0.0, f32(max(ch - 1, 0)))"
            } else {
                "f32(sy)"
            };
            let a_expr = if !matches!(read_layout.alpha, AlphaMode::None) {
                format!("f32(textureLoad(src_tex_2, vec2<i32>(ix, iy), 0).x){scale_expr}")
            } else {
                "1.0".to_string()
            };
            let _ = writeln!(
                wgsl,
                "    let y_val = f32(textureLoad(src_tex_0, vec2<i32>(ix, iy), 0).x){scale_expr};\n\
                 \x20   let c_dims = vec2<i32>(textureDimensions(src_tex_1));\n\
                 \x20   let cw = max(c_dims.x, 1);\n\
                 \x20   let ch = max(c_dims.y, 1);\n\
                 \x20   let fx = {fx_expr};\n\
                 \x20   let fy = {fy_expr};\n\
                 \x20   let cx0 = clamp(i32(floor(fx)), 0, cw - 1);\n\
                 \x20   let cx1 = clamp(cx0 + 1, 0, cw - 1);\n\
                 \x20   let cy0 = clamp(i32(floor(fy)), 0, ch - 1);\n\
                 \x20   let cy1 = clamp(cy0 + 1, 0, ch - 1);\n\
                 \x20   let wx = fx - f32(cx0);\n\
                 \x20   let wy = fy - f32(cy0);\n\
                 \x20   let uv00 = vec4<f32>(textureLoad(src_tex_1, vec2<i32>(cx0, cy0), 0)){scale_expr};\n\
                 \x20   let uv10 = vec4<f32>(textureLoad(src_tex_1, vec2<i32>(cx1, cy0), 0)){scale_expr};\n\
                 \x20   let uv01 = vec4<f32>(textureLoad(src_tex_1, vec2<i32>(cx0, cy1), 0)){scale_expr};\n\
                 \x20   let uv11 = vec4<f32>(textureLoad(src_tex_1, vec2<i32>(cx1, cy1), 0)){scale_expr};\n\
                 \x20   let u_val = bilerp_f32(uv00.x, uv10.x, uv01.x, uv11.x, wx, wy);\n\
                 \x20   let v_val = bilerp_f32(uv00.y, uv10.y, uv01.y, uv11.y, wx, wy);\n\
                 \x20   let a_val = {a_expr};\n\
                 \x20   return vec4<f32>(y_val, u_val, v_val, a_val);"
            );
        }
        PlanePacking::Planar => {
            if read_layout.model == ColorModel::Gray {
                let a_expr = if !matches!(read_layout.alpha, AlphaMode::None) {
                    format!("f32(textureLoad(src_tex_1, vec2<i32>(ix, iy), 0).x){scale_expr}")
                } else {
                    "1.0".to_string()
                };
                let _ = writeln!(
                    wgsl,
                    "    let y_val = f32(textureLoad(src_tex_0, vec2<i32>(ix, iy), 0).x){scale_expr};\n\
                     \x20   let a_val = {a_expr};\n\
                     \x20   return vec4<f32>(y_val, 0.0, 0.0, a_val);"
                );
            } else if matches!(read_layout.subsampling, Subsampling::Cs444) {
                let a_expr = if !matches!(read_layout.alpha, AlphaMode::None) {
                    format!("f32(textureLoad(src_tex_3, vec2<i32>(ix, iy), 0).x){scale_expr}")
                } else {
                    "1.0".to_string()
                };
                let _ = writeln!(
                    wgsl,
                    "    let c0 = f32(textureLoad(src_tex_0, vec2<i32>(ix, iy), 0).x){scale_expr};\n\
                     \x20   let c1 = f32(textureLoad(src_tex_1, vec2<i32>(ix, iy), 0).x){scale_expr};\n\
                     \x20   let c2 = f32(textureLoad(src_tex_2, vec2<i32>(ix, iy), 0).x){scale_expr};\n\
                     \x20   let a_val = {a_expr};\n\
                     \x20   return vec4<f32>(c0, c1, c2, a_val);"
                );
            } else {
                let sub_y = matches!(read_layout.subsampling, Subsampling::Cs420);
                let fy_expr = if sub_y {
                    "clamp((f32(sy) - params.chroma_offset_y) * 0.5, 0.0, f32(max(ch - 1, 0)))"
                } else {
                    "f32(sy)"
                };
                let a_expr = if !matches!(read_layout.alpha, AlphaMode::None) {
                    format!("f32(textureLoad(src_tex_3, vec2<i32>(ix, iy), 0).x){scale_expr}")
                } else {
                    "1.0".to_string()
                };
                let _ = writeln!(
                    wgsl,
                    "    let c0 = f32(textureLoad(src_tex_0, vec2<i32>(ix, iy), 0).x){scale_expr};\n\
                     \x20   let c_dims = vec2<i32>(textureDimensions(src_tex_1));\n\
                     \x20   let cw = max(c_dims.x, 1);\n\
                     \x20   let ch = max(c_dims.y, 1);\n\
                     \x20   let fx = clamp((f32(sx) - params.chroma_offset_x) * 0.5, 0.0, f32(max(cw - 1, 0)));\n\
                     \x20   let fy = {fy_expr};\n\
                     \x20   let cx0 = clamp(i32(floor(fx)), 0, cw - 1);\n\
                     \x20   let cx1 = clamp(cx0 + 1, 0, cw - 1);\n\
                     \x20   let cy0 = clamp(i32(floor(fy)), 0, ch - 1);\n\
                     \x20   let cy1 = clamp(cy0 + 1, 0, ch - 1);\n\
                     \x20   let wx = fx - f32(cx0);\n\
                     \x20   let wy = fy - f32(cy0);\n\
                     \x20   let u00 = f32(textureLoad(src_tex_1, vec2<i32>(cx0, cy0), 0).x){scale_expr};\n\
                     \x20   let u10 = f32(textureLoad(src_tex_1, vec2<i32>(cx1, cy0), 0).x){scale_expr};\n\
                     \x20   let u01 = f32(textureLoad(src_tex_1, vec2<i32>(cx0, cy1), 0).x){scale_expr};\n\
                     \x20   let u11 = f32(textureLoad(src_tex_1, vec2<i32>(cx1, cy1), 0).x){scale_expr};\n\
                     \x20   let v00 = f32(textureLoad(src_tex_2, vec2<i32>(cx0, cy0), 0).x){scale_expr};\n\
                     \x20   let v10 = f32(textureLoad(src_tex_2, vec2<i32>(cx1, cy0), 0).x){scale_expr};\n\
                     \x20   let v01 = f32(textureLoad(src_tex_2, vec2<i32>(cx0, cy1), 0).x){scale_expr};\n\
                     \x20   let v11 = f32(textureLoad(src_tex_2, vec2<i32>(cx1, cy1), 0).x){scale_expr};\n\
                     \x20   let c1 = bilerp_f32(u00, u10, u01, u11, wx, wy);\n\
                     \x20   let c2 = bilerp_f32(v00, v10, v01, v11, wx, wy);\n\
                     \x20   let a_val = {a_expr};\n\
                     \x20   return vec4<f32>(c0, c1, c2, a_val);"
                );
            }
        }
    }

    wgsl.push_str("}\n\n");
}

fn emit_pointwise_ops(
    wgsl: &mut String,
    ops: &[SwsOp],
    read_layout: PixelLayout,
    write_layout: PixelLayout,
    coord_x_var: &str,
    coord_y_var: &str,
    mat_cursor: &mut usize,
) {
    let has_alpha = !matches!(read_layout.alpha, AlphaMode::None);
    let is_gray = read_layout.model == ColorModel::Gray;
    let dst_quant_max = match write_layout.depth {
        SampleDepth::F32 | SampleDepth::F16 => 1.0f32,
        d => ((1u32 << d.bits()) - 1) as f32,
    };

    for op in ops {
        match *op {
            SwsOp::Unpack { depth, shift } => {
                let max_val = match depth {
                    SampleDepth::F32 | SampleDepth::F16 => 1.0f32,
                    d => ((1u32 << d.bits()) - 1) as f32,
                };
                if shift == 0 {
                    let _ = writeln!(wgsl, "    px = px * (1.0 / {max_val:.1});");
                } else {
                    let _ = writeln!(
                        wgsl,
                        "    px = vec4<f32>(vec4<u32>(max(px, vec4<f32>(0.0))) >> vec4<u32>({shift}u)) * (1.0 / {max_val:.1});"
                    );
                }
                if !has_alpha {
                    wgsl.push_str("    px.w = 1.0;\n");
                }
            }
            SwsOp::Swizzle(s) => {
                const COMP: [&str; 4] = ["x", "y", "z", "w"];
                let c0 = COMP[(s[0] as usize) & 3];
                let c1 = COMP[(s[1] as usize) & 3];
                let c2 = COMP[(s[2] as usize) & 3];
                let c3 = COMP[(s[3] as usize) & 3];
                let _ = writeln!(
                    wgsl,
                    "    px = vec4<f32>(px.{c0}, px.{c1}, px.{c2}, px.{c3});"
                );
            }
            SwsOp::ColorMatrix(_) => {
                let base = (*mat_cursor) * 3;
                *mat_cursor += 1;
                let r0 = base;
                let r1 = base + 1;
                let r2 = base + 2;
                let _ = writeln!(
                    wgsl,
                    "    {{\n\
                     \x20       let m0 = params.matrix_rows[{r0}u];\n\
                     \x20       let m1 = params.matrix_rows[{r1}u];\n\
                     \x20       let m2 = params.matrix_rows[{r2}u];\n\
                     \x20       let c_in = vec4<f32>(px.x, px.y, px.z, 1.0);\n\
                     \x20       px = vec4<f32>(dot(m0, c_in), dot(m1, c_in), dot(m2, c_in), px.w);\n\
                     \x20   }}"
                );
            }
            SwsOp::Linearize(trc) => match color::canonical_transfer(trc) {
                TransferCharacteristic::Linear => {}
                TransferCharacteristic::Bt709 => {
                    wgsl.push_str(
                        "    px = vec4<f32>(eotf_bt709_ch(px.x), eotf_bt709_ch(px.y), eotf_bt709_ch(px.z), px.w);\n",
                    );
                }
                TransferCharacteristic::Srgb => {
                    wgsl.push_str(
                        "    px = vec4<f32>(eotf_srgb_ch(px.x), eotf_srgb_ch(px.y), eotf_srgb_ch(px.z), px.w);\n",
                    );
                }
                TransferCharacteristic::Gamma22 => {
                    wgsl.push_str(
                        "    px = vec4<f32>(pow(max(px.x, 0.0), 2.2), pow(max(px.y, 0.0), 2.2), pow(max(px.z, 0.0), 2.2), px.w);\n",
                    );
                }
                TransferCharacteristic::Gamma28 => {
                    wgsl.push_str(
                        "    px = vec4<f32>(pow(max(px.x, 0.0), 2.8), pow(max(px.y, 0.0), 2.8), pow(max(px.z, 0.0), 2.8), px.w);\n",
                    );
                }
                TransferCharacteristic::Smpte240m => {
                    wgsl.push_str(
                        "    px = vec4<f32>(eotf_smpte240m_ch(px.x), eotf_smpte240m_ch(px.y), eotf_smpte240m_ch(px.z), px.w);\n",
                    );
                }
                TransferCharacteristic::Smpte428 => {
                    wgsl.push_str(
                        "    px = vec4<f32>((52.37 / 48.0) * pow(max(px.x, 0.0), 2.6), (52.37 / 48.0) * pow(max(px.y, 0.0), 2.6), (52.37 / 48.0) * pow(max(px.z, 0.0), 2.6), px.w);\n",
                    );
                }
                TransferCharacteristic::Pq => {
                    wgsl.push_str(
                        "    px = vec4<f32>(pq_eotf_nits(px.x) / SDR_WHITE_NITS, pq_eotf_nits(px.y) / SDR_WHITE_NITS, pq_eotf_nits(px.z) / SDR_WHITE_NITS, px.w);\n",
                    );
                }
                TransferCharacteristic::Hlg => {
                    wgsl.push_str(
                        "    {\n\
                         \x20       let lin = linearize_hlg(px.xyz);\n\
                         \x20       px = vec4<f32>(lin.x, lin.y, lin.z, px.w);\n\
                         \x20   }\n",
                    );
                }
                _ => {}
            },
            SwsOp::Delinearize(trc) => match color::canonical_transfer(trc) {
                TransferCharacteristic::Linear => {}
                TransferCharacteristic::Bt709 => {
                    wgsl.push_str(
                        "    px = vec4<f32>(oetf_bt709_ch(px.x), oetf_bt709_ch(px.y), oetf_bt709_ch(px.z), px.w);\n",
                    );
                }
                TransferCharacteristic::Srgb => {
                    wgsl.push_str(
                        "    px = vec4<f32>(oetf_srgb_ch(px.x), oetf_srgb_ch(px.y), oetf_srgb_ch(px.z), px.w);\n",
                    );
                }
                TransferCharacteristic::Gamma22 => {
                    wgsl.push_str(
                        "    px = vec4<f32>(pow(max(px.x, 0.0), 1.0 / 2.2), pow(max(px.y, 0.0), 1.0 / 2.2), pow(max(px.z, 0.0), 1.0 / 2.2), px.w);\n",
                    );
                }
                TransferCharacteristic::Gamma28 => {
                    wgsl.push_str(
                        "    px = vec4<f32>(pow(max(px.x, 0.0), 1.0 / 2.8), pow(max(px.y, 0.0), 1.0 / 2.8), pow(max(px.z, 0.0), 1.0 / 2.8), px.w);\n",
                    );
                }
                TransferCharacteristic::Smpte240m => {
                    wgsl.push_str(
                        "    px = vec4<f32>(oetf_smpte240m_ch(px.x), oetf_smpte240m_ch(px.y), oetf_smpte240m_ch(px.z), px.w);\n",
                    );
                }
                TransferCharacteristic::Smpte428 => {
                    wgsl.push_str(
                        "    px = vec4<f32>(pow((48.0 / 52.37) * max(px.x, 0.0), 1.0 / 2.6), pow((48.0 / 52.37) * max(px.y, 0.0), 1.0 / 2.6), pow((48.0 / 52.37) * max(px.z, 0.0), 1.0 / 2.6), px.w);\n",
                    );
                }
                TransferCharacteristic::Pq => {
                    wgsl.push_str(
                        "    px = vec4<f32>(pq_oetf_nits(px.x * SDR_WHITE_NITS), pq_oetf_nits(px.y * SDR_WHITE_NITS), pq_oetf_nits(px.z * SDR_WHITE_NITS), px.w);\n",
                    );
                }
                TransferCharacteristic::Hlg => {
                    wgsl.push_str(
                        "    {\n\
                         \x20       let nl = delinearize_hlg(px.xyz);\n\
                         \x20       px = vec4<f32>(nl.x, nl.y, nl.z, px.w);\n\
                         \x20   }\n",
                    );
                }
                _ => {}
            },
            SwsOp::ToneMap(tm) => {
                let fn_name = match tm {
                    ToneMapOp::None => None,
                    ToneMapOp::Clip => Some("tone_map_clip"),
                    ToneMapOp::Hable => Some("tone_map_hable"),
                    ToneMapOp::Bt2390 => Some("tone_map_bt2390"),
                };
                if let Some(f) = fn_name {
                    let _ = writeln!(
                        wgsl,
                        "    {{\n\
                         \x20       let tm = {f}(px.xyz, params.src_peak_nits, params.dst_peak_nits);\n\
                         \x20       px = vec4<f32>(tm.x, tm.y, tm.z, px.w);\n\
                         \x20   }}"
                    );
                }
            }
            SwsOp::BlendAlpha => {
                wgsl.push_str(
                    "    {\n\
                     \x20       let a = clamp(px.w, 0.0, 1.0);\n\
                     \x20       px = vec4<f32>(px.x * a, px.y * a, px.z * a, 1.0);\n\
                     \x20   }\n",
                );
            }
            SwsOp::Dither(DitherMode::Ordered | DitherMode::ErrorDiffusion) => {
                let inv_q = 1.0 / dst_quant_max;
                if is_gray {
                    let _ = writeln!(
                        wgsl,
                        "    {{\n\
                         \x20       let d = bayer_threshold_16x16({coord_x_var} & 15u, {coord_y_var} & 15u) * {inv_q:.9};\n\
                         \x20       let da = bayer_threshold_16x16({coord_x_var} & 15u, ({coord_y_var} + 5u) & 15u) * {inv_q:.9};\n\
                         \x20       px = px + vec4<f32>(d, d, d, da);\n\
                         \x20   }}"
                    );
                } else {
                    let _ = writeln!(
                        wgsl,
                        "    {{\n\
                         \x20       let gx = {coord_x_var} & 15u;\n\
                         \x20       let d0 = bayer_threshold_16x16(gx, ({coord_y_var} + 0u) & 15u) * {inv_q:.9};\n\
                         \x20       let d1 = bayer_threshold_16x16(gx, ({coord_y_var} + 3u) & 15u) * {inv_q:.9};\n\
                         \x20       let d2 = bayer_threshold_16x16(gx, ({coord_y_var} + 2u) & 15u) * {inv_q:.9};\n\
                         \x20       let d3 = bayer_threshold_16x16(gx, ({coord_y_var} + 5u) & 15u) * {inv_q:.9};\n\
                         \x20       px = px + vec4<f32>(d0, d1, d2, d3);\n\
                         \x20   }}"
                    );
                }
            }
            SwsOp::Dither(DitherMode::None) => {}
            SwsOp::Pack(dst_layout) => {
                let shift = layout_shift(dst_layout);
                let max_val = match dst_layout.depth {
                    SampleDepth::F32 | SampleDepth::F16 => 1.0f32,
                    d => ((1u32 << d.bits()) - 1) as f32,
                };
                if matches!(dst_layout.depth, SampleDepth::F32) {
                    wgsl.push_str("    px = clamp(px, vec4<f32>(0.0), vec4<f32>(1.0));\n");
                } else if shift == 0 {
                    let _ = writeln!(
                        wgsl,
                        "    px = clamp(round(px * {max_val:.1}), vec4<f32>(0.0), vec4<f32>({max_val:.1}));"
                    );
                } else {
                    let _ = writeln!(
                        wgsl,
                        "    {{\n\
                         \x20       let q = vec4<u32>(clamp(round(px * {max_val:.1}), vec4<f32>(0.0), vec4<f32>({max_val:.1})));\n\
                         \x20       px = vec4<f32>(q << vec4<u32>({shift}u));\n\
                         \x20   }}"
                    );
                }
            }
            _ => {}
        }
    }
}

fn emit_pre_scale_fn(
    wgsl: &mut String,
    read_layout: PixelLayout,
    write_layout: PixelLayout,
    pre_scale_ops: &[SwsOp],
    mat_cursor: &mut usize,
) {
    wgsl.push_str(
        "fn eval_pre_scale(ox: u32, oy: u32) -> vec4<f32> {\n\
         \x20   let src_xy = map_oriented_to_src(ox, oy);\n\
         \x20   var px = read_raw_pixel(src_xy.x, src_xy.y);\n",
    );
    emit_pointwise_ops(
        wgsl,
        pre_scale_ops,
        read_layout,
        write_layout,
        "ox",
        "oy",
        mat_cursor,
    );
    wgsl.push_str("    return px;\n}\n\n");
}

#[allow(clippy::too_many_arguments)]
fn emit_dst_pixel_fn(
    wgsl: &mut String,
    read_layout: PixelLayout,
    write_layout: PixelLayout,
    scale_h: Option<SwsScaler>,
    scale_v: Option<SwsScaler>,
    between_scale_ops: &[SwsOp],
    post_scale_ops: &[SwsOp],
    mat_cursor: &mut usize,
) {
    // If horizontal scaling is present, emit `eval_h_scaled(dx: u32, sy: u32) -> vec4<f32>`;
    // otherwise `eval_h_scaled` simply delegates to `eval_pre_scale(dx, sy)`.
    if let Some(hk) = scale_h {
        let kind = scaler_tag(hk);
        let base_radius = match hk {
            SwsScaler::Point => 0.5f32,
            SwsScaler::Bilinear => 1.0 - 1e-6,
            SwsScaler::Bicubic => 2.0 - 1e-6,
            SwsScaler::Lanczos => 3.0 - 1e-6,
        };
        let _ = writeln!(
            wgsl,
            "fn eval_h_scaled(dx: u32, sy: u32) -> vec4<f32> {{\n\
             \x20   let src_n = max(params.oriented_w, 1u);\n\
             \x20   let dst_n = max(params.dst_w, 1u);\n\
             \x20   let ratio = f32(dst_n) / f32(src_n);\n\
             \x20   let stretch = select(1.0, 1.0 / ratio, ratio < 1.0 && {kind}u != 1u);\n\
             \x20   let radius = select({base_radius:.6} * stretch, 0.5, {kind}u == 1u);\n\
             \x20   let taps = select(clamp(u32(ceil(radius * 2.0)), 1u, src_n), 1u, {kind}u == 1u);\n\
             \x20   let src_pos = (f32(dx) + 0.5) / ratio - 0.5;\n\
             \x20   var px = vec4<f32>(0.0);\n\
             \x20   if (taps == 1u) {{\n\
             \x20       let idx = u32(clamp(round(src_pos), 0.0, f32(src_n - 1u)));\n\
             \x20       px = eval_pre_scale(idx, sy);\n\
             \x20   }} else {{\n\
             \x20       let max_start = i32(src_n - taps);\n\
             \x20       let start_idx = u32(clamp(i32(ceil(src_pos - radius)), 0, max(max_start, 0)));\n\
             \x20       var w_sum = 0.0;\n\
             \x20       for (var i: u32 = 0u; i < taps; i = i + 1u) {{\n\
             \x20           let sx = min(start_idx + i, src_n - 1u);\n\
             \x20           let dist = abs(f32(sx) - src_pos) / stretch;\n\
             \x20           let w = kernel_weight({kind}u, dist);\n\
             \x20           px = px + eval_pre_scale(sx, sy) * w;\n\
             \x20           w_sum = w_sum + w;\n\
             \x20       }}\n\
             \x20       if (abs(w_sum) > 1e-8) {{\n\
             \x20           px = px * (1.0 / w_sum);\n\
             \x20       }}\n\
             \x20   }}"
        );
        emit_pointwise_ops(
            wgsl,
            between_scale_ops,
            read_layout,
            write_layout,
            "dx",
            "sy",
            mat_cursor,
        );
        wgsl.push_str("    return px;\n}\n\n");
    } else {
        wgsl.push_str(
            "fn eval_h_scaled(dx: u32, sy: u32) -> vec4<f32> {\n\
             \x20   return eval_pre_scale(dx, sy);\n\
             }\n\n",
        );
    }

    wgsl.push_str("fn eval_dst_pixel(dx: u32, dy: u32) -> vec4<f32> {\n");
    if let Some(vk) = scale_v {
        let kind = scaler_tag(vk);
        let base_radius = match vk {
            SwsScaler::Point => 0.5f32,
            SwsScaler::Bilinear => 1.0 - 1e-6,
            SwsScaler::Bicubic => 2.0 - 1e-6,
            SwsScaler::Lanczos => 3.0 - 1e-6,
        };
        let _ = writeln!(
            wgsl,
            "    let src_n = max(params.oriented_h, 1u);\n\
             \x20   let dst_n = max(params.dst_h, 1u);\n\
             \x20   let ratio = f32(dst_n) / f32(src_n);\n\
             \x20   let stretch = select(1.0, 1.0 / ratio, ratio < 1.0 && {kind}u != 1u);\n\
             \x20   let radius = select({base_radius:.6} * stretch, 0.5, {kind}u == 1u);\n\
             \x20   let taps = select(clamp(u32(ceil(radius * 2.0)), 1u, src_n), 1u, {kind}u == 1u);\n\
             \x20   let src_pos = (f32(dy) + 0.5) / ratio - 0.5;\n\
             \x20   var px = vec4<f32>(0.0);\n\
             \x20   if (taps == 1u) {{\n\
             \x20       let idx = u32(clamp(round(src_pos), 0.0, f32(src_n - 1u)));\n\
             \x20       px = eval_h_scaled(dx, idx);\n\
             \x20   }} else {{\n\
             \x20       let max_start = i32(src_n - taps);\n\
             \x20       let start_idx = u32(clamp(i32(ceil(src_pos - radius)), 0, max(max_start, 0)));\n\
             \x20       var w_sum = 0.0;\n\
             \x20       for (var i: u32 = 0u; i < taps; i = i + 1u) {{\n\
             \x20           let sy = min(start_idx + i, src_n - 1u);\n\
             \x20           let dist = abs(f32(sy) - src_pos) / stretch;\n\
             \x20           let w = kernel_weight({kind}u, dist);\n\
             \x20           px = px + eval_h_scaled(dx, sy) * w;\n\
             \x20           w_sum = w_sum + w;\n\
             \x20       }}\n\
             \x20       if (abs(w_sum) > 1e-8) {{\n\
             \x20           px = px * (1.0 / w_sum);\n\
             \x20       }}\n\
             \x20   }}"
        );
    } else {
        wgsl.push_str("    var px = eval_h_scaled(dx, dy);\n");
    }

    emit_pointwise_ops(
        wgsl,
        post_scale_ops,
        read_layout,
        write_layout,
        "dx",
        "dy",
        mat_cursor,
    );
    wgsl.push_str("    return px;\n}\n\n");
}

fn emit_main_entry(
    wgsl: &mut String,
    write_layout: PixelLayout,
    output_binding: WgslOutputBinding,
) {
    wgsl.push_str(
        "@compute @workgroup_size(16, 16, 1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
         \x20   if (gid.x >= params.dst_w || gid.y >= params.dst_h) {\n\
         \x20       return;\n\
         \x20   }\n\
         \x20   let px = eval_dst_pixel(gid.x, gid.y);\n",
    );

    match output_binding {
        WgslOutputBinding::StorageTexture(fmt) => match fmt {
            wgpu::TextureFormat::Rgba8Unorm => {
                wgsl.push_str(
                    "    let out_norm = clamp(px * (1.0 / 255.0), vec4<f32>(0.0), vec4<f32>(1.0));\n\
                     \x20   textureStore(dst_tex_0, vec2<i32>(i32(gid.x), i32(gid.y)), out_norm);\n",
                );
            }
            wgpu::TextureFormat::Bgra8Unorm => {
                // `canonical_to_write_swizzle(BGRA8)` swizzled `px` to `[B, G, R, A]` memory order,
                // while `textureStore` on `bgra8unorm` expects logical `(R, G, B, A)` in `.xyzw`!
                wgsl.push_str(
                    "    let out_norm = clamp(vec4<f32>(px.z, px.y, px.x, px.w) * (1.0 / 255.0), vec4<f32>(0.0), vec4<f32>(1.0));\n\
                     \x20   textureStore(dst_tex_0, vec2<i32>(i32(gid.x), i32(gid.y)), out_norm);\n",
                );
            }
            wgpu::TextureFormat::Rgba32Float => {
                wgsl.push_str(
                    "    textureStore(dst_tex_0, vec2<i32>(i32(gid.x), i32(gid.y)), px);\n",
                );
            }
            _ => {}
        },
        WgslOutputBinding::PlanarStorageBuffers { plane_count } => {
            let luma_idx = "gid.y * params.dst_w + gid.x";
            match write_layout.packing {
                PlanePacking::Packed => match write_layout.order {
                    CompOrder::Gray => {
                        wgsl.push_str(&format!(
                            "    dst_plane_0[{luma_idx}] = u32(clamp(round(px.x), 0.0, 65535.0));\n"
                        ));
                    }
                    CompOrder::Bgra | CompOrder::Rgba | CompOrder::Argb | CompOrder::Abgr => {
                        wgsl.push_str(&format!(
                            "    let b0 = u32(clamp(round(px.x), 0.0, 255.0));\n\
                             \x20   let b1 = u32(clamp(round(px.y), 0.0, 255.0));\n\
                             \x20   let b2 = u32(clamp(round(px.z), 0.0, 255.0));\n\
                             \x20   let b3 = u32(clamp(round(px.w), 0.0, 255.0));\n\
                             \x20   dst_plane_0[{luma_idx}] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);\n"
                        ));
                    }
                    CompOrder::Yuyv | CompOrder::Uyvy => {
                        let is_yuyv = write_layout.order == CompOrder::Yuyv;
                        wgsl.push_str(
                            "    if ((gid.x & 1u) == 0u) {\n\
                             \x20       let x1 = min(gid.x + 1u, params.dst_w - 1u);\n\
                             \x20       let p1 = select(px, eval_dst_pixel(x1, gid.y), x1 != gid.x);\n\
                             \x20       let y0 = u32(clamp(round(px.x), 0.0, 255.0));\n\
                             \x20       let y1 = u32(clamp(round(p1.x), 0.0, 255.0));\n\
                             \x20       let u_v = u32(clamp(round((px.y + p1.y) * 0.5), 0.0, 255.0));\n\
                             \x20       let v_v = u32(clamp(round((px.z + p1.z) * 0.5), 0.0, 255.0));\n\
                             \x20       let cw = (params.dst_w + 1u) / 2u;\n\
                             \x20       let m_idx = gid.y * cw + (gid.x / 2u);\n",
                        );
                        if is_yuyv {
                            wgsl.push_str(
                                "        dst_plane_0[m_idx] = y0 | (u_v << 8u) | (y1 << 16u) | (v_v << 24u);\n    }\n",
                            );
                        } else {
                            wgsl.push_str(
                                "        dst_plane_0[m_idx] = u_v | (y0 << 8u) | (v_v << 16u) | (y1 << 24u);\n    }\n",
                            );
                        }
                    }
                    _ => {}
                },
                PlanePacking::SemiPlanar | PlanePacking::Planar => {
                    if matches!(write_layout.depth, SampleDepth::F32) {
                        wgsl.push_str(&format!(
                            "    dst_plane_0[{luma_idx}] = bitcast<u32>(px.x);\n"
                        ));
                    } else {
                        wgsl.push_str(&format!(
                            "    dst_plane_0[{luma_idx}] = u32(clamp(round(px.x), 0.0, 65535.0));\n"
                        ));
                    }

                    if write_layout.model == ColorModel::Gray {
                        if plane_count > 1 {
                            wgsl.push_str(&format!(
                                "    dst_plane_1[{luma_idx}] = u32(clamp(round(px.w), 0.0, 65535.0));\n"
                            ));
                        }
                    } else {
                        match write_layout.subsampling {
                            Subsampling::Cs444 => {
                                if matches!(write_layout.packing, PlanePacking::SemiPlanar) {
                                    wgsl.push_str(&format!(
                                        "    let u_q = u32(clamp(round(px.y), 0.0, 65535.0));\n\
                                         \x20   let v_q = u32(clamp(round(px.z), 0.0, 65535.0));\n\
                                         \x20   dst_plane_1[({luma_idx}) * 2u] = u_q;\n\
                                         \x20   dst_plane_1[({luma_idx}) * 2u + 1u] = v_q;\n"
                                    ));
                                } else {
                                    wgsl.push_str(&format!(
                                        "    dst_plane_1[{luma_idx}] = u32(clamp(round(px.y), 0.0, 65535.0));\n\
                                         \x20   dst_plane_2[{luma_idx}] = u32(clamp(round(px.z), 0.0, 65535.0));\n"
                                    ));
                                }
                            }
                            Subsampling::Cs422 => {
                                wgsl.push_str(
                                    "    if ((gid.x & 1u) == 0u) {\n\
                                     \x20       let x1 = min(gid.x + 1u, params.dst_w - 1u);\n\
                                     \x20       let p1 = select(px, eval_dst_pixel(x1, gid.y), x1 != gid.x);\n\
                                     \x20       let u_q = u32(clamp(round((px.y + p1.y) * 0.5), 0.0, 65535.0));\n\
                                     \x20       let v_q = u32(clamp(round((px.z + p1.z) * 0.5), 0.0, 65535.0));\n\
                                     \x20       let cw = (params.dst_w + 1u) / 2u;\n\
                                     \x20       let c_idx = gid.y * cw + (gid.x / 2u);\n",
                                );
                                if matches!(write_layout.packing, PlanePacking::SemiPlanar) {
                                    wgsl.push_str(
                                        "        dst_plane_1[c_idx * 2u] = u_q;\n\
                                         \x20       dst_plane_1[c_idx * 2u + 1u] = v_q;\n    }\n",
                                    );
                                } else {
                                    wgsl.push_str(
                                        "        dst_plane_1[c_idx] = u_q;\n\
                                         \x20       dst_plane_2[c_idx] = v_q;\n    }\n",
                                    );
                                }
                            }
                            Subsampling::Cs420 => {
                                wgsl.push_str(
                                    "    if (((gid.x & 1u) == 0u) && ((gid.y & 1u) == 0u)) {\n\
                                     \x20       let x1 = min(gid.x + 1u, params.dst_w - 1u);\n\
                                     \x20       let y1 = min(gid.y + 1u, params.dst_h - 1u);\n\
                                     \x20       let p10 = select(px, eval_dst_pixel(x1, gid.y), x1 != gid.x);\n\
                                     \x20       let p01 = select(px, eval_dst_pixel(gid.x, y1), y1 != gid.y);\n\
                                     \x20       let p11 = select(p01, eval_dst_pixel(x1, y1), x1 != gid.x && y1 != gid.y);\n\
                                     \x20       let u_q = u32(clamp(round((px.y + p10.y + p01.y + p11.y) * 0.25), 0.0, 65535.0));\n\
                                     \x20       let v_q = u32(clamp(round((px.z + p10.z + p01.z + p11.z) * 0.25), 0.0, 65535.0));\n\
                                     \x20       let cw = (params.dst_w + 1u) / 2u;\n\
                                     \x20       let c_idx = (gid.y / 2u) * cw + (gid.x / 2u);\n",
                                );
                                if matches!(write_layout.packing, PlanePacking::SemiPlanar) {
                                    wgsl.push_str(
                                        "        dst_plane_1[c_idx * 2u] = u_q;\n\
                                         \x20       dst_plane_1[c_idx * 2u + 1u] = v_q;\n    }\n",
                                    );
                                } else {
                                    wgsl.push_str(
                                        "        dst_plane_1[c_idx] = u_q;\n\
                                         \x20       dst_plane_2[c_idx] = v_q;\n    }\n",
                                    );
                                }
                            }
                            _ => {}
                        }

                        if !matches!(write_layout.alpha, AlphaMode::None) {
                            let a_plane =
                                if matches!(write_layout.packing, PlanePacking::SemiPlanar) {
                                    2
                                } else {
                                    3
                                };
                            if a_plane < plane_count {
                                wgsl.push_str(&format!(
                                    "    dst_plane_{a_plane}[{luma_idx}] = u32(clamp(round(px.w), 0.0, 65535.0));\n"
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    wgsl.push_str("}\n");
}

pub(crate) fn compile_cached_pipeline(
    device: &wgpu::Device,
    op_list: &SwsOpList,
) -> Result<CachedPipeline> {
    let mut ops = op_list.clone();
    optimizer::optimize(&mut ops);

    let allow_bgra_storage = device
        .features()
        .contains(wgpu::Features::BGRA8UNORM_STORAGE);
    let generated = generate_wgsl_internal(&ops, allow_bgra_storage)?;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wedeo-sws-wgsl"),
        source: wgpu::ShaderSource::Wgsl(generated.source.as_str().into()),
    });

    let mut entries = Vec::with_capacity(1 + generated.src_plane_count + 4);
    // @binding(0): SwsUniforms
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(SWS_UNIFORMS_SIZE as u64),
        },
        count: None,
    });

    let sample_type = match generated.read_layout.depth {
        SampleDepth::U8 => wgpu::TextureSampleType::Float { filterable: false },
        SampleDepth::F32 | SampleDepth::F16 => wgpu::TextureSampleType::Float { filterable: false },
        SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16 => {
            wgpu::TextureSampleType::Uint
        }
    };

    for p in 0..generated.src_plane_count {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (p + 1) as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }

    let dst_base = (generated.src_plane_count + 1) as u32;
    match generated.output_binding {
        WgslOutputBinding::StorageTexture(format) => {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: dst_base,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            });
        }
        WgslOutputBinding::PlanarStorageBuffers { plane_count } => {
            for p in 0..plane_count {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: dst_base + (p as u32),
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
        }
    }

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wedeo-sws-bgl"),
        entries: &entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("wedeo-sws-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wedeo-sws-compute-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(generated.entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok(CachedPipeline {
        structural_hash: generated.structural_hash,
        wgsl_source: Arc::from(generated.source),
        output_binding: generated.output_binding,
        src_plane_count: generated.src_plane_count,
        pipeline,
        bind_group_layout,
    })
}

/// Compiled user-supplied custom WGSL compute pass for the `WgslFilter` framework.
///
/// Binding convention (`@group(0)`):
/// - `@binding(0)`: `var src_tex: texture_2d<f32>;` (input RGBA8/RGBA32F frame)
/// - `@binding(1)`: `var dst_tex: texture_storage_2d<rgba8unorm, write>;` (output frame)
/// - `@binding(2)` (optional, when `has_uniforms == true`): `var<uniform> params: ...;`
#[derive(Debug, Clone)]
pub struct WgslComputePass {
    wgsl_source: Arc<str>,
    entry_point: Arc<str>,
    workgroup_size: [u32; 3],
    has_uniforms: bool,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WgslComputePass {
    /// Validate and compile a custom user WGSL compute shader string on `ctx`.
    pub fn compile(
        ctx: &WgpuContext,
        wgsl_source: &str,
        entry_point: &str,
        workgroup_size: [u32; 3],
        has_uniforms: bool,
    ) -> Result<Self> {
        if workgroup_size[0] == 0 || workgroup_size[1] == 0 || workgroup_size[2] == 0 {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "WgslComputePass workgroup_size dimensions must all be non-zero",
            ));
        }
        validate_wgsl(wgsl_source)?;

        let device = ctx.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wedeo-user-wgsl-filter"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let mut entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ];

        if has_uniforms {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wedeo-user-wgsl-bgl"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wedeo-user-wgsl-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wedeo-user-wgsl-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            wgsl_source: Arc::from(wgsl_source),
            entry_point: Arc::from(entry_point),
            workgroup_size,
            has_uniforms,
            pipeline,
            bind_group_layout,
        })
    }

    #[inline]
    pub fn wgsl_source(&self) -> &str {
        &self.wgsl_source
    }

    #[inline]
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    #[inline]
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    #[inline]
    pub fn has_uniforms(&self) -> bool {
        self.has_uniforms
    }

    #[inline]
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Execute this compiled user WGSL compute pass on an `RGBA8` frame, binding `uniforms`
    /// at `@group(0) @binding(2)` when `self.has_uniforms()` is `true`.
    ///
    /// Preserves the input frame's memory residency (`Cpu` or `Gpu(ctx.device_id())`) and
    /// all VFR timing and side data metadata.
    pub fn execute(
        &self,
        ctx: &WgpuContext,
        src: &VideoFrame,
        uniforms: &[u8],
        pool: &BufferPool,
    ) -> Result<VideoFrame> {
        if src.layout != PixelLayout::RGBA8 {
            return Err(Error::unsupported(
                "wedeo-gpu-wgpu",
                "WgslComputePass expects PixelLayout::RGBA8 frames",
            ));
        }
        let return_cpu = src.domain() == MemoryDomain::Cpu;
        let gpu_src = if return_cpu {
            ctx.upload_frame(src)?
        } else {
            src.clone()
        };

        let src_surf = gpu_src
            .gpu_surface()
            .and_then(|s| WgpuContext::unwrap_surface(s.as_ref()))
            .ok_or_else(|| Error::invalid_argument("wedeo-gpu-wgpu", "missing source texture"))?;
        let src_tex = src_surf.texture().ok_or_else(|| {
            Error::invalid_argument("wedeo-gpu-wgpu", "virtual surface has no backing texture")
        })?;
        let src_view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let dst_tex = ctx.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("wedeo-user-wgsl-dst"),
            size: wgpu::Extent3d {
                width: src.width,
                height: src.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dst_view = dst_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buf = if self.has_uniforms {
            // Pad uniform buffer to at least 16 bytes (multiple of 16 for WebGPU std140 safety).
            let padded_len = uniforms.len().max(16).div_ceil(16) * 16;
            let mut padded = vec![0u8; padded_len];
            padded[..uniforms.len()].copy_from_slice(uniforms);

            let buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("wedeo-user-wgsl-uniforms"),
                size: padded_len as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue().write_buffer(&buf, 0, &padded);
            Some(buf)
        } else {
            None
        };

        let mut bg_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&dst_view),
            },
        ];
        if let Some(ref ubuf) = uniform_buf {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: ubuf.as_entire_binding(),
            });
        }

        let bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wedeo-user-wgsl-bg"),
            layout: &self.bind_group_layout,
            entries: &bg_entries,
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wedeo-user-wgsl-encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wedeo-user-wgsl-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = src.width.div_ceil(self.workgroup_size[0]);
            let wg_y = src.height.div_ceil(self.workgroup_size[1]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        ctx.queue().submit(Some(encoder.finish()));

        let dst_surface = ctx.wrap_texture(dst_tex, PixelLayout::RGBA8);
        let mut gpu_out = VideoFrame::from_gpu(dst_surface);
        crate::context::copy_frame_metadata(src, &mut gpu_out);

        if return_cpu {
            ctx.download_frame(&gpu_out, pool)
        } else {
            Ok(gpu_out)
        }
    }
}
