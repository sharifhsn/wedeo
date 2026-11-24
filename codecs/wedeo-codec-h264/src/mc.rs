#![forbid(unsafe_code)]
// H.264/AVC motion compensation — quarter-pixel luma, eighth-pixel chroma.
//
// Luma uses a 6-tap FIR filter {1, -5, 20, 20, -5, 1} for half-pel positions,
// with bilinear averaging to reach quarter-pel. Chroma uses bilinear
// interpolation at 1/8-pel precision.
//
// This is a clean-room scalar implementation targeting correctness, not speed.
// Reference: ITU-T H.264 section 8.4.2, FFmpeg libavcodec/h264qpel_template.c
// and h264chroma_template.c.

use crate::deblock::PictureBuffer;

// ---------------------------------------------------------------------------
// Pre-allocated scratch buffers for MC (eliminates per-call heap allocations)
// ---------------------------------------------------------------------------

/// Scratch buffers reused across all MC calls within a frame.
///
/// Replaces ~15k-20k short-lived `vec![]` allocations per 1080p frame with
/// 4 persistent buffers totaling ~4 KB.
pub struct McScratch {
    /// Legacy integer reference workspace retained for callers that reuse it.
    pub ref_buf: Vec<i32>,
    /// Intermediate horizontal sums for two-dimensional half-pixel filtering.
    pub ref_buf2: Vec<i32>,
    /// u8 buffer for first half-pel filter output. Worst case: 16×16 = 256.
    pub half_a: Vec<u8>,
    /// Second predictor followed by space for a clamped reference tile.
    pub half_b: Vec<u8>,
}

impl Default for McScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl McScratch {
    pub fn new() -> Self {
        // Pre-allocate for the common worst case (16×16 block with 6-tap filter).
        let max_ref = (16 + 5) * (16 + 5); // 441
        let max_pix = 16 * 16; // 256
        Self {
            ref_buf: vec![0i32; max_ref],
            ref_buf2: vec![0i32; max_ref],
            half_a: vec![0u8; max_pix],
            half_b: vec![0u8; max_pix + max_ref],
        }
    }

    /// Ensure all buffers are large enough for a block of size `w × h`.
    /// Never shrinks — only grows if needed.
    #[inline]
    fn ensure_capacity(&mut self, w: usize, h: usize) {
        let max_ref = (w + 5) * (h + 5);
        let max_pix = w * h;
        if self.ref_buf.len() < max_ref {
            self.ref_buf.resize(max_ref, 0);
        }
        if self.ref_buf2.len() < max_ref {
            self.ref_buf2.resize(max_ref, 0);
        }
        if self.half_a.len() < max_pix {
            self.half_a.resize(max_pix, 0);
        }
        if self.half_b.len() < max_pix {
            self.half_b.resize(max_pix, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel helpers
// ---------------------------------------------------------------------------

/// Read a reference pixel with edge clamping.
#[inline(always)]
fn get_ref_pixel(ref_data: &[u8], stride: usize, x: i32, y: i32, width: i32, height: i32) -> u8 {
    let cx = x.clamp(0, width - 1) as usize;
    let cy = y.clamp(0, height - 1) as usize;
    ref_data[cy * stride + cx]
}

/// Copy a rectangular block. Common widths are passed as literals for specialization.
#[inline(always)]
fn copy_block_direct(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_offset: usize,
    block_w: usize,
    block_h: usize,
) {
    for j in 0..block_h {
        let s = src_offset + j * src_stride;
        let d = j * dst_stride;
        dst[d..d + block_w].copy_from_slice(&src[s..s + block_w]);
    }
}

/// Check if a block plus its filter tap margin fits entirely within the picture.
/// When true, we can read directly from the reference plane without edge clamping.
#[inline(always)]
fn is_interior(
    ref_x: i32,
    ref_y: i32,
    block_w: usize,
    block_h: usize,
    pic_w: i32,
    pic_h: i32,
    tap_margin: i32,
) -> bool {
    ref_x - tap_margin >= 0
        && ref_y - tap_margin >= 0
        && ref_x + block_w as i32 + tap_margin <= pic_w
        && ref_y + block_h as i32 + tap_margin <= pic_h
}

/// Materialize only the taps the requested phase needs. Extend each edge
/// with its closest reference sample, as in FFmpeg's emulated-edge path.
#[allow(clippy::too_many_arguments)]
fn extend_luma_edge(
    dst: &mut [u8],
    width: usize,
    src: &[u8],
    stride: usize,
    x: i64,
    y: i64,
    pic_width: i64,
    pic_height: i64,
) {
    let left = (-x).clamp(0, width as i64) as usize;
    let start = x.clamp(0, pic_width) as usize;
    let middle = (width - left).min(pic_width as usize - start);
    for (i, row) in dst.chunks_exact_mut(width).enumerate() {
        let offset = (y + i as i64).clamp(0, pic_height - 1) as usize * stride;
        if left != 0 {
            row[..left].fill(src[offset]);
        }
        if middle != 0 {
            row[left..left + middle].copy_from_slice(&src[offset + start..offset + start + middle]);
        }
        if left + middle < width {
            row[left + middle..].fill(src[offset + pic_width as usize - 1]);
        }
    }
}

/// Copies a block from the reference luma plane to `dst`, applying sub-pixel
/// interpolation as determined by the fractional MV components `dx`, `dy`
/// (each 0..3, in quarter-pel units).
///
/// The 16 (dx, dy) combinations map to spec positions as follows (FFmpeg naming
/// mc{dx}{dy}):
///
/// ```text
///   dx=0      dx=1       dx=2       dx=3
///   ----      ----       ----       ----
///   mc00(G)   mc10(a)    mc20(b)    mc30(c)
///   mc01(d)   mc11(e)    mc21(f)    mc31(g)
///   mc02(h)   mc12(i)    mc22(j)    mc32(k)
///   mc03(l)   mc13(m)    mc23(n)    mc33(o)
/// ```
///
/// # Arguments
///
/// * `dst` - Output buffer
/// * `dst_stride` - Output row stride
/// * `ref_y` - Reference luma plane data
/// * `ref_stride` - Reference plane stride
/// * `ref_x`, `ref_y_pos` - Integer part of the reference position
/// * `dx` - Fractional x offset (0-3, quarter-pel)
/// * `dy` - Fractional y offset (0-3, quarter-pel)
/// * `block_w`, `block_h` - Block dimensions (typically 4, 8, or 16)
/// * `pic_width`, `pic_height` - Reference frame dimensions for edge clamping
#[allow(clippy::too_many_arguments)] // Matches FFmpeg's MC function signature: dst, ref, position, fraction, block size, frame size
pub fn mc_luma(
    dst: &mut [u8],
    dst_stride: usize,
    ref_y: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    dx: u8,
    dy: u8,
    block_w: usize,
    block_h: usize,
    pic_width: u32,
    pic_height: u32,
    scratch: &mut McScratch,
) {
    assert!(dx < 4 && dy < 4, "invalid luma quarter-pixel position");
    if block_w == 0 || block_h == 0 {
        return;
    }
    scratch.ensure_capacity(block_w, block_h);
    let horizontal = usize::from(dx != 0);
    let vertical = usize::from(dy != 0);
    let x = i64::from(ref_x) - 2 * horizontal as i64;
    let y = i64::from(ref_y_pos) - 2 * vertical as i64;
    let width = block_w + 5 * horizontal;
    let height = block_h + 5 * vertical;
    let pixels = block_w * block_h;

    // Interior samples borrow the picture directly. Border samples get the
    // same filter kernel on a small padded tile in the reusable scratch.
    let inside = x >= 0
        && y >= 0
        && x + width as i64 <= i64::from(pic_width)
        && y + height as i64 <= i64::from(pic_height);
    if !inside && scratch.half_b.len() < pixels + width * height {
        scratch.half_b.resize(pixels + width * height, 0);
    }
    let (b, edge) = scratch.half_b.split_at_mut(pixels);
    let (src, stride) = if inside {
        (&ref_y[y as usize * ref_stride + x as usize..], ref_stride)
    } else {
        let tile = &mut edge[..width * height];
        extend_luma_edge(
            tile,
            width,
            ref_y,
            ref_stride,
            x,
            y,
            pic_width.into(),
            pic_height.into(),
        );
        (&*tile, width)
    };

    let center = 2 * vertical * stride + 2 * horizontal;
    let full = |x: usize, y: usize| &src[center + y * stride + x..];
    let h = |out: &mut [u8], pitch, row| {
        crate::simd::luma_6tap_h(
            out,
            pitch,
            &src[center - 2 + row * stride..],
            stride,
            block_w,
            block_h,
        )
    };
    let v = |out: &mut [u8], pitch, col| {
        crate::simd::luma_6tap_v(
            out,
            pitch,
            &src[center - 2 * stride + col..],
            stride,
            block_w,
            block_h,
        )
    };
    let mut hv = |out: &mut [u8], pitch| {
        crate::simd::luma_6tap_hv(
            out,
            pitch,
            src,
            stride,
            block_w,
            block_h,
            &mut scratch.ref_buf2,
        )
    };
    let a = &mut scratch.half_a[..pixels];

    // Integer and half-pixel positions need just one prediction. The remaining
    // phases average a pair of neighboring integer/half-pixel predictions.
    let (second, second_stride): (&[u8], usize) = match (dx, dy) {
        (0, 0) => {
            match block_w {
                16 => crate::simd::put_pixels_16x16(dst, dst_stride, src, stride, block_h),
                8 => copy_block_direct(dst, dst_stride, src, stride, 0, 8, block_h),
                4 => copy_block_direct(dst, dst_stride, src, stride, 0, 4, block_h),
                _ => copy_block_direct(dst, dst_stride, src, stride, 0, block_w, block_h),
            }
            return;
        }
        (2, 0) => return h(dst, dst_stride, 0),
        (0, 2) => return v(dst, dst_stride, 0),
        (2, 2) => return hv(dst, dst_stride),
        (1 | 3, 0) => {
            h(a, block_w, 0);
            (full(usize::from(dx == 3), 0), stride)
        }
        (0, 1 | 3) => {
            v(a, block_w, 0);
            (full(0, usize::from(dy == 3)), stride)
        }
        (1 | 3, 1 | 3) => {
            h(a, block_w, usize::from(dy == 3));
            v(b, block_w, usize::from(dx == 3));
            (b, block_w)
        }
        (2, 1 | 3) => {
            h(a, block_w, usize::from(dy == 3));
            hv(b, block_w);
            (b, block_w)
        }
        (1 | 3, 2) => {
            v(a, block_w, usize::from(dx == 3));
            hv(b, block_w);
            (b, block_w)
        }
        _ => unreachable!(),
    };
    crate::simd::avg_pixels(
        dst,
        dst_stride,
        a,
        block_w,
        second,
        second_stride,
        block_w,
        block_h,
    );
}

// ---------------------------------------------------------------------------
// B-frame weighted prediction helper
// ---------------------------------------------------------------------------

/// Average `dst` with a second prediction (for B-frame bi-prediction).
///
/// After this call, `dst[i] = (dst[i] + pred2[i] + 1) >> 1`.
pub fn mc_avg(
    dst: &mut [u8],
    dst_stride: usize,
    pred2: &[u8],
    pred2_stride: usize,
    block_w: usize,
    block_h: usize,
) {
    crate::simd::avg_pixels_inplace(dst, dst_stride, pred2, pred2_stride, block_w, block_h);
}

// ---------------------------------------------------------------------------
// Chroma motion compensation (1/8-pel bilinear)
// ---------------------------------------------------------------------------

/// Perform chroma motion compensation (1/8-pel bilinear interpolation).
///
/// Chroma MVs are derived from luma MVs: in 4:2:0, chroma resolution is
/// half of luma. The fractional part has 1/8-pel precision (8 steps per
/// integer chroma pixel).
///
/// The bilinear filter weights are:
/// ```text
///   A = (8 - dx) * (8 - dy)
///   B = dx       * (8 - dy)
///   C = (8 - dx) * dy
///   D = dx       * dy
///   pixel = (A*s00 + B*s10 + C*s01 + D*s11 + 32) >> 6
/// ```
///
/// Matches FFmpeg's `h264_chroma_mc*` (op_put variant).
#[allow(clippy::too_many_arguments)] // Matches FFmpeg's chroma MC signature: dst, ref, position, fraction, block size, frame size
pub fn mc_chroma(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    dx: u8,
    dy: u8,
    block_w: usize,
    block_h: usize,
    pic_width: u32,
    pic_height: u32,
) {
    debug_assert!(dx < 8 && dy < 8);

    chroma_mc_scalar(
        dst,
        dst_stride,
        ref_uv,
        ref_stride,
        ref_x,
        ref_y_pos,
        dx,
        dy,
        block_w,
        block_h,
        pic_width as i32,
        pic_height as i32,
        false,
    );
}

/// Perform chroma MC avg (bi-prediction averaging) into an existing destination.
///
/// Reads the existing dst pixels, performs bilinear MC from `ref_uv`, and
/// averages the two: `dst[i] = (dst[i] + mc_result[i] + 1) >> 1`.
///
/// Used for unweighted B-frame chroma bi-prediction, replacing the
/// mc_chroma-into-tmp + avg_pixels_inplace pattern.
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma_avg(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    dx: u8,
    dy: u8,
    block_w: usize,
    block_h: usize,
    pic_width: u32,
    pic_height: u32,
) {
    debug_assert!(dx < 8 && dy < 8);

    chroma_mc_scalar(
        dst,
        dst_stride,
        ref_uv,
        ref_stride,
        ref_x,
        ref_y_pos,
        dx,
        dy,
        block_w,
        block_h,
        pic_width as i32,
        pic_height as i32,
        true,
    );
}

/// Scalar chroma bilinear MC. When `avg` is true, averages with existing dst.
#[allow(clippy::too_many_arguments)]
fn chroma_mc_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    dx: u8,
    dy: u8,
    block_w: usize,
    block_h: usize,
    pw: i32,
    ph: i32,
    avg: bool,
) {
    let a = (8 - dx as i32) * (8 - dy as i32);
    let b = (dx as i32) * (8 - dy as i32);
    let c = (8 - dx as i32) * (dy as i32);
    let d = (dx as i32) * (dy as i32);

    // Bilinear filter reads at most 1 pixel beyond the block in each direction.
    if is_interior(ref_x, ref_y_pos, block_w, block_h, pw, ph, 1) {
        // Interior fast path — no clamping needed for any pixel access.
        let src_offset = ref_y_pos as usize * ref_stride + ref_x as usize;

        // Dispatch to fixed-width specializations for auto-vectorization.
        if d != 0 {
            if matches!(block_w, 4 | 8 | 16) {
                crate::simd::chroma_mc_bilinear(
                    dst,
                    dst_stride,
                    &ref_uv[src_offset..],
                    ref_stride,
                    block_w,
                    block_h,
                    a,
                    b,
                    c,
                    d,
                    avg,
                );
            } else {
                match block_w {
                    2 => chroma_mc_bilinear(
                        dst, dst_stride, ref_uv, ref_stride, src_offset, 2, block_h, a, b, c, d,
                        avg,
                    ),
                    _ => chroma_mc_bilinear(
                        dst, dst_stride, ref_uv, ref_stride, src_offset, block_w, block_h, a, b, c,
                        d, avg,
                    ),
                }
            }
        } else if b + c != 0 {
            let e = b + c;
            let step = if c != 0 { ref_stride } else { 1 };
            if matches!(block_w, 4 | 8 | 16) {
                crate::simd::chroma_mc_2tap(
                    dst,
                    dst_stride,
                    &ref_uv[src_offset..],
                    ref_stride,
                    block_w,
                    block_h,
                    a,
                    e,
                    step,
                    avg,
                );
            } else {
                match block_w {
                    2 => chroma_mc_2tap(
                        dst, dst_stride, ref_uv, ref_stride, src_offset, 2, block_h, a, e, step,
                        avg,
                    ),
                    _ => chroma_mc_2tap(
                        dst, dst_stride, ref_uv, ref_stride, src_offset, block_w, block_h, a, e,
                        step, avg,
                    ),
                }
            }
        } else if !avg {
            // Pure full-pel copy: a==64, (64*s0+32)>>6 == s0. Use fixed-width memcpy.
            match block_w {
                8 => copy_block_direct(dst, dst_stride, ref_uv, ref_stride, src_offset, 8, block_h),
                4 => copy_block_direct(dst, dst_stride, ref_uv, ref_stride, src_offset, 4, block_h),
                2 => copy_block_direct(dst, dst_stride, ref_uv, ref_stride, src_offset, 2, block_h),
                _ => copy_block_direct(
                    dst, dst_stride, ref_uv, ref_stride, src_offset, block_w, block_h,
                ),
            }
        } else {
            // Full-pel avg: copy + average with existing dst.
            match block_w {
                2 => chroma_mc_copy(
                    dst, dst_stride, ref_uv, ref_stride, src_offset, 2, block_h, a, avg,
                ),
                4 => chroma_mc_copy(
                    dst, dst_stride, ref_uv, ref_stride, src_offset, 4, block_h, a, avg,
                ),
                8 => chroma_mc_copy(
                    dst, dst_stride, ref_uv, ref_stride, src_offset, 8, block_h, a, avg,
                ),
                _ => chroma_mc_copy(
                    dst, dst_stride, ref_uv, ref_stride, src_offset, block_w, block_h, a, avg,
                ),
            }
        }
    } else {
        // Edge fallback — use get_ref_pixel with clamping.
        for j in 0..block_h {
            for i in 0..block_w {
                let sx = ref_x + i as i32;
                let sy = ref_y_pos + j as i32;

                let mc_val = if d != 0 {
                    let s00 = get_ref_pixel(ref_uv, ref_stride, sx, sy, pw, ph) as i32;
                    let s10 = get_ref_pixel(ref_uv, ref_stride, sx + 1, sy, pw, ph) as i32;
                    let s01 = get_ref_pixel(ref_uv, ref_stride, sx, sy + 1, pw, ph) as i32;
                    let s11 = get_ref_pixel(ref_uv, ref_stride, sx + 1, sy + 1, pw, ph) as i32;
                    ((a * s00 + b * s10 + c * s01 + d * s11 + 32) >> 6) as u8
                } else if b + c != 0 {
                    let e = b + c;
                    let (step_x, step_y): (i32, i32) = if c != 0 { (0, 1) } else { (1, 0) };
                    let s0 = get_ref_pixel(ref_uv, ref_stride, sx, sy, pw, ph) as i32;
                    let s1 =
                        get_ref_pixel(ref_uv, ref_stride, sx + step_x, sy + step_y, pw, ph) as i32;
                    ((a * s0 + e * s1 + 32) >> 6) as u8
                } else {
                    let s0 = get_ref_pixel(ref_uv, ref_stride, sx, sy, pw, ph) as i32;
                    ((a * s0 + 32) >> 6) as u8
                };

                let idx = j * dst_stride + i;
                if avg {
                    dst[idx] = ((dst[idx] as u16 + mc_val as u16 + 1) >> 1) as u8;
                } else {
                    dst[idx] = mc_val;
                }
            }
        }
    }
}

/// Scalar chroma bilinear (4-tap).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn chroma_mc_bilinear(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    src_offset: usize,
    block_w: usize,
    block_h: usize,
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    avg: bool,
) {
    for j in 0..block_h {
        let row_off = src_offset + j * ref_stride;
        let dst_row = &mut dst[j * dst_stride..j * dst_stride + block_w];
        for (i, dst_px) in dst_row.iter_mut().enumerate().take(block_w) {
            let off = row_off + i;
            let s00 = ref_uv[off] as i32;
            let s10 = ref_uv[off + 1] as i32;
            let s01 = ref_uv[off + ref_stride] as i32;
            let s11 = ref_uv[off + ref_stride + 1] as i32;
            let val = ((a * s00 + b * s10 + c * s01 + d * s11 + 32) >> 6) as u8;
            if avg {
                *dst_px = ((*dst_px as u16 + val as u16 + 1) >> 1) as u8;
            } else {
                *dst_px = val;
            }
        }
    }
}

/// Scalar chroma 2-tap.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn chroma_mc_2tap(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    src_offset: usize,
    block_w: usize,
    block_h: usize,
    a: i32,
    e: i32,
    step: usize,
    avg: bool,
) {
    for j in 0..block_h {
        let row_off = src_offset + j * ref_stride;
        let dst_row = &mut dst[j * dst_stride..j * dst_stride + block_w];
        for (i, dst_px) in dst_row.iter_mut().enumerate().take(block_w) {
            let off = row_off + i;
            let s0 = ref_uv[off] as i32;
            let s1 = ref_uv[off + step] as i32;
            let val = ((a * s0 + e * s1 + 32) >> 6) as u8;
            if avg {
                *dst_px = ((*dst_px as u16 + val as u16 + 1) >> 1) as u8;
            } else {
                *dst_px = val;
            }
        }
    }
}

/// Scalar chroma full-pel copy.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn chroma_mc_copy(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    src_offset: usize,
    block_w: usize,
    block_h: usize,
    a: i32,
    avg: bool,
) {
    for j in 0..block_h {
        let row_off = src_offset + j * ref_stride;
        let dst_row = &mut dst[j * dst_stride..j * dst_stride + block_w];
        for i in 0..block_w {
            let s0 = ref_uv[row_off + i] as i32;
            let val = ((a * s0 + 32) >> 6) as u8;
            if avg {
                dst_row[i] = ((dst_row[i] as u16 + val as u16 + 1) >> 1) as u8;
            } else {
                dst_row[i] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MV to reference position conversion
// ---------------------------------------------------------------------------

/// Convert a quarter-pel luma MV to integer pixel position and fractional offset.
///
/// # Arguments
///
/// * `mv` - Motion vector in quarter-pel units `[x, y]`
/// * `mb_x`, `mb_y` - Macroblock position (in MB units)
/// * `blk_x`, `blk_y` - 4x4 block offset within the macroblock (0..3 each)
///
/// # Returns
///
/// `(ref_x, ref_y, dx, dy)` where `ref_x`/`ref_y` are integer pixel positions
/// and `dx`/`dy` are fractional offsets in quarter-pel units (0..3).
///
/// Negative MVs are handled correctly via `div_euclid`/`rem_euclid` so that
/// `dx`/`dy` are always non-negative.
pub fn mv_to_ref_pos_luma(
    mv: [i16; 2],
    mb_x: u32,
    mb_y: u32,
    blk_x: u32,
    blk_y: u32,
) -> (i32, i32, u8, u8) {
    let full_qpel_x = (mb_x * 16 + blk_x * 4) as i32 * 4 + mv[0] as i32;
    let full_qpel_y = (mb_y * 16 + blk_y * 4) as i32 * 4 + mv[1] as i32;
    let px = full_qpel_x.div_euclid(4);
    let py = full_qpel_y.div_euclid(4);
    let dx = full_qpel_x.rem_euclid(4) as u8;
    let dy = full_qpel_y.rem_euclid(4) as u8;
    (px, py, dx, dy)
}

/// Convert a luma MV to chroma reference position and fractional offset.
///
/// In 4:2:0, chroma has half the resolution of luma. A luma quarter-pel unit
/// corresponds to a chroma eighth-pel unit (since halving the pixel grid doubles
/// the fractional precision per pixel).
///
/// The conversion is:
/// ```text
/// luma_pos_in_qpel = (mb_x*16 + blk_x*4) * 4 + mv_x
/// chroma_pos_in_eighth = luma_pos_in_qpel
///     (because 1 luma qpel = 1/4 luma pixel = 1/8 chroma pixel when chroma = luma/2)
/// chroma_int = chroma_eighth / 8
/// chroma_frac = chroma_eighth % 8
/// ```
///
/// # Returns
///
/// `(ref_x, ref_y, dx, dy)` where `dx`/`dy` are in 1/8-pel units (0..7).
pub fn mv_to_ref_pos_chroma(
    mv: [i16; 2],
    mb_x: u32,
    mb_y: u32,
    blk_x: u32,
    blk_y: u32,
) -> (i32, i32, u8, u8) {
    // Luma position in quarter-pel units.
    let luma_qpel_x = (mb_x * 16 + blk_x * 4) as i32 * 4 + mv[0] as i32;
    let luma_qpel_y = (mb_y * 16 + blk_y * 4) as i32 * 4 + mv[1] as i32;

    // Chroma position in eighth-pel units = luma position in quarter-pel units,
    // because 1 luma quarter-pel = 1 chroma eighth-pel in 4:2:0.
    let cx_int = luma_qpel_x.div_euclid(8);
    let cx_frac = luma_qpel_x.rem_euclid(8) as u8;
    let cy_int = luma_qpel_y.div_euclid(8);
    let cy_frac = luma_qpel_y.rem_euclid(8) as u8;
    (cx_int, cy_int, cx_frac, cy_frac)
}

// ---------------------------------------------------------------------------
// High-level MC dispatcher for a macroblock partition
// ---------------------------------------------------------------------------

/// Perform motion compensation for a single partition of a macroblock,
/// writing the predicted block into the picture buffer.
///
/// This dispatches luma and both chroma planes, converting the luma MV
/// to chroma coordinates automatically.
#[allow(clippy::too_many_arguments)] // High-level dispatch needs picture buffers, MV, MB position, block position, block size
pub fn mc_block(
    pic: &mut PictureBuffer,
    ref_pic: &PictureBuffer,
    mv: [i16; 2],
    mb_x: u32,
    mb_y: u32,
    blk_x: u32,
    blk_y: u32,
    block_w: usize,
    block_h: usize,
    scratch: &mut McScratch,
) {
    // Luma.
    let (lx, ly, ldx, ldy) = mv_to_ref_pos_luma(mv, mb_x, mb_y, blk_x, blk_y);
    let luma_dst_x = (mb_x * 16 + blk_x * 4) as usize;
    let luma_dst_y = (mb_y * 16 + blk_y * 4) as usize;
    let luma_dst_off = luma_dst_y * pic.y_stride + luma_dst_x;

    mc_luma(
        &mut pic.y[luma_dst_off..],
        pic.y_stride,
        &ref_pic.y,
        ref_pic.y_stride,
        lx,
        ly,
        ldx,
        ldy,
        block_w,
        block_h,
        ref_pic.width,
        ref_pic.height,
        scratch,
    );

    // Chroma (4:2:0 — half resolution each way).
    let chroma_w = block_w / 2;
    let chroma_h = block_h / 2;
    // Only perform chroma MC when the block is at least 2x2 in chroma space
    // (i.e. luma partition is at least 4x4, which is always true for H.264).
    if chroma_w > 0 && chroma_h > 0 {
        let (cx, cy, cdx, cdy) = mv_to_ref_pos_chroma(mv, mb_x, mb_y, blk_x, blk_y);
        let chroma_dst_x = (mb_x * 8 + blk_x * 2) as usize;
        let chroma_dst_y = (mb_y * 8 + blk_y * 2) as usize;
        let chroma_dst_off = chroma_dst_y * pic.uv_stride + chroma_dst_x;
        let chroma_pic_w = ref_pic.width / 2;
        let chroma_pic_h = ref_pic.height / 2;

        mc_chroma(
            &mut pic.u[chroma_dst_off..],
            pic.uv_stride,
            &ref_pic.u,
            ref_pic.uv_stride,
            cx,
            cy,
            cdx,
            cdy,
            chroma_w,
            chroma_h,
            chroma_pic_w,
            chroma_pic_h,
        );

        mc_chroma(
            &mut pic.v[chroma_dst_off..],
            pic.uv_stride,
            &ref_pic.v,
            ref_pic.uv_stride,
            cx,
            cy,
            cdx,
            cdy,
            chroma_w,
            chroma_h,
            chroma_pic_w,
            chroma_pic_h,
        );
    }
}

// ---------------------------------------------------------------------------
// Bi-directional averaging (in-place)
// ---------------------------------------------------------------------------

/// Average dst with src in-place: dst[i] = (dst[i] + src[i] + 1) >> 1.
///
/// Used for unweighted bi-directional prediction (weighted_bipred_idc == 0).
/// `dst` already contains the L0 prediction; `src` is the L1 prediction.
pub fn avg_pixels_inplace(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    crate::simd::avg_pixels_inplace(dst, dst_stride, src, src_stride, w, h);
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple 8x8 test reference frame with a known gradient pattern.
    fn make_test_ref(w: usize, h: usize) -> (Vec<u8>, usize) {
        let stride = w;
        let mut data = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                // Gradient: pixel = (x * 16 + y * 8) mod 256.
                data[y * stride + x] = ((x * 16 + y * 8) % 256) as u8;
            }
        }
        (data, stride)
    }

    #[test]
    fn test_fullpel_copy() {
        let (ref_data, ref_stride) = make_test_ref(16, 16);
        let mut dst = vec![0u8; 4 * 4];
        let mut scratch = McScratch::new();
        mc_luma(
            &mut dst,
            4,
            &ref_data,
            ref_stride,
            2,
            3, // ref position
            0,
            0, // full-pel
            4,
            4, // block size
            16,
            16,
            &mut scratch,
        );
        // Verify each pixel matches the reference at (2,3).
        for j in 0..4 {
            for i in 0..4 {
                let expected = ref_data[(3 + j) * ref_stride + (2 + i)];
                assert_eq!(dst[j * 4 + i], expected, "mismatch at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_horizontal_halfpel() {
        // Construct a 1-row reference where we can manually compute the 6-tap result.
        // Ref: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        // Half-pel at position 2 (with dx=2): filter taps centered on pixel 2.
        // filter6(src[0], src[1], src[2], src[3], src[4], src[5])
        // = 10 - 5*20 + 20*30 + 20*40 - 5*50 + 60
        // = 10 - 100 + 600 + 800 - 250 + 60 = 1120
        // result = clip((1120 + 16) >> 5) = clip(35) = 35
        let ref_data: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let ref_stride = 10;
        let mut dst = [0u8; 1];
        let mut scratch = McScratch::new();
        mc_luma(
            &mut dst,
            1,
            &ref_data,
            ref_stride,
            2,
            0, // ref position (integer part)
            2,
            0, // dx=2 (half-pel horizontal), dy=0
            1,
            1, // 1x1 block
            10,
            1,
            &mut scratch,
        );
        // Expected: clip((10 - 100 + 600 + 800 - 250 + 60 + 16) >> 5) = clip(1136 >> 5) = clip(35) = 35
        assert_eq!(dst[0], 35);
    }

    #[test]
    fn test_vertical_halfpel() {
        // 1-column reference, same values vertically.
        let ref_stride = 1;
        let ref_data: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let mut dst = [0u8; 1];
        let mut scratch = McScratch::new();
        mc_luma(
            &mut dst,
            1,
            &ref_data,
            ref_stride,
            0,
            2, // ref position
            0,
            2, // dy=2 (half-pel vertical)
            1,
            1,
            1,
            10,
            &mut scratch,
        );
        // Same calculation as horizontal: 35
        assert_eq!(dst[0], 35);
    }

    #[test]
    fn test_2d_halfpel() {
        // Test the "j" position (dx=2, dy=2).
        // Use a larger reference with known values. The 2D filter is the
        // horizontal 6-tap applied first, then vertical 6-tap on the intermediate.
        // For a uniform reference (all pixels = 128), the output must also be 128.
        let w = 16;
        let h = 16;
        let ref_data = vec![128u8; w * h];
        let mut dst = [0u8; 4 * 4];
        let mut scratch = McScratch::new();
        mc_luma(
            &mut dst,
            4,
            &ref_data,
            w,
            4,
            4,
            2,
            2,
            4,
            4,
            w as u32,
            h as u32,
            &mut scratch,
        );
        for p in &dst {
            assert_eq!(*p, 128, "uniform reference should produce uniform output");
        }
    }

    #[test]
    fn test_quarterpel_averaging() {
        // For dx=1,dy=0: result = avg(full_pel, h_lowpass).
        // With a uniform reference of 100, both full-pel and h_lowpass give 100,
        // so avg = (100 + 100 + 1) >> 1 = 100.
        let w = 16;
        let h = 16;
        let ref_data = vec![100u8; w * h];
        let mut dst = [0u8; 4 * 4];
        let mut scratch = McScratch::new();
        mc_luma(
            &mut dst,
            4,
            &ref_data,
            w,
            4,
            4,
            1,
            0,
            4,
            4,
            w as u32,
            h as u32,
            &mut scratch,
        );
        for p in &dst {
            assert_eq!(*p, 100);
        }
    }

    #[test]
    fn test_chroma_fullpel() {
        // dx=0, dy=0: simple copy (weight A=64, others=0).
        let ref_data: Vec<u8> = (0..64).collect();
        let mut dst = [0u8; 4 * 4];
        mc_chroma(&mut dst, 4, &ref_data, 8, 1, 1, 0, 0, 4, 4, 8, 8);
        for j in 0..4 {
            for i in 0..4 {
                let expected = ref_data[(1 + j) * 8 + (1 + i)];
                assert_eq!(
                    dst[j * 4 + i],
                    expected,
                    "chroma fullpel mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_chroma_bilinear_center() {
        // dx=4, dy=4: all four weights equal = 4*4 = 16 each.
        // pixel = (16*s00 + 16*s10 + 16*s01 + 16*s11 + 32) >> 6
        //       = (16*(s00+s10+s01+s11) + 32) >> 6
        // With uniform value 80: (16*320 + 32) >> 6 = 5152 >> 6 = 80
        let ref_data = vec![80u8; 8 * 8];
        let mut dst = [0u8; 2 * 2];
        mc_chroma(&mut dst, 2, &ref_data, 8, 2, 2, 4, 4, 2, 2, 8, 8);
        for p in &dst {
            assert_eq!(*p, 80);
        }
    }

    #[test]
    fn test_chroma_bilinear_nonuniform() {
        // 2x2 reference region:
        //   [10, 20]
        //   [30, 40]
        // With dx=4, dy=4 (center):
        // pixel = (16*10 + 16*20 + 16*30 + 16*40 + 32) >> 6
        //       = (160 + 320 + 480 + 640 + 32) >> 6
        //       = 1632 >> 6 = 25
        let ref_data: Vec<u8> = vec![10, 20, 0, 0, 30, 40, 0, 0, 0, 0, 0, 0];
        let mut dst = [0u8; 1];
        mc_chroma(&mut dst, 1, &ref_data, 4, 0, 0, 4, 4, 1, 1, 4, 3);
        assert_eq!(dst[0], 25);
    }

    #[test]
    fn test_edge_clamping() {
        // MV pointing outside the frame: reference position (-1, -1).
        // The edge-clamped pixel at (-1,-1) should be ref[0,0].
        let ref_data = vec![42u8; 4 * 4];
        let mut dst = [0u8; 2 * 2];
        let mut scratch = McScratch::new();
        mc_luma(
            &mut dst,
            2,
            &ref_data,
            4,
            -1,
            -1,
            0,
            0,
            2,
            2,
            4,
            4,
            &mut scratch,
        );
        assert_eq!(dst[0], 42);
        assert_eq!(dst[1], 42);
        assert_eq!(dst[2], 42);
        assert_eq!(dst[3], 42);
    }

    #[test]
    fn test_edge_clamping_bottom_right() {
        // MV pointing past the bottom-right corner.
        let ref_data = vec![99u8; 8 * 8];
        let mut dst = [0u8; 2 * 2];
        let mut scratch = McScratch::new();
        mc_luma(
            &mut dst,
            2,
            &ref_data,
            8,
            7,
            7, // starts at last pixel
            0,
            0,
            2,
            2,
            8,
            8,
            &mut scratch,
        );
        // All clamped to ref[7,7] = 99.
        for p in &dst {
            assert_eq!(*p, 99);
        }
    }

    #[test]
    fn test_mv_to_ref_pos_luma_positive() {
        // MV = [5, 9] in quarter-pel. MB at (1, 2), block offset (1, 1).
        // Luma pixel = mb*16 + blk*4 = (16+4, 32+4) = (20, 36).
        // In quarter-pel: (80+5, 144+9) = (85, 153).
        // Integer: 85/4=21, frac: 85%4=1 ; 153/4=38, frac: 153%4=1
        let (px, py, dx, dy) = mv_to_ref_pos_luma([5, 9], 1, 2, 1, 1);
        assert_eq!(px, 21);
        assert_eq!(py, 38);
        assert_eq!(dx, 1);
        assert_eq!(dy, 1);
    }

    #[test]
    fn test_mv_to_ref_pos_luma_negative() {
        // MV = [-3, -7]. MB at (0, 0), block (0, 0).
        // Quarter-pel: (0-3, 0-7) = (-3, -7).
        // div_euclid(-3, 4) = -1, rem_euclid(-3, 4) = 1
        // div_euclid(-7, 4) = -2, rem_euclid(-7, 4) = 1
        let (px, py, dx, dy) = mv_to_ref_pos_luma([-3, -7], 0, 0, 0, 0);
        assert_eq!(px, -1);
        assert_eq!(py, -2);
        assert_eq!(dx, 1);
        assert_eq!(dy, 1);
    }

    #[test]
    fn test_mv_to_ref_pos_chroma() {
        // MV = [8, 16] in quarter-pel. MB at (0, 0), block (0, 0).
        // Luma qpel = (0+8, 0+16) = (8, 16).
        // Chroma eighth-pel = luma qpel = (8, 16).
        // Chroma int = 8/8=1, frac = 8%8=0 ; 16/8=2, frac = 16%8=0
        let (cx, cy, cdx, cdy) = mv_to_ref_pos_chroma([8, 16], 0, 0, 0, 0);
        assert_eq!(cx, 1);
        assert_eq!(cy, 2);
        assert_eq!(cdx, 0);
        assert_eq!(cdy, 0);
    }

    #[test]
    fn test_mv_to_ref_pos_chroma_fractional() {
        // MV = [3, 5] in quarter-pel. MB at (0, 0), block (0, 0).
        // Luma qpel = (3, 5). Chroma eighth = (3, 5).
        // 3/8 = 0 rem 3; 5/8 = 0 rem 5.
        let (cx, cy, cdx, cdy) = mv_to_ref_pos_chroma([3, 5], 0, 0, 0, 0);
        assert_eq!(cx, 0);
        assert_eq!(cy, 0);
        assert_eq!(cdx, 3);
        assert_eq!(cdy, 5);
    }

    #[test]
    fn test_all_16_qpel_positions_uniform() {
        // For a uniform reference, every quarter-pel position should produce
        // the same value (the filter is unity-gain for DC).
        let w = 32;
        let h = 32;
        let val = 77u8;
        let ref_data = vec![val; w * h];
        let mut scratch = McScratch::new();

        for dy in 0..4u8 {
            for dx in 0..4u8 {
                let mut dst = [0u8; 4 * 4];
                mc_luma(
                    &mut dst,
                    4,
                    &ref_data,
                    w,
                    8,
                    8,
                    dx,
                    dy,
                    4,
                    4,
                    w as u32,
                    h as u32,
                    &mut scratch,
                );
                for (idx, p) in dst.iter().enumerate() {
                    assert_eq!(
                        *p, val,
                        "uniform ref failed for dx={}, dy={}, pixel {}",
                        dx, dy, idx
                    );
                }
            }
        }
    }

    #[test]
    fn test_chroma_edge_clamping() {
        // Chroma MC with position outside the frame.
        let ref_data = vec![55u8; 4 * 4];
        let mut dst = [0u8; 2 * 2];
        mc_chroma(&mut dst, 2, &ref_data, 4, -1, -1, 0, 0, 2, 2, 4, 4);
        // All clamped to corner pixel.
        for p in &dst {
            assert_eq!(*p, 55);
        }
    }

    #[test]
    fn test_horizontal_halfpel_4x4() {
        // 4x4 block with a horizontal ramp.
        // Reference: each row is [0, 10, 20, 30, 40, 50, 60, 70, 80].
        // Horizontal half-pel at x=2: taps on [0, 10, 20, 30, 40, 50].
        // filter6(0, 10, 20, 30, 40, 50) = 0 - 50 + 400 + 600 - 200 + 50 = 800
        // (800 + 16) >> 5 = 25
        let w = 9;
        let h = 4;
        let mut ref_data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                ref_data[y * w + x] = (x * 10) as u8;
            }
        }
        let mut dst = [0u8; 4 * 4];
        let mut scratch = McScratch::new();
        mc_luma(
            &mut dst,
            4,
            &ref_data,
            w,
            2,
            0,
            2,
            0, // horizontal half-pel
            4,
            4,
            w as u32,
            h as u32,
            &mut scratch,
        );
        // Row 0, col 0: filter6(0, 10, 20, 30, 40, 50) = 800, (816)>>5 = 25
        assert_eq!(dst[0], 25);
        // Row 0, col 1: filter6(10, 20, 30, 40, 50, 60) = 10-100+600+800-250+60 = 1120, (1136)>>5 = 35
        assert_eq!(dst[1], 35);
    }

    #[test]
    fn test_filter6_known_values() {
        // A 32-level impulse cancels the filter's shift by five.
        for (tap, weight) in [1, -5, 20, 20, -5, 1].into_iter().enumerate() {
            let mut src = [128; 6];
            src[tap] += 32;
            let mut dst = [0];
            crate::simd::luma_6tap_h(&mut dst, 1, &src, 6, 1, 1);
            assert_eq!(i32::from(dst[0]) - 128, weight);
        }
        for value in [0, 1, 32] {
            let mut dst = [255];
            crate::simd::luma_6tap_h(&mut dst, 1, &[value; 6], 6, 1, 1);
            assert_eq!(dst[0], value);
        }
    }
}
