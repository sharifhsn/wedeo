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
    /// i32 buffer for extract_ref_block output.
    /// Worst case: (16+5) × (16+5) = 441 i32s (for hv_lowpass src).
    pub ref_buf: Vec<i32>,
    /// Second i32 buffer for functions needing two extract_ref_block calls
    /// or hv_lowpass tmp.
    pub ref_buf2: Vec<i32>,
    /// u8 buffer for first half-pel filter output. Worst case: 16×16 = 256.
    pub half_a: Vec<u8>,
    /// u8 buffer for second half-pel output (avg_pixels needs two live).
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
            half_b: vec![0u8; max_pix],
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

/// Clamp a value to [0, 255].
#[inline(always)]
fn clip_u8(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

/// Read a reference pixel with edge clamping.
#[inline(always)]
fn get_ref_pixel(ref_data: &[u8], stride: usize, x: i32, y: i32, width: i32, height: i32) -> u8 {
    let cx = x.clamp(0, width - 1) as usize;
    let cy = y.clamp(0, height - 1) as usize;
    ref_data[cy * stride + cx]
}

// ---------------------------------------------------------------------------
// 6-tap FIR half-pel filter (H.264 spec coefficients: 1, -5, 20, 20, -5, 1)
// ---------------------------------------------------------------------------

/// Apply the 6-tap FIR filter to six samples.
///
/// Returns the unshifted, unclipped filter output. The caller must add the
/// rounding offset and shift right by 5 (for direct half-pel) or 10 (for the
/// second pass of the 2D HV filter).
#[inline(always)]
fn filter6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

// ---------------------------------------------------------------------------
// Edge-clamped reference block extraction
// ---------------------------------------------------------------------------

/// Copy a reference block with edge clamping into a contiguous temporary
/// buffer. The output has `out_w` columns and `out_h` rows, stride = `out_w`.
///
/// `rx`, `ry` are the top-left position (may be negative for edge cases).
#[allow(clippy::too_many_arguments)] // MC primitives inherently need position + dimension + picture size params
fn extract_ref_block(
    out: &mut [i32],
    ref_data: &[u8],
    ref_stride: usize,
    rx: i32,
    ry: i32,
    out_w: usize,
    out_h: usize,
    pic_w: i32,
    pic_h: i32,
) {
    for j in 0..out_h {
        let sy = ry + j as i32;
        for i in 0..out_w {
            let sx = rx + i as i32;
            out[j * out_w + i] = get_ref_pixel(ref_data, ref_stride, sx, sy, pic_w, pic_h) as i32;
        }
    }
}

// ---------------------------------------------------------------------------
// Luma half-pel lowpass filters
// ---------------------------------------------------------------------------

/// Horizontal half-pel lowpass: apply the 6-tap filter across each row.
///
/// Input: `src` is an edge-clamped block of size `(block_w + 5) x block_h`,
/// stride = `src_stride`.
/// Output: `dst` is `block_w x block_h`, stride = `dst_stride`.
/// Each output sample is `clip((filter6(...) + 16) >> 5)`.
fn h_lowpass(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[i32],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
) {
    for j in 0..block_h {
        for i in 0..block_w {
            let s = &src[j * src_stride + i..];
            let val = filter6(s[0], s[1], s[2], s[3], s[4], s[5]);
            dst[j * dst_stride + i] = clip_u8((val + 16) >> 5);
        }
    }
}

/// Vertical half-pel lowpass: apply the 6-tap filter down each column.
///
/// Input: `src` is an edge-clamped block of size `block_w x (block_h + 5)`,
/// stride = `src_stride`.
/// Output: `dst` is `block_w x block_h`, stride = `dst_stride`.
fn v_lowpass(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[i32],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
) {
    for i in 0..block_w {
        for j in 0..block_h {
            let a = src[(j) * src_stride + i];
            let b = src[(j + 1) * src_stride + i];
            let c = src[(j + 2) * src_stride + i];
            let d = src[(j + 3) * src_stride + i];
            let e = src[(j + 4) * src_stride + i];
            let f = src[(j + 5) * src_stride + i];
            let val = filter6(a, b, c, d, e, f);
            dst[j * dst_stride + i] = clip_u8((val + 16) >> 5);
        }
    }
}

/// 2D half-pel lowpass (the "j" position in the spec): horizontal first pass
/// into a temporary i16 buffer, then vertical second pass.
///
/// Input: `src` is an edge-clamped block of size `(block_w + 5) x (block_h + 5)`,
/// stride = `src_stride`.
/// Output: `dst` is `block_w x block_h`, stride = `dst_stride`.
fn hv_lowpass(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[i32],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
    tmp: &mut [i32],
) {
    // First pass: horizontal filter into tmp.
    // We need (block_h + 5) rows of horizontally-filtered output so the
    // vertical pass has enough taps.
    let tmp_h = block_h + 5;
    let tmp_stride = block_w;

    for j in 0..tmp_h {
        for i in 0..block_w {
            let s = &src[j * src_stride + i..];
            // No shift/clip here — the intermediate values stay as i32.
            tmp[j * tmp_stride + i] = filter6(s[0], s[1], s[2], s[3], s[4], s[5]);
        }
    }

    // Second pass: vertical filter on the tmp buffer.
    // The rounding for the two-pass filter is (512) >> 10.
    for i in 0..block_w {
        for j in 0..block_h {
            let a = tmp[(j) * tmp_stride + i];
            let b = tmp[(j + 1) * tmp_stride + i];
            let c = tmp[(j + 2) * tmp_stride + i];
            let d = tmp[(j + 3) * tmp_stride + i];
            let e = tmp[(j + 4) * tmp_stride + i];
            let f = tmp[(j + 5) * tmp_stride + i];
            let val = filter6(a, b, c, d, e, f);
            dst[j * dst_stride + i] = clip_u8((val + 512) >> 10);
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel averaging (for quarter-pel from half-pel + integer/half-pel)
// ---------------------------------------------------------------------------

/// Average two pixel buffers: `dst[i] = (a[i] + b[i] + 1) >> 1`.
#[allow(clippy::too_many_arguments)] // Two source buffers plus destination each need data + stride
fn avg_pixels(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    block_w: usize,
    block_h: usize,
) {
    for j in 0..block_h {
        for i in 0..block_w {
            let va = a[j * a_stride + i] as u16;
            let vb = b[j * b_stride + i] as u16;
            dst[j * dst_stride + i] = ((va + vb + 1) >> 1) as u8;
        }
    }
}

/// Copy a block of pixels.
#[allow(dead_code)] // Will be used by MC callers for direct block copies
fn copy_block(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
) {
    for j in 0..block_h {
        dst[j * dst_stride..j * dst_stride + block_w]
            .copy_from_slice(&src[j * src_stride..j * src_stride + block_w]);
    }
}

// ---------------------------------------------------------------------------
// Quarter-pel luma helper functions
// ---------------------------------------------------------------------------

/// Group B helper: compute h_lowpass, then average with a full-pel column
/// extracted at column offset `full_pel_col_offset` within the extracted block.
///
/// `full_pel_col_offset` is 2 for (1,0) and 3 for (3,0).
#[inline]
#[allow(clippy::too_many_arguments)]
fn qpel_h_avg(
    dst: &mut [u8],
    dst_stride: usize,
    ref_y: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    block_w: usize,
    block_h: usize,
    pw: i32,
    ph: i32,
    full_pel_col_offset: usize,
    scratch: &mut McScratch,
) {
    let src_w = block_w + 5;
    let src_h = block_h;
    let needed = src_w * src_h;
    scratch.ref_buf[..needed].fill(0);
    extract_ref_block(
        &mut scratch.ref_buf[..needed],
        ref_y,
        ref_stride,
        ref_x - 2,
        ref_y_pos,
        src_w,
        src_h,
        pw,
        ph,
    );

    let pix = block_w * block_h;
    scratch.half_a[..pix].fill(0);
    h_lowpass(
        &mut scratch.half_a[..pix],
        block_w,
        &scratch.ref_buf[..needed],
        src_w,
        block_w,
        block_h,
    );

    scratch.half_b[..pix].fill(0);
    for j in 0..block_h {
        for i in 0..block_w {
            scratch.half_b[j * block_w + i] =
                scratch.ref_buf[j * src_w + i + full_pel_col_offset] as u8;
        }
    }
    avg_pixels(
        dst,
        dst_stride,
        &scratch.half_b[..pix],
        block_w,
        &scratch.half_a[..pix],
        block_w,
        block_w,
        block_h,
    );
}

/// Group B helper: compute v_lowpass, then average with a full-pel row
/// extracted at row offset `full_pel_row_offset` within the extracted block.
///
/// `full_pel_row_offset` is 2 for (0,1) and 3 for (0,3).
#[inline]
#[allow(clippy::too_many_arguments)]
fn qpel_v_avg(
    dst: &mut [u8],
    dst_stride: usize,
    ref_y: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    block_w: usize,
    block_h: usize,
    pw: i32,
    ph: i32,
    full_pel_row_offset: usize,
    scratch: &mut McScratch,
) {
    let src_w = block_w;
    let src_h = block_h + 5;
    let needed = src_w * src_h;
    scratch.ref_buf[..needed].fill(0);
    extract_ref_block(
        &mut scratch.ref_buf[..needed],
        ref_y,
        ref_stride,
        ref_x,
        ref_y_pos - 2,
        src_w,
        src_h,
        pw,
        ph,
    );

    let pix = block_w * block_h;
    scratch.half_a[..pix].fill(0);
    v_lowpass(
        &mut scratch.half_a[..pix],
        block_w,
        &scratch.ref_buf[..needed],
        src_w,
        block_w,
        block_h,
    );

    scratch.half_b[..pix].fill(0);
    for j in 0..block_h {
        for i in 0..block_w {
            scratch.half_b[j * block_w + i] =
                scratch.ref_buf[(j + full_pel_row_offset) * src_w + i] as u8;
        }
    }
    avg_pixels(
        dst,
        dst_stride,
        &scratch.half_b[..pix],
        block_w,
        &scratch.half_a[..pix],
        block_w,
        block_w,
        block_h,
    );
}

/// Group C helper: diagonal quarter-pel = avg(h_lowpass, v_lowpass).
///
/// `h_row_delta` is the vertical offset added to `ref_y_pos` for the
/// horizontal half-pel extraction (0 for top-adjacent, +1 for below).
/// `v_col_delta` is the horizontal offset added to `ref_x` for the
/// vertical half-pel extraction (0 for current col, +1 for right).
#[inline]
#[allow(clippy::too_many_arguments)]
fn qpel_diagonal(
    dst: &mut [u8],
    dst_stride: usize,
    ref_y: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    block_w: usize,
    block_h: usize,
    pw: i32,
    ph: i32,
    h_row_delta: i32,
    v_col_delta: i32,
    scratch: &mut McScratch,
) {
    let pix = block_w * block_h;

    // First extract → ref_buf, h_lowpass → half_a
    let src_h_w = block_w + 5;
    let src_h_h = block_h;
    let needed_h = src_h_w * src_h_h;
    scratch.ref_buf[..needed_h].fill(0);
    extract_ref_block(
        &mut scratch.ref_buf[..needed_h],
        ref_y,
        ref_stride,
        ref_x - 2,
        ref_y_pos + h_row_delta,
        src_h_w,
        src_h_h,
        pw,
        ph,
    );
    scratch.half_a[..pix].fill(0);
    h_lowpass(
        &mut scratch.half_a[..pix],
        block_w,
        &scratch.ref_buf[..needed_h],
        src_h_w,
        block_w,
        block_h,
    );

    // Second extract → ref_buf2, v_lowpass → half_b
    let src_v_w = block_w;
    let src_v_h = block_h + 5;
    let needed_v = src_v_w * src_v_h;
    scratch.ref_buf2[..needed_v].fill(0);
    extract_ref_block(
        &mut scratch.ref_buf2[..needed_v],
        ref_y,
        ref_stride,
        ref_x + v_col_delta,
        ref_y_pos - 2,
        src_v_w,
        src_v_h,
        pw,
        ph,
    );
    scratch.half_b[..pix].fill(0);
    v_lowpass(
        &mut scratch.half_b[..pix],
        block_w,
        &scratch.ref_buf2[..needed_v],
        src_v_w,
        block_w,
        block_h,
    );

    avg_pixels(
        dst,
        dst_stride,
        &scratch.half_a[..pix],
        block_w,
        &scratch.half_b[..pix],
        block_w,
        block_w,
        block_h,
    );
}

/// Group D helper: h_lowpass (at row `ref_y_pos + h_row_delta`) + hv_lowpass, averaged.
///
/// Used for (2,1) with `h_row_delta=0` and (2,3) with `h_row_delta=1`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn qpel_mixed_h_hv(
    dst: &mut [u8],
    dst_stride: usize,
    ref_y: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    block_w: usize,
    block_h: usize,
    pw: i32,
    ph: i32,
    h_row_delta: i32,
    scratch: &mut McScratch,
) {
    let pix = block_w * block_h;

    // h_lowpass extract → ref_buf, filter → half_a
    let src_h_w = block_w + 5;
    let src_h_h = block_h;
    let needed_h = src_h_w * src_h_h;
    scratch.ref_buf[..needed_h].fill(0);
    extract_ref_block(
        &mut scratch.ref_buf[..needed_h],
        ref_y,
        ref_stride,
        ref_x - 2,
        ref_y_pos + h_row_delta,
        src_h_w,
        src_h_h,
        pw,
        ph,
    );
    scratch.half_a[..pix].fill(0);
    h_lowpass(
        &mut scratch.half_a[..pix],
        block_w,
        &scratch.ref_buf[..needed_h],
        src_h_w,
        block_w,
        block_h,
    );

    // hv_lowpass extract → ref_buf2, hv_lowpass uses ref_buf as tmp → half_b
    let src_hv_w = block_w + 5;
    let src_hv_h = block_h + 5;
    let needed_hv = src_hv_w * src_hv_h;
    scratch.ref_buf2[..needed_hv].fill(0);
    extract_ref_block(
        &mut scratch.ref_buf2[..needed_hv],
        ref_y,
        ref_stride,
        ref_x - 2,
        ref_y_pos - 2,
        src_hv_w,
        src_hv_h,
        pw,
        ph,
    );
    scratch.half_b[..pix].fill(0);
    // hv_lowpass reads from ref_buf2, uses ref_buf as tmp (ref_buf is free after h_lowpass consumed it)
    hv_lowpass(
        &mut scratch.half_b[..pix],
        block_w,
        &scratch.ref_buf2[..needed_hv],
        src_hv_w,
        block_w,
        block_h,
        &mut scratch.ref_buf,
    );

    avg_pixels(
        dst,
        dst_stride,
        &scratch.half_a[..pix],
        block_w,
        &scratch.half_b[..pix],
        block_w,
        block_w,
        block_h,
    );
}

/// Group D helper: v_lowpass (at col `ref_x + v_col_delta`) + hv_lowpass, averaged.
///
/// Used for (1,2) with `v_col_delta=0` and (3,2) with `v_col_delta=1`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn qpel_mixed_v_hv(
    dst: &mut [u8],
    dst_stride: usize,
    ref_y: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    block_w: usize,
    block_h: usize,
    pw: i32,
    ph: i32,
    v_col_delta: i32,
    scratch: &mut McScratch,
) {
    let pix = block_w * block_h;

    // v_lowpass extract → ref_buf, filter → half_a
    let src_v_w = block_w;
    let src_v_h = block_h + 5;
    let needed_v = src_v_w * src_v_h;
    scratch.ref_buf[..needed_v].fill(0);
    extract_ref_block(
        &mut scratch.ref_buf[..needed_v],
        ref_y,
        ref_stride,
        ref_x + v_col_delta,
        ref_y_pos - 2,
        src_v_w,
        src_v_h,
        pw,
        ph,
    );
    scratch.half_a[..pix].fill(0);
    v_lowpass(
        &mut scratch.half_a[..pix],
        block_w,
        &scratch.ref_buf[..needed_v],
        src_v_w,
        block_w,
        block_h,
    );

    // hv_lowpass extract → ref_buf2, hv_lowpass uses ref_buf as tmp → half_b
    let src_hv_w = block_w + 5;
    let src_hv_h = block_h + 5;
    let needed_hv = src_hv_w * src_hv_h;
    scratch.ref_buf2[..needed_hv].fill(0);
    extract_ref_block(
        &mut scratch.ref_buf2[..needed_hv],
        ref_y,
        ref_stride,
        ref_x - 2,
        ref_y_pos - 2,
        src_hv_w,
        src_hv_h,
        pw,
        ph,
    );
    scratch.half_b[..pix].fill(0);
    // hv_lowpass reads from ref_buf2, uses ref_buf as tmp (ref_buf is free after v_lowpass consumed it)
    hv_lowpass(
        &mut scratch.half_b[..pix],
        block_w,
        &scratch.ref_buf2[..needed_hv],
        src_hv_w,
        block_w,
        block_h,
        &mut scratch.ref_buf,
    );

    avg_pixels(
        dst,
        dst_stride,
        &scratch.half_a[..pix],
        block_w,
        &scratch.half_b[..pix],
        block_w,
        block_w,
        block_h,
    );
}

// ---------------------------------------------------------------------------
// Luma motion compensation (quarter-pel precision)
// ---------------------------------------------------------------------------

/// Perform luma motion compensation (quarter-pel precision).
///
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
    // Try NEON assembly for interior 16×16 and 8×8 blocks.
    #[cfg(has_asm)]
    if crate::asm_dispatch::mc_luma_asm(
        dst,
        dst_stride,
        ref_y,
        ref_stride,
        ref_x,
        ref_y_pos,
        dx,
        dy,
        block_w,
        block_h,
        pic_width as i32,
        pic_height as i32,
    ) {
        return;
    }

    let pw = pic_width as i32;
    let ph = pic_height as i32;

    scratch.ensure_capacity(block_w, block_h);

    match (dx, dy) {
        // ------------------------------------------------------------------
        // (0,0): full-pel copy
        // ------------------------------------------------------------------
        (0, 0) => {
            let src_w = block_w;
            let src_h = block_h;
            let needed = src_w * src_h;
            scratch.ref_buf[..needed].fill(0);
            extract_ref_block(
                &mut scratch.ref_buf[..needed],
                ref_y,
                ref_stride,
                ref_x,
                ref_y_pos,
                src_w,
                src_h,
                pw,
                ph,
            );
            for j in 0..block_h {
                for i in 0..block_w {
                    dst[j * dst_stride + i] = scratch.ref_buf[j * src_w + i] as u8;
                }
            }
        }

        // ------------------------------------------------------------------
        // (2,0): horizontal half-pel (the "b" position)
        // ------------------------------------------------------------------
        (2, 0) => {
            let src_w = block_w + 5;
            let src_h = block_h;
            let needed = src_w * src_h;
            scratch.ref_buf[..needed].fill(0);
            extract_ref_block(
                &mut scratch.ref_buf[..needed],
                ref_y,
                ref_stride,
                ref_x - 2,
                ref_y_pos,
                src_w,
                src_h,
                pw,
                ph,
            );
            h_lowpass(
                dst,
                dst_stride,
                &scratch.ref_buf[..needed],
                src_w,
                block_w,
                block_h,
            );
        }

        // ------------------------------------------------------------------
        // (0,2): vertical half-pel (the "h" position)
        // ------------------------------------------------------------------
        (0, 2) => {
            let src_w = block_w;
            let src_h = block_h + 5;
            let needed = src_w * src_h;
            scratch.ref_buf[..needed].fill(0);
            extract_ref_block(
                &mut scratch.ref_buf[..needed],
                ref_y,
                ref_stride,
                ref_x,
                ref_y_pos - 2,
                src_w,
                src_h,
                pw,
                ph,
            );
            v_lowpass(
                dst,
                dst_stride,
                &scratch.ref_buf[..needed],
                src_w,
                block_w,
                block_h,
            );
        }

        // ------------------------------------------------------------------
        // (2,2): 2D half-pel (the "j" position)
        // ------------------------------------------------------------------
        (2, 2) => {
            let src_w = block_w + 5;
            let src_h = block_h + 5;
            let needed = src_w * src_h;
            scratch.ref_buf[..needed].fill(0);
            extract_ref_block(
                &mut scratch.ref_buf[..needed],
                ref_y,
                ref_stride,
                ref_x - 2,
                ref_y_pos - 2,
                src_w,
                src_h,
                pw,
                ph,
            );
            hv_lowpass(
                dst,
                dst_stride,
                &scratch.ref_buf[..needed],
                src_w,
                block_w,
                block_h,
                &mut scratch.ref_buf2,
            );
        }

        // ------------------------------------------------------------------
        // (1,0): quarter-pel = avg(G, b) — avg of full-pel and horizontal half
        // ------------------------------------------------------------------
        (1, 0) => qpel_h_avg(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 2,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (3,0): quarter-pel = avg(G+1, b) — avg of full-pel+1 and horizontal half
        // ------------------------------------------------------------------
        (3, 0) => qpel_h_avg(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 3,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (0,1): quarter-pel = avg(G, h) — avg of full-pel and vertical half
        // ------------------------------------------------------------------
        (0, 1) => qpel_v_avg(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 2,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (0,3): quarter-pel = avg(G_below, h) — avg of full-pel+1row and vertical half
        // ------------------------------------------------------------------
        (0, 3) => qpel_v_avg(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 3,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (1,1): avg(halfH, halfV) — diagonal quarter-pel "e"
        // ------------------------------------------------------------------
        (1, 1) => qpel_diagonal(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 0, 0,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (3,1): avg(halfH, halfV_right) — "g" position
        // ------------------------------------------------------------------
        (3, 1) => qpel_diagonal(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 0, 1,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (1,3): avg(halfH_below, halfV) — "m" position
        // ------------------------------------------------------------------
        (1, 3) => qpel_diagonal(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 1, 0,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (3,3): avg(halfH_below, halfV_right) — "o" position
        // ------------------------------------------------------------------
        (3, 3) => qpel_diagonal(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 1, 1,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (2,1): avg(halfH, halfHV) — "f" position
        // ------------------------------------------------------------------
        (2, 1) => qpel_mixed_h_hv(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 0,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (2,3): avg(halfH_below, halfHV) — "n" position
        // ------------------------------------------------------------------
        (2, 3) => qpel_mixed_h_hv(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 1,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (1,2): avg(halfV, halfHV) — "i" position
        // ------------------------------------------------------------------
        (1, 2) => qpel_mixed_v_hv(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 0,
            scratch,
        ),

        // ------------------------------------------------------------------
        // (3,2): avg(halfV_right, halfHV) — "k" position
        // ------------------------------------------------------------------
        (3, 2) => qpel_mixed_v_hv(
            dst, dst_stride, ref_y, ref_stride, ref_x, ref_y_pos, block_w, block_h, pw, ph, 1,
            scratch,
        ),

        _ => unreachable!("dx and dy must be in 0..4, got ({}, {})", dx, dy),
    }
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
    for j in 0..block_h {
        for i in 0..block_w {
            let a = dst[j * dst_stride + i] as u16;
            let b = pred2[j * pred2_stride + i] as u16;
            dst[j * dst_stride + i] = ((a + b + 1) >> 1) as u8;
        }
    }
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

    // Try NEON assembly for interior blocks with matching strides.
    #[cfg(has_asm)]
    if crate::asm_dispatch::mc_chroma_asm(
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
    ) {
        return;
    }

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

    #[cfg(has_asm)]
    if crate::asm_dispatch::mc_chroma_avg_asm(
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
    ) {
        return;
    }

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
                let s1 = get_ref_pixel(ref_uv, ref_stride, sx + step_x, sy + step_y, pw, ph) as i32;
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
    for row in 0..h {
        let d_off = row * dst_stride;
        let s_off = row * src_stride;
        for col in 0..w {
            let a = dst[d_off + col] as u16;
            let b = src[s_off + col] as u16;
            dst[d_off + col] = ((a + b + 1) >> 1) as u8;
        }
    }
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
        // Direct test of filter6.
        assert_eq!(filter6(1, 1, 1, 1, 1, 1), 32);
        assert_eq!(filter6(0, 0, 0, 0, 0, 0), 0);
        assert_eq!(filter6(0, 0, 1, 0, 0, 0), 20);
        assert_eq!(filter6(0, 0, 0, 1, 0, 0), 20);
        assert_eq!(filter6(0, 1, 0, 0, 0, 0), -5);
        assert_eq!(filter6(1, 0, 0, 0, 0, 0), 1);
    }
}
