// Assembly dispatch layer: bridges wedeo's MC/IDCT/deblock calling conventions
// to FFmpeg's NEON assembly functions.
//
// For MC, the key design is the "interior block" check: FFmpeg's qpel assembly
// reads src[-2*stride] through src[(h+3)*stride] — it needs 2 pixels of padding
// on each side for the 6-tap filter. Wedeo's frames have no padding, so we only
// dispatch to assembly for blocks whose filter window stays within picture bounds.
// Edge blocks (~10% of a 1080p frame) fall back to the scalar Rust path.

use crate::asm_ffi;

// ---------------------------------------------------------------------------
// Interior block check for luma qpel MC
// ---------------------------------------------------------------------------

/// Returns true if the 6-tap filter window for this MC call stays entirely
/// within the reference picture bounds (no edge clamping needed).
///
/// The filter reads from `src[ref_y - 2]` to `src[ref_y + block_h + 2]` vertically,
/// and `src[ref_x - 2]` to `src[ref_x + block_w + 2]` horizontally.
/// For sub-pel positions, the filter window extends further.
#[inline]
#[allow(clippy::too_many_arguments)]
fn is_interior(
    ref_x: i32,
    ref_y: i32,
    dx: u8,
    dy: u8,
    block_w: usize,
    block_h: usize,
    pic_w: i32,
    pic_h: i32,
) -> bool {
    // The 6-tap filter needs 2 pixels before and 3 pixels after the reference
    // position (taps at -2, -1, 0, 1, 2, 3 relative to the center).
    // For sub-pel positions, the center is at ref_x/ref_y, and the filter
    // reads 2 pixels before and 3 after.
    let extra_before = 2;
    let extra_after = 3; // filter taps go to position +3

    let x_min = ref_x - extra_before;
    let y_min = ref_y - extra_before;
    let x_max = ref_x + block_w as i32 + extra_after;
    let y_max = ref_y + block_h as i32 + extra_after;

    // For fractional positions, the assembly may read one extra row/column.
    // dx != 0 means horizontal sub-pel → already covered by extra_after
    // dy != 0 means vertical sub-pel → already covered by extra_after
    // But double-check: the hv path reads (block_h + 5) rows and (block_w + 5) cols.
    // That's ref_y - 2 to ref_y + block_h + 3 — exactly our bounds.
    let _ = (dx, dy); // bounds are the same for all sub-pel positions

    x_min >= 0 && y_min >= 0 && x_max <= pic_w && y_max <= pic_h
}

// ---------------------------------------------------------------------------
// Luma MC dispatch
// ---------------------------------------------------------------------------

/// Try to perform luma MC via NEON assembly.
///
/// Returns `true` if the assembly path was used, `false` if the caller should
/// fall back to the scalar Rust implementation.
///
/// Conditions for assembly dispatch:
/// 1. Block size is 16×16, 8×8, 16×8, or 8×16 (16×8 and 8×16 decompose into 2× 8×8)
/// 2. Block is "interior" — the 6-tap filter window doesn't touch picture edges
/// 3. dst_stride == ref_stride (FFmpeg takes a single stride for both)
#[inline]
#[allow(clippy::too_many_arguments)] // Mirrors mc_luma signature
pub fn mc_luma_asm(
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
    pic_w: i32,
    pic_h: i32,
) -> bool {
    // FFmpeg qpel takes a single stride for both src and dst.
    if dst_stride != ref_stride {
        return false;
    }

    // Check that the filter window stays within picture bounds.
    if !is_interior(ref_x, ref_y_pos, dx, dy, block_w, block_h, pic_w, pic_h) {
        return false;
    }

    // Determine which table to use based on block dimensions.
    // Square 16×16 and 8×8 are direct. Non-square 16×8 and 8×16 decompose
    // into two 8×8 calls.
    let stride = ref_stride as isize;
    let idx = dy as usize * 4 + dx as usize;
    debug_assert!(idx < 16);

    match (block_w, block_h) {
        (16, 16) => {
            mc_luma_single(dst, ref_y, ref_stride, ref_x, ref_y_pos, idx, &asm_ffi::QPEL_PUT_16, stride, 16, 16)
        }
        (8, 8) => {
            mc_luma_single(dst, ref_y, ref_stride, ref_x, ref_y_pos, idx, &asm_ffi::QPEL_PUT_8, stride, 8, 8)
        }
        (16, 8) => {
            // Split width: left 8×8 + right 8×8
            let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
            mc_luma_half(dst, ref_y, src_off, ref_stride, idx, &asm_ffi::QPEL_PUT_8, stride, 8, 0)
        }
        (8, 16) => {
            // Split height: top 8×8 + bottom 8×8
            let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
            mc_luma_half(dst, ref_y, src_off, ref_stride, idx, &asm_ffi::QPEL_PUT_8, stride, 0, 8)
        }
        _ => false,
    }
}

/// Execute a single MC call for a square block.
#[inline]
#[allow(clippy::too_many_arguments)]
fn mc_luma_single(
    dst: &mut [u8], ref_y: &[u8], ref_stride: usize,
    ref_x: i32, ref_y_pos: i32, idx: usize,
    table: &[Option<asm_ffi::QpelFn>; 16], stride: isize,
    block_w: usize, block_h: usize,
) -> bool {
    let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
    match table[idx] {
        Some(func) => {
            // SAFETY: Interior check passed, block dims match function.
            unsafe { func(dst.as_mut_ptr(), ref_y[src_off..].as_ptr(), stride); }
            true
        }
        None => {
            // mc00 (fullpel copy)
            for j in 0..block_h {
                let d = j * ref_stride;
                let s = src_off + j * ref_stride;
                dst[d..d + block_w].copy_from_slice(&ref_y[s..s + block_w]);
            }
            true
        }
    }
}

/// Execute two 8×8 MC calls for a non-square block (16×8 or 8×16).
/// `col_off` is the horizontal offset for the second call (8 for 16×8, 0 for 8×16).
/// `row_off` is the vertical offset in rows (0 for 16×8, 8 for 8×16).
#[inline]
#[allow(clippy::too_many_arguments)]
fn mc_luma_half(
    dst: &mut [u8], ref_y: &[u8], src_off: usize, ref_stride: usize,
    idx: usize, table: &[Option<asm_ffi::QpelFn>; 16], stride: isize,
    col_off: usize, row_off: usize,
) -> bool {
    let stride_off = row_off * ref_stride;
    match table[idx] {
        Some(func) => {
            // SAFETY: Interior check covered the full non-square window.
            unsafe {
                func(dst.as_mut_ptr(), ref_y[src_off..].as_ptr(), stride);
                func(
                    dst[col_off + stride_off..].as_mut_ptr(),
                    ref_y[src_off + col_off + stride_off..].as_ptr(),
                    stride,
                );
            }
            true
        }
        None => {
            // mc00 (fullpel copy) for both halves
            for j in 0..8 {
                let d = j * ref_stride;
                let s = src_off + j * ref_stride;
                dst[d..d + 8].copy_from_slice(&ref_y[s..s + 8]);
            }
            for j in 0..8 {
                let d = col_off + stride_off + j * ref_stride;
                let s = src_off + col_off + stride_off + j * ref_stride;
                dst[d..d + 8].copy_from_slice(&ref_y[s..s + 8]);
            }
            true
        }
    }
}

/// Try to perform luma MC avg (bi-prediction) via NEON assembly.
///
/// Same constraints as `mc_luma_asm`. The `avg` variant reads the existing
/// dst pixels and averages them with the MC result.
#[inline]
#[allow(clippy::too_many_arguments)] // Mirrors mc_luma signature
pub fn mc_avg_asm(
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
    pic_w: i32,
    pic_h: i32,
) -> bool {
    if dst_stride != ref_stride {
        return false;
    }
    if !is_interior(ref_x, ref_y_pos, dx, dy, block_w, block_h, pic_w, pic_h) {
        return false;
    }

    let stride = ref_stride as isize;
    let idx = dy as usize * 4 + dx as usize;

    match (block_w, block_h) {
        (16, 16) => {
            mc_avg_single(dst, ref_y, ref_stride, ref_x, ref_y_pos, idx, &asm_ffi::QPEL_AVG_16, stride, 16, 16)
        }
        (8, 8) => {
            mc_avg_single(dst, ref_y, ref_stride, ref_x, ref_y_pos, idx, &asm_ffi::QPEL_AVG_8, stride, 8, 8)
        }
        (16, 8) => {
            let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
            mc_avg_half(dst, ref_y, src_off, ref_stride, idx, &asm_ffi::QPEL_AVG_8, stride, 8, 0)
        }
        (8, 16) => {
            let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
            mc_avg_half(dst, ref_y, src_off, ref_stride, idx, &asm_ffi::QPEL_AVG_8, stride, 0, 8)
        }
        _ => false,
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn mc_avg_single(
    dst: &mut [u8], ref_y: &[u8], ref_stride: usize,
    ref_x: i32, ref_y_pos: i32, idx: usize,
    table: &[Option<asm_ffi::QpelFn>; 16], stride: isize,
    block_w: usize, block_h: usize,
) -> bool {
    let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
    match table[idx] {
        Some(func) => {
            // SAFETY: Interior check passed, block dims match function.
            unsafe { func(dst.as_mut_ptr(), ref_y[src_off..].as_ptr(), stride); }
            true
        }
        None => {
            // mc00 avg: average with existing dst
            for j in 0..block_h {
                for i in 0..block_w {
                    let d = j * ref_stride + i;
                    let s = src_off + j * ref_stride + i;
                    let a = dst[d] as u16;
                    let b = ref_y[s] as u16;
                    dst[d] = ((a + b + 1) >> 1) as u8;
                }
            }
            true
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn mc_avg_half(
    dst: &mut [u8], ref_y: &[u8], src_off: usize, ref_stride: usize,
    idx: usize, table: &[Option<asm_ffi::QpelFn>; 16], stride: isize,
    col_off: usize, row_off: usize,
) -> bool {
    let stride_off = row_off * ref_stride;
    match table[idx] {
        Some(func) => {
            // SAFETY: Interior check covered the full non-square window.
            unsafe {
                func(dst.as_mut_ptr(), ref_y[src_off..].as_ptr(), stride);
                func(
                    dst[col_off + stride_off..].as_mut_ptr(),
                    ref_y[src_off + col_off + stride_off..].as_ptr(),
                    stride,
                );
            }
            true
        }
        None => {
            // mc00 avg for both halves
            for j in 0..8 {
                for i in 0..8 {
                    let d = j * ref_stride + i;
                    let s = src_off + j * ref_stride + i;
                    let a = dst[d] as u16;
                    let b = ref_y[s] as u16;
                    dst[d] = ((a + b + 1) >> 1) as u8;
                }
            }
            for j in 0..8 {
                for i in 0..8 {
                    let d = col_off + stride_off + j * ref_stride + i;
                    let s = src_off + col_off + stride_off + j * ref_stride + i;
                    let a = dst[d] as u16;
                    let b = ref_y[s] as u16;
                    dst[d] = ((a + b + 1) >> 1) as u8;
                }
            }
            true
        }
    }
}

// ---------------------------------------------------------------------------
// Chroma MC dispatch
// ---------------------------------------------------------------------------

/// Chroma MC function type: `fn(dst, src, stride, h, x, y)`.
type ChromaMcFn =
    unsafe extern "C" fn(*mut u8, *const u8, isize, i32, i32, i32);

/// Try to perform chroma MC via NEON assembly.
///
/// Returns `true` if assembly was used.
/// Requires: interior block, dst_stride == ref_stride, block_w ∈ {2, 4, 8}.
#[inline]
#[allow(clippy::too_many_arguments)] // Mirrors mc_chroma signature
pub fn mc_chroma_asm(
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
    pic_w: i32,
    pic_h: i32,
) -> bool {
    if dst_stride != ref_stride {
        return false;
    }

    // Chroma bilinear reads src[x..x+block_w+1] and src[y..y+block_h+1].
    // Check bounds: need ref_x >= 0, ref_y >= 0,
    // ref_x + block_w + 1 <= pic_w, ref_y + block_h + 1 <= pic_h.
    let dx_extra = if dx > 0 { 1 } else { 0 };
    let dy_extra = if dy > 0 { 1 } else { 0 };
    if ref_x < 0
        || ref_y_pos < 0
        || ref_x + block_w as i32 + dx_extra > pic_w
        || ref_y_pos + block_h as i32 + dy_extra > pic_h
    {
        return false;
    }

    let func: ChromaMcFn = match block_w {
        8 => asm_ffi::ff_put_h264_chroma_mc8_neon,
        4 => asm_ffi::ff_put_h264_chroma_mc4_neon,
        2 => asm_ffi::ff_put_h264_chroma_mc2_neon,
        _ => return false,
    };

    let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
    let stride = ref_stride as isize;

    // SAFETY: Bounds checked above, block dimensions match function expectations.
    unsafe {
        func(
            dst.as_mut_ptr(),
            ref_uv[src_off..].as_ptr(),
            stride,
            block_h as i32,
            dx as i32,
            dy as i32,
        );
    }
    true
}

/// Try to perform chroma MC avg (bi-prediction) via NEON assembly.
///
/// Same constraints as `mc_chroma_asm`. The `avg` variant reads the existing
/// dst pixels and averages them with the MC result.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma_avg_asm(
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
    pic_w: i32,
    pic_h: i32,
) -> bool {
    if dst_stride != ref_stride {
        return false;
    }

    let dx_extra = if dx > 0 { 1 } else { 0 };
    let dy_extra = if dy > 0 { 1 } else { 0 };
    if ref_x < 0
        || ref_y_pos < 0
        || ref_x + block_w as i32 + dx_extra > pic_w
        || ref_y_pos + block_h as i32 + dy_extra > pic_h
    {
        return false;
    }

    let func: ChromaMcFn = match block_w {
        8 => asm_ffi::ff_avg_h264_chroma_mc8_neon,
        4 => asm_ffi::ff_avg_h264_chroma_mc4_neon,
        2 => asm_ffi::ff_avg_h264_chroma_mc2_neon,
        _ => return false,
    };

    let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
    let stride = ref_stride as isize;

    // SAFETY: Bounds checked above, block dimensions match function expectations.
    unsafe {
        func(
            dst.as_mut_ptr(),
            ref_uv[src_off..].as_ptr(),
            stride,
            block_h as i32,
            dx as i32,
            dy as i32,
        );
    }
    true
}

// ---------------------------------------------------------------------------
// IDCT dispatch
// ---------------------------------------------------------------------------
//
// FFmpeg's NEON IDCT expects column-major (transposed) coefficients. Wedeo
// stores row-major, so we transpose in-place before calling NEON.
//
// Rounding: The Rust scalar path adds +32 to block[0] then uses plain >>6.
// The NEON path uses `srshr #6` = `(v + 32) >> 6` per element with NO +32
// to block[0]. These are equivalent because +32 enters via the even butterfly
// path (never hits >>1 truncation) and distributes uniformly to all outputs.
// Therefore: do NOT add +32 before calling NEON.
//
// DC-only functions only read block[0], so layout is irrelevant — direct call.

/// In-place transpose of a 4x4 coefficient matrix stored as [i16; 16].
/// Swaps off-diagonal elements: (row, col) <-> (col, row).
#[inline]
fn transpose_4x4(coeffs: &mut [i16; 16]) {
    for i in 0..4 {
        for j in (i + 1)..4 {
            coeffs.swap(i * 4 + j, j * 4 + i);
        }
    }
}

/// In-place transpose of an 8x8 coefficient matrix stored as [i16; 64].
#[inline]
fn transpose_8x8(coeffs: &mut [i16; 64]) {
    for i in 0..8 {
        for j in (i + 1)..8 {
            coeffs.swap(i * 8 + j, j * 8 + i);
        }
    }
}

/// 4x4 DC-only IDCT via NEON. Only reads block[0], no transpose needed.
#[inline]
pub fn idct4x4_dc_add_asm(dst: &mut [u8], stride: usize, dc: &mut i16) {
    // SAFETY: NEON dc_add only reads/writes [x1] (one i16). The `srshr #6`
    // computes (dc + 32) >> 6, matching the scalar path. The function zeros
    // block[0] via `strh w3=0, [x1]`.
    unsafe {
        asm_ffi::ff_h264_idct_dc_add_neon(
            dst.as_mut_ptr(),
            dc as *mut i16,
            stride as i32,
        );
    }
}

/// 8x8 DC-only IDCT via NEON. Only reads block[0], no transpose needed.
#[inline]
pub fn idct8x8_dc_add_asm(dst: &mut [u8], stride: usize, dc: &mut i16) {
    // SAFETY: Same as 4x4 dc_add — only touches [x1].
    unsafe {
        asm_ffi::ff_h264_idct8_dc_add_neon(
            dst.as_mut_ptr(),
            dc as *mut i16,
            stride as i32,
        );
    }
}

/// 4x4 full IDCT via NEON with in-place transpose.
///
/// Transposes coefficients from row-major to column-major (what the NEON
/// butterfly expects), then calls the assembly function. The NEON function
/// handles rounding via `srshr #6` and zeros the coefficient block.
#[inline]
pub fn idct4x4_add_asm(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 16]) {
    transpose_4x4(coeffs);
    // SAFETY: coeffs is a contiguous 32-byte block. The NEON function reads
    // all 16 coefficients, performs the butterfly + srshr + add-to-dst, and
    // zeros the block via st1 of zero vectors.
    unsafe {
        asm_ffi::ff_h264_idct_add_neon(
            dst.as_mut_ptr(),
            coeffs.as_mut_ptr(),
            stride as i32,
        );
    }
}

/// 8x8 full IDCT via NEON with in-place transpose.
#[inline]
pub fn idct8x8_add_asm(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 64]) {
    transpose_8x8(coeffs);
    // SAFETY: coeffs is a contiguous 128-byte block. Same guarantees as 4x4.
    unsafe {
        asm_ffi::ff_h264_idct8_add_neon(
            dst.as_mut_ptr(),
            coeffs.as_mut_ptr(),
            stride as i32,
        );
    }
}
