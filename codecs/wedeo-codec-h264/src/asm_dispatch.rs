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
    if dst_stride != ref_stride {
        return false;
    }
    if !is_interior(ref_x, ref_y_pos, dx, dy, block_w, block_h, pic_w, pic_h) {
        return false;
    }
    mc_luma_dispatch(
        dst,
        ref_y,
        ref_stride,
        ref_x,
        ref_y_pos,
        dx,
        dy,
        block_w,
        block_h,
        &asm_ffi::QPEL_PUT_16,
        &asm_ffi::QPEL_PUT_8,
        fullpel_put,
    )
}

/// Fullpel (mc00) put: copy src to dst.
#[inline]
fn fullpel_put(dst: &mut [u8], ref_y: &[u8], src_off: usize, stride: usize, w: usize, h: usize) {
    for j in 0..h {
        let d = j * stride;
        let s = src_off + j * stride;
        dst[d..d + w].copy_from_slice(&ref_y[s..s + w]);
    }
}

/// Fullpel (mc00) avg: average src with existing dst.
#[inline]
fn fullpel_avg(dst: &mut [u8], ref_y: &[u8], src_off: usize, stride: usize, w: usize, h: usize) {
    for j in 0..h {
        for i in 0..w {
            let d = j * stride + i;
            let s = src_off + j * stride + i;
            dst[d] = ((dst[d] as u16 + ref_y[s] as u16 + 1) >> 1) as u8;
        }
    }
}

/// Dispatch a single square MC block (8×8 or 16×16) to NEON or fullpel fallback.
#[inline]
#[allow(clippy::too_many_arguments)]
fn mc_dispatch_single(
    dst: &mut [u8],
    ref_y: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    idx: usize,
    table: &[Option<asm_ffi::QpelFn>; 16],
    stride: isize,
    block_w: usize,
    block_h: usize,
    fullpel: fn(&mut [u8], &[u8], usize, usize, usize, usize),
) -> bool {
    let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
    match table[idx] {
        Some(func) => {
            // SAFETY: Interior check passed, block dims match function.
            unsafe {
                func(dst.as_mut_ptr(), ref_y[src_off..].as_ptr(), stride);
            }
            true
        }
        None => {
            fullpel(dst, ref_y, src_off, ref_stride, block_w, block_h);
            true
        }
    }
}

/// Dispatch two 8×8 MC calls for a non-square block (16×8 or 8×16).
#[inline]
#[allow(clippy::too_many_arguments)]
fn mc_dispatch_half(
    dst: &mut [u8],
    ref_y: &[u8],
    src_off: usize,
    ref_stride: usize,
    idx: usize,
    table: &[Option<asm_ffi::QpelFn>; 16],
    stride: isize,
    col_off: usize,
    row_off: usize,
    fullpel: fn(&mut [u8], &[u8], usize, usize, usize, usize),
) -> bool {
    let off2 = col_off + row_off * ref_stride;
    match table[idx] {
        Some(func) => {
            // SAFETY: Interior check covered the full non-square window.
            unsafe {
                func(dst.as_mut_ptr(), ref_y[src_off..].as_ptr(), stride);
                func(
                    dst[off2..].as_mut_ptr(),
                    ref_y[src_off + off2..].as_ptr(),
                    stride,
                );
            }
            true
        }
        None => {
            fullpel(dst, ref_y, src_off, ref_stride, 8, 8);
            fullpel(&mut dst[off2..], ref_y, src_off + off2, ref_stride, 8, 8);
            true
        }
    }
}

/// Common dispatch logic for mc_luma_asm / mc_avg_asm.
#[inline]
#[allow(clippy::too_many_arguments)]
fn mc_luma_dispatch(
    dst: &mut [u8],
    ref_y: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    dx: u8,
    dy: u8,
    block_w: usize,
    block_h: usize,
    put_16: &[Option<asm_ffi::QpelFn>; 16],
    put_8: &[Option<asm_ffi::QpelFn>; 16],
    fullpel: fn(&mut [u8], &[u8], usize, usize, usize, usize),
) -> bool {
    let stride = ref_stride as isize;
    let idx = dy as usize * 4 + dx as usize;
    debug_assert!(idx < 16);

    match (block_w, block_h) {
        (16, 16) => mc_dispatch_single(
            dst, ref_y, ref_stride, ref_x, ref_y_pos, idx, put_16, stride, 16, 16, fullpel,
        ),
        (8, 8) => mc_dispatch_single(
            dst, ref_y, ref_stride, ref_x, ref_y_pos, idx, put_8, stride, 8, 8, fullpel,
        ),
        (16, 8) => {
            let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
            mc_dispatch_half(
                dst, ref_y, src_off, ref_stride, idx, put_8, stride, 8, 0, fullpel,
            )
        }
        (8, 16) => {
            let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
            mc_dispatch_half(
                dst, ref_y, src_off, ref_stride, idx, put_8, stride, 0, 8, fullpel,
            )
        }
        _ => false,
    }
}

/// Try to perform luma MC avg (bi-prediction) via NEON assembly.
///
/// Same constraints as `mc_luma_asm`. The `avg` variant reads the existing
/// dst pixels and averages them with the MC result.
#[inline]
#[allow(clippy::too_many_arguments)]
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
    mc_luma_dispatch(
        dst,
        ref_y,
        ref_stride,
        ref_x,
        ref_y_pos,
        dx,
        dy,
        block_w,
        block_h,
        &asm_ffi::QPEL_AVG_16,
        &asm_ffi::QPEL_AVG_8,
        fullpel_avg,
    )
}

// ---------------------------------------------------------------------------
// Chroma MC dispatch
// ---------------------------------------------------------------------------

/// Chroma MC function type: `fn(dst, src, stride, h, x, y)`.
type ChromaMcFn = unsafe extern "C" fn(*mut u8, *const u8, isize, i32, i32, i32);

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

    if !is_interior_chroma(ref_x, ref_y_pos, dx, dy, block_w, block_h, pic_w, pic_h) {
        return false;
    }

    mc_chroma_dispatch(
        dst, ref_uv, ref_stride, ref_x, ref_y_pos, dx, dy, block_w, block_h, false,
    )
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
    if !is_interior_chroma(ref_x, ref_y_pos, dx, dy, block_w, block_h, pic_w, pic_h) {
        return false;
    }

    mc_chroma_dispatch(
        dst, ref_uv, ref_stride, ref_x, ref_y_pos, dx, dy, block_w, block_h, true,
    )
}

/// Bounds check for chroma MC: bilinear filter reads one extra pixel in each
/// sub-pel direction.
#[inline]
#[allow(clippy::too_many_arguments)]
fn is_interior_chroma(
    ref_x: i32,
    ref_y: i32,
    dx: u8,
    dy: u8,
    block_w: usize,
    block_h: usize,
    pic_w: i32,
    pic_h: i32,
) -> bool {
    let dx_extra = if dx > 0 { 1 } else { 0 };
    let dy_extra = if dy > 0 { 1 } else { 0 };
    ref_x >= 0
        && ref_y >= 0
        && ref_x + block_w as i32 + dx_extra <= pic_w
        && ref_y + block_h as i32 + dy_extra <= pic_h
}

/// Shared dispatch for chroma put/avg.
#[inline]
#[allow(clippy::too_many_arguments)]
fn mc_chroma_dispatch(
    dst: &mut [u8],
    ref_uv: &[u8],
    ref_stride: usize,
    ref_x: i32,
    ref_y_pos: i32,
    dx: u8,
    dy: u8,
    block_w: usize,
    block_h: usize,
    avg: bool,
) -> bool {
    let func: ChromaMcFn = match (block_w, avg) {
        (8, false) => asm_ffi::ff_put_h264_chroma_mc8_neon,
        (4, false) => asm_ffi::ff_put_h264_chroma_mc4_neon,
        (2, false) => asm_ffi::ff_put_h264_chroma_mc2_neon,
        (8, true) => asm_ffi::ff_avg_h264_chroma_mc8_neon,
        (4, true) => asm_ffi::ff_avg_h264_chroma_mc4_neon,
        (2, true) => asm_ffi::ff_avg_h264_chroma_mc2_neon,
        _ => return false,
    };

    let src_off = ref_y_pos as usize * ref_stride + ref_x as usize;
    // SAFETY: Bounds checked by is_interior_chroma, block dims match function.
    unsafe {
        func(
            dst.as_mut_ptr(),
            ref_uv[src_off..].as_ptr(),
            ref_stride as isize,
            block_h as i32,
            dx as i32,
            dy as i32,
        );
    }
    true
}

// ---------------------------------------------------------------------------
// Deblock loop filter dispatch
// ---------------------------------------------------------------------------
//
// FFmpeg NEON naming is counter-intuitive:
//   v_loop_filter = horizontal edge (reads vertically across the edge)
//   h_loop_filter = vertical edge (reads horizontally across the edge)
//
// Mapped from wedeo's `is_vertical` flag:
//   is_vertical=true  → h_loop_filter (vertical boundary, reads horizontally)
//   is_vertical=false → v_loop_filter (horizontal boundary, reads vertically)

/// tc0 table in NEON format: indexed by [index_a][bS] for bS 0..3.
/// bS=0 → -1 (NEON skips pairs where tc0 < 0).
/// bS=1..3 → same values as TC0_TABLE[index_a][bS-1].
/// Matches FFmpeg's tc0_table (h264_loopfilter.c:69).
#[rustfmt::skip]
const TC0_NEON: [[i8; 4]; 52] = [
    [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0],
    [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0],
    [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0],
    [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0], [-1, 0, 0, 0],
    [-1, 0, 0, 0], [-1, 0, 0, 1], [-1, 0, 0, 1], [-1, 0, 0, 1],
    [-1, 0, 0, 1], [-1, 0, 1, 1], [-1, 0, 1, 1], [-1, 1, 1, 1],
    [-1, 1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, 1], [-1, 1, 1, 2],
    [-1, 1, 1, 2], [-1, 1, 1, 2], [-1, 1, 1, 2], [-1, 1, 2, 3],
    [-1, 1, 2, 3], [-1, 2, 2, 3], [-1, 2, 2, 4], [-1, 2, 3, 4],
    [-1, 2, 3, 4], [-1, 3, 3, 5], [-1, 3, 4, 6], [-1, 3, 4, 6],
    [-1, 4, 5, 7], [-1, 4, 5, 8], [-1, 4, 6, 9], [-1, 5, 7,10],
    [-1, 6, 8,11], [-1, 6, 8,13], [-1, 7,10,14], [-1, 8,11,16],
    [-1, 9,12,18], [-1,10,13,20], [-1,11,15,23], [-1,13,17,25],
];

/// Try to filter a luma edge via NEON assembly.
///
/// Returns `true` if NEON handled the edge, `false` if scalar fallback is needed.
/// Mixed bS (some 4, some <4) cannot be handled by NEON — returns false.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn deblock_luma_edge_asm(
    plane: &mut [u8],
    base: usize,
    stride: usize,
    is_vertical: bool,
    bs: [u8; 4],
    index_a: usize,
    alpha: i32,
    beta: i32,
) -> bool {
    let intra_count = bs.iter().filter(|&&b| b == 4).count();
    let all_intra = intra_count == 4;
    let any_intra = intra_count > 0;

    if all_intra {
        // All bS=4: use intra (strong) NEON filter
        let pix = &mut plane[base..];
        let s = stride as isize;
        // SAFETY: The caller verified alpha/beta thresholds and base offset.
        // The NEON intra filter processes 16 pixel pairs.
        unsafe {
            if is_vertical {
                asm_ffi::ff_h264_h_loop_filter_luma_intra_neon(pix.as_mut_ptr(), s, alpha, beta);
            } else {
                asm_ffi::ff_h264_v_loop_filter_luma_intra_neon(pix.as_mut_ptr(), s, alpha, beta);
            }
        }
        return true;
    }

    if any_intra {
        // Mixed bS=4 and bS<4: NEON can't handle, fall back to scalar
        return false;
    }

    // All bS < 4: build tc0 array from table (luma: no +1)
    let tc0: [i8; 4] = [
        TC0_NEON[index_a][bs[0] as usize],
        TC0_NEON[index_a][bs[1] as usize],
        TC0_NEON[index_a][bs[2] as usize],
        TC0_NEON[index_a][bs[3] as usize],
    ];

    let pix = &mut plane[base..];
    let s = stride as isize;
    // SAFETY: tc0 is a valid 4-byte array. Same base/stride guarantees as above.
    unsafe {
        if is_vertical {
            asm_ffi::ff_h264_h_loop_filter_luma_neon(
                pix.as_mut_ptr(),
                s,
                alpha,
                beta,
                tc0.as_ptr(),
            );
        } else {
            asm_ffi::ff_h264_v_loop_filter_luma_neon(
                pix.as_mut_ptr(),
                s,
                alpha,
                beta,
                tc0.as_ptr(),
            );
        }
    }
    true
}

/// Try to filter a chroma edge via NEON assembly.
///
/// Returns `true` if NEON handled the edge, `false` if scalar fallback is needed.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn deblock_chroma_edge_asm(
    plane: &mut [u8],
    base: usize,
    stride: usize,
    is_vertical: bool,
    bs: [u8; 4],
    index_a: usize,
    alpha: i32,
    beta: i32,
) -> bool {
    let intra_count = bs.iter().filter(|&&b| b == 4).count();
    let all_intra = intra_count == 4;
    let any_intra = intra_count > 0;

    if all_intra {
        let pix = &mut plane[base..];
        let s = stride as isize;
        // SAFETY: Same guarantees as luma. Chroma intra processes 8 pixel pairs.
        unsafe {
            if is_vertical {
                asm_ffi::ff_h264_h_loop_filter_chroma_intra_neon(pix.as_mut_ptr(), s, alpha, beta);
            } else {
                asm_ffi::ff_h264_v_loop_filter_chroma_intra_neon(pix.as_mut_ptr(), s, alpha, beta);
            }
        }
        return true;
    }

    if any_intra {
        return false;
    }

    // Chroma: tc0 = table value + 1 (matches FFmpeg h264_loopfilter.c:134)
    let tc0: [i8; 4] = [
        TC0_NEON[index_a][bs[0] as usize] + 1,
        TC0_NEON[index_a][bs[1] as usize] + 1,
        TC0_NEON[index_a][bs[2] as usize] + 1,
        TC0_NEON[index_a][bs[3] as usize] + 1,
    ];

    let pix = &mut plane[base..];
    let s = stride as isize;
    // SAFETY: Same as luma. Chroma normal filter processes 8 pixel pairs.
    unsafe {
        if is_vertical {
            asm_ffi::ff_h264_h_loop_filter_chroma_neon(
                pix.as_mut_ptr(),
                s,
                alpha,
                beta,
                tc0.as_ptr(),
            );
        } else {
            asm_ffi::ff_h264_v_loop_filter_chroma_neon(
                pix.as_mut_ptr(),
                s,
                alpha,
                beta,
                tc0.as_ptr(),
            );
        }
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
        asm_ffi::ff_h264_idct_dc_add_neon(dst.as_mut_ptr(), dc as *mut i16, stride as i32);
    }
}

/// 8x8 DC-only IDCT via NEON. Only reads block[0], no transpose needed.
#[inline]
pub fn idct8x8_dc_add_asm(dst: &mut [u8], stride: usize, dc: &mut i16) {
    // SAFETY: Same as 4x4 dc_add — only touches [x1].
    unsafe {
        asm_ffi::ff_h264_idct8_dc_add_neon(dst.as_mut_ptr(), dc as *mut i16, stride as i32);
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
        asm_ffi::ff_h264_idct_add_neon(dst.as_mut_ptr(), coeffs.as_mut_ptr(), stride as i32);
    }
}

/// 8x8 full IDCT via NEON with in-place transpose.
#[inline]
pub fn idct8x8_add_asm(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 64]) {
    transpose_8x8(coeffs);
    // SAFETY: coeffs is a contiguous 128-byte block. Same guarantees as 4x4.
    unsafe {
        asm_ffi::ff_h264_idct8_add_neon(dst.as_mut_ptr(), coeffs.as_mut_ptr(), stride as i32);
    }
}
