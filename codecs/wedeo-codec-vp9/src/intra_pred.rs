// VP9 intra prediction modes.
//
// Translated from FFmpeg's libavcodec/vp9dsp_template.c (Ronald S. Bultje,
// Clément Bœsch). All 15 intra prediction modes are implemented for block
// sizes 4, 8, 16, and 32.
//
// Conventions (matching vp9dsp_template.c):
//  • `above[0..size]` — pixels directly above the block (first `size` pixels).
//  • `above[-1]` (i.e., `top_left`) — the top-left corner pixel; callers
//    pass this as `above_left` (a separate parameter or `above[size]` sentinel).
//    In practice the interface puts top-left at `above[-1]` in C; here we
//    accept it via a dedicated parameter.
//  • `left[0..size]` — pixels to the left of the block, stored bottom-to-top
//    in FFmpeg (left[0] = bottom-left). **This module preserves that ordering.**
//
// Public interface:
//  `intra_pred(dst, stride, above, left, top_left, mode, size, bit_depth)`
//
// LGPL-2.1-or-later — same licence as FFmpeg.

use crate::types::IntraMode;

/// Clamp a pixel value to [0, max_val].
#[inline(always)]
fn clip(v: i32, max_val: i32) -> u8 {
    v.clamp(0, max_val) as u8
}

/// Access `dst[x + y * stride]` as an index (helper macro substitute).
#[inline(always)]
fn dst_idx(x: usize, y: usize, stride: usize) -> usize {
    x + y * stride
}

// ---------------------------------------------------------------------------
// Perform intra prediction
// ---------------------------------------------------------------------------

/// Perform VP9 intra prediction for one block.
///
/// # Arguments
/// * `dst` — output buffer (prediction written here; must be `stride * size` bytes).
/// * `stride` — row stride of `dst`.
/// * `above` — row of `size` pixels immediately above the block.
/// * `left` — column of `size` pixels to the left of the block, stored in
///   bottom-to-top order (left[0] is the bottom-left pixel).
/// * `top_left` — the single pixel at position (row-1, col-1) relative to the block.
/// * `mode` — intra prediction mode.
/// * `size` — block dimension in pixels (4, 8, 16, or 32).
/// * `bit_depth` — pixel bit depth (8, 10, or 12).
// The VP9 intra prediction interface requires these 8 arguments to match the
// vp9dsp_template.c calling convention. A wrapper struct would add indirection
// without benefit.
#[allow(clippy::too_many_arguments)]
pub fn intra_pred(
    dst: &mut [u8],
    stride: usize,
    above: &[u8], // above[0..size]
    left: &[u8],  // left[0..size], bottom-to-top
    top_left: u8, // pixel at (row-1, col-1)
    mode: IntraMode,
    size: usize,
    bit_depth: u8,
) {
    let max_val = (1_i32 << bit_depth) - 1;
    match mode {
        IntraMode::VertPred => vert_pred(dst, stride, above, size),
        IntraMode::HorPred => hor_pred(dst, stride, left, size),
        IntraMode::DcPred => dc_pred(dst, stride, above, left, size),
        IntraMode::DiagDownLeftPred => diag_downleft(dst, stride, above, size),
        IntraMode::DiagDownRightPred => diag_downright(dst, stride, above, left, top_left, size),
        IntraMode::VertRightPred => vert_right(dst, stride, above, left, top_left, size),
        IntraMode::HorDownPred => hor_down(dst, stride, above, left, top_left, size),
        IntraMode::VertLeftPred => vert_left(dst, stride, above, size),
        IntraMode::HorUpPred => hor_up(dst, stride, left, size),
        IntraMode::TmVp8Pred => tm_pred(dst, stride, above, left, top_left, size, max_val),
        IntraMode::LeftDcPred => dc_left_pred(dst, stride, left, size),
        IntraMode::TopDcPred => dc_top_pred(dst, stride, above, size),
        IntraMode::Dc128Pred => dc_128_pred(dst, stride, size, bit_depth),
        IntraMode::Dc127Pred => dc_127_pred(dst, stride, size, bit_depth),
        IntraMode::Dc129Pred => dc_129_pred(dst, stride, size, bit_depth),
    }
}

// ---------------------------------------------------------------------------
// V_PRED — copy above row to every row
// ---------------------------------------------------------------------------
fn vert_pred(dst: &mut [u8], stride: usize, above: &[u8], size: usize) {
    for y in 0..size {
        dst[y * stride..y * stride + size].copy_from_slice(&above[..size]);
    }
}

// ---------------------------------------------------------------------------
// H_PRED — copy left pixel to each row
// Stored bottom-to-top: left[0]=bottom, left[size-1]=top.
// Row 0 of dst is top → use left[size-1-y].
// ---------------------------------------------------------------------------
fn hor_pred(dst: &mut [u8], stride: usize, left: &[u8], size: usize) {
    for y in 0..size {
        let v = left[size - 1 - y];
        for x in 0..size {
            dst[dst_idx(x, y, stride)] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// DC_PRED — average of above + left
// ---------------------------------------------------------------------------
fn dc_pred(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8], size: usize) {
    let sum: u32 = above[..size].iter().map(|&v| v as u32).sum::<u32>()
        + left[..size].iter().map(|&v| v as u32).sum::<u32>();
    let dc = ((sum + size as u32) >> (size.trailing_zeros() + 1)) as u8;
    fill_block(dst, stride, dc, size);
}

// ---------------------------------------------------------------------------
// LEFT_DC_PRED — DC from left column only
// ---------------------------------------------------------------------------
fn dc_left_pred(dst: &mut [u8], stride: usize, left: &[u8], size: usize) {
    let sum: u32 = left[..size].iter().map(|&v| v as u32).sum();
    let half = (size / 2) as u32;
    let dc = ((sum + half) >> size.trailing_zeros()) as u8;
    fill_block(dst, stride, dc, size);
}

// ---------------------------------------------------------------------------
// TOP_DC_PRED — DC from above row only
// ---------------------------------------------------------------------------
fn dc_top_pred(dst: &mut [u8], stride: usize, above: &[u8], size: usize) {
    let sum: u32 = above[..size].iter().map(|&v| v as u32).sum();
    let half = (size / 2) as u32;
    let dc = ((sum + half) >> size.trailing_zeros()) as u8;
    fill_block(dst, stride, dc, size);
}

// ---------------------------------------------------------------------------
// DC_128_PRED, DC_127_PRED, DC_129_PRED
// ---------------------------------------------------------------------------
fn dc_128_pred(dst: &mut [u8], stride: usize, size: usize, bit_depth: u8) {
    let v = 128_u32 << (bit_depth - 8);
    fill_block(dst, stride, v as u8, size);
}

fn dc_127_pred(dst: &mut [u8], stride: usize, size: usize, bit_depth: u8) {
    let v = (128_u32 << (bit_depth - 8)) - 1;
    fill_block(dst, stride, v as u8, size);
}

fn dc_129_pred(dst: &mut [u8], stride: usize, size: usize, bit_depth: u8) {
    let v = (128_u32 << (bit_depth - 8)) + 1;
    fill_block(dst, stride, v as u8, size);
}

// ---------------------------------------------------------------------------
// TM_PRED (TrueMotion)
// ---------------------------------------------------------------------------
fn tm_pred(
    dst: &mut [u8],
    stride: usize,
    above: &[u8],
    left: &[u8],
    top_left: u8,
    size: usize,
    max_val: i32,
) {
    let tl = top_left as i32;
    for y in 0..size {
        let l_m_tl = left[size - 1 - y] as i32 - tl;
        for x in 0..size {
            let v = above[x] as i32 + l_m_tl;
            dst[dst_idx(x, y, stride)] = clip(v, max_val);
        }
    }
}

// ---------------------------------------------------------------------------
// D45_PRED (DIAG_DOWN_LEFT) — projects down and to the left at 45 degrees
// The C reference reads 2*size pixels from `top` (above[0..2*size]).
// We treat above[size-1] as the repeated-last value when beyond the block.
// ---------------------------------------------------------------------------
fn diag_downleft(dst: &mut [u8], stride: usize, above: &[u8], size: usize) {
    // Helper: read from `above`, clamping at the last pixel.
    let a = |k: usize| -> i32 { above[k.min(2 * size - 1)] as i32 };

    if size == 4 {
        // Verbatim 4×4 case from vp9dsp_template.c.
        let (a0, a1, a2, a3, a4, a5, a6, a7) = (a(0), a(1), a(2), a(3), a(4), a(5), a(6), a(7));
        let dst_set = |x: usize, y: usize, v: i32, d: &mut [u8]| {
            d[dst_idx(x, y, stride)] = v as u8;
        };
        let f = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;
        dst_set(0, 0, f(a0, a1, a2), dst);
        let v01 = f(a1, a2, a3);
        dst[dst_idx(1, 0, stride)] = v01 as u8;
        dst[dst_idx(0, 1, stride)] = v01 as u8;
        let v02 = f(a2, a3, a4);
        dst[dst_idx(2, 0, stride)] = v02 as u8;
        dst[dst_idx(1, 1, stride)] = v02 as u8;
        dst[dst_idx(0, 2, stride)] = v02 as u8;
        let v03 = f(a3, a4, a5);
        dst[dst_idx(3, 0, stride)] = v03 as u8;
        dst[dst_idx(2, 1, stride)] = v03 as u8;
        dst[dst_idx(1, 2, stride)] = v03 as u8;
        dst[dst_idx(0, 3, stride)] = v03 as u8;
        let v11 = f(a4, a5, a6);
        dst[dst_idx(3, 1, stride)] = v11 as u8;
        dst[dst_idx(2, 2, stride)] = v11 as u8;
        dst[dst_idx(1, 3, stride)] = v11 as u8;
        let v22 = f(a5, a6, a7);
        dst[dst_idx(3, 2, stride)] = v22 as u8;
        dst[dst_idx(2, 3, stride)] = v22 as u8;
        dst[dst_idx(3, 3, stride)] = a7 as u8;
    } else {
        // Generic case matching `def_diag_downleft(size)`.
        // We need the index k to read a(k), a(k+1), a(k+2).
        #[allow(clippy::needless_range_loop)]
        let v: Vec<i32> = (0..size - 1)
            .map(|k| {
                if k < size - 2 {
                    (a(k) + a(k + 1) * 2 + a(k + 2) + 2) >> 2
                } else {
                    (a(size - 2) + a(size - 1) * 3 + 2) >> 2
                }
            })
            .collect();

        for j in 0..size {
            for x in 0..(size - 1 - j) {
                dst[dst_idx(x, j, stride)] = v[j + x] as u8;
            }
            // Fill the remaining positions with the last above pixel.
            let last = above[size - 1];
            for x in (size - 1 - j)..size {
                dst[dst_idx(x, j, stride)] = last;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D135_PRED (DIAG_DOWN_RIGHT)
// ---------------------------------------------------------------------------
fn diag_downright(
    dst: &mut [u8],
    stride: usize,
    above: &[u8],
    left: &[u8],
    top_left: u8,
    size: usize,
) {
    let tl = top_left as i32;

    if size == 4 {
        let a = |k: usize| above[k] as i32;
        let l = |k: usize| left[k] as i32; // left[0]=bottom, left[3]=top-left-adjacent
        let (a0, a1, a2, a3) = (a(0), a(1), a(2), a(3));
        let (l0, l1, l2, l3) = (l(3), l(2), l(1), l(0));

        let f = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;
        dst[dst_idx(0, 3, stride)] = f(l1, l2, l3) as u8;
        let v = f(l0, l1, l2);
        dst[dst_idx(0, 2, stride)] = v as u8;
        dst[dst_idx(1, 3, stride)] = v as u8;
        let v = f(tl, l0, l1);
        dst[dst_idx(0, 1, stride)] = v as u8;
        dst[dst_idx(1, 2, stride)] = v as u8;
        dst[dst_idx(2, 3, stride)] = v as u8;
        let v = f(l0, tl, a0);
        dst[dst_idx(0, 0, stride)] = v as u8;
        dst[dst_idx(1, 1, stride)] = v as u8;
        dst[dst_idx(2, 2, stride)] = v as u8;
        dst[dst_idx(3, 3, stride)] = v as u8;
        let v = f(tl, a0, a1);
        dst[dst_idx(1, 0, stride)] = v as u8;
        dst[dst_idx(2, 1, stride)] = v as u8;
        dst[dst_idx(3, 2, stride)] = v as u8;
        let v = f(a0, a1, a2);
        dst[dst_idx(2, 0, stride)] = v as u8;
        dst[dst_idx(3, 1, stride)] = v as u8;
        dst[dst_idx(3, 0, stride)] = f(a1, a2, a3) as u8;
    } else {
        // Generic case matching `def_diag_downright(size)`.
        let mut v = vec![0i32; 2 * size - 1];
        let f3 = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;

        for k in 0..(size - 2) {
            v[k] = f3(left[k] as i32, left[k + 1] as i32, left[k + 2] as i32);
            v[size + 1 + k] = f3(above[k] as i32, above[k + 1] as i32, above[k + 2] as i32);
        }
        // left indices in C are reversed (left[0]=bottom=farthest from corner).
        v[size - 2] = f3(left[size - 2] as i32, left[size - 1] as i32, tl);
        v[size - 1] = f3(left[size - 1] as i32, tl, above[0] as i32);
        v[size] = f3(tl, above[0] as i32, above[1] as i32);

        for j in 0..size {
            for x in 0..size {
                dst[dst_idx(x, j, stride)] = v[size - 1 - j + x] as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D117_PRED (VERT_RIGHT)
// ---------------------------------------------------------------------------
fn vert_right(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8], top_left: u8, size: usize) {
    let tl = top_left as i32;

    if size == 4 {
        let a = |k: usize| above[k] as i32;
        let l = |k: usize| left[k] as i32;
        // In FFmpeg, l0=left[3] (row just below TL), l2=left[1].
        let (l0, l1, l2) = (l(3), l(2), l(1));
        let (a0, a1, a2, a3) = (a(0), a(1), a(2), a(3));

        let f2 = |p: i32, q: i32| (p + q + 1) >> 1;
        let f3 = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;

        dst[dst_idx(0, 3, stride)] = f3(l0, l1, l2) as u8;
        dst[dst_idx(0, 2, stride)] = f3(tl, l0, l1) as u8;
        let v = f2(tl, a0);
        dst[dst_idx(0, 0, stride)] = v as u8;
        dst[dst_idx(1, 2, stride)] = v as u8;
        let v = f3(l0, tl, a0);
        dst[dst_idx(0, 1, stride)] = v as u8;
        dst[dst_idx(1, 3, stride)] = v as u8;
        let v = f2(a0, a1);
        dst[dst_idx(1, 0, stride)] = v as u8;
        dst[dst_idx(2, 2, stride)] = v as u8;
        let v = f3(tl, a0, a1);
        dst[dst_idx(1, 1, stride)] = v as u8;
        dst[dst_idx(2, 3, stride)] = v as u8;
        let v = f2(a1, a2);
        dst[dst_idx(2, 0, stride)] = v as u8;
        dst[dst_idx(3, 2, stride)] = v as u8;
        let v = f3(a0, a1, a2);
        dst[dst_idx(2, 1, stride)] = v as u8;
        dst[dst_idx(3, 3, stride)] = v as u8;
        dst[dst_idx(3, 0, stride)] = f2(a2, a3) as u8;
        dst[dst_idx(3, 1, stride)] = f3(a1, a2, a3) as u8;
    } else {
        // Generic case matching `def_vert_right(size)`.
        let h = size / 2;
        let mut ve = vec![0i32; size + h - 1];
        let mut vo = vec![0i32; size + h - 1];
        let f2 = |p: i32, q: i32| (p + q + 1) >> 1;
        let f3 = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;

        for k in 0..(h - 2) {
            // In C: left[0]=bottom, left[size-1]=top-adjacent.
            // Rows counted from top: vo[k] uses left at positions 2k+3, 2k+2, 2k+1.
            vo[k] = f3(
                left[2 * k + 3] as i32,
                left[2 * k + 2] as i32,
                left[2 * k + 1] as i32,
            );
            ve[k] = f3(
                left[2 * k + 4] as i32,
                left[2 * k + 3] as i32,
                left[2 * k + 2] as i32,
            );
        }
        vo[h - 2] = f3(
            left[size - 1] as i32,
            left[size - 2] as i32,
            left[size - 3] as i32,
        );
        ve[h - 2] = f3(tl, left[size - 1] as i32, left[size - 2] as i32);
        ve[h - 1] = f2(tl, above[0] as i32);
        vo[h - 1] = f3(left[size - 1] as i32, tl, above[0] as i32);
        for k in 0..(size - 1) {
            ve[h + k] = f2(above[k] as i32, above[k + 1] as i32);
            vo[h + k] = f3(
                if k > 0 { above[k - 1] as i32 } else { tl },
                above[k] as i32,
                above[k + 1] as i32,
            );
        }
        for j in 0..h {
            let base_e = h - 1 - j;
            let base_o = h - 1 - j;
            for x in 0..size {
                dst[dst_idx(x, j * 2, stride)] = ve[base_e + x] as u8;
                dst[dst_idx(x, j * 2 + 1, stride)] = vo[base_o + x] as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D153_PRED (HOR_DOWN)
// ---------------------------------------------------------------------------
fn hor_down(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8], top_left: u8, size: usize) {
    let tl = top_left as i32;

    if size == 4 {
        let a = |k: usize| above[k] as i32;
        // In FFmpeg: l0=left[3], l1=left[2], l2=left[1], l3=left[0].
        let (l0, l1, l2, l3) = (
            left[3] as i32,
            left[2] as i32,
            left[1] as i32,
            left[0] as i32,
        );
        let (a0, a1, a2) = (a(0), a(1), a(2));
        let f2 = |p: i32, q: i32| (p + q + 1) >> 1;
        let f3 = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;

        dst[dst_idx(2, 0, stride)] = f3(tl, a0, a1) as u8;
        dst[dst_idx(3, 0, stride)] = f3(a0, a1, a2) as u8;
        let v = f2(tl, l0);
        dst[dst_idx(0, 0, stride)] = v as u8;
        dst[dst_idx(2, 1, stride)] = v as u8;
        let v = f3(a0, tl, l0);
        dst[dst_idx(1, 0, stride)] = v as u8;
        dst[dst_idx(3, 1, stride)] = v as u8;
        let v = f2(l0, l1);
        dst[dst_idx(0, 1, stride)] = v as u8;
        dst[dst_idx(2, 2, stride)] = v as u8;
        let v = f3(tl, l0, l1);
        dst[dst_idx(1, 1, stride)] = v as u8;
        dst[dst_idx(3, 2, stride)] = v as u8;
        let v = f2(l1, l2);
        dst[dst_idx(0, 2, stride)] = v as u8;
        dst[dst_idx(2, 3, stride)] = v as u8;
        let v = f3(l0, l1, l2);
        dst[dst_idx(1, 2, stride)] = v as u8;
        dst[dst_idx(3, 3, stride)] = v as u8;
        dst[dst_idx(0, 3, stride)] = f2(l2, l3) as u8;
        dst[dst_idx(1, 3, stride)] = f3(l1, l2, l3) as u8;
    } else {
        // Generic case matching `def_hor_down(size)`.
        let mut v = vec![0i32; 3 * size - 2];
        let f2 = |p: i32, q: i32| (p + q + 1) >> 1;
        let f3 = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;

        for k in 0..(size - 2) {
            v[k * 2] = f2(left[k + 1] as i32, left[k] as i32);
            v[k * 2 + 1] = f3(left[k + 2] as i32, left[k + 1] as i32, left[k] as i32);
            v[size * 2 + k] = f3(
                if k > 0 { above[k - 1] as i32 } else { tl },
                above[k] as i32,
                above[k + 1] as i32,
            );
        }
        v[size * 2 - 2] = f2(tl, left[size - 1] as i32);
        v[size * 2 - 4] = f2(left[size - 1] as i32, left[size - 2] as i32);
        v[size * 2 - 1] = f3(above[0] as i32, tl, left[size - 1] as i32);
        v[size * 2 - 3] = f3(tl, left[size - 1] as i32, left[size - 2] as i32);

        for j in 0..size {
            for x in 0..size {
                let idx = size * 2 - 2 - j * 2 + x;
                dst[dst_idx(x, j, stride)] = v[idx] as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D207_PRED (VERT_LEFT)
// ---------------------------------------------------------------------------
fn vert_left(dst: &mut [u8], stride: usize, above: &[u8], size: usize) {
    let a = |k: usize| above[k.min(2 * size - 1)] as i32;

    if size == 4 {
        let f2 = |p: i32, q: i32| (p + q + 1) >> 1;
        let f3 = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;
        let (a0, a1, a2, a3, a4, a5, a6) = (a(0), a(1), a(2), a(3), a(4), a(5), a(6));

        dst[dst_idx(0, 0, stride)] = f2(a0, a1) as u8;
        dst[dst_idx(0, 1, stride)] = f3(a0, a1, a2) as u8;
        let v = f2(a1, a2);
        dst[dst_idx(1, 0, stride)] = v as u8;
        dst[dst_idx(0, 2, stride)] = v as u8;
        let v = f3(a1, a2, a3);
        dst[dst_idx(1, 1, stride)] = v as u8;
        dst[dst_idx(0, 3, stride)] = v as u8;
        let v = f2(a2, a3);
        dst[dst_idx(2, 0, stride)] = v as u8;
        dst[dst_idx(1, 2, stride)] = v as u8;
        let v = f3(a2, a3, a4);
        dst[dst_idx(2, 1, stride)] = v as u8;
        dst[dst_idx(1, 3, stride)] = v as u8;
        let v = f2(a3, a4);
        dst[dst_idx(3, 0, stride)] = v as u8;
        dst[dst_idx(2, 2, stride)] = v as u8;
        let v = f3(a3, a4, a5);
        dst[dst_idx(3, 1, stride)] = v as u8;
        dst[dst_idx(2, 3, stride)] = v as u8;
        dst[dst_idx(3, 2, stride)] = f2(a4, a5) as u8;
        dst[dst_idx(3, 3, stride)] = f3(a4, a5, a6) as u8;
    } else {
        // Generic case matching `def_vert_left(size)`.
        let mut ve = vec![0i32; size - 1];
        let mut vo = vec![0i32; size - 1];
        let f2 = |p: i32, q: i32| (p + q + 1) >> 1;
        let f3 = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;

        for k in 0..(size - 2) {
            ve[k] = f2(a(k), a(k + 1));
            vo[k] = f3(a(k), a(k + 1), a(k + 2));
        }
        ve[size - 2] = f2(a(size - 2), a(size - 1));
        vo[size - 2] = f3(a(size - 2), a(size - 1), a(size - 1)); // clamp last

        for j in 0..(size / 2) {
            for x in 0..(size - j - 1) {
                dst[dst_idx(x, j * 2, stride)] = ve[j + x] as u8;
                dst[dst_idx(x, j * 2 + 1, stride)] = vo[j + x] as u8;
            }
            let last = above[size - 1];
            for x in (size - j - 1)..size {
                dst[dst_idx(x, j * 2, stride)] = last;
                dst[dst_idx(x, j * 2 + 1, stride)] = last;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D63_PRED (HOR_UP)
// left[0]=bottom-left, left[size-1]=top-left-adjacent.
// The C function uses left[0..size] with left[0]=top of left column.
// ---------------------------------------------------------------------------
fn hor_up(dst: &mut [u8], stride: usize, left: &[u8], size: usize) {
    if size == 4 {
        let l = |k: usize| left[k] as i32;
        let (l0, l1, l2, l3) = (l(0), l(1), l(2), l(3));
        let f2 = |p: i32, q: i32| (p + q + 1) >> 1;
        let f3 = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;

        dst[dst_idx(0, 0, stride)] = f2(l0, l1) as u8;
        dst[dst_idx(1, 0, stride)] = f3(l0, l1, l2) as u8;
        let v = f2(l1, l2);
        dst[dst_idx(0, 1, stride)] = v as u8;
        dst[dst_idx(2, 0, stride)] = v as u8;
        let v = f3(l1, l2, l3);
        dst[dst_idx(1, 1, stride)] = v as u8;
        dst[dst_idx(3, 0, stride)] = v as u8;
        let v = f2(l2, l3);
        dst[dst_idx(0, 2, stride)] = v as u8;
        dst[dst_idx(2, 1, stride)] = v as u8;
        let v = (l2 + l3 * 3 + 2) >> 2; // f3-like with 3*l3
        dst[dst_idx(1, 2, stride)] = v as u8;
        dst[dst_idx(3, 1, stride)] = v as u8;
        // Bottom-right quadrant filled with l3.
        for &(x, y) in &[(0, 3), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)] {
            dst[dst_idx(x, y, stride)] = l3 as u8;
        }
    } else {
        // Generic case matching `def_hor_up(size)`.
        let mut v = vec![0i32; 2 * size - 2];
        let f2 = |p: i32, q: i32| (p + q + 1) >> 1;
        let f3 = |p: i32, q: i32, r: i32| (p + q * 2 + r + 2) >> 2;

        for k in 0..(size - 2) {
            v[k * 2] = f2(left[k] as i32, left[k + 1] as i32);
            v[k * 2 + 1] = f3(left[k] as i32, left[k + 1] as i32, left[k + 2] as i32);
        }
        v[2 * size - 4] = f2(left[size - 2] as i32, left[size - 1] as i32);
        v[2 * size - 3] = (left[size - 2] as i32 + left[size - 1] as i32 * 3 + 2) >> 2;

        let last = left[size - 1];
        for j in 0..(size / 2) {
            for x in 0..size {
                dst[dst_idx(x, j, stride)] = v[j * 2 + x] as u8;
            }
        }
        for j in (size / 2)..size {
            let avail = 2 * size - 2 - j * 2;
            for x in 0..avail {
                dst[dst_idx(x, j, stride)] = v[j * 2 + x] as u8;
            }
            for x in avail..size {
                dst[dst_idx(x, j, stride)] = last;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: fill entire block with constant value
// ---------------------------------------------------------------------------
fn fill_block(dst: &mut [u8], stride: usize, val: u8, size: usize) {
    for y in 0..size {
        for x in 0..size {
            dst[dst_idx(x, y, stride)] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_above(size: usize, val: u8) -> Vec<u8> {
        vec![val; size]
    }
    fn make_left(size: usize, val: u8) -> Vec<u8> {
        vec![val; size]
    }

    /// V_PRED: every pixel should equal the above value.
    #[test]
    fn vert_pred_flat() {
        let size = 4;
        let above = make_above(size, 200);
        let left = make_left(size, 100);
        let mut dst = vec![0u8; size * size];
        intra_pred(
            &mut dst,
            size,
            &above,
            &left,
            150,
            IntraMode::VertPred,
            size,
            8,
        );
        assert!(dst.iter().all(|&v| v == 200));
    }

    /// H_PRED: each row should be filled with the corresponding left pixel.
    #[test]
    fn hor_pred_flat() {
        let size = 4;
        let above = make_above(size, 100);
        let left: Vec<u8> = (0..size as u8).collect(); // left[0]=bottom
        let mut dst = vec![0u8; size * size];
        intra_pred(
            &mut dst,
            size,
            &above,
            &left,
            150,
            IntraMode::HorPred,
            size,
            8,
        );
        // Row y maps to left[size-1-y].
        for y in 0..size {
            let expected = left[size - 1 - y];
            for x in 0..size {
                assert_eq!(dst[x + y * size], expected, "y={y} x={x}");
            }
        }
    }

    /// DC_PRED: all pixels should equal the average of above+left.
    #[test]
    fn dc_pred_flat() {
        let size = 4;
        let above = make_above(size, 100);
        let left = make_left(size, 200);
        let mut dst = vec![0u8; size * size];
        intra_pred(
            &mut dst,
            size,
            &above,
            &left,
            150,
            IntraMode::DcPred,
            size,
            8,
        );
        // Average of 4×100 + 4×200 = 1200/8 = 150.
        assert!(dst.iter().all(|&v| v == 150), "DC pred failed: {:?}", dst);
    }

    /// TM_PRED: each pixel = above[x] + left[size-1-y] - top_left.
    #[test]
    fn tm_pred_basic() {
        let size = 4;
        let above: Vec<u8> = (1..=4).collect();
        let left: Vec<u8> = (5..=8).collect(); // left[0]=bottom=5, left[3]=top=8
        let top_left = 4u8;
        let mut dst = vec![0u8; size * size];
        intra_pred(
            &mut dst,
            size,
            &above,
            &left,
            top_left,
            IntraMode::TmVp8Pred,
            size,
            8,
        );
        // Row 0: left[3]=8, l_m_tl = 8-4 = 4. Pixel (0,0) = above[0]+4 = 5.
        assert_eq!(dst[0], 5);
    }

    /// DC_128_PRED: all pixels equal 128 for 8-bit.
    #[test]
    fn dc128_pred() {
        let size = 4;
        let mut dst = vec![0u8; size * size];
        intra_pred(
            &mut dst,
            size,
            &[0u8; 4],
            &[0u8; 4],
            0,
            IntraMode::Dc128Pred,
            size,
            8,
        );
        assert!(dst.iter().all(|&v| v == 128));
    }

    /// DIAG_DOWN_LEFT 4×4: spot-check corner.
    /// The VP9 4×4 diag-downleft reads 8 above pixels (2×block_size).
    #[test]
    fn diag_downleft_4x4_corner() {
        // Provide 8 above pixels as required by the 4×4 formula (a0..a7).
        let above: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let left = [0u8; 4];
        let mut dst = vec![0u8; 16];
        intra_pred(
            &mut dst,
            4,
            &above, // pass all 8 above pixels
            &left,
            0,
            IntraMode::DiagDownLeftPred,
            4,
            8,
        );
        // DST(3,3) = a7 = 80
        assert_eq!(dst[3 + 3 * 4], 80);
        // DST(0,0) = (a0+a1*2+a2+2)>>2 = (10+40+30+2)>>2 = 82>>2 = 20
        assert_eq!(dst[0], 20);
    }
}
