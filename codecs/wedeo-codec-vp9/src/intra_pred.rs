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
    match size {
        4 => predict::<4>(dst, stride, above, left, top_left, mode, bit_depth),
        8 => predict::<8>(dst, stride, above, left, top_left, mode, bit_depth),
        16 => predict::<16>(dst, stride, above, left, top_left, mode, bit_depth),
        32 => predict::<32>(dst, stride, above, left, top_left, mode, bit_depth),
        _ => unreachable!("invalid VP9 transform size {size}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn predict<const N: usize>(
    dst: &mut [u8],
    stride: usize,
    above: &[u8], // above[0..size]
    left: &[u8],  // left[0..size], bottom-to-top
    top_left: u8, // pixel at (row-1, col-1)
    mode: IntraMode,
    bit_depth: u8,
) {
    let max_val = (1_i32 << bit_depth) - 1;
    match mode {
        IntraMode::VertPred => vert_pred(dst, stride, above, N),
        IntraMode::HorPred => hor_pred(dst, stride, left, N),
        IntraMode::DcPred => dc_pred(dst, stride, above, left, N),
        IntraMode::DiagDownLeftPred => diag_downleft(dst, stride, above, N),
        IntraMode::DiagDownRightPred => diag_downright(dst, stride, above, left, top_left, N),
        IntraMode::VertRightPred => vert_right(dst, stride, above, left, top_left, N),
        IntraMode::HorDownPred => hor_down(dst, stride, above, left, top_left, N),
        IntraMode::VertLeftPred => vert_left(dst, stride, above, N),
        IntraMode::HorUpPred => hor_up(dst, stride, left, N),
        IntraMode::TmVp8Pred => tm_pred(dst, stride, above, left, top_left, N, max_val),
        IntraMode::LeftDcPred => dc_left_pred(dst, stride, left, N),
        IntraMode::TopDcPred => dc_top_pred(dst, stride, above, N),
        IntraMode::Dc128Pred => dc_128_pred(dst, stride, N, bit_depth),
        IntraMode::Dc127Pred => dc_127_pred(dst, stride, N, bit_depth),
        IntraMode::Dc129Pred => dc_129_pred(dst, stride, N, bit_depth),
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

// Each directional predictor filters a short strip of boundary pixels once.
// A block row is then a contiguous window into that strip. VP9 blocks
// are at most 32 pixels wide.
#[inline]
fn avg2(a: u8, b: u8) -> u8 {
    (a as u16 + b as u16).div_ceil(2) as u8
}

#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2 * b as u16 + c as u16 + 2) >> 2) as u8
}

#[inline]
fn put_row(dst: &mut [u8], stride: usize, y: usize, strip: &[u8], offset: usize, size: usize) {
    dst[y * stride..y * stride + size].copy_from_slice(&strip[offset..offset + size]);
}

#[inline]
fn diag_downleft(dst: &mut [u8], stride: usize, above: &[u8], size: usize) {
    // Only the 4x4 predictor has real top-right neighbors.
    let len = if size == 4 { 8 } else { size };
    let last = above[len - 1];
    let mut strip = [last; 64];
    for i in 0..len - 2 {
        strip[i] = avg3(above[i], above[i + 1], above[i + 2]);
    }
    if size != 4 {
        strip[len - 2] = avg3(above[len - 2], last, last);
    }
    for y in 0..size {
        put_row(dst, stride, y, &strip, y, size);
    }
}

#[inline]
fn diag_downright(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8], tl: u8, size: usize) {
    let mut edge = [0; 65];
    edge[..size].copy_from_slice(&left[..size]);
    edge[size] = tl;
    edge[size + 1..2 * size + 1].copy_from_slice(&above[..size]);
    let mut strip = [0; 63];
    for i in 0..2 * size - 1 {
        strip[i] = avg3(edge[i], edge[i + 1], edge[i + 2]);
    }
    for y in 0..size {
        put_row(dst, stride, y, &strip, size - 1 - y, size);
    }
}

#[inline]
fn vert_right(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8], top_left: u8, size: usize) {
    let tl = top_left;

    let h = size / 2;
    let mut ve = [0u8; 47];
    let mut vo = [0u8; 47];

    for k in 0..(h - 2) {
        // In C: left[0]=bottom, left[size-1]=top-adjacent.
        // Rows counted from top: vo[k] uses left at positions 2k+3, 2k+2, 2k+1.
        vo[k] = avg3(left[2 * k + 3], left[2 * k + 2], left[2 * k + 1]);
        ve[k] = avg3(left[2 * k + 4], left[2 * k + 3], left[2 * k + 2]);
    }
    vo[h - 2] = avg3(left[size - 1], left[size - 2], left[size - 3]);
    ve[h - 2] = avg3(tl, left[size - 1], left[size - 2]);
    ve[h - 1] = avg2(tl, above[0]);
    vo[h - 1] = avg3(left[size - 1], tl, above[0]);
    for k in 0..(size - 1) {
        ve[h + k] = avg2(above[k], above[k + 1]);
        vo[h + k] = avg3(
            if k > 0 { above[k - 1] } else { tl },
            above[k],
            above[k + 1],
        );
    }
    for j in 0..h {
        let offset = h - 1 - j;
        put_row(dst, stride, j * 2, &ve, offset, size);
        put_row(dst, stride, j * 2 + 1, &vo, offset, size);
    }
}

// ---------------------------------------------------------------------------
// D153_PRED (HOR_DOWN)
// ---------------------------------------------------------------------------
#[inline]
fn hor_down(dst: &mut [u8], stride: usize, above: &[u8], left: &[u8], top_left: u8, size: usize) {
    let tl = top_left;

    let mut v = [0u8; 94];

    for k in 0..(size - 2) {
        v[k * 2] = avg2(left[k + 1], left[k]);
        v[k * 2 + 1] = avg3(left[k + 2], left[k + 1], left[k]);
        v[size * 2 + k] = avg3(
            if k > 0 { above[k - 1] } else { tl },
            above[k],
            above[k + 1],
        );
    }
    v[size * 2 - 2] = avg2(tl, left[size - 1]);
    v[size * 2 - 4] = avg2(left[size - 1], left[size - 2]);
    v[size * 2 - 1] = avg3(above[0], tl, left[size - 1]);
    v[size * 2 - 3] = avg3(tl, left[size - 1], left[size - 2]);

    for j in 0..size {
        put_row(dst, stride, j, &v, size * 2 - 2 - j * 2, size);
    }
}

// ---------------------------------------------------------------------------
// D207_PRED (VERT_LEFT)
// ---------------------------------------------------------------------------
#[inline]
fn vert_left(dst: &mut [u8], stride: usize, above: &[u8], size: usize) {
    let len = if size == 4 { 7 } else { size };
    let last = above[len - 1];
    let mut even = [last; 48];
    let mut odd = [last; 48];
    for i in 0..len - 1 {
        even[i] = avg2(above[i], above[i + 1]);
        odd[i] = avg3(above[i], above[i + 1], above[(i + 2).min(len - 1)]);
    }
    for j in 0..size / 2 {
        put_row(dst, stride, j * 2, &even, j, size);
        put_row(dst, stride, j * 2 + 1, &odd, j, size);
    }
}

#[inline]
fn hor_up(dst: &mut [u8], stride: usize, left: &[u8], size: usize) {
    let mut strip = [left[size - 1]; 94];
    for i in 0..size - 1 {
        strip[2 * i] = avg2(left[i], left[i + 1]);
        strip[2 * i + 1] = avg3(left[i], left[i + 1], left[(i + 2).min(size - 1)]);
    }
    for y in 0..size {
        put_row(dst, stride, y, &strip, 2 * y, size);
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
