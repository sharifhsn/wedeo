// H.264/AVC in-loop deblocking filter.
//
// Reduces blocking artifacts at macroblock and 4x4 block boundaries.
// Runs after all macroblocks in a slice are decoded; the filtered output
// is used for inter prediction of future frames (in-loop).
//
// Reference: ITU-T H.264 spec section 8.7, FFmpeg libavcodec/h264_loopfilter.c
// and h264dsp_template.c.

use tracing::debug;

use crate::tables::CHROMA_QP_TABLE;

// ---------------------------------------------------------------------------
// Threshold tables from the H.264 spec (Table 8-16) and FFmpeg.
// ---------------------------------------------------------------------------

/// Alpha threshold table indexed by indexA (0..51).
const ALPHA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

/// Beta threshold table indexed by indexB (0..51).
const BETA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// tc0 table indexed by [indexA][bS-1] for bS 1..3.
/// bS=4 uses the strong (intra) filter, not tc0.
/// From H.264 spec Table 8-17 and FFmpeg tc0_table.
const TC0_TABLE: [[i32; 3]; 52] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 2, 3],
    [1, 2, 3],
    [2, 2, 3],
    [2, 2, 4],
    [2, 3, 4],
    [2, 3, 4],
    [3, 3, 5],
    [3, 4, 6],
    [3, 4, 6],
    [4, 5, 7],
    [4, 5, 8],
    [4, 6, 9],
    [5, 7, 10],
    [6, 8, 11],
    [6, 8, 13],
    [7, 10, 14],
    [8, 11, 16],
    [9, 12, 18],
    [10, 13, 20],
    [11, 15, 23],
    [13, 17, 25],
];

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Information about a decoded macroblock needed for deblocking.
#[derive(Debug, Clone)]
pub struct MbDeblockInfo {
    /// Macroblock type classification for deblocking.
    pub is_intra: bool,
    /// QP value for this macroblock.
    pub qp: u8,
    /// Number of reference lists used: 1 for P-slice MBs, 2 for B-slice MBs.
    /// Used by the two-permutation BS check for B-frames (FFmpeg check_mv).
    pub list_count: u8,
    /// Non-zero coefficient count per 4x4 block (16 luma + 8 chroma).
    /// Luma: indices 0..16 in raster scan of 4x4 sub-blocks.
    /// Chroma Cb: indices 16..20, Cr: indices 20..24.
    pub non_zero_count: [u8; 24],
    /// Reference picture identity per 4x4 block (list 0).
    /// B-slices: stores POC so cross-list (L0/L1) identity comparison works.
    /// P-slices: stores DPB index (cast to i32) — handles ref list duplicates
    /// from ref_pic_list_modification and POC collisions from MMCO-5.
    /// i32::MIN = unavailable.
    pub ref_poc: [i32; 16],
    /// Motion vectors per 4x4 block (list 0) [x, y].
    pub mv: [[i16; 2]; 16],
    /// Reference picture POC per 4x4 block (list 1, for B-slice MBs).
    /// i32::MIN = unavailable.
    pub ref_poc_l1: [i32; 16],
    /// Motion vectors per 4x4 block (list 1, for B-slice MBs) [x, y].
    pub mv_l1: [[i16; 2]; 16],
}

impl Default for MbDeblockInfo {
    fn default() -> Self {
        Self {
            is_intra: false,
            qp: 0,
            list_count: 1,
            non_zero_count: [0; 24],
            ref_poc: [i32::MIN; 16],
            mv: [[0; 2]; 16],
            ref_poc_l1: [i32::MIN; 16],
            mv_l1: [[0; 2]; 16],
        }
    }
}

/// Picture buffer with Y, U, V planes.
#[derive(Clone)]
pub struct PictureBuffer {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub y_stride: usize,
    pub uv_stride: usize,
    pub width: u32,
    pub height: u32,
    pub mb_width: u32,
    pub mb_height: u32,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Clamp `x` to the range `[lo, hi]`.
#[inline(always)]
fn clip3(lo: i32, hi: i32, x: i32) -> i32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Clamp a pixel value to [0, 255].
#[inline(always)]
fn clip_pixel(x: i32) -> u8 {
    clip3(0, 255, x) as u8
}

/// Map luma QP to chroma QP using the spec table, applying the PPS
/// chroma_qp_index_offset before lookup (H.264 spec eq. 7-30).
#[inline(always)]
fn chroma_qp(luma_qp: u8, chroma_qp_index_offset: i32) -> u8 {
    let idx = (luma_qp as i32 + chroma_qp_index_offset).clamp(0, 51) as usize;
    CHROMA_QP_TABLE[idx]
}

/// Compute the average QP between two blocks for edge filtering.
#[inline(always)]
fn avg_qp(qp_p: u8, qp_q: u8) -> u8 {
    ((qp_p as u16 + qp_q as u16 + 1) >> 1) as u8
}

/// Compute alpha and beta thresholds from QP and offsets.
/// Returns (alpha, beta), or (0, 0) if no filtering needed.
#[inline]
fn get_thresholds(qp: u8, alpha_c0_offset: i32, beta_offset: i32) -> (i32, i32) {
    let index_a = clip3(0, 51, qp as i32 + alpha_c0_offset) as usize;
    let index_b = clip3(0, 51, qp as i32 + beta_offset) as usize;
    (ALPHA_TABLE[index_a], BETA_TABLE[index_b])
}

/// Get tc0 value from the table for a given QP, offset, and bS (1-3).
#[inline]
fn get_tc0(qp: u8, alpha_c0_offset: i32, bs: u8) -> i32 {
    debug_assert!((1..=3).contains(&bs));
    let index_a = clip3(0, 51, qp as i32 + alpha_c0_offset) as usize;
    TC0_TABLE[index_a][(bs - 1) as usize]
}

// ---------------------------------------------------------------------------
// 4x4 block index helpers
// ---------------------------------------------------------------------------

/// Convert (block_x, block_y) within a macroblock to a linear 4x4 luma index (0..15).
/// block_x, block_y are in 4x4 block units (0..3).
#[inline(always)]
fn luma_block_idx(block_x: usize, block_y: usize) -> usize {
    block_y * 4 + block_x
}

// ---------------------------------------------------------------------------
// Boundary strength calculation
// ---------------------------------------------------------------------------

/// Check if two blocks have different motion, implementing FFmpeg's check_mv()
/// (h264_loopfilter.c:438-466) with the two-permutation check for B-frames.
///
/// For B-frames, blocks can use opposite reference lists with identical predictions.
/// The two-permutation check detects this: L0↔L0 + L1↔L1, then cross L0↔L1 + L1↔L0.
#[allow(clippy::too_many_arguments)] // mirrors FFmpeg check_mv's per-block L0+L1 parameters
#[inline]
fn check_mv(
    p_ref: i32,
    q_ref: i32,
    p_mv: [i16; 2],
    q_mv: [i16; 2],
    p_ref_l1: i32,
    q_ref_l1: i32,
    p_mv_l1: [i16; 2],
    q_mv_l1: [i16; 2],
    list_count: u8,
) -> bool {
    // Step 1: Check L0 ref and MV (same for P and B slices)
    let mut v = p_ref != q_ref
        || (p_mv[0] - q_mv[0]).unsigned_abs() >= 4
        || (p_mv[1] - q_mv[1]).unsigned_abs() >= 4;

    if list_count == 2 {
        // Step 2: If L0 passed (v==false), also check L1
        if !v {
            v = p_ref_l1 != q_ref_l1
                || (p_mv_l1[0] - q_mv_l1[0]).unsigned_abs() >= 4
                || (p_mv_l1[1] - q_mv_l1[1]).unsigned_abs() >= 4;
        }
        // Step 3: If either same-list check failed, try cross-list permutation.
        // Blocks might use opposite lists with the same prediction.
        if v {
            if p_ref != q_ref_l1 || p_ref_l1 != q_ref {
                return true;
            }
            return (p_mv[0] - q_mv_l1[0]).unsigned_abs() >= 4
                || (p_mv[1] - q_mv_l1[1]).unsigned_abs() >= 4
                || (p_mv_l1[0] - q_mv[0]).unsigned_abs() >= 4
                || (p_mv_l1[1] - q_mv[1]).unsigned_abs() >= 4;
        }
    }

    v
}

/// Compute boundary strength for an edge between block P and block Q.
///
/// `is_mb_edge`: true if this is a macroblock boundary edge (edge 0)
/// `p_intra`, `q_intra`: whether blocks are intra
/// `p_nnz`, `q_nnz`: non-zero coefficient counts
/// `p_ref`/`q_ref`: L0 reference indices; `p_ref_l1`/`q_ref_l1`: L1 reference indices
/// `p_mv`/`q_mv`: L0 motion vectors; `p_mv_l1`/`q_mv_l1`: L1 motion vectors
/// `list_count`: max of P/Q list counts (1 for P-only, 2 if either is from B-slice)
#[allow(clippy::too_many_arguments)] // matches the H.264 spec's per-edge decision tree
pub fn compute_bs(
    is_mb_edge: bool,
    p_intra: bool,
    q_intra: bool,
    p_nnz: u8,
    q_nnz: u8,
    p_ref: i32,
    q_ref: i32,
    p_mv: [i16; 2],
    q_mv: [i16; 2],
    p_ref_l1: i32,
    q_ref_l1: i32,
    p_mv_l1: [i16; 2],
    q_mv_l1: [i16; 2],
    list_count: u8,
) -> u8 {
    if p_intra || q_intra {
        if is_mb_edge { 4 } else { 3 }
    } else if p_nnz != 0 || q_nnz != 0 {
        2
    } else if check_mv(
        p_ref, q_ref, p_mv, q_mv, p_ref_l1, q_ref_l1, p_mv_l1, q_mv_l1, list_count,
    ) {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Normal filter (bS = 1, 2, or 3) — luma
// ---------------------------------------------------------------------------

/// Apply the normal deblocking filter to a single pixel pair on a luma edge.
///
/// `p0..p2` and `q0..q2` are the 3 pixels on each side of the edge.
/// Returns (p0', p1', q0', q1') -- only p0/q0 always change; p1/q1 may remain unchanged.
#[allow(clippy::too_many_arguments)] // mirrors FFmpeg h264_loop_filter_luma per-pixel logic
#[inline]
fn filter_normal_luma(
    p0: i32,
    p1: i32,
    p2: i32,
    q0: i32,
    q1: i32,
    q2: i32,
    alpha: i32,
    beta: i32,
    tc0: i32,
) -> Option<(u8, u8, u8, u8)> {
    // Threshold check
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return None;
    }

    let mut tc = tc0;
    let mut new_p1 = p1 as u8;
    let mut new_q1 = q1 as u8;

    // Optionally filter p1
    if (p2 - p0).abs() < beta {
        if tc0 != 0 {
            new_p1 = (p1 + clip3(-tc0, tc0, ((p2 + ((p0 + q0 + 1) >> 1)) >> 1) - p1)) as u8;
        }
        tc += 1;
    }

    // Optionally filter q1
    if (q2 - q0).abs() < beta {
        if tc0 != 0 {
            new_q1 = (q1 + clip3(-tc0, tc0, ((q2 + ((p0 + q0 + 1) >> 1)) >> 1) - q1)) as u8;
        }
        tc += 1;
    }

    // Filter p0, q0
    let delta = clip3(-tc, tc, ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3);
    let new_p0 = clip_pixel(p0 + delta);
    let new_q0 = clip_pixel(q0 - delta);

    Some((new_p0, new_p1, new_q0, new_q1))
}

// ---------------------------------------------------------------------------
// Normal filter (bS = 1, 2, or 3) — chroma
// ---------------------------------------------------------------------------

/// Apply the normal deblocking filter to a single pixel pair on a chroma edge.
///
/// Chroma normal filter only modifies p0 and q0; tc = tc0 + 1.
#[inline]
fn filter_normal_chroma(
    p0: i32,
    p1: i32,
    q0: i32,
    q1: i32,
    alpha: i32,
    beta: i32,
    tc: i32,
) -> Option<(u8, u8)> {
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return None;
    }

    let delta = clip3(-tc, tc, ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3);
    let new_p0 = clip_pixel(p0 + delta);
    let new_q0 = clip_pixel(q0 - delta);

    Some((new_p0, new_q0))
}

// ---------------------------------------------------------------------------
// Strong filter (bS = 4) — luma
// ---------------------------------------------------------------------------

/// Apply the strong (intra) deblocking filter to a luma pixel column/row.
///
/// Returns (p0', p1', p2', q0', q1', q2') or None if thresholds not met.
#[allow(clippy::too_many_arguments)] // mirrors FFmpeg h264_loop_filter_luma_intra per-pixel logic
#[inline]
fn filter_strong_luma(
    p0: i32,
    p1: i32,
    p2: i32,
    p3: i32,
    q0: i32,
    q1: i32,
    q2: i32,
    q3: i32,
    alpha: i32,
    beta: i32,
) -> Option<(u8, u8, u8, u8, u8, u8)> {
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return None;
    }

    let ap = (p2 - p0).abs();
    let aq = (q2 - q0).abs();
    let small_gap = (p0 - q0).abs() < ((alpha >> 2) + 2);

    let (new_p0, new_p1, new_p2);
    let (new_q0, new_q1, new_q2);

    if small_gap {
        if ap < beta {
            new_p0 = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
            new_p1 = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
            new_p2 = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
        } else {
            new_p0 = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            new_p1 = p1 as u8;
            new_p2 = p2 as u8;
        }

        if aq < beta {
            new_q0 = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) as u8;
            new_q1 = ((p0 + q0 + q1 + q2 + 2) >> 2) as u8;
            new_q2 = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
        } else {
            new_q0 = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
            new_q1 = q1 as u8;
            new_q2 = q2 as u8;
        }
    } else {
        // Weak form of the strong filter — only modify p0 and q0
        new_p0 = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
        new_p1 = p1 as u8;
        new_p2 = p2 as u8;
        new_q0 = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
        new_q1 = q1 as u8;
        new_q2 = q2 as u8;
    }

    Some((new_p0, new_p1, new_p2, new_q0, new_q1, new_q2))
}

// ---------------------------------------------------------------------------
// Strong filter (bS = 4) — chroma
// ---------------------------------------------------------------------------

/// Apply the strong (intra) deblocking filter to a chroma pixel pair.
///
/// Chroma intra filter: p0' = (2*p1 + p0 + q1 + 2) >> 2
///                       q0' = (2*q1 + q0 + p1 + 2) >> 2
#[inline]
fn filter_strong_chroma(
    p0: i32,
    p1: i32,
    q0: i32,
    q1: i32,
    alpha: i32,
    beta: i32,
) -> Option<(u8, u8)> {
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return None;
    }

    let new_p0 = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
    let new_q0 = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;

    Some((new_p0, new_q0))
}

// ---------------------------------------------------------------------------
// Edge filtering functions — luma (vertical and horizontal)
// ---------------------------------------------------------------------------

/// Filter a luma edge (vertical or horizontal), spanning 4 pixel pairs.
///
/// `is_vertical`: true = vertical edge (filter across columns), false = horizontal (filter across rows).
/// `edge`: 0 = MB boundary, 1..3 = internal edges.
/// `bs`: boundary strength for each of the 4 pixel pairs along the edge.
#[allow(clippy::too_many_arguments)] // edge filtering requires position, bS, QP, and offsets
fn filter_mb_edge_luma(
    is_vertical: bool,
    pic: &mut PictureBuffer,
    mb_x: u32,
    mb_y: u32,
    edge: usize,
    bs: [u8; 4],
    qp: u8,
    alpha_offset: i32,
    beta_offset: i32,
) {
    let (alpha, beta) = get_thresholds(qp, alpha_offset, beta_offset);
    if alpha == 0 || beta == 0 {
        return;
    }

    let stride = pic.y_stride;
    // For vertical edges the fixed coordinate is x; for horizontal it is y.
    let (x_base, y_base) = if is_vertical {
        (mb_x as usize * 16 + edge * 4, mb_y as usize * 16)
    } else {
        (mb_x as usize * 16, mb_y as usize * 16 + edge * 4)
    };

    for i in 0..4u8 {
        let cur_bs = bs[i as usize];
        if cur_bs == 0 {
            continue;
        }

        for d in 0..4usize {
            // Walk along the edge: for vertical, y varies; for horizontal, x varies.
            let off = if is_vertical {
                (y_base + i as usize * 4 + d) * stride + x_base
            } else {
                y_base * stride + x_base + i as usize * 4 + d
            };

            // Step size across the edge boundary.
            let step = if is_vertical { 1 } else { stride };

            let p0 = pic.y[off - step] as i32;
            let p1 = pic.y[off - 2 * step] as i32;
            let p2 = pic.y[off - 3 * step] as i32;
            let q0 = pic.y[off] as i32;
            let q1 = pic.y[off + step] as i32;
            let q2 = pic.y[off + 2 * step] as i32;

            if cur_bs < 4 {
                let tc0 = get_tc0(qp, alpha_offset, cur_bs);
                if let Some((new_p0, new_p1, new_q0, new_q1)) =
                    filter_normal_luma(p0, p1, p2, q0, q1, q2, alpha, beta, tc0)
                {
                    pic.y[off - step] = new_p0;
                    pic.y[off - 2 * step] = new_p1;
                    pic.y[off] = new_q0;
                    pic.y[off + step] = new_q1;
                }
            } else {
                let p3 = pic.y[off - 4 * step] as i32;
                let q3 = pic.y[off + 3 * step] as i32;
                if let Some((new_p0, new_p1, new_p2, new_q0, new_q1, new_q2)) =
                    filter_strong_luma(p0, p1, p2, p3, q0, q1, q2, q3, alpha, beta)
                {
                    pic.y[off - step] = new_p0;
                    pic.y[off - 2 * step] = new_p1;
                    pic.y[off - 3 * step] = new_p2;
                    pic.y[off] = new_q0;
                    pic.y[off + step] = new_q1;
                    pic.y[off + 2 * step] = new_q2;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Edge filtering functions — chroma (vertical and horizontal)
// ---------------------------------------------------------------------------

/// Filter a chroma edge (vertical or horizontal).
///
/// For 4:2:0, chroma MBs are 8x8 with 2 edges per direction
/// (0 = MB boundary, 1 = internal). Each edge spans 4 pixel pairs
/// (2 sub-blocks of 2 pixels each).
/// `is_vertical`: true = vertical edge (step=1), false = horizontal (step=stride).
#[allow(clippy::too_many_arguments)] // edge filtering requires plane, stride, position, bS, QP, and offsets
fn filter_mb_edge_chroma(
    is_vertical: bool,
    plane: &mut [u8],
    stride: usize,
    mb_x: u32,
    mb_y: u32,
    edge: usize,
    bs: [u8; 4],
    qp: u8,
    alpha_offset: i32,
    beta_offset: i32,
) {
    let (alpha, beta) = get_thresholds(qp, alpha_offset, beta_offset);
    if alpha == 0 || beta == 0 {
        return;
    }

    let (x_base, y_base) = if is_vertical {
        (mb_x as usize * 8 + edge * 4, mb_y as usize * 8)
    } else {
        (mb_x as usize * 8, mb_y as usize * 8 + edge * 4)
    };
    let step = if is_vertical { 1 } else { stride };

    for i in 0..4u8 {
        let cur_bs = bs[i as usize];
        if cur_bs == 0 {
            continue;
        }

        for d in 0..2usize {
            // Walk along the edge: for vertical, y varies; for horizontal, x varies.
            let off = if is_vertical {
                (y_base + i as usize * 2 + d) * stride + x_base
            } else {
                y_base * stride + x_base + i as usize * 2 + d
            };

            let p0 = plane[off - step] as i32;
            let p1 = plane[off - 2 * step] as i32;
            let q0 = plane[off] as i32;
            let q1 = plane[off + step] as i32;

            if cur_bs < 4 {
                let tc = get_tc0(qp, alpha_offset, cur_bs) + 1;
                if let Some((new_p0, new_q0)) =
                    filter_normal_chroma(p0, p1, q0, q1, alpha, beta, tc)
                {
                    plane[off - step] = new_p0;
                    plane[off] = new_q0;
                }
            } else if let Some((new_p0, new_q0)) = filter_strong_chroma(p0, p1, q0, q1, alpha, beta)
            {
                plane[off - step] = new_p0;
                plane[off] = new_q0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-macroblock deblocking
// ---------------------------------------------------------------------------

/// Compute boundary strengths for all 4 luma edges of a macroblock.
///
/// `is_vertical`: true = vertical edges (left neighbor), false = horizontal edges (above neighbor).
/// Returns `bs[edge][pair]` — 4 edges, each with 4 pairs of bS values.
fn compute_luma_bs(
    is_vertical: bool,
    mb_info: &[MbDeblockInfo],
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
) -> [[u8; 4]; 4] {
    let mut bs = [[0u8; 4]; 4];
    let mb_idx = (mb_y * mb_width + mb_x) as usize;
    let cur = &mb_info[mb_idx];

    // Use the current MB's list_count for the two-permutation check.
    // This matches FFmpeg's check_mv which uses sl->list_count (current slice).
    let list_count = cur.list_count;

    for (edge, bs_edge) in bs.iter_mut().enumerate() {
        let is_mb_edge = edge == 0;

        // For edge 0, the P block is in the neighboring macroblock.
        // Skip if there is no such neighbor.
        if is_mb_edge && (if is_vertical { mb_x == 0 } else { mb_y == 0 }) {
            continue;
        }

        for (pair, bs_val) in bs_edge.iter_mut().enumerate() {
            // For vertical edges: Q block at (col=edge, row=pair).
            // For horizontal edges: Q block at (col=pair, row=edge).
            let (q_bx, q_by) = if is_vertical {
                (edge, pair)
            } else {
                (pair, edge)
            };
            let q_idx = luma_block_idx(q_bx, q_by);
            let q_intra = cur.is_intra;
            let q_nnz = cur.non_zero_count[q_idx];
            let q_ref = cur.ref_poc[q_idx];
            let q_mv = cur.mv[q_idx];
            let q_ref_l1 = cur.ref_poc_l1[q_idx];
            let q_mv_l1 = cur.mv_l1[q_idx];

            // P block: one step in the opposite direction.
            let (p_intra, p_nnz, p_ref, p_mv, p_ref_l1, p_mv_l1) = if is_mb_edge {
                if is_vertical {
                    // P is in the left macroblock, rightmost column
                    let left = &mb_info[mb_idx - 1];
                    let p_idx = luma_block_idx(3, q_by);
                    (
                        left.is_intra,
                        left.non_zero_count[p_idx],
                        left.ref_poc[p_idx],
                        left.mv[p_idx],
                        left.ref_poc_l1[p_idx],
                        left.mv_l1[p_idx],
                    )
                } else {
                    // P is in the above macroblock, bottom row
                    let above = &mb_info[mb_idx - mb_width as usize];
                    let p_idx = luma_block_idx(q_bx, 3);
                    (
                        above.is_intra,
                        above.non_zero_count[p_idx],
                        above.ref_poc[p_idx],
                        above.mv[p_idx],
                        above.ref_poc_l1[p_idx],
                        above.mv_l1[p_idx],
                    )
                }
            } else {
                let p_idx = if is_vertical {
                    luma_block_idx(q_bx - 1, q_by)
                } else {
                    luma_block_idx(q_bx, q_by - 1)
                };
                (
                    cur.is_intra,
                    cur.non_zero_count[p_idx],
                    cur.ref_poc[p_idx],
                    cur.mv[p_idx],
                    cur.ref_poc_l1[p_idx],
                    cur.mv_l1[p_idx],
                )
            };

            *bs_val = compute_bs(
                is_mb_edge, p_intra, q_intra, p_nnz, q_nnz, p_ref, q_ref, p_mv, q_mv, p_ref_l1,
                q_ref_l1, p_mv_l1, q_mv_l1, list_count,
            );
        }
    }

    bs
}

/// Derive chroma boundary strengths from luma bS for 4:2:0.
///
/// In H.264, chroma deblocking uses the SAME bS as luma (computed purely from
/// luma block properties). The bS is mapped from the 4x4 luma grid to the 2x2
/// chroma grid by taking the maximum bS across the two luma pairs that correspond
/// to each chroma pair.
///
/// For vertical edges:
///   - Chroma edge 0 corresponds to luma edge 0, chroma edge 1 to luma edge 2
///   - Chroma pair 0 = max(luma_bs[pair=0], luma_bs[pair=1])
///   - Chroma pair 1 = max(luma_bs[pair=2], luma_bs[pair=3])
///
/// For horizontal edges: same mapping applies.
/// Derive chroma bS from luma bS for 4:2:0.
///
/// Chroma edges: 2 per direction (edge 0 = MB boundary, edge 1 = internal).
/// Each chroma edge has 4 bS values (one per 2-pixel pair along the 8-pixel edge),
/// mapped 1:1 from the corresponding luma edge's bS values.
/// Chroma edge 0 corresponds to luma edge 0; chroma edge 1 to luma edge 2.
fn derive_chroma_bs(luma_bs: &[[u8; 4]; 4]) -> [[u8; 4]; 2] {
    [luma_bs[0], luma_bs[2]]
}

/// Deblock a single macroblock (all luma and chroma edges).
fn deblock_mb(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    mb_x: u32,
    mb_y: u32,
    alpha_c0_offset: i32,
    beta_offset: i32,
    chroma_qp_index_offset: [i32; 2],
) {
    let mb_width = pic.mb_width;
    let mb_idx = (mb_y * mb_width + mb_x) as usize;
    let cur_qp = mb_info[mb_idx].qp;

    // Process vertical (is_vertical=true) then horizontal (is_vertical=false) edges.
    // Chroma uses the same bS as luma (derived by mapping the 4x4 luma grid to 4:2:0).
    for is_vertical in [true, false] {
        let luma_bs = compute_luma_bs(is_vertical, mb_info, mb_x, mb_y, mb_width);

        // --- Luma edges ---
        for (edge, &bs_edge) in luma_bs.iter().enumerate() {
            if bs_edge == [0, 0, 0, 0] {
                continue;
            }
            let qp = if edge == 0 {
                let neighbor_qp = if is_vertical && mb_x > 0 {
                    Some(mb_info[mb_idx - 1].qp)
                } else if !is_vertical && mb_y > 0 {
                    Some(mb_info[mb_idx - mb_width as usize].qp)
                } else {
                    None
                };
                neighbor_qp.map_or(cur_qp, |nq| avg_qp(cur_qp, nq))
            } else {
                cur_qp
            };
            filter_mb_edge_luma(
                is_vertical,
                pic,
                mb_x,
                mb_y,
                edge,
                bs_edge,
                qp,
                alpha_c0_offset,
                beta_offset,
            );
        }

        // --- Chroma edges (4:2:0: 2 edges per direction) ---
        // Cb and Cr use separate chroma_qp_index_offsets (PPS [0] and [1]).
        let chroma_bs = derive_chroma_bs(&luma_bs);
        for (edge, &bs_edge) in chroma_bs.iter().enumerate() {
            if bs_edge == [0, 0, 0, 0] {
                continue;
            }

            let uv_stride = pic.uv_stride;
            for (comp, &offset) in chroma_qp_index_offset.iter().enumerate() {
                let chroma_qp_cur = chroma_qp(cur_qp, offset);
                let qp = if edge == 0 {
                    let neighbor_qp = if is_vertical && mb_x > 0 {
                        Some(chroma_qp(mb_info[mb_idx - 1].qp, offset))
                    } else if !is_vertical && mb_y > 0 {
                        Some(chroma_qp(mb_info[mb_idx - mb_width as usize].qp, offset))
                    } else {
                        None
                    };
                    neighbor_qp.map_or(chroma_qp_cur, |nq| avg_qp(chroma_qp_cur, nq))
                } else {
                    chroma_qp_cur
                };

                let plane = if comp == 0 { &mut pic.u } else { &mut pic.v };
                filter_mb_edge_chroma(
                    is_vertical,
                    plane,
                    uv_stride,
                    mb_x,
                    mb_y,
                    edge,
                    bs_edge,
                    qp,
                    alpha_c0_offset,
                    beta_offset,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply the H.264 in-loop deblocking filter to an entire frame.
///
/// This must be called after all macroblocks in a slice have been decoded.
/// The filter modifies the picture buffer in-place, and the filtered output
/// is used as the reference for inter prediction of future frames.
///
/// # Arguments
///
/// * `pic` - the decoded picture buffer (modified in-place)
/// * `mb_info` - per-macroblock deblocking info (mb_width * mb_height entries, raster order)
/// * `disable_deblocking_filter_idc` - 0 = filter all edges, 1 = disable filter,
///   2 = disable filter across slice boundaries (treated as 0 in this implementation since
///   we don't track slice boundaries per-MB)
/// * `alpha_c0_offset` - from slice header (already multiplied by 2)
/// * `beta_offset` - from slice header (already multiplied by 2)
/// * `chroma_qp_index_offset` - from PPS: \[0\] for Cb, \[1\] for Cr
#[cfg_attr(feature = "tracing-detail", tracing::instrument(skip_all, fields(mb_width = pic.mb_width, mb_height = pic.mb_height)))]
pub fn deblock_frame(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    disable_deblocking_filter_idc: u32,
    alpha_c0_offset: i32,
    beta_offset: i32,
    chroma_qp_index_offset: [i32; 2],
) {
    if disable_deblocking_filter_idc == 1 {
        return;
    }

    let mb_width = pic.mb_width;
    let mb_height = pic.mb_height;

    debug!(mb_width, mb_height, "deblocking frame");

    debug_assert_eq!(
        mb_info.len(),
        (mb_width * mb_height) as usize,
        "mb_info length must equal mb_width * mb_height"
    );

    // Process macroblocks in raster order
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            deblock_mb(
                pic,
                mb_info,
                mb_x,
                mb_y,
                alpha_c0_offset,
                beta_offset,
                chroma_qp_index_offset,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal PictureBuffer for testing (1 MB = 16x16 luma, 8x8 chroma).
    fn make_pic_1mb(y_data: &[u8; 16 * 16]) -> PictureBuffer {
        PictureBuffer {
            y: y_data.to_vec(),
            u: vec![128; 8 * 8],
            v: vec![128; 8 * 8],
            y_stride: 16,
            uv_stride: 8,
            width: 16,
            height: 16,
            mb_width: 1,
            mb_height: 1,
        }
    }

    /// Create a 2x1 MB picture (32x16 luma, 16x8 chroma) for testing MB boundary edges.
    fn make_pic_2x1(y_data: &[u8]) -> PictureBuffer {
        assert_eq!(y_data.len(), 32 * 16);
        PictureBuffer {
            y: y_data.to_vec(),
            u: vec![128; 16 * 8],
            v: vec![128; 16 * 8],
            y_stride: 32,
            uv_stride: 16,
            width: 32,
            height: 16,
            mb_width: 2,
            mb_height: 1,
        }
    }

    /// Create a 1x2 MB picture (16x32 luma, 8x16 chroma) for testing horizontal MB edges.
    fn make_pic_1x2(y_data: &[u8]) -> PictureBuffer {
        assert_eq!(y_data.len(), 16 * 32);
        PictureBuffer {
            y: y_data.to_vec(),
            u: vec![128; 8 * 16],
            v: vec![128; 8 * 16],
            y_stride: 16,
            uv_stride: 8,
            width: 16,
            height: 32,
            mb_width: 1,
            mb_height: 2,
        }
    }

    // --- Boundary strength tests ---

    /// P-slice helper: compute_bs with list_count=1 (no L1 data).
    fn bs_p(
        is_mb_edge: bool,
        p_intra: bool,
        q_intra: bool,
        p_nnz: u8,
        q_nnz: u8,
        p_ref: i32,
        q_ref: i32,
        p_mv: [i16; 2],
        q_mv: [i16; 2],
    ) -> u8 {
        compute_bs(
            is_mb_edge,
            p_intra,
            q_intra,
            p_nnz,
            q_nnz,
            p_ref,
            q_ref,
            p_mv,
            q_mv,
            i32::MIN,
            i32::MIN,
            [0, 0],
            [0, 0],
            1,
        )
    }

    #[test]
    fn bs_both_intra_mb_edge() {
        assert_eq!(bs_p(true, true, true, 0, 0, 0, 0, [0, 0], [0, 0]), 4);
    }

    #[test]
    fn bs_one_intra_mb_edge() {
        assert_eq!(bs_p(true, true, false, 0, 0, 0, 0, [0, 0], [0, 0]), 4);
        assert_eq!(bs_p(true, false, true, 0, 0, 0, 0, [0, 0], [0, 0]), 4);
    }

    #[test]
    fn bs_intra_internal_edge() {
        assert_eq!(bs_p(false, true, true, 0, 0, 0, 0, [0, 0], [0, 0]), 3);
        assert_eq!(bs_p(false, true, false, 0, 0, 0, 0, [0, 0], [0, 0]), 3);
    }

    #[test]
    fn bs_nonzero_coeffs() {
        assert_eq!(bs_p(false, false, false, 1, 0, 0, 0, [0, 0], [0, 0]), 2);
        assert_eq!(bs_p(false, false, false, 0, 1, 0, 0, [0, 0], [0, 0]), 2);
        assert_eq!(bs_p(false, false, false, 5, 3, 0, 0, [0, 0], [0, 0]), 2);
    }

    #[test]
    fn bs_different_ref() {
        assert_eq!(bs_p(false, false, false, 0, 0, 0, 1, [0, 0], [0, 0]), 1);
    }

    #[test]
    fn bs_mv_diff_x() {
        assert_eq!(bs_p(false, false, false, 0, 0, 0, 0, [0, 0], [4, 0]), 1);
        assert_eq!(bs_p(false, false, false, 0, 0, 0, 0, [0, 0], [3, 0]), 0);
        assert_eq!(bs_p(false, false, false, 0, 0, 0, 0, [2, 0], [-2, 0]), 1);
    }

    #[test]
    fn bs_mv_diff_y() {
        assert_eq!(bs_p(false, false, false, 0, 0, 0, 0, [0, 0], [0, 4]), 1);
        assert_eq!(bs_p(false, false, false, 0, 0, 0, 0, [0, 0], [0, 3]), 0);
    }

    #[test]
    fn bs_zero_no_filtering() {
        assert_eq!(bs_p(false, false, false, 0, 0, 0, 0, [0, 0], [0, 0]), 0);
        assert_eq!(bs_p(false, false, false, 0, 0, 5, 5, [1, 2], [3, 4]), 0);
    }

    #[test]
    fn bs_b_frame_two_permutation() {
        // B-frame: blocks with opposite lists should NOT be filtered.
        // P: L0 ref=0 mv=[0,0], L1 ref=1 mv=[2,0]
        // Q: L0 ref=1 mv=[2,0], L1 ref=0 mv=[0,0]
        // Same-list check fails (L0 refs differ, L1 refs differ),
        // but cross-list matches (P.L0==Q.L1, P.L1==Q.L0, MVs match).
        assert_eq!(
            compute_bs(
                false,
                false,
                false,
                0,
                0,
                0,
                1,
                [0, 0],
                [2, 0],
                1,
                0,
                [2, 0],
                [0, 0],
                2,
            ),
            0
        );

        // B-frame: same-list check passes → bS=0 without needing cross-check.
        assert_eq!(
            compute_bs(
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                [0, 0],
                [0, 0],
                1,
                1,
                [1, 1],
                [1, 1],
                2,
            ),
            0
        );

        // B-frame: different refs on both lists, no cross match → bS=1.
        assert_eq!(
            compute_bs(
                false,
                false,
                false,
                0,
                0,
                0,
                1,
                [0, 0],
                [0, 0],
                0,
                1,
                [0, 0],
                [0, 0],
                2,
            ),
            1
        );
    }

    // --- Threshold table tests ---

    #[test]
    fn alpha_table_boundary_values() {
        assert_eq!(ALPHA_TABLE[0], 0);
        assert_eq!(ALPHA_TABLE[15], 0);
        assert_eq!(ALPHA_TABLE[16], 4);
        assert_eq!(ALPHA_TABLE[51], 255);
    }

    #[test]
    fn beta_table_boundary_values() {
        assert_eq!(BETA_TABLE[0], 0);
        assert_eq!(BETA_TABLE[15], 0);
        assert_eq!(BETA_TABLE[16], 2);
        assert_eq!(BETA_TABLE[51], 18);
    }

    #[test]
    fn tc0_table_matches_ffmpeg() {
        // Full table verified against FFmpeg tc0_table in h264_loopfilter.c
        // (bS=1,2,3 columns; bS=0 column is -1 in FFmpeg, not stored here)
        let expected: [[i32; 3]; 52] = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 2],
            [1, 1, 2],
            [1, 2, 3],
            [1, 2, 3],
            [2, 2, 3],
            [2, 2, 4],
            [2, 3, 4],
            [2, 3, 4],
            [3, 3, 5],
            [3, 4, 6],
            [3, 4, 6],
            [4, 5, 7],
            [4, 5, 8],
            [4, 6, 9],
            [5, 7, 10],
            [6, 8, 11],
            [6, 8, 13],
            [7, 10, 14],
            [8, 11, 16],
            [9, 12, 18],
            [10, 13, 20],
            [11, 15, 23],
            [13, 17, 25],
        ];
        for i in 0..52 {
            assert_eq!(
                TC0_TABLE[i], expected[i],
                "TC0_TABLE[{i}]: got {:?}, expected {:?}",
                TC0_TABLE[i], expected[i]
            );
        }
    }

    #[test]
    fn get_thresholds_clamping() {
        // QP 0 + offset -10 should clamp to index 0
        let (alpha, beta) = get_thresholds(0, -10, -10);
        assert_eq!(alpha, 0);
        assert_eq!(beta, 0);

        // QP 51 + offset +10 should clamp to index 51
        let (alpha, beta) = get_thresholds(51, 10, 10);
        assert_eq!(alpha, 255);
        assert_eq!(beta, 18);
    }

    // --- Strong filter (bS=4) tests ---

    #[test]
    fn strong_filter_luma_basic() {
        // Test with pixels that should trigger filtering.
        // p3=120, p2=125, p1=130, p0=140, | q0=160, q1=155, q2=150, q3=145
        // QP=35: alpha=45, beta=10
        // |p0-q0|=20 < 45, |p1-p0|=10 == beta => fails threshold
        // Use QP=36 where beta=11 so |p1-p0|=10 < 11
        let (alpha, beta) = get_thresholds(36, 0, 0);
        assert_eq!(alpha, 50);
        assert_eq!(beta, 11);

        let result = filter_strong_luma(140, 130, 125, 120, 160, 155, 150, 145, alpha, beta);
        assert!(result.is_some());

        let (p0, p1, p2, q0, q1, q2) = result.unwrap();
        // Verify p0 changed (should be filtered)
        assert_ne!(p0, 140);
        assert_ne!(q0, 160);

        // Check the small_gap condition: |p0-q0|=20 < (50/4+2)=14 is false.
        // So it uses the weak form: p0' = (2*130 + 140 + 155 + 2) >> 2 = 139
        assert_eq!(p0, ((2 * 130 + 140 + 155 + 2) >> 2) as u8);
        assert_eq!(q0, ((2 * 155 + 160 + 130 + 2) >> 2) as u8);
        // p1, p2, q1, q2 unchanged in weak form
        assert_eq!(p1, 130);
        assert_eq!(p2, 125);
        assert_eq!(q1, 155);
        assert_eq!(q2, 150);
    }

    #[test]
    fn strong_filter_luma_full_strong() {
        // Craft pixels where |p0-q0| < (alpha >> 2) + 2 and |p2-p0| < beta
        // to trigger full strong filtering.
        // QP=40: alpha=80, beta=13
        let (alpha, beta) = get_thresholds(40, 0, 0);
        assert_eq!(alpha, 80);
        assert_eq!(beta, 13);

        // p3=125, p2=128, p1=130, p0=132, | q0=135, q1=137, q2=139, q3=141
        // |p0-q0| = 3 < (80/4+2) = 22 => small_gap = true
        // |p2-p0| = 4 < 13 => full p-side strong filter
        // |q2-q0| = 4 < 13 => full q-side strong filter
        let result = filter_strong_luma(132, 130, 128, 125, 135, 137, 139, 141, alpha, beta);
        assert!(result.is_some());
        let (p0, p1, p2, q0, q1, q2) = result.unwrap();

        // p0' = (p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3
        //     = (128 + 260 + 264 + 270 + 137 + 4) >> 3 = 1063 >> 3 = 132
        assert_eq!(
            p0,
            ((128 + 2 * 130 + 2 * 132 + 2 * 135 + 137 + 4) >> 3) as u8
        );
        // p1' = (p2 + p1 + p0 + q0 + 2) >> 2
        assert_eq!(p1, ((128 + 130 + 132 + 135 + 2) >> 2) as u8);
        // p2' = (2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3
        assert_eq!(p2, ((2 * 125 + 3 * 128 + 130 + 132 + 135 + 4) >> 3) as u8);
        // q0' = (p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4) >> 3
        assert_eq!(
            q0,
            ((130 + 2 * 132 + 2 * 135 + 2 * 137 + 139 + 4) >> 3) as u8
        );
        // q1' = (p0 + q0 + q1 + q2 + 2) >> 2
        assert_eq!(q1, ((132 + 135 + 137 + 139 + 2) >> 2) as u8);
        // q2' = (2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3
        assert_eq!(q2, ((2 * 141 + 3 * 139 + 137 + 135 + 132 + 4) >> 3) as u8);
    }

    #[test]
    fn strong_filter_threshold_not_met() {
        // |p0-q0| >= alpha => no filtering
        let result = filter_strong_luma(100, 90, 80, 70, 200, 210, 220, 230, 10, 5);
        assert!(result.is_none());
    }

    // --- Normal filter tests ---

    #[test]
    fn normal_filter_luma_basic() {
        // QP=30, bS=2 => tc0 = TC0_TABLE[30][1] = 1
        let tc0 = get_tc0(30, 0, 2);
        assert_eq!(tc0, 1);

        let (alpha, beta) = get_thresholds(30, 0, 0);
        // alpha=25, beta=8

        // p2=120, p1=125, p0=130, q0=135, q1=140, q2=145
        // |p0-q0|=5 < 25, |p1-p0|=5 < 8, |q1-q0|=5 < 8 => filter
        let result = filter_normal_luma(130, 125, 120, 135, 140, 145, alpha, beta, tc0);
        assert!(result.is_some());

        let (new_p0, new_p1, new_q0, new_q1) = result.unwrap();
        // delta = clip3(-tc, tc, ((135-130)*4 + (125-140) + 4) >> 3)
        //       = clip3(-tc, tc, (20 - 15 + 4) >> 3) = clip3(-tc, tc, 9 >> 3) = clip3(-tc, tc, 1)
        // Need to compute tc: |p2-p0|=10 >= beta=8 => no p1 filter, no tc increment from p
        //                      |q2-q0|=10 >= beta=8 => no q1 filter, no tc increment from q
        // tc = tc0 = 1
        // delta = clip3(-1, 1, 1) = 1
        assert_eq!(new_p0, 131); // 130 + 1
        assert_eq!(new_q0, 134); // 135 - 1
        // p1 and q1 unchanged since |p2-p0| and |q2-q0| >= beta
        assert_eq!(new_p1, 125);
        assert_eq!(new_q1, 140);
    }

    #[test]
    fn normal_filter_luma_with_p1_q1() {
        // Use QP=40 where beta=13 so |p2-p0| < beta triggers p1 filtering
        let tc0 = get_tc0(40, 0, 2);
        let (alpha, beta) = get_thresholds(40, 0, 0);
        // alpha=80, beta=13

        // p2=128, p1=130, p0=132, q0=138, q1=140, q2=142
        // |p0-q0|=6 < 80, |p1-p0|=2 < 13, |q1-q0|=2 < 13 => filter
        // |p2-p0|=4 < 13 => filter p1, tc++
        // |q2-q0|=4 < 11 => filter q1, tc++
        let result = filter_normal_luma(132, 130, 128, 138, 140, 142, alpha, beta, tc0);
        assert!(result.is_some());

        let (new_p0, _new_p1, new_q0, _new_q1) = result.unwrap();
        // p1 and q1 should be modified
        // tc = tc0 + 2 (both p2 and q2 close)
        // delta = clip3(-(tc0+2), (tc0+2), ((138-132)*4 + (130-140) + 4) >> 3)
        //       = clip3(-(tc0+2), (tc0+2), (24 - 10 + 4) >> 3)
        //       = clip3(-(tc0+2), (tc0+2), 18 >> 3) = clip3(-(tc0+2), (tc0+2), 2)
        assert!(new_p0 > 132); // p0 increased
        assert!(new_q0 < 138); // q0 decreased
    }

    #[test]
    fn normal_filter_threshold_not_met() {
        // |p0-q0| >= alpha => no filtering
        let result = filter_normal_luma(100, 90, 80, 200, 210, 220, 10, 5, 2);
        assert!(result.is_none());

        // |p1-p0| >= beta => no filtering
        let result = filter_normal_luma(100, 80, 70, 105, 110, 120, 255, 5, 2);
        assert!(result.is_none());
    }

    // --- Chroma filter tests ---

    #[test]
    fn strong_filter_chroma_basic() {
        let (alpha, beta) = get_thresholds(35, 0, 0);

        // p1=125, p0=130, q0=135, q1=140
        // |p0-q0|=5 < alpha, |p1-p0|=5 < beta, |q1-q0|=5 < beta
        let result = filter_strong_chroma(130, 125, 135, 140, alpha, beta);
        assert!(result.is_some());
        let (new_p0, new_q0) = result.unwrap();
        assert_eq!(new_p0, ((2 * 125 + 130 + 140 + 2) >> 2) as u8);
        assert_eq!(new_q0, ((2 * 140 + 135 + 125 + 2) >> 2) as u8);
    }

    #[test]
    fn normal_filter_chroma_basic() {
        let (alpha, beta) = get_thresholds(35, 0, 0);
        let tc = get_tc0(35, 0, 2) + 1;

        let result = filter_normal_chroma(130, 125, 135, 140, alpha, beta, tc);
        assert!(result.is_some());
        let (new_p0, new_q0) = result.unwrap();
        // delta = clip3(-tc, tc, ((135-130)*4 + (125-140) + 4) >> 3)
        //       = clip3(-tc, tc, (20 - 15 + 4) >> 3) = clip3(-tc, tc, 1)
        assert_eq!(new_p0, 131);
        assert_eq!(new_q0, 134);
    }

    // --- Filter symmetry test ---

    #[test]
    fn filter_symmetry() {
        // When p and q sides are symmetric (p1=q1 relative to the edge),
        // the filter should produce symmetric results. Use values where
        // p1-p0 == q1-q0 to ensure the (p1-q1) term is zero.
        let alpha = 100;
        let beta = 20;
        // Symmetric setup: p0=130, p1=125, p2=120, q0=140, q1=145, q2=150
        // Mirror:           p0=140, p1=145, p2=150, q0=130, q1=125, q2=120
        // The p1-q1 term is (125-145)=-20 for r1 and (145-125)=20 for r2.
        // delta_r1 = clip(-tc, tc, ((140-130)*4 + (125-145) + 4) >> 3) = clip(-tc, tc, (40-20+4)>>3) = clip(-tc, tc, 3)
        // delta_r2 = clip(-tc, tc, ((130-140)*4 + (145-125) + 4) >> 3) = clip(-tc, tc, (-40+20+4)>>3) = clip(-tc, tc, -2)
        // Not equal due to +4 rounding bias. This is by design in H.264.

        // Instead test that the strong chroma filter IS symmetric
        // since it has no rounding bias term.
        let r1 = filter_strong_chroma(130, 120, 140, 150, alpha, beta);
        let r2 = filter_strong_chroma(140, 150, 130, 120, alpha, beta);

        assert!(r1.is_some());
        assert!(r2.is_some());

        let (p0_1, q0_1) = r1.unwrap();
        let (p0_2, q0_2) = r2.unwrap();

        // p0 from first call should equal q0 from second call and vice versa
        assert_eq!(p0_1, q0_2);
        assert_eq!(q0_1, p0_2);
    }

    // --- bS=0 produces no changes ---

    #[test]
    fn bs_zero_no_filtering_2() {
        let mut y_data = [128u8; 16 * 16];
        // Put a sharp edge at x=4 (edge 1 within the single MB)
        for row in 0..16 {
            for col in 0..4 {
                y_data[row * 16 + col] = 100;
            }
            for col in 4..16 {
                y_data[row * 16 + col] = 200;
            }
        }

        let y_original = y_data;
        let mut pic = make_pic_1mb(&y_data);

        // Single intra MB (no neighbors) — no MB boundary edges to filter,
        // internal edges have bS=3 for intra. Let's use a non-intra MB with
        // all zero NNZ and same ref/mv to get bS=0.
        let mb_info = vec![MbDeblockInfo {
            is_intra: false,
            qp: 30,
            non_zero_count: [0; 24],
            ref_poc: [0; 16],
            mv: [[0, 0]; 16],
            ..Default::default()
        }];

        deblock_frame(&mut pic, &mb_info, 0, 0, 0, [0, 0]);

        // With bS=0 everywhere (single MB, no neighbors, inter with identical motion),
        // nothing should change.
        assert_eq!(&pic.y[..], &y_original[..]);
    }

    // --- Integration: deblock_frame with disable_deblocking_filter_idc=1 ---

    #[test]
    fn deblock_disabled() {
        let y_data = [128u8; 16 * 16];
        let mut pic = make_pic_1mb(&y_data);
        let mb_info = vec![MbDeblockInfo {
            is_intra: true,
            qp: 30,
            non_zero_count: [1; 24],
            ref_poc: [0; 16],
            mv: [[0, 0]; 16],
            ..Default::default()
        }];

        let y_before = pic.y.clone();
        deblock_frame(&mut pic, &mb_info, 1, 0, 0, [0, 0]);
        assert_eq!(pic.y, y_before);
    }

    // --- Integration: vertical MB boundary with bS=4 ---

    #[test]
    fn deblock_vertical_mb_boundary_strong() {
        // 2 MBs side by side, both intra. The boundary at x=16 gets bS=4.
        // Use QP=51 so alpha=255, which is large enough for the 100->200 step.
        let mut y_data = vec![0u8; 32 * 16];
        for row in 0..16 {
            for col in 0..16 {
                y_data[row * 32 + col] = 100;
            }
            for col in 16..32 {
                y_data[row * 32 + col] = 200;
            }
        }

        let y_before = y_data.clone();
        let mut pic = make_pic_2x1(&y_data);

        let mb_info = vec![
            MbDeblockInfo {
                is_intra: true,
                qp: 51,
                non_zero_count: [0; 24],
                ref_poc: [0; 16],
                mv: [[0, 0]; 16],
                ..Default::default()
            },
            MbDeblockInfo {
                is_intra: true,
                qp: 51,
                non_zero_count: [0; 24],
                ref_poc: [0; 16],
                mv: [[0, 0]; 16],
                ..Default::default()
            },
        ];

        deblock_frame(&mut pic, &mb_info, 0, 0, 0, [0, 0]);

        // The pixels near x=16 boundary should have been modified
        let mut boundary_changed = false;
        for row in 0..16 {
            for col in 13..19 {
                if pic.y[row * 32 + col] != y_before[row * 32 + col] {
                    boundary_changed = true;
                }
            }
        }
        assert!(
            boundary_changed,
            "Strong filter should modify boundary pixels"
        );

        // Pixels far from any edge should be unchanged
        assert_eq!(pic.y[0], 100); // row 0, col 0
        assert_eq!(pic.y[31], 200); // row 0, col 31
    }

    // --- Integration: horizontal MB boundary with bS=4 ---

    #[test]
    fn deblock_horizontal_mb_boundary_strong() {
        // 2 MBs stacked vertically, both intra.
        // Use QP=51 so alpha=255, large enough for the 100->200 step.
        let mut y_data = vec![0u8; 16 * 32];
        for row in 0..16 {
            for col in 0..16 {
                y_data[row * 16 + col] = 100;
            }
        }
        for row in 16..32 {
            for col in 0..16 {
                y_data[row * 16 + col] = 200;
            }
        }

        let y_before = y_data.clone();
        let mut pic = make_pic_1x2(&y_data);

        let mb_info = vec![
            MbDeblockInfo {
                is_intra: true,
                qp: 51,
                non_zero_count: [0; 24],
                ref_poc: [0; 16],
                mv: [[0, 0]; 16],
                ..Default::default()
            },
            MbDeblockInfo {
                is_intra: true,
                qp: 51,
                non_zero_count: [0; 24],
                ref_poc: [0; 16],
                mv: [[0, 0]; 16],
                ..Default::default()
            },
        ];

        deblock_frame(&mut pic, &mb_info, 0, 0, 0, [0, 0]);

        // The pixels near y=16 boundary should have been modified
        let mut boundary_changed = false;
        for row in 13..19 {
            for col in 0..16 {
                if pic.y[row * 16 + col] != y_before[row * 16 + col] {
                    boundary_changed = true;
                }
            }
        }
        assert!(
            boundary_changed,
            "Strong filter should modify horizontal boundary pixels"
        );
    }

    // --- Chroma QP mapping ---

    #[test]
    fn chroma_qp_mapping() {
        // offset=0: identity through table
        assert_eq!(chroma_qp(0, 0), 0);
        assert_eq!(chroma_qp(29, 0), 29);
        assert_eq!(chroma_qp(30, 0), 29);
        assert_eq!(chroma_qp(51, 0), 39);
        // negative offset: shifts QP down before lookup
        assert_eq!(chroma_qp(30, -1), 29); // 30-1=29 → table[29]=29
        assert_eq!(chroma_qp(10, -12), 0); // 10-12=-2 → clamped to 0 → table[0]=0
    }

    // --- Average QP ---

    #[test]
    fn avg_qp_values() {
        assert_eq!(avg_qp(20, 30), 25);
        assert_eq!(avg_qp(21, 30), 26); // (21+30+1)/2 = 26
        assert_eq!(avg_qp(0, 0), 0);
        assert_eq!(avg_qp(51, 51), 51);
    }

    // --- clip3 and clip_pixel ---

    #[test]
    fn clip3_basic() {
        assert_eq!(clip3(-5, 5, 10), 5);
        assert_eq!(clip3(-5, 5, -10), -5);
        assert_eq!(clip3(-5, 5, 3), 3);
    }

    #[test]
    fn clip_pixel_basic() {
        assert_eq!(clip_pixel(300), 255);
        assert_eq!(clip_pixel(-10), 0);
        assert_eq!(clip_pixel(128), 128);
    }

    // --- Normal filter: bS=1 known values ---

    #[test]
    fn normal_filter_bs1_known() {
        // bS=1, QP=30: tc0 = TC0_TABLE[30][0] = 1
        let tc0 = get_tc0(30, 0, 1);
        assert_eq!(tc0, 1);

        let (alpha, beta) = get_thresholds(30, 0, 0);

        // Small difference at boundary: p0=130, q0=135, p1=128, q1=137, p2=126, q2=139
        // |p0-q0|=5 < 25=alpha, |p1-p0|=2 < 8=beta, |q1-q0|=2 < 8=beta => filter
        let result = filter_normal_luma(130, 128, 126, 135, 137, 139, alpha, beta, tc0);
        assert!(result.is_some());

        let (new_p0, new_p1, new_q0, new_q1) = result.unwrap();

        // |p2-p0|=4 < 8 => filter p1 (tc0=1 != 0), tc becomes 2
        // |q2-q0|=4 < 8 => filter q1 (tc0=1 != 0), tc becomes 3
        // delta = clip3(-3, 3, ((135-130)*4 + (128-137) + 4) >> 3)
        //       = clip3(-3, 3, (20 - 9 + 4) >> 3) = clip3(-3, 3, 15 >> 3) = clip3(-3, 3, 1) = 1
        assert_eq!(new_p0, 131); // 130 + 1
        assert_eq!(new_q0, 134); // 135 - 1

        // p1' = p1 + clip3(-1, 1, (p2 + ((p0+q0+1)>>1))>>1 - p1)
        //     = 128 + clip3(-1, 1, (126 + ((130+135+1)>>1))>>1 - 128)
        //     = 128 + clip3(-1, 1, (126 + 133)>>1 - 128)
        //     = 128 + clip3(-1, 1, 129 - 128) = 128 + 1 = 129
        assert_eq!(new_p1, 129);

        // q1' = q1 + clip3(-1, 1, (q2 + ((p0+q0+1)>>1))>>1 - q1)
        //     = 137 + clip3(-1, 1, (139 + 133)>>1 - 137)
        //     = 137 + clip3(-1, 1, 136 - 137) = 137 + (-1) = 136
        assert_eq!(new_q1, 136);
    }
}
