// H.264/AVC in-loop deblocking filter.
//
// Reduces blocking artifacts at macroblock and 4x4 block boundaries.
// Runs after all macroblocks in a slice are decoded; the filtered output
// is used for inter prediction of future frames (in-loop).
//
// Reference: ITU-T H.264 spec section 8.7, FFmpeg libavcodec/h264_loopfilter.c
// and h264dsp_template.c.

use std::sync::atomic::{AtomicU32, Ordering};

use tracing::{debug, trace};

use crate::tables::CHROMA_QP_TABLE;

/// Wrapper to send a `*mut PictureBuffer` across scoped threads.
///
/// SAFETY: Only used within `std::thread::scope` where the wavefront ordering
/// guarantees disjoint row access. All threads are joined before the pointer
/// is invalidated.
struct SyncPic(*mut PictureBuffer);
unsafe impl Send for SyncPic {}
unsafe impl Sync for SyncPic {}

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

/// Per-slice deblocking filter parameters.
/// Each slice in a frame can have different alpha/beta offsets and idc.
#[derive(Clone, Default)]
pub struct SliceDeblockParams {
    pub alpha_c0_offset: i32,
    pub beta_offset: i32,
    pub disable_deblocking_filter_idc: u32,
    pub chroma_qp_index_offset: [i32; 2],
}

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
    /// True if this MB uses 8x8 transform (High profile).
    pub transform_8x8: bool,
    /// Coded block pattern (luma bits 0-3) for deblocking NNZ override.
    /// For CAVLC 8x8 DCT: derived from NNZ sums (!!nnz_sum per 8x8 block),
    /// matching FFmpeg's cbp_table bits 12-15. NOT the bitstream CBP.
    pub cbp: u8,
    /// True if this MB was decoded with CABAC entropy coding.
    /// FFmpeg gates the 8x8 NNZ override on `!CABAC` (h264_slice.c:2396).
    pub is_cabac: bool,
    /// True if this MB was decoded in field mode (MBAFF).
    /// Controls mvy_limit (2 vs 4), pixel stride doubling, and bS for interlaced edges.
    pub mb_field: bool,
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
            transform_8x8: false,
            cbp: 0,
            is_cabac: false,
            mb_field: false,
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

impl PictureBuffer {
    /// Zero-sized sentinel for `mem::replace` in SharedPicture::Drop.
    pub fn empty() -> Self {
        Self {
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            y_stride: 0,
            uv_stride: 0,
            width: 0,
            height: 0,
            mb_width: 0,
            mb_height: 0,
        }
    }
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
    mvy_limit: u16,
) -> bool {
    // Step 1: Check L0 ref and MV (same for P and B slices)
    // X threshold is always 4 (FFmpeg: `+ 3 >= 7U`); Y threshold is mvy_limit
    // (2 for field MBs, 4 for frame MBs — FFmpeg h264_loopfilter.c:723).
    let mut v = p_ref != q_ref
        || (p_mv[0] - q_mv[0]).unsigned_abs() >= 4
        || (p_mv[1] - q_mv[1]).unsigned_abs() >= mvy_limit;

    if list_count == 2 {
        // Step 2: If L0 passed (v==false), also check L1
        if !v {
            v = p_ref_l1 != q_ref_l1
                || (p_mv_l1[0] - q_mv_l1[0]).unsigned_abs() >= 4
                || (p_mv_l1[1] - q_mv_l1[1]).unsigned_abs() >= mvy_limit;
        }
        // Step 3: If either same-list check failed, try cross-list permutation.
        // Blocks might use opposite lists with the same prediction.
        if v {
            if p_ref != q_ref_l1 || p_ref_l1 != q_ref {
                return true;
            }
            return (p_mv[0] - q_mv_l1[0]).unsigned_abs() >= 4
                || (p_mv[1] - q_mv_l1[1]).unsigned_abs() >= mvy_limit
                || (p_mv_l1[0] - q_mv[0]).unsigned_abs() >= 4
                || (p_mv_l1[1] - q_mv[1]).unsigned_abs() >= mvy_limit;
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
    mvy_limit: u16,
    is_interlaced_edge: bool,
) -> u8 {
    if p_intra || q_intra {
        // FFmpeg h264_loopfilter.c:547-552: bS=3 when IS_INTERLACED(mb_type|mbm_type)
        // for MB boundary edges; internal intra edges are always bS=3.
        if is_mb_edge && !is_interlaced_edge {
            4
        } else {
            3
        }
    } else if p_nnz != 0 || q_nnz != 0 {
        2
    } else if check_mv(
        p_ref, q_ref, p_mv, q_mv, p_ref_l1, q_ref_l1, p_mv_l1, q_mv_l1, list_count, mvy_limit,
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
/// `mb_base_offset`: pre-computed byte offset of the MB's top-left pixel in the Y plane.
/// `stride`: row stride (doubled for field-mode MBs in MBAFF).
#[allow(clippy::too_many_arguments)] // edge filtering requires position, bS, QP, and offsets
fn filter_mb_edge_luma(
    is_vertical: bool,
    plane: &mut [u8],
    stride: usize,
    mb_base_offset: usize,
    edge: usize,
    bs: [u8; 4],
    qp: u8,
    alpha_offset: i32,
    beta_offset: i32,
    mb_x: u32,
    mb_y: u32,
) {
    let (alpha, beta) = get_thresholds(qp, alpha_offset, beta_offset);
    if alpha == 0 || beta == 0 {
        return;
    }

    // For vertical edges: fixed x = edge*4, varying y.
    // For horizontal edges: fixed y = edge*4, varying x.
    let edge_pixel_offset = if is_vertical {
        edge * 4 // x offset within MB
    } else {
        edge * 4 * stride // y offset within MB
    };
    let base = mb_base_offset + edge_pixel_offset;

    for i in 0..4u8 {
        let cur_bs = bs[i as usize];
        if cur_bs == 0 {
            continue;
        }

        for d in 0..4usize {
            // Walk along the edge: for vertical, y varies; for horizontal, x varies.
            let off = if is_vertical {
                base + (i as usize * 4 + d) * stride
            } else {
                base + i as usize * 4 + d
            };

            // Step size across the edge boundary.
            let step = if is_vertical { 1 } else { stride };

            let p0 = plane[off - step] as i32;
            let p1 = plane[off - 2 * step] as i32;
            let p2 = plane[off - 3 * step] as i32;
            let q0 = plane[off] as i32;
            let q1 = plane[off + step] as i32;
            let q2 = plane[off + 2 * step] as i32;

            trace!(
                mb_x,
                mb_y,
                is_vertical,
                edge,
                i,
                d,
                cur_bs,
                p0,
                p1,
                p2,
                q0,
                q1,
                q2,
                alpha,
                beta,
                qp,
                "DEBLOCK_PIXEL_IN"
            );

            if cur_bs < 4 {
                let tc0 = get_tc0(qp, alpha_offset, cur_bs);
                if let Some((new_p0, new_p1, new_q0, new_q1)) =
                    filter_normal_luma(p0, p1, p2, q0, q1, q2, alpha, beta, tc0)
                {
                    trace!(
                        mb_x,
                        mb_y, is_vertical, edge, i, d, new_p0, new_q0, tc0, "DEBLOCK_PIXEL_OUT"
                    );
                    plane[off - step] = new_p0;
                    plane[off - 2 * step] = new_p1;
                    plane[off] = new_q0;
                    plane[off + step] = new_q1;
                }
            } else {
                let p3 = plane[off - 4 * step] as i32;
                let q3 = plane[off + 3 * step] as i32;
                if let Some((new_p0, new_p1, new_p2, new_q0, new_q1, new_q2)) =
                    filter_strong_luma(p0, p1, p2, p3, q0, q1, q2, q3, alpha, beta)
                {
                    trace!(
                        mb_x,
                        mb_y, is_vertical, edge, i, d, new_p0, new_q0, "DEBLOCK_PIXEL_OUT"
                    );
                    plane[off - step] = new_p0;
                    plane[off - 2 * step] = new_p1;
                    plane[off - 3 * step] = new_p2;
                    plane[off] = new_q0;
                    plane[off + step] = new_q1;
                    plane[off + 2 * step] = new_q2;
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
/// `mb_base_offset`: pre-computed byte offset of the MB's top-left chroma pixel.
/// `stride`: row stride (doubled for field-mode MBs in MBAFF).
#[allow(clippy::too_many_arguments)] // edge filtering requires plane, stride, position, bS, QP, and offsets
fn filter_mb_edge_chroma(
    is_vertical: bool,
    plane: &mut [u8],
    stride: usize,
    mb_base_offset: usize,
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

    let edge_pixel_offset = if is_vertical {
        edge * 4
    } else {
        edge * 4 * stride
    };
    let base = mb_base_offset + edge_pixel_offset;
    let step = if is_vertical { 1 } else { stride };

    for i in 0..4u8 {
        let cur_bs = bs[i as usize];
        if cur_bs == 0 {
            continue;
        }

        for d in 0..2usize {
            // Walk along the edge: for vertical, y varies; for horizontal, x varies.
            let off = if is_vertical {
                base + (i as usize * 2 + d) * stride
            } else {
                base + i as usize * 2 + d
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

/// Get the effective NNZ for deblocking at a 4x4 block position.
///
/// For CAVLC 8x8 DCT MBs, individual sub-block NNZ values are not meaningful
/// for deblocking. FFmpeg replaces them with CBP-derived values: if the
/// containing 8x8 block was coded, ALL sub-blocks are treated as having
/// non-zero coefficients.
///
/// Reference: FFmpeg h264_slice.c:2396-2432 (fill_filter_caches).
#[inline]
fn deblock_nnz(info: &MbDeblockInfo, block_idx: usize) -> u8 {
    if info.transform_8x8 && !info.is_cabac {
        // CAVLC 8x8 DCT: use CBP-based NNZ (whether the 8x8 block has non-zero
        // coefficients). FFmpeg gates this on `!CABAC` (h264_slice.c:2396).
        // CABAC uses raw per-4x4 NNZ which is already broadcast for 8x8 blocks.
        let bx = block_idx % 4;
        let by = block_idx / 4;
        let i8x8 = (by / 2) * 2 + (bx / 2);
        if info.cbp & (1 << i8x8) != 0 { 1 } else { 0 }
    } else {
        info.non_zero_count[block_idx]
    }
}

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

    // FFmpeg h264_loopfilter.c:723: mvy_limit = IS_INTERLACED(mb_type) ? 2 : 4
    let mvy_limit: u16 = if cur.mb_field { 2 } else { 4 };

    // For field MBs in MBAFF, the above neighbor is 2*mb_width away (same field
    // of the pair above), not 1*mb_width (other field of same pair).
    // FFmpeg: top_xy = mb_xy - (mb_stride << MB_FIELD(sl))
    let above_stride = if cur.mb_field {
        2 * mb_width as usize
    } else {
        mb_width as usize
    };

    // When 8x8 transform: skip internal 4-pixel edges (edges 1 and 3).
    // Only edge 0 (MB boundary) and edge 2 (8-pixel boundary) are filtered.
    let cur_t8x8 = cur.transform_8x8;

    for (edge, bs_edge) in bs.iter_mut().enumerate() {
        // Skip internal 4-pixel edges for 8x8 transform MBs
        if cur_t8x8 && (edge == 1 || edge == 3) {
            continue;
        }
        let is_mb_edge = edge == 0;

        // For edge 0, the P block is in the neighboring macroblock.
        // Skip if there is no such neighbor.
        let has_above = if cur.mb_field { mb_y >= 2 } else { mb_y > 0 };
        if is_mb_edge && (if is_vertical { mb_x == 0 } else { !has_above }) {
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
            let q_nnz = deblock_nnz(cur, q_idx);
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
                        deblock_nnz(left, p_idx),
                        left.ref_poc[p_idx],
                        left.mv[p_idx],
                        left.ref_poc_l1[p_idx],
                        left.mv_l1[p_idx],
                    )
                } else {
                    // P is in the above macroblock, bottom row.
                    // For field MBs: above = same field of pair above (2*mb_width away).
                    let above = &mb_info[mb_idx - above_stride];
                    let p_idx = luma_block_idx(q_bx, 3);
                    (
                        above.is_intra,
                        deblock_nnz(above, p_idx),
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
                    deblock_nnz(cur, p_idx),
                    cur.ref_poc[p_idx],
                    cur.mv[p_idx],
                    cur.ref_poc_l1[p_idx],
                    cur.mv_l1[p_idx],
                )
            };

            // FFmpeg h264_loopfilter.c:547-552: intra MB boundary bS=3 when
            // IS_INTERLACED(mb_type|mbm_type) AND NOT (FRAME_MBAFF && dir==0).
            // In MBAFF: vertical (dir=0) edges always get bS=4; horizontal (dir=1)
            // edges get bS=3 when either MB is interlaced.
            let is_interlaced_edge = if is_mb_edge && !is_vertical {
                let neighbor_field = mb_info[mb_idx - above_stride].mb_field;
                cur.mb_field || neighbor_field
            } else {
                false
            };

            // FFmpeg h264_loopfilter.c:557-559: horizontal inter MB boundary
            // with mixed interlace modes → bS=1, skip MV check.
            if is_mb_edge && !is_vertical && !p_intra && !q_intra {
                let neighbor_field = mb_info[mb_idx - above_stride].mb_field;
                if cur.mb_field != neighbor_field {
                    *bs_val = 1;
                    continue;
                }
            }

            *bs_val = compute_bs(
                is_mb_edge,
                p_intra,
                q_intra,
                p_nnz,
                q_nnz,
                p_ref,
                q_ref,
                p_mv,
                q_mv,
                p_ref_l1,
                q_ref_l1,
                p_mv_l1,
                q_mv_l1,
                list_count,
                mvy_limit,
                is_interlaced_edge,
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

// ---------------------------------------------------------------------------
// MBAFF mixed-interlace first vertical edge filter
// ---------------------------------------------------------------------------

/// Filter 8 pixels on the first vertical edge when adjacent pairs have different
/// interlace modes (one field, one frame). This is the MBAFF special case from
/// FFmpeg h264_loopfilter.c:730-832.
///
/// `pix_offset`: byte offset of the leftmost Q pixel (column 0 of current MB).
/// `stride`: row stride for the pixel walk.
/// `bs`: 4 boundary strengths for this half-edge.
/// `inner_iters`: 2 for luma, 1 for chroma.
#[allow(clippy::too_many_arguments)]
fn filter_mbaff_edge_luma(
    plane: &mut [u8],
    pix_offset: usize,
    stride: usize,
    bs: &[u8],
    qp: u8,
    alpha_offset: i32,
    beta_offset: i32,
) {
    let (alpha, beta) = get_thresholds(qp, alpha_offset, beta_offset);
    if alpha == 0 || beta == 0 {
        return;
    }

    // MBAFF luma: 4 bS groups × 2 pixels each = 8 pixels
    let mut off = pix_offset;
    for &cur_bs in bs.iter() {
        for _d in 0..2usize {
            if cur_bs == 0 {
                off += stride;
                continue;
            }
            let p0 = plane[off - 1] as i32;
            let p1 = plane[off - 2] as i32;
            let p2 = plane[off - 3] as i32;
            let q0 = plane[off] as i32;
            let q1 = plane[off + 1] as i32;
            let q2 = plane[off + 2] as i32;

            if cur_bs < 4 {
                let tc0 = get_tc0(qp, alpha_offset, cur_bs);
                if let Some((new_p0, new_p1, new_q0, new_q1)) =
                    filter_normal_luma(p0, p1, p2, q0, q1, q2, alpha, beta, tc0)
                {
                    plane[off - 1] = new_p0;
                    plane[off - 2] = new_p1;
                    plane[off] = new_q0;
                    plane[off + 1] = new_q1;
                }
            } else {
                let p3 = plane[off - 4] as i32;
                let q3 = plane[off + 3] as i32;
                if let Some((new_p0, new_p1, new_p2, new_q0, new_q1, new_q2)) =
                    filter_strong_luma(p0, p1, p2, p3, q0, q1, q2, q3, alpha, beta)
                {
                    plane[off - 1] = new_p0;
                    plane[off - 2] = new_p1;
                    plane[off - 3] = new_p2;
                    plane[off] = new_q0;
                    plane[off + 1] = new_q1;
                    plane[off + 2] = new_q2;
                }
            }
            off += stride;
        }
    }
}

/// MBAFF chroma edge filter (1 pixel per bS group × 4 groups = 4 pixels per call).
#[allow(clippy::too_many_arguments)]
fn filter_mbaff_edge_chroma(
    plane: &mut [u8],
    pix_offset: usize,
    stride: usize,
    bs: &[u8],
    qp: u8,
    alpha_offset: i32,
    beta_offset: i32,
) {
    let (alpha, beta) = get_thresholds(qp, alpha_offset, beta_offset);
    if alpha == 0 || beta == 0 {
        return;
    }

    // MBAFF chroma: 4 bS groups × 1 pixel each = 4 pixels
    let mut off = pix_offset;
    for &cur_bs in bs.iter() {
        if cur_bs == 0 {
            off += stride;
            continue;
        }
        let p0 = plane[off - 1] as i32;
        let p1 = plane[off - 2] as i32;
        let q0 = plane[off] as i32;
        let q1 = plane[off + 1] as i32;

        if cur_bs < 4 {
            let tc = get_tc0(qp, alpha_offset, cur_bs) + 1;
            if let Some((new_p0, new_q0)) = filter_normal_chroma(p0, p1, q0, q1, alpha, beta, tc) {
                plane[off - 1] = new_p0;
                plane[off] = new_q0;
            }
        } else if let Some((new_p0, new_q0)) = filter_strong_chroma(p0, p1, q0, q1, alpha, beta) {
            plane[off - 1] = new_p0;
            plane[off] = new_q0;
        }
        off += stride;
    }
}

/// Compute 8 bS values for the mixed-interlace first vertical edge.
///
/// Returns `[bS0..bS7]` — 8 boundary strengths mapping 8 pixel-pair positions
/// along the 16-pixel edge. The mapping depends on field/frame mode of current MB.
///
/// FFmpeg ref: h264_loopfilter.c:746-774
fn compute_mbaff_vert_bs(
    mb_info: &[MbDeblockInfo],
    mb_idx: usize,
    mb_y: u32,
    mb_width: u32,
    cur_mb_field: bool,
) -> [u8; 8] {
    let cur = &mb_info[mb_idx];
    let mut bs = [0u8; 8];

    if cur.is_intra {
        // All intra: bS = 4 for all 8 positions
        bs = [4; 8];
        return bs;
    }

    // Offset table: maps pixel position i to the neighbor sub-block row.
    // offset[MB_FIELD][mb_y&1][i] gives the left neighbor's NNZ block index (row within MB).
    // FFmpeg h264_loopfilter.c:750-757
    //
    // When current is field (MB_FIELD=1):
    //   row 0: offsets into left_top  (rows 0,1,2,3,0,1,2,3)
    //   row 1: offsets into left_top  (rows 0,1,2,3,0,1,2,3)
    // When current is frame (MB_FIELD=0):
    //   row 0: offsets into left_top  (rows 0,0,0,0,1,1,1,1)
    //   row 1: offsets into left_top  (rows 2,2,2,2,3,3,3,3)

    // The neighbor 4x4 block row for each of the 8 bS positions
    let neighbor_block_row: [usize; 8] = if cur_mb_field {
        [0, 1, 2, 3, 0, 1, 2, 3]
    } else if mb_y & 1 == 0 {
        [0, 0, 0, 0, 1, 1, 1, 1]
    } else {
        [2, 2, 2, 2, 3, 3, 3, 3]
    };

    // j selects which neighbor MB (top=0 or bottom=1 of the left pair)
    // FFmpeg: j = MB_FIELD ? i>>2 : i&1
    for i in 0..8usize {
        let j = if cur_mb_field { i >> 2 } else { i & 1 };

        // Left neighbor MB: top (j=0) or bottom (j=1) of the left pair
        let left_mb_idx = if j == 0 {
            // Top of left pair: (mb_y & !1) * mb_width + (mb_x - 1)
            ((mb_y & !1) * mb_width + (mb_idx as u32 % mb_width) - 1) as usize
        } else {
            // Bottom of left pair: (mb_y | 1) * mb_width + (mb_x - 1)
            ((mb_y | 1) * mb_width + (mb_idx as u32 % mb_width) - 1) as usize
        };

        let left = &mb_info[left_mb_idx];

        if left.is_intra {
            bs[i] = 4;
            continue;
        }

        // Current side NNZ: leftmost column (block_x=0), row = i>>1 in 4x4 units
        let cur_block_row = i >> 1;
        let cur_nnz = deblock_nnz(cur, luma_block_idx(0, cur_block_row));

        // Neighbor side NNZ: rightmost column (block_x=3), row from offset table
        let nbr_row = neighbor_block_row[i];
        let nbr_nnz = deblock_nnz(left, luma_block_idx(3, nbr_row));

        if cur_nnz != 0 || nbr_nnz != 0 {
            bs[i] = 2;
        } else {
            // MV check — use NNZ-based bS=1 minimum, skip MV check for simplicity
            // The MV check is complex for mixed-mode and rarely matters vs NNZ.
            // Use bS=1 conservatively for non-zero MV difference.
            let cur_blk = luma_block_idx(0, cur_block_row);
            let nbr_blk = luma_block_idx(3, nbr_row);
            let mvy_limit: u16 = if cur.mb_field { 2 } else { 4 };
            if check_mv(
                cur.ref_poc[cur_blk],
                left.ref_poc[nbr_blk],
                cur.mv[cur_blk],
                left.mv[nbr_blk],
                cur.ref_poc_l1[cur_blk],
                left.ref_poc_l1[nbr_blk],
                cur.mv_l1[cur_blk],
                left.mv_l1[nbr_blk],
                cur.list_count,
                mvy_limit,
            ) {
                bs[i] = 1;
            }
        }
    }

    bs
}

/// Apply the MBAFF mixed-interlace first vertical edge filter for a macroblock.
///
/// When the current pair and left pair have different interlace modes,
/// the first vertical edge uses 8 bS values, 2 QPs, and the MBAFF filter
/// (2 pixels per bS for luma, 1 pixel per bS for chroma).
///
/// Returns true if the edge was handled (caller should skip normal edge 0).
#[allow(clippy::too_many_arguments)]
fn deblock_mbaff_first_vert_edge(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    alpha_c0_offset: i32,
    beta_offset: i32,
    chroma_qp_index_offset: [i32; 2],
) -> bool {
    let mb_idx = (mb_y * mb_width + mb_x) as usize;
    let cur = &mb_info[mb_idx];

    // Only applies when current and left pairs have different interlace modes
    if mb_x == 0 {
        return false;
    }
    let left_top_idx = ((mb_y & !1) * mb_width + mb_x - 1) as usize;
    let left_bot_idx = ((mb_y | 1) * mb_width + mb_x - 1) as usize;

    // Check if left pair has a different interlace mode
    // (use top-of-left-pair's field flag as representative of the pair)
    let left_field = mb_info[left_top_idx].mb_field;
    if !(cur.mb_field ^ left_field) {
        return false; // Same interlace mode, no special handling needed
    }

    // Compute 8 bS values
    let bs8 = compute_mbaff_vert_bs(mb_info, mb_idx, mb_y, mb_width, cur.mb_field);

    // Compute 2 QP values: one for each half (top and bottom of left pair)
    let mb_qp = cur.qp;
    let left_top_qp = mb_info[left_top_idx].qp;
    let left_bot_qp = mb_info[left_bot_idx].qp;
    let qp0 = avg_qp(mb_qp, left_top_qp);
    let qp1 = avg_qp(mb_qp, left_bot_qp);

    // Compute the base pixel offset for this MB's top-left in the frame buffer.
    // For MBAFF, img_y points to the MB's first pixel including field offset.
    let frame_y_stride = pic.y_stride;
    let frame_uv_stride = pic.uv_stride;

    // The pixel base is always at the pair's position using the current MB's
    // field-aware addressing.
    let (luma_base, luma_stride) = deblock_luma_offset(frame_y_stride, mb_x, mb_y, cur.mb_field);
    let (chroma_base_u, chroma_stride) =
        deblock_chroma_offset(frame_uv_stride, mb_x, mb_y, cur.mb_field);
    let (chroma_base_v, _) = deblock_chroma_offset(frame_uv_stride, mb_x, mb_y, cur.mb_field);

    // --- Luma ---
    // FFmpeg h264_loopfilter.c:794-831
    // Field mode: bsi=1, two 8-row strips → bS[0..3] and bS[4..7]
    // Frame mode: bsi=2, even/odd rows → bS[0,2,4,6] and bS[1,3,5,7]
    let (bs_half0, bs_half1): ([u8; 4], [u8; 4]) = if cur.mb_field {
        (
            [bs8[0], bs8[1], bs8[2], bs8[3]],
            [bs8[4], bs8[5], bs8[6], bs8[7]],
        )
    } else {
        (
            [bs8[0], bs8[2], bs8[4], bs8[6]],
            [bs8[1], bs8[3], bs8[5], bs8[7]],
        )
    };

    if cur.mb_field {
        // Current is field: two 8-row strips with field stride.
        filter_mbaff_edge_luma(
            &mut pic.y,
            luma_base,
            luma_stride,
            &bs_half0,
            qp0,
            alpha_c0_offset,
            beta_offset,
        );
        filter_mbaff_edge_luma(
            &mut pic.y,
            luma_base + 8 * luma_stride,
            luma_stride,
            &bs_half1,
            qp1,
            alpha_c0_offset,
            beta_offset,
        );
    } else {
        // Current is frame: even/odd rows interleaved with 2x stride.
        let double_stride = 2 * luma_stride;
        filter_mbaff_edge_luma(
            &mut pic.y,
            luma_base,
            double_stride,
            &bs_half0,
            qp0,
            alpha_c0_offset,
            beta_offset,
        );
        filter_mbaff_edge_luma(
            &mut pic.y,
            luma_base + luma_stride,
            double_stride,
            &bs_half1,
            qp1,
            alpha_c0_offset,
            beta_offset,
        );
    }

    // --- Chroma (4:2:0) ---
    for (comp, &offset) in chroma_qp_index_offset.iter().enumerate() {
        let chroma_qp_cur = chroma_qp(mb_qp, offset);
        let cqp0 = avg_qp(chroma_qp_cur, chroma_qp(left_top_qp, offset));
        let cqp1 = avg_qp(chroma_qp_cur, chroma_qp(left_bot_qp, offset));

        let (plane, base) = if comp == 0 {
            (&mut pic.u, chroma_base_u)
        } else {
            (&mut pic.v, chroma_base_v)
        };

        if cur.mb_field {
            filter_mbaff_edge_chroma(
                plane,
                base,
                chroma_stride,
                &bs_half0,
                cqp0,
                alpha_c0_offset,
                beta_offset,
            );
            filter_mbaff_edge_chroma(
                plane,
                base + 4 * chroma_stride,
                chroma_stride,
                &bs_half1,
                cqp1,
                alpha_c0_offset,
                beta_offset,
            );
        } else {
            let double_stride = 2 * chroma_stride;
            filter_mbaff_edge_chroma(
                plane,
                base,
                double_stride,
                &bs_half0,
                cqp0,
                alpha_c0_offset,
                beta_offset,
            );
            filter_mbaff_edge_chroma(
                plane,
                base + chroma_stride,
                double_stride,
                &bs_half1,
                cqp1,
                alpha_c0_offset,
                beta_offset,
            );
        }
    }

    true
}

/// Handle the MBAFF mixed-mode horizontal edge 0 special case.
///
/// Applies when: MBAFF, horizontal edge (dir=1), top of pair (mb_y & 1 == 0),
/// and the pair above is interlaced while the current is NOT interlaced.
/// FFmpeg h264_loopfilter.c:494-542.
///
/// The filter runs twice (once per field) with doubled stride, using separate
/// bS and QP for each field pass.
///
/// Returns true if the edge was handled.
#[allow(clippy::too_many_arguments)]
fn deblock_mbaff_horiz_edge0(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    alpha_c0_offset: i32,
    beta_offset: i32,
    chroma_qp_index_offset: [i32; 2],
) -> bool {
    // Only applies at top of pair when above pair is field and current is frame
    if mb_y & 1 != 0 || mb_y < 2 {
        return false;
    }
    let mb_idx = (mb_y * mb_width + mb_x) as usize;
    let cur = &mb_info[mb_idx];
    if cur.mb_field {
        return false; // Current must be frame-mode for this special case
    }

    // Above pair: top at mb_y-2, bottom at mb_y-1
    let above_top_idx = ((mb_y - 2) * mb_width + mb_x) as usize;
    let above_bot_idx = ((mb_y - 1) * mb_width + mb_x) as usize;

    // The above pair must be interlaced (field-mode)
    if !mb_info[above_top_idx].mb_field {
        return false;
    }

    let frame_y_stride = pic.y_stride;
    let frame_uv_stride = pic.uv_stride;
    let (luma_base, luma_stride) = deblock_luma_offset(frame_y_stride, mb_x, mb_y, cur.mb_field);
    let (chroma_base_u, chroma_stride) =
        deblock_chroma_offset(frame_uv_stride, mb_x, mb_y, cur.mb_field);
    let (chroma_base_v, _) = deblock_chroma_offset(frame_uv_stride, mb_x, mb_y, cur.mb_field);

    let double_luma_stride = 2 * luma_stride;
    let double_chroma_stride = 2 * chroma_stride;

    // Two passes: j=0 for even field (top of above pair), j=1 for odd field (bottom)
    for j in 0..2usize {
        let above_idx = if j == 0 { above_top_idx } else { above_bot_idx };
        let above = &mb_info[above_idx];

        // Compute bS for this field pass
        let mut bs = [0u8; 4];
        if cur.is_intra || above.is_intra {
            bs = [3; 4]; // bS=3 for interlaced intra edges
        } else {
            for (i, bs_val) in bs.iter_mut().enumerate() {
                // Current: top row, block column i
                let cur_nnz = deblock_nnz(cur, luma_block_idx(i, 0));
                // Above: bottom row (row 3), block column i
                let above_nnz = deblock_nnz(above, luma_block_idx(i, 3));
                if cur_nnz != 0 || above_nnz != 0 {
                    *bs_val = 2;
                } else {
                    let cur_blk = luma_block_idx(i, 0);
                    let above_blk = luma_block_idx(i, 3);
                    let mvy_limit: u16 = if cur.mb_field { 2 } else { 4 };
                    if check_mv(
                        cur.ref_poc[cur_blk],
                        above.ref_poc[above_blk],
                        cur.mv[cur_blk],
                        above.mv[above_blk],
                        cur.ref_poc_l1[cur_blk],
                        above.ref_poc_l1[above_blk],
                        cur.mv_l1[cur_blk],
                        above.mv_l1[above_blk],
                        cur.list_count,
                        mvy_limit,
                    ) {
                        *bs_val = 1;
                    }
                }
            }
        }

        // QP: average of current and the above-field MB
        let qp = avg_qp(cur.qp, above.qp);

        // Luma: filter at j*single_stride offset, using doubled stride
        filter_mb_edge_luma(
            false, // horizontal
            &mut pic.y,
            double_luma_stride,
            luma_base + j * luma_stride,
            0,
            bs,
            qp,
            alpha_c0_offset,
            beta_offset,
            mb_x,
            mb_y,
        );

        // Chroma
        for (comp, &offset) in chroma_qp_index_offset.iter().enumerate() {
            let cqp = avg_qp(chroma_qp(cur.qp, offset), chroma_qp(above.qp, offset));
            let (plane, base) = if comp == 0 {
                (&mut pic.u, chroma_base_u)
            } else {
                (&mut pic.v, chroma_base_v)
            };
            filter_mb_edge_chroma(
                false, // horizontal
                plane,
                double_chroma_stride,
                base + j * chroma_stride,
                0,
                bs,
                cqp,
                alpha_c0_offset,
                beta_offset,
            );
        }
    }

    true
}

/// Compute the byte offset and stride for a macroblock's luma plane.
///
/// For field-mode MBs (MBAFF), the stride is doubled and the bottom field
/// (mb_y & 1 == 1) starts one frame-row below the pair's top-left corner.
/// This matches FFmpeg h264_slice.c:2473-2484.
#[inline]
fn deblock_luma_offset(y_stride: usize, mb_x: u32, mb_y: u32, mb_field: bool) -> (usize, usize) {
    let x = mb_x as usize * 16;
    if mb_field {
        let pair_base_y = (mb_y & !1) as usize * 16;
        let field_offset = if mb_y & 1 == 1 { y_stride } else { 0 };
        (pair_base_y * y_stride + field_offset + x, y_stride * 2)
    } else {
        (mb_y as usize * 16 * y_stride + x, y_stride)
    }
}

/// Compute the byte offset and stride for a macroblock's chroma plane.
#[inline]
fn deblock_chroma_offset(uv_stride: usize, mb_x: u32, mb_y: u32, mb_field: bool) -> (usize, usize) {
    let x = mb_x as usize * 8;
    if mb_field {
        let pair_base_y = (mb_y & !1) as usize * 8;
        let field_offset = if mb_y & 1 == 1 { uv_stride } else { 0 };
        (pair_base_y * uv_stride + field_offset + x, uv_stride * 2)
    } else {
        (mb_y as usize * 8 * uv_stride + x, uv_stride)
    }
}

/// Deblock a single macroblock (all luma and chroma edges).
fn deblock_mb(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    mb_x: u32,
    mb_y: u32,
    slice_table: &[u16],
    slice_params: &[SliceDeblockParams],
) {
    let mb_width = pic.mb_width;
    let mb_idx = (mb_y * mb_width + mb_x) as usize;
    let cur_qp = mb_info[mb_idx].qp;
    let cur_mb_field = mb_info[mb_idx].mb_field;

    // Look up this MB's slice deblock params
    let slice_id = slice_table[mb_idx] as usize;
    let params = &slice_params[slice_id.min(slice_params.len() - 1)];
    let alpha_c0_offset = params.alpha_c0_offset;
    let beta_offset = params.beta_offset;
    let chroma_qp_index_offset = params.chroma_qp_index_offset;

    // Compute field-aware pixel offsets and strides for this MB.
    trace!(
        mb_x,
        mb_y,
        cur_mb_field,
        cur_qp,
        y_stride = pic.y_stride,
        "DEBLOCK_MB_START"
    );
    let (luma_base, luma_stride) = deblock_luma_offset(pic.y_stride, mb_x, mb_y, cur_mb_field);
    let (chroma_base_u, chroma_stride) =
        deblock_chroma_offset(pic.uv_stride, mb_x, mb_y, cur_mb_field);
    let (chroma_base_v, _) = deblock_chroma_offset(pic.uv_stride, mb_x, mb_y, cur_mb_field);

    // For field MBs in MBAFF, the above neighbor is 2*mb_width away (same field
    // of pair above). For frame MBs, it's 1*mb_width (standard).
    let above_stride = if cur_mb_field {
        2 * mb_width as usize
    } else {
        mb_width as usize
    };
    let has_above = if cur_mb_field { mb_y >= 2 } else { mb_y > 0 };

    // For idc == 2, determine which neighbors are in a different slice.
    // Neighbors in different slices are treated as unavailable (bS=0 for that edge).
    let cur_slice = slice_table[mb_idx];
    let left_available = if params.disable_deblocking_filter_idc == 2 && mb_x > 0 {
        slice_table[mb_idx - 1] == cur_slice
    } else {
        mb_x > 0
    };
    let top_available = if params.disable_deblocking_filter_idc == 2 && has_above {
        slice_table[mb_idx - above_stride] == cur_slice
    } else {
        has_above
    };

    // Process vertical (is_vertical=true) then horizontal (is_vertical=false) edges.
    // MBAFF special handlers run at the start of each direction pass, matching FFmpeg's
    // filter_mb_dir(dir=0) then filter_mb_dir(dir=1) structure. The MBAFF vert edge 0
    // must run before any vert edges (so vert edge 1+ see its output), and the MBAFF
    // horiz edge 0 must run after all vert edges (so it sees vert-filtered pixels).
    for is_vertical in [true, false] {
        // MBAFF mixed-interlace edge 0: handled before normal edges in each direction.
        let first_edge_done = if is_vertical {
            left_available
                && deblock_mbaff_first_vert_edge(
                    pic,
                    mb_info,
                    mb_x,
                    mb_y,
                    mb_width,
                    alpha_c0_offset,
                    beta_offset,
                    chroma_qp_index_offset,
                )
        } else {
            top_available
                && deblock_mbaff_horiz_edge0(
                    pic,
                    mb_info,
                    mb_x,
                    mb_y,
                    mb_width,
                    alpha_c0_offset,
                    beta_offset,
                    chroma_qp_index_offset,
                )
        };

        // For idc==2, skip the MB boundary edge if the neighbor is in a different slice
        let skip_mb_edge = if is_vertical {
            !left_available
        } else {
            !top_available
        };
        let luma_bs = compute_luma_bs(is_vertical, mb_info, mb_x, mb_y, mb_width);

        trace!(
            mb_x, mb_y, is_vertical,
            bs0 = ?luma_bs[0], bs1 = ?luma_bs[1], bs2 = ?luma_bs[2], bs3 = ?luma_bs[3],
            cur_qp,
            alpha_c0_offset, beta_offset,
            "DEBLOCK_EDGE"
        );

        // --- Luma edges ---
        for (edge, &bs_edge) in luma_bs.iter().enumerate() {
            if bs_edge == [0, 0, 0, 0] {
                continue;
            }
            // Skip MB boundary edge if cross-slice (idc==2) or already handled by MBAFF
            if edge == 0 && skip_mb_edge {
                continue;
            }
            if edge == 0 && first_edge_done {
                continue;
            }
            let qp = if edge == 0 {
                let neighbor_qp = if is_vertical && mb_x > 0 {
                    Some(mb_info[mb_idx - 1].qp)
                } else if !is_vertical && has_above {
                    Some(mb_info[mb_idx - above_stride].qp)
                } else {
                    None
                };
                neighbor_qp.map_or(cur_qp, |nq| avg_qp(cur_qp, nq))
            } else {
                cur_qp
            };
            filter_mb_edge_luma(
                is_vertical,
                &mut pic.y,
                luma_stride,
                luma_base,
                edge,
                bs_edge,
                qp,
                alpha_c0_offset,
                beta_offset,
                mb_x,
                mb_y,
            );
        }

        // --- Chroma edges (4:2:0: 2 edges per direction) ---
        // Cb and Cr use separate chroma_qp_index_offsets (PPS [0] and [1]).
        let chroma_bs = derive_chroma_bs(&luma_bs);
        for (edge, &bs_edge) in chroma_bs.iter().enumerate() {
            if bs_edge == [0, 0, 0, 0] {
                continue;
            }
            // Skip MB boundary edge if cross-slice (idc==2) or already handled by MBAFF
            if edge == 0 && skip_mb_edge {
                continue;
            }
            if edge == 0 && first_edge_done {
                continue;
            }

            for (comp, &offset) in chroma_qp_index_offset.iter().enumerate() {
                let chroma_qp_cur = chroma_qp(cur_qp, offset);
                let qp = if edge == 0 {
                    let neighbor_qp = if is_vertical && mb_x > 0 {
                        Some(chroma_qp(mb_info[mb_idx - 1].qp, offset))
                    } else if !is_vertical && has_above {
                        Some(chroma_qp(mb_info[mb_idx - above_stride].qp, offset))
                    } else {
                        None
                    };
                    neighbor_qp.map_or(chroma_qp_cur, |nq| avg_qp(chroma_qp_cur, nq))
                } else {
                    chroma_qp_cur
                };

                let (plane, base) = if comp == 0 {
                    (&mut pic.u, chroma_base_u)
                } else {
                    (&mut pic.v, chroma_base_v)
                };
                filter_mb_edge_chroma(
                    is_vertical,
                    plane,
                    chroma_stride,
                    base,
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
/// Uses per-slice deblocking parameters: each MB is filtered with the
/// alpha/beta offsets and idc from its own slice header.
///
/// # Arguments
///
/// * `pic` - the decoded picture buffer (modified in-place)
/// * `mb_info` - per-macroblock deblocking info (mb_width * mb_height entries, raster order)
/// * `slice_table` - per-macroblock slice number (same layout as mb_info)
/// * `slice_params` - deblocking parameters for each slice
#[tracing::instrument(skip_all, fields(mb_width = pic.mb_width, mb_height = pic.mb_height))]
pub fn deblock_frame(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    slice_table: &[u16],
    slice_params: &[SliceDeblockParams],
    is_mbaff: bool,
) {
    let mb_width = pic.mb_width;
    let mb_height = pic.mb_height;

    debug!(mb_width, mb_height, is_mbaff, "deblocking frame");

    debug_assert_eq!(
        mb_info.len(),
        (mb_width * mb_height) as usize,
        "mb_info length must equal mb_width * mb_height"
    );

    // MBAFF uses column-first pair iteration — complex ordering, keep sequential.
    // Tiny frames (<=2 rows) have no parallelism benefit.
    if is_mbaff || mb_height <= 2 {
        deblock_frame_sequential(pic, mb_info, slice_table, slice_params, is_mbaff);
        return;
    }

    // Progressive parallel deblocking: row-wavefront with work-stealing.
    //
    // SAFETY argument: Each MB row Y writes to luma pixel rows [Y*16-3 .. Y*16+15]
    // and chroma [Y*8-1 .. Y*8+7]. The wavefront guarantee (row Y-1 complete before
    // row Y starts) ensures no concurrent writes to overlapping regions.
    // The Acquire/Release ordering ensures pixel writes from row Y-1 are visible to
    // row Y's thread.

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get().min(mb_height as usize / 2).max(1))
        .unwrap_or(1);

    if num_threads <= 1 {
        deblock_frame_sequential(pic, mb_info, slice_table, slice_params, false);
        return;
    }

    let row_progress: Vec<AtomicU32> = (0..mb_height).map(|_| AtomicU32::new(0)).collect();
    let next_row = AtomicU32::new(0);

    // SAFETY: thread::scope joins all threads before returning, and the
    // wavefront ensures disjoint row access. See SyncPic doc comment.
    let sync_pic = SyncPic(pic as *mut PictureBuffer);

    std::thread::scope(|s| {
        for _ in 0..num_threads {
            let sync_pic = &sync_pic;
            let row_progress = &row_progress;
            let next_row = &next_row;
            s.spawn(move || {
                loop {
                    let my_row = next_row.fetch_add(1, Ordering::Relaxed);
                    if my_row >= mb_height {
                        break;
                    }

                    // Wait for dependency: row my_row - 1 must be done
                    if my_row > 0 {
                        while row_progress[(my_row - 1) as usize].load(Ordering::Acquire) == 0 {
                            std::hint::spin_loop();
                        }
                    }

                    // SAFETY: Row my_row writes to pixels disjoint from all other active rows.
                    // Row my_row-1 is complete (checked above), so its pixel writes are visible.
                    let pic = unsafe { &mut *sync_pic.0 };
                    for mb_x in 0..mb_width {
                        deblock_mb_if_enabled(
                            pic,
                            mb_info,
                            mb_x,
                            my_row,
                            mb_width,
                            slice_table,
                            slice_params,
                        );
                    }

                    row_progress[my_row as usize].store(1, Ordering::Release);
                }
            });
        }
    });
}

/// Deblock a single MB row (progressive, non-MBAFF).
///
/// Calls `deblock_mb_if_enabled` for each MB in row `mb_y`.
/// Used by inline deblock in the decode loop (per-row deblock with 1-row delay).
pub fn deblock_row(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    slice_table: &[u16],
    slice_params: &[SliceDeblockParams],
    mb_y: u32,
    mb_width: u32,
) {
    for mb_x in 0..mb_width {
        deblock_mb_if_enabled(
            pic,
            mb_info,
            mb_x,
            mb_y,
            mb_width,
            slice_table,
            slice_params,
        );
    }
}

/// Deblock a single MBAFF pair row (both top and bottom MB of the pair).
///
/// MBAFF uses column-first pair iteration: for each column, process top then
/// bottom MB. `pair_row` is the pair index (0-based), so the MB rows are
/// `pair_row * 2` and `pair_row * 2 + 1`.
pub fn deblock_row_mbaff(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    slice_table: &[u16],
    slice_params: &[SliceDeblockParams],
    pair_row: u32,
    mb_width: u32,
) {
    for mb_x in 0..mb_width {
        for sub in 0..2u32 {
            let mb_y = pair_row * 2 + sub;
            deblock_mb_if_enabled(
                pic,
                mb_info,
                mb_x,
                mb_y,
                mb_width,
                slice_table,
                slice_params,
            );
        }
    }
}

/// Sequential deblocking fallback (MBAFF or single-threaded).
fn deblock_frame_sequential(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    slice_table: &[u16],
    slice_params: &[SliceDeblockParams],
    is_mbaff: bool,
) {
    let mb_width = pic.mb_width;
    let mb_height = pic.mb_height;

    if is_mbaff {
        for pair_row in 0..(mb_height / 2) {
            for mb_x in 0..mb_width {
                for sub in 0..2u32 {
                    let mb_y = pair_row * 2 + sub;
                    deblock_mb_if_enabled(
                        pic,
                        mb_info,
                        mb_x,
                        mb_y,
                        mb_width,
                        slice_table,
                        slice_params,
                    );
                }
            }
        }
    } else {
        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                deblock_mb_if_enabled(
                    pic,
                    mb_info,
                    mb_x,
                    mb_y,
                    mb_width,
                    slice_table,
                    slice_params,
                );
            }
        }
    }
}

/// Check idc and deblock a single MB, with tracing.
fn deblock_mb_if_enabled(
    pic: &mut PictureBuffer,
    mb_info: &[MbDeblockInfo],
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    slice_table: &[u16],
    slice_params: &[SliceDeblockParams],
) {
    let mb_idx = (mb_y * mb_width + mb_x) as usize;
    let slice_id = slice_table[mb_idx] as usize;
    let params = &slice_params[slice_id.min(slice_params.len() - 1)];

    // idc == 1 means disable filter for this slice's MBs
    if params.disable_deblocking_filter_idc == 1 {
        return;
    }

    deblock_mb(pic, mb_info, mb_x, mb_y, slice_table, slice_params);

    {
        let mb_field = mb_info[mb_idx].mb_field;
        let (y_base, y_stride) = deblock_luma_offset(pic.y_stride, mb_x, mb_y, mb_field);
        let (u_base, uv_stride) = deblock_chroma_offset(pic.uv_stride, mb_x, mb_y, mb_field);
        let (v_base, _) = deblock_chroma_offset(pic.uv_stride, mb_x, mb_y, mb_field);
        let y_sum = deblock_plane_sum(&pic.y, y_base, y_stride, 16);
        let u_sum = deblock_plane_sum(&pic.u, u_base, uv_stride, 8);
        let v_sum = deblock_plane_sum(&pic.v, v_base, uv_stride, 8);
        tracing::trace!(mb_x, mb_y, y_sum, u_sum, v_sum, "MB_DEBLOCK");
    }
}

/// Compute pixel sum for a macroblock region (for deblock tracing).
/// Uses pre-computed base offset and stride (field-aware for MBAFF).
fn deblock_plane_sum(plane: &[u8], base_offset: usize, stride: usize, size: u32) -> u32 {
    let mut sum = 0u32;
    for dy in 0..size as usize {
        let row_start = base_offset + dy * stride;
        for dx in 0..size as usize {
            sum = sum.wrapping_add(plane[row_start + dx] as u32);
        }
    }
    sum
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
            4,     // mvy_limit: frame mode default
            false, // is_interlaced_edge: non-MBAFF default
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
                4,
                false,
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
                4,
                false,
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
                4,
                false,
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

        let st = vec![0u16; mb_info.len()];
        deblock_frame(
            &mut pic,
            &mb_info,
            &st,
            &[SliceDeblockParams::default()],
            false,
        );

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
        let disabled_params = SliceDeblockParams {
            disable_deblocking_filter_idc: 1,
            ..Default::default()
        };
        let st = vec![0u16; mb_info.len()];
        deblock_frame(&mut pic, &mb_info, &st, &[disabled_params], false);
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

        let st = vec![0u16; mb_info.len()];
        deblock_frame(
            &mut pic,
            &mb_info,
            &st,
            &[SliceDeblockParams::default()],
            false,
        );

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

        let st = vec![0u16; mb_info.len()];
        deblock_frame(
            &mut pic,
            &mb_info,
            &st,
            &[SliceDeblockParams::default()],
            false,
        );

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

    /// Create a multi-MB picture with non-trivial pixel data for deblock comparison.
    /// 4x3 MBs (64x48 luma, 32x24 chroma) with gradient-like data.
    fn make_pic_4x3() -> PictureBuffer {
        let mb_w = 4u32;
        let mb_h = 3u32;
        let w = (mb_w * 16) as usize;
        let h = (mb_h * 16) as usize;
        let cw = (mb_w * 8) as usize;
        let ch = (mb_h * 8) as usize;
        let mut y = vec![128u8; w * h];
        let mut u = vec![128u8; cw * ch];
        let mut v = vec![128u8; cw * ch];
        // Create variations across MB boundaries to exercise the filter.
        for row in 0..h {
            for col in 0..w {
                y[row * w + col] = ((row * 7 + col * 13 + 50) % 256) as u8;
            }
        }
        for row in 0..ch {
            for col in 0..cw {
                u[row * cw + col] = ((row * 11 + col * 5 + 80) % 256) as u8;
                v[row * cw + col] = ((row * 3 + col * 9 + 120) % 256) as u8;
            }
        }
        PictureBuffer {
            y,
            u,
            v,
            y_stride: w,
            uv_stride: cw,
            width: w as u32,
            height: h as u32,
            mb_width: mb_w,
            mb_height: mb_h,
        }
    }

    /// Build MB info and slice table for a uniform I-slice frame (all intra, single slice).
    fn make_mb_info_intra(
        mb_w: u32,
        mb_h: u32,
    ) -> (Vec<MbDeblockInfo>, Vec<u16>, Vec<SliceDeblockParams>) {
        let total = (mb_w * mb_h) as usize;
        let mb_info: Vec<MbDeblockInfo> = (0..total)
            .map(|_| MbDeblockInfo {
                qp: 28,
                is_intra: true,
                non_zero_count: [16; 24],
                ..Default::default()
            })
            .collect();
        let slice_table = vec![0u16; total];
        let slice_params = vec![SliceDeblockParams {
            alpha_c0_offset: 0,
            beta_offset: 0,
            disable_deblocking_filter_idc: 0,
            chroma_qp_index_offset: [0, 0],
        }];
        (mb_info, slice_table, slice_params)
    }

    #[test]
    fn deblock_row_matches_deblock_frame() {
        let pic_frame = make_pic_4x3();
        let mut pic_row = pic_frame.clone();
        let mut pic_frame_mut = pic_frame;
        let mb_w = pic_row.mb_width;
        let mb_h = pic_row.mb_height;
        let (mb_info, slice_table, slice_params) = make_mb_info_intra(mb_w, mb_h);

        // Deblock whole frame
        deblock_frame(
            &mut pic_frame_mut,
            &mb_info,
            &slice_table,
            &slice_params,
            false,
        );

        // Deblock row by row
        for mb_y in 0..mb_h {
            deblock_row(
                &mut pic_row,
                &mb_info,
                &slice_table,
                &slice_params,
                mb_y,
                mb_w,
            );
        }

        assert_eq!(
            pic_row.y, pic_frame_mut.y,
            "Y plane mismatch between deblock_row and deblock_frame"
        );
        assert_eq!(pic_row.u, pic_frame_mut.u, "U plane mismatch");
        assert_eq!(pic_row.v, pic_frame_mut.v, "V plane mismatch");
    }

    #[test]
    fn deblock_row_mbaff_matches_deblock_frame() {
        // 4x4 MBs (2 pair rows) for MBAFF
        let mb_w = 4u32;
        let mb_h = 4u32;
        let w = (mb_w * 16) as usize;
        let h = (mb_h * 16) as usize;
        let cw = (mb_w * 8) as usize;
        let ch = (mb_h * 8) as usize;
        let mut y = vec![128u8; w * h];
        for row in 0..h {
            for col in 0..w {
                y[row * w + col] = ((row * 7 + col * 13 + 50) % 256) as u8;
            }
        }
        let u = vec![128u8; cw * ch];
        let v = vec![128u8; cw * ch];
        let pic = PictureBuffer {
            y,
            u,
            v,
            y_stride: w,
            uv_stride: cw,
            width: w as u32,
            height: h as u32,
            mb_width: mb_w,
            mb_height: mb_h,
        };
        let mut pic_frame = pic.clone();
        let mut pic_row = pic;
        let (mb_info, slice_table, slice_params) = make_mb_info_intra(mb_w, mb_h);

        // Deblock whole frame (MBAFF)
        deblock_frame(&mut pic_frame, &mb_info, &slice_table, &slice_params, true);

        // Deblock pair row by pair row
        for pair_row in 0..(mb_h / 2) {
            deblock_row_mbaff(
                &mut pic_row,
                &mb_info,
                &slice_table,
                &slice_params,
                pair_row,
                mb_w,
            );
        }

        assert_eq!(
            pic_row.y, pic_frame.y,
            "Y plane mismatch in MBAFF deblock_row"
        );
        assert_eq!(pic_row.u, pic_frame.u, "U plane mismatch in MBAFF");
        assert_eq!(pic_row.v, pic_frame.v, "V plane mismatch in MBAFF");
    }
}
