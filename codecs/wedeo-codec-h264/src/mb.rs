// H.264 macroblock decode orchestrator.
//
// Decodes a single macroblock by calling CAVLC, dequant, intra prediction,
// and IDCT in the correct order. Handles I_4x4, I_16x16, and I_PCM macroblocks.
//
// Reference: FFmpeg libavcodec/h264_mb.c, h264_slice.c

use std::sync::{Arc, Mutex};

use tracing::{debug, trace};
use wedeo_codec::bitstream::BitReadBE;
use wedeo_core::error::Result;

use crate::cavlc::{Macroblock, NeighborContext, decode_mb_cavlc};
use crate::deblock::{MbDeblockInfo, PictureBuffer, SliceDeblockParams};
use crate::dequant::{self, Dequant4Table, Dequant8Table};
use crate::idct;
use crate::intra_pred;
use crate::mc;
use crate::mvpred::{self, MvContext};
use crate::pps::Pps;
use crate::shared_picture::{BufferPool, PicHandle, SharedPicture};
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::tables::CHROMA_QP_TABLE;

// ---------------------------------------------------------------------------
// Per-MB pixel checksum for pipeline-stage tracing
// ---------------------------------------------------------------------------

/// Compute a quick sum of all pixel values in an MB-sized block of a plane.
/// Used for per-MB reconstruction checksums to compare against FFmpeg.
/// Uses (offset, stride) for MBAFF field-mode compatibility.
fn mb_plane_sum(plane: &[u8], offset: usize, stride: usize, size: u32) -> u32 {
    let mut sum = 0u32;
    for dy in 0..size as usize {
        let row_start = offset + dy * stride;
        for dx in 0..size as usize {
            sum = sum.wrapping_add(plane[row_start + dx] as u32);
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// Block scanning order
// ---------------------------------------------------------------------------

/// Maps 4x4 block index (0..15) to (blk_x, blk_y) within the macroblock.
/// Blocks are scanned in raster order within 8x8 blocks:
///   Block 0..3  = top-left 8x8
///   Block 4..7  = top-right 8x8
///   Block 8..11 = bottom-left 8x8
///   Block 12..15 = bottom-right 8x8
const BLOCK_INDEX_TO_XY: [(u32, u32); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];

/// Maps block index (0..15) to a raster-order index (row-major 4x4 grid)
/// for neighbor context lookups.
/// raster_idx = blk_y * 4 + blk_x
fn block_to_raster(block: usize) -> usize {
    let (bx, by) = BLOCK_INDEX_TO_XY[block];
    (by * 4 + bx) as usize
}

// ---------------------------------------------------------------------------
// MBAFF neighbor indices
// ---------------------------------------------------------------------------

/// Pre-computed MBAFF neighbor MB indices for context derivation.
///
/// Matches FFmpeg's `fill_decode_neighbors()` output: `left_mb_xy[LTOP]`,
/// `left_mb_xy[LBOT]`, and `top_mb_xy`. For non-MBAFF (progressive) frames,
/// these are just `mb_idx - 1` / `mb_idx - mb_width`. For MBAFF with mixed
/// field/frame pairs, the indices are adjusted per spec Table 6-4.
///
/// `None` means the neighbor is unavailable (out of bounds or different slice).
///
/// Reference: FFmpeg h264_mvpred.h:487-574 (`fill_decode_neighbors`).
#[derive(Clone, Debug)]
pub struct MbaffNeighbors {
    /// Left neighbor MB index (LTOP). Used by most left-context lookups.
    pub left_idx: Option<usize>,
    /// Left neighbor MB index (LBOT). May differ from `left_idx` when
    /// the current MB is field-mode and the left pair has different mode.
    pub left_idx_bot: Option<usize>,
    /// Top neighbor MB index.
    pub top_idx: Option<usize>,
    /// Left block remapping option (0-3) for MBAFF field/frame mode mismatch.
    ///
    /// Selects which sub-blocks of the left neighbor MB are used for NNZ,
    /// CBP, and intra4x4 mode context. See FFmpeg `left_block_options[4][32]`
    /// in h264_mvpred.h:491-496.
    ///
    /// - 0: default (same mode or non-MBAFF)
    /// - 1: bottom pair, curr frame, left field
    /// - 2: top pair, curr frame, left field
    /// - 3: curr field, left frame
    pub left_block_option: u8,
}

/// Compute MBAFF-aware neighbor MB indices for context derivation.
///
/// For progressive (`!is_mbaff`), returns simple indices.
/// For MBAFF, adjusts based on current/neighbor field/frame mode.
///
/// Reference: FFmpeg h264_mvpred.h:487-574 (`fill_decode_neighbors`).
#[allow(clippy::too_many_arguments)] // mirrors FFmpeg fill_decode_neighbors parameter set
pub fn compute_mbaff_neighbors(
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    is_mbaff: bool,
    curr_field: bool,
    mb_field_flags: &[bool],
    slice_table: &[u16],
    current_slice: u16,
) -> MbaffNeighbors {
    let mb_idx = (mb_y * mb_width + mb_x) as usize;
    let mb_stride = mb_width as usize;

    if !is_mbaff {
        // Progressive: simple indices with slice boundary check
        let left = if mb_x > 0 {
            let li = mb_idx - 1;
            if slice_table[li] == current_slice {
                Some(li)
            } else {
                None
            }
        } else {
            None
        };
        let top = if mb_y > 0 {
            let ti = mb_idx - mb_stride;
            if slice_table[ti] == current_slice {
                Some(ti)
            } else {
                None
            }
        } else {
            None
        };
        return MbaffNeighbors {
            left_idx: left,
            left_idx_bot: left,
            top_idx: top,
            left_block_option: 0,
        };
    }

    // --- MBAFF ---
    // Default: same as progressive but top uses field-adjusted stride
    // FFmpeg: top_xy = mb_xy - (mb_stride << MB_FIELD)
    let top_stride = if curr_field { 2 * mb_stride } else { mb_stride };
    let mut top_xy = if mb_idx >= top_stride {
        mb_idx - top_stride
    } else {
        usize::MAX
    };

    let mut left_xy_top = if mb_x > 0 { mb_idx - 1 } else { usize::MAX };
    let mut left_xy_bot = left_xy_top;
    let mut left_block_option = 0u8;

    if mb_x > 0 {
        let left_is_field = mb_field_flags.get(mb_idx - 1).copied().unwrap_or(false);
        let mode_mismatch = left_is_field != curr_field;

        if mb_y & 1 == 1 {
            // Bottom of pair
            if mode_mismatch {
                // Left = top of left pair
                let top_of_left = if mb_idx > mb_stride {
                    mb_idx - mb_stride - 1
                } else {
                    usize::MAX
                };
                left_xy_top = top_of_left;
                left_xy_bot = top_of_left;
                if curr_field && top_of_left != usize::MAX {
                    // Current is field: LBOT = bottom of left pair
                    left_xy_bot = top_of_left + mb_stride;
                    left_block_option = 3;
                } else {
                    left_block_option = 1;
                }
            }
        } else if mode_mismatch {
            // Top of pair
            if curr_field {
                // Current field, left frame: LBOT = bottom of left pair
                left_xy_bot = left_xy_top + mb_stride;
                left_block_option = 3;
            } else {
                // Current frame, left field: no index change
                left_block_option = 2;
            }
        }
    }

    // Top adjustment for top-of-pair field-mode current
    if curr_field && mb_y & 1 == 0 && top_xy != usize::MAX && top_xy < mb_field_flags.len() {
        // If above neighbor pair is frame-mode, adjust top to bottom of above pair
        let above_is_field = mb_field_flags.get(top_xy).copied().unwrap_or(false);
        if !above_is_field {
            top_xy += mb_stride;
        }
    }

    // Apply slice boundary checks
    let left_top = if left_xy_top != usize::MAX
        && left_xy_top < slice_table.len()
        && slice_table[left_xy_top] == current_slice
    {
        Some(left_xy_top)
    } else {
        None
    };
    let left_bot = if left_xy_bot != usize::MAX
        && left_xy_bot < slice_table.len()
        && slice_table[left_xy_bot] == current_slice
    {
        Some(left_xy_bot)
    } else {
        None
    };
    let top = if top_xy != usize::MAX
        && top_xy < slice_table.len()
        && slice_table[top_xy] == current_slice
    {
        Some(top_xy)
    } else {
        None
    };

    MbaffNeighbors {
        left_idx: left_top,
        left_idx_bot: left_bot,
        top_idx: top,
        left_block_option,
    }
}

/// Compute MBAFF-aware neighbor indices for CABAC skip context.
///
/// The skip function uses DIFFERENT neighbor logic from `fill_decode_neighbors`:
/// it works at the pair level and has its own field/frame adjustment.
///
/// Reference: FFmpeg h264_cabac.c:1336-1371 (`decode_cabac_mb_skip`).
#[allow(clippy::too_many_arguments)] // mirrors FFmpeg decode_cabac_mb_skip parameter set
pub fn mbaff_skip_neighbors(
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    is_mbaff: bool,
    curr_field: bool,
    mb_field_flags: &[bool],
    slice_table: &[u16],
    current_slice: u16,
) -> (Option<usize>, Option<usize>) {
    let mb_stride = mb_width as usize;

    if !is_mbaff {
        let mb_idx = (mb_y * mb_width + mb_x) as usize;
        let left = if mb_x > 0 {
            let li = mb_idx - 1;
            if slice_table[li] == current_slice {
                Some(li)
            } else {
                None
            }
        } else {
            None
        };
        let top = if mb_y > 0 {
            let ti = mb_idx - mb_stride;
            if slice_table[ti] == current_slice {
                Some(ti)
            } else {
                None
            }
        } else {
            None
        };
        return (left, top);
    }

    // MBAFF skip: mb_xy = mb_x + (mb_y & ~1) * mb_stride (top of pair)
    let pair_top_xy = mb_x as usize + ((mb_y & !1) as usize) * mb_stride;

    // Left (mba_xy): starts at top of left pair
    let mba = if mb_x > 0 {
        let mut mba_xy = pair_top_xy - 1;
        if mb_y & 1 == 1 {
            // Bottom MB: if left is same slice and same field/frame mode, go to bottom
            if mba_xy < slice_table.len()
                && slice_table[mba_xy] == current_slice
                && curr_field == mb_field_flags.get(mba_xy).copied().unwrap_or(false)
            {
                mba_xy += mb_stride;
            }
        }
        if mba_xy < slice_table.len() && slice_table[mba_xy] == current_slice {
            Some(mba_xy)
        } else {
            None
        }
    } else {
        None
    };

    // Top (mbb_xy)
    let mbb = if curr_field {
        // Field mode: mbb = pair_top - stride (= bottom of above pair)
        if pair_top_xy >= mb_stride {
            let mut mbb_xy = pair_top_xy - mb_stride;
            // If top of current pair and above is field-mode, go to top of above pair
            if mb_y & 1 == 0
                && mbb_xy < slice_table.len()
                && slice_table[mbb_xy] == current_slice
                && mb_field_flags.get(mbb_xy).copied().unwrap_or(false)
                && mbb_xy >= mb_stride
            {
                mbb_xy -= mb_stride;
            }
            if mbb_xy < slice_table.len() && slice_table[mbb_xy] == current_slice {
                Some(mbb_xy)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        // Frame mode: mbb = mb_x + (mb_y - 1) * stride
        if mb_y > 0 {
            let mbb_xy = mb_x as usize + (mb_y as usize - 1) * mb_stride;
            if mbb_xy < slice_table.len() && slice_table[mbb_xy] == current_slice {
                Some(mbb_xy)
            } else {
                None
            }
        } else {
            None
        }
    };

    (mba, mbb)
}

// ---------------------------------------------------------------------------
// Frame-level decode context
// ---------------------------------------------------------------------------

/// Frame-level decode context.
pub struct FrameDecodeContext {
    pub pic: PicHandle,
    pub mb_info: Vec<MbDeblockInfo>,
    pub neighbor_ctx: NeighborContext,
    /// Current QP (starts from PPS init_qp + slice_qp_delta, modified per-MB).
    pub qp: u8,
    pub mb_width: u32,
    pub mb_height: u32,
    /// Pre-computed dequantization tables.
    pub dequant4: Dequant4Table,
    /// Pre-computed 8x8 dequantization tables (High profile).
    pub dequant8: Dequant8Table,
    /// Per-MB transform_size_8x8_flag for deblocking edge selection.
    pub transform_8x8: Vec<bool>,
    /// Per-MB interlace flag for MBAFF. True if the MB was decoded in field mode.
    /// Used for neighbor lookups across field/frame pair boundaries.
    pub mb_field_flag: Vec<bool>,
    /// Motion vector context for inter prediction across MBs.
    pub mv_ctx: MvContext,
    /// Per-MB slice number for cross-slice neighbor availability.
    /// H.264 spec requires neighbors from different slices to be unavailable.
    pub slice_table: Vec<u16>,
    /// Current slice number (incremented per slice within a frame).
    pub current_slice: u16,
    /// Colocated L0 motion vectors from L1[0] reference frame (for spatial direct).
    pub col_mv: Vec<[i16; 2]>,
    /// Colocated L0 reference indices from L1[0] reference frame.
    pub col_ref: Vec<i8>,
    /// Colocated L1 MVs from L1[0] reference frame (fallback when col L0 ref < 0).
    pub col_mv_l1: Vec<[i16; 2]>,
    /// Colocated L1 reference indices from L1[0] reference frame.
    pub col_ref_l1: Vec<i8>,
    /// Per-MB intra flag from L1[0] reference frame.
    pub col_mb_intra: Vec<bool>,
    /// PPS constrained_intra_pred flag: inter neighbor pixels unavailable for intra prediction.
    pub constrained_intra_pred: bool,
    /// SPS direct_8x8_inference_flag: when false, spatial direct uses 4x4 col blocks.
    pub direct_8x8_inference_flag: bool,
    /// POC of colocated frame (L1[0]), for temporal direct mode.
    pub col_poc: i32,
    /// True if the colocated picture (L1[0]) is a long-term reference.
    /// When true, FFmpeg disables the col_zero_flag optimization in spatial
    /// direct mode (h264_direct.c:374,405,443).
    pub col_l1_is_long_term: bool,
    /// L0 ref POCs stored from the colocated frame, for temporal direct mode.
    /// Maps col_ref[blk] → ref_poc_l0[col_ref[blk]] → POC.
    pub col_ref_poc_l0: Vec<i32>,
    /// L1 ref POCs stored from the colocated frame, for temporal direct L1 fallback.
    pub col_ref_poc_l1: Vec<i32>,
    /// Current frame's L0 ref POCs (for temporal direct dist_scale_factor).
    pub cur_l0_ref_poc: Vec<i32>,
    /// Current frame's L0 ref DPB indices (for deblocking ref identity).
    /// DPB index uniquely identifies a picture even when POC collides
    /// (e.g., after MMCO-5 Reset).
    pub cur_l0_ref_dpb: Vec<usize>,
    /// Current frame's L1 ref POCs (for implicit weighted bipred).
    pub cur_l1_ref_poc: Vec<i32>,
    /// Current frame's L1 ref DPB indices (for deblocking ref identity).
    pub cur_l1_ref_dpb: Vec<usize>,
    /// Current picture POC (for temporal direct dist_scale_factor).
    pub cur_poc: i32,
    /// Pre-computed implicit weights for weighted_bipred_idc=2.
    /// `implicit_weight[ref0][ref1]` = w0 weight for biweight formula.
    /// w1 = 64 - w0, log2_denom = 5, offset = 0.
    pub implicit_weight: Vec<Vec<i32>>,
    /// Previous macroblock's qscale_diff (used by CABAC to select mb_qp_delta context).
    pub last_qscale_diff: i32,
    /// Per-slice deblocking filter parameters.
    /// Indexed by slice number (fdc.current_slice).
    pub slice_deblock_params: Vec<SliceDeblockParams>,
    /// True if chroma planes exist (chroma_format_idc != 0).
    /// When false (monochrome), chroma bitstream data is absent and U/V planes are 128-filled.
    pub decode_chroma: bool,
    /// True if the SPS has frame_mbs_only_flag=false (MBAFF capable).
    /// Used by `compute_mbaff_neighbors` to decide neighbor addressing.
    pub is_mbaff: bool,
    /// MBAFF field-coded MB flag. When true, this MB is part of a field-coded pair:
    /// stride is doubled (writes to every other row), and the bottom field (mb_y & 1 == 1)
    /// starts one row below the pair's top-left corner.
    /// Set by the decoder loop before calling apply_macroblock/decode_macroblock.
    pub mb_field: bool,
}

impl FrameDecodeContext {
    /// Create a new frame decode context for the given SPS and PPS.
    /// If `buffer_pool` is provided, acquires the PictureBuffer from it
    /// (reusing a previous allocation when possible).
    pub fn new(sps: &Sps, pps: &Pps, buffer_pool: Option<&Arc<Mutex<BufferPool>>>) -> Self {
        let mb_width = sps.mb_width;
        let mb_height = sps.mb_height;
        let width = mb_width * 16;
        let height = mb_height * 16;

        let decode_chroma = sps.chroma_format_idc != 0;
        // Monochrome: fill U/V with 128 (mid-gray). FFmpeg outputs monochrome as
        // yuv420p with U/V=128, not gray8. Chroma prediction/MC on 128-filled planes
        // naturally produces 128, so no additional guards needed in reconstruction.
        let uv_fill = if decode_chroma { 0u8 } else { 128u8 };

        let pic = if let Some(pool) = buffer_pool {
            let buf = pool
                .lock()
                .unwrap()
                .acquire(width, height, mb_width, mb_height, uv_fill);
            PicHandle::new_pooled(buf, pool)
        } else {
            let y_stride = width as usize;
            let uv_stride = (width / 2) as usize;
            PicHandle::new(PictureBuffer {
                y: vec![0u8; y_stride * height as usize],
                u: vec![uv_fill; uv_stride * (height / 2) as usize],
                v: vec![uv_fill; uv_stride * (height / 2) as usize],
                y_stride,
                uv_stride,
                width,
                height,
                mb_width,
                mb_height,
            })
        };

        let total_mbs = (mb_width * mb_height) as usize;

        // Log dequant table checksums for diagnosing scaling matrix issues
        debug!(
            dq4_intra_y_sum = pps.scaling_matrix4[0]
                .iter()
                .map(|&x| x as u32)
                .sum::<u32>(),
            dq4_inter_y_sum = pps.scaling_matrix4[3]
                .iter()
                .map(|&x| x as u32)
                .sum::<u32>(),
            dq8_intra_y_sum = pps.scaling_matrix8[0]
                .iter()
                .map(|&x| x as u32)
                .sum::<u32>(),
            dq8_inter_y_sum = pps.scaling_matrix8[3]
                .iter()
                .map(|&x| x as u32)
                .sum::<u32>(),
            scaling_present = pps.scaling_matrix_present,
            "DEQUANT_TABLES"
        );

        Self {
            pic,
            mb_info: vec![MbDeblockInfo::default(); total_mbs],
            neighbor_ctx: NeighborContext::new(mb_width),
            qp: 0,
            mb_width,
            mb_height,
            dequant4: Dequant4Table::new(&pps.scaling_matrix4),
            dequant8: Dequant8Table::new(&pps.scaling_matrix8),
            transform_8x8: vec![false; total_mbs],
            mb_field_flag: vec![false; total_mbs],
            mv_ctx: MvContext::new(mb_width, mb_height),
            slice_table: vec![u16::MAX; total_mbs],
            current_slice: 0,
            col_mv: Vec::new(),
            col_ref: Vec::new(),
            col_mv_l1: Vec::new(),
            col_ref_l1: Vec::new(),
            col_mb_intra: Vec::new(),
            col_l1_is_long_term: false,
            constrained_intra_pred: pps.constrained_intra_pred,
            direct_8x8_inference_flag: sps.direct_8x8_inference_flag,
            col_poc: 0,
            col_ref_poc_l0: Vec::new(),
            col_ref_poc_l1: Vec::new(),
            cur_l0_ref_poc: Vec::new(),
            cur_l0_ref_dpb: Vec::new(),
            cur_l1_ref_poc: Vec::new(),
            cur_l1_ref_dpb: Vec::new(),
            cur_poc: 0,
            implicit_weight: Vec::new(),
            last_qscale_diff: 0,
            slice_deblock_params: Vec::new(),
            decode_chroma,
            is_mbaff: !sps.frame_mbs_only_flag,
            mb_field: false,
        }
    }

    /// Compute implicit weights for weighted_bipred_idc=2.
    ///
    /// For each (ref0, ref1) pair, computes w0 from POC distances:
    ///   td = clip(-128, 127, POC[ref1] - POC[ref0])
    ///   tb = clip(-128, 127, POC[current] - POC[ref0])
    ///   dist_scale_factor = (tb * ((16384 + abs(td)/2) / td) + 32) >> 8
    ///   w0 = 64 - dist_scale_factor  (if in range [-64, 128])
    ///
    /// Reference: FFmpeg h264_slice.c:688-747 implicit_weight_table()
    pub fn compute_implicit_weights(&mut self) {
        let cur_poc = self.cur_poc;
        let ref_count0 = self.cur_l0_ref_poc.len();
        let ref_count1 = self.cur_l1_ref_poc.len();

        // Early exit: single ref on each list with symmetric POC → no weighting
        if ref_count0 == 1
            && ref_count1 == 1
            && self.cur_l0_ref_poc[0] as i64 + self.cur_l1_ref_poc[0] as i64 == 2 * cur_poc as i64
        {
            self.implicit_weight = vec![vec![32; ref_count1]; ref_count0];
            return;
        }

        let mut weights = vec![vec![32i32; ref_count1]; ref_count0];
        for (r0, &poc0) in self.cur_l0_ref_poc.iter().enumerate() {
            for (r1, &poc1) in self.cur_l1_ref_poc.iter().enumerate() {
                let td = (poc1 - poc0).clamp(-128, 127);
                if td != 0 {
                    let tb = (cur_poc - poc0).clamp(-128, 127);
                    let tx = (16384 + (td.abs() >> 1)) / td;
                    let dist_scale_factor = (tb * tx + 32) >> 8;
                    if (-64..=128).contains(&dist_scale_factor) {
                        weights[r0][r1] = 64 - dist_scale_factor;
                    }
                }
            }
        }
        self.implicit_weight = weights;
    }
}

// ---------------------------------------------------------------------------
// Deblock info helper
// ---------------------------------------------------------------------------

/// Finalize a skip/direct MB: update slice table, neighbor context, and deblock info.
///
/// Called at the end of `decode_skip_mb` and `decode_b_skip_mb` after MC is applied.
/// Skip MBs have no residual and no intra prediction, so non_zero_count is all zeros.
/// Inter MBs make neighbors unavailable under constrained_intra_pred.
#[inline]
fn finalize_skip_mb(ctx: &mut FrameDecodeContext, mb_x: u32, mb_y: u32, is_b_slice: bool) {
    let mb_idx = (mb_y * ctx.mb_width + mb_x) as usize;
    ctx.slice_table[mb_idx] = ctx.current_slice;
    ctx.mb_field_flag[mb_idx] = ctx.mb_field;
    let nz = [0u8; 24];
    let modes = if ctx.constrained_intra_pred {
        [-1i8; 16] // unavailable for constrained intra prediction
    } else {
        [2i8; 16] // DC_PRED for inter MBs
    };
    ctx.neighbor_ctx.update_after_mb(mb_x, &nz, &modes);
    ctx.neighbor_ctx.left_available = true;
    let qp = ctx.qp;
    store_deblock_info(ctx, mb_idx, false, is_b_slice, qp, [0; 24], 0, false);
}

/// Store `MbDeblockInfo` for the deblocking filter after decoding a macroblock.
///
/// Copies the current MV and ref_idx from `mv_ctx` into `mb_info[mb_idx]`,
/// converting list-relative ref_idx to picture POC for identity comparison.
/// Called from `apply_macroblock` and `finalize_skip_mb`.
/// `is_b_slice`: true for B-slice MBs (copies L1 data for two-permutation BS check).
#[allow(clippy::too_many_arguments)] // all parameters are logically required
#[inline]
fn store_deblock_info(
    ctx: &mut FrameDecodeContext,
    mb_idx: usize,
    is_intra: bool,
    is_b_slice: bool,
    qp: u8,
    non_zero_count: [u8; 24],
    cbp: u8,
    is_cabac: bool,
) {
    let mb_idx_base = mb_idx * 16;
    let mut deblock_mv = [[0i16; 2]; 16];
    let mut deblock_ref_poc = [i32::MIN; 16];
    let mut deblock_mv_l1 = [[0i16; 2]; 16];
    let mut deblock_ref_poc_l1 = [i32::MIN; 16];
    if !is_intra && mb_idx_base + 16 <= ctx.mv_ctx.mv.len() {
        deblock_mv.copy_from_slice(&ctx.mv_ctx.mv[mb_idx_base..mb_idx_base + 16]);
        // Convert ref_idx to DPB index for deblocking ref identity comparison.
        // DPB index uniquely identifies a picture regardless of slice type,
        // ensuring consistent bS computation at P/B slice boundaries and
        // correct behavior with MMCO-5 (where POC can collide).
        for (blk, id) in deblock_ref_poc.iter_mut().enumerate() {
            let ri = ctx.mv_ctx.ref_idx[mb_idx_base + blk];
            *id = if ri >= 0 {
                ctx.cur_l0_ref_dpb
                    .get(ri as usize)
                    .map(|&dpb_idx| dpb_idx as i32)
                    .unwrap_or(i32::MIN)
            } else {
                i32::MIN
            };
        }
        if is_b_slice {
            deblock_mv_l1.copy_from_slice(&ctx.mv_ctx.mv_l1[mb_idx_base..mb_idx_base + 16]);
            for (blk, id) in deblock_ref_poc_l1.iter_mut().enumerate() {
                let ri = ctx.mv_ctx.ref_idx_l1[mb_idx_base + blk];
                *id = if ri >= 0 {
                    ctx.cur_l1_ref_dpb
                        .get(ri as usize)
                        .map(|&dpb_idx| dpb_idx as i32)
                        .unwrap_or(i32::MIN)
                } else {
                    i32::MIN
                };
            }
        }
    }
    ctx.mb_info[mb_idx] = MbDeblockInfo {
        is_intra,
        qp,
        list_count: if is_b_slice { 2 } else { 1 },
        non_zero_count,
        ref_poc: deblock_ref_poc,
        mv: deblock_mv,
        ref_poc_l1: deblock_ref_poc_l1,
        mv_l1: deblock_mv_l1,
        transform_8x8: ctx.transform_8x8[mb_idx],
        cbp,
        is_cabac,
        mb_field: ctx.mb_field,
    };
}

// ---------------------------------------------------------------------------
// Pixel access helpers
// ---------------------------------------------------------------------------

/// Get the offset and stride for a 4x4 luma block within the picture buffer.
///
/// In MBAFF field mode (`mb_field=true`), the stride is doubled and the
/// bottom field (mb_y & 1 == 1) starts one frame-row below the pair's top-left.
/// Reference: FFmpeg h264_mb_template.c:65-96 (MB_FIELD linesize adjustment).
#[inline]
fn luma_block_offset(
    pic: &PictureBuffer,
    mb_x: u32,
    mb_y: u32,
    blk_x: u32,
    blk_y: u32,
    mb_field: bool,
) -> (usize, usize) {
    let x = (mb_x * 16 + blk_x * 4) as usize;
    let stride = pic.y_stride;
    if mb_field {
        let pair_base_y = ((mb_y & !1) * 16) as usize;
        let field_offset = if mb_y & 1 == 1 { stride } else { 0 };
        let y_in_mb = (blk_y * 4) as usize;
        (
            pair_base_y * stride + field_offset + y_in_mb * stride * 2 + x,
            stride * 2,
        )
    } else {
        let y = (mb_y * 16 + blk_y * 4) as usize;
        (y * stride + x, stride)
    }
}

/// Get the offset and stride for a 4x4 chroma block within the picture buffer.
///
/// Same MBAFF field-mode adjustment as `luma_block_offset` but for chroma planes.
#[inline]
fn chroma_block_offset(
    pic: &PictureBuffer,
    mb_x: u32,
    mb_y: u32,
    blk_x: u32,
    blk_y: u32,
    mb_field: bool,
) -> (usize, usize) {
    let x = (mb_x * 8 + blk_x * 4) as usize;
    let stride = pic.uv_stride;
    if mb_field {
        let pair_base_y = ((mb_y & !1) * 8) as usize;
        let field_offset = if mb_y & 1 == 1 { stride } else { 0 };
        let y_in_mb = (blk_y * 4) as usize;
        (
            pair_base_y * stride + field_offset + y_in_mb * stride * 2 + x,
            stride * 2,
        )
    } else {
        let y = (mb_y * 8 + blk_y * 4) as usize;
        (y * stride + x, stride)
    }
}

/// Get the offset and stride for the top-left of a luma MB.
#[inline]
fn luma_mb_offset(pic: &PictureBuffer, mb_x: u32, mb_y: u32, mb_field: bool) -> (usize, usize) {
    luma_block_offset(pic, mb_x, mb_y, 0, 0, mb_field)
}

/// Get the offset and stride for the top-left of a chroma MB.
#[inline]
fn chroma_mb_offset(pic: &PictureBuffer, mb_x: u32, mb_y: u32, mb_field: bool) -> (usize, usize) {
    chroma_block_offset(pic, mb_x, mb_y, 0, 0, mb_field)
}

/// Gather the top 4 neighbor pixels for a 4x4 luma block.
/// Returns up to 8 values (4 top + 4 top-right for diagonal modes).
///
/// Uses offset/stride addressing for MBAFF field-mode compatibility.
/// `offset` is the block's top-left position in the luma buffer.
/// `stride` is the field-adjusted stride (doubled in field mode).
fn gather_top_luma(
    plane: &[u8],
    offset: usize,
    stride: usize,
    has_top: bool,
    has_top_right: bool,
) -> [u8; 8] {
    let mut top = [128u8; 8];
    if has_top && offset >= stride {
        let row_above = offset - stride;
        top[..4].copy_from_slice(&plane[row_above..row_above + 4]);
        if has_top_right {
            top[4..8].copy_from_slice(&plane[row_above + 4..row_above + 8]);
        } else {
            let v = top[3];
            top[4..8].fill(v);
        }
    }
    top
}

/// Gather the left 4 neighbor pixels for a 4x4 luma block.
///
/// Uses offset/stride addressing. `offset` is the block's top-left.
fn gather_left_luma(plane: &[u8], offset: usize, stride: usize, has_left: bool) -> [u8; 4] {
    let mut left = [128u8; 4];
    if has_left {
        for (i, l) in left.iter_mut().enumerate() {
            *l = plane[offset + i * stride - 1];
        }
    }
    left
}

/// Gather the top-left corner pixel.
///
/// Uses offset/stride addressing.
fn gather_top_left_luma(
    plane: &[u8],
    offset: usize,
    stride: usize,
    has_top: bool,
    has_left: bool,
) -> u8 {
    if has_top && has_left && offset >= stride {
        plane[offset - stride - 1]
    } else {
        128
    }
}

/// Gather N pixels from the row above a block.
///
/// Uses offset/stride addressing for MBAFF field-mode compatibility.
/// `offset` is the block's top-left position in the plane buffer.
/// Returns `[128; N]` if no row above (offset < stride).
#[inline]
fn gather_top<const N: usize>(plane: &[u8], offset: usize, stride: usize) -> [u8; N] {
    let mut top = [128u8; N];
    if offset >= stride {
        let row_above = offset - stride;
        top.copy_from_slice(&plane[row_above..row_above + N]);
    }
    top
}

/// Gather N pixels from the column left of a block.
///
/// Uses offset/stride addressing for MBAFF field-mode compatibility.
/// `offset` is the block's top-left position in the plane buffer.
/// Returns `[128; N]` if `has_left` is false.
#[inline]
fn gather_left<const N: usize>(
    plane: &[u8],
    offset: usize,
    stride: usize,
    has_left: bool,
) -> [u8; N] {
    let mut left = [128u8; N];
    if has_left {
        for (i, l) in left.iter_mut().enumerate() {
            *l = plane[offset + i * stride - 1];
        }
    }
    left
}

/// Gather the top-left corner pixel.
///
/// Uses offset/stride addressing for MBAFF field-mode compatibility.
/// Returns `128` if `has_top` or `has_left` is false.
#[inline]
fn gather_top_left(
    plane: &[u8],
    offset: usize,
    stride: usize,
    has_top: bool,
    has_left: bool,
) -> u8 {
    if has_top && has_left && offset >= stride {
        plane[offset - stride - 1]
    } else {
        128
    }
}

// ---------------------------------------------------------------------------
// Chroma decode helper
// ---------------------------------------------------------------------------

/// Decode chroma planes (U and V) for an intra macroblock.
#[allow(clippy::too_many_arguments)] // H.264 decode requires all these parameters
///
/// Applies chroma intra prediction, then chroma DC Hadamard + dequant,
/// then AC dequant + IDCT for each 4x4 chroma block.
fn decode_chroma(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    mb_x: u32,
    mb_y: u32,
    chroma_qp: [u8; 2],
    has_top: bool,
    has_left: bool,
) {
    let cbp_chroma = (mb.cbp >> 4) & 3;

    for (plane_idx, &c_qp) in chroma_qp.iter().enumerate() {
        // Compute field-aware offset before borrowing plane data mutably
        let (offset, c_stride) = chroma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);

        // Select the correct chroma plane
        let plane_data = if plane_idx == 0 {
            &mut ctx.pic.u as &mut Vec<u8>
        } else {
            &mut ctx.pic.v as &mut Vec<u8>
        };
        let top: [u8; 8] = gather_top(plane_data, offset, c_stride);
        let left: [u8; 8] = gather_left(plane_data, offset, c_stride, has_left);
        let top_left = gather_top_left(plane_data, offset, c_stride, has_top, has_left);
        intra_pred::predict_chroma_8x8(
            &mut plane_data[offset..],
            c_stride,
            mb.chroma_pred_mode,
            &top,
            &left,
            top_left,
            has_top,
            has_left,
        );

        {
            let mut pred_row0 = [0u8; 8];
            pred_row0.copy_from_slice(&plane_data[offset..offset + 8]);
            let plane_name = if plane_idx == 0 { "U" } else { "V" };
            trace!(mb_x, mb_y, plane = plane_name, row0 = ?pred_row0, "chroma prediction");
        }

        if cbp_chroma > 0 {
            // Chroma DC: inverse Hadamard + dequant (i32 output to avoid i16 overflow)
            // FFmpeg uses list 1+plane for intra (1=Cb, 2=Cr)
            let dc_cqm = 1 + plane_idx;
            let qmul = dequant::dc_dequant_scale(&ctx.dequant4, dc_cqm, c_qp);
            let mut chroma_dc_out = [0i32; 4];
            idct::chroma_dc_dequant_idct(&mut chroma_dc_out, &mb.chroma_dc[plane_idx], qmul);

            // For each 4x4 chroma block
            for (blk_idx, &dc_val) in chroma_dc_out.iter().enumerate() {
                let blk_x = (blk_idx & 1) as u32;
                let blk_y = (blk_idx >> 1) as u32;

                let (c_offset, c_stride) =
                    chroma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y, ctx.mb_field);
                let plane_data = if plane_idx == 0 {
                    &mut ctx.pic.u
                } else {
                    &mut ctx.pic.v
                };

                if cbp_chroma >= 2 {
                    // AC coefficients present — combine DC into coeffs[0] and
                    // process everything through a single IDCT pass (matching
                    // FFmpeg, which applies one +32 rounding bias for DC+AC).
                    let cqm = 1 + plane_idx; // intra: 1=Cb, 2=Cr
                    mb.chroma_ac[plane_idx][blk_idx][0] = dc_val as i16;
                    dequant::dequant_4x4(
                        &mut mb.chroma_ac[plane_idx][blk_idx],
                        &ctx.dequant4.coeffs[cqm][c_qp as usize],
                    );
                    // dequant_4x4 scaled [0] by the AC dequant factor, but
                    // DC was already fully dequantized by the Hadamard. Restore it.
                    mb.chroma_ac[plane_idx][blk_idx][0] = dc_val as i16;
                    idct::idct4x4_add(
                        &mut plane_data[c_offset..],
                        c_stride,
                        &mut mb.chroma_ac[plane_idx][blk_idx],
                    );
                } else {
                    // DC-only: apply directly with rounding.
                    let dc_add = (dc_val + 32) >> 6;
                    for j in 0..4 {
                        for i in 0..4 {
                            let idx = c_offset + j * c_stride + i;
                            plane_data[idx] = (plane_data[idx] as i32 + dc_add).clamp(0, 255) as u8;
                        }
                    }
                }

                {
                    let mut final_px = [0u8; 16];
                    for r in 0..4 {
                        let off = c_offset + r * c_stride;
                        final_px[r * 4..r * 4 + 4].copy_from_slice(&plane_data[off..off + 4]);
                    }
                    let plane_name = if plane_idx == 0 { "U" } else { "V" };
                    trace!(mb_x, mb_y, plane = plane_name, blk_x, blk_y, pixels = ?final_px, "chroma final");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main macroblock decode function
// ---------------------------------------------------------------------------

/// Decode one macroblock at position (mb_x, mb_y).
///
/// Reads syntax elements from the bitstream via CAVLC, applies dequantization,
/// intra prediction, and IDCT to produce decoded pixels in the picture buffer.
///
/// `ref_pics` contains the decoded reference picture buffers for inter prediction
/// (list 0). Index into this slice corresponds to ref_idx values from CAVLC.
#[allow(clippy::too_many_arguments)] // H.264 MB decode requires all these parameters
#[tracing::instrument(skip_all, fields(mb_x, mb_y))]
pub fn decode_macroblock(
    ctx: &mut FrameDecodeContext,
    br: &mut BitReadBE,
    slice_hdr: &SliceHeader,
    _sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    ref_pics: &[Arc<SharedPicture>],
    ref_pics_l1: &[Arc<SharedPicture>],
) -> Result<()> {
    // 1. Parse macroblock syntax via CAVLC
    let mut mb = decode_mb_cavlc(
        br,
        slice_hdr.slice_type,
        pps,
        &ctx.neighbor_ctx,
        mb_x,
        mb_y,
        ctx.mb_width,
        slice_hdr.num_ref_idx_l0_active,
        slice_hdr.num_ref_idx_l1_active,
        ctx.direct_8x8_inference_flag,
        ctx.decode_chroma,
        ctx.mb_field,
    )?;

    // 2. Apply entropy-agnostic processing
    apply_macroblock(
        ctx,
        &mut mb,
        slice_hdr,
        pps,
        mb_x,
        mb_y,
        ref_pics,
        ref_pics_l1,
    )
}

/// Apply entropy-agnostic macroblock processing.
///
/// Takes a parsed `Macroblock` (from either CAVLC or CABAC) and applies:
/// QP update, dequantization, IDCT, intra prediction, motion compensation,
/// neighbor context update, and deblock info storage.
///
/// This function is the shared backend for both CAVLC and CABAC decode paths.
#[allow(clippy::too_many_arguments)]
pub fn apply_macroblock(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    slice_hdr: &SliceHeader,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    ref_pics: &[Arc<SharedPicture>],
    ref_pics_l1: &[Arc<SharedPicture>],
) -> Result<()> {
    let mb_idx = (mb_y * ctx.mb_width + mb_x) as usize;

    // Update QP with mb_qp_delta
    // I_PCM: QP=0 for deblocking table, but running QP stays unchanged
    // (FFmpeg h264_cavlc.c: sl->qscale is NOT modified for I_PCM).
    if mb.mb_qp_delta != 0 {
        ctx.qp = ((ctx.qp as i32 + mb.mb_qp_delta).rem_euclid(52)) as u8;
    }
    ctx.last_qscale_diff = mb.mb_qp_delta;
    let qp = if mb.is_pcm { 0 } else { ctx.qp };

    // Compute per-plane chroma QP (Cb uses offset[0], Cr uses offset[1])
    let c_qp = [
        CHROMA_QP_TABLE[(qp as i32 + pps.chroma_qp_index_offset[0]).clamp(0, 51) as usize],
        CHROMA_QP_TABLE[(qp as i32 + pps.chroma_qp_index_offset[1]).clamp(0, 51) as usize],
    ];

    // Check neighbor availability with slice boundary awareness.
    // H.264 spec: neighbors from different slices are unavailable.
    // For MBAFF, use adjusted neighbor indices from fill_decode_neighbors logic.
    let nb = compute_mbaff_neighbors(
        mb_x,
        mb_y,
        ctx.mb_width,
        ctx.is_mbaff,
        ctx.mb_field,
        &ctx.mb_field_flag,
        &ctx.slice_table,
        ctx.current_slice,
    );
    let mut has_top = nb.top_idx.is_some();
    let mut has_left = nb.left_idx.is_some();

    // Constrained intra prediction: inter neighbor pixels are unavailable
    // for intra prediction (H.264 spec, FFmpeg h264_mvpred.h:598).
    if ctx.constrained_intra_pred && (mb.is_intra4x4 || mb.is_intra16x16 || mb.is_pcm) {
        if let Some(ti) = nb.top_idx
            && !ctx.mb_info[ti].is_intra
        {
            has_top = false;
        }
        if let Some(li) = nb.left_idx
            && !ctx.mb_info[li].is_intra
        {
            has_left = false;
        }
    }

    trace!(
        mb_x,
        mb_y,
        qp,
        mb_type = mb.mb_type,
        cbp = mb.cbp,
        is_intra4x4 = mb.is_intra4x4,
        is_intra16x16 = mb.is_intra16x16,
        t8x8 = mb.transform_size_8x8_flag,
        "decoded MB"
    );
    trace!(
        mb_x,
        mb_y,
        mb_field = ctx.mb_field,
        has_top,
        has_left,
        left_block_option = nb.left_block_option,
        "INTRA_RECON_START"
    );

    // Decode based on macroblock type
    if mb.is_pcm {
        // I_PCM: raw samples already in the entropy coder output
        // Copy luma samples (16x16) with field-aware addressing
        let (luma_base, luma_stride) = luma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
        for y in 0..16u32 {
            for x in 0..16u32 {
                let blk = ((y / 4) * 4 + (x / 4)) as usize;
                let sub_y = (y % 4) as usize;
                let sub_x = (x % 4) as usize;
                ctx.pic.y[luma_base + y as usize * luma_stride + x as usize] =
                    mb.luma_coeffs[blk][sub_y * 4 + sub_x] as u8;
            }
        }
        // Copy chroma samples (8x8 each) — skip for monochrome (U/V stay 128).
        if ctx.decode_chroma {
            let (chroma_base, chroma_stride) = chroma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
            for plane_idx in 0..2usize {
                let plane = if plane_idx == 0 {
                    &mut ctx.pic.u
                } else {
                    &mut ctx.pic.v
                };
                for y in 0..8u32 {
                    for x in 0..8u32 {
                        let blk = ((y / 4) * 2 + (x / 4)) as usize;
                        let sub_y = (y % 4) as usize;
                        let sub_x = (x % 4) as usize;
                        plane[chroma_base + y as usize * chroma_stride + x as usize] =
                            mb.chroma_ac[plane_idx][blk][sub_y * 4 + sub_x] as u8;
                    }
                }
            }
        } // if ctx.decode_chroma
    } else if mb.is_intra4x4 {
        if mb.transform_size_8x8_flag {
            decode_intra8x8(ctx, mb, mb_x, mb_y, qp, c_qp, has_top, has_left);
        } else {
            decode_intra4x4(ctx, mb, mb_x, mb_y, qp, c_qp, has_top, has_left);
        }
    } else if mb.is_intra16x16 {
        decode_intra16x16(ctx, mb, mb_x, mb_y, qp, c_qp, has_top, has_left);
    }

    // Intra MBs: set MV context to ref=-1 (LIST_NOT_USED), mv=[0,0].
    // Without this, the initialization value of -2 (PART_NOT_AVAILABLE) remains,
    // which causes spatial direct prediction to compute wrong ref_idx when an
    // intra MB is a neighbor (unsigned min treats -2 as 254, not 255).
    // Reference: FFmpeg fill_decode_caches sets LIST_NOT_USED for intra neighbors.
    if mb.is_intra {
        for blk in 0..16 {
            ctx.mv_ctx.set(mb_x, mb_y, blk, [0, 0], -1);
            ctx.mv_ctx.set_l1(mb_x, mb_y, blk, [0, 0], -1);
        }
    }

    if !mb.is_intra {
        // Inter macroblock (P or B)
        decode_inter_mb(
            ctx,
            mb,
            slice_hdr,
            mb_x,
            mb_y,
            qp,
            c_qp,
            ref_pics,
            ref_pics_l1,
        );
    }

    // Update neighbor context
    // Build intra4x4 mode array for neighbor tracking.
    // For I_4x4 MBs: pass the decoded modes.
    // For other MBs (I_16x16, I_PCM, inter): pass 2 (DC_PRED).
    //
    // H.264 spec 8.3.1.1: when a neighbor is not Intra_4x4, its prediction
    // mode is inferred as DC_PRED (2). FFmpeg's fill_decode_caches uses
    // `2 - 3 * !(type & type_mask)` which yields 2 for all non-I_4x4 types
    // (both intra-16x16 and inter).  Previously we stored -1 for inter MBs,
    // which made the "unavailable" path fire and always predicted DC_PRED=2
    // — correct in isolation, but wrong when min(left, top) would yield a
    // smaller predicted mode (e.g., min(2, 1) = 1 ≠ 2).
    let intra4x4_modes: [i8; 16] = if mb.is_intra4x4 {
        let mut modes = [-1i8; 16];
        for (i, mode) in modes.iter_mut().enumerate() {
            *mode = mb.intra4x4_pred_mode[i] as i8;
        }
        modes
    } else if ctx.constrained_intra_pred && !mb.is_intra {
        [-1i8; 16] // unavailable for constrained intra prediction
    } else {
        [2i8; 16] // DC_PRED for I_16x16, I_PCM, and inter MBs
    };
    ctx.slice_table[mb_idx] = ctx.current_slice;
    ctx.transform_8x8[mb_idx] = mb.transform_size_8x8_flag;
    ctx.mb_field_flag[mb_idx] = ctx.mb_field;
    ctx.neighbor_ctx
        .update_after_mb(mb_x, &mb.non_zero_count, &intra4x4_modes);
    ctx.neighbor_ctx.left_available = true;

    // Per-MB pipeline-stage checksum for diffing against FFmpeg.
    // Captures reconstruction state BEFORE deblocking. Compare with FFmpeg via:
    //   scripts/mb_recon_compare.py <file> --frame N
    // To enable: RUST_LOG=wedeo_codec_h264::mb=trace (appears as MB_RECON lines).
    {
        let (y_off, y_stride) = luma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
        let (uv_off, uv_stride) = chroma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
        let y_sum = mb_plane_sum(&ctx.pic.y, y_off, y_stride, 16);
        let u_sum = mb_plane_sum(&ctx.pic.u, uv_off, uv_stride, 8);
        let v_sum = mb_plane_sum(&ctx.pic.v, uv_off, uv_stride, 8);
        trace!(
            mb_x,
            mb_y,
            mb_type = mb.mb_type,
            qp,
            cbp = mb.cbp,
            t8x8 = mb.transform_size_8x8_flag,
            is_intra = mb.is_intra,
            y_sum,
            u_sum,
            v_sum,
            "MB_RECON"
        );
        trace!(
            mb_x,
            mb_y,
            y_row0 = ?&ctx.pic.y[y_off..y_off + 16],
            y_row1 = ?&ctx.pic.y[y_off + y_stride..y_off + y_stride + 16],
            y_row2 = ?&ctx.pic.y[y_off + 2 * y_stride..y_off + 2 * y_stride + 16],
            y_row3 = ?&ctx.pic.y[y_off + 3 * y_stride..y_off + 3 * y_stride + 16],
            "INTRA_FINAL_ROWS"
        );
    }

    // Store MbDeblockInfo for the deblocking filter.
    // For CAVLC 8x8 DCT, compute CBP from NNZ sums (matching FFmpeg's cbp_table
    // bits 12-15 = !!nnz_sum per 8x8 block) rather than bitstream CBP.
    // A block can be "coded" in CBP but have all-zero coefficients.
    let is_cabac = pps.entropy_coding_mode_flag;
    let deblock_cbp = if ctx.transform_8x8[mb_idx] && !is_cabac {
        // First raster position of each 8x8 block holds the NNZ sum after
        // CAVLC broadcast (cavlc.rs: nnz[r0] += nnz[r1] + nnz[r2] + nnz[r3]).
        const FIRST_RASTER: [usize; 4] = [0, 2, 8, 10];
        let mut cbp: u8 = 0;
        for (i, &r0) in FIRST_RASTER.iter().enumerate() {
            if mb.non_zero_count[r0] > 0 {
                cbp |= 1 << i;
            }
        }
        cbp
    } else {
        (mb.cbp & 0x0F) as u8
    };
    store_deblock_info(
        ctx,
        mb_idx,
        mb.is_intra,
        slice_hdr.slice_type.is_b(),
        qp,
        mb.non_zero_count,
        deblock_cbp,
        is_cabac,
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Inter MB decode (P-slice)
// ---------------------------------------------------------------------------

/// Decode an inter macroblock: compute MV from prediction + MVD, apply motion
/// compensation to get a prediction block, then add the dequantized/IDCT residual.
///
/// Handles P_L0_16x16, P_L0_L0_16x8, P_L0_L0_8x16, P_8x8, and P_SKIP.
/// P_SKIP is signalled by mb_type == u32::MAX from the caller (mb_skip_run handling).
#[allow(clippy::too_many_arguments)] // H.264 decode requires all these parameters
fn decode_inter_mb(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    slice_hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: [u8; 2],
    ref_pics: &[Arc<SharedPicture>],
    ref_pics_l1: &[Arc<SharedPicture>],
) {
    // Dispatch B-frame inter MBs to dedicated handler
    if slice_hdr.slice_type.is_b() {
        decode_b_inter_mb(
            ctx,
            mb,
            slice_hdr,
            mb_x,
            mb_y,
            qp,
            chroma_qp,
            ref_pics,
            ref_pics_l1,
        );
        return;
    }

    if ref_pics.is_empty() {
        // No reference frames available — fill with gray and return.
        fill_mb_gray(ctx, mb_x, mb_y);
        // Still need to set MV context for neighbor prediction
        for blk in 0..16 {
            ctx.mv_ctx.set(mb_x, mb_y, blk, [0, 0], 0);
        }
        return;
    }

    trace!(mb_x, mb_y, mb_type = mb.mb_type, "inter MB");

    match mb.mb_type {
        0 => {
            // P_L0_16x16: one 16x16 partition
            let ref_idx = mb.ref_idx_l0[0].max(0) as usize;
            let n = ctx.mv_ctx.get_neighbors_slice(
                mb_x,
                mb_y,
                0,
                0,
                4,
                Some(&ctx.slice_table),
                ctx.current_slice,
            );
            trace!(
                mb_x, mb_y,
                mv_a = ?n.mv_a, ref_a = n.ref_a, a_avail = n.a_avail,
                mv_b = ?n.mv_b, ref_b = n.ref_b, b_avail = n.b_avail,
                mv_c = ?n.mv_c, ref_c = n.ref_c, c_avail = n.c_avail,
                ref_idx, "16x16 neighbors"
            );
            let mvp = mvpred::predict_mv(
                n.mv_a,
                n.mv_b,
                n.mv_c,
                n.ref_a,
                n.ref_b,
                n.ref_c,
                ref_idx as i8,
                n.a_avail,
                n.b_avail,
                n.c_avail,
            );
            let mv = [
                mvp[0].wrapping_add(mb.mvd_l0[0][0]),
                mvp[1].wrapping_add(mb.mvd_l0[0][1]),
            ];
            trace!(mb_x, mb_y, mvp = ?mvp, mvd = ?mb.mvd_l0[0], mv = ?mv, "16x16 MV");

            let ref_pic = &ref_pics[ref_idx.min(ref_pics.len() - 1)];
            apply_mc_partition(ctx, ref_pic, mb_x, mb_y, 0, 0, 16, 16, mv);
            if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                apply_weight_p(ctx, slice_hdr, mb_x, mb_y, 0, 0, 16, 16, ref_idx);
            }

            // Fill MV context for all 16 4x4 blocks
            for blk in 0..16 {
                ctx.mv_ctx.set(mb_x, mb_y, blk, mv, ref_idx as i8);
            }
        }
        1 => {
            // P_L0_L0_16x8: two 16x8 partitions
            for part in 0..2u32 {
                let ref_idx = mb.ref_idx_l0[part as usize].max(0) as usize;
                let blk_y = part * 2; // 0 for top, 2 for bottom
                let n = ctx.mv_ctx.get_neighbors_slice(
                    mb_x,
                    mb_y,
                    0,
                    blk_y,
                    4,
                    Some(&ctx.slice_table),
                    ctx.current_slice,
                );
                trace!(mb_x, mb_y, part, ref_idx, blk_y,
                    mv_a = ?n.mv_a, ref_a = n.ref_a, a_avail = n.a_avail,
                    mv_b = ?n.mv_b, ref_b = n.ref_b, b_avail = n.b_avail,
                    mv_c = ?n.mv_c, ref_c = n.ref_c, c_avail = n.c_avail,
                    ref_pics_len = ref_pics.len(),
                    "16x8 neighbors");
                let mvp = mvpred::predict_mv_16x8(
                    n.mv_a,
                    n.mv_b,
                    n.mv_c,
                    n.ref_a,
                    n.ref_b,
                    n.ref_c,
                    ref_idx as i8,
                    n.a_avail,
                    n.b_avail,
                    n.c_avail,
                    part == 0,
                );
                let mv = [
                    mvp[0].wrapping_add(mb.mvd_l0[part as usize][0]),
                    mvp[1].wrapping_add(mb.mvd_l0[part as usize][1]),
                ];

                trace!(mb_x, mb_y, part, mvp = ?mvp, mvd = ?mb.mvd_l0[part as usize], mv = ?mv, ref_idx, "16x8 MV");

                let ref_pic = &ref_pics[ref_idx.min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, 0, blk_y * 4, 16, 8, mv);
                if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                    apply_weight_p(ctx, slice_hdr, mb_x, mb_y, 0, blk_y * 4, 16, 8, ref_idx);
                }

                // Fill MV context for the 8 4x4 blocks in this partition
                for by in blk_y..blk_y + 2 {
                    for bx in 0..4u32 {
                        let blk_idx = (bx + by * 4) as usize;
                        ctx.mv_ctx.set(mb_x, mb_y, blk_idx, mv, ref_idx as i8);
                    }
                }
            }
        }
        2 => {
            // P_L0_L0_8x16: two 8x16 partitions
            for part in 0..2u32 {
                let ref_idx = mb.ref_idx_l0[part as usize].max(0) as usize;
                let blk_x = part * 2; // 0 for left, 2 for right
                let n = ctx.mv_ctx.get_neighbors_slice(
                    mb_x,
                    mb_y,
                    blk_x,
                    0,
                    2,
                    Some(&ctx.slice_table),
                    ctx.current_slice,
                );
                let mvp = mvpred::predict_mv_8x16(
                    n.mv_a,
                    n.mv_b,
                    n.mv_c,
                    n.ref_a,
                    n.ref_b,
                    n.ref_c,
                    ref_idx as i8,
                    n.a_avail,
                    n.b_avail,
                    n.c_avail,
                    part == 0,
                );
                let mv = [
                    mvp[0].wrapping_add(mb.mvd_l0[part as usize][0]),
                    mvp[1].wrapping_add(mb.mvd_l0[part as usize][1]),
                ];

                let ref_pic = &ref_pics[ref_idx.min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, blk_x * 4, 0, 8, 16, mv);
                if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                    apply_weight_p(ctx, slice_hdr, mb_x, mb_y, blk_x * 4, 0, 8, 16, ref_idx);
                }

                // Fill MV context for the 8 4x4 blocks in this partition
                for by in 0..4u32 {
                    for bx in blk_x..blk_x + 2 {
                        let blk_idx = (bx + by * 4) as usize;
                        ctx.mv_ctx.set(mb_x, mb_y, blk_idx, mv, ref_idx as i8);
                    }
                }
            }
        }
        3 | 4 => {
            // P_8x8 / P_8x8ref0: four 8x8 sub-partitions
            // Each sub-partition may be further divided (8x4, 4x8, 4x4).
            // For simplicity, we handle 8x8 sub-partition type (sub_mb_type == 0)
            // and treat 8x4, 4x8, 4x4 sub-partitions as 8x8 with the same MV.
            for i8x8 in 0..4usize {
                let ref_idx = mb.ref_idx_l0[i8x8].max(0) as usize;
                let part_x = (i8x8 % 2) as u32 * 2; // 0 or 2 in 4x4 block units
                let part_y = (i8x8 / 2) as u32 * 2; // 0 or 2

                let sub_type = mb.sub_mb_type[i8x8];
                trace!(mb_x, mb_y, i8x8, sub_type, ref_idx, "P_8x8 sub");

                match sub_type {
                    0 => {
                        // 8x8 sub-partition
                        let n = ctx.mv_ctx.get_neighbors_slice(
                            mb_x,
                            mb_y,
                            part_x,
                            part_y,
                            2,
                            Some(&ctx.slice_table),
                            ctx.current_slice,
                        );
                        trace!(
                            mb_x, mb_y, i8x8, part_x, part_y,
                            mv_a = ?n.mv_a, ref_a = n.ref_a, a_avail = n.a_avail,
                            mv_b = ?n.mv_b, ref_b = n.ref_b, b_avail = n.b_avail,
                            mv_c = ?n.mv_c, ref_c = n.ref_c, c_avail = n.c_avail,
                            ref_idx, "P_8x8 neighbors"
                        );
                        let mvp = mvpred::predict_mv(
                            n.mv_a,
                            n.mv_b,
                            n.mv_c,
                            n.ref_a,
                            n.ref_b,
                            n.ref_c,
                            ref_idx as i8,
                            n.a_avail,
                            n.b_avail,
                            n.c_avail,
                        );
                        let mvd_idx = i8x8 * 4;
                        let mv = [
                            mvp[0].wrapping_add(mb.mvd_l0[mvd_idx][0]),
                            mvp[1].wrapping_add(mb.mvd_l0[mvd_idx][1]),
                        ];
                        trace!(mb_x, mb_y, i8x8, mvp = ?mvp, mvd = ?mb.mvd_l0[mvd_idx], mv = ?mv, "P_8x8 MV");

                        let ref_pic = &ref_pics[ref_idx.min(ref_pics.len() - 1)];
                        apply_mc_partition(
                            ctx,
                            ref_pic,
                            mb_x,
                            mb_y,
                            part_x * 4,
                            part_y * 4,
                            8,
                            8,
                            mv,
                        );
                        if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                            apply_weight_p(
                                ctx,
                                slice_hdr,
                                mb_x,
                                mb_y,
                                part_x * 4,
                                part_y * 4,
                                8,
                                8,
                                ref_idx,
                            );
                        }

                        for by in part_y..part_y + 2 {
                            for bx in part_x..part_x + 2 {
                                ctx.mv_ctx.set(
                                    mb_x,
                                    mb_y,
                                    (bx + by * 4) as usize,
                                    mv,
                                    ref_idx as i8,
                                );
                            }
                        }
                    }
                    1 => {
                        // 8x4 sub-partition: two 8x4 blocks
                        for sub in 0..2u32 {
                            let sub_y = part_y + sub;
                            let n = ctx.mv_ctx.get_neighbors_slice(
                                mb_x,
                                mb_y,
                                part_x,
                                sub_y,
                                2,
                                Some(&ctx.slice_table),
                                ctx.current_slice,
                            );
                            let mvp = mvpred::predict_mv(
                                n.mv_a,
                                n.mv_b,
                                n.mv_c,
                                n.ref_a,
                                n.ref_b,
                                n.ref_c,
                                ref_idx as i8,
                                n.a_avail,
                                n.b_avail,
                                n.c_avail,
                            );
                            let mvd_idx = i8x8 * 4 + sub as usize;
                            let mv = [
                                mvp[0].wrapping_add(mb.mvd_l0[mvd_idx][0]),
                                mvp[1].wrapping_add(mb.mvd_l0[mvd_idx][1]),
                            ];
                            trace!(mb_x, mb_y, i8x8, sub, part_x, sub_y,
                                mv_a = ?n.mv_a, ref_a = n.ref_a, a_avail = n.a_avail,
                                mv_b = ?n.mv_b, ref_b = n.ref_b, b_avail = n.b_avail,
                                mv_c = ?n.mv_c, ref_c = n.ref_c, c_avail = n.c_avail,
                                mvp = ?mvp, mvd = ?mb.mvd_l0[mvd_idx], mv = ?mv, "P_8x4 MV");

                            let ref_pic = &ref_pics[ref_idx.min(ref_pics.len() - 1)];
                            apply_mc_partition(
                                ctx,
                                ref_pic,
                                mb_x,
                                mb_y,
                                part_x * 4,
                                sub_y * 4,
                                8,
                                4,
                                mv,
                            );
                            if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                                apply_weight_p(
                                    ctx,
                                    slice_hdr,
                                    mb_x,
                                    mb_y,
                                    part_x * 4,
                                    sub_y * 4,
                                    8,
                                    4,
                                    ref_idx,
                                );
                            }

                            for bx in part_x..part_x + 2 {
                                ctx.mv_ctx.set(
                                    mb_x,
                                    mb_y,
                                    (bx + sub_y * 4) as usize,
                                    mv,
                                    ref_idx as i8,
                                );
                            }
                        }
                    }
                    2 => {
                        // 4x8 sub-partition: two 4x8 blocks
                        for sub in 0..2u32 {
                            let sub_x = part_x + sub;
                            let n = ctx.mv_ctx.get_neighbors_slice(
                                mb_x,
                                mb_y,
                                sub_x,
                                part_y,
                                1,
                                Some(&ctx.slice_table),
                                ctx.current_slice,
                            );
                            let mvp = mvpred::predict_mv(
                                n.mv_a,
                                n.mv_b,
                                n.mv_c,
                                n.ref_a,
                                n.ref_b,
                                n.ref_c,
                                ref_idx as i8,
                                n.a_avail,
                                n.b_avail,
                                n.c_avail,
                            );
                            let mvd_idx = i8x8 * 4 + sub as usize;
                            let mv = [
                                mvp[0].wrapping_add(mb.mvd_l0[mvd_idx][0]),
                                mvp[1].wrapping_add(mb.mvd_l0[mvd_idx][1]),
                            ];
                            trace!(mb_x, mb_y, i8x8, sub, sub_x, part_y,
                                mv_a = ?n.mv_a, ref_a = n.ref_a, a_avail = n.a_avail,
                                mv_b = ?n.mv_b, ref_b = n.ref_b, b_avail = n.b_avail,
                                mv_c = ?n.mv_c, ref_c = n.ref_c, c_avail = n.c_avail,
                                mvp = ?mvp, mvd = ?mb.mvd_l0[mvd_idx], mv = ?mv, "P_4x8 MV");

                            let ref_pic = &ref_pics[ref_idx.min(ref_pics.len() - 1)];
                            apply_mc_partition(
                                ctx,
                                ref_pic,
                                mb_x,
                                mb_y,
                                sub_x * 4,
                                part_y * 4,
                                4,
                                8,
                                mv,
                            );
                            if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                                apply_weight_p(
                                    ctx,
                                    slice_hdr,
                                    mb_x,
                                    mb_y,
                                    sub_x * 4,
                                    part_y * 4,
                                    4,
                                    8,
                                    ref_idx,
                                );
                            }

                            for by in part_y..part_y + 2 {
                                ctx.mv_ctx.set(
                                    mb_x,
                                    mb_y,
                                    (sub_x + by * 4) as usize,
                                    mv,
                                    ref_idx as i8,
                                );
                            }
                        }
                    }
                    3 => {
                        // 4x4 sub-partition: four 4x4 blocks
                        for sub in 0..4u32 {
                            let sub_x = part_x + (sub % 2);
                            let sub_y = part_y + (sub / 2);
                            let n = ctx.mv_ctx.get_neighbors_slice(
                                mb_x,
                                mb_y,
                                sub_x,
                                sub_y,
                                1,
                                Some(&ctx.slice_table),
                                ctx.current_slice,
                            );
                            let mvp = mvpred::predict_mv(
                                n.mv_a,
                                n.mv_b,
                                n.mv_c,
                                n.ref_a,
                                n.ref_b,
                                n.ref_c,
                                ref_idx as i8,
                                n.a_avail,
                                n.b_avail,
                                n.c_avail,
                            );
                            let mvd_idx = i8x8 * 4 + sub as usize;
                            let mv = [
                                mvp[0].wrapping_add(mb.mvd_l0[mvd_idx][0]),
                                mvp[1].wrapping_add(mb.mvd_l0[mvd_idx][1]),
                            ];

                            let ref_pic = &ref_pics[ref_idx.min(ref_pics.len() - 1)];
                            apply_mc_partition(
                                ctx,
                                ref_pic,
                                mb_x,
                                mb_y,
                                sub_x * 4,
                                sub_y * 4,
                                4,
                                4,
                                mv,
                            );
                            if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                                apply_weight_p(
                                    ctx,
                                    slice_hdr,
                                    mb_x,
                                    mb_y,
                                    sub_x * 4,
                                    sub_y * 4,
                                    4,
                                    4,
                                    ref_idx,
                                );
                            }

                            ctx.mv_ctx.set(
                                mb_x,
                                mb_y,
                                (sub_x + sub_y * 4) as usize,
                                mv,
                                ref_idx as i8,
                            );
                        }
                    }
                    _ => {
                        // Invalid sub_mb_type — fill gray
                        fill_mb_gray(ctx, mb_x, mb_y);
                    }
                }
            }
        }
        _ => {
            // Unknown inter mb_type — fill gray
            fill_mb_gray(ctx, mb_x, mb_y);
            for blk in 0..16 {
                ctx.mv_ctx.set(mb_x, mb_y, blk, [0, 0], 0);
            }
        }
    }

    {
        let (luma_off, _stride) = luma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
        let mut mc_row0 = [0u8; 16];
        mc_row0.copy_from_slice(&ctx.pic.y[luma_off..luma_off + 16]);
        trace!(mb_x, mb_y, row0 = ?mc_row0, "inter MC luma");
    }

    // Add residual on top of the motion-compensated prediction
    let cbp_luma = mb.cbp & 0x0F;
    if cbp_luma != 0 {
        if mb.transform_size_8x8_flag {
            // 8x8 transform inter residual (High profile)
            for (i8x8, &(bx, by)) in BLOCK_8X8_OFFSET.iter().enumerate() {
                if cbp_luma & (1 << i8x8) != 0 {
                    let cqm = 3; // inter
                    let dequant_table = &ctx.dequant8.coeffs[cqm][qp as usize];
                    dequant::dequant_8x8(&mut mb.luma_8x8_coeffs[i8x8], dequant_table);
                    trace!(
                        mb_x,
                        mb_y,
                        block_idx = i8x8,
                        qp,
                        cqm,
                        coeff_sum = mb.luma_8x8_coeffs[i8x8]
                            .iter()
                            .map(|&c| c.unsigned_abs() as u32)
                            .sum::<u32>(),
                        dc = mb.luma_8x8_coeffs[i8x8][0],
                        "DEQUANT"
                    );
                    // bx/by are pixel offsets within MB (0 or 8)
                    let blk_x_4 = bx / 4; // 0 or 2 in 4x4-block units
                    let blk_y_4 = by / 4;
                    let (offset, field_stride) =
                        luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x_4, blk_y_4, ctx.mb_field);
                    idct::idct8x8_add(
                        &mut ctx.pic.y[offset..],
                        field_stride,
                        &mut mb.luma_8x8_coeffs[i8x8],
                    );
                    {
                        let mut post_sum = 0u32;
                        for dy in 0..8usize {
                            for dx in 0..8usize {
                                post_sum = post_sum.wrapping_add(
                                    ctx.pic.y[offset + dy * field_stride + dx] as u32,
                                );
                            }
                        }
                        trace!(mb_x, mb_y, i8x8, post_sum, "POST_IDCT_8X8");
                    }
                }
            }
        } else {
            let dq_table = &ctx.dequant4.coeffs[3][qp as usize]; // inter Y
            for (block, &(blk_x, blk_y)) in BLOCK_INDEX_TO_XY.iter().enumerate() {
                let group_8x8 = (blk_y / 2) * 2 + (blk_x / 2);
                if cbp_luma & (1 << group_8x8) != 0 {
                    let raster_idx = block_to_raster(block);
                    dequant::dequant_4x4(&mut mb.luma_coeffs[raster_idx], dq_table);
                    trace!(
                        mb_x,
                        mb_y,
                        block_idx = block,
                        qp,
                        coeff_sum = mb.luma_coeffs[raster_idx]
                            .iter()
                            .map(|&c| c.unsigned_abs() as u32)
                            .sum::<u32>(),
                        dc = mb.luma_coeffs[raster_idx][0],
                        "DEQUANT"
                    );
                    let (offset, stride) =
                        luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y, ctx.mb_field);
                    idct::idct4x4_add(
                        &mut ctx.pic.y[offset..],
                        stride,
                        &mut mb.luma_coeffs[raster_idx],
                    );
                }
            }
        }
    }

    {
        let (luma_off, _stride) = luma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
        let mut final_row0 = [0u8; 16];
        final_row0.copy_from_slice(&ctx.pic.y[luma_off..luma_off + 16]);
        trace!(mb_x, mb_y, row0 = ?final_row0, "inter final luma");
    }

    // Decode chroma for inter MB (same as intra, but prediction is from MC)
    decode_chroma_inter(ctx, mb, mb_x, mb_y, chroma_qp);
}

/// Apply motion compensation for one partition block.
///
/// Copies the motion-compensated luma and chroma prediction from `ref_pic` into
/// the destination picture buffer at (mb_x*16 + px_offset_x, mb_y*16 + px_offset_y).
#[allow(clippy::too_many_arguments)]
fn apply_mc_partition(
    ctx: &mut FrameDecodeContext,
    ref_pic: &SharedPicture,
    mb_x: u32,
    mb_y: u32,
    px_offset_x: u32,
    px_offset_y: u32,
    block_w: usize,
    block_h: usize,
    mv: [i16; 2],
) {
    // For MV-based reference lookup, use frame-mode coordinates (not field-adjusted).
    let dst_y = (mb_y * 16 + px_offset_y) as i32;
    let mvy = mv[1] as i32;

    // Compute the highest reference pixel row needed.
    // MV is quarter-pel; 6-tap luma filter reads 2 pixels above + 3 below.
    let ref_y_bottom = dst_y + (mvy >> 2) + block_h as i32 + 3;
    let needed_mb_row = (ref_y_bottom.max(0) as u32 / 16).min(ref_pic.mb_height() - 1);

    // Block until reference picture has decoded this row
    ref_pic.wait_for_row(needed_mb_row as i32);

    // SAFETY: wait_for_row guarantees the needed rows are published.
    let rp = unsafe { ref_pic.data() };

    // Compute field-aware destination offset and stride.
    // In field mode, the dst stride is doubled and the offset is adjusted for bottom field.
    // The reference picture is always frame-mode (full-frame stride), so ref addressing is unchanged.
    let mb_field = ctx.mb_field;
    let (luma_mb_off, luma_dst_stride) = luma_mb_offset(&ctx.pic, mb_x, mb_y, mb_field);
    let luma_offset = luma_mb_off + px_offset_y as usize * luma_dst_stride + px_offset_x as usize;

    let dst_x = (mb_x * 16 + px_offset_x) as i32;

    // Quarter-pixel MV components
    let mvx = mv[0] as i32;

    // Luma: quarter-pixel precision
    let luma_ref_x = dst_x + (mvx >> 2);
    let luma_ref_y = dst_y + (mvy >> 2);
    let luma_dx = (mvx & 3) as u8;
    let luma_dy = (mvy & 3) as u8;

    mc::mc_luma(
        &mut ctx.pic.y[luma_offset..],
        luma_dst_stride,
        &rp.y,
        rp.y_stride,
        luma_ref_x,
        luma_ref_y,
        luma_dx,
        luma_dy,
        block_w,
        block_h,
        rp.width,
        rp.height,
    );

    // Chroma: eighth-pixel precision (MV divided by 2 with rounding)
    let chroma_w = block_w / 2;
    let chroma_h = block_h / 2;
    if chroma_w == 0 || chroma_h == 0 {
        return; // Partitions smaller than 4 pixels wide don't have separate chroma
    }

    let (chroma_mb_off, chroma_dst_stride) = chroma_mb_offset(&ctx.pic, mb_x, mb_y, mb_field);
    let chroma_offset =
        chroma_mb_off + (px_offset_y / 2) as usize * chroma_dst_stride + (px_offset_x / 2) as usize;

    // For MV-based reference lookup, use frame-mode coordinates.
    let chroma_dst_x = (mb_x * 8 + px_offset_x / 2) as i32;
    let chroma_dst_y = (mb_y * 8 + px_offset_y / 2) as i32;

    let cmvx = mvx;
    let cmvy = mvy;
    let chroma_ref_x = chroma_dst_x + (cmvx >> 3);
    let chroma_ref_y = chroma_dst_y + (cmvy >> 3);
    let chroma_dx = (cmvx & 7) as u8;
    let chroma_dy = (cmvy & 7) as u8;

    mc::mc_chroma(
        &mut ctx.pic.u[chroma_offset..],
        chroma_dst_stride,
        &rp.u,
        rp.uv_stride,
        chroma_ref_x,
        chroma_ref_y,
        chroma_dx,
        chroma_dy,
        chroma_w,
        chroma_h,
        rp.width / 2,
        rp.height / 2,
    );

    mc::mc_chroma(
        &mut ctx.pic.v[chroma_offset..],
        chroma_dst_stride,
        &rp.v,
        rp.uv_stride,
        chroma_ref_x,
        chroma_ref_y,
        chroma_dx,
        chroma_dy,
        chroma_w,
        chroma_h,
        rp.width / 2,
        rp.height / 2,
    );
}

/// Decode a P_SKIP macroblock.
///
/// P_SKIP macroblocks have no residual, no QP delta, and no bitstream data.
/// The motion vector is derived from neighbors (A=left, B=top, C=top-right).
/// QP carries forward from the previous macroblock.
///
/// Reference: ITU-T H.264 Section 7.4.5, 8.4.1.1
pub fn decode_skip_mb(
    ctx: &mut FrameDecodeContext,
    slice_hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    ref_pics: &[Arc<SharedPicture>],
    _ref_pics_l1: &[Arc<SharedPicture>],
) {
    if ref_pics.is_empty() {
        fill_mb_gray(ctx, mb_x, mb_y);
        for blk in 0..16 {
            ctx.mv_ctx.set(mb_x, mb_y, blk, [0, 0], 0);
        }
    } else {
        // Compute skip MV from neighbors
        let n = ctx.mv_ctx.get_neighbors_slice(
            mb_x,
            mb_y,
            0,
            0,
            4,
            Some(&ctx.slice_table),
            ctx.current_slice,
        );
        trace!(
            mb_x, mb_y,
            mv_a = ?n.mv_a, ref_a = n.ref_a, a_avail = n.a_avail,
            mv_b = ?n.mv_b, ref_b = n.ref_b, b_avail = n.b_avail,
            mv_c = ?n.mv_c, ref_c = n.ref_c, c_avail = n.c_avail,
            "P_SKIP neighbors"
        );
        let mv = mvpred::predict_mv_skip_full(
            n.mv_a, n.mv_b, n.mv_c, n.ref_a, n.ref_b, n.ref_c, n.a_avail, n.b_avail, n.c_avail,
        );
        trace!(mb_x, mb_y, mv = ?mv, "P_SKIP MV");

        // Apply motion compensation from ref_pics[0]
        apply_mc_partition(ctx, &ref_pics[0], mb_x, mb_y, 0, 0, 16, 16, mv);
        if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
            apply_weight_p(ctx, slice_hdr, mb_x, mb_y, 0, 0, 16, 16, 0);
        }

        // Fill MV context for all 16 4x4 blocks
        for blk in 0..16 {
            ctx.mv_ctx.set(mb_x, mb_y, blk, mv, 0);
        }
    }

    // Record slice ownership, update neighbor context, and store deblock info.
    finalize_skip_mb(ctx, mb_x, mb_y, false);

    let _ = slice_hdr; // reserved for future use (e.g. weighted prediction)
}

/// Decode a B_Skip macroblock (direct prediction, no residual).
///
/// B_Skip uses direct prediction (spatial or temporal based on slice header)
/// for both L0 and L1 MVs, then averages the two MC predictions.
pub fn decode_b_skip_mb(
    ctx: &mut FrameDecodeContext,
    slice_hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    ref_pics: &[Arc<SharedPicture>],
    ref_pics_l1: &[Arc<SharedPicture>],
) {
    if ref_pics.is_empty() || ref_pics_l1.is_empty() {
        fill_mb_gray(ctx, mb_x, mb_y);
        for blk in 0..16 {
            ctx.mv_ctx.set(mb_x, mb_y, blk, [0, 0], 0);
            ctx.mv_ctx.set_l1(mb_x, mb_y, blk, [0, 0], 0);
        }
    } else {
        // Compute direct MVs per 4x4 block
        let direct = if slice_hdr.direct_spatial_mv_pred_flag {
            pred_spatial_direct(ctx, mb_x, mb_y)
        } else {
            pred_temporal_direct(ctx, mb_x, mb_y)
        };

        for (i4, &(mv_l0, ref_l0, mv_l1, ref_l1)) in direct.iter().enumerate() {
            let bx = (i4 % 4) as u32;
            let by = (i4 / 4) as u32;
            let use_l0 = ref_l0 >= 0;
            let use_l1 = ref_l1 >= 0;
            let px_x = bx * 4;
            let px_y = by * 4;

            if use_l0 && use_l1 {
                apply_mc_bi_partition(
                    ctx,
                    ref_pics,
                    ref_pics_l1,
                    mb_x,
                    mb_y,
                    px_x,
                    px_y,
                    4,
                    4,
                    mv_l0,
                    ref_l0,
                    mv_l1,
                    ref_l1,
                    slice_hdr,
                );
            } else if use_l0 {
                let ref_pic = &ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 4, 4, mv_l0);
                if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                    apply_weight_list(
                        ctx,
                        slice_hdr,
                        mb_x,
                        mb_y,
                        px_x,
                        px_y,
                        4,
                        4,
                        ref_l0.max(0) as usize,
                        0,
                    );
                }
            } else if use_l1 {
                let ref_pic = &ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 4, 4, mv_l1);
                if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                    apply_weight_list(
                        ctx,
                        slice_hdr,
                        mb_x,
                        mb_y,
                        px_x,
                        px_y,
                        4,
                        4,
                        ref_l1.max(0) as usize,
                        1,
                    );
                }
            }

            ctx.mv_ctx.set(mb_x, mb_y, i4, mv_l0, ref_l0);
            ctx.mv_ctx.set_l1(mb_x, mb_y, i4, mv_l1, ref_l1);
        }
    }

    // Record slice ownership, update neighbor context, and store deblock info.
    finalize_skip_mb(ctx, mb_x, mb_y, true);
}

/// Decode a B-frame inter macroblock.
///
/// Handles all B-frame partition types: B_Direct_16x16, B_L0/L1/Bi 16x16,
/// B_L0/L1/Bi 16x8/8x16, and B_8x8.
#[allow(clippy::too_many_arguments)]
fn decode_b_inter_mb(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    slice_hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: [u8; 2],
    ref_pics: &[Arc<SharedPicture>],
    ref_pics_l1: &[Arc<SharedPicture>],
) {
    if ref_pics.is_empty() && ref_pics_l1.is_empty() {
        fill_mb_gray(ctx, mb_x, mb_y);
        for blk in 0..16 {
            ctx.mv_ctx.set(mb_x, mb_y, blk, [0, 0], 0);
            ctx.mv_ctx.set_l1(mb_x, mb_y, blk, [0, 0], 0);
        }
        add_b_residual(ctx, mb, mb_x, mb_y, qp, chroma_qp);
        return;
    }

    if mb.is_direct {
        // B_Direct_16x16: dispatch based on direct_spatial_mv_pred_flag
        let direct = if slice_hdr.direct_spatial_mv_pred_flag {
            pred_spatial_direct(ctx, mb_x, mb_y)
        } else {
            pred_temporal_direct(ctx, mb_x, mb_y)
        };

        // Apply MC per 4x4 block. Adjacent blocks with identical MVs could be
        // coalesced into larger partitions, but for correctness (especially
        // when direct_8x8_inference_flag=0) we apply per-4x4.
        for (i4, &(mv_l0, ref_l0, mv_l1, ref_l1)) in direct.iter().enumerate() {
            let bx = (i4 % 4) as u32;
            let by = (i4 / 4) as u32;
            let use_l0 = ref_l0 >= 0 && !ref_pics.is_empty();
            let use_l1 = ref_l1 >= 0 && !ref_pics_l1.is_empty();
            let px_x = bx * 4;
            let px_y = by * 4;

            if use_l0 && use_l1 {
                apply_mc_bi_partition(
                    ctx,
                    ref_pics,
                    ref_pics_l1,
                    mb_x,
                    mb_y,
                    px_x,
                    px_y,
                    4,
                    4,
                    mv_l0,
                    ref_l0,
                    mv_l1,
                    ref_l1,
                    slice_hdr,
                );
            } else if use_l0 {
                let ref_pic = &ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 4, 4, mv_l0);
                if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                    apply_weight_list(
                        ctx,
                        slice_hdr,
                        mb_x,
                        mb_y,
                        px_x,
                        px_y,
                        4,
                        4,
                        ref_l0.max(0) as usize,
                        0,
                    );
                }
            } else if use_l1 {
                let ref_pic = &ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 4, 4, mv_l1);
                if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                    apply_weight_list(
                        ctx,
                        slice_hdr,
                        mb_x,
                        mb_y,
                        px_x,
                        px_y,
                        4,
                        4,
                        ref_l1.max(0) as usize,
                        1,
                    );
                }
            }

            ctx.mv_ctx.set(mb_x, mb_y, i4, mv_l0, ref_l0);
            ctx.mv_ctx.set_l1(mb_x, mb_y, i4, mv_l1, ref_l1);
        }
    } else if mb.mb_type == 22 {
        // B_8x8: per-8x8-partition decode with sub_mb_type
        decode_b_8x8_mb(ctx, mb, slice_hdr, mb_x, mb_y, ref_pics, ref_pics_l1);
    } else {
        // Non-direct B-frame partitions (16x16, 16x8, 8x16)
        let part_size = mb.b_part_size;
        let part_count = mb.partition_count.min(2) as usize;

        for part in 0..part_count {
            let uses_l0 = mb.b_list_flags[part][0];
            let uses_l1 = mb.b_list_flags[part][1];
            let ref_l0 = if uses_l0 {
                mb.ref_idx_l0[part].max(0)
            } else {
                -1
            };
            let ref_l1 = if uses_l1 {
                mb.ref_idx_l1[part].max(0)
            } else {
                -1
            };

            // Compute partition geometry
            let (blk_x, blk_y, pw, ph) = match part_size {
                0 => (0u32, 0u32, 16usize, 16usize), // 16x16
                1 => (0, part as u32 * 2, 16, 8),    // 16x8
                2 => (part as u32 * 2, 0, 8, 16),    // 8x16
                _ => (0, 0, 16, 16),
            };

            let part_width_4x4 = (pw / 4) as u32;

            // Compute L0 MV
            let mv_l0 = if uses_l0 {
                let n = ctx.mv_ctx.get_neighbors_slice(
                    mb_x,
                    mb_y,
                    blk_x,
                    blk_y,
                    part_width_4x4,
                    Some(&ctx.slice_table),
                    ctx.current_slice,
                );
                let mvp = match part_size {
                    1 => mvpred::predict_mv_16x8(
                        n.mv_a,
                        n.mv_b,
                        n.mv_c,
                        n.ref_a,
                        n.ref_b,
                        n.ref_c,
                        ref_l0,
                        n.a_avail,
                        n.b_avail,
                        n.c_avail,
                        part == 0,
                    ),
                    2 => mvpred::predict_mv_8x16(
                        n.mv_a,
                        n.mv_b,
                        n.mv_c,
                        n.ref_a,
                        n.ref_b,
                        n.ref_c,
                        ref_l0,
                        n.a_avail,
                        n.b_avail,
                        n.c_avail,
                        part == 0,
                    ),
                    _ => mvpred::predict_mv(
                        n.mv_a, n.mv_b, n.mv_c, n.ref_a, n.ref_b, n.ref_c, ref_l0, n.a_avail,
                        n.b_avail, n.c_avail,
                    ),
                };
                [
                    mvp[0].wrapping_add(mb.mvd_l0[part][0]),
                    mvp[1].wrapping_add(mb.mvd_l0[part][1]),
                ]
            } else {
                [0, 0]
            };

            // Store L0 MV context before computing L1 (so L1 neighbors can see it)
            let blk_h_4x4 = (ph / 4) as u32;
            for by in blk_y..blk_y + blk_h_4x4 {
                for bx in blk_x..blk_x + part_width_4x4 {
                    ctx.mv_ctx
                        .set(mb_x, mb_y, (bx + by * 4) as usize, mv_l0, ref_l0);
                }
            }

            // Compute L1 MV using L1 neighbor context
            let mv_l1 = if uses_l1 {
                let n = ctx.mv_ctx.get_neighbors_list(
                    mb_x,
                    mb_y,
                    blk_x,
                    blk_y,
                    part_width_4x4,
                    Some(&ctx.slice_table),
                    ctx.current_slice,
                    1,
                );
                let mvp = match part_size {
                    1 => mvpred::predict_mv_16x8(
                        n.mv_a,
                        n.mv_b,
                        n.mv_c,
                        n.ref_a,
                        n.ref_b,
                        n.ref_c,
                        ref_l1,
                        n.a_avail,
                        n.b_avail,
                        n.c_avail,
                        part == 0,
                    ),
                    2 => mvpred::predict_mv_8x16(
                        n.mv_a,
                        n.mv_b,
                        n.mv_c,
                        n.ref_a,
                        n.ref_b,
                        n.ref_c,
                        ref_l1,
                        n.a_avail,
                        n.b_avail,
                        n.c_avail,
                        part == 0,
                    ),
                    _ => mvpred::predict_mv(
                        n.mv_a, n.mv_b, n.mv_c, n.ref_a, n.ref_b, n.ref_c, ref_l1, n.a_avail,
                        n.b_avail, n.c_avail,
                    ),
                };
                [
                    mvp[0].wrapping_add(mb.mvd_l1[part][0]),
                    mvp[1].wrapping_add(mb.mvd_l1[part][1]),
                ]
            } else {
                [0, 0]
            };

            // Store L1 MV context
            for by in blk_y..blk_y + blk_h_4x4 {
                for bx in blk_x..blk_x + part_width_4x4 {
                    ctx.mv_ctx
                        .set_l1(mb_x, mb_y, (bx + by * 4) as usize, mv_l1, ref_l1);
                }
            }

            // Apply motion compensation
            let px_x = blk_x * 4;
            let px_y = blk_y * 4;

            if uses_l0 && uses_l1 && !ref_pics.is_empty() && !ref_pics_l1.is_empty() {
                apply_mc_bi_partition(
                    ctx,
                    ref_pics,
                    ref_pics_l1,
                    mb_x,
                    mb_y,
                    px_x,
                    px_y,
                    pw,
                    ph,
                    mv_l0,
                    ref_l0,
                    mv_l1,
                    ref_l1,
                    slice_hdr,
                );
            } else if uses_l0 && !ref_pics.is_empty() {
                let ref_pic = &ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, pw, ph, mv_l0);
                if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                    apply_weight_list(
                        ctx,
                        slice_hdr,
                        mb_x,
                        mb_y,
                        px_x,
                        px_y,
                        pw,
                        ph,
                        ref_l0.max(0) as usize,
                        0,
                    );
                }
            } else if uses_l1 && !ref_pics_l1.is_empty() {
                let ref_pic = &ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, pw, ph, mv_l1);
                if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                    apply_weight_list(
                        ctx,
                        slice_hdr,
                        mb_x,
                        mb_y,
                        px_x,
                        px_y,
                        pw,
                        ph,
                        ref_l1.max(0) as usize,
                        1,
                    );
                }
            }
        }
    }

    // Add residual
    add_b_residual(ctx, mb, mb_x, mb_y, qp, chroma_qp);
}

/// Decode a B_8x8 macroblock: each 8x8 partition has its own sub_mb_type.
///
/// Sub-partition types: B_Direct_8x8 (0) uses spatial direct prediction;
/// B_L0_8x8 (1), B_L1_8x8 (2), B_Bi_8x8 (3) use explicit MC;
/// Sub-types 4-12 have smaller sub-partitions (8x4, 4x8, 4x4) but are
/// handled as 8x8 with the first sub-partition's MV for simplicity.
///
/// Reference: FFmpeg h264_mb.c hl_decode_mb_predict_luma B_8x8 path
#[allow(clippy::too_many_arguments)]
fn decode_b_8x8_mb(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    slice_hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    ref_pics: &[Arc<SharedPicture>],
    ref_pics_l1: &[Arc<SharedPicture>],
) {
    use crate::tables::B_SUB_MB_TYPE_INFO;

    // 8x8 partition layout: [0]=(0,0), [1]=(2,0), [2]=(0,2), [3]=(2,2) in 4x4-block units
    const PART_XY: [(u32, u32); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];

    // Pre-compute spatial direct for all 4 8x8 blocks (needed if any sub_mb_type==0).
    // Neighbor computation is MB-level so results are valid for all partitions.
    let has_direct = mb.sub_mb_type.contains(&0);
    let direct = if has_direct {
        Some(if slice_hdr.direct_spatial_mv_pred_flag {
            pred_spatial_direct(ctx, mb_x, mb_y)
        } else {
            pred_temporal_direct(ctx, mb_x, mb_y)
        })
    } else {
        None
    };

    for (part, &(blk_x, blk_y)) in PART_XY.iter().enumerate() {
        let sub_type = mb.sub_mb_type[part];

        if sub_type == 0 {
            // B_Direct_8x8: per-4x4 spatial direct result
            let d = direct.as_ref().unwrap();
            for by in blk_y..blk_y + 2 {
                for bx in blk_x..blk_x + 2 {
                    let i4 = (bx + by * 4) as usize;
                    let (mv_l0, ref_l0, mv_l1, ref_l1) = d[i4];
                    let use_l0 = ref_l0 >= 0 && !ref_pics.is_empty();
                    let use_l1 = ref_l1 >= 0 && !ref_pics_l1.is_empty();
                    let px_x = bx * 4;
                    let px_y = by * 4;

                    if use_l0 && use_l1 {
                        apply_mc_bi_partition(
                            ctx,
                            ref_pics,
                            ref_pics_l1,
                            mb_x,
                            mb_y,
                            px_x,
                            px_y,
                            4,
                            4,
                            mv_l0,
                            ref_l0,
                            mv_l1,
                            ref_l1,
                            slice_hdr,
                        );
                    } else if use_l0 {
                        let ref_pic = &ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                        apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 4, 4, mv_l0);
                        if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                            apply_weight_list(
                                ctx,
                                slice_hdr,
                                mb_x,
                                mb_y,
                                px_x,
                                px_y,
                                4,
                                4,
                                ref_l0.max(0) as usize,
                                0,
                            );
                        }
                    } else if use_l1 {
                        let ref_pic = &ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                        apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 4, 4, mv_l1);
                        if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                            apply_weight_list(
                                ctx,
                                slice_hdr,
                                mb_x,
                                mb_y,
                                px_x,
                                px_y,
                                4,
                                4,
                                ref_l1.max(0) as usize,
                                1,
                            );
                        }
                    }

                    ctx.mv_ctx.set(mb_x, mb_y, i4, mv_l0, ref_l0);
                    ctx.mv_ctx.set_l1(mb_x, mb_y, i4, mv_l1, ref_l1);
                }
            }
        } else {
            // Non-direct 8x8 sub-partition with per-sub-partition MC.
            // Sub-partition layout: 0→8x8, 1→8x4, 2→4x8, 3→4x4
            let info = &B_SUB_MB_TYPE_INFO[sub_type as usize];
            let sub_count = info.0 as usize;
            let part_size = info.1;
            let uses_l0 = info.2;
            let uses_l1 = info.3;
            let ref_l0 = if uses_l0 {
                mb.ref_idx_l0[part].max(0)
            } else {
                -1
            };
            let ref_l1 = if uses_l1 {
                mb.ref_idx_l1[part].max(0)
            } else {
                -1
            };

            // Sub-partition geometry: (dx, dy, w, h) in 4x4-block units
            let sub_parts: &[(u32, u32, u32, u32)] = match part_size {
                1 => &[(0, 0, 2, 1), (0, 1, 2, 1)], // 8x4
                2 => &[(0, 0, 1, 2), (1, 0, 1, 2)], // 4x8
                3 => &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)], // 4x4
                _ => &[(0, 0, 2, 2)],               // 8x8
            };

            for (j, &(sdx, sdy, sw, sh)) in sub_parts.iter().enumerate().take(sub_count) {
                let sub_blk_x = blk_x + sdx;
                let sub_blk_y = blk_y + sdy;
                let pw = (sw * 4) as usize;
                let ph = (sh * 4) as usize;
                let mvd_idx = part * 4 + j;

                // Compute L0 MV
                let mv_l0 = if uses_l0 {
                    let n = ctx.mv_ctx.get_neighbors_slice(
                        mb_x,
                        mb_y,
                        sub_blk_x,
                        sub_blk_y,
                        sw,
                        Some(&ctx.slice_table),
                        ctx.current_slice,
                    );
                    let mvp = mvpred::predict_mv(
                        n.mv_a, n.mv_b, n.mv_c, n.ref_a, n.ref_b, n.ref_c, ref_l0, n.a_avail,
                        n.b_avail, n.c_avail,
                    );
                    trace!(
                        mb_x,
                        mb_y,
                        part,
                        j,
                        list = 0,
                        sub_blk_x,
                        sub_blk_y,
                        sw,
                        a_mv_x = n.mv_a[0],
                        a_mv_y = n.mv_a[1],
                        a_ref = n.ref_a,
                        a_avail = n.a_avail,
                        b_mv_x = n.mv_b[0],
                        b_mv_y = n.mv_b[1],
                        b_ref = n.ref_b,
                        b_avail = n.b_avail,
                        c_mv_x = n.mv_c[0],
                        c_mv_y = n.mv_c[1],
                        c_ref = n.ref_c,
                        c_avail = n.c_avail,
                        mvp_x = mvp[0],
                        mvp_y = mvp[1],
                        "B_8x8 neighbors"
                    );
                    [
                        mvp[0].wrapping_add(mb.mvd_l0[mvd_idx][0]),
                        mvp[1].wrapping_add(mb.mvd_l0[mvd_idx][1]),
                    ]
                } else {
                    [0, 0]
                };

                // Store L0 context
                for by in sub_blk_y..sub_blk_y + sh {
                    for bx in sub_blk_x..sub_blk_x + sw {
                        ctx.mv_ctx
                            .set(mb_x, mb_y, (bx + by * 4) as usize, mv_l0, ref_l0);
                    }
                }

                // Compute L1 MV
                let mv_l1 = if uses_l1 {
                    let n = ctx.mv_ctx.get_neighbors_list(
                        mb_x,
                        mb_y,
                        sub_blk_x,
                        sub_blk_y,
                        sw,
                        Some(&ctx.slice_table),
                        ctx.current_slice,
                        1,
                    );
                    let mvp = mvpred::predict_mv(
                        n.mv_a, n.mv_b, n.mv_c, n.ref_a, n.ref_b, n.ref_c, ref_l1, n.a_avail,
                        n.b_avail, n.c_avail,
                    );
                    trace!(
                        mb_x,
                        mb_y,
                        part,
                        j,
                        list = 1,
                        sub_blk_x,
                        sub_blk_y,
                        sw,
                        a_mv_x = n.mv_a[0],
                        a_mv_y = n.mv_a[1],
                        a_ref = n.ref_a,
                        a_avail = n.a_avail,
                        b_mv_x = n.mv_b[0],
                        b_mv_y = n.mv_b[1],
                        b_ref = n.ref_b,
                        b_avail = n.b_avail,
                        c_mv_x = n.mv_c[0],
                        c_mv_y = n.mv_c[1],
                        c_ref = n.ref_c,
                        c_avail = n.c_avail,
                        mvp_x = mvp[0],
                        mvp_y = mvp[1],
                        "B_8x8 neighbors"
                    );
                    [
                        mvp[0].wrapping_add(mb.mvd_l1[mvd_idx][0]),
                        mvp[1].wrapping_add(mb.mvd_l1[mvd_idx][1]),
                    ]
                } else {
                    [0, 0]
                };

                // Store L1 context
                for by in sub_blk_y..sub_blk_y + sh {
                    for bx in sub_blk_x..sub_blk_x + sw {
                        ctx.mv_ctx
                            .set_l1(mb_x, mb_y, (bx + by * 4) as usize, mv_l1, ref_l1);
                    }
                }

                trace!(
                    mb_x,
                    mb_y,
                    part,
                    j,
                    sub_type,
                    sub_blk_x,
                    sub_blk_y,
                    mv_l0_x = mv_l0[0],
                    mv_l0_y = mv_l0[1],
                    ref_l0,
                    mv_l1_x = mv_l1[0],
                    mv_l1_y = mv_l1[1],
                    ref_l1,
                    mvd_l0_x = mb.mvd_l0[mvd_idx][0],
                    mvd_l0_y = mb.mvd_l0[mvd_idx][1],
                    mvd_l1_x = mb.mvd_l1[mvd_idx][0],
                    mvd_l1_y = mb.mvd_l1[mvd_idx][1],
                    "B_8x8 sub-partition MV"
                );

                // Apply MC at sub-partition size
                let px_x = sub_blk_x * 4;
                let px_y = sub_blk_y * 4;
                if uses_l0 && uses_l1 && !ref_pics.is_empty() && !ref_pics_l1.is_empty() {
                    apply_mc_bi_partition(
                        ctx,
                        ref_pics,
                        ref_pics_l1,
                        mb_x,
                        mb_y,
                        px_x,
                        px_y,
                        pw,
                        ph,
                        mv_l0,
                        ref_l0,
                        mv_l1,
                        ref_l1,
                        slice_hdr,
                    );
                } else if uses_l0 && !ref_pics.is_empty() {
                    let ref_pic = &ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                    apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, pw, ph, mv_l0);
                    if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                        apply_weight_list(
                            ctx,
                            slice_hdr,
                            mb_x,
                            mb_y,
                            px_x,
                            px_y,
                            pw,
                            ph,
                            ref_l0.max(0) as usize,
                            0,
                        );
                    }
                } else if uses_l1 && !ref_pics_l1.is_empty() {
                    let ref_pic = &ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                    apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, pw, ph, mv_l1);
                    if slice_hdr.use_weight || slice_hdr.use_weight_chroma {
                        apply_weight_list(
                            ctx,
                            slice_hdr,
                            mb_x,
                            mb_y,
                            px_x,
                            px_y,
                            pw,
                            ph,
                            ref_l1.max(0) as usize,
                            1,
                        );
                    }
                }
            }
        }
    }
}

/// Add residual coefficients on top of B-frame MC prediction.
#[allow(clippy::too_many_arguments)]
fn add_b_residual(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: [u8; 2],
) {
    let cbp_luma = mb.cbp & 0x0F;
    if cbp_luma != 0 {
        if mb.transform_size_8x8_flag {
            // 8x8 transform inter residual (High profile)
            for (i8x8, &(bx, by)) in BLOCK_8X8_OFFSET.iter().enumerate() {
                if cbp_luma & (1 << i8x8) != 0 {
                    let cqm = 3; // inter
                    let dequant_table = &ctx.dequant8.coeffs[cqm][qp as usize];
                    dequant::dequant_8x8(&mut mb.luma_8x8_coeffs[i8x8], dequant_table);
                    trace!(
                        mb_x,
                        mb_y,
                        block_idx = i8x8,
                        qp,
                        cqm,
                        coeff_sum = mb.luma_8x8_coeffs[i8x8]
                            .iter()
                            .map(|&c| c.unsigned_abs() as u32)
                            .sum::<u32>(),
                        dc = mb.luma_8x8_coeffs[i8x8][0],
                        "DEQUANT"
                    );
                    // bx/by are pixel offsets within MB (0 or 8)
                    let blk_x_4 = bx / 4; // 0 or 2 in 4x4-block units
                    let blk_y_4 = by / 4;
                    let (offset, field_stride) =
                        luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x_4, blk_y_4, ctx.mb_field);
                    idct::idct8x8_add(
                        &mut ctx.pic.y[offset..],
                        field_stride,
                        &mut mb.luma_8x8_coeffs[i8x8],
                    );
                    {
                        let mut post_sum = 0u32;
                        for dy in 0..8usize {
                            for dx in 0..8usize {
                                post_sum = post_sum.wrapping_add(
                                    ctx.pic.y[offset + dy * field_stride + dx] as u32,
                                );
                            }
                        }
                        trace!(mb_x, mb_y, i8x8, post_sum, "POST_IDCT_8X8");
                    }
                }
            }
        } else {
            let dq_table = &ctx.dequant4.coeffs[3][qp as usize]; // inter Y
            for (block, &(blk_x, blk_y)) in BLOCK_INDEX_TO_XY.iter().enumerate() {
                let group_8x8 = (blk_y / 2) * 2 + (blk_x / 2);
                if cbp_luma & (1 << group_8x8) != 0 {
                    let raster_idx = block_to_raster(block);
                    dequant::dequant_4x4(&mut mb.luma_coeffs[raster_idx], dq_table);
                    trace!(
                        mb_x,
                        mb_y,
                        block_idx = block,
                        qp,
                        coeff_sum = mb.luma_coeffs[raster_idx]
                            .iter()
                            .map(|&c| c.unsigned_abs() as u32)
                            .sum::<u32>(),
                        dc = mb.luma_coeffs[raster_idx][0],
                        "DEQUANT"
                    );
                    let (offset, stride) =
                        luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y, ctx.mb_field);
                    idct::idct4x4_add(
                        &mut ctx.pic.y[offset..],
                        stride,
                        &mut mb.luma_coeffs[raster_idx],
                    );
                }
            }
        }
    }
    decode_chroma_inter(ctx, mb, mb_x, mb_y, chroma_qp);
}

/// Spatial direct prediction for a B-frame macroblock, returning per-8x8-block results.
///
/// Derives L0 and L1 MVs from neighbors per H.264 spec 8.4.1.2.3.
/// Returns 4 tuples (mv_l0, ref_l0, mv_l1, ref_l1), one per 8x8 partition:
/// [0]=top-left, [1]=top-right, [2]=bottom-left, [3]=bottom-right.
///
/// The col_zero_flag optimization is applied per-8x8-block, checking the
/// colocated reference frame's L0 ref_idx and MV at the corner of each 8x8.
///
/// Reference: FFmpeg h264_direct.c:pred_spatial_direct_motion (lines 199-484)
///
/// Returns per-4x4 block results (16 entries in raster order).
/// When direct_8x8_inference_flag=1, each 8x8 group of four has identical values.
/// When direct_8x8_inference_flag=0, each 4x4 may differ (col_zero_flag per-4x4).
fn pred_spatial_direct(
    ctx: &FrameDecodeContext,
    mb_x: u32,
    mb_y: u32,
) -> [([i16; 2], i8, [i16; 2], i8); 16] {
    let mut ref_idx = [-1i8; 2];
    let mut mv = [[0i16; 2]; 2];

    // For each list: ref = min(left_ref, top_ref, topright_ref) using unsigned comparison.
    // If ref >= 0, compute MV via median prediction matching the selected ref.
    // Reference: FFmpeg h264_direct.c lines 224-267
    for list in 0..2u8 {
        let n = ctx.mv_ctx.get_neighbors_list(
            mb_x,
            mb_y,
            0,
            0,
            4,
            Some(&ctx.slice_table),
            ctx.current_slice,
            list,
        );

        let ref_a_u = if n.a_avail { n.ref_a as u8 } else { u8::MAX };
        let ref_b_u = if n.b_avail { n.ref_b as u8 } else { u8::MAX };
        let ref_c_u = if n.c_avail { n.ref_c as u8 } else { u8::MAX };

        let min_ref_u = ref_a_u.min(ref_b_u).min(ref_c_u);
        let r = if min_ref_u == u8::MAX {
            -1i8
        } else {
            min_ref_u as i8
        };
        ref_idx[list as usize] = r;

        if r >= 0 {
            mv[list as usize] = mvpred::predict_mv(
                n.mv_a, n.mv_b, n.mv_c, n.ref_a, n.ref_b, n.ref_c, r, n.a_avail, n.b_avail,
                n.c_avail,
            );
        }
        // else: mv stays [0, 0], ref stays -1
    }

    // If both refs negative, set both to 0 (use both lists with zero MV).
    // Reference: FFmpeg h264_direct.c lines 268-273
    if ref_idx[0] < 0 && ref_idx[1] < 0 {
        ref_idx[0] = 0;
        ref_idx[1] = 0;
    }

    let base = (mv[0], ref_idx[0], mv[1], ref_idx[1]);

    // Fast path: if both MVs are zero, skip col_zero_flag check.
    // Reference: FFmpeg h264_direct.c lines 275-284
    if mv[0] == [0, 0] && mv[1] == [0, 0] {
        return [base; 16];
    }

    // Col_zero_flag optimization: check the colocated block in the L1[0] reference.
    // If colocated ref_idx=0 and |mv| <= 1, suppress the spatial MV for lists
    // where ref == 0.
    // Reference: FFmpeg h264_direct.c lines 424-477
    let mb_idx = (mb_y * ctx.mb_width + mb_x) as usize;
    let col_is_intra = ctx.col_mb_intra.get(mb_idx).copied().unwrap_or(true);

    let mut results = [base; 16];

    // FFmpeg h264_direct.c:374,405,443: col_zero_flag is only applied when
    // L1[0] is NOT a long-term reference. When it IS long-term, skip the
    // col_zero_flag optimization entirely (keep the spatial MV as-is).
    if !col_is_intra && !ctx.col_ref.is_empty() && !ctx.col_l1_is_long_term {
        let blk_base = mb_idx * 16;

        if ctx.direct_8x8_inference_flag {
            // Per-8x8: one colocated MV per 8x8 block
            // REF_CORNERS: top-left 4x4 of each 8x8 (for ref)
            const REF_CORNERS: [usize; 4] = [0, 2, 8, 10];
            // MV_CORNERS: bottom-right 4x4 of each 8x8 (for MV)
            const MV_CORNERS: [usize; 4] = [0, 3, 12, 15];
            // 4x4 indices within each 8x8 in raster order
            const BLOCK_8X8_TO_4X4: [[usize; 4]; 4] =
                [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]];

            for i8 in 0..4 {
                let col_ref0 = ctx
                    .col_ref
                    .get(blk_base + REF_CORNERS[i8])
                    .copied()
                    .unwrap_or(-1);

                // col_zero_flag: check colocated L0 ref first, fall back to L1.
                // Reference: FFmpeg h264_direct.c lines 443-447
                //   (l1ref0[i8] == 0 || (l1ref0[i8] < 0 && l1ref1[i8] == 0))
                let col_mv_to_check = if col_ref0 == 0 {
                    // Colocated L0 ref is 0: use L0 MV
                    Some(
                        ctx.col_mv
                            .get(blk_base + MV_CORNERS[i8])
                            .copied()
                            .unwrap_or([0, 0]),
                    )
                } else if col_ref0 < 0 {
                    // Colocated L0 ref not used: check L1 ref
                    let col_ref1 = ctx
                        .col_ref_l1
                        .get(blk_base + REF_CORNERS[i8])
                        .copied()
                        .unwrap_or(-1);
                    if col_ref1 == 0 {
                        // L1 ref is 0: use L1 MV
                        Some(
                            ctx.col_mv_l1
                                .get(blk_base + MV_CORNERS[i8])
                                .copied()
                                .unwrap_or([0, 0]),
                        )
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(col_mv_val) = col_mv_to_check
                    && col_mv_val[0].abs() <= 1
                    && col_mv_val[1].abs() <= 1
                {
                    let mut a = mv[0];
                    let mut b = mv[1];
                    if ref_idx[0] == 0 {
                        a = [0, 0];
                    }
                    if ref_idx[1] == 0 {
                        b = [0, 0];
                    }
                    for &b4 in &BLOCK_8X8_TO_4X4[i8] {
                        results[b4] = (a, ref_idx[0], b, ref_idx[1]);
                    }
                }
            }
        } else {
            // Per-4x4: each 4x4 block gets its own colocated check
            // Reference: FFmpeg h264_direct.c lines 460-477 (IS_SUB_8X8 else branch)
            for (i4, result) in results.iter_mut().enumerate() {
                let col_ref0 = ctx.col_ref.get(blk_base + i4).copied().unwrap_or(-1);

                // col_zero_flag: check colocated L0 ref first, fall back to L1.
                // Reference: FFmpeg h264_direct.c lines 443-447
                let col_mv_to_check = if col_ref0 == 0 {
                    Some(ctx.col_mv.get(blk_base + i4).copied().unwrap_or([0, 0]))
                } else if col_ref0 < 0 {
                    let col_ref1 = ctx.col_ref_l1.get(blk_base + i4).copied().unwrap_or(-1);
                    if col_ref1 == 0 {
                        Some(ctx.col_mv_l1.get(blk_base + i4).copied().unwrap_or([0, 0]))
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(col_mv_val) = col_mv_to_check
                    && col_mv_val[0].abs() <= 1
                    && col_mv_val[1].abs() <= 1
                {
                    let mut a = mv[0];
                    let mut b = mv[1];
                    if ref_idx[0] == 0 {
                        a = [0, 0];
                    }
                    if ref_idx[1] == 0 {
                        b = [0, 0];
                    }
                    *result = (a, ref_idx[0], b, ref_idx[1]);
                }
            }
        }
    }

    results
}

/// Temporal direct prediction for a B-frame macroblock.
///
/// Derives L0 and L1 MVs by scaling the colocated MB's MV from L1[0].
/// Formula: L0_MV = (scale * col_mv + 128) >> 8
///          L1_MV = L0_MV - col_mv
/// where scale = 256 * (cur_poc - l0_ref_poc) / (col_poc - col_ref_poc)
///
/// Returns per-4x4 block results matching pred_spatial_direct's format.
///
/// Reference: FFmpeg h264_direct.c:pred_temp_direct_motion (lines 486-718)
fn pred_temporal_direct(
    ctx: &FrameDecodeContext,
    mb_x: u32,
    mb_y: u32,
) -> [([i16; 2], i8, [i16; 2], i8); 16] {
    let mb_idx = (mb_y * ctx.mb_width + mb_x) as usize;
    let col_is_intra = ctx.col_mb_intra.get(mb_idx).copied().unwrap_or(true);
    let blk_base = mb_idx * 16;

    let mut results = [([0i16; 2], 0i8, [0i16; 2], 0i8); 16];

    // For intra colocated, all MVs are zero with ref_idx 0
    if col_is_intra || ctx.col_ref.is_empty() {
        return results;
    }

    // Precompute dist_scale_factor for each L0 ref
    // scale = clip(256 * td / tb, -1024, 1023) where
    //   tb = clip(cur_poc - l0_ref_poc, -128, 127)
    //   td = clip(col_poc - col_ref_poc, -128, 127)
    let compute_scale = |l0_ref_poc: i32, col_ref_poc_val: i32| -> i32 {
        let tb = (ctx.cur_poc - l0_ref_poc).clamp(-128, 127);
        let td = (ctx.col_poc - col_ref_poc_val).clamp(-128, 127);
        if td == 0 {
            return 256;
        }
        let tx = (16384 + (td.abs() >> 1)) / td;
        ((tb * tx + 32) >> 6).clamp(-1024, 1023)
    };

    if ctx.direct_8x8_inference_flag {
        // Per-8x8 block: ref uses 8x8 grid positions (top-left of each 8x8),
        // MV uses FFmpeg's x8*3 + y8*3*4 positions matching h264_direct.c
        const REF_POS: [usize; 4] = [0, 2, 8, 10];
        const MV_POS: [usize; 4] = [0, 3, 12, 15];
        const FILL: [[usize; 4]; 4] =
            [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]];

        for i8x8 in 0..4 {
            let col_ref_idx_l0 = ctx
                .col_ref
                .get(blk_base + REF_POS[i8x8])
                .copied()
                .unwrap_or(-1);

            // FFmpeg h264_direct.c: if L0 ref >= 0, use it; else fall back to L1.
            // If both are < 0, the block is intra → zero MV, ref 0.
            let (col_ref_poc_val, col_mv) = if col_ref_idx_l0 >= 0 {
                let poc = ctx
                    .col_ref_poc_l0
                    .get(col_ref_idx_l0 as usize)
                    .copied()
                    .unwrap_or(ctx.col_poc);
                let mv = ctx
                    .col_mv
                    .get(blk_base + MV_POS[i8x8])
                    .copied()
                    .unwrap_or([0, 0]);
                (poc, mv)
            } else {
                let col_ref_idx_l1 = ctx
                    .col_ref_l1
                    .get(blk_base + REF_POS[i8x8])
                    .copied()
                    .unwrap_or(-1);
                if col_ref_idx_l1 < 0 {
                    // Intra sub-block: zero MV, ref 0
                    for &blk in &FILL[i8x8] {
                        results[blk] = ([0, 0], 0, [0, 0], 0);
                    }
                    continue;
                }
                let poc = ctx
                    .col_ref_poc_l1
                    .get(col_ref_idx_l1 as usize)
                    .copied()
                    .unwrap_or(ctx.col_poc);
                let mv = ctx
                    .col_mv_l1
                    .get(blk_base + MV_POS[i8x8])
                    .copied()
                    .unwrap_or([0, 0]);
                (poc, mv)
            };

            // Map colocated ref POC to current L0 index
            let l0_ref = ctx
                .cur_l0_ref_poc
                .iter()
                .position(|&poc| poc == col_ref_poc_val)
                .unwrap_or(0);

            let l0_ref_poc = ctx.cur_l0_ref_poc.get(l0_ref).copied().unwrap_or(0);
            let scale = compute_scale(l0_ref_poc, col_ref_poc_val);

            let mv_l0 = [
                ((scale * col_mv[0] as i32 + 128) >> 8) as i16,
                ((scale * col_mv[1] as i32 + 128) >> 8) as i16,
            ];
            let mv_l1 = [mv_l0[0] - col_mv[0], mv_l0[1] - col_mv[1]];

            for &blk in &FILL[i8x8] {
                results[blk] = (mv_l0, l0_ref as i8, mv_l1, 0);
            }
        }
    } else {
        // Per-4x4 block
        for (blk, result) in results.iter_mut().enumerate() {
            let col_blk = blk_base + blk;
            let col_ref_idx_l0 = ctx.col_ref.get(col_blk).copied().unwrap_or(-1);

            // L1 fallback: if colocated L0 ref < 0, use L1 ref and MV
            let (col_ref_poc_val, col_mv) = if col_ref_idx_l0 >= 0 {
                let poc = ctx
                    .col_ref_poc_l0
                    .get(col_ref_idx_l0 as usize)
                    .copied()
                    .unwrap_or(ctx.col_poc);
                let mv = ctx.col_mv.get(col_blk).copied().unwrap_or([0, 0]);
                (poc, mv)
            } else {
                let col_ref_idx_l1 = ctx.col_ref_l1.get(col_blk).copied().unwrap_or(-1);
                if col_ref_idx_l1 < 0 {
                    *result = ([0, 0], 0, [0, 0], 0);
                    continue;
                }
                let poc = ctx
                    .col_ref_poc_l1
                    .get(col_ref_idx_l1 as usize)
                    .copied()
                    .unwrap_or(ctx.col_poc);
                let mv = ctx.col_mv_l1.get(col_blk).copied().unwrap_or([0, 0]);
                (poc, mv)
            };

            let l0_ref = ctx
                .cur_l0_ref_poc
                .iter()
                .position(|&poc| poc == col_ref_poc_val)
                .unwrap_or(0);

            let l0_ref_poc = ctx.cur_l0_ref_poc.get(l0_ref).copied().unwrap_or(0);
            let scale = compute_scale(l0_ref_poc, col_ref_poc_val);

            let mv_l0 = [
                ((scale * col_mv[0] as i32 + 128) >> 8) as i16,
                ((scale * col_mv[1] as i32 + 128) >> 8) as i16,
            ];
            let mv_l1 = [mv_l0[0] - col_mv[0], mv_l0[1] - col_mv[1]];

            *result = (mv_l0, l0_ref as i8, mv_l1, 0);
        }
    }

    results
}

/// Apply weighted prediction for a P-slice partition (luma + chroma).
#[allow(clippy::too_many_arguments)]
fn apply_weight_p(
    ctx: &mut FrameDecodeContext,
    hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    px_x: u32,
    px_y: u32,
    pw: usize,
    ph: usize,
    ref_idx: usize,
) {
    apply_weight_list(ctx, hdr, mb_x, mb_y, px_x, px_y, pw, ph, ref_idx, 0);
}

/// Apply uni-directional weighted prediction for a specific list (0=L0, 1=L1).
///
/// Called for P-slice weighted pred (list=0) and B-slice uni-directional
/// weighted pred (B_L0 uses list=0, B_L1 uses list=1).
#[allow(clippy::too_many_arguments)]
fn apply_weight_list(
    ctx: &mut FrameDecodeContext,
    hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    px_x: u32,
    px_y: u32,
    pw: usize,
    ph: usize,
    ref_idx: usize,
    list: u8,
) {
    let luma_weights = if list == 0 {
        &hdr.luma_weight_l0
    } else {
        &hdr.luma_weight_l1
    };
    let chroma_weights = if list == 0 {
        &hdr.chroma_weight_l0
    } else {
        &hdr.chroma_weight_l1
    };

    // Field-aware destination offsets
    let (luma_mb_off, luma_stride) = luma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
    let luma_off = luma_mb_off + px_y as usize * luma_stride + px_x as usize;

    // Luma — fall back to default weight (1<<denom, 0) when ref_idx exceeds
    // the parsed weight table.
    if hdr.use_weight {
        let (w, o) = if ref_idx < luma_weights.len() {
            luma_weights[ref_idx]
        } else {
            (1i32 << hdr.luma_log2_weight_denom, 0i32)
        };
        trace!(
            ref_idx,
            list,
            luma_weight = w,
            luma_offset = o,
            denom = hdr.luma_log2_weight_denom,
            "weighted pred luma"
        );
        apply_weight_uni_at(
            &mut ctx.pic.y,
            luma_off,
            luma_stride,
            pw,
            ph,
            hdr.luma_log2_weight_denom,
            w,
            o,
        );
    }

    // Chroma — same default fallback for chroma weights.
    if hdr.use_weight_chroma {
        let (chroma_mb_off, chroma_stride) = chroma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
        let chroma_off = chroma_mb_off + (px_y / 2) as usize * chroma_stride + (px_x / 2) as usize;
        let cw = pw / 2;
        let ch = ph / 2;
        let cw_entry = if ref_idx < chroma_weights.len() {
            chroma_weights[ref_idx]
        } else {
            [
                (1i32 << hdr.chroma_log2_weight_denom, 0i32),
                (1i32 << hdr.chroma_log2_weight_denom, 0i32),
            ]
        };
        // Cb
        apply_weight_uni_at(
            &mut ctx.pic.u,
            chroma_off,
            chroma_stride,
            cw,
            ch,
            hdr.chroma_log2_weight_denom,
            cw_entry[0].0,
            cw_entry[0].1,
        );
        // Cr
        apply_weight_uni_at(
            &mut ctx.pic.v,
            chroma_off,
            chroma_stride,
            cw,
            ch,
            hdr.chroma_log2_weight_denom,
            cw_entry[1].0,
            cw_entry[1].1,
        );
    }
}

/// Apply uni-directional weighted prediction to pixels already written by MC.
///
/// Uses offset-based addressing for MBAFF field-mode compatibility.
/// `base_offset` is the top-left of the block in the buffer, `stride` is field-adjusted.
///
/// For each pixel: `clip((pixel * weight + offset_scaled) >> log2_denom)`
///
/// Reference: FFmpeg h264dsp_template.c:30-61
#[allow(clippy::too_many_arguments)]
fn apply_weight_uni_at(
    buf: &mut [u8],
    base_offset: usize,
    stride: usize,
    w: usize,
    h: usize,
    log2_denom: u32,
    weight: i32,
    offset: i32,
) {
    let offset_scaled = if log2_denom > 0 {
        (offset << log2_denom) + (1 << (log2_denom - 1))
    } else {
        offset
    };
    for row in 0..h {
        let base = base_offset + row * stride;
        for col in 0..w {
            let idx = base + col;
            if idx < buf.len() {
                let val = (buf[idx] as i32 * weight + offset_scaled) >> log2_denom;
                buf[idx] = val.clamp(0, 255) as u8;
            }
        }
    }
}

/// Apply bidirectional motion compensation: MC from both L0 and L1, then average.
///
/// Bi-directional MC with optional weighted prediction.
///
/// Unweighted (weighted_bipred_idc == 0):
///   result[i] = (L0[i] + L1[i] + 1) >> 1
///
/// Explicit weighted (weighted_bipred_idc == 1):
///   result[i] = clip((L0[i]*w0 + L1[i]*w1 + offset) >> (denom+1))
///
/// Reference: FFmpeg h264dsp_template.c:63-92, h264_mb.c:406-456
#[allow(clippy::too_many_arguments)]
fn apply_mc_bi_partition(
    ctx: &mut FrameDecodeContext,
    ref_pics: &[Arc<SharedPicture>],
    ref_pics_l1: &[Arc<SharedPicture>],
    mb_x: u32,
    mb_y: u32,
    px_offset_x: u32,
    px_offset_y: u32,
    block_w: usize,
    block_h: usize,
    mv_l0: [i16; 2],
    ref_idx_l0: i8,
    mv_l1: [i16; 2],
    ref_idx_l1: i8,
    slice_hdr: &SliceHeader,
) {
    let ref_l0 = &ref_pics[(ref_idx_l0.max(0) as usize).min(ref_pics.len() - 1)];
    let ref_l1 = &ref_pics_l1[(ref_idx_l1.max(0) as usize).min(ref_pics_l1.len() - 1)];

    // Await L1 reference progress (L0 is awaited inside apply_mc_partition)
    let dst_y_await = (mb_y * 16 + px_offset_y) as i32;
    let mvy_l1 = mv_l1[1] as i32;
    let ref_y_bottom_l1 = dst_y_await + (mvy_l1 >> 2) + block_h as i32 + 3;
    let needed_row_l1 = (ref_y_bottom_l1.max(0) as u32 / 16).min(ref_l1.mb_height() - 1);
    ref_l1.wait_for_row(needed_row_l1 as i32);

    // SAFETY: wait_for_row guarantees the needed rows are published.
    let rp_l1 = unsafe { ref_l1.data() };

    // MC L0 into destination
    apply_mc_partition(
        ctx,
        ref_l0,
        mb_x,
        mb_y,
        px_offset_x,
        px_offset_y,
        block_w,
        block_h,
        mv_l0,
    );

    // MC L1 into temp buffers, then average with destination.
    // Use frame-mode coordinates for reference lookup, field-aware offsets for destination.
    let dst_x = (mb_x * 16 + px_offset_x) as i32;
    let dst_y = (mb_y * 16 + px_offset_y) as i32;
    let mb_field = ctx.mb_field;

    // Luma L1
    let mvx1 = mv_l1[0] as i32;
    let mvy1 = mv_l1[1] as i32;
    let l1_ref_x = dst_x + (mvx1 >> 2);
    let l1_ref_y = dst_y + (mvy1 >> 2);
    let l1_dx = (mvx1 & 3) as u8;
    let l1_dy = (mvy1 & 3) as u8;

    let mut tmp_y = vec![0u8; block_w * block_h];
    mc::mc_luma(
        &mut tmp_y,
        block_w,
        &rp_l1.y,
        rp_l1.y_stride,
        l1_ref_x,
        l1_ref_y,
        l1_dx,
        l1_dy,
        block_w,
        block_h,
        rp_l1.width,
        rp_l1.height,
    );

    // Average or weighted-average luma (field-aware destination stride)
    let (luma_mb_off, luma_dst_stride) = luma_mb_offset(&ctx.pic, mb_x, mb_y, mb_field);
    let luma_offset = luma_mb_off + px_offset_y as usize * luma_dst_stride + px_offset_x as usize;
    let l0_idx = ref_idx_l0.max(0) as usize;
    let l1_idx = ref_idx_l1.max(0) as usize;

    if slice_hdr.weighted_bipred_idc == 2
        && l0_idx < ctx.implicit_weight.len()
        && l1_idx < ctx.implicit_weight.get(l0_idx).map_or(0, |v| v.len())
    {
        let w0 = ctx.implicit_weight[l0_idx][l1_idx];
        let w1 = 64 - w0;
        biweight_pixels(
            &mut ctx.pic.y[luma_offset..],
            luma_dst_stride,
            &tmp_y,
            block_w,
            block_w,
            block_h,
            5,
            w0,
            w1,
            0,
        );
    } else if slice_hdr.use_weight
        && l0_idx < slice_hdr.luma_weight_l0.len()
        && l1_idx < slice_hdr.luma_weight_l1.len()
    {
        let (w0, o0) = slice_hdr.luma_weight_l0[l0_idx];
        let (w1, o1) = slice_hdr.luma_weight_l1[l1_idx];
        let denom = slice_hdr.luma_log2_weight_denom;
        biweight_pixels(
            &mut ctx.pic.y[luma_offset..],
            luma_dst_stride,
            &tmp_y,
            block_w,
            block_w,
            block_h,
            denom,
            w0,
            w1,
            o0 + o1,
        );
    } else {
        mc::avg_pixels_inplace(
            &mut ctx.pic.y[luma_offset..],
            luma_dst_stride,
            &tmp_y,
            block_w,
            block_w,
            block_h,
        );
    }

    // Chroma L1
    let chroma_w = block_w / 2;
    let chroma_h = block_h / 2;
    if chroma_w == 0 || chroma_h == 0 {
        return;
    }

    // Frame-mode coordinates for reference lookup
    let chroma_dst_x = (mb_x * 8 + px_offset_x / 2) as i32;
    let chroma_dst_y = (mb_y * 8 + px_offset_y / 2) as i32;
    let chroma_ref_x = chroma_dst_x + (mvx1 >> 3);
    let chroma_ref_y = chroma_dst_y + (mvy1 >> 3);
    let chroma_dx = (mvx1 & 7) as u8;
    let chroma_dy = (mvy1 & 7) as u8;

    let mut tmp_u = vec![0u8; chroma_w * chroma_h];
    mc::mc_chroma(
        &mut tmp_u,
        chroma_w,
        &rp_l1.u,
        rp_l1.uv_stride,
        chroma_ref_x,
        chroma_ref_y,
        chroma_dx,
        chroma_dy,
        chroma_w,
        chroma_h,
        rp_l1.width / 2,
        rp_l1.height / 2,
    );
    // Field-aware destination offset
    let (chroma_mb_off, chroma_dst_stride) = chroma_mb_offset(&ctx.pic, mb_x, mb_y, mb_field);
    let chroma_off =
        chroma_mb_off + (px_offset_y / 2) as usize * chroma_dst_stride + (px_offset_x / 2) as usize;

    let mut tmp_v = vec![0u8; chroma_w * chroma_h];
    mc::mc_chroma(
        &mut tmp_v,
        chroma_w,
        &rp_l1.v,
        rp_l1.uv_stride,
        chroma_ref_x,
        chroma_ref_y,
        chroma_dx,
        chroma_dy,
        chroma_w,
        chroma_h,
        rp_l1.width / 2,
        rp_l1.height / 2,
    );

    if slice_hdr.weighted_bipred_idc == 2
        && l0_idx < ctx.implicit_weight.len()
        && l1_idx < ctx.implicit_weight.get(l0_idx).map_or(0, |v| v.len())
    {
        let w0 = ctx.implicit_weight[l0_idx][l1_idx];
        let w1 = 64 - w0;
        biweight_pixels(
            &mut ctx.pic.u[chroma_off..],
            chroma_dst_stride,
            &tmp_u,
            chroma_w,
            chroma_w,
            chroma_h,
            5,
            w0,
            w1,
            0,
        );
        biweight_pixels(
            &mut ctx.pic.v[chroma_off..],
            chroma_dst_stride,
            &tmp_v,
            chroma_w,
            chroma_w,
            chroma_h,
            5,
            w0,
            w1,
            0,
        );
    } else if slice_hdr.use_weight_chroma
        && l0_idx < slice_hdr.chroma_weight_l0.len()
        && l1_idx < slice_hdr.chroma_weight_l1.len()
    {
        let cw0 = slice_hdr.chroma_weight_l0[l0_idx];
        let cw1 = slice_hdr.chroma_weight_l1[l1_idx];
        let denom = slice_hdr.chroma_log2_weight_denom;
        biweight_pixels(
            &mut ctx.pic.u[chroma_off..],
            chroma_dst_stride,
            &tmp_u,
            chroma_w,
            chroma_w,
            chroma_h,
            denom,
            cw0[0].0,
            cw1[0].0,
            cw0[0].1 + cw1[0].1,
        );
        biweight_pixels(
            &mut ctx.pic.v[chroma_off..],
            chroma_dst_stride,
            &tmp_v,
            chroma_w,
            chroma_w,
            chroma_h,
            denom,
            cw0[1].0,
            cw1[1].0,
            cw0[1].1 + cw1[1].1,
        );
    } else {
        mc::avg_pixels_inplace(
            &mut ctx.pic.u[chroma_off..],
            chroma_dst_stride,
            &tmp_u,
            chroma_w,
            chroma_w,
            chroma_h,
        );
        mc::avg_pixels_inplace(
            &mut ctx.pic.v[chroma_off..],
            chroma_dst_stride,
            &tmp_v,
            chroma_w,
            chroma_w,
            chroma_h,
        );
    }
}

/// Apply weighted bi-prediction to pixels.
///
/// FFmpeg formula: clip((dst * w0 + src * w1 + offset) >> (denom + 1))
/// where offset = ((o0 + o1 + 1) | 1) << denom
///
/// Reference: FFmpeg h264dsp_template.c:63-92
#[allow(clippy::too_many_arguments)]
fn biweight_pixels(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    log2_denom: u32,
    weight0: i32,
    weight1: i32,
    offset_sum: i32,
) {
    let offset = ((offset_sum + 1) | 1) << log2_denom;
    let shift = log2_denom + 1;
    for row in 0..h {
        let d_off = row * dst_stride;
        let s_off = row * src_stride;
        for col in 0..w {
            let val =
                (dst[d_off + col] as i32 * weight0 + src[s_off + col] as i32 * weight1 + offset)
                    >> shift;
            dst[d_off + col] = val.clamp(0, 255) as u8;
        }
    }
}

/// Fill a macroblock with gray (128) for all planes.
///
/// Used when no reference frame is available for inter prediction.
fn fill_mb_gray(ctx: &mut FrameDecodeContext, mb_x: u32, mb_y: u32) {
    // Luma
    let (luma_base, luma_stride) = luma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
    for y in 0..16 {
        let offset = luma_base + y * luma_stride;
        ctx.pic.y[offset..offset + 16].fill(128);
    }
    // Chroma
    let (chroma_base, chroma_stride) = chroma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
    for y in 0..8 {
        let uv_offset = chroma_base + y * chroma_stride;
        ctx.pic.u[uv_offset..uv_offset + 8].fill(128);
        ctx.pic.v[uv_offset..uv_offset + 8].fill(128);
    }
}

/// Decode chroma for an inter macroblock.
///
/// For inter MBs, the chroma prediction comes from motion compensation (already
/// applied). We only need to add the residual (chroma DC + AC coefficients).
#[allow(clippy::too_many_arguments)]
fn decode_chroma_inter(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    mb_x: u32,
    mb_y: u32,
    chroma_qp: [u8; 2],
) {
    let cbp_chroma = (mb.cbp >> 4) & 3;

    if cbp_chroma > 0 {
        for (plane_idx, &c_qp) in chroma_qp.iter().enumerate() {
            // Chroma DC: inverse Hadamard + dequant (i32 output to avoid i16 overflow)
            // FFmpeg uses list 4+plane for inter (4=Cb, 5=Cr)
            let dc_cqm = 4 + plane_idx;
            let qmul = dequant::dc_dequant_scale(&ctx.dequant4, dc_cqm, c_qp);
            let mut chroma_dc_out = [0i32; 4];
            idct::chroma_dc_dequant_idct(&mut chroma_dc_out, &mb.chroma_dc[plane_idx], qmul);

            {
                let plane_name = if plane_idx == 0 { "U" } else { "V" };
                trace!(mb_x, mb_y, plane = plane_name,
                    dc_in = ?mb.chroma_dc[plane_idx], dc_out = ?chroma_dc_out,
                    qmul, c_qp, "inter chroma DC");
            }

            for (blk_idx, &dc_val) in chroma_dc_out.iter().enumerate() {
                let blk_x = (blk_idx & 1) as u32;
                let blk_y = (blk_idx >> 1) as u32;

                let (c_offset, c_stride) =
                    chroma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y, ctx.mb_field);
                let plane_data = if plane_idx == 0 {
                    &mut ctx.pic.u
                } else {
                    &mut ctx.pic.v
                };

                if cbp_chroma >= 2 {
                    // AC present: combine DC into coeffs[0] for single IDCT pass
                    let ac_cqm = 4 + plane_idx; // inter: 4=Cb, 5=Cr
                    mb.chroma_ac[plane_idx][blk_idx][0] = dc_val as i16;
                    dequant::dequant_4x4(
                        &mut mb.chroma_ac[plane_idx][blk_idx],
                        &ctx.dequant4.coeffs[ac_cqm][c_qp as usize],
                    );
                    mb.chroma_ac[plane_idx][blk_idx][0] = dc_val as i16;
                    idct::idct4x4_add(
                        &mut plane_data[c_offset..],
                        c_stride,
                        &mut mb.chroma_ac[plane_idx][blk_idx],
                    );
                } else {
                    // DC-only
                    let dc_add = (dc_val + 32) >> 6;
                    {
                        let plane_name = if plane_idx == 0 { "U" } else { "V" };
                        let mc_row0: Vec<u8> = (0..4).map(|i| plane_data[c_offset + i]).collect();
                        trace!(mb_x, mb_y, plane = plane_name, blk_x, blk_y,
                            dc_val, dc_add, mc_pred_row0 = ?mc_row0, "inter chroma DC-add");
                    }
                    for j in 0..4 {
                        for i in 0..4 {
                            let idx = c_offset + j * c_stride + i;
                            plane_data[idx] = (plane_data[idx] as i32 + dc_add).clamp(0, 255) as u8;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// I_8x8 decode (High profile)
// ---------------------------------------------------------------------------

/// Maps 8x8 block index (0..3) to top-left pixel offset within the macroblock.
const BLOCK_8X8_OFFSET: [(u32, u32); 4] = [(0, 0), (8, 0), (0, 8), (8, 8)];

/// Decode an intra 8x8 macroblock (High profile, transform_size_8x8_flag=1).
#[allow(clippy::too_many_arguments)]
fn decode_intra8x8(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: [u8; 2],
    has_top: bool,
    has_left: bool,
) {
    let cbp_luma = mb.cbp & 0x0F;

    for i8x8 in 0..4usize {
        let (bx, by) = BLOCK_8X8_OFFSET[i8x8];
        // Compute field-aware offset and stride for this 8x8 block.
        // bx/by are pixel offsets (0 or 8) → convert to 4x4-block units.
        let blk_x_4 = bx / 4; // 0 or 2
        let blk_y_4 = by / 4;
        let (offset, stride) =
            luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x_4, blk_y_4, ctx.mb_field);

        // Neighbor availability for this 8x8 block
        let block_has_top = by > 0 || has_top;
        let block_has_left = bx > 0 || has_left;
        let block_has_top_left = (by > 0 || has_top) && (bx > 0 || has_left);
        let block_has_top_right = if by > 0 {
            // Within the MB: top-right 8x8 block is available if bx==0
            bx == 0
        } else if has_top {
            // First row: depends on neighbor MB availability
            if bx == 0 {
                true // top[8..15] from this MB or right part of top MB
            } else {
                // Top-right of the right 8x8 block = top-right MB.
                // Must check slice_table to ensure the above-right MB has
                // been decoded. In MBAFF field mode, the above row is from
                // the previous pair row (mb_y-2), not mb_y-1.
                let tr_y_opt = if ctx.is_mbaff && ctx.mb_field {
                    if mb_y >= 2 { Some(mb_y - 2) } else { None }
                } else {
                    if mb_y > 0 { Some(mb_y - 1) } else { None }
                };
                if mb_x + 1 < ctx.mb_width
                    && let Some(tr_y) = tr_y_opt
                {
                    let tr_idx = (tr_y * ctx.mb_width + mb_x + 1) as usize;
                    let decoded = ctx.slice_table[tr_idx] == ctx.current_slice;
                    if decoded && ctx.constrained_intra_pred {
                        ctx.mb_info[tr_idx].is_intra
                    } else {
                        decoded
                    }
                } else {
                    false
                }
            }
        } else {
            false
        };

        // Gather reference samples using offset/stride (field-mode aware)
        // Top: 16 samples (8 top + 8 top-right, replicated if unavailable)
        let mut top = [128u8; 16];
        if block_has_top && offset >= stride {
            let row_above = offset - stride;
            top[..8].copy_from_slice(&ctx.pic.y[row_above..row_above + 8]);
            if block_has_top_right {
                top[8..16].copy_from_slice(&ctx.pic.y[row_above + 8..row_above + 16]);
            } else {
                let v = top[7];
                top[8..16].fill(v);
            }
        }

        // Left: 8 samples
        let mut left = [128u8; 8];
        if block_has_left {
            for (i, lv) in left.iter_mut().enumerate() {
                *lv = ctx.pic.y[offset + i * stride - 1];
            }
        }

        // Top-left: 1 sample
        let top_left = if block_has_top_left && offset >= stride {
            ctx.pic.y[offset - stride - 1]
        } else {
            128
        };

        // Get prediction mode — stored per 8x8 block (first 4x4 sub-block within each)
        let raster_for_mode = SCAN_TO_RASTER[i8x8 * 4];
        let mode = mb.intra4x4_pred_mode[raster_for_mode];

        // Apply 8x8 intra prediction (filtering is done inside predict_8x8l)
        intra_pred::predict_8x8l(
            &mut ctx.pic.y[offset..],
            stride,
            mode,
            &top,
            &left,
            top_left,
            block_has_top,
            block_has_left,
            block_has_top_left,
            block_has_top_right,
        );
        // Prediction checksum (before residual addition)
        {
            let mut pred_sum = 0u32;
            for dy in 0..8usize {
                for dx in 0..8usize {
                    pred_sum = pred_sum.wrapping_add(ctx.pic.y[offset + dy * stride + dx] as u32);
                }
            }
            trace!(mb_x, mb_y, i8x8, mode, pred_sum, "INTRA8x8_PRED");
        }

        // Dequant and IDCT residual if CBP indicates this 8x8 block is coded
        if cbp_luma & (1 << i8x8) != 0 {
            // Determine CQM index: 0 for intra, 3 for inter
            let cqm = 0; // intra
            let dequant_table = &ctx.dequant8.coeffs[cqm][qp as usize];
            dequant::dequant_8x8(&mut mb.luma_8x8_coeffs[i8x8], dequant_table);
            trace!(
                mb_x,
                mb_y,
                block_idx = i8x8,
                qp,
                cqm,
                coeff_sum = mb.luma_8x8_coeffs[i8x8]
                    .iter()
                    .map(|&c| c.unsigned_abs() as u32)
                    .sum::<u32>(),
                dc = mb.luma_8x8_coeffs[i8x8][0],
                "DEQUANT"
            );
            idct::idct8x8_add(
                &mut ctx.pic.y[offset..],
                stride,
                &mut mb.luma_8x8_coeffs[i8x8],
            );
            // Post-IDCT pixel checksum for this 8x8 block
            {
                let mut post_sum = 0u32;
                for dy in 0..8usize {
                    for dx in 0..8usize {
                        post_sum =
                            post_sum.wrapping_add(ctx.pic.y[offset + dy * stride + dx] as u32);
                    }
                }
                trace!(mb_x, mb_y, i8x8, post_sum, "POST_IDCT_8X8");
            }
        }
    }

    // Decode chroma (same as 4x4)
    decode_chroma(ctx, mb, mb_x, mb_y, chroma_qp, has_top, has_left);
}

// Scan-to-raster mapping constant (re-exported from cavlc.rs for local use)
const SCAN_TO_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

// ---------------------------------------------------------------------------
// I_4x4 decode
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)] // H.264 decode requires all these parameters
fn decode_intra4x4(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: [u8; 2],
    has_top: bool,
    has_left: bool,
) {
    let cbp_luma = mb.cbp & 0x0F;

    // Raw neighbor pixels from frame buffer for MBAFF debugging.
    // Dumps the 16-pixel top row, left column, and top-left at MB-level
    // before any per-block gather. Compare with FFmpeg via ffmpeg_recon_extract.py.
    {
        let (mb_off, mb_str) = luma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);
        let raw_top: [u8; 16] = if mb_off >= mb_str {
            let row = mb_off - mb_str;
            let mut t = [0u8; 16];
            t.copy_from_slice(&ctx.pic.y[row..row + 16]);
            t
        } else {
            [0u8; 16]
        };
        let raw_left: [u8; 16] = if has_left {
            let mut l = [128u8; 16];
            for (i, lv) in l.iter_mut().enumerate() {
                *lv = ctx.pic.y[mb_off + i * mb_str - 1];
            }
            l
        } else {
            [128u8; 16]
        };
        let raw_tl = if has_top && has_left && mb_off >= mb_str {
            ctx.pic.y[mb_off - mb_str - 1]
        } else {
            128
        };
        trace!(
            mb_x, mb_y,
            offset = mb_off, stride = mb_str,
            top_row = ?raw_top,
            left_col = ?raw_left,
            top_left = raw_tl,
            "INTRA_RAW_NEIGHBORS"
        );
    }

    // Decode each of the 16 4x4 luma blocks in block scan order
    for (block, &(blk_x, blk_y)) in BLOCK_INDEX_TO_XY.iter().enumerate() {
        let (offset, stride) = luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y, ctx.mb_field);

        // Determine neighbor availability for this 4x4 block
        let block_has_top = blk_y > 0 || has_top;
        let block_has_left = blk_x > 0 || has_left;

        // Top-right availability: available if the block at (blk_x+1, blk_y-1) exists
        let block_has_top_right = if blk_y > 0 {
            // Within the MB, top-right is available for blocks not at the right edge of an 8x8 group
            blk_x < 3
                && !(blk_x == 1 && blk_y == 1)
                && !(blk_x == 1 && blk_y == 3)
                && !(blk_x == 3 && blk_y == 1)
                && !(blk_x == 3 && blk_y == 3)
        } else if has_top {
            // First row of blocks: top-right available from neighbor MB
            if blk_x < 3 {
                true
            } else {
                // blk_x == 3: top-right is from the MB to the upper-right.
                // Must check slice_table (not just bounds) to handle MBAFF
                // decode order where MB(x+1, y-1) may not be decoded yet.
                //
                // MBAFF field mode: the "row above" is from the previous
                // pair row (2 MB-rows above), not the immediately adjacent
                // MB row. The previous pair row is always fully decoded,
                // so the check should use mb_y-2 instead of mb_y-1.
                let tr_y_opt = if ctx.is_mbaff && ctx.mb_field {
                    // Field mode: above row is from previous pair row
                    if mb_y >= 2 { Some(mb_y - 2) } else { None }
                } else {
                    if mb_y > 0 { Some(mb_y - 1) } else { None }
                };
                if mb_x + 1 < ctx.mb_width
                    && let Some(tr_y) = tr_y_opt
                {
                    let tr_idx = (tr_y * ctx.mb_width + mb_x + 1) as usize;
                    let decoded = ctx.slice_table[tr_idx] == ctx.current_slice;
                    if decoded && ctx.constrained_intra_pred {
                        ctx.mb_info[tr_idx].is_intra
                    } else {
                        decoded
                    }
                } else {
                    false
                }
            }
        } else {
            false
        };

        // Gather neighbor pixels using offset/stride (field-mode aware)
        let top = gather_top_luma(
            &ctx.pic.y,
            offset,
            stride,
            block_has_top,
            block_has_top_right,
        );
        let left = gather_left_luma(&ctx.pic.y, offset, stride, block_has_left);
        let top_left =
            gather_top_left_luma(&ctx.pic.y, offset, stride, block_has_top, block_has_left);

        // Apply intra 4x4 prediction (modes stored in raster order)
        let raster_for_mode = block_to_raster(block);
        let mode = mb.intra4x4_pred_mode[raster_for_mode];
        intra_pred::predict_4x4(
            &mut ctx.pic.y[offset..],
            stride,
            mode,
            &top,
            &left,
            top_left,
            block_has_top,
            block_has_left,
            block_has_top_right,
        );

        {
            let mut pred = [0u8; 16];
            for r in 0..4 {
                let off = offset + r * stride;
                pred[r * 4..r * 4 + 4].copy_from_slice(&ctx.pic.y[off..off + 4]);
            }
            trace!(mb_x, mb_y, blk_x, blk_y, pred = ?pred, "intra4x4 prediction");
        }

        // Dequant and IDCT residual if CBP indicates this 8x8 group is coded
        let group_8x8 = (blk_y / 2) * 2 + (blk_x / 2);
        if cbp_luma & (1 << group_8x8) != 0 {
            let raster_idx = block_to_raster(block);

            {
                trace!(mb_x, mb_y, blk_x, blk_y, coeffs = ?mb.luma_coeffs[raster_idx], "intra4x4 pre-dequant");
            }

            dequant::dequant_4x4(
                &mut mb.luma_coeffs[raster_idx],
                &ctx.dequant4.coeffs[0][qp as usize],
            ); // intra Y

            {
                trace!(mb_x, mb_y, blk_x, blk_y, coeffs = ?mb.luma_coeffs[raster_idx], "intra4x4 post-dequant");
            }

            let (offset, stride) =
                luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y, ctx.mb_field);
            idct::idct4x4_add(
                &mut ctx.pic.y[offset..],
                stride,
                &mut mb.luma_coeffs[raster_idx],
            );

            {
                let mut final_px = [0u8; 16];
                for r in 0..4 {
                    let off = offset + r * stride;
                    final_px[r * 4..r * 4 + 4].copy_from_slice(&ctx.pic.y[off..off + 4]);
                }
                trace!(mb_x, mb_y, blk_x, blk_y, pixels = ?final_px, "intra4x4 post-IDCT");
            }
        }
    }

    // Decode chroma
    decode_chroma(ctx, mb, mb_x, mb_y, chroma_qp, has_top, has_left);
}

// ---------------------------------------------------------------------------
// I_16x16 decode
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)] // H.264 decode requires all these parameters
fn decode_intra16x16(
    ctx: &mut FrameDecodeContext,
    mb: &mut Macroblock,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: [u8; 2],
    has_top: bool,
    has_left: bool,
) {
    // Apply 16x16 intra prediction
    let (offset, stride) = luma_mb_offset(&ctx.pic, mb_x, mb_y, ctx.mb_field);

    // Raw neighbor pixels from frame buffer for MBAFF debugging
    {
        let raw_top: [u8; 16] = if offset >= stride {
            let row = offset - stride;
            let mut t = [0u8; 16];
            t.copy_from_slice(&ctx.pic.y[row..row + 16]);
            t
        } else {
            [0u8; 16]
        };
        let raw_left: [u8; 16] = if has_left {
            let mut l = [128u8; 16];
            for (i, lv) in l.iter_mut().enumerate() {
                *lv = ctx.pic.y[offset + i * stride - 1];
            }
            l
        } else {
            [128u8; 16]
        };
        let raw_tl = if has_top && has_left && offset >= stride {
            ctx.pic.y[offset - stride - 1]
        } else {
            128
        };
        trace!(
            mb_x, mb_y,
            offset, stride,
            top_row = ?raw_top,
            left_col = ?raw_left,
            top_left = raw_tl,
            "INTRA_RAW_NEIGHBORS"
        );
    }

    let top: [u8; 16] = gather_top(&ctx.pic.y, offset, stride);
    let left: [u8; 16] = gather_left(&ctx.pic.y, offset, stride, has_left);
    let top_left = gather_top_left(&ctx.pic.y, offset, stride, has_top, has_left);
    intra_pred::predict_16x16(
        &mut ctx.pic.y[offset..],
        stride,
        mb.intra16x16_mode,
        &top,
        &left,
        top_left,
        has_top,
        has_left,
    );

    {
        let mut pred_row0 = [0u8; 16];
        pred_row0.copy_from_slice(&ctx.pic.y[offset..offset + 16]);
        trace!(mb_x, mb_y, mode = mb.intra16x16_mode, row0 = ?pred_row0, "intra16x16 prediction");
    }

    // Luma DC: inverse Hadamard + dequant (i32 output to avoid i16 overflow)
    let qmul = dequant::dc_dequant_scale(&ctx.dequant4, 0, qp);
    let mut luma_dc_out = [0i32; 16];
    idct::luma_dc_dequant_idct(&mut luma_dc_out, &mb.luma_dc, qmul);

    {
        trace!(mb_x, mb_y, dc_out = ?luma_dc_out, "intra16x16 luma DC Hadamard");
    }

    let cbp_luma = mb.cbp & 0x0F;

    // For each 4x4 luma block: combine DC from Hadamard with AC coefficients.
    for (block, &(blk_x, blk_y)) in BLOCK_INDEX_TO_XY.iter().enumerate() {
        let raster_idx = block_to_raster(block);

        let group_8x8 = (blk_y / 2) * 2 + (blk_x / 2);
        let (offset, stride) = luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y, ctx.mb_field);

        if cbp_luma & (1 << group_8x8) != 0 {
            // AC present: combine DC into coeffs[0] for single IDCT pass
            // (avoids double +32 rounding bias from separate DC add + IDCT).
            // Dequant all 16 positions using pre-computed table, then restore
            // DC from Hadamard (which was already fully dequantized).
            mb.luma_coeffs[raster_idx][0] = luma_dc_out[block] as i16;
            dequant::dequant_4x4(
                &mut mb.luma_coeffs[raster_idx],
                &ctx.dequant4.coeffs[0][qp as usize],
            ); // intra Y
            mb.luma_coeffs[raster_idx][0] = luma_dc_out[block] as i16;
            idct::idct4x4_add(
                &mut ctx.pic.y[offset..],
                stride,
                &mut mb.luma_coeffs[raster_idx],
            );
        } else {
            // DC-only: apply with rounding
            let dc_add = (luma_dc_out[block] + 32) >> 6;
            for j in 0..4 {
                for i in 0..4 {
                    let idx = offset + j * stride + i;
                    ctx.pic.y[idx] = (ctx.pic.y[idx] as i32 + dc_add).clamp(0, 255) as u8;
                }
            }
        }
    }

    // Decode chroma
    decode_chroma(ctx, mb, mb_x, mb_y, chroma_qp, has_top, has_left);
}
