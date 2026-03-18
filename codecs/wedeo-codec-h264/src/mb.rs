// H.264 macroblock decode orchestrator.
//
// Decodes a single macroblock by calling CAVLC, dequant, intra prediction,
// and IDCT in the correct order. Handles I_4x4, I_16x16, and I_PCM macroblocks.
//
// Reference: FFmpeg libavcodec/h264_mb.c, h264_slice.c

use tracing::trace;
use wedeo_codec::bitstream::BitReadBE;
use wedeo_core::error::Result;

use crate::cavlc::{MacroblockCavlc, NeighborContext, decode_mb_cavlc};
use crate::deblock::{MbDeblockInfo, PictureBuffer};
use crate::dequant::{self, Dequant4Table};
use crate::idct;
use crate::intra_pred;
use crate::mc;
use crate::mvpred::{self, MvContext};
use crate::pps::Pps;
use crate::slice::SliceHeader;
use crate::sps::Sps;
use crate::tables::CHROMA_QP_TABLE;

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
// Frame-level decode context
// ---------------------------------------------------------------------------

/// Frame-level decode context.
pub struct FrameDecodeContext {
    pub pic: PictureBuffer,
    pub mb_info: Vec<MbDeblockInfo>,
    pub neighbor_ctx: NeighborContext,
    /// Current QP (starts from PPS init_qp + slice_qp_delta, modified per-MB).
    pub qp: u8,
    pub mb_width: u32,
    pub mb_height: u32,
    /// Pre-computed dequantization tables.
    pub dequant4: Dequant4Table,
    /// Motion vector context for inter prediction across MBs.
    pub mv_ctx: MvContext,
}

impl FrameDecodeContext {
    /// Create a new frame decode context for the given SPS and PPS.
    pub fn new(sps: &Sps, _pps: &Pps) -> Self {
        let mb_width = sps.mb_width;
        let mb_height = sps.mb_height;
        let width = mb_width * 16;
        let height = mb_height * 16;

        let y_stride = width as usize;
        let uv_stride = (width / 2) as usize;

        let pic = PictureBuffer {
            y: vec![0u8; y_stride * height as usize],
            u: vec![0u8; uv_stride * (height / 2) as usize],
            v: vec![0u8; uv_stride * (height / 2) as usize],
            y_stride,
            uv_stride,
            width,
            height,
            mb_width,
            mb_height,
        };

        let total_mbs = (mb_width * mb_height) as usize;

        Self {
            pic,
            mb_info: vec![MbDeblockInfo::default(); total_mbs],
            neighbor_ctx: NeighborContext::new(mb_width),
            qp: 0,
            mb_width,
            mb_height,
            dequant4: Dequant4Table::new(&sps.scaling_matrix4),
            mv_ctx: MvContext::new(mb_width, mb_height),
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel access helpers
// ---------------------------------------------------------------------------

/// Get the offset and stride for a 4x4 luma block within the picture buffer.
#[inline]
fn luma_block_offset(
    pic: &PictureBuffer,
    mb_x: u32,
    mb_y: u32,
    blk_x: u32,
    blk_y: u32,
) -> (usize, usize) {
    let x = (mb_x * 16 + blk_x * 4) as usize;
    let y = (mb_y * 16 + blk_y * 4) as usize;
    let stride = pic.y_stride;
    (y * stride + x, stride)
}

/// Get the offset and stride for a 4x4 chroma block within the picture buffer.
#[inline]
fn chroma_block_offset(
    pic: &PictureBuffer,
    mb_x: u32,
    mb_y: u32,
    blk_x: u32,
    blk_y: u32,
) -> (usize, usize) {
    let x = (mb_x * 8 + blk_x * 4) as usize;
    let y = (mb_y * 8 + blk_y * 4) as usize;
    let stride = pic.uv_stride;
    (y * stride + x, stride)
}

/// Gather the top 4 neighbor pixels for a 4x4 luma block.
/// Returns up to 8 values (4 top + 4 top-right for diagonal modes).
fn gather_top_luma(
    pic: &PictureBuffer,
    px: usize,
    py: usize,
    has_top: bool,
    has_top_right: bool,
) -> [u8; 8] {
    let stride = pic.y_stride;
    let mut top = [128u8; 8];
    if has_top && py > 0 {
        let row_above = (py - 1) * stride + px;
        top[..4].copy_from_slice(&pic.y[row_above..row_above + 4]);
        if has_top_right {
            top[4..8].copy_from_slice(&pic.y[row_above + 4..row_above + 8]);
        } else {
            // Replicate last top pixel for modes that need top-right
            let v = top[3];
            top[4..8].fill(v);
        }
    }
    top
}

/// Gather the left 4 neighbor pixels for a 4x4 luma block.
fn gather_left_luma(pic: &PictureBuffer, px: usize, py: usize, has_left: bool) -> [u8; 4] {
    let stride = pic.y_stride;
    let mut left = [128u8; 4];
    if has_left && px > 0 {
        for (i, l) in left.iter_mut().enumerate() {
            *l = pic.y[(py + i) * stride + px - 1];
        }
    }
    left
}

/// Gather the top-left corner pixel.
fn gather_top_left_luma(
    pic: &PictureBuffer,
    px: usize,
    py: usize,
    has_top: bool,
    has_left: bool,
) -> u8 {
    if has_top && has_left && px > 0 && py > 0 {
        pic.y[(py - 1) * pic.y_stride + px - 1]
    } else {
        128
    }
}

/// Gather top 16 pixels for 16x16 luma prediction.
fn gather_top_16(pic: &PictureBuffer, mb_x: u32, mb_y: u32) -> [u8; 16] {
    let mut top = [128u8; 16];
    if mb_y > 0 {
        let px = (mb_x * 16) as usize;
        let py = (mb_y * 16) as usize;
        let row_above = (py - 1) * pic.y_stride + px;
        top.copy_from_slice(&pic.y[row_above..row_above + 16]);
    }
    top
}

/// Gather left 16 pixels for 16x16 luma prediction.
fn gather_left_16(pic: &PictureBuffer, mb_x: u32, mb_y: u32) -> [u8; 16] {
    let mut left = [128u8; 16];
    if mb_x > 0 {
        let px = (mb_x * 16) as usize;
        let py = (mb_y * 16) as usize;
        for (i, l) in left.iter_mut().enumerate() {
            *l = pic.y[(py + i) * pic.y_stride + px - 1];
        }
    }
    left
}

/// Gather top-left pixel for 16x16 luma prediction.
fn gather_top_left_16(pic: &PictureBuffer, mb_x: u32, mb_y: u32) -> u8 {
    if mb_x > 0 && mb_y > 0 {
        let px = (mb_x * 16) as usize;
        let py = (mb_y * 16) as usize;
        pic.y[(py - 1) * pic.y_stride + px - 1]
    } else {
        128
    }
}

/// Gather top 8 pixels for chroma 8x8 prediction.
fn gather_top_chroma(plane: &[u8], stride: usize, mb_x: u32, mb_y: u32) -> [u8; 8] {
    let mut top = [128u8; 8];
    if mb_y > 0 {
        let px = (mb_x * 8) as usize;
        let py = (mb_y * 8) as usize;
        let row_above = (py - 1) * stride + px;
        top.copy_from_slice(&plane[row_above..row_above + 8]);
    }
    top
}

/// Gather left 8 pixels for chroma 8x8 prediction.
fn gather_left_chroma(plane: &[u8], stride: usize, mb_x: u32, mb_y: u32) -> [u8; 8] {
    let mut left = [128u8; 8];
    if mb_x > 0 {
        let px = (mb_x * 8) as usize;
        let py = (mb_y * 8) as usize;
        for (i, l) in left.iter_mut().enumerate() {
            *l = plane[(py + i) * stride + px - 1];
        }
    }
    left
}

/// Gather top-left pixel for chroma prediction.
fn gather_top_left_chroma(plane: &[u8], stride: usize, mb_x: u32, mb_y: u32) -> u8 {
    if mb_x > 0 && mb_y > 0 {
        let px = (mb_x * 8) as usize;
        let py = (mb_y * 8) as usize;
        plane[(py - 1) * stride + px - 1]
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
    mb: &mut MacroblockCavlc,
    mb_x: u32,
    mb_y: u32,
    chroma_qp: u8,
) {
    let has_top = mb_y > 0;
    let has_left = mb_x > 0;
    let cbp_chroma = (mb.cbp >> 4) & 3;

    for plane_idx in 0..2usize {
        // Select the correct chroma plane
        let (plane_data, stride) = if plane_idx == 0 {
            (&mut ctx.pic.u as &mut Vec<u8>, ctx.pic.uv_stride)
        } else {
            (&mut ctx.pic.v as &mut Vec<u8>, ctx.pic.uv_stride)
        };

        // Gather neighbors for chroma prediction
        let top = gather_top_chroma(plane_data, stride, mb_x, mb_y);
        let left = gather_left_chroma(plane_data, stride, mb_x, mb_y);
        let top_left = gather_top_left_chroma(plane_data, stride, mb_x, mb_y);

        // Apply chroma prediction
        let px = (mb_x * 8) as usize;
        let py = (mb_y * 8) as usize;
        let offset = py * stride + px;
        intra_pred::predict_chroma_8x8(
            &mut plane_data[offset..],
            stride,
            mb.chroma_pred_mode,
            &top,
            &left,
            top_left,
            has_top,
            has_left,
        );

        if cbp_chroma > 0 {
            // Chroma DC: inverse Hadamard + dequant (i32 output to avoid i16 overflow)
            let qmul = dequant::dc_dequant_scale(&ctx.dequant4, 1, chroma_qp);
            let mut chroma_dc_out = [0i32; 4];
            idct::chroma_dc_dequant_idct(&mut chroma_dc_out, &mb.chroma_dc[plane_idx], qmul);

            // For each 4x4 chroma block
            for (blk_idx, &dc_val) in chroma_dc_out.iter().enumerate() {
                let blk_x = (blk_idx & 1) as u32;
                let blk_y = (blk_idx >> 1) as u32;

                let (c_offset, c_stride) = chroma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y);
                let plane_data = if plane_idx == 0 {
                    &mut ctx.pic.u
                } else {
                    &mut ctx.pic.v
                };

                if cbp_chroma >= 2 {
                    // AC coefficients present — combine DC into coeffs[0] and
                    // process everything through a single IDCT pass (matching
                    // FFmpeg, which applies one +32 rounding bias for DC+AC).
                    mb.chroma_ac[plane_idx][blk_idx][0] = dc_val as i16;
                    dequant::dequant_4x4_flat(&mut mb.chroma_ac[plane_idx][blk_idx], chroma_qp);
                    // dequant_4x4_flat scaled [0] by the AC dequant factor, but
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
#[cfg_attr(
    feature = "tracing-detail",
    tracing::instrument(skip_all, fields(mb_x, mb_y))
)]
pub fn decode_macroblock(
    ctx: &mut FrameDecodeContext,
    br: &mut BitReadBE,
    slice_hdr: &SliceHeader,
    _sps: &Sps,
    pps: &Pps,
    mb_x: u32,
    mb_y: u32,
    ref_pics: &[&PictureBuffer],
) -> Result<()> {
    let mb_idx = (mb_y * ctx.mb_width + mb_x) as usize;

    // 1. Parse macroblock syntax via CAVLC
    let mut mb = decode_mb_cavlc(
        br,
        slice_hdr.slice_type,
        pps,
        &ctx.neighbor_ctx,
        mb_x,
        mb_y,
        ctx.mb_width,
    )?;

    // 2. Update QP with mb_qp_delta
    if mb.mb_qp_delta != 0 || mb.is_pcm {
        if mb.is_pcm {
            ctx.qp = 0;
        } else {
            ctx.qp = ((ctx.qp as i32 + mb.mb_qp_delta).rem_euclid(52)) as u8;
        }
    }
    let qp = ctx.qp;

    // Compute chroma QP
    let chroma_qp_idx = (qp as i32 + pps.chroma_qp_index_offset[0]).clamp(0, 51) as usize;
    let c_qp = CHROMA_QP_TABLE[chroma_qp_idx];

    let has_top = mb_y > 0;
    let has_left = mb_x > 0;

    trace!(
        mb_x,
        mb_y,
        qp,
        mb_type = mb.mb_type,
        cbp = mb.cbp,
        is_intra4x4 = mb.is_intra4x4,
        is_intra16x16 = mb.is_intra16x16,
        "decoded MB"
    );

    // 3. Decode based on macroblock type
    if mb.is_pcm {
        // I_PCM: raw samples already in the CAVLC output
        // Copy luma samples (16x16)
        for y in 0..16u32 {
            for x in 0..16u32 {
                let px = (mb_x * 16 + x) as usize;
                let py = (mb_y * 16 + y) as usize;
                let blk = ((y / 4) * 4 + (x / 4)) as usize;
                let sub_y = (y % 4) as usize;
                let sub_x = (x % 4) as usize;
                ctx.pic.y[py * ctx.pic.y_stride + px] =
                    mb.luma_coeffs[blk][sub_y * 4 + sub_x] as u8;
            }
        }
        // Copy chroma samples (8x8 each)
        for plane_idx in 0..2usize {
            let plane = if plane_idx == 0 {
                &mut ctx.pic.u
            } else {
                &mut ctx.pic.v
            };
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let px = (mb_x * 8 + x) as usize;
                    let py = (mb_y * 8 + y) as usize;
                    let blk = ((y / 4) * 2 + (x / 4)) as usize;
                    let sub_y = (y % 4) as usize;
                    let sub_x = (x % 4) as usize;
                    plane[py * ctx.pic.uv_stride + px] =
                        mb.chroma_ac[plane_idx][blk][sub_y * 4 + sub_x] as u8;
                }
            }
        }
    } else if mb.is_intra4x4 {
        decode_intra4x4(ctx, &mut mb, mb_x, mb_y, qp, c_qp, has_top, has_left);
    } else if mb.is_intra16x16 {
        decode_intra16x16(ctx, &mut mb, mb_x, mb_y, qp, c_qp, has_top, has_left);
    } else if !mb.is_intra {
        // Inter macroblock (P or B)
        decode_inter_mb(ctx, &mut mb, slice_hdr, mb_x, mb_y, qp, c_qp, ref_pics);
    }

    // 6. Update neighbor context
    // Build intra4x4 mode array for neighbor tracking.
    // For I_4x4 MBs, pass the decoded modes. For others, pass -1 (unavailable).
    // Build intra4x4 mode array for neighbor tracking.
    // For I_4x4 MBs: pass the decoded modes.
    // For other intra MBs (I_16x16, I_PCM): pass 2 (DC_PRED), matching FFmpeg's
    //   fill_decode_caches which sets `2 - 3 * !(type & type_mask)` = 2 for intra.
    // For inter MBs: pass -1 (unavailable).
    let intra4x4_modes: [i8; 16] = if mb.is_intra4x4 {
        let mut modes = [-1i8; 16];
        for (i, mode) in modes.iter_mut().enumerate() {
            *mode = mb.intra4x4_pred_mode[i] as i8;
        }
        modes
    } else if mb.is_intra {
        [2i8; 16] // DC_PRED for I_16x16 / I_PCM
    } else {
        [-1i8; 16]
    };
    ctx.neighbor_ctx
        .update_after_mb(mb_x, &mb.non_zero_count, &intra4x4_modes);
    ctx.neighbor_ctx.left_available = true;

    // 7. Store MbDeblockInfo for the deblocking filter
    let mb_idx_base = mb_idx * 16;
    let mut deblock_ref = [-1i8; 16];
    let mut deblock_mv = [[0i16; 2]; 16];
    if !mb.is_intra && mb_idx_base + 16 <= ctx.mv_ctx.mv.len() {
        deblock_ref.copy_from_slice(&ctx.mv_ctx.ref_idx[mb_idx_base..mb_idx_base + 16]);
        deblock_mv.copy_from_slice(&ctx.mv_ctx.mv[mb_idx_base..mb_idx_base + 16]);
    }
    ctx.mb_info[mb_idx] = MbDeblockInfo {
        is_intra: mb.is_intra,
        qp,
        non_zero_count: mb.non_zero_count,
        ref_idx: deblock_ref,
        mv: deblock_mv,
    };

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
    mb: &mut MacroblockCavlc,
    _slice_hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: u8,
    ref_pics: &[&PictureBuffer],
) {
    if ref_pics.is_empty() {
        // No reference frames available — fill with gray and return.
        fill_mb_gray(ctx, mb_x, mb_y);
        // Still need to set MV context for neighbor prediction
        for blk in 0..16 {
            ctx.mv_ctx.set(mb_x, mb_y, blk, [0, 0], 0);
        }
        return;
    }

    match mb.mb_type {
        0 => {
            // P_L0_16x16: one 16x16 partition
            let ref_idx = mb.ref_idx_l0[0].max(0) as usize;
            let n = ctx.mv_ctx.get_neighbors(mb_x, mb_y, 0, 0, 4);
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

            let ref_pic = ref_pics[ref_idx.min(ref_pics.len() - 1)];
            apply_mc_partition(ctx, ref_pic, mb_x, mb_y, 0, 0, 16, 16, mv);

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
                let n = ctx.mv_ctx.get_neighbors(mb_x, mb_y, 0, blk_y, 4);
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

                let ref_pic = ref_pics[ref_idx.min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, 0, blk_y * 4, 16, 8, mv);

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
                let n = ctx.mv_ctx.get_neighbors(mb_x, mb_y, blk_x, 0, 2);
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

                let ref_pic = ref_pics[ref_idx.min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, blk_x * 4, 0, 8, 16, mv);

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

                match sub_type {
                    0 => {
                        // 8x8 sub-partition
                        let n = ctx.mv_ctx.get_neighbors(mb_x, mb_y, part_x, part_y, 2);
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

                        let ref_pic = ref_pics[ref_idx.min(ref_pics.len() - 1)];
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
                            let n = ctx.mv_ctx.get_neighbors(mb_x, mb_y, part_x, sub_y, 2);
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

                            let ref_pic = ref_pics[ref_idx.min(ref_pics.len() - 1)];
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
                            let n = ctx.mv_ctx.get_neighbors(mb_x, mb_y, sub_x, part_y, 1);
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

                            let ref_pic = ref_pics[ref_idx.min(ref_pics.len() - 1)];
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
                            let n = ctx.mv_ctx.get_neighbors(mb_x, mb_y, sub_x, sub_y, 1);
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

                            let ref_pic = ref_pics[ref_idx.min(ref_pics.len() - 1)];
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

    // Add residual on top of the motion-compensated prediction
    let cbp_luma = mb.cbp & 0x0F;
    if cbp_luma != 0 {
        for (block, &(blk_x, blk_y)) in BLOCK_INDEX_TO_XY.iter().enumerate() {
            let group_8x8 = (blk_y / 2) * 2 + (blk_x / 2);
            if cbp_luma & (1 << group_8x8) != 0 {
                let raster_idx = block_to_raster(block);
                dequant::dequant_4x4_flat(&mut mb.luma_coeffs[raster_idx], qp);
                let (offset, stride) = luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y);
                idct::idct4x4_add(
                    &mut ctx.pic.y[offset..],
                    stride,
                    &mut mb.luma_coeffs[raster_idx],
                );
            }
        }
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
    ref_pic: &PictureBuffer,
    mb_x: u32,
    mb_y: u32,
    px_offset_x: u32,
    px_offset_y: u32,
    block_w: usize,
    block_h: usize,
    mv: [i16; 2],
) {
    let dst_x = (mb_x * 16 + px_offset_x) as i32;
    let dst_y = (mb_y * 16 + px_offset_y) as i32;

    // Quarter-pixel MV components
    let mvx = mv[0] as i32;
    let mvy = mv[1] as i32;

    // Luma: quarter-pixel precision
    let luma_ref_x = dst_x + (mvx >> 2);
    let luma_ref_y = dst_y + (mvy >> 2);
    let luma_dx = (mvx & 3) as u8;
    let luma_dy = (mvy & 3) as u8;

    let luma_offset = dst_y as usize * ctx.pic.y_stride + dst_x as usize;
    mc::mc_luma(
        &mut ctx.pic.y[luma_offset..],
        ctx.pic.y_stride,
        &ref_pic.y,
        ref_pic.y_stride,
        luma_ref_x,
        luma_ref_y,
        luma_dx,
        luma_dy,
        block_w,
        block_h,
        ref_pic.width,
        ref_pic.height,
    );

    // Chroma: eighth-pixel precision (MV divided by 2 with rounding)
    let chroma_w = block_w / 2;
    let chroma_h = block_h / 2;
    if chroma_w == 0 || chroma_h == 0 {
        return; // Partitions smaller than 4 pixels wide don't have separate chroma
    }

    let chroma_dst_x = (mb_x * 8 + px_offset_x / 2) as i32;
    let chroma_dst_y = (mb_y * 8 + px_offset_y / 2) as i32;

    // Chroma MV is luma MV / 2 at eighth-pixel precision
    // Spec: chroma MV = (luma_mv / 2) with rounding toward zero for the fractional part
    let cmvx = mvx;
    let cmvy = mvy;
    let chroma_ref_x = chroma_dst_x + (cmvx >> 3);
    let chroma_ref_y = chroma_dst_y + (cmvy >> 3);
    let chroma_dx = (cmvx & 7) as u8;
    let chroma_dy = (cmvy & 7) as u8;

    let chroma_offset_u = chroma_dst_y as usize * ctx.pic.uv_stride + chroma_dst_x as usize;
    mc::mc_chroma(
        &mut ctx.pic.u[chroma_offset_u..],
        ctx.pic.uv_stride,
        &ref_pic.u,
        ref_pic.uv_stride,
        chroma_ref_x,
        chroma_ref_y,
        chroma_dx,
        chroma_dy,
        chroma_w,
        chroma_h,
        ref_pic.width / 2,
        ref_pic.height / 2,
    );

    let chroma_offset_v = chroma_dst_y as usize * ctx.pic.uv_stride + chroma_dst_x as usize;
    mc::mc_chroma(
        &mut ctx.pic.v[chroma_offset_v..],
        ctx.pic.uv_stride,
        &ref_pic.v,
        ref_pic.uv_stride,
        chroma_ref_x,
        chroma_ref_y,
        chroma_dx,
        chroma_dy,
        chroma_w,
        chroma_h,
        ref_pic.width / 2,
        ref_pic.height / 2,
    );
}

/// Fill a macroblock with gray (128) for all planes.
///
/// Used when no reference frame is available for inter prediction.
fn fill_mb_gray(ctx: &mut FrameDecodeContext, mb_x: u32, mb_y: u32) {
    // Luma
    let luma_x = (mb_x * 16) as usize;
    let luma_y = (mb_y * 16) as usize;
    for y in 0..16 {
        let offset = (luma_y + y) * ctx.pic.y_stride + luma_x;
        ctx.pic.y[offset..offset + 16].fill(128);
    }
    // Chroma
    let chroma_x = (mb_x * 8) as usize;
    let chroma_y = (mb_y * 8) as usize;
    for y in 0..8 {
        let u_offset = (chroma_y + y) * ctx.pic.uv_stride + chroma_x;
        ctx.pic.u[u_offset..u_offset + 8].fill(128);
        let v_offset = (chroma_y + y) * ctx.pic.uv_stride + chroma_x;
        ctx.pic.v[v_offset..v_offset + 8].fill(128);
    }
}

/// Decode chroma for an inter macroblock.
///
/// For inter MBs, the chroma prediction comes from motion compensation (already
/// applied). We only need to add the residual (chroma DC + AC coefficients).
#[allow(clippy::too_many_arguments)]
fn decode_chroma_inter(
    ctx: &mut FrameDecodeContext,
    mb: &mut MacroblockCavlc,
    mb_x: u32,
    mb_y: u32,
    chroma_qp: u8,
) {
    let cbp_chroma = (mb.cbp >> 4) & 3;

    if cbp_chroma > 0 {
        for plane_idx in 0..2usize {
            // Chroma DC: inverse Hadamard + dequant (i32 output to avoid i16 overflow)
            let qmul = dequant::dc_dequant_scale(&ctx.dequant4, 1, chroma_qp);
            let mut chroma_dc_out = [0i32; 4];
            idct::chroma_dc_dequant_idct(&mut chroma_dc_out, &mb.chroma_dc[plane_idx], qmul);

            for (blk_idx, &dc_val) in chroma_dc_out.iter().enumerate() {
                let blk_x = (blk_idx & 1) as u32;
                let blk_y = (blk_idx >> 1) as u32;

                let (c_offset, c_stride) = chroma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y);
                let plane_data = if plane_idx == 0 {
                    &mut ctx.pic.u
                } else {
                    &mut ctx.pic.v
                };

                if cbp_chroma >= 2 {
                    // AC present: combine DC into coeffs[0] for single IDCT pass
                    mb.chroma_ac[plane_idx][blk_idx][0] = dc_val as i16;
                    dequant::dequant_4x4_flat(&mut mb.chroma_ac[plane_idx][blk_idx], chroma_qp);
                    mb.chroma_ac[plane_idx][blk_idx][0] = dc_val as i16;
                    idct::idct4x4_add(
                        &mut plane_data[c_offset..],
                        c_stride,
                        &mut mb.chroma_ac[plane_idx][blk_idx],
                    );
                } else {
                    // DC-only
                    let dc_add = (dc_val + 32) >> 6;
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
// I_4x4 decode
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)] // H.264 decode requires all these parameters
fn decode_intra4x4(
    ctx: &mut FrameDecodeContext,
    mb: &mut MacroblockCavlc,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: u8,
    has_top: bool,
    has_left: bool,
) {
    let cbp_luma = mb.cbp & 0x0F;

    // Decode each of the 16 4x4 luma blocks in block scan order
    for (block, &(blk_x, blk_y)) in BLOCK_INDEX_TO_XY.iter().enumerate() {
        let px = (mb_x * 16 + blk_x * 4) as usize;
        let py = (mb_y * 16 + blk_y * 4) as usize;

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
                // blk_x == 3: top-right is from the MB to the right
                mb_x + 1 < ctx.mb_width
            }
        } else {
            false
        };

        // Gather neighbor pixels
        let top = gather_top_luma(&ctx.pic, px, py, block_has_top, block_has_top_right);
        let left = gather_left_luma(&ctx.pic, px, py, block_has_left);
        let top_left = gather_top_left_luma(&ctx.pic, px, py, block_has_top, block_has_left);

        // Apply intra 4x4 prediction (modes stored in raster order)
        let raster_for_mode = block_to_raster(block);
        let mode = mb.intra4x4_pred_mode[raster_for_mode];
        let (offset, stride) = luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y);
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

        // Dequant and IDCT residual if CBP indicates this 8x8 group is coded
        let group_8x8 = (blk_y / 2) * 2 + (blk_x / 2);
        if cbp_luma & (1 << group_8x8) != 0 {
            let raster_idx = block_to_raster(block);
            dequant::dequant_4x4_flat(&mut mb.luma_coeffs[raster_idx], qp);
            let (offset, stride) = luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y);
            idct::idct4x4_add(
                &mut ctx.pic.y[offset..],
                stride,
                &mut mb.luma_coeffs[raster_idx],
            );
        }
    }

    // Decode chroma
    decode_chroma(ctx, mb, mb_x, mb_y, chroma_qp);
}

// ---------------------------------------------------------------------------
// I_16x16 decode
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)] // H.264 decode requires all these parameters
fn decode_intra16x16(
    ctx: &mut FrameDecodeContext,
    mb: &mut MacroblockCavlc,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: u8,
    has_top: bool,
    has_left: bool,
) {
    // Apply 16x16 intra prediction
    let top = gather_top_16(&ctx.pic, mb_x, mb_y);
    let left = gather_left_16(&ctx.pic, mb_x, mb_y);
    let top_left = gather_top_left_16(&ctx.pic, mb_x, mb_y);

    let px = (mb_x * 16) as usize;
    let py = (mb_y * 16) as usize;
    let offset = py * ctx.pic.y_stride + px;
    intra_pred::predict_16x16(
        &mut ctx.pic.y[offset..],
        ctx.pic.y_stride,
        mb.intra16x16_mode,
        &top,
        &left,
        top_left,
        has_top,
        has_left,
    );

    // Luma DC: inverse Hadamard + dequant (i32 output to avoid i16 overflow)
    let qmul = dequant::dc_dequant_scale(&ctx.dequant4, 0, qp);
    let mut luma_dc_out = [0i32; 16];
    idct::luma_dc_dequant_idct(&mut luma_dc_out, &mb.luma_dc, qmul);

    let cbp_luma = mb.cbp & 0x0F;

    // For each 4x4 luma block: combine DC from Hadamard with AC coefficients.
    for (block, &(blk_x, blk_y)) in BLOCK_INDEX_TO_XY.iter().enumerate() {
        let raster_idx = block_to_raster(block);

        let group_8x8 = (blk_y / 2) * 2 + (blk_x / 2);
        let (offset, stride) = luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y);

        if cbp_luma & (1 << group_8x8) != 0 {
            // AC present: combine DC into coeffs[0] for single IDCT pass
            // (avoids double +32 rounding bias from separate DC add + IDCT).
            mb.luma_coeffs[raster_idx][0] = luma_dc_out[block] as i16;
            let qp_per = (qp / 6) as u32;
            let qp_rem = (qp % 6) as usize;
            const POS_CLASS: [usize; 16] = [0, 1, 0, 1, 1, 2, 1, 2, 0, 1, 0, 1, 1, 2, 1, 2];
            let scale = &crate::tables::DEQUANT4_COEFF_INIT[qp_rem];
            for i in 1..16 {
                let s = scale[POS_CLASS[i]] as i32;
                mb.luma_coeffs[raster_idx][i] =
                    ((mb.luma_coeffs[raster_idx][i] as i32 * s) << qp_per) as i16;
            }
            // Restore DC after flat dequant (dequant doesn't touch [0] since
            // we only dequant [1..15], but be explicit):
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
    decode_chroma(ctx, mb, mb_x, mb_y, chroma_qp);
}
