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
// Deblocking helper
// ---------------------------------------------------------------------------

/// Map a raw ref_idx to a canonical picture ID using the reference picture list.
/// Returns the picture buffer's pointer address as i64, or -1 if unused.
/// This ensures cross-list comparisons (L0 vs L1) correctly identify same-picture.
fn ref_pic_id_from_list(ref_idx: i8, ref_pics: &[&PictureBuffer]) -> i64 {
    if ref_idx < 0 {
        return -1;
    }
    match ref_pics.get(ref_idx as usize) {
        Some(pic) => *pic as *const PictureBuffer as i64,
        None => -1,
    }
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
    /// Per-MB slice number for cross-slice neighbor availability.
    /// H.264 spec requires neighbors from different slices to be unavailable.
    pub slice_table: Vec<u16>,
    /// Current slice number (incremented per slice within a frame).
    pub current_slice: u16,
    /// Colocated L0 motion vectors from L1[0] reference frame (for spatial direct).
    pub col_mv: Vec<[i16; 2]>,
    /// Colocated L0 reference indices from L1[0] reference frame.
    pub col_ref: Vec<i8>,
    /// Per-MB intra flag from L1[0] reference frame.
    pub col_mb_intra: Vec<bool>,
    /// PPS constrained_intra_pred flag: inter neighbor pixels unavailable for intra prediction.
    pub constrained_intra_pred: bool,
}

impl FrameDecodeContext {
    /// Create a new frame decode context for the given SPS and PPS.
    pub fn new(sps: &Sps, pps: &Pps) -> Self {
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
            slice_table: vec![u16::MAX; total_mbs],
            current_slice: 0,
            col_mv: Vec::new(),
            col_ref: Vec::new(),
            col_mb_intra: Vec::new(),
            constrained_intra_pred: pps.constrained_intra_pred,
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
    has_top: bool,
    has_left: bool,
) {
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

        #[cfg(feature = "tracing-detail")]
        {
            let mut pred_row0 = [0u8; 8];
            pred_row0.copy_from_slice(&plane_data[offset..offset + 8]);
            let plane_name = if plane_idx == 0 { "U" } else { "V" };
            trace!(mb_x, mb_y, plane = plane_name, row0 = ?pred_row0, "chroma prediction");
        }

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

                #[cfg(feature = "tracing-detail")]
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
    ref_pics_l1: &[&PictureBuffer],
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
        slice_hdr.num_ref_idx_l0_active,
        slice_hdr.num_ref_idx_l1_active,
    )?;

    // 2. Update QP with mb_qp_delta
    // I_PCM: QP=0 for deblocking table, but running QP stays unchanged
    // (FFmpeg h264_cavlc.c: sl->qscale is NOT modified for I_PCM).
    if mb.mb_qp_delta != 0 {
        ctx.qp = ((ctx.qp as i32 + mb.mb_qp_delta).rem_euclid(52)) as u8;
    }
    let qp = if mb.is_pcm { 0 } else { ctx.qp };

    // Compute chroma QP
    let chroma_qp_idx = (qp as i32 + pps.chroma_qp_index_offset[0]).clamp(0, 51) as usize;
    let c_qp = CHROMA_QP_TABLE[chroma_qp_idx];

    // Check neighbor availability with slice boundary awareness.
    // H.264 spec: neighbors from different slices are unavailable.
    let cur_slice = ctx.current_slice;
    let mut has_top = mb_y > 0 && ctx.slice_table[mb_idx - ctx.mb_width as usize] == cur_slice;
    let mut has_left = mb_x > 0 && ctx.slice_table[mb_idx - 1] == cur_slice;

    // Constrained intra prediction: inter neighbor pixels are unavailable
    // for intra prediction (H.264 spec, FFmpeg h264_mvpred.h:598).
    if ctx.constrained_intra_pred && (mb.is_intra4x4 || mb.is_intra16x16 || mb.is_pcm) {
        if has_top && !ctx.mb_info[mb_idx - ctx.mb_width as usize].is_intra {
            has_top = false;
        }
        if has_left && !ctx.mb_info[mb_idx - 1].is_intra {
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
        decode_inter_mb(
            ctx,
            &mut mb,
            slice_hdr,
            mb_x,
            mb_y,
            qp,
            c_qp,
            ref_pics,
            ref_pics_l1,
        );
    }

    // 6. Update neighbor context
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
    ctx.neighbor_ctx
        .update_after_mb(mb_x, &mb.non_zero_count, &intra4x4_modes);
    ctx.neighbor_ctx.left_available = true;

    // 7. Store MbDeblockInfo for the deblocking filter
    let mb_idx_base = mb_idx * 16;
    let mut deblock_mv = [[0i16; 2]; 16];
    let mut deblock_mv_l1 = [[0i16; 2]; 16];
    let mut deblock_pic_id = [-1i64; 16];
    let mut deblock_pic_id_l1 = [-1i64; 16];
    if !mb.is_intra && mb_idx_base + 16 <= ctx.mv_ctx.mv.len() {
        deblock_mv.copy_from_slice(&ctx.mv_ctx.mv[mb_idx_base..mb_idx_base + 16]);
        deblock_mv_l1.copy_from_slice(&ctx.mv_ctx.mv_l1[mb_idx_base..mb_idx_base + 16]);
        // Map raw ref_idx to canonical picture IDs (pointer identity)
        for blk in 0..16 {
            let r0 = ctx.mv_ctx.ref_idx[mb_idx_base + blk];
            deblock_pic_id[blk] = ref_pic_id_from_list(r0, ref_pics);
            let r1 = ctx.mv_ctx.ref_idx_l1[mb_idx_base + blk];
            deblock_pic_id_l1[blk] = ref_pic_id_from_list(r1, ref_pics_l1);
        }
    }
    let list_count = if slice_hdr.slice_type.is_b() { 2 } else { 1 };
    ctx.mb_info[mb_idx] = MbDeblockInfo {
        is_intra: mb.is_intra,
        qp,
        non_zero_count: mb.non_zero_count,
        ref_pic_id: deblock_pic_id,
        mv: deblock_mv,
        ref_pic_id_l1: deblock_pic_id_l1,
        mv_l1: deblock_mv_l1,
        list_count,
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
    slice_hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: u8,
    ref_pics: &[&PictureBuffer],
    ref_pics_l1: &[&PictureBuffer],
) {
    // Dispatch B-frame inter MBs to dedicated handler
    if slice_hdr.slice_type.is_b() {
        decode_b_inter_mb(ctx, mb, mb_x, mb_y, qp, chroma_qp, ref_pics, ref_pics_l1);
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

    #[cfg(feature = "tracing-detail")]
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
            #[cfg(feature = "tracing-detail")]
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
            #[cfg(feature = "tracing-detail")]
            trace!(mb_x, mb_y, mvp = ?mvp, mvd = ?mb.mvd_l0[0], mv = ?mv, "16x16 MV");

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
                let n = ctx.mv_ctx.get_neighbors_slice(
                    mb_x,
                    mb_y,
                    0,
                    blk_y,
                    4,
                    Some(&ctx.slice_table),
                    ctx.current_slice,
                );
                #[cfg(feature = "tracing-detail")]
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

                #[cfg(feature = "tracing-detail")]
                trace!(mb_x, mb_y, part, mvp = ?mvp, mvd = ?mb.mvd_l0[part as usize], mv = ?mv, ref_idx, "16x8 MV");

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
                #[cfg(feature = "tracing-detail")]
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
                        #[cfg(feature = "tracing-detail")]
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
                        #[cfg(feature = "tracing-detail")]
                        trace!(mb_x, mb_y, i8x8, mvp = ?mvp, mvd = ?mb.mvd_l0[mvd_idx], mv = ?mv, "P_8x8 MV");

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
                            #[cfg(feature = "tracing-detail")]
                            trace!(mb_x, mb_y, i8x8, sub, part_x, sub_y,
                                mv_a = ?n.mv_a, ref_a = n.ref_a, a_avail = n.a_avail,
                                mv_b = ?n.mv_b, ref_b = n.ref_b, b_avail = n.b_avail,
                                mv_c = ?n.mv_c, ref_c = n.ref_c, c_avail = n.c_avail,
                                mvp = ?mvp, mvd = ?mb.mvd_l0[mvd_idx], mv = ?mv, "P_8x4 MV");

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
                            #[cfg(feature = "tracing-detail")]
                            trace!(mb_x, mb_y, i8x8, sub, sub_x, part_y,
                                mv_a = ?n.mv_a, ref_a = n.ref_a, a_avail = n.a_avail,
                                mv_b = ?n.mv_b, ref_b = n.ref_b, b_avail = n.b_avail,
                                mv_c = ?n.mv_c, ref_c = n.ref_c, c_avail = n.c_avail,
                                mvp = ?mvp, mvd = ?mb.mvd_l0[mvd_idx], mv = ?mv, "P_4x8 MV");

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

    #[cfg(feature = "tracing-detail")]
    {
        let lx = (mb_x * 16) as usize;
        let ly = (mb_y * 16) as usize;
        let stride = ctx.pic.y_stride;
        let mut mc_row0 = [0u8; 16];
        mc_row0.copy_from_slice(&ctx.pic.y[ly * stride + lx..ly * stride + lx + 16]);
        trace!(mb_x, mb_y, row0 = ?mc_row0, "inter MC luma");
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

    #[cfg(feature = "tracing-detail")]
    {
        let lx = (mb_x * 16) as usize;
        let ly = (mb_y * 16) as usize;
        let stride = ctx.pic.y_stride;
        let mut final_row0 = [0u8; 16];
        final_row0.copy_from_slice(&ctx.pic.y[ly * stride + lx..ly * stride + lx + 16]);
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

    // Temporary debug trace for SVA_Base_B MB(10,2) investigation
    #[cfg(feature = "tracing-detail")]
    if mb_x == 10 && mb_y == 2 {
        let ref_val = ref_pic.y[32 * ref_pic.y_stride + 160];
        let ref_val2 = ref_pic.y[32 * ref_pic.y_stride + 162];
        trace!(mb_x, mb_y, px_offset_y, mv = ?mv,
            ref_y_ptr = ?ref_pic.y.as_ptr(),
            ref_stride = ref_pic.y_stride,
            ref_at_160_32 = ref_val,
            ref_at_162_32 = ref_val2,
            "MC ref check");
    }

    // Quarter-pixel MV components
    let mvx = mv[0] as i32;
    let mvy = mv[1] as i32;

    // Luma: quarter-pixel precision
    let luma_ref_x = dst_x + (mvx >> 2);
    let luma_ref_y = dst_y + (mvy >> 2);
    let luma_dx = (mvx & 3) as u8;
    let luma_dy = (mvy & 3) as u8;

    let luma_offset = dst_y as usize * ctx.pic.y_stride + dst_x as usize;

    #[cfg(feature = "tracing-detail")]
    if mb_x == 10 && mb_y == 2 {
        // Print first row of ref data that the MC will read
        let ry = luma_ref_y.clamp(0, ref_pic.height as i32 - 1) as usize;
        let stride = ref_pic.y_stride;
        let ref_ptr = ref_pic.y.as_ptr();
        let cur_ptr = ctx.pic.y.as_ptr();
        let same_buf = std::ptr::eq(ref_ptr, cur_ptr);
        let ref_row: Vec<u8> = (0..21)
            .map(|i| {
                let rx =
                    (luma_ref_x as i32 - 2 + i as i32).clamp(0, ref_pic.width as i32 - 1) as usize;
                ref_pic.y[ry * stride + rx]
            })
            .collect();
        trace!(mb_x, mb_y, luma_ref_x, luma_ref_y, luma_dx, luma_dy,
               ref_ptr = ?ref_ptr, cur_ptr = ?cur_ptr, same_buf,
               ref_pic_width = ref_pic.width, ref_pic_height = ref_pic.height,
               ref_row = ?ref_row, "MC ref row");
    }

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
    ref_pics: &[&PictureBuffer],
    _ref_pics_l1: &[&PictureBuffer],
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
        #[cfg(feature = "tracing-detail")]
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
        #[cfg(feature = "tracing-detail")]
        trace!(mb_x, mb_y, mv = ?mv, "P_SKIP MV");

        // Apply motion compensation from ref_pics[0]
        apply_mc_partition(ctx, ref_pics[0], mb_x, mb_y, 0, 0, 16, 16, mv);

        // Fill MV context for all 16 4x4 blocks
        for blk in 0..16 {
            ctx.mv_ctx.set(mb_x, mb_y, blk, mv, 0);
        }
    }

    // Record slice ownership and update neighbor context.
    let mb_idx = (mb_y * ctx.mb_width + mb_x) as usize;
    ctx.slice_table[mb_idx] = ctx.current_slice;
    let nz = [0u8; 24];
    // Skip MBs are inter: use -1 (unavailable) when constrained_intra_pred
    let modes = if ctx.constrained_intra_pred {
        [-1i8; 16]
    } else {
        [2i8; 16]
    };
    ctx.neighbor_ctx.update_after_mb(mb_x, &nz, &modes);
    ctx.neighbor_ctx.left_available = true;

    // Store deblocking info (P_SKIP: only L0, list_count=1)
    let mb_idx_base = mb_idx * 16;
    let mut deblock_mv = [[0i16; 2]; 16];
    let mut deblock_pic_id = [-1i64; 16];
    if mb_idx_base + 16 <= ctx.mv_ctx.mv.len() {
        deblock_mv.copy_from_slice(&ctx.mv_ctx.mv[mb_idx_base..mb_idx_base + 16]);
        for (blk, pic_id) in deblock_pic_id.iter_mut().enumerate() {
            let r0 = ctx.mv_ctx.ref_idx[mb_idx_base + blk];
            *pic_id = ref_pic_id_from_list(r0, ref_pics);
        }
    }
    ctx.mb_info[mb_idx] = MbDeblockInfo {
        is_intra: false,
        qp: ctx.qp,
        non_zero_count: [0; 24],
        ref_pic_id: deblock_pic_id,
        mv: deblock_mv,
        ..Default::default()
    };

    let _ = slice_hdr; // reserved for future use (e.g. weighted prediction)
}

/// Decode a B_Skip macroblock (spatial direct prediction, no residual).
///
/// B_Skip uses spatial direct prediction for both L0 and L1 MVs,
/// then averages the two MC predictions. No residual data is present.
pub fn decode_b_skip_mb(
    ctx: &mut FrameDecodeContext,
    _slice_hdr: &SliceHeader,
    mb_x: u32,
    mb_y: u32,
    ref_pics: &[&PictureBuffer],
    ref_pics_l1: &[&PictureBuffer],
) {
    if ref_pics.is_empty() || ref_pics_l1.is_empty() {
        fill_mb_gray(ctx, mb_x, mb_y);
        for blk in 0..16 {
            ctx.mv_ctx.set(mb_x, mb_y, blk, [0, 0], 0);
            ctx.mv_ctx.set_l1(mb_x, mb_y, blk, [0, 0], 0);
        }
    } else {
        // Compute spatial direct MVs per 8x8 block
        let direct = pred_spatial_direct(ctx, mb_x, mb_y);
        // 8x8 partition offsets in 4x4-block units and pixel units
        const PART_8X8: [(u32, u32); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];

        for (i8, &(bx, by)) in PART_8X8.iter().enumerate() {
            let (mv_l0, ref_l0, mv_l1, ref_l1) = direct[i8];
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
                    8,
                    8,
                    mv_l0,
                    ref_l0,
                    mv_l1,
                    ref_l1,
                );
            } else if use_l0 {
                let ref_pic = ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 8, 8, mv_l0);
            } else if use_l1 {
                let ref_pic = ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 8, 8, mv_l1);
            }

            // Store MV context for this 8x8 block
            for dy in by..by + 2 {
                for dx in bx..bx + 2 {
                    ctx.mv_ctx
                        .set(mb_x, mb_y, (dx + dy * 4) as usize, mv_l0, ref_l0);
                    ctx.mv_ctx
                        .set_l1(mb_x, mb_y, (dx + dy * 4) as usize, mv_l1, ref_l1);
                }
            }
        }
    }

    // Record slice ownership and update neighbor context
    let mb_idx = (mb_y * ctx.mb_width + mb_x) as usize;
    ctx.slice_table[mb_idx] = ctx.current_slice;
    let nz = [0u8; 24];
    // B_Skip MBs are inter: use -1 (unavailable) when constrained_intra_pred
    let modes = if ctx.constrained_intra_pred {
        [-1i8; 16]
    } else {
        [2i8; 16]
    };
    ctx.neighbor_ctx.update_after_mb(mb_x, &nz, &modes);
    ctx.neighbor_ctx.left_available = true;

    // Store deblocking info (B_SKIP: always list_count=2)
    let mb_idx_base = mb_idx * 16;
    let mut deblock_mv = [[0i16; 2]; 16];
    let mut deblock_mv_l1 = [[0i16; 2]; 16];
    let mut deblock_pic_id = [-1i64; 16];
    let mut deblock_pic_id_l1 = [-1i64; 16];
    if mb_idx_base + 16 <= ctx.mv_ctx.mv.len() {
        deblock_mv.copy_from_slice(&ctx.mv_ctx.mv[mb_idx_base..mb_idx_base + 16]);
        deblock_mv_l1.copy_from_slice(&ctx.mv_ctx.mv_l1[mb_idx_base..mb_idx_base + 16]);
        for blk in 0..16 {
            let r0 = ctx.mv_ctx.ref_idx[mb_idx_base + blk];
            deblock_pic_id[blk] = ref_pic_id_from_list(r0, ref_pics);
            let r1 = ctx.mv_ctx.ref_idx_l1[mb_idx_base + blk];
            deblock_pic_id_l1[blk] = ref_pic_id_from_list(r1, ref_pics_l1);
        }
    }
    ctx.mb_info[mb_idx] = MbDeblockInfo {
        is_intra: false,
        qp: ctx.qp,
        non_zero_count: [0; 24],
        ref_pic_id: deblock_pic_id,
        mv: deblock_mv,
        ref_pic_id_l1: deblock_pic_id_l1,
        mv_l1: deblock_mv_l1,
        list_count: 2,
    };
}

/// Decode a B-frame inter macroblock.
///
/// Handles all B-frame partition types: B_Direct_16x16, B_L0/L1/Bi 16x16,
/// B_L0/L1/Bi 16x8/8x16, and B_8x8.
#[allow(clippy::too_many_arguments)]
fn decode_b_inter_mb(
    ctx: &mut FrameDecodeContext,
    mb: &mut MacroblockCavlc,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: u8,
    ref_pics: &[&PictureBuffer],
    ref_pics_l1: &[&PictureBuffer],
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
        // B_Direct_16x16: spatial direct prediction per 8x8 block
        let direct = pred_spatial_direct(ctx, mb_x, mb_y);
        const PART_8X8: [(u32, u32); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];

        for (i8, &(bx, by)) in PART_8X8.iter().enumerate() {
            let (mv_l0, ref_l0, mv_l1, ref_l1) = direct[i8];
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
                    8,
                    8,
                    mv_l0,
                    ref_l0,
                    mv_l1,
                    ref_l1,
                );
            } else if use_l0 {
                let ref_pic = ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 8, 8, mv_l0);
            } else if use_l1 {
                let ref_pic = ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 8, 8, mv_l1);
            }

            for dy in by..by + 2 {
                for dx in bx..bx + 2 {
                    ctx.mv_ctx
                        .set(mb_x, mb_y, (dx + dy * 4) as usize, mv_l0, ref_l0);
                    ctx.mv_ctx
                        .set_l1(mb_x, mb_y, (dx + dy * 4) as usize, mv_l1, ref_l1);
                }
            }
        }
    } else if mb.mb_type == 22 {
        // B_8x8: per-8x8-partition decode with sub_mb_type
        decode_b_8x8_mb(ctx, mb, mb_x, mb_y, ref_pics, ref_pics_l1);
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
                );
            } else if uses_l0 && !ref_pics.is_empty() {
                let ref_pic = ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, pw, ph, mv_l0);
            } else if uses_l1 && !ref_pics_l1.is_empty() {
                let ref_pic = ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, pw, ph, mv_l1);
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
    mb: &mut MacroblockCavlc,
    mb_x: u32,
    mb_y: u32,
    ref_pics: &[&PictureBuffer],
    ref_pics_l1: &[&PictureBuffer],
) {
    use crate::tables::B_SUB_MB_TYPE_INFO;

    // 8x8 partition layout: [0]=(0,0), [1]=(2,0), [2]=(0,2), [3]=(2,2) in 4x4-block units
    const PART_XY: [(u32, u32); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];

    // Pre-compute spatial direct for all 4 8x8 blocks (needed if any sub_mb_type==0).
    // Neighbor computation is MB-level so results are valid for all partitions.
    let has_direct = mb.sub_mb_type.contains(&0);
    let direct = if has_direct {
        Some(pred_spatial_direct(ctx, mb_x, mb_y))
    } else {
        None
    };

    for (part, &(blk_x, blk_y)) in PART_XY.iter().enumerate() {
        let sub_type = mb.sub_mb_type[part];

        if sub_type == 0 {
            // B_Direct_8x8: use per-8x8 spatial direct result
            let (mv_l0, ref_l0, mv_l1, ref_l1) = direct.unwrap()[part];

            let use_l0 = ref_l0 >= 0 && !ref_pics.is_empty();
            let use_l1 = ref_l1 >= 0 && !ref_pics_l1.is_empty();

            let px_x = blk_x * 4;
            let px_y = blk_y * 4;

            if use_l0 && use_l1 {
                apply_mc_bi_partition(
                    ctx,
                    ref_pics,
                    ref_pics_l1,
                    mb_x,
                    mb_y,
                    px_x,
                    px_y,
                    8,
                    8,
                    mv_l0,
                    ref_l0,
                    mv_l1,
                    ref_l1,
                );
            } else if use_l0 {
                let ref_pic = ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 8, 8, mv_l0);
            } else if use_l1 {
                let ref_pic = ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, 8, 8, mv_l1);
            }

            // Store MV context for the 8x8 block
            for by in blk_y..blk_y + 2 {
                for bx in blk_x..blk_x + 2 {
                    ctx.mv_ctx
                        .set(mb_x, mb_y, (bx + by * 4) as usize, mv_l0, ref_l0);
                    ctx.mv_ctx
                        .set_l1(mb_x, mb_y, (bx + by * 4) as usize, mv_l1, ref_l1);
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
                    );
                } else if uses_l0 && !ref_pics.is_empty() {
                    let ref_pic = ref_pics[(ref_l0 as usize).min(ref_pics.len() - 1)];
                    apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, pw, ph, mv_l0);
                } else if uses_l1 && !ref_pics_l1.is_empty() {
                    let ref_pic = ref_pics_l1[(ref_l1 as usize).min(ref_pics_l1.len() - 1)];
                    apply_mc_partition(ctx, ref_pic, mb_x, mb_y, px_x, px_y, pw, ph, mv_l1);
                }
            }
        }
    }
}

/// Add residual coefficients on top of B-frame MC prediction.
#[allow(clippy::too_many_arguments)]
fn add_b_residual(
    ctx: &mut FrameDecodeContext,
    mb: &mut MacroblockCavlc,
    mb_x: u32,
    mb_y: u32,
    qp: u8,
    chroma_qp: u8,
) {
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
fn pred_spatial_direct(
    ctx: &FrameDecodeContext,
    mb_x: u32,
    mb_y: u32,
) -> [([i16; 2], i8, [i16; 2], i8); 4] {
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
        return [base; 4];
    }

    // Per-8x8-block col_zero_flag optimization.
    // Check the colocated block in the L1[0] reference frame per 8x8 partition.
    // If colocated ref_idx=0 and |mv| <= 1, suppress the spatial MV for lists
    // where ref == 0.
    // Reference: FFmpeg h264_direct.c lines 424-477 (per-8x8 path)
    let mb_idx = (mb_y * ctx.mb_width + mb_x) as usize;
    let col_is_intra = ctx.col_mb_intra.get(mb_idx).copied().unwrap_or(true);

    let mut results = [base; 4];

    if !col_is_intra && !ctx.col_ref.is_empty() {
        let blk_base = mb_idx * 16;
        // Raster indices for ref at corner of each 8x8 block (one ref per 8x8):
        // FFmpeg: l1ref0[i8] where i8=0..3, stored per-8x8
        // Wedeo: per-4x4, use top-left corner of each 8x8
        const REF_CORNERS: [usize; 4] = [0, 2, 8, 10];
        // Raster indices for MV at corner of each 8x8 (direct_8x8_inference_flag=1):
        // FFmpeg: l1mv[x8*3 + y8*3*b4_stride] → bottom-right 4x4 of each 8x8
        // In raster order: (0,0)=0, (3,0)=3, (0,3)=12, (3,3)=15
        const MV_CORNERS: [usize; 4] = [0, 3, 12, 15];

        for i8 in 0..4 {
            let col_ref0 = ctx
                .col_ref
                .get(blk_base + REF_CORNERS[i8])
                .copied()
                .unwrap_or(-1);

            if col_ref0 == 0 {
                let col_mv0 = ctx
                    .col_mv
                    .get(blk_base + MV_CORNERS[i8])
                    .copied()
                    .unwrap_or([0, 0]);
                if col_mv0[0].abs() <= 1 && col_mv0[1].abs() <= 1 {
                    let mut a = mv[0];
                    let mut b = mv[1];
                    if ref_idx[0] == 0 {
                        a = [0, 0];
                    }
                    if ref_idx[1] == 0 {
                        b = [0, 0];
                    }
                    results[i8] = (a, ref_idx[0], b, ref_idx[1]);
                }
            }
        }
    }

    results
}

/// Apply bidirectional motion compensation: MC from both L0 and L1, then average.
///
/// This handles the unweighted case (weighted_bipred_idc == 0):
/// result[i] = (L0[i] + L1[i] + 1) >> 1
#[allow(clippy::too_many_arguments)]
fn apply_mc_bi_partition(
    ctx: &mut FrameDecodeContext,
    ref_pics: &[&PictureBuffer],
    ref_pics_l1: &[&PictureBuffer],
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
) {
    let ref_l0 = ref_pics[(ref_idx_l0.max(0) as usize).min(ref_pics.len() - 1)];
    let ref_l1 = ref_pics_l1[(ref_idx_l1.max(0) as usize).min(ref_pics_l1.len() - 1)];

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

    // MC L1 into temp buffers, then average with destination
    let dst_x = (mb_x * 16 + px_offset_x) as i32;
    let dst_y = (mb_y * 16 + px_offset_y) as i32;

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
        &ref_l1.y,
        ref_l1.y_stride,
        l1_ref_x,
        l1_ref_y,
        l1_dx,
        l1_dy,
        block_w,
        block_h,
        ref_l1.width,
        ref_l1.height,
    );

    // Average luma
    let luma_offset = dst_y as usize * ctx.pic.y_stride + dst_x as usize;
    mc::avg_pixels_inplace(
        &mut ctx.pic.y[luma_offset..],
        ctx.pic.y_stride,
        &tmp_y,
        block_w,
        block_w,
        block_h,
    );

    // Chroma L1
    let chroma_w = block_w / 2;
    let chroma_h = block_h / 2;
    if chroma_w == 0 || chroma_h == 0 {
        return;
    }

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
        &ref_l1.u,
        ref_l1.uv_stride,
        chroma_ref_x,
        chroma_ref_y,
        chroma_dx,
        chroma_dy,
        chroma_w,
        chroma_h,
        ref_l1.width / 2,
        ref_l1.height / 2,
    );
    let chroma_offset = chroma_dst_y as usize * ctx.pic.uv_stride + chroma_dst_x as usize;
    mc::avg_pixels_inplace(
        &mut ctx.pic.u[chroma_offset..],
        ctx.pic.uv_stride,
        &tmp_u,
        chroma_w,
        chroma_w,
        chroma_h,
    );

    let mut tmp_v = vec![0u8; chroma_w * chroma_h];
    mc::mc_chroma(
        &mut tmp_v,
        chroma_w,
        &ref_l1.v,
        ref_l1.uv_stride,
        chroma_ref_x,
        chroma_ref_y,
        chroma_dx,
        chroma_dy,
        chroma_w,
        chroma_h,
        ref_l1.width / 2,
        ref_l1.height / 2,
    );
    mc::avg_pixels_inplace(
        &mut ctx.pic.v[chroma_offset..],
        ctx.pic.uv_stride,
        &tmp_v,
        chroma_w,
        chroma_w,
        chroma_h,
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

            #[cfg(feature = "tracing-detail")]
            {
                let plane_name = if plane_idx == 0 { "U" } else { "V" };
                trace!(mb_x, mb_y, plane = plane_name,
                    dc_in = ?mb.chroma_dc[plane_idx], dc_out = ?chroma_dc_out,
                    qmul, chroma_qp, "inter chroma DC");
            }

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
                    #[cfg(feature = "tracing-detail")]
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
                // blk_x == 3: top-right is from the MB to the upper-right
                let tr_available = mb_x + 1 < ctx.mb_width;
                if tr_available && ctx.constrained_intra_pred {
                    // Upper-right MB must be intra for constrained intra pred
                    let tr_idx = ((mb_y - 1) * ctx.mb_width + mb_x + 1) as usize;
                    ctx.mb_info[tr_idx].is_intra
                } else {
                    tr_available
                }
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

        #[cfg(feature = "tracing-detail")]
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

            #[cfg(feature = "tracing-detail")]
            {
                trace!(mb_x, mb_y, blk_x, blk_y, coeffs = ?mb.luma_coeffs[raster_idx], "intra4x4 pre-dequant");
            }

            dequant::dequant_4x4_flat(&mut mb.luma_coeffs[raster_idx], qp);

            #[cfg(feature = "tracing-detail")]
            {
                trace!(mb_x, mb_y, blk_x, blk_y, coeffs = ?mb.luma_coeffs[raster_idx], "intra4x4 post-dequant");
            }

            let (offset, stride) = luma_block_offset(&ctx.pic, mb_x, mb_y, blk_x, blk_y);
            idct::idct4x4_add(
                &mut ctx.pic.y[offset..],
                stride,
                &mut mb.luma_coeffs[raster_idx],
            );

            #[cfg(feature = "tracing-detail")]
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

    #[cfg(feature = "tracing-detail")]
    {
        let mut pred_row0 = [0u8; 16];
        pred_row0.copy_from_slice(&ctx.pic.y[offset..offset + 16]);
        trace!(mb_x, mb_y, mode = mb.intra16x16_mode, row0 = ?pred_row0, "intra16x16 prediction");
    }

    // Luma DC: inverse Hadamard + dequant (i32 output to avoid i16 overflow)
    let qmul = dequant::dc_dequant_scale(&ctx.dequant4, 0, qp);
    let mut luma_dc_out = [0i32; 16];
    idct::luma_dc_dequant_idct(&mut luma_dc_out, &mb.luma_dc, qmul);

    #[cfg(feature = "tracing-detail")]
    {
        trace!(mb_x, mb_y, dc_out = ?luma_dc_out, "intra16x16 luma DC Hadamard");
    }

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
    decode_chroma(ctx, mb, mb_x, mb_y, chroma_qp, has_top, has_left);
}
