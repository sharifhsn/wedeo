// VP9 block-level parsing: partition tree, mode decoding, coefficient decoding.
//
// Translated from FFmpeg's libavcodec/vp9block.c (decode_mode, decode_coeffs,
// decode_coeffs_b_generic) and libavcodec/vp9.c (decode_sb).
//
// Only the keyframe (intra) path is implemented.  Inter-frame paths return
// Error::PatchwelcomeNotImplemented.

use wedeo_core::error::{Error, Result};

use crate::bool_decoder::BoolDecoder;
use crate::context::{AboveContext, LeftContext};
use crate::data::{
    BWH_TAB, COL_SCAN_4X4_NB, COL_SCAN_8X8_NB, COL_SCAN_16X16_NB, DEFAULT_SCAN_4X4_NB,
    DEFAULT_SCAN_8X8_NB, DEFAULT_SCAN_16X16_NB, DEFAULT_SCAN_32X32_NB, FILTER_LUT, FILTER_TREE,
    INTER_MODE_CTX_LUT, INTER_MODE_CTX_OFF, INTER_MODE_TREE, INTRA_TXFM_TYPE, INTRAMODE_TREE,
    KF_PARTITION_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, PARETO8, PARTITION_TREE,
    ROW_SCAN_4X4_NB, ROW_SCAN_8X8_NB, ROW_SCAN_16X16_NB, SIZE_GROUP,
};
use crate::header::{FrameHeader, FrameType, TxMode};
use crate::mvs::{MvSearchCtx, fill_mv};
use crate::prob::CountContext;
use crate::quant::{get_ac_quant, get_dc_quant};
use crate::refs::MvRefPair;
use crate::types::{BlockLevel, BlockPartition, BlockSize, IntraMode, TxSize};

// ---------------------------------------------------------------------------
// Per-block decoded data
// ---------------------------------------------------------------------------

/// All decoded data for one VP9 prediction block (one leaf of the partition
/// tree).
///
/// Stored so that a reconstruction agent (Agent 5) can iterate over blocks in
/// raster order and apply intra prediction + IDCT.
#[derive(Debug)]
pub struct BlockInfo {
    /// Row position in 4×4 units.
    pub row: usize,
    /// Column position in 4×4 units.
    pub col: usize,
    /// Block size.
    pub bs: BlockSize,
    /// Luma transform size.
    pub tx_size: TxSize,
    /// UV transform size.
    pub uv_tx_size: TxSize,
    /// Luma prediction modes (up to 4 sub-4×4 modes for small blocks).
    pub y_mode: [IntraMode; 4],
    /// UV prediction mode (single value for the block).
    pub uv_mode: IntraMode,
    /// Skip flag (no transform coefficients if true).
    pub skip: bool,
    /// Segment ID.
    pub segment_id: u8,
    /// Tile column start in 4×4 units (for intra edge detection at tile boundaries).
    pub tile_col_start: usize,
    // --- Inter-prediction fields ---
    /// True if this block uses inter prediction.
    pub is_inter: bool,
    /// True if compound (two-reference) prediction is used.
    pub comp: bool,
    /// Reference frame indices. ref_frame[0] is always valid for inter;
    /// ref_frame[1] is valid when comp=true. Values: 0=LAST, 1=GOLDEN,
    /// 2=ALTREF. -1 = unused (intra or single-ref second slot).
    pub ref_frame: [i8; 2],
    /// Per-sub-block inter prediction mode (NEARESTMV=10..NEWMV=13).
    /// For blocks >= 8×8, all 4 entries are the same.
    pub inter_mode: [u8; 4],
    /// Per-sub-block motion vectors. `mv[sub_block][ref_idx] = [x, y]`
    /// in 1/8-pel units.
    pub mv: [[[i16; 2]; 2]; 4],
    /// Interpolation filter index (0=Smooth, 1=Regular, 2=Sharp, 3=Bilinear).
    pub filter: u8,
    /// Dequantised luma coefficients, one flat array per transform block.
    /// The number of transform blocks is `(w4 / tx_step) * (h4 / tx_step)`
    /// where `tx_step` is the transform size in 4×4 units.
    pub coefs_y: Vec<i32>,
    /// Dequantised Cb coefficients.
    pub coefs_u: Vec<i32>,
    /// Dequantised Cr coefficients.
    pub coefs_v: Vec<i32>,
}

// ---------------------------------------------------------------------------
// Tile decode context
// ---------------------------------------------------------------------------

/// Per-tile decode state for one VP9 tile.
///
/// Mirrors the relevant fields of VP9TileData (vp9dec.h) that are needed
/// during the parse-only phase (no pixel reconstruction).
pub struct TileDecodeContext<'a> {
    /// Boolean arithmetic decoder (one per tile data stream).
    pub bd: BoolDecoder<'a>,
    /// Parsed frame header.
    pub header: &'a FrameHeader,
    /// Above-row context (shared across the whole frame).
    pub above: &'a mut AboveContext,
    /// Left-column context (local to the current superblock row).
    pub left: &'a mut LeftContext,
    /// Symbol counts accumulated during this tile (for probability adaptation).
    pub counts: CountContext,
    /// Decoded blocks in raster order (row-major, then column-major).
    pub blocks: Vec<BlockInfo>,
    /// Frame dimensions in superblock units.
    pub sb_cols: usize,
    pub sb_rows: usize,
    /// Frame dimensions in 4×4 units.
    pub rows_4x4: usize,
    pub cols_4x4: usize,
    /// First 4×4 column that belongs to this tile.
    pub tile_col_start: usize,
    // --- Inter-prediction state ---
    /// Current frame's in-progress per-4×4 MV grid.
    pub cur_mv_grid: &'a mut [MvRefPair],
    /// Previous frame's MV grid for temporal MV prediction.
    pub prev_mv_grid: Option<&'a [MvRefPair]>,
    /// Previous frame's cols_4x4.
    pub prev_cols_4x4: usize,
    /// Sign bias per reference frame (from header).
    pub sign_bias: [bool; 3],
}

impl<'a> TileDecodeContext<'a> {
    /// Create a new tile decode context.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bd: BoolDecoder<'a>,
        header: &'a FrameHeader,
        above: &'a mut AboveContext,
        left: &'a mut LeftContext,
        tile_col_start: usize,
        cur_mv_grid: &'a mut [MvRefPair],
        prev_mv_grid: Option<&'a [MvRefPair]>,
        prev_cols_4x4: usize,
    ) -> Self {
        let width = header.width as usize;
        let height = header.height as usize;
        let cols_4x4 = width.div_ceil(4);
        let rows_4x4 = height.div_ceil(4);
        let sb_cols = cols_4x4.div_ceil(16);
        let sb_rows = rows_4x4.div_ceil(16);
        Self {
            bd,
            header,
            above,
            left,
            counts: CountContext::default(),
            blocks: Vec::new(),
            sb_cols,
            sb_rows,
            rows_4x4,
            cols_4x4,
            tile_col_start,
            cur_mv_grid,
            prev_mv_grid,
            prev_cols_4x4,
            sign_bias: header.sign_bias,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers: partition context
// ---------------------------------------------------------------------------

/// Per-block-size values written to the partition context arrays, indexed by
/// BlockSize.  These match `left_ctx[N_BS_SIZES]` and `above_ctx[N_BS_SIZES]`
/// from decode_mode() in vp9block.c.
const LEFT_PARTITION_CTX: [u8; 13] = [
    0x0, 0x8, 0x0, 0x8, 0xc, 0x8, 0xc, 0xe, 0xc, 0xe, 0xf, 0xe, 0xf,
];
const ABOVE_PARTITION_CTX: [u8; 13] = [
    0x0, 0x0, 0x8, 0x8, 0x8, 0xc, 0xc, 0xc, 0xe, 0xe, 0xe, 0xf, 0xf,
];

/// Maximum transform size per block size (max_tx_for_bl_bp in vp9block.c).
const MAX_TX_FOR_BS: [u8; 13] = [3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0];

// ---------------------------------------------------------------------------
// decode_sb — partition tree recursion  (vp9.c: decode_sb)
// ---------------------------------------------------------------------------

/// Recursively partition a superblock starting at (`row`, `col`) in 4×4 units
/// at block level `bl` and decode each leaf block.
///
/// Corresponds to `decode_sb` in FFmpeg's vp9.c.
pub fn decode_sb(
    td: &mut TileDecodeContext<'_>,
    row: usize,
    col: usize,
    bl: BlockLevel,
) -> Result<()> {
    // Half-block size in 4×4 units: 8 >> bl  (64→8, 32→4, 16→2, 8→1)
    let hbs: usize = 8 >> (bl as usize);
    let is_kf = td.header.frame_type == FrameType::KeyFrame || td.header.intra_only;
    // Derive partition probability context from above/left partition context.
    // FFmpeg:  c = ((above_partition_ctx[col] >> (3-bl)) & 1)
    //            | (((left_partition_ctx[row&7] >> (3-bl)) & 1) << 1)
    let above_bit = if col < td.above.partition.len() {
        (td.above.partition[col] >> (3 - bl as u8)) & 1
    } else {
        0
    };
    let row7 = (row >> 1) & 7;
    let left_bit = (td.left.partition[row7] >> (3 - bl as u8)) & 1;
    let c = (above_bit | (left_bit << 1)) as usize;

    // Choose probability table.
    let p: [u8; 3] = if is_kf {
        KF_PARTITION_PROBS[bl as usize][c]
    } else {
        td.header.prob.partition[bl as usize][above_bit as usize][left_bit as usize]
    };

    let cols = td.cols_4x4;
    let rows = td.rows_4x4;

    let bp: BlockPartition;

    if bl == BlockLevel::Bl8x8 {
        let tree_val = td.bd.get_tree(&PARTITION_TREE, &p);
        bp = BlockPartition::try_from(tree_val as u8).map_err(|_| Error::InvalidData)?;
        decode_block(td, row, col, bl, bp)?;
    } else if col + hbs < cols {
        if row + hbs < rows {
            let tree_val = td.bd.get_tree(&PARTITION_TREE, &p);
            bp = BlockPartition::try_from(tree_val as u8).map_err(|_| Error::InvalidData)?;
            match bp {
                BlockPartition::None => {
                    decode_block(td, row, col, bl, bp)?;
                }
                BlockPartition::H => {
                    decode_block(td, row, col, bl, bp)?;
                    decode_block(td, row + hbs, col, bl, bp)?;
                }
                BlockPartition::V => {
                    decode_block(td, row, col, bl, bp)?;
                    decode_block(td, row, col + hbs, bl, bp)?;
                }
                BlockPartition::Split => {
                    let next_bl =
                        BlockLevel::try_from(bl as u8 + 1).map_err(|_| Error::InvalidData)?;
                    decode_sb(td, row, col, next_bl)?;
                    decode_sb(td, row, col + hbs, next_bl)?;
                    decode_sb(td, row + hbs, col, next_bl)?;
                    decode_sb(td, row + hbs, col + hbs, next_bl)?;
                }
            }
        } else {
            // Row overhang: can only be SPLIT or H.
            if td.bd.get_prob(p[1]) {
                bp = BlockPartition::Split;
                let next_bl = BlockLevel::try_from(bl as u8 + 1).map_err(|_| Error::InvalidData)?;
                decode_sb(td, row, col, next_bl)?;
                decode_sb(td, row, col + hbs, next_bl)?;
            } else {
                bp = BlockPartition::H;
                decode_block(td, row, col, bl, bp)?;
            }
        }
    } else if row + hbs < rows {
        // Column overhang: can only be SPLIT or V.
        if td.bd.get_prob(p[2]) {
            bp = BlockPartition::Split;
            let next_bl = BlockLevel::try_from(bl as u8 + 1).map_err(|_| Error::InvalidData)?;
            decode_sb(td, row, col, next_bl)?;
            decode_sb(td, row + hbs, col, next_bl)?;
        } else {
            bp = BlockPartition::V;
            decode_block(td, row, col, bl, bp)?;
        }
    } else {
        // Corner block, must split.
        bp = BlockPartition::Split;
        let next_bl = BlockLevel::try_from(bl as u8 + 1).map_err(|_| Error::InvalidData)?;
        decode_sb(td, row, col, next_bl)?;
    }

    // Accumulate partition count.
    // counts.partition is [1][4][4][4] (only used for inter; index [0] here).
    td.counts.partition[0][bl as usize][c][bp as usize] += 1;

    Ok(())
}

// ---------------------------------------------------------------------------
// decode_block — single block decode  (vp9block.c: ff_vp9_decode_block)
// ---------------------------------------------------------------------------

/// Derive the block size from its level and partition.
///
/// Mirrors `enum BlockSize bs = bl * 3 + bp` in vp9block.c.
fn bs_from_bl_bp(bl: BlockLevel, bp: BlockPartition) -> Result<BlockSize> {
    let v = bl as u8 * 3 + bp as u8;
    BlockSize::try_from(v).map_err(|_| Error::InvalidData)
}

/// Decode one block at (`row`, `col`) in 4×4 units.
///
/// Corresponds to `ff_vp9_decode_block` in vp9block.c.
fn decode_block(
    td: &mut TileDecodeContext<'_>,
    row: usize,
    col: usize,
    bl: BlockLevel,
    bp: BlockPartition,
) -> Result<()> {
    let bs = bs_from_bl_bp(bl, bp)?;
    let is_kf = td.header.frame_type == FrameType::KeyFrame || td.header.intra_only;

    // --- Segment ID ---
    let segment_id: u8 = if td.header.segmentation.enabled && td.header.segmentation.update_map {
        td.bd.get_tree(
            &crate::data::SEGMENTATION_TREE,
            &td.header.segmentation.prob,
        ) as u8
    } else {
        0
    };

    // --- Skip flag ---
    let skip = if td.header.segmentation.enabled
        && td.header.segmentation.feat[segment_id as usize].skip_enabled
    {
        true
    } else {
        let row7 = (row >> 1) & 7;
        let c = td.left.skip[row7] as usize + td.above.skip.get(col).copied().unwrap_or(0) as usize;
        let s = td.bd.get_prob(td.header.prob.skip[c.min(2)]);
        td.counts.skip[c.min(2)][usize::from(s)] += 1;
        s
    };

    // --- Intra/inter decision ---
    let is_intra = if is_kf {
        true
    } else if td.header.segmentation.enabled
        && td.header.segmentation.feat[segment_id as usize].ref_enabled
    {
        td.header.segmentation.feat[segment_id as usize].ref_val == 0
    } else {
        let row7 = (row >> 1) & 7;
        let have_a = row > 0;
        let have_l = col > td.tile_col_start;
        let c = if have_a && have_l {
            let ai = td.above.intra.get(col).copied().unwrap_or(1) as usize;
            let li = td.left.intra[row7] as usize;
            let s = ai + li;
            s + (s == 2) as usize
        } else if have_a {
            2 * td.above.intra.get(col).copied().unwrap_or(1) as usize
        } else if have_l {
            2 * td.left.intra[row7] as usize
        } else {
            0
        };
        let bit = td.bd.get_prob(td.header.prob.intra[c.min(3)]);
        td.counts.intra[c.min(3)][bit as usize] += 1;
        !bit
    };

    // --- Transform size ---
    let max_tx = MAX_TX_FOR_BS[bs as usize];
    let tx_size = if (is_intra || !skip) && td.header.tx_mode == TxMode::TxModeSelect {
        decode_tx_size(td, bs, max_tx, skip, row, col, is_intra)?
    } else {
        let mode_limit = match td.header.tx_mode {
            TxMode::Only4x4 => 0u8,
            TxMode::Allow8x8 => 1u8,
            TxMode::Allow16x16 => 2u8,
            TxMode::Allow32x32 => 3u8,
            TxMode::TxModeSelect => max_tx,
        };
        TxSize::try_from(mode_limit.min(max_tx)).map_err(|_| Error::InvalidData)?
    };

    // --- UV transform size ---
    let w4 = BWH_TAB[1][bs as usize][0] as usize;
    let h4 = BWH_TAB[1][bs as usize][1] as usize;
    let ss_h = td.header.subsampling_x;
    let ss_v = td.header.subsampling_y;
    let tx_val = tx_size as u8;
    let uv_tx_val = tx_val.saturating_sub(
        u8::from(ss_h && w4 * 2 == (1usize << tx_val))
            | u8::from(ss_v && h4 * 2 == (1usize << tx_val)),
    );
    let uv_tx_size = TxSize::try_from(uv_tx_val).unwrap_or(TxSize::Tx4x4);

    // --- Mode decode ---
    let mut comp = false;
    let mut ref_frame: [i8; 2] = [-1, -1];
    let mut inter_mode_arr = [0u8; 4];
    let mut mv = [[[0i16; 2]; 2]; 4];
    let mut filter = 0u8;
    let mut filter_id = 0u8;

    let (y_mode, uv_mode) = if is_kf {
        // Keyframe intra mode
        decode_intra_mode(td, bs, row, col)?
    } else if is_intra {
        // Non-keyframe intra mode
        comp = false;
        decode_non_kf_intra_mode(td, bs)?
    } else {
        // Inter mode
        let r = decode_inter_mode(td, bs, row, col, skip, segment_id)?;
        comp = r.comp;
        ref_frame = r.ref_frame;
        inter_mode_arr = r.inter_mode;
        mv = r.mv;
        filter = r.filter;
        filter_id = r.filter_id;
        // Inter blocks: y_mode unused for decode, but mode context needs
        // inter_mode[3] (10-13).  We still store DcPred in BlockInfo.y_mode;
        // the correct value is passed to update_ctx via mode_ctx_val below.
        ([IntraMode::DcPred; 4], IntraMode::DcPred)
    };

    let is_inter = !is_intra;
    // --- Coefficient decoding ---
    let mut skip = skip;
    let (coefs_y, coefs_u, coefs_v) = if skip {
        zero_nnz_ctx(td.above, td.left, col, row, w4, h4, ss_h, ss_v);
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        let result = decode_coeffs(
            td, bs, tx_size, uv_tx_size, segment_id, row, col, &y_mode, is_inter,
        )?;
        // FFmpeg vp9block.c:1310–1314: upgrade skip for inter blocks >= 8x8
        // when all coefficients are zero.
        if is_inter
            && (bs as u8) <= (BlockSize::Bs8x8 as u8)
            && result.0.iter().all(|&c| c == 0)
            && result.1.iter().all(|&c| c == 0)
            && result.2.iter().all(|&c| c == 0)
        {
            skip = true;
            let row7 = (row >> 1) & 7;
            for c in col..col + w4 {
                if c < td.above.skip.len() {
                    td.above.skip[c] = 1;
                }
            }
            for r in row7..row7 + h4 {
                if r < td.left.skip.len() {
                    td.left.skip[r] = 1;
                }
            }
        }
        result
    };

    // --- vref for context ---
    let vref = if is_inter {
        let hdr = td.header;
        ref_frame[if comp {
            hdr.sign_bias[hdr.var_comp_ref[0] as usize] as usize
        } else {
            0
        }] as u8
    } else {
        0
    };

    // --- Update context arrays ---
    // For inter blocks, FFmpeg's SET_CTXS writes b->mode[3] (inter mode 10-13)
    // to the y_mode context — NOT IntraMode::DcPred.  This affects inter mode
    // context derivation and filter context for subsequent blocks.
    let mode_ctx_val = if is_inter {
        inter_mode_arr[3]
    } else {
        y_mode[3] as u8
    };
    update_ctx(
        td,
        bs,
        tx_size,
        skip,
        segment_id,
        mode_ctx_val,
        col,
        row,
        is_kf,
        is_inter,
        comp,
        &ref_frame,
        filter,
        filter_id,
        vref,
    );

    // --- MV context and grid update ---
    if !is_kf {
        update_mv_ctx(
            td,
            bs,
            row,
            col,
            is_intra,
            comp,
            &ref_frame,
            &inter_mode_arr,
            &mv,
        );
    }

    td.blocks.push(BlockInfo {
        row,
        col,
        bs,
        tx_size,
        uv_tx_size,
        y_mode,
        uv_mode,
        skip,
        segment_id,
        tile_col_start: td.tile_col_start,
        is_inter,
        comp,
        ref_frame,
        inter_mode: inter_mode_arr,
        mv,
        filter,
        coefs_y,
        coefs_u,
        coefs_v,
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// decode_tx_size
// ---------------------------------------------------------------------------

/// Decode the transform size for a block.
///
/// Mirrors the tx-size decoding in `decode_mode()` (vp9block.c lines 171–215).
fn decode_tx_size(
    td: &mut TileDecodeContext<'_>,
    _bs: BlockSize,
    max_tx: u8,
    _skip: bool,
    row: usize,
    col: usize,
    _is_intra: bool,
) -> Result<TxSize> {
    // The caller already gates on `(is_intra || !skip) && TxModeSelect`.
    // FFmpeg's condition (b->intra || !b->skip) is equivalent — no inner check needed.
    {
        let row7 = (row >> 1) & 7;
        // Derive context: see vp9block.c lines 172–212
        let have_a = row > 0;
        let have_l = col > td.tile_col_start;
        let c = if have_a {
            if have_l {
                let a_tx = if td.above.skip.get(col).copied().unwrap_or(0) != 0 {
                    max_tx
                } else {
                    td.above.tx_size.get(col).copied().unwrap_or(0)
                };
                let l_tx = if td.left.skip[row7] != 0 {
                    max_tx
                } else {
                    td.left.tx_size[row7]
                };
                usize::from((a_tx + l_tx) > max_tx)
            } else {
                let a_skip = td.above.skip.get(col).copied().unwrap_or(0) != 0;
                if a_skip {
                    1
                } else {
                    usize::from(td.above.tx_size.get(col).copied().unwrap_or(0) * 2 > max_tx)
                }
            }
        } else if have_l {
            let l_skip = td.left.skip[row7] != 0;
            if l_skip {
                1
            } else {
                usize::from(td.left.tx_size[row7] * 2 > max_tx)
            }
        } else {
            1
        };

        let tx = match max_tx {
            3 => {
                // TX_32X32
                let b0 = td.bd.get_prob(td.header.prob.tx32p[c][0]);
                let t = if b0 {
                    let b1 = td.bd.get_prob(td.header.prob.tx32p[c][1]);
                    if b1 {
                        let b2 = td.bd.get_prob(td.header.prob.tx32p[c][2]);
                        2 + u8::from(b2)
                    } else {
                        1
                    }
                } else {
                    0
                };
                td.counts.tx32p[c][t as usize] += 1;
                t
            }
            2 => {
                // TX_16X16
                let b0 = td.bd.get_prob(td.header.prob.tx16p[c][0]);
                let t = if b0 {
                    let b1 = td.bd.get_prob(td.header.prob.tx16p[c][1]);
                    1 + u8::from(b1)
                } else {
                    0
                };
                td.counts.tx16p[c][t as usize] += 1;
                t
            }
            1 => {
                // TX_8X8
                let b0 = td.bd.get_prob(td.header.prob.tx8p[c]);
                let t = u8::from(b0);
                td.counts.tx8p[c][t as usize] += 1;
                t
            }
            _ => 0, // TX_4X4
        };
        TxSize::try_from(tx).map_err(|_| Error::InvalidData)
    }
}

// ---------------------------------------------------------------------------
// decode_intra_mode  (keyframe path of decode_mode in vp9block.c)
// ---------------------------------------------------------------------------

/// Decode intra prediction modes for a keyframe block.
///
/// Returns `([y_mode; 4], uv_mode)`.  For blocks larger than 8×8 the four
/// entries cover the four 4×4-sub-block corners; for 8×8 all four are equal.
///
/// Mirrors the keyframe branch in `decode_mode()` (vp9block.c lines 217–270).
fn decode_intra_mode(
    td: &mut TileDecodeContext<'_>,
    bs: BlockSize,
    row: usize,
    col: usize,
) -> Result<([IntraMode; 4], IntraMode)> {
    // above_mode_ctx and left_mode_ctx pointers in FFmpeg; 2 entries per 4×4.
    // We read them from our context arrays.
    let a0 = *td.above.y_mode.get(col * 2).unwrap_or(&2u8);
    let a1 = *td.above.y_mode.get(col * 2 + 1).unwrap_or(&2u8);
    let row7 = (row >> 1) & 7;
    let l0 = td.left.y_mode[row7 * 2];
    let l1 = td.left.y_mode[row7 * 2 + 1];

    let is_sub8x8 = matches!(bs, BlockSize::Bs8x4 | BlockSize::Bs4x8 | BlockSize::Bs4x4);

    let (y_modes, final_a0, final_a1, final_l0, final_l1) = if is_sub8x8 {
        // Sub-8×8: multiple 4×4 modes
        let m0 = read_kf_y_mode(&mut td.bd, a0, l0)?;
        // mode[1]: read if bs != BS_8x4 (FFmpeg vp9block.c:229)
        let (m1, al1) = if bs != BlockSize::Bs8x4 {
            let m = read_kf_y_mode(&mut td.bd, a1, m0 as u8)?;
            (m, m as u8)
        } else {
            (m0, m0 as u8)
        };

        // mode[2]: read if bs != BS_4x8 (FFmpeg vp9block.c:239)
        // FFmpeg: a[0] was overwritten with mode[0] before this read.
        let (m2, al0_new) = if bs != BlockSize::Bs4x8 {
            let m = read_kf_y_mode(&mut td.bd, m0 as u8, l1)?;
            (m, m as u8)
        } else {
            (m0, m0 as u8)
        };
        let (m3, al1_new) = if bs == BlockSize::Bs4x4 {
            let m = read_kf_y_mode(&mut td.bd, al1, m2 as u8)?;
            (m, m as u8)
        } else if bs == BlockSize::Bs4x8 {
            (m1, al1)
        } else {
            // Bs8x4
            (m2, al0_new)
        };

        ([m0, m1, m2, m3], al0_new, al1_new, m1 as u8, m3 as u8)
    } else {
        // 8×8 or larger: single mode for the block.
        // bwh[0] entries wide, bwh[1] entries tall in above_mode_ctx;
        // update_mode_ctx handles the fill.
        let m = read_kf_y_mode(&mut td.bd, a0, l0)?;
        let mu = m as u8;
        ([m, m, m, m], mu, mu, mu, mu)
    };

    // Update above/left mode context.
    update_mode_ctx(
        td.above,
        td.left,
        bs,
        col,
        (row >> 1) & 7,
        ModeCtxUpdate {
            a0: final_a0,
            a1: final_a1,
            l0: final_l0,
            l1: final_l1,
        },
    );

    // UV mode: always from last y mode.
    let last_y = y_modes[3] as u8;
    let last_y_clamped = (last_y as usize).min(9); // KF_UV_MODE_PROBS has 10 entries
    let uv_val = td
        .bd
        .get_tree(&INTRAMODE_TREE, &KF_UV_MODE_PROBS[last_y_clamped]);
    let uv_mode = IntraMode::try_from(uv_val as u8).map_err(|_| Error::InvalidData)?;

    Ok((y_modes, uv_mode))
}

/// Read one keyframe Y intra mode using KF_Y_MODE_PROBS[above][left].
#[inline]
fn read_kf_y_mode(bd: &mut BoolDecoder<'_>, above: u8, left: u8) -> Result<IntraMode> {
    let a = (above as usize).min(9);
    let l = (left as usize).min(9);
    let v = bd.get_tree(&INTRAMODE_TREE, &KF_Y_MODE_PROBS[a][l]);
    IntraMode::try_from(v as u8).map_err(|_| Error::InvalidData)
}

/// Packed above/left mode context values for update_mode_ctx.
struct ModeCtxUpdate {
    a0: u8,
    a1: u8,
    l0: u8,
    l1: u8,
}

/// Write intra mode values back to the above/left mode context arrays.
///
/// For blocks larger than 4×4 we fill the width/height of the block.
fn update_mode_ctx(
    above: &mut AboveContext,
    left: &mut LeftContext,
    bs: BlockSize,
    col: usize,
    row7: usize,
    ctx: ModeCtxUpdate,
) {
    // Above mode context: 2 entries per 4×4 col → use BWH_TAB[0] width (4×4 units).
    let bw = BWH_TAB[0][bs as usize][0] as usize;
    // Left mode context: 2 entries per 8×8 row → use BWH_TAB[1] height (8×8 units).
    let bh = BWH_TAB[1][bs as usize][1] as usize;

    // above_mode_ctx uses 2 entries per 4×4 column.
    let base_a = col * 2;
    let is_sub = matches!(bs, BlockSize::Bs8x4 | BlockSize::Bs4x8 | BlockSize::Bs4x4);
    if is_sub {
        if base_a < above.y_mode.len() {
            above.y_mode[base_a] = ctx.a0;
        }
        if base_a + 1 < above.y_mode.len() {
            above.y_mode[base_a + 1] = ctx.a1;
        }
        let base_l = row7 * 2;
        if base_l < left.y_mode.len() {
            left.y_mode[base_l] = ctx.l0;
        }
        if base_l + 1 < left.y_mode.len() {
            left.y_mode[base_l + 1] = ctx.l1;
        }
    } else {
        // Fill bw*2 above entries and bh*2 left entries with ctx.a0.
        let end_a = (base_a + bw * 2).min(above.y_mode.len());
        above.y_mode[base_a..end_a].fill(ctx.a0);
        let base_l = row7 * 2;
        let end_l = (base_l + bh * 2).min(left.y_mode.len());
        left.y_mode[base_l..end_l].fill(ctx.l0);
    }
}

// ---------------------------------------------------------------------------
// decode_non_kf_intra_mode  (non-keyframe intra, vp9block.c lines 271-314)
// ---------------------------------------------------------------------------

/// Decode intra prediction modes for a non-keyframe intra block.
///
/// Uses `prob.y_mode` / `prob.uv_mode` instead of the KF tables.
/// Mirrors FFmpeg vp9block.c lines 271-314.
fn decode_non_kf_intra_mode(
    td: &mut TileDecodeContext<'_>,
    bs: BlockSize,
) -> Result<([IntraMode; 4], IntraMode)> {
    let is_sub8x8 = (bs as u8) > (BlockSize::Bs8x8 as u8);

    let y_modes = if is_sub8x8 {
        // Sub-8×8: read up to 4 modes from prob.y_mode[0] (size group 0).
        let m0 = td.bd.get_tree(&INTRAMODE_TREE, &td.header.prob.y_mode[0]);
        let m0 = IntraMode::try_from(m0 as u8).map_err(|_| Error::InvalidData)?;
        td.counts.y_mode[0][m0 as usize] += 1;

        let m1 = if bs != BlockSize::Bs8x4 {
            let m = td.bd.get_tree(&INTRAMODE_TREE, &td.header.prob.y_mode[0]);
            let m = IntraMode::try_from(m as u8).map_err(|_| Error::InvalidData)?;
            td.counts.y_mode[0][m as usize] += 1;
            m
        } else {
            m0
        };

        let (m2, m3) = if bs != BlockSize::Bs4x8 {
            let m2 = td.bd.get_tree(&INTRAMODE_TREE, &td.header.prob.y_mode[0]);
            let m2 = IntraMode::try_from(m2 as u8).map_err(|_| Error::InvalidData)?;
            td.counts.y_mode[0][m2 as usize] += 1;
            let m3 = if bs != BlockSize::Bs8x4 {
                let m = td.bd.get_tree(&INTRAMODE_TREE, &td.header.prob.y_mode[0]);
                let m = IntraMode::try_from(m as u8).map_err(|_| Error::InvalidData)?;
                td.counts.y_mode[0][m as usize] += 1;
                m
            } else {
                m2
            };
            (m2, m3)
        } else {
            (m0, m1)
        };

        [m0, m1, m2, m3]
    } else {
        // >=8×8: single mode from prob.y_mode[SIZE_GROUP[bs]].
        let sz = SIZE_GROUP[bs as usize] as usize;
        let m = td.bd.get_tree(&INTRAMODE_TREE, &td.header.prob.y_mode[sz]);
        let m = IntraMode::try_from(m as u8).map_err(|_| Error::InvalidData)?;
        td.counts.y_mode[sz][m as usize] += 1;
        [m, m, m, m]
    };

    // UV mode from prob.uv_mode[last_y_mode].
    let last_y = y_modes[3] as usize;
    let uv_val = td
        .bd
        .get_tree(&INTRAMODE_TREE, &td.header.prob.uv_mode[last_y.min(9)]);
    let uv_mode = IntraMode::try_from(uv_val as u8).map_err(|_| Error::InvalidData)?;
    td.counts.uv_mode[last_y.min(9)][uv_mode as usize] += 1;

    Ok((y_modes, uv_mode))
}

// ---------------------------------------------------------------------------
// decode_inter_mode  (inter prediction, vp9block.c lines 315-680)
// ---------------------------------------------------------------------------

/// Output struct for inter mode decode.
struct InterModeResult {
    comp: bool,
    ref_frame: [i8; 2],
    inter_mode: [u8; 4],
    mv: [[[i16; 2]; 2]; 4],
    filter: u8,
    filter_id: u8,
}

/// Decode inter prediction mode: comp flag, ref frames, inter mode, MVs, filter.
///
/// Translates FFmpeg vp9block.c lines 315-680.
#[allow(clippy::too_many_arguments)]
fn decode_inter_mode(
    td: &mut TileDecodeContext<'_>,
    bs: BlockSize,
    row: usize,
    col: usize,
    _skip: bool,
    segment_id: u8,
) -> Result<InterModeResult> {
    let row7 = (row >> 1) & 7;
    let have_a = row > 0;
    let have_l = col > td.tile_col_start;
    let hdr = td.header;
    let seg_feat = &hdr.segmentation.feat[segment_id as usize];

    let comp;
    let mut ref_frame: [i8; 2] = [-1, -1];

    // --- Comp pred flag + ref frame selection ---
    if hdr.segmentation.enabled && seg_feat.ref_enabled {
        comp = false;
        ref_frame[0] = seg_feat.ref_val as i8 - 1;
    } else {
        // Comp pred flag
        if hdr.comp_pred_mode != 2 {
            // Not switchable: forced
            comp = hdr.comp_pred_mode == 1;
        } else {
            // Switchable: derive context and read
            let c = comp_pred_ctx(
                td.above,
                td.left,
                col,
                row7,
                have_a,
                have_l,
                hdr.fix_comp_ref,
            );
            comp = td.bd.get_prob(hdr.prob.comp[c]);
            td.counts.comp[c][comp as usize] += 1;
        }

        if comp {
            // Compound reference selection
            let fix_idx = hdr.sign_bias[hdr.fix_comp_ref as usize] as usize;
            let var_idx = 1 - fix_idx;
            ref_frame[fix_idx] = hdr.fix_comp_ref as i8;

            let c = comp_ref_ctx(td.above, td.left, col, row7, have_a, have_l, hdr);
            let bit = td.bd.get_prob(hdr.prob.comp_ref[c]);
            td.counts.comp_ref[c][bit as usize] += 1;
            ref_frame[var_idx] = hdr.var_comp_ref[bit as usize] as i8;
        } else {
            // Single reference selection
            let c = single_ref_ctx_0(td.above, td.left, col, row7, have_a, have_l, hdr);
            let bit0 = td.bd.get_prob(hdr.prob.single_ref[c][0]);
            td.counts.single_ref[c][0][bit0 as usize] += 1;
            if !bit0 {
                ref_frame[0] = 0; // LAST
            } else {
                let c2 = single_ref_ctx_1(td.above, td.left, col, row7, have_a, have_l, hdr);
                let bit1 = td.bd.get_prob(hdr.prob.single_ref[c2][1]);
                td.counts.single_ref[c2][1][bit1 as usize] += 1;
                ref_frame[0] = 1 + bit1 as i8; // GOLDEN or ALTREF
            }
        }
    }

    let mut inter_mode = [0u8; 4];
    let mut mv = [[[0i16; 2]; 2]; 4];

    let is_sub8x8 = (bs as u8) > (BlockSize::Bs8x8 as u8);

    // --- Inter mode for >=8x8 blocks (read before filter) ---
    if !is_sub8x8 {
        if hdr.segmentation.enabled && seg_feat.skip_enabled {
            inter_mode = [12; 4]; // ZEROMV
        } else {
            let off = INTER_MODE_CTX_OFF[bs as usize] as usize;
            // Inter-frame mode ctx uses 1 entry per 8×8 unit: col/2 + off, row7 + off
            let a_mode = td.above.y_mode.get(col / 2 + off).copied().unwrap_or(10) as usize;
            let l_mode = td.left.y_mode.get(row7 + off).copied().unwrap_or(10) as usize;
            let c = INTER_MODE_CTX_LUT[a_mode.min(13)][l_mode.min(13)] as usize;
            let m = td.bd.get_tree(&INTER_MODE_TREE, &hdr.prob.mv_mode[c]);
            inter_mode = [m as u8; 4];
            td.counts.mv_mode[c][(m as u8).wrapping_sub(10) as usize] += 1;
        }
    }

    // --- Filter mode ---
    let filter;
    let filter_id;
    if hdr.filter_mode == 4 {
        // FILTER_SWITCHABLE
        // Inter-frame mode ctx uses 1 entry per 8×8: above_mode_ctx[col], left_mode_ctx[row7]
        let c = if have_a && td.above.y_mode.get(col / 2).copied().unwrap_or(0) >= 10 {
            if have_l && td.left.y_mode[row7] >= 10 {
                let af = td.above.filter.get(col).copied().unwrap_or(3);
                let lf = td.left.filter[row7];
                if af == lf { lf as usize } else { 3 }
            } else {
                td.above.filter.get(col).copied().unwrap_or(3) as usize
            }
        } else if have_l && td.left.y_mode[row7] >= 10 {
            td.left.filter[row7] as usize
        } else {
            3
        };
        filter_id = td.bd.get_tree(&FILTER_TREE, &hdr.prob.filter[c.min(3)]) as u8;
        td.counts.filter[c.min(3)][filter_id as usize] += 1;
        filter = FILTER_LUT[filter_id as usize];
    } else {
        filter = hdr.filter_mode;
        filter_id = 0;
    }



    // --- MV search context ---
    // MV clamping bounds: cols_4x4/rows_4x4 and col/row are in 4×4 units,
    // so bw4/bh4 must also be in 4×4 units (BWH_TAB[0], not BWH_TAB[1]).
    let bw4 = BWH_TAB[0][bs as usize][0] as i32;
    let bh4 = BWH_TAB[0][bs as usize][1] as i32;
    let mv_ctx = MvSearchCtx {
        min_mv: [
            (-(128 + col as i32 * 32)).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            (-(128 + row as i32 * 32)).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        ],
        max_mv: [
            (128 + (td.cols_4x4 as i32 - col as i32 - bw4) * 32)
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            (128 + (td.rows_4x4 as i32 - row as i32 - bh4) * 32)
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        ],
        tile_col_start: td.tile_col_start,
        cols_4x4: td.cols_4x4,
        rows_4x4: td.rows_4x4,
        sign_bias: td.sign_bias,
    };

    // --- Sub-8x8: per-sub-block mode + MV fill ---
    // Macro-like helper: call fill_mv with snapshot of sub_mvs to avoid borrow conflict.
    macro_rules! do_fill_mv {
        ($td:expr, $mv:expr, $inter_mode:expr, $sb:expr) => {{
            let sub_mvs_snap = $mv; // copy current state
            fill_mv(
                &mut $td.bd,
                &mv_ctx,
                $td.cur_mv_grid,
                $td.prev_mv_grid,
                $td.prev_cols_4x4,
                &$td.above.mv[..],
                &$td.left.mv[..],
                &mut $mv[$sb as usize],
                $inter_mode[$sb as usize],
                $sb,
                row,
                col,
                row7,
                bs,
                comp,
                &ref_frame,
                &sub_mvs_snap,
                &hdr.prob,
                &mut $td.counts,
                hdr.high_precision_mvs,
                hdr.use_last_frame_mvs,
            );
        }};
    }

    if is_sub8x8 {
        // Inter-frame mode ctx: 1 entry per 8×8 → col/2, row7
        let a_mode = td.above.y_mode.get(col / 2).copied().unwrap_or(10) as usize;
        let l_mode = (td.left.y_mode[row7] as usize).min(13);
        let c = INTER_MODE_CTX_LUT[a_mode][l_mode] as usize;
        // Read mode[0]
        inter_mode[0] = td.bd.get_tree(&INTER_MODE_TREE, &hdr.prob.mv_mode[c]) as u8;
        td.counts.mv_mode[c][inter_mode[0].wrapping_sub(10) as usize] += 1;
        do_fill_mv!(td, mv, inter_mode, 0i32);

        if bs != BlockSize::Bs8x4 {
            inter_mode[1] = td.bd.get_tree(&INTER_MODE_TREE, &hdr.prob.mv_mode[c]) as u8;
            td.counts.mv_mode[c][inter_mode[1].wrapping_sub(10) as usize] += 1;
            do_fill_mv!(td, mv, inter_mode, 1i32);
        } else {
            inter_mode[1] = inter_mode[0];
            mv[1] = mv[0];
        }

        if bs != BlockSize::Bs4x8 {
            inter_mode[2] = td.bd.get_tree(&INTER_MODE_TREE, &hdr.prob.mv_mode[c]) as u8;
            td.counts.mv_mode[c][inter_mode[2].wrapping_sub(10) as usize] += 1;
            do_fill_mv!(td, mv, inter_mode, 2i32);

            if bs != BlockSize::Bs8x4 {
                inter_mode[3] = td.bd.get_tree(&INTER_MODE_TREE, &hdr.prob.mv_mode[c]) as u8;
                td.counts.mv_mode[c][inter_mode[3].wrapping_sub(10) as usize] += 1;
                do_fill_mv!(td, mv, inter_mode, 3i32);
            } else {
                inter_mode[3] = inter_mode[2];
                mv[3] = mv[2];
            }
        } else {
            inter_mode[2] = inter_mode[0];
            mv[2] = mv[0];
            inter_mode[3] = inter_mode[1];
            mv[3] = mv[1];
        }
    } else {
        // >=8x8: single fill_mv with sb=-1, broadcast
        let sub_mvs_snap = mv;
        fill_mv(
            &mut td.bd,
            &mv_ctx,
            td.cur_mv_grid,
            td.prev_mv_grid,
            td.prev_cols_4x4,
            &td.above.mv[..],
            &td.left.mv[..],
            &mut mv[0],
            inter_mode[0],
            -1,
            row,
            col,
            row7,
            bs,
            comp,
            &ref_frame,
            &sub_mvs_snap,
            &hdr.prob,
            &mut td.counts,
            hdr.high_precision_mvs,
            hdr.use_last_frame_mvs,
        );
        mv[1] = mv[0];
        mv[2] = mv[0];
        mv[3] = mv[0];
    }

    Ok(InterModeResult {
        comp,
        ref_frame,
        inter_mode,
        mv,
        filter,
        filter_id,
    })
}

// ---------------------------------------------------------------------------
// Inter context derivation helpers
// ---------------------------------------------------------------------------

/// Comp pred context derivation (FFmpeg vp9block.c lines 333-373).
fn comp_pred_ctx(
    above: &AboveContext,
    left: &LeftContext,
    col: usize,
    row7: usize,
    have_a: bool,
    have_l: bool,
    fix_comp_ref: u8,
) -> usize {
    if have_a {
        if have_l {
            let ac = above.comp.get(col).copied().unwrap_or(0) != 0;
            let lc = left.comp[row7] != 0;
            if ac && lc {
                4
            } else if ac {
                2 + (left.intra[row7] != 0 || left.ref_frame[row7] == fix_comp_ref) as usize
            } else if lc {
                2 + (above.intra.get(col).copied().unwrap_or(1) != 0
                    || above.ref_frame.get(col).copied().unwrap_or(0) == fix_comp_ref)
                    as usize
            } else {
                let a_match = above.intra.get(col).copied().unwrap_or(1) == 0
                    && above.ref_frame.get(col).copied().unwrap_or(0) == fix_comp_ref;
                let l_match = left.intra[row7] == 0 && left.ref_frame[row7] == fix_comp_ref;
                (a_match ^ l_match) as usize
            }
        } else {
            let ac = above.comp.get(col).copied().unwrap_or(0) != 0;
            if ac {
                3
            } else {
                (above.intra.get(col).copied().unwrap_or(1) == 0
                    && above.ref_frame.get(col).copied().unwrap_or(0) == fix_comp_ref)
                    as usize
            }
        }
    } else if have_l {
        let lc = left.comp[row7] != 0;
        if lc {
            3
        } else {
            (left.intra[row7] == 0 && left.ref_frame[row7] == fix_comp_ref) as usize
        }
    } else {
        1
    }
}

/// Compound ref context derivation (FFmpeg vp9block.c lines 378-445).
fn comp_ref_ctx(
    above: &AboveContext,
    left: &LeftContext,
    col: usize,
    row7: usize,
    have_a: bool,
    have_l: bool,
    hdr: &FrameHeader,
) -> usize {
    let vcr1 = hdr.var_comp_ref[1];
    let vcr0 = hdr.var_comp_ref[0];
    let fcr = hdr.fix_comp_ref;
    if have_a {
        if have_l {
            let ai = above.intra.get(col).copied().unwrap_or(1) != 0;
            let li = left.intra[row7] != 0;
            if ai {
                if li {
                    2
                } else {
                    1 + 2 * (left.ref_frame[row7] != vcr1) as usize
                }
            } else if li {
                1 + 2 * (above.ref_frame.get(col).copied().unwrap_or(0) != vcr1) as usize
            } else {
                let refl = left.ref_frame[row7];
                let refa = above.ref_frame.get(col).copied().unwrap_or(0);
                let ac = above.comp.get(col).copied().unwrap_or(0) != 0;
                let lc = left.comp[row7] != 0;

                if refl == refa && refa == vcr1 {
                    0
                } else if !lc && !ac {
                    if (refa == fcr && refl == vcr0) || (refl == fcr && refa == vcr0) {
                        4
                    } else if refa == refl {
                        3
                    } else {
                        1
                    }
                } else if !lc {
                    if refa == vcr1 && refl != vcr1 {
                        1
                    } else if refl == vcr1 && refa != vcr1 {
                        2
                    } else {
                        4
                    }
                } else if !ac {
                    if refl == vcr1 && refa != vcr1 {
                        1
                    } else if refa == vcr1 && refl != vcr1 {
                        2
                    } else {
                        4
                    }
                } else if refl == refa {
                    4
                } else {
                    2
                }
            }
        } else {
            let ai = above.intra.get(col).copied().unwrap_or(1) != 0;
            if ai {
                2
            } else if above.comp.get(col).copied().unwrap_or(0) != 0 {
                4 * (above.ref_frame.get(col).copied().unwrap_or(0) != vcr1) as usize
            } else {
                3 * (above.ref_frame.get(col).copied().unwrap_or(0) != vcr1) as usize
            }
        }
    } else if have_l {
        let li = left.intra[row7] != 0;
        if li {
            2
        } else if left.comp[row7] != 0 {
            4 * (left.ref_frame[row7] != vcr1) as usize
        } else {
            3 * (left.ref_frame[row7] != vcr1) as usize
        }
    } else {
        2
    }
}

/// Single ref context derivation — first bit (FFmpeg vp9block.c lines 446-500).
fn single_ref_ctx_0(
    above: &AboveContext,
    left: &LeftContext,
    col: usize,
    row7: usize,
    have_a: bool,
    have_l: bool,
    hdr: &FrameHeader,
) -> usize {
    let fcr = hdr.fix_comp_ref;
    if have_a && above.intra.get(col).copied().unwrap_or(1) == 0 {
        if have_l && left.intra[row7] == 0 {
            let lc = left.comp[row7] != 0;
            let ac = above.comp.get(col).copied().unwrap_or(0) != 0;
            if lc {
                if ac {
                    1 + (fcr == 0
                        || left.ref_frame[row7] == 0
                        || above.ref_frame.get(col).copied().unwrap_or(0) == 0)
                        as usize
                } else {
                    let ar = above.ref_frame.get(col).copied().unwrap_or(0);
                    (3 * (ar == 0) as usize) + (fcr == 0 || left.ref_frame[row7] == 0) as usize
                }
            } else if ac {
                let lr = left.ref_frame[row7];
                (3 * (lr == 0) as usize)
                    + (fcr == 0 || above.ref_frame.get(col).copied().unwrap_or(0) == 0) as usize
            } else {
                let lr = left.ref_frame[row7];
                let ar = above.ref_frame.get(col).copied().unwrap_or(0);
                2 * (lr == 0) as usize + 2 * (ar == 0) as usize
            }
        } else {
            let ai = above.intra.get(col).copied().unwrap_or(1) != 0;
            if ai {
                2
            } else {
                let ac = above.comp.get(col).copied().unwrap_or(0) != 0;
                if ac {
                    1 + (fcr == 0 || above.ref_frame.get(col).copied().unwrap_or(0) == 0) as usize
                } else {
                    4 * (above.ref_frame.get(col).copied().unwrap_or(0) == 0) as usize
                }
            }
        }
    } else if have_l && left.intra[row7] == 0 {
        let li = left.intra[row7] != 0;
        if li {
            2
        } else {
            let lc = left.comp[row7] != 0;
            if lc {
                1 + (fcr == 0 || left.ref_frame[row7] == 0) as usize
            } else {
                4 * (left.ref_frame[row7] == 0) as usize
            }
        }
    } else {
        2
    }
}

/// Single ref context derivation — second bit (FFmpeg vp9block.c lines 500-573).
fn single_ref_ctx_1(
    above: &AboveContext,
    left: &LeftContext,
    col: usize,
    row7: usize,
    have_a: bool,
    have_l: bool,
    hdr: &FrameHeader,
) -> usize {
    let fcr = hdr.fix_comp_ref;
    if have_a {
        if have_l {
            let ai = above.intra.get(col).copied().unwrap_or(1) != 0;
            let li = left.intra[row7] != 0;
            if li {
                if ai {
                    return 2;
                }
                let ac = above.comp.get(col).copied().unwrap_or(0) != 0;
                if ac {
                    return 1 + 2
                        * (fcr == 1 || above.ref_frame.get(col).copied().unwrap_or(0) == 1)
                            as usize;
                }
                let ar = above.ref_frame.get(col).copied().unwrap_or(0);
                return if ar == 0 { 3 } else { 4 * (ar == 1) as usize };
            } else if ai {
                if li {
                    return 2;
                }
                let lc = left.comp[row7] != 0;
                if lc {
                    return 1 + 2 * (fcr == 1 || left.ref_frame[row7] == 1) as usize;
                }
                let lr = left.ref_frame[row7];
                return if lr == 0 { 3 } else { 4 * (lr == 1) as usize };
            }
            let ac = above.comp.get(col).copied().unwrap_or(0) != 0;
            let lc = left.comp[row7] != 0;
            let lr = left.ref_frame[row7];
            let ar = above.ref_frame.get(col).copied().unwrap_or(0);
            if ac {
                if lc {
                    if lr == ar {
                        3 * (fcr == 1 || lr == 1) as usize
                    } else {
                        2
                    }
                } else if lr == 0 {
                    1 + 2 * (fcr == 1 || ar == 1) as usize
                } else {
                    3 * (lr == 1) as usize + (fcr == 1 || ar == 1) as usize
                }
            } else if lc {
                if ar == 0 {
                    1 + 2 * (fcr == 1 || lr == 1) as usize
                } else {
                    3 * (ar == 1) as usize + (fcr == 1 || lr == 1) as usize
                }
            } else if ar == 0 {
                if lr == 0 { 3 } else { 4 * (lr == 1) as usize }
            } else if lr == 0 {
                4 * (ar == 1) as usize
            } else {
                2 * (lr == 1) as usize + 2 * (ar == 1) as usize
            }
        } else {
            let ai = above.intra.get(col).copied().unwrap_or(1) != 0;
            let ac = above.comp.get(col).copied().unwrap_or(0) != 0;
            let ar = above.ref_frame.get(col).copied().unwrap_or(0);
            if ai || (!ac && ar == 0) {
                2
            } else if ac {
                3 * (fcr == 1 || ar == 1) as usize
            } else {
                4 * (ar == 1) as usize
            }
        }
    } else if have_l {
        let li = left.intra[row7] != 0;
        let lc = left.comp[row7] != 0;
        let lr = left.ref_frame[row7];
        if li || (!lc && lr == 0) {
            2
        } else if lc {
            3 * (fcr == 1 || lr == 1) as usize
        } else {
            4 * (lr == 1) as usize
        }
    } else {
        2
    }
}

// ---------------------------------------------------------------------------
// MV context and grid update (vp9block.c lines 751-801)
// ---------------------------------------------------------------------------

/// Update above/left MV context and the current frame's MV grid after
/// decoding an inter block.
///
/// Mirrors FFmpeg vp9block.c lines 751-801.
#[allow(clippy::too_many_arguments)]
fn update_mv_ctx(
    td: &mut TileDecodeContext<'_>,
    bs: BlockSize,
    row: usize,
    col: usize,
    is_intra: bool,
    comp: bool,
    ref_frame: &[i8; 2],
    _inter_mode: &[u8; 4],
    mv: &[[[i16; 2]; 2]; 4],
) {
    let w4 = BWH_TAB[1][bs as usize][0] as usize;
    let h4 = BWH_TAB[1][bs as usize][1] as usize;
    let row7 = (row >> 1) & 7;
    let is_sub8x8 = (bs as u8) > (BlockSize::Bs8x8 as u8);

    // Above/left MV context update
    if !is_intra {
        if is_sub8x8 {
            // Sub-8×8: write specific sub-block MVs
            let mv3 = mv[3];
            // Left MV context
            if row7 * 2 < td.left.mv.len() {
                td.left.mv[row7 * 2] = mv[1];
            }
            if row7 * 2 + 1 < td.left.mv.len() {
                td.left.mv[row7 * 2 + 1] = mv3;
            }
            // Above MV context
            let ac = col * 2;
            if ac < td.above.mv.len() {
                td.above.mv[ac] = mv[2];
            }
            if ac + 1 < td.above.mv.len() {
                td.above.mv[ac + 1] = mv3;
            }
        } else {
            // >=8×8: broadcast mv[3] across w4*2 above and h4*2 left
            let mv3 = mv[3];
            for n in 0..w4 * 2 {
                let idx = col * 2 + n;
                if idx < td.above.mv.len() {
                    td.above.mv[idx] = mv3;
                }
            }
            for n in 0..h4 * 2 {
                let idx = row7 * 2 + n;
                if idx < td.left.mv.len() {
                    td.left.mv[idx] = mv3;
                }
            }
        }
    }

    // MV grid update: write MvRefPair to cur_mv_grid for each 4×4 in the block.
    // Use BWH_TAB[0] (luma dimensions in 4×4 units) since our grid has 4×4
    // granularity, unlike the above/left context arrays which use BWH_TAB[1].
    let grid_w4 = BWH_TAB[0][bs as usize][0] as usize;
    let grid_h4 = BWH_TAB[0][bs as usize][1] as usize;
    for y in 0..grid_h4 {
        let grid_row = row + y;
        for x in 0..grid_w4 {
            let grid_col = col + x;
            let idx = grid_row * td.cols_4x4 + grid_col;
            if idx < td.cur_mv_grid.len() {
                if is_intra {
                    td.cur_mv_grid[idx] = MvRefPair {
                        mv: [[0, 0], [0, 0]],
                        ref_frame: [-1, -1],
                    };
                } else if comp {
                    td.cur_mv_grid[idx] = MvRefPair {
                        mv: mv[3],
                        ref_frame: *ref_frame,
                    };
                } else {
                    td.cur_mv_grid[idx] = MvRefPair {
                        mv: [mv[3][0], [0, 0]],
                        ref_frame: [ref_frame[0], -1],
                    };
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Coefficient decoding
// ---------------------------------------------------------------------------

/// Band counts for each transform size (band_counts in vp9block.c).
/// Indexed by [tx_size][band_index].
const BAND_COUNTS: [[i16; 6]; 4] = [
    [1, 2, 3, 4, 3, 3],     // 4×4  — sums to 16
    [1, 2, 3, 4, 11, 43],   // 8×8  — sums to 64
    [1, 2, 3, 4, 11, 235],  // 16×16 — sums to 256
    [1, 2, 3, 4, 11, 1003], // 32×32 — sums to 1024
];

/// Decode transform coefficients for all planes of a block.
///
/// Returns `(coefs_y, coefs_u, coefs_v)`.
/// Each is a flat array of `n_txb * n_coef` dequantised coefficients,
/// where `n_coef = (1 << tx_size)^2`.
///
/// Corresponds to `decode_coeffs` in vp9block.c (8-bit path).
// Direct translation of FFmpeg's decode_coeffs which takes the same number of
// parameters; the high argument count reflects the VP9 spec's per-plane state.
#[allow(clippy::too_many_arguments)]
fn decode_coeffs(
    td: &mut TileDecodeContext<'_>,
    bs: BlockSize,
    tx_size: TxSize,
    uv_tx_size: TxSize,
    seg_id: u8,
    row: usize,
    col: usize,
    y_modes: &[IntraMode; 4],
    is_inter: bool,
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
    let hdr = td.header;
    let intra_idx = usize::from(is_inter); // 0 = intra, 1 = inter

    // Quantizer multipliers [dc, ac].
    let dc_y = get_dc_quant(hdr.base_q_idx, hdr.y_dc_delta_q, hdr.bit_depth);
    let ac_y = get_ac_quant(hdr.base_q_idx, 0, hdr.bit_depth);
    let dc_uv = get_dc_quant(hdr.base_q_idx, hdr.uv_dc_delta_q, hdr.bit_depth);
    let ac_uv = get_ac_quant(hdr.base_q_idx, hdr.uv_ac_delta_q, hdr.bit_depth);
    // Note: lossless does NOT override quantizers — the lookup table returns
    // the correct values (e.g., dc=4 for base_q_idx=0). The lossless flag
    // only selects the WHT inverse transform.

    // Segment quantizer (if segmentation Q feature is enabled).
    // When absolute_vals is true, q_val is the absolute Q index.
    // When false, q_val is a delta added to base_q_idx.
    let seg_feat = &hdr.segmentation.feat[seg_id as usize];
    let (qmul_y_dc, qmul_y_ac, qmul_uv_dc, qmul_uv_ac) =
        if hdr.segmentation.enabled && seg_feat.q_enabled {
            let q_base = if hdr.segmentation.absolute_vals {
                (seg_feat.q_val).clamp(0, 255) as u8
            } else {
                (hdr.base_q_idx as i16 + seg_feat.q_val).clamp(0, 255) as u8
            };
            let dc_y2 = get_dc_quant(q_base, hdr.y_dc_delta_q, hdr.bit_depth);
            let ac_y2 = get_ac_quant(q_base, 0, hdr.bit_depth);
            let dc_uv2 = get_dc_quant(q_base, hdr.uv_dc_delta_q, hdr.bit_depth);
            let ac_uv2 = get_ac_quant(q_base, hdr.uv_ac_delta_q, hdr.bit_depth);
            (dc_y2, ac_y2, dc_uv2, ac_uv2)
        } else {
            (dc_y, ac_y, dc_uv, ac_uv)
        };

    // Block dimensions in 4×4 chroma units.
    // FFmpeg: w4 = bwh_tab[1][bs][0]<<1, h4 = bwh_tab[1][bs][1]<<1
    let w4_y = BWH_TAB[1][bs as usize][0] as usize * 2;
    let h4_y = BWH_TAB[1][bs as usize][1] as usize * 2;
    let end_x_y = w4_y.min(td.cols_4x4.saturating_sub(col));
    let end_y_y = h4_y.min(td.rows_4x4.saturating_sub(row));

    let ss_h_shift = usize::from(hdr.subsampling_x);
    let ss_v_shift = usize::from(hdr.subsampling_y);
    let w4_uv = w4_y >> ss_h_shift;
    let h4_uv = h4_y >> ss_v_shift;
    let end_x_uv = end_x_y >> ss_h_shift;
    let end_y_uv = end_y_y >> ss_v_shift;

    // Step size for each tx dimension (in 4×4 "double" units from FFmpeg).
    let tx_step_y = 1usize << (tx_size as usize); // 1, 2, 4, 8
    let tx_step_uv = 1usize << (uv_tx_size as usize);
    let n_coef_y = tx_step_y * tx_step_y * 16; // = (tx_dim)^2
    let n_coef_uv = tx_step_uv * tx_step_uv * 16;

    // NNZ (non-zero) context arrays for luma.
    // FFmpeg uses above_y_nnz_ctx[col*2] where col is in 8x8 units (= 4x4 index).
    // Our col is already in 4x4 units, so no multiplication needed.
    // left_y_nnz_ctx uses (row & 15) since left context spans one 64px SB row.
    let a_y_base = col;
    let l_y_base = row & 15;

    // --- Merge NNZ context for larger transforms (MERGE_CTX) ---
    // For tx_step_y == 2 we OR-reduce pairs of NNZ entries into one.
    // We work on local copies and write back at the end.
    let a_y_len = end_x_y.min(td.above.coef[0].len().saturating_sub(a_y_base));
    let mut a_y: Vec<u8> = td.above.coef[0][a_y_base..a_y_base + a_y_len].to_vec();
    let l_y_avail = 16usize.saturating_sub(l_y_base);
    let mut l_y: Vec<u8> = td.left.coef[0][l_y_base..l_y_base + end_y_y.min(l_y_avail)].to_vec();

    merge_ctx(&mut a_y, end_x_y, tx_step_y);
    merge_ctx(&mut l_y, end_y_y, tx_step_y);

    let band_counts_y = &BAND_COUNTS[tx_size as usize];

    let mut coefs_y =
        Vec::with_capacity(((end_y_y / tx_step_y) * (end_x_y / tx_step_y)) * n_coef_y);

    // Luma coefficient decode loop.
    let tx_idx_y = if hdr.lossless { 4 } else { tx_size as usize };
    let is_tx32 = tx_size == TxSize::Tx32x32 && !hdr.lossless;

    for y_off in (0..end_y_y).step_by(tx_step_y) {
        for x_off in (0..end_x_y).step_by(tx_step_y) {
            let mode_idx = if matches!(bs, BlockSize::Bs8x4 | BlockSize::Bs4x8 | BlockSize::Bs4x4) {
                // sub-8×8: use per-4×4 mode.  FFmpeg: mode_index = n (tx4x4 only).
                let nt = y_off / tx_step_y * (end_x_y / tx_step_y) + x_off / tx_step_y;
                nt.min(3)
            } else {
                0
            };
            let txtp = if is_inter {
                0 // DCT_DCT for inter blocks
            } else {
                let y_mode_val = y_modes[mode_idx] as u8;
                if y_mode_val < 14 {
                    INTRA_TXFM_TYPE[y_mode_val as usize] as usize
                } else {
                    0
                }
            };
            // select scan and nb tables
            let (scan, nb) = select_scan_y(tx_idx_y, txtp);
            let nnz_a = *a_y.get(x_off).unwrap_or(&0);
            let nnz_l = *l_y.get(y_off).unwrap_or(&0);
            let nnz = (nnz_a + nnz_l) as usize;
            let p = &td.header.coef[tx_size as usize][0][intra_idx];
            let c = &mut td.counts.coef[tx_size as usize][0][intra_idx];
            let e = &mut td.counts.eob[tx_size as usize][0][intra_idx];
            let mut txb = vec![0i32; n_coef_y];
            let ret = decode_coeffs_block(
                &mut td.bd,
                &mut txb,
                n_coef_y,
                is_tx32,
                p,
                nnz,
                scan,
                nb,
                band_counts_y,
                qmul_y_dc,
                qmul_y_ac,
                c,
                e,
            )?;
            // Update NNZ context.
            let nnz_out = u8::from(ret > 0);
            splat_val(&mut a_y, x_off, nnz_out, tx_step_y, end_x_y == w4_y);
            splat_val(&mut l_y, y_off, nnz_out, tx_step_y, end_y_y == h4_y);
            coefs_y.append(&mut txb);
        }
    }

    // Write back NNZ context for luma.
    let a_y_end = (a_y_base + a_y_len).min(td.above.coef[0].len());
    td.above.coef[0][a_y_base..a_y_end].copy_from_slice(&a_y[..a_y_end - a_y_base]);
    let l_y_end = l_y_base + l_y.len();
    td.left.coef[0][l_y_base..l_y_end.min(16)].copy_from_slice(&l_y[..l_y_end.min(16) - l_y_base]);

    // --- UV planes ---
    let band_counts_uv = &BAND_COUNTS[uv_tx_size as usize];
    let uv_tx_idx = if hdr.lossless { 4 } else { uv_tx_size as usize };
    let is_tx32_uv = uv_tx_size == TxSize::Tx32x32 && !hdr.lossless;

    let mut coefs_u = Vec::new();
    let mut coefs_v = Vec::new();

    for pl in 0..2usize {
        // FFmpeg: above_uv_nnz_ctx[pl][col << !ss_h], left_uv_nnz_ctx[pl][(row & 7) << !ss_v]
        // Our col/row are in luma 4x4 units; chroma 4x4 = luma 4x4 >> ss.
        let a_uv_base = col >> ss_h_shift;
        let l_uv_base = (row & 15) >> ss_v_shift;
        let a_uv_len = end_x_uv.min(td.above.coef[pl + 1].len().saturating_sub(a_uv_base));
        let mut a_uv: Vec<u8> = td.above.coef[pl + 1][a_uv_base..a_uv_base + a_uv_len].to_vec();
        let l_uv_len = end_y_uv.min(16 - l_uv_base.min(16));
        let mut l_uv: Vec<u8> = td.left.coef[pl + 1][l_uv_base..l_uv_base + l_uv_len].to_vec();

        merge_ctx(&mut a_uv, end_x_uv, tx_step_uv);
        merge_ctx(&mut l_uv, end_y_uv, tx_step_uv);

        // Both chroma planes use the same quantizer multipliers.
        let (dc_qm, ac_qm) = (qmul_uv_dc, qmul_uv_ac);

        let p_uv = &td.header.coef[uv_tx_size as usize][1][intra_idx];
        let (scan_uv, nb_uv) = select_scan_y(uv_tx_idx, 0 /* DCT_DCT */);

        let mut plane_coefs = Vec::with_capacity(
            ((end_y_uv / tx_step_uv.max(1)) * (end_x_uv / tx_step_uv.max(1))) * n_coef_uv,
        );

        for y_off in (0..end_y_uv).step_by(tx_step_uv.max(1)) {
            for x_off in (0..end_x_uv).step_by(tx_step_uv.max(1)) {
                let nnz_a = *a_uv.get(x_off).unwrap_or(&0);
                let nnz_l = *l_uv.get(y_off).unwrap_or(&0);
                let nnz = (nnz_a + nnz_l) as usize;
                let c = &mut td.counts.coef[uv_tx_size as usize][1][intra_idx];
                let e = &mut td.counts.eob[uv_tx_size as usize][1][intra_idx];
                let mut txb = vec![0i32; n_coef_uv];
                let ret = decode_coeffs_block(
                    &mut td.bd,
                    &mut txb,
                    n_coef_uv,
                    is_tx32_uv,
                    p_uv,
                    nnz,
                    scan_uv,
                    nb_uv,
                    band_counts_uv,
                    dc_qm,
                    ac_qm,
                    c,
                    e,
                )?;
                let nnz_out = u8::from(ret > 0);
                splat_val(
                    &mut a_uv,
                    x_off,
                    nnz_out,
                    tx_step_uv.max(1),
                    end_x_uv == w4_uv,
                );
                splat_val(
                    &mut l_uv,
                    y_off,
                    nnz_out,
                    tx_step_uv.max(1),
                    end_y_uv == h4_uv,
                );
                plane_coefs.append(&mut txb);
            }
        }

        // Write back UV NNZ context.
        let a_uv_end = (a_uv_base + a_uv_len).min(td.above.coef[pl + 1].len());
        td.above.coef[pl + 1][a_uv_base..a_uv_end].copy_from_slice(&a_uv[..a_uv_end - a_uv_base]);
        let l_uv_end = l_uv_base + l_uv.len();
        td.left.coef[pl + 1][l_uv_base..l_uv_end.min(16)]
            .copy_from_slice(&l_uv[..l_uv_end.min(16) - l_uv_base]);

        if pl == 0 {
            coefs_u = plane_coefs;
        } else {
            coefs_v = plane_coefs;
        }
    }

    Ok((coefs_y, coefs_u, coefs_v))
}

/// Select scan order and neighbour table for a given tx index and transform type.
///
/// `tx_idx` = 0..3 (normal) or 4 (lossless/4×4).
/// `txtp` = 0=DCT_DCT, 1=DCT_ADST, 2=ADST_DCT, 3=ADST_ADST.
///
/// Mirrors `ff_vp9_scans[tx][txtp]` and `ff_vp9_scans_nb[tx][txtp]` from vp9data.c.
fn select_scan_y(tx_idx: usize, txtp: usize) -> (&'static [i16], &'static [[i16; 2]]) {
    // For 32×32 and lossless (tx_idx==4) only default DCT scan is used.
    // Scan selection maps txtp (TxType enum value) to the scan pattern.
    // FFmpeg: scans[tx][0]=default, scans[tx][1]=col, scans[tx][2]=row, scans[tx][3]=default.
    // FFmpeg enum: DCT_ADST=1→col_scan, ADST_DCT=2→row_scan.
    // Wedeo enum: DctAdst=1, AdstDct=2 — same values, so 1→col, 2→row.
    match tx_idx {
        4 => {
            // Lossless 4×4 (WHT): always default scan, matching FFmpeg's
            // ff_vp9_scans[4] = { default, default, default, default }.
            (&crate::data::DEFAULT_SCAN_4X4, &DEFAULT_SCAN_4X4_NB)
        }
        0 => {
            // 4×4
            match txtp {
                1 => (&crate::data::COL_SCAN_4X4, &COL_SCAN_4X4_NB), // DCT_ADST → col_scan
                2 => (&crate::data::ROW_SCAN_4X4, &ROW_SCAN_4X4_NB), // ADST_DCT → row_scan
                _ => (&crate::data::DEFAULT_SCAN_4X4, &DEFAULT_SCAN_4X4_NB),
            }
        }
        1 => {
            // 8×8
            match txtp {
                1 => (&crate::data::COL_SCAN_8X8, &COL_SCAN_8X8_NB),
                2 => (&crate::data::ROW_SCAN_8X8, &ROW_SCAN_8X8_NB),
                _ => (&crate::data::DEFAULT_SCAN_8X8, &DEFAULT_SCAN_8X8_NB),
            }
        }
        2 => {
            // 16×16
            match txtp {
                1 => (&crate::data::COL_SCAN_16X16, &COL_SCAN_16X16_NB),
                2 => (&crate::data::ROW_SCAN_16X16, &ROW_SCAN_16X16_NB),
                _ => (&crate::data::DEFAULT_SCAN_16X16, &DEFAULT_SCAN_16X16_NB),
            }
        }
        _ => {
            // 32×32 — only default scan
            (&crate::data::DEFAULT_SCAN_32X32, &DEFAULT_SCAN_32X32_NB)
        }
    }
}

/// Expand a 3-probability coef entry `[p0, p1, p2]` to the 11-probability
/// form used during decoding.
///
/// FFmpeg stores `prob.coef[tx][bt][is_inter][band][ctx]` as 11 bytes:
///   [0] = EOB prob, [1] = zero prob, [2] = one-vs-more prob,
///   [3..11] = model_pareto8[p[2]] for the higher magnitude tokens.
/// The compressed form stores only [0..3] and reconstructs [3..11] on demand.
#[inline]
fn expand_coef_prob(p3: &[u8; 3]) -> [u8; 11] {
    let mut out = [0u8; 11];
    out[0] = p3[0];
    out[1] = p3[1];
    out[2] = p3[2];
    let pareto = &PARETO8[p3[2] as usize];
    out[3..11].copy_from_slice(pareto);
    out
}

/// Decode one transform block's coefficients.
///
/// Corresponds to `decode_coeffs_b_generic` in vp9block.c (8-bit path,
/// `is8bitsperpixel = 1`).
///
/// Returns the number of non-zero coefficients decoded (same as FFmpeg's
/// return value used to populate `eob[]`).
// Direct translation from C; the high argument count matches FFmpeg exactly.
#[allow(clippy::too_many_arguments)]
fn decode_coeffs_block(
    bd: &mut BoolDecoder<'_>,
    coef: &mut [i32],
    n_coeffs: usize,
    is_tx32x32: bool,
    p: &[[[u8; 3]; 6]; 6], // [band][ctx][3]
    nnz: usize,
    scan: &[i16],
    nb: &[[i16; 2]],
    band_counts: &[i16; 6],
    qmul_dc: i16,
    qmul_ac: i16,
    cnt: &mut [[[u32; 3]; 6]; 6],
    eob_cnt: &mut [[[u32; 2]; 6]; 6],
) -> Result<usize> {
    // The prob/count arrays are [band][ctx][3].
    // Clamp initial nnz to the valid range [0..4].
    let mut nnz = nnz.min(5);
    let mut band: usize = 0;
    let mut band_left = band_counts[0] as usize;
    let mut cache = [0u8; 1024];
    let mut i: usize = 0;

    // Outer loop: read EOB flag, then enter inner "skip_eob" loop for zero runs.
    // Matches FFmpeg's goto skip_eob structure in decode_coeffs_b_generic.
    'outer: loop {
        if i >= n_coeffs {
            break;
        }
        // Expand 3 base probs to 11 using PARETO8 for the current band/ctx.
        let tp = expand_coef_prob(&p[band][nnz]);
        // EOB flag — only read for the first coeff and after each non-zero coeff.
        let more_coefs = bd.get_prob(tp[0]);
        eob_cnt[band][nnz][usize::from(more_coefs)] += 1;
        if !more_coefs {
            break; // end of block
        }

        // skip_eob: inner loop for zero-coefficient runs.
        // After a zero coefficient, we loop back here (NOT to the EOB read).
        loop {
            let tp = expand_coef_prob(&p[band][nnz]);
            if !bd.get_prob(tp[1]) {
                // zero coefficient
                cnt[band][nnz][0] += 1;
                band_left -= 1;
                if band_left == 0 {
                    band += 1;
                    if band < 6 {
                        band_left = band_counts[band] as usize;
                    }
                }
                let rc = scan[i] as usize;
                if rc < cache.len() {
                    cache[rc] = 0;
                }
                let nb0 = nb.get(i).map(|n| n[0] as usize).unwrap_or(0);
                let nb1 = nb.get(i).map(|n| n[1] as usize).unwrap_or(0);
                let c0 = cache.get(nb0).copied().unwrap_or(0);
                let c1 = cache.get(nb1).copied().unwrap_or(0);
                nnz = ((1u32 + c0 as u32 + c1 as u32) >> 1) as usize;
                nnz = nnz.min(5);
                i += 1;
                if i >= n_coeffs {
                    break 'outer; // invalid: blocks should end with EOB
                }
                continue; // goto skip_eob (inner loop)
            }
            break; // non-zero coefficient — fall through to decode it
        }

        let tp = expand_coef_prob(&p[band][nnz]);
        let rc = scan[i] as usize;
        // Decode coefficient magnitude.
        let val: i32 = if !bd.get_prob(tp[2]) {
            // one
            cnt[band][nnz][1] += 1;
            if rc < cache.len() {
                cache[rc] = 1;
            }
            1
        } else {
            cnt[band][nnz][2] += 1;
            if !bd.get_prob(tp[3]) {
                // 2, 3, or 4
                if !bd.get_prob(tp[4]) {
                    if rc < cache.len() {
                        cache[rc] = 2;
                    }
                    2
                } else {
                    let extra = bd.get_prob(tp[5]);
                    if rc < cache.len() {
                        cache[rc] = 3;
                    }
                    3 + i32::from(extra)
                }
            } else if !bd.get_prob(tp[6]) {
                // cat 1 / cat 2
                if rc < cache.len() {
                    cache[rc] = 4;
                }
                if !bd.get_prob(tp[7]) {
                    // cat1: 1 extra bit
                    5 + i32::from(bd.get_prob(159))
                } else {
                    // cat2: 2 extra bits
                    7 + (i32::from(bd.get_prob(165)) << 1) + i32::from(bd.get_prob(145))
                }
            } else {
                // cat 3-6
                if rc < cache.len() {
                    cache[rc] = 5;
                }
                if !bd.get_prob(tp[8]) {
                    if !bd.get_prob(tp[9]) {
                        // cat3: 3 extra bits
                        11 + (i32::from(bd.get_prob(173)) << 2)
                            + (i32::from(bd.get_prob(148)) << 1)
                            + i32::from(bd.get_prob(140))
                    } else {
                        // cat4: 4 extra bits
                        19 + (i32::from(bd.get_prob(176)) << 3)
                            + (i32::from(bd.get_prob(155)) << 2)
                            + (i32::from(bd.get_prob(140)) << 1)
                            + i32::from(bd.get_prob(135))
                    }
                } else if !bd.get_prob(tp[10]) {
                    // cat5: 5 extra bits
                    35 + (i32::from(bd.get_prob(180)) << 4)
                        + (i32::from(bd.get_prob(157)) << 3)
                        + (i32::from(bd.get_prob(141)) << 2)
                        + (i32::from(bd.get_prob(134)) << 1)
                        + i32::from(bd.get_prob(130))
                } else {
                    // cat6: variable extra bits (8-bit path: 14 extra bits)
                    67 + (i32::from(bd.get_prob(254)) << 13)
                        + (i32::from(bd.get_prob(254)) << 12)
                        + (i32::from(bd.get_prob(254)) << 11)
                        + (i32::from(bd.get_prob(252)) << 10)
                        + (i32::from(bd.get_prob(249)) << 9)
                        + (i32::from(bd.get_prob(243)) << 8)
                        + (i32::from(bd.get_prob(230)) << 7)
                        + (i32::from(bd.get_prob(196)) << 6)
                        + (i32::from(bd.get_prob(177)) << 5)
                        + (i32::from(bd.get_prob(153)) << 4)
                        + (i32::from(bd.get_prob(140)) << 3)
                        + (i32::from(bd.get_prob(133)) << 2)
                        + (i32::from(bd.get_prob(130)) << 1)
                        + i32::from(bd.get_prob(129))
                }
            }
        };

        // Advance band.
        band_left -= 1;
        if band_left == 0 {
            band += 1;
            if band < 6 {
                band_left = band_counts[band] as usize;
            }
        }

        // Sign bit.
        let sign = bd.get();
        let signed_val = if sign { -val } else { val };

        // Apply dequantization.
        let qmul = if i == 0 { qmul_dc } else { qmul_ac };
        let dequant = if is_tx32x32 {
            // 32×32: divide by 2 to avoid overflow (FFmpeg: (val * qmul) / 2).
            signed_val.wrapping_mul(i32::from(qmul)) / 2
        } else {
            signed_val.wrapping_mul(i32::from(qmul))
        };

        if rc < coef.len() {
            // FFmpeg stores coefficients as int16_t for 8bpp, truncating to 16 bits.
            coef[rc] = dequant as i16 as i32;
        }

        // Update NNZ neighbour context.
        let nb0 = nb.get(i).map(|n| n[0] as usize).unwrap_or(0);
        let nb1 = nb.get(i).map(|n| n[1] as usize).unwrap_or(0);
        let c0 = cache.get(nb0).copied().unwrap_or(0);
        let c1 = cache.get(nb1).copied().unwrap_or(0);
        nnz = ((1u32 + c0 as u32 + c1 as u32) >> 1) as usize;
        nnz = nnz.min(5);

        i += 1;
    }

    Ok(i)
}

// ---------------------------------------------------------------------------
// Context helpers
// ---------------------------------------------------------------------------

/// Merge NNZ context entries for transforms larger than 4×4.
///
/// For `step == 2` each pair of adjacent NNZ entries is OR-reduced.
/// Mirrors the `MERGE` macro in vp9block.c.
fn merge_ctx(ctx: &mut [u8], end: usize, step: usize) {
    if step < 2 {
        return;
    }
    let end = end.min(ctx.len());
    let mut n = 0;
    while n < end {
        let v = if n + step <= ctx.len() {
            ctx[n..n + step].iter().any(|&x| x != 0) as u8
        } else {
            ctx.get(n).copied().unwrap_or(0)
        };
        if n < ctx.len() {
            ctx[n] = v;
        }
        n += step;
    }
}

/// Splat an NNZ value across `step` consecutive context entries.
///
/// Mirrors the `SPLAT` macro in vp9block.c.
fn splat_val(ctx: &mut [u8], off: usize, val: u8, step: usize, full: bool) {
    if step <= 1 {
        if off < ctx.len() {
            ctx[off] = val;
        }
        return;
    }
    let end = if full {
        (off + step).min(ctx.len())
    } else {
        // Partial block: fill starting from off+1 up to min(off+step-1, end).
        let fill_end = (off + step).min(ctx.len());
        for x in ctx.iter_mut().take(fill_end).skip(off + 1) {
            *x = val;
        }
        if off < ctx.len() {
            ctx[off] = val;
        }
        return;
    };
    ctx[off..end].fill(val);
}

/// Zero-out the NNZ context for a skipped block.
///
/// Mirrors the `SPLAT_ZERO_YUV` macros in vp9block.c.
// VP9 spec requires separate above/left, per-plane, per-subsampling parameters —
// refactoring would obscure the 1-to-1 correspondence with FFmpeg's C macros.
#[allow(clippy::too_many_arguments)]
fn zero_nnz_ctx(
    above: &mut AboveContext,
    left: &mut LeftContext,
    col: usize,
    row: usize,
    w4: usize,
    h4: usize,
    ss_h: bool,
    ss_v: bool,
) {
    // Luma NNZ: FFmpeg uses SPLAT_ZERO_CTX(dir_y_nnz_ctx[off*2], n*2) where
    // off/n are in 8x8 units.  Our col/w4 are in 4x4 units, and the NNZ
    // context has 2 entries per 4x4 column → span = w4 * 2 entries.
    let base_a = col;
    let end_a = (base_a + w4 * 2).min(above.coef[0].len());
    above.coef[0][base_a..end_a].fill(0);

    let base_l = row & 15;
    let end_l = (base_l + h4 * 2).min(16);
    left.coef[0][base_l..end_l].fill(0);

    // Chroma NNZ — FFmpeg's SPLAT_ZERO_YUV:
    //   ss=1: index = col/row7 (8×8 units), span = w4/h4 (8×8 units)
    //   ss=0: index = col*2/row7*2 (4×4 units), span = w4*2/h4*2 (4×4 units)
    // Wedeo's col/row are in 4×4 luma units; w4/h4 are in 8×8 units.
    let ss_h_shift = usize::from(ss_h);
    let ss_v_shift = usize::from(ss_v);
    let uv_col = col >> ss_h_shift;
    let uv_row = (row & 15) >> ss_v_shift;
    let uv_w = if ss_h { w4 } else { w4 * 2 };
    let uv_h = if ss_v { h4 } else { h4 * 2 };

    for pl in 1..3usize {
        let end_ua = (uv_col + uv_w).min(above.coef[pl].len());
        if uv_col < above.coef[pl].len() {
            above.coef[pl][uv_col..end_ua].fill(0);
        }
        let end_ul = (uv_row + uv_h).min(16);
        left.coef[pl][uv_row..end_ul].fill(0);
    }
}

/// Update the above/left context arrays after decoding a block.
///
/// Mirrors `SET_CTXS` in vp9block.c.
#[allow(clippy::too_many_arguments)]
fn update_ctx(
    td: &mut TileDecodeContext<'_>,
    bs: BlockSize,
    tx_size: TxSize,
    skip: bool,
    _segment_id: u8,
    mode_ctx_val: u8,
    col: usize,
    row: usize,
    is_keyframe: bool,
    is_inter: bool,
    comp: bool,
    _ref_frame: &[i8; 2],
    _filter: u8,
    filter_id: u8,
    vref: u8,
) {
    // Above context arrays are cols_4x4 entries (4×4 units) → use BWH_TAB[0] for width.
    let w4 = BWH_TAB[0][bs as usize][0] as usize;
    // Left context arrays are 8 entries (8×8 units) → use BWH_TAB[1] for height.
    let h4 = BWH_TAB[1][bs as usize][1] as usize;
    // row is in 4×4 units; left context needs 8×8-unit row within the SB.
    let row7 = (row >> 1) & 7;

    let above_part_val = ABOVE_PARTITION_CTX[bs as usize];
    let left_part_val = LEFT_PARTITION_CTX[bs as usize];

    // Above context updates (indexed by col, spans w4 entries).
    for i in 0..w4 {
        let c = col + i;
        if c < td.above.skip.len() {
            td.above.skip[c] = u8::from(skip);
        }
        if c < td.above.tx_size.len() {
            td.above.tx_size[c] = tx_size as u8;
        }
        if c < td.above.partition.len() {
            td.above.partition[c] = above_part_val;
        }
    }

    // For non-keyframes: SET_CTXS inter context
    if !is_keyframe {
        let last_y = mode_ctx_val;
        // intra/comp/ref/filter use 1 entry per 4×4 column (col in 4×4 units)
        for i in 0..w4 {
            let c = col + i;
            if c < td.above.intra.len() {
                td.above.intra[c] = !is_inter as u8;
            }
            if c < td.above.comp.len() {
                td.above.comp[c] = comp as u8;
            }
            if is_inter && c < td.above.ref_frame.len() {
                td.above.ref_frame[c] = vref;
            }
            if is_inter && td.header.filter_mode == 4 && c < td.above.filter.len() {
                td.above.filter[c] = filter_id;
            }
        }
        // Mode context above: 1 entry per 8×8 column for non-keyframe
        // FFmpeg: memset(&s->above_mode_ctx[col], mode[3], w4) where col is 8×8
        let bw = BWH_TAB[1][bs as usize][0] as usize;
        let col_8x8 = col / 2;
        for i in 0..bw {
            let c = col_8x8 + i;
            if c < td.above.y_mode.len() {
                td.above.y_mode[c] = last_y;
            }
        }
    }

    // Left context updates (indexed by row7, spans h4 entries).
    for i in 0..h4 {
        let r = row7 + i;
        if r < 8 {
            td.left.skip[r] = u8::from(skip);
            td.left.tx_size[r] = tx_size as u8;
            td.left.partition[r] = left_part_val;
        }
    }

    if !is_keyframe {
        let last_y = mode_ctx_val;
        for i in 0..h4 {
            let r = row7 + i;
            if r < 8 {
                td.left.intra[r] = !is_inter as u8;
                td.left.comp[r] = comp as u8;
                if is_inter {
                    td.left.ref_frame[r] = vref;
                    if td.header.filter_mode == 4 {
                        td.left.filter[r] = filter_id;
                    }
                }
            }
        }
        // Mode context left (1 entry per 8×8 row for non-keyframe)
        for i in 0..h4 {
            let r = row7 + i;
            if r < 8 {
                td.left.y_mode[r] = last_y;
            }
        }
    }
}
