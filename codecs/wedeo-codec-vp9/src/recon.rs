// VP9 intra reconstruction: prediction + inverse transform + add.
//
// Translated from FFmpeg's libavcodec/vp9recon.c (`intra_recon`,
// `check_intra_mode`).  Only the 8-bit / Profile 0 (4:2:0) path is
// implemented; higher bit-depths and other chroma formats are not supported
// yet.
//
// LGPL-2.1-or-later — same licence as FFmpeg.

use std::sync::Arc;

use wedeo_core::error::Result;

use crate::block::BlockInfo;
use crate::data::{BWH_TAB, INTRA_TXFM_TYPE};
use crate::header::FrameHeader;
use crate::idct::{itxfm_add, itxfm_add_lossless};
use crate::intra_pred::intra_pred;
use crate::mc;
use crate::refs::RefFrame;
use crate::types::{IntraMode, TxSize, TxType};

// ---------------------------------------------------------------------------
// Frame buffer
// ---------------------------------------------------------------------------

/// YUV 4:2:0 frame buffer holding one decoded frame.
pub struct FrameBuffer {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub y_stride: usize,
    pub uv_stride: usize,
    pub width: u32,
    pub height: u32,
}

impl FrameBuffer {
    /// Allocate a zeroed frame buffer for a frame of the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let w = width as usize;
        let h = height as usize;
        // Round strides up to a multiple of 64 for alignment headroom.
        let y_stride = (w + 63) & !63;
        let uv_w = w.div_ceil(2);
        let uv_stride = (uv_w + 63) & !63;
        // Pad rows to the next superblock boundary (64 luma, 32 chroma)
        // so edge-SB intra prediction can write past the visible frame.
        let y_rows = (h + 63) & !63;
        let uv_rows = (h.div_ceil(2) + 31) & !31;
        Self {
            y: vec![0u8; y_stride * y_rows],
            u: vec![0u8; uv_stride * uv_rows],
            v: vec![0u8; uv_stride * uv_rows],
            y_stride,
            uv_stride,
            width,
            height,
        }
    }
}

// ---------------------------------------------------------------------------
// Mode conversion table  (check_intra_mode in vp9recon.c)
// ---------------------------------------------------------------------------

/// `mode_conv[mode][have_left][have_top]` — maps a bitstream intra mode to
/// the appropriate fallback when some neighbors are missing.
///
/// Indices: 0 = no-left/no-top → left/top DC variants; same layout as FFmpeg.
static MODE_CONV: [[[IntraMode; 2]; 2]; 10] = [
    // VERT_PRED
    [
        [IntraMode::Dc127Pred, IntraMode::VertPred],
        [IntraMode::Dc127Pred, IntraMode::VertPred],
    ],
    // HOR_PRED
    [
        [IntraMode::Dc129Pred, IntraMode::Dc129Pred],
        [IntraMode::HorPred, IntraMode::HorPred],
    ],
    // DC_PRED
    [
        [IntraMode::Dc128Pred, IntraMode::TopDcPred],
        [IntraMode::LeftDcPred, IntraMode::DcPred],
    ],
    // DIAG_DOWN_LEFT_PRED
    [
        [IntraMode::Dc127Pred, IntraMode::DiagDownLeftPred],
        [IntraMode::Dc127Pred, IntraMode::DiagDownLeftPred],
    ],
    // DIAG_DOWN_RIGHT_PRED
    [
        [IntraMode::DiagDownRightPred, IntraMode::DiagDownRightPred],
        [IntraMode::DiagDownRightPred, IntraMode::DiagDownRightPred],
    ],
    // VERT_RIGHT_PRED
    [
        [IntraMode::VertRightPred, IntraMode::VertRightPred],
        [IntraMode::VertRightPred, IntraMode::VertRightPred],
    ],
    // HOR_DOWN_PRED
    [
        [IntraMode::HorDownPred, IntraMode::HorDownPred],
        [IntraMode::HorDownPred, IntraMode::HorDownPred],
    ],
    // VERT_LEFT_PRED
    [
        [IntraMode::Dc127Pred, IntraMode::VertLeftPred],
        [IntraMode::Dc127Pred, IntraMode::VertLeftPred],
    ],
    // HOR_UP_PRED
    [
        [IntraMode::Dc129Pred, IntraMode::Dc129Pred],
        [IntraMode::HorUpPred, IntraMode::HorUpPred],
    ],
    // TM_VP8_PRED
    [
        [IntraMode::Dc129Pred, IntraMode::VertPred],
        [IntraMode::HorPred, IntraMode::TmVp8Pred],
    ],
];

/// Which neighbor pixels each mode requires.
struct EdgeNeeds {
    needs_left: bool,
    needs_top: bool,
    needs_topleft: bool,
    needs_topright: bool,
    /// HOR_UP_PRED: collect left pixels top-to-bottom instead of bottom-to-top.
    invert_left: bool,
}

static EDGE_NEEDS: [EdgeNeeds; 15] = [
    // VERT_PRED
    EdgeNeeds {
        needs_left: false,
        needs_top: true,
        needs_topleft: false,
        needs_topright: false,
        invert_left: false,
    },
    // HOR_PRED
    EdgeNeeds {
        needs_left: true,
        needs_top: false,
        needs_topleft: false,
        needs_topright: false,
        invert_left: false,
    },
    // DC_PRED
    EdgeNeeds {
        needs_left: true,
        needs_top: true,
        needs_topleft: false,
        needs_topright: false,
        invert_left: false,
    },
    // DIAG_DOWN_LEFT_PRED
    EdgeNeeds {
        needs_left: false,
        needs_top: true,
        needs_topleft: false,
        needs_topright: true,
        invert_left: false,
    },
    // DIAG_DOWN_RIGHT_PRED
    EdgeNeeds {
        needs_left: true,
        needs_top: true,
        needs_topleft: true,
        needs_topright: false,
        invert_left: false,
    },
    // VERT_RIGHT_PRED
    EdgeNeeds {
        needs_left: true,
        needs_top: true,
        needs_topleft: true,
        needs_topright: false,
        invert_left: false,
    },
    // HOR_DOWN_PRED
    EdgeNeeds {
        needs_left: true,
        needs_top: true,
        needs_topleft: true,
        needs_topright: false,
        invert_left: false,
    },
    // VERT_LEFT_PRED
    EdgeNeeds {
        needs_left: false,
        needs_top: true,
        needs_topleft: false,
        needs_topright: true,
        invert_left: false,
    },
    // HOR_UP_PRED
    EdgeNeeds {
        needs_left: true,
        needs_top: false,
        needs_topleft: false,
        needs_topright: false,
        invert_left: true,
    },
    // TM_VP8_PRED
    EdgeNeeds {
        needs_left: true,
        needs_top: true,
        needs_topleft: true,
        needs_topright: false,
        invert_left: false,
    },
    // LEFT_DC_PRED (10)
    EdgeNeeds {
        needs_left: true,
        needs_top: false,
        needs_topleft: false,
        needs_topright: false,
        invert_left: false,
    },
    // TOP_DC_PRED (11)
    EdgeNeeds {
        needs_left: false,
        needs_top: true,
        needs_topleft: false,
        needs_topright: false,
        invert_left: false,
    },
    // DC_128_PRED (12)
    EdgeNeeds {
        needs_left: false,
        needs_top: false,
        needs_topleft: false,
        needs_topright: false,
        invert_left: false,
    },
    // DC_127_PRED (13)
    EdgeNeeds {
        needs_left: false,
        needs_top: false,
        needs_topleft: false,
        needs_topright: false,
        invert_left: false,
    },
    // DC_129_PRED (14)
    EdgeNeeds {
        needs_left: false,
        needs_top: false,
        needs_topleft: false,
        needs_topright: false,
        invert_left: false,
    },
];

// ---------------------------------------------------------------------------
// Neighbor pixel preparation  (check_intra_mode)
// ---------------------------------------------------------------------------

/// Prepare the `above` and `left` neighbor pixel arrays for one transform-block
/// and return the (possibly adjusted) intra mode.
///
/// * `plane_buf`   — the plane pixel buffer (Y, U, or V).
/// * `stride`      — row stride for `plane_buf`.
/// * `px`, `py`    — pixel-coordinate top-left corner of this transform block.
/// * `frame_w`, `frame_h` — frame dimensions for the plane (pixels).
/// * `tile_col_start_px` — first pixel column in the current tile.
/// * `tx_px`       — transform block size in pixels (4/8/16/32).
/// * `above_buf`   — caller-provided scratch buffer (≥ `tx_px + 1` entries;
///   entry `[0]` is the top-left, `[1..]` are the above pixels).
/// * `left_buf`    — caller-provided scratch buffer (≥ `tx_px` entries).
/// * `mode`        — bitstream intra mode (0..9).
///
/// Returns the resolved mode.
#[allow(clippy::too_many_arguments)]
fn prepare_intra_edges(
    plane_buf: &[u8],
    stride: usize,
    px: usize,
    py: usize,
    frame_w: usize,
    frame_h: usize,
    tile_col_start_px: usize,
    tx_px: usize,
    above_buf: &mut [u8], // length >= tx_px + 1; above_buf[1..] = above pixels; above_buf[0] = top-left
    left_buf: &mut [u8],  // length >= tx_px
    mode: IntraMode,
    have_right: bool, // FFmpeg: x < w - 1 (within-block check)
) -> IntraMode {
    let mode_idx = mode as usize;
    debug_assert!(
        mode_idx < 10,
        "prepare_intra_edges called with non-bitstream mode"
    );

    let have_top = py > 0;
    let have_left = px > tile_col_start_px;

    // Apply mode_conv: map bitstream mode to the mode suitable for the
    // available neighbors.
    let resolved = MODE_CONV[mode_idx][have_left as usize][have_top as usize];
    let r = resolved as usize;
    let needs = &EDGE_NEEDS[r];

    // --- above pixels ---
    if needs.needs_top {
        // FFmpeg: n_px_have = (((s->cols - col) << !ss_h) - x) * 4
        // Total pixels available from px to right edge of frame.
        let n_px_have = frame_w.saturating_sub(px);
        let n_px_need = tx_px;
        // Include top-right pixels if needed (only for TX_4X4).
        let n_need_tr = if tx_px == 4 && needs.needs_topright && have_right {
            4
        } else {
            0
        };

        // FFmpeg fast-path: direct pointer when all required pixels are available.
        //   have_top && (!needs_topleft || (have_left && top == topleft))
        //   && (tx != TX_4X4 || !needs_topright || have_right)
        //   && n_px_need + n_px_need_tr <= n_px_have
        if have_top
            && (!needs.needs_topleft || have_left)
            && (tx_px != 4 || !needs.needs_topright || have_right)
            && n_px_need + n_need_tr <= n_px_have
        {
            // Fast path: can point directly; but since we own above_buf, copy.
            let src_row = py - 1;
            for i in 0..tx_px {
                above_buf[i + 1] = plane_buf[src_row * stride + px + i];
            }
            if n_need_tr > 0 {
                for i in 0..n_need_tr {
                    above_buf[tx_px + 1 + i] = plane_buf[src_row * stride + px + tx_px + i];
                }
            }
        } else {
            // Slow path: copy what we have, pad the rest.
            if have_top {
                let src_row = py - 1;
                if n_px_have >= n_px_need {
                    for i in 0..n_px_need {
                        above_buf[i + 1] = plane_buf[src_row * stride + px + i];
                    }
                } else {
                    for i in 0..n_px_have {
                        above_buf[i + 1] = plane_buf[src_row * stride + px + i];
                    }
                    // Pad with last available pixel.
                    let pad = if n_px_have > 0 {
                        plane_buf[src_row * stride + px + n_px_have - 1]
                    } else {
                        127
                    };
                    for i in n_px_have..n_px_need {
                        above_buf[i + 1] = pad;
                    }
                }
            } else {
                // No top neighbor — fill with 127.
                for i in 0..n_px_need {
                    above_buf[i + 1] = 127;
                }
            }
            if needs.needs_topright {
                if have_top && have_right && n_px_need + n_need_tr <= n_px_have {
                    // FFmpeg: memcpy(&(*a)[4], &top[4], 4)
                    let src_row = py - 1;
                    for i in 0..4 {
                        above_buf[tx_px + 1 + i] = plane_buf[src_row * stride + px + tx_px + i];
                    }
                } else {
                    // Replicate last above pixel.
                    let last = above_buf[tx_px];
                    for i in 0..4 {
                        above_buf[tx_px + 1 + i] = last;
                    }
                }
            }
        }

        // Top-left pixel.
        if needs.needs_topleft {
            if have_left && have_top {
                above_buf[0] = plane_buf[(py - 1) * stride + px - 1];
            } else if have_top {
                above_buf[0] = 128u8.wrapping_add(1); // 129
            } else {
                above_buf[0] = 128u8.wrapping_sub(1); // 127
            }
        }
    }

    // --- left pixels (bottom-to-top ordering expected by intra_pred) ---
    if needs.needs_left {
        let n_have_rows = frame_h.saturating_sub(py).min(tx_px);

        if have_left {
            if needs.invert_left {
                // HOR_UP_PRED: collect top-to-bottom (left[0] = topmost row).
                if n_have_rows >= tx_px {
                    for i in 0..tx_px {
                        left_buf[i] = plane_buf[(py + i) * stride + px - 1];
                    }
                } else {
                    for i in 0..n_have_rows {
                        left_buf[i] = plane_buf[(py + i) * stride + px - 1];
                    }
                    let pad = if n_have_rows > 0 {
                        plane_buf[(py + n_have_rows - 1) * stride + px - 1]
                    } else {
                        129
                    };
                    left_buf[n_have_rows..tx_px].fill(pad);
                }
            } else {
                // Normal: bottom-to-top (left[0] = bottom-most row).
                if n_have_rows >= tx_px {
                    for i in 0..tx_px {
                        left_buf[tx_px - 1 - i] = plane_buf[(py + i) * stride + px - 1];
                    }
                } else {
                    for i in 0..n_have_rows {
                        left_buf[tx_px - 1 - i] = plane_buf[(py + i) * stride + px - 1];
                    }
                    let pad = if n_have_rows > 0 {
                        plane_buf[(py + n_have_rows - 1) * stride + px - 1]
                    } else {
                        129
                    };
                    left_buf[0..(tx_px - n_have_rows)].fill(pad);
                }
            }
        } else {
            // No left neighbor — fill with 129.
            left_buf[..tx_px].fill(129);
        }
    }

    resolved
}

// ---------------------------------------------------------------------------
// Public reconstruction entry point
// ---------------------------------------------------------------------------

/// Reconstruct one decoded block: run intra prediction then add the inverse
/// transform of the dequantized coefficients.
///
/// # Arguments
/// * `fb`     — frame buffer (modified in-place).
/// * `block`  — decoded block info from the entropy-coding phase.
/// * `header` — frame header (for lossless flag, dimensions, tile info).
/// * `tile_col_start` — first 4×4 column of the current tile.
pub fn reconstruct_intra_block(
    fb: &mut FrameBuffer,
    block: &BlockInfo,
    header: &FrameHeader,
    tile_col_start: usize,
) -> Result<()> {
    let bit_depth = header.bit_depth;
    let bs = block.bs as usize;

    // Block dimensions in 4×4 "double" units — matches FFmpeg's
    // `w4 = bwh_tab[1][bs][0] << 1` in intra_recon (vp9recon.c:225).
    // This is the FULL coefficient grid, not the sub-block size.
    let w4 = (BWH_TAB[1][bs][0] as usize) * 2;
    let h4 = (BWH_TAB[1][bs][1] as usize) * 2;

    // Frame-boundary clamping — FFmpeg:
    //   end_x = FFMIN(2 * (s->cols - col), w4)
    //   end_y = FFMIN(2 * (s->rows - row), h4)
    // where cols/rows are in 8×8 units and col/row are also 8×8.
    // Our col/row are in 4×4 units and cols_4x4 is also 4×4, so we
    // convert: 2 * (cols_8x8 - col_8x8) = cols_4x4*2 - col_4x4*2
    //        = 2 * (header.cols_4x4 - block.col)  (treating block.col as 4×4).
    let cols_8x8 = (header.width as usize).div_ceil(8);
    let rows_8x8 = (header.height as usize).div_ceil(8);
    let col_8x8 = block.col / 2;
    let row_8x8 = block.row / 2;
    let end_x = w4.min(2 * (cols_8x8.saturating_sub(col_8x8)));
    let end_y = h4.min(2 * (rows_8x8.saturating_sub(row_8x8)));

    // Luma transform step in 4×4 units.
    let tx_step = 1usize << (block.tx_size as usize);
    // TX size in pixels.
    let tx_px = tx_step * 4;

    // Luma reconstruction.
    {
        // Scratch buffers for neighbor pixels.  Above needs tx_px + 4 entries
        // (one for top-left at index 0, tx_px for above, and up to 4 for
        // top-right).  Left needs tx_px entries.
        let mut above_scratch = vec![128u8; tx_px + 5];
        let mut left_scratch = vec![129u8; tx_px];

        let mut coef_idx = 0usize;
        let coef_per_tx = tx_px * tx_px;

        for ty in (0..end_y).step_by(tx_step) {
            for tx in (0..end_x).step_by(tx_step) {
                // Pixel coordinates of this transform block.
                let px = (block.col + tx) * 4;
                let py = (block.row + ty) * 4;

                // For blocks larger than 8×8 and using 4×4 transforms, each
                // 4×4 within the 16-element y_mode array is indexed by
                // `ty * 2 + tx` (matching `b->mode[b->bs > BS_8x8 && b->tx == TX_4X4 ? y * 2 + x : 0]`
                // from vp9recon.c where x and y are in tx steps).
                let mode_idx = if bs > 9 && block.tx_size == TxSize::Tx4x4 {
                    // bs > 9 means sub-8×8 blocks (Bs8x4=10, Bs4x8=11, Bs4x4=12).
                    // FFmpeg: b->mode[b->bs > BS_8x8 && b->tx == TX_4X4 ? y*2+x : 0]
                    let sub_y = ty / tx_step;
                    let sub_x = tx / tx_step;
                    (sub_y * 2 + sub_x).min(3)
                } else {
                    0
                };
                let mode = block.y_mode[mode_idx];

                // Prepare neighbors and resolve the mode.
                // Use 8×8-grid-based dimensions (not pixel-exact) to match
                // FFmpeg's n_px_have = (((s->cols - col) << !ss_h) - x) * 4.
                let frame_w = cols_8x8 * 8;
                let frame_h = rows_8x8 * 8;
                let tile_col_px = tile_col_start * 4;

                // FFmpeg: have_right = x < w - 1 (against full block width, not end_x).
                let have_right = tx + tx_step < w4;

                let resolved = prepare_intra_edges(
                    &fb.y,
                    fb.y_stride,
                    px,
                    py,
                    frame_w,
                    frame_h,
                    tile_col_px,
                    tx_px,
                    &mut above_scratch,
                    &mut left_scratch,
                    mode,
                    have_right,
                );

                // intra_pred writes to a sub-slice of the frame buffer.
                let dst_offset = py * fb.y_stride + px;
                // above = &above_scratch[1..] (above_scratch[0] is top-left).
                let above_slice = &above_scratch[1..];
                let top_left = above_scratch[0];

                intra_pred(
                    &mut fb.y[dst_offset..],
                    fb.y_stride,
                    above_slice,
                    &left_scratch,
                    top_left,
                    resolved,
                    tx_px,
                    bit_depth,
                );

                // Inverse transform + add.
                if !block.skip {
                    let coefs = &block.coefs_y[coef_idx..coef_idx + coef_per_tx];
                    // Determine EOB (last non-zero coefficient + 1).
                    let eob = coefs.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
                    if eob > 0 {
                        if header.lossless {
                            itxfm_add_lossless(
                                &mut fb.y[dst_offset..],
                                fb.y_stride,
                                coefs,
                                bit_depth,
                            );
                        } else {
                            let tx_type_raw = INTRA_TXFM_TYPE[mode as usize];
                            let tx_type = TxType::try_from(tx_type_raw).unwrap_or(TxType::DctDct);
                            itxfm_add(
                                &mut fb.y[dst_offset..],
                                fb.y_stride,
                                coefs,
                                block.tx_size,
                                tx_type,
                                bit_depth,
                                eob,
                            );
                        }

                    }
                }

                coef_idx += coef_per_tx;
            }
        }
    }

    // Chroma (U and V) reconstruction.
    // For 4:2:0: chroma dimensions are half luma (rounded up).
    let ss_h = if header.subsampling_x { 1usize } else { 0 };
    let ss_v = if header.subsampling_y { 1usize } else { 0 };

    // Chroma end dimensions derived from luma end, matching FFmpeg:
    //   w4 >>= s->ss_h; end_x >>= s->ss_h; end_y >>= s->ss_v;
    let uv_end_x = end_x >> ss_h;
    let uv_end_y = end_y >> ss_v;
    let uv_w4 = w4 >> ss_h; // FFmpeg: w4 >>= s->ss_h
    let uv_tx_step = 1usize << (block.uv_tx_size as usize);
    let uv_tx_px = uv_tx_step * 4;

    // Grid-based chroma dimensions matching FFmpeg's n_px_have formula.
    let uv_frame_w = cols_8x8 * (8 >> ss_h);
    let uv_frame_h = rows_8x8 * (8 >> ss_v);
    let uv_tile_col_px = (tile_col_start * 4) >> ss_h;

    let mut uv_above_scratch = vec![128u8; uv_tx_px + 5];
    let mut uv_left_scratch = vec![129u8; uv_tx_px];
    let uv_stride = fb.uv_stride;
    let uv_mode = block.uv_mode;
    let coef_per_uv_tx = uv_tx_px * uv_tx_px;

    // Process U (Cb) plane.
    {
        let mut coef_idx = 0usize;
        for ty in (0..uv_end_y).step_by(uv_tx_step) {
            for tx in (0..uv_end_x).step_by(uv_tx_step) {
                let px = ((block.col >> ss_h) + tx) * 4;
                let py = ((block.row >> ss_v) + ty) * 4;

                let dst_end = py
                    .saturating_add(uv_tx_px)
                    .saturating_sub(1)
                    .saturating_mul(uv_stride)
                    .saturating_add(px)
                    .saturating_add(uv_tx_px);
                let uv_buf_ok = |buf: &[u8]| {
                    dst_end <= buf.len()
                        && px + uv_tx_px <= uv_frame_w
                        && py + uv_tx_px <= uv_frame_h
                };
                if !uv_buf_ok(&fb.u) || !uv_buf_ok(&fb.v) {
                    coef_idx += coef_per_uv_tx;
                    continue;
                }

                let uv_have_right = tx + uv_tx_step < uv_w4;

                let resolved = prepare_intra_edges(
                    &fb.u,
                    uv_stride,
                    px,
                    py,
                    uv_frame_w,
                    uv_frame_h,
                    uv_tile_col_px,
                    uv_tx_px,
                    &mut uv_above_scratch,
                    &mut uv_left_scratch,
                    uv_mode,
                    uv_have_right,
                );

                let dst_offset = py * uv_stride + px;
                let above_slice = &uv_above_scratch[1..];
                let top_left = uv_above_scratch[0];

                intra_pred(
                    &mut fb.u[dst_offset..],
                    uv_stride,
                    above_slice,
                    &uv_left_scratch,
                    top_left,
                    resolved,
                    uv_tx_px,
                    bit_depth,
                );

                if !block.skip {
                    let coefs = &block.coefs_u[coef_idx..coef_idx + coef_per_uv_tx];
                    let eob = coefs.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
                    if eob > 0 {
                        if header.lossless {
                            itxfm_add_lossless(
                                &mut fb.u[dst_offset..],
                                uv_stride,
                                coefs,
                                bit_depth,
                            );
                        } else {
                            itxfm_add(
                                &mut fb.u[dst_offset..],
                                uv_stride,
                                coefs,
                                block.uv_tx_size,
                                TxType::DctDct,
                                bit_depth,
                                eob,
                            );
                        }
                    }
                }

                coef_idx += coef_per_uv_tx;
            }
        }
    }

    // Process V (Cr) plane.
    {
        let mut coef_idx = 0usize;
        for ty in (0..uv_end_y).step_by(uv_tx_step) {
            for tx in (0..uv_end_x).step_by(uv_tx_step) {
                let px = ((block.col >> ss_h) + tx) * 4;
                let py = ((block.row >> ss_v) + ty) * 4;

                let dst_end = py
                    .saturating_add(uv_tx_px)
                    .saturating_sub(1)
                    .saturating_mul(uv_stride)
                    .saturating_add(px)
                    .saturating_add(uv_tx_px);
                let uv_buf_ok = |buf: &[u8]| {
                    dst_end <= buf.len()
                        && px + uv_tx_px <= uv_frame_w
                        && py + uv_tx_px <= uv_frame_h
                };
                if !uv_buf_ok(&fb.u) || !uv_buf_ok(&fb.v) {
                    coef_idx += coef_per_uv_tx;
                    continue;
                }

                let uv_have_right = tx + uv_tx_step < uv_w4;

                let resolved = prepare_intra_edges(
                    &fb.v,
                    uv_stride,
                    px,
                    py,
                    uv_frame_w,
                    uv_frame_h,
                    uv_tile_col_px,
                    uv_tx_px,
                    &mut uv_above_scratch,
                    &mut uv_left_scratch,
                    uv_mode,
                    uv_have_right,
                );

                let dst_offset = py * uv_stride + px;
                let above_slice = &uv_above_scratch[1..];
                let top_left = uv_above_scratch[0];

                intra_pred(
                    &mut fb.v[dst_offset..],
                    uv_stride,
                    above_slice,
                    &uv_left_scratch,
                    top_left,
                    resolved,
                    uv_tx_px,
                    bit_depth,
                );

                if !block.skip {
                    let coefs = &block.coefs_v[coef_idx..coef_idx + coef_per_uv_tx];
                    let eob = coefs.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
                    if eob > 0 {
                        if header.lossless {
                            itxfm_add_lossless(
                                &mut fb.v[dst_offset..],
                                uv_stride,
                                coefs,
                                bit_depth,
                            );
                        } else {
                            // UV always uses DCT_DCT (vp9recon.c line 280).
                            itxfm_add(
                                &mut fb.v[dst_offset..],
                                uv_stride,
                                coefs,
                                block.uv_tx_size,
                                TxType::DctDct,
                                bit_depth,
                                eob,
                            );
                        }
                    }
                }

                coef_idx += coef_per_uv_tx;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Inter block reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct one inter-predicted block: motion compensation then add residual.
///
/// Translated from `inter_recon` in vp9recon.c. Inter blocks always use
/// `TxType::DctDct` for the inverse transform (no ADST for inter).
pub fn reconstruct_inter_block(
    fb: &mut FrameBuffer,
    block: &BlockInfo,
    header: &FrameHeader,
    ref_slots: &[Option<Arc<RefFrame>>; 8],
) -> Result<()> {
    let bit_depth = header.bit_depth;

    // 1. Motion compensation: write prediction into fb.
    mc::inter_pred(
        fb,
        block,
        ref_slots,
        &header.ref_idx,
        header.subsampling_x,
        header.subsampling_y,
    );

    // 2. If skip, no residual to add.
    if block.skip {
        return Ok(());
    }

    let bs = block.bs as usize;

    // Luma residual.
    // FFmpeg vp9recon.c:611-614: dimensions from bwh_tab[1]<<1, clamped to frame edge.
    let cols_8x8 = (header.width as usize).div_ceil(8);
    let rows_8x8 = (header.height as usize).div_ceil(8);
    let w4 = (BWH_TAB[1][bs][0] as usize) * 2;
    let h4 = (BWH_TAB[1][bs][1] as usize) * 2;
    let end_x = w4.min(2 * cols_8x8.saturating_sub(block.col / 2));
    let end_y = h4.min(2 * rows_8x8.saturating_sub(block.row / 2));
    {
        let tx_step = 1usize << (block.tx_size as usize);
        let tx_px = tx_step * 4;
        let coef_per_tx = tx_px * tx_px;
        let mut coef_idx = 0usize;

        for ty in (0..end_y).step_by(tx_step) {
            for tx in (0..end_x).step_by(tx_step) {
                let px = (block.col + tx) * 4;
                let py = (block.row + ty) * 4;

                let coefs = &block.coefs_y[coef_idx..coef_idx + coef_per_tx];
                let eob = coefs.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
                if eob > 0 {
                    let dst_offset = py * fb.y_stride + px;
                    if header.lossless {
                        itxfm_add_lossless(&mut fb.y[dst_offset..], fb.y_stride, coefs, bit_depth);
                    } else {
                        itxfm_add(
                            &mut fb.y[dst_offset..],
                            fb.y_stride,
                            coefs,
                            block.tx_size,
                            TxType::DctDct, // Inter always uses DCT_DCT.
                            bit_depth,
                            eob,
                        );
                    }
                }

                coef_idx += coef_per_tx;
            }
        }
    }

    // Chroma residual (U and V).
    let ss_h = if header.subsampling_x { 1usize } else { 0 };
    let ss_v = if header.subsampling_y { 1usize } else { 0 };
    let uv_w4 = BWH_TAB[1][bs][0] as usize;
    let uv_h4 = BWH_TAB[1][bs][1] as usize;
    let uv_end_x = uv_w4.min(end_x >> ss_h);
    let uv_end_y = uv_h4.min(end_y >> ss_v);
    let uv_tx_step = 1usize << (block.uv_tx_size as usize);
    let uv_tx_px = uv_tx_step * 4;
    let coef_per_uv_tx = uv_tx_px * uv_tx_px;
    let uv_stride = fb.uv_stride;

    // U plane.
    {
        let mut coef_idx = 0usize;
        for ty in (0..uv_end_y).step_by(uv_tx_step) {
            for tx in (0..uv_end_x).step_by(uv_tx_step) {
                let px = ((block.col >> ss_h) + tx) * 4;
                let py = ((block.row >> ss_v) + ty) * 4;

                let coefs = &block.coefs_u[coef_idx..coef_idx + coef_per_uv_tx];
                let eob = coefs.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
                if eob > 0 {
                    let dst_offset = py * uv_stride + px;
                    if header.lossless {
                        itxfm_add_lossless(&mut fb.u[dst_offset..], uv_stride, coefs, bit_depth);
                    } else {
                        itxfm_add(
                            &mut fb.u[dst_offset..],
                            uv_stride,
                            coefs,
                            block.uv_tx_size,
                            TxType::DctDct,
                            bit_depth,
                            eob,
                        );
                    }
                }

                coef_idx += coef_per_uv_tx;
            }
        }
    }

    // V plane.
    {
        let mut coef_idx = 0usize;
        for ty in (0..uv_end_y).step_by(uv_tx_step) {
            for tx in (0..uv_end_x).step_by(uv_tx_step) {
                let px = ((block.col >> ss_h) + tx) * 4;
                let py = ((block.row >> ss_v) + ty) * 4;

                let coefs = &block.coefs_v[coef_idx..coef_idx + coef_per_uv_tx];
                let eob = coefs.iter().rposition(|&c| c != 0).map_or(0, |p| p + 1);
                if eob > 0 {
                    let dst_offset = py * uv_stride + px;
                    if header.lossless {
                        itxfm_add_lossless(&mut fb.v[dst_offset..], uv_stride, coefs, bit_depth);
                    } else {
                        itxfm_add(
                            &mut fb.v[dst_offset..],
                            uv_stride,
                            coefs,
                            block.uv_tx_size,
                            TxType::DctDct,
                            bit_depth,
                            eob,
                        );
                    }
                }

                coef_idx += coef_per_uv_tx;
            }
        }
    }

    Ok(())
}
