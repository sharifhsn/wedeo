// CAVLC (Context-Adaptive Variable-Length Coding) decode functions.
//
// Implements macroblock-level CAVLC parsing for H.264 Baseline/Main profile.
//
// Reference: ITU-T H.264 Section 9.2, FFmpeg libavcodec/h264_cavlc.c
// (`decode_residual`, `ff_h264_decode_mb_cavlc`)

use tracing::trace;
use wedeo_codec::bitstream::{BitRead, BitReadBE, get_se_golomb, get_ue_golomb};
use wedeo_core::error::{Error, Result};

use crate::cavlc_tables::{
    read_coeff_token, read_run_before, read_total_zeros, read_total_zeros_chroma_dc,
};
use crate::pps::Pps;
use crate::slice::SliceType;
use crate::tables::{
    B_MB_TYPE_INFO, B_SUB_MB_TYPE_INFO, GOLOMB_TO_INTER_CBP, GOLOMB_TO_INTRA4X4_CBP,
};

// ---------------------------------------------------------------------------
// Neighbor context for nC computation
// ---------------------------------------------------------------------------

/// Context for CAVLC neighbor-based nC computation.
///
/// nC is the predicted number of non-zero coefficients, computed from the
/// left and top neighboring blocks (H.264 spec section 9.2.1).
///
/// Layout:
/// - `top_nz`: Non-zero counts for the row above, indexed by 4x4 block column
///   across the full picture width. Length = mb_width * 4 for luma,
///   mb_width * 2 for each chroma component.
/// - `left_nz`: Non-zero counts for the column to the left of the current MB.
///   [0..3] = luma left column (top to bottom),
///   [4..5] = Cb left column,
///   [6..7] = Cr left column.
/// - `left_available`: Whether the left neighbor is available (false for left edge).
/// - `top_available`: Per-column top availability, indexed like top_nz.
#[derive(Debug, Clone)]
pub struct NeighborContext {
    /// Non-zero counts for the row above (luma: mb_width*4, Cb: mb_width*2, Cr: mb_width*2).
    pub top_nz_luma: Vec<u8>,
    pub top_nz_cb: Vec<u8>,
    pub top_nz_cr: Vec<u8>,
    /// Non-zero counts for the block to the left.
    /// [0..3] = luma (top to bottom within MB), [4..5] = Cb, [6..7] = Cr.
    pub left_nz: [u8; 8],
    /// Whether the left neighbor MB is available.
    pub left_available: bool,
    /// Whether the top neighbor MB is available.
    pub top_available: bool,
    /// Intra4x4 prediction modes for the bottom row of the row above.
    /// Indexed by 4x4 block column across the full picture width (mb_width * 4).
    /// -1 means unavailable (inter MB or I_16x16).
    pub top_intra4x4_mode: Vec<i8>,
    /// Intra4x4 prediction modes for the right column of the left neighbor MB.
    /// [0..3] = modes for blk_y 0..3 (top to bottom).
    /// -1 means unavailable.
    pub left_intra4x4_mode: [i8; 4],
}

impl NeighborContext {
    /// Create a new context for a picture of the given width in macroblocks.
    pub fn new(mb_width: u32) -> Self {
        Self {
            top_nz_luma: vec![0; mb_width as usize * 4],
            top_nz_cb: vec![0; mb_width as usize * 2],
            top_nz_cr: vec![0; mb_width as usize * 2],
            left_nz: [0; 8],
            left_available: false,
            top_available: false,
            top_intra4x4_mode: vec![-1; mb_width as usize * 4],
            left_intra4x4_mode: [-1; 4],
        }
    }

    /// Compute nC for a luma 4x4 block at position (blk_x, blk_y) within the MB,
    /// where the MB is at position (mb_x, mb_y).
    ///
    /// blk_x, blk_y are in 0..3 (4x4 block coordinates within the 16x16 MB).
    pub fn predict_luma_nz(&self, mb_x: u32, blk_x: u32, blk_y: u32) -> i32 {
        let abs_blk_x = mb_x * 4 + blk_x;

        // Left neighbor
        let left = if blk_x > 0 {
            // Within the same MB — use the non_zero_count we've already decoded.
            // Caller must update the neighbor context incrementally.
            // For now, return a sentinel that means "use current MB data".
            // Actually, we need to handle this at a higher level, so we'll
            // return the available left nz.
            None // Handled by caller (intra-MB left is in the output array)
        } else if self.left_available {
            Some(self.left_nz[blk_y as usize] as i32)
        } else {
            None
        };

        // Top neighbor
        let top = if blk_y > 0 {
            None // Handled by caller (intra-MB top is in the output array)
        } else if self.top_available {
            Some(self.top_nz_luma[abs_blk_x as usize] as i32)
        } else {
            None
        };

        match (left, top) {
            (Some(a), Some(b)) => (a + b + 1) >> 1,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => 0,
        }
    }

    /// Compute nC for a chroma 4x4 block.
    ///
    /// `plane`: 0 = Cb, 1 = Cr.
    /// `blk_x`, `blk_y`: 0..1 within the 2x2 chroma block grid.
    pub fn predict_chroma_nz(&self, mb_x: u32, plane: usize, blk_x: u32, blk_y: u32) -> i32 {
        let abs_blk_x = mb_x * 2 + blk_x;

        let left = if blk_x > 0 {
            None // intra-MB, handled by caller
        } else if self.left_available {
            let offset = if plane == 0 { 4 } else { 6 };
            Some(self.left_nz[offset + blk_y as usize] as i32)
        } else {
            None
        };

        let top = if blk_y > 0 {
            None // intra-MB, handled by caller
        } else if self.top_available {
            let top_nz = if plane == 0 {
                &self.top_nz_cb
            } else {
                &self.top_nz_cr
            };
            Some(top_nz[abs_blk_x as usize] as i32)
        } else {
            None
        };

        match (left, top) {
            (Some(a), Some(b)) => (a + b + 1) >> 1,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => 0,
        }
    }

    /// Update the context after decoding a macroblock.
    ///
    /// Stores the bottom row of luma non-zero counts into `top_nz_luma`,
    /// the right column into `left_nz`, etc.
    /// `intra4x4_modes` contains the 16 prediction modes in raster order
    /// (or all -1 if the MB is not I_4x4).
    pub fn update_after_mb(&mut self, mb_x: u32, nz: &[u8; 24], intra4x4_modes: &[i8; 16]) {
        // Luma: nz[0..15] in raster order (row-major 4x4)
        // Bottom row of luma (blk_y=3): indices 12, 13, 14, 15
        let luma_base = mb_x as usize * 4;
        self.top_nz_luma[luma_base] = nz[12];
        self.top_nz_luma[luma_base + 1] = nz[13];
        self.top_nz_luma[luma_base + 2] = nz[14];
        self.top_nz_luma[luma_base + 3] = nz[15];

        // Right column of luma (blk_x=3): indices 3, 7, 11, 15
        self.left_nz[0] = nz[3];
        self.left_nz[1] = nz[7];
        self.left_nz[2] = nz[11];
        self.left_nz[3] = nz[15];

        // Chroma Cb: nz[16..19] in raster order (2x2)
        let cb_base = mb_x as usize * 2;
        self.top_nz_cb[cb_base] = nz[18]; // bottom row: indices 18, 19
        self.top_nz_cb[cb_base + 1] = nz[19];
        self.left_nz[4] = nz[17]; // right column: 17, 19
        self.left_nz[5] = nz[19];

        // Chroma Cr: nz[20..23]
        let cr_base = mb_x as usize * 2;
        self.top_nz_cr[cr_base] = nz[22];
        self.top_nz_cr[cr_base + 1] = nz[23];
        self.left_nz[6] = nz[21];
        self.left_nz[7] = nz[23];

        // Intra4x4 prediction modes: store bottom row and right column.
        // Bottom row (blk_y=3): raster indices 12, 13, 14, 15
        let mode_base = mb_x as usize * 4;
        self.top_intra4x4_mode[mode_base] = intra4x4_modes[12];
        self.top_intra4x4_mode[mode_base + 1] = intra4x4_modes[13];
        self.top_intra4x4_mode[mode_base + 2] = intra4x4_modes[14];
        self.top_intra4x4_mode[mode_base + 3] = intra4x4_modes[15];
        // Right column (blk_x=3): raster indices 3, 7, 11, 15
        self.left_intra4x4_mode[0] = intra4x4_modes[3];
        self.left_intra4x4_mode[1] = intra4x4_modes[7];
        self.left_intra4x4_mode[2] = intra4x4_modes[11];
        self.left_intra4x4_mode[3] = intra4x4_modes[15];
    }

    /// Reset left context at the start of a new MB row.
    pub fn new_row(&mut self) {
        self.left_available = false;
        self.left_nz = [0; 8];
        self.left_intra4x4_mode = [-1; 4];
    }
}

/// Compute nC from already-decoded non_zero_count values within the current MB
/// and from the neighbor context for edge blocks.
///
/// `nz_cache` contains the non-zero counts already decoded for the current MB
/// (indexed in raster order 0..15 for luma, 16..23 for chroma).
/// Returns the nC value suitable for `read_coeff_token`.
///
/// `block_idx`: raster index of the 4x4 block within the MB:
///   0..15 = luma, 16..19 = Cb, 20..23 = Cr.
pub fn compute_nc(
    block_idx: usize,
    mb_x: u32,
    neighbor: &NeighborContext,
    nz_cache: &[u8; 24],
) -> i32 {
    if block_idx < 16 {
        // Luma
        let blk_x = (block_idx % 4) as u32;
        let blk_y = (block_idx / 4) as u32;

        let left = if blk_x > 0 {
            Some(nz_cache[block_idx - 1] as i32)
        } else if neighbor.left_available {
            Some(neighbor.left_nz[blk_y as usize] as i32)
        } else {
            None
        };

        let top = if blk_y > 0 {
            Some(nz_cache[block_idx - 4] as i32)
        } else if neighbor.top_available {
            let abs_blk_x = mb_x * 4 + blk_x;
            Some(neighbor.top_nz_luma[abs_blk_x as usize] as i32)
        } else {
            None
        };

        match (left, top) {
            (Some(a), Some(b)) => (a + b + 1) >> 1,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => 0,
        }
    } else if block_idx < 20 {
        // Chroma Cb
        let cb_idx = block_idx - 16; // 0..3
        let blk_x = (cb_idx % 2) as u32;
        let blk_y = (cb_idx / 2) as u32;

        let left = if blk_x > 0 {
            Some(nz_cache[block_idx - 1] as i32)
        } else if neighbor.left_available {
            Some(neighbor.left_nz[4 + blk_y as usize] as i32)
        } else {
            None
        };

        let top = if blk_y > 0 {
            Some(nz_cache[block_idx - 2] as i32)
        } else if neighbor.top_available {
            let abs_blk_x = mb_x * 2 + blk_x;
            Some(neighbor.top_nz_cb[abs_blk_x as usize] as i32)
        } else {
            None
        };

        match (left, top) {
            (Some(a), Some(b)) => (a + b + 1) >> 1,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => 0,
        }
    } else {
        // Chroma Cr
        let cr_idx = block_idx - 20; // 0..3
        let blk_x = (cr_idx % 2) as u32;
        let blk_y = (cr_idx / 2) as u32;

        let left = if blk_x > 0 {
            Some(nz_cache[block_idx - 1] as i32)
        } else if neighbor.left_available {
            Some(neighbor.left_nz[6 + blk_y as usize] as i32)
        } else {
            None
        };

        let top = if blk_y > 0 {
            Some(nz_cache[block_idx - 2] as i32)
        } else if neighbor.top_available {
            let abs_blk_x = mb_x * 2 + blk_x;
            Some(neighbor.top_nz_cr[abs_blk_x as usize] as i32)
        } else {
            None
        };

        match (left, top) {
            (Some(a), Some(b)) => (a + b + 1) >> 1,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// decode_residual — the core CAVLC coefficient decoding function
// ---------------------------------------------------------------------------

/// Read level_prefix: count consecutive zero bits before a '1' bit.
///
/// Matches FFmpeg's `get_level_prefix` in h264_cavlc.c.
fn get_level_prefix(br: &mut BitReadBE<'_>) -> Result<u32> {
    // Peek 32 bits and count leading zeros.
    let buf = br.peek_bits_32(32);
    if buf == 0 {
        return Err(Error::InvalidData);
    }
    let lz = buf.leading_zeros();
    // Skip the leading zeros plus the '1' bit.
    br.skip_bits(lz as usize + 1);
    Ok(lz)
}

/// Decode a block of residual coefficients using CAVLC.
///
/// Reads coeff_token, trailing ones sign bits, remaining levels,
/// total_zeros, and run_before values to produce the final coefficient array.
///
/// This follows the algorithm in H.264 spec section 9.2 and the implementation
/// in FFmpeg's `decode_residual` in h264_cavlc.c.
///
/// Parameters:
/// - `br`: bitstream reader
/// - `nc`: context number for coeff_token table selection (nC).
///   Use -1 for chroma DC (4:2:0), 0..16 for luma/chroma AC.
/// - `max_coeff`: maximum number of coefficients (16 for 4x4 luma,
///   15 for luma/chroma AC in I16x16/chroma, 4 for chroma DC 4:2:0).
///
/// Returns `(coefficients, non_zero_count)`.
/// `coefficients` is a 16-element array (only first `max_coeff` positions used).
/// Coefficients are placed in reverse scan order (highest-frequency first in
/// the levels array, then distributed using run_before into scan positions).
pub fn decode_residual(
    br: &mut BitReadBE<'_>,
    nc: i32,
    max_coeff: usize,
) -> Result<([i16; 16], u8)> {
    debug_assert!(max_coeff <= 16);

    let mut coeffs = [0i16; 16];

    // 1. Read coeff_token
    let (total_coeff, trailing_ones) = read_coeff_token(br, nc)?;

    if total_coeff == 0 {
        return Ok((coeffs, 0));
    }

    if total_coeff as usize > max_coeff {
        return Err(Error::InvalidData);
    }

    // 2. Read levels
    let mut levels = [0i32; 16];

    // 2a. Trailing ones: read sign bits (1 bit each).
    // The spec says trailing ones are decoded in reverse order (last non-zero first),
    // but their signs are read MSB-first from the bitstream (most recent first).
    // FFmpeg reads 3 sign bits with show_bits(3) and maps them:
    //   level[0] = 1 - ((bits & 4) >> 1)  =>  +1 or -1
    //   level[1] = 1 - (bits & 2)          =>  +1 or -1
    //   level[2] = 1 - ((bits & 1) << 1)   =>  +1 or -1
    // We follow a simpler approach: read one bit per trailing one.
    for level in levels.iter_mut().take(trailing_ones as usize) {
        let sign = br.get_bit(); // true = negative
        *level = if sign { -1 } else { 1 };
    }

    // 2b. Read remaining levels (total_coeff - trailing_ones levels).
    //
    // suffix_length controls the number of suffix bits read for each level.
    // It starts at 0 (or 1 if total_coeff > 10 and trailing_ones < 3) and
    // increases as larger levels are encountered.
    //
    // The first level after trailing ones has special handling when
    // suffix_length == 0: no suffix bits are read for prefix < 14, and
    // if trailing_ones < 3, the level_code is offset by +2 to avoid
    // encoding +/-1 (which would be redundant with trailing ones).
    //
    // Reference: H.264 spec section 9.2.2, FFmpeg h264_cavlc.c decode_residual.
    let remaining = total_coeff as usize - trailing_ones as usize;
    if remaining > 0 {
        let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 {
            1
        } else {
            0
        };

        for i in 0..remaining {
            let level_idx = trailing_ones as usize + i;
            let is_first = i == 0;

            // Read level_prefix (count of consecutive zeros before a '1').
            let prefix = get_level_prefix(br)?;

            if prefix > 25 + 3 {
                return Err(Error::InvalidData);
            }

            // Compute level_code from prefix and suffix.
            let level_code: i32;

            if is_first && suffix_length == 0 {
                // First coefficient with suffix_length == 0: special VLC structure.
                // prefix < 14: no suffix; prefix == 14: 4-bit suffix;
                // prefix >= 15: (prefix-3)-bit suffix with base offset 30.
                if prefix < 14 {
                    level_code = prefix as i32;
                } else if prefix == 14 {
                    let suffix = br.get_bits_32(4) as i32;
                    level_code = 14 + suffix;
                } else {
                    let mut lc = 30i32;
                    if prefix >= 16 {
                        lc += (1i32 << (prefix - 3)) - 4096;
                    }
                    let suffix = br.get_bits_32((prefix - 3) as usize) as i32;
                    level_code = lc + suffix;
                }
            } else {
                // Subsequent coefficients or first with suffix_length == 1.
                if prefix < 15 {
                    let suffix = if suffix_length > 0 {
                        br.get_bits_32(suffix_length as usize) as i32
                    } else {
                        0
                    };
                    level_code = (prefix as i32) * (1 << suffix_length) + suffix;
                } else {
                    let mut lc = 15i32 * (1 << suffix_length);
                    if prefix >= 16 {
                        lc += (1i32 << (prefix - 3)) - 4096;
                    }
                    let suffix = br.get_bits_32((prefix - 3) as usize) as i32;
                    level_code = lc + suffix;
                }
            }

            // Apply trailing_ones < 3 offset for first level.
            // When trailing_ones < 3, the first non-trailing level cannot be +/-1
            // (those are reserved for trailing ones), so offset by +2.
            let adjusted_code = if is_first && trailing_ones < 3 {
                level_code + 2
            } else {
                level_code
            };

            // Convert level_code to signed level value.
            // Even codes => negative, odd codes => positive.
            // level = ((code + 2) >> 1) with sign from bit 0.
            let mask = -(adjusted_code & 1); // 0 for odd (positive), -1 for even (negative)
            levels[level_idx] = (((adjusted_code + 2) >> 1) ^ mask) - mask;

            // Update suffix_length for the next level.
            // FFmpeg approach: use suffix_limit thresholds.
            let abs_level = levels[level_idx].unsigned_abs();
            if is_first {
                // After first level: suffix_length = 1 + (|level| > 3).
                // This is equivalent to FFmpeg's: suffix_length = 1 + (level_code + 3U > 6U)
                // applied to the adjusted level_code, or just checking |level| > 3.
                suffix_length = if abs_level > 3 { 2 } else { 1 };
            } else {
                // Subsequent levels: increment suffix_length when |level| exceeds threshold.
                // suffix_limit = [0, 3, 6, 12, 24, 48, MAX]
                const SUFFIX_LIMIT: [u32; 7] = [0, 3, 6, 12, 24, 48, u32::MAX];
                if suffix_length < 6 && abs_level > SUFFIX_LIMIT[suffix_length as usize] {
                    suffix_length += 1;
                }
            }
        }
    }

    // 3. Read total_zeros
    let total_zeros = if (total_coeff as usize) < max_coeff {
        if max_coeff <= 4 {
            // Chroma DC
            read_total_zeros_chroma_dc(br, total_coeff)?
        } else {
            read_total_zeros(br, total_coeff)?
        }
    } else {
        0u8
    };

    // 4. Read run_before and place coefficients
    // The levels array has levels in decoding order:
    //   levels[0..trailing_ones] = trailing ones (most recent first)
    //   levels[trailing_ones..total_coeff] = remaining levels (most recent first)
    // They are placed from the highest scan position downward.

    let mut zeros_left = total_zeros as i32;
    let mut coeff_idx = (total_coeff as i32 + total_zeros as i32 - 1) as usize;

    let total = total_coeff as usize;
    for (i, &level) in levels.iter().enumerate().take(total) {
        let is_last = i == total - 1;
        let run = if !is_last && zeros_left > 0 {
            read_run_before(br, zeros_left as u8)? as i32
        } else if !is_last {
            0i32
        } else {
            // Last coefficient: remaining zeros_left is implicit.
            zeros_left
        };

        if coeff_idx >= max_coeff {
            return Err(Error::InvalidData);
        }
        coeffs[coeff_idx] = level as i16;

        zeros_left -= run;
        if zeros_left < 0 {
            return Err(Error::InvalidData);
        }

        if !is_last {
            coeff_idx = coeff_idx
                .checked_sub((1 + run) as usize)
                .ok_or(Error::InvalidData)?;
        }
    }

    #[cfg(feature = "tracing-detail")]
    tracing::trace!(nc, total_coeff, trailing_ones, "CAVLC residual");

    Ok((coeffs, total_coeff))
}

// ---------------------------------------------------------------------------
// Macroblock-level types
// ---------------------------------------------------------------------------

/// Decoded macroblock data from CAVLC parsing.
#[derive(Debug, Clone)]
pub struct MacroblockCavlc {
    /// Raw mb_type value from the bitstream (before mapping to internal flags).
    pub mb_type: u32,
    /// True if this is an intra macroblock.
    pub is_intra: bool,
    /// True if this is an I_PCM macroblock (raw samples, no transform).
    pub is_pcm: bool,
    /// True if this is an I_4x4 macroblock.
    pub is_intra4x4: bool,
    /// True if this is an I_16x16 macroblock.
    pub is_intra16x16: bool,
    /// Intra prediction modes for 4x4 luma (16 entries), only valid for I_4x4.
    pub intra4x4_pred_mode: [u8; 16],
    /// Intra 16x16 prediction mode (0-3), only valid for I_16x16.
    pub intra16x16_mode: u8,
    /// Chroma intra prediction mode (0-3).
    pub chroma_pred_mode: u8,
    /// Coded block pattern: bits 0-3 for luma 8x8, bits 4-5 for chroma.
    pub cbp: u32,
    /// Luma coefficients for each 4x4 block (16 blocks, 16 coeffs each).
    /// Coefficients are in scan order (zigzag or field scan applied separately).
    pub luma_coeffs: [[i16; 16]; 16],
    /// DC coefficients for Intra16x16 (16 entries, before inverse Hadamard).
    pub luma_dc: [i16; 16],
    /// Chroma DC coefficients per chroma plane (2 planes, 4 DCs each for 4:2:0).
    pub chroma_dc: [[i16; 4]; 2],
    /// Chroma AC coefficients per chroma plane (2 planes, 4 blocks, 15 ACs each).
    pub chroma_ac: [[[i16; 16]; 4]; 2],
    /// Non-zero count per 4x4 block (used as context for neighbors).
    /// Layout: [0..15] = luma (raster), [16..19] = Cb, [20..23] = Cr.
    pub non_zero_count: [u8; 24],
    /// QP delta from slice QP.
    pub mb_qp_delta: i32,
    /// Sub-macroblock types for P_8x8 / B_8x8 (4 entries).
    pub sub_mb_type: [u8; 4],
    /// Reference indices for list 0 (4 entries for 4 partitions).
    pub ref_idx_l0: [i8; 4],
    /// Reference indices for list 1 (4 entries for 4 partitions).
    pub ref_idx_l1: [i8; 4],
    /// Motion vector differences for list 0.
    /// 16 entries: 4 partitions x up to 4 sub-partitions, each [dx, dy].
    pub mvd_l0: [[i16; 2]; 16],
    /// Motion vector differences for list 1.
    pub mvd_l1: [[i16; 2]; 16],
    /// Number of partitions for this macroblock type.
    pub partition_count: u8,
    /// True for B_Direct_16x16 (mb_type 0 in B-slice).
    pub is_direct: bool,
    /// Per-partition L0/L1 usage flags for B-slices.
    /// [partition][0=l0, 1=l1]. Only valid for B-slice inter MBs.
    pub b_list_flags: [[bool; 2]; 2],
    /// Partition size for B-slices: 0=16x16, 1=16x8, 2=8x16, 3=8x8.
    pub b_part_size: u8,
}

impl Default for MacroblockCavlc {
    fn default() -> Self {
        Self {
            mb_type: 0,
            is_intra: false,
            is_pcm: false,
            is_intra4x4: false,
            is_intra16x16: false,
            intra4x4_pred_mode: [0; 16],
            intra16x16_mode: 0,
            chroma_pred_mode: 0,
            cbp: 0,
            luma_coeffs: [[0; 16]; 16],
            luma_dc: [0; 16],
            chroma_dc: [[0; 4]; 2],
            chroma_ac: [[[0; 16]; 4]; 2],
            non_zero_count: [0; 24],
            mb_qp_delta: 0,
            sub_mb_type: [0; 4],
            ref_idx_l0: [-1; 4],
            ref_idx_l1: [-1; 4],
            mvd_l0: [[0; 2]; 16],
            mvd_l1: [[0; 2]; 16],
            partition_count: 0,
            is_direct: false,
            b_list_flags: [[false; 2]; 2],
            b_part_size: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Zigzag scan tables
// ---------------------------------------------------------------------------

/// Standard 4x4 zigzag scan: maps scan position (0..15) to raster position
/// (row * 4 + col) within the 4x4 block.
///
/// From FFmpeg `ff_zigzag_scan` in mathtables.c.
const ZIGZAG_SCAN_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Apply zigzag descan: convert coefficients from scan order to raster order.
fn zigzag_descan_4x4(scan_order: &[i16; 16], max_coeff: usize) -> [i16; 16] {
    let mut raster = [0i16; 16];
    for scan_pos in 0..max_coeff {
        raster[ZIGZAG_SCAN_4X4[scan_pos]] = scan_order[scan_pos];
    }
    raster
}

// ---------------------------------------------------------------------------
// Block scan order
// ---------------------------------------------------------------------------

/// Maps H.264 block scan index (i8x8 * 4 + i4x4, 0..15) to raster-order index
/// (blk_y * 4 + blk_x, 0..15).
///
/// H.264 scans 4x4 blocks within each 8x8 group:
///   i8x8=0 (top-left 8x8):     blocks at (0,0),(1,0),(0,1),(1,1) -> raster 0,1,4,5
///   i8x8=1 (top-right 8x8):    blocks at (2,0),(3,0),(2,1),(3,1) -> raster 2,3,6,7
///   i8x8=2 (bottom-left 8x8):  blocks at (0,2),(1,2),(0,3),(1,3) -> raster 8,9,12,13
///   i8x8=3 (bottom-right 8x8): blocks at (2,2),(3,2),(2,3),(3,3) -> raster 10,11,14,15
const SCAN_TO_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

// ---------------------------------------------------------------------------
// I-slice mb_type info table (from FFmpeg ff_h264_i_mb_type_info)
// ---------------------------------------------------------------------------

/// I-slice mb_type info: (prediction_mode, cbp).
///
/// Index 0 = I_4x4. Indices 1..24 = I_16x16 variants. Index 25 = I_PCM.
/// For I_16x16:
///   pred_mode = (mb_type - 1) % 4
///   cbp_chroma = ((mb_type - 1) / 4) % 3   (0=none, 1=DC only, 2=DC+AC)
///   cbp_luma = if (mb_type - 1) / 12 != 0 { 15 } else { 0 }
///
/// cbp encoding: bits 0-3 = luma (0 or 15), bits 4-5 = chroma (0, 1, or 2 mapped to 16, 32).
const I_MB_TYPE_INFO: [(i8, u8); 26] = [
    (-1, 0), // 0: I_4x4
    (0, 0),  // 1: I_16x16_0_0_0
    (1, 0),  // 2: I_16x16_1_0_0
    (2, 0),  // 3: I_16x16_2_0_0
    (3, 0),  // 4: I_16x16_3_0_0
    (0, 16), // 5: I_16x16_0_1_0
    (1, 16), // 6: I_16x16_1_1_0
    (2, 16), // 7: I_16x16_2_1_0
    (3, 16), // 8: I_16x16_3_1_0
    (0, 32), // 9: I_16x16_0_2_0
    (1, 32), // 10: I_16x16_1_2_0
    (2, 32), // 11: I_16x16_2_2_0
    (3, 32), // 12: I_16x16_3_2_0
    (0, 15), // 13: I_16x16_0_0_1 (luma CBP=15, chroma CBP=0)
    (1, 15), // 14: I_16x16_1_0_1
    (2, 15), // 15: I_16x16_2_0_1
    (3, 15), // 16: I_16x16_3_0_1
    (0, 31), // 17: I_16x16_0_1_1
    (1, 31), // 18: I_16x16_1_1_1
    (2, 31), // 19: I_16x16_2_1_1
    (3, 31), // 20: I_16x16_3_1_1
    (0, 47), // 21: I_16x16_0_2_1
    (1, 47), // 22: I_16x16_1_2_1
    (2, 47), // 23: I_16x16_2_2_1
    (3, 47), // 24: I_16x16_3_2_1
    (-1, 0), // 25: I_PCM
];

// ---------------------------------------------------------------------------
// P-slice mb_type info (from FFmpeg ff_h264_p_mb_type_info)
// ---------------------------------------------------------------------------

/// P-slice mb_type info: (partition_count, num_ref_lists).
/// mb_type 0 = P_L0_16x16 (1 partition)
/// mb_type 1 = P_L0_L0_16x8 (2 partitions)
/// mb_type 2 = P_L0_L0_8x16 (2 partitions)
/// mb_type 3 = P_8x8 (4 sub-partitions)
/// mb_type 4 = P_8x8ref0 (4 sub-partitions, ref_idx forced to 0)
const P_MB_PARTITION_COUNT: [u8; 5] = [1, 2, 2, 4, 4];

/// P sub-macroblock type partition counts.
/// sub_mb_type 0 = 8x8 (1), 1 = 8x4 (2), 2 = 4x8 (2), 3 = 4x4 (4)
const P_SUB_MB_PARTITION_COUNT: [u8; 4] = [1, 2, 2, 4];

// ---------------------------------------------------------------------------
// decode_mb_cavlc — macroblock-level CAVLC syntax parsing
// ---------------------------------------------------------------------------

/// Decode one macroblock's syntax elements using CAVLC.
///
/// This handles:
/// - mb_type parsing (I-slice vs P-slice mb_type tables)
/// - Intra prediction mode parsing (intra4x4 or intra16x16)
/// - Inter prediction parsing (ref_idx, mvd for P macroblocks)
/// - CBP parsing (coded block pattern)
/// - mb_qp_delta
/// - Residual coefficient decoding via decode_residual
///
/// Reference: FFmpeg `ff_h264_decode_mb_cavlc` in h264_cavlc.c.
#[cfg_attr(feature = "tracing-detail", tracing::instrument(skip_all, fields(mb_x, mb_y = _mb_y, slice_type = ?slice_type)))]
#[allow(clippy::too_many_arguments)]
pub fn decode_mb_cavlc(
    br: &mut BitReadBE<'_>,
    slice_type: SliceType,
    _pps: &Pps,
    neighbor: &NeighborContext,
    mb_x: u32,
    _mb_y: u32,
    _mb_width: u32,
    num_ref_idx_l0_active: u32,
    _num_ref_idx_l1_active: u32,
) -> Result<MacroblockCavlc> {
    let mut mb = MacroblockCavlc::default();

    // 1. Parse mb_type
    let raw_mb_type = get_ue_golomb(br)?;

    let is_intra;
    match slice_type {
        SliceType::I | SliceType::SI => {
            let mut mt = raw_mb_type;
            if slice_type == SliceType::SI && mt > 0 {
                mt -= 1;
            }
            if mt > 25 {
                return Err(Error::InvalidData);
            }
            is_intra = true;
            decode_intra_mb_type(&mut mb, mt)?;
        }
        SliceType::P | SliceType::SP => {
            if raw_mb_type < 5 {
                // Inter P macroblock
                is_intra = false;
                mb.mb_type = raw_mb_type;
                mb.partition_count = P_MB_PARTITION_COUNT[raw_mb_type as usize];
            } else {
                // Intra macroblock in P-slice (offset by 5)
                let mt = raw_mb_type - 5;
                if mt > 25 {
                    return Err(Error::InvalidData);
                }
                is_intra = true;
                decode_intra_mb_type(&mut mb, mt)?;
            }
        }
        SliceType::B => {
            if raw_mb_type < 23 {
                is_intra = false;
                mb.mb_type = raw_mb_type;
                let info = &B_MB_TYPE_INFO[raw_mb_type as usize];
                mb.partition_count = info.0;
                mb.b_part_size = info.1;
                mb.b_list_flags = info.2;
                mb.is_direct = raw_mb_type == 0;
            } else {
                let mt = raw_mb_type - 23;
                if mt > 25 {
                    return Err(Error::InvalidData);
                }
                is_intra = true;
                decode_intra_mb_type(&mut mb, mt)?;
            }
        }
    }
    mb.is_intra = is_intra;

    trace!(
        raw_mb_type,
        is_intra,
        is_pcm = mb.is_pcm,
        is_intra4x4 = mb.is_intra4x4,
        is_intra16x16 = mb.is_intra16x16,
        "MB type parsed"
    );

    // 2. Handle I_PCM
    if mb.is_pcm {
        // I_PCM: all coefficients are raw, non_zero_count = 16 for all blocks.
        mb.non_zero_count = [16; 24];
        // The caller must read raw PCM samples from the bitstream (byte-aligned).
        // We don't handle the actual PCM data here since it requires knowing
        // bit depth and chroma format.
        return Ok(mb);
    }

    // 3. Intra prediction modes
    #[cfg(feature = "tracing-detail")]
    trace!(
        bits_after_mb_type = br.consumed(),
        mb_x,
        mb_y = _mb_y,
        "intra4x4 modes start"
    );
    if mb.is_intra4x4 {
        // Parse and resolve intra4x4 prediction modes for each 4x4 block.
        //
        // Block scan order maps (i8x8*4 + i4x4) -> (blk_x, blk_y) via BLOCK_INDEX_TO_XY
        // from mb.rs, which is equivalent to SCAN_TO_RASTER converting to raster order.
        //
        // For each block, the predicted mode is min(left_mode, top_mode), defaulting
        // to DC_PRED (2) if a neighbor is unavailable.
        //
        // If prev_intra4x4_pred_mode_flag == 1: use predicted mode.
        // If prev_intra4x4_pred_mode_flag == 0: actual = rem_mode + (rem_mode >= predicted).
        //
        // Reference: FFmpeg pred_intra_mode() in h264_mvpred.h
        const DC_PRED: u8 = 2;

        // Temporary raster-order mode cache for resolving predictions within the MB.
        // -1 (as i8) means unavailable.
        let mut mode_cache = [-1i8; 16]; // indexed in raster order (blk_y * 4 + blk_x)

        for &raster_idx in &SCAN_TO_RASTER {
            let blk_x = raster_idx % 4;
            let blk_y = raster_idx / 4;

            // Get left neighbor's mode
            let left_mode: i8 = if blk_x > 0 {
                mode_cache[raster_idx - 1]
            } else if neighbor.left_available {
                // Left neighbor is from the previous MB's right column.
                neighbor.left_intra4x4_mode[blk_y]
            } else {
                -1
            };

            // Get top neighbor's mode
            let top_mode: i8 = if blk_y > 0 {
                mode_cache[raster_idx - 4]
            } else if neighbor.top_available {
                // Top neighbor is from the MB above's bottom row.
                let abs_blk_x = mb_x as usize * 4 + blk_x;
                neighbor.top_intra4x4_mode[abs_blk_x]
            } else {
                -1
            };

            // Predicted mode = min(left, top), or DC_PRED if either unavailable
            let predicted = if left_mode < 0 || top_mode < 0 {
                DC_PRED
            } else {
                (left_mode as u8).min(top_mode as u8)
            };

            let prev_flag = br.get_bit();
            let mode = if prev_flag {
                predicted
            } else {
                let rem_mode = br.get_bits_32(3) as u8;
                if rem_mode >= predicted {
                    rem_mode + 1
                } else {
                    rem_mode
                }
            };

            mode_cache[raster_idx] = mode as i8;
            mb.intra4x4_pred_mode[raster_idx] = mode;
        }
    }

    #[cfg(feature = "tracing-detail")]
    tracing::trace!(bits_after_pred_modes = br.consumed(), "CAVLC MB header");
    if is_intra {
        // Parse chroma intra prediction mode.
        mb.chroma_pred_mode = get_ue_golomb(br)? as u8;
        if mb.chroma_pred_mode > 3 {
            return Err(Error::InvalidData);
        }
    }

    // 4. Inter prediction (P-slice, not intra)
    if !is_intra && (slice_type == SliceType::P || slice_type == SliceType::SP) {
        match mb.mb_type {
            0 => {
                // P_L0_16x16: one ref_idx, one mvd
                mb.ref_idx_l0[0] = read_ref_idx(br, num_ref_idx_l0_active)? as i8;
                mb.mvd_l0[0][0] = get_se_golomb(br)? as i16;
                mb.mvd_l0[0][1] = get_se_golomb(br)? as i16;
            }
            1 => {
                // P_L0_L0_16x8: two ref_idx, two mvd
                mb.ref_idx_l0[0] = read_ref_idx(br, num_ref_idx_l0_active)? as i8;
                mb.ref_idx_l0[1] = read_ref_idx(br, num_ref_idx_l0_active)? as i8;
                mb.mvd_l0[0][0] = get_se_golomb(br)? as i16;
                mb.mvd_l0[0][1] = get_se_golomb(br)? as i16;
                mb.mvd_l0[1][0] = get_se_golomb(br)? as i16;
                mb.mvd_l0[1][1] = get_se_golomb(br)? as i16;
            }
            2 => {
                // P_L0_L0_8x16: two ref_idx, two mvd
                mb.ref_idx_l0[0] = read_ref_idx(br, num_ref_idx_l0_active)? as i8;
                mb.ref_idx_l0[1] = read_ref_idx(br, num_ref_idx_l0_active)? as i8;
                mb.mvd_l0[0][0] = get_se_golomb(br)? as i16;
                mb.mvd_l0[0][1] = get_se_golomb(br)? as i16;
                mb.mvd_l0[1][0] = get_se_golomb(br)? as i16;
                mb.mvd_l0[1][1] = get_se_golomb(br)? as i16;
            }
            3 | 4 => {
                // P_8x8 / P_8x8ref0: parse sub_mb_type, then ref_idx and mvd per sub-partition.
                for i in 0..4 {
                    let sub_mt = get_ue_golomb(br)?;
                    if sub_mt >= 4 {
                        return Err(Error::InvalidData);
                    }
                    mb.sub_mb_type[i] = sub_mt as u8;
                }

                // ref_idx for each 8x8 partition
                let ref_count = if mb.mb_type == 4 {
                    1 // P_8x8ref0: all refs forced to 0
                } else {
                    num_ref_idx_l0_active
                };
                for i in 0..4 {
                    mb.ref_idx_l0[i] = read_ref_idx(br, ref_count)? as i8;
                }

                // mvd for each sub-partition
                for i in 0..4 {
                    let sub_part_count = P_SUB_MB_PARTITION_COUNT[mb.sub_mb_type[i] as usize];
                    for j in 0..sub_part_count as usize {
                        let mvd_idx = i * 4 + j;
                        if mvd_idx < 16 {
                            mb.mvd_l0[mvd_idx][0] = get_se_golomb(br)? as i16;
                            mb.mvd_l0[mvd_idx][1] = get_se_golomb(br)? as i16;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // 4b. Inter prediction (B-slice, not intra)
    if !is_intra && slice_type == SliceType::B && !mb.is_direct {
        let part_count = mb.partition_count as usize;

        if mb.mb_type == 22 {
            // B_8x8: parse sub_mb_type for each 8x8 partition
            for i in 0..4 {
                let sub_mt = get_ue_golomb(br)?;
                if sub_mt >= 13 {
                    return Err(Error::InvalidData);
                }
                mb.sub_mb_type[i] = sub_mt as u8;
            }

            // Parse ref_idx L0 for each 8x8 partition that uses L0
            for i in 0..4 {
                let info = &B_SUB_MB_TYPE_INFO[mb.sub_mb_type[i] as usize];
                if info.2 {
                    // uses L0
                    mb.ref_idx_l0[i] = read_ref_idx(br, num_ref_idx_l0_active)? as i8;
                }
            }
            // Parse ref_idx L1 for each 8x8 partition that uses L1
            for i in 0..4 {
                let info = &B_SUB_MB_TYPE_INFO[mb.sub_mb_type[i] as usize];
                if info.3 {
                    // uses L1
                    mb.ref_idx_l1[i] = read_ref_idx(br, _num_ref_idx_l1_active)? as i8;
                }
            }
            // Parse mvd L0 for each sub-partition
            for i in 0..4 {
                let info = &B_SUB_MB_TYPE_INFO[mb.sub_mb_type[i] as usize];
                if info.2 {
                    let sub_part_count = info.0 as usize;
                    for j in 0..sub_part_count {
                        let mvd_idx = i * 4 + j;
                        if mvd_idx < 16 {
                            mb.mvd_l0[mvd_idx][0] = get_se_golomb(br)? as i16;
                            mb.mvd_l0[mvd_idx][1] = get_se_golomb(br)? as i16;
                        }
                    }
                }
            }
            // Parse mvd L1 for each sub-partition
            for i in 0..4 {
                let info = &B_SUB_MB_TYPE_INFO[mb.sub_mb_type[i] as usize];
                if info.3 {
                    let sub_part_count = info.0 as usize;
                    for j in 0..sub_part_count {
                        let mvd_idx = i * 4 + j;
                        if mvd_idx < 16 {
                            mb.mvd_l1[mvd_idx][0] = get_se_golomb(br)? as i16;
                            mb.mvd_l1[mvd_idx][1] = get_se_golomb(br)? as i16;
                        }
                    }
                }
            }
        } else {
            // Non-8x8 B-slice partitions (1 or 2 partitions)
            // Parse ref_idx L0 for all partitions that use L0
            for p in 0..part_count {
                if mb.b_list_flags[p][0] {
                    mb.ref_idx_l0[p] = read_ref_idx(br, num_ref_idx_l0_active)? as i8;
                }
            }
            // Parse ref_idx L1 for all partitions that use L1
            for p in 0..part_count {
                if mb.b_list_flags[p][1] {
                    mb.ref_idx_l1[p] = read_ref_idx(br, _num_ref_idx_l1_active)? as i8;
                }
            }
            // Parse mvd L0 for all partitions that use L0
            for p in 0..part_count {
                if mb.b_list_flags[p][0] {
                    mb.mvd_l0[p][0] = get_se_golomb(br)? as i16;
                    mb.mvd_l0[p][1] = get_se_golomb(br)? as i16;
                }
            }
            // Parse mvd L1 for all partitions that use L1
            for p in 0..part_count {
                if mb.b_list_flags[p][1] {
                    mb.mvd_l1[p][0] = get_se_golomb(br)? as i16;
                    mb.mvd_l1[p][1] = get_se_golomb(br)? as i16;
                }
            }
        }
    }

    // 5. CBP
    #[cfg(feature = "tracing-detail")]
    tracing::trace!(bits_before_cbp = br.consumed(), "CAVLC MB header");
    if !mb.is_intra16x16 {
        let cbp_code = get_ue_golomb(br)?;
        if cbp_code > 47 {
            return Err(Error::InvalidData);
        }
        mb.cbp = if mb.is_intra4x4 || (is_intra && !mb.is_intra16x16) {
            GOLOMB_TO_INTRA4X4_CBP[cbp_code as usize] as u32
        } else {
            GOLOMB_TO_INTER_CBP[cbp_code as usize] as u32
        };
    }
    // For I_16x16, cbp was already set from mb_type.

    // 6. mb_qp_delta and residual coefficients
    #[cfg(feature = "tracing-detail")]
    tracing::trace!(bits_before_qp_delta = br.consumed(), "CAVLC MB header");
    if mb.cbp > 0 || mb.is_intra16x16 {
        mb.mb_qp_delta = get_se_golomb(br)?;
        #[cfg(feature = "tracing-detail")]
        tracing::trace!(
            bits_after_qp_delta = br.consumed(),
            mb_qp_delta = mb.mb_qp_delta,
            "CAVLC MB header"
        );
        decode_residual_blocks(br, &mut mb, neighbor, mb_x)?;
    }

    Ok(mb)
}

/// Decode intra mb_type and fill in the MacroblockCavlc fields.
fn decode_intra_mb_type(mb: &mut MacroblockCavlc, mt: u32) -> Result<()> {
    if mt > 25 {
        return Err(Error::InvalidData);
    }

    if mt == 0 {
        // I_4x4
        mb.is_intra4x4 = true;
        mb.mb_type = mt;
    } else if mt == 25 {
        // I_PCM
        mb.is_pcm = true;
        mb.mb_type = mt;
    } else {
        // I_16x16 (mt 1..24)
        mb.is_intra16x16 = true;
        mb.mb_type = mt;
        let (pred_mode, cbp) = I_MB_TYPE_INFO[mt as usize];
        mb.intra16x16_mode = pred_mode as u8;
        mb.cbp = cbp as u32;
    }

    Ok(())
}

/// Read a reference index.
///
/// Matches FFmpeg's truncated exp-Golomb approach:
/// - If ref_count == 1, return 0 without reading.
/// - If ref_count == 2, read 1 bit XOR 1.
/// - Otherwise, read ue(v).
fn read_ref_idx(br: &mut BitReadBE<'_>, ref_count: u32) -> Result<u32> {
    if ref_count == 1 {
        Ok(0)
    } else if ref_count == 2 {
        Ok(u32::from(br.get_bit()) ^ 1)
    } else {
        let val = get_ue_golomb(br)?;
        if val >= ref_count {
            return Err(Error::InvalidData);
        }
        Ok(val)
    }
}

/// Decode residual blocks for a macroblock.
///
/// Handles I_16x16 (luma DC + AC), I_4x4 (luma), and chroma DC + AC.
fn decode_residual_blocks(
    br: &mut BitReadBE<'_>,
    mb: &mut MacroblockCavlc,
    neighbor: &NeighborContext,
    mb_x: u32,
) -> Result<()> {
    let cbp = mb.cbp;

    if mb.is_intra16x16 {
        // Luma DC: 16 coefficients, nC from luma context of block 0.
        let nc = compute_nc(0, mb_x, neighbor, &mb.non_zero_count);
        let (dc_coeffs_scan, _dc_nz) = decode_residual(br, nc, 16)?;
        // Descan from zigzag to raster order for the Hadamard input.
        mb.luma_dc = zigzag_descan_4x4(&dc_coeffs_scan, 16);

        // Luma AC: 15 coefficients per block (DC is separate), only if luma CBP is non-zero.
        if cbp & 0x0F != 0 {
            for i8x8 in 0..4 {
                for i4x4 in 0..4 {
                    let scan_idx = i8x8 * 4 + i4x4;
                    let raster_idx = SCAN_TO_RASTER[scan_idx];
                    let nc = compute_nc(raster_idx, mb_x, neighbor, &mb.non_zero_count);
                    let (ac_coeffs_scan, nz) = decode_residual(br, nc, 15)?;
                    // Descan AC coefficients: scan positions 0..14 map to
                    // raster positions via zigzag_scan[1..16] (shifted by 1 since DC is separate).
                    for scan_pos in 0..15 {
                        mb.luma_coeffs[raster_idx][ZIGZAG_SCAN_4X4[scan_pos + 1]] =
                            ac_coeffs_scan[scan_pos];
                    }
                    mb.non_zero_count[raster_idx] = nz;
                }
            }
        }
    } else {
        // I_4x4 or inter: luma 4x4 blocks, 16 coefficients each.
        for i8x8 in 0..4 {
            if cbp & (1 << i8x8) != 0 {
                for i4x4 in 0..4 {
                    let scan_idx = i8x8 * 4 + i4x4;
                    let raster_idx = SCAN_TO_RASTER[scan_idx];
                    let nc = compute_nc(raster_idx, mb_x, neighbor, &mb.non_zero_count);
                    let (block_coeffs_scan, nz) = decode_residual(br, nc, 16)?;
                    // Descan from zigzag to raster order.
                    mb.luma_coeffs[raster_idx] = zigzag_descan_4x4(&block_coeffs_scan, 16);
                    mb.non_zero_count[raster_idx] = nz;
                }
            } else {
                // No coded coefficients in this 8x8 block.
                for i4x4 in 0..4 {
                    let scan_idx = i8x8 * 4 + i4x4;
                    let raster_idx = SCAN_TO_RASTER[scan_idx];
                    mb.non_zero_count[raster_idx] = 0;
                }
            }
        }
    }

    // Chroma (4:2:0)
    if cbp & 0x30 != 0 {
        // Chroma DC: 4 coefficients per plane, nC = -1.
        for chroma_idx in 0..2 {
            let (dc_coeffs, _dc_nz) = decode_residual(br, -1, 4)?;
            mb.chroma_dc[chroma_idx] = [dc_coeffs[0], dc_coeffs[1], dc_coeffs[2], dc_coeffs[3]];
        }
    }

    if cbp & 0x20 != 0 {
        // Chroma AC: 15 coefficients per block, 4 blocks per plane.
        for chroma_idx in 0..2 {
            for blk in 0..4 {
                let block_idx = 16 + chroma_idx * 4 + blk;
                let nc = compute_nc(block_idx, mb_x, neighbor, &mb.non_zero_count);
                let (ac_coeffs_scan, nz) = decode_residual(br, nc, 15)?;
                // Descan AC coefficients: scan positions 0..14 -> raster via zigzag_scan[1..16].
                for scan_pos in 0..15 {
                    mb.chroma_ac[chroma_idx][blk][ZIGZAG_SCAN_4X4[scan_pos + 1]] =
                        ac_coeffs_scan[scan_pos];
                }
                mb.non_zero_count[block_idx] = nz;
            }
        }
    } else {
        // Zero out chroma non_zero_count.
        for i in 16..24 {
            mb.non_zero_count[i] = 0;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a bitstream from individual bit values.
    fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
        let padded_len = ((bits.len() + 7) / 8) * 8 + 64; // pad for BitReadBE
        let mut padded_bits = bits.to_vec();
        padded_bits.resize(padded_len, 0);

        let mut bytes = Vec::new();
        for chunk in padded_bits.chunks(8) {
            let mut byte = 0u8;
            for (j, &b) in chunk.iter().enumerate() {
                byte |= (b & 1) << (7 - j);
            }
            bytes.push(byte);
        }
        bytes
    }

    /// Helper to build a bitstream from a binary string.
    fn bits_str_to_bytes(s: &str) -> Vec<u8> {
        let bits: Vec<u8> = s.chars().map(|c| if c == '1' { 1 } else { 0 }).collect();
        bits_to_bytes(&bits)
    }

    // --- decode_residual tests ---

    #[test]
    fn decode_residual_zero_coeffs() {
        // nC=0: coeff_token for total_coeff=0 is "1" (len=1, bits=1).
        let data = bits_str_to_bytes("1");
        let mut br = BitReadBE::new(&data);
        let (coeffs, nz) = decode_residual(&mut br, 0, 16).unwrap();
        assert_eq!(nz, 0);
        assert_eq!(coeffs, [0i16; 16]);
    }

    #[test]
    fn decode_residual_single_trailing_one_positive() {
        // nC=0: total_coeff=1, trailing_ones=1 => coeff_token "01" (len=2, bits=1)
        // trailing one sign bit: 0 => +1
        // total_zeros: total_coeff=1, max_coeff=4 => chroma DC table
        //   But let's use max_coeff=16 so we use the 4x4 table.
        //   total_zeros for total_coeff=1: we need total_zeros=0 => "1" (len=1, bits=1)
        // No run_before needed (only 1 coeff).
        //
        // Bitstream: 01 | 0 | 1
        // = "0101"
        let data = bits_str_to_bytes("0101");
        let mut br = BitReadBE::new(&data);
        let (coeffs, nz) = decode_residual(&mut br, 0, 16).unwrap();
        assert_eq!(nz, 1);
        // Single coefficient at position 0 (last scan position = total_coeff + total_zeros - 1 = 0).
        assert_eq!(coeffs[0], 1);
        for i in 1..16 {
            assert_eq!(coeffs[i], 0);
        }
    }

    #[test]
    fn decode_residual_single_trailing_one_negative() {
        // Same as above but trailing one sign bit = 1 => -1.
        // Bitstream: 01 | 1 | 1  = "0111"
        let data = bits_str_to_bytes("0111");
        let mut br = BitReadBE::new(&data);
        let (coeffs, nz) = decode_residual(&mut br, 0, 16).unwrap();
        assert_eq!(nz, 1);
        assert_eq!(coeffs[0], -1);
    }

    #[test]
    fn decode_residual_two_trailing_ones() {
        // nC=0: total_coeff=2, trailing_ones=2 => coeff_token "001" (len=3, bits=1)
        // trailing ones signs: bit0=0 (+1), bit1=1 (-1)
        // total_zeros for total_coeff=2, max_coeff=16: need total_zeros=0
        //   Table row 1 (total_coeff=2): total_zeros=0 => len=3, bits=7 => "111"
        // No run_before (total_zeros=0, both coeffs adjacent).
        //
        // Bitstream: 001 | 01 | 111
        // = "00101111"
        let data = bits_str_to_bytes("00101111");
        let mut br = BitReadBE::new(&data);
        let (coeffs, nz) = decode_residual(&mut br, 0, 16).unwrap();
        assert_eq!(nz, 2);
        // Two coefficients at positions 0 and 1.
        // levels[0] = +1 (first trailing one, highest frequency)
        // levels[1] = -1 (second trailing one, lower frequency)
        // With total_zeros=0, they go to positions 1 and 0.
        assert_eq!(coeffs[1], 1);
        assert_eq!(coeffs[0], -1);
    }

    #[test]
    fn decode_residual_chroma_dc_zero() {
        // Chroma DC: nC=-1, max_coeff=4.
        // coeff_token for total_coeff=0: len=2, bits=1 => "01"
        let data = bits_str_to_bytes("01");
        let mut br = BitReadBE::new(&data);
        let (coeffs, nz) = decode_residual(&mut br, -1, 4).unwrap();
        assert_eq!(nz, 0);
        assert_eq!(coeffs[0..4], [0, 0, 0, 0]);
    }

    // --- compute_nc tests ---

    #[test]
    fn compute_nc_no_neighbors() {
        let neighbor = NeighborContext::new(10);
        let nz_cache = [0u8; 24];
        // Block 0 (top-left of MB), no left or top available.
        let nc = compute_nc(0, 0, &neighbor, &nz_cache);
        assert_eq!(nc, 0);
    }

    #[test]
    fn compute_nc_top_only() {
        let mut neighbor = NeighborContext::new(10);
        neighbor.top_available = true;
        neighbor.top_nz_luma[0] = 4;
        let nz_cache = [0u8; 24];
        // Block 0: has top neighbor with nz=4, no left.
        let nc = compute_nc(0, 0, &neighbor, &nz_cache);
        assert_eq!(nc, 4);
    }

    #[test]
    fn compute_nc_left_only() {
        let mut neighbor = NeighborContext::new(10);
        neighbor.left_available = true;
        neighbor.left_nz[0] = 6;
        let nz_cache = [0u8; 24];
        // Block 0: has left neighbor with nz=6, no top.
        let nc = compute_nc(0, 0, &neighbor, &nz_cache);
        assert_eq!(nc, 6);
    }

    #[test]
    fn compute_nc_both_neighbors() {
        let mut neighbor = NeighborContext::new(10);
        neighbor.left_available = true;
        neighbor.top_available = true;
        neighbor.left_nz[0] = 4;
        neighbor.top_nz_luma[0] = 6;
        let nz_cache = [0u8; 24];
        // Block 0: both available => (4 + 6 + 1) >> 1 = 5
        let nc = compute_nc(0, 0, &neighbor, &nz_cache);
        assert_eq!(nc, 5);
    }

    #[test]
    fn compute_nc_intra_mb_left() {
        let neighbor = NeighborContext::new(10);
        let mut nz_cache = [0u8; 24];
        nz_cache[0] = 3; // block 0 has 3 non-zero coeffs
        // Block 1 (to the right of block 0 within MB): left = nz_cache[0] = 3, no top.
        let nc = compute_nc(1, 0, &neighbor, &nz_cache);
        assert_eq!(nc, 3);
    }

    #[test]
    fn compute_nc_intra_mb_top() {
        let neighbor = NeighborContext::new(10);
        let mut nz_cache = [0u8; 24];
        nz_cache[0] = 5; // block 0
        // Block 4 (below block 0 within MB): top = nz_cache[0] = 5, no left.
        let nc = compute_nc(4, 0, &neighbor, &nz_cache);
        assert_eq!(nc, 5);
    }

    #[test]
    fn compute_nc_intra_mb_both() {
        let neighbor = NeighborContext::new(10);
        let mut nz_cache = [0u8; 24];
        nz_cache[1] = 4; // block 1 (left of block 5)
        nz_cache[2] = 6; // block 2 (above is block 5? No, above block 5 is block 1)
        // Block 5 = (1, 1): left = block 4, top = block 1.
        nz_cache[4] = 2;
        nz_cache[1] = 8;
        // block 5: blk_x=1, blk_y=1. left = nz_cache[4] = 2, top = nz_cache[1] = 8.
        let nc = compute_nc(5, 0, &neighbor, &nz_cache);
        assert_eq!(nc, (2 + 8 + 1) >> 1); // 5
    }

    // --- NeighborContext update tests ---

    #[test]
    fn neighbor_context_update() {
        let mut ctx = NeighborContext::new(2);
        let nz: [u8; 24] = [
            1, 2, 3, 4, // luma row 0
            5, 6, 7, 8, // luma row 1
            9, 10, 11, 12, // luma row 2
            13, 14, 15, 16, // luma row 3
            17, 18, 19, 20, // Cb
            21, 22, 23, 24, // Cr
        ];
        ctx.update_after_mb(0, &nz, &[-1i8; 16]);

        // Top: bottom row of luma
        assert_eq!(ctx.top_nz_luma[0], 13);
        assert_eq!(ctx.top_nz_luma[1], 14);
        assert_eq!(ctx.top_nz_luma[2], 15);
        assert_eq!(ctx.top_nz_luma[3], 16);

        // Left: right column of luma
        assert_eq!(ctx.left_nz[0], 4);
        assert_eq!(ctx.left_nz[1], 8);
        assert_eq!(ctx.left_nz[2], 12);
        assert_eq!(ctx.left_nz[3], 16);

        // Cb: bottom row
        assert_eq!(ctx.top_nz_cb[0], 19);
        assert_eq!(ctx.top_nz_cb[1], 20);
        // Cb: right column
        assert_eq!(ctx.left_nz[4], 18);
        assert_eq!(ctx.left_nz[5], 20);

        // Cr: bottom row
        assert_eq!(ctx.top_nz_cr[0], 23);
        assert_eq!(ctx.top_nz_cr[1], 24);
        // Cr: right column
        assert_eq!(ctx.left_nz[6], 22);
        assert_eq!(ctx.left_nz[7], 24);
    }

    // --- Level decoding edge cases ---

    #[test]
    fn get_level_prefix_zero() {
        // level_prefix = 0: bitstream starts with "1" => 0 leading zeros.
        let data = bits_str_to_bytes("1");
        let mut br = BitReadBE::new(&data);
        assert_eq!(get_level_prefix(&mut br).unwrap(), 0);
    }

    #[test]
    fn get_level_prefix_three() {
        // level_prefix = 3: "0001" (3 leading zeros then 1).
        let data = bits_str_to_bytes("0001");
        let mut br = BitReadBE::new(&data);
        assert_eq!(get_level_prefix(&mut br).unwrap(), 3);
    }

    #[test]
    fn get_level_prefix_error() {
        // All zeros => error.
        let data = [0u8; 16];
        let mut br = BitReadBE::new(&data);
        assert!(get_level_prefix(&mut br).is_err());
    }

    // --- decode_intra_mb_type tests ---

    #[test]
    fn intra_mb_type_i4x4() {
        let mut mb = MacroblockCavlc::default();
        decode_intra_mb_type(&mut mb, 0).unwrap();
        assert!(mb.is_intra4x4);
        assert!(!mb.is_intra16x16);
        assert!(!mb.is_pcm);
    }

    #[test]
    fn intra_mb_type_i16x16() {
        let mut mb = MacroblockCavlc::default();
        decode_intra_mb_type(&mut mb, 1).unwrap();
        assert!(!mb.is_intra4x4);
        assert!(mb.is_intra16x16);
        // mb_type 1: pred_mode=0, cbp=0 (H.264 Table 7-11)
        assert_eq!(mb.intra16x16_mode, 0);
        assert_eq!(mb.cbp, 0);
    }

    #[test]
    fn intra_mb_type_i16x16_with_cbp() {
        let mut mb = MacroblockCavlc::default();
        decode_intra_mb_type(&mut mb, 13).unwrap();
        assert!(mb.is_intra16x16);
        // mb_type 13: pred_mode=0, cbp=15 (H.264 Table 7-11)
        assert_eq!(mb.intra16x16_mode, 0);
        assert_eq!(mb.cbp, 15);
    }

    #[test]
    fn intra_mb_type_i16x16_full_cbp() {
        let mut mb = MacroblockCavlc::default();
        decode_intra_mb_type(&mut mb, 24).unwrap();
        assert!(mb.is_intra16x16);
        // mb_type 24: pred_mode=3, cbp=47 (luma=15, chroma=32)
        assert_eq!(mb.intra16x16_mode, 3);
        assert_eq!(mb.cbp, 47);
    }

    #[test]
    fn intra_mb_type_ipcm() {
        let mut mb = MacroblockCavlc::default();
        decode_intra_mb_type(&mut mb, 25).unwrap();
        assert!(mb.is_pcm);
        assert!(!mb.is_intra4x4);
        assert!(!mb.is_intra16x16);
    }

    #[test]
    fn intra_mb_type_invalid() {
        let mut mb = MacroblockCavlc::default();
        assert!(decode_intra_mb_type(&mut mb, 26).is_err());
    }

    // --- Chroma nC computation ---

    #[test]
    fn compute_nc_chroma_no_neighbors() {
        let neighbor = NeighborContext::new(10);
        let nz_cache = [0u8; 24];
        // Cb block 0 (index 16), no neighbors.
        let nc = compute_nc(16, 0, &neighbor, &nz_cache);
        assert_eq!(nc, 0);
    }

    #[test]
    fn compute_nc_chroma_with_top() {
        let mut neighbor = NeighborContext::new(10);
        neighbor.top_available = true;
        neighbor.top_nz_cb[0] = 3;
        let nz_cache = [0u8; 24];
        let nc = compute_nc(16, 0, &neighbor, &nz_cache);
        assert_eq!(nc, 3);
    }

    #[test]
    fn compute_nc_chroma_intra_mb() {
        let neighbor = NeighborContext::new(10);
        let mut nz_cache = [0u8; 24];
        nz_cache[16] = 2; // Cb block 0
        // Cb block 1 (index 17): left neighbor is block 16.
        let nc = compute_nc(17, 0, &neighbor, &nz_cache);
        assert_eq!(nc, 2);
    }

    // --- read_ref_idx tests ---

    #[test]
    fn read_ref_idx_single_ref() {
        let data = bits_str_to_bytes("1");
        let mut br = BitReadBE::new(&data);
        assert_eq!(read_ref_idx(&mut br, 1).unwrap(), 0);
    }

    #[test]
    fn read_ref_idx_two_refs() {
        // bit=0 => 0^1 = 1
        let data = bits_str_to_bytes("0");
        let mut br = BitReadBE::new(&data);
        assert_eq!(read_ref_idx(&mut br, 2).unwrap(), 1);

        // bit=1 => 1^1 = 0
        let data = bits_str_to_bytes("1");
        let mut br = BitReadBE::new(&data);
        assert_eq!(read_ref_idx(&mut br, 2).unwrap(), 0);
    }

    #[test]
    fn read_ref_idx_multiple_refs() {
        // ue(0) = "1" => ref_idx = 0
        let data = bits_str_to_bytes("1");
        let mut br = BitReadBE::new(&data);
        assert_eq!(read_ref_idx(&mut br, 4).unwrap(), 0);
    }

    #[test]
    fn read_ref_idx_overflow() {
        // ue(4) = "00101" => ref_idx = 4, but ref_count = 4 => overflow
        let data = bits_str_to_bytes("00101");
        let mut br = BitReadBE::new(&data);
        assert!(read_ref_idx(&mut br, 4).is_err());
    }
}
