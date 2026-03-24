// CABAC (Context-Adaptive Binary Arithmetic Coding) decoder.
//
// Binary arithmetic decoder for H.264 Main/High profile entropy coding.
// Port of FFmpeg libavcodec/cabac.c + cabac_functions.h + h264_cabac.c.
//
// All arithmetic uses i32, matching FFmpeg's branchless tricks that rely
// on `>> 31` sign extension.
//
// Reference: FFmpeg libavcodec/cabac.c, cabac_functions.h, cabac.h, h264_cabac.c

use tracing::trace;
use wedeo_core::error::{Error, Result};

use crate::cabac_tables::{
    COEFF_ABS_LEVEL_M1_OFFSET, COEFF_ABS_LEVEL_TRANSITION, COEFF_ABS_LEVEL1_CTX,
    COEFF_ABS_LEVELGT1_CTX, LAST_COEFF_FLAG_OFFSET, LAST_COEFF_FLAG_OFFSET_8X8, LPS_RANGE,
    MLPS_STATE, NORM_SHIFT, SIGNIFICANT_COEFF_FLAG_OFFSET, SIGNIFICANT_COEFF_FLAG_OFFSET_8X8,
};
use crate::cavlc::{Macroblock, NeighborContext};
use crate::pps::Pps;
use crate::slice::SliceType;
use crate::tables::{B_MB_TYPE_INFO, B_SUB_MB_TYPE_INFO};

/// CABAC uses 16-bit precision internally (matching FFmpeg's CABAC_BITS=16).
const CABAC_BITS: u32 = 16;
const CABAC_MASK: i32 = (1 << CABAC_BITS) - 1;

/// CABAC binary arithmetic decoder.
///
/// Reads a byte-aligned CABAC bitstream and decodes binary symbols using
/// context-adaptive probability models.
pub struct CabacReader<'a> {
    /// Current interval offset (scaled by 2^CABAC_BITS).
    low: i32,
    /// Current interval range (9-bit, 256..510).
    range: i32,
    /// Current read position in the data buffer.
    pos: usize,
    /// RBSP data (byte-aligned, after slice header).
    data: &'a [u8],
}

impl<'a> CabacReader<'a> {
    /// Initialize a CABAC decoder from byte-aligned RBSP data.
    ///
    /// Uses the aligned init path from FFmpeg's `ff_init_cabac_decoder`
    /// (CABAC_BITS=16). Both aligned and unaligned paths produce equivalent
    /// decoded bits — the `low` offset difference is cosmetic (verified
    /// experimentally). Using a fixed path avoids tracking pointer alignment
    /// through the NAL parsing chain.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidData);
        }

        let low = ((data[0] as i32) << 18) | ((data[1] as i32) << 10) | (1 << 9);
        let range = 0x1FE;

        if (range << (CABAC_BITS + 1)) < low {
            return Err(Error::InvalidData);
        }

        Ok(Self {
            low,
            range,
            pos: 2,
            data,
        })
    }

    /// Refill the low register by reading 2 bytes from the bitstream.
    /// Called when the lower CABAC_BITS of low are all zero.
    ///
    /// Reference: FFmpeg `refill` (CABAC_BITS=16 path).
    #[inline(always)]
    fn refill(&mut self) {
        let b0 = self.byte_at(self.pos) as i32;
        let b1 = self.byte_at(self.pos + 1) as i32;
        self.low += (b0 << 9) + (b1 << 1);
        self.low -= CABAC_MASK;
        self.pos += 2;
    }

    /// Refill variant used by get_cabac. Computes the shift amount from
    /// the current low value to determine how many bits to inject.
    ///
    /// Reference: FFmpeg `refill2` (non-CLZ path, CABAC_BITS=16).
    #[inline(always)]
    fn refill2(&mut self) {
        // Compute number of consumed bits since last refill
        let x = (self.low ^ (self.low.wrapping_sub(1))) as u32;
        let i = 7 - NORM_SHIFT[(x >> (CABAC_BITS - 1)) as usize] as i32;

        let b0 = self.byte_at(self.pos) as i32;
        let b1 = self.byte_at(self.pos + 1) as i32;
        let x = -CABAC_MASK + (b0 << 9) + (b1 << 1);

        self.low += x << i;
        self.pos += 2;
    }

    /// Read a byte from the data buffer, returning 0 if past the end.
    #[inline(always)]
    fn byte_at(&self, pos: usize) -> u8 {
        if pos < self.data.len() {
            self.data[pos]
        } else {
            0
        }
    }

    /// Decode one context-adaptive binary symbol.
    ///
    /// `state` is the 7-bit probability state (bit 0 = MPS value,
    /// bits 1..6 = probability index). Updated in place after decode.
    ///
    /// Reference: FFmpeg `get_cabac_inline`.
    #[inline]
    pub fn get_cabac(&mut self, state: &mut u8) -> u8 {
        let pre_state = *state;
        let pre_low = self.low;
        let pre_range = self.range;

        let s = *state as i32;
        let range_lps = LPS_RANGE[(2 * (self.range & 0xC0) + s) as usize] as i32;

        self.range -= range_lps;
        let lps_mask = ((self.range << (CABAC_BITS + 1)) - self.low) >> 31;

        self.low -= (self.range << (CABAC_BITS + 1)) & lps_mask;
        self.range += (range_lps - self.range) & lps_mask;

        let s = s ^ lps_mask;
        *state = MLPS_STATE[(128 + s) as usize];
        let bit = s & 1;

        let shift = NORM_SHIFT[self.range as usize] as i32;
        self.range <<= shift;
        self.low <<= shift;
        if self.low & CABAC_MASK == 0 {
            self.refill2();
        }

        {
            use std::sync::atomic::{AtomicI32, Ordering};
            static BIN_COUNT: AtomicI32 = AtomicI32::new(0);
            static MAX_BINS: AtomicI32 = AtomicI32::new(-1);
            let max = MAX_BINS.load(Ordering::Relaxed);
            let max = if max < 0 {
                let v = std::env::var("CABAC_MAX_BINS")
                    .ok()
                    .and_then(|s| s.parse::<i32>().ok())
                    .unwrap_or(10000);
                MAX_BINS.store(v, Ordering::Relaxed);
                v
            } else {
                max
            };
            let n = BIN_COUNT.fetch_add(1, Ordering::Relaxed);
            if n < max {
                trace!(
                    "CABAC_BIN {} state={} low={} range={} -> bit={} post_low={} post_range={}",
                    n, pre_state, pre_low, pre_range, bit, self.low, self.range
                );
            }
        }

        bit as u8
    }

    /// Decode one equiprobable (bypass) binary symbol.
    ///
    /// Used for sign bits, exp-golomb suffixes, and other uniform-probability
    /// syntax elements.
    ///
    /// Reference: FFmpeg `get_cabac_bypass`.
    #[inline]
    pub fn get_cabac_bypass(&mut self) -> u8 {
        let pre_low = self.low;
        let pre_range = self.range;

        self.low += self.low;

        if self.low & CABAC_MASK == 0 {
            self.refill();
        }

        let range = self.range << (CABAC_BITS + 1);
        let bit = if self.low < range {
            0
        } else {
            self.low -= range;
            1
        };

        {
            use std::sync::atomic::{AtomicI32, Ordering};
            static BYPASS_COUNT: AtomicI32 = AtomicI32::new(0);
            static MAX_BINS_BP: AtomicI32 = AtomicI32::new(-1);
            let max = MAX_BINS_BP.load(Ordering::Relaxed);
            let max = if max < 0 {
                let v = std::env::var("CABAC_MAX_BINS")
                    .ok()
                    .and_then(|s| s.parse::<i32>().ok())
                    .unwrap_or(10000);
                MAX_BINS_BP.store(v, Ordering::Relaxed);
                v
            } else {
                max
            };
            let n = BYPASS_COUNT.fetch_add(1, Ordering::Relaxed);
            if n < max {
                trace!(
                    "CABAC_BYPASS {} low={} range={} -> bit={} post_low={}",
                    n, pre_low, pre_range, bit, self.low
                );
            }
        }

        bit
    }

    /// Decode a bypass symbol and apply it as a sign to `val`.
    ///
    /// Returns `val` if MPS (0), `-val` if LPS (1).
    /// Uses branchless arithmetic: `(val ^ mask) - mask` where
    /// mask = low >> 31 (sign extension).
    ///
    /// Reference: FFmpeg `get_cabac_bypass_sign`.
    #[inline]
    pub fn get_cabac_bypass_sign(&mut self, val: i32) -> i32 {
        let pre_low = self.low;
        let pre_range = self.range;

        self.low += self.low;

        if self.low & CABAC_MASK == 0 {
            self.refill();
        }

        let range = self.range << (CABAC_BITS + 1);
        self.low -= range;
        let mask = self.low >> 31;
        self.low += range & mask;
        let result = (val ^ mask) - mask;

        {
            use std::sync::atomic::{AtomicI32, Ordering};
            static BYPASS_SIGN_COUNT: AtomicI32 = AtomicI32::new(0);
            static MAX_BINS_BPS: AtomicI32 = AtomicI32::new(-1);
            let max = MAX_BINS_BPS.load(Ordering::Relaxed);
            let max = if max < 0 {
                let v = std::env::var("CABAC_MAX_BINS")
                    .ok()
                    .and_then(|s| s.parse::<i32>().ok())
                    .unwrap_or(10000);
                MAX_BINS_BPS.store(v, Ordering::Relaxed);
                v
            } else {
                max
            };
            let n = BYPASS_SIGN_COUNT.fetch_add(1, Ordering::Relaxed);
            if n < max {
                // bit=1 means negative (LPS), bit=0 means positive (MPS)
                let bit = if mask == 0 { 0 } else { 1 };
                trace!(
                    "CABAC_BYPASS_SIGN {} low={} range={} val={} -> bit={} result={} post_low={}",
                    n, pre_low, pre_range, val, bit, result, self.low
                );
            }
        }

        result
    }

    /// Check for end-of-slice (terminate symbol).
    ///
    /// Returns true if the slice ends here. The terminate symbol uses
    /// a fixed range reduction of 2.
    ///
    /// Reference: FFmpeg `get_cabac_terminate`.
    #[inline]
    pub fn get_cabac_terminate(&mut self) -> bool {
        let pre_low = self.low;
        let pre_range = self.range;

        self.range -= 2;
        let result = if self.low < (self.range << (CABAC_BITS + 1)) {
            // Not terminated: renormalize once
            let shift = ((self.range as u32).wrapping_sub(0x100) >> 31) as i32;
            self.range <<= shift;
            self.low <<= shift;
            if self.low & CABAC_MASK == 0 {
                self.refill();
            }
            false
        } else {
            true
        };

        {
            use std::sync::atomic::{AtomicI32, Ordering};
            static TERM_COUNT: AtomicI32 = AtomicI32::new(0);
            let n = TERM_COUNT.fetch_add(1, Ordering::Relaxed);
            trace!(
                "CABAC_TERM {} low={} range={} -> result={} post_low={} post_range={}",
                n, pre_low, pre_range, result as i32, self.low, self.range
            );
        }

        result
    }

    /// Skip `n` bytes and re-initialize the CABAC engine.
    ///
    /// Used after I_PCM macroblocks: raw sample bytes are read directly,
    /// then the CABAC engine must be re-initialized from the new position.
    ///
    /// Reference: FFmpeg `skip_bytes`.
    pub fn skip_bytes(&mut self, n: usize) -> Result<()> {
        // Recover the actual byte position from the CABAC state.
        // The engine may have read ahead; adjust backwards based on
        // whether the low bits indicate unconsumed refill data.
        let mut ptr = self.pos;
        if self.low & 0x1 != 0 {
            ptr -= 1;
        }
        if self.low & 0x1FF != 0 {
            ptr -= 1;
        }

        let new_start = ptr + n;
        if new_start + 2 > self.data.len() {
            return Err(Error::InvalidData);
        }

        // Re-init the engine from the new position (aligned path)
        self.low = (self.data[new_start] as i32) << 18
            | (self.data[new_start + 1] as i32) << 10
            | (1 << 9);
        self.range = 0x1FE;
        self.pos = new_start + 2;

        Ok(())
    }

    /// Return the current byte position in the data buffer.
    /// Useful for debugging and I_PCM byte reads.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Return the current low value (for debugging).
    pub fn low(&self) -> i32 {
        self.low
    }

    /// Return the current range value (for debugging).
    pub fn range(&self) -> i32 {
        self.range
    }

    /// Get the number of bytes remaining in the data buffer.
    pub fn bytes_remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Read a raw byte from the current position and advance.
    /// Used for I_PCM sample data within CABAC slices.
    pub fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(Error::InvalidData);
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    /// Check if bit 0 of low is set (for byte position recovery).
    /// Used by I_PCM to compute the true byte position.
    pub fn low_bit0(&self) -> bool {
        self.low & 0x1 != 0
    }

    /// Check if bits 0..8 of low are non-zero (for byte position recovery).
    /// Used by I_PCM to compute the true byte position.
    pub fn low_bits9(&self) -> bool {
        self.low & 0x1FF != 0
    }

    /// Get a reference to the underlying data buffer.
    pub fn data(&self) -> &[u8] {
        self.data
    }
}

// ---------------------------------------------------------------------------
// CABAC neighbor context
// ---------------------------------------------------------------------------

/// CABAC neighbor context — tracks per-MB state needed for context derivation.
///
/// CABAC needs different neighbor info than CAVLC: skip flags, intra16x16/PCM
/// flags, CBP values, and chroma prediction modes from left and top MBs.
///
/// Reference: FFmpeg h264_cabac.c context derivation functions.
pub struct CabacNeighborCtx {
    /// Picture width in macroblocks.
    pub mb_width: u32,
    /// Per-MB skip flag (true = skipped). Length = mb_width * mb_height.
    pub mb_skip: Vec<bool>,
    /// Per-MB flag: true if mb_type is I_16x16 or I_PCM. Length = mb_width * mb_height.
    pub mb_type_intra16x16_or_pcm: Vec<bool>,
    /// Per-MB CBP. Bits 0-3 = luma 8x8 blocks, bits 4-5 = chroma.
    /// Bits 6-7 = chroma DC coded flags (set by residual decode).
    /// Bits 8-9 = luma DC coded flags (for I16x16).
    /// Length = mb_width * mb_height.
    pub cbp: Vec<u32>,
    /// Per-MB chroma prediction mode. Length = mb_width * mb_height.
    pub chroma_pred_mode: Vec<u8>,
    /// Per-MB non-zero count for CABAC CBF context derivation.
    /// Layout per MB: [0..15] luma raster, [16..19] Cb, [20..23] Cr.
    /// Total length = (mb_width * mb_height) * 24.
    pub nz_count: Vec<u8>,
    /// Per-MB intra flag (true = any intra type). Length = mb_width * mb_height.
    pub mb_intra: Vec<bool>,
    /// Per-MB direct flag for B-slice. Length = mb_width * mb_height.
    pub mb_direct: Vec<bool>,
    /// Per-4x4-block absolute MVD values for left/top neighbor lookup.
    /// Layout: mvd_cache_l0[mb_idx * 16 + blk][component] = abs(mvd).
    /// Length = (mb_width * mb_height) * 16.
    pub mvd_cache_l0: Vec<[u8; 2]>,
    /// Same for L1.
    pub mvd_cache_l1: Vec<[u8; 2]>,
    /// Per-8x8-partition reference indices for CABAC ref context derivation.
    /// Layout: ref_cache_l0[mb_idx * 4 + part8x8] = ref_idx (-1 if intra/unavailable).
    /// Length = (mb_width * mb_height) * 4.
    pub ref_cache_l0: Vec<i8>,
    /// Same for L1.
    pub ref_cache_l1: Vec<i8>,
    /// Per-MB transform_size_8x8_flag for CABAC context 399 derivation.
    pub transform_8x8: Vec<bool>,
}

impl CabacNeighborCtx {
    /// Create a new CABAC neighbor context for a picture of the given dimensions.
    pub fn new(mb_width: u32, mb_height: u32) -> Self {
        let total_mbs = (mb_width * mb_height) as usize;
        Self {
            mb_width,
            mb_skip: vec![false; total_mbs],
            mb_type_intra16x16_or_pcm: vec![false; total_mbs],
            cbp: vec![0; total_mbs],
            chroma_pred_mode: vec![0; total_mbs],
            nz_count: vec![0; total_mbs * 24],
            mb_intra: vec![false; total_mbs],
            mb_direct: vec![false; total_mbs],
            mvd_cache_l0: vec![[0; 2]; total_mbs * 16],
            mvd_cache_l1: vec![[0; 2]; total_mbs * 16],
            ref_cache_l0: vec![-1; total_mbs * 4],
            ref_cache_l1: vec![-1; total_mbs * 4],
            transform_8x8: vec![false; total_mbs],
        }
    }

    /// Get the left CBP for the current MB.
    ///
    /// When the left neighbor is unavailable, returns a default value matching
    /// FFmpeg's `fill_decode_caches`: `0x7CF` for intra MBs, `0x00F` for inter.
    /// Both defaults have luma=0xF (all coded) and chroma CBP field=0.
    /// The intra default also sets chroma DC and luma DC coded flags (bits 6-10),
    /// which affects residual coded_block_flag context derivation.
    ///
    /// Reference: FFmpeg h264_mvpred.h:726-732.
    #[inline]
    fn left_cbp(
        &self,
        mb_idx: usize,
        mb_x: u32,
        slice_table: &[u16],
        cur_slice: u16,
        is_intra: bool,
    ) -> u32 {
        if mb_x == 0 {
            return if is_intra { 0x7CF } else { 0x00F };
        }
        let left_idx = mb_idx - 1;
        if slice_table[left_idx] != cur_slice {
            return if is_intra { 0x7CF } else { 0x00F };
        }
        self.cbp[left_idx]
    }

    /// Get the top CBP for the current MB.
    ///
    /// When the top neighbor is unavailable, returns the same default as `left_cbp`.
    ///
    /// Reference: FFmpeg h264_mvpred.h:722-725.
    #[inline]
    fn top_cbp(
        &self,
        mb_idx: usize,
        mb_y: u32,
        mb_width: u32,
        slice_table: &[u16],
        cur_slice: u16,
        is_intra: bool,
    ) -> u32 {
        if mb_y == 0 {
            return if is_intra { 0x7CF } else { 0x00F };
        }
        let top_idx = mb_idx - mb_width as usize;
        if slice_table[top_idx] != cur_slice {
            return if is_intra { 0x7CF } else { 0x00F };
        }
        self.cbp[top_idx]
    }

    /// Store the CABAC-relevant scalar state after decoding a macroblock.
    /// MVD and ref_idx are written separately by `CabacDecodeCache::write_back`
    /// (for coded MBs) or `update_mvd_ref_skip` (for skip MBs).
    #[allow(clippy::too_many_arguments)]
    pub fn update_after_mb(
        &mut self,
        mb_idx: usize,
        is_skip: bool,
        is_intra16x16_or_pcm: bool,
        is_intra: bool,
        is_direct: bool,
        cbp: u32,
        chroma_pred: u8,
        nz: &[u8; 24],
        transform_8x8_flag: bool,
    ) {
        self.mb_skip[mb_idx] = is_skip;
        self.mb_type_intra16x16_or_pcm[mb_idx] = is_intra16x16_or_pcm;
        self.mb_intra[mb_idx] = is_intra;
        self.mb_direct[mb_idx] = is_direct;
        self.cbp[mb_idx] = cbp;
        self.chroma_pred_mode[mb_idx] = chroma_pred;
        let nz_base = mb_idx * 24;
        self.nz_count[nz_base..nz_base + 24].copy_from_slice(nz);
        self.transform_8x8[mb_idx] = transform_8x8_flag;
    }

    /// Write skip-MB defaults for MVD and ref_idx.
    pub fn update_mvd_ref_skip(&mut self, mb_idx: usize) {
        let mvd_base = mb_idx * 16;
        self.mvd_cache_l0[mvd_base..mvd_base + 16].fill([0; 2]);
        self.mvd_cache_l1[mvd_base..mvd_base + 16].fill([0; 2]);
        let ref_base = mb_idx * 4;
        self.ref_cache_l0[ref_base..ref_base + 4].fill(0);
        self.ref_cache_l1[ref_base..ref_base + 4].fill(-1);
    }
}

// ---------------------------------------------------------------------------
// scan8-based per-MB decode cache
// ---------------------------------------------------------------------------

/// scan8 table: maps H.264 block index (0..50) to position in stride-8 cache.
///
/// Layout: luma 4x4 blocks [0-15] in Z-scan within 8x8 partitions,
/// chroma Cb [16-19], Cr [20-23] interleaved in cols 4-5/6-7,
/// 4:2:2 extensions [24-47], DC positions [48-50].
///
/// Reference: FFmpeg h264_parse.h:40-54.
pub const SCAN8: [usize; 51] = [
    12, 13, 20, 21, // luma [0-3]: top-left 8x8      (col 4-5, rows 1-2)
    14, 15, 22, 23, // luma [4-7]: top-right 8x8     (col 6-7, rows 1-2)
    28, 29, 36, 37, // luma [8-11]: bottom-left 8x8  (col 4-5, rows 3-4)
    30, 31, 38, 39, // luma [12-15]: bottom-right 8x8 (col 6-7, rows 3-4)
    52, 53, 60, 61, // chroma Cb [16-19]              (col 4-5, rows 6-7)
    54, 55, 62, 63, // chroma Cr [20-23] (4:2:0)     (col 6-7, rows 6-7)
    68, 69, 76, 77, // [24-27]
    70, 71, 78, 79, // [28-31]
    92, 93, 100, 101, // [32-35]
    94, 95, 102, 103, // [36-39]
    108, 109, 116, 117, // [40-43]
    110, 111, 118, 119, // [44-47]
    0, 40, 80, // DC [48-50]
];

/// Sentinel: reference index not used by this list.
pub const LIST_NOT_USED: i8 = -1;

/// Per-MB decode cache matching FFmpeg's scan8-indexed layout.
///
/// Filled by `fill()` before each MB's inter prediction, updated during
/// decode via `fill_rectangle`, persisted by `write_back()` after each MB.
///
/// Reference: FFmpeg h264dec.h:290-302 (H264SliceContext cache fields).
pub struct CabacDecodeCache {
    /// Reference indices per list. Layout: [list][5*8].
    /// Values: >=0 ref_idx, LIST_NOT_USED (-1).
    pub ref_cache: [[i8; 40]; 2],
    /// Absolute MVD for CABAC context. Layout: [list][5*8][2].
    pub mvd_cache: [[[u8; 2]; 40]; 2],
}

/// Fill a w×h rectangle in a stride-8 cache starting at position `pos`.
///
/// Reference: FFmpeg rectangle.h fill_rectangle().
#[inline]
pub fn fill_rectangle<T: Copy>(
    cache: &mut [T],
    pos: usize,
    w: usize,
    h: usize,
    stride: usize,
    val: T,
) {
    for row in 0..h {
        for col in 0..w {
            cache[pos + row * stride + col] = val;
        }
    }
}

impl Default for CabacDecodeCache {
    fn default() -> Self {
        Self {
            ref_cache: [[LIST_NOT_USED; 40]; 2],
            mvd_cache: [[[0u8; 2]; 40]; 2],
        }
    }
}

impl CabacDecodeCache {
    /// Create a cache initialized with unavailable defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fill the decode cache from neighbor data before decoding inter prediction.
    ///
    /// Populates the left column (col 3) and top row (row 0) of each cache
    /// from the neighboring MBs' stored values. Internal positions (rows 1-4,
    /// cols 4-7) are filled by `fill_rectangle` during decode.
    ///
    /// Reference: FFmpeg h264_mvpred.h:576-929 (fill_decode_caches).
    #[allow(clippy::too_many_arguments)]
    pub fn fill(
        &mut self,
        cabac_nb: &CabacNeighborCtx,
        slice_table: &[u16],
        cur_slice: u16,
        mb_idx: usize,
        mb_x: u32,
        mb_y: u32,
        mb_width: u32,
    ) {
        let has_top = mb_y > 0 && slice_table[mb_idx - mb_width as usize] == cur_slice;
        let has_left = mb_x > 0 && slice_table[mb_idx - 1] == cur_slice;

        for list in 0..2 {
            let ref_store = if list == 0 {
                &cabac_nb.ref_cache_l0
            } else {
                &cabac_nb.ref_cache_l1
            };
            let mvd_store = if list == 0 {
                &cabac_nb.mvd_cache_l0
            } else {
                &cabac_nb.mvd_cache_l1
            };

            // Top neighbor: bottom row of top MB
            if has_top {
                let top_idx = mb_idx - mb_width as usize;
                if !cabac_nb.mb_intra[top_idx] {
                    // ref: bottom-left (part 2) fills cols 4-5, bottom-right (part 3) fills cols 6-7
                    let r2 = ref_store[top_idx * 4 + 2];
                    let r3 = ref_store[top_idx * 4 + 3];
                    self.ref_cache[list][4] = r2;
                    self.ref_cache[list][5] = r2;
                    self.ref_cache[list][6] = r3;
                    self.ref_cache[list][7] = r3;
                    // mvd: bottom row = raster indices 12..15
                    for col in 0..4 {
                        self.mvd_cache[list][4 + col] = mvd_store[top_idx * 16 + 12 + col];
                    }
                }
                // else: intra → ref stays LIST_NOT_USED, mvd stays [0,0]
            }

            // Left neighbor: right column of left MB
            if has_left {
                let left_idx = mb_idx - 1;
                if !cabac_nb.mb_intra[left_idx] {
                    // ref: top-right (part 1) → rows 1-2, bottom-right (part 3) → rows 3-4
                    let r1 = ref_store[left_idx * 4 + 1];
                    let r3 = ref_store[left_idx * 4 + 3];
                    self.ref_cache[list][SCAN8[0] - 1] = r1; // col 3, row 1
                    self.ref_cache[list][SCAN8[2] - 1] = r1; // col 3, row 2
                    self.ref_cache[list][SCAN8[8] - 1] = r3; // col 3, row 3
                    self.ref_cache[list][SCAN8[10] - 1] = r3; // col 3, row 4
                    // mvd: right column = raster indices 3,7,11,15
                    self.mvd_cache[list][SCAN8[0] - 1] = mvd_store[left_idx * 16 + 3];
                    self.mvd_cache[list][SCAN8[2] - 1] = mvd_store[left_idx * 16 + 7];
                    self.mvd_cache[list][SCAN8[8] - 1] = mvd_store[left_idx * 16 + 11];
                    self.mvd_cache[list][SCAN8[10] - 1] = mvd_store[left_idx * 16 + 15];
                }
            }
        }
    }

    /// Write back MVD and ref_idx from the cache to persistent flat storage.
    ///
    /// Reference: FFmpeg h264_mvpred.h:94-128 (write_back_motion_list).
    pub fn write_back(&self, cabac_nb: &mut CabacNeighborCtx, mb_idx: usize) {
        for list in 0..2 {
            let ref_store = if list == 0 {
                &mut cabac_nb.ref_cache_l0
            } else {
                &mut cabac_nb.ref_cache_l1
            };
            let mvd_store = if list == 0 {
                &mut cabac_nb.mvd_cache_l0
            } else {
                &mut cabac_nb.mvd_cache_l1
            };

            // Write back ref indices (4 per MB, one per 8x8 partition)
            let ref_base = mb_idx * 4;
            ref_store[ref_base] = self.ref_cache[list][SCAN8[0]];
            ref_store[ref_base + 1] = self.ref_cache[list][SCAN8[4]];
            ref_store[ref_base + 2] = self.ref_cache[list][SCAN8[8]];
            ref_store[ref_base + 3] = self.ref_cache[list][SCAN8[12]];

            // Write back MVD (16 per MB, raster-ordered from scan8 positions)
            let mvd_base = mb_idx * 16;
            for raster in 0..16 {
                let cache_pos = (4 + raster % 4) + (1 + raster / 4) * 8;
                mvd_store[mvd_base + raster] = self.mvd_cache[list][cache_pos];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CABAC macroblock type info table (I-slice)
// ---------------------------------------------------------------------------

/// I-slice mb_type info: (prediction_mode, cbp).
/// Same as in cavlc.rs. Index 0 = I_4x4, 1..24 = I_16x16, 25 = I_PCM.
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
    (0, 15), // 13: I_16x16_0_0_1
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

/// Block scan order: maps H.264 block scan index to raster-order index.
/// Identical to SCAN_TO_RASTER in cavlc.rs.
const SCAN_TO_RASTER: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

/// Standard 4x4 zigzag scan: maps scan position (0..15) to raster position
/// (row * 4 + col) within the 4x4 block.
///
/// From FFmpeg `ff_zigzag_scan` in mathtables.c.
const ZIGZAG_SCAN_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// 8x8 zigzag scan as usize for CABAC residual decode.
/// Same values as tables::ZIGZAG_SCAN_8X8 but typed as usize.
const ZIGZAG_SCAN_8X8_USIZE: [usize; 64] = {
    let src = crate::tables::ZIGZAG_SCAN_8X8;
    let mut out = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        out[i] = src[i] as usize;
        i += 1;
    }
    out
};

/// P sub-macroblock type partition counts.
pub const P_SUB_MB_PARTITION_COUNT: [u8; 4] = [1, 2, 2, 4];

// ---------------------------------------------------------------------------
// CABAC syntax element decode functions
// ---------------------------------------------------------------------------

/// Decode the mb_skip_flag for CABAC.
///
/// Context base: 11 for P-slices, 24 for B-slices.
/// Left/top non-skip neighbors add to the context index.
///
/// Reference: FFmpeg h264_cabac.c:1336-1371 (decode_cabac_mb_skip).
#[allow(clippy::too_many_arguments)]
pub fn decode_cabac_mb_skip(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    is_b_slice: bool,
) -> u8 {
    let mb_idx = (mb_y * mb_width + mb_x) as usize;
    let mut ctx = 0u32;

    // Check left neighbor
    if mb_x > 0 {
        let left_idx = mb_idx - 1;
        if slice_table[left_idx] == cur_slice && !cabac_nb.mb_skip[left_idx] {
            ctx += 1;
        }
    }

    // Check top neighbor
    if mb_y > 0 {
        let top_idx = mb_idx - mb_width as usize;
        if slice_table[top_idx] == cur_slice && !cabac_nb.mb_skip[top_idx] {
            ctx += 1;
        }
    }

    if is_b_slice {
        ctx += 13;
    }

    reader.get_cabac(&mut state[11 + ctx as usize])
}

/// Decode an intra mb_type from CABAC (I_4x4, I_16x16 variants, or I_PCM).
///
/// Reference: FFmpeg h264_cabac.c:1304-1334 (decode_cabac_intra_mb_type).
#[allow(clippy::too_many_arguments)]
fn decode_cabac_intra_mb_type(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    ctx_base: usize,
    intra_slice: bool,
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_idx: usize,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
) -> u32 {
    if intra_slice {
        let mut ctx = 0usize;
        // Left neighbor: check if I_16x16 or I_PCM
        if mb_x > 0 {
            let left_idx = mb_idx - 1;
            if slice_table[left_idx] == cur_slice && cabac_nb.mb_type_intra16x16_or_pcm[left_idx] {
                ctx += 1;
            }
        }
        // Top neighbor
        if mb_y > 0 {
            let top_idx = mb_idx - mb_width as usize;
            if slice_table[top_idx] == cur_slice && cabac_nb.mb_type_intra16x16_or_pcm[top_idx] {
                ctx += 1;
            }
        }
        trace!(
            "CABAC_INTRA_MB_TYPE ctx_base={} ctx={} state_idx={}",
            ctx_base,
            ctx,
            ctx_base + ctx
        );
        if reader.get_cabac(&mut state[ctx_base + ctx]) == 0 {
            return 0; // I_4x4
        }
    } else {
        trace!(
            "CABAC_INTRA_MB_TYPE ctx_base={} ctx=0 state_idx={} (inter_slice)",
            ctx_base, ctx_base
        );
        if reader.get_cabac(&mut state[ctx_base]) == 0 {
            return 0; // I_4x4
        }
    }

    // Check for I_PCM
    if reader.get_cabac_terminate() {
        return 25; // I_PCM
    }

    // I_16x16: decode sub-fields
    let intra_offset = if intra_slice { 1usize } else { 0 };
    // FFmpeg does `state += 2` for intra_slice then indexes from state[1],
    // so the absolute offset is ctx_base + 3. For inter, no pointer advance,
    // so state[1] = ctx_base + 1.
    let s_base = ctx_base + if intra_slice { 3 } else { 1 };

    let mut mb_type = 1u32;
    mb_type += 12 * reader.get_cabac(&mut state[s_base]) as u32; // cbp_luma != 0
    if reader.get_cabac(&mut state[s_base + 1]) != 0 {
        // cbp_chroma
        mb_type += 4 + 4 * reader.get_cabac(&mut state[s_base + 1 + intra_offset]) as u32;
    }
    mb_type += 2 * reader.get_cabac(&mut state[s_base + 2 + intra_offset]) as u32;
    mb_type += reader.get_cabac(&mut state[s_base + 2 + 2 * intra_offset]) as u32;

    mb_type
}

/// Decode intra 4x4 prediction mode using CABAC.
///
/// Reference: FFmpeg h264_cabac.c:1373-1385.
fn decode_cabac_mb_intra4x4_pred_mode(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    pred_mode: u8,
) -> u8 {
    if reader.get_cabac(&mut state[68]) != 0 {
        return pred_mode;
    }

    let mut mode = 0u8;
    mode += reader.get_cabac(&mut state[69]);
    mode += 2 * reader.get_cabac(&mut state[69]);
    mode += 4 * reader.get_cabac(&mut state[69]);

    if mode >= pred_mode { mode + 1 } else { mode }
}

/// Decode chroma prediction mode using CABAC (truncated unary, max 3).
///
/// Reference: FFmpeg h264_cabac.c:1387-1410.
#[allow(clippy::too_many_arguments)]
fn decode_cabac_mb_chroma_pre_mode(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_idx: usize,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
) -> u8 {
    let mut ctx = 0usize;

    // Left neighbor: chroma_pred_mode != 0
    if mb_x > 0 {
        let left_idx = mb_idx - 1;
        if slice_table[left_idx] == cur_slice && cabac_nb.chroma_pred_mode[left_idx] != 0 {
            ctx += 1;
        }
    }
    // Top neighbor
    if mb_y > 0 {
        let top_idx = mb_idx - mb_width as usize;
        if slice_table[top_idx] == cur_slice && cabac_nb.chroma_pred_mode[top_idx] != 0 {
            ctx += 1;
        }
    }

    if reader.get_cabac(&mut state[64 + ctx]) == 0 {
        return 0;
    }
    if reader.get_cabac(&mut state[64 + 3]) == 0 {
        return 1;
    }
    if reader.get_cabac(&mut state[64 + 3]) == 0 {
        2
    } else {
        3
    }
}

/// Decode luma CBP (4 bits, one per 8x8 block) using CABAC.
///
/// Reference: FFmpeg h264_cabac.c:1412-1428.
fn decode_cabac_mb_cbp_luma(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    left_cbp: u32,
    top_cbp: u32,
) -> u32 {
    let cbp_a = left_cbp;
    let cbp_b = top_cbp;

    // Bit 0 (top-left 8x8): left=bit1 of cbp_a, top=bit2 of cbp_b
    let ctx = (cbp_a & 0x02 == 0) as usize + 2 * (cbp_b & 0x04 == 0) as usize;
    let mut cbp = reader.get_cabac(&mut state[73 + ctx]) as u32;

    // Bit 1 (top-right 8x8): left=bit0 of cbp, top=bit3 of cbp_b
    let ctx = (cbp & 0x01 == 0) as usize + 2 * (cbp_b & 0x08 == 0) as usize;
    cbp += (reader.get_cabac(&mut state[73 + ctx]) as u32) << 1;

    // Bit 2 (bottom-left 8x8): left=bit3 of cbp_a, top=bit0 of cbp
    let ctx = (cbp_a & 0x08 == 0) as usize + 2 * (cbp & 0x01 == 0) as usize;
    cbp += (reader.get_cabac(&mut state[73 + ctx]) as u32) << 2;

    // Bit 3 (bottom-right 8x8): left=bit2 of cbp, top=bit1 of cbp
    let ctx = (cbp & 0x04 == 0) as usize + 2 * (cbp & 0x02 == 0) as usize;
    cbp += (reader.get_cabac(&mut state[73 + ctx]) as u32) << 3;

    cbp
}

/// Decode chroma CBP (2 bits encoded in two steps) using CABAC.
///
/// Reference: FFmpeg h264_cabac.c:1429-1447.
fn decode_cabac_mb_cbp_chroma(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    left_cbp: u32,
    top_cbp: u32,
) -> u32 {
    let cbp_a = (left_cbp >> 4) & 0x03;
    let cbp_b = (top_cbp >> 4) & 0x03;

    let mut ctx = 0usize;
    if cbp_a > 0 {
        ctx += 1;
    }
    if cbp_b > 0 {
        ctx += 2;
    }
    if reader.get_cabac(&mut state[77 + ctx]) == 0 {
        return 0;
    }

    let mut ctx = 4usize;
    if cbp_a == 2 {
        ctx += 1;
    }
    if cbp_b == 2 {
        ctx += 2;
    }
    1 + reader.get_cabac(&mut state[77 + ctx]) as u32
}

/// Decode P sub-macroblock type using CABAC.
///
/// Reference: FFmpeg h264_cabac.c:1449-1458.
fn decode_cabac_p_mb_sub_type(reader: &mut CabacReader, state: &mut [u8; 1024]) -> u8 {
    if reader.get_cabac(&mut state[21]) != 0 {
        return 0; // 8x8
    }
    if reader.get_cabac(&mut state[22]) == 0 {
        return 1; // 8x4
    }
    if reader.get_cabac(&mut state[23]) != 0 {
        2 // 4x8
    } else {
        3 // 4x4
    }
}

/// Decode B sub-macroblock type using CABAC.
///
/// Reference: FFmpeg h264_cabac.c:1459-1475.
fn decode_cabac_b_mb_sub_type(reader: &mut CabacReader, state: &mut [u8; 1024]) -> u8 {
    if reader.get_cabac(&mut state[36]) == 0 {
        return 0; // B_Direct_8x8
    }
    if reader.get_cabac(&mut state[37]) == 0 {
        return 1 + reader.get_cabac(&mut state[39]); // B_L0_8x8 or B_L1_8x8
    }
    let mut sub_type = 3u8;
    if reader.get_cabac(&mut state[38]) != 0 {
        if reader.get_cabac(&mut state[39]) != 0 {
            return 11 + reader.get_cabac(&mut state[39]); // B_L1_4x4 or B_Bi_4x4
        }
        sub_type += 4;
    }
    sub_type += 2 * reader.get_cabac(&mut state[39]);
    sub_type += reader.get_cabac(&mut state[39]);
    sub_type
}

/// Decode reference index using CABAC (unary code).
///
/// `cache_pos` is the scan8 cache position of the partition's top-left block.
/// Left and top ref neighbors are read from `cache.ref_cache[list]` at
/// `cache_pos - 1` and `cache_pos - 8`, which reach the correct neighbor
/// positions thanks to the scan8 layout.
///
/// Reference: FFmpeg h264_cabac.c:1477-1503.
#[allow(clippy::too_many_arguments)]
fn decode_cabac_mb_ref(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    cache: &CabacDecodeCache,
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_idx: usize,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    list: usize,
    cache_pos: usize,
    is_b_slice: bool,
) -> i32 {
    let ref_a = cache.ref_cache[list][cache_pos - 1] as i32;
    let ref_b = cache.ref_cache[list][cache_pos - 8] as i32;

    let mut ctx = 0usize;

    if is_b_slice {
        let left_pos = cache_pos - 1;
        let top_pos = cache_pos - 8;
        // B-slice: check ref > 0 and neighbor is not direct-mode.
        // Left column (col 3) → left MB, top row (row 0) → top MB,
        // anything else is intra-MB (never direct).
        if ref_a > 0
            && !is_cache_pos_direct(
                left_pos,
                cabac_nb,
                slice_table,
                cur_slice,
                mb_idx,
                mb_x,
                mb_y,
                mb_width,
            )
        {
            ctx += 1;
        }
        if ref_b > 0
            && !is_cache_pos_direct(
                top_pos,
                cabac_nb,
                slice_table,
                cur_slice,
                mb_idx,
                mb_x,
                mb_y,
                mb_width,
            )
        {
            ctx += 2;
        }
    } else {
        if ref_a > 0 {
            ctx += 1;
        }
        if ref_b > 0 {
            ctx += 2;
        }
    }

    let mut ref_idx = 0i32;
    while reader.get_cabac(&mut state[54 + ctx]) != 0 {
        ref_idx += 1;
        ctx = (ctx >> 2) + 4;
        if ref_idx >= 32 {
            return -1;
        }
    }
    ref_idx
}

/// Check if a cache position refers to a direct-mode neighbor MB.
/// Col 3 → left MB, row 0 → top MB, anything else → intra-MB (not direct).
#[allow(clippy::too_many_arguments)]
fn is_cache_pos_direct(
    pos: usize,
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_idx: usize,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
) -> bool {
    if pos % 8 == 3 {
        // Left column → check left MB
        is_neighbor_direct(
            cabac_nb,
            slice_table,
            cur_slice,
            mb_idx,
            mb_x,
            mb_y,
            mb_width,
            true,
        )
    } else if pos < 8 {
        // Top row → check top MB
        is_neighbor_direct(
            cabac_nb,
            slice_table,
            cur_slice,
            mb_idx,
            mb_x,
            mb_y,
            mb_width,
            false,
        )
    } else {
        false
    }
}

/// Check if a neighbor MB is direct mode (for B-slice ref context).
#[allow(clippy::too_many_arguments)]
fn is_neighbor_direct(
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_idx: usize,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    is_left: bool, // true = check left, false = check top
) -> bool {
    let nb_idx = if is_left {
        if mb_x == 0 {
            return false;
        }
        mb_idx - 1
    } else {
        if mb_y == 0 {
            return false;
        }
        mb_idx - mb_width as usize
    };
    if slice_table[nb_idx] != cur_slice {
        return false;
    }
    cabac_nb.mb_direct[nb_idx]
}

/// Decode one component of a motion vector difference using CABAC.
///
/// Reference: FFmpeg h264_cabac.c:1506-1541.
fn decode_cabac_mb_mvd_component(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    ctx_base: usize,
    amvd: i32,
) -> (i32, u8) {
    // Branchless context: ctx_base + ((amvd-3) >> 31) + ((amvd-33) >> 31) + 2
    // This maps: amvd <= 2 -> +2, 3 <= amvd <= 32 -> +1, amvd >= 33 -> +0
    let ctx_offset = (((amvd - 3) >> 31) + ((amvd - 33) >> 31) + 2) as usize;

    if reader.get_cabac(&mut state[ctx_base + ctx_offset]) == 0 {
        return (0, 0);
    }

    let mut mvd = 1i32;
    let mut ctx = ctx_base + 3;
    while mvd < 9 && reader.get_cabac(&mut state[ctx]) != 0 {
        if mvd < 4 {
            ctx += 1;
        }
        mvd += 1;
    }

    let abs_mvd = if mvd >= 9 {
        // Exp-golomb suffix via bypass
        let mut k = 3;
        while reader.get_cabac_bypass() != 0 {
            mvd += 1 << k;
            k += 1;
            if k > 24 {
                return (i32::MIN, 70);
            }
        }
        while k > 0 {
            k -= 1;
            mvd += (reader.get_cabac_bypass() as i32) << k;
        }
        if mvd < 70 { mvd as u8 } else { 70 }
    } else {
        mvd as u8
    };

    let signed_mvd = reader.get_cabac_bypass_sign(-mvd);
    (signed_mvd, abs_mvd)
}

/// Decode MVD for a block, returning (mvd_x, mvd_y, abs_mvd_x, abs_mvd_y).
///
/// Decode MVD for a block, returning (mvd_x, mvd_y, abs_mvd_x, abs_mvd_y).
///
/// `cache_pos` is the scan8 cache position for this block. Left and top
/// MVD neighbors are read from `cache.mvd_cache[list]` at `cache_pos - 1`
/// and `cache_pos - 8`.
///
/// Reference: FFmpeg DECODE_CABAC_MB_MVD macro, h264_cabac.c:1543-1556.
fn decode_cabac_mb_mvd(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    cache: &CabacDecodeCache,
    list: usize,
    cache_pos: usize,
) -> (i16, i16, u8, u8) {
    let left = &cache.mvd_cache[list][cache_pos - 1];
    let top = &cache.mvd_cache[list][cache_pos - 8];
    let amvd0 = left[0] as i32 + top[0] as i32;
    let amvd1 = left[1] as i32 + top[1] as i32;

    let (mx, abs_x) = decode_cabac_mb_mvd_component(reader, state, 40, amvd0);
    let (my, abs_y) = decode_cabac_mb_mvd_component(reader, state, 47, amvd1);

    if mx == i32::MIN || my == i32::MIN {
        return (0, 0, 0, 0); // overflow
    }

    (mx as i16, my as i16, abs_x, abs_y)
}

/// Sub-block offset within an 8x8 partition for P_8x8 sub_mb_types.
/// Returns (dx, dy, width, height) in 4x4 units for sub-partition index j.
pub const fn p8x8_sub_blk(sub_mb_type: u32, j: usize) -> (u32, u32, u32, u32) {
    match sub_mb_type {
        0 => (0, 0, 2, 2),                            // 8x8
        1 => (0, j as u32, 2, 1),                     // 8x4
        2 => (j as u32, 0, 1, 2),                     // 4x8
        3 => ((j & 1) as u32, (j >> 1) as u32, 1, 1), // 4x4
        _ => (0, 0, 2, 2),
    }
}

/// Decode QP delta using CABAC.
///
/// Reference: FFmpeg h264_cabac.c:2398-2426.
fn decode_cabac_mb_dqp(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    last_qscale_diff: i32,
) -> Result<i32> {
    let ctx0 = if last_qscale_diff != 0 { 1 } else { 0 };
    if reader.get_cabac(&mut state[60 + ctx0]) == 0 {
        return Ok(0);
    }

    let mut val = 1i32;
    let mut ctx = 2usize;
    while reader.get_cabac(&mut state[60 + ctx]) != 0 {
        ctx = 3;
        val += 1;
        if val > 102 {
            // prevent infinite loop (2 * max_qp)
            return Err(Error::InvalidData);
        }
    }

    if val & 0x01 != 0 {
        Ok((val + 1) >> 1)
    } else {
        Ok(-((val + 1) >> 1))
    }
}

// ---------------------------------------------------------------------------
// CABAC residual coefficient decode
// ---------------------------------------------------------------------------

/// CBF (coded block flag) context base offsets per block category.
///
/// From FFmpeg h264_cabac.c:1564.
const CBF_CTX_BASE: [u16; 14] = [
    85, 89, 93, 97, 101, 1012, 460, 464, 468, 1016, 472, 476, 480, 1020,
];

/// Compute the CBF context index for a given block.
///
/// Reference: FFmpeg h264_cabac.c:1558-1588 (get_cabac_cbf_ctx).
#[allow(clippy::too_many_arguments)]
fn get_cabac_cbf_ctx(
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_idx: usize,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    cat: usize,
    block_idx: usize,
    is_dc: bool,
    nz_cache: &[u8; 24],
    is_intra: bool,
) -> usize {
    let mut ctx = 0usize;
    // For unavailable neighbors, FFmpeg's fill_decode_caches uses:
    //   left_cbp/top_cbp = 0x7CF (intra) or 0x00F (inter)
    //   non_zero_count_cache = 64 (intra) or 0 (inter)
    // Reference: FFmpeg h264_mvpred.h:fill_decode_caches.
    let dc_unavail = if is_intra { 1u32 } else { 0 };

    if is_dc {
        if cat == 3 {
            // Chroma DC: idx is 0 or 1 (Cb or Cr)
            let chroma_idx = block_idx; // 0=Cb, 1=Cr
            // Left neighbor: check chroma DC coded flag
            let nza = if mb_x > 0 && slice_table[mb_idx - 1] == cur_slice {
                (cabac_nb.cbp[mb_idx - 1] >> (6 + chroma_idx)) & 0x01
            } else {
                dc_unavail
            };
            let nzb = if mb_y > 0 && slice_table[mb_idx - mb_width as usize] == cur_slice {
                (cabac_nb.cbp[mb_idx - mb_width as usize] >> (6 + chroma_idx)) & 0x01
            } else {
                dc_unavail
            };
            if nza > 0 {
                ctx += 1;
            }
            if nzb > 0 {
                ctx += 2;
            }
        } else {
            // Luma DC (I16x16): idx is 0
            let nza = if mb_x > 0 && slice_table[mb_idx - 1] == cur_slice {
                cabac_nb.cbp[mb_idx - 1] & 0x100
            } else if is_intra {
                0x100
            } else {
                0
            };
            let nzb = if mb_y > 0 && slice_table[mb_idx - mb_width as usize] == cur_slice {
                cabac_nb.cbp[mb_idx - mb_width as usize] & 0x100
            } else if is_intra {
                0x100
            } else {
                0
            };
            if nza > 0 {
                ctx += 1;
            }
            if nzb > 0 {
                ctx += 2;
            }
        }
    } else {
        // Non-DC: use non-zero count cache from left and top blocks.
        // block_idx is the 4x4 block scan index (0..15 for luma, 16..19 Cb, 20..23 Cr).
        let unavail_nz = if is_intra { 64 } else { 0 };
        let (nza, nzb) = get_nz_neighbors(
            cabac_nb,
            slice_table,
            cur_slice,
            mb_idx,
            mb_x,
            mb_y,
            mb_width,
            block_idx,
            nz_cache,
            unavail_nz,
        );
        if nza > 0 {
            ctx += 1;
        }
        if nzb > 0 {
            ctx += 2;
        }
    }

    CBF_CTX_BASE[cat] as usize + ctx
}

/// Get left and top non-zero count neighbors for a 4x4 block.
///
/// When the neighbor is unavailable, returns `unavail_nz` (64 for intra, 0 for inter).
/// This matches FFmpeg's `fill_decode_caches` which uses `CABAC && !IS_INTRA ? 0 : 64`.
///
/// Returns (nz_left, nz_top).
///
/// Reference: FFmpeg h264_mvpred.h:fill_decode_caches (non_zero_count_cache init).
#[allow(clippy::too_many_arguments)]
fn get_nz_neighbors(
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_idx: usize,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    block_idx: usize,
    nz_cache: &[u8; 24],
    unavail_nz: u8,
) -> (u8, u8) {
    if block_idx < 16 {
        // Luma block: raster index = block_idx
        let blk_x = block_idx % 4;
        let blk_y = block_idx / 4;

        let nza = if blk_x > 0 {
            nz_cache[block_idx - 1]
        } else if mb_x > 0 && slice_table[mb_idx - 1] == cur_slice {
            // Right column of left MB
            let left_idx = mb_idx - 1;
            let left_blk = blk_y * 4 + 3;
            cabac_nb.nz_count[left_idx * 24 + left_blk]
        } else {
            unavail_nz
        };

        let nzb = if blk_y > 0 {
            nz_cache[block_idx - 4]
        } else if mb_y > 0 && slice_table[mb_idx - mb_width as usize] == cur_slice {
            // Bottom row of top MB
            let top_idx = mb_idx - mb_width as usize;
            let top_blk = 12 + blk_x;
            cabac_nb.nz_count[top_idx * 24 + top_blk]
        } else {
            unavail_nz
        };

        (nza, nzb)
    } else if block_idx < 20 {
        // Cb chroma block: block_idx 16..19, raster within 2x2
        let c_idx = block_idx - 16;
        let blk_x = c_idx % 2;
        let blk_y = c_idx / 2;

        let nza = if blk_x > 0 {
            nz_cache[block_idx - 1]
        } else if mb_x > 0 && slice_table[mb_idx - 1] == cur_slice {
            let left_idx = mb_idx - 1;
            let left_blk = 16 + blk_y * 2 + 1;
            cabac_nb.nz_count[left_idx * 24 + left_blk]
        } else {
            unavail_nz
        };

        let nzb = if blk_y > 0 {
            nz_cache[block_idx - 2]
        } else if mb_y > 0 && slice_table[mb_idx - mb_width as usize] == cur_slice {
            let top_idx = mb_idx - mb_width as usize;
            let top_blk = 16 + 2 + blk_x;
            cabac_nb.nz_count[top_idx * 24 + top_blk]
        } else {
            unavail_nz
        };

        (nza, nzb)
    } else {
        // Cr chroma block: block_idx 20..23
        let c_idx = block_idx - 20;
        let blk_x = c_idx % 2;
        let blk_y = c_idx / 2;

        let nza = if blk_x > 0 {
            nz_cache[block_idx - 1]
        } else if mb_x > 0 && slice_table[mb_idx - 1] == cur_slice {
            let left_idx = mb_idx - 1;
            let left_blk = 20 + blk_y * 2 + 1;
            cabac_nb.nz_count[left_idx * 24 + left_blk]
        } else {
            unavail_nz
        };

        let nzb = if blk_y > 0 {
            nz_cache[block_idx - 2]
        } else if mb_y > 0 && slice_table[mb_idx - mb_width as usize] == cur_slice {
            let top_idx = mb_idx - mb_width as usize;
            let top_blk = 20 + 2 + blk_x;
            cabac_nb.nz_count[top_idx * 24 + top_blk]
        } else {
            unavail_nz
        };

        (nza, nzb)
    }
}

/// Decode a CABAC residual block (significance map + coefficient levels).
///
/// Outputs raw coefficients in scan order (same as CAVLC). The existing
/// dequant/IDCT pipeline handles the rest.
///
/// `cat`: block category (0=luma DC, 1=luma AC, 2=luma 4x4, 3=chroma DC, 4=chroma AC)
/// `max_coeff`: maximum number of coefficients (16 for 4x4, 15 for AC, 4 for chroma DC)
///
/// Returns the number of non-zero coefficients.
///
/// Reference: FFmpeg h264_cabac.c:1590-1776 (decode_cabac_residual_internal).
///
/// `scantable` maps significance-map position to raster position within the
/// coefficient block. For DC and full 4x4 blocks this is `ZIGZAG_SCAN_4X4`;
/// for AC blocks (where DC is separate) this is `&ZIGZAG_SCAN_4X4[1..]`
/// (offset by 1 so that scan position 0 maps to raster position 1 in the
/// zigzag order, skipping the DC position).
fn decode_cabac_residual(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    block: &mut [i16],
    cat: usize,
    max_coeff: usize,
    scantable: &[usize],
) -> usize {
    let sig_ctx_base = SIGNIFICANT_COEFF_FLAG_OFFSET[cat] as usize;
    let last_ctx_base = LAST_COEFF_FLAG_OFFSET[cat] as usize;
    let abs_level_ctx_base = COEFF_ABS_LEVEL_M1_OFFSET[cat] as usize;
    let is_8x8 = cat == 5;

    // Phase 1: significance map — find which positions have non-zero coefficients.
    //
    // Reference: FFmpeg DECODE_SIGNIFICANCE macro (h264_cabac.c:1669-1683).
    // For cat=5 (8x8 luma), context offsets are remapped via
    // SIGNIFICANT_COEFF_FLAG_OFFSET_8X8 and LAST_COEFF_FLAG_OFFSET_8X8.
    let mut index = [0usize; 64];
    let mut coeff_count = 0usize;
    let mut terminated = false;

    for last in 0..max_coeff.saturating_sub(1) {
        let sig_ctx = if is_8x8 {
            sig_ctx_base + SIGNIFICANT_COEFF_FLAG_OFFSET_8X8[last] as usize
        } else {
            sig_ctx_base + last
        };
        if reader.get_cabac(&mut state[sig_ctx]) != 0 {
            index[coeff_count] = last;
            coeff_count += 1;
            let last_ctx = if is_8x8 {
                last_ctx_base + LAST_COEFF_FLAG_OFFSET_8X8[last] as usize
            } else {
                last_ctx_base + last
            };
            if reader.get_cabac(&mut state[last_ctx]) != 0 {
                terminated = true;
                break;
            }
        }
    }
    if !terminated && max_coeff > 0 {
        // The last position (max_coeff - 1) is implicitly significant
        index[coeff_count] = max_coeff - 1;
        coeff_count += 1;
    }

    if coeff_count == 0 {
        return 0;
    }

    // Phase 2: decode coefficient levels (in reverse order).
    // Apply the scan table to convert significance-map positions to raster
    // positions within the block, matching FFmpeg's `j = scantable[index[...]]`.
    let mut node_ctx = 0usize;

    for i in (0..coeff_count).rev() {
        let raster_pos = scantable[index[i]];
        let ctx_idx = COEFF_ABS_LEVEL1_CTX[node_ctx] as usize + abs_level_ctx_base;

        let coeff_abs;
        if reader.get_cabac(&mut state[ctx_idx]) == 0 {
            // |coeff| == 1
            node_ctx = COEFF_ABS_LEVEL_TRANSITION[0][node_ctx] as usize;
            coeff_abs = 1;
        } else {
            // |coeff| >= 2
            let ctx_gt1 = COEFF_ABS_LEVELGT1_CTX[0][node_ctx] as usize + abs_level_ctx_base;
            node_ctx = COEFF_ABS_LEVEL_TRANSITION[1][node_ctx] as usize;

            let mut abs_val = 2u32;
            while abs_val < 15 && reader.get_cabac(&mut state[ctx_gt1]) != 0 {
                abs_val += 1;
            }

            if abs_val >= 15 {
                // Exp-golomb suffix
                let mut j = 0u32;
                while reader.get_cabac_bypass() != 0 && j < 23 {
                    j += 1;
                }
                let mut val = 1u32;
                for _k in (0..j).rev() {
                    val = val * 2 + reader.get_cabac_bypass() as u32;
                }
                abs_val = val + 14;
            }
            coeff_abs = abs_val;
        }

        // Decode sign via bypass
        let signed_val = reader.get_cabac_bypass_sign(-(coeff_abs as i32));
        block[raster_pos] = signed_val as i16;
    }

    coeff_count
}

// ---------------------------------------------------------------------------
// CABAC luma residual decode
// ---------------------------------------------------------------------------

/// Decode luma residual coefficients for a macroblock using CABAC.
///
/// Handles both I16x16 (DC + AC) and non-I16x16 (4x4 blocks per 8x8).
///
/// Reference: FFmpeg h264_cabac.c:1870-1918 (decode_cabac_luma_residual).
#[allow(clippy::too_many_arguments)]
fn decode_cabac_luma_residual(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    mb: &mut Macroblock,
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_idx: usize,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    cbp: u32,
    is_intra16x16: bool,
    is_intra: bool,
) {
    let nz_cache = &mut mb.non_zero_count;

    if is_intra16x16 {
        // Luma DC (cat=0): 16 coefficients
        let cbf_ctx = get_cabac_cbf_ctx(
            cabac_nb,
            slice_table,
            cur_slice,
            mb_idx,
            mb_x,
            mb_y,
            mb_width,
            0,
            0,
            true,
            nz_cache,
            is_intra,
        );
        if reader.get_cabac(&mut state[cbf_ctx]) != 0 {
            // Store DC coded flag in cbp (bit 8)
            // (We'll set this in the caller's cbp tracking)
            let nz = decode_cabac_residual(reader, state, &mut mb.luma_dc, 0, 16, &ZIGZAG_SCAN_4X4);
            let _ = nz; // nz used for DC coded flag
        }

        // Luma AC (cat=1): 15 coefficients per 4x4 block, only if cbp luma != 0
        if cbp & 15 != 0 {
            for &raster_idx in &SCAN_TO_RASTER {
                let cbf_ctx = get_cabac_cbf_ctx(
                    cabac_nb,
                    slice_table,
                    cur_slice,
                    mb_idx,
                    mb_x,
                    mb_y,
                    mb_width,
                    1,
                    raster_idx,
                    false,
                    nz_cache,
                    is_intra,
                );
                if reader.get_cabac(&mut state[cbf_ctx]) != 0 {
                    let nz = decode_cabac_residual(
                        reader,
                        state,
                        &mut mb.luma_coeffs[raster_idx],
                        1,
                        15,
                        &ZIGZAG_SCAN_4X4[1..],
                    );
                    nz_cache[raster_idx] = nz as u8;
                }
            }
        }
    } else if mb.transform_size_8x8_flag {
        // 8x8 transform (High profile CABAC): decode 64 coefficients per 8x8 block.
        // Uses cat=5 with CABAC-specific context remapping.
        // Reference: FFmpeg h264_cabac.c:1898-1901
        for i8x8 in 0..4 {
            if cbp & (1 << i8x8) != 0 {
                // CBF context: use the first 4x4 sub-block's position
                let raster_idx = SCAN_TO_RASTER[i8x8 * 4];
                let cbf_ctx = get_cabac_cbf_ctx(
                    cabac_nb,
                    slice_table,
                    cur_slice,
                    mb_idx,
                    mb_x,
                    mb_y,
                    mb_width,
                    5,
                    raster_idx,
                    false,
                    nz_cache,
                    is_intra,
                );
                if reader.get_cabac(&mut state[cbf_ctx]) != 0 {
                    let nz = decode_cabac_residual(
                        reader,
                        state,
                        &mut mb.luma_8x8_coeffs[i8x8],
                        5,
                        64,
                        &ZIGZAG_SCAN_8X8_USIZE,
                    );
                    // Broadcast NNZ to all 4 sub-blocks for deblocking/neighbor context.
                    let nz_val = (nz as u8).min(16);
                    for k in 0..4 {
                        nz_cache[SCAN_TO_RASTER[i8x8 * 4 + k]] = nz_val;
                    }
                }
            }
        }
    } else {
        // Non-I16x16: 4x4 blocks grouped by 8x8
        for i8x8 in 0..4 {
            if cbp & (1 << i8x8) != 0 {
                // Decode 4 4x4 blocks within this 8x8 block
                for i4x4 in 0..4 {
                    let scan_idx = i8x8 * 4 + i4x4;
                    let raster_idx = SCAN_TO_RASTER[scan_idx];
                    let cbf_ctx = get_cabac_cbf_ctx(
                        cabac_nb,
                        slice_table,
                        cur_slice,
                        mb_idx,
                        mb_x,
                        mb_y,
                        mb_width,
                        2,
                        raster_idx,
                        false,
                        nz_cache,
                        is_intra,
                    );
                    if reader.get_cabac(&mut state[cbf_ctx]) != 0 {
                        let nz = decode_cabac_residual(
                            reader,
                            state,
                            &mut mb.luma_coeffs[raster_idx],
                            2,
                            16,
                            &ZIGZAG_SCAN_4X4,
                        );
                        nz_cache[raster_idx] = nz as u8;
                    }
                }
            }
        }
    }
}

/// Decode chroma residual coefficients (DC + AC) using CABAC.
///
/// Reference: FFmpeg h264_cabac.c:2466-2487 (yuv420 chroma decode).
#[allow(clippy::too_many_arguments)]
fn decode_cabac_chroma_residual(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    mb: &mut Macroblock,
    cabac_nb: &CabacNeighborCtx,
    slice_table: &[u16],
    cur_slice: u16,
    mb_idx: usize,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    cbp: u32,
    stored_cbp: &mut u32,
    is_intra: bool,
) {
    let nz_cache = &mut mb.non_zero_count;

    // Chroma DC (cat=3): 4 coefficients per plane
    if cbp & 0x30 != 0 {
        for c in 0..2u32 {
            let cbf_ctx = get_cabac_cbf_ctx(
                cabac_nb,
                slice_table,
                cur_slice,
                mb_idx,
                mb_x,
                mb_y,
                mb_width,
                3,
                c as usize,
                true,
                nz_cache,
                is_intra,
            );
            if reader.get_cabac(&mut state[cbf_ctx]) != 0 {
                // Chroma DC 2x2 uses identity scan (positions map 1:1 to the
                // chroma_dc[0..4] array, matching CAVLC's direct storage).
                const CHROMA_DC_SCAN: [usize; 4] = [0, 1, 2, 3];
                decode_cabac_residual(
                    reader,
                    state,
                    &mut mb.chroma_dc[c as usize],
                    3,
                    4,
                    &CHROMA_DC_SCAN,
                );
                // Set chroma DC coded flag in cbp (bits 6..7)
                *stored_cbp |= 0x40 << c;
            }
        }
    }

    // Chroma AC (cat=4): 15 coefficients per 4x4 block
    if cbp & 0x20 != 0 {
        for c in 0..2usize {
            for i in 0..4usize {
                let _block_idx = 16 + 4 * c + i;
                let nz_idx = 16 + 4 * c + i;
                let cbf_ctx = get_cabac_cbf_ctx(
                    cabac_nb,
                    slice_table,
                    cur_slice,
                    mb_idx,
                    mb_x,
                    mb_y,
                    mb_width,
                    4,
                    nz_idx,
                    false,
                    nz_cache,
                    is_intra,
                );
                if reader.get_cabac(&mut state[cbf_ctx]) != 0 {
                    let nz = decode_cabac_residual(
                        reader,
                        state,
                        &mut mb.chroma_ac[c][i],
                        4,
                        15,
                        &ZIGZAG_SCAN_4X4[1..],
                    );
                    nz_cache[nz_idx] = nz as u8;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main CABAC macroblock decode function
// ---------------------------------------------------------------------------

/// Decode one macroblock using CABAC, producing a `Macroblock` struct
/// identical to what CAVLC produces.
///
/// Reference: FFmpeg h264_cabac.c:1920-2499 (ff_h264_decode_mb_cabac).
#[allow(clippy::too_many_arguments)]
pub fn decode_mb_cabac(
    reader: &mut CabacReader,
    state: &mut [u8; 1024],
    slice_type: SliceType,
    _pps: &Pps,
    cabac_nb: &CabacNeighborCtx,
    neighbor: &NeighborContext,
    slice_table: &[u16],
    cur_slice: u16,
    mb_x: u32,
    mb_y: u32,
    mb_width: u32,
    num_ref_idx_l0_active: u32,
    num_ref_idx_l1_active: u32,
    last_qscale_diff: i32,
    cache: &mut CabacDecodeCache,
    direct_8x8_inference_flag: bool,
) -> Result<Macroblock> {
    let mut mb = Macroblock::default();
    let mb_idx = (mb_y * mb_width + mb_x) as usize;

    trace!(
        "CABAC_MB_START mb_x={} mb_y={} slice_type={:?}",
        mb_x, mb_y, slice_type
    );

    // 1. Parse mb_type
    let is_intra;
    let mut cbp;

    match slice_type {
        SliceType::I | SliceType::SI => {
            let raw_mt = decode_cabac_intra_mb_type(
                reader,
                state,
                3,
                true,
                cabac_nb,
                slice_table,
                cur_slice,
                mb_idx,
                mb_x,
                mb_y,
                mb_width,
            );
            is_intra = true;
            decode_intra_mb_cabac(&mut mb, raw_mt)?;
            cbp = mb.cbp;
        }
        SliceType::P | SliceType::SP => {
            // P-slice: first check if inter or intra
            if reader.get_cabac(&mut state[14]) == 0 {
                // P-type inter
                is_intra = false;
                if reader.get_cabac(&mut state[15]) == 0 {
                    // P_L0_16x16 or P_8x8
                    mb.mb_type = 3 * reader.get_cabac(&mut state[16]) as u32;
                } else {
                    // P_L0_8x16 or P_L0_16x8
                    mb.mb_type = 2 - reader.get_cabac(&mut state[17]) as u32;
                }
                mb.partition_count = [1, 2, 2, 4, 4][mb.mb_type as usize];
                cbp = 0;
            } else {
                // Intra MB in P-slice
                let raw_mt = decode_cabac_intra_mb_type(
                    reader,
                    state,
                    17,
                    false,
                    cabac_nb,
                    slice_table,
                    cur_slice,
                    mb_idx,
                    mb_x,
                    mb_y,
                    mb_width,
                );
                is_intra = true;
                decode_intra_mb_cabac(&mut mb, raw_mt)?;
                cbp = mb.cbp;
            }
        }
        SliceType::B => {
            // B-slice mb_type decode
            let mut b_ctx = 0usize;
            // Left neighbor: not direct
            if mb_x > 0 && slice_table[mb_idx - 1] == cur_slice {
                let left_idx = mb_idx - 1;
                if !cabac_nb.mb_direct[left_idx] && !cabac_nb.mb_skip[left_idx] {
                    b_ctx += 1;
                }
            }
            // Top neighbor: not direct
            if mb_y > 0 && slice_table[mb_idx - mb_width as usize] == cur_slice {
                let top_idx = mb_idx - mb_width as usize;
                if !cabac_nb.mb_direct[top_idx] && !cabac_nb.mb_skip[top_idx] {
                    b_ctx += 1;
                }
            }

            trace!("CABAC_B_MB_TYPE b_ctx={} state_idx={}", b_ctx, 27 + b_ctx);

            if reader.get_cabac(&mut state[27 + b_ctx]) == 0 {
                // B_Direct_16x16
                is_intra = false;
                mb.mb_type = 0;
                mb.is_direct = true;
                let info = &B_MB_TYPE_INFO[0];
                mb.partition_count = info.0;
                mb.b_part_size = info.1;
                mb.b_list_flags = info.2;
                cbp = 0;
            } else if reader.get_cabac(&mut state[27 + 3]) == 0 {
                // B_L0_16x16 or B_L1_16x16
                is_intra = false;
                mb.mb_type = 1 + reader.get_cabac(&mut state[27 + 5]) as u32;
                let info = &B_MB_TYPE_INFO[mb.mb_type as usize];
                mb.partition_count = info.0;
                mb.b_part_size = info.1;
                mb.b_list_flags = info.2;
                cbp = 0;
            } else {
                let mut bits = (reader.get_cabac(&mut state[27 + 4]) as u32) << 3;
                bits += (reader.get_cabac(&mut state[27 + 5]) as u32) << 2;
                bits += (reader.get_cabac(&mut state[27 + 5]) as u32) << 1;
                bits += reader.get_cabac(&mut state[27 + 5]) as u32;

                if bits < 8 {
                    is_intra = false;
                    mb.mb_type = bits + 3;
                    let info = &B_MB_TYPE_INFO[mb.mb_type as usize];
                    mb.partition_count = info.0;
                    mb.b_part_size = info.1;
                    mb.b_list_flags = info.2;
                    cbp = 0;
                } else if bits == 13 {
                    // Intra MB in B-slice
                    let raw_mt = decode_cabac_intra_mb_type(
                        reader,
                        state,
                        32,
                        false,
                        cabac_nb,
                        slice_table,
                        cur_slice,
                        mb_idx,
                        mb_x,
                        mb_y,
                        mb_width,
                    );
                    is_intra = true;
                    decode_intra_mb_cabac(&mut mb, raw_mt)?;
                    cbp = mb.cbp;
                } else if bits == 14 {
                    is_intra = false;
                    mb.mb_type = 11; // B_L1_L0_8x16
                    let info = &B_MB_TYPE_INFO[11];
                    mb.partition_count = info.0;
                    mb.b_part_size = info.1;
                    mb.b_list_flags = info.2;
                    cbp = 0;
                } else if bits == 15 {
                    is_intra = false;
                    mb.mb_type = 22; // B_8x8
                    let info = &B_MB_TYPE_INFO[22];
                    mb.partition_count = info.0;
                    mb.b_part_size = info.1;
                    mb.b_list_flags = info.2;
                    cbp = 0;
                } else {
                    is_intra = false;
                    let ext_bits = (bits << 1) + reader.get_cabac(&mut state[27 + 5]) as u32;
                    mb.mb_type = ext_bits - 4;
                    let info = &B_MB_TYPE_INFO[mb.mb_type as usize];
                    mb.partition_count = info.0;
                    mb.b_part_size = info.1;
                    mb.b_list_flags = info.2;
                    cbp = 0;
                }
            }
        }
    }
    mb.is_intra = is_intra;

    // 2. Handle I_PCM
    if mb.is_pcm {
        // I_PCM in CABAC: recover byte position from the arithmetic engine,
        // read 384 raw sample bytes (256 luma + 64 Cb + 64 Cr for 8-bit 4:2:0),
        // then re-initialize the CABAC engine.
        //
        // Reference: FFmpeg h264_cabac.c:2035-2069.
        //
        // We use skip_bytes which: (1) recovers the true byte position from
        // the CABAC engine state, (2) skips N bytes, (3) re-inits the engine.
        // First, recover the byte position and read the raw PCM bytes.
        let mb_size = 384usize; // 8-bit 4:2:0: 256 + 64 + 64

        // Recover actual byte position (engine may have read ahead)
        let mut ptr = reader.pos();
        if reader.low_bit0() {
            ptr -= 1;
        }
        if reader.low_bits9() {
            ptr -= 1;
        }

        // Read raw samples from the recovered position
        let pcm_data = reader.data();
        if ptr + mb_size > pcm_data.len() {
            return Err(Error::InvalidData);
        }

        let mut byte_pos = ptr;
        // Read 256 luma samples in raster order
        for y in 0..16u32 {
            for x in 0..16u32 {
                let blk = ((y / 4) * 4 + (x / 4)) as usize;
                let sub = ((y % 4) * 4 + (x % 4)) as usize;
                mb.luma_coeffs[blk][sub] = pcm_data[byte_pos] as i16;
                byte_pos += 1;
            }
        }
        // Read 64 Cb then 64 Cr
        for plane_idx in 0..2usize {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let blk = ((y / 4) * 2 + (x / 4)) as usize;
                    let sub = ((y % 4) * 4 + (x % 4)) as usize;
                    mb.chroma_ac[plane_idx][blk][sub] = pcm_data[byte_pos] as i16;
                    byte_pos += 1;
                }
            }
        }

        // Re-init CABAC engine from after the PCM data
        reader.skip_bytes(mb_size)?;

        mb.non_zero_count = [16; 24];
        mb.mb_qp_delta = 0;
        // I_PCM: all blocks are coded. Set CBP so neighbor context derivation
        // sees correct values. Matches FFmpeg's h->cbp_table[mb_xy] = 0xf7ef.
        // Bits: luma=0xF, chroma=2<<4, Cb DC=0x40, Cr DC=0x80, luma DC=0x100.
        mb.cbp = 0x1EF;
        return Ok(mb);
    }

    let mut dct8x8_allowed = _pps.transform_8x8_mode;

    // 3. Intra prediction modes
    trace!("CABAC_SECTION intra_pred_modes");

    // Compute neighbor_transform_size for CABAC context 399.
    // Reference: FFmpeg h264_mvpred.h:928
    let neighbor_ts = {
        let mut ts = 0usize;
        // Top neighbor
        if mb_y > 0
            && slice_table[mb_idx - mb_width as usize] == cur_slice
            && cabac_nb.transform_8x8[mb_idx - mb_width as usize]
        {
            ts += 1;
        }
        // Left neighbor
        if mb_x > 0 && slice_table[mb_idx - 1] == cur_slice && cabac_nb.transform_8x8[mb_idx - 1] {
            ts += 1;
        }
        ts
    };

    if mb.is_intra4x4 {
        // High profile: decode transform_size_8x8_flag via CABAC context 399.
        // Reference: FFmpeg h264_cabac.c:2077
        if _pps.transform_8x8_mode {
            mb.transform_size_8x8_flag = reader.get_cabac(&mut state[399 + neighbor_ts]) != 0;
        }

        // Decode intra 4x4 prediction modes.
        // When 8x8 transform: decode 4 modes and broadcast to 2x2 sub-blocks.
        const DC_PRED: u8 = 2;
        let di = if mb.transform_size_8x8_flag { 4 } else { 1 };
        let mut mode_cache = [-1i8; 16]; // raster order

        let mut scan_idx = 0;
        while scan_idx < 16 {
            let raster_idx = SCAN_TO_RASTER[scan_idx];
            let blk_x = raster_idx % 4;
            let blk_y = raster_idx / 4;

            let left_mode: i8 = if blk_x > 0 {
                mode_cache[raster_idx - 1]
            } else if neighbor.left_available {
                neighbor.left_intra4x4_mode[blk_y]
            } else {
                -1
            };

            let top_mode: i8 = if blk_y > 0 {
                mode_cache[raster_idx - 4]
            } else if neighbor.top_available {
                let abs_blk_x = mb_x as usize * 4 + blk_x;
                neighbor.top_intra4x4_mode[abs_blk_x]
            } else {
                -1
            };

            let predicted = if left_mode < 0 || top_mode < 0 {
                DC_PRED
            } else {
                (left_mode as u8).min(top_mode as u8)
            };

            let mode = decode_cabac_mb_intra4x4_pred_mode(reader, state, predicted);

            if di == 4 {
                for k in 0..4 {
                    let r = SCAN_TO_RASTER[scan_idx + k];
                    mode_cache[r] = mode as i8;
                    mb.intra4x4_pred_mode[r] = mode;
                }
            } else {
                mode_cache[raster_idx] = mode as i8;
                mb.intra4x4_pred_mode[raster_idx] = mode;
            }

            scan_idx += di;
        }
    }

    // 4. Chroma prediction mode (for intra MBs)
    trace!("CABAC_SECTION chroma_pred_mode");
    if is_intra {
        mb.chroma_pred_mode = decode_cabac_mb_chroma_pre_mode(
            reader,
            state,
            cabac_nb,
            slice_table,
            cur_slice,
            mb_idx,
            mb_x,
            mb_y,
            mb_width,
        );
    }

    // 5. Inter prediction — fill scan8 cache and decode ref/mvd.
    // The cache handles both cross-MB and intra-MB neighbor lookups via scan8.
    if !is_intra {
        cache.fill(
            cabac_nb,
            slice_table,
            cur_slice,
            mb_idx,
            mb_x,
            mb_y,
            mb_width,
        );
    }

    if !is_intra && (slice_type == SliceType::P || slice_type == SliceType::SP) {
        match mb.mb_type {
            0 => {
                // P_L0_16x16: one ref, one mvd
                if num_ref_idx_l0_active > 1 {
                    mb.ref_idx_l0[0] = decode_cabac_mb_ref(
                        reader,
                        state,
                        cache,
                        cabac_nb,
                        slice_table,
                        cur_slice,
                        mb_idx,
                        mb_x,
                        mb_y,
                        mb_width,
                        0,
                        SCAN8[0],
                        false,
                    ) as i8;
                } else {
                    mb.ref_idx_l0[0] = 0;
                }
                fill_rectangle(&mut cache.ref_cache[0], SCAN8[0], 4, 4, 8, mb.ref_idx_l0[0]);
                let (mx, my, ax, ay) = decode_cabac_mb_mvd(reader, state, cache, 0, SCAN8[0]);
                mb.mvd_l0[0] = [mx, my];
                fill_rectangle(&mut cache.mvd_cache[0], SCAN8[0], 4, 4, 8, [ax, ay]);
            }
            1 => {
                // P_L0_L0_16x8
                for part in 0..2u32 {
                    let scan8_n = SCAN8[part as usize * 8];
                    if num_ref_idx_l0_active > 1 {
                        mb.ref_idx_l0[part as usize] = decode_cabac_mb_ref(
                            reader,
                            state,
                            cache,
                            cabac_nb,
                            slice_table,
                            cur_slice,
                            mb_idx,
                            mb_x,
                            mb_y,
                            mb_width,
                            0,
                            scan8_n,
                            false,
                        ) as i8;
                    } else {
                        mb.ref_idx_l0[part as usize] = 0;
                    }
                    fill_rectangle(
                        &mut cache.ref_cache[0],
                        scan8_n,
                        4,
                        2,
                        8,
                        mb.ref_idx_l0[part as usize],
                    );
                }
                for part in 0..2u32 {
                    let scan8_n = SCAN8[part as usize * 8];
                    let (mx, my, ax, ay) = decode_cabac_mb_mvd(reader, state, cache, 0, scan8_n);
                    mb.mvd_l0[part as usize] = [mx, my];
                    fill_rectangle(&mut cache.mvd_cache[0], scan8_n, 4, 2, 8, [ax, ay]);
                }
            }
            2 => {
                // P_L0_L0_8x16
                for part in 0..2u32 {
                    let scan8_n = SCAN8[part as usize * 4];
                    if num_ref_idx_l0_active > 1 {
                        mb.ref_idx_l0[part as usize] = decode_cabac_mb_ref(
                            reader,
                            state,
                            cache,
                            cabac_nb,
                            slice_table,
                            cur_slice,
                            mb_idx,
                            mb_x,
                            mb_y,
                            mb_width,
                            0,
                            scan8_n,
                            false,
                        ) as i8;
                    } else {
                        mb.ref_idx_l0[part as usize] = 0;
                    }
                    fill_rectangle(
                        &mut cache.ref_cache[0],
                        scan8_n,
                        2,
                        4,
                        8,
                        mb.ref_idx_l0[part as usize],
                    );
                }
                for part in 0..2u32 {
                    let scan8_n = SCAN8[part as usize * 4];
                    let (mx, my, ax, ay) = decode_cabac_mb_mvd(reader, state, cache, 0, scan8_n);
                    mb.mvd_l0[part as usize] = [mx, my];
                    fill_rectangle(&mut cache.mvd_cache[0], scan8_n, 2, 4, 8, [ax, ay]);
                }
            }
            3 | 4 => {
                // P_8x8 / P_8x8ref0
                for i in 0..4 {
                    mb.sub_mb_type[i] = decode_cabac_p_mb_sub_type(reader, state);
                }
                // Restrict dct8x8_allowed: only 8x8 sub-partitions (raw type 0) allow 8x8 DCT.
                // Reference: FFmpeg h264_mvpred.h:157 get_dct8x8_allowed()
                if dct8x8_allowed {
                    dct8x8_allowed = mb.sub_mb_type.iter().all(|&t| t == 0);
                }
                let ref_count = if mb.mb_type == 4 {
                    1
                } else {
                    num_ref_idx_l0_active
                };
                for i in 0..4 {
                    if ref_count > 1 {
                        mb.ref_idx_l0[i] = decode_cabac_mb_ref(
                            reader,
                            state,
                            cache,
                            cabac_nb,
                            slice_table,
                            cur_slice,
                            mb_idx,
                            mb_x,
                            mb_y,
                            mb_width,
                            0,
                            SCAN8[i * 4],
                            false,
                        ) as i8;
                    } else {
                        mb.ref_idx_l0[i] = 0;
                    }
                    fill_rectangle(
                        &mut cache.ref_cache[0],
                        SCAN8[i * 4],
                        2,
                        2,
                        8,
                        mb.ref_idx_l0[i],
                    );
                }
                for i in 0..4 {
                    let base_x = (i & 1) as u32 * 2;
                    let base_y = (i >> 1) as u32 * 2;
                    let sub_part_count = P_SUB_MB_PARTITION_COUNT[mb.sub_mb_type[i] as usize];
                    for j in 0..sub_part_count as usize {
                        let mvd_idx = i * 4 + j;
                        if mvd_idx < 16 {
                            let (dx, dy, w, h) = p8x8_sub_blk(mb.sub_mb_type[i] as u32, j);
                            let blk_x = base_x + dx;
                            let blk_y = base_y + dy;
                            let cache_pos = (4 + blk_x as usize) + (1 + blk_y as usize) * 8;
                            let (mx, my, ax, ay) =
                                decode_cabac_mb_mvd(reader, state, cache, 0, cache_pos);
                            mb.mvd_l0[mvd_idx] = [mx, my];
                            fill_rectangle(
                                &mut cache.mvd_cache[0],
                                cache_pos,
                                w as usize,
                                h as usize,
                                8,
                                [ax, ay],
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // 5b. Inter prediction (B-slice, not intra)
    if !is_intra && slice_type == SliceType::B && !mb.is_direct {
        if mb.mb_type == 22 {
            // B_8x8
            for i in 0..4 {
                mb.sub_mb_type[i] = decode_cabac_b_mb_sub_type(reader, state);
            }
            // Restrict dct8x8_allowed for B_8x8: sub-8x8 partitions (raw type >= 4) disallow,
            // and B_Direct_8x8 (raw type 0) requires direct_8x8_inference_flag.
            // Reference: FFmpeg h264_mvpred.h:157 get_dct8x8_allowed()
            if dct8x8_allowed {
                let all_8x8 = mb.sub_mb_type.iter().all(|&t| t <= 3);
                let has_direct = mb.sub_mb_type.contains(&0);
                dct8x8_allowed = all_8x8 && (!has_direct || direct_8x8_inference_flag);
            }
            // ref_idx L0
            for i in 0..4 {
                if mb.sub_mb_type[i] == 0 {
                    continue;
                }
                let info = &B_SUB_MB_TYPE_INFO[mb.sub_mb_type[i] as usize];
                if info.2 && num_ref_idx_l0_active > 1 {
                    mb.ref_idx_l0[i] = decode_cabac_mb_ref(
                        reader,
                        state,
                        cache,
                        cabac_nb,
                        slice_table,
                        cur_slice,
                        mb_idx,
                        mb_x,
                        mb_y,
                        mb_width,
                        0,
                        SCAN8[i * 4],
                        true,
                    ) as i8;
                } else if info.2 {
                    mb.ref_idx_l0[i] = 0;
                }
                fill_rectangle(
                    &mut cache.ref_cache[0],
                    SCAN8[i * 4],
                    2,
                    2,
                    8,
                    mb.ref_idx_l0[i],
                );
            }
            // ref_idx L1
            for i in 0..4 {
                if mb.sub_mb_type[i] == 0 {
                    continue;
                }
                let info = &B_SUB_MB_TYPE_INFO[mb.sub_mb_type[i] as usize];
                if info.3 && num_ref_idx_l1_active > 1 {
                    mb.ref_idx_l1[i] = decode_cabac_mb_ref(
                        reader,
                        state,
                        cache,
                        cabac_nb,
                        slice_table,
                        cur_slice,
                        mb_idx,
                        mb_x,
                        mb_y,
                        mb_width,
                        1,
                        SCAN8[i * 4],
                        true,
                    ) as i8;
                } else if info.3 {
                    mb.ref_idx_l1[i] = 0;
                }
                fill_rectangle(
                    &mut cache.ref_cache[1],
                    SCAN8[i * 4],
                    2,
                    2,
                    8,
                    mb.ref_idx_l1[i],
                );
            }
            // mvd L0
            for i in 0..4 {
                if mb.sub_mb_type[i] == 0 {
                    continue;
                }
                let info = &B_SUB_MB_TYPE_INFO[mb.sub_mb_type[i] as usize];
                if info.2 {
                    let base_x = (i & 1) as u32 * 2;
                    let base_y = (i >> 1) as u32 * 2;
                    for j in 0..info.0 as usize {
                        let mvd_idx = i * 4 + j;
                        if mvd_idx < 16 {
                            let (dx, dy, w, h) = p8x8_sub_blk(info.1 as u32, j);
                            let blk_x = base_x + dx;
                            let blk_y = base_y + dy;
                            let cache_pos = (4 + blk_x as usize) + (1 + blk_y as usize) * 8;
                            let (mx, my, ax, ay) =
                                decode_cabac_mb_mvd(reader, state, cache, 0, cache_pos);
                            mb.mvd_l0[mvd_idx] = [mx, my];
                            fill_rectangle(
                                &mut cache.mvd_cache[0],
                                cache_pos,
                                w as usize,
                                h as usize,
                                8,
                                [ax, ay],
                            );
                        }
                    }
                }
            }
            // mvd L1
            for i in 0..4 {
                if mb.sub_mb_type[i] == 0 {
                    continue;
                }
                let info = &B_SUB_MB_TYPE_INFO[mb.sub_mb_type[i] as usize];
                if info.3 {
                    let base_x = (i & 1) as u32 * 2;
                    let base_y = (i >> 1) as u32 * 2;
                    for j in 0..info.0 as usize {
                        let mvd_idx = i * 4 + j;
                        if mvd_idx < 16 {
                            let (dx, dy, w, h) = p8x8_sub_blk(info.1 as u32, j);
                            let blk_x = base_x + dx;
                            let blk_y = base_y + dy;
                            let cache_pos = (4 + blk_x as usize) + (1 + blk_y as usize) * 8;
                            let (mx, my, ax, ay) =
                                decode_cabac_mb_mvd(reader, state, cache, 1, cache_pos);
                            mb.mvd_l1[mvd_idx] = [mx, my];
                            fill_rectangle(
                                &mut cache.mvd_cache[1],
                                cache_pos,
                                w as usize,
                                h as usize,
                                8,
                                [ax, ay],
                            );
                        }
                    }
                }
            }
        } else {
            // Non-8x8 B-slice partitions
            let part_count = mb.partition_count as usize;
            // Scan8 positions for B partitions
            let ref_scan8 = |p: usize| -> usize {
                match mb.b_part_size {
                    1 => SCAN8[p * 8], // 16x8
                    2 => SCAN8[p * 4], // 8x16
                    _ => SCAN8[0],     // 16x16
                }
            };
            let part_dims = |_p: usize| -> (usize, usize) {
                match mb.b_part_size {
                    1 => (4, 2), // 16x8
                    2 => (2, 4), // 8x16
                    _ => (4, 4), // 16x16
                }
            };

            // ref_idx L0
            for p in 0..part_count {
                if mb.b_list_flags[p][0] {
                    let scan8_n = ref_scan8(p);
                    if num_ref_idx_l0_active > 1 {
                        mb.ref_idx_l0[p] = decode_cabac_mb_ref(
                            reader,
                            state,
                            cache,
                            cabac_nb,
                            slice_table,
                            cur_slice,
                            mb_idx,
                            mb_x,
                            mb_y,
                            mb_width,
                            0,
                            scan8_n,
                            true,
                        ) as i8;
                    } else {
                        mb.ref_idx_l0[p] = 0;
                    }
                    let (w, h) = part_dims(p);
                    fill_rectangle(&mut cache.ref_cache[0], scan8_n, w, h, 8, mb.ref_idx_l0[p]);
                }
            }
            // ref_idx L1
            for p in 0..part_count {
                if mb.b_list_flags[p][1] {
                    let scan8_n = ref_scan8(p);
                    if num_ref_idx_l1_active > 1 {
                        mb.ref_idx_l1[p] = decode_cabac_mb_ref(
                            reader,
                            state,
                            cache,
                            cabac_nb,
                            slice_table,
                            cur_slice,
                            mb_idx,
                            mb_x,
                            mb_y,
                            mb_width,
                            1,
                            scan8_n,
                            true,
                        ) as i8;
                    } else {
                        mb.ref_idx_l1[p] = 0;
                    }
                    let (w, h) = part_dims(p);
                    fill_rectangle(&mut cache.ref_cache[1], scan8_n, w, h, 8, mb.ref_idx_l1[p]);
                }
            }
            // mvd L0
            for p in 0..part_count {
                if mb.b_list_flags[p][0] {
                    let scan8_n = ref_scan8(p);
                    let (w, h) = part_dims(p);
                    let (mx, my, ax, ay) = decode_cabac_mb_mvd(reader, state, cache, 0, scan8_n);
                    mb.mvd_l0[p] = [mx, my];
                    fill_rectangle(&mut cache.mvd_cache[0], scan8_n, w, h, 8, [ax, ay]);
                }
            }
            // mvd L1
            for p in 0..part_count {
                if mb.b_list_flags[p][1] {
                    let scan8_n = ref_scan8(p);
                    let (w, h) = part_dims(p);
                    let (mx, my, ax, ay) = decode_cabac_mb_mvd(reader, state, cache, 1, scan8_n);
                    mb.mvd_l1[p] = [mx, my];
                    fill_rectangle(&mut cache.mvd_cache[1], scan8_n, w, h, 8, [ax, ay]);
                }
            }
        }
    }

    // 6. CBP (for non-I16x16)
    trace!("CABAC_SECTION cbp");
    if !mb.is_intra16x16 {
        let left_cbp = cabac_nb.left_cbp(mb_idx, mb_x, slice_table, cur_slice, is_intra);
        let top_cbp = cabac_nb.top_cbp(mb_idx, mb_y, mb_width, slice_table, cur_slice, is_intra);
        cbp = decode_cabac_mb_cbp_luma(reader, state, left_cbp, top_cbp);
        cbp |= decode_cabac_mb_cbp_chroma(reader, state, left_cbp, top_cbp) << 4;
    }
    mb.cbp = cbp;

    // B_Direct_16x16: restrict 8x8 DCT unless direct_8x8_inference_flag.
    // Reference: FFmpeg h264_cabac.c (same pattern as CAVLC line 917)
    if mb.is_direct && !direct_8x8_inference_flag {
        dct8x8_allowed = false;
    }

    // 6b. Inter transform_size_8x8_flag (after CBP, before QP delta).
    // Reference: FFmpeg h264_cabac.c:2347-2348
    if dct8x8_allowed && (cbp & 15) != 0 && !is_intra {
        mb.transform_size_8x8_flag = reader.get_cabac(&mut state[399 + neighbor_ts]) != 0;
    }

    // 7. QP delta and residual coefficients
    trace!("CABAC_SECTION qp_delta_and_residual cbp={}", cbp);
    let mut stored_cbp = cbp;
    let is_i16x16 = mb.is_intra16x16;
    if cbp > 0 || is_i16x16 {
        // Decode QP delta
        mb.mb_qp_delta = decode_cabac_mb_dqp(reader, state, last_qscale_diff)?;

        // Decode luma residual
        trace!("CABAC_SECTION luma_residual");
        decode_cabac_luma_residual(
            reader,
            state,
            &mut mb,
            cabac_nb,
            slice_table,
            cur_slice,
            mb_idx,
            mb_x,
            mb_y,
            mb_width,
            cbp,
            is_i16x16,
            is_intra,
        );

        // Decode chroma residual (4:2:0)
        trace!("CABAC_SECTION chroma_residual");
        decode_cabac_chroma_residual(
            reader,
            state,
            &mut mb,
            cabac_nb,
            slice_table,
            cur_slice,
            mb_idx,
            mb_x,
            mb_y,
            mb_width,
            cbp,
            &mut stored_cbp,
            is_intra,
        );

        // Store the luma DC coded flag if I16x16 had non-zero DC
        if is_i16x16 && mb.luma_dc.iter().any(|&c| c != 0) {
            stored_cbp |= 0x100;
        }
    }

    // Update stored cbp
    mb.cbp = stored_cbp;

    trace!(
        mb_x,
        mb_y,
        mb_type = mb.mb_type,
        cbp = mb.cbp,
        is_intra4x4 = mb.is_intra4x4,
        is_intra16x16 = mb.is_intra16x16,
        is_pcm = mb.is_pcm,
        "CABAC decoded MB"
    );

    Ok(mb)
}

/// Decode intra mb_type fields into the Macroblock struct.
fn decode_intra_mb_cabac(mb: &mut Macroblock, raw_mt: u32) -> Result<()> {
    if raw_mt > 25 {
        return Err(Error::InvalidData);
    }

    if raw_mt == 0 {
        mb.is_intra4x4 = true;
        mb.is_intra = true;
    } else if raw_mt == 25 {
        mb.is_pcm = true;
        mb.is_intra = true;
    } else {
        mb.is_intra16x16 = true;
        mb.is_intra = true;
        let info = I_MB_TYPE_INFO[raw_mt as usize];
        mb.intra16x16_mode = info.0 as u8;
        mb.cbp = info.1 as u32;
    }
    mb.mb_type = raw_mt;
    mb.partition_count = 0;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cabac_init_zero() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let reader = CabacReader::new(&data).unwrap();
        assert_eq!(reader.range, 0x1FE);
        assert_eq!(reader.pos, 2);
        assert_eq!(reader.low, 1 << 9); // 0<<18 + 0<<10 + (1<<9)
    }

    #[test]
    fn test_cabac_init_nonzero() {
        // Use values small enough to pass the validity check
        let data = [0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00];
        let reader = CabacReader::new(&data).unwrap();
        assert_eq!(reader.range, 0x1FE);
        let expected_low = (0x01 << 18) | (0x02 << 10) | (1 << 9);
        assert_eq!(reader.low, expected_low);
    }

    #[test]
    fn test_cabac_init_too_short() {
        let data = [0x00];
        assert!(CabacReader::new(&data).is_err());
    }

    #[test]
    fn test_cabac_init_invalid_range() {
        // 0xFF bytes cause low > range<<17, which is invalid
        let data = [0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(CabacReader::new(&data).is_err());
    }

    #[test]
    fn test_cabac_terminate() {
        let data = vec![0x00; 32];
        let mut reader = CabacReader::new(&data).unwrap();
        let _ = reader.get_cabac_terminate();
    }

    #[test]
    fn test_cabac_bypass() {
        // 0xAA has low = 0xAA<<18 + 0xAA<<10 + (1<<9)
        // = 0x2AA0000 + 0x2A800 + 0x200 = 0x2ACAA00
        // range<<17 = 0x3FC0000 > 0x2ACAA00, so valid
        let data = vec![0xAA; 32];
        let mut reader = CabacReader::new(&data).unwrap();
        for _ in 0..16 {
            let bit = reader.get_cabac_bypass();
            assert!(bit <= 1);
        }
    }

    #[test]
    fn test_cabac_context_decode() {
        let data = vec![0x55; 32];
        let mut reader = CabacReader::new(&data).unwrap();
        let mut state = 0u8;
        for _ in 0..16 {
            let bit = reader.get_cabac(&mut state);
            assert!(bit <= 1);
        }
    }

    #[test]
    fn test_cabac_bypass_sign() {
        let data = vec![0x20; 32];
        let mut reader = CabacReader::new(&data).unwrap();
        let val = reader.get_cabac_bypass_sign(42);
        assert!(val == 42 || val == -42);
    }
}
