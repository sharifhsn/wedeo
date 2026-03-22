// CABAC (Context-Adaptive Binary Arithmetic Coding) decoder.
//
// Binary arithmetic decoder for H.264 Main/High profile entropy coding.
// Port of FFmpeg libavcodec/cabac.c + cabac_functions.h.
//
// All arithmetic uses i32, matching FFmpeg's branchless tricks that rely
// on `>> 31` sign extension.
//
// Reference: FFmpeg libavcodec/cabac.c, cabac_functions.h, cabac.h

use wedeo_core::error::{Error, Result};

use crate::cabac_tables::{LPS_RANGE, MLPS_STATE, NORM_SHIFT};

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
    /// Reads the first 2-3 bytes to initialize the arithmetic engine.
    /// Reference: FFmpeg `ff_init_cabac_decoder` (CABAC_BITS=16 path).
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 3 {
            return Err(Error::InvalidData);
        }

        // Read first two bytes into low, shifted for 16-bit precision.
        // Then read a third byte for additional precision.
        // This matches FFmpeg's "unaligned" init path which always works
        // correctly regardless of buffer alignment.
        let low = (data[0] as i32) << 18 | (data[1] as i32) << 10 | (data[2] as i32) << 2 | 2;
        let range = 0x1FE;

        // Validity check: range << (CABAC_BITS+1) must be >= low
        if (range << (CABAC_BITS + 1)) < low {
            return Err(Error::InvalidData);
        }

        Ok(Self {
            low,
            range,
            pos: 3,
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
        self.low += self.low;

        if self.low & CABAC_MASK == 0 {
            self.refill();
        }

        let range = self.range << (CABAC_BITS + 1);
        if self.low < range {
            0
        } else {
            self.low -= range;
            1
        }
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
        self.low += self.low;

        if self.low & CABAC_MASK == 0 {
            self.refill();
        }

        let range = self.range << (CABAC_BITS + 1);
        self.low -= range;
        let mask = self.low >> 31;
        self.low += range & mask;
        (val ^ mask) - mask
    }

    /// Check for end-of-slice (terminate symbol).
    ///
    /// Returns true if the slice ends here. The terminate symbol uses
    /// a fixed range reduction of 2.
    ///
    /// Reference: FFmpeg `get_cabac_terminate`.
    #[inline]
    pub fn get_cabac_terminate(&mut self) -> bool {
        self.range -= 2;
        if self.low < (self.range << (CABAC_BITS + 1)) {
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
        }
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
        if new_start + 3 > self.data.len() {
            return Err(Error::InvalidData);
        }

        // Re-init the engine from the new position
        self.low = (self.data[new_start] as i32) << 18
            | (self.data[new_start + 1] as i32) << 10
            | (self.data[new_start + 2] as i32) << 2
            | 2;
        self.range = 0x1FE;
        self.pos = new_start + 3;

        Ok(())
    }

    /// Return the current byte position in the data buffer.
    /// Useful for debugging and I_PCM byte reads.
    pub fn pos(&self) -> usize {
        self.pos
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cabac_init_zero() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let reader = CabacReader::new(&data).unwrap();
        assert_eq!(reader.range, 0x1FE);
        assert_eq!(reader.pos, 3);
        assert_eq!(reader.low, 2); // 0<<18 + 0<<10 + 0<<2 + 2
    }

    #[test]
    fn test_cabac_init_nonzero() {
        // Use values small enough to pass the validity check
        let data = [0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00];
        let reader = CabacReader::new(&data).unwrap();
        assert_eq!(reader.range, 0x1FE);
        let expected_low = (0x01 << 18) | (0x02 << 10) | (0x03 << 2) | 2;
        assert_eq!(reader.low, expected_low);
    }

    #[test]
    fn test_cabac_init_too_short() {
        let data = [0x00, 0x00];
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
        // 0xAA has low = 0xAA<<18 + 0xAA<<10 + 0xAA<<2 + 2
        // = 0x2AA0000 + 0x2A800 + 0x2A8 + 2 = 0x2ACABAA
        // range<<17 = 0x3FC0000 > 0x2ACABAA, so valid
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
