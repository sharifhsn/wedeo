// VP9 / VPX boolean arithmetic (range) decoder.
//
// Translated from FFmpeg's libavcodec/vpx_rac.h, vpx_rac.c, and vp89_rac.h.
// The range coder maintains a probability interval [0, high) encoded in
// a code_word register. Renormalization shifts bits in from the input
// byte-stream to keep `high` in [128, 256).
//
// Key invariants (from vpx_rac.h):
//   - `high` is stored in [0, 255].
//   - `bits` is stored negated: a *negative* value means there are
//     (-bits) bits still available in `code_word` before the next
//     refill from the buffer.
//   - `code_word` holds at most 32 bits of lookahead.
//   - Initialisation: high=255, bits=-16, code_word = first 3 bytes BE.

/// Number-of-leading-zeros lookup: `NORM_SHIFT[x]` = number of leading
/// zero bits in `x` (for x in 0..=255), i.e. how much to left-shift
/// to move the MSB into bit 7.  `NORM_SHIFT[0] = 8` (special case).
///
/// Verbatim copy of `ff_vpx_norm_shift` from vpx_rac.c.
const NORM_SHIFT: [u8; 256] = [
    8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// VP9 / VPX boolean arithmetic decoder.
///
/// Corresponds to `VPXRangeCoder` in FFmpeg, together with the inline
/// functions `vpx_rac_renorm`, `vpx_rac_get_prob`, `vpx_rac_get`,
/// `vp89_rac_get`, and `vp89_rac_get_uint`.
pub struct BoolDecoder<'a> {
    /// Upper end of the current probability interval (stored in [0, 255]).
    high: u32,
    /// Negated count of valid bits remaining in `code_word` before
    /// the next 16-bit refill.  A value of -16 means 16 bits are
    /// available; ≥0 means the buffer is exhausted.
    bits: i32,
    /// Remaining input bytes.
    buffer: &'a [u8],
    /// Current position within `buffer`.
    pos: usize,
    /// Prefetched code word (up to 32 bits of lookahead).
    code_word: u32,
    /// How many times the end of the buffer has been reached.
    end_reached: u32,
}

impl<'a> BoolDecoder<'a> {
    /// Return (high, code_word, bits, pos) for diagnostics.
    pub fn diag_state(&self) -> (u32, u32, i32, usize) {
        (self.high, self.code_word, self.bits, self.pos)
    }
    /// Initialise a `BoolDecoder` from a byte slice.
    ///
    /// Corresponds to `ff_vpx_init_range_decoder` in vpx_rac.c.
    /// Returns `None` if `data` is empty.
    pub fn new(data: &'a [u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        // Consume the first three bytes as the initial code_word (big-endian 24 bits).
        let (code_word, pos) = if data.len() >= 3 {
            let cw = ((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32);
            (cw, 3)
        } else if data.len() == 2 {
            let cw = ((data[0] as u32) << 16) | ((data[1] as u32) << 8);
            (cw, 2)
        } else {
            let cw = (data[0] as u32) << 16;
            (cw, 1)
        };

        Some(Self {
            high: 255,
            bits: -16,
            buffer: data,
            pos,
            code_word,
            end_reached: 0,
        })
    }

    /// Returns `true` once the end of the input stream has been reached.
    ///
    /// Corresponds to `vpx_rac_is_end` in vpx_rac.h.
    #[inline]
    pub fn is_end(&mut self) -> bool {
        if self.pos >= self.buffer.len() && self.bits >= 0 {
            self.end_reached = self.end_reached.saturating_add(1);
        }
        self.end_reached > 10
    }

    /// Renormalise the interval, consuming bytes from the input as needed.
    ///
    /// After renormalisation `high` is in [128, 256) and `code_word`
    /// holds fresh bits at the top of its 32-bit register.
    ///
    /// Corresponds to `vpx_rac_renorm` in vpx_rac.h.
    #[inline]
    fn renorm(&mut self) -> u32 {
        let shift = NORM_SHIFT[self.high as usize] as i32;
        self.high <<= shift;
        let mut code_word = self.code_word << shift;
        self.bits += shift;

        // Refill from the buffer if we have room (bits >= 0 means we need
        // more data — recall bits is stored *negated*, so bits >= 0 means
        // ≤0 bits are left before the next refill is due).
        if self.bits >= 0 && self.pos < self.buffer.len() {
            // Read up to 2 bytes (16 bits) as big-endian.
            let b0 = self.buffer[self.pos] as u32;
            self.pos += 1;
            let b1 = if self.pos < self.buffer.len() {
                let v = self.buffer[self.pos] as u32;
                self.pos += 1;
                v
            } else {
                0
            };
            code_word |= ((b0 << 8) | b1) << self.bits as u32;
            self.bits -= 16;
        }

        self.code_word = code_word;
        code_word
    }

    /// Decode one boolean with the given probability.
    ///
    /// Returns `true` for the "1" / "high" branch.
    /// `prob` is in [0, 255]; higher means "1" is more likely.
    ///
    /// Corresponds to `vpx_rac_get_prob` in vpx_rac.h.
    #[inline]
    pub fn get_prob(&mut self, prob: u8) -> bool {
        let code_word = self.renorm();
        let low = 1 + (((self.high - 1).wrapping_mul(prob as u32)) >> 8);
        let low_shift = low << 16;
        let bit = code_word >= low_shift;
        if bit {
            self.high -= low;
            self.code_word = code_word - low_shift;
        } else {
            self.high = low;
            self.code_word = code_word;
        }
        bit
    }

    /// Decode one equiprobable bit (prob = 128).
    ///
    /// This is the VP8/VP9 variant of the equiprobable read. It differs
    /// slightly from `vpx_rac_get` in rounding — it calls `get_prob(128)`
    /// which corresponds to `vp89_rac_get` in vp89_rac.h.
    #[inline]
    pub fn get(&mut self) -> bool {
        self.get_prob(128)
    }

    /// Decode `bits` literal (equiprobable) bits, MSB first.
    ///
    /// Corresponds to `vp89_rac_get_uint` in vp89_rac.h.
    #[inline]
    pub fn get_uint(&mut self, bits: u32) -> u32 {
        let mut value: u32 = 0;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.get());
        }
        value
    }

    /// Decode a tree-coded symbol.
    ///
    /// `tree` is a table of `[left, right]` pairs; negative values are
    /// leaf encodings (`-leaf_value`). Corresponds to `vp89_rac_get_tree`
    /// in vp89_rac.h.
    pub fn get_tree(&mut self, tree: &[[i8; 2]], probs: &[u8]) -> i32 {
        let mut i: usize = 0;
        loop {
            let branch = self.get_prob(probs[i]);
            let next = tree[i][branch as usize];
            if next <= 0 {
                return (-next) as i32;
            }
            i = next as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_data_returns_none() {
        assert!(BoolDecoder::new(&[]).is_none());
    }

    #[test]
    fn test_single_byte_initialises() {
        let bd = BoolDecoder::new(&[0x80]);
        assert!(bd.is_some());
    }

    #[test]
    fn test_get_uint_all_ones() {
        // A stream of 0xFF bytes should decode all-ones.
        let data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let mut bd = BoolDecoder::new(&data).unwrap();
        // Reading 8 bits: with all-ones data every `get()` should be 1.
        let v = bd.get_uint(8);
        assert_eq!(v, 0xFF);
    }

    #[test]
    fn test_norm_shift_table_spot_checks() {
        // NORM_SHIFT[0] = 8 (special), NORM_SHIFT[128] = 0, NORM_SHIFT[1] = 7
        assert_eq!(NORM_SHIFT[0], 8);
        assert_eq!(NORM_SHIFT[1], 7);
        assert_eq!(NORM_SHIFT[128], 0);
        assert_eq!(NORM_SHIFT[255], 0);
        assert_eq!(NORM_SHIFT[64], 1);
    }
}
