use crate::error::{Error, Result};

/// Fast, 100% safe big-endian bitstream reader.
///
/// Exploits 64-byte SIMD tail padding when constructed via `from_padded`, while
/// remaining completely bounds-checked and panic-free on unpadded or truncated slices.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Logical payload length in bits (excluding any trailing padding).
    bit_len: usize,
    /// Total logical bits consumed so far.
    bits_read: usize,
    /// Next byte index to pull into `cache`.
    byte_pos: usize,
    /// Left-aligned bit buffer (most-significant bit is the next bit to read).
    cache: u64,
    /// Number of valid bits currently in `cache` (0..=64).
    cache_bits: u32,
}

impl<'a> BitReader<'a> {
    /// Create a reader over an exact payload slice (no padding assumed).
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self::from_padded(data, data.len())
    }

    /// Create a reader over a slice that may include trailing padding bytes beyond `logical_len`.
    /// Passing `Buffer::data_with_padding()` with `logical_len = buf.len()` keeps the fast 4-byte
    /// refill path active up to the final bit with zero extra allocations.
    #[inline]
    pub fn from_padded(data: &'a [u8], logical_len: usize) -> Self {
        let logical_len = logical_len.min(data.len());
        let mut r = Self {
            data,
            bit_len: logical_len.saturating_mul(8),
            bits_read: 0,
            byte_pos: 0,
            cache: 0,
            cache_bits: 0,
        };
        r.refill();
        r
    }

    /// Total logical bit count in the payload.
    #[inline]
    pub const fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// Current bit offset from the beginning of the payload.
    #[inline]
    pub const fn position(&self) -> usize {
        self.bits_read
    }

    /// Remaining unread logical bits.
    #[inline]
    pub const fn bits_left(&self) -> usize {
        self.bit_len.saturating_sub(self.bits_read)
    }

    /// Returns `true` if `bits_read` is on an 8-bit boundary.
    #[inline]
    pub const fn is_byte_aligned(&self) -> bool {
        self.bits_read.is_multiple_of(8)
    }

    #[inline]
    fn refill(&mut self) {
        if self.cache_bits <= 32 {
            if let Some(chunk) = self.data.get(self.byte_pos..self.byte_pos + 4) {
                let word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u64;
                self.cache |= word << (32 - self.cache_bits);
                self.byte_pos += 4;
                self.cache_bits += 32;
            } else {
                while self.cache_bits <= 56 && self.byte_pos < self.data.len() {
                    let b = self.data[self.byte_pos] as u64;
                    self.cache |= b << (56 - self.cache_bits);
                    self.byte_pos += 1;
                    self.cache_bits += 8;
                }
            }
        }
    }

    /// Peek `n` bits (`0..=32`) without advancing the cursor. Missing bits past EOF read as 0.
    #[inline]
    pub fn peek_bits(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            return 0;
        }
        if self.cache_bits < n {
            self.refill();
        }
        let val = (self.cache >> (64 - n)) as u32;
        // Mask out bits past logical EOF so padded subslices never leak adjacent bytes.
        let avail = self.bits_left();
        if avail >= n as usize {
            val
        } else if avail == 0 {
            0
        } else {
            let valid_mask = !((1u32 << (n - avail as u32)) - 1);
            val & valid_mask
        }
    }

    /// Advance the cursor by `n` bits. Returns `Err(InvalidData)` if fewer than `n` bits remain.
    #[inline]
    pub fn skip_bits(&mut self, n: usize) -> Result<()> {
        if n > self.bits_left() {
            return Err(Error::invalid_data(
                "bitstream",
                "unexpected end of bitstream",
            ));
        }
        if n <= self.cache_bits as usize {
            let shift = n as u32;
            self.cache = if shift == 64 { 0 } else { self.cache << shift };
            self.cache_bits -= shift;
            self.bits_read += n;
            return Ok(());
        }
        // Large skip: jump directly via byte position and clear cache.
        self.bits_read += n;
        let target_byte = self.bits_read / 8;
        let rem_bits = (self.bits_read % 8) as u32;
        self.byte_pos = target_byte;
        self.cache = 0;
        self.cache_bits = 0;
        self.refill();
        if rem_bits > 0 {
            self.cache <<= rem_bits;
            self.cache_bits = self.cache_bits.saturating_sub(rem_bits);
        }
        Ok(())
    }

    /// Read a single bit (`0` or `1`).
    #[inline]
    pub fn read_bit(&mut self) -> Result<u32> {
        self.read_bits(1)
    }

    /// Read a single bit as `bool`.
    #[inline]
    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Read `n` bits (`0..=32`) big-endian.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n > 32 {
            return Err(Error::invalid_argument("bitstream", "read_bits n > 32"));
        }
        if n == 0 {
            return Ok(0);
        }
        if (n as usize) > self.bits_left() {
            return Err(Error::invalid_data(
                "bitstream",
                "unexpected end of bitstream",
            ));
        }
        if self.cache_bits < n {
            self.refill();
        }
        let val = (self.cache >> (64 - n)) as u32;
        self.cache = if n == 64 { 0 } else { self.cache << n };
        self.cache_bits -= n;
        self.bits_read += n as usize;
        Ok(val)
    }

    /// Skip 0..7 bits to align to the next byte boundary.
    #[inline]
    pub fn align_to_byte(&mut self) -> Result<()> {
        let rem = self.bits_read % 8;
        if rem != 0 {
            self.skip_bits(8 - rem)?;
        }
        Ok(())
    }

    /// Read an unsigned Exp-Golomb code `ue(v)`, matching FFmpeg's `get_ue_golomb_long`.
    pub fn read_ue(&mut self) -> Result<u32> {
        if self.bits_left() == 0 {
            return Err(Error::invalid_data("bitstream", "EOF in ue(v)"));
        }
        let buf = self.peek_bits(32);
        if buf == 0 {
            return Err(Error::invalid_data(
                "bitstream",
                "invalid ue(v): more than 31 leading zeros",
            ));
        }
        let leading_zeros = buf.leading_zeros();
        let total_bits = leading_zeros as usize * 2 + 1;
        if total_bits > self.bits_left() {
            return Err(Error::invalid_data("bitstream", "truncated ue(v)"));
        }
        self.skip_bits(leading_zeros as usize)?;
        let val = self.read_bits(leading_zeros + 1)?;
        Ok(val - 1)
    }

    /// Read a signed Exp-Golomb code `se(v)`, matching FFmpeg's `get_se_golomb_long`
    /// with exact C wrapping arithmetic (panic-free even on `ue == u32::MAX`).
    pub fn read_se(&mut self) -> Result<i32> {
        let buf = self.read_ue()?;
        let sign = (buf & 1).wrapping_sub(1) as i32; // 0 for odd, -1 for even
        Ok((((buf >> 1) as i32) ^ sign).wrapping_add(1))
    }

    /// Read a truncated Exp-Golomb code `te(v)`, matching FFmpeg's `get_te_golomb`.
    /// Returns `Err(InvalidArgument)` when `range == 0` instead of panicking.
    pub fn read_te(&mut self, range: u32) -> Result<u32> {
        if range == 0 {
            return Err(Error::invalid_argument(
                "bitstream",
                "read_te: range must be >= 1",
            ));
        }
        if range == 2 {
            Ok(self.read_bit()? ^ 1)
        } else {
            self.read_ue()
        }
    }

    /// Read a truncated Exp-Golomb code `te(v)` where `range == 1` returns `0` without reading.
    /// Matches FFmpeg's `get_te0_golomb`. Returns `Err(InvalidArgument)` when `range == 0`.
    pub fn read_te0(&mut self, range: u32) -> Result<u32> {
        if range == 0 {
            return Err(Error::invalid_argument(
                "bitstream",
                "read_te0: range must be >= 1",
            ));
        }
        if range == 1 {
            Ok(0)
        } else if range == 2 {
            Ok(self.read_bit()? ^ 1)
        } else {
            self.read_ue()
        }
    }
}

/// Fast, 100% safe little-endian bitstream reader (for VP8/VP9/WAV/Deflate-style bitstreams).
pub struct BitReaderLe<'a> {
    data: &'a [u8],
    bit_len: usize,
    bits_read: usize,
    byte_pos: usize,
    /// Right-aligned bit buffer (least-significant bit is the next bit to read).
    cache: u64,
    cache_bits: u32,
}

impl<'a> BitReaderLe<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self::from_padded(data, data.len())
    }

    #[inline]
    pub fn from_padded(data: &'a [u8], logical_len: usize) -> Self {
        let logical_len = logical_len.min(data.len());
        let mut r = Self {
            data,
            bit_len: logical_len.saturating_mul(8),
            bits_read: 0,
            byte_pos: 0,
            cache: 0,
            cache_bits: 0,
        };
        r.refill();
        r
    }

    #[inline]
    pub const fn bits_left(&self) -> usize {
        self.bit_len.saturating_sub(self.bits_read)
    }

    #[inline]
    pub const fn position(&self) -> usize {
        self.bits_read
    }

    #[inline]
    fn refill(&mut self) {
        if self.cache_bits <= 32 {
            if let Some(chunk) = self.data.get(self.byte_pos..self.byte_pos + 4) {
                let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u64;
                self.cache |= word << self.cache_bits;
                self.byte_pos += 4;
                self.cache_bits += 32;
            } else {
                while self.cache_bits <= 56 && self.byte_pos < self.data.len() {
                    let b = self.data[self.byte_pos] as u64;
                    self.cache |= b << self.cache_bits;
                    self.byte_pos += 1;
                    self.cache_bits += 8;
                }
            }
        }
    }

    #[inline]
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n > 32 {
            return Err(Error::invalid_argument("bitstream", "read_bits n > 32"));
        }
        if n == 0 {
            return Ok(0);
        }
        if (n as usize) > self.bits_left() {
            return Err(Error::invalid_data(
                "bitstream",
                "unexpected end of LE bitstream",
            ));
        }
        if self.cache_bits < n {
            self.refill();
        }
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let val = (self.cache as u32) & mask;
        self.cache >>= n;
        self.cache_bits -= n;
        self.bits_read += n as usize;
        Ok(val)
    }

    #[inline]
    pub fn read_bit(&mut self) -> Result<u32> {
        self.read_bits(1)
    }
}

/// Safe big-endian bitstream writer (for headers, extradata, and encoders).
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    bits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            acc: 0,
            bits: 0,
        }
    }

    /// Write the low `n` bits (`0..=32`) of `val` in big-endian order.
    pub fn write_bits(&mut self, val: u32, n: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        let masked = if n == 32 {
            val as u64
        } else {
            (val as u64) & ((1u64 << n) - 1)
        };
        self.acc = (self.acc << n) | masked;
        self.bits += n;
        while self.bits >= 8 {
            self.bits -= 8;
            self.buf.push((self.acc >> self.bits) as u8);
            if self.bits == 0 {
                self.acc = 0;
            } else {
                self.acc &= (1u64 << self.bits) - 1;
            }
        }
    }

    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(u32::from(bit), 1);
    }

    /// Write an unsigned Exp-Golomb code `ue(v)`.
    pub fn write_ue(&mut self, val: u32) {
        let code = val as u64 + 1;
        let len = 64 - code.leading_zeros();
        let leading_zeros = len - 1;
        if leading_zeros > 0 {
            self.write_bits(0, leading_zeros);
        }
        if len <= 32 {
            self.write_bits(code as u32, len);
        } else {
            self.write_bits((code >> 32) as u32, len - 32);
            self.write_bits(code as u32, 32);
        }
    }

    /// Write a signed Exp-Golomb code `se(v)`.
    pub fn write_se(&mut self, val: i32) {
        let code = if val <= 0 {
            (val.unsigned_abs()).wrapping_mul(2)
        } else {
            (val as u32).wrapping_mul(2).wrapping_sub(1)
        };
        self.write_ue(code);
    }

    /// Pad with 0 bits to the next byte boundary and return the written bytes.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            let pad = 8 - self.bits;
            self.buf.push((self.acc << pad) as u8);
        }
        self.buf
    }

    /// Current total number of written bits.
    #[inline]
    pub fn bit_len(&self) -> usize {
        self.buf.len() * 8 + self.bits as usize
    }
}

/// FFmpeg-compatible Adler-32 (`libavutil/adler32.c`): uses `s1 = 0, s2 = 0`
/// initial state when called with `init = 0` (matching FFmpeg's `framecrc` muxer).
pub fn adler32_ffmpeg(init: u32, data: &[u8]) -> u32 {
    const BASE: u32 = 65521;
    let mut s1 = init & 0xFFFF;
    let mut s2 = init >> 16;

    for chunk in data.chunks(5552) {
        for &b in chunk {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= BASE;
        s2 %= BASE;
    }

    (s2 << 16) | s1
}

/// Standard IEEE 802.3 CRC-32 (polynomial `0xEDB88320` reflected).
pub fn crc32(init: u32, data: &[u8]) -> u32 {
    let mut crc = !init;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitreader_roundtrip_and_golomb() {
        let mut w = BitWriter::new();
        w.write_bits(0b10110, 5);
        for v in [0u32, 1, 2, 3, 7, 42, 1024, 100_000] {
            w.write_ue(v);
        }
        for v in [0i32, 1, -1, 2, -2, 17, -17, 5000, -5000] {
            w.write_se(v);
        }
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(5).unwrap(), 0b10110);
        for v in [0u32, 1, 2, 3, 7, 42, 1024, 100_000] {
            assert_eq!(r.read_ue().unwrap(), v);
        }
        for v in [0i32, 1, -1, 2, -2, 17, -17, 5000, -5000] {
            assert_eq!(r.read_se().unwrap(), v);
        }
    }

    #[test]
    fn golomb_overflow_and_zero_range_are_safe() {
        // Max valid 32-bit ue(v) matching FFmpeg get_ue_golomb_long: 31 zeros + 32 ones => u32::MAX - 1
        // Bits: 31 zeros, then 32 ones = 63 bits (8 bytes):
        // bytes[0..3] = 0x00,0x00,0x00,0x01; bytes[4..7] = 0xFF,0xFF,0xFF,0xFE
        let data = [0x00, 0x00, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFE];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), u32::MAX - 1);

        let mut r2 = BitReader::new(&data);
        // Matches FFmpeg get_se_golomb_long: ((0xFFFFFFFE >> 1) ^ -1) + 1 = i32::MIN + 1
        assert_eq!(r2.read_se().unwrap(), i32::MIN + 1);

        // range == 0 returns Err instead of panicking
        let mut r3 = BitReader::new(&[0x80]);
        assert!(r3.read_te(0).is_err());
        assert!(r3.read_te0(0).is_err());
        assert_eq!(r3.read_te0(1).unwrap(), 0);
        assert_eq!(r3.read_te(2).unwrap(), 0); // first bit is 1 -> 1^1 = 0
    }

    #[test]
    fn padded_subslice_does_not_leak_adjacent_bytes() {
        // Logical payload is 1 byte (0x00), followed by 0xFF padding bytes
        let mut padded = vec![0xFFu8; 65];
        padded[0] = 0x00;
        let mut r = BitReader::from_padded(&padded, 1);
        assert_eq!(r.peek_bits(32), 0);
        assert!(r.read_ue().is_err());
        assert_eq!(r.read_bits(8).unwrap(), 0);
        assert!(r.read_bit().is_err());
    }

    #[test]
    fn bitreader_le_reads_lsb_first() {
        let data = [0b1011_0101u8, 0b0100_1110];
        let mut r = BitReaderLe::new(&data);
        assert_eq!(r.read_bits(4).unwrap(), 0b0101);
        assert_eq!(r.read_bits(4).unwrap(), 0b1011);
        assert_eq!(r.read_bits(8).unwrap(), 0b0100_1110);
        assert!(r.read_bit().is_err());
    }

    #[test]
    fn checksums_match_known_vectors() {
        // FFmpeg adler32 with init = 0 over [1, 2, 3]: s1 = 6, s2 = 1 + 3 + 6 = 10 -> (10 << 16) | 6
        assert_eq!(adler32_ffmpeg(0, &[1, 2, 3]), (10 << 16) | 6);
        // Standard RFC 1950 check with init = 1 over "Wikipedia" = 0x11E60398
        assert_eq!(adler32_ffmpeg(1, b"Wikipedia"), 0x11E6_0398);
        // IEEE CRC-32 over "123456789" = 0xCBF43926
        assert_eq!(crc32(0, b"123456789"), 0xCBF4_3926);
    }
}
