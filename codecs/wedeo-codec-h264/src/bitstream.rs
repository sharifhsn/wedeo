#![forbid(unsafe_code)]
//! Big-endian bitstream reader and exp-Golomb parsers for H.264.
//!
//! Matches FFmpeg's `libavcodec/get_bits.h` and `libavcodec/golomb.h` in 100% safe Rust.

use wedeo_core::error::{Error, Result};

/// Maximum number of leading zeros allowed in an exp-Golomb code.
const MAX_EXP_GOLOMB_LEADING_ZEROS: u32 = 31;

#[inline]
fn get_u32b(buf: &[u8]) -> u32 {
    u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]])
}

#[inline]
fn get_u64b(buf: &[u8]) -> u64 {
    u64::from_be_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ])
}

/// Bitstream reader trait kept for call-site compatibility.
pub trait BitRead<'a>: Copy {
    fn new(buffer: &'a [u8]) -> Self;
    fn consumed(&self) -> usize;
    fn available(&self) -> usize;
    fn skip_bits(&mut self, size: usize);
    fn get_bit(&mut self) -> bool;
    fn get_bits_64(&mut self, n: usize) -> u64;
    fn get_bits_32(&mut self, n: usize) -> u32;
    fn peek_bit(&mut self) -> bool;
    fn peek_bits_32(&mut self, n: usize) -> u32;
    fn peek_bits_64(&self, n: usize) -> u64;
    fn align_bits(&mut self);
}

/// Big-endian 64-bit cached bitstream reader.
#[derive(Debug, Clone, Copy)]
pub struct BitReadBE<'a> {
    buffer: &'a [u8],
    index: usize,
    cache: u64,
    left: usize,
}

impl<'a> BitReadBE<'a> {
    #[inline]
    fn can_refill(&self) -> bool {
        self.index + 8 <= self.buffer.len()
    }

    #[inline]
    fn fill32(&self) -> u64 {
        u64::from(get_u32b(&self.buffer[self.index..]))
    }

    #[inline]
    fn fill64(&self) -> u64 {
        get_u64b(&self.buffer[self.index..])
    }

    #[inline]
    fn peek_val(&self, n: usize) -> u64 {
        self.cache.checked_shr(64 - n as u32).unwrap_or(0)
    }

    #[inline]
    fn skip_rem(&mut self, n: usize) {
        self.cache = self.cache.checked_shl(n as u32).unwrap_or(0);
        self.left = self.left.saturating_sub(n);
    }

    #[inline]
    fn merge_val(msp: u64, lsp: u64, _msb: usize, lsb: usize) -> u64 {
        msp | lsp.checked_shl(lsb as u32).unwrap_or(0)
    }

    #[inline]
    fn build_cache(cache: u64, refill: u64, cache_size: usize) -> u64 {
        cache | refill << (32 - cache_size)
    }

    #[inline]
    fn refill32(&mut self) {
        if !self.can_refill() {
            return;
        }
        let val = self.fill32();
        self.cache = Self::build_cache(self.cache, val, self.left);
        self.index += 4;
        self.left += 32;
    }

    #[inline]
    fn refill64(&mut self) {
        if !self.can_refill() {
            return;
        }
        self.cache = self.fill64();
        self.index += 8;
        self.left = 64;
    }

    #[inline]
    fn get_val(&mut self, n: usize) -> u64 {
        let ret = self.peek_val(n);
        self.skip_rem(n);
        ret
    }
}

impl<'a> BitRead<'a> for BitReadBE<'a> {
    fn new(buffer: &'a [u8]) -> Self {
        let mut reader = Self {
            buffer,
            index: 0,
            cache: 0,
            left: 0,
        };
        reader.refill64();
        reader
    }

    #[inline]
    fn consumed(&self) -> usize {
        self.index * 8 - self.left
    }

    #[inline]
    fn available(&self) -> usize {
        (self.buffer.len() - self.index) * 8 + self.left
    }

    #[inline]
    fn skip_bits(&mut self, mut n: usize) {
        if self.left < n {
            n -= self.left;
            if n > 64 {
                let skip = n / 8;
                n -= skip * 8;
                self.index += skip;
            }
            self.skip_rem(n);
            self.refill64();
        }
        self.skip_rem(n);
    }

    #[inline]
    fn get_bit(&mut self) -> bool {
        if self.left == 0 {
            self.refill64();
        }
        self.get_val(1) != 0
    }

    #[inline]
    fn get_bits_64(&mut self, mut n: usize) -> u64 {
        let mut left = 0;
        let mut ret = 0;
        if n == 0 {
            return 0;
        }
        if self.left < n {
            n -= self.left;
            left = self.left;
            ret = self.get_val(left);
            self.refill64();
        }
        Self::merge_val(self.get_val(n), ret, left, n)
    }

    #[inline]
    fn get_bits_32(&mut self, n: usize) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.left <= n {
            self.refill32();
        }
        self.get_val(n) as u32
    }

    #[inline]
    fn peek_bit(&mut self) -> bool {
        let mut tmp = *self;
        tmp.get_bit()
    }

    #[inline]
    fn peek_bits_32(&mut self, n: usize) -> u32 {
        let mut tmp = *self;
        tmp.get_bits_32(n)
    }

    #[inline]
    fn peek_bits_64(&self, n: usize) -> u64 {
        let mut tmp = *self;
        tmp.get_bits_64(n)
    }

    #[inline]
    fn align_bits(&mut self) {
        let left = self.left & 7;
        self.skip_bits(left);
    }
}

/// Read an unsigned exp-Golomb code (`ue(v)`) from a big-endian bitstream reader.
pub fn get_ue_golomb(br: &mut BitReadBE<'_>) -> Result<u32> {
    let buf = br.peek_bits_32(32);
    if buf == 0 {
        return Err(Error::invalid_data(
            "h264/golomb",
            "exp-Golomb leading zeros exceed 31",
        ));
    }
    let leading_zeros = buf.leading_zeros();
    if leading_zeros > MAX_EXP_GOLOMB_LEADING_ZEROS {
        return Err(Error::invalid_data(
            "h264/golomb",
            "exp-Golomb leading zeros exceed 31",
        ));
    }
    br.skip_bits(leading_zeros as usize);
    let val = br.get_bits_32(leading_zeros as usize + 1);
    Ok(val - 1)
}

/// Read a signed exp-Golomb code (`se(v)`) from a big-endian bitstream reader.
pub fn get_se_golomb(br: &mut BitReadBE<'_>) -> Result<i32> {
    let buf = get_ue_golomb(br)?;
    let sign = (buf & 1) as i32 - 1;
    Ok(((buf >> 1) as i32 ^ sign) + 1)
}
