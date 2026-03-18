//! Bitstream reading utilities for codec implementations.
//!
//! Re-exports the `av-bitstream` crate's reader types and provides
//! exp-Golomb parsing functions matching FFmpeg's `libavcodec/golomb.h`.

pub use av_bitstream::bitread::{BitRead, BitReadBE, BitReadLE};

use wedeo_core::{Error, Result};

/// Maximum number of leading zeros allowed in an exp-Golomb code.
///
/// FFmpeg's `get_ue_golomb_long` supports up to 31 leading zeros (reading
/// a 32-bit peek). We use the same limit.
const MAX_EXP_GOLOMB_LEADING_ZEROS: u32 = 31;

/// Read an unsigned exp-Golomb code (ue(v)) from a big-endian bitstream reader.
///
/// Matches FFmpeg's `get_ue_golomb_long` from `libavcodec/golomb.h`:
/// count leading zeros by peeking 32 bits, then read the value bits.
///
/// Returns `Error::InvalidData` if the code has more than 31 leading zeros
/// (which would require reading more than 32 bits total).
pub fn get_ue_golomb(br: &mut BitReadBE<'_>) -> Result<u32> {
    // Peek up to 32 bits to find the leading '1' bit.
    // We need at least 1 bit available; peek_bits_32 returns 0 if no bits.
    let buf = br.peek_bits_32(32);

    if buf == 0 {
        // All 32 peeked bits are zero — the code is too long.
        return Err(Error::InvalidData);
    }

    // Count leading zeros: position of highest set bit.
    // For buf != 0, leading_zeros is 0..=31.
    let leading_zeros = buf.leading_zeros();

    if leading_zeros > MAX_EXP_GOLOMB_LEADING_ZEROS {
        return Err(Error::InvalidData);
    }

    // Skip the leading zeros.
    br.skip_bits(leading_zeros as usize);

    // Read (leading_zeros + 1) bits: the '1' prefix plus the value suffix.
    let val = br.get_bits_32(leading_zeros as usize + 1);

    // The decoded value is val - 1.
    Ok(val - 1)
}

/// Read a signed exp-Golomb code (se(v)) from a big-endian bitstream reader.
///
/// Matches FFmpeg's `get_se_golomb_long` from `libavcodec/golomb.h`:
/// reads an unsigned code, then maps even values to negative and odd to positive.
///
/// Mapping: ue=0 -> 0, ue=1 -> 1, ue=2 -> -1, ue=3 -> 2, ue=4 -> -2, ...
pub fn get_se_golomb(br: &mut BitReadBE<'_>) -> Result<i32> {
    let buf = get_ue_golomb(br)?;

    // FFmpeg formula: sign = (buf & 1) - 1; result = ((buf >> 1) ^ sign) + 1
    // When buf is odd:  sign = 0,  result = (buf >> 1) + 1     (positive)
    // When buf is even: sign = -1, result = -(buf >> 1)         (negative, or 0 when buf=0)
    let sign = (buf & 1) as i32 - 1; // 0 for odd, -1 for even
    Ok(((buf >> 1) as i32 ^ sign) + 1)
}

/// Read a truncated exp-Golomb code (te(v)) from a big-endian bitstream reader.
///
/// Matches FFmpeg's `get_te_golomb` from `libavcodec/golomb.h`:
/// if range == 2, reads a single bit and XORs with 1; otherwise falls through
/// to unsigned exp-Golomb.
///
/// Panics if `range` < 1 (matching FFmpeg's `av_assert2(range >= 1)`).
pub fn get_te_golomb(br: &mut BitReadBE<'_>, range: u32) -> Result<u32> {
    assert!(range >= 1, "get_te_golomb: range must be >= 1");

    if range == 2 {
        Ok(u32::from(br.get_bit()) ^ 1)
    } else {
        get_ue_golomb(br)
    }
}

/// Read a truncated exp-Golomb code (te(v)) with range==1 returning 0.
///
/// Matches FFmpeg's `get_te0_golomb` from `libavcodec/golomb.h`:
/// if range == 1, returns 0 without reading; if range == 2, reads a single
/// bit XOR 1; otherwise falls through to unsigned exp-Golomb.
///
/// Panics if `range` < 1 (matching FFmpeg's `av_assert2(range >= 1)`).
pub fn get_te0_golomb(br: &mut BitReadBE<'_>, range: u32) -> Result<u32> {
    assert!(range >= 1, "get_te0_golomb: range must be >= 1");

    if range == 1 {
        Ok(0)
    } else if range == 2 {
        Ok(u32::from(br.get_bit()) ^ 1)
    } else {
        get_ue_golomb(br)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a BitReadBE from a byte slice.
    ///
    /// The `av-bitstream` reader does 8-byte reads for cache refills, so we
    /// pad the input to at least 8 bytes to avoid reading out of bounds.
    fn make_reader(data: &[u8]) -> BitReadBE<'_> {
        BitReadBE::new(data)
    }

    // --- Unsigned exp-Golomb (ue) tests ---

    #[test]
    fn ue_value_0() {
        // 0 => code "1" => bits: 1000_0000 = 0x80
        let data = [0x80, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 0);
    }

    #[test]
    fn ue_value_1() {
        // 1 => code "010" => bits: 0100_0000 = 0x40
        let data = [0x40, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 1);
    }

    #[test]
    fn ue_value_2() {
        // 2 => code "011" => bits: 0110_0000 = 0x60
        let data = [0x60, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 2);
    }

    #[test]
    fn ue_value_3() {
        // 3 => code "00100" => bits: 0010_0000 = 0x20
        let data = [0x20, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 3);
    }

    #[test]
    fn ue_value_4() {
        // 4 => code "00101" => bits: 0010_1000 = 0x28
        let data = [0x28, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 4);
    }

    #[test]
    fn ue_value_5() {
        // 5 => code "00110" => bits: 0011_0000 = 0x30
        let data = [0x30, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 5);
    }

    #[test]
    fn ue_value_6() {
        // 6 => code "00111" => bits: 0011_1000 = 0x38
        let data = [0x38, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 6);
    }

    #[test]
    fn ue_sequential_values() {
        // Read multiple ue codes in sequence: 0 (1), 1 (010), 2 (011)
        // Bits: 1 010 011 0 = 0xA6 (1010_0110)
        let data = [0xA6, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 0); // "1"
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 1); // "010"
        assert_eq!(get_ue_golomb(&mut br).unwrap(), 2); // "011"
    }

    #[test]
    fn ue_error_all_zeros() {
        // All zeros — no '1' bit found in 32-bit peek.
        let data = [0u8; 8];
        let mut br = make_reader(&data);
        assert_eq!(get_ue_golomb(&mut br), Err(Error::InvalidData));
    }

    // --- Signed exp-Golomb (se) tests ---

    #[test]
    fn se_mapping() {
        // ue=0 -> se=0, ue=1 -> se=1, ue=2 -> se=-1, ue=3 -> se=2, ue=4 -> se=-2

        // se=0: code for ue=0 is "1" = 0x80
        let data = [0x80, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_se_golomb(&mut br).unwrap(), 0);

        // se=1: code for ue=1 is "010" = 0x40
        let data = [0x40, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_se_golomb(&mut br).unwrap(), 1);

        // se=-1: code for ue=2 is "011" = 0x60
        let data = [0x60, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_se_golomb(&mut br).unwrap(), -1);

        // se=2: code for ue=3 is "00100" = 0x20
        let data = [0x20, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_se_golomb(&mut br).unwrap(), 2);

        // se=-2: code for ue=4 is "00101" = 0x28
        let data = [0x28, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_se_golomb(&mut br).unwrap(), -2);
    }

    // --- Truncated exp-Golomb (te) tests ---

    #[test]
    fn te_range_2_bit_0() {
        // range==2: read 1 bit, XOR with 1.
        // Bit = 0 => result = 0 ^ 1 = 1
        let data = [0x00, 0, 0, 0, 0, 0, 0, 0]; // first bit is 0
        let mut br = make_reader(&data);
        assert_eq!(get_te_golomb(&mut br, 2).unwrap(), 1);
    }

    #[test]
    fn te_range_2_bit_1() {
        // range==2: read 1 bit, XOR with 1.
        // Bit = 1 => result = 1 ^ 1 = 0
        let data = [0x80, 0, 0, 0, 0, 0, 0, 0]; // first bit is 1
        let mut br = make_reader(&data);
        assert_eq!(get_te_golomb(&mut br, 2).unwrap(), 0);
    }

    #[test]
    fn te_range_gt2_falls_through_to_ue() {
        // range > 2: fall through to ue. Code "1" => ue=0
        let data = [0x80, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_te_golomb(&mut br, 5).unwrap(), 0);
    }

    // --- te0 variant tests ---

    #[test]
    fn te0_range_1_returns_zero() {
        // range==1: returns 0 without reading any bits.
        let data = [0xFF, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_te0_golomb(&mut br, 1).unwrap(), 0);
    }

    #[test]
    fn te0_range_2_reads_bit() {
        // range==2: same as te, read 1 bit XOR 1.
        let data = [0x80, 0, 0, 0, 0, 0, 0, 0]; // first bit = 1 => 1^1=0
        let mut br = make_reader(&data);
        assert_eq!(get_te0_golomb(&mut br, 2).unwrap(), 0);
    }

    #[test]
    fn te0_range_gt2_falls_through_to_ue() {
        // range > 2: fall through to ue. Code "010" => ue=1
        let data = [0x40, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        assert_eq!(get_te0_golomb(&mut br, 8).unwrap(), 1);
    }

    // --- Panic tests ---

    #[test]
    #[should_panic(expected = "range must be >= 1")]
    fn te_panics_on_range_0() {
        let data = [0x80, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        let _ = get_te_golomb(&mut br, 0);
    }

    #[test]
    #[should_panic(expected = "range must be >= 1")]
    fn te0_panics_on_range_0() {
        let data = [0x80, 0, 0, 0, 0, 0, 0, 0];
        let mut br = make_reader(&data);
        let _ = get_te0_golomb(&mut br, 0);
    }
}
