// VP9 quantizer lookup helpers.
//
// Translated from FFmpeg's libavcodec/vp9.c (quantization section).
// Maps a quantizer index + delta to actual DC/AC quantizer step sizes
// using the lookup tables in data.rs.

use crate::data::{AC_QLOOKUP, DC_QLOOKUP};

/// Return the bit-depth index (0=8-bit, 1=10-bit, 2=12-bit).
fn bpp_index(bit_depth: u8) -> usize {
    match bit_depth {
        10 => 1,
        12 => 2,
        _ => 0,
    }
}

/// Clamp a quantizer index sum into [0, 255].
fn clamp_q(base: u8, delta: i8) -> usize {
    (base as i16 + delta as i16).clamp(0, 255) as usize
}

/// Look up the DC quantizer step for a given base index, delta, and bit depth.
///
/// Mirrors `ff_vp9_dc_qlookup[bpp_index][clamp(q_idx + delta, 0, 255)]`
/// from FFmpeg's vp9.c.
pub fn get_dc_quant(q_idx: u8, delta: i8, bit_depth: u8) -> i16 {
    DC_QLOOKUP[bpp_index(bit_depth)][clamp_q(q_idx, delta)]
}

/// Look up the AC quantizer step for a given base index, delta, and bit depth.
///
/// Mirrors `ff_vp9_ac_qlookup[bpp_index][clamp(q_idx + delta, 0, 255)]`
/// from FFmpeg's vp9.c.
pub fn get_ac_quant(q_idx: u8, delta: i8, bit_depth: u8) -> i16 {
    AC_QLOOKUP[bpp_index(bit_depth)][clamp_q(q_idx, delta)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_quant_zero_index() {
        // q_idx=0, delta=0 → DC_QLOOKUP[0][0] = 4 (from VP9 spec Table)
        let q = get_dc_quant(0, 0, 8);
        assert!(q > 0, "DC quantizer must be positive");
    }

    #[test]
    fn test_ac_quant_zero_index() {
        let q = get_ac_quant(0, 0, 8);
        assert!(q > 0, "AC quantizer must be positive");
    }

    #[test]
    fn test_quant_clamp_low() {
        // delta makes index go below 0 → clamp to 0
        let q = get_dc_quant(0, -10, 8);
        assert_eq!(q, get_dc_quant(0, 0, 8));
    }

    #[test]
    fn test_quant_clamp_high() {
        // delta makes index go above 255 → clamp to 255
        let q = get_dc_quant(250, 100, 8);
        assert_eq!(q, get_dc_quant(255, 0, 8));
    }
}
