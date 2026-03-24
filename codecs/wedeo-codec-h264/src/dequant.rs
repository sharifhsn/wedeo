// H.264 dequantization (inverse quantization).
//
// Implements dequantization for 4x4 and 8x8 transform blocks, including
// pre-computed coefficient tables and the DC scale factor lookup used by
// the luma/chroma DC Hadamard transforms.
//
// Reference: ITU-T H.264 Section 8.5.12.1, FFmpeg libavcodec/h264_ps.c
// (`init_dequant4_coeff_table`, `init_dequant_tables`)

use crate::tables::{
    DEFAULT_SCALING4, DEFAULT_SCALING8, DEQUANT4_COEFF_INIT, DEQUANT8_COEFF_INIT,
    DEQUANT8_COEFF_INIT_SCAN,
};

/// Pre-computed dequantization table for 4x4 blocks with the default flat
/// scaling matrix.
///
/// Indexed as `DEQUANT4_DEFAULT[qp][raster_position]`.
///
/// Built by replicating FFmpeg's `init_dequant4_coeff_table` logic with the
/// default intra scaling matrix (all entries 16 for flat, but the actual
/// default_scaling4[0] values are used).
///
/// For each QP:
///   shift = qp/6 + 2
///   idx   = qp%6
///   for each position x in raster order:
///     scale_idx  = (x & 1) + ((x >> 2) & 1)
///     table[qp][x] = dequant4_coeff_init[idx][scale_idx]
///                     * scaling_matrix4[x] << shift
///
/// Maximum QP for 8-bit depth is 51.
///
/// Reference: FFmpeg `init_dequant4_coeff_table` in h264_ps.c
pub struct Dequant4Table {
    /// `coeffs[qp][position_in_raster_order]` for each of the 6 scaling lists.
    /// Index 0-2 = intra (Y, Cr, Cb), index 3-5 = inter (Y, Cr, Cb).
    pub coeffs: [[[u32; 16]; 52]; 6],
}

/// Pre-computed dequantization table for 8x8 blocks with the default flat
/// scaling matrix.
///
/// Indexed as `DEQUANT8_DEFAULT[qp][raster_position]`.
///
/// Reference: FFmpeg `init_dequant8_coeff_table` in h264_ps.c
pub struct Dequant8Table {
    /// `coeffs[qp][position_in_raster_order]` for each of the 6 scaling lists.
    pub coeffs: [[[u32; 64]; 52]; 6],
}

impl Dequant4Table {
    /// Build dequantization tables from 4x4 scaling matrices.
    ///
    /// `scaling_matrix4` is the array of 6 scaling lists (from PPS/SPS, or the
    /// default if none was signaled). Each list has 16 entries in raster order.
    ///
    /// This replicates FFmpeg's `init_dequant4_coeff_table` exactly.
    pub fn new(scaling_matrix4: &[[u8; 16]; 6]) -> Self {
        let mut table = Dequant4Table {
            coeffs: [[[0u32; 16]; 52]; 6],
        };

        for (i, sm) in scaling_matrix4.iter().enumerate() {
            for q in 0..52 {
                let shift = (q / 6) + 2;
                let idx = q % 6;
                for x in 0..16u32 {
                    // No transpose — both coefficients and scaling matrices are in raster order.
                    // FFmpeg also stores at position x directly (init_dequant4_coeff_table, h264_ps.c).
                    let scale_idx = ((x & 1) + ((x >> 2) & 1)) as usize;
                    table.coeffs[i][q][x as usize] = ((DEQUANT4_COEFF_INIT[idx][scale_idx] as u32)
                        * (sm[x as usize] as u32))
                        << shift;
                }
            }
        }

        table
    }

    /// Build dequantization tables using the default scaling matrices.
    pub fn default_tables() -> Self {
        // The 6 scaling lists: indices 0-2 use intra default, 3-5 use inter default.
        let mut scaling = [[0u8; 16]; 6];
        scaling[0] = DEFAULT_SCALING4[0];
        scaling[1] = DEFAULT_SCALING4[0];
        scaling[2] = DEFAULT_SCALING4[0];
        scaling[3] = DEFAULT_SCALING4[1];
        scaling[4] = DEFAULT_SCALING4[1];
        scaling[5] = DEFAULT_SCALING4[1];
        Self::new(&scaling)
    }
}

impl Dequant8Table {
    /// Build dequantization tables from 8x8 scaling matrices.
    ///
    /// `scaling_matrix8` is the array of 6 scaling lists (from PPS/SPS, or the
    /// default if none was signaled). Each list has 64 entries in raster order.
    ///
    /// This replicates FFmpeg's `init_dequant8_coeff_table` exactly.
    pub fn new(scaling_matrix8: &[[u8; 64]; 6]) -> Self {
        let mut table = Dequant8Table {
            coeffs: [[[0u32; 64]; 52]; 6],
        };

        for (i, sm) in scaling_matrix8.iter().enumerate() {
            for q in 0..52 {
                let shift = q / 6;
                let idx = q % 6;
                for x in 0..64u32 {
                    // FFmpeg stores dequant values at transposed positions because
                    // its coefficients are in transposed (column-major) order.
                    // Wedeo's coefficients are in standard row-major order, so we
                    // store at the standard position x directly (no transpose).
                    let scan_idx =
                        DEQUANT8_COEFF_INIT_SCAN[(((x >> 1) & 12) | (x & 3)) as usize] as usize;
                    table.coeffs[i][q][x as usize] = ((DEQUANT8_COEFF_INIT[idx][scan_idx] as u32)
                        * (sm[x as usize] as u32))
                        << shift;
                }
            }
        }

        table
    }

    /// Build dequantization tables using the default scaling matrices.
    pub fn default_tables() -> Self {
        let mut scaling = [[0u8; 64]; 6];
        scaling[0] = DEFAULT_SCALING8[0];
        scaling[1] = DEFAULT_SCALING8[0];
        scaling[2] = DEFAULT_SCALING8[0];
        scaling[3] = DEFAULT_SCALING8[1];
        scaling[4] = DEFAULT_SCALING8[1];
        scaling[5] = DEFAULT_SCALING8[1];
        Self::new(&scaling)
    }
}

/// Dequantize a 4x4 block of coefficients in-place using a pre-computed table.
///
/// `coeffs` are the quantized transform coefficients (from CAVLC/CABAC decode).
/// `dequant` is the row from the pre-computed dequant table for this QP and
/// scaling list: `dequant4_table.coeffs[list_idx][qp]`.
///
/// Applies `(level * qmul + 32) >> 6` per coefficient, matching FFmpeg's
/// inline dequant in `decode_residual` STORE_BLOCK for 4x4 blocks
/// (h264_cavlc.c line 564).
///
/// The pre-computed table includes `INIT * scaling_matrix << (qp/6 + 2)`.
/// The `>> 6` here normalizes the extra `<< 2` shift and the scaling matrix
/// factor (which is 16 for the default flat matrix, making the `<< 2` and
/// `* 16` combine to `<< 6`, perfectly canceled by `>> 6`).
///
/// For flat scaling (default, all entries 16), this produces exactly the
/// same result as `dequant_4x4_flat`: `level * INIT << (qp/6)`.
pub fn dequant_4x4(coeffs: &mut [i16; 16], dequant: &[u32; 16]) {
    for i in 0..16 {
        coeffs[i] = ((coeffs[i] as i32 * dequant[i] as i32 + 32) >> 6) as i16;
    }
}

/// Dequantize an 8x8 block of coefficients in-place using a pre-computed table.
///
/// Applies `(level * qmul + 32) >> 6` per coefficient, matching FFmpeg's
/// inline dequant in `decode_residual` for 8x8 blocks (h264_cavlc.c STORE_BLOCK).
///
/// The pre-computed table includes `INIT * scaling_matrix << (qp/6)`. The
/// `>> 6` normalization here reduces the coefficient magnitude to the range
/// expected by the IDCT (which applies its own `>> 6` at the output).
///
/// Without this normalization, coefficients would be 64x too large, causing
/// massive overflow in the IDCT and garbled output.
pub fn dequant_8x8(coeffs: &mut [i16; 64], dequant: &[u32; 64]) {
    for i in 0..64 {
        coeffs[i] = ((coeffs[i] as i32 * dequant[i] as i32 + 32) >> 6) as i16;
    }
}

/// Dequantize a 4x4 block using simple per-QP scaling (no pre-computed table).
///
/// This is a standalone dequantization path that computes scaling on the fly,
/// useful when pre-computed tables are not available.
///
/// For each position:
///   `coeffs[i] = coeffs[i] * scale_factor[qp%6][pos_class] << (qp/6)`
///
/// The position class depends on (row, col) parity:
///   - (even, even) -> class 0 (diagonal positions)
///   - (even, odd) or (odd, even) -> class 1 (off-diagonal)
///   - (odd, odd) -> class 2 (corner positions)
///
/// The scale factors per qp%6 are from `DEQUANT4_COEFF_INIT`.
pub fn dequant_4x4_flat(coeffs: &mut [i16; 16], qp: u8) {
    let qp_per = (qp / 6) as u32;
    let qp_rem = (qp % 6) as usize;

    // Position class for each raster position in a 4x4 block.
    // class = (row & 1) + (col & 1), mapping to DEQUANT4_COEFF_INIT indices.
    const POS_CLASS: [usize; 16] = [
        0, 1, 0, 1, // row 0: (0,0)=0, (0,1)=1, (0,2)=0, (0,3)=1
        1, 2, 1, 2, // row 1: (1,0)=1, (1,1)=2, (1,2)=1, (1,3)=2
        0, 1, 0, 1, // row 2: same as row 0
        1, 2, 1, 2, // row 3: same as row 1
    ];

    let scale = &DEQUANT4_COEFF_INIT[qp_rem];
    for i in 0..16 {
        let s = scale[POS_CLASS[i]] as i32;
        coeffs[i] = ((coeffs[i] as i32 * s) << qp_per) as i16;
    }
}

/// Get the DC dequantization scale factor for a given QP.
///
/// Used as the `qmul` argument to `luma_dc_dequant_idct` and
/// `chroma_dc_dequant_idct`. This is the dequant table value at position [0]
/// (the DC position) for the given QP and scaling list.
///
/// For the default flat scaling matrix (intra, list 0):
///   `qmul = dequant4_coeff_init[qp%6][0] * scaling_matrix4[0][0] << (qp/6 + 2)`
///
/// With the default intra scaling matrix, `scaling_matrix4[0][0] = 6`, so:
///   `qmul = dequant4_coeff_init[qp%6][0] * 6 << (qp/6 + 2)`
///
/// This matches FFmpeg's `h->dequant4_coeff[0][qp][0]`.
pub fn dc_dequant_scale(dequant_table: &Dequant4Table, list_idx: usize, qp: u8) -> i32 {
    dequant_table.coeffs[list_idx][qp as usize][0] as i32
}

/// Get the DC dequant scale factor using the default flat intra scaling matrix.
///
/// Convenience function that computes the scale on the fly without needing
/// a pre-built table. Uses scaling list index 0 (intra Y).
pub fn dc_dequant_scale_default(qp: u8) -> i32 {
    let qp_per = (qp / 6) as u32;
    let qp_rem = (qp % 6) as usize;
    // Position (0,0) has class 0 -> scale = DEQUANT4_COEFF_INIT[qp_rem][0]
    // Default intra scaling_matrix4[0][0] = 6
    ((DEQUANT4_COEFF_INIT[qp_rem][0] as i32) * (DEFAULT_SCALING4[0][0] as i32)) << (qp_per + 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant4_table_qp0_position0() {
        // At QP=0: shift = 0/6 + 2 = 2, idx = 0%6 = 0
        // Position x=0: raster_pos = (0>>2)|((0<<2)&0xF) = 0
        //   scale_idx = (0&1) + ((0>>2)&1) = 0
        //   value = dequant4_coeff_init[0][0] * scaling_matrix4[0][0] << 2
        //         = 10 * 6 << 2 = 60 << 2 = 240
        let table = Dequant4Table::default_tables();
        assert_eq!(table.coeffs[0][0][0], 240);
    }

    #[test]
    fn dequant4_table_qp6_has_extra_shift() {
        // At QP=6: shift = 6/6 + 2 = 3, idx = 6%6 = 0
        // Position x=0: same scale_idx as QP=0
        //   value = 10 * 6 << 3 = 60 << 3 = 480
        let table = Dequant4Table::default_tables();
        assert_eq!(table.coeffs[0][6][0], 480);
        // Should be exactly 2x the QP=0 value (one extra shift bit)
        assert_eq!(table.coeffs[0][6][0], table.coeffs[0][0][0] * 2);
    }

    #[test]
    fn dequant4_table_qp12_double_shift() {
        // At QP=12: shift = 12/6 + 2 = 4, idx = 12%6 = 0
        // value = 10 * 6 << 4 = 960
        let table = Dequant4Table::default_tables();
        assert_eq!(table.coeffs[0][12][0], 960);
        assert_eq!(table.coeffs[0][12][0], table.coeffs[0][0][0] * 4);
    }

    #[test]
    fn dequant4_table_qp1_uses_second_init_row() {
        // At QP=1: shift = 0/6 + 2 = 2, idx = 1%6 = 1
        // Position x=0: scale_idx = 0
        //   value = dequant4_coeff_init[1][0] * 6 << 2 = 11 * 6 << 2 = 264
        let table = Dequant4Table::default_tables();
        assert_eq!(table.coeffs[0][1][0], 264);
    }

    #[test]
    fn dequant4_table_intra_vs_inter() {
        // Intra list 0 and inter list 3 should differ because they use
        // different default scaling matrices.
        let table = Dequant4Table::default_tables();
        // Position 0: intra scaling[0]=6, inter scaling[0]=10
        // QP=0: intra = 10*6<<2 = 240, inter = 10*10<<2 = 400
        assert_eq!(table.coeffs[0][0][0], 240); // intra
        assert_eq!(table.coeffs[3][0][0], 400); // inter
    }

    #[test]
    fn dequant4_table_position_variation() {
        // Different positions within the 4x4 block should have different
        // scale factors (because the default scaling matrix is not flat).
        let table = Dequant4Table::default_tables();
        // Position (0,0) vs position (0,1) at QP=0
        // These should differ because DEFAULT_SCALING4 is not uniform.
        assert_ne!(table.coeffs[0][0][0], table.coeffs[0][0][1]);
    }

    #[test]
    fn dequant_4x4_in_place() {
        let table = Dequant4Table::default_tables();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 1;

        dequant_4x4(&mut coeffs, &table.coeffs[0][0]);

        // coeffs[0] = (1 * 240 + 32) >> 6 = 272 >> 6 = 4
        // (default intra scaling_matrix4[0][0] = 6, INIT[0][0] = 10,
        //  table = 10 * 6 << 2 = 240, dequant applies (+32)>>6)
        assert_eq!(coeffs[0], 4);
        // All other coefficients remain 0
        for i in 1..16 {
            assert_eq!(coeffs[i], 0, "coeff {i}");
        }
    }

    #[test]
    fn dequant_4x4_flat_qp0() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 1;

        dequant_4x4_flat(&mut coeffs, 0);

        // Position (0,0), class 0: scale = DEQUANT4_COEFF_INIT[0][0] = 10
        // qp_per = 0, so no shift: result = 1 * 10 << 0 = 10
        assert_eq!(coeffs[0], 10);
    }

    #[test]
    fn dequant_4x4_flat_qp6() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 1;

        dequant_4x4_flat(&mut coeffs, 6);

        // QP=6: qp_per = 1, qp_rem = 0
        // Position (0,0), class 0: scale = 10
        // result = 1 * 10 << 1 = 20
        assert_eq!(coeffs[0], 20);
    }

    #[test]
    fn dequant_4x4_flat_position_classes() {
        // Verify that different position classes get different scale factors
        let mut coeffs = [1i16; 16];
        dequant_4x4_flat(&mut coeffs, 0);

        // Position (0,0) class 0: scale = 10
        assert_eq!(coeffs[0], 10);
        // Position (0,1) class 1: scale = 13
        assert_eq!(coeffs[1], 13);
        // Position (1,1) class 2: scale = 16
        assert_eq!(coeffs[5], 16);
    }

    #[test]
    fn dc_dequant_scale_matches_table() {
        let table = Dequant4Table::default_tables();

        for qp in 0..52u8 {
            let from_table = dc_dequant_scale(&table, 0, qp);
            let from_fn = dc_dequant_scale_default(qp);
            assert_eq!(
                from_table, from_fn,
                "DC scale mismatch at QP={qp}: table={from_table}, fn={from_fn}"
            );
        }
    }

    #[test]
    fn dc_dequant_scale_default_qp0() {
        // QP=0: qp_per=0, qp_rem=0
        // scale = 10 * 6 << 0 << 2 = 60 << 2 = 240
        assert_eq!(dc_dequant_scale_default(0), 240);
    }

    #[test]
    fn dc_dequant_scale_default_qp6() {
        // QP=6: qp_per=1, qp_rem=0
        // scale = 10 * 6 << 1 << 2 = 120 << 2 = 480
        assert_eq!(dc_dequant_scale_default(6), 480);
    }

    #[test]
    fn dequant8_table_qp0_position0() {
        // At QP=0: shift = 0/6 = 0, idx = 0%6 = 0
        // Position x=0: raster_pos = (0>>3)|((0&7)<<3) = 0
        //   scan_idx = DEQUANT8_COEFF_INIT_SCAN[((0>>1)&12)|(0&3)] = SCAN[0] = 0
        //   value = dequant8_coeff_init[0][0] * scaling_matrix8[0][0] << 0
        //         = 20 * 6 = 120
        let table = Dequant8Table::default_tables();
        assert_eq!(table.coeffs[0][0][0], 120);
    }

    #[test]
    fn dequant8_table_qp6_extra_shift() {
        // At QP=6: shift = 1
        // value = 20 * 6 << 1 = 240
        let table = Dequant8Table::default_tables();
        assert_eq!(table.coeffs[0][6][0], 240);
        assert_eq!(table.coeffs[0][6][0], table.coeffs[0][0][0] * 2);
    }

    #[test]
    fn dequant_8x8_in_place() {
        let table = Dequant8Table::default_tables();
        let mut coeffs = [0i16; 64];
        coeffs[0] = 1;

        dequant_8x8(&mut coeffs, &table.coeffs[0][0]);

        // (1 * 120 + 32) >> 6 = 152 >> 6 = 2
        assert_eq!(coeffs[0], 2);
        for i in 1..64 {
            assert_eq!(coeffs[i], 0, "coeff {i}");
        }
    }
}
