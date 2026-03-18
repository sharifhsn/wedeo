// H.264/AVC lookup tables.
//
// Scan orders, default scaling matrices, QP mapping, CBP tables, and level limits.
//
// Reference: ITU-T H.264 spec, FFmpeg libavcodec/h264data.c, mathtables.c, h264_ps.c

/// Zigzag scan order for 4x4 blocks (H.264 spec Table 8-13).
///
/// Maps linear index to raster position in a 4x4 block.
/// From FFmpeg `ff_zigzag_scan` in mathtables.c.
pub const ZIGZAG_SCAN_4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Field scan order for 4x4 blocks.
///
/// Column-first scan used for interlaced (field) macroblocks.
/// From FFmpeg `field_scan` in h264_slice.c.
pub const FIELD_SCAN_4X4: [u8; 16] = [0, 4, 1, 8, 12, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// Zigzag scan order for 8x8 blocks (H.264 spec Table 8-14).
///
/// Maps linear index to raster position in an 8x8 block.
/// From FFmpeg `ff_zigzag_direct` in mathtables.c.
pub const ZIGZAG_SCAN_8X8: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Field scan order for 8x8 blocks.
///
/// Column-first scan used for interlaced (field) macroblocks with 8x8 transforms.
/// From FFmpeg `field_scan8x8` in h264_slice.c.
pub const FIELD_SCAN_8X8: [u8; 64] = [
    0, 8, 16, 1, 9, 24, 32, 17, 2, 25, 40, 48, 56, 33, 10, 3, 18, 41, 49, 57, 26, 11, 4, 19, 34,
    42, 50, 58, 27, 12, 5, 20, 35, 43, 51, 59, 28, 13, 6, 21, 36, 44, 52, 60, 29, 14, 22, 37, 45,
    53, 61, 30, 7, 15, 38, 46, 54, 62, 23, 31, 39, 47, 55, 63,
];

/// QP-to-chroma-QP mapping for 8-bit depth (H.264 spec Table 8-15).
///
/// Maps luma QP (0-51) to chroma QP. Used for chroma deblocking and
/// dequantization. From FFmpeg `ff_h264_chroma_qp` (depth=8 row).
pub const CHROMA_QP_TABLE: [u8; 52] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39,
    39, 39,
];

/// Default scaling matrices for 4x4 blocks (intra and inter).
///
/// Index 0 = intra (Table 7-3), index 1 = inter (Table 7-4).
/// From FFmpeg `default_scaling4` in h264_ps.c.
pub const DEFAULT_SCALING4: [[u8; 16]; 2] = [
    [
        6, 13, 20, 28, 13, 20, 28, 32, 20, 28, 32, 37, 28, 32, 37, 42,
    ],
    [
        10, 14, 20, 24, 14, 20, 24, 27, 20, 24, 27, 30, 24, 27, 30, 34,
    ],
];

/// Default scaling matrices for 8x8 blocks (intra and inter).
///
/// Index 0 = intra (Table 7-5), index 1 = inter (Table 7-6).
/// From FFmpeg `default_scaling8` in h264_ps.c.
pub const DEFAULT_SCALING8: [[u8; 64]; 2] = [
    [
        6, 10, 13, 16, 18, 23, 25, 27, 10, 11, 16, 18, 23, 25, 27, 29, 13, 16, 18, 23, 25, 27, 29,
        31, 16, 18, 23, 25, 27, 29, 31, 33, 18, 23, 25, 27, 29, 31, 33, 36, 23, 25, 27, 29, 31, 33,
        36, 38, 25, 27, 29, 31, 33, 36, 38, 40, 27, 29, 31, 33, 36, 38, 40, 42,
    ],
    [
        9, 13, 15, 17, 19, 21, 22, 24, 13, 13, 17, 19, 21, 22, 24, 25, 15, 17, 19, 21, 22, 24, 25,
        27, 17, 19, 21, 22, 24, 25, 27, 28, 19, 21, 22, 24, 25, 27, 28, 30, 21, 22, 24, 25, 27, 28,
        30, 32, 22, 24, 25, 27, 28, 30, 32, 33, 24, 25, 27, 28, 30, 32, 33, 35,
    ],
];

/// Level limits: maximum DPB MBs per level.
///
/// Each entry is (level_idc * 10, max_dpb_mbs). Level 3.1 is stored as 31.
/// From FFmpeg `level_max_dpb_mbs` in h264_ps.c.
pub const LEVEL_MAX_DPB_MBS: [(u32, u32); 16] = [
    (10, 396),
    (11, 900),
    (12, 2376),
    (13, 2376),
    (20, 2376),
    (21, 4752),
    (22, 8100),
    (30, 8100),
    (31, 18000),
    (32, 20480),
    (40, 32768),
    (41, 32768),
    (42, 34816),
    (50, 110400),
    (51, 184320),
    (52, 184320),
];

/// CBP mapping for I-slice 4x4 macroblocks (golomb code to CBP).
///
/// Maps unsigned exp-Golomb coded value to coded block pattern for
/// Intra_4x4 macroblocks (H.264 spec Table 9-4, column "Intra_4x4").
/// From FFmpeg `ff_h264_golomb_to_intra4x4_cbp` in h264data.c.
pub const GOLOMB_TO_INTRA4X4_CBP: [u8; 48] = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28,
    35, 37, 42, 44, 1, 2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];

/// CBP mapping for P/B-slice macroblocks (golomb code to CBP).
///
/// Maps unsigned exp-Golomb coded value to coded block pattern for
/// Inter macroblocks (H.264 spec Table 9-4, column "Inter").
/// From FFmpeg `ff_h264_golomb_to_inter_cbp` in h264data.c.
pub const GOLOMB_TO_INTER_CBP: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34,
    36, 40, 39, 43, 45, 46, 17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];

/// Chroma DC scan order for 4:2:0 (2x2 block).
///
/// From FFmpeg `ff_h264_chroma_dc_scan` in h264data.c.
pub const CHROMA_DC_SCAN: [u8; 4] = [0, 16, 32, 48];

/// Chroma DC scan order for 4:2:2 (2x4 block).
///
/// From FFmpeg `ff_h264_chroma422_dc_scan` in h264data.c.
pub const CHROMA422_DC_SCAN: [u8; 8] = [0, 32, 16, 64, 96, 48, 80, 112];

/// Dequantization coefficient init table for 4x4 blocks.
///
/// From FFmpeg `ff_h264_dequant4_coeff_init` in h264data.c.
pub const DEQUANT4_COEFF_INIT: [[u8; 3]; 6] = [
    [10, 13, 16],
    [11, 14, 18],
    [13, 16, 20],
    [14, 18, 23],
    [16, 20, 25],
    [18, 23, 29],
];

/// Dequantization scan pattern for 8x8 blocks.
///
/// From FFmpeg `ff_h264_dequant8_coeff_init_scan` in h264data.c.
pub const DEQUANT8_COEFF_INIT_SCAN: [u8; 16] = [0, 3, 4, 3, 3, 1, 5, 1, 4, 5, 2, 5, 3, 1, 5, 1];

/// Dequantization coefficient init table for 8x8 blocks.
///
/// From FFmpeg `ff_h264_dequant8_coeff_init` in h264data.c.
pub const DEQUANT8_COEFF_INIT: [[u8; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

// ---------------------------------------------------------------------------
// B-frame macroblock type tables
// ---------------------------------------------------------------------------

/// B-slice macroblock type info.
///
/// Each entry: (partition_size, partition_count, [part0_l0, part0_l1, part1_l0, part1_l1])
/// where partition_size is: 0=16x16, 1=16x8, 2=8x16, 3=8x8, 4=direct.
/// Per-partition L0/L1 flags indicate which reference lists are used.
///
/// Index 0 = B_Direct_16x16 (spatial/temporal direct mode).
/// Indices 1-22 = various partition types.
///
/// From FFmpeg `ff_h264_b_mb_type_info` in h264data.c, decoded into a
/// more usable representation.
///
/// Fields: (partition_count, part_size, [[l0, l1]; 2])
/// part_size: 0=16x16, 1=16x8, 2=8x16, 3=8x8
pub const B_MB_TYPE_INFO: [(u8, u8, [[bool; 2]; 2]); 23] = [
    // 0: B_Direct_16x16
    (4, 0, [[true, true], [true, true]]),
    // 1: B_L0_16x16
    (1, 0, [[true, false], [false, false]]),
    // 2: B_L1_16x16
    (1, 0, [[false, true], [false, false]]),
    // 3: B_Bi_16x16
    (1, 0, [[true, true], [false, false]]),
    // 4: B_L0_L0_16x8
    (2, 1, [[true, false], [true, false]]),
    // 5: B_L0_L0_8x16
    (2, 2, [[true, false], [true, false]]),
    // 6: B_L1_L1_16x8
    (2, 1, [[false, true], [false, true]]),
    // 7: B_L1_L1_8x16
    (2, 2, [[false, true], [false, true]]),
    // 8: B_L0_L1_16x8
    (2, 1, [[true, false], [false, true]]),
    // 9: B_L0_L1_8x16
    (2, 2, [[true, false], [false, true]]),
    // 10: B_L1_L0_16x8
    (2, 1, [[false, true], [true, false]]),
    // 11: B_L1_L0_8x16
    (2, 2, [[false, true], [true, false]]),
    // 12: B_L0_Bi_16x8
    (2, 1, [[true, false], [true, true]]),
    // 13: B_L0_Bi_8x16
    (2, 2, [[true, false], [true, true]]),
    // 14: B_L1_Bi_16x8
    (2, 1, [[false, true], [true, true]]),
    // 15: B_L1_Bi_8x16
    (2, 2, [[false, true], [true, true]]),
    // 16: B_Bi_L0_16x8
    (2, 1, [[true, true], [true, false]]),
    // 17: B_Bi_L0_8x16
    (2, 2, [[true, true], [true, false]]),
    // 18: B_Bi_L1_16x8
    (2, 1, [[true, true], [false, true]]),
    // 19: B_Bi_L1_8x16
    (2, 2, [[true, true], [false, true]]),
    // 20: B_Bi_Bi_16x8
    (2, 1, [[true, true], [true, true]]),
    // 21: B_Bi_Bi_8x16
    (2, 2, [[true, true], [true, true]]),
    // 22: B_8x8
    (4, 3, [[true, true], [true, true]]),
];

/// B-slice sub-macroblock type info (for B_8x8 partitions).
///
/// Each entry: (sub_partition_count, sub_part_size, l0, l1)
/// sub_part_size: 0=8x8, 1=8x4, 2=4x8, 3=4x4, 4=direct
///
/// From FFmpeg `ff_h264_b_sub_mb_type_info` in h264data.c.
pub const B_SUB_MB_TYPE_INFO: [(u8, u8, bool, bool); 13] = [
    // 0: B_Direct_8x8
    (4, 4, true, true),
    // 1: B_L0_8x8
    (1, 0, true, false),
    // 2: B_L1_8x8
    (1, 0, false, true),
    // 3: B_Bi_8x8
    (1, 0, true, true),
    // 4: B_L0_8x4
    (2, 1, true, false),
    // 5: B_L0_4x8
    (2, 2, true, false),
    // 6: B_L1_8x4
    (2, 1, false, true),
    // 7: B_L1_4x8
    (2, 2, false, true),
    // 8: B_Bi_8x4
    (2, 1, true, true),
    // 9: B_Bi_4x8
    (2, 2, true, true),
    // 10: B_L0_4x4
    (4, 3, true, false),
    // 11: B_L1_4x4
    (4, 3, false, true),
    // 12: B_Bi_4x4
    (4, 3, true, true),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_4x4_first_and_last() {
        assert_eq!(ZIGZAG_SCAN_4X4[0], 0);
        assert_eq!(ZIGZAG_SCAN_4X4[15], 15);
        // The second element scans right (col 1, row 0)
        assert_eq!(ZIGZAG_SCAN_4X4[1], 1);
        // Then diagonally down-left to (col 0, row 1) = position 4
        assert_eq!(ZIGZAG_SCAN_4X4[2], 4);
    }

    #[test]
    fn zigzag_8x8_first_and_last() {
        assert_eq!(ZIGZAG_SCAN_8X8[0], 0);
        assert_eq!(ZIGZAG_SCAN_8X8[63], 63);
        // Second element: (col 1, row 0) = 1
        assert_eq!(ZIGZAG_SCAN_8X8[1], 1);
        // Third element: (col 0, row 1) = 8
        assert_eq!(ZIGZAG_SCAN_8X8[2], 8);
    }

    #[test]
    fn field_scan_4x4_column_first() {
        // Field scan goes down columns first
        assert_eq!(FIELD_SCAN_4X4[0], 0); // (0,0)
        assert_eq!(FIELD_SCAN_4X4[1], 4); // (0,1)
        assert_eq!(FIELD_SCAN_4X4[2], 1); // (1,0)
        assert_eq!(FIELD_SCAN_4X4[3], 8); // (0,2)
        assert_eq!(FIELD_SCAN_4X4[15], 15); // (3,3)
    }

    #[test]
    fn chroma_qp_identity_below_30() {
        // For QP 0-29, chroma QP equals luma QP
        for qp in 0..30 {
            assert_eq!(CHROMA_QP_TABLE[qp], qp as u8);
        }
    }

    #[test]
    fn chroma_qp_clamps_above_29() {
        // Full table from H.264 spec Table 8-15, verified against FFmpeg CHROMA_QP_TABLE_END(8)
        let expected: [u8; 52] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37,
            37, 38, 38, 38, 39, 39, 39, 39,
        ];
        for i in 0..52 {
            assert_eq!(
                CHROMA_QP_TABLE[i], expected[i],
                "CHROMA_QP_TABLE[{i}]: got {}, expected {}",
                CHROMA_QP_TABLE[i], expected[i]
            );
        }
    }

    #[test]
    fn default_scaling4_intra_first() {
        assert_eq!(DEFAULT_SCALING4[0][0], 6);
        assert_eq!(DEFAULT_SCALING4[0][15], 42);
    }

    #[test]
    fn default_scaling4_inter_first() {
        assert_eq!(DEFAULT_SCALING4[1][0], 10);
        assert_eq!(DEFAULT_SCALING4[1][15], 34);
    }

    #[test]
    fn default_scaling8_intra_corners() {
        assert_eq!(DEFAULT_SCALING8[0][0], 6);
        assert_eq!(DEFAULT_SCALING8[0][63], 42);
    }

    #[test]
    fn default_scaling8_inter_corners() {
        assert_eq!(DEFAULT_SCALING8[1][0], 9);
        assert_eq!(DEFAULT_SCALING8[1][63], 35);
    }

    #[test]
    fn golomb_to_intra4x4_cbp_known_values() {
        // Golomb code 0 maps to CBP 47 (all luma + all chroma)
        assert_eq!(GOLOMB_TO_INTRA4X4_CBP[0], 47);
        // Golomb code 3 maps to CBP 0 (nothing coded)
        assert_eq!(GOLOMB_TO_INTRA4X4_CBP[3], 0);
    }

    #[test]
    fn golomb_to_inter_cbp_known_values() {
        // Golomb code 0 maps to CBP 0 (nothing coded — most common for inter)
        assert_eq!(GOLOMB_TO_INTER_CBP[0], 0);
        // Golomb code 12 maps to CBP 47 (all coded)
        assert_eq!(GOLOMB_TO_INTER_CBP[12], 47);
    }

    #[test]
    fn level_max_dpb_mbs_level31() {
        // Level 3.1 (stored as 31) has 18000 max DPB MBs
        let entry = LEVEL_MAX_DPB_MBS.iter().find(|(l, _)| *l == 31);
        assert_eq!(entry, Some(&(31, 18000)));
    }

    #[test]
    fn level_max_dpb_mbs_level51() {
        // Level 5.1 (stored as 51) has 184320 max DPB MBs
        let entry = LEVEL_MAX_DPB_MBS.iter().find(|(l, _)| *l == 51);
        assert_eq!(entry, Some(&(51, 184320)));
    }

    #[test]
    fn chroma_dc_scan_values() {
        // (0+0*2)*16=0, (1+0*2)*16=16, (0+1*2)*16=32, (1+1*2)*16=48
        assert_eq!(CHROMA_DC_SCAN, [0, 16, 32, 48]);
    }

    #[test]
    fn chroma422_dc_scan_values() {
        assert_eq!(CHROMA422_DC_SCAN.len(), 8);
        assert_eq!(CHROMA422_DC_SCAN[0], 0);
        assert_eq!(CHROMA422_DC_SCAN[1], 32);
        assert_eq!(CHROMA422_DC_SCAN[7], 112);
    }

    #[test]
    fn field_scan_8x8_first_and_last() {
        assert_eq!(FIELD_SCAN_8X8[0], 0); // (0,0)
        assert_eq!(FIELD_SCAN_8X8[1], 8); // (0,1)
        assert_eq!(FIELD_SCAN_8X8[63], 63); // (7,7)
    }
}
