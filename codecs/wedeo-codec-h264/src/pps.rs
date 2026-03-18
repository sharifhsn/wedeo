// H.264 PPS (Picture Parameter Set) parsing.
//
// Reference: ITU-T H.264 Section 7.3.2.2, FFmpeg libavcodec/h264_ps.c
// (`ff_h264_decode_picture_parameter_set`)

use wedeo_codec::bitstream::{BitRead, BitReadBE, get_se_golomb, get_ue_golomb};
use wedeo_core::error::{Error, Result};

use crate::sps::Sps;
use crate::tables::{DEFAULT_SCALING4, DEFAULT_SCALING8, ZIGZAG_SCAN_4X4, ZIGZAG_SCAN_8X8};

/// Maximum PPS count (H.264 spec allows pps_id 0..255).
pub const MAX_PPS_COUNT: usize = 256;

/// Maximum SPS count (H.264 spec allows sps_id 0..31).
pub const MAX_SPS_COUNT: usize = 32;

/// Picture Parameter Set (PPS).
///
/// Reference: ITU-T H.264 Section 7.3.2.2, FFmpeg libavcodec/h264_ps.h
#[derive(Debug, Clone, PartialEq)]
pub struct Pps {
    /// PPS identifier (0..255).
    pub pps_id: u32,
    /// SPS identifier this PPS references (0..31).
    pub sps_id: u32,
    /// Entropy coding mode: false = CAVLC, true = CABAC.
    pub entropy_coding_mode_flag: bool,
    /// bottom_field_pic_order_in_frame_present_flag.
    pub bottom_field_pic_order_in_frame_present: bool,
    /// num_slice_groups_minus1 + 1 (always 1 for Baseline/Main/High).
    pub num_slice_groups: u32,
    /// num_ref_idx_l0_default_active_minus1 + 1.
    pub num_ref_idx_l0_default_active: u32,
    /// num_ref_idx_l1_default_active_minus1 + 1.
    pub num_ref_idx_l1_default_active: u32,
    /// weighted_pred_flag.
    pub weighted_pred_flag: bool,
    /// weighted_bipred_idc (0..2).
    pub weighted_bipred_idc: u8,
    /// pic_init_qp_minus26 + 26 (+ qp_bd_offset for high bit depth).
    pub pic_init_qp: i32,
    /// pic_init_qs_minus26 + 26 (+ qp_bd_offset for high bit depth).
    pub pic_init_qs: i32,
    /// chroma_qp_index_offset[0] from PPS, [1] = second_chroma_qp_index_offset
    /// or copy of [0] if not high profile.
    pub chroma_qp_index_offset: [i32; 2],
    /// deblocking_filter_parameters_present_flag.
    pub deblocking_filter_parameters_present: bool,
    /// constrained_intra_pred_flag.
    pub constrained_intra_pred: bool,
    /// redundant_pic_cnt_present_flag.
    pub redundant_pic_cnt_present: bool,
    /// transform_8x8_mode_flag (only set in High profile and above).
    pub transform_8x8_mode: bool,
    /// 4x4 scaling lists (6 lists of 16 coefficients each).
    pub scaling_matrix4: [[u8; 16]; 6],
    /// 8x8 scaling lists (6 lists of 64 coefficients each).
    pub scaling_matrix8: [[u8; 64]; 6],
    /// Whether pic_scaling_matrix_present_flag was set in the PPS.
    pub scaling_matrix_present: bool,
}

/// Returns true if the SPS profile allows more RBSP data in PPS
/// (transform_8x8_mode, scaling lists, second_chroma_qp_index_offset).
///
/// Baseline (66), Main (77), and Extended (88) profiles with all
/// constraint_set flags set do NOT have additional PPS data.
/// From FFmpeg `more_rbsp_data_in_pps` in h264_ps.c.
fn more_rbsp_data_in_pps(sps: &Sps) -> bool {
    let profile = sps.profile_idc;
    let constraint_flags_low3 = u8::from(sps.constraint_set0_flag)
        | (u8::from(sps.constraint_set1_flag) << 1)
        | (u8::from(sps.constraint_set2_flag) << 2);
    if (profile == 66 || profile == 77 || profile == 88) && constraint_flags_low3 != 0 {
        return false;
    }
    true
}

/// Decode a single scaling list from the bitstream.
///
/// If seq_scaling_list_present_flag is 0, copies the fallback list.
/// If the first delta_scale produces next==0, copies the JVT default list.
/// Otherwise, reads delta_scale values and fills the list via zigzag scan.
///
/// Reference: H.264 spec 7.3.2.1.1, FFmpeg `decode_scaling_list` in h264_ps.c.
fn decode_scaling_list(
    br: &mut BitReadBE<'_>,
    factors: &mut [u8],
    size: usize,
    jvt_list: &[u8],
    fallback_list: &[u8],
) -> Result<bool> {
    let scan: &[u8] = if size == 16 {
        &ZIGZAG_SCAN_4X4
    } else {
        &ZIGZAG_SCAN_8X8
    };

    let present = br.get_bit();
    if !present {
        // Matrix not written, use the predicted (fallback) one.
        factors[..size].copy_from_slice(&fallback_list[..size]);
        return Ok(false);
    }

    let mut last: u8 = 8;
    let mut next: u8 = 8;
    for i in 0..size {
        if next != 0 {
            let delta = get_se_golomb(br)?;
            if !(-128..=127).contains(&delta) {
                return Err(Error::InvalidData);
            }
            next = (last as i32 + delta) as u8; // wraps via & 0xff naturally
        }
        if i == 0 && next == 0 {
            // Matrix not written, use the JVT default one.
            factors[..size].copy_from_slice(&jvt_list[..size]);
            return Ok(true);
        }
        let val = if next != 0 { next } else { last };
        factors[scan[i] as usize] = val;
        last = val;
    }
    Ok(true)
}

/// Decode scaling matrices for PPS.
///
/// Reads up to 6 4x4 lists and (if transform_8x8_mode) up to 6 8x8 lists.
/// Fallback behavior depends on whether the SPS had scaling matrices.
///
/// Reference: FFmpeg `decode_scaling_matrices` in h264_ps.c.
fn decode_scaling_matrices(
    br: &mut BitReadBE<'_>,
    sps: &Sps,
    transform_8x8_mode: bool,
    present_flag: bool,
    scaling_matrix4: &mut [[u8; 16]; 6],
    scaling_matrix8: &mut [[u8; 64]; 6],
) -> Result<()> {
    let fallback_sps = sps.scaling_matrix_present;

    let fallback4_intra: &[u8; 16] = if fallback_sps {
        &sps.scaling_matrix4[0]
    } else {
        &DEFAULT_SCALING4[0]
    };
    let fallback4_inter: &[u8; 16] = if fallback_sps {
        &sps.scaling_matrix4[3]
    } else {
        &DEFAULT_SCALING4[1]
    };
    let fallback8_intra: &[u8; 64] = if fallback_sps {
        &sps.scaling_matrix8[0]
    } else {
        &DEFAULT_SCALING8[0]
    };
    let fallback8_inter: &[u8; 64] = if fallback_sps {
        &sps.scaling_matrix8[3]
    } else {
        &DEFAULT_SCALING8[1]
    };

    if present_flag {
        // 4x4 scaling lists
        decode_scaling_list(
            br,
            &mut scaling_matrix4[0],
            16,
            &DEFAULT_SCALING4[0],
            fallback4_intra,
        )?; // Intra, Y
        let fb1 = scaling_matrix4[0];
        decode_scaling_list(br, &mut scaling_matrix4[1], 16, &DEFAULT_SCALING4[0], &fb1)?; // Intra, Cr
        let fb2 = scaling_matrix4[1];
        decode_scaling_list(br, &mut scaling_matrix4[2], 16, &DEFAULT_SCALING4[0], &fb2)?; // Intra, Cb
        decode_scaling_list(
            br,
            &mut scaling_matrix4[3],
            16,
            &DEFAULT_SCALING4[1],
            fallback4_inter,
        )?; // Inter, Y
        let fb4 = scaling_matrix4[3];
        decode_scaling_list(br, &mut scaling_matrix4[4], 16, &DEFAULT_SCALING4[1], &fb4)?; // Inter, Cr
        let fb5 = scaling_matrix4[4];
        decode_scaling_list(br, &mut scaling_matrix4[5], 16, &DEFAULT_SCALING4[1], &fb5)?; // Inter, Cb

        if transform_8x8_mode {
            // 8x8 scaling lists
            decode_scaling_list(
                br,
                &mut scaling_matrix8[0],
                64,
                &DEFAULT_SCALING8[0],
                fallback8_intra,
            )?; // Intra, Y
            decode_scaling_list(
                br,
                &mut scaling_matrix8[3],
                64,
                &DEFAULT_SCALING8[1],
                fallback8_inter,
            )?; // Inter, Y

            if sps.chroma_format_idc == 3 {
                // 4:4:4 -- separate Cr/Cb scaling lists
                let fb8_0 = scaling_matrix8[0];
                decode_scaling_list(
                    br,
                    &mut scaling_matrix8[1],
                    64,
                    &DEFAULT_SCALING8[0],
                    &fb8_0,
                )?; // Intra, Cr
                let fb8_3 = scaling_matrix8[3];
                decode_scaling_list(
                    br,
                    &mut scaling_matrix8[4],
                    64,
                    &DEFAULT_SCALING8[1],
                    &fb8_3,
                )?; // Inter, Cr
                let fb8_1 = scaling_matrix8[1];
                decode_scaling_list(
                    br,
                    &mut scaling_matrix8[2],
                    64,
                    &DEFAULT_SCALING8[0],
                    &fb8_1,
                )?; // Intra, Cb
                let fb8_4 = scaling_matrix8[4];
                decode_scaling_list(
                    br,
                    &mut scaling_matrix8[5],
                    64,
                    &DEFAULT_SCALING8[1],
                    &fb8_4,
                )?; // Inter, Cb
            }
        }
    }

    Ok(())
}

/// Parse a Picture Parameter Set from raw RBSP data.
///
/// `data` is the PPS NAL unit payload after emulation prevention byte removal
/// (i.e., the RBSP). `sps_list` is the array of previously parsed SPS entries.
///
/// Returns the parsed PPS, or an error if the data is malformed or references
/// an unknown SPS.
///
/// Reference: FFmpeg `ff_h264_decode_picture_parameter_set` in h264_ps.c.
pub fn parse_pps(data: &[u8], sps_list: &[Option<Sps>; 32]) -> Result<Pps> {
    // Pad data for safe bitstream reading (av-bitstream does 8-byte cache refills).
    let mut padded = Vec::with_capacity(data.len() + 8);
    padded.extend_from_slice(data);
    padded.resize(data.len() + 8, 0);
    let mut br = BitReadBE::new(&padded);

    let bit_length = data.len() * 8;

    // pps_id
    let pps_id = get_ue_golomb(&mut br)?;
    if pps_id >= MAX_PPS_COUNT as u32 {
        return Err(Error::InvalidData);
    }

    // sps_id
    let sps_id = get_ue_golomb(&mut br)?;
    if sps_id >= MAX_SPS_COUNT as u32 {
        return Err(Error::InvalidData);
    }

    let sps = sps_list[sps_id as usize]
        .as_ref()
        .ok_or(Error::InvalidData)?;

    // Validate bit depth
    if sps.bit_depth_luma > 14 {
        return Err(Error::InvalidData);
    }
    if sps.bit_depth_luma == 11 || sps.bit_depth_luma == 13 {
        return Err(Error::PatchwelcomeNotImplemented);
    }

    let entropy_coding_mode_flag = br.get_bit();
    let bottom_field_pic_order_in_frame_present = br.get_bit();

    let num_slice_groups_minus1 = get_ue_golomb(&mut br)?;
    let num_slice_groups = num_slice_groups_minus1 + 1;
    if num_slice_groups > 1 {
        // FMO (Flexible Macroblock Ordering) not supported.
        return Err(Error::PatchwelcomeNotImplemented);
    }

    let ref_count_l0 = get_ue_golomb(&mut br)? + 1;
    let ref_count_l1 = get_ue_golomb(&mut br)? + 1;
    if ref_count_l0 > 32 || ref_count_l1 > 32 {
        return Err(Error::InvalidData);
    }

    let qp_bd_offset = 6 * (sps.bit_depth_luma as i32 - 8);

    let weighted_pred_flag = br.get_bit();
    let weighted_bipred_idc = br.get_bits_32(2) as u8;

    let pic_init_qp = get_se_golomb(&mut br)? + 26 + qp_bd_offset;
    let pic_init_qs = get_se_golomb(&mut br)? + 26 + qp_bd_offset;

    let chroma_qp_index_offset_0 = get_se_golomb(&mut br)?;
    if !(-12..=12).contains(&chroma_qp_index_offset_0) {
        return Err(Error::InvalidData);
    }

    let deblocking_filter_parameters_present = br.get_bit();
    let constrained_intra_pred = br.get_bit();
    let redundant_pic_cnt_present = br.get_bit();

    // Initialize scaling matrices from SPS
    let mut scaling_matrix4 = sps.scaling_matrix4;
    let mut scaling_matrix8 = sps.scaling_matrix8;
    let mut transform_8x8_mode = false;
    let mut scaling_matrix_present = false;
    let mut chroma_qp_index_offset_1 = chroma_qp_index_offset_0;

    // Check for more RBSP data (high profile extensions)
    let bits_consumed = br.consumed();
    let bits_left = bit_length.saturating_sub(bits_consumed);

    if bits_left > 0 && more_rbsp_data_in_pps(sps) {
        transform_8x8_mode = br.get_bit();
        let pic_scaling_matrix_present_flag = br.get_bit();
        scaling_matrix_present = pic_scaling_matrix_present_flag;

        decode_scaling_matrices(
            &mut br,
            sps,
            transform_8x8_mode,
            pic_scaling_matrix_present_flag,
            &mut scaling_matrix4,
            &mut scaling_matrix8,
        )?;

        // second_chroma_qp_index_offset
        chroma_qp_index_offset_1 = get_se_golomb(&mut br)?;
        if !(-12..=12).contains(&chroma_qp_index_offset_1) {
            return Err(Error::InvalidData);
        }
    }

    Ok(Pps {
        pps_id,
        sps_id,
        entropy_coding_mode_flag,
        bottom_field_pic_order_in_frame_present,
        num_slice_groups,
        num_ref_idx_l0_default_active: ref_count_l0,
        num_ref_idx_l1_default_active: ref_count_l1,
        weighted_pred_flag,
        weighted_bipred_idc,
        pic_init_qp,
        pic_init_qs,
        chroma_qp_index_offset: [chroma_qp_index_offset_0, chroma_qp_index_offset_1],
        deblocking_filter_parameters_present,
        constrained_intra_pred,
        redundant_pic_cnt_present,
        transform_8x8_mode,
        scaling_matrix4,
        scaling_matrix8,
        scaling_matrix_present,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal SPS for testing PPS parsing.
    fn test_sps() -> Sps {
        Sps {
            profile_idc: 100, // High profile
            ..Sps::default()
        }
    }

    /// Build an SPS list with a single SPS at index 0.
    fn sps_list_with(sps: Sps) -> [Option<Sps>; 32] {
        let mut list: [Option<Sps>; 32] = Default::default();
        let idx = sps.sps_id as usize;
        list[idx] = Some(sps);
        list
    }

    /// Encode a value as unsigned exp-Golomb and push bits to a bit vector.
    fn encode_ue(bits: &mut Vec<bool>, val: u32) {
        let code = val + 1;
        let n = 32 - code.leading_zeros(); // number of bits in (val+1)
        // Leading zeros
        for _ in 0..n - 1 {
            bits.push(false);
        }
        // The code itself (n bits, MSB first)
        for i in (0..n).rev() {
            bits.push((code >> i) & 1 != 0);
        }
    }

    /// Encode a value as signed exp-Golomb and push bits to a bit vector.
    fn encode_se(bits: &mut Vec<bool>, val: i32) {
        let ue_val = if val <= 0 {
            (-2 * val) as u32
        } else {
            (2 * val - 1) as u32
        };
        encode_ue(bits, ue_val);
    }

    /// Convert a bit vector to a byte vector (MSB first, zero-padded).
    fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
        let num_bytes = (bits.len() + 7) / 8;
        let mut bytes = vec![0u8; num_bytes];
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        bytes
    }

    /// Build a minimal Baseline PPS bitstream (no high profile extensions).
    fn build_baseline_pps_bits() -> Vec<u8> {
        let mut bits = Vec::new();

        encode_ue(&mut bits, 0); // pps_id = 0
        encode_ue(&mut bits, 0); // sps_id = 0
        bits.push(false); // entropy_coding_mode_flag = 0 (CAVLC)
        bits.push(false); // bottom_field_pic_order_in_frame_present = 0
        encode_ue(&mut bits, 0); // num_slice_groups_minus1 = 0
        encode_ue(&mut bits, 0); // num_ref_idx_l0_default_active_minus1 = 0
        encode_ue(&mut bits, 0); // num_ref_idx_l1_default_active_minus1 = 0
        bits.push(false); // weighted_pred_flag = 0
        bits.push(false); // weighted_bipred_idc = 0 (2 bits)
        bits.push(false);
        encode_se(&mut bits, 0); // pic_init_qp_minus26 = 0
        encode_se(&mut bits, 0); // pic_init_qs_minus26 = 0
        encode_se(&mut bits, 0); // chroma_qp_index_offset = 0
        bits.push(true); // deblocking_filter_parameters_present = 1
        bits.push(false); // constrained_intra_pred = 0
        bits.push(false); // redundant_pic_cnt_present = 0

        bits_to_bytes(&bits)
    }

    #[test]
    fn parse_baseline_pps() {
        let sps = Sps {
            profile_idc: 66, // Baseline
            constraint_set0_flag: true,
            constraint_set1_flag: true,
            constraint_set2_flag: true,
            ..Sps::default()
        };
        let sps_list = sps_list_with(sps);
        let data = build_baseline_pps_bits();

        let pps = parse_pps(&data, &sps_list).unwrap();

        assert_eq!(pps.pps_id, 0);
        assert_eq!(pps.sps_id, 0);
        assert!(!pps.entropy_coding_mode_flag);
        assert!(!pps.bottom_field_pic_order_in_frame_present);
        assert_eq!(pps.num_slice_groups, 1);
        assert_eq!(pps.num_ref_idx_l0_default_active, 1);
        assert_eq!(pps.num_ref_idx_l1_default_active, 1);
        assert!(!pps.weighted_pred_flag);
        assert_eq!(pps.weighted_bipred_idc, 0);
        assert_eq!(pps.pic_init_qp, 26);
        assert_eq!(pps.pic_init_qs, 26);
        assert_eq!(pps.chroma_qp_index_offset, [0, 0]);
        assert!(pps.deblocking_filter_parameters_present);
        assert!(!pps.constrained_intra_pred);
        assert!(!pps.redundant_pic_cnt_present);
        assert!(!pps.transform_8x8_mode);
    }

    #[test]
    fn parse_high_profile_pps_with_cabac() {
        let sps = test_sps();
        let sps_list = sps_list_with(sps);

        let mut bits = Vec::new();
        encode_ue(&mut bits, 1); // pps_id = 1
        encode_ue(&mut bits, 0); // sps_id = 0
        bits.push(true); // entropy_coding_mode_flag = 1 (CABAC)
        bits.push(false); // bottom_field_pic_order_in_frame_present = 0
        encode_ue(&mut bits, 0); // num_slice_groups_minus1 = 0
        encode_ue(&mut bits, 2); // num_ref_idx_l0_default_active_minus1 = 2 -> 3
        encode_ue(&mut bits, 1); // num_ref_idx_l1_default_active_minus1 = 1 -> 2
        bits.push(true); // weighted_pred_flag = 1
        bits.push(true); // weighted_bipred_idc = 2 (binary 10)
        bits.push(false);
        encode_se(&mut bits, -2); // pic_init_qp_minus26 = -2 -> 24
        encode_se(&mut bits, 0); // pic_init_qs_minus26 = 0 -> 26
        encode_se(&mut bits, -3); // chroma_qp_index_offset = -3
        bits.push(true); // deblocking_filter_parameters_present = 1
        bits.push(true); // constrained_intra_pred = 1
        bits.push(false); // redundant_pic_cnt_present = 0
        // High profile extensions
        bits.push(true); // transform_8x8_mode = 1
        bits.push(false); // pic_scaling_matrix_present_flag = 0
        encode_se(&mut bits, 5); // second_chroma_qp_index_offset = 5

        let data = bits_to_bytes(&bits);
        let pps = parse_pps(&data, &sps_list).unwrap();

        assert_eq!(pps.pps_id, 1);
        assert!(pps.entropy_coding_mode_flag);
        assert_eq!(pps.num_ref_idx_l0_default_active, 3);
        assert_eq!(pps.num_ref_idx_l1_default_active, 2);
        assert!(pps.weighted_pred_flag);
        assert_eq!(pps.weighted_bipred_idc, 2);
        assert_eq!(pps.pic_init_qp, 24);
        assert_eq!(pps.pic_init_qs, 26);
        assert_eq!(pps.chroma_qp_index_offset[0], -3);
        assert_eq!(pps.chroma_qp_index_offset[1], 5);
        assert!(pps.transform_8x8_mode);
        assert!(pps.constrained_intra_pred);
    }

    #[test]
    fn parse_pps_missing_sps() {
        let sps_list: [Option<Sps>; 32] = Default::default();
        let data = build_baseline_pps_bits();
        let result = parse_pps(&data, &sps_list);
        assert!(result.is_err());
    }

    #[test]
    fn parse_pps_invalid_chroma_qp_offset() {
        let sps = test_sps();
        let sps_list = sps_list_with(sps);

        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // pps_id
        encode_ue(&mut bits, 0); // sps_id
        bits.push(false); // entropy_coding_mode_flag
        bits.push(false); // bottom_field_pic_order_in_frame_present
        encode_ue(&mut bits, 0); // num_slice_groups_minus1
        encode_ue(&mut bits, 0); // ref l0
        encode_ue(&mut bits, 0); // ref l1
        bits.push(false); // weighted_pred_flag
        bits.push(false); // weighted_bipred_idc (2 bits)
        bits.push(false);
        encode_se(&mut bits, 0); // pic_init_qp_minus26
        encode_se(&mut bits, 0); // pic_init_qs_minus26
        encode_se(&mut bits, 13); // chroma_qp_index_offset = 13 (out of range)

        let data = bits_to_bytes(&bits);
        let result = parse_pps(&data, &sps_list);
        assert_eq!(result, Err(Error::InvalidData));
    }

    #[test]
    fn parse_pps_ref_count_overflow() {
        let sps = test_sps();
        let sps_list = sps_list_with(sps);

        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // pps_id
        encode_ue(&mut bits, 0); // sps_id
        bits.push(false); // entropy_coding_mode_flag
        bits.push(false); // bottom_field_pic_order_in_frame_present
        encode_ue(&mut bits, 0); // num_slice_groups_minus1
        encode_ue(&mut bits, 32); // num_ref_idx_l0 = 33 (overflow!)
        encode_ue(&mut bits, 0); // ref l1

        let data = bits_to_bytes(&bits);
        let result = parse_pps(&data, &sps_list);
        assert_eq!(result, Err(Error::InvalidData));
    }

    #[test]
    fn baseline_profile_no_high_extensions() {
        // When Baseline profile with constraint_set flags, second chroma QP offset
        // should equal the first (no high profile RBSP data read).
        let sps = Sps {
            profile_idc: 66,
            constraint_set0_flag: true,
            constraint_set1_flag: true,
            constraint_set2_flag: true,
            ..Sps::default()
        };
        let sps_list = sps_list_with(sps);

        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // pps_id
        encode_ue(&mut bits, 0); // sps_id
        bits.push(false); // entropy_coding_mode_flag
        bits.push(false); // bottom_field_pic_order_in_frame_present
        encode_ue(&mut bits, 0); // num_slice_groups_minus1
        encode_ue(&mut bits, 0); // ref l0
        encode_ue(&mut bits, 0); // ref l1
        bits.push(false); // weighted_pred_flag
        bits.push(false); // weighted_bipred_idc (2 bits)
        bits.push(false);
        encode_se(&mut bits, 0); // pic_init_qp_minus26
        encode_se(&mut bits, 0); // pic_init_qs_minus26
        encode_se(&mut bits, -5); // chroma_qp_index_offset = -5
        bits.push(false); // deblocking_filter_parameters_present
        bits.push(false); // constrained_intra_pred
        bits.push(false); // redundant_pic_cnt_present
        // Extra bits follow but should be ignored for Baseline with constraint flags

        let data = bits_to_bytes(&bits);
        let pps = parse_pps(&data, &sps_list).unwrap();

        // Both offsets should be -5 (second copied from first)
        assert_eq!(pps.chroma_qp_index_offset[0], -5);
        assert_eq!(pps.chroma_qp_index_offset[1], -5);
        assert!(!pps.transform_8x8_mode);
    }
}
