//! H.264 Sequence Parameter Set (SPS) parsing.
//!
//! Implements parsing of the SPS NAL unit as specified in ITU-T H.264
//! section 7.3.2.1. The parsing logic follows FFmpeg's
//! `ff_h264_decode_seq_parameter_set` in `libavcodec/h264_ps.c`.

use wedeo_codec::bitstream::{BitRead, BitReadBE, get_se_golomb, get_ue_golomb};
use wedeo_core::rational::Rational;
use wedeo_core::{Error, Result};

use crate::tables::{DEFAULT_SCALING4, DEFAULT_SCALING8, ZIGZAG_SCAN_4X4, ZIGZAG_SCAN_8X8};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_SPS_COUNT: u32 = 32;
const MIN_LOG2_MAX_FRAME_NUM: u32 = 4;
const MAX_LOG2_MAX_FRAME_NUM: u32 = 16; // 12 + 4
const EXTENDED_SAR: u32 = 255;
const MAX_POC_CYCLE_LENGTH: usize = 256;

/// SAR aspect ratio table (H.264 Table E-1).
/// Index 0 is "Unspecified", indices 1..=16 are defined, 255 is Extended_SAR.
const SAR_TABLE: [(i32, i32); 17] = [
    (0, 1),    // 0: Unspecified
    (1, 1),    // 1: 1:1
    (12, 11),  // 2: 12:11
    (10, 11),  // 3: 10:11
    (16, 11),  // 4: 16:11
    (40, 33),  // 5: 40:33
    (24, 11),  // 6: 24:11
    (20, 11),  // 7: 20:11
    (32, 11),  // 8: 32:11
    (80, 33),  // 9: 80:33
    (18, 11),  // 10: 18:11
    (15, 11),  // 11: 15:11
    (64, 33),  // 12: 64:33
    (160, 99), // 13: 160:99
    (4, 3),    // 14: 4:3
    (3, 2),    // 15: 3:2
    (2, 1),    // 16: 2:1
];

// ---------------------------------------------------------------------------
// High-profile profile_idc values that trigger extended SPS fields
// ---------------------------------------------------------------------------

/// Returns true if this profile_idc requires parsing the high-profile
/// extensions (chroma_format_idc, bit_depth, scaling lists, etc.).
fn is_high_profile(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100  // High
        | 110 // High10
        | 122 // High422
        | 244 // High444 Predictive
        | 44  // Cavlc444
        | 83  // Scalable Constrained High (SVC)
        | 86  // Scalable High Intra (SVC)
        | 118 // Stereo High (MVC)
        | 128 // Multiview High (MVC)
        | 138 // Multiview Depth High (MVCD)
        | 139 // 3D-AVC High (Amendment 8)
        | 134 // MFC Depth High (Amendment 9)
        | 135 // Enhanced Multiview Depth High (Amendment 10)
        | 144 // old High444
    )
}

// ---------------------------------------------------------------------------
// SPS struct
// ---------------------------------------------------------------------------

/// H.264 Sequence Parameter Set.
#[derive(Debug, Clone, PartialEq)]
pub struct Sps {
    // Identity
    pub sps_id: u32,
    pub profile_idc: u8,
    pub level_idc: u8,

    // Constraint set flags
    pub constraint_set0_flag: bool,
    pub constraint_set1_flag: bool,
    pub constraint_set2_flag: bool,
    pub constraint_set3_flag: bool,
    pub constraint_set4_flag: bool,
    pub constraint_set5_flag: bool,

    // Chroma / bit depth
    pub chroma_format_idc: u8,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub residual_color_transform_flag: bool,
    pub transform_bypass: bool,

    // Frame numbering
    pub log2_max_frame_num: u8,

    // Picture order count
    pub poc_type: u8,
    // poc_type == 0
    pub log2_max_poc_lsb: u8,
    // poc_type == 1
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub num_ref_frames_in_poc_cycle: u8,
    pub offset_for_ref_frame: Vec<i32>,

    // Reference frames
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_allowed: bool,

    // Dimensions in macroblocks
    /// pic_width_in_mbs_minus1 + 1
    pub mb_width: u32,
    /// (pic_height_in_map_units_minus1 + 1) * (2 - frame_mbs_only_flag)
    pub mb_height: u32,
    pub frame_mbs_only_flag: bool,
    /// mb_adaptive_frame_field_flag (only meaningful when !frame_mbs_only_flag)
    pub mb_aff: bool,
    pub direct_8x8_inference_flag: bool,

    // Cropping (in luma samples, already scaled by SubWidth/SubHeight)
    pub crop_left: u32,
    pub crop_right: u32,
    pub crop_top: u32,
    pub crop_bottom: u32,

    // VUI
    pub vui_parameters_present: bool,
    pub timing_info_present: bool,
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub fixed_frame_rate: bool,
    pub sar: Rational,

    // VUI bitstream restriction
    /// Number of frames that need reordering (from VUI bitstream_restriction).
    /// -1 means not signalled (infer from profile/level).
    pub num_reorder_frames: i32,

    // VUI video signal type
    pub video_signal_type_present: bool,
    pub video_format: u8,
    pub video_full_range_flag: bool,
    pub colour_description_present: bool,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,

    // Scaling lists
    pub scaling_matrix4: [[u8; 16]; 6],
    pub scaling_matrix8: [[u8; 64]; 6],
    pub scaling_matrix_present: bool,
}

impl Default for Sps {
    fn default() -> Self {
        // Initialize scaling matrices to flat 16 (matching FFmpeg's memset(..., 16, ...))
        Self {
            sps_id: 0,
            profile_idc: 0,
            level_idc: 0,
            constraint_set0_flag: false,
            constraint_set1_flag: false,
            constraint_set2_flag: false,
            constraint_set3_flag: false,
            constraint_set4_flag: false,
            constraint_set5_flag: false,
            chroma_format_idc: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            residual_color_transform_flag: false,
            transform_bypass: false,
            log2_max_frame_num: 0,
            poc_type: 0,
            log2_max_poc_lsb: 0,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            num_ref_frames_in_poc_cycle: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 0,
            gaps_in_frame_num_allowed: false,
            mb_width: 0,
            mb_height: 0,
            frame_mbs_only_flag: true,
            mb_aff: false,
            direct_8x8_inference_flag: false,
            crop_left: 0,
            crop_right: 0,
            crop_top: 0,
            crop_bottom: 0,
            vui_parameters_present: false,
            timing_info_present: false,
            num_units_in_tick: 0,
            time_scale: 0,
            fixed_frame_rate: false,
            sar: Rational::new(0, 1),
            num_reorder_frames: -1,
            video_signal_type_present: false,
            video_format: 5, // "Unspecified" per H.264 spec
            video_full_range_flag: false,
            colour_description_present: false,
            colour_primaries: 2,         // "Unspecified"
            transfer_characteristics: 2, // "Unspecified"
            matrix_coefficients: 2,      // "Unspecified"
            scaling_matrix4: [[16; 16]; 6],
            scaling_matrix8: [[16; 64]; 6],
            scaling_matrix_present: false,
        }
    }
}

impl Sps {
    /// Pixel width after cropping.
    pub fn width(&self) -> u32 {
        self.mb_width * 16 - self.crop_left - self.crop_right
    }

    /// Pixel height after cropping.
    pub fn height(&self) -> u32 {
        self.mb_height * 16 - self.crop_top - self.crop_bottom
    }

    /// Frame rate derived from VUI timing info, if present.
    ///
    /// H.264 specifies the field rate as `time_scale / num_units_in_tick`.
    /// For progressive content the frame rate is `time_scale / (2 * num_units_in_tick)`.
    /// We always return the frame-level rate (dividing by 2).
    pub fn frame_rate(&self) -> Option<Rational> {
        if self.timing_info_present && self.num_units_in_tick > 0 && self.time_scale > 0 {
            Some(Rational::new(
                self.time_scale as i32,
                (self.num_units_in_tick * 2) as i32,
            ))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Scaling list parsing
// ---------------------------------------------------------------------------

/// Decode a single scaling list from the bitstream.
///
/// Matches FFmpeg's `decode_scaling_list` in `h264_ps.c`.
///
/// - `factors`: output scaling list (length `size`, either 16 or 64)
/// - `size`: 16 for 4x4, 64 for 8x8
/// - `jvt_list`: default JVT scaling list (used when "use default" is signalled)
/// - `fallback_list`: fallback list used when seq_scaling_list_present_flag == 0
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
        // Matrix not written -- use the fallback (previous list or default).
        factors[..size].copy_from_slice(&fallback_list[..size]);
        return Ok(false);
    }

    let mut last: i32 = 8;
    let mut next: i32 = 8;
    for i in 0..size {
        if next != 0 {
            let delta = get_se_golomb(br)?;
            if !(-128..=127).contains(&delta) {
                return Err(Error::InvalidData);
            }
            next = (last + delta) & 0xff;
        }
        if i == 0 && next == 0 {
            // "use default scaling matrix" signal
            factors[..size].copy_from_slice(&jvt_list[..size]);
            return Ok(true);
        }
        let val = if next != 0 { next } else { last };
        factors[scan[i] as usize] = val as u8;
        last = val;
    }
    Ok(true)
}

/// Decode all scaling matrices for an SPS.
///
/// Matches FFmpeg's `decode_scaling_matrices` (SPS path: `is_sps=1`).
fn decode_sps_scaling_matrices(
    br: &mut BitReadBE<'_>,
    chroma_format_idc: u8,
    scaling_matrix4: &mut [[u8; 16]; 6],
    scaling_matrix8: &mut [[u8; 64]; 6],
) -> Result<bool> {
    let present = br.get_bit();
    if !present {
        return Ok(false);
    }

    // 4x4 matrices
    // Intra Y -- fallback is default_scaling4[0]
    decode_scaling_list(
        br,
        &mut scaling_matrix4[0],
        16,
        &DEFAULT_SCALING4[0],
        &DEFAULT_SCALING4[0],
    )?;
    // Intra Cr -- fallback is scaling_matrix4[0]
    let fallback1 = scaling_matrix4[0];
    decode_scaling_list(
        br,
        &mut scaling_matrix4[1],
        16,
        &DEFAULT_SCALING4[0],
        &fallback1,
    )?;
    // Intra Cb -- fallback is scaling_matrix4[1]
    let fallback2 = scaling_matrix4[1];
    decode_scaling_list(
        br,
        &mut scaling_matrix4[2],
        16,
        &DEFAULT_SCALING4[0],
        &fallback2,
    )?;
    // Inter Y -- fallback is default_scaling4[1]
    decode_scaling_list(
        br,
        &mut scaling_matrix4[3],
        16,
        &DEFAULT_SCALING4[1],
        &DEFAULT_SCALING4[1],
    )?;
    // Inter Cr -- fallback is scaling_matrix4[3]
    let fallback4 = scaling_matrix4[3];
    decode_scaling_list(
        br,
        &mut scaling_matrix4[4],
        16,
        &DEFAULT_SCALING4[1],
        &fallback4,
    )?;
    // Inter Cb -- fallback is scaling_matrix4[4]
    let fallback5 = scaling_matrix4[4];
    decode_scaling_list(
        br,
        &mut scaling_matrix4[5],
        16,
        &DEFAULT_SCALING4[1],
        &fallback5,
    )?;

    // 8x8 matrices (always present for SPS when scaling_matrix_present_flag is set)
    // Intra Y
    decode_scaling_list(
        br,
        &mut scaling_matrix8[0],
        64,
        &DEFAULT_SCALING8[0],
        &DEFAULT_SCALING8[0],
    )?;
    // Inter Y
    decode_scaling_list(
        br,
        &mut scaling_matrix8[3],
        64,
        &DEFAULT_SCALING8[1],
        &DEFAULT_SCALING8[1],
    )?;

    if chroma_format_idc == 3 {
        // 4:4:4 -- additional Cr/Cb 8x8 matrices
        let fallback_intra = scaling_matrix8[0];
        decode_scaling_list(
            br,
            &mut scaling_matrix8[1],
            64,
            &DEFAULT_SCALING8[0],
            &fallback_intra,
        )?;
        let fallback_inter = scaling_matrix8[3];
        decode_scaling_list(
            br,
            &mut scaling_matrix8[4],
            64,
            &DEFAULT_SCALING8[1],
            &fallback_inter,
        )?;
        let fallback_intra2 = scaling_matrix8[1];
        decode_scaling_list(
            br,
            &mut scaling_matrix8[2],
            64,
            &DEFAULT_SCALING8[0],
            &fallback_intra2,
        )?;
        let fallback_inter2 = scaling_matrix8[4];
        decode_scaling_list(
            br,
            &mut scaling_matrix8[5],
            64,
            &DEFAULT_SCALING8[1],
            &fallback_inter2,
        )?;
    }

    Ok(true)
}

// ---------------------------------------------------------------------------
// VUI parsing
// ---------------------------------------------------------------------------

/// Parse VUI parameters from the bitstream.
///
/// Follows FFmpeg's `decode_vui_parameters` and `ff_h2645_decode_common_vui_params`.
fn parse_vui(br: &mut BitReadBE<'_>, sps: &mut Sps) -> Result<()> {
    // -- Common VUI params (ff_h2645_decode_common_vui_params) --

    // aspect_ratio_info
    let aspect_ratio_info_present = br.get_bit();
    if aspect_ratio_info_present {
        let aspect_ratio_idc = br.get_bits_32(8);
        if (aspect_ratio_idc as usize) < SAR_TABLE.len() {
            let (num, den) = SAR_TABLE[aspect_ratio_idc as usize];
            sps.sar = Rational::new(num, den);
        } else if aspect_ratio_idc == EXTENDED_SAR {
            let sar_width = br.get_bits_32(16) as i32;
            let sar_height = br.get_bits_32(16) as i32;
            sps.sar = Rational::new(sar_width, sar_height);
        }
        // else: unknown SAR index, leave as default (0/1)
    } else {
        sps.sar = Rational::new(0, 1);
    }

    // overscan_info
    let overscan_info_present = br.get_bit();
    if overscan_info_present {
        let _overscan_appropriate = br.get_bit();
    }

    // video_signal_type
    sps.video_signal_type_present = br.get_bit();
    if sps.video_signal_type_present {
        sps.video_format = br.get_bits_32(3) as u8;
        sps.video_full_range_flag = br.get_bit();
        sps.colour_description_present = br.get_bit();
        if sps.colour_description_present {
            sps.colour_primaries = br.get_bits_32(8) as u8;
            sps.transfer_characteristics = br.get_bits_32(8) as u8;
            sps.matrix_coefficients = br.get_bits_32(8) as u8;
        }
    }

    // chroma_loc_info
    let chroma_loc_info_present = br.get_bit();
    if chroma_loc_info_present {
        let _chroma_sample_loc_type_top = get_ue_golomb(br)?;
        let _chroma_sample_loc_type_bottom = get_ue_golomb(br)?;
    }

    // -- H.264-specific VUI params --

    // timing_info
    sps.timing_info_present = br.get_bit();
    if sps.timing_info_present {
        let num_units_in_tick = br.get_bits_32(32);
        let time_scale = br.get_bits_32(32);
        if num_units_in_tick == 0 || time_scale == 0 {
            // Invalid timing info -- mark as not present (matching FFmpeg behavior).
            sps.timing_info_present = false;
        } else {
            sps.num_units_in_tick = num_units_in_tick;
            sps.time_scale = time_scale;
        }
        sps.fixed_frame_rate = br.get_bit();
    }

    // nal_hrd_parameters
    let nal_hrd_present = br.get_bit();
    if nal_hrd_present {
        skip_hrd_parameters(br)?;
    }

    // vcl_hrd_parameters
    let vcl_hrd_present = br.get_bit();
    if vcl_hrd_present {
        skip_hrd_parameters(br)?;
    }

    if nal_hrd_present || vcl_hrd_present {
        let _low_delay_hrd = br.get_bit();
    }

    let _pic_struct_present = br.get_bit();

    // bitstream_restriction
    let bitstream_restriction = br.get_bit();
    if bitstream_restriction {
        let _motion_vectors_over_pic_boundaries = br.get_bit();
        let _max_bytes_per_pic_denom = get_ue_golomb(br)?;
        let _max_bits_per_mb_denom = get_ue_golomb(br)?;
        let _log2_max_mv_length_horizontal = get_ue_golomb(br)?;
        let _log2_max_mv_length_vertical = get_ue_golomb(br)?;
        let num_reorder_frames = get_ue_golomb(br)?;
        let _max_dec_frame_buffering = get_ue_golomb(br)?;
        sps.num_reorder_frames = num_reorder_frames.min(16) as i32;
    }

    Ok(())
}

/// Skip HRD parameters in the bitstream.
///
/// Matches FFmpeg's `decode_hrd_parameters` but discards the values.
fn skip_hrd_parameters(br: &mut BitReadBE<'_>) -> Result<()> {
    let cpb_count = get_ue_golomb(br)? + 1;
    if cpb_count > 32 {
        return Err(Error::InvalidData);
    }
    let _bit_rate_scale = br.get_bits_32(4);
    let _cpb_size_scale = br.get_bits_32(4);
    for _ in 0..cpb_count {
        let _bit_rate_value = get_ue_golomb(br)?;
        let _cpb_size_value = get_ue_golomb(br)?;
        let _cbr_flag = br.get_bit();
    }
    let _initial_cpb_removal_delay_length = br.get_bits_32(5);
    let _cpb_removal_delay_length = br.get_bits_32(5);
    let _dpb_output_delay_length = br.get_bits_32(5);
    let _time_offset_length = br.get_bits_32(5);
    Ok(())
}

// ---------------------------------------------------------------------------
// Main SPS parser
// ---------------------------------------------------------------------------

/// Parse an H.264 SPS from RBSP data (after NAL header byte is stripped).
///
/// The input `data` should be the raw SPS bytes with emulation prevention
/// bytes already removed (RBSP), starting at `profile_idc`.
///
/// # Errors
///
/// Returns `Error::InvalidData` if the SPS is malformed:
/// - `sps_id` >= 32
/// - Invalid `log2_max_frame_num_minus4` (must be 0..=12)
/// - Invalid `poc_type` (must be 0, 1, or 2)
/// - Bit depth out of range
/// - Crop values exceeding frame dimensions
// Fields are assigned sequentially from the bitstream, so struct-init syntax
// with `..Default::default()` would be less readable than incremental assignment.
#[allow(clippy::field_reassign_with_default)]
pub fn parse_sps(data: &[u8]) -> Result<Sps> {
    // The bitstream reader does 8-byte cache reads, so we need to ensure
    // the data is padded to avoid out-of-bounds reads. We pad to at least
    // data.len() + 8 bytes with zeros.
    let mut padded = Vec::with_capacity(data.len() + 8);
    padded.extend_from_slice(data);
    padded.resize(data.len() + 8, 0);

    let mut br = BitReadBE::new(&padded);
    let mut sps = Sps::default();

    // 1. Profile, constraint flags, level
    sps.profile_idc = br.get_bits_32(8) as u8;
    sps.constraint_set0_flag = br.get_bit();
    sps.constraint_set1_flag = br.get_bit();
    sps.constraint_set2_flag = br.get_bit();
    sps.constraint_set3_flag = br.get_bit();
    sps.constraint_set4_flag = br.get_bit();
    sps.constraint_set5_flag = br.get_bit();
    br.skip_bits(2); // reserved_zero_2bits
    sps.level_idc = br.get_bits_32(8) as u8;

    // 2. sps_id
    sps.sps_id = get_ue_golomb(&mut br)?;
    if sps.sps_id >= MAX_SPS_COUNT {
        return Err(Error::InvalidData);
    }

    // 3. High profile extensions
    if is_high_profile(sps.profile_idc) {
        sps.chroma_format_idc = get_ue_golomb(&mut br)? as u8;
        if sps.chroma_format_idc > 3 {
            return Err(Error::InvalidData);
        }
        if sps.chroma_format_idc == 3 {
            sps.residual_color_transform_flag = br.get_bit();
            if sps.residual_color_transform_flag {
                // Separate color planes not supported
                return Err(Error::InvalidData);
            }
        }

        let bit_depth_luma_minus8 = get_ue_golomb(&mut br)?;
        let bit_depth_chroma_minus8 = get_ue_golomb(&mut br)?;
        sps.bit_depth_luma = (bit_depth_luma_minus8 + 8) as u8;
        sps.bit_depth_chroma = (bit_depth_chroma_minus8 + 8) as u8;

        if sps.bit_depth_luma < 8
            || sps.bit_depth_luma > 14
            || sps.bit_depth_chroma < 8
            || sps.bit_depth_chroma > 14
        {
            return Err(Error::InvalidData);
        }

        sps.transform_bypass = br.get_bit();

        // Scaling matrices
        sps.scaling_matrix_present = decode_sps_scaling_matrices(
            &mut br,
            sps.chroma_format_idc,
            &mut sps.scaling_matrix4,
            &mut sps.scaling_matrix8,
        )?;
    } else {
        sps.chroma_format_idc = 1;
        sps.bit_depth_luma = 8;
        sps.bit_depth_chroma = 8;
    }

    // 4. log2_max_frame_num
    let log2_max_frame_num_minus4 = get_ue_golomb(&mut br)?;
    if log2_max_frame_num_minus4 > MAX_LOG2_MAX_FRAME_NUM - MIN_LOG2_MAX_FRAME_NUM {
        return Err(Error::InvalidData);
    }
    sps.log2_max_frame_num = (log2_max_frame_num_minus4 + 4) as u8;

    // 5. POC type
    sps.poc_type = get_ue_golomb(&mut br)? as u8;
    match sps.poc_type {
        0 => {
            let log2_max_poc_lsb_minus4 = get_ue_golomb(&mut br)?;
            if log2_max_poc_lsb_minus4 > 12 {
                return Err(Error::InvalidData);
            }
            sps.log2_max_poc_lsb = (log2_max_poc_lsb_minus4 + 4) as u8;
        }
        1 => {
            sps.delta_pic_order_always_zero_flag = br.get_bit();
            sps.offset_for_non_ref_pic = get_se_golomb(&mut br)?;
            sps.offset_for_top_to_bottom_field = get_se_golomb(&mut br)?;

            let poc_cycle_length = get_ue_golomb(&mut br)? as usize;
            if poc_cycle_length >= MAX_POC_CYCLE_LENGTH {
                return Err(Error::InvalidData);
            }
            sps.num_ref_frames_in_poc_cycle = poc_cycle_length as u8;

            sps.offset_for_ref_frame = Vec::with_capacity(poc_cycle_length);
            for _ in 0..poc_cycle_length {
                sps.offset_for_ref_frame.push(get_se_golomb(&mut br)?);
            }
        }
        2 => {
            // No additional data for poc_type 2
        }
        _ => {
            return Err(Error::InvalidData);
        }
    }

    // 6. Reference frames
    sps.max_num_ref_frames = get_ue_golomb(&mut br)?;
    if sps.max_num_ref_frames > 16 {
        return Err(Error::InvalidData);
    }
    sps.gaps_in_frame_num_allowed = br.get_bit();

    // 7. Dimensions
    let pic_width_in_mbs_minus1 = get_ue_golomb(&mut br)?;
    let pic_height_in_map_units_minus1 = get_ue_golomb(&mut br)?;
    sps.mb_width = pic_width_in_mbs_minus1 + 1;

    // 8. frame_mbs_only, mb_aff
    sps.frame_mbs_only_flag = br.get_bit();
    // mb_height = (pic_height_in_map_units_minus1 + 1) * (2 - frame_mbs_only_flag)
    let mbs_only_mult: u32 = if sps.frame_mbs_only_flag { 1 } else { 2 };
    sps.mb_height = (pic_height_in_map_units_minus1 + 1) * mbs_only_mult;

    if !sps.frame_mbs_only_flag {
        sps.mb_aff = br.get_bit();
    }

    sps.direct_8x8_inference_flag = br.get_bit();

    // 9. Cropping
    let crop_flag = br.get_bit();
    if crop_flag {
        let crop_left = get_ue_golomb(&mut br)?;
        let crop_right = get_ue_golomb(&mut br)?;
        let crop_top = get_ue_golomb(&mut br)?;
        let crop_bottom = get_ue_golomb(&mut br)?;

        // SubWidthC and SubHeightC depend on chroma_format_idc
        let vsub: u32 = u32::from(sps.chroma_format_idc == 1);
        let hsub: u32 = u32::from(sps.chroma_format_idc == 1 || sps.chroma_format_idc == 2);
        let step_x: u32 = 1 << hsub;
        let step_y: u32 = (2 - u32::from(sps.frame_mbs_only_flag)) << vsub;

        let width = sps.mb_width * 16;
        let height = sps.mb_height * 16;

        // Validate crop values don't exceed frame dimensions
        if (crop_left + crop_right) * step_x >= width || (crop_top + crop_bottom) * step_y >= height
        {
            return Err(Error::InvalidData);
        }

        sps.crop_left = crop_left * step_x;
        sps.crop_right = crop_right * step_x;
        sps.crop_top = crop_top * step_y;
        sps.crop_bottom = crop_bottom * step_y;
    }

    // 10. VUI
    sps.vui_parameters_present = br.get_bit();
    if sps.vui_parameters_present {
        parse_vui(&mut br, &mut sps)?;
    }

    // Ensure SAR denominator is non-zero
    if sps.sar.den == 0 {
        sps.sar.den = 1;
    }

    Ok(sps)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helper method tests --

    #[test]
    fn test_width_height_no_crop() {
        let sps = Sps {
            mb_width: 120, // 1920 / 16
            mb_height: 68, // 1088 / 16 (not 1080 -- padded to MB boundary)
            frame_mbs_only_flag: true,
            ..Default::default()
        };
        assert_eq!(sps.width(), 1920);
        assert_eq!(sps.height(), 1088);
    }

    #[test]
    fn test_width_height_with_crop() {
        // 1920x1080 encoded as 120x68 MBs with bottom crop of 8
        let sps = Sps {
            mb_width: 120,
            mb_height: 68,
            frame_mbs_only_flag: true,
            crop_bottom: 8,
            ..Default::default()
        };
        assert_eq!(sps.width(), 1920);
        assert_eq!(sps.height(), 1080);
    }

    #[test]
    fn test_width_height_interlaced() {
        // 1920x1080i: pic_height_in_map_units = 34, frame_mbs_only=false
        // mb_height = 34 * 2 = 68
        let sps = Sps {
            mb_width: 120,
            mb_height: 68,
            frame_mbs_only_flag: false,
            crop_bottom: 8,
            ..Default::default()
        };
        assert_eq!(sps.width(), 1920);
        assert_eq!(sps.height(), 1080);
    }

    #[test]
    fn test_frame_rate() {
        // 30000/1001 fps (NTSC 29.97): time_scale=60000, num_units_in_tick=1001
        let sps = Sps {
            timing_info_present: true,
            time_scale: 60000,
            num_units_in_tick: 1001,
            ..Default::default()
        };
        let rate = sps.frame_rate().unwrap();
        assert_eq!(rate.num, 60000);
        assert_eq!(rate.den, 2002);
    }

    #[test]
    fn test_frame_rate_none() {
        let sps = Sps::default();
        assert!(sps.frame_rate().is_none());
    }

    #[test]
    fn test_frame_rate_25fps() {
        // 25 fps: time_scale=50, num_units_in_tick=1
        let sps = Sps {
            timing_info_present: true,
            time_scale: 50,
            num_units_in_tick: 1,
            ..Default::default()
        };
        let rate = sps.frame_rate().unwrap();
        assert_eq!(rate.num, 50);
        assert_eq!(rate.den, 2);
    }

    // -- SAR table tests --

    #[test]
    fn test_sar_table_values() {
        assert_eq!(SAR_TABLE[0], (0, 1));
        assert_eq!(SAR_TABLE[1], (1, 1));
        assert_eq!(SAR_TABLE[2], (12, 11));
        assert_eq!(SAR_TABLE[13], (160, 99));
        assert_eq!(SAR_TABLE[14], (4, 3));
        assert_eq!(SAR_TABLE[16], (2, 1));
    }

    // -- Scaling list tests --

    #[test]
    fn test_decode_scaling_list_not_present() {
        // When present flag is 0 (first bit = 0), use fallback list.
        let mut data = vec![0x00u8]; // bit 0 = 0
        data.resize(data.len() + 8, 0);
        let mut br = BitReadBE::new(&data);
        let mut factors = [0u8; 16];
        let fallback = [1u8; 16];
        let jvt = [2u8; 16];
        let present = decode_scaling_list(&mut br, &mut factors, 16, &jvt, &fallback).unwrap();
        assert!(!present);
        assert_eq!(factors, [1u8; 16]);
    }

    #[test]
    fn test_decode_scaling_list_use_default() {
        // When present flag is 1 and the first delta_scale produces next=0
        // at i=0, use jvt_list.
        //
        // present_flag = 1 (1 bit)
        // delta_scale se(v) = -8 to make next = (8 + (-8)) & 0xff = 0
        //   se(-8) -> ue(16): 16+1 = 17 = 0b10001 (5 bits), prefix = 4 zeros
        //   => 0000_10001 (9 bits total for ue(16))
        //
        // Total: 1 bit (present) + 9 bits (se) = 10 bits
        // Bit sequence: 1_0000_10001
        // Byte 0 (bits 0-7): 1000_0100 = 0x84
        // Byte 1 (bits 8-9): 01xx_xxxx = 0x40
        let mut data = vec![0x84u8, 0x40];
        data.resize(data.len() + 8, 0);
        let mut br = BitReadBE::new(&data);
        let mut factors = [0u8; 16];
        let jvt = DEFAULT_SCALING4[0];
        let fallback = [99u8; 16];
        let present = decode_scaling_list(&mut br, &mut factors, 16, &jvt, &fallback).unwrap();
        assert!(present);
        assert_eq!(factors, DEFAULT_SCALING4[0]);
    }

    // -- Default values test --

    #[test]
    fn test_default_sps() {
        let sps = Sps::default();
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.bit_depth_luma, 8);
        assert_eq!(sps.bit_depth_chroma, 8);
        assert!(sps.frame_mbs_only_flag);
        assert!(!sps.mb_aff);
        // Scaling matrices default to flat 16
        assert_eq!(sps.scaling_matrix4[0][0], 16);
        assert_eq!(sps.scaling_matrix8[0][0], 16);
    }

    // -- Baseline SPS parsing test --

    #[test]
    fn test_parse_baseline_sps() {
        // Hand-crafted Baseline SPS for 320x240, 30fps.
        //
        // profile_idc = 66 (Baseline), constraint_set0 = 1, level = 30
        // sps_id = 0, log2_max_frame_num_minus4 = 0, poc_type = 0,
        // log2_max_poc_lsb_minus4 = 0, max_num_ref_frames = 1, gaps = 0,
        // pic_width_in_mbs_minus1 = 19 (320/16 - 1),
        // pic_height_in_map_units_minus1 = 14 (240/16 - 1),
        // frame_mbs_only = 1, direct_8x8 = 1, crop = 0, vui = 0
        //
        // Bit layout:
        // Byte 0: 0x42 = profile_idc 66
        // Byte 1: 0x80 = constraint_set0=1, rest=0, reserved=00
        // Byte 2: 0x1E = level_idc 30
        // Byte 3: 0xF4 = ue(0) ue(0) ue(0) ue(0) ue(1)=010 gaps=0
        // Byte 4: 0x0A = ue(19)=000010100 starts here
        // Byte 5: 0x0F = ...finish ue(19), ue(14)=0001111 starts
        // Byte 6: 0xC0 = frame_mbs_only=1 direct_8x8=1 crop=0 vui=0

        let data: &[u8] = &[0x42, 0x80, 0x1E, 0xF4, 0x0A, 0x0F, 0xC0];
        let sps = parse_sps(data).expect("should parse baseline SPS");

        assert_eq!(sps.profile_idc, 66);
        assert_eq!(sps.level_idc, 30);
        assert!(sps.constraint_set0_flag);
        assert!(!sps.constraint_set1_flag);
        assert_eq!(sps.sps_id, 0);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.bit_depth_luma, 8);
        assert_eq!(sps.bit_depth_chroma, 8);
        assert_eq!(sps.log2_max_frame_num, 4);
        assert_eq!(sps.poc_type, 0);
        assert_eq!(sps.log2_max_poc_lsb, 4);
        assert_eq!(sps.max_num_ref_frames, 1);
        assert!(!sps.gaps_in_frame_num_allowed);
        assert_eq!(sps.mb_width, 20); // 320 / 16
        assert_eq!(sps.mb_height, 15); // 240 / 16
        assert!(sps.frame_mbs_only_flag);
        assert!(!sps.mb_aff);
        assert!(sps.direct_8x8_inference_flag);
        assert_eq!(sps.crop_left, 0);
        assert_eq!(sps.crop_right, 0);
        assert_eq!(sps.crop_top, 0);
        assert_eq!(sps.crop_bottom, 0);
        assert!(!sps.vui_parameters_present);

        assert_eq!(sps.width(), 320);
        assert_eq!(sps.height(), 240);
    }

    #[test]
    fn test_parse_sps_invalid_sps_id() {
        // Craft an SPS with sps_id = 32 (invalid, must be < 32).
        // profile_idc=66, cs=0, reserved=0, level=30,
        // sps_id ue(32): 32+1=33=0b100001 (6 bits), prefix=5 zeros => 00000_100001 (11 bits)
        //
        // Byte 0: 0x42 (profile 66)
        // Byte 1: 0x00 (no constraint flags)
        // Byte 2: 0x1E (level 30)
        // Byte 3-4: ue(32) = 00000_100001 = 0000_0100_001x_xxxx
        //           byte3 = 0x04, byte4 = 0x20
        let data: &[u8] = &[0x42, 0x00, 0x1E, 0x04, 0x20];
        assert_eq!(parse_sps(data), Err(Error::InvalidData));
    }

    #[test]
    fn test_parse_sps_poc_type_2() {
        // Baseline SPS with poc_type=2 and 640x480.
        // profile_idc=66, cs0=1, level=30, sps_id=0,
        // log2_max_frame_num_minus4=0, poc_type=2 (ue(2)="011"),
        // max_num_ref_frames=0, gaps=0,
        // width: ue(39) = 00000_101000 (11 bits)
        // height: ue(29) = 0000_11110 (9 bits)
        // frame_mbs_only=1, direct_8x8=1, crop=0, vui=0
        //
        // Byte 3: bits 24-31: 1_1_011_1_0_0 = 0xDC
        // Byte 4: bits 32-39: 00001010 = 0x0A
        // Byte 5: bits 40-47: 00000011 = 0x03
        // Byte 6: bits 48-55: 11011000 = 0xD8

        let data: &[u8] = &[0x42, 0x80, 0x1E, 0xDC, 0x0A, 0x03, 0xD8];
        let sps = parse_sps(data).expect("should parse poc_type=2 SPS");

        assert_eq!(sps.poc_type, 2);
        assert_eq!(sps.mb_width, 40); // 640 / 16
        assert_eq!(sps.mb_height, 30); // 480 / 16
        assert_eq!(sps.width(), 640);
        assert_eq!(sps.height(), 480);
    }

    // -- is_high_profile tests --

    #[test]
    fn test_is_high_profile() {
        assert!(is_high_profile(100));
        assert!(is_high_profile(110));
        assert!(is_high_profile(122));
        assert!(is_high_profile(244));
        assert!(is_high_profile(44));
        assert!(!is_high_profile(66)); // Baseline
        assert!(!is_high_profile(77)); // Main
        assert!(!is_high_profile(88)); // Extended
    }
}
