//! H.264 / AVC bitstream syntax parser, Annex B AU framer, and `avcC` BSF helpers.
//!
//! Faithfully ports FFmpeg `n9.1` `libavcodec/h264_ps.c`, `libavcodec/h2645_vui.c`,
//! `libavcodec/h264_parser.c`, and `libavformat/avc.c`.

use wedeo_core::{
    AlphaMode, BitReader, ChromaLocation, CodecId, ColorDesc, ColorModel, ColorPrimaries,
    ColorRange, CompOrder, CropRect, Endian, Error, MatrixCoefficients, PixelLayout, PlanePacking,
    Rational, Result, SampleDepth, StreamParams, Subsampling, TransferCharacteristic,
    VideoStreamParams,
};

use crate::raw::{ElementaryFramer, FrameParams, FrameStep};
use crate::startcode::{
    AnnexBIter, RawAnnexBNal, annexb_to_length_prefixed, ebsp_to_rbsp, ebsp_to_rbsp_into,
    find_start_code, length_prefixed_to_annexb, rbsp_bit_length, rbsp_to_ebsp,
};

pub const MAX_SPS_COUNT: usize = 32;
pub const MAX_PPS_COUNT: usize = 256;
pub const MAX_DPB_FRAMES: u32 = 16;
const MIN_LOG2_MAX_FRAME_NUM: u32 = 4;
const MAX_LOG2_MAX_FRAME_NUM: u32 = 16;
const MAX_POC_CYCLE_LENGTH: usize = 256;
const EXTENDED_SAR: u8 = 255;

/// H.264 Table E-1 sample aspect ratios (`ff_h2645_pixel_aspect`).
pub const SAR_TABLE: [(i32, i32); 17] = [
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

/// 4x4 zigzag scan (`ff_zigzag_scan`).
pub const ZIGZAG_SCAN_4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// 8x8 zigzag scan (`ff_zigzag_direct`).
pub const ZIGZAG_SCAN_8X8: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Default JVT 4x4 scaling matrices (`default_scaling4` in `h264_ps.c`).
pub const DEFAULT_SCALING4: [[u8; 16]; 2] = [
    [
        6, 13, 20, 28, 13, 20, 28, 32, 20, 28, 32, 37, 28, 32, 37, 42,
    ],
    [
        10, 14, 20, 24, 14, 20, 24, 27, 20, 24, 27, 30, 24, 27, 30, 34,
    ],
];

/// Default JVT 8x8 scaling matrices (`default_scaling8` in `h264_ps.c`).
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

/// Level-to-MaxDpbMbs table (`level_max_dpb_mbs` in `h264_ps.c`).
const LEVEL_MAX_DPB_MBS: [(u8, u32); 16] = [
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

/// H.264 NAL unit types (ITU-T H.264 Table 7-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NalUnitType {
    Unspecified,
    Slice,
    Dpa,
    Dpb,
    Dpc,
    IdrSlice,
    Sei,
    Sps,
    Pps,
    Aud,
    EndSequence,
    EndStream,
    FillerData,
    SpsExt,
    Prefix,
    SubSps,
    AuxSlice,
    Other(u8),
}

impl NalUnitType {
    pub const fn from_u8(v: u8) -> Self {
        match v & 0x1F {
            0 => Self::Unspecified,
            1 => Self::Slice,
            2 => Self::Dpa,
            3 => Self::Dpb,
            4 => Self::Dpc,
            5 => Self::IdrSlice,
            6 => Self::Sei,
            7 => Self::Sps,
            8 => Self::Pps,
            9 => Self::Aud,
            10 => Self::EndSequence,
            11 => Self::EndStream,
            12 => Self::FillerData,
            13 => Self::SpsExt,
            14 => Self::Prefix,
            15 => Self::SubSps,
            19 => Self::AuxSlice,
            other => Self::Other(other),
        }
    }

    pub const fn code(self) -> u8 {
        match self {
            Self::Unspecified => 0,
            Self::Slice => 1,
            Self::Dpa => 2,
            Self::Dpb => 3,
            Self::Dpc => 4,
            Self::IdrSlice => 5,
            Self::Sei => 6,
            Self::Sps => 7,
            Self::Pps => 8,
            Self::Aud => 9,
            Self::EndSequence => 10,
            Self::EndStream => 11,
            Self::FillerData => 12,
            Self::SpsExt => 13,
            Self::Prefix => 14,
            Self::SubSps => 15,
            Self::AuxSlice => 19,
            Self::Other(v) => v & 0x1F,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::Slice => "slice",
            Self::Dpa => "dpa",
            Self::Dpb => "dpb",
            Self::Dpc => "dpc",
            Self::IdrSlice => "idr",
            Self::Sei => "sei",
            Self::Sps => "sps",
            Self::Pps => "pps",
            Self::Aud => "aud",
            Self::EndSequence => "end_seq",
            Self::EndStream => "end_stream",
            Self::FillerData => "filler",
            Self::SpsExt => "sps_ext",
            Self::Prefix => "prefix",
            Self::SubSps => "sub_sps",
            Self::AuxSlice => "aux_slice",
            Self::Other(_) => "reserved",
        }
    }

    /// Returns `true` for VCL NAL units carrying coded slices (`1..=5`).
    #[inline]
    pub const fn is_vcl(self) -> bool {
        matches!(
            self,
            Self::Slice | Self::Dpa | Self::Dpb | Self::Dpc | Self::IdrSlice
        )
    }

    /// Returns `true` for VCL NAL units beginning with a slice header (`Slice`, `Dpa`, `IdrSlice`).
    #[inline]
    pub const fn has_slice_header(self) -> bool {
        matches!(self, Self::Slice | Self::Dpa | Self::IdrSlice)
    }

    /// Returns `true` if this non-VCL NAL starts a new Access Unit when appearing after a VCL NAL
    /// (ITU-T H.264 §7.4.1.2.3).
    #[inline]
    pub const fn starts_new_au_after_vcl(self) -> bool {
        matches!(
            self.code(),
            6..=9 | 14..=18
        )
    }
}

/// Decoded 1-byte H.264 NAL unit header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NalHeader {
    pub forbidden_zero_bit: bool,
    pub nal_ref_idc: u8,
    pub nal_unit_type: NalUnitType,
}

impl NalHeader {
    pub fn parse(byte: u8) -> Result<Self> {
        let forbidden_zero_bit = (byte & 0x80) != 0;
        if forbidden_zero_bit {
            return Err(Error::invalid_data(
                "h264/nal",
                "NAL forbidden_zero_bit is set",
            ));
        }
        Ok(Self {
            forbidden_zero_bit,
            nal_ref_idc: (byte >> 5) & 0x03,
            nal_unit_type: NalUnitType::from_u8(byte & 0x1F),
        })
    }

    pub const fn to_byte(self) -> u8 {
        ((self.forbidden_zero_bit as u8) << 7)
            | ((self.nal_ref_idc & 0x03) << 5)
            | self.nal_unit_type.code()
    }
}

/// An Annex B NAL unit with parsed [`NalHeader`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnnexBNal<'a> {
    pub start_code_offset: usize,
    pub start_code_len: usize,
    pub header: NalHeader,
    /// Complete NAL bytes including the 1-byte header.
    pub raw: &'a [u8],
}

/// Find the earliest Annex B start code (`00 00 01` or `00 00 00 01`) at or after `from`.
pub fn find_annexb_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    find_start_code(data, from)
}

/// Iterate over valid Annex B NAL units in `data`.
pub fn iter_annexb_nals(data: &[u8]) -> impl Iterator<Item = Result<AnnexBNal<'_>>> {
    AnnexBIter::new(data).filter_map(|raw: RawAnnexBNal<'_>| {
        if raw.payload.is_empty() {
            return None;
        }
        Some(NalHeader::parse(raw.payload[0]).map(|header| AnnexBNal {
            start_code_offset: raw.start_code_offset,
            start_code_len: raw.start_code_len,
            header,
            raw: raw.payload,
        }))
    })
}

/// Strip emulation prevention bytes (`00 00 03` -> `00 00`) from an EBSP slice.
pub fn nal_to_rbsp(ebsp: &[u8]) -> Vec<u8> {
    ebsp_to_rbsp(ebsp)
}

/// Insert emulation prevention bytes (`00 00 {00..=03}` -> `00 00 03 {00..=03}`) into an RBSP slice.
pub fn rbsp_to_nal(rbsp: &[u8]) -> Vec<u8> {
    rbsp_to_ebsp(rbsp)
}

/// Returns `true` if `profile_idc` carries High-profile SPS extensions (chroma/bit-depth/scaling).
pub const fn is_high_profile(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135 | 144
    )
}

/// Parsed H.264 HRD parameters (`decode_hrd_parameters` in `h264_ps.c`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HrdParameters {
    pub cpb_cnt: u8,
    pub bit_rate_scale: u8,
    pub cpb_size_scale: u8,
    pub bit_rate_value: Vec<u32>,
    pub cpb_size_value: Vec<u32>,
    pub cbr_flag_mask: u32,
    pub initial_cpb_removal_delay_length: u8,
    pub cpb_removal_delay_length: u8,
    pub dpb_output_delay_length: u8,
    pub time_offset_length: u8,
}

/// Parsed H.264 VUI parameters (`decode_vui_parameters` + `ff_h2645_decode_common_vui_params`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VuiParameters {
    pub aspect_ratio_info_present: bool,
    pub aspect_ratio_idc: u8,
    pub sar: Rational,
    pub overscan_info_present: bool,
    pub overscan_appropriate: bool,
    pub video_signal_type_present: bool,
    pub video_format: u8,
    pub colour_description_present: bool,
    pub color: ColorDesc,
    pub chroma_loc_info_present: bool,
    pub chroma_sample_loc_type_top_field: u32,
    pub chroma_sample_loc_type_bottom_field: u32,
    pub timing_info_present: bool,
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub fixed_frame_rate_flag: bool,
    pub nal_hrd: Option<HrdParameters>,
    pub vcl_hrd: Option<HrdParameters>,
    pub low_delay_hrd_flag: bool,
    pub pic_struct_present_flag: bool,
    pub bitstream_restriction_flag: bool,
    pub num_reorder_frames: i32,
    pub max_dec_frame_buffering: u32,
}

impl Default for VuiParameters {
    fn default() -> Self {
        Self {
            aspect_ratio_info_present: false,
            aspect_ratio_idc: 0,
            sar: Rational::new(0, 1),
            overscan_info_present: false,
            overscan_appropriate: false,
            video_signal_type_present: false,
            video_format: 5, // Unspecified
            colour_description_present: false,
            color: ColorDesc::UNSPECIFIED,
            chroma_loc_info_present: false,
            chroma_sample_loc_type_top_field: 0,
            chroma_sample_loc_type_bottom_field: 0,
            timing_info_present: false,
            num_units_in_tick: 0,
            time_scale: 0,
            fixed_frame_rate_flag: false,
            nal_hrd: None,
            vcl_hrd: None,
            low_delay_hrd_flag: false,
            pic_struct_present_flag: false,
            bitstream_restriction_flag: false,
            num_reorder_frames: -1,
            max_dec_frame_buffering: 0,
        }
    }
}

/// Complete H.264 Sequence Parameter Set (SPS), ported faithfully from FFmpeg `n9.1` `h264_ps.c`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sps {
    pub sps_id: u32,
    pub profile_idc: u8,
    pub constraint_set_flags: u8,
    pub level_idc: u8,

    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub qpprime_y_zero_transform_bypass: bool,

    pub scaling_matrix_present: bool,
    pub scaling_matrix_present_mask: u16,
    pub scaling_matrix4: [[u8; 16]; 6],
    pub scaling_matrix8: [[u8; 64]; 6],

    pub log2_max_frame_num: u8,
    pub pic_order_cnt_type: u8,
    pub log2_max_pic_order_cnt_lsb: u8,
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub offset_for_ref_frame: Vec<i32>,

    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_allowed_flag: bool,

    pub mb_width: u32,
    pub mb_height: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,

    pub frame_cropping_flag: bool,
    pub crop: CropRect,

    pub vui_parameters_present_flag: bool,
    pub vui: VuiParameters,
    pub num_reorder_frames: i32,
}

impl Sps {
    /// Parse an SPS from a complete NAL unit (including the 1-byte NAL header).
    pub fn parse_nal(nal: &[u8]) -> Result<Self> {
        if nal.is_empty() {
            return Err(Error::invalid_data("h264/sps", "empty SPS NAL"));
        }
        let hdr = NalHeader::parse(nal[0])?;
        if hdr.nal_unit_type != NalUnitType::Sps {
            return Err(Error::invalid_data(
                "h264/sps",
                "expected SPS NAL unit type (7)",
            ));
        }
        let rbsp = ebsp_to_rbsp(&nal[1..]);
        Self::parse_rbsp(&rbsp)
    }

    /// Parse an SPS from its emulation-prevention-stripped RBSP payload (after the 1-byte NAL header).
    pub fn parse_rbsp(rbsp: &[u8]) -> Result<Self> {
        let mut br = BitReader::new(rbsp);

        let profile_idc = br.read_bits(8)? as u8;
        let mut constraint_set_flags = 0u8;
        for bit in 0..6u8 {
            if br.read_bool()? {
                constraint_set_flags |= 1 << bit;
            }
        }
        br.skip_bits(2)?; // reserved_zero_2bits
        let level_idc = br.read_bits(8)? as u8;
        let sps_id = br.read_ue()?;
        if sps_id as usize >= MAX_SPS_COUNT {
            return Err(Error::invalid_data("h264/sps", "sps_id out of range"));
        }

        let mut scaling_matrix4 = [[16u8; 16]; 6];
        let mut scaling_matrix8 = [[16u8; 64]; 6];
        let mut scaling_matrix_present = false;
        let mut scaling_matrix_present_mask = 0u16;

        let (
            chroma_format_idc,
            separate_colour_plane_flag,
            bit_depth_luma,
            bit_depth_chroma,
            qpprime_y_zero_transform_bypass,
        ) = if is_high_profile(profile_idc) {
            let cf = br.read_ue()?;
            if cf > 3 {
                return Err(Error::invalid_data(
                    "h264/sps",
                    "chroma_format_idc out of range",
                ));
            }
            let sep = if cf == 3 { br.read_bool()? } else { false };
            let bd_luma = br.read_ue()?.saturating_add(8);
            let bd_chroma = br.read_ue()?.saturating_add(8);
            if bd_luma > 14 || bd_chroma > 14 {
                return Err(Error::invalid_data(
                    "h264/sps",
                    "bit_depth out of range (8..=14)",
                ));
            }
            let bypass = br.read_bool()?;
            let seq_scaling_present = br.read_bool()?;
            if seq_scaling_present {
                scaling_matrix_present = true;
                decode_scaling_matrices(
                    &mut br,
                    cf as u8,
                    None,
                    true,
                    true,
                    &mut scaling_matrix_present_mask,
                    &mut scaling_matrix4,
                    &mut scaling_matrix8,
                )?;
            }
            (cf as u8, sep, bd_luma as u8, bd_chroma as u8, bypass)
        } else {
            (1u8, false, 8u8, 8u8, false)
        };

        let log2_max_frame_num_minus4 = br.read_ue()?;
        if log2_max_frame_num_minus4 > MAX_LOG2_MAX_FRAME_NUM - MIN_LOG2_MAX_FRAME_NUM {
            return Err(Error::invalid_data(
                "h264/sps",
                "log2_max_frame_num_minus4 out of range (0..=12)",
            ));
        }
        let log2_max_frame_num = (log2_max_frame_num_minus4 + 4) as u8;

        let pic_order_cnt_type = br.read_ue()?;
        let mut log2_max_pic_order_cnt_lsb = 0u8;
        let mut delta_pic_order_always_zero_flag = false;
        let mut offset_for_non_ref_pic = 0i32;
        let mut offset_for_top_to_bottom_field = 0i32;
        let mut offset_for_ref_frame = Vec::new();

        match pic_order_cnt_type {
            0 => {
                let t = br.read_ue()?;
                if t > 12 {
                    return Err(Error::invalid_data(
                        "h264/sps",
                        "log2_max_pic_order_cnt_lsb_minus4 out of range",
                    ));
                }
                log2_max_pic_order_cnt_lsb = (t + 4) as u8;
            }
            1 => {
                delta_pic_order_always_zero_flag = br.read_bool()?;
                offset_for_non_ref_pic = br.read_se()?;
                offset_for_top_to_bottom_field = br.read_se()?;
                if offset_for_non_ref_pic == i32::MIN || offset_for_top_to_bottom_field == i32::MIN
                {
                    return Err(Error::invalid_data("h264/sps", "POC offset out of range"));
                }
                let cycle_len = br.read_ue()? as usize;
                if cycle_len >= MAX_POC_CYCLE_LENGTH {
                    return Err(Error::invalid_data(
                        "h264/sps",
                        "num_ref_frames_in_pic_order_cnt_cycle overflow",
                    ));
                }
                offset_for_ref_frame.reserve_exact(cycle_len);
                for _ in 0..cycle_len {
                    let off = br.read_se()?;
                    if off == i32::MIN {
                        return Err(Error::invalid_data(
                            "h264/sps",
                            "offset_for_ref_frame out of range",
                        ));
                    }
                    offset_for_ref_frame.push(off);
                }
            }
            2 => {}
            _ => {
                return Err(Error::invalid_data(
                    "h264/sps",
                    "illegal pic_order_cnt_type",
                ));
            }
        }

        let max_num_ref_frames = br.read_ue()?;
        if max_num_ref_frames > MAX_DPB_FRAMES {
            return Err(Error::invalid_data(
                "h264/sps",
                "max_num_ref_frames exceeds 16",
            ));
        }
        let gaps_in_frame_num_allowed_flag = br.read_bool()?;

        let mb_width = br
            .read_ue()?
            .checked_add(1)
            .ok_or_else(|| Error::invalid_data("h264/sps", "mb_width overflow"))?;
        let map_height = br
            .read_ue()?
            .checked_add(1)
            .ok_or_else(|| Error::invalid_data("h264/sps", "mb_height overflow"))?;
        let frame_mbs_only_flag = br.read_bool()?;
        let mb_height = map_height
            .checked_mul(if frame_mbs_only_flag { 1 } else { 2 })
            .ok_or_else(|| Error::invalid_data("h264/sps", "mb_height overflow"))?;

        let mb_adaptive_frame_field_flag = if !frame_mbs_only_flag {
            br.read_bool()?
        } else {
            false
        };

        let coded_w = mb_width
            .checked_mul(16)
            .ok_or_else(|| Error::invalid_data("h264/sps", "coded width overflow"))?;
        let coded_h = mb_height
            .checked_mul(16)
            .ok_or_else(|| Error::invalid_data("h264/sps", "coded height overflow"))?;
        if coded_w > i32::MAX as u32 || coded_h > i32::MAX as u32 {
            return Err(Error::invalid_data("h264/sps", "coded dimensions overflow"));
        }

        let direct_8x8_inference_flag = br.read_bool()?;

        let frame_cropping_flag = br.read_bool()?;
        let crop = if frame_cropping_flag {
            let crop_left = br.read_ue()?;
            let crop_right = br.read_ue()?;
            let crop_top = br.read_ue()?;
            let crop_bottom = br.read_ue()?;

            let eff_cf = if separate_colour_plane_flag {
                0
            } else {
                chroma_format_idc
            };
            let vsub = u32::from(eff_cf == 1);
            let hsub = u32::from(eff_cf == 1 || eff_cf == 2);
            let step_x = 1u32 << hsub;
            let step_y = (if frame_mbs_only_flag { 1u32 } else { 2u32 }) << vsub;

            let left = crop_left
                .checked_mul(step_x)
                .ok_or_else(|| Error::invalid_data("h264/sps", "crop_left overflow"))?;
            let right = crop_right
                .checked_mul(step_x)
                .ok_or_else(|| Error::invalid_data("h264/sps", "crop_right overflow"))?;
            let top = crop_top
                .checked_mul(step_y)
                .ok_or_else(|| Error::invalid_data("h264/sps", "crop_top overflow"))?;
            let bottom = crop_bottom
                .checked_mul(step_y)
                .ok_or_else(|| Error::invalid_data("h264/sps", "crop_bottom overflow"))?;

            if left.saturating_add(right) >= coded_w || top.saturating_add(bottom) >= coded_h {
                return Err(Error::invalid_data(
                    "h264/sps",
                    "crop margins exceed coded picture dimensions",
                ));
            }
            CropRect {
                top,
                bottom,
                left,
                right,
            }
        } else {
            CropRect::ZERO
        };

        let vui_parameters_present_flag = br.read_bool()?;
        let vui = if vui_parameters_present_flag {
            decode_vui_parameters(&mut br)?
        } else {
            VuiParameters::default()
        };

        // Derive num_reorder_frames matching FFmpeg h264_ps.c lines 545-555.
        let num_reorder_frames = if vui.bitstream_restriction_flag {
            vui.num_reorder_frames
        } else if max_num_ref_frames > 0 {
            let mut reorder = (MAX_DPB_FRAMES - 1) as i32;
            let pic_mbs = mb_width.saturating_mul(mb_height).max(1);
            for &(lvl, max_dpb_mbs) in &LEVEL_MAX_DPB_MBS {
                if lvl == level_idc {
                    reorder = ((max_dpb_mbs / pic_mbs) as i32).min(reorder);
                    break;
                }
            }
            reorder
        } else {
            0
        };

        Ok(Self {
            sps_id,
            profile_idc,
            constraint_set_flags,
            level_idc,
            chroma_format_idc,
            separate_colour_plane_flag,
            bit_depth_luma,
            bit_depth_chroma,
            qpprime_y_zero_transform_bypass,
            scaling_matrix_present,
            scaling_matrix_present_mask,
            scaling_matrix4,
            scaling_matrix8,
            log2_max_frame_num,
            pic_order_cnt_type: pic_order_cnt_type as u8,
            log2_max_pic_order_cnt_lsb,
            delta_pic_order_always_zero_flag,
            offset_for_non_ref_pic,
            offset_for_top_to_bottom_field,
            offset_for_ref_frame,
            max_num_ref_frames,
            gaps_in_frame_num_allowed_flag,
            mb_width,
            mb_height,
            frame_mbs_only_flag,
            mb_adaptive_frame_field_flag,
            direct_8x8_inference_flag,
            frame_cropping_flag,
            crop,
            vui_parameters_present_flag,
            vui,
            num_reorder_frames,
        })
    }

    #[inline]
    pub const fn coded_width(&self) -> u32 {
        self.mb_width * 16
    }

    #[inline]
    pub const fn coded_height(&self) -> u32 {
        self.mb_height * 16
    }

    #[inline]
    pub const fn display_width(&self) -> u32 {
        self.crop
            .display_size(self.coded_width(), self.coded_height())
            .0
    }

    #[inline]
    pub const fn display_height(&self) -> u32 {
        self.crop
            .display_size(self.coded_width(), self.coded_height())
            .1
    }

    /// Derive the orthogonal [`PixelLayout`] matching FFmpeg's H.264 pixel-format selection.
    pub const fn pixel_layout(&self) -> PixelLayout {
        let depth = match self.bit_depth_luma {
            8 => SampleDepth::U8,
            9 | 10 => SampleDepth::U10,
            11 | 12 => SampleDepth::U12,
            13 | 14 => SampleDepth::U14,
            _ => SampleDepth::U16,
        };
        if self.chroma_format_idc == 0 {
            return PixelLayout {
                model: ColorModel::Gray,
                packing: PlanePacking::Planar,
                subsampling: Subsampling::Cs400,
                depth,
                endian: Endian::Little,
                alpha: AlphaMode::None,
                order: CompOrder::Gray,
            };
        }
        if self.chroma_format_idc == 3
            && matches!(self.vui.color.matrix, MatrixCoefficients::Identity)
        {
            return PixelLayout {
                model: ColorModel::Rgb,
                packing: PlanePacking::Planar,
                subsampling: Subsampling::Cs444,
                depth,
                endian: Endian::Little,
                alpha: AlphaMode::None,
                order: CompOrder::Gbr,
            };
        }
        let subsampling = match self.chroma_format_idc {
            2 => Subsampling::Cs422,
            3 => Subsampling::Cs444,
            _ => Subsampling::Cs420,
        };
        PixelLayout {
            model: ColorModel::Ycbcr,
            packing: PlanePacking::Planar,
            subsampling,
            depth,
            endian: Endian::Little,
            alpha: AlphaMode::None,
            order: CompOrder::Yuv,
        }
    }

    /// Advisory frame rate derived from VUI timing info (`time_scale / (2 * num_units_in_tick)`).
    pub fn frame_rate(&self) -> Option<Rational> {
        if self.vui.timing_info_present && self.vui.num_units_in_tick > 0 && self.vui.time_scale > 0
        {
            let mut r = Rational::new(
                self.vui.time_scale.min(i32::MAX as u32) as i32,
                self.vui
                    .num_units_in_tick
                    .saturating_mul(2)
                    .min(i32::MAX as u32) as i32,
            );
            r.reduce(i32::MAX as i64);
            Some(r)
        } else {
            None
        }
    }

    /// Convert SPS fields into a [`VideoStreamParams`].
    pub fn to_video_stream_params(&self) -> VideoStreamParams {
        let mut params = VideoStreamParams::new(
            self.display_width(),
            self.display_height(),
            self.pixel_layout(),
        );
        params.color = self.vui.color;
        params.crop = self.crop;
        params.sar = if self.vui.sar.is_valid() {
            self.vui.sar
        } else {
            Rational::new(1, 1)
        };
        if let Some(fr) = self.frame_rate() {
            params.avg_frame_rate = fr;
        }
        params
    }
}

fn decode_hrd_parameters(br: &mut BitReader<'_>) -> Result<HrdParameters> {
    let cpb_cnt = br.read_ue()?.saturating_add(1);
    if cpb_cnt > 32 {
        return Err(Error::invalid_data("h264/hrd", "cpb_cnt exceeds 32"));
    }
    let bit_rate_scale = br.read_bits(4)? as u8;
    let cpb_size_scale = br.read_bits(4)? as u8;
    let mut bit_rate_value = Vec::with_capacity(cpb_cnt as usize);
    let mut cpb_size_value = Vec::with_capacity(cpb_cnt as usize);
    let mut cbr_flag_mask = 0u32;
    for i in 0..cpb_cnt {
        bit_rate_value.push(br.read_ue()?.wrapping_add(1));
        cpb_size_value.push(br.read_ue()?.wrapping_add(1));
        if br.read_bool()? {
            cbr_flag_mask |= 1u32 << i;
        }
    }
    let initial_cpb_removal_delay_length = br.read_bits(5)? as u8 + 1;
    let cpb_removal_delay_length = br.read_bits(5)? as u8 + 1;
    let dpb_output_delay_length = br.read_bits(5)? as u8 + 1;
    let time_offset_length = br.read_bits(5)? as u8;
    Ok(HrdParameters {
        cpb_cnt: cpb_cnt as u8,
        bit_rate_scale,
        cpb_size_scale,
        bit_rate_value,
        cpb_size_value,
        cbr_flag_mask,
        initial_cpb_removal_delay_length,
        cpb_removal_delay_length,
        dpb_output_delay_length,
        time_offset_length,
    })
}

fn decode_vui_parameters(br: &mut BitReader<'_>) -> Result<VuiParameters> {
    let aspect_ratio_info_present = br.read_bool()?;
    let mut vui = VuiParameters {
        aspect_ratio_info_present,
        ..VuiParameters::default()
    };

    if vui.aspect_ratio_info_present {
        vui.aspect_ratio_idc = br.read_bits(8)? as u8;
        if (vui.aspect_ratio_idc as usize) < SAR_TABLE.len() {
            let (n, d) = SAR_TABLE[vui.aspect_ratio_idc as usize];
            vui.sar = Rational::new(n, d);
        } else if vui.aspect_ratio_idc == EXTENDED_SAR {
            let num = br.read_bits(16)? as i32;
            let den = br.read_bits(16)? as i32;
            if num > 0 && den > 0 {
                vui.sar = Rational::new(num, den);
            } else {
                vui.sar = Rational::new(0, 1);
            }
        }
    }

    vui.overscan_info_present = br.read_bool()?;
    if vui.overscan_info_present {
        vui.overscan_appropriate = br.read_bool()?;
    }

    vui.video_signal_type_present = br.read_bool()?;
    if vui.video_signal_type_present {
        vui.video_format = br.read_bits(3)? as u8;
        let full_range = br.read_bool()?;
        vui.color.range = if full_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        vui.colour_description_present = br.read_bool()?;
        if vui.colour_description_present {
            let primaries = br.read_bits(8)? as u8;
            let transfer = br.read_bits(8)? as u8;
            let matrix = br.read_bits(8)? as u8;

            // Match FFmpeg h2645_vui.c reserved-to-unspecified sanitization.
            vui.color.primaries = match primaries {
                0 | 3 => ColorPrimaries::Unspecified,
                v => ColorPrimaries::from_cicp(v),
            };
            vui.color.transfer = match transfer {
                0 | 3 => TransferCharacteristic::Unspecified,
                v => TransferCharacteristic::from_cicp(v),
            };
            vui.color.matrix = match matrix {
                3 => MatrixCoefficients::Unspecified,
                v => MatrixCoefficients::from_cicp(v),
            };
        }
    }

    vui.chroma_loc_info_present = br.read_bool()?;
    if vui.chroma_loc_info_present {
        vui.chroma_sample_loc_type_top_field = br.read_ue()?;
        vui.chroma_sample_loc_type_bottom_field = br.read_ue()?;
        vui.color.chroma_location = match vui.chroma_sample_loc_type_top_field {
            0 => ChromaLocation::Left,
            1 => ChromaLocation::Center,
            2 => ChromaLocation::TopLeft,
            3 => ChromaLocation::Top,
            4 => ChromaLocation::BottomLeft,
            5 => ChromaLocation::Bottom,
            _ => ChromaLocation::Unspecified,
        };
    } else {
        // Matches FFmpeg h2645_vui.c line 99.
        vui.color.chroma_location = ChromaLocation::Left;
    }

    // Match FFmpeg h264_ps.c line 138 truncated-VUI guard.
    if br.bits_left() == 0 || (br.peek_bits(1) != 0 && br.bits_left() < 10) {
        return Ok(vui);
    }

    vui.timing_info_present = br.read_bool()?;
    if vui.timing_info_present {
        let num_units_in_tick = br.read_bits(32)?;
        let time_scale = br.read_bits(32)?;
        if num_units_in_tick == 0 || time_scale == 0 {
            vui.timing_info_present = false;
        } else {
            vui.num_units_in_tick = num_units_in_tick;
            vui.time_scale = time_scale;
        }
        vui.fixed_frame_rate_flag = br.read_bool()?;
    }

    if br.bits_left() == 0 {
        return Ok(vui);
    }
    let nal_hrd_present = br.read_bool()?;
    if nal_hrd_present {
        vui.nal_hrd = Some(decode_hrd_parameters(br)?);
    }
    let vcl_hrd_present = br.read_bool()?;
    if vcl_hrd_present {
        vui.vcl_hrd = Some(decode_hrd_parameters(br)?);
    }
    if nal_hrd_present || vcl_hrd_present {
        vui.low_delay_hrd_flag = br.read_bool()?;
    }
    if br.bits_left() == 0 {
        return Ok(vui);
    }
    vui.pic_struct_present_flag = br.read_bool()?;
    if br.bits_left() == 0 {
        return Ok(vui);
    }
    vui.bitstream_restriction_flag = br.read_bool()?;
    if vui.bitstream_restriction_flag {
        let _mv_over_pic_boundaries = br.read_bool()?;
        let _max_bytes_per_pic_denom = br.read_ue()?;
        let _max_bits_per_mb_denom = br.read_ue()?;
        let _log2_max_mv_len_h = br.read_ue()?;
        let _log2_max_mv_len_v = br.read_ue()?;
        let num_reorder = br.read_ue()?;
        let max_dec_buf = br.read_ue()?;
        if num_reorder > MAX_DPB_FRAMES {
            return Err(Error::invalid_data(
                "h264/vui",
                "num_reorder_frames exceeds 16",
            ));
        }
        vui.num_reorder_frames = num_reorder as i32;
        vui.max_dec_frame_buffering = max_dec_buf;
    }

    Ok(vui)
}

fn decode_scaling_list(
    br: &mut BitReader<'_>,
    factors: &mut [u8],
    jvt_list: &[u8],
    fallback_list: &[u8],
    mask: &mut u16,
    pos: u32,
) -> Result<()> {
    let size = factors.len();
    let scan: &[u8] = if size == 16 {
        &ZIGZAG_SCAN_4X4
    } else {
        &ZIGZAG_SCAN_8X8
    };
    let present = br.read_bool()?;
    if present {
        *mask |= 1u16 << pos;
    }
    if !present {
        factors.copy_from_slice(fallback_list);
        return Ok(());
    }
    let mut last: i32 = 8;
    let mut next: i32 = 8;
    for i in 0..size {
        if next != 0 {
            let delta = br.read_se()?;
            if !(-128..=127).contains(&delta) {
                return Err(Error::invalid_data(
                    "h264/scaling",
                    "delta_scale out of range (-128..=127)",
                ));
            }
            next = (last + delta) & 0xFF;
        }
        if i == 0 && next == 0 {
            factors.copy_from_slice(jvt_list);
            return Ok(());
        }
        let val = if next != 0 { next } else { last };
        factors[scan[i] as usize] = val as u8;
        last = val;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_scaling_matrices(
    br: &mut BitReader<'_>,
    chroma_format_idc: u8,
    sps_fallback: Option<&Sps>,
    is_sps: bool,
    parse_8x8: bool,
    mask: &mut u16,
    scaling_matrix4: &mut [[u8; 16]; 6],
    scaling_matrix8: &mut [[u8; 64]; 6],
) -> Result<()> {
    *mask = 0;
    let fallback_sps = sps_fallback.filter(|s| !is_sps && s.scaling_matrix_present);
    let fb4_intra = fallback_sps.map_or(&DEFAULT_SCALING4[0], |s| &s.scaling_matrix4[0]);
    let fb4_inter = fallback_sps.map_or(&DEFAULT_SCALING4[1], |s| &s.scaling_matrix4[3]);
    let fb8_intra = fallback_sps.map_or(&DEFAULT_SCALING8[0], |s| &s.scaling_matrix8[0]);
    let fb8_inter = fallback_sps.map_or(&DEFAULT_SCALING8[1], |s| &s.scaling_matrix8[3]);

    decode_scaling_list(
        br,
        &mut scaling_matrix4[0],
        &DEFAULT_SCALING4[0],
        fb4_intra,
        mask,
        0,
    )?;
    let prev = scaling_matrix4[0];
    decode_scaling_list(
        br,
        &mut scaling_matrix4[1],
        &DEFAULT_SCALING4[0],
        &prev,
        mask,
        1,
    )?;
    let prev = scaling_matrix4[1];
    decode_scaling_list(
        br,
        &mut scaling_matrix4[2],
        &DEFAULT_SCALING4[0],
        &prev,
        mask,
        2,
    )?;
    decode_scaling_list(
        br,
        &mut scaling_matrix4[3],
        &DEFAULT_SCALING4[1],
        fb4_inter,
        mask,
        3,
    )?;
    let prev = scaling_matrix4[3];
    decode_scaling_list(
        br,
        &mut scaling_matrix4[4],
        &DEFAULT_SCALING4[1],
        &prev,
        mask,
        4,
    )?;
    let prev = scaling_matrix4[4];
    decode_scaling_list(
        br,
        &mut scaling_matrix4[5],
        &DEFAULT_SCALING4[1],
        &prev,
        mask,
        5,
    )?;

    if parse_8x8 {
        decode_scaling_list(
            br,
            &mut scaling_matrix8[0],
            &DEFAULT_SCALING8[0],
            fb8_intra,
            mask,
            6,
        )?;
        decode_scaling_list(
            br,
            &mut scaling_matrix8[3],
            &DEFAULT_SCALING8[1],
            fb8_inter,
            mask,
            7,
        )?;
        if chroma_format_idc == 3 {
            let prev = scaling_matrix8[0];
            decode_scaling_list(
                br,
                &mut scaling_matrix8[1],
                &DEFAULT_SCALING8[0],
                &prev,
                mask,
                8,
            )?;
            let prev = scaling_matrix8[3];
            decode_scaling_list(
                br,
                &mut scaling_matrix8[4],
                &DEFAULT_SCALING8[1],
                &prev,
                mask,
                9,
            )?;
            let prev = scaling_matrix8[1];
            decode_scaling_list(
                br,
                &mut scaling_matrix8[2],
                &DEFAULT_SCALING8[0],
                &prev,
                mask,
                10,
            )?;
            let prev = scaling_matrix8[4];
            decode_scaling_list(
                br,
                &mut scaling_matrix8[5],
                &DEFAULT_SCALING8[1],
                &prev,
                mask,
                11,
            )?;
        }
    }
    Ok(())
}

/// Complete H.264 Picture Parameter Set (PPS), ported faithfully from FFmpeg `n9.1` `h264_ps.c`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pps {
    pub pps_id: u32,
    pub sps_id: u32,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present: bool,
    pub slice_group_count: u32,
    pub ref_count: [u32; 2],
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u8,
    pub init_qp: i32,
    pub init_qs: i32,
    pub chroma_qp_index_offset: [i32; 2],
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    pub pic_scaling_matrix_present: bool,
    pub pic_scaling_matrix_present_mask: u16,
    pub scaling_matrix4: [[u8; 16]; 6],
    pub scaling_matrix8: [[u8; 64]; 6],
}

impl Pps {
    /// Parse a PPS from a complete NAL unit (including the 1-byte NAL header), looking up the
    /// referenced SPS via `get_sps`.
    pub fn parse_nal<'a>(nal: &[u8], get_sps: impl Fn(u32) -> Option<&'a Sps>) -> Result<Self> {
        if nal.is_empty() {
            return Err(Error::invalid_data("h264/pps", "empty PPS NAL"));
        }
        let hdr = NalHeader::parse(nal[0])?;
        if hdr.nal_unit_type != NalUnitType::Pps {
            return Err(Error::invalid_data(
                "h264/pps",
                "expected PPS NAL unit type (8)",
            ));
        }
        let rbsp = ebsp_to_rbsp(&nal[1..]);
        Self::parse_rbsp(&rbsp, get_sps)
    }

    /// Parse a PPS from its emulation-prevention-stripped RBSP payload.
    pub fn parse_rbsp<'a>(rbsp: &[u8], get_sps: impl Fn(u32) -> Option<&'a Sps>) -> Result<Self> {
        let bit_length = rbsp_bit_length(rbsp);
        let mut br = BitReader::new(rbsp);

        let pps_id = br.read_ue()?;
        if pps_id as usize >= MAX_PPS_COUNT {
            return Err(Error::invalid_data("h264/pps", "pps_id out of range"));
        }
        let sps_id = br.read_ue()?;
        if sps_id as usize >= MAX_SPS_COUNT {
            return Err(Error::invalid_data("h264/pps", "sps_id out of range"));
        }
        let sps = get_sps(sps_id)
            .ok_or_else(|| Error::invalid_data("h264/pps", "referenced SPS not found"))?;

        let entropy_coding_mode_flag = br.read_bool()?;
        let bottom_field_pic_order_in_frame_present = br.read_bool()?;
        let slice_group_count = br.read_ue()?.saturating_add(1);
        if slice_group_count > 1 {
            return Err(Error::unsupported("h264/pps", "FMO slice groups > 1"));
        }

        let ref0 = br.read_ue()?.saturating_add(1);
        let ref1 = br.read_ue()?.saturating_add(1);
        if ref0 > 32 || ref1 > 32 {
            return Err(Error::invalid_data(
                "h264/pps",
                "num_ref_idx_default_active exceeds 32",
            ));
        }

        let qp_bd_offset = 6 * i32::from(sps.bit_depth_luma.saturating_sub(8));
        let weighted_pred = br.read_bool()?;
        let weighted_bipred_idc = br.read_bits(2)? as u8;
        let init_qp = br.read_se()?.wrapping_add(26).wrapping_add(qp_bd_offset);
        let init_qs = br.read_se()?.wrapping_add(26).wrapping_add(qp_bd_offset);
        let chroma_qp_0 = br.read_se()?;
        if !(-12..=12).contains(&chroma_qp_0) {
            return Err(Error::invalid_data(
                "h264/pps",
                "chroma_qp_index_offset out of range",
            ));
        }
        let deblocking_filter_control_present = br.read_bool()?;
        let constrained_intra_pred = br.read_bool()?;
        let redundant_pic_cnt_present = br.read_bool()?;

        let mut transform_8x8_mode = false;
        let mut pic_scaling_matrix_present = false;
        let mut pic_scaling_matrix_present_mask = 0u16;
        let mut scaling_matrix4 = sps.scaling_matrix4;
        let mut scaling_matrix8 = sps.scaling_matrix8;
        let mut chroma_qp_1 = chroma_qp_0;

        // Match FFmpeg `more_rbsp_data_in_pps` in h264_ps.c.
        let allow_more =
            !((sps.profile_idc == 66 || sps.profile_idc == 77 || sps.profile_idc == 88)
                && (sps.constraint_set_flags & 7) != 0);
        if allow_more && br.position() < bit_length {
            transform_8x8_mode = br.read_bool()?;
            pic_scaling_matrix_present = br.read_bool()?;
            if pic_scaling_matrix_present {
                decode_scaling_matrices(
                    &mut br,
                    sps.chroma_format_idc,
                    Some(sps),
                    false,
                    transform_8x8_mode,
                    &mut pic_scaling_matrix_present_mask,
                    &mut scaling_matrix4,
                    &mut scaling_matrix8,
                )?;
            }
            chroma_qp_1 = br.read_se()?;
            if !(-12..=12).contains(&chroma_qp_1) {
                return Err(Error::invalid_data(
                    "h264/pps",
                    "second_chroma_qp_index_offset out of range",
                ));
            }
        }

        Ok(Self {
            pps_id,
            sps_id,
            entropy_coding_mode_flag,
            bottom_field_pic_order_in_frame_present,
            slice_group_count,
            ref_count: [ref0, ref1],
            weighted_pred,
            weighted_bipred_idc,
            init_qp,
            init_qs,
            chroma_qp_index_offset: [chroma_qp_0, chroma_qp_1],
            deblocking_filter_control_present,
            constrained_intra_pred,
            redundant_pic_cnt_present,
            transform_8x8_mode,
            pic_scaling_matrix_present,
            pic_scaling_matrix_present_mask,
            scaling_matrix4,
            scaling_matrix8,
        })
    }
}

/// Active SPS/PPS parameter set table.
#[derive(Debug, Clone, Default)]
pub struct ParamSetStore {
    sps: Vec<Option<Sps>>,
    pps: Vec<Option<Pps>>,
}

impl ParamSetStore {
    pub fn new() -> Self {
        Self {
            sps: vec![None; MAX_SPS_COUNT],
            pps: vec![None; MAX_PPS_COUNT],
        }
    }

    pub fn parse_sps_nal(&mut self, nal: &[u8]) -> Result<&Sps> {
        let sps = Sps::parse_nal(nal)?;
        let id = sps.sps_id as usize;
        self.sps[id] = Some(sps);
        Ok(self.sps[id].as_ref().expect("just inserted"))
    }

    pub fn parse_pps_nal(&mut self, nal: &[u8]) -> Result<&Pps> {
        let pps = Pps::parse_nal(nal, |id| self.sps.get(id as usize).and_then(Option::as_ref))?;
        let id = pps.pps_id as usize;
        self.pps[id] = Some(pps);
        Ok(self.pps[id].as_ref().expect("just inserted"))
    }

    pub fn sps(&self, id: u32) -> Option<&Sps> {
        self.sps.get(id as usize).and_then(Option::as_ref)
    }

    pub fn pps(&self, id: u32) -> Option<&Pps> {
        self.pps.get(id as usize).and_then(Option::as_ref)
    }

    pub fn clear(&mut self) {
        self.sps.fill(None);
        self.pps.fill(None);
    }
}

/// Decoded slice header prefix sufficient for ITU-T H.264 §7.4.1.2.4 AU boundary detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceHeaderPrefix {
    pub nal_ref_idc: u8,
    pub nal_unit_type: NalUnitType,
    pub first_mb_in_slice: u32,
    pub slice_type: u8,
    pub pps_id: u32,
    /// Full §7.4.1.2.4 picture identity fields (populated when SPS & PPS are known).
    pub pic_id: Option<SlicePicIdentity>,
}

/// Picture-identifying fields from H.264 §7.4.1.2.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlicePicIdentity {
    pub frame_num: u32,
    pub field_pic_flag: bool,
    pub bottom_field_flag: Option<bool>,
    pub idr_pic_id: Option<u32>,
    pub pic_order_cnt_type: u8,
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
}

impl SliceHeaderPrefix {
    /// Parse the slice header prefix of a VCL NAL (`Slice`, `Dpa`, or `IdrSlice`).
    pub fn parse_nal(nal: &[u8], store: &ParamSetStore, scratch: &mut Vec<u8>) -> Result<Self> {
        if nal.len() < 2 {
            return Err(Error::invalid_data("h264/slice", "truncated VCL NAL"));
        }
        let hdr = NalHeader::parse(nal[0])?;
        if !hdr.nal_unit_type.has_slice_header() {
            return Err(Error::invalid_data(
                "h264/slice",
                "NAL does not begin with a slice header",
            ));
        }
        // Only the leading bytes are needed to decode through POC fields.
        let prefix_len = nal.len().min(64);
        ebsp_to_rbsp_into(&nal[1..prefix_len], scratch);
        let mut br = BitReader::new(scratch);

        let first_mb_in_slice = br.read_ue()?;
        let slice_type = (br.read_ue()? % 5) as u8;
        let pps_id = br.read_ue()?;

        let pic_id = if let Some(pps) = store.pps(pps_id)
            && let Some(sps) = store.sps(pps.sps_id)
        {
            if sps.separate_colour_plane_flag {
                let _colour_plane_id = br.read_bits(2)?;
            }
            let frame_num = br.read_bits(u32::from(sps.log2_max_frame_num))?;
            let (field_pic_flag, bottom_field_flag) = if !sps.frame_mbs_only_flag {
                let field = br.read_bool()?;
                let bottom = if field { Some(br.read_bool()?) } else { None };
                (field, bottom)
            } else {
                (false, None)
            };
            let idr_pic_id = if hdr.nal_unit_type == NalUnitType::IdrSlice {
                Some(br.read_ue()?)
            } else {
                None
            };
            let mut pic_order_cnt_lsb = 0u32;
            let mut delta_pic_order_cnt_bottom = 0i32;
            let mut delta_pic_order_cnt = [0i32; 2];

            if sps.pic_order_cnt_type == 0 {
                pic_order_cnt_lsb = br.read_bits(u32::from(sps.log2_max_pic_order_cnt_lsb))?;
                if pps.bottom_field_pic_order_in_frame_present && !field_pic_flag {
                    delta_pic_order_cnt_bottom = br.read_se()?;
                }
            } else if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
                delta_pic_order_cnt[0] = br.read_se()?;
                if pps.bottom_field_pic_order_in_frame_present && !field_pic_flag {
                    delta_pic_order_cnt[1] = br.read_se()?;
                }
            }

            Some(SlicePicIdentity {
                frame_num,
                field_pic_flag,
                bottom_field_flag,
                idr_pic_id,
                pic_order_cnt_type: sps.pic_order_cnt_type,
                pic_order_cnt_lsb,
                delta_pic_order_cnt_bottom,
                delta_pic_order_cnt,
            })
        } else {
            None
        };

        Ok(Self {
            nal_ref_idc: hdr.nal_ref_idc,
            nal_unit_type: hdr.nal_unit_type,
            first_mb_in_slice,
            slice_type,
            pps_id,
            pic_id,
        })
    }

    /// Returns `true` if `self` starts a new primary coded picture relative to `prev`,
    /// implementing ITU-T H.264 §7.4.1.2.4 exactly.
    pub fn is_new_picture_after(&self, prev: &Self) -> bool {
        if self.pps_id != prev.pps_id {
            return true;
        }
        let self_idr = self.nal_unit_type == NalUnitType::IdrSlice;
        let prev_idr = prev.nal_unit_type == NalUnitType::IdrSlice;
        if self_idr != prev_idr {
            return true;
        }
        if (self.nal_ref_idc == 0) != (prev.nal_ref_idc == 0) {
            return true;
        }
        if let (Some(a), Some(b)) = (self.pic_id, prev.pic_id) {
            if a.frame_num != b.frame_num {
                return true;
            }
            if a.field_pic_flag != b.field_pic_flag {
                return true;
            }
            if let (Some(ab), Some(bb)) = (a.bottom_field_flag, b.bottom_field_flag)
                && ab != bb
            {
                return true;
            }
            if a.pic_order_cnt_type == 0
                && b.pic_order_cnt_type == 0
                && (a.pic_order_cnt_lsb != b.pic_order_cnt_lsb
                    || a.delta_pic_order_cnt_bottom != b.delta_pic_order_cnt_bottom)
            {
                return true;
            }
            if a.pic_order_cnt_type == 1
                && b.pic_order_cnt_type == 1
                && a.delta_pic_order_cnt != b.delta_pic_order_cnt
            {
                return true;
            }
            if self_idr && prev_idr && a.idr_pic_id != b.idr_pic_id {
                return true;
            }
            // Same picture: multi-slice frame!
            false
        } else {
            // Fallback when SPS/PPS not yet available: FFmpeg's `mb <= last_mb` rule.
            self.first_mb_in_slice <= prev.first_mb_in_slice
        }
    }
}

/// H.264 Annex B Access Unit framer implementing [`ElementaryFramer`] with real §7.4.1.2.4
/// slice-header primary picture boundary detection.
pub struct H264AnnexBFramer {
    store: ParamSetStore,
    rbsp_scratch: Vec<u8>,
    last_emitted_sps: Option<Sps>,
}

impl H264AnnexBFramer {
    pub fn new() -> Self {
        Self {
            store: ParamSetStore::new(),
            rbsp_scratch: Vec::with_capacity(64),
            last_emitted_sps: None,
        }
    }

    pub fn param_sets(&self) -> &ParamSetStore {
        &self.store
    }
}

impl Default for H264AnnexBFramer {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementaryFramer for H264AnnexBFramer {
    fn format_name(&self) -> &'static str {
        "h264"
    }

    fn codec_id(&self) -> CodecId {
        CodecId::H264
    }

    fn initial_params(&self) -> StreamParams {
        StreamParams::Video(VideoStreamParams::new(0, 0, PixelLayout::YUV420P))
    }

    fn probe(head: &[u8]) -> u8 {
        let mut sps_ok = false;
        let mut pps_ok = false;
        let mut vcl_count = 0u32;
        let mut invalid = 0u32;
        let mut store = ParamSetStore::new();

        for item in iter_annexb_nals(head).take(32) {
            match item {
                Ok(nal) => match nal.header.nal_unit_type {
                    NalUnitType::Sps => {
                        if store.parse_sps_nal(nal.raw).is_ok() {
                            sps_ok = true;
                        } else {
                            invalid += 1;
                        }
                    }
                    NalUnitType::Pps => {
                        if store.parse_pps_nal(nal.raw).is_ok() {
                            pps_ok = true;
                        }
                    }
                    t if t.has_slice_header() => vcl_count += 1,
                    _ => {}
                },
                Err(_) => invalid += 1,
            }
        }
        if invalid > 2 {
            return 0;
        }
        if sps_ok && pps_ok && vcl_count > 0 {
            95
        } else if sps_ok && (pps_ok || vcl_count > 0) {
            85
        } else if sps_ok {
            65
        } else if vcl_count >= 2 {
            40
        } else {
            0
        }
    }

    fn next_unit(&mut self, buf: &[u8], eof: bool) -> Result<FrameStep> {
        let Some((first_sc, _)) = find_start_code(buf, 0) else {
            // Keep up to 3 trailing bytes in case a start code straddles a chunk boundary.
            if !eof && buf.len() > 3 {
                return Ok(FrameStep::Skip(buf.len() - 3));
            }
            return Ok(FrameStep::NeedMore);
        };
        if first_sc > 0 {
            return Ok(FrameStep::Skip(first_sc));
        }

        let mut seen_vcl = false;
        let mut is_key = false;
        let mut prev_slice: Option<SliceHeaderPrefix> = None;
        let mut updated_params: Option<StreamParams> = None;

        // Walk complete NALs (those followed by a subsequent start code, or at EOF).
        let mut sc = Some((0usize, find_start_code(buf, 0).expect("checked").1));
        while let Some((sc_off, sc_len)) = sc {
            let payload_start = sc_off + sc_len;
            let next_sc = find_start_code(buf, payload_start);
            if next_sc.is_none() && !eof {
                // This NAL is still incomplete; wait for more bytes before parsing it.
                return Ok(FrameStep::NeedMore);
            }
            let payload_end = next_sc.map_or(buf.len(), |(off, _)| off);
            let nal = &buf[payload_start..payload_end];

            if !nal.is_empty()
                && let Ok(hdr) = NalHeader::parse(nal[0])
            {
                let ntype = hdr.nal_unit_type;
                if seen_vcl && ntype.starts_new_au_after_vcl() {
                    return Ok(FrameStep::Emit {
                        range: 0..sc_off,
                        consume: sc_off,
                        is_key,
                        params: updated_params.map(|sp| FrameParams {
                            duration: None,
                            time_base: None,
                            stream_params: Some(sp),
                        }),
                    });
                }

                match ntype {
                    NalUnitType::Sps => {
                        if let Ok(sps) = self.store.parse_sps_nal(nal)
                            && self.last_emitted_sps.as_ref() != Some(sps)
                        {
                            updated_params =
                                Some(StreamParams::Video(sps.to_video_stream_params()));
                            self.last_emitted_sps = Some(sps.clone());
                        }
                    }
                    NalUnitType::Pps => {
                        let _ = self.store.parse_pps_nal(nal);
                    }
                    t if t.has_slice_header() => {
                        if let Ok(slice) =
                            SliceHeaderPrefix::parse_nal(nal, &self.store, &mut self.rbsp_scratch)
                        {
                            if let Some(ref prev) = prev_slice
                                && slice.is_new_picture_after(prev)
                            {
                                return Ok(FrameStep::Emit {
                                    range: 0..sc_off,
                                    consume: sc_off,
                                    is_key,
                                    params: updated_params.map(|sp| FrameParams {
                                        duration: None,
                                        time_base: None,
                                        stream_params: Some(sp),
                                    }),
                                });
                            }
                            if t == NalUnitType::IdrSlice {
                                is_key = true;
                            }
                            seen_vcl = true;
                            prev_slice = Some(slice);
                        } else {
                            if t == NalUnitType::IdrSlice {
                                is_key = true;
                            }
                            seen_vcl = true;
                        }
                    }
                    _ => {}
                }
            }

            sc = next_sc;
        }

        if eof && !buf.is_empty() {
            Ok(FrameStep::Emit {
                range: 0..buf.len(),
                consume: buf.len(),
                is_key,
                params: updated_params.map(|sp| FrameParams {
                    duration: None,
                    time_base: None,
                    stream_params: Some(sp),
                }),
            })
        } else {
            Ok(FrameStep::NeedMore)
        }
    }

    fn reset(&mut self) {
        self.store.clear();
        self.last_emitted_sps = None;
    }
}

/// ISO/IEC 14496-15 `AVCDecoderConfigurationRecord` (`avcC` extradata).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvcDecoderConfig {
    pub configuration_version: u8,
    pub avc_profile_indication: u8,
    pub profile_compatibility: u8,
    pub avc_level_indication: u8,
    /// Byte length of NAL unit length prefixes (`1..=4`).
    pub nal_length_size: u8,
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
    /// High-profile extension fields (present when profile is not 66/77/88).
    pub chroma_format: Option<u8>,
    pub bit_depth_luma_minus8: Option<u8>,
    pub bit_depth_chroma_minus8: Option<u8>,
    pub sps_ext: Vec<Vec<u8>>,
}

/// Parse an ISO/IEC 14496-15 `avcC` decoder configuration record.
pub fn parse_avcc(data: &[u8]) -> Result<AvcDecoderConfig> {
    if data.len() < 7 {
        return Err(Error::invalid_data(
            "h264/avcc",
            "avcC record shorter than 7 bytes",
        ));
    }
    let configuration_version = data[0];
    if configuration_version != 1 {
        return Err(Error::invalid_data(
            "h264/avcc",
            "unsupported avcC configurationVersion (expected 1)",
        ));
    }
    let avc_profile_indication = data[1];
    let profile_compatibility = data[2];
    let avc_level_indication = data[3];
    let nal_length_size = (data[4] & 0x03) + 1;
    let num_sps = (data[5] & 0x1F) as usize;

    let mut pos = 6usize;
    let mut sps = Vec::with_capacity(num_sps);
    for _ in 0..num_sps {
        if data.len() - pos < 2 {
            return Err(Error::invalid_data("h264/avcc", "truncated SPS length"));
        }
        let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if data.len() - pos < len {
            return Err(Error::invalid_data("h264/avcc", "truncated SPS payload"));
        }
        sps.push(data[pos..pos + len].to_vec());
        pos += len;
    }

    if data.len() - pos < 1 {
        return Err(Error::invalid_data("h264/avcc", "missing PPS count"));
    }
    let num_pps = data[pos] as usize;
    pos += 1;
    let mut pps = Vec::with_capacity(num_pps);
    for _ in 0..num_pps {
        if data.len() - pos < 2 {
            return Err(Error::invalid_data("h264/avcc", "truncated PPS length"));
        }
        let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if data.len() - pos < len {
            return Err(Error::invalid_data("h264/avcc", "truncated PPS payload"));
        }
        pps.push(data[pos..pos + len].to_vec());
        pos += len;
    }

    let mut chroma_format = None;
    let mut bit_depth_luma_minus8 = None;
    let mut bit_depth_chroma_minus8 = None;
    let mut sps_ext = Vec::new();

    if !matches!(avc_profile_indication, 66 | 77 | 88) && data.len() - pos >= 4 {
        chroma_format = Some(data[pos] & 0x03);
        bit_depth_luma_minus8 = Some(data[pos + 1] & 0x07);
        bit_depth_chroma_minus8 = Some(data[pos + 2] & 0x07);
        let num_ext = data[pos + 3] as usize;
        pos += 4;
        for _ in 0..num_ext {
            if data.len() - pos < 2 {
                return Err(Error::invalid_data("h264/avcc", "truncated SPS ext length"));
            }
            let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            if data.len() - pos < len {
                return Err(Error::invalid_data(
                    "h264/avcc",
                    "truncated SPS ext payload",
                ));
            }
            sps_ext.push(data[pos..pos + len].to_vec());
            pos += len;
        }
    }

    Ok(AvcDecoderConfig {
        configuration_version,
        avc_profile_indication,
        profile_compatibility,
        avc_level_indication,
        nal_length_size,
        sps,
        pps,
        chroma_format,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        sps_ext,
    })
}

/// Serialize an [`AvcDecoderConfig`] into an ISO/IEC 14496-15 `avcC` record
/// (matching FFmpeg's `ff_isom_write_avcc`).
pub fn build_avcc(config: &AvcDecoderConfig) -> Result<Vec<u8>> {
    if !(1..=4).contains(&config.nal_length_size) {
        return Err(Error::invalid_argument(
            "h264/avcc",
            "nal_length_size must be in 1..=4",
        ));
    }
    if config.sps.len() > 31 {
        return Err(Error::invalid_argument(
            "h264/avcc",
            "avcC supports at most 31 SPS NAL units",
        ));
    }
    if config.pps.len() > 255 {
        return Err(Error::invalid_argument(
            "h264/avcc",
            "avcC supports at most 255 PPS NAL units",
        ));
    }
    if config.sps_ext.len() > 255 {
        return Err(Error::invalid_argument(
            "h264/avcc",
            "avcC supports at most 255 SPS extension NAL units",
        ));
    }

    let mut out = Vec::with_capacity(32);
    out.push(config.configuration_version);
    out.push(config.avc_profile_indication);
    out.push(config.profile_compatibility);
    out.push(config.avc_level_indication);
    out.push(0xFC | (config.nal_length_size - 1));
    out.push(0xE0 | (config.sps.len() as u8));

    for s in &config.sps {
        if s.len() > u16::MAX as usize {
            return Err(Error::invalid_data("h264/avcc", "SPS exceeds 65535 bytes"));
        }
        out.extend_from_slice(&(s.len() as u16).to_be_bytes());
        out.extend_from_slice(s);
    }

    out.push(config.pps.len() as u8);
    for p in &config.pps {
        if p.len() > u16::MAX as usize {
            return Err(Error::invalid_data("h264/avcc", "PPS exceeds 65535 bytes"));
        }
        out.extend_from_slice(&(p.len() as u16).to_be_bytes());
        out.extend_from_slice(p);
    }

    if !matches!(config.avc_profile_indication, 66 | 77 | 88)
        && let (Some(cf), Some(bd_l), Some(bd_c)) = (
            config.chroma_format,
            config.bit_depth_luma_minus8,
            config.bit_depth_chroma_minus8,
        )
    {
        out.push(0xFC | (cf & 0x03));
        out.push(0xF8 | (bd_l & 0x07));
        out.push(0xF8 | (bd_c & 0x07));
        out.push(config.sps_ext.len() as u8);
        for ext in &config.sps_ext {
            if ext.len() > u16::MAX as usize {
                return Err(Error::invalid_data(
                    "h264/avcc",
                    "SPS ext exceeds 65535 bytes",
                ));
            }
            out.extend_from_slice(&(ext.len() as u16).to_be_bytes());
            out.extend_from_slice(ext);
        }
    }

    Ok(out)
}

/// Build an `avcC` extradata record directly from an Annex B byte stream containing SPS and PPS.
pub fn avcc_from_annexb(annexb: &[u8]) -> Result<Vec<u8>> {
    let mut sps_list = Vec::new();
    let mut pps_list = Vec::new();
    let mut sps_ext_list = Vec::new();

    for nal in iter_annexb_nals(annexb) {
        let nal = nal?;
        match nal.header.nal_unit_type {
            NalUnitType::Sps => sps_list.push(nal.raw.to_vec()),
            NalUnitType::Pps => pps_list.push(nal.raw.to_vec()),
            NalUnitType::SpsExt => sps_ext_list.push(nal.raw.to_vec()),
            _ => {}
        }
    }
    let first_sps = sps_list
        .first()
        .ok_or_else(|| Error::invalid_data("h264/avcc", "no SPS found in Annex B data"))?;
    if first_sps.len() < 4 || pps_list.is_empty() {
        return Err(Error::invalid_data(
            "h264/avcc",
            "incomplete SPS/PPS in Annex B data",
        ));
    }

    let parsed = Sps::parse_nal(first_sps)?;
    let is_ext_profile = !matches!(first_sps[1], 66 | 77 | 88);
    let config = AvcDecoderConfig {
        configuration_version: 1,
        avc_profile_indication: first_sps[1],
        profile_compatibility: first_sps[2],
        avc_level_indication: first_sps[3],
        nal_length_size: 4,
        sps: sps_list,
        pps: pps_list,
        chroma_format: is_ext_profile.then_some(parsed.chroma_format_idc),
        bit_depth_luma_minus8: is_ext_profile.then_some(parsed.bit_depth_luma - 8),
        bit_depth_chroma_minus8: is_ext_profile.then_some(parsed.bit_depth_chroma - 8),
        sps_ext: sps_ext_list,
    };
    build_avcc(&config)
}

/// Convert `avcC` parameter sets (SPS/PPS/SPS_EXT) into an Annex B parameter-set prefix.
pub fn avcc_extradata_to_annexb(config: &AvcDecoderConfig) -> Vec<u8> {
    let mut out = Vec::new();
    for s in &config.sps {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(s);
    }
    for s in &config.sps_ext {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(s);
    }
    for p in &config.pps {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(p);
    }
    out
}

/// Losslessly convert a length-prefixed AVCC packet to Annex B (4-byte start codes).
pub fn avcc_to_annexb(packet: &[u8], nal_length_size: u8) -> Result<Vec<u8>> {
    length_prefixed_to_annexb(packet, nal_length_size)
}

/// Losslessly convert an Annex B packet to length-prefixed AVCC format.
pub fn annexb_to_avcc(packet: &[u8], nal_length_size: u8) -> Result<Vec<u8>> {
    annexb_to_length_prefixed(packet, nal_length_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::RawStreamDemuxer;
    use wedeo_core::{BitWriter, Demuxer, ErrorPolicy, MemorySource, PacketFlags, RecvStatus};

    /// Build a synthetic SPS NAL with configurable profile, chroma, bit depth, crop, and VUI.
    #[allow(clippy::too_many_arguments)]
    fn make_sps_nal(
        profile_idc: u8,
        level_idc: u8,
        chroma_format_idc: u8,
        bit_depth: u8,
        mb_w: u32,
        mb_h: u32,
        crop: Option<(u32, u32, u32, u32)>,
        vui: bool,
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(u32::from(profile_idc), 8);
        w.write_bits(0, 8); // constraint flags + reserved
        w.write_bits(u32::from(level_idc), 8);
        w.write_ue(0); // sps_id

        if is_high_profile(profile_idc) {
            w.write_ue(u32::from(chroma_format_idc));
            if chroma_format_idc == 3 {
                w.write_bit(false); // separate_colour_plane_flag
            }
            w.write_ue(u32::from(bit_depth - 8)); // bit_depth_luma_minus8
            w.write_ue(u32::from(bit_depth - 8)); // bit_depth_chroma_minus8
            w.write_bit(false); // transform_bypass
            w.write_bit(false); // seq_scaling_matrix_present_flag
        }

        w.write_ue(0); // log2_max_frame_num_minus4 => 4
        w.write_ue(0); // pic_order_cnt_type = 0
        w.write_ue(2); // log2_max_pic_order_cnt_lsb_minus4 => 6
        w.write_ue(2); // max_num_ref_frames
        w.write_bit(false); // gaps_in_frame_num_allowed
        w.write_ue(mb_w - 1);
        w.write_ue(mb_h - 1);
        w.write_bit(true); // frame_mbs_only_flag
        w.write_bit(true); // direct_8x8_inference_flag

        if let Some((l, r, t, b)) = crop {
            w.write_bit(true);
            w.write_ue(l);
            w.write_ue(r);
            w.write_ue(t);
            w.write_ue(b);
        } else {
            w.write_bit(false);
        }

        w.write_bit(vui);
        if vui {
            w.write_bit(true); // aspect_ratio_info_present
            w.write_bits(255, 8); // EXTENDED_SAR
            w.write_bits(16, 16); // sar_width
            w.write_bits(15, 16); // sar_height
            w.write_bit(false); // overscan_info_present
            w.write_bit(true); // video_signal_type_present
            w.write_bits(5, 3); // video_format
            w.write_bit(false); // video_full_range_flag = limited
            w.write_bit(true); // colour_description_present
            w.write_bits(1, 8); // BT.709 primaries
            w.write_bits(1, 8); // BT.709 transfer
            w.write_bits(1, 8); // BT.709 matrix
            w.write_bit(true); // chroma_loc_info_present
            w.write_ue(2); // TopLeft (2 -> +1 = TopLeft)
            w.write_ue(2);
            w.write_bit(true); // timing_info_present
            w.write_bits(1001, 32); // num_units_in_tick
            w.write_bits(60000, 32); // time_scale => 30000/1001 fps
            w.write_bit(true); // fixed_frame_rate
            w.write_bit(false); // nal_hrd
            w.write_bit(false); // vcl_hrd
            w.write_bit(false); // pic_struct_present
            w.write_bit(true); // bitstream_restriction
            w.write_bit(true);
            w.write_ue(0);
            w.write_ue(0);
            w.write_ue(10);
            w.write_ue(10);
            w.write_ue(2); // num_reorder_frames
            w.write_ue(4); // max_dec_frame_buffering
        }
        w.write_bit(true); // rbsp_stop_one_bit
        let rbsp = w.finish();
        let mut nal = vec![0x67]; // ref_idc=3, type=7 (SPS)
        nal.extend_from_slice(&rbsp_to_nal(&rbsp));
        nal
    }

    fn make_pps_nal(pps_id: u32, sps_id: u32, transform_8x8: bool) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_ue(pps_id);
        w.write_ue(sps_id);
        w.write_bit(true); // CABAC
        w.write_bit(false); // bottom_field_pic_order_in_frame_present
        w.write_ue(0); // num_slice_groups_minus1
        w.write_ue(0); // ref_l0 - 1
        w.write_ue(0); // ref_l1 - 1
        w.write_bit(false); // weighted_pred
        w.write_bits(0, 2); // weighted_bipred_idc
        w.write_se(0); // init_qp_minus26
        w.write_se(0); // init_qs_minus26
        w.write_se(-2); // chroma_qp_index_offset
        w.write_bit(true); // deblocking_filter_control_present
        w.write_bit(false); // constrained_intra_pred
        w.write_bit(false); // redundant_pic_cnt_present
        if transform_8x8 {
            w.write_bit(true); // transform_8x8_mode
            w.write_bit(false); // pic_scaling_matrix_present
            w.write_se(2); // second_chroma_qp_index_offset
        }
        w.write_bit(true); // rbsp_stop_one_bit
        let rbsp = w.finish();
        let mut nal = vec![0x68];
        nal.extend_from_slice(&rbsp_to_nal(&rbsp));
        nal
    }

    #[allow(clippy::too_many_arguments)]
    fn make_slice_nal(
        is_idr: bool,
        nal_ref_idc: u8,
        first_mb: u32,
        slice_type: u32,
        pps_id: u32,
        frame_num: u32,
        idr_pic_id: Option<u32>,
        poc_lsb: u32,
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_ue(first_mb);
        w.write_ue(slice_type);
        w.write_ue(pps_id);
        w.write_bits(frame_num, 4); // log2_max_frame_num = 4
        if let Some(id) = idr_pic_id {
            w.write_ue(id);
        }
        w.write_bits(poc_lsb, 6); // log2_max_pic_order_cnt_lsb = 6
        w.write_bit(true);
        let rbsp = w.finish();
        let hdr = NalHeader {
            forbidden_zero_bit: false,
            nal_ref_idc,
            nal_unit_type: if is_idr {
                NalUnitType::IdrSlice
            } else {
                NalUnitType::Slice
            },
        };
        let mut nal = vec![hdr.to_byte()];
        nal.extend_from_slice(&rbsp_to_nal(&rbsp));
        nal
    }

    #[test]
    fn rbsp_emulation_prevention_roundtrip() {
        let rbsp = vec![
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00, 0x03, 0x04,
        ];
        let ebsp = rbsp_to_nal(&rbsp);
        assert!(!ebsp.windows(3).any(|w| w == [0, 0, 0] || w == [0, 0, 1]));
        let decoded = nal_to_rbsp(&ebsp);
        assert_eq!(decoded, rbsp);
    }

    #[test]
    fn sps_vui_odd_dim_high444_10bit_and_pps_parse() {
        // High 4:4:4 Predictive (244), 10-bit, 4:4:4 allows odd 1-pixel crops!
        // Coded 32x32, crop right=1, bottom=3 => display 31x29 (odd dimensions!)
        let sps_nal = make_sps_nal(244, 41, 3, 10, 2, 2, Some((0, 1, 0, 3)), true);
        let sps = Sps::parse_nal(&sps_nal).unwrap();
        assert_eq!(sps.profile_idc, 244);
        assert_eq!(sps.level_idc, 41);
        assert_eq!(sps.bit_depth_luma, 10);
        assert_eq!(sps.bit_depth_chroma, 10);
        assert_eq!(sps.coded_width(), 32);
        assert_eq!(sps.coded_height(), 32);
        assert_eq!(sps.display_width(), 31);
        assert_eq!(sps.display_height(), 29);
        assert_eq!(sps.pixel_layout(), PixelLayout::YUV444P10LE);
        assert_eq!(sps.vui.sar, Rational::new(16, 15));
        assert_eq!(sps.vui.color.primaries, ColorPrimaries::Bt709);
        assert_eq!(sps.vui.color.transfer, TransferCharacteristic::Bt709);
        assert_eq!(sps.vui.color.matrix, MatrixCoefficients::Bt709);
        assert_eq!(sps.vui.color.range, ColorRange::Limited);
        assert_eq!(sps.vui.color.chroma_location, ChromaLocation::TopLeft);
        assert_eq!(sps.frame_rate(), Some(Rational::new(30000, 1001)));
        assert_eq!(sps.num_reorder_frames, 2);

        let pps_nal = make_pps_nal(0, 0, true);
        let pps = Pps::parse_nal(&pps_nal, |id| (id == 0).then_some(&sps)).unwrap();
        assert!(pps.entropy_coding_mode_flag);
        assert!(pps.transform_8x8_mode);
        assert_eq!(pps.chroma_qp_index_offset, [-2, 2]);
        // init_qp = 0 + 26 + 6*(10-8) = 38
        assert_eq!(pps.init_qp, 38);
    }

    #[test]
    fn avcc_roundtrip_and_packet_bsf_lossless() {
        let sps_nal = make_sps_nal(100, 40, 1, 8, 80, 45, Some((0, 0, 0, 4)), false);
        let pps_nal = make_pps_nal(0, 0, true);
        let slice_nal = make_slice_nal(true, 3, 0, 2, 0, 0, Some(0), 0);

        let mut annexb = Vec::new();
        for nal in [&sps_nal, &pps_nal, &slice_nal] {
            annexb.extend_from_slice(&[0, 0, 0, 1]);
            annexb.extend_from_slice(nal);
        }

        let avcc_bytes = avcc_from_annexb(&annexb).unwrap();
        let cfg = parse_avcc(&avcc_bytes).unwrap();
        assert_eq!(cfg.avc_profile_indication, 100);
        assert_eq!(cfg.avc_level_indication, 40);
        assert_eq!(cfg.nal_length_size, 4);
        assert_eq!(cfg.sps, vec![sps_nal.clone()]);
        assert_eq!(cfg.pps, vec![pps_nal.clone()]);
        assert_eq!(cfg.chroma_format, Some(1));
        assert_eq!(cfg.bit_depth_luma_minus8, Some(0));
        assert_eq!(build_avcc(&cfg).unwrap(), avcc_bytes);

        let avcc_pkt = annexb_to_avcc(&annexb, 4).unwrap();
        let roundtrip_annexb = avcc_to_annexb(&avcc_pkt, 4).unwrap();
        assert_eq!(roundtrip_annexb, annexb);

        // Also verify 2-byte length prefix roundtrip
        let avcc_pkt2 = annexb_to_avcc(&annexb, 2).unwrap();
        assert_eq!(avcc_to_annexb(&avcc_pkt2, 2).unwrap(), annexb);
    }

    #[test]
    fn multi_slice_h264_au_splitting() {
        let sps_nal = make_sps_nal(100, 31, 1, 8, 20, 15, None, false);
        let pps_nal = make_pps_nal(0, 0, false);

        // AU 0 (IDR keyframe) composed of TWO slices (first_mb=0 and first_mb=100) with identical frame_num=0, idr_pic_id=0, poc_lsb=0
        let idr_slice0 = make_slice_nal(true, 3, 0, 2, 0, 0, Some(0), 0);
        let idr_slice1 = make_slice_nal(true, 3, 100, 2, 0, 0, Some(0), 0);

        // AU 1 (P frame) composed of TWO slices (first_mb=0 and first_mb=100) with frame_num=1, poc_lsb=2
        let p_slice0 = make_slice_nal(false, 2, 0, 0, 0, 1, None, 2);
        let p_slice1 = make_slice_nal(false, 2, 100, 0, 0, 1, None, 2);

        // AU 2 (back-to-back IDR frame with idr_pic_id=1, no SPS/PPS/AUD in between!)
        let idr2_slice0 = make_slice_nal(true, 3, 0, 2, 0, 0, Some(1), 0);

        let mut stream = Vec::new();
        for nal in [
            &sps_nal,
            &pps_nal,
            &idr_slice0,
            &idr_slice1,
            &p_slice0,
            &p_slice1,
            &idr2_slice0,
        ] {
            stream.extend_from_slice(&[0, 0, 0, 1]);
            stream.extend_from_slice(nal);
        }

        assert!(H264AnnexBFramer::probe(&stream) >= 90);

        let mut demux = RawStreamDemuxer::open(
            Box::new(MemorySource::new(stream)),
            H264AnnexBFramer::new(),
            ErrorPolicy::Strict,
        )
        .unwrap();

        let vp = demux.header().tracks[0].params.video().unwrap();
        assert_eq!(vp.width, 320);
        assert_eq!(vp.height, 240);

        let mut pkts = Vec::new();
        while let RecvStatus::Ready(p) = demux.read_packet().unwrap() {
            pkts.push(p);
        }
        assert_eq!(
            pkts.len(),
            3,
            "expected 3 AUs (multi-slice IDR, multi-slice P, second IDR)"
        );
        assert!(pkts[0].flags.contains(PacketFlags::KEY));
        assert!(!pkts[1].flags.contains(PacketFlags::KEY));
        assert!(pkts[2].flags.contains(PacketFlags::KEY));

        // Verify AU 0 has 4 NALs (SPS, PPS, slice0, slice1), AU 1 has 2 slices, AU 2 has 1 slice.
        assert_eq!(
            iter_annexb_nals(pkts[0].data.as_slice().unwrap()).count(),
            4
        );
        assert_eq!(
            iter_annexb_nals(pkts[1].data.as_slice().unwrap()).count(),
            2
        );
        assert_eq!(
            iter_annexb_nals(pkts[2].data.as_slice().unwrap()).count(),
            1
        );
    }

    #[test]
    fn sps_scaling_matrices_poc1_hrd_and_interlaced_crop() {
        let mut w = BitWriter::new();
        w.write_bits(100, 8); // High profile
        w.write_bits(0, 8);
        w.write_bits(40, 8); // level 4.0
        w.write_ue(1); // sps_id = 1
        w.write_ue(1); // chroma_format_idc = 4:2:0
        w.write_ue(0); // 8-bit luma
        w.write_ue(0); // 8-bit chroma
        w.write_bit(false); // bypass
        // seq_scaling_matrix_present_flag = 1
        w.write_bit(true);
        // List 0 (Intra 4x4 Y): present = 1, first delta = -8 => next = 0 => use JVT DEFAULT_SCALING4[0]
        w.write_bit(true);
        w.write_se(-8);
        // Lists 1..=5: not present (fallback chain)
        for _ in 1..6 {
            w.write_bit(false);
        }
        // 8x8 Intra Y (idx 6): present = 1, constant 12 (delta=+4, then 0 => repeat 12)
        w.write_bit(true);
        w.write_se(4); // 8 + 4 = 12
        w.write_se(-12); // 12 + (-12) = 0 => repeat last (12) for remaining 63 entries!
        // 8x8 Inter Y (idx 7): not present => fallback DEFAULT_SCALING8[1]
        w.write_bit(false);

        w.write_ue(1); // log2_max_frame_num = 5
        w.write_ue(1); // pic_order_cnt_type = 1
        w.write_bit(false); // delta_pic_order_always_zero_flag
        w.write_se(-4); // offset_for_non_ref_pic
        w.write_se(2); // offset_for_top_to_bottom_field
        w.write_ue(2); // cycle len = 2
        w.write_se(2);
        w.write_se(4);
        w.write_ue(4); // max_num_ref_frames
        w.write_bit(true); // gaps_in_frame_num_allowed
        w.write_ue(44); // mb_width = 45 (720)
        w.write_ue(17); // map_height = 18; frame_mbs_only=0 => mb_height = 36 (576)
        w.write_bit(false); // frame_mbs_only_flag = 0 (interlaced)
        w.write_bit(true); // mb_adaptive_frame_field_flag = 1 (MBAFF)
        w.write_bit(true); // direct_8x8_inference
        // Frame crop: 4:2:0 interlaced => step_x=2, step_y=4
        w.write_bit(true);
        w.write_ue(1); // left = 2
        w.write_ue(1); // right = 2
        w.write_ue(1); // top = 4
        w.write_ue(1); // bottom = 4
        // VUI with NAL HRD
        w.write_bit(true);
        w.write_bit(true); // aspect_ratio_info_present
        w.write_bits(12, 8); // SAR table 12 => 64/33
        w.write_bit(false); // overscan
        w.write_bit(false); // video_signal_type
        w.write_bit(false); // chroma_loc => defaults to Left
        w.write_bit(false); // timing_info
        w.write_bit(true); // nal_hrd_parameters_present
        w.write_ue(0); // cpb_cnt_minus1 = 0 => 1
        w.write_bits(4, 4); // bit_rate_scale
        w.write_bits(2, 4); // cpb_size_scale
        w.write_ue(999); // bit_rate_value_minus1
        w.write_ue(1999); // cpb_size_value_minus1
        w.write_bit(true); // cbr_flag[0]
        w.write_bits(23, 5); // initial_cpb_removal_delay_length_minus1
        w.write_bits(23, 5); // cpb_removal_delay_length_minus1
        w.write_bits(23, 5); // dpb_output_delay_length_minus1
        w.write_bits(24, 5); // time_offset_length
        w.write_bit(false); // vcl_hrd_present
        w.write_bit(false); // low_delay_hrd_flag
        w.write_bit(true); // pic_struct_present_flag
        w.write_bit(false); // bitstream_restriction_flag
        w.write_bit(true); // stop bit

        let rbsp = w.finish();
        let sps = Sps::parse_rbsp(&rbsp).unwrap();
        assert_eq!(sps.sps_id, 1);
        assert!(sps.scaling_matrix_present);
        assert_eq!(sps.scaling_matrix4[0], DEFAULT_SCALING4[0]);
        assert_eq!(sps.scaling_matrix4[2], DEFAULT_SCALING4[0]);
        assert_eq!(sps.scaling_matrix4[3], DEFAULT_SCALING4[1]);
        assert_eq!(sps.scaling_matrix8[0], [12u8; 64]);
        assert_eq!(sps.scaling_matrix8[3], DEFAULT_SCALING8[1]);
        assert_eq!(sps.pic_order_cnt_type, 1);
        assert_eq!(sps.offset_for_non_ref_pic, -4);
        assert_eq!(sps.offset_for_ref_frame, vec![2, 4]);
        assert!(!sps.frame_mbs_only_flag);
        assert!(sps.mb_adaptive_frame_field_flag);
        assert_eq!(sps.coded_width(), 720);
        assert_eq!(sps.coded_height(), 576);
        assert_eq!(
            sps.crop,
            CropRect {
                top: 4,
                bottom: 4,
                left: 2,
                right: 2
            }
        );
        assert_eq!(sps.display_width(), 716);
        assert_eq!(sps.display_height(), 568);
        assert_eq!(sps.vui.sar, Rational::new(64, 33));
        assert_eq!(sps.vui.color.chroma_location, ChromaLocation::Left);
        let hrd = sps.vui.nal_hrd.as_ref().unwrap();
        assert_eq!(hrd.cpb_cnt, 1);
        assert_eq!(hrd.bit_rate_value, vec![1000]);
        assert_eq!(hrd.cpb_size_value, vec![2000]);
        assert_eq!(hrd.cbr_flag_mask, 1);
    }

    #[test]
    fn malformed_h264_inputs_never_panic() {
        for bad in [
            &[][..],
            &[0x80],
            &[0x67],
            &[0x67, 0xFF, 0xFF],
            &[0x01, 0x00],
        ] {
            let _ = Sps::parse_nal(bad);
            let _ = parse_avcc(bad);
            let _ = avcc_to_annexb(bad, 4);
            let _ = annexb_to_avcc(bad, 4);
        }
    }
}
