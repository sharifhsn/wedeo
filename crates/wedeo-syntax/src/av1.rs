//! AV1 OBU parser, Sequence Header parser, raw OBU temporal-unit framer, and `av1C` helpers.
//!
//! Faithfully ports FFmpeg `n9.1` `libavcodec/av1_parse.h`, `libavcodec/cbs_av1_syntax_template.c`,
//! and `libavformat/av1.c`.

use std::ops::Range;

use wedeo_core::{
    AlphaMode, BitReader, BitWriter, ChromaLocation, CodecId, ColorDesc, ColorModel,
    ColorPrimaries, ColorRange, CompOrder, Endian, Error, MatrixCoefficients, PixelLayout,
    PlanePacking, Result, SampleDepth, StreamParams, Subsampling, TransferCharacteristic,
    VideoStreamParams,
};

use crate::raw::{ElementaryFramer, FrameParams, FrameStep};

/// AV1 OBU types (AV1 spec §6.2.2, `libavcodec/av1.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObuType {
    SequenceHeader,
    TemporalDelimiter,
    FrameHeader,
    TileGroup,
    Metadata,
    Frame,
    RedundantFrameHeader,
    TileList,
    Padding,
    Reserved(u8),
}

impl ObuType {
    pub const fn from_u8(v: u8) -> Self {
        match v & 0x0F {
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            other => Self::Reserved(other),
        }
    }

    pub const fn code(self) -> u8 {
        match self {
            Self::SequenceHeader => 1,
            Self::TemporalDelimiter => 2,
            Self::FrameHeader => 3,
            Self::TileGroup => 4,
            Self::Metadata => 5,
            Self::Frame => 6,
            Self::RedundantFrameHeader => 7,
            Self::TileList => 8,
            Self::Padding => 15,
            Self::Reserved(v) => v & 0x0F,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::SequenceHeader => "sequence_header",
            Self::TemporalDelimiter => "temporal_delimiter",
            Self::FrameHeader => "frame_header",
            Self::TileGroup => "tile_group",
            Self::Metadata => "metadata",
            Self::Frame => "frame",
            Self::RedundantFrameHeader => "redundant_frame_header",
            Self::TileList => "tile_list",
            Self::Padding => "padding",
            Self::Reserved(_) => "reserved",
        }
    }
}

/// Read an AV1 unsigned LEB128 integer (up to 8 bytes, value `<= u32::MAX`).
///
/// Returns `Ok(None)` if `data` ends mid-varint (more bytes needed), `Ok(Some((value, bytes_read)))`
/// on success, or `Err(InvalidData)` on malformed/overflowing varints.
pub fn try_read_leb128(data: &[u8]) -> Result<Option<(u64, usize)>> {
    let mut value = 0u64;
    for i in 0..8usize {
        let Some(&byte) = data.get(i) else {
            return Ok(None);
        };
        value |= u64::from(byte & 0x7F) << (i * 7);
        if (byte & 0x80) == 0 {
            if value > u64::from(u32::MAX) {
                return Err(Error::invalid_data(
                    "av1/leb128",
                    "LEB128 value exceeds u32::MAX",
                ));
            }
            return Ok(Some((value, i + 1)));
        }
    }
    Err(Error::invalid_data(
        "av1/leb128",
        "LEB128 exceeds 8-byte maximum length",
    ))
}

/// Read a complete AV1 LEB128 integer from `data`.
pub fn read_leb128(data: &[u8]) -> Result<(u64, usize)> {
    try_read_leb128(data)?
        .ok_or_else(|| Error::invalid_data("av1/leb128", "truncated LEB128 integer"))
}

/// Encode `value` as a minimal AV1 LEB128 integer.
pub fn write_leb128(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Parsed AV1 OBU header (`parse_obu_header` in `libavcodec/av1_parse.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObuHeader {
    pub obu_type: ObuType,
    pub extension_flag: bool,
    pub has_size_field: bool,
    pub temporal_id: u8,
    pub spatial_id: u8,
    /// Total byte length of the header (1 or 2 bytes + optional LEB128 size bytes).
    pub header_size: usize,
    /// Declared OBU payload size (or remaining slice length when `has_size_field == false`).
    pub payload_size: usize,
}

/// Try to parse an AV1 OBU header from `data`.
///
/// Returns `Ok(None)` when `data` is too short to hold the header + LEB128 size field.
pub fn try_parse_obu_header(data: &[u8]) -> Result<Option<ObuHeader>> {
    let Some(&b0) = data.first() else {
        return Ok(None);
    };
    if (b0 & 0x80) != 0 {
        return Err(Error::invalid_data(
            "av1/obu",
            "obu_forbidden_bit is non-zero",
        ));
    }
    let obu_type = ObuType::from_u8((b0 >> 3) & 0x0F);
    let extension_flag = (b0 & 0x04) != 0;
    let has_size_field = (b0 & 0x02) != 0;

    let mut pos = 1usize;
    let (temporal_id, spatial_id) = if extension_flag {
        let Some(&b1) = data.get(pos) else {
            return Ok(None);
        };
        pos += 1;
        ((b1 >> 5) & 0x07, (b1 >> 3) & 0x03)
    } else {
        (0, 0)
    };

    let payload_size = if has_size_field {
        let Some((sz, leb_len)) = try_read_leb128(&data[pos..])? else {
            return Ok(None);
        };
        pos += leb_len;
        sz as usize
    } else {
        data.len() - pos
    };

    Ok(Some(ObuHeader {
        obu_type,
        extension_flag,
        has_size_field,
        temporal_id,
        spatial_id,
        header_size: pos,
        payload_size,
    }))
}

/// Parse a complete AV1 OBU header from `data`.
pub fn parse_obu_header(data: &[u8]) -> Result<ObuHeader> {
    let hdr = try_parse_obu_header(data)?
        .ok_or_else(|| Error::invalid_data("av1/obu", "truncated OBU header"))?;
    if data.len().saturating_sub(hdr.header_size) < hdr.payload_size {
        return Err(Error::invalid_data(
            "av1/obu",
            "OBU payload exceeds remaining buffer",
        ));
    }
    Ok(hdr)
}

/// A parsed OBU slice inside an AV1 packet or stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObuSlice<'a> {
    pub header: ObuHeader,
    pub raw_range: Range<usize>,
    pub payload: &'a [u8],
}

/// Iterate over OBUs in an AV1 byte slice (`ff_av1_packet_split`).
pub fn iter_obus(data: &[u8]) -> impl Iterator<Item = Result<ObuSlice<'_>>> {
    struct ObuIter<'a> {
        data: &'a [u8],
        pos: usize,
        failed: bool,
    }
    impl<'a> Iterator for ObuIter<'a> {
        type Item = Result<ObuSlice<'a>>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.failed || self.pos >= self.data.len() {
                return None;
            }
            let start = self.pos;
            match parse_obu_header(&self.data[start..]) {
                Ok(header) => {
                    let payload_start = start + header.header_size;
                    let end = payload_start + header.payload_size;
                    self.pos = end;
                    Some(Ok(ObuSlice {
                        header,
                        raw_range: start..end,
                        payload: &self.data[payload_start..end],
                    }))
                }
                Err(e) => {
                    self.failed = true;
                    Some(Err(e))
                }
            }
        }
    }
    ObuIter {
        data,
        pos: 0,
        failed: false,
    }
}

/// Compute AV1 OBU bit length excluding trailing `1` bit and zero padding (`get_obu_bit_length`).
pub fn obu_bit_length(payload: &[u8], obu_type: ObuType) -> usize {
    if matches!(
        obu_type,
        ObuType::TileGroup | ObuType::TileList | ObuType::Frame
    ) {
        return payload.len() * 8;
    }
    let mut len = payload.len();
    while len > 0 && payload[len - 1] == 0 {
        len -= 1;
    }
    if len == 0 {
        return 0;
    }
    let last = payload[len - 1];
    len * 8 - (last.trailing_zeros() as usize + 1)
}

/// Parsed AV1 Sequence Header OBU (`cbs_av1_syntax_template.c` + `libavformat/av1.c`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: u8,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub frame_id_numbers_present: bool,
    pub use_128x128_superblock: bool,
    pub enable_order_hint: bool,
    pub order_hint_bits: u8,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    /// `0` = unknown, `1` = vertical (left), `2` = colocated (top-left).
    pub chroma_sample_position: u8,
    pub color: ColorDesc,
    pub film_grain_params_present: bool,
}

impl Av1SequenceHeader {
    /// Derive the orthogonal [`PixelLayout`] matching FFmpeg `av1_parser.c`.
    pub const fn pixel_layout(&self) -> PixelLayout {
        let depth = match self.bit_depth {
            8 => SampleDepth::U8,
            10 => SampleDepth::U10,
            12 => SampleDepth::U12,
            _ => SampleDepth::U8,
        };
        if self.monochrome {
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
        if matches!(self.color.matrix, MatrixCoefficients::Identity) {
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
        let subsampling = match (self.subsampling_x, self.subsampling_y) {
            (true, true) => Subsampling::Cs420,
            (true, false) => Subsampling::Cs422,
            _ => Subsampling::Cs444,
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

    /// Convert to [`VideoStreamParams`].
    pub fn to_video_stream_params(&self) -> VideoStreamParams {
        let mut params = VideoStreamParams::new(
            self.max_frame_width,
            self.max_frame_height,
            self.pixel_layout(),
        );
        params.color = self.color;
        params
    }
}

fn skip_uvlc(br: &mut BitReader<'_>) -> Result<()> {
    let mut leading_zeros = 0usize;
    while leading_zeros < 32 {
        if br.read_bool()? {
            break;
        }
        leading_zeros += 1;
    }
    if leading_zeros < 32 && leading_zeros > 0 {
        br.skip_bits(leading_zeros)?;
    }
    Ok(())
}

/// Parse an AV1 `OBU_SEQUENCE_HEADER` payload (without the OBU header).
pub fn parse_sequence_header(payload: &[u8]) -> Result<Av1SequenceHeader> {
    if payload.is_empty() {
        return Err(Error::invalid_data(
            "av1/seq_hdr",
            "empty sequence header OBU",
        ));
    }
    let mut br = BitReader::new(payload);

    let seq_profile = br.read_bits(3)? as u8;
    if seq_profile > 2 {
        return Err(Error::invalid_data(
            "av1/seq_hdr",
            "unsupported AV1 seq_profile > 2",
        ));
    }
    let still_picture = br.read_bool()?;
    let reduced_still_picture_header = br.read_bool()?;

    let mut seq_level_idx_0 = 0u8;
    let mut seq_tier_0 = 0u8;

    if reduced_still_picture_header {
        seq_level_idx_0 = br.read_bits(5)? as u8;
    } else {
        let timing_info_present = br.read_bool()?;
        let mut decoder_model_info_present = false;
        let mut buffer_delay_length_minus_1 = 0u32;

        if timing_info_present {
            br.skip_bits(32)?; // num_units_in_display_tick
            br.skip_bits(32)?; // time_scale
            if br.read_bool()? {
                skip_uvlc(&mut br)?; // num_ticks_per_picture_minus_1
            }
            decoder_model_info_present = br.read_bool()?;
            if decoder_model_info_present {
                buffer_delay_length_minus_1 = br.read_bits(5)?;
                br.skip_bits(32)?; // num_units_in_decoding_tick
                br.skip_bits(10)?; // buffer_removal_time_length_minus_1 + frame_presentation_time_length_minus_1
            }
        }

        let initial_display_delay_present = br.read_bool()?;
        let operating_points_cnt_minus_1 = br.read_bits(5)?;
        for i in 0..=operating_points_cnt_minus_1 {
            br.skip_bits(12)?; // operating_point_idc
            let level = br.read_bits(5)? as u8;
            let tier = if level > 7 { br.read_bit()? as u8 } else { 0 };
            if decoder_model_info_present && br.read_bool()? {
                let n = (buffer_delay_length_minus_1 + 1) as usize;
                br.skip_bits(n)?; // decoder_buffer_delay
                br.skip_bits(n)?; // encoder_buffer_delay
                br.skip_bits(1)?; // low_delay_mode_flag
            }
            if initial_display_delay_present && br.read_bool()? {
                br.skip_bits(4)?; // initial_display_delay_minus_1
            }
            if i == 0 {
                seq_level_idx_0 = level;
                seq_tier_0 = tier;
            }
        }
    }

    let frame_width_bits_minus_1 = br.read_bits(4)?;
    let frame_height_bits_minus_1 = br.read_bits(4)?;
    let max_frame_width = br.read_bits(frame_width_bits_minus_1 + 1)? + 1;
    let max_frame_height = br.read_bits(frame_height_bits_minus_1 + 1)? + 1;

    let frame_id_numbers_present = if reduced_still_picture_header {
        false
    } else {
        br.read_bool()?
    };
    if frame_id_numbers_present {
        br.skip_bits(7)?; // delta_frame_id_length_minus_2 (4) + additional_frame_id_length_minus_1 (3)
    }

    let use_128x128_superblock = br.read_bool()?;
    br.skip_bits(2)?; // enable_filter_intra, enable_intra_edge_filter

    let mut enable_order_hint = false;
    let mut order_hint_bits = 0u8;
    if !reduced_still_picture_header {
        br.skip_bits(4)?; // interintra, masked, warped, dual_filter
        enable_order_hint = br.read_bool()?;
        if enable_order_hint {
            br.skip_bits(2)?; // enable_jnt_comp, enable_ref_frame_mvs
        }
        let seq_force_screen_content = if br.read_bool()? { 2 } else { br.read_bit()? };
        if seq_force_screen_content > 0 && !br.read_bool()? {
            br.skip_bits(1)?; // seq_force_integer_mv
        }
        if enable_order_hint {
            order_hint_bits = br.read_bits(3)? as u8 + 1;
        }
    }

    br.skip_bits(3)?; // enable_superres, enable_cdef, enable_restoration

    // color_config()
    let high_bitdepth = br.read_bool()?;
    let twelve_bit = if seq_profile == 2 && high_bitdepth {
        br.read_bool()?
    } else {
        false
    };
    let bit_depth = 8 + if high_bitdepth { 2 } else { 0 } + if twelve_bit { 2 } else { 0 };

    let monochrome = if seq_profile == 1 {
        false
    } else {
        br.read_bool()?
    };

    let color_description_present = br.read_bool()?;
    let (primaries, transfer, matrix) = if color_description_present {
        let p = ColorPrimaries::from_cicp(br.read_bits(8)? as u8);
        let t = TransferCharacteristic::from_cicp(br.read_bits(8)? as u8);
        let m = MatrixCoefficients::from_cicp(br.read_bits(8)? as u8);
        (p, t, m)
    } else {
        (
            ColorPrimaries::Unspecified,
            TransferCharacteristic::Unspecified,
            MatrixCoefficients::Unspecified,
        )
    };

    let (range, subsampling_x, subsampling_y, chroma_sample_position) = if monochrome {
        let full = br.read_bool()?;
        (
            if full {
                ColorRange::Full
            } else {
                ColorRange::Limited
            },
            true,
            true,
            0u8,
        )
    } else if primaries == ColorPrimaries::Bt709
        && transfer == TransferCharacteristic::Srgb
        && matrix == MatrixCoefficients::Identity
    {
        br.skip_bits(1)?; // separate_uv_delta_q
        (ColorRange::Full, false, false, 0u8)
    } else {
        let full = br.read_bool()?;
        let (sx, sy) = match seq_profile {
            0 => (true, true),
            1 => (false, false),
            _ => {
                if twelve_bit {
                    let sx = br.read_bool()?;
                    let sy = if sx { br.read_bool()? } else { false };
                    (sx, sy)
                } else {
                    (true, false)
                }
            }
        };
        let csp = if sx && sy {
            br.read_bits(2)? as u8
        } else {
            0u8
        };
        br.skip_bits(1)?; // separate_uv_delta_q
        (
            if full {
                ColorRange::Full
            } else {
                ColorRange::Limited
            },
            sx,
            sy,
            csp,
        )
    };

    let chroma_location = match chroma_sample_position {
        1 => ChromaLocation::Left,
        2 => ChromaLocation::TopLeft,
        _ => ChromaLocation::Unspecified,
    };

    let film_grain_params_present = br.read_bool()?;

    Ok(Av1SequenceHeader {
        seq_profile,
        still_picture,
        reduced_still_picture_header,
        seq_level_idx_0,
        seq_tier_0,
        max_frame_width,
        max_frame_height,
        frame_id_numbers_present,
        use_128x128_superblock,
        enable_order_hint,
        order_hint_bits,
        bit_depth,
        monochrome,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
        color: ColorDesc {
            primaries,
            transfer,
            matrix,
            range,
            chroma_location,
        },
        film_grain_params_present,
    })
}

/// Peek whether an AV1 `Frame` or `FrameHeader` OBU payload is a keyframe (`frame_type == 0`).
fn is_av1_frame_keyframe(payload: &[u8], reduced_still: bool) -> bool {
    if reduced_still {
        return true;
    }
    let Some(&first) = payload.first() else {
        return false;
    };
    let show_existing = (first & 0x80) != 0;
    if show_existing {
        return false;
    }
    let frame_type = (first >> 5) & 0x03;
    frame_type == 0
}

/// Raw AV1 Low-Overhead Bitstream Format (OBU) temporal-unit framer implementing [`ElementaryFramer`].
pub struct Av1ObuFramer {
    seq_hdr: Option<Av1SequenceHeader>,
}

impl Av1ObuFramer {
    pub fn new() -> Self {
        Self { seq_hdr: None }
    }

    pub fn sequence_header(&self) -> Option<&Av1SequenceHeader> {
        self.seq_hdr.as_ref()
    }
}

impl Default for Av1ObuFramer {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementaryFramer for Av1ObuFramer {
    fn format_name(&self) -> &'static str {
        "av1"
    }

    fn codec_id(&self) -> CodecId {
        CodecId::Av1
    }

    fn initial_params(&self) -> StreamParams {
        StreamParams::Video(VideoStreamParams::new(0, 0, PixelLayout::YUV420P))
    }

    fn probe(head: &[u8]) -> u8 {
        if head.len() < 4 {
            return 0;
        }
        let mut pos = 0usize;
        let mut saw_td = false;
        let mut saw_valid_seq = false;
        let mut valid_obus = 0u32;

        while pos < head.len() && valid_obus < 16 {
            let Ok(Some(hdr)) = try_parse_obu_header(&head[pos..]) else {
                break;
            };
            if !hdr.has_size_field {
                return 0;
            }
            if matches!(hdr.obu_type, ObuType::Reserved(_)) {
                return 0;
            }
            let total = hdr.header_size.saturating_add(hdr.payload_size);
            if hdr.obu_type == ObuType::TemporalDelimiter {
                if hdr.payload_size != 0 {
                    return 0;
                }
                saw_td = true;
            }
            if pos + total > head.len() {
                break;
            }
            if hdr.obu_type == ObuType::SequenceHeader
                && parse_sequence_header(&head[pos + hdr.header_size..pos + total]).is_ok()
            {
                saw_valid_seq = true;
            }
            valid_obus += 1;
            pos += total;
        }

        if saw_td && saw_valid_seq {
            95
        } else if saw_valid_seq && valid_obus >= 2 {
            85
        } else if saw_valid_seq {
            65
        } else {
            0
        }
    }

    fn next_unit(&mut self, buf: &[u8], eof: bool) -> Result<FrameStep> {
        let mut pos = 0usize;
        let mut saw_any_obu = false;
        let mut saw_frame = false;
        let mut is_key = false;
        let mut updated_params: Option<StreamParams> = None;

        while pos < buf.len() {
            let hdr = match try_parse_obu_header(&buf[pos..]) {
                Ok(Some(h)) => h,
                Ok(None) => {
                    if !eof {
                        return Ok(FrameStep::NeedMore);
                    }
                    break;
                }
                Err(e) => {
                    if pos == 0 {
                        return Ok(FrameStep::Skip(1));
                    }
                    return Err(e);
                }
            };

            if !hdr.has_size_field {
                return Err(Error::invalid_data(
                    "av1/framer",
                    "raw AV1 OBU stream requires obu_has_size_field == 1",
                ));
            }
            let total = hdr.header_size.saturating_add(hdr.payload_size);
            if buf.len() - pos < total {
                if !eof {
                    return Ok(FrameStep::NeedMore);
                }
                break;
            }

            // A Temporal Delimiter after any previous OBU starts the next Temporal Unit.
            if saw_any_obu && hdr.obu_type == ObuType::TemporalDelimiter {
                return Ok(FrameStep::Emit {
                    range: 0..pos,
                    consume: pos,
                    is_key,
                    params: updated_params.map(|sp| FrameParams {
                        duration: None,
                        time_base: None,
                        stream_params: Some(sp),
                    }),
                });
            }

            // Streams without Temporal Delimiters split before a new SequenceHeader after a frame.
            if saw_frame && hdr.obu_type == ObuType::SequenceHeader {
                return Ok(FrameStep::Emit {
                    range: 0..pos,
                    consume: pos,
                    is_key,
                    params: updated_params.map(|sp| FrameParams {
                        duration: None,
                        time_base: None,
                        stream_params: Some(sp),
                    }),
                });
            }

            let payload = &buf[pos + hdr.header_size..pos + total];
            match hdr.obu_type {
                ObuType::SequenceHeader => {
                    if let Ok(seq) = parse_sequence_header(payload)
                        && self.seq_hdr.as_ref() != Some(&seq)
                    {
                        updated_params = Some(StreamParams::Video(seq.to_video_stream_params()));
                        self.seq_hdr = Some(seq);
                    }
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    let reduced = self
                        .seq_hdr
                        .as_ref()
                        .is_some_and(|s| s.reduced_still_picture_header);
                    if is_av1_frame_keyframe(payload, reduced) {
                        is_key = true;
                    }
                    saw_frame = true;
                }
                _ => {}
            }

            saw_any_obu = true;
            pos += total;
        }

        if eof && pos > 0 {
            Ok(FrameStep::Emit {
                range: 0..pos,
                consume: pos,
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
        self.seq_hdr = None;
    }
}

/// AV1-in-ISOBMFF `AV1CodecConfigurationRecord` (`av1C` extradata, version 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1CodecConfig {
    pub version: u8,
    pub seq_profile: u8,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: u8,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub initial_presentation_delay_minus_one: Option<u8>,
    pub config_obus: Vec<u8>,
}

impl Av1CodecConfig {
    pub const fn bit_depth(&self) -> u8 {
        8 + if self.high_bitdepth { 2 } else { 0 } + if self.twelve_bit { 2 } else { 0 }
    }
}

/// Parse an `av1C` (`AV1CodecConfigurationRecord`) byte slice.
pub fn parse_av1c(data: &[u8]) -> Result<Av1CodecConfig> {
    if data.len() < 4 {
        return Err(Error::invalid_data(
            "av1/av1c",
            "av1C record shorter than 4 bytes",
        ));
    }
    if (data[0] & 0x80) == 0 {
        return Err(Error::invalid_data("av1/av1c", "av1C marker bit must be 1"));
    }
    let version = data[0] & 0x7F;
    if version != 1 {
        return Err(Error::invalid_data(
            "av1/av1c",
            "unsupported av1C version (expected 1)",
        ));
    }
    let seq_profile = (data[1] >> 5) & 0x07;
    let seq_level_idx_0 = data[1] & 0x1F;
    let seq_tier_0 = (data[2] >> 7) & 0x01;
    let high_bitdepth = (data[2] & 0x40) != 0;
    let twelve_bit = (data[2] & 0x20) != 0;
    let monochrome = (data[2] & 0x10) != 0;
    let chroma_subsampling_x = (data[2] & 0x08) != 0;
    let chroma_subsampling_y = (data[2] & 0x04) != 0;
    let chroma_sample_position = data[2] & 0x03;
    let initial_presentation_delay_minus_one = if (data[3] & 0x10) != 0 {
        Some(data[3] & 0x0F)
    } else {
        None
    };

    Ok(Av1CodecConfig {
        version,
        seq_profile,
        seq_level_idx_0,
        seq_tier_0,
        high_bitdepth,
        twelve_bit,
        monochrome,
        chroma_subsampling_x,
        chroma_subsampling_y,
        chroma_sample_position,
        initial_presentation_delay_minus_one,
        config_obus: data[4..].to_vec(),
    })
}

/// Serialize an [`Av1CodecConfig`] into an `av1C` byte slice matching FFmpeg's `ff_isom_write_av1c`.
pub fn build_av1c(cfg: &Av1CodecConfig) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(4 + cfg.config_obus.len());
    w.write_bit(true); // marker = 1
    w.write_bits(u32::from(cfg.version & 0x7F), 7);
    w.write_bits(u32::from(cfg.seq_profile), 3);
    w.write_bits(u32::from(cfg.seq_level_idx_0), 5);
    w.write_bits(u32::from(cfg.seq_tier_0), 1);
    w.write_bit(cfg.high_bitdepth);
    w.write_bit(cfg.twelve_bit);
    w.write_bit(cfg.monochrome);
    w.write_bit(cfg.chroma_subsampling_x);
    w.write_bit(cfg.chroma_subsampling_y);
    w.write_bits(u32::from(cfg.chroma_sample_position), 2);
    w.write_bits(0, 3); // reserved
    if let Some(delay) = cfg.initial_presentation_delay_minus_one {
        w.write_bit(true);
        w.write_bits(u32::from(delay), 4);
    } else {
        w.write_bit(false);
        w.write_bits(0, 4);
    }
    let mut out = w.finish();
    out.extend_from_slice(&cfg.config_obus);
    out
}

/// Build an `av1C` record directly from a sequence of raw AV1 OBUs containing a Sequence Header.
pub fn av1c_from_obus(obus: &[u8], include_seq_header: bool) -> Result<Vec<u8>> {
    let mut seq: Option<(Av1SequenceHeader, &[u8])> = None;
    let mut metadata_obus = Vec::new();

    for item in iter_obus(obus) {
        let obu = item?;
        match obu.header.obu_type {
            ObuType::SequenceHeader => {
                if seq.is_some() {
                    return Err(Error::invalid_data(
                        "av1/av1c",
                        "multiple sequence header OBUs",
                    ));
                }
                let parsed = parse_sequence_header(obu.payload)?;
                seq = Some((parsed, &obus[obu.raw_range]));
            }
            ObuType::Metadata => {
                metadata_obus.extend_from_slice(&obus[obu.raw_range]);
            }
            _ => {}
        }
    }

    let (parsed, seq_raw) =
        seq.ok_or_else(|| Error::invalid_data("av1/av1c", "no sequence header OBU found"))?;
    let mut config_obus = Vec::new();
    if include_seq_header {
        config_obus.extend_from_slice(seq_raw);
    }
    config_obus.extend_from_slice(&metadata_obus);

    let cfg = Av1CodecConfig {
        version: 1,
        seq_profile: parsed.seq_profile,
        seq_level_idx_0: parsed.seq_level_idx_0,
        seq_tier_0: parsed.seq_tier_0,
        high_bitdepth: parsed.bit_depth > 8,
        twelve_bit: parsed.bit_depth == 12,
        monochrome: parsed.monochrome,
        chroma_subsampling_x: parsed.subsampling_x,
        chroma_subsampling_y: parsed.subsampling_y,
        chroma_sample_position: parsed.chroma_sample_position,
        initial_presentation_delay_minus_one: None,
        config_obus,
    };
    Ok(build_av1c(&cfg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::RawStreamDemuxer;
    use wedeo_core::{Demuxer, ErrorPolicy, MemorySource, PacketFlags, RecvStatus};

    fn make_obu(obu_type: ObuType, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        // forbidden=0, type(4), extension=0, has_size=1, reserved=0
        out.push((obu_type.code() << 3) | 0x02);
        write_leb128(payload.len() as u64, &mut out);
        out.extend_from_slice(payload);
        out
    }

    fn make_seq_hdr_payload(width: u32, height: u32, ten_bit: bool) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(0, 3); // seq_profile = Main (0)
        w.write_bit(false); // still_picture
        w.write_bit(false); // reduced_still_picture_header
        w.write_bit(false); // timing_info_present
        w.write_bit(false); // initial_display_delay_present
        w.write_bits(0, 5); // operating_points_cnt_minus_1 = 0
        w.write_bits(0, 12); // operating_point_idc
        w.write_bits(8, 5); // seq_level_idx = 8 (>7 => tier follows)
        w.write_bit(false); // seq_tier = 0
        w.write_bits(11, 4); // frame_width_bits_minus_1 => 12 bits
        w.write_bits(11, 4); // frame_height_bits_minus_1 => 12 bits
        w.write_bits(width - 1, 12);
        w.write_bits(height - 1, 12);
        w.write_bit(false); // frame_id_numbers_present
        w.write_bit(false); // use_128x128_superblock
        w.write_bit(true); // enable_filter_intra
        w.write_bit(true); // enable_intra_edge_filter
        w.write_bits(0, 4); // interintra, masked, warped, dual_filter
        w.write_bit(true); // enable_order_hint
        w.write_bit(true); // enable_jnt_comp
        w.write_bit(true); // enable_ref_frame_mvs
        w.write_bit(true); // seq_choose_screen_content_tools
        w.write_bit(true); // seq_choose_integer_mv
        w.write_bits(6, 3); // order_hint_bits_minus_1 => 7 bits
        w.write_bit(false); // enable_superres
        w.write_bit(true); // enable_cdef
        w.write_bit(true); // enable_restoration
        // color_config:
        w.write_bit(ten_bit); // high_bitdepth
        w.write_bit(false); // monochrome
        w.write_bit(true); // color_description_present
        w.write_bits(9, 8); // BT.2020 primaries
        w.write_bits(16, 8); // PQ transfer
        w.write_bits(9, 8); // BT.2020 NCL matrix
        w.write_bit(false); // limited range
        w.write_bits(2, 2); // chroma_sample_position = colocated (TopLeft)
        w.write_bit(false); // separate_uv_delta_q
        w.write_bit(false); // film_grain_params_present
        w.write_bit(true); // trailing_one_bit
        w.finish()
    }

    #[test]
    fn leb128_roundtrip_and_limits() {
        for &v in &[0u64, 1, 127, 128, 16384, 0x0012_3456, u64::from(u32::MAX)] {
            let mut buf = Vec::new();
            write_leb128(v, &mut buf);
            let (decoded, len) = read_leb128(&buf).unwrap();
            assert_eq!(decoded, v);
            assert_eq!(len, buf.len());
        }
        assert!(read_leb128(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]).is_err());
    }

    #[test]
    fn av1_seq_hdr_av1c_and_obu_framer() {
        let seq_payload = make_seq_hdr_payload(1920, 1080, true);
        let seq_obu = make_obu(ObuType::SequenceHeader, &seq_payload);
        let td_obu = make_obu(ObuType::TemporalDelimiter, &[]);
        // Keyframe OBU: show_existing=0, frame_type=0 (KEY) => byte 0 = 0x00
        let kf_obu = make_obu(ObuType::Frame, &[0x00, 0x10, 0x20]);
        // Inter frame OBU: show_existing=0, frame_type=1 (INTER) => byte 0 = 0x20
        let inter_obu = make_obu(ObuType::Frame, &[0x20, 0x30, 0x40]);

        let parsed = parse_sequence_header(&seq_payload).unwrap();
        assert_eq!(parsed.max_frame_width, 1920);
        assert_eq!(parsed.max_frame_height, 1080);
        assert_eq!(parsed.bit_depth, 10);
        assert_eq!(parsed.pixel_layout(), PixelLayout::YUV420P10LE);
        assert_eq!(parsed.color.primaries, ColorPrimaries::Bt2020);
        assert_eq!(parsed.color.transfer, TransferCharacteristic::Pq);
        assert_eq!(parsed.color.matrix, MatrixCoefficients::Bt2020Ncl);
        assert_eq!(parsed.color.chroma_location, ChromaLocation::TopLeft);

        // av1C roundtrip
        let av1c = av1c_from_obus(&seq_obu, true).unwrap();
        let cfg = parse_av1c(&av1c).unwrap();
        assert_eq!(cfg.bit_depth(), 10);
        assert_eq!(cfg.seq_profile, 0);
        assert_eq!(cfg.seq_level_idx_0, 8);
        assert_eq!(cfg.config_obus, seq_obu);
        assert_eq!(build_av1c(&cfg), av1c);

        // Frame two Temporal Units through RawStreamDemuxer
        let mut stream = Vec::new();
        stream.extend_from_slice(&td_obu);
        stream.extend_from_slice(&seq_obu);
        stream.extend_from_slice(&kf_obu);
        stream.extend_from_slice(&td_obu);
        stream.extend_from_slice(&inter_obu);

        assert!(Av1ObuFramer::probe(&stream) >= 90);

        let mut demux = RawStreamDemuxer::open(
            Box::new(MemorySource::new(stream)),
            Av1ObuFramer::new(),
            ErrorPolicy::Strict,
        )
        .unwrap();

        let vp = demux.header().tracks[0].params.video().unwrap();
        assert_eq!(vp.width, 1920);
        assert_eq!(vp.height, 1080);

        let mut pkts = Vec::new();
        while let RecvStatus::Ready(p) = demux.read_packet().unwrap() {
            pkts.push(p);
        }
        assert_eq!(pkts.len(), 2);
        assert!(pkts[0].flags.contains(PacketFlags::KEY));
        assert!(!pkts[1].flags.contains(PacketFlags::KEY));
    }

    #[test]
    fn av1_reduced_still_srgb_and_extension_header() {
        // Profile 1 (High), reduced_still_picture_header=1, sRGB identity RGB
        let mut w = BitWriter::new();
        w.write_bits(1, 3); // seq_profile = 1 (High => mono_chrome inferred 0)
        w.write_bit(true); // still_picture
        w.write_bit(true); // reduced_still_picture_header
        w.write_bits(4, 5); // seq_level_idx[0] = 4
        w.write_bits(7, 4); // 8-bit width
        w.write_bits(7, 4); // 8-bit height
        w.write_bits(199, 8); // 200 px
        w.write_bits(149, 8); // 150 px
        w.write_bit(false); // use_128x128_superblock
        w.write_bit(false); // enable_filter_intra
        w.write_bit(false); // enable_intra_edge_filter
        w.write_bit(false); // enable_superres
        w.write_bit(false); // enable_cdef
        w.write_bit(false); // enable_restoration
        // color_config
        w.write_bit(false); // 8-bit
        w.write_bit(true); // color_description_present
        w.write_bits(1, 8); // BT.709 primaries
        w.write_bits(13, 8); // sRGB transfer
        w.write_bits(0, 8); // Identity matrix => RGB shortcut (full range, 4:4:4)
        w.write_bit(false); // separate_uv_delta_q
        w.write_bit(false); // film_grain_params_present
        w.write_bit(true); // trailing_one_bit
        let seq = parse_sequence_header(&w.finish()).unwrap();
        assert!(seq.reduced_still_picture_header);
        assert_eq!(seq.max_frame_width, 200);
        assert_eq!(seq.max_frame_height, 150);
        assert_eq!(seq.pixel_layout(), PixelLayout::GBRP);
        assert_eq!(seq.color.range, ColorRange::Full);

        // OBU with extension header: type=6 (Frame), ext=1, has_size=1 => (6<<3)|0x06 = 0x36
        // extension byte: temporal_id=3 (011), spatial_id=2 (10), reserved=000 => 0b0111_0000 = 0x70
        let ext_obu = [0x36, 0x70, 0x02, 0xAA, 0xBB];
        let hdr = parse_obu_header(&ext_obu).unwrap();
        assert_eq!(hdr.obu_type, ObuType::Frame);
        assert!(hdr.extension_flag);
        assert_eq!(hdr.temporal_id, 3);
        assert_eq!(hdr.spatial_id, 2);
        assert_eq!(hdr.header_size, 3);
        assert_eq!(hdr.payload_size, 2);
    }
}
