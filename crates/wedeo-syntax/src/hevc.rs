//! HEVC / H.265 NAL header parser, Annex B AU framer, and `hvcC` configuration helpers.
//!
//! Faithfully ports FFmpeg `n9.1` `libavcodec/hevc/parser.c` (`hevc_find_frame_end`) and
//! `libavformat/hevc.c` (`hvcc_write`).

use wedeo_core::{CodecId, Error, PixelLayout, Result, StreamParams, VideoStreamParams};

use crate::raw::{ElementaryFramer, FrameStep};
use crate::startcode::{
    AnnexBIter, RawAnnexBNal, annexb_to_length_prefixed, find_start_code, length_prefixed_to_annexb,
};

/// HEVC NAL unit types (ITU-T H.265 Table 7-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HevcNalUnitType {
    TrailN,
    TrailR,
    TsaN,
    TsaR,
    StsaN,
    StsaR,
    RadlN,
    RadlR,
    RaslN,
    RaslR,
    BlaWLp,
    BlaWRadl,
    BlaNLp,
    IdrWRadl,
    IdrNLp,
    CraNut,
    Vps,
    Sps,
    Pps,
    Aud,
    Eos,
    Eob,
    Fd,
    SeiPrefix,
    SeiSuffix,
    Other(u8),
}

impl HevcNalUnitType {
    pub const fn from_u8(v: u8) -> Self {
        match v & 0x3F {
            0 => Self::TrailN,
            1 => Self::TrailR,
            2 => Self::TsaN,
            3 => Self::TsaR,
            4 => Self::StsaN,
            5 => Self::StsaR,
            6 => Self::RadlN,
            7 => Self::RadlR,
            8 => Self::RaslN,
            9 => Self::RaslR,
            16 => Self::BlaWLp,
            17 => Self::BlaWRadl,
            18 => Self::BlaNLp,
            19 => Self::IdrWRadl,
            20 => Self::IdrNLp,
            21 => Self::CraNut,
            32 => Self::Vps,
            33 => Self::Sps,
            34 => Self::Pps,
            35 => Self::Aud,
            36 => Self::Eos,
            37 => Self::Eob,
            38 => Self::Fd,
            39 => Self::SeiPrefix,
            40 => Self::SeiSuffix,
            other => Self::Other(other),
        }
    }

    pub const fn code(self) -> u8 {
        match self {
            Self::TrailN => 0,
            Self::TrailR => 1,
            Self::TsaN => 2,
            Self::TsaR => 3,
            Self::StsaN => 4,
            Self::StsaR => 5,
            Self::RadlN => 6,
            Self::RadlR => 7,
            Self::RaslN => 8,
            Self::RaslR => 9,
            Self::BlaWLp => 16,
            Self::BlaWRadl => 17,
            Self::BlaNLp => 18,
            Self::IdrWRadl => 19,
            Self::IdrNLp => 20,
            Self::CraNut => 21,
            Self::Vps => 32,
            Self::Sps => 33,
            Self::Pps => 34,
            Self::Aud => 35,
            Self::Eos => 36,
            Self::Eob => 37,
            Self::Fd => 38,
            Self::SeiPrefix => 39,
            Self::SeiSuffix => 40,
            Self::Other(v) => v & 0x3F,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::TrailN => "trail_n",
            Self::TrailR => "trail_r",
            Self::TsaN => "tsa_n",
            Self::TsaR => "tsa_r",
            Self::StsaN => "stsa_n",
            Self::StsaR => "stsa_r",
            Self::RadlN => "radl_n",
            Self::RadlR => "radl_r",
            Self::RaslN => "rasl_n",
            Self::RaslR => "rasl_r",
            Self::BlaWLp => "bla_w_lp",
            Self::BlaWRadl => "bla_w_radl",
            Self::BlaNLp => "bla_n_lp",
            Self::IdrWRadl => "idr_w_radl",
            Self::IdrNLp => "idr_n_lp",
            Self::CraNut => "cra",
            Self::Vps => "vps",
            Self::Sps => "sps",
            Self::Pps => "pps",
            Self::Aud => "aud",
            Self::Eos => "eos",
            Self::Eob => "eob",
            Self::Fd => "filler",
            Self::SeiPrefix => "sei_prefix",
            Self::SeiSuffix => "sei_suffix",
            Self::Other(_) => "other",
        }
    }

    #[inline]
    pub const fn is_vcl(self) -> bool {
        self.code() <= 31
    }

    /// Returns `true` for Intra Random Access Point NALs (`16..=23`, FFmpeg `IS_IRAP_NAL`).
    #[inline]
    pub const fn is_irap(self) -> bool {
        let c = self.code();
        c >= 16 && c <= 23
    }

    /// Returns `true` for IDR NALs (`IDR_W_RADL` or `IDR_N_LP`).
    #[inline]
    pub const fn is_idr(self) -> bool {
        matches!(self, Self::IdrWRadl | Self::IdrNLp)
    }

    /// Matches FFmpeg `hevc_find_frame_end` AU-prefix NAL types.
    #[inline]
    pub const fn starts_new_au_after_vcl(self) -> bool {
        let c = self.code();
        (c >= 32 && c <= 37) || c == 39 || (c >= 41 && c <= 44) || (c >= 48 && c <= 55)
    }

    /// Matches FFmpeg `hevc_find_frame_end` VCL slice types (`<= RASL_R || BLA_W_LP..=CRA_NUT`).
    #[inline]
    pub const fn is_coded_slice(self) -> bool {
        let c = self.code();
        c <= 9 || (c >= 16 && c <= 21)
    }
}

/// Decoded 2-byte HEVC NAL unit header (H.265 §7.3.1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HevcNalHeader {
    pub forbidden_zero_bit: bool,
    pub nal_unit_type: HevcNalUnitType,
    pub nuh_layer_id: u8,
    pub nuh_temporal_id_plus1: u8,
}

impl HevcNalHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 2 {
            return Err(Error::invalid_data(
                "hevc/nal",
                "HEVC NAL header shorter than 2 bytes",
            ));
        }
        let b0 = bytes[0];
        let b1 = bytes[1];
        let forbidden_zero_bit = (b0 & 0x80) != 0;
        if forbidden_zero_bit {
            return Err(Error::invalid_data(
                "hevc/nal",
                "HEVC NAL forbidden_zero_bit is set",
            ));
        }
        let nal_unit_type = HevcNalUnitType::from_u8((b0 >> 1) & 0x3F);
        let nuh_layer_id = ((b0 & 0x01) << 5) | ((b1 >> 3) & 0x1F);
        let nuh_temporal_id_plus1 = b1 & 0x07;
        if nuh_temporal_id_plus1 == 0 {
            return Err(Error::invalid_data(
                "hevc/nal",
                "HEVC nuh_temporal_id_plus1 must be >= 1",
            ));
        }
        Ok(Self {
            forbidden_zero_bit,
            nal_unit_type,
            nuh_layer_id,
            nuh_temporal_id_plus1,
        })
    }

    pub const fn to_bytes(self) -> [u8; 2] {
        let b0 = ((self.forbidden_zero_bit as u8) << 7)
            | ((self.nal_unit_type.code() & 0x3F) << 1)
            | ((self.nuh_layer_id >> 5) & 0x01);
        let b1 = ((self.nuh_layer_id & 0x1F) << 3) | (self.nuh_temporal_id_plus1 & 0x07);
        [b0, b1]
    }
}

/// An HEVC Annex B NAL unit with parsed 2-byte [`HevcNalHeader`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HevcAnnexBNal<'a> {
    pub start_code_offset: usize,
    pub start_code_len: usize,
    pub header: HevcNalHeader,
    pub raw: &'a [u8],
}

/// Iterate over HEVC Annex B NAL units in `data`.
pub fn iter_annexb_nals(data: &[u8]) -> impl Iterator<Item = Result<HevcAnnexBNal<'_>>> {
    AnnexBIter::new(data).filter_map(|raw: RawAnnexBNal<'_>| {
        if raw.payload.is_empty() {
            return None;
        }
        Some(
            HevcNalHeader::parse(raw.payload).map(|header| HevcAnnexBNal {
                start_code_offset: raw.start_code_offset,
                start_code_len: raw.start_code_len,
                header,
                raw: raw.payload,
            }),
        )
    })
}

/// HEVC Annex B Access Unit framer implementing [`ElementaryFramer`], faithfully porting
/// FFmpeg `n9.1` `hevc_find_frame_end` (`first_slice_segment_in_pic_flag` + NAL ordering).
#[derive(Debug, Default)]
pub struct HevcAnnexBFramer;

impl HevcAnnexBFramer {
    pub fn new() -> Self {
        Self
    }
}

impl ElementaryFramer for HevcAnnexBFramer {
    fn format_name(&self) -> &'static str {
        "hevc"
    }

    fn codec_id(&self) -> CodecId {
        CodecId::Hevc
    }

    fn initial_params(&self) -> StreamParams {
        StreamParams::Video(VideoStreamParams::new(0, 0, PixelLayout::YUV420P))
    }

    fn probe(head: &[u8]) -> u8 {
        let mut vps = false;
        let mut sps = false;
        let mut pps = false;
        let mut vcl = 0u32;
        let mut invalid = 0u32;

        for item in iter_annexb_nals(head).take(32) {
            match item {
                Ok(nal) => match nal.header.nal_unit_type {
                    HevcNalUnitType::Vps => vps = true,
                    HevcNalUnitType::Sps => sps = true,
                    HevcNalUnitType::Pps => pps = true,
                    t if t.is_coded_slice() => vcl += 1,
                    _ => {}
                },
                Err(_) => invalid += 1,
            }
        }
        if invalid > 2 {
            return 0;
        }
        if vps && sps && pps && vcl > 0 {
            95
        } else if (vps || sps) && pps && vcl > 0 {
            85
        } else if vps && sps {
            70
        } else if vcl >= 2 {
            35
        } else {
            0
        }
    }

    fn next_unit(&mut self, buf: &[u8], eof: bool) -> Result<FrameStep> {
        let Some((first_sc, first_sc_len)) = find_start_code(buf, 0) else {
            if !eof && buf.len() > 3 {
                return Ok(FrameStep::Skip(buf.len() - 3));
            }
            return Ok(FrameStep::NeedMore);
        };
        if first_sc > 0 {
            return Ok(FrameStep::Skip(first_sc));
        }

        let mut frame_start_found = false;
        let mut is_key = false;
        let mut sc = Some((0usize, first_sc_len));

        while let Some((sc_off, sc_len)) = sc {
            let payload_start = sc_off + sc_len;
            let next_sc = find_start_code(buf, payload_start);
            if next_sc.is_none() && !eof {
                return Ok(FrameStep::NeedMore);
            }
            let payload_end = next_sc.map_or(buf.len(), |(off, _)| off);
            let nal = &buf[payload_start..payload_end];

            if let Ok(hdr) = HevcNalHeader::parse(nal)
                && hdr.nuh_layer_id == 0
            {
                let nut = hdr.nal_unit_type;
                if nut.starts_new_au_after_vcl() {
                    if frame_start_found {
                        return Ok(FrameStep::Emit {
                            range: 0..sc_off,
                            consume: sc_off,
                            is_key,
                            params: None,
                        });
                    }
                } else if nut.is_coded_slice() {
                    let first_slice_segment_in_pic_flag =
                        nal.get(2).is_some_and(|&b| (b & 0x80) != 0);
                    if first_slice_segment_in_pic_flag {
                        if frame_start_found {
                            return Ok(FrameStep::Emit {
                                range: 0..sc_off,
                                consume: sc_off,
                                is_key,
                                params: None,
                            });
                        }
                        frame_start_found = true;
                    }
                    if nut.is_irap() {
                        is_key = true;
                    }
                }
            }

            sc = next_sc;
        }

        if eof && !buf.is_empty() {
            Ok(FrameStep::Emit {
                range: 0..buf.len(),
                consume: buf.len(),
                is_key,
                params: None,
            })
        } else {
            Ok(FrameStep::NeedMore)
        }
    }

    fn reset(&mut self) {}
}

/// One NAL unit array entry inside an ISO/IEC 14496-15 `hvcC` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HvccNalArray {
    pub array_completeness: bool,
    pub nal_unit_type: HevcNalUnitType,
    pub nalus: Vec<Vec<u8>>,
}

/// ISO/IEC 14496-15 `HEVCDecoderConfigurationRecord` (`hvcC` extradata).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcDecoderConfig {
    pub configuration_version: u8,
    pub general_profile_space: u8,
    pub general_tier_flag: bool,
    pub general_profile_idc: u8,
    pub general_profile_compatibility_flags: u32,
    /// 48-bit constraint indicator flags stored in the low 48 bits.
    pub general_constraint_indicator_flags: u64,
    pub general_level_idc: u8,
    pub min_spatial_segmentation_idc: u16,
    pub parallelism_type: u8,
    pub chroma_format_idc: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub avg_frame_rate: u16,
    pub constant_frame_rate: u8,
    pub num_temporal_layers: u8,
    pub temporal_id_nested: bool,
    /// Byte length of NAL unit length prefixes (`1..=4`).
    pub nal_length_size: u8,
    pub arrays: Vec<HvccNalArray>,
}

/// Parse an ISO/IEC 14496-15 `hvcC` decoder configuration record (minimum 23 bytes).
pub fn parse_hvcc(data: &[u8]) -> Result<HevcDecoderConfig> {
    if data.len() < 23 {
        return Err(Error::invalid_data(
            "hevc/hvcc",
            "hvcC record shorter than 23 bytes",
        ));
    }
    let configuration_version = data[0];
    if configuration_version != 1 {
        return Err(Error::invalid_data(
            "hevc/hvcc",
            "unsupported hvcC configurationVersion (expected 1)",
        ));
    }
    let general_profile_space = (data[1] >> 6) & 0x03;
    let general_tier_flag = (data[1] & 0x20) != 0;
    let general_profile_idc = data[1] & 0x1F;
    let general_profile_compatibility_flags =
        u32::from_be_bytes([data[2], data[3], data[4], data[5]]);
    let general_constraint_indicator_flags = (u64::from(data[6]) << 40)
        | (u64::from(data[7]) << 32)
        | (u64::from(data[8]) << 24)
        | (u64::from(data[9]) << 16)
        | (u64::from(data[10]) << 8)
        | u64::from(data[11]);
    let general_level_idc = data[12];
    let min_spatial_segmentation_idc = u16::from_be_bytes([data[13], data[14]]) & 0x0FFF;
    let parallelism_type = data[15] & 0x03;
    let chroma_format_idc = data[16] & 0x03;
    let bit_depth_luma_minus8 = data[17] & 0x07;
    let bit_depth_chroma_minus8 = data[18] & 0x07;
    let avg_frame_rate = u16::from_be_bytes([data[19], data[20]]);
    let constant_frame_rate = (data[21] >> 6) & 0x03;
    let num_temporal_layers = (data[21] >> 3) & 0x07;
    let temporal_id_nested = (data[21] & 0x04) != 0;
    let nal_length_size = (data[21] & 0x03) + 1;
    let num_arrays = data[22] as usize;

    let mut pos = 23usize;
    let mut arrays = Vec::with_capacity(num_arrays);
    for _ in 0..num_arrays {
        if data.len() - pos < 3 {
            return Err(Error::invalid_data(
                "hevc/hvcc",
                "truncated hvcC array header",
            ));
        }
        let array_completeness = (data[pos] & 0x80) != 0;
        let nal_unit_type = HevcNalUnitType::from_u8(data[pos] & 0x3F);
        let num_nalus = u16::from_be_bytes([data[pos + 1], data[pos + 2]]) as usize;
        pos += 3;

        let mut nalus = Vec::with_capacity(num_nalus);
        for _ in 0..num_nalus {
            if data.len() - pos < 2 {
                return Err(Error::invalid_data(
                    "hevc/hvcc",
                    "truncated hvcC NAL unit length",
                ));
            }
            let len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            if data.len() - pos < len {
                return Err(Error::invalid_data(
                    "hevc/hvcc",
                    "truncated hvcC NAL unit payload",
                ));
            }
            nalus.push(data[pos..pos + len].to_vec());
            pos += len;
        }
        arrays.push(HvccNalArray {
            array_completeness,
            nal_unit_type,
            nalus,
        });
    }

    Ok(HevcDecoderConfig {
        configuration_version,
        general_profile_space,
        general_tier_flag,
        general_profile_idc,
        general_profile_compatibility_flags,
        general_constraint_indicator_flags,
        general_level_idc,
        min_spatial_segmentation_idc,
        parallelism_type,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        avg_frame_rate,
        constant_frame_rate,
        num_temporal_layers,
        temporal_id_nested,
        nal_length_size,
        arrays,
    })
}

/// Serialize an [`HevcDecoderConfig`] into an ISO/IEC 14496-15 `hvcC` record
/// (matching FFmpeg's `hvcc_write` in `libavformat/hevc.c`).
pub fn build_hvcc(cfg: &HevcDecoderConfig) -> Result<Vec<u8>> {
    if !(1..=4).contains(&cfg.nal_length_size) {
        return Err(Error::invalid_argument(
            "hevc/hvcc",
            "nal_length_size must be in 1..=4",
        ));
    }
    if cfg.arrays.len() > u8::MAX as usize {
        return Err(Error::invalid_argument(
            "hevc/hvcc",
            "hvcC supports at most 255 NAL arrays",
        ));
    }

    let mut out = Vec::with_capacity(64);
    out.push(cfg.configuration_version);
    out.push(
        ((cfg.general_profile_space & 0x03) << 6)
            | (u8::from(cfg.general_tier_flag) << 5)
            | (cfg.general_profile_idc & 0x1F),
    );
    out.extend_from_slice(&cfg.general_profile_compatibility_flags.to_be_bytes());
    let ci = cfg.general_constraint_indicator_flags;
    out.push(((ci >> 40) & 0xFF) as u8);
    out.push(((ci >> 32) & 0xFF) as u8);
    out.push(((ci >> 24) & 0xFF) as u8);
    out.push(((ci >> 16) & 0xFF) as u8);
    out.push(((ci >> 8) & 0xFF) as u8);
    out.push((ci & 0xFF) as u8);
    out.push(cfg.general_level_idc);
    out.extend_from_slice(&(0xF000 | (cfg.min_spatial_segmentation_idc & 0x0FFF)).to_be_bytes());
    out.push(0xFC | (cfg.parallelism_type & 0x03));
    out.push(0xFC | (cfg.chroma_format_idc & 0x03));
    out.push(0xF8 | (cfg.bit_depth_luma_minus8 & 0x07));
    out.push(0xF8 | (cfg.bit_depth_chroma_minus8 & 0x07));
    out.extend_from_slice(&cfg.avg_frame_rate.to_be_bytes());
    out.push(
        ((cfg.constant_frame_rate & 0x03) << 6)
            | ((cfg.num_temporal_layers & 0x07) << 3)
            | (u8::from(cfg.temporal_id_nested) << 2)
            | (cfg.nal_length_size - 1),
    );
    out.push(cfg.arrays.len() as u8);

    for arr in &cfg.arrays {
        if arr.nalus.len() > u16::MAX as usize {
            return Err(Error::invalid_data(
                "hevc/hvcc",
                "too many NAL units in hvcC array",
            ));
        }
        out.push((u8::from(arr.array_completeness) << 7) | (arr.nal_unit_type.code() & 0x3F));
        out.extend_from_slice(&(arr.nalus.len() as u16).to_be_bytes());
        for nalu in &arr.nalus {
            if nalu.len() > u16::MAX as usize {
                return Err(Error::invalid_data(
                    "hevc/hvcc",
                    "HEVC parameter set NAL exceeds 65535 bytes",
                ));
            }
            out.extend_from_slice(&(nalu.len() as u16).to_be_bytes());
            out.extend_from_slice(nalu);
        }
    }
    Ok(out)
}

/// Convert all parameter-set NAL units in an [`HevcDecoderConfig`] into an Annex B prefix.
pub fn hvcc_extradata_to_annexb(cfg: &HevcDecoderConfig) -> Vec<u8> {
    let mut out = Vec::new();
    for arr in &cfg.arrays {
        for nalu in &arr.nalus {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nalu);
        }
    }
    out
}

/// Losslessly convert a length-prefixed HVCC packet to Annex B (4-byte start codes).
pub fn hvcc_to_annexb(packet: &[u8], nal_length_size: u8) -> Result<Vec<u8>> {
    length_prefixed_to_annexb(packet, nal_length_size)
}

/// Losslessly convert an Annex B packet to length-prefixed HVCC format.
pub fn annexb_to_hvcc(packet: &[u8], nal_length_size: u8) -> Result<Vec<u8>> {
    annexb_to_length_prefixed(packet, nal_length_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::RawStreamDemuxer;
    use wedeo_core::{Demuxer, ErrorPolicy, MemorySource, PacketFlags, RecvStatus};

    fn make_hevc_nal(nut: HevcNalUnitType, body: &[u8]) -> Vec<u8> {
        let hdr = HevcNalHeader {
            forbidden_zero_bit: false,
            nal_unit_type: nut,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: 1,
        };
        let mut out = hdr.to_bytes().to_vec();
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn hevc_annexb_framer_splits_multi_slice_aus() {
        let vps = make_hevc_nal(HevcNalUnitType::Vps, &[0x1C, 0x01]);
        let sps = make_hevc_nal(HevcNalUnitType::Sps, &[0x01, 0x02]);
        let pps = make_hevc_nal(HevcNalUnitType::Pps, &[0xC1, 0x72]);
        // AU 0: IDR with 2 slice segments (first_slice=1 -> 0x80, then first_slice=0 -> 0x00)
        let idr_s0 = make_hevc_nal(HevcNalUnitType::IdrWRadl, &[0x80, 0xAA]);
        let idr_s1 = make_hevc_nal(HevcNalUnitType::IdrWRadl, &[0x00, 0xBB]);
        // AU 1: TRAIL_R with first_slice=1
        let trail_s0 = make_hevc_nal(HevcNalUnitType::TrailR, &[0x80, 0xCC]);

        let mut stream = Vec::new();
        for n in [&vps, &sps, &pps, &idr_s0, &idr_s1, &trail_s0] {
            stream.extend_from_slice(&[0, 0, 0, 1]);
            stream.extend_from_slice(n);
        }

        assert!(HevcAnnexBFramer::probe(&stream) >= 90);

        let mut demux = RawStreamDemuxer::open(
            Box::new(MemorySource::new(stream)),
            HevcAnnexBFramer::new(),
            ErrorPolicy::Strict,
        )
        .unwrap();

        let mut pkts = Vec::new();
        while let RecvStatus::Ready(p) = demux.read_packet().unwrap() {
            pkts.push(p);
        }
        assert_eq!(pkts.len(), 2);
        assert!(pkts[0].flags.contains(PacketFlags::KEY));
        assert!(!pkts[1].flags.contains(PacketFlags::KEY));
        assert_eq!(
            iter_annexb_nals(pkts[0].data.as_slice().unwrap()).count(),
            5
        );
        assert_eq!(
            iter_annexb_nals(pkts[1].data.as_slice().unwrap()).count(),
            1
        );
    }

    #[test]
    fn hvcc_roundtrip_and_packet_bsf() {
        let vps = make_hevc_nal(HevcNalUnitType::Vps, &[0x1C, 0x01, 0xFF]);
        let sps = make_hevc_nal(HevcNalUnitType::Sps, &[0x01, 0x02, 0x03]);
        let pps = make_hevc_nal(HevcNalUnitType::Pps, &[0xC1, 0x72]);

        let cfg = HevcDecoderConfig {
            configuration_version: 1,
            general_profile_space: 0,
            general_tier_flag: false,
            general_profile_idc: 2, // Main 10
            general_profile_compatibility_flags: 0x2000_0000,
            general_constraint_indicator_flags: 0x9000_0000_0000,
            general_level_idc: 120,
            min_spatial_segmentation_idc: 0,
            parallelism_type: 0,
            chroma_format_idc: 1,
            bit_depth_luma_minus8: 2,
            bit_depth_chroma_minus8: 2,
            avg_frame_rate: 0,
            constant_frame_rate: 0,
            num_temporal_layers: 1,
            temporal_id_nested: true,
            nal_length_size: 4,
            arrays: vec![
                HvccNalArray {
                    array_completeness: true,
                    nal_unit_type: HevcNalUnitType::Vps,
                    nalus: vec![vps.clone()],
                },
                HvccNalArray {
                    array_completeness: true,
                    nal_unit_type: HevcNalUnitType::Sps,
                    nalus: vec![sps.clone()],
                },
                HvccNalArray {
                    array_completeness: true,
                    nal_unit_type: HevcNalUnitType::Pps,
                    nalus: vec![pps.clone()],
                },
            ],
        };

        let bytes = build_hvcc(&cfg).unwrap();
        let parsed = parse_hvcc(&bytes).unwrap();
        assert_eq!(parsed, cfg);

        let annexb = hvcc_extradata_to_annexb(&parsed);
        let hvcc_pkt = annexb_to_hvcc(&annexb, 4).unwrap();
        assert_eq!(hvcc_to_annexb(&hvcc_pkt, 4).unwrap(), annexb);
    }
}
