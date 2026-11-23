//! Pure-Rust, `#![forbid(unsafe_code)]` bitstream syntax parsing, coded-bitstream inspection,
//! lossless BSF/extradata helpers, and generic raw elementary stream framing (`RawStreamDemuxer`).
//!
//! Codec modules are strictly feature-gated (`h264`, `vp9`, `av1`, `aac`, `hevc`) so audio-only
//! builds (`--no-default-features --features aac`) compile zero video syntax code.

#![forbid(unsafe_code)]

pub mod raw;

#[cfg(any(feature = "h264", feature = "hevc"))]
mod startcode;

#[cfg(feature = "h264")]
pub mod h264;

#[cfg(feature = "vp9")]
pub mod vp9;

#[cfg(feature = "av1")]
pub mod av1;

#[cfg(feature = "aac")]
pub mod aac;

#[cfg(feature = "hevc")]
pub mod hevc;

pub use raw::{ElementaryFramer, FrameParams, FrameStep, RawStreamDemuxer};

use wedeo_core::{CodecId, ColorDesc, Error, PixelLayout, Result};

/// One syntax unit (NAL, OBU, VP9 sub-frame, or ADTS frame) extracted for `wedeo inspect`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitstreamUnit {
    /// Byte offset of this unit within the inspected packet.
    pub offset: usize,
    /// Total byte size of this unit (including framing/headers).
    pub size: usize,
    /// Codec-specific unit type code (e.g. H.264/HEVC NAL unit type, AV1 OBU type, VP9 profile).
    pub type_code: u8,
    /// Human-readable unit type name (e.g. `"sps"`, `"idr"`, `"sequence_header"`, `"adts"`).
    pub type_name: &'static str,
    /// Whether this unit carries or indicates a keyframe / random access point.
    pub is_key: bool,
}

/// Structured packet-level syntax inspection summary for `wedeo inspect`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketInspection {
    pub codec_id: CodecId,
    pub total_bytes: usize,
    pub is_key: bool,
    pub profile: Option<u8>,
    pub level: Option<u8>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub pixel_layout: Option<PixelLayout>,
    pub color: Option<ColorDesc>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub units: Vec<BitstreamUnit>,
}

/// Inspect a single compressed packet and return its structured syntax breakdown for `wedeo inspect`.
///
/// Works for Annex B H.264/HEVC, VP9 (including superframes), AV1 OBUs, and AAC ADTS packets
/// according to enabled cargo features.
pub fn inspect_packet(codec_id: CodecId, data: &[u8]) -> Result<PacketInspection> {
    #[allow(unused_mut, unused_variables)]
    let mut out = PacketInspection {
        codec_id,
        total_bytes: data.len(),
        is_key: false,
        profile: None,
        level: None,
        width: None,
        height: None,
        pixel_layout: None,
        color: None,
        sample_rate: None,
        channels: None,
        units: Vec::new(),
    };

    match codec_id {
        #[cfg(feature = "h264")]
        CodecId::H264 => {
            for item in h264::iter_annexb_nals(data) {
                let nal = item?;
                let ntype = nal.header.nal_unit_type;
                let is_key = ntype == h264::NalUnitType::IdrSlice;
                if is_key {
                    out.is_key = true;
                }
                if ntype == h264::NalUnitType::Sps
                    && let Ok(sps) = h264::Sps::parse_nal(nal.raw)
                {
                    out.profile = Some(sps.profile_idc);
                    out.level = Some(sps.level_idc);
                    out.width = Some(sps.display_width());
                    out.height = Some(sps.display_height());
                    out.pixel_layout = Some(sps.pixel_layout());
                    out.color = Some(sps.vui.color);
                }
                out.units.push(BitstreamUnit {
                    offset: nal.start_code_offset,
                    size: nal.start_code_len + nal.raw.len(),
                    type_code: ntype.code(),
                    type_name: ntype.name(),
                    is_key,
                });
            }
            Ok(out)
        }
        #[cfg(feature = "hevc")]
        CodecId::Hevc => {
            for item in hevc::iter_annexb_nals(data) {
                let nal = item?;
                let nut = nal.header.nal_unit_type;
                let is_key = nut.is_irap();
                if is_key {
                    out.is_key = true;
                }
                out.units.push(BitstreamUnit {
                    offset: nal.start_code_offset,
                    size: nal.start_code_len + nal.raw.len(),
                    type_code: nut.code(),
                    type_name: nut.name(),
                    is_key,
                });
            }
            Ok(out)
        }
        #[cfg(feature = "vp9")]
        CodecId::Vp9 => {
            let subframes = vp9::parse_superframe(data)?;
            for range in subframes {
                let slice = &data[range.clone()];
                let hdr = vp9::parse_frame_header(slice)?;
                if hdr.is_keyframe {
                    out.is_key = true;
                }
                out.profile = out.profile.or(Some(hdr.profile));
                if out.width.is_none() {
                    out.width = hdr.width;
                    out.height = hdr.height;
                }
                if out.pixel_layout.is_none() {
                    out.pixel_layout = hdr.pixel_layout;
                }
                if out.color.is_none() && hdr.vp9_color_space.is_some() {
                    out.color = Some(hdr.color);
                }
                let type_name = if hdr.show_existing_frame.is_some() {
                    "show_existing"
                } else if hdr.is_keyframe {
                    "keyframe"
                } else if hdr.intra_only {
                    "intra_only"
                } else {
                    "inter"
                };
                out.units.push(BitstreamUnit {
                    offset: range.start,
                    size: range.len(),
                    type_code: hdr.profile,
                    type_name,
                    is_key: hdr.is_keyframe,
                });
            }
            Ok(out)
        }
        #[cfg(feature = "av1")]
        CodecId::Av1 => {
            let mut reduced_still = false;
            for item in av1::iter_obus(data) {
                let obu = item?;
                let mut unit_key = false;
                match obu.header.obu_type {
                    av1::ObuType::SequenceHeader => {
                        if let Ok(seq) = av1::parse_sequence_header(obu.payload) {
                            reduced_still = seq.reduced_still_picture_header;
                            out.profile = Some(seq.seq_profile);
                            out.level = Some(seq.seq_level_idx_0);
                            out.width = Some(seq.max_frame_width);
                            out.height = Some(seq.max_frame_height);
                            out.pixel_layout = Some(seq.pixel_layout());
                            out.color = Some(seq.color);
                        }
                    }
                    av1::ObuType::Frame | av1::ObuType::FrameHeader => {
                        let is_kf = if reduced_still {
                            true
                        } else if let Some(&b0) = obu.payload.first() {
                            (b0 & 0x80) == 0 && ((b0 >> 5) & 0x03) == 0
                        } else {
                            false
                        };
                        if is_kf {
                            unit_key = true;
                            out.is_key = true;
                        }
                    }
                    _ => {}
                }
                out.units.push(BitstreamUnit {
                    offset: obu.raw_range.start,
                    size: obu.raw_range.len(),
                    type_code: obu.header.obu_type.code(),
                    type_name: obu.header.obu_type.name(),
                    is_key: unit_key,
                });
            }
            Ok(out)
        }
        #[cfg(feature = "aac")]
        CodecId::Aac => {
            let hdr = aac::parse_adts_header(data)?;
            out.is_key = true;
            out.profile = Some(hdr.object_type);
            out.sample_rate = Some(hdr.sample_rate);
            out.channels = Some(hdr.channels);
            out.units.push(BitstreamUnit {
                offset: 0,
                size: (hdr.frame_length as usize).min(data.len()),
                type_code: hdr.object_type,
                type_name: "adts",
                is_key: true,
            });
            Ok(out)
        }
        _ => Err(Error::unsupported(
            "syntax/inspect",
            "packet inspection unsupported or feature disabled for codec",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspect_unsupported_codec_returns_error() {
        assert!(inspect_packet(CodecId::PcmS16le, &[0, 1, 2, 3]).is_err());
    }

    #[cfg(feature = "aac")]
    #[test]
    fn inspect_aac_adts_packet() {
        let hdr = aac::AdtsHeader {
            mpeg_version: 0,
            protection_absent: true,
            object_type: aac::aot::AAC_LC,
            sampling_index: 4, // 44.1 kHz
            sample_rate: 44_100,
            private_bit: false,
            channel_config: 2,
            channels: 2,
            original_copy: false,
            home: false,
            frame_length: 11,
            buffer_fullness: 0x7FF,
            num_raw_data_blocks: 1,
        };
        let mut pkt = aac::build_adts_header(&hdr).unwrap().to_vec();
        pkt.extend_from_slice(&[1, 2, 3, 4]);
        let info = inspect_packet(CodecId::Aac, &pkt).unwrap();
        assert!(info.is_key);
        assert_eq!(info.sample_rate, Some(44_100));
        assert_eq!(info.channels, Some(2));
        assert_eq!(info.units.len(), 1);
        assert_eq!(info.units[0].type_name, "adts");
    }

    #[cfg(feature = "h264")]
    #[test]
    fn inspect_h264_and_unsupported_codec() {
        let mut pkt = Vec::new();
        pkt.extend_from_slice(&[0, 0, 0, 1, 0x09, 0xF0]); // AUD
        pkt.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x80]); // IDR
        let info = inspect_packet(CodecId::H264, &pkt).unwrap();
        assert!(info.is_key);
        assert_eq!(info.units.len(), 2);
        assert_eq!(info.units[0].type_name, "aud");
        assert_eq!(info.units[1].type_name, "idr");
        assert!(inspect_packet(CodecId::PcmS16le, &pkt).is_err());
    }

    #[cfg(feature = "hevc")]
    #[test]
    fn inspect_hevc_packet() {
        let mut pkt = Vec::new();
        pkt.extend_from_slice(&[0, 0, 0, 1, 0x40, 0x01, 0xAA]); // VPS (32)
        pkt.extend_from_slice(&[0, 0, 0, 1, 0x26, 0x01, 0x80]); // IDR_W_RADL (19)
        let info = inspect_packet(CodecId::Hevc, &pkt).unwrap();
        assert!(info.is_key);
        assert_eq!(info.units.len(), 2);
        assert_eq!(info.units[0].type_name, "vps");
        assert_eq!(info.units[1].type_name, "idr_w_radl");
    }
}
