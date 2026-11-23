//! AAC ADTS elementary stream framer and MPEG-4 `AudioSpecificConfig` parser/builder.
//!
//! Faithfully ports FFmpeg `n9.1` `libavcodec/adts_header.c` and `libavcodec/mpeg4audio.c`.
//! Has zero video or GPU dependencies so audio-only consumers can compile with
//! `--no-default-features --features aac`.

use wedeo_core::{
    AudioStreamParams, BitReader, BitWriter, Channel, ChannelLayout, ChannelOrder, CodecId, Error,
    Rational, Result, SampleFormat, StreamParams,
};

use crate::raw::{ElementaryFramer, FrameParams, FrameStep};

/// Minimum ADTS header length without CRC (`protection_absent == 1`).
pub const ADTS_HEADER_SIZE: usize = 7;

/// ADTS header length with 16-bit CRC (`protection_absent == 0`).
pub const ADTS_HEADER_SIZE_WITH_CRC: usize = 9;

/// MPEG-4 Audio sampling frequency table (`ff_mpeg4audio_sample_rates`).
pub const MPEG4_AUDIO_SAMPLE_RATES: [u32; 16] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350, 0, 0, 0,
];

/// MPEG-4 Audio channel count lookup by `channel_configuration` (`ff_mpeg4audio_channels`).
pub const MPEG4_AUDIO_CHANNELS: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 8, 0, 0, 0, 7, 8, 24, 8];

/// Standard MPEG-4 Audio Object Types (`mpeg4audio.h`).
pub mod aot {
    pub const NULL: u8 = 0;
    pub const AAC_MAIN: u8 = 1;
    pub const AAC_LC: u8 = 2;
    pub const AAC_SSR: u8 = 3;
    pub const AAC_LTP: u8 = 4;
    pub const SBR: u8 = 5;
    pub const ER_AAC_LC: u8 = 17;
    pub const ER_BSAC: u8 = 22;
    pub const ER_AAC_LD: u8 = 23;
    pub const PS: u8 = 29;
    pub const ESCAPE: u8 = 31;
    pub const ER_AAC_ELD: u8 = 39;
    pub const USAC: u8 = 42;
}

/// Map an MPEG-4 Audio `channel_configuration` (`1..=7`) to a [`ChannelLayout`].
pub fn channel_layout_from_config(chan_config: u8) -> ChannelLayout {
    use Channel::*;
    match chan_config {
        1 => ChannelLayout::mono(),
        2 => ChannelLayout::stereo(),
        3 => ChannelLayout {
            order: ChannelOrder::Native,
            nb_channels: 3,
            channels: vec![FrontLeft, FrontRight, FrontCenter],
        },
        4 => ChannelLayout {
            order: ChannelOrder::Native,
            nb_channels: 4,
            channels: vec![FrontLeft, FrontRight, FrontCenter, BackCenter],
        },
        5 => ChannelLayout {
            order: ChannelOrder::Native,
            nb_channels: 5,
            channels: vec![FrontLeft, FrontRight, FrontCenter, BackLeft, BackRight],
        },
        6 => ChannelLayout::surround_5_1(),
        7 => ChannelLayout::surround_7_1(),
        idx => {
            let count = MPEG4_AUDIO_CHANNELS.get(idx as usize).copied().unwrap_or(0);
            ChannelLayout::default_for_count(u32::from(count))
        }
    }
}

/// Parsed 7-byte (or 9-byte with CRC) AAC ADTS frame header (`ff_adts_header_parse`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdtsHeader {
    /// `0` = MPEG-4, `1` = MPEG-2.
    pub mpeg_version: u8,
    /// `true` when CRC is absent (7-byte header); `false` when 16-bit CRC follows (9-byte header).
    pub protection_absent: bool,
    /// MPEG-4 Audio Object Type (`profile_bits + 1`: `1`=Main, `2`=LC, `3`=SSR, `4`=LTP).
    pub object_type: u8,
    /// Index into [`MPEG4_AUDIO_SAMPLE_RATES`] (`0..=12`).
    pub sampling_index: u8,
    pub sample_rate: u32,
    pub private_bit: bool,
    pub channel_config: u8,
    pub channels: u8,
    pub original_copy: bool,
    pub home: bool,
    /// Total frame length in bytes including the ADTS header (13-bit field).
    pub frame_length: u16,
    /// 11-bit ADTS buffer fullness (`0x7FF` for VBR).
    pub buffer_fullness: u16,
    /// Number of raw AAC data blocks (`1..=4`).
    pub num_raw_data_blocks: u8,
}

impl AdtsHeader {
    #[inline]
    pub const fn header_size(&self) -> usize {
        if self.protection_absent {
            ADTS_HEADER_SIZE
        } else {
            ADTS_HEADER_SIZE_WITH_CRC
        }
    }

    /// Exact decoded PCM sample count (`1024 * num_raw_data_blocks`).
    #[inline]
    pub const fn samples(&self) -> u32 {
        (self.num_raw_data_blocks as u32) * 1024
    }

    /// Exact packet time base (`1 / sample_rate`).
    #[inline]
    pub const fn time_base(&self) -> Rational {
        Rational::new(1, self.sample_rate as i32)
    }

    pub fn channel_layout(&self) -> ChannelLayout {
        channel_layout_from_config(self.channel_config)
    }

    /// Build the corresponding [`AudioSpecificConfig`] for container extradata (`esds`).
    pub fn to_audio_specific_config(&self) -> AudioSpecificConfig {
        AudioSpecificConfig {
            object_type: self.object_type,
            sampling_index: self.sampling_index,
            sample_rate: self.sample_rate,
            channel_config: self.channel_config,
            channels: self.channels,
            frame_length: 1024,
            sbr: None,
            ps: None,
            ext_object_type: aot::NULL,
            ext_sampling_index: 0,
            ext_sample_rate: 0,
        }
    }

    /// Convert into [`AudioStreamParams`] including synthesized `AudioSpecificConfig` extradata.
    pub fn to_audio_stream_params(&self) -> AudioStreamParams {
        let mut params = AudioStreamParams::new(
            self.sample_rate,
            SampleFormat::F32Planar,
            self.channel_layout(),
        );
        params.frame_size = 1024;
        if let Ok(asc) = build_audio_specific_config(&self.to_audio_specific_config()) {
            params.extradata = asc;
        }
        params
    }
}

/// Parse an AAC ADTS header from `buf` (requires at least 7 bytes).
pub fn parse_adts_header(buf: &[u8]) -> Result<AdtsHeader> {
    if buf.len() < ADTS_HEADER_SIZE {
        return Err(Error::invalid_data(
            "aac/adts",
            "ADTS header shorter than 7 bytes",
        ));
    }
    let mut br = BitReader::new(&buf[..ADTS_HEADER_SIZE]);
    if br.read_bits(12)? != 0xFFF {
        return Err(Error::invalid_data(
            "aac/adts",
            "invalid ADTS 12-bit syncword",
        ));
    }
    let mpeg_version = br.read_bit()? as u8;
    let layer = br.read_bits(2)?;
    if layer != 0 {
        return Err(Error::invalid_data("aac/adts", "ADTS layer must be 0"));
    }
    let protection_absent = br.read_bool()?;
    let profile = br.read_bits(2)? as u8;
    let sampling_index = br.read_bits(4)? as u8;
    let sample_rate = MPEG4_AUDIO_SAMPLE_RATES[sampling_index as usize];
    if sample_rate == 0 {
        return Err(Error::invalid_data(
            "aac/adts",
            "invalid ADTS sampling_frequency_index",
        ));
    }
    let private_bit = br.read_bool()?;
    let channel_config = br.read_bits(3)? as u8;
    let original_copy = br.read_bool()?;
    let home = br.read_bool()?;

    let _copyright_id_bit = br.read_bit()?;
    let _copyright_id_start = br.read_bit()?;
    let frame_length = br.read_bits(13)? as u16;
    let min_len = if protection_absent {
        ADTS_HEADER_SIZE
    } else {
        ADTS_HEADER_SIZE_WITH_CRC
    };
    if (frame_length as usize) < min_len {
        return Err(Error::invalid_data(
            "aac/adts",
            "ADTS aac_frame_length smaller than header size",
        ));
    }
    let buffer_fullness = br.read_bits(11)? as u16;
    let rdb = br.read_bits(2)? as u8;

    let channels = MPEG4_AUDIO_CHANNELS[channel_config as usize];

    Ok(AdtsHeader {
        mpeg_version,
        protection_absent,
        object_type: profile + 1,
        sampling_index,
        sample_rate,
        private_bit,
        channel_config,
        channels,
        original_copy,
        home,
        frame_length,
        buffer_fullness,
        num_raw_data_blocks: rdb + 1,
    })
}

/// Serialize a 7-byte ADTS header (CRC-absent if `protection_absent == true`; caller appends 2 CRC bytes if `false`).
pub fn build_adts_header(hdr: &AdtsHeader) -> Result<[u8; ADTS_HEADER_SIZE]> {
    if !(1..=4).contains(&hdr.object_type) {
        return Err(Error::invalid_argument(
            "aac/adts",
            "ADTS object_type must be in 1..=4",
        ));
    }
    if hdr.sampling_index > 12 {
        return Err(Error::invalid_argument(
            "aac/adts",
            "ADTS sampling_index must be in 0..=12",
        ));
    }
    if hdr.channel_config > 7 {
        return Err(Error::invalid_argument(
            "aac/adts",
            "ADTS channel_config must be in 0..=7",
        ));
    }
    if hdr.frame_length >= (1 << 13) || (hdr.frame_length as usize) < hdr.header_size() {
        return Err(Error::invalid_argument(
            "aac/adts",
            "invalid ADTS frame_length",
        ));
    }
    if !(1..=4).contains(&hdr.num_raw_data_blocks) {
        return Err(Error::invalid_argument(
            "aac/adts",
            "ADTS num_raw_data_blocks must be in 1..=4",
        ));
    }

    let mut w = BitWriter::with_capacity(ADTS_HEADER_SIZE);
    w.write_bits(0xFFF, 12);
    w.write_bits(u32::from(hdr.mpeg_version & 1), 1);
    w.write_bits(0, 2); // layer = 0
    w.write_bit(hdr.protection_absent);
    w.write_bits(u32::from(hdr.object_type - 1), 2);
    w.write_bits(u32::from(hdr.sampling_index), 4);
    w.write_bit(hdr.private_bit);
    w.write_bits(u32::from(hdr.channel_config), 3);
    w.write_bit(hdr.original_copy);
    w.write_bit(hdr.home);
    w.write_bit(false); // copyright_id_bit
    w.write_bit(false); // copyright_id_start
    w.write_bits(u32::from(hdr.frame_length), 13);
    w.write_bits(u32::from(hdr.buffer_fullness & 0x7FF), 11);
    w.write_bits(u32::from(hdr.num_raw_data_blocks - 1), 2);
    let bytes = w.finish();
    let mut out = [0u8; ADTS_HEADER_SIZE];
    out.copy_from_slice(&bytes);
    Ok(out)
}

/// Find the earliest candidate 12-bit `0xFFF` ADTS syncword (`FF Fx` with `layer == 0`).
fn find_adts_sync(buf: &[u8]) -> Option<usize> {
    if buf.len() < 2 {
        return None;
    }
    (0..buf.len() - 1).find(|&i| buf[i] == 0xFF && (buf[i + 1] & 0xF6) == 0xF0)
}

/// Raw AAC ADTS framer implementing [`ElementaryFramer`] with exact `1024 / sample_rate` timing.
pub struct AacAdtsFramer {
    last_config: Option<(u8, u32, u8)>,
}

impl AacAdtsFramer {
    pub fn new() -> Self {
        Self { last_config: None }
    }
}

impl Default for AacAdtsFramer {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementaryFramer for AacAdtsFramer {
    fn format_name(&self) -> &'static str {
        "aac"
    }

    fn codec_id(&self) -> CodecId {
        CodecId::Aac
    }

    fn initial_params(&self) -> StreamParams {
        StreamParams::Audio(AudioStreamParams::new(
            0,
            SampleFormat::F32Planar,
            ChannelLayout::unspec(0),
        ))
    }

    fn probe(head: &[u8]) -> u8 {
        let Some(mut pos) = find_adts_sync(head) else {
            return 0;
        };
        let mut frames = 0u32;
        let mut first_sr = 0u32;
        let mut first_ch = 0u8;

        while pos + ADTS_HEADER_SIZE <= head.len() && frames < 8 {
            let Ok(hdr) = parse_adts_header(&head[pos..]) else {
                break;
            };
            let flen = hdr.frame_length as usize;
            if frames == 0 {
                first_sr = hdr.sample_rate;
                first_ch = hdr.channel_config;
            } else if hdr.sample_rate != first_sr || hdr.channel_config != first_ch {
                break;
            }
            frames += 1;
            if pos + flen > head.len() {
                break;
            }
            pos += flen;
        }

        match frames {
            0 => 0,
            1 => 25,
            2 => 60,
            _ => 90,
        }
    }

    fn next_unit(&mut self, buf: &[u8], eof: bool) -> Result<FrameStep> {
        let Some(sync_pos) = find_adts_sync(buf) else {
            // Preserve the final byte in case `0xFF` straddles a chunk boundary.
            if !eof && buf.len() > 1 {
                return Ok(FrameStep::Skip(buf.len() - 1));
            }
            return Ok(FrameStep::NeedMore);
        };
        if sync_pos > 0 {
            return Ok(FrameStep::Skip(sync_pos));
        }
        if buf.len() < ADTS_HEADER_SIZE {
            return Ok(FrameStep::NeedMore);
        }

        let Ok(hdr) = parse_adts_header(buf) else {
            return Ok(FrameStep::Skip(1));
        };
        let frame_len = hdr.frame_length as usize;

        if buf.len() < frame_len {
            return Ok(FrameStep::NeedMore);
        }

        // When not at EOF, verify the next frame also starts with a valid ADTS syncword
        // to reject false syncs inside corrupt data.
        if !eof {
            if buf.len() < frame_len + 2 {
                return Ok(FrameStep::NeedMore);
            }
            if buf[frame_len] != 0xFF || (buf[frame_len + 1] & 0xF6) != 0xF0 {
                return Ok(FrameStep::Skip(1));
            }
        }

        let key = (hdr.object_type, hdr.sample_rate, hdr.channel_config);
        let stream_params = if self.last_config != Some(key) {
            self.last_config = Some(key);
            Some(StreamParams::Audio(hdr.to_audio_stream_params()))
        } else {
            None
        };

        Ok(FrameStep::Emit {
            range: 0..frame_len,
            consume: frame_len,
            is_key: true,
            params: Some(FrameParams {
                duration: Some(i64::from(hdr.samples())),
                time_base: Some(hdr.time_base()),
                stream_params,
            }),
        })
    }

    fn reset(&mut self) {
        self.last_config = None;
    }
}

/// Parsed ISO/IEC 14496-3 `AudioSpecificConfig` (`ff_mpeg4audio_get_config_gb`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    pub object_type: u8,
    pub sampling_index: u8,
    pub sample_rate: u32,
    pub channel_config: u8,
    pub channels: u8,
    /// Core frame length (`1024`, `960`, `512`, `480`, or `0` if unspecified).
    pub frame_length: u32,
    /// Spectral Band Replication (`Some(true)` = explicit/present, `Some(false)` = absent, `None` = implicit/unknown).
    pub sbr: Option<bool>,
    /// Parametric Stereo (`Some(true)` = present, `Some(false)` = absent, `None` = implicit/unknown).
    pub ps: Option<bool>,
    pub ext_object_type: u8,
    pub ext_sampling_index: u8,
    pub ext_sample_rate: u32,
}

impl AudioSpecificConfig {
    /// Effective output sample rate (accounts for explicit SBR extension rate).
    pub const fn effective_sample_rate(&self) -> u32 {
        if self.ext_sample_rate > 0 {
            self.ext_sample_rate
        } else {
            self.sample_rate
        }
    }

    pub fn channel_layout(&self) -> ChannelLayout {
        channel_layout_from_config(self.channel_config)
    }
}

fn read_object_type(br: &mut BitReader<'_>) -> Result<u8> {
    let mut ot = br.read_bits(5)? as u8;
    if ot == aot::ESCAPE {
        ot = 32 + br.read_bits(6)? as u8;
    }
    Ok(ot)
}

fn read_sample_rate(br: &mut BitReader<'_>) -> Result<(u8, u32)> {
    let idx = br.read_bits(4)? as u8;
    let rate = if idx == 0x0F {
        br.read_bits(24)?
    } else {
        MPEG4_AUDIO_SAMPLE_RATES[idx as usize]
    };
    Ok((idx, rate))
}

/// Parse an ISO/IEC 14496-3 `AudioSpecificConfig` (matching FFmpeg `avpriv_mpeg4audio_get_config2`).
pub fn parse_audio_specific_config(data: &[u8]) -> Result<AudioSpecificConfig> {
    if data.is_empty() {
        return Err(Error::invalid_data("aac/asc", "empty AudioSpecificConfig"));
    }
    let mut br = BitReader::new(data);
    let mut object_type = read_object_type(&mut br)?;
    let (sampling_index, sample_rate) = read_sample_rate(&mut br)?;
    let channel_config = br.read_bits(4)? as u8;
    let Some(&channels) = MPEG4_AUDIO_CHANNELS.get(channel_config as usize) else {
        return Err(Error::invalid_data(
            "aac/asc",
            "invalid channel_configuration in AudioSpecificConfig",
        ));
    };

    let mut sbr: Option<bool> = None;
    let mut ps: Option<bool> = None;
    let mut ext_object_type = aot::NULL;
    let mut ext_sampling_index = 0u8;
    let mut ext_sample_rate = 0u32;

    if object_type == aot::SBR || object_type == aot::PS {
        if object_type == aot::PS {
            ps = Some(true);
        }
        ext_object_type = aot::SBR;
        sbr = Some(true);
        let (e_idx, e_rate) = read_sample_rate(&mut br)?;
        ext_sampling_index = e_idx;
        ext_sample_rate = e_rate;
        object_type = read_object_type(&mut br)?;
        if object_type == aot::ER_BSAC && br.bits_left() >= 4 {
            let _ext_chan_cfg = br.read_bits(4)?;
        }
    }

    let frame_length = match object_type {
        aot::AAC_MAIN | aot::AAC_LC | aot::AAC_SSR | aot::AAC_LTP | aot::ER_AAC_LC => {
            if br.bits_left() >= 1 && br.read_bool()? {
                960
            } else {
                1024
            }
        }
        aot::ER_AAC_LD | aot::ER_AAC_ELD => {
            if br.bits_left() >= 1 && br.read_bool()? {
                480
            } else {
                512
            }
        }
        _ => 0,
    };

    // Scan for optional sync extension (0x2b7 = SBR, 0x548 = PS), matching mpeg4audio.c.
    if ext_object_type != aot::SBR {
        while br.bits_left() > 15 {
            if br.peek_bits(11) == 0x2B7 {
                br.skip_bits(11)?;
                ext_object_type = read_object_type(&mut br)?;
                if ext_object_type == aot::SBR {
                    let sbr_present = br.read_bool()?;
                    sbr = Some(sbr_present);
                    if sbr_present {
                        let (e_idx, e_rate) = read_sample_rate(&mut br)?;
                        ext_sampling_index = e_idx;
                        ext_sample_rate = e_rate;
                        if ext_sample_rate == sample_rate {
                            sbr = None;
                        }
                    }
                }
                if br.bits_left() > 11 && br.peek_bits(11) == 0x548 {
                    br.skip_bits(11)?;
                    ps = Some(br.read_bool()?);
                }
                break;
            } else {
                br.skip_bits(1)?;
            }
        }
    }

    if sbr == Some(false) {
        ps = Some(false);
    }
    if (ps.is_none() && object_type != aot::AAC_LC) || (channels & !0x01) != 0 {
        ps = Some(false);
    }

    Ok(AudioSpecificConfig {
        object_type,
        sampling_index,
        sample_rate,
        channel_config,
        channels,
        frame_length,
        sbr,
        ps,
        ext_object_type,
        ext_sampling_index,
        ext_sample_rate,
    })
}

fn write_object_type(w: &mut BitWriter, ot: u8) {
    if ot >= 32 {
        w.write_bits(u32::from(aot::ESCAPE), 5);
        w.write_bits(u32::from(ot - 32), 6);
    } else {
        w.write_bits(u32::from(ot), 5);
    }
}

fn write_sample_rate(w: &mut BitWriter, rate: u32) -> Result<()> {
    if let Some(idx) = MPEG4_AUDIO_SAMPLE_RATES[..13]
        .iter()
        .position(|&r| r == rate)
    {
        w.write_bits(idx as u32, 4);
    } else if rate > 0 && rate < (1 << 24) {
        w.write_bits(0x0F, 4);
        w.write_bits(rate, 24);
    } else {
        return Err(Error::invalid_argument(
            "aac/asc",
            "invalid sample_rate for AudioSpecificConfig",
        ));
    }
    Ok(())
}

/// Build an ISO/IEC 14496-3 `AudioSpecificConfig` byte sequence.
pub fn build_audio_specific_config(cfg: &AudioSpecificConfig) -> Result<Vec<u8>> {
    if cfg.channel_config >= MPEG4_AUDIO_CHANNELS.len() as u8 {
        return Err(Error::invalid_argument(
            "aac/asc",
            "channel_config out of range",
        ));
    }
    let mut w = BitWriter::with_capacity(5);
    let explicit_sbr = cfg.ext_object_type == aot::SBR && cfg.ext_sample_rate > 0;
    if explicit_sbr {
        let outer = if cfg.ps == Some(true) {
            aot::PS
        } else {
            aot::SBR
        };
        write_object_type(&mut w, outer);
        write_sample_rate(&mut w, cfg.sample_rate)?;
        w.write_bits(u32::from(cfg.channel_config), 4);
        write_sample_rate(&mut w, cfg.ext_sample_rate)?;
        write_object_type(&mut w, cfg.object_type);
    } else {
        write_object_type(&mut w, cfg.object_type);
        write_sample_rate(&mut w, cfg.sample_rate)?;
        w.write_bits(u32::from(cfg.channel_config), 4);
    }

    // GASpecificConfig minimum 3 bits: frameLengthFlag, dependsOnCoreCoder=0, extensionFlag=0
    if matches!(
        cfg.object_type,
        aot::AAC_MAIN | aot::AAC_LC | aot::AAC_SSR | aot::AAC_LTP | aot::ER_AAC_LC
    ) {
        w.write_bit(cfg.frame_length == 960);
        w.write_bit(false);
        w.write_bit(false);
    }

    Ok(w.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::RawStreamDemuxer;
    use wedeo_core::{Demuxer, ErrorPolicy, MemorySource, PacketFlags, RecvStatus, Timestamp};

    fn make_adts_frame(sample_rate_idx: u8, chan_cfg: u8, payload: &[u8]) -> Vec<u8> {
        let hdr = AdtsHeader {
            mpeg_version: 0,
            protection_absent: true,
            object_type: aot::AAC_LC,
            sampling_index: sample_rate_idx,
            sample_rate: MPEG4_AUDIO_SAMPLE_RATES[sample_rate_idx as usize],
            private_bit: false,
            channel_config: chan_cfg,
            channels: MPEG4_AUDIO_CHANNELS[chan_cfg as usize],
            original_copy: false,
            home: false,
            frame_length: (ADTS_HEADER_SIZE + payload.len()) as u16,
            buffer_fullness: 0x7FF,
            num_raw_data_blocks: 1,
        };
        let mut out = build_adts_header(&hdr).unwrap().to_vec();
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn adts_framing_exact_duration_and_resync() {
        // 48 kHz (idx 3), stereo (chan_cfg 2)
        let f0 = make_adts_frame(3, 2, &[0x21, 0x00, 0x49, 0x90, 0x02, 0x19]);
        let f1 = make_adts_frame(3, 2, &[0x21, 0x10, 0x33, 0x44]);
        let f2 = make_adts_frame(3, 2, &[0x21, 0x20, 0x55]);

        let mut stream = vec![0xDE, 0xAD, 0xBE, 0xEF]; // 4 bytes leading junk
        stream.extend_from_slice(&f0);
        stream.extend_from_slice(&f1);
        stream.extend_from_slice(&f2);

        assert!(AacAdtsFramer::probe(&stream) >= 85);

        let mut demux = RawStreamDemuxer::open(
            Box::new(MemorySource::new(stream)),
            AacAdtsFramer::new(),
            ErrorPolicy::Resilient,
        )
        .unwrap();

        let ap = demux.header().tracks[0].params.audio().unwrap();
        assert_eq!(ap.sample_rate, 48_000);
        assert_eq!(ap.channel_layout, ChannelLayout::stereo());
        assert_eq!(ap.extradata, vec![0x11, 0x90]); // standard AAC-LC 48kHz stereo ASC!

        let mut pkts = Vec::new();
        while let RecvStatus::Ready(p) = demux.read_packet().unwrap() {
            pkts.push(p);
        }
        assert_eq!(pkts.len(), 3);
        assert_eq!(pkts[0].data.as_slice().unwrap(), f0.as_slice());
        assert_eq!(pkts[1].data.as_slice().unwrap(), f1.as_slice());
        assert_eq!(pkts[2].data.as_slice().unwrap(), f2.as_slice());
        for p in &pkts {
            assert!(p.flags.contains(PacketFlags::KEY));
            assert_eq!(p.pts, Timestamp::NONE);
            assert_eq!(p.duration, 1024);
            assert_eq!(p.time_base, Rational::new(1, 48_000));
        }
        assert_eq!(demux.health().dropped_bytes, 4);
    }

    #[test]
    fn audio_specific_config_lc_and_he_aac_roundtrip() {
        // Standard AAC-LC 44.1 kHz stereo = [0x12, 0x10]
        let parsed = parse_audio_specific_config(&[0x12, 0x10]).unwrap();
        assert_eq!(parsed.object_type, aot::AAC_LC);
        assert_eq!(parsed.sample_rate, 44_100);
        assert_eq!(parsed.channels, 2);
        assert_eq!(parsed.frame_length, 1024);
        assert_eq!(
            build_audio_specific_config(&parsed).unwrap(),
            vec![0x12, 0x10]
        );

        // Explicit HE-AAC (SBR): 24 kHz core -> 48 kHz output, stereo
        let he_cfg = AudioSpecificConfig {
            object_type: aot::AAC_LC,
            sampling_index: 6,
            sample_rate: 24_000,
            channel_config: 2,
            channels: 2,
            frame_length: 1024,
            sbr: Some(true),
            ps: Some(false),
            ext_object_type: aot::SBR,
            ext_sampling_index: 3,
            ext_sample_rate: 48_000,
        };
        let bytes = build_audio_specific_config(&he_cfg).unwrap();
        let rt = parse_audio_specific_config(&bytes).unwrap();
        assert_eq!(rt.object_type, aot::AAC_LC);
        assert_eq!(rt.sample_rate, 24_000);
        assert_eq!(rt.ext_sample_rate, 48_000);
        assert_eq!(rt.effective_sample_rate(), 48_000);
        assert_eq!(rt.sbr, Some(true));
        assert_eq!(rt.channels, 2);
    }

    #[test]
    fn adts_crc_header_multi_rdb_and_custom_asc_rate() {
        let hdr = AdtsHeader {
            mpeg_version: 1,
            protection_absent: false, // 9-byte header with 2-byte CRC
            object_type: aot::AAC_LC,
            sampling_index: 4, // 44.1 kHz
            sample_rate: 44_100,
            private_bit: false,
            channel_config: 6, // 5.1
            channels: 6,
            original_copy: false,
            home: false,
            frame_length: 13, // 9-byte header + 4 payload bytes
            buffer_fullness: 0x7FF,
            num_raw_data_blocks: 2, // 2048 samples!
        };
        let mut frame = build_adts_header(&hdr).unwrap().to_vec();
        frame.extend_from_slice(&[0xAB, 0xCD, 1, 2, 3, 4]); // 2-byte CRC + 4-byte payload
        let parsed = parse_adts_header(&frame).unwrap();
        assert_eq!(parsed.header_size(), 9);
        assert_eq!(parsed.samples(), 2048);
        assert_eq!(parsed.channel_layout(), ChannelLayout::surround_5_1());

        // Custom 24-bit explicit sample rate (e.g. 44101 Hz) in AudioSpecificConfig
        let custom = AudioSpecificConfig {
            object_type: aot::AAC_LC,
            sampling_index: 0x0F,
            sample_rate: 44_101,
            channel_config: 1,
            channels: 1,
            frame_length: 960,
            sbr: None,
            ps: None,
            ext_object_type: aot::NULL,
            ext_sampling_index: 0,
            ext_sample_rate: 0,
        };
        let bytes = build_audio_specific_config(&custom).unwrap();
        let rt = parse_audio_specific_config(&bytes).unwrap();
        assert_eq!(rt.sample_rate, 44_101);
        assert_eq!(rt.frame_length, 960);
        assert_eq!(rt.channels, 1);
    }
}
