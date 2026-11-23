use std::collections::HashMap;
use std::sync::Arc;

use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo, Track};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use wedeo_core::{
    AudioStreamParams, Buffer, ByteSource, ChannelLayout, CodecId, ContainerHeader, Demuxer,
    DemuxerDescriptor, DemuxerFactory, Error, ErrorPolicy, HealthEvent, HealthIssue, Metadata,
    OptionSchema, Packet, PacketFlags, Rational, RecvStatus, Registry, Result, Rounding,
    SampleFormat, SeekTarget, SideData, StreamHealth, StreamParams, TIME_BASE_Q, Timestamp,
    TrackDescriptor, TrackId,
};

use crate::channel_layout::channels_to_layout;
use crate::codec_map::{default_bits_for_codec, symphonia_to_wedeo};
use crate::error::from_symphonia;
use crate::io_bridge::WedeoMediaSource;

const COMPONENT: &str = "symphonia/demuxer";
const MAX_CONSECUTIVE_DEMUX_ERRORS: u32 = 64;

static SYMPHONIA_DEMUXER_DESCRIPTOR: DemuxerDescriptor = DemuxerDescriptor {
    name: "symphonia",
    long_name: "Symphonia multi-format audio demuxer",
    extensions: &[
        "wav", "wave", "ogg", "oga", "ogx", "spx", "opus", "flac", "mp4", "m4a", "m4b", "m4v",
        "mov", "mkv", "mka", "webm", "aiff", "aif", "aifc", "caf", "mp3", "mp2", "mp1",
    ],
    mime_types: &[
        "audio/wav",
        "audio/x-wav",
        "audio/ogg",
        "application/ogg",
        "audio/flac",
        "audio/x-flac",
        "audio/mp4",
        "video/mp4",
        "audio/x-matroska",
        "video/x-matroska",
        "video/webm",
        "audio/aiff",
        "audio/x-aiff",
        "audio/x-caf",
        "audio/mpeg",
    ],
};

static SYMPHONIA_DEMUXER_OPTIONS: &[OptionSchema] = &[];

/// Parse the `iTunSMPB` metadata tag used by Apple encoders to signal
/// encoder delay and padding for AAC in MP4/M4A containers.
///
/// Format: `" 00000000 XXXXXXXX YYYYYYYY ZZZZZZZZZZZZZZZZ"`
/// where `XXXXXXXX` is encoder delay (hex) and `YYYYYYYY` is padding (hex).
fn parse_itunsmpb(metadata: &Metadata) -> (u32, u32) {
    for (key, value) in metadata.iter() {
        if key.contains("iTunSMPB") {
            let parts: Vec<&str> = value.split_whitespace().collect();
            if parts.len() >= 3 {
                let delay = u32::from_str_radix(parts[1], 16).unwrap_or(0);
                let padding = u32::from_str_radix(parts[2], 16).unwrap_or(0);
                return (delay, padding);
            }
        }
    }
    (0, 0)
}

/// Try to determine channel count from codec-specific extradata when the
/// container doesn't provide it.
fn guess_channels_from_extradata(codec_id: CodecId, extra: &Option<Box<[u8]>>) -> u32 {
    let Some(data) = extra else { return 0 };
    match codec_id {
        // AAC AudioSpecificConfig: 5 bits audioObjectType + 4 bits frequencyIndex +
        // 4 bits channelConfiguration (bits 9..=12).
        CodecId::Aac if data.len() >= 2 => {
            let channel_config = (data[1] >> 3) & 0x0F;
            match channel_config {
                1 => 1,
                2 => 2,
                3 => 3,
                4 => 4,
                5 => 5,
                6 => 6,
                7 => 8,
                _ => 0,
            }
        }
        // Opus ID header: byte 9 = channel count (RFC 7845)
        CodecId::Opus if data.len() >= 10 && data.starts_with(b"OpusHead") => data[9] as u32,
        _ => 0,
    }
}

/// Guess the output sample format based on codec and bit depth.
fn guess_sample_format(codec_id: CodecId, bits_per_sample: u32) -> SampleFormat {
    match codec_id {
        CodecId::PcmU8 | CodecId::PcmS8 => SampleFormat::U8,
        CodecId::PcmS16le
        | CodecId::PcmS16be
        | CodecId::PcmU16le
        | CodecId::PcmU16be
        | CodecId::PcmAlaw
        | CodecId::PcmMulaw => SampleFormat::S16,
        CodecId::PcmS24le
        | CodecId::PcmS24be
        | CodecId::PcmU24le
        | CodecId::PcmU24be
        | CodecId::PcmS32le
        | CodecId::PcmS32be
        | CodecId::PcmU32le
        | CodecId::PcmU32be => SampleFormat::S32,
        CodecId::PcmF32le | CodecId::PcmF32be => SampleFormat::F32,
        CodecId::PcmF64le | CodecId::PcmF64be => SampleFormat::F64,
        CodecId::Flac | CodecId::Alac | CodecId::WavPack => match bits_per_sample {
            0..=16 => SampleFormat::S16,
            _ => SampleFormat::S32,
        },
        CodecId::Mp1
        | CodecId::Mp2
        | CodecId::Mp3
        | CodecId::Aac
        | CodecId::Vorbis
        | CodecId::Opus => SampleFormat::F32,
        CodecId::AdpcmImaWav | CodecId::AdpcmMs => SampleFormat::S16,
        _ => SampleFormat::S32,
    }
}

/// Convert a symphonia `Track` into a v2 `TrackDescriptor` with sequential `TrackId(index)`.
fn track_to_descriptor(
    track: &Track,
    index: usize,
    aac_delay: u32,
    aac_padding: u32,
) -> TrackDescriptor {
    let codec_id = symphonia_to_wedeo(track.codec_params.codec);
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);

    let channel_layout = if let Some(channels) = track.codec_params.channels {
        let count = channels.count();
        channels_to_layout(channels, count)
    } else {
        let ch_count = guess_channels_from_extradata(codec_id, &track.codec_params.extra_data);
        if ch_count > 0 {
            ChannelLayout::default_for_count(ch_count)
        } else {
            ChannelLayout::unspec(0)
        }
    };

    let raw_bits = track.codec_params.bits_per_sample.unwrap_or(0);
    let coded_bits = track
        .codec_params
        .bits_per_coded_sample
        .or(track.codec_params.bits_per_sample)
        .unwrap_or_else(|| default_bits_for_codec(codec_id));

    let format = guess_sample_format(codec_id, if raw_bits > 0 { raw_bits } else { coded_bits });
    let mut audio_params = AudioStreamParams::new(sample_rate, format, channel_layout);
    audio_params.bits_per_coded_sample = coded_bits;
    if let Some(max_frames) = track.codec_params.max_frames_per_packet {
        audio_params.frame_size = max_frames as u32;
    }
    audio_params.initial_padding = track.codec_params.delay.unwrap_or(aac_delay);
    audio_params.trailing_padding = track.codec_params.padding.unwrap_or(aac_padding);
    if let Some(extra) = &track.codec_params.extra_data {
        audio_params.extradata = extra.to_vec();
    }

    let time_base = if let Some(tb) = track.codec_params.time_base
        && tb.numer > 0
        && tb.denom > 0
    {
        Rational::new(tb.numer as i32, tb.denom as i32)
    } else if sample_rate > 0 {
        Rational::new(1, sample_rate as i32)
    } else {
        TIME_BASE_Q
    };

    let mut desc = TrackDescriptor::new(
        TrackId(index as u32),
        codec_id,
        time_base,
        StreamParams::Audio(audio_params),
    );
    desc.language = track.language.clone();

    if let Some(n_frames) = track.codec_params.n_frames {
        desc.nb_frames = Some(n_frames);
        desc.duration = n_frames as i64;
    }

    desc
}

#[derive(Debug, Clone)]
struct TrackState {
    wedeo_id: TrackId,
    sym_id: u32,
    time_base: Rational,
    initial_trim_start: u32,
    remaining_trim_start: u32,
}

/// Symphonia container demuxer owning its `ByteSource` cleanly via `WedeoMediaSource`.
pub struct SymphoniaDemuxer {
    reader: Box<dyn FormatReader>,
    header: ContainerHeader,
    track_states: Vec<TrackState>,
    sym_to_track_idx: HashMap<u32, usize>,
    is_seekable: bool,
    policy: ErrorPolicy,
    health: StreamHealth,
}

impl SymphoniaDemuxer {
    pub fn open(source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Self> {
        let is_seekable = source.is_seekable();
        let media_source = WedeoMediaSource::new(source);
        let mss = MediaSourceStream::new(Box::new(media_source), Default::default());

        let fmt_opts = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let meta_opts = MetadataOptions::default();
        let hint = Hint::new();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(|e| from_symphonia(COMPONENT, e))?;

        let mut reader = probed.format;

        let metadata = if let Some(rev) = reader.metadata().current() {
            crate::metadata::convert_metadata(rev)
        } else {
            Metadata::new()
        };

        let (aac_start, aac_end) = parse_itunsmpb(&metadata);

        let sym_tracks = reader.tracks();
        let mut tracks = Vec::with_capacity(sym_tracks.len());
        let mut track_states = Vec::with_capacity(sym_tracks.len());
        let mut sym_to_track_idx = HashMap::with_capacity(sym_tracks.len());

        for (idx, sym_track) in sym_tracks.iter().enumerate() {
            let desc = track_to_descriptor(sym_track, idx, aac_start, aac_end);
            let wedeo_id = desc.id;
            let time_base = desc.time_base;
            let sym_id = sym_track.id;

            sym_to_track_idx.insert(sym_id, idx);
            track_states.push(TrackState {
                wedeo_id,
                sym_id,
                time_base,
                initial_trim_start: aac_start,
                remaining_trim_start: aac_start,
            });
            tracks.push(desc);
        }

        let mut header = ContainerHeader::new("symphonia");
        header.tracks = tracks;
        header.metadata = metadata;

        Ok(Self {
            reader,
            header,
            track_states,
            sym_to_track_idx,
            is_seekable,
            policy,
            health: StreamHealth::new(),
        })
    }
}

impl Demuxer for SymphoniaDemuxer {
    fn header(&self) -> &ContainerHeader {
        &self.header
    }

    fn read_packet(&mut self) -> Result<RecvStatus<Packet>> {
        let mut consecutive_errors = 0u32;

        loop {
            match self.reader.next_packet() {
                Ok(sym_pkt) => {
                    let Some(&track_idx) = self.sym_to_track_idx.get(&sym_pkt.track_id()) else {
                        if self.policy == ErrorPolicy::Strict {
                            return Err(Error::corrupt(
                                COMPONENT,
                                format!(
                                    "packet references unknown symphonia track id {}",
                                    sym_pkt.track_id()
                                ),
                            ));
                        }
                        self.health.record(HealthEvent {
                            track: TrackId(0),
                            pts: Timestamp::NONE,
                            byte_offset: None,
                            kind: HealthIssue::CorruptSyntax,
                            detail: "packet with unknown symphonia track id",
                        });
                        consecutive_errors += 1;
                        if consecutive_errors >= MAX_CONSECUTIVE_DEMUX_ERRORS {
                            return Err(Error::corrupt(
                                COMPONENT,
                                "too many consecutive demux errors",
                            ));
                        }
                        continue;
                    };

                    let track_state = &mut self.track_states[track_idx];
                    let data = Buffer::from_slice(sym_pkt.buf());
                    let mut pkt = Packet::new(data, track_state.wedeo_id, track_state.time_base);

                    let raw_ts = sym_pkt.ts();
                    pkt.pts = if raw_ts == u64::MAX {
                        Timestamp::NONE
                    } else {
                        Timestamp::new(raw_ts as i64)
                    };
                    pkt.dts = pkt.pts;
                    pkt.duration = sym_pkt.dur() as i64;
                    pkt.flags = PacketFlags::KEY;

                    let mut trim_start = sym_pkt.trim_start();
                    let trim_end = sym_pkt.trim_end();

                    if track_state.remaining_trim_start > 0 && trim_start == 0 {
                        let packet_dur = pkt.duration.max(0) as u32;
                        let trim = if packet_dur > 0 {
                            track_state.remaining_trim_start.min(packet_dur)
                        } else {
                            track_state.remaining_trim_start
                        };
                        trim_start = trim;
                        track_state.remaining_trim_start -= trim;
                    }

                    if trim_start > 0 || trim_end > 0 {
                        pkt.side_data.push(SideData::SampleTrim {
                            skip_start: trim_start,
                            skip_end: trim_end,
                        });
                    }

                    return Ok(RecvStatus::Ready(pkt));
                }
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(RecvStatus::Eof);
                }
                Err(symphonia::core::errors::Error::DecodeError(msg)) => {
                    if self.policy == ErrorPolicy::Strict {
                        return Err(Error::corrupt(
                            COMPONENT,
                            format!("symphonia demux decode error: {msg}"),
                        ));
                    }
                    self.health.record(HealthEvent {
                        track: TrackId(0),
                        pts: Timestamp::NONE,
                        byte_offset: None,
                        kind: HealthIssue::CorruptSyntax,
                        detail: "symphonia demux decode error",
                    });
                    consecutive_errors += 1;
                    if consecutive_errors >= MAX_CONSECUTIVE_DEMUX_ERRORS {
                        return Err(Error::corrupt(
                            COMPONENT,
                            format!("too many consecutive demux decode errors: {msg}"),
                        ));
                    }
                }
                Err(e) => return Err(from_symphonia(COMPONENT, e)),
            }
        }
    }

    fn seek(&mut self, target: SeekTarget) -> Result<()> {
        if !self.is_seekable {
            return Err(Error::not_seekable(
                COMPONENT,
                "underlying byte source is not seekable",
            ));
        }

        let Some(raw_ts) = target.timestamp.get() else {
            return Err(Error::invalid_argument(
                COMPONENT,
                "cannot seek to Timestamp::NONE",
            ));
        };

        let track_state = if let Some(track_id) = target.track {
            self.track_states
                .iter()
                .find(|t| t.wedeo_id == track_id)
                .ok_or_else(|| {
                    Error::not_found(COMPONENT, format!("track {} not found", track_id.0))
                })?
        } else {
            self.track_states
                .first()
                .ok_or_else(|| Error::not_found(COMPONENT, "container has no tracks"))?
        };

        let track_tb = track_state.time_base;
        let sym_track_id = track_state.sym_id;

        let rnd = if target.backward {
            Rounding::Down
        } else {
            Rounding::NearInf
        };

        let rescaled_ts = if target.time_base.is_valid() && track_tb.is_valid() {
            target
                .timestamp
                .rescale(target.time_base, track_tb, rnd)
                .get()
                .unwrap_or(raw_ts)
        } else {
            raw_ts
        };

        let ts = rescaled_ts.max(0) as u64;
        let seek_to = SeekTo::TimeStamp {
            ts,
            track_id: sym_track_id,
        };

        self.reader
            .seek(SeekMode::Accurate, seek_to)
            .map_err(|e| from_symphonia(COMPONENT, e))?;

        // Reset per-track state after seek.
        for st in &mut self.track_states {
            st.remaining_trim_start = if ts == 0 { st.initial_trim_start } else { 0 };
        }

        Ok(())
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

/// Probe container bytes and optional extension for Symphonia-supported formats.
///
/// Always returns a confidence score `<= 50` (and exactly `50` for RIFF/WAVE magic)
/// so native demuxers (score 100) take precedence.
pub fn probe_symphonia(head: &[u8], ext: Option<&str>) -> u8 {
    // RIFF / RIFX / RF64 / BW64 + WAVE -> exactly 50
    if head.len() >= 12 {
        let magic = &head[0..4];
        let form = &head[8..12];
        if (magic == b"RIFF" || magic == b"RIFX" || magic == b"RF64" || magic == b"BW64")
            && form == b"WAVE"
        {
            return 50;
        }
        // AIFF / AIFC
        if magic == b"FORM" && (form == b"AIFF" || form == b"AIFC") {
            return 50;
        }
    }

    if head.len() >= 4 {
        let magic4 = &head[0..4];
        // OGG, FLAC, MKV/WebM (EBML), CAF
        if magic4 == b"OggS"
            || magic4 == b"fLaC"
            || magic4 == [0x1A, 0x45, 0xDF, 0xA3]
            || magic4 == b"caff"
        {
            return 50;
        }
    }

    // ISOBMFF / MP4 / M4A / MOV
    if head.len() >= 8 {
        let box_type = &head[4..8];
        if matches!(
            box_type,
            b"ftyp" | b"moov" | b"mdat" | b"free" | b"wide" | b"skip" | b"styp"
        ) {
            return 45;
        }
    }

    // ID3v2 header (MP3/audio with ID3 tag)
    if head.len() >= 3 && &head[0..3] == b"ID3" {
        return 45;
    }

    // MPEG audio frame sync word (0xFFE0 mask)
    if head.len() >= 2 && head[0] == 0xFF && (head[1] & 0xE0) == 0xE0 {
        return 30;
    }

    // Weak extension fallback
    if let Some(ext) = ext {
        let clean = ext.trim_start_matches('.');
        if SYMPHONIA_DEMUXER_DESCRIPTOR
            .extensions
            .iter()
            .any(|e| e.eq_ignore_ascii_case(clean))
        {
            return 20;
        }
    }

    0
}

/// Factory for the `"symphonia"` fallback audio demuxer.
pub struct SymphoniaDemuxerFactory;

impl DemuxerFactory for SymphoniaDemuxerFactory {
    fn descriptor(&self) -> &DemuxerDescriptor {
        &SYMPHONIA_DEMUXER_DESCRIPTOR
    }

    fn probe(&self, head: &[u8], ext: Option<&str>) -> u8 {
        probe_symphonia(head, ext)
    }

    fn open(&self, source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Box<dyn Demuxer>> {
        Ok(Box::new(SymphoniaDemuxer::open(source, policy)?))
    }

    fn options(&self) -> &'static [OptionSchema] {
        SYMPHONIA_DEMUXER_OPTIONS
    }
}

pub(crate) fn register_demuxer(reg: &mut Registry) {
    reg.register_demuxer(Arc::new(SymphoniaDemuxerFactory));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_zero_symphonia_track_id_maps_to_sequential_track_id() {
        let mut sym_track = Track::new(42, symphonia::core::codecs::CodecParameters::new());
        sym_track
            .codec_params
            .for_codec(symphonia::core::codecs::CODEC_TYPE_VORBIS)
            .with_sample_rate(48_000);

        let desc = track_to_descriptor(&sym_track, 0, 0, 0);
        // Arbitrary Symphonia track.id (42) must map to sequential TrackId(0).
        assert_eq!(desc.id, TrackId(0));
        assert_eq!(desc.codec_id, CodecId::Vorbis);
        assert_eq!(desc.time_base, Rational::new(1, 48_000));
    }

    #[test]
    fn itunsmpb_and_extradata_channel_guessing() {
        let mut meta = Metadata::new();
        meta.set(
            "iTunSMPB",
            " 00000000 00000840 000001C0 0000000000046000 00000000 00000000",
        );
        let (delay, pad) = parse_itunsmpb(&meta);
        assert_eq!(delay, 0x840);
        assert_eq!(pad, 0x1c0);

        // AAC AudioSpecificConfig for 48kHz stereo: objectType=2 (AAC-LC), freqIdx=3, chanCfg=2 -> 0x11, 0x90
        let aac_asc: Option<Box<[u8]>> = Some(vec![0x11, 0x90].into_boxed_slice());
        assert_eq!(guess_channels_from_extradata(CodecId::Aac, &aac_asc), 2);

        // OpusHead with 1 channel at byte 9
        let mut opus_head = b"OpusHead\x01\x01".to_vec();
        opus_head.extend_from_slice(&[0u8; 9]);
        let opus_extra: Option<Box<[u8]>> = Some(opus_head.into_boxed_slice());
        assert_eq!(guess_channels_from_extradata(CodecId::Opus, &opus_extra), 1);
    }
}
