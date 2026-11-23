use std::collections::VecDeque;
use std::sync::Arc;

use symphonia::core::codecs::DecoderOptions as SymphoniaDecoderOptions;
use symphonia::core::formats::Packet as SymphoniaPacket;

use wedeo_core::{
    AudioBlock, ChannelLayout, CodecContext, CodecId, Decoder, DecoderDescriptor, DecoderFactory,
    Error, ErrorPolicy, HealthEvent, HealthIssue, MediaType, MediaUnit, OptionSchema, Packet,
    Rational, RecvStatus, Registry, Result, SendStatus, SideData, StreamHealth, TrackDescriptor,
    TrackId,
};

use crate::codec_map::{default_bits_for_codec, wedeo_to_symphonia};
use crate::error::from_symphonia;
use crate::sample_format::audio_buffer_to_block;

const COMPONENT: &str = "symphonia/decoder";
static SYMPHONIA_DECODER_OPTIONS: &[OptionSchema] = &[];

/// Check if a codec ID is a PCM or ADPCM variant (these require
/// `max_frames_per_packet` to be set on symphonia's `CodecParameters`).
fn is_pcm_or_adpcm(id: CodecId) -> bool {
    matches!(
        id,
        CodecId::PcmS8
            | CodecId::PcmS16le
            | CodecId::PcmS16be
            | CodecId::PcmS24le
            | CodecId::PcmS24be
            | CodecId::PcmS32le
            | CodecId::PcmS32be
            | CodecId::PcmU8
            | CodecId::PcmU16le
            | CodecId::PcmU16be
            | CodecId::PcmU24le
            | CodecId::PcmU24be
            | CodecId::PcmU32le
            | CodecId::PcmU32be
            | CodecId::PcmF32le
            | CodecId::PcmF32be
            | CodecId::PcmF64le
            | CodecId::PcmF64be
            | CodecId::PcmAlaw
            | CodecId::PcmMulaw
            | CodecId::AdpcmMs
            | CodecId::AdpcmImaWav
    )
}

/// Extract cumulative `(skip_start, skip_end)` sample counts from `Packet::side_data`.
pub(crate) fn extract_sample_trim(pkt: &Packet) -> (usize, usize) {
    let mut skip_start = 0usize;
    let mut skip_end = 0usize;
    for sd in &pkt.side_data {
        if let SideData::SampleTrim {
            skip_start: s,
            skip_end: e,
        } = *sd
        {
            skip_start = skip_start.saturating_add(s as usize);
            skip_end = skip_end.saturating_add(e as usize);
        }
    }
    (skip_start, skip_end)
}

/// Copy non-`SampleTrim` side data entries from a packet onto a decoded `AudioBlock`.
pub(crate) fn propagate_non_trim_side_data(pkt: &Packet, block: &mut AudioBlock) {
    for sd in &pkt.side_data {
        if !matches!(sd, SideData::SampleTrim { .. }) {
            block.side_data.push(sd.clone());
        }
    }
}

/// Trim samples from the start and/or end of an `AudioBlock` for gapless playback.
///
/// Uses zero-copy `Buffer::subslice` views over the block's plane buffers.
pub(crate) fn trim_audio_block(block: &mut AudioBlock, trim_start: usize, trim_end: usize) {
    let nb_channels = block.channel_layout.nb_channels.max(1) as usize;
    let bytes_per_sample = block.format.bytes_per_sample();
    let total_samples = block.nb_samples as usize;

    let new_start = trim_start.min(total_samples);
    let new_end = total_samples.saturating_sub(trim_end).max(new_start);
    let new_nb_samples = new_end - new_start;

    if new_nb_samples == 0 {
        block.nb_samples = 0;
        block.planes.clear();
        block.duration = 0;
        return;
    }

    let sample_stride = if block.format.is_planar() {
        bytes_per_sample
    } else {
        nb_channels * bytes_per_sample
    };
    let start_byte = new_start * sample_stride;
    let end_byte = new_end * sample_stride;

    for plane in &mut block.planes {
        let actual_end = end_byte.min(plane.len());
        let actual_start = start_byte.min(actual_end);
        if let Some(sub) = plane.subslice(actual_start..actual_end) {
            *plane = sub;
        }
    }

    block.nb_samples = new_nb_samples as u32;
    block.duration = new_nb_samples as i64;
}

/// Symphonia decoder wrapper implementing `wedeo_core::Decoder`.
pub struct SymphoniaDecoderWrapper {
    inner: Box<dyn symphonia::core::codecs::Decoder>,
    pending_packets: VecDeque<Packet>,
    drained: bool,
    sample_rate: u32,
    channel_layout: ChannelLayout,
    track_id: TrackId,
    track_time_base: Rational,
    ctx: CodecContext,
    health: StreamHealth,
}

impl SymphoniaDecoderWrapper {
    pub fn new(track: &TrackDescriptor, ctx: CodecContext) -> Result<Self> {
        let Some(audio) = track.params.audio() else {
            return Err(Error::invalid_argument(
                COMPONENT,
                "track does not carry AudioStreamParams",
            ));
        };

        let symphonia_codec = wedeo_to_symphonia(track.codec_id);
        if symphonia_codec == symphonia::core::codecs::CODEC_TYPE_NULL {
            return Err(Error::unsupported(
                COMPONENT,
                format!("unsupported codec {:?}", track.codec_id),
            ));
        }

        let mut sp = symphonia::core::codecs::CodecParameters::new();
        sp.for_codec(symphonia_codec);
        if audio.sample_rate > 0 {
            sp.with_sample_rate(audio.sample_rate);
        }

        if audio.channel_layout.nb_channels > 0 {
            let channels = crate::channel_layout::layout_to_channels(&audio.channel_layout);
            if !channels.is_empty() {
                sp.with_channels(channels);
            }
        }

        if !audio.extradata.is_empty() {
            sp.with_extra_data(audio.extradata.clone().into_boxed_slice());
        }

        let bits = if audio.bits_per_coded_sample > 0 {
            audio.bits_per_coded_sample
        } else {
            default_bits_for_codec(track.codec_id)
        };
        if bits > 0 {
            sp.with_bits_per_coded_sample(bits);
            sp.with_bits_per_sample(bits);
        }

        if audio.frame_size > 0 {
            sp.with_max_frames_per_packet(audio.frame_size as u64);
        } else if is_pcm_or_adpcm(track.codec_id) {
            sp.with_max_frames_per_packet(1152);
        }

        let opts = SymphoniaDecoderOptions::default();
        let registry = symphonia::default::get_codecs();
        let inner = registry
            .make(&sp, &opts)
            .map_err(|e| from_symphonia(COMPONENT, e))?;

        let track_time_base = if track.time_base.is_valid() {
            track.time_base
        } else if audio.sample_rate > 0 {
            Rational::new(1, audio.sample_rate as i32)
        } else {
            Rational::new(1, 48_000)
        };

        Ok(Self {
            inner,
            pending_packets: VecDeque::new(),
            drained: false,
            sample_rate: audio.sample_rate,
            channel_layout: audio.channel_layout.clone(),
            track_id: track.id,
            track_time_base,
            ctx,
            health: StreamHealth::new(),
        })
    }
}

impl Decoder for SymphoniaDecoderWrapper {
    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<SendStatus> {
        match pkt {
            Some(p) => {
                self.pending_packets.push_back(p.clone());
                Ok(SendStatus::Accepted)
            }
            None => {
                self.drained = true;
                Ok(SendStatus::Accepted)
            }
        }
    }

    fn receive(&mut self) -> Result<RecvStatus<MediaUnit>> {
        while let Some(pkt) = self.pending_packets.pop_front() {
            let raw_ts = pkt.pts.get().unwrap_or(0).max(0) as u64;
            let raw_dur = pkt.duration.max(0) as u64;
            let data = pkt.data.data().to_vec();
            let sym_pkt = SymphoniaPacket::new_from_boxed_slice(
                self.track_id.0,
                raw_ts,
                raw_dur,
                data.into_boxed_slice(),
            );

            let decoded = match self.inner.decode(&sym_pkt) {
                Ok(buf) => buf,
                Err(e) => {
                    if self.ctx.error_policy == ErrorPolicy::Strict {
                        return Err(Error::corrupt(
                            COMPONENT,
                            format!("symphonia decode error: {e}"),
                        ));
                    }
                    self.health.record(HealthEvent {
                        track: pkt.track_id,
                        pts: pkt.pts,
                        byte_offset: pkt.pos,
                        kind: HealthIssue::CorruptSyntax,
                        detail: "symphonia audio decode error",
                    });
                    continue;
                }
            };

            let spec = decoded.spec();
            if self.channel_layout.nb_channels == 0 && spec.channels.count() > 0 {
                self.channel_layout =
                    crate::channel_layout::channels_to_layout(spec.channels, spec.channels.count());
            }
            let effective_sr = if self.sample_rate > 0 {
                self.sample_rate
            } else {
                spec.rate
            };
            let effective_tb = if pkt.time_base.is_valid() {
                pkt.time_base
            } else {
                self.track_time_base
            };

            let mut block = audio_buffer_to_block(
                &decoded,
                effective_sr,
                self.channel_layout.clone(),
                pkt.pts,
                pkt.duration,
                effective_tb,
                &self.ctx.pool,
            )?;

            let (trim_start, trim_end) = extract_sample_trim(&pkt);
            if trim_start > 0 || trim_end > 0 {
                trim_audio_block(&mut block, trim_start, trim_end);
            }
            propagate_non_trim_side_data(&pkt, &mut block);

            return Ok(RecvStatus::Ready(MediaUnit::Audio(block)));
        }

        if self.drained {
            Ok(RecvStatus::Eof)
        } else {
            Ok(RecvStatus::NeedMore)
        }
    }

    fn flush(&mut self) {
        self.inner.reset();
        self.pending_packets.clear();
        self.drained = false;
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

const fn audio_desc(
    codec_id: CodecId,
    name: &'static str,
    long_name: &'static str,
) -> DecoderDescriptor {
    DecoderDescriptor {
        codec_id,
        name,
        long_name,
        media_type: MediaType::Audio,
    }
}

static SYMPHONIA_DECODER_DESCRIPTORS: &[DecoderDescriptor] = &[
    // Compressed & lossless codecs
    audio_desc(
        CodecId::Aac,
        "aac_symphonia",
        "AAC (Advanced Audio Coding) [symphonia]",
    ),
    audio_desc(
        CodecId::Alac,
        "alac_symphonia",
        "ALAC (Apple Lossless Audio Codec) [symphonia]",
    ),
    audio_desc(
        CodecId::Flac,
        "flac_symphonia",
        "FLAC (Free Lossless Audio Codec) [symphonia]",
    ),
    audio_desc(CodecId::WavPack, "wavpack_symphonia", "WavPack [symphonia]"),
    audio_desc(
        CodecId::Mp1,
        "mp1_symphonia",
        "MP1 (MPEG audio layer 1) [symphonia]",
    ),
    audio_desc(
        CodecId::Mp2,
        "mp2_symphonia",
        "MP2 (MPEG audio layer 2) [symphonia]",
    ),
    audio_desc(
        CodecId::Mp3,
        "mp3_symphonia",
        "MP3 (MPEG audio layer 3) [symphonia]",
    ),
    audio_desc(CodecId::Vorbis, "vorbis_symphonia", "Vorbis [symphonia]"),
    // ADPCM codecs
    audio_desc(
        CodecId::AdpcmImaWav,
        "adpcm_ima_wav_symphonia",
        "ADPCM IMA WAV [symphonia]",
    ),
    audio_desc(
        CodecId::AdpcmMs,
        "adpcm_ms_symphonia",
        "ADPCM Microsoft [symphonia]",
    ),
    // PCM codecs
    audio_desc(
        CodecId::PcmS8,
        "pcm_s8_symphonia",
        "PCM signed 8-bit [symphonia]",
    ),
    audio_desc(
        CodecId::PcmU8,
        "pcm_u8_symphonia",
        "PCM unsigned 8-bit [symphonia]",
    ),
    audio_desc(
        CodecId::PcmS16le,
        "pcm_s16le_symphonia",
        "PCM signed 16-bit little-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmS16be,
        "pcm_s16be_symphonia",
        "PCM signed 16-bit big-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmU16le,
        "pcm_u16le_symphonia",
        "PCM unsigned 16-bit little-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmU16be,
        "pcm_u16be_symphonia",
        "PCM unsigned 16-bit big-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmS24le,
        "pcm_s24le_symphonia",
        "PCM signed 24-bit little-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmS24be,
        "pcm_s24be_symphonia",
        "PCM signed 24-bit big-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmU24le,
        "pcm_u24le_symphonia",
        "PCM unsigned 24-bit little-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmU24be,
        "pcm_u24be_symphonia",
        "PCM unsigned 24-bit big-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmS32le,
        "pcm_s32le_symphonia",
        "PCM signed 32-bit little-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmS32be,
        "pcm_s32be_symphonia",
        "PCM signed 32-bit big-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmU32le,
        "pcm_u32le_symphonia",
        "PCM unsigned 32-bit little-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmU32be,
        "pcm_u32be_symphonia",
        "PCM unsigned 32-bit big-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmF32le,
        "pcm_f32le_symphonia",
        "PCM 32-bit float little-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmF32be,
        "pcm_f32be_symphonia",
        "PCM 32-bit float big-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmF64le,
        "pcm_f64le_symphonia",
        "PCM 64-bit float little-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmF64be,
        "pcm_f64be_symphonia",
        "PCM 64-bit float big-endian [symphonia]",
    ),
    audio_desc(
        CodecId::PcmAlaw,
        "pcm_alaw_symphonia",
        "PCM A-law [symphonia]",
    ),
    audio_desc(
        CodecId::PcmMulaw,
        "pcm_mulaw_symphonia",
        "PCM mu-law [symphonia]",
    ),
];

struct SymphoniaDecoderFactory {
    descriptor: &'static DecoderDescriptor,
}

impl DecoderFactory for SymphoniaDecoderFactory {
    fn descriptor(&self) -> &DecoderDescriptor {
        self.descriptor
    }

    fn open(&self, track: &TrackDescriptor, ctx: CodecContext) -> Result<Box<dyn Decoder>> {
        Ok(Box::new(SymphoniaDecoderWrapper::new(track, ctx)?))
    }

    fn options(&self) -> &'static [OptionSchema] {
        SYMPHONIA_DECODER_OPTIONS
    }
}

pub(crate) fn register_decoders(reg: &mut Registry) {
    for desc in SYMPHONIA_DECODER_DESCRIPTORS {
        reg.register_decoder(Arc::new(SymphoniaDecoderFactory { descriptor: desc }));
    }
}
