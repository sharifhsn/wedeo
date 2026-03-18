use std::collections::VecDeque;

use symphonia::core::codecs::DecoderOptions as SymphoniaDecoderOptions;
use symphonia::core::formats::Packet as SymphoniaPacket;

use wedeo_codec::decoder::{CodecParameters, Decoder, DecoderDescriptor};
use wedeo_codec::descriptor::{CodecCapabilities, CodecDescriptor, CodecProperties};
use wedeo_codec::registry::DecoderFactory;
use wedeo_core::buffer::Buffer;
use wedeo_core::channel_layout::ChannelLayout;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::frame::{Frame, FrameData};
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::Packet;

use crate::codec_map::wedeo_to_symphonia;
use crate::error::from_symphonia;
use crate::sample_format::audio_buffer_to_frame;

/// Check if a codec ID is a PCM or ADPCM variant (these require
/// `max_frames_per_packet` to be set on symphonia's CodecParameters).
fn is_pcm_or_adpcm(id: CodecId) -> bool {
    matches!(
        id,
        CodecId::PcmS16le
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

/// Trim samples from the start and/or end of an audio frame for gapless playback.
///
/// This removes encoder delay (trim_start) and padding (trim_end) samples from
/// the decoded audio data, adjusting nb_samples, duration, PTS, and plane buffers.
pub(crate) fn trim_frame(mut frame: Frame, trim_start: usize, trim_end: usize) -> Result<Frame> {
    if let FrameData::Audio(ref mut audio) = frame.data {
        let nb_channels = audio.channel_layout.nb_channels.max(1) as usize;
        let bytes_per_sample = audio.format.bytes_per_sample();
        let total_samples = audio.nb_samples as usize;

        let new_start = trim_start.min(total_samples);
        let new_end = total_samples.saturating_sub(trim_end).max(new_start);
        let new_nb_samples = new_end - new_start;

        if new_nb_samples == 0 {
            audio.nb_samples = 0;
            audio.planes.clear();
            frame.duration = 0;
            return Ok(frame);
        }

        let frame_size = nb_channels * bytes_per_sample;
        let start_byte = new_start * frame_size;
        let end_byte = new_end * frame_size;

        for plane in &mut audio.planes {
            let data = plane.buffer.data();
            let actual_end = end_byte.min(data.len());
            let trimmed = data[start_byte..actual_end].to_vec();
            plane.buffer = Buffer::from_slice(&trimmed);
            plane.offset = 0;
            plane.linesize = trimmed.len();
        }

        audio.nb_samples = new_nb_samples as u32;
        frame.duration = new_nb_samples as i64;

        // Note: PTS is NOT adjusted here. When trim comes from symphonia's
        // gapless mode (MP3, OGG), the packet TS is already adjusted.
        // When trim comes from iTunSMPB (AAC), the demuxer adjusts the
        // packet PTS directly.
    }
    Ok(frame)
}

/// Symphonia decoder wrapper — implements wedeo's Decoder trait using
/// a symphonia decoder underneath.
struct SymphoniaDecoderWrapper {
    inner: Box<dyn symphonia::core::codecs::Decoder>,
    pending_packets: VecDeque<Packet>,
    drained: bool,
    sample_rate: u32,
    channel_layout: ChannelLayout,
    codec_descriptor: CodecDescriptor,
    /// Track number to use in symphonia packets (always 0 for single-stream).
    track_id: u32,
}

impl SymphoniaDecoderWrapper {
    fn new(params: CodecParameters, track_id: u32) -> Result<Self> {
        let symphonia_codec = wedeo_to_symphonia(params.codec_id);

        // Build a symphonia CodecParameters
        let mut sp = symphonia::core::codecs::CodecParameters::new();
        sp.for_codec(symphonia_codec)
            .with_sample_rate(params.sample_rate);

        if params.channel_layout.nb_channels > 0 {
            let channels = crate::channel_layout::layout_to_channels(&params.channel_layout);
            sp.with_channels(channels);
        }

        if !params.extradata.is_empty() {
            sp.with_extra_data(params.extradata.into());
        }

        if params.bits_per_coded_sample > 0 {
            sp.with_bits_per_coded_sample(params.bits_per_coded_sample);
        }

        if params.bits_per_raw_sample > 0 {
            sp.with_bits_per_sample(params.bits_per_raw_sample);
        }

        // PCM and ADPCM codecs require max_frames_per_packet to be set.
        // Use 1152 which matches symphonia's internal MAX_FRAMES_PER_PACKET
        // for WAV/AIFF PCM streams.
        if is_pcm_or_adpcm(params.codec_id) {
            sp.with_max_frames_per_packet(1152);
        }

        let opts = SymphoniaDecoderOptions::default();

        let registry = symphonia::default::get_codecs();
        let inner = registry.make(&sp, &opts).map_err(from_symphonia)?;

        Ok(Self {
            inner,
            pending_packets: VecDeque::new(),
            drained: false,
            sample_rate: params.sample_rate,
            channel_layout: params.channel_layout,
            codec_descriptor: CodecDescriptor {
                id: params.codec_id,
                media_type: MediaType::Audio,
                name: params.codec_id.name(),
                long_name: "Symphonia audio decoder",
                properties: CodecProperties::empty(),
                profiles: &[],
            },
            track_id,
        })
    }
}

impl Decoder for SymphoniaDecoderWrapper {
    fn send_packet(&mut self, packet: Option<&Packet>) -> Result<()> {
        match packet {
            Some(pkt) => {
                self.pending_packets.push_back(pkt.clone());
                Ok(())
            }
            None => {
                self.drained = true;
                Ok(())
            }
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(pkt) = self.pending_packets.pop_front() {
            // Convert wedeo Packet to symphonia Packet
            let data = pkt.data.data().to_vec();
            let sym_pkt = SymphoniaPacket::new_from_boxed_slice(
                self.track_id,
                pkt.pts as u64,
                pkt.duration as u64,
                data.into_boxed_slice(),
            );

            let decoded = self.inner.decode(&sym_pkt).map_err(from_symphonia)?;

            // Update channel layout from decoded output if not yet known.
            // Some codecs (AAC) determine channels from the bitstream, not
            // from container metadata.
            let spec = decoded.spec();
            if self.channel_layout.nb_channels == 0 && spec.channels.count() > 0 {
                self.channel_layout =
                    crate::channel_layout::channels_to_layout(spec.channels, spec.channels.count());
            }

            let mut frame = audio_buffer_to_frame(
                &decoded,
                self.sample_rate,
                self.channel_layout.clone(),
                pkt.pts,
            )?;

            // Apply gapless trim if the packet carries trim information
            if pkt.trim_start > 0 || pkt.trim_end > 0 {
                frame = trim_frame(frame, pkt.trim_start as usize, pkt.trim_end as usize)?;
            }

            Ok(frame)
        } else if self.drained {
            Err(Error::Eof)
        } else {
            Err(Error::Again)
        }
    }

    fn flush(&mut self) {
        self.inner.reset();
        self.pending_packets.clear();
        self.drained = false;
    }

    fn descriptor(&self) -> &CodecDescriptor {
        &self.codec_descriptor
    }
}

// --- Factory registration macro ---

macro_rules! register_symphonia_decoder {
    ($factory_name:ident, $codec_id:expr, $name:expr, $long_name:expr, $props:expr) => {
        register_symphonia_decoder_pri!($factory_name, $codec_id, $name, $long_name, $props, 50);
    };
}

macro_rules! register_symphonia_decoder_pri {
    ($factory_name:ident, $codec_id:expr, $name:expr, $long_name:expr, $props:expr, $pri:expr) => {
        struct $factory_name;

        impl DecoderFactory for $factory_name {
            fn descriptor(&self) -> &DecoderDescriptor {
                static DESC: DecoderDescriptor = DecoderDescriptor {
                    codec_id: $codec_id,
                    name: $name,
                    long_name: $long_name,
                    media_type: MediaType::Audio,
                    capabilities: CodecCapabilities::empty(),
                    properties: $props,
                    priority: $pri,
                };
                &DESC
            }

            fn create(&self, params: CodecParameters) -> Result<Box<dyn Decoder>> {
                Ok(Box::new(SymphoniaDecoderWrapper::new(params, 0)?))
            }
        }

        inventory::submit!(&$factory_name as &dyn DecoderFactory);
    };
}

// Lossless codecs
register_symphonia_decoder!(
    SymphoniaFlacDecoderFactory,
    CodecId::Flac,
    "flac_symphonia",
    "FLAC (Free Lossless Audio Codec) [symphonia]",
    CodecProperties::LOSSLESS
);
register_symphonia_decoder!(
    SymphoniaAlacDecoderFactory,
    CodecId::Alac,
    "alac_symphonia",
    "ALAC (Apple Lossless Audio Codec) [symphonia]",
    CodecProperties::LOSSLESS
);
register_symphonia_decoder!(
    SymphoniaWavPackDecoderFactory,
    CodecId::WavPack,
    "wavpack_symphonia",
    "WavPack [symphonia]",
    CodecProperties::LOSSLESS
);

// Lossy codecs
register_symphonia_decoder!(
    SymphoniaMp1DecoderFactory,
    CodecId::Mp1,
    "mp1_symphonia",
    "MP1 (MPEG audio layer 1) [symphonia]",
    CodecProperties::LOSSY
);
register_symphonia_decoder!(
    SymphoniaMp2DecoderFactory,
    CodecId::Mp2,
    "mp2_symphonia",
    "MP2 (MPEG audio layer 2) [symphonia]",
    CodecProperties::LOSSY
);
register_symphonia_decoder!(
    SymphoniaMp3DecoderFactory,
    CodecId::Mp3,
    "mp3_symphonia",
    "MP3 (MPEG audio layer 3) [symphonia]",
    CodecProperties::LOSSY
);
register_symphonia_decoder!(
    SymphoniaAacDecoderFactory,
    CodecId::Aac,
    "aac_symphonia",
    "AAC (Advanced Audio Coding) [symphonia]",
    CodecProperties::LOSSY
);
register_symphonia_decoder!(
    SymphoniaVorbisDecoderFactory,
    CodecId::Vorbis,
    "vorbis_symphonia",
    "Vorbis [symphonia]",
    CodecProperties::LOSSY
);

// ADPCM
register_symphonia_decoder!(
    SymphoniaAdpcmImaWavDecoderFactory,
    CodecId::AdpcmImaWav,
    "adpcm_ima_wav_symphonia",
    "ADPCM IMA WAV [symphonia]",
    CodecProperties::LOSSY
);
register_symphonia_decoder!(
    SymphoniaAdpcmMsDecoderFactory,
    CodecId::AdpcmMs,
    "adpcm_ms_symphonia",
    "ADPCM Microsoft [symphonia]",
    CodecProperties::LOSSY
);

// PCM decoders — replace native implementations with symphonia (priority 100)
register_symphonia_decoder_pri!(
    SymphoniaPcmS16leFactory,
    CodecId::PcmS16le,
    "pcm_s16le",
    "PCM signed 16-bit little-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmS16beFactory,
    CodecId::PcmS16be,
    "pcm_s16be",
    "PCM signed 16-bit big-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmU8Factory,
    CodecId::PcmU8,
    "pcm_u8",
    "PCM unsigned 8-bit [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmU16leFactory,
    CodecId::PcmU16le,
    "pcm_u16le",
    "PCM unsigned 16-bit little-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmU16beFactory,
    CodecId::PcmU16be,
    "pcm_u16be",
    "PCM unsigned 16-bit big-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmS32leFactory,
    CodecId::PcmS32le,
    "pcm_s32le",
    "PCM signed 32-bit little-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmS32beFactory,
    CodecId::PcmS32be,
    "pcm_s32be",
    "PCM signed 32-bit big-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmU32leFactory,
    CodecId::PcmU32le,
    "pcm_u32le",
    "PCM unsigned 32-bit little-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmU32beFactory,
    CodecId::PcmU32be,
    "pcm_u32be",
    "PCM unsigned 32-bit big-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmS24leFactory,
    CodecId::PcmS24le,
    "pcm_s24le",
    "PCM signed 24-bit little-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmS24beFactory,
    CodecId::PcmS24be,
    "pcm_s24be",
    "PCM signed 24-bit big-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmU24leFactory,
    CodecId::PcmU24le,
    "pcm_u24le",
    "PCM unsigned 24-bit little-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmU24beFactory,
    CodecId::PcmU24be,
    "pcm_u24be",
    "PCM unsigned 24-bit big-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmF32leFactory,
    CodecId::PcmF32le,
    "pcm_f32le",
    "PCM 32-bit float little-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmF32beFactory,
    CodecId::PcmF32be,
    "pcm_f32be",
    "PCM 32-bit float big-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmF64leFactory,
    CodecId::PcmF64le,
    "pcm_f64le",
    "PCM 64-bit float little-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
register_symphonia_decoder_pri!(
    SymphoniaPcmF64beFactory,
    CodecId::PcmF64be,
    "pcm_f64be",
    "PCM 64-bit float big-endian [symphonia]",
    CodecProperties::LOSSLESS,
    100
);
