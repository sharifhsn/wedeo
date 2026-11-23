use std::collections::VecDeque;
use std::sync::Arc;

use wedeo_core::{
    AudioBlock, ChannelLayout, CodecContext, CodecId, Decoder, DecoderDescriptor, DecoderFactory,
    Error, ErrorPolicy, HealthEvent, HealthIssue, MediaType, MediaUnit, OptionSchema, Packet,
    Rational, RecvStatus, Registry, Result, SampleFormat, SendStatus, StreamHealth,
    TrackDescriptor,
};

use crate::decoder::{extract_sample_trim, propagate_non_trim_side_data, trim_audio_block};

const COMPONENT: &str = "symphonia/opus";

static OPUS_DECODER_DESCRIPTOR: DecoderDescriptor = DecoderDescriptor {
    codec_id: CodecId::Opus,
    name: "opus_symphonia",
    long_name: "Opus audio decoder [opus-decoder]",
    media_type: MediaType::Audio,
};

static OPUS_DECODER_OPTIONS: &[OptionSchema] = &[];

/// Opus decoder wrapper using the pure-Rust `opus-decoder` crate.
///
/// Opus always decodes at 48 kHz internally. The `opus-decoder` crate supports
/// output at 8/12/16/24/48 kHz. We decode to `SampleFormat::F32` at the stream's sample rate.
pub struct OpusDecoderWrapper {
    inner: opus_decoder::OpusDecoder,
    pending_packets: VecDeque<Packet>,
    drained: bool,
    sample_rate: u32,
    channels: usize,
    channel_layout: ChannelLayout,
    track_time_base: Rational,
    ctx: CodecContext,
    health: StreamHealth,
    /// Reusable decode buffer (`f32`, interleaved, max frame size).
    decode_buf: Vec<f32>,
}

impl OpusDecoderWrapper {
    pub fn new(track: &TrackDescriptor, ctx: CodecContext) -> Result<Self> {
        let Some(audio) = track.params.audio() else {
            return Err(Error::invalid_argument(
                COMPONENT,
                "track does not carry AudioStreamParams",
            ));
        };

        // Determine channel count. Priority:
        // 1. Explicit channel layout from container
        // 2. Opus ID header in extradata (byte 9 = channel count, per RFC 7845)
        // 3. Default to stereo (most common for Opus)
        let channels = if audio.channel_layout.nb_channels > 0 {
            audio.channel_layout.nb_channels as usize
        } else if audio.extradata.len() >= 10 && &audio.extradata[..8] == b"OpusHead" {
            audio.extradata[9] as usize
        } else {
            2
        };

        if channels > 2 {
            return Err(Error::unsupported(
                COMPONENT,
                format!("opus-decoder: only mono/stereo supported, got {channels} channels"),
            ));
        }
        let channels = channels.max(1);

        let sample_rate = if audio.sample_rate > 0 {
            audio.sample_rate
        } else {
            48_000
        };

        let inner = opus_decoder::OpusDecoder::new(sample_rate, channels)
            .map_err(|e| Error::invalid_argument(COMPONENT, format!("opus-decoder init: {e}")))?;

        let max_frame = inner.max_frame_size_per_channel() * channels;

        let channel_layout = if audio.channel_layout.nb_channels > 0 {
            audio.channel_layout.clone()
        } else {
            ChannelLayout::default_for_count(channels as u32)
        };

        let track_time_base = if track.time_base.is_valid() {
            track.time_base
        } else {
            Rational::new(1, sample_rate as i32)
        };

        Ok(Self {
            inner,
            pending_packets: VecDeque::new(),
            drained: false,
            sample_rate,
            channels,
            channel_layout,
            track_time_base,
            ctx,
            health: StreamHealth::new(),
            decode_buf: vec![0.0f32; max_frame],
        })
    }
}

impl Decoder for OpusDecoderWrapper {
    fn send_packet(&mut self, packet: Option<&Packet>) -> Result<SendStatus> {
        match packet {
            Some(pkt) => {
                self.pending_packets.push_back(pkt.clone());
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
            let data = pkt.data.data();

            let samples_per_channel =
                match self.inner.decode_float(data, &mut self.decode_buf, false) {
                    Ok(n) => n,
                    Err(e) => {
                        if self.ctx.error_policy == ErrorPolicy::Strict {
                            return Err(Error::corrupt(COMPONENT, format!("opus decode: {e}")));
                        }
                        self.health.record(HealthEvent {
                            track: pkt.track_id,
                            pts: pkt.pts,
                            byte_offset: pkt.pos,
                            kind: HealthIssue::CorruptSyntax,
                            detail: "opus audio decode error",
                        });
                        continue;
                    }
                };

            let total_samples = samples_per_channel * self.channels;
            let mut pcm_bytes = Vec::with_capacity(total_samples * 4);
            for &sample in &self.decode_buf[..total_samples] {
                pcm_bytes.extend_from_slice(&sample.to_ne_bytes());
            }

            let plane = self.ctx.pool.acquire_from_slice(&pcm_bytes);
            let mut block = AudioBlock::new(
                vec![plane],
                samples_per_channel as u32,
                SampleFormat::F32,
                self.sample_rate,
                self.channel_layout.clone(),
            );
            block.pts = pkt.pts;
            block.duration = if pkt.duration > 0 {
                pkt.duration
            } else {
                samples_per_channel as i64
            };
            block.time_base = if pkt.time_base.is_valid() {
                pkt.time_base
            } else {
                self.track_time_base
            };

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

pub struct OpusDecoderFactory;

impl DecoderFactory for OpusDecoderFactory {
    fn descriptor(&self) -> &DecoderDescriptor {
        &OPUS_DECODER_DESCRIPTOR
    }

    fn open(&self, track: &TrackDescriptor, ctx: CodecContext) -> Result<Box<dyn Decoder>> {
        Ok(Box::new(OpusDecoderWrapper::new(track, ctx)?))
    }

    fn options(&self) -> &'static [OptionSchema] {
        OPUS_DECODER_OPTIONS
    }
}

pub(crate) fn register_opus_decoder(reg: &mut Registry) {
    reg.register_decoder(Arc::new(OpusDecoderFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::{AudioStreamParams, Buffer, ErrorKind, StreamParams, Timestamp, TrackId};

    #[test]
    fn opus_decoder_open_and_error_policy() {
        let mut audio = AudioStreamParams::new(48_000, SampleFormat::F32, ChannelLayout::unspec(0));
        // Extradata with OpusHead specifying 2 channels
        let mut head = b"OpusHead\x01\x02".to_vec();
        head.extend_from_slice(&[0u8; 9]);
        audio.extradata = head;

        let track = TrackDescriptor::new(
            TrackId(0),
            CodecId::Opus,
            Rational::new(1, 48_000),
            StreamParams::Audio(audio),
        );

        // Resilient mode records health event on empty/invalid opus packet
        let mut dec = OpusDecoderWrapper::new(&track, CodecContext::default()).unwrap();
        assert_eq!(dec.channel_layout, ChannelLayout::stereo());

        let empty_pkt = Packet::new(
            Buffer::from_slice(&[0x03]),
            TrackId(0),
            Rational::new(1, 48_000),
        );
        dec.send_packet(Some(&empty_pkt)).unwrap();
        assert!(matches!(dec.receive().unwrap(), RecvStatus::NeedMore));
        assert_eq!(dec.health().corrupt_packets, 1);

        // Strict mode returns CorruptStream on invalid packet
        let strict_ctx = CodecContext {
            error_policy: ErrorPolicy::Strict,
            ..Default::default()
        };
        let mut strict_dec = OpusDecoderWrapper::new(&track, strict_ctx).unwrap();
        let mut bad_pkt = Packet::new(
            Buffer::from_slice(&[0x03]),
            TrackId(0),
            Rational::new(1, 48_000),
        );
        bad_pkt.pts = Timestamp::new(0);
        strict_dec.send_packet(Some(&bad_pkt)).unwrap();
        assert_eq!(
            strict_dec.receive().unwrap_err().kind,
            ErrorKind::CorruptStream
        );
    }
}
