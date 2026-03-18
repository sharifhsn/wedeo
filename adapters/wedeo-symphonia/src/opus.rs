use std::collections::VecDeque;

use wedeo_codec::decoder::{CodecParameters, Decoder, DecoderDescriptor};
use wedeo_codec::descriptor::{CodecCapabilities, CodecDescriptor, CodecProperties};
use wedeo_codec::registry::DecoderFactory;
use wedeo_core::buffer::Buffer;
use wedeo_core::channel_layout::ChannelLayout;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::frame::{Frame, FrameData, FramePlane};
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::Packet;
use wedeo_core::sample_format::SampleFormat;

use crate::decoder::trim_frame;

/// Opus decoder wrapper using the `opus-decoder` crate.
///
/// Opus always decodes at 48 kHz internally. The opus-decoder crate supports
/// output at 8/12/16/24/48 kHz. We decode to f32 at the stream's sample rate.
struct OpusDecoderWrapper {
    inner: opus_decoder::OpusDecoder,
    pending_packets: VecDeque<Packet>,
    drained: bool,
    sample_rate: u32,
    channels: usize,
    channel_layout: ChannelLayout,
    codec_descriptor: CodecDescriptor,
    /// Reusable decode buffer (f32, interleaved, max frame size).
    decode_buf: Vec<f32>,
}

impl OpusDecoderWrapper {
    fn new(params: CodecParameters) -> Result<Self> {
        // Determine channel count. Priority:
        // 1. Explicit channel layout from container
        // 2. Opus ID header in extradata (byte 9 = channel count, per RFC 7845)
        // 3. Default to stereo (most common for Opus)
        let channels = if params.channel_layout.nb_channels > 0 {
            params.channel_layout.nb_channels as usize
        } else if params.extradata.len() >= 10 && &params.extradata[..8] == b"OpusHead" {
            params.extradata[9] as usize
        } else {
            2 // Opus default: stereo
        };
        // opus-decoder supports 1 or 2 channels (mono/stereo).
        // For surround, you'd need OpusMultistreamDecoder.
        if channels > 2 {
            return Err(Error::Other(format!(
                "opus-decoder: only mono/stereo supported, got {channels} channels"
            )));
        }
        let channels = channels.max(1);

        let sample_rate = if params.sample_rate > 0 {
            params.sample_rate
        } else {
            48000 // Opus default
        };

        let inner = opus_decoder::OpusDecoder::new(sample_rate, channels)
            .map_err(|e| Error::Other(format!("opus-decoder init: {e}")))?;

        let max_frame = inner.max_frame_size_per_channel() * channels;

        // Use the resolved channel count for layout
        let channel_layout = if params.channel_layout.nb_channels > 0 {
            params.channel_layout
        } else {
            match channels {
                1 => ChannelLayout::mono(),
                2 => ChannelLayout::stereo(),
                _ => ChannelLayout::unspec(channels as i32),
            }
        };

        Ok(Self {
            inner,
            pending_packets: VecDeque::new(),
            drained: false,
            sample_rate,
            channels,
            channel_layout,
            codec_descriptor: CodecDescriptor {
                id: CodecId::Opus,
                media_type: MediaType::Audio,
                name: "opus",
                long_name: "Opus audio decoder [opus-decoder]",
                properties: CodecProperties::LOSSY,
                profiles: &[],
            },
            decode_buf: vec![0.0f32; max_frame],
        })
    }
}

impl Decoder for OpusDecoderWrapper {
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
            let data = pkt.data.data();

            let samples_per_channel = self
                .inner
                .decode_float(data, &mut self.decode_buf, false)
                .map_err(|e| Error::Other(format!("opus decode: {e}")))?;

            let total_samples = samples_per_channel * self.channels;

            // Convert f32 interleaved to bytes
            let mut pcm_bytes = Vec::with_capacity(total_samples * 4);
            for &sample in &self.decode_buf[..total_samples] {
                pcm_bytes.extend_from_slice(&sample.to_ne_bytes());
            }

            let buffer = Buffer::from_slice(&pcm_bytes);
            let plane = FramePlane {
                buffer,
                offset: 0,
                linesize: pcm_bytes.len(),
            };

            let mut frame = Frame::new_audio(
                samples_per_channel as u32,
                SampleFormat::Flt,
                self.sample_rate,
                self.channel_layout.clone(),
            );
            frame.pts = pkt.pts;
            frame.duration = samples_per_channel as i64;

            if let FrameData::Audio(ref mut audio) = frame.data {
                audio.planes = vec![plane];
            }

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

// --- Factory registration ---

struct OpusDecoderFactory;

impl DecoderFactory for OpusDecoderFactory {
    fn descriptor(&self) -> &DecoderDescriptor {
        static DESC: DecoderDescriptor = DecoderDescriptor {
            codec_id: CodecId::Opus,
            name: "opus",
            long_name: "Opus audio decoder [opus-decoder]",
            media_type: MediaType::Audio,
            capabilities: CodecCapabilities::empty(),
            properties: CodecProperties::LOSSY,
            priority: 100,
        };
        &DESC
    }

    fn create(&self, params: CodecParameters) -> Result<Box<dyn Decoder>> {
        Ok(Box::new(OpusDecoderWrapper::new(params)?))
    }
}

inventory::submit!(&OpusDecoderFactory as &dyn DecoderFactory);
