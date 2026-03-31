//! rav1e AV1 encoder adapter for wedeo.
//!
//! Wraps the `rav1e` crate (pure Rust AV1 encoder) behind wedeo's `Encoder`
//! trait, following the same adapter pattern as `wedeo-rav1d` for decoding.
//!
//! ## Supported pixel formats
//!
//! 8-bit YUV420p only. 10-bit (`Context<u16>`) can be added later.

use std::collections::VecDeque;

use rav1e::prelude::*;

use wedeo_codec::descriptor::{CodecCapabilities, CodecProperties};
use wedeo_codec::encoder::{Encoder, EncoderBuilder, EncoderDescriptor};
use wedeo_codec::registry::EncoderFactory;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::frame::Frame;
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::{Packet, PacketFlags};
use wedeo_core::pixel_format::PixelFormat;

/// AV1 encoder wrapper backed by rav1e.
struct Rav1eEncoderWrapper {
    ctx: Context<u8>,
    pending_packets: VecDeque<Packet>,
    /// Raw OBU sequence header bytes (for MP4 av1C extradata).
    sequence_header: Vec<u8>,
}

impl Rav1eEncoderWrapper {
    fn new(builder: &EncoderBuilder) -> Result<Self> {
        if builder.pixel_format != PixelFormat::Yuv420p {
            tracing::warn!(
                "unsupported pixel format for rav1e: {:?}",
                builder.pixel_format
            );
            return Err(Error::PatchwelcomeNotImplemented);
        }

        let speed: u8 = builder
            .options
            .get("speed")
            .or_else(|| builder.options.get("preset"))
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);

        let mut enc = EncoderConfig::with_speed_preset(speed);
        enc.width = builder.width as usize;
        enc.height = builder.height as usize;
        enc.chroma_sampling = ChromaSampling::Cs420;
        enc.bit_depth = 8;

        // Time base
        if builder.time_base.den > 0 && builder.time_base.num > 0 {
            enc.time_base =
                Rational::new(builder.time_base.num as u64, builder.time_base.den as u64);
        }

        // Rate control: bitrate mode if bit_rate is set, otherwise quantizer mode
        if builder.bit_rate > 0 {
            enc.bitrate = builder.bit_rate as i32;
        } else {
            let qp: usize = builder
                .options
                .get("quantizer")
                .or_else(|| builder.options.get("qp"))
                .and_then(|v| v.parse().ok())
                .unwrap_or(100);
            enc.quantizer = qp;
        }

        // Keyframe interval
        if let Some(keyint) = builder
            .options
            .get("keyint")
            .and_then(|v| v.parse::<u64>().ok())
        {
            enc.max_key_frame_interval = keyint;
        }
        if let Some(min_keyint) = builder
            .options
            .get("min-keyint")
            .and_then(|v| v.parse::<u64>().ok())
        {
            enc.min_key_frame_interval = min_keyint;
        }

        // Tiling
        if let Some(tiles) = builder
            .options
            .get("tiles")
            .and_then(|v| v.parse::<usize>().ok())
        {
            enc.tiles = tiles;
        }

        let cfg = Config::new().with_encoder_config(enc).with_threads(0);

        let ctx: Context<u8> = cfg.new_context().map_err(|e| {
            tracing::error!("rav1e context creation failed: {e}");
            Error::InvalidArgument
        })?;

        // Extract the full AV1CodecConfigurationRecord for MP4 av1C extradata.
        // container_sequence_header() returns the complete av1C content:
        // 4-byte header (marker/version/profile/level/chroma) + configOBUs.
        // The muxer writes this directly into the av1C box. The demuxer
        // strips the 4-byte header before passing OBUs to the decoder.
        let sequence_header = ctx.container_sequence_header();

        Ok(Self {
            ctx,
            pending_packets: VecDeque::new(),
            sequence_header,
        })
    }

    /// Convert a wedeo Frame to a rav1e Frame by copying Y/U/V plane data.
    fn convert_frame(&self, frame: &Frame) -> Result<rav1e::prelude::Frame<u8>> {
        let video = frame.video().ok_or(Error::InvalidData)?;
        let mut enc_frame = self.ctx.new_frame();

        // Y plane
        let y_plane = &video.planes[0];
        let y_data = y_plane.buffer.data();
        enc_frame.planes[0].copy_from_raw_u8(
            &y_data[y_plane.offset..],
            y_plane.linesize,
            1, // bytes per pixel
        );

        // U plane
        if video.planes.len() > 1 {
            let u_plane = &video.planes[1];
            let u_data = u_plane.buffer.data();
            enc_frame.planes[1].copy_from_raw_u8(&u_data[u_plane.offset..], u_plane.linesize, 1);
        }

        // V plane
        if video.planes.len() > 2 {
            let v_plane = &video.planes[2];
            let v_data = v_plane.buffer.data();
            enc_frame.planes[2].copy_from_raw_u8(&v_data[v_plane.offset..], v_plane.linesize, 1);
        }

        Ok(enc_frame)
    }
}

impl Encoder for Rav1eEncoderWrapper {
    fn send_frame(&mut self, frame: Option<&Frame>) -> Result<()> {
        match frame {
            Some(f) => {
                let enc_frame = self.convert_frame(f)?;
                match self.ctx.send_frame(enc_frame) {
                    Ok(()) => Ok(()),
                    Err(EncoderStatus::EnoughData) => {
                        // Internal buffer full — caller should drain packets first
                        Err(Error::Again)
                    }
                    Err(e) => {
                        tracing::warn!("rav1e send_frame error: {e}");
                        Err(Error::InvalidData)
                    }
                }
            }
            None => {
                self.ctx.flush();
                Ok(())
            }
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        // Return buffered packets first
        if let Some(pkt) = self.pending_packets.pop_front() {
            return Ok(pkt);
        }

        // rav1e's canonical loop: `Encoded` means "frame processed internally,
        // call receive_packet again to get the actual packet data". We retry
        // up to a bounded limit to avoid an infinite loop if rav1e misbehaves.
        for _ in 0..1000 {
            match self.ctx.receive_packet() {
                Ok(pkt) => {
                    let mut wedeo_pkt = Packet::from_slice(&pkt.data);
                    wedeo_pkt.pts = pkt.input_frameno as i64;
                    wedeo_pkt.dts = wedeo_pkt.pts;
                    wedeo_pkt.duration = 1;
                    if pkt.frame_type == FrameType::KEY {
                        wedeo_pkt.flags |= PacketFlags::KEY;
                    }
                    return Ok(wedeo_pkt);
                }
                Err(EncoderStatus::Encoded) => {
                    // Frame encoded internally but no packet emitted yet — retry.
                    continue;
                }
                Err(EncoderStatus::NeedMoreData) => return Err(Error::Again),
                Err(EncoderStatus::LimitReached) => return Err(Error::Eof),
                Err(e) => {
                    tracing::warn!("rav1e receive_packet error: {e}");
                    return Err(Error::InvalidData);
                }
            }
        }
        // Exhausted retries without a packet or terminal status
        Err(Error::Again)
    }

    fn flush(&mut self) {
        // rav1e cannot reset a context — only clear our pending state.
        self.pending_packets.clear();
    }
}

// --- Factory registration ---

struct Rav1eAv1EncoderFactory;

impl EncoderFactory for Rav1eAv1EncoderFactory {
    fn descriptor(&self) -> &EncoderDescriptor {
        static DESC: EncoderDescriptor = EncoderDescriptor {
            codec_id: CodecId::Av1,
            name: "av1_rav1e",
            long_name: "AV1 Video [rav1e]",
            media_type: MediaType::Video,
            capabilities: CodecCapabilities::DELAY, // rav1e buffers frames internally
            properties: CodecProperties::LOSSY,
            supported_sample_formats: &[],
            supported_pixel_formats: &[PixelFormat::Yuv420p],
            priority: 50, // adapter, not native
        };
        &DESC
    }

    fn create(&self, builder: EncoderBuilder) -> Result<Box<dyn Encoder>> {
        Ok(Box::new(Rav1eEncoderWrapper::new(&builder)?))
    }
}

inventory::submit!(&Rav1eAv1EncoderFactory as &dyn EncoderFactory);

// --- Public API for callers needing the sequence header ---

/// Create an AV1 encoder and return `(encoder, sequence_header_extradata)`.
///
/// The extradata is the full AV1CodecConfigurationRecord (4-byte header +
/// configOBUs) from rav1e's `container_sequence_header()`. Pass it as
/// `CodecParameters.extradata` when muxing to MP4.
pub fn create_av1_encoder(builder: EncoderBuilder) -> Result<(Box<dyn Encoder>, Vec<u8>)> {
    let wrapper = Rav1eEncoderWrapper::new(&builder)?;
    let extradata = wrapper.sequence_header.clone();
    Ok((Box::new(wrapper), extradata))
}
