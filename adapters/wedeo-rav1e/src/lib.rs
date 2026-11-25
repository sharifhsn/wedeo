#![forbid(unsafe_code)]
//! Segregated `rav1e` AV1 encoder adapter for Wedeo.
//!
//! Wraps the pure-Rust `rav1e` AV1 encoder behind `wedeo_core::Encoder`:
//! - Exposes the `av1C` codec configuration record immediately after `open()` via
//!   `Encoder::extradata()`.
//! - Preserves exact VFR input `Timestamp` (`pts`) and `duration` / `time_base`
//!   across encoder reordering/delay.
//! - Honors `EncodeHint::force_keyframe`.
//! - Maps `ColorDesc` (`primaries`, `transfer`, `matrix`, `range`, `chroma_location`)
//!   into `rav1e` sequence headers.
//! - Clean `flush()` resetting internal encoder state.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use rav1e::prelude::{
    ChromaSamplePosition, ChromaSampling, ColorDescription, ColorPrimaries as Rav1ePrimaries,
    Config, Context, EncoderConfig, EncoderStatus, FrameParameters, FrameType, FrameTypeOverride,
    MatrixCoefficients as Rav1eMatrix, PixelRange, Rational as Rav1eRational,
    TransferCharacteristics as Rav1eTransfer,
};

use wedeo_core::{
    ChromaLocation, CodecContext, CodecId, ColorDesc, ColorPrimaries, ColorRange, EncodeHint,
    Encoder, EncoderDescriptor, EncoderFactory, Error, MatrixCoefficients, MediaType, MediaUnit,
    Packet, PacketFlags, PassMode, PixelLayout, Rational, RecvStatus, Registry, Result,
    SampleDepth, SendStatus, Subsampling, Timestamp, TrackDescriptor, TrackId,
    TransferCharacteristic, VideoFrame,
};

const COMPONENT: &str = "rav1e";

#[derive(Debug, Clone, Copy)]
struct InputTiming {
    pts: Timestamp,
    duration: i64,
    time_base: Rational,
}

/// AV1 encoder wrapper backed by `rav1e`.
pub struct Rav1eEncoder {
    enc_cfg: EncoderConfig,
    threads: usize,
    ctx: Context<u8>,
    extradata: Vec<u8>,
    track_id: TrackId,
    default_time_base: Rational,
    next_input_frameno: u64,
    timing_map: HashMap<u64, InputTiming>,
    pending_packets: VecDeque<Packet>,
    draining: bool,
}

fn map_primaries(p: ColorPrimaries) -> Rav1ePrimaries {
    match p {
        ColorPrimaries::Bt709 => Rav1ePrimaries::BT709,
        ColorPrimaries::Bt470m => Rav1ePrimaries::BT470M,
        ColorPrimaries::Bt470bg => Rav1ePrimaries::BT470BG,
        ColorPrimaries::Smpte170m => Rav1ePrimaries::BT601,
        ColorPrimaries::Smpte240m => Rav1ePrimaries::SMPTE240,
        ColorPrimaries::Film => Rav1ePrimaries::GenericFilm,
        ColorPrimaries::Bt2020 => Rav1ePrimaries::BT2020,
        ColorPrimaries::Smpte428 => Rav1ePrimaries::XYZ,
        ColorPrimaries::Smpte431 => Rav1ePrimaries::SMPTE431,
        ColorPrimaries::Smpte432 => Rav1ePrimaries::SMPTE432,
        ColorPrimaries::Ebu3213 => Rav1ePrimaries::EBU3213,
        ColorPrimaries::Reserved0 | ColorPrimaries::Unspecified | ColorPrimaries::Reserved => {
            Rav1ePrimaries::Unspecified
        }
    }
}

fn map_transfer(t: TransferCharacteristic) -> Rav1eTransfer {
    match t {
        TransferCharacteristic::Bt709 => Rav1eTransfer::BT709,
        TransferCharacteristic::Gamma22 => Rav1eTransfer::BT470M,
        TransferCharacteristic::Gamma28 => Rav1eTransfer::BT470BG,
        TransferCharacteristic::Smpte170m => Rav1eTransfer::BT601,
        TransferCharacteristic::Smpte240m => Rav1eTransfer::SMPTE240,
        TransferCharacteristic::Linear => Rav1eTransfer::Linear,
        TransferCharacteristic::Log => Rav1eTransfer::Log100,
        TransferCharacteristic::LogSqrt => Rav1eTransfer::Log100Sqrt10,
        TransferCharacteristic::Iec61966_2_4 => Rav1eTransfer::IEC61966,
        TransferCharacteristic::Bt1361Ecg => Rav1eTransfer::BT1361,
        TransferCharacteristic::Srgb => Rav1eTransfer::SRGB,
        TransferCharacteristic::Bt2020_10 => Rav1eTransfer::BT2020_10Bit,
        TransferCharacteristic::Bt2020_12 => Rav1eTransfer::BT2020_12Bit,
        TransferCharacteristic::Pq => Rav1eTransfer::SMPTE2084,
        TransferCharacteristic::Smpte428 => Rav1eTransfer::SMPTE428,
        TransferCharacteristic::Hlg => Rav1eTransfer::HLG,
        TransferCharacteristic::Reserved0
        | TransferCharacteristic::Unspecified
        | TransferCharacteristic::Reserved => Rav1eTransfer::Unspecified,
    }
}

fn map_matrix(m: MatrixCoefficients) -> Rav1eMatrix {
    match m {
        MatrixCoefficients::Identity => Rav1eMatrix::Identity,
        MatrixCoefficients::Bt709 => Rav1eMatrix::BT709,
        MatrixCoefficients::Fcc => Rav1eMatrix::FCC,
        MatrixCoefficients::Bt470bg => Rav1eMatrix::BT470BG,
        MatrixCoefficients::Smpte170m => Rav1eMatrix::BT601,
        MatrixCoefficients::Smpte240m => Rav1eMatrix::SMPTE240,
        MatrixCoefficients::Ycgco => Rav1eMatrix::YCgCo,
        MatrixCoefficients::Bt2020Ncl => Rav1eMatrix::BT2020NCL,
        MatrixCoefficients::Bt2020Cl => Rav1eMatrix::BT2020CL,
        MatrixCoefficients::Smpte2085 => Rav1eMatrix::SMPTE2085,
        MatrixCoefficients::ChromaDerivedNcl => Rav1eMatrix::ChromatNCL,
        MatrixCoefficients::ChromaDerivedCl => Rav1eMatrix::ChromatCL,
        MatrixCoefficients::Ictcp => Rav1eMatrix::ICtCp,
        MatrixCoefficients::Unspecified => Rav1eMatrix::Unspecified,
    }
}

fn map_color_desc(color: ColorDesc) -> Option<ColorDescription> {
    if color.primaries == ColorPrimaries::Unspecified
        && color.transfer == TransferCharacteristic::Unspecified
        && color.matrix == MatrixCoefficients::Unspecified
    {
        return None;
    }
    Some(ColorDescription {
        color_primaries: map_primaries(color.primaries),
        transfer_characteristics: map_transfer(color.transfer),
        matrix_coefficients: map_matrix(color.matrix),
    })
}

fn map_chroma_sampling(layout: PixelLayout) -> Result<ChromaSampling> {
    if layout.depth != SampleDepth::U8 {
        return Err(Error::unsupported(
            COMPONENT,
            format!(
                "rav1e 8-bit adapter does not support depth {:?}",
                layout.depth
            ),
        ));
    }
    match layout.subsampling {
        Subsampling::Cs420 => Ok(ChromaSampling::Cs420),
        Subsampling::Cs422 => Ok(ChromaSampling::Cs422),
        Subsampling::Cs444 => Ok(ChromaSampling::Cs444),
        Subsampling::Cs400 => Ok(ChromaSampling::Cs400),
        other => Err(Error::unsupported(
            COMPONENT,
            format!("unsupported chroma subsampling for rav1e: {other:?}"),
        )),
    }
}

impl Rav1eEncoder {
    fn open(track: &TrackDescriptor, codec_ctx: CodecContext, pass: PassMode) -> Result<Self> {
        if !matches!(pass, PassMode::Single) {
            return Err(Error::unsupported(
                COMPONENT,
                "rav1e adapter currently supports PassMode::Single only",
            ));
        }

        let video = track.params.video().ok_or_else(|| {
            Error::invalid_argument(COMPONENT, "rav1e encoder requires video stream params")
        })?;
        if video.width == 0 || video.height == 0 {
            return Err(Error::invalid_argument(
                COMPONENT,
                "video width and height must be non-zero",
            ));
        }

        let chroma_sampling = map_chroma_sampling(video.layout)?;

        let mut enc = EncoderConfig::with_speed_preset(6);
        enc.width = video.width as usize;
        enc.height = video.height as usize;
        enc.chroma_sampling = chroma_sampling;
        enc.bit_depth = 8;

        if track.time_base.is_valid() {
            enc.time_base =
                Rav1eRational::new(track.time_base.num as u64, track.time_base.den as u64);
        } else {
            enc.time_base = Rav1eRational::new(1, 1_000_000);
        }

        if video.bit_rate > 0 {
            enc.bitrate = video.bit_rate.min(i64::from(i32::MAX)) as i32;
        } else {
            enc.quantizer = 100;
        }

        enc.color_description = map_color_desc(video.color);
        enc.pixel_range = match video.color.range {
            ColorRange::Full => PixelRange::Full,
            ColorRange::Limited | ColorRange::Unspecified => PixelRange::Limited,
        };
        enc.chroma_sample_position = match video.color.chroma_location {
            ChromaLocation::Left => ChromaSamplePosition::Vertical,
            ChromaLocation::TopLeft => ChromaSamplePosition::Colocated,
            _ => ChromaSamplePosition::Unknown,
        };

        if video.sar.is_valid() {
            enc.sample_aspect_ratio =
                Rav1eRational::new(video.sar.num as u64, video.sar.den as u64);
        }

        let cfg = Config::new()
            .with_encoder_config(enc.clone())
            .with_threads(codec_ctx.threads);

        let ctx: Context<u8> = cfg
            .new_context()
            .map_err(|e| Error::invalid_argument(COMPONENT, format!("rav1e open failed: {e}")))?;

        let extradata = ctx.container_sequence_header();

        Ok(Self {
            enc_cfg: enc,
            threads: codec_ctx.threads,
            ctx,
            extradata,
            track_id: track.id,
            default_time_base: if track.time_base.is_valid() {
                track.time_base
            } else {
                Rational::new(1, 1_000_000)
            },
            next_input_frameno: 0,
            timing_map: HashMap::new(),
            pending_packets: VecDeque::new(),
            draining: false,
        })
    }

    fn convert_frame(&self, video: &VideoFrame) -> Result<rav1e::prelude::Frame<u8>> {
        let planes = video.planes().ok_or_else(|| {
            Error::unsupported(COMPONENT, "rav1e requires CPU-resident video frames")
        })?;
        let expected_planes = video.layout.plane_count();
        if planes.len() < expected_planes {
            return Err(Error::invalid_argument(
                COMPONENT,
                format!(
                    "expected {expected_planes} planes for {:?}, got {}",
                    video.layout,
                    planes.len()
                ),
            ));
        }

        let mut enc_frame = self.ctx.new_frame();
        for (idx, src_plane) in planes.iter().take(expected_planes).enumerate() {
            let dst_plane = &mut enc_frame.planes[idx];
            let dst_stride = dst_plane.cfg.stride;
            let dst_origin = dst_plane.cfg.yorigin * dst_stride + dst_plane.cfg.xorigin;
            let dst_slice = &mut dst_plane.data[dst_origin..];
            let copy_w = src_plane.width_bytes.min(dst_plane.cfg.width);
            let copy_h = src_plane.height.min(dst_plane.cfg.height);

            for y in 0..copy_h {
                let src_row = src_plane.row(y).ok_or_else(|| {
                    Error::invalid_argument(COMPONENT, format!("plane {idx} row {y} out of bounds"))
                })?;
                let dst_start = y * dst_stride;
                dst_slice[dst_start..dst_start + copy_w].copy_from_slice(&src_row[..copy_w]);
            }
        }

        Ok(enc_frame)
    }

    fn build_packet(&mut self, pkt: rav1e::prelude::Packet<u8>) -> Packet {
        let timing = self
            .timing_map
            .remove(&pkt.input_frameno)
            .unwrap_or(InputTiming {
                pts: Timestamp::new(pkt.input_frameno as i64),
                duration: 1,
                time_base: self.default_time_base,
            });

        let mut out = Packet::new(
            wedeo_core::Buffer::from_slice(&pkt.data),
            self.track_id,
            timing.time_base,
        );
        out.pts = timing.pts;
        out.dts = timing.pts;
        out.duration = timing.duration;
        if pkt.frame_type == FrameType::KEY {
            out.flags |= PacketFlags::KEY;
        }
        out
    }

    fn drain_available(&mut self) -> Result<bool> {
        for _ in 0..1024 {
            match self.ctx.receive_packet() {
                Ok(pkt) => {
                    let out = self.build_packet(pkt);
                    self.pending_packets.push_back(out);
                    return Ok(false);
                }
                Err(EncoderStatus::Encoded) => continue,
                Err(EncoderStatus::NeedMoreData) => return Ok(false),
                Err(EncoderStatus::LimitReached) => return Ok(true),
                Err(e) => {
                    return Err(Error::invalid_data(
                        COMPONENT,
                        format!("rav1e receive_packet error: {e}"),
                    ));
                }
            }
        }
        Ok(false)
    }
}

impl Encoder for Rav1eEncoder {
    fn extradata(&self) -> &[u8] {
        &self.extradata
    }

    fn send_unit(&mut self, unit: Option<&MediaUnit>, hint: EncodeHint) -> Result<SendStatus> {
        match unit {
            Some(unit) => {
                if self.draining {
                    return Ok(SendStatus::Full);
                }
                let video = unit.as_video().ok_or_else(|| {
                    Error::invalid_argument(COMPONENT, "rav1e encoder only accepts video frames")
                })?;

                let enc_frame = self.convert_frame(video)?;
                let mut params = FrameParameters::default();
                if hint.force_keyframe {
                    params.frame_type_override = FrameTypeOverride::Key;
                }

                match self.ctx.send_frame((enc_frame, params)) {
                    Ok(()) => {
                        let frameno = self.next_input_frameno;
                        self.next_input_frameno += 1;
                        let tb = if video.time_base.is_valid() {
                            video.time_base
                        } else {
                            self.default_time_base
                        };
                        self.timing_map.insert(
                            frameno,
                            InputTiming {
                                pts: video.pts,
                                duration: video.duration,
                                time_base: tb,
                            },
                        );
                        Ok(SendStatus::Accepted)
                    }
                    Err(EncoderStatus::EnoughData) => Ok(SendStatus::Full),
                    Err(e) => Err(Error::invalid_data(
                        COMPONENT,
                        format!("rav1e send_frame error: {e}"),
                    )),
                }
            }
            None => {
                self.draining = true;
                self.ctx.flush();
                Ok(SendStatus::Accepted)
            }
        }
    }

    fn receive_packet(&mut self) -> Result<RecvStatus<Packet>> {
        if let Some(pkt) = self.pending_packets.pop_front() {
            return Ok(RecvStatus::Ready(pkt));
        }

        let limit_reached = self.drain_available()?;
        if let Some(pkt) = self.pending_packets.pop_front() {
            return Ok(RecvStatus::Ready(pkt));
        }

        if limit_reached || self.draining {
            Ok(RecvStatus::Eof)
        } else {
            Ok(RecvStatus::NeedMore)
        }
    }

    fn flush(&mut self) {
        self.pending_packets.clear();
        self.timing_map.clear();
        self.next_input_frameno = 0;
        self.draining = false;

        let cfg = Config::new()
            .with_encoder_config(self.enc_cfg.clone())
            .with_threads(self.threads);
        if let Ok(ctx) = cfg.new_context() {
            self.ctx = ctx;
        }
    }
}

/// Factory for opening the segregated `rav1e` AV1 encoder.
pub struct Rav1eAv1EncoderFactory;

impl EncoderFactory for Rav1eAv1EncoderFactory {
    fn descriptor(&self) -> &EncoderDescriptor {
        static DESC: EncoderDescriptor = EncoderDescriptor {
            codec_id: CodecId::Av1,
            name: "av1_rav1e",
            long_name: "AV1 Video [rav1e]",
            media_type: MediaType::Video,
        };
        &DESC
    }

    fn open(
        &self,
        track: &TrackDescriptor,
        ctx: CodecContext,
        pass: PassMode,
    ) -> Result<Box<dyn Encoder>> {
        Ok(Box::new(Rav1eEncoder::open(track, ctx, pass)?))
    }
}

/// Register the `rav1e` AV1 encoder factory into `reg`.
pub fn register_rav1e(reg: &mut Registry) {
    reg.register_encoder(Arc::new(Rav1eAv1EncoderFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::{BorderPadding, BorderedPlane, BufferPool, StreamParams, VideoStreamParams};

    fn make_test_frame(
        pool: &Arc<BufferPool>,
        width: u32,
        height: u32,
        luma_val: u8,
        pts: i64,
        time_base: Rational,
        color: ColorDesc,
    ) -> VideoFrame {
        let mut y = BorderedPlane::new(pool, width as usize, height as usize, BorderPadding::NONE);
        for row in 0..height as usize {
            y.row_mut(row).unwrap().fill(luma_val);
        }
        let cw = width.div_ceil(2) as usize;
        let ch = height.div_ceil(2) as usize;
        let mut u = BorderedPlane::new(pool, cw, ch, BorderPadding::NONE);
        let mut v = BorderedPlane::new(pool, cw, ch, BorderPadding::NONE);
        for row in 0..ch {
            u.row_mut(row).unwrap().fill(128);
            v.row_mut(row).unwrap().fill(128);
        }
        let mut frame = VideoFrame::from_planes(width, height, PixelLayout::YUV420P, vec![y, u, v]);
        frame.pts = Timestamp::new(pts);
        frame.duration = 1000;
        frame.time_base = time_base;
        frame.color = color;
        frame
    }

    #[test]
    fn rav1e_extradata_vfr_pts_keyframe_hint_and_flush() {
        let mut reg = Registry::new();
        register_rav1e(&mut reg);

        let factory = reg.find_encoder(CodecId::Av1).unwrap();
        assert_eq!(factory.descriptor().name, "av1_rav1e");

        let tb = Rational::new(1, 90_000);
        let mut vparams = VideoStreamParams::new(64, 64, PixelLayout::YUV420P);
        vparams.color = ColorDesc::BT709_LIMITED;
        let track =
            TrackDescriptor::new(TrackId(0), CodecId::Av1, tb, StreamParams::Video(vparams));

        let pool = BufferPool::new();
        let ctx = CodecContext::new(Arc::clone(&pool));
        let mut enc = factory.open(&track, ctx, PassMode::Single).unwrap();

        // av1C extradata must be non-empty immediately after open
        let extradata = enc.extradata();
        assert!(extradata.len() >= 4, "expected av1C header");
        assert_eq!(extradata[0] & 0x80, 0x80); // marker bit

        let vfr_pts = [0i64, 3003, 9009, 12012];
        for (i, &pts) in vfr_pts.iter().enumerate() {
            let frame = make_test_frame(
                &pool,
                64,
                64,
                16 + (i as u8 * 20),
                pts,
                tb,
                ColorDesc::BT709_LIMITED,
            );
            let hint = EncodeHint {
                force_keyframe: i == 2,
                target_qp: None,
            };
            assert_eq!(
                enc.send_unit(Some(&MediaUnit::Video(frame)), hint).unwrap(),
                SendStatus::Accepted
            );
        }
        assert_eq!(
            enc.send_unit(None, EncodeHint::default()).unwrap(),
            SendStatus::Accepted
        );

        let mut packets = Vec::new();
        loop {
            match enc.receive_packet().unwrap() {
                RecvStatus::Ready(p) => packets.push(p),
                RecvStatus::NeedMore => continue,
                RecvStatus::Eof => break,
            }
        }
        assert_eq!(packets.len(), 4);
        assert!(packets[0].flags.contains(PacketFlags::KEY));
        assert!(
            packets[2].flags.contains(PacketFlags::KEY),
            "expected forced keyframe at index 2"
        );
        for (pkt, &expected_pts) in packets.iter().zip(vfr_pts.iter()) {
            assert_eq!(pkt.pts, Timestamp::new(expected_pts));
            assert_eq!(pkt.time_base, tb);
            assert_eq!(pkt.duration, 1000);
        }

        // Flush and encode another frame to verify clean reset
        enc.flush();
        let f2 = make_test_frame(&pool, 64, 64, 80, 55555, tb, ColorDesc::BT709_LIMITED);
        assert_eq!(
            enc.send_unit(Some(&MediaUnit::Video(f2)), EncodeHint::default())
                .unwrap(),
            SendStatus::Accepted
        );
        assert_eq!(
            enc.send_unit(None, EncodeHint::default()).unwrap(),
            SendStatus::Accepted
        );
        let mut post_flush = Vec::new();
        loop {
            match enc.receive_packet().unwrap() {
                RecvStatus::Ready(p) => post_flush.push(p),
                RecvStatus::NeedMore => continue,
                RecvStatus::Eof => break,
            }
        }
        assert_eq!(post_flush.len(), 1);
        assert_eq!(post_flush[0].pts, Timestamp::new(55555));
    }
}
