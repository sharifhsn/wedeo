//! Segregated `rav1d` AV1 decoder adapter for Wedeo.
//!
//! Wraps the pure-Rust `rav1d` AV1 decoder behind `wedeo_core::Decoder`:
//! - **Zero-copy output planes** via `wedeo_core::ExternalCpuBuffer` holding a shared,
//!   reference-counted `Dav1dPictureGuard` that calls `dav1d_picture_unref`
//!   once all plane buffers for a frame are dropped (with automatic clean
//!   fallback to pooled copy if a non-positive or narrow stride is encountered).
//! - **Odd-dimension chroma** computed with ceiling division (`w.div_ceil(2)`,
//!   `h.div_ceil(2)`).
//! - **Full CICP [`ColorDesc`]** (`primaries`, `transfer`, `matrix`, `range`,
//!   `chroma_location`) populated from `Dav1dSequenceHeader` on every output
//!   [`VideoFrame`], plus HDR side data (`MasteringDisplay`, `ContentLightLevel`)
//!   and sample aspect ratio (`sar`).
//! - **8-bit, 10-bit, and 12-bit** support across `4:2:0`, `4:2:2`, `4:4:4`
//!   (including GBR identity matrix), and `4:0:0` monochrome.

#![deny(unsafe_code)]

mod ffi_guard;

use std::collections::VecDeque;
use std::ffi::c_int;
use std::sync::Arc;

use rav1d::include::dav1d::headers::{
    DAV1D_CHR_COLOCATED, DAV1D_CHR_VERTICAL, DAV1D_FRAME_TYPE_INTER, DAV1D_FRAME_TYPE_INTRA,
    DAV1D_FRAME_TYPE_KEY, DAV1D_FRAME_TYPE_SWITCH, DAV1D_PIXEL_LAYOUT_I400,
    DAV1D_PIXEL_LAYOUT_I420, DAV1D_PIXEL_LAYOUT_I422, DAV1D_PIXEL_LAYOUT_I444, Dav1dPixelLayout,
    Dav1dSequenceHeader,
};

use wedeo_core::{
    BufferPool, ChromaLocation, CodecContext, CodecId, ColorDesc, ColorPrimaries, ColorRange,
    Decoder, DecoderDescriptor, DecoderFactory, Error, ErrorPolicy, FrameFlags, HealthEvent,
    HealthIssue, MasteringDisplayMeta, MatrixCoefficients, MediaType, MediaUnit, Packet,
    PictureType, PixelLayout, Rational, RecvStatus, Registry, Result, SampleDepth, SendStatus,
    SideData, StreamHealth, Timestamp, TrackDescriptor, TrackId, TransferCharacteristic,
    VideoFrame, reduce,
};

use self::ffi_guard::{Dav1dContextGuard, Dav1dPictureGuard};

const COMPONENT: &str = "rav1d";

/// Internal outcome of feeding raw OBU bytes to `dav1d_send_data`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawSendOutcome {
    Accepted,
    TryAgain,
    InvalidData,
}

/// Map a `rav1d` pixel layout, bit depth (`8`, `10`, `12`), and matrix coefficients
/// to an orthogonal Wedeo [`PixelLayout`].
pub fn map_pixel_layout(
    layout: Dav1dPixelLayout,
    bpc: c_int,
    matrix: MatrixCoefficients,
) -> Result<PixelLayout> {
    let depth = match bpc {
        8 => SampleDepth::U8,
        10 => SampleDepth::U10,
        12 => SampleDepth::U12,
        _ => {
            return Err(Error::unsupported(
                COMPONENT,
                format!("unsupported rav1d bit depth: {bpc}"),
            ));
        }
    };

    match layout {
        DAV1D_PIXEL_LAYOUT_I400 => Ok(PixelLayout {
            depth,
            ..PixelLayout::GRAY8
        }),
        DAV1D_PIXEL_LAYOUT_I420 => Ok(PixelLayout {
            depth,
            ..PixelLayout::YUV420P
        }),
        DAV1D_PIXEL_LAYOUT_I422 => Ok(PixelLayout {
            depth,
            ..PixelLayout::YUV422P
        }),
        DAV1D_PIXEL_LAYOUT_I444 => {
            if matrix == MatrixCoefficients::Identity {
                Ok(PixelLayout {
                    depth,
                    ..PixelLayout::GBRP
                })
            } else {
                Ok(PixelLayout {
                    depth,
                    ..PixelLayout::YUV444P
                })
            }
        }
        _ => Err(Error::unsupported(
            COMPONENT,
            format!("unsupported rav1d pixel layout: {layout}"),
        )),
    }
}

/// Chroma dimensions in pixels for a given `Dav1dPixelLayout`, using ceiling division
/// (`div_ceil(2)`) so odd frame dimensions never truncate the last chroma sample.
#[inline]
pub const fn chroma_dims(layout: Dav1dPixelLayout, w: usize, h: usize) -> (usize, usize) {
    match layout {
        DAV1D_PIXEL_LAYOUT_I420 => (w.div_ceil(2), h.div_ceil(2)),
        DAV1D_PIXEL_LAYOUT_I422 => (w.div_ceil(2), h),
        DAV1D_PIXEL_LAYOUT_I444 => (w, h),
        _ => (0, 0), // I400 monochrome
    }
}

/// Extract CICP [`ColorDesc`] from a [`Dav1dSequenceHeader`], falling back to
/// `fallback` (`TrackDescriptor` stream colorimetry) when the bitstream omits
/// explicit `color_description_present`.
pub fn extract_color_desc(seq_hdr: &Dav1dSequenceHeader, fallback: ColorDesc) -> ColorDesc {
    let (primaries, transfer, matrix) = if seq_hdr.color_description_present != 0 {
        let pri = u8::try_from(seq_hdr.pri)
            .map(ColorPrimaries::from_cicp)
            .unwrap_or(ColorPrimaries::Unspecified);
        let trc = u8::try_from(seq_hdr.trc)
            .map(TransferCharacteristic::from_cicp)
            .unwrap_or(TransferCharacteristic::Unspecified);
        let mtrx = u8::try_from(seq_hdr.mtrx)
            .map(MatrixCoefficients::from_cicp)
            .unwrap_or(MatrixCoefficients::Unspecified);
        (
            if pri != ColorPrimaries::Unspecified {
                pri
            } else {
                fallback.primaries
            },
            if trc != TransferCharacteristic::Unspecified {
                trc
            } else {
                fallback.transfer
            },
            if mtrx != MatrixCoefficients::Unspecified {
                mtrx
            } else {
                fallback.matrix
            },
        )
    } else {
        (fallback.primaries, fallback.transfer, fallback.matrix)
    };

    let range = if seq_hdr.color_range != 0 {
        ColorRange::Full
    } else {
        ColorRange::Limited
    };

    let chroma_location = match seq_hdr.chr {
        DAV1D_CHR_VERTICAL => ChromaLocation::Left,
        DAV1D_CHR_COLOCATED => ChromaLocation::TopLeft,
        _ => fallback.chroma_location,
    };

    ColorDesc {
        primaries,
        transfer,
        matrix,
        range,
        chroma_location,
    }
}

/// Convert a decoded `Dav1dPictureGuard` into a Wedeo [`VideoFrame`].
fn picture_to_video_frame(
    guard: Arc<Dav1dPictureGuard>,
    pool: &Arc<BufferPool>,
    fallback_color: ColorDesc,
    fallback_sar: Rational,
    time_base: Rational,
) -> Result<VideoFrame> {
    let pic = guard.picture();
    if pic.p.w <= 0 || pic.p.h <= 0 {
        return Err(Error::invalid_data(
            COMPONENT,
            "non-positive picture dimensions from rav1d",
        ));
    }
    let w = pic.p.w as usize;
    let h = pic.p.h as usize;
    let bpc = pic.p.bpc;
    let layout = pic.p.layout;

    // Extract sequence header CICP colorimetry before choosing pixel layout
    // (since 4:4:4 + Identity matrix maps to GBRP).
    let color = if let Some(seq_hdr) = guard.seq_hdr() {
        extract_color_desc(seq_hdr, fallback_color)
    } else {
        fallback_color
    };

    let pixel_layout = map_pixel_layout(layout, bpc, color.matrix)?;
    let bytes_per_sample = pixel_layout.depth.bytes_per_sample();

    // Compute guaranteed allocation sizes from rav1d's default picture allocator
    // (`aligned_h = (h + 127) & !127`, chroma rows `aligned_h >> ss_ver`).
    let aligned_h = (h + 127) & !127;
    let ss_ver = usize::from(layout == DAV1D_PIXEL_LAYOUT_I420);
    let y_stride = pic.stride[0];
    let uv_stride = pic.stride[1];
    let y_alloc_bytes = y_stride
        .unsigned_abs()
        .checked_mul(aligned_h)
        .ok_or_else(|| Error::invalid_data(COMPONENT, "rav1d luma plane byte size overflow"))?;
    let uv_alloc_bytes = uv_stride
        .unsigned_abs()
        .checked_mul(aligned_h >> ss_ver)
        .ok_or_else(|| Error::invalid_data(COMPONENT, "rav1d chroma plane byte size overflow"))?;

    let y_width_bytes = w
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| Error::invalid_data(COMPONENT, "rav1d luma row byte width overflow"))?;
    let y_plane = Dav1dPictureGuard::build_plane(&guard, pool, 0, y_width_bytes, h, y_alloc_bytes)
        .ok_or_else(|| Error::invalid_data(COMPONENT, "missing luma plane pointer"))?;

    let (cw, ch) = chroma_dims(layout, w, h);
    let planes = if cw > 0 && ch > 0 {
        let c_width_bytes = cw.checked_mul(bytes_per_sample).ok_or_else(|| {
            Error::invalid_data(COMPONENT, "rav1d chroma row byte width overflow")
        })?;

        let u_plane =
            Dav1dPictureGuard::build_plane(&guard, pool, 1, c_width_bytes, ch, uv_alloc_bytes)
                .ok_or_else(|| Error::invalid_data(COMPONENT, "missing U chroma plane pointer"))?;
        let v_plane =
            Dav1dPictureGuard::build_plane(&guard, pool, 2, c_width_bytes, ch, uv_alloc_bytes)
                .ok_or_else(|| Error::invalid_data(COMPONENT, "missing V chroma plane pointer"))?;
        vec![y_plane, u_plane, v_plane]
    } else {
        vec![y_plane]
    };

    let mut frame = VideoFrame::from_planes(w as u32, h as u32, pixel_layout, planes);
    frame.color = color;
    frame.pts = Timestamp::new(pic.m.timestamp);
    frame.pkt_dts = frame.pts;
    frame.duration = pic.m.duration;
    frame.time_base = time_base;
    frame.sar = fallback_sar;

    if let Some(hdr) = guard.frame_hdr() {
        frame.picture_type = match hdr.frame_type {
            DAV1D_FRAME_TYPE_KEY | DAV1D_FRAME_TYPE_INTRA => PictureType::I,
            DAV1D_FRAME_TYPE_INTER => PictureType::P,
            DAV1D_FRAME_TYPE_SWITCH => PictureType::Sp,
            _ => PictureType::None,
        };
        if hdr.frame_type == DAV1D_FRAME_TYPE_KEY {
            frame.flags |= FrameFlags::KEY;
        }
        if hdr.render_width > 0 && hdr.render_height > 0 {
            let (sar_num, sar_den, _) = reduce(
                h as i64 * i64::from(hdr.render_width),
                w as i64 * i64::from(hdr.render_height),
                i64::from(i32::MAX),
            );
            if sar_num > 0 && sar_den > 0 {
                frame.sar = Rational::new(sar_num, sar_den);
            }
        }
    }

    if let Some(md) = guard.mastering_display() {
        frame
            .side_data
            .push(SideData::MasteringDisplay(MasteringDisplayMeta {
                display_primaries: md.primaries.map(|[x, y]| {
                    [
                        Rational::new(i32::from(x), 1 << 16),
                        Rational::new(i32::from(y), 1 << 16),
                    ]
                }),
                white_point: [
                    Rational::new(i32::from(md.white_point[0]), 1 << 16),
                    Rational::new(i32::from(md.white_point[1]), 1 << 16),
                ],
                max_luminance: Rational::new(md.max_luminance as i32, 1 << 8),
                min_luminance: Rational::new(md.min_luminance as i32, 1 << 14),
            }));
    }

    if let Some(cll) = guard.content_light() {
        frame.side_data.push(SideData::ContentLightLevel {
            max_cll: u32::from(cll.max_content_light_level),
            max_fall: u32::from(cll.max_frame_average_light_level),
        });
    }

    Ok(frame)
}

/// Strip the 4-byte `av1C` (`AV1CodecConfigurationRecord`) header if `extradata`
/// starts with the `av1C` marker bit (`0x81`), returning any trailing `configOBUs`.
fn extract_config_obus(extradata: &[u8], policy: ErrorPolicy) -> Result<&[u8]> {
    if extradata.is_empty() {
        return Ok(&[]);
    }
    if (extradata[0] & 0x80) != 0 {
        let version = extradata[0] & 0x7F;
        if version != 1 || extradata.len() < 4 {
            if policy == ErrorPolicy::Strict {
                return Err(Error::corrupt(COMPONENT, "malformed av1C extradata header"));
            }
            return Ok(&[]);
        }
        return Ok(&extradata[4..]);
    }
    Ok(extradata)
}

/// `rav1d` AV1 decoder implementing [`wedeo_core::Decoder`].
pub struct Rav1dDecoder {
    ctx: Dav1dContextGuard,
    codec_ctx: CodecContext,
    track_id: TrackId,
    time_base: Rational,
    fallback_color: ColorDesc,
    fallback_sar: Rational,
    pending_packets: VecDeque<Packet>,
    pending_frames: VecDeque<VideoFrame>,
    draining: bool,
    health: StreamHealth,
}

impl Rav1dDecoder {
    pub fn open(track: &TrackDescriptor, codec_ctx: CodecContext) -> Result<Self> {
        if track.codec_id != CodecId::Av1 {
            return Err(Error::unsupported(
                COMPONENT,
                format!("expected CodecId::Av1, got {:?}", track.codec_id),
            ));
        }

        let ctx = Dav1dContextGuard::open(
            codec_ctx.threads,
            codec_ctx.error_policy == ErrorPolicy::Strict,
        )?;

        let (fallback_color, fallback_sar, extradata) = match track.params.video() {
            Some(vp) => {
                let sar = if vp.sar.is_valid() {
                    vp.sar
                } else {
                    Rational::new(1, 1)
                };
                (vp.color, sar, vp.extradata.as_slice())
            }
            None => (ColorDesc::UNSPECIFIED, Rational::new(1, 1), &[][..]),
        };

        let config_obus = extract_config_obus(extradata, codec_ctx.error_policy)?;

        let mut decoder = Self {
            ctx,
            codec_ctx,
            track_id: track.id,
            time_base: track.time_base,
            fallback_color,
            fallback_sar,
            pending_packets: VecDeque::new(),
            pending_frames: VecDeque::new(),
            draining: false,
            health: StreamHealth::new(),
        };

        if !config_obus.is_empty() {
            match decoder.send_raw_data(config_obus, Timestamp::NONE.raw(), 0)? {
                RawSendOutcome::Accepted | RawSendOutcome::TryAgain => {
                    decoder.drain_pictures()?;
                }
                RawSendOutcome::InvalidData => {
                    decoder.record_corrupt(
                        Timestamp::NONE,
                        None,
                        "rav1d rejected corrupt AV1 extradata configOBUs",
                    )?;
                }
            }
        }

        Ok(decoder)
    }

    fn record_corrupt(
        &mut self,
        pts: Timestamp,
        byte_offset: Option<u64>,
        detail: &'static str,
    ) -> Result<()> {
        self.health.record(HealthEvent {
            track: self.track_id,
            pts,
            byte_offset,
            kind: HealthIssue::CorruptSyntax,
            detail,
        });
        if self.codec_ctx.error_policy == ErrorPolicy::Strict {
            return Err(Error::corrupt(COMPONENT, detail));
        }
        Ok(())
    }

    /// Feed raw OBU bytes to `dav1d_send_data`.
    fn send_raw_data(
        &mut self,
        data: &[u8],
        timestamp: i64,
        duration: i64,
    ) -> Result<RawSendOutcome> {
        if data.is_empty() {
            return Ok(RawSendOutcome::Accepted);
        }

        let code = self.ctx.send_data(data, timestamp, duration)?;
        if code == 0 {
            Ok(RawSendOutcome::Accepted)
        } else if code == -(libc::EAGAIN as c_int) {
            Ok(RawSendOutcome::TryAgain)
        } else {
            Ok(RawSendOutcome::InvalidData)
        }
    }

    /// Pull all ready pictures from `rav1d` into `self.pending_frames`.
    fn drain_pictures(&mut self) -> Result<()> {
        loop {
            let guard = match self.ctx.get_picture() {
                Ok(Some(guard)) => guard,
                Ok(None) => break,
                Err(_) => {
                    self.record_corrupt(
                        Timestamp::NONE,
                        None,
                        "rav1d reported corrupt picture during decode",
                    )?;
                    break;
                }
            };

            match picture_to_video_frame(
                guard,
                &self.codec_ctx.pool,
                self.fallback_color,
                self.fallback_sar,
                self.time_base,
            ) {
                Ok(frame) => self.pending_frames.push_back(frame),
                Err(e) => {
                    if self.codec_ctx.error_policy == ErrorPolicy::Strict {
                        return Err(e);
                    }
                    self.health.record(HealthEvent {
                        track: self.track_id,
                        pts: Timestamp::NONE,
                        byte_offset: None,
                        kind: HealthIssue::CorruptSyntax,
                        detail: "failed to convert decoded rav1d picture to VideoFrame",
                    });
                }
            }
        }

        Ok(())
    }

    /// Feed queued packets into `rav1d` until a frame is produced or the queue is empty.
    fn pump_pending_packets(&mut self) -> Result<()> {
        while let Some(pkt) = self.pending_packets.pop_front() {
            if pkt.time_base.is_valid() {
                self.time_base = pkt.time_base;
            }
            let outcome = self.send_raw_data(pkt.data.data(), pkt.pts.raw(), pkt.duration)?;
            match outcome {
                RawSendOutcome::Accepted => {
                    self.drain_pictures()?;
                    if !self.pending_frames.is_empty() {
                        break;
                    }
                }
                RawSendOutcome::TryAgain => {
                    self.drain_pictures()?;
                    self.pending_packets.push_front(pkt);
                    break;
                }
                RawSendOutcome::InvalidData => {
                    self.record_corrupt(
                        pkt.pts,
                        pkt.pos,
                        "rav1d rejected corrupt AV1 packet payload",
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl Decoder for Rav1dDecoder {
    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<SendStatus> {
        match pkt {
            Some(pkt) => {
                if self.draining {
                    return Ok(SendStatus::Full);
                }
                if pkt.time_base.is_valid() {
                    self.time_base = pkt.time_base;
                }
                if pkt.is_empty() {
                    return Ok(SendStatus::Accepted);
                }

                // If earlier packets are still queued waiting on `EAGAIN`, append behind them.
                if !self.pending_packets.is_empty() {
                    self.pump_pending_packets()?;
                    if !self.pending_packets.is_empty() {
                        if self.pending_packets.len() >= 64 {
                            return Ok(SendStatus::Full);
                        }
                        self.pending_packets.push_back(pkt.clone());
                        return Ok(SendStatus::Accepted);
                    }
                }

                // Fast path: feed directly from `&pkt.data` without cloning the packet.
                match self.send_raw_data(pkt.data.data(), pkt.pts.raw(), pkt.duration)? {
                    RawSendOutcome::Accepted => {
                        self.drain_pictures()?;
                        Ok(SendStatus::Accepted)
                    }
                    RawSendOutcome::TryAgain => {
                        self.drain_pictures()?;
                        match self.send_raw_data(pkt.data.data(), pkt.pts.raw(), pkt.duration)? {
                            RawSendOutcome::Accepted => {
                                self.drain_pictures()?;
                                Ok(SendStatus::Accepted)
                            }
                            RawSendOutcome::TryAgain => {
                                self.pending_packets.push_back(pkt.clone());
                                Ok(SendStatus::Accepted)
                            }
                            RawSendOutcome::InvalidData => {
                                self.record_corrupt(
                                    pkt.pts,
                                    pkt.pos,
                                    "rav1d rejected corrupt AV1 packet payload",
                                )?;
                                Ok(SendStatus::Accepted)
                            }
                        }
                    }
                    RawSendOutcome::InvalidData => {
                        self.record_corrupt(
                            pkt.pts,
                            pkt.pos,
                            "rav1d rejected corrupt AV1 packet payload",
                        )?;
                        Ok(SendStatus::Accepted)
                    }
                }
            }
            None => {
                self.draining = true;
                Ok(SendStatus::Accepted)
            }
        }
    }

    fn receive(&mut self) -> Result<RecvStatus<MediaUnit>> {
        if let Some(frame) = self.pending_frames.pop_front() {
            return Ok(RecvStatus::Ready(MediaUnit::Video(frame)));
        }

        self.pump_pending_packets()?;

        if let Some(frame) = self.pending_frames.pop_front() {
            return Ok(RecvStatus::Ready(MediaUnit::Video(frame)));
        }

        if self.draining {
            self.drain_pictures()?;
            if let Some(frame) = self.pending_frames.pop_front() {
                return Ok(RecvStatus::Ready(MediaUnit::Video(frame)));
            }
            if self.pending_packets.is_empty() {
                return Ok(RecvStatus::Eof);
            }
        }

        Ok(RecvStatus::NeedMore)
    }

    fn flush(&mut self) {
        self.ctx.flush();
        self.pending_packets.clear();
        self.pending_frames.clear();
        self.draining = false;
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

/// Factory for opening the segregated `rav1d` AV1 decoder.
pub struct Rav1dAv1DecoderFactory;

impl DecoderFactory for Rav1dAv1DecoderFactory {
    fn descriptor(&self) -> &DecoderDescriptor {
        static DESC: DecoderDescriptor = DecoderDescriptor {
            codec_id: CodecId::Av1,
            name: "av1_rav1d",
            long_name: "AV1 Video [rav1d]",
            media_type: MediaType::Video,
        };
        &DESC
    }

    fn open(&self, track: &TrackDescriptor, ctx: CodecContext) -> Result<Box<dyn Decoder>> {
        Ok(Box::new(Rav1dDecoder::open(track, ctx)?))
    }
}

/// Register the `rav1d` AV1 decoder factory into `reg`.
pub fn register_rav1d(reg: &mut Registry) {
    reg.register_decoder(Arc::new(Rav1dAv1DecoderFactory));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_dimension_chroma_uses_div_ceil() {
        assert_eq!(chroma_dims(DAV1D_PIXEL_LAYOUT_I420, 101, 75), (51, 38));
        assert_eq!(chroma_dims(DAV1D_PIXEL_LAYOUT_I422, 101, 75), (51, 75));
        assert_eq!(chroma_dims(DAV1D_PIXEL_LAYOUT_I444, 101, 75), (101, 75));
        assert_eq!(chroma_dims(DAV1D_PIXEL_LAYOUT_I400, 101, 75), (0, 0));
    }

    #[test]
    fn map_pixel_layout_supports_8_10_12_bit_all_subsamplings() {
        assert_eq!(
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I420, 8, MatrixCoefficients::Bt709).unwrap(),
            PixelLayout::YUV420P
        );
        assert_eq!(
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I420, 10, MatrixCoefficients::Bt709).unwrap(),
            PixelLayout::YUV420P10LE
        );
        assert_eq!(
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I422, 8, MatrixCoefficients::Bt709).unwrap(),
            PixelLayout::YUV422P
        );
        assert_eq!(
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I422, 10, MatrixCoefficients::Bt709).unwrap(),
            PixelLayout::YUV422P10LE
        );
        assert_eq!(
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I444, 8, MatrixCoefficients::Bt709).unwrap(),
            PixelLayout::YUV444P
        );
        assert_eq!(
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I444, 10, MatrixCoefficients::Bt709).unwrap(),
            PixelLayout::YUV444P10LE
        );
        assert_eq!(
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I444, 8, MatrixCoefficients::Identity).unwrap(),
            PixelLayout::GBRP
        );
        assert_eq!(
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I444, 10, MatrixCoefficients::Identity).unwrap(),
            PixelLayout::GBRP10LE
        );
        assert_eq!(
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I400, 8, MatrixCoefficients::Unspecified).unwrap(),
            PixelLayout::GRAY8
        );
        let gray10 =
            map_pixel_layout(DAV1D_PIXEL_LAYOUT_I400, 10, MatrixCoefficients::Unspecified).unwrap();
        assert_eq!(gray10.depth, SampleDepth::U10);
        assert_eq!(gray10.plane_count(), 1);
    }

    #[test]
    fn register_rav1d_populates_registry() {
        let mut reg = Registry::new();
        register_rav1d(&mut reg);
        let dec = reg.find_decoder(CodecId::Av1).unwrap();
        assert_eq!(dec.descriptor().name, "av1_rav1d");
    }
}
