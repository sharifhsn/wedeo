// wedeo-scale: Pixel format conversion (libswscale equivalent).
//
// Wraps dcv-color-primitives to convert between YUV planar formats and packed
// RGB/BGRA formats.  No spatial scaling yet — dimensions must match exactly.

use dcv_color_primitives as dcp;

use wedeo_core::Frame;
use wedeo_core::buffer::Buffer;
use wedeo_core::error::{Error, Result};
use wedeo_core::frame::{ColorRange, ColorSpace, FrameData, FramePlane, VideoFrameData};
use wedeo_core::pixel_format::PixelFormat;

// ---------------------------------------------------------------------------
// Mapping helpers
// ---------------------------------------------------------------------------

/// Map a wedeo `PixelFormat` to the corresponding `dcp::PixelFormat`.
fn to_dcp_pixel_format(pf: PixelFormat) -> Option<dcp::PixelFormat> {
    match pf {
        PixelFormat::Yuv420p => Some(dcp::PixelFormat::I420),
        PixelFormat::Nv12 => Some(dcp::PixelFormat::Nv12),
        PixelFormat::Rgb24 => Some(dcp::PixelFormat::Rgb),
        PixelFormat::Bgr24 => Some(dcp::PixelFormat::Bgr),
        PixelFormat::Rgba => Some(dcp::PixelFormat::Rgba),
        PixelFormat::Bgra => Some(dcp::PixelFormat::Bgra),
        PixelFormat::Argb => Some(dcp::PixelFormat::Argb),
        _ => None,
    }
}

/// Number of planes expected by dcp for a given pixel format.
fn dcp_num_planes(pf: dcp::PixelFormat) -> u32 {
    match pf {
        dcp::PixelFormat::I420 | dcp::PixelFormat::I444 | dcp::PixelFormat::I422 => 3,
        dcp::PixelFormat::Nv12 => 2,
        dcp::PixelFormat::Argb
        | dcp::PixelFormat::Bgra
        | dcp::PixelFormat::Bgr
        | dcp::PixelFormat::Rgba
        | dcp::PixelFormat::Rgb => 1,
    }
}

/// Determine the `dcp::ColorSpace` to use for a conversion.
///
/// RGB pixel formats always use `dcp::ColorSpace::Rgb`.
/// YUV pixel formats choose between BT.601 / BT.709 (full / limited range)
/// based on the wedeo `ColorSpace` and `ColorRange` carried in the frame.
fn dcp_color_space(
    dcp_pf: dcp::PixelFormat,
    color_space: ColorSpace,
    color_range: ColorRange,
) -> dcp::ColorSpace {
    match dcp_pf {
        // Packed RGB formats are always in the RGB color model.
        dcp::PixelFormat::Argb
        | dcp::PixelFormat::Bgra
        | dcp::PixelFormat::Bgr
        | dcp::PixelFormat::Rgba
        | dcp::PixelFormat::Rgb => dcp::ColorSpace::Rgb,

        // YUV formats — pick BT.601/709 + range.
        _ => {
            let full_range = matches!(color_range, ColorRange::Jpeg);
            match (color_space, full_range) {
                (ColorSpace::Bt709, false) => dcp::ColorSpace::Bt709,
                (ColorSpace::Bt709, true) => dcp::ColorSpace::Bt709FR,
                // Default to BT.601 for anything not explicitly BT.709.
                (_, false) => dcp::ColorSpace::Bt601,
                (_, true) => dcp::ColorSpace::Bt601FR,
            }
        }
    }
}

/// Check whether a conversion between two dcp pixel formats is supported.
fn is_conversion_supported(src: dcp::PixelFormat, dst: dcp::PixelFormat) -> bool {
    use dcp::PixelFormat::*;
    matches!(
        (src, dst),
        // YUV → packed RGB
        (I420, Rgb) | (I420, Bgr) | (I420, Rgba) | (I420, Bgra)
        | (Nv12, Rgb) | (Nv12, Bgr) | (Nv12, Rgba) | (Nv12, Bgra)
        // packed RGB → YUV
        | (Argb, I420) | (Argb, Nv12) | (Argb, Rgb)
        | (Bgra, I420) | (Bgra, Nv12) | (Bgra, Rgb)
        | (Bgr, I420)  | (Bgr, Nv12)  | (Bgr, Rgb)
        | (Rgb, Bgra)
    )
}

// ---------------------------------------------------------------------------
// Converter
// ---------------------------------------------------------------------------

/// Pixel format converter backed by `dcv-color-primitives`.
///
/// Converts video frames between YUV planar formats (I420 / NV12) and packed
/// RGB-family formats (RGB24, BGR24, RGBA, BGRA, ARGB) without rescaling.
pub struct Converter {
    src_format: PixelFormat,
    dst_format: PixelFormat,
    width: u32,
    height: u32,
    src_dcp: dcp::PixelFormat,
    dst_dcp: dcp::PixelFormat,
}

impl Converter {
    /// Create a new converter for the given format pair and dimensions.
    ///
    /// Returns `Error::InvalidArgument` if the conversion is not supported or
    /// if the dimensions are not valid for the source/destination formats
    /// (YUV 4:2:0 requires even width and height).
    pub fn new(src: PixelFormat, dst: PixelFormat, w: u32, h: u32) -> Result<Self> {
        let src_dcp = to_dcp_pixel_format(src).ok_or(Error::InvalidArgument)?;
        let dst_dcp = to_dcp_pixel_format(dst).ok_or(Error::InvalidArgument)?;

        // Same format is always allowed (noop).
        if src as i32 != dst as i32 && !is_conversion_supported(src_dcp, dst_dcp) {
            return Err(Error::InvalidArgument);
        }

        // YUV 4:2:0 needs even dimensions.
        let needs_even = matches!(src_dcp, dcp::PixelFormat::I420 | dcp::PixelFormat::Nv12)
            || matches!(dst_dcp, dcp::PixelFormat::I420 | dcp::PixelFormat::Nv12);
        if needs_even && (!w.is_multiple_of(2) || !h.is_multiple_of(2)) {
            return Err(Error::InvalidArgument);
        }

        if w == 0 || h == 0 {
            return Err(Error::InvalidArgument);
        }

        Ok(Self {
            src_format: src,
            dst_format: dst,
            width: w,
            height: h,
            src_dcp,
            dst_dcp,
        })
    }

    /// Returns `true` when source and destination formats are the same (no
    /// conversion needed).
    pub fn is_noop(&self) -> bool {
        self.src_format as i32 == self.dst_format as i32
    }

    /// Convert a source video `Frame` to the destination pixel format.
    ///
    /// All frame metadata (pts, duration, time_base, flags, side_data, etc.)
    /// is preserved.  The video-specific fields (picture_type, color_range,
    /// color_space, sample_aspect_ratio, crop) are copied from the source.
    ///
    /// Returns `Error::InvalidData` if the frame dimensions do not match the
    /// converter or if the frame is not a video frame.
    pub fn convert(&self, src: &Frame) -> Result<Frame> {
        let video = src.video().ok_or(Error::InvalidData)?;

        if video.width != self.width || video.height != self.height {
            return Err(Error::InvalidData);
        }
        if video.format as i32 != self.src_format as i32 {
            return Err(Error::InvalidData);
        }

        // Noop fast path — just clone.
        if self.is_noop() {
            let mut out = src.clone();
            if let Some(v) = out.video_mut() {
                v.format = self.dst_format;
            }
            return Ok(out);
        }

        // Build dcp ImageFormats.
        let src_cs = dcp_color_space(self.src_dcp, video.color_space, video.color_range);
        let dst_cs = dcp_color_space(self.dst_dcp, video.color_space, video.color_range);

        let src_img_fmt = dcp::ImageFormat {
            pixel_format: self.src_dcp,
            color_space: src_cs,
            num_planes: dcp_num_planes(self.src_dcp),
        };

        let dst_img_fmt = dcp::ImageFormat {
            pixel_format: self.dst_dcp,
            color_space: dst_cs,
            num_planes: dcp_num_planes(self.dst_dcp),
        };

        // Collect source plane byte slices.
        let expected_planes = dcp_num_planes(self.src_dcp) as usize;
        let src_refs = self.extract_src_planes(video)?;

        // Collect source strides (only for the planes we actually use).
        let src_strides: Vec<usize> = video.planes[..expected_planes]
            .iter()
            .map(|p| p.linesize)
            .collect();

        // Allocate destination buffers.
        let dst_num_planes = dcp_num_planes(self.dst_dcp) as usize;
        let mut dst_sizes = vec![0usize; dst_num_planes];
        dcp::get_buffers_size(self.width, self.height, &dst_img_fmt, None, &mut dst_sizes)
            .map_err(|_| Error::InvalidArgument)?;

        let mut dst_bufs: Vec<Vec<u8>> = dst_sizes.iter().map(|&sz| vec![0u8; sz]).collect();

        // Build mutable slice references for dcp.
        let mut dst_slices: Vec<&mut [u8]> =
            dst_bufs.iter_mut().map(|v| v.as_mut_slice()).collect();

        dcp::convert_image(
            self.width,
            self.height,
            &src_img_fmt,
            Some(&src_strides),
            &src_refs,
            &dst_img_fmt,
            None,
            &mut dst_slices,
        )
        .map_err(|e| Error::Other(format!("dcp convert_image failed: {e}")))?;

        // Build output frame.
        let dst_planes: Vec<FramePlane> = dst_bufs
            .into_iter()
            .enumerate()
            .map(|(i, buf)| {
                let linesize = self.compute_dst_linesize(i);
                FramePlane {
                    buffer: Buffer::from_slice(&buf),
                    offset: 0,
                    linesize,
                }
            })
            .collect();

        let dst_video = VideoFrameData {
            planes: dst_planes,
            width: self.width,
            height: self.height,
            format: self.dst_format,
            picture_type: video.picture_type,
            color_range: video.color_range,
            color_space: video.color_space,
            color_primaries: video.color_primaries,
            color_trc: video.color_trc,
            chroma_location: video.chroma_location,
            sample_aspect_ratio: video.sample_aspect_ratio,
            crop_top: video.crop_top,
            crop_bottom: video.crop_bottom,
            crop_left: video.crop_left,
            crop_right: video.crop_right,
        };

        Ok(Frame {
            data: FrameData::Video(dst_video),
            pts: src.pts,
            pkt_dts: src.pkt_dts,
            best_effort_timestamp: src.best_effort_timestamp,
            duration: src.duration,
            time_base: src.time_base,
            flags: src.flags,
            repeat_pict: src.repeat_pict,
            metadata: src.metadata.clone(),
            side_data: src.side_data.clone(),
        })
    }

    /// Extract source plane byte slices from frame planes, honoring the
    /// `offset` field.
    fn extract_src_planes<'a>(&self, video: &'a VideoFrameData) -> Result<Vec<&'a [u8]>> {
        let expected = dcp_num_planes(self.src_dcp) as usize;
        if video.planes.len() < expected {
            return Err(Error::InvalidData);
        }

        let mut out = Vec::with_capacity(expected);
        for plane in &video.planes[..expected] {
            let data = plane.buffer.data();
            if plane.offset > data.len() {
                return Err(Error::InvalidData);
            }
            out.push(&data[plane.offset..]);
        }
        Ok(out)
    }

    /// Compute the expected linesize for a destination plane.
    fn compute_dst_linesize(&self, plane_idx: usize) -> usize {
        let w = self.width as usize;
        match self.dst_dcp {
            dcp::PixelFormat::Argb | dcp::PixelFormat::Bgra | dcp::PixelFormat::Rgba => w * 4,
            dcp::PixelFormat::Rgb | dcp::PixelFormat::Bgr => w * 3,
            dcp::PixelFormat::I420 | dcp::PixelFormat::I444 | dcp::PixelFormat::I422 => {
                if plane_idx == 0 {
                    w
                } else {
                    // Chroma width: same as luma for I444, half for I422/I420.
                    match self.dst_dcp {
                        dcp::PixelFormat::I444 => w,
                        _ => w.div_ceil(2),
                    }
                }
            }
            dcp::PixelFormat::Nv12 => {
                if plane_idx == 0 {
                    w
                } else {
                    // Interleaved UV, same width as luma in bytes.
                    w
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// One-shot conversion: creates a temporary `Converter` and converts a single
/// frame.
pub fn convert_frame(src: &Frame, dst_format: PixelFormat) -> Result<Frame> {
    let video = src.video().ok_or(Error::InvalidData)?;
    let conv = Converter::new(video.format, dst_format, video.width, video.height)?;
    conv.convert(src)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::frame::{ColorRange, ColorSpace, FrameFlags, PictureType};
    use wedeo_core::metadata::Metadata;
    use wedeo_core::rational::Rational;
    use wedeo_core::timestamp::NOPTS_VALUE;

    /// Helper: build a minimal I420 frame with solid-color Y/U/V data.
    fn make_i420_frame(w: u32, h: u32, y_val: u8, u_val: u8, v_val: u8) -> Frame {
        let y_size = (w as usize) * (h as usize);
        let uv_w = (w as usize).div_ceil(2);
        let uv_h = (h as usize).div_ceil(2);
        let uv_size = uv_w * uv_h;

        let y_buf = Buffer::from_slice(&vec![y_val; y_size]);
        let u_buf = Buffer::from_slice(&vec![u_val; uv_size]);
        let v_buf = Buffer::from_slice(&vec![v_val; uv_size]);

        let planes = vec![
            FramePlane {
                buffer: y_buf,
                offset: 0,
                linesize: w as usize,
            },
            FramePlane {
                buffer: u_buf,
                offset: 0,
                linesize: uv_w,
            },
            FramePlane {
                buffer: v_buf,
                offset: 0,
                linesize: uv_w,
            },
        ];

        Frame {
            data: FrameData::Video(VideoFrameData {
                planes,
                width: w,
                height: h,
                format: PixelFormat::Yuv420p,
                picture_type: PictureType::I,
                color_range: ColorRange::Mpeg,
                color_space: ColorSpace::Bt709,
                color_primaries: wedeo_core::ColorPrimaries::default(),
                color_trc: wedeo_core::ColorTransferCharacteristic::default(),
                chroma_location: wedeo_core::ChromaLocation::default(),
                sample_aspect_ratio: Rational::new(1, 1),
                crop_top: 0,
                crop_bottom: 0,
                crop_left: 0,
                crop_right: 0,
            }),
            pts: 42,
            pkt_dts: NOPTS_VALUE,
            best_effort_timestamp: NOPTS_VALUE,
            duration: 1024,
            time_base: Rational::new(1, 44100),
            flags: FrameFlags::KEY,
            repeat_pict: 0,
            metadata: Metadata::new(),
            side_data: Vec::new(),
        }
    }

    #[test]
    fn noop_same_format() {
        let conv = Converter::new(PixelFormat::Yuv420p, PixelFormat::Yuv420p, 4, 2).unwrap();
        assert!(conv.is_noop());
    }

    #[test]
    fn noop_different_format() {
        let conv = Converter::new(PixelFormat::Yuv420p, PixelFormat::Rgb24, 4, 2).unwrap();
        assert!(!conv.is_noop());
    }

    #[test]
    fn unsupported_format_pair() {
        // Gray8 is not mappable to a dcp pixel format.
        let result = Converter::new(PixelFormat::Gray8, PixelFormat::Rgb24, 4, 2);
        assert!(result.is_err());
    }

    #[test]
    fn odd_dimensions_rejected_for_yuv420() {
        let result = Converter::new(PixelFormat::Yuv420p, PixelFormat::Rgb24, 3, 2);
        assert!(result.is_err());
        let result = Converter::new(PixelFormat::Yuv420p, PixelFormat::Rgb24, 4, 3);
        assert!(result.is_err());
    }

    #[test]
    fn zero_dimensions_rejected() {
        let result = Converter::new(PixelFormat::Rgb24, PixelFormat::Bgra, 0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn i420_to_rgb24_4x2() {
        // BT.601 limited range:
        // Y=16 U=128 V=128 → R=0 G=0 B=0 (black)
        let src = make_i420_frame(4, 2, 16, 128, 128);
        let conv = Converter::new(PixelFormat::Yuv420p, PixelFormat::Rgb24, 4, 2).unwrap();
        let dst = conv.convert(&src).unwrap();

        let video = dst.video().unwrap();
        assert_eq!(video.format, PixelFormat::Rgb24);
        assert_eq!(video.width, 4);
        assert_eq!(video.height, 2);
        assert_eq!(video.planes.len(), 1);

        // All RGB pixels should be near-black.
        let data = video.planes[0].buffer.data();
        let expected_size = 4 * 2 * 3; // w * h * 3 bytes per pixel
        assert!(data.len() >= expected_size);
        for (i, &byte) in data.iter().enumerate().take(expected_size) {
            // Allow small rounding tolerance (dcp uses integer approximation).
            assert!(byte <= 2, "pixel byte {i} = {byte}, expected near 0",);
        }
    }

    #[test]
    fn i420_to_rgb24_white() {
        // BT.601 limited range:
        // Y=235 U=128 V=128 → R=255 G=255 B=255 (white)
        let src = make_i420_frame(4, 2, 235, 128, 128);

        // Use BT.601 instead of BT.709 for this test — the make_i420_frame
        // helper sets Bt709 but we can override.
        let mut src = src;
        if let Some(v) = src.video_mut() {
            v.color_space = ColorSpace::Smpte170m; // BT.601
        }

        let conv = Converter::new(PixelFormat::Yuv420p, PixelFormat::Rgb24, 4, 2).unwrap();
        let dst = conv.convert(&src).unwrap();
        let video = dst.video().unwrap();
        let data = video.planes[0].buffer.data();

        for (i, &byte) in data.iter().enumerate().take(4 * 2 * 3) {
            assert!(byte >= 253, "pixel byte {i} = {byte}, expected near 255",);
        }
    }

    #[test]
    fn metadata_preserved() {
        let src = make_i420_frame(4, 2, 16, 128, 128);
        let conv = Converter::new(PixelFormat::Yuv420p, PixelFormat::Rgb24, 4, 2).unwrap();
        let dst = conv.convert(&src).unwrap();

        assert_eq!(dst.pts, 42);
        assert_eq!(dst.duration, 1024);
        assert_eq!(dst.time_base, Rational::new(1, 44100));
        assert_eq!(dst.flags, FrameFlags::KEY);

        let video = dst.video().unwrap();
        assert_eq!(video.picture_type, PictureType::I);
        assert_eq!(video.color_range, ColorRange::Mpeg);
        assert_eq!(video.color_space, ColorSpace::Bt709);
        assert_eq!(video.sample_aspect_ratio, Rational::new(1, 1));
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let src = make_i420_frame(4, 2, 16, 128, 128);
        let conv = Converter::new(PixelFormat::Yuv420p, PixelFormat::Rgb24, 8, 4).unwrap();
        let result = conv.convert(&src);
        assert!(result.is_err());
    }

    #[test]
    fn convert_frame_convenience() {
        let src = make_i420_frame(4, 2, 16, 128, 128);
        let dst = convert_frame(&src, PixelFormat::Rgba).unwrap();
        let video = dst.video().unwrap();
        assert_eq!(video.format, PixelFormat::Rgba);
        // RGBA: 4 bytes per pixel, single plane.
        assert_eq!(video.planes.len(), 1);
        assert!(video.planes[0].buffer.data().len() >= 4 * 2 * 4);
    }

    #[test]
    fn noop_convert_preserves_data() {
        let src = make_i420_frame(4, 2, 100, 50, 200);
        let conv = Converter::new(PixelFormat::Yuv420p, PixelFormat::Yuv420p, 4, 2).unwrap();
        let dst = conv.convert(&src).unwrap();

        let sv = src.video().unwrap();
        let dv = dst.video().unwrap();
        assert_eq!(sv.planes.len(), dv.planes.len());
        for (sp, dp) in sv.planes.iter().zip(dv.planes.iter()) {
            assert_eq!(
                &sp.buffer.data()[sp.offset..],
                &dp.buffer.data()[dp.offset..]
            );
        }
    }
}
