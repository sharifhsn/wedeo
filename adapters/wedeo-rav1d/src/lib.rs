//! rav1d AV1 decoder adapter for wedeo.
//!
//! Wraps the `rav1d` crate (pure Rust AV1 decoder, port of dav1d) behind
//! wedeo's `Decoder` trait, following the same pattern as `wedeo-symphonia`.
//!
//! ## Supported pixel formats
//!
//! AV1 Profile 0 (8-bit 420) and Profile 0 (10-bit 420) are fully supported.
//! 8-bit 422/444/monochrome are also handled. 10-bit 422/444 and 12-bit content
//! (Profile 2) require `PixelFormat` variants that wedeo doesn't have yet.

use std::collections::VecDeque;
use std::ffi::c_int;
use std::ffi::c_void;
use std::ptr::NonNull;

use rav1d::Dav1dResult;
use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::headers::{
    DAV1D_FRAME_TYPE_INTER, DAV1D_FRAME_TYPE_INTRA, DAV1D_FRAME_TYPE_KEY, DAV1D_FRAME_TYPE_SWITCH,
    DAV1D_PIXEL_LAYOUT_I400, DAV1D_PIXEL_LAYOUT_I420, DAV1D_PIXEL_LAYOUT_I422,
    DAV1D_PIXEL_LAYOUT_I444, Dav1dPixelLayout,
};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_flush,
    dav1d_get_picture, dav1d_open, dav1d_picture_unref, dav1d_send_data,
};

use wedeo_codec::decoder::{CodecParameters, Decoder, DecoderDescriptor};
use wedeo_codec::descriptor::{CodecCapabilities, CodecDescriptor, CodecProperties};
use wedeo_codec::registry::DecoderFactory;
use wedeo_core::buffer::Buffer;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::frame::{Frame, FrameData, FrameFlags, FramePlane, PictureType};
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::Packet;
use wedeo_core::pixel_format::PixelFormat;

/// Convert a `Dav1dResult` to a wedeo `Result`.
fn check_result(r: Dav1dResult) -> Result<()> {
    let code = r.0;
    if code == 0 {
        Ok(())
    } else if code == -(libc::EAGAIN as c_int) {
        Err(Error::Again)
    } else {
        Err(Error::InvalidData)
    }
}

/// Map a rav1d pixel layout + bit depth to a wedeo PixelFormat.
fn map_pixel_format(layout: Dav1dPixelLayout, bpc: c_int) -> Result<PixelFormat> {
    match (layout, bpc) {
        (DAV1D_PIXEL_LAYOUT_I420, 8) => Ok(PixelFormat::Yuv420p),
        (DAV1D_PIXEL_LAYOUT_I420, 10) => Ok(PixelFormat::Yuv420p10le),
        (DAV1D_PIXEL_LAYOUT_I422, 8) => Ok(PixelFormat::Yuv422p),
        (DAV1D_PIXEL_LAYOUT_I444, 8) => Ok(PixelFormat::Yuv444p),
        (DAV1D_PIXEL_LAYOUT_I400, 8) => Ok(PixelFormat::Gray8),
        // 10-bit 422/444 and all 12-bit layouts need PixelFormat variants
        // that wedeo doesn't have yet (AV1 Profile 2 / Professional).
        _ => {
            tracing::warn!("unsupported rav1d pixel format: layout={layout:?} bpc={bpc}");
            Err(Error::PatchwelcomeNotImplemented)
        }
    }
}

/// Chroma dimensions for a given layout.
fn chroma_dims(layout: Dav1dPixelLayout, w: usize, h: usize) -> (usize, usize) {
    match layout {
        DAV1D_PIXEL_LAYOUT_I420 => (w / 2, h / 2),
        DAV1D_PIXEL_LAYOUT_I422 => (w / 2, h),
        DAV1D_PIXEL_LAYOUT_I444 => (w, h),
        _ => (0, 0), // I400 monochrome
    }
}

/// Copy pixel data from a `Dav1dPicture` plane into a compact byte buffer.
///
/// Handles both positive strides (top-down) and negative strides (bottom-up).
/// For 8-bit, copies `width` bytes per row. For 10/12-bit, copies `width * 2`.
///
/// # Safety
/// `plane_ptr` must point to valid pixel data with the given stride for `height` rows.
unsafe fn copy_plane(
    plane_ptr: NonNull<c_void>,
    stride: isize,
    width: usize,
    height: usize,
    bpc: c_int,
) -> Vec<u8> {
    let bytes_per_pixel = if bpc > 8 { 2 } else { 1 };
    let row_bytes = width * bytes_per_pixel;
    let mut buf = Vec::with_capacity(row_bytes * height);

    let base = plane_ptr.as_ptr() as *const u8;
    for row in 0..height {
        // SAFETY: Caller guarantees plane_ptr is valid for height rows at the given stride.
        // Negative strides are valid — dav1d adjusts the base pointer so that
        // base + row * stride always lands within the allocated region.
        let row_ptr = unsafe { base.offset(row as isize * stride) };
        let slice = unsafe { std::slice::from_raw_parts(row_ptr, row_bytes) };
        buf.extend_from_slice(slice);
    }

    buf
}

/// RAII guard that calls `dav1d_picture_unref` on drop, preventing leaks
/// if pixel copying panics (e.g. OOM in Vec allocation).
struct PictureGuard(Option<Dav1dPicture>);

impl Drop for PictureGuard {
    fn drop(&mut self) {
        if let Some(ref mut pic) = self.0 {
            // SAFETY: The picture was successfully obtained from dav1d_get_picture.
            unsafe {
                dav1d_picture_unref(NonNull::new(pic as *mut _));
            }
        }
    }
}

/// Convert a `Dav1dPicture` to a wedeo `Frame`.
///
/// # Safety
/// The picture must contain valid plane pointers and be fully decoded.
unsafe fn picture_to_frame(pic: &Dav1dPicture) -> Result<Frame> {
    let w = pic.p.w;
    let h = pic.p.h;
    debug_assert!(w >= 0 && h >= 0, "negative picture dimensions from rav1d");
    let w = w as usize;
    let h = h as usize;
    let bpc = pic.p.bpc;
    let layout = pic.p.layout;
    let pixel_format = map_pixel_format(layout, bpc)?;

    let bytes_per_pixel = if bpc > 8 { 2usize } else { 1 };

    let mut frame = Frame::new_video(w as u32, h as u32, pixel_format);
    frame.pts = pic.m.timestamp;
    frame.duration = pic.m.duration;

    // Luma plane
    let y_ptr = pic.data[0].ok_or(Error::InvalidData)?;
    let y_stride = pic.stride[0];
    // SAFETY: Picture planes are valid after successful dav1d_get_picture.
    let y_data = unsafe { copy_plane(y_ptr, y_stride, w, h, bpc) };

    let y_plane = FramePlane {
        buffer: Buffer::from_slice(&y_data),
        offset: 0,
        linesize: w * bytes_per_pixel,
    };

    // Chroma planes (if not monochrome)
    let (cw, ch) = chroma_dims(layout, w, h);
    let planes = if cw > 0 && ch > 0 {
        let u_ptr = pic.data[1].ok_or(Error::InvalidData)?;
        let v_ptr = pic.data[2].ok_or(Error::InvalidData)?;
        let c_stride = pic.stride[1];

        // SAFETY: Picture planes are valid after successful dav1d_get_picture.
        let u_data = unsafe { copy_plane(u_ptr, c_stride, cw, ch, bpc) };
        let v_data = unsafe { copy_plane(v_ptr, c_stride, cw, ch, bpc) };

        vec![
            y_plane,
            FramePlane {
                buffer: Buffer::from_slice(&u_data),
                offset: 0,
                linesize: cw * bytes_per_pixel,
            },
            FramePlane {
                buffer: Buffer::from_slice(&v_data),
                offset: 0,
                linesize: cw * bytes_per_pixel,
            },
        ]
    } else {
        vec![y_plane]
    };

    // Determine picture type from frame header
    let pict_type = if let Some(frame_hdr) = pic.frame_hdr {
        // SAFETY: frame_hdr is valid while the picture (and its PictureGuard) is alive.
        let hdr = unsafe { &*frame_hdr.as_ptr() };
        if hdr.frame_type == DAV1D_FRAME_TYPE_KEY || hdr.frame_type == DAV1D_FRAME_TYPE_INTRA {
            PictureType::I
        } else if hdr.frame_type == DAV1D_FRAME_TYPE_INTER
            || hdr.frame_type == DAV1D_FRAME_TYPE_SWITCH
        {
            PictureType::P
        } else {
            PictureType::None
        }
    } else {
        PictureType::None
    };

    if pict_type == PictureType::I {
        frame.flags |= FrameFlags::KEY;
    }

    if let FrameData::Video(ref mut video) = frame.data {
        video.planes = planes;
        video.picture_type = pict_type;
    }

    Ok(frame)
}

/// rav1d AV1 decoder wrapper — implements wedeo's Decoder trait.
///
/// # Safety
/// `Dav1dContext` is `RawArc<Rav1dContext>` — a `Copy` raw pointer to an
/// `Arc<Rav1dContext>`. `RawArc` deliberately doesn't impl `Send` to force
/// C API callers to reason about ownership, but the underlying `Arc` is
/// `Send + Sync` and rav1d uses `Mutex`/atomics for internal synchronization.
/// We maintain single logical ownership: `ctx` lives in `Option<Dav1dContext>`,
/// is only freed once in `Drop` via `dav1d_close`, and intermediate copies
/// (from `Copy`) are only used for read-access API calls.
unsafe impl Send for Rav1dDecoderWrapper {}

struct Rav1dDecoderWrapper {
    ctx: Option<Dav1dContext>,
    pending_packets: VecDeque<Packet>,
    pending_frames: VecDeque<Frame>,
    drained: bool,
    codec_descriptor: CodecDescriptor,
}

impl Rav1dDecoderWrapper {
    fn new(params: CodecParameters) -> Result<Self> {
        let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();

        // SAFETY: dav1d_default_settings initializes the settings struct.
        unsafe {
            dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap());
        }
        let mut settings = unsafe { settings.assume_init() };

        // Configure threading
        if params.thread_count > 0 {
            settings.n_threads = params.thread_count as c_int;
        }
        // Apply film grain by default (matches FFmpeg behavior)
        settings.apply_grain = 1;

        let mut ctx: Option<Dav1dContext> = None;

        // SAFETY: dav1d_open writes a context handle to ctx.
        let result = unsafe {
            dav1d_open(
                Some(NonNull::new(&mut ctx as *mut _).unwrap()),
                Some(NonNull::new(&mut settings as *mut _).unwrap()),
            )
        };
        check_result(result)?;

        let ctx = ctx.ok_or(Error::InvalidData)?;

        let mut wrapper = Self {
            ctx: Some(ctx),
            pending_packets: VecDeque::new(),
            pending_frames: VecDeque::new(),
            drained: false,
            codec_descriptor: CodecDescriptor {
                id: params.codec_id,
                media_type: MediaType::Video,
                name: "av1_rav1d",
                long_name: "AV1 Video [rav1d]",
                properties: CodecProperties::LOSSY,
                profiles: &[],
            },
        };

        // If extradata is provided (av1C from MP4), send it as initial data
        if !params.extradata.is_empty() {
            let _ = wrapper.send_raw_data(&params.extradata, i64::MIN, 0);
            // Drain any frames produced by sequence header
            wrapper.drain_pictures();
        }

        Ok(wrapper)
    }

    /// Send raw OBU data to the decoder with a timestamp and duration.
    fn send_raw_data(&mut self, data: &[u8], timestamp: i64, duration: i64) -> Result<()> {
        let ctx = self.ctx.ok_or(Error::InvalidData)?;

        let mut dav1d_data = std::mem::MaybeUninit::<Dav1dData>::zeroed();
        let dav1d_data_ptr = NonNull::new(dav1d_data.as_mut_ptr()).unwrap();

        // SAFETY: dav1d_data_create allocates a buffer of the given size.
        // Returns a pointer to the allocated buffer, or null on error.
        let buf_ptr = unsafe { dav1d_data_create(Some(dav1d_data_ptr), data.len()) };
        if buf_ptr.is_null() {
            return Err(Error::InvalidData);
        }

        // SAFETY: dav1d_data_create succeeded, struct is now initialized.
        let mut dav1d_data = unsafe { dav1d_data.assume_init() };

        // Copy packet data into the rav1d buffer
        if let Some(ptr) = dav1d_data.data {
            // SAFETY: dav1d_data_create allocated a buffer of exactly data.len() bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len());
            }
        }

        // Set timestamp and duration so rav1d propagates them to the output picture
        dav1d_data.m.timestamp = timestamp;
        dav1d_data.m.duration = duration;

        // SAFETY: We pass valid data to the decoder.
        let result = unsafe {
            dav1d_send_data(
                Some(ctx),
                Some(NonNull::new(&mut dav1d_data as *mut _).unwrap()),
            )
        };

        // On EAGAIN, rav1d didn't consume the data (internal buffer busy).
        // On other errors, data may not have been consumed either.
        // On success, rav1d takes ownership (clears dav1d_data internally).
        // Free our allocation in the non-success cases.
        if dav1d_data.sz > 0 {
            // SAFETY: Unreferencing data that rav1d did not consume.
            unsafe {
                dav1d_data_unref(Some(NonNull::new(&mut dav1d_data as *mut _).unwrap()));
            }
        }

        check_result(result)
    }

    /// Pull all available pictures from the decoder into our frame queue.
    fn drain_pictures(&mut self) {
        let Some(ctx) = self.ctx else { return };

        loop {
            let mut pic = std::mem::MaybeUninit::<Dav1dPicture>::zeroed();
            let pic_ptr = NonNull::new(pic.as_mut_ptr()).unwrap();

            // SAFETY: dav1d_get_picture writes a picture to the output.
            let result = unsafe { dav1d_get_picture(Some(ctx), Some(pic_ptr)) };

            if result.0 < 0 {
                break; // EAGAIN or error — no more pictures available
            }

            // SAFETY: dav1d_get_picture succeeded, pic is now initialized.
            // Wrap in PictureGuard to ensure dav1d_picture_unref is called
            // even if pixel copying panics (e.g. OOM).
            let guard = PictureGuard(Some(unsafe { pic.assume_init() }));

            // SAFETY: picture planes are valid while the guard holds the picture.
            let frame_result = unsafe { picture_to_frame(guard.0.as_ref().unwrap()) };

            // Explicitly unref via the guard (drop releases it)
            drop(guard);

            match frame_result {
                Ok(frame) => self.pending_frames.push_back(frame),
                Err(e) => {
                    tracing::warn!("failed to convert rav1d picture to frame: {e:?}");
                }
            }
        }
    }
}

impl Decoder for Rav1dDecoderWrapper {
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
        // First check if we already have decoded frames queued
        if let Some(frame) = self.pending_frames.pop_front() {
            return Ok(frame);
        }

        // Try to decode from pending packets
        while let Some(pkt) = self.pending_packets.pop_front() {
            let data = pkt.data.data().to_vec();
            let pts = pkt.pts;
            let duration = pkt.duration;

            match self.send_raw_data(&data, pts, duration) {
                Ok(()) => {}
                Err(Error::Again) => {
                    // Decoder's internal buffer is busy with previous data.
                    // Drain available pictures, then return to caller so it
                    // can call receive_frame() again (avoids infinite loop).
                    self.drain_pictures();
                    self.pending_packets.push_front(pkt);
                    return if let Some(frame) = self.pending_frames.pop_front() {
                        Ok(frame)
                    } else {
                        Err(Error::Again)
                    };
                }
                Err(e) => return Err(e),
            }

            // Try to get pictures after sending data
            self.drain_pictures();

            if let Some(frame) = self.pending_frames.pop_front() {
                return Ok(frame);
            }
        }

        // If draining, pull remaining frames from the decoder.
        // dav1d_get_picture automatically enters drain mode when no more
        // data is sent (it sets state.drain = true internally).
        if self.drained {
            self.drain_pictures();

            if let Some(frame) = self.pending_frames.pop_front() {
                return Ok(frame);
            }
            return Err(Error::Eof);
        }

        Err(Error::Again)
    }

    fn flush(&mut self) {
        if let Some(ctx) = self.ctx {
            // SAFETY: dav1d_flush resets the decoder state.
            unsafe {
                dav1d_flush(ctx);
            }
        }
        self.pending_packets.clear();
        self.pending_frames.clear();
        self.drained = false;
    }

    fn descriptor(&self) -> &CodecDescriptor {
        &self.codec_descriptor
    }
}

impl Drop for Rav1dDecoderWrapper {
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.take() {
            let mut ctx_opt: Option<Dav1dContext> = Some(ctx);
            // SAFETY: dav1d_close reads the context from ctx_opt, frees it,
            // and writes None back.
            unsafe {
                dav1d_close(NonNull::new(&mut ctx_opt as *mut _));
            }
        }
    }
}

// --- Factory registration ---

struct Rav1dAv1DecoderFactory;

impl DecoderFactory for Rav1dAv1DecoderFactory {
    fn descriptor(&self) -> &DecoderDescriptor {
        static DESC: DecoderDescriptor = DecoderDescriptor {
            codec_id: CodecId::Av1,
            name: "av1_rav1d",
            long_name: "AV1 Video [rav1d]",
            media_type: MediaType::Video,
            capabilities: CodecCapabilities::empty(),
            properties: CodecProperties::LOSSY,
            priority: 50, // adapter, not native
        };
        &DESC
    }

    fn create(&self, params: CodecParameters) -> Result<Box<dyn Decoder>> {
        Ok(Box::new(Rav1dDecoderWrapper::new(params)?))
    }
}

inventory::submit!(&Rav1dAv1DecoderFactory as &dyn DecoderFactory);
