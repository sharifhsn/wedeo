//! Isolated `unsafe` FFI and zero-copy buffer guards for `rav1d`.
//!
//! All raw pointer operations, `rav1d::dav1d_*` FFI calls, `Send`/`Sync` unsafe
//! markers, and `Drop` unref calls live in this module behind a safe `pub(crate)`
//! boundary so the rest of `wedeo-rav1d` compiles under `#![deny(unsafe_code)]`.

use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;

use rav1d::Dav1dResult;
use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{DAV1D_MAX_THREADS, Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::headers::{
    Dav1dFrameHeader, Dav1dSequenceHeader, Rav1dContentLightLevel, Rav1dMasteringDisplay,
};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_flush,
    dav1d_get_picture, dav1d_open, dav1d_picture_unref, dav1d_send_data,
};

use wedeo_core::{
    BorderPadding, BorderedPlane, Buffer, BufferPool, Error, ExternalCpuBuffer, Result,
};

const COMPONENT: &str = "rav1d";

/// Convert a `Dav1dResult` from `dav1d_open` to a Wedeo `Result`.
fn check_open_result(r: Dav1dResult) -> Result<()> {
    if r.0 == 0 {
        Ok(())
    } else {
        Err(Error::invalid_argument(
            COMPONENT,
            format!("dav1d_open failed with code {}", r.0),
        ))
    }
}

/// Clamp a caller-supplied thread count to `[0, DAV1D_MAX_THREADS]` without
/// wrapping large `usize` values into negative or small `c_int` values.
fn clamp_thread_count(threads: usize) -> c_int {
    let max_threads = usize::try_from(DAV1D_MAX_THREADS).unwrap_or(0);
    threads.min(max_threads) as c_int
}

/// RAII owner for an opened `Dav1dContext`.
pub(crate) struct Dav1dContextGuard(Option<Dav1dContext>);

// SAFETY: `Dav1dContext` is `RawArc<Rav1dContext>`, a raw pointer to an `Arc<Rav1dContext>`.
// `Rav1dContext` is thread-safe internally (`Mutex` + atomics), and `Dav1dContextGuard`
// holds unique ownership of its `Dav1dContext` handle, freeing it once in `Drop` via
// `dav1d_close`.
#[allow(unsafe_code)]
unsafe impl Send for Dav1dContextGuard {}

/// RAII owner for an initialized `Dav1dData` buffer passed to `dav1d_send_data`.
struct Dav1dDataGuard(Dav1dData);

impl Drop for Dav1dDataGuard {
    fn drop(&mut self) {
        if self.0.sz > 0 || self.0.r#ref.is_some() {
            // SAFETY: `self.0` still owns an unconsumed `rav1d` allocation (`sz > 0` or
            // `ref` is `Some`), and `dav1d_data_unref` decrements/frees it cleanly.
            #[allow(unsafe_code)]
            unsafe {
                dav1d_data_unref(NonNull::new(&mut self.0));
            }
        }
    }
}

impl Dav1dContextGuard {
    /// Open a new `rav1d` decoder context with the requested thread count and strictness.
    pub(crate) fn open(threads: usize, strict: bool) -> Result<Self> {
        let mut settings = MaybeUninit::<Dav1dSettings>::uninit();
        let settings_ptr = NonNull::new(settings.as_mut_ptr())
            .ok_or_else(|| Error::internal(COMPONENT, "null Dav1dSettings pointer"))?;
        // SAFETY: `dav1d_default_settings` initializes the `Dav1dSettings` struct.
        #[allow(unsafe_code)]
        unsafe {
            dav1d_default_settings(settings_ptr);
        }
        // SAFETY: `settings` was initialized by `dav1d_default_settings`.
        #[allow(unsafe_code)]
        let mut settings = unsafe { settings.assume_init() };

        let n_threads = clamp_thread_count(threads);
        if n_threads > 0 {
            settings.n_threads = n_threads;
            if n_threads == 1 {
                settings.max_frame_delay = 1;
            }
        }
        settings.apply_grain = 1;
        settings.strict_std_compliance = c_int::from(strict);

        let mut raw_ctx: Option<Dav1dContext> = None;
        // SAFETY: `dav1d_open` writes a newly opened context handle into `raw_ctx`.
        #[allow(unsafe_code)]
        let open_res = unsafe {
            dav1d_open(
                Some(NonNull::from(&mut raw_ctx)),
                Some(NonNull::from(&mut settings)),
            )
        };
        check_open_result(open_res)?;

        let ctx = raw_ctx
            .ok_or_else(|| Error::internal(COMPONENT, "dav1d_open returned null context"))?;
        Ok(Self(Some(ctx)))
    }

    /// Feed raw OBU bytes into `dav1d_send_data`, returning the raw `Dav1dResult` code.
    pub(crate) fn send_data(
        &mut self,
        data: &[u8],
        timestamp: i64,
        duration: i64,
    ) -> Result<c_int> {
        let ctx = self
            .0
            .ok_or_else(|| Error::internal(COMPONENT, "rav1d context already closed"))?;

        let mut dav1d_data = MaybeUninit::<Dav1dData>::zeroed();
        let dav1d_data_ptr = NonNull::new(dav1d_data.as_mut_ptr())
            .ok_or_else(|| Error::internal(COMPONENT, "null Dav1dData pointer"))?;

        // SAFETY: `dav1d_data_create` initializes `dav1d_data` and returns a pointer to an
        // internal buffer of `data.len()` bytes (or null on allocation failure).
        #[allow(unsafe_code)]
        let buf_ptr = unsafe { dav1d_data_create(Some(dav1d_data_ptr), data.len()) };
        if buf_ptr.is_null() {
            return Err(Error::internal(
                COMPONENT,
                "dav1d_data_create failed to allocate buffer",
            ));
        }

        // SAFETY: `dav1d_data_create` succeeded, so `dav1d_data` is initialized.
        #[allow(unsafe_code)]
        let mut dav1d_data = Dav1dDataGuard(unsafe { dav1d_data.assume_init() });

        if let Some(ptr) = dav1d_data.0.data {
            // SAFETY: `dav1d_data_create` allocated exactly `data.len()` bytes at `ptr`.
            #[allow(unsafe_code)]
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len());
            }
        }

        dav1d_data.0.m.timestamp = timestamp;
        dav1d_data.0.m.duration = duration;

        // SAFETY: `ctx` is an open `Dav1dContext` and `dav1d_data.0` is initialized.
        // Any unconsumed allocation is freed by `Dav1dDataGuard::drop`.
        #[allow(unsafe_code)]
        let result = unsafe { dav1d_send_data(Some(ctx), Some(NonNull::from(&mut dav1d_data.0))) };

        Ok(result.0)
    }

    /// Pull the next decoded picture from `rav1d`.
    ///
    /// Returns `Ok(Some(guard))` on a decoded picture, `Ok(None)` on `EAGAIN`,
    /// or `Err(code)` on any negative error code other than `EAGAIN`.
    pub(crate) fn get_picture(
        &mut self,
    ) -> std::result::Result<Option<Arc<Dav1dPictureGuard>>, c_int> {
        let Some(ctx) = self.0 else {
            return Ok(None);
        };

        let mut pic = MaybeUninit::<Dav1dPicture>::zeroed();
        let Some(pic_ptr) = NonNull::new(pic.as_mut_ptr()) else {
            return Err(-(libc::EINVAL as c_int));
        };

        // SAFETY: `dav1d_get_picture` writes a decoded picture into `pic` on success (`0`).
        #[allow(unsafe_code)]
        let result = unsafe { dav1d_get_picture(Some(ctx), Some(pic_ptr)) };
        let code = result.0;
        if code < 0 {
            if code == -(libc::EAGAIN as c_int) {
                return Ok(None);
            }
            return Err(code);
        }

        // SAFETY: `dav1d_get_picture` returned `0`, so `pic` is a valid initialized
        // `Dav1dPicture`. Wrapping in `Arc<Dav1dPictureGuard>` ensures `dav1d_picture_unref`
        // runs exactly once when all zero-copy plane buffers drop.
        #[allow(unsafe_code)]
        let decoded = unsafe { pic.assume_init() };
        Ok(Some(Arc::new(Dav1dPictureGuard::from_decoded(decoded))))
    }

    /// Flush buffered state inside the `rav1d` decoder context.
    pub(crate) fn flush(&mut self) {
        if let Some(ctx) = self.0 {
            // SAFETY: `ctx` is a valid open `Dav1dContext`; `dav1d_flush` resets decoder state.
            #[allow(unsafe_code)]
            unsafe {
                dav1d_flush(ctx);
            }
        }
    }
}

impl Drop for Dav1dContextGuard {
    fn drop(&mut self) {
        if let Some(ctx) = self.0.take() {
            let mut ctx_opt: Option<Dav1dContext> = Some(ctx);
            // SAFETY: `dav1d_close` takes the context from `ctx_opt`, stops worker threads,
            // frees the context, and writes `None` back.
            #[allow(unsafe_code)]
            unsafe {
                dav1d_close(NonNull::new(&mut ctx_opt));
            }
        }
    }
}

#[cfg(test)]
type TestUnrefHook = Box<dyn Fn(&mut Dav1dPicture) + Send + Sync>;

/// Shared RAII owner for a decoded `Dav1dPicture`.
///
/// Calls `dav1d_picture_unref` exactly once when the final plane buffer referencing
/// this decoded picture is dropped.
pub(crate) struct Dav1dPictureGuard {
    pic: Dav1dPicture,
    #[cfg(test)]
    test_unref: Option<TestUnrefHook>,
    #[cfg(test)]
    _test_backing: Option<Box<dyn std::any::Any + Send + Sync>>,
}

// SAFETY: Once returned by `dav1d_get_picture`, a `Dav1dPicture`'s pixel planes and
// reference-counted headers (`RawArc<Rav1dPictureData>`, `RawArc<DRav1d<...>>`) are
// immutable for the lifetime of the picture. On drop, `dav1d_picture_unref` converts
// those `RawArc` pointers back into `Arc`s and drops them, returning the backing buffer
// to `rav1d`'s thread-safe `Arc<MemPool<u8>>`.
#[allow(unsafe_code)]
unsafe impl Send for Dav1dPictureGuard {}

// SAFETY: `Dav1dPictureGuard` only exposes shared immutable access to the decoded
// plane bytes and headers after `dav1d_get_picture` completes.
#[allow(unsafe_code)]
unsafe impl Sync for Dav1dPictureGuard {}

impl Drop for Dav1dPictureGuard {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(ref on_unref) = self.test_unref {
            on_unref(&mut self.pic);
            return;
        }

        // SAFETY: `self.pic` was populated by a successful `dav1d_get_picture` call and
        // is unreferenced exactly once here when the `Arc<Dav1dPictureGuard>` drops.
        #[allow(unsafe_code)]
        unsafe {
            dav1d_picture_unref(NonNull::new(&mut self.pic));
        }
    }
}

impl Dav1dPictureGuard {
    fn from_decoded(pic: Dav1dPicture) -> Self {
        Self {
            pic,
            #[cfg(test)]
            test_unref: None,
            #[cfg(test)]
            _test_backing: None,
        }
    }

    #[cfg(test)]
    fn for_test(
        pic: Dav1dPicture,
        on_unref: impl Fn(&mut Dav1dPicture) + Send + Sync + 'static,
        backing: impl std::any::Any + Send + Sync,
    ) -> Arc<Self> {
        Arc::new(Self {
            pic,
            test_unref: Some(Box::new(on_unref)),
            _test_backing: Some(Box::new(backing)),
        })
    }

    /// Borrow the underlying `Dav1dPicture` metadata fields.
    #[inline]
    pub(crate) fn picture(&self) -> &Dav1dPicture {
        &self.pic
    }

    /// Safely borrow the sequence header if present on the picture.
    #[inline]
    pub(crate) fn seq_hdr(&self) -> Option<&Dav1dSequenceHeader> {
        let ptr = self.pic.seq_hdr?;
        // SAFETY: `seq_hdr` is valid and immutable for the lifetime of `self.pic`.
        #[allow(unsafe_code)]
        Some(unsafe { ptr.as_ref() })
    }

    /// Safely borrow the frame header if present on the picture.
    #[inline]
    pub(crate) fn frame_hdr(&self) -> Option<&Dav1dFrameHeader> {
        let ptr = self.pic.frame_hdr?;
        // SAFETY: `frame_hdr` is valid and immutable for the lifetime of `self.pic`.
        #[allow(unsafe_code)]
        Some(unsafe { ptr.as_ref() })
    }

    /// Safely borrow mastering display metadata if present on the picture.
    #[inline]
    pub(crate) fn mastering_display(&self) -> Option<&Rav1dMasteringDisplay> {
        let ptr = self.pic.mastering_display?;
        // SAFETY: `mastering_display` is valid and immutable for the lifetime of `self.pic`.
        #[allow(unsafe_code)]
        Some(unsafe { ptr.as_ref() })
    }

    /// Safely borrow content light level metadata if present on the picture.
    #[inline]
    pub(crate) fn content_light(&self) -> Option<&Rav1dContentLightLevel> {
        let ptr = self.pic.content_light?;
        // SAFETY: `content_light` is valid and immutable for the lifetime of `self.pic`.
        #[allow(unsafe_code)]
        Some(unsafe { ptr.as_ref() })
    }

    /// Build a `BorderedPlane` for `plane_idx` (`0..3`), returning `None` if that plane
    /// pointer is missing on the decoded picture.
    ///
    /// Uses zero-copy `ExternalCpuBuffer` whenever `stride > 0`, checked spans fit within
    /// `alloc_component_bytes` and `isize::MAX`, and falls back cleanly to a pooled copy
    /// otherwise.
    pub(crate) fn build_plane(
        guard: &Arc<Self>,
        pool: &Arc<BufferPool>,
        plane_idx: usize,
        width_bytes: usize,
        height: usize,
        alloc_component_bytes: usize,
    ) -> Option<BorderedPlane> {
        if plane_idx >= 3 {
            return None;
        }
        let plane_ptr = guard.pic.data[plane_idx]?;
        let stride = guard.pic.stride[usize::from(plane_idx > 0)];

        if let Some((logical_len, padded_len)) =
            compute_zero_copy_spans(stride, width_bytes, height, alloc_component_bytes)
        {
            let ext = Rav1dPlaneBuffer {
                _guard: Arc::clone(guard),
                ptr: plane_ptr.cast::<u8>(),
                logical_len,
                padded_len,
            };
            let buffer = Buffer::from_external(Box::new(ext));
            return Some(BorderedPlane::from_buffer(
                buffer,
                0,
                stride,
                width_bytes,
                height,
            ));
        }

        // SAFETY: `plane_ptr` comes from `guard.pic.data[plane_idx]`, kept alive by `guard`
        // for the duration of this copy and valid for `height` rows of `width_bytes` at `stride`.
        #[allow(unsafe_code)]
        let copied = unsafe { copy_plane_fallback(pool, plane_ptr, stride, width_bytes, height) };
        Some(copied)
    }
}

/// Compute `(logical_len, padded_len)` for zero-copy plane viewing using checked arithmetic
/// and `isize::MAX` bounds so `slice::from_raw_parts` preconditions always hold.
fn compute_zero_copy_spans(
    stride: isize,
    width_bytes: usize,
    height: usize,
    alloc_component_bytes: usize,
) -> Option<(usize, usize)> {
    if height == 0 || width_bytes == 0 || stride <= 0 {
        return None;
    }
    if alloc_component_bytes > isize::MAX as usize {
        return None;
    }
    let stride_usize = stride as usize;
    if stride_usize < width_bytes {
        return None;
    }
    let min_span = (height - 1)
        .checked_mul(stride_usize)?
        .checked_add(width_bytes)?;
    let logical_len = match height.checked_mul(stride_usize) {
        Some(full_span) if full_span <= alloc_component_bytes => full_span,
        _ => min_span,
    };
    if logical_len <= alloc_component_bytes && logical_len <= isize::MAX as usize {
        let padded_len = alloc_component_bytes.max(logical_len);
        if padded_len <= isize::MAX as usize {
            return Some((logical_len, padded_len));
        }
    }
    None
}

/// Zero-copy `ExternalCpuBuffer` view over one plane of a `Dav1dPictureGuard`.
struct Rav1dPlaneBuffer {
    _guard: Arc<Dav1dPictureGuard>,
    ptr: NonNull<u8>,
    logical_len: usize,
    padded_len: usize,
}

// SAFETY: `ptr` points into immutable plane memory owned by `_guard`, which stays alive
// and free of mutable aliases until `_guard` drops.
#[allow(unsafe_code)]
unsafe impl Send for Rav1dPlaneBuffer {}

// SAFETY: `as_slice` and `as_padded_slice` only return shared immutable `&[u8]` views.
#[allow(unsafe_code)]
unsafe impl Sync for Rav1dPlaneBuffer {}

impl ExternalCpuBuffer for Rav1dPlaneBuffer {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` points to at least `self.logical_len` (`<= isize::MAX`)
        // initialized, immutable bytes inside the decoded `Dav1dPicture` owned by
        // `self._guard`.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.logical_len)
        }
    }

    #[inline]
    fn as_padded_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` points to at least `self.padded_len` (`>= self.logical_len`,
        // `<= isize::MAX`) initialized, immutable bytes inside the `rav1d` picture
        // component allocation kept alive by `self._guard`.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.padded_len)
        }
    }
}

/// Fallback row-by-row copy into a pooled `BorderedPlane` when a picture plane
/// uses a negative (bottom-up) or unexpectedly narrow stride.
///
/// # Safety
/// `plane_ptr` must point to valid pixel data for `height` rows at signed byte `stride`,
/// with at least `width_bytes` (`<= isize::MAX`) readable bytes per row.
#[allow(unsafe_code)]
unsafe fn copy_plane_fallback(
    pool: &Arc<BufferPool>,
    plane_ptr: NonNull<c_void>,
    stride: isize,
    width_bytes: usize,
    height: usize,
) -> BorderedPlane {
    assert!(
        width_bytes <= isize::MAX as usize,
        "plane width_bytes exceeds isize::MAX"
    );
    let mut plane = BorderedPlane::new_reused(pool, width_bytes, height, BorderPadding::NONE);
    let base = plane_ptr.as_ptr() as *const u8;
    for row in 0..height {
        let row_offset = isize::try_from(row)
            .ok()
            .and_then(|r| r.checked_mul(stride))
            .expect("plane row byte offset overflowed isize");
        // SAFETY: Caller guarantees `plane_ptr` is valid for `height` rows at `stride`.
        // For negative strides, `base.offset(row_offset)` steps backward within the
        // allocated slab.
        #[allow(unsafe_code)]
        let src_row = unsafe {
            let row_ptr = base.offset(row_offset);
            std::slice::from_raw_parts(row_ptr, width_bytes)
        };
        if let Some(dst_row) = plane.row_mut(row) {
            dst_row.copy_from_slice(src_row);
        }
    }
    plane
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rav1d::include::dav1d::headers::{
        DAV1D_CHR_VERTICAL, DAV1D_COLOR_PRI_BT709, DAV1D_FRAME_TYPE_KEY, DAV1D_MC_BT709,
        DAV1D_PIXEL_LAYOUT_I420, DAV1D_TRC_BT709,
    };

    use super::*;

    struct TestSlabs {
        _y: Vec<u8>,
        _u: Vec<u8>,
        _v: Vec<u8>,
    }

    struct TestMetadata {
        _seq: Arc<Dav1dSequenceHeader>,
        _frame: Arc<Dav1dFrameHeader>,
        _md: Arc<Rav1dMasteringDisplay>,
        _cll: Arc<Rav1dContentLightLevel>,
    }

    #[test]
    fn picture_guard_zero_copy_plane_refcount_and_single_drop() {
        let pool = BufferPool::new();
        let unref_count = Arc::new(AtomicUsize::new(0));

        // 4x4 Y plane (stride 8, padded tail >= 64 bytes) and 2x2 U/V planes (stride 4).
        let mut y_slab = vec![0u8; 8 * 4 + 64];
        for r in 0..4 {
            for c in 0..4 {
                y_slab[r * 8 + c] = (r * 10 + c) as u8;
            }
        }
        let u_slab = vec![11u8; 4 * 2 + 64];
        let v_slab = vec![22u8; 4 * 2 + 64];

        let mut pic = Dav1dPicture::default();
        pic.p.w = 4;
        pic.p.h = 4;
        pic.p.bpc = 8;
        pic.p.layout = DAV1D_PIXEL_LAYOUT_I420;
        pic.stride = [8, 4];
        pic.data = [
            NonNull::new(y_slab.as_ptr().cast_mut()).map(NonNull::cast),
            NonNull::new(u_slab.as_ptr().cast_mut()).map(NonNull::cast),
            NonNull::new(v_slab.as_ptr().cast_mut()).map(NonNull::cast),
        ];

        let counter = Arc::clone(&unref_count);
        let guard = Dav1dPictureGuard::for_test(
            pic,
            move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            },
            TestSlabs {
                _y: y_slab,
                _u: u_slab,
                _v: v_slab,
            },
        );

        let y_plane = Dav1dPictureGuard::build_plane(&guard, &pool, 0, 4, 4, 8 * 4 + 64).unwrap();
        let u_plane = Dav1dPictureGuard::build_plane(&guard, &pool, 1, 2, 2, 4 * 2 + 64).unwrap();
        let v_plane = Dav1dPictureGuard::build_plane(&guard, &pool, 2, 2, 2, 4 * 2 + 64).unwrap();
        drop(guard);

        assert_eq!(unref_count.load(Ordering::SeqCst), 0);
        assert_eq!(y_plane.row(0).unwrap(), &[0, 1, 2, 3]);
        assert_eq!(y_plane.row(3).unwrap(), &[30, 31, 32, 33]);
        assert!(y_plane.buffer.data_with_padding().len() >= 8 * 4 + 64);
        assert_eq!(u_plane.row(1).unwrap(), &[11, 11]);
        assert_eq!(v_plane.row(1).unwrap(), &[22, 22]);

        drop(y_plane);
        drop(u_plane);
        assert_eq!(unref_count.load(Ordering::SeqCst), 0);
        drop(v_plane);
        assert_eq!(unref_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn picture_guard_send_and_sync_across_threads() {
        let pool = BufferPool::new();
        let unref_count = Arc::new(AtomicUsize::new(0));

        let mut y_slab = vec![0u8; 8 * 2 + 64];
        y_slab[0..4].copy_from_slice(&[1, 2, 3, 4]);
        y_slab[8..12].copy_from_slice(&[5, 6, 7, 8]);

        let pic = Dav1dPicture {
            stride: [8, 0],
            data: [
                NonNull::new(y_slab.as_ptr().cast_mut()).map(NonNull::cast),
                None,
                None,
            ],
            ..Default::default()
        };

        let counter = Arc::clone(&unref_count);
        let guard = Dav1dPictureGuard::for_test(
            pic,
            move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            },
            y_slab,
        );

        let plane_a = Dav1dPictureGuard::build_plane(&guard, &pool, 0, 4, 2, 8 * 2 + 64).unwrap();
        let plane_b = plane_a.clone();
        let guard_clone = Arc::clone(&guard);
        drop(guard);

        let t1 = std::thread::spawn(move || {
            assert_eq!(plane_a.row(0).unwrap(), &[1, 2, 3, 4]);
            assert!(plane_a.buffer.data_with_padding().len() >= 16);
        });
        let t2 = std::thread::spawn(move || {
            assert_eq!(plane_b.row(1).unwrap(), &[5, 6, 7, 8]);
            assert_eq!(guard_clone.picture().stride[0], 8);
        });

        t1.join().unwrap();
        t2.join().unwrap();
        assert_eq!(unref_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn build_plane_negative_and_narrow_stride_uses_copy_fallback() {
        let pool = BufferPool::new();
        let unref_count = Arc::new(AtomicUsize::new(0));

        // Bottom-up Y plane: memory stores bottom row at offset 0 and top row at offset 8.
        let mut y_neg = vec![0u8; 8 * 3];
        y_neg[0..4].copy_from_slice(&[20, 21, 22, 23]);
        y_neg[8..12].copy_from_slice(&[10, 11, 12, 13]);
        y_neg[16..20].copy_from_slice(&[1, 2, 3, 4]);
        // Narrow U plane: stride < width_bytes (alloc_component_bytes < min_span).
        let u_narrow: Vec<u8> = vec![41, 42, 43, 44, 45, 46];

        // SAFETY: `y_neg.as_ptr().add(16)` stays inside the single `y_neg` Vec allocation,
        // allowing `copy_plane_fallback` to step backward across rows `0..3` at stride `-8`.
        #[allow(unsafe_code)]
        let y_top_ptr = unsafe { y_neg.as_ptr().add(16).cast_mut() };
        let pic = Dav1dPicture {
            stride: [-8, 3],
            data: [
                NonNull::new(y_top_ptr).map(NonNull::cast),
                NonNull::new(u_narrow.as_ptr().cast_mut()).map(NonNull::cast),
                None,
            ],
            ..Default::default()
        };

        let counter = Arc::clone(&unref_count);
        let guard = Dav1dPictureGuard::for_test(
            pic,
            move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            },
            (y_neg, u_narrow),
        );

        let y_plane = Dav1dPictureGuard::build_plane(&guard, &pool, 0, 4, 3, 24).unwrap();
        let u_plane = Dav1dPictureGuard::build_plane(&guard, &pool, 1, 3, 2, 4).unwrap();
        drop(guard);

        // Fallback copied into pooled planes rather than retaining `guard`.
        assert_eq!(unref_count.load(Ordering::SeqCst), 1);
        assert_eq!(y_plane.row(0).unwrap(), &[1, 2, 3, 4]);
        assert_eq!(y_plane.row(1).unwrap(), &[10, 11, 12, 13]);
        assert_eq!(y_plane.row(2).unwrap(), &[20, 21, 22, 23]);
        assert_eq!(u_plane.row(0).unwrap(), &[41, 42, 43]);
        assert_eq!(u_plane.row(1).unwrap(), &[44, 45, 46]);
    }

    #[test]
    fn picture_guard_metadata_accessors_read_valid_headers() {
        let guard_empty = Dav1dPictureGuard::for_test(Dav1dPicture::default(), |_| {}, ());
        assert!(guard_empty.seq_hdr().is_none());
        assert!(guard_empty.frame_hdr().is_none());
        assert!(guard_empty.mastering_display().is_none());
        assert!(guard_empty.content_light().is_none());

        // SAFETY: `Dav1dSequenceHeader` and `Dav1dFrameHeader` are plain `#[repr(C)]`
        // integer/array structs for which all-zero bit patterns are valid.
        #[allow(unsafe_code)]
        let mut seq_hdr: Dav1dSequenceHeader = unsafe { MaybeUninit::zeroed().assume_init() };
        seq_hdr.color_description_present = 1;
        seq_hdr.pri = DAV1D_COLOR_PRI_BT709;
        seq_hdr.trc = DAV1D_TRC_BT709;
        seq_hdr.mtrx = DAV1D_MC_BT709;
        seq_hdr.chr = DAV1D_CHR_VERTICAL;
        let seq_hdr = Arc::new(seq_hdr);

        // SAFETY: Same as `seq_hdr`.
        #[allow(unsafe_code)]
        let mut frame_hdr: Dav1dFrameHeader = unsafe { MaybeUninit::zeroed().assume_init() };
        frame_hdr.frame_type = DAV1D_FRAME_TYPE_KEY;
        frame_hdr.render_width = 1920;
        frame_hdr.render_height = 1080;
        let frame_hdr = Arc::new(frame_hdr);

        let md = Arc::new(Rav1dMasteringDisplay {
            primaries: [[34000, 16000], [13250, 34500], [7500, 3000]],
            white_point: [15635, 16450],
            max_luminance: 1000 << 8,
            min_luminance: 50,
        });
        let cll = Arc::new(Rav1dContentLightLevel {
            max_content_light_level: 1000,
            max_frame_average_light_level: 400,
        });

        let pic = Dav1dPicture {
            seq_hdr: Some(NonNull::from(&*seq_hdr)),
            frame_hdr: Some(NonNull::from(&*frame_hdr)),
            mastering_display: Some(NonNull::from(&*md)),
            content_light: Some(NonNull::from(&*cll)),
            ..Default::default()
        };

        let guard = Dav1dPictureGuard::for_test(
            pic,
            |_| {},
            TestMetadata {
                _seq: seq_hdr,
                _frame: frame_hdr,
                _md: md,
                _cll: cll,
            },
        );

        assert_eq!(guard.seq_hdr().unwrap().pri, DAV1D_COLOR_PRI_BT709);
        assert_eq!(guard.frame_hdr().unwrap().frame_type, DAV1D_FRAME_TYPE_KEY);
        assert_eq!(guard.frame_hdr().unwrap().render_width, 1920);
        assert_eq!(guard.mastering_display().unwrap().max_luminance, 1000 << 8);
        assert_eq!(guard.content_light().unwrap().max_content_light_level, 1000);
    }

    #[test]
    fn build_plane_overflow_bounds_do_not_create_oversized_slice() {
        assert_eq!(
            compute_zero_copy_spans(8, 4, 2, usize::MAX),
            None,
            "alloc_component_bytes > isize::MAX must not produce an oversized slice"
        );
        assert_eq!(
            compute_zero_copy_spans(isize::MAX, 4, usize::MAX, isize::MAX as usize),
            None,
            "height * stride overflow must return None"
        );
        assert_eq!(
            compute_zero_copy_spans(16, 8, usize::MAX / 8, isize::MAX as usize),
            None,
            "min_span overflow must return None"
        );

        let pool = BufferPool::new();
        let unref_count = Arc::new(AtomicUsize::new(0));
        let slab: Vec<u8> = vec![9, 8, 7, 6, 0, 0, 0, 0, 5, 4, 3, 2, 0, 0, 0, 0];

        let pic = Dav1dPicture {
            stride: [8, 0],
            data: [
                NonNull::new(slab.as_ptr().cast_mut()).map(NonNull::cast),
                None,
                None,
            ],
            ..Default::default()
        };

        let counter = Arc::clone(&unref_count);
        let guard = Dav1dPictureGuard::for_test(
            pic,
            move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            },
            slab,
        );

        // Passing `usize::MAX` for `alloc_component_bytes` must cleanly fall back to a
        // pooled copy rather than wrapping `Rav1dPlaneBuffer::padded_len` to `usize::MAX`.
        let plane = Dav1dPictureGuard::build_plane(&guard, &pool, 0, 4, 2, usize::MAX).unwrap();
        drop(guard);

        assert_eq!(unref_count.load(Ordering::SeqCst), 1);
        assert_eq!(plane.row(0).unwrap(), &[9, 8, 7, 6]);
        assert_eq!(plane.row(1).unwrap(), &[5, 4, 3, 2]);
    }

    #[test]
    fn clamp_thread_count_does_not_wrap_large_usize() {
        assert_eq!(clamp_thread_count(0), 0);
        assert_eq!(clamp_thread_count(1), 1);
        assert_eq!(
            clamp_thread_count(DAV1D_MAX_THREADS as usize),
            DAV1D_MAX_THREADS
        );
        assert_eq!(clamp_thread_count(0x8000_0000), DAV1D_MAX_THREADS);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(clamp_thread_count(0x1_0000_0001), DAV1D_MAX_THREADS);
        assert_eq!(clamp_thread_count(usize::MAX), DAV1D_MAX_THREADS);
    }
}
