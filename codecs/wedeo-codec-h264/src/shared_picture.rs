#![forbid(unsafe_code)]
// Safe pooled picture buffers and immutable reference pictures for H.264 decode.
//
// `PicHandle` is uniquely owned (`&mut`) by `FrameDecodeContext` during macroblock
// decoding and in-loop deblocking. Once decoding finishes, `PicHandle::into_shared()`
// moves the completed `PictureBuffer` into an immutable `Arc<SharedPicture>` for DPB
// reference storage, returning the underlying buffer to `BufferPool` when the last
// reference drops — with zero `UnsafeCell` and zero `unsafe` code.

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use crate::deblock::PictureBuffer;

/// Progress sentinel: all rows decoded + deblocked.
pub const PROGRESS_COMPLETE: i32 = i32::MAX;

// ---------------------------------------------------------------------------
// BufferPool — reuse PictureBuffer allocations across frames
// ---------------------------------------------------------------------------

/// Pool of reusable `PictureBuffer`s to avoid per-frame allocations.
pub struct BufferPool {
    free: Vec<PictureBuffer>,
    width: u32,
    height: u32,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            free: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Acquire a buffer of the given dimensions, pre-filled with `0` (luma) / `uv_fill` (chroma).
    pub fn acquire(
        &mut self,
        width: u32,
        height: u32,
        mb_width: u32,
        mb_height: u32,
        uv_fill: u8,
    ) -> PictureBuffer {
        if width != self.width || height != self.height {
            self.free.clear();
            self.width = width;
            self.height = height;
        }

        if let Some(mut buf) = self.free.pop() {
            buf.y.fill(0);
            buf.u.fill(uv_fill);
            buf.v.fill(uv_fill);
            buf
        } else {
            let y_stride = width as usize;
            let uv_stride = (width / 2) as usize;
            PictureBuffer {
                y: vec![0u8; y_stride * height as usize],
                u: vec![uv_fill; uv_stride * (height / 2) as usize],
                v: vec![uv_fill; uv_stride * (height / 2) as usize],
                y_stride,
                uv_stride,
                width,
                height,
                mb_width,
                mb_height,
            }
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn reclaim(&mut self, buf: PictureBuffer) {
        if buf.width == self.width && buf.height == self.height && !buf.y.is_empty() {
            self.free.push(buf);
        }
    }
}

/// Immutable reference picture stored in the DPB and shared across worker threads.
pub struct SharedPicture {
    data: PictureBuffer,
    mb_height: u32,
    return_to: Option<Arc<Mutex<BufferPool>>>,
}

impl SharedPicture {
    pub fn new(pic: PictureBuffer) -> Arc<Self> {
        let mb_height = pic.mb_height;
        Arc::new(Self {
            data: pic,
            mb_height,
            return_to: None,
        })
    }

    /// Create a `SharedPicture` that returns its buffer to `pool` on drop.
    pub fn new_pooled(pic: PictureBuffer, pool: &Arc<Mutex<BufferPool>>) -> Arc<Self> {
        let mb_height = pic.mb_height;
        Arc::new(Self {
            data: pic,
            mb_height,
            return_to: Some(Arc::clone(pool)),
        })
    }

    /// Get a shared reference to the completed `PictureBuffer`.
    #[inline]
    pub fn data(&self) -> &PictureBuffer {
        &self.data
    }

    /// No-op for completed immutable reference pictures (kept for call-site compatibility).
    #[inline]
    pub fn publish_row(&self, _row: i32) {}

    /// No-op for completed immutable reference pictures (all reference frames are joined before dispatch).
    #[inline]
    pub fn wait_for_row(&self, _row: i32) {}

    /// No-op for completed immutable reference pictures.
    #[inline]
    pub fn mark_complete(&self) {}

    #[inline]
    pub fn is_complete(&self) -> bool {
        true
    }

    #[inline]
    pub fn mb_height(&self) -> u32 {
        self.mb_height
    }
}

impl Drop for SharedPicture {
    fn drop(&mut self) {
        if let Some(pool) = self.return_to.take() {
            let buf = std::mem::replace(&mut self.data, PictureBuffer::empty());
            if let Ok(mut pool) = pool.lock() {
                pool.reclaim(buf);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PicHandle — single-owner mutable wrapper for current-frame decode
// ---------------------------------------------------------------------------

/// Handle to the current frame's picture during decode.
///
/// Uniquely owns `PictureBuffer` during decode (`Deref` + `DerefMut` with zero `unsafe`),
/// then transitions into an immutable `Arc<SharedPicture>` via `into_shared()` when stored
/// in the DPB. If dropped without `into_shared()` (e.g. non-reference B-frame), returns
/// its buffer directly to the pool.
pub struct PicHandle {
    pic: PictureBuffer,
    return_to: Option<Arc<Mutex<BufferPool>>>,
}

impl PicHandle {
    pub fn new(pic: PictureBuffer) -> Self {
        Self {
            pic,
            return_to: None,
        }
    }

    /// Create a `PicHandle` whose buffer returns to `pool` when all references drop.
    pub fn new_pooled(pic: PictureBuffer, pool: &Arc<Mutex<BufferPool>>) -> Self {
        Self {
            pic,
            return_to: Some(Arc::clone(pool)),
        }
    }

    /// Convert into an immutable `Arc<SharedPicture>` for DPB storage.
    pub fn into_shared(mut self) -> Arc<SharedPicture> {
        let pic = std::mem::replace(&mut self.pic, PictureBuffer::empty());
        let return_to = self.return_to.take();
        let mb_height = pic.mb_height;
        Arc::new(SharedPicture {
            data: pic,
            mb_height,
            return_to,
        })
    }

    /// Signal a row is decoded.
    #[inline]
    pub fn publish_row(&self, _row: i32) {}

    /// Mark picture as fully complete.
    #[inline]
    pub fn mark_complete(&self) {}
}

impl Deref for PicHandle {
    type Target = PictureBuffer;

    #[inline]
    fn deref(&self) -> &PictureBuffer {
        &self.pic
    }
}

impl DerefMut for PicHandle {
    #[inline]
    fn deref_mut(&mut self) -> &mut PictureBuffer {
        &mut self.pic
    }
}

impl Drop for PicHandle {
    fn drop(&mut self) {
        if let Some(pool) = self.return_to.take() {
            let buf = std::mem::replace(&mut self.pic, PictureBuffer::empty());
            if let Ok(mut pool) = pool.lock() {
                pool.reclaim(buf);
            }
        }
    }
}
