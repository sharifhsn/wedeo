// Thread-safe picture with row-level progress tracking for frame-level threading.
//
// One writer thread decodes rows top-to-bottom, calling `publish_row()` after
// each MB row. Reader threads (decoding later frames) call `wait_for_row()`
// before accessing reference pixels.
//
// Reference: FFmpeg libavcodec/pthread_frame.c (ff_thread_report_progress,
// ff_thread_await_progress), WEDEO_SYNTHESIS.md §4 (hybrid spin→condvar).

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::deblock::PictureBuffer;

/// Progress sentinel: all rows decoded + deblocked.
pub const PROGRESS_COMPLETE: i32 = i32::MAX;

/// Thread-safe picture with row-level progress tracking.
///
/// SAFETY: Concurrent access is mediated by `row_progress`:
/// - Writer has exclusive access to rows > row_progress
/// - Readers may access rows <= row_progress.load(Acquire)
pub struct SharedPicture {
    data: UnsafeCell<PictureBuffer>,
    row_progress: AtomicI32,
    mb_height: u32,
    // For blocking wait when spin is insufficient
    progress_mutex: Mutex<()>,
    progress_cond: Condvar,
}

// SAFETY: Concurrent access is mediated by `row_progress` atomic.
// Writer has exclusive access to rows > row_progress; readers only
// access rows <= row_progress.load(Acquire).
unsafe impl Sync for SharedPicture {}

impl SharedPicture {
    pub fn new(pic: PictureBuffer) -> Arc<Self> {
        let mb_height = pic.mb_height;
        Arc::new(Self {
            data: UnsafeCell::new(pic),
            row_progress: AtomicI32::new(-1),
            mb_height,
            progress_mutex: Mutex::new(()),
            progress_cond: Condvar::new(),
        })
    }

    /// Get a shared reference to the inner PictureBuffer.
    ///
    /// # Safety
    ///
    /// Caller must only access pixel rows that have been published
    /// (rows <= self.row_progress). For fully-complete pictures
    /// (PROGRESS_COMPLETE), all access is safe.
    pub unsafe fn data(&self) -> &PictureBuffer {
        unsafe { &*self.data.get() }
    }

    /// Get a mutable reference to the inner PictureBuffer.
    ///
    /// # Safety
    ///
    /// Caller must be the sole writer, accessing rows > row_progress.
    /// Only the decode thread for this picture should call this.
    /// This returns `&mut` from `&self` via `UnsafeCell`, which is the
    /// standard interior mutability pattern for concurrent data structures.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn data_mut(&self) -> &mut PictureBuffer {
        unsafe { &mut *self.data.get() }
    }

    /// Called by decode thread after completing MB row `row`.
    pub fn publish_row(&self, row: i32) {
        self.row_progress.store(row, Ordering::Release);
        self.progress_cond.notify_all();
    }

    /// Called by reader threads before accessing reference pixels from `row`.
    /// Uses hybrid spin → condvar strategy (WEDEO_SYNTHESIS §4).
    pub fn wait_for_row(&self, row: i32) {
        // Fast path: already done
        if self.row_progress.load(Ordering::Acquire) >= row {
            return;
        }

        // Phase 1: brief spin (microsecond waits)
        let mut spin = 0;
        while self.row_progress.load(Ordering::Relaxed) < row && spin < 100 {
            spin += 1;
            std::hint::spin_loop();
        }
        if self.row_progress.load(Ordering::Acquire) >= row {
            return;
        }

        // Phase 2: condvar wait (millisecond+ waits)
        let mut guard = self.progress_mutex.lock().unwrap();
        while self.row_progress.load(Ordering::Acquire) < row {
            guard = self.progress_cond.wait(guard).unwrap();
        }
    }

    /// Mark picture as fully complete (all rows decoded + deblocked).
    pub fn mark_complete(&self) {
        self.publish_row(PROGRESS_COMPLETE);
    }

    pub fn is_complete(&self) -> bool {
        self.row_progress.load(Ordering::Relaxed) == PROGRESS_COMPLETE
    }

    pub fn mb_height(&self) -> u32 {
        self.mb_height
    }
}

// ---------------------------------------------------------------------------
// PicHandle — transparent Deref wrapper for current-frame decode
// ---------------------------------------------------------------------------

/// Handle to the current frame's picture during decode.
///
/// Derefs to `PictureBuffer` transparently, so `ctx.pic.y[offset]` works
/// without any changes to mb.rs, mc.rs, intra_pred.rs. Holds an
/// `Arc<SharedPicture>` that can be extracted for DPB storage.
pub struct PicHandle {
    shared: Arc<SharedPicture>,
}

impl PicHandle {
    pub fn new(pic: PictureBuffer) -> Self {
        Self {
            shared: SharedPicture::new(pic),
        }
    }

    /// Extract the inner Arc<SharedPicture> (consumes the handle).
    /// Used when storing in the DPB after decode + deblock.
    pub fn into_shared(self) -> Arc<SharedPicture> {
        self.shared
    }

    /// Get the Arc for cloning (e.g., for in-flight ref pic lists).
    pub fn shared(&self) -> &Arc<SharedPicture> {
        &self.shared
    }

    /// Signal a row is decoded (delegates to SharedPicture).
    pub fn publish_row(&self, row: i32) {
        self.shared.publish_row(row);
    }

    /// Mark picture as fully complete.
    pub fn mark_complete(&self) {
        self.shared.mark_complete();
    }
}

impl Deref for PicHandle {
    type Target = PictureBuffer;

    fn deref(&self) -> &PictureBuffer {
        // SAFETY: During decode, only one PicHandle exists per frame, and the
        // decode thread is the sole writer. Deref for read access is safe because
        // no other thread accesses this picture's data until it's published.
        unsafe { self.shared.data() }
    }
}

impl DerefMut for PicHandle {
    fn deref_mut(&mut self) -> &mut PictureBuffer {
        // SAFETY: PicHandle has exclusive ownership semantics — only one exists
        // per frame, and &mut self guarantees no concurrent borrows. The decode
        // thread is the sole writer to this picture's data.
        unsafe { self.shared.data_mut() }
    }
}
