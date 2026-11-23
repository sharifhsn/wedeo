use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::buffer::{
    BorderPadding, BorderedPlane, Buffer, BufferPool, PooledRaw, SIMD_ALIGN, align_up,
    extend_horizontal_rows,
};
use crate::error::{Error, Result};

/// Sentinel stored in `row_progress` when all rows and borders are finalized.
const PROGRESS_COMPLETE: i32 = i32::MAX;

/// Sentinel stored in `row_progress` if the writer drops without calling `finish()`.
const PROGRESS_ABORTED: i32 = i32::MIN;

struct RowPlaneShared {
    pooled: Arc<PooledRaw>,
    total_len: usize,
    origin_offset: usize,
    stride: usize,
    width_bytes: usize,
    height: usize,
    border: BorderPadding,
    pixel_bytes: usize,
    /// Number of published active rows (`0..=height`), or `PROGRESS_COMPLETE` / `PROGRESS_ABORTED`.
    row_progress: AtomicI32,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
}

impl RowPlaneShared {
    /// Byte boundary separating immutable published rows `[0..split)` from
    /// writer-exclusive unpublished rows `[split..total_len)`.
    #[inline]
    fn split_byte(&self, published_rows: usize) -> usize {
        if published_rows == 0 {
            0
        } else {
            (self.border.top + published_rows) * self.stride
        }
    }
}

/// Allocate a synchronized row-progress 2D plane backed by `pool`, returning a unique
/// `PlaneRowWriter` (`!Clone`) and a shareable `PlaneRowReader` (`Clone + Send + Sync`).
///
/// # Soundness Model (Miri-verifiable)
/// - At every instant, `PlaneRowWriter` can only form `&mut [u8]` over bytes
///   `[split_byte(published_end) .. total_len)`.
/// - `PlaneRowReader` waits on `row_progress` (`Acquire`) and can only form `&[u8]` over
///   bytes `[0 .. split_byte(observed_published)]` (or `[0 .. total_len)` once complete).
/// - Because `observed_published <= published_end` and the writer advances `published_end`
///   *before* storing `Release`, mutable and shared byte slices are strictly disjoint.
/// - `PlaneRowWriter::finish()` transfers the exact same `Arc<PooledRaw>` zero-copy into
///   the output `BorderedPlane`.
pub fn row_progress_plane(
    pool: &Arc<BufferPool>,
    width_bytes: usize,
    height: usize,
    border: BorderPadding,
    pixel_bytes: usize,
) -> (PlaneRowWriter, PlaneRowReader) {
    let min_stride = border
        .left
        .checked_add(width_bytes)
        .and_then(|v| v.checked_add(border.right))
        .expect("row plane stride overflow");
    let stride = align_up(min_stride, SIMD_ALIGN).max(SIMD_ALIGN);
    let total_rows = border
        .top
        .checked_add(height)
        .and_then(|v| v.checked_add(border.bottom))
        .expect("row plane height overflow");
    let total_len = stride
        .checked_mul(total_rows)
        .expect("row plane size overflow");
    let origin_offset = border.top * stride + border.left;
    let pooled = pool.acquire_raw(total_len, true);

    let shared = Arc::new(RowPlaneShared {
        pooled,
        total_len,
        origin_offset,
        stride,
        width_bytes,
        height,
        border,
        pixel_bytes: pixel_bytes.max(1),
        row_progress: AtomicI32::new(0),
        wait_lock: Mutex::new(()),
        wait_cv: Condvar::new(),
    });

    let writer = PlaneRowWriter {
        shared: Arc::clone(&shared),
        published_end: 0,
        finished: false,
    };
    let reader = PlaneRowReader { shared };
    (writer, reader)
}

/// Exclusive writer handle (`!Clone`) for a `RowProgressPlane`.
pub struct PlaneRowWriter {
    shared: Arc<RowPlaneShared>,
    /// Strictly monotonic lower bound: writer can NEVER mutably borrow active rows `< published_end`.
    published_end: usize,
    finished: bool,
}

impl PlaneRowWriter {
    #[inline]
    pub fn width_bytes(&self) -> usize {
        self.shared.width_bytes
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.shared.height
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.shared.stride
    }

    #[inline]
    pub fn border(&self) -> BorderPadding {
        self.shared.border
    }

    #[inline]
    pub fn published_rows(&self) -> usize {
        self.published_end
    }

    /// Borrow ONLY the unpublished tail `[published_end..height]` mutably.
    pub fn unpublished_rows_mut(&mut self) -> UnpublishedRowsMut<'_> {
        let start = self.shared.split_byte(self.published_end);
        let len = self.shared.total_len - start;
        // SAFETY: `PlaneRowWriter` is unique (`!Clone`), and readers can only access
        // bytes `< split_byte(self.published_end)`. Therefore `[start..total_len)` is
        // strictly disjoint from every active reader slice.
        let slice = unsafe {
            std::slice::from_raw_parts_mut(self.shared.pooled.base_ptr().as_ptr().add(start), len)
        };
        UnpublishedRowsMut {
            slice,
            first_row: self.published_end,
            height: self.shared.height,
            stride: self.shared.stride,
            width_bytes: self.shared.width_bytes,
            border: self.shared.border,
            includes_top_border: self.published_end == 0,
        }
    }

    /// Finalize horizontal borders (and top border on first publish) for active rows
    /// `published_end..new_published_end` and atomically publish them to concurrent readers.
    pub fn publish_up_to(&mut self, new_published_end: usize) {
        let new_end = new_published_end.min(self.shared.height);
        if new_end <= self.published_end {
            return;
        }
        self.extend_band(self.published_end, new_end);
        self.published_end = new_end;
        self.shared
            .row_progress
            .store(new_end as i32, Ordering::Release);
        // Lock before notify_all to prevent lost wakeups with waiting readers.
        let _guard = self
            .shared
            .wait_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.shared.wait_cv.notify_all();
    }

    /// Finalize any remaining unpublished rows and the bottom border, mark the plane
    /// permanently complete, and return a zero-copy `BorderedPlane`.
    pub fn finish(mut self) -> BorderedPlane {
        let height = self.shared.height;
        if self.published_end < height {
            self.extend_band(self.published_end, height);
            self.published_end = height;
        }
        self.extend_bottom_border();
        self.finished = true;

        self.shared
            .row_progress
            .store(PROGRESS_COMPLETE, Ordering::Release);
        {
            let _guard = self
                .shared
                .wait_lock
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            self.shared.wait_cv.notify_all();
        }

        let buffer =
            Buffer::from_pooled_raw(Arc::clone(&self.shared.pooled), self.shared.total_len);
        BorderedPlane {
            buffer,
            origin_offset: self.shared.origin_offset,
            stride: self.shared.stride as isize,
            width_bytes: self.shared.width_bytes,
            height: self.shared.height,
            border: self.shared.border,
        }
    }

    /// Extend horizontal borders for active rows `from_row..to_row` (and top border if `from_row == 0`),
    /// mutating ONLY bytes within `[split_byte(from_row)..split_byte(to_row))`.
    fn extend_band(&mut self, from_row: usize, to_row: usize) {
        let s = &self.shared;
        let band_start = s.split_byte(from_row);
        let band_end = s.split_byte(to_row);
        if band_end <= band_start {
            return;
        }

        // SAFETY: `from_row == self.published_end`, so `[band_start..band_end)` lies strictly
        // in the unpublished region exclusively owned by `self`.
        let band = unsafe {
            std::slice::from_raw_parts_mut(
                s.pooled.base_ptr().as_ptr().add(band_start),
                band_end - band_start,
            )
        };

        let first_phys_in_band = if from_row == 0 { s.border.top } else { 0 };
        extend_horizontal_rows(
            band,
            first_phys_in_band,
            to_row - from_row,
            s.stride,
            s.border.left,
            s.border.right,
            s.width_bytes,
            s.pixel_bytes,
        );

        // On first publish (from_row == 0), copy active row 0 into top border rows inside this band.
        if from_row == 0 && s.border.top > 0 {
            let src_off = s.border.top * s.stride;
            let (top_area, rest) = band.split_at_mut(src_off);
            let first_row = &rest[..s.stride];
            for dst_row in top_area.chunks_exact_mut(s.stride) {
                dst_row.copy_from_slice(first_row);
            }
        }
    }

    /// Extend the bottom border from the last active row (if `height > 0 && border.bottom > 0`).
    fn extend_bottom_border(&mut self) {
        let s = &self.shared;
        if s.height == 0 || s.border.bottom == 0 {
            return;
        }
        let last_row_off = (s.border.top + s.height - 1) * s.stride;
        let bottom_off = (s.border.top + s.height) * s.stride;
        let bottom_len = s.border.bottom * s.stride;

        // SAFETY:
        // - Active row `height - 1` is immutable (either already published or writer is exclusive
        //   and no longer writes it), so reading `[last_row_off .. bottom_off)` as `&[u8]` is safe.
        // - Bottom border `[bottom_off .. bottom_off + bottom_len)` is NEVER visible to readers
        //   until `PROGRESS_COMPLETE` is stored after this call, so mutating it as `&mut [u8]` is
        //   strictly disjoint from `[last_row_off .. bottom_off)` and all reader slices.
        unsafe {
            let base = s.pooled.base_ptr().as_ptr();
            let last_row = std::slice::from_raw_parts(base.add(last_row_off), s.stride);
            let bottom_area = std::slice::from_raw_parts_mut(base.add(bottom_off), bottom_len);
            for dst_row in bottom_area.chunks_exact_mut(s.stride) {
                dst_row.copy_from_slice(last_row);
            }
        }
    }
}

impl Drop for PlaneRowWriter {
    fn drop(&mut self) {
        if !self.finished {
            self.shared
                .row_progress
                .store(PROGRESS_ABORTED, Ordering::Release);
            let _guard = self
                .shared
                .wait_lock
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            self.shared.wait_cv.notify_all();
        }
    }
}

/// Disjoint mutable view over unpublished rows `[first_row..height]` of a `RowProgressPlane`.
pub struct UnpublishedRowsMut<'a> {
    slice: &'a mut [u8],
    first_row: usize,
    height: usize,
    stride: usize,
    width_bytes: usize,
    border: BorderPadding,
    includes_top_border: bool,
}

impl<'a> UnpublishedRowsMut<'a> {
    #[inline]
    pub fn first_row(&self) -> usize {
        self.first_row
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Byte offset of active row `active_y` within `self.raw_slice_mut()`.
    /// Returns `None` if `active_y` has already been published or is out of range.
    #[inline]
    pub fn row_offset(&self, active_y: usize) -> Option<usize> {
        if active_y < self.first_row || active_y >= self.height {
            return None;
        }
        let rel_row = if self.includes_top_border {
            self.border.top + active_y
        } else {
            active_y - self.first_row
        };
        Some(rel_row * self.stride + self.border.left)
    }

    /// Mutable active pixel slice for unpublished row `active_y`.
    #[inline]
    pub fn row_mut(&mut self, active_y: usize) -> Option<&mut [u8]> {
        let off = self.row_offset(active_y)?;
        Some(&mut self.slice[off..off + self.width_bytes])
    }

    /// Raw mutable slice covering all unpublished rows (including their horizontal borders).
    #[inline]
    pub fn raw_slice_mut(&mut self) -> &mut [u8] {
        self.slice
    }
}

/// Shareable concurrent reader handle for a `RowProgressPlane`.
#[derive(Clone)]
pub struct PlaneRowReader {
    shared: Arc<RowPlaneShared>,
}

impl PlaneRowReader {
    #[inline]
    pub fn width_bytes(&self) -> usize {
        self.shared.width_bytes
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.shared.height
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.shared.stride
    }

    #[inline]
    pub fn origin_offset(&self) -> usize {
        self.shared.origin_offset
    }

    #[inline]
    pub fn border(&self) -> BorderPadding {
        self.shared.border
    }

    /// Wait until at least `required_rows` active rows (`0..required_rows`) have been published.
    /// If `required_rows > self.height()`, waits until the entire plane (including bottom border)
    /// is complete.
    pub fn wait_for_rows(&self, required_rows: usize) -> Result<PublishedPlaneView<'_>> {
        let target = if required_rows > self.shared.height {
            PROGRESS_COMPLETE
        } else {
            required_rows as i32
        };
        let observed = self.wait_progress(target)?;
        Ok(self.make_view(observed))
    }

    /// Wait until the entire plane (all rows + all 4 borders) is permanently complete.
    pub fn wait_complete(&self) -> Result<PublishedPlaneView<'_>> {
        let observed = self.wait_progress(PROGRESS_COMPLETE)?;
        Ok(self.make_view(observed))
    }

    fn wait_progress(&self, target: i32) -> Result<i32> {
        // Fast path: already published.
        let p = self.shared.row_progress.load(Ordering::Acquire);
        if p == PROGRESS_ABORTED {
            return Err(Error::internal(
                "sync_plane",
                "plane writer aborted before completion",
            ));
        }
        if p >= target {
            return Ok(p);
        }

        // Brief spin for microsecond producer-consumer handoffs.
        for _ in 0..64 {
            std::hint::spin_loop();
            let p = self.shared.row_progress.load(Ordering::Acquire);
            if p == PROGRESS_ABORTED {
                return Err(Error::internal(
                    "sync_plane",
                    "plane writer aborted before completion",
                ));
            }
            if p >= target {
                return Ok(p);
            }
        }

        // Blocking wait on condvar.
        let mut guard = self
            .shared
            .wait_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        loop {
            let p = self.shared.row_progress.load(Ordering::Acquire);
            if p == PROGRESS_ABORTED {
                return Err(Error::internal(
                    "sync_plane",
                    "plane writer aborted before completion",
                ));
            }
            if p >= target {
                return Ok(p);
            }
            guard = self
                .shared
                .wait_cv
                .wait(guard)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    fn make_view(&self, observed: i32) -> PublishedPlaneView<'_> {
        let (published_rows, byte_len, complete) = if observed == PROGRESS_COMPLETE {
            (self.shared.height, self.shared.total_len, true)
        } else {
            let rows = (observed.max(0) as usize).min(self.shared.height);
            (rows, self.shared.split_byte(rows), false)
        };

        // SAFETY: `observed` was loaded with `Ordering::Acquire` synchronized with the writer's
        // `Ordering::Release`. The writer has permanently relinquished bytes `[0..byte_len)`.
        let slice =
            unsafe { std::slice::from_raw_parts(self.shared.pooled.base_ptr().as_ptr(), byte_len) };
        PublishedPlaneView {
            slice,
            origin_offset: self.shared.origin_offset,
            stride: self.shared.stride,
            width_bytes: self.shared.width_bytes,
            published_rows,
            complete,
        }
    }
}

/// Immutable shared view of all published rows (`0..published_rows`) plus their top/side borders
/// (and bottom border when `is_complete()`).
pub struct PublishedPlaneView<'a> {
    slice: &'a [u8],
    origin_offset: usize,
    stride: usize,
    width_bytes: usize,
    published_rows: usize,
    complete: bool,
}

impl<'a> PublishedPlaneView<'a> {
    #[inline]
    pub fn published_rows(&self) -> usize {
        self.published_rows
    }

    #[inline]
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    #[inline]
    pub fn origin_offset(&self) -> usize {
        self.origin_offset
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Full readable slice covering top border + published rows (and bottom border if complete).
    #[inline]
    pub fn padded_slice(&self) -> &'a [u8] {
        self.slice
    }

    /// Active pixel slice for published row `active_y`.
    #[inline]
    pub fn row(&self, active_y: usize) -> Option<&'a [u8]> {
        if active_y >= self.published_rows {
            return None;
        }
        let off = self.origin_offset + active_y * self.stride;
        Some(&self.slice[off..off + self.width_bytes])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn concurrent_row_producer_and_multiple_consumers() {
        let pool = BufferPool::new();
        let width = 64usize;
        let height = 32usize;
        let border = BorderPadding::uniform(4);
        let (mut writer, reader) = row_progress_plane(&pool, width, height, border, 1);

        let consumers: Vec<_> = (0..4)
            .map(|_| {
                let r = reader.clone();
                thread::spawn(move || {
                    for y in 0..height {
                        let view = r.wait_for_rows(y + 1).expect("reader wait failed");
                        assert!(view.published_rows() > y);
                        let row = view.row(y).expect("row must be published");
                        assert!(row.iter().all(|&b| b == (y as u8).wrapping_mul(7)));
                        // Verify horizontal left border of published row y was extended!
                        let left_off = view.origin_offset() + y * view.stride() - border.left;
                        assert_eq!(
                            &view.padded_slice()[left_off..left_off + border.left],
                            &[(y as u8).wrapping_mul(7); 4]
                        );
                    }
                    let full = r.wait_complete().expect("wait_complete failed");
                    assert!(full.is_complete());
                    // Verify bottom border replicates last row (height - 1)
                    let expected = ((height - 1) as u8).wrapping_mul(7);
                    let bot_off = full.origin_offset() + height * full.stride();
                    assert_eq!(
                        &full.padded_slice()[bot_off..bot_off + width],
                        &[expected; 64]
                    );
                })
            })
            .collect();

        // Producer writes 4 rows at a time and publishes progressively
        for band in (0..height).step_by(4) {
            {
                let mut unpub = writer.unpublished_rows_mut();
                assert_eq!(unpub.first_row(), band);
                // Cannot mutably borrow already-published rows
                if band > 0 {
                    assert!(unpub.row_mut(band - 1).is_none());
                }
                for y in band..(band + 4).min(height) {
                    let row = unpub.row_mut(y).unwrap();
                    row.fill((y as u8).wrapping_mul(7));
                }
            }
            writer.publish_up_to(band + 4);
        }

        let plane = writer.finish();
        // Zero-copy output plane matches all rows even while consumers may still hold readers
        for y in 0..height {
            assert_eq!(plane.row(y).unwrap(), &[(y as u8).wrapping_mul(7); 64]);
        }

        for handle in consumers {
            handle.join().unwrap();
        }
        // Once all readers drop, plane's buffer is uniquely writable again and returns to pool on drop
        drop(reader);
        assert!(plane.buffer.is_writable());
        drop(plane);
        assert_eq!(pool.pooled_count(), 1);
    }

    #[test]
    fn dropped_writer_wakes_reader_with_error() {
        let pool = BufferPool::new();
        let (writer, reader) = row_progress_plane(&pool, 16, 16, BorderPadding::NONE, 1);

        let t = thread::spawn(move || reader.wait_for_rows(8).is_err());
        drop(writer);
        assert!(t.join().unwrap());
    }

    #[test]
    fn wait_beyond_height_blocks_until_bottom_border_finished() {
        let pool = BufferPool::new();
        let border = BorderPadding::uniform(2);
        let (mut writer, reader) = row_progress_plane(&pool, 8, 4, border, 1);

        let r = reader.clone();
        let handle = thread::spawn(move || {
            // Requesting > height waits for PROGRESS_COMPLETE (bottom border finalized)
            let view = r.wait_for_rows(5).unwrap();
            assert!(view.is_complete());
            let bot_row = view.origin_offset() + 4 * view.stride();
            assert_eq!(&view.padded_slice()[bot_row..bot_row + 8], &[0xCC; 8]);
        });

        for y in 0..4 {
            writer
                .unpublished_rows_mut()
                .row_mut(y)
                .unwrap()
                .fill(if y == 3 { 0xCC } else { y as u8 });
            writer.publish_up_to(y + 1);
        }
        let plane = writer.finish();
        handle.join().unwrap();
        assert_eq!(plane.row(3).unwrap(), &[0xCC; 8]);
    }
}
