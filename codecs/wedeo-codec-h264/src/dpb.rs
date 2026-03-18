// Decoded Picture Buffer (DPB) for H.264/AVC.
//
// The DPB stores decoded reference pictures for inter prediction and manages
// picture output ordering. For Baseline profile (no B-frames), output order
// equals decode order.
//
// Reference: ITU-T H.264 Annex A (level limits), Section 8.2.5 (DPB),
// FFmpeg libavcodec/h264_refs.c

use crate::deblock::PictureBuffer;
use crate::tables::LEVEL_MAX_DPB_MBS;

// ---------------------------------------------------------------------------
// RefStatus
// ---------------------------------------------------------------------------

/// Reference picture status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefStatus {
    /// Not used for reference.
    Unused,
    /// Short-term reference picture.
    ShortTerm,
    /// Long-term reference picture.
    LongTerm,
}

// ---------------------------------------------------------------------------
// DpbEntry
// ---------------------------------------------------------------------------

/// A decoded picture in the DPB.
pub struct DpbEntry {
    /// Decoded picture data (Y/U/V planes).
    pub pic: PictureBuffer,
    /// Picture Order Count.
    pub poc: i32,
    /// frame_num from the slice header.
    pub frame_num: u32,
    /// Reference status.
    pub status: RefStatus,
    /// Long-term frame index (only valid if status == LongTerm).
    pub long_term_frame_idx: u32,
    /// Per-4x4-block motion vectors (16 entries per MB, row-major).
    pub mv_info: Vec<[i16; 2]>,
    /// Per-4x4-block reference indices (16 entries per MB, row-major).
    pub ref_info: Vec<i8>,
    /// Whether this picture is needed for output (not yet displayed).
    pub needs_output: bool,
}

// ---------------------------------------------------------------------------
// Dpb
// ---------------------------------------------------------------------------

/// Decoded Picture Buffer.
///
/// Holds up to `max_size` decoded pictures (capped at 16 per the spec).
/// Entries are stored in a fixed-size Vec of Options for O(1) access.
pub struct Dpb {
    /// DPB entries (max 16).
    pub entries: Vec<Option<DpbEntry>>,
    /// Maximum DPB size (from SPS level limits).
    pub max_size: usize,
}

impl Dpb {
    /// Create a new DPB with the given maximum size.
    ///
    /// The size is clamped to [1, 16] per the H.264 spec.
    pub fn new(max_size: usize) -> Self {
        let clamped = max_size.clamp(1, 16);
        let mut entries = Vec::with_capacity(clamped);
        for _ in 0..clamped {
            entries.push(None);
        }
        Self {
            entries,
            max_size: clamped,
        }
    }

    /// Find a free slot in the DPB.
    ///
    /// Returns the index of the first empty slot, or `None` if the DPB is full.
    pub fn find_free_slot(&self) -> Option<usize> {
        self.entries.iter().position(|e| e.is_none())
    }

    /// Store a decoded picture in the DPB.
    ///
    /// Returns the slot index where it was stored, or `None` if no free
    /// slot is available.
    pub fn store(&mut self, entry: DpbEntry) -> Option<usize> {
        if let Some(idx) = self.find_free_slot() {
            self.entries[idx] = Some(entry);
            Some(idx)
        } else {
            None
        }
    }

    /// Get a reference picture by DPB index.
    pub fn get(&self, idx: usize) -> Option<&DpbEntry> {
        self.entries.get(idx).and_then(|e| e.as_ref())
    }

    /// Get a mutable reference to a picture by DPB index.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut DpbEntry> {
        self.entries.get_mut(idx).and_then(|e| e.as_mut())
    }

    /// Get the number of short-term references.
    pub fn num_short_term(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, Some(entry) if entry.status == RefStatus::ShortTerm))
            .count()
    }

    /// Get the number of long-term references.
    pub fn num_long_term(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, Some(entry) if entry.status == RefStatus::LongTerm))
            .count()
    }

    /// Get the total number of reference pictures (short-term + long-term).
    pub fn num_refs(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Some(entry) if entry.status == RefStatus::ShortTerm
                                   || entry.status == RefStatus::LongTerm
                )
            })
            .count()
    }

    /// Remove the oldest short-term reference (smallest frame_num).
    ///
    /// This implements the sliding window marking process from H.264
    /// Section 8.2.5.3. When the DPB is full, the short-term reference
    /// with the smallest frame_num is marked as "unused for reference".
    pub fn remove_oldest_short_term(&mut self) {
        let mut oldest_idx: Option<usize> = None;
        let mut oldest_frame_num = u32::MAX;

        for (i, entry) in self.entries.iter().enumerate() {
            if let Some(e) = entry
                && e.status == RefStatus::ShortTerm
                && e.frame_num < oldest_frame_num
            {
                oldest_frame_num = e.frame_num;
                oldest_idx = Some(i);
            }
        }

        if let Some(idx) = oldest_idx {
            self.mark_unused(idx);
        }
    }

    /// Clear all references (for IDR).
    ///
    /// Marks all entries as unused. Entries that still need output are kept
    /// but marked as non-reference; others are removed entirely.
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            if let Some(e) = entry {
                if e.needs_output {
                    e.status = RefStatus::Unused;
                } else {
                    *entry = None;
                }
            }
        }
    }

    /// Get indices of pictures ready for output, sorted by POC (ascending).
    ///
    /// For Baseline profile, output order matches decode order, but we
    /// still sort by POC for correctness with Main/High profile streams.
    pub fn get_output_pictures(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if let Some(entry) = e
                    && entry.needs_output
                {
                    return Some(i);
                }
                None
            })
            .collect();

        // Sort by POC ascending (display order)
        indices.sort_by_key(|&i| self.entries[i].as_ref().map(|e| e.poc).unwrap_or(i32::MAX));

        indices
    }

    /// Mark a picture as unused for reference.
    ///
    /// If the picture also doesn't need output, the slot is freed entirely.
    pub fn mark_unused(&mut self, idx: usize) {
        if let Some(entry) = self.entries.get_mut(idx)
            && let Some(e) = entry.as_mut()
        {
            e.status = RefStatus::Unused;
            if !e.needs_output {
                *entry = None;
            }
        }
    }

    /// Mark a picture as no longer needing output.
    ///
    /// If the picture is also unused for reference, the slot is freed.
    pub fn mark_output_done(&mut self, idx: usize) {
        if let Some(entry) = self.entries.get_mut(idx)
            && let Some(e) = entry.as_mut()
        {
            e.needs_output = false;
            if e.status == RefStatus::Unused {
                *entry = None;
            }
        }
    }

    /// Find a short-term reference by frame_num.
    pub fn find_short_term(&self, frame_num: u32) -> Option<usize> {
        self.entries.iter().position(|e| {
            matches!(e, Some(entry) if entry.status == RefStatus::ShortTerm
                                       && entry.frame_num == frame_num)
        })
    }

    /// Find a long-term reference by long_term_frame_idx.
    pub fn find_long_term(&self, lt_idx: u32) -> Option<usize> {
        self.entries.iter().position(|e| {
            matches!(e, Some(entry) if entry.status == RefStatus::LongTerm
                                       && entry.long_term_frame_idx == lt_idx)
        })
    }

    /// Returns true if the DPB is full (no free slots).
    pub fn is_full(&self) -> bool {
        self.find_free_slot().is_none()
    }

    /// Number of occupied slots (reference + output-pending).
    pub fn num_occupied(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }
}

// ---------------------------------------------------------------------------
// DPB sizing
// ---------------------------------------------------------------------------

/// Compute the maximum number of DPB frames from the SPS level.
///
/// The spec defines MaxDpbFrames = Min(MaxDpbMbs / (PicWidthInMbs * FrameHeightInMbs), 16).
/// We clamp to [1, 16].
///
/// Reference: ITU-T H.264 Table A-1, FFmpeg `h264_get_max_num_ref_frames`.
pub fn max_dpb_frames(level_idc: u8, mb_width: u32, mb_height: u32) -> usize {
    let max_dpb_mbs = LEVEL_MAX_DPB_MBS
        .iter()
        .find(|(level, _)| *level == level_idc as u32)
        .map(|(_, mbs)| *mbs)
        .unwrap_or(8100); // default to level 3.0

    let frame_mbs = mb_width * mb_height;
    if frame_mbs == 0 {
        return 1;
    }
    let max_frames = max_dpb_mbs / frame_mbs;
    (max_frames as usize).clamp(1, 16)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal DpbEntry for testing.
    fn make_entry(frame_num: u32, poc: i32, status: RefStatus) -> DpbEntry {
        DpbEntry {
            pic: PictureBuffer {
                y: vec![128; 16 * 16],
                u: vec![128; 8 * 8],
                v: vec![128; 8 * 8],
                y_stride: 16,
                uv_stride: 8,
                width: 16,
                height: 16,
                mb_width: 1,
                mb_height: 1,
            },
            poc,
            frame_num,
            status,
            long_term_frame_idx: 0,
            mv_info: vec![[0i16; 2]; 16],
            ref_info: vec![-1i8; 16],
            needs_output: true,
        }
    }

    #[test]
    fn test_dpb_new() {
        let dpb = Dpb::new(4);
        assert_eq!(dpb.max_size, 4);
        assert_eq!(dpb.entries.len(), 4);
        assert!(dpb.entries.iter().all(|e| e.is_none()));
    }

    #[test]
    fn test_dpb_new_clamped() {
        // Size 0 -> clamped to 1
        let dpb = Dpb::new(0);
        assert_eq!(dpb.max_size, 1);

        // Size 20 -> clamped to 16
        let dpb = Dpb::new(20);
        assert_eq!(dpb.max_size, 16);
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut dpb = Dpb::new(4);
        let entry = make_entry(0, 0, RefStatus::ShortTerm);
        let idx = dpb.store(entry).unwrap();

        let e = dpb.get(idx).unwrap();
        assert_eq!(e.frame_num, 0);
        assert_eq!(e.poc, 0);
        assert_eq!(e.status, RefStatus::ShortTerm);
    }

    #[test]
    fn test_store_full_dpb() {
        let mut dpb = Dpb::new(2);
        dpb.store(make_entry(0, 0, RefStatus::ShortTerm)).unwrap();
        dpb.store(make_entry(1, 2, RefStatus::ShortTerm)).unwrap();

        // DPB is full
        assert!(dpb.store(make_entry(2, 4, RefStatus::ShortTerm)).is_none());
        assert!(dpb.is_full());
    }

    #[test]
    fn test_num_short_term_long_term() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(0, 0, RefStatus::ShortTerm));
        dpb.store(make_entry(1, 2, RefStatus::ShortTerm));
        dpb.store(make_entry(2, 4, RefStatus::LongTerm));

        assert_eq!(dpb.num_short_term(), 2);
        assert_eq!(dpb.num_long_term(), 1);
        assert_eq!(dpb.num_refs(), 3);
    }

    #[test]
    fn test_sliding_window_removes_oldest() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(5, 10, RefStatus::ShortTerm));
        dpb.store(make_entry(2, 4, RefStatus::ShortTerm));
        dpb.store(make_entry(8, 16, RefStatus::ShortTerm));

        // Oldest is frame_num=2 (poc=4)
        dpb.remove_oldest_short_term();

        // frame_num=2 should be gone (or marked unused)
        assert!(dpb.find_short_term(2).is_none());
        assert!(dpb.find_short_term(5).is_some());
        assert!(dpb.find_short_term(8).is_some());
        assert_eq!(dpb.num_short_term(), 2);
    }

    #[test]
    fn test_idr_clears_all_refs() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(0, 0, RefStatus::ShortTerm));
        dpb.store(make_entry(1, 2, RefStatus::ShortTerm));
        dpb.store(make_entry(2, 4, RefStatus::LongTerm));

        dpb.clear();

        // All references should be cleared
        assert_eq!(dpb.num_short_term(), 0);
        assert_eq!(dpb.num_long_term(), 0);
        assert_eq!(dpb.num_refs(), 0);
        // But entries with needs_output=true are kept (just marked unused)
        assert!(dpb.num_occupied() > 0);
    }

    #[test]
    fn test_find_short_term() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(3, 6, RefStatus::ShortTerm));
        dpb.store(make_entry(7, 14, RefStatus::ShortTerm));

        assert!(dpb.find_short_term(3).is_some());
        assert!(dpb.find_short_term(7).is_some());
        assert!(dpb.find_short_term(5).is_none());
    }

    #[test]
    fn test_find_long_term() {
        let mut dpb = Dpb::new(4);
        let mut entry = make_entry(0, 0, RefStatus::LongTerm);
        entry.long_term_frame_idx = 2;
        dpb.store(entry);

        assert!(dpb.find_long_term(2).is_some());
        assert!(dpb.find_long_term(0).is_none());
    }

    #[test]
    fn test_mark_unused() {
        let mut dpb = Dpb::new(4);
        let idx = dpb.store(make_entry(0, 0, RefStatus::ShortTerm)).unwrap();

        // Still needs output, so marking unused keeps it
        dpb.mark_unused(idx);
        assert!(dpb.get(idx).is_some());
        assert_eq!(dpb.get(idx).unwrap().status, RefStatus::Unused);
        assert_eq!(dpb.num_short_term(), 0);

        // Now mark output done -> entry should be freed
        dpb.mark_output_done(idx);
        assert!(dpb.get(idx).is_none());
    }

    #[test]
    fn test_mark_unused_no_output() {
        let mut dpb = Dpb::new(4);
        let mut entry = make_entry(0, 0, RefStatus::ShortTerm);
        entry.needs_output = false;
        let idx = dpb.store(entry).unwrap();

        // Entry doesn't need output, so marking unused frees it
        dpb.mark_unused(idx);
        assert!(dpb.get(idx).is_none());
    }

    #[test]
    fn test_get_output_pictures_sorted_by_poc() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(2, 8, RefStatus::ShortTerm));
        dpb.store(make_entry(0, 0, RefStatus::ShortTerm));
        dpb.store(make_entry(1, 4, RefStatus::ShortTerm));

        let output = dpb.get_output_pictures();
        assert_eq!(output.len(), 3);
        // Should be sorted by POC: 0, 4, 8
        let pocs: Vec<i32> = output.iter().map(|&i| dpb.get(i).unwrap().poc).collect();
        assert_eq!(pocs, vec![0, 4, 8]);
    }

    #[test]
    fn test_max_dpb_frames_level31() {
        // Level 3.1: 18000 max DPB MBs
        // 1920x1080 = 120x68 = 8160 MBs per frame
        // 18000 / 8160 = 2
        assert_eq!(max_dpb_frames(31, 120, 68), 2);
    }

    #[test]
    fn test_max_dpb_frames_level40() {
        // Level 4.0: 32768 max DPB MBs
        // 1920x1080 = 120x68 = 8160 MBs per frame
        // 32768 / 8160 = 4
        assert_eq!(max_dpb_frames(40, 120, 68), 4);
    }

    #[test]
    fn test_max_dpb_frames_small_resolution() {
        // Level 3.0: 8100 max DPB MBs
        // 320x240 = 20x15 = 300 MBs per frame
        // 8100 / 300 = 27 -> clamped to 16
        assert_eq!(max_dpb_frames(30, 20, 15), 16);
    }

    #[test]
    fn test_max_dpb_frames_zero_mbs() {
        assert_eq!(max_dpb_frames(30, 0, 0), 1);
    }

    #[test]
    fn test_max_dpb_frames_unknown_level() {
        // Unknown level -> defaults to 8100 (level 3.0)
        // 1920x1080 = 8160 MBs
        // 8100 / 8160 = 0 -> clamped to 1
        assert_eq!(max_dpb_frames(99, 120, 68), 1);
    }
}
