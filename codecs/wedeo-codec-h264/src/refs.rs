// H.264/AVC reference picture list construction and marking.
//
// For Baseline profile (no B-frames), only list 0 is used.
// Reference pictures are managed through sliding window or adaptive (MMCO)
// marking.
//
// Reference: ITU-T H.264 Sections 8.2.4 (list construction) and
// 8.2.5 (reference picture marking), FFmpeg libavcodec/h264_refs.c

use crate::dpb::{Dpb, RefStatus};
use crate::slice::{MmcoOp, RefPicListModification, SliceHeader};

// ---------------------------------------------------------------------------
// Reference list construction
// ---------------------------------------------------------------------------

/// Build reference list 0 for a P-slice (Baseline profile, frame-only).
///
/// Short-term references are sorted by pic_num descending (most recent
/// first). pic_num handles frame_num wrap-around per H.264 spec 8.2.4.1:
///   pic_num = frame_num                    if frame_num <= CurrFrameNum
///   pic_num = frame_num - MaxFrameNum      if frame_num > CurrFrameNum
/// where MaxFrameNum = 2^log2_max_frame_num.
///
/// Long-term references are appended after, sorted by
/// long_term_frame_idx ascending.
///
/// The list is then optionally reordered by ref_pic_list_modification
/// commands from the slice header.
///
/// Returns a Vec of DPB indices representing list 0.
///
/// Reference: FFmpeg `h264_initialise_ref_list` and `ff_h264_build_ref_list`.
pub fn build_ref_list_p(
    dpb: &Dpb,
    slice_hdr: &SliceHeader,
    current_frame_num: u32,
    max_frame_num: u32, // also MaxPicNum for frame mode
) -> Vec<usize> {
    // Compute pic_num with wrap-around handling (H.264 spec 8.2.4.1).
    let pic_num = |fn_: u32| -> i64 {
        if fn_ <= current_frame_num {
            fn_ as i64
        } else {
            fn_ as i64 - max_frame_num as i64
        }
    };

    // Collect short-term references with their pic_num values.
    let mut short_term: Vec<(usize, i64)> = Vec::new();
    for (i, entry) in dpb.entries.iter().enumerate() {
        if let Some(e) = entry
            && e.status == RefStatus::ShortTerm
        {
            short_term.push((i, pic_num(e.frame_num)));
        }
    }

    // Sort by pic_num descending (most recent first).
    short_term.sort_by_key(|&(_, p)| std::cmp::Reverse(p));

    // Collect long-term references
    let mut long_term: Vec<(usize, u32)> = Vec::new();
    for (i, entry) in dpb.entries.iter().enumerate() {
        if let Some(e) = entry
            && e.status == RefStatus::LongTerm
        {
            long_term.push((i, e.long_term_frame_idx));
        }
    }

    // Sort by long_term_frame_idx ascending
    long_term.sort_by_key(|&(_, lt_idx)| lt_idx);

    // Build initial list: short-term first, then long-term
    let mut list: Vec<usize> = short_term
        .iter()
        .map(|&(idx, _)| idx)
        .chain(long_term.iter().map(|&(idx, _)| idx))
        .collect();

    // Apply ref_pic_list_modification commands if present
    if !slice_hdr.ref_pic_list_modification_l0.is_empty() {
        apply_ref_pic_list_modification(
            &mut list,
            &slice_hdr.ref_pic_list_modification_l0,
            dpb,
            current_frame_num,
            slice_hdr.num_ref_idx_l0_active as usize,
            max_frame_num,
        );
    }

    // Truncate to num_ref_idx_l0_active
    let max_refs = slice_hdr.num_ref_idx_l0_active as usize;
    if list.len() > max_refs {
        list.truncate(max_refs);
    }

    list
}

/// Build reference lists L0 and L1 for a B-slice (frame-only).
///
/// L0: short-term with POC <= current (descending by POC),
///     then short-term with POC > current (ascending by POC),
///     then long-term (ascending by long_term_frame_idx).
/// L1: short-term with POC > current (ascending by POC),
///     then short-term with POC <= current (descending by POC),
///     then long-term (ascending by long_term_frame_idx).
/// If L0 == L1 and len > 1, swap L1[0] and L1[1].
///
/// Returns (list0, list1) as DPB indices.
///
/// Reference: FFmpeg `h264_initialise_ref_list`, H.264 spec 8.2.4.2.3-4.
pub fn build_ref_list_b(
    dpb: &Dpb,
    slice_hdr: &SliceHeader,
    current_poc: i32,
    max_frame_num: u32,
) -> (Vec<usize>, Vec<usize>) {
    // Collect short-term references split by POC relative to current.
    let mut st_before: Vec<(usize, i32)> = Vec::new(); // POC <= current
    let mut st_after: Vec<(usize, i32)> = Vec::new(); // POC > current

    for (i, entry) in dpb.entries.iter().enumerate() {
        if let Some(e) = entry
            && e.status == RefStatus::ShortTerm
        {
            if e.poc <= current_poc {
                st_before.push((i, e.poc));
            } else {
                st_after.push((i, e.poc));
            }
        }
    }

    // Sort: before = descending POC, after = ascending POC
    st_before.sort_by_key(|&(_, poc)| std::cmp::Reverse(poc));
    st_after.sort_by_key(|&(_, poc)| poc);

    // Collect long-term references sorted by long_term_frame_idx ascending
    let mut long_term: Vec<(usize, u32)> = Vec::new();
    for (i, entry) in dpb.entries.iter().enumerate() {
        if let Some(e) = entry
            && e.status == RefStatus::LongTerm
        {
            long_term.push((i, e.long_term_frame_idx));
        }
    }
    long_term.sort_by_key(|&(_, lt_idx)| lt_idx);
    let lt_indices: Vec<usize> = long_term.iter().map(|&(idx, _)| idx).collect();

    // Build L0: before + after + long-term
    let mut list0: Vec<usize> = st_before
        .iter()
        .map(|&(idx, _)| idx)
        .chain(st_after.iter().map(|&(idx, _)| idx))
        .chain(lt_indices.iter().copied())
        .collect();

    // Build L1: after + before + long-term
    let mut list1: Vec<usize> = st_after
        .iter()
        .map(|&(idx, _)| idx)
        .chain(st_before.iter().map(|&(idx, _)| idx))
        .chain(lt_indices.iter().copied())
        .collect();

    // If L0 == L1 and len > 1, swap L1[0] and L1[1]
    if list0.len() > 1 && list0 == list1 {
        list1.swap(0, 1);
    }

    // Apply ref_pic_list_modification for L0
    if !slice_hdr.ref_pic_list_modification_l0.is_empty() {
        apply_ref_pic_list_modification(
            &mut list0,
            &slice_hdr.ref_pic_list_modification_l0,
            dpb,
            slice_hdr.frame_num,
            slice_hdr.num_ref_idx_l0_active as usize,
            max_frame_num,
        );
    }
    // Apply ref_pic_list_modification for L1
    if !slice_hdr.ref_pic_list_modification_l1.is_empty() {
        apply_ref_pic_list_modification(
            &mut list1,
            &slice_hdr.ref_pic_list_modification_l1,
            dpb,
            slice_hdr.frame_num,
            slice_hdr.num_ref_idx_l1_active as usize,
            max_frame_num,
        );
    }

    // Truncate to active counts
    let max_l0 = slice_hdr.num_ref_idx_l0_active as usize;
    if list0.len() > max_l0 {
        list0.truncate(max_l0);
    }
    let max_l1 = slice_hdr.num_ref_idx_l1_active as usize;
    if list1.len() > max_l1 {
        list1.truncate(max_l1);
    }

    (list0, list1)
}

/// Apply ref_pic_list_modification() reordering commands to a reference list.
///
/// `max_pic_num` = MaxFrameNum for frame mode (2^log2_max_frame_num).
/// Wrap-around uses `& (max_pic_num - 1)` per H.264 spec 8.2.4.3.1.
fn apply_ref_pic_list_modification(
    list: &mut Vec<usize>,
    mods: &[RefPicListModification],
    dpb: &Dpb,
    current_frame_num: u32,
    max_ref_count: usize,
    max_pic_num: u32,
) {
    let mut pred_pic_num = current_frame_num;
    let mask = max_pic_num.wrapping_sub(1);

    for (index, modification) in mods.iter().enumerate() {
        match modification.idc {
            0 | 1 => {
                let abs_diff_pic_num = modification.val + 1;
                if modification.idc == 0 {
                    pred_pic_num = pred_pic_num.wrapping_sub(abs_diff_pic_num);
                } else {
                    pred_pic_num = pred_pic_num.wrapping_add(abs_diff_pic_num);
                }
                // Wrap modulo MaxPicNum (= MaxFrameNum for frame mode)
                pred_pic_num &= mask;
                if let Some(dpb_idx) = dpb.find_short_term(pred_pic_num) {
                    reorder_list(list, dpb_idx, index, max_ref_count);
                }
            }
            2 => {
                let long_term_pic_num = modification.val;
                if let Some(dpb_idx) = dpb.find_long_term(long_term_pic_num) {
                    reorder_list(list, dpb_idx, index, max_ref_count);
                }
            }
            _ => {}
        }
    }
}

/// Reorder a reference list by moving `dpb_idx` to position `target_pos`.
fn reorder_list(list: &mut Vec<usize>, dpb_idx: usize, target_pos: usize, max_len: usize) {
    let found_pos = list.iter().position(|&x| x == dpb_idx);

    let remove_pos = if let Some(pos) = found_pos {
        if pos >= target_pos {
            pos
        } else {
            return;
        }
    } else {
        list.push(dpb_idx);
        list.len() - 1
    };

    if remove_pos > target_pos {
        for i in (target_pos..remove_pos).rev() {
            list.swap(i, i + 1);
        }
    }

    if target_pos < list.len() {
        list[target_pos] = dpb_idx;
    }

    if list.len() > max_len {
        list.truncate(max_len);
    }
}

// ---------------------------------------------------------------------------
// Reference picture marking
// ---------------------------------------------------------------------------

/// Apply reference picture marking.
///
/// For IDR: clear all references, mark current as short-term (or long-term
/// if `long_term_reference_flag` is set).
///
/// For non-IDR with adaptive marking: apply MMCO operations.
/// For non-IDR with sliding window: remove oldest short-term ref if DPB full.
///
/// Reference: FFmpeg `ff_h264_execute_ref_pic_marking`.
pub fn mark_reference(
    dpb: &mut Dpb,
    slice_hdr: &SliceHeader,
    is_idr: bool,
    current_frame_num: u32,
    max_frame_num: u32,
    max_num_ref_frames: u32,
    current_dpb_idx: Option<usize>,
) {
    if is_idr {
        // Clear all entries except the current one. The current entry
        // was just stored with RefStatus::Unused and needs to survive
        // the clear so it can be marked as a reference picture.
        for (i, entry) in dpb.entries.iter_mut().enumerate() {
            if Some(i) == current_dpb_idx {
                continue;
            }
            if let Some(e) = entry {
                if e.needs_output {
                    e.status = RefStatus::Unused;
                } else {
                    *entry = None;
                }
            }
        }
        if let Some(idx) = current_dpb_idx
            && let Some(entry) = dpb.get_mut(idx)
        {
            if slice_hdr.long_term_reference_flag {
                entry.status = RefStatus::LongTerm;
                entry.long_term_frame_idx = 0;
            } else {
                entry.status = RefStatus::ShortTerm;
            }
        }
    } else if slice_hdr.adaptive_ref_pic_marking {
        apply_mmco(
            dpb,
            &slice_hdr.mmco_ops,
            current_frame_num,
            max_frame_num,
            current_dpb_idx,
        );
    } else {
        sliding_window_mark(dpb, max_num_ref_frames, current_dpb_idx);
    }
}

/// Sliding window reference picture marking.
fn sliding_window_mark(dpb: &mut Dpb, max_num_ref_frames: u32, current_dpb_idx: Option<usize>) {
    let num_st = dpb.num_short_term();
    let num_lt = dpb.num_long_term();

    if num_st > 0 && (num_st + num_lt) as u32 >= max_num_ref_frames.max(1) {
        dpb.remove_oldest_short_term();
    }

    if let Some(idx) = current_dpb_idx
        && let Some(entry) = dpb.get_mut(idx)
        && entry.status == RefStatus::Unused
    {
        entry.status = RefStatus::ShortTerm;
    }
}

/// Apply Memory Management Control Operations (MMCO).
fn apply_mmco(
    dpb: &mut Dpb,
    ops: &[MmcoOp],
    current_frame_num: u32,
    max_frame_num: u32,
    current_dpb_idx: Option<usize>,
) {
    let mut current_marked = false;

    for op in ops {
        match op {
            MmcoOp::End => break,

            MmcoOp::ShortTermUnused {
                difference_of_pic_nums_minus1,
            } => {
                let pic_num = current_frame_num.wrapping_sub(difference_of_pic_nums_minus1 + 1)
                    % max_frame_num;
                if let Some(idx) = dpb.find_short_term(pic_num) {
                    dpb.mark_unused(idx);
                }
            }

            MmcoOp::LongTermUnused { long_term_pic_num } => {
                if let Some(idx) = dpb.find_long_term(*long_term_pic_num) {
                    dpb.mark_unused(idx);
                }
            }

            MmcoOp::ShortTermToLongTerm {
                difference_of_pic_nums_minus1,
                long_term_frame_idx,
            } => {
                let pic_num = current_frame_num.wrapping_sub(difference_of_pic_nums_minus1 + 1)
                    % max_frame_num;
                if let Some(old_lt) = dpb.find_long_term(*long_term_frame_idx) {
                    dpb.mark_unused(old_lt);
                }
                if let Some(idx) = dpb.find_short_term(pic_num)
                    && let Some(entry) = dpb.get_mut(idx)
                {
                    entry.status = RefStatus::LongTerm;
                    entry.long_term_frame_idx = *long_term_frame_idx;
                }
            }

            MmcoOp::MaxLongTermFrameIdx {
                max_long_term_frame_idx_plus1,
            } => {
                let max_idx = *max_long_term_frame_idx_plus1;
                let to_remove: Vec<usize> = (0..dpb.entries.len())
                    .filter(|&i| {
                        if let Some(entry) = dpb.get(i) {
                            entry.status == RefStatus::LongTerm
                                && (max_idx == 0 || entry.long_term_frame_idx >= max_idx)
                        } else {
                            false
                        }
                    })
                    .collect();
                for i in to_remove {
                    dpb.mark_unused(i);
                }
            }

            MmcoOp::Reset => {
                // Clear all entries except the current one
                for (i, entry) in dpb.entries.iter_mut().enumerate() {
                    if Some(i) == current_dpb_idx {
                        continue;
                    }
                    if let Some(e) = entry {
                        if e.needs_output {
                            e.status = RefStatus::Unused;
                        } else {
                            *entry = None;
                        }
                    }
                }
                current_marked = true;
                if let Some(idx) = current_dpb_idx
                    && let Some(entry) = dpb.get_mut(idx)
                {
                    entry.status = RefStatus::ShortTerm;
                    entry.frame_num = 0;
                }
            }

            MmcoOp::CurrentToLongTerm {
                long_term_frame_idx,
            } => {
                if let Some(old_lt) = dpb.find_long_term(*long_term_frame_idx) {
                    dpb.mark_unused(old_lt);
                }
                if let Some(idx) = current_dpb_idx
                    && let Some(entry) = dpb.get_mut(idx)
                {
                    entry.status = RefStatus::LongTerm;
                    entry.long_term_frame_idx = *long_term_frame_idx;
                }
                current_marked = true;
            }
        }
    }

    if !current_marked
        && let Some(idx) = current_dpb_idx
        && let Some(entry) = dpb.get_mut(idx)
        && entry.status == RefStatus::Unused
    {
        entry.status = RefStatus::ShortTerm;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deblock::PictureBuffer;
    use crate::dpb::DpbEntry;
    use crate::slice::SliceType;

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
            mb_intra: vec![false; 1],
            needs_output: false,
        }
    }

    fn default_slice_header() -> SliceHeader {
        SliceHeader {
            slice_type: SliceType::P,
            num_ref_idx_l0_active: 4,
            ..Default::default()
        }
    }

    #[test]
    fn test_build_ref_list_p_sorted_by_frame_num_desc() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(3, 6, RefStatus::ShortTerm));
        dpb.store(make_entry(7, 14, RefStatus::ShortTerm));
        dpb.store(make_entry(5, 10, RefStatus::ShortTerm));

        let hdr = default_slice_header();
        let list = build_ref_list_p(&dpb, &hdr, 8, 256);

        assert_eq!(list.len(), 3);
        let frame_nums: Vec<u32> = list
            .iter()
            .map(|&idx| dpb.get(idx).unwrap().frame_num)
            .collect();
        assert_eq!(frame_nums, vec![7, 5, 3]);
    }

    #[test]
    fn test_build_ref_list_p_long_term_appended() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(5, 10, RefStatus::ShortTerm));

        let mut lt_entry = make_entry(0, 0, RefStatus::LongTerm);
        lt_entry.long_term_frame_idx = 1;
        dpb.store(lt_entry);

        let hdr = default_slice_header();
        let list = build_ref_list_p(&dpb, &hdr, 8, 256);

        assert_eq!(list.len(), 2);
        assert_eq!(dpb.get(list[0]).unwrap().status, RefStatus::ShortTerm);
        assert_eq!(dpb.get(list[1]).unwrap().status, RefStatus::LongTerm);
    }

    #[test]
    fn test_build_ref_list_p_truncated_to_active_count() {
        let mut dpb = Dpb::new(8);
        for i in 0..6 {
            dpb.store(make_entry(i, i as i32 * 2, RefStatus::ShortTerm));
        }

        let mut hdr = default_slice_header();
        hdr.num_ref_idx_l0_active = 3;
        let list = build_ref_list_p(&dpb, &hdr, 10, 256);

        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_build_ref_list_p_empty_dpb() {
        let dpb = Dpb::new(4);
        let hdr = default_slice_header();
        let list = build_ref_list_p(&dpb, &hdr, 0, 16);
        assert!(list.is_empty());
    }

    /// Test that frame_num wrap-around is handled correctly.
    ///
    /// Scenario: max_frame_num=16, CurrFrameNum=1 (frame 17 in decode order,
    /// which has H.264 frame_num=1 after wrap). The DPB contains:
    ///   - frame_num=0 (frame 16, just decoded before wrap — most recent)
    ///   - frame_num=15 (frame 15, older)
    ///   - frame_num=14 (frame 14, oldest)
    ///
    /// Without wrap-around, raw u32 sort gives [15, 14, 0] (wrong).
    /// With wrap-around: pic_num(0)=0, pic_num(15)=15-16=-1, pic_num(14)=-2
    /// → sorted desc: [0, -1, -2] → frame_nums [0, 15, 14] (correct).
    #[test]
    fn test_build_ref_list_p_frame_num_wraparound() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(0, 0, RefStatus::ShortTerm)); // most recent (after wrap)
        dpb.store(make_entry(15, 30, RefStatus::ShortTerm)); // one before wrap
        dpb.store(make_entry(14, 28, RefStatus::ShortTerm)); // two before wrap

        let hdr = default_slice_header();
        // CurrFrameNum=1 (wrapped), MaxFrameNum=16
        let list = build_ref_list_p(&dpb, &hdr, 1, 16);

        assert_eq!(list.len(), 3);
        let frame_nums: Vec<u32> = list
            .iter()
            .map(|&idx| dpb.get(idx).unwrap().frame_num)
            .collect();
        // frame_num=0 (pic_num=0) must be first (most recent)
        assert_eq!(frame_nums, vec![0, 15, 14]);
    }

    #[test]
    fn test_sliding_window_removes_oldest() {
        let mut dpb = Dpb::new(8);
        dpb.store(make_entry(2, 4, RefStatus::ShortTerm)).unwrap();
        dpb.store(make_entry(5, 10, RefStatus::ShortTerm)).unwrap();
        dpb.store(make_entry(8, 16, RefStatus::ShortTerm)).unwrap();
        dpb.store(make_entry(10, 20, RefStatus::ShortTerm)).unwrap();

        let hdr = SliceHeader {
            adaptive_ref_pic_marking: false,
            ..Default::default()
        };

        let new_idx = dpb.store(make_entry(12, 24, RefStatus::Unused)).unwrap();
        mark_reference(&mut dpb, &hdr, false, 12, 256, 4, Some(new_idx));

        assert!(dpb.find_short_term(2).is_none());
        assert!(dpb.find_short_term(12).is_some());
    }

    #[test]
    fn test_idr_marking() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(0, 0, RefStatus::ShortTerm));
        dpb.store(make_entry(1, 2, RefStatus::ShortTerm));

        // Current picture needs_output=true so it survives dpb.clear()
        let mut cur = make_entry(0, 0, RefStatus::Unused);
        cur.needs_output = true;
        let new_idx = dpb.store(cur).unwrap();

        let hdr = SliceHeader {
            long_term_reference_flag: false,
            ..Default::default()
        };

        mark_reference(&mut dpb, &hdr, true, 0, 256, 4, Some(new_idx));

        assert_eq!(dpb.num_refs(), 1);
        assert!(dpb.find_short_term(0).is_some());
    }

    #[test]
    fn test_idr_long_term_flag() {
        let mut dpb = Dpb::new(4);
        // Current picture needs_output=true so it survives dpb.clear()
        let mut cur = make_entry(0, 0, RefStatus::Unused);
        cur.needs_output = true;
        let new_idx = dpb.store(cur).unwrap();

        let hdr = SliceHeader {
            long_term_reference_flag: true,
            ..Default::default()
        };

        mark_reference(&mut dpb, &hdr, true, 0, 256, 4, Some(new_idx));

        let entry = dpb.get(new_idx).unwrap();
        assert_eq!(entry.status, RefStatus::LongTerm);
        assert_eq!(entry.long_term_frame_idx, 0);
    }

    #[test]
    fn test_mmco_short_term_unused() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(3, 6, RefStatus::ShortTerm));
        dpb.store(make_entry(5, 10, RefStatus::ShortTerm));

        let ops = vec![
            MmcoOp::ShortTermUnused {
                difference_of_pic_nums_minus1: 1, // pic_num = 7 - 2 = 5
            },
            MmcoOp::End,
        ];

        apply_mmco(&mut dpb, &ops, 7, 256, None);

        assert!(dpb.find_short_term(5).is_none());
        assert!(dpb.find_short_term(3).is_some());
    }

    #[test]
    fn test_mmco_current_to_long_term() {
        let mut dpb = Dpb::new(4);
        let idx = dpb.store(make_entry(5, 10, RefStatus::Unused)).unwrap();

        let ops = vec![
            MmcoOp::CurrentToLongTerm {
                long_term_frame_idx: 3,
            },
            MmcoOp::End,
        ];

        apply_mmco(&mut dpb, &ops, 5, 256, Some(idx));

        let entry = dpb.get(idx).unwrap();
        assert_eq!(entry.status, RefStatus::LongTerm);
        assert_eq!(entry.long_term_frame_idx, 3);
    }

    #[test]
    fn test_mmco_short_to_long_term() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(3, 6, RefStatus::ShortTerm));

        let ops = vec![
            MmcoOp::ShortTermToLongTerm {
                difference_of_pic_nums_minus1: 1,
                long_term_frame_idx: 2,
            },
            MmcoOp::End,
        ];

        apply_mmco(&mut dpb, &ops, 5, 256, None);

        assert!(dpb.find_short_term(3).is_none());
        let lt_idx = dpb.find_long_term(2).unwrap();
        assert_eq!(dpb.get(lt_idx).unwrap().frame_num, 3);
    }

    #[test]
    fn test_mmco_reset() {
        let mut dpb = Dpb::new(4);
        dpb.store(make_entry(1, 2, RefStatus::ShortTerm));
        dpb.store(make_entry(2, 4, RefStatus::ShortTerm));
        // Current picture needs_output=true so it survives dpb.clear()
        let mut cur = make_entry(3, 6, RefStatus::Unused);
        cur.needs_output = true;
        let cur_idx = dpb.store(cur).unwrap();

        let ops = vec![MmcoOp::Reset, MmcoOp::End];
        apply_mmco(&mut dpb, &ops, 3, 256, Some(cur_idx));

        assert!(dpb.find_short_term(1).is_none());
        assert!(dpb.find_short_term(2).is_none());
        let entry = dpb.get(cur_idx).unwrap();
        assert_eq!(entry.status, RefStatus::ShortTerm);
        assert_eq!(entry.frame_num, 0);
    }

    #[test]
    fn test_mmco_max_long_term_idx() {
        let mut dpb = Dpb::new(8);

        let mut e0 = make_entry(0, 0, RefStatus::LongTerm);
        e0.long_term_frame_idx = 0;
        dpb.store(e0);

        let mut e1 = make_entry(1, 2, RefStatus::LongTerm);
        e1.long_term_frame_idx = 1;
        dpb.store(e1);

        let mut e2 = make_entry(2, 4, RefStatus::LongTerm);
        e2.long_term_frame_idx = 3;
        dpb.store(e2);

        let ops = vec![
            MmcoOp::MaxLongTermFrameIdx {
                max_long_term_frame_idx_plus1: 2,
            },
            MmcoOp::End,
        ];

        apply_mmco(&mut dpb, &ops, 5, 256, None);

        assert!(dpb.find_long_term(0).is_some());
        assert!(dpb.find_long_term(1).is_some());
        assert!(dpb.find_long_term(3).is_none());
    }

    #[test]
    fn test_mmco_max_long_term_idx_zero_removes_all() {
        let mut dpb = Dpb::new(4);

        let mut e0 = make_entry(0, 0, RefStatus::LongTerm);
        e0.long_term_frame_idx = 0;
        dpb.store(e0);

        let ops = vec![
            MmcoOp::MaxLongTermFrameIdx {
                max_long_term_frame_idx_plus1: 0,
            },
            MmcoOp::End,
        ];

        apply_mmco(&mut dpb, &ops, 5, 256, None);

        assert_eq!(dpb.num_long_term(), 0);
    }
}
