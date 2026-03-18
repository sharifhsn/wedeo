// H.264/AVC motion vector prediction.
//
// In H.264, motion vectors are differentially coded. The predicted MV is
// computed from neighboring blocks (A=left, B=top, C=top-right or top-left).
//
// Reference: ITU-T H.264 Section 8.4.1, FFmpeg libavcodec/h264_mvpred.h

// ---------------------------------------------------------------------------
// Median helper
// ---------------------------------------------------------------------------

/// Compute the median of three values.
///
/// Equivalent to FFmpeg's `mid_pred`: returns the value that is neither the
/// minimum nor the maximum.
#[inline(always)]
fn median(a: i16, b: i16, c: i16) -> i16 {
    a + b + c - a.min(b).min(c) - a.max(b).max(c)
}

// ---------------------------------------------------------------------------
// MV prediction
// ---------------------------------------------------------------------------

/// Compute the predicted motion vector from neighbors.
///
/// `mv_a`, `mv_b`, `mv_c`: motion vectors of neighbors A (left), B (top),
/// C (top-right or top-left).
/// `ref_a`, `ref_b`, `ref_c`: reference indices of neighbors (-1 = unavailable).
/// `ref_idx`: target reference index.
/// `a_avail`, `b_avail`, `c_avail`: whether each neighbor is available.
///
/// Returns the predicted MV `[mvp_x, mvp_y]`.
///
/// Reference: ITU-T H.264 Section 8.4.1.3, FFmpeg `pred_motion`.
#[allow(clippy::too_many_arguments)]
pub fn predict_mv(
    mv_a: [i16; 2],
    mv_b: [i16; 2],
    mv_c: [i16; 2],
    ref_a: i8,
    ref_b: i8,
    ref_c: i8,
    ref_idx: i8,
    a_avail: bool,
    b_avail: bool,
    c_avail: bool,
) -> [i16; 2] {
    // Effective values: unavailable neighbors use MV=[0,0], ref=-1
    let (eff_mv_a, eff_ref_a) = if a_avail { (mv_a, ref_a) } else { ([0, 0], -1) };
    let (eff_mv_b, eff_ref_b) = if b_avail { (mv_b, ref_b) } else { ([0, 0], -1) };
    let (eff_mv_c, eff_ref_c) = if c_avail { (mv_c, ref_c) } else { ([0, 0], -1) };

    // Count how many neighbors match the target reference index.
    let match_count =
        (eff_ref_a == ref_idx) as u8 + (eff_ref_b == ref_idx) as u8 + (eff_ref_c == ref_idx) as u8;

    if match_count > 1 {
        // Most common: multiple matches -> median
        [
            median(eff_mv_a[0], eff_mv_b[0], eff_mv_c[0]),
            median(eff_mv_a[1], eff_mv_b[1], eff_mv_c[1]),
        ]
    } else if match_count == 1 {
        // Exactly one neighbor matches -> use that neighbor's MV
        if eff_ref_a == ref_idx {
            eff_mv_a
        } else if eff_ref_b == ref_idx {
            eff_mv_b
        } else {
            eff_mv_c
        }
    } else {
        // No match: if only A is available (B and C both unavailable), use A.
        // Otherwise, use median.
        if !b_avail && !c_avail && a_avail {
            eff_mv_a
        } else {
            [
                median(eff_mv_a[0], eff_mv_b[0], eff_mv_c[0]),
                median(eff_mv_a[1], eff_mv_b[1], eff_mv_c[1]),
            ]
        }
    }
}

/// Directionally predicted MV for 16x8 partitions.
///
/// For the top partition (n=0), prefer B (top neighbor).
/// For the bottom partition (n=1), prefer A (left neighbor).
/// If the preferred neighbor doesn't match, fall back to general prediction.
///
/// Reference: FFmpeg `pred_16x8_motion`.
#[allow(clippy::too_many_arguments)]
pub fn predict_mv_16x8(
    mv_a: [i16; 2],
    mv_b: [i16; 2],
    mv_c: [i16; 2],
    ref_a: i8,
    ref_b: i8,
    ref_c: i8,
    ref_idx: i8,
    a_avail: bool,
    b_avail: bool,
    c_avail: bool,
    is_top: bool,
) -> [i16; 2] {
    if is_top {
        // Top partition: prefer B
        if b_avail && ref_b == ref_idx {
            return mv_b;
        }
    } else {
        // Bottom partition: prefer A
        if a_avail && ref_a == ref_idx {
            return mv_a;
        }
    }
    // Fallback to general prediction
    predict_mv(
        mv_a, mv_b, mv_c, ref_a, ref_b, ref_c, ref_idx, a_avail, b_avail, c_avail,
    )
}

/// Directionally predicted MV for 8x16 partitions.
///
/// For the left partition (n=0), prefer A (left neighbor).
/// For the right partition (n=1), prefer C (top-right/top-left neighbor).
/// If the preferred neighbor doesn't match, fall back to general prediction.
///
/// Reference: FFmpeg `pred_8x16_motion`.
#[allow(clippy::too_many_arguments)]
pub fn predict_mv_8x16(
    mv_a: [i16; 2],
    mv_b: [i16; 2],
    mv_c: [i16; 2],
    ref_a: i8,
    ref_b: i8,
    ref_c: i8,
    ref_idx: i8,
    a_avail: bool,
    b_avail: bool,
    c_avail: bool,
    is_left: bool,
) -> [i16; 2] {
    if is_left {
        // Left partition: prefer A
        if a_avail && ref_a == ref_idx {
            return mv_a;
        }
    } else {
        // Right partition: prefer C
        if c_avail && ref_c == ref_idx {
            return mv_c;
        }
    }
    // Fallback to general prediction
    predict_mv(
        mv_a, mv_b, mv_c, ref_a, ref_b, ref_c, ref_idx, a_avail, b_avail, c_avail,
    )
}

// ---------------------------------------------------------------------------
// P_SKIP motion vector
// ---------------------------------------------------------------------------

/// Compute the motion vector for a P_SKIP macroblock.
///
/// For P_SKIP: if A is unavailable or (ref_a == 0 && mv_a == [0,0]),
/// and B is unavailable or (ref_b == 0 && mv_b == [0,0]),
/// then MV = [0,0]. Otherwise, use median prediction with ref_idx = 0.
///
/// Reference: ITU-T H.264 Section 8.4.1.1, FFmpeg `pred_pskip_motion`.
pub fn predict_mv_skip(
    mv_a: [i16; 2],
    mv_b: [i16; 2],
    ref_a: i8,
    ref_b: i8,
    a_avail: bool,
    b_avail: bool,
) -> [i16; 2] {
    // If A is not available, MV = [0,0]
    if !a_avail {
        return [0, 0];
    }
    // If B is not available, MV = [0,0]
    if !b_avail {
        return [0, 0];
    }
    // If A is ref 0 with zero MV, and B is ref 0 with zero MV => [0,0]
    let a_zero = ref_a == 0 && mv_a == [0, 0];
    let b_zero = ref_b == 0 && mv_b == [0, 0];
    if a_zero || b_zero {
        return [0, 0];
    }

    // Otherwise, use median prediction with ref_idx=0.
    // For P_SKIP, C (top-right) neighbor is also needed but here we use the
    // simplified two-neighbor form. The caller should provide full neighbor
    // data and use predict_mv() for complete accuracy.
    // This matches the FFmpeg fast-path where both A and B have ref=0 and
    // non-zero MV: the median of (A, B, C) with ref=0.
    [median(mv_a[0], mv_b[0], 0), median(mv_a[1], mv_b[1], 0)]
}

/// Full P_SKIP motion vector prediction with three neighbors.
///
/// This is the complete version that takes the C neighbor (top-right or
/// top-left fallback) into account, matching FFmpeg's pred_pskip_motion.
#[allow(clippy::too_many_arguments)]
pub fn predict_mv_skip_full(
    mv_a: [i16; 2],
    mv_b: [i16; 2],
    mv_c: [i16; 2],
    ref_a: i8,
    ref_b: i8,
    ref_c: i8,
    a_avail: bool,
    b_avail: bool,
    c_avail: bool,
) -> [i16; 2] {
    // If A is not available or B is not available, MV = [0,0]
    if !a_avail || !b_avail {
        return [0, 0];
    }
    // If A or B is (ref=0, mv=[0,0]), MV = [0,0]
    if (ref_a == 0 && mv_a == [0, 0]) || (ref_b == 0 && mv_b == [0, 0]) {
        return [0, 0];
    }

    // Use general prediction with ref_idx=0
    predict_mv(
        mv_a, mv_b, mv_c, ref_a, ref_b, ref_c, 0, a_avail, b_avail, c_avail,
    )
}

// ---------------------------------------------------------------------------
// MvContext — neighbor lookup
// ---------------------------------------------------------------------------

/// Context for looking up neighbor MVs and reference indices.
///
/// Stores motion vectors and reference indices per 4x4 block for the current
/// frame, laid out as `[mb_addr * 16 + blk_idx]` where `blk_idx` is the
/// raster-scan index of the 4x4 block within a macroblock (0..16).
pub struct MvContext {
    /// Motion vectors per 4x4 block for the current frame.
    pub mv: Vec<[i16; 2]>,
    /// Reference indices per 4x4 block.
    pub ref_idx: Vec<i8>,
    /// MB width in the current frame.
    pub mb_width: u32,
    /// MB height in the current frame.
    pub mb_height: u32,
}

/// Map a 4x4 block index (0..16, raster scan of 4x4 sub-blocks within MB)
/// to (blk_x, blk_y) within the MB, where each unit is one 4x4 block.
///
/// The layout matches H.264's raster scan order within a macroblock:
///   0  1  2  3
///   4  5  6  7
///   8  9 10 11
///  12 13 14 15
#[inline(always)]
pub fn blk_idx_to_xy(blk_idx: usize) -> (u32, u32) {
    let blk_x = (blk_idx % 4) as u32;
    let blk_y = (blk_idx / 4) as u32;
    (blk_x, blk_y)
}

impl MvContext {
    /// Create a new MvContext for a frame of the given MB dimensions.
    pub fn new(mb_width: u32, mb_height: u32) -> Self {
        let total_blocks = (mb_width * mb_height * 16) as usize;
        Self {
            mv: vec![[0i16; 2]; total_blocks],
            ref_idx: vec![-1i8; total_blocks],
            mb_width,
            mb_height,
        }
    }

    /// Linear index for a 4x4 block at (mb_x, mb_y, blk_idx).
    #[inline(always)]
    fn linear_idx(&self, mb_x: u32, mb_y: u32, blk_idx: usize) -> usize {
        let mb_addr = (mb_y * self.mb_width + mb_x) as usize;
        mb_addr * 16 + blk_idx
    }

    /// Get the MV and ref_idx for a 4x4 block at (mb_x, mb_y, blk_idx).
    pub fn get(&self, mb_x: u32, mb_y: u32, blk_idx: usize) -> ([i16; 2], i8) {
        let idx = self.linear_idx(mb_x, mb_y, blk_idx);
        (self.mv[idx], self.ref_idx[idx])
    }

    /// Set the MV and ref_idx for a 4x4 block.
    pub fn set(&mut self, mb_x: u32, mb_y: u32, blk_idx: usize, mv: [i16; 2], ref_idx: i8) {
        let idx = self.linear_idx(mb_x, mb_y, blk_idx);
        self.mv[idx] = mv;
        self.ref_idx[idx] = ref_idx;
    }

    /// Get neighbor A (left) for a block at position (blk_x, blk_y) within
    /// MB (mb_x, mb_y).
    ///
    /// Returns `None` if the neighbor is outside the picture.
    pub fn neighbor_a(
        &self,
        mb_x: u32,
        mb_y: u32,
        blk_x: u32,
        blk_y: u32,
    ) -> Option<([i16; 2], i8)> {
        if blk_x > 0 {
            // Within the same MB
            let blk_idx = ((blk_x - 1) + blk_y * 4) as usize;
            Some(self.get(mb_x, mb_y, blk_idx))
        } else if mb_x > 0 {
            // Left MB, rightmost column (blk_x=3)
            let blk_idx = (3 + blk_y * 4) as usize;
            Some(self.get(mb_x - 1, mb_y, blk_idx))
        } else {
            None
        }
    }

    /// Get neighbor B (top) for a block at position (blk_x, blk_y) within
    /// MB (mb_x, mb_y).
    ///
    /// Returns `None` if the neighbor is outside the picture.
    pub fn neighbor_b(
        &self,
        mb_x: u32,
        mb_y: u32,
        blk_x: u32,
        blk_y: u32,
    ) -> Option<([i16; 2], i8)> {
        if blk_y > 0 {
            // Within the same MB
            let blk_idx = (blk_x + (blk_y - 1) * 4) as usize;
            Some(self.get(mb_x, mb_y, blk_idx))
        } else if mb_y > 0 {
            // Above MB, bottom row (blk_y=3)
            let blk_idx = (blk_x + 3 * 4) as usize;
            Some(self.get(mb_x, mb_y - 1, blk_idx))
        } else {
            None
        }
    }

    /// Get neighbor C (top-right, falling back to top-left D).
    ///
    /// C is the block at (blk_x + part_width, blk_y - 1). C is only
    /// available if the block has already been decoded, i.e. it belongs to
    /// a macroblock in a previous row, or the same row to the left/same MB.
    /// Same-row MBs to the right (mb_x + 1, mb_y, …) are NOT yet decoded
    /// and must be treated as PART_NOT_AVAILABLE → fall back to D.
    ///
    /// `part_width` is the width of the partition in 4x4 block units
    /// (1 for 4-wide, 2 for 8-wide, 4 for 16-wide).
    ///
    /// Returns `None` if neither C nor D is available.
    ///
    /// Reference: ITU-T H.264 8.4.1.2.1, FFmpeg fill_decode_caches
    pub fn neighbor_c(
        &self,
        mb_x: u32,
        mb_y: u32,
        blk_x: u32,
        blk_y: u32,
        part_width: u32,
    ) -> Option<([i16; 2], i8)> {
        // Try top-right (C)
        let cr_x = blk_x + part_width;
        let cr_y = blk_y.wrapping_sub(1); // u32::MAX when blk_y==0 → wraps to -1 in i32

        // Compute absolute block coordinates to determine whether C is in a
        // decoded (past) macroblock.  cr_y wraps to u32::MAX when blk_y=0,
        // which becomes -1 in i32 arithmetic → targets the row above (OK).
        let abs_cr_x = mb_x as i32 * 4 + cr_x as i32;
        let abs_cr_y = mb_y as i32 * 4 + cr_y as i32;

        // C is in a "future" (not-yet-decoded) macroblock when its row is the
        // same as the current MB and it is to the right (target_mb_x > mb_x),
        // or when it is in a future row.  In those cases fall through to D.
        let (c_is_past, c_is_same_mb) = if abs_cr_x >= 0 && abs_cr_y >= 0 {
            let target_mb_x = (abs_cr_x as u32) / 4;
            let target_mb_y = (abs_cr_y as u32) / 4;
            let past = target_mb_y < mb_y || (target_mb_y == mb_y && target_mb_x <= mb_x);
            let same = target_mb_y == mb_y && target_mb_x == mb_x;
            (past, same)
        } else {
            // Negative coordinates → out of picture → try_get_neighbor will return None.
            // The bounds check is handled there; mark as "past" so we attempt the lookup.
            (true, false)
        };

        if c_is_past
            && let Some(result) = self.try_get_neighbor(mb_x, mb_y, cr_x, cr_y)
        {
            // If C is within the current MB and ref_idx=-1, the target 4x4 block
            // has not been decoded yet (e.g., sub 3 when processing sub 2 of a
            // P_8x8 MB).  FFmpeg marks these positions PART_NOT_AVAILABLE and
            // falls back to D; wedeo must do the same.  ref_idx=-1 from a
            // *different* MB means intra-coded (LIST_NOT_USED) and is valid.
            if !c_is_same_mb || result.1 >= 0 {
                return Some(result);
            }
            // Same MB + ref=-1 → fall through to D
        }

        // Fall back to top-left (D)
        let dl_x = blk_x.wrapping_sub(1);
        let dl_y = blk_y.wrapping_sub(1);
        self.try_get_neighbor(mb_x, mb_y, dl_x, dl_y)
    }

    /// Simplified neighbor C lookup for 16x16 partitions.
    ///
    /// For a 16x16 partition, the top-right neighbor is the bottom-left 4x4
    /// block of the MB above-right. If that MB doesn't exist, falls back
    /// to the bottom-right 4x4 block of the MB above-left.
    pub fn neighbor_c_16x16(&self, mb_x: u32, mb_y: u32) -> Option<([i16; 2], i8)> {
        self.neighbor_c(mb_x, mb_y, 0, 0, 4)
    }

    /// Try to get a 4x4 block at absolute position (blk_x, blk_y) relative
    /// to MB (mb_x, mb_y), handling cross-MB boundaries.
    ///
    /// blk_x and blk_y may be negative (via wrapping) or >= 4.
    fn try_get_neighbor(
        &self,
        mb_x: u32,
        mb_y: u32,
        blk_x: u32,
        blk_y: u32,
    ) -> Option<([i16; 2], i8)> {
        // Use signed arithmetic to handle wrapping (negative) block coordinates
        let abs_blk_x = mb_x as i32 * 4 + blk_x as i32;
        let abs_blk_y = mb_y as i32 * 4 + blk_y as i32;

        // Check bounds (wrapping u32 values become large positive i32 which is > max)
        if abs_blk_x < 0
            || abs_blk_y < 0
            || abs_blk_x >= self.mb_width as i32 * 4
            || abs_blk_y >= self.mb_height as i32 * 4
        {
            return None;
        }
        let abs_blk_x = abs_blk_x as u32;
        let abs_blk_y = abs_blk_y as u32;

        let target_mb_x = abs_blk_x / 4;
        let target_mb_y = abs_blk_y / 4;
        let target_blk_x = abs_blk_x % 4;
        let target_blk_y = abs_blk_y % 4;
        let blk_idx = (target_blk_x + target_blk_y * 4) as usize;

        Some(self.get(target_mb_x, target_mb_y, blk_idx))
    }

    /// Get neighbors A, B, C for a partition starting at 4x4 block (blk_x, blk_y)
    /// within MB (mb_x, mb_y), with partition width `part_width` in 4x4 units.
    ///
    /// Returns (mv_a, ref_a, a_avail, mv_b, ref_b, b_avail, mv_c, ref_c, c_avail).
    pub fn get_neighbors(
        &self,
        mb_x: u32,
        mb_y: u32,
        blk_x: u32,
        blk_y: u32,
        part_width: u32,
    ) -> NeighborInfo {
        self.get_neighbors_slice(mb_x, mb_y, blk_x, blk_y, part_width, None, 0)
    }

    /// Like `get_neighbors` but with slice-boundary awareness.
    /// If `slice_table` is Some, neighbors from different slices are unavailable.
    #[allow(clippy::too_many_arguments)] // slice-awareness needs both table + current_slice
    pub fn get_neighbors_slice(
        &self,
        mb_x: u32,
        mb_y: u32,
        blk_x: u32,
        blk_y: u32,
        part_width: u32,
        slice_table: Option<&[u16]>,
        current_slice: u16,
    ) -> NeighborInfo {
        // Helper: check if a neighbor MB is in the same slice
        let same_slice = |nb_mb_x: u32, nb_mb_y: u32| -> bool {
            match slice_table {
                None => true,
                Some(st) => {
                    let idx = (nb_mb_y * self.mb_width + nb_mb_x) as usize;
                    idx < st.len() && st[idx] == current_slice
                }
            }
        };

        // Neighbor A (left)
        let (mv_a, ref_a, a_avail) = if blk_x > 0 {
            // Within same MB — always available
            match self.neighbor_a(mb_x, mb_y, blk_x, blk_y) {
                Some((mv, r)) => (mv, r, true),
                None => ([0, 0], -1, false),
            }
        } else if mb_x > 0 && same_slice(mb_x - 1, mb_y) {
            match self.neighbor_a(mb_x, mb_y, blk_x, blk_y) {
                Some((mv, r)) => (mv, r, true),
                None => ([0, 0], -1, false),
            }
        } else {
            ([0, 0], -1, false)
        };

        // Neighbor B (top)
        let (mv_b, ref_b, b_avail) = if blk_y > 0 {
            // Within same MB
            match self.neighbor_b(mb_x, mb_y, blk_x, blk_y) {
                Some((mv, r)) => (mv, r, true),
                None => ([0, 0], -1, false),
            }
        } else if mb_y > 0 && same_slice(mb_x, mb_y - 1) {
            match self.neighbor_b(mb_x, mb_y, blk_x, blk_y) {
                Some((mv, r)) => (mv, r, true),
                None => ([0, 0], -1, false),
            }
        } else {
            ([0, 0], -1, false)
        };

        // Neighbor C (top-right, falling back to D=top-left)
        // The neighbor_c method handles the complex availability logic
        // within a MB. For cross-MB access, check slice boundary.
        let (mv_c, ref_c, c_avail) = if slice_table.is_some() {
            // Determine which MB the C/D result comes from and check slices.
            // C candidate: (blk_x + part_width, blk_y - 1)
            let cr_x = blk_x + part_width;
            let cr_y = blk_y.wrapping_sub(1);
            let abs_cr_x = mb_x as i32 * 4 + cr_x as i32;
            let abs_cr_y = mb_y as i32 * 4 + cr_y as i32;

            // Check C availability: must be in-bounds and in a decoded MB
            let c_mb_ok = if abs_cr_x >= 0
                && abs_cr_y >= 0
                && abs_cr_x < self.mb_width as i32 * 4
                && abs_cr_y < self.mb_height as i32 * 4
            {
                let c_mb_x = abs_cr_x as u32 / 4;
                let c_mb_y = abs_cr_y as u32 / 4;
                let is_same_mb = c_mb_x == mb_x && c_mb_y == mb_y;
                let is_past = c_mb_y < mb_y || (c_mb_y == mb_y && c_mb_x <= mb_x);
                is_same_mb || (is_past && same_slice(c_mb_x, c_mb_y))
            } else {
                false
            };

            if c_mb_ok {
                match self.neighbor_c(mb_x, mb_y, blk_x, blk_y, part_width) {
                    Some((mv, r)) => (mv, r, true),
                    None => ([0, 0], -1, false),
                }
            } else {
                // C not available — try D (top-left fallback)
                let dl_x = blk_x.wrapping_sub(1);
                let dl_y = blk_y.wrapping_sub(1);
                let abs_dl_x = mb_x as i32 * 4 + dl_x as i32;
                let abs_dl_y = mb_y as i32 * 4 + dl_y as i32;

                let d_mb_ok = if abs_dl_x >= 0
                    && abs_dl_y >= 0
                    && abs_dl_x < self.mb_width as i32 * 4
                    && abs_dl_y < self.mb_height as i32 * 4
                {
                    let d_mb_x = abs_dl_x as u32 / 4;
                    let d_mb_y = abs_dl_y as u32 / 4;
                    let is_same_mb = d_mb_x == mb_x && d_mb_y == mb_y;
                    is_same_mb || same_slice(d_mb_x, d_mb_y)
                } else {
                    false
                };

                if d_mb_ok {
                    match self.try_get_neighbor(mb_x, mb_y, dl_x, dl_y) {
                        Some((mv, r)) => (mv, r, true),
                        None => ([0, 0], -1, false),
                    }
                } else {
                    ([0, 0], -1, false)
                }
            }
        } else {
            match self.neighbor_c(mb_x, mb_y, blk_x, blk_y, part_width) {
                Some((mv, r)) => (mv, r, true),
                None => ([0, 0], -1, false),
            }
        };

        NeighborInfo {
            mv_a,
            ref_a,
            a_avail,
            mv_b,
            ref_b,
            b_avail,
            mv_c,
            ref_c,
            c_avail,
        }
    }
}

/// Neighbor information for MV prediction.
#[derive(Debug, Clone, Copy)]
pub struct NeighborInfo {
    pub mv_a: [i16; 2],
    pub ref_a: i8,
    pub a_avail: bool,
    pub mv_b: [i16; 2],
    pub ref_b: i8,
    pub b_avail: bool,
    pub mv_c: [i16; 2],
    pub ref_c: i8,
    pub c_avail: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median() {
        assert_eq!(median(1, 5, 3), 3);
        assert_eq!(median(5, 1, 3), 3);
        assert_eq!(median(3, 5, 1), 3);
        assert_eq!(median(2, 2, 2), 2);
        assert_eq!(median(-5, 0, 10), 0);
        assert_eq!(median(0, 0, 0), 0);
    }

    #[test]
    fn test_median_prediction_basic() {
        // Three neighbors with different refs; ref_idx matches none -> median
        let mv = predict_mv(
            [1, 2], // A
            [5, 6], // B
            [3, 4], // C
            1,
            2,
            3, // ref_a, ref_b, ref_c
            0, // ref_idx (matches none)
            true,
            true,
            true,
        );
        assert_eq!(mv, [3, 4]); // median([1,5,3], [2,6,4]) = [3, 4]
    }

    #[test]
    fn test_single_match_uses_matching_neighbor() {
        // Only A matches ref_idx=1
        let mv = predict_mv(
            [10, 20], // A
            [30, 40], // B
            [50, 60], // C
            1,
            2,
            3, // only ref_a==1 matches
            1, // ref_idx
            true,
            true,
            true,
        );
        assert_eq!(mv, [10, 20]);

        // Only B matches ref_idx=2
        let mv = predict_mv(
            [10, 20], // A
            [30, 40], // B
            [50, 60], // C
            1,
            2,
            3, // only ref_b==2 matches
            2, // ref_idx
            true,
            true,
            true,
        );
        assert_eq!(mv, [30, 40]);

        // Only C matches ref_idx=3
        let mv = predict_mv(
            [10, 20], // A
            [30, 40], // B
            [50, 60], // C
            1,
            2,
            3, // only ref_c==3 matches
            3, // ref_idx
            true,
            true,
            true,
        );
        assert_eq!(mv, [50, 60]);
    }

    #[test]
    fn test_only_a_available() {
        // Only A available, no match -> use A
        let mv = predict_mv([7, 8], [0, 0], [0, 0], 5, -1, -1, 0, true, false, false);
        assert_eq!(mv, [7, 8]);
    }

    #[test]
    fn test_no_neighbors_available() {
        // No neighbors available -> [0, 0]
        let mv = predict_mv([0, 0], [0, 0], [0, 0], -1, -1, -1, 0, false, false, false);
        assert_eq!(mv, [0, 0]);
    }

    #[test]
    fn test_predict_mv_skip_zero_case() {
        // A unavailable -> [0, 0]
        let mv = predict_mv_skip([0, 0], [0, 0], -1, -1, false, false);
        assert_eq!(mv, [0, 0]);

        // B unavailable -> [0, 0]
        let mv = predict_mv_skip([5, 6], [0, 0], 0, -1, true, false);
        assert_eq!(mv, [0, 0]);

        // A is (ref=0, mv=[0,0]) -> [0, 0]
        let mv = predict_mv_skip([0, 0], [3, 4], 0, 0, true, true);
        assert_eq!(mv, [0, 0]);

        // B is (ref=0, mv=[0,0]) -> [0, 0]
        let mv = predict_mv_skip([3, 4], [0, 0], 0, 0, true, true);
        assert_eq!(mv, [0, 0]);
    }

    #[test]
    fn test_predict_mv_skip_nonzero() {
        // Both A and B have non-zero ref or MV -> use median with C=[0,0]
        let mv = predict_mv_skip([4, 6], [2, 8], 0, 0, true, true);
        // median(4,2,0)=2, median(6,8,0)=6
        assert_eq!(mv, [2, 6]);
    }

    #[test]
    fn test_predict_mv_16x8_top_prefers_b() {
        // Top partition: B matches -> use B
        let mv = predict_mv_16x8([1, 1], [5, 5], [3, 3], 0, 0, 0, 0, true, true, true, true);
        assert_eq!(mv, [5, 5]);
    }

    #[test]
    fn test_predict_mv_16x8_bottom_prefers_a() {
        // Bottom partition: A matches -> use A
        let mv = predict_mv_16x8([1, 1], [5, 5], [3, 3], 0, 0, 0, 0, true, true, true, false);
        assert_eq!(mv, [1, 1]);
    }

    #[test]
    fn test_predict_mv_8x16_left_prefers_a() {
        // Left partition: A matches -> use A
        let mv = predict_mv_8x16([1, 1], [5, 5], [3, 3], 0, 0, 0, 0, true, true, true, true);
        assert_eq!(mv, [1, 1]);
    }

    #[test]
    fn test_predict_mv_8x16_right_prefers_c() {
        // Right partition: C matches -> use C
        let mv = predict_mv_8x16([1, 1], [5, 5], [3, 3], 0, 0, 0, 0, true, true, true, false);
        assert_eq!(mv, [3, 3]);
    }

    #[test]
    fn test_mv_context_basic() {
        let mut ctx = MvContext::new(2, 2);

        // Set MV for MB (0,0), block 0
        ctx.set(0, 0, 0, [10, 20], 0);
        let (mv, r) = ctx.get(0, 0, 0);
        assert_eq!(mv, [10, 20]);
        assert_eq!(r, 0);

        // Default values
        let (mv, r) = ctx.get(1, 1, 15);
        assert_eq!(mv, [0, 0]);
        assert_eq!(r, -1);
    }

    #[test]
    fn test_neighbor_a_within_mb() {
        let mut ctx = MvContext::new(2, 2);
        ctx.set(0, 0, 0, [5, 6], 1);

        // Block 1 (blk_x=1, blk_y=0) should find block 0 (blk_x=0) as left neighbor
        let result = ctx.neighbor_a(0, 0, 1, 0);
        assert_eq!(result, Some(([5, 6], 1)));
    }

    #[test]
    fn test_neighbor_a_cross_mb() {
        let mut ctx = MvContext::new(2, 2);
        // Set right column of MB (0,0): blocks 3, 7, 11, 15
        ctx.set(0, 0, 3, [10, 11], 0);

        // Block 0 (blk_x=0, blk_y=0) of MB (1,0) should find block 3 of MB (0,0)
        let result = ctx.neighbor_a(1, 0, 0, 0);
        assert_eq!(result, Some(([10, 11], 0)));
    }

    #[test]
    fn test_neighbor_a_left_edge() {
        let ctx = MvContext::new(2, 2);
        // Block at blk_x=0 in MB (0,0) has no left neighbor
        let result = ctx.neighbor_a(0, 0, 0, 0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_neighbor_b_within_mb() {
        let mut ctx = MvContext::new(2, 2);
        ctx.set(0, 0, 0, [5, 6], 1);

        // Block 4 (blk_x=0, blk_y=1) should find block 0 (blk_y=0) as top neighbor
        let result = ctx.neighbor_b(0, 0, 0, 1);
        assert_eq!(result, Some(([5, 6], 1)));
    }

    #[test]
    fn test_neighbor_b_cross_mb() {
        let mut ctx = MvContext::new(2, 2);
        // Set bottom row of MB (0,0): blocks 12..15
        ctx.set(0, 0, 12, [7, 8], 2);

        // Block 0 (blk_y=0) of MB (0,1) should find block 12 of MB (0,0)
        let result = ctx.neighbor_b(0, 1, 0, 0);
        assert_eq!(result, Some(([7, 8], 2)));
    }

    #[test]
    fn test_neighbor_b_top_edge() {
        let ctx = MvContext::new(2, 2);
        // Block at blk_y=0 in MB (0,0) has no top neighbor
        let result = ctx.neighbor_b(0, 0, 0, 0);
        assert_eq!(result, None);
    }

    #[test]
    fn test_blk_idx_to_xy() {
        assert_eq!(blk_idx_to_xy(0), (0, 0));
        assert_eq!(blk_idx_to_xy(1), (1, 0));
        assert_eq!(blk_idx_to_xy(4), (0, 1));
        assert_eq!(blk_idx_to_xy(5), (1, 1));
        assert_eq!(blk_idx_to_xy(15), (3, 3));
    }

    #[test]
    fn test_get_neighbors() {
        let mut ctx = MvContext::new(3, 3);
        // MB (1,1), set some neighbors
        // A = left: MB (0,1), block (3,0) = blk_idx 3
        ctx.set(0, 1, 3, [1, 2], 0);
        // B = top: MB (1,0), block (0,3) = blk_idx 12
        ctx.set(1, 0, 12, [3, 4], 1);
        // C (top-right for 16x16 at MB 1,1): MB (2,0), block (0,3) = blk_idx 12
        ctx.set(2, 0, 12, [5, 6], 2);

        let n = ctx.get_neighbors(1, 1, 0, 0, 4);
        assert!(n.a_avail);
        assert_eq!(n.mv_a, [1, 2]);
        assert_eq!(n.ref_a, 0);
        assert!(n.b_avail);
        assert_eq!(n.mv_b, [3, 4]);
        assert_eq!(n.ref_b, 1);
        assert!(n.c_avail);
        assert_eq!(n.mv_c, [5, 6]);
        assert_eq!(n.ref_c, 2);
    }
}
