// VP9 above-row and left-column context arrays.
//
// Translated from FFmpeg's libavcodec/vp9dec.h (VP9Context above_* arrays)
// and VP9TileData left_* arrays.
//
// These arrays carry neighbour state from previously decoded blocks to allow
// entropy-coding context derivation.  Above arrays are per-4×4-column and span
// the whole frame width; left arrays are local to one superblock row (16
// entries = 64 pixels / 4).

// ---------------------------------------------------------------------------
// Above-row context
// ---------------------------------------------------------------------------

/// Above-row context arrays, one element per 4×4-pixel column in the frame.
///
/// Corresponds to the `above_*` arrays in VP9Context (vp9dec.h).
pub struct AboveContext {
    /// Partition context per 8×8 column (above_partition_ctx, cols entries).
    pub partition: Vec<u8>,
    /// Skip flag per 4×4 column (above_skip_ctx).
    pub skip: Vec<u8>,
    /// Transform size per 4×4 column (above_txfm_ctx).
    pub tx_size: Vec<u8>,
    /// Intra-mode per 4×4 column — 2 entries per 4×4 for sub-block modes
    /// (above_mode_ctx, cols*2 entries).
    pub y_mode: Vec<u8>,
    /// UV intra-mode per 4×4 column (above_mode_ctx is shared for UV
    /// in practice; stored separately here for clarity).
    pub uv_mode: Vec<u8>,
    /// Segment-ID prediction context (above_segpred_ctx).
    pub seg_id: Vec<u8>,
    /// Non-zero coefficient context per plane (above_y_nnz_ctx and
    /// above_uv_nnz_ctx[2]).  Index 0 = luma, 1 = Cb, 2 = Cr.
    /// Luma has `sb_cols * 16` entries; each chroma plane has
    /// `sb_cols * 16 >> ss_h` entries.  For simplicity we store all three
    /// at the luma width and let the caller index appropriately.
    pub coef: [Vec<u8>; 3],
    // --- Inter-prediction context arrays ---
    /// Intra flag per 4×4 column (above_intra_ctx). 1 = intra, 0 = inter.
    pub intra: Vec<u8>,
    /// Compound prediction flag per 4×4 column (above_comp_ctx).
    pub comp: Vec<u8>,
    /// Primary reference frame per 4×4 column (above_ref_ctx).
    pub ref_frame: Vec<u8>,
    /// Interpolation filter per 4×4 column (above_filter_ctx).
    pub filter: Vec<u8>,
    /// Motion vectors per 4×4 column. 2 entries per column for sub-block
    /// modes (above_mv_ctx, cols*2 entries). Each entry holds MVs for
    /// both reference frames: `[ref_idx] = (x, y)`.
    pub mv: Vec<[[i16; 2]; 2]>,
}

impl AboveContext {
    /// Allocate above-context arrays for a frame with `cols_4x4` 4×4 columns
    /// (= `ceil(width / 4)`).
    ///
    /// `sb_cols` = number of 64×64 superblock columns = `ceil(cols_4x4 / 16)`.
    pub fn new(cols_4x4: usize, sb_cols: usize) -> Self {
        // above_mode_ctx in FFmpeg is cols*2 bytes (2 mode entries per 4×4
        // column to support sub-4×4-block modes such as BS_8x4 / BS_4x8).
        let mode_len = cols_4x4 * 2;
        // NNZ arrays: luma is sb_cols*16; chroma is sb_cols*16 >> ss_h.
        // We allocate luma-sized for all three and the caller handles ss_h.
        let nnz_len = sb_cols * 16;
        Self {
            partition: vec![0u8; cols_4x4],
            skip: vec![0u8; cols_4x4],
            tx_size: vec![0u8; cols_4x4],
            y_mode: vec![0u8; mode_len],
            uv_mode: vec![0u8; mode_len],
            seg_id: vec![0u8; cols_4x4],
            coef: [vec![0u8; nnz_len], vec![0u8; nnz_len], vec![0u8; nnz_len]],
            intra: vec![0u8; cols_4x4],
            comp: vec![0u8; cols_4x4],
            ref_frame: vec![0u8; cols_4x4],
            filter: vec![0u8; cols_4x4],
            mv: vec![[[0i16; 2]; 2]; mode_len],
        }
    }

    /// Reset above context at the start of a new tile / frame decode.
    ///
    /// For keyframes, intra_only: mode context is reset to DC_PRED (= 2).
    /// For inter frames: y_mode context is reset to NEARESTMV (= 10).
    pub fn reset(&mut self, keyframe_or_intraonly: bool) {
        self.partition.fill(0);
        self.skip.fill(0);
        self.seg_id.fill(0);
        if keyframe_or_intraonly {
            // FFmpeg: memset(s->above_mode_ctx, DC_PRED, cols*2)
            self.y_mode.fill(2);
        } else {
            // FFmpeg: memset(s->above_mode_ctx, NEARESTMV, cols)
            // Only first `cols` entries, not cols*2
            let cols = self.partition.len();
            self.y_mode[..cols].fill(10);
        }
        for plane in &mut self.coef {
            plane.fill(0);
        }
        // Do NOT reset: tx_size, intra, comp, ref_frame, filter, mv, uv_mode
        // FFmpeg intentionally preserves these across frame resets
        // (vp9.c:1693–1703 only resets the fields above)
    }
}

// ---------------------------------------------------------------------------
// Left-column context
// ---------------------------------------------------------------------------

/// Left-column context for one superblock row (16 × 4 pixels = 64 pixels tall).
///
/// Corresponds to the `left_*` arrays in VP9TileData (vp9dec.h).
/// Index 0 is the top row of 4×4 blocks in the current SB row.
pub struct LeftContext {
    /// Partition context per 8×8 row (left_partition_ctx, 8 entries).
    pub partition: [u8; 8],
    /// Skip flag per 4×4 row (left_skip_ctx, 8 entries — indexed by row7).
    pub skip: [u8; 8],
    /// Transform size per 4×4 row (left_txfm_ctx, 8 entries).
    pub tx_size: [u8; 8],
    /// Intra-mode per 4×4 row — 16 entries, 2 per 4×4 block for sub-block
    /// modes (left_mode_ctx in FFmpeg).
    pub y_mode: [u8; 16],
    /// UV intra-mode per 4×4 row (16 entries).
    pub uv_mode: [u8; 16],
    /// Segment-ID prediction context per 4×4 row (left_segpred_ctx, 8 entries).
    pub seg_id: [u8; 8],
    /// Non-zero coefficient context per plane (left_y_nnz_ctx,
    /// left_uv_nnz_ctx[2]).  Index 0 = luma (16 entries), 1–2 = chroma.
    pub coef: [[u8; 16]; 3],
    // --- Inter-prediction context arrays ---
    /// Intra flag per 4×4 row (left_intra_ctx, 8 entries).
    pub intra: [u8; 8],
    /// Compound prediction flag per 4×4 row (left_comp_ctx, 8 entries).
    pub comp: [u8; 8],
    /// Primary reference frame per 4×4 row (left_ref_ctx, 8 entries).
    pub ref_frame: [u8; 8],
    /// Interpolation filter per 4×4 row (left_filter_ctx, 8 entries).
    pub filter: [u8; 8],
    /// Motion vectors per 4×4 row. 16 entries (2 per 4×4 row for sub-block modes).
    /// Each entry holds MVs for both reference frames: `[ref_idx] = (x, y)`.
    pub mv: [[[i16; 2]; 2]; 16],
}

impl LeftContext {
    /// Create a zeroed left context.
    pub const fn new() -> Self {
        Self {
            partition: [0u8; 8],
            skip: [0u8; 8],
            tx_size: [0u8; 8],
            y_mode: [0u8; 16],
            uv_mode: [0u8; 16],
            seg_id: [0u8; 8],
            coef: [[0u8; 16]; 3],
            intra: [0u8; 8],
            comp: [0u8; 8],
            ref_frame: [0u8; 8],
            filter: [0u8; 8],
            mv: [[[0i16; 2]; 2]; 16],
        }
    }

    /// Reset all left context at the start of a new superblock row.
    ///
    /// Mirrors the `memset` calls in the SB-row loop in vp9.c.
    /// `keyframe_or_intraonly` should be true for keyframes and intra-only frames.
    pub fn reset(&mut self, keyframe_or_intraonly: bool) {
        self.partition = [0u8; 8];
        self.skip = [0u8; 8];
        if keyframe_or_intraonly {
            // FFmpeg: memset(td->left_mode_ctx, DC_PRED, 16)
            // DC_PRED = 2, 16 entries for keyframe/intra-only
            self.y_mode = [2u8; 16];
        } else {
            // FFmpeg: memset(td->left_mode_ctx, NEARESTMV, 8)
            // NEARESTMV = 10, only 8 entries for inter frames
            self.y_mode[..8].fill(10);
        }
        self.coef = [[0u8; 16]; 3];
        self.seg_id = [0u8; 8];
        // Do NOT reset: tx_size, intra, comp, ref_frame, filter, mv, uv_mode
        // FFmpeg intentionally preserves these across SB row boundaries
        // (vp9.c:1343–1352 only resets the fields above)
    }

    /// Write a partition context value to `n` consecutive 8×8 rows starting
    /// at `row7` (row within the current 64-pixel SB, 0–7).
    pub fn set_partition(&mut self, row7: usize, val: u8, n: usize) {
        let end = (row7 + n).min(8);
        self.partition[row7..end].fill(val);
    }

    /// Write a skip value to `n` consecutive 4×4 rows starting at `row7`.
    pub fn set_skip(&mut self, row7: usize, val: u8, n: usize) {
        let end = (row7 + n).min(8);
        self.skip[row7..end].fill(val);
    }

    /// Write a tx_size value to `n` consecutive 4×4 rows starting at `row7`.
    pub fn set_tx_size(&mut self, row7: usize, val: u8, n: usize) {
        let end = (row7 + n).min(8);
        self.tx_size[row7..end].fill(val);
    }

    /// Write a y_mode value to `n` consecutive 4×4 rows starting at `row7*2`
    /// (the mode context uses 2 entries per 4×4 row for sub-block modes).
    pub fn set_y_mode(&mut self, row7: usize, val: u8, n: usize) {
        let base = row7 * 2;
        let end = (base + n * 2).min(16);
        self.y_mode[base..end].fill(val);
    }
}

impl Default for LeftContext {
    fn default() -> Self {
        Self::new()
    }
}
