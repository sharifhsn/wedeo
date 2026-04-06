// VP9 reference frame storage.
//
// Manages the 8 reference frame slots used by VP9 inter prediction,
// plus the previous frame's per-4×4 MV grid for temporal MV prediction.
//
// Translated from FFmpeg's VP9SharedContext.refs[], VP9Frame, and the
// reference management logic in vp9.c (vp9_frame_alloc/unref/replace,
// refresh logic in vp9_decode_frame).
//
// LGPL-2.1-or-later — same licence as FFmpeg.

use std::sync::Arc;

use crate::recon::FrameBuffer;

// ---------------------------------------------------------------------------
// Per-4×4 MV reference pair
// ---------------------------------------------------------------------------

/// Motion vector pair stored per 4×4 block, used for temporal MV prediction.
///
/// Mirrors `VP9mvrefPair` in vp9shared.h.
#[derive(Clone, Copy, Default)]
pub struct MvRefPair {
    /// Motion vectors for up to two reference frames: `mv[ref_idx] = [x, y]`
    /// in 1/8-pel units.
    pub mv: [[i16; 2]; 2],
    /// Reference frame indices. -1 = intra / unused.
    /// 0 = LAST, 1 = GOLDEN, 2 = ALTREF.
    pub ref_frame: [i8; 2],
}

// ---------------------------------------------------------------------------
// Decoded reference frame
// ---------------------------------------------------------------------------

/// A fully decoded reference frame: pixel data + per-4×4 MV grid.
///
/// Once constructed, a `RefFrame` is immutable and shared via `Arc` across
/// reference slots and tile-decode threads.
pub struct RefFrame {
    /// Decoded YUV pixel data.
    pub fb: FrameBuffer,
    /// Per-4×4 MV grid (`rows_4x4 * cols_4x4` entries).
    pub mv_grid: Vec<MvRefPair>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Columns in 4×4 units (`ceil(width / 4)`).
    pub cols_4x4: usize,
    /// Rows in 4×4 units (`ceil(height / 4)`).
    pub rows_4x4: usize,
}

// ---------------------------------------------------------------------------
// Reference frame store
// ---------------------------------------------------------------------------

/// Storage for 8 reference frame slots plus the previous frame's MV data.
///
/// Mirrors the reference management in `VP9SharedContext` (vp9shared.h) and
/// the refresh logic in `vp9_decode_frame` (vp9.c).
pub struct RefStore {
    /// 8 reference frame slots, indexed by `header.ref_idx[0..2]`.
    pub slots: [Option<Arc<RefFrame>>; 8],
    /// Previous frame's MV grid for temporal MV prediction
    /// (`frames[REF_FRAME_MVPAIR].mv` in FFmpeg).
    pub prev_frame_mvs: Option<Vec<MvRefPair>>,
    /// Previous frame's cols_4x4 (needed to index prev_frame_mvs correctly).
    pub prev_cols_4x4: usize,
}

impl RefStore {
    /// Create an empty reference store.
    pub fn new() -> Self {
        Self {
            slots: Default::default(),
            prev_frame_mvs: None,
            prev_cols_4x4: 0,
        }
    }

    /// Update reference slots based on `refresh_ref_mask`.
    ///
    /// Each set bit `i` in the mask replaces `slots[i]` with `frame`.
    /// Multiple bits can point to the same `Arc<RefFrame>`.
    pub fn refresh(&mut self, mask: u8, frame: Arc<RefFrame>) {
        for i in 0..8 {
            if mask & (1 << i) != 0 {
                self.slots[i] = Some(Arc::clone(&frame));
            }
        }
    }

    /// Rotate the current frame's MV grid into `prev_frame_mvs` for the
    /// next frame's temporal prediction.
    pub fn rotate_mvpair(&mut self, cur_mvs: Vec<MvRefPair>, cols_4x4: usize) {
        self.prev_frame_mvs = Some(cur_mvs);
        self.prev_cols_4x4 = cols_4x4;
    }

    /// Clear all reference state (e.g. on keyframe or error).
    pub fn clear(&mut self) {
        self.slots = Default::default();
        self.prev_frame_mvs = None;
        self.prev_cols_4x4 = 0;
    }
}

impl Default for RefStore {
    fn default() -> Self {
        Self::new()
    }
}
