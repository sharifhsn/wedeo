// VP9 motion vector prediction and parsing.
//
// Translated from FFmpeg's vp9mvs.c: find_ref_mvs, read_mv_component,
// ff_vp9_fill_mv.
//
// LGPL-2.1-or-later — same licence as FFmpeg.

use crate::bool_decoder::BoolDecoder;
use crate::data::{MV_CLASS_TREE, MV_FP_TREE, MV_JOINT_TREE, MV_REF_BLK_OFF};
use crate::prob::MvCompCounts;
use crate::refs::MvRefPair;
use crate::types::{BlockSize, MvCompProbs};

// ---------------------------------------------------------------------------
// MV clamping
// ---------------------------------------------------------------------------

/// Clamp an MV to tile boundaries.
#[inline]
fn clamp_mv(mv: [i16; 2], min_mv: [i16; 2], max_mv: [i16; 2]) -> [i16; 2] {
    [
        mv[0].clamp(min_mv[0], max_mv[0]),
        mv[1].clamp(min_mv[1], max_mv[1]),
    ]
}

/// Round half-pel away from zero when high-precision is disabled.
#[inline]
fn round_mv_comp(v: i16) -> i16 {
    if v & 1 != 0 {
        if v < 0 { v + 1 } else { v - 1 }
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// MV packing — treat [i16; 2] as a u32 for dedup comparison (matches FFmpeg)
// ---------------------------------------------------------------------------

#[inline]
fn mv_as_u32(mv: [i16; 2]) -> u32 {
    (mv[0] as u16 as u32) | ((mv[1] as u16 as u32) << 16)
}

const INVALID_MV: u32 = 0x8000_8000;

// ---------------------------------------------------------------------------
// find_ref_mvs — spatial + temporal MV prediction
// ---------------------------------------------------------------------------

/// Context for MV search boundaries (computed per-block in decode_block).
pub struct MvSearchCtx {
    pub min_mv: [i16; 2],
    pub max_mv: [i16; 2],
    pub tile_col_start: usize,
    pub cols_4x4: usize,
    pub rows_4x4: usize,
    /// sign_bias[ref_frame], indexed 0-2 for LAST/GOLDEN/ALTREF.
    /// -1 indices will never be looked up.
    pub sign_bias: [bool; 3],
}

/// Find reference MV prediction for one block.
///
/// Translated from `find_ref_mvs` in vp9mvs.c.
///
/// # Arguments
/// * `cur_mv_grid` — current frame's in-progress per-4×4 MV grid.
/// * `prev_mv_grid` — previous frame's MV grid (for temporal prediction).
/// * `prev_cols_4x4` — previous frame's cols_4x4 (= sb_cols*8 in FFmpeg).
/// * `above_mv_ctx` — above MV context (from AboveContext.mv).
/// * `left_mv_ctx` — left MV context (from LeftContext.mv).
/// * `pmv` — output: the predicted MV (clamped).
/// * `ref_idx` — reference frame index we're predicting for (0=LAST, etc.).
/// * `z` — which ref slot to read from MV pairs (0 or 1 for compound).
/// * `idx` — 0 for nearest, 1 for near.
/// * `sb` — sub-block index (-1 for >=8x8).
/// * `row`, `col` — block position in 4×4 units.
/// * `bs` — block size enum.
/// * `sub_mvs` — previously decoded MVs for sub-blocks within this 8×8
///   (`b->mv[sub][z]`), only meaningful when sb >= 0.
/// * `use_last_frame_mvs` — whether temporal MV prediction is enabled.
#[allow(clippy::too_many_arguments)]
pub fn find_ref_mvs(
    ctx: &MvSearchCtx,
    cur_mv_grid: &[MvRefPair],
    prev_mv_grid: Option<&[MvRefPair]>,
    prev_cols_4x4: usize,
    above_mv_ctx: &[[[i16; 2]; 2]],
    left_mv_ctx: &[[[i16; 2]; 2]],
    pmv: &mut [i16; 2],
    ref_idx: i8,
    z: usize,
    idx: usize,
    sb: i32,
    row: usize,
    col: usize,
    row7: usize,
    bs: BlockSize,
    sub_mvs: &[[[i16; 2]; 2]; 4],
    use_last_frame_mvs: bool,
) {
    let p = &MV_REF_BLK_OFF[bs as usize];
    let mut mem: u32 = INVALID_MV;
    let mut mem_sub8x8: u32 = INVALID_MV;

    // --- Macros translated as closures with early-return via `return` ---
    // Since Rust closures can't `return` from the enclosing function, we use
    // a single big function body with a label-like approach via a helper.

    // Helper: RETURN_DIRECT_MV logic. Returns Some(mv) if we should return.
    #[inline(always)]
    fn try_direct_mv(mv: [i16; 2], idx: usize, mem: &mut u32, pmv: &mut [i16; 2]) -> bool {
        let m = mv_as_u32(mv);
        if idx == 0 {
            *pmv = mv;
            return true;
        } else if *mem == INVALID_MV {
            *mem = m;
        } else if m != *mem {
            *pmv = mv;
            return true;
        }
        false
    }

    // Helper: RETURN_MV logic for sb <= 0 path.
    #[inline(always)]
    fn try_mv_normal(
        mv: [i16; 2],
        idx: usize,
        mem: &mut u32,
        pmv: &mut [i16; 2],
        min_mv: [i16; 2],
        max_mv: [i16; 2],
    ) -> bool {
        let m = mv_as_u32(mv);
        if idx == 0 {
            *pmv = clamp_mv(mv, min_mv, max_mv);
            return true;
        } else if *mem == INVALID_MV {
            *mem = m;
        } else if m != *mem {
            *pmv = clamp_mv(mv, min_mv, max_mv);
            return true;
        }
        false
    }

    // Helper: RETURN_MV logic for sb > 0 path.
    #[inline(always)]
    fn try_mv_sub8x8(
        mv: [i16; 2],
        mem: &u32,
        mem_sub8x8: &mut u32,
        pmv: &mut [i16; 2],
        min_mv: [i16; 2],
        max_mv: [i16; 2],
    ) -> bool {
        // In the sub-8x8 path, idx is always 1 and mem is always valid.
        if *mem_sub8x8 == INVALID_MV {
            let tmp = clamp_mv(mv, min_mv, max_mv);
            let m = mv_as_u32(tmp);
            if m != *mem {
                *pmv = tmp;
                return true;
            }
            *mem_sub8x8 = mv_as_u32(mv);
        } else if *mem_sub8x8 != mv_as_u32(mv) {
            let tmp = clamp_mv(mv, min_mv, max_mv);
            let m = mv_as_u32(tmp);
            if m != *mem {
                *pmv = tmp;
            } else {
                // BUG — libvpx writes zero here
                *pmv = [0, 0];
            }
            return true;
        }
        false
    }

    // Macro-like dispatcher for RETURN_MV
    macro_rules! return_mv {
        ($mv:expr) => {
            if sb > 0 {
                if try_mv_sub8x8($mv, &mem, &mut mem_sub8x8, pmv, ctx.min_mv, ctx.max_mv) {
                    return;
                }
            } else {
                if try_mv_normal($mv, idx, &mut mem, pmv, ctx.min_mv, ctx.max_mv) {
                    return;
                }
            }
        };
    }

    macro_rules! return_scale_mv {
        ($mv:expr, $scale:expr) => {
            if $scale {
                let mv_temp: [i16; 2] = [$mv[0].wrapping_neg(), $mv[1].wrapping_neg()];
                return_mv!(mv_temp);
            } else {
                return_mv!($mv);
            }
        };
    }

    let sb_cols_8 = ctx.cols_4x4; // In FFmpeg: s->sb_cols * 8. For us, cols_4x4.

    let i_start = if sb >= 0 {
        // Sub-8x8: check previously decoded sub-blocks.
        if sb == 2 || sb == 1 {
            if try_direct_mv(sub_mvs[0][z], idx, &mut mem, pmv) {
                return;
            }
        } else if sb == 3 {
            if try_direct_mv(sub_mvs[2][z], idx, &mut mem, pmv) {
                return;
            }
            if try_direct_mv(sub_mvs[1][z], idx, &mut mem, pmv) {
                return;
            }
            if try_direct_mv(sub_mvs[0][z], idx, &mut mem, pmv) {
                return;
            }
        }

        // Check above neighbor (using above_mv_ctx).
        if row > 0 {
            let mv_pair = &cur_mv_grid[(row - 1) * sb_cols_8 + col];
            let above_idx = 2 * col + (sb as usize & 1);
            if mv_pair.ref_frame[0] == ref_idx && above_idx < above_mv_ctx.len() {
                return_mv!(above_mv_ctx[above_idx][0]);
            } else if mv_pair.ref_frame[1] == ref_idx && above_idx < above_mv_ctx.len() {
                return_mv!(above_mv_ctx[above_idx][1]);
            }
        }
        // Check left neighbor (using left_mv_ctx).
        if col > ctx.tile_col_start {
            let mv_pair = &cur_mv_grid[row * sb_cols_8 + col - 1];
            let left_idx = 2 * row7 + (sb as usize >> 1);
            if mv_pair.ref_frame[0] == ref_idx && left_idx < left_mv_ctx.len() {
                return_mv!(left_mv_ctx[left_idx][0]);
            } else if mv_pair.ref_frame[1] == ref_idx && left_idx < left_mv_ctx.len() {
                return_mv!(left_mv_ctx[left_idx][1]);
            }
        }
        2
    } else {
        0
    };

    // Previously coded MVs in this neighborhood, same reference frame.
    // MV_REF_BLK_OFF offsets are in 8×8 units (matching FFmpeg's coord system),
    // but our col/row are in 4×4 units, so multiply offsets by 2.
    for off in &p[i_start..] {
        let c = col as isize + off[0] as isize * 2;
        let r = row as isize + off[1] as isize * 2;

        if c >= ctx.tile_col_start as isize
            && (c as usize) < ctx.cols_4x4
            && r >= 0
            && (r as usize) < ctx.rows_4x4
        {
            let mv_pair = &cur_mv_grid[r as usize * sb_cols_8 + c as usize];
            if mv_pair.ref_frame[0] == ref_idx {
                return_mv!(mv_pair.mv[0]);
            } else if mv_pair.ref_frame[1] == ref_idx {
                return_mv!(mv_pair.mv[1]);
            }
        }
    }

    // MV at this position in previous frame, same reference frame.
    if let Some(prev) = prev_mv_grid.filter(|_| use_last_frame_mvs) {
        let prev_idx = row * prev_cols_4x4 + col;
        if prev_idx < prev.len() {
            let mv_pair = &prev[prev_idx];
            if mv_pair.ref_frame[0] == ref_idx {
                return_mv!(mv_pair.mv[0]);
            } else if mv_pair.ref_frame[1] == ref_idx {
                return_mv!(mv_pair.mv[1]);
            }
        }
    }

    // Previously coded MVs in this neighborhood, DIFFERENT reference frame
    // (sign-flipped if sign_bias differs).
    // Same 8×8→4×4 offset scaling as above.
    for off in p {
        let c = col as isize + off[0] as isize * 2;
        let r = row as isize + off[1] as isize * 2;

        if c >= ctx.tile_col_start as isize
            && (c as usize) < ctx.cols_4x4
            && r >= 0
            && (r as usize) < ctx.rows_4x4
        {
            let mv_pair = &cur_mv_grid[r as usize * sb_cols_8 + c as usize];
            if mv_pair.ref_frame[0] != ref_idx && mv_pair.ref_frame[0] >= 0 {
                let scale =
                    ctx.sign_bias[mv_pair.ref_frame[0] as usize] != ctx.sign_bias[ref_idx as usize];
                return_scale_mv!(mv_pair.mv[0], scale);
            }
            if mv_pair.ref_frame[1] != ref_idx
                && mv_pair.ref_frame[1] >= 0
                // BUG — libvpx has this condition regardless of whether
                // we used the first ref MV and pre-scaling
                && mv_as_u32(mv_pair.mv[0]) != mv_as_u32(mv_pair.mv[1])
            {
                let scale =
                    ctx.sign_bias[mv_pair.ref_frame[1] as usize] != ctx.sign_bias[ref_idx as usize];
                return_scale_mv!(mv_pair.mv[1], scale);
            }
        }
    }

    // MV at this position in previous frame, different reference frame.
    if let Some(prev) = prev_mv_grid.filter(|_| use_last_frame_mvs) {
        let prev_idx = row * prev_cols_4x4 + col;
        if prev_idx < prev.len() {
            let mv_pair = &prev[prev_idx];
            if mv_pair.ref_frame[0] != ref_idx && mv_pair.ref_frame[0] >= 0 {
                let scale =
                    ctx.sign_bias[mv_pair.ref_frame[0] as usize] != ctx.sign_bias[ref_idx as usize];
                return_scale_mv!(mv_pair.mv[0], scale);
            }
            if mv_pair.ref_frame[1] != ref_idx
                && mv_pair.ref_frame[1] >= 0
                && mv_as_u32(mv_pair.mv[0]) != mv_as_u32(mv_pair.mv[1])
            {
                let scale =
                    ctx.sign_bias[mv_pair.ref_frame[1] as usize] != ctx.sign_bias[ref_idx as usize];
                return_scale_mv!(mv_pair.mv[1], scale);
            }
        }
    }

    // Fallback: zero MV.
    *pmv = clamp_mv([0, 0], ctx.min_mv, ctx.max_mv);
}

// ---------------------------------------------------------------------------
// read_mv_component — reads one MV component from the bitstream
// ---------------------------------------------------------------------------

/// Read one MV component (x or y) from the boolean decoder.
///
/// Translated from `read_mv_component` in vp9mvs.c.
///
/// * `idx` — 0 = vertical (y), 1 = horizontal (x). Selects which
///   `mv_comp` probability set to use.
/// * `hp` — whether high-precision MVs are enabled for this block.
///
/// Returns the signed MV component value in 1/8-pel units.
pub fn read_mv_component(
    bd: &mut BoolDecoder<'_>,
    prob: &MvCompProbs,
    counts: &mut MvCompCounts,
    hp: bool,
) -> i16 {
    let sign = bd.get_prob(prob.sign);
    let c = bd.get_tree(&MV_CLASS_TREE, &prob.classes) as usize;

    counts.sign[sign as usize] += 1;
    counts.classes[c] += 1;

    let n = if c > 0 {
        // Classes 1-10: c magnitude bits + 3-bit fractional + optional HP bit.
        let mut mantissa: i32 = 0;
        for m in 0..c {
            let bit = bd.get_prob(prob.bits[m]);
            mantissa |= (bit as i32) << m;
            counts.bits[m][bit as usize] += 1;
        }
        mantissa <<= 3;

        let fp = bd.get_tree(&MV_FP_TREE, &prob.fp);
        mantissa |= fp << 1;
        counts.fp[fp as usize] += 1;

        if hp {
            let bit = bd.get_prob(prob.hp);
            counts.hp[bit as usize] += 1;
            mantissa |= bit as i32;
        } else {
            mantissa |= 1;
            // BUG in libvpx — count HP=1 even when not coded.
            counts.hp[1] += 1;
        }
        mantissa + (8 << c)
    } else {
        // Class 0: class0 bit + class0_fp + optional HP bit.
        let class0 = bd.get_prob(prob.class0) as usize;
        counts.class0[class0] += 1;

        let fp = bd.get_tree(&MV_FP_TREE, &prob.class0_fp[class0]);
        counts.class0_fp[class0][fp as usize] += 1;

        let mut val = (class0 as i32) << 3 | (fp << 1);

        if hp {
            let bit = bd.get_prob(prob.class0_hp);
            counts.class0_hp[bit as usize] += 1;
            val |= bit as i32;
        } else {
            val |= 1;
            // BUG in libvpx — count HP=1 even when not coded.
            counts.class0_hp[1] += 1;
        }
        val
    };

    if sign { -(n as i16 + 1) } else { n as i16 + 1 }
}

// ---------------------------------------------------------------------------
// fill_mv — top-level MV fill dispatch
// ---------------------------------------------------------------------------

/// Fill motion vectors for one block or sub-block.
///
/// Translated from `ff_vp9_fill_mv` in vp9mvs.c.
///
/// * `mv_out` — output: `mv_out[0]` = ref0 MV, `mv_out[1]` = ref1 MV.
/// * `mode` — inter prediction mode (NEARESTMV=10, NEARMV=11, ZEROMV=12, NEWMV=13).
/// * `sb` — sub-block index (-1 for >=8x8).
/// * `comp` — whether compound (two-reference) prediction.
/// * `ref_frames` — `[ref0, ref1]` frame indices (0-2).
/// * `sub_mvs` — previously decoded sub-block MVs (`b->mv[0..4]`).
#[allow(clippy::too_many_arguments)]
pub fn fill_mv(
    bd: &mut BoolDecoder<'_>,
    ctx: &MvSearchCtx,
    cur_mv_grid: &[MvRefPair],
    prev_mv_grid: Option<&[MvRefPair]>,
    prev_cols_4x4: usize,
    above_mv_ctx: &[[[i16; 2]; 2]],
    left_mv_ctx: &[[[i16; 2]; 2]],
    mv_out: &mut [[i16; 2]; 2],
    mode: u8,
    sb: i32,
    row: usize,
    col: usize,
    row7: usize,
    bs: BlockSize,
    comp: bool,
    ref_frames: &[i8; 2],
    sub_mvs: &[[[i16; 2]; 2]; 4],
    prob: &crate::types::ProbContext,
    counts: &mut crate::prob::CountContext,
    high_precision_mvs: bool,
    use_last_frame_mvs: bool,
) {
    const ZEROMV: u8 = 12;
    const NEARMV: u8 = 11;
    const NEWMV: u8 = 13;

    if mode == ZEROMV {
        mv_out[0] = [0, 0];
        mv_out[1] = [0, 0];
        return;
    }

    // Predict MV for ref[0].
    find_ref_mvs(
        ctx,
        cur_mv_grid,
        prev_mv_grid,
        prev_cols_4x4,
        above_mv_ctx,
        left_mv_ctx,
        &mut mv_out[0],
        ref_frames[0],
        0,
        if mode == NEARMV { 1 } else { 0 },
        if mode == NEWMV { -1 } else { sb },
        row,
        col,
        row7,
        bs,
        sub_mvs,
        use_last_frame_mvs,
    );

    // Round to half-pel if high-precision is disabled or MV is too large.
    let hp0 =
        high_precision_mvs && (mv_out[0][0] as i32).abs() < 64 && (mv_out[0][1] as i32).abs() < 64;
    if (mode == NEWMV || sb == -1) && !hp0 {
        mv_out[0][0] = round_mv_comp(mv_out[0][0]);
        mv_out[0][1] = round_mv_comp(mv_out[0][1]);
    }

    if mode == NEWMV {
        let j = bd.get_tree(&MV_JOINT_TREE, &prob.mv_joint) as usize;
        counts.mv_joint[j] += 1;
        // MV_JOINT_V = 2, MV_JOINT_HV = 3
        if j >= 2 {
            mv_out[0][1] = mv_out[0][1].wrapping_add(read_mv_component(
                bd,
                &prob.mv_comp[0],
                &mut counts.mv_comp[0],
                hp0,
            ));
        }
        if j & 1 != 0 {
            mv_out[0][0] = mv_out[0][0].wrapping_add(read_mv_component(
                bd,
                &prob.mv_comp[1],
                &mut counts.mv_comp[1],
                hp0,
            ));
        }
    }

    if comp {
        // Predict MV for ref[1].
        find_ref_mvs(
            ctx,
            cur_mv_grid,
            prev_mv_grid,
            prev_cols_4x4,
            above_mv_ctx,
            left_mv_ctx,
            &mut mv_out[1],
            ref_frames[1],
            1,
            if mode == NEARMV { 1 } else { 0 },
            if mode == NEWMV { -1 } else { sb },
            row,
            col,
            row7,
            bs,
            sub_mvs,
            use_last_frame_mvs,
        );

        let hp1 = high_precision_mvs
            && (mv_out[1][0] as i32).abs() < 64
            && (mv_out[1][1] as i32).abs() < 64;
        if (mode == NEWMV || sb == -1) && !hp1 {
            mv_out[1][0] = round_mv_comp(mv_out[1][0]);
            mv_out[1][1] = round_mv_comp(mv_out[1][1]);
        }

        if mode == NEWMV {
            let j = bd.get_tree(&MV_JOINT_TREE, &prob.mv_joint) as usize;
            counts.mv_joint[j] += 1;
            if j >= 2 {
                mv_out[1][1] = mv_out[1][1].wrapping_add(read_mv_component(
                    bd,
                    &prob.mv_comp[0],
                    &mut counts.mv_comp[0],
                    hp1,
                ));
            }
            if j & 1 != 0 {
                mv_out[1][0] = mv_out[1][0].wrapping_add(read_mv_component(
                    bd,
                    &prob.mv_comp[1],
                    &mut counts.mv_comp[1],
                    hp1,
                ));
            }
        }
    }
}
