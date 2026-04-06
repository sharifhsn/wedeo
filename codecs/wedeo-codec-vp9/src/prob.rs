// VP9 probability adaptation.
//
// Translated from FFmpeg's libavcodec/vp9prob.c (`ff_vp9_adapt_probs`).
// After decoding a frame the accumulated symbol counts are used to update
// the probability tables for the next frame.

use crate::types::{CompPredMode, MvCompProbs, ProbContext};

/// Type alias for the 6-dimensional coefficient probability array (3 probs per entry).
/// Dimensions: [tx_size(4)][block_type(2)][intra(2)][band(6)][ctx(6)][3].
#[allow(clippy::type_complexity)]
pub type CoefProbArray = [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4];

/// Type alias for the 6-dimensional coefficient count array (3 counts per entry).
/// Dimensions: [tx_size(4)][block_type(2)][intra(2)][band(6)][ctx(6)][3].
#[allow(clippy::type_complexity)]
pub type CoefCountArray = [[[[[[u32; 3]; 6]; 6]; 2]; 2]; 4];

/// Type alias for the 6-dimensional eob count array (2 counts per entry).
/// Dimensions: [tx_size(4)][block_type(2)][intra(2)][band(6)][ctx(6)][2].
#[allow(clippy::type_complexity)]
pub type EobCountArray = [[[[[[u32; 2]; 6]; 6]; 2]; 2]; 4];

// ---------------------------------------------------------------------------
// Count context — mirrors the `counts` sub-struct of VP9TileData in vp9dec.h.
// ---------------------------------------------------------------------------

/// Per MV-component counts (mirrors the anonymous struct inside VP9TileData.counts).
#[derive(Clone, Debug, Default)]
pub struct MvCompCounts {
    pub sign: [u32; 2],
    pub classes: [u32; 11],
    pub class0: [u32; 2],
    pub bits: [[u32; 2]; 10],
    pub class0_fp: [[u32; 4]; 2],
    pub fp: [u32; 4],
    pub class0_hp: [u32; 2],
    pub hp: [u32; 2],
}

/// Full frame-level symbol counts accumulated during tile decoding.
///
/// All array dimensions match the corresponding probability arrays in
/// `ProbContext` (plus one dimension for the binary choice).
#[derive(Clone, Debug, Default)]
pub struct CountContext {
    pub y_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub filter: [[u32; 3]; 4],
    pub mv_mode: [[u32; 4]; 7],
    pub intra: [[u32; 2]; 4],
    pub comp: [[u32; 2]; 5],
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub tx32p: [[u32; 4]; 2],
    pub tx16p: [[u32; 3]; 2],
    pub tx8p: [[u32; 2]; 2],
    pub skip: [[u32; 2]; 3],
    pub mv_joint: [u32; 4],
    pub mv_comp: [MvCompCounts; 2],
    pub partition: [[[[u32; 4]; 4]; 4]; 1], // [1][4][4][4] — only used for inter
    pub coef: CoefCountArray,
    pub eob: EobCountArray,
}

impl CountContext {
    /// Element-wise addition of all count fields from `other` into `self`.
    pub fn merge(&mut self, other: &CountContext) {
        for (a, b) in self
            .y_mode
            .iter_mut()
            .flatten()
            .zip(other.y_mode.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self
            .uv_mode
            .iter_mut()
            .flatten()
            .zip(other.uv_mode.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self
            .filter
            .iter_mut()
            .flatten()
            .zip(other.filter.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self
            .mv_mode
            .iter_mut()
            .flatten()
            .zip(other.mv_mode.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self
            .intra
            .iter_mut()
            .flatten()
            .zip(other.intra.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self
            .comp
            .iter_mut()
            .flatten()
            .zip(other.comp.iter().flatten())
        {
            *a += b;
        }
        for i in 0..5 {
            for j in 0..2 {
                for k in 0..2 {
                    self.single_ref[i][j][k] += other.single_ref[i][j][k];
                }
            }
        }
        for (a, b) in self
            .comp_ref
            .iter_mut()
            .flatten()
            .zip(other.comp_ref.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self
            .tx32p
            .iter_mut()
            .flatten()
            .zip(other.tx32p.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self
            .tx16p
            .iter_mut()
            .flatten()
            .zip(other.tx16p.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self
            .tx8p
            .iter_mut()
            .flatten()
            .zip(other.tx8p.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self
            .skip
            .iter_mut()
            .flatten()
            .zip(other.skip.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in self.mv_joint.iter_mut().zip(other.mv_joint.iter()) {
            *a += b;
        }
        for c in 0..2 {
            for (a, b) in self.mv_comp[c]
                .sign
                .iter_mut()
                .zip(other.mv_comp[c].sign.iter())
            {
                *a += b;
            }
            for (a, b) in self.mv_comp[c]
                .classes
                .iter_mut()
                .zip(other.mv_comp[c].classes.iter())
            {
                *a += b;
            }
            for (a, b) in self.mv_comp[c]
                .class0
                .iter_mut()
                .zip(other.mv_comp[c].class0.iter())
            {
                *a += b;
            }
            for (a, b) in self.mv_comp[c]
                .bits
                .iter_mut()
                .flatten()
                .zip(other.mv_comp[c].bits.iter().flatten())
            {
                *a += b;
            }
            for (a, b) in self.mv_comp[c]
                .class0_fp
                .iter_mut()
                .flatten()
                .zip(other.mv_comp[c].class0_fp.iter().flatten())
            {
                *a += b;
            }
            for (a, b) in self.mv_comp[c]
                .fp
                .iter_mut()
                .zip(other.mv_comp[c].fp.iter())
            {
                *a += b;
            }
            for (a, b) in self.mv_comp[c]
                .class0_hp
                .iter_mut()
                .zip(other.mv_comp[c].class0_hp.iter())
            {
                *a += b;
            }
            for (a, b) in self.mv_comp[c]
                .hp
                .iter_mut()
                .zip(other.mv_comp[c].hp.iter())
            {
                *a += b;
            }
        }
        for i in 0..1 {
            for j in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        self.partition[i][j][k][l] += other.partition[i][j][k][l];
                    }
                }
            }
        }
        // Coefficient and EOB counts — 6D arrays.
        for (a, b) in self
            .coef
            .iter_mut()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
            .zip(
                other
                    .coef
                    .iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .flatten()
                    .flatten(),
            )
        {
            *a += b;
        }
        for (a, b) in self
            .eob
            .iter_mut()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
            .flatten()
            .zip(
                other
                    .eob
                    .iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .flatten()
                    .flatten(),
            )
        {
            *a += b;
        }
    }
}

// ---------------------------------------------------------------------------
// Core adaptation helper
// ---------------------------------------------------------------------------

/// Weighted-average probability update.
///
/// Mirrors `adapt_prob` in vp9prob.c.
/// `ct0` = count for the "zero" (false) branch.
/// `ct1` = count for the "one" (true) branch.
/// `max_count` caps the total count before scaling.
/// `update_factor` controls how fast the probability shifts.
#[inline]
fn adapt_prob(p: &mut u8, ct0: u32, ct1: u32, max_count: u32, update_factor: u32) {
    let ct = ct0 + ct1;
    if ct == 0 {
        return;
    }
    // Scale update_factor by min(ct, max_count) / max_count.
    // Integer division matches FFmpeg's FASTDIV macro (exact for small values).
    let uf = update_factor * ct.min(max_count) / max_count;
    let p1 = *p as u32;
    // New probability = (ct0 << 8 + ct/2) / ct, clamped to [1, 255].
    let p2 = (((ct0 as u64) << 8) + (ct as u64 / 2)) / ct as u64;
    let p2 = (p2 as u32).clamp(1, 255);
    // Blend: p1 + ((p2 - p1) * uf + 128) >> 8  (signed arithmetic).
    let delta = (p2 as i32 - p1 as i32) * uf as i32;
    *p = (p1 as i32 + ((delta + 128) >> 8)).clamp(1, 255) as u8;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Adapt all frame-level probabilities based on symbol counts from this frame.
///
/// Mirrors `ff_vp9_adapt_probs` in vp9prob.c.
///
/// `keyframe_or_intraonly` should be `true` for keyframes and intra-only frames.
/// `last_was_keyframe` is `true` when the previous frame was a keyframe.
/// `highprecisionmvs` enables adaptation of the high-precision MV probabilities.
/// `filtermode_switchable` enables filter probability adaptation.
/// `txfm_switchable` enables tx-size probability adaptation.
/// `comppredmode` controls which reference-frame probability sets are adapted.
// Parameters mirror `ff_vp9_adapt_probs` in vp9prob.c.
#[allow(clippy::too_many_arguments)]
pub fn adapt_probs(
    prob: &mut ProbContext,
    coef: &mut CoefProbArray,
    counts: &CountContext,
    keyframe_or_intraonly: bool,
    last_was_keyframe: bool,
    highprecisionmvs: bool,
    filtermode_switchable: bool,
    txfm_switchable: bool,
    comppredmode: CompPredMode,
) {
    // update_factor: slower adaptation for keyframe/intra, faster for inter.
    let uf: u32 = if keyframe_or_intraonly || !last_was_keyframe {
        112
    } else {
        128
    };

    // Coefficient probabilities — always adapted.
    // Explicit index loops required because the DC band (l=0) breaks early at m=3.
    #[allow(clippy::needless_range_loop)]
    for i in 0..4usize {
        #[allow(clippy::needless_range_loop)]
        for j in 0..2usize {
            #[allow(clippy::needless_range_loop)]
            for k in 0..2usize {
                #[allow(clippy::needless_range_loop)]
                for l in 0..6usize {
                    #[allow(clippy::needless_range_loop)]
                    for m in 0..6usize {
                        if l == 0 && m >= 3 {
                            // DC only has 3 context points.
                            break;
                        }
                        let pp = &mut coef[i][j][k][l][m];
                        let e = &counts.eob[i][j][k][l][m];
                        let c = &counts.coef[i][j][k][l][m];
                        adapt_prob(&mut pp[0], e[0], e[1], 24, uf);
                        adapt_prob(&mut pp[1], c[0], c[1] + c[2], 24, uf);
                        adapt_prob(&mut pp[2], c[1], c[2], 24, uf);
                    }
                }
            }
        }
    }

    if keyframe_or_intraonly {
        // For keyframes/intra-only, the non-coef probs are reset to the
        // current frame's probs (already set during header parsing).
        return;
    }

    // Skip probabilities.
    for i in 0..3 {
        adapt_prob(
            &mut prob.skip[i],
            counts.skip[i][0],
            counts.skip[i][1],
            20,
            128,
        );
    }

    // Intra/inter flag.
    for i in 0..4 {
        adapt_prob(
            &mut prob.intra[i],
            counts.intra[i][0],
            counts.intra[i][1],
            20,
            128,
        );
    }

    // Compound prediction flag.
    if comppredmode == CompPredMode::Switchable {
        for i in 0..5 {
            adapt_prob(
                &mut prob.comp[i],
                counts.comp[i][0],
                counts.comp[i][1],
                20,
                128,
            );
        }
    }

    // Reference frames.
    if comppredmode != CompPredMode::SingleRef {
        for i in 0..5 {
            adapt_prob(
                &mut prob.comp_ref[i],
                counts.comp_ref[i][0],
                counts.comp_ref[i][1],
                20,
                128,
            );
        }
    }

    if comppredmode != CompPredMode::CompRef {
        for i in 0..5 {
            adapt_prob(
                &mut prob.single_ref[i][0],
                counts.single_ref[i][0][0],
                counts.single_ref[i][0][1],
                20,
                128,
            );
            adapt_prob(
                &mut prob.single_ref[i][1],
                counts.single_ref[i][1][0],
                counts.single_ref[i][1][1],
                20,
                128,
            );
        }
    }

    // Block partitioning.
    // FFmpeg iterates [4 levels][4 combined_ctx] where combined_ctx = above_bit | (left_bit<<1).
    // Rust ProbContext.partition is [block_level][above_bit][left_bit][3].
    for i in 0..4usize {
        for j in 0..4usize {
            // Map combined context j to separate above/left bits.
            let above_bit = j & 1;
            let left_bit = j >> 1;
            let c = &counts.partition[0][i][j];
            adapt_prob(
                &mut prob.partition[i][above_bit][left_bit][0],
                c[0],
                c[1] + c[2] + c[3],
                20,
                128,
            );
            adapt_prob(
                &mut prob.partition[i][above_bit][left_bit][1],
                c[1],
                c[2] + c[3],
                20,
                128,
            );
            adapt_prob(
                &mut prob.partition[i][above_bit][left_bit][2],
                c[2],
                c[3],
                20,
                128,
            );
        }
    }

    // TX size.
    if txfm_switchable {
        for i in 0..2 {
            let c8 = &counts.tx8p[i];
            let c16 = &counts.tx16p[i];
            let c32 = &counts.tx32p[i];

            adapt_prob(&mut prob.tx8p[i], c8[0], c8[1], 20, 128);
            adapt_prob(&mut prob.tx16p[i][0], c16[0], c16[1] + c16[2], 20, 128);
            adapt_prob(&mut prob.tx16p[i][1], c16[1], c16[2], 20, 128);
            adapt_prob(
                &mut prob.tx32p[i][0],
                c32[0],
                c32[1] + c32[2] + c32[3],
                20,
                128,
            );
            adapt_prob(&mut prob.tx32p[i][1], c32[1], c32[2] + c32[3], 20, 128);
            adapt_prob(&mut prob.tx32p[i][2], c32[2], c32[3], 20, 128);
        }
    }

    // Interpolation filter.
    if filtermode_switchable {
        for i in 0..4 {
            adapt_prob(
                &mut prob.filter[i][0],
                counts.filter[i][0],
                counts.filter[i][1] + counts.filter[i][2],
                20,
                128,
            );
            adapt_prob(
                &mut prob.filter[i][1],
                counts.filter[i][1],
                counts.filter[i][2],
                20,
                128,
            );
        }
    }

    // Inter modes.
    for i in 0..7 {
        let c = &counts.mv_mode[i];
        adapt_prob(&mut prob.mv_mode[i][0], c[2], c[1] + c[0] + c[3], 20, 128);
        adapt_prob(&mut prob.mv_mode[i][1], c[0], c[1] + c[3], 20, 128);
        adapt_prob(&mut prob.mv_mode[i][2], c[1], c[3], 20, 128);
    }

    // MV joints.
    {
        let c = &counts.mv_joint;
        adapt_prob(&mut prob.mv_joint[0], c[0], c[1] + c[2] + c[3], 20, 128);
        adapt_prob(&mut prob.mv_joint[1], c[1], c[2] + c[3], 20, 128);
        adapt_prob(&mut prob.mv_joint[2], c[2], c[3], 20, 128);
    }

    // MV components.
    for i in 0..2 {
        adapt_mv_comp(&mut prob.mv_comp[i], &counts.mv_comp[i], highprecisionmvs);
    }

    // Y intra modes.
    for i in 0..4 {
        adapt_y_mode(&mut prob.y_mode[i], &counts.y_mode[i]);
    }

    // UV intra modes.
    for i in 0..10 {
        adapt_uv_mode(&mut prob.uv_mode[i], &counts.uv_mode[i]);
    }
}

/// Adapt a single MV component's probability set.
fn adapt_mv_comp(pp: &mut MvCompProbs, c: &MvCompCounts, highprecisionmvs: bool) {
    adapt_prob(&mut pp.sign, c.sign[0], c.sign[1], 20, 128);

    // classes (11 values, 10 probabilities — binary tree).
    let classes = &c.classes;
    let sum_all = classes[1]
        + classes[2]
        + classes[3]
        + classes[4]
        + classes[5]
        + classes[6]
        + classes[7]
        + classes[8]
        + classes[9]
        + classes[10];
    let mut sum = sum_all;
    adapt_prob(&mut pp.classes[0], classes[0], sum, 20, 128);
    sum -= classes[1];
    adapt_prob(&mut pp.classes[1], classes[1], sum, 20, 128);
    sum -= classes[2] + classes[3];
    adapt_prob(&mut pp.classes[2], classes[2] + classes[3], sum, 20, 128);
    adapt_prob(&mut pp.classes[3], classes[2], classes[3], 20, 128);
    sum -= classes[4] + classes[5];
    adapt_prob(&mut pp.classes[4], classes[4] + classes[5], sum, 20, 128);
    adapt_prob(&mut pp.classes[5], classes[4], classes[5], 20, 128);
    sum -= classes[6];
    adapt_prob(&mut pp.classes[6], classes[6], sum, 20, 128);
    adapt_prob(
        &mut pp.classes[7],
        classes[7] + classes[8],
        classes[9] + classes[10],
        20,
        128,
    );
    adapt_prob(&mut pp.classes[8], classes[7], classes[8], 20, 128);
    adapt_prob(&mut pp.classes[9], classes[9], classes[10], 20, 128);

    adapt_prob(&mut pp.class0, c.class0[0], c.class0[1], 20, 128);
    for j in 0..10 {
        adapt_prob(&mut pp.bits[j], c.bits[j][0], c.bits[j][1], 20, 128);
    }

    for j in 0..2 {
        adapt_prob(
            &mut pp.class0_fp[j][0],
            c.class0_fp[j][0],
            c.class0_fp[j][1] + c.class0_fp[j][2] + c.class0_fp[j][3],
            20,
            128,
        );
        adapt_prob(
            &mut pp.class0_fp[j][1],
            c.class0_fp[j][1],
            c.class0_fp[j][2] + c.class0_fp[j][3],
            20,
            128,
        );
        adapt_prob(
            &mut pp.class0_fp[j][2],
            c.class0_fp[j][2],
            c.class0_fp[j][3],
            20,
            128,
        );
    }
    adapt_prob(&mut pp.fp[0], c.fp[0], c.fp[1] + c.fp[2] + c.fp[3], 20, 128);
    adapt_prob(&mut pp.fp[1], c.fp[1], c.fp[2] + c.fp[3], 20, 128);
    adapt_prob(&mut pp.fp[2], c.fp[2], c.fp[3], 20, 128);

    if highprecisionmvs {
        adapt_prob(&mut pp.class0_hp, c.class0_hp[0], c.class0_hp[1], 20, 128);
        adapt_prob(&mut pp.hp, c.hp[0], c.hp[1], 20, 128);
    }
}

/// Adapt Y intra-mode probabilities for one block-size context.
///
/// The intra mode tree is non-trivially binary; counts must be mapped
/// to the 9 probability entries in the same order as FFmpeg's vp9prob.c.
fn adapt_y_mode(pp: &mut [u8; 9], c: &[u32; 10]) {
    // Index mapping: DC_PRED=2, TM_VP8_PRED=9, VERT_PRED=0, HOR_PRED=1,
    // DIAG_DOWN_RIGHT_PRED=4, VERT_RIGHT_PRED=5, DIAG_DOWN_LEFT_PRED=3,
    // VERT_LEFT_PRED=7, HOR_DOWN_PRED=6, HOR_UP_PRED=8.
    let mut sum = c[0] + c[1] + c[3] + c[4] + c[5] + c[6] + c[7] + c[8] + c[9];
    adapt_prob(&mut pp[0], c[2], sum, 20, 128); // DC_PRED
    sum -= c[9];
    adapt_prob(&mut pp[1], c[9], sum, 20, 128); // TM_VP8_PRED
    sum -= c[0];
    adapt_prob(&mut pp[2], c[0], sum, 20, 128); // VERT_PRED
    let s2 = c[1] + c[4] + c[5];
    sum -= s2;
    adapt_prob(&mut pp[3], s2, sum, 20, 128);
    let s2b = s2 - c[1];
    adapt_prob(&mut pp[4], c[1], s2b, 20, 128); // HOR_PRED
    adapt_prob(&mut pp[5], c[4], c[5], 20, 128); // DIAG_DOWN_RIGHT vs VERT_RIGHT
    sum -= c[3];
    adapt_prob(&mut pp[6], c[3], sum, 20, 128); // DIAG_DOWN_LEFT
    sum -= c[7];
    adapt_prob(&mut pp[7], c[7], sum, 20, 128); // VERT_LEFT
    adapt_prob(&mut pp[8], c[6], c[8], 20, 128); // HOR_DOWN vs HOR_UP
}

/// Adapt UV intra-mode probabilities (same tree structure as Y).
fn adapt_uv_mode(pp: &mut [u8; 9], c: &[u32; 10]) {
    adapt_y_mode(pp, c);
}
