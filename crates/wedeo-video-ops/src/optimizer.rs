//! Faithful Rust port of FFmpeg `n9.1` `libswscale/ops_optimizer.c` algebraic passes.

use wedeo_core::{AlphaMode, ChromaLocation, CropRect, Subsampling, TransferCharacteristic};

use crate::color::{canonical_transfer, fuse_color_matrices, transfers_equivalent};
use crate::ops::{DisplayTransform, SwsOp, SwsOpList, ToneMapOp, layout_shift};

const MAT_EPS: f32 = 1e-5;
const MAX_OPT_ITERS: usize = 64;

#[inline]
pub fn is_identity_swizzle(s: &[u8; 4]) -> bool {
    *s == [0, 1, 2, 3]
}

#[inline]
pub fn is_identity_matrix(m: &[[f32; 4]; 3]) -> bool {
    for (i, row) in m.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (val - expected).abs() > MAT_EPS {
                return false;
            }
        }
    }
    true
}

#[inline]
fn compose_swizzle(first: [u8; 4], second: [u8; 4]) -> [u8; 4] {
    [
        first[second[0] as usize],
        first[second[1] as usize],
        first[second[2] as usize],
        first[second[3] as usize],
    ]
}

#[inline]
fn can_fold_swizzle_with_matrix(s: &[u8; 4]) -> bool {
    s[3] == 3 && s[0] < 3 && s[1] < 3 && s[2] < 3
}

/// Fold `Swizzle(s)` followed by `ColorMatrix(m)` into a single `ColorMatrix`.
fn fold_swizzle_into_matrix_left(s: [u8; 4], m: &[[f32; 4]; 3]) -> [[f32; 4]; 3] {
    let mut out = [[0.0f32; 4]; 3];
    for i in 0..3 {
        for k in 0..3 {
            let src_idx = s[k] as usize;
            out[i][src_idx] += m[i][k];
        }
        out[i][3] = m[i][3];
    }
    out
}

/// Fold `ColorMatrix(m)` followed by `Swizzle(s)` into a single `ColorMatrix`.
fn fold_swizzle_into_matrix_right(m: &[[f32; 4]; 3], s: [u8; 4]) -> [[f32; 4]; 3] {
    [m[s[0] as usize], m[s[1] as usize], m[s[2] as usize]]
}

fn current_dims_before(list: &SwsOpList, idx: usize) -> (u32, u32) {
    let (cw, ch) = list.src.crop.display_size(list.src.width, list.src.height);
    let (mut w, mut h) = DisplayTransform::from_matrix(&list.src.display_matrix)
        .map(|dt| dt.output_dims(cw, ch))
        .unwrap_or((cw, ch));
    for op in list.ops.iter().take(idx) {
        match *op {
            SwsOp::ScaleH { dst_w, .. } => w = dst_w,
            SwsOp::ScaleV { dst_h, .. } => h = dst_h,
            _ => {}
        }
    }
    (w, h)
}

/// Optimize `list.ops` in-place using FFmpeg `n9.1` `ops_optimizer.c` algebraic rules.
pub fn optimize(list: &mut SwsOpList) {
    for _ in 0..MAX_OPT_ITERS {
        if !optimize_step(list) {
            break;
        }
    }
}

fn is_exact_before_dither(list: &SwsOpList, dither_idx: usize) -> bool {
    let dst_bits = list.dst.layout.depth.bits();
    if matches!(
        list.src.layout.depth,
        wedeo_core::SampleDepth::F32 | wedeo_core::SampleDepth::F16
    ) {
        return false;
    }
    if list.src.layout.depth.bits() > dst_bits {
        return false;
    }
    for op in &list.ops[..dither_idx] {
        match op {
            SwsOp::Read { .. } | SwsOp::Unpack { .. } | SwsOp::Swizzle(_) => {}
            _ => return false,
        }
    }
    true
}

fn optimize_step(list: &mut SwsOpList) -> bool {
    let src_has_alpha = !matches!(list.src.layout.alpha, AlphaMode::None);
    let dst_has_alpha = !matches!(list.dst.layout.alpha, AlphaMode::None);

    // 1. Push ScaleH / ScaleV leftwards across Swizzle and ColorMatrix
    //    (FFmpeg ops_optimizer.c: push filters towards input to unblock linear fusion).
    for i in 1..list.ops.len() {
        let is_filter = matches!(list.ops[i], SwsOp::ScaleH { .. } | SwsOp::ScaleV { .. });
        if is_filter && matches!(list.ops[i - 1], SwsOp::Swizzle(_) | SwsOp::ColorMatrix(_)) {
            list.ops.swap(i - 1, i);
            return true;
        }
    }

    // 2. If only ScaleH/ScaleV (and/or BlendAlpha) sit between Linearize(a) and Delinearize(b)
    //    where transfers_equivalent(a, b), commute ScaleH/ScaleV past Delinearize so the
    //    inverse transfer curves cancel (matching FFmpeg format.c same-transfer resize).
    for i in 0..list.ops.len() {
        if let SwsOp::Linearize(trc_a) = list.ops[i] {
            let mut j = i + 1;
            while j < list.ops.len()
                && matches!(
                    list.ops[j],
                    SwsOp::ScaleH { .. } | SwsOp::ScaleV { .. } | SwsOp::BlendAlpha
                )
            {
                j += 1;
            }
            if j > i + 1
                && j < list.ops.len()
                && let SwsOp::Delinearize(trc_b) = list.ops[j]
                && transfers_equivalent(trc_a, trc_b)
            {
                // Move Delinearize right next to Linearize (if BlendAlpha is present, keep
                // BlendAlpha after Delinearize? No, BlendAlpha works in either domain on premul,
                // wait: straight BlendAlpha over black `rgb * a` is only linear if done in linear,
                // so only commute across ScaleH/ScaleV when no BlendAlpha sits inside).
                let has_blend = list.ops[i + 1..j]
                    .iter()
                    .any(|op| matches!(op, SwsOp::BlendAlpha));
                if !has_blend {
                    let delin = list.ops.remove(j);
                    list.ops.insert(i + 1, delin);
                    return true;
                }
            }
        }
    }

    // 3. Single-op no-op elimination
    for i in 0..list.ops.len() {
        let remove = match &list.ops[i] {
            SwsOp::Swizzle(s) => is_identity_swizzle(s),
            SwsOp::ColorMatrix(m) => is_identity_matrix(m),
            SwsOp::Linearize(trc) | SwsOp::Delinearize(trc) => {
                canonical_transfer(*trc) == TransferCharacteristic::Linear
            }
            SwsOp::ToneMap(ToneMapOp::None) => true,
            SwsOp::ScaleH { dst_w, .. } => {
                let (cur_w, _) = current_dims_before(list, i);
                *dst_w == cur_w
            }
            SwsOp::ScaleV { dst_h, .. } => {
                let (_, cur_h) = current_dims_before(list, i);
                *dst_h == cur_h
            }
            SwsOp::Dither(crate::ops::DitherMode::None) => true,
            SwsOp::Dither(_) => is_exact_before_dither(list, i),
            SwsOp::BlendAlpha => !src_has_alpha,
            _ => false,
        };
        if remove {
            list.ops.remove(i);
            return true;
        }
    }

    // 4. Pairwise fusions & cancellations
    for i in 0..list.ops.len().saturating_sub(1) {
        match (&list.ops[i], &list.ops[i + 1]) {
            // Fuse adjacent ColorMatrix ops: B * A
            (SwsOp::ColorMatrix(a), SwsOp::ColorMatrix(b)) => {
                let fused = fuse_color_matrices(a, b);
                list.ops[i] = SwsOp::ColorMatrix(fused);
                list.ops.remove(i + 1);
                return true;
            }
            // Cancel inverse Linearize / Delinearize
            (SwsOp::Linearize(a), SwsOp::Delinearize(b))
            | (SwsOp::Delinearize(a), SwsOp::Linearize(b))
                if transfers_equivalent(*a, *b) =>
            {
                list.ops.remove(i + 1);
                list.ops.remove(i);
                return true;
            }
            // Compose adjacent Swizzles
            (SwsOp::Swizzle(a), SwsOp::Swizzle(b)) => {
                let composed = compose_swizzle(*a, *b);
                list.ops[i] = SwsOp::Swizzle(composed);
                list.ops.remove(i + 1);
                return true;
            }
            // Fold Swizzle into ColorMatrix
            (SwsOp::Swizzle(s), SwsOp::ColorMatrix(m)) if can_fold_swizzle_with_matrix(s) => {
                let folded = fold_swizzle_into_matrix_left(*s, m);
                list.ops[i] = SwsOp::ColorMatrix(folded);
                list.ops.remove(i + 1);
                return true;
            }
            (SwsOp::ColorMatrix(m), SwsOp::Swizzle(s)) if can_fold_swizzle_with_matrix(s) => {
                let folded = fold_swizzle_into_matrix_right(m, *s);
                list.ops[i] = SwsOp::ColorMatrix(folded);
                list.ops.remove(i + 1);
                return true;
            }
            // Cancel directly adjacent Unpack + Pack when depth, shift, and alpha presence match
            (SwsOp::Unpack { depth, shift }, SwsOp::Pack(dst_layout))
                if *depth == dst_layout.depth
                    && *shift == layout_shift(*dst_layout)
                    && (src_has_alpha || !dst_has_alpha) =>
            {
                list.ops.remove(i + 1);
                list.ops.remove(i);
                return true;
            }
            _ => {}
        }
    }

    // 5. Cancel Unpack + Swizzle + Pack when depth, shift, and alpha presence match
    for i in 0..list.ops.len().saturating_sub(2) {
        if let (SwsOp::Unpack { depth, shift }, SwsOp::Swizzle(_), SwsOp::Pack(dst_layout)) =
            (&list.ops[i], &list.ops[i + 1], &list.ops[i + 2])
            && *depth == dst_layout.depth
            && *shift == layout_shift(*dst_layout)
            && (src_has_alpha || !dst_has_alpha)
        {
            list.ops.remove(i + 2);
            list.ops.remove(i);
            return true;
        }
    }

    // 6. Special case: Gray -> Gray with identical model/range collapses unused chroma rows
    //    (decode_to_rgb_matrix broadcasts Y to [Y,Y,Y], encode_from_rgb_matrix computes Y = kr*Y + kg*Y + kb*Y = Y).
    //    After ColorMatrix fusion, row 0 is [1, 0, 0, 0] while rows 1,2 are [0,0,0,0.5] (unused for Gray output).
    //    Recognize this when dst is Gray and replace with identity so Unpack+Pack can also cancel.
    if list.dst.layout.model == wedeo_core::ColorModel::Gray {
        for i in 0..list.ops.len() {
            if let SwsOp::ColorMatrix(m) = list.ops[i]
                && (m[0][0] - 1.0).abs() < MAT_EPS
                && m[0][1].abs() < MAT_EPS
                && m[0][2].abs() < MAT_EPS
                && m[0][3].abs() < MAT_EPS
            {
                list.ops.remove(i);
                return true;
            }
        }
    }

    false
}

/// Returns `true` if `list` is a pure zero-copy identity pass-through
/// (`ff_sws_op_list_is_noop` from FFmpeg `n9.1` `ops.c`).
pub fn is_noop(list: &SwsOpList) -> bool {
    if list.ops.len() != 2 {
        return false;
    }
    let Some(SwsOp::Read {
        layout: read_layout,
        crop,
        display_matrix,
    }) = list.ops.first()
    else {
        return false;
    };
    let Some(SwsOp::Write(write_layout)) = list.ops.last() else {
        return false;
    };

    if read_layout != write_layout {
        return false;
    }
    if *crop != CropRect::ZERO {
        return false;
    }
    if !DisplayTransform::from_matrix(display_matrix).is_ok_and(|dt| dt.is_identity()) {
        return false;
    }
    if list.src.width != list.dst.width
        || list.src.height != list.dst.height
        || list.src.domain != list.dst.domain
    {
        return false;
    }

    let is_subsampled = !matches!(
        list.src.layout.subsampling,
        Subsampling::Cs444 | Subsampling::Cs400
    );
    if is_subsampled {
        let sl = list.src.color.chroma_location;
        let dl = list.dst.color.chroma_location;
        if sl != ChromaLocation::Unspecified && dl != ChromaLocation::Unspecified && sl != dl {
            return false;
        }
    }

    true
}
