// VP9 motion compensation.
//
// 8-tap subpel interpolation, edge emulation, and inter prediction dispatch.
// Translated from FFmpeg's vp9recon.c (mc_luma_unscaled, mc_chroma_unscaled),
// vp9_mc_template.c (inter_pred), and vp9dsp_template.c (8-tap filter kernels).
//
// Only 8-bit unscaled prediction is implemented. Scaled prediction (when ref
// dimensions differ from current frame) returns an error.
//
// LGPL-2.1-or-later — same licence as FFmpeg.

use std::sync::Arc;

use crate::block::BlockInfo;
use crate::data::{BWH_TAB, VP9_SUBPEL_FILTERS};
use crate::recon::FrameBuffer;
use crate::refs::RefFrame;
use crate::types::BlockSize;

// ---------------------------------------------------------------------------
// 8-tap subpel filter application
// ---------------------------------------------------------------------------

/// Apply an 8-tap filter horizontally (put, not avg).
///
/// `src` must have at least `w + 7` accessible columns (3 before, 4 after).
/// The filter operates on `src[x - 3 .. x + 4]` for each output pixel.
fn filter_h_8tap(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    filter: &[i16; 8],
) {
    for y in 0..h {
        let src_row = &src[y * src_stride..];
        let dst_row = &mut dst[y * dst_stride..];
        for x in 0..w {
            let val = filter[0] as i32 * src_row[x] as i32     // x + (-3) + 3 = x
                + filter[1] as i32 * src_row[x + 1] as i32
                + filter[2] as i32 * src_row[x + 2] as i32
                + filter[3] as i32 * src_row[x + 3] as i32
                + filter[4] as i32 * src_row[x + 4] as i32
                + filter[5] as i32 * src_row[x + 5] as i32
                + filter[6] as i32 * src_row[x + 6] as i32
                + filter[7] as i32 * src_row[x + 7] as i32;
            dst_row[x] = ((val + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

/// Apply an 8-tap filter vertically (put, not avg).
///
/// `src` must have at least `h + 7` accessible rows (3 before, 4 after).
fn filter_v_8tap(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    filter: &[i16; 8],
) {
    for y in 0..h {
        let dst_row = &mut dst[y * dst_stride..];
        for x in 0..w {
            let val = filter[0] as i32 * src[y * src_stride + x] as i32
                + filter[1] as i32 * src[(y + 1) * src_stride + x] as i32
                + filter[2] as i32 * src[(y + 2) * src_stride + x] as i32
                + filter[3] as i32 * src[(y + 3) * src_stride + x] as i32
                + filter[4] as i32 * src[(y + 4) * src_stride + x] as i32
                + filter[5] as i32 * src[(y + 5) * src_stride + x] as i32
                + filter[6] as i32 * src[(y + 6) * src_stride + x] as i32
                + filter[7] as i32 * src[(y + 7) * src_stride + x] as i32;
            dst_row[x] = ((val + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

/// Apply an 8-tap filter in both dimensions (2-pass: horizontal into i16
/// intermediate with stride 64, then vertical to u8).
///
/// `src` must have `(h + 7)` rows and `(w + 7)` cols of accessible data
/// (3 pixels of margin on each side).
#[allow(clippy::too_many_arguments)]
fn filter_hv_8tap(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    h_filter: &[i16; 8],
    v_filter: &[i16; 8],
) {
    // Horizontal pass into i16 intermediate (stride = 64, height = h + 7).
    let tmp_h = h + 7;
    let mut tmp = [0i16; 64 * 71]; // 64 * (64 + 7) max
    for y in 0..tmp_h {
        let src_row = &src[y * src_stride..];
        for x in 0..w {
            let val = h_filter[0] as i32 * src_row[x] as i32
                + h_filter[1] as i32 * src_row[x + 1] as i32
                + h_filter[2] as i32 * src_row[x + 2] as i32
                + h_filter[3] as i32 * src_row[x + 3] as i32
                + h_filter[4] as i32 * src_row[x + 4] as i32
                + h_filter[5] as i32 * src_row[x + 5] as i32
                + h_filter[6] as i32 * src_row[x + 6] as i32
                + h_filter[7] as i32 * src_row[x + 7] as i32;
            // Store as i16 (no clipping yet; that happens after vertical pass).
            // FFmpeg: FILTER_8TAP returns clipped pixel, but the C template
            // stores into `int16_t tmp[]` without clipping for the 2D case.
            // Actually in the C code tmp_ptr[x] = FILTER_8TAP which clips to pixel.
            // For 8-bit, pixel = uint8_t, so the clip is to [0, 255].
            // But the intermediate value is stored as int16_t for the vertical pass.
            tmp[y * 64 + x] = ((val + 64) >> 7).clamp(0, 255) as i16;
        }
    }

    // Vertical pass from i16 intermediate to u8 output.
    for y in 0..h {
        let dst_row = &mut dst[y * dst_stride..];
        for x in 0..w {
            let val = v_filter[0] as i32 * tmp[y * 64 + x] as i32
                + v_filter[1] as i32 * tmp[(y + 1) * 64 + x] as i32
                + v_filter[2] as i32 * tmp[(y + 2) * 64 + x] as i32
                + v_filter[3] as i32 * tmp[(y + 3) * 64 + x] as i32
                + v_filter[4] as i32 * tmp[(y + 4) * 64 + x] as i32
                + v_filter[5] as i32 * tmp[(y + 5) * 64 + x] as i32
                + v_filter[6] as i32 * tmp[(y + 6) * 64 + x] as i32
                + v_filter[7] as i32 * tmp[(y + 7) * 64 + x] as i32;
            dst_row[x] = ((val + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

/// Simple pixel copy (integer-pel, no filtering).
fn copy_block(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        dst[y * dst_stride..y * dst_stride + w]
            .copy_from_slice(&src[y * src_stride..y * src_stride + w]);
    }
}

/// Average dst with src: `dst[i] = (dst[i] + src[i] + 1) >> 1`.
fn avg_block(dst: &mut [u8], dst_stride: usize, src: &[u8], src_stride: usize, w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            let d = dst[y * dst_stride + x] as u16;
            let s = src[y * src_stride + x] as u16;
            dst[y * dst_stride + x] = ((d + s + 1) >> 1) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Edge emulation
// ---------------------------------------------------------------------------

/// Emulated edge MC: copy a block from `src` with clamped border extension.
///
/// Translated from `ff_emulated_edge_mc_template` in videodsp_template.c.
/// `src_x`, `src_y` may be negative (the copy window extends before the frame).
#[allow(clippy::too_many_arguments)]
fn emu_edge(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
    src_x: isize,
    src_y: isize,
    frame_w: usize,
    frame_h: usize,
) {
    for y in 0..block_h {
        let sy = (src_y + y as isize).clamp(0, frame_h as isize - 1) as usize;
        for x in 0..block_w {
            let sx = (src_x + x as isize).clamp(0, frame_w as isize - 1) as usize;
            dst[y * dst_stride + x] = src[sy * src_stride + sx];
        }
    }
}

// ---------------------------------------------------------------------------
// Per-block MC
// ---------------------------------------------------------------------------

/// MC for one luma block (unscaled, 8-bit).
///
/// Translated from `mc_luma_unscaled` in vp9recon.c.
///
/// * `row_px`, `col_px` — pixel position of the block's top-left corner.
/// * `mv` — `[x, y]` in 1/8-pel units.
/// * `bw`, `bh` — block dimensions in pixels.
/// * `filter_idx` — 0=Smooth, 1=Regular, 2=Sharp.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma(
    dst: &mut [u8],
    dst_stride: usize,
    ref_plane: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    row_px: usize,
    col_px: usize,
    mv: &[i16; 2],
    bw: usize,
    bh: usize,
    filter_idx: u8,
    emu_buf: &mut [u8],
) {
    let mx = mv[0] as isize;
    let my = mv[1] as isize;

    let x = col_px as isize + (mx >> 3);
    let y = row_px as isize + (my >> 3);
    let mx_frac = (mx & 7) as usize;
    let my_frac = (my & 7) as usize;

    // The 8-tap filter needs 3 pixels before and 4 after the block in each
    // dimension.  Use emu_edge when any filter tap would fall outside the
    // reference frame (matches FFmpeg mc_luma_unscaled / mc_chroma_unscaled).
    // FFmpeg uses !!my * 5 for the vertical bound check (not 4).
    let need_emu = x < (mx_frac != 0) as isize * 3
        || y < (my_frac != 0) as isize * 3
        || x + (mx_frac != 0) as isize * 4 + bw as isize > ref_w as isize
        || y + (my_frac != 0) as isize * 5 + bh as isize > ref_h as isize;

    if need_emu {
        let emu_w = bw + if mx_frac != 0 { 7 } else { 0 };
        let emu_h = bh + if my_frac != 0 { 7 } else { 0 };
        let emu_x = x - if mx_frac != 0 { 3 } else { 0 };
        let emu_y = y - if my_frac != 0 { 3 } else { 0 };
        emu_edge(
            emu_buf, 160, ref_plane, ref_stride, emu_w, emu_h, emu_x, emu_y, ref_w, ref_h,
        );

        // The emu buffer starts at (emu_x, emu_y) = (integer - 3, y - 3),
        // which is already the position our filter functions expect (they read
        // src[0..7] with center at tap[3] = integer position). No offset needed.
        // (FFmpeg adds +3 here because its FILTER_8TAP reads src[x-3..x+4].)
        mc_apply(
            dst, dst_stride, emu_buf, 160, bw, bh, mx_frac, my_frac, filter_idx, true,
        );
    } else {
        // Position src 3 rows above and 3 pixels left of the block start
        // so the 8-tap filter center (tap[3]) lands on the block.
        let offset_y = y as usize - if my_frac != 0 { 3 } else { 0 };
        let offset_x = x as usize - if mx_frac != 0 { 3 } else { 0 };
        let src_off = offset_y * ref_stride + offset_x;
        let src = &ref_plane[src_off..];
        mc_apply(
            dst, dst_stride, src, ref_stride, bw, bh, mx_frac, my_frac, filter_idx, true,
        );
    }
}

/// MC for one chroma block pair (U+V, unscaled, 8-bit).
///
/// Translated from `mc_chroma_unscaled` in vp9recon.c.
/// Chroma MV uses 4-bit subpel: `mx = mv.x * (1 << !ss_h) & 15`.
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma(
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    dst_stride: usize,
    ref_u: &[u8],
    ref_v: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    row_px: usize,
    col_px: usize,
    mv: &[i16; 2],
    bw: usize,
    bh: usize,
    filter_idx: u8,
    ss_h: bool,
    ss_v: bool,
    emu_buf: &mut [u8],
) {
    // Chroma MV: multiply by 2 if not subsampled in that dimension.
    let mx = mv[0] as isize * (1 << (!ss_h as isize));
    let my = mv[1] as isize * (1 << (!ss_v as isize));

    let x = col_px as isize + (mx >> 4);
    let y = row_px as isize + (my >> 4);
    let mx_frac = (mx & 15) as usize;
    let my_frac = (my & 15) as usize;

    // FFmpeg uses !!my * 5 for the vertical bound check (not 4).
    let need_emu = x < (mx_frac != 0) as isize * 3
        || y < (my_frac != 0) as isize * 3
        || x + (mx_frac != 0) as isize * 4 + bw as isize > ref_w as isize
        || y + (my_frac != 0) as isize * 5 + bh as isize > ref_h as isize;

    if need_emu {
        let emu_w = bw + if mx_frac != 0 { 7 } else { 0 };
        let emu_h = bh + if my_frac != 0 { 7 } else { 0 };
        let emu_x = x - if mx_frac != 0 { 3 } else { 0 };
        let emu_y = y - if my_frac != 0 { 3 } else { 0 };

        // U plane — emu buffer starts at (x-3, y-3), no offset needed
        // (see mc_luma comment for why).
        emu_edge(
            emu_buf, 160, ref_u, ref_stride, emu_w, emu_h, emu_x, emu_y, ref_w, ref_h,
        );
        mc_apply(
            dst_u, dst_stride, emu_buf, 160, bw, bh, mx_frac, my_frac, filter_idx, false,
        );

        // V plane
        emu_edge(
            emu_buf, 160, ref_v, ref_stride, emu_w, emu_h, emu_x, emu_y, ref_w, ref_h,
        );
        mc_apply(
            dst_v, dst_stride, emu_buf, 160, bw, bh, mx_frac, my_frac, filter_idx, false,
        );
    } else {
        let offset_y = y as usize - if my_frac != 0 { 3 } else { 0 };
        let offset_x = x as usize - if mx_frac != 0 { 3 } else { 0 };
        let src_off = offset_y * ref_stride + offset_x;
        mc_apply(
            dst_u,
            dst_stride,
            &ref_u[src_off..],
            ref_stride,
            bw,
            bh,
            mx_frac,
            my_frac,
            filter_idx,
            false,
        );
        mc_apply(
            dst_v,
            dst_stride,
            &ref_v[src_off..],
            ref_stride,
            bw,
            bh,
            mx_frac,
            my_frac,
            filter_idx,
            false,
        );
    }
}

/// Dispatch the appropriate MC kernel based on subpel fractions.
///
/// For luma: `mx`/`my` are in [0, 7] (3-bit). Filter lookup uses `mx << 1`.
/// For chroma: `mx`/`my` are in [0, 15] (4-bit). Filter lookup uses `mx` directly.
///
/// The `luma` flag distinguishes luma (3-bit → shift left) from chroma (4-bit
/// → use directly).  This avoids the ambiguity of guessing from the value.
#[allow(clippy::too_many_arguments)]
fn mc_apply(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: usize,
    my: usize,
    filter_idx: u8,
    luma: bool,
) {
    let mx_pos = if luma { mx << 1 } else { mx };
    let my_pos = if luma { my << 1 } else { my };

    let fi = filter_idx as usize;
    let h_filter = &VP9_SUBPEL_FILTERS[fi][mx_pos];
    let v_filter = &VP9_SUBPEL_FILTERS[fi][my_pos];

    if mx_pos == 0 && my_pos == 0 {
        copy_block(dst, dst_stride, src, src_stride, w, h);
    } else if my_pos == 0 {
        filter_h_8tap(dst, dst_stride, src, src_stride, w, h, h_filter);
    } else if mx_pos == 0 {
        filter_v_8tap(dst, dst_stride, src, src_stride, w, h, v_filter);
    } else {
        filter_hv_8tap(dst, dst_stride, src, src_stride, w, h, h_filter, v_filter);
    }
}

// ---------------------------------------------------------------------------
// Chroma MV averaging for sub-8x8 blocks
// ---------------------------------------------------------------------------

/// FFmpeg's ROUNDED_DIV: round away from zero.
/// `(a >= 0 ? a + (b >> 1) : a - (b >> 1)) / b`
#[inline]
fn rounded_div(a: i32, b: i32) -> i16 {
    ((if a >= 0 { a + (b >> 1) } else { a - (b >> 1) }) / b) as i16
}

/// Average two MVs using ROUNDED_DIV (for 4:2:0 chroma).
fn rounded_div_mv_x2(a: [i16; 2], b: [i16; 2]) -> [i16; 2] {
    [
        rounded_div(a[0] as i32 + b[0] as i32, 2),
        rounded_div(a[1] as i32 + b[1] as i32, 2),
    ]
}

/// Average four MVs using ROUNDED_DIV (for 4:2:0 chroma with 4x4 sub-blocks).
fn rounded_div_mv_x4(a: [i16; 2], b: [i16; 2], c: [i16; 2], d: [i16; 2]) -> [i16; 2] {
    [
        rounded_div(a[0] as i32 + b[0] as i32 + c[0] as i32 + d[0] as i32, 4),
        rounded_div(a[1] as i32 + b[1] as i32 + c[1] as i32 + d[1] as i32, 4),
    ]
}

// ---------------------------------------------------------------------------
// Top-level inter prediction dispatch
// ---------------------------------------------------------------------------

/// Perform inter prediction for one decoded block.
///
/// Translated from `inter_pred` in vp9_mc_template.c (unscaled 8bpp path).
/// Handles all block sizes including sub-8x8 with chroma MV averaging.
///
/// For compound prediction: predicts ref0 into `fb`, predicts ref1 into a temp
/// buffer, then averages.
#[allow(clippy::too_many_arguments)]
pub fn inter_pred(
    fb: &mut FrameBuffer,
    block: &BlockInfo,
    ref_slots: &[Option<Arc<RefFrame>>; 8],
    ref_idx: &[u8; 3],
    ss_h: bool,
    ss_v: bool,
) {
    let bs = block.bs as usize;
    let bw4 = BWH_TAB[0][bs][0] as usize;
    let bh4 = BWH_TAB[0][bs][1] as usize;
    let bw = bw4 * 4;
    let bh = bh4 * 4;
    let col_px = block.col * 4;
    let row_px = block.row * 4;

    // Resolve the reference frame for ref[0].
    let ref0_slot = ref_idx[block.ref_frame[0] as usize] as usize;
    let ref0 = match &ref_slots[ref0_slot] {
        Some(r) => r,
        None => return, // Missing ref frame — skip silently.
    };

    let filter = block.filter;

    // Allocate an edge-emulation buffer (160-byte stride, enough for 64+7 rows).
    let mut emu_buf = vec![0u8; 160 * 80];

    // --- Luma MC for ref[0] ---
    if bs > BlockSize::Bs8x8 as usize {
        // Sub-8x8: per-sub-block MC.
        sub8x8_luma_mc(fb, block, ref0, col_px, row_px, filter, &mut emu_buf);
    } else {
        // Normal block: single MV.
        let dst_off = row_px * fb.y_stride + col_px;
        mc_luma(
            &mut fb.y[dst_off..],
            fb.y_stride,
            &ref0.fb.y,
            ref0.fb.y_stride,
            ref0.width as usize,
            ref0.height as usize,
            row_px,
            col_px,
            &block.mv[0][0],
            bw,
            bh,
            filter,
            &mut emu_buf,
        );
    }

    // --- Chroma MC for ref[0] ---
    let uv_bw4 = BWH_TAB[1][bs][0] as usize;
    let uv_bh4 = BWH_TAB[1][bs][1] as usize;
    let uv_bw = uv_bw4 * 4;
    let uv_bh = uv_bh4 * 4;
    let uv_col = col_px >> ss_h as usize;
    let uv_row = row_px >> ss_v as usize;
    let ref_uv_w = (ref0.width as usize).div_ceil(1 << ss_h as usize);
    let ref_uv_h = (ref0.height as usize).div_ceil(1 << ss_v as usize);

    if bs > BlockSize::Bs8x8 as usize {
        sub8x8_chroma_mc(
            fb,
            block,
            ref0,
            uv_col,
            uv_row,
            uv_bw,
            uv_bh,
            ref_uv_w,
            ref_uv_h,
            filter,
            ss_h,
            ss_v,
            &mut emu_buf,
        );
    } else {
        let u_off = uv_row * fb.uv_stride + uv_col;
        let v_off = u_off;
        mc_chroma(
            &mut fb.u[u_off..],
            &mut fb.v[v_off..],
            fb.uv_stride,
            &ref0.fb.u,
            &ref0.fb.v,
            ref0.fb.uv_stride,
            ref_uv_w,
            ref_uv_h,
            uv_row,
            uv_col,
            &block.mv[0][0],
            uv_bw,
            uv_bh,
            filter,
            ss_h,
            ss_v,
            &mut emu_buf,
        );
    }

    // --- Compound prediction: average with ref[1] ---
    if block.comp {
        let ref1_slot = ref_idx[block.ref_frame[1] as usize] as usize;
        let ref1 = match &ref_slots[ref1_slot] {
            Some(r) => r,
            None => return,
        };

        // MC ref1 into temp buffers, then average with fb.
        let mut tmp_y = vec![0u8; fb.y_stride * bh];
        let mut tmp_u = vec![0u8; fb.uv_stride * uv_bh];
        let mut tmp_v = vec![0u8; fb.uv_stride * uv_bh];

        // Luma ref1 — per-sub-block for sub-8x8, single MC for >=8x8.
        if bs > BlockSize::Bs8x8 as usize {
            sub8x8_luma_mc_buf(
                &mut tmp_y,
                fb.y_stride,
                block,
                ref1,
                col_px,
                row_px,
                filter,
                &mut emu_buf,
                1, // ref_idx=1
            );
        } else {
            mc_luma(
                &mut tmp_y,
                fb.y_stride,
                &ref1.fb.y,
                ref1.fb.y_stride,
                ref1.width as usize,
                ref1.height as usize,
                row_px,
                col_px,
                &block.mv[0][1],
                bw,
                bh,
                filter,
                &mut emu_buf,
            );
        }
        // Average luma.
        let dst_off = row_px * fb.y_stride + col_px;
        avg_block(
            &mut fb.y[dst_off..],
            fb.y_stride,
            &tmp_y,
            fb.y_stride,
            bw,
            bh,
        );

        // Chroma ref1 — per-sub-block averaging for sub-8x8.
        let ref1_uv_w = (ref1.width as usize).div_ceil(1 << ss_h as usize);
        let ref1_uv_h = (ref1.height as usize).div_ceil(1 << ss_v as usize);
        if bs > BlockSize::Bs8x8 as usize {
            sub8x8_chroma_mc_buf(
                &mut tmp_u,
                &mut tmp_v,
                fb.uv_stride,
                block,
                ref1,
                uv_col,
                uv_row,
                uv_bw,
                uv_bh,
                ref1_uv_w,
                ref1_uv_h,
                filter,
                ss_h,
                ss_v,
                &mut emu_buf,
                1, // ref_idx=1
            );
        } else {
            mc_chroma(
                &mut tmp_u,
                &mut tmp_v,
                fb.uv_stride,
                &ref1.fb.u,
                &ref1.fb.v,
                ref1.fb.uv_stride,
                ref1_uv_w,
                ref1_uv_h,
                uv_row,
                uv_col,
                &block.mv[0][1],
                uv_bw,
                uv_bh,
                filter,
                ss_h,
                ss_v,
                &mut emu_buf,
            );
        }
        let u_off = uv_row * fb.uv_stride + uv_col;
        avg_block(
            &mut fb.u[u_off..],
            fb.uv_stride,
            &tmp_u,
            fb.uv_stride,
            uv_bw,
            uv_bh,
        );
        avg_block(
            &mut fb.v[u_off..],
            fb.uv_stride,
            &tmp_v,
            fb.uv_stride,
            uv_bw,
            uv_bh,
        );
    }
}

// ---------------------------------------------------------------------------
// Sub-8x8 block MC helpers
// ---------------------------------------------------------------------------

/// MC luma for sub-8x8 blocks into an arbitrary buffer (for compound ref[1]).
///
/// `ri` selects which reference MV to use: 0 or 1.
#[allow(clippy::too_many_arguments)]
fn sub8x8_luma_mc_buf(
    dst: &mut [u8],
    dst_stride: usize,
    block: &BlockInfo,
    ref_frame: &RefFrame,
    col_px: usize,
    row_px: usize,
    filter: u8,
    emu_buf: &mut [u8],
    ri: usize,
) {
    let bs = block.bs;
    let ref_w = ref_frame.width as usize;
    let ref_h = ref_frame.height as usize;

    match bs {
        BlockSize::Bs8x4 => {
            for sub in 0..2 {
                let sub_row = row_px + sub * 4;
                let dst_off = sub * 4 * dst_stride;
                mc_luma(
                    &mut dst[dst_off..],
                    dst_stride,
                    &ref_frame.fb.y,
                    ref_frame.fb.y_stride,
                    ref_w,
                    ref_h,
                    sub_row,
                    col_px,
                    &block.mv[sub * 2][ri],
                    8,
                    4,
                    filter,
                    emu_buf,
                );
            }
        }
        BlockSize::Bs4x8 => {
            for sub in 0..2 {
                let sub_col = col_px + sub * 4;
                let dst_off = sub * 4;
                mc_luma(
                    &mut dst[dst_off..],
                    dst_stride,
                    &ref_frame.fb.y,
                    ref_frame.fb.y_stride,
                    ref_w,
                    ref_h,
                    row_px,
                    sub_col,
                    &block.mv[sub][ri],
                    4,
                    8,
                    filter,
                    emu_buf,
                );
            }
        }
        BlockSize::Bs4x4 => {
            for sub in 0..4 {
                let sub_row = (sub >> 1) * 4;
                let sub_col = (sub & 1) * 4;
                let dst_off = sub_row * dst_stride + sub_col;
                mc_luma(
                    &mut dst[dst_off..],
                    dst_stride,
                    &ref_frame.fb.y,
                    ref_frame.fb.y_stride,
                    ref_w,
                    ref_h,
                    row_px + sub_row,
                    col_px + sub_col,
                    &block.mv[sub][ri],
                    4,
                    4,
                    filter,
                    emu_buf,
                );
            }
        }
        _ => {}
    }
}

/// MC chroma for sub-8x8 blocks into arbitrary buffers (for compound ref[1]).
///
/// `ri` selects which reference MV to use: 0 or 1.
#[allow(clippy::too_many_arguments)]
fn sub8x8_chroma_mc_buf(
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    dst_stride: usize,
    block: &BlockInfo,
    ref_frame: &RefFrame,
    uv_col: usize,
    uv_row: usize,
    uv_bw: usize,
    uv_bh: usize,
    ref_uv_w: usize,
    ref_uv_h: usize,
    filter: u8,
    ss_h: bool,
    ss_v: bool,
    emu_buf: &mut [u8],
    ri: usize,
) {
    let bs = block.bs;
    let chroma_mv = match bs {
        BlockSize::Bs8x4 if ss_v => rounded_div_mv_x2(block.mv[0][ri], block.mv[2][ri]),
        BlockSize::Bs4x8 if ss_h => rounded_div_mv_x2(block.mv[0][ri], block.mv[1][ri]),
        BlockSize::Bs4x4 if ss_h && ss_v => rounded_div_mv_x4(
            block.mv[0][ri],
            block.mv[1][ri],
            block.mv[2][ri],
            block.mv[3][ri],
        ),
        _ => block.mv[0][ri],
    };
    mc_chroma(
        dst_u,
        dst_v,
        dst_stride,
        &ref_frame.fb.u,
        &ref_frame.fb.v,
        ref_frame.fb.uv_stride,
        ref_uv_w,
        ref_uv_h,
        uv_row,
        uv_col,
        &chroma_mv,
        uv_bw,
        uv_bh,
        filter,
        ss_h,
        ss_v,
        emu_buf,
    );
}

/// MC luma for sub-8x8 blocks (per-sub-block MVs).
fn sub8x8_luma_mc(
    fb: &mut FrameBuffer,
    block: &BlockInfo,
    ref_frame: &RefFrame,
    col_px: usize,
    row_px: usize,
    filter: u8,
    emu_buf: &mut [u8],
) {
    let bs = block.bs;
    let ref_w = ref_frame.width as usize;
    let ref_h = ref_frame.height as usize;

    match bs {
        BlockSize::Bs8x4 => {
            // Two 8x4 sub-blocks stacked vertically.
            for sub in 0..2 {
                let sub_row = row_px + sub * 4;
                let dst_off = sub_row * fb.y_stride + col_px;
                mc_luma(
                    &mut fb.y[dst_off..],
                    fb.y_stride,
                    &ref_frame.fb.y,
                    ref_frame.fb.y_stride,
                    ref_w,
                    ref_h,
                    sub_row,
                    col_px,
                    &block.mv[sub * 2][0],
                    8,
                    4,
                    filter,
                    emu_buf,
                );
            }
        }
        BlockSize::Bs4x8 => {
            // Two 4x8 sub-blocks side by side.
            for sub in 0..2 {
                let sub_col = col_px + sub * 4;
                let dst_off = row_px * fb.y_stride + sub_col;
                mc_luma(
                    &mut fb.y[dst_off..],
                    fb.y_stride,
                    &ref_frame.fb.y,
                    ref_frame.fb.y_stride,
                    ref_w,
                    ref_h,
                    row_px,
                    sub_col,
                    &block.mv[sub][0],
                    4,
                    8,
                    filter,
                    emu_buf,
                );
            }
        }
        BlockSize::Bs4x4 => {
            // Four 4x4 sub-blocks.
            for sub in 0..4 {
                let sub_row = row_px + (sub >> 1) * 4;
                let sub_col = col_px + (sub & 1) * 4;
                let dst_off = sub_row * fb.y_stride + sub_col;
                mc_luma(
                    &mut fb.y[dst_off..],
                    fb.y_stride,
                    &ref_frame.fb.y,
                    ref_frame.fb.y_stride,
                    ref_w,
                    ref_h,
                    sub_row,
                    sub_col,
                    &block.mv[sub][0],
                    4,
                    4,
                    filter,
                    emu_buf,
                );
            }
        }
        _ => {}
    }
}

/// MC chroma for sub-8x8 blocks (averaged MVs for 4:2:0).
#[allow(clippy::too_many_arguments)]
fn sub8x8_chroma_mc(
    fb: &mut FrameBuffer,
    block: &BlockInfo,
    ref_frame: &RefFrame,
    uv_col: usize,
    uv_row: usize,
    uv_bw: usize,
    uv_bh: usize,
    ref_uv_w: usize,
    ref_uv_h: usize,
    filter: u8,
    ss_h: bool,
    ss_v: bool,
    emu_buf: &mut [u8],
) {
    let bs = block.bs;

    // For 4:2:0, sub-8x8 chroma uses averaged MVs from the luma sub-blocks.
    let chroma_mv = match bs {
        BlockSize::Bs8x4 if ss_v => {
            // Two 8x4 luma sub-blocks → one chroma block. Average MVs.
            rounded_div_mv_x2(block.mv[0][0], block.mv[2][0])
        }
        BlockSize::Bs4x8 if ss_h => {
            // Two 4x8 luma sub-blocks → one chroma block. Average MVs.
            rounded_div_mv_x2(block.mv[0][0], block.mv[1][0])
        }
        BlockSize::Bs4x4 if ss_h && ss_v => {
            // Four 4x4 luma sub-blocks → one chroma block. Average all 4 MVs.
            rounded_div_mv_x4(
                block.mv[0][0],
                block.mv[1][0],
                block.mv[2][0],
                block.mv[3][0],
            )
        }
        _ => {
            // No averaging needed — use the first sub-block MV.
            block.mv[0][0]
        }
    };

    let u_off = uv_row * fb.uv_stride + uv_col;
    mc_chroma(
        &mut fb.u[u_off..],
        &mut fb.v[u_off..],
        fb.uv_stride,
        &ref_frame.fb.u,
        &ref_frame.fb.v,
        ref_frame.fb.uv_stride,
        ref_uv_w,
        ref_uv_h,
        uv_row,
        uv_col,
        &chroma_mv,
        uv_bw,
        uv_bh,
        filter,
        ss_h,
        ss_v,
        emu_buf,
    );
}
