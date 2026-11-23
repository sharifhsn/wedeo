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

/// Stride of the compound-prediction temporary planes (max block width).
pub const TMP_STRIDE: usize = 64;
/// Stride of the edge-emulation buffer (64 + 7 columns fit comfortably).
const EMU_STRIDE: usize = 160;
const EMU_ROWS: usize = 80;

/// Reusable motion-compensation scratch buffers (allocated once per decoder).
pub struct McScratch {
    /// Edge-emulation buffer (`EMU_STRIDE` x `EMU_ROWS`).
    pub emu: Vec<u8>,
    /// Second-reference luma prediction for compound blocks.
    pub tmp_y: Vec<u8>,
    /// Second-reference U prediction for compound blocks.
    pub tmp_u: Vec<u8>,
    /// Second-reference V prediction for compound blocks.
    pub tmp_v: Vec<u8>,
}

impl McScratch {
    /// Allocate the scratch buffers.
    pub fn new() -> Self {
        Self {
            emu: vec![0u8; EMU_STRIDE * EMU_ROWS],
            tmp_y: vec![0u8; TMP_STRIDE * 64],
            tmp_u: vec![0u8; TMP_STRIDE * 64],
            tmp_v: vec![0u8; TMP_STRIDE * 64],
        }
    }
}

impl Default for McScratch {
    fn default() -> Self {
        Self::new()
    }
}

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
            emu_buf, EMU_STRIDE, ref_plane, ref_stride, emu_w, emu_h, emu_x, emu_y, ref_w, ref_h,
        );

        // The emu buffer starts at (emu_x, emu_y) = (integer - 3, y - 3),
        // which is already the position our filter functions expect (they read
        // src[0..7] with center at tap[3] = integer position). No offset needed.
        // (FFmpeg adds +3 here because its FILTER_8TAP reads src[x-3..x+4].)
        mc_apply(
            dst, dst_stride, emu_buf, EMU_STRIDE, bw, bh, mx_frac, my_frac, filter_idx, true,
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
            emu_buf, EMU_STRIDE, ref_u, ref_stride, emu_w, emu_h, emu_x, emu_y, ref_w, ref_h,
        );
        mc_apply(
            dst_u, dst_stride, emu_buf, EMU_STRIDE, bw, bh, mx_frac, my_frac, filter_idx, false,
        );

        // V plane
        emu_edge(
            emu_buf, EMU_STRIDE, ref_v, ref_stride, emu_w, emu_h, emu_x, emu_y, ref_w, ref_h,
        );
        mc_apply(
            dst_v, dst_stride, emu_buf, EMU_STRIDE, bw, bh, mx_frac, my_frac, filter_idx, false,
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
    scratch: &mut McScratch,
) {
    let McScratch {
        emu: emu_buf,
        tmp_y,
        tmp_u,
        tmp_v,
    } = scratch;
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

    // --- Luma MC for ref[0] ---
    if bs > BlockSize::Bs8x8 as usize {
        // Sub-8x8: per-sub-block MC.
        let dst_off = row_px * fb.y_stride + col_px;
        sub8x8_luma_mc_buf(
            &mut fb.y[dst_off..],
            fb.y_stride,
            block,
            ref0,
            col_px,
            row_px,
            filter,
            emu_buf,
            0,
        );
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
            emu_buf,
        );
    }

    // --- Chroma MC for ref[0] ---
    let uv_bw4 = BWH_TAB[usize::from(ss_h)][bs][0] as usize;
    let uv_bh4 = BWH_TAB[usize::from(ss_v)][bs][1] as usize;
    let uv_bw = uv_bw4 * 4;
    let uv_bh = uv_bh4 * 4;
    let uv_col = col_px >> usize::from(ss_h);
    let uv_row = row_px >> usize::from(ss_v);
    let ref_uv_w = (ref0.width as usize).div_ceil(1 << usize::from(ss_h));
    let ref_uv_h = (ref0.height as usize).div_ceil(1 << usize::from(ss_v));

    if bs > BlockSize::Bs8x8 as usize {
        let u_off = uv_row * fb.uv_stride + uv_col;
        sub8x8_chroma_mc_dst(
            &mut fb.u[u_off..],
            &mut fb.v[u_off..],
            fb.uv_stride,
            block,
            ref0,
            uv_col,
            uv_row,
            ref_uv_w,
            ref_uv_h,
            filter,
            ss_h,
            ss_v,
            emu_buf,
            0,
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
            emu_buf,
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

        // Luma ref1 — per-sub-block for sub-8x8, single MC for >=8x8.
        if bs > BlockSize::Bs8x8 as usize {
            sub8x8_luma_mc_buf(
                tmp_y, TMP_STRIDE, block, ref1, col_px, row_px, filter, emu_buf,
                1, // ref_idx=1
            );
        } else {
            mc_luma(
                tmp_y,
                TMP_STRIDE,
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
                emu_buf,
            );
        }
        // Average luma.
        let dst_off = row_px * fb.y_stride + col_px;
        avg_block(&mut fb.y[dst_off..], fb.y_stride, tmp_y, TMP_STRIDE, bw, bh);

        // Chroma ref1 — per-sub-block averaging for sub-8x8.
        let ref1_uv_w = (ref1.width as usize).div_ceil(1 << ss_h as usize);
        let ref1_uv_h = (ref1.height as usize).div_ceil(1 << ss_v as usize);
        if bs > BlockSize::Bs8x8 as usize {
            sub8x8_chroma_mc_dst(
                tmp_u, tmp_v, TMP_STRIDE, block, ref1, uv_col, uv_row, ref1_uv_w, ref1_uv_h,
                filter, ss_h, ss_v, emu_buf, 1, // ref_idx=1
            );
        } else {
            mc_chroma(
                tmp_u,
                tmp_v,
                TMP_STRIDE,
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
                emu_buf,
            );
        }
        let u_off = uv_row * fb.uv_stride + uv_col;
        avg_block(
            &mut fb.u[u_off..],
            fb.uv_stride,
            tmp_u,
            TMP_STRIDE,
            uv_bw,
            uv_bh,
        );
        avg_block(
            &mut fb.v[u_off..],
            fb.uv_stride,
            tmp_v,
            TMP_STRIDE,
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
            for (sub, mv_pair) in block.mv.iter().take(2).enumerate() {
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
                    &mv_pair[ri],
                    4,
                    8,
                    filter,
                    emu_buf,
                );
            }
        }
        BlockSize::Bs4x4 => {
            for (sub, mv_pair) in block.mv.iter().enumerate() {
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
                    &mv_pair[ri],
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

/// MC chroma for sub-8x8 blocks into `(dst_u, dst_v)` for reference `ri` (0 or 1).
#[allow(clippy::too_many_arguments)]
fn sub8x8_chroma_mc_dst(
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    dst_stride: usize,
    block: &BlockInfo,
    ref_frame: &RefFrame,
    uv_col: usize,
    uv_row: usize,
    ref_uv_w: usize,
    ref_uv_h: usize,
    filter: u8,
    ss_h: bool,
    ss_v: bool,
    emu_buf: &mut [u8],
    ri: usize,
) {
    let mut call = |dx: usize, dy: usize, w: usize, h: usize, mv: &[i16; 2]| {
        let off = dy * dst_stride + dx;
        mc_chroma(
            &mut dst_u[off..],
            &mut dst_v[off..],
            dst_stride,
            &ref_frame.fb.u,
            &ref_frame.fb.v,
            ref_frame.fb.uv_stride,
            ref_uv_w,
            ref_uv_h,
            uv_row + dy,
            uv_col + dx,
            mv,
            w,
            h,
            filter,
            ss_h,
            ss_v,
            emu_buf,
        );
    };

    match block.bs {
        BlockSize::Bs8x4 => {
            let bw = 8 >> usize::from(ss_h);
            if ss_v {
                let mv = rounded_div_mv_x2(block.mv[0][ri], block.mv[2][ri]);
                call(0, 0, bw, 4, &mv);
            } else {
                call(0, 0, bw, 4, &block.mv[0][ri]);
                let mv2 = if !ss_h {
                    block.mv[2][ri]
                } else {
                    rounded_div_mv_x2(block.mv[0][ri], block.mv[2][ri])
                };
                call(0, 4, bw, 4, &mv2);
            }
        }
        BlockSize::Bs4x8 => {
            let bh = 8 >> usize::from(ss_v);
            if ss_h {
                let mv = rounded_div_mv_x2(block.mv[0][ri], block.mv[1][ri]);
                call(0, 0, 4, bh, &mv);
            } else {
                call(0, 0, 4, bh, &block.mv[0][ri]);
                call(4, 0, 4, bh, &block.mv[1][ri]);
            }
        }
        BlockSize::Bs4x4 => match (ss_h, ss_v) {
            (true, true) => {
                let mv = rounded_div_mv_x4(
                    block.mv[0][ri],
                    block.mv[1][ri],
                    block.mv[2][ri],
                    block.mv[3][ri],
                );
                call(0, 0, 4, 4, &mv);
            }
            (false, true) => {
                let mv0 = rounded_div_mv_x2(block.mv[0][ri], block.mv[2][ri]);
                let mv1 = rounded_div_mv_x2(block.mv[1][ri], block.mv[3][ri]);
                call(0, 0, 4, 4, &mv0);
                call(4, 0, 4, 4, &mv1);
            }
            (true, false) => {
                let mv0 = rounded_div_mv_x2(block.mv[0][ri], block.mv[1][ri]);
                let mv1 = rounded_div_mv_x2(block.mv[1][ri], block.mv[2][ri]);
                call(0, 0, 4, 4, &mv0);
                call(0, 4, 4, 4, &mv1);
            }
            (false, false) => {
                for (sub, mv_pair) in block.mv.iter().enumerate() {
                    let dx = (sub & 1) * 4;
                    let dy = (sub >> 1) * 4;
                    call(dx, dy, 4, 4, &mv_pair[ri]);
                }
            }
        },
        _ => {}
    }
}
