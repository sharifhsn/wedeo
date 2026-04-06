// VP9 loop filter — mask-based, per-superblock architecture.
//
// Translated from FFmpeg's libavcodec/vp9lpf.c (filter dispatch),
// libavcodec/vp9block.c (mask_edges), and libavcodec/vp9dsp_template.c
// (filter kernels).
//
// LGPL-2.1-or-later — same licence as FFmpeg.

use crate::block::BlockInfo;
use crate::data::BWH_TAB;
use crate::header::FrameHeader;
use crate::recon::FrameBuffer;
// BlockSize/TxSize imported for doc reference but types used via as usize casts.

// ---------------------------------------------------------------------------
// Filter parameter LUTs (vp9.c filter_lut computation)
// ---------------------------------------------------------------------------

/// Compute lim_lut and mblim_lut for all 64 levels.
fn build_filter_luts(sharpness: u8) -> ([u8; 64], [u8; 64]) {
    let mut lim_lut = [0u8; 64];
    let mut mblim_lut = [0u8; 64];
    for level in 1..64u8 {
        let mut limit = level as i32;
        if sharpness > 0 {
            limit >>= ((sharpness as i32) + 3) >> 2;
            limit = limit.min(9 - sharpness as i32);
        }
        limit = limit.max(1);
        lim_lut[level as usize] = limit as u8;
        mblim_lut[level as usize] = (2 * (level as i32 + 2) + limit).min(255) as u8;
    }
    (lim_lut, mblim_lut)
}

// ---------------------------------------------------------------------------
// VP9Filter — per-superblock filter state  (vp9dec.h VP9Filter)
// ---------------------------------------------------------------------------

/// Per-superblock filter state. Matches FFmpeg's `VP9Filter`.
///
/// `level[row8*8+col8]` — filter level for each 8×8 cell in the 64×64 SB.
/// `mask[plane][col_or_row][row8][class]`:
///   - plane: 0=luma, 1=chroma (shared U/V)
///   - col_or_row: 0=column (vertical edges), 1=row (horizontal edges)
///   - row8: row within superblock in 8-pixel units (0..7)
///   - class: 0=16-tap, 1=8-tap, 2=4-tap, 3=inner-4-tap
#[derive(Clone)]
struct Vp9Filter {
    level: [u8; 64],
    mask: [[[[u8; 4]; 8]; 2]; 2],
}

impl Default for Vp9Filter {
    fn default() -> Self {
        Self {
            level: [0; 64],
            mask: [[[[0; 4]; 8]; 2]; 2],
        }
    }
}

// ---------------------------------------------------------------------------
// setctx_2d — fill a 2D region of level[] with a value  (vp9block.c:34)
// ---------------------------------------------------------------------------

fn setctx_2d(level: &mut [u8; 64], col: usize, row: usize, w: usize, h: usize, val: u8) {
    for r in row..row + h {
        for c in col..col + w {
            if r < 8 && c < 8 {
                level[r * 8 + c] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mask_edges — translate from vp9block.c:1142–1261
// ---------------------------------------------------------------------------

/// Build filter masks for one block within a superblock.
///
/// `mask` is `[col_or_row][8 rows][4 classes]` for one plane.
/// Direct translation of FFmpeg's `mask_edges` (vp9block.c:1142–1261).
// Index-based loop is a direct C translation; iterator form would obscure
// the 1-to-1 correspondence with FFmpeg's mask_edges.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn mask_edges(
    mask: &mut [[[u8; 4]; 8]; 2],
    ss_h: usize,
    ss_v: usize,
    row_and_7: usize,
    col_and_7: usize,
    mut w: usize,
    mut h: usize,
    col_end: usize,
    row_end: usize,
    tx: usize, // TxSize as usize: 0=4x4, 1=8x8, 2=16x16, 3=32x32
    skip_inter: bool,
) {
    const WIDE_FILTER_COL_MASK: [u32; 2] = [0x11, 0x01];
    const WIDE_FILTER_ROW_MASK: [u32; 2] = [0x03, 0x07];

    // For UV with TX_4X4, ignore odd sub-blocks that fall on chroma boundaries.
    if tx == 0 && (ss_v | ss_h) != 0 {
        if h == ss_v {
            if row_and_7 & 1 != 0 {
                return;
            }
            if row_end == 0 {
                h += 1;
            }
        }
        if w == ss_h {
            if col_and_7 & 1 != 0 {
                return;
            }
            if col_end == 0 {
                w += 1;
            }
        }
    }

    if tx == 0 && !skip_inter {
        // TX_4X4, non-skip inter (or intra)
        let t = 1u32 << col_and_7;
        let m_col = (t << w) - t;
        let m_row_8 = m_col & WIDE_FILTER_COL_MASK[ss_h];
        let m_row_4 = m_col - m_row_8;

        for y in row_and_7..h + row_and_7 {
            let col_mask_id = 2 - usize::from(y as u32 & WIDE_FILTER_ROW_MASK[ss_v] == 0);

            mask[0][y][1] |= m_row_8 as u8;
            mask[0][y][2] |= m_row_4 as u8;

            if (ss_h & ss_v) != 0 && (col_end & 1) != 0 && (y & 1) != 0 {
                mask[1][y][col_mask_id] |= ((t << (w - 1)) - t) as u8;
            } else {
                mask[1][y][col_mask_id] |= m_col as u8;
            }
            if ss_h == 0 {
                mask[0][y][3] |= m_col as u8;
            }
            if ss_v == 0 {
                if ss_h != 0 && (col_end & 1) != 0 {
                    mask[1][y][3] |= ((t << (w - 1)) - t) as u8;
                } else {
                    mask[1][y][3] |= m_col as u8;
                }
            }
        }
    } else {
        let t = 1u32 << col_and_7;
        let m_col = (t << w) - t;

        if !skip_inter {
            let mask_id = usize::from(tx == 1); // TX_8X8
            let l2 = tx + ss_h - 1;
            const MASKS: [u32; 4] = [0xff, 0x55, 0x11, 0x01];
            let m_row = m_col & MASKS[l2];

            // Odd UV col/row edges for tx16/tx32: force 8-wide filter.
            if ss_h != 0 && tx > 1 && (w ^ (w - 1)) == 1 {
                let m_row_16 = ((t << (w - 1)) - t) & MASKS[l2];
                let m_row_8 = m_row - m_row_16;
                for y in row_and_7..h + row_and_7 {
                    mask[0][y][0] |= m_row_16 as u8;
                    mask[0][y][1] |= m_row_8 as u8;
                }
            } else {
                for y in row_and_7..h + row_and_7 {
                    mask[0][y][mask_id] |= m_row as u8;
                }
            }

            let l2 = tx + ss_v - 1;
            let step1d = 1usize << l2;
            if ss_v != 0 && tx > 1 && (h ^ (h - 1)) == 1 {
                let mut y = row_and_7;
                while y < h + row_and_7 - 1 {
                    mask[1][y][0] |= m_col as u8;
                    y += step1d;
                }
                if y - row_and_7 == h - 1 {
                    mask[1][y][1] |= m_col as u8;
                }
            } else {
                let mut y = row_and_7;
                while y < h + row_and_7 {
                    mask[1][y][mask_id] |= m_col as u8;
                    y += step1d;
                }
            }
        } else if tx != 0 {
            // skip_inter && tx != TX_4X4: only block-boundary edges
            let mask_id = usize::from(tx == 1 || h == ss_v);
            mask[1][row_and_7][mask_id] |= m_col as u8;
            let mask_id = usize::from(tx == 1 || w == ss_h);
            for y in row_and_7..h + row_and_7 {
                mask[0][y][mask_id] |= t as u8;
            }
        } else {
            // skip_inter && TX_4X4
            let t8 = t & WIDE_FILTER_COL_MASK[ss_h];
            let t4 = t - t8;
            for y in row_and_7..h + row_and_7 {
                mask[0][y][2] |= t4 as u8;
                mask[0][y][1] |= t8 as u8;
            }
            let col_mask_id = 2 - usize::from(row_and_7 as u32 & WIDE_FILTER_ROW_MASK[ss_v] == 0);
            mask[1][row_and_7][col_mask_id] |= m_col as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Filter kernels
// ---------------------------------------------------------------------------

#[inline(always)]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// 4-tap filter — vp9dsp_template.c:1862–1886.
#[inline(always)]
fn filter4(p1: u8, p0: u8, q0: u8, q1: u8, hev_thresh: u8) -> (u8, u8, u8, u8) {
    let p1i = p1 as i32;
    let p0i = p0 as i32;
    let q0i = q0 as i32;
    let q1i = q1 as i32;

    let hev = (p1i - p0i).abs() > hev_thresh as i32 || (q1i - q0i).abs() > hev_thresh as i32;

    if hev {
        let f = (p1i - q1i).clamp(-128, 127);
        let f = (3 * (q0i - p0i) + f).clamp(-128, 127);
        let f1 = (f + 4).min(127) >> 3;
        let f2 = (f + 3).min(127) >> 3;
        (p1, clip_u8(p0i + f2), clip_u8(q0i - f1), q1)
    } else {
        let f = (3 * (q0i - p0i)).clamp(-128, 127);
        let f1 = (f + 4).min(127) >> 3;
        let f2 = (f + 3).min(127) >> 3;
        let f3 = (f1 + 1) >> 1;
        (
            clip_u8(p1i + f3),
            clip_u8(p0i + f2),
            clip_u8(q0i - f1),
            clip_u8(q1i - f3),
        )
    }
}

/// 8-tap filter — vp9dsp_template.c:1856–1861.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn filter8(p3: u8, p2: u8, p1: u8, p0: u8, q0: u8, q1: u8, q2: u8, q3: u8) -> [u8; 8] {
    let p = [p0 as i32, p1 as i32, p2 as i32, p3 as i32];
    let q = [q0 as i32, q1 as i32, q2 as i32, q3 as i32];
    [
        p3,
        clip_u8((p[3] + p[3] + p[3] + 2 * p[2] + p[1] + p[0] + q[0] + 4) >> 3),
        clip_u8((p[3] + p[3] + p[2] + 2 * p[1] + p[0] + q[0] + q[1] + 4) >> 3),
        clip_u8((p[3] + p[2] + p[1] + 2 * p[0] + q[0] + q[1] + q[2] + 4) >> 3),
        clip_u8((p[2] + p[1] + p[0] + 2 * q[0] + q[1] + q[2] + q[3] + 4) >> 3),
        clip_u8((p[1] + p[0] + q[0] + 2 * q[1] + q[2] + q[3] + q[3] + 4) >> 3),
        clip_u8((p[0] + q[0] + q[1] + 2 * q[2] + q[3] + q[3] + q[3] + 4) >> 3),
        q3,
    ]
}

/// 16-tap filter — vp9dsp_template.c:1827–1854.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn filter16(
    p7: u8,
    p6: u8,
    p5: u8,
    p4: u8,
    p3: u8,
    p2: u8,
    p1: u8,
    p0: u8,
    q0: u8,
    q1: u8,
    q2: u8,
    q3: u8,
    q4: u8,
    q5: u8,
    q6: u8,
    q7: u8,
) -> [u8; 16] {
    let p = [
        p0 as i32, p1 as i32, p2 as i32, p3 as i32, p4 as i32, p5 as i32, p6 as i32, p7 as i32,
    ];
    let q = [
        q0 as i32, q1 as i32, q2 as i32, q3 as i32, q4 as i32, q5 as i32, q6 as i32, q7 as i32,
    ];
    [
        p7,
        clip_u8((p[7] * 7 + p[6] * 2 + p[5] + p[4] + p[3] + p[2] + p[1] + p[0] + q[0] + 8) >> 4),
        clip_u8(
            (p[7] * 6 + p[6] + p[5] * 2 + p[4] + p[3] + p[2] + p[1] + p[0] + q[0] + q[1] + 8) >> 4,
        ),
        clip_u8(
            (p[7] * 5
                + p[6]
                + p[5]
                + p[4] * 2
                + p[3]
                + p[2]
                + p[1]
                + p[0]
                + q[0]
                + q[1]
                + q[2]
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[7] * 4
                + p[6]
                + p[5]
                + p[4]
                + p[3] * 2
                + p[2]
                + p[1]
                + p[0]
                + q[0]
                + q[1]
                + q[2]
                + q[3]
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[7] * 3
                + p[6]
                + p[5]
                + p[4]
                + p[3]
                + p[2] * 2
                + p[1]
                + p[0]
                + q[0]
                + q[1]
                + q[2]
                + q[3]
                + q[4]
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[7] * 2
                + p[6]
                + p[5]
                + p[4]
                + p[3]
                + p[2]
                + p[1] * 2
                + p[0]
                + q[0]
                + q[1]
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[7]
                + p[6]
                + p[5]
                + p[4]
                + p[3]
                + p[2]
                + p[1]
                + p[0] * 2
                + q[0]
                + q[1]
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + q[6]
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[6]
                + p[5]
                + p[4]
                + p[3]
                + p[2]
                + p[1]
                + p[0]
                + q[0] * 2
                + q[1]
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + q[6]
                + q[7]
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[5]
                + p[4]
                + p[3]
                + p[2]
                + p[1]
                + p[0]
                + q[0]
                + q[1] * 2
                + q[2]
                + q[3]
                + q[4]
                + q[5]
                + q[6]
                + q[7] * 2
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[4]
                + p[3]
                + p[2]
                + p[1]
                + p[0]
                + q[0]
                + q[1]
                + q[2] * 2
                + q[3]
                + q[4]
                + q[5]
                + q[6]
                + q[7] * 3
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[3]
                + p[2]
                + p[1]
                + p[0]
                + q[0]
                + q[1]
                + q[2]
                + q[3] * 2
                + q[4]
                + q[5]
                + q[6]
                + q[7] * 4
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[2]
                + p[1]
                + p[0]
                + q[0]
                + q[1]
                + q[2]
                + q[3]
                + q[4] * 2
                + q[5]
                + q[6]
                + q[7] * 5
                + 8)
                >> 4,
        ),
        clip_u8(
            (p[1] + p[0] + q[0] + q[1] + q[2] + q[3] + q[4] + q[5] * 2 + q[6] + q[7] * 6 + 8) >> 4,
        ),
        clip_u8((p[0] + q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] * 2 + q[7] * 7 + 8) >> 4),
        q7,
    ]
}

/// Filter-mask check — vp9dsp_template.c:1796–1799.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn needs_filter(
    p3: i32,
    p2: i32,
    p1: i32,
    p0: i32,
    q0: i32,
    q1: i32,
    q2: i32,
    q3: i32,
    inner_limit: i32,
    outer_limit: i32,
) -> bool {
    (p3 - p2).abs() <= inner_limit
        && (p2 - p1).abs() <= inner_limit
        && (p1 - p0).abs() <= inner_limit
        && (q1 - q0).abs() <= inner_limit
        && (q2 - q1).abs() <= inner_limit
        && (q3 - q2).abs() <= inner_limit
        && 2 * (p0 - q0).abs() + ((p1 - q1).abs() >> 1) <= outer_limit
}

/// Flat8in check — vp9dsp_template.c:1821–1824.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn is_flat8in(p3: i32, p2: i32, p1: i32, p0: i32, q0: i32, q1: i32, q2: i32, q3: i32) -> bool {
    (p1 - p0).abs() <= 1
        && (q1 - q0).abs() <= 1
        && (p2 - p0).abs() <= 1
        && (q2 - q0).abs() <= 1
        && (p3 - p0).abs() <= 1
        && (q3 - q0).abs() <= 1
}

/// Flat8out check — vp9dsp_template.c:1815–1818.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn is_flat8out(
    p7: i32,
    p6: i32,
    p5: i32,
    p4: i32,
    p0: i32,
    q0: i32,
    q4: i32,
    q5: i32,
    q6: i32,
    q7: i32,
) -> bool {
    (p7 - p0).abs() <= 1
        && (p6 - p0).abs() <= 1
        && (p5 - p0).abs() <= 1
        && (p4 - p0).abs() <= 1
        && (q4 - q0).abs() <= 1
        && (q5 - q0).abs() <= 1
        && (q6 - q0).abs() <= 1
        && (q7 - q0).abs() <= 1
}

// ---------------------------------------------------------------------------
// Unified loop_filter — vp9dsp_template.c:1780–1888
//
// Filters 8 pixels along an edge. `stridea` advances to next row along the
// edge; `strideb` advances perpendicular to the edge.
// For vertical edges:   stridea = stride (next row), strideb = 1 (next col)
// For horizontal edges: stridea = 1 (next col), strideb = stride (next row)
// ---------------------------------------------------------------------------

/// Apply the unified VP9 loop filter for one 8-pixel edge.
#[allow(clippy::too_many_arguments)]
fn loop_filter_edge(
    dst: &mut [u8],
    offset: usize,
    stridea: usize, // along the edge (iterate 8 pixels)
    strideb: isize, // perpendicular (into p/q sides)
    e_val: i32,
    i_val: i32,
    h_val: i32,
    wd: usize,
) {
    let dst_len = dst.len();
    for i in 0..8 {
        let base = offset + i * stridea;
        // Bounds check: all filter paths need offsets -4..+3 (p3..q3).
        // FFmpeg relies on buffer padding and never checks bounds.
        let min_idx = base as isize - 4 * strideb;
        let max_idx = base as isize + 3 * strideb;
        if min_idx < 0 || max_idx as usize >= dst_len {
            continue;
        }
        let at = |off: isize| -> i32 { dst[(base as isize + off * strideb) as usize] as i32 };

        let p3 = at(-4);
        let p2 = at(-3);
        let p1 = at(-2);
        let p0 = at(-1);
        let q0 = at(0);
        let q1 = at(1);
        let q2 = at(2);
        let q3 = at(3);

        if !needs_filter(p3, p2, p1, p0, q0, q1, q2, q3, i_val, e_val) {
            continue;
        }

        let flat8in = if wd >= 8 {
            is_flat8in(p3, p2, p1, p0, q0, q1, q2, q3)
        } else {
            false
        };

        // For 16-tap, need offsets -8..+7 — check bounds before reading.
        let wide_ok = if wd >= 16 && flat8in {
            let min16 = base as isize - 8 * strideb;
            let max16 = base as isize + 7 * strideb;
            min16 >= 0 && (max16 as usize) < dst_len
        } else {
            false
        };

        if wide_ok {
            let p4 = at(-5);
            let p5 = at(-6);
            let p6 = at(-7);
            let p7 = at(-8);
            let q4 = at(4);
            let q5 = at(5);
            let q6 = at(6);
            let q7 = at(7);
            if is_flat8out(p7, p6, p5, p4, p0, q0, q4, q5, q6, q7) {
                let r = filter16(
                    p7 as u8, p6 as u8, p5 as u8, p4 as u8, p3 as u8, p2 as u8, p1 as u8, p0 as u8,
                    q0 as u8, q1 as u8, q2 as u8, q3 as u8, q4 as u8, q5 as u8, q6 as u8, q7 as u8,
                );
                // Write p6..p1, p0, q0, q1..q6
                // r[0]=p7 (unchanged), r[15]=q7 (unchanged)
                for (off, val) in [
                    (-7isize, r[1]),
                    (-6, r[2]),
                    (-5, r[3]),
                    (-4, r[4]),
                    (-3, r[5]),
                    (-2, r[6]),
                    (-1, r[7]),
                    (0, r[8]),
                    (1, r[9]),
                    (2, r[10]),
                    (3, r[11]),
                    (4, r[12]),
                    (5, r[13]),
                    (6, r[14]),
                ] {
                    dst[(base as isize + off * strideb) as usize] = val;
                }
                continue;
            }
        }

        if wd >= 8 && flat8in {
            let r = filter8(
                p3 as u8, p2 as u8, p1 as u8, p0 as u8, q0 as u8, q1 as u8, q2 as u8, q3 as u8,
            );
            for (off, val) in [
                (-3isize, r[1]),
                (-2, r[2]),
                (-1, r[3]),
                (0, r[4]),
                (1, r[5]),
                (2, r[6]),
            ] {
                dst[(base as isize + off * strideb) as usize] = val;
            }
        } else {
            let (np1, np0, nq0, nq1) = filter4(p1 as u8, p0 as u8, q0 as u8, q1 as u8, h_val as u8);
            for (off, val) in [(-2isize, np1), (-1, np0), (0, nq0), (1, nq1)] {
                dst[(base as isize + off * strideb) as usize] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// filter_plane_cols — vp9lpf.c:27–99
// ---------------------------------------------------------------------------

/// Filter vertical (column) edges for one plane within one superblock.
///
/// Simplified version: no mix2 batching — each stacked pair is filtered
/// as two separate 8-pixel calls.  Produces identical results, just slower.
#[allow(clippy::too_many_arguments)]
fn filter_plane_cols(
    plane: &mut [u8],
    plane_off: usize,
    stride: usize,
    sb_col: usize,
    ss_h: usize,
    ss_v: usize,
    level: &[u8; 64],
    mask: &[[u8; 4]; 8],
    lim_lut: &[u8; 64],
    mblim_lut: &[u8; 64],
) {
    // lvl_idx walks through level[64] with stride 8, stepping 2 rows per
    // outer iteration for luma (ss_v=0) or 4 rows for chroma (ss_v=1).
    let mut lvl_idx: usize = 0; // index into level[]
    let mut dst_off = plane_off; // pixel offset for current row-pair

    for y in (0..8).step_by(2 << ss_v) {
        let y2 = y + 1 + ss_v;
        let hmask1 = mask[y];
        let hmask2 = if y2 < 8 { mask[y2] } else { [0; 4] };

        let hm1 = hmask1[0] | hmask1[1] | hmask1[2];
        let hm13 = hmask1[3];
        let hm2 = hmask2[1] | hmask2[2];
        let hm23 = hmask2[3];
        let hm = hm1 | hm2 | hm13 | hm23;

        let mut x: u32 = 1;
        let mut ptr = dst_off;
        let mut l_idx = lvl_idx; // per-column level pointer

        while hm as u32 & !(x - 1) != 0 {
            if sb_col > 0 || x > 1 {
                if hm1 & x as u8 != 0 {
                    let l = level[l_idx.min(63)] as usize;
                    let h = l >> 4;
                    let e = mblim_lut[l] as i32;
                    let ii = lim_lut[l] as i32;

                    if hmask1[0] & x as u8 != 0 {
                        if hmask2[0] & x as u8 != 0 {
                            // 16-tap: two stacked 8-pixel edges (lf_16_fn with stridea=stride)
                            loop_filter_edge(plane, ptr, stride, 1, e, ii, h as i32, 16);
                            loop_filter_edge(
                                plane,
                                ptr + 8 * stride,
                                stride,
                                1,
                                e,
                                ii,
                                h as i32,
                                16,
                            );
                        } else {
                            loop_filter_edge(plane, ptr, stride, 1, e, ii, h as i32, 8);
                        }
                    } else if hm2 & x as u8 != 0 {
                        let wd1 = if hmask1[1] & x as u8 != 0 { 8 } else { 4 };
                        loop_filter_edge(plane, ptr, stride, 1, e, ii, h as i32, wd1);
                        let l2 = level[(l_idx + (8 << ss_v)).min(63)] as usize;
                        let h2 = l2 >> 4;
                        let e2 = mblim_lut[l2] as i32;
                        let i2 = lim_lut[l2] as i32;
                        let wd2 = if hmask2[1] & x as u8 != 0 { 8 } else { 4 };
                        loop_filter_edge(
                            plane,
                            ptr + 8 * stride,
                            stride,
                            1,
                            e2,
                            i2,
                            h2 as i32,
                            wd2,
                        );
                    } else {
                        let wd = if hmask1[1] & x as u8 != 0 { 8 } else { 4 };
                        loop_filter_edge(plane, ptr, stride, 1, e, ii, h as i32, wd);
                    }
                } else if hm2 & x as u8 != 0 {
                    let l2 = level[(l_idx + (8 << ss_v)).min(63)] as usize;
                    let h = l2 >> 4;
                    let e = mblim_lut[l2] as i32;
                    let ii = lim_lut[l2] as i32;
                    let wd = if hmask2[1] & x as u8 != 0 { 8 } else { 4 };
                    loop_filter_edge(plane, ptr + 8 * stride, stride, 1, e, ii, h as i32, wd);
                }
            }

            // Inner 4-pixel edges and level pointer advance
            if ss_h != 0 {
                if x & 0xAA != 0 {
                    l_idx += 2;
                }
            } else {
                // Luma: process inner-4 edges (mask[3])
                // NOTE: inner4 edges are NOT guarded by (col || x > 1) in FFmpeg.
                if hm13 & x as u8 != 0 {
                    let l = level[l_idx.min(63)] as usize;
                    let h = l >> 4;
                    let e = mblim_lut[l] as i32;
                    let ii = lim_lut[l] as i32;

                    if hm23 & x as u8 != 0 {
                        loop_filter_edge(plane, ptr + 4, stride, 1, e, ii, h as i32, 4);
                        let l2 = level[(l_idx + (8 << ss_v)).min(63)] as usize;
                        let h2 = l2 >> 4;
                        let e2 = mblim_lut[l2] as i32;
                        let i2 = lim_lut[l2] as i32;
                        loop_filter_edge(
                            plane,
                            ptr + 4 + 8 * stride,
                            stride,
                            1,
                            e2,
                            i2,
                            h2 as i32,
                            4,
                        );
                    } else {
                        loop_filter_edge(plane, ptr + 4, stride, 1, e, ii, h as i32, 4);
                    }
                } else if hm23 & x as u8 != 0 {
                    let l2 = level[(l_idx + (8 << ss_v)).min(63)] as usize;
                    let h = l2 >> 4;
                    let e = mblim_lut[l2] as i32;
                    let ii = lim_lut[l2] as i32;
                    loop_filter_edge(plane, ptr + 4 + 8 * stride, stride, 1, e, ii, h as i32, 4);
                }
                l_idx += 1;
            }

            x <<= 1;
            ptr += 8 >> ss_h; // advance pixel pointer by 8 or 4 pixels
        }

        dst_off += 16 * stride; // advance by 16 pixel rows
        lvl_idx += 16 << ss_v; // advance level pointer by 16 or 32
    }
}

// ---------------------------------------------------------------------------
// filter_plane_rows — vp9lpf.c:102–177
// ---------------------------------------------------------------------------

/// Filter horizontal (row) edges for one plane within one superblock.
// Index-based loop preserves 1-to-1 correspondence with FFmpeg's filter_plane_rows.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn filter_plane_rows(
    plane: &mut [u8],
    plane_off: usize,
    stride: usize,
    sb_row: usize,
    ss_h: usize,
    ss_v: usize,
    level: &[u8; 64],
    mask: &[[u8; 4]; 8],
    lim_lut: &[u8; 64],
    mblim_lut: &[u8; 64],
) {
    let mut lvl_idx: usize = 0;
    let mut dst_off = plane_off;

    for y in 0..8usize {
        let vmask = mask[y];
        let vm = vmask[0] | vmask[1] | vmask[2];
        let vm3 = vmask[3];

        let mut x: u32 = 1;
        let mut ptr = dst_off;
        let mut l_idx = lvl_idx;

        while vm as u32 & !(x - 1) != 0 {
            let x_next = x << (1 + ss_h);

            if sb_row > 0 || y > 0 {
                if vm & x as u8 != 0 {
                    let l = level[l_idx.min(63)] as usize;
                    let h = l >> 4;
                    let e = mblim_lut[l] as i32;
                    let ii = lim_lut[l] as i32;

                    if vmask[0] & x as u8 != 0 {
                        if vmask[0] & x_next as u8 != 0 {
                            // 16-tap: two side-by-side 8-pixel edges (lf_16_fn with stridea=1)
                            loop_filter_edge(plane, ptr, 1, stride as isize, e, ii, h as i32, 16);
                            loop_filter_edge(
                                plane,
                                ptr + 8,
                                1,
                                stride as isize,
                                e,
                                ii,
                                h as i32,
                                16,
                            );
                        } else {
                            loop_filter_edge(plane, ptr, 1, stride as isize, e, ii, h as i32, 8);
                        }
                    } else if vm & x_next as u8 != 0 {
                        let wd1 = if vmask[1] & x as u8 != 0 { 8 } else { 4 };
                        loop_filter_edge(plane, ptr, 1, stride as isize, e, ii, h as i32, wd1);
                        let l2 = level[(l_idx + 1 + ss_h).min(63)] as usize;
                        let h2 = l2 >> 4;
                        let e2 = mblim_lut[l2] as i32;
                        let i2 = lim_lut[l2] as i32;
                        let wd2 = if vmask[1] & x_next as u8 != 0 { 8 } else { 4 };
                        loop_filter_edge(
                            plane,
                            ptr + 8,
                            1,
                            stride as isize,
                            e2,
                            i2,
                            h2 as i32,
                            wd2,
                        );
                    } else {
                        let wd = if vmask[1] & x as u8 != 0 { 8 } else { 4 };
                        loop_filter_edge(plane, ptr, 1, stride as isize, e, ii, h as i32, wd);
                    }
                } else if vm & x_next as u8 != 0 {
                    let l2 = level[(l_idx + 1 + ss_h).min(63)] as usize;
                    let h = l2 >> 4;
                    let e = mblim_lut[l2] as i32;
                    let ii = lim_lut[l2] as i32;
                    let wd = if vmask[1] & x_next as u8 != 0 { 8 } else { 4 };
                    loop_filter_edge(plane, ptr + 8, 1, stride as isize, e, ii, h as i32, wd);
                }
            }

            // Inner 4-pixel row edges (mask[3])
            // NOTE: inner4 edges are NOT guarded by (row || y) in FFmpeg.
            if ss_v == 0 {
                if vm3 & x as u8 != 0 {
                    let l = level[l_idx.min(63)] as usize;
                    let h = l >> 4;
                    let e = mblim_lut[l] as i32;
                    let ii = lim_lut[l] as i32;

                    if vm3 & x_next as u8 != 0 {
                        loop_filter_edge(
                            plane,
                            ptr + stride * 4,
                            1,
                            stride as isize,
                            e,
                            ii,
                            h as i32,
                            4,
                        );
                        let l2 = level[(l_idx + 1 + ss_h).min(63)] as usize;
                        let h2 = l2 >> 4;
                        let e2 = mblim_lut[l2] as i32;
                        let i2 = lim_lut[l2] as i32;
                        loop_filter_edge(
                            plane,
                            ptr + stride * 4 + 8,
                            1,
                            stride as isize,
                            e2,
                            i2,
                            h2 as i32,
                            4,
                        );
                    } else {
                        loop_filter_edge(
                            plane,
                            ptr + stride * 4,
                            1,
                            stride as isize,
                            e,
                            ii,
                            h as i32,
                            4,
                        );
                    }
                } else if vm3 & x_next as u8 != 0 {
                    let l2 = level[(l_idx + 1 + ss_h).min(63)] as usize;
                    let h = l2 >> 4;
                    let e = mblim_lut[l2] as i32;
                    let ii = lim_lut[l2] as i32;
                    loop_filter_edge(
                        plane,
                        ptr + stride * 4 + 8,
                        1,
                        stride as isize,
                        e,
                        ii,
                        h as i32,
                        4,
                    );
                }
            }

            // Advance: step = 2 << ss_h (2 for luma, 4 for chroma)
            l_idx += 2 << ss_h;
            x <<= 2 << ss_h;
            ptr += 16; // 16 pixels per step (2 8-pixel blocks)
        }

        // Advance level row pointer
        if ss_v != 0 {
            if y & 1 != 0 {
                lvl_idx += 16;
            }
        } else {
            lvl_idx += 8;
        }
        dst_off += (8 >> ss_v) * stride;
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Apply the VP9 loop filter to all block edges in `blocks`.
///
/// Two-phase, per-superblock approach matching FFmpeg's architecture:
/// 1. Build per-SB filter masks from block info
/// 2. Apply column then row filters using the masks
pub fn loop_filter_frame(fb: &mut FrameBuffer, blocks: &[BlockInfo], header: &FrameHeader) {
    if header.filter_level == 0 {
        return;
    }

    let ss_h = usize::from(header.subsampling_x);
    let ss_v = usize::from(header.subsampling_y);

    let width = fb.width as usize;
    let height = fb.height as usize;
    let y_stride = fb.y_stride;
    let uv_stride = fb.uv_stride;

    // FFmpeg: cols = (w + 7) >> 3, rows = (h + 7) >> 3 (in 8-pixel units)
    let cols8 = (width + 7) >> 3;
    let rows8 = (height + 7) >> 3;
    let sb_cols = (width + 63) >> 6;
    let sb_rows = (height + 63) >> 6;

    let (lim_lut, mblim_lut) = build_filter_luts(header.sharpness_level);

    // Pre-index blocks by superblock.  Key: (sb_row, sb_col).
    // We iterate blocks once, assign each to its SB.
    let mut sb_blocks: Vec<Vec<usize>> = vec![Vec::new(); sb_rows * sb_cols];
    for (idx, block) in blocks.iter().enumerate() {
        // block.row / block.col are in 4-pixel units.
        let sb_r = (block.row * 4) >> 6;
        let sb_c = (block.col * 4) >> 6;
        if sb_r < sb_rows && sb_c < sb_cols {
            sb_blocks[sb_r * sb_cols + sb_c].push(idx);
        }
    }

    for sb_row in 0..sb_rows {
        for sb_col in 0..sb_cols {
            let mut lflvl = Vp9Filter::default();

            // Phase 1: Build masks for this superblock.
            let sb_block_indices = &sb_blocks[sb_row * sb_cols + sb_col];
            for &bi in sb_block_indices {
                let block = &blocks[bi];
                let seg_feat = &header.segmentation.feat[block.segment_id as usize];

                // Compute filter level for this block.
                let ref_idx = if block.is_inter {
                    (block.ref_frame[0] + 1) as usize
                } else {
                    0
                };
                // FFmpeg: b->mode[3] != ZEROMV.  For intra, lflvl[0][0] ==
                // lflvl[0][1] (no mode delta), so mode_idx doesn't matter.
                let mode_idx = if block.is_inter {
                    usize::from(block.inter_mode[3] != 12) // mode[3] != ZEROMV
                } else {
                    0
                };
                let lvl = seg_feat.lflvl[ref_idx.min(3)][mode_idx];
                if lvl == 0 {
                    continue;
                }

                let bs = block.bs as usize;
                let w4 = BWH_TAB[1][bs][0] as usize; // in 8-pixel units
                let h4 = BWH_TAB[1][bs][1] as usize;

                // Block position in 8-pixel units within the superblock.
                // block.row/col are in 4-pixel units. FFmpeg's row/col are in 8-pixel units.
                let block_col8 = block.col >> 1; // convert 4px → 8px
                let block_row8 = block.row >> 1;
                let col7 = block_col8 & 7;
                let row7 = block_row8 & 7;

                // Clamp to frame edge.
                let x_end = w4.min(cols8.saturating_sub(block_col8));
                let y_end = h4.min(rows8.saturating_sub(block_row8));

                let skip_inter = block.is_inter && block.skip;

                // Fill level array — uses FULL block size (w4 x h4), NOT the
                // clamped x_end/y_end.  FFmpeg: setctx_2d(..., w4, h4, 8, lvl).
                setctx_2d(&mut lflvl.level, col7, row7, w4, h4, lvl);

                // Build luma masks.
                mask_edges(
                    &mut lflvl.mask[0],
                    0,
                    0,
                    row7,
                    col7,
                    x_end,
                    y_end,
                    0,
                    0,
                    block.tx_size as usize,
                    skip_inter,
                );

                // Build chroma masks.
                if ss_h != 0 || ss_v != 0 {
                    let col_end = if cols8 & 1 != 0 && block_col8 + w4 >= cols8 {
                        cols8 & 7
                    } else {
                        0
                    };
                    let row_end = if rows8 & 1 != 0 && block_row8 + h4 >= rows8 {
                        rows8 & 7
                    } else {
                        0
                    };
                    mask_edges(
                        &mut lflvl.mask[1],
                        ss_h,
                        ss_v,
                        row7,
                        col7,
                        x_end,
                        y_end,
                        col_end,
                        row_end,
                        block.uv_tx_size as usize,
                        skip_inter,
                    );
                }
            }

            // Phase 2: Apply filters using masks.
            let yoff = sb_row * 64 * y_stride + sb_col * 64;
            let uvoff = sb_row * (64 >> ss_v) * uv_stride + sb_col * (64 >> ss_h);

            // Luma: col edges, then row edges.
            filter_plane_cols(
                &mut fb.y,
                yoff,
                y_stride,
                sb_col,
                0,
                0,
                &lflvl.level,
                &lflvl.mask[0][0],
                &lim_lut,
                &mblim_lut,
            );
            filter_plane_rows(
                &mut fb.y,
                yoff,
                y_stride,
                sb_row,
                0,
                0,
                &lflvl.level,
                &lflvl.mask[0][1],
                &lim_lut,
                &mblim_lut,
            );

            // Chroma.
            let uv_mask = &lflvl.mask[usize::from(ss_h != 0 || ss_v != 0)];
            filter_plane_cols(
                &mut fb.u,
                uvoff,
                uv_stride,
                sb_col,
                ss_h,
                ss_v,
                &lflvl.level,
                &uv_mask[0],
                &lim_lut,
                &mblim_lut,
            );
            filter_plane_rows(
                &mut fb.u,
                uvoff,
                uv_stride,
                sb_row,
                ss_h,
                ss_v,
                &lflvl.level,
                &uv_mask[1],
                &lim_lut,
                &mblim_lut,
            );
            filter_plane_cols(
                &mut fb.v,
                uvoff,
                uv_stride,
                sb_col,
                ss_h,
                ss_v,
                &lflvl.level,
                &uv_mask[0],
                &lim_lut,
                &mblim_lut,
            );
            filter_plane_rows(
                &mut fb.v,
                uvoff,
                uv_stride,
                sb_row,
                ss_h,
                ss_v,
                &lflvl.level,
                &uv_mask[1],
                &lim_lut,
                &mblim_lut,
            );
        }
    }
}
