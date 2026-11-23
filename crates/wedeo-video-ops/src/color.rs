//! Exact CICP transfer curves, YCbCr↔RGB and gamut primaries color matrices,
//! and perceptual HDR→SDR tone-mapping operators matching FFmpeg `n9.1`.

use wedeo_core::{
    AlphaMode, ChromaLocation, ColorDesc, ColorModel, ColorPrimaries, ColorRange, Error,
    MatrixCoefficients, PixelLayout, Result, SampleDepth, Subsampling, TransferCharacteristic,
};

use crate::ops::ToneMapOp;

/// ITU-R BT.2408 reference SDR diffuse white luminance in cd/m² (nits).
pub const SDR_WHITE_NITS: f32 = 203.0;

/// SMPTE ST 2084 (PQ) peak luminance in cd/m² (nits).
pub const PQ_PEAK_NITS: f32 = 10_000.0;

/// ARIB STD-B67 (HLG) reference display peak luminance in cd/m² (nits).
pub const HLG_PEAK_NITS: f32 = 1_000.0;

const BT709_ALPHA: f64 = 1.099_296_826_809_442;
const BT709_BETA: f64 = 0.018_053_968_510_807;

const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 32.0 * 2413.0 / 4096.0;
const PQ_C3: f64 = 32.0 * 2392.0 / 4096.0;
const PQ_M1: f64 = 0.25 * 2610.0 / 4096.0;
const PQ_M2: f64 = 128.0 * 2523.0 / 4096.0;

const HLG_A: f64 = 0.178_832_77;
const HLG_B: f64 = 0.284_668_92;
const HLG_C: f64 = 0.559_910_73;

/// Sanitize and resolve `Unspecified` `ColorDesc` fields cleanly per FFmpeg `n9.1` `format.c`.
///
/// Guarantees that RGB outputs never inherit YUV `matrix` or `chroma_location` metadata.
pub fn resolve_color(
    layout: PixelLayout,
    color: ColorDesc,
    ref_color: Option<ColorDesc>,
) -> ColorDesc {
    let mut out = color;

    match layout.model {
        ColorModel::Rgb => {
            out.matrix = MatrixCoefficients::Identity;
            out.range = ColorRange::Full;
            out.chroma_location = ChromaLocation::Unspecified;
        }
        ColorModel::Gray => {
            out.matrix = MatrixCoefficients::Unspecified;
            out.chroma_location = ChromaLocation::Unspecified;
            if out.range == ColorRange::Unspecified {
                out.range = ColorRange::Full;
            }
        }
        ColorModel::Ycbcr | ColorModel::Xyz => {
            if out.matrix == MatrixCoefficients::Unspecified {
                out.matrix = MatrixCoefficients::Smpte170m;
            }
            if out.range == ColorRange::Unspecified {
                out.range = ColorRange::Limited;
            }
            if matches!(layout.subsampling, Subsampling::Cs444 | Subsampling::Cs400) {
                out.chroma_location = ChromaLocation::Unspecified;
            } else if out.chroma_location == ChromaLocation::Unspecified {
                out.chroma_location = ChromaLocation::Left;
            }
        }
    }

    if out.primaries == ColorPrimaries::Unspecified {
        out.primaries = match ref_color.map(|r| r.primaries) {
            Some(
                p @ (ColorPrimaries::Bt709
                | ColorPrimaries::Bt470m
                | ColorPrimaries::Bt470bg
                | ColorPrimaries::Smpte170m
                | ColorPrimaries::Smpte240m),
            ) => p,
            _ => ColorPrimaries::Bt709,
        };
    }

    if out.transfer == TransferCharacteristic::Unspecified {
        out.transfer = match ref_color.map(|r| r.transfer) {
            None
            | Some(
                TransferCharacteristic::Unspecified
                | TransferCharacteristic::Pq
                | TransferCharacteristic::Hlg,
            ) => TransferCharacteristic::Bt709,
            Some(trc) => trc,
        };
    }

    out
}

/// Sanitize destination `ColorDesc` for attaching to an output `VideoFrame`
/// (ensuring YUV metadata is never leaked onto RGB/Gray frames).
pub fn sanitize_output_color(
    dst_layout: PixelLayout,
    src_color: ColorDesc,
    dst_color: ColorDesc,
) -> ColorDesc {
    let resolved = resolve_color(dst_layout, dst_color, Some(src_color));
    match dst_layout.model {
        ColorModel::Rgb => ColorDesc {
            primaries: resolved.primaries,
            transfer: resolved.transfer,
            matrix: MatrixCoefficients::Identity,
            range: ColorRange::Full,
            chroma_location: ChromaLocation::Unspecified,
        },
        ColorModel::Gray => ColorDesc {
            primaries: resolved.primaries,
            transfer: resolved.transfer,
            matrix: MatrixCoefficients::Unspecified,
            range: resolved.range,
            chroma_location: ChromaLocation::Unspecified,
        },
        ColorModel::Ycbcr | ColorModel::Xyz => resolved,
    }
}

/// Returns `true` if two `TransferCharacteristic` values represent the exact same curve.
#[inline]
pub const fn transfers_equivalent(a: TransferCharacteristic, b: TransferCharacteristic) -> bool {
    let ca = canonical_transfer(a) as u8;
    let cb = canonical_transfer(b) as u8;
    ca == cb
}

#[inline]
pub const fn canonical_transfer(trc: TransferCharacteristic) -> TransferCharacteristic {
    match trc {
        TransferCharacteristic::Smpte170m
        | TransferCharacteristic::Bt2020_10
        | TransferCharacteristic::Bt2020_12
        | TransferCharacteristic::Bt1361Ecg => TransferCharacteristic::Bt709,
        TransferCharacteristic::Unspecified => TransferCharacteristic::Bt709,
        other => other,
    }
}

/// SMPTE ST 2084 (PQ) EOTF: normalized signal `[0, 1]` -> absolute luminance `[0, 10000]` nits.
#[inline]
pub fn pq_eotf_nits(e: f32) -> f32 {
    if e <= 0.0 {
        return 0.0;
    }
    let e_clamped = (e as f64).min(1.0);
    let em = e_clamped.powf(1.0 / PQ_M2);
    if em <= PQ_C1 {
        return 0.0;
    }
    let num = em - PQ_C1;
    let den = (PQ_C2 - PQ_C3 * em).max(1e-12);
    (PQ_PEAK_NITS as f64 * (num / den).powf(1.0 / PQ_M1)) as f32
}

/// SMPTE ST 2084 (PQ) inverse EOTF (OETF): absolute luminance `[0, 10000]` nits -> signal `[0, 1]`.
#[inline]
pub fn pq_oetf_nits(nits: f32) -> f32 {
    if nits <= 0.0 {
        return 0.0;
    }
    let l = (nits as f64 / PQ_PEAK_NITS as f64).clamp(0.0, 1.0);
    let ln = l.powf(PQ_M1);
    (((PQ_C1 + PQ_C2 * ln) / (1.0 + PQ_C3 * ln)).powf(PQ_M2)) as f32
}

/// Scalar 1D EOTF: non-linear signal `E` -> linear light (where `1.0 == 203 nits` SDR white).
///
/// Note: For HLG, prefer [`linearize_rgb`] which applies the luminance-coupled BT.2100 OOTF.
#[inline]
pub fn eotf_channel(trc: TransferCharacteristic, e: f32) -> f32 {
    let e64 = e as f64;
    let out = match canonical_transfer(trc) {
        TransferCharacteristic::Linear => e64,
        TransferCharacteristic::Bt709 => {
            if e64 <= 0.0 {
                0.0
            } else if e64 < 4.5 * BT709_BETA {
                e64 / 4.5
            } else {
                ((e64 + (BT709_ALPHA - 1.0)) / BT709_ALPHA).powf(1.0 / 0.45)
            }
        }
        TransferCharacteristic::Srgb => {
            const A: f64 = 1.055;
            const B: f64 = 0.003_130_8;
            if e64 <= 0.0 {
                0.0
            } else if e64 < 12.92 * B {
                e64 / 12.92
            } else {
                ((e64 + (A - 1.0)) / A).powf(2.4)
            }
        }
        TransferCharacteristic::Gamma22 => {
            if e64 <= 0.0 {
                0.0
            } else {
                e64.powf(2.2)
            }
        }
        TransferCharacteristic::Gamma28 => {
            if e64 <= 0.0 {
                0.0
            } else {
                e64.powf(2.8)
            }
        }
        TransferCharacteristic::Smpte240m => {
            const A: f64 = 1.1115;
            const B: f64 = 0.0228;
            if e64 <= 0.0 {
                0.0
            } else if e64 < 4.0 * B {
                e64 / 4.0
            } else {
                ((e64 + (A - 1.0)) / A).powf(1.0 / 0.45)
            }
        }
        TransferCharacteristic::Smpte428 => {
            if e64 <= 0.0 {
                0.0
            } else {
                (52.37 / 48.0) * e64.powf(2.6)
            }
        }
        TransferCharacteristic::Pq => (pq_eotf_nits(e) / SDR_WHITE_NITS) as f64,
        TransferCharacteristic::Hlg => {
            // Scene-referred inverse OETF * HLG peak / SDR white (OOTF applied in linearize_rgb)
            let s = if e64 <= 0.0 {
                0.0
            } else if e64 <= 0.5 {
                e64 * e64 / 3.0
            } else {
                (((e64 - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
            };
            s * (HLG_PEAK_NITS / SDR_WHITE_NITS) as f64
        }
        _ => e64,
    };
    out as f32
}

/// Scalar 1D inverse EOTF (OETF): linear light (where `1.0 == 203 nits`) -> non-linear signal `E`.
#[inline]
pub fn oetf_channel(trc: TransferCharacteristic, l: f32) -> f32 {
    let l64 = l as f64;
    let out = match canonical_transfer(trc) {
        TransferCharacteristic::Linear => l64,
        TransferCharacteristic::Bt709 => {
            if l64 <= 0.0 {
                0.0
            } else if l64 < BT709_BETA {
                4.5 * l64
            } else {
                BT709_ALPHA * l64.powf(0.45) - (BT709_ALPHA - 1.0)
            }
        }
        TransferCharacteristic::Srgb => {
            const A: f64 = 1.055;
            const B: f64 = 0.003_130_8;
            if l64 <= 0.0 {
                0.0
            } else if l64 < B {
                12.92 * l64
            } else {
                A * l64.powf(1.0 / 2.4) - (A - 1.0)
            }
        }
        TransferCharacteristic::Gamma22 => {
            if l64 <= 0.0 {
                0.0
            } else {
                l64.powf(1.0 / 2.2)
            }
        }
        TransferCharacteristic::Gamma28 => {
            if l64 <= 0.0 {
                0.0
            } else {
                l64.powf(1.0 / 2.8)
            }
        }
        TransferCharacteristic::Smpte240m => {
            const A: f64 = 1.1115;
            const B: f64 = 0.0228;
            if l64 <= 0.0 {
                0.0
            } else if l64 < B {
                4.0 * l64
            } else {
                A * l64.powf(0.45) - (A - 1.0)
            }
        }
        TransferCharacteristic::Smpte428 => {
            if l64 <= 0.0 {
                0.0
            } else {
                ((48.0 / 52.37) * l64).powf(1.0 / 2.6)
            }
        }
        TransferCharacteristic::Pq => pq_oetf_nits(l * SDR_WHITE_NITS) as f64,
        TransferCharacteristic::Hlg => {
            let s = (l64 * SDR_WHITE_NITS as f64 / HLG_PEAK_NITS as f64).max(0.0);
            if s <= 1.0 / 12.0 {
                (3.0 * s).sqrt()
            } else {
                HLG_A * (12.0 * s - HLG_B).ln() + HLG_C
            }
        }
        _ => l64,
    };
    out as f32
}

/// Apply display-referred EOTF to an RGB triplet (output linear RGB in SDR-relative units,
/// where `1.0 == 203 nits`). Handles HLG's coupled BT.2100 OOTF (`gamma = 1.2`).
#[inline]
pub fn linearize_rgb(trc: TransferCharacteristic, rgb: [f32; 3]) -> [f32; 3] {
    if matches!(trc, TransferCharacteristic::Hlg) {
        let inv_oetf = |e: f32| -> f64 {
            let e64 = e as f64;
            if e64 <= 0.0 {
                0.0
            } else if e64 <= 0.5 {
                e64 * e64 / 3.0
            } else {
                (((e64 - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
            }
        };
        let rs = inv_oetf(rgb[0]);
        let gs = inv_oetf(rgb[1]);
        let bs = inv_oetf(rgb[2]);
        let ys = (0.2627 * rs + 0.6780 * gs + 0.0593 * bs).max(0.0);
        // Reference display Lw = 1000 nits -> gamma = 1.2 -> gamma - 1 = 0.2
        let scale = (HLG_PEAK_NITS / SDR_WHITE_NITS) as f64 * ys.powf(0.2);
        return [
            (rs * scale) as f32,
            (gs * scale) as f32,
            (bs * scale) as f32,
        ];
    }

    [
        eotf_channel(trc, rgb[0]),
        eotf_channel(trc, rgb[1]),
        eotf_channel(trc, rgb[2]),
    ]
}

/// Apply inverse display-referred EOTF (OETF + inverse OOTF for HLG) to an RGB triplet
/// in SDR-relative linear units (`1.0 == 203 nits`).
#[inline]
pub fn delinearize_rgb(trc: TransferCharacteristic, rgb: [f32; 3]) -> [f32; 3] {
    if matches!(trc, TransferCharacteristic::Hlg) {
        let rd = (rgb[0] as f64 * SDR_WHITE_NITS as f64).max(0.0);
        let gd = (rgb[1] as f64 * SDR_WHITE_NITS as f64).max(0.0);
        let bd = (rgb[2] as f64 * SDR_WHITE_NITS as f64).max(0.0);
        let yd = 0.2627 * rd + 0.6780 * gd + 0.0593 * bd;
        let lw = HLG_PEAK_NITS as f64;
        if yd <= 0.0 {
            return [0.0, 0.0, 0.0];
        }
        // gamma = 1.2 => (1 - gamma) / gamma = -0.2 / 1.2 = -1/6
        let gain = (yd / lw).powf(-1.0 / 6.0) / lw;
        let oetf = |s: f64| -> f32 {
            let s = s.max(0.0);
            let e = if s <= 1.0 / 12.0 {
                (3.0 * s).sqrt()
            } else {
                HLG_A * (12.0 * s - HLG_B).ln() + HLG_C
            };
            e as f32
        };
        return [oetf(rd * gain), oetf(gd * gain), oetf(bd * gain)];
    }

    [
        oetf_channel(trc, rgb[0]),
        oetf_channel(trc, rgb[1]),
        oetf_channel(trc, rgb[2]),
    ]
}

/// Return ITU/CICP luma coefficients `(kr, kg, kb)` matching FFmpeg `libavutil/csp.c`.
pub fn luma_coefficients(matrix: MatrixCoefficients) -> Result<(f64, f64, f64)> {
    match matrix {
        MatrixCoefficients::Bt709 => Ok((0.2126, 0.7152, 0.0722)),
        MatrixCoefficients::Bt470bg
        | MatrixCoefficients::Smpte170m
        | MatrixCoefficients::Unspecified => Ok((0.299, 0.587, 0.114)),
        MatrixCoefficients::Fcc => Ok((0.30, 0.59, 0.11)),
        MatrixCoefficients::Smpte240m => Ok((0.212, 0.701, 0.087)),
        MatrixCoefficients::Bt2020Ncl => Ok((0.2627, 0.6780, 0.0593)),
        _ => Err(Error::unsupported(
            "wedeo-video-ops",
            "unsupported matrix coefficients",
        )),
    }
}

/// Compute range `(y_min, y_span, c_mid, c_span)` in `[0.0, 1.0]` normalized sample space,
/// matching FFmpeg `libswscale/format.c` `fmt_encode_range()`.
fn range_params(
    depth: SampleDepth,
    range: ColorRange,
    is_rgb_or_gray: bool,
) -> (f64, f64, f64, f64) {
    if matches!(depth, SampleDepth::F32 | SampleDepth::F16) {
        return (0.0, 1.0, 0.5, 1.0);
    }
    let bits = depth.bits() as u32;
    let max_int = ((1u64 << bits) - 1) as f64;
    if is_rgb_or_gray {
        if range == ColorRange::Limited {
            let s = (1u64 << (bits - 8)) as f64;
            (16.0 * s / max_int, 219.0 * s / max_int, 0.0, 1.0)
        } else {
            (0.0, 1.0, 0.0, 1.0)
        }
    } else if range == ColorRange::Full {
        let c_mid = (1u64 << (bits - 1)) as f64 / max_int;
        (0.0, 1.0, c_mid, 1.0)
    } else {
        let s = (1u64 << (bits - 8)) as f64;
        let y_min = 16.0 * s / max_int;
        let y_span = 219.0 * s / max_int;
        let c_mid = 128.0 * s / max_int;
        let c_span = 224.0 * s / max_int;
        (y_min, y_span, c_mid, c_span)
    }
}

/// Build the 3×4 matrix converting normalized `[0, 1]` source samples in canonical order
/// (`[R, G, B]` or `[Y, Cb, Cr]` or `[Gray, _, _]`) to full-range non-linear R'G'B' `[0, 1]`.
pub fn decode_to_rgb_matrix(layout: PixelLayout, color: ColorDesc) -> Result<[[f32; 4]; 3]> {
    match layout.model {
        ColorModel::Rgb => {
            let (y_min, y_span, _, _) = range_params(layout.depth, color.range, true);
            let s = (1.0 / y_span) as f32;
            let o = (-y_min / y_span) as f32;
            Ok([[s, 0.0, 0.0, o], [0.0, s, 0.0, o], [0.0, 0.0, s, o]])
        }
        ColorModel::Gray => {
            let (y_min, y_span, _, _) = range_params(layout.depth, color.range, true);
            let s = (1.0 / y_span) as f32;
            let o = (-y_min / y_span) as f32;
            Ok([[s, 0.0, 0.0, o], [s, 0.0, 0.0, o], [s, 0.0, 0.0, o]])
        }
        ColorModel::Ycbcr => {
            let (y_min, y_span, c_mid, c_span) = range_params(layout.depth, color.range, false);
            let m_yuv2rgb = if color.matrix == MatrixCoefficients::Ycgco {
                [[1.0, -1.0, 1.0], [1.0, 1.0, 0.0], [1.0, -1.0, -1.0]]
            } else {
                let (kr, kg, kb) = luma_coefficients(color.matrix)?;
                let m02 = 2.0 * (1.0 - kr);
                let m21 = 2.0 * (1.0 - kb);
                let m11 = -(kb / kg) * m21;
                let m12 = -(kr / kg) * m02;
                [[1.0, 0.0, m02], [1.0, m11, m12], [1.0, m21, 0.0]]
            };

            let mut out = [[0.0f32; 4]; 3];
            for i in 0..3 {
                let c0 = m_yuv2rgb[i][0] / y_span;
                let c1 = m_yuv2rgb[i][1] / c_span;
                let c2 = m_yuv2rgb[i][2] / c_span;
                let off = -(c0 * y_min + c1 * c_mid + c2 * c_mid);
                out[i] = [c0 as f32, c1 as f32, c2 as f32, off as f32];
            }
            Ok(out)
        }
        ColorModel::Xyz => Err(Error::unsupported(
            "wedeo-video-ops",
            "XYZ decode is not supported",
        )),
    }
}

/// Build the 3×4 matrix converting full-range non-linear R'G'B' `[0, 1]` to normalized `[0, 1]`
/// destination samples in canonical order (`[R, G, B]` or `[Y, Cb, Cr]` or `[Gray, _, _]`).
pub fn encode_from_rgb_matrix(layout: PixelLayout, color: ColorDesc) -> Result<[[f32; 4]; 3]> {
    match layout.model {
        ColorModel::Rgb => {
            let (y_min, y_span, _, _) = range_params(layout.depth, color.range, true);
            let s = y_span as f32;
            let o = y_min as f32;
            Ok([[s, 0.0, 0.0, o], [0.0, s, 0.0, o], [0.0, 0.0, s, o]])
        }
        ColorModel::Gray => {
            let (y_min, y_span, _, _) = range_params(layout.depth, color.range, true);
            let (kr, kg, kb) = match color.primaries {
                ColorPrimaries::Bt2020 => (0.2627, 0.6780, 0.0593),
                ColorPrimaries::Bt709 => (0.2126, 0.7152, 0.0722),
                _ => (0.299, 0.587, 0.114),
            };
            Ok([
                [
                    (kr * y_span) as f32,
                    (kg * y_span) as f32,
                    (kb * y_span) as f32,
                    y_min as f32,
                ],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.5],
            ])
        }
        ColorModel::Ycbcr => {
            let (y_min, y_span, c_mid, c_span) = range_params(layout.depth, color.range, false);
            let m_rgb2yuv = if color.matrix == MatrixCoefficients::Ycgco {
                [[0.25, 0.5, 0.25], [0.5, 0.0, -0.5], [-0.25, 0.5, -0.25]]
            } else {
                let (kr, kg, kb) = luma_coefficients(color.matrix)?;
                let cb1 = kb - 1.0;
                let cr1 = kr - 1.0;
                [
                    [kr, kg, kb],
                    [0.5 * kr / cb1, 0.5 * kg / cb1, 0.5],
                    [0.5, 0.5 * kg / cr1, 0.5 * kb / cr1],
                ]
            };

            let mut out = [[0.0f32; 4]; 3];
            let spans = [y_span, c_span, c_span];
            let offs = [y_min, c_mid, c_mid];
            for i in 0..3 {
                out[i] = [
                    (m_rgb2yuv[i][0] * spans[i]) as f32,
                    (m_rgb2yuv[i][1] * spans[i]) as f32,
                    (m_rgb2yuv[i][2] * spans[i]) as f32,
                    offs[i] as f32,
                ];
            }
            Ok(out)
        }
        ColorModel::Xyz => Err(Error::unsupported(
            "wedeo-video-ops",
            "XYZ encode is not supported",
        )),
    }
}

/// CIE xy chromaticity coordinates `(wp, r, g, b)` from FFmpeg `libavutil/csp.c`.
struct PrimariesDesc {
    wp: [f64; 2],
    r: [f64; 2],
    g: [f64; 2],
    b: [f64; 2],
}

fn primaries_desc(p: ColorPrimaries) -> Option<PrimariesDesc> {
    const D65: [f64; 2] = [0.3127, 0.3290];
    const WP_C: [f64; 2] = [0.3100, 0.3160];
    const WP_DCI: [f64; 2] = [0.3140, 0.3510];
    const WP_E: [f64; 2] = [1.0 / 3.0, 1.0 / 3.0];

    match p {
        ColorPrimaries::Bt709 => Some(PrimariesDesc {
            wp: D65,
            r: [0.640, 0.330],
            g: [0.300, 0.600],
            b: [0.150, 0.060],
        }),
        ColorPrimaries::Bt470m => Some(PrimariesDesc {
            wp: WP_C,
            r: [0.670, 0.330],
            g: [0.210, 0.710],
            b: [0.140, 0.080],
        }),
        ColorPrimaries::Bt470bg | ColorPrimaries::Ebu3213 => Some(PrimariesDesc {
            wp: D65,
            r: [0.640, 0.330],
            g: [0.290, 0.600],
            b: [0.150, 0.060],
        }),
        ColorPrimaries::Smpte170m | ColorPrimaries::Smpte240m => Some(PrimariesDesc {
            wp: D65,
            r: [0.630, 0.340],
            g: [0.310, 0.595],
            b: [0.155, 0.070],
        }),
        ColorPrimaries::Film => Some(PrimariesDesc {
            wp: WP_C,
            r: [0.681, 0.319],
            g: [0.243, 0.692],
            b: [0.145, 0.049],
        }),
        ColorPrimaries::Bt2020 => Some(PrimariesDesc {
            wp: D65,
            r: [0.708, 0.292],
            g: [0.170, 0.797],
            b: [0.131, 0.046],
        }),
        ColorPrimaries::Smpte428 => Some(PrimariesDesc {
            wp: WP_E,
            r: [0.735, 0.265],
            g: [0.274, 0.718],
            b: [0.167, 0.009],
        }),
        ColorPrimaries::Smpte431 => Some(PrimariesDesc {
            wp: WP_DCI,
            r: [0.680, 0.320],
            g: [0.265, 0.690],
            b: [0.150, 0.060],
        }),
        ColorPrimaries::Smpte432 => Some(PrimariesDesc {
            wp: D65,
            r: [0.680, 0.320],
            g: [0.265, 0.690],
            b: [0.150, 0.060],
        }),
        _ => None,
    }
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat3_invert(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a00 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
    let a01 = -(m[0][1] * m[2][2] - m[2][1] * m[0][2]);
    let a02 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
    let a10 = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]);
    let a11 = m[0][0] * m[2][2] - m[2][0] * m[0][2];
    let a12 = -(m[0][0] * m[1][2] - m[1][0] * m[0][2]);
    let a20 = m[1][0] * m[2][1] - m[2][0] * m[1][1];
    let a21 = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]);
    let a22 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    let det = m[0][0] * a00 + m[1][0] * a01 + m[2][0] * a02;
    let inv = 1.0 / det;
    [
        [a00 * inv, a01 * inv, a02 * inv],
        [a10 * inv, a11 * inv, a12 * inv],
        [a20 * inv, a21 * inv, a22 * inv],
    ]
}

fn rgb2xyz(desc: &PrimariesDesc) -> [[f64; 3]; 3] {
    let cie_x = |xy: [f64; 2]| xy[0] / xy[1];
    let cie_z = |xy: [f64; 2]| (1.0 - xy[0] - xy[1]) / xy[1];

    let xr = [cie_x(desc.r), cie_x(desc.g), cie_x(desc.b)];
    let zr = [cie_z(desc.r), cie_z(desc.g), cie_z(desc.b)];
    let xw = cie_x(desc.wp);
    let zw = cie_z(desc.wp);

    let m = [xr, [1.0, 1.0, 1.0], zr];
    let inv = mat3_invert(&m);
    let mut s = [0.0; 3];
    for i in 0..3 {
        s[i] = inv[i][0] * xw + inv[i][1] + inv[i][2] * zw;
    }

    [
        [s[0] * xr[0], s[1] * xr[1], s[2] * xr[2]],
        [s[0], s[1], s[2]],
        [s[0] * zr[0], s[1] * zr[1], s[2] * zr[2]],
    ]
}

/// CAT16 chromatic adaptation matrix (`FFmpeg/libswscale/csputils.c`).
fn cat16_adaptation(src_wp: [f64; 2], dst_wp: [f64; 2]) -> [[f64; 3]; 3] {
    if (src_wp[0] - dst_wp[0]).abs() < 1e-6 && (src_wp[1] - dst_wp[1]).abs() < 1e-6 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    const M_CAT16: [[f64; 3]; 3] = [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ];
    let inv = mat3_invert(&M_CAT16);
    let sx = src_wp[0] / src_wp[1];
    let sz = (1.0 - src_wp[0] - src_wp[1]) / src_wp[1];
    let dx = dst_wp[0] / dst_wp[1];
    let dz = (1.0 - dst_wp[0] - dst_wp[1]) / dst_wp[1];

    let mut diag = [[0.0; 3]; 3];
    for i in 0..3 {
        let cs = M_CAT16[i][0] * sx + M_CAT16[i][1] + M_CAT16[i][2] * sz;
        let cd = M_CAT16[i][0] * dx + M_CAT16[i][1] + M_CAT16[i][2] * dz;
        diag[i][i] = cd / cs;
    }
    mat3_mul(&inv, &mat3_mul(&diag, &M_CAT16))
}

/// Compute 3×4 linear-RGB gamut conversion matrix from `src` primaries to `dst` primaries.
pub fn primaries_matrix(src: ColorPrimaries, dst: ColorPrimaries) -> [[f32; 4]; 3] {
    const IDENT: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    if src == dst || src == ColorPrimaries::Unspecified || dst == ColorPrimaries::Unspecified {
        return IDENT;
    }
    let (Some(sd), Some(dd)) = (primaries_desc(src), primaries_desc(dst)) else {
        return IDENT;
    };
    let s2x = rgb2xyz(&sd);
    let x2d = mat3_invert(&rgb2xyz(&dd));
    let cat = cat16_adaptation(sd.wp, dd.wp);
    let m = mat3_mul(&x2d, &mat3_mul(&cat, &s2x));
    [
        [m[0][0] as f32, m[0][1] as f32, m[0][2] as f32, 0.0],
        [m[1][0] as f32, m[1][1] as f32, m[1][2] as f32, 0.0],
        [m[2][0] as f32, m[2][1] as f32, m[2][2] as f32, 0.0],
    ]
}

/// Multiply two 3×4 affine color matrices (`B * A`).
#[inline]
pub fn fuse_color_matrices(a: &[[f32; 4]; 3], b: &[[f32; 4]; 3]) -> [[f32; 4]; 3] {
    let mut c = [[0.0f32; 4]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = b[i][0] * a[0][j] + b[i][1] * a[1][j] + b[i][2] * a[2][j];
        }
        c[i][3] = b[i][0] * a[0][3] + b[i][1] * a[1][3] + b[i][2] * a[2][3] + b[i][3];
    }
    c
}

#[inline]
fn hable_curve(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (x * A + B * C) + D * E) / (x * (x * A + B) + D * F)) - E / F
}

/// ITU-R BT.2390-10 perceptual EETF Hermite spline on SDR-relative linear RGB (`1.0 == 203 nits`).
pub fn tone_map_bt2390(rgb: [f32; 3], src_peak_nits: f32, dst_peak_nits: f32) -> [f32; 3] {
    let r = rgb[0].max(0.0);
    let g = rgb[1].max(0.0);
    let b = rgb[2].max(0.0);
    let sig = r.max(g).max(b);
    if sig <= 1e-6 {
        return [0.0, 0.0, 0.0];
    }

    let dst_peak_nits = dst_peak_nits.max(1.0);
    let src_peak_nits = src_peak_nits.max(dst_peak_nits);
    let dst_max_rel = dst_peak_nits / SDR_WHITE_NITS;

    if (src_peak_nits - dst_peak_nits).abs() < 1e-3 {
        let new_sig = sig.min(dst_max_rel);
        let scale = new_sig / sig;
        return [r * scale, g * scale, b * scale];
    }

    let sig_nits = sig * SDR_WHITE_NITS;
    let lw = pq_oetf_nits(src_peak_nits);
    let lmax = pq_oetf_nits(dst_peak_nits);
    let e_in = pq_oetf_nits(sig_nits);

    let e1 = (e_in / lw).clamp(0.0, 1.0);
    let max_lum = (lmax / lw).clamp(0.0, 1.0);
    let ks = 1.5 * max_lum - 0.5;

    let e2 = if ks <= 0.0 || e1 >= ks {
        let ks_c = ks.max(0.0);
        let denom = (1.0 - ks_c).max(1e-6);
        let t = ((e1 - ks_c) / denom).clamp(0.0, 1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        (2.0 * t3 - 3.0 * t2 + 1.0) * ks_c
            + (t3 - 2.0 * t2 + t) * (1.0 - ks_c)
            + (-2.0 * t3 + 3.0 * t2) * max_lum
    } else {
        e1
    };

    let out_nits = pq_eotf_nits((e2 * lw).clamp(0.0, 1.0));
    let new_sig = (out_nits / SDR_WHITE_NITS).clamp(0.0, dst_max_rel);
    let scale = new_sig / sig;
    [r * scale, g * scale, b * scale]
}

/// Apply perceptual HDR→SDR tone-mapping to an SDR-relative linear RGB triplet (`1.0 == 203 nits`).
#[inline]
pub fn tone_map_rgb(
    op: ToneMapOp,
    rgb: [f32; 3],
    src_peak_nits: f32,
    dst_peak_nits: f32,
) -> [f32; 3] {
    match op {
        ToneMapOp::None => rgb,
        ToneMapOp::Clip => {
            let max_rel = (dst_peak_nits / SDR_WHITE_NITS).max(1e-6);
            let r = rgb[0].max(0.0);
            let g = rgb[1].max(0.0);
            let b = rgb[2].max(0.0);
            let sig = r.max(g).max(b);
            if sig <= max_rel {
                [r, g, b]
            } else {
                let s = max_rel / sig;
                [r * s, g * s, b * s]
            }
        }
        ToneMapOp::Hable => {
            let r = rgb[0].max(0.0);
            let g = rgb[1].max(0.0);
            let b = rgb[2].max(0.0);
            let sig = r.max(g).max(b);
            if sig <= 1e-6 {
                return [0.0, 0.0, 0.0];
            }
            let src_peak_rel = (src_peak_nits / SDR_WHITE_NITS).max(1.0);
            let dst_peak_rel = (dst_peak_nits / SDR_WHITE_NITS).max(1e-6);
            let mapped = (hable_curve(sig) / hable_curve(src_peak_rel)) * dst_peak_rel;
            let s = mapped / sig;
            [r * s, g * s, b * s]
        }
        ToneMapOp::Bt2390 => tone_map_bt2390(rgb, src_peak_nits, dst_peak_nits),
    }
}

/// Resolve effective source peak luminance (in nits) from options and `TransferCharacteristic`.
#[inline]
pub fn effective_src_peak_nits(trc: TransferCharacteristic, explicit_peak: f32) -> f32 {
    if explicit_peak > 0.0 {
        explicit_peak
    } else {
        match trc {
            TransferCharacteristic::Pq => PQ_PEAK_NITS,
            TransferCharacteristic::Hlg => HLG_PEAK_NITS,
            _ => SDR_WHITE_NITS,
        }
    }
}

/// Convert a single overlay pixel from `src_layout` into canonical `[R, G, B, A]` in `[0.0, 1.0]`
/// for alpha compositing.
pub fn unpack_alpha_overlay_pixel(
    layout: PixelLayout,
    raw: [f32; 4],
    color: ColorDesc,
) -> [f32; 4] {
    let max_val = match layout.depth {
        SampleDepth::F32 | SampleDepth::F16 => 1.0,
        d => ((1u32 << d.bits()) - 1) as f32,
    };
    let norm = [
        raw[0] / max_val,
        raw[1] / max_val,
        raw[2] / max_val,
        if matches!(layout.alpha, AlphaMode::None) {
            1.0
        } else {
            raw[3] / max_val
        },
    ];
    let swiz = crate::ops::read_to_canonical_swizzle(layout);
    let canon = [
        norm[swiz[0] as usize],
        norm[swiz[1] as usize],
        norm[swiz[2] as usize],
        norm[swiz[3] as usize],
    ];
    let mat = decode_to_rgb_matrix(layout, color).unwrap_or([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]);
    [
        (mat[0][0] * canon[0] + mat[0][1] * canon[1] + mat[0][2] * canon[2] + mat[0][3])
            .clamp(0.0, 1.0),
        (mat[1][0] * canon[0] + mat[1][1] * canon[1] + mat[1][2] * canon[2] + mat[1][3])
            .clamp(0.0, 1.0),
        (mat[2][0] * canon[0] + mat[2][1] * canon[1] + mat[2][2] * canon[2] + mat[2][3])
            .clamp(0.0, 1.0),
        canon[3].clamp(0.0, 1.0),
    ]
}
