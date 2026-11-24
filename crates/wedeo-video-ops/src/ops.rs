//! Pure-Rust port of FFmpeg `n9.1` `SwsOpList` intermediate representation.

use wedeo_core::{
    ChromaLocation, ColorDesc, ColorModel, CompOrder, CropRect, Error, MemoryDomain, PixelLayout,
    PlanePacking, Result, SampleDepth, SideData, TransferCharacteristic, VideoFrame,
};

use crate::color;

/// Identity 3×3 display matrix in 16.16 / 2.30 fixed point (matching FFmpeg `av_display_rotation_set(0)`).
pub const IDENTITY_DISPLAY_MATRIX: [i32; 9] = [1 << 16, 0, 0, 0, 1 << 16, 0, 0, 0, 1 << 30];

/// Construct a FFmpeg-compatible `[i32; 9]` display matrix from clockwise rotation
/// (`0`, `90`, `180`, `270` degrees) and optional horizontal/vertical flips.
pub fn display_matrix(clockwise_deg: i32, hflip: bool, vflip: bool) -> [i32; 9] {
    let deg = clockwise_deg.rem_euclid(360);
    let (cos_i, sin_i) = match deg {
        0 => (1, 0),
        90 => (0, -1),
        180 => (-1, 0),
        270 => (0, 1),
        _ => {
            let rad = -(clockwise_deg as f64) * std::f64::consts::PI / 180.0;
            let c = (rad.cos() * 65536.0).round() as i32;
            let s = (rad.sin() * 65536.0).round() as i32;
            let mut m = [c, -s, 0, s, c, 0, 0, 0, 1 << 30];
            apply_flips(&mut m, hflip, vflip);
            return m;
        }
    };
    let mut m = [
        cos_i * (1 << 16),
        -sin_i * (1 << 16),
        0,
        sin_i * (1 << 16),
        cos_i * (1 << 16),
        0,
        0,
        0,
        1 << 30,
    ];
    apply_flips(&mut m, hflip, vflip);
    m
}

fn apply_flips(m: &mut [i32; 9], hflip: bool, vflip: bool) {
    let flip = [if hflip { -1 } else { 1 }, if vflip { -1 } else { 1 }, 1];
    if hflip || vflip {
        for (i, v) in m.iter_mut().enumerate() {
            *v *= flip[i % 3];
        }
    }
}

/// Decoded orthogonal 2D dihedral transform (`D_4`) extracted from a `display_matrix`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayTransform {
    /// Forward 2×2 matrix elements mapping `(p, q) -> (a*p + c*q, b*p + d*q)` in `{-1, 0, 1}`.
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub d: i32,
}

impl DisplayTransform {
    pub const IDENTITY: Self = Self {
        a: 1,
        b: 0,
        c: 0,
        d: 1,
    };

    /// Parse an FFmpeg 16.16 fixed-point `display_matrix` into an exact 0°/90°/180°/270°
    /// rotation + flip transform.
    ///
    /// Note: `[0; 9]` is treated as identity (convenient for default-initialized structs).
    pub fn from_matrix(m: &[i32; 9]) -> Result<Self> {
        if *m == [0; 9] || *m == IDENTITY_DISPLAY_MATRIX {
            return Ok(Self::IDENTITY);
        }
        let scale_x = ((m[0] as f64).hypot(m[3] as f64)) / 65536.0;
        let scale_y = ((m[1] as f64).hypot(m[4] as f64)) / 65536.0;
        if scale_x < 1e-3 || scale_y < 1e-3 {
            return Err(Error::invalid_argument(
                "wedeo-video-ops",
                "singular display_matrix",
            ));
        }
        let norm = |val: i32, s: f64| -> Result<i32> {
            let f = (val as f64 / 65536.0) / s;
            let r = f.round();
            if (f - r).abs() > 0.05 {
                return Err(Error::unsupported(
                    "wedeo-video-ops",
                    "only orthogonal 0/90/180/270 display_matrix rotations/flips are supported",
                ));
            }
            Ok(r as i32)
        };
        let a = norm(m[0], scale_x)?;
        let c = norm(m[3], scale_x)?;
        let b = norm(m[1], scale_y)?;
        let d = norm(m[4], scale_y)?;
        let det = a * d - b * c;
        if (a * a + b * b) != 1 || (c * c + d * d) != 1 || det.abs() != 1 {
            return Err(Error::unsupported(
                "wedeo-video-ops",
                "display_matrix is not a valid orthogonal 90-degree rotation/flip",
            ));
        }
        Ok(Self { a, b, c, d })
    }

    #[inline]
    pub const fn is_identity(self) -> bool {
        self.a == 1 && self.b == 0 && self.c == 0 && self.d == 1
    }

    #[inline]
    pub const fn swaps_axes(self) -> bool {
        self.a == 0
    }

    /// Output dimensions after applying this transform to a `cw x ch` cropped source.
    #[inline]
    pub const fn output_dims(self, cw: u32, ch: u32) -> (u32, u32) {
        if self.swaps_axes() {
            (ch, cw)
        } else {
            (cw, ch)
        }
    }

    /// Map an oriented pixel `(x_out, y_out)` in `[0..out_w) x [0..out_h)` back to
    /// cropped source coordinates `(x_crop, y_crop)` in `[0..cw) x [0..ch)`.
    #[inline]
    pub const fn map_to_src(self, x_out: u32, y_out: u32, cw: u32, ch: u32) -> (u32, u32) {
        let cw_i = cw as i32;
        let ch_i = ch as i32;
        let min_p = if self.a < 0 { self.a * (cw_i - 1) } else { 0 }
            + if self.c < 0 { self.c * (ch_i - 1) } else { 0 };
        let min_q = if self.b < 0 { self.b * (cw_i - 1) } else { 0 }
            + if self.d < 0 { self.d * (ch_i - 1) } else { 0 };
        let p_raw = x_out as i32 + min_p;
        let q_raw = y_out as i32 + min_q;
        let x_crop = self.a * p_raw + self.b * q_raw;
        let y_crop = self.c * p_raw + self.d * q_raw;
        (x_crop as u32, y_crop as u32)
    }
}

/// Complete video frame contract describing dimensions, layout, colorimetry,
/// crop, orientation, and memory residency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoContract {
    pub width: u32,
    pub height: u32,
    pub layout: PixelLayout,
    pub color: ColorDesc,
    pub crop: CropRect,
    pub display_matrix: [i32; 9],
    pub domain: MemoryDomain,
}

impl VideoContract {
    pub const fn new(width: u32, height: u32, layout: PixelLayout) -> Self {
        Self {
            width,
            height,
            layout,
            color: ColorDesc::UNSPECIFIED,
            crop: CropRect::ZERO,
            display_matrix: IDENTITY_DISPLAY_MATRIX,
            domain: MemoryDomain::Cpu,
        }
    }

    pub fn with_color(mut self, color: ColorDesc) -> Self {
        self.color = color;
        self
    }

    pub fn with_crop(mut self, crop: CropRect) -> Self {
        self.crop = crop;
        self
    }

    pub fn with_display_matrix(mut self, display_matrix: [i32; 9]) -> Self {
        self.display_matrix = display_matrix;
        self
    }

    pub fn with_domain(mut self, domain: MemoryDomain) -> Self {
        self.domain = domain;
        self
    }

    /// Build a `VideoContract` directly from a `VideoFrame`, extracting any
    /// `SideData::DisplayMatrix` attached to the frame.
    pub fn from_frame(frame: &VideoFrame) -> Self {
        let display_matrix = frame
            .side_data
            .iter()
            .find_map(|sd| match sd {
                SideData::DisplayMatrix(m) => Some(*m),
                _ => None,
            })
            .unwrap_or(IDENTITY_DISPLAY_MATRIX);

        Self {
            width: frame.width,
            height: frame.height,
            layout: frame.layout,
            color: frame.color,
            crop: frame.crop,
            display_matrix,
            domain: frame.domain(),
        }
    }

    /// Effective display dimensions after applying `crop` and `display_matrix` rotation.
    pub fn effective_dimensions(&self) -> Result<(u32, u32)> {
        let (cw, ch) = self.crop.display_size(self.width, self.height);
        let dt = DisplayTransform::from_matrix(&self.display_matrix)?;
        Ok(dt.output_dims(cw, ch))
    }
}

/// Resampling kernel used by `ScaleH` and `ScaleV`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SwsScaler {
    Point,
    Bilinear,
    #[default]
    Bicubic,
    Lanczos,
}

/// Perceptual HDR-to-SDR tone-mapping operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ToneMapOp {
    #[default]
    None,
    Clip,
    Hable,
    Bt2390,
}

/// Dithering strategy applied prior to quantization (`Pack`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DitherMode {
    None,
    #[default]
    Ordered,
    ErrorDiffusion,
}

/// Configuration options for `SwsOpList::build`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwsOptions {
    pub scaler: SwsScaler,
    pub tone_map: ToneMapOp,
    pub dither: DitherMode,
    /// Optional explicit source peak luminance in nits (`0.0` = infer from transfer:
    /// 10000 nits for PQ, 1000 nits for HLG, 203 nits for SDR).
    pub src_peak_nits: f32,
    /// Target peak luminance in nits for HDR-to-SDR tone-mapping (default: `203.0` per BT.2408).
    pub dst_peak_nits: f32,
    /// Use +0.5 rounding when quantizing without dither.
    pub accurate_rounding: bool,
}

impl Default for SwsOptions {
    fn default() -> Self {
        Self {
            scaler: SwsScaler::Bicubic,
            tone_map: ToneMapOp::Bt2390,
            dither: DitherMode::Ordered,
            src_peak_nits: 0.0,
            dst_peak_nits: color::SDR_WHITE_NITS,
            accurate_rounding: true,
        }
    }
}

/// Elementary video pipeline operation (FFmpeg `n9.1` `SwsOp` IR).
#[derive(Debug, Clone, PartialEq)]
pub enum SwsOp {
    Read {
        layout: PixelLayout,
        crop: CropRect,
        display_matrix: [i32; 9],
    },
    Unpack {
        depth: SampleDepth,
        shift: u8,
    },
    Swizzle([u8; 4]),
    Linearize(TransferCharacteristic),
    ColorMatrix([[f32; 4]; 3]),
    ToneMap(ToneMapOp),
    ScaleH {
        dst_w: u32,
        kernel: SwsScaler,
        chroma_loc: ChromaLocation,
    },
    ScaleV {
        dst_h: u32,
        kernel: SwsScaler,
        chroma_loc: ChromaLocation,
    },
    BlendAlpha,
    Delinearize(TransferCharacteristic),
    Dither(DitherMode),
    Pack(PixelLayout),
    Write(PixelLayout),
}

/// Complete, optimizable video transformation operation list.
#[derive(Debug, Clone, PartialEq)]
pub struct SwsOpList {
    pub ops: Vec<SwsOp>,
    pub src: VideoContract,
    pub dst: VideoContract,
    pub options: SwsOptions,
}

impl SwsOpList {
    /// Propagate frame metadata after applying this conversion, accounting for
    /// rotation, crop, tone mapping and the destination pixel layout.
    pub fn populate_output_metadata(
        &self,
        src: &VideoFrame,
        out: &mut VideoFrame,
        transform: DisplayTransform,
    ) {
        out.pts = src.pts;
        out.pkt_dts = src.pkt_dts;
        out.duration = src.duration;
        out.time_base = src.time_base;
        out.picture_type = src.picture_type;
        out.flags = src.flags;
        out.repeat_pict = src.repeat_pict;
        out.sar = if transform.swaps_axes() && src.sar.num > 0 && src.sar.den > 0 {
            wedeo_core::Rational::new(src.sar.den, src.sar.num)
        } else {
            src.sar
        };
        out.crop = CropRect::ZERO;
        out.color = color::sanitize_output_color(out.layout, src.color, self.dst.color);

        let tone_mapped = self.src.color.transfer.is_hdr()
            && !out.color.transfer.is_hdr()
            && self.options.tone_map != ToneMapOp::None;
        out.side_data.clear();
        for sd in &src.side_data {
            match sd {
                SideData::DisplayMatrix(_) => {}
                SideData::MasteringDisplay(_)
                | SideData::ContentLightLevel { .. }
                | SideData::Hdr10Plus(_)
                    if tone_mapped => {}
                other => out.side_data.push(other.clone()),
            }
        }
    }

    pub fn new(src: VideoContract, dst: VideoContract, options: SwsOptions) -> Self {
        Self {
            ops: Vec::new(),
            src,
            dst,
            options,
        }
    }

    /// Generate and optimize a complete `SwsOpList` converting `src` to `dst`.
    pub fn build(src: &VideoContract, dst: &VideoContract, opts: SwsOptions) -> Result<Self> {
        validate_contract(src)?;
        validate_contract(dst)?;

        let (cw, ch) = src.crop.display_size(src.width, src.height);
        if cw == 0 || ch == 0 {
            return Err(Error::invalid_argument(
                "wedeo-video-ops",
                "crop leaves zero active dimensions",
            ));
        }

        let dt = DisplayTransform::from_matrix(&src.display_matrix)?;
        let (oriented_w, oriented_h) = dt.output_dims(cw, ch);

        let src_color = color::resolve_color(src.layout, src.color, None);
        let dst_color = color::resolve_color(dst.layout, dst.color, Some(src_color));

        let mut list = Self::new(src.clone(), dst.clone(), opts);
        list.src.color = src_color;
        list.dst.color = dst_color;

        // 1. Read + crop + rotation/flip
        list.ops.push(SwsOp::Read {
            layout: src.layout,
            crop: src.crop,
            display_matrix: src.display_matrix,
        });

        // 2. Unpack bit shift + normalize to [0, 1]
        list.ops.push(SwsOp::Unpack {
            depth: src.layout.depth,
            shift: layout_shift(src.layout),
        });

        // 3. Canonicalize component order: [R, G, B, A] or [Y, Cb, Cr, A]
        list.ops
            .push(SwsOp::Swizzle(read_to_canonical_swizzle(src.layout)));

        // 4. Decode range + YCbCr/Gray -> non-linear R'G'B' [0, 1]
        let decode_mat = color::decode_to_rgb_matrix(src.layout, src_color)?;
        list.ops.push(SwsOp::ColorMatrix(decode_mat));

        // 5. Linearize transfer curve
        list.ops.push(SwsOp::Linearize(src_color.transfer));

        // 6. Gamut primaries conversion in linear light
        let gamut_mat = color::primaries_matrix(src_color.primaries, dst_color.primaries);
        list.ops.push(SwsOp::ColorMatrix(gamut_mat));

        // 7. HDR -> SDR tone-mapping (only when converting HDR -> SDR)
        if src_color.transfer.is_hdr()
            && !dst_color.transfer.is_hdr()
            && opts.tone_map != ToneMapOp::None
        {
            list.ops.push(SwsOp::ToneMap(opts.tone_map));
        }

        // 8. Separable scaling
        if oriented_w != dst.width {
            list.ops.push(SwsOp::ScaleH {
                dst_w: dst.width,
                kernel: opts.scaler,
                chroma_loc: dst_color.chroma_location,
            });
        }
        if oriented_h != dst.height {
            list.ops.push(SwsOp::ScaleV {
                dst_h: dst.height,
                kernel: opts.scaler,
                chroma_loc: dst_color.chroma_location,
            });
        }

        // 9. Blend away alpha if source has alpha and destination does not
        if !matches!(src.layout.alpha, wedeo_core::AlphaMode::None)
            && matches!(dst.layout.alpha, wedeo_core::AlphaMode::None)
        {
            list.ops.push(SwsOp::BlendAlpha);
        }

        // 10. Delinearize transfer curve
        list.ops.push(SwsOp::Delinearize(dst_color.transfer));

        // 11. Encode non-linear R'G'B' -> dst YCbCr/Gray + range
        let encode_mat = color::encode_from_rgb_matrix(dst.layout, dst_color)?;
        list.ops.push(SwsOp::ColorMatrix(encode_mat));

        // 12. Dither before quantization
        if dst.layout.depth != SampleDepth::F32 && opts.dither != DitherMode::None {
            list.ops.push(SwsOp::Dither(opts.dither));
        }

        // 13. Swizzle canonical -> destination memory order
        list.ops
            .push(SwsOp::Swizzle(canonical_to_write_swizzle(dst.layout)));

        // 14. Pack [0, 1] -> destination integer sample representation & bit shift
        list.ops.push(SwsOp::Pack(dst.layout));

        // 15. Write to destination planes (with chroma downsampling if subsampled)
        list.ops.push(SwsOp::Write(dst.layout));

        list.optimize();
        Ok(list)
    }

    /// Run FFmpeg `n9.1` `ops_optimizer.c` algebraic simplification passes.
    pub fn optimize(&mut self) {
        crate::optimizer::optimize(self);
    }

    /// Returns `true` if the operation list is a true zero-copy identity pass-through.
    pub fn is_noop(&self) -> bool {
        crate::optimizer::is_noop(self)
    }
}

/// Bit shift required for MSB-aligned packed 10/12/14-bit formats (such as `P010`).
#[inline]
pub const fn layout_shift(layout: PixelLayout) -> u8 {
    if matches!(layout.packing, PlanePacking::SemiPlanar)
        && matches!(
            layout.depth,
            SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14
        )
    {
        16 - layout.depth.bits()
    } else {
        0
    }
}

/// Component permutation mapping `Read` output order to canonical `[R/Y, G/Cb, B/Cr, A]`.
pub const fn read_to_canonical_swizzle(layout: PixelLayout) -> [u8; 4] {
    match layout.order {
        CompOrder::Yuv
        | CompOrder::Rgb
        | CompOrder::Rgba
        | CompOrder::Gray
        | CompOrder::Yuyv
        | CompOrder::Uyvy => [0, 1, 2, 3],
        CompOrder::Yvu => [0, 2, 1, 3],
        CompOrder::Bgr | CompOrder::Bgra => [2, 1, 0, 3],
        CompOrder::Argb => [1, 2, 3, 0],
        CompOrder::Abgr => [3, 2, 1, 0],
        CompOrder::Gbr => [2, 0, 1, 3],
    }
}

/// Component permutation mapping canonical `[R/Y, G/Cb, B/Cr, A]` to `Write` memory order.
pub const fn canonical_to_write_swizzle(layout: PixelLayout) -> [u8; 4] {
    match layout.order {
        CompOrder::Yuv
        | CompOrder::Rgb
        | CompOrder::Rgba
        | CompOrder::Gray
        | CompOrder::Yuyv
        | CompOrder::Uyvy => [0, 1, 2, 3],
        CompOrder::Yvu => [0, 2, 1, 3],
        CompOrder::Bgr | CompOrder::Bgra => [2, 1, 0, 3],
        CompOrder::Argb => [3, 0, 1, 2],
        CompOrder::Abgr => [3, 2, 1, 0],
        CompOrder::Gbr => [1, 2, 0, 3],
    }
}

pub(crate) fn validate_contract(c: &VideoContract) -> Result<()> {
    if c.width == 0 || c.height == 0 {
        return Err(Error::invalid_argument(
            "wedeo-video-ops",
            "video dimensions must be non-zero",
        ));
    }
    if c.crop.left.saturating_add(c.crop.right) >= c.width
        || c.crop.top.saturating_add(c.crop.bottom) >= c.height
    {
        return Err(Error::invalid_argument(
            "wedeo-video-ops",
            "crop rectangle exceeds frame dimensions",
        ));
    }
    if matches!(c.layout.model, ColorModel::Xyz) {
        return Err(Error::unsupported(
            "wedeo-video-ops",
            "XYZ color model is not supported",
        ));
    }
    if matches!(c.layout.depth, SampleDepth::F16) {
        return Err(Error::unsupported(
            "wedeo-video-ops",
            "F16 sample depth is not yet supported",
        ));
    }
    DisplayTransform::from_matrix(&c.display_matrix)?;
    Ok(())
}
