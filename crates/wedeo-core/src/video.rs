use std::fmt;
use std::sync::Arc;

use bitflags::bitflags;

use crate::buffer::{BorderedPlane, GpuSurface, MemoryDomain};
use crate::media::SideData;
use crate::time::{Rational, Timestamp};

/// Orthogonal color model stored in pixel planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorModel {
    Ycbcr,
    Rgb,
    Gray,
    Xyz,
}

/// Plane memory arrangement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanePacking {
    /// Separate plane per component (e.g. YUV420p, GBRP).
    Planar,
    /// Luma plane + interleaved chroma plane (e.g. NV12, NV21, P010).
    SemiPlanar,
    /// All components interleaved in a single plane (e.g. RGBA8, RGB24, YUYV422).
    Packed,
}

/// Chroma subsampling geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Subsampling {
    Cs444,
    Cs422,
    Cs420,
    Cs411,
    Cs410,
    Cs400,
}

impl Subsampling {
    /// Compute chroma plane dimensions in pixels using ceiling division (`div_ceil`)
    /// so odd frame dimensions match FFmpeg / AV1 / VP9 / H.264 rules.
    #[inline]
    pub const fn chroma_dims(self, width: u32, height: u32) -> (u32, u32) {
        match self {
            Self::Cs444 => (width, height),
            Self::Cs422 => (width.div_ceil(2), height),
            Self::Cs420 => (width.div_ceil(2), height.div_ceil(2)),
            Self::Cs411 => (width.div_ceil(4), height),
            Self::Cs410 => (width.div_ceil(4), height.div_ceil(4)),
            Self::Cs400 => (0, 0),
        }
    }
}

/// Component sample precision & numeric representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleDepth {
    U8,
    U10,
    U12,
    U14,
    U16,
    F16,
    F32,
}

impl SampleDepth {
    #[inline]
    pub const fn bits(self) -> u8 {
        match self {
            Self::U8 => 8,
            Self::U10 => 10,
            Self::U12 => 12,
            Self::U14 => 14,
            Self::U16 | Self::F16 => 16,
            Self::F32 => 32,
        }
    }

    #[inline]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U10 | Self::U12 | Self::U14 | Self::U16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Component byte order for multi-byte samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Endian {
    Little,
    Big,
}

/// Alpha channel semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    None,
    Straight,
    Premultiplied,
}

/// Component ordering within planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompOrder {
    Yuv,
    Yvu,
    Rgb,
    Bgr,
    Gbr,
    Rgba,
    Bgra,
    Argb,
    Abgr,
    Yuyv,
    Uyvy,
    Gray,
}

/// Orthogonal memory layout description for video frames (decoupled from `ColorDesc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelLayout {
    pub model: ColorModel,
    pub packing: PlanePacking,
    pub subsampling: Subsampling,
    pub depth: SampleDepth,
    pub endian: Endian,
    pub alpha: AlphaMode,
    pub order: CompOrder,
}

impl PixelLayout {
    pub const YUV420P: Self = Self {
        model: ColorModel::Ycbcr,
        packing: PlanePacking::Planar,
        subsampling: Subsampling::Cs420,
        depth: SampleDepth::U8,
        endian: Endian::Little,
        alpha: AlphaMode::None,
        order: CompOrder::Yuv,
    };

    pub const YUV420P10LE: Self = Self {
        depth: SampleDepth::U10,
        ..Self::YUV420P
    };

    pub const YUV420P10BE: Self = Self {
        depth: SampleDepth::U10,
        endian: Endian::Big,
        ..Self::YUV420P
    };

    pub const YUV420P12LE: Self = Self {
        depth: SampleDepth::U12,
        ..Self::YUV420P
    };

    pub const YUV420P16LE: Self = Self {
        depth: SampleDepth::U16,
        ..Self::YUV420P
    };

    pub const YUV422P: Self = Self {
        subsampling: Subsampling::Cs422,
        ..Self::YUV420P
    };

    pub const YUV422P10LE: Self = Self {
        depth: SampleDepth::U10,
        ..Self::YUV422P
    };

    pub const YUV444P: Self = Self {
        subsampling: Subsampling::Cs444,
        ..Self::YUV420P
    };

    pub const YUV444P10LE: Self = Self {
        depth: SampleDepth::U10,
        ..Self::YUV444P
    };

    pub const NV12: Self = Self {
        model: ColorModel::Ycbcr,
        packing: PlanePacking::SemiPlanar,
        subsampling: Subsampling::Cs420,
        depth: SampleDepth::U8,
        endian: Endian::Little,
        alpha: AlphaMode::None,
        order: CompOrder::Yuv,
    };

    pub const NV21: Self = Self {
        order: CompOrder::Yvu,
        ..Self::NV12
    };

    pub const P010: Self = Self {
        depth: SampleDepth::U10,
        ..Self::NV12
    };

    pub const RGB24: Self = Self {
        model: ColorModel::Rgb,
        packing: PlanePacking::Packed,
        subsampling: Subsampling::Cs444,
        depth: SampleDepth::U8,
        endian: Endian::Little,
        alpha: AlphaMode::None,
        order: CompOrder::Rgb,
    };

    pub const BGR24: Self = Self {
        order: CompOrder::Bgr,
        ..Self::RGB24
    };

    pub const RGBA8: Self = Self {
        alpha: AlphaMode::Straight,
        order: CompOrder::Rgba,
        ..Self::RGB24
    };

    pub const BGRA8: Self = Self {
        order: CompOrder::Bgra,
        ..Self::RGBA8
    };

    pub const ARGB8: Self = Self {
        order: CompOrder::Argb,
        ..Self::RGBA8
    };

    pub const ABGR8: Self = Self {
        order: CompOrder::Abgr,
        ..Self::RGBA8
    };

    pub const RGBA16F: Self = Self {
        depth: SampleDepth::F16,
        ..Self::RGBA8
    };

    pub const RGBA32F: Self = Self {
        depth: SampleDepth::F32,
        ..Self::RGBA8
    };

    pub const GBRP: Self = Self {
        model: ColorModel::Rgb,
        packing: PlanePacking::Planar,
        subsampling: Subsampling::Cs444,
        depth: SampleDepth::U8,
        endian: Endian::Little,
        alpha: AlphaMode::None,
        order: CompOrder::Gbr,
    };

    pub const GBRP10LE: Self = Self {
        depth: SampleDepth::U10,
        ..Self::GBRP
    };

    pub const GRAY8: Self = Self {
        model: ColorModel::Gray,
        packing: PlanePacking::Planar,
        subsampling: Subsampling::Cs400,
        depth: SampleDepth::U8,
        endian: Endian::Little,
        alpha: AlphaMode::None,
        order: CompOrder::Gray,
    };

    pub const GRAY16LE: Self = Self {
        depth: SampleDepth::U16,
        ..Self::GRAY8
    };

    pub const YUYV422: Self = Self {
        model: ColorModel::Ycbcr,
        packing: PlanePacking::Packed,
        subsampling: Subsampling::Cs422,
        depth: SampleDepth::U8,
        endian: Endian::Little,
        alpha: AlphaMode::None,
        order: CompOrder::Yuyv,
    };

    pub const UYVY422: Self = Self {
        order: CompOrder::Uyvy,
        ..Self::YUYV422
    };

    /// Number of memory planes required for this layout.
    pub const fn plane_count(self) -> usize {
        let alpha = !matches!(self.alpha, AlphaMode::None) as usize;
        match self.packing {
            PlanePacking::Packed => 1,
            PlanePacking::SemiPlanar => 2 + alpha,
            PlanePacking::Planar => match self.model {
                ColorModel::Gray => 1 + alpha,
                ColorModel::Ycbcr | ColorModel::Rgb | ColorModel::Xyz => 3 + alpha,
            },
        }
    }

    /// Compute `(width_bytes, height_rows)` for plane `plane_idx` of a `width x height` frame,
    /// using ceiling division (`div_ceil`) for odd-dimension chroma subsampling.
    pub const fn plane_dims(
        self,
        width: u32,
        height: u32,
        plane_idx: usize,
    ) -> Option<(usize, usize)> {
        if plane_idx >= self.plane_count() {
            return None;
        }
        let bps = self.depth.bytes_per_sample();
        let w = width as usize;
        let h = height as usize;

        match self.packing {
            PlanePacking::Packed => {
                let row_bytes = match self.order {
                    CompOrder::Yuyv | CompOrder::Uyvy => w.div_ceil(2) * 4 * bps,
                    CompOrder::Rgb | CompOrder::Bgr => w * 3 * bps,
                    CompOrder::Rgba | CompOrder::Bgra | CompOrder::Argb | CompOrder::Abgr => {
                        w * 4 * bps
                    }
                    CompOrder::Gray => w * bps,
                    _ => w * 3 * bps,
                };
                Some((row_bytes, h))
            }
            PlanePacking::SemiPlanar => {
                if plane_idx == 0 || plane_idx == 2 {
                    Some((w * bps, h))
                } else {
                    let (cw, ch) = self.subsampling.chroma_dims(width, height);
                    // Interleaved UV: 2 chroma components per chroma sample
                    Some((cw as usize * 2 * bps, ch as usize))
                }
            }
            PlanePacking::Planar => {
                if plane_idx == 0 || plane_idx == 3 {
                    Some((w * bps, h))
                } else {
                    let (cw, ch) = self.subsampling.chroma_dims(width, height);
                    Some((cw as usize * bps, ch as usize))
                }
            }
        }
    }

    /// FFmpeg-compatible canonical name for common layouts.
    pub fn name(self) -> &'static str {
        match self {
            Self::YUV420P => "yuv420p",
            Self::YUV420P10LE => "yuv420p10le",
            Self::YUV420P10BE => "yuv420p10be",
            Self::YUV420P12LE => "yuv420p12le",
            Self::YUV420P16LE => "yuv420p16le",
            Self::YUV422P => "yuv422p",
            Self::YUV422P10LE => "yuv422p10le",
            Self::YUV444P => "yuv444p",
            Self::YUV444P10LE => "yuv444p10le",
            Self::NV12 => "nv12",
            Self::NV21 => "nv21",
            Self::P010 => "p010le",
            Self::RGB24 => "rgb24",
            Self::BGR24 => "bgr24",
            Self::RGBA8 => "rgba",
            Self::BGRA8 => "bgra",
            Self::ARGB8 => "argb",
            Self::ABGR8 => "abgr",
            Self::RGBA16F => "rgbaf16le",
            Self::RGBA32F => "rgbaf32le",
            Self::GBRP => "gbrp",
            Self::GBRP10LE => "gbrp10le",
            Self::GRAY8 => "gray",
            Self::GRAY16LE => "gray16le",
            Self::YUYV422 => "yuyv422",
            Self::UYVY422 => "uyvy422",
            _ => "custom",
        }
    }
}

impl fmt::Display for PixelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// CICP / ITU-T H.273 color primaries (matches FFmpeg `AVColorPrimaries`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ColorPrimaries {
    Reserved0 = 0,
    Bt709 = 1,
    #[default]
    Unspecified = 2,
    Reserved = 3,
    Bt470m = 4,
    Bt470bg = 5,
    Smpte170m = 6,
    Smpte240m = 7,
    Film = 8,
    Bt2020 = 9,
    Smpte428 = 10,
    Smpte431 = 11,
    Smpte432 = 12,
    Ebu3213 = 22,
}

impl ColorPrimaries {
    pub fn from_cicp(v: u8) -> Self {
        match v {
            0 => Self::Reserved0,
            1 => Self::Bt709,
            2 => Self::Unspecified,
            3 => Self::Reserved,
            4 => Self::Bt470m,
            5 => Self::Bt470bg,
            6 => Self::Smpte170m,
            7 => Self::Smpte240m,
            8 => Self::Film,
            9 => Self::Bt2020,
            10 => Self::Smpte428,
            11 => Self::Smpte431,
            12 => Self::Smpte432,
            22 => Self::Ebu3213,
            _ => Self::Unspecified,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Reserved0 | Self::Reserved => "reserved",
            Self::Bt709 => "bt709",
            Self::Unspecified => "unknown",
            Self::Bt470m => "bt470m",
            Self::Bt470bg => "bt470bg",
            Self::Smpte170m => "smpte170m",
            Self::Smpte240m => "smpte240m",
            Self::Film => "film",
            Self::Bt2020 => "bt2020",
            Self::Smpte428 => "smpte428",
            Self::Smpte431 => "smpte431",
            Self::Smpte432 => "smpte432",
            Self::Ebu3213 => "ebu3213",
        }
    }
}

/// CICP / ITU-T H.273 transfer characteristics (matches FFmpeg `AVColorTransferCharacteristic`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum TransferCharacteristic {
    Reserved0 = 0,
    Bt709 = 1,
    #[default]
    Unspecified = 2,
    Reserved = 3,
    Gamma22 = 4,
    Gamma28 = 5,
    Smpte170m = 6,
    Smpte240m = 7,
    Linear = 8,
    Log = 9,
    LogSqrt = 10,
    Iec61966_2_4 = 11,
    Bt1361Ecg = 12,
    Srgb = 13, // IEC 61966-2-1
    Bt2020_10 = 14,
    Bt2020_12 = 15,
    Pq = 16, // SMPTE ST 2084
    Smpte428 = 17,
    Hlg = 18, // ARIB STD-B67
}

impl TransferCharacteristic {
    pub fn from_cicp(v: u8) -> Self {
        match v {
            0 => Self::Reserved0,
            1 => Self::Bt709,
            2 => Self::Unspecified,
            3 => Self::Reserved,
            4 => Self::Gamma22,
            5 => Self::Gamma28,
            6 => Self::Smpte170m,
            7 => Self::Smpte240m,
            8 => Self::Linear,
            9 => Self::Log,
            10 => Self::LogSqrt,
            11 => Self::Iec61966_2_4,
            12 => Self::Bt1361Ecg,
            13 => Self::Srgb,
            14 => Self::Bt2020_10,
            15 => Self::Bt2020_12,
            16 => Self::Pq,
            17 => Self::Smpte428,
            18 => Self::Hlg,
            _ => Self::Unspecified,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Reserved0 | Self::Reserved => "reserved",
            Self::Bt709 => "bt709",
            Self::Unspecified => "unknown",
            Self::Gamma22 => "bt470m",
            Self::Gamma28 => "bt470bg",
            Self::Smpte170m => "smpte170m",
            Self::Smpte240m => "smpte240m",
            Self::Linear => "linear",
            Self::Log => "log100",
            Self::LogSqrt => "log316",
            Self::Iec61966_2_4 => "iec61966-2-4",
            Self::Bt1361Ecg => "bt1361e",
            Self::Srgb => "iec61966-2-1",
            Self::Bt2020_10 => "bt2020-10",
            Self::Bt2020_12 => "bt2020-12",
            Self::Pq => "smpte2084",
            Self::Smpte428 => "smpte428",
            Self::Hlg => "arib-std-b67",
        }
    }

    #[inline]
    pub const fn is_hdr(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }
}

/// CICP / ITU-T H.273 matrix coefficients (matches FFmpeg `AVColorSpace`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Identity = 0, // RGB / GBR
    Bt709 = 1,
    #[default]
    Unspecified = 2,
    Fcc = 4,
    Bt470bg = 5,
    Smpte170m = 6,
    Smpte240m = 7,
    Ycgco = 8,
    Bt2020Ncl = 9,
    Bt2020Cl = 10,
    Smpte2085 = 11,
    ChromaDerivedNcl = 12,
    ChromaDerivedCl = 13,
    Ictcp = 14,
}

impl MatrixCoefficients {
    pub fn from_cicp(v: u8) -> Self {
        match v {
            0 => Self::Identity,
            1 => Self::Bt709,
            2 => Self::Unspecified,
            4 => Self::Fcc,
            5 => Self::Bt470bg,
            6 => Self::Smpte170m,
            7 => Self::Smpte240m,
            8 => Self::Ycgco,
            9 => Self::Bt2020Ncl,
            10 => Self::Bt2020Cl,
            11 => Self::Smpte2085,
            12 => Self::ChromaDerivedNcl,
            13 => Self::ChromaDerivedCl,
            14 => Self::Ictcp,
            _ => Self::Unspecified,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Identity => "gbr",
            Self::Bt709 => "bt709",
            Self::Unspecified => "unknown",
            Self::Fcc => "fcc",
            Self::Bt470bg => "bt470bg",
            Self::Smpte170m => "smpte170m",
            Self::Smpte240m => "smpte240m",
            Self::Ycgco => "ycgco",
            Self::Bt2020Ncl => "bt2020nc",
            Self::Bt2020Cl => "bt2020c",
            Self::Smpte2085 => "smpte2085",
            Self::ChromaDerivedNcl => "chroma-derived-nc",
            Self::ChromaDerivedCl => "chroma-derived-c",
            Self::Ictcp => "ictcp",
        }
    }
}

/// Signal nominal range (matches FFmpeg `AVColorRange`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ColorRange {
    #[default]
    Unspecified = 0,
    /// Limited ("TV" / MPEG) range: `Y ∈ [16, 235]`, `Cb/Cr ∈ [16, 240]` (8-bit).
    Limited = 1,
    /// Full ("PC" / JPEG) range: `[0, 2^n - 1]`.
    Full = 2,
}

/// Chroma sample siting (matches FFmpeg `AVChromaLocation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ChromaLocation {
    #[default]
    Unspecified = 0,
    Left = 1,
    Center = 2,
    TopLeft = 3,
    Top = 4,
    BottomLeft = 5,
    Bottom = 6,
}

/// Complete, explicit colorimetry description attached to every `VideoFrame` and stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ColorDesc {
    pub primaries: ColorPrimaries,
    pub transfer: TransferCharacteristic,
    pub matrix: MatrixCoefficients,
    pub range: ColorRange,
    pub chroma_location: ChromaLocation,
}

impl ColorDesc {
    pub const UNSPECIFIED: Self = Self {
        primaries: ColorPrimaries::Unspecified,
        transfer: TransferCharacteristic::Unspecified,
        matrix: MatrixCoefficients::Unspecified,
        range: ColorRange::Unspecified,
        chroma_location: ChromaLocation::Unspecified,
    };

    pub const BT709_LIMITED: Self = Self {
        primaries: ColorPrimaries::Bt709,
        transfer: TransferCharacteristic::Bt709,
        matrix: MatrixCoefficients::Bt709,
        range: ColorRange::Limited,
        chroma_location: ChromaLocation::Left,
    };

    pub const BT601_LIMITED: Self = Self {
        primaries: ColorPrimaries::Smpte170m,
        transfer: TransferCharacteristic::Smpte170m,
        matrix: MatrixCoefficients::Smpte170m,
        range: ColorRange::Limited,
        chroma_location: ChromaLocation::Left,
    };

    pub const BT2020_PQ: Self = Self {
        primaries: ColorPrimaries::Bt2020,
        transfer: TransferCharacteristic::Pq,
        matrix: MatrixCoefficients::Bt2020Ncl,
        range: ColorRange::Limited,
        chroma_location: ChromaLocation::TopLeft,
    };

    pub const BT2020_HLG: Self = Self {
        primaries: ColorPrimaries::Bt2020,
        transfer: TransferCharacteristic::Hlg,
        matrix: MatrixCoefficients::Bt2020Ncl,
        range: ColorRange::Limited,
        chroma_location: ChromaLocation::TopLeft,
    };

    pub const SRGB: Self = Self {
        primaries: ColorPrimaries::Bt709,
        transfer: TransferCharacteristic::Srgb,
        matrix: MatrixCoefficients::Identity,
        range: ColorRange::Full,
        chroma_location: ChromaLocation::Unspecified,
    };
}

/// Crop margins in pixels to discard from each edge for presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CropRect {
    pub top: u32,
    pub bottom: u32,
    pub left: u32,
    pub right: u32,
}

impl CropRect {
    pub const ZERO: Self = Self {
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
    };

    #[inline]
    pub const fn display_size(self, coded_w: u32, coded_h: u32) -> (u32, u32) {
        let w = coded_w.saturating_sub(self.left).saturating_sub(self.right);
        let h = coded_h.saturating_sub(self.top).saturating_sub(self.bottom);
        (w, h)
    }
}

/// Picture coding type, matching FFmpeg's `AVPictureType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum PictureType {
    #[default]
    None = 0,
    I = 1,
    P = 2,
    B = 3,
    S = 4,
    Si = 5,
    Sp = 6,
    Bi = 7,
}

bitflags! {
    /// Video frame flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct FrameFlags: u32 {
        const CORRUPT         = 1 << 0;
        const KEY             = 1 << 1;
        const DISCARD         = 1 << 2;
        const INTERLACED      = 1 << 3;
        const TOP_FIELD_FIRST = 1 << 4;
    }
}

/// Video stream-level parameters.
///
/// VFR-native: `avg_frame_rate` is advisory metadata only, never used to compute PTS.
#[derive(Debug, Clone)]
pub struct VideoStreamParams {
    pub width: u32,
    pub height: u32,
    pub layout: PixelLayout,
    pub color: ColorDesc,
    pub crop: CropRect,
    pub sar: Rational,
    pub avg_frame_rate: Rational,
    pub display_matrix: Option<[i32; 9]>,
    pub bit_rate: i64,
    pub extradata: Vec<u8>,
}

impl VideoStreamParams {
    pub fn new(width: u32, height: u32, layout: PixelLayout) -> Self {
        Self {
            width,
            height,
            layout,
            color: ColorDesc::UNSPECIFIED,
            crop: CropRect::ZERO,
            sar: Rational::new(1, 1),
            avg_frame_rate: Rational::new(0, 1),
            display_matrix: None,
            bit_rate: 0,
            extradata: Vec::new(),
        }
    }
}

/// Backing storage for a `VideoFrame` — either CPU planes or a zero-copy GPU surface.
#[derive(Clone)]
pub enum VideoStorage {
    Cpu(Vec<BorderedPlane>),
    Gpu(Arc<dyn GpuSurface>),
}

impl fmt::Debug for VideoStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu(planes) => f.debug_tuple("Cpu").field(&planes.len()).finish(),
            Self::Gpu(surf) => f
                .debug_struct("Gpu")
                .field("device_id", &surf.device_id())
                .field("layout", &surf.layout())
                .field("width", &surf.width())
                .field("height", &surf.height())
                .finish(),
        }
    }
}

/// VFR-native decoded video frame supporting CPU `BorderedPlane`s and zero-copy GPU surfaces.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub storage: VideoStorage,
    pub width: u32,
    pub height: u32,
    pub layout: PixelLayout,
    pub color: ColorDesc,
    pub crop: CropRect,
    pub sar: Rational,
    pub picture_type: PictureType,
    pub flags: FrameFlags,
    pub repeat_pict: i32,
    pub pts: Timestamp,
    pub pkt_dts: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
    pub side_data: Vec<SideData>,
}

impl VideoFrame {
    pub fn from_planes(
        width: u32,
        height: u32,
        layout: PixelLayout,
        planes: Vec<BorderedPlane>,
    ) -> Self {
        Self::with_storage(width, height, layout, VideoStorage::Cpu(planes))
    }

    pub fn from_gpu(surface: Arc<dyn GpuSurface>) -> Self {
        Self::with_storage(
            surface.width(),
            surface.height(),
            surface.layout(),
            VideoStorage::Gpu(surface),
        )
    }

    fn with_storage(width: u32, height: u32, layout: PixelLayout, storage: VideoStorage) -> Self {
        Self {
            storage,
            width,
            height,
            layout,
            color: ColorDesc::UNSPECIFIED,
            crop: CropRect::ZERO,
            sar: Rational::new(1, 1),
            picture_type: PictureType::None,
            flags: FrameFlags::empty(),
            repeat_pict: 0,
            pts: Timestamp::NONE,
            pkt_dts: Timestamp::NONE,
            duration: 0,
            time_base: Rational::new(0, 1),
            side_data: Vec::new(),
        }
    }

    #[inline]
    pub fn domain(&self) -> MemoryDomain {
        match &self.storage {
            VideoStorage::Cpu(_) => MemoryDomain::Cpu,
            VideoStorage::Gpu(surf) => MemoryDomain::Gpu(surf.device_id()),
        }
    }

    #[inline]
    pub fn planes(&self) -> Option<&[BorderedPlane]> {
        match &self.storage {
            VideoStorage::Cpu(p) => Some(p),
            VideoStorage::Gpu(_) => None,
        }
    }

    #[inline]
    pub fn planes_mut(&mut self) -> Option<&mut [BorderedPlane]> {
        match &mut self.storage {
            VideoStorage::Cpu(p) => Some(p),
            VideoStorage::Gpu(_) => None,
        }
    }

    #[inline]
    pub fn gpu_surface(&self) -> Option<&Arc<dyn GpuSurface>> {
        match &self.storage {
            VideoStorage::Gpu(s) => Some(s),
            VideoStorage::Cpu(_) => None,
        }
    }

    #[inline]
    pub const fn display_size(&self) -> (u32, u32) {
        self.crop.display_size(self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_layout_plane_dims_uses_div_ceil_for_odd_sizes() {
        // 100x75 YUV420p (like vp9_blue_100x75.ivf!)
        assert_eq!(PixelLayout::YUV420P.plane_count(), 3);
        assert_eq!(PixelLayout::YUV420P.plane_dims(100, 75, 0), Some((100, 75)));
        assert_eq!(PixelLayout::YUV420P.plane_dims(100, 75, 1), Some((50, 38)));
        assert_eq!(PixelLayout::YUV420P.plane_dims(100, 75, 2), Some((50, 38)));
        assert_eq!(PixelLayout::YUV420P.plane_dims(100, 75, 3), None);

        // 10-bit P010: plane 0 = 2 bytes/px luma, plane 1 = 4 bytes/chroma-pair
        assert_eq!(PixelLayout::P010.plane_count(), 2);
        assert_eq!(PixelLayout::P010.plane_dims(101, 75, 0), Some((202, 75)));
        assert_eq!(PixelLayout::P010.plane_dims(101, 75, 1), Some((204, 38)));

        // GBRP (VP9 Profile 1)
        assert_eq!(PixelLayout::GBRP.plane_count(), 3);
        assert_eq!(PixelLayout::GBRP.plane_dims(64, 64, 1), Some((64, 64)));
    }
}
