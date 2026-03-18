use bitflags::bitflags;

/// Pixel format, matching a subset of FFmpeg's AVPixelFormat.
/// Only the most common formats are included initially.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum PixelFormat {
    None = -1,
    Yuv420p = 0,
    Yuyv422 = 1,
    Rgb24 = 2,
    Bgr24 = 3,
    Yuv422p = 4,
    Yuv444p = 5,
    Yuv410p = 6,
    Yuv411p = 7,
    Gray8 = 8,
    MonoWhite = 9,
    MonoBlack = 10,
    Pal8 = 11,
    Yuvj420p = 12,
    Yuvj422p = 13,
    Yuvj444p = 14,
    Uyvy422 = 15,
    Nv12 = 25,
    Nv21 = 26,
    Argb = 27,
    Rgba = 28,
    Abgr = 29,
    Bgra = 30,
    Gray16be = 31,
    Gray16le = 32,
    Yuv420p16le = 55,
    Yuv420p16be = 56,
    Rgb48be = 41,
    Rgb48le = 42,
    Yuv420p10le = 90,
    Yuv420p10be = 91,
}

impl PixelFormat {
    pub fn name(self) -> &'static str {
        match self {
            PixelFormat::None => "none",
            PixelFormat::Yuv420p => "yuv420p",
            PixelFormat::Yuyv422 => "yuyv422",
            PixelFormat::Rgb24 => "rgb24",
            PixelFormat::Bgr24 => "bgr24",
            PixelFormat::Yuv422p => "yuv422p",
            PixelFormat::Yuv444p => "yuv444p",
            PixelFormat::Yuv410p => "yuv410p",
            PixelFormat::Yuv411p => "yuv411p",
            PixelFormat::Gray8 => "gray8",
            PixelFormat::MonoWhite => "monowhite",
            PixelFormat::MonoBlack => "monoblack",
            PixelFormat::Pal8 => "pal8",
            PixelFormat::Yuvj420p => "yuvj420p",
            PixelFormat::Yuvj422p => "yuvj422p",
            PixelFormat::Yuvj444p => "yuvj444p",
            PixelFormat::Uyvy422 => "uyvy422",
            PixelFormat::Nv12 => "nv12",
            PixelFormat::Nv21 => "nv21",
            PixelFormat::Argb => "argb",
            PixelFormat::Rgba => "rgba",
            PixelFormat::Abgr => "abgr",
            PixelFormat::Bgra => "bgra",
            PixelFormat::Gray16be => "gray16be",
            PixelFormat::Gray16le => "gray16le",
            PixelFormat::Yuv420p16le => "yuv420p16le",
            PixelFormat::Yuv420p16be => "yuv420p16be",
            PixelFormat::Rgb48be => "rgb48be",
            PixelFormat::Rgb48le => "rgb48le",
            PixelFormat::Yuv420p10le => "yuv420p10le",
            PixelFormat::Yuv420p10be => "yuv420p10be",
        }
    }
}

bitflags! {
    /// Pixel format descriptor flags, matching FFmpeg's AV_PIX_FMT_FLAG_*.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PixelFormatFlags: u64 {
        const BIG_ENDIAN  = 1 << 0;
        const PAL         = 1 << 1;
        const BITSTREAM   = 1 << 2;
        const HWACCEL     = 1 << 3;
        const PLANAR      = 1 << 4;
        const RGB         = 1 << 5;
        const ALPHA       = 1 << 7;
        const BAYER       = 1 << 8;
        const FLOAT       = 1 << 9;
    }
}

/// Descriptor for a pixel format component.
#[derive(Debug, Clone, Copy)]
pub struct PixelFormatComponentDescriptor {
    /// Which of the 4 planes contains the component.
    pub plane: u16,
    /// Number of elements between two horizontally consecutive pixels.
    pub step: u16,
    /// Number of elements before the component of the first pixel.
    pub offset: u16,
    /// Number of least significant bits that must be shifted away to get the value.
    pub shift: u16,
    /// Number of bits in the component.
    pub depth: u16,
}

/// Descriptor for a pixel format.
#[derive(Debug, Clone)]
pub struct PixelFormatDescriptor {
    pub name: &'static str,
    pub nb_components: u8,
    /// Amount to shift the luma width right to find the chroma width.
    pub log2_chroma_w: u8,
    /// Amount to shift the luma height right to find the chroma height.
    pub log2_chroma_h: u8,
    pub flags: PixelFormatFlags,
    pub components: [PixelFormatComponentDescriptor; 4],
}
