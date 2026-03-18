use std::fmt;

/// Audio sample format, matching FFmpeg's AVSampleFormat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SampleFormat {
    None = -1,
    /// Unsigned 8 bits.
    U8 = 0,
    /// Signed 16 bits.
    S16 = 1,
    /// Signed 32 bits.
    S32 = 2,
    /// Float.
    Flt = 3,
    /// Double.
    Dbl = 4,
    /// Unsigned 8 bits, planar.
    U8p = 5,
    /// Signed 16 bits, planar.
    S16p = 6,
    /// Signed 32 bits, planar.
    S32p = 7,
    /// Float, planar.
    Fltp = 8,
    /// Double, planar.
    Dblp = 9,
    /// Signed 64 bits.
    S64 = 10,
    /// Signed 64 bits, planar.
    S64p = 11,
}

impl SampleFormat {
    /// Return the number of bytes per sample.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            SampleFormat::None => 0,
            SampleFormat::U8 | SampleFormat::U8p => 1,
            SampleFormat::S16 | SampleFormat::S16p => 2,
            SampleFormat::S32 | SampleFormat::S32p | SampleFormat::Flt | SampleFormat::Fltp => 4,
            SampleFormat::Dbl | SampleFormat::Dblp | SampleFormat::S64 | SampleFormat::S64p => 8,
        }
    }

    /// Check if the sample format is planar.
    pub fn is_planar(self) -> bool {
        matches!(
            self,
            SampleFormat::U8p
                | SampleFormat::S16p
                | SampleFormat::S32p
                | SampleFormat::Fltp
                | SampleFormat::Dblp
                | SampleFormat::S64p
        )
    }

    /// Get the packed equivalent of a planar format (or return self if already packed).
    pub fn packed(self) -> Self {
        match self {
            SampleFormat::U8p => SampleFormat::U8,
            SampleFormat::S16p => SampleFormat::S16,
            SampleFormat::S32p => SampleFormat::S32,
            SampleFormat::Fltp => SampleFormat::Flt,
            SampleFormat::Dblp => SampleFormat::Dbl,
            SampleFormat::S64p => SampleFormat::S64,
            other => other,
        }
    }

    /// Get the planar equivalent of a packed format (or return self if already planar).
    pub fn planar(self) -> Self {
        match self {
            SampleFormat::U8 => SampleFormat::U8p,
            SampleFormat::S16 => SampleFormat::S16p,
            SampleFormat::S32 => SampleFormat::S32p,
            SampleFormat::Flt => SampleFormat::Fltp,
            SampleFormat::Dbl => SampleFormat::Dblp,
            SampleFormat::S64 => SampleFormat::S64p,
            other => other,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SampleFormat::None => "none",
            SampleFormat::U8 => "u8",
            SampleFormat::S16 => "s16",
            SampleFormat::S32 => "s32",
            SampleFormat::Flt => "flt",
            SampleFormat::Dbl => "dbl",
            SampleFormat::U8p => "u8p",
            SampleFormat::S16p => "s16p",
            SampleFormat::S32p => "s32p",
            SampleFormat::Fltp => "fltp",
            SampleFormat::Dblp => "dblp",
            SampleFormat::S64 => "s64",
            SampleFormat::S64p => "s64p",
        }
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
