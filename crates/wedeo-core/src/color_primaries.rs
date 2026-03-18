use std::fmt;

/// Color primaries, matching FFmpeg's AVColorPrimaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
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
    /// Returns the FFmpeg-compatible string name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Reserved0 => "reserved",
            Self::Bt709 => "bt709",
            Self::Unspecified => "unknown",
            Self::Reserved => "reserved",
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

impl fmt::Display for ColorPrimaries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_unspecified() {
        assert_eq!(ColorPrimaries::default(), ColorPrimaries::Unspecified);
    }

    #[test]
    fn test_repr_values() {
        assert_eq!(ColorPrimaries::Reserved0 as i32, 0);
        assert_eq!(ColorPrimaries::Bt709 as i32, 1);
        assert_eq!(ColorPrimaries::Bt2020 as i32, 9);
        assert_eq!(ColorPrimaries::Ebu3213 as i32, 22);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ColorPrimaries::Bt709), "bt709");
        assert_eq!(format!("{}", ColorPrimaries::Unspecified), "unknown");
        assert_eq!(format!("{}", ColorPrimaries::Smpte170m), "smpte170m");
    }
}
