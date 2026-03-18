use std::fmt;

/// Chroma sample location, matching FFmpeg's AVChromaLocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
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

impl ChromaLocation {
    /// Returns the FFmpeg-compatible string name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::Left => "left",
            Self::Center => "center",
            Self::TopLeft => "topleft",
            Self::Top => "top",
            Self::BottomLeft => "bottomleft",
            Self::Bottom => "bottom",
        }
    }
}

impl fmt::Display for ChromaLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_unspecified() {
        assert_eq!(ChromaLocation::default(), ChromaLocation::Unspecified);
    }

    #[test]
    fn test_repr_values() {
        assert_eq!(ChromaLocation::Unspecified as i32, 0);
        assert_eq!(ChromaLocation::Left as i32, 1);
        assert_eq!(ChromaLocation::Center as i32, 2);
        assert_eq!(ChromaLocation::Bottom as i32, 6);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ChromaLocation::Left), "left");
        assert_eq!(format!("{}", ChromaLocation::TopLeft), "topleft");
    }
}
