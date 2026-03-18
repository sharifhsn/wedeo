use std::fmt;

/// Media type, matching FFmpeg's AVMediaType.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum MediaType {
    Video = 0,
    Audio = 1,
    Data = 2,
    Subtitle = 3,
    Attachment = 4,
}

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediaType::Video => write!(f, "video"),
            MediaType::Audio => write!(f, "audio"),
            MediaType::Data => write!(f, "data"),
            MediaType::Subtitle => write!(f, "subtitle"),
            MediaType::Attachment => write!(f, "attachment"),
        }
    }
}
