use wedeo_core::error::Result;
use wedeo_core::packet::Packet;

/// A single subtitle rectangle/region.
#[derive(Debug, Clone)]
pub enum SubtitleRect {
    /// Bitmap subtitle (e.g., DVD subtitles).
    Bitmap {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: Vec<u8>,
        palette: Vec<u32>,
    },
    /// Text subtitle.
    Text(String),
    /// ASS/SSA formatted subtitle.
    Ass(String),
}

/// A subtitle event with timing and content.
#[derive(Debug, Clone)]
pub struct Subtitle {
    pub start_display_time: u32,
    pub end_display_time: u32,
    pub pts: i64,
    pub rects: Vec<SubtitleRect>,
}

/// Subtitle decoder trait.
pub trait SubtitleDecoder: Send {
    fn decode(&mut self, packet: &Packet) -> Result<Option<Subtitle>>;
    fn flush(&mut self);
}

/// Subtitle encoder trait.
pub trait SubtitleEncoder: Send {
    fn encode(&mut self, subtitle: &Subtitle) -> Result<Packet>;
}
