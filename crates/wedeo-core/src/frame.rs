use bitflags::bitflags;

use crate::buffer::Buffer;
use crate::channel_layout::ChannelLayout;
use crate::chroma_location::ChromaLocation;
use crate::color_primaries::ColorPrimaries;
use crate::color_trc::ColorTransferCharacteristic;
use crate::frame_side_data::{FrameSideData, FrameSideDataType};
use crate::metadata::Metadata;
use crate::pixel_format::PixelFormat;
use crate::rational::Rational;
use crate::sample_format::SampleFormat;
use crate::timestamp::NOPTS_VALUE;

/// Maximum number of data planes in a video frame.
pub const VIDEO_MAX_PLANES: usize = 4;

/// Maximum number of data planes in an audio frame.
pub const AUDIO_MAX_PLANES: usize = 8;

/// Picture type, matching FFmpeg's AVPictureType.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PictureType {
    None = 0,
    I = 1,
    P = 2,
    B = 3,
    S = 4,
    Si = 5,
    Sp = 6,
    Bi = 7,
}

/// Color range, matching FFmpeg's AVColorRange.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ColorRange {
    Unspecified = 0,
    /// The normal 219*2^(n-8) "MPEG" YUV ranges.
    Mpeg = 1,
    /// The normal 2^n-1 "JPEG" YUV ranges.
    Jpeg = 2,
}

/// Color space, matching FFmpeg's AVColorSpace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ColorSpace {
    Rgb = 0,
    Bt709 = 1,
    Unspecified = 2,
    Fcc = 4,
    Bt470bg = 5,
    Smpte170m = 6,
    Smpte240m = 7,
    Ycgco = 8,
    Bt2020Ncl = 9,
    Bt2020Cl = 10,
}

bitflags! {
    /// Frame flags, matching FFmpeg's AV_FRAME_FLAG_*.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FrameFlags: u32 {
        const CORRUPT    = 1 << 0;
        const KEY        = 1 << 1;
        const DISCARD    = 1 << 2;
        const INTERLACED = 1 << 3;
        const TOP_FIRST  = 1 << 4;
    }
}

/// A plane of frame data (pointer into a Buffer).
#[derive(Debug, Clone)]
pub struct FramePlane {
    pub buffer: Buffer,
    /// Offset into the buffer where this plane's data starts.
    pub offset: usize,
    /// Stride (bytes per row for video, total bytes for audio plane).
    pub linesize: usize,
}

/// Video-specific frame data.
#[derive(Debug, Clone)]
pub struct VideoFrameData {
    pub planes: Vec<FramePlane>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub picture_type: PictureType,
    pub color_range: ColorRange,
    pub color_space: ColorSpace,
    pub color_primaries: ColorPrimaries,
    pub color_trc: ColorTransferCharacteristic,
    pub chroma_location: ChromaLocation,
    pub sample_aspect_ratio: Rational,
    /// Crop rectangle — number of pixels to discard from each edge.
    pub crop_top: u32,
    pub crop_bottom: u32,
    pub crop_left: u32,
    pub crop_right: u32,
}

/// Audio-specific frame data.
#[derive(Debug, Clone)]
pub struct AudioFrameData {
    pub planes: Vec<FramePlane>,
    pub nb_samples: u32,
    pub format: SampleFormat,
    pub sample_rate: u32,
    pub channel_layout: ChannelLayout,
}

/// Frame data — either video or audio.
/// Using an enum prevents accessing width on audio frames (compile-time safety).
#[derive(Debug, Clone)]
pub enum FrameData {
    Video(VideoFrameData),
    Audio(AudioFrameData),
}

/// Decoded frame, matching FFmpeg's AVFrame concept.
#[derive(Debug, Clone)]
pub struct Frame {
    pub data: FrameData,
    /// Presentation timestamp in time_base units.
    pub pts: i64,
    /// Decompression timestamp, copied from the packet that triggered output.
    pub pkt_dts: i64,
    /// Best-effort timestamp estimated by the framework.
    pub best_effort_timestamp: i64,
    /// Duration in time_base units.
    pub duration: i64,
    /// Time base for pts/duration.
    pub time_base: Rational,
    pub flags: FrameFlags,
    /// Extra flag to indicate a picture should be decoded but displayed twice
    /// (or according to repeat_pict). Matches FFmpeg's AVFrame.repeat_pict.
    pub repeat_pict: i32,
    /// Metadata key-value pairs.
    pub metadata: Metadata,
    /// Side data associated with this frame.
    pub side_data: Vec<FrameSideData>,
}

impl Frame {
    /// Create a new video frame.
    pub fn new_video(width: u32, height: u32, format: PixelFormat) -> Self {
        Self {
            data: FrameData::Video(VideoFrameData {
                planes: Vec::new(),
                width,
                height,
                format,
                picture_type: PictureType::None,
                color_range: ColorRange::Unspecified,
                color_space: ColorSpace::Unspecified,
                color_primaries: ColorPrimaries::Unspecified,
                color_trc: ColorTransferCharacteristic::Unspecified,
                chroma_location: ChromaLocation::Unspecified,
                sample_aspect_ratio: Rational::new(0, 1),
                crop_top: 0,
                crop_bottom: 0,
                crop_left: 0,
                crop_right: 0,
            }),
            pts: NOPTS_VALUE,
            pkt_dts: NOPTS_VALUE,
            best_effort_timestamp: NOPTS_VALUE,
            duration: 0,
            time_base: Rational::new(0, 1),
            flags: FrameFlags::empty(),
            repeat_pict: 0,
            metadata: Metadata::new(),
            side_data: Vec::new(),
        }
    }

    /// Create a new audio frame.
    pub fn new_audio(
        nb_samples: u32,
        format: SampleFormat,
        sample_rate: u32,
        channel_layout: ChannelLayout,
    ) -> Self {
        Self {
            data: FrameData::Audio(AudioFrameData {
                planes: Vec::new(),
                nb_samples,
                format,
                sample_rate,
                channel_layout,
            }),
            pts: NOPTS_VALUE,
            pkt_dts: NOPTS_VALUE,
            best_effort_timestamp: NOPTS_VALUE,
            duration: 0,
            time_base: Rational::new(0, 1),
            flags: FrameFlags::empty(),
            repeat_pict: 0,
            metadata: Metadata::new(),
            side_data: Vec::new(),
        }
    }

    /// Returns true if this is a video frame.
    pub fn is_video(&self) -> bool {
        matches!(self.data, FrameData::Video(_))
    }

    /// Returns true if this is an audio frame.
    pub fn is_audio(&self) -> bool {
        matches!(self.data, FrameData::Audio(_))
    }

    /// Get video data, if this is a video frame.
    pub fn video(&self) -> Option<&VideoFrameData> {
        match &self.data {
            FrameData::Video(v) => Some(v),
            _ => None,
        }
    }

    /// Get mutable video data, if this is a video frame.
    pub fn video_mut(&mut self) -> Option<&mut VideoFrameData> {
        match &mut self.data {
            FrameData::Video(v) => Some(v),
            _ => None,
        }
    }

    /// Get audio data, if this is an audio frame.
    pub fn audio(&self) -> Option<&AudioFrameData> {
        match &self.data {
            FrameData::Audio(a) => Some(a),
            _ => None,
        }
    }

    /// Get mutable audio data, if this is an audio frame.
    pub fn audio_mut(&mut self) -> Option<&mut AudioFrameData> {
        match &mut self.data {
            FrameData::Audio(a) => Some(a),
            _ => None,
        }
    }

    /// Get side data of a specific type.
    pub fn get_side_data(&self, data_type: FrameSideDataType) -> Option<&FrameSideData> {
        self.side_data.iter().find(|sd| sd.data_type == data_type)
    }

    /// Add side data to the frame.
    ///
    /// Does NOT replace existing data of the same type — multiple entries of
    /// the same type are allowed, matching FFmpeg's `av_frame_new_side_data`.
    /// Use `remove_side_data` first if you need unique-per-type behavior.
    pub fn add_side_data(&mut self, side_data: FrameSideData) {
        self.side_data.push(side_data);
    }

    /// Remove side data of a specific type.
    pub fn remove_side_data(&mut self, data_type: FrameSideDataType) {
        self.side_data.retain(|sd| sd.data_type != data_type);
    }
}
