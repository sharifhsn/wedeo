use std::fmt;

use bitflags::bitflags;

use crate::audio::{AudioBlock, AudioStreamParams};
use crate::buffer::{BorderedPlane, Buffer};
use crate::time::{Rational, Timestamp};
use crate::video::{PixelLayout, VideoFrame, VideoStreamParams};

/// High-level media domain of a track or codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaType {
    Video,
    Audio,
    Subtitle,
    Data,
    Attachment,
}

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => f.write_str("video"),
            Self::Audio => f.write_str("audio"),
            Self::Subtitle => f.write_str("subtitle"),
            Self::Data => f.write_str("data"),
            Self::Attachment => f.write_str("attachment"),
        }
    }
}

/// Strongly-typed track identifier within a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TrackId(pub u32);

/// Codec identifier covering video, audio, and subtitle codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodecId {
    None,
    // Video
    Rawvideo,
    Mpeg1video,
    Mpeg2video,
    Mpeg4,
    Mjpeg,
    H264,
    Hevc,
    Vp8,
    Vp9,
    Av1,
    Ffv1,
    Prores,
    // PCM Audio
    PcmS8,
    PcmU8,
    PcmS16le,
    PcmS16be,
    PcmU16le,
    PcmU16be,
    PcmS24le,
    PcmS24be,
    PcmU24le,
    PcmU24be,
    PcmS32le,
    PcmS32be,
    PcmU32le,
    PcmU32be,
    PcmF32le,
    PcmF32be,
    PcmF64le,
    PcmF64be,
    PcmMulaw,
    PcmAlaw,
    // ADPCM & Compressed Audio
    AdpcmImaWav,
    AdpcmMs,
    Mp1,
    Mp2,
    Mp3,
    Aac,
    Ac3,
    Vorbis,
    Flac,
    Alac,
    WavPack,
    Opus,
    // Subtitles
    SubText,
    SubSrt,
    SubWebvtt,
    SubAss,
    SubMovText,
    SubPgs,
    SubDvd,
    SubDvb,
}

impl CodecId {
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Rawvideo => "rawvideo",
            Self::Mpeg1video => "mpeg1video",
            Self::Mpeg2video => "mpeg2video",
            Self::Mpeg4 => "mpeg4",
            Self::Mjpeg => "mjpeg",
            Self::H264 => "h264",
            Self::Hevc => "hevc",
            Self::Vp8 => "vp8",
            Self::Vp9 => "vp9",
            Self::Av1 => "av1",
            Self::Ffv1 => "ffv1",
            Self::Prores => "prores",
            Self::PcmS8 => "pcm_s8",
            Self::PcmU8 => "pcm_u8",
            Self::PcmS16le => "pcm_s16le",
            Self::PcmS16be => "pcm_s16be",
            Self::PcmU16le => "pcm_u16le",
            Self::PcmU16be => "pcm_u16be",
            Self::PcmS24le => "pcm_s24le",
            Self::PcmS24be => "pcm_s24be",
            Self::PcmU24le => "pcm_u24le",
            Self::PcmU24be => "pcm_u24be",
            Self::PcmS32le => "pcm_s32le",
            Self::PcmS32be => "pcm_s32be",
            Self::PcmU32le => "pcm_u32le",
            Self::PcmU32be => "pcm_u32be",
            Self::PcmF32le => "pcm_f32le",
            Self::PcmF32be => "pcm_f32be",
            Self::PcmF64le => "pcm_f64le",
            Self::PcmF64be => "pcm_f64be",
            Self::PcmMulaw => "pcm_mulaw",
            Self::PcmAlaw => "pcm_alaw",
            Self::AdpcmImaWav => "adpcm_ima_wav",
            Self::AdpcmMs => "adpcm_ms",
            Self::Mp1 => "mp1",
            Self::Mp2 => "mp2",
            Self::Mp3 => "mp3",
            Self::Aac => "aac",
            Self::Ac3 => "ac3",
            Self::Vorbis => "vorbis",
            Self::Flac => "flac",
            Self::Alac => "alac",
            Self::WavPack => "wavpack",
            Self::Opus => "opus",
            Self::SubText => "text",
            Self::SubSrt => "srt",
            Self::SubWebvtt => "webvtt",
            Self::SubAss => "ass",
            Self::SubMovText => "mov_text",
            Self::SubPgs => "hdmv_pgs_subtitle",
            Self::SubDvd => "dvd_subtitle",
            Self::SubDvb => "dvb_subtitle",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        let id = match name {
            "none" => Self::None,
            "rawvideo" => Self::Rawvideo,
            "mpeg1video" => Self::Mpeg1video,
            "mpeg2video" => Self::Mpeg2video,
            "mpeg4" => Self::Mpeg4,
            "mjpeg" => Self::Mjpeg,
            "h264" | "avc" | "libx264" => Self::H264,
            "hevc" | "h265" | "libx265" => Self::Hevc,
            "vp8" | "libvpx" => Self::Vp8,
            "vp9" | "libvpx-vp9" => Self::Vp9,
            "av1" | "libaom-av1" | "librav1e" | "libsvtav1" => Self::Av1,
            "ffv1" => Self::Ffv1,
            "prores" => Self::Prores,
            "pcm_s8" => Self::PcmS8,
            "pcm_u8" => Self::PcmU8,
            "pcm_s16le" => Self::PcmS16le,
            "pcm_s16be" => Self::PcmS16be,
            "pcm_u16le" => Self::PcmU16le,
            "pcm_u16be" => Self::PcmU16be,
            "pcm_s24le" => Self::PcmS24le,
            "pcm_s24be" => Self::PcmS24be,
            "pcm_u24le" => Self::PcmU24le,
            "pcm_u24be" => Self::PcmU24be,
            "pcm_s32le" => Self::PcmS32le,
            "pcm_s32be" => Self::PcmS32be,
            "pcm_u32le" => Self::PcmU32le,
            "pcm_u32be" => Self::PcmU32be,
            "pcm_f32le" => Self::PcmF32le,
            "pcm_f32be" => Self::PcmF32be,
            "pcm_f64le" => Self::PcmF64le,
            "pcm_f64be" => Self::PcmF64be,
            "pcm_mulaw" => Self::PcmMulaw,
            "pcm_alaw" => Self::PcmAlaw,
            "adpcm_ima_wav" => Self::AdpcmImaWav,
            "adpcm_ms" => Self::AdpcmMs,
            "mp1" => Self::Mp1,
            "mp2" => Self::Mp2,
            "mp3" | "libmp3lame" => Self::Mp3,
            "aac" | "libfdk_aac" => Self::Aac,
            "ac3" => Self::Ac3,
            "vorbis" | "libvorbis" => Self::Vorbis,
            "flac" => Self::Flac,
            "alac" => Self::Alac,
            "wavpack" => Self::WavPack,
            "opus" | "libopus" => Self::Opus,
            "text" => Self::SubText,
            "srt" | "subrip" => Self::SubSrt,
            "webvtt" => Self::SubWebvtt,
            "ass" | "ssa" => Self::SubAss,
            "mov_text" => Self::SubMovText,
            "pgs" | "hdmv_pgs_subtitle" => Self::SubPgs,
            "dvd_subtitle" => Self::SubDvd,
            "dvb_subtitle" => Self::SubDvb,
            _ => return None,
        };
        Some(id)
    }

    pub const fn media_type(self) -> Option<MediaType> {
        match self {
            Self::None => None,
            Self::Rawvideo
            | Self::Mpeg1video
            | Self::Mpeg2video
            | Self::Mpeg4
            | Self::Mjpeg
            | Self::H264
            | Self::Hevc
            | Self::Vp8
            | Self::Vp9
            | Self::Av1
            | Self::Ffv1
            | Self::Prores => Some(MediaType::Video),
            Self::PcmS8
            | Self::PcmU8
            | Self::PcmS16le
            | Self::PcmS16be
            | Self::PcmU16le
            | Self::PcmU16be
            | Self::PcmS24le
            | Self::PcmS24be
            | Self::PcmU24le
            | Self::PcmU24be
            | Self::PcmS32le
            | Self::PcmS32be
            | Self::PcmU32le
            | Self::PcmU32be
            | Self::PcmF32le
            | Self::PcmF32be
            | Self::PcmF64le
            | Self::PcmF64be
            | Self::PcmMulaw
            | Self::PcmAlaw
            | Self::AdpcmImaWav
            | Self::AdpcmMs
            | Self::Mp1
            | Self::Mp2
            | Self::Mp3
            | Self::Aac
            | Self::Ac3
            | Self::Vorbis
            | Self::Flac
            | Self::Alac
            | Self::WavPack
            | Self::Opus => Some(MediaType::Audio),
            Self::SubText
            | Self::SubSrt
            | Self::SubWebvtt
            | Self::SubAss
            | Self::SubMovText
            | Self::SubPgs
            | Self::SubDvd
            | Self::SubDvb => Some(MediaType::Subtitle),
        }
    }
}

/// Ordered key-value metadata (`Vec<(String, String)>` for deterministic iteration)
/// with case-insensitive key lookup.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Metadata {
    entries: Vec<(String, String)>,
}

impl Metadata {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|(k, _)| k.eq_ignore_ascii_case(&key))
        {
            entry.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.entries
            .iter()
            .position(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|pos| self.entries.remove(pos).1)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// SMPTE ST 2086 mastering display color volume metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasteringDisplayMeta {
    /// Display primaries `[R, G, B]` as `(x, y)` rationals.
    pub display_primaries: [[Rational; 2]; 3],
    /// White point `(x, y)` rationals.
    pub white_point: [Rational; 2],
    pub min_luminance: Rational,
    pub max_luminance: Rational,
}

/// ReplayGain loudness metadata (gains in 1/100,000 dB, peaks in 1/100,000 scale).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReplayGain {
    pub track_gain: i32,
    pub track_peak: u32,
    pub album_gain: i32,
    pub album_peak: u32,
}

/// Strictly typed side data attached to packets and decoded media units.
#[derive(Debug, Clone)]
pub enum SideData {
    DisplayMatrix([i32; 9]),
    MasteringDisplay(MasteringDisplayMeta),
    ContentLightLevel { max_cll: u32, max_fall: u32 },
    Hdr10Plus(Buffer),
    DolbyVisionRpu(Buffer),
    IccProfile(Buffer),
    SampleTrim { skip_start: u32, skip_end: u32 },
    ReplayGain(ReplayGain),
    ClosedCaptionsA53(Buffer),
}

bitflags! {
    /// Compressed packet flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct PacketFlags: u32 {
        const KEY        = 1 << 0;
        const CORRUPT    = 1 << 1;
        const DISCARD    = 1 << 2;
        const DISPOSABLE = 1 << 3;
    }
}

/// Self-describing compressed packet with explicit `time_base` and VFR timestamps.
#[derive(Debug, Clone)]
pub struct Packet {
    pub data: Buffer,
    pub track_id: TrackId,
    pub pts: Timestamp,
    pub dts: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
    pub pos: Option<u64>,
    pub flags: PacketFlags,
    pub side_data: Vec<SideData>,
}

impl Packet {
    pub fn new(data: Buffer, track_id: TrackId, time_base: Rational) -> Self {
        Self {
            data,
            track_id,
            pts: Timestamp::NONE,
            dts: Timestamp::NONE,
            duration: 0,
            time_base,
            pos: None,
            flags: PacketFlags::empty(),
            side_data: Vec::new(),
        }
    }

    pub fn from_slice(data: &[u8]) -> Self {
        Self::new(Buffer::from_slice(data), TrackId(0), Rational::new(0, 1))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Subtitle encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubtitleFormat {
    PlainText,
    Srt,
    WebVtt,
    Ass,
    MovText,
    Pgs,
    Dvd,
    Dvb,
}

/// Positioned alpha/RGBA plane for bitmap subtitles (`PGS`/`DVD`) and overlays.
#[derive(Debug, Clone)]
pub struct PositionedPlane {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub layout: PixelLayout,
    pub plane: BorderedPlane,
}

/// Decoded subtitle event payload.
#[derive(Debug, Clone)]
pub enum SubtitlePayload {
    Text(String),
    Ass(String),
    WebVtt(String),
    Bitmap(Vec<PositionedPlane>),
}

/// Timed subtitle event.
#[derive(Debug, Clone)]
pub struct SubtitleEvent {
    pub pts: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
    pub payload: SubtitlePayload,
}

/// Subtitle track parameters.
#[derive(Debug, Clone)]
pub struct SubtitleStreamParams {
    pub format: SubtitleFormat,
    pub canvas_width: u32,
    pub canvas_height: u32,
    /// Optional ASS/SSA `[Script Info]` + `[V4+ Styles]` header.
    pub header_script: Vec<u8>,
}

/// Timed non-AV side stream payload (SCTE-35, KLV, telemetry, AI metadata).
#[derive(Debug, Clone)]
pub struct TimedData {
    pub schema_tag: String,
    pub payload: Buffer,
    pub pts: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
}

/// Timed data track parameters.
#[derive(Debug, Clone)]
pub struct DataStreamParams {
    pub schema_tag: String,
    pub extradata: Vec<u8>,
}

/// Strictly typed per-track parameters (replaces flat union-like `CodecParameters`).
#[derive(Debug, Clone)]
pub enum StreamParams {
    Video(VideoStreamParams),
    Audio(AudioStreamParams),
    Subtitle(SubtitleStreamParams),
    Data(DataStreamParams),
}

impl StreamParams {
    pub fn media_type(&self) -> MediaType {
        match self {
            Self::Video(_) => MediaType::Video,
            Self::Audio(_) => MediaType::Audio,
            Self::Subtitle(_) => MediaType::Subtitle,
            Self::Data(_) => MediaType::Data,
        }
    }

    pub fn video(&self) -> Option<&VideoStreamParams> {
        match self {
            Self::Video(v) => Some(v),
            _ => None,
        }
    }

    pub fn audio(&self) -> Option<&AudioStreamParams> {
        match self {
            Self::Audio(a) => Some(a),
            _ => None,
        }
    }
}

/// Unified decoded media unit flowing through decoders, encoders, and graphs.
#[derive(Debug, Clone)]
pub enum MediaUnit {
    Video(VideoFrame),
    Audio(AudioBlock),
    Subtitle(SubtitleEvent),
    TimedData(TimedData),
}

impl MediaUnit {
    pub fn pts(&self) -> Timestamp {
        match self {
            Self::Video(v) => v.pts,
            Self::Audio(a) => a.pts,
            Self::Subtitle(s) => s.pts,
            Self::TimedData(d) => d.pts,
        }
    }

    pub fn duration(&self) -> i64 {
        match self {
            Self::Video(v) => v.duration,
            Self::Audio(a) => a.duration,
            Self::Subtitle(s) => s.duration,
            Self::TimedData(d) => d.duration,
        }
    }

    pub fn time_base(&self) -> Rational {
        match self {
            Self::Video(v) => v.time_base,
            Self::Audio(a) => a.time_base,
            Self::Subtitle(s) => s.time_base,
            Self::TimedData(d) => d.time_base,
        }
    }

    pub fn as_video(&self) -> Option<&VideoFrame> {
        match self {
            Self::Video(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_audio(&self) -> Option<&AudioBlock> {
        match self {
            Self::Audio(a) => Some(a),
            _ => None,
        }
    }

    pub fn into_video(self) -> Option<VideoFrame> {
        match self {
            Self::Video(v) => Some(v),
            _ => None,
        }
    }

    pub fn into_audio(self) -> Option<AudioBlock> {
        match self {
            Self::Audio(a) => Some(a),
            _ => None,
        }
    }
}

/// Container edit-list segment (`MP4 elst`) mapping coded track time to presentation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditSegment {
    /// Start time in coded track media coordinates (`Timestamp::NONE` for empty dwell).
    pub media_time: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
    pub rate: Rational,
}

/// Container chapter entry.
#[derive(Debug, Clone)]
pub struct Chapter {
    pub id: i64,
    pub start: Timestamp,
    pub end: Timestamp,
    pub time_base: Rational,
    pub metadata: Metadata,
}

/// Container attachment (fonts, cover art, thumbnails).
#[derive(Debug, Clone)]
pub struct Attachment {
    pub name: String,
    pub mime_type: String,
    pub data: Buffer,
}

bitflags! {
    /// Track disposition flags preserved across `--preserve-all` pipelines.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct TrackDisposition: u32 {
        const DEFAULT          = 1 << 0;
        const FORCED           = 1 << 1;
        const HEARING_IMPAIRED = 1 << 2;
        const VISUAL_IMPAIRED  = 1 << 3;
        const COMMENTARY       = 1 << 4;
        const ATTACHED_PIC     = 1 << 5;
    }
}

/// Complete description of one logical track within a container.
#[derive(Debug, Clone)]
pub struct TrackDescriptor {
    pub id: TrackId,
    pub codec_id: CodecId,
    pub codec_tag: u32,
    pub time_base: Rational,
    pub duration: i64,
    pub nb_frames: Option<u64>,
    pub language: Option<String>,
    pub disposition: TrackDisposition,
    pub params: StreamParams,
    pub edit_list: Vec<EditSegment>,
    pub metadata: Metadata,
}

impl TrackDescriptor {
    pub fn new(id: TrackId, codec_id: CodecId, time_base: Rational, params: StreamParams) -> Self {
        Self {
            id,
            codec_id,
            codec_tag: 0,
            time_base,
            duration: 0,
            nb_frames: None,
            language: None,
            disposition: TrackDisposition::empty(),
            params,
            edit_list: Vec::new(),
            metadata: Metadata::new(),
        }
    }
}

/// Top-level container metadata, tracks, chapters, and attachments.
#[derive(Debug, Clone)]
pub struct ContainerHeader {
    pub format_name: &'static str,
    pub tracks: Vec<TrackDescriptor>,
    pub chapters: Vec<Chapter>,
    pub attachments: Vec<Attachment>,
    pub duration: Option<i64>,
    pub start_time: Timestamp,
    pub time_base: Rational,
    pub metadata: Metadata,
}

impl ContainerHeader {
    pub fn new(format_name: &'static str) -> Self {
        Self {
            format_name,
            tracks: Vec::new(),
            chapters: Vec::new(),
            attachments: Vec::new(),
            duration: None,
            start_time: Timestamp::NONE,
            time_base: crate::time::TIME_BASE_Q,
            metadata: Metadata::new(),
        }
    }

    pub fn track(&self, id: TrackId) -> Option<&TrackDescriptor> {
        self.tracks.iter().find(|t| t.id == id)
    }
}

/// Explicit seek request for `Demuxer::seek`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeekTarget {
    pub track: Option<TrackId>,
    pub timestamp: Timestamp,
    pub time_base: Rational,
    /// When `true`, seek to the closest keyframe at or before `timestamp`.
    pub backward: bool,
}
