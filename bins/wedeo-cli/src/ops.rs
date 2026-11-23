//! Typed operations backing all `wedeo` frontends (`CLI`, `wedeo ffmpeg`, `wedeo mcp`):
//!
//! 1. `InspectReq` -> `InspectReport`
//! 2. `ExplainReq` -> `ExecutionPlan`
//! 3. `TranscodeReq` -> `RunReport`
//! 4. `CutReq` -> `RunReport`
//! 5. `MuxReq` -> `RunReport`

use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use wedeo::audio_ops::AudioContract;
use wedeo::core::{
    BufByteSource, Rounding, StreamHealth as CoreStreamHealth, TrackDisposition, rescale_q,
    rescale_q_rnd,
};
use wedeo::formats::mkv::{EbmlElement, parse_ebml_tree};
use wedeo::formats::mp4::boxes::{Mp4BoxNode, inspect_boxes, matrix_rotation_degrees};
use wedeo::graph::{
    CompileOpts, CompiledGraph, ExecutionPlan, GraphBuilder, GraphItem, GraphNode, NodeId,
    PacketContract, PacketFraming, PortContract, TrimFilter,
};
use wedeo::syntax::{BitstreamUnit, PacketInspection, h264, hevc, inspect_packet};
use wedeo::video_ops::VideoContract;
use wedeo::{
    AudioStreamParams, BufferPool, ByteSink, ChannelLayout, CodecContext, CodecId, ContainerHeader,
    Decoder, Demuxer, EncodeHint, Encoder, Error, ErrorPolicy, FileSink, FileSource, HealthEvent,
    MediaType, MediaUnit, MemorySink, Packet, PacketFlags, PassMode, PixelLayout, Rational,
    RecvStatus, Registry, Result, SampleFormat, SendStatus, StreamHealth, StreamParams, Timestamp,
    TrackDescriptor, TrackId, VideoStreamParams,
};

use crate::json::JsonValue;

// ============================================================================
// Shared Helpers (Demuxing, Probing, Format/Codec Resolution)
// ============================================================================

const PROBE_BYTES: usize = 8192;

pub fn open_demuxer_path(
    reg: &Registry,
    path: &str,
    policy: ErrorPolicy,
) -> Result<Box<dyn Demuxer>> {
    let ext = Path::new(path).extension().and_then(|e| e.to_str());
    let mut file = FileSource::open(path)?;
    let mut head = Vec::with_capacity(PROBE_BYTES);
    let mut buf = [0u8; 4096];
    while head.len() < PROBE_BYTES {
        let want = (PROBE_BYTES - head.len()).min(buf.len());
        match file.read(&mut buf[..want]) {
            Ok(0) => break,
            Ok(n) => head.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::io("wedeo-cli/open", &e)),
        }
    }
    let factory = reg.probe_demuxer(&head, ext).ok_or_else(|| {
        Error::not_found(
            "wedeo-cli/open",
            format!("no registered demuxer matched '{path}'"),
        )
    })?;
    wedeo::core::ByteSource::seek(&mut file, SeekFrom::Start(0))
        .map_err(|e| Error::io("wedeo-cli/open", &e))?;
    factory.open(Box::new(file), policy)
}

pub fn resolve_muxer_name(
    reg: &Registry,
    output_path: &str,
    explicit_format: Option<&str>,
) -> Result<String> {
    if let Some(fmt) = explicit_format {
        let factory = reg.find_muxer_by_name(fmt).ok_or_else(|| {
            Error::not_found("wedeo-cli/muxer", format!("unknown output format '{fmt}'"))
        })?;
        return Ok(factory.descriptor().name.to_string());
    }
    let ext = Path::new(output_path)
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| {
            Error::invalid_argument(
                "wedeo-cli/muxer",
                format!("cannot infer output format from '{output_path}'; pass --format / -f"),
            )
        })?;
    let factory = reg.find_muxer_by_ext(ext).ok_or_else(|| {
        Error::not_found(
            "wedeo-cli/muxer",
            format!("no muxer registered for extension '.{ext}'"),
        )
    })?;
    Ok(factory.descriptor().name.to_string())
}

pub fn parse_pixel_layout(name: &str) -> Option<PixelLayout> {
    match name.to_ascii_lowercase().as_str() {
        "yuv420p" => Some(PixelLayout::YUV420P),
        "yuv420p10le" => Some(PixelLayout::YUV420P10LE),
        "yuv420p10be" => Some(PixelLayout::YUV420P10BE),
        "yuv420p12le" => Some(PixelLayout::YUV420P12LE),
        "yuv420p16le" => Some(PixelLayout::YUV420P16LE),
        "yuv422p" => Some(PixelLayout::YUV422P),
        "yuv422p10le" => Some(PixelLayout::YUV422P10LE),
        "yuv444p" => Some(PixelLayout::YUV444P),
        "yuv444p10le" => Some(PixelLayout::YUV444P10LE),
        "nv12" => Some(PixelLayout::NV12),
        "nv21" => Some(PixelLayout::NV21),
        "p010le" | "p010" => Some(PixelLayout::P010),
        "rgb24" => Some(PixelLayout::RGB24),
        "bgr24" => Some(PixelLayout::BGR24),
        "rgba" | "rgba8" => Some(PixelLayout::RGBA8),
        "bgra" | "bgra8" => Some(PixelLayout::BGRA8),
        "argb" | "argb8" => Some(PixelLayout::ARGB8),
        "abgr" | "abgr8" => Some(PixelLayout::ABGR8),
        "gbrp" => Some(PixelLayout::GBRP),
        "gbrp10le" => Some(PixelLayout::GBRP10LE),
        "gray" | "gray8" => Some(PixelLayout::GRAY8),
        "gray16le" => Some(PixelLayout::GRAY16LE),
        "yuyv422" => Some(PixelLayout::YUYV422),
        "uyvy422" => Some(PixelLayout::UYVY422),
        _ => None,
    }
}

pub fn parse_size(s: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = s.split(['x', 'X']).collect();
    if parts.len() != 2 {
        return Err(Error::invalid_argument(
            "wedeo-cli",
            format!("invalid size '{s}', expected WxH (e.g. 1280x720)"),
        ));
    }
    let w: u32 = parts[0]
        .parse()
        .map_err(|_| Error::invalid_argument("wedeo-cli", format!("invalid width in '{s}'")))?;
    let h: u32 = parts[1]
        .parse()
        .map_err(|_| Error::invalid_argument("wedeo-cli", format!("invalid height in '{s}'")))?;
    if w == 0 || h == 0 {
        return Err(Error::invalid_argument(
            "wedeo-cli",
            "video dimensions must be non-zero",
        ));
    }
    Ok((w, h))
}

pub fn parse_timestamp_sec(s: &str) -> Result<f64> {
    let s = s.trim();
    if s.is_empty() {
        return Err(Error::invalid_argument("wedeo-cli", "empty timestamp"));
    }
    if !s.contains(':') {
        return s
            .parse::<f64>()
            .map_err(|_| Error::invalid_argument("wedeo-cli", format!("invalid timestamp '{s}'")));
    }
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        2 => {
            let m: f64 = parts[0].parse().map_err(|_| {
                Error::invalid_argument("wedeo-cli", format!("invalid minutes in '{s}'"))
            })?;
            let sec: f64 = parts[1].parse().map_err(|_| {
                Error::invalid_argument("wedeo-cli", format!("invalid seconds in '{s}'"))
            })?;
            Ok(m * 60.0 + sec)
        }
        3 => {
            let h: f64 = parts[0].parse().map_err(|_| {
                Error::invalid_argument("wedeo-cli", format!("invalid hours in '{s}'"))
            })?;
            let m: f64 = parts[1].parse().map_err(|_| {
                Error::invalid_argument("wedeo-cli", format!("invalid minutes in '{s}'"))
            })?;
            let sec: f64 = parts[2].parse().map_err(|_| {
                Error::invalid_argument("wedeo-cli", format!("invalid seconds in '{s}'"))
            })?;
            Ok(h * 3600.0 + m * 60.0 + sec)
        }
        _ => Err(Error::invalid_argument(
            "wedeo-cli",
            format!("invalid timestamp '{s}'"),
        )),
    }
}

fn check_overwrite(output_path: &str, overwrite: bool) -> Result<()> {
    if !overwrite && output_path != "-" && Path::new(output_path).exists() {
        return Err(Error::invalid_argument(
            "wedeo-cli",
            format!("output file '{output_path}' already exists (pass -y / --overwrite)"),
        ));
    }
    Ok(())
}

fn format_duration_hms(secs: f64) -> String {
    let total_ms = (secs.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

// ============================================================================
// HealthSummary (shared across InspectReport & RunReport)
// ============================================================================

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HealthEventSummary {
    pub track_id: u32,
    pub pts: Option<i64>,
    pub byte_offset: Option<u64>,
    pub kind: String,
    pub detail: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HealthSummary {
    pub is_clean: bool,
    pub corrupt_packets: u64,
    pub concealed_frames: u64,
    pub concealed_blocks: u64,
    pub timestamp_glitches: u64,
    pub dropped_bytes: u64,
    pub clipped_audio_samples: u64,
    pub events: Vec<HealthEventSummary>,
}

impl HealthSummary {
    pub fn from_stream_health(h: &CoreStreamHealth) -> Self {
        Self {
            is_clean: h.is_clean(),
            corrupt_packets: h.corrupt_packets,
            concealed_frames: h.concealed_frames,
            concealed_blocks: h.concealed_blocks,
            timestamp_glitches: h.timestamp_glitches,
            dropped_bytes: h.dropped_bytes,
            clipped_audio_samples: h.clipped_audio_samples,
            events: h
                .events
                .iter()
                .map(HealthEventSummary::from_event)
                .collect(),
        }
    }

    pub fn to_json_value(&self) -> JsonValue {
        let events: Vec<JsonValue> = self
            .events
            .iter()
            .map(|e| {
                JsonValue::obj(vec![
                    ("track_id", e.track_id.into()),
                    ("pts", e.pts.into()),
                    ("byte_offset", e.byte_offset.into()),
                    ("kind", e.kind.clone().into()),
                    ("detail", e.detail.clone().into()),
                ])
            })
            .collect();
        JsonValue::obj(vec![
            ("is_clean", self.is_clean.into()),
            ("corrupt_packets", self.corrupt_packets.into()),
            ("concealed_frames", self.concealed_frames.into()),
            ("concealed_blocks", self.concealed_blocks.into()),
            ("timestamp_glitches", self.timestamp_glitches.into()),
            ("dropped_bytes", self.dropped_bytes.into()),
            ("clipped_audio_samples", self.clipped_audio_samples.into()),
            ("events", JsonValue::Arr(events)),
        ])
    }

    pub fn format_warning_line(&self) -> String {
        format!(
            "StreamHealth warning: corrupt_packets={}, concealed_frames={}, concealed_blocks={}, timestamp_glitches={}, dropped_bytes={}, clipped_audio_samples={}",
            self.corrupt_packets,
            self.concealed_frames,
            self.concealed_blocks,
            self.timestamp_glitches,
            self.dropped_bytes,
            self.clipped_audio_samples
        )
    }
}

impl HealthEventSummary {
    fn from_event(e: &HealthEvent) -> Self {
        Self {
            track_id: e.track.0,
            pts: e.pts.get(),
            byte_offset: e.byte_offset,
            kind: format!("{:?}", e.kind),
            detail: e.detail.to_string(),
        }
    }
}

// ============================================================================
// 1. InspectReq -> InspectReport
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InspectLevel {
    #[default]
    Container,
    Packets,
    Gop,
    Nal,
}

impl InspectLevel {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "container" => Ok(Self::Container),
            "packets" | "packet" => Ok(Self::Packets),
            "gop" | "gops" => Ok(Self::Gop),
            "nal" | "nals" | "obu" | "bitstream" => Ok(Self::Nal),
            other => Err(Error::invalid_argument(
                "inspect",
                format!("unknown inspect level '{other}'; expected container|packets|gop|nal"),
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Container => "container",
            Self::Packets => "packets",
            Self::Gop => "gop",
            Self::Nal => "nal",
        }
    }
}

#[derive(Debug, Clone)]
pub struct InspectReq {
    pub input: String,
    pub level: InspectLevel,
    pub verify: bool,
    pub max_packets: Option<usize>,
}

impl InspectReq {
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            level: InspectLevel::Container,
            verify: false,
            max_packets: None,
        }
    }

    pub fn from_json(v: &JsonValue) -> Result<Self> {
        let input = v
            .get("input")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| Error::invalid_argument("inspect", "missing string field 'input'"))?
            .to_string();
        let level = match v.get("level").and_then(JsonValue::as_str) {
            Some(s) => InspectLevel::parse(s)?,
            None => InspectLevel::Container,
        };
        let verify = v
            .get("verify")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        let max_packets = v
            .get("max_packets")
            .and_then(JsonValue::as_u64)
            .map(|n| n as usize);
        Ok(Self {
            input,
            level,
            verify,
            max_packets,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerNode {
    pub kind: String,
    pub offset: u64,
    pub size: u64,
    pub summary: Option<String>,
    pub children: Vec<ContainerNode>,
}

impl ContainerNode {
    fn from_mp4(b: &Mp4BoxNode) -> Self {
        Self {
            kind: b.type_name(),
            offset: b.offset,
            size: b.size,
            summary: b.summary.clone(),
            children: b.children.iter().map(Self::from_mp4).collect(),
        }
    }

    fn from_ebml(e: &EbmlElement) -> Self {
        let summary = if e.size == u64::MAX {
            Some(format!("id=0x{:X} size=unknown", e.id))
        } else {
            Some(format!("id=0x{:X}", e.id))
        };
        Self {
            kind: e.name.to_string(),
            offset: e.header_offset,
            size: e.size,
            summary,
            children: e.children.iter().map(Self::from_ebml).collect(),
        }
    }

    fn to_json_value(&self) -> JsonValue {
        JsonValue::obj(vec![
            ("kind", self.kind.clone().into()),
            ("offset", self.offset.into()),
            ("size", self.size.into()),
            ("summary", self.summary.clone().into()),
            (
                "children",
                JsonValue::Arr(self.children.iter().map(Self::to_json_value).collect()),
            ),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct TrackInspectSummary {
    pub id: u32,
    pub media_type: String,
    pub codec: String,
    pub time_base: String,
    pub duration: i64,
    pub nb_frames: Option<u64>,
    pub language: Option<String>,
    pub disposition: Vec<String>,
    pub metadata: Vec<(String, String)>,
    // Video
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub pixel_layout: Option<String>,
    pub color_primaries: Option<String>,
    pub color_transfer: Option<String>,
    pub color_matrix: Option<String>,
    pub color_range: Option<String>,
    pub chroma_location: Option<String>,
    pub sar: Option<String>,
    pub avg_frame_rate: Option<String>,
    pub rotation_deg: Option<i32>,
    // Audio
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub channel_layout: Option<String>,
    pub sample_format: Option<String>,
    pub frame_size: Option<u32>,
    // Subtitle / Data
    pub subtitle_format: Option<String>,
    pub schema_tag: Option<String>,
    pub bit_rate: Option<i64>,
    pub extradata_bytes: usize,
}

impl TrackInspectSummary {
    fn from_descriptor(t: &TrackDescriptor) -> Self {
        let mut disp = Vec::new();
        if t.disposition.contains(TrackDisposition::DEFAULT) {
            disp.push("default".to_string());
        }
        if t.disposition.contains(TrackDisposition::FORCED) {
            disp.push("forced".to_string());
        }
        if t.disposition.contains(TrackDisposition::HEARING_IMPAIRED) {
            disp.push("hearing_impaired".to_string());
        }
        if t.disposition.contains(TrackDisposition::VISUAL_IMPAIRED) {
            disp.push("visual_impaired".to_string());
        }
        if t.disposition.contains(TrackDisposition::COMMENTARY) {
            disp.push("commentary".to_string());
        }
        if t.disposition.contains(TrackDisposition::ATTACHED_PIC) {
            disp.push("attached_pic".to_string());
        }

        let mut summary = Self {
            id: t.id.0,
            media_type: t.params.media_type().to_string(),
            codec: t.codec_id.name().to_string(),
            time_base: t.time_base.to_string(),
            duration: t.duration,
            nb_frames: t.nb_frames,
            language: t.language.clone(),
            disposition: disp,
            metadata: t
                .metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            width: None,
            height: None,
            pixel_layout: None,
            color_primaries: None,
            color_transfer: None,
            color_matrix: None,
            color_range: None,
            chroma_location: None,
            sar: None,
            avg_frame_rate: None,
            rotation_deg: None,
            sample_rate: None,
            channels: None,
            channel_layout: None,
            sample_format: None,
            frame_size: None,
            subtitle_format: None,
            schema_tag: None,
            bit_rate: None,
            extradata_bytes: 0,
        };

        match &t.params {
            StreamParams::Video(v) => {
                summary.width = Some(v.width);
                summary.height = Some(v.height);
                summary.pixel_layout = Some(v.layout.name().to_string());
                summary.color_primaries = Some(v.color.primaries.name().to_string());
                summary.color_transfer = Some(v.color.transfer.name().to_string());
                summary.color_matrix = Some(v.color.matrix.name().to_string());
                summary.color_range = Some(format!("{:?}", v.color.range).to_ascii_lowercase());
                summary.chroma_location =
                    Some(format!("{:?}", v.color.chroma_location).to_ascii_lowercase());
                summary.sar = Some(v.sar.to_string());
                if v.avg_frame_rate.is_valid() {
                    summary.avg_frame_rate = Some(v.avg_frame_rate.to_string());
                }
                if let Some(m) = v.display_matrix {
                    summary.rotation_deg = Some(matrix_rotation_degrees(&m));
                }
                if v.bit_rate > 0 {
                    summary.bit_rate = Some(v.bit_rate);
                }
                summary.extradata_bytes = v.extradata.len();
            }
            StreamParams::Audio(a) => {
                summary.sample_rate = Some(a.sample_rate);
                summary.channels = Some(a.channel_layout.nb_channels);
                summary.channel_layout = Some(a.channel_layout.to_string());
                summary.sample_format = Some(a.format.name().to_string());
                if a.frame_size > 0 {
                    summary.frame_size = Some(a.frame_size);
                }
                if a.bit_rate > 0 {
                    summary.bit_rate = Some(a.bit_rate);
                }
                summary.extradata_bytes = a.extradata.len();
            }
            StreamParams::Subtitle(s) => {
                summary.subtitle_format = Some(format!("{:?}", s.format));
                summary.width = Some(s.canvas_width);
                summary.height = Some(s.canvas_height);
                summary.extradata_bytes = s.header_script.len();
            }
            StreamParams::Data(d) => {
                summary.schema_tag = Some(d.schema_tag.clone());
                summary.extradata_bytes = d.extradata.len();
            }
        }
        summary
    }

    fn to_json_value(&self) -> JsonValue {
        let meta: Vec<JsonValue> = self
            .metadata
            .iter()
            .map(|(k, v)| {
                JsonValue::obj(vec![("key", k.clone().into()), ("value", v.clone().into())])
            })
            .collect();
        JsonValue::obj(vec![
            ("id", self.id.into()),
            ("media_type", self.media_type.clone().into()),
            ("codec", self.codec.clone().into()),
            ("time_base", self.time_base.clone().into()),
            ("duration", self.duration.into()),
            ("nb_frames", self.nb_frames.into()),
            ("language", self.language.clone().into()),
            ("disposition", self.disposition.clone().into()),
            ("width", self.width.into()),
            ("height", self.height.into()),
            ("pixel_layout", self.pixel_layout.clone().into()),
            ("color_primaries", self.color_primaries.clone().into()),
            ("color_transfer", self.color_transfer.clone().into()),
            ("color_matrix", self.color_matrix.clone().into()),
            ("color_range", self.color_range.clone().into()),
            ("chroma_location", self.chroma_location.clone().into()),
            ("sar", self.sar.clone().into()),
            ("avg_frame_rate", self.avg_frame_rate.clone().into()),
            ("rotation_deg", self.rotation_deg.into()),
            ("sample_rate", self.sample_rate.into()),
            ("channels", self.channels.into()),
            ("channel_layout", self.channel_layout.clone().into()),
            ("sample_format", self.sample_format.clone().into()),
            ("frame_size", self.frame_size.into()),
            ("subtitle_format", self.subtitle_format.clone().into()),
            ("schema_tag", self.schema_tag.clone().into()),
            ("bit_rate", self.bit_rate.into()),
            ("extradata_bytes", self.extradata_bytes.into()),
            ("metadata", JsonValue::Arr(meta)),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct ChapterSummary {
    pub id: i64,
    pub start_sec: Option<f64>,
    pub end_sec: Option<f64>,
    pub title: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AttachmentSummary {
    pub name: String,
    pub mime_type: String,
    pub size_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct NalUnitSummary {
    pub offset: usize,
    pub size: usize,
    pub type_code: u8,
    pub type_name: String,
    pub is_key: bool,
}

#[derive(Debug, Clone)]
pub struct PacketSummary {
    pub index: usize,
    pub track_id: u32,
    pub pts: Option<i64>,
    pub dts: Option<i64>,
    pub duration: i64,
    pub time_base: String,
    pub size: usize,
    pub pos: Option<u64>,
    pub is_key: bool,
    pub units: Vec<NalUnitSummary>,
}

#[derive(Debug, Clone)]
pub struct GopSummary {
    pub track_id: u32,
    pub gop_index: usize,
    pub keyframe_pts: Option<i64>,
    pub end_pts: Option<i64>,
    pub packet_count: usize,
    pub total_bytes: u64,
    pub time_base: String,
}

#[derive(Debug, Clone)]
pub struct InspectReport {
    pub input: String,
    pub format_name: String,
    pub level: InspectLevel,
    pub verified: bool,
    pub decoded_units: u64,
    pub duration_sec: Option<f64>,
    pub time_base: String,
    pub metadata: Vec<(String, String)>,
    pub tracks: Vec<TrackInspectSummary>,
    pub chapters: Vec<ChapterSummary>,
    pub attachments: Vec<AttachmentSummary>,
    pub container_ast: Vec<ContainerNode>,
    pub packets: Vec<PacketSummary>,
    pub gops: Vec<GopSummary>,
    pub health: HealthSummary,
}

impl InspectReport {
    pub fn format_human(&self) -> String {
        let mut out = String::with_capacity(1024);
        let _ = writeln!(out, "Input: {} ({})", self.input, self.format_name);
        if let Some(dur) = self.duration_sec {
            let _ = writeln!(
                out,
                "  Duration: {} ({:.3}s, time_base={})",
                format_duration_hms(dur),
                dur,
                self.time_base
            );
        }
        if !self.metadata.is_empty() {
            out.push_str("  Metadata:\n");
            for (k, v) in &self.metadata {
                let _ = writeln!(out, "    {k}: {v}");
            }
        }

        out.push_str("  Tracks:\n");
        for t in &self.tracks {
            match t.media_type.as_str() {
                "video" => {
                    let w = t.width.unwrap_or(0);
                    let h = t.height.unwrap_or(0);
                    let pix = t.pixel_layout.as_deref().unwrap_or("unknown");
                    let _ = write!(
                        out,
                        "    #{} [video] {} {}x{} {} tb={}",
                        t.id, t.codec, w, h, pix, t.time_base
                    );
                    if let (Some(p), Some(tr), Some(m), Some(r)) = (
                        &t.color_primaries,
                        &t.color_transfer,
                        &t.color_matrix,
                        &t.color_range,
                    ) {
                        let _ = write!(out, " color={p}/{tr}/{m}({r})");
                    }
                    if let Some(rot) = t.rotation_deg
                        && rot != 0
                    {
                        let _ = write!(out, " rot={rot}deg");
                    }
                    if let Some(lang) = &t.language {
                        let _ = write!(out, " lang={lang}");
                    }
                    out.push('\n');
                }
                "audio" => {
                    let sr = t.sample_rate.unwrap_or(0);
                    let ch = t.channel_layout.as_deref().unwrap_or("?");
                    let sf = t.sample_format.as_deref().unwrap_or("?");
                    let _ = write!(
                        out,
                        "    #{} [audio] {} {}Hz {} {} tb={}",
                        t.id, t.codec, sr, ch, sf, t.time_base
                    );
                    if let Some(lang) = &t.language {
                        let _ = write!(out, " lang={lang}");
                    }
                    out.push('\n');
                }
                other => {
                    let _ = writeln!(
                        out,
                        "    #{} [{other}] {} tb={}",
                        t.id, t.codec, t.time_base
                    );
                }
            }
        }

        if !self.chapters.is_empty() {
            let _ = writeln!(out, "  Chapters ({}):", self.chapters.len());
            for ch in &self.chapters {
                let title = ch.title.as_deref().unwrap_or("");
                let s = ch.start_sec.unwrap_or(0.0);
                let e = ch.end_sec.unwrap_or(0.0);
                let _ = writeln!(out, "    #{} [{s:.3}s..{e:.3}s] {title}", ch.id);
            }
        }

        if !self.attachments.is_empty() {
            let _ = writeln!(out, "  Attachments ({}):", self.attachments.len());
            for att in &self.attachments {
                let _ = writeln!(
                    out,
                    "    {} ({}, {} bytes)",
                    att.name, att.mime_type, att.size_bytes
                );
            }
        }

        if !self.container_ast.is_empty() {
            out.push_str("  Container AST:\n");
            for node in &self.container_ast {
                fmt_container_node(node, 2, &mut out);
            }
        }

        if !self.gops.is_empty() {
            let _ = writeln!(out, "  GOPs ({}):", self.gops.len());
            let _ = writeln!(
                out,
                "    {:<6} {:<6} {:<12} {:<12} {:<8} {:<10}",
                "GOP", "Track", "KeyPTS", "EndPTS", "Pkts", "Bytes"
            );
            for g in &self.gops {
                let kpts = g
                    .keyframe_pts
                    .map_or_else(|| "NOPTS".to_string(), |v| v.to_string());
                let epts = g
                    .end_pts
                    .map_or_else(|| "NOPTS".to_string(), |v| v.to_string());
                let _ = writeln!(
                    out,
                    "    {:<6} {:<6} {:<12} {:<12} {:<8} {:<10}",
                    g.gop_index, g.track_id, kpts, epts, g.packet_count, g.total_bytes
                );
            }
        }

        if !self.packets.is_empty() {
            let _ = writeln!(out, "  Packets ({}):", self.packets.len());
            let _ = writeln!(
                out,
                "    {:<6} {:<5} {:<12} {:<12} {:<8} {:<8} {:<4}",
                "Idx", "Trk", "PTS", "DTS", "Dur", "Size", "Key"
            );
            for p in &self.packets {
                let pts = p.pts.map_or_else(|| "NOPTS".to_string(), |v| v.to_string());
                let dts = p.dts.map_or_else(|| "NOPTS".to_string(), |v| v.to_string());
                let key = if p.is_key { "K" } else { "." };
                let _ = writeln!(
                    out,
                    "    {:<6} {:<5} {:<12} {:<12} {:<8} {:<8} {:<4}",
                    p.index, p.track_id, pts, dts, p.duration, p.size, key
                );
                for u in &p.units {
                    let _ = writeln!(
                        out,
                        "      + unit off={} size={} type={}({}) key={}",
                        u.offset, u.size, u.type_name, u.type_code, u.is_key
                    );
                }
            }
        }

        if self.verified || !self.health.is_clean {
            let status = if self.health.is_clean {
                "clean"
            } else {
                "issues_detected"
            };
            let _ = writeln!(
                out,
                "  StreamHealth ({status}, verified={}, decoded_units={}): corrupt_packets={} concealed_frames={} concealed_blocks={} timestamp_glitches={} dropped_bytes={} clipped_audio={}",
                self.verified,
                self.decoded_units,
                self.health.corrupt_packets,
                self.health.concealed_frames,
                self.health.concealed_blocks,
                self.health.timestamp_glitches,
                self.health.dropped_bytes,
                self.health.clipped_audio_samples
            );
            for ev in &self.health.events {
                let _ = writeln!(
                    out,
                    "    ! trk={} pts={:?} off={:?} kind={} detail={}",
                    ev.track_id, ev.pts, ev.byte_offset, ev.kind, ev.detail
                );
            }
        }

        out
    }

    pub fn to_json(&self) -> String {
        let meta: Vec<JsonValue> = self
            .metadata
            .iter()
            .map(|(k, v)| {
                JsonValue::obj(vec![("key", k.clone().into()), ("value", v.clone().into())])
            })
            .collect();
        let tracks: Vec<JsonValue> = self
            .tracks
            .iter()
            .map(TrackInspectSummary::to_json_value)
            .collect();
        let chapters: Vec<JsonValue> = self
            .chapters
            .iter()
            .map(|c| {
                JsonValue::obj(vec![
                    ("id", c.id.into()),
                    ("start_sec", c.start_sec.into()),
                    ("end_sec", c.end_sec.into()),
                    ("title", c.title.clone().into()),
                ])
            })
            .collect();
        let attachments: Vec<JsonValue> = self
            .attachments
            .iter()
            .map(|a| {
                JsonValue::obj(vec![
                    ("name", a.name.clone().into()),
                    ("mime_type", a.mime_type.clone().into()),
                    ("size_bytes", a.size_bytes.into()),
                ])
            })
            .collect();
        let ast: Vec<JsonValue> = self
            .container_ast
            .iter()
            .map(ContainerNode::to_json_value)
            .collect();
        let gops: Vec<JsonValue> = self
            .gops
            .iter()
            .map(|g| {
                JsonValue::obj(vec![
                    ("track_id", g.track_id.into()),
                    ("gop_index", g.gop_index.into()),
                    ("keyframe_pts", g.keyframe_pts.into()),
                    ("end_pts", g.end_pts.into()),
                    ("packet_count", g.packet_count.into()),
                    ("total_bytes", g.total_bytes.into()),
                    ("time_base", g.time_base.clone().into()),
                ])
            })
            .collect();
        let packets: Vec<JsonValue> = self
            .packets
            .iter()
            .map(|p| {
                let units: Vec<JsonValue> = p
                    .units
                    .iter()
                    .map(|u| {
                        JsonValue::obj(vec![
                            ("offset", u.offset.into()),
                            ("size", u.size.into()),
                            ("type_code", u32::from(u.type_code).into()),
                            ("type_name", u.type_name.clone().into()),
                            ("is_key", u.is_key.into()),
                        ])
                    })
                    .collect();
                JsonValue::obj(vec![
                    ("index", p.index.into()),
                    ("track_id", p.track_id.into()),
                    ("pts", p.pts.into()),
                    ("dts", p.dts.into()),
                    ("duration", p.duration.into()),
                    ("time_base", p.time_base.clone().into()),
                    ("size", p.size.into()),
                    ("pos", p.pos.into()),
                    ("is_key", p.is_key.into()),
                    ("units", JsonValue::Arr(units)),
                ])
            })
            .collect();

        JsonValue::obj(vec![
            ("input", self.input.clone().into()),
            ("format_name", self.format_name.clone().into()),
            ("level", self.level.as_str().into()),
            ("verified", self.verified.into()),
            ("decoded_units", self.decoded_units.into()),
            ("duration_sec", self.duration_sec.into()),
            ("time_base", self.time_base.clone().into()),
            ("metadata", JsonValue::Arr(meta)),
            ("tracks", JsonValue::Arr(tracks)),
            ("chapters", JsonValue::Arr(chapters)),
            ("attachments", JsonValue::Arr(attachments)),
            ("container_ast", JsonValue::Arr(ast)),
            ("gops", JsonValue::Arr(gops)),
            ("packets", JsonValue::Arr(packets)),
            ("health", self.health.to_json_value()),
        ])
        .to_pretty()
    }
}

fn fmt_container_node(node: &ContainerNode, indent_level: usize, out: &mut String) {
    let pad = "  ".repeat(indent_level);
    let sum = node
        .summary
        .as_deref()
        .map(|s| format!(" ({s})"))
        .unwrap_or_default();
    let _ = writeln!(
        out,
        "{pad}[{}] offset={} size={}{sum}",
        node.kind, node.offset, node.size
    );
    for child in &node.children {
        fmt_container_node(child, indent_level + 1, out);
    }
}

pub fn run_inspect(reg: &Registry, req: &InspectReq) -> Result<InspectReport> {
    let mut demuxer = open_demuxer_path(reg, &req.input, ErrorPolicy::Resilient)?;
    let hdr = demuxer.header().clone();

    let duration_sec = hdr.duration.and_then(|d| {
        if hdr.time_base.is_valid() {
            Some(d as f64 * hdr.time_base.to_f64())
        } else {
            None
        }
    });

    let tracks: Vec<TrackInspectSummary> = hdr
        .tracks
        .iter()
        .map(TrackInspectSummary::from_descriptor)
        .collect();
    let chapters: Vec<ChapterSummary> = hdr
        .chapters
        .iter()
        .map(|c| ChapterSummary {
            id: c.id,
            start_sec: c.start.to_seconds(c.time_base),
            end_sec: c.end.to_seconds(c.time_base),
            title: c.metadata.get("title").map(str::to_string),
        })
        .collect();
    let attachments: Vec<AttachmentSummary> = hdr
        .attachments
        .iter()
        .map(|a| AttachmentSummary {
            name: a.name.clone(),
            mime_type: a.mime_type.clone(),
            size_bytes: a.data.len(),
        })
        .collect();

    let container_ast = inspect_container_ast(&req.input, hdr.format_name)?;

    let need_packet_loop = req.verify || req.level != InspectLevel::Container;
    let mut packets = Vec::new();
    let mut gops = Vec::new();
    let mut decoded_units = 0u64;
    let mut agg_health = StreamHealth::new();

    if need_packet_loop {
        let mut decoders: BTreeMap<u32, Box<dyn Decoder>> = BTreeMap::new();
        if req.verify {
            let pool = BufferPool::new();
            for t in &hdr.tracks {
                if let Some(factory) = reg.find_decoder(t.codec_id) {
                    let ctx = CodecContext::new(Arc::clone(&pool));
                    if let Ok(dec) = factory.open(t, ctx) {
                        decoders.insert(t.id.0, dec);
                    }
                }
            }
        }

        // Per-track GOP accumulator
        struct GopAcc {
            gop_index: usize,
            keyframe_pts: Option<i64>,
            end_pts: Option<i64>,
            packet_count: usize,
            total_bytes: u64,
            time_base: String,
        }
        let mut active_gop: BTreeMap<u32, GopAcc> = BTreeMap::new();
        let mut gop_counts: BTreeMap<u32, usize> = BTreeMap::new();

        let mut pkt_idx = 0usize;
        let max_pkts = req.max_packets.unwrap_or(usize::MAX);

        loop {
            match demuxer.read_packet()? {
                RecvStatus::Ready(pkt) => {
                    let track_desc = hdr.track(pkt.track_id);

                    // Collect packet / NAL summaries
                    if matches!(req.level, InspectLevel::Packets | InspectLevel::Nal)
                        && packets.len() < max_pkts
                    {
                        let units = if req.level == InspectLevel::Nal {
                            inspect_packet_nals(track_desc, &pkt)
                        } else {
                            Vec::new()
                        };
                        packets.push(PacketSummary {
                            index: pkt_idx,
                            track_id: pkt.track_id.0,
                            pts: pkt.pts.get(),
                            dts: pkt.dts.get(),
                            duration: pkt.duration,
                            time_base: pkt.time_base.to_string(),
                            size: pkt.len(),
                            pos: pkt.pos,
                            is_key: pkt.flags.contains(PacketFlags::KEY),
                            units,
                        });
                    }

                    // Collect GOP summary
                    if req.level == InspectLevel::Gop {
                        let is_key = pkt.flags.contains(PacketFlags::KEY);
                        let pkt_end = pkt.pts.get().map(|p| p.saturating_add(pkt.duration.max(0)));
                        if is_key {
                            if let Some(prev) = active_gop.remove(&pkt.track_id.0) {
                                gops.push(GopSummary {
                                    track_id: pkt.track_id.0,
                                    gop_index: prev.gop_index,
                                    keyframe_pts: prev.keyframe_pts,
                                    end_pts: prev.end_pts,
                                    packet_count: prev.packet_count,
                                    total_bytes: prev.total_bytes,
                                    time_base: prev.time_base,
                                });
                            }
                            let gidx = gop_counts.entry(pkt.track_id.0).or_insert(0);
                            let idx = *gidx;
                            *gidx += 1;
                            active_gop.insert(
                                pkt.track_id.0,
                                GopAcc {
                                    gop_index: idx,
                                    keyframe_pts: pkt.pts.get(),
                                    end_pts: pkt_end,
                                    packet_count: 1,
                                    total_bytes: pkt.len() as u64,
                                    time_base: pkt.time_base.to_string(),
                                },
                            );
                        } else if let Some(cur) = active_gop.get_mut(&pkt.track_id.0) {
                            cur.packet_count += 1;
                            cur.total_bytes += pkt.len() as u64;
                            if let Some(e) = pkt_end {
                                cur.end_pts = Some(cur.end_pts.map_or(e, |prev| prev.max(e)));
                            }
                        } else {
                            let gidx = gop_counts.entry(pkt.track_id.0).or_insert(0);
                            let idx = *gidx;
                            *gidx += 1;
                            active_gop.insert(
                                pkt.track_id.0,
                                GopAcc {
                                    gop_index: idx,
                                    keyframe_pts: pkt.pts.get(),
                                    end_pts: pkt_end,
                                    packet_count: 1,
                                    total_bytes: pkt.len() as u64,
                                    time_base: pkt.time_base.to_string(),
                                },
                            );
                        }
                    }

                    // Decode if --verify
                    if req.verify
                        && let Some(dec) = decoders.get_mut(&pkt.track_id.0)
                    {
                        if dec.send_packet(Some(&pkt))? == SendStatus::Full {
                            decoded_units += drain_decoder_count(dec.as_mut())?;
                            let _ = dec.send_packet(Some(&pkt))?;
                        }
                        decoded_units += drain_decoder_count(dec.as_mut())?;
                    }

                    pkt_idx += 1;
                }
                RecvStatus::NeedMore => continue,
                RecvStatus::Eof => break,
            }
        }

        for (tid, acc) in active_gop {
            gops.push(GopSummary {
                track_id: tid,
                gop_index: acc.gop_index,
                keyframe_pts: acc.keyframe_pts,
                end_pts: acc.end_pts,
                packet_count: acc.packet_count,
                total_bytes: acc.total_bytes,
                time_base: acc.time_base,
            });
        }

        if req.verify {
            for dec in decoders.values_mut() {
                let _ = dec.send_packet(None)?;
                decoded_units += drain_decoder_count(dec.as_mut())?;
                agg_health.merge(dec.health());
            }
        }
    }

    agg_health.merge(demuxer.health());

    Ok(InspectReport {
        input: req.input.clone(),
        format_name: hdr.format_name.to_string(),
        level: req.level,
        verified: req.verify,
        decoded_units,
        duration_sec,
        time_base: hdr.time_base.to_string(),
        metadata: hdr
            .metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        tracks,
        chapters,
        attachments,
        container_ast,
        packets,
        gops,
        health: HealthSummary::from_stream_health(&agg_health),
    })
}

fn drain_decoder_count(dec: &mut dyn Decoder) -> Result<u64> {
    let mut count = 0u64;
    while let RecvStatus::Ready(_) = dec.receive()? {
        count += 1;
    }
    Ok(count)
}

fn inspect_packet_nals(track: Option<&TrackDescriptor>, pkt: &Packet) -> Vec<NalUnitSummary> {
    let Some(track) = track else {
        return Vec::new();
    };
    let raw = pkt.data.data();
    let converted: Vec<u8>;
    let payload: &[u8] = match track.codec_id {
        CodecId::H264 => {
            if let StreamParams::Video(vp) = &track.params
                && vp.extradata.len() >= 5
                && vp.extradata[0] == 1
            {
                let len_size = (vp.extradata[4] & 0x03) + 1;
                if let Ok(annexb) = h264::avcc_to_annexb(raw, len_size) {
                    converted = annexb;
                    &converted
                } else {
                    raw
                }
            } else {
                raw
            }
        }
        CodecId::Hevc => {
            if let StreamParams::Video(vp) = &track.params
                && vp.extradata.len() >= 22
                && vp.extradata[0] == 1
            {
                let len_size = (vp.extradata[21] & 0x03) + 1;
                if let Ok(annexb) = hevc::hvcc_to_annexb(raw, len_size) {
                    converted = annexb;
                    &converted
                } else {
                    raw
                }
            } else {
                raw
            }
        }
        _ => raw,
    };

    let Ok(PacketInspection { units, .. }) = inspect_packet(track.codec_id, payload) else {
        return Vec::new();
    };
    units.into_iter().map(map_unit).collect()
}

fn map_unit(u: BitstreamUnit) -> NalUnitSummary {
    NalUnitSummary {
        offset: u.offset,
        size: u.size,
        type_code: u.type_code,
        type_name: u.type_name.to_string(),
        is_key: u.is_key,
    }
}

fn inspect_container_ast(path: &str, format_name: &str) -> Result<Vec<ContainerNode>> {
    match format_name {
        "mp4" | "mov" => {
            let src = FileSource::open(path)?;
            let mut io = BufByteSource::new(Box::new(src));
            let boxes = inspect_boxes(&mut io)?;
            Ok(boxes.iter().map(ContainerNode::from_mp4).collect())
        }
        "matroska" | "mkv" | "webm" => {
            let src = FileSource::open(path)?;
            let mut io = BufByteSource::new(Box::new(src));
            let elems = parse_ebml_tree(&mut io, 4)?;
            Ok(elems.iter().map(ContainerNode::from_ebml).collect())
        }
        "wav" => inspect_wav_chunks(path),
        "ivf" => inspect_ivf_header(path),
        _ => Ok(Vec::new()),
    }
}

fn inspect_wav_chunks(path: &str) -> Result<Vec<ContainerNode>> {
    let mut file = File::open(path).map_err(|e| Error::io("inspect/wav", &e))?;
    let file_len = file
        .metadata()
        .map_err(|e| Error::io("inspect/wav", &e))?
        .len();
    let mut hdr = [0u8; 12];
    if file.read_exact(&mut hdr).is_err() {
        return Ok(Vec::new());
    }
    let magic = &hdr[0..4];
    if !matches!(magic, b"RIFF" | b"RIFX" | b"RF64" | b"BW64") {
        return Ok(Vec::new());
    }
    let big_endian = magic == b"RIFX";
    let read_u32 = |b: [u8; 4]| -> u32 {
        if big_endian {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        }
    };
    let root_size = u64::from(read_u32([hdr[4], hdr[5], hdr[6], hdr[7]]));
    let form_type = String::from_utf8_lossy(&hdr[8..12]).to_string();
    let mut children = Vec::new();
    let mut pos = 12u64;
    let mut ds64_data_size: Option<u64> = None;

    while pos + 8 <= file_len {
        file.seek(SeekFrom::Start(pos))
            .map_err(|e| Error::io("inspect/wav", &e))?;
        let mut ch = [0u8; 8];
        if file.read_exact(&mut ch).is_err() {
            break;
        }
        let cid = [ch[0], ch[1], ch[2], ch[3]];
        let raw_sz = u64::from(read_u32([ch[4], ch[5], ch[6], ch[7]]));
        let size = if &cid == b"data" && raw_sz == 0xFFFF_FFFF {
            ds64_data_size.unwrap_or(raw_sz)
        } else {
            raw_sz
        };
        if &cid == b"ds64" && size >= 16 {
            let mut ds = [0u8; 16];
            if file.read_exact(&mut ds).is_ok() {
                ds64_data_size = Some(u64::from_le_bytes(ds[8..16].try_into().unwrap_or([0; 8])));
            }
        }
        let kind = String::from_utf8_lossy(&cid).to_string();
        children.push(ContainerNode {
            kind,
            offset: pos,
            size,
            summary: None,
            children: Vec::new(),
        });
        let padded = size.saturating_add(size & 1);
        pos = pos.saturating_add(8).saturating_add(padded);
    }

    Ok(vec![ContainerNode {
        kind: String::from_utf8_lossy(magic).to_string(),
        offset: 0,
        size: root_size.saturating_add(8),
        summary: Some(format!("form={form_type}")),
        children,
    }])
}

fn inspect_ivf_header(path: &str) -> Result<Vec<ContainerNode>> {
    let mut file = File::open(path).map_err(|e| Error::io("inspect/ivf", &e))?;
    let mut hdr = [0u8; 32];
    if file.read_exact(&mut hdr).is_err() || &hdr[0..4] != b"DKIF" {
        return Ok(Vec::new());
    }
    let fourcc = String::from_utf8_lossy(&hdr[8..12]).to_string();
    let w = u16::from_le_bytes([hdr[12], hdr[13]]);
    let h = u16::from_le_bytes([hdr[14], hdr[15]]);
    let num = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
    let den = u32::from_le_bytes([hdr[20], hdr[21], hdr[22], hdr[23]]);
    let frames = u32::from_le_bytes([hdr[24], hdr[25], hdr[26], hdr[27]]);
    Ok(vec![ContainerNode {
        kind: "DKIF".to_string(),
        offset: 0,
        size: 32,
        summary: Some(format!(
            "fourcc={fourcc} {w}x{h} rate={num}/{den} frames={frames}"
        )),
        children: Vec::new(),
    }])
}

// ============================================================================
// 2. ExplainReq & 3. TranscodeReq -> ExecutionPlan / RunReport
// ============================================================================

#[derive(Debug, Clone)]
pub struct TranscodeReq {
    pub input: String,
    pub output: String,
    pub format: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub video_bitrate: Option<i64>,
    pub audio_bitrate: Option<i64>,
    pub frames_video: Option<u64>,
    pub no_video: bool,
    pub no_audio: bool,
    pub size: Option<(u32, u32)>,
    pub pixel_layout: Option<PixelLayout>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub start_sec: Option<f64>,
    pub duration_sec: Option<f64>,
    pub strict: bool,
    pub preserve_all: bool,
    pub overwrite: bool,
    pub prefer_gpu: bool,
    pub codec_options: Vec<(String, String)>,
}

impl TranscodeReq {
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            format: None,
            video_codec: None,
            audio_codec: None,
            video_bitrate: None,
            audio_bitrate: None,
            frames_video: None,
            no_video: false,
            no_audio: false,
            size: None,
            pixel_layout: None,
            sample_rate: None,
            channels: None,
            start_sec: None,
            duration_sec: None,
            strict: false,
            preserve_all: true,
            overwrite: false,
            prefer_gpu: false,
            codec_options: Vec::new(),
        }
    }

    pub fn from_json(v: &JsonValue) -> Result<Self> {
        let input = v
            .get("input")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| Error::invalid_argument("transcode", "missing string field 'input'"))?
            .to_string();
        let output = v
            .get("output")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| Error::invalid_argument("transcode", "missing string field 'output'"))?
            .to_string();
        let mut req = Self::new(input, output);
        req.format = v
            .get("format")
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        req.video_codec = v
            .get("video_codec")
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        req.audio_codec = v
            .get("audio_codec")
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        req.video_bitrate = v.get("video_bitrate").and_then(JsonValue::as_i64);
        req.audio_bitrate = v.get("audio_bitrate").and_then(JsonValue::as_i64);
        req.frames_video = v.get("frames").and_then(JsonValue::as_u64);
        req.no_video = v
            .get("no_video")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        req.no_audio = v
            .get("no_audio")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        if let Some(sz) = v.get("size").and_then(JsonValue::as_str) {
            req.size = Some(parse_size(sz)?);
        }
        if let Some(pf) = v.get("pixel_layout").and_then(JsonValue::as_str) {
            req.pixel_layout = Some(parse_pixel_layout(pf).ok_or_else(|| {
                Error::invalid_argument("transcode", format!("unknown pixel layout '{pf}'"))
            })?);
        }
        req.sample_rate = v
            .get("sample_rate")
            .and_then(JsonValue::as_u64)
            .map(|n| n as u32);
        req.channels = v
            .get("channels")
            .and_then(JsonValue::as_u64)
            .map(|n| n as u32);
        req.start_sec = v.get("start_sec").and_then(JsonValue::as_f64);
        req.duration_sec = v.get("duration_sec").and_then(JsonValue::as_f64);
        req.strict = v
            .get("strict")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        req.preserve_all = v
            .get("preserve_all")
            .and_then(JsonValue::as_bool)
            .unwrap_or(true);
        req.overwrite = v
            .get("overwrite")
            .and_then(JsonValue::as_bool)
            .unwrap_or(true);
        req.prefer_gpu = v
            .get("prefer_gpu")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        Ok(req)
    }
}

#[derive(Debug, Clone)]
pub struct ExplainReq {
    pub transcode: TranscodeReq,
}

impl ExplainReq {
    pub fn new(transcode: TranscodeReq) -> Self {
        Self { transcode }
    }

    pub fn from_json(v: &JsonValue) -> Result<Self> {
        Ok(Self {
            transcode: TranscodeReq::from_json(v)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RunReport {
    pub operation: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub format: String,
    pub tracks: usize,
    pub packets_processed: u64,
    pub units_processed: u64,
    pub conversion_notes: Vec<String>,
    pub gpu_warnings: Vec<String>,
    pub smart_cut_segments: Vec<String>,
    pub health: HealthSummary,
}

impl RunReport {
    pub fn format_human(&self) -> String {
        let mut out = String::with_capacity(256);
        let _ = writeln!(
            out,
            "{}: {} -> {} (format={}, tracks={}, packets={}, units={})",
            self.operation,
            self.inputs.join(", "),
            self.output,
            self.format,
            self.tracks,
            self.packets_processed,
            self.units_processed
        );
        for note in &self.conversion_notes {
            let _ = writeln!(out, "  note: {note}");
        }
        for warn in &self.gpu_warnings {
            let _ = writeln!(out, "  gpu: {warn}");
        }
        for seg in &self.smart_cut_segments {
            let _ = writeln!(out, "  segment: {seg}");
        }
        if !self.health.is_clean {
            let _ = writeln!(out, "  WARNING: {}", self.health.format_warning_line());
        }
        out
    }

    pub fn to_json(&self) -> String {
        JsonValue::obj(vec![
            ("operation", self.operation.clone().into()),
            ("inputs", self.inputs.clone().into()),
            ("output", self.output.clone().into()),
            ("format", self.format.clone().into()),
            ("tracks", self.tracks.into()),
            ("packets_processed", self.packets_processed.into()),
            ("units_processed", self.units_processed.into()),
            ("conversion_notes", self.conversion_notes.clone().into()),
            ("gpu_warnings", self.gpu_warnings.clone().into()),
            ("smart_cut_segments", self.smart_cut_segments.clone().into()),
            ("health", self.health.to_json_value()),
        ])
        .to_pretty()
    }
}

pub fn run_explain(reg: &Registry, req: &ExplainReq) -> Result<ExecutionPlan> {
    let (_graph, plan, _fmt, _tracks) = build_transcode_graph(reg, &req.transcode, true)?;
    Ok(plan)
}

pub fn run_transcode(reg: &Registry, req: &TranscodeReq) -> Result<RunReport> {
    check_overwrite(&req.output, req.overwrite)?;
    let (mut graph, plan, out_fmt, track_count) = build_transcode_graph(reg, req, false)?;
    let pool = BufferPool::new();
    let report = graph.run(&pool)?;
    Ok(RunReport {
        operation: "transcode".to_string(),
        inputs: vec![req.input.clone()],
        output: req.output.clone(),
        format: out_fmt,
        tracks: track_count,
        packets_processed: report.packets_processed,
        units_processed: report.units_processed,
        conversion_notes: plan
            .conversion_notes
            .iter()
            .map(ToString::to_string)
            .collect(),
        gpu_warnings: plan.gpu_warnings,
        smart_cut_segments: Vec::new(),
        health: HealthSummary::from_stream_health(&report.health),
    })
}

fn build_transcode_graph(
    reg: &Registry,
    req: &TranscodeReq,
    compile_only: bool,
) -> Result<(CompiledGraph, ExecutionPlan, String, usize)> {
    let policy = if req.strict {
        ErrorPolicy::Strict
    } else {
        ErrorPolicy::Resilient
    };
    let demuxer = open_demuxer_path(reg, &req.input, policy)?;
    let in_hdr = demuxer.header().clone();
    let out_fmt = resolve_muxer_name(reg, &req.output, req.format.as_deref())?;
    let muxer_factory = reg
        .find_muxer_by_name(&out_fmt)
        .ok_or_else(|| Error::not_found("transcode", format!("muxer '{out_fmt}' not found")))?;
    let muxer_desc = muxer_factory.descriptor();

    let copy_video = req.video_codec.as_deref() == Some("copy");
    let copy_audio = req.audio_codec.as_deref() == Some("copy");

    let mut builder = GraphBuilder::new();
    let n_demux = builder.add_demuxer(demuxer);

    let mut out_header = ContainerHeader::new(muxer_desc.name);
    out_header.time_base = in_hdr.time_base;
    if req.preserve_all {
        out_header.metadata = in_hdr.metadata.clone();
        out_header.chapters = in_hdr.chapters.clone();
        out_header.attachments = in_hdr.attachments.clone();
    }

    let mut mux_inputs: Vec<PortContract> = Vec::new();
    let mut pending_mux_edges: Vec<(NodeId, usize)> = Vec::new();

    let mut seen_video = false;
    let mut seen_audio = false;
    let pool = BufferPool::new();

    let trim_clip = build_clip_range_from_sec(req.start_sec, req.duration_sec)?;

    for (in_port, in_track) in in_hdr.tracks.iter().enumerate() {
        let media_type = in_track.params.media_type();
        match media_type {
            MediaType::Video => {
                if req.no_video {
                    continue;
                }
                if !muxer_supports_media(&out_fmt, MediaType::Video) {
                    continue;
                }
                if !req.preserve_all && seen_video {
                    continue;
                }
                seen_video = true;

                let out_id = TrackId(out_header.tracks.len() as u32);
                let in_pkt_contract = packet_contract_from_track(in_track);

                if copy_video {
                    let mut out_track = in_track.clone();
                    out_track.id = out_id;
                    let pkt_contract =
                        packet_contract_for_mux(&out_fmt, &out_track, &in_pkt_contract);
                    let mut remap = PacketTransformNode::new(
                        format!("remap_copy(v:{})", out_id.0),
                        in_pkt_contract,
                        pkt_contract.clone(),
                        out_id,
                    );
                    remap.max_packets = req.frames_video;
                    if let Some(clip) = trim_clip {
                        remap.trim = Some((clip, true));
                    }
                    let n_remap = builder.add_node(Box::new(remap));
                    builder.connect(n_demux, in_port, n_remap, 0)?;
                    mux_inputs.push(PortContract::Packet(pkt_contract));
                    pending_mux_edges.push((n_remap, 0));
                    out_header.tracks.push(out_track);
                    continue;
                }

                // Decode + optional filter + encode
                let vp = in_track.params.video().ok_or_else(|| {
                    Error::internal("transcode", "video track missing VideoStreamParams")
                })?;
                let dec_factory = reg.find_decoder(in_track.codec_id).ok_or_else(|| {
                    Error::not_found(
                        "transcode",
                        format!("no decoder for '{}'", in_track.codec_id.name()),
                    )
                })?;
                let mut dec_ctx = CodecContext::new(Arc::clone(&pool));
                dec_ctx.error_policy = policy;
                let decoder = dec_factory.open(in_track, dec_ctx)?;

                let out_codec_id = match req.video_codec.as_deref() {
                    Some(name) => CodecId::from_name(name).ok_or_else(|| {
                        Error::invalid_argument(
                            "transcode",
                            format!("unknown video codec '{name}'"),
                        )
                    })?,
                    None => {
                        if muxer_desc.default_video == CodecId::None {
                            return Err(Error::invalid_argument(
                                "transcode",
                                format!("format '{out_fmt}' has no default video codec"),
                            ));
                        }
                        muxer_desc.default_video
                    }
                };
                let enc_factory = reg.find_encoder(out_codec_id).ok_or_else(|| {
                    Error::not_found(
                        "transcode",
                        format!("no encoder for '{}'", out_codec_id.name()),
                    )
                })?;

                let mut dec_contract = VideoContract::new(vp.width, vp.height, vp.layout)
                    .with_color(vp.color)
                    .with_crop(vp.crop);
                if let Some(m) = vp.display_matrix {
                    dec_contract = dec_contract.with_display_matrix(m);
                }

                let (dst_w, dst_h) = req.size.unwrap_or((vp.width, vp.height));
                let dst_layout = req
                    .pixel_layout
                    .unwrap_or_else(|| compatible_video_encoder_layout(out_codec_id, vp.layout));
                let dst_color = vp.color;
                let mut enc_contract =
                    VideoContract::new(dst_w, dst_h, dst_layout).with_color(dst_color);
                if req.size.is_none()
                    && req.pixel_layout.is_none()
                    && let Some(m) = vp.display_matrix
                {
                    enc_contract = enc_contract.with_display_matrix(m);
                }

                let mut out_vp = VideoStreamParams::new(dst_w, dst_h, dst_layout);
                out_vp.color = dst_color;
                out_vp.sar = vp.sar;
                out_vp.avg_frame_rate = vp.avg_frame_rate;
                out_vp.bit_rate = req.video_bitrate.unwrap_or(vp.bit_rate);
                if req.preserve_all && req.size.is_none() {
                    out_vp.display_matrix = vp.display_matrix;
                }

                let mut out_track = TrackDescriptor::new(
                    out_id,
                    out_codec_id,
                    in_track.time_base,
                    StreamParams::Video(out_vp),
                );
                if req.preserve_all {
                    out_track.language = in_track.language.clone();
                    out_track.disposition = in_track.disposition;
                    out_track.metadata = in_track.metadata.clone();
                }

                let enc_ctx = CodecContext::new(Arc::clone(&pool));
                let encoder = enc_factory.open(&out_track, enc_ctx, PassMode::Single)?;
                let extradata = encoder.extradata().to_vec();
                if let StreamParams::Video(ref mut vp_mut) = out_track.params {
                    vp_mut.extradata = extradata.clone();
                }

                let enc_pkt_contract = encoded_packet_contract(&out_fmt, out_codec_id, extradata);

                let n_dec = builder.add_decoder(
                    decoder,
                    in_pkt_contract,
                    PortContract::Video(dec_contract.clone()),
                );
                builder.connect(n_demux, in_port, n_dec, 0)?;

                let mut cur_node = n_dec;
                let cur_contract = dec_contract;

                if let Some(clip) = trim_clip {
                    let n_trim = builder.add_video_filter(
                        "trim_video",
                        cur_contract.clone(),
                        TrimFilter::new(clip, true),
                    )?;
                    builder.connect(cur_node, 0, n_trim, 0)?;
                    cur_node = n_trim;
                }

                if let Some(limit) = req.frames_video {
                    let n_limit = builder
                        .add_node(Box::new(VideoLimitNode::new(cur_contract.clone(), limit)));
                    builder.connect(cur_node, 0, n_limit, 0)?;
                    cur_node = n_limit;
                }

                let n_enc = builder.add_encoder(
                    encoder,
                    PortContract::Video(enc_contract),
                    enc_pkt_contract.clone(),
                );
                builder.connect(cur_node, 0, n_enc, 0)?;
                mux_inputs.push(PortContract::Packet(enc_pkt_contract));
                pending_mux_edges.push((n_enc, 0));
                out_header.tracks.push(out_track);
            }
            MediaType::Audio => {
                if req.no_audio {
                    continue;
                }
                if !muxer_supports_media(&out_fmt, MediaType::Audio) {
                    continue;
                }
                if !req.preserve_all && seen_audio {
                    continue;
                }
                seen_audio = true;

                let out_id = TrackId(out_header.tracks.len() as u32);
                let in_pkt_contract = packet_contract_from_track(in_track);

                if copy_audio {
                    let mut out_track = in_track.clone();
                    out_track.id = out_id;
                    let pkt_contract =
                        packet_contract_for_mux(&out_fmt, &out_track, &in_pkt_contract);
                    let mut remap = PacketTransformNode::new(
                        format!("remap_copy(a:{})", out_id.0),
                        in_pkt_contract,
                        pkt_contract.clone(),
                        out_id,
                    );
                    if let Some(clip) = trim_clip {
                        remap.trim = Some((clip, true));
                    }
                    let n_remap = builder.add_node(Box::new(remap));
                    builder.connect(n_demux, in_port, n_remap, 0)?;
                    mux_inputs.push(PortContract::Packet(pkt_contract));
                    pending_mux_edges.push((n_remap, 0));
                    out_header.tracks.push(out_track);
                    continue;
                }

                let ap = in_track.params.audio().ok_or_else(|| {
                    Error::internal("transcode", "audio track missing AudioStreamParams")
                })?;
                let dec_factory = reg.find_decoder(in_track.codec_id).ok_or_else(|| {
                    Error::not_found(
                        "transcode",
                        format!("no decoder for '{}'", in_track.codec_id.name()),
                    )
                })?;
                let mut dec_ctx = CodecContext::new(Arc::clone(&pool));
                dec_ctx.error_policy = policy;
                let decoder = dec_factory.open(in_track, dec_ctx)?;

                let out_codec_id = match req.audio_codec.as_deref() {
                    Some(name) => CodecId::from_name(name).ok_or_else(|| {
                        Error::invalid_argument(
                            "transcode",
                            format!("unknown audio codec '{name}'"),
                        )
                    })?,
                    None => {
                        if muxer_desc.default_audio == CodecId::None {
                            return Err(Error::invalid_argument(
                                "transcode",
                                format!("format '{out_fmt}' has no default audio codec"),
                            ));
                        }
                        muxer_desc.default_audio
                    }
                };
                let enc_factory = reg.find_encoder(out_codec_id).ok_or_else(|| {
                    Error::not_found(
                        "transcode",
                        format!("no encoder for '{}'", out_codec_id.name()),
                    )
                })?;

                let mut dec_contract =
                    AudioContract::new(ap.format, ap.sample_rate, ap.channel_layout.clone());
                if ap.frame_size > 0 {
                    dec_contract = dec_contract.with_frame_size(Some(ap.frame_size));
                }

                let dst_sr = req.sample_rate.unwrap_or(ap.sample_rate);
                let dst_layout = req
                    .channels
                    .map(ChannelLayout::default_for_count)
                    .unwrap_or_else(|| ap.channel_layout.clone());
                let dst_fmt = compatible_audio_encoder_format(out_codec_id, ap.format);
                let enc_contract = AudioContract::new(dst_fmt, dst_sr, dst_layout.clone());

                let mut out_ap = AudioStreamParams::new(dst_sr, dst_fmt, dst_layout);
                out_ap.bit_rate = req.audio_bitrate.unwrap_or(ap.bit_rate);

                let mut out_track = TrackDescriptor::new(
                    out_id,
                    out_codec_id,
                    in_track.time_base,
                    StreamParams::Audio(out_ap),
                );
                if req.preserve_all {
                    out_track.language = in_track.language.clone();
                    out_track.disposition = in_track.disposition;
                    out_track.metadata = in_track.metadata.clone();
                }

                let enc_ctx = CodecContext::new(Arc::clone(&pool));
                let encoder = enc_factory.open(&out_track, enc_ctx, PassMode::Single)?;
                let extradata = encoder.extradata().to_vec();
                if let StreamParams::Audio(ref mut ap_mut) = out_track.params {
                    ap_mut.extradata = extradata.clone();
                }
                let enc_pkt_contract = encoded_packet_contract(&out_fmt, out_codec_id, extradata);

                let n_dec = builder.add_decoder(
                    decoder,
                    in_pkt_contract,
                    PortContract::Audio(dec_contract.clone()),
                );
                builder.connect(n_demux, in_port, n_dec, 0)?;

                let mut cur_node = n_dec;
                if let Some(clip) = trim_clip {
                    let n_trim = builder.add_audio_filter(
                        "trim_audio",
                        dec_contract,
                        TrimFilter::new(clip, true),
                    )?;
                    builder.connect(cur_node, 0, n_trim, 0)?;
                    cur_node = n_trim;
                }

                let n_enc = builder.add_encoder(
                    encoder,
                    PortContract::Audio(enc_contract),
                    enc_pkt_contract.clone(),
                );
                builder.connect(cur_node, 0, n_enc, 0)?;
                mux_inputs.push(PortContract::Packet(enc_pkt_contract));
                pending_mux_edges.push((n_enc, 0));
                out_header.tracks.push(out_track);
            }
            MediaType::Subtitle
                if req.preserve_all && muxer_supports_media(&out_fmt, MediaType::Subtitle) =>
            {
                let out_id = TrackId(out_header.tracks.len() as u32);
                let in_pkt_contract = packet_contract_from_track(in_track);
                let mut out_track = in_track.clone();
                out_track.id = out_id;
                let mut remap = PacketTransformNode::new(
                    format!("remap_sub({})", out_id.0),
                    in_pkt_contract.clone(),
                    in_pkt_contract.clone(),
                    out_id,
                );
                if let Some(clip) = trim_clip {
                    remap.trim = Some((clip, true));
                }
                let n_remap = builder.add_node(Box::new(remap));
                builder.connect(n_demux, in_port, n_remap, 0)?;
                mux_inputs.push(PortContract::Packet(in_pkt_contract));
                pending_mux_edges.push((n_remap, 0));
                out_header.tracks.push(out_track);
            }
            _ => {}
        }
    }

    if out_header.tracks.is_empty() {
        return Err(Error::invalid_argument(
            "transcode",
            format!("no compatible tracks selected for output format '{out_fmt}'"),
        ));
    }

    let sink: Box<dyn ByteSink> = if compile_only {
        Box::new(MemorySink::new())
    } else {
        Box::new(FileSink::create(&req.output)?)
    };
    let muxer = muxer_factory.create(sink)?;
    let track_count = out_header.tracks.len();
    let n_mux = builder.add_muxer(muxer, out_header, mux_inputs);

    for (to_port, (from_node, from_port)) in pending_mux_edges.into_iter().enumerate() {
        builder.connect(from_node, from_port, n_mux, to_port)?;
    }

    let mut compile_opts = CompileOpts::default()
        .with_strict(req.strict)
        .with_preserve_all(req.preserve_all);
    if req.prefer_gpu {
        compile_opts = compile_opts.with_prefer_gpu(Some(wedeo::GpuDeviceId(0)));
    }

    let (graph, plan) = builder.compile(compile_opts)?;
    Ok((graph, plan, out_fmt, track_count))
}

fn build_clip_range_from_sec(
    start_sec: Option<f64>,
    duration_sec: Option<f64>,
) -> Result<Option<wedeo::graph::ClipRange>> {
    if start_sec.is_none() && duration_sec.is_none() {
        return Ok(None);
    }
    let tb = Rational::new(1, 1_000_000);
    let s_us = (start_sec.unwrap_or(0.0).max(0.0) * 1_000_000.0).round() as i64;
    let e_us = match duration_sec {
        Some(d) => {
            if d <= 0.0 {
                return Err(Error::invalid_argument(
                    "transcode",
                    "duration (-t) must be > 0",
                ));
            }
            s_us.saturating_add((d * 1_000_000.0).round() as i64)
        }
        None => i64::MAX / 2,
    };
    let clip = wedeo::graph::ClipRange::from_ticks(s_us, e_us, tb);
    clip.validate()?;
    Ok(Some(clip))
}

// ============================================================================
// 4. CutReq -> RunReport (powered by wedeo_graph::timeline)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CutMode {
    Keyframe,
    #[default]
    Smart,
    Reencode,
}

impl CutMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "keyframe" | "copy" => Ok(Self::Keyframe),
            "smart" => Ok(Self::Smart),
            "reencode" | "re-encode" => Ok(Self::Reencode),
            other => Err(Error::invalid_argument(
                "cut",
                format!("unknown cut mode '{other}'; expected keyframe|smart|reencode"),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CutReq {
    pub input: String,
    pub output: String,
    pub from_sec: f64,
    pub to_sec: f64,
    pub mode: CutMode,
    pub format: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub strict: bool,
    pub preserve_all: bool,
    pub overwrite: bool,
}

impl CutReq {
    pub fn new(
        input: impl Into<String>,
        output: impl Into<String>,
        from_sec: f64,
        to_sec: f64,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            from_sec,
            to_sec,
            mode: CutMode::Smart,
            format: None,
            video_codec: None,
            audio_codec: None,
            strict: false,
            preserve_all: true,
            overwrite: false,
        }
    }

    pub fn from_json(v: &JsonValue) -> Result<Self> {
        let input = v
            .get("input")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| Error::invalid_argument("cut", "missing string field 'input'"))?
            .to_string();
        let output = v
            .get("output")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| Error::invalid_argument("cut", "missing string field 'output'"))?
            .to_string();
        let from_sec = v
            .get("from_sec")
            .and_then(JsonValue::as_f64)
            .ok_or_else(|| Error::invalid_argument("cut", "missing numeric field 'from_sec'"))?;
        let to_sec = v
            .get("to_sec")
            .and_then(JsonValue::as_f64)
            .ok_or_else(|| Error::invalid_argument("cut", "missing numeric field 'to_sec'"))?;
        let mode = match v.get("mode").and_then(JsonValue::as_str) {
            Some(m) => CutMode::parse(m)?,
            None => CutMode::Smart,
        };
        let mut req = Self::new(input, output, from_sec, to_sec);
        req.mode = mode;
        req.format = v
            .get("format")
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        req.video_codec = v
            .get("video_codec")
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        req.audio_codec = v
            .get("audio_codec")
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        req.strict = v
            .get("strict")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        req.preserve_all = v
            .get("preserve_all")
            .and_then(JsonValue::as_bool)
            .unwrap_or(true);
        req.overwrite = v
            .get("overwrite")
            .and_then(JsonValue::as_bool)
            .unwrap_or(true);
        Ok(req)
    }
}

pub fn run_cut(reg: &Registry, req: &CutReq) -> Result<RunReport> {
    if req.from_sec < 0.0 || req.to_sec <= req.from_sec {
        return Err(Error::invalid_argument(
            "cut",
            "cut range requires 0 <= from < to",
        ));
    }
    check_overwrite(&req.output, req.overwrite)?;

    let policy = if req.strict {
        ErrorPolicy::Strict
    } else {
        ErrorPolicy::Resilient
    };

    // Index keyframes on the input file for SmartCutPlan
    let mut scan_demuxer = open_demuxer_path(reg, &req.input, policy)?;
    let in_hdr = scan_demuxer.header().clone();
    let video_track = in_hdr
        .tracks
        .iter()
        .find(|t| t.params.media_type() == MediaType::Video)
        .cloned();

    let us_tb = Rational::new(1, 1_000_000);
    let mut keyframes = Vec::new();
    let mut stream_end = Timestamp::NONE;
    if let Some(ref vt) = video_track {
        let tb = if vt.time_base.is_valid() {
            vt.time_base
        } else {
            us_tb
        };
        if vt.duration > 0 {
            let dur_us = rescale_q(vt.duration, tb, us_tb);
            stream_end = Timestamp::new(dur_us);
        }
        if let Some(hdr_dur) = in_hdr.duration
            && hdr_dur > 0
            && in_hdr.time_base.is_valid()
        {
            let dur_us = rescale_q(hdr_dur, in_hdr.time_base, us_tb);
            stream_end = Timestamp::new(stream_end.get().map_or(dur_us, |c| c.max(dur_us)));
        }
        let mut last_kf_pts: Option<i64> = None;
        loop {
            match scan_demuxer.read_packet()? {
                RecvStatus::Ready(pkt) => {
                    if pkt.track_id == vt.id {
                        let pts = pkt.pts.get().or_else(|| pkt.dts.get());
                        if let Some(p) = pts {
                            let pkt_tb = if pkt.time_base.is_valid() {
                                pkt.time_base
                            } else {
                                tb
                            };
                            let end_p = p.saturating_add(pkt.duration.max(1));
                            let end_us = rescale_q(end_p, pkt_tb, us_tb);
                            stream_end = Timestamp::new(
                                stream_end.get().map_or(end_us, |cur| cur.max(end_us)),
                            );
                            if pkt.flags.contains(PacketFlags::KEY)
                                && last_kf_pts.is_none_or(|prev| p > prev)
                            {
                                keyframes.push(wedeo::graph::KeyframeEntry::new(
                                    Timestamp::new(p),
                                    pkt.dts,
                                    pkt.pos,
                                    pkt_tb,
                                ));
                                last_kf_pts = Some(p);
                            }
                        }
                    }
                }
                RecvStatus::NeedMore => continue,
                RecvStatus::Eof => break,
            }
        }
    }

    let clip = wedeo::graph::ClipRange::from_ticks(
        (req.from_sec * 1_000_000.0).round() as i64,
        (req.to_sec * 1_000_000.0).round() as i64,
        us_tb,
    );
    clip.validate()?;

    let mut smart_plan_opt: Option<wedeo::graph::SmartCutPlan> = None;
    let mut segment_notes: Vec<String> = Vec::new();
    if !keyframes.is_empty() {
        let plan = wedeo::graph::SmartCutPlan::compute(clip, &keyframes, stream_end)?;
        for seg in &plan.segments {
            segment_notes.push(format!("{seg:?}"));
        }
        smart_plan_opt = Some(plan);
    }

    match req.mode {
        CutMode::Reencode => {
            let mut tr = TranscodeReq::new(&req.input, &req.output);
            tr.format = req.format.clone();
            tr.video_codec = req.video_codec.clone();
            tr.audio_codec = req.audio_codec.clone();
            tr.start_sec = Some(req.from_sec);
            tr.duration_sec = Some(req.to_sec - req.from_sec);
            tr.strict = req.strict;
            tr.preserve_all = req.preserve_all;
            tr.overwrite = req.overwrite;
            let mut report = run_transcode(reg, &tr)?;
            report.operation = "cut(reencode)".to_string();
            report.smart_cut_segments = segment_notes;
            Ok(report)
        }
        CutMode::Keyframe => {
            let snapped_clip = if let Some(ref sp) = smart_plan_opt
                && let Some(wedeo::graph::GopSegment::ReencodeHead {
                    decode_from,
                    time_base,
                    ..
                }) = sp.head()
            {
                let snapped_start =
                    rescale_q(decode_from.raw(), *time_base, us_tb).min(clip.start.raw());
                wedeo::graph::ClipRange::from_ticks(snapped_start, clip.end.raw(), us_tb)
            } else {
                clip
            };
            let mut tr = TranscodeReq::new(&req.input, &req.output);
            tr.format = req.format.clone();
            tr.video_codec = Some("copy".to_string());
            tr.audio_codec = Some("copy".to_string());
            tr.start_sec = snapped_clip.start.to_seconds(us_tb);
            tr.duration_sec = snapped_clip.duration().map(|d| d as f64 / 1_000_000.0);
            tr.strict = req.strict;
            tr.preserve_all = req.preserve_all;
            tr.overwrite = req.overwrite;
            let mut report = run_transcode(reg, &tr)?;
            report.operation = "cut(keyframe)".to_string();
            report.smart_cut_segments = segment_notes;
            Ok(report)
        }
        CutMode::Smart => {
            let need_reencode_edges = smart_plan_opt
                .as_ref()
                .is_some_and(|p| p.head().is_some() || p.tail().is_some());
            let same_codec_ok = video_track.as_ref().is_some_and(|vt| {
                req.video_codec
                    .as_deref()
                    .is_none_or(|c| c == "copy" || CodecId::from_name(c) == Some(vt.codec_id))
            });

            if !need_reencode_edges && same_codec_ok {
                let mut tr = TranscodeReq::new(&req.input, &req.output);
                tr.format = req.format.clone();
                tr.video_codec = Some("copy".to_string());
                tr.audio_codec = Some("copy".to_string());
                tr.start_sec = Some(req.from_sec);
                tr.duration_sec = Some(req.to_sec - req.from_sec);
                tr.strict = req.strict;
                tr.preserve_all = req.preserve_all;
                tr.overwrite = req.overwrite;
                let mut report = run_transcode(reg, &tr)?;
                report.operation = "cut(smart-copy)".to_string();
                report.smart_cut_segments = segment_notes;
                return Ok(report);
            }

            if let Some(vt) = video_track.as_ref()
                && let Some(smart_plan) = smart_plan_opt.as_ref()
                && reg.find_encoder(vt.codec_id).is_some()
                && req
                    .video_codec
                    .as_deref()
                    .is_none_or(|c| CodecId::from_name(c) == Some(vt.codec_id))
            {
                return run_mixed_smart_cut(reg, req, vt, smart_plan, segment_notes, policy);
            }

            if req.strict {
                return Err(Error::unsupported(
                    "cut",
                    "strict smart cut requires a matching-codec encoder for head/tail GOPs",
                ));
            }

            let mut tr = TranscodeReq::new(&req.input, &req.output);
            tr.format = req.format.clone();
            tr.video_codec = req.video_codec.clone();
            tr.audio_codec = req.audio_codec.clone();
            tr.start_sec = Some(req.from_sec);
            tr.duration_sec = Some(req.to_sec - req.from_sec);
            tr.strict = false;
            tr.preserve_all = req.preserve_all;
            tr.overwrite = req.overwrite;
            let mut report = run_transcode(reg, &tr)?;
            report.operation = "cut(smart-reencode)".to_string();
            report.conversion_notes.push(
                "smart cut fell back to full-clip re-encode (no same-codec encoder for head/tail GOP)"
                    .to_string(),
            );
            report.smart_cut_segments = segment_notes;
            Ok(report)
        }
    }
}

fn run_mixed_smart_cut(
    reg: &Registry,
    req: &CutReq,
    video_track: &TrackDescriptor,
    smart_plan: &wedeo::graph::SmartCutPlan,
    segment_notes: Vec<String>,
    policy: ErrorPolicy,
) -> Result<RunReport> {
    let demuxer = open_demuxer_path(reg, &req.input, policy)?;
    let in_hdr = demuxer.header().clone();
    let out_fmt = resolve_muxer_name(reg, &req.output, req.format.as_deref())?;
    let muxer_factory = reg
        .find_muxer_by_name(&out_fmt)
        .ok_or_else(|| Error::not_found("cut", format!("muxer '{out_fmt}' not found")))?;
    let muxer_desc = muxer_factory.descriptor();
    let pool = BufferPool::new();

    let mut builder = GraphBuilder::new();
    let n_demux = builder.add_demuxer(demuxer);

    let mut out_header = ContainerHeader::new(muxer_desc.name);
    out_header.time_base = in_hdr.time_base;
    if req.preserve_all {
        out_header.metadata = in_hdr.metadata.clone();
        out_header.chapters = in_hdr.chapters.clone();
        out_header.attachments = in_hdr.attachments.clone();
    }

    let mut mux_inputs = Vec::new();
    let mut pending_mux_edges = Vec::new();
    let timeline = wedeo::graph::Timeline::single(smart_plan.clip);

    for (in_port, in_track) in in_hdr.tracks.iter().enumerate() {
        let out_id = TrackId(out_header.tracks.len() as u32);
        let in_pkt_contract = packet_contract_from_track(in_track);

        if in_track.id == video_track.id {
            let dec_factory = reg.find_decoder(in_track.codec_id).ok_or_else(|| {
                Error::not_found(
                    "cut",
                    format!("no decoder for '{}'", in_track.codec_id.name()),
                )
            })?;
            let enc_factory = reg.find_encoder(in_track.codec_id).ok_or_else(|| {
                Error::not_found(
                    "cut",
                    format!("no encoder for '{}'", in_track.codec_id.name()),
                )
            })?;

            let mut out_track = in_track.clone();
            out_track.id = out_id;

            let mut dec_ctx = CodecContext::new(Arc::clone(&pool));
            dec_ctx.error_policy = policy;
            let decoder = dec_factory.open(in_track, dec_ctx)?;
            let enc_ctx = CodecContext::new(Arc::clone(&pool));
            let encoder = enc_factory.open(&out_track, enc_ctx, PassMode::Single)?;
            if !encoder.extradata().is_empty()
                && let StreamParams::Video(ref mut vp) = out_track.params
            {
                vp.extradata = encoder.extradata().to_vec();
            }

            let out_pkt_contract = packet_contract_for_mux(&out_fmt, &out_track, &in_pkt_contract);
            let smart_node = SmartCutVideoNode::new(
                in_pkt_contract,
                out_pkt_contract.clone(),
                out_id,
                smart_plan.clone(),
                timeline.clone(),
                decoder,
                encoder,
            );
            let n_smart = builder.add_node(Box::new(smart_node));
            builder.connect(n_demux, in_port, n_smart, 0)?;
            mux_inputs.push(PortContract::Packet(out_pkt_contract));
            pending_mux_edges.push((n_smart, 0));
            out_header.tracks.push(out_track);
        } else if muxer_supports_media(&out_fmt, in_track.params.media_type()) {
            let mut out_track = in_track.clone();
            out_track.id = out_id;
            let pkt_contract = packet_contract_for_mux(&out_fmt, &out_track, &in_pkt_contract);
            let mut remap = PacketTransformNode::new(
                format!("cut_audio_copy({})", out_id.0),
                in_pkt_contract,
                pkt_contract.clone(),
                out_id,
            );
            remap.trim = Some((smart_plan.clip, true));
            let n_remap = builder.add_node(Box::new(remap));
            builder.connect(n_demux, in_port, n_remap, 0)?;
            mux_inputs.push(PortContract::Packet(pkt_contract));
            pending_mux_edges.push((n_remap, 0));
            out_header.tracks.push(out_track);
        }
    }

    let sink = Box::new(FileSink::create(&req.output)?);
    let muxer = muxer_factory.create(sink)?;
    let track_count = out_header.tracks.len();
    let n_mux = builder.add_muxer(muxer, out_header, mux_inputs);
    for (to_port, (from_node, from_port)) in pending_mux_edges.into_iter().enumerate() {
        builder.connect(from_node, from_port, n_mux, to_port)?;
    }

    let (mut graph, plan) = builder.compile(
        CompileOpts::default()
            .with_strict(req.strict)
            .with_preserve_all(req.preserve_all),
    )?;
    let report = graph.run(&pool)?;
    Ok(RunReport {
        operation: "cut(smart)".to_string(),
        inputs: vec![req.input.clone()],
        output: req.output.clone(),
        format: out_fmt,
        tracks: track_count,
        packets_processed: report.packets_processed,
        units_processed: report.units_processed,
        conversion_notes: plan
            .conversion_notes
            .iter()
            .map(ToString::to_string)
            .collect(),
        gpu_warnings: plan.gpu_warnings,
        smart_cut_segments: segment_notes,
        health: HealthSummary::from_stream_health(&report.health),
    })
}

// ============================================================================
// 5. MuxReq -> RunReport
// ============================================================================

#[derive(Debug, Clone)]
pub struct MuxReq {
    pub video_inputs: Vec<String>,
    pub audio_inputs: Vec<String>,
    pub extra_inputs: Vec<String>,
    pub output: String,
    pub format: Option<String>,
    pub tags: Vec<(String, String)>,
    pub preserve_all: bool,
    pub strict: bool,
    pub overwrite: bool,
}

impl MuxReq {
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            video_inputs: Vec::new(),
            audio_inputs: Vec::new(),
            extra_inputs: Vec::new(),
            output: output.into(),
            format: None,
            tags: Vec::new(),
            preserve_all: true,
            strict: false,
            overwrite: false,
        }
    }

    pub fn from_json(v: &JsonValue) -> Result<Self> {
        let output = v
            .get("output")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| Error::invalid_argument("mux", "missing string field 'output'"))?
            .to_string();
        let mut req = Self::new(output);
        if let Some(arr) = v.get("video").and_then(JsonValue::as_arr) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    req.video_inputs.push(s.to_string());
                }
            }
        } else if let Some(s) = v.get("video").and_then(JsonValue::as_str) {
            req.video_inputs.push(s.to_string());
        }
        if let Some(arr) = v.get("audio").and_then(JsonValue::as_arr) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    req.audio_inputs.push(s.to_string());
                }
            }
        } else if let Some(s) = v.get("audio").and_then(JsonValue::as_str) {
            req.audio_inputs.push(s.to_string());
        }
        if let Some(arr) = v.get("inputs").and_then(JsonValue::as_arr) {
            for item in arr {
                if let Some(s) = item.as_str() {
                    req.extra_inputs.push(s.to_string());
                }
            }
        }
        req.format = v
            .get("format")
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        if let Some(obj) = v.get("tags").and_then(JsonValue::as_obj) {
            for (k, val) in obj {
                if let Some(vs) = val.as_str() {
                    req.tags.push((k.clone(), vs.to_string()));
                }
            }
        }
        req.preserve_all = v
            .get("preserve_all")
            .and_then(JsonValue::as_bool)
            .unwrap_or(true);
        req.strict = v
            .get("strict")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        req.overwrite = v
            .get("overwrite")
            .and_then(JsonValue::as_bool)
            .unwrap_or(true);
        Ok(req)
    }
}

pub fn run_mux(reg: &Registry, req: &MuxReq) -> Result<RunReport> {
    if req.video_inputs.is_empty() && req.audio_inputs.is_empty() && req.extra_inputs.is_empty() {
        return Err(Error::invalid_argument(
            "mux",
            "at least one input (--video, --audio, or --input) is required",
        ));
    }
    check_overwrite(&req.output, req.overwrite)?;

    let policy = if req.strict {
        ErrorPolicy::Strict
    } else {
        ErrorPolicy::Resilient
    };
    let out_fmt = resolve_muxer_name(reg, &req.output, req.format.as_deref())?;
    let muxer_factory = reg
        .find_muxer_by_name(&out_fmt)
        .ok_or_else(|| Error::not_found("mux", format!("muxer '{out_fmt}' not found")))?;
    let muxer_desc = muxer_factory.descriptor();

    let mut builder = GraphBuilder::new();
    let mut out_header = ContainerHeader::new(muxer_desc.name);
    let mut mux_inputs = Vec::new();
    let mut pending_mux_edges = Vec::new();
    let mut all_inputs = Vec::new();

    #[derive(Clone, Copy)]
    enum TrackFilter {
        VideoOnly,
        AudioOnly,
        AllSupported,
    }

    let mut sources: Vec<(&str, TrackFilter)> = Vec::new();
    for p in &req.video_inputs {
        sources.push((p.as_str(), TrackFilter::VideoOnly));
    }
    for p in &req.audio_inputs {
        sources.push((p.as_str(), TrackFilter::AudioOnly));
    }
    for p in &req.extra_inputs {
        sources.push((p.as_str(), TrackFilter::AllSupported));
    }

    for (path, filter) in sources {
        all_inputs.push(path.to_string());
        let demuxer = open_demuxer_path(reg, path, policy)?;
        let in_hdr = demuxer.header().clone();
        if req.preserve_all {
            for (k, v) in in_hdr.metadata.iter() {
                if out_header.metadata.get(k).is_none() {
                    out_header.metadata.set(k, v);
                }
            }
            out_header.chapters.extend(in_hdr.chapters.clone());
            out_header.attachments.extend(in_hdr.attachments.clone());
        }

        let n_demux = builder.add_demuxer(demuxer);
        let mut matched_for_source = 0usize;

        for (in_port, in_track) in in_hdr.tracks.iter().enumerate() {
            let mt = in_track.params.media_type();
            let take = match filter {
                TrackFilter::VideoOnly => mt == MediaType::Video,
                TrackFilter::AudioOnly => mt == MediaType::Audio,
                TrackFilter::AllSupported => muxer_supports_media(&out_fmt, mt),
            };
            if !take {
                continue;
            }
            if !muxer_supports_media(&out_fmt, mt) {
                return Err(Error::unsupported(
                    "mux",
                    format!("output format '{out_fmt}' does not support {mt} tracks"),
                ));
            }

            matched_for_source += 1;
            let out_id = TrackId(out_header.tracks.len() as u32);
            let mut out_track = in_track.clone();
            out_track.id = out_id;

            let in_pkt_contract = packet_contract_from_track(in_track);
            let out_pkt_contract = packet_contract_for_mux(&out_fmt, &out_track, &in_pkt_contract);

            let remap = PacketTransformNode::new(
                format!("mux_remap({})", out_id.0),
                in_pkt_contract,
                out_pkt_contract.clone(),
                out_id,
            );
            let n_remap = builder.add_node(Box::new(remap));
            builder.connect(n_demux, in_port, n_remap, 0)?;
            mux_inputs.push(PortContract::Packet(out_pkt_contract));
            pending_mux_edges.push((n_remap, 0));
            out_header.tracks.push(out_track);
        }

        if matched_for_source == 0 {
            return Err(Error::not_found(
                "mux",
                format!("no matching tracks found in '{path}'"),
            ));
        }
    }

    for (k, v) in &req.tags {
        out_header.metadata.set(k, v);
    }

    let sink = Box::new(FileSink::create(&req.output)?);
    let muxer = muxer_factory.create(sink)?;
    let track_count = out_header.tracks.len();
    let n_mux = builder.add_muxer(muxer, out_header, mux_inputs);
    for (to_port, (from_node, from_port)) in pending_mux_edges.into_iter().enumerate() {
        builder.connect(from_node, from_port, n_mux, to_port)?;
    }

    let (mut graph, plan) = builder.compile(
        CompileOpts::default()
            .with_strict(req.strict)
            .with_preserve_all(req.preserve_all),
    )?;
    let pool = BufferPool::new();
    let report = graph.run(&pool)?;

    Ok(RunReport {
        operation: "mux".to_string(),
        inputs: all_inputs,
        output: req.output.clone(),
        format: out_fmt,
        tracks: track_count,
        packets_processed: report.packets_processed,
        units_processed: report.units_processed,
        conversion_notes: plan
            .conversion_notes
            .iter()
            .map(ToString::to_string)
            .collect(),
        gpu_warnings: plan.gpu_warnings,
        smart_cut_segments: Vec::new(),
        health: HealthSummary::from_stream_health(&report.health),
    })
}

// ============================================================================
// Codec & Format Listing Helpers
// ============================================================================

pub fn format_codecs_table(reg: &Registry) -> String {
    let mut seen: BTreeMap<&str, (bool, bool, CodecId, &str)> = BTreeMap::new();
    for f in reg.decoders() {
        let d = f.descriptor();
        let entry =
            seen.entry(d.codec_id.name())
                .or_insert((false, false, d.codec_id, d.long_name));
        entry.0 = true;
    }
    for f in reg.encoders() {
        let d = f.descriptor();
        let entry =
            seen.entry(d.codec_id.name())
                .or_insert((false, false, d.codec_id, d.long_name));
        entry.1 = true;
    }

    let mut out = String::from(
        "Codecs:\n D..... = Decoding supported\n .E.... = Encoding supported\n ..V... = Video codec\n ..A... = Audio codec\n ..S... = Subtitle codec\n ------\n",
    );
    for (name, (has_dec, has_enc, id, long_name)) in seen {
        let d = if has_dec { 'D' } else { '.' };
        let e = if has_enc { 'E' } else { '.' };
        let t = match id.media_type() {
            Some(MediaType::Video) => 'V',
            Some(MediaType::Audio) => 'A',
            Some(MediaType::Subtitle) => 'S',
            _ => '.',
        };
        let _ = writeln!(out, " {d}{e}{t}... {name:<20} {long_name}");
    }
    out
}

pub fn format_formats_table(reg: &Registry) -> String {
    let mut seen: BTreeMap<&str, (bool, bool, &str)> = BTreeMap::new();
    for f in reg.demuxers() {
        let d = f.descriptor();
        let entry = seen.entry(d.name).or_insert((false, false, d.long_name));
        entry.0 = true;
    }
    for f in reg.muxers() {
        let d = f.descriptor();
        let entry = seen.entry(d.name).or_insert((false, false, d.long_name));
        entry.1 = true;
    }

    let mut out = String::from("Formats:\n D. = Demuxing supported\n .M = Muxing supported\n --\n");
    for (name, (has_d, has_m, long_name)) in seen {
        let d = if has_d { 'D' } else { '.' };
        let m = if has_m { 'M' } else { '.' };
        let _ = writeln!(out, " {d}{m} {name:<20} {long_name}");
    }
    out
}

pub fn format_decoders_table(reg: &Registry) -> String {
    let mut out = String::from("Decoders:\n");
    for f in reg.decoders() {
        let d = f.descriptor();
        let _ = writeln!(out, "  {:<20} {} ({})", d.name, d.long_name, d.media_type);
    }
    out
}

pub fn format_encoders_table(reg: &Registry) -> String {
    let mut out = String::from("Encoders:\n");
    for f in reg.encoders() {
        let d = f.descriptor();
        let _ = writeln!(out, "  {:<20} {} ({})", d.name, d.long_name, d.media_type);
    }
    out
}

// ============================================================================
// Internal Graph Helper Nodes & Contract Helpers
// ============================================================================

fn packet_contract_from_track(t: &TrackDescriptor) -> PacketContract {
    let extradata = match &t.params {
        StreamParams::Video(v) => v.extradata.clone(),
        StreamParams::Audio(a) => a.extradata.clone(),
        StreamParams::Subtitle(s) => s.header_script.clone(),
        StreamParams::Data(d) => d.extradata.clone(),
    };
    let framing = match t.codec_id {
        CodecId::H264 => {
            if extradata.len() >= 7 && extradata[0] == 1 {
                PacketFraming::Avcc((extradata[4] & 0x03) + 1)
            } else {
                PacketFraming::AnnexB
            }
        }
        CodecId::Hevc => {
            if extradata.len() >= 23 && extradata[0] == 1 {
                PacketFraming::Avcc((extradata[21] & 0x03) + 1)
            } else {
                PacketFraming::AnnexB
            }
        }
        CodecId::Av1 => PacketFraming::Obu,
        CodecId::Aac => {
            if extradata.is_empty() {
                PacketFraming::Adts
            } else {
                PacketFraming::Raw
            }
        }
        _ => PacketFraming::Raw,
    };
    PacketContract::new(t.codec_id, framing).with_extradata(extradata)
}

fn muxer_supports_media(fmt: &str, mt: MediaType) -> bool {
    match fmt {
        "wav" => mt == MediaType::Audio,
        "ivf" => mt == MediaType::Video,
        "mp4" | "mov" => matches!(mt, MediaType::Video | MediaType::Audio),
        "matroska" | "mkv" | "webm" => {
            matches!(
                mt,
                MediaType::Video | MediaType::Audio | MediaType::Subtitle
            )
        }
        _ => true,
    }
}

fn compatible_video_encoder_layout(codec: CodecId, src: PixelLayout) -> PixelLayout {
    if codec == CodecId::Av1 {
        if src.depth != wedeo::core::SampleDepth::U8 {
            return PixelLayout::YUV420P;
        }
        match src.subsampling {
            wedeo::core::Subsampling::Cs420
            | wedeo::core::Subsampling::Cs422
            | wedeo::core::Subsampling::Cs444
            | wedeo::core::Subsampling::Cs400 => {
                if src.model == wedeo::core::ColorModel::Ycbcr
                    || src.model == wedeo::core::ColorModel::Gray
                {
                    src
                } else {
                    PixelLayout::YUV420P
                }
            }
            _ => PixelLayout::YUV420P,
        }
    } else {
        src
    }
}

fn compatible_audio_encoder_format(codec: CodecId, src: SampleFormat) -> SampleFormat {
    match codec {
        CodecId::PcmU8 => SampleFormat::U8,
        CodecId::PcmS8
        | CodecId::PcmS16le
        | CodecId::PcmS16be
        | CodecId::PcmU16le
        | CodecId::PcmU16be
        | CodecId::PcmAlaw
        | CodecId::PcmMulaw => SampleFormat::S16,
        CodecId::PcmS24le
        | CodecId::PcmS24be
        | CodecId::PcmU24le
        | CodecId::PcmU24be
        | CodecId::PcmS32le
        | CodecId::PcmS32be
        | CodecId::PcmU32le
        | CodecId::PcmU32be => SampleFormat::S32,
        CodecId::PcmF32le | CodecId::PcmF32be => SampleFormat::F32,
        CodecId::PcmF64le | CodecId::PcmF64be => SampleFormat::F64,
        _ => src,
    }
}

fn encoded_packet_contract(fmt: &str, codec_id: CodecId, extradata: Vec<u8>) -> PacketContract {
    let framing = match codec_id {
        CodecId::H264 => {
            if matches!(fmt, "mp4" | "mov" | "matroska" | "mkv") {
                PacketFraming::Avcc(4)
            } else {
                PacketFraming::AnnexB
            }
        }
        CodecId::Hevc => {
            if matches!(fmt, "mp4" | "mov" | "matroska" | "mkv") {
                PacketFraming::Avcc(4)
            } else {
                PacketFraming::AnnexB
            }
        }
        CodecId::Av1 => PacketFraming::Obu,
        _ => PacketFraming::Raw,
    };
    let mut pc = PacketContract::new(codec_id, framing);
    if !extradata.is_empty() {
        pc = pc
            .with_extradata(extradata)
            .with_extradata_mode(wedeo::graph::ExtradataMode::OutOfBand);
    }
    pc
}

fn packet_contract_for_mux(
    fmt: &str,
    track: &TrackDescriptor,
    src_contract: &PacketContract,
) -> PacketContract {
    let mut dst = src_contract.clone();
    match track.codec_id {
        CodecId::H264 | CodecId::Hevc
            if matches!(fmt, "mp4" | "mov" | "matroska" | "mkv")
                && matches!(src_contract.framing, PacketFraming::AnnexB) =>
        {
            dst.framing = PacketFraming::Avcc(4);
        }
        _ => {}
    }
    dst
}

/// Remaps `pkt.track_id` (fixing `-vn` / `-an` / multi-input stream-index mismatch),
/// optionally limits packet count, and optionally filters/shifts packet timestamps.
struct PacketTransformNode {
    name: String,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
    target_track_id: TrackId,
    max_packets: Option<u64>,
    sent_packets: u64,
    trim: Option<(wedeo::graph::ClipRange, bool)>,
}

impl PacketTransformNode {
    fn new(
        name: String,
        input: PacketContract,
        output: PacketContract,
        target_track_id: TrackId,
    ) -> Self {
        Self {
            name,
            inputs: [PortContract::Packet(input)],
            outputs: [PortContract::Packet(output)],
            target_track_id,
            max_packets: None,
            sent_packets: 0,
            trim: None,
        }
    }
}

impl GraphNode for PacketTransformNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Packet(mut pkt) = item else {
            return Err(Error::invalid_argument(
                "packet_remap",
                "expected GraphItem::Packet",
            ));
        };
        if let Some(max) = self.max_packets
            && self.sent_packets >= max
        {
            return Ok(Vec::new());
        }
        if let Some((clip, zero_based)) = self.trim {
            let ts = if pkt.pts.is_known() { pkt.pts } else { pkt.dts };
            if !ts.is_known() {
                return Ok(Vec::new());
            }
            let tb = if pkt.time_base.is_valid() {
                pkt.time_base
            } else {
                clip.time_base
            };
            if !clip.contains(ts, tb) {
                return Ok(Vec::new());
            }
            if zero_based && let Some(start_raw) = clip.start.get() {
                let start_in_tb = if clip.time_base.is_valid() && tb.is_valid() {
                    rescale_q(start_raw, clip.time_base, tb)
                } else {
                    start_raw
                };
                pkt.pts = pkt.pts.checked_sub(start_in_tb).unwrap_or(pkt.pts);
                pkt.dts = pkt.dts.checked_sub(start_in_tb).unwrap_or(pkt.dts);
            }
        }
        pkt.track_id = self.target_track_id;
        self.sent_packets += 1;
        Ok(vec![(0, GraphItem::Packet(pkt))])
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Limits the number of forwarded `VideoFrame`s (`--frames`).
struct VideoLimitNode {
    name: String,
    contracts: [PortContract; 1],
    max_frames: u64,
    sent: u64,
}

impl VideoLimitNode {
    fn new(contract: VideoContract, max_frames: u64) -> Self {
        Self {
            name: format!("limit_frames({max_frames})"),
            contracts: [PortContract::Video(contract)],
            max_frames,
            sent: 0,
        }
    }
}

impl GraphNode for VideoLimitNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.contracts
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.contracts
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        if self.sent >= self.max_frames {
            return Ok(Vec::new());
        }
        self.sent += 1;
        Ok(vec![(0, item)])
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Executes a mixed-mode `SmartCutPlan` (`ReencodeHead` + `CopyInterior` + `ReencodeTail`)
/// inside a single `wedeo-graph` node when a same-codec encoder is available.
struct SmartCutVideoNode {
    name: String,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
    target_track_id: TrackId,
    plan: wedeo::graph::SmartCutPlan,
    timeline: wedeo::graph::Timeline,
    decoder: Box<dyn Decoder>,
    encoder: Box<dyn Encoder>,
    active_seg: usize,
    force_next_key: bool,
    health: StreamHealth,
    buffered: VecDeque<Packet>,
}

impl SmartCutVideoNode {
    fn new(
        in_contract: PacketContract,
        out_contract: PacketContract,
        target_track_id: TrackId,
        plan: wedeo::graph::SmartCutPlan,
        timeline: wedeo::graph::Timeline,
        decoder: Box<dyn Decoder>,
        encoder: Box<dyn Encoder>,
    ) -> Self {
        Self {
            name: "smart_cut_video".to_string(),
            inputs: [PortContract::Packet(in_contract)],
            outputs: [PortContract::Packet(out_contract)],
            target_track_id,
            plan,
            timeline,
            decoder,
            encoder,
            active_seg: 0,
            force_next_key: true,
            health: StreamHealth::new(),
            buffered: VecDeque::new(),
        }
    }

    fn map_pts(&self, pts: Timestamp, tb: Rational) -> Timestamp {
        self.timeline
            .coded_to_presentation(wedeo::graph::CodedTime(pts), tb)
            .map_or(pts, |p| p.0)
    }

    fn flush_reencode_segment(
        &mut self,
        pts_start: Timestamp,
        pts_end: Timestamp,
        seg_tb: Rational,
        out: &mut Vec<(usize, GraphItem)>,
    ) -> Result<()> {
        let _ = self.decoder.send_packet(None)?;
        loop {
            match self.decoder.receive()? {
                RecvStatus::Ready(MediaUnit::Video(mut frame)) => {
                    let ftb = if frame.time_base.is_valid() {
                        frame.time_base
                    } else {
                        seg_tb
                    };
                    let in_range = frame.pts.get().is_some_and(|p| {
                        let p_seg = rescale_q_rnd(p, ftb, seg_tb, Rounding::NearInf, true);
                        p_seg >= pts_start.raw() && p_seg < pts_end.raw()
                    });
                    if in_range {
                        frame.pts = self.map_pts(frame.pts, ftb);
                        let hint = EncodeHint {
                            force_keyframe: self.force_next_key,
                            target_qp: None,
                        };
                        self.force_next_key = false;
                        if self
                            .encoder
                            .send_unit(Some(&MediaUnit::Video(frame.clone())), hint)?
                            == SendStatus::Full
                        {
                            self.drain_encoder(out)?;
                            let _ = self
                                .encoder
                                .send_unit(Some(&MediaUnit::Video(frame)), hint)?;
                        }
                        self.drain_encoder(out)?;
                    }
                }
                RecvStatus::Ready(_) => {}
                RecvStatus::NeedMore | RecvStatus::Eof => break,
            }
        }
        let _ = self.encoder.send_unit(None, EncodeHint::default())?;
        self.drain_encoder(out)?;
        self.decoder.flush();
        self.encoder.flush();
        self.force_next_key = true;
        Ok(())
    }

    fn drain_encoder(&mut self, out: &mut Vec<(usize, GraphItem)>) -> Result<()> {
        while let RecvStatus::Ready(mut pkt) = self.encoder.receive_packet()? {
            pkt.track_id = self.target_track_id;
            out.push((0, GraphItem::Packet(pkt)));
        }
        Ok(())
    }

    fn process_buffered(&mut self, out: &mut Vec<(usize, GraphItem)>) -> Result<()> {
        while self.active_seg < self.plan.segments.len() {
            let seg = self.plan.segments[self.active_seg].clone();
            match seg {
                wedeo::graph::GopSegment::CopyInterior {
                    pts_start,
                    pts_end,
                    time_base,
                    ..
                } => {
                    while let Some(pkt) = self.buffered.front() {
                        let ptb = if pkt.time_base.is_valid() {
                            pkt.time_base
                        } else {
                            time_base
                        };
                        let p_raw = pkt.pts.get().or_else(|| pkt.dts.get()).unwrap_or(0);
                        let p_seg = rescale_q(p_raw, ptb, time_base);
                        if p_seg < pts_start.raw() {
                            self.buffered.pop_front();
                            continue;
                        }
                        if p_seg >= pts_end.raw() {
                            self.active_seg += 1;
                            break;
                        }
                        let mut cp = self.buffered.pop_front().unwrap();
                        cp.pts = self.map_pts(cp.pts, ptb);
                        cp.dts = self.map_pts(cp.dts, ptb);
                        cp.track_id = self.target_track_id;
                        out.push((0, GraphItem::Packet(cp)));
                    }
                    if self.buffered.is_empty() {
                        break;
                    }
                }
                wedeo::graph::GopSegment::ReencodeHead {
                    decode_from,
                    pts_start,
                    pts_end,
                    time_base,
                    ..
                }
                | wedeo::graph::GopSegment::ReencodeTail {
                    decode_from,
                    pts_start,
                    pts_end,
                    time_base,
                    ..
                } => {
                    let mut finished_seg = false;
                    while let Some(pkt) = self.buffered.front() {
                        let ptb = if pkt.time_base.is_valid() {
                            pkt.time_base
                        } else {
                            time_base
                        };
                        let p_raw = pkt.pts.get().or_else(|| pkt.dts.get()).unwrap_or(0);
                        let p_seg = rescale_q(p_raw, ptb, time_base);
                        if p_seg < decode_from.raw() {
                            self.buffered.pop_front();
                            continue;
                        }
                        if p_seg >= pts_end.raw() {
                            self.flush_reencode_segment(pts_start, pts_end, time_base, out)?;
                            self.active_seg += 1;
                            finished_seg = true;
                            break;
                        }
                        let pkt = self.buffered.pop_front().unwrap();
                        let _ = self.decoder.send_packet(Some(&pkt))?;
                    }
                    if !finished_seg {
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}

impl GraphNode for SmartCutVideoNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Packet(pkt) = item else {
            return Err(Error::invalid_argument(
                "smart_cut_video",
                "expected GraphItem::Packet",
            ));
        };
        self.buffered.push_back(pkt);
        let mut out = Vec::new();
        self.process_buffered(&mut out)?;
        Ok(out)
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        let mut out = Vec::new();
        self.process_buffered(&mut out)?;
        if self.active_seg < self.plan.segments.len() {
            let seg = self.plan.segments[self.active_seg].clone();
            match seg {
                wedeo::graph::GopSegment::ReencodeHead {
                    pts_start,
                    pts_end,
                    time_base,
                    ..
                }
                | wedeo::graph::GopSegment::ReencodeTail {
                    pts_start,
                    pts_end,
                    time_base,
                    ..
                } => {
                    self.flush_reencode_segment(pts_start, pts_end, time_base, &mut out)?;
                }
                wedeo::graph::GopSegment::CopyInterior { .. } => {}
            }
            self.active_seg += 1;
        }
        self.health.merge(self.decoder.health());
        Ok(out)
    }

    fn health(&self) -> Option<&StreamHealth> {
        Some(&self.health)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
