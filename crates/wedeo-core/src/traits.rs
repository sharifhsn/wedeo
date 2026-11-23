use std::sync::Arc;

use crate::buffer::{Buffer, BufferPool};
use crate::error::{ErrorPolicy, RecvStatus, Result, SendStatus, StreamHealth};
use crate::io::{ByteSink, ByteSource};
use crate::media::{
    CodecId, ContainerHeader, MediaType, MediaUnit, Packet, SeekTarget, TrackDescriptor,
};

/// Boundary-only schema metadata for CLI/JSON option validation (library code uses typed structs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionKind {
    Int { min: i64, max: i64 },
    Float { min: f64, max: f64 },
    Bool,
    Str,
    Enum(&'static [&'static str]),
}

/// Static descriptor for one configurable option exposed at CLI/JSON boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionSchema {
    pub name: &'static str,
    pub kind: OptionKind,
    pub default: &'static str,
    pub help: &'static str,
}

/// Static descriptor for a container demuxer implementation.
#[derive(Debug, Clone, Copy)]
pub struct DemuxerDescriptor {
    pub name: &'static str,
    pub long_name: &'static str,
    pub extensions: &'static [&'static str],
    pub mime_types: &'static [&'static str],
}

/// Container demuxer owning its `ByteSource` cleanly (no borrowed-I/O `DeadIo` hacks).
pub trait Demuxer: Send {
    fn header(&self) -> &ContainerHeader;
    fn read_packet(&mut self) -> Result<RecvStatus<Packet>>;
    fn seek(&mut self, target: SeekTarget) -> Result<()>;
    fn health(&self) -> &StreamHealth;
}

/// Factory for probing and opening a container demuxer.
pub trait DemuxerFactory: Send + Sync {
    fn descriptor(&self) -> &DemuxerDescriptor;
    /// Return confidence score in `0..=100` (`0` = no match, `100` = definite match).
    fn probe(&self, head: &[u8], ext: Option<&str>) -> u8;
    fn open(&self, source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Box<dyn Demuxer>>;
    fn options(&self) -> &'static [OptionSchema] {
        &[]
    }
}

/// Static descriptor for a container muxer implementation.
#[derive(Debug, Clone, Copy)]
pub struct MuxerDescriptor {
    pub name: &'static str,
    pub long_name: &'static str,
    pub extensions: &'static [&'static str],
    pub mime_types: &'static [&'static str],
    pub default_video: CodecId,
    pub default_audio: CodecId,
}

/// Container muxer owning its `ByteSink` (supports both seekable files and non-seekable pipes).
pub trait Muxer: Send {
    fn write_header(&mut self, header: &ContainerHeader) -> Result<()>;
    fn write_packet(&mut self, pkt: &Packet) -> Result<()>;
    fn finalize(self: Box<Self>) -> Result<()>;
}

/// Factory for creating a container muxer.
pub trait MuxerFactory: Send + Sync {
    fn descriptor(&self) -> &MuxerDescriptor;
    fn create(&self, sink: Box<dyn ByteSink>) -> Result<Box<dyn Muxer>>;
    fn options(&self) -> &'static [OptionSchema] {
        &[]
    }
}

/// Shared runtime resources passed when opening a `Decoder` or `Encoder`.
#[derive(Debug, Clone)]
pub struct CodecContext {
    pub pool: Arc<BufferPool>,
    /// Worker thread budget (`0` = auto, `1` = inline/no threads spawned).
    pub threads: usize,
    pub error_policy: ErrorPolicy,
}

impl CodecContext {
    pub fn new(pool: Arc<BufferPool>) -> Self {
        Self {
            pool,
            threads: 0,
            error_policy: ErrorPolicy::Resilient,
        }
    }
}

impl Default for CodecContext {
    fn default() -> Self {
        Self::new(BufferPool::new())
    }
}

/// Static descriptor for a decoder implementation.
#[derive(Debug, Clone, Copy)]
pub struct DecoderDescriptor {
    pub codec_id: CodecId,
    pub name: &'static str,
    pub long_name: &'static str,
    pub media_type: MediaType,
}

/// Stateful media decoder (`Packet` -> `MediaUnit`).
pub trait Decoder: Send {
    /// Submit a compressed packet, or `None` to flush/drain end-of-stream.
    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<SendStatus>;
    /// Pull the next decoded `MediaUnit`.
    fn receive(&mut self) -> Result<RecvStatus<MediaUnit>>;
    /// Reset internal decoder state (e.g. after seeking).
    fn flush(&mut self);
    fn health(&self) -> &StreamHealth;
}

/// Factory for opening a `Decoder`.
pub trait DecoderFactory: Send + Sync {
    fn descriptor(&self) -> &DecoderDescriptor;
    fn open(&self, track: &TrackDescriptor, ctx: CodecContext) -> Result<Box<dyn Decoder>>;
    fn options(&self) -> &'static [OptionSchema] {
        &[]
    }
}

/// Per-frame encoder hint (keyframe forcing for smart-cut/CMAF, optional QP override).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncodeHint {
    pub force_keyframe: bool,
    pub target_qp: Option<i32>,
}

/// Rate-control pass mode for encoders supporting 2-pass encoding.
#[derive(Debug, Clone, Default)]
pub enum PassMode {
    #[default]
    Single,
    FirstPass,
    SecondPass(Buffer),
}

/// Static descriptor for an encoder implementation.
#[derive(Debug, Clone, Copy)]
pub struct EncoderDescriptor {
    pub codec_id: CodecId,
    pub name: &'static str,
    pub long_name: &'static str,
    pub media_type: MediaType,
}

/// Stateful media encoder (`MediaUnit` -> `Packet`) with first-class `extradata()` plumbing.
pub trait Encoder: Send {
    /// Codec configuration record (`avcC`, `hvcC`, `av1C`, `AudioSpecificConfig`), ready after `open`.
    fn extradata(&self) -> &[u8];
    /// Submit a decoded unit with optional hint, or `None` to drain end-of-stream.
    fn send_unit(&mut self, unit: Option<&MediaUnit>, hint: EncodeHint) -> Result<SendStatus>;
    /// Pull the next encoded `Packet`.
    fn receive_packet(&mut self) -> Result<RecvStatus<Packet>>;
    /// Retrieve collected first-pass rate-control statistics (if `PassMode::FirstPass`).
    fn take_pass_stats(&mut self) -> Option<Buffer> {
        None
    }
    /// Reset internal encoder state.
    fn flush(&mut self);
}

/// Factory for opening an `Encoder`.
pub trait EncoderFactory: Send + Sync {
    fn descriptor(&self) -> &EncoderDescriptor;
    fn open(
        &self,
        track: &TrackDescriptor,
        ctx: CodecContext,
        pass: PassMode,
    ) -> Result<Box<dyn Encoder>>;
    fn options(&self) -> &'static [OptionSchema] {
        &[]
    }
}
