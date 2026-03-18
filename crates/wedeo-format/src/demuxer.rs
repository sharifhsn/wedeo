use bitflags::bitflags;

use wedeo_codec::decoder::CodecParameters;
use wedeo_core::error::Result;
use wedeo_core::metadata::Metadata;
use wedeo_core::packet::Packet;
use wedeo_core::rational::Rational;

use crate::io::BufferedIo;

bitflags! {
    /// Input format flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct InputFormatFlags: u32 {
        const NOFILE       = 1 << 0;
        const NEEDNUMBER   = 1 << 1;
        const NOBINSEARCH  = 1 << 3;
        const NOGENSEARCH  = 1 << 4;
        const NOBYTESEEK   = 1 << 5;
        const SEEK_TO_PTS  = 1 << 26;
    }
}

bitflags! {
    /// Seek flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SeekFlags: u32 {
        const BACKWARD = 1;
        const BYTE     = 2;
        const ANY      = 4;
        const FRAME    = 8;
    }
}

/// Discard level for stream packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum Discard {
    None = -16,
    Default = 0,
    NonRef = 8,
    Bidir = 16,
    NonIntra = 24,
    NonKey = 32,
    All = 48,
}

/// Data used for format probing.
#[derive(Debug)]
pub struct ProbeData<'a> {
    pub filename: &'a str,
    pub buf: &'a [u8],
}

/// Probe score thresholds.
pub const PROBE_SCORE_EXTENSION: i32 = 50;
pub const PROBE_SCORE_MAX: i32 = 100;

/// A stream within a demuxed file.
#[derive(Debug, Clone)]
pub struct Stream {
    pub index: usize,
    pub codec_params: CodecParameters,
    pub time_base: Rational,
    pub duration: i64,
    pub nb_frames: i64,
    pub metadata: Metadata,
    pub discard: Discard,
}

impl Stream {
    pub fn new(index: usize, codec_params: CodecParameters) -> Self {
        Self {
            index,
            codec_params,
            time_base: Rational::new(0, 1),
            duration: 0,
            nb_frames: 0,
            metadata: Metadata::new(),
            discard: Discard::Default,
        }
    }
}

/// Result of reading a demuxer header.
pub struct DemuxerHeader {
    pub streams: Vec<Stream>,
    pub metadata: Metadata,
    pub duration: i64,
    pub start_time: i64,
}

/// Demuxer trait — the main abstraction for all demuxers.
pub trait Demuxer: Send {
    /// Read the file header, returning streams and metadata.
    fn read_header(&mut self, io: &mut BufferedIo) -> Result<DemuxerHeader>;

    /// Read the next packet from the file.
    fn read_packet(&mut self, io: &mut BufferedIo) -> Result<Packet>;

    /// Seek to a timestamp.
    fn seek(
        &mut self,
        io: &mut BufferedIo,
        stream_index: usize,
        timestamp: i64,
        flags: SeekFlags,
    ) -> Result<()>;
}

/// Descriptor for an input format.
#[derive(Debug, Clone)]
pub struct InputFormatDescriptor {
    pub name: &'static str,
    pub long_name: &'static str,
    pub extensions: &'static str,
    pub mime_types: &'static str,
    pub flags: InputFormatFlags,
    /// Priority for probe tie-breaking. Higher priority wins when multiple
    /// demuxers return the same probe score. Native implementations use 100,
    /// wrapper/adapter implementations use 50.
    pub priority: i32,
}
