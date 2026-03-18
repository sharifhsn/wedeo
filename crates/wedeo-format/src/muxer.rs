use bitflags::bitflags;

use wedeo_core::codec_id::CodecId;
use wedeo_core::error::Result;
use wedeo_core::packet::Packet;

use crate::demuxer::Stream;
use crate::io::BufferedIo;

bitflags! {
    /// Output format flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct OutputFormatFlags: u32 {
        const NOFILE        = 1 << 0;
        const NEEDNUMBER    = 1 << 1;
        const GLOBALHEADER  = 1 << 6;
        const NOTIMESTAMPS  = 1 << 7;
        const VARIABLE_FPS  = 1 << 10;
        const NODIMENSIONS  = 1 << 11;
        const NOSTREAMS     = 1 << 12;
        const TS_NONSTRICT  = 1 << 17;
        const TS_NEGATIVE   = 1 << 18;
    }
}

/// Muxer trait — the main abstraction for all muxers.
pub trait Muxer: Send {
    /// Write the file header.
    fn write_header(&mut self, io: &mut BufferedIo, streams: &[Stream]) -> Result<()>;

    /// Write a single packet.
    fn write_packet(&mut self, io: &mut BufferedIo, packet: &Packet) -> Result<()>;

    /// Write the file trailer (finalize).
    fn write_trailer(&mut self, io: &mut BufferedIo) -> Result<()>;
}

/// Descriptor for an output format.
#[derive(Debug, Clone)]
pub struct OutputFormatDescriptor {
    pub name: &'static str,
    pub long_name: &'static str,
    pub extensions: &'static str,
    pub mime_types: &'static str,
    pub flags: OutputFormatFlags,
    pub audio_codec: CodecId,
    pub video_codec: CodecId,
}
