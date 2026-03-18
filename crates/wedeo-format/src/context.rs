use tracing::debug;
use wedeo_core::error::{Error, Result};
use wedeo_core::metadata::Metadata;
use wedeo_core::packet::Packet;

use crate::demuxer::{Demuxer, ProbeData, SeekFlags, Stream};
use crate::io::{BufferedIo, FileIo};
use crate::muxer::Muxer;
use crate::registry;

/// High-level input context — wraps demuxer + I/O for easy demuxing.
pub struct InputContext {
    demuxer: Box<dyn Demuxer>,
    io: BufferedIo,
    pub streams: Vec<Stream>,
    pub metadata: Metadata,
    pub duration: i64,
    pub start_time: i64,
}

impl InputContext {
    /// Open a file and auto-detect the format.
    pub fn open(path: &str) -> Result<Self> {
        let file_io = FileIo::open(path)?;
        let mut io = BufferedIo::new(Box::new(file_io));

        // Read probe data (up to 4096 bytes, file may be smaller)
        let file_size = io.size().unwrap_or(4096);
        let probe_size = file_size.min(4096) as usize;
        let probe_buf = io.read_bytes(probe_size)?;
        io.seek(0)?;

        let probe_data = ProbeData {
            filename: path,
            buf: &probe_buf,
        };

        let factory = registry::probe(&probe_data).ok_or(Error::DemuxerNotFound)?;

        let mut demuxer = factory.create()?;
        let header = demuxer.read_header(&mut io)?;

        let format_name = factory.descriptor().name;
        debug!(
            path,
            format = format_name,
            streams = header.streams.len(),
            "InputContext opened"
        );

        Ok(Self {
            demuxer,
            io,
            streams: header.streams,
            metadata: header.metadata,
            duration: header.duration,
            start_time: header.start_time,
        })
    }

    /// Read the next packet.
    pub fn read_packet(&mut self) -> Result<Packet> {
        self.demuxer.read_packet(&mut self.io)
    }

    /// Seek to a timestamp in the specified stream.
    pub fn seek(&mut self, stream_index: usize, timestamp: i64, flags: SeekFlags) -> Result<()> {
        self.demuxer
            .seek(&mut self.io, stream_index, timestamp, flags)
    }

    /// Get a stream by index.
    pub fn stream(&self, index: usize) -> Option<&Stream> {
        self.streams.get(index)
    }

    /// Number of streams.
    pub fn nb_streams(&self) -> usize {
        self.streams.len()
    }
}

/// High-level output context — wraps muxer + I/O for easy muxing.
pub struct OutputContext {
    muxer: Box<dyn Muxer>,
    io: BufferedIo,
}

impl OutputContext {
    /// Create an output file with the specified format.
    pub fn create(path: &str, format_name: &str, streams: &[Stream]) -> Result<Self> {
        let factory = registry::find_muxer_by_name(format_name).ok_or(Error::MuxerNotFound)?;
        let mut muxer = factory.create()?;
        let file_io = FileIo::create(path)?;
        let mut io = BufferedIo::new(Box::new(file_io));
        muxer.write_header(&mut io, streams)?;
        Ok(Self { muxer, io })
    }

    /// Write a single packet.
    pub fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        self.muxer.write_packet(&mut self.io, packet)
    }

    /// Finalize the output: write trailer and flush.
    pub fn finish(mut self) -> Result<()> {
        self.muxer.write_trailer(&mut self.io)?;
        self.io.flush()
    }
}
