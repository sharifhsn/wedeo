// IVF container demuxer for wedeo.
//
// IVF is a simple raw video container used by the VP8, VP9, and AV1 codecs.
// Format reference: https://wiki.multimedia.cx/index.php/IVF
//
// File structure:
//   [32-byte file header]
//   [12-byte frame header] [frame_size bytes of frame data]
//   [12-byte frame header] [frame_size bytes of frame data]
//   ...
//
// File header layout (all little-endian):
//   bytes  0-3 : magic "DKIF"
//   bytes  4-5 : version (u16le) — must be 0
//   bytes  6-7 : header_len (u16le) — must be 32
//   bytes  8-11: fourcc (4 ASCII bytes, e.g. "VP80", "VP90", "AV01")
//   bytes 12-13: width (u16le)
//   bytes 14-15: height (u16le)
//   bytes 16-19: fps_num (u32le)
//   bytes 20-23: fps_den (u32le)
//   bytes 24-27: num_frames (u32le)
//   bytes 28-31: unused (u32le)
//
// Frame header layout (all little-endian):
//   bytes  0-3 : frame_size (u32le)
//   bytes  4-11: pts (u64le)

use tracing::{debug, trace};
use wedeo_codec::decoder::CodecParameters;
use wedeo_core::buffer::Buffer;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::media_type::MediaType;
use wedeo_core::metadata::Metadata;
use wedeo_core::packet::{Packet, PacketFlags};
use wedeo_core::pixel_format::PixelFormat;
use wedeo_core::rational::Rational;
use wedeo_format::demuxer::{
    Demuxer, DemuxerHeader, InputFormatDescriptor, InputFormatFlags, PROBE_SCORE_MAX, ProbeData,
    SeekFlags, Stream,
};
use wedeo_format::io::BufferedIo;
use wedeo_format::registry::DemuxerFactory;

// ---------------------------------------------------------------------------
// IVF file header constants
// ---------------------------------------------------------------------------

const IVF_MAGIC: &[u8; 4] = b"DKIF";
const IVF_FILE_HEADER_LEN: usize = 32;
const IVF_FRAME_HEADER_LEN: usize = 12;

// ---------------------------------------------------------------------------
// IVF demuxer state
// ---------------------------------------------------------------------------

struct IvfDemuxer {
    /// Frame counter (used for DTS when PTS is unavailable).
    frame_count: i64,
    /// Whether the file header has been read.
    header_read: bool,
}

impl IvfDemuxer {
    fn new() -> Self {
        Self {
            frame_count: 0,
            header_read: false,
        }
    }
}

/// Map a 4-byte IVF fourcc to a `CodecId`.
///
/// Known mappings (case-insensitive in the spec; we match exact bytes):
///   "VP80" → Vp8
///   "VP90" → Vp9
///   "AV01" → Av1
fn fourcc_to_codec_id(fourcc: &[u8; 4]) -> Option<CodecId> {
    match fourcc {
        b"VP80" => Some(CodecId::Vp8),
        b"VP90" => Some(CodecId::Vp9),
        b"AV01" => Some(CodecId::Av1),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Demuxer trait implementation
// ---------------------------------------------------------------------------

impl Demuxer for IvfDemuxer {
    fn read_header(&mut self, io: &mut BufferedIo) -> Result<DemuxerHeader> {
        // Read the 32-byte file header.
        let mut hdr = [0u8; IVF_FILE_HEADER_LEN];
        io.read_exact(&mut hdr)?;

        // Validate magic.
        if &hdr[0..4] != IVF_MAGIC {
            return Err(Error::InvalidData);
        }

        // Parse fields.
        let version = u16::from_le_bytes([hdr[4], hdr[5]]);
        let header_len = u16::from_le_bytes([hdr[6], hdr[7]]) as usize;
        let fourcc: [u8; 4] = [hdr[8], hdr[9], hdr[10], hdr[11]];
        let width = u16::from_le_bytes([hdr[12], hdr[13]]) as u32;
        let height = u16::from_le_bytes([hdr[14], hdr[15]]) as u32;
        let fps_num = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
        let fps_den = u32::from_le_bytes([hdr[20], hdr[21], hdr[22], hdr[23]]);

        debug!(
            version,
            header_len,
            ?fourcc,
            width,
            height,
            fps_num,
            fps_den,
            "IVF file header"
        );

        // Skip extra header bytes if header_len > 32.
        if header_len > IVF_FILE_HEADER_LEN {
            let extra = header_len - IVF_FILE_HEADER_LEN;
            let _ = io.read_up_to(extra)?;
        }

        // Validate version.
        if version != 0 {
            return Err(Error::InvalidData);
        }

        // Map fourcc → CodecId.
        let codec_id = fourcc_to_codec_id(&fourcc).unwrap_or(CodecId::None);

        // Build codec parameters.
        let mut params = CodecParameters::new(codec_id, MediaType::Video);
        params.width = width;
        params.height = height;
        params.pixel_format = PixelFormat::Yuv420p; // default; decoder will refine

        // Compute time base from fps.
        let time_base = if fps_num > 0 && fps_den > 0 {
            Rational::new(fps_den as i32, fps_num as i32)
        } else {
            Rational::new(1, 30)
        };

        let mut stream = Stream::new(0, params);
        stream.time_base = time_base;

        self.header_read = true;

        Ok(DemuxerHeader {
            streams: vec![stream],
            metadata: Metadata::new(),
            duration: 0,
            start_time: 0,
        })
    }

    fn read_packet(&mut self, io: &mut BufferedIo) -> Result<Packet> {
        if !self.header_read {
            return Err(Error::InvalidData);
        }

        // Read the 12-byte frame header.
        let mut fhdr = [0u8; IVF_FRAME_HEADER_LEN];
        match io.read_exact(&mut fhdr) {
            Ok(()) => {}
            Err(Error::Eof) => return Err(Error::Eof),
            Err(e) => return Err(e),
        }

        let frame_size = u32::from_le_bytes([fhdr[0], fhdr[1], fhdr[2], fhdr[3]]) as usize;
        let pts = i64::from_le_bytes([
            fhdr[4], fhdr[5], fhdr[6], fhdr[7], fhdr[8], fhdr[9], fhdr[10], fhdr[11],
        ]);

        trace!(
            frame = self.frame_count,
            frame_size, pts, "IVF frame header"
        );

        if frame_size == 0 {
            return Err(Error::InvalidData);
        }

        // Read the frame payload.
        let frame_data = io.read_up_to(frame_size)?;
        if frame_data.is_empty() {
            return Err(Error::Eof);
        }
        if frame_data.len() < frame_size {
            // Truncated frame — return what we have.
        }

        let mut pkt = Packet::new();
        pkt.data = Buffer::from_slice(&frame_data);
        pkt.pts = pts;
        pkt.dts = pts;
        pkt.duration = 1;
        pkt.stream_index = 0;

        // IVF doesn't have a keyframe flag; mark frame 0 as keyframe,
        // and for VP9 the decoder will refine this from the frame header.
        if self.frame_count == 0 {
            pkt.flags = PacketFlags::KEY;
        }

        self.frame_count += 1;
        Ok(pkt)
    }

    fn seek(
        &mut self,
        _io: &mut BufferedIo,
        _stream_index: usize,
        _timestamp: i64,
        _flags: SeekFlags,
    ) -> Result<()> {
        Err(Error::PatchwelcomeNotImplemented)
    }
}

// ---------------------------------------------------------------------------
// DemuxerFactory
// ---------------------------------------------------------------------------

struct IvfDemuxerFactory;

impl DemuxerFactory for IvfDemuxerFactory {
    fn descriptor(&self) -> &InputFormatDescriptor {
        static DESC: InputFormatDescriptor = InputFormatDescriptor {
            name: "ivf",
            long_name: "IVF (On2/Google VP8/VP9 Container)",
            extensions: "ivf",
            mime_types: "video/x-ivf",
            flags: InputFormatFlags::empty(),
            priority: 100,
        };
        &DESC
    }

    fn probe(&self, data: &ProbeData<'_>) -> i32 {
        let buf = data.buf;
        if buf.len() >= 4 && &buf[0..4] == IVF_MAGIC {
            PROBE_SCORE_MAX
        } else {
            0
        }
    }

    fn create(&self) -> Result<Box<dyn Demuxer>> {
        Ok(Box::new(IvfDemuxer::new()))
    }
}

inventory::submit!(&IvfDemuxerFactory as &dyn DemuxerFactory);
