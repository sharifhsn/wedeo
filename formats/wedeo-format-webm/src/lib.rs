// WebM/Matroska container demuxer for wedeo.
//
// WebM is a subset of the Matroska container format. This demuxer handles
// video-only demuxing — audio tracks are silently skipped.
//
// Format reference: https://www.matroska.org/technical/elements.html
// EBML spec: https://matroska-org.github.io/libebml/specs.html
//
// File structure (EBML):
//   [EBML Header]
//   [Segment]
//     [SeekHead] [Info] [Tracks] [Cluster [SimpleBlock]* ]* [Cues] ...

use tracing::{debug, trace, warn};
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
// EBML element IDs
// ---------------------------------------------------------------------------

// EBML Header
const EBML_HEADER: u32 = 0x1A45_DFA3;
const EBML_DOCTYPE: u32 = 0x4282;
const EBML_DOC_TYPE_READ_VERSION: u32 = 0x4285;
const EBML_VERSION: u32 = 0x4286;
const EBML_READ_VERSION: u32 = 0x42F7;
const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;

// Segment
const SEGMENT: u32 = 0x1853_8067;

// Info
const INFO: u32 = 0x1549_A966;
const TIMECODE_SCALE: u32 = 0x2AD7B1;
const DURATION: u32 = 0x4489;

// Tracks
const TRACKS: u32 = 0x1654_AE6B;
const TRACK_ENTRY: u32 = 0xAE;
const TRACK_NUMBER: u32 = 0xD7;
const TRACK_TYPE: u32 = 0x83;
const CODEC_ID: u32 = 0x86;
const CODEC_PRIVATE: u32 = 0x63A2;
const DEFAULT_DURATION: u32 = 0x0023_E383;

// Video
const VIDEO: u32 = 0xE0;
const PIXEL_WIDTH: u32 = 0xB0;
const PIXEL_HEIGHT: u32 = 0xBA;

// Cluster
const CLUSTER: u32 = 0x1F43_B675;
const CLUSTER_TIMECODE: u32 = 0xE7;
const SIMPLE_BLOCK: u32 = 0xA3;

// EBML global elements (can appear anywhere)
const EBML_VOID: u32 = 0xEC;
const EBML_CRC32: u32 = 0xBF;

// Other top-level elements to skip
const SEEK_HEAD: u32 = 0x114D_9B74;
const CUES: u32 = 0x1C53_BB6B;
const CHAPTERS: u32 = 0x1043_A770;
const TAGS: u32 = 0x1254_C367;
const ATTACHMENTS: u32 = 0x1941_A469;

// Matroska track types
const TRACK_TYPE_VIDEO: u8 = 1;

// ---------------------------------------------------------------------------
// EBML variable-length integer reading
// ---------------------------------------------------------------------------

/// Read an EBML element ID. The leading-zero pattern is part of the ID value.
/// Returns (id, bytes_consumed).
fn read_element_id(io: &mut BufferedIo) -> Result<u32> {
    let mut first = [0u8; 1];
    io.read_exact(&mut first)?;
    let b0 = first[0];

    if b0 == 0 {
        return Err(Error::InvalidData);
    }

    if b0 & 0x80 != 0 {
        // 1-byte ID
        Ok(b0 as u32)
    } else if b0 & 0x40 != 0 {
        // 2-byte ID
        let mut buf = [0u8; 1];
        io.read_exact(&mut buf)?;
        Ok(((b0 as u32) << 8) | buf[0] as u32)
    } else if b0 & 0x20 != 0 {
        // 3-byte ID
        let mut buf = [0u8; 2];
        io.read_exact(&mut buf)?;
        Ok(((b0 as u32) << 16) | ((buf[0] as u32) << 8) | buf[1] as u32)
    } else if b0 & 0x10 != 0 {
        // 4-byte ID
        let mut buf = [0u8; 3];
        io.read_exact(&mut buf)?;
        Ok(((b0 as u32) << 24) | ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | buf[2] as u32)
    } else {
        // IDs longer than 4 bytes are not used in WebM/Matroska in practice.
        Err(Error::InvalidData)
    }
}

/// Size value indicating "unknown size" — the element continues until a
/// same-level or higher-level element is encountered.
const SIZE_UNKNOWN: u64 = u64::MAX;

/// Read an EBML variable-length size. The leading bits are masked off to get
/// the numeric value. An all-ones payload means "unknown size".
fn read_vint_size(io: &mut BufferedIo) -> Result<u64> {
    let mut first = [0u8; 1];
    io.read_exact(&mut first)?;
    let b0 = first[0];

    if b0 == 0 {
        return Err(Error::InvalidData);
    }

    // Determine width from leading zeros.
    let width = b0.leading_zeros() + 1; // 1..=8
    if width > 8 {
        return Err(Error::InvalidData);
    }

    // Mask off the leading 1-bit marker.
    let mask = 0xFFu8.checked_shr(width).unwrap_or(0);
    let mut value = (b0 & mask) as u64;

    // Read remaining bytes.
    let extra = (width - 1) as usize;
    if extra > 0 {
        let mut buf = [0u8; 7];
        io.read_exact(&mut buf[..extra])?;
        for &b in &buf[..extra] {
            value = (value << 8) | b as u64;
        }
    }

    // Check for "unknown size" (all data bits set to 1).
    let all_ones = (1u64 << (7 * width)) - 1;
    if value == all_ones {
        return Ok(SIZE_UNKNOWN);
    }

    Ok(value)
}

/// Read a VINT used for track number in SimpleBlock (uses size encoding,
/// i.e. leading bits are masked off).
fn read_vint_track_number(data: &[u8]) -> Result<(u64, usize)> {
    if data.is_empty() {
        return Err(Error::InvalidData);
    }
    let b0 = data[0];
    if b0 == 0 {
        return Err(Error::InvalidData);
    }

    let width = (b0.leading_zeros() + 1) as usize;
    if width > 8 || width > data.len() {
        return Err(Error::InvalidData);
    }

    let mask = 0xFFu8.checked_shr(width as u32).unwrap_or(0);
    let mut value = (b0 & mask) as u64;
    for &b in &data[1..width] {
        value = (value << 8) | b as u64;
    }

    Ok((value, width))
}

// ---------------------------------------------------------------------------
// EBML element helper: read an unsigned integer element body
// ---------------------------------------------------------------------------

fn read_uint(io: &mut BufferedIo, size: u64) -> Result<u64> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidData);
    }
    let mut buf = [0u8; 8];
    let n = size as usize;
    io.read_exact(&mut buf[..n])?;
    let mut val = 0u64;
    for &b in &buf[..n] {
        val = (val << 8) | b as u64;
    }
    Ok(val)
}

/// Read an EBML float element (4 or 8 bytes).
fn read_float(io: &mut BufferedIo, size: u64) -> Result<f64> {
    match size {
        4 => {
            let mut buf = [0u8; 4];
            io.read_exact(&mut buf)?;
            Ok(f32::from_be_bytes(buf) as f64)
        }
        8 => {
            let mut buf = [0u8; 8];
            io.read_exact(&mut buf)?;
            Ok(f64::from_be_bytes(buf))
        }
        _ => Err(Error::InvalidData),
    }
}

/// Read an EBML string (ASCII) element body.
fn read_string(io: &mut BufferedIo, size: u64) -> Result<String> {
    if size > 1024 {
        return Err(Error::InvalidData);
    }
    let expected = size as usize;
    let data = io.read_up_to(expected)?;
    if data.len() != expected {
        return Err(Error::Eof);
    }
    // Strip trailing NULs (Matroska allows NUL-padded strings).
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    String::from_utf8(data[..end].to_vec()).map_err(|_| Error::InvalidData)
}

/// Skip `size` bytes in the stream.
fn skip_element(io: &mut BufferedIo, size: u64) -> Result<()> {
    if size == SIZE_UNKNOWN {
        // Cannot skip an unknown-size element by byte count; caller must handle.
        return Err(Error::InvalidData);
    }
    // Read and discard in chunks. Use u64 to avoid truncation on 32-bit.
    let mut remaining = size;
    while remaining > 0 {
        let chunk = remaining.min(65536) as usize;
        let read = io.read_up_to(chunk)?;
        if read.is_empty() {
            return Err(Error::Eof);
        }
        remaining -= read.len() as u64;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Codec ID mapping
// ---------------------------------------------------------------------------

fn codec_id_from_matroska(s: &str) -> CodecId {
    match s {
        "V_VP9" => CodecId::Vp9,
        "V_VP8" => CodecId::Vp8,
        "V_AV1" => CodecId::Av1,
        _ => CodecId::None,
    }
}

// ---------------------------------------------------------------------------
// Track info parsed from the Tracks element
// ---------------------------------------------------------------------------

struct TrackInfo {
    /// Matroska track number (used in SimpleBlock headers).
    number: u64,
    /// Track type (1 = video, 2 = audio, etc.).
    /// Only video tracks (type 1) are stored; kept for debugging.
    _track_type: u8,
    /// Mapped codec ID.
    codec_id: CodecId,
    /// Video pixel width.
    width: u32,
    /// Video pixel height.
    height: u32,
    /// Default frame duration in nanoseconds (0 if not present).
    default_duration_ns: u64,
    /// CodecPrivate data (e.g. SPS/PPS for H.264).
    codec_private: Vec<u8>,
    /// Index into the output streams vector.
    stream_index: usize,
}

// ---------------------------------------------------------------------------
// WebM demuxer state
// ---------------------------------------------------------------------------

struct WebmDemuxer {
    /// Parsed video tracks.
    tracks: Vec<TrackInfo>,
    /// TimecodeScale from the Info element (nanoseconds per timestamp unit).
    /// Default is 1_000_000 (= 1 ms per unit).
    timecode_scale_ns: u64,
    /// Current cluster's base timecode.
    cluster_timecode: i64,
    /// Whether we are currently positioned inside a Cluster element.
    in_cluster: bool,
    /// Byte position of the end of the current cluster (None if unknown size).
    cluster_end: Option<u64>,
    /// Byte position of the end of the segment (None if unknown size).
    segment_end: Option<u64>,
}

impl WebmDemuxer {
    fn new() -> Self {
        Self {
            tracks: Vec::new(),
            timecode_scale_ns: 1_000_000,
            cluster_timecode: 0,
            in_cluster: false,
            cluster_end: None,
            segment_end: None,
        }
    }

    /// Find the output stream index for a given Matroska track number.
    /// Returns None if the track is not a video track we care about.
    fn find_stream_index(&self, track_number: u64) -> Option<usize> {
        self.tracks
            .iter()
            .find(|t| t.number == track_number)
            .map(|t| t.stream_index)
    }

    // -----------------------------------------------------------------------
    // Header parsing helpers
    // -----------------------------------------------------------------------

    /// Parse the EBML header and validate DocType.
    fn parse_ebml_header(&mut self, io: &mut BufferedIo) -> Result<()> {
        let id = read_element_id(io)?;
        if id != EBML_HEADER {
            return Err(Error::InvalidData);
        }
        let header_size = read_vint_size(io)?;
        if header_size == SIZE_UNKNOWN {
            return Err(Error::InvalidData);
        }

        let header_end = io.tell()? + header_size;
        let mut doc_type = String::new();

        while io.tell()? < header_end {
            let child_id = read_element_id(io)?;
            let child_size = read_vint_size(io)?;

            match child_id {
                EBML_DOCTYPE => {
                    doc_type = read_string(io, child_size)?;
                    debug!(doc_type = %doc_type, "EBML DocType");
                }
                EBML_VERSION
                | EBML_READ_VERSION
                | EBML_DOC_TYPE_READ_VERSION
                | EBML_MAX_ID_LENGTH
                | EBML_MAX_SIZE_LENGTH => {
                    // Read and discard — we don't enforce version constraints.
                    skip_element(io, child_size)?;
                }
                _ => {
                    trace!(
                        id = child_id,
                        size = child_size,
                        "skipping unknown EBML header child"
                    );
                    skip_element(io, child_size)?;
                }
            }
        }

        if doc_type != "webm" && doc_type != "matroska" {
            warn!(doc_type = %doc_type, "unsupported EBML DocType");
            return Err(Error::InvalidData);
        }

        Ok(())
    }

    /// Parse the Info element to extract TimecodeScale.
    fn parse_info(&mut self, io: &mut BufferedIo, size: u64) -> Result<()> {
        let end = io.tell()? + size;

        while io.tell()? < end {
            let child_id = read_element_id(io)?;
            let child_size = read_vint_size(io)?;

            match child_id {
                TIMECODE_SCALE => {
                    self.timecode_scale_ns = read_uint(io, child_size)?;
                    debug!(timecode_scale_ns = self.timecode_scale_ns, "TimecodeScale");
                }
                DURATION => {
                    let dur = read_float(io, child_size)?;
                    debug!(duration_ms = dur, "Segment duration");
                }
                _ => {
                    skip_element(io, child_size)?;
                }
            }
        }

        Ok(())
    }

    /// Parse the Video sub-element of a TrackEntry.
    fn parse_video_element(
        io: &mut BufferedIo,
        size: u64,
        width: &mut u32,
        height: &mut u32,
    ) -> Result<()> {
        let end = io.tell()? + size;

        while io.tell()? < end {
            let child_id = read_element_id(io)?;
            let child_size = read_vint_size(io)?;

            match child_id {
                PIXEL_WIDTH => {
                    *width = read_uint(io, child_size)? as u32;
                }
                PIXEL_HEIGHT => {
                    *height = read_uint(io, child_size)? as u32;
                }
                _ => {
                    skip_element(io, child_size)?;
                }
            }
        }

        Ok(())
    }

    /// Parse a single TrackEntry element.
    fn parse_track_entry(&mut self, io: &mut BufferedIo, size: u64) -> Result<()> {
        let end = io.tell()? + size;

        let mut track_number: u64 = 0;
        let mut track_type: u8 = 0;
        let mut codec_string = String::new();
        let mut codec_private: Vec<u8> = Vec::new();
        let mut width: u32 = 0;
        let mut height: u32 = 0;
        let mut default_duration_ns: u64 = 0;

        while io.tell()? < end {
            let child_id = read_element_id(io)?;
            let child_size = read_vint_size(io)?;

            match child_id {
                TRACK_NUMBER => {
                    track_number = read_uint(io, child_size)?;
                }
                TRACK_TYPE => {
                    track_type = read_uint(io, child_size)? as u8;
                }
                CODEC_ID => {
                    codec_string = read_string(io, child_size)?;
                }
                CODEC_PRIVATE => {
                    if child_size > 10 * 1024 * 1024 {
                        return Err(Error::InvalidData);
                    }
                    let expected = child_size as usize;
                    codec_private = io.read_up_to(expected)?;
                    if codec_private.len() != expected {
                        return Err(Error::Eof);
                    }
                }
                DEFAULT_DURATION => {
                    default_duration_ns = read_uint(io, child_size)?;
                }
                VIDEO => {
                    Self::parse_video_element(io, child_size, &mut width, &mut height)?;
                }
                _ => {
                    skip_element(io, child_size)?;
                }
            }
        }

        // Only keep video tracks.
        if track_type != TRACK_TYPE_VIDEO {
            trace!(track_number, track_type, "skipping non-video track");
            return Ok(());
        }

        let codec_id = codec_id_from_matroska(&codec_string);
        let stream_index = self.tracks.len();

        debug!(
            track_number,
            codec = %codec_string,
            width,
            height,
            default_duration_ns,
            stream_index,
            "video track"
        );

        self.tracks.push(TrackInfo {
            number: track_number,
            _track_type: track_type,
            codec_id,
            width,
            height,
            default_duration_ns,
            codec_private,
            stream_index,
        });

        Ok(())
    }

    /// Parse the Tracks element.
    fn parse_tracks(&mut self, io: &mut BufferedIo, size: u64) -> Result<()> {
        let end = io.tell()? + size;

        while io.tell()? < end {
            let child_id = read_element_id(io)?;
            let child_size = read_vint_size(io)?;

            if child_id == TRACK_ENTRY {
                self.parse_track_entry(io, child_size)?;
            } else {
                skip_element(io, child_size)?;
            }
        }

        Ok(())
    }

    /// Enter a cluster: parse the Cluster element header and extract the
    /// cluster timecode. After this, the IO position is at the first child
    /// element after the timecode (or at the start of child elements if no
    /// timecode comes first).
    fn enter_cluster(&mut self, io: &mut BufferedIo, size: u64) -> Result<()> {
        self.in_cluster = true;
        if size != SIZE_UNKNOWN {
            self.cluster_end = Some(io.tell()? + size);
        } else {
            self.cluster_end = None;
        }

        // The first child of a Cluster is typically the Timecode element,
        // but we'll handle it in read_packet's main loop.
        Ok(())
    }

    /// Check whether we've reached the end of the current cluster.
    fn past_cluster_end(&self, pos: u64) -> bool {
        if let Some(end) = self.cluster_end {
            pos >= end
        } else {
            false
        }
    }

    /// Check whether we've reached the end of the segment.
    fn past_segment_end(&self, pos: u64) -> bool {
        if let Some(end) = self.segment_end {
            pos >= end
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Demuxer trait implementation
// ---------------------------------------------------------------------------

impl Demuxer for WebmDemuxer {
    fn read_header(&mut self, io: &mut BufferedIo) -> Result<DemuxerHeader> {
        // 1. Parse the EBML header.
        self.parse_ebml_header(io)?;

        // 2. Read the Segment element.
        let seg_id = read_element_id(io)?;
        if seg_id != SEGMENT {
            return Err(Error::InvalidData);
        }
        let seg_size = read_vint_size(io)?;
        let seg_data_start = io.tell()?;
        if seg_size != SIZE_UNKNOWN {
            self.segment_end = Some(seg_data_start + seg_size);
        }

        // 3. Scan Segment children until we hit the first Cluster.
        loop {
            let pos = io.tell()?;
            if self.past_segment_end(pos) {
                break;
            }

            let child_id = match read_element_id(io) {
                Ok(id) => id,
                Err(Error::Eof) => break,
                Err(e) => return Err(e),
            };
            let child_size = read_vint_size(io)?;

            match child_id {
                INFO => {
                    if child_size == SIZE_UNKNOWN {
                        return Err(Error::InvalidData);
                    }
                    self.parse_info(io, child_size)?;
                }
                TRACKS => {
                    if child_size == SIZE_UNKNOWN {
                        return Err(Error::InvalidData);
                    }
                    self.parse_tracks(io, child_size)?;
                }
                CLUSTER => {
                    // We've reached the first cluster. Enter it but don't
                    // consume its children yet — read_packet will do that.
                    self.enter_cluster(io, child_size)?;
                    break;
                }
                SEEK_HEAD | CUES | CHAPTERS | TAGS | ATTACHMENTS => {
                    trace!(
                        id = child_id,
                        size = child_size,
                        "skipping top-level element"
                    );
                    if child_size == SIZE_UNKNOWN {
                        return Err(Error::InvalidData);
                    }
                    skip_element(io, child_size)?;
                }
                _ => {
                    trace!(
                        id = child_id,
                        size = child_size,
                        "skipping unknown segment child"
                    );
                    if child_size == SIZE_UNKNOWN {
                        return Err(Error::InvalidData);
                    }
                    skip_element(io, child_size)?;
                }
            }
        }

        if self.tracks.is_empty() {
            warn!("no video tracks found");
            return Err(Error::InvalidData);
        }

        if self.timecode_scale_ns == 0 {
            warn!("TimecodeScale is 0, invalid");
            return Err(Error::InvalidData);
        }

        // 4. Build output streams.
        let mut streams = Vec::with_capacity(self.tracks.len());
        for track in &self.tracks {
            let mut params = CodecParameters::new(track.codec_id, MediaType::Video);
            params.width = track.width;
            params.height = track.height;
            params.pixel_format = PixelFormat::Yuv420p; // default; decoder will refine
            if !track.codec_private.is_empty() {
                params.extradata = track.codec_private.clone();
            }

            let mut stream = Stream::new(track.stream_index, params);

            // Time base: TimecodeScale is in nanoseconds per timestamp unit.
            // With default 1_000_000 ns/unit, timestamps are in milliseconds,
            // so time_base = 1/1000.
            //
            // General formula: time_base = timecode_scale_ns / 1_000_000_000
            // We reduce to avoid overflow. For the common case (1_000_000):
            //   time_base = Rational(1, 1000)
            let tb_num = self.timecode_scale_ns as i64;
            let tb_den = 1_000_000_000i64;
            let g = gcd(tb_num, tb_den);
            let reduced_num = tb_num / g;
            let reduced_den = tb_den / g;
            // Guard against values that don't fit in i32 (would require a
            // TimecodeScale > ~2.1 billion which is extremely unusual).
            if reduced_num > i32::MAX as i64 || reduced_den > i32::MAX as i64 {
                return Err(Error::InvalidData);
            }
            stream.time_base = Rational::new(reduced_num as i32, reduced_den as i32);

            streams.push(stream);
        }

        debug!(
            num_tracks = streams.len(),
            timecode_scale_ns = self.timecode_scale_ns,
            "WebM header parsed"
        );

        Ok(DemuxerHeader {
            streams,
            metadata: Metadata::new(),
            duration: 0,
            start_time: 0,
        })
    }

    fn read_packet(&mut self, io: &mut BufferedIo) -> Result<Packet> {
        loop {
            let pos = io.tell()?;

            // Check if we've left the current cluster.
            if self.in_cluster && self.past_cluster_end(pos) {
                self.in_cluster = false;
            }

            // Check if we've left the segment entirely.
            if self.past_segment_end(pos) {
                return Err(Error::Eof);
            }

            // Read the next EBML element.
            let elem_id = match read_element_id(io) {
                Ok(id) => id,
                Err(Error::Eof) => return Err(Error::Eof),
                Err(e) => return Err(e),
            };
            let elem_size = read_vint_size(io)?;

            match elem_id {
                CLUSTER => {
                    // New cluster — enter it and continue to read children.
                    self.enter_cluster(io, elem_size)?;
                    trace!("entered new cluster");
                    continue;
                }
                CLUSTER_TIMECODE => {
                    // Cluster timecode — update state and continue.
                    self.cluster_timecode = read_uint(io, elem_size)? as i64;
                    trace!(cluster_timecode = self.cluster_timecode, "cluster timecode");
                    continue;
                }
                SIMPLE_BLOCK => {
                    if elem_size == SIZE_UNKNOWN || elem_size < 4 {
                        return Err(Error::InvalidData);
                    }

                    // Read the entire SimpleBlock payload.
                    let expected_len = elem_size as usize;
                    let data = io.read_up_to(expected_len)?;
                    if data.len() != expected_len {
                        return Err(Error::Eof);
                    }
                    if data.len() < 4 {
                        return Err(Error::InvalidData);
                    }

                    // Parse track number (VINT with size encoding).
                    let (track_number, vint_len) = read_vint_track_number(&data)?;

                    if data.len() < vint_len + 3 {
                        return Err(Error::InvalidData);
                    }

                    // Timecode delta: signed 16-bit big-endian.
                    let timecode_delta = i16::from_be_bytes([data[vint_len], data[vint_len + 1]]);

                    // Flags byte.
                    let flags_byte = data[vint_len + 2];
                    let keyframe = flags_byte & 0x80 != 0;
                    let lacing = (flags_byte >> 1) & 0x03;

                    if lacing != 0 {
                        // We don't implement lacing — VP9 video should never use it.
                        return Err(Error::InvalidData);
                    }

                    // Frame data starts after track_number + 2 bytes timecode + 1 byte flags.
                    let header_len = vint_len + 3;
                    let frame_data = &data[header_len..];

                    // Look up the track — skip non-video tracks silently.
                    let stream_index = match self.find_stream_index(track_number) {
                        Some(idx) => idx,
                        None => {
                            trace!(track_number, "skipping block for non-video track");
                            continue;
                        }
                    };

                    // Compute PTS.
                    let pts = self.cluster_timecode + timecode_delta as i64;

                    let mut pkt = Packet::new();
                    pkt.data = Buffer::from_slice(frame_data);
                    pkt.pts = pts;
                    pkt.dts = pts;
                    pkt.stream_index = stream_index;
                    pkt.pos = pos as i64;

                    if keyframe {
                        pkt.flags = PacketFlags::KEY;
                    }

                    // Duration from DefaultDuration if available.
                    if let Some(track) = self.tracks.iter().find(|t| t.number == track_number)
                        && track.default_duration_ns > 0
                        && self.timecode_scale_ns > 0
                    {
                        // Convert nanoseconds to timestamp units.
                        pkt.duration = (track.default_duration_ns / self.timecode_scale_ns) as i64;
                    }

                    trace!(
                        stream_index,
                        track_number,
                        pts,
                        keyframe,
                        size = frame_data.len(),
                        "SimpleBlock"
                    );

                    return Ok(pkt);
                }
                // EBML global elements — silently skip.
                EBML_VOID | EBML_CRC32 => {
                    if elem_size == SIZE_UNKNOWN {
                        return Err(Error::InvalidData);
                    }
                    skip_element(io, elem_size)?;
                    continue;
                }
                // Skip known segment-level elements that may appear between clusters.
                SEEK_HEAD | CUES | CHAPTERS | TAGS | ATTACHMENTS | INFO | TRACKS => {
                    if elem_size == SIZE_UNKNOWN {
                        return Err(Error::InvalidData);
                    }
                    trace!(
                        id = elem_id,
                        size = elem_size,
                        "skipping segment-level element"
                    );
                    skip_element(io, elem_size)?;
                    continue;
                }
                _ => {
                    // Unknown element — skip if size is known.
                    if elem_size == SIZE_UNKNOWN {
                        // Unknown size for an unknown element inside a cluster:
                        // we cannot safely skip. Treat as end of cluster.
                        if self.in_cluster {
                            self.in_cluster = false;
                            continue;
                        }
                        return Err(Error::InvalidData);
                    }
                    trace!(id = elem_id, size = elem_size, "skipping unknown element");
                    skip_element(io, elem_size)?;
                    continue;
                }
            }
        }
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
// GCD helper
// ---------------------------------------------------------------------------

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ---------------------------------------------------------------------------
// DemuxerFactory
// ---------------------------------------------------------------------------

struct WebmDemuxerFactory;

impl DemuxerFactory for WebmDemuxerFactory {
    fn descriptor(&self) -> &InputFormatDescriptor {
        static DESC: InputFormatDescriptor = InputFormatDescriptor {
            name: "webm",
            long_name: "WebM (Matroska subset)",
            extensions: "webm,mkv",
            mime_types: "video/webm,video/x-matroska",
            flags: InputFormatFlags::empty(),
            priority: 100,
        };
        &DESC
    }

    fn probe(&self, data: &ProbeData<'_>) -> i32 {
        // EBML files start with the EBML Header element ID: 0x1A 0x45 0xDF 0xA3.
        // We also look for "webm" or "matroska" DocType strings in the probe
        // buffer to distinguish from other EBML-based formats.
        if data.buf.len() < 4 || data.buf[0..4] != [0x1A, 0x45, 0xDF, 0xA3] {
            return 0;
        }
        // Search for DocType string in the probe buffer. The DocType element
        // (0x4282) is followed by a VINT size and then the string payload.
        // A quick heuristic: look for "webm" or "matroska" as a substring.
        let buf = data.buf;
        for window in buf.windows(4) {
            if window == b"webm" {
                return PROBE_SCORE_MAX;
            }
        }
        for window in buf.windows(8) {
            if window == b"matroska" {
                return PROBE_SCORE_MAX;
            }
        }
        // EBML header present but DocType not found in probe buffer — score lower.
        PROBE_SCORE_MAX / 2
    }

    fn create(&self) -> Result<Box<dyn Demuxer>> {
        Ok(Box::new(WebmDemuxer::new()))
    }
}

inventory::submit!(&WebmDemuxerFactory as &dyn DemuxerFactory);
