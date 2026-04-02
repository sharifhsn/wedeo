// H.264 Annex B raw bitstream demuxer.
//
// Reads raw H.264 elementary streams delimited by Annex B start codes
// (0x000001 or 0x00000001). Groups NAL units into access units (one
// per frame) and returns them as packets.
//
// Reference: FFmpeg libavformat/h264dec.c, ITU-T H.264 Annex B.

use wedeo_codec::decoder::CodecParameters;
use wedeo_codec_h264::nal::{NalUnitType, split_annex_b};
use wedeo_codec_h264::sps::{Sps, parse_sps};
use wedeo_core::buffer::Buffer;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::media_type::MediaType;
use wedeo_core::metadata::Metadata;
use wedeo_core::packet::{Packet, PacketFlags};
use wedeo_core::pixel_format::PixelFormat;
use wedeo_core::rational::Rational;
use wedeo_format::demuxer::{
    Demuxer, DemuxerHeader, InputFormatDescriptor, InputFormatFlags, PROBE_SCORE_EXTENSION,
    PROBE_SCORE_MAX, ProbeData, SeekFlags, Stream,
};
use wedeo_format::io::BufferedIo;
use wedeo_format::registry::DemuxerFactory;

/// Size of the initial read buffer for probing and header parsing.
const INITIAL_READ_SIZE: usize = 64 * 1024;

/// Maximum bytes to scan in probe data looking for start codes.
const PROBE_SCAN_LIMIT: usize = 4096;

// ---------------------------------------------------------------------------
// Start code scanning utilities
// ---------------------------------------------------------------------------

/// Find the next Annex B start code in `data` starting at position `pos`.
///
/// Returns `Some((start_code_pos, start_code_len))` where `start_code_len` is
/// 3 for `0x000001` or 4 for `0x00000001`. Returns `None` if no start code found.
fn find_start_code(data: &[u8], pos: usize) -> Option<(usize, usize)> {
    let mut i = pos;
    while i + 2 < data.len() {
        if data[i] == 0x00 && data[i + 1] == 0x00 {
            if data[i + 2] == 0x01 {
                return Some((i, 3));
            }
            if i + 3 < data.len() && data[i + 2] == 0x00 && data[i + 3] == 0x01 {
                return Some((i, 4));
            }
        }
        i += 1;
    }
    None
}

/// Check if a NAL unit type starts a new access unit boundary.
///
/// Per ITU-T H.264 Section 7.4.1.2.3, an access unit boundary is detected at:
///   - AUD (type 9)
///   - SPS (type 7)
///   - PPS (type 8)
///   - SEI (type 6)
///   - Prefix NAL types 14, 15, 16-18 (but we don't handle those)
///   - Slice or IDR slice with first_mb_in_slice == 0
fn is_au_boundary_nal_type(nal_type: u8) -> bool {
    matches!(nal_type, 6 | 7 | 8 | 9 | 14 | 15)
}

/// Check if the first exp-golomb coded value (first_mb_in_slice) is 0.
///
/// In exp-golomb coding, the value 0 is encoded as a single '1' bit.
/// So first_mb_in_slice == 0 iff the first bit of the slice data (after
/// the NAL header byte) is '1'.
fn is_first_mb_zero(nal_payload: &[u8]) -> bool {
    if nal_payload.is_empty() {
        return false;
    }
    // The first bit of the first byte is the MSB
    (nal_payload[0] & 0x80) != 0
}

// ---------------------------------------------------------------------------
// H.264 Demuxer
// ---------------------------------------------------------------------------

struct H264Demuxer {
    /// Buffered data from the stream that hasn't been returned yet.
    pending: Vec<u8>,
    /// Frame counter for PTS generation.
    frame_count: i64,
}

impl H264Demuxer {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            frame_count: 0,
        }
    }

    /// Read more data from I/O into the pending buffer.
    /// Returns the number of bytes read (0 at EOF).
    fn fill_pending(&mut self, io: &mut BufferedIo, min_size: usize) -> Result<usize> {
        if self.pending.len() >= min_size {
            return Ok(0);
        }
        let need = min_size.saturating_sub(self.pending.len()).max(32768);
        let chunk = io.read_up_to(need)?;
        let n = chunk.len();
        self.pending.extend_from_slice(&chunk);
        Ok(n)
    }
}

impl Demuxer for H264Demuxer {
    fn read_header(&mut self, io: &mut BufferedIo) -> Result<DemuxerHeader> {
        // Read initial data to find the SPS
        let data = io.read_up_to(INITIAL_READ_SIZE)?;
        if data.is_empty() {
            return Err(Error::InvalidData);
        }

        // Parse NAL units from the initial buffer
        let nalus = split_annex_b(&data);
        let mut sps_parsed: Option<Sps> = None;

        for nalu in &nalus {
            if nalu.nal_type == NalUnitType::Sps
                && let Ok(sps) = parse_sps(&nalu.data)
            {
                sps_parsed = Some(sps);
                break;
            }
        }

        // Build codec parameters
        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);

        // Determine time_base from SPS VUI timing info, or use 1/25 default
        let time_base;

        if let Some(ref sps) = sps_parsed {
            params.width = sps.width();
            params.height = sps.height();

            // Map chroma_format_idc to pixel format
            params.pixel_format = match sps.chroma_format_idc {
                0 => PixelFormat::Gray8,
                1 => PixelFormat::Yuv420p,
                2 => PixelFormat::Yuv422p,
                3 => PixelFormat::Yuv444p,
                _ => PixelFormat::Yuv420p,
            };

            // Use SPS frame rate for time_base if available
            if let Some(rate) = sps.frame_rate() {
                // time_base = 1/fps = num_units_in_tick*2 / time_scale
                time_base = Rational::new(rate.den, rate.num);
            } else {
                time_base = Rational::new(1, 25);
            }
        } else {
            // No SPS found; set defaults
            params.pixel_format = PixelFormat::Yuv420p;
            time_base = Rational::new(1, 25);
        }

        let mut stream = Stream::new(0, params);
        stream.time_base = time_base;


        // Seek back to start so read_packet gets the full stream
        if io.is_seekable() {
            io.seek(0)?;
        } else {
            // Non-seekable: keep the data we read as pending
            self.pending = data;
        }

        Ok(DemuxerHeader {
            streams: vec![stream],
            metadata: Metadata::new(),
            duration: 0,
            start_time: 0,
        })
    }

    fn read_packet(&mut self, io: &mut BufferedIo) -> Result<Packet> {
        // Ensure we have some data to work with
        if self.pending.is_empty() {
            let chunk = io.read_up_to(32768)?;
            if chunk.is_empty() {
                return Err(Error::Eof);
            }
            self.pending = chunk;
        }

        // Strategy: collect NAL units one at a time. The first slice NAL in the
        // current AU establishes that we have a slice. Each subsequent NAL is
        // checked for AU boundary conditions. When a boundary is detected, we
        // return everything up to that point as one packet.

        loop {
            // Find the first start code
            let first_sc = match find_start_code(&self.pending, 0) {
                Some(sc) => sc,
                None => {
                    let n = self.fill_pending(io, self.pending.len() + 32768)?;
                    if n == 0 {
                        if self.pending.is_empty() {
                            return Err(Error::Eof);
                        }
                        let data = std::mem::take(&mut self.pending);
                        return self.make_packet(data);
                    }
                    continue;
                }
            };

            let au_start = first_sc.0;

            // Build a list of (start_code_pos, nal_type) for each NAL in the
            // pending buffer. We scan forward collecting NAL boundaries.
            let mut nal_positions: Vec<(usize, u8)> = Vec::new(); // (start_code_pos, nal_type)
            let mut scan_pos = au_start;
            let mut need_more_data = false;

            while let Some(sc) = find_start_code(&self.pending, scan_pos) {
                let nal_header_pos = sc.0 + sc.1;
                if nal_header_pos >= self.pending.len() {
                    // Need more data to see the NAL header
                    let n = self.fill_pending(io, nal_header_pos + 2)?;
                    if n == 0 {
                        break;
                    }
                    need_more_data = true;
                    break;
                }
                let nal_header = self.pending[nal_header_pos];
                if (nal_header & 0x80) != 0 {
                    // forbidden_zero_bit set — skip
                    scan_pos = nal_header_pos + 1;
                    continue;
                }
                let nal_type = nal_header & 0x1F;

                // For slice NALs, ensure we can read at least one payload byte
                // (for first_mb_in_slice check)
                if (nal_type == 1 || nal_type == 5) && nal_header_pos + 1 >= self.pending.len() {
                    let n = self.fill_pending(io, nal_header_pos + 2)?;
                    if n == 0 {
                        // EOF; record what we have and break
                        nal_positions.push((sc.0, nal_type));
                        break;
                    }
                    need_more_data = true;
                    break;
                }

                nal_positions.push((sc.0, nal_type));
                scan_pos = nal_header_pos + 1;
            }

            if need_more_data {
                continue;
            }

            if nal_positions.is_empty() {
                // No start codes found at all; return everything as last packet
                let data = std::mem::take(&mut self.pending);
                if data.is_empty() {
                    return Err(Error::Eof);
                }
                return self.make_packet(data);
            }

            // If we only have one NAL and haven't hit EOF, try to get more data
            // so we can determine if there's a boundary after it.
            if nal_positions.len() == 1
                && find_start_code(&self.pending, nal_positions[0].0 + 4).is_none()
            {
                let n = self.fill_pending(io, self.pending.len() + 32768)?;
                if n > 0 {
                    continue; // Re-scan with more data
                }
                // EOF: return everything as last packet
                let au_data: Vec<u8> = self.pending[au_start..].to_vec();
                self.pending.clear();
                if au_data.is_empty() {
                    return Err(Error::Eof);
                }
                return self.make_packet(au_data);
            }

            // Now detect AU boundaries. Walk through NALs starting from the
            // second one; the first NAL is always part of the current AU.
            let mut current_au_has_slice = false;
            let mut au_boundary_idx: Option<usize> = None;

            // Check if the first NAL is a slice
            let first_nal_type = nal_positions[0].1;
            if first_nal_type == 1 || first_nal_type == 5 {
                current_au_has_slice = true;
            }

            for (i, &(sc_pos, nal_type)) in nal_positions.iter().enumerate().skip(1) {
                // Check if this NAL starts a new AU
                let starts_new_au = if is_au_boundary_nal_type(nal_type) {
                    // SPS, PPS, SEI, AUD etc. start a new AU only if the
                    // current AU already contains a slice.
                    current_au_has_slice
                } else if nal_type == 1 || nal_type == 5 {
                    if current_au_has_slice {
                        // A new slice with first_mb_in_slice == 0 starts a new AU.
                        let sc = find_start_code(&self.pending, sc_pos).unwrap();
                        let payload_pos = sc.0 + sc.1 + 1; // after NAL header byte
                        if payload_pos < self.pending.len() {
                            is_first_mb_zero(&self.pending[payload_pos..])
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if starts_new_au {
                    au_boundary_idx = Some(i);
                    break;
                }

                // Update slice tracking for the current NAL
                if nal_type == 1 || nal_type == 5 {
                    current_au_has_slice = true;
                }
            }

            if let Some(boundary) = au_boundary_idx {
                let au_end = nal_positions[boundary].0;
                let au_data: Vec<u8> = self.pending[au_start..au_end].to_vec();
                self.pending = self.pending[au_end..].to_vec();
                if au_data.is_empty() {
                    continue;
                }
                return self.make_packet(au_data);
            }

            // No boundary found yet — try to read more data to find the next
            // AU's start code. Without more data we can't tell if this is the
            // last AU or just an incomplete read.
            let n = self.fill_pending(io, self.pending.len() + 32768)?;
            if n > 0 {
                continue; // Re-scan with more data
            }

            // Truly at EOF: return everything as last AU
            let au_data: Vec<u8> = self.pending[au_start..].to_vec();
            self.pending.clear();
            if au_data.is_empty() {
                return Err(Error::Eof);
            }
            return self.make_packet(au_data);
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

impl H264Demuxer {
    /// Create a Packet from raw AU data (with Annex B start codes preserved).
    fn make_packet(&mut self, data: Vec<u8>) -> Result<Packet> {
        // Determine if this AU contains an IDR slice
        let mut is_keyframe = false;
        let nalus = split_annex_b(&data);
        for nalu in &nalus {
            if nalu.nal_type == NalUnitType::Idr {
                is_keyframe = true;
                break;
            }
        }

        let mut pkt = Packet::new();
        pkt.data = Buffer::from_slice(&data);
        pkt.pts = self.frame_count;
        pkt.dts = self.frame_count;
        pkt.duration = 1;
        pkt.stream_index = 0;
        if is_keyframe {
            pkt.flags = PacketFlags::KEY;
        }
        self.frame_count += 1;
        Ok(pkt)
    }
}

// ---------------------------------------------------------------------------
// DemuxerFactory
// ---------------------------------------------------------------------------

struct H264DemuxerFactory;

impl DemuxerFactory for H264DemuxerFactory {
    fn descriptor(&self) -> &InputFormatDescriptor {
        static DESC: InputFormatDescriptor = InputFormatDescriptor {
            name: "h264",
            long_name: "raw H.264 video",
            extensions: "h26l,h264,264,avc",
            mime_types: "video/h264",
            flags: InputFormatFlags::empty(),
            priority: 100,
        };
        &DESC
    }

    fn probe(&self, data: &ProbeData<'_>) -> i32 {
        let buf = data.buf;

        // Check file extension for a baseline score
        let ext_score = {
            let filename = data.filename.to_lowercase();
            if filename.ends_with(".264")
                || filename.ends_with(".h264")
                || filename.ends_with(".h26l")
                || filename.ends_with(".avc")
            {
                PROBE_SCORE_EXTENSION
            } else {
                0
            }
        };

        // Scan for Annex B start codes
        let scan_len = buf.len().min(PROBE_SCAN_LIMIT);
        let scan_buf = &buf[..scan_len];

        let mut best = ext_score;

        if let Some((sc_pos, sc_len)) = find_start_code(scan_buf, 0) {
            let nal_header_pos = sc_pos + sc_len;
            if nal_header_pos < scan_len {
                let nal_header = scan_buf[nal_header_pos];
                // Check forbidden_zero_bit
                if (nal_header & 0x80) == 0 {
                    let nal_type = nal_header & 0x1F;
                    if nal_type == 7 {
                        // SPS found right after start code — strong match
                        return PROBE_SCORE_MAX;
                    }
                    // Start codes found but first NAL isn't SPS.
                    // Look further for an SPS.
                    let mut pos = nal_header_pos + 1;
                    while let Some((next_sc_pos, next_sc_len)) = find_start_code(scan_buf, pos) {
                        let next_nal_pos = next_sc_pos + next_sc_len;
                        if next_nal_pos >= scan_len {
                            break;
                        }
                        let next_header = scan_buf[next_nal_pos];
                        if (next_header & 0x80) == 0 {
                            let next_type = next_header & 0x1F;
                            if next_type == 7 {
                                return PROBE_SCORE_MAX;
                            }
                        }
                        pos = next_nal_pos + 1;
                    }
                    // Start codes found but no SPS
                    best = best.max(25);
                }
            }
        }

        best
    }

    fn create(&self) -> Result<Box<dyn Demuxer>> {
        Ok(Box::new(H264Demuxer::new()))
    }
}

inventory::submit!(&H264DemuxerFactory as &dyn DemuxerFactory);
