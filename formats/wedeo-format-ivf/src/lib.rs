//! Native IVF (On2/Google VP8, VP9, AV1) container demuxer and muxer.
//!
//! File layout (all integer fields little-endian):
//! - 32-byte file header:
//!   - `0..4`: `"DKIF"` magic
//!   - `4..6`: version (`0`)
//!   - `6..8`: header length (`32`)
//!   - `8..12`: fourcc (`"VP80"`, `"VP90"`, `"AV01"`)
//!   - `12..14`: width (`u16`)
//!   - `14..16`: height (`u16`)
//!   - `16..20`: timebase denominator / frame rate numerator (`u32`)
//!   - `20..24`: timebase numerator / frame rate denominator (`u32`)
//!   - `24..28`: frame count (`u32`)
//!   - `28..32`: unused (`u32`)
//! - Sequence of frames:
//!   - `0..4`: frame byte length (`u32`)
//!   - `4..12`: presentation timestamp (`u64`)
//!   - `12..12 + frame_size`: compressed frame payload

#![forbid(unsafe_code)]

use std::io::{self, Read, SeekFrom};
use std::sync::Arc;

use wedeo_core::{
    BitReader, BufByteSource, Buffer, ByteSink, ByteSource, CodecId, ContainerHeader, Demuxer,
    DemuxerDescriptor, DemuxerFactory, Error, ErrorPolicy, HealthEvent, HealthIssue, Muxer,
    MuxerDescriptor, MuxerFactory, Packet, PacketFlags, PixelLayout, Rational, RecvStatus,
    Registry, Result, Rounding, SeekTarget, StreamHealth, StreamParams, Timestamp, TrackDescriptor,
    TrackDisposition, TrackId, VideoStreamParams, rescale_q_rnd,
};

/// IVF file header magic bytes.
pub const IVF_MAGIC: [u8; 4] = *b"DKIF";
/// Standard IVF file header length in bytes.
pub const IVF_FILE_HEADER_LEN: usize = 32;
/// IVF per-frame header length in bytes.
pub const IVF_FRAME_HEADER_LEN: usize = 12;

const VP9_SYNC_CODE: u32 = 0x49_83_42;

/// Map a 4-byte IVF FourCC to a [`CodecId`].
pub fn fourcc_to_codec_id(fourcc: &[u8; 4]) -> Option<CodecId> {
    if fourcc.eq_ignore_ascii_case(b"VP80") {
        Some(CodecId::Vp8)
    } else if fourcc.eq_ignore_ascii_case(b"VP90") {
        Some(CodecId::Vp9)
    } else if fourcc.eq_ignore_ascii_case(b"AV01") {
        Some(CodecId::Av1)
    } else {
        None
    }
}

/// Map a [`CodecId`] to its canonical 4-byte IVF FourCC.
pub const fn codec_id_to_fourcc(codec_id: CodecId) -> Option<[u8; 4]> {
    match codec_id {
        CodecId::Vp8 => Some(*b"VP80"),
        CodecId::Vp9 => Some(*b"VP90"),
        CodecId::Av1 => Some(*b"AV01"),
        _ => None,
    }
}

/// Detect whether a compressed VP8, VP9, or AV1 packet payload is a keyframe.
///
/// Inspects the uncompressed bitstream header directly:
/// - **VP8**: RFC 6386 §9.1 frame tag bit 0 (`frame_type == 0`) and optional start code `0x9D 0x01 0x2A`.
/// - **VP9**: VP9 uncompressed header (`frame_marker == 2`, `!show_existing_frame`, `!frame_type`),
///   including VP9 superframe index parsing.
/// - **AV1**: OBU stream scan for `OBU_SEQUENCE_HEADER` (`reduced_still_picture_header`) and
///   `OBU_FRAME` / `OBU_FRAME_HEADER` (`!show_existing_frame && frame_type == KEY_FRAME`).
pub fn detect_keyframe(codec_id: CodecId, data: &[u8]) -> bool {
    let mut reduced_still = false;
    detect_keyframe_stateful(codec_id, data, &mut reduced_still).unwrap_or(false)
}

fn detect_keyframe_stateful(
    codec_id: CodecId,
    data: &[u8],
    av1_reduced_still: &mut bool,
) -> Option<bool> {
    match codec_id {
        CodecId::Vp8 => detect_vp8_keyframe(data),
        CodecId::Vp9 => detect_vp9_keyframe(data),
        CodecId::Av1 => detect_av1_keyframe(data, av1_reduced_still),
        _ => None,
    }
}

fn detect_vp8_keyframe(data: &[u8]) -> Option<bool> {
    let &b0 = data.first()?;
    let is_key = (b0 & 0x01) == 0;
    if is_key && data.len() >= 6 {
        // Real VP8 keyframes carry the 3-byte start code 0x9D 0x01 0x2A at bytes 3..6.
        if data[3..6] != [0x9D, 0x01, 0x2A] {
            return None;
        }
    }
    Some(is_key)
}

fn detect_vp9_keyframe(data: &[u8]) -> Option<bool> {
    if data.is_empty() {
        return None;
    }
    // If this is a VP9 superframe, check each subframe in order.
    if let Some(subframes) = parse_vp9_superframe_ranges(data) {
        let mut any_valid = false;
        for (start, len) in subframes {
            if let Some(is_key) = detect_vp9_single_frame(&data[start..start + len]) {
                any_valid = true;
                if is_key {
                    return Some(true);
                }
            }
        }
        if any_valid {
            return Some(false);
        }
    }
    detect_vp9_single_frame(data)
}

fn detect_vp9_single_frame(data: &[u8]) -> Option<bool> {
    if data.is_empty() {
        return None;
    }
    let mut br = BitReader::new(data);
    let frame_marker = br.read_bits(2).ok()?;
    if frame_marker != 0x2 {
        return None;
    }
    let profile_low = br.read_bit().ok()?;
    let profile_high = br.read_bit().ok()?;
    let mut profile = (profile_low | (profile_high << 1)) as u8;
    if profile == 3 {
        profile += br.read_bit().ok()? as u8;
    }
    if profile > 3 {
        return None;
    }
    let show_existing_frame = br.read_bool().ok()?;
    if show_existing_frame {
        return Some(false);
    }
    let is_keyframe = !br.read_bool().ok()?;
    if is_keyframe && br.bits_left() >= 26 {
        // Skip show_frame (1) and error_resilient_mode (1), then verify 24-bit VP9 sync code.
        let _show_frame = br.read_bool().ok()?;
        let _error_resilient = br.read_bool().ok()?;
        let sync = br.read_bits(24).ok()?;
        if sync != VP9_SYNC_CODE {
            return None;
        }
    }
    Some(is_keyframe)
}

fn parse_vp9_superframe_ranges(packet: &[u8]) -> Option<Vec<(usize, usize)>> {
    let &marker = packet.last()?;
    if (marker & 0xE0) != 0xC0 {
        return None;
    }
    let bytes_per_size = (((marker >> 3) & 0x03) + 1) as usize;
    let frame_count = ((marker & 0x07) + 1) as usize;
    let index_size = 2 + frame_count * bytes_per_size;
    if packet.len() < index_size {
        return None;
    }
    let index_start = packet.len() - index_size;
    if packet[index_start] != marker {
        return None;
    }
    let mut ranges = Vec::with_capacity(frame_count);
    let mut offset = 0usize;
    let mut idx_pos = index_start + 1;
    for _ in 0..frame_count {
        let mut sz = 0usize;
        for b in 0..bytes_per_size {
            sz |= (packet[idx_pos + b] as usize) << (b * 8);
        }
        idx_pos += bytes_per_size;
        if sz == 0 || offset.checked_add(sz)? > index_start {
            return None;
        }
        ranges.push((offset, sz));
        offset += sz;
    }
    Some(ranges)
}

fn try_read_leb128(data: &[u8]) -> Option<(usize, usize)> {
    let mut value = 0u64;
    for (i, &byte) in data.iter().take(8).enumerate() {
        value |= u64::from(byte & 0x7F) << (i * 7);
        if (byte & 0x80) == 0 {
            if value > u32::MAX as u64 {
                return None;
            }
            return Some((value as usize, i + 1));
        }
    }
    None
}

fn detect_av1_keyframe(data: &[u8], reduced_still: &mut bool) -> Option<bool> {
    if data.is_empty() {
        return None;
    }
    let mut pos = 0usize;
    let mut saw_valid_obu = false;
    let mut saw_seq_header = false;

    while pos < data.len() {
        let b0 = data[pos];
        // obu_forbidden_bit must be 0
        if (b0 & 0x80) != 0 {
            return None;
        }
        let obu_type = (b0 >> 3) & 0x0F;
        let extension_flag = (b0 & 0x04) != 0;
        let has_size_field = (b0 & 0x02) != 0;
        // Reserved bit must be 0
        if (b0 & 0x01) != 0 {
            return None;
        }

        let mut hdr_end = pos + 1;
        if extension_flag {
            if hdr_end >= data.len() {
                return None;
            }
            hdr_end += 1;
        }

        let (payload_size, payload_start) = if has_size_field {
            let (sz, leb_len) = try_read_leb128(&data[hdr_end..])?;
            (sz, hdr_end + leb_len)
        } else {
            (data.len().checked_sub(hdr_end)?, hdr_end)
        };

        let payload_end = payload_start.checked_add(payload_size)?;
        if payload_end > data.len() {
            return None;
        }
        let payload = &data[payload_start..payload_end];
        saw_valid_obu = true;

        match obu_type {
            // OBU_SEQUENCE_HEADER = 1
            1 => {
                saw_seq_header = true;
                if let Some(&first) = payload.first() {
                    // seq_profile (3 bits), still_picture (1 bit), reduced_still_picture_header (1 bit = bit 3)
                    let reduced = (first & 0x08) != 0;
                    *reduced_still = reduced;
                }
            }
            // OBU_FRAME_HEADER = 3, OBU_FRAME = 6, OBU_REDUNDANT_FRAME_HEADER = 7
            3 | 6 | 7 => {
                if *reduced_still {
                    return Some(true);
                }
                let &first = payload.first()?;
                let show_existing_frame = (first & 0x80) != 0;
                if show_existing_frame {
                    return Some(false);
                }
                let frame_type = (first >> 5) & 0x03;
                // KEY_FRAME == 0
                return Some(frame_type == 0);
            }
            _ => {}
        }

        pos = payload_end;
    }

    if saw_valid_obu && saw_seq_header {
        Some(true)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
struct IvfIndexEntry {
    offset: u64,
    pts: i64,
    is_key: bool,
}

/// Native IVF container demuxer owning its [`ByteSource`].
pub struct IvfDemuxer {
    io: BufByteSource,
    header: ContainerHeader,
    health: StreamHealth,
    policy: ErrorPolicy,
    codec_id: CodecId,
    time_base: Rational,
    data_start_offset: u64,
    frame_count: u64,
    av1_reduced_still: bool,
    index: Vec<IvfIndexEntry>,
}

impl IvfDemuxer {
    /// Open an IVF stream from a [`ByteSource`] and parse its 32-byte file header.
    pub fn open(source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Self> {
        let mut io = BufByteSource::new(source);
        let mut hdr = [0u8; IVF_FILE_HEADER_LEN];
        io.read_exact_bytes(&mut hdr)
            .map_err(|e| Error::io("ivf/demuxer", &e))?;

        if hdr[0..4] != IVF_MAGIC {
            return Err(Error::invalid_data("ivf/demuxer", "invalid IVF magic"));
        }

        let version = u16::from_le_bytes([hdr[4], hdr[5]]);
        if version != 0 {
            return Err(Error::unsupported(
                "ivf/demuxer",
                "unsupported IVF version (expected 0)",
            ));
        }

        let header_len = u16::from_le_bytes([hdr[6], hdr[7]]) as usize;
        if header_len < IVF_FILE_HEADER_LEN {
            return Err(Error::invalid_data(
                "ivf/demuxer",
                "IVF header length shorter than 32 bytes",
            ));
        }

        let fourcc = [hdr[8], hdr[9], hdr[10], hdr[11]];
        let codec_id = fourcc_to_codec_id(&fourcc).ok_or_else(|| {
            Error::unsupported("ivf/demuxer", "unsupported codec FourCC in IVF header")
        })?;

        let width = u32::from(u16::from_le_bytes([hdr[12], hdr[13]]));
        let height = u32::from(u16::from_le_bytes([hdr[14], hdr[15]]));
        let fps_num = u32::from_le_bytes([hdr[16], hdr[17], hdr[18], hdr[19]]);
        let fps_den = u32::from_le_bytes([hdr[20], hdr[21], hdr[22], hdr[23]]);
        let num_frames = u32::from_le_bytes([hdr[24], hdr[25], hdr[26], hdr[27]]);

        if header_len > IVF_FILE_HEADER_LEN {
            io.skip((header_len - IVF_FILE_HEADER_LEN) as u64)
                .map_err(|e| Error::io("ivf/demuxer", &e))?;
        }
        let data_start_offset = io.tell();

        let (time_base, avg_frame_rate) = if fps_num > 0
            && fps_den > 0
            && fps_num <= i32::MAX as u32
            && fps_den <= i32::MAX as u32
        {
            (
                Rational::new(fps_den as i32, fps_num as i32),
                Rational::new(fps_num as i32, fps_den as i32),
            )
        } else {
            (Rational::new(1, 30), Rational::new(30, 1))
        };

        let mut video_params = VideoStreamParams::new(width, height, PixelLayout::YUV420P);
        video_params.avg_frame_rate = avg_frame_rate;

        let mut track = TrackDescriptor::new(
            TrackId(0),
            codec_id,
            time_base,
            StreamParams::Video(video_params),
        );
        track.codec_tag = u32::from_le_bytes(fourcc);
        track.disposition = TrackDisposition::DEFAULT;
        if num_frames > 0 {
            track.nb_frames = Some(u64::from(num_frames));
            track.duration = i64::from(num_frames);
        }

        let mut container = ContainerHeader::new("ivf");
        container.time_base = time_base;
        if num_frames > 0 {
            container.duration = Some(i64::from(num_frames));
        }
        container.tracks.push(track);

        Ok(Self {
            io,
            header: container,
            health: StreamHealth::new(),
            policy,
            codec_id,
            time_base,
            data_start_offset,
            frame_count: 0,
            av1_reduced_still: false,
            index: Vec::new(),
        })
    }

    fn record_index(&mut self, offset: u64, pts: i64, is_key: bool) {
        if self.index.last().is_none_or(|last| offset > last.offset) {
            self.index.push(IvfIndexEntry {
                offset,
                pts,
                is_key,
            });
        }
    }
}

impl Demuxer for IvfDemuxer {
    fn header(&self) -> &ContainerHeader {
        &self.header
    }

    fn read_packet(&mut self) -> Result<RecvStatus<Packet>> {
        let frame_offset = self.io.tell();
        let mut fhdr = [0u8; IVF_FRAME_HEADER_LEN];

        let first_byte = match self.io.read_u8() {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                return Ok(RecvStatus::Eof);
            }
            Err(e) => return Err(Error::io("ivf/demuxer", &e)),
        };
        fhdr[0] = first_byte;

        if let Err(e) = self.io.read_exact_bytes(&mut fhdr[1..]) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                self.health.record(HealthEvent {
                    track: TrackId(0),
                    pts: Timestamp::NONE,
                    byte_offset: Some(frame_offset),
                    kind: HealthIssue::TruncatedPacket,
                    detail: "truncated IVF frame header",
                });
                if self.policy == ErrorPolicy::Strict {
                    return Err(Error::corrupt(
                        "ivf/demuxer",
                        "truncated IVF frame header at EOF",
                    ));
                }
                return Ok(RecvStatus::Eof);
            }
            return Err(Error::io("ivf/demuxer", &e));
        }

        let frame_size = u32::from_le_bytes([fhdr[0], fhdr[1], fhdr[2], fhdr[3]]) as usize;
        let pts_raw = i64::from_le_bytes([
            fhdr[4], fhdr[5], fhdr[6], fhdr[7], fhdr[8], fhdr[9], fhdr[10], fhdr[11],
        ]);

        if frame_size == 0 {
            self.health.record(HealthEvent {
                track: TrackId(0),
                pts: Timestamp::new(pts_raw),
                byte_offset: Some(frame_offset),
                kind: HealthIssue::CorruptSyntax,
                detail: "zero-length IVF frame",
            });
            if self.policy == ErrorPolicy::Strict {
                return Err(Error::corrupt("ivf/demuxer", "zero-length IVF frame"));
            }
        }

        let mut payload_buf = Buffer::new(frame_size);
        let mut read_total = 0usize;
        if let Some(dst) = payload_buf.try_data_mut() {
            while read_total < frame_size {
                match self.io.read(&mut dst[read_total..]) {
                    Ok(0) => break,
                    Ok(n) => read_total += n,
                    Err(e) => return Err(Error::io("ivf/demuxer", &e)),
                }
            }
        }

        let mut flags = PacketFlags::empty();
        if read_total < frame_size {
            payload_buf = payload_buf
                .subslice(0..read_total)
                .unwrap_or_else(|| Buffer::new(0));
            self.health.record(HealthEvent {
                track: TrackId(0),
                pts: Timestamp::new(pts_raw),
                byte_offset: Some(frame_offset),
                kind: HealthIssue::TruncatedPacket,
                detail: "truncated IVF frame payload",
            });
            if self.policy == ErrorPolicy::Strict {
                return Err(Error::corrupt("ivf/demuxer", "truncated IVF frame payload"));
            }
            if payload_buf.is_empty() {
                return Ok(RecvStatus::Eof);
            }
            flags |= PacketFlags::CORRUPT;
        }

        let is_key = detect_keyframe_stateful(
            self.codec_id,
            payload_buf.data(),
            &mut self.av1_reduced_still,
        )
        .unwrap_or(self.frame_count == 0);
        if is_key {
            flags |= PacketFlags::KEY;
        }

        self.record_index(frame_offset, pts_raw, is_key);

        if self.header.start_time.is_none() {
            self.header.start_time = Timestamp::new(pts_raw);
        }

        let mut pkt = Packet::new(payload_buf, TrackId(0), self.time_base);
        pkt.pts = Timestamp::new(pts_raw);
        pkt.dts = Timestamp::new(pts_raw);
        pkt.duration = 1;
        pkt.pos = Some(frame_offset);
        pkt.flags = flags;

        self.frame_count += 1;
        Ok(RecvStatus::Ready(pkt))
    }

    fn seek(&mut self, target: SeekTarget) -> Result<()> {
        if !self.io.is_seekable() {
            return Err(Error::not_seekable(
                "ivf/demuxer",
                "cannot seek non-seekable IVF source",
            ));
        }
        if let Some(track) = target.track
            && track != TrackId(0)
        {
            return Err(Error::not_found("ivf/demuxer", "track not found in IVF"));
        }

        let target_pts = if target.timestamp.is_known() {
            let raw = target.timestamp.raw();
            if target.time_base.is_valid() && target.time_base != self.time_base {
                rescale_q_rnd(
                    raw,
                    target.time_base,
                    self.time_base,
                    Rounding::NearInf,
                    true,
                )
            } else {
                raw
            }
        } else {
            0
        };

        // Scan forward to index frames if we haven't indexed up to target_pts yet.
        if self.index.last().is_none_or(|last| last.pts < target_pts) {
            let start_scan = self
                .index
                .last()
                .map_or(self.data_start_offset, |entry| entry.offset);
            self.io
                .seek(SeekFrom::Start(start_scan))
                .map_err(|e| Error::io("ivf/demuxer", &e))?;

            while let Ok(pkt_status) = self.read_packet() {
                match pkt_status {
                    RecvStatus::Ready(pkt) => {
                        if pkt.pts.raw() >= target_pts && !pkt.flags.contains(PacketFlags::KEY) {
                            // Keep scanning only until we pass target_pts (or find next keyframe if !backward)
                            if target.backward {
                                break;
                            }
                        } else if pkt.pts.raw() >= target_pts
                            && pkt.flags.contains(PacketFlags::KEY)
                        {
                            break;
                        }
                    }
                    RecvStatus::NeedMore | RecvStatus::Eof => break,
                }
            }
        }

        if self.index.is_empty() {
            self.io
                .seek(SeekFrom::Start(self.data_start_offset))
                .map_err(|e| Error::io("ivf/demuxer", &e))?;
            self.frame_count = 0;
            return Ok(());
        }

        let chosen_idx = if target.backward {
            let pos = self
                .index
                .partition_point(|e| e.pts <= target_pts)
                .saturating_sub(1);
            let mut key_idx = pos;
            while key_idx > 0 && !self.index[key_idx].is_key {
                key_idx -= 1;
            }
            key_idx
        } else {
            let pos = self.index.partition_point(|e| e.pts < target_pts);
            pos.min(self.index.len() - 1)
        };

        let entry = self.index[chosen_idx];
        self.io
            .seek(SeekFrom::Start(entry.offset))
            .map_err(|e| Error::io("ivf/demuxer", &e))?;
        self.frame_count = chosen_idx as u64;
        Ok(())
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

/// Native IVF container muxer owning its [`ByteSink`].
pub struct IvfMuxer {
    sink: Box<dyn ByteSink>,
    track_id: TrackId,
    time_base: Rational,
    frame_count: u32,
    header_written: bool,
}

impl IvfMuxer {
    /// Create a new IVF muxer writing to `sink`.
    pub fn new(sink: Box<dyn ByteSink>) -> Self {
        Self {
            sink,
            track_id: TrackId(0),
            time_base: Rational::new(1, 30),
            frame_count: 0,
            header_written: false,
        }
    }
}

impl Muxer for IvfMuxer {
    fn write_header(&mut self, header: &ContainerHeader) -> Result<()> {
        if self.header_written {
            return Err(Error::invalid_argument(
                "ivf/muxer",
                "IVF header already written",
            ));
        }
        if header.tracks.len() != 1 {
            return Err(Error::invalid_argument(
                "ivf/muxer",
                "IVF container requires exactly 1 video track",
            ));
        }
        let track = &header.tracks[0];
        let fourcc = codec_id_to_fourcc(track.codec_id).ok_or_else(|| {
            Error::unsupported("ivf/muxer", "IVF muxer only supports VP8, VP9, and AV1")
        })?;
        let video = track.params.video().ok_or_else(|| {
            Error::invalid_argument("ivf/muxer", "IVF track must carry VideoStreamParams")
        })?;

        if video.width == 0
            || video.height == 0
            || video.width > u32::from(u16::MAX)
            || video.height > u32::from(u16::MAX)
        {
            return Err(Error::invalid_argument(
                "ivf/muxer",
                "IVF video dimensions must be in 1..=65535",
            ));
        }

        let time_base = if track.time_base.is_valid() && track.time_base.num > 0 {
            track.time_base
        } else if video.avg_frame_rate.is_valid() && video.avg_frame_rate.num > 0 {
            video.avg_frame_rate.invert()
        } else {
            Rational::new(1, 30)
        };

        self.track_id = track.id;
        self.time_base = time_base;

        let fps_num = time_base.den as u32;
        let fps_den = time_base.num as u32;
        let initial_frames = track.nb_frames.unwrap_or(0).min(u64::from(u32::MAX)) as u32;

        let mut hdr = [0u8; IVF_FILE_HEADER_LEN];
        hdr[0..4].copy_from_slice(&IVF_MAGIC);
        hdr[4..6].copy_from_slice(&0u16.to_le_bytes()); // version = 0
        hdr[6..8].copy_from_slice(&(IVF_FILE_HEADER_LEN as u16).to_le_bytes());
        hdr[8..12].copy_from_slice(&fourcc);
        hdr[12..14].copy_from_slice(&(video.width as u16).to_le_bytes());
        hdr[14..16].copy_from_slice(&(video.height as u16).to_le_bytes());
        hdr[16..20].copy_from_slice(&fps_num.to_le_bytes());
        hdr[20..24].copy_from_slice(&fps_den.to_le_bytes());
        hdr[24..28].copy_from_slice(&initial_frames.to_le_bytes());
        hdr[28..32].copy_from_slice(&0u32.to_le_bytes());

        self.sink
            .write_all(&hdr)
            .map_err(|e| Error::io("ivf/muxer", &e))?;
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, pkt: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid_argument(
                "ivf/muxer",
                "write_header must be called before write_packet",
            ));
        }
        if pkt.track_id != self.track_id {
            return Err(Error::invalid_argument(
                "ivf/muxer",
                "packet track_id does not match IVF video track",
            ));
        }
        let data = pkt.data.data();
        if data.len() > u32::MAX as usize {
            return Err(Error::invalid_argument(
                "ivf/muxer",
                "IVF frame size exceeds u32::MAX",
            ));
        }

        let raw_ts = if pkt.pts.is_known() {
            pkt.pts
        } else if pkt.dts.is_known() {
            pkt.dts
        } else {
            Timestamp::new(i64::from(self.frame_count))
        };

        let pts =
            if pkt.time_base.is_valid() && pkt.time_base.num > 0 && pkt.time_base != self.time_base
            {
                raw_ts
                    .rescale(pkt.time_base, self.time_base, Rounding::NearInf)
                    .raw()
            } else {
                raw_ts.raw()
            };

        let mut fhdr = [0u8; IVF_FRAME_HEADER_LEN];
        fhdr[0..4].copy_from_slice(&(data.len() as u32).to_le_bytes());
        fhdr[4..12].copy_from_slice(&(pts as u64).to_le_bytes());

        self.sink
            .write_all(&fhdr)
            .map_err(|e| Error::io("ivf/muxer", &e))?;
        self.sink
            .write_all(data)
            .map_err(|e| Error::io("ivf/muxer", &e))?;

        self.frame_count = self.frame_count.saturating_add(1);
        Ok(())
    }

    fn finalize(mut self: Box<Self>) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid_argument(
                "ivf/muxer",
                "cannot finalize IVF muxer before write_header",
            ));
        }
        if self.sink.is_seekable() {
            self.sink
                .seek(SeekFrom::Start(24))
                .map_err(|e| Error::io("ivf/muxer", &e))?;
            self.sink
                .write_all(&self.frame_count.to_le_bytes())
                .map_err(|e| Error::io("ivf/muxer", &e))?;
        }
        self.sink.flush().map_err(|e| Error::io("ivf/muxer", &e))?;
        Ok(())
    }
}

/// Factory for probing and opening [`IvfDemuxer`].
pub struct IvfDemuxerFactory;

impl DemuxerFactory for IvfDemuxerFactory {
    fn descriptor(&self) -> &DemuxerDescriptor {
        static DESC: DemuxerDescriptor = DemuxerDescriptor {
            name: "ivf",
            long_name: "On2 IVF (VP8/VP9/AV1)",
            extensions: &["ivf"],
            mime_types: &["video/x-ivf"],
        };
        &DESC
    }

    fn probe(&self, head: &[u8], ext: Option<&str>) -> u8 {
        if head.len() >= 12 && head[0..4] == IVF_MAGIC {
            let version = u16::from_le_bytes([head[4], head[5]]);
            let hdr_len = u16::from_le_bytes([head[6], head[7]]);
            let fourcc = [head[8], head[9], head[10], head[11]];
            if version == 0 && hdr_len >= 32 && fourcc_to_codec_id(&fourcc).is_some() {
                return 100;
            }
            return 80;
        }
        if head.len() >= 4 && head[0..4] == IVF_MAGIC {
            return 80;
        }
        if ext.is_some_and(|e| e.trim_start_matches('.').eq_ignore_ascii_case("ivf")) {
            return 25;
        }
        0
    }

    fn open(&self, source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Box<dyn Demuxer>> {
        Ok(Box::new(IvfDemuxer::open(source, policy)?))
    }
}

/// Factory for creating [`IvfMuxer`].
pub struct IvfMuxerFactory;

impl MuxerFactory for IvfMuxerFactory {
    fn descriptor(&self) -> &MuxerDescriptor {
        static DESC: MuxerDescriptor = MuxerDescriptor {
            name: "ivf",
            long_name: "On2 IVF (VP8/VP9/AV1)",
            extensions: &["ivf"],
            mime_types: &["video/x-ivf"],
            default_video: CodecId::Vp9,
            default_audio: CodecId::None,
        };
        &DESC
    }

    fn create(&self, sink: Box<dyn ByteSink>) -> Result<Box<dyn Muxer>> {
        Ok(Box::new(IvfMuxer::new(sink)))
    }
}

/// Register the IVF demuxer and muxer with a [`Registry`].
pub fn register_ivf(registry: &mut Registry) {
    registry.register_demuxer(Arc::new(IvfDemuxerFactory));
    registry.register_muxer(Arc::new(IvfMuxerFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::{BitWriter, MemorySink, MemorySource};

    fn make_vp9_keyframe_bytes() -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.write_bits(0b10, 2); // frame_marker = 2
        bw.write_bit(false); // profile_low = 0
        bw.write_bit(false); // profile_high = 0
        bw.write_bit(false); // show_existing_frame = 0
        bw.write_bit(false); // frame_type = 0 (keyframe)
        bw.write_bit(true); // show_frame = 1
        bw.write_bit(false); // error_resilient_mode = 0
        bw.write_bits(VP9_SYNC_CODE, 24);
        bw.write_bits(0, 16); // trailing payload
        bw.finish()
    }

    fn make_vp9_interframe_bytes() -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.write_bits(0b10, 2); // frame_marker = 2
        bw.write_bit(false); // profile_low = 0
        bw.write_bit(false); // profile_high = 0
        bw.write_bit(false); // show_existing_frame = 0
        bw.write_bit(true); // frame_type = 1 (interframe)
        bw.write_bit(true); // show_frame = 1
        bw.write_bit(false); // error_resilient_mode = 0
        bw.write_bits(0xAA55, 16);
        bw.finish()
    }

    fn make_av1_keyframe_obus() -> Vec<u8> {
        // OBU_TEMPORAL_DELIMITER (type=2, has_size=1, size=0)
        // + OBU_SEQUENCE_HEADER (type=1, has_size=1, size=4, reduced_still=0)
        // + OBU_FRAME (type=6, has_size=1, size=4, show_existing=0, frame_type=0 KEY_FRAME)
        vec![
            0x12, 0x00, // Temporal Delimiter OBU
            0x0A, 0x04, 0x00, 0x00, 0x00, 0x00, // Sequence Header OBU
            0x32, 0x04, 0x10, 0xAA, 0xBB,
            0xCC, // Frame OBU: show_existing=0, frame_type=0 (KEY)
        ]
    }

    fn make_av1_interframe_obus() -> Vec<u8> {
        // OBU_TEMPORAL_DELIMITER + OBU_FRAME (show_existing=0, frame_type=1 INTER_FRAME -> 0b0010_0000 = 0x20)
        vec![
            0x12, 0x00, // Temporal Delimiter OBU
            0x32, 0x04, 0x20, 0x11, 0x22, 0x33, // Frame OBU: frame_type=1 (INTER)
        ]
    }

    #[test]
    fn vp9_and_av1_keyframe_detection_accuracy() {
        let vp9_kf = make_vp9_keyframe_bytes();
        let vp9_inter = make_vp9_interframe_bytes();
        assert!(detect_keyframe(CodecId::Vp9, &vp9_kf));
        assert!(!detect_keyframe(CodecId::Vp9, &vp9_inter));

        let av1_kf = make_av1_keyframe_obus();
        let av1_inter = make_av1_interframe_obus();
        assert!(detect_keyframe(CodecId::Av1, &av1_kf));
        assert!(!detect_keyframe(CodecId::Av1, &av1_inter));

        // VP8 keyframe (bit 0 == 0 + start code 0x9D 0x01 0x2A) vs interframe (bit 0 == 1)
        let vp8_kf = vec![0x10, 0x00, 0x00, 0x9D, 0x01, 0x2A, 0x40, 0x01];
        let vp8_inter = vec![0x11, 0x00, 0x00, 0x01, 0x02, 0x03];
        assert!(detect_keyframe(CodecId::Vp8, &vp8_kf));
        assert!(!detect_keyframe(CodecId::Vp8, &vp8_inter));
    }

    #[test]
    fn ivf_mux_demux_vfr_roundtrip_and_seek() {
        let mut reg = Registry::new();
        register_ivf(&mut reg);

        let tb = Rational::new(1, 90_000);
        let mut video = VideoStreamParams::new(640, 360, PixelLayout::YUV420P);
        video.avg_frame_rate = Rational::new(30, 1);
        let track = TrackDescriptor::new(TrackId(0), CodecId::Vp9, tb, StreamParams::Video(video));
        let mut hdr = ContainerHeader::new("ivf");
        hdr.time_base = tb;
        hdr.tracks.push(track);

        let frames = [
            (0i64, make_vp9_keyframe_bytes(), true),
            (3000i64, make_vp9_interframe_bytes(), false),
            (7500i64, make_vp9_keyframe_bytes(), true),
            (10500i64, make_vp9_interframe_bytes(), false),
        ];

        // Use a shared Vec via custom ByteSink or direct IvfMuxer over MemorySink
        let sink = MemorySink::new();
        let mut muxer: Box<dyn Muxer> = Box::new(IvfMuxer::new(Box::new(sink)));
        muxer.write_header(&hdr).unwrap();

        // To inspect bytes after finalize, let's test over a Temporary file or shared buffer sink.
        let tmp_path = "/tmp/wedeo_test_ivf_roundtrip.ivf";
        let file_sink = wedeo_core::FileSink::create(tmp_path).unwrap();
        let mux_factory = reg.find_muxer_by_name("ivf").unwrap();
        let mut muxer = mux_factory.create(Box::new(file_sink)).unwrap();
        muxer.write_header(&hdr).unwrap();

        for (pts, payload, _) in &frames {
            let mut pkt = Packet::new(Buffer::from_slice(payload), TrackId(0), tb);
            pkt.pts = Timestamp::new(*pts);
            pkt.dts = Timestamp::new(*pts);
            pkt.duration = 3000;
            muxer.write_packet(&pkt).unwrap();
        }
        muxer.finalize().unwrap();

        let bytes = std::fs::read(tmp_path).unwrap();
        std::fs::remove_file(tmp_path).ok();

        // Check patched frame count in header bytes 24..28
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            4
        );

        // Probe via registry
        let demux_factory = reg.probe_demuxer(&bytes, Some("ivf")).unwrap();
        assert_eq!(demux_factory.descriptor().name, "ivf");

        let mut demuxer = demux_factory
            .open(
                Box::new(MemorySource::new(bytes.clone())),
                ErrorPolicy::Strict,
            )
            .unwrap();
        assert_eq!(demuxer.header().tracks.len(), 1);
        let dtrack = &demuxer.header().tracks[0];
        assert_eq!(dtrack.codec_id, CodecId::Vp9);
        assert_eq!(dtrack.time_base, tb);
        assert_eq!(dtrack.nb_frames, Some(4));
        assert_eq!(dtrack.params.video().unwrap().width, 640);
        assert_eq!(dtrack.params.video().unwrap().height, 360);

        let mut read_pkts = Vec::new();
        while let RecvStatus::Ready(pkt) = demuxer.read_packet().unwrap() {
            read_pkts.push(pkt);
        }
        assert_eq!(read_pkts.len(), 4);
        for (i, (expected_pts, expected_data, expected_key)) in frames.iter().enumerate() {
            assert_eq!(read_pkts[i].pts, Timestamp::new(*expected_pts));
            assert_eq!(read_pkts[i].dts, Timestamp::new(*expected_pts));
            assert_eq!(read_pkts[i].data.data(), expected_data.as_slice());
            assert_eq!(
                read_pkts[i].flags.contains(PacketFlags::KEY),
                *expected_key,
                "frame {i} keyframe flag mismatch"
            );
        }
        assert!(demuxer.health().is_clean());

        // Test backward keyframe seek to pts 9000 -> should land on keyframe at pts 7500
        demuxer
            .seek(SeekTarget {
                track: Some(TrackId(0)),
                timestamp: Timestamp::new(9000),
                time_base: tb,
                backward: true,
            })
            .unwrap();
        match demuxer.read_packet().unwrap() {
            RecvStatus::Ready(pkt) => {
                assert_eq!(pkt.pts, Timestamp::new(7500));
                assert!(pkt.flags.contains(PacketFlags::KEY));
            }
            other => panic!("expected Ready packet after seek, got {other:?}"),
        }
    }

    #[test]
    fn ivf_av1_roundtrip() {
        let tb = Rational::new(1, 24);
        let video = VideoStreamParams::new(1280, 720, PixelLayout::YUV420P);
        let track = TrackDescriptor::new(TrackId(0), CodecId::Av1, tb, StreamParams::Video(video));
        let mut hdr = ContainerHeader::new("ivf");
        hdr.time_base = tb;
        hdr.tracks.push(track);

        let tmp_path = "/tmp/wedeo_test_ivf_av1_roundtrip.ivf";
        let file_sink = wedeo_core::FileSink::create(tmp_path).unwrap();
        let mut muxer: Box<dyn Muxer> = Box::new(IvfMuxer::new(Box::new(file_sink)));
        muxer.write_header(&hdr).unwrap();

        let kf = make_av1_keyframe_obus();
        let inter = make_av1_interframe_obus();
        for (i, payload) in [&kf, &inter].iter().enumerate() {
            let mut pkt = Packet::new(Buffer::from_slice(payload), TrackId(0), tb);
            pkt.pts = Timestamp::new(i as i64);
            pkt.dts = Timestamp::new(i as i64);
            pkt.duration = 1;
            muxer.write_packet(&pkt).unwrap();
        }
        muxer.finalize().unwrap();

        let bytes = std::fs::read(tmp_path).unwrap();
        std::fs::remove_file(tmp_path).ok();

        let mut demuxer =
            IvfDemuxer::open(Box::new(MemorySource::new(bytes)), ErrorPolicy::Strict).unwrap();
        let RecvStatus::Ready(p0) = demuxer.read_packet().unwrap() else {
            panic!("expected packet 0");
        };
        let RecvStatus::Ready(p1) = demuxer.read_packet().unwrap() else {
            panic!("expected packet 1");
        };
        assert!(p0.flags.contains(PacketFlags::KEY));
        assert!(!p1.flags.contains(PacketFlags::KEY));
        assert!(matches!(demuxer.read_packet().unwrap(), RecvStatus::Eof));
    }

    #[test]
    fn ivf_ffprobe_oracle_and_error_policy() {
        let tb = Rational::new(1, 30);
        let mut video = VideoStreamParams::new(320, 240, PixelLayout::YUV420P);
        video.avg_frame_rate = Rational::new(30, 1);
        let track = TrackDescriptor::new(TrackId(0), CodecId::Vp9, tb, StreamParams::Video(video));
        let mut hdr = ContainerHeader::new("ivf");
        hdr.time_base = tb;
        hdr.tracks.push(track);

        let tmp_path = "/tmp/wedeo_test_ivf_ffprobe.ivf";
        let file_sink = wedeo_core::FileSink::create(tmp_path).unwrap();
        let mut muxer: Box<dyn Muxer> = Box::new(IvfMuxer::new(Box::new(file_sink)));
        muxer.write_header(&hdr).unwrap();

        for i in 0..3 {
            let payload = if i == 0 {
                make_vp9_keyframe_bytes()
            } else {
                make_vp9_interframe_bytes()
            };
            let mut pkt = Packet::new(Buffer::from_slice(&payload), TrackId(0), tb);
            pkt.pts = Timestamp::new(i);
            pkt.dts = Timestamp::new(i);
            pkt.duration = 1;
            muxer.write_packet(&pkt).unwrap();
        }
        muxer.finalize().unwrap();

        let ffprobe = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/ffmpeg-oracle/ffprobe"
        );
        if std::path::Path::new(ffprobe).exists() {
            let out = std::process::Command::new(ffprobe)
                .args([
                    "-v",
                    "error",
                    "-show_format",
                    "-show_streams",
                    "-of",
                    "json",
                    tmp_path,
                ])
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(out.status.success(), "ffprobe failed: {stderr}");
            assert!(stdout.contains("\"codec_name\": \"vp9\""));
            assert!(stdout.contains("\"width\": 320"));
            assert!(stdout.contains("\"height\": 240"));
        }

        let mut bytes = std::fs::read(tmp_path).unwrap();
        std::fs::remove_file(tmp_path).ok();

        // Append a truncated frame header + payload to test Strict vs Resilient policy
        bytes.extend_from_slice(&100u32.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(&[0x82, 0x49, 0x83]); // only 3 bytes instead of 100

        // Strict policy fails on truncated frame
        let mut strict_demux = IvfDemuxer::open(
            Box::new(MemorySource::new(bytes.clone())),
            ErrorPolicy::Strict,
        )
        .unwrap();
        for _ in 0..3 {
            assert!(matches!(
                strict_demux.read_packet().unwrap(),
                RecvStatus::Ready(_)
            ));
        }
        assert!(strict_demux.read_packet().is_err());

        // Resilient policy returns CORRUPT packet and records health event
        let mut res_demux =
            IvfDemuxer::open(Box::new(MemorySource::new(bytes)), ErrorPolicy::Resilient).unwrap();
        for _ in 0..3 {
            assert!(matches!(
                res_demux.read_packet().unwrap(),
                RecvStatus::Ready(_)
            ));
        }
        let RecvStatus::Ready(corrupt_pkt) = res_demux.read_packet().unwrap() else {
            panic!("expected corrupt packet");
        };
        assert!(corrupt_pkt.flags.contains(PacketFlags::CORRUPT));
        assert_eq!(res_demux.health().corrupt_packets, 1);
    }
}
