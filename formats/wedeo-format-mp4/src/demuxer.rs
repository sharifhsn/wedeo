// MP4 demuxer implementation.
//
// Reads standard (non-fragmented) MP4/M4A/M4V files with H.264 video and/or
// AAC audio. Builds a flat per-sample index at header time for O(1) packet
// reads and efficient seeking.
//
// Reference: ISO 14496-12 (ISOBMFF), ISO 14496-14 (MP4 file format).

use wedeo_codec::decoder::CodecParameters;
use wedeo_core::buffer::Buffer;
use wedeo_core::channel_layout::ChannelLayout;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::media_type::MediaType;
use wedeo_core::metadata::Metadata;
use wedeo_core::packet::{Packet, PacketFlags};
use wedeo_core::pixel_format::PixelFormat;
use wedeo_core::rational::Rational;
use wedeo_format::demuxer::{
    Demuxer, DemuxerHeader, InputFormatDescriptor, InputFormatFlags, ProbeData, SeekFlags, Stream,
};
use wedeo_format::io::BufferedIo;
use wedeo_format::registry::DemuxerFactory;

use crate::boxes;

// ---------------------------------------------------------------------------
// Sample index
// ---------------------------------------------------------------------------

/// Pre-computed per-sample entry for O(1) packet reads.
struct SampleEntry {
    /// Byte offset in the file.
    offset: u64,
    /// Byte length of the sample.
    size: u32,
    /// Decode timestamp in media timescale.
    dts: i64,
    /// Composition time offset (PTS = DTS + cts_offset).
    cts_offset: i32,
    /// Sample duration in media timescale.
    duration: u32,
    /// Whether this sample is a sync (keyframe) sample.
    is_keyframe: bool,
}

/// Build the flat sample index from the parsed sample table boxes.
///
/// Combines stts (time-to-sample), stsc (sample-to-chunk), stsz (sample sizes),
/// stco/co64 (chunk offsets), stss (sync samples), and ctts (composition offsets)
/// into a single Vec<SampleEntry>.
fn build_sample_index(
    stts: &[(u32, u32)],
    stsc: &[(u32, u32, u32)],
    fixed_sample_size: u32,
    sample_sizes: &[u32],
    chunk_offsets: &[u64],
    sync_samples: &[u32],
    ctts: &[(u32, i32)],
) -> Vec<SampleEntry> {
    // Count total samples from stts
    let total_samples: u64 = stts.iter().map(|&(count, _)| count as u64).sum();
    let total_samples = total_samples as usize;
    if total_samples == 0 {
        return Vec::new();
    }

    let mut samples = Vec::with_capacity(total_samples);

    // Step 1: Expand stts → cumulative DTS and duration per sample
    let mut dts: i64 = 0;
    for &(count, delta) in stts {
        for _ in 0..count {
            samples.push(SampleEntry {
                offset: 0,
                size: 0,
                dts,
                cts_offset: 0,
                duration: delta,
                is_keyframe: false,
            });
            dts += delta as i64;
        }
    }

    // Step 2: Expand ctts → composition time offsets
    if !ctts.is_empty() {
        let mut sample_idx = 0;
        for &(count, offset) in ctts {
            for _ in 0..count {
                if sample_idx < samples.len() {
                    samples[sample_idx].cts_offset = offset;
                    sample_idx += 1;
                }
            }
        }
    }

    // Step 3: Compute file offsets via stsc + stco + stsz
    //
    // stsc entries are (first_chunk_1indexed, samples_per_chunk, desc_index).
    // They define ranges: entry[i] applies from first_chunk[i] to first_chunk[i+1]-1
    // (or to the last chunk for the final entry).
    let mut sample_idx: usize = 0;

    for (chunk_idx, &chunk_offset) in chunk_offsets.iter().enumerate() {
        let chunk_1indexed = chunk_idx as u32 + 1;

        // Find which stsc entry applies to this chunk.
        // The last entry whose first_chunk <= chunk_1indexed wins.
        let mut samples_in_chunk = 0u32;
        for entry in stsc.iter().rev() {
            if entry.0 <= chunk_1indexed {
                samples_in_chunk = entry.1;
                break;
            }
        }

        let mut offset_in_chunk: u64 = 0;
        for _ in 0..samples_in_chunk {
            if sample_idx >= samples.len() {
                break;
            }
            let size = if fixed_sample_size > 0 {
                fixed_sample_size
            } else if sample_idx < sample_sizes.len() {
                sample_sizes[sample_idx]
            } else {
                0
            };
            samples[sample_idx].offset = chunk_offset + offset_in_chunk;
            samples[sample_idx].size = size;
            offset_in_chunk += size as u64;
            sample_idx += 1;
        }
    }

    // Step 4: Mark keyframes from stss (1-indexed).
    // If stss is absent, all samples are keyframes.
    if sync_samples.is_empty() {
        for s in &mut samples {
            s.is_keyframe = true;
        }
    } else {
        for &sync_idx in sync_samples {
            let idx = sync_idx.saturating_sub(1) as usize;
            if idx < samples.len() {
                samples[idx].is_keyframe = true;
            }
        }
    }

    samples
}

// ---------------------------------------------------------------------------
// Track parsing state (temporary, used during read_header)
// ---------------------------------------------------------------------------

struct TrackParseState {
    handler_type: [u8; 4],
    timescale: u32,
    duration: u64,
    width: u32,
    height: u32,
    stsd_entry: Option<boxes::StsdEntry>,
    stts: Vec<(u32, u32)>,
    stsc: Vec<(u32, u32, u32)>,
    fixed_sample_size: u32,
    sample_sizes: Vec<u32>,
    chunk_offsets: Vec<u64>,
    sync_samples: Vec<u32>,
    ctts: Vec<(u32, i32)>,
}

impl TrackParseState {
    fn new() -> Self {
        Self {
            handler_type: [0; 4],
            timescale: 0,
            duration: 0,
            width: 0,
            height: 0,
            stsd_entry: None,
            stts: Vec::new(),
            stsc: Vec::new(),
            fixed_sample_size: 0,
            sample_sizes: Vec::new(),
            chunk_offsets: Vec::new(),
            sync_samples: Vec::new(),
            ctts: Vec::new(),
        }
    }

    fn is_video(&self) -> bool {
        &self.handler_type == b"vide"
    }

    fn is_audio(&self) -> bool {
        &self.handler_type == b"soun"
    }
}

// ---------------------------------------------------------------------------
// Demuxer state
// ---------------------------------------------------------------------------

struct Mp4TrackState {
    stream_index: usize,
    samples: Vec<SampleEntry>,
    current_sample: usize,
    timescale: u32,
}

pub struct Mp4Demuxer {
    tracks: Vec<Mp4TrackState>,
}

impl Default for Mp4Demuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp4Demuxer {
    pub fn new() -> Self {
        Self { tracks: Vec::new() }
    }
}

impl Demuxer for Mp4Demuxer {
    fn read_header(&mut self, io: &mut BufferedIo) -> Result<DemuxerHeader> {
        let file_size = io.size().unwrap_or(u64::MAX);
        let mut movie_timescale = 1000u32;
        let mut movie_duration = 0u64;
        let mut track_states: Vec<TrackParseState> = Vec::new();

        // Scan top-level boxes until we find moov
        let mut pos: u64 = 0;
        let mut found_moov = false;

        while pos < file_size {
            io.seek(pos)?;
            let header = match boxes::read_box_header(io) {
                Ok(h) => h,
                Err(Error::Eof) => break,
                Err(e) => return Err(e),
            };

            let box_end = if header.size == 0 {
                file_size
            } else {
                pos + header.size
            };

            if &header.box_type == b"moov" {
                found_moov = true;
                parse_moov(
                    io,
                    box_end,
                    &mut movie_timescale,
                    &mut movie_duration,
                    &mut track_states,
                )?;
                break;
            }

            // Skip non-moov boxes (ftyp, mdat, free, etc.)
            pos = box_end;
        }

        if !found_moov {
            return Err(Error::InvalidData);
        }

        // Build streams and sample indices from parsed tracks
        let mut streams = Vec::new();

        for tps in &track_states {
            if !tps.is_video() && !tps.is_audio() {
                continue;
            }

            let stream_index = streams.len();
            let stsd = match &tps.stsd_entry {
                Some(e) => e,
                None => continue,
            };

            let samples = build_sample_index(
                &tps.stts,
                &tps.stsc,
                tps.fixed_sample_size,
                &tps.sample_sizes,
                &tps.chunk_offsets,
                &tps.sync_samples,
                &tps.ctts,
            );

            let nb_frames = samples.len() as i64;

            if tps.is_video() {
                let codec_id = match &stsd.fourcc {
                    b"avc1" | b"avc3" => CodecId::H264,
                    b"av01" => CodecId::Av1,
                    _ => continue,
                };
                let mut params = CodecParameters::new(codec_id, MediaType::Video);
                params.width = tps.width.max(stsd.width as u32);
                params.height = tps.height.max(stsd.height as u32);
                params.pixel_format = PixelFormat::Yuv420p;
                params.extradata = stsd.extradata.clone();
                params.time_base = Rational::new(1, tps.timescale as i32);

                let mut stream = Stream::new(stream_index, params);
                stream.time_base = Rational::new(1, tps.timescale as i32);
                stream.duration = tps.duration as i64;
                stream.nb_frames = nb_frames;
                streams.push(stream);
            } else {
                // Audio
                let codec_id = match &stsd.fourcc {
                    b"mp4a" => CodecId::Aac,
                    _ => continue,
                };
                let mut params = CodecParameters::new(codec_id, MediaType::Audio);
                params.sample_rate = stsd.sample_rate;
                params.channel_layout = if stsd.channel_count == 1 {
                    ChannelLayout::mono()
                } else if stsd.channel_count == 2 {
                    ChannelLayout::stereo()
                } else {
                    ChannelLayout::unspec(stsd.channel_count as i32)
                };
                params.extradata = stsd.extradata.clone();
                params.time_base = Rational::new(1, tps.timescale as i32);

                let mut stream = Stream::new(stream_index, params);
                stream.time_base = Rational::new(1, tps.timescale as i32);
                stream.duration = tps.duration as i64;
                stream.nb_frames = nb_frames;
                streams.push(stream);
            }

            self.tracks.push(Mp4TrackState {
                stream_index,
                samples,
                current_sample: 0,
                timescale: tps.timescale,
            });
        }

        if streams.is_empty() {
            return Err(Error::StreamNotFound);
        }

        // Compute file-level duration in AV_TIME_BASE (1_000_000)
        let duration = if movie_timescale > 0 && movie_duration > 0 {
            (movie_duration as i64 * 1_000_000) / movie_timescale as i64
        } else {
            0
        };

        Ok(DemuxerHeader {
            streams,
            metadata: Metadata::new(),
            duration,
            start_time: 0,
        })
    }

    fn read_packet(&mut self, io: &mut BufferedIo) -> Result<Packet> {
        // Find the track with the lowest DTS among remaining samples
        let mut best_track: Option<usize> = None;
        let mut best_dts: i64 = i64::MAX;

        for (i, track) in self.tracks.iter().enumerate() {
            if track.current_sample >= track.samples.len() {
                continue;
            }
            let sample = &track.samples[track.current_sample];
            // Normalize DTS to a common timebase for comparison.
            // Multiply by 1_000_000 / timescale to get microseconds.
            let dts_us = if track.timescale > 0 {
                sample.dts * 1_000_000 / track.timescale as i64
            } else {
                sample.dts
            };
            if dts_us < best_dts {
                best_dts = dts_us;
                best_track = Some(i);
            }
        }

        let track_idx = best_track.ok_or(Error::Eof)?;
        let track = &mut self.tracks[track_idx];
        let sample = &track.samples[track.current_sample];

        // Seek to sample position and read data
        io.seek(sample.offset)?;
        let data = io.read_bytes(sample.size as usize)?;

        let mut pkt = Packet::new();
        pkt.data = Buffer::from_slice(&data);
        pkt.dts = sample.dts;
        pkt.pts = sample.dts + sample.cts_offset as i64;
        pkt.duration = sample.duration as i64;
        pkt.stream_index = track.stream_index;
        pkt.pos = sample.offset as i64;
        if sample.is_keyframe {
            pkt.flags = PacketFlags::KEY;
        }

        track.current_sample += 1;
        Ok(pkt)
    }

    fn seek(
        &mut self,
        _io: &mut BufferedIo,
        stream_index: usize,
        timestamp: i64,
        _flags: SeekFlags,
    ) -> Result<()> {
        // Find the target track
        let track = self
            .tracks
            .iter_mut()
            .find(|t| t.stream_index == stream_index)
            .ok_or(Error::StreamNotFound)?;

        // Binary search for the nearest keyframe at or before the target timestamp
        let target_sample = match track.samples.binary_search_by_key(&timestamp, |s| s.dts) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };

        // Walk backwards to find a keyframe
        let mut keyframe_idx = target_sample;
        while keyframe_idx > 0 && !track.samples[keyframe_idx].is_keyframe {
            keyframe_idx -= 1;
        }

        track.current_sample = keyframe_idx;

        // Reset other tracks to approximately the same time position
        let target_dts = track.samples[keyframe_idx].dts;
        let target_timescale = track.timescale;

        for other in self.tracks.iter_mut() {
            if other.stream_index == stream_index {
                continue;
            }
            // Convert target_dts to other track's timescale
            let other_dts = if target_timescale > 0 && other.timescale > 0 {
                target_dts * other.timescale as i64 / target_timescale as i64
            } else {
                target_dts
            };

            // Find the sample at or before other_dts
            let idx = match other.samples.binary_search_by_key(&other_dts, |s| s.dts) {
                Ok(i) => i,
                Err(i) => i.saturating_sub(1),
            };
            other.current_sample = idx;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// moov box recursive parser
// ---------------------------------------------------------------------------

fn parse_moov(
    io: &mut BufferedIo,
    end_pos: u64,
    movie_timescale: &mut u32,
    movie_duration: &mut u64,
    tracks: &mut Vec<TrackParseState>,
) -> Result<()> {
    while io.tell()? + 8 <= end_pos {
        let header = boxes::read_box_header(io)?;
        let box_end = io.tell()?.saturating_sub(header.header_size as u64) + header.size;
        let box_end = box_end.min(end_pos);

        match &header.box_type {
            b"mvhd" => {
                let mvhd = boxes::parse_mvhd(io)?;
                *movie_timescale = mvhd.timescale;
                *movie_duration = mvhd.duration;
            }
            b"trak" => {
                let mut tps = TrackParseState::new();
                parse_trak(io, box_end, &mut tps)?;
                tracks.push(tps);
            }
            _ => {}
        }

        // Ensure we're at the box end
        let current = io.tell()?;
        if current < box_end {
            io.skip(box_end - current)?;
        }
    }
    Ok(())
}

fn parse_trak(io: &mut BufferedIo, end_pos: u64, tps: &mut TrackParseState) -> Result<()> {
    while io.tell()? + 8 <= end_pos {
        let header = boxes::read_box_header(io)?;
        let box_end = io.tell()?.saturating_sub(header.header_size as u64) + header.size;
        let box_end = box_end.min(end_pos);

        match &header.box_type {
            b"tkhd" => {
                let tkhd = boxes::parse_tkhd(io)?;
                tps.width = tkhd.width;
                tps.height = tkhd.height;
            }
            b"mdia" => {
                parse_mdia(io, box_end, tps)?;
            }
            _ => {}
        }

        let current = io.tell()?;
        if current < box_end {
            io.skip(box_end - current)?;
        }
    }
    Ok(())
}

fn parse_mdia(io: &mut BufferedIo, end_pos: u64, tps: &mut TrackParseState) -> Result<()> {
    while io.tell()? + 8 <= end_pos {
        let header = boxes::read_box_header(io)?;
        let box_end = io.tell()?.saturating_sub(header.header_size as u64) + header.size;
        let box_end = box_end.min(end_pos);
        let payload_size = header.payload_size().unwrap_or(0);

        match &header.box_type {
            b"mdhd" => {
                let mdhd = boxes::parse_mdhd(io)?;
                tps.timescale = mdhd.timescale;
                tps.duration = mdhd.duration;
            }
            b"hdlr" => {
                let hdlr = boxes::parse_hdlr(io, payload_size)?;
                tps.handler_type = hdlr.handler_type;
            }
            b"minf" => {
                parse_minf(io, box_end, tps)?;
            }
            _ => {}
        }

        let current = io.tell()?;
        if current < box_end {
            io.skip(box_end - current)?;
        }
    }
    Ok(())
}

fn parse_minf(io: &mut BufferedIo, end_pos: u64, tps: &mut TrackParseState) -> Result<()> {
    while io.tell()? + 8 <= end_pos {
        let header = boxes::read_box_header(io)?;
        let box_end = io.tell()?.saturating_sub(header.header_size as u64) + header.size;
        let box_end = box_end.min(end_pos);

        if &header.box_type == b"stbl" {
            parse_stbl(io, box_end, tps)?;
        }

        let current = io.tell()?;
        if current < box_end {
            io.skip(box_end - current)?;
        }
    }
    Ok(())
}

fn parse_stbl(io: &mut BufferedIo, end_pos: u64, tps: &mut TrackParseState) -> Result<()> {
    while io.tell()? + 8 <= end_pos {
        let header = boxes::read_box_header(io)?;
        let box_end = io.tell()?.saturating_sub(header.header_size as u64) + header.size;
        let box_end = box_end.min(end_pos);
        let payload_size = header.payload_size().unwrap_or(0);

        match &header.box_type {
            b"stsd" => {
                tps.stsd_entry = Some(boxes::parse_stsd(io, payload_size)?);
            }
            b"stts" => {
                tps.stts = boxes::parse_stts(io)?;
            }
            b"stsc" => {
                tps.stsc = boxes::parse_stsc(io)?;
            }
            b"stsz" => {
                let (fixed, per_sample) = boxes::parse_stsz(io)?;
                tps.fixed_sample_size = fixed;
                tps.sample_sizes = per_sample;
            }
            b"stco" => {
                tps.chunk_offsets = boxes::parse_stco(io)?;
            }
            b"co64" => {
                tps.chunk_offsets = boxes::parse_co64(io)?;
            }
            b"stss" => {
                tps.sync_samples = boxes::parse_stss(io)?;
            }
            b"ctts" => {
                tps.ctts = boxes::parse_ctts(io)?;
            }
            _ => {
                let current = io.tell()?;
                if current < box_end {
                    io.skip(box_end - current)?;
                }
            }
        }

        // Ensure we're at the box end
        let current = io.tell()?;
        if current < box_end {
            io.skip(box_end - current)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Probe function
// ---------------------------------------------------------------------------

/// Probe score for MP4 files. Returns 0-100.
fn probe_mp4(data: &ProbeData<'_>) -> i32 {
    let buf = data.buf;
    if buf.len() < 8 {
        return 0;
    }

    // Check for ftyp box at offset 0
    if buf.len() >= 8 && &buf[4..8] == b"ftyp" {
        return 90;
    }

    // Check for moov/mdat/free/wide at offset 4..8 (some files start with these)
    if buf.len() >= 8
        && (&buf[4..8] == b"moov"
            || &buf[4..8] == b"mdat"
            || &buf[4..8] == b"free"
            || &buf[4..8] == b"wide")
    {
        return 70;
    }

    // Extension-based fallback
    let filename = data.filename.to_lowercase();
    if filename.ends_with(".mp4")
        || filename.ends_with(".m4a")
        || filename.ends_with(".m4v")
        || filename.ends_with(".mov")
    {
        return 50;
    }

    0
}

// ---------------------------------------------------------------------------
// Factory registration
// ---------------------------------------------------------------------------

struct Mp4DemuxerFactory;

impl DemuxerFactory for Mp4DemuxerFactory {
    fn descriptor(&self) -> &InputFormatDescriptor {
        static DESC: InputFormatDescriptor = InputFormatDescriptor {
            name: "mp4",
            long_name: "MP4 (MPEG-4 Part 14)",
            extensions: "mp4,m4a,m4v,mov",
            mime_types: "video/mp4,audio/mp4",
            flags: InputFormatFlags::empty(),
            priority: 100,
        };
        &DESC
    }

    fn probe(&self, data: &ProbeData<'_>) -> i32 {
        probe_mp4(data)
    }

    fn create(&self) -> Result<Box<dyn Demuxer>> {
        Ok(Box::new(Mp4Demuxer::new()))
    }
}

inventory::submit!(&Mp4DemuxerFactory as &dyn DemuxerFactory);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::muxer::Mp4Muxer;
    use wedeo_codec::decoder::CodecParameters;
    use wedeo_core::buffer::Buffer;
    use wedeo_core::rational::Rational;
    use wedeo_format::io::{BufferedIo, FileIo};
    use wedeo_format::muxer::Muxer;

    /// Minimal avcC extradata for tests.
    fn test_avcc() -> Vec<u8> {
        vec![
            0x01, 0x42, 0xC0, 0x1E, 0xFF, 0xE1, 0x00, 0x04, 0x67, 0x42, 0xC0, 0x1E, 0x01, 0x00,
            0x02, 0x68, 0xCE,
        ]
    }

    /// Helper: mux an MP4 file with the given video packets and return the path.
    fn mux_test_mp4(
        path: &str,
        width: u32,
        height: u32,
        packets: &[(Vec<u8>, i64, i64, i64, bool)], // (data, pts, dts, duration, is_key)
    ) {
        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = width;
        params.height = height;
        params.time_base = Rational::new(1, 25);
        params.extradata = test_avcc();
        let stream = Stream::new(0, params);

        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));
        muxer.write_header(&mut io, &[stream]).unwrap();

        for (data, pts, dts, duration, is_key) in packets {
            let mut pkt = Packet::new();
            pkt.data = Buffer::from_slice(data);
            pkt.stream_index = 0;
            pkt.pts = *pts;
            pkt.dts = *dts;
            pkt.duration = *duration;
            if *is_key {
                pkt.flags = PacketFlags::KEY;
            }
            muxer.write_packet(&mut io, &pkt).unwrap();
        }

        muxer.write_trailer(&mut io).unwrap();
    }

    #[test]
    fn test_demux_roundtrip() {
        let path = "/tmp/wedeo_test_mp4_demux_roundtrip.mp4";

        // Write 5 packets with varying sizes
        let packets: Vec<(Vec<u8>, i64, i64, i64, bool)> = (0..5)
            .map(|i| (vec![0xAA; 100 + i * 20], i as i64, i as i64, 1i64, i == 0))
            .collect();
        mux_test_mp4(path, 320, 240, &packets);

        // Now demux it
        let mut demuxer = Mp4Demuxer::new();
        let file_io = FileIo::open(path).expect("open file");
        let mut io = BufferedIo::new(Box::new(file_io));
        let header = demuxer.read_header(&mut io).unwrap();

        // Verify stream info
        assert_eq!(header.streams.len(), 1);
        assert_eq!(header.streams[0].codec_params.codec_id, CodecId::H264);
        assert_eq!(header.streams[0].codec_params.width, 320);
        assert_eq!(header.streams[0].codec_params.height, 240);
        assert!(!header.streams[0].codec_params.extradata.is_empty());

        // Read all packets back
        let mut read_packets = Vec::new();
        loop {
            match demuxer.read_packet(&mut io) {
                Ok(pkt) => read_packets.push(pkt),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }

        // Verify packet count and sizes
        assert_eq!(read_packets.len(), 5);
        for (i, pkt) in read_packets.iter().enumerate() {
            assert_eq!(pkt.data.size(), 100 + i * 20, "packet {i} size mismatch");
            assert_eq!(pkt.stream_index, 0);
            assert_eq!(pkt.dts, i as i64);
            assert_eq!(pkt.duration, 1);
        }

        // First packet should be keyframe
        assert!(read_packets[0].flags.contains(PacketFlags::KEY));
        // Others should not
        for pkt in &read_packets[1..] {
            assert!(!pkt.flags.contains(PacketFlags::KEY));
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_demux_probe() {
        // ftyp header → score 90
        let mut ftyp_buf = vec![0u8; 32];
        ftyp_buf[4..8].copy_from_slice(b"ftyp");
        let probe = ProbeData {
            filename: "test.mp4",
            buf: &ftyp_buf,
        };
        assert_eq!(probe_mp4(&probe), 90);

        // moov header → score 70
        let mut moov_buf = vec![0u8; 32];
        moov_buf[4..8].copy_from_slice(b"moov");
        let probe = ProbeData {
            filename: "test.bin",
            buf: &moov_buf,
        };
        assert_eq!(probe_mp4(&probe), 70);

        // Extension only → score 50
        let probe = ProbeData {
            filename: "test.mp4",
            buf: &[0u8; 32],
        };
        assert_eq!(probe_mp4(&probe), 50);

        // No match → score 0
        let probe = ProbeData {
            filename: "test.wav",
            buf: b"RIFF....WAVE",
        };
        assert_eq!(probe_mp4(&probe), 0);
    }

    #[test]
    fn test_demux_seek() {
        let path = "/tmp/wedeo_test_mp4_demux_seek.mp4";

        // Write packets with keyframes at 0 and 3
        let packets: Vec<(Vec<u8>, i64, i64, i64, bool)> = (0..6)
            .map(|i| {
                let is_key = i == 0 || i == 3;
                (vec![i as u8; 50], i as i64, i as i64, 1i64, is_key)
            })
            .collect();
        mux_test_mp4(path, 320, 240, &packets);

        let mut demuxer = Mp4Demuxer::new();
        let file_io = FileIo::open(path).expect("open file");
        let mut io = BufferedIo::new(Box::new(file_io));
        demuxer.read_header(&mut io).unwrap();

        // Seek to DTS=4 — should snap back to keyframe at DTS=3
        demuxer.seek(&mut io, 0, 4, SeekFlags::BACKWARD).unwrap();
        let pkt = demuxer.read_packet(&mut io).unwrap();
        assert_eq!(pkt.dts, 3);
        assert!(pkt.flags.contains(PacketFlags::KEY));

        // Seek to DTS=0 — should be at the beginning
        demuxer.seek(&mut io, 0, 0, SeekFlags::BACKWARD).unwrap();
        let pkt = demuxer.read_packet(&mut io).unwrap();
        assert_eq!(pkt.dts, 0);
        assert!(pkt.flags.contains(PacketFlags::KEY));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_demux_cts_offsets() {
        let path = "/tmp/wedeo_test_mp4_demux_cts.mp4";

        // Mux with CTS offsets (PTS != DTS for B-frame reordering)
        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        params.extradata = test_avcc();
        let stream = Stream::new(0, params);

        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));
        muxer.write_header(&mut io, &[stream]).unwrap();

        // DTS order: 0, 1, 2, 3
        // PTS order: 1, 0, 3, 2 (simulating B-frame reorder)
        let frame_data: Vec<(i64, i64, bool)> = vec![
            (1, 0, true),  // I-frame: pts=1, dts=0
            (0, 1, false), // B-frame: pts=0, dts=1
            (3, 2, false), // P-frame: pts=3, dts=2
            (2, 3, false), // B-frame: pts=2, dts=3
        ];

        for (pts, dts, is_key) in &frame_data {
            let mut pkt = Packet::new();
            pkt.data = Buffer::from_slice(&[0u8; 64]);
            pkt.stream_index = 0;
            pkt.pts = *pts;
            pkt.dts = *dts;
            pkt.duration = 1;
            if *is_key {
                pkt.flags = PacketFlags::KEY;
            }
            muxer.write_packet(&mut io, &pkt).unwrap();
        }

        muxer.write_trailer(&mut io).unwrap();
        drop(io);

        // Demux and verify PTS/DTS
        let mut demuxer = Mp4Demuxer::new();
        let file_io = FileIo::open(path).expect("open file");
        let mut io = BufferedIo::new(Box::new(file_io));
        demuxer.read_header(&mut io).unwrap();

        let mut read_pkts = Vec::new();
        loop {
            match demuxer.read_packet(&mut io) {
                Ok(pkt) => read_pkts.push(pkt),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }

        assert_eq!(read_pkts.len(), 4);
        // Verify DTS is monotonically increasing
        for (i, pkt) in read_pkts.iter().enumerate() {
            assert_eq!(pkt.dts, i as i64, "dts mismatch at {i}");
        }
        // Verify PTS matches what we wrote
        assert_eq!(read_pkts[0].pts, 1); // I: pts=1, dts=0
        assert_eq!(read_pkts[1].pts, 0); // B: pts=0, dts=1
        assert_eq!(read_pkts[2].pts, 3); // P: pts=3, dts=2
        assert_eq!(read_pkts[3].pts, 2); // B: pts=2, dts=3

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_demux_faststart() {
        let path = "/tmp/wedeo_test_mp4_demux_faststart.mp4";

        // Use faststart (moov before mdat)
        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 160;
        params.height = 120;
        params.time_base = Rational::new(1, 30);
        params.extradata = test_avcc();
        let stream = Stream::new(0, params);

        let mut muxer =
            crate::muxer::Mp4Muxer::with_options(crate::muxer::Mp4MuxerOptions { faststart: true });
        let file_io = FileIo::create(path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));
        muxer.write_header(&mut io, &[stream]).unwrap();

        for i in 0..3 {
            let mut pkt = Packet::new();
            pkt.data = Buffer::from_slice(&[0u8; 80]);
            pkt.stream_index = 0;
            pkt.pts = i;
            pkt.dts = i;
            pkt.duration = 1;
            if i == 0 {
                pkt.flags = PacketFlags::KEY;
            }
            muxer.write_packet(&mut io, &pkt).unwrap();
        }

        muxer.write_trailer(&mut io).unwrap();
        drop(io);

        // Demux — faststart puts moov before mdat, demuxer should handle both layouts
        let mut demuxer = Mp4Demuxer::new();
        let file_io = FileIo::open(path).expect("open file");
        let mut io = BufferedIo::new(Box::new(file_io));
        let header = demuxer.read_header(&mut io).unwrap();

        assert_eq!(header.streams.len(), 1);
        assert_eq!(header.streams[0].codec_params.width, 160);
        assert_eq!(header.streams[0].codec_params.height, 120);

        let mut count = 0;
        while demuxer.read_packet(&mut io).is_ok() {
            count += 1;
        }
        assert_eq!(count, 3);

        std::fs::remove_file(path).ok();
    }
}
