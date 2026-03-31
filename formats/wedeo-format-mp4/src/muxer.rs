// MP4 muxer implementation.
//
// Two-pass approach: mdat is written first (incrementally), moov is written at
// the end with all sample tables. Requires a seekable output.

use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::{Packet, PacketFlags};
use wedeo_format::demuxer::Stream;
use wedeo_format::io::BufferedIo;
use wedeo_format::muxer::Muxer;

use crate::atoms;

/// Per-track state accumulated during muxing.
pub struct TrackState {
    pub stream_index: usize,
    pub codec_id: CodecId,
    pub media_type: MediaType,
    pub timescale: u32,

    // Video
    pub width: u32,
    pub height: u32,

    // Audio
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub bit_rate: i64,

    pub extradata: Vec<u8>,

    // Sample table data (accumulated during write_packet)
    pub sample_count: u32,
    pub sample_sizes: Vec<u32>,
    pub sample_durations: Vec<u32>,
    pub chunk_offsets: Vec<u64>,
    pub sync_samples: Vec<u32>,
    pub cts_offsets: Vec<i32>,
    pub has_cts: bool,

    pub duration_ts: u64,
}

/// Options for controlling MP4 muxer behavior.
#[derive(Default)]
pub struct Mp4MuxerOptions {
    /// Place moov before mdat for progressive playback (equivalent to `-movflags +faststart`).
    pub faststart: bool,
}

/// MP4 muxer state.
pub struct Mp4Muxer {
    tracks: Vec<TrackState>,
    /// File offset where the mdat box starts (at the size field).
    mdat_pos: u64,
    /// File offset of the 64-bit largesize field in the mdat header.
    mdat_largesize_pos: u64,
    /// Movie-level timescale (typically 1000).
    movie_timescale: u32,
    /// Enable moov-before-mdat relocation.
    faststart: bool,
}

impl Default for Mp4Muxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp4Muxer {
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            mdat_pos: 0,
            mdat_largesize_pos: 0,
            movie_timescale: 1000,
            faststart: false,
        }
    }

    pub fn with_options(options: Mp4MuxerOptions) -> Self {
        Self {
            tracks: Vec::new(),
            mdat_pos: 0,
            mdat_largesize_pos: 0,
            movie_timescale: 1000,
            faststart: options.faststart,
        }
    }

    fn find_track(&self, stream_index: usize) -> Option<usize> {
        self.tracks
            .iter()
            .position(|t| t.stream_index == stream_index)
    }
}

impl Muxer for Mp4Muxer {
    fn write_header(&mut self, io: &mut BufferedIo, streams: &[Stream]) -> Result<()> {
        if streams.is_empty() {
            return Err(Error::StreamNotFound);
        }

        // Two-pass muxing requires seek to patch mdat size
        if !io.is_seekable() {
            return Err(Error::Other("MP4 muxer requires seekable output".into()));
        }

        // Validate codecs and initialize track state
        let has_video = streams
            .iter()
            .any(|s| s.codec_params.media_type == MediaType::Video);
        let has_audio = streams
            .iter()
            .any(|s| s.codec_params.media_type == MediaType::Audio);

        for stream in streams {
            let params = &stream.codec_params;

            if !is_supported_codec(params.codec_id) {
                return Err(Error::PatchwelcomeNotImplemented);
            }

            let timescale = compute_timescale(params);

            self.tracks.push(TrackState {
                stream_index: stream.index,
                codec_id: params.codec_id,
                media_type: params.media_type,
                timescale,
                width: params.width,
                height: params.height,
                sample_rate: params.sample_rate,
                channels: params.channel_layout.nb_channels as u16,
                bits_per_sample: params.bits_per_coded_sample as u16,
                bit_rate: params.bit_rate,
                extradata: params.extradata.clone(),
                sample_count: 0,
                sample_sizes: Vec::new(),
                sample_durations: Vec::new(),
                chunk_offsets: Vec::new(),
                sync_samples: Vec::new(),
                cts_offsets: Vec::new(),
                has_cts: false,
                duration_ts: 0,
            });
        }

        // Validate extradata requirements
        for track in &self.tracks {
            match track.codec_id {
                CodecId::H264 => {
                    if track.extradata.is_empty() {
                        return Err(Error::InvalidArgument);
                    }
                    if track.extradata[0] != 0x01 {
                        return Err(Error::InvalidData);
                    }
                }
                CodecId::Av1 if track.extradata.is_empty() => {
                    return Err(Error::InvalidArgument);
                }
                CodecId::Aac if track.extradata.is_empty() => {
                    return Err(Error::InvalidArgument);
                }
                _ => {}
            }
        }

        // Write ftyp
        let ftyp = atoms::write_ftyp(has_video, has_audio);
        io.write_bytes(&ftyp)?;

        // Write extended mdat header (16 bytes: size=1 + "mdat" + largesize placeholder)
        self.mdat_pos = io.tell()?;
        io.write_u32be(1)?; // size = 1 means "use largesize"
        io.write_bytes(b"mdat")?;
        self.mdat_largesize_pos = io.tell()?;
        io.write_bytes(&0u64.to_be_bytes())?; // largesize placeholder

        io.flush()?;
        Ok(())
    }

    fn write_packet(&mut self, io: &mut BufferedIo, packet: &Packet) -> Result<()> {
        let track_idx = self
            .find_track(packet.stream_index)
            .ok_or(Error::StreamNotFound)?;

        let offset = io.tell()?;
        let data = packet.data.data();
        io.write_bytes(data)?;

        let track = &mut self.tracks[track_idx];
        track.sample_count += 1;
        track.sample_sizes.push(data.len() as u32);
        track.chunk_offsets.push(offset);

        // Duration: use packet.duration if available, otherwise default to 1
        let duration = if packet.duration > 0 {
            packet.duration as u32
        } else {
            1
        };
        track.sample_durations.push(duration);
        track.duration_ts += duration as u64;

        // Sync samples (keyframes) — 1-indexed
        if packet.flags.contains(PacketFlags::KEY) {
            track.sync_samples.push(track.sample_count);
        }

        // CTS offset (pts - dts)
        let cts_offset = if packet.pts != i64::MIN && packet.dts != i64::MIN {
            (packet.pts - packet.dts) as i32
        } else {
            0
        };
        track.cts_offsets.push(cts_offset);
        if cts_offset != 0 {
            track.has_cts = true;
        }

        Ok(())
    }

    fn write_trailer(&mut self, io: &mut BufferedIo) -> Result<()> {
        io.flush()?;

        let mdat_end = io.tell()?;
        let mdat_size = mdat_end - self.mdat_pos;

        // Patch 64-bit largesize in the extended mdat header
        io.seek(self.mdat_largesize_pos)?;
        io.write_bytes(&mdat_size.to_be_bytes())?;
        io.flush()?;

        if self.faststart {
            // Build moov with adjusted chunk offsets (shifted forward by moov size)
            let moov = self.build_faststart_moov()?;
            let moov_size = moov.len() as u64;

            // Shift mdat data forward by moov_size using backwards chunk copy
            shift_data_forward(io, self.mdat_pos, mdat_size, moov_size)?;

            // Write moov into the gap before mdat
            io.seek(self.mdat_pos)?;
            io.write_bytes(&moov)?;
            io.flush()?;
        } else {
            // Standard layout: moov at end
            io.seek(mdat_end)?;
            let moov = atoms::write_moov(&self.tracks, self.movie_timescale);
            io.write_bytes(&moov)?;
            io.flush()?;
        }

        Ok(())
    }
}

impl Mp4Muxer {
    /// Build moov with chunk offsets adjusted for faststart relocation.
    ///
    /// Handles stco→co64 size convergence: adding offset bytes may push offsets
    /// past u32::MAX, upgrading stco to co64 and changing moov size. Converges
    /// in at most 2 iterations since co64 is the maximum size.
    fn build_faststart_moov(&mut self) -> Result<Vec<u8>> {
        // Save original chunk offsets
        let original_offsets: Vec<Vec<u64>> = self
            .tracks
            .iter()
            .map(|t| t.chunk_offsets.clone())
            .collect();

        // Iteration 1: trial build to get moov size
        let trial_moov = atoms::write_moov(&self.tracks, self.movie_timescale);
        let mut moov_size = trial_moov.len() as u64;

        for max_iter in 0..2 {
            // Adjust all chunk offsets by +moov_size
            for (track, orig) in self.tracks.iter_mut().zip(original_offsets.iter()) {
                track.chunk_offsets = orig.iter().map(|&o| o + moov_size).collect();
            }

            let moov = atoms::write_moov(&self.tracks, self.movie_timescale);
            let new_size = moov.len() as u64;

            if new_size == moov_size || max_iter == 1 {
                return Ok(moov);
            }

            // Size changed (stco upgraded to co64), retry with new size
            moov_size = new_size;
        }

        unreachable!()
    }
}

/// Copy `len` bytes from position `start` to `start + shift`, working backwards
/// in chunks to avoid overwriting unread data.
fn shift_data_forward(io: &mut BufferedIo, start: u64, len: u64, shift: u64) -> Result<()> {
    if len == 0 || shift == 0 {
        return Ok(());
    }

    const CHUNK_SIZE: u64 = 8 * 1024 * 1024; // 8 MiB
    let mut buf = vec![0u8; CHUNK_SIZE as usize];
    let mut remaining = len;

    while remaining > 0 {
        let chunk = remaining.min(CHUNK_SIZE);
        let read_pos = start + remaining - chunk;

        io.seek(read_pos)?;
        io.read_exact(&mut buf[..chunk as usize])?;

        io.seek(read_pos + shift)?;
        io.write_bytes(&buf[..chunk as usize])?;
        io.flush()?;

        remaining -= chunk;
    }

    Ok(())
}

/// Check if a codec is supported by the MP4 muxer.
fn is_supported_codec(codec_id: CodecId) -> bool {
    matches!(
        codec_id,
        CodecId::H264
            | CodecId::Av1
            | CodecId::Aac
            | CodecId::PcmS16be
            | CodecId::PcmS16le
            | CodecId::PcmS24be
            | CodecId::PcmS24le
            | CodecId::PcmS32be
            | CodecId::PcmS32le
            | CodecId::PcmF32be
            | CodecId::PcmF32le
            | CodecId::PcmF64be
            | CodecId::PcmF64le
    )
}

/// Compute media timescale for a track from its codec parameters.
fn compute_timescale(params: &wedeo_codec::decoder::CodecParameters) -> u32 {
    match params.media_type {
        MediaType::Audio => {
            if params.sample_rate > 0 {
                params.sample_rate
            } else {
                44100 // fallback
            }
        }
        MediaType::Video => {
            // Use the stream time_base denominator if sensible
            if params.time_base.den > 0 {
                params.time_base.den as u32
            } else {
                90000 // fallback (common for MPEG-TS / H.264)
            }
        }
        _ => 1000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_codec::decoder::CodecParameters;
    use wedeo_core::buffer::Buffer;
    use wedeo_core::rational::Rational;
    use wedeo_format::io::{BufferedIo, FileIo};

    /// Roundtrip test: create a minimal H.264 MP4, verify it has the expected
    /// box structure (ftyp, mdat, moov).
    #[test]
    fn test_mp4_muxer_basic_structure() {
        let tmp_path = "/tmp/wedeo_test_mp4_muxer.mp4";

        // Set up a video stream
        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25); // 25 fps
        // Minimal avcC extradata (version 1, profile 66 baseline, level 30)
        params.extradata = vec![
            0x01, 0x42, 0xC0, 0x1E, 0xFF,
            0xE1, // configurationVersion, profile, compat, level, NALU len - 1, numSPS
            0x00, 0x04, // SPS length
            0x67, 0x42, 0xC0, 0x1E, // fake SPS
            0x01, // numPPS
            0x00, 0x02, // PPS length
            0x68, 0xCE, // fake PPS
        ];

        let stream = Stream::new(0, params);

        // Create muxer and write
        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));

        muxer
            .write_header(&mut io, &[stream])
            .expect("write_header");

        // Write 3 packets (keyframe, non-key, non-key)
        for i in 0..3 {
            let data = vec![0u8; 100 + i * 10]; // varying sizes
            let mut pkt = Packet::new();
            pkt.data = Buffer::from_slice(&data);
            pkt.stream_index = 0;
            pkt.pts = i as i64;
            pkt.dts = i as i64;
            pkt.duration = 1;
            if i == 0 {
                pkt.flags = PacketFlags::KEY;
            }
            muxer.write_packet(&mut io, &pkt).expect("write_packet");
        }

        muxer.write_trailer(&mut io).expect("write_trailer");
        drop(io);

        // Read back and verify box structure
        let data = std::fs::read(tmp_path).expect("read file");
        std::fs::remove_file(tmp_path).ok();

        // Parse top-level boxes
        let boxes = parse_top_level_boxes(&data);
        let box_types: Vec<&str> = boxes.iter().map(|(t, _, _)| t.as_str()).collect();
        assert_eq!(box_types, vec!["ftyp", "mdat", "moov"]);

        // Verify ftyp contains isom brand
        let (_, ftyp_start, ftyp_size) = &boxes[0];
        let ftyp_data = &data[*ftyp_start..*ftyp_start + *ftyp_size];
        assert_eq!(&ftyp_data[8..12], b"isom"); // major brand

        // Verify mdat size accounts for packet data (100 + 110 + 120 = 330 + 16 extended header)
        let (_, _, mdat_size) = &boxes[1];
        assert_eq!(*mdat_size, 330 + 16);

        // Verify moov has mvhd and trak inside
        let (_, moov_start, moov_size) = &boxes[2];
        let moov_data = &data[*moov_start + 8..*moov_start + *moov_size];
        let inner_boxes = parse_top_level_boxes(moov_data);
        let inner_types: Vec<&str> = inner_boxes.iter().map(|(t, _, _)| t.as_str()).collect();
        assert_eq!(inner_types, vec!["mvhd", "trak"]);
    }

    /// Verify ffprobe can read the MP4 we produce.
    #[test]
    fn test_mp4_ffprobe_validates() {
        let tmp_path = "/tmp/wedeo_test_mp4_ffprobe.mp4";

        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        // Minimal avcC
        params.extradata = vec![
            0x01, 0x42, 0xC0, 0x1E, 0xFF, 0xE1, 0x00, 0x04, 0x67, 0x42, 0xC0, 0x1E, 0x01, 0x00,
            0x02, 0x68, 0xCE,
        ];

        let stream = Stream::new(0, params);
        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));

        muxer.write_header(&mut io, &[stream]).unwrap();

        for i in 0..5 {
            let data = vec![0u8; 200];
            let mut pkt = Packet::new();
            pkt.data = Buffer::from_slice(&data);
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

        // Run ffprobe on the output
        let output = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                tmp_path,
            ])
            .output();

        std::fs::remove_file(tmp_path).ok();

        match output {
            Ok(result) => {
                let stdout = String::from_utf8_lossy(&result.stdout);
                let stderr = String::from_utf8_lossy(&result.stderr);
                assert!(result.status.success(), "ffprobe failed: {}", stderr);
                // Verify it detected the video stream
                assert!(stdout.contains("\"codec_type\": \"video\""));
                assert!(stdout.contains("\"width\": 320"));
                assert!(stdout.contains("\"height\": 240"));
            }
            Err(_) => {
                // ffprobe not available — skip silently
            }
        }
    }

    /// Non-seekable I/O for testing the seekable guard.
    struct NonSeekableIo(Vec<u8>);

    impl wedeo_format::io::IoContext for NonSeekableIo {
        fn read(&mut self, _buf: &mut [u8]) -> wedeo_core::error::Result<usize> {
            Ok(0)
        }
        fn write(&mut self, buf: &[u8]) -> wedeo_core::error::Result<usize> {
            self.0.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn seek(&mut self, _pos: std::io::SeekFrom) -> wedeo_core::error::Result<u64> {
            Err(Error::Other("not seekable".into()))
        }
        fn tell(&mut self) -> wedeo_core::error::Result<u64> {
            Ok(self.0.len() as u64)
        }
        fn size(&mut self) -> wedeo_core::error::Result<u64> {
            Ok(self.0.len() as u64)
        }
        fn is_seekable(&self) -> bool {
            false
        }
    }

    /// Minimal avcC extradata for tests.
    fn test_avcc() -> Vec<u8> {
        vec![
            0x01, 0x42, 0xC0, 0x1E, 0xFF, 0xE1, 0x00, 0x04, 0x67, 0x42, 0xC0, 0x1E, 0x01, 0x00,
            0x02, 0x68, 0xCE,
        ]
    }

    #[test]
    fn test_reject_non_seekable() {
        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        params.extradata = test_avcc();
        let stream = Stream::new(0, params);

        let mut muxer = Mp4Muxer::new();
        let mut io = BufferedIo::new(Box::new(NonSeekableIo(Vec::new())));
        let err = muxer.write_header(&mut io, &[stream]).unwrap_err();
        assert!(matches!(err, Error::Other(_)));
    }

    #[test]
    fn test_reject_unsupported_codec() {
        let mut params = CodecParameters::new(CodecId::Vp9, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        let stream = Stream::new(0, params);

        let tmp_path = "/tmp/wedeo_test_reject_codec.mp4";
        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));
        let err = muxer.write_header(&mut io, &[stream]).unwrap_err();
        std::fs::remove_file(tmp_path).ok();
        assert_eq!(err, Error::PatchwelcomeNotImplemented);
    }

    #[test]
    fn test_reject_missing_extradata() {
        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        // No extradata
        let stream = Stream::new(0, params);

        let tmp_path = "/tmp/wedeo_test_reject_noextra.mp4";
        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));
        let err = muxer.write_header(&mut io, &[stream]).unwrap_err();
        std::fs::remove_file(tmp_path).ok();
        assert_eq!(err, Error::InvalidArgument);
    }

    #[test]
    fn test_reject_invalid_avcc() {
        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        // Annex B start code instead of avcC (version byte != 0x01)
        params.extradata = vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42];
        let stream = Stream::new(0, params);

        let tmp_path = "/tmp/wedeo_test_reject_annexb.mp4";
        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));
        let err = muxer.write_header(&mut io, &[stream]).unwrap_err();
        std::fs::remove_file(tmp_path).ok();
        assert_eq!(err, Error::InvalidData);
    }

    #[test]
    fn test_96khz_audio() {
        let tmp_path = "/tmp/wedeo_test_96khz.m4a";

        let mut params = CodecParameters::new(CodecId::Aac, MediaType::Audio);
        params.sample_rate = 96000;
        params.channel_layout.nb_channels = 2;
        params.bits_per_coded_sample = 16;
        // Minimal AAC AudioSpecificConfig (2 bytes)
        params.extradata = vec![0x11, 0x90]; // AAC-LC 96kHz stereo
        let stream = Stream::new(0, params);

        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));
        muxer.write_header(&mut io, &[stream]).unwrap();

        // Write a packet
        let mut pkt = Packet::new();
        pkt.data = Buffer::from_slice(&[0u8; 100]);
        pkt.stream_index = 0;
        pkt.pts = 0;
        pkt.dts = 0;
        pkt.duration = 1024;
        pkt.flags = PacketFlags::KEY;
        muxer.write_packet(&mut io, &pkt).unwrap();

        muxer.write_trailer(&mut io).unwrap();
        drop(io);

        // If ffprobe is available, verify it can read the file
        let output = std::process::Command::new("ffprobe")
            .args(["-v", "error", "-show_streams", "-of", "json", tmp_path])
            .output();
        std::fs::remove_file(tmp_path).ok();

        if let Ok(result) = output {
            assert!(
                result.status.success(),
                "ffprobe failed: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }

    #[test]
    fn test_audio_only_m4a() {
        let tmp_path = "/tmp/wedeo_test_m4a_brand.m4a";

        let mut params = CodecParameters::new(CodecId::Aac, MediaType::Audio);
        params.sample_rate = 44100;
        params.channel_layout.nb_channels = 2;
        params.bits_per_coded_sample = 16;
        params.extradata = vec![0x12, 0x10]; // AAC-LC 44100 stereo
        let stream = Stream::new(0, params);

        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));
        muxer.write_header(&mut io, &[stream]).unwrap();

        let mut pkt = Packet::new();
        pkt.data = Buffer::from_slice(&[0u8; 100]);
        pkt.stream_index = 0;
        pkt.pts = 0;
        pkt.dts = 0;
        pkt.duration = 1024;
        pkt.flags = PacketFlags::KEY;
        muxer.write_packet(&mut io, &pkt).unwrap();
        muxer.write_trailer(&mut io).unwrap();
        drop(io);

        // Verify M4A branding in the raw file
        let data = std::fs::read(tmp_path).expect("read file");
        std::fs::remove_file(tmp_path).ok();

        let boxes = parse_top_level_boxes(&data);
        let (_, ftyp_start, ftyp_size) = &boxes[0];
        let ftyp_data = &data[*ftyp_start..*ftyp_start + *ftyp_size];
        assert_eq!(&ftyp_data[8..12], b"M4A "); // major brand
    }

    #[test]
    fn test_large_duration() {
        // Simulate a track with duration > u32::MAX to trigger version-1 boxes
        let track = TrackState {
            stream_index: 0,
            codec_id: CodecId::Aac,
            media_type: MediaType::Audio,
            timescale: 1000, // match movie_timescale so rescale is 1:1
            width: 0,
            height: 0,
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
            bit_rate: 128000,
            extradata: vec![0x12, 0x10],
            sample_count: 1,
            sample_sizes: vec![100],
            sample_durations: vec![1024],
            chunk_offsets: vec![100],
            sync_samples: vec![1],
            cts_offsets: vec![0],
            has_cts: false,
            duration_ts: u32::MAX as u64 + 1000, // exceeds u32
        };

        let moov = atoms::write_moov(&[track], 1000);

        // Parse moov → mvhd, check it uses version 1
        let inner = parse_top_level_boxes(&moov[8..]); // skip moov header
        // mvhd is first inner box
        let (ref mvhd_type, mvhd_start, mvhd_size) = inner[0];
        assert_eq!(mvhd_type, "mvhd");
        // version is the first byte after the box header (8 bytes)
        let mvhd_data = &moov[8 + mvhd_start..8 + mvhd_start + mvhd_size];
        let version = mvhd_data[8]; // offset 8 = after size(4) + type(4)
        assert_eq!(version, 1, "mvhd should use version 1 for large durations");

        // Also check trak → tkhd uses version 1
        let (ref trak_type, trak_start, trak_size) = inner[1];
        assert_eq!(trak_type, "trak");
        let trak_data = &moov[8 + trak_start + 8..8 + trak_start + trak_size];
        let trak_inner = parse_top_level_boxes(trak_data);
        let (ref tkhd_type, tkhd_start, _) = trak_inner[0];
        assert_eq!(tkhd_type, "tkhd");
        let tkhd_version = trak_data[tkhd_start + 8];
        assert_eq!(
            tkhd_version, 1,
            "tkhd should use version 1 for large durations"
        );
    }

    #[test]
    fn test_faststart_box_order() {
        let tmp_path = "/tmp/wedeo_test_faststart_order.mp4";

        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        params.extradata = test_avcc();
        let stream = Stream::new(0, params);

        let mut muxer = Mp4Muxer::with_options(Mp4MuxerOptions { faststart: true });
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));

        muxer.write_header(&mut io, &[stream]).unwrap();

        for i in 0..3 {
            let data = vec![0xABu8; 100 + i * 10];
            let mut pkt = Packet::new();
            pkt.data = Buffer::from_slice(&data);
            pkt.stream_index = 0;
            pkt.pts = i as i64;
            pkt.dts = i as i64;
            pkt.duration = 1;
            if i == 0 {
                pkt.flags = PacketFlags::KEY;
            }
            muxer.write_packet(&mut io, &pkt).unwrap();
        }

        muxer.write_trailer(&mut io).unwrap();
        drop(io);

        let data = std::fs::read(tmp_path).expect("read file");
        std::fs::remove_file(tmp_path).ok();

        // Verify box order: ftyp, moov, mdat
        let boxes = parse_top_level_boxes(&data);
        let box_types: Vec<&str> = boxes.iter().map(|(t, _, _)| t.as_str()).collect();
        assert_eq!(box_types, vec!["ftyp", "moov", "mdat"]);

        // Verify mdat size: 100 + 110 + 120 = 330 payload + 16 extended header
        let (_, _, mdat_size) = &boxes[2];
        assert_eq!(*mdat_size, 330 + 16);
    }

    #[test]
    fn test_faststart_ffprobe() {
        let tmp_path = "/tmp/wedeo_test_faststart_ffprobe.mp4";

        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        params.extradata = test_avcc();
        let stream = Stream::new(0, params);

        let mut muxer = Mp4Muxer::with_options(Mp4MuxerOptions { faststart: true });
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));

        muxer.write_header(&mut io, &[stream]).unwrap();

        for i in 0..5 {
            let data = vec![0u8; 200];
            let mut pkt = Packet::new();
            pkt.data = Buffer::from_slice(&data);
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

        let output = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                tmp_path,
            ])
            .output();

        std::fs::remove_file(tmp_path).ok();

        if let Ok(result) = output {
            let stdout = String::from_utf8_lossy(&result.stdout);
            let stderr = String::from_utf8_lossy(&result.stderr);
            assert!(result.status.success(), "ffprobe failed: {}", stderr);
            assert!(stdout.contains("\"codec_type\": \"video\""));
            assert!(stdout.contains("\"width\": 320"));
        }
    }

    #[test]
    fn test_faststart_data_integrity() {
        let tmp_path = "/tmp/wedeo_test_faststart_integrity.mp4";

        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        params.extradata = test_avcc();
        let stream = Stream::new(0, params);

        let mut muxer = Mp4Muxer::with_options(Mp4MuxerOptions { faststart: true });
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));

        muxer.write_header(&mut io, &[stream]).unwrap();

        // Write packets with known content patterns
        let expected_payloads: Vec<Vec<u8>> = (0..5)
            .map(|i| vec![(i as u8).wrapping_mul(0x37); 100 + i * 50])
            .collect();
        for (i, payload) in expected_payloads.iter().enumerate() {
            let mut pkt = Packet::new();
            pkt.data = Buffer::from_slice(payload);
            pkt.stream_index = 0;
            pkt.pts = i as i64;
            pkt.dts = i as i64;
            pkt.duration = 1;
            if i == 0 {
                pkt.flags = PacketFlags::KEY;
            }
            muxer.write_packet(&mut io, &pkt).unwrap();
        }

        muxer.write_trailer(&mut io).unwrap();
        drop(io);

        let data = std::fs::read(tmp_path).expect("read file");
        std::fs::remove_file(tmp_path).ok();

        // Find mdat box and extract payload
        let boxes = parse_top_level_boxes(&data);
        let (ref mdat_type, mdat_start, mdat_size) = boxes[2];
        assert_eq!(mdat_type, "mdat");

        // mdat payload starts after extended header (16 bytes)
        let mdat_payload = &data[mdat_start + 16..mdat_start + mdat_size];

        // Verify all packet payloads are intact and contiguous
        let mut offset = 0;
        for payload in &expected_payloads {
            assert_eq!(
                &mdat_payload[offset..offset + payload.len()],
                payload.as_slice(),
                "packet data corrupted after faststart relocation"
            );
            offset += payload.len();
        }
        assert_eq!(offset, mdat_payload.len());
    }

    #[test]
    fn test_default_no_faststart() {
        let tmp_path = "/tmp/wedeo_test_default_no_faststart.mp4";

        let mut params = CodecParameters::new(CodecId::H264, MediaType::Video);
        params.width = 320;
        params.height = 240;
        params.time_base = Rational::new(1, 25);
        params.extradata = test_avcc();
        let stream = Stream::new(0, params);

        // Default constructor — no faststart
        let mut muxer = Mp4Muxer::new();
        let file_io = FileIo::create(tmp_path).expect("create file");
        let mut io = BufferedIo::new(Box::new(file_io));

        muxer.write_header(&mut io, &[stream]).unwrap();

        let mut pkt = Packet::new();
        pkt.data = Buffer::from_slice(&[0u8; 100]);
        pkt.stream_index = 0;
        pkt.pts = 0;
        pkt.dts = 0;
        pkt.duration = 1;
        pkt.flags = PacketFlags::KEY;
        muxer.write_packet(&mut io, &pkt).unwrap();

        muxer.write_trailer(&mut io).unwrap();
        drop(io);

        let data = std::fs::read(tmp_path).expect("read file");
        std::fs::remove_file(tmp_path).ok();

        // Verify standard order: ftyp, mdat, moov
        let boxes = parse_top_level_boxes(&data);
        let box_types: Vec<&str> = boxes.iter().map(|(t, _, _)| t.as_str()).collect();
        assert_eq!(box_types, vec!["ftyp", "mdat", "moov"]);
    }

    /// Parse top-level MP4 boxes from raw data (handles extended size).
    fn parse_top_level_boxes(data: &[u8]) -> Vec<(String, usize, usize)> {
        let mut boxes = Vec::new();
        let mut offset = 0;
        while offset + 8 <= data.len() {
            let size32 = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap());
            let box_type = String::from_utf8_lossy(&data[offset + 4..offset + 8]).to_string();
            let size = if size32 == 1 {
                // Extended size: 8-byte largesize follows the type field
                if offset + 16 > data.len() {
                    break;
                }
                u64::from_be_bytes(data[offset + 8..offset + 16].try_into().unwrap()) as usize
            } else {
                size32 as usize
            };
            if size < 8 || offset + size > data.len() {
                break;
            }
            boxes.push((box_type, offset, size));
            offset += size;
        }
        boxes
    }
}
