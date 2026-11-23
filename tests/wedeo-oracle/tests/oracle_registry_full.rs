//! Full-workspace `wedeo::default_registry()` integration tests proving
//! zero format/codec hijacking and clean end-to-end `Encoder::extradata()` plumbing.

use std::io::Cursor;
use std::process::Command;
use std::sync::{Arc, Mutex};

use wedeo::core::{
    BorderPadding, BorderedPlane, BufferPool, ByteSink, CodecContext, CodecId, ColorDesc,
    ContainerHeader, EncodeHint, ErrorPolicy, MediaType, MediaUnit, MemorySource, Packet, PassMode,
    PixelLayout, Rational, RecvStatus, SendStatus, StreamParams, Timestamp, TrackDescriptor,
    TrackId, VideoFrame, VideoStreamParams,
};
use wedeo::{default_registry, native_registry};

/// In-memory seekable `ByteSink` that retains written bytes in a shared buffer
/// even after `Muxer::finalize(self: Box<Self>)` drops the sink.
#[derive(Clone)]
struct SharedVecSink {
    buf: Arc<Mutex<Vec<u8>>>,
    pos: usize,
}

impl SharedVecSink {
    fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                buf: Arc::clone(&buf),
                pos: 0,
            },
            buf,
        )
    }
}

impl std::io::Write for SharedVecSink {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let mut guard = self.buf.lock().unwrap();
        let end = self.pos + data.len();
        if end > guard.len() {
            guard.resize(end, 0);
        }
        guard[self.pos..end].copy_from_slice(data);
        self.pos = end;
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl ByteSink for SharedVecSink {
    fn is_seekable(&self) -> bool {
        true
    }

    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let len = self.buf.lock().unwrap().len() as i64;
        let next = match pos {
            std::io::SeekFrom::Start(n) => n as i64,
            std::io::SeekFrom::End(d) => len + d,
            std::io::SeekFrom::Current(d) => self.pos as i64 + d,
        };
        if next < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "negative seek",
            ));
        }
        self.pos = next as usize;
        Ok(self.pos as u64)
    }
}

fn make_minimal_wav_header() -> Vec<u8> {
    let mut h = Vec::with_capacity(64);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&1000u32.to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.resize(64, 0);
    h
}

#[test]
fn default_registry_native_wav_and_pcm_beat_symphonia() {
    let reg = default_registry();

    // Native WAV demuxer must score 100 and beat Symphonia (score <= 50).
    let head = make_minimal_wav_header();
    let winner = reg
        .probe_demuxer(&head, Some("wav"))
        .expect("WAV demuxer probe");
    assert_eq!(winner.descriptor().name, "wav");
    assert_eq!(winner.probe(&head, Some("wav")), 100);

    let symphonia = reg
        .find_demuxer_by_name("symphonia")
        .expect("symphonia demuxer registered");
    assert!(symphonia.probe(&head, Some("wav")) <= 50);

    // Native PCM decoders must win over Symphonia fallback decoders.
    for codec_id in [
        CodecId::PcmU8,
        CodecId::PcmS16le,
        CodecId::PcmS16be,
        CodecId::PcmS24le,
        CodecId::PcmS32le,
        CodecId::PcmF32le,
        CodecId::PcmF64le,
    ] {
        let dec = reg
            .find_decoder(codec_id)
            .unwrap_or_else(|| panic!("missing decoder for {codec_id:?}"));
        assert!(
            !dec.descriptor().name.contains("symphonia"),
            "symphonia hijacked {codec_id:?} (got {})",
            dec.descriptor().name
        );
    }

    // Native registry must exclude external adapters completely.
    let native = native_registry();
    assert!(native.find_demuxer_by_name("symphonia").is_none());
    assert!(native.find_decoder_by_name("av1_rav1d").is_none());
    assert!(native.find_encoder_by_name("av1_rav1e").is_none());
}

#[test]
fn default_registry_mkv_keeps_video_and_audio_without_symphonia_hijack() {
    let dir = wedeo_oracle::temp_test_dir("registry_mkv");
    let mkv_path = dir.join("av_mux.mkv");

    let status = Command::new(wedeo_oracle::system_ffmpeg_bin())
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=25:duration=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=0.2",
            "-c:v",
            "libvpx-vp9",
            "-c:a",
            "pcm_s16le",
            mkv_path.to_str().unwrap(),
        ])
        .status()
        .expect("run system ffmpeg to build MKV fixture");
    assert!(status.success(), "ffmpeg MKV generation failed");

    let mkv_bytes = std::fs::read(&mkv_path).unwrap();
    let reg = default_registry();

    // Native MKV demuxer must win over Symphonia.
    let head = &mkv_bytes[..mkv_bytes.len().min(8192)];
    let winner = reg.probe_demuxer(head, Some("mkv")).expect("MKV probe");
    assert_eq!(winner.descriptor().name, "matroska");

    let mut demuxer = winner
        .open(
            Box::new(MemorySource::new(mkv_bytes)),
            ErrorPolicy::Resilient,
        )
        .expect("open MKV");
    let (video_track_id, audio_track_id) = {
        let tracks = &demuxer.header().tracks;
        let video_track = tracks
            .iter()
            .find(|t| t.params.media_type() == MediaType::Video)
            .expect("MKV must keep video track");
        let audio_track = tracks
            .iter()
            .find(|t| t.params.media_type() == MediaType::Audio)
            .expect("MKV must keep audio track");
        assert_eq!(video_track.codec_id, CodecId::Vp9);
        assert_eq!(audio_track.codec_id, CodecId::PcmS16le);
        (video_track.id, audio_track.id)
    };

    let mut video_pkts = 0usize;
    let mut audio_pkts = 0usize;
    loop {
        match demuxer.read_packet().unwrap() {
            RecvStatus::Ready(pkt) => {
                if pkt.track_id == video_track_id {
                    video_pkts += 1;
                } else if pkt.track_id == audio_track_id {
                    audio_pkts += 1;
                }
            }
            RecvStatus::NeedMore => continue,
            RecvStatus::Eof => break,
        }
    }
    assert!(video_pkts >= 5, "expected >=5 video pkts, got {video_pkts}");
    assert!(audio_pkts >= 1, "expected >=1 audio pkts, got {audio_pkts}");
}

#[test]
fn default_registry_mp4_handles_video_and_audio() {
    let dir = wedeo_oracle::temp_test_dir("registry_mp4");
    let mp4_path = dir.join("av_mux.mp4");

    let status = Command::new(wedeo_oracle::system_ffmpeg_bin())
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=25:duration=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=0.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            mp4_path.to_str().unwrap(),
        ])
        .status()
        .expect("run system ffmpeg to build MP4 fixture");
    assert!(status.success(), "ffmpeg MP4 generation failed");

    let mp4_bytes = std::fs::read(&mp4_path).unwrap();
    let reg = default_registry();

    let head = &mp4_bytes[..mp4_bytes.len().min(8192)];
    let winner = reg.probe_demuxer(head, Some("mp4")).expect("MP4 probe");
    assert_eq!(winner.descriptor().name, "mp4");

    // Verify AudioReader decodes the AAC track and VideoReader decodes the H.264 track.
    let mut vreader = wedeo::video::VideoReader::open(&reg, &mp4_path).expect("VideoReader MP4");
    assert_eq!(vreader.header().format_name, "mp4");
    let mut vframes = 0usize;
    while let Some(f) = vreader.next_frame().unwrap() {
        assert_eq!(f.display_size(), (64, 64));
        vframes += 1;
    }
    assert_eq!(vframes, 5);

    let mut areader = wedeo::audio::AudioReader::open(&reg, &mp4_path).expect("AudioReader MP4");
    assert_eq!(areader.header().format_name, "mp4");
    let mut ablocks = 0usize;
    while let Some(b) = areader.next_block().unwrap() {
        assert!(b.nb_samples > 0);
        ablocks += 1;
    }
    assert!(ablocks >= 1, "expected decoded AAC audio blocks from MP4");
}

#[test]
fn default_registry_av1_encode_mp4_decode_via_encoder_extradata() {
    let reg = default_registry();
    let pool = BufferPool::new();
    let width = 64u32;
    let height = 64u32;
    let tb = Rational::new(1, 30);

    let mut vparams = VideoStreamParams::new(width, height, PixelLayout::YUV420P);
    vparams.color = ColorDesc::BT709_LIMITED;
    let track = TrackDescriptor::new(
        TrackId(1),
        CodecId::Av1,
        tb,
        StreamParams::Video(vparams.clone()),
    );

    let enc_factory = reg.find_encoder(CodecId::Av1).expect("av1_rav1e");
    assert_eq!(enc_factory.descriptor().name, "av1_rav1e");
    let mut encoder = enc_factory
        .open(
            &track,
            CodecContext::new(Arc::clone(&pool)),
            PassMode::Single,
        )
        .expect("open rav1e");

    // Zero backdoors: Encoder::extradata() provides av1C directly after open().
    let extradata = encoder.extradata().to_vec();
    assert!(
        extradata.len() >= 4 && (extradata[0] & 0x80) != 0,
        "expected valid av1C extradata"
    );

    let mut encoded_packets: Vec<Packet> = Vec::new();
    for i in 0..4u32 {
        let mut y = BorderedPlane::new(&pool, width as usize, height as usize, BorderPadding::NONE);
        for r in 0..height as usize {
            y.row_mut(r)
                .unwrap()
                .fill(((r * 3 + i as usize * 19) % 200 + 16) as u8);
        }
        let cw = width.div_ceil(2) as usize;
        let ch = height.div_ceil(2) as usize;
        let mut u = BorderedPlane::new(&pool, cw, ch, BorderPadding::NONE);
        let mut v = BorderedPlane::new(&pool, cw, ch, BorderPadding::NONE);
        for r in 0..ch {
            u.row_mut(r).unwrap().fill(128);
            v.row_mut(r).unwrap().fill(128);
        }
        let mut frame = VideoFrame::from_planes(width, height, PixelLayout::YUV420P, vec![y, u, v]);
        frame.pts = Timestamp::new(i as i64 * 1000);
        frame.duration = 1000;
        frame.time_base = tb;
        frame.color = ColorDesc::BT709_LIMITED;

        assert_eq!(
            encoder
                .send_unit(Some(&MediaUnit::Video(frame)), EncodeHint::default())
                .unwrap(),
            SendStatus::Accepted
        );
        while let RecvStatus::Ready(pkt) = encoder.receive_packet().unwrap() {
            encoded_packets.push(pkt);
        }
    }
    assert_eq!(
        encoder.send_unit(None, EncodeHint::default()).unwrap(),
        SendStatus::Accepted
    );
    loop {
        match encoder.receive_packet().unwrap() {
            RecvStatus::Ready(pkt) => encoded_packets.push(pkt),
            RecvStatus::NeedMore => continue,
            RecvStatus::Eof => break,
        }
    }
    assert_eq!(encoded_packets.len(), 4);

    // Mux into MP4 using public Mp4Muxer + Encoder::extradata().
    let (sink, mp4_buf) = SharedVecSink::new();
    let mux_factory = reg.find_muxer_by_name("mp4").expect("mp4 muxer");
    let mut muxer = mux_factory
        .create(Box::new(sink))
        .expect("create mp4 muxer");

    let mut mux_vparams = vparams;
    mux_vparams.extradata = extradata;
    let mux_track = TrackDescriptor::new(
        TrackId(1),
        CodecId::Av1,
        tb,
        StreamParams::Video(mux_vparams),
    );
    let mut header = ContainerHeader::new("mp4");
    header.tracks.push(mux_track);
    muxer.write_header(&header).unwrap();
    for pkt in &encoded_packets {
        muxer.write_packet(pkt).unwrap();
    }
    muxer.finalize().unwrap();

    let mp4_bytes = mp4_buf.lock().unwrap().clone();
    let probe_head = &mp4_bytes[..mp4_bytes.len().min(8192)];
    let demux_factory = reg.probe_demuxer(probe_head, Some("mp4")).unwrap();
    assert_eq!(demux_factory.descriptor().name, "mp4");

    let mut reader = wedeo::video::VideoReader::from_source(
        &reg,
        Box::new(MemorySource::new(mp4_bytes)),
        ErrorPolicy::Strict,
    )
    .expect("VideoReader on rav1e MP4");
    assert_eq!(reader.params().codec_id_or_extradata_check(), ());
    let mut decoded_frames = 0usize;
    while let Some(frame) = reader.next_frame().unwrap() {
        assert_eq!(frame.display_size(), (64, 64));
        assert_eq!(frame.color.primaries, ColorDesc::BT709_LIMITED.primaries);
        decoded_frames += 1;
    }
    assert_eq!(decoded_frames, 4);
    let _ = Cursor::new(Vec::<u8>::new());
}

trait CheckParamsExt {
    fn codec_id_or_extradata_check(&self);
}
impl CheckParamsExt for VideoStreamParams {
    fn codec_id_or_extradata_check(&self) {
        assert_eq!(self.width, 64);
        assert_eq!(self.height, 64);
        assert!(!self.extradata.is_empty(), "av1C extradata missing in MP4");
    }
}
