use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};

use wedeo_core::{
    AudioStreamParams, BufByteSource, Buffer, ByteSink, ByteSource, ChannelLayout, CodecId,
    ColorDesc, ContainerHeader, EditSegment, ErrorKind, ErrorPolicy, MemorySink, MemorySource,
    Packet, PacketFlags, PixelLayout, Rational, RecvStatus, Registry, SampleFormat, SeekTarget,
    SideData, StreamParams, Timestamp, TrackDescriptor, TrackId, VideoStreamParams,
};
use wedeo_format_mp4::{boxes, register_mp4};

struct PipeSink {
    buf: Arc<Mutex<Vec<u8>>>,
}

impl Write for PipeSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buf.lock().unwrap().extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl ByteSink for PipeSink {
    fn is_seekable(&self) -> bool {
        false
    }
}

struct PipeSource {
    data: Vec<u8>,
    pos: usize,
}

impl Read for PipeSource {
    fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        let avail = self.data.len().saturating_sub(self.pos);
        let n = avail.min(dst.len());
        dst[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

impl ByteSource for PipeSource {
    fn is_seekable(&self) -> bool {
        false
    }
}

/// Valid minimal avcC (Baseline Profile, Level 3.0, 320x240)
fn test_avcc() -> Vec<u8> {
    vec![
        0x01, 0x42, 0xC0, 0x1E, 0xFF, 0xE1, 0x00, 0x09, 0x67, 0x42, 0xC0, 0x1E, 0xD9, 0x01, 0x41,
        0xFB, 0x01, 0x01, 0x00, 0x04, 0x68, 0xCE, 0x3C, 0x80,
    ]
}

/// AAC-LC 44.1 kHz stereo AudioSpecificConfig
fn test_aac_asc() -> Vec<u8> {
    vec![0x12, 0x10]
}

/// Minimal AVCC-framed H.264 IDR slice NAL (4-byte length + NAL)
fn test_h264_idr_pkt() -> Vec<u8> {
    let nal = [0x65, 0x88, 0x84, 0x21, 0xA0];
    let mut out = Vec::with_capacity(4 + nal.len());
    out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
    out.extend_from_slice(&nal);
    out
}

/// Minimal AVCC-framed H.264 non-IDR slice NAL
fn test_h264_p_pkt() -> Vec<u8> {
    let nal = [0x41, 0x9A, 0x24, 0x6C, 0x41];
    let mut out = Vec::with_capacity(4 + nal.len());
    out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
    out.extend_from_slice(&nal);
    out
}

#[test]
fn register_mp4_and_probe() {
    let mut reg = Registry::new();
    register_mp4(&mut reg);

    assert!(reg.find_demuxer_by_name("mp4").is_some());
    assert!(reg.find_muxer_by_name("mp4").is_some());
    assert!(reg.find_muxer_by_ext("m4a").is_some());

    let demux = reg.find_demuxer_by_name("mp4").unwrap();
    let ftyp_isom = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2";
    let score = demux.probe(ftyp_isom, Some("mp4"));
    assert!(
        (90..=100).contains(&score),
        "unexpected ftyp score: {score}"
    );

    assert_eq!(demux.probe(b"RIFF\x00\x00\x00\x00WAVEfmt ", Some("wav")), 0);
    assert_eq!(demux.probe(b"OggS\x00\x02\x00\x00", Some("mp4")), 0);
    assert_eq!(demux.probe(b"short", None), 0);
}

#[test]
fn seekable_faststart_roundtrip_and_box_ast() {
    let mut reg = Registry::new();
    register_mp4(&mut reg);

    let mut hdr = ContainerHeader::new("mp4");

    let mut vp = VideoStreamParams::new(320, 240, PixelLayout::YUV420P);
    vp.color = ColorDesc::BT709_LIMITED;
    vp.sar = Rational::new(4, 3);
    vp.display_matrix = Some(boxes::rotation_degrees_to_matrix(90));
    vp.extradata = test_avcc();
    let mut vtrack = TrackDescriptor::new(
        TrackId(0),
        CodecId::H264,
        Rational::new(1, 90_000),
        StreamParams::Video(vp),
    );
    vtrack.language = Some("eng".to_string());
    vtrack.edit_list.push(EditSegment {
        media_time: Timestamp::new(3000),
        duration: 9000,
        time_base: Rational::new(1, 90_000),
        rate: Rational::new(1, 1),
    });

    let mut ap = AudioStreamParams::new(44_100, SampleFormat::F32Planar, ChannelLayout::stereo());
    ap.extradata = test_aac_asc();
    let mut atrack = TrackDescriptor::new(
        TrackId(1),
        CodecId::Aac,
        Rational::new(1, 44_100),
        StreamParams::Audio(ap),
    );
    atrack.language = Some("und".to_string());

    hdr.tracks.push(vtrack);
    hdr.tracks.push(atrack);

    let sink = MemorySink::new();
    let mut muxer = reg
        .find_muxer_by_name("mp4")
        .unwrap()
        .create(Box::new(sink))
        .unwrap();
    muxer.write_header(&hdr).unwrap();

    // VFR video packets with B-frame reordering (including negative CTS offset -> ctts v1)
    let v_frames = [
        (3000i64, 0i64, 3000i64, true),
        (0, 3000, 1500, false),
        (6000, 4500, 4500, false),
        (9000, 9000, 3000, true),
    ];
    for &(pts, dts, dur, is_key) in &v_frames {
        let raw = if is_key {
            test_h264_idr_pkt()
        } else {
            test_h264_p_pkt()
        };
        let mut pkt = Packet::new(
            Buffer::from_slice(&raw),
            TrackId(0),
            Rational::new(1, 90_000),
        );
        pkt.pts = Timestamp::new(pts);
        pkt.dts = Timestamp::new(dts);
        pkt.duration = dur;
        if is_key {
            pkt.flags |= PacketFlags::KEY;
        }
        muxer.write_packet(&pkt).unwrap();
    }

    for i in 0..4i64 {
        let mut pkt = Packet::new(
            Buffer::from_slice(&[0xDE, 0x04, 0x00, 0x20]),
            TrackId(1),
            Rational::new(1, 44_100),
        );
        pkt.pts = Timestamp::new(i * 1024);
        pkt.dts = Timestamp::new(i * 1024);
        pkt.duration = 1024;
        pkt.flags |= PacketFlags::KEY;
        muxer.write_packet(&pkt).unwrap();
    }

    // Extract bytes by using a shared Pipe-like wrapper? MemorySink is owned by Box<dyn ByteSink>.
    // Write to a temp file via FileSink so we can inspect & demux!
    drop(muxer);

    let path = "/tmp/wedeo_test_faststart_v2.mp4";
    let file_sink = wedeo_core::FileSink::create(path).unwrap();
    let mut muxer = reg
        .find_muxer_by_name("mp4")
        .unwrap()
        .create(Box::new(file_sink))
        .unwrap();
    muxer.write_header(&hdr).unwrap();
    for &(pts, dts, dur, is_key) in &v_frames {
        let raw = if is_key {
            test_h264_idr_pkt()
        } else {
            test_h264_p_pkt()
        };
        let mut pkt = Packet::new(
            Buffer::from_slice(&raw),
            TrackId(0),
            Rational::new(1, 90_000),
        );
        pkt.pts = Timestamp::new(pts);
        pkt.dts = Timestamp::new(dts);
        pkt.duration = dur;
        if is_key {
            pkt.flags |= PacketFlags::KEY;
        }
        muxer.write_packet(&pkt).unwrap();
    }
    for i in 0..4i64 {
        let mut pkt = Packet::new(
            Buffer::from_slice(&[0xDE, 0x04, 0x00, 0x20]),
            TrackId(1),
            Rational::new(1, 44_100),
        );
        pkt.pts = Timestamp::new(i * 1024);
        pkt.dts = Timestamp::new(i * 1024);
        pkt.duration = 1024;
        pkt.flags |= PacketFlags::KEY;
        muxer.write_packet(&pkt).unwrap();
    }
    muxer.finalize().unwrap();

    let mp4_bytes = std::fs::read(path).unwrap();

    // 1. Verify box AST (`parse_box_tree_from_slice` & `parse_box_tree`)
    let ast = boxes::parse_box_tree_from_slice(&mp4_bytes).unwrap();
    assert_eq!(ast.len(), 3);
    assert_eq!(ast[0].type_name(), "ftyp");
    assert_eq!(ast[1].type_name(), "moov"); // faststart: moov before mdat!
    assert_eq!(ast[2].type_name(), "mdat");

    let mut buf_src = BufByteSource::new(Box::new(MemorySource::new(mp4_bytes.clone())));
    let ast2 = boxes::parse_box_tree(&mut buf_src).unwrap();
    assert_eq!(ast, ast2);

    let moov = &ast[1];
    assert!(moov.child(b"mvhd").is_some());
    let trak0 = moov.child(b"trak").unwrap();
    assert!(trak0.child(b"edts").unwrap().child(b"elst").is_some());

    // 2. Demux and verify all metadata, VFR timestamps, side data, and seek
    let demux_factory = reg.probe_demuxer(&mp4_bytes, Some("mp4")).unwrap();
    let mut demuxer = demux_factory
        .open(Box::new(MemorySource::new(mp4_bytes)), ErrorPolicy::Strict)
        .unwrap();

    let dh = demuxer.header();
    assert_eq!(dh.tracks.len(), 2);

    let vt = &dh.tracks[0];
    assert_eq!(vt.codec_id, CodecId::H264);
    assert_eq!(vt.language.as_deref(), Some("eng"));
    assert_eq!(vt.edit_list.len(), 1);
    assert_eq!(vt.edit_list[0].media_time, Timestamp::new(3000));
    let vparams = vt.params.video().unwrap();
    assert_eq!(vparams.width, 320);
    assert_eq!(vparams.height, 240);
    assert_eq!(vparams.color.primaries, ColorDesc::BT709_LIMITED.primaries);
    assert_eq!(vparams.color.transfer, ColorDesc::BT709_LIMITED.transfer);
    assert_eq!(vparams.color.matrix, ColorDesc::BT709_LIMITED.matrix);
    assert_eq!(vparams.color.range, ColorDesc::BT709_LIMITED.range);
    assert_eq!(vparams.sar, Rational::new(4, 3));
    assert_eq!(
        vparams.display_matrix,
        Some(boxes::rotation_degrees_to_matrix(90))
    );

    let at = &dh.tracks[1];
    assert_eq!(at.codec_id, CodecId::Aac);
    let aparams = at.params.audio().unwrap();
    assert_eq!(aparams.sample_rate, 44_100);
    assert_eq!(aparams.channel_layout.nb_channels, 2);

    let mut v_out = Vec::new();
    let mut a_out = Vec::new();
    let mut saw_display_matrix = false;
    loop {
        match demuxer.read_packet().unwrap() {
            RecvStatus::Ready(pkt) => {
                if pkt.track_id == TrackId(0) {
                    if pkt
                        .side_data
                        .iter()
                        .any(|s| matches!(s, SideData::DisplayMatrix(_)))
                    {
                        saw_display_matrix = true;
                    }
                    v_out.push(pkt);
                } else {
                    a_out.push(pkt);
                }
            }
            RecvStatus::Eof => break,
            RecvStatus::NeedMore => unreachable!(),
        }
    }

    assert!(saw_display_matrix);
    assert_eq!(v_out.len(), 4);
    assert_eq!(a_out.len(), 4);
    assert!(demuxer.health().is_clean());

    // Verify VFR ctts v1 negative & positive offsets preserved
    for (pkt, &(pts, dts, dur, is_key)) in v_out.iter().zip(v_frames.iter()) {
        assert_eq!(pkt.pts, Timestamp::new(pts));
        assert_eq!(pkt.dts, Timestamp::new(dts));
        assert_eq!(pkt.duration, dur);
        assert_eq!(pkt.flags.contains(PacketFlags::KEY), is_key);
    }

    // Verify backward keyframe seek
    demuxer
        .seek(SeekTarget {
            track: Some(TrackId(0)),
            timestamp: Timestamp::new(4500),
            time_base: Rational::new(1, 90_000),
            backward: true,
        })
        .unwrap();
    match demuxer.read_packet().unwrap() {
        RecvStatus::Ready(pkt) => {
            assert_eq!(pkt.track_id, TrackId(0));
            assert_eq!(pkt.dts, Timestamp::new(0));
            assert!(pkt.flags.contains(PacketFlags::KEY));
        }
        other => panic!("expected Ready packet after seek, got {other:?}"),
    }

    std::fs::remove_file(path).ok();
}

#[test]
fn non_seekable_pipe_auto_fmp4_and_ffprobe() {
    let mut reg = Registry::new();
    register_mp4(&mut reg);

    let shared = Arc::new(Mutex::new(Vec::new()));
    let pipe_sink = PipeSink {
        buf: Arc::clone(&shared),
    };

    let mut hdr = ContainerHeader::new("mp4");
    let mut vp = VideoStreamParams::new(320, 240, PixelLayout::YUV420P);
    vp.extradata = test_avcc();
    hdr.tracks.push(TrackDescriptor::new(
        TrackId(0),
        CodecId::H264,
        Rational::new(1, 30),
        StreamParams::Video(vp),
    ));
    let mut ap = AudioStreamParams::new(44_100, SampleFormat::F32Planar, ChannelLayout::stereo());
    ap.extradata = test_aac_asc();
    hdr.tracks.push(TrackDescriptor::new(
        TrackId(1),
        CodecId::Aac,
        Rational::new(1, 44_100),
        StreamParams::Audio(ap),
    ));

    let mut muxer = reg
        .find_muxer_by_name("mp4")
        .unwrap()
        .create(Box::new(pipe_sink))
        .unwrap();
    muxer.write_header(&hdr).unwrap();

    // 2 GOPs of 3 frames each -> should emit 2 moof/mdat fragments
    for i in 0..6i64 {
        let is_key = i % 3 == 0;
        let raw = if is_key {
            test_h264_idr_pkt()
        } else {
            test_h264_p_pkt()
        };
        let mut pkt = Packet::new(Buffer::from_slice(&raw), TrackId(0), Rational::new(1, 30));
        pkt.pts = Timestamp::new(i);
        pkt.dts = Timestamp::new(i);
        pkt.duration = 1;
        if is_key {
            pkt.flags |= PacketFlags::KEY;
        }
        muxer.write_packet(&pkt).unwrap();
    }

    for i in 0..2i64 {
        let mut pkt = Packet::new(
            Buffer::from_slice(&[0xDE, 0x04, 0x00, 0x20]),
            TrackId(1),
            Rational::new(1, 44_100),
        );
        pkt.pts = Timestamp::new(i * 1024);
        pkt.dts = Timestamp::new(i * 1024);
        pkt.duration = 1024;
        pkt.flags |= PacketFlags::KEY;
        muxer.write_packet(&pkt).unwrap();
    }

    muxer.finalize().unwrap();
    let fmp4_bytes = shared.lock().unwrap().clone();

    // Verify box AST has ftyp + moov(mvex) + multiple moof/mdat
    let ast = boxes::parse_box_tree_from_slice(&fmp4_bytes).unwrap();
    assert_eq!(ast[0].type_name(), "ftyp");
    assert_eq!(ast[1].type_name(), "moov");
    assert!(ast[1].child(b"mvex").is_some());
    let moof_count = ast.iter().filter(|n| n.type_name() == "moof").count();
    let mdat_count = ast.iter().filter(|n| n.type_name() == "mdat").count();
    assert_eq!(moof_count, 2);
    assert_eq!(mdat_count, 2);

    // Verify our own demuxer reads all packets from a non-seekable PipeSource
    let demux_factory = reg.probe_demuxer(&fmp4_bytes, Some("mp4")).unwrap();
    let mut demuxer = demux_factory
        .open(
            Box::new(PipeSource {
                data: fmp4_bytes.clone(),
                pos: 0,
            }),
            ErrorPolicy::Strict,
        )
        .unwrap();
    let mut v_count = 0;
    let mut a_count = 0;
    while let RecvStatus::Ready(pkt) = demuxer.read_packet().unwrap() {
        if pkt.track_id == TrackId(0) {
            v_count += 1;
        } else {
            a_count += 1;
        }
    }
    assert_eq!(v_count, 6);
    assert_eq!(a_count, 2);

    // Verify with FFmpeg oracle ffprobe n9.1
    let ffprobe_path = "/home/sharif/wedeo/target/ffmpeg-oracle/ffprobe";
    if std::path::Path::new(ffprobe_path).exists() {
        let tmp_fmp4 = "/tmp/wedeo_test_auto_fmp4_oracle.mp4";
        std::fs::write(tmp_fmp4, &fmp4_bytes).unwrap();
        let out = std::process::Command::new(ffprobe_path)
            .args([
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                tmp_fmp4,
            ])
            .output()
            .expect("run ffprobe");
        std::fs::remove_file(tmp_fmp4).ok();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "ffprobe failed (status={}): stderr={stderr}",
            out.status
        );
        assert!(stdout.contains("\"codec_name\": \"h264\""), "{stdout}");
        assert!(stdout.contains("\"codec_name\": \"aac\""), "{stdout}");
        assert!(stdout.contains("\"width\": 320"), "{stdout}");
        assert!(stdout.contains("\"height\": 240"), "{stdout}");
    }
}

#[test]
fn opus_alac_pcm_and_error_policy() {
    let mut reg = Registry::new();
    register_mp4(&mut reg);

    let path = "/tmp/wedeo_test_audio_codecs.mp4";
    let mut hdr = ContainerHeader::new("mp4");

    // Opus with OpusHead extradata
    let mut opus_head = b"OpusHead\x01\x02\x38\x01\x80\xBB\x00\x00\x00\x00\x00".to_vec();
    let mut opus_ap = AudioStreamParams::new(48_000, SampleFormat::F32, ChannelLayout::stereo());
    opus_ap.extradata = std::mem::take(&mut opus_head);
    hdr.tracks.push(TrackDescriptor::new(
        TrackId(0),
        CodecId::Opus,
        Rational::new(1, 48_000),
        StreamParams::Audio(opus_ap),
    ));

    // PCM S16LE
    let pcm_ap = AudioStreamParams::new(44_100, SampleFormat::S16, ChannelLayout::stereo());
    hdr.tracks.push(TrackDescriptor::new(
        TrackId(1),
        CodecId::PcmS16le,
        Rational::new(1, 44_100),
        StreamParams::Audio(pcm_ap),
    ));

    let file_sink = wedeo_core::FileSink::create(path).unwrap();
    let mut muxer = reg
        .find_muxer_by_name("mp4")
        .unwrap()
        .create(Box::new(file_sink))
        .unwrap();
    muxer.write_header(&hdr).unwrap();

    let mut p0 = Packet::new(
        Buffer::from_slice(&[0xFC, 0x01, 0x02, 0x03]),
        TrackId(0),
        Rational::new(1, 48_000),
    );
    p0.pts = Timestamp::new(0);
    p0.dts = Timestamp::new(0);
    p0.duration = 960;
    p0.flags |= PacketFlags::KEY;
    muxer.write_packet(&p0).unwrap();

    let mut p1 = Packet::new(
        Buffer::from_slice(&[0x01, 0x02, 0x03, 0x04]),
        TrackId(1),
        Rational::new(1, 44_100),
    );
    p1.pts = Timestamp::new(0);
    p1.dts = Timestamp::new(0);
    p1.duration = 1;
    p1.flags |= PacketFlags::KEY;
    muxer.write_packet(&p1).unwrap();

    muxer.finalize().unwrap();
    let bytes = std::fs::read(path).unwrap();
    std::fs::remove_file(path).ok();

    let demux_factory = reg.find_demuxer_by_name("mp4").unwrap();
    let demux = demux_factory
        .open(
            Box::new(MemorySource::new(bytes.clone())),
            ErrorPolicy::Strict,
        )
        .unwrap();
    assert_eq!(demux.header().tracks[0].codec_id, CodecId::Opus);
    assert_eq!(demux.header().tracks[1].codec_id, CodecId::PcmS16le);

    // Truncate last 2 bytes to test ErrorPolicy::Strict vs ErrorPolicy::Resilient
    let truncated = bytes[..bytes.len() - 2].to_vec();

    let mut strict_d = demux_factory
        .open(
            Box::new(MemorySource::new(truncated.clone())),
            ErrorPolicy::Strict,
        )
        .unwrap();
    let _ = strict_d.read_packet().unwrap();
    let err = strict_d.read_packet().unwrap_err();
    assert_eq!(err.kind, ErrorKind::CorruptStream);

    let mut resilient_d = demux_factory
        .open(
            Box::new(MemorySource::new(truncated)),
            ErrorPolicy::Resilient,
        )
        .unwrap();
    let _ = resilient_d.read_packet().unwrap();
    match resilient_d.read_packet().unwrap() {
        RecvStatus::Ready(pkt) => assert!(pkt.flags.contains(PacketFlags::CORRUPT)),
        other => panic!("expected corrupt packet in Resilient mode, got {other:?}"),
    }
    assert_eq!(resilient_d.health().corrupt_packets, 1);
}
