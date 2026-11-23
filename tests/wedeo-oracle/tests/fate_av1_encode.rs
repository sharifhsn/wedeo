//! FATE-style live AV1 encode (`rav1e` `Encoder::extradata()` → `Mp4Muxer`)
//! verified against FFmpeg `n9.1` (`ffprobe` + `ffmpeg` `framecrc`) and `rav1d`.

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use wedeo::core::{
    BorderPadding, BorderedPlane, BufferPool, CodecContext, CodecId, ColorDesc, ContainerHeader,
    EncodeHint, FileSink, MediaUnit, PassMode, PixelLayout, Rational, RecvStatus, SendStatus,
    StreamParams, Timestamp, TrackDescriptor, TrackId, VideoFrame, VideoStreamParams,
};
use wedeo::default_registry;
use wedeo_oracle::{
    FramecrcCompareOpts, compare_framecrc_full, ffprobe_oracle_bin, run_wedeo_framecrc,
    system_ffmpeg_bin, temp_test_dir,
};

fn build_synthetic_frame(
    pool: &Arc<BufferPool>,
    width: u32,
    height: u32,
    frame_idx: u32,
    pts: i64,
    time_base: Rational,
) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);

    let mut y = BorderedPlane::new(pool, w, h, BorderPadding::NONE);
    for row in 0..h {
        let line = y.row_mut(row).unwrap();
        for (col, px) in line.iter_mut().enumerate() {
            *px = (((col + row * 2 + frame_idx as usize * 17) % 219) + 16) as u8;
        }
    }

    let mut u = BorderedPlane::new(pool, cw, ch, BorderPadding::NONE);
    let mut v = BorderedPlane::new(pool, cw, ch, BorderPadding::NONE);
    for row in 0..ch {
        u.row_mut(row).unwrap().fill(128);
        v.row_mut(row).unwrap().fill(128);
    }

    let mut frame = VideoFrame::from_planes(width, height, PixelLayout::YUV420P, vec![y, u, v]);
    frame.pts = Timestamp::new(pts);
    frame.duration = 1;
    frame.time_base = time_base;
    frame.color = ColorDesc::BT709_LIMITED;
    frame
}

fn encode_rav1e_to_mp4(
    width: u32,
    height: u32,
    num_frames: u32,
    mp4_path: &Path,
) -> (usize, Vec<u8>) {
    let reg = default_registry();
    let pool = BufferPool::new();
    let tb = Rational::new(1, 30);

    let mut vparams = VideoStreamParams::new(width, height, PixelLayout::YUV420P);
    vparams.color = ColorDesc::BT709_LIMITED;
    let track = TrackDescriptor::new(
        TrackId(1),
        CodecId::Av1,
        tb,
        StreamParams::Video(vparams.clone()),
    );

    let enc_factory = reg.find_encoder(CodecId::Av1).expect("av1_rav1e encoder");
    assert_eq!(enc_factory.descriptor().name, "av1_rav1e");
    let mut encoder = enc_factory
        .open(
            &track,
            CodecContext::new(Arc::clone(&pool)),
            PassMode::Single,
        )
        .expect("open av1_rav1e");

    // Zero-backdoor av1C extradata directly from Encoder::extradata().
    let extradata = encoder.extradata().to_vec();
    assert!(
        extradata.len() >= 4 && (extradata[0] & 0x80) != 0,
        "expected valid av1C extradata"
    );

    let mut packets = Vec::new();
    for i in 0..num_frames {
        let frame = build_synthetic_frame(&pool, width, height, i, i as i64, tb);
        let hint = EncodeHint {
            force_keyframe: i == 0 || i == num_frames / 2,
            target_qp: None,
        };
        assert_eq!(
            encoder
                .send_unit(Some(&MediaUnit::Video(frame)), hint)
                .unwrap(),
            SendStatus::Accepted
        );
        while let RecvStatus::Ready(pkt) = encoder.receive_packet().unwrap() {
            packets.push(pkt);
        }
    }

    assert_eq!(
        encoder.send_unit(None, EncodeHint::default()).unwrap(),
        SendStatus::Accepted
    );
    loop {
        match encoder.receive_packet().unwrap() {
            RecvStatus::Ready(pkt) => packets.push(pkt),
            RecvStatus::NeedMore => continue,
            RecvStatus::Eof => break,
        }
    }

    let mut mux_params = vparams;
    mux_params.extradata = extradata.clone();
    let mux_track = TrackDescriptor::new(
        TrackId(1),
        CodecId::Av1,
        tb,
        StreamParams::Video(mux_params),
    );
    let mut header = ContainerHeader::new("mp4");
    header.tracks.push(mux_track);

    let mux_factory = reg.find_muxer_by_name("mp4").expect("mp4 muxer");
    let sink = FileSink::create(mp4_path).expect("create mp4 file");
    let mut muxer = mux_factory
        .create(Box::new(sink))
        .expect("create mp4 muxer");
    muxer.write_header(&header).unwrap();
    for pkt in &packets {
        muxer.write_packet(pkt).unwrap();
    }
    muxer.finalize().unwrap();

    (packets.len(), extradata)
}

#[test]
fn av1_encode_mp4_validated_and_decoded_bitexact_by_ffmpeg_oracle_and_rav1d() {
    let dir = temp_test_dir("av1_encode");
    let mp4_path = dir.join("rav1e_bt709.mp4");

    let (pkt_count, extradata) = encode_rav1e_to_mp4(176, 144, 6, &mp4_path);
    assert_eq!(pkt_count, 6);
    assert!(!extradata.is_empty());

    // 1. ffprobe oracle confirms valid AV1 MP4 stream with BT.709 metadata.
    let probe = Command::new(ffprobe_oracle_bin())
        .args(["-v", "error", "-show_streams", mp4_path.to_str().unwrap()])
        .output()
        .expect("run ffprobe oracle");
    assert!(
        probe.status.success(),
        "ffprobe oracle failed: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let probe_out = String::from_utf8_lossy(&probe.stdout);
    assert!(probe_out.contains("codec_name=av1"), "{probe_out}");
    assert!(probe_out.contains("width=176"), "{probe_out}");
    assert!(probe_out.contains("height=144"), "{probe_out}");
    assert!(probe_out.contains("color_primaries=bt709"), "{probe_out}");

    // 2. FFmpeg (libdav1d) decodes MP4 cleanly, and wedeo-framecrc matches it bit-exact!
    let ff_out = Command::new(system_ffmpeg_bin())
        .args([
            "-v",
            "error",
            "-bitexact",
            "-c:v",
            "libdav1d",
            "-i",
            mp4_path.to_str().unwrap(),
            "-f",
            "framecrc",
            "-flags",
            "+bitexact",
            "-fflags",
            "+bitexact",
            "-",
        ])
        .output()
        .expect("run ffmpeg libdav1d framecrc on rav1e MP4");
    assert!(
        ff_out.status.success(),
        "ffmpeg libdav1d decode failed: {}",
        String::from_utf8_lossy(&ff_out.stderr)
    );
    let ff_lines: Vec<String> = String::from_utf8_lossy(&ff_out.stdout)
        .lines()
        .map(ToOwned::to_owned)
        .collect();
    let wd_lines = run_wedeo_framecrc(&mp4_path).expect("wedeo-framecrc decode rav1e MP4");
    compare_framecrc_full(
        &wd_lines,
        &ff_lines,
        FramecrcCompareOpts::STRICT_TIMING_AND_HEADERS,
    )
    .expect("rav1e MP4 decode parity mismatch between wedeo (rav1d) and FFmpeg libdav1d");

    // 3. Also verify VideoReader decodes all 6 frames cleanly.
    let reg = default_registry();
    let mut reader = wedeo::video::VideoReader::open(&reg, &mp4_path).unwrap();
    let mut count = 0usize;
    while let Some(f) = reader.next_frame().unwrap() {
        assert_eq!(f.display_size(), (176, 144));
        assert_eq!(f.color.primaries, ColorDesc::BT709_LIMITED.primaries);
        count += 1;
    }
    assert_eq!(count, 6);
    assert!(reader.health().is_clean());
}
