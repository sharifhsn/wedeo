//! FATE-style live H.264 decode parity tests against FFmpeg `n9.1`.
//!
//! Generates High-profile CABAC (with B-pyramid, BT.709 VUI, SAR 16:15) and
//! non-mod16 cropped streams via system FFmpeg (`libx264`), then verifies:
//! - 100% bit-exact `framecrc` parity against `/home/sharif/wedeo/target/ffmpeg-oracle/ffmpeg`
//! - BT.709 VUI colorimetry, SAR, and `CropRect` display dimensions
//! - `threads=1 == threads=2` determinism
//! - Clean `StreamHealth`

use std::io::Read;
use std::path::Path;
use std::process::Command;

use wedeo::core::{
    CodecContext, CodecId, ColorDesc, CropRect, ErrorPolicy, FileSource, FrameFlags, PictureType,
    Rational, RecvStatus, SendStatus, VideoFrame,
};
use wedeo::default_registry;
use wedeo_oracle::{
    FramecrcCompareOpts, compare_framecrc_full, parse_framecrc_headers, run_ffmpeg_framecrc,
    run_wedeo_framecrc, system_ffmpeg_bin, temp_test_dir,
};

fn decode_h264_frames(path: &Path, ext: &str, threads: usize) -> Vec<VideoFrame> {
    let reg = default_registry();
    let mut head = [0u8; 8192];
    let n = {
        let mut f = FileSource::open(path).unwrap();
        f.read(&mut head).unwrap()
    };
    let demux_factory = reg
        .probe_demuxer(&head[..n], Some(ext))
        .expect("probe demuxer for H.264");
    let mut demuxer = demux_factory
        .open(
            Box::new(FileSource::open(path).unwrap()),
            ErrorPolicy::Strict,
        )
        .expect("open H.264 demuxer");

    let track = demuxer
        .header()
        .tracks
        .iter()
        .find(|t| t.codec_id == CodecId::H264)
        .expect("H.264 track")
        .clone();

    let ctx = CodecContext {
        threads,
        error_policy: ErrorPolicy::Strict,
        ..Default::default()
    };

    let mut decoder = reg
        .find_decoder(CodecId::H264)
        .expect("H.264 decoder")
        .open(&track, ctx)
        .expect("open H.264 decoder");

    let mut frames = Vec::new();
    loop {
        match demuxer.read_packet().unwrap() {
            RecvStatus::Ready(pkt) => {
                if pkt.track_id != track.id {
                    continue;
                }
                let mut pending = Some(pkt);
                while let Some(p) = pending.take() {
                    if decoder.send_packet(Some(&p)).unwrap() == SendStatus::Full {
                        pending = Some(p);
                    }
                    while let RecvStatus::Ready(unit) = decoder.receive().unwrap() {
                        frames.push(unit.into_video().unwrap());
                    }
                }
            }
            RecvStatus::NeedMore => continue,
            RecvStatus::Eof => break,
        }
    }
    assert_eq!(decoder.send_packet(None).unwrap(), SendStatus::Accepted);
    while let RecvStatus::Ready(unit) = decoder.receive().unwrap() {
        frames.push(unit.into_video().unwrap());
    }
    assert!(decoder.health().is_clean());
    assert!(demuxer.health().is_clean());
    frames
}

fn frame_display_bytes(frame: &VideoFrame) -> Vec<u8> {
    let (dw, dh) = frame.display_size();
    let planes = frame.planes().unwrap();
    let crop = frame.crop;
    let cw = (dw as usize).div_ceil(2);
    let ch = (dh as usize).div_ceil(2);
    let mut out = Vec::with_capacity(dw as usize * dh as usize + cw * ch * 2);

    for r in 0..dh as usize {
        let row = planes[0].row(crop.top as usize + r).unwrap();
        out.extend_from_slice(&row[crop.left as usize..crop.left as usize + dw as usize]);
    }
    for plane in &planes[1..=2] {
        for r in 0..ch {
            let row = plane.row((crop.top as usize / 2) + r).unwrap();
            out.extend_from_slice(&row[(crop.left as usize / 2)..(crop.left as usize / 2) + cw]);
        }
    }
    out
}

#[test]
fn h264_high_profile_cabac_bt709_sar_and_threads_parity_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("h264_bt709");
    let h264_path = dir.join("bt709_cabac.h264");
    let mp4_path = dir.join("bt709_cabac.mp4");

    // Generate High-profile CABAC H.264 Annex B stream with B-pyramid, BT.709 VUI, SAR 16:15.
    let st = Command::new(system_ffmpeg_bin())
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x112:rate=30:duration=0.6",
            "-vf",
            "setsar=16/15",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-crf",
            "20",
            "-g",
            "9",
            "-keyint_min",
            "9",
            "-bf",
            "2",
            "-refs",
            "2",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-color_range",
            "tv",
            "-x264-params",
            "aud=1:scenecut=0:b-pyramid=normal:b-adapt=0:colorprim=bt709:transfer=bt709:colormatrix=bt709",
            h264_path.to_str().unwrap(),
        ])
        .status()
        .expect("run system ffmpeg libx264");
    assert!(st.success(), "libx264 Annex B generation failed");

    // Generate matching MP4 directly via libx264 so container PTS/DTS/ctts are valid.
    let st_mp4 = Command::new(system_ffmpeg_bin())
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x112:rate=30:duration=0.6",
            "-vf",
            "setsar=16/15",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-crf",
            "20",
            "-g",
            "9",
            "-keyint_min",
            "9",
            "-bf",
            "2",
            "-refs",
            "2",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-color_range",
            "tv",
            "-x264-params",
            "aud=1:scenecut=0:b-pyramid=normal:b-adapt=0:colorprim=bt709:transfer=bt709:colormatrix=bt709",
            mp4_path.to_str().unwrap(),
        ])
        .status()
        .expect("generate H.264 MP4");
    assert!(st_mp4.success(), "libx264 MP4 generation failed");

    // 1. Annex B raw stream bit-exact framecrc vs FFmpeg n9.1 oracle.
    let wd_annexb = run_wedeo_framecrc(&h264_path).expect("wedeo-framecrc Annex B");
    let ff_annexb = run_ffmpeg_framecrc(&h264_path).expect("ffmpeg oracle Annex B");
    compare_framecrc_full(&wd_annexb, &ff_annexb, FramecrcCompareOpts::DATA_ONLY)
        .expect("H.264 Annex B framecrc mismatch vs FFmpeg n9.1");
    let wd_hdrs = parse_framecrc_headers(&wd_annexb);
    let ff_hdrs = parse_framecrc_headers(&ff_annexb);
    assert_eq!(wd_hdrs[0].dimensions, ff_hdrs[0].dimensions);
    assert_eq!(wd_hdrs[0].sar.as_deref(), Some("16/15"));

    // 2. MP4 container bit-exact framecrc vs FFmpeg n9.1 oracle.
    let wd_mp4 = run_wedeo_framecrc(&mp4_path).expect("wedeo-framecrc MP4");
    let ff_mp4 = run_ffmpeg_framecrc(&mp4_path).expect("ffmpeg oracle MP4");
    compare_framecrc_full(&wd_mp4, &ff_mp4, FramecrcCompareOpts::DATA_ONLY)
        .expect("H.264 MP4 framecrc mismatch vs FFmpeg n9.1");

    // 3. BT.709 VUI colorimetry, SAR, B-frames, KEY flags, and threads=1 == threads=2 parity.
    let t1 = decode_h264_frames(&h264_path, "h264", 1);
    let t2 = decode_h264_frames(&h264_path, "h264", 2);
    assert_eq!(t1.len(), 18);
    assert_eq!(t2.len(), 18);

    let mut saw_b = false;
    for (i, (f1, f2)) in t1.iter().zip(t2.iter()).enumerate() {
        assert_eq!(f1.display_size(), (160, 112));
        assert_eq!(f1.crop, CropRect::ZERO);
        assert_eq!(f1.color, ColorDesc::BT709_LIMITED);
        assert_eq!(f1.sar, Rational::new(16, 15));
        assert_eq!(f1.flags.contains(FrameFlags::KEY), i == 0 || i == 9);
        if f1.picture_type == PictureType::B {
            saw_b = true;
        }
        assert_eq!(
            frame_display_bytes(f1),
            frame_display_bytes(f2),
            "H.264 threads=1 vs threads=2 mismatch at frame {i}"
        );
    }
    assert!(saw_b, "expected B-frames in High-profile stream");
}

#[test]
fn h264_non_mod16_crop_and_full_range_bitexact_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("h264_crop");
    let h264_path = dir.join("crop_130x98.h264");

    // 130x98 forces coded 144x112 with CropRect { top: 0, bottom: 14, left: 0, right: 14 }.
    let st = Command::new(system_ffmpeg_bin())
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=130x98:rate=25:duration=0.48",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-profile:v",
            "main",
            "-crf",
            "18",
            "-g",
            "6",
            "-keyint_min",
            "6",
            "-bf",
            "0",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-color_range",
            "pc",
            "-x264-params",
            "aud=1:scenecut=0:colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=on",
            h264_path.to_str().unwrap(),
        ])
        .status()
        .expect("run system ffmpeg libx264 crop");
    assert!(st.success(), "libx264 crop stream generation failed");

    let wd_lines = run_wedeo_framecrc(&h264_path).expect("wedeo-framecrc crop");
    let ff_lines = run_ffmpeg_framecrc(&h264_path).expect("ffmpeg oracle crop");
    compare_framecrc_full(&wd_lines, &ff_lines, FramecrcCompareOpts::DATA_ONLY)
        .expect("H.264 cropped 130x98 framecrc mismatch vs FFmpeg n9.1");

    let wd_hdrs = parse_framecrc_headers(&wd_lines);
    assert_eq!(wd_hdrs[0].dimensions.as_deref(), Some("130x98"));

    let frames = decode_h264_frames(&h264_path, "h264", 1);
    assert_eq!(frames.len(), 12);
    for f in &frames {
        assert_eq!(f.width, 144);
        assert_eq!(f.height, 112);
        assert_eq!(f.display_size(), (130, 98));
        assert_eq!(
            f.crop,
            CropRect {
                top: 0,
                bottom: 14,
                left: 0,
                right: 14,
            }
        );
        assert_eq!(f.color.primaries, ColorDesc::BT709_LIMITED.primaries);
        assert_eq!(f.color.range, wedeo::core::ColorRange::Full);
    }
}
