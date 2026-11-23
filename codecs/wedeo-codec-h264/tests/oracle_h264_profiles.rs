//! Multi-profile H.264 oracle parity: Baseline CAVLC, Main CABAC and High 8x8
//! CABAC B-pyramid streams (libx264) decoded by wedeo (with runtime SIMD
//! dispatch active) must be byte-identical to the FFmpeg `n9.1` oracle.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use wedeo_codec_h264::register_h264;
use wedeo_core::{
    CodecContext, CodecId, MediaUnit, Packet, PixelLayout, Rational, RecvStatus, Registry,
    SendStatus, StreamParams, Timestamp, TrackDescriptor, TrackId, VideoFrame, VideoStreamParams,
};
use wedeo_syntax::h264::{NalUnitType, find_annexb_start_code};

const SYSTEM_FFMPEG: &str = "/usr/bin/ffmpeg";
const ORACLE_FFMPEG: &str = "/home/sharif/wedeo/target/ffmpeg-oracle/ffmpeg";

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "wedeo_h264_profiles_{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }
    fn path(&self, f: &str) -> PathBuf {
        self.0.join(f)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn run(bin: &str, args: &[&str]) {
    assert!(Path::new(bin).exists(), "missing binary {bin}");
    let out = Command::new(bin).args(args).output().expect("spawn");
    assert!(
        out.status.success(),
        "{bin} {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn split_aus(annexb: &[u8]) -> Vec<Vec<u8>> {
    let mut auds = Vec::new();
    let mut off = 0usize;
    while let Some((sc, plen)) = find_annexb_start_code(annexb, off) {
        let start = sc + plen;
        if start < annexb.len() && NalUnitType::from_u8(annexb[start] & 0x1F) == NalUnitType::Aud {
            auds.push(sc);
        }
        off = start + 1;
    }
    assert!(!auds.is_empty(), "no AUDs in stream");
    auds.iter()
        .enumerate()
        .map(|(i, &s)| annexb[s..auds.get(i + 1).copied().unwrap_or(annexb.len())].to_vec())
        .collect()
}

fn decode(annexb: &[u8]) -> Vec<VideoFrame> {
    let mut reg = Registry::new();
    register_h264(&mut reg);
    let factory = reg.find_decoder(CodecId::H264).expect("h264 registered");
    let track = TrackDescriptor::new(
        TrackId(1),
        CodecId::H264,
        Rational::new(1, 90_000),
        StreamParams::Video(VideoStreamParams::new(0, 0, PixelLayout::YUV420P)),
    );
    let mut dec = factory
        .open(
            &track,
            CodecContext {
                threads: 1,
                ..Default::default()
            },
        )
        .expect("open");
    let mut frames = Vec::new();
    for (i, au) in split_aus(annexb).iter().enumerate() {
        let mut pkt = Packet::from_slice(au);
        pkt.track_id = TrackId(1);
        pkt.time_base = Rational::new(1, 90_000);
        pkt.dts = Timestamp::new(i as i64 * 3003);
        pkt.pts = Timestamp::new(i as i64 * 3003);
        pkt.duration = 3003;
        assert_eq!(dec.send_packet(Some(&pkt)).unwrap(), SendStatus::Accepted);
        while let RecvStatus::Ready(MediaUnit::Video(f)) = dec.receive().unwrap() {
            frames.push(f);
        }
    }
    dec.send_packet(None).unwrap();
    loop {
        match dec.receive().unwrap() {
            RecvStatus::Ready(MediaUnit::Video(f)) => frames.push(f),
            RecvStatus::Ready(_) => panic!("non-video unit"),
            RecvStatus::NeedMore | RecvStatus::Eof => break,
        }
    }
    assert!(dec.health().is_clean(), "unclean health {:?}", dec.health());
    frames
}

fn frame_to_yuv(f: &VideoFrame) -> Vec<u8> {
    let (w, h) = f.display_size();
    let (w, h) = (w as usize, h as usize);
    let crop = f.crop;
    let planes = f.planes().expect("planes");
    let mut out = Vec::with_capacity(w * h * 3 / 2);
    for row in 0..h {
        let r = planes[0].row(crop.top as usize + row).unwrap();
        out.extend_from_slice(&r[crop.left as usize..crop.left as usize + w]);
    }
    for plane in &planes[1..3] {
        for row in 0..h / 2 {
            let r = plane.row(crop.top as usize / 2 + row).unwrap();
            let l = crop.left as usize / 2;
            out.extend_from_slice(&r[l..l + w / 2]);
        }
    }
    out
}

fn check_profile(name: &str, size: &str, encode_args: &[&str]) {
    let tmp = TempDir::new(name);
    let h264 = tmp.path("s.h264");
    let yuv = tmp.path("o.yuv");
    let h264_s = h264.to_str().unwrap();
    let yuv_s = yuv.to_str().unwrap();

    let src = format!("testsrc2=size={size}:rate=30:duration=0.8");
    let mut args = vec![
        "-y",
        "-f",
        "lavfi",
        "-i",
        src.as_str(),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
    ];
    args.extend_from_slice(encode_args);
    args.push(h264_s);
    run(SYSTEM_FFMPEG, &args);
    run(
        ORACLE_FFMPEG,
        &[
            "-y", "-i", h264_s, "-f", "rawvideo", "-pix_fmt", "yuv420p", yuv_s,
        ],
    );

    let oracle = fs::read(&yuv).unwrap();
    let frames = decode(&fs::read(&h264).unwrap());
    let mut got = Vec::new();
    for f in &frames {
        got.extend(frame_to_yuv(f));
    }
    assert_eq!(got.len(), oracle.len(), "{name}: output size mismatch");
    if let Some(pos) = got.iter().zip(&oracle).position(|(a, b)| a != b) {
        panic!(
            "{name}: first mismatch at byte {pos} (SIMD dispatch = {:?})",
            wedeo_codec_h264::simd::detected_backend()
        );
    }
    eprintln!(
        "{name}: {} frames bit-exact vs FFmpeg n9.1 (SIMD = {:?})",
        frames.len(),
        wedeo_codec_h264::simd::detected_backend()
    );
}

#[test]
fn oracle_baseline_cavlc_parity() {
    check_profile(
        "baseline",
        "176x144",
        &[
            "-profile:v",
            "baseline",
            "-crf",
            "22",
            "-g",
            "10",
            "-bf",
            "0",
            "-refs",
            "3",
            "-x264-params",
            "aud=1:scenecut=0:cabac=0",
        ],
    );
}

#[test]
fn oracle_main_cabac_parity() {
    check_profile(
        "main",
        "160x112",
        &[
            "-profile:v",
            "main",
            "-crf",
            "20",
            "-g",
            "12",
            "-bf",
            "2",
            "-refs",
            "2",
            "-x264-params",
            "aud=1:scenecut=0:b-adapt=0",
        ],
    );
}

#[test]
fn oracle_high_8x8_cabac_bpyramid_parity() {
    check_profile(
        "high",
        "192x128",
        &[
            "-profile:v",
            "high",
            "-crf",
            "18",
            "-g",
            "12",
            "-bf",
            "3",
            "-refs",
            "4",
            "-x264-params",
            "aud=1:scenecut=0:8x8dct=1:b-pyramid=normal:b-adapt=0:weightp=0",
        ],
    );
}
