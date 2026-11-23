//! FATE-style live AV1 decode tests via `wedeo::default_registry()` (`rav1d` zero-copy
//! `CpuExternal` + odd-dimension `div_ceil` chroma + CICP colorimetry) vs FFmpeg.

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

fn generate_even_av1_ivf(path: &Path, width: u32, height: u32, frames: u32) {
    let status = Command::new(system_ffmpeg_bin())
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!(
                "testsrc2=size={width}x{height}:rate=25:duration={:.3}",
                frames as f64 / 25.0
            ),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "32",
            path.to_str().unwrap(),
        ])
        .status()
        .expect("run system ffmpeg libaom-av1");
    assert!(status.success(), "libaom-av1 IVF generation failed");
}

fn generate_odd_bt709_av1_ivf(path: &Path, width: u32, height: u32, frames: u32) {
    let reg = default_registry();
    let pool = BufferPool::new();
    let tb = Rational::new(1, 25);

    let mut vparams = VideoStreamParams::new(width, height, PixelLayout::YUV420P);
    vparams.color = ColorDesc::BT709_LIMITED;
    let track = TrackDescriptor::new(
        TrackId(0),
        CodecId::Av1,
        tb,
        StreamParams::Video(vparams.clone()),
    );
    let mut enc = reg
        .find_encoder(CodecId::Av1)
        .unwrap()
        .open(
            &track,
            CodecContext::new(Arc::clone(&pool)),
            PassMode::Single,
        )
        .unwrap();

    let mut packets = Vec::new();
    for i in 0..frames {
        let mut y = BorderedPlane::new(&pool, width as usize, height as usize, BorderPadding::NONE);
        for r in 0..height as usize {
            let row = y.row_mut(r).unwrap();
            for (c, px) in row.iter_mut().enumerate() {
                *px = (((c * 3 + r * 5 + i as usize * 13) % 219) + 16) as u8;
            }
        }
        let cw = width.div_ceil(2) as usize;
        let ch = height.div_ceil(2) as usize;
        let mut u = BorderedPlane::new(&pool, cw, ch, BorderPadding::NONE);
        let mut v = BorderedPlane::new(&pool, cw, ch, BorderPadding::NONE);
        for r in 0..ch {
            u.row_mut(r).unwrap().fill(110);
            v.row_mut(r).unwrap().fill(145);
        }
        let mut frame = VideoFrame::from_planes(width, height, PixelLayout::YUV420P, vec![y, u, v]);
        frame.pts = Timestamp::new(i as i64);
        frame.duration = 1;
        frame.time_base = tb;
        frame.color = ColorDesc::BT709_LIMITED;
        assert_eq!(
            enc.send_unit(Some(&MediaUnit::Video(frame)), EncodeHint::default())
                .unwrap(),
            SendStatus::Accepted
        );
        while let RecvStatus::Ready(p) = enc.receive_packet().unwrap() {
            packets.push(p);
        }
    }
    assert_eq!(
        enc.send_unit(None, EncodeHint::default()).unwrap(),
        SendStatus::Accepted
    );
    loop {
        match enc.receive_packet().unwrap() {
            RecvStatus::Ready(p) => packets.push(p),
            RecvStatus::NeedMore => continue,
            RecvStatus::Eof => break,
        }
    }

    let mut hdr = ContainerHeader::new("ivf");
    hdr.tracks.push(track);
    let sink = FileSink::create(path).unwrap();
    let mut muxer = reg
        .find_muxer_by_name("ivf")
        .unwrap()
        .create(Box::new(sink))
        .unwrap();
    muxer.write_header(&hdr).unwrap();
    for p in &packets {
        muxer.write_packet(p).unwrap();
    }
    muxer.finalize().unwrap();
}

/// Run AV1 `framecrc` decode using `libdav1d` on system FFmpeg (the pinned
/// `ffmpeg-oracle` build omits `libdav1d` and only builds hwaccel `av1`).
fn run_av1_ffmpeg_framecrc(path: &Path) -> Vec<String> {
    let out = Command::new(system_ffmpeg_bin())
        .args([
            "-v",
            "error",
            "-bitexact",
            "-c:v",
            "libdav1d",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "framecrc",
            "-flags",
            "+bitexact",
            "-fflags",
            "+bitexact",
            "-",
        ])
        .output()
        .expect("run ffmpeg libdav1d framecrc");
    assert!(
        out.status.success(),
        "ffmpeg libdav1d framecrc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(ToOwned::to_owned)
        .collect()
}

#[test]
fn av1_decoder_opens_via_default_registry() {
    let reg = default_registry();
    let factory = reg
        .find_decoder(CodecId::Av1)
        .expect("av1_rav1d registered");
    assert_eq!(factory.descriptor().name, "av1_rav1d");
}

#[test]
fn av1_even_and_odd_dims_bitexact_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("av1_decode");

    let even_ivf = dir.join("av1_176x144.ivf");
    generate_even_av1_ivf(&even_ivf, 176, 144, 8);

    let odd_ivf = dir.join("av1_odd_101x75_bt709.ivf");
    generate_odd_bt709_av1_ivf(&odd_ivf, 101, 75, 5);

    // Confirm ffprobe oracle sees exact odd dimensions 101x75.
    let probe = Command::new(ffprobe_oracle_bin())
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
            odd_ivf.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(probe.status.success());
    let probe_txt = String::from_utf8_lossy(&probe.stdout);
    assert!(probe_txt.contains("width=101"), "{probe_txt}");
    assert!(probe_txt.contains("height=75"), "{probe_txt}");

    for (ivf, w, h, count, bt709) in [
        (&even_ivf, 176u32, 144u32, 8u32, false),
        (&odd_ivf, 101, 75, 5, true),
    ] {
        let wd_lines = run_wedeo_framecrc(ivf).expect("wedeo-framecrc AV1 IVF");
        let ff_lines = run_av1_ffmpeg_framecrc(ivf);

        compare_framecrc_full(
            &wd_lines,
            &ff_lines,
            FramecrcCompareOpts::STRICT_TIMING_AND_HEADERS,
        )
        .unwrap_or_else(|e| panic!("AV1 decode framecrc mismatch on {}:\n{e}", ivf.display()));

        let reg = default_registry();
        let mut reader = wedeo::video::VideoReader::open(&reg, ivf).unwrap();
        assert_eq!(reader.header().format_name, "ivf");

        let mut prev_pts = None;
        let mut decoded = 0u32;
        while let Some(frame) = reader.next_frame().unwrap() {
            assert_eq!(frame.display_size(), (w, h));
            assert_eq!(frame.layout, PixelLayout::YUV420P);
            let planes = frame.planes().expect("CPU planes");
            assert_eq!(planes.len(), 3);
            assert_eq!(planes[1].width_bytes, w.div_ceil(2) as usize);
            assert_eq!(planes[1].height, h.div_ceil(2) as usize);
            if bt709 {
                assert_eq!(frame.color.primaries, ColorDesc::BT709_LIMITED.primaries);
                assert_eq!(frame.color.transfer, ColorDesc::BT709_LIMITED.transfer);
                assert_eq!(frame.color.matrix, ColorDesc::BT709_LIMITED.matrix);
            }
            let pts = frame.pts.get().unwrap();
            if let Some(p) = prev_pts {
                assert!(pts >= p, "PTS went backwards: {p} -> {pts}");
            }
            prev_pts = Some(pts);
            decoded += 1;
        }
        assert_eq!(decoded, count);
        assert!(reader.health().is_clean());
    }
}
