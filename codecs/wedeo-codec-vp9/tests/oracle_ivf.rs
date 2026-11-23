use std::path::PathBuf;
use std::process::Command;

use wedeo_codec_vp9::register_vp9;
use wedeo_core::{
    CodecContext, CodecId, ColorPrimaries, ColorRange, ErrorKind, MatrixCoefficients, MediaUnit,
    Packet, PixelLayout, Rational, RecvStatus, Registry, SendStatus, StreamParams, TrackDescriptor,
    TrackId, TransferCharacteristic, VideoStreamParams,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Minimal IVF demuxer returning `(width, height, packets)`.
fn read_ivf_packets(data: &[u8]) -> (u32, u32, Vec<(i64, Vec<u8>)>) {
    assert!(data.len() >= 32, "IVF header too short");
    assert_eq!(&data[0..4], b"DKIF", "invalid IVF magic");
    let width = u16::from_le_bytes([data[12], data[13]]) as u32;
    let height = u16::from_le_bytes([data[14], data[15]]) as u32;

    let mut pos = 32usize;
    let mut pkts = Vec::new();
    while pos + 12 <= data.len() {
        let frame_size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let pts = i64::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
            data[pos + 8],
            data[pos + 9],
            data[pos + 10],
            data[pos + 11],
        ]);
        pos += 12;
        assert!(pos + frame_size <= data.len(), "truncated IVF packet");
        pkts.push((pts, data[pos..pos + frame_size].to_vec()));
        pos += frame_size;
    }
    (width, height, pkts)
}

fn oracle_raw_video(ivf_path: &std::path::Path) -> Vec<u8> {
    let ffmpeg = workspace_root().join("target/ffmpeg-oracle/ffmpeg");
    assert!(
        ffmpeg.exists(),
        "FFmpeg n9.1 oracle binary missing at {}",
        ffmpeg.display()
    );
    let out = Command::new(&ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(ivf_path)
        .args(["-f", "rawvideo", "-"])
        .output()
        .expect("failed to execute ffmpeg-oracle");
    assert!(
        out.status.success(),
        "ffmpeg-oracle failed on {}: {}",
        ivf_path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn decode_ivf_with_wedeo(ivf_path: &std::path::Path) -> Vec<wedeo_core::VideoFrame> {
    let bytes = std::fs::read(ivf_path).expect("read IVF file");
    let (w, h, pkts) = read_ivf_packets(&bytes);

    let mut reg = Registry::new();
    register_vp9(&mut reg);
    let factory = reg.find_decoder(CodecId::Vp9).expect("vp9 registered");
    assert_eq!(
        reg.find_decoder_by_name("vp9").unwrap().descriptor().name,
        "vp9"
    );

    let tb = Rational::new(1, 25);
    let track = TrackDescriptor::new(
        TrackId(0),
        CodecId::Vp9,
        tb,
        StreamParams::Video(VideoStreamParams::new(w, h, PixelLayout::YUV420P)),
    );
    let mut dec = factory.open(&track, CodecContext::default()).unwrap();

    let mut frames = Vec::new();
    for (pts, payload) in pkts {
        let mut pkt = Packet::new(wedeo_core::Buffer::from_slice(&payload), TrackId(0), tb);
        pkt.pts = wedeo_core::Timestamp::new(pts);
        pkt.dts = wedeo_core::Timestamp::new(pts);
        assert_eq!(dec.send_packet(Some(&pkt)).unwrap(), SendStatus::Accepted);
        loop {
            match dec.receive().unwrap() {
                RecvStatus::Ready(MediaUnit::Video(vf)) => frames.push(vf),
                RecvStatus::Ready(_) => panic!("unexpected non-video unit"),
                RecvStatus::NeedMore => break,
                RecvStatus::Eof => panic!("unexpected EOF before drain"),
            }
        }
    }
    assert_eq!(dec.send_packet(None).unwrap(), SendStatus::Accepted);
    loop {
        match dec.receive().unwrap() {
            RecvStatus::Ready(MediaUnit::Video(vf)) => frames.push(vf),
            RecvStatus::Ready(_) => panic!("unexpected non-video unit"),
            RecvStatus::NeedMore => panic!("unexpected NeedMore while draining"),
            RecvStatus::Eof => break,
        }
    }
    assert!(dec.health().is_clean(), "expected clean StreamHealth");
    frames
}

fn flatten_planes(vf: &wedeo_core::VideoFrame) -> Vec<u8> {
    let planes = vf.planes().expect("CPU planes");
    let mut out = Vec::new();
    for (p, plane) in planes.iter().enumerate() {
        let (w_bytes, h_rows) = vf
            .layout
            .plane_dims(vf.width, vf.height, p)
            .expect("valid plane dims");
        for y in 0..h_rows {
            let row = plane.row(y).expect("row in bounds");
            out.extend_from_slice(&row[..w_bytes]);
        }
    }
    out
}

#[test]
fn vp9_bundled_ivf_bit_exact_against_ffmpeg_oracle() {
    let data_dir = workspace_root().join("tests/wedeo-oracle/data");
    let cases = [
        ("vp9_red_64x64.ivf", 64, 64, PixelLayout::YUV420P),
        ("vp9_testsrc_128x128.ivf", 128, 128, PixelLayout::YUV420P),
        ("vp9_testsrc_320x240.ivf", 320, 240, PixelLayout::YUV420P),
        ("vp9_blue_100x75.ivf", 100, 75, PixelLayout::YUV420P),
        ("vp9_keyframe_64x64.ivf", 64, 64, PixelLayout::GBRP),
    ];

    for (name, exp_w, exp_h, exp_layout) in cases {
        let path = data_dir.join(name);
        let frames = decode_ivf_with_wedeo(&path);
        assert!(!frames.is_empty(), "no frames decoded for {name}");

        let mut actual_raw = Vec::new();
        for vf in &frames {
            assert_eq!(vf.width, exp_w, "width mismatch for {name}");
            assert_eq!(vf.height, exp_h, "height mismatch for {name}");
            assert_eq!(vf.layout, exp_layout, "layout mismatch for {name}");
            if exp_layout == PixelLayout::GBRP {
                assert_eq!(vf.color.range, ColorRange::Full);
                assert_eq!(vf.color.matrix, MatrixCoefficients::Identity);
                assert_eq!(vf.color.primaries, ColorPrimaries::Bt709);
                assert_eq!(vf.color.transfer, TransferCharacteristic::Srgb);
            } else {
                assert_eq!(vf.color.range, ColorRange::Limited);
            }
            actual_raw.extend_from_slice(&flatten_planes(vf));
        }

        let oracle_raw = oracle_raw_video(&path);
        assert_eq!(
            actual_raw.len(),
            oracle_raw.len(),
            "byte length mismatch for {name}"
        );
        if actual_raw != oracle_raw {
            let mut diffs = Vec::new();
            for (i, (a, b)) in actual_raw.iter().zip(oracle_raw.iter()).enumerate() {
                if a != b {
                    diffs.push((i, *a, *b));
                    if diffs.len() >= 16 {
                        break;
                    }
                }
            }
            let total_diffs = actual_raw
                .iter()
                .zip(oracle_raw.iter())
                .filter(|(a, b)| a != b)
                .count();
            panic!(
                "pixel parity mismatch vs FFmpeg n9.1 oracle for {name}: {total_diffs} differing bytes; first diffs (idx, actual, oracle): {diffs:?}"
            );
        }
    }
}

#[test]
fn vp9_profile2_10bit_returns_explicit_unsupported_error() {
    let mut reg = Registry::new();
    register_vp9(&mut reg);
    let factory = reg.find_decoder(CodecId::Vp9).unwrap();
    let track = TrackDescriptor::new(
        TrackId(0),
        CodecId::Vp9,
        Rational::new(1, 25),
        StreamParams::Video(VideoStreamParams::new(16, 16, PixelLayout::YUV420P10LE)),
    );
    let mut dec = factory.open(&track, CodecContext::default()).unwrap();

    // Minimal valid VP9 Profile 2 (10-bit) keyframe header + 4-byte compressed header.
    struct BitWriter(Vec<u8>, u8, u8);
    impl BitWriter {
        fn push_bits(&mut self, val: u32, n: u8) {
            for i in (0..n).rev() {
                self.1 = (self.1 << 1) | (((val >> i) & 1) as u8);
                self.2 += 1;
                if self.2 == 8 {
                    self.0.push(self.1);
                    self.1 = 0;
                    self.2 = 0;
                }
            }
        }
        fn align(&mut self) {
            if self.2 != 0 {
                self.1 <<= 8 - self.2;
                self.0.push(self.1);
                self.1 = 0;
                self.2 = 0;
            }
        }
    }
    let mut bw = BitWriter(Vec::new(), 0, 0);
    bw.push_bits(0b10, 2); // frame_marker
    bw.push_bits(0, 1); // profile_low
    bw.push_bits(1, 1); // profile_high => profile 2
    bw.push_bits(0, 1); // show_existing_frame
    bw.push_bits(0, 1); // frame_type = KeyFrame
    bw.push_bits(1, 1); // show_frame
    bw.push_bits(0, 1); // error_resilient
    bw.push_bits(0x49_83_42, 24); // sync code
    bw.push_bits(0, 1); // ten_or_twelve_bit = 0 (10-bit)
    bw.push_bits(0, 3); // color_space = Unknown
    bw.push_bits(0, 1); // color_range = Limited
    bw.push_bits(15, 16); // width - 1
    bw.push_bits(15, 16); // height - 1
    bw.push_bits(0, 1); // render_and_frame_size_different
    bw.push_bits(0, 1); // refresh_frame_context
    bw.push_bits(0, 1); // frame_parallel_decoding_mode
    bw.push_bits(0, 2); // frame_context_idx
    bw.push_bits(0, 6); // loop_filter_level
    bw.push_bits(0, 3); // sharpness_level
    bw.push_bits(0, 1); // mode_ref_delta_enabled
    bw.push_bits(0, 8); // base_q_idx
    bw.push_bits(0, 1); // delta_q_y_dc
    bw.push_bits(0, 1); // delta_q_uv_dc
    bw.push_bits(0, 1); // delta_q_uv_ac
    bw.push_bits(0, 1); // segmentation_enabled
    bw.push_bits(0, 1); // tile_rows_log2 = 0
    bw.push_bits(4, 16); // compressed_header_size = 4 bytes
    bw.align();
    bw.0.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    let pkt = Packet::from_slice(&bw.0);
    let err = dec
        .send_packet(Some(&pkt))
        .expect_err("expected explicit unsupported error for 10-bit Profile 2");
    assert_eq!(err.kind, ErrorKind::Unsupported);
}

#[test]
fn vp9_corrupt_packet_records_stream_health_and_flush_resets() {
    let mut reg = Registry::new();
    register_vp9(&mut reg);
    let factory = reg.find_decoder(CodecId::Vp9).unwrap();
    let track = TrackDescriptor::new(
        TrackId(0),
        CodecId::Vp9,
        Rational::new(1, 25),
        StreamParams::Video(VideoStreamParams::new(64, 64, PixelLayout::YUV420P)),
    );
    let mut dec = factory.open(&track, CodecContext::default()).unwrap();

    let bad_pkt = Packet::from_slice(&[0x00; 16]);
    let err = dec.send_packet(Some(&bad_pkt)).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidData);
    assert_eq!(dec.health().corrupt_packets, 1);

    assert_eq!(dec.send_packet(None).unwrap(), SendStatus::Accepted);
    assert!(matches!(dec.receive().unwrap(), RecvStatus::Eof));
    dec.flush();
    assert!(matches!(dec.receive().unwrap(), RecvStatus::NeedMore));
}

fn generate_libvpx_ivf(
    name: &str,
    width: u32,
    height: u32,
    frames: u32,
    extra_args: &[&str],
) -> PathBuf {
    let out_path =
        std::env::temp_dir().join(format!("wedeo_vp9_{name}_{}.ivf", std::process::id()));
    let filter = format!("testsrc2=size={width}x{height}:rate=25");
    let frames_str = frames.to_string();
    let mut cmd = Command::new("/usr/bin/ffmpeg");
    cmd.args([
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        &filter,
        "-frames:v",
        &frames_str,
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "200k",
    ]);
    cmd.args(extra_args);
    cmd.arg(&out_path);
    let out = cmd.output().expect("failed to execute /usr/bin/ffmpeg");
    assert!(
        out.status.success(),
        "/usr/bin/ffmpeg failed for {name}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out_path
}

fn assert_ivf_bit_exact(
    name: &str,
    ivf_path: &std::path::Path,
    exp_w: u32,
    exp_h: u32,
    exp_frames: usize,
) {
    let frames = decode_ivf_with_wedeo(ivf_path);
    assert_eq!(
        frames.len(),
        exp_frames,
        "decoded frame count mismatch for {name}"
    );

    let oracle_raw = oracle_raw_video(ivf_path);
    let y_bytes = (exp_w * exp_h) as usize;
    let uv_bytes = (exp_w.div_ceil(2) * exp_h.div_ceil(2)) as usize;
    let frame_bytes = y_bytes + 2 * uv_bytes;
    assert_eq!(
        oracle_raw.len(),
        exp_frames * frame_bytes,
        "oracle raw length mismatch for {name}"
    );

    let mut actual_raw = Vec::with_capacity(oracle_raw.len());
    for (idx, vf) in frames.iter().enumerate() {
        assert_eq!(vf.width, exp_w, "width mismatch at frame {idx} for {name}");
        assert_eq!(
            vf.height, exp_h,
            "height mismatch at frame {idx} for {name}"
        );
        assert_eq!(
            vf.layout,
            PixelLayout::YUV420P,
            "layout mismatch at frame {idx} for {name}"
        );
        let actual_frame = flatten_planes(vf);
        let oracle_frame = &oracle_raw[idx * frame_bytes..(idx + 1) * frame_bytes];
        if actual_frame != oracle_frame {
            let mut diffs = Vec::new();
            let mut total_diffs = 0usize;
            for (i, (a, b)) in actual_frame.iter().zip(oracle_frame.iter()).enumerate() {
                if a != b {
                    total_diffs += 1;
                    if diffs.len() < 12 {
                        let (plane, py, px) = if i < y_bytes {
                            ("Y", i / exp_w as usize, i % exp_w as usize)
                        } else if i < y_bytes + uv_bytes {
                            let j = i - y_bytes;
                            let cw = exp_w.div_ceil(2) as usize;
                            ("U", j / cw, j % cw)
                        } else {
                            let j = i - y_bytes - uv_bytes;
                            let cw = exp_w.div_ceil(2) as usize;
                            ("V", j / cw, j % cw)
                        };
                        diffs.push(format!("{plane}({px},{py}): act={a} exp={b}"));
                    }
                }
            }
            panic!(
                "pixel parity mismatch vs FFmpeg n9.1 oracle for {name} at frame {idx}: {total_diffs} differing bytes; first diffs: {diffs:?}"
            );
        }
        actual_raw.extend_from_slice(&actual_frame);
    }
    assert_eq!(actual_raw, oracle_raw);
}

#[test]
fn vp9_inter_clip1_p_frames_bit_exact_against_ffmpeg_oracle() {
    let ivf = generate_libvpx_ivf("clip1_inter", 160, 112, 15, &["-auto-alt-ref", "0"]);
    assert_ivf_bit_exact("clip1_inter", &ivf, 160, 112, 15);
    let _ = std::fs::remove_file(&ivf);
}

#[test]
fn vp9_inter_clip2_altref_compound_bit_exact_against_ffmpeg_oracle() {
    let ivf = generate_libvpx_ivf(
        "clip2_altref",
        160,
        112,
        15,
        &["-auto-alt-ref", "1", "-lag-in-frames", "10"],
    );
    assert_ivf_bit_exact("clip2_altref", &ivf, 160, 112, 15);
    let _ = std::fs::remove_file(&ivf);
}

#[test]
fn vp9_inter_clip3_odd_dim_130x98_bit_exact_against_ffmpeg_oracle() {
    let ivf = generate_libvpx_ivf("clip3_odd", 130, 98, 10, &["-auto-alt-ref", "0"]);
    assert_ivf_bit_exact("clip3_odd", &ivf, 130, 98, 10);
    let _ = std::fs::remove_file(&ivf);
}

#[test]
fn vp9_inter_clip4_multi_tile_320x240_bit_exact_against_ffmpeg_oracle() {
    let ivf = generate_libvpx_ivf(
        "clip4_tiles",
        320,
        240,
        10,
        &["-tile-columns", "1", "-auto-alt-ref", "0"],
    );
    assert_ivf_bit_exact("clip4_tiles", &ivf, 320, 240, 10);
    let _ = std::fs::remove_file(&ivf);
}
