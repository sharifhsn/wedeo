#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use wedeo_codec_h264::{H264Decoder, register_h264};
use wedeo_core::{
    CodecContext, CodecId, ColorDesc, ColorPrimaries, ColorRange, CropRect, Decoder, ErrorPolicy,
    FrameFlags, MatrixCoefficients, MediaUnit, Packet, PictureType, PixelLayout, Rational,
    RecvStatus, Registry, SendStatus, StreamParams, Timestamp, TrackDescriptor, TrackId,
    TransferCharacteristic, VideoFrame, VideoStreamParams,
};
use wedeo_syntax::h264::{NalUnitType, find_annexb_start_code, iter_annexb_nals};

const SYSTEM_FFMPEG: &str = "/usr/bin/ffmpeg";
const ORACLE_FFMPEG: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../target/ffmpeg-oracle/ffmpeg"
);

struct TempDirGuard {
    dir: PathBuf,
}

impl TempDirGuard {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "wedeo_h264_oracle_{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self { dir }
    }

    fn path(&self, file: &str) -> PathBuf {
        self.dir.join(file)
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn run_ffmpeg(bin: &str, args: &[&str]) {
    assert!(Path::new(bin).exists(), "missing ffmpeg binary at {bin}");
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {bin}: {e}"));
    assert!(
        out.status.success(),
        "{bin} {:?} failed:\nstdout: {}\nstderr: {}",
        args,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Split an Annex B stream containing AUD (`nal_unit_type == 9`) boundaries into
/// one `Vec<u8>` per Access Unit so we can attach per-AU VFR timestamps.
fn split_access_units_by_aud(annexb: &[u8]) -> Vec<Vec<u8>> {
    let mut aud_offsets = Vec::new();
    let mut offset = 0usize;
    while let Some((sc_pos, prefix_len)) = find_annexb_start_code(annexb, offset) {
        let nal_start = sc_pos + prefix_len;
        if nal_start < annexb.len() {
            let nal_type = NalUnitType::from_u8(annexb[nal_start] & 0x1F);
            if nal_type == NalUnitType::Aud {
                aud_offsets.push(sc_pos);
            }
        }
        offset = nal_start + 1;
    }
    assert!(
        !aud_offsets.is_empty(),
        "expected AUD NALUs in generated Annex B test stream"
    );

    let mut aus = Vec::with_capacity(aud_offsets.len());
    for (idx, &start) in aud_offsets.iter().enumerate() {
        let end = aud_offsets.get(idx + 1).copied().unwrap_or(annexb.len());
        aus.push(annexb[start..end].to_vec());
    }
    aus
}

fn make_h264_track() -> TrackDescriptor {
    TrackDescriptor::new(
        TrackId(1),
        CodecId::H264,
        Rational::new(1, 90_000),
        StreamParams::Video(VideoStreamParams::new(0, 0, PixelLayout::YUV420P)),
    )
}

fn decode_all_frames(annexb: &[u8], threads: usize) -> (Vec<VideoFrame>, wedeo_core::StreamHealth) {
    let mut reg = Registry::new();
    register_h264(&mut reg);
    let factory = reg
        .find_decoder(CodecId::H264)
        .expect("H.264 decoder registered");

    let ctx = CodecContext {
        threads,
        ..Default::default()
    };
    let mut decoder = factory.open(&make_h264_track(), ctx).expect("open decoder");

    let aus = split_access_units_by_aud(annexb);
    let tb = Rational::new(1, 90_000);
    let mut frames = Vec::new();

    for (i, au) in aus.iter().enumerate() {
        let mut pkt = Packet::from_slice(au);
        pkt.track_id = TrackId(1);
        pkt.time_base = tb;
        pkt.dts = Timestamp::new((i as i64) * 3003);
        pkt.pts = Timestamp::new((i as i64) * 3003);
        pkt.duration = 3003;

        assert_eq!(
            decoder.send_packet(Some(&pkt)).expect("send_packet"),
            SendStatus::Accepted
        );
        loop {
            match decoder.receive().expect("receive") {
                RecvStatus::Ready(MediaUnit::Video(f)) => frames.push(f),
                RecvStatus::Ready(_) => panic!("unexpected non-video media unit"),
                RecvStatus::NeedMore => break,
                RecvStatus::Eof => panic!("unexpected Eof before drain"),
            }
        }
    }

    assert_eq!(
        decoder.send_packet(None).expect("drain"),
        SendStatus::Accepted
    );
    loop {
        match decoder.receive().expect("receive after drain") {
            RecvStatus::Ready(MediaUnit::Video(f)) => frames.push(f),
            RecvStatus::Ready(_) => panic!("unexpected non-video media unit"),
            RecvStatus::NeedMore => panic!("unexpected NeedMore during drain"),
            RecvStatus::Eof => break,
        }
    }

    let health = decoder.health().clone();
    (frames, health)
}

fn assert_yuv420p_frame_matches_oracle(frame: &VideoFrame, oracle_yuv: &[u8], frame_idx: usize) {
    let (disp_w, disp_h) = frame.display_size();
    let w = disp_w as usize;
    let h = disp_h as usize;
    let cw = w / 2;
    let ch = h / 2;
    let crop = frame.crop;

    let y_bytes = w * h;
    let u_bytes = cw * ch;
    let frame_size = y_bytes + u_bytes * 2;
    let base = frame_idx * frame_size;
    let oracle_frame = &oracle_yuv[base..base + frame_size];
    let oracle_y = &oracle_frame[..y_bytes];
    let oracle_u = &oracle_frame[y_bytes..y_bytes + u_bytes];
    let oracle_v = &oracle_frame[y_bytes + u_bytes..];

    let planes = frame.planes().expect("CPU planes");
    assert_eq!(planes.len(), 3);

    for row in 0..h {
        let src_row = planes[0]
            .row(crop.top as usize + row)
            .expect("Y plane row in bounds");
        let got = &src_row[crop.left as usize..crop.left as usize + w];
        let exp = &oracle_y[row * w..(row + 1) * w];
        assert_eq!(got, exp, "Y plane mismatch at frame {frame_idx}, row {row}");
    }

    let c_left = (crop.left / 2) as usize;
    let c_top = (crop.top / 2) as usize;
    for row in 0..ch {
        let u_row = planes[1].row(c_top + row).expect("U plane row in bounds");
        assert_eq!(
            &u_row[c_left..c_left + cw],
            &oracle_u[row * cw..(row + 1) * cw],
            "U plane mismatch at frame {frame_idx}, row {row}"
        );

        let v_row = planes[2].row(c_top + row).expect("V plane row in bounds");
        assert_eq!(
            &v_row[c_left..c_left + cw],
            &oracle_v[row * cw..(row + 1) * cw],
            "V plane mismatch at frame {frame_idx}, row {row}"
        );
    }
}

#[test]
fn oracle_bt709_high_profile_cabac_bframes_and_sar_parity() {
    let tmp = TempDirGuard::new("bt709");
    let h264_path = tmp.path("bt709.h264");
    let yuv_path = tmp.path("bt709_oracle.yuv");

    let h264_str = h264_path.to_str().unwrap();
    let yuv_str = yuv_path.to_str().unwrap();

    // Generate real High-profile CABAC H.264 stream with B-frames, B-pyramid,
    // BT.709 VUI colorimetry, non-1:1 SAR (16:15), and periodic IDR frames.
    run_ffmpeg(
        SYSTEM_FFMPEG,
        &[
            "-y",
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
            h264_str,
        ],
    );

    // Decode with FFmpeg n9.1 golden oracle to raw YUV420p.
    run_ffmpeg(
        ORACLE_FFMPEG,
        &[
            "-y", "-i", h264_str, "-f", "rawvideo", "-pix_fmt", "yuv420p", yuv_str,
        ],
    );

    let annexb = fs::read(&h264_path).expect("read h264");
    let oracle_yuv = fs::read(&yuv_path).expect("read oracle yuv");

    for threads in [1usize, 2usize] {
        let (frames, health) = decode_all_frames(&annexb, threads);
        assert!(health.is_clean(), "expected clean health, got {health:?}");
        assert_eq!(frames.len(), 18, "expected 18 decoded frames");
        assert_eq!(oracle_yuv.len(), 18 * (160 * 112 + 80 * 56 * 2));

        let mut saw_b_frame = false;
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.width, 160);
            assert_eq!(frame.height, 112);
            assert_eq!(frame.crop, CropRect::ZERO);
            assert_eq!(frame.display_size(), (160, 112));
            assert_eq!(frame.color, ColorDesc::BT709_LIMITED);
            assert_eq!(frame.sar, Rational::new(16, 15));
            assert_eq!(frame.time_base, Rational::new(1, 90_000));
            assert_eq!(frame.duration, 3003);

            // Only IDR keyframes (frame 0 and frame 9) must carry FrameFlags::KEY.
            let expected_key = i == 0 || i == 9;
            assert_eq!(
                frame.flags.contains(FrameFlags::KEY),
                expected_key,
                "unexpected KEY flag at frame {i} (picture_type={:?})",
                frame.picture_type
            );
            if frame.picture_type == PictureType::B {
                saw_b_frame = true;
            }

            assert_yuv420p_frame_matches_oracle(frame, &oracle_yuv, i);
        }
        assert!(saw_b_frame, "test stream should exercise B-frames");
    }
}

#[test]
fn oracle_non_mod16_crop_and_full_range_bt709_parity() {
    let tmp = TempDirGuard::new("crop");
    let h264_path = tmp.path("crop.h264");
    let yuv_path = tmp.path("crop_oracle.yuv");

    let h264_str = h264_path.to_str().unwrap();
    let yuv_str = yuv_path.to_str().unwrap();

    // 130x98 forces coded 144x112 with CropRect { top: 0, bottom: 14, left: 0, right: 14 }.
    run_ffmpeg(
        SYSTEM_FFMPEG,
        &[
            "-y",
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
            h264_str,
        ],
    );

    run_ffmpeg(
        ORACLE_FFMPEG,
        &[
            "-y", "-i", h264_str, "-f", "rawvideo", "-pix_fmt", "yuv420p", yuv_str,
        ],
    );

    let annexb = fs::read(&h264_path).expect("read crop.h264");
    let oracle_yuv = fs::read(&yuv_path).expect("read crop_oracle.yuv");

    let (frames, health) = decode_all_frames(&annexb, 1);
    assert!(health.is_clean());
    assert_eq!(frames.len(), 12);
    assert_eq!(oracle_yuv.len(), 12 * (130 * 98 + 65 * 49 * 2));

    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(frame.width, 144);
        assert_eq!(frame.height, 112);
        assert_eq!(
            frame.crop,
            CropRect {
                top: 0,
                bottom: 14,
                left: 0,
                right: 14,
            }
        );
        assert_eq!(frame.display_size(), (130, 98));
        assert_eq!(frame.color.primaries, ColorPrimaries::Bt709);
        assert_eq!(frame.color.transfer, TransferCharacteristic::Bt709);
        assert_eq!(frame.color.matrix, MatrixCoefficients::Bt709);
        assert_eq!(frame.color.range, ColorRange::Full);

        assert_yuv420p_frame_matches_oracle(frame, &oracle_yuv, i);
    }
}

#[test]
fn non_idr_intra_slice_is_not_key_without_recovery_point_sei() {
    let tmp = TempDirGuard::new("open_gop");
    let h264_path = tmp.path("open_gop.h264");
    let h264_str = h264_path.to_str().unwrap();

    // open-gop=1 turns periodic keyframes after frame 0 into non-IDR I-frames
    // accompanied by a recovery_point SEI. We decode both with SEI intact and
    // with SEI stripped to verify FrameFlags::KEY strictly tracks IDR / recovery_point.
    run_ffmpeg(
        SYSTEM_FFMPEG,
        &[
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=30:duration=0.3",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-g",
            "4",
            "-keyint_min",
            "4",
            "-bf",
            "2",
            "-x264-params",
            "aud=1:open-gop=1:scenecut=0:b-adapt=0",
            h264_str,
        ],
    );

    let annexb = fs::read(&h264_path).expect("read open_gop.h264");

    // Strip SEI NAL units while keeping AUD/SPS/PPS/Slice/IDR intact.
    let mut no_sei_annexb = Vec::with_capacity(annexb.len());
    for item in iter_annexb_nals(&annexb) {
        let nal = item.expect("valid annex b nal");
        if nal.header.nal_unit_type != NalUnitType::Sei {
            no_sei_annexb.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            no_sei_annexb.extend_from_slice(nal.raw);
        }
    }

    let (frames_no_sei, _) = decode_all_frames(&no_sei_annexb, 1);
    assert!(frames_no_sei.len() >= 8);
    // Frame 0 is IDR -> KEY. Later I-frames without recovery_point SEI must NOT be KEY.
    assert!(frames_no_sei[0].flags.contains(FrameFlags::KEY));
    let non_idr_i_frames: Vec<_> = frames_no_sei
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, f)| f.picture_type == PictureType::I)
        .collect();
    assert!(
        !non_idr_i_frames.is_empty(),
        "open-gop stream should contain at least one non-IDR I-frame"
    );
    for (idx, frame) in non_idr_i_frames {
        assert!(
            !frame.flags.contains(FrameFlags::KEY),
            "non-IDR I-frame {idx} without recovery_point SEI must not have FrameFlags::KEY"
        );
    }

    // With SEI intact, recovery_point SEI promotes the open-GOP I-frame to KEY.
    let (frames_with_sei, _) = decode_all_frames(&annexb, 1);
    assert_eq!(frames_with_sei.len(), frames_no_sei.len());
    assert!(frames_with_sei[0].flags.contains(FrameFlags::KEY));
    for (idx, frame) in frames_with_sei
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, f)| f.picture_type == PictureType::I)
    {
        assert!(
            frame.flags.contains(FrameFlags::KEY),
            "open-gop I-frame {idx} with recovery_point SEI should have FrameFlags::KEY"
        );
    }
}

#[test]
fn error_policy_strict_vs_resilient_tracks_stream_health() {
    // Minimal SPS + PPS + malformed truncated slice NAL (nal_unit_type = 5, 1 invalid byte).
    let sps_rbsp: &[u8] = &[0x42, 0x80, 0x1E, 0xF4, 0x0A, 0x0F, 0xC0];
    let bad_stream: Vec<u8> = [
        &[0x00, 0x00, 0x00, 0x01, 0x67][..],
        sps_rbsp,
        &[0x00, 0x00, 0x00, 0x01, 0x65, 0x00][..], // IDR referencing nonexistent PPS
    ]
    .concat();

    let pkt = Packet::from_slice(&bad_stream);
    let track = make_h264_track();

    // Resilient mode: records corruption in StreamHealth and does not return Err.
    let mut resilient = H264Decoder::open(
        &track,
        CodecContext {
            threads: 1,
            error_policy: ErrorPolicy::Resilient,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        resilient.send_packet(Some(&pkt)).unwrap(),
        SendStatus::Accepted
    );
    assert!(!resilient.health().is_clean());
    assert!(resilient.health().corrupt_packets >= 1);

    // Strict mode: returns Err AND records the issue in StreamHealth.
    let mut strict = H264Decoder::open(
        &track,
        CodecContext {
            threads: 1,
            error_policy: ErrorPolicy::Strict,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(strict.send_packet(Some(&pkt)).is_err());
    assert!(strict.health().corrupt_packets >= 1);
}
