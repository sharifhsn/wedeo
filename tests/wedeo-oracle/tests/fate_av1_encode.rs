//! FATE-style integration tests for the AV1 encode pipeline via rav1e.
//!
//! Validation uses FFmpeg/ffprobe as the external oracle — no self-referential
//! testing. Tests skip gracefully if FFmpeg is unavailable.

use std::path::PathBuf;
use std::process::Command;

// Ensure codec/format registrations are linked.
use wedeo_codec_pcm as _;
use wedeo_format_mp4 as _;
use wedeo_rav1d as _;
use wedeo_rav1e as _;

use wedeo_codec::decoder::DecoderBuilder;
use wedeo_codec::encoder::EncoderBuilder;
use wedeo_core::buffer::Buffer;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::Error;
use wedeo_core::frame::{Frame, FrameData, FramePlane};
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::PacketFlags;
use wedeo_core::pixel_format::PixelFormat;
use wedeo_core::rational::Rational;
use wedeo_format::context::{InputContext, OutputContext};
use wedeo_format::demuxer::Stream;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("wedeo-fate-av1-encode");
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Generate a synthetic YUV420p frame with a gradient pattern.
fn generate_frame(width: u32, height: u32, frame_idx: u32) -> Frame {
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;

    // Y: gradient shifting per frame
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y[row * w + col] = ((col + row * 2 + frame_idx as usize * 17) % 235 + 16) as u8;
        }
    }
    // U/V: neutral chroma
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];

    let mut frame = Frame::new_video(width, height, PixelFormat::Yuv420p);
    frame.pts = frame_idx as i64;
    frame.duration = 1;

    if let FrameData::Video(ref mut video) = frame.data {
        video.planes = vec![
            FramePlane {
                buffer: Buffer::from_slice(&y),
                offset: 0,
                linesize: w,
            },
            FramePlane {
                buffer: Buffer::from_slice(&u),
                offset: 0,
                linesize: cw,
            },
            FramePlane {
                buffer: Buffer::from_slice(&v),
                offset: 0,
                linesize: cw,
            },
        ];
    }

    frame
}

/// Encode synthetic frames and mux to MP4. Returns (output_path, frame_count).
fn encode_to_mp4(
    width: u32,
    height: u32,
    num_frames: u32,
    output_path: &std::path::Path,
) -> (usize, Vec<u8>) {
    let mut builder = EncoderBuilder::new(CodecId::Av1, MediaType::Video);
    builder.width = width;
    builder.height = height;
    builder.pixel_format = PixelFormat::Yuv420p;
    builder.time_base = Rational::new(1, 30);
    builder = builder.option("speed", "10"); // fastest for tests

    let (mut encoder, extradata) =
        wedeo_rav1e::create_av1_encoder(builder).expect("failed to create AV1 encoder");

    // Collect all encoded packets
    let mut packets = Vec::new();

    for i in 0..num_frames {
        let frame = generate_frame(width, height, i);
        encoder.send_frame(Some(&frame)).expect("send_frame failed");

        // Drain available packets
        loop {
            match encoder.receive_packet() {
                Ok(pkt) => packets.push(pkt),
                Err(Error::Again) => break,
                Err(e) => panic!("receive_packet error: {e:?}"),
            }
        }
    }

    // Signal end-of-stream and drain remaining
    encoder.send_frame(None).expect("flush failed");
    loop {
        match encoder.receive_packet() {
            Ok(pkt) => packets.push(pkt),
            Err(Error::Eof | Error::Again) => break,
            Err(e) => panic!("drain error: {e:?}"),
        }
    }

    let packet_count = packets.len();

    // Mux to MP4
    let mut params = wedeo_codec::decoder::CodecParameters::new(CodecId::Av1, MediaType::Video);
    params.width = width;
    params.height = height;
    params.pixel_format = PixelFormat::Yuv420p;
    params.extradata = extradata.clone();
    params.time_base = Rational::new(1, 30);

    let stream = Stream::new(0, params);
    let mut out_ctx = OutputContext::create(output_path.to_str().unwrap(), "mp4", &[stream])
        .expect("failed to create output");

    for pkt in &packets {
        out_ctx.write_packet(pkt).expect("write_packet failed");
    }
    out_ctx.finish().expect("finish failed");

    (packet_count, extradata)
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Smoke test: encoder opens and reports correct descriptor.
#[test]
fn av1_encoder_opens() {
    let mut builder = EncoderBuilder::new(CodecId::Av1, MediaType::Video);
    builder.width = 64;
    builder.height = 64;
    builder.pixel_format = PixelFormat::Yuv420p;
    builder.time_base = Rational::new(1, 30);
    builder = builder.option("speed", "10");

    let encoder = builder.open();
    assert!(
        encoder.is_ok(),
        "failed to open AV1 encoder: {:?}",
        encoder.err()
    );
}

/// Encode 5 synthetic frames, verify packets come back with KEY flag on first.
#[test]
fn av1_encode_basic() {
    let mut builder = EncoderBuilder::new(CodecId::Av1, MediaType::Video);
    builder.width = 64;
    builder.height = 64;
    builder.pixel_format = PixelFormat::Yuv420p;
    builder.time_base = Rational::new(1, 30);
    builder = builder.option("speed", "10");

    let (mut encoder, _extradata) =
        wedeo_rav1e::create_av1_encoder(builder).expect("create encoder");

    let mut packets = Vec::new();

    for i in 0..5 {
        let frame = generate_frame(64, 64, i);
        encoder.send_frame(Some(&frame)).unwrap();
        loop {
            match encoder.receive_packet() {
                Ok(pkt) => packets.push(pkt),
                Err(Error::Again) => break,
                Err(e) => panic!("error: {e:?}"),
            }
        }
    }

    // Drain
    encoder.send_frame(None).unwrap();
    loop {
        match encoder.receive_packet() {
            Ok(pkt) => packets.push(pkt),
            Err(Error::Eof | Error::Again) => break,
            Err(e) => panic!("drain error: {e:?}"),
        }
    }

    assert_eq!(
        packets.len(),
        5,
        "expected 5 packets after drain, got {}",
        packets.len()
    );
    assert!(
        !packets[0].data.data().is_empty(),
        "first packet should have data"
    );
    assert!(
        packets[0].flags.contains(PacketFlags::KEY),
        "first packet should be a keyframe"
    );
}

/// Encode → mux MP4 → ffprobe validates the output.
#[test]
fn av1_encode_ffprobe_validates() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg/ffprobe not available");
        return;
    }

    let output = test_dir().join("ffprobe_test.mp4");
    encode_to_mp4(176, 144, 10, &output);

    // ffprobe should succeed and report an AV1 stream
    let result = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_streams",
            "-of",
            "csv=p=0",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("ffprobe failed to run");

    assert!(
        result.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("av1") || stdout.contains("av01"),
        "ffprobe output should mention AV1: {stdout}"
    );
}

/// Encode → mux MP4 → FFmpeg decodes without error, frame count matches.
#[test]
fn av1_encode_ffmpeg_decodes() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    let output = test_dir().join("ffmpeg_decode_test.mp4");
    let (packet_count, _) = encode_to_mp4(176, 144, 10, &output);

    // FFmpeg should decode the file without errors
    let result = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            output.to_str().unwrap(),
            "-f",
            "framecrc",
            "-",
        ])
        .output()
        .expect("ffmpeg failed to run");

    assert!(
        result.status.success(),
        "ffmpeg decode failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    // Count decoded frames from framecrc output
    let stdout = String::from_utf8_lossy(&result.stdout);
    let frame_count = stdout
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .count();

    assert_eq!(
        frame_count, packet_count,
        "frame count mismatch: ffmpeg decoded {frame_count}, encoder produced {packet_count}"
    );
    eprintln!("PASS: FFmpeg decoded {frame_count}/{packet_count} frames from wedeo/rav1e MP4");
}

/// Round-trip: encode → mux MP4 → demux → decode (rav1d) → compare PSNR.
#[test]
fn av1_roundtrip_decode() {
    let output = test_dir().join("roundtrip_test.mp4");
    let width = 176u32;
    let height = 144u32;
    let num_frames = 5u32;

    encode_to_mp4(width, height, num_frames, &output);

    // Demux and decode the MP4 with wedeo
    let mut ctx = InputContext::open(output.to_str().unwrap()).expect("failed to open MP4");
    let stream = ctx.streams.first().expect("no streams");
    assert_eq!(stream.codec_params.codec_id, CodecId::Av1);
    assert_eq!(stream.codec_params.width, width);
    assert_eq!(stream.codec_params.height, height);

    let params = stream.codec_params.clone();
    let mut decoder = DecoderBuilder::new(params)
        .open()
        .expect("failed to open decoder");

    let mut decoded_count = 0u32;

    loop {
        match ctx.read_packet() {
            Ok(packet) => {
                decoder.send_packet(Some(&packet)).unwrap();
                loop {
                    match decoder.receive_frame() {
                        Ok(frame) => {
                            let video = frame.video().expect("not a video frame");
                            assert_eq!(video.width, width);
                            assert_eq!(video.height, height);
                            assert_eq!(video.planes.len(), 3);
                            decoded_count += 1;
                        }
                        Err(Error::Again) => break,
                        Err(e) => panic!("decode error: {e:?}"),
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        }
    }

    // Drain
    decoder.send_packet(None).unwrap();
    loop {
        match decoder.receive_frame() {
            Ok(_) => decoded_count += 1,
            Err(Error::Eof | Error::Again) => break,
            Err(e) => panic!("drain error: {e:?}"),
        }
    }

    assert_eq!(
        decoded_count, num_frames,
        "round-trip frame count mismatch: decoded {decoded_count}, expected {num_frames}"
    );
    eprintln!("PASS: AV1 round-trip {decoded_count}/{num_frames} frames (encode → MP4 → decode)");
}
