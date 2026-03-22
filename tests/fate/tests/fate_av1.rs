//! FATE-style integration tests for the AV1 decode pipeline via rav1d.
//!
//! Since wedeo doesn't have an IVF/OBU demuxer, these tests parse IVF
//! inline (trivial format) and feed packets directly to the decoder.
//! Decoded YUV frames are checksummed and compared against FFmpeg's
//! framecrc output for bitexact conformance.

use std::path::{Path, PathBuf};
use std::process::Command;

// Ensure the rav1d decoder is registered.
use wedeo_rav1d as _;

use wedeo_codec::decoder::{CodecParameters, Decoder, DecoderBuilder};
use wedeo_core::buffer::Buffer;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::Error;
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::Packet;

// ---------------------------------------------------------------------------
// Minimal IVF parser
// ---------------------------------------------------------------------------

#[allow(dead_code)] // parsed but not always used; kept for debugging
struct IvfHeader {
    width: u16,
    height: u16,
}

struct IvfFrame {
    timestamp: i64,
    data: Vec<u8>,
}

fn parse_ivf(data: &[u8]) -> Option<(IvfHeader, Vec<IvfFrame>)> {
    if data.len() < 32 {
        return None;
    }
    // Signature: "DKIF"
    if &data[0..4] != b"DKIF" {
        return None;
    }
    let header_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let width = u16::from_le_bytes([data[12], data[13]]);
    let height = u16::from_le_bytes([data[14], data[15]]);

    let header = IvfHeader { width, height };

    let mut frames = Vec::new();
    let mut pos = header_len.max(32);
    while pos + 12 <= data.len() {
        let frame_size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let timestamp = i64::from_le_bytes(data[pos + 4..pos + 12].try_into().ok()?);
        pos += 12;
        if pos + frame_size > data.len() {
            break;
        }
        frames.push(IvfFrame {
            timestamp,
            data: data[pos..pos + frame_size].to_vec(),
        });
        pos += frame_size;
    }

    Some((header, frames))
}

// ---------------------------------------------------------------------------
// Adler-32 matching FFmpeg's non-standard init (s1=0, s2=0)
// ---------------------------------------------------------------------------

fn adler32(data: &[u8]) -> u32 {
    let mut hasher = adler2::Adler32::from_checksum(0);
    hasher.write_slice(data);
    hasher.checksum()
}

// ---------------------------------------------------------------------------
// Decode an IVF file through wedeo's rav1d adapter
// ---------------------------------------------------------------------------

/// Decoded frame info for comparison.
struct DecodedFrame {
    pts: i64,
    size: usize,
    crc: u32,
}

fn decode_ivf(path: &Path) -> Vec<DecodedFrame> {
    let file_data = std::fs::read(path).expect("failed to read IVF file");
    let (_header, frames) = parse_ivf(&file_data).expect("failed to parse IVF");

    let params = CodecParameters::new(CodecId::Av1, MediaType::Video);
    let mut decoder = DecoderBuilder::new(params)
        .open()
        .expect("failed to open AV1 decoder");

    let mut decoded = Vec::new();

    // Feed all packets
    for ivf_frame in &frames {
        let mut pkt = Packet::new();
        pkt.data = Buffer::from_slice(&ivf_frame.data);
        pkt.pts = ivf_frame.timestamp;
        pkt.duration = 1;
        pkt.stream_index = 0;

        decoder.send_packet(Some(&pkt)).expect("send_packet failed");
        drain_decoder(&mut *decoder, &mut decoded);
    }

    // Drain remaining frames
    decoder.send_packet(None).expect("send_packet(None) failed");
    drain_decoder(&mut *decoder, &mut decoded);

    decoded
}

fn drain_decoder(decoder: &mut dyn Decoder, out: &mut Vec<DecodedFrame>) {
    while let Some(frame) = drain_one(decoder) {
        out.push(frame);
    }
}

fn drain_one(decoder: &mut dyn Decoder) -> Option<DecodedFrame> {
    match decoder.receive_frame() {
        Ok(frame) => {
            let video = frame.video()?;
            let width = video.width as usize;
            let height = video.height as usize;
            let chroma_width = width / 2;
            let chroma_height = height / 2;

            let mut raw = Vec::with_capacity(width * height * 3 / 2);

            // Y plane
            let y_plane = &video.planes[0];
            let y_data = y_plane.buffer.data();
            for row in 0..height {
                let start = y_plane.offset + row * y_plane.linesize;
                raw.extend_from_slice(&y_data[start..start + width]);
            }

            // U plane
            if video.planes.len() > 1 {
                let u_plane = &video.planes[1];
                let u_data = u_plane.buffer.data();
                for row in 0..chroma_height {
                    let start = u_plane.offset + row * u_plane.linesize;
                    raw.extend_from_slice(&u_data[start..start + chroma_width]);
                }
            }

            // V plane
            if video.planes.len() > 2 {
                let v_plane = &video.planes[2];
                let v_data = v_plane.buffer.data();
                for row in 0..chroma_height {
                    let start = v_plane.offset + row * v_plane.linesize;
                    raw.extend_from_slice(&v_data[start..start + chroma_width]);
                }
            }

            let crc = adler32(&raw);
            Some(DecodedFrame {
                pts: frame.pts,
                size: raw.len(),
                crc,
            })
        }
        Err(Error::Again) | Err(Error::Eof) => None,
        Err(e) => panic!("decode error: {e:?}"),
    }
}

// ---------------------------------------------------------------------------
// Run FFmpeg framecrc on an IVF file
// ---------------------------------------------------------------------------

fn run_ffmpeg_framecrc(path: &Path) -> Option<Vec<String>> {
    let output = Command::new("ffmpeg")
        .args([
            "-bitexact",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "framecrc",
            "-",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(
        stdout
            .lines()
            .filter(|l| !l.starts_with('#'))
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect(),
    )
}

/// Parse a framecrc data line into (size, checksum).
fn parse_framecrc_line(line: &str) -> Option<(usize, String)> {
    let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
    if fields.len() < 6 {
        return None;
    }
    let size: usize = fields[4].parse().ok()?;
    let crc = fields[5].to_string();
    Some((size, crc))
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn generate_test_ivf(path: &Path, width: u32, height: u32, frames: u32) -> bool {
    let result = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!(
                "testsrc=size={width}x{height}:rate=25:duration={}",
                frames as f64 / 25.0
            ),
            "-c:v",
            "libsvtav1",
            "-preset",
            "8",
            "-crf",
            "35",
            path.to_str().unwrap(),
        ])
        .output();

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
}

fn test_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("wedeo-fate-av1");
    std::fs::create_dir_all(&dir).ok();
    dir
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Smoke test: create decoder, verify it can be opened for AV1.
#[test]
fn av1_decoder_opens() {
    let params = CodecParameters::new(CodecId::Av1, MediaType::Video);
    let decoder = DecoderBuilder::new(params).open();
    assert!(
        decoder.is_ok(),
        "failed to open AV1 decoder: {:?}",
        decoder.err()
    );
    assert_eq!(decoder.unwrap().descriptor().name, "av1_rav1d");
}

/// Decode a small AV1 IVF file and verify we get the expected number of frames.
#[test]
fn av1_decode_frame_count() {
    let ivf_path = test_dir().join("test_framecount.ivf");

    if !generate_test_ivf(&ivf_path, 176, 144, 5) {
        eprintln!("SKIP: ffmpeg with libsvtav1 not available");
        return;
    }

    let decoded = decode_ivf(&ivf_path);
    assert!(
        !decoded.is_empty(),
        "decoder produced no frames from 5-frame IVF"
    );
    assert_eq!(
        decoded.len(),
        5,
        "expected 5 decoded frames, got {}",
        decoded.len()
    );

    // Verify frame dimensions via size: YUV420p 176x144 = 176*144*3/2 = 38016
    for (i, frame) in decoded.iter().enumerate() {
        assert_eq!(
            frame.size, 38016,
            "frame {i} has unexpected size {} (expected 38016 for 176x144 YUV420p)",
            frame.size
        );
    }
}

/// Bitexact test: decode AV1 IVF and compare YUV checksums against FFmpeg.
#[test]
fn av1_bitexact_vs_ffmpeg() {
    let ivf_path = test_dir().join("test_bitexact.ivf");

    if !generate_test_ivf(&ivf_path, 176, 144, 10) {
        eprintln!("SKIP: ffmpeg with libsvtav1 not available");
        return;
    }

    // Decode with wedeo/rav1d
    let decoded = decode_ivf(&ivf_path);
    assert!(!decoded.is_empty(), "decoder produced no frames");

    // Get FFmpeg reference
    let Some(ffmpeg_lines) = run_ffmpeg_framecrc(&ivf_path) else {
        eprintln!("SKIP: ffmpeg not available for cross-validation");
        return;
    };

    assert_eq!(
        decoded.len(),
        ffmpeg_lines.len(),
        "frame count mismatch: wedeo={}, ffmpeg={}",
        decoded.len(),
        ffmpeg_lines.len()
    );

    let mut mismatches = 0;
    for (i, (wedeo_frame, ffmpeg_line)) in decoded.iter().zip(ffmpeg_lines.iter()).enumerate() {
        let Some((ff_size, ff_crc)) = parse_framecrc_line(ffmpeg_line) else {
            panic!("failed to parse ffmpeg framecrc line {i}: {ffmpeg_line}");
        };

        let wedeo_crc = format!("0x{:08x}", wedeo_frame.crc);

        if wedeo_frame.size != ff_size || wedeo_crc != ff_crc {
            eprintln!(
                "DIFF frame {i}: wedeo(size={}, crc={wedeo_crc}) vs ffmpeg(size={ff_size}, crc={ff_crc})",
                wedeo_frame.size,
            );
            mismatches += 1;
        }
    }

    if mismatches == 0 {
        eprintln!(
            "PASS: AV1 BITEXACT — {}/{} frames match FFmpeg",
            decoded.len(),
            decoded.len()
        );
    } else {
        // AV1 is an integer-only spec, so bitexact SHOULD be achievable.
        // If mismatches occur, it could be film grain or a rav1d/dav1d difference.
        panic!(
            "AV1 NOT BITEXACT: {mismatches}/{} frames differ from FFmpeg",
            decoded.len()
        );
    }
}

/// Decode a 10-frame sequence and verify PTS values are monotonically increasing.
#[test]
fn av1_pts_ordering() {
    let ivf_path = test_dir().join("test_pts.ivf");

    if !generate_test_ivf(&ivf_path, 176, 144, 10) {
        eprintln!("SKIP: ffmpeg with libsvtav1 not available");
        return;
    }

    let decoded = decode_ivf(&ivf_path);
    assert!(!decoded.is_empty(), "no frames decoded");

    // PTS should be non-decreasing (may have reordering for B-frames,
    // but testsrc at low frame count typically produces I+P only)
    for window in decoded.windows(2) {
        assert!(
            window[1].pts >= window[0].pts,
            "PTS went backwards: {} -> {}",
            window[0].pts,
            window[1].pts
        );
    }
}
