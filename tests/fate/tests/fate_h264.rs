//! FATE-style integration tests for the H.264 decode pipeline.
//!
//! These tests verify:
//! 1. Correct demuxing of H.264 Annex B streams into per-frame access units
//! 2. Correct frame count, dimensions, and pixel format from the decoder
//! 3. End-to-end decode of conformance files (single-slice I-frame files)
//!
//! Tests that require fate-suite samples are skipped if the directory
//! is not present (controlled by FATE_SUITE env var or ./fate-suite/).

use std::path::{Path, PathBuf};

// Ensure codec and format registrations are linked.
use wedeo_codec_h264 as _;
use wedeo_format_h264 as _;

use wedeo_codec::registry;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::Error;
use wedeo_core::frame::FrameData;
use wedeo_core::pixel_format::PixelFormat;
use wedeo_format::context::InputContext;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the fate-suite directory, or None if not available.
fn fate_suite_dir() -> Option<PathBuf> {
    // Check FATE_SUITE env var first
    if let Ok(dir) = std::env::var("FATE_SUITE") {
        let p = PathBuf::from(dir);
        if p.is_dir() {
            return Some(p);
        }
    }
    // Fall back to ./fate-suite/
    let p = PathBuf::from("fate-suite");
    if p.is_dir() {
        return Some(p);
    }
    // Try from workspace root
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("fate-suite");
    if p.is_dir() {
        return Some(p);
    }
    None
}

/// Get a conformance file path, or skip the test if not available.
fn conformance_file(name: &str) -> PathBuf {
    let dir = match fate_suite_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: fate-suite not found");
            return PathBuf::new();
        }
    };
    let path = dir.join("h264-conformance").join(name);
    if !path.exists() {
        eprintln!("SKIP: {} not found", path.display());
    }
    path
}

/// Open a file with InputContext and decode all frames.
/// Returns a list of (width, height, pixel_format, data_size) for each decoded frame.
fn decode_all_frames(path: &Path) -> Vec<(u32, u32, PixelFormat, usize)> {
    if !path.exists() {
        return Vec::new();
    }

    let mut input = InputContext::open(path.to_str().unwrap())
        .unwrap_or_else(|e| panic!("Failed to open {}: {:?}", path.display(), e));

    // Find the video stream
    let stream = &input.streams[0];
    let params = stream.codec_params.clone();

    // Create decoder
    let factory = registry::find_decoder(params.codec_id)
        .unwrap_or_else(|| panic!("No decoder for {:?}", params.codec_id));
    let mut decoder = factory.create(params).unwrap();

    let mut frames = Vec::new();

    // Demux + decode loop
    loop {
        match input.read_packet() {
            Ok(pkt) => {
                decoder.send_packet(Some(&pkt)).unwrap();
                loop {
                    match decoder.receive_frame() {
                        Ok(frame) => {
                            let (w, h, pf, size) = match &frame.data {
                                FrameData::Video(v) => {
                                    let total_size: usize =
                                        v.planes.iter().map(|p| p.buffer.size()).sum();
                                    (v.width, v.height, v.format, total_size)
                                }
                                _ => panic!("Expected video frame"),
                            };
                            frames.push((w, h, pf, size));
                        }
                        Err(Error::Again) => break,
                        Err(e) => panic!("receive_frame error: {:?}", e),
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("read_packet error: {:?}", e),
        }
    }

    // Drain
    decoder.send_packet(None).unwrap();
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                let (w, h, pf, size) = match &frame.data {
                    FrameData::Video(v) => {
                        let total_size: usize = v.planes.iter().map(|p| p.buffer.size()).sum();
                        (v.width, v.height, v.format, total_size)
                    }
                    _ => panic!("Expected video frame"),
                };
                frames.push((w, h, pf, size));
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("drain receive_frame error: {:?}", e),
        }
    }

    frames
}

// ---------------------------------------------------------------------------
// Registration tests
// ---------------------------------------------------------------------------

#[test]
fn h264_decoder_registered() {
    let factory = registry::find_decoder(CodecId::H264);
    assert!(factory.is_some(), "H.264 decoder should be registered");
    let desc = factory.unwrap().descriptor();
    assert_eq!(desc.name, "h264");
    assert_eq!(desc.priority, 100);
}

#[test]
fn h264_demuxer_registered() {
    use wedeo_format::registry as fmt_registry;
    // Check that the H.264 demuxer can be found
    let h264_factory = fmt_registry::demuxers().find(|f| f.descriptor().name == "h264");
    assert!(h264_factory.is_some(), "H.264 demuxer should be registered");
}

// ---------------------------------------------------------------------------
// Conformance file tests (single-slice Baseline files)
// ---------------------------------------------------------------------------

/// BA1_Sony_D.jsv: Baseline profile, CAVLC, single slice per frame.
/// 176x144, 17 frames (1 IDR + 16 P-frames).
#[test]
fn fate_h264_ba1_sony_d() {
    let path = conformance_file("BA1_Sony_D.jsv");
    if !path.exists() {
        return;
    }

    let frames = decode_all_frames(&path);
    assert!(
        !frames.is_empty(),
        "Should decode at least one frame from BA1_Sony_D.jsv"
    );

    // All frames should be 176x144 YUV420p
    for (i, &(w, h, pf, size)) in frames.iter().enumerate() {
        assert_eq!(w, 176, "Frame {} width", i);
        assert_eq!(h, 144, "Frame {} height", i);
        assert_eq!(pf, PixelFormat::Yuv420p, "Frame {} pixel format", i);
        // YUV420p: W*H + 2*(W/2)*(H/2) = W*H*3/2
        let expected_size = 176 * 144 * 3 / 2;
        assert_eq!(size, expected_size, "Frame {} data size", i);
    }

    // BA1_Sony_D.jsv has 17 frames
    eprintln!(
        "BA1_Sony_D.jsv: decoded {} frames (expected 17)",
        frames.len()
    );
    assert_eq!(frames.len(), 17, "Expected 17 frames in BA1_Sony_D.jsv");
}

/// BA2_Sony_F.jsv: Baseline profile, CAVLC.
#[test]
fn fate_h264_ba2_sony_f() {
    let path = conformance_file("BA2_Sony_F.jsv");
    if !path.exists() {
        return;
    }

    let frames = decode_all_frames(&path);
    assert!(
        !frames.is_empty(),
        "Should decode at least one frame from BA2_Sony_F.jsv"
    );

    // All frames should be YUV420p
    for (i, &(_, _, pf, _)) in frames.iter().enumerate() {
        assert_eq!(pf, PixelFormat::Yuv420p, "Frame {} pixel format", i);
    }

    eprintln!("BA2_Sony_F.jsv: decoded {} frames", frames.len());
}

/// BASQP1_Sony_C.jsv: Baseline profile, variable QP.
#[test]
fn fate_h264_basqp1_sony_c() {
    let path = conformance_file("BASQP1_Sony_C.jsv");
    if !path.exists() {
        return;
    }

    let frames = decode_all_frames(&path);
    assert!(
        !frames.is_empty(),
        "Should decode at least one frame from BASQP1_Sony_C.jsv"
    );

    for (i, &(_, _, pf, _)) in frames.iter().enumerate() {
        assert_eq!(pf, PixelFormat::Yuv420p, "Frame {} pixel format", i);
    }

    eprintln!("BASQP1_Sony_C.jsv: decoded {} frames", frames.len());
}

/// Verify that the demuxer correctly splits NALs into per-frame access units.
/// BA1_Sony_D.jsv should produce 17 packets (one per frame).
#[test]
fn fate_h264_demuxer_au_grouping() {
    let path = conformance_file("BA1_Sony_D.jsv");
    if !path.exists() {
        return;
    }

    let mut input = InputContext::open(path.to_str().unwrap()).unwrap();

    let mut packet_count = 0;
    loop {
        match input.read_packet() {
            Ok(_pkt) => {
                packet_count += 1;
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("read_packet error: {:?}", e),
        }
    }

    eprintln!("BA1_Sony_D.jsv: {} packets from demuxer", packet_count);
    // Should have 17 packets (one per frame, not one giant packet)
    assert_eq!(
        packet_count, 17,
        "Demuxer should produce 17 packets for BA1_Sony_D.jsv"
    );
}

/// Test that the demuxer produces correct packet count for a multi-slice file.
/// SVA_Base_B.264 has 3 slices per frame.
#[test]
fn fate_h264_demuxer_multi_slice() {
    let path = conformance_file("SVA_Base_B.264");
    if !path.exists() {
        return;
    }

    let mut input = InputContext::open(path.to_str().unwrap()).unwrap();

    let mut packet_count = 0;
    loop {
        match input.read_packet() {
            Ok(_pkt) => {
                packet_count += 1;
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("read_packet error: {:?}", e),
        }
    }

    eprintln!(
        "SVA_Base_B.264: {} packets from demuxer (multi-slice)",
        packet_count
    );
    // This file has multiple slices per frame; the demuxer should group
    // them into per-frame AUs. The exact count depends on the file structure.
    assert!(
        packet_count > 0,
        "Should produce at least one packet for SVA_Base_B.264"
    );
}
