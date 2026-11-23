//! FATE-style integration tests for the VP9 decode pipeline.
//!
//! These tests verify:
//! 1. Correct demuxing of IVF streams into per-frame packets
//! 2. Correct frame count, dimensions, and pixel format from the VP9 decoder
//! 3. End-to-end decode of a synthetic keyframe IVF file

// Ensure codec and format registrations are linked.
use wedeo_codec_vp9 as _;
use wedeo_format_ivf as _;
use wedeo_format_webm as _;

use wedeo_codec::registry;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::Error;
use wedeo_core::frame::FrameData;
use wedeo_core::pixel_format::PixelFormat;
use wedeo_format::context::InputContext;

#[test]
fn vp9_keyframe_decode_64x64() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("vp9_keyframe_64x64.ivf");

    let mut input = InputContext::open(path.to_str().unwrap()).unwrap();
    let params = input.streams[0].codec_params.clone();
    assert_eq!(params.codec_id, CodecId::Vp9);

    let mut decoder = registry::find_decoder(CodecId::Vp9)
        .expect("VP9 decoder not found")
        .create(params)
        .unwrap();

    let mut frames = Vec::new();
    loop {
        match input.read_packet() {
            Ok(pkt) => {
                decoder.send_packet(Some(&pkt)).unwrap();
                loop {
                    match decoder.receive_frame() {
                        Ok(frame) => frames.push(frame),
                        Err(Error::Again) => break,
                        Err(e) => panic!("receive_frame error: {e:?}"),
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("read_packet error: {e:?}"),
        }
    }
    decoder.send_packet(None).unwrap();
    loop {
        match decoder.receive_frame() {
            Ok(frame) => frames.push(frame),
            Err(Error::Eof) => break,
            Err(e) => panic!("drain error: {e:?}"),
        }
    }

    assert_eq!(frames.len(), 1, "Expected exactly 1 decoded frame");
    if let FrameData::Video(v) = &frames[0].data {
        assert_eq!(v.width, 64, "Expected width 64");
        assert_eq!(v.height, 64, "Expected height 64");
        assert_eq!(v.format, PixelFormat::Yuv420p, "Expected YUV420p");
        assert_eq!(v.planes.len(), 3, "Expected 3 planes");
        assert!(v.planes[0].buffer.size() > 0, "Y plane should be non-empty");
    } else {
        panic!("Expected video frame");
    }
}
