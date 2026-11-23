//! FATE-style bit-exact VP9 conformance tests against FFmpeg `n9.1` `framecrc`.
//!
//! Verifies all 5 bundled IVF fixtures (including odd-dimension `100x75` and
//! Profile 1 `gbrp`) plus multi-frame intra-predicted tiled VP9 streams generated
//! by system FFmpeg (`libvpx-vp9`) against `/home/sharif/wedeo/target/ffmpeg-oracle/ffmpeg`.

use std::io::Read;
use std::process::Command;

use wedeo::core::{
    CodecContext, CodecId, ErrorPolicy, FileSource, PixelLayout, RecvStatus, SendStatus,
};
use wedeo::default_registry;
use wedeo_oracle::{
    FramecrcCompareOpts, compare_framecrc_full, oracle_data_dir, run_ffmpeg_framecrc,
    run_wedeo_framecrc, system_ffmpeg_bin, temp_test_dir,
};

#[test]
fn vp9_all_five_bundled_ivf_bitexact_vs_ffmpeg_oracle() {
    let fixtures = [
        ("vp9_red_64x64.ivf", 64u32, 64u32, PixelLayout::YUV420P),
        ("vp9_testsrc_128x128.ivf", 128, 128, PixelLayout::YUV420P),
        ("vp9_testsrc_320x240.ivf", 320, 240, PixelLayout::YUV420P),
        ("vp9_blue_100x75.ivf", 100, 75, PixelLayout::YUV420P),
        ("vp9_keyframe_64x64.ivf", 64, 64, PixelLayout::GBRP),
    ];

    let reg = default_registry();

    for (name, exp_w, exp_h, exp_layout) in fixtures {
        let path = oracle_data_dir().join(name);
        assert!(
            path.exists(),
            "missing bundled IVF fixture {}",
            path.display()
        );

        let wedeo_lines = run_wedeo_framecrc(&path)
            .unwrap_or_else(|e| panic!("wedeo-framecrc failed on {name}: {e}"));
        let ffmpeg_lines = run_ffmpeg_framecrc(&path)
            .unwrap_or_else(|e| panic!("ffmpeg oracle failed on {name}: {e}"));

        compare_framecrc_full(
            &wedeo_lines,
            &ffmpeg_lines,
            FramecrcCompareOpts::STRICT_TIMING_AND_HEADERS,
        )
        .unwrap_or_else(|e| panic!("VP9 bit-exact mismatch on {name}:\n{e}"));

        // Also check VideoReader metadata & clean health.
        let mut reader = wedeo::video::VideoReader::open(&reg, &path)
            .unwrap_or_else(|e| panic!("VideoReader failed on {name}: {e}"));
        assert_eq!(reader.header().format_name, "ivf");
        assert_eq!(reader.params().width, exp_w);
        assert_eq!(reader.params().height, exp_h);

        let frame = reader
            .next_frame()
            .unwrap()
            .unwrap_or_else(|| panic!("no frame decoded from {name}"));
        assert_eq!(frame.display_size(), (exp_w, exp_h));
        assert_eq!(frame.layout, exp_layout, "layout mismatch on {name}");
        assert!(reader.next_frame().unwrap().is_none());
        assert!(reader.health().is_clean(), "unclean health on {name}");
    }
}

#[test]
fn vp9_multiframe_tiled_intra_and_threads_parity_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("vp9_intra_multi");
    let ivf_path = dir.join("vp9_intra_160x114.ivf");

    // Odd height (114) + multi-frame intra-only keyframes + 2 tile columns.
    let status = Command::new(system_ffmpeg_bin())
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x114:rate=25:duration=0.24",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "300k",
            "-g",
            "1",
            "-tile-columns",
            "1",
            ivf_path.to_str().unwrap(),
        ])
        .status()
        .expect("run system ffmpeg libvpx-vp9");
    assert!(status.success(), "libvpx-vp9 generation failed");

    let wedeo_lines = run_wedeo_framecrc(&ivf_path).expect("wedeo-framecrc on multi IVF");
    let ffmpeg_lines = run_ffmpeg_framecrc(&ivf_path).expect("ffmpeg oracle on multi IVF");

    compare_framecrc_full(
        &wedeo_lines,
        &ffmpeg_lines,
        FramecrcCompareOpts::STRICT_TIMING_AND_HEADERS,
    )
    .expect("VP9 multi-frame tiled stream must match FFmpeg n9.1 bit-exact");

    // Verify threads=1 vs threads=2 produce identical planes.
    let reg = default_registry();
    let decode_with_threads = |threads: usize| -> Vec<Vec<u8>> {
        let src = FileSource::open(&ivf_path).unwrap();
        let mut head = [0u8; 64];
        let mut peek = FileSource::open(&ivf_path).unwrap();
        let n = peek.read(&mut head).unwrap();
        let mut demux = reg
            .probe_demuxer(&head[..n], Some("ivf"))
            .unwrap()
            .open(Box::new(src), ErrorPolicy::Strict)
            .unwrap();
        let track = demux.header().tracks[0].clone();
        assert_eq!(track.codec_id, CodecId::Vp9);
        let ctx = CodecContext {
            threads,
            ..Default::default()
        };
        let mut dec = reg
            .find_decoder(CodecId::Vp9)
            .unwrap()
            .open(&track, ctx)
            .unwrap();

        let mut out = Vec::new();
        loop {
            match demux.read_packet().unwrap() {
                RecvStatus::Ready(pkt) => {
                    assert_eq!(dec.send_packet(Some(&pkt)).unwrap(), SendStatus::Accepted);
                    while let RecvStatus::Ready(u) = dec.receive().unwrap() {
                        let f = u.into_video().unwrap();
                        let planes = f.planes().unwrap();
                        let mut bytes = Vec::new();
                        for (idx, p) in planes.iter().enumerate() {
                            let (wb, h) = f.layout.plane_dims(f.width, f.height, idx).unwrap();
                            for r in 0..h {
                                bytes.extend_from_slice(&p.row(r).unwrap()[..wb]);
                            }
                        }
                        out.push(bytes);
                    }
                }
                RecvStatus::NeedMore => continue,
                RecvStatus::Eof => break,
            }
        }
        assert_eq!(dec.send_packet(None).unwrap(), SendStatus::Accepted);
        while let RecvStatus::Ready(u) = dec.receive().unwrap() {
            let f = u.into_video().unwrap();
            let planes = f.planes().unwrap();
            let mut bytes = Vec::new();
            for (idx, p) in planes.iter().enumerate() {
                let (wb, h) = f.layout.plane_dims(f.width, f.height, idx).unwrap();
                for r in 0..h {
                    bytes.extend_from_slice(&p.row(r).unwrap()[..wb]);
                }
            }
            out.push(bytes);
        }
        out
    };

    let t1 = decode_with_threads(1);
    let t2 = decode_with_threads(2);
    assert_eq!(t1.len(), 6);
    assert_eq!(t1, t2, "VP9 threads=1 vs threads=2 mismatch");
}
