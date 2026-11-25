//! Pure-Rust port of FFmpeg `n9.1` `SwsOpList` video scaling, pixel layout,
//! chroma siting, colorimetry, HDR tone-mapping, crop, rotation, and overlay ops.

#![forbid(unsafe_code)]

pub mod color;
pub mod cpu;
pub mod ops;
pub mod optimizer;

pub use cpu::{
    CompiledSws, CompiledWgsl, CpuCompiledSws, CpuSwsBackend, SwsBackend, TILE_H, TILE_W,
};
pub use ops::{
    DisplayTransform, DitherMode, IDENTITY_DISPLAY_MATRIX, SwsOp, SwsOpList, SwsOptions, SwsScaler,
    ToneMapOp, VideoContract, display_matrix,
};

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::{
        BorderPadding, BorderedPlane, BufferPool, ChromaLocation, ColorDesc, ColorRange, CropRect,
        MasteringDisplayMeta, MatrixCoefficients, MemoryDomain, PixelLayout, PositionedPlane,
        Rational, SideData, Timestamp, TransferCharacteristic, VideoFrame,
    };

    fn make_solid_yuv420p(
        pool: &std::sync::Arc<BufferPool>,
        w: u32,
        h: u32,
        y: u8,
        u: u8,
        v: u8,
        color: ColorDesc,
    ) -> VideoFrame {
        let mut py = BorderedPlane::new(pool, w as usize, h as usize, BorderPadding::NONE);
        for row in 0..h as usize {
            py.row_mut(row).unwrap().fill(y);
        }
        let (cw, ch) = PixelLayout::YUV420P.subsampling.chroma_dims(w, h);
        let mut pu = BorderedPlane::new(pool, cw as usize, ch as usize, BorderPadding::NONE);
        let mut pv = BorderedPlane::new(pool, cw as usize, ch as usize, BorderPadding::NONE);
        for row in 0..ch as usize {
            pu.row_mut(row).unwrap().fill(u);
            pv.row_mut(row).unwrap().fill(v);
        }
        let mut f = VideoFrame::from_planes(w, h, PixelLayout::YUV420P, vec![py, pu, pv]);
        f.color = color;
        f
    }

    fn make_solid_yuv420p10le(
        pool: &std::sync::Arc<BufferPool>,
        w: u32,
        h: u32,
        y: u16,
        u: u16,
        v: u16,
        color: ColorDesc,
    ) -> VideoFrame {
        let mut py = BorderedPlane::new(pool, w as usize * 2, h as usize, BorderPadding::NONE);
        let y_bytes = y.to_le_bytes();
        for row in 0..h as usize {
            let r = py.row_mut(row).unwrap();
            for chunk in r.as_chunks_mut::<2>().0 {
                chunk.copy_from_slice(&y_bytes);
            }
        }
        let (cw, ch) = PixelLayout::YUV420P10LE.subsampling.chroma_dims(w, h);
        let mut pu = BorderedPlane::new(pool, cw as usize * 2, ch as usize, BorderPadding::NONE);
        let mut pv = BorderedPlane::new(pool, cw as usize * 2, ch as usize, BorderPadding::NONE);
        let u_bytes = u.to_le_bytes();
        let v_bytes = v.to_le_bytes();
        for row in 0..ch as usize {
            for chunk in pu.row_mut(row).unwrap().as_chunks_mut::<2>().0 {
                chunk.copy_from_slice(&u_bytes);
            }
            for chunk in pv.row_mut(row).unwrap().as_chunks_mut::<2>().0 {
                chunk.copy_from_slice(&v_bytes);
            }
        }
        let mut f = VideoFrame::from_planes(w, h, PixelLayout::YUV420P10LE, vec![py, pu, pv]);
        f.color = color;
        f
    }

    #[test]
    fn vertical_downscale_retains_the_first_rows_of_each_band() {
        let pool = BufferPool::new();
        let mut plane = BorderedPlane::new(&pool, 4, 96, BorderPadding::NONE);
        for y in 0..96 {
            plane
                .row_mut(y)
                .unwrap()
                .copy_from_slice(&[y as u8, 0, 0, 255]);
        }
        let mut src = VideoFrame::from_planes(1, 96, PixelLayout::RGBA8, vec![plane]);
        src.color = ColorDesc::SRGB;
        let from = VideoContract::from_frame(&src);
        let to = VideoContract::new(1, 32, PixelLayout::RGBA8).with_color(src.color);
        let options = SwsOptions {
            scaler: SwsScaler::Point,
            dither: DitherMode::None,
            ..Default::default()
        };
        let ops = SwsOpList::build(&from, &to, options).unwrap();
        let result = CpuSwsBackend::new()
            .compile(&ops)
            .unwrap()
            .execute(&src, &pool)
            .unwrap();
        for y in 0..32 {
            assert_eq!(
                result.planes().unwrap()[0].row(y).unwrap()[0],
                (3 * y + 1) as u8
            );
        }
    }

    #[test]
    fn noop_zero_copy_passthrough_preserves_buffer_and_vfr() {
        let pool = BufferPool::new();
        let mut frame = make_solid_yuv420p(&pool, 64, 48, 120, 128, 128, ColorDesc::BT709_LIMITED);
        frame.pts = Timestamp::new(90090);
        frame.pkt_dts = Timestamp::new(87087);
        frame.duration = 3003;
        frame.time_base = Rational::new(1, 90000);

        let src_c = VideoContract::from_frame(&frame);
        let dst_c = src_c.clone();
        let ops = SwsOpList::build(&src_c, &dst_c, SwsOptions::default()).unwrap();

        assert!(
            ops.is_noop(),
            "expected is_noop() == true, ops={:?}",
            ops.ops
        );
        assert_eq!(ops.ops.len(), 2);

        let mut compiled = CpuSwsBackend::new().compile(&ops).unwrap();
        let out = compiled.execute(&frame, &pool).unwrap();

        // Zero-copy: output plane buffer points to the exact same allocation!
        assert_eq!(
            out.planes().unwrap()[0].buffer.data().as_ptr(),
            frame.planes().unwrap()[0].buffer.data().as_ptr()
        );
        assert_eq!(out.pts, Timestamp::new(90090));
        assert_eq!(out.pkt_dts, Timestamp::new(87087));
        assert_eq!(out.duration, 3003);
        assert_eq!(out.time_base, Rational::new(1, 90000));
    }

    #[test]
    fn yuv420p_rgba8_bt709_and_bt601_roundtrip() {
        let pool = BufferPool::new();
        for color in [ColorDesc::BT709_LIMITED, ColorDesc::BT601_LIMITED] {
            let mut rgba_plane = BorderedPlane::new(&pool, 32 * 4, 32, BorderPadding::NONE);
            // Distinct saturated + neutral blocks (aligned to 2x2 so 4:2:0 subsampling is exact)
            for y in 0..32usize {
                let row = rgba_plane.row_mut(y).unwrap();
                for x in 0..32usize {
                    let px = match ((x / 16), (y / 16)) {
                        (0, 0) => [200u8, 40, 60, 255],
                        (1, 0) => [40, 190, 80, 255],
                        (0, 1) => [50, 70, 210, 255],
                        _ => [128, 128, 128, 255],
                    };
                    row[x * 4..x * 4 + 4].copy_from_slice(&px);
                }
            }
            let mut src_frame =
                VideoFrame::from_planes(32, 32, PixelLayout::RGBA8, vec![rgba_plane]);
            src_frame.color = ColorDesc {
                primaries: color.primaries,
                transfer: color.transfer,
                matrix: MatrixCoefficients::Identity,
                range: ColorRange::Full,
                chroma_location: ChromaLocation::Unspecified,
            };
            src_frame.pts = Timestamp::new(42);
            src_frame.duration = 1;
            src_frame.time_base = Rational::new(1, 30);

            let c_rgba = VideoContract::from_frame(&src_frame);
            let c_yuv = VideoContract::new(32, 32, PixelLayout::YUV420P).with_color(color);

            let opts = SwsOptions {
                dither: DitherMode::None,
                ..SwsOptions::default()
            };
            let to_yuv_ops = SwsOpList::build(&c_rgba, &c_yuv, opts).unwrap();
            let mut to_yuv = CpuSwsBackend::new().compile(&to_yuv_ops).unwrap();
            let yuv_frame = to_yuv.execute(&src_frame, &pool).unwrap();
            assert_eq!(yuv_frame.layout, PixelLayout::YUV420P);
            assert_eq!(yuv_frame.color.matrix, color.matrix);
            assert_eq!(yuv_frame.pts, Timestamp::new(42));

            let back_rgba_ops = SwsOpList::build(&c_yuv, &c_rgba, opts).unwrap();
            let mut back_rgba = CpuSwsBackend::new().compile(&back_rgba_ops).unwrap();
            let roundtrip = back_rgba.execute(&yuv_frame, &pool).unwrap();

            // RGB output must NEVER leak YUV matrix or chroma_location
            assert_eq!(roundtrip.color.matrix, MatrixCoefficients::Identity);
            assert_eq!(roundtrip.color.range, ColorRange::Full);
            assert_eq!(roundtrip.color.chroma_location, ChromaLocation::Unspecified);
            assert_eq!(roundtrip.pts, Timestamp::new(42));

            // Check interior of each 16x16 solid block matches within <= 2 LSBs
            let out_p = &roundtrip.planes().unwrap()[0];
            let src_p = &src_frame.planes().unwrap()[0];
            for &(bx, by) in &[(8usize, 8usize), (24, 8), (8, 24), (24, 24)] {
                let expected = &src_p.row(by).unwrap()[bx * 4..bx * 4 + 4];
                let actual = &out_p.row(by).unwrap()[bx * 4..bx * 4 + 4];
                for c in 0..4 {
                    let diff = (expected[c] as i32 - actual[c] as i32).abs();
                    assert!(
                        diff <= 2,
                        "mismatch at ({bx},{by}) ch {c} for {:?}: exp={} act={}",
                        color.matrix,
                        expected[c],
                        actual[c]
                    );
                }
            }
        }
    }

    #[test]
    fn hdr_10bit_bt2020_pq_to_bt709_sdr_tonemap() {
        let pool = BufferPool::new();
        // Create two 10-bit PQ frames: mid-gray (SDR white ~203 nits -> PQ ~0.5807 -> 594)
        // and very bright HDR highlight (~4000 nits -> PQ ~0.902 -> 923)
        let pq_sdr_white = (color::pq_oetf_nits(203.0) * 1023.0).round() as u16;
        let pq_hdr_peak = (color::pq_oetf_nits(4000.0) * 1023.0).round() as u16;

        let mut mid_frame =
            make_solid_yuv420p10le(&pool, 16, 16, pq_sdr_white, 512, 512, ColorDesc::BT2020_PQ);
        mid_frame.pts = Timestamp::new(1000);
        mid_frame
            .side_data
            .push(SideData::MasteringDisplay(MasteringDisplayMeta {
                display_primaries: [[Rational::new(0, 1); 2]; 3],
                white_point: [Rational::new(0, 1); 2],
                min_luminance: Rational::new(1, 10000),
                max_luminance: Rational::new(4000, 1),
            }));

        let peak_frame =
            make_solid_yuv420p10le(&pool, 16, 16, pq_hdr_peak, 512, 512, ColorDesc::BT2020_PQ);

        let src_c = VideoContract::from_frame(&mid_frame);
        let dst_c = VideoContract::new(16, 16, PixelLayout::RGBA8).with_color(ColorDesc::SRGB);

        for tm in [ToneMapOp::Bt2390, ToneMapOp::Hable, ToneMapOp::Clip] {
            let opts = SwsOptions {
                tone_map: tm,
                dither: DitherMode::None,
                ..SwsOptions::default()
            };
            let ops = SwsOpList::build(&src_c, &dst_c, opts).unwrap();
            assert!(
                ops.ops
                    .iter()
                    .any(|op| matches!(op, SwsOp::ToneMap(t) if *t == tm)),
                "expected ToneMap({tm:?}) in ops: {:?}",
                ops.ops
            );

            let mut compiled = CpuSwsBackend::new().compile(&ops).unwrap();
            let out_mid = compiled.execute(&mid_frame, &pool).unwrap();
            let out_peak = compiled.execute(&peak_frame, &pool).unwrap();

            // HDR metadata must be stripped when tone-mapped to SDR, while VFR PTS is kept
            assert!(out_mid.side_data.is_empty());
            assert_eq!(out_mid.pts, Timestamp::new(1000));
            assert_eq!(out_mid.color.transfer, TransferCharacteristic::Srgb);
            assert_eq!(out_mid.color.matrix, MatrixCoefficients::Identity);

            let mid_px = &out_mid.planes().unwrap()[0].row(8).unwrap()[32..36];
            let peak_px = &out_peak.planes().unwrap()[0].row(8).unwrap()[32..36];

            // Neutral gray (Cb=512, Cr=512): R ≈ G ≈ B
            assert!((mid_px[0] as i32 - mid_px[1] as i32).abs() <= 2);
            assert!((mid_px[1] as i32 - mid_px[2] as i32).abs() <= 2);
            assert_eq!(mid_px[3], 255);

            if tm == ToneMapOp::Bt2390 {
                // BT.2390 preserves brightness ordering and stays within SDR [0, 255]
                assert!(
                    mid_px[0] > 140 && mid_px[0] < 245,
                    "BT.2390 mid SDR white out of range: {}",
                    mid_px[0]
                );
                assert!(
                    peak_px[0] >= mid_px[0],
                    "BT.2390 peak {} should be >= mid {}",
                    peak_px[0],
                    mid_px[0]
                );
            } else {
                assert!(peak_px[0] >= mid_px[0]);
            }
        }
    }

    #[test]
    fn lanczos_resize_odd_dimensions_and_chroma_siting() {
        let pool = BufferPool::new();
        // 101x75 YUV420P (odd width & height -> chroma 51x38 via div_ceil(2))
        let (src_w, src_h) = (101u32, 75u32);
        let (dst_w, dst_h) = (151u32, 113u32);

        let mut py = BorderedPlane::new(&pool, src_w as usize, src_h as usize, BorderPadding::NONE);
        for y in 0..src_h as usize {
            let row = py.row_mut(y).unwrap();
            for (x, b) in row.iter_mut().enumerate() {
                *b = ((x * 2 + y) & 0xFF) as u8;
            }
        }
        let (cw, ch) = PixelLayout::YUV420P.subsampling.chroma_dims(src_w, src_h);
        assert_eq!((cw, ch), (51, 38));
        let mut pu = BorderedPlane::new(&pool, cw as usize, ch as usize, BorderPadding::NONE);
        let mut pv = BorderedPlane::new(&pool, cw as usize, ch as usize, BorderPadding::NONE);
        for y in 0..ch as usize {
            pu.row_mut(y).unwrap().fill(110);
            pv.row_mut(y).unwrap().fill(145);
        }
        let mut src_frame =
            VideoFrame::from_planes(src_w, src_h, PixelLayout::YUV420P, vec![py, pu, pv]);
        src_frame.color = ColorDesc {
            chroma_location: ChromaLocation::TopLeft,
            ..ColorDesc::BT709_LIMITED
        };

        let src_c = VideoContract::from_frame(&src_frame);
        let dst_c =
            VideoContract::new(dst_w, dst_h, PixelLayout::RGBA8).with_color(ColorDesc::SRGB);
        let opts = SwsOptions {
            scaler: SwsScaler::Lanczos,
            ..SwsOptions::default()
        };
        let ops = SwsOpList::build(&src_c, &dst_c, opts).unwrap();
        let mut compiled = CpuSwsBackend::new().compile(&ops).unwrap();
        let out = compiled.execute(&src_frame, &pool).unwrap();
        assert_eq!((out.width, out.height), (dst_w, dst_h));
        assert_eq!(out.layout, PixelLayout::RGBA8);
        assert_eq!(out.planes().unwrap()[0].height, dst_h as usize);
        assert_eq!(out.planes().unwrap()[0].width_bytes, dst_w as usize * 4);

        // Also test odd-dim Lanczos downscale back to YUV420P 53x39
        let dst_down =
            VideoContract::new(53, 39, PixelLayout::YUV420P).with_color(ColorDesc::BT709_LIMITED);
        let ops_down = SwsOpList::build(&dst_c, &dst_down, opts).unwrap();
        let mut comp_down = CpuSwsBackend::new().compile(&ops_down).unwrap();
        let down = comp_down.execute(&out, &pool).unwrap();
        assert_eq!((down.width, down.height), (53, 39));
        let (dcw, dch) = PixelLayout::YUV420P.subsampling.chroma_dims(53, 39);
        assert_eq!((dcw, dch), (27, 20));
        assert_eq!(down.planes().unwrap()[1].height, 20);
        assert_eq!(down.planes().unwrap()[1].width_bytes, 27);
    }

    #[test]
    fn rotation_90_and_crop_fused_into_read() {
        let pool = BufferPool::new();
        // 8x6 RGBA8 image, crop 2px left, 1px top, 1px right, 1px bottom => 5x4 active region:
        // Active (xc in 0..5, yc in 0..4) encoded with unique color (xc*40, yc*50, 200, 255).
        let mut plane = BorderedPlane::new(&pool, 8 * 4, 6, BorderPadding::NONE);
        for sy in 0..6usize {
            let row = plane.row_mut(sy).unwrap();
            for sx in 0..8usize {
                let px = if (2..7).contains(&sx) && (1..5).contains(&sy) {
                    let xc = (sx - 2) as u8;
                    let yc = (sy - 1) as u8;
                    [xc * 40, yc * 50, 200, 255]
                } else {
                    [0, 0, 0, 255] // cropped border
                };
                row[sx * 4..sx * 4 + 4].copy_from_slice(&px);
            }
        }
        let crop = CropRect {
            left: 2,
            right: 1,
            top: 1,
            bottom: 1,
        };
        let rot90 = display_matrix(90, false, false);

        let mut src_frame = VideoFrame::from_planes(8, 6, PixelLayout::RGBA8, vec![plane]);
        src_frame.color = ColorDesc::SRGB;
        src_frame.crop = crop;
        src_frame.side_data.push(SideData::DisplayMatrix(rot90));

        let src_c = VideoContract::from_frame(&src_frame);
        // 5x4 cropped rotated 90 deg CW => 4x5 output!
        assert_eq!(src_c.effective_dimensions().unwrap(), (4, 5));
        let dst_c = VideoContract::new(4, 5, PixelLayout::RGBA8).with_color(ColorDesc::SRGB);

        let ops = SwsOpList::build(
            &src_c,
            &dst_c,
            SwsOptions {
                dither: DitherMode::None,
                ..SwsOptions::default()
            },
        )
        .unwrap();
        // Verify crop & display_matrix were fused into SwsOp::Read and DisplayMatrix side data is stripped
        assert!(matches!(
            ops.ops[0],
            SwsOp::Read {
                crop: c,
                display_matrix: m,
                ..
            } if c == crop && m == rot90
        ));

        let mut compiled = CpuSwsBackend::new().compile(&ops).unwrap();
        let out = compiled.execute(&src_frame, &pool).unwrap();
        assert_eq!((out.width, out.height), (4, 5));
        assert_eq!(out.crop, CropRect::ZERO);
        assert!(out.side_data.is_empty());

        let out_p = &out.planes().unwrap()[0];
        // Under 90 deg CW rotation:
        // top-left of cropped src (xc=0, yc=0) -> top-right of output (x_out=3, y_out=0)
        let top_right = &out_p.row(0).unwrap()[3 * 4..3 * 4 + 4];
        assert_eq!(top_right, &[0, 0, 200, 255]);
        // bottom-left of cropped src (xc=0, yc=3) -> top-left of output (x_out=0, y_out=0)
        let top_left = &out_p.row(0).unwrap()[0..4];
        assert_eq!(top_left, &[0, 150, 200, 255]);
        // top-right of cropped src (xc=4, yc=0) -> bottom-right of output (x_out=3, y_out=4)
        let bot_right = &out_p.row(4).unwrap()[3 * 4..3 * 4 + 4];
        assert_eq!(bot_right, &[160, 0, 200, 255]);
    }

    #[test]
    fn optimizer_fuses_matrices_and_folds_swizzles() {
        // 1. Matrix fusion: BT.601 matrix (on BT.709 primaries) -> BT.709 matrix
        //    Linearize(Bt709) + Delinearize(Bt709) cancel, allowing YUV->RGB and RGB->YUV
        //    to fuse into ONE 3x4 ColorMatrix!
        let src_c = VideoContract::new(16, 16, PixelLayout::YUV420P).with_color(ColorDesc {
            matrix: MatrixCoefficients::Smpte170m,
            ..ColorDesc::BT709_LIMITED
        });
        let dst_c =
            VideoContract::new(16, 16, PixelLayout::YUV420P).with_color(ColorDesc::BT709_LIMITED);

        let ops = SwsOpList::build(
            &src_c,
            &dst_c,
            SwsOptions {
                dither: DitherMode::None,
                ..SwsOptions::default()
            },
        )
        .unwrap();

        let mat_count = ops
            .ops
            .iter()
            .filter(|op| matches!(op, SwsOp::ColorMatrix(_)))
            .count();
        assert_eq!(
            mat_count, 1,
            "expected 1 fused ColorMatrix, got {:?}",
            ops.ops
        );
        assert!(
            !ops.ops
                .iter()
                .any(|op| matches!(op, SwsOp::Linearize(_) | SwsOp::Delinearize(_))),
            "expected Linearize/Delinearize eliminated: {:?}",
            ops.ops
        );

        // 2. When primaries actually differ (Smpte170m -> Bt709), Linearize/Delinearize stay
        let src_diff_prim =
            VideoContract::new(16, 16, PixelLayout::YUV420P).with_color(ColorDesc::BT601_LIMITED);
        let ops_prim = SwsOpList::build(&src_diff_prim, &dst_c, SwsOptions::default()).unwrap();
        assert!(
            ops_prim
                .ops
                .iter()
                .any(|op| matches!(op, SwsOp::Linearize(_))),
            "expected Linearize preserved across gamut change"
        );

        // 3. Test explicit Swizzle folding into ColorMatrix + identity matrix elimination
        let mut custom = SwsOpList::new(
            VideoContract::new(8, 8, PixelLayout::BGRA8),
            VideoContract::new(8, 8, PixelLayout::RGBA8),
            SwsOptions::default(),
        );
        custom.ops = vec![
            SwsOp::Read {
                layout: PixelLayout::BGRA8,
                crop: CropRect::ZERO,
                display_matrix: IDENTITY_DISPLAY_MATRIX,
            },
            SwsOp::Swizzle([2, 1, 0, 3]),
            SwsOp::ColorMatrix([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]),
            SwsOp::Swizzle([2, 1, 0, 3]),
            SwsOp::Write(PixelLayout::RGBA8),
        ];
        custom.optimize();
        assert_eq!(
            custom.ops.len(),
            2,
            "double swizzle + identity matrix should collapse: {:?}",
            custom.ops
        );
    }

    #[test]
    fn blend_alpha_and_overlay_compositing() {
        let pool = BufferPool::new();

        // 1. Test SwsOp::BlendAlpha in pipeline: semi-transparent red RGBA8 -> RGB24
        let mut src_plane = BorderedPlane::new(&pool, 8 * 4, 8, BorderPadding::NONE);
        for y in 0..8usize {
            let row = src_plane.row_mut(y).unwrap();
            for x in 0..8usize {
                // Red = 255, Green = 0, Blue = 0, Alpha = 128 (~0.5)
                row[x * 4..x * 4 + 4].copy_from_slice(&[255, 0, 0, 128]);
            }
        }
        let mut src_rgba = VideoFrame::from_planes(8, 8, PixelLayout::RGBA8, vec![src_plane]);
        src_rgba.color = ColorDesc::SRGB;

        let src_c = VideoContract::from_frame(&src_rgba);
        let dst_c = VideoContract::new(8, 8, PixelLayout::RGB24).with_color(ColorDesc::SRGB);
        let ops = SwsOpList::build(
            &src_c,
            &dst_c,
            SwsOptions {
                dither: DitherMode::None,
                ..SwsOptions::default()
            },
        )
        .unwrap();
        assert!(ops.ops.iter().any(|op| matches!(op, SwsOp::BlendAlpha)));

        let mut compiled = CpuSwsBackend::new().compile(&ops).unwrap();
        let blended_rgb24 = compiled.execute(&src_rgba, &pool).unwrap();
        assert_eq!(blended_rgb24.layout, PixelLayout::RGB24);
        let px = &blended_rgb24.planes().unwrap()[0].row(0).unwrap()[0..3];
        // Linear blend of sRGB(1.0) * 0.502 -> linear 0.502 -> sRGB ~0.735 -> ~187
        assert!(
            px[0] > 160 && px[0] < 200 && px[1] == 0 && px[2] == 0,
            "expected gamma-correct blended red, got {px:?}"
        );

        // 2. Test CompiledSws::blend_overlay on YUV420P frame
        let mut yuv_base =
            make_solid_yuv420p(&pool, 16, 16, 16, 128, 128, ColorDesc::BT709_LIMITED);
        let mut ov_plane = BorderedPlane::new(&pool, 4 * 4, 4, BorderPadding::NONE);
        for y in 0..4usize {
            let row = ov_plane.row_mut(y).unwrap();
            for x in 0..4usize {
                // Opaque white overlay
                row[x * 4..x * 4 + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let overlay = PositionedPlane {
            x: 4,
            y: 4,
            width: 4,
            height: 4,
            layout: PixelLayout::RGBA8,
            plane: ov_plane,
        };
        compiled
            .blend_overlay(&mut yuv_base, &[overlay], &pool)
            .unwrap();

        let yp = &yuv_base.planes().unwrap()[0];
        // Inside overlay (4, 4): white -> Y = 235
        assert!((yp.row(4).unwrap()[4] as i32 - 235).abs() <= 1);
        // Outside overlay (0, 0): black -> Y = 16
        assert_eq!(yp.row(0).unwrap()[0], 16);
    }

    #[test]
    fn supports_all_required_pixel_formats_and_dither_modes() {
        let pool = BufferPool::new();
        let layouts = [
            PixelLayout::YUV420P,
            PixelLayout::YUV420P10LE,
            PixelLayout::YUV422P,
            PixelLayout::YUV444P,
            PixelLayout::NV12,
            PixelLayout::P010,
            PixelLayout::GBRP,
            PixelLayout::RGB24,
            PixelLayout::RGBA8,
            PixelLayout::BGRA8,
            PixelLayout::GRAY8,
        ];

        // Start from 16x16 RGBA8
        let mut p = BorderedPlane::new(&pool, 16 * 4, 16, BorderPadding::NONE);
        for y in 0..16usize {
            let row = p.row_mut(y).unwrap();
            for x in 0..16usize {
                row[x * 4..x * 4 + 4].copy_from_slice(&[(x * 16) as u8, (y * 16) as u8, 128, 255]);
            }
        }
        let mut src = VideoFrame::from_planes(16, 16, PixelLayout::RGBA8, vec![p]);
        src.color = ColorDesc::SRGB;
        let src_c = VideoContract::from_frame(&src);

        for &layout in &layouts {
            let color = match layout.model {
                wedeo_core::ColorModel::Ycbcr => ColorDesc::BT709_LIMITED,
                wedeo_core::ColorModel::Gray => ColorDesc {
                    range: ColorRange::Full,
                    ..ColorDesc::UNSPECIFIED
                },
                _ => ColorDesc::SRGB,
            };
            let dst_c = VideoContract::new(16, 16, layout).with_color(color);
            for dither in [DitherMode::Ordered, DitherMode::ErrorDiffusion] {
                let ops = SwsOpList::build(
                    &src_c,
                    &dst_c,
                    SwsOptions {
                        dither,
                        ..SwsOptions::default()
                    },
                )
                .unwrap();
                let mut compiled = CpuSwsBackend::new().compile(&ops).unwrap();
                let converted = compiled.execute(&src, &pool).unwrap();
                assert_eq!(converted.layout, layout);
                assert_eq!(converted.domain(), MemoryDomain::Cpu);

                // Roundtrip back to RGBA8
                let back_ops = SwsOpList::build(&dst_c, &src_c, SwsOptions::default()).unwrap();
                let mut back_c = CpuSwsBackend::new().compile(&back_ops).unwrap();
                let back = back_c.execute(&converted, &pool).unwrap();
                assert_eq!(back.layout, PixelLayout::RGBA8);
                assert_eq!(back.color.matrix, MatrixCoefficients::Identity);
            }
        }
    }

    #[test]
    fn multi_tile_scaling_and_hlg_roundtrip() {
        let pool = BufferPool::new();
        // Cross 256x16 tile boundary: 320x24 -> 280x20 Lanczos resize
        let (w, h) = (320u32, 24u32);
        let mut plane = BorderedPlane::new(&pool, w as usize * 4, h as usize, BorderPadding::NONE);
        for y in 0..h as usize {
            let row = plane.row_mut(y).unwrap();
            for x in 0..w as usize {
                row[x * 4..x * 4 + 4].copy_from_slice(&[
                    (x % 256) as u8,
                    ((y * 10) % 256) as u8,
                    128,
                    255,
                ]);
            }
        }
        let mut src = VideoFrame::from_planes(w, h, PixelLayout::RGBA8, vec![plane]);
        src.color = ColorDesc::SRGB;

        let src_c = VideoContract::from_frame(&src);
        let dst_c = VideoContract::new(280, 20, PixelLayout::RGBA8).with_color(ColorDesc::SRGB);
        let ops = SwsOpList::build(
            &src_c,
            &dst_c,
            SwsOptions {
                scaler: SwsScaler::Lanczos,
                dither: DitherMode::ErrorDiffusion,
                ..SwsOptions::default()
            },
        )
        .unwrap();
        let mut compiled = CpuSwsBackend::new().compile(&ops).unwrap();
        let out = compiled.execute(&src, &pool).unwrap();
        assert_eq!((out.width, out.height), (280, 20));

        // HLG transfer curve roundtrip check
        let rgb = [0.5f32, 0.75, 0.25];
        let lin = color::linearize_rgb(TransferCharacteristic::Hlg, rgb);
        let back = color::delinearize_rgb(TransferCharacteristic::Hlg, lin);
        for i in 0..3 {
            assert!((rgb[i] - back[i]).abs() < 1e-5);
        }
    }
}
