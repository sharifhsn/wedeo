//! Optional `wgpu` WGSL compute backend for [`wedeo_video_ops::SwsOpList`], custom
//! `WgslFilter` compute passes, and zero-copy `wgpu::Texture` (`GpuSurface`) interop.

#![forbid(unsafe_code)]

pub mod backend;
pub mod context;
pub mod wgsl_gen;

pub use backend::{WgpuCompiledSws, WgpuSwsBackend};
pub use context::{
    WgpuContext, WgpuTextureSurface, plane_texel_geometry, plane_texture_format,
    supports_storage_binding,
};
pub use wgsl_gen::{
    CachedPipeline, GeneratedWgsl, MAX_COLOR_MATRICES, SWS_UNIFORMS_SIZE, SwsUniforms,
    WORKGROUP_SIZE, WgslComputePass, WgslOutputBinding, generate_wgsl, structural_op_list_hash,
    validate_wgsl,
};

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use wedeo_core::{
        BorderPadding, BorderedPlane, Buffer, BufferPool, ChromaLocation, ColorDesc,
        ColorPrimaries, ColorRange, CropRect, GpuDeviceId, GpuSurface, MatrixCoefficients,
        MemoryDomain, PixelLayout, Rational, SideData, Timestamp, TransferCharacteristic,
        VideoFrame,
    };
    use wedeo_video_ops::{
        CpuSwsBackend, DitherMode, SwsBackend, SwsOp, SwsOpList, SwsOptions, SwsScaler, ToneMapOp,
        VideoContract, color, display_matrix,
    };

    use super::*;

    fn make_solid_yuv420p(
        pool: &Arc<BufferPool>,
        w: u32,
        h: u32,
        y: u8,
        u: u8,
        v: u8,
        color_desc: ColorDesc,
    ) -> VideoFrame {
        let mut py = BorderedPlane::new(pool, w as usize, h as usize, BorderPadding::NONE);
        for r in 0..h as usize {
            py.row_mut(r).unwrap().fill(y);
        }
        let (cw, ch) = PixelLayout::YUV420P.subsampling.chroma_dims(w, h);
        let mut pu = BorderedPlane::new(pool, cw as usize, ch as usize, BorderPadding::NONE);
        let mut pv = BorderedPlane::new(pool, cw as usize, ch as usize, BorderPadding::NONE);
        for r in 0..ch as usize {
            pu.row_mut(r).unwrap().fill(u);
            pv.row_mut(r).unwrap().fill(v);
        }
        let mut frame = VideoFrame::from_planes(w, h, PixelLayout::YUV420P, vec![py, pu, pv]);
        frame.color = color_desc;
        frame
    }

    #[test]
    fn wgsl_codegen_structural_invariance_across_color_matrices_and_crops() {
        // Two YUV420P -> RGBA8 conversions differing in ColorMatrix (BT.709 vs BT.601),
        // crop rectangle, display_matrix rotation (0 vs 90 deg), and dimensions:
        // because ColorMatrix, crop, display_matrix, and dimensions live in SwsUniforms,
        // the generated WGSL source and structural cache key MUST be 100% identical!
        let src_bt709 = VideoContract::new(1920, 1080, PixelLayout::YUV420P)
            .with_color(ColorDesc::BT709_LIMITED);
        let dst_bt709 = VideoContract::new(1920, 1080, PixelLayout::RGBA8).with_color(ColorDesc {
            primaries: ColorPrimaries::Bt709,
            transfer: TransferCharacteristic::Bt709,
            matrix: MatrixCoefficients::Identity,
            range: ColorRange::Full,
            chroma_location: ChromaLocation::Unspecified,
        });

        let src_bt601 = VideoContract::new(1280, 720, PixelLayout::YUV420P)
            .with_color(ColorDesc {
                // Same transfer (BT.709 curve) and primaries, but SMPTE 170M (BT.601) YUV matrix
                // and full range -> completely different 3x4 ColorMatrix float coefficients!
                primaries: ColorPrimaries::Bt709,
                transfer: TransferCharacteristic::Bt709,
                matrix: MatrixCoefficients::Smpte170m,
                range: ColorRange::Full,
                chroma_location: ChromaLocation::Center,
            })
            .with_crop(CropRect {
                top: 8,
                bottom: 8,
                left: 16,
                right: 16,
            })
            .with_display_matrix(display_matrix(90, true, false));
        // After crop (1248x704) and 90 deg rotation -> (704x1248) so no ScaleH/ScaleV is added
        let dst_bt601 = VideoContract::new(704, 1248, PixelLayout::RGBA8).with_color(ColorDesc {
            primaries: ColorPrimaries::Bt709,
            transfer: TransferCharacteristic::Bt709,
            matrix: MatrixCoefficients::Identity,
            range: ColorRange::Full,
            chroma_location: ChromaLocation::Unspecified,
        });

        let opts = SwsOptions {
            dither: DitherMode::Ordered,
            ..SwsOptions::default()
        };
        let ops_709 = SwsOpList::build(&src_bt709, &dst_bt709, opts).unwrap();
        let ops_601 = SwsOpList::build(&src_bt601, &dst_bt601, opts).unwrap();

        // Confirm the two SwsOpLists genuinely have different ColorMatrix float coefficients
        let mat_709 = ops_709
            .ops
            .iter()
            .find_map(|op| match op {
                SwsOp::ColorMatrix(m) => Some(*m),
                _ => None,
            })
            .expect("ops_709 has ColorMatrix");
        let mat_601 = ops_601
            .ops
            .iter()
            .find_map(|op| match op {
                SwsOp::ColorMatrix(m) => Some(*m),
                _ => None,
            })
            .expect("ops_601 has ColorMatrix");
        assert_ne!(
            mat_709, mat_601,
            "BT.709 limited and BT.601 full matrices must differ"
        );

        // Yet structural hash and generated WGSL must be identical!
        let hash_709 = structural_op_list_hash(&ops_709);
        let hash_601 = structural_op_list_hash(&ops_601);
        assert_eq!(
            hash_709, hash_601,
            "structural cache key must be invariant to ColorMatrix values, crop, and display_matrix"
        );

        let wgsl_709 = generate_wgsl(&ops_709).unwrap();
        let wgsl_601 = generate_wgsl(&ops_601).unwrap();
        assert_eq!(wgsl_709.structural_hash, hash_709);
        assert_eq!(wgsl_709.source, wgsl_601.source);
        assert!(
            wgsl_709
                .source
                .contains("@compute @workgroup_size(16, 16, 1)")
        );
        assert!(
            wgsl_709
                .source
                .contains("texture_storage_2d<rgba8unorm, write>")
        );

        // Verify the matrices ARE differentiated in SwsUniforms bytes
        let u_709 = SwsUniforms::from_op_list(&ops_709).unwrap();
        let u_601 = SwsUniforms::from_op_list(&ops_601).unwrap();
        assert_ne!(u_709.to_bytes(), u_601.to_bytes());
        assert_eq!(u_709.matrices[0], mat_709);
        assert_eq!(u_601.matrices[0], mat_601);
    }

    #[test]
    fn wgsl_codegen_validates_all_transfers_tonemaps_scalers_and_outputs() {
        // 1. HDR 10-bit PQ + HLG -> SDR RGBA8 & BGRA8 across BT.2390, Hable, and Clip tone-mappers
        for src_color in [ColorDesc::BT2020_PQ, ColorDesc::BT2020_HLG] {
            let src =
                VideoContract::new(3840, 2160, PixelLayout::YUV420P10LE).with_color(src_color);
            for dst_layout in [PixelLayout::RGBA8, PixelLayout::BGRA8, PixelLayout::RGBA32F] {
                let dst = VideoContract::new(1920, 1080, dst_layout).with_color(ColorDesc::SRGB);
                for tone_map in [ToneMapOp::Bt2390, ToneMapOp::Hable, ToneMapOp::Clip] {
                    for scaler in [
                        SwsScaler::Point,
                        SwsScaler::Bilinear,
                        SwsScaler::Bicubic,
                        SwsScaler::Lanczos,
                    ] {
                        let ops = SwsOpList::build(
                            &src,
                            &dst,
                            SwsOptions {
                                scaler,
                                tone_map,
                                dither: DitherMode::Ordered,
                                ..SwsOptions::default()
                            },
                        )
                        .unwrap();
                        let generated = generate_wgsl(&ops).unwrap();
                        validate_wgsl(&generated.source).unwrap();
                        if dst_layout == PixelLayout::BGRA8 {
                            assert!(
                                generated
                                    .source
                                    .contains("texture_storage_2d<bgra8unorm, write>")
                            );
                        } else if dst_layout == PixelLayout::RGBA8 {
                            assert!(
                                generated
                                    .source
                                    .contains("texture_storage_2d<rgba8unorm, write>")
                            );
                        }
                    }
                }
            }
        }

        // 2. Planar & semi-planar outputs: YUV420P, YUV420P10LE, YUV422P, YUV444P, NV12, P010, GBRP, GRAY8, YUYV422
        let src_rgba = VideoContract::new(128, 96, PixelLayout::RGBA8).with_color(ColorDesc::SRGB);
        let dst_layouts = [
            PixelLayout::YUV420P,
            PixelLayout::YUV420P10LE,
            PixelLayout::YUV422P,
            PixelLayout::YUV444P,
            PixelLayout::NV12,
            PixelLayout::P010,
            PixelLayout::GBRP,
            PixelLayout::GRAY8,
            PixelLayout::GRAY16LE,
            PixelLayout::YUYV422,
            PixelLayout::UYVY422,
        ];
        for layout in dst_layouts {
            let dst = VideoContract::new(64, 48, layout).with_color(ColorDesc::BT709_LIMITED);
            let ops = SwsOpList::build(&src_rgba, &dst, SwsOptions::default()).unwrap();
            let generated = generate_wgsl(&ops).unwrap();
            validate_wgsl(&generated.source).unwrap();
            assert_eq!(
                generated.output_binding,
                WgslOutputBinding::PlanarStorageBuffers {
                    plane_count: layout.plane_count()
                }
            );

            // And reverse direction: layout -> RGBA8
            let back_ops = SwsOpList::build(&dst, &src_rgba, SwsOptions::default()).unwrap();
            let back_gen = generate_wgsl(&back_ops).unwrap();
            validate_wgsl(&back_gen.source).unwrap();
        }
    }

    #[test]
    fn wgpu_texture_surface_downcast_and_memory_domain_roundtrip() {
        let dev_id = GpuDeviceId(4242);
        let surf: Arc<dyn GpuSurface> = Arc::new(WgpuTextureSurface::new_virtual(
            1920,
            1080,
            PixelLayout::RGBA8,
            dev_id,
        ));

        assert_eq!(surf.device_id(), dev_id);
        assert_eq!(surf.width(), 1920);
        assert_eq!(surf.height(), 1080);
        assert_eq!(surf.layout(), PixelLayout::RGBA8);

        // Downcast via GpuSurface::as_any() and WgpuContext::unwrap_surface
        let downcasted = WgpuContext::unwrap_surface(surf.as_ref()).expect("downcast succeeds");
        assert_eq!(downcasted.device_id(), dev_id);
        assert_eq!(downcasted.width(), 1920);
        assert_eq!(downcasted.height(), 1080);
        assert_eq!(downcasted.layout(), PixelLayout::RGBA8);

        // Buffer::from_gpu roundtrip
        let gpu_buf = Buffer::from_gpu(Arc::clone(&surf));
        assert!(gpu_buf.is_gpu());
        assert_eq!(gpu_buf.domain(), MemoryDomain::Gpu(dev_id));
        let buf_surf = gpu_buf.gpu_surface().expect("has gpu_surface");
        assert_eq!(buf_surf.device_id(), dev_id);

        // VideoFrame::from_gpu roundtrip
        let gpu_frame = VideoFrame::from_gpu(Arc::clone(&surf));
        assert_eq!(gpu_frame.domain(), MemoryDomain::Gpu(dev_id));
        assert_eq!(gpu_frame.width, 1920);
        assert_eq!(gpu_frame.height, 1080);
        assert_eq!(gpu_frame.layout, PixelLayout::RGBA8);
        assert!(gpu_frame.planes().is_none());
        let frame_surf = gpu_frame
            .gpu_surface()
            .and_then(|s| WgpuContext::unwrap_surface(s.as_ref()))
            .expect("unwrap frame surface");
        assert_eq!(frame_surf.device_id(), dev_id);
    }

    #[test]
    fn custom_wgsl_filter_shader_validates_and_live_gpu_executes_when_available() {
        let custom_vignette_wgsl = r#"
struct VignetteParams {
    strength: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: VignetteParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(src_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let px = textureLoad(src_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    let out_px = vec4<f32>(px.rgb * (1.0 - params.strength * 0.5), px.a);
    textureStore(dst_tex, vec2<i32>(i32(gid.x), i32(gid.y)), out_px);
}
"#;
        validate_wgsl(custom_vignette_wgsl).expect("custom vignette WGSL is valid");

        // Attempt to acquire a headless wgpu adapter. If available on this machine,
        // run full live GPU upload/download, WgpuTextureSurface unwrap_texture,
        // WgpuSwsBackend (YUV420P <-> RGBA8, HDR PQ BT.2390 tonemap, 90° rotation + crop),
        // pipeline cache hit verification, and WgslComputePass execution!
        let Ok(ctx) = WgpuContext::new_headless() else {
            return;
        };

        let pool = BufferPool::new();

        // 1. Live wgpu::Texture wrap_texture & unwrap_texture zero-copy Tessera handoff
        let raw_tex = ctx.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("tessera-external-tex"),
            size: wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let wrapped = ctx.wrap_texture(raw_tex.clone(), PixelLayout::RGBA8);
        assert_eq!(wrapped.device_id(), ctx.device_id());
        let unwrapped =
            WgpuContext::unwrap_texture(wrapped.as_ref()).expect("unwrap_texture succeeds");
        assert_eq!(unwrapped.width(), 32);
        assert_eq!(unwrapped.height(), 32);
        assert_eq!(unwrapped.format(), wgpu::TextureFormat::Rgba8Unorm);

        // 2. CPU -> GPU upload_frame and GPU -> CPU download_frame roundtrip
        let mut rgba_plane = BorderedPlane::new(&pool, 16 * 4, 16, BorderPadding::NONE);
        for y in 0..16usize {
            let row = rgba_plane.row_mut(y).unwrap();
            for x in 0..16usize {
                row[x * 4..x * 4 + 4].copy_from_slice(&[(x * 15) as u8, (y * 15) as u8, 180, 255]);
            }
        }
        let mut cpu_rgba = VideoFrame::from_planes(16, 16, PixelLayout::RGBA8, vec![rgba_plane]);
        cpu_rgba.color = ColorDesc::SRGB;
        cpu_rgba.pts = Timestamp::new(3003);
        cpu_rgba.duration = 1001;
        cpu_rgba.time_base = Rational::new(1, 30000);

        let gpu_rgba = ctx.upload_frame(&cpu_rgba).unwrap();
        assert_eq!(gpu_rgba.domain(), MemoryDomain::Gpu(ctx.device_id()));
        assert_eq!(gpu_rgba.pts, Timestamp::new(3003));
        assert!(WgpuContext::unwrap_texture(gpu_rgba.gpu_surface().unwrap().as_ref()).is_some());

        let downloaded = ctx.download_frame(&gpu_rgba, &pool).unwrap();
        assert_eq!(downloaded.domain(), MemoryDomain::Cpu);
        assert_eq!(downloaded.pts, Timestamp::new(3003));
        for y in 0..16usize {
            assert_eq!(
                downloaded.planes().unwrap()[0].row(y).unwrap(),
                cpu_rgba.planes().unwrap()[0].row(y).unwrap()
            );
        }

        // 3. Live WgpuSwsBackend vs CpuSwsBackend parity + pipeline cache reuse across matrices
        let backend = ctx.sws_backend();
        let cpu_backend = CpuSwsBackend::new();
        let bt601_matrix_same_primaries = ColorDesc {
            matrix: MatrixCoefficients::Smpte170m,
            range: ColorRange::Full,
            ..ColorDesc::BT709_LIMITED
        };
        for color_desc in [ColorDesc::BT709_LIMITED, bt601_matrix_same_primaries] {
            let mut src_yuv = make_solid_yuv420p(&pool, 16, 16, 140, 100, 160, color_desc);
            src_yuv.pts = Timestamp::new(777);

            let src_c = VideoContract::from_frame(&src_yuv);
            let dst_c = VideoContract::new(16, 16, PixelLayout::RGBA8).with_color(ColorDesc {
                primaries: ColorPrimaries::Bt709,
                transfer: TransferCharacteristic::Bt709,
                matrix: MatrixCoefficients::Identity,
                range: ColorRange::Full,
                chroma_location: ChromaLocation::Unspecified,
            });
            let opts = SwsOptions {
                dither: DitherMode::None,
                ..SwsOptions::default()
            };
            let ops = SwsOpList::build(&src_c, &dst_c, opts).unwrap();

            let mut gpu_comp = backend.compile(&ops).unwrap();
            let mut cpu_comp = cpu_backend.compile(&ops).unwrap();

            let gpu_out = gpu_comp.execute(&src_yuv, &pool).unwrap();
            let cpu_out = cpu_comp.execute(&src_yuv, &pool).unwrap();

            assert_eq!(gpu_out.pts, Timestamp::new(777));
            let g_px = &gpu_out.planes().unwrap()[0].row(8).unwrap()[32..36];
            let c_px = &cpu_out.planes().unwrap()[0].row(8).unwrap()[32..36];
            for ch in 0..4 {
                let diff = (g_px[ch] as i32 - c_px[ch] as i32).abs();
                assert!(
                    diff <= 1,
                    "GPU vs CPU mismatch for {:?} ch {ch}: gpu={} cpu={}",
                    color_desc.matrix,
                    g_px[ch],
                    c_px[ch]
                );
            }
        }
        // Both BT.709-Limited and BT.601-Full YUV420P->RGBA8 pipelines share the SAME structural cache entry!
        assert_eq!(ctx.cached_pipeline_count(), 1);

        // 4. Live RGBA8 -> YUV420P planar storage buffer output test
        let c_rgba = VideoContract::from_frame(&cpu_rgba);
        let c_yuv =
            VideoContract::new(16, 16, PixelLayout::YUV420P).with_color(ColorDesc::BT709_LIMITED);
        let to_yuv_ops = SwsOpList::build(
            &c_rgba,
            &c_yuv,
            SwsOptions {
                dither: DitherMode::None,
                ..SwsOptions::default()
            },
        )
        .unwrap();
        let mut gpu_to_yuv = backend.compile(&to_yuv_ops).unwrap();
        let mut cpu_to_yuv = cpu_backend.compile(&to_yuv_ops).unwrap();
        let g_yuv = gpu_to_yuv.execute(&cpu_rgba, &pool).unwrap();
        let c_yuv_frame = cpu_to_yuv.execute(&cpu_rgba, &pool).unwrap();
        for p in 0..3 {
            let gr = g_yuv.planes().unwrap()[p].row(2).unwrap();
            let cr = c_yuv_frame.planes().unwrap()[p].row(2).unwrap();
            for (gv, cv) in gr.iter().zip(cr.iter()) {
                assert!((*gv as i32 - *cv as i32).abs() <= 1);
            }
        }

        // 5. Live 90-degree rotation + crop test on GPU vs CPU
        let crop = CropRect {
            left: 2,
            right: 2,
            top: 2,
            bottom: 4,
        };
        let rot90 = display_matrix(90, false, false);
        let mut rot_src = cpu_rgba.clone();
        rot_src.crop = crop;
        rot_src.side_data.push(SideData::DisplayMatrix(rot90));
        let rot_src_c = VideoContract::from_frame(&rot_src);
        // 16x16 cropped by (2,2,2,4) -> 12x10, rotated 90 CW -> 10x12
        let rot_dst_c = VideoContract::new(10, 12, PixelLayout::RGBA8).with_color(ColorDesc::SRGB);
        let rot_ops = SwsOpList::build(
            &rot_src_c,
            &rot_dst_c,
            SwsOptions {
                dither: DitherMode::None,
                ..SwsOptions::default()
            },
        )
        .unwrap();
        let mut gpu_rot = backend.compile(&rot_ops).unwrap();
        let mut cpu_rot = cpu_backend.compile(&rot_ops).unwrap();
        let g_rot_out = gpu_rot.execute(&rot_src, &pool).unwrap();
        let c_rot_out = cpu_rot.execute(&rot_src, &pool).unwrap();
        assert_eq!((g_rot_out.width, g_rot_out.height), (10, 12));
        for y in 0..12usize {
            assert_eq!(
                g_rot_out.planes().unwrap()[0].row(y).unwrap(),
                c_rot_out.planes().unwrap()[0].row(y).unwrap()
            );
        }

        // 6. Live HDR 10-bit PQ -> SDR RGBA8 BT.2390 tone-mapping test on GPU vs CPU
        let pq_val = (color::pq_oetf_nits(1000.0) * 1023.0).round() as u16;
        let mut py10 = BorderedPlane::new(&pool, 16 * 2, 16, BorderPadding::NONE);
        let (cw10, ch10) = (8usize, 8usize);
        let mut pu10 = BorderedPlane::new(&pool, cw10 * 2, ch10, BorderPadding::NONE);
        let mut pv10 = BorderedPlane::new(&pool, cw10 * 2, ch10, BorderPadding::NONE);
        for r in 0..16 {
            for chunk in py10.row_mut(r).unwrap().as_chunks_mut::<2>().0 {
                chunk.copy_from_slice(&pq_val.to_le_bytes());
            }
        }
        for r in 0..ch10 {
            for chunk in pu10.row_mut(r).unwrap().as_chunks_mut::<2>().0 {
                chunk.copy_from_slice(&512u16.to_le_bytes());
            }
            for chunk in pv10.row_mut(r).unwrap().as_chunks_mut::<2>().0 {
                chunk.copy_from_slice(&512u16.to_le_bytes());
            }
        }
        let mut hdr_frame =
            VideoFrame::from_planes(16, 16, PixelLayout::YUV420P10LE, vec![py10, pu10, pv10]);
        hdr_frame.color = ColorDesc::BT2020_PQ;
        let hdr_c = VideoContract::from_frame(&hdr_frame);
        let sdr_c = VideoContract::new(16, 16, PixelLayout::RGBA8).with_color(ColorDesc::SRGB);
        let hdr_ops = SwsOpList::build(
            &hdr_c,
            &sdr_c,
            SwsOptions {
                tone_map: ToneMapOp::Bt2390,
                dither: DitherMode::None,
                ..SwsOptions::default()
            },
        )
        .unwrap();
        let mut gpu_hdr = backend.compile(&hdr_ops).unwrap();
        let mut cpu_hdr = cpu_backend.compile(&hdr_ops).unwrap();
        let g_sdr = gpu_hdr.execute(&hdr_frame, &pool).unwrap();
        let c_sdr = cpu_hdr.execute(&hdr_frame, &pool).unwrap();
        let g_hdr_px = &g_sdr.planes().unwrap()[0].row(8).unwrap()[32..36];
        let c_hdr_px = &c_sdr.planes().unwrap()[0].row(8).unwrap()[32..36];
        for ch in 0..4 {
            assert!(
                (g_hdr_px[ch] as i32 - c_hdr_px[ch] as i32).abs() <= 1,
                "HDR BT.2390 GPU vs CPU ch {ch}: gpu={} cpu={}",
                g_hdr_px[ch],
                c_hdr_px[ch]
            );
        }

        // 7. Live custom user WgslComputePass execution with uniform buffer binding
        let pass = WgslComputePass::compile(&ctx, custom_vignette_wgsl, "main", [16, 16, 1], true)
            .unwrap();
        let strength_bytes = 1.0f32.to_le_bytes(); // strength = 1.0 -> scales RGB by 0.5
        let filtered = pass
            .execute(&ctx, &cpu_rgba, &strength_bytes, &pool)
            .unwrap();
        let orig_px = &cpu_rgba.planes().unwrap()[0].row(10).unwrap()[10 * 4..10 * 4 + 4];
        let filt_px = &filtered.planes().unwrap()[0].row(10).unwrap()[10 * 4..10 * 4 + 4];
        for ch in 0..3 {
            let expected = (orig_px[ch] as f32 * 0.5).round() as i32;
            assert!((filt_px[ch] as i32 - expected).abs() <= 1);
        }
        assert_eq!(filt_px[3], 255);
    }

    #[test]
    fn display_matrix_all_orientations_and_flips_share_structural_wgsl() {
        let mut ref_hash = None;
        let mut ref_wgsl = None;

        for deg in [0, 90, 180, 270] {
            for hflip in [false, true] {
                for vflip in [false, true] {
                    let dm = display_matrix(deg, hflip, vflip);
                    let src = VideoContract::new(64, 48, PixelLayout::RGBA8)
                        .with_color(ColorDesc::SRGB)
                        .with_crop(CropRect {
                            left: 4,
                            right: 4,
                            top: 2,
                            bottom: 2,
                        })
                        .with_display_matrix(dm);
                    let (ow, oh) = src.effective_dimensions().unwrap();
                    let dst =
                        VideoContract::new(ow, oh, PixelLayout::BGRA8).with_color(ColorDesc::SRGB);

                    let ops = SwsOpList::build(&src, &dst, SwsOptions::default()).unwrap();
                    let generated = generate_wgsl(&ops).unwrap();
                    validate_wgsl(&generated.source).unwrap();

                    let uniforms = SwsUniforms::from_op_list(&ops).unwrap();
                    assert_eq!((uniforms.oriented_w, uniforms.oriented_h), (ow, oh));

                    if let Some(h) = ref_hash {
                        assert_eq!(generated.structural_hash, h);
                        assert_eq!(Some(generated.source.as_str()), ref_wgsl.as_deref());
                    } else {
                        ref_hash = Some(generated.structural_hash);
                        ref_wgsl = Some(generated.source);
                    }
                }
            }
        }
    }

    #[test]
    fn gpu_domain_zero_ping_pong_and_lanczos_scaling_when_adapter_available() {
        let Ok(ctx) = WgpuContext::new_headless() else {
            return;
        };
        let pool = BufferPool::new();
        let gpu_dom = MemoryDomain::Gpu(ctx.device_id());

        // Create a 24x16 CPU frame, upload once to GPU, run GPU->GPU Lanczos resize (24x16 -> 16x12)
        // keeping the output in MemoryDomain::Gpu(ctx.device_id()), then unwrap_texture and download.
        let mut plane = BorderedPlane::new(&pool, 24 * 4, 16, BorderPadding::NONE);
        for y in 0..16usize {
            let row = plane.row_mut(y).unwrap();
            for x in 0..24usize {
                row[x * 4..x * 4 + 4].copy_from_slice(&[(x * 10) as u8, (y * 14) as u8, 200, 255]);
            }
        }
        let mut cpu_src = VideoFrame::from_planes(24, 16, PixelLayout::RGBA8, vec![plane]);
        cpu_src.color = ColorDesc::SRGB;

        let gpu_src = ctx.upload_frame(&cpu_src).unwrap();
        assert_eq!(gpu_src.domain(), gpu_dom);

        let src_c = VideoContract::from_frame(&gpu_src);
        assert_eq!(src_c.domain, gpu_dom);
        let dst_c = VideoContract::new(16, 12, PixelLayout::RGBA8)
            .with_color(ColorDesc::SRGB)
            .with_domain(gpu_dom);

        let opts = SwsOptions {
            scaler: SwsScaler::Lanczos,
            dither: DitherMode::None,
            ..SwsOptions::default()
        };
        let ops = SwsOpList::build(&src_c, &dst_c, opts).unwrap();
        let mut compiled = ctx.sws_backend().compile(&ops).unwrap();
        let gpu_dst = compiled.execute(&gpu_src, &pool).unwrap();

        // Output stays on GPU without CPU roundtrip!
        assert_eq!(gpu_dst.domain(), gpu_dom);
        assert_eq!((gpu_dst.width, gpu_dst.height), (16, 12));
        let dst_tex = WgpuContext::unwrap_texture(gpu_dst.gpu_surface().unwrap().as_ref()).unwrap();
        assert_eq!((dst_tex.width(), dst_tex.height()), (16, 12));

        // Compare downloaded GPU Lanczos output against CPU reference Lanczos output
        let downloaded = ctx.download_frame(&gpu_dst, &pool).unwrap();
        let cpu_ops = SwsOpList::build(
            &src_c.with_domain(MemoryDomain::Cpu),
            &dst_c.with_domain(MemoryDomain::Cpu),
            opts,
        )
        .unwrap();
        let mut cpu_compiled = CpuSwsBackend::new().compile(&cpu_ops).unwrap();
        let cpu_dst = cpu_compiled.execute(&cpu_src, &pool).unwrap();

        for y in 0..12usize {
            let gr = downloaded.planes().unwrap()[0].row(y).unwrap();
            let cr = cpu_dst.planes().unwrap()[0].row(y).unwrap();
            for (gv, cv) in gr.iter().zip(cr.iter()) {
                assert!((*gv as i32 - *cv as i32).abs() <= 1);
            }
        }
    }
}
