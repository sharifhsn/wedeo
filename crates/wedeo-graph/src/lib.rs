//! VFR-native flat-indexed Data-Oriented Design (DOD) media DAG, auto-adapter compiler,
//! GPU residency domain placement, unified `Timeline` algebra (edit lists & GOP smart-cut),
//! and ergonomic custom `VideoFilter` / `AudioFilter` / `WgslFilter` framework.

#![forbid(unsafe_code)]

pub mod contract;
pub mod dag;
pub mod filter;
pub mod timeline;

pub use contract::{
    CompileOpts, ConversionNote, EdgeId, ExecutionPlan, ExtradataMode, NodeId, PacketContract,
    PacketFraming, PlannedEdge, PlannedNode, PortContract, classify_audio_conversion_loss,
    classify_video_conversion_loss,
};
pub use dag::{
    AudioFilterNode, AudioOpsNode, BsfNode, CompiledGraph, CpuBackedGpuSurface, DecoderNode,
    DemuxerNode, DomainTransferNode, EncoderNode, GraphBuilder, GraphItem, GraphNode, MuxerNode,
    RunReport, SinkNode, SourceNode, SwsNode, VideoFilterNode, WgslFilterNode,
    download_frame_to_cpu, upload_frame_to_gpu,
};
pub use filter::{
    AmixFilter, AudioFilter, BlackDetectFilter, BlackSpan, BwdifFilter, BwdifMode, BwdifParity,
    DrawBoxFilter, FadeFilter, FadeKind, FpsFilter, OverlayFilter, PadFilter, SetPtsFilter,
    SetPtsMode, SilenceDetectFilter, SilenceSpan, SubtitleRasterizer, TrimFilter, VideoFilter,
    WgslFilter,
};
pub use timeline::{
    ClipRange, CodedTime, GopSegment, KeyframeEntry, PresentationTime, SmartCutPlan, Timeline,
};
pub use wedeo_audio_ops::AudioContract;
pub use wedeo_video_ops::VideoContract;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use wedeo_core::{
        AudioBlock, BorderPadding, BorderedPlane, Buffer, BufferPool, ChannelLayout, CodecId,
        ColorDesc, FrameFlags, GpuDeviceId, MemoryDomain, Packet, PacketFlags, PixelLayout,
        PositionedPlane, Rational, SampleFormat, SubtitleEvent, SubtitlePayload, Timestamp,
        TrackId, VideoFrame,
    };
    use wedeo_video_ops::{SwsOpList, SwsOptions};

    #[allow(clippy::too_many_arguments)]
    fn make_yuv420p_frame(
        pool: &Arc<BufferPool>,
        w: u32,
        h: u32,
        y_val: u8,
        u_val: u8,
        v_val: u8,
        pts: i64,
        dur: i64,
        tb: Rational,
    ) -> VideoFrame {
        let mut py = BorderedPlane::new(pool, w as usize, h as usize, BorderPadding::NONE);
        for r in 0..h as usize {
            py.row_mut(r).unwrap().fill(y_val);
        }
        let (cw, ch) = PixelLayout::YUV420P.subsampling.chroma_dims(w, h);
        let mut pu = BorderedPlane::new(pool, cw as usize, ch as usize, BorderPadding::NONE);
        let mut pv = BorderedPlane::new(pool, cw as usize, ch as usize, BorderPadding::NONE);
        for r in 0..ch as usize {
            pu.row_mut(r).unwrap().fill(u_val);
            pv.row_mut(r).unwrap().fill(v_val);
        }
        let mut f = VideoFrame::from_planes(w, h, PixelLayout::YUV420P, vec![py, pu, pv]);
        f.color = ColorDesc::BT709_LIMITED;
        f.pts = Timestamp::new(pts);
        f.pkt_dts = Timestamp::new(pts);
        f.duration = dur;
        f.time_base = tb;
        f
    }

    fn make_f32p_audio_block(
        sample_rate: u32,
        layout: ChannelLayout,
        nb_samples: u32,
        val: f32,
        pts: i64,
        tb: Rational,
    ) -> AudioBlock {
        let channels = layout.nb_channels as usize;
        let mut planes = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut bytes = Vec::with_capacity(nb_samples as usize * 4);
            for _ in 0..nb_samples {
                bytes.extend_from_slice(&val.to_ne_bytes());
            }
            planes.push(Buffer::from_slice(&bytes));
        }
        let mut blk = AudioBlock::new(
            planes,
            nb_samples,
            SampleFormat::F32Planar,
            sample_rate,
            layout,
        );
        blk.pts = Timestamp::new(pts);
        blk.duration = nb_samples as i64;
        blk.time_base = tb;
        blk
    }

    // 1. End-to-end synthetic VFR video graph: Source -> Custom VideoFilter -> SwsOpList -> Sink
    #[test]
    fn end_to_end_vfr_video_graph_source_custom_filter_sws_sink() {
        struct LumaBoostFilter {
            delta: u8,
        }
        impl VideoFilter for LumaBoostFilter {
            fn process(
                &mut self,
                mut frame: VideoFrame,
                _pool: &BufferPool,
            ) -> wedeo_core::Result<Option<VideoFrame>> {
                crate::filter::ensure_planes_writable(&mut frame)?;
                let p0 = &mut frame.planes_mut().unwrap()[0];
                for y in 0..p0.height {
                    for b in p0.row_mut(y).unwrap() {
                        *b = b.saturating_add(self.delta);
                    }
                }
                Ok(Some(frame))
            }
        }

        let pool = BufferPool::new();
        let tb = Rational::new(1, 90_000);
        // Non-uniform VFR timestamps and durations: 0 (+1500), 1500 (+4500), 6000 (+2250)
        let vfr_schedule = [(0i64, 1500i64), (1500, 4500), (6000, 2250)];
        let frames: Vec<VideoFrame> = vfr_schedule
            .iter()
            .map(|&(pts, dur)| make_yuv420p_frame(&pool, 16, 16, 100, 128, 128, pts, dur, tb))
            .collect();

        let src_c =
            VideoContract::new(16, 16, PixelLayout::YUV420P).with_color(ColorDesc::BT709_LIMITED);
        let dst_c = VideoContract::new(32, 32, PixelLayout::RGBA8).with_color(ColorDesc::SRGB);

        let mut builder = GraphBuilder::new();
        let n_src = builder.add_node(Box::new(SourceNode::video(
            "vfr_src",
            src_c.clone(),
            frames,
        )));
        let n_flt = builder
            .add_video_filter("luma_boost", src_c.clone(), LumaBoostFilter { delta: 20 })
            .unwrap();
        let sws_ops = SwsOpList::build(&src_c, &dst_c, SwsOptions::default()).unwrap();
        let n_sws = builder.add_sws(sws_ops).unwrap();
        let n_sink = builder.add_sink("rgba_sink", PortContract::Video(dst_c));

        builder.connect(n_src, 0, n_flt, 0).unwrap();
        builder.connect(n_flt, 0, n_sws, 0).unwrap();
        builder.connect(n_sws, 0, n_sink, 0).unwrap();

        let (mut graph, plan) = builder.compile(CompileOpts::default()).unwrap();
        assert!(plan.auto_bridges.is_empty());
        assert!(plan.explain().contains("luma_boost"));
        assert!(plan.to_json().contains("\"name\": \"sws_op_list\""));

        let report = graph.run(&pool).unwrap();
        assert!(report.health.is_clean());

        let items = graph.take_sink_items(n_sink).unwrap();
        assert_eq!(items.len(), 3);
        for (i, item) in items.into_iter().enumerate() {
            let f = item.into_video().unwrap();
            assert_eq!((f.width, f.height), (32, 32));
            assert_eq!(f.layout, PixelLayout::RGBA8);
            assert_eq!(f.pts, Timestamp::new(vfr_schedule[i].0));
            assert_eq!(f.duration, vfr_schedule[i].1);
            assert_eq!(f.time_base, tb);
        }
    }

    // 2. Auto-adapter inserting Avcc -> AnnexB BSF automatically
    #[test]
    fn auto_adapter_inserts_avcc_to_annexb_bsf_losslessly() {
        let pool = BufferPool::new();
        let tb = Rational::new(1, 90_000);

        // Build an AVCC packet with 4-byte length prefix containing an IDR slice NAL `[0x65, 0x88, 0x80, 0x20]`
        let nal = [0x65u8, 0x88, 0x80, 0x20];
        let mut avcc_bytes = Vec::new();
        avcc_bytes.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        avcc_bytes.extend_from_slice(&nal);

        let mut pkt = Packet::new(Buffer::from_slice(&avcc_bytes), TrackId(0), tb);
        pkt.pts = Timestamp::new(3003);
        pkt.dts = Timestamp::new(3003);
        pkt.duration = 3003;
        pkt.flags = PacketFlags::KEY;

        let src_contract = PacketContract::new(CodecId::H264, PacketFraming::Avcc(4));
        let dst_contract = PacketContract::new(CodecId::H264, PacketFraming::AnnexB);

        let mut builder = GraphBuilder::new();
        let n_src = builder.add_node(Box::new(SourceNode::packets(
            "mp4_avcc_src",
            src_contract,
            vec![pkt],
        )));
        let n_sink = builder.add_sink("annexb_sink", PortContract::Packet(dst_contract));
        builder.connect(n_src, 0, n_sink, 0).unwrap();

        // Should succeed even in strict mode because Avcc -> AnnexB is 100% lossless!
        let (mut graph, plan) = builder
            .compile(CompileOpts::default().with_strict(true))
            .unwrap();
        assert_eq!(plan.auto_bridges.len(), 1);
        assert_eq!(plan.conversion_notes.len(), 1);
        assert!(!plan.conversion_notes[0].is_lossy);
        assert!(
            plan.conversion_notes[0]
                .one_line()
                .contains("avcc(4) -> annexb")
        );

        let report = graph.run(&pool).unwrap();
        assert_eq!(report.packets_processed, 2); // 1 from source + 1 from BSF bridge

        let out_items = graph.take_sink_items(n_sink).unwrap();
        assert_eq!(out_items.len(), 1);
        let out_pkt = out_items.into_iter().next().unwrap().into_packet().unwrap();
        assert_eq!(out_pkt.data.data(), &[0, 0, 0, 1, 0x65, 0x88, 0x80, 0x20]);
        assert_eq!(out_pkt.pts, Timestamp::new(3003));
        assert_eq!(out_pkt.duration, 3003);
    }

    // 3. Auto-adapter emitting ConversionNote on HDR -> SDR lossy bridge AND failing under strict: true
    #[test]
    fn auto_adapter_emits_conversion_note_on_hdr_to_sdr_and_fails_in_strict_mode() {
        let hdr_contract = VideoContract::new(1920, 1080, PixelLayout::YUV420P10LE)
            .with_color(ColorDesc::BT2020_PQ);
        let sdr_contract = VideoContract::new(1280, 720, PixelLayout::YUV420P)
            .with_color(ColorDesc::BT709_LIMITED);

        // Non-strict compile succeeds and records detailed lossy ConversionNote
        let mut b_permissive = GraphBuilder::new();
        let s1 = b_permissive.add_source("hdr_src", PortContract::Video(hdr_contract.clone()), []);
        let k1 = b_permissive.add_sink("sdr_sink", PortContract::Video(sdr_contract.clone()));
        b_permissive.connect(s1, 0, k1, 0).unwrap();

        let (_graph, plan) = b_permissive
            .compile(CompileOpts::default().with_strict(false))
            .unwrap();
        assert_eq!(plan.auto_bridges.len(), 1);
        assert_eq!(plan.conversion_notes.len(), 1);
        let note = &plan.conversion_notes[0];
        assert!(note.is_lossy);
        let line = note.one_line();
        assert!(line.contains("HDR->SDR tone-map"), "got note: {line}");
        assert!(line.contains("color gamut narrowing"), "got note: {line}");
        assert!(line.contains("downscale"), "got note: {line}");
        assert!(line.contains("bit-depth reduction"), "got note: {line}");

        // Strict compile rejects the implicit lossy HDR -> SDR bridge
        let mut b_strict = GraphBuilder::new();
        let s2 = b_strict.add_source("hdr_src", PortContract::Video(hdr_contract), []);
        let k2 = b_strict.add_sink("sdr_sink", PortContract::Video(sdr_contract));
        b_strict.connect(s2, 0, k2, 0).unwrap();

        let strict_res = b_strict.compile(CompileOpts::default().with_strict(true));
        assert!(strict_res.is_err());
        let err_msg = strict_res.err().unwrap().to_string();
        assert!(
            err_msg.contains("strict mode rejected implicit lossy video bridge"),
            "unexpected error: {err_msg}"
        );
    }

    // 4. GPU domain placement test (detecting GPU->CPU->GPU ping-pong warning in ExecutionPlan)
    #[test]
    fn gpu_domain_placement_detects_forced_ping_pong_and_promotes_flexible_sws() {
        struct DummyWgsl;
        impl WgslFilter for DummyWgsl {
            fn wgsl_source(&self) -> &str {
                "@compute @workgroup_size(16, 16, 1) fn main() {}"
            }
        }

        struct CpuOnlyFilter;
        impl VideoFilter for CpuOnlyFilter {
            fn process(
                &mut self,
                frame: VideoFrame,
                _pool: &BufferPool,
            ) -> wedeo_core::Result<Option<VideoFrame>> {
                Ok(Some(frame))
            }
        }

        let dev = GpuDeviceId(7);
        let gpu_c = VideoContract::new(64, 64, PixelLayout::RGBA8)
            .with_color(ColorDesc::SRGB)
            .with_domain(MemoryDomain::Gpu(dev));
        let cpu_c = VideoContract::new(64, 64, PixelLayout::RGBA8)
            .with_color(ColorDesc::SRGB)
            .with_domain(MemoryDomain::Cpu);

        // Case A: GPU WgslFilter -> CPU-only VideoFilter -> GPU WgslFilter => flags forced GPU->CPU->GPU bounce!
        let mut builder = GraphBuilder::new();
        let n_w1 = builder
            .add_wgsl_filter("wgsl_stage_1", gpu_c.clone(), DummyWgsl)
            .unwrap();
        let n_cpu = builder
            .add_video_filter("cpu_draw_box", cpu_c.clone(), CpuOnlyFilter)
            .unwrap();
        let n_w2 = builder
            .add_wgsl_filter("wgsl_stage_2", gpu_c.clone(), DummyWgsl)
            .unwrap();

        builder.connect(n_w1, 0, n_cpu, 0).unwrap();
        builder.connect(n_cpu, 0, n_w2, 0).unwrap();

        let (_g, plan) = builder
            .compile(CompileOpts::default().with_prefer_gpu(Some(dev)))
            .unwrap();
        assert_eq!(
            plan.gpu_warnings.len(),
            1,
            "expected 1 GPU->CPU->GPU warning, got {:?}",
            plan.gpu_warnings
        );
        assert!(
            plan.gpu_warnings[0].contains("forced GPU->CPU->GPU bounce"),
            "got: {}",
            plan.gpu_warnings[0]
        );
        assert!(plan.explain().contains("GPU Residency Warnings:"));

        // Case B: GPU WgslFilter -> flexible SwsNode -> GPU WgslFilter => SwsNode promoted to GPU with 0 warnings!
        let mut builder2 = GraphBuilder::new();
        let w1 = builder2
            .add_wgsl_filter("wgsl_1", gpu_c.clone(), DummyWgsl)
            .unwrap();
        let sws_ops = SwsOpList::build(&cpu_c, &cpu_c, SwsOptions::default()).unwrap();
        let sws = builder2.add_sws(sws_ops).unwrap();
        let w2 = builder2
            .add_wgsl_filter("wgsl_2", gpu_c, DummyWgsl)
            .unwrap();
        builder2.connect(w1, 0, sws, 0).unwrap();
        builder2.connect(sws, 0, w2, 0).unwrap();

        let (_g2, plan2) = builder2
            .compile(CompileOpts::default().with_prefer_gpu(Some(dev)))
            .unwrap();
        assert!(
            plan2.gpu_warnings.is_empty(),
            "flexible SwsNode between GPU nodes should be promoted to GPU without bounce: {:?}",
            plan2.gpu_warnings
        );
        let sws_planned = plan2.nodes.iter().find(|n| n.id == sws).unwrap();
        assert_eq!(sws_planned.domain, MemoryDomain::Gpu(dev));
    }

    // 5. Custom 15-line VideoFilter & AudioFilter & WgslFilter declaration test
    #[test]
    fn custom_15_line_video_audio_and_wgsl_filter_declarations() {
        // ~15-line custom VideoFilter
        struct InvertLuma;
        impl VideoFilter for InvertLuma {
            fn process(
                &mut self,
                mut frame: VideoFrame,
                _pool: &BufferPool,
            ) -> wedeo_core::Result<Option<VideoFrame>> {
                crate::filter::ensure_planes_writable(&mut frame)?;
                for y in 0..frame.height as usize {
                    for px in frame.planes_mut().unwrap()[0].row_mut(y).unwrap() {
                        *px = 255 - *px;
                    }
                }
                Ok(Some(frame))
            }
        }

        // ~15-line custom AudioFilter
        struct HalfVolume;
        impl AudioFilter for HalfVolume {
            fn process(
                &mut self,
                mut block: AudioBlock,
                _pool: &BufferPool,
            ) -> wedeo_core::Result<Option<AudioBlock>> {
                for plane in &mut block.planes {
                    let mut out = Vec::with_capacity(plane.len());
                    for &chunk in plane.data().as_chunks::<4>().0 {
                        let s = f32::from_ne_bytes(chunk) * 0.5;
                        out.extend_from_slice(&s.to_ne_bytes());
                    }
                    *plane = Buffer::from_slice(&out);
                }
                Ok(Some(block))
            }
        }

        // ~15-line custom WgslFilter
        struct Vignette {
            strength_bytes: [u8; 4],
        }
        impl WgslFilter for Vignette {
            fn wgsl_source(&self) -> &str {
                "@compute @workgroup_size(16, 16, 1) fn main() {}"
            }
            fn uniforms(&self, _pts_sec: f64, _w: u32, _h: u32) -> &[u8] {
                &self.strength_bytes
            }
        }

        let pool = BufferPool::new();
        let tb = Rational::new(1, 1000);

        // Execute custom VideoFilter + WgslFilter in one graph
        let vc =
            VideoContract::new(8, 8, PixelLayout::YUV420P).with_color(ColorDesc::BT709_LIMITED);
        let f_in = make_yuv420p_frame(&pool, 8, 8, 55, 128, 128, 500, 33, tb);

        let mut vb = GraphBuilder::new();
        let vs = vb.add_node(Box::new(SourceNode::video("v_src", vc.clone(), vec![f_in])));
        let vf = vb
            .add_video_filter("invert", vc.clone(), InvertLuma)
            .unwrap();
        let vw = vb
            .add_wgsl_filter(
                "vignette",
                vc.clone().with_domain(MemoryDomain::Gpu(GpuDeviceId(1))),
                Vignette {
                    strength_bytes: 0.4_f32.to_le_bytes(),
                },
            )
            .unwrap();
        let vk = vb.add_sink("v_sink", PortContract::Video(vc));
        vb.connect(vs, 0, vf, 0).unwrap();
        vb.connect(vf, 0, vw, 0).unwrap();
        vb.connect(vw, 0, vk, 0).unwrap();

        let (mut vg, _vplan) = vb.compile(CompileOpts::default()).unwrap();
        vg.run(&pool).unwrap();
        let out_v = vg
            .take_sink_items(vk)
            .unwrap()
            .pop()
            .unwrap()
            .into_video()
            .unwrap();
        assert_eq!(out_v.planes().unwrap()[0].row(0).unwrap()[0], 200);
        assert_eq!(out_v.pts, Timestamp::new(500));

        // Execute custom AudioFilter
        let ac = AudioContract::new(SampleFormat::F32Planar, 48_000, ChannelLayout::mono());
        let a_in = make_f32p_audio_block(
            48_000,
            ChannelLayout::mono(),
            64,
            0.8,
            0,
            Rational::new(1, 48_000),
        );
        let mut ab = GraphBuilder::new();
        let a_src = ab.add_node(Box::new(SourceNode::audio("a_src", ac.clone(), vec![a_in])));
        let a_flt = ab
            .add_audio_filter("half_vol", ac.clone(), HalfVolume)
            .unwrap();
        let a_sink = ab.add_sink("a_sink", PortContract::Audio(ac));
        ab.connect(a_src, 0, a_flt, 0).unwrap();
        ab.connect(a_flt, 0, a_sink, 0).unwrap();

        let (mut ag, _) = ab.compile(CompileOpts::default()).unwrap();
        ag.run(&pool).unwrap();
        let out_a = ag
            .take_sink_items(a_sink)
            .unwrap()
            .pop()
            .unwrap()
            .into_audio()
            .unwrap();
        let s0 = f32::from_ne_bytes(out_a.planes[0].data()[0..4].try_into().unwrap());
        assert!((s0 - 0.4).abs() < 1e-6);
    }

    // 6. Timeline smart-cut GOP plan test (ReencodeHead + CopyInterior + ReencodeTail)
    #[test]
    fn timeline_smart_cut_gop_plan_head_interior_tail_and_algebra() {
        let tb = Rational::new(1, 1000);
        // Keyframes every 1000ms: 0, 1000, 2000, 3000, 4000; stream_end = 5000
        let keyframes = vec![
            KeyframeEntry::from_pts(0, tb),
            KeyframeEntry::from_pts(1000, tb),
            KeyframeEntry::from_pts(2000, tb),
            KeyframeEntry::from_pts(3000, tb),
            KeyframeEntry::from_pts(4000, tb),
        ];

        // Cut from 400ms to 3600ms:
        // - GOP 0 [0..1000) -> ReencodeHead { decode_from: 0, pts_start: 400, pts_end: 1000 }
        // - GOP 1..3 [1000..3000) -> CopyInterior { pts_start: 1000, pts_end: 3000, gop_range: 1..3 }
        // - GOP 3 [3000..4000) -> ReencodeTail { decode_from: 3000, pts_start: 3000, pts_end: 3600 }
        let clip = ClipRange::from_ticks(400, 3600, tb);
        let plan = SmartCutPlan::compute(clip, &keyframes, Timestamp::new(5000)).unwrap();

        assert_eq!(plan.segments.len(), 3);
        assert_eq!(
            plan.segments[0],
            GopSegment::ReencodeHead {
                decode_from: Timestamp::new(0),
                pts_start: Timestamp::new(400),
                pts_end: Timestamp::new(1000),
                gop_index: 0,
                time_base: tb,
            }
        );
        assert_eq!(
            plan.segments[1],
            GopSegment::CopyInterior {
                pts_start: Timestamp::new(1000),
                pts_end: Timestamp::new(3000),
                gop_range: 1..3,
                time_base: tb,
            }
        );
        assert_eq!(
            plan.segments[2],
            GopSegment::ReencodeTail {
                decode_from: Timestamp::new(3000),
                pts_start: Timestamp::new(3000),
                pts_end: Timestamp::new(3600),
                gop_index: 3,
                time_base: tb,
            }
        );
        assert!(plan.head().unwrap().is_reencode());
        assert!(plan.interior().unwrap().is_copy());
        assert!(plan.tail().unwrap().is_reencode());

        // Test multi-segment Timeline CodedTime <-> PresentationTime mapping & EditSegment roundtrip
        let timeline = Timeline::new(vec![
            ClipRange::from_ticks(1000, 2000, tb), // 1000ms duration -> pres [0..1000)
            ClipRange::from_ticks(3500, 5000, tb), // 1500ms duration -> pres [1000..2500)
        ]);
        assert_eq!(timeline.total_duration(tb), 2500);
        assert_eq!(
            timeline.coded_to_presentation(CodedTime(Timestamp::new(500)), tb),
            None
        );
        assert_eq!(
            timeline.coded_to_presentation(CodedTime(Timestamp::new(1250)), tb),
            Some(PresentationTime(Timestamp::new(250)))
        );
        assert_eq!(
            timeline.coded_to_presentation(CodedTime(Timestamp::new(4000)), tb),
            Some(PresentationTime(Timestamp::new(1500)))
        );
        assert_eq!(
            timeline.presentation_to_coded(PresentationTime(Timestamp::new(1500)), tb),
            Some((1, CodedTime(Timestamp::new(4000))))
        );

        let elst = timeline.to_edit_list();
        assert_eq!(Timeline::from_edit_list(&elst), timeline);
    }

    // 7. Audio-only AudioFilter + AudioOpList graph test
    #[test]
    fn audio_only_filter_and_audio_op_list_graph() {
        let pool = BufferPool::new();
        let tb = Rational::new(1, 48_000);

        // Input: 48 kHz 5.1 F32Planar -> FadeFilter -> SilenceDetectFilter -> auto AudioOpList (48kHz Stereo S16) -> Sink
        let src_c = AudioContract::new(
            SampleFormat::F32Planar,
            48_000,
            ChannelLayout::surround_5_1(),
        );
        let dst_c = AudioContract::new(SampleFormat::S16, 48_000, ChannelLayout::stereo())
            .with_frame_size(Some(512));

        let b0 = make_f32p_audio_block(48_000, ChannelLayout::surround_5_1(), 512, 0.25, 0, tb);
        let b1_silent =
            make_f32p_audio_block(48_000, ChannelLayout::surround_5_1(), 512, 0.0, 512, tb);

        let mut builder = GraphBuilder::new();
        let n_src = builder.add_node(Box::new(SourceNode::audio(
            "audio_src",
            src_c.clone(),
            vec![b0, b1_silent],
        )));
        let n_fade = builder
            .add_audio_filter(
                "audio_fade_in",
                src_c.clone(),
                FadeFilter::new(FadeKind::In, Timestamp::new(0), 256, tb),
            )
            .unwrap();
        let n_silence = builder
            .add_audio_filter(
                "silence_detect",
                src_c,
                SilenceDetectFilter::new(-45.0, 0.005),
            )
            .unwrap();
        let n_sink = builder.add_sink("stereo_s16_sink", PortContract::Audio(dst_c));

        builder.connect(n_src, 0, n_fade, 0).unwrap();
        builder.connect(n_fade, 0, n_silence, 0).unwrap();
        // Auto-adapter inserts AudioOpList (5.1 F32Planar -> Stereo S16 frame=512)
        builder.connect(n_silence, 0, n_sink, 0).unwrap();

        let (mut graph, plan) = builder.compile(CompileOpts::default()).unwrap();
        assert_eq!(plan.auto_bridges.len(), 1);
        assert!(plan.conversion_notes[0].is_lossy);
        assert!(
            plan.conversion_notes[0]
                .one_line()
                .contains("channel downmix")
        );

        let report = graph.run(&pool).unwrap();
        assert!(report.health.is_clean());

        // Verify SilenceDetectFilter recorded the second 512-sample block as silent
        let sd_node = graph
            .node(n_silence)
            .unwrap()
            .as_any()
            .downcast_ref::<AudioFilterNode<SilenceDetectFilter>>()
            .unwrap();
        let silences = sd_node.filter().detected_silences();
        assert_eq!(silences.len(), 1);
        assert_eq!(silences[0].start, Timestamp::new(512));
        assert_eq!(silences[0].end, Timestamp::new(1024));

        let out_blocks: Vec<AudioBlock> = graph
            .take_sink_items(n_sink)
            .unwrap()
            .into_iter()
            .map(|i| i.into_audio().unwrap())
            .collect();
        assert_eq!(out_blocks.len(), 2);
        assert_eq!(out_blocks[0].format, SampleFormat::S16);
        assert_eq!(out_blocks[0].channel_layout, ChannelLayout::stereo());
        assert_eq!(out_blocks[0].nb_samples, 512);
    }

    // 8. Built-in essential filters comprehensive test
    #[test]
    fn builtin_essential_filters_vfr_behavior() {
        let pool = BufferPool::new();
        let tb = Rational::new(1, 1000);

        // A. PadFilter + DrawBoxFilter + OverlayFilter (with bitmap PGS subtitle) + BlackDetectFilter
        let f_black = make_yuv420p_frame(&pool, 8, 8, 16, 128, 128, 0, 100, tb);
        let f_bright = make_yuv420p_frame(&pool, 8, 8, 180, 128, 128, 100, 100, tb);

        let mut pad = PadFilter::new(16, 16, 4, 4);
        let f_padded = pad.process(f_bright, &pool).unwrap().unwrap();
        assert_eq!((f_padded.width, f_padded.height), (16, 16));
        assert_eq!(f_padded.pts, Timestamp::new(100));
        // Border (0,0) is Y=16 black; interior (6,6) is Y=180
        assert_eq!(f_padded.planes().unwrap()[0].row(0).unwrap()[0], 16);
        assert_eq!(f_padded.planes().unwrap()[0].row(6).unwrap()[6], 180);

        // DrawBoxFilter filled box at (0, 0, 4, 4) with Y=235
        let mut drawbox = DrawBoxFilter::new(0, 0, 4, 4, [235, 128, 128, 255], 1, true);
        let f_boxed = drawbox.process(f_padded, &pool).unwrap().unwrap();
        assert_eq!(f_boxed.planes().unwrap()[0].row(1).unwrap()[1], 235);

        // OverlayFilter with timed PGS/DVD bitmap SubtitleEvent active in [90..150)
        let mut sub_plane = BorderedPlane::new(&pool, 4 * 4, 4, BorderPadding::NONE);
        for y in 0..4 {
            sub_plane.row_mut(y).unwrap().fill(255); // opaque white RGBA8
        }
        let sub_ev = SubtitleEvent {
            pts: Timestamp::new(90),
            duration: 60,
            time_base: tb,
            payload: SubtitlePayload::Bitmap(vec![PositionedPlane {
                x: 8,
                y: 8,
                width: 4,
                height: 4,
                layout: PixelLayout::RGBA8,
                plane: sub_plane,
            }]),
        };
        let mut overlay = OverlayFilter::new(0, 0);
        overlay.push_subtitle(sub_ev);
        let f_ov = overlay.process(f_boxed, &pool).unwrap().unwrap();
        assert!((f_ov.planes().unwrap()[0].row(9).unwrap()[9] as i32 - 235).abs() <= 1);

        // BlackDetectFilter detects f_black [0..100) and stops on f_ov [100..200)
        let mut bdetect = BlackDetectFilter::new(0.10, 0.95, 0.05);
        bdetect.process(f_black, &pool).unwrap();
        bdetect.process(f_ov, &pool).unwrap();
        bdetect.flush(&pool).unwrap();
        assert_eq!(bdetect.detected_black_spans().len(), 1);
        assert_eq!(bdetect.detected_black_spans()[0].start, Timestamp::new(0));
        assert_eq!(bdetect.detected_black_spans()[0].end, Timestamp::new(100));

        // B. FpsFilter explicit VFR -> CFR conversion (10 fps => 100ms period)
        let mut fps = FpsFilter::new(Rational::new(10, 1)).unwrap();
        // Frame 0 at 0ms with 200ms duration -> emits CFR pts=0 and queues duplicate at CFR pts=1
        let v0 = make_yuv420p_frame(&pool, 8, 8, 100, 128, 128, 0, 200, tb);
        let c0 = fps.process(v0, &pool).unwrap().unwrap();
        let c1 = fps.flush(&pool).unwrap().unwrap();
        assert_eq!(c0.pts, Timestamp::new(0));
        assert_eq!(c1.pts, Timestamp::new(1));
        assert_eq!(c1.time_base, Rational::new(1, 10));

        // C. SetPtsFilter + TrimFilter + BwdifFilter + AmixFilter
        let mut setpts = SetPtsFilter::offset(500);
        let c1_shifted = VideoFilter::process(&mut setpts, c1, &pool)
            .unwrap()
            .unwrap();
        assert_eq!(c1_shifted.pts, Timestamp::new(501));

        let mut trim = TrimFilter::new(ClipRange::from_ticks(500, 502, Rational::new(1, 10)), true);
        let c1_trimmed = VideoFilter::process(&mut trim, c1_shifted, &pool)
            .unwrap()
            .unwrap();
        assert_eq!(c1_trimmed.pts, Timestamp::new(1));

        let mut interlaced = make_yuv420p_frame(&pool, 8, 8, 120, 128, 128, 0, 40, tb);
        interlaced.flags = FrameFlags::INTERLACED | FrameFlags::TOP_FIELD_FIRST;
        let mut bwdif = BwdifFilter::new(BwdifMode::SendField, BwdifParity::Auto, false);
        let field0 = bwdif.process(interlaced, &pool).unwrap().unwrap();
        let field1 = bwdif.flush(&pool).unwrap().unwrap();
        assert!(!field0.flags.contains(FrameFlags::INTERLACED));
        assert_eq!(field0.pts, Timestamp::new(0));
        assert_eq!(field1.pts, Timestamp::new(20));

        let a1 = make_f32p_audio_block(
            48_000,
            ChannelLayout::mono(),
            16,
            0.6,
            0,
            Rational::new(1, 48_000),
        );
        let a2 = make_f32p_audio_block(
            48_000,
            ChannelLayout::mono(),
            16,
            0.2,
            0,
            Rational::new(1, 48_000),
        );
        let mut amix = AmixFilter::new(2, true);
        amix.push_secondary(a2);
        let mixed = amix.process(a1, &pool).unwrap().unwrap();
        let m0 = f32::from_ne_bytes(mixed.planes[0].data()[0..4].try_into().unwrap());
        assert!((m0 - 0.4).abs() < 1e-6);
    }

    // 9. Cycle detection and ExecutionPlan JSON/explain test
    #[test]
    fn cycle_detection_rejects_cyclic_graph() {
        struct PassThrough;
        impl VideoFilter for PassThrough {
            fn process(
                &mut self,
                frame: VideoFrame,
                _pool: &BufferPool,
            ) -> wedeo_core::Result<Option<VideoFrame>> {
                Ok(Some(frame))
            }
        }

        let vc =
            VideoContract::new(16, 16, PixelLayout::YUV420P).with_color(ColorDesc::BT709_LIMITED);
        let mut builder = GraphBuilder::new();
        let n0 = builder
            .add_video_filter("a", vc.clone(), PassThrough)
            .unwrap();
        let n1 = builder.add_video_filter("b", vc, PassThrough).unwrap();
        builder.connect(n0, 0, n1, 0).unwrap();
        builder.connect(n1, 0, n0, 0).unwrap();

        let res = builder.compile(CompileOpts::default());
        assert!(res.is_err());
        assert!(res.err().unwrap().to_string().contains("cycle detected"));
    }
}
