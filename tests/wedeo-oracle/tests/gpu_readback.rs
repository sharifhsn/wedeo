//! Run with --features gpu-readback on a host with a hardware or software adapter.
#![cfg(feature = "gpu-readback")]

use std::sync::Arc;
use wedeo::gpu_wgpu::{WgpuContext, WgpuTextureSurface};
use wedeo::graph::download_frame_to_cpu;
use wedeo::{
    BorderPadding, BorderedPlane, BufferPool, GpuDeviceId, PixelLayout, Timestamp, VideoFrame,
};

#[test]
fn graph_reads_patterned_planes_after_context_handle_drops() {
    let pool = BufferPool::new();
    let layout = PixelLayout::YUV420P;
    let mut planes = Vec::new();
    for p in 0..layout.plane_count() {
        let (bytes, rows) = layout.plane_dims(17, 7, p).unwrap();
        let mut plane = BorderedPlane::new(&pool, bytes, rows, BorderPadding::NONE);
        for y in 0..rows {
            for (x, v) in plane.row_mut(y).unwrap().iter_mut().enumerate() {
                *v = (p * 67 + y * 13 + x) as u8;
            }
        }
        planes.push(plane);
    }
    let mut src = VideoFrame::from_planes(17, 7, layout, planes);
    src.pts = Timestamp::new(123);
    let resident = WgpuContext::new_headless()
        .unwrap()
        .upload_frame(&src)
        .unwrap();
    let out = download_frame_to_cpu(resident).unwrap();
    assert_eq!(out.pts, src.pts);
    let actual = out.planes().unwrap();
    assert_eq!(actual.len(), 3);
    for (a, b) in actual.iter().zip(src.planes().unwrap()) {
        assert_eq!((a.width_bytes, a.height), (b.width_bytes, b.height));
        for y in 0..a.height {
            assert_eq!(a.row(y).unwrap(), b.row(y).unwrap());
        }
    }
}

#[test]
fn unreadable_surface_returns_an_error() {
    let frame = VideoFrame::from_gpu(Arc::new(WgpuTextureSurface::new_virtual(
        17,
        7,
        PixelLayout::RGBA8,
        GpuDeviceId(5),
    )));
    assert!(download_frame_to_cpu(frame).is_err());
}

#[test]
fn graph_dispatches_two_shaders_and_preserves_vfr_timestamps() {
    use wedeo::graph::{
        CompileOpts, GraphBuilder, PortContract, SourceNode, VideoContract, WgslFilter,
    };
    struct AddRed;
    impl WgslFilter for AddRed {
        fn wgsl_source(&self) -> &str {
            r#"
            @group(0) @binding(0) var src: texture_2d<f32>;
            @group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
            @group(0) @binding(2) var<uniform> red: f32;
            @compute @workgroup_size(8,4,1)
            fn paint(@builtin(global_invocation_id) id: vec3<u32>) {
                if (any(id.xy >= textureDimensions(src))) { return; }
                let p = textureLoad(src, id.xy, 0);
                textureStore(dst, id.xy, p + vec4<f32>(red,0.,0.,0.));
            }"#
        }
        fn entry_point(&self) -> &str {
            "paint"
        }
        fn workgroup_size(&self) -> [u32; 3] {
            [8, 4, 1]
        }
        fn uniforms(&self, pts: f64, _: u32, _: u32) -> &[u8] {
            const FIRST: [u8; 4] = (10.0f32 / 255.0).to_ne_bytes();
            const SECOND: [u8; 4] = (20.0f32 / 255.0).to_ne_bytes();
            if pts < 0.4 { &FIRST } else { &SECOND }
        }
    }

    let pool = BufferPool::new();
    let mut plane = BorderedPlane::new(&pool, 19 * 4, 7, BorderPadding::NONE);
    for y in 0..7 {
        for (x, p) in plane.row_mut(y).unwrap().chunks_exact_mut(4).enumerate() {
            p.copy_from_slice(&[x as u8, y as u8, 99, 255]);
        }
    }
    let mut first = VideoFrame::from_planes(19, 7, PixelLayout::RGBA8, vec![plane]);
    first.time_base = wedeo::Rational::new(1, 1000);
    first.pts = Timestamp::new(70);
    let mut second = first.clone();
    second.pts = Timestamp::new(530);
    let cpu = VideoContract::from_frame(&first);
    let ctx = WgpuContext::new_headless().unwrap();
    let info = ctx.device().adapter_info();
    eprintln!(
        "graph shader adapter: {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    let gpu = cpu
        .clone()
        .with_domain(wedeo::MemoryDomain::Gpu(ctx.device_id()));
    let mut builder = GraphBuilder::new();
    builder.set_sws_backend(Box::new(ctx.sws_backend()));
    let source = builder.add_node(Box::new(SourceNode::video(
        "source",
        cpu.clone(),
        vec![first, second],
    )));
    let a = builder
        .add_wgsl_filter("first", gpu.clone(), AddRed)
        .unwrap();
    let b = builder.add_wgsl_filter("second", gpu, AddRed).unwrap();
    let sink = builder.add_sink("sink", PortContract::Video(cpu));
    for (src, dst) in [(source, a), (a, b), (b, sink)] {
        builder.connect(src, 0, dst, 0).unwrap();
    }
    let (mut graph, _) = builder.compile(CompileOpts::default()).unwrap();
    graph.run(&pool).unwrap();
    let items = graph.take_sink_items(sink).unwrap();
    assert_eq!(items.len(), 2);
    for (item, (pts, delta)) in items.into_iter().zip([(70, 20), (530, 40)]) {
        let frame = item.into_video().unwrap();
        assert_eq!(frame.pts, Timestamp::new(pts));
        for y in 0..7 {
            for (x, px) in frame.planes().unwrap()[0]
                .row(y)
                .unwrap()
                .chunks_exact(4)
                .enumerate()
            {
                assert_eq!(px, [x as u8 + delta, y as u8, 99, 255]);
            }
        }
    }
}
