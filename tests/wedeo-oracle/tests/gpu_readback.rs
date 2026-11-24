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
