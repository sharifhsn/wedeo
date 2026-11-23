//! Data-Oriented Design (DOD) steady-state allocation & performance invariant
//! tests using [`wedeo_oracle::CountingAlloc`].

use std::sync::Arc;

use wedeo::audio_ops::{AudioContract, AudioDither, AudioOp, AudioOpList, CompiledAudioOps};
use wedeo::core::{
    AudioBlock, BorderPadding, BorderedPlane, Buffer, BufferPool, ChannelLayout, ColorDesc,
    PixelLayout, Rational, SampleFormat, Timestamp, VideoFrame,
};
use wedeo::video_ops::{CpuSwsBackend, SwsBackend, SwsOpList, SwsOptions, VideoContract};
use wedeo_oracle::{assert_zero_allocs, measure_allocs};

#[test]
fn dod_buffer_pool_warm_acquire_reuses_backing_pages() {
    const PLANE_BYTES: usize = 1920 * 1080;
    let pool = BufferPool::new();

    let (_, cold_stats) = measure_allocs(|| {
        let buf = pool.acquire(PLANE_BYTES);
        assert_eq!(buf.len(), PLANE_BYTES);
    });
    assert!(
        cold_stats.bytes_allocated >= PLANE_BYTES,
        "expected cold acquire to allocate backing page (>= {PLANE_BYTES} B), got {cold_stats:?}"
    );
    assert_eq!(pool.pooled_count(), 1);

    let (_, warm_stats) = measure_allocs(|| {
        for _ in 0..16 {
            let buf = pool.acquire_reused(PLANE_BYTES);
            assert_eq!(buf.len(), PLANE_BYTES);
        }
    });
    assert_eq!(
        warm_stats.reallocs, 0,
        "warm BufferPool acquire should never realloc: {warm_stats:?}"
    );
    // Only tiny Arc<PooledRaw> wrapper handles may be allocated; backing pages are reused.
    assert!(
        warm_stats.bytes_allocated <= 16 * 128,
        "warm BufferPool acquire allocated backing pages instead of reusing: {warm_stats:?}"
    );
    assert_eq!(pool.pooled_count(), 1);
}

#[test]
fn dod_cpu_sws_backend_zero_scratch_alloc_on_warm_frames() {
    let src_pool = BufferPool::new();
    let dst_pool = BufferPool::new();

    let src_w = 640u32;
    let src_h = 360u32;
    let dst_w = 640u32;
    let dst_h = 360u32;

    let py = BorderedPlane::new(
        &src_pool,
        src_w as usize,
        src_h as usize,
        BorderPadding::NONE,
    );
    let (cw, ch) = PixelLayout::YUV420P.subsampling.chroma_dims(src_w, src_h);
    let pu = BorderedPlane::new(&src_pool, cw as usize, ch as usize, BorderPadding::NONE);
    let pv = BorderedPlane::new(&src_pool, cw as usize, ch as usize, BorderPadding::NONE);
    let mut src = VideoFrame::from_planes(src_w, src_h, PixelLayout::YUV420P, vec![py, pu, pv]);
    src.color = ColorDesc::BT709_LIMITED;

    let src_c = VideoContract::from_frame(&src);
    let dst_c = VideoContract {
        width: dst_w,
        height: dst_h,
        layout: PixelLayout::RGBA8,
        color: ColorDesc::SRGB,
        ..src_c.clone()
    };

    let ops = SwsOpList::build(&src_c, &dst_c, SwsOptions::default()).expect("build SwsOpList");
    let mut compiled = CpuSwsBackend::new()
        .compile(&ops)
        .expect("compile CpuSwsBackend");

    // Warm up dst_pool with one frame so output plane backing pages are pooled.
    let (_, cold_stats) = measure_allocs(|| {
        let out = compiled.execute(&src, &dst_pool).expect("warmup execute");
        assert_eq!(out.width, dst_w);
        assert_eq!(out.height, dst_h);
    });
    let expected_dst_bytes = (dst_w * dst_h * 4) as usize;
    assert!(
        cold_stats.bytes_allocated >= expected_dst_bytes,
        "expected cold frame to allocate output plane (>= {expected_dst_bytes} B), got {cold_stats:?}"
    );

    // Steady-state warm frames must perform zero internal 256x16 scratch or output page allocs.
    let (_, warm_stats) = measure_allocs(|| {
        for _ in 0..4 {
            let out = compiled.execute(&src, &dst_pool).expect("warm execute");
            assert_eq!(out.width, dst_w);
        }
    });
    assert_eq!(
        warm_stats.reallocs, 0,
        "CpuSwsBackend warm frames must not realloc: {warm_stats:?}"
    );
    assert!(
        warm_stats.bytes_allocated < 2048,
        "CpuSwsBackend allocated internal scratch or output pages on warm frames: {warm_stats:?}"
    );
}

#[test]
fn dod_audio_ops_zero_growth_alloc_after_warmup() {
    let pool = BufferPool::new();
    let src_c = AudioContract {
        sample_rate: 48_000,
        format: SampleFormat::S16,
        channel_layout: ChannelLayout::stereo(),
        frame_size: None,
    };

    let op_list = AudioOpList::builder(src_c)
        .format(SampleFormat::F32Planar)
        .sample_rate(44_100)
        .channels(ChannelLayout::mono())
        .dither(AudioDither::Triangular)
        .reframe(1024)
        .build()
        .expect("build AudioOpList");
    assert!(
        op_list
            .ops
            .iter()
            .any(|op| matches!(op, AudioOp::Resample(_))),
        "expected Resample op in AudioOpList"
    );

    let mut compiled = CompiledAudioOps::compile(&op_list).expect("compile AudioOpList");
    let mut out = Vec::with_capacity(8);

    let in_samples = 1024usize;
    let in_bytes = in_samples * 2 * 2; // S16 stereo
    let template_buf = Buffer::from_slice(&vec![0u8; in_bytes]);

    // Warm up internal 1024-sample slabs, ring buffer, and BufferPool.
    for i in 0..4 {
        let mut blk = AudioBlock::new(
            vec![template_buf.clone()],
            in_samples as u32,
            SampleFormat::S16,
            48_000,
            ChannelLayout::stereo(),
        );
        blk.pts = Timestamp::new(i * in_samples as i64);
        blk.time_base = Rational::new(1, 48_000);
        compiled
            .process_into(blk, &pool, &mut out)
            .expect("warmup process_into");
        out.clear();
    }

    // Pre-build input blocks outside the measurement window so we measure only CompiledAudioOps.
    let mut blocks = Vec::with_capacity(8);
    for i in 4..12 {
        let mut blk = AudioBlock::new(
            vec![template_buf.clone()],
            in_samples as u32,
            SampleFormat::S16,
            48_000,
            ChannelLayout::stereo(),
        );
        blk.pts = Timestamp::new(i * in_samples as i64);
        blk.time_base = Rational::new(1, 48_000);
        blocks.push(blk);
    }

    let (_, warm_stats) = measure_allocs(|| {
        for blk in blocks {
            compiled
                .process_into(blk, &pool, &mut out)
                .expect("steady-state process_into");
            out.clear();
        }
    });
    assert_eq!(
        warm_stats.reallocs, 0,
        "CompiledAudioOps must not grow/realloc buffers in steady state: {warm_stats:?}"
    );
    assert!(
        warm_stats.bytes_allocated < 4096,
        "CompiledAudioOps performed unexpected steady-state allocations: {warm_stats:?}"
    );
}

#[test]
fn dod_bordered_plane_extend_edges_strict_zero_allocs() {
    let pool = Arc::clone(&BufferPool::new());
    let mut plane = BorderedPlane::new(&pool, 1920, 1080, BorderPadding::uniform(32));
    for y in 0..1080 {
        plane.row_mut(y).unwrap().fill((y & 0xff) as u8);
    }

    assert_zero_allocs("BorderedPlane::extend_edges", || {
        assert!(plane.extend_edges(1));
    });
}
