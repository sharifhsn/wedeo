//! Run with a raw YUV420P frame, its width and its height. Prints per-stage microseconds.
use std::time::Instant;
use wedeo_core::{
    BorderPadding, BorderedPlane, BufferPool, ColorDesc, MemoryDomain, PixelLayout, VideoFrame,
};
use wedeo_gpu_wgpu::WgpuContext;
use wedeo_video_ops::{CpuSwsBackend, SwsBackend, SwsOpList, SwsOptions, VideoContract};

fn main() {
    let mut args = std::env::args().skip(1);
    let raw = std::fs::read(args.next().expect("raw YUV420P path")).unwrap();
    let w: u32 = args.next().expect("width").parse().unwrap();
    let h: u32 = args.next().expect("height").parse().unwrap();
    let pool = BufferPool::new();
    let layout = PixelLayout::YUV420P;
    let mut data = raw.as_slice();
    let mut planes = Vec::new();
    for i in 0..3 {
        let (bytes, rows) = layout.plane_dims(w, h, i).unwrap();
        let mut plane = BorderedPlane::new(&pool, bytes, rows, BorderPadding::NONE);
        for y in 0..rows {
            let (row, remaining) = data.split_at(bytes);
            plane.row_mut(y).unwrap().copy_from_slice(row);
            data = remaining;
        }
        planes.push(plane);
    }
    assert!(data.is_empty());
    let mut frame = VideoFrame::from_planes(w, h, layout, planes);
    frame.color = ColorDesc::BT709_LIMITED;
    let startup = Instant::now();
    let ctx = WgpuContext::new_headless().unwrap();
    let mode = args.next();
    if mode.as_deref() == Some("upload") {
        println!("trial,upload_us");
        for trial in 0..120 {
            let t = Instant::now();
            let resident = ctx.upload_frame(&frame).unwrap();
            ctx.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            println!("{trial},{}", t.elapsed().as_micros());
            if trial == 119 {
                let copy = ctx.download_frame(&resident, &pool).unwrap();
                for (original, downloaded) in
                    frame.planes().unwrap().iter().zip(copy.planes().unwrap())
                {
                    for y in 0..original.height {
                        assert_eq!(original.row(y), downloaded.row(y));
                    }
                }
            }
        }
        return;
    }
    let src = VideoContract::from_frame(&frame);
    if mode.as_deref() == Some("planar-loop") {
        use std::io::Write;
        let dst = VideoContract::new(w / 2, h / 2, layout).with_color(frame.color);
        let ops = SwsOpList::build(&src, &dst, SwsOptions::default()).unwrap();
        let mut pass = ctx.sws_backend().compile(&ops).unwrap();
        eprintln!(
            "adapter and pipeline setup: {} us",
            startup.elapsed().as_micros()
        );
        let mut output = std::io::BufWriter::new(
            std::fs::File::create(args.next().expect("output frame dump")).unwrap(),
        );
        println!("frame,conversion_and_readback_us");
        for i in 0..32 {
            // Change visible input each time so stale storage is observable.
            frame.planes_mut().unwrap()[0].row_mut(0).unwrap()[0] ^= 255;
            let t = Instant::now();
            let result = pass.execute(&frame, &pool).unwrap();
            println!("{i},{}", t.elapsed().as_micros());
            if i == 30 || i == 31 {
                for plane in result.planes().unwrap() {
                    for y in 0..plane.height {
                        output.write_all(plane.row(y).unwrap()).unwrap();
                    }
                }
            }
        }
        output.flush().unwrap();
        return;
    }
    if mode.as_deref() == Some("planar") {
        let dst = VideoContract::new(w / 2, h / 2, layout).with_color(frame.color);
        let ops = SwsOpList::build(&src, &dst, SwsOptions::default()).unwrap();
        let gpu = ctx
            .sws_backend()
            .compile(&ops)
            .unwrap()
            .execute(&frame, &pool)
            .unwrap();
        let cpu = CpuSwsBackend::new()
            .compile(&ops)
            .unwrap()
            .execute(&frame, &pool)
            .unwrap();
        for (index, (a, b)) in cpu
            .planes()
            .unwrap()
            .iter()
            .zip(gpu.planes().unwrap())
            .enumerate()
        {
            assert_eq!((a.width_bytes, a.height), (b.width_bytes, b.height));
            let mut sum = 0u64;
            let mut peak = 0;
            for y in 0..a.height {
                for (&x, &v) in a.row(y).unwrap().iter().zip(b.row(y).unwrap()) {
                    let diff = x.abs_diff(v);
                    sum += u64::from(diff);
                    peak = peak.max(diff);
                }
            }
            println!("{index},{sum},{peak},{}", a.width_bytes * a.height);
        }
        return;
    }
    let dst = VideoContract::new(w / 2, h / 2, PixelLayout::RGBA8).with_color(ColorDesc::SRGB);
    let cpu_ops = SwsOpList::build(&src, &dst, SwsOptions::default()).unwrap();
    let gpu_domain = MemoryDomain::Gpu(ctx.device_id());
    let ops = SwsOpList::build(
        &src.with_domain(gpu_domain),
        &dst.with_domain(gpu_domain),
        SwsOptions::default(),
    )
    .unwrap();
    let mut cpu = CpuSwsBackend::new().compile(&cpu_ops).unwrap();
    let mut gpu = ctx.sws_backend().compile(&ops).unwrap();
    let sync = || {
        ctx.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    };
    println!("trial,upload_us,dispatch_us,readback_us,cpu_us,abs_sum,max_diff,samples");
    for trial in 0..12 {
        let t = Instant::now();
        let resident = ctx.upload_frame(&frame).unwrap();
        sync();
        let uploaded = Instant::now();
        let rendered = gpu.execute(&resident, &pool).unwrap();
        assert_eq!(rendered.domain(), gpu_domain);
        sync();
        let dispatched = Instant::now();
        let got = ctx.download_frame(&rendered, &pool).unwrap();
        let downloaded = Instant::now();
        let reference = cpu.execute(&frame, &pool).unwrap();
        let finished = Instant::now();
        let (mut sum, mut max, mut samples) = (0u64, 0u8, 0usize);
        for y in 0..got.height as usize {
            let a = got.planes().unwrap()[0].row(y).unwrap();
            let b = reference.planes().unwrap()[0].row(y).unwrap();
            assert_eq!(a.len(), b.len());
            for (&x, &v) in a.iter().zip(b) {
                let diff = x.abs_diff(v);
                sum += u64::from(diff);
                max = max.max(diff);
            }
            samples += a.len();
        }
        println!(
            "{trial},{},{},{},{},{sum},{max},{samples}",
            (uploaded - t).as_micros(),
            (dispatched - uploaded).as_micros(),
            (downloaded - dispatched).as_micros(),
            (finished - downloaded).as_micros()
        );
    }
}
