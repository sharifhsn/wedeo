//! Standalone comparison with the pinned FFmpeg x86 fpel object.
//! Build FFmpeg's libavcodec/x86/fpel.o with its normal configure/make, then:
//! rustc --edition=2024 -O -C link-arg=FFmpeg/libavcodec/x86/fpel.o \
//!   scripts/bench_h264_avg_ffmpeg.rs -o /tmp/avg_ffmpeg
//! Run pinned to one CPU, with other builds stopped. FFmpeg requires aligned
//! 16-byte destination rows and identical source and destination strides.
#![allow(dead_code)]
#[path = "../codecs/wedeo-codec-h264/src/simd.rs"]
mod simd;
use std::{hint::black_box, time::Instant};
unsafe extern "C" {
    fn ff_avg_pixels8x8_sse2(dst: *mut u8, src: *const u8, stride: isize);
    fn ff_avg_pixels16x16_sse2(dst: *mut u8, src: *const u8, stride: isize);
}
#[repr(align(32))]
#[derive(Clone)]
struct Block([u8; 512]);
fn main() {
    assert!(std::arch::is_x86_feature_detected!("sse2"));
    let mut seed = 1u32;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as u8
    };
    for (size, stride) in [(8usize, 8usize), (8, 32), (16, 16), (16, 32)] {
        let src: Vec<Block> = (0..64)
            .map(|_| Block(std::array::from_fn(|_| random())))
            .collect();
        let initial: Vec<Block> = (0..64)
            .map(|_| Block(std::array::from_fn(|_| random())))
            .collect();
        let mut a = initial.clone();
        let mut b = initial.clone();
        for i in 0..64 {
            simd::avg_pixels_inplace(&mut a[i].0, stride, &src[i].0, stride, size, size);
            unsafe {
                if size == 8 {
                    ff_avg_pixels8x8_sse2(b[i].0.as_mut_ptr(), src[i].0.as_ptr(), stride as isize)
                } else {
                    ff_avg_pixels16x16_sse2(b[i].0.as_mut_ptr(), src[i].0.as_ptr(), stride as isize)
                }
            }
            assert_eq!(a[i].0, b[i].0, "size {size}, block {i}");
        }
        for trial in 0..7 {
            for ff in [trial % 2 == 0, trial % 2 != 0] {
                let mut dst = initial.clone();
                let start = Instant::now();
                for n in 0..500_000 {
                    let i = black_box(n & 63);
                    if ff {
                        unsafe {
                            if size == 8 {
                                ff_avg_pixels8x8_sse2(
                                    dst[i].0.as_mut_ptr(),
                                    src[i].0.as_ptr(),
                                    stride as isize,
                                )
                            } else {
                                ff_avg_pixels16x16_sse2(
                                    dst[i].0.as_mut_ptr(),
                                    src[i].0.as_ptr(),
                                    stride as isize,
                                )
                            }
                        }
                    } else {
                        simd::avg_pixels_inplace(
                            &mut dst[i].0,
                            stride,
                            &src[i].0,
                            stride,
                            size,
                            size,
                        );
                    }
                }
                println!(
                    "size={size} stride={stride} trial={trial} {} {:.3} ns/block",
                    if ff { "ffmpeg" } else { "wedeo" },
                    start.elapsed().as_secs_f64() * 1e9 / 500_000.0
                );
                black_box(dst);
            }
        }
    }
}
