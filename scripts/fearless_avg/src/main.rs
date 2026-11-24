use fearless_simd::{Level, dispatch, prelude::*};
use fearless_simd_macros::simd;
#[simd]
fn avg<S: Simd>(s: S, d: &mut [u8], src: &[u8]) {
    let n = S::u8s::LEN;
    let mut chunks = d.chunks_exact_mut(n);
    for (x, y) in chunks.by_ref().zip(src.chunks_exact(n)) {
        let a = S::u8s::from_slice(s, x);
        let b = S::u8s::from_slice(s, y);
        let v = (a & b) + ((a ^ b) >> 1) + ((a ^ b) & S::u8s::splat(s, 1));
        v.store_slice(x);
    }
    let rem = chunks.into_remainder();
    let start = src.len() - rem.len();
    for (x, y) in rem.iter_mut().zip(&src[start..]) {
        *x = ((*x as u16 + *y as u16 + 1) >> 1) as u8;
    }
}

#[simd]
fn avg_block<S: Simd>(s: S, d: &mut [u8], src: &[u8], size: usize, stride: usize) {
    if size == stride {
        avg(s, &mut d[..size * size], &src[..size * size]);
    } else {
        for row in 0..size {
            let o = row * stride;
            avg(s, &mut d[o..o + size], &src[o..o + size]);
        }
    }
}

#[path = "../../../codecs/wedeo-codec-h264/src/simd.rs"]
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
            let mut f = initial[i].clone();
            dispatch!(Level::new(), s => avg_block(s, &mut f.0, &src[i].0, size, stride));
            assert_eq!(f.0, b[i].0, "fearless size {size}, block {i}");
        }
        for trial in 0..7 {
            for ff in 0..3 {
                let mut dst = initial.clone();
                let start = Instant::now();
                for n in 0..500_000 {
                    let i = black_box(n & 63);
                    if ff == 0 {
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
                    } else if ff == 1 {
                        simd::avg_pixels_inplace(
                            &mut dst[i].0,
                            stride,
                            &src[i].0,
                            stride,
                            size,
                            size,
                        );
                    } else {
                        dispatch!(Level::new(), s => avg_block(s, &mut dst[i].0, &src[i].0, size, stride));
                    }
                }
                println!(
                    "size={size} stride={stride} trial={trial} {} {:.3} ns/block",
                    match ff {
                        0 => "ffmpeg",
                        1 => "intrinsics",
                        _ => "fearless",
                    },
                    start.elapsed().as_secs_f64() * 1e9 / 500_000.0
                );
                black_box(dst);
            }
        }
    }
}
