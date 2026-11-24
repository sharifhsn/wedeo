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

#[cfg(test)]
mod tests {
    use super::*;

    // The assembly permits unaligned source rows. For 16x16 it uses aligned
    // destination accesses, so both the destination offset and pitch must be
    // multiples of 16. Independent Wedeo strides are covered in checkasm.
    #[repr(align(32))]
    #[derive(Clone)]
    struct Plane([u8; 16 * 2048 + 64]);

    fn check_oracle(size: usize, pitch: usize, d: usize, s: usize, pair_base: usize) {
        let end = (size - 1) * pitch + size;
        let mut src = Plane([0xcd; 16 * 2048 + 64]);
        let mut input = Plane([0x39; 16 * 2048 + 64]);
        for row in 0..size {
            for col in 0..size {
                let pair = (pair_base + row * size + col) as u16;
                input.0[d + row * pitch + col] = (pair >> 8) as u8;
                src.0[s + row * pitch + col] = pair as u8;
            }
        }
        let src = &src.0[s..s + end];
        let mut expected = input.clone();
        let dst = &mut expected.0[d..d + end];
        // SAFETY: callers below use only 8/16, valid pitches/offsets per
        // fpel.asm and exact accessible extents for both blocks on x86-64.
        assert!(size == 8 || (size == 16 && dst.as_ptr().align_offset(16) == 0 && pitch % 16 == 0));
        assert!(pitch >= size);
        unsafe {
            let f = if size == 8 {
                ff_avg_pixels8x8_sse2
            } else {
                ff_avg_pixels16x16_sse2
            };
            f(dst.as_mut_ptr(), src.as_ptr(), pitch as isize);
        }
        let check = |label: &str, actual: &Plane| {
            if let Some(byte) = actual.0.iter().zip(&expected.0).position(|(a, b)| a != b) {
                panic!(
                    "{label} {size}x{size} pitch={pitch} dst={d} src={s} base={pair_base} byte={byte}"
                );
            }
        };
        for backend in std::iter::once(None).chain(simd::available_backends().into_iter().map(Some))
        {
            let mut actual = input.clone();
            let dst = &mut actual.0[d..d + end];
            if let Some(b) = backend {
                simd::avg_pixels_inplace_with_backend(dst, pitch, src, pitch, size, size, b);
            } else {
                simd::avg_pixels_inplace(dst, pitch, src, pitch, size, size);
            }
            check(&format!("wedeo {backend:?}"), &actual);
        }
        let check_level = |level| {
            let mut actual = input.clone();
            dispatch!(level, token => avg_block(token, &mut actual.0[d..d+end], src, size, pitch));
            check(&format!("fearless {level:?}"), &actual);
        };
        check_level(Level::new());
        #[cfg(feature = "fallback")]
        check_level(Level::fallback());
    }

    #[test]
    fn packed_every_input_pair_matches_ffmpeg() {
        for size in [8, 16] {
            for base in (0..65536).step_by(size * size) {
                check_oracle(size, size, 16, 1, base);
            }
        }
    }

    #[test]
    fn padded_and_offset_rows_match_ffmpeg() {
        for (size, pitch) in [(8, 9), (8, 32), (8, 2048), (16, 32), (16, 2048)] {
            for s in 0..16 {
                for d in if size == 8 { [1, 15] } else { [0, 16] } {
                    check_oracle(size, pitch, d, s, 0xaf00 + s * 17 + d);
                }
            }
        }
    }
}
