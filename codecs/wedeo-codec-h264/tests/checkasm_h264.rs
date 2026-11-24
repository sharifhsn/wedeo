//! `checkasm`-style bit-exactness suite: every `core::arch` SIMD kernel in
//! `wedeo_codec_h264::simd` must produce a 0-bit difference against its
//! pure-Rust scalar reference over thousands of randomized edge-case vectors
//! (extremal pixels, extremal coefficients, odd strides, canary guard bytes).

#![forbid(unsafe_code)]

use wedeo_codec_h264::simd::{self, SimdBackend, scalar};

const VECTORS: usize = 10_000;
const GUARD: usize = 32;
const CANARY: u8 = 0xCD;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: usize, hi_incl: usize) -> usize {
        lo + (self.next() >> 11) as usize % (hi_incl - lo + 1)
    }
}

/// Pixel fill modes: uniform random, all-0, all-255, extremes only, near-extremes.
fn fill_pixels(rng: &mut Rng, buf: &mut [u8]) {
    match rng.range(0, 5) {
        0 | 1 => buf.iter_mut().for_each(|b| *b = rng.next() as u8),
        2 => buf.fill(0),
        3 => buf.fill(255),
        4 => buf
            .iter_mut()
            .for_each(|b| *b = if rng.next() & 1 == 0 { 0 } else { 255 }),
        _ => buf.iter_mut().for_each(|b| {
            *b = match rng.next() % 4 {
                0 => 0,
                1 => 1,
                2 => 254,
                _ => 255,
            }
        }),
    }
}

/// Coefficient fill modes: zero, small, medium, full-range, extremal +/-.
fn fill_coeffs(rng: &mut Rng, c: &mut [i16]) {
    match rng.range(0, 6) {
        0 => c.fill(0),
        1 => c
            .iter_mut()
            .for_each(|v| *v = (rng.next() % 33) as i16 - 16),
        2 => c
            .iter_mut()
            .for_each(|v| *v = (rng.next() % 4097) as i16 - 2048),
        3 => c.iter_mut().for_each(|v| *v = rng.next() as i16),
        4 => c.iter_mut().for_each(|v| {
            *v = if rng.next() & 1 == 0 {
                i16::MAX
            } else {
                i16::MIN
            }
        }),
        5 => {
            c.fill(0);
            c[0] = rng.next() as i16; // DC-dominant
        }
        _ => c.iter_mut().for_each(|v| {
            *v = match rng.next() % 4 {
                0 => i16::MAX,
                1 => i16::MIN,
                2 => 0,
                _ => (rng.next() % 129) as i16 - 64,
            }
        }),
    }
}

/// A dst buffer with canary guards before and after the `needed`-byte window.
struct Guarded {
    buf: Vec<u8>,
    needed: usize,
}

impl Guarded {
    fn new(rng: &mut Rng, needed: usize) -> Self {
        let mut buf = vec![CANARY; GUARD + needed + GUARD];
        fill_pixels(rng, &mut buf[GUARD..GUARD + needed]);
        Self { buf, needed }
    }
    fn window(&mut self) -> &mut [u8] {
        // Tail guard stays inside the slice so out-of-window writes are detected.
        &mut self.buf[GUARD..]
    }
    fn check_canaries(&self, what: &str) {
        assert!(
            self.buf[..GUARD].iter().all(|&b| b == CANARY),
            "{what}: head canary clobbered"
        );
        assert!(
            self.buf[GUARD + self.needed..].iter().all(|&b| b == CANARY),
            "{what}: tail canary clobbered"
        );
    }
}

fn compare(a: &Guarded, b: &Guarded, what: &str, backend: SimdBackend, iter: usize) {
    b.check_canaries(what);
    assert_eq!(
        a.buf, b.buf,
        "{what}: SIMD ({backend:?}) != scalar at vector {iter}"
    );
}

fn simd_backends() -> Vec<SimdBackend> {
    simd::available_backends()
        .into_iter()
        .filter(|b| *b != SimdBackend::Scalar)
        .collect()
}

#[test]
fn checkasm_reports_backends() {
    let all = simd::available_backends();
    eprintln!(
        "checkasm backends: {all:?}, dispatch = {:?}",
        simd::detected_backend()
    );
    assert!(all.contains(&SimdBackend::Scalar));
}

#[test]
fn checkasm_idct_add_4x4() {
    for backend in simd_backends() {
        let mut rng = Rng::new(0x1001);
        for it in 0..VECTORS {
            let stride = rng.range(4, 41);
            let needed = 3 * stride + 4;
            let mut r = Guarded::new(&mut rng, needed);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed,
            };
            let mut cr = [0i16; 16];
            fill_coeffs(&mut rng, &mut cr);
            let mut cs = cr;
            scalar::idct_add_4x4(&mut r.window()[..needed], stride, &mut cr);
            simd::idct_add_4x4_with_backend(&mut s.window()[..needed], stride, &mut cs, backend);
            assert_eq!(cr, cs, "coeffs not zeroed identically");
            compare(&r, &s, "idct_add_4x4", backend, it);
        }
    }
}

#[test]
fn checkasm_idct_dc_add_4x4() {
    for backend in simd_backends() {
        let mut rng = Rng::new(0x1002);
        for it in 0..VECTORS {
            let stride = rng.range(4, 41);
            let needed = 3 * stride + 4;
            let mut r = Guarded::new(&mut rng, needed);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed,
            };
            let mut c = [0i16; 1];
            fill_coeffs(&mut rng, &mut c);
            let (mut dr, mut ds) = (c[0], c[0]);
            scalar::idct_dc_add_4x4(&mut r.window()[..needed], stride, &mut dr);
            simd::idct_dc_add_4x4_with_backend(&mut s.window()[..needed], stride, &mut ds, backend);
            assert_eq!((dr, ds), (0, 0));
            compare(&r, &s, "idct_dc_add_4x4", backend, it);
        }
    }
}

#[test]
fn checkasm_idct_add_8x8() {
    for backend in simd_backends() {
        let mut rng = Rng::new(0x1003);
        for it in 0..VECTORS {
            let stride = rng.range(8, 49);
            let needed = 7 * stride + 8;
            let mut r = Guarded::new(&mut rng, needed);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed,
            };
            let mut cr = [0i16; 64];
            fill_coeffs(&mut rng, &mut cr);
            let mut cs = cr;
            scalar::idct_add_8x8(&mut r.window()[..needed], stride, &mut cr);
            simd::idct_add_8x8_with_backend(&mut s.window()[..needed], stride, &mut cs, backend);
            assert_eq!(cr, cs, "coeffs not zeroed identically");
            compare(&r, &s, "idct_add_8x8", backend, it);
        }
    }
}

#[test]
fn checkasm_idct_dc_add_8x8() {
    for backend in simd_backends() {
        let mut rng = Rng::new(0x1004);
        for it in 0..VECTORS {
            let stride = rng.range(8, 49);
            let needed = 7 * stride + 8;
            let mut r = Guarded::new(&mut rng, needed);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed,
            };
            let mut c = [0i16; 1];
            fill_coeffs(&mut rng, &mut c);
            let (mut dr, mut ds) = (c[0], c[0]);
            scalar::idct_dc_add_8x8(&mut r.window()[..needed], stride, &mut dr);
            simd::idct_dc_add_8x8_with_backend(&mut s.window()[..needed], stride, &mut ds, backend);
            assert_eq!((dr, ds), (0, 0));
            compare(&r, &s, "idct_dc_add_8x8", backend, it);
        }
    }
}

#[test]
fn checkasm_put_and_avg_pixels() {
    for backend in simd_backends() {
        let mut rng = Rng::new(0x2001);
        for it in 0..VECTORS {
            let bw = [4usize, 8, 16][rng.range(0, 2)];
            let bh = rng.range(1, 16);
            let ds = bw + rng.range(0, 31);
            let as_ = bw + rng.range(0, 31);
            let bs = bw + rng.range(0, 31);
            let dn = (bh - 1) * ds + bw;
            let an = (bh - 1) * as_ + bw;
            let bn = (bh - 1) * bs + bw;
            let mut a = vec![0u8; an];
            let mut b = vec![0u8; bn];
            fill_pixels(&mut rng, &mut a);
            fill_pixels(&mut rng, &mut b);

            // avg_pixels (out-of-place)
            let mut r = Guarded::new(&mut rng, dn);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed: dn,
            };
            scalar::avg_pixels(&mut r.window()[..dn], ds, &a, as_, &b, bs, bw, bh);
            simd::avg_pixels_with_backend(
                &mut s.window()[..dn],
                ds,
                &a,
                as_,
                &b,
                bs,
                bw,
                bh,
                backend,
            );
            compare(&r, &s, "avg_pixels", backend, it);

            // avg_pixels_inplace
            let mut r = Guarded::new(&mut rng, dn);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed: dn,
            };
            scalar::avg_pixels_inplace(&mut r.window()[..dn], ds, &a, as_, bw, bh);
            simd::avg_pixels_inplace_with_backend(
                &mut s.window()[..dn],
                ds,
                &a,
                as_,
                bw,
                bh,
                backend,
            );
            compare(&r, &s, "avg_pixels_inplace", backend, it);

            // put_pixels (16 and 8)
            if bw == 16 || bw == 8 {
                let mut r = Guarded::new(&mut rng, dn);
                let mut s = Guarded {
                    buf: r.buf.clone(),
                    needed: dn,
                };
                scalar::put_pixels(&mut r.window()[..dn], ds, &a, as_, bw, bh);
                if bw == 16 {
                    simd::put_pixels_16x16_with_backend(
                        &mut s.window()[..dn],
                        ds,
                        &a,
                        as_,
                        bh,
                        backend,
                    );
                } else {
                    simd::put_pixels_8x8_with_backend(
                        &mut s.window()[..dn],
                        ds,
                        &a,
                        as_,
                        bh,
                        backend,
                    );
                }
                compare(&r, &s, "put_pixels", backend, it);
            }
        }
    }
}

#[test]
fn checkasm_public_dispatch_avg_wrappers() {
    let mut rng = Rng::new(0x2002);
    for it in 0..VECTORS {
        let bw = [8usize, 16][rng.range(0, 1)];
        let bh = rng.range(1, 16);
        let ds = bw + rng.range(0, 31);
        let as_ = bw + rng.range(0, 31);
        let bs = bw + rng.range(0, 31);
        let dn = (bh - 1) * ds + bw;
        let mut a = vec![0u8; (bh - 1) * as_ + bw];
        let mut b = vec![0u8; (bh - 1) * bs + bw];
        fill_pixels(&mut rng, &mut a);
        fill_pixels(&mut rng, &mut b);
        let mut r = Guarded::new(&mut rng, dn);
        let mut s = Guarded {
            buf: r.buf.clone(),
            needed: dn,
        };
        scalar::avg_pixels(&mut r.window()[..dn], ds, &a, as_, &b, bs, bw, bh);
        if bw == 16 {
            simd::avg_pixels_16x16(&mut s.window()[..dn], ds, &a, as_, &b, bs, bh);
        } else {
            simd::avg_pixels_8x8(&mut s.window()[..dn], ds, &a, as_, &b, bs, bh);
        }
        compare(
            &r,
            &s,
            "avg_pixels_{16x16,8x8}",
            simd::detected_backend(),
            it,
        );
    }
}

#[test]
fn checkasm_chroma_mc() {
    for backend in simd_backends() {
        let mut rng = Rng::new(0x3001);
        for it in 0..VECTORS {
            let bw = [4usize, 8, 16][rng.range(0, 2)];
            let bh = rng.range(1, 16);
            let ds = bw + rng.range(0, 31);
            let rs = bw + 1 + rng.range(0, 31);
            let avg = rng.next() & 1 == 1;
            let dn = (bh - 1) * ds + bw;

            // Bilinear: random 1/8-pel fractions (a+b+c+d == 64).
            let dx = rng.range(0, 7) as i32;
            let dy = rng.range(0, 7) as i32;
            let (a, b, c, d) = ((8 - dx) * (8 - dy), dx * (8 - dy), (8 - dx) * dy, dx * dy);
            let rn = bh * rs + bw + 1;
            let mut refp = vec![0u8; rn];
            fill_pixels(&mut rng, &mut refp);
            let mut r = Guarded::new(&mut rng, dn);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed: dn,
            };
            scalar::chroma_mc_bilinear(
                &mut r.window()[..dn],
                ds,
                &refp,
                rs,
                bw,
                bh,
                a,
                b,
                c,
                d,
                avg,
            );
            simd::chroma_mc_bilinear_with_backend(
                &mut s.window()[..dn],
                ds,
                &refp,
                rs,
                bw,
                bh,
                a,
                b,
                c,
                d,
                avg,
                backend,
            );
            compare(&r, &s, "chroma_mc_bilinear", backend, it);

            // 2-tap: horizontal (step 1) or vertical (step = stride).
            let e = (rng.range(1, 7) * 8) as i32;
            let a2 = 64 - e;
            let step = if rng.next() & 1 == 0 { 1 } else { rs };
            let rn2 = (bh - 1) * rs + step + bw;
            let mut refp2 = vec![0u8; rn2];
            fill_pixels(&mut rng, &mut refp2);
            let mut r = Guarded::new(&mut rng, dn);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed: dn,
            };
            scalar::chroma_mc_2tap(
                &mut r.window()[..dn],
                ds,
                &refp2,
                rs,
                bw,
                bh,
                a2,
                e,
                step,
                avg,
            );
            simd::chroma_mc_2tap_with_backend(
                &mut s.window()[..dn],
                ds,
                &refp2,
                rs,
                bw,
                bh,
                a2,
                e,
                step,
                avg,
                backend,
            );
            compare(&r, &s, "chroma_mc_2tap", backend, it);
        }
    }
}

#[test]
fn checkasm_luma_6tap() {
    for backend in simd_backends() {
        let mut rng = Rng::new(0x4001);
        for it in 0..VECTORS {
            let bw = [4usize, 8, 16][rng.range(0, 2)];
            let bh = rng.range(1, 16);
            let ds = bw + rng.range(0, 31);
            let ss = bw + 5 + rng.range(0, 31);
            let dn = (bh - 1) * ds + bw;

            // Horizontal: needs (bh-1)*ss + bw + 5.
            let hn = (bh - 1) * ss + bw + 5;
            let mut src_h = vec![0u8; hn];
            fill_pixels(&mut rng, &mut src_h);
            let mut r = Guarded::new(&mut rng, dn);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed: dn,
            };
            scalar::luma_6tap_h(&mut r.window()[..dn], ds, &src_h, ss, bw, bh);
            simd::luma_6tap_h_with_backend(&mut s.window()[..dn], ds, &src_h, ss, bw, bh, backend);
            compare(&r, &s, "luma_6tap_h", backend, it);

            // Vertical: needs (bh+4)*ss + bw.
            let vn = (bh + 4) * ss + bw;
            let mut src_v = vec![0u8; vn];
            fill_pixels(&mut rng, &mut src_v);
            let mut r = Guarded::new(&mut rng, dn);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed: dn,
            };
            scalar::luma_6tap_v(&mut r.window()[..dn], ds, &src_v, ss, bw, bh);
            simd::luma_6tap_v_with_backend(&mut s.window()[..dn], ds, &src_v, ss, bw, bh, backend);
            compare(&r, &s, "luma_6tap_v", backend, it);

            // 2D: needs (bh+4)*ss + bw + 5.
            let hvn = (bh + 4) * ss + bw + 5;
            let mut src_hv = vec![0u8; hvn];
            fill_pixels(&mut rng, &mut src_hv);
            let mut r = Guarded::new(&mut rng, dn);
            let mut s = Guarded {
                buf: r.buf.clone(),
                needed: dn,
            };
            let mut tr = vec![0i32; (bh + 5) * bw];
            let mut ts = vec![0i32; (bh + 5) * bw];
            scalar::luma_6tap_hv(&mut r.window()[..dn], ds, &src_hv, ss, bw, bh, &mut tr);
            simd::luma_6tap_hv_with_backend(
                &mut s.window()[..dn],
                ds,
                &src_hv,
                ss,
                bw,
                bh,
                &mut ts,
                backend,
            );
            compare(&r, &s, "luma_6tap_hv", backend, it);
        }
    }
}

#[test]
fn checkasm_public_dispatch_idct() {
    // The auto-dispatched entry points (what the decoder actually calls).
    let mut rng = Rng::new(0x5001);
    for it in 0..VECTORS {
        let stride = rng.range(8, 49);
        let needed = 7 * stride + 8;
        let mut r = Guarded::new(&mut rng, needed);
        let mut s = Guarded {
            buf: r.buf.clone(),
            needed,
        };
        let mut c8 = [0i16; 64];
        fill_coeffs(&mut rng, &mut c8);
        let mut c8s = c8;
        scalar::idct_add_8x8(&mut r.window()[..needed], stride, &mut c8);
        simd::idct_add_8x8(&mut s.window()[..needed], stride, &mut c8s);
        compare(
            &r,
            &s,
            "idct_add_8x8 dispatch",
            simd::detected_backend(),
            it,
        );

        let mut c4 = [0i16; 16];
        fill_coeffs(&mut rng, &mut c4);
        let mut c4s = c4;
        scalar::idct_add_4x4(&mut r.window()[..needed], stride, &mut c4);
        simd::idct_add_4x4(&mut s.window()[..needed], stride, &mut c4s);
        compare(
            &r,
            &s,
            "idct_add_4x4 dispatch",
            simd::detected_backend(),
            it,
        );
    }
}

#[test]
fn dispatched_square_average_matches_scalar_with_padding_and_overlap() {
    for size in [8usize, 16] {
        for stride in [1, size, size + 3, 32] {
            for seed in 0..128usize {
                let len = (size - 1) * stride + size + 13;
                let src: Vec<u8> = (0..len)
                    .map(|i| (i.wrapping_mul(71).wrapping_add(seed * 37)) as u8)
                    .collect();
                let initial: Vec<u8> = (0..len)
                    .map(|i| (i.wrapping_mul(113).wrapping_add(seed * 23)) as u8)
                    .collect();
                let mut expected = initial.clone();
                let mut actual = initial;
                scalar::avg_pixels_inplace(&mut expected, stride, &src, stride, size, size);
                simd::avg_pixels_inplace(&mut actual, stride, &src, stride, size, size);
                assert_eq!(actual, expected, "size={size} stride={stride} seed={seed}");
            }
        }
    }
}
