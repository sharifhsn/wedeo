//! Randomized bit-exactness verification (`checkasm`) of every pure-Rust
//! `core::arch` SIMD kernel in `wedeo-codec-vp9` against the scalar reference.

use wedeo_codec_vp9::data::VP9_SUBPEL_FILTERS;
use wedeo_codec_vp9::simd::{
    SimdBackend, active_backend, available_backends, avg_block_with_backend,
    filter_h_8tap_with_backend, filter_hv_8tap_with_backend, filter_v_8tap_with_backend,
};

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill_bytes(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b = (self.next_u64() & 0xFF) as u8;
        }
    }
}

const BLOCK_DIMS: &[(usize, usize)] = &[
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 32),
    (64, 64),
];

#[test]
fn checkasm_vp9_reports_backends() {
    let backends = available_backends();
    assert!(backends.contains(&SimdBackend::Scalar));
    assert!(backends.contains(&active_backend()));
}

#[test]
fn checkasm_vp9_filter_h_8tap_10k_vectors() {
    let mut rng = SplitMix64::new(0x5650_395F_4838);
    let backends = available_backends();
    let mut src = vec![0u8; 80 * 80];
    let mut ref_dst = vec![0u8; 80 * 80];
    let mut simd_dst = vec![0u8; 80 * 80];

    // 10,000 randomized tests covering all 4 filter types, all 16 phases, and all VP9 block sizes.
    for iter in 0..10_000usize {
        let (w, h) = BLOCK_DIMS[iter % BLOCK_DIMS.len()];
        let fi = (rng.next_u64() as usize) % 4;
        let phase = (rng.next_u64() as usize) % 16;
        let filter = &VP9_SUBPEL_FILTERS[fi][phase];
        let src_stride = w + 7 + ((rng.next_u64() as usize) % 9);
        let dst_stride = w + ((rng.next_u64() as usize) % 9);

        // Edge-case extreme values on first 16 iterations, random bytes thereafter.
        let src_len = h * src_stride;
        if iter < 4 {
            src[..src_len].fill(if iter % 2 == 0 { 0 } else { 255 });
        } else {
            rng.fill_bytes(&mut src[..src_len]);
        }

        let dst_len = h * dst_stride;
        ref_dst[..dst_len].fill(0xA5);
        filter_h_8tap_with_backend(
            &mut ref_dst[..dst_len],
            dst_stride,
            &src[..src_len],
            src_stride,
            w,
            h,
            filter,
            SimdBackend::Scalar,
        );

        for &backend in &backends {
            if backend == SimdBackend::Scalar {
                continue;
            }
            simd_dst[..dst_len].fill(0xA5);
            filter_h_8tap_with_backend(
                &mut simd_dst[..dst_len],
                dst_stride,
                &src[..src_len],
                src_stride,
                w,
                h,
                filter,
                backend,
            );
            assert_eq!(
                &simd_dst[..dst_len],
                &ref_dst[..dst_len],
                "filter_h_8tap mismatch for {backend:?} at iter={iter} {w}x{h} fi={fi} phase={phase}"
            );
        }
    }
}

#[test]
fn checkasm_vp9_filter_v_8tap_10k_vectors() {
    let mut rng = SplitMix64::new(0x5650_395F_5638);
    let backends = available_backends();
    let mut src = vec![0u8; 80 * 80];
    let mut ref_dst = vec![0u8; 80 * 80];
    let mut simd_dst = vec![0u8; 80 * 80];

    for iter in 0..10_000usize {
        let (w, h) = BLOCK_DIMS[iter % BLOCK_DIMS.len()];
        let fi = (rng.next_u64() as usize) % 4;
        let phase = (rng.next_u64() as usize) % 16;
        let filter = &VP9_SUBPEL_FILTERS[fi][phase];
        let src_stride = w + ((rng.next_u64() as usize) % 9);
        let dst_stride = w + ((rng.next_u64() as usize) % 9);
        let src_len = (h + 7) * src_stride;
        let dst_len = h * dst_stride;

        rng.fill_bytes(&mut src[..src_len]);
        ref_dst[..dst_len].fill(0x5A);
        filter_v_8tap_with_backend(
            &mut ref_dst[..dst_len],
            dst_stride,
            &src[..src_len],
            src_stride,
            w,
            h,
            filter,
            SimdBackend::Scalar,
        );

        for &backend in &backends {
            if backend == SimdBackend::Scalar {
                continue;
            }
            simd_dst[..dst_len].fill(0x5A);
            filter_v_8tap_with_backend(
                &mut simd_dst[..dst_len],
                dst_stride,
                &src[..src_len],
                src_stride,
                w,
                h,
                filter,
                backend,
            );
            assert_eq!(
                &simd_dst[..dst_len],
                &ref_dst[..dst_len],
                "filter_v_8tap mismatch for {backend:?} at iter={iter} {w}x{h} fi={fi} phase={phase}"
            );
        }
    }
}

#[test]
fn checkasm_vp9_filter_hv_and_avg_block() {
    let mut rng = SplitMix64::new(0x5650_395F_4856);
    let backends = available_backends();
    let mut src = vec![0u8; 80 * 80];
    let mut ref_dst = vec![0u8; 80 * 80];
    let mut simd_dst = vec![0u8; 80 * 80];
    let mut tmp_ref = vec![0u8; 64 * 71];
    let mut tmp_simd = vec![0u8; 64 * 71];

    for iter in 0..2_000usize {
        let (w, h) = BLOCK_DIMS[iter % BLOCK_DIMS.len()];
        let fi = (rng.next_u64() as usize) % 4;
        let px = (rng.next_u64() as usize) % 16;
        let py = (rng.next_u64() as usize) % 16;
        let hf = &VP9_SUBPEL_FILTERS[fi][px];
        let vf = &VP9_SUBPEL_FILTERS[fi][py];
        let src_stride = w + 7 + ((rng.next_u64() as usize) % 7);
        let dst_stride = w + ((rng.next_u64() as usize) % 7);
        let src_len = (h + 7) * src_stride;
        let dst_len = h * dst_stride;

        rng.fill_bytes(&mut src[..src_len]);
        ref_dst[..dst_len].fill(0x33);
        filter_hv_8tap_with_backend(
            &mut ref_dst[..dst_len],
            dst_stride,
            &src[..src_len],
            src_stride,
            w,
            h,
            hf,
            vf,
            &mut tmp_ref,
            SimdBackend::Scalar,
        );

        for &backend in &backends {
            if backend == SimdBackend::Scalar {
                continue;
            }
            simd_dst[..dst_len].fill(0x33);
            filter_hv_8tap_with_backend(
                &mut simd_dst[..dst_len],
                dst_stride,
                &src[..src_len],
                src_stride,
                w,
                h,
                hf,
                vf,
                &mut tmp_simd,
                backend,
            );
            assert_eq!(
                &simd_dst[..dst_len],
                &ref_dst[..dst_len],
                "filter_hv_8tap mismatch for {backend:?} at iter={iter} {w}x{h}"
            );
        }

        // avg_block check
        rng.fill_bytes(&mut ref_dst[..dst_len]);
        simd_dst[..dst_len].copy_from_slice(&ref_dst[..dst_len]);
        let mut ref_avg = ref_dst[..dst_len].to_vec();
        avg_block_with_backend(
            &mut ref_avg,
            dst_stride,
            &src[..dst_len],
            dst_stride,
            w,
            h,
            SimdBackend::Scalar,
        );
        for &backend in &backends {
            if backend == SimdBackend::Scalar {
                continue;
            }
            let mut simd_avg = simd_dst[..dst_len].to_vec();
            avg_block_with_backend(
                &mut simd_avg,
                dst_stride,
                &src[..dst_len],
                dst_stride,
                w,
                h,
                backend,
            );
            assert_eq!(
                simd_avg, ref_avg,
                "avg_block mismatch for {backend:?} at iter={iter} {w}x{h}"
            );
        }
    }
}
