//! Pure-Rust `core::arch` SIMD hot kernels for VP9 motion compensation and
//! block averaging.
//!
//! All `unsafe` in `wedeo-codec-vp9` is strictly isolated in this module and
//! verified bit-exact against the scalar reference across 10,000+ randomized
//! vectors in `tests/checkasm_vp9.rs`.

#![allow(unsafe_code)]

/// Executable SIMD backend selected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Detect the fastest SIMD backend available on the current CPU.
#[inline]
pub fn active_backend() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return SimdBackend::Avx2;
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            return SimdBackend::Sse2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return SimdBackend::Neon;
        }
    }
    SimdBackend::Scalar
}

/// Return all backends executable on the current CPU (used by `checkasm_vp9`).
pub fn available_backends() -> Vec<SimdBackend> {
    #[allow(unused_mut)]
    let mut out = vec![SimdBackend::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            out.push(SimdBackend::Sse2);
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push(SimdBackend::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            out.push(SimdBackend::Neon);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Public runtime-dispatched entry points
// ---------------------------------------------------------------------------

/// 8-tap horizontal subpel filter (`((sum + 64) >> 7).clamp(0, 255)`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn filter_h_8tap(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    filter: &[i16; 8],
) {
    filter_h_8tap_with_backend(
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        filter,
        active_backend(),
    );
}

/// 8-tap vertical subpel filter (`((sum + 64) >> 7).clamp(0, 255)`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn filter_v_8tap(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    filter: &[i16; 8],
) {
    filter_v_8tap_with_backend(
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        filter,
        active_backend(),
    );
}

/// 2D separable 8-tap horizontal + vertical subpel filter using caller scratch (`>= 64 * (h + 7)`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn filter_hv_8tap(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    h_filter: &[i16; 8],
    v_filter: &[i16; 8],
    tmp: &mut [u8],
) {
    filter_hv_8tap_with_backend(
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        h_filter,
        v_filter,
        tmp,
        active_backend(),
    );
}

/// Compound prediction averaging: `dst[i] = (dst[i] + src[i] + 1) >> 1`.
#[inline]
pub fn avg_block(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    avg_block_with_backend(dst, dst_stride, src, src_stride, w, h, active_backend());
}

// ---------------------------------------------------------------------------
// Explicit-backend dispatch (used by checkasm)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn filter_h_8tap_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    filter: &[i16; 8],
    backend: SimdBackend,
) {
    if w == 0 || h == 0 {
        return;
    }
    assert!((h - 1) * dst_stride + w <= dst.len());
    assert!((h - 1) * src_stride + w + 7 <= src.len());

    match backend {
        SimdBackend::Scalar => {
            scalar::filter_h_8tap(dst, dst_stride, src, src_stride, w, h, filter)
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Sse2 => {
            // SAFETY: Caller / active_backend verified SSE2 is supported and bounds are asserted above.
            unsafe { x86_sse2::filter_h_8tap(dst, dst_stride, src, src_stride, w, h, filter) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => {
            // SAFETY: Caller / active_backend verified AVX2 is supported and bounds are asserted above.
            unsafe { x86_avx2::filter_h_8tap(dst, dst_stride, src, src_stride, w, h, filter) }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            // SAFETY: Caller / active_backend verified NEON is supported and bounds are asserted above.
            unsafe { arm_neon::filter_h_8tap(dst, dst_stride, src, src_stride, w, h, filter) }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn filter_v_8tap_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    filter: &[i16; 8],
    backend: SimdBackend,
) {
    if w == 0 || h == 0 {
        return;
    }
    assert!((h - 1) * dst_stride + w <= dst.len());
    assert!((h + 6) * src_stride + w <= src.len());

    match backend {
        SimdBackend::Scalar => {
            scalar::filter_v_8tap(dst, dst_stride, src, src_stride, w, h, filter)
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Sse2 => {
            // SAFETY: SSE2 support and slice bounds verified above.
            unsafe { x86_sse2::filter_v_8tap(dst, dst_stride, src, src_stride, w, h, filter) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => {
            // SAFETY: AVX2 support and slice bounds verified above.
            unsafe { x86_avx2::filter_v_8tap(dst, dst_stride, src, src_stride, w, h, filter) }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            // SAFETY: NEON support and slice bounds verified above.
            unsafe { arm_neon::filter_v_8tap(dst, dst_stride, src, src_stride, w, h, filter) }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn filter_hv_8tap_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    h_filter: &[i16; 8],
    v_filter: &[i16; 8],
    tmp: &mut [u8],
    backend: SimdBackend,
) {
    if w == 0 || h == 0 {
        return;
    }
    const TMP_STRIDE: usize = 64;
    debug_assert!(w <= TMP_STRIDE);
    assert!((h + 7) * TMP_STRIDE <= tmp.len());
    filter_h_8tap_with_backend(
        tmp,
        TMP_STRIDE,
        src,
        src_stride,
        w,
        h + 7,
        h_filter,
        backend,
    );
    filter_v_8tap_with_backend(dst, dst_stride, tmp, TMP_STRIDE, w, h, v_filter, backend);
}

pub fn avg_block_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    backend: SimdBackend,
) {
    if w == 0 || h == 0 {
        return;
    }
    assert!((h - 1) * dst_stride + w <= dst.len());
    assert!((h - 1) * src_stride + w <= src.len());

    match backend {
        SimdBackend::Scalar => scalar::avg_block(dst, dst_stride, src, src_stride, w, h),
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Sse2 => {
            // SAFETY: SSE2 support and slice bounds verified above.
            unsafe { x86_sse2::avg_block(dst, dst_stride, src, src_stride, w, h) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 => {
            // SAFETY: AVX2 support and slice bounds verified above.
            unsafe { x86_avx2::avg_block(dst, dst_stride, src, src_stride, w, h) }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            // SAFETY: NEON support and slice bounds verified above.
            unsafe { arm_neon::avg_block(dst, dst_stride, src, src_stride, w, h) }
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar reference kernels
// ---------------------------------------------------------------------------

pub mod scalar {
    #[allow(clippy::too_many_arguments)]
    pub fn filter_h_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        filter: &[i16; 8],
    ) {
        for y in 0..h {
            let src_row = &src[y * src_stride..];
            let dst_row = &mut dst[y * dst_stride..];
            for x in 0..w {
                let val = filter[0] as i32 * src_row[x] as i32
                    + filter[1] as i32 * src_row[x + 1] as i32
                    + filter[2] as i32 * src_row[x + 2] as i32
                    + filter[3] as i32 * src_row[x + 3] as i32
                    + filter[4] as i32 * src_row[x + 4] as i32
                    + filter[5] as i32 * src_row[x + 5] as i32
                    + filter[6] as i32 * src_row[x + 6] as i32
                    + filter[7] as i32 * src_row[x + 7] as i32;
                dst_row[x] = ((val + 64) >> 7).clamp(0, 255) as u8;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn filter_v_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        filter: &[i16; 8],
    ) {
        for y in 0..h {
            let dst_row = &mut dst[y * dst_stride..];
            for x in 0..w {
                let val = filter[0] as i32 * src[y * src_stride + x] as i32
                    + filter[1] as i32 * src[(y + 1) * src_stride + x] as i32
                    + filter[2] as i32 * src[(y + 2) * src_stride + x] as i32
                    + filter[3] as i32 * src[(y + 3) * src_stride + x] as i32
                    + filter[4] as i32 * src[(y + 4) * src_stride + x] as i32
                    + filter[5] as i32 * src[(y + 5) * src_stride + x] as i32
                    + filter[6] as i32 * src[(y + 6) * src_stride + x] as i32
                    + filter[7] as i32 * src[(y + 7) * src_stride + x] as i32;
                dst_row[x] = ((val + 64) >> 7).clamp(0, 255) as u8;
            }
        }
    }

    pub fn avg_block(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
    ) {
        for y in 0..h {
            let drow = &mut dst[y * dst_stride..y * dst_stride + w];
            let srow = &src[y * src_stride..y * src_stride + w];
            for x in 0..w {
                drow[x] = ((drow[x] as u16 + srow[x] as u16 + 1) >> 1) as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// x86_64 SSE2 kernels (exact 32-bit _mm_madd_epi16 accumulation)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86_sse2 {
    use core::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load_4_u8_to_i16(ptr: *const u8) -> __m128i {
        // Load 4 bytes as u32, zero-extend u8 -> i16 in low 4 lanes.
        let v = unsafe { (ptr as *const u32).read_unaligned() } as i32;
        _mm_unpacklo_epi8(_mm_cvtsi32_si128(v), _mm_setzero_si128())
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load_8_u8_to_i16(ptr: *const u8) -> __m128i {
        let v = unsafe { (ptr as *const u64).read_unaligned() } as i64;
        _mm_unpacklo_epi8(_mm_cvtsi64_si128(v), _mm_setzero_si128())
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn pair_f(a: i16, b: i16) -> __m128i {
        let packed = (a as u16 as u32) | ((b as u16 as u32) << 16);
        _mm_set1_epi32(packed as i32)
    }

    #[target_feature(enable = "sse2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn filter_h_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        filter: &[i16; 8],
    ) {
        let f01 = unsafe { pair_f(filter[0], filter[1]) };
        let f23 = unsafe { pair_f(filter[2], filter[3]) };
        let f45 = unsafe { pair_f(filter[4], filter[5]) };
        let f67 = unsafe { pair_f(filter[6], filter[7]) };
        let round = _mm_set1_epi32(64);

        for y in 0..h {
            let srow = unsafe { src.as_ptr().add(y * src_stride) };
            let drow = unsafe { dst.as_mut_ptr().add(y * dst_stride) };
            let mut x = 0usize;
            while x + 4 <= w {
                let s0 = unsafe { load_4_u8_to_i16(srow.add(x)) };
                let s1 = unsafe { load_4_u8_to_i16(srow.add(x + 1)) };
                let s2 = unsafe { load_4_u8_to_i16(srow.add(x + 2)) };
                let s3 = unsafe { load_4_u8_to_i16(srow.add(x + 3)) };
                let s4 = unsafe { load_4_u8_to_i16(srow.add(x + 4)) };
                let s5 = unsafe { load_4_u8_to_i16(srow.add(x + 5)) };
                let s6 = unsafe { load_4_u8_to_i16(srow.add(x + 6)) };
                let s7 = unsafe { load_4_u8_to_i16(srow.add(x + 7)) };

                let p01 = _mm_madd_epi16(_mm_unpacklo_epi16(s0, s1), f01);
                let p23 = _mm_madd_epi16(_mm_unpacklo_epi16(s2, s3), f23);
                let p45 = _mm_madd_epi16(_mm_unpacklo_epi16(s4, s5), f45);
                let p67 = _mm_madd_epi16(_mm_unpacklo_epi16(s6, s7), f67);

                let sum = _mm_add_epi32(_mm_add_epi32(p01, p23), _mm_add_epi32(p45, p67));
                let shifted = _mm_srai_epi32(_mm_add_epi32(sum, round), 7);
                let packed16 = _mm_packs_epi32(shifted, shifted);
                let packed8 = _mm_packus_epi16(packed16, packed16);
                let out32 = _mm_cvtsi128_si32(packed8) as u32;
                unsafe { (drow.add(x) as *mut u32).write_unaligned(out32) };
                x += 4;
            }
            while x < w {
                let mut acc = 0i32;
                for (t, &ft) in filter.iter().enumerate() {
                    acc += ft as i32 * unsafe { *srow.add(x + t) } as i32;
                }
                unsafe { *drow.add(x) = ((acc + 64) >> 7).clamp(0, 255) as u8 };
                x += 1;
            }
        }
    }

    #[target_feature(enable = "sse2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn filter_v_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        filter: &[i16; 8],
    ) {
        let f01 = unsafe { pair_f(filter[0], filter[1]) };
        let f23 = unsafe { pair_f(filter[2], filter[3]) };
        let f45 = unsafe { pair_f(filter[4], filter[5]) };
        let f67 = unsafe { pair_f(filter[6], filter[7]) };
        let round = _mm_set1_epi32(64);

        for y in 0..h {
            let drow = unsafe { dst.as_mut_ptr().add(y * dst_stride) };
            let r0 = unsafe { src.as_ptr().add(y * src_stride) };
            let r1 = unsafe { src.as_ptr().add((y + 1) * src_stride) };
            let r2 = unsafe { src.as_ptr().add((y + 2) * src_stride) };
            let r3 = unsafe { src.as_ptr().add((y + 3) * src_stride) };
            let r4 = unsafe { src.as_ptr().add((y + 4) * src_stride) };
            let r5 = unsafe { src.as_ptr().add((y + 5) * src_stride) };
            let r6 = unsafe { src.as_ptr().add((y + 6) * src_stride) };
            let r7 = unsafe { src.as_ptr().add((y + 7) * src_stride) };

            let mut x = 0usize;
            while x + 8 <= w {
                let a0 = unsafe { load_8_u8_to_i16(r0.add(x)) };
                let a1 = unsafe { load_8_u8_to_i16(r1.add(x)) };
                let a2 = unsafe { load_8_u8_to_i16(r2.add(x)) };
                let a3 = unsafe { load_8_u8_to_i16(r3.add(x)) };
                let a4 = unsafe { load_8_u8_to_i16(r4.add(x)) };
                let a5 = unsafe { load_8_u8_to_i16(r5.add(x)) };
                let a6 = unsafe { load_8_u8_to_i16(r6.add(x)) };
                let a7 = unsafe { load_8_u8_to_i16(r7.add(x)) };

                let lo = _mm_add_epi32(
                    _mm_add_epi32(
                        _mm_madd_epi16(_mm_unpacklo_epi16(a0, a1), f01),
                        _mm_madd_epi16(_mm_unpacklo_epi16(a2, a3), f23),
                    ),
                    _mm_add_epi32(
                        _mm_madd_epi16(_mm_unpacklo_epi16(a4, a5), f45),
                        _mm_madd_epi16(_mm_unpacklo_epi16(a6, a7), f67),
                    ),
                );
                let hi = _mm_add_epi32(
                    _mm_add_epi32(
                        _mm_madd_epi16(_mm_unpackhi_epi16(a0, a1), f01),
                        _mm_madd_epi16(_mm_unpackhi_epi16(a2, a3), f23),
                    ),
                    _mm_add_epi32(
                        _mm_madd_epi16(_mm_unpackhi_epi16(a4, a5), f45),
                        _mm_madd_epi16(_mm_unpackhi_epi16(a6, a7), f67),
                    ),
                );
                let lo_s = _mm_srai_epi32(_mm_add_epi32(lo, round), 7);
                let hi_s = _mm_srai_epi32(_mm_add_epi32(hi, round), 7);
                let packed16 = _mm_packs_epi32(lo_s, hi_s);
                let packed8 = _mm_packus_epi16(packed16, packed16);
                let out64 = _mm_cvtsi128_si64(packed8) as u64;
                unsafe { (drow.add(x) as *mut u64).write_unaligned(out64) };
                x += 8;
            }
            while x + 4 <= w {
                let a0 = unsafe { load_4_u8_to_i16(r0.add(x)) };
                let a1 = unsafe { load_4_u8_to_i16(r1.add(x)) };
                let a2 = unsafe { load_4_u8_to_i16(r2.add(x)) };
                let a3 = unsafe { load_4_u8_to_i16(r3.add(x)) };
                let a4 = unsafe { load_4_u8_to_i16(r4.add(x)) };
                let a5 = unsafe { load_4_u8_to_i16(r5.add(x)) };
                let a6 = unsafe { load_4_u8_to_i16(r6.add(x)) };
                let a7 = unsafe { load_4_u8_to_i16(r7.add(x)) };

                let lo = _mm_add_epi32(
                    _mm_add_epi32(
                        _mm_madd_epi16(_mm_unpacklo_epi16(a0, a1), f01),
                        _mm_madd_epi16(_mm_unpacklo_epi16(a2, a3), f23),
                    ),
                    _mm_add_epi32(
                        _mm_madd_epi16(_mm_unpacklo_epi16(a4, a5), f45),
                        _mm_madd_epi16(_mm_unpacklo_epi16(a6, a7), f67),
                    ),
                );
                let lo_s = _mm_srai_epi32(_mm_add_epi32(lo, round), 7);
                let packed16 = _mm_packs_epi32(lo_s, lo_s);
                let packed8 = _mm_packus_epi16(packed16, packed16);
                let out32 = _mm_cvtsi128_si32(packed8) as u32;
                unsafe { (drow.add(x) as *mut u32).write_unaligned(out32) };
                x += 4;
            }
            while x < w {
                let acc = filter[0] as i32 * unsafe { *r0.add(x) } as i32
                    + filter[1] as i32 * unsafe { *r1.add(x) } as i32
                    + filter[2] as i32 * unsafe { *r2.add(x) } as i32
                    + filter[3] as i32 * unsafe { *r3.add(x) } as i32
                    + filter[4] as i32 * unsafe { *r4.add(x) } as i32
                    + filter[5] as i32 * unsafe { *r5.add(x) } as i32
                    + filter[6] as i32 * unsafe { *r6.add(x) } as i32
                    + filter[7] as i32 * unsafe { *r7.add(x) } as i32;
                unsafe { *drow.add(x) = ((acc + 64) >> 7).clamp(0, 255) as u8 };
                x += 1;
            }
        }
    }

    #[target_feature(enable = "sse2")]
    pub unsafe fn avg_block(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
    ) {
        for y in 0..h {
            let drow = unsafe { dst.as_mut_ptr().add(y * dst_stride) };
            let srow = unsafe { src.as_ptr().add(y * src_stride) };
            let mut x = 0usize;
            while x + 16 <= w {
                let vd = unsafe { _mm_loadu_si128(drow.add(x) as *const __m128i) };
                let vs = unsafe { _mm_loadu_si128(srow.add(x) as *const __m128i) };
                let avg = _mm_avg_epu8(vd, vs);
                unsafe { _mm_storeu_si128(drow.add(x) as *mut __m128i, avg) };
                x += 16;
            }
            while x + 8 <= w {
                let vd = _mm_cvtsi64_si128(unsafe { (drow.add(x) as *const i64).read_unaligned() });
                let vs = _mm_cvtsi64_si128(unsafe { (srow.add(x) as *const i64).read_unaligned() });
                let avg = _mm_avg_epu8(vd, vs);
                unsafe { (drow.add(x) as *mut i64).write_unaligned(_mm_cvtsi128_si64(avg)) };
                x += 8;
            }
            while x < w {
                unsafe {
                    let d = *drow.add(x) as u16;
                    let s = *srow.add(x) as u16;
                    *drow.add(x) = ((d + s + 1) >> 1) as u8;
                }
                x += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 kernels (16 pixels/iter for v-filter, 8-16 for h-filter, 32 for avg)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86_avx2 {
    use core::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn pair_f256(a: i16, b: i16) -> __m256i {
        let packed = (a as u16 as u32) | ((b as u16 as u32) << 16);
        _mm256_set1_epi32(packed as i32)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_16_u8_to_i16(ptr: *const u8) -> __m256i {
        let v128 = unsafe { _mm_loadu_si128(ptr as *const __m128i) };
        _mm256_cvtepu8_epi16(v128)
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn filter_h_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        filter: &[i16; 8],
    ) {
        if w < 16 {
            unsafe {
                super::x86_sse2::filter_h_8tap(dst, dst_stride, src, src_stride, w, h, filter);
            }
            return;
        }

        let f01 = unsafe { pair_f256(filter[0], filter[1]) };
        let f23 = unsafe { pair_f256(filter[2], filter[3]) };
        let f45 = unsafe { pair_f256(filter[4], filter[5]) };
        let f67 = unsafe { pair_f256(filter[6], filter[7]) };
        let round = _mm256_set1_epi32(64);

        for y in 0..h {
            let srow = unsafe { src.as_ptr().add(y * src_stride) };
            let drow = unsafe { dst.as_mut_ptr().add(y * dst_stride) };
            let mut x = 0usize;
            while x + 16 <= w {
                let s0 = unsafe { load_16_u8_to_i16(srow.add(x)) };
                let s1 = unsafe { load_16_u8_to_i16(srow.add(x + 1)) };
                let s2 = unsafe { load_16_u8_to_i16(srow.add(x + 2)) };
                let s3 = unsafe { load_16_u8_to_i16(srow.add(x + 3)) };
                let s4 = unsafe { load_16_u8_to_i16(srow.add(x + 4)) };
                let s5 = unsafe { load_16_u8_to_i16(srow.add(x + 5)) };
                let s6 = unsafe { load_16_u8_to_i16(srow.add(x + 6)) };
                let s7 = unsafe { load_16_u8_to_i16(srow.add(x + 7)) };

                let lo = _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_unpacklo_epi16(s0, s1), f01),
                        _mm256_madd_epi16(_mm256_unpacklo_epi16(s2, s3), f23),
                    ),
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_unpacklo_epi16(s4, s5), f45),
                        _mm256_madd_epi16(_mm256_unpacklo_epi16(s6, s7), f67),
                    ),
                );
                let hi = _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_unpackhi_epi16(s0, s1), f01),
                        _mm256_madd_epi16(_mm256_unpackhi_epi16(s2, s3), f23),
                    ),
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_unpackhi_epi16(s4, s5), f45),
                        _mm256_madd_epi16(_mm256_unpackhi_epi16(s6, s7), f67),
                    ),
                );
                let lo_s = _mm256_srai_epi32(_mm256_add_epi32(lo, round), 7);
                let hi_s = _mm256_srai_epi32(_mm256_add_epi32(hi, round), 7);
                // Within each 128-bit lane: packs_epi32(lo_s, hi_s) restores linear order [0..7] & [8..15]!
                let p16 = _mm256_packs_epi32(lo_s, hi_s);
                let p16_lo = _mm256_castsi256_si128(p16);
                let p16_hi = _mm256_extracti128_si256::<1>(p16);
                let p8 = _mm_packus_epi16(p16_lo, p16_hi);
                unsafe { _mm_storeu_si128(drow.add(x) as *mut __m128i, p8) };
                x += 16;
            }
            if x < w {
                let rem = w - x;
                unsafe {
                    super::x86_sse2::filter_h_8tap(
                        &mut dst[y * dst_stride + x..],
                        dst_stride,
                        &src[y * src_stride + x..],
                        src_stride,
                        rem,
                        1,
                        filter,
                    );
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn filter_v_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        filter: &[i16; 8],
    ) {
        if w < 16 {
            unsafe {
                super::x86_sse2::filter_v_8tap(dst, dst_stride, src, src_stride, w, h, filter);
            }
            return;
        }

        let f01 = unsafe { pair_f256(filter[0], filter[1]) };
        let f23 = unsafe { pair_f256(filter[2], filter[3]) };
        let f45 = unsafe { pair_f256(filter[4], filter[5]) };
        let f67 = unsafe { pair_f256(filter[6], filter[7]) };
        let round = _mm256_set1_epi32(64);

        for y in 0..h {
            let drow = unsafe { dst.as_mut_ptr().add(y * dst_stride) };
            let r0 = unsafe { src.as_ptr().add(y * src_stride) };
            let r1 = unsafe { src.as_ptr().add((y + 1) * src_stride) };
            let r2 = unsafe { src.as_ptr().add((y + 2) * src_stride) };
            let r3 = unsafe { src.as_ptr().add((y + 3) * src_stride) };
            let r4 = unsafe { src.as_ptr().add((y + 4) * src_stride) };
            let r5 = unsafe { src.as_ptr().add((y + 5) * src_stride) };
            let r6 = unsafe { src.as_ptr().add((y + 6) * src_stride) };
            let r7 = unsafe { src.as_ptr().add((y + 7) * src_stride) };

            let mut x = 0usize;
            while x + 16 <= w {
                let a0 = unsafe { load_16_u8_to_i16(r0.add(x)) };
                let a1 = unsafe { load_16_u8_to_i16(r1.add(x)) };
                let a2 = unsafe { load_16_u8_to_i16(r2.add(x)) };
                let a3 = unsafe { load_16_u8_to_i16(r3.add(x)) };
                let a4 = unsafe { load_16_u8_to_i16(r4.add(x)) };
                let a5 = unsafe { load_16_u8_to_i16(r5.add(x)) };
                let a6 = unsafe { load_16_u8_to_i16(r6.add(x)) };
                let a7 = unsafe { load_16_u8_to_i16(r7.add(x)) };

                let lo = _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_unpacklo_epi16(a0, a1), f01),
                        _mm256_madd_epi16(_mm256_unpacklo_epi16(a2, a3), f23),
                    ),
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_unpacklo_epi16(a4, a5), f45),
                        _mm256_madd_epi16(_mm256_unpacklo_epi16(a6, a7), f67),
                    ),
                );
                let hi = _mm256_add_epi32(
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_unpackhi_epi16(a0, a1), f01),
                        _mm256_madd_epi16(_mm256_unpackhi_epi16(a2, a3), f23),
                    ),
                    _mm256_add_epi32(
                        _mm256_madd_epi16(_mm256_unpackhi_epi16(a4, a5), f45),
                        _mm256_madd_epi16(_mm256_unpackhi_epi16(a6, a7), f67),
                    ),
                );
                let lo_s = _mm256_srai_epi32(_mm256_add_epi32(lo, round), 7);
                let hi_s = _mm256_srai_epi32(_mm256_add_epi32(hi, round), 7);
                let p16 = _mm256_packs_epi32(lo_s, hi_s);
                let p16_lo = _mm256_castsi256_si128(p16);
                let p16_hi = _mm256_extracti128_si256::<1>(p16);
                let p8 = _mm_packus_epi16(p16_lo, p16_hi);
                unsafe { _mm_storeu_si128(drow.add(x) as *mut __m128i, p8) };
                x += 16;
            }
            if x < w {
                let rem = w - x;
                unsafe {
                    super::x86_sse2::filter_v_8tap(
                        &mut dst[y * dst_stride + x..],
                        dst_stride,
                        &src[y * src_stride + x..],
                        src_stride,
                        rem,
                        1,
                        filter,
                    );
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn avg_block(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
    ) {
        for y in 0..h {
            let drow = unsafe { dst.as_mut_ptr().add(y * dst_stride) };
            let srow = unsafe { src.as_ptr().add(y * src_stride) };
            let mut x = 0usize;
            while x + 32 <= w {
                let vd = unsafe { _mm256_loadu_si256(drow.add(x) as *const __m256i) };
                let vs = unsafe { _mm256_loadu_si256(srow.add(x) as *const __m256i) };
                let avg = _mm256_avg_epu8(vd, vs);
                unsafe { _mm256_storeu_si256(drow.add(x) as *mut __m256i, avg) };
                x += 32;
            }
            while x + 16 <= w {
                let vd = unsafe { _mm_loadu_si128(drow.add(x) as *const __m128i) };
                let vs = unsafe { _mm_loadu_si128(srow.add(x) as *const __m128i) };
                let avg = _mm_avg_epu8(vd, vs);
                unsafe { _mm_storeu_si128(drow.add(x) as *mut __m128i, avg) };
                x += 16;
            }
            while x < w {
                unsafe {
                    let d = *drow.add(x) as u16;
                    let s = *srow.add(x) as u16;
                    *drow.add(x) = ((d + s + 1) >> 1) as u8;
                }
                x += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod arm_neon {
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn filter_h_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        filter: &[i16; 8],
    ) {
        let f: [int16x4_t; 8] = [
            vdup_n_s16(filter[0]),
            vdup_n_s16(filter[1]),
            vdup_n_s16(filter[2]),
            vdup_n_s16(filter[3]),
            vdup_n_s16(filter[4]),
            vdup_n_s16(filter[5]),
            vdup_n_s16(filter[6]),
            vdup_n_s16(filter[7]),
        ];
        let round = vdupq_n_s32(64);

        for y in 0..h {
            let srow = unsafe { src.as_ptr().add(y * src_stride) };
            let drow = unsafe { dst.as_mut_ptr().add(y * dst_stride) };
            let mut x = 0usize;
            while x + 8 <= w {
                let mut lo = round;
                let mut hi = round;
                for t in 0..8 {
                    let v8 = unsafe { vld1_u8(srow.add(x + t)) };
                    let v16 = vreinterpretq_s16_u16(vmovl_u8(v8));
                    lo = vmlal_s16(lo, vget_low_s16(v16), f[t]);
                    hi = vmlal_s16(hi, vget_high_s16(v16), f[t]);
                }
                let lo16 = vqmovn_s32(vshrq_n_s32::<7>(lo));
                let hi16 = vqmovn_s32(vshrq_n_s32::<7>(hi));
                let out8 = vqmovun_s16(vcombine_s16(lo16, hi16));
                unsafe { vst1_u8(drow.add(x), out8) };
                x += 8;
            }
            while x < w {
                let mut acc = 0i32;
                for (t, &ft) in filter.iter().enumerate() {
                    acc += ft as i32 * unsafe { *srow.add(x + t) } as i32;
                }
                unsafe { *drow.add(x) = ((acc + 64) >> 7).clamp(0, 255) as u8 };
                x += 1;
            }
        }
    }

    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn filter_v_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        filter: &[i16; 8],
    ) {
        let f: [int16x4_t; 8] = [
            vdup_n_s16(filter[0]),
            vdup_n_s16(filter[1]),
            vdup_n_s16(filter[2]),
            vdup_n_s16(filter[3]),
            vdup_n_s16(filter[4]),
            vdup_n_s16(filter[5]),
            vdup_n_s16(filter[6]),
            vdup_n_s16(filter[7]),
        ];
        let round = vdupq_n_s32(64);

        for y in 0..h {
            let drow = unsafe { dst.as_mut_ptr().add(y * dst_stride) };
            let mut x = 0usize;
            while x + 8 <= w {
                let mut lo = round;
                let mut hi = round;
                for t in 0..8 {
                    let row_ptr = unsafe { src.as_ptr().add((y + t) * src_stride + x) };
                    let v8 = unsafe { vld1_u8(row_ptr) };
                    let v16 = vreinterpretq_s16_u16(vmovl_u8(v8));
                    lo = vmlal_s16(lo, vget_low_s16(v16), f[t]);
                    hi = vmlal_s16(hi, vget_high_s16(v16), f[t]);
                }
                let lo16 = vqmovn_s32(vshrq_n_s32::<7>(lo));
                let hi16 = vqmovn_s32(vshrq_n_s32::<7>(hi));
                let out8 = vqmovun_s16(vcombine_s16(lo16, hi16));
                unsafe { vst1_u8(drow.add(x), out8) };
                x += 8;
            }
            while x < w {
                let mut acc = 0i32;
                for t in 0..8 {
                    acc += filter[t] as i32
                        * unsafe { *src.as_ptr().add((y + t) * src_stride + x) } as i32;
                }
                unsafe { *drow.add(x) = ((acc + 64) >> 7).clamp(0, 255) as u8 };
                x += 1;
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn avg_block(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
    ) {
        for y in 0..h {
            let drow = unsafe { dst.as_mut_ptr().add(y * dst_stride) };
            let srow = unsafe { src.as_ptr().add(y * src_stride) };
            let mut x = 0usize;
            while x + 16 <= w {
                let vd = unsafe { vld1q_u8(drow.add(x)) };
                let vs = unsafe { vld1q_u8(srow.add(x)) };
                unsafe { vst1q_u8(drow.add(x), vrhaddq_u8(vd, vs)) };
                x += 16;
            }
            while x + 8 <= w {
                let vd = unsafe { vld1_u8(drow.add(x)) };
                let vs = unsafe { vld1_u8(srow.add(x)) };
                unsafe { vst1_u8(drow.add(x), vrhadd_u8(vd, vs)) };
                x += 8;
            }
            while x < w {
                unsafe {
                    let d = *drow.add(x) as u16;
                    let s = *srow.add(x) as u16;
                    *drow.add(x) = ((d + s + 1) >> 1) as u8;
                }
                x += 1;
            }
        }
    }
}
