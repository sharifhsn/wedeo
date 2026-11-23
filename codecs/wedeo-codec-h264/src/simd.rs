//! Pure-Rust `core::arch` SIMD kernels for H.264/AVC IDCT and motion compensation.
//!
//! All `unsafe` `#[target_feature]` intrinsics in `wedeo-codec-h264` are strictly
//! isolated inside this module. Every algorithmic module in the crate remains
//! `#[forbid(unsafe_code)]`.
//!
//! Runtime CPU detection (`is_x86_feature_detected!`) selects AVX2 or SSE2 on
//! `x86_64`, NEON on `aarch64` (for straightforward byte kernels), and falls
//! back to safe scalar implementations on all other targets or degenerate strides.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop)]

/// Available SIMD execution backends for H.264 kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    Sse2,
    Avx2,
    Neon,
}

/// Return the highest-priority SIMD backend supported by the current CPU.
#[inline]
pub fn detected_backend() -> SimdBackend {
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

/// Return all backends executable on the current CPU (used by `checkasm`).
pub fn available_backends() -> Vec<SimdBackend> {
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

// ===========================================================================
// Public runtime-dispatched entrypoints
// ===========================================================================

/// 4x4 integer IDCT + saturating add to `dst`, zeroing `coeffs`.
#[inline]
pub fn idct_add_4x4(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 16]) {
    idct_add_4x4_with_backend(dst, stride, coeffs, detected_backend());
}

/// 4x4 IDCT with explicit backend selection (for `checkasm`).
#[inline]
pub fn idct_add_4x4_with_backend(
    dst: &mut [u8],
    stride: usize,
    coeffs: &mut [i16; 16],
    backend: SimdBackend,
) {
    assert!(3 * stride + 3 < dst.len());
    if stride < 4 {
        scalar::idct_add_4x4(dst, stride, coeffs);
        return;
    }
    match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: CPUID verified AVX2 support; `dst` has >= 3*stride+4 bytes
            // and `stride >= 4` ensures non-overlapping 4-byte rows.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::idct_add_4x4(dst, stride, coeffs);
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Sse2 | SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("sse2") => {
            // SAFETY: CPUID verified SSE2 support; `dst` has >= 3*stride+4 bytes
            // and `stride >= 4` ensures non-overlapping 4-byte rows.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::idct_add_4x4(dst, stride, coeffs);
            }
        }
        _ => scalar::idct_add_4x4(dst, stride, coeffs),
    }
}

/// 4x4 DC-only IDCT shortcut: add `(dc + 32) >> 6` to 4x4 block and zero `dc`.
#[inline]
pub fn idct_dc_add_4x4(dst: &mut [u8], stride: usize, dc: &mut i16) {
    idct_dc_add_4x4_with_backend(dst, stride, dc, detected_backend());
}

/// 4x4 DC-only IDCT with explicit backend selection (for `checkasm`).
#[inline]
pub fn idct_dc_add_4x4_with_backend(
    dst: &mut [u8],
    stride: usize,
    dc: &mut i16,
    backend: SimdBackend,
) {
    assert!(3 * stride + 3 < dst.len());
    if stride < 4 {
        scalar::idct_dc_add_4x4(dst, stride, dc);
        return;
    }
    match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: CPUID verified AVX2 support; `dst` has >= 3*stride+4 bytes
            // and `stride >= 4` guarantees non-overlapping 4-byte rows.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::idct_dc_add_4x4(dst, stride, dc);
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Sse2 | SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("sse2") => {
            // SAFETY: CPUID verified SSE2 support; `dst` has >= 3*stride+4 bytes
            // and `stride >= 4` guarantees non-overlapping 4-byte rows.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::idct_dc_add_4x4(dst, stride, dc);
            }
        }
        _ => scalar::idct_dc_add_4x4(dst, stride, dc),
    }
}

/// 8x8 integer IDCT + saturating add to `dst`, zeroing `coeffs`.
#[inline]
pub fn idct_add_8x8(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 64]) {
    idct_add_8x8_with_backend(dst, stride, coeffs, detected_backend());
}

/// 8x8 IDCT with explicit backend selection (for `checkasm`).
#[inline]
pub fn idct_add_8x8_with_backend(
    dst: &mut [u8],
    stride: usize,
    coeffs: &mut [i16; 64],
    backend: SimdBackend,
) {
    assert!(7 * stride + 7 < dst.len());
    if stride < 8 {
        scalar::idct_add_8x8(dst, stride, coeffs);
        return;
    }
    match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: CPUID verified AVX2 support; `dst` has >= 7*stride+8 bytes
            // and `stride >= 8` guarantees non-overlapping 8-byte rows.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::idct_add_8x8(dst, stride, coeffs);
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Sse2 | SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("sse2") => {
            // SAFETY: CPUID verified SSE2 support; `dst` has >= 7*stride+8 bytes
            // and `stride >= 8` guarantees non-overlapping 8-byte rows.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::idct_add_8x8(dst, stride, coeffs);
            }
        }
        _ => scalar::idct_add_8x8(dst, stride, coeffs),
    }
}

/// 8x8 DC-only IDCT shortcut: add `(dc + 32) >> 6` to 8x8 block and zero `dc`.
#[inline]
pub fn idct_dc_add_8x8(dst: &mut [u8], stride: usize, dc: &mut i16) {
    idct_dc_add_8x8_with_backend(dst, stride, dc, detected_backend());
}

/// 8x8 DC-only IDCT with explicit backend selection (for `checkasm`).
#[inline]
pub fn idct_dc_add_8x8_with_backend(
    dst: &mut [u8],
    stride: usize,
    dc: &mut i16,
    backend: SimdBackend,
) {
    assert!(7 * stride + 7 < dst.len());
    if stride < 8 {
        scalar::idct_dc_add_8x8(dst, stride, dc);
        return;
    }
    match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: CPUID verified AVX2 support; `dst` has >= 7*stride+8 bytes
            // and `stride >= 8` guarantees non-overlapping 8-byte rows.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::idct_dc_add_8x8(dst, stride, dc);
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Sse2 | SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("sse2") => {
            // SAFETY: CPUID verified SSE2 support; `dst` has >= 7*stride+8 bytes
            // and `stride >= 8` guarantees non-overlapping 8-byte rows.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::idct_dc_add_8x8(dst, stride, dc);
            }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
            // SAFETY: NEON support detected; `dst` has >= 7*stride+8 bytes
            // and `stride >= 8` guarantees non-overlapping 8-byte rows.
            #[allow(unsafe_code)]
            unsafe {
                aarch64_neon::idct_dc_add_8x8(dst, stride, dc);
            }
        }
        _ => scalar::idct_dc_add_8x8(dst, stride, dc),
    }
}

/// Copy a `16 x block_h` pixel block from `src` to `dst`.
#[inline]
pub fn put_pixels_16x16(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_h: usize,
) {
    put_pixels_16x16_with_backend(
        dst,
        dst_stride,
        src,
        src_stride,
        block_h,
        detected_backend(),
    );
}

/// `put_pixels_16x16` with explicit backend selection (for `checkasm`).
#[inline]
pub fn put_pixels_16x16_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_h: usize,
    backend: SimdBackend,
) {
    if block_h == 0 {
        return;
    }
    assert!((block_h - 1) * dst_stride + 16 <= dst.len());
    assert!((block_h - 1) * src_stride + 16 <= src.len());
    if dst_stride < 16 || src_stride < 16 {
        scalar::put_pixels(dst, dst_stride, src, src_stride, 16, block_h);
        return;
    }
    match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: AVX2 detected; bounds and non-overlapping strides checked above.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::put_pixels_16(dst, dst_stride, src, src_stride, block_h);
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Sse2 | SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("sse2") => {
            // SAFETY: SSE2 detected; bounds and non-overlapping strides checked above.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::put_pixels_16(dst, dst_stride, src, src_stride, block_h);
            }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
            // SAFETY: NEON detected; bounds and non-overlapping strides checked above.
            #[allow(unsafe_code)]
            unsafe {
                aarch64_neon::put_pixels_16(dst, dst_stride, src, src_stride, block_h);
            }
        }
        _ => scalar::put_pixels(dst, dst_stride, src, src_stride, 16, block_h),
    }
}

/// Copy an `8 x block_h` pixel block from `src` to `dst`.
#[inline]
pub fn put_pixels_8x8(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_h: usize,
) {
    put_pixels_8x8_with_backend(
        dst,
        dst_stride,
        src,
        src_stride,
        block_h,
        detected_backend(),
    );
}

/// `put_pixels_8x8` with explicit backend selection (for `checkasm`).
#[inline]
pub fn put_pixels_8x8_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_h: usize,
    backend: SimdBackend,
) {
    if block_h == 0 {
        return;
    }
    assert!((block_h - 1) * dst_stride + 8 <= dst.len());
    assert!((block_h - 1) * src_stride + 8 <= src.len());
    if dst_stride < 8 || src_stride < 8 {
        scalar::put_pixels(dst, dst_stride, src, src_stride, 8, block_h);
        return;
    }
    match backend {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: AVX2 detected; bounds and non-overlapping strides checked above.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::put_pixels_8(dst, dst_stride, src, src_stride, block_h);
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdBackend::Sse2 | SimdBackend::Avx2 if std::arch::is_x86_feature_detected!("sse2") => {
            // SAFETY: SSE2 detected; bounds and non-overlapping strides checked above.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::put_pixels_8(dst, dst_stride, src, src_stride, block_h);
            }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
            // SAFETY: NEON detected; bounds and non-overlapping strides checked above.
            #[allow(unsafe_code)]
            unsafe {
                aarch64_neon::put_pixels_8(dst, dst_stride, src, src_stride, block_h);
            }
        }
        _ => scalar::put_pixels(dst, dst_stride, src, src_stride, 8, block_h),
    }
}

/// Out-of-place pixel averaging `dst = (a + b + 1) >> 1` for width 16.
#[inline]
pub fn avg_pixels_16x16(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    block_h: usize,
) {
    avg_pixels_with_backend(
        dst,
        dst_stride,
        a,
        a_stride,
        b,
        b_stride,
        16,
        block_h,
        detected_backend(),
    );
}

/// Out-of-place pixel averaging `dst = (a + b + 1) >> 1` for width 8.
#[inline]
pub fn avg_pixels_8x8(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    block_h: usize,
) {
    avg_pixels_with_backend(
        dst,
        dst_stride,
        a,
        a_stride,
        b,
        b_stride,
        8,
        block_h,
        detected_backend(),
    );
}

/// Out-of-place pixel averaging `dst = (a + b + 1) >> 1` for width 4, 8, or 16.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn avg_pixels(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    block_w: usize,
    block_h: usize,
) {
    avg_pixels_with_backend(
        dst,
        dst_stride,
        a,
        a_stride,
        b,
        b_stride,
        block_w,
        block_h,
        detected_backend(),
    );
}

/// Out-of-place pixel averaging with explicit backend selection (for `checkasm`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn avg_pixels_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    block_w: usize,
    block_h: usize,
    backend: SimdBackend,
) {
    if block_h == 0 || block_w == 0 {
        return;
    }
    assert!((block_h - 1) * dst_stride + block_w <= dst.len());
    assert!((block_h - 1) * a_stride + block_w <= a.len());
    assert!((block_h - 1) * b_stride + block_w <= b.len());
    if dst_stride < block_w || a_stride < block_w || b_stride < block_w {
        scalar::avg_pixels(dst, dst_stride, a, a_stride, b, b_stride, block_w, block_h);
        return;
    }
    match (backend, block_w) {
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Avx2, 4 | 8 | 16) if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: AVX2 detected; bounds for all three buffers and non-overlapping
            // strides verified above; `block_w` is 4, 8, or 16.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::avg_pixels(dst, dst_stride, a, a_stride, b, b_stride, block_w, block_h);
            }
        }
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Sse2 | SimdBackend::Avx2, 4 | 8 | 16)
            if std::arch::is_x86_feature_detected!("sse2") =>
        {
            // SAFETY: SSE2 detected; bounds for all three buffers and non-overlapping
            // strides verified above; `block_w` is 4, 8, or 16.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::avg_pixels(dst, dst_stride, a, a_stride, b, b_stride, block_w, block_h);
            }
        }
        #[cfg(target_arch = "aarch64")]
        (SimdBackend::Neon, 8 | 16) if std::arch::is_aarch64_feature_detected!("neon") => {
            // SAFETY: NEON detected; bounds and strides verified above; `block_w` is 8 or 16.
            #[allow(unsafe_code)]
            unsafe {
                aarch64_neon::avg_pixels(
                    dst, dst_stride, a, a_stride, b, b_stride, block_w, block_h,
                );
            }
        }
        _ => scalar::avg_pixels(dst, dst_stride, a, a_stride, b, b_stride, block_w, block_h),
    }
}

/// In-place pixel averaging `dst = (dst + src + 1) >> 1` for width 4, 8, or 16.
#[inline]
pub fn avg_pixels_inplace(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
) {
    avg_pixels_inplace_with_backend(
        dst,
        dst_stride,
        src,
        src_stride,
        block_w,
        block_h,
        detected_backend(),
    );
}

/// In-place pixel averaging with explicit backend selection (for `checkasm`).
#[inline]
pub fn avg_pixels_inplace_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
    backend: SimdBackend,
) {
    if block_h == 0 || block_w == 0 {
        return;
    }
    assert!((block_h - 1) * dst_stride + block_w <= dst.len());
    assert!((block_h - 1) * src_stride + block_w <= src.len());
    if dst_stride < block_w || src_stride < block_w {
        scalar::avg_pixels_inplace(dst, dst_stride, src, src_stride, block_w, block_h);
        return;
    }
    match (backend, block_w) {
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Avx2, 4 | 8 | 16) if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: AVX2 detected; bounds and non-overlapping strides verified above.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::avg_pixels_inplace(dst, dst_stride, src, src_stride, block_w, block_h);
            }
        }
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Sse2 | SimdBackend::Avx2, 4 | 8 | 16)
            if std::arch::is_x86_feature_detected!("sse2") =>
        {
            // SAFETY: SSE2 detected; bounds and non-overlapping strides verified above.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::avg_pixels_inplace(dst, dst_stride, src, src_stride, block_w, block_h);
            }
        }
        #[cfg(target_arch = "aarch64")]
        (SimdBackend::Neon, 8 | 16) if std::arch::is_aarch64_feature_detected!("neon") => {
            // SAFETY: NEON detected; bounds and strides verified above.
            #[allow(unsafe_code)]
            unsafe {
                aarch64_neon::avg_pixels_inplace(
                    dst, dst_stride, src, src_stride, block_w, block_h,
                );
            }
        }
        _ => scalar::avg_pixels_inplace(dst, dst_stride, src, src_stride, block_w, block_h),
    }
}

/// 1/8-pel chroma bilinear interpolation `(a*s00 + b*s10 + c*s01 + d*s11 + 32) >> 6`.
///
/// If `avg` is `true`, computes `(dst + mc_val + 1) >> 1`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc_bilinear(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    block_w: usize,
    block_h: usize,
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    avg: bool,
) {
    chroma_mc_bilinear_with_backend(
        dst,
        dst_stride,
        ref_uv,
        ref_stride,
        block_w,
        block_h,
        a,
        b,
        c,
        d,
        avg,
        detected_backend(),
    );
}

/// `chroma_mc_bilinear` with explicit backend selection (for `checkasm`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc_bilinear_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    block_w: usize,
    block_h: usize,
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    avg: bool,
    backend: SimdBackend,
) {
    if block_h == 0 || block_w == 0 {
        return;
    }
    assert!((block_h - 1) * dst_stride + block_w <= dst.len());
    assert!(block_h * ref_stride + block_w < ref_uv.len());
    if dst_stride < block_w
        || ref_stride < block_w + 1
        || !(0..=64).contains(&a)
        || !(0..=64).contains(&b)
        || !(0..=64).contains(&c)
        || !(0..=64).contains(&d)
        || a + b + c + d != 64
    {
        scalar::chroma_mc_bilinear(
            dst, dst_stride, ref_uv, ref_stride, block_w, block_h, a, b, c, d, avg,
        );
        return;
    }
    match (backend, block_w) {
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Avx2, 4 | 8 | 16) if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: AVX2 detected; bounds on `dst` and `ref_uv` (including +1 col/+1 row)
            // verified above and weights are non-negative summing to 64 (no u16/i16 overflow).
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::chroma_mc_bilinear(
                    dst, dst_stride, ref_uv, ref_stride, block_w, block_h, a, b, c, d, avg,
                );
            }
        }
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Sse2 | SimdBackend::Avx2, 4 | 8 | 16)
            if std::arch::is_x86_feature_detected!("sse2") =>
        {
            // SAFETY: SSE2 detected; bounds and weight invariants verified above.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::chroma_mc_bilinear(
                    dst, dst_stride, ref_uv, ref_stride, block_w, block_h, a, b, c, d, avg,
                );
            }
        }
        _ => scalar::chroma_mc_bilinear(
            dst, dst_stride, ref_uv, ref_stride, block_w, block_h, a, b, c, d, avg,
        ),
    }
}

/// 1/8-pel chroma 2-tap interpolation `(a*s0 + e*s1 + 32) >> 6` where `s1` is at `+step`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc_2tap(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    block_w: usize,
    block_h: usize,
    a: i32,
    e: i32,
    step: usize,
    avg: bool,
) {
    chroma_mc_2tap_with_backend(
        dst,
        dst_stride,
        ref_uv,
        ref_stride,
        block_w,
        block_h,
        a,
        e,
        step,
        avg,
        detected_backend(),
    );
}

/// `chroma_mc_2tap` with explicit backend selection (for `checkasm`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn chroma_mc_2tap_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    ref_uv: &[u8],
    ref_stride: usize,
    block_w: usize,
    block_h: usize,
    a: i32,
    e: i32,
    step: usize,
    avg: bool,
    backend: SimdBackend,
) {
    if block_h == 0 || block_w == 0 {
        return;
    }
    assert!((block_h - 1) * dst_stride + block_w <= dst.len());
    assert!((block_h - 1) * ref_stride + step + block_w <= ref_uv.len());
    if dst_stride < block_w
        || ref_stride < block_w
        || !(0..=64).contains(&a)
        || !(0..=64).contains(&e)
        || a + e != 64
    {
        scalar::chroma_mc_2tap(
            dst, dst_stride, ref_uv, ref_stride, block_w, block_h, a, e, step, avg,
        );
        return;
    }
    match (backend, block_w) {
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Avx2, 4 | 8 | 16) if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: AVX2 detected; bounds including `+step` verified above.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::chroma_mc_2tap(
                    dst, dst_stride, ref_uv, ref_stride, block_w, block_h, a, e, step, avg,
                );
            }
        }
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Sse2 | SimdBackend::Avx2, 4 | 8 | 16)
            if std::arch::is_x86_feature_detected!("sse2") =>
        {
            // SAFETY: SSE2 detected; bounds including `+step` verified above.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::chroma_mc_2tap(
                    dst, dst_stride, ref_uv, ref_stride, block_w, block_h, a, e, step, avg,
                );
            }
        }
        _ => scalar::chroma_mc_2tap(
            dst, dst_stride, ref_uv, ref_stride, block_w, block_h, a, e, step, avg,
        ),
    }
}

/// Horizontal 6-tap FIR luma half-pel filter `clip((a - 5b + 20c + 20d - 5e + f + 16) >> 5)`.
///
/// `src` must point to tap `-2` (so each row reads `block_w + 5` bytes).
#[inline]
pub fn luma_6tap_h(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
) {
    luma_6tap_h_with_backend(
        dst,
        dst_stride,
        src,
        src_stride,
        block_w,
        block_h,
        detected_backend(),
    );
}

/// `luma_6tap_h` with explicit backend selection (for `checkasm`).
#[inline]
pub fn luma_6tap_h_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
    backend: SimdBackend,
) {
    if block_h == 0 || block_w == 0 {
        return;
    }
    assert!((block_h - 1) * dst_stride + block_w <= dst.len());
    assert!((block_h - 1) * src_stride + block_w + 5 <= src.len());
    if dst_stride < block_w || src_stride < block_w + 5 {
        scalar::luma_6tap_h(dst, dst_stride, src, src_stride, block_w, block_h);
        return;
    }
    match (backend, block_w) {
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Avx2, 4 | 8 | 16) if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: AVX2 detected; `src` has `(block_h-1)*src_stride + block_w + 5`
            // and `dst` has `(block_h-1)*dst_stride + block_w` bytes.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::luma_6tap_h(dst, dst_stride, src, src_stride, block_w, block_h);
            }
        }
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Sse2 | SimdBackend::Avx2, 4 | 8 | 16)
            if std::arch::is_x86_feature_detected!("sse2") =>
        {
            // SAFETY: SSE2 detected; bounds verified above.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::luma_6tap_h(dst, dst_stride, src, src_stride, block_w, block_h);
            }
        }
        _ => scalar::luma_6tap_h(dst, dst_stride, src, src_stride, block_w, block_h),
    }
}

/// Vertical 6-tap FIR luma half-pel filter `clip((a - 5b + 20c + 20d - 5e + f + 16) >> 5)`.
///
/// `src` must point to row `-2` (so it reads `block_h + 5` rows).
#[inline]
pub fn luma_6tap_v(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
) {
    luma_6tap_v_with_backend(
        dst,
        dst_stride,
        src,
        src_stride,
        block_w,
        block_h,
        detected_backend(),
    );
}

/// `luma_6tap_v` with explicit backend selection (for `checkasm`).
#[inline]
pub fn luma_6tap_v_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
    backend: SimdBackend,
) {
    if block_h == 0 || block_w == 0 {
        return;
    }
    assert!((block_h - 1) * dst_stride + block_w <= dst.len());
    assert!((block_h + 4) * src_stride + block_w <= src.len());
    if dst_stride < block_w || src_stride < block_w {
        scalar::luma_6tap_v(dst, dst_stride, src, src_stride, block_w, block_h);
        return;
    }
    match (backend, block_w) {
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Avx2, 4 | 8 | 16) if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: AVX2 detected; `src` has `(block_h+4)*src_stride + block_w` bytes
            // and `dst` has `(block_h-1)*dst_stride + block_w` bytes.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::luma_6tap_v(dst, dst_stride, src, src_stride, block_w, block_h);
            }
        }
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Sse2 | SimdBackend::Avx2, 4 | 8 | 16)
            if std::arch::is_x86_feature_detected!("sse2") =>
        {
            // SAFETY: SSE2 detected; bounds verified above.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::luma_6tap_v(dst, dst_stride, src, src_stride, block_w, block_h);
            }
        }
        _ => scalar::luma_6tap_v(dst, dst_stride, src, src_stride, block_w, block_h),
    }
}

/// 2D half-pel 6-tap FIR filter ("j" position) reading directly from `u8` reference.
///
/// `src` must point to `(-2, -2)` so `(block_h + 5)` rows of `(block_w + 5)` bytes are valid.
/// `tmp` must have length `>= (block_h + 5) * block_w`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn luma_6tap_hv(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
    tmp: &mut [i32],
) {
    luma_6tap_hv_with_backend(
        dst,
        dst_stride,
        src,
        src_stride,
        block_w,
        block_h,
        tmp,
        detected_backend(),
    );
}

/// `luma_6tap_hv` with explicit backend selection (for `checkasm`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn luma_6tap_hv_with_backend(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    block_w: usize,
    block_h: usize,
    tmp: &mut [i32],
    backend: SimdBackend,
) {
    if block_h == 0 || block_w == 0 {
        return;
    }
    assert!((block_h - 1) * dst_stride + block_w <= dst.len());
    assert!((block_h + 4) * src_stride + block_w + 5 <= src.len());
    assert!((block_h + 5) * block_w <= tmp.len());
    if dst_stride < block_w || src_stride < block_w + 5 {
        scalar::luma_6tap_hv(dst, dst_stride, src, src_stride, block_w, block_h, tmp);
        return;
    }
    match (backend, block_w) {
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Avx2, 4 | 8 | 16) if std::arch::is_x86_feature_detected!("avx2") => {
            // SAFETY: AVX2 detected; `src`, `dst`, and `tmp` bounds verified above.
            #[allow(unsafe_code)]
            unsafe {
                x86_avx2::luma_6tap_hv(dst, dst_stride, src, src_stride, block_w, block_h, tmp);
            }
        }
        #[cfg(target_arch = "x86_64")]
        (SimdBackend::Sse2 | SimdBackend::Avx2, 4 | 8 | 16)
            if std::arch::is_x86_feature_detected!("sse2") =>
        {
            // SAFETY: SSE2 detected; `src`, `dst`, and `tmp` bounds verified above.
            #[allow(unsafe_code)]
            unsafe {
                x86_sse2::luma_6tap_hv(dst, dst_stride, src, src_stride, block_w, block_h, tmp);
            }
        }
        _ => scalar::luma_6tap_hv(dst, dst_stride, src, src_stride, block_w, block_h, tmp),
    }
}

// ===========================================================================
// Pure-Rust scalar reference kernels (100% safe)
// ===========================================================================

pub mod scalar {
    #[inline(always)]
    fn clip_u8(x: i32) -> u8 {
        x.clamp(0, 255) as u8
    }

    #[inline(always)]
    fn filter6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
        a - 5 * b + 20 * c + 20 * d - 5 * e + f
    }

    pub fn idct_add_4x4(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 16]) {
        coeffs[0] = coeffs[0].wrapping_add(32);

        for i in 0..4 {
            let row = 4 * i;
            let z0 = coeffs[row] as i32 + coeffs[row + 2] as i32;
            let z1 = coeffs[row] as i32 - coeffs[row + 2] as i32;
            let z2 = (coeffs[row + 1] as i32 >> 1) - coeffs[row + 3] as i32;
            let z3 = coeffs[row + 1] as i32 + (coeffs[row + 3] as i32 >> 1);

            coeffs[row] = (z0 + z3) as i16;
            coeffs[row + 1] = (z1 + z2) as i16;
            coeffs[row + 2] = (z1 - z2) as i16;
            coeffs[row + 3] = (z0 - z3) as i16;
        }

        assert!(3 * stride + 3 < dst.len());
        for i in 0..4 {
            let z0 = coeffs[i] as i32 + coeffs[i + 8] as i32;
            let z1 = coeffs[i] as i32 - coeffs[i + 8] as i32;
            let z2 = (coeffs[i + 4] as i32 >> 1) - coeffs[i + 12] as i32;
            let z3 = coeffs[i + 4] as i32 + (coeffs[i + 12] as i32 >> 1);

            dst[i] = clip_u8(dst[i] as i32 + ((z0 + z3) >> 6));
            dst[stride + i] = clip_u8(dst[stride + i] as i32 + ((z1 + z2) >> 6));
            dst[2 * stride + i] = clip_u8(dst[2 * stride + i] as i32 + ((z1 - z2) >> 6));
            dst[3 * stride + i] = clip_u8(dst[3 * stride + i] as i32 + ((z0 - z3) >> 6));
        }

        *coeffs = [0i16; 16];
    }

    pub fn idct_dc_add_4x4(dst: &mut [u8], stride: usize, dc: &mut i16) {
        let val = (*dc as i32 + 32) >> 6;
        *dc = 0;

        assert!(3 * stride + 3 < dst.len());
        for j in 0..4 {
            let row = j * stride;
            for px in &mut dst[row..row + 4] {
                *px = clip_u8(*px as i32 + val);
            }
        }
    }

    pub fn idct_add_8x8(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 64]) {
        coeffs[0] = coeffs[0].wrapping_add(32);

        for i in 0..8 {
            let row = i * 8;
            let a0 = coeffs[row] as i32 + coeffs[row + 4] as i32;
            let a2 = coeffs[row] as i32 - coeffs[row + 4] as i32;
            let a4 = (coeffs[row + 2] as i32 >> 1) - coeffs[row + 6] as i32;
            let a6 = (coeffs[row + 6] as i32 >> 1) + coeffs[row + 2] as i32;

            let b0 = a0 + a6;
            let b2 = a2 + a4;
            let b4 = a2 - a4;
            let b6 = a0 - a6;

            let a1 = -(coeffs[row + 3] as i32) + coeffs[row + 5] as i32
                - coeffs[row + 7] as i32
                - (coeffs[row + 7] as i32 >> 1);
            let a3 = coeffs[row + 1] as i32 + coeffs[row + 7] as i32
                - coeffs[row + 3] as i32
                - (coeffs[row + 3] as i32 >> 1);
            let a5 = -(coeffs[row + 1] as i32)
                + coeffs[row + 7] as i32
                + coeffs[row + 5] as i32
                + (coeffs[row + 5] as i32 >> 1);
            let a7 = coeffs[row + 3] as i32
                + coeffs[row + 5] as i32
                + coeffs[row + 1] as i32
                + (coeffs[row + 1] as i32 >> 1);

            let b1 = (a7 >> 2) + a1;
            let b3 = a3 + (a5 >> 2);
            let b5 = (a3 >> 2) - a5;
            let b7 = a7 - (a1 >> 2);

            coeffs[row] = (b0 + b7) as i16;
            coeffs[row + 7] = (b0 - b7) as i16;
            coeffs[row + 1] = (b2 + b5) as i16;
            coeffs[row + 6] = (b2 - b5) as i16;
            coeffs[row + 2] = (b4 + b3) as i16;
            coeffs[row + 5] = (b4 - b3) as i16;
            coeffs[row + 3] = (b6 + b1) as i16;
            coeffs[row + 4] = (b6 - b1) as i16;
        }

        assert!(7 * stride + 7 < dst.len());
        for i in 0..8 {
            let a0 = coeffs[i] as i32 + coeffs[i + 32] as i32;
            let a2 = coeffs[i] as i32 - coeffs[i + 32] as i32;
            let a4 = (coeffs[i + 16] as i32 >> 1) - coeffs[i + 48] as i32;
            let a6 = (coeffs[i + 48] as i32 >> 1) + coeffs[i + 16] as i32;

            let b0 = a0 + a6;
            let b2 = a2 + a4;
            let b4 = a2 - a4;
            let b6 = a0 - a6;

            let a1 = -(coeffs[i + 24] as i32) + coeffs[i + 40] as i32
                - coeffs[i + 56] as i32
                - (coeffs[i + 56] as i32 >> 1);
            let a3 = coeffs[i + 8] as i32 + coeffs[i + 56] as i32
                - coeffs[i + 24] as i32
                - (coeffs[i + 24] as i32 >> 1);
            let a5 = -(coeffs[i + 8] as i32)
                + coeffs[i + 56] as i32
                + coeffs[i + 40] as i32
                + (coeffs[i + 40] as i32 >> 1);
            let a7 = coeffs[i + 24] as i32
                + coeffs[i + 40] as i32
                + coeffs[i + 8] as i32
                + (coeffs[i + 8] as i32 >> 1);

            let b1 = (a7 >> 2) + a1;
            let b3 = a3 + (a5 >> 2);
            let b5 = (a3 >> 2) - a5;
            let b7 = a7 - (a1 >> 2);

            dst[i] = clip_u8(dst[i] as i32 + ((b0 + b7) >> 6));
            dst[stride + i] = clip_u8(dst[stride + i] as i32 + ((b2 + b5) >> 6));
            dst[2 * stride + i] = clip_u8(dst[2 * stride + i] as i32 + ((b4 + b3) >> 6));
            dst[3 * stride + i] = clip_u8(dst[3 * stride + i] as i32 + ((b6 + b1) >> 6));
            dst[4 * stride + i] = clip_u8(dst[4 * stride + i] as i32 + ((b6 - b1) >> 6));
            dst[5 * stride + i] = clip_u8(dst[5 * stride + i] as i32 + ((b4 - b3) >> 6));
            dst[6 * stride + i] = clip_u8(dst[6 * stride + i] as i32 + ((b2 - b5) >> 6));
            dst[7 * stride + i] = clip_u8(dst[7 * stride + i] as i32 + ((b0 - b7) >> 6));
        }

        *coeffs = [0i16; 64];
    }

    pub fn idct_dc_add_8x8(dst: &mut [u8], stride: usize, dc: &mut i16) {
        let val = (*dc as i32 + 32) >> 6;
        *dc = 0;

        assert!(7 * stride + 7 < dst.len());
        for j in 0..8 {
            let row = j * stride;
            for px in &mut dst[row..row + 8] {
                *px = clip_u8(*px as i32 + val);
            }
        }
    }

    pub fn put_pixels(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        for j in 0..block_h {
            let s = j * src_stride;
            let d = j * dst_stride;
            dst[d..d + block_w].copy_from_slice(&src[s..s + block_w]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn avg_pixels(
        dst: &mut [u8],
        dst_stride: usize,
        a: &[u8],
        a_stride: usize,
        b: &[u8],
        b_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        for j in 0..block_h {
            for i in 0..block_w {
                let va = a[j * a_stride + i] as u16;
                let vb = b[j * b_stride + i] as u16;
                dst[j * dst_stride + i] = ((va + vb + 1) >> 1) as u8;
            }
        }
    }

    pub fn avg_pixels_inplace(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        for j in 0..block_h {
            let d_off = j * dst_stride;
            let s_off = j * src_stride;
            for i in 0..block_w {
                let a = dst[d_off + i] as u16;
                let b = src[s_off + i] as u16;
                dst[d_off + i] = ((a + b + 1) >> 1) as u8;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chroma_mc_bilinear(
        dst: &mut [u8],
        dst_stride: usize,
        ref_uv: &[u8],
        ref_stride: usize,
        block_w: usize,
        block_h: usize,
        a: i32,
        b: i32,
        c: i32,
        d: i32,
        avg: bool,
    ) {
        for j in 0..block_h {
            let row_off = j * ref_stride;
            let dst_row = &mut dst[j * dst_stride..j * dst_stride + block_w];
            for (i, dst_px) in dst_row.iter_mut().enumerate().take(block_w) {
                let off = row_off + i;
                let s00 = ref_uv[off] as i32;
                let s10 = ref_uv[off + 1] as i32;
                let s01 = ref_uv[off + ref_stride] as i32;
                let s11 = ref_uv[off + ref_stride + 1] as i32;
                let val = ((a * s00 + b * s10 + c * s01 + d * s11 + 32) >> 6) as u8;
                if avg {
                    *dst_px = ((*dst_px as u16 + val as u16 + 1) >> 1) as u8;
                } else {
                    *dst_px = val;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chroma_mc_2tap(
        dst: &mut [u8],
        dst_stride: usize,
        ref_uv: &[u8],
        ref_stride: usize,
        block_w: usize,
        block_h: usize,
        a: i32,
        e: i32,
        step: usize,
        avg: bool,
    ) {
        for j in 0..block_h {
            let row_off = j * ref_stride;
            let dst_row = &mut dst[j * dst_stride..j * dst_stride + block_w];
            for (i, dst_px) in dst_row.iter_mut().enumerate().take(block_w) {
                let off = row_off + i;
                let s0 = ref_uv[off] as i32;
                let s1 = ref_uv[off + step] as i32;
                let val = ((a * s0 + e * s1 + 32) >> 6) as u8;
                if avg {
                    *dst_px = ((*dst_px as u16 + val as u16 + 1) >> 1) as u8;
                } else {
                    *dst_px = val;
                }
            }
        }
    }

    pub fn luma_6tap_h(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        for j in 0..block_h {
            let src_row = &src[j * src_stride..j * src_stride + block_w + 5];
            let dst_row = &mut dst[j * dst_stride..j * dst_stride + block_w];
            for i in 0..block_w {
                let val = filter6(
                    src_row[i] as i32,
                    src_row[i + 1] as i32,
                    src_row[i + 2] as i32,
                    src_row[i + 3] as i32,
                    src_row[i + 4] as i32,
                    src_row[i + 5] as i32,
                );
                dst_row[i] = clip_u8((val + 16) >> 5);
            }
        }
    }

    pub fn luma_6tap_v(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        for j in 0..block_h {
            let row_base = j * src_stride;
            let dst_row = &mut dst[j * dst_stride..j * dst_stride + block_w];
            for (i, dst_px) in dst_row.iter_mut().enumerate().take(block_w) {
                let base = row_base + i;
                let val = filter6(
                    src[base] as i32,
                    src[base + src_stride] as i32,
                    src[base + 2 * src_stride] as i32,
                    src[base + 3 * src_stride] as i32,
                    src[base + 4 * src_stride] as i32,
                    src[base + 5 * src_stride] as i32,
                );
                *dst_px = clip_u8((val + 16) >> 5);
            }
        }
    }

    pub fn luma_6tap_hv(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
        tmp: &mut [i32],
    ) {
        let tmp_h = block_h + 5;
        for j in 0..tmp_h {
            let src_row = &src[j * src_stride..j * src_stride + block_w + 5];
            let tmp_row = &mut tmp[j * block_w..j * block_w + block_w];
            for i in 0..block_w {
                tmp_row[i] = filter6(
                    src_row[i] as i32,
                    src_row[i + 1] as i32,
                    src_row[i + 2] as i32,
                    src_row[i + 3] as i32,
                    src_row[i + 4] as i32,
                    src_row[i + 5] as i32,
                );
            }
        }
        for j in 0..block_h {
            let dst_row = &mut dst[j * dst_stride..j * dst_stride + block_w];
            for (i, dst_px) in dst_row.iter_mut().enumerate().take(block_w) {
                let val = filter6(
                    tmp[j * block_w + i],
                    tmp[(j + 1) * block_w + i],
                    tmp[(j + 2) * block_w + i],
                    tmp[(j + 3) * block_w + i],
                    tmp[(j + 4) * block_w + i],
                    tmp[(j + 5) * block_w + i],
                );
                *dst_px = clip_u8((val + 512) >> 10);
            }
        }
    }
}

// ===========================================================================
// x86_64 SSE2 kernels
// ===========================================================================

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86_sse2 {
    use core::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn load_u32_xmm(p: *const u8) -> __m128i {
        // SAFETY: Caller guarantees `p` is valid for 4-byte unaligned read.
        let v = unsafe { core::ptr::read_unaligned(p as *const u32) };
        _mm_cvtsi32_si128(v as i32)
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn store_u32_xmm(p: *mut u8, v: __m128i) {
        let bits = _mm_cvtsi128_si32(v) as u32;
        // SAFETY: Caller guarantees `p` is valid for 4-byte unaligned write.
        unsafe { core::ptr::write_unaligned(p as *mut u32, bits) };
    }

    #[inline(always)]
    unsafe fn load_u64_xmm(p: *const u8) -> __m128i {
        // SAFETY: Caller guarantees `p` is valid for 8-byte unaligned read.
        unsafe { _mm_loadl_epi64(p as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn store_u64_xmm(p: *mut u8, v: __m128i) {
        // SAFETY: Caller guarantees `p` is valid for 8-byte unaligned write.
        unsafe { _mm_storel_epi64(p as *mut __m128i, v) };
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    pub(super) unsafe fn transpose_8x8_i16(r: &mut [__m128i; 8]) {
        let t0 = _mm_unpacklo_epi16(r[0], r[1]);
        let t1 = _mm_unpackhi_epi16(r[0], r[1]);
        let t2 = _mm_unpacklo_epi16(r[2], r[3]);
        let t3 = _mm_unpackhi_epi16(r[2], r[3]);
        let t4 = _mm_unpacklo_epi16(r[4], r[5]);
        let t5 = _mm_unpackhi_epi16(r[4], r[5]);
        let t6 = _mm_unpacklo_epi16(r[6], r[7]);
        let t7 = _mm_unpackhi_epi16(r[6], r[7]);

        let u0 = _mm_unpacklo_epi32(t0, t2);
        let u1 = _mm_unpackhi_epi32(t0, t2);
        let u2 = _mm_unpacklo_epi32(t1, t3);
        let u3 = _mm_unpackhi_epi32(t1, t3);
        let u4 = _mm_unpacklo_epi32(t4, t6);
        let u5 = _mm_unpackhi_epi32(t4, t6);
        let u6 = _mm_unpacklo_epi32(t5, t7);
        let u7 = _mm_unpackhi_epi32(t5, t7);

        r[0] = _mm_unpacklo_epi64(u0, u4);
        r[1] = _mm_unpackhi_epi64(u0, u4);
        r[2] = _mm_unpacklo_epi64(u1, u5);
        r[3] = _mm_unpackhi_epi64(u1, u5);
        r[4] = _mm_unpacklo_epi64(u2, u6);
        r[5] = _mm_unpackhi_epi64(u2, u6);
        r[6] = _mm_unpacklo_epi64(u3, u7);
        r[7] = _mm_unpackhi_epi64(u3, u7);
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn idct_add_4x4(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 16]) {
        // SAFETY: Caller guarantees `3 * stride + 3 < dst.len()` and `stride >= 4`.
        let cp = coeffs.as_ptr();
        let mut r0 = unsafe { _mm_loadl_epi64(cp as *const __m128i) };
        let r1 = unsafe { _mm_loadl_epi64(cp.add(4) as *const __m128i) };
        let r2 = unsafe { _mm_loadl_epi64(cp.add(8) as *const __m128i) };
        let r3 = unsafe { _mm_loadl_epi64(cp.add(12) as *const __m128i) };

        r0 = _mm_add_epi16(r0, _mm_setr_epi16(32, 0, 0, 0, 0, 0, 0, 0));

        // Transpose 4x4 i16 so c0..c3 hold columns 0..3 across rows 0..3 in low 4 lanes.
        let t0 = _mm_unpacklo_epi16(r0, r1);
        let t1 = _mm_unpacklo_epi16(r2, r3);
        let c0 = _mm_unpacklo_epi32(t0, t1);
        let c2 = _mm_unpackhi_epi32(t0, t1);
        let c1 = _mm_srli_si128::<8>(c0);
        let c3 = _mm_srli_si128::<8>(c2);

        // Pass 1 (row butterfly in wrapping i16 — bit-exact to scalar i32 -> i16 truncation).
        let z0 = _mm_add_epi16(c0, c2);
        let z1 = _mm_sub_epi16(c0, c2);
        let z2 = _mm_sub_epi16(_mm_srai_epi16::<1>(c1), c3);
        let z3 = _mm_add_epi16(c1, _mm_srai_epi16::<1>(c3));

        let o0 = _mm_add_epi16(z0, z3);
        let o1 = _mm_add_epi16(z1, z2);
        let o2 = _mm_sub_epi16(z1, z2);
        let o3 = _mm_sub_epi16(z0, z3);

        // Transpose back to rows: row01 = [row0, row1], row23 = [row2, row3].
        let u0 = _mm_unpacklo_epi16(o0, o1);
        let u1 = _mm_unpacklo_epi16(o2, o3);
        let row01 = _mm_unpacklo_epi32(u0, u1);
        let row23 = _mm_unpackhi_epi32(u0, u1);

        // Pass 2 in i32 (prevents any i16 overflow on extreme residual coefficients).
        let p0 = _mm_srai_epi32::<16>(_mm_unpacklo_epi16(row01, row01));
        let p1 = _mm_srai_epi32::<16>(_mm_unpackhi_epi16(row01, row01));
        let p2 = _mm_srai_epi32::<16>(_mm_unpacklo_epi16(row23, row23));
        let p3 = _mm_srai_epi32::<16>(_mm_unpackhi_epi16(row23, row23));

        let cz0 = _mm_add_epi32(p0, p2);
        let cz1 = _mm_sub_epi32(p0, p2);
        let cz2 = _mm_sub_epi32(_mm_srai_epi32::<1>(p1), p3);
        let cz3 = _mm_add_epi32(p1, _mm_srai_epi32::<1>(p3));

        let d0 = _mm_srai_epi32::<6>(_mm_add_epi32(cz0, cz3));
        let d1 = _mm_srai_epi32::<6>(_mm_add_epi32(cz1, cz2));
        let d2 = _mm_srai_epi32::<6>(_mm_sub_epi32(cz1, cz2));
        let d3 = _mm_srai_epi32::<6>(_mm_sub_epi32(cz0, cz3));

        let dp = dst.as_mut_ptr();
        let zero = _mm_setzero_si128();
        // SAFETY: `dp.add(j * stride)` is within `dst` bounds for `j in 0..4`.
        let s0 = unsafe { load_u32_xmm(dp) };
        let s1 = unsafe { load_u32_xmm(dp.add(stride)) };
        let s2 = unsafe { load_u32_xmm(dp.add(2 * stride)) };
        let s3 = unsafe { load_u32_xmm(dp.add(3 * stride)) };

        let x0 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(s0, zero), zero);
        let x1 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(s1, zero), zero);
        let x2 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(s2, zero), zero);
        let x3 = _mm_unpacklo_epi16(_mm_unpacklo_epi8(s3, zero), zero);

        let sum01 = _mm_packs_epi32(_mm_add_epi32(x0, d0), _mm_add_epi32(x1, d1));
        let sum23 = _mm_packs_epi32(_mm_add_epi32(x2, d2), _mm_add_epi32(x3, d3));
        let packed = _mm_packus_epi16(sum01, sum23);

        // SAFETY: Each row write is 4 bytes and `stride >= 4` so rows are disjoint.
        unsafe {
            store_u32_xmm(dp, packed);
            store_u32_xmm(dp.add(stride), _mm_shuffle_epi32::<0x55>(packed));
            store_u32_xmm(dp.add(2 * stride), _mm_shuffle_epi32::<0xAA>(packed));
            store_u32_xmm(dp.add(3 * stride), _mm_shuffle_epi32::<0xFF>(packed));
        }

        *coeffs = [0i16; 16];
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn idct_dc_add_4x4(dst: &mut [u8], stride: usize, dc: &mut i16) {
        let val = (*dc as i32 + 32) >> 6;
        *dc = 0;
        if val == 0 {
            return;
        }
        let dp = dst.as_mut_ptr();
        if val > 0 {
            let add = _mm_set1_epi8(val.min(255) as i8);
            for j in 0..4 {
                // SAFETY: `3 * stride + 3 < dst.len()` checked by caller.
                unsafe {
                    let row = dp.add(j * stride);
                    let px = load_u32_xmm(row);
                    store_u32_xmm(row, _mm_adds_epu8(px, add));
                }
            }
        } else {
            let sub = _mm_set1_epi8((-val).min(255) as i8);
            for j in 0..4 {
                // SAFETY: `3 * stride + 3 < dst.len()` checked by caller.
                unsafe {
                    let row = dp.add(j * stride);
                    let px = load_u32_xmm(row);
                    store_u32_xmm(row, _mm_subs_epu8(px, sub));
                }
            }
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn idct_dc_add_8x8(dst: &mut [u8], stride: usize, dc: &mut i16) {
        let val = (*dc as i32 + 32) >> 6;
        *dc = 0;
        if val == 0 {
            return;
        }
        let dp = dst.as_mut_ptr();
        if val > 0 {
            let add = _mm_set1_epi8(val.min(255) as i8);
            for j in 0..8 {
                // SAFETY: `7 * stride + 7 < dst.len()` checked by caller.
                unsafe {
                    let row = dp.add(j * stride);
                    let px = load_u64_xmm(row);
                    store_u64_xmm(row, _mm_adds_epu8(px, add));
                }
            }
        } else {
            let sub = _mm_set1_epi8((-val).min(255) as i8);
            for j in 0..8 {
                // SAFETY: `7 * stride + 7 < dst.len()` checked by caller.
                unsafe {
                    let row = dp.add(j * stride);
                    let px = load_u64_xmm(row);
                    store_u64_xmm(row, _mm_subs_epu8(px, sub));
                }
            }
        }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn idct8_butterfly_4(v: [__m128i; 8]) -> [__m128i; 8] {
        let a0 = _mm_add_epi32(v[0], v[4]);
        let a2 = _mm_sub_epi32(v[0], v[4]);
        let a4 = _mm_sub_epi32(_mm_srai_epi32::<1>(v[2]), v[6]);
        let a6 = _mm_add_epi32(_mm_srai_epi32::<1>(v[6]), v[2]);

        let b0 = _mm_add_epi32(a0, a6);
        let b2 = _mm_add_epi32(a2, a4);
        let b4 = _mm_sub_epi32(a2, a4);
        let b6 = _mm_sub_epi32(a0, a6);

        let a1 = _mm_sub_epi32(
            _mm_sub_epi32(_mm_sub_epi32(v[5], v[3]), v[7]),
            _mm_srai_epi32::<1>(v[7]),
        );
        let a3 = _mm_sub_epi32(
            _mm_sub_epi32(_mm_add_epi32(v[1], v[7]), v[3]),
            _mm_srai_epi32::<1>(v[3]),
        );
        let a5 = _mm_add_epi32(
            _mm_add_epi32(_mm_sub_epi32(v[7], v[1]), v[5]),
            _mm_srai_epi32::<1>(v[5]),
        );
        let a7 = _mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(v[3], v[5]), v[1]),
            _mm_srai_epi32::<1>(v[1]),
        );

        let b1 = _mm_add_epi32(_mm_srai_epi32::<2>(a7), a1);
        let b3 = _mm_add_epi32(a3, _mm_srai_epi32::<2>(a5));
        let b5 = _mm_sub_epi32(_mm_srai_epi32::<2>(a3), a5);
        let b7 = _mm_sub_epi32(a7, _mm_srai_epi32::<2>(a1));

        [
            _mm_add_epi32(b0, b7),
            _mm_add_epi32(b2, b5),
            _mm_add_epi32(b4, b3),
            _mm_add_epi32(b6, b1),
            _mm_sub_epi32(b6, b1),
            _mm_sub_epi32(b4, b3),
            _mm_sub_epi32(b2, b5),
            _mm_sub_epi32(b0, b7),
        ]
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn idct_add_8x8(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 64]) {
        let cp = coeffs.as_ptr();
        // SAFETY: `coeffs` has 64 `i16` elements (8 rows of 8).
        let mut r = unsafe {
            [
                _mm_loadu_si128(cp as *const __m128i),
                _mm_loadu_si128(cp.add(8) as *const __m128i),
                _mm_loadu_si128(cp.add(16) as *const __m128i),
                _mm_loadu_si128(cp.add(24) as *const __m128i),
                _mm_loadu_si128(cp.add(32) as *const __m128i),
                _mm_loadu_si128(cp.add(40) as *const __m128i),
                _mm_loadu_si128(cp.add(48) as *const __m128i),
                _mm_loadu_si128(cp.add(56) as *const __m128i),
            ]
        };
        r[0] = _mm_add_epi16(r[0], _mm_setr_epi16(32, 0, 0, 0, 0, 0, 0, 0));

        // SAFETY: Pure register SIMD transpose.
        unsafe { transpose_8x8_i16(&mut r) };

        // Pass 1 in i32 (rows 0..3 in lo, rows 4..7 in hi), wrapping back to i16.
        let mut lo_in = [_mm_setzero_si128(); 8];
        let mut hi_in = [_mm_setzero_si128(); 8];
        for k in 0..8 {
            lo_in[k] = _mm_srai_epi32::<16>(_mm_unpacklo_epi16(r[k], r[k]));
            hi_in[k] = _mm_srai_epi32::<16>(_mm_unpackhi_epi16(r[k], r[k]));
        }
        // SAFETY: SSE2 enabled.
        let lo_out = unsafe { idct8_butterfly_4(lo_in) };
        let hi_out = unsafe { idct8_butterfly_4(hi_in) };

        let mut mid = [_mm_setzero_si128(); 8];
        for k in 0..8 {
            let l = _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(lo_out[k]));
            let h = _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(hi_out[k]));
            mid[k] = _mm_packs_epi32(l, h);
        }

        // Transpose back to rows.
        // SAFETY: Pure register SIMD transpose.
        unsafe { transpose_8x8_i16(&mut mid) };

        // Pass 2 in i32 (cols 0..3 in lo, cols 4..7 in hi), shift >> 6, add dst, clamp u8.
        for k in 0..8 {
            lo_in[k] = _mm_srai_epi32::<16>(_mm_unpacklo_epi16(mid[k], mid[k]));
            hi_in[k] = _mm_srai_epi32::<16>(_mm_unpackhi_epi16(mid[k], mid[k]));
        }
        // SAFETY: SSE2 enabled.
        let res_lo = unsafe { idct8_butterfly_4(lo_in) };
        let res_hi = unsafe { idct8_butterfly_4(hi_in) };

        let dp = dst.as_mut_ptr();
        let zero = _mm_setzero_si128();
        for j in 0..8 {
            // SAFETY: Caller verified `7 * stride + 7 < dst.len()` and `stride >= 8`.
            unsafe {
                let row_ptr = dp.add(j * stride);
                let src8 = load_u64_xmm(row_ptr);
                let src16 = _mm_unpacklo_epi8(src8, zero);
                let src_lo = _mm_unpacklo_epi16(src16, zero);
                let src_hi = _mm_unpackhi_epi16(src16, zero);

                let sum_lo = _mm_add_epi32(src_lo, _mm_srai_epi32::<6>(res_lo[j]));
                let sum_hi = _mm_add_epi32(src_hi, _mm_srai_epi32::<6>(res_hi[j]));
                let sum16 = _mm_packs_epi32(sum_lo, sum_hi);
                let out8 = _mm_packus_epi16(sum16, sum16);
                store_u64_xmm(row_ptr, out8);
            }
        }

        *coeffs = [0i16; 64];
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn put_pixels_16(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        for j in 0..block_h {
            // SAFETY: Caller verified bounds for `block_h` rows of 16 bytes.
            unsafe {
                let v = _mm_loadu_si128(sp.add(j * src_stride) as *const __m128i);
                _mm_storeu_si128(dp.add(j * dst_stride) as *mut __m128i, v);
            }
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn put_pixels_8(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        for j in 0..block_h {
            // SAFETY: Caller verified bounds for `block_h` rows of 8 bytes.
            unsafe {
                let v = load_u64_xmm(sp.add(j * src_stride));
                store_u64_xmm(dp.add(j * dst_stride), v);
            }
        }
    }

    #[target_feature(enable = "sse2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn avg_pixels(
        dst: &mut [u8],
        dst_stride: usize,
        a: &[u8],
        a_stride: usize,
        b: &[u8],
        b_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        match block_w {
            16 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 16-byte rows.
                    unsafe {
                        let va = _mm_loadu_si128(ap.add(j * a_stride) as *const __m128i);
                        let vb = _mm_loadu_si128(bp.add(j * b_stride) as *const __m128i);
                        _mm_storeu_si128(
                            dp.add(j * dst_stride) as *mut __m128i,
                            _mm_avg_epu8(va, vb),
                        );
                    }
                }
            }
            8 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 8-byte rows.
                    unsafe {
                        let va = load_u64_xmm(ap.add(j * a_stride));
                        let vb = load_u64_xmm(bp.add(j * b_stride));
                        store_u64_xmm(dp.add(j * dst_stride), _mm_avg_epu8(va, vb));
                    }
                }
            }
            4 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 4-byte rows.
                    unsafe {
                        let va = load_u32_xmm(ap.add(j * a_stride));
                        let vb = load_u32_xmm(bp.add(j * b_stride));
                        store_u32_xmm(dp.add(j * dst_stride), _mm_avg_epu8(va, vb));
                    }
                }
            }
            _ => {}
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn avg_pixels_inplace(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        match block_w {
            16 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 16-byte rows.
                    unsafe {
                        let d_row = dp.add(j * dst_stride);
                        let va = _mm_loadu_si128(d_row as *const __m128i);
                        let vb = _mm_loadu_si128(sp.add(j * src_stride) as *const __m128i);
                        _mm_storeu_si128(d_row as *mut __m128i, _mm_avg_epu8(va, vb));
                    }
                }
            }
            8 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 8-byte rows.
                    unsafe {
                        let d_row = dp.add(j * dst_stride);
                        let va = load_u64_xmm(d_row);
                        let vb = load_u64_xmm(sp.add(j * src_stride));
                        store_u64_xmm(d_row, _mm_avg_epu8(va, vb));
                    }
                }
            }
            4 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 4-byte rows.
                    unsafe {
                        let d_row = dp.add(j * dst_stride);
                        let va = load_u32_xmm(d_row);
                        let vb = load_u32_xmm(sp.add(j * src_stride));
                        store_u32_xmm(d_row, _mm_avg_epu8(va, vb));
                    }
                }
            }
            _ => {}
        }
    }

    #[target_feature(enable = "sse2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn chroma_mc_bilinear(
        dst: &mut [u8],
        dst_stride: usize,
        ref_uv: &[u8],
        ref_stride: usize,
        block_w: usize,
        block_h: usize,
        a: i32,
        b: i32,
        c: i32,
        d: i32,
        avg: bool,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = ref_uv.as_ptr();
        let wa = _mm_set1_epi16(a as i16);
        let wb = _mm_set1_epi16(b as i16);
        let wc = _mm_set1_epi16(c as i16);
        let wd = _mm_set1_epi16(d as i16);
        let c32 = _mm_set1_epi16(32);
        let zero = _mm_setzero_si128();

        let chunks = block_w / 8;
        if chunks > 0 {
            for j in 0..block_h {
                for ch in 0..chunks {
                    let x = ch * 8;
                    // SAFETY: Caller verified bounds including `+1` column and `+ref_stride` row.
                    unsafe {
                        let r0 = sp.add(j * ref_stride + x);
                        let r1 = r0.add(ref_stride);
                        let s00 = _mm_unpacklo_epi8(load_u64_xmm(r0), zero);
                        let s10 = _mm_unpacklo_epi8(load_u64_xmm(r0.add(1)), zero);
                        let s01 = _mm_unpacklo_epi8(load_u64_xmm(r1), zero);
                        let s11 = _mm_unpacklo_epi8(load_u64_xmm(r1.add(1)), zero);

                        let mut sum =
                            _mm_add_epi16(_mm_mullo_epi16(s00, wa), _mm_mullo_epi16(s10, wb));
                        sum = _mm_add_epi16(sum, _mm_mullo_epi16(s01, wc));
                        sum = _mm_add_epi16(sum, _mm_mullo_epi16(s11, wd));
                        sum = _mm_srli_epi16::<6>(_mm_add_epi16(sum, c32));
                        let mut val = _mm_packus_epi16(sum, sum);
                        let dst_ptr = dp.add(j * dst_stride + x);
                        if avg {
                            val = _mm_avg_epu8(val, load_u64_xmm(dst_ptr));
                        }
                        store_u64_xmm(dst_ptr, val);
                    }
                }
            }
        } else {
            // block_w == 4
            for j in 0..block_h {
                // SAFETY: Caller verified bounds including `+1` column and `+ref_stride` row.
                unsafe {
                    let r0 = sp.add(j * ref_stride);
                    let r1 = r0.add(ref_stride);
                    let s00 = _mm_unpacklo_epi8(load_u32_xmm(r0), zero);
                    let s10 = _mm_unpacklo_epi8(load_u32_xmm(r0.add(1)), zero);
                    let s01 = _mm_unpacklo_epi8(load_u32_xmm(r1), zero);
                    let s11 = _mm_unpacklo_epi8(load_u32_xmm(r1.add(1)), zero);

                    let mut sum = _mm_add_epi16(_mm_mullo_epi16(s00, wa), _mm_mullo_epi16(s10, wb));
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(s01, wc));
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(s11, wd));
                    sum = _mm_srli_epi16::<6>(_mm_add_epi16(sum, c32));
                    let mut val = _mm_packus_epi16(sum, sum);
                    let dst_ptr = dp.add(j * dst_stride);
                    if avg {
                        val = _mm_avg_epu8(val, load_u32_xmm(dst_ptr));
                    }
                    store_u32_xmm(dst_ptr, val);
                }
            }
        }
    }

    #[target_feature(enable = "sse2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn chroma_mc_2tap(
        dst: &mut [u8],
        dst_stride: usize,
        ref_uv: &[u8],
        ref_stride: usize,
        block_w: usize,
        block_h: usize,
        a: i32,
        e: i32,
        step: usize,
        avg: bool,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = ref_uv.as_ptr();
        let wa = _mm_set1_epi16(a as i16);
        let we = _mm_set1_epi16(e as i16);
        let c32 = _mm_set1_epi16(32);
        let zero = _mm_setzero_si128();

        let chunks = block_w / 8;
        if chunks > 0 {
            for j in 0..block_h {
                for ch in 0..chunks {
                    let x = ch * 8;
                    // SAFETY: Caller verified bounds including `+step`.
                    unsafe {
                        let r0 = sp.add(j * ref_stride + x);
                        let s0 = _mm_unpacklo_epi8(load_u64_xmm(r0), zero);
                        let s1 = _mm_unpacklo_epi8(load_u64_xmm(r0.add(step)), zero);

                        let mut sum =
                            _mm_add_epi16(_mm_mullo_epi16(s0, wa), _mm_mullo_epi16(s1, we));
                        sum = _mm_srli_epi16::<6>(_mm_add_epi16(sum, c32));
                        let mut val = _mm_packus_epi16(sum, sum);
                        let dst_ptr = dp.add(j * dst_stride + x);
                        if avg {
                            val = _mm_avg_epu8(val, load_u64_xmm(dst_ptr));
                        }
                        store_u64_xmm(dst_ptr, val);
                    }
                }
            }
        } else {
            for j in 0..block_h {
                // SAFETY: Caller verified bounds including `+step`.
                unsafe {
                    let r0 = sp.add(j * ref_stride);
                    let s0 = _mm_unpacklo_epi8(load_u32_xmm(r0), zero);
                    let s1 = _mm_unpacklo_epi8(load_u32_xmm(r0.add(step)), zero);

                    let mut sum = _mm_add_epi16(_mm_mullo_epi16(s0, wa), _mm_mullo_epi16(s1, we));
                    sum = _mm_srli_epi16::<6>(_mm_add_epi16(sum, c32));
                    let mut val = _mm_packus_epi16(sum, sum);
                    let dst_ptr = dp.add(j * dst_stride);
                    if avg {
                        val = _mm_avg_epu8(val, load_u32_xmm(dst_ptr));
                    }
                    store_u32_xmm(dst_ptr, val);
                }
            }
        }
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn fir6_i16(
        a: __m128i,
        b: __m128i,
        c: __m128i,
        d: __m128i,
        e: __m128i,
        f: __m128i,
    ) -> __m128i {
        let c5 = _mm_set1_epi16(5);
        let c20 = _mm_set1_epi16(20);
        let af = _mm_add_epi16(a, f);
        let be = _mm_mullo_epi16(_mm_add_epi16(b, e), c5);
        let cd = _mm_mullo_epi16(_mm_add_epi16(c, d), c20);
        _mm_sub_epi16(_mm_add_epi16(af, cd), be)
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn luma_6tap_h(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let zero = _mm_setzero_si128();
        let c16 = _mm_set1_epi16(16);

        let chunks = block_w / 8;
        if chunks > 0 {
            for j in 0..block_h {
                for ch in 0..chunks {
                    let x = ch * 8;
                    // SAFETY: Caller verified `block_w + 5` bytes per row in `src`.
                    unsafe {
                        let r = sp.add(j * src_stride + x);
                        let a = _mm_unpacklo_epi8(load_u64_xmm(r), zero);
                        let b = _mm_unpacklo_epi8(load_u64_xmm(r.add(1)), zero);
                        let c = _mm_unpacklo_epi8(load_u64_xmm(r.add(2)), zero);
                        let d = _mm_unpacklo_epi8(load_u64_xmm(r.add(3)), zero);
                        let e = _mm_unpacklo_epi8(load_u64_xmm(r.add(4)), zero);
                        let f = _mm_unpacklo_epi8(load_u64_xmm(r.add(5)), zero);
                        let sum = _mm_add_epi16(fir6_i16(a, b, c, d, e, f), c16);
                        let out = _mm_packus_epi16(_mm_srai_epi16::<5>(sum), zero);
                        store_u64_xmm(dp.add(j * dst_stride + x), out);
                    }
                }
            }
        } else {
            for j in 0..block_h {
                // SAFETY: Caller verified `4 + 5 = 9` bytes per row in `src`.
                unsafe {
                    let r = sp.add(j * src_stride);
                    let a = _mm_unpacklo_epi8(load_u32_xmm(r), zero);
                    let b = _mm_unpacklo_epi8(load_u32_xmm(r.add(1)), zero);
                    let c = _mm_unpacklo_epi8(load_u32_xmm(r.add(2)), zero);
                    let d = _mm_unpacklo_epi8(load_u32_xmm(r.add(3)), zero);
                    let e = _mm_unpacklo_epi8(load_u32_xmm(r.add(4)), zero);
                    let f = _mm_unpacklo_epi8(load_u32_xmm(r.add(5)), zero);
                    let sum = _mm_add_epi16(fir6_i16(a, b, c, d, e, f), c16);
                    let out = _mm_packus_epi16(_mm_srai_epi16::<5>(sum), zero);
                    store_u32_xmm(dp.add(j * dst_stride), out);
                }
            }
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn luma_6tap_v(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let zero = _mm_setzero_si128();
        let c16 = _mm_set1_epi16(16);

        let chunks = block_w / 8;
        if chunks > 0 {
            for ch in 0..chunks {
                let x = ch * 8;
                // SAFETY: Caller verified `(block_h + 4) * src_stride + block_w <= src.len()`.
                unsafe {
                    let col_sp = sp.add(x);
                    let mut r0 = _mm_unpacklo_epi8(load_u64_xmm(col_sp), zero);
                    let mut r1 = _mm_unpacklo_epi8(load_u64_xmm(col_sp.add(src_stride)), zero);
                    let mut r2 = _mm_unpacklo_epi8(load_u64_xmm(col_sp.add(2 * src_stride)), zero);
                    let mut r3 = _mm_unpacklo_epi8(load_u64_xmm(col_sp.add(3 * src_stride)), zero);
                    let mut r4 = _mm_unpacklo_epi8(load_u64_xmm(col_sp.add(4 * src_stride)), zero);

                    for j in 0..block_h {
                        let r5 =
                            _mm_unpacklo_epi8(load_u64_xmm(col_sp.add((j + 5) * src_stride)), zero);
                        let sum = _mm_add_epi16(fir6_i16(r0, r1, r2, r3, r4, r5), c16);
                        let out = _mm_packus_epi16(_mm_srai_epi16::<5>(sum), zero);
                        store_u64_xmm(dp.add(j * dst_stride + x), out);
                        r0 = r1;
                        r1 = r2;
                        r2 = r3;
                        r3 = r4;
                        r4 = r5;
                    }
                }
            }
        } else {
            // SAFETY: Caller verified `(block_h + 4) * src_stride + 4 <= src.len()`.
            unsafe {
                let mut r0 = _mm_unpacklo_epi8(load_u32_xmm(sp), zero);
                let mut r1 = _mm_unpacklo_epi8(load_u32_xmm(sp.add(src_stride)), zero);
                let mut r2 = _mm_unpacklo_epi8(load_u32_xmm(sp.add(2 * src_stride)), zero);
                let mut r3 = _mm_unpacklo_epi8(load_u32_xmm(sp.add(3 * src_stride)), zero);
                let mut r4 = _mm_unpacklo_epi8(load_u32_xmm(sp.add(4 * src_stride)), zero);

                for j in 0..block_h {
                    let r5 = _mm_unpacklo_epi8(load_u32_xmm(sp.add((j + 5) * src_stride)), zero);
                    let sum = _mm_add_epi16(fir6_i16(r0, r1, r2, r3, r4, r5), c16);
                    let out = _mm_packus_epi16(_mm_srai_epi16::<5>(sum), zero);
                    store_u32_xmm(dp.add(j * dst_stride), out);
                    r0 = r1;
                    r1 = r2;
                    r2 = r3;
                    r3 = r4;
                    r4 = r5;
                }
            }
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn luma_6tap_hv(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
        tmp: &mut [i32],
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let tp = tmp.as_mut_ptr();
        let zero = _mm_setzero_si128();
        let tmp_h = block_h + 5;

        // Pass 1: horizontal 6-tap into i32 tmp (unshifted).
        let chunks8 = block_w / 8;
        if chunks8 > 0 {
            for j in 0..tmp_h {
                for ch in 0..chunks8 {
                    let x = ch * 8;
                    // SAFETY: Caller verified `src` and `tmp` bounds.
                    unsafe {
                        let r = sp.add(j * src_stride + x);
                        let a = _mm_unpacklo_epi8(load_u64_xmm(r), zero);
                        let b = _mm_unpacklo_epi8(load_u64_xmm(r.add(1)), zero);
                        let c = _mm_unpacklo_epi8(load_u64_xmm(r.add(2)), zero);
                        let d = _mm_unpacklo_epi8(load_u64_xmm(r.add(3)), zero);
                        let e = _mm_unpacklo_epi8(load_u64_xmm(r.add(4)), zero);
                        let f = _mm_unpacklo_epi8(load_u64_xmm(r.add(5)), zero);
                        let v16 = fir6_i16(a, b, c, d, e, f);
                        let lo32 = _mm_srai_epi32::<16>(_mm_unpacklo_epi16(v16, v16));
                        let hi32 = _mm_srai_epi32::<16>(_mm_unpackhi_epi16(v16, v16));
                        let out_p = tp.add(j * block_w + x);
                        _mm_storeu_si128(out_p as *mut __m128i, lo32);
                        _mm_storeu_si128(out_p.add(4) as *mut __m128i, hi32);
                    }
                }
            }
        } else {
            for j in 0..tmp_h {
                // SAFETY: Caller verified `src` and `tmp` bounds for `block_w == 4`.
                unsafe {
                    let r = sp.add(j * src_stride);
                    let a = _mm_unpacklo_epi8(load_u32_xmm(r), zero);
                    let b = _mm_unpacklo_epi8(load_u32_xmm(r.add(1)), zero);
                    let c = _mm_unpacklo_epi8(load_u32_xmm(r.add(2)), zero);
                    let d = _mm_unpacklo_epi8(load_u32_xmm(r.add(3)), zero);
                    let e = _mm_unpacklo_epi8(load_u32_xmm(r.add(4)), zero);
                    let f = _mm_unpacklo_epi8(load_u32_xmm(r.add(5)), zero);
                    let v16 = fir6_i16(a, b, c, d, e, f);
                    let lo32 = _mm_srai_epi32::<16>(_mm_unpacklo_epi16(v16, v16));
                    _mm_storeu_si128(tp.add(j * block_w) as *mut __m128i, lo32);
                }
            }
        }

        // Pass 2: vertical 6-tap in i32 on `tmp` (4 columns per XMM).
        let c512 = _mm_set1_epi32(512);
        let chunks4 = block_w / 4;
        for ch in 0..chunks4 {
            let x = ch * 4;
            // SAFETY: `tp` has `(block_h + 5) * block_w` i32s; `dp` has `block_h` rows.
            unsafe {
                let col_tp = tp.add(x);
                let mut r0 = _mm_loadu_si128(col_tp as *const __m128i);
                let mut r1 = _mm_loadu_si128(col_tp.add(block_w) as *const __m128i);
                let mut r2 = _mm_loadu_si128(col_tp.add(2 * block_w) as *const __m128i);
                let mut r3 = _mm_loadu_si128(col_tp.add(3 * block_w) as *const __m128i);
                let mut r4 = _mm_loadu_si128(col_tp.add(4 * block_w) as *const __m128i);
                for j in 0..block_h {
                    let r5 = _mm_loadu_si128(col_tp.add((j + 5) * block_w) as *const __m128i);
                    let af = _mm_add_epi32(r0, r5);
                    let be = _mm_add_epi32(r1, r4);
                    let be5 = _mm_add_epi32(be, _mm_slli_epi32::<2>(be));
                    let cd = _mm_add_epi32(r2, r3);
                    let cd20 = _mm_add_epi32(_mm_slli_epi32::<4>(cd), _mm_slli_epi32::<2>(cd));
                    let sum = _mm_add_epi32(_mm_sub_epi32(_mm_add_epi32(af, cd20), be5), c512);
                    let shifted = _mm_srai_epi32::<10>(sum);
                    let p16 = _mm_packs_epi32(shifted, shifted);
                    let p8 = _mm_packus_epi16(p16, p16);
                    store_u32_xmm(dp.add(j * dst_stride + x), p8);
                    r0 = r1;
                    r1 = r2;
                    r2 = r3;
                    r3 = r4;
                    r4 = r5;
                }
            }
        }
    }
}

// ===========================================================================
// x86_64 AVX2 kernels
// ===========================================================================

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86_avx2 {
    use core::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn load_u32_xmm(p: *const u8) -> __m128i {
        // SAFETY: Caller guarantees 4 readable bytes at `p`.
        let v = unsafe { core::ptr::read_unaligned(p as *const u32) };
        _mm_cvtsi32_si128(v as i32)
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn store_u32_xmm(p: *mut u8, v: __m128i) {
        let bits = _mm_cvtsi128_si32(v) as u32;
        // SAFETY: Caller guarantees 4 writable bytes at `p`.
        unsafe { core::ptr::write_unaligned(p as *mut u32, bits) };
    }

    #[inline(always)]
    unsafe fn load_u64_xmm(p: *const u8) -> __m128i {
        // SAFETY: Caller guarantees 8 readable bytes at `p`.
        unsafe { _mm_loadl_epi64(p as *const __m128i) }
    }

    #[inline(always)]
    unsafe fn store_u64_xmm(p: *mut u8, v: __m128i) {
        // SAFETY: Caller guarantees 8 writable bytes at `p`.
        unsafe { _mm_storel_epi64(p as *mut __m128i, v) };
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn transpose_8x8_i16(r: &mut [__m128i; 8]) {
        let t0 = _mm_unpacklo_epi16(r[0], r[1]);
        let t1 = _mm_unpackhi_epi16(r[0], r[1]);
        let t2 = _mm_unpacklo_epi16(r[2], r[3]);
        let t3 = _mm_unpackhi_epi16(r[2], r[3]);
        let t4 = _mm_unpacklo_epi16(r[4], r[5]);
        let t5 = _mm_unpackhi_epi16(r[4], r[5]);
        let t6 = _mm_unpacklo_epi16(r[6], r[7]);
        let t7 = _mm_unpackhi_epi16(r[6], r[7]);

        let u0 = _mm_unpacklo_epi32(t0, t2);
        let u1 = _mm_unpackhi_epi32(t0, t2);
        let u2 = _mm_unpacklo_epi32(t1, t3);
        let u3 = _mm_unpackhi_epi32(t1, t3);
        let u4 = _mm_unpacklo_epi32(t4, t6);
        let u5 = _mm_unpackhi_epi32(t4, t6);
        let u6 = _mm_unpacklo_epi32(t5, t7);
        let u7 = _mm_unpackhi_epi32(t5, t7);

        r[0] = _mm_unpacklo_epi64(u0, u4);
        r[1] = _mm_unpackhi_epi64(u0, u4);
        r[2] = _mm_unpacklo_epi64(u1, u5);
        r[3] = _mm_unpackhi_epi64(u1, u5);
        r[4] = _mm_unpacklo_epi64(u2, u6);
        r[5] = _mm_unpackhi_epi64(u2, u6);
        r[6] = _mm_unpacklo_epi64(u3, u7);
        r[7] = _mm_unpackhi_epi64(u3, u7);
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn idct_add_4x4(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 16]) {
        // VEX-encoded 128-bit execution avoids AVX->SSE transition penalties.
        let cp = coeffs.as_ptr();
        // SAFETY: `coeffs` has 16 `i16` elements.
        let mut r0 = unsafe { _mm_loadl_epi64(cp as *const __m128i) };
        let r1 = unsafe { _mm_loadl_epi64(cp.add(4) as *const __m128i) };
        let r2 = unsafe { _mm_loadl_epi64(cp.add(8) as *const __m128i) };
        let r3 = unsafe { _mm_loadl_epi64(cp.add(12) as *const __m128i) };

        r0 = _mm_add_epi16(r0, _mm_setr_epi16(32, 0, 0, 0, 0, 0, 0, 0));

        let t0 = _mm_unpacklo_epi16(r0, r1);
        let t1 = _mm_unpacklo_epi16(r2, r3);
        let c0 = _mm_unpacklo_epi32(t0, t1);
        let c2 = _mm_unpackhi_epi32(t0, t1);
        let c1 = _mm_srli_si128::<8>(c0);
        let c3 = _mm_srli_si128::<8>(c2);

        let z0 = _mm_add_epi16(c0, c2);
        let z1 = _mm_sub_epi16(c0, c2);
        let z2 = _mm_sub_epi16(_mm_srai_epi16::<1>(c1), c3);
        let z3 = _mm_add_epi16(c1, _mm_srai_epi16::<1>(c3));

        let o0 = _mm_add_epi16(z0, z3);
        let o1 = _mm_add_epi16(z1, z2);
        let o2 = _mm_sub_epi16(z1, z2);
        let o3 = _mm_sub_epi16(z0, z3);

        let u0 = _mm_unpacklo_epi16(o0, o1);
        let u1 = _mm_unpacklo_epi16(o2, o3);
        let row01 = _mm_unpacklo_epi32(u0, u1);
        let row23 = _mm_unpackhi_epi32(u0, u1);

        let p0 = _mm_cvtepi16_epi32(row01);
        let p1 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(row01));
        let p2 = _mm_cvtepi16_epi32(row23);
        let p3 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(row23));

        let cz0 = _mm_add_epi32(p0, p2);
        let cz1 = _mm_sub_epi32(p0, p2);
        let cz2 = _mm_sub_epi32(_mm_srai_epi32::<1>(p1), p3);
        let cz3 = _mm_add_epi32(p1, _mm_srai_epi32::<1>(p3));

        let d0 = _mm_srai_epi32::<6>(_mm_add_epi32(cz0, cz3));
        let d1 = _mm_srai_epi32::<6>(_mm_add_epi32(cz1, cz2));
        let d2 = _mm_srai_epi32::<6>(_mm_sub_epi32(cz1, cz2));
        let d3 = _mm_srai_epi32::<6>(_mm_sub_epi32(cz0, cz3));

        let dp = dst.as_mut_ptr();
        // SAFETY: Caller verified `3 * stride + 3 < dst.len()` and `stride >= 4`.
        let (x0, x1, x2, x3) = unsafe {
            (
                _mm_cvtepu8_epi32(load_u32_xmm(dp)),
                _mm_cvtepu8_epi32(load_u32_xmm(dp.add(stride))),
                _mm_cvtepu8_epi32(load_u32_xmm(dp.add(2 * stride))),
                _mm_cvtepu8_epi32(load_u32_xmm(dp.add(3 * stride))),
            )
        };

        let sum01 = _mm_packs_epi32(_mm_add_epi32(x0, d0), _mm_add_epi32(x1, d1));
        let sum23 = _mm_packs_epi32(_mm_add_epi32(x2, d2), _mm_add_epi32(x3, d3));
        let packed = _mm_packus_epi16(sum01, sum23);

        // SAFETY: Disjoint 4-byte writes within `dst`.
        unsafe {
            store_u32_xmm(dp, packed);
            store_u32_xmm(dp.add(stride), _mm_shuffle_epi32::<0x55>(packed));
            store_u32_xmm(dp.add(2 * stride), _mm_shuffle_epi32::<0xAA>(packed));
            store_u32_xmm(dp.add(3 * stride), _mm_shuffle_epi32::<0xFF>(packed));
        }

        *coeffs = [0i16; 16];
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn idct_dc_add_4x4(dst: &mut [u8], stride: usize, dc: &mut i16) {
        let val = (*dc as i32 + 32) >> 6;
        *dc = 0;
        if val == 0 {
            return;
        }
        let dp = dst.as_mut_ptr();
        if val > 0 {
            let add = _mm_set1_epi8(val.min(255) as i8);
            for j in 0..4 {
                // SAFETY: `3 * stride + 3 < dst.len()` verified by caller.
                unsafe {
                    let row = dp.add(j * stride);
                    store_u32_xmm(row, _mm_adds_epu8(load_u32_xmm(row), add));
                }
            }
        } else {
            let sub = _mm_set1_epi8((-val).min(255) as i8);
            for j in 0..4 {
                // SAFETY: `3 * stride + 3 < dst.len()` verified by caller.
                unsafe {
                    let row = dp.add(j * stride);
                    store_u32_xmm(row, _mm_subs_epu8(load_u32_xmm(row), sub));
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn idct_dc_add_8x8(dst: &mut [u8], stride: usize, dc: &mut i16) {
        let val = (*dc as i32 + 32) >> 6;
        *dc = 0;
        if val == 0 {
            return;
        }
        let dp = dst.as_mut_ptr();
        if val > 0 {
            let add = _mm_set1_epi8(val.min(255) as i8);
            for j in 0..8 {
                // SAFETY: `7 * stride + 7 < dst.len()` verified by caller.
                unsafe {
                    let row = dp.add(j * stride);
                    store_u64_xmm(row, _mm_adds_epu8(load_u64_xmm(row), add));
                }
            }
        } else {
            let sub = _mm_set1_epi8((-val).min(255) as i8);
            for j in 0..8 {
                // SAFETY: `7 * stride + 7 < dst.len()` verified by caller.
                unsafe {
                    let row = dp.add(j * stride);
                    store_u64_xmm(row, _mm_subs_epu8(load_u64_xmm(row), sub));
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn idct8_butterfly_8(v: [__m256i; 8]) -> [__m256i; 8] {
        let a0 = _mm256_add_epi32(v[0], v[4]);
        let a2 = _mm256_sub_epi32(v[0], v[4]);
        let a4 = _mm256_sub_epi32(_mm256_srai_epi32::<1>(v[2]), v[6]);
        let a6 = _mm256_add_epi32(_mm256_srai_epi32::<1>(v[6]), v[2]);

        let b0 = _mm256_add_epi32(a0, a6);
        let b2 = _mm256_add_epi32(a2, a4);
        let b4 = _mm256_sub_epi32(a2, a4);
        let b6 = _mm256_sub_epi32(a0, a6);

        let a1 = _mm256_sub_epi32(
            _mm256_sub_epi32(_mm256_sub_epi32(v[5], v[3]), v[7]),
            _mm256_srai_epi32::<1>(v[7]),
        );
        let a3 = _mm256_sub_epi32(
            _mm256_sub_epi32(_mm256_add_epi32(v[1], v[7]), v[3]),
            _mm256_srai_epi32::<1>(v[3]),
        );
        let a5 = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_sub_epi32(v[7], v[1]), v[5]),
            _mm256_srai_epi32::<1>(v[5]),
        );
        let a7 = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(v[3], v[5]), v[1]),
            _mm256_srai_epi32::<1>(v[1]),
        );

        let b1 = _mm256_add_epi32(_mm256_srai_epi32::<2>(a7), a1);
        let b3 = _mm256_add_epi32(a3, _mm256_srai_epi32::<2>(a5));
        let b5 = _mm256_sub_epi32(_mm256_srai_epi32::<2>(a3), a5);
        let b7 = _mm256_sub_epi32(a7, _mm256_srai_epi32::<2>(a1));

        [
            _mm256_add_epi32(b0, b7),
            _mm256_add_epi32(b2, b5),
            _mm256_add_epi32(b4, b3),
            _mm256_add_epi32(b6, b1),
            _mm256_sub_epi32(b6, b1),
            _mm256_sub_epi32(b4, b3),
            _mm256_sub_epi32(b2, b5),
            _mm256_sub_epi32(b0, b7),
        ]
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn idct_add_8x8(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 64]) {
        let cp = coeffs.as_ptr();
        // SAFETY: `coeffs` has 64 `i16` elements (8 rows of 8).
        let mut r = unsafe {
            [
                _mm_loadu_si128(cp as *const __m128i),
                _mm_loadu_si128(cp.add(8) as *const __m128i),
                _mm_loadu_si128(cp.add(16) as *const __m128i),
                _mm_loadu_si128(cp.add(24) as *const __m128i),
                _mm_loadu_si128(cp.add(32) as *const __m128i),
                _mm_loadu_si128(cp.add(40) as *const __m128i),
                _mm_loadu_si128(cp.add(48) as *const __m128i),
                _mm_loadu_si128(cp.add(56) as *const __m128i),
            ]
        };
        r[0] = _mm_add_epi16(r[0], _mm_setr_epi16(32, 0, 0, 0, 0, 0, 0, 0));

        // Transpose so each r[k] holds column k across all 8 rows.
        // SAFETY: Pure register SIMD transpose.
        unsafe { transpose_8x8_i16(&mut r) };

        // Pass 1 in AVX2 i32 (all 8 rows simultaneously per column).
        let mut v_in = [_mm256_setzero_si256(); 8];
        for k in 0..8 {
            v_in[k] = _mm256_cvtepi16_epi32(r[k]);
        }
        // SAFETY: AVX2 enabled.
        let v_out = unsafe { idct8_butterfly_8(v_in) };

        // Truncate wrapping i32 -> i16 matching scalar `as i16`.
        let mut mid = [_mm_setzero_si128(); 8];
        for k in 0..8 {
            let s = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(v_out[k]));
            let p = _mm256_packs_epi32(s, s);
            let lo = _mm256_castsi256_si128(p);
            let hi = _mm256_extracti128_si256::<1>(p);
            mid[k] = _mm_unpacklo_epi64(lo, hi);
        }

        // Transpose back so mid[j] holds row j across all 8 columns.
        // SAFETY: Pure register SIMD transpose.
        unsafe { transpose_8x8_i16(&mut mid) };

        // Pass 2 in AVX2 i32 (all 8 columns simultaneously per row).
        for k in 0..8 {
            v_in[k] = _mm256_cvtepi16_epi32(mid[k]);
        }
        // SAFETY: AVX2 enabled.
        let res = unsafe { idct8_butterfly_8(v_in) };

        let dp = dst.as_mut_ptr();
        for (j, &res_row) in res.iter().enumerate() {
            // SAFETY: Caller verified `7 * stride + 7 < dst.len()` and `stride >= 8`.
            unsafe {
                let row_ptr = dp.add(j * stride);
                let src8 = load_u64_xmm(row_ptr);
                let src32 = _mm256_cvtepu8_epi32(src8);
                let sum = _mm256_add_epi32(src32, _mm256_srai_epi32::<6>(res_row));
                let p = _mm256_packs_epi32(sum, sum);
                let lo = _mm256_castsi256_si128(p);
                let hi = _mm256_extracti128_si256::<1>(p);
                let row16 = _mm_unpacklo_epi64(lo, hi);
                let out8 = _mm_packus_epi16(row16, row16);
                store_u64_xmm(row_ptr, out8);
            }
        }

        *coeffs = [0i16; 64];
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn put_pixels_16(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        for j in 0..block_h {
            // SAFETY: Caller verified bounds for `block_h` rows of 16 bytes.
            unsafe {
                let v = _mm_loadu_si128(sp.add(j * src_stride) as *const __m128i);
                _mm_storeu_si128(dp.add(j * dst_stride) as *mut __m128i, v);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn put_pixels_8(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        for j in 0..block_h {
            // SAFETY: Caller verified bounds for `block_h` rows of 8 bytes.
            unsafe {
                let v = load_u64_xmm(sp.add(j * src_stride));
                store_u64_xmm(dp.add(j * dst_stride), v);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn avg_pixels(
        dst: &mut [u8],
        dst_stride: usize,
        a: &[u8],
        a_stride: usize,
        b: &[u8],
        b_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        match block_w {
            16 => {
                let mut j = 0;
                while j + 1 < block_h {
                    // SAFETY: Caller verified bounds for `block_h` rows of 16 bytes.
                    unsafe {
                        let a0 = _mm_loadu_si128(ap.add(j * a_stride) as *const __m128i);
                        let a1 = _mm_loadu_si128(ap.add((j + 1) * a_stride) as *const __m128i);
                        let b0 = _mm_loadu_si128(bp.add(j * b_stride) as *const __m128i);
                        let b1 = _mm_loadu_si128(bp.add((j + 1) * b_stride) as *const __m128i);
                        let va = _mm256_set_m128i(a1, a0);
                        let vb = _mm256_set_m128i(b1, b0);
                        let avg = _mm256_avg_epu8(va, vb);
                        _mm_storeu_si128(
                            dp.add(j * dst_stride) as *mut __m128i,
                            _mm256_castsi256_si128(avg),
                        );
                        _mm_storeu_si128(
                            dp.add((j + 1) * dst_stride) as *mut __m128i,
                            _mm256_extracti128_si256::<1>(avg),
                        );
                    }
                    j += 2;
                }
                if j < block_h {
                    // SAFETY: Caller verified bounds for `block_h` rows of 16 bytes.
                    unsafe {
                        let va = _mm_loadu_si128(ap.add(j * a_stride) as *const __m128i);
                        let vb = _mm_loadu_si128(bp.add(j * b_stride) as *const __m128i);
                        _mm_storeu_si128(
                            dp.add(j * dst_stride) as *mut __m128i,
                            _mm_avg_epu8(va, vb),
                        );
                    }
                }
            }
            8 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 8-byte rows.
                    unsafe {
                        let va = load_u64_xmm(ap.add(j * a_stride));
                        let vb = load_u64_xmm(bp.add(j * b_stride));
                        store_u64_xmm(dp.add(j * dst_stride), _mm_avg_epu8(va, vb));
                    }
                }
            }
            4 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 4-byte rows.
                    unsafe {
                        let va = load_u32_xmm(ap.add(j * a_stride));
                        let vb = load_u32_xmm(bp.add(j * b_stride));
                        store_u32_xmm(dp.add(j * dst_stride), _mm_avg_epu8(va, vb));
                    }
                }
            }
            _ => {}
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avg_pixels_inplace(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        match block_w {
            16 => {
                let mut j = 0;
                while j + 1 < block_h {
                    // SAFETY: Caller verified bounds for `block_h` rows of 16 bytes.
                    unsafe {
                        let d0_p = dp.add(j * dst_stride);
                        let d1_p = dp.add((j + 1) * dst_stride);
                        let d0 = _mm_loadu_si128(d0_p as *const __m128i);
                        let d1 = _mm_loadu_si128(d1_p as *const __m128i);
                        let s0 = _mm_loadu_si128(sp.add(j * src_stride) as *const __m128i);
                        let s1 = _mm_loadu_si128(sp.add((j + 1) * src_stride) as *const __m128i);
                        let vd = _mm256_set_m128i(d1, d0);
                        let vs = _mm256_set_m128i(s1, s0);
                        let avg = _mm256_avg_epu8(vd, vs);
                        _mm_storeu_si128(d0_p as *mut __m128i, _mm256_castsi256_si128(avg));
                        _mm_storeu_si128(d1_p as *mut __m128i, _mm256_extracti128_si256::<1>(avg));
                    }
                    j += 2;
                }
                if j < block_h {
                    // SAFETY: Caller verified bounds for `block_h` rows of 16 bytes.
                    unsafe {
                        let d_p = dp.add(j * dst_stride);
                        let vd = _mm_loadu_si128(d_p as *const __m128i);
                        let vs = _mm_loadu_si128(sp.add(j * src_stride) as *const __m128i);
                        _mm_storeu_si128(d_p as *mut __m128i, _mm_avg_epu8(vd, vs));
                    }
                }
            }
            8 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 8-byte rows.
                    unsafe {
                        let d_p = dp.add(j * dst_stride);
                        let vd = load_u64_xmm(d_p);
                        let vs = load_u64_xmm(sp.add(j * src_stride));
                        store_u64_xmm(d_p, _mm_avg_epu8(vd, vs));
                    }
                }
            }
            4 => {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds for 4-byte rows.
                    unsafe {
                        let d_p = dp.add(j * dst_stride);
                        let vd = load_u32_xmm(d_p);
                        let vs = load_u32_xmm(sp.add(j * src_stride));
                        store_u32_xmm(d_p, _mm_avg_epu8(vd, vs));
                    }
                }
            }
            _ => {}
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn chroma_mc_bilinear(
        dst: &mut [u8],
        dst_stride: usize,
        ref_uv: &[u8],
        ref_stride: usize,
        block_w: usize,
        block_h: usize,
        a: i32,
        b: i32,
        c: i32,
        d: i32,
        avg: bool,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = ref_uv.as_ptr();
        let wa = _mm256_set1_epi16(a as i16);
        let wb = _mm256_set1_epi16(b as i16);
        let wc = _mm256_set1_epi16(c as i16);
        let wd = _mm256_set1_epi16(d as i16);
        let c32 = _mm256_set1_epi16(32);

        if block_w == 16 {
            for j in 0..block_h {
                // SAFETY: Caller verified bounds including `+1` col and `+ref_stride` row.
                unsafe {
                    let r0 = sp.add(j * ref_stride);
                    let r1 = r0.add(ref_stride);
                    let s00 = _mm256_cvtepu8_epi16(_mm_loadu_si128(r0 as *const __m128i));
                    let s10 = _mm256_cvtepu8_epi16(_mm_loadu_si128(r0.add(1) as *const __m128i));
                    let s01 = _mm256_cvtepu8_epi16(_mm_loadu_si128(r1 as *const __m128i));
                    let s11 = _mm256_cvtepu8_epi16(_mm_loadu_si128(r1.add(1) as *const __m128i));

                    let mut sum =
                        _mm256_add_epi16(_mm256_mullo_epi16(s00, wa), _mm256_mullo_epi16(s10, wb));
                    sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(s01, wc));
                    sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(s11, wd));
                    sum = _mm256_srli_epi16::<6>(_mm256_add_epi16(sum, c32));
                    let mut val = _mm_packus_epi16(
                        _mm256_castsi256_si128(sum),
                        _mm256_extracti128_si256::<1>(sum),
                    );
                    let dst_ptr = dp.add(j * dst_stride);
                    if avg {
                        val = _mm_avg_epu8(val, _mm_loadu_si128(dst_ptr as *const __m128i));
                    }
                    _mm_storeu_si128(dst_ptr as *mut __m128i, val);
                }
            }
        } else if block_w == 8 {
            // Process 2 rows at a time in AVX2 (16 pixels across two 8-pixel rows).
            let mut j = 0;
            while j + 1 < block_h {
                // SAFETY: Caller verified bounds for `block_h` rows including `+1` col and `+ref_stride` row.
                unsafe {
                    let r0a = sp.add(j * ref_stride);
                    let r1a = r0a.add(ref_stride);
                    let r0b = r1a;
                    let r1b = r0b.add(ref_stride);

                    let p00 = _mm_unpacklo_epi64(load_u64_xmm(r0a), load_u64_xmm(r0b));
                    let p10 =
                        _mm_unpacklo_epi64(load_u64_xmm(r0a.add(1)), load_u64_xmm(r0b.add(1)));
                    let p01 = _mm_unpacklo_epi64(load_u64_xmm(r1a), load_u64_xmm(r1b));
                    let p11 =
                        _mm_unpacklo_epi64(load_u64_xmm(r1a.add(1)), load_u64_xmm(r1b.add(1)));

                    let mut sum = _mm256_add_epi16(
                        _mm256_mullo_epi16(_mm256_cvtepu8_epi16(p00), wa),
                        _mm256_mullo_epi16(_mm256_cvtepu8_epi16(p10), wb),
                    );
                    sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(p01), wc));
                    sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(p11), wd));
                    sum = _mm256_srli_epi16::<6>(_mm256_add_epi16(sum, c32));
                    let mut val = _mm_packus_epi16(
                        _mm256_castsi256_si128(sum),
                        _mm256_extracti128_si256::<1>(sum),
                    );
                    let dst_a = dp.add(j * dst_stride);
                    let dst_b = dp.add((j + 1) * dst_stride);
                    if avg {
                        let d_pair = _mm_unpacklo_epi64(load_u64_xmm(dst_a), load_u64_xmm(dst_b));
                        val = _mm_avg_epu8(val, d_pair);
                    }
                    store_u64_xmm(dst_a, val);
                    store_u64_xmm(dst_b, _mm_srli_si128::<8>(val));
                }
                j += 2;
            }
            if j < block_h {
                // SAFETY: Caller verified bounds including `+1` col and `+ref_stride` row.
                unsafe {
                    let r0 = sp.add(j * ref_stride);
                    let r1 = r0.add(ref_stride);
                    let wa128 = _mm256_castsi256_si128(wa);
                    let wb128 = _mm256_castsi256_si128(wb);
                    let wc128 = _mm256_castsi256_si128(wc);
                    let wd128 = _mm256_castsi256_si128(wd);
                    let c32_128 = _mm256_castsi256_si128(c32);
                    let zero = _mm_setzero_si128();
                    let s00 = _mm_unpacklo_epi8(load_u64_xmm(r0), zero);
                    let s10 = _mm_unpacklo_epi8(load_u64_xmm(r0.add(1)), zero);
                    let s01 = _mm_unpacklo_epi8(load_u64_xmm(r1), zero);
                    let s11 = _mm_unpacklo_epi8(load_u64_xmm(r1.add(1)), zero);
                    let mut sum =
                        _mm_add_epi16(_mm_mullo_epi16(s00, wa128), _mm_mullo_epi16(s10, wb128));
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(s01, wc128));
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(s11, wd128));
                    sum = _mm_srli_epi16::<6>(_mm_add_epi16(sum, c32_128));
                    let mut val = _mm_packus_epi16(sum, sum);
                    let dst_ptr = dp.add(j * dst_stride);
                    if avg {
                        val = _mm_avg_epu8(val, load_u64_xmm(dst_ptr));
                    }
                    store_u64_xmm(dst_ptr, val);
                }
            }
        } else {
            // block_w == 4 (VEX 128-bit).
            let wa128 = _mm256_castsi256_si128(wa);
            let wb128 = _mm256_castsi256_si128(wb);
            let wc128 = _mm256_castsi256_si128(wc);
            let wd128 = _mm256_castsi256_si128(wd);
            let c32_128 = _mm256_castsi256_si128(c32);
            let zero = _mm_setzero_si128();
            for j in 0..block_h {
                // SAFETY: Caller verified bounds including `+1` col and `+ref_stride` row.
                unsafe {
                    let r0 = sp.add(j * ref_stride);
                    let r1 = r0.add(ref_stride);
                    let s00 = _mm_unpacklo_epi8(load_u32_xmm(r0), zero);
                    let s10 = _mm_unpacklo_epi8(load_u32_xmm(r0.add(1)), zero);
                    let s01 = _mm_unpacklo_epi8(load_u32_xmm(r1), zero);
                    let s11 = _mm_unpacklo_epi8(load_u32_xmm(r1.add(1)), zero);
                    let mut sum =
                        _mm_add_epi16(_mm_mullo_epi16(s00, wa128), _mm_mullo_epi16(s10, wb128));
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(s01, wc128));
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(s11, wd128));
                    sum = _mm_srli_epi16::<6>(_mm_add_epi16(sum, c32_128));
                    let mut val = _mm_packus_epi16(sum, sum);
                    let dst_ptr = dp.add(j * dst_stride);
                    if avg {
                        val = _mm_avg_epu8(val, load_u32_xmm(dst_ptr));
                    }
                    store_u32_xmm(dst_ptr, val);
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn chroma_mc_2tap(
        dst: &mut [u8],
        dst_stride: usize,
        ref_uv: &[u8],
        ref_stride: usize,
        block_w: usize,
        block_h: usize,
        a: i32,
        e: i32,
        step: usize,
        avg: bool,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = ref_uv.as_ptr();
        let wa = _mm256_set1_epi16(a as i16);
        let we = _mm256_set1_epi16(e as i16);
        let c32 = _mm256_set1_epi16(32);

        if block_w == 16 {
            for j in 0..block_h {
                // SAFETY: Caller verified bounds including `+step`.
                unsafe {
                    let r0 = sp.add(j * ref_stride);
                    let s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(r0 as *const __m128i));
                    let s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(r0.add(step) as *const __m128i));
                    let mut sum =
                        _mm256_add_epi16(_mm256_mullo_epi16(s0, wa), _mm256_mullo_epi16(s1, we));
                    sum = _mm256_srli_epi16::<6>(_mm256_add_epi16(sum, c32));
                    let mut val = _mm_packus_epi16(
                        _mm256_castsi256_si128(sum),
                        _mm256_extracti128_si256::<1>(sum),
                    );
                    let dst_ptr = dp.add(j * dst_stride);
                    if avg {
                        val = _mm_avg_epu8(val, _mm_loadu_si128(dst_ptr as *const __m128i));
                    }
                    _mm_storeu_si128(dst_ptr as *mut __m128i, val);
                }
            }
        } else {
            // SAFETY: AVX2 implies SSE2; delegate 4/8 width via SSE2 function inlined into AVX2?
            // Keep pure VEX 128-bit loop here to avoid legacy SSE transition.
            let wa128 = _mm256_castsi256_si128(wa);
            let we128 = _mm256_castsi256_si128(we);
            let c32_128 = _mm256_castsi256_si128(c32);
            let zero = _mm_setzero_si128();
            if block_w == 8 {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds including `+step`.
                    unsafe {
                        let r0 = sp.add(j * ref_stride);
                        let s0 = _mm_unpacklo_epi8(load_u64_xmm(r0), zero);
                        let s1 = _mm_unpacklo_epi8(load_u64_xmm(r0.add(step)), zero);
                        let mut sum =
                            _mm_add_epi16(_mm_mullo_epi16(s0, wa128), _mm_mullo_epi16(s1, we128));
                        sum = _mm_srli_epi16::<6>(_mm_add_epi16(sum, c32_128));
                        let mut val = _mm_packus_epi16(sum, sum);
                        let dst_ptr = dp.add(j * dst_stride);
                        if avg {
                            val = _mm_avg_epu8(val, load_u64_xmm(dst_ptr));
                        }
                        store_u64_xmm(dst_ptr, val);
                    }
                }
            } else {
                for j in 0..block_h {
                    // SAFETY: Caller verified bounds including `+step`.
                    unsafe {
                        let r0 = sp.add(j * ref_stride);
                        let s0 = _mm_unpacklo_epi8(load_u32_xmm(r0), zero);
                        let s1 = _mm_unpacklo_epi8(load_u32_xmm(r0.add(step)), zero);
                        let mut sum =
                            _mm_add_epi16(_mm_mullo_epi16(s0, wa128), _mm_mullo_epi16(s1, we128));
                        sum = _mm_srli_epi16::<6>(_mm_add_epi16(sum, c32_128));
                        let mut val = _mm_packus_epi16(sum, sum);
                        let dst_ptr = dp.add(j * dst_stride);
                        if avg {
                            val = _mm_avg_epu8(val, load_u32_xmm(dst_ptr));
                        }
                        store_u32_xmm(dst_ptr, val);
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn fir6_i16_256(
        a: __m256i,
        b: __m256i,
        c: __m256i,
        d: __m256i,
        e: __m256i,
        f: __m256i,
    ) -> __m256i {
        let c5 = _mm256_set1_epi16(5);
        let c20 = _mm256_set1_epi16(20);
        let af = _mm256_add_epi16(a, f);
        let be = _mm256_mullo_epi16(_mm256_add_epi16(b, e), c5);
        let cd = _mm256_mullo_epi16(_mm256_add_epi16(c, d), c20);
        _mm256_sub_epi16(_mm256_add_epi16(af, cd), be)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn fir6_i16_128(
        a: __m128i,
        b: __m128i,
        c: __m128i,
        d: __m128i,
        e: __m128i,
        f: __m128i,
    ) -> __m128i {
        let c5 = _mm_set1_epi16(5);
        let c20 = _mm_set1_epi16(20);
        let af = _mm_add_epi16(a, f);
        let be = _mm_mullo_epi16(_mm_add_epi16(b, e), c5);
        let cd = _mm_mullo_epi16(_mm_add_epi16(c, d), c20);
        _mm_sub_epi16(_mm_add_epi16(af, cd), be)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn luma_6tap_h(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        if block_w == 16 {
            let c16 = _mm256_set1_epi16(16);
            for j in 0..block_h {
                // SAFETY: Caller verified `16 + 5 = 21` bytes per row in `src`.
                unsafe {
                    let r = sp.add(j * src_stride);
                    let a = _mm256_cvtepu8_epi16(_mm_loadu_si128(r as *const __m128i));
                    let b = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(1) as *const __m128i));
                    let c = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(2) as *const __m128i));
                    let d = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(3) as *const __m128i));
                    let e = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(4) as *const __m128i));
                    let f = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(5) as *const __m128i));
                    let sum = _mm256_add_epi16(fir6_i16_256(a, b, c, d, e, f), c16);
                    let res = _mm256_srai_epi16::<5>(sum);
                    let packed = _mm_packus_epi16(
                        _mm256_castsi256_si128(res),
                        _mm256_extracti128_si256::<1>(res),
                    );
                    _mm_storeu_si128(dp.add(j * dst_stride) as *mut __m128i, packed);
                }
            }
        } else {
            let zero = _mm_setzero_si128();
            let c16 = _mm_set1_epi16(16);
            if block_w == 8 {
                for j in 0..block_h {
                    // SAFETY: Caller verified `8 + 5 = 13` bytes per row in `src`.
                    unsafe {
                        let r = sp.add(j * src_stride);
                        let a = _mm_unpacklo_epi8(load_u64_xmm(r), zero);
                        let b = _mm_unpacklo_epi8(load_u64_xmm(r.add(1)), zero);
                        let c = _mm_unpacklo_epi8(load_u64_xmm(r.add(2)), zero);
                        let d = _mm_unpacklo_epi8(load_u64_xmm(r.add(3)), zero);
                        let e = _mm_unpacklo_epi8(load_u64_xmm(r.add(4)), zero);
                        let f = _mm_unpacklo_epi8(load_u64_xmm(r.add(5)), zero);
                        let sum = _mm_add_epi16(fir6_i16_128(a, b, c, d, e, f), c16);
                        let out = _mm_packus_epi16(_mm_srai_epi16::<5>(sum), zero);
                        store_u64_xmm(dp.add(j * dst_stride), out);
                    }
                }
            } else {
                for j in 0..block_h {
                    // SAFETY: Caller verified `4 + 5 = 9` bytes per row in `src`.
                    unsafe {
                        let r = sp.add(j * src_stride);
                        let a = _mm_unpacklo_epi8(load_u32_xmm(r), zero);
                        let b = _mm_unpacklo_epi8(load_u32_xmm(r.add(1)), zero);
                        let c = _mm_unpacklo_epi8(load_u32_xmm(r.add(2)), zero);
                        let d = _mm_unpacklo_epi8(load_u32_xmm(r.add(3)), zero);
                        let e = _mm_unpacklo_epi8(load_u32_xmm(r.add(4)), zero);
                        let f = _mm_unpacklo_epi8(load_u32_xmm(r.add(5)), zero);
                        let sum = _mm_add_epi16(fir6_i16_128(a, b, c, d, e, f), c16);
                        let out = _mm_packus_epi16(_mm_srai_epi16::<5>(sum), zero);
                        store_u32_xmm(dp.add(j * dst_stride), out);
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn luma_6tap_v(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        if block_w == 16 {
            let c16 = _mm256_set1_epi16(16);
            // SAFETY: Caller verified `(block_h + 4) * src_stride + 16 <= src.len()`.
            unsafe {
                let mut r0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(sp as *const __m128i));
                let mut r1 =
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(sp.add(src_stride) as *const __m128i));
                let mut r2 =
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(sp.add(2 * src_stride) as *const __m128i));
                let mut r3 =
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(sp.add(3 * src_stride) as *const __m128i));
                let mut r4 =
                    _mm256_cvtepu8_epi16(_mm_loadu_si128(sp.add(4 * src_stride) as *const __m128i));
                for j in 0..block_h {
                    let r5 = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                        sp.add((j + 5) * src_stride) as *const __m128i
                    ));
                    let sum = _mm256_add_epi16(fir6_i16_256(r0, r1, r2, r3, r4, r5), c16);
                    let res = _mm256_srai_epi16::<5>(sum);
                    let packed = _mm_packus_epi16(
                        _mm256_castsi256_si128(res),
                        _mm256_extracti128_si256::<1>(res),
                    );
                    _mm_storeu_si128(dp.add(j * dst_stride) as *mut __m128i, packed);
                    r0 = r1;
                    r1 = r2;
                    r2 = r3;
                    r3 = r4;
                    r4 = r5;
                }
            }
        } else {
            let zero = _mm_setzero_si128();
            let c16 = _mm_set1_epi16(16);
            if block_w == 8 {
                // SAFETY: Caller verified `(block_h + 4) * src_stride + 8 <= src.len()`.
                unsafe {
                    let mut r0 = _mm_unpacklo_epi8(load_u64_xmm(sp), zero);
                    let mut r1 = _mm_unpacklo_epi8(load_u64_xmm(sp.add(src_stride)), zero);
                    let mut r2 = _mm_unpacklo_epi8(load_u64_xmm(sp.add(2 * src_stride)), zero);
                    let mut r3 = _mm_unpacklo_epi8(load_u64_xmm(sp.add(3 * src_stride)), zero);
                    let mut r4 = _mm_unpacklo_epi8(load_u64_xmm(sp.add(4 * src_stride)), zero);
                    for j in 0..block_h {
                        let r5 =
                            _mm_unpacklo_epi8(load_u64_xmm(sp.add((j + 5) * src_stride)), zero);
                        let sum = _mm_add_epi16(fir6_i16_128(r0, r1, r2, r3, r4, r5), c16);
                        let out = _mm_packus_epi16(_mm_srai_epi16::<5>(sum), zero);
                        store_u64_xmm(dp.add(j * dst_stride), out);
                        r0 = r1;
                        r1 = r2;
                        r2 = r3;
                        r3 = r4;
                        r4 = r5;
                    }
                }
            } else {
                // SAFETY: Caller verified `(block_h + 4) * src_stride + 4 <= src.len()`.
                unsafe {
                    let mut r0 = _mm_unpacklo_epi8(load_u32_xmm(sp), zero);
                    let mut r1 = _mm_unpacklo_epi8(load_u32_xmm(sp.add(src_stride)), zero);
                    let mut r2 = _mm_unpacklo_epi8(load_u32_xmm(sp.add(2 * src_stride)), zero);
                    let mut r3 = _mm_unpacklo_epi8(load_u32_xmm(sp.add(3 * src_stride)), zero);
                    let mut r4 = _mm_unpacklo_epi8(load_u32_xmm(sp.add(4 * src_stride)), zero);
                    for j in 0..block_h {
                        let r5 =
                            _mm_unpacklo_epi8(load_u32_xmm(sp.add((j + 5) * src_stride)), zero);
                        let sum = _mm_add_epi16(fir6_i16_128(r0, r1, r2, r3, r4, r5), c16);
                        let out = _mm_packus_epi16(_mm_srai_epi16::<5>(sum), zero);
                        store_u32_xmm(dp.add(j * dst_stride), out);
                        r0 = r1;
                        r1 = r2;
                        r2 = r3;
                        r3 = r4;
                        r4 = r5;
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn luma_6tap_hv(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
        tmp: &mut [i32],
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let tp = tmp.as_mut_ptr();
        let tmp_h = block_h + 5;

        // Pass 1: horizontal 6-tap into i32 tmp.
        if block_w == 16 {
            for j in 0..tmp_h {
                // SAFETY: Caller verified `src` and `tmp` bounds for width 16.
                unsafe {
                    let r = sp.add(j * src_stride);
                    let a = _mm256_cvtepu8_epi16(_mm_loadu_si128(r as *const __m128i));
                    let b = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(1) as *const __m128i));
                    let c = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(2) as *const __m128i));
                    let d = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(3) as *const __m128i));
                    let e = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(4) as *const __m128i));
                    let f = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.add(5) as *const __m128i));
                    let v16 = fir6_i16_256(a, b, c, d, e, f);
                    let out_p = tp.add(j * 16);
                    _mm256_storeu_si256(
                        out_p as *mut __m256i,
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v16)),
                    );
                    _mm256_storeu_si256(
                        out_p.add(8) as *mut __m256i,
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v16)),
                    );
                }
            }
        } else {
            let zero = _mm_setzero_si128();
            if block_w == 8 {
                for j in 0..tmp_h {
                    // SAFETY: Caller verified `src` and `tmp` bounds for width 8.
                    unsafe {
                        let r = sp.add(j * src_stride);
                        let a = _mm_unpacklo_epi8(load_u64_xmm(r), zero);
                        let b = _mm_unpacklo_epi8(load_u64_xmm(r.add(1)), zero);
                        let c = _mm_unpacklo_epi8(load_u64_xmm(r.add(2)), zero);
                        let d = _mm_unpacklo_epi8(load_u64_xmm(r.add(3)), zero);
                        let e = _mm_unpacklo_epi8(load_u64_xmm(r.add(4)), zero);
                        let f = _mm_unpacklo_epi8(load_u64_xmm(r.add(5)), zero);
                        let v16 = fir6_i16_128(a, b, c, d, e, f);
                        _mm256_storeu_si256(
                            tp.add(j * 8) as *mut __m256i,
                            _mm256_cvtepi16_epi32(v16),
                        );
                    }
                }
            } else {
                for j in 0..tmp_h {
                    // SAFETY: Caller verified `src` and `tmp` bounds for width 4.
                    unsafe {
                        let r = sp.add(j * src_stride);
                        let a = _mm_unpacklo_epi8(load_u32_xmm(r), zero);
                        let b = _mm_unpacklo_epi8(load_u32_xmm(r.add(1)), zero);
                        let c = _mm_unpacklo_epi8(load_u32_xmm(r.add(2)), zero);
                        let d = _mm_unpacklo_epi8(load_u32_xmm(r.add(3)), zero);
                        let e = _mm_unpacklo_epi8(load_u32_xmm(r.add(4)), zero);
                        let f = _mm_unpacklo_epi8(load_u32_xmm(r.add(5)), zero);
                        let v16 = fir6_i16_128(a, b, c, d, e, f);
                        _mm_storeu_si128(tp.add(j * 4) as *mut __m128i, _mm_cvtepi16_epi32(v16));
                    }
                }
            }
        }

        // Pass 2: vertical 6-tap in i32 on `tmp`.
        if block_w >= 8 {
            let c512 = _mm256_set1_epi32(512);
            let chunks8 = block_w / 8;
            for ch in 0..chunks8 {
                let x = ch * 8;
                // SAFETY: Caller verified `tmp` and `dst` bounds.
                unsafe {
                    let col_tp = tp.add(x);
                    let mut r0 = _mm256_loadu_si256(col_tp as *const __m256i);
                    let mut r1 = _mm256_loadu_si256(col_tp.add(block_w) as *const __m256i);
                    let mut r2 = _mm256_loadu_si256(col_tp.add(2 * block_w) as *const __m256i);
                    let mut r3 = _mm256_loadu_si256(col_tp.add(3 * block_w) as *const __m256i);
                    let mut r4 = _mm256_loadu_si256(col_tp.add(4 * block_w) as *const __m256i);
                    for j in 0..block_h {
                        let r5 =
                            _mm256_loadu_si256(col_tp.add((j + 5) * block_w) as *const __m256i);
                        let af = _mm256_add_epi32(r0, r5);
                        let be = _mm256_add_epi32(r1, r4);
                        let be5 = _mm256_add_epi32(be, _mm256_slli_epi32::<2>(be));
                        let cd = _mm256_add_epi32(r2, r3);
                        let cd20 = _mm256_add_epi32(
                            _mm256_slli_epi32::<4>(cd),
                            _mm256_slli_epi32::<2>(cd),
                        );
                        let sum = _mm256_add_epi32(
                            _mm256_sub_epi32(_mm256_add_epi32(af, cd20), be5),
                            c512,
                        );
                        let shifted = _mm256_srai_epi32::<10>(sum);
                        let p16 = _mm256_packs_epi32(shifted, shifted);
                        let row16 = _mm_unpacklo_epi64(
                            _mm256_castsi256_si128(p16),
                            _mm256_extracti128_si256::<1>(p16),
                        );
                        let p8 = _mm_packus_epi16(row16, row16);
                        store_u64_xmm(dp.add(j * dst_stride + x), p8);
                        r0 = r1;
                        r1 = r2;
                        r2 = r3;
                        r3 = r4;
                        r4 = r5;
                    }
                }
            }
        } else {
            let c512 = _mm_set1_epi32(512);
            // SAFETY: Caller verified `tmp` and `dst` bounds for width 4.
            unsafe {
                let mut r0 = _mm_loadu_si128(tp as *const __m128i);
                let mut r1 = _mm_loadu_si128(tp.add(4) as *const __m128i);
                let mut r2 = _mm_loadu_si128(tp.add(8) as *const __m128i);
                let mut r3 = _mm_loadu_si128(tp.add(12) as *const __m128i);
                let mut r4 = _mm_loadu_si128(tp.add(16) as *const __m128i);
                for j in 0..block_h {
                    let r5 = _mm_loadu_si128(tp.add((j + 5) * 4) as *const __m128i);
                    let af = _mm_add_epi32(r0, r5);
                    let be = _mm_add_epi32(r1, r4);
                    let be5 = _mm_add_epi32(be, _mm_slli_epi32::<2>(be));
                    let cd = _mm_add_epi32(r2, r3);
                    let cd20 = _mm_add_epi32(_mm_slli_epi32::<4>(cd), _mm_slli_epi32::<2>(cd));
                    let sum = _mm_add_epi32(_mm_sub_epi32(_mm_add_epi32(af, cd20), be5), c512);
                    let shifted = _mm_srai_epi32::<10>(sum);
                    let p16 = _mm_packs_epi32(shifted, shifted);
                    let p8 = _mm_packus_epi16(p16, p16);
                    store_u32_xmm(dp.add(j * dst_stride), p8);
                    r0 = r1;
                    r1 = r2;
                    r2 = r3;
                    r3 = r4;
                    r4 = r5;
                }
            }
        }
    }
}

// ===========================================================================
// aarch64 NEON kernels (straightforward byte-level paths)
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod aarch64_neon {
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn idct_dc_add_8x8(dst: &mut [u8], stride: usize, dc: &mut i16) {
        let val = (*dc as i32 + 32) >> 6;
        *dc = 0;
        if val == 0 {
            return;
        }
        let dp = dst.as_mut_ptr();
        if val > 0 {
            let add = vdup_n_u8(val.min(255) as u8);
            for j in 0..8 {
                // SAFETY: Caller verified `7 * stride + 7 < dst.len()` and `stride >= 8`.
                unsafe {
                    let row = dp.add(j * stride);
                    let px = vld1_u8(row);
                    vst1_u8(row, vqadd_u8(px, add));
                }
            }
        } else {
            let sub = vdup_n_u8((-val).min(255) as u8);
            for j in 0..8 {
                // SAFETY: Caller verified `7 * stride + 7 < dst.len()` and `stride >= 8`.
                unsafe {
                    let row = dp.add(j * stride);
                    let px = vld1_u8(row);
                    vst1_u8(row, vqsub_u8(px, sub));
                }
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn put_pixels_16(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        for j in 0..block_h {
            // SAFETY: Caller verified bounds for `block_h` rows of 16 bytes.
            unsafe {
                let v = vld1q_u8(sp.add(j * src_stride));
                vst1q_u8(dp.add(j * dst_stride), v);
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn put_pixels_8(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        for j in 0..block_h {
            // SAFETY: Caller verified bounds for `block_h` rows of 8 bytes.
            unsafe {
                let v = vld1_u8(sp.add(j * src_stride));
                vst1_u8(dp.add(j * dst_stride), v);
            }
        }
    }

    #[target_feature(enable = "neon")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn avg_pixels(
        dst: &mut [u8],
        dst_stride: usize,
        a: &[u8],
        a_stride: usize,
        b: &[u8],
        b_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        if block_w == 16 {
            for j in 0..block_h {
                // SAFETY: Caller verified bounds for 16-byte rows.
                unsafe {
                    let va = vld1q_u8(ap.add(j * a_stride));
                    let vb = vld1q_u8(bp.add(j * b_stride));
                    vst1q_u8(dp.add(j * dst_stride), vrhaddq_u8(va, vb));
                }
            }
        } else if block_w == 8 {
            for j in 0..block_h {
                // SAFETY: Caller verified bounds for 8-byte rows.
                unsafe {
                    let va = vld1_u8(ap.add(j * a_stride));
                    let vb = vld1_u8(bp.add(j * b_stride));
                    vst1_u8(dp.add(j * dst_stride), vrhadd_u8(va, vb));
                }
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn avg_pixels_inplace(
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        block_w: usize,
        block_h: usize,
    ) {
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        if block_w == 16 {
            for j in 0..block_h {
                // SAFETY: Caller verified bounds for 16-byte rows.
                unsafe {
                    let d_row = dp.add(j * dst_stride);
                    let va = vld1q_u8(d_row);
                    let vb = vld1q_u8(sp.add(j * src_stride));
                    vst1q_u8(d_row, vrhaddq_u8(va, vb));
                }
            }
        } else if block_w == 8 {
            for j in 0..block_h {
                // SAFETY: Caller verified bounds for 8-byte rows.
                unsafe {
                    let d_row = dp.add(j * dst_stride);
                    let va = vld1_u8(d_row);
                    let vb = vld1_u8(sp.add(j * src_stride));
                    vst1_u8(d_row, vrhadd_u8(va, vb));
                }
            }
        }
    }
}
