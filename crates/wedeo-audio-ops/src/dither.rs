//! Deterministic audio dithering and `SampleFormat` unpack/pack across all
//! planar and interleaved formats (`U8`, `S16`, `S32`, `S64`, `F32`, `F64`),
//! with clipping accounting into [`wedeo_core::StreamHealth`].

use wedeo_core::{
    Buffer, Error, HealthEvent, HealthIssue, Result, SampleFormat, StreamHealth, Timestamp, TrackId,
};

use crate::ops::AudioDither;

const INV_128_F64: f64 = 1.0 / 128.0;
const INV_32768_F64: f64 = 1.0 / 32768.0;
const INV_I32_F64: f64 = 1.0 / 2_147_483_648.0;
const INV_I64_F64: f64 = 1.0 / 9_223_372_036_854_775_808.0;

/// Per-channel deterministic LCG state matching FFmpeg `n9.1` `libswresample/dither.c`.
#[derive(Debug, Clone)]
pub struct DitherState {
    mode: AudioDither,
    seeds: Vec<u32>,
}

impl DitherState {
    /// Initialize deterministic per-channel seeds (`swresample.c` line 709).
    pub fn new(mode: AudioDither, channels: usize) -> Self {
        let mut s = Self {
            mode,
            seeds: Vec::with_capacity(channels),
        };
        s.reset_for(channels);
        s
    }

    pub fn mode(&self) -> AudioDither {
        self.mode
    }

    /// Reset deterministic PRNG seeds to their initial stream state.
    pub fn reset(&mut self) {
        let ch_count = self.seeds.len();
        self.reset_for(ch_count);
    }

    fn reset_for(&mut self, channels: usize) {
        self.seeds.clear();
        for ch in 0..channels {
            let seed = ((12_345_678_913_579u64
                .wrapping_mul(ch as u64)
                .wrapping_add(3_141_592))
                % 2_718_281_828) as u32;
            self.seeds.push(seed);
        }
    }

    /// Quantization noise scale in normalized `[-1.0, 1.0)` float space (`dither.c`).
    #[inline]
    pub const fn scale_for_format(fmt: SampleFormat) -> f64 {
        match fmt {
            SampleFormat::U8 | SampleFormat::U8Planar => INV_128_F64,
            SampleFormat::S16 | SampleFormat::S16Planar => INV_32768_F64,
            SampleFormat::S32 | SampleFormat::S32Planar => INV_I32_F64,
            SampleFormat::S64 | SampleFormat::S64Planar => INV_I64_F64,
            SampleFormat::F32
            | SampleFormat::F32Planar
            | SampleFormat::F64
            | SampleFormat::F64Planar => 0.0,
        }
    }

    /// Generate the next normalized dither sample for channel `ch` scaled by `scale`.
    #[inline]
    pub fn next_noise(&mut self, ch: usize, scale: f64) -> f64 {
        if scale == 0.0 || self.mode == AudioDither::None {
            return 0.0;
        }
        let seed = &mut self.seeds[ch];
        match self.mode {
            AudioDither::None => 0.0,
            AudioDither::Rectangular => {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((*seed as f64) / (u32::MAX as f64) - 0.5) * scale
            }
            AudioDither::Triangular => {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let a = (*seed as f64) / (u32::MAX as f64);
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let b = (*seed as f64) / (u32::MAX as f64);
                (a - b) * scale
            }
        }
    }
}

/// Record clipped sample counts into `StreamHealth` with at most one event per block.
#[inline]
pub fn record_clipping(health: &mut StreamHealth, pts: Timestamp, clipped: u64) {
    if clipped == 0 {
        return;
    }
    health.record(HealthEvent {
        track: TrackId(0),
        pts,
        byte_offset: None,
        kind: HealthIssue::AudioClip,
        detail: "audio sample clipped during format quantization",
    });
    if clipped > 1 {
        health.clipped_audio_samples = health.clipped_audio_samples.saturating_add(clipped - 1);
    }
}

trait UnpackSample: Sized {
    fn from_f64(sample: f64) -> Self;
    fn unpack_contiguous(src: &[u8], fmt: SampleFormat, dst: &mut [Self]);
}

impl UnpackSample for f32 {
    fn from_f64(sample: f64) -> Self {
        sample as f32
    }
    fn unpack_contiguous(src: &[u8], fmt: SampleFormat, dst: &mut [Self]) {
        unpack_contiguous_f32(src, fmt, dst);
    }
}

impl UnpackSample for f64 {
    fn from_f64(sample: f64) -> Self {
        sample
    }
    fn unpack_contiguous(src: &[u8], fmt: SampleFormat, dst: &mut [Self]) {
        unpack_contiguous_f64(src, fmt, dst);
    }
}

/// Unpack `count` samples starting at `sample_offset` from `planes` (in `fmt`) into planar `f32` slabs.
pub fn unpack_to_f32(
    planes: &[Buffer],
    fmt: SampleFormat,
    channels: usize,
    sample_offset: usize,
    count: usize,
    dst: &mut [&mut [f32]],
) -> Result<()> {
    unpack_planes(planes, fmt, channels, sample_offset, count, dst)
}

/// Unpack `count` samples starting at `sample_offset` from `planes` (in `fmt`) into planar `f64` slabs.
pub fn unpack_to_f64(
    planes: &[Buffer],
    fmt: SampleFormat,
    channels: usize,
    sample_offset: usize,
    count: usize,
    dst: &mut [&mut [f64]],
) -> Result<()> {
    unpack_planes(planes, fmt, channels, sample_offset, count, dst)
}

fn unpack_planes<T: UnpackSample>(
    planes: &[Buffer],
    fmt: SampleFormat,
    channels: usize,
    sample_offset: usize,
    count: usize,
    dst: &mut [&mut [T]],
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    if dst.len() != channels {
        return Err(Error::invalid_argument(
            "audio_unpack",
            "destination plane count mismatch",
        ));
    }

    if fmt.is_planar() {
        if planes.len() != channels {
            return Err(Error::invalid_argument(
                "audio_unpack",
                "planar AudioBlock plane count mismatch",
            ));
        }
        let bps = fmt.bytes_per_sample();
        let start_byte = sample_offset * bps;
        let end_byte = (sample_offset + count) * bps;
        for (ch, plane) in planes.iter().enumerate() {
            let bytes = plane.data();
            if bytes.len() < end_byte {
                return Err(Error::invalid_data(
                    "audio_unpack",
                    "planar audio buffer smaller than declared nb_samples",
                ));
            }
            let src = &bytes[start_byte..end_byte];
            let out = &mut dst[ch][..count];
            T::unpack_contiguous(src, fmt, out);
        }
    } else {
        if planes.len() != 1 {
            return Err(Error::invalid_argument(
                "audio_unpack",
                "interleaved AudioBlock must have exactly 1 plane",
            ));
        }
        let bps = fmt.bytes_per_sample();
        let stride = bps * channels;
        let start_byte = sample_offset * stride;
        let end_byte = (sample_offset + count) * stride;
        let bytes = planes[0].data();
        if bytes.len() < end_byte {
            return Err(Error::invalid_data(
                "audio_unpack",
                "interleaved audio buffer smaller than declared nb_samples",
            ));
        }
        let src = &bytes[start_byte..end_byte];
        for i in 0..count {
            let frame = &src[i * stride..(i + 1) * stride];
            for (ch, plane_out) in dst.iter_mut().enumerate() {
                let b = &frame[ch * bps..(ch + 1) * bps];
                plane_out[i] = T::from_f64(decode_sample_f64(b, fmt));
            }
        }
    }
    Ok(())
}

#[inline]
fn decode_sample_f64(b: &[u8], fmt: SampleFormat) -> f64 {
    match fmt {
        SampleFormat::U8 | SampleFormat::U8Planar => (b[0] as i32 - 0x80) as f64 * INV_128_F64,
        SampleFormat::S16 | SampleFormat::S16Planar => {
            i16::from_ne_bytes([b[0], b[1]]) as f64 * INV_32768_F64
        }
        SampleFormat::S32 | SampleFormat::S32Planar => {
            i32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f64 * INV_I32_F64
        }
        SampleFormat::S64 | SampleFormat::S64Planar => {
            i64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f64
                * INV_I64_F64
        }
        SampleFormat::F32 | SampleFormat::F32Planar => {
            f32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f64
        }
        SampleFormat::F64 | SampleFormat::F64Planar => {
            f64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        }
    }
}

fn unpack_contiguous_f32(src: &[u8], fmt: SampleFormat, dst: &mut [f32]) {
    match fmt {
        SampleFormat::U8 | SampleFormat::U8Planar => {
            for (d, &b) in dst.iter_mut().zip(src.iter()) {
                *d = (b as i32 - 0x80) as f32 * (INV_128_F64 as f32);
            }
        }
        SampleFormat::S16 | SampleFormat::S16Planar => {
            let (chunks, _) = src.as_chunks::<2>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = i16::from_ne_bytes(c) as f32 * (INV_32768_F64 as f32);
            }
        }
        SampleFormat::S32 | SampleFormat::S32Planar => {
            let (chunks, _) = src.as_chunks::<4>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = (i32::from_ne_bytes(c) as f64 * INV_I32_F64) as f32;
            }
        }
        SampleFormat::S64 | SampleFormat::S64Planar => {
            let (chunks, _) = src.as_chunks::<8>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = (i64::from_ne_bytes(c) as f64 * INV_I64_F64) as f32;
            }
        }
        SampleFormat::F32 | SampleFormat::F32Planar => {
            let (chunks, _) = src.as_chunks::<4>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = f32::from_ne_bytes(c);
            }
        }
        SampleFormat::F64 | SampleFormat::F64Planar => {
            let (chunks, _) = src.as_chunks::<8>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = f64::from_ne_bytes(c) as f32;
            }
        }
    }
}

fn unpack_contiguous_f64(src: &[u8], fmt: SampleFormat, dst: &mut [f64]) {
    match fmt {
        SampleFormat::U8 | SampleFormat::U8Planar => {
            for (d, &b) in dst.iter_mut().zip(src.iter()) {
                *d = (b as i32 - 0x80) as f64 * INV_128_F64;
            }
        }
        SampleFormat::S16 | SampleFormat::S16Planar => {
            let (chunks, _) = src.as_chunks::<2>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = i16::from_ne_bytes(c) as f64 * INV_32768_F64;
            }
        }
        SampleFormat::S32 | SampleFormat::S32Planar => {
            let (chunks, _) = src.as_chunks::<4>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = i32::from_ne_bytes(c) as f64 * INV_I32_F64;
            }
        }
        SampleFormat::S64 | SampleFormat::S64Planar => {
            let (chunks, _) = src.as_chunks::<8>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = i64::from_ne_bytes(c) as f64 * INV_I64_F64;
            }
        }
        SampleFormat::F32 | SampleFormat::F32Planar => {
            let (chunks, _) = src.as_chunks::<4>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = f32::from_ne_bytes(c) as f64;
            }
        }
        SampleFormat::F64 | SampleFormat::F64Planar => {
            let (chunks, _) = src.as_chunks::<8>();
            for (d, &c) in dst.iter_mut().zip(chunks) {
                *d = f64::from_ne_bytes(c);
            }
        }
    }
}

#[inline]
fn encode_sample_f64(val: f64, fmt: SampleFormat, out: &mut [u8], clipped: &mut u64) {
    match fmt {
        SampleFormat::U8 | SampleFormat::U8Planar => {
            let scaled = (val * 128.0).round() + 128.0;
            let byte = if !scaled.is_finite() {
                *clipped += 1;
                0x80
            } else if scaled < 0.0 {
                *clipped += 1;
                0
            } else if scaled > 255.0 {
                *clipped += 1;
                255
            } else {
                scaled as u8
            };
            out[0] = byte;
        }
        SampleFormat::S16 | SampleFormat::S16Planar => {
            let scaled = (val * 32768.0).round();
            let sample = if !scaled.is_finite() {
                *clipped += 1;
                0
            } else if scaled < -32768.0 {
                *clipped += 1;
                i16::MIN
            } else if scaled > 32767.0 {
                *clipped += 1;
                i16::MAX
            } else {
                scaled as i16
            };
            out[..2].copy_from_slice(&sample.to_ne_bytes());
        }
        SampleFormat::S32 | SampleFormat::S32Planar => {
            let scaled = (val * 2_147_483_648.0).round();
            let sample = if !scaled.is_finite() {
                *clipped += 1;
                0
            } else if scaled < -2_147_483_648.0 {
                *clipped += 1;
                i32::MIN
            } else if scaled > 2_147_483_647.0 {
                *clipped += 1;
                i32::MAX
            } else {
                scaled as i32
            };
            out[..4].copy_from_slice(&sample.to_ne_bytes());
        }
        SampleFormat::S64 | SampleFormat::S64Planar => {
            let scaled = (val * 9_223_372_036_854_775_808.0).round();
            let sample = if !scaled.is_finite() {
                *clipped += 1;
                0
            } else if scaled < -9_223_372_036_854_775_808.0 {
                *clipped += 1;
                i64::MIN
            } else if scaled >= 9_223_372_036_854_775_808.0 {
                *clipped += 1;
                i64::MAX
            } else {
                scaled as i64
            };
            out[..8].copy_from_slice(&sample.to_ne_bytes());
        }
        SampleFormat::F32 | SampleFormat::F32Planar => {
            out[..4].copy_from_slice(&(val as f32).to_ne_bytes());
        }
        SampleFormat::F64 | SampleFormat::F64Planar => {
            out[..8].copy_from_slice(&val.to_ne_bytes());
        }
    }
}

/// Pack `count` planar `f32` samples into `dst_planes` (planar or packed bytes) with optional dither,
/// returning the number of clipped samples.
pub fn pack_from_f32(
    src: &[&[f32]],
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_planes: &mut [&mut [u8]],
) -> u64 {
    pack_planes(src, count, fmt, dither, dst_planes)
}

/// Pack `count` planar `f64` samples into `dst_planes` (planar or packed bytes) with optional dither,
/// returning the number of clipped samples.
pub fn pack_from_f64(
    src: &[&[f64]],
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_planes: &mut [&mut [u8]],
) -> u64 {
    pack_planes(src, count, fmt, dither, dst_planes)
}

fn pack_planes<T: Copy + Into<f64>>(
    src: &[&[T]],
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_planes: &mut [&mut [u8]],
) -> u64 {
    if count == 0 {
        return 0;
    }
    let scale = DitherState::scale_for_format(fmt);
    let bps = fmt.bytes_per_sample();
    let mut clipped = 0u64;

    if fmt.is_planar() {
        for (ch, plane_src) in src.iter().enumerate() {
            let out = &mut dst_planes[ch][..count * bps];
            for i in 0..count {
                let v = plane_src[i].into() + dither.next_noise(ch, scale);
                encode_sample_f64(v, fmt, &mut out[i * bps..(i + 1) * bps], &mut clipped);
            }
        }
    } else {
        let channels = src.len();
        let stride = bps * channels;
        let out = &mut dst_planes[0][..count * stride];
        for i in 0..count {
            let frame = &mut out[i * stride..(i + 1) * stride];
            for (ch, plane_src) in src.iter().enumerate() {
                let v = plane_src[i].into() + dither.next_noise(ch, scale);
                encode_sample_f64(v, fmt, &mut frame[ch * bps..(ch + 1) * bps], &mut clipped);
            }
        }
    }
    clipped
}

/// Unpack `count` samples starting at `sample_offset` from `planes` into a flat planar `f32` slab
/// (`ch * stride .. ch * stride + count`).
pub fn unpack_slab_f32(
    planes: &[Buffer],
    fmt: SampleFormat,
    channels: usize,
    sample_offset: usize,
    count: usize,
    slab: &mut [f32],
    stride: usize,
) -> Result<()> {
    unpack_slab(planes, fmt, channels, sample_offset, count, slab, stride)
}

/// Unpack `count` samples starting at `sample_offset` from `planes` into a flat planar `f64` slab
/// (`ch * stride .. ch * stride + count`).
pub fn unpack_slab_f64(
    planes: &[Buffer],
    fmt: SampleFormat,
    channels: usize,
    sample_offset: usize,
    count: usize,
    slab: &mut [f64],
    stride: usize,
) -> Result<()> {
    unpack_slab(planes, fmt, channels, sample_offset, count, slab, stride)
}

fn unpack_slab<T: UnpackSample>(
    planes: &[Buffer],
    fmt: SampleFormat,
    channels: usize,
    sample_offset: usize,
    count: usize,
    slab: &mut [T],
    stride: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    if slab.len() < channels * stride || count > stride {
        return Err(Error::invalid_argument("audio_unpack", "slab too small"));
    }
    if fmt.is_planar() {
        if planes.len() != channels {
            return Err(Error::invalid_argument(
                "audio_unpack",
                "planar AudioBlock plane count mismatch",
            ));
        }
        let bps = fmt.bytes_per_sample();
        let start_byte = sample_offset * bps;
        let end_byte = (sample_offset + count) * bps;
        for (ch, plane) in planes.iter().enumerate() {
            let bytes = plane.data();
            if bytes.len() < end_byte {
                return Err(Error::invalid_data(
                    "audio_unpack",
                    "planar audio buffer smaller than declared nb_samples",
                ));
            }
            T::unpack_contiguous(
                &bytes[start_byte..end_byte],
                fmt,
                &mut slab[ch * stride..ch * stride + count],
            );
        }
    } else {
        if planes.len() != 1 {
            return Err(Error::invalid_argument(
                "audio_unpack",
                "interleaved AudioBlock must have exactly 1 plane",
            ));
        }
        let bps = fmt.bytes_per_sample();
        let frame_stride = bps * channels;
        let start_byte = sample_offset * frame_stride;
        let end_byte = (sample_offset + count) * frame_stride;
        let bytes = planes[0].data();
        if bytes.len() < end_byte {
            return Err(Error::invalid_data(
                "audio_unpack",
                "interleaved audio buffer smaller than declared nb_samples",
            ));
        }
        let src = &bytes[start_byte..end_byte];
        for i in 0..count {
            let frame = &src[i * frame_stride..(i + 1) * frame_stride];
            for ch in 0..channels {
                let b = &frame[ch * bps..(ch + 1) * bps];
                slab[ch * stride + i] = T::from_f64(decode_sample_f64(b, fmt));
            }
        }
    }
    Ok(())
}

/// Pack `count` samples from a flat planar `f32` slab into pooled `Buffer`s at `dst_sample_offset`.
#[allow(clippy::too_many_arguments)]
pub fn pack_slab_f32_to_buffers(
    slab: &[f32],
    stride: usize,
    channels: usize,
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_buffers: &mut [Buffer],
    dst_sample_offset: usize,
) -> u64 {
    pack_slab_buffers(
        slab,
        stride,
        channels,
        count,
        fmt,
        dither,
        dst_buffers,
        dst_sample_offset,
    )
}

/// Pack `count` samples from a flat planar `f64` slab into pooled `Buffer`s at `dst_sample_offset`.
#[allow(clippy::too_many_arguments)]
pub fn pack_slab_f64_to_buffers(
    slab: &[f64],
    stride: usize,
    channels: usize,
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_buffers: &mut [Buffer],
    dst_sample_offset: usize,
) -> u64 {
    pack_slab_buffers(
        slab,
        stride,
        channels,
        count,
        fmt,
        dither,
        dst_buffers,
        dst_sample_offset,
    )
}

#[allow(clippy::too_many_arguments)]
fn pack_slab_buffers<T: Copy + Into<f64>>(
    slab: &[T],
    stride: usize,
    channels: usize,
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_buffers: &mut [Buffer],
    dst_sample_offset: usize,
) -> u64 {
    if count == 0 {
        return 0;
    }
    let scale = DitherState::scale_for_format(fmt);
    let bps = fmt.bytes_per_sample();
    let mut clipped = 0u64;

    if fmt.is_planar() {
        let byte_off = dst_sample_offset * bps;
        for (ch, buf) in dst_buffers.iter_mut().enumerate().take(channels) {
            if let Some(out_all) = buf.try_data_mut() {
                let out = &mut out_all[byte_off..byte_off + count * bps];
                let src = &slab[ch * stride..ch * stride + count];
                for i in 0..count {
                    let v = src[i].into() + dither.next_noise(ch, scale);
                    encode_sample_f64(v, fmt, &mut out[i * bps..(i + 1) * bps], &mut clipped);
                }
            }
        }
    } else if let Some(out_all) = dst_buffers[0].try_data_mut() {
        let frame_stride = bps * channels;
        let byte_off = dst_sample_offset * frame_stride;
        let out = &mut out_all[byte_off..byte_off + count * frame_stride];
        for i in 0..count {
            let frame = &mut out[i * frame_stride..(i + 1) * frame_stride];
            for ch in 0..channels {
                let v = slab[ch * stride + i].into() + dither.next_noise(ch, scale);
                encode_sample_f64(v, fmt, &mut frame[ch * bps..(ch + 1) * bps], &mut clipped);
            }
        }
    }
    clipped
}

/// Pack `count` samples from a flat planar `f32` slab into pre-allocated `Vec<u8>` planes.
pub fn pack_slab_f32_to_vecs(
    slab: &[f32],
    stride: usize,
    channels: usize,
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_vecs: &mut [Vec<u8>],
) -> u64 {
    pack_slab_vecs(slab, stride, channels, count, fmt, dither, dst_vecs)
}

/// Pack `count` samples from a flat planar `f64` slab into pre-allocated `Vec<u8>` planes.
pub fn pack_slab_f64_to_vecs(
    slab: &[f64],
    stride: usize,
    channels: usize,
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_vecs: &mut [Vec<u8>],
) -> u64 {
    pack_slab_vecs(slab, stride, channels, count, fmt, dither, dst_vecs)
}

fn pack_slab_vecs<T: Copy + Into<f64>>(
    slab: &[T],
    stride: usize,
    channels: usize,
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_vecs: &mut [Vec<u8>],
) -> u64 {
    if count == 0 {
        return 0;
    }
    let scale = DitherState::scale_for_format(fmt);
    let bps = fmt.bytes_per_sample();
    let mut clipped = 0u64;

    if fmt.is_planar() {
        for (ch, vec_out) in dst_vecs.iter_mut().enumerate().take(channels) {
            let out = &mut vec_out[..count * bps];
            let src = &slab[ch * stride..ch * stride + count];
            for i in 0..count {
                let v = src[i].into() + dither.next_noise(ch, scale);
                encode_sample_f64(v, fmt, &mut out[i * bps..(i + 1) * bps], &mut clipped);
            }
        }
    } else {
        let frame_stride = bps * channels;
        let out = &mut dst_vecs[0][..count * frame_stride];
        for i in 0..count {
            let frame = &mut out[i * frame_stride..(i + 1) * frame_stride];
            for ch in 0..channels {
                let v = slab[ch * stride + i].into() + dither.next_noise(ch, scale);
                encode_sample_f64(v, fmt, &mut frame[ch * bps..(ch + 1) * bps], &mut clipped);
            }
        }
    }
    clipped
}
