//! Faithful pure-Rust port of FFmpeg `n9.1` `libswresample/resample.c`
//! and `libavutil/mathematics.c` (`av_bessel_i0`).
//!
//! Implements a Kaiser-windowed sinc polyphase filter bank with:
//! - Exact rational phase stepping (`src_incr`, `dst_incr`, `index`, `frac`)
//! - Symmetric boundary reflection (`invert_initial_buffer`) and trailing `flush()`
//! - Delay tracking (`delay_samples`) and output upper-bound calculation
//! - Dynamic fractional ratio / compensation support for VFR clock drift

use std::f64::consts::PI;

use wedeo_core::{Error, Result, reduce, rescale};

use crate::ops::SwrPolyphaseConfig;

#[inline]
fn eval_poly(coeff: &[f64], x: f64) -> f64 {
    let mut sum = coeff[coeff.len() - 1];
    for &c in coeff[..coeff.len() - 1].iter().rev() {
        sum = sum * x + c;
    }
    sum
}

/// 0th-order modified Bessel function of the first kind (`libavutil/mathematics.c`: `av_bessel_i0`).
#[allow(clippy::excessive_precision)]
pub fn bessel_i0(x: f64) -> f64 {
    const P1: [f64; 15] = [
        -2.233_558_263_947_437_5e+15,
        -5.505_036_967_301_843e+14,
        -3.294_008_762_740_775e+13,
        -8.492_510_124_711_416e+11,
        -1.191_274_610_498_523_7e+10,
        -1.031_306_670_873_798_1e+08,
        -5.954_562_601_984_79e+05,
        -2.412_519_587_604_189_7e+03,
        -7.093_534_744_921_055e+00,
        -1.545_397_779_178_685_1e-02,
        -2.517_264_467_068_897_5e-05,
        -3.051_722_645_045_106_7e-08,
        -2.684_344_857_346_848_3e-11,
        -1.598_222_667_565_318_5e-14,
        -5.248_786_662_794_57e-18,
    ];
    const Q1: [f64; 6] = [
        -2.233_558_263_947_437_5e+15,
        7.885_869_256_675_100_3e+12,
        -1.220_706_739_780_898e+10,
        1.037_708_105_806_216_6e+07,
        -4.852_756_017_996_277e+03,
        1.0,
    ];
    const P2: [f64; 7] = [
        -2.221_026_223_330_657_3e-04,
        1.306_739_203_810_692_4e-02,
        -4.470_080_572_117_445_4e-01,
        5.567_451_837_124_076e+00,
        -2.351_794_567_923_948e+01,
        3.161_132_281_870_113e+01,
        -9.609_002_196_865_618e+00,
    ];
    const Q2: [f64; 8] = [
        -5.519_433_023_100_548e-04,
        3.254_769_759_481_961_5e-02,
        -1.115_175_918_874_131_3e+00,
        1.398_259_535_389_285_2e+01,
        -6.022_800_206_674_334e+01,
        8.553_956_325_801_293e+01,
        -3.144_669_027_513_549e+01,
        1.0,
    ];

    if x == 0.0 {
        return 1.0;
    }
    let ax = x.abs();
    if ax <= 15.0 {
        let y = ax * ax;
        eval_poly(&P1, y) / eval_poly(&Q1, y)
    } else {
        let y = 1.0 / ax - 1.0 / 15.0;
        let r = eval_poly(&P2, y) / eval_poly(&Q2, y);
        let factor = ax.exp() / ax.sqrt();
        factor * r
    }
}

/// Build a normalized Kaiser-windowed sinc polyphase filter bank (`resample.c`: `build_filter`).
fn build_filter_bank_f64(
    factor: f64,
    tap_count: usize,
    filter_alloc: usize,
    phase_count: usize,
    kaiser_beta: f64,
) -> Vec<f64> {
    let mut bank = vec![0.0_f64; (phase_count + 1) * filter_alloc];
    let ph_nb = if !phase_count.is_multiple_of(2) {
        phase_count
    } else {
        phase_count / 2 + 1
    };
    let center = ((tap_count - 1) / 2) as isize;
    let effective_factor = factor.min(1.0);
    let mut tab = vec![0.0_f64; tap_count];
    let mut sin_lut = vec![0.0_f64; ph_nb];

    if effective_factor == 1.0 {
        let sign = if (center & 1) != 0 { 1.0 } else { -1.0 };
        for (ph, slot) in sin_lut.iter_mut().enumerate() {
            *slot = (PI * ph as f64 / phase_count as f64).sin() * sign;
        }
    }

    let mut norm = 0.0_f64;
    for (ph, &lut_val) in sin_lut.iter().enumerate() {
        let mut s = lut_val;
        for (i, tab_slot) in tab.iter_mut().enumerate() {
            let x = PI
                * ((i as isize - center) as f64 - ph as f64 / phase_count as f64)
                * effective_factor;
            let sinc = if x == 0.0 {
                1.0
            } else if effective_factor == 1.0 {
                s / x
            } else {
                x.sin() / x
            };
            let w = 2.0 * x / (effective_factor * tap_count as f64 * PI);
            let kaiser = bessel_i0(kaiser_beta * (1.0 - w * w).max(0.0).sqrt());
            let y = sinc * kaiser;
            *tab_slot = y;
            s = -s;
            if ph == 0 {
                norm += y;
            }
        }

        let inv_norm = if norm != 0.0 { 1.0 / norm } else { 1.0 };
        for (i, &val) in tab.iter().enumerate() {
            bank[ph * filter_alloc + i] = val * inv_norm;
        }
        if phase_count.is_multiple_of(2) {
            let mirror_ph = phase_count - ph;
            for i in 0..tap_count {
                bank[mirror_ph * filter_alloc + tap_count - 1 - i] = bank[ph * filter_alloc + i];
            }
        }
    }

    // Populate wrap-around phase row `phase_count` for linear interpolation at `index == phase_count - 1`:
    // phase `1.0` at `sample_index` equals phase `0.0` at `sample_index + 1`.
    bank[phase_count * filter_alloc] = 0.0;
    for i in 1..tap_count {
        bank[phase_count * filter_alloc + i] = bank[i - 1];
    }

    bank
}

/// Internal sample buffer precision for polyphase convolution.
enum ResampleStorage {
    F32 {
        filter_bank: Vec<f32>,
        in_planes: Vec<Vec<f32>>,
    },
    F64 {
        filter_bank: Vec<f64>,
        in_planes: Vec<Vec<f64>>,
    },
}

/// Stateful Kaiser-windowed sinc polyphase resampler matching FFmpeg `n9.1` `libswresample`.
pub struct PolyphaseResampler {
    in_rate: u32,
    out_rate: u32,
    channels: usize,
    factor: f64,
    kaiser_beta: f64,
    linear_interp: bool,
    filter_length: usize,
    filter_alloc: usize,
    phase_count: usize,
    phase_count_compensation: usize,
    src_incr: i64,
    dst_incr: i64,
    ideal_dst_incr: i64,
    dst_incr_div: i64,
    dst_incr_mod: i64,
    index: i64,
    frac: i64,
    compensation_distance: usize,
    in_offset: usize,
    in_len: usize,
    plane_cap: usize,
    started: bool,
    flushed: bool,
    storage: ResampleStorage,
}

impl PolyphaseResampler {
    /// Default FFmpeg `libswresample` cutoff ratio (`0.97`).
    pub const DEFAULT_CUTOFF: f64 = 0.97;

    /// Construct a `PolyphaseResampler` using `f32` or `f64` internal precision.
    pub fn new(config: SwrPolyphaseConfig, channels: usize, use_f64: bool) -> Result<Self> {
        config.validate()?;
        if channels == 0 {
            return Err(Error::invalid_argument(
                "resample",
                "channels must be non-zero",
            ));
        }

        let cutoff = Self::DEFAULT_CUTOFF;
        let factor = ((config.out_rate as f64 * cutoff) / config.in_rate as f64).min(1.0);
        let mut phase_count = 1usize << config.phase_shift;
        let mut phase_count_compensation = phase_count;

        let mut filter_length = ((config.filter_size as f64 / factor).ceil() as usize).max(1);
        if filter_length > 1 {
            filter_length = (filter_length + 1) & !1;
        }

        let (exact_num, _, _) = reduce(
            config.out_rate as i64,
            config.in_rate as i64,
            i32::MAX as i64,
        );
        if exact_num > 0 && (exact_num as usize) <= phase_count {
            let exact = exact_num as usize;
            phase_count_compensation = exact * (phase_count / exact);
            phase_count = exact;
        }

        let filter_alloc = (filter_length + 7) & !7;
        let bank_f64 = build_filter_bank_f64(
            factor,
            filter_length,
            filter_alloc,
            phase_count,
            config.kaiser_beta,
        );

        let (mut src_incr_i32, mut dst_incr_i32, exact) = reduce(
            config.out_rate as i64,
            config.in_rate as i64 * phase_count as i64,
            (i32::MAX / 2) as i64,
        );
        if !exact && src_incr_i32 == 0 {
            return Err(Error::invalid_argument(
                "resample",
                "resampling ratio exceeds rational bounds",
            ));
        }
        let mut src_incr = src_incr_i32 as i64;
        let mut dst_incr = dst_incr_i32 as i64;
        let _ = (&mut src_incr_i32, &mut dst_incr_i32);
        while dst_incr < (1 << 20) && src_incr < (1 << 20) {
            dst_incr *= 2;
            src_incr *= 2;
        }

        let ideal_dst_incr = dst_incr;
        let dst_incr_div = dst_incr / src_incr;
        let dst_incr_mod = dst_incr % src_incr;
        let index = -(phase_count as i64) * ((filter_length as i64 - 1) / 2);

        // Pre-allocate ample planar history capacity so steady-state processing never allocates.
        let plane_cap = (filter_length * 3 + 4096).max(4096);
        let in_offset = filter_length;

        let storage = if use_f64 {
            ResampleStorage::F64 {
                filter_bank: bank_f64,
                in_planes: (0..channels).map(|_| vec![0.0_f64; plane_cap]).collect(),
            }
        } else {
            ResampleStorage::F32 {
                filter_bank: bank_f64.into_iter().map(|v| v as f32).collect(),
                in_planes: (0..channels).map(|_| vec![0.0_f32; plane_cap]).collect(),
            }
        };

        Ok(Self {
            in_rate: config.in_rate,
            out_rate: config.out_rate,
            channels,
            factor,
            kaiser_beta: config.kaiser_beta,
            linear_interp: config.linear_interp,
            filter_length,
            filter_alloc,
            phase_count,
            phase_count_compensation,
            src_incr,
            dst_incr,
            ideal_dst_incr,
            dst_incr_div,
            dst_incr_mod,
            index,
            frac: 0,
            compensation_distance: 0,
            in_offset,
            in_len: 0,
            plane_cap,
            started: false,
            flushed: false,
            storage,
        })
    }

    #[inline]
    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    #[inline]
    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }

    #[inline]
    pub fn filter_length(&self) -> usize {
        self.filter_length
    }

    #[inline]
    pub fn phase_count(&self) -> usize {
        self.phase_count
    }

    /// Reset internal history and phase accumulators for reuse on a new stream / seek.
    pub fn reset(&mut self) {
        self.dst_incr = self.ideal_dst_incr;
        self.dst_incr_div = self.dst_incr / self.src_incr;
        self.dst_incr_mod = self.dst_incr % self.src_incr;
        self.index = -(self.phase_count as i64) * ((self.filter_length as i64 - 1) / 2);
        self.frac = 0;
        self.compensation_distance = 0;
        self.in_offset = self.filter_length;
        self.in_len = 0;
        self.started = false;
        self.flushed = false;
    }

    /// Expand the polyphase bank to `phase_count_compensation` phases for smooth
    /// fractional ratio / clock drift interpolation (`resample.c`: `rebuild_filter_bank_with_compensation`).
    fn ensure_compensation_bank(&mut self) -> Result<()> {
        let target_phases = self.phase_count_compensation;
        if target_phases == self.phase_count {
            return Ok(());
        }
        let ratio_mul = (target_phases / self.phase_count) as i64;
        let bank_f64 = build_filter_bank_f64(
            self.factor,
            self.filter_length,
            self.filter_alloc,
            target_phases,
            self.kaiser_beta,
        );
        let (new_src_i32, new_dst_i32, exact) = reduce(
            self.src_incr,
            self.dst_incr * ratio_mul,
            (i32::MAX / 2) as i64,
        );
        if !exact && new_src_i32 == 0 {
            return Err(Error::invalid_argument(
                "resample",
                "compensation ratio exceeds rational bounds",
            ));
        }
        let mut new_src = new_src_i32 as i64;
        let mut new_dst = new_dst_i32 as i64;
        while new_dst < (1 << 20) && new_src < (1 << 20) {
            new_dst *= 2;
            new_src *= 2;
        }
        self.src_incr = new_src;
        self.dst_incr = new_dst;
        self.ideal_dst_incr = new_dst;
        self.dst_incr_div = new_dst / new_src;
        self.dst_incr_mod = new_dst % new_src;
        self.index *= ratio_mul;
        self.phase_count = target_phases;

        match &mut self.storage {
            ResampleStorage::F32 { filter_bank, .. } => {
                *filter_bank = bank_f64.into_iter().map(|v| v as f32).collect();
            }
            ResampleStorage::F64 { filter_bank, .. } => {
                *filter_bank = bank_f64;
            }
        }
        Ok(())
    }

    /// Apply FFmpeg `swr_set_compensation`: stretch or squeeze by `sample_delta` samples
    /// over `compensation_distance` output samples to track VFR clock drift.
    pub fn set_compensation(
        &mut self,
        sample_delta: i32,
        compensation_distance: usize,
    ) -> Result<()> {
        if compensation_distance > 0 && sample_delta != 0 {
            self.ensure_compensation_bank()?;
        }
        self.compensation_distance = compensation_distance;
        if compensation_distance > 0 {
            self.dst_incr = self.ideal_dst_incr
                - self.ideal_dst_incr * sample_delta as i64 / compensation_distance as i64;
        } else {
            self.dst_incr = self.ideal_dst_incr;
        }
        if self.dst_incr <= 0 {
            return Err(Error::invalid_argument(
                "resample",
                "compensation sample_delta is too large for compensation_distance",
            ));
        }
        self.dst_incr_div = self.dst_incr / self.src_incr;
        self.dst_incr_mod = self.dst_incr % self.src_incr;
        Ok(())
    }

    /// Continuously adjust the resampling ratio (`effective_out_rate / effective_in_rate`)
    /// for VFR clock-drift tracking (e.g. `44100.0 / 48000.0 * 1.0002`).
    pub fn set_fractional_ratio(&mut self, out_over_in: f64) -> Result<()> {
        if !out_over_in.is_finite() || out_over_in <= 0.0 {
            return Err(Error::invalid_argument(
                "resample",
                "fractional resampling ratio must be finite and positive",
            ));
        }
        self.ensure_compensation_bank()?;
        self.linear_interp = true;
        self.compensation_distance = 0;
        let target = ((self.src_incr as f64 * self.phase_count as f64) / out_over_in)
            .round()
            .max(1.0);
        self.dst_incr = target as i64;
        self.dst_incr_div = self.dst_incr / self.src_incr;
        self.dst_incr_mod = self.dst_incr % self.src_incr;
        Ok(())
    }

    /// Current buffered delay expressed in ticks of `1 / base` seconds (`resample.c`: `get_delay`).
    pub fn delay_samples(&self, base: u32) -> i64 {
        let mut num = self.in_len as i64 - (self.filter_length as i64 - 1) / 2;
        num *= self.phase_count as i64;
        num -= self.index;
        num *= self.src_incr;
        num -= self.frac;
        rescale(
            num,
            base as i64,
            self.in_rate as i64 * self.src_incr * self.phase_count as i64,
        )
    }

    /// Conservative upper bound on output samples producible for `in_samples` additional inputs.
    pub fn out_samples_bound(&self, in_samples: usize) -> usize {
        let mut num = self.in_len as i64 + 2 + in_samples as i64;
        num *= self.phase_count as i64;
        num -= self.index;
        let den = self.in_rate as i64 * self.phase_count as i64;
        let out = ((num * self.out_rate as i64 + den - 1) / den + 2).max(0);
        if self.compensation_distance > 0 {
            out.max((out * self.ideal_dst_incr - 1) / self.dst_incr + 1) as usize
        } else {
            out as usize
        }
    }

    fn ensure_append_capacity(&mut self, add: usize) {
        if self.in_offset + self.in_len + add <= self.plane_cap {
            return;
        }
        let min_offset = if self.started { 0 } else { self.filter_length };
        if min_offset + self.in_len + add <= self.plane_cap {
            match &mut self.storage {
                ResampleStorage::F32 { in_planes, .. } => {
                    for p in in_planes {
                        p.copy_within(self.in_offset..self.in_offset + self.in_len, min_offset);
                    }
                }
                ResampleStorage::F64 { in_planes, .. } => {
                    for p in in_planes {
                        p.copy_within(self.in_offset..self.in_offset + self.in_len, min_offset);
                    }
                }
            }
            self.in_offset = min_offset;
        } else {
            let new_cap = (min_offset + self.in_len + add)
                .next_power_of_two()
                .max(self.plane_cap * 2);
            match &mut self.storage {
                ResampleStorage::F32 { in_planes, .. } => {
                    for p in in_planes {
                        p.copy_within(self.in_offset..self.in_offset + self.in_len, min_offset);
                        p.resize(new_cap, 0.0);
                    }
                }
                ResampleStorage::F64 { in_planes, .. } => {
                    for p in in_planes {
                        p.copy_within(self.in_offset..self.in_offset + self.in_len, min_offset);
                        p.resize(new_cap, 0.0);
                    }
                }
            }
            self.in_offset = min_offset;
            self.plane_cap = new_cap;
        }
    }

    /// Push `nb_samples` of planar `f32` audio into the resampler's history buffer.
    pub fn push_f32(&mut self, planes: &[&[f32]], nb_samples: usize) -> Result<()> {
        if planes.len() != self.channels {
            return Err(Error::invalid_argument(
                "resample",
                format!("expected {} channels, got {}", self.channels, planes.len()),
            ));
        }
        if nb_samples == 0 {
            return Ok(());
        }
        self.flushed = false;
        self.ensure_append_capacity(nb_samples);
        let write_pos = self.in_offset + self.in_len;
        match &mut self.storage {
            ResampleStorage::F32 { in_planes, .. } => {
                for (dst, src) in in_planes.iter_mut().zip(planes.iter()) {
                    dst[write_pos..write_pos + nb_samples].copy_from_slice(&src[..nb_samples]);
                }
            }
            ResampleStorage::F64 { in_planes, .. } => {
                for (dst, src) in in_planes.iter_mut().zip(planes.iter()) {
                    for (d, &s) in dst[write_pos..write_pos + nb_samples]
                        .iter_mut()
                        .zip(&src[..nb_samples])
                    {
                        *d = s as f64;
                    }
                }
            }
        }
        self.in_len += nb_samples;
        self.try_invert_initial(false);
        Ok(())
    }

    /// Push `nb_samples` of planar `f64` audio into the resampler's history buffer.
    pub fn push_f64(&mut self, planes: &[&[f64]], nb_samples: usize) -> Result<()> {
        if planes.len() != self.channels {
            return Err(Error::invalid_argument(
                "resample",
                format!("expected {} channels, got {}", self.channels, planes.len()),
            ));
        }
        if nb_samples == 0 {
            return Ok(());
        }
        self.flushed = false;
        self.ensure_append_capacity(nb_samples);
        let write_pos = self.in_offset + self.in_len;
        match &mut self.storage {
            ResampleStorage::F32 { in_planes, .. } => {
                for (dst, src) in in_planes.iter_mut().zip(planes.iter()) {
                    for (d, &s) in dst[write_pos..write_pos + nb_samples]
                        .iter_mut()
                        .zip(&src[..nb_samples])
                    {
                        *d = s as f32;
                    }
                }
            }
            ResampleStorage::F64 { in_planes, .. } => {
                for (dst, src) in in_planes.iter_mut().zip(planes.iter()) {
                    dst[write_pos..write_pos + nb_samples].copy_from_slice(&src[..nb_samples]);
                }
            }
        }
        self.in_len += nb_samples;
        self.try_invert_initial(false);
        Ok(())
    }

    /// Push `nb_samples` from a flat planar slab (`ch * stride .. ch * stride + nb_samples`).
    pub fn push_f32_slab(&mut self, slab: &[f32], stride: usize, nb_samples: usize) -> Result<()> {
        if slab.len() < self.channels * stride || nb_samples > stride {
            return Err(Error::invalid_argument(
                "resample",
                "slab dimensions too small",
            ));
        }
        if nb_samples == 0 {
            return Ok(());
        }
        self.flushed = false;
        self.ensure_append_capacity(nb_samples);
        let write_pos = self.in_offset + self.in_len;
        match &mut self.storage {
            ResampleStorage::F32 { in_planes, .. } => {
                for (ch, dst) in in_planes.iter_mut().enumerate() {
                    let src = &slab[ch * stride..ch * stride + nb_samples];
                    dst[write_pos..write_pos + nb_samples].copy_from_slice(src);
                }
            }
            ResampleStorage::F64 { in_planes, .. } => {
                for (ch, dst) in in_planes.iter_mut().enumerate() {
                    let src = &slab[ch * stride..ch * stride + nb_samples];
                    for (d, &s) in dst[write_pos..write_pos + nb_samples].iter_mut().zip(src) {
                        *d = s as f64;
                    }
                }
            }
        }
        self.in_len += nb_samples;
        self.try_invert_initial(false);
        Ok(())
    }

    /// Push `nb_samples` from a flat planar `f64` slab (`ch * stride .. ch * stride + nb_samples`).
    pub fn push_f64_slab(&mut self, slab: &[f64], stride: usize, nb_samples: usize) -> Result<()> {
        if slab.len() < self.channels * stride || nb_samples > stride {
            return Err(Error::invalid_argument(
                "resample",
                "slab dimensions too small",
            ));
        }
        if nb_samples == 0 {
            return Ok(());
        }
        self.flushed = false;
        self.ensure_append_capacity(nb_samples);
        let write_pos = self.in_offset + self.in_len;
        match &mut self.storage {
            ResampleStorage::F32 { in_planes, .. } => {
                for (ch, dst) in in_planes.iter_mut().enumerate() {
                    let src = &slab[ch * stride..ch * stride + nb_samples];
                    for (d, &s) in dst[write_pos..write_pos + nb_samples].iter_mut().zip(src) {
                        *d = s as f32;
                    }
                }
            }
            ResampleStorage::F64 { in_planes, .. } => {
                for (ch, dst) in in_planes.iter_mut().enumerate() {
                    let src = &slab[ch * stride..ch * stride + nb_samples];
                    dst[write_pos..write_pos + nb_samples].copy_from_slice(src);
                }
            }
        }
        self.in_len += nb_samples;
        self.try_invert_initial(false);
        Ok(())
    }

    /// Symmetric boundary reflection for the initial `filter_length` taps (`resample.c`: `invert_initial_buffer`).
    fn try_invert_initial(&mut self, force: bool) {
        if self.started || self.index >= 0 {
            self.started = true;
            return;
        }
        if !force && self.in_len <= self.filter_length {
            return;
        }
        if self.in_len == 0 {
            return;
        }

        let origin = self.in_offset;
        let available = self.in_len;
        let fl = self.filter_length;

        match &mut self.storage {
            ResampleStorage::F32 { in_planes, .. } => {
                for p in in_planes {
                    for n in 1..=fl {
                        let mirror = if available == 1 {
                            0
                        } else {
                            let period = (available - 1) * 2;
                            let m = n % period;
                            if m < available { m } else { period - m }
                        };
                        p[origin - n] = p[origin + mirror];
                    }
                }
            }
            ResampleStorage::F64 { in_planes, .. } => {
                for p in in_planes {
                    for n in 1..=fl {
                        let mirror = if available == 1 {
                            0
                        } else {
                            let period = (available - 1) * 2;
                            let m = n % period;
                            if m < available { m } else { period - m }
                        };
                        p[origin - n] = p[origin + mirror];
                    }
                }
            }
        }

        let mut left_pad = 0usize;
        while self.index < 0 {
            left_pad += 1;
            self.index += self.phase_count as i64;
        }
        self.in_offset -= left_pad;
        self.in_len += left_pad;
        self.started = true;
    }

    /// Reflect trailing boundary samples so all remaining samples drain cleanly on EOS (`resample.c`: `resample_flush`).
    pub fn flush(&mut self) {
        if self.flushed {
            return;
        }
        if !self.started {
            self.try_invert_initial(true);
        }
        if self.started && self.in_len > 0 && self.filter_length > 1 {
            let reflection = self.in_len.min(self.filter_length).div_ceil(2);
            self.ensure_append_capacity(reflection);
            let end = self.in_offset + self.in_len;
            match &mut self.storage {
                ResampleStorage::F32 { in_planes, .. } => {
                    for p in in_planes {
                        for j in 0..reflection {
                            let idx = end - 1 - j.min(self.in_len - 1);
                            p[end + j] = p[idx];
                        }
                    }
                }
                ResampleStorage::F64 { in_planes, .. } => {
                    for p in in_planes {
                        for j in 0..reflection {
                            let idx = end - 1 - j.min(self.in_len - 1);
                            p[end + j] = p[idx];
                        }
                    }
                }
            }
            self.in_len += reflection;
        }
        self.flushed = true;
    }

    /// Compute how many output samples can be drained right now (clamped to `max_out`).
    fn available_out_samples(&self, max_out: usize) -> usize {
        if !self.started || max_out == 0 {
            return 0;
        }
        let mut avail = if self.filter_length == 1 && self.phase_count == 1 {
            let num = self.in_len as i64 * self.src_incr - self.frac + self.dst_incr - 1;
            (num / self.dst_incr).max(0) as usize
        } else {
            if self.in_len < self.filter_length {
                return 0;
            }
            let end_index =
                (1i64 + self.in_len as i64 - self.filter_length as i64) * self.phase_count as i64;
            let delta_frac = (end_index - self.index) * self.src_incr - self.frac;
            if delta_frac <= 0 {
                return 0;
            }
            ((delta_frac + self.dst_incr - 1) / self.dst_incr).max(0) as usize
        };
        if self.compensation_distance > 0 {
            avail = avail.min(self.compensation_distance);
        }
        avail.min(max_out)
    }

    fn advance_compensation(&mut self, produced: usize) {
        if self.compensation_distance > 0 {
            self.compensation_distance = self.compensation_distance.saturating_sub(produced);
            if self.compensation_distance == 0 {
                self.dst_incr = self.ideal_dst_incr;
                self.dst_incr_div = self.dst_incr / self.src_incr;
                self.dst_incr_mod = self.dst_incr % self.src_incr;
            }
        }
    }

    /// Drain up to `max_out` resampled planar `f32` samples. Returns the number of samples written.
    pub fn drain_f32(&mut self, out_planes: &mut [&mut [f32]], max_out: usize) -> Result<usize> {
        if out_planes.len() != self.channels {
            return Err(Error::invalid_argument(
                "resample",
                format!(
                    "expected {} output channels, got {}",
                    self.channels,
                    out_planes.len()
                ),
            ));
        }
        Ok(self.drain_channels(
            max_out,
            out_planes.iter_mut().map(|plane| &mut **plane),
            resample_channel_f32,
            resample_channel_f64_to_f32,
        ))
    }

    /// Drain up to `max_out` resampled planar `f64` samples. Returns the number of samples written.
    pub fn drain_f64(&mut self, out_planes: &mut [&mut [f64]], max_out: usize) -> Result<usize> {
        if out_planes.len() != self.channels {
            return Err(Error::invalid_argument(
                "resample",
                format!(
                    "expected {} output channels, got {}",
                    self.channels,
                    out_planes.len()
                ),
            ));
        }
        Ok(self.drain_channels(
            max_out,
            out_planes.iter_mut().map(|plane| &mut **plane),
            resample_channel_f32_to_f64,
            resample_channel_f64,
        ))
    }

    /// Drain up to `max_out` resampled samples into a flat planar `f32` slab (`ch * stride .. ch * stride + n`).
    pub fn drain_f32_slab(
        &mut self,
        slab: &mut [f32],
        stride: usize,
        max_out: usize,
    ) -> Result<usize> {
        if slab.len() < self.channels * stride || max_out > stride {
            return Err(Error::invalid_argument(
                "resample",
                "slab dimensions too small",
            ));
        }
        Ok(self.drain_channels(
            max_out,
            slab.chunks_mut(stride.max(1)),
            resample_channel_f32,
            resample_channel_f64_to_f32,
        ))
    }

    /// Drain up to `max_out` resampled samples into a flat planar `f64` slab (`ch * stride .. ch * stride + n`).
    pub fn drain_f64_slab(
        &mut self,
        slab: &mut [f64],
        stride: usize,
        max_out: usize,
    ) -> Result<usize> {
        if slab.len() < self.channels * stride || max_out > stride {
            return Err(Error::invalid_argument(
                "resample",
                "slab dimensions too small",
            ));
        }
        Ok(self.drain_channels(
            max_out,
            slab.chunks_mut(stride.max(1)),
            resample_channel_f32_to_f64,
            resample_channel_f64,
        ))
    }
    fn drain_channels<'a, T: 'a>(
        &mut self,
        max_out: usize,
        output: impl Iterator<Item = &'a mut [T]>,
        from_f32: impl Fn(&[f32], &mut [T], &[f32], FilterStep) -> (usize, i64, i64),
        from_f64: impl Fn(&[f64], &mut [T], &[f64], FilterStep) -> (usize, i64, i64),
    ) -> usize {
        let n = self.available_out_samples(max_out);
        if n == 0 {
            return 0;
        }
        let step = FilterStep {
            filter_length: self.filter_length,
            filter_alloc: self.filter_alloc,
            phase_count: self.phase_count as i64,
            src_incr: self.src_incr,
            dst_incr_div: self.dst_incr_div,
            dst_incr_mod: self.dst_incr_mod,
            index: self.index,
            frac: self.frac,
            use_linear: self.linear_interp && (self.frac != 0 || self.dst_incr_mod != 0),
        };
        let input = self.in_offset..self.in_offset + self.in_len;
        let mut consumed = 0;
        for (ch, plane) in output.take(self.channels).enumerate() {
            let dst = &mut plane[..n];
            (consumed, self.index, self.frac) = match &self.storage {
                ResampleStorage::F32 {
                    filter_bank,
                    in_planes,
                } => from_f32(&in_planes[ch][input.clone()], dst, filter_bank, step),
                ResampleStorage::F64 {
                    filter_bank,
                    in_planes,
                } => from_f64(&in_planes[ch][input.clone()], dst, filter_bank, step),
            };
        }
        self.in_offset += consumed;
        self.in_len -= consumed;
        if self.in_len == 0 && self.started {
            self.in_offset = 0;
        }
        self.advance_compensation(n);
        n
    }
}

#[derive(Clone, Copy)]
struct FilterStep {
    filter_length: usize,
    filter_alloc: usize,
    phase_count: i64,
    src_incr: i64,
    dst_incr_div: i64,
    dst_incr_mod: i64,
    index: i64,
    frac: i64,
    use_linear: bool,
}

fn resample_channel_f32(
    src: &[f32],
    dst: &mut [f32],
    filter_bank: &[f32],
    step: FilterStep,
) -> (usize, i64, i64) {
    let FilterStep {
        filter_length,
        filter_alloc,
        phase_count,
        src_incr,
        dst_incr_div,
        dst_incr_mod,
        mut index,
        mut frac,
        use_linear,
    } = step;
    let mut sample_index = 0usize;
    while index >= phase_count {
        sample_index += 1;
        index -= phase_count;
    }

    let inv_src_incr = 1.0_f32 / src_incr as f32;
    for out_slot in dst.iter_mut() {
        let base = filter_alloc * index as usize;
        let f = &filter_bank[base..base + filter_length];
        let s = &src[sample_index..sample_index + filter_length];

        if use_linear {
            let f_next = &filter_bank[base + filter_alloc..base + filter_alloc + filter_length];
            let mut v1 = 0.0_f32;
            let mut v2 = 0.0_f32;
            for i in 0..filter_length {
                v1 += s[i] * f[i];
                v2 += s[i] * f_next[i];
            }
            *out_slot = v1 + (v2 - v1) * inv_src_incr * frac as f32;
        } else {
            let mut val = 0.0_f32;
            let mut val2 = 0.0_f32;
            let mut i = 0;
            while i + 1 < filter_length {
                val += s[i] * f[i];
                val2 += s[i + 1] * f[i + 1];
                i += 2;
            }
            if i < filter_length {
                val += s[i] * f[i];
            }
            *out_slot = val + val2;
        }

        frac += dst_incr_mod;
        index += dst_incr_div;
        if frac >= src_incr {
            frac -= src_incr;
            index += 1;
        }
        while index >= phase_count {
            sample_index += 1;
            index -= phase_count;
        }
    }

    (sample_index, index, frac)
}

fn resample_channel_f64(
    src: &[f64],
    dst: &mut [f64],
    filter_bank: &[f64],
    step: FilterStep,
) -> (usize, i64, i64) {
    let FilterStep {
        filter_length,
        filter_alloc,
        phase_count,
        src_incr,
        dst_incr_div,
        dst_incr_mod,
        mut index,
        mut frac,
        use_linear,
    } = step;
    let mut sample_index = 0usize;
    while index >= phase_count {
        sample_index += 1;
        index -= phase_count;
    }

    let inv_src_incr = 1.0_f64 / src_incr as f64;
    for out_slot in dst.iter_mut() {
        let base = filter_alloc * index as usize;
        let f = &filter_bank[base..base + filter_length];
        let s = &src[sample_index..sample_index + filter_length];

        if use_linear {
            let f_next = &filter_bank[base + filter_alloc..base + filter_alloc + filter_length];
            let mut v1 = 0.0_f64;
            let mut v2 = 0.0_f64;
            for i in 0..filter_length {
                v1 += s[i] * f[i];
                v2 += s[i] * f_next[i];
            }
            *out_slot = v1 + (v2 - v1) * inv_src_incr * frac as f64;
        } else {
            let mut val = 0.0_f64;
            let mut val2 = 0.0_f64;
            let mut i = 0;
            while i + 1 < filter_length {
                val += s[i] * f[i];
                val2 += s[i + 1] * f[i + 1];
                i += 2;
            }
            if i < filter_length {
                val += s[i] * f[i];
            }
            *out_slot = val + val2;
        }

        frac += dst_incr_mod;
        index += dst_incr_div;
        if frac >= src_incr {
            frac -= src_incr;
            index += 1;
        }
        while index >= phase_count {
            sample_index += 1;
            index -= phase_count;
        }
    }

    (sample_index, index, frac)
}

fn resample_channel_f64_to_f32(
    src: &[f64],
    dst: &mut [f32],
    filter_bank: &[f64],
    step: FilterStep,
) -> (usize, i64, i64) {
    let FilterStep {
        filter_length,
        filter_alloc,
        phase_count,
        src_incr,
        dst_incr_div,
        dst_incr_mod,
        mut index,
        mut frac,
        use_linear,
    } = step;
    let mut sample_index = 0usize;
    while index >= phase_count {
        sample_index += 1;
        index -= phase_count;
    }

    let inv_src_incr = 1.0_f64 / src_incr as f64;
    for out_slot in dst.iter_mut() {
        let base = filter_alloc * index as usize;
        let f = &filter_bank[base..base + filter_length];
        let s = &src[sample_index..sample_index + filter_length];
        let res = if use_linear {
            let f_next = &filter_bank[base + filter_alloc..base + filter_alloc + filter_length];
            let mut v1 = 0.0_f64;
            let mut v2 = 0.0_f64;
            for i in 0..filter_length {
                v1 += s[i] * f[i];
                v2 += s[i] * f_next[i];
            }
            v1 + (v2 - v1) * inv_src_incr * frac as f64
        } else {
            let mut val = 0.0_f64;
            for i in 0..filter_length {
                val += s[i] * f[i];
            }
            val
        };
        *out_slot = res as f32;

        frac += dst_incr_mod;
        index += dst_incr_div;
        if frac >= src_incr {
            frac -= src_incr;
            index += 1;
        }
        while index >= phase_count {
            sample_index += 1;
            index -= phase_count;
        }
    }
    (sample_index, index, frac)
}

fn resample_channel_f32_to_f64(
    src: &[f32],
    dst: &mut [f64],
    filter_bank: &[f32],
    step: FilterStep,
) -> (usize, i64, i64) {
    let FilterStep {
        filter_length,
        filter_alloc,
        phase_count,
        src_incr,
        dst_incr_div,
        dst_incr_mod,
        mut index,
        mut frac,
        use_linear,
    } = step;
    let mut sample_index = 0usize;
    while index >= phase_count {
        sample_index += 1;
        index -= phase_count;
    }

    let inv_src_incr = 1.0_f32 / src_incr as f32;
    for out_slot in dst.iter_mut() {
        let base = filter_alloc * index as usize;
        let f = &filter_bank[base..base + filter_length];
        let s = &src[sample_index..sample_index + filter_length];
        let res = if use_linear {
            let f_next = &filter_bank[base + filter_alloc..base + filter_alloc + filter_length];
            let mut v1 = 0.0_f32;
            let mut v2 = 0.0_f32;
            for i in 0..filter_length {
                v1 += s[i] * f[i];
                v2 += s[i] * f_next[i];
            }
            v1 + (v2 - v1) * inv_src_incr * frac as f32
        } else {
            let mut val = 0.0_f32;
            for i in 0..filter_length {
                val += s[i] * f[i];
            }
            val
        };
        *out_slot = res as f64;

        frac += dst_incr_mod;
        index += dst_incr_div;
        if frac >= src_incr {
            frac -= src_incr;
            index += 1;
        }
        while index >= phase_count {
            sample_index += 1;
            index -= phase_count;
        }
    }
    (sample_index, index, frac)
}
