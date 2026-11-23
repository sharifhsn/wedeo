#![forbid(unsafe_code)]

//! Pure-Rust `AudioOpList` compiler and execution engine porting FFmpeg `n9.1`
//! `libswresample` (`rematrix.c`, `resample.c`, `dither.c`, `audioconvert.c`)
//! with cost-optimal downmix-before-resample ordering, L1-cache-resident 1024-sample
//! planar slabs, zero-alloc circular reframing, and VFR timestamp preservation.

pub mod chain;
pub mod dither;
pub mod ops;
pub mod optimizer;
pub mod rematrix;
pub mod resample;

pub use chain::{AUDIO_SLAB_SAMPLES, CompiledAudioOps};
pub use dither::{DitherState, pack_from_f32, pack_from_f64, unpack_to_f32, unpack_to_f64};
pub use ops::{
    AudioContract, AudioDither, AudioOp, AudioOpList, AudioOpListBuilder, ChannelMatrix,
    SwrPolyphaseConfig,
};
pub use optimizer::{is_noop, optimize_ops};
pub use rematrix::{MatrixEncoding, RematrixOptions, build_rematrix_matrix};
pub use resample::{PolyphaseResampler, bessel_i0};

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

    use wedeo_core::{
        AudioBlock, Buffer, BufferPool, ChannelLayout, Rational, SampleFormat, SampleTrim,
        Timestamp,
    };

    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn make_f32p_sine_block(
        sample_rate: u32,
        layout: ChannelLayout,
        nb_samples: usize,
        freq_hz: f64,
        amp: f64,
        pts: Timestamp,
        time_base: Rational,
        sample_offset: usize,
    ) -> AudioBlock {
        let channels = layout.nb_channels as usize;
        let mut planes = Vec::with_capacity(channels);
        for ch in 0..channels {
            let mut bytes = vec![0u8; nb_samples * 4];
            for i in 0..nb_samples {
                let t = (sample_offset + i) as f64 / sample_rate as f64;
                let phase = ch as f64 * 0.25;
                let val = (amp * (2.0 * PI * freq_hz * t + phase).sin()) as f32;
                bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_ne_bytes());
            }
            planes.push(Buffer::from_slice(&bytes));
        }
        let mut block = AudioBlock::new(
            planes,
            nb_samples as u32,
            SampleFormat::F32Planar,
            sample_rate,
            layout,
        );
        block.pts = pts;
        block.time_base = time_base;
        block.duration = wedeo_core::rescale_q(
            nb_samples as i64,
            Rational::new(1, sample_rate as i32),
            time_base,
        );
        block
    }

    fn read_f32p_channel(blocks: &[AudioBlock], ch: usize) -> Vec<f32> {
        let mut samples = Vec::new();
        for b in blocks {
            assert_eq!(b.format, SampleFormat::F32Planar);
            let data = b.planes[ch].data();
            let count = b.nb_samples as usize;
            for i in 0..count {
                samples.push(f32::from_ne_bytes([
                    data[i * 4],
                    data[i * 4 + 1],
                    data[i * 4 + 2],
                    data[i * 4 + 3],
                ]));
            }
        }
        samples
    }

    fn measure_sine_snr_db(
        samples: &[f32],
        sample_rate: u32,
        freq_hz: f64,
        trim_edges: usize,
    ) -> f64 {
        let slice = &samples[trim_edges..samples.len() - trim_edges];
        // Exact 2x2 least-squares fit of A*cos(w*t) + B*sin(w*t) (no window leakage on non-integer cycles).
        let w = 2.0 * PI * freq_hz / sample_rate as f64;
        let mut s_cc = 0.0_f64;
        let mut s_ss = 0.0_f64;
        let mut s_cs = 0.0_f64;
        let mut s_yc = 0.0_f64;
        let mut s_ys = 0.0_f64;
        for (i, &s) in slice.iter().enumerate() {
            let idx = (trim_edges + i) as f64;
            let c = (w * idx).cos();
            let sn = (w * idx).sin();
            let y = s as f64;
            s_cc += c * c;
            s_ss += sn * sn;
            s_cs += c * sn;
            s_yc += y * c;
            s_ys += y * sn;
        }
        let det = s_cc * s_ss - s_cs * s_cs;
        let a = (s_yc * s_ss - s_ys * s_cs) / det;
        let b = (s_cc * s_ys - s_cs * s_yc) / det;

        let mut sig_pow = 0.0_f64;
        let mut err_pow = 0.0_f64;
        for (i, &s) in slice.iter().enumerate() {
            let idx = (trim_edges + i) as f64;
            let ref_val = a * (w * idx).cos() + b * (w * idx).sin();
            let diff = s as f64 - ref_val;
            sig_pow += ref_val * ref_val;
            err_pow += diff * diff;
        }
        10.0 * (sig_pow / err_pow.max(1e-30)).log10()
    }

    #[test]
    fn optimizer_downmix_before_resample_and_upmix_after_resample() {
        // Downmix: 5.1 (6ch) -> Stereo (2ch) + gain + resample 48k -> 44.1k
        let src_5_1 = AudioContract::new(
            SampleFormat::F32Planar,
            48_000,
            ChannelLayout::surround_5_1(),
        );
        let downmix = AudioOpList::builder(src_5_1.clone())
            .channels(ChannelLayout::stereo())
            .sample_rate(44_100)
            .gain_db(-6.0)
            .build()
            .expect("downmix build");

        let rematrix_idx = downmix
            .ops
            .iter()
            .position(|op| matches!(op, AudioOp::Rematrix(_)))
            .expect("has rematrix");
        let resample_idx = downmix
            .ops
            .iter()
            .position(|op| matches!(op, AudioOp::Resample(_)))
            .expect("has resample");
        assert!(
            rematrix_idx < resample_idx,
            "downmix Rematrix must be ordered BEFORE Resample: {:?}",
            downmix.ops
        );
        assert_eq!(
            downmix
                .ops
                .iter()
                .filter(|op| matches!(op, AudioOp::Rematrix(_)))
                .count(),
            1,
            "layout matrix and gain_db must fuse into ONE Rematrix"
        );

        // Verify exact fused coefficient: FL->FL should equal 10^(-6/20)
        if let AudioOp::Rematrix(m) = &downmix.ops[rematrix_idx] {
            let expected_gain = 10.0_f64.powf(-6.0 / 20.0);
            assert!((m.get(0, 0) - expected_gain).abs() < 1e-12);
            assert!((m.get(0, 2) - FRAC_1_SQRT_2 * expected_gain).abs() < 1e-12);
        }

        // Upmix: Stereo (2ch) -> 5.1 (6ch) + resample 44.1k -> 48k
        let src_stereo =
            AudioContract::new(SampleFormat::F32Planar, 44_100, ChannelLayout::stereo());
        let upmix = AudioOpList::builder(src_stereo)
            .channels(ChannelLayout::surround_5_1())
            .sample_rate(48_000)
            .gain_db(-3.0)
            .build()
            .expect("upmix build");

        let rematrix_idx = upmix
            .ops
            .iter()
            .position(|op| matches!(op, AudioOp::Rematrix(_)))
            .expect("has rematrix");
        let resample_idx = upmix
            .ops
            .iter()
            .position(|op| matches!(op, AudioOp::Resample(_)))
            .expect("has resample");
        assert!(
            resample_idx < rematrix_idx,
            "upmix Rematrix must be ordered AFTER Resample: {:?}",
            upmix.ops
        );

        // Explicit matrix before Resample + gain after Resample on upmix must still fuse and move after Resample.
        let mut manual = AudioOpList {
            src: AudioContract::new(SampleFormat::F32Planar, 44_100, ChannelLayout::stereo()),
            dst: AudioContract::new(
                SampleFormat::F32Planar,
                48_000,
                ChannelLayout::surround_5_1(),
            ),
            ops: vec![
                AudioOp::Read(SampleFormat::F32Planar),
                AudioOp::Rematrix(
                    build_rematrix_matrix(
                        &ChannelLayout::stereo(),
                        &ChannelLayout::surround_5_1(),
                        &RematrixOptions::default(),
                    )
                    .unwrap(),
                ),
                AudioOp::Resample(SwrPolyphaseConfig::new(44_100, 48_000)),
                AudioOp::Rematrix(ChannelMatrix::gain(6, 0.5)),
                AudioOp::Write(SampleFormat::F32Planar),
            ],
        };
        manual.optimize();
        assert_eq!(
            manual.ops.len(),
            4,
            "expected [Read, Resample, Rematrix, Write], got {:?}",
            manual.ops
        );
        assert!(matches!(manual.ops[1], AudioOp::Resample(_)));
        assert!(matches!(manual.ops[2], AudioOp::Rematrix(_)));
        if let AudioOp::Rematrix(m) = &manual.ops[2] {
            assert!((m.get(0, 0) - 0.5).abs() < 1e-12);
            assert!((m.get(1, 1) - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn optimizer_elides_identity_and_detects_noop() {
        let contract = AudioContract::new(SampleFormat::S16, 48_000, ChannelLayout::stereo());
        let ops = AudioOpList::builder(contract.clone())
            .gain_db(-6.0)
            .gain_db(6.0)
            .dither(AudioDither::None)
            .build()
            .expect("noop build");

        assert!(
            ops.is_noop(),
            "inverse gains + same contract should be noop: {:?}",
            ops.ops
        );
        assert_eq!(
            ops.ops,
            vec![
                AudioOp::Read(SampleFormat::S16),
                AudioOp::Write(SampleFormat::S16)
            ]
        );

        let mut compiled = CompiledAudioOps::compile(&ops).expect("compile noop");
        assert!(compiled.is_noop());

        let pool = BufferPool::new();
        let blk = AudioBlock::new(
            vec![Buffer::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8])],
            2,
            SampleFormat::S16,
            48_000,
            ChannelLayout::stereo(),
        );
        let out = compiled.process(blk, &pool).expect("process noop");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].planes[0].data(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn rematrix_5_1_to_stereo_exact_ffmpeg_coefficients_and_normalization() {
        let m = build_rematrix_matrix(
            &ChannelLayout::surround_5_1(),
            &ChannelLayout::stereo(),
            &RematrixOptions::default(),
        )
        .expect("5.1 -> 2.0");

        assert_eq!(m.in_channels, 6);
        assert_eq!(m.out_channels, 2);
        // Channel order in 5.1: [FL, FR, FC, LFE, BL, BR]
        let expected_fl = [1.0, 0.0, FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2, 0.0];
        let expected_fr = [0.0, 1.0, FRAC_1_SQRT_2, 0.0, 0.0, FRAC_1_SQRT_2];
        for i in 0..6 {
            assert!((m.get(0, i) - expected_fl[i]).abs() < 1e-15);
            assert!((m.get(1, i) - expected_fr[i]).abs() < 1e-15);
        }

        // Normalized variant: row sum == 1.0 + sqrt(2)
        let norm_m = build_rematrix_matrix(
            &ChannelLayout::surround_5_1(),
            &ChannelLayout::stereo(),
            &RematrixOptions::normalized(),
        )
        .expect("5.1 -> 2.0 normalized");
        let div = 1.0 + SQRT_2;
        for i in 0..6 {
            assert!((norm_m.get(0, i) - expected_fl[i] / div).abs() < 1e-15);
            assert!((norm_m.get(1, i) - expected_fr[i] / div).abs() < 1e-15);
        }

        // 7.1 -> 5.1: SL/SR fold into BL/BR at SQRT1_2
        let m71 = build_rematrix_matrix(
            &ChannelLayout::surround_7_1(),
            &ChannelLayout::surround_5_1(),
            &RematrixOptions::default(),
        )
        .expect("7.1 -> 5.1");
        assert_eq!(m71.in_channels, 8);
        assert_eq!(m71.out_channels, 6);
        assert!((m71.get(4, 4) - 1.0).abs() < 1e-15);
        assert!((m71.get(4, 6) - FRAC_1_SQRT_2).abs() < 1e-15);
        assert!((m71.get(5, 5) - 1.0).abs() < 1e-15);
        assert!((m71.get(5, 7) - FRAC_1_SQRT_2).abs() < 1e-15);

        // Stereo <-> Mono
        let st_to_mono = build_rematrix_matrix(
            &ChannelLayout::stereo(),
            &ChannelLayout::mono(),
            &RematrixOptions::default(),
        )
        .unwrap();
        assert!((st_to_mono.get(0, 0) - FRAC_1_SQRT_2).abs() < 1e-15);
        assert!((st_to_mono.get(0, 1) - FRAC_1_SQRT_2).abs() < 1e-15);

        let mono_to_st = build_rematrix_matrix(
            &ChannelLayout::mono(),
            &ChannelLayout::stereo(),
            &RematrixOptions::default(),
        )
        .unwrap();
        assert!((mono_to_st.get(0, 0) - FRAC_1_SQRT_2).abs() < 1e-15);
        assert!((mono_to_st.get(1, 0) - FRAC_1_SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn bessel_i0_matches_reference_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-13);
        assert!((bessel_i0(9.0) - 1_093.588_354_511_374_7).abs() < 1e-9);
        assert!((bessel_i0(16.0) - 893_446.227_920_105_1).abs() < 1e-6);
    }

    #[test]
    fn resample_48khz_to_44100hz_and_16khz_sine_snr_above_90db() {
        let pool = BufferPool::new();
        for &(out_rate, freq_hz) in &[(44_100u32, 1000.0_f64), (16_000u32, 1000.0_f64)] {
            let src = AudioContract::new(SampleFormat::F32Planar, 48_000, ChannelLayout::stereo());
            let list = AudioOpList::builder(src)
                .sample_rate(out_rate)
                .build()
                .expect("build resampler");
            let mut compiled = CompiledAudioOps::compile(&list).expect("compile");

            let total_in = 48_000usize;
            let mut out_blocks = Vec::new();
            let mut fed = 0usize;
            while fed < total_in {
                let chunk = (total_in - fed).min(1536);
                let blk = make_f32p_sine_block(
                    48_000,
                    ChannelLayout::stereo(),
                    chunk,
                    freq_hz,
                    0.8,
                    Timestamp::new(fed as i64),
                    Rational::new(1, 48_000),
                    fed,
                );
                compiled.process_into(blk, &pool, &mut out_blocks).unwrap();
                fed += chunk;
            }
            compiled.flush_into(&pool, &mut out_blocks).unwrap();

            let ch0 = read_f32p_channel(&out_blocks, 0);
            assert_eq!(
                ch0.len(),
                out_rate as usize,
                "1.0s at {out_rate}Hz should yield {out_rate} samples, got {}",
                ch0.len()
            );

            let snr = measure_sine_snr_db(&ch0, out_rate, freq_hz, 256);
            assert!(
                snr > 90.0,
                "48kHz -> {out_rate}Hz sine SNR = {snr:.2} dB (expected > 90 dB)"
            );
        }
    }

    #[test]
    fn resample_fractional_clock_drift_ratio_and_delay_tracking() {
        let cfg = SwrPolyphaseConfig::new(48_000, 48_000);
        let mut r = PolyphaseResampler::new(cfg, 1, true).unwrap();
        // +1% VFR clock drift
        r.set_fractional_ratio(1.01).unwrap();

        let n_in = 10_000usize;
        let input: Vec<f64> = (0..n_in)
            .map(|i| (2.0 * PI * 440.0 * (i as f64 / 48_000.0)).sin())
            .collect();
        r.push_f64(&[&input], n_in).unwrap();
        let delay = r.delay_samples(48_000);
        assert!(delay > 0);
        r.flush();

        let mut out = vec![0.0_f64; 12_000];
        let mut total = 0usize;
        loop {
            let n = r.drain_f64(&mut [&mut out[total..]], 1024).unwrap();
            if n == 0 {
                break;
            }
            total += n;
        }
        // Expect ~10100 samples for 1.01x ratio
        assert!(
            (total as isize - 10_100).abs() <= 4,
            "expected ~10100 samples at 1.01x drift, got {total}"
        );
    }

    #[test]
    fn sample_trim_exact_start_and_end_with_vfr_pts() {
        let pool = BufferPool::new();
        let src = AudioContract::new(SampleFormat::F32Planar, 48_000, ChannelLayout::mono());
        let list = AudioOpList::builder(src)
            .trim(SampleTrim {
                skip_start: 150,
                skip_end: 250,
            })
            .build()
            .unwrap();
        let mut compiled = CompiledAudioOps::compile(&list).unwrap();

        // Feed 3 blocks: 100, 500, 400 samples = 1000 samples total (pts in 1/48000)
        let mut out = Vec::new();
        let sizes = [100usize, 500, 400];
        let mut cursor = 0usize;
        for &sz in &sizes {
            let mut bytes = vec![0u8; sz * 4];
            for i in 0..sz {
                let val = (cursor + i) as f32;
                bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_ne_bytes());
            }
            let mut blk = AudioBlock::new(
                vec![Buffer::from_slice(&bytes)],
                sz as u32,
                SampleFormat::F32Planar,
                48_000,
                ChannelLayout::mono(),
            );
            blk.pts = Timestamp::new(cursor as i64);
            blk.time_base = Rational::new(1, 48_000);
            blk.duration = sz as i64;
            compiled.process_into(blk, &pool, &mut out).unwrap();
            cursor += sz;
        }
        compiled.flush_into(&pool, &mut out).unwrap();

        let kept = read_f32p_channel(&out, 0);
        // 1000 - 150 - 250 = 600 samples, values [150 .. 750)
        assert_eq!(kept.len(), 600);
        for (i, &v) in kept.iter().enumerate() {
            assert_eq!(v, (150 + i) as f32);
        }
        assert_eq!(out[0].pts, Timestamp::new(150));
        let total_dur: i64 = out.iter().map(|b| b.duration).sum();
        assert_eq!(total_dur, 600);
    }

    #[test]
    fn reframe_1024_and_flush_preserves_sample_count_and_vfr_timestamps() {
        let pool = BufferPool::new();
        let tb = Rational::new(1, 90_000);
        let src = AudioContract::new(SampleFormat::F32Planar, 48_000, ChannelLayout::stereo());
        let list = AudioOpList::builder(src).reframe(1024).build().unwrap();
        let mut compiled = CompiledAudioOps::compile(&list).unwrap();

        // Feed irregular VFR-sized chunks: 333, 777, 1500, 400 = 3010 samples
        let chunks = [333usize, 777, 1500, 400];
        let mut out = Vec::new();
        let mut fed = 0usize;
        for &sz in &chunks {
            let pts = Timestamp::new(wedeo_core::rescale_q(
                fed as i64,
                Rational::new(1, 48_000),
                tb,
            ));
            let blk = make_f32p_sine_block(
                48_000,
                ChannelLayout::stereo(),
                sz,
                440.0,
                0.5,
                pts,
                tb,
                fed,
            );
            compiled.process_into(blk, &pool, &mut out).unwrap();
            fed += sz;
        }
        compiled.flush_into(&pool, &mut out).unwrap();

        // 3010 = 1024 + 1024 + 962
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].nb_samples, 1024);
        assert_eq!(out[1].nb_samples, 1024);
        assert_eq!(out[2].nb_samples, 962);
        assert_eq!(out[0].time_base, tb);
        assert_eq!(out[0].pts, Timestamp::new(0));
        assert_eq!(out[0].pts.checked_add(out[0].duration), Some(out[1].pts));
        assert_eq!(out[1].pts.checked_add(out[1].duration), Some(out[2].pts));
        let expected_end = wedeo_core::rescale_q(3010, Rational::new(1, 48_000), tb);
        assert_eq!(out[2].pts.raw() + out[2].duration, expected_end);
    }

    #[test]
    fn sample_formats_roundtrip_dither_and_clip_accounting() {
        let pool = BufferPool::new();
        let all_formats = [
            SampleFormat::U8,
            SampleFormat::S16,
            SampleFormat::S32,
            SampleFormat::S64,
            SampleFormat::F32,
            SampleFormat::F64,
            SampleFormat::U8Planar,
            SampleFormat::S16Planar,
            SampleFormat::S32Planar,
            SampleFormat::S64Planar,
            SampleFormat::F32Planar,
            SampleFormat::F64Planar,
        ];

        // Roundtrip F64Planar -> every format -> F64Planar
        let n = 64usize;
        let layout = ChannelLayout::stereo();
        let src_vals: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let s = ((i as f64 / n as f64) * 2.0 - 1.0) * 0.75;
                [s, -s]
            })
            .collect();

        for &mid_fmt in &all_formats {
            let mut p0 = vec![0u8; n * 8];
            let mut p1 = vec![0u8; n * 8];
            for i in 0..n {
                p0[i * 8..(i + 1) * 8].copy_from_slice(&src_vals[i][0].to_ne_bytes());
                p1[i * 8..(i + 1) * 8].copy_from_slice(&src_vals[i][1].to_ne_bytes());
            }
            let src_block = AudioBlock::new(
                vec![Buffer::from_slice(&p0), Buffer::from_slice(&p1)],
                n as u32,
                SampleFormat::F64Planar,
                48_000,
                layout.clone(),
            );

            let to_mid = AudioOpList::builder(AudioContract::new(
                SampleFormat::F64Planar,
                48_000,
                layout.clone(),
            ))
            .format(mid_fmt)
            .build()
            .unwrap();
            let to_f64 = AudioOpList::builder(AudioContract::new(mid_fmt, 48_000, layout.clone()))
                .format(SampleFormat::F64Planar)
                .build()
                .unwrap();

            let mut c1 = CompiledAudioOps::compile(&to_mid).unwrap();
            let mut c2 = CompiledAudioOps::compile(&to_f64).unwrap();
            let mid_blocks = c1.process(src_block, &pool).unwrap();
            assert!(c1.health().is_clean());
            let back_blocks = c2
                .process(mid_blocks.into_iter().next().unwrap(), &pool)
                .unwrap();

            let tol = match mid_fmt {
                SampleFormat::U8 | SampleFormat::U8Planar => 1.0 / 128.0 + 1e-6,
                SampleFormat::S16 | SampleFormat::S16Planar => 1.0 / 32768.0 + 1e-6,
                _ => 1e-6,
            };
            let b0 = back_blocks[0].planes[0].data();
            let b1 = back_blocks[0].planes[1].data();
            for i in 0..n {
                let r0 = f64::from_ne_bytes(b0[i * 8..(i + 1) * 8].try_into().unwrap());
                let r1 = f64::from_ne_bytes(b1[i * 8..(i + 1) * 8].try_into().unwrap());
                assert!(
                    (r0 - src_vals[i][0]).abs() <= tol,
                    "format {mid_fmt} ch0 sample {i}: got {r0}, expected {}",
                    src_vals[i][0]
                );
                assert!(
                    (r1 - src_vals[i][1]).abs() <= tol,
                    "format {mid_fmt} ch1 sample {i}: got {r1}, expected {}",
                    src_vals[i][1]
                );
            }
        }

        // Deterministic TPDF dither & clipping accounting into StreamHealth
        let list_clip = AudioOpList::builder(AudioContract::new(
            SampleFormat::F32Planar,
            48_000,
            ChannelLayout::mono(),
        ))
        .format(SampleFormat::S16)
        .dither(AudioDither::Triangular)
        .build()
        .unwrap();

        let mut c_a = CompiledAudioOps::compile(&list_clip).unwrap();
        let mut c_b = CompiledAudioOps::compile(&list_clip).unwrap();
        // Samples: [1.5 (clips), -1.5 (clips), 0.25, -0.25]
        let mut bytes = Vec::new();
        for &s in &[1.5_f32, -1.5_f32, 0.25_f32, -0.25_f32] {
            bytes.extend_from_slice(&s.to_ne_bytes());
        }
        let blk_a = AudioBlock::new(
            vec![Buffer::from_slice(&bytes)],
            4,
            SampleFormat::F32Planar,
            48_000,
            ChannelLayout::mono(),
        );
        let blk_b = blk_a.clone();
        let out_a = c_a.process(blk_a, &pool).unwrap();
        let out_b = c_b.process(blk_b, &pool).unwrap();

        // Deterministic PRNG produces bit-identical output across runs.
        assert_eq!(out_a[0].planes[0].data(), out_b[0].planes[0].data());
        assert_eq!(c_a.health().clipped_audio_samples, 2);
        assert_eq!(c_a.health().events.len(), 1);
    }

    #[test]
    fn end_to_end_5_1_to_stereo_48k_to_44100_reframe_1024_and_vfr_gap() {
        let pool = BufferPool::new();
        let src = AudioContract::new(
            SampleFormat::F32Planar,
            48_000,
            ChannelLayout::surround_5_1(),
        );
        let list = AudioOpList::builder(src)
            .channels(ChannelLayout::stereo())
            .sample_rate(44_100)
            .reframe(1024)
            .format(SampleFormat::S16)
            .dither(AudioDither::Triangular)
            .build()
            .unwrap();
        let mut compiled = CompiledAudioOps::compile(&list).unwrap();

        // Feed 0.5s (24000 samples) at pts=0, then a VFR gap jumping to pts=48000 for 0.5s
        let tb = Rational::new(1, 48_000);
        let mut out = Vec::new();
        let b0 = make_f32p_sine_block(
            48_000,
            ChannelLayout::surround_5_1(),
            24_000,
            1000.0,
            0.2,
            Timestamp::new(0),
            tb,
            0,
        );
        compiled.process_into(b0, &pool, &mut out).unwrap();
        let b1 = make_f32p_sine_block(
            48_000,
            ChannelLayout::surround_5_1(),
            24_000,
            1000.0,
            0.2,
            Timestamp::new(48_000),
            tb,
            24_000,
        );
        compiled.process_into(b1, &pool, &mut out).unwrap();
        compiled.flush_into(&pool, &mut out).unwrap();

        let total_samples: u32 = out.iter().map(|b| b.nb_samples).sum();
        assert_eq!(total_samples, 44_100);
        assert!(out[..out.len() - 1].iter().all(|b| b.nb_samples == 1024));
        assert!(out.iter().all(|b| b.format == SampleFormat::S16));
        assert!(out.iter().all(|b| b.sample_rate == 44_100));
        assert!(
            out.iter()
                .all(|b| b.channel_layout == ChannelLayout::stereo())
        );
        // Output time_base switched cleanly from 1/48000 -> 1/44100; after 22050 samples (0.5s),
        // the VFR jump of +1.0s (44100 ticks) appears!
        assert_eq!(out[0].time_base, Rational::new(1, 44_100));
        assert_eq!(out[0].pts, Timestamp::new(0));
        assert!(out.last().unwrap().pts.raw() >= 44_100);
    }

    #[test]
    fn short_stream_resample_and_dolby_rematrix_encoding() {
        let pool = BufferPool::new();
        let src = AudioContract::new(SampleFormat::F32Planar, 48_000, ChannelLayout::mono());
        let list = AudioOpList::builder(src)
            .sample_rate(44_100)
            .build()
            .unwrap();
        let mut compiled = CompiledAudioOps::compile(&list).unwrap();

        // Feed fewer samples (16) than filter_length (~36), then flush immediately.
        let blk = make_f32p_sine_block(
            48_000,
            ChannelLayout::mono(),
            16,
            500.0,
            0.5,
            Timestamp::new(0),
            Rational::new(1, 48_000),
            0,
        );
        let mut out = compiled.process(blk, &pool).unwrap();
        compiled.flush_into(&pool, &mut out).unwrap();
        let n: u32 = out.iter().map(|b| b.nb_samples).sum();
        assert!(
            (14..=16).contains(&n),
            "expected ~15 samples from 16@48k->44.1k, got {n}"
        );

        // Dolby Pro Logic II 5.1 -> stereo matrix coefficients check
        let dpl2 = build_rematrix_matrix(
            &ChannelLayout::surround_5_1(),
            &ChannelLayout::stereo(),
            &RematrixOptions {
                matrix_encoding: MatrixEncoding::DplII,
                ..RematrixOptions::default()
            },
        )
        .unwrap();
        // FL gets negative BL/BR and FR gets positive BL/BR in DPLII
        assert!(dpl2.get(0, 4) < 0.0);
        assert!(dpl2.get(0, 5) < 0.0);
        assert!(dpl2.get(1, 4) > 0.0);
        assert!(dpl2.get(1, 5) > 0.0);
    }
}
