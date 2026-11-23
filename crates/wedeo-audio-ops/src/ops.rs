//! Audio operation IR (`AudioOp`, `AudioOpList`) and fluent builder API.

use wedeo_core::{ChannelLayout, Error, ReplayGain, Result, SampleFormat, SampleTrim};

use crate::optimizer;
use crate::rematrix::{RematrixOptions, build_rematrix_matrix};

/// Contract describing an audio stream's sample format, rate, layout, and optional framing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AudioContract {
    pub format: SampleFormat,
    pub sample_rate: u32,
    pub channel_layout: ChannelLayout,
    pub frame_size: Option<u32>,
}

impl AudioContract {
    /// Construct an `AudioContract` without fixed reframing.
    pub fn new(format: SampleFormat, sample_rate: u32, channel_layout: ChannelLayout) -> Self {
        Self {
            format,
            sample_rate,
            channel_layout,
            frame_size: None,
        }
    }

    /// Set an optional fixed output frame size (e.g. `1024` for AAC, `960` for Opus).
    pub fn with_frame_size(mut self, frame_size: Option<u32>) -> Self {
        self.frame_size = frame_size;
        self
    }

    /// Validate that sample rate, channel count, and optional frame size are well-formed.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(Error::invalid_argument(
                "audio_contract",
                "sample_rate must be non-zero",
            ));
        }
        if self.channel_layout.nb_channels == 0 {
            return Err(Error::invalid_argument(
                "audio_contract",
                "channel_layout must have at least 1 channel",
            ));
        }
        if !self.channel_layout.channels.is_empty()
            && self.channel_layout.channels.len() != self.channel_layout.nb_channels as usize
        {
            return Err(Error::invalid_argument(
                "audio_contract",
                "channel_layout nb_channels does not match channels vector length",
            ));
        }
        if let Some(0) = self.frame_size {
            return Err(Error::invalid_argument(
                "audio_contract",
                "frame_size must be non-zero when specified",
            ));
        }
        Ok(())
    }
}

/// Audio quantization dithering mode (matching FFmpeg `libswresample`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum AudioDither {
    #[default]
    None,
    Triangular,
    Rectangular,
}

/// Configuration for the FFmpeg `n9.1` `libswresample` Kaiser-windowed sinc polyphase resampler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwrPolyphaseConfig {
    pub in_rate: u32,
    pub out_rate: u32,
    pub filter_size: u32,
    pub phase_shift: u32,
    pub kaiser_beta: f64,
    pub linear_interp: bool,
}

impl SwrPolyphaseConfig {
    /// Construct a polyphase config with FFmpeg `libswresample` defaults
    /// (`filter_size = 32`, `phase_shift = 10`, `kaiser_beta = 9.0`, `linear_interp = true`).
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        Self {
            in_rate,
            out_rate,
            filter_size: 32,
            phase_shift: 10,
            kaiser_beta: 9.0,
            linear_interp: true,
        }
    }

    /// Validate resampler parameters.
    pub fn validate(&self) -> Result<()> {
        if self.in_rate == 0 || self.out_rate == 0 {
            return Err(Error::invalid_argument(
                "resample",
                "in_rate and out_rate must be non-zero",
            ));
        }
        if self.filter_size == 0 {
            return Err(Error::invalid_argument(
                "resample",
                "filter_size must be non-zero",
            ));
        }
        if self.phase_shift > 24 {
            return Err(Error::invalid_argument(
                "resample",
                "phase_shift must be <= 24",
            ));
        }
        if !self.kaiser_beta.is_finite() || self.kaiser_beta < 0.0 {
            return Err(Error::invalid_argument(
                "resample",
                "kaiser_beta must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

impl Default for SwrPolyphaseConfig {
    fn default() -> Self {
        Self::new(48_000, 48_000)
    }
}

/// Row-major `out_channels x in_channels` linear audio mixing matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelMatrix {
    pub in_channels: usize,
    pub out_channels: usize,
    pub coeffs: Vec<f64>,
}

impl ChannelMatrix {
    /// Construct a validated channel matrix from row-major coefficients (`out * in_channels + in`).
    pub fn new(in_channels: usize, out_channels: usize, coeffs: Vec<f64>) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(Error::invalid_argument(
                "channel_matrix",
                "in_channels and out_channels must be non-zero",
            ));
        }
        let expected = in_channels
            .checked_mul(out_channels)
            .ok_or_else(|| Error::invalid_argument("channel_matrix", "matrix size overflow"))?;
        if coeffs.len() != expected {
            return Err(Error::invalid_argument(
                "channel_matrix",
                format!(
                    "expected {expected} coefficients for {out_channels}x{in_channels} matrix, got {}",
                    coeffs.len()
                ),
            ));
        }
        if coeffs.iter().any(|c| !c.is_finite()) {
            return Err(Error::invalid_argument(
                "channel_matrix",
                "matrix coefficients must be finite",
            ));
        }
        Ok(Self {
            in_channels,
            out_channels,
            coeffs,
        })
    }

    /// Construct a zeroed `out_channels x in_channels` matrix.
    pub fn zeros(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            coeffs: vec![0.0; in_channels * out_channels],
        }
    }

    /// Construct an identity matrix for `channels` channels.
    pub fn identity(channels: usize) -> Self {
        Self::gain(channels, 1.0)
    }

    /// Construct a diagonal gain matrix (`gain * I`) for `channels` channels.
    pub fn gain(channels: usize, gain_linear: f64) -> Self {
        let mut coeffs = vec![0.0; channels * channels];
        for i in 0..channels {
            coeffs[i * channels + i] = gain_linear;
        }
        Self {
            in_channels: channels,
            out_channels: channels,
            coeffs,
        }
    }

    /// Construct a diagonal gain matrix from a gain specified in decibels.
    pub fn gain_db(channels: usize, gain_db: f64) -> Self {
        let linear = 10.0_f64.powf(gain_db / 20.0);
        Self::gain(channels, linear)
    }

    /// Read coefficient at `(out_ch, in_ch)`.
    #[inline]
    pub fn get(&self, out_ch: usize, in_ch: usize) -> f64 {
        self.coeffs[out_ch * self.in_channels + in_ch]
    }

    /// Set coefficient at `(out_ch, in_ch)`.
    #[inline]
    pub fn set(&mut self, out_ch: usize, in_ch: usize, value: f64) {
        self.coeffs[out_ch * self.in_channels + in_ch] = value;
    }

    /// Compose `self` followed by `next` (`out = next * self`).
    pub fn then(&self, next: &Self) -> Result<Self> {
        if self.out_channels != next.in_channels {
            return Err(Error::invalid_argument(
                "channel_matrix",
                format!(
                    "cannot compose {}x{} matrix followed by {}x{} matrix",
                    self.out_channels, self.in_channels, next.out_channels, next.in_channels
                ),
            ));
        }
        let mut out = Self::zeros(self.in_channels, next.out_channels);
        for o in 0..next.out_channels {
            for i in 0..self.in_channels {
                let mut sum = 0.0_f64;
                for k in 0..self.out_channels {
                    sum += next.get(o, k) * self.get(k, i);
                }
                out.set(o, i, sum);
            }
        }
        Ok(out)
    }

    /// Returns `true` if this matrix is square and within `1e-12` of the identity matrix.
    pub fn is_identity(&self) -> bool {
        if self.in_channels != self.out_channels {
            return false;
        }
        for o in 0..self.out_channels {
            for i in 0..self.in_channels {
                let target = if o == i { 1.0 } else { 0.0 };
                if (self.get(o, i) - target).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }
}

/// Single audio processing operation inside an [`AudioOpList`].
#[derive(Debug, Clone, PartialEq)]
pub enum AudioOp {
    Read(SampleFormat),
    Trim(SampleTrim),
    Rematrix(ChannelMatrix),
    Resample(SwrPolyphaseConfig),
    Dither(AudioDither),
    Reframe { frame_size: u32 },
    Write(SampleFormat),
}

/// High-level declarative audio conversion pipeline mirroring `SwsOpList`.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioOpList {
    pub src: AudioContract,
    pub dst: AudioContract,
    pub ops: Vec<AudioOp>,
}

impl AudioOpList {
    /// Start a fluent [`AudioOpListBuilder`] from a source [`AudioContract`].
    pub fn builder(src: AudioContract) -> AudioOpListBuilder {
        AudioOpListBuilder::new(src)
    }

    /// Build and optimize an `AudioOpList` converting directly from `src` to `dst`.
    pub fn build(src: AudioContract, dst: AudioContract) -> Result<Self> {
        let mut builder = AudioOpListBuilder::new(src)
            .format(dst.format)
            .sample_rate(dst.sample_rate)
            .channels(dst.channel_layout);
        if let Some(fs) = dst.frame_size {
            builder = builder.reframe(fs);
        }
        builder.build()
    }

    /// Fuse linear channel/gain matrices, reorder `Rematrix` vs `Resample` for minimum cost,
    /// and elide identity / no-op operations.
    pub fn optimize(&mut self) {
        optimizer::optimize_ops(&mut self.ops);
    }

    /// Returns `true` when this list is a pure zero-copy pass-through.
    pub fn is_noop(&self) -> bool {
        optimizer::is_noop(&self.src, &self.dst, &self.ops)
    }
}

enum LinearStep {
    GainLinear(f64),
    Matrix(ChannelMatrix),
}

/// Fluent builder for constructing and optimizing an [`AudioOpList`].
pub struct AudioOpListBuilder {
    src: AudioContract,
    dst_format: SampleFormat,
    dst_sample_rate: u32,
    dst_channel_layout: ChannelLayout,
    dst_frame_size: Option<u32>,
    trim: Option<SampleTrim>,
    dither: AudioDither,
    linear_steps: Vec<LinearStep>,
    rematrix_opts: RematrixOptions,
    filter_size: u32,
    phase_shift: u32,
    kaiser_beta: f64,
    linear_interp: bool,
}

impl AudioOpListBuilder {
    /// Create a builder whose initial target contract matches `src`.
    pub fn new(src: AudioContract) -> Self {
        let dst_format = src.format;
        let dst_sample_rate = src.sample_rate;
        let dst_channel_layout = src.channel_layout.clone();
        let dst_frame_size = src.frame_size;
        Self {
            src,
            dst_format,
            dst_sample_rate,
            dst_channel_layout,
            dst_frame_size,
            trim: None,
            dither: AudioDither::None,
            linear_steps: Vec::new(),
            rematrix_opts: RematrixOptions::default(),
            filter_size: 32,
            phase_shift: 10,
            kaiser_beta: 9.0,
            linear_interp: true,
        }
    }

    /// Set target sample rate in Hz.
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.dst_sample_rate = sample_rate;
        self
    }

    /// Set target channel layout.
    pub fn channels(mut self, channel_layout: ChannelLayout) -> Self {
        self.dst_channel_layout = channel_layout;
        self
    }

    /// Set target sample format.
    pub fn format(mut self, format: SampleFormat) -> Self {
        self.dst_format = format;
        self
    }

    /// Apply a linear gain specified in decibels.
    pub fn gain_db(mut self, gain_db: f64) -> Self {
        self.linear_steps
            .push(LinearStep::GainLinear(10.0_f64.powf(gain_db / 20.0)));
        self
    }

    /// Apply a linear gain multiplier.
    pub fn gain(mut self, gain_linear: f64) -> Self {
        self.linear_steps.push(LinearStep::GainLinear(gain_linear));
        self
    }

    /// Apply ReplayGain metadata (`track` or `album` gain in 1/100,000 dB).
    pub fn replay_gain(self, rg: ReplayGain, album: bool) -> Self {
        let gain_hundred_thousandths = if album { rg.album_gain } else { rg.track_gain };
        let db = gain_hundred_thousandths as f64 / 100_000.0;
        self.gain_db(db)
    }

    /// Append a custom channel mixing/pan matrix.
    pub fn rematrix(mut self, matrix: ChannelMatrix) -> Self {
        self.linear_steps.push(LinearStep::Matrix(matrix));
        self
    }

    /// Configure automatic rematrixing options (center/surround/LFE levels, normalization).
    pub fn rematrix_options(mut self, opts: RematrixOptions) -> Self {
        self.rematrix_opts = opts;
        self
    }

    /// Enable or disable row-sum clipping normalization (`maxval = 1.0`) for channel downmix.
    pub fn normalize_rematrix(mut self, normalize: bool) -> Self {
        self.rematrix_opts.maxval = if normalize { Some(1.0) } else { None };
        self
    }

    /// Set quantization dither mode.
    pub fn dither(mut self, dither: AudioDither) -> Self {
        self.dither = dither;
        self
    }

    /// Set fixed output frame size in samples per channel.
    pub fn reframe(mut self, frame_size: u32) -> Self {
        self.dst_frame_size = Some(frame_size);
        self
    }

    /// Set gapless sample trim (`skip_start`, `skip_end`).
    pub fn trim(mut self, trim: SampleTrim) -> Self {
        self.trim = Some(trim);
        self
    }

    /// Set polyphase sinc filter tap count (default: 32).
    pub fn filter_size(mut self, filter_size: u32) -> Self {
        self.filter_size = filter_size;
        self
    }

    /// Set polyphase bank `log2(phase_count)` (default: 10 -> 1024 phases).
    pub fn phase_shift(mut self, phase_shift: u32) -> Self {
        self.phase_shift = phase_shift;
        self
    }

    /// Set Kaiser window beta parameter (default: 9.0).
    pub fn kaiser_beta(mut self, kaiser_beta: f64) -> Self {
        self.kaiser_beta = kaiser_beta;
        self
    }

    /// Enable or disable linear phase interpolation (default: true).
    pub fn linear_interp(mut self, linear_interp: bool) -> Self {
        self.linear_interp = linear_interp;
        self
    }

    /// Build the raw unoptimized [`AudioOpList`].
    pub fn build_unoptimized(self) -> Result<AudioOpList> {
        self.src.validate()?;
        let dst = AudioContract {
            format: self.dst_format,
            sample_rate: self.dst_sample_rate,
            channel_layout: self.dst_channel_layout,
            frame_size: self.dst_frame_size,
        };
        dst.validate()?;

        let mut ops = Vec::with_capacity(8);
        ops.push(AudioOp::Read(self.src.format));

        if let Some(trim) = self.trim
            && (trim.skip_start > 0 || trim.skip_end > 0)
        {
            ops.push(AudioOp::Trim(trim));
        }

        let mut current_ch = self.src.channel_layout.nb_channels as usize;
        let mut has_custom_matrix = false;
        for step in self.linear_steps {
            match step {
                LinearStep::GainLinear(g) => {
                    if !g.is_finite() {
                        return Err(Error::invalid_argument("audio_ops", "gain must be finite"));
                    }
                    ops.push(AudioOp::Rematrix(ChannelMatrix::gain(current_ch, g)));
                }
                LinearStep::Matrix(m) => {
                    if m.in_channels != current_ch {
                        return Err(Error::invalid_argument(
                            "audio_ops",
                            format!(
                                "custom rematrix input channels ({}) does not match stream channels ({current_ch})",
                                m.in_channels
                            ),
                        ));
                    }
                    current_ch = m.out_channels;
                    has_custom_matrix = true;
                    ops.push(AudioOp::Rematrix(m));
                }
            }
        }

        if has_custom_matrix {
            if current_ch != dst.channel_layout.nb_channels as usize {
                return Err(Error::invalid_argument(
                    "audio_ops",
                    format!(
                        "custom rematrix output channels ({current_ch}) does not match destination channels ({})",
                        dst.channel_layout.nb_channels
                    ),
                ));
            }
        } else if self.src.channel_layout != dst.channel_layout
            || self.rematrix_opts.rematrix_volume != 1.0
        {
            let m = build_rematrix_matrix(
                &self.src.channel_layout,
                &dst.channel_layout,
                &self.rematrix_opts,
            )?;
            ops.push(AudioOp::Rematrix(m));
        }

        if self.src.sample_rate != dst.sample_rate {
            let cfg = SwrPolyphaseConfig {
                in_rate: self.src.sample_rate,
                out_rate: dst.sample_rate,
                filter_size: self.filter_size,
                phase_shift: self.phase_shift,
                kaiser_beta: self.kaiser_beta,
                linear_interp: self.linear_interp,
            };
            cfg.validate()?;
            ops.push(AudioOp::Resample(cfg));
        }

        if self.dither != AudioDither::None {
            ops.push(AudioOp::Dither(self.dither));
        }

        if let Some(frame_size) = dst.frame_size {
            ops.push(AudioOp::Reframe { frame_size });
        }

        ops.push(AudioOp::Write(dst.format));

        Ok(AudioOpList {
            src: self.src,
            dst,
            ops,
        })
    }

    /// Build and optimize the [`AudioOpList`] (fusing matrices, ordering downmix-before-resample
    /// and upmix-after-resample, and eliding identity/no-op steps).
    pub fn build(self) -> Result<AudioOpList> {
        let mut list = self.build_unoptimized()?;
        list.optimize();
        Ok(list)
    }
}
