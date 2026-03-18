// wedeo-resample: Audio resampling (libswresample equivalent)
// Wraps the rubato crate behind a simple interleaved f32 interface.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, PolynomialDegree, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use wedeo_core::error::{Error, Result};

/// Resampling quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Polynomial (cubic) interpolation — fastest, lowest quality.
    /// No anti-aliasing filter; suitable when speed matters more than fidelity.
    Fast,
    /// Sinc interpolation with moderate filter length — good balance.
    Normal,
    /// Sinc interpolation with long filter — best quality, highest CPU cost.
    High,
}

/// Audio resampler that accepts interleaved f32 samples.
///
/// Internally buffers partial chunks and drives rubato's fixed-input-size
/// resampler, producing interleaved f32 output.
pub struct Resampler {
    inner: Box<dyn rubato::Resampler<f32>>,
    channels: usize,
    from_rate: u32,
    to_rate: u32,
    /// rubato's required input chunk size (frames per channel).
    chunk_size: usize,
    /// Accumulator for input frames that don't fill a complete chunk.
    pending: Vec<f32>,
}

/// Default chunk size fed to rubato constructors.
const DEFAULT_CHUNK_SIZE: usize = 1024;

fn map_rubato_err<E: std::fmt::Display>(e: E) -> Error {
    Error::Other(format!("resample: {e}"))
}

impl Resampler {
    /// Create a new resampler.
    ///
    /// * `from_rate` / `to_rate` — input / output sample rates in Hz.
    /// * `channels` — number of interleaved channels (must be >= 1).
    /// * `quality` — algorithm selection (see [`Quality`]).
    pub fn new(from_rate: u32, to_rate: u32, channels: usize, quality: Quality) -> Result<Self> {
        if from_rate == 0 || to_rate == 0 {
            return Err(Error::InvalidArgument);
        }
        if channels == 0 {
            return Err(Error::InvalidArgument);
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let chunk_size = DEFAULT_CHUNK_SIZE;

        let inner: Box<dyn rubato::Resampler<f32>> = match quality {
            Quality::Fast => Box::new(
                Async::<f32>::new_poly(
                    ratio,
                    1.0,
                    PolynomialDegree::Cubic,
                    chunk_size,
                    channels,
                    FixedAsync::Input,
                )
                .map_err(map_rubato_err)?,
            ),
            Quality::Normal => {
                let params = SincInterpolationParameters {
                    sinc_len: 128,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 128,
                    window: WindowFunction::BlackmanHarris2,
                };
                Box::new(
                    Async::<f32>::new_sinc(
                        ratio,
                        1.0,
                        &params,
                        chunk_size,
                        channels,
                        FixedAsync::Input,
                    )
                    .map_err(map_rubato_err)?,
                )
            }
            Quality::High => {
                let params = SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Cubic,
                    oversampling_factor: 256,
                    window: WindowFunction::BlackmanHarris2,
                };
                Box::new(
                    Async::<f32>::new_sinc(
                        ratio,
                        1.0,
                        &params,
                        chunk_size,
                        channels,
                        FixedAsync::Input,
                    )
                    .map_err(map_rubato_err)?,
                )
            }
        };

        Ok(Self {
            inner,
            channels,
            from_rate,
            to_rate,
            chunk_size,
            pending: Vec::new(),
        })
    }

    /// Feed interleaved f32 samples and receive resampled interleaved output.
    ///
    /// Input length must be a multiple of `channels`. Output length will also
    /// be a multiple of `channels`.
    ///
    /// The resampler buffers internally, so output may be shorter or longer
    /// than a naive ratio calculation — call [`flush`](Self::flush) at the end
    /// to drain remaining samples.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if !input.len().is_multiple_of(self.channels) {
            return Err(Error::InvalidArgument);
        }

        // Append new data to pending buffer.
        self.pending.extend_from_slice(input);

        let samples_per_chunk = self.chunk_size * self.channels;
        let mut output = Vec::new();

        // Process as many full chunks as we can.
        while self.pending.len() >= samples_per_chunk {
            // Split off one chunk to avoid borrowing self.pending while calling
            // process_chunk(&mut self).
            let rest = self.pending.split_off(samples_per_chunk);
            let chunk = std::mem::replace(&mut self.pending, rest);
            self.process_chunk(&chunk, &mut output)?;
        }

        Ok(output)
    }

    /// Flush remaining buffered samples by zero-padding to a full chunk.
    ///
    /// Call this once after the last `process()` call to retrieve the tail
    /// of the resampled signal.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }

        let samples_per_chunk = self.chunk_size * self.channels;

        // Pad pending data to a full chunk with zeros.
        let pending_frames = self.pending.len() / self.channels;
        let partial_samples = self.pending.len() % self.channels;

        // If pending isn't frame-aligned, pad to the next frame boundary first.
        if partial_samples != 0 {
            self.pending
                .resize(self.pending.len() + (self.channels - partial_samples), 0.0);
        }

        // Now pad to a full chunk.
        self.pending.resize(samples_per_chunk, 0.0);

        let mut output = Vec::new();

        let input_adapter = InterleavedSlice::new(&self.pending, self.channels, self.chunk_size)
            .map_err(map_rubato_err)?;

        let result = self
            .inner
            .process(&input_adapter, 0, None)
            .map_err(map_rubato_err)?;

        // The result is an InterleavedOwned — extract the interleaved data.
        let out_data = result.take_data();
        let out_frames = self.inner.output_frames_next();
        // rubato may have allocated more than needed; only take valid frames.
        let valid_samples = out_frames.min(out_data.len() / self.channels) * self.channels;
        output.extend_from_slice(&out_data[..valid_samples]);

        // Trim output to the expected number of frames based on the real
        // pending frame count (not the zero-padded chunk).
        let expected_out_frames = self.output_frames_estimate(pending_frames);
        let expected_samples = expected_out_frames * self.channels;
        if output.len() > expected_samples {
            output.truncate(expected_samples);
        }

        self.pending.clear();
        Ok(output)
    }

    /// Estimate the number of output frames for a given number of input frames.
    pub fn output_frames_estimate(&self, input_frames: usize) -> usize {
        (input_frames as u64 * self.to_rate as u64).div_ceil(self.from_rate as u64) as usize
    }

    /// Get the input sample rate.
    pub fn from_rate(&self) -> u32 {
        self.from_rate
    }

    /// Get the output sample rate.
    pub fn to_rate(&self) -> u32 {
        self.to_rate
    }

    /// Get the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Reset the resampler state and clear all internal buffers.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.pending.clear();
    }

    /// Process exactly one chunk of interleaved data through rubato and
    /// append the interleaved output to `output`.
    fn process_chunk(&mut self, chunk: &[f32], output: &mut Vec<f32>) -> Result<()> {
        let frames = chunk.len() / self.channels;
        let input_adapter =
            InterleavedSlice::new(chunk, self.channels, frames).map_err(map_rubato_err)?;

        let result = self
            .inner
            .process(&input_adapter, 0, None)
            .map_err(map_rubato_err)?;

        let out_data = result.take_data();
        let out_frames = out_data.len() / self.channels;
        output.extend_from_slice(&out_data[..out_frames * self.channels]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_resampler_all_qualities() {
        for quality in [Quality::Fast, Quality::Normal, Quality::High] {
            let r = Resampler::new(44100, 48000, 2, quality);
            assert!(r.is_ok(), "Failed to create resampler with {quality:?}");
            let r = r.unwrap();
            assert_eq!(r.from_rate(), 44100);
            assert_eq!(r.to_rate(), 48000);
            assert_eq!(r.channels(), 2);
        }
    }

    #[test]
    fn invalid_params() {
        assert!(Resampler::new(0, 48000, 2, Quality::Fast).is_err());
        assert!(Resampler::new(44100, 0, 2, Quality::Fast).is_err());
        assert!(Resampler::new(44100, 48000, 0, Quality::Fast).is_err());
    }

    #[test]
    fn process_silence() {
        let mut r = Resampler::new(44100, 48000, 1, Quality::Fast).unwrap();
        let input = vec![0.0f32; 44100]; // 1 second mono
        let output = r.process(&input).unwrap();
        let tail = r.flush().unwrap();
        let total_frames = output.len() + tail.len();
        // Should be approximately 48000 frames (1 second at 48 kHz).
        // Allow generous margin for resampler latency.
        assert!(
            total_frames > 40000 && total_frames < 56000,
            "Unexpected output length: {total_frames}"
        );
        // All input was silence, so output should be (approximately) silence.
        for &s in output.iter().chain(tail.iter()) {
            assert!(s.abs() < 1e-6, "Non-silent sample in silence resample: {s}");
        }
    }

    #[test]
    fn process_non_multiple_of_chunk() {
        // Feed an odd number of frames that doesn't align with chunk_size.
        let mut r = Resampler::new(48000, 16000, 2, Quality::Fast).unwrap();
        let frames = 3000; // not a multiple of 1024
        let input = vec![0.0f32; frames * 2];
        let output = r.process(&input).unwrap();
        let tail = r.flush().unwrap();
        let total_frames = (output.len() + tail.len()) / 2;
        let expected = r.output_frames_estimate(frames);
        // Allow 20% tolerance.
        assert!(
            (total_frames as f64 - expected as f64).abs() / expected as f64 <= 0.2,
            "Output frame count {total_frames} too far from estimate {expected}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut r = Resampler::new(44100, 48000, 1, Quality::Fast).unwrap();
        let _ = r.process(&vec![0.5f32; 500]).unwrap();
        r.reset();
        // After reset, pending should be empty.
        let tail = r.flush().unwrap();
        assert!(tail.is_empty());
    }

    #[test]
    fn channel_mismatch_rejected() {
        let mut r = Resampler::new(44100, 48000, 2, Quality::Fast).unwrap();
        // 3 samples is not a multiple of 2 channels.
        let result = r.process(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }
}
