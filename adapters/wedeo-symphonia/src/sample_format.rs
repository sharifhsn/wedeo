use std::sync::Arc;

use symphonia::core::audio::{AudioBufferRef, Signal};
use wedeo_core::{
    AudioBlock, BufferPool, ChannelLayout, Rational, Result, SampleFormat, Timestamp,
};

/// Convert a symphonia `AudioBufferRef` to a wedeo `AudioBlock`.
///
/// Symphonia decodes into planar typed buffers. We interleave them into a
/// single packed buffer matching FFmpeg's default packed output format.
pub fn audio_buffer_to_block(
    buf: &AudioBufferRef<'_>,
    sample_rate: u32,
    channel_layout: ChannelLayout,
    pts: Timestamp,
    pkt_duration: i64,
    time_base: Rational,
    pool: &Arc<BufferPool>,
) -> Result<AudioBlock> {
    let spec_channels = buf.spec().channels.count();
    let nb_channels = (channel_layout.nb_channels as usize)
        .max(spec_channels)
        .max(1);
    let avail_channels = spec_channels.max(1);
    let nb_samples = buf.frames() as u32;

    let (sample_format, data) = match buf {
        AudioBufferRef::U8(b) => {
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.push(b.chan(ch.min(avail_channels - 1))[s]);
                }
            }
            (SampleFormat::U8, out)
        }
        AudioBufferRef::U16(b) => {
            // U16 -> S16 (subtract 0x8000) to match FFmpeg behavior
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 2);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    let v = b.chan(ch.min(avail_channels - 1))[s].wrapping_sub(0x8000) as i16;
                    out.extend_from_slice(&v.to_ne_bytes());
                }
            }
            (SampleFormat::S16, out)
        }
        AudioBufferRef::U24(b) => {
            // U24 -> S32: subtract 0x800000, shift left 8
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 4);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    let v = b.chan(ch.min(avail_channels - 1))[s].0;
                    let sample = v.wrapping_sub(0x800000) << 8;
                    out.extend_from_slice(&sample.to_ne_bytes());
                }
            }
            (SampleFormat::S32, out)
        }
        AudioBufferRef::U32(b) => {
            // U32 -> S32: subtract 0x80000000
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 4);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    let v = b.chan(ch.min(avail_channels - 1))[s].wrapping_sub(0x8000_0000) as i32;
                    out.extend_from_slice(&v.to_ne_bytes());
                }
            }
            (SampleFormat::S32, out)
        }
        AudioBufferRef::S8(b) => {
            // S8 stored as U8 with 128 offset in wedeo (FFmpeg u8 is unsigned 8-bit)
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    let v = (b.chan(ch.min(avail_channels - 1))[s] as i16 + 128) as u8;
                    out.push(v);
                }
            }
            (SampleFormat::U8, out)
        }
        AudioBufferRef::S16(b) => {
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 2);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.extend_from_slice(&b.chan(ch.min(avail_channels - 1))[s].to_ne_bytes());
                }
            }
            (SampleFormat::S16, out)
        }
        AudioBufferRef::S24(b) => {
            // S24 -> S32 with 8-bit left shift, matching FFmpeg's behavior
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 4);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    let v = b.chan(ch.min(avail_channels - 1))[s].0;
                    let sample = (v as u32) << 8;
                    out.extend_from_slice(&sample.to_ne_bytes());
                }
            }
            (SampleFormat::S32, out)
        }
        AudioBufferRef::S32(b) => {
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 4);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.extend_from_slice(&b.chan(ch.min(avail_channels - 1))[s].to_ne_bytes());
                }
            }
            (SampleFormat::S32, out)
        }
        AudioBufferRef::F32(b) => {
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 4);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.extend_from_slice(&b.chan(ch.min(avail_channels - 1))[s].to_ne_bytes());
                }
            }
            (SampleFormat::F32, out)
        }
        AudioBufferRef::F64(b) => {
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 8);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.extend_from_slice(&b.chan(ch.min(avail_channels - 1))[s].to_ne_bytes());
                }
            }
            (SampleFormat::F64, out)
        }
    };

    let plane = pool.acquire_from_slice(&data);
    let mut block = AudioBlock::new(
        vec![plane],
        nb_samples,
        sample_format,
        sample_rate,
        channel_layout,
    );
    block.pts = pts;
    block.duration = if pkt_duration > 0 {
        pkt_duration
    } else {
        nb_samples as i64
    };
    block.time_base = if time_base.is_valid() {
        time_base
    } else {
        Rational::new(1, sample_rate.max(1) as i32)
    };

    Ok(block)
}
