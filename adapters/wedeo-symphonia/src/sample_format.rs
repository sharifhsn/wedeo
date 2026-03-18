use symphonia::core::audio::{AudioBufferRef, Signal};
use wedeo_core::buffer::Buffer;
use wedeo_core::channel_layout::ChannelLayout;
use wedeo_core::error::Result;
use wedeo_core::frame::{Frame, FrameData, FramePlane};
use wedeo_core::sample_format::SampleFormat;

/// Convert a symphonia AudioBufferRef to a wedeo Frame.
///
/// Symphonia decodes into planar typed buffers. We interleave them into a
/// single packed buffer matching FFmpeg's output format.
pub fn audio_buffer_to_frame(
    buf: &AudioBufferRef<'_>,
    sample_rate: u32,
    channel_layout: ChannelLayout,
    pts: i64,
) -> Result<Frame> {
    let nb_channels = channel_layout.nb_channels.max(1) as usize;
    let nb_samples = buf.frames() as u32;

    let (sample_format, data) = match buf {
        AudioBufferRef::U8(b) => {
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.push(b.chan(ch)[s]);
                }
            }
            (SampleFormat::U8, out)
        }
        AudioBufferRef::U16(b) => {
            // U16 → S16 (subtract 0x8000) to match FFmpeg behavior
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 2);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    let v = b.chan(ch)[s].wrapping_sub(0x8000) as i16;
                    out.extend_from_slice(&v.to_ne_bytes());
                }
            }
            (SampleFormat::S16, out)
        }
        AudioBufferRef::U24(b) => {
            // U24 → S32: subtract 0x800000, shift left 8
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 4);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    let v = b.chan(ch)[s].0;
                    let sample = v.wrapping_sub(0x800000) << 8;
                    out.extend_from_slice(&sample.to_ne_bytes());
                }
            }
            (SampleFormat::S32, out)
        }
        AudioBufferRef::U32(b) => {
            // U32 → S32: subtract 0x80000000
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 4);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    let v = b.chan(ch)[s].wrapping_sub(0x80000000) as i32;
                    out.extend_from_slice(&v.to_ne_bytes());
                }
            }
            (SampleFormat::S32, out)
        }
        AudioBufferRef::S8(b) => {
            // S8 stored as U8 with 128 offset in wedeo
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.push(b.chan(ch)[s] as u8);
                }
            }
            (SampleFormat::U8, out)
        }
        AudioBufferRef::S16(b) => {
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 2);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.extend_from_slice(&b.chan(ch)[s].to_ne_bytes());
                }
            }
            (SampleFormat::S16, out)
        }
        AudioBufferRef::S24(b) => {
            // S24 → S32 with 8-bit left shift, matching FFmpeg's behavior
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 4);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    let v = b.chan(ch)[s].0;
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
                    out.extend_from_slice(&b.chan(ch)[s].to_ne_bytes());
                }
            }
            (SampleFormat::S32, out)
        }
        AudioBufferRef::F32(b) => {
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 4);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.extend_from_slice(&b.chan(ch)[s].to_ne_bytes());
                }
            }
            (SampleFormat::Flt, out)
        }
        AudioBufferRef::F64(b) => {
            let mut out = Vec::with_capacity(nb_samples as usize * nb_channels * 8);
            for s in 0..nb_samples as usize {
                for ch in 0..nb_channels {
                    out.extend_from_slice(&b.chan(ch)[s].to_ne_bytes());
                }
            }
            (SampleFormat::Dbl, out)
        }
    };

    let buffer = Buffer::from_slice(&data);
    let plane = FramePlane {
        buffer,
        offset: 0,
        linesize: data.len(),
    };

    let mut frame = Frame::new_audio(nb_samples, sample_format, sample_rate, channel_layout);
    frame.pts = pts;
    frame.duration = nb_samples as i64;

    if let FrameData::Audio(ref mut audio) = frame.data {
        audio.planes = vec![plane];
    }

    Ok(frame)
}
