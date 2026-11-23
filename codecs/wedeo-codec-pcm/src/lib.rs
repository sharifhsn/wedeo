//! Native bit-exact FFmpeg `n9.1` PCM audio decoders and encoders (`libavcodec/pcm.c`).
//!
//! Supports all 20 PCM codec IDs in `wedeo_core::CodecId`:
//! - Signed integer: `PcmS8`, `PcmS16le`, `PcmS16be`, `PcmS24le`, `PcmS24be`, `PcmS32le`, `PcmS32be`
//! - Unsigned integer: `PcmU8`, `PcmU16le`, `PcmU16be`, `PcmU24le`, `PcmU24be`, `PcmU32le`, `PcmU32be`
//! - IEEE 754 floating-point: `PcmF32le`, `PcmF32be`, `PcmF64le`, `PcmF64be`
//! - ITU-T G.711 companded: `PcmAlaw`, `PcmMulaw` (exact FFmpeg `pcm_tablegen.h` lookup tables)

#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::sync::Arc;

use wedeo_core::{
    AudioBlock, ChannelLayout, CodecContext, CodecId, Decoder, DecoderDescriptor, DecoderFactory,
    EncodeHint, Encoder, EncoderDescriptor, EncoderFactory, Error, ErrorPolicy, HealthEvent,
    HealthIssue, MediaType, MediaUnit, Packet, PacketFlags, PassMode, Rational, RecvStatus,
    Registry, Result, SampleFormat, SendStatus, StreamHealth, TrackDescriptor, TrackId, rescale_q,
};

// =============================================================================
// G.711 A-law and mu-law exact FFmpeg n9.1 tables (libavcodec/pcm_tablegen.h)
// =============================================================================

const SIGN_BIT: u8 = 0x80;
const QUANT_MASK: u8 = 0x0f;
const SEG_SHIFT: u8 = 4;
const SEG_MASK: u8 = 0x70;
const BIAS: i32 = 0x84;

/// Convert an 8-bit A-law codeword to 16-bit linear PCM (`libavcodec/pcm_tablegen.h:alaw2linear`).
#[inline]
pub const fn alaw2linear(mut a_val: u8) -> i16 {
    a_val ^= 0x55;

    let mut t = (a_val & QUANT_MASK) as i32;
    let seg = ((a_val & SEG_MASK) >> SEG_SHIFT) as i32;
    if seg != 0 {
        t = (t + t + 1 + 32) << (seg + 2);
    } else {
        t = (t + t + 1) << 3;
    }

    if (a_val & SIGN_BIT) != 0 {
        t as i16
    } else {
        (-t) as i16
    }
}

/// Convert an 8-bit mu-law codeword to 16-bit linear PCM (`libavcodec/pcm_tablegen.h:ulaw2linear`).
#[inline]
pub const fn ulaw2linear(mut u_val: u8) -> i16 {
    u_val = !u_val;

    let mut t = (((u_val & QUANT_MASK) as i32) << 3) + BIAS;
    t <<= ((u_val & SEG_MASK) >> SEG_SHIFT) as i32;

    if (u_val & SIGN_BIT) != 0 {
        (BIAS - t) as i16
    } else {
        (t - BIAS) as i16
    }
}

const fn build_alaw_decode_lut() -> [i16; 256] {
    let mut table = [0i16; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = alaw2linear(i as u8);
        i += 1;
    }
    table
}

const fn build_ulaw_decode_lut() -> [i16; 256] {
    let mut table = [0i16; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = ulaw2linear(i as u8);
        i += 1;
    }
    table
}

/// Build the 16384-entry linear-to-xlaw encode lookup table (`libavcodec/pcm_tablegen.h:build_xlaw_table`).
const fn build_alaw_encode_lut() -> [u8; 16384] {
    let mask: i32 = 0xd5;
    let mut linear_to_xlaw = [0u8; 16384];
    let mut j: i32 = 1;
    linear_to_xlaw[8192] = mask as u8;
    let mut i: i32 = 0;
    while i < 127 {
        let v1 = alaw2linear((i ^ mask) as u8) as i32;
        let v2 = alaw2linear(((i + 1) ^ mask) as u8) as i32;
        let v = (v1 + v2 + 4) >> 3;
        while j < v {
            linear_to_xlaw[(8192 - j) as usize] = (i ^ (mask ^ 0x80)) as u8;
            linear_to_xlaw[(8192 + j) as usize] = (i ^ mask) as u8;
            j += 1;
        }
        i += 1;
    }
    while j < 8192 {
        linear_to_xlaw[(8192 - j) as usize] = (127 ^ (mask ^ 0x80)) as u8;
        linear_to_xlaw[(8192 + j) as usize] = (127 ^ mask) as u8;
        j += 1;
    }
    linear_to_xlaw[0] = linear_to_xlaw[1];
    linear_to_xlaw
}

/// Build the 16384-entry linear-to-ulaw encode lookup table (`libavcodec/pcm_tablegen.h:build_xlaw_table`).
const fn build_ulaw_encode_lut() -> [u8; 16384] {
    let mask: i32 = 0xff;
    let mut linear_to_xlaw = [0u8; 16384];
    let mut j: i32 = 1;
    linear_to_xlaw[8192] = mask as u8;
    let mut i: i32 = 0;
    while i < 127 {
        let v1 = ulaw2linear((i ^ mask) as u8) as i32;
        let v2 = ulaw2linear(((i + 1) ^ mask) as u8) as i32;
        let v = (v1 + v2 + 4) >> 3;
        while j < v {
            linear_to_xlaw[(8192 - j) as usize] = (i ^ (mask ^ 0x80)) as u8;
            linear_to_xlaw[(8192 + j) as usize] = (i ^ mask) as u8;
            j += 1;
        }
        i += 1;
    }
    while j < 8192 {
        linear_to_xlaw[(8192 - j) as usize] = (127 ^ (mask ^ 0x80)) as u8;
        linear_to_xlaw[(8192 + j) as usize] = (127 ^ mask) as u8;
        j += 1;
    }
    linear_to_xlaw[0] = linear_to_xlaw[1];
    linear_to_xlaw
}

static ALAW_TO_LINEAR: [i16; 256] = build_alaw_decode_lut();
static ULAW_TO_LINEAR: [i16; 256] = build_ulaw_decode_lut();
static LINEAR_TO_ALAW: [u8; 16384] = build_alaw_encode_lut();
static LINEAR_TO_ULAW: [u8; 16384] = build_ulaw_encode_lut();

/// Encode a 16-bit linear PCM sample to A-law (`libavcodec/pcm.c:233`).
#[inline]
pub fn linear2alaw(sample: i16) -> u8 {
    let idx = (((sample as i32) + 32768) >> 2) as usize;
    LINEAR_TO_ALAW[idx]
}

/// Encode a 16-bit linear PCM sample to mu-law (`libavcodec/pcm.c:241`).
#[inline]
pub fn linear2ulaw(sample: i16) -> u8 {
    let idx = (((sample as i32) + 32768) >> 2) as usize;
    LINEAR_TO_ULAW[idx]
}

// =============================================================================
// PCM format tables & helpers
// =============================================================================

/// Map a PCM `CodecId` to FFmpeg's decoded `SampleFormat` (`libavcodec/pcm.c:pcm_decode_init`).
#[inline]
pub const fn codec_id_to_sample_format(id: CodecId) -> Option<SampleFormat> {
    match id {
        CodecId::PcmS8 | CodecId::PcmU8 => Some(SampleFormat::U8),
        CodecId::PcmS16le
        | CodecId::PcmS16be
        | CodecId::PcmU16le
        | CodecId::PcmU16be
        | CodecId::PcmAlaw
        | CodecId::PcmMulaw => Some(SampleFormat::S16),
        CodecId::PcmS24le
        | CodecId::PcmS24be
        | CodecId::PcmU24le
        | CodecId::PcmU24be
        | CodecId::PcmS32le
        | CodecId::PcmS32be
        | CodecId::PcmU32le
        | CodecId::PcmU32be => Some(SampleFormat::S32),
        CodecId::PcmF32le | CodecId::PcmF32be => Some(SampleFormat::F32),
        CodecId::PcmF64le | CodecId::PcmF64be => Some(SampleFormat::F64),
        _ => None,
    }
}

/// Coded byte size of a single channel sample in the compressed PCM bitstream.
#[inline]
pub const fn coded_bytes_per_sample(id: CodecId) -> Option<usize> {
    match id {
        CodecId::PcmS8 | CodecId::PcmU8 | CodecId::PcmAlaw | CodecId::PcmMulaw => Some(1),
        CodecId::PcmS16le | CodecId::PcmS16be | CodecId::PcmU16le | CodecId::PcmU16be => Some(2),
        CodecId::PcmS24le | CodecId::PcmS24be | CodecId::PcmU24le | CodecId::PcmU24be => Some(3),
        CodecId::PcmS32le
        | CodecId::PcmS32be
        | CodecId::PcmU32le
        | CodecId::PcmU32be
        | CodecId::PcmF32le
        | CodecId::PcmF32be => Some(4),
        CodecId::PcmF64le | CodecId::PcmF64be => Some(8),
        _ => None,
    }
}

/// Exact bits per coded sample for a PCM `CodecId` (`av_get_bits_per_sample`).
#[inline]
pub const fn bits_per_coded_sample(id: CodecId) -> Option<u32> {
    match coded_bytes_per_sample(id) {
        Some(bytes) => Some((bytes as u32) * 8),
        None => None,
    }
}

/// Decode `n_samples` (total samples across all channels) from `src` into `dst`.
///
/// `src` must have length `n_samples * coded_bytes_per_sample(codec_id)` and
/// `dst` must have length `n_samples * codec_id_to_sample_format(codec_id).bytes_per_sample()`.
fn decode_samples_into(codec_id: CodecId, src: &[u8], dst: &mut [u8]) {
    match codec_id {
        CodecId::PcmU8 => {
            dst.copy_from_slice(src);
        }
        CodecId::PcmS8 => {
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.wrapping_add(128);
            }
        }
        CodecId::PcmAlaw => {
            for (&s, d) in src.iter().zip(dst.as_chunks_mut::<2>().0.iter_mut()) {
                *d = ALAW_TO_LINEAR[s as usize].to_ne_bytes();
            }
        }
        CodecId::PcmMulaw => {
            for (&s, d) in src.iter().zip(dst.as_chunks_mut::<2>().0.iter_mut()) {
                *d = ULAW_TO_LINEAR[s as usize].to_ne_bytes();
            }
        }
        CodecId::PcmS16le => {
            if cfg!(target_endian = "little") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<2>().0.iter_mut())
                {
                    *d = i16::from_le_bytes(s).to_ne_bytes();
                }
            }
        }
        CodecId::PcmS16be => {
            if cfg!(target_endian = "big") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<2>().0.iter_mut())
                {
                    *d = i16::from_be_bytes(s).to_ne_bytes();
                }
            }
        }
        CodecId::PcmU16le => {
            for (&s, d) in src
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<2>().0.iter_mut())
            {
                let v = u16::from_le_bytes(s).wrapping_sub(0x8000);
                *d = v.to_ne_bytes();
            }
        }
        CodecId::PcmU16be => {
            for (&s, d) in src
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<2>().0.iter_mut())
            {
                let v = u16::from_be_bytes(s).wrapping_sub(0x8000);
                *d = v.to_ne_bytes();
            }
        }
        CodecId::PcmS24le => {
            for (&s, d) in src
                .as_chunks::<3>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<4>().0.iter_mut())
            {
                let v = (s[0] as u32) | ((s[1] as u32) << 8) | ((s[2] as u32) << 16);
                *d = (v << 8).to_ne_bytes();
            }
        }
        CodecId::PcmS24be => {
            for (&s, d) in src
                .as_chunks::<3>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<4>().0.iter_mut())
            {
                let v = ((s[0] as u32) << 16) | ((s[1] as u32) << 8) | (s[2] as u32);
                *d = (v << 8).to_ne_bytes();
            }
        }
        CodecId::PcmU24le => {
            for (&s, d) in src
                .as_chunks::<3>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<4>().0.iter_mut())
            {
                let v = (s[0] as u32) | ((s[1] as u32) << 8) | ((s[2] as u32) << 16);
                let out = v.wrapping_sub(0x80_0000) << 8;
                *d = out.to_ne_bytes();
            }
        }
        CodecId::PcmU24be => {
            for (&s, d) in src
                .as_chunks::<3>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<4>().0.iter_mut())
            {
                let v = ((s[0] as u32) << 16) | ((s[1] as u32) << 8) | (s[2] as u32);
                let out = v.wrapping_sub(0x80_0000) << 8;
                *d = out.to_ne_bytes();
            }
        }
        CodecId::PcmS32le | CodecId::PcmF32le => {
            if cfg!(target_endian = "little") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<4>().0.iter_mut())
                {
                    *d = u32::from_le_bytes(s).to_ne_bytes();
                }
            }
        }
        CodecId::PcmS32be | CodecId::PcmF32be => {
            if cfg!(target_endian = "big") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<4>().0.iter_mut())
                {
                    *d = u32::from_be_bytes(s).to_ne_bytes();
                }
            }
        }
        CodecId::PcmU32le => {
            for (&s, d) in src
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<4>().0.iter_mut())
            {
                let v = u32::from_le_bytes(s).wrapping_sub(0x8000_0000);
                *d = v.to_ne_bytes();
            }
        }
        CodecId::PcmU32be => {
            for (&s, d) in src
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<4>().0.iter_mut())
            {
                let v = u32::from_be_bytes(s).wrapping_sub(0x8000_0000);
                *d = v.to_ne_bytes();
            }
        }
        CodecId::PcmF64le => {
            if cfg!(target_endian = "little") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<8>().0.iter_mut())
                {
                    *d = u64::from_le_bytes(s).to_ne_bytes();
                }
            }
        }
        CodecId::PcmF64be => {
            if cfg!(target_endian = "big") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<8>().0.iter_mut())
                {
                    *d = u64::from_be_bytes(s).to_ne_bytes();
                }
            }
        }
        _ => {}
    }
}

/// Encode `n_samples` (interleaved native samples in `src`) into `dst` (`libavcodec/pcm.c:pcm_encode_frame`).
fn encode_samples_into(codec_id: CodecId, src: &[u8], dst: &mut [u8]) {
    match codec_id {
        CodecId::PcmU8 => {
            dst.copy_from_slice(src);
        }
        CodecId::PcmS8 => {
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = s.wrapping_sub(128);
            }
        }
        CodecId::PcmAlaw => {
            for (&s, d) in src.as_chunks::<2>().0.iter().zip(dst.iter_mut()) {
                let sample = i16::from_ne_bytes(s);
                *d = linear2alaw(sample);
            }
        }
        CodecId::PcmMulaw => {
            for (&s, d) in src.as_chunks::<2>().0.iter().zip(dst.iter_mut()) {
                let sample = i16::from_ne_bytes(s);
                *d = linear2ulaw(sample);
            }
        }
        CodecId::PcmS16le => {
            if cfg!(target_endian = "little") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<2>().0.iter_mut())
                {
                    *d = i16::from_ne_bytes(s).to_le_bytes();
                }
            }
        }
        CodecId::PcmS16be => {
            if cfg!(target_endian = "big") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<2>().0.iter_mut())
                {
                    *d = i16::from_ne_bytes(s).to_be_bytes();
                }
            }
        }
        CodecId::PcmU16le => {
            for (&s, d) in src
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<2>().0.iter_mut())
            {
                let v = u16::from_ne_bytes(s).wrapping_add(0x8000);
                *d = v.to_le_bytes();
            }
        }
        CodecId::PcmU16be => {
            for (&s, d) in src
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<2>().0.iter_mut())
            {
                let v = u16::from_ne_bytes(s).wrapping_add(0x8000);
                *d = v.to_be_bytes();
            }
        }
        CodecId::PcmS24le => {
            for (&s, d) in src
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<3>().0.iter_mut())
            {
                let sample = i32::from_ne_bytes(s);
                let v = (sample >> 8) as u32;
                *d = [v as u8, (v >> 8) as u8, (v >> 16) as u8];
            }
        }
        CodecId::PcmS24be => {
            for (&s, d) in src
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<3>().0.iter_mut())
            {
                let sample = i32::from_ne_bytes(s);
                let v = (sample >> 8) as u32;
                *d = [(v >> 16) as u8, (v >> 8) as u8, v as u8];
            }
        }
        CodecId::PcmU24le => {
            for (&s, d) in src
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<3>().0.iter_mut())
            {
                let sample = u32::from_ne_bytes(s);
                let v = (sample >> 8).wrapping_add(0x80_0000);
                *d = [v as u8, (v >> 8) as u8, (v >> 16) as u8];
            }
        }
        CodecId::PcmU24be => {
            for (&s, d) in src
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<3>().0.iter_mut())
            {
                let sample = u32::from_ne_bytes(s);
                let v = (sample >> 8).wrapping_add(0x80_0000);
                *d = [(v >> 16) as u8, (v >> 8) as u8, v as u8];
            }
        }
        CodecId::PcmS32le | CodecId::PcmF32le => {
            if cfg!(target_endian = "little") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<4>().0.iter_mut())
                {
                    *d = u32::from_ne_bytes(s).to_le_bytes();
                }
            }
        }
        CodecId::PcmS32be | CodecId::PcmF32be => {
            if cfg!(target_endian = "big") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<4>().0.iter_mut())
                {
                    *d = u32::from_ne_bytes(s).to_be_bytes();
                }
            }
        }
        CodecId::PcmU32le => {
            for (&s, d) in src
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<4>().0.iter_mut())
            {
                let v = u32::from_ne_bytes(s).wrapping_add(0x8000_0000);
                *d = v.to_le_bytes();
            }
        }
        CodecId::PcmU32be => {
            for (&s, d) in src
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst.as_chunks_mut::<4>().0.iter_mut())
            {
                let v = u32::from_ne_bytes(s).wrapping_add(0x8000_0000);
                *d = v.to_be_bytes();
            }
        }
        CodecId::PcmF64le => {
            if cfg!(target_endian = "little") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<8>().0.iter_mut())
                {
                    *d = u64::from_ne_bytes(s).to_le_bytes();
                }
            }
        }
        CodecId::PcmF64be => {
            if cfg!(target_endian = "big") {
                dst.copy_from_slice(src);
            } else {
                for (&s, d) in src
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(dst.as_chunks_mut::<8>().0.iter_mut())
                {
                    *d = u64::from_ne_bytes(s).to_be_bytes();
                }
            }
        }
        _ => {}
    }
}

// =============================================================================
// PCM Decoder (v2 wedeo_core::Decoder)
// =============================================================================

/// Native FFmpeg `n9.1` PCM decoder.
pub struct PcmDecoder {
    codec_id: CodecId,
    sample_format: SampleFormat,
    coded_bps: usize,
    sample_rate: u32,
    channel_layout: ChannelLayout,
    time_base: Rational,
    ctx: CodecContext,
    pending: VecDeque<MediaUnit>,
    draining: bool,
    health: StreamHealth,
}

impl PcmDecoder {
    pub fn open(
        expected_codec: CodecId,
        track: &TrackDescriptor,
        ctx: CodecContext,
    ) -> Result<Self> {
        if track.codec_id != expected_codec {
            return Err(Error::invalid_argument(
                "pcm_dec",
                format!(
                    "codec id mismatch: decoder is {}, track is {}",
                    expected_codec.name(),
                    track.codec_id.name()
                ),
            ));
        }

        let sample_format = codec_id_to_sample_format(expected_codec)
            .ok_or_else(|| Error::unsupported("pcm_dec", "unsupported PCM codec id"))?;
        let coded_bps = coded_bytes_per_sample(expected_codec)
            .ok_or_else(|| Error::unsupported("pcm_dec", "unsupported PCM codec id"))?;

        let audio = track
            .params
            .audio()
            .ok_or_else(|| Error::invalid_argument("pcm_dec", "track is not an audio stream"))?;

        if audio.channel_layout.nb_channels == 0 {
            return Err(Error::invalid_argument(
                "pcm_dec",
                "invalid number of channels (0)",
            ));
        }
        if audio.sample_rate == 0 {
            return Err(Error::invalid_argument(
                "pcm_dec",
                "invalid sample rate (0)",
            ));
        }

        let time_base = if track.time_base.is_valid() {
            track.time_base
        } else {
            Rational::new(1, audio.sample_rate as i32)
        };

        Ok(Self {
            codec_id: expected_codec,
            sample_format,
            coded_bps,
            sample_rate: audio.sample_rate,
            channel_layout: audio.channel_layout.clone(),
            time_base,
            ctx,
            pending: VecDeque::new(),
            draining: false,
            health: StreamHealth::new(),
        })
    }

    fn decode_packet(&mut self, pkt: &Packet) -> Result<AudioBlock> {
        let channels = self.channel_layout.nb_channels as usize;
        let block_bytes = channels * self.coded_bps;
        let raw = pkt.data.data();
        let mut buf_size = raw.len();

        if block_bytes > 0 && !buf_size.is_multiple_of(block_bytes) {
            self.health.record(HealthEvent {
                track: pkt.track_id,
                pts: pkt.pts,
                byte_offset: pkt.pos,
                kind: HealthIssue::TruncatedPacket,
                detail: "unaligned PCM packet size",
            });
            if self.ctx.error_policy == ErrorPolicy::Strict {
                return Err(Error::corrupt(
                    "pcm_dec",
                    "PCM packet size is not a multiple of channel block size",
                ));
            }
            if buf_size < block_bytes {
                return Err(Error::invalid_data(
                    "pcm_dec",
                    format!(
                        "invalid PCM packet: size {buf_size} is smaller than block size {block_bytes}"
                    ),
                ));
            }
            buf_size -= buf_size % block_bytes;
        } else if buf_size < block_bytes {
            self.health.record(HealthEvent {
                track: pkt.track_id,
                pts: pkt.pts,
                byte_offset: pkt.pos,
                kind: HealthIssue::TruncatedPacket,
                detail: "empty PCM packet",
            });
            if self.ctx.error_policy == ErrorPolicy::Strict {
                return Err(Error::corrupt("pcm_dec", "empty PCM packet"));
            }
            return Err(Error::invalid_data("pcm_dec", "empty PCM packet"));
        }

        let total_samples = buf_size / self.coded_bps;
        let nb_samples = (total_samples / channels) as u32;
        let decoded_bps = self.sample_format.bytes_per_sample();
        let out_bytes = total_samples * decoded_bps;

        let mut out_buf = self.ctx.pool.acquire_reused(out_bytes);
        let dst = out_buf
            .try_data_mut()
            .ok_or_else(|| Error::internal("pcm_dec", "pooled buffer not uniquely writable"))?;

        decode_samples_into(self.codec_id, &raw[..buf_size], dst);

        let time_base = if pkt.time_base.is_valid() {
            pkt.time_base
        } else {
            self.time_base
        };
        let pts = if pkt.pts.is_known() { pkt.pts } else { pkt.dts };
        let duration = if pkt.duration > 0 {
            pkt.duration
        } else {
            rescale_q(
                nb_samples as i64,
                Rational::new(1, self.sample_rate as i32),
                time_base,
            )
        };

        let mut block = AudioBlock::new(
            vec![out_buf],
            nb_samples,
            self.sample_format,
            self.sample_rate,
            self.channel_layout.clone(),
        );
        block.pts = pts;
        block.duration = duration;
        block.time_base = time_base;
        block.side_data.clone_from(&pkt.side_data);

        Ok(block)
    }
}

impl Decoder for PcmDecoder {
    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<SendStatus> {
        match pkt {
            Some(p) => {
                if self.draining {
                    return Err(Error::invalid_argument(
                        "pcm_dec",
                        "decoder is draining; call flush() before sending new packets",
                    ));
                }
                let block = self.decode_packet(p)?;
                self.pending.push_back(MediaUnit::Audio(block));
                Ok(SendStatus::Accepted)
            }
            None => {
                self.draining = true;
                Ok(SendStatus::Accepted)
            }
        }
    }

    fn receive(&mut self) -> Result<RecvStatus<MediaUnit>> {
        if let Some(unit) = self.pending.pop_front() {
            Ok(RecvStatus::Ready(unit))
        } else if self.draining {
            Ok(RecvStatus::Eof)
        } else {
            Ok(RecvStatus::NeedMore)
        }
    }

    fn flush(&mut self) {
        self.pending.clear();
        self.draining = false;
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

// =============================================================================
// PCM Encoder (v2 wedeo_core::Encoder)
// =============================================================================

/// Native FFmpeg `n9.1` PCM encoder.
pub struct PcmEncoder {
    codec_id: CodecId,
    track_id: TrackId,
    expected_format: SampleFormat,
    coded_bps: usize,
    sample_rate: u32,
    channel_layout: ChannelLayout,
    time_base: Rational,
    ctx: CodecContext,
    pending: VecDeque<Packet>,
    draining: bool,
    interleave_scratch: Vec<u8>,
}

impl PcmEncoder {
    pub fn open(
        expected_codec: CodecId,
        track: &TrackDescriptor,
        ctx: CodecContext,
    ) -> Result<Self> {
        if track.codec_id != expected_codec {
            return Err(Error::invalid_argument(
                "pcm_enc",
                format!(
                    "codec id mismatch: encoder is {}, track is {}",
                    expected_codec.name(),
                    track.codec_id.name()
                ),
            ));
        }

        let expected_format = codec_id_to_sample_format(expected_codec)
            .ok_or_else(|| Error::unsupported("pcm_enc", "unsupported PCM codec id"))?;
        let coded_bps = coded_bytes_per_sample(expected_codec)
            .ok_or_else(|| Error::unsupported("pcm_enc", "unsupported PCM codec id"))?;

        let audio = track
            .params
            .audio()
            .ok_or_else(|| Error::invalid_argument("pcm_enc", "track is not an audio stream"))?;

        if audio.channel_layout.nb_channels == 0 {
            return Err(Error::invalid_argument(
                "pcm_enc",
                "invalid number of channels (0)",
            ));
        }
        if audio.sample_rate == 0 {
            return Err(Error::invalid_argument(
                "pcm_enc",
                "invalid sample rate (0)",
            ));
        }

        let time_base = if track.time_base.is_valid() {
            track.time_base
        } else {
            Rational::new(1, audio.sample_rate as i32)
        };

        Ok(Self {
            codec_id: expected_codec,
            track_id: track.id,
            expected_format,
            coded_bps,
            sample_rate: audio.sample_rate,
            channel_layout: audio.channel_layout.clone(),
            time_base,
            ctx,
            pending: VecDeque::new(),
            draining: false,
            interleave_scratch: Vec::new(),
        })
    }

    fn encode_block(&mut self, block: &AudioBlock) -> Result<Packet> {
        if block.format.packed() != self.expected_format {
            return Err(Error::invalid_argument(
                "pcm_enc",
                format!(
                    "PCM encoder {} expects sample format {}, got {}",
                    self.codec_id.name(),
                    self.expected_format,
                    block.format
                ),
            ));
        }

        let channels = self.channel_layout.nb_channels as usize;
        if block.channel_layout.nb_channels as usize != channels {
            return Err(Error::invalid_argument(
                "pcm_enc",
                format!(
                    "channel count mismatch: encoder configured for {}, block has {}",
                    channels, block.channel_layout.nb_channels
                ),
            ));
        }

        let nb_samples = block.nb_samples as usize;
        let total_samples = nb_samples * channels;
        let in_bps = self.expected_format.bytes_per_sample();
        let expected_in_bytes = total_samples * in_bps;
        let out_bytes = total_samples * self.coded_bps;

        let mut out_buf = self.ctx.pool.acquire_reused(out_bytes);
        let dst = out_buf
            .try_data_mut()
            .ok_or_else(|| Error::internal("pcm_enc", "pooled buffer not uniquely writable"))?;

        if block.format.is_planar() {
            if block.planes.len() < channels {
                return Err(Error::invalid_data(
                    "pcm_enc",
                    "planar AudioBlock has fewer planes than channels",
                ));
            }
            let plane_bytes = nb_samples * in_bps;
            for p in &block.planes[..channels] {
                if p.len() < plane_bytes {
                    return Err(Error::invalid_data(
                        "pcm_enc",
                        "planar AudioBlock plane buffer is too small for nb_samples",
                    ));
                }
            }
            self.interleave_scratch.resize(expected_in_bytes, 0);
            for s in 0..nb_samples {
                let src_off = s * in_bps;
                for (ch, plane) in block.planes.iter().enumerate().take(channels) {
                    let dst_off = (s * channels + ch) * in_bps;
                    self.interleave_scratch[dst_off..dst_off + in_bps]
                        .copy_from_slice(&plane.data()[src_off..src_off + in_bps]);
                }
            }
            encode_samples_into(self.codec_id, &self.interleave_scratch, dst);
        } else {
            if block.planes.is_empty() {
                return Err(Error::invalid_data("pcm_enc", "AudioBlock has no planes"));
            }
            let src = block.planes[0].data();
            if src.len() < expected_in_bytes {
                return Err(Error::invalid_data(
                    "pcm_enc",
                    format!(
                        "AudioBlock plane too small: got {} bytes, need {}",
                        src.len(),
                        expected_in_bytes
                    ),
                ));
            }
            encode_samples_into(self.codec_id, &src[..expected_in_bytes], dst);
        }

        let time_base = if block.time_base.is_valid() {
            block.time_base
        } else {
            self.time_base
        };
        let duration = if block.duration > 0 {
            block.duration
        } else {
            let sr = if block.sample_rate > 0 {
                block.sample_rate
            } else {
                self.sample_rate
            };
            rescale_q(
                block.nb_samples as i64,
                Rational::new(1, sr as i32),
                time_base,
            )
        };

        let mut pkt = Packet::new(out_buf, self.track_id, time_base);
        pkt.pts = block.pts;
        pkt.dts = block.pts;
        pkt.duration = duration;
        pkt.flags = PacketFlags::KEY;
        pkt.side_data.clone_from(&block.side_data);

        Ok(pkt)
    }
}

impl Encoder for PcmEncoder {
    fn extradata(&self) -> &[u8] {
        &[]
    }

    fn send_unit(&mut self, unit: Option<&MediaUnit>, _hint: EncodeHint) -> Result<SendStatus> {
        match unit {
            Some(u) => {
                if self.draining {
                    return Err(Error::invalid_argument(
                        "pcm_enc",
                        "encoder is draining; call flush() before sending new units",
                    ));
                }
                let block = u.as_audio().ok_or_else(|| {
                    Error::invalid_argument("pcm_enc", "PCM encoder requires MediaUnit::Audio")
                })?;
                let pkt = self.encode_block(block)?;
                self.pending.push_back(pkt);
                Ok(SendStatus::Accepted)
            }
            None => {
                self.draining = true;
                Ok(SendStatus::Accepted)
            }
        }
    }

    fn receive_packet(&mut self) -> Result<RecvStatus<Packet>> {
        if let Some(pkt) = self.pending.pop_front() {
            Ok(RecvStatus::Ready(pkt))
        } else if self.draining {
            Ok(RecvStatus::Eof)
        } else {
            Ok(RecvStatus::NeedMore)
        }
    }

    fn flush(&mut self) {
        self.pending.clear();
        self.draining = false;
    }
}

// =============================================================================
// Static Factory Definitions & Explicit Registry Wiring
// =============================================================================

macro_rules! define_pcm_codec {
    ($dec_factory:ident, $enc_factory:ident, $codec_id:expr, $name:expr, $long_name:expr) => {
        pub struct $dec_factory;

        impl DecoderFactory for $dec_factory {
            fn descriptor(&self) -> &DecoderDescriptor {
                static DESC: DecoderDescriptor = DecoderDescriptor {
                    codec_id: $codec_id,
                    name: $name,
                    long_name: $long_name,
                    media_type: MediaType::Audio,
                };
                &DESC
            }

            fn open(&self, track: &TrackDescriptor, ctx: CodecContext) -> Result<Box<dyn Decoder>> {
                Ok(Box::new(PcmDecoder::open($codec_id, track, ctx)?))
            }
        }

        pub struct $enc_factory;

        impl EncoderFactory for $enc_factory {
            fn descriptor(&self) -> &EncoderDescriptor {
                static DESC: EncoderDescriptor = EncoderDescriptor {
                    codec_id: $codec_id,
                    name: $name,
                    long_name: $long_name,
                    media_type: MediaType::Audio,
                };
                &DESC
            }

            fn open(
                &self,
                track: &TrackDescriptor,
                ctx: CodecContext,
                _pass: PassMode,
            ) -> Result<Box<dyn Encoder>> {
                Ok(Box::new(PcmEncoder::open($codec_id, track, ctx)?))
            }
        }
    };
}

define_pcm_codec!(
    PcmS8DecoderFactory,
    PcmS8EncoderFactory,
    CodecId::PcmS8,
    "pcm_s8",
    "PCM signed 8-bit"
);
define_pcm_codec!(
    PcmU8DecoderFactory,
    PcmU8EncoderFactory,
    CodecId::PcmU8,
    "pcm_u8",
    "PCM unsigned 8-bit"
);
define_pcm_codec!(
    PcmS16leDecoderFactory,
    PcmS16leEncoderFactory,
    CodecId::PcmS16le,
    "pcm_s16le",
    "PCM signed 16-bit little-endian"
);
define_pcm_codec!(
    PcmS16beDecoderFactory,
    PcmS16beEncoderFactory,
    CodecId::PcmS16be,
    "pcm_s16be",
    "PCM signed 16-bit big-endian"
);
define_pcm_codec!(
    PcmU16leDecoderFactory,
    PcmU16leEncoderFactory,
    CodecId::PcmU16le,
    "pcm_u16le",
    "PCM unsigned 16-bit little-endian"
);
define_pcm_codec!(
    PcmU16beDecoderFactory,
    PcmU16beEncoderFactory,
    CodecId::PcmU16be,
    "pcm_u16be",
    "PCM unsigned 16-bit big-endian"
);
define_pcm_codec!(
    PcmS24leDecoderFactory,
    PcmS24leEncoderFactory,
    CodecId::PcmS24le,
    "pcm_s24le",
    "PCM signed 24-bit little-endian"
);
define_pcm_codec!(
    PcmS24beDecoderFactory,
    PcmS24beEncoderFactory,
    CodecId::PcmS24be,
    "pcm_s24be",
    "PCM signed 24-bit big-endian"
);
define_pcm_codec!(
    PcmU24leDecoderFactory,
    PcmU24leEncoderFactory,
    CodecId::PcmU24le,
    "pcm_u24le",
    "PCM unsigned 24-bit little-endian"
);
define_pcm_codec!(
    PcmU24beDecoderFactory,
    PcmU24beEncoderFactory,
    CodecId::PcmU24be,
    "pcm_u24be",
    "PCM unsigned 24-bit big-endian"
);
define_pcm_codec!(
    PcmS32leDecoderFactory,
    PcmS32leEncoderFactory,
    CodecId::PcmS32le,
    "pcm_s32le",
    "PCM signed 32-bit little-endian"
);
define_pcm_codec!(
    PcmS32beDecoderFactory,
    PcmS32beEncoderFactory,
    CodecId::PcmS32be,
    "pcm_s32be",
    "PCM signed 32-bit big-endian"
);
define_pcm_codec!(
    PcmU32leDecoderFactory,
    PcmU32leEncoderFactory,
    CodecId::PcmU32le,
    "pcm_u32le",
    "PCM unsigned 32-bit little-endian"
);
define_pcm_codec!(
    PcmU32beDecoderFactory,
    PcmU32beEncoderFactory,
    CodecId::PcmU32be,
    "pcm_u32be",
    "PCM unsigned 32-bit big-endian"
);
define_pcm_codec!(
    PcmF32leDecoderFactory,
    PcmF32leEncoderFactory,
    CodecId::PcmF32le,
    "pcm_f32le",
    "PCM 32-bit floating point little-endian"
);
define_pcm_codec!(
    PcmF32beDecoderFactory,
    PcmF32beEncoderFactory,
    CodecId::PcmF32be,
    "pcm_f32be",
    "PCM 32-bit floating point big-endian"
);
define_pcm_codec!(
    PcmF64leDecoderFactory,
    PcmF64leEncoderFactory,
    CodecId::PcmF64le,
    "pcm_f64le",
    "PCM 64-bit floating point little-endian"
);
define_pcm_codec!(
    PcmF64beDecoderFactory,
    PcmF64beEncoderFactory,
    CodecId::PcmF64be,
    "pcm_f64be",
    "PCM 64-bit floating point big-endian"
);
define_pcm_codec!(
    PcmAlawDecoderFactory,
    PcmAlawEncoderFactory,
    CodecId::PcmAlaw,
    "pcm_alaw",
    "PCM A-law / G.711 A-law"
);
define_pcm_codec!(
    PcmMulawDecoderFactory,
    PcmMulawEncoderFactory,
    CodecId::PcmMulaw,
    "pcm_mulaw",
    "PCM mu-law / G.711 mu-law"
);

/// Register all 20 native PCM decoders and 20 native PCM encoders into `reg`.
pub fn register_pcm(reg: &mut Registry) {
    // Decoders
    reg.register_decoder(Arc::new(PcmS8DecoderFactory));
    reg.register_decoder(Arc::new(PcmU8DecoderFactory));
    reg.register_decoder(Arc::new(PcmS16leDecoderFactory));
    reg.register_decoder(Arc::new(PcmS16beDecoderFactory));
    reg.register_decoder(Arc::new(PcmU16leDecoderFactory));
    reg.register_decoder(Arc::new(PcmU16beDecoderFactory));
    reg.register_decoder(Arc::new(PcmS24leDecoderFactory));
    reg.register_decoder(Arc::new(PcmS24beDecoderFactory));
    reg.register_decoder(Arc::new(PcmU24leDecoderFactory));
    reg.register_decoder(Arc::new(PcmU24beDecoderFactory));
    reg.register_decoder(Arc::new(PcmS32leDecoderFactory));
    reg.register_decoder(Arc::new(PcmS32beDecoderFactory));
    reg.register_decoder(Arc::new(PcmU32leDecoderFactory));
    reg.register_decoder(Arc::new(PcmU32beDecoderFactory));
    reg.register_decoder(Arc::new(PcmF32leDecoderFactory));
    reg.register_decoder(Arc::new(PcmF32beDecoderFactory));
    reg.register_decoder(Arc::new(PcmF64leDecoderFactory));
    reg.register_decoder(Arc::new(PcmF64beDecoderFactory));
    reg.register_decoder(Arc::new(PcmAlawDecoderFactory));
    reg.register_decoder(Arc::new(PcmMulawDecoderFactory));

    // Encoders
    reg.register_encoder(Arc::new(PcmS8EncoderFactory));
    reg.register_encoder(Arc::new(PcmU8EncoderFactory));
    reg.register_encoder(Arc::new(PcmS16leEncoderFactory));
    reg.register_encoder(Arc::new(PcmS16beEncoderFactory));
    reg.register_encoder(Arc::new(PcmU16leEncoderFactory));
    reg.register_encoder(Arc::new(PcmU16beEncoderFactory));
    reg.register_encoder(Arc::new(PcmS24leEncoderFactory));
    reg.register_encoder(Arc::new(PcmS24beEncoderFactory));
    reg.register_encoder(Arc::new(PcmU24leEncoderFactory));
    reg.register_encoder(Arc::new(PcmU24beEncoderFactory));
    reg.register_encoder(Arc::new(PcmS32leEncoderFactory));
    reg.register_encoder(Arc::new(PcmS32beEncoderFactory));
    reg.register_encoder(Arc::new(PcmU32leEncoderFactory));
    reg.register_encoder(Arc::new(PcmU32beEncoderFactory));
    reg.register_encoder(Arc::new(PcmF32leEncoderFactory));
    reg.register_encoder(Arc::new(PcmF32beEncoderFactory));
    reg.register_encoder(Arc::new(PcmF64leEncoderFactory));
    reg.register_encoder(Arc::new(PcmF64beEncoderFactory));
    reg.register_encoder(Arc::new(PcmAlawEncoderFactory));
    reg.register_encoder(Arc::new(PcmMulawEncoderFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::{AudioStreamParams, Buffer, ErrorKind, SideData, StreamParams, Timestamp};

    const ALL_PCM_CODECS: [CodecId; 20] = [
        CodecId::PcmS8,
        CodecId::PcmU8,
        CodecId::PcmS16le,
        CodecId::PcmS16be,
        CodecId::PcmU16le,
        CodecId::PcmU16be,
        CodecId::PcmS24le,
        CodecId::PcmS24be,
        CodecId::PcmU24le,
        CodecId::PcmU24be,
        CodecId::PcmS32le,
        CodecId::PcmS32be,
        CodecId::PcmU32le,
        CodecId::PcmU32be,
        CodecId::PcmF32le,
        CodecId::PcmF32be,
        CodecId::PcmF64le,
        CodecId::PcmF64be,
        CodecId::PcmAlaw,
        CodecId::PcmMulaw,
    ];

    fn make_track(codec_id: CodecId, sample_rate: u32, layout: ChannelLayout) -> TrackDescriptor {
        let fmt = codec_id_to_sample_format(codec_id).unwrap();
        let mut params = AudioStreamParams::new(sample_rate, fmt, layout);
        params.bits_per_coded_sample = bits_per_coded_sample(codec_id).unwrap();
        params.block_align = params.channel_layout.nb_channels * (params.bits_per_coded_sample / 8);
        params.bit_rate = params.sample_rate as i64 * params.block_align as i64 * 8;
        TrackDescriptor::new(
            TrackId(0),
            codec_id,
            Rational::new(1, sample_rate as i32),
            StreamParams::Audio(params),
        )
    }

    #[test]
    fn register_pcm_registers_all_20_decoders_and_encoders() {
        let mut reg = Registry::new();
        register_pcm(&mut reg);

        assert_eq!(reg.decoders().count(), 20);
        assert_eq!(reg.encoders().count(), 20);

        for &id in &ALL_PCM_CODECS {
            let dec = reg.find_decoder(id).expect("missing decoder");
            assert_eq!(dec.descriptor().codec_id, id);
            assert_eq!(dec.descriptor().name, id.name());

            let enc = reg.find_encoder(id).expect("missing encoder");
            assert_eq!(enc.descriptor().codec_id, id);
            assert_eq!(enc.descriptor().name, id.name());
        }
    }

    #[test]
    fn g711_alaw_and_mulaw_exact_roundtrip_and_values() {
        // Every 8-bit A-law and mu-law codeword must decode to a 16-bit value that
        // re-encodes back to the exact original 8-bit codeword (except -0 vs +0 in mu-law 0x7f -> 0xff).
        for code in 0u8..=255 {
            let lin_a = alaw2linear(code);
            let re_a = linear2alaw(lin_a);
            assert_eq!(
                re_a, code,
                "A-law codeword 0x{code:02x} -> {lin_a} -> 0x{re_a:02x}"
            );

            let lin_u = ulaw2linear(code);
            let re_u = linear2ulaw(lin_u);
            if code == 0x7f {
                // In G.711 mu-law, both 0x7F (-0) and 0xFF (+0) decode to 0, which encodes to 0xFF.
                assert_eq!(lin_u, 0);
                assert_eq!(re_u, 0xff);
            } else {
                assert_eq!(
                    re_u, code,
                    "mu-law codeword 0x{code:02x} -> {lin_u} -> 0x{re_u:02x}"
                );
            }
        }
    }

    #[test]
    fn roundtrip_all_20_pcm_codecs_preserves_samples_and_vfr_timestamps() {
        let mut reg = Registry::new();
        register_pcm(&mut reg);

        for &codec_id in &ALL_PCM_CODECS {
            let track = make_track(codec_id, 48_000, ChannelLayout::stereo());
            let dec_factory = reg.find_decoder(codec_id).unwrap();
            let enc_factory = reg.find_encoder(codec_id).unwrap();

            let mut dec = dec_factory.open(&track, CodecContext::default()).unwrap();
            let mut enc = enc_factory
                .open(&track, CodecContext::default(), PassMode::Single)
                .unwrap();
            assert!(enc.extradata().is_empty());

            let coded_bps = coded_bytes_per_sample(codec_id).unwrap();
            let nb_samples = 64usize;
            let channels = 2usize;
            let total_samples = nb_samples * channels;

            // Construct valid coded payload for each codec
            let mut raw = vec![0u8; total_samples * coded_bps];
            match codec_id {
                CodecId::PcmF32le => {
                    for (i, chunk) in raw.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        let f = ((i as f32) - 64.0) * 0.01;
                        *chunk = f.to_le_bytes();
                    }
                }
                CodecId::PcmF32be => {
                    for (i, chunk) in raw.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        let f = ((i as f32) - 64.0) * 0.01;
                        *chunk = f.to_be_bytes();
                    }
                }
                CodecId::PcmF64le => {
                    for (i, chunk) in raw.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let f = ((i as f64) - 64.0) * 0.005;
                        *chunk = f.to_le_bytes();
                    }
                }
                CodecId::PcmF64be => {
                    for (i, chunk) in raw.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        let f = ((i as f64) - 64.0) * 0.005;
                        *chunk = f.to_be_bytes();
                    }
                }
                CodecId::PcmMulaw => {
                    for (i, b) in raw.iter_mut().enumerate() {
                        let c = (i as u8).wrapping_mul(17);
                        *b = if c == 0x7f { 0xff } else { c };
                    }
                }
                _ => {
                    for (i, b) in raw.iter_mut().enumerate() {
                        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
                    }
                }
            }

            let tb = Rational::new(1, 90_000);
            let mut in_pkt = Packet::new(Buffer::from_slice(&raw), TrackId(0), tb);
            in_pkt.pts = Timestamp::new(12_345);
            in_pkt.dts = Timestamp::new(12_345);
            in_pkt.duration = 120;
            in_pkt.side_data.push(SideData::SampleTrim {
                skip_start: 4,
                skip_end: 2,
            });

            // Decode
            assert_eq!(
                dec.send_packet(Some(&in_pkt)).unwrap(),
                SendStatus::Accepted
            );
            let RecvStatus::Ready(unit) = dec.receive().unwrap() else {
                panic!("expected decoded unit for {}", codec_id.name());
            };
            assert!(matches!(dec.receive().unwrap(), RecvStatus::NeedMore));

            let block = unit.as_audio().unwrap();
            assert_eq!(block.nb_samples, nb_samples as u32);
            assert_eq!(block.pts, Timestamp::new(12_345));
            assert_eq!(block.duration, 120);
            assert_eq!(block.time_base, tb);
            assert_eq!(block.side_data.len(), 1);

            // Re-encode
            assert_eq!(
                enc.send_unit(Some(&unit), EncodeHint::default()).unwrap(),
                SendStatus::Accepted
            );
            let RecvStatus::Ready(out_pkt) = enc.receive_packet().unwrap() else {
                panic!("expected encoded packet for {}", codec_id.name());
            };
            assert!(matches!(
                enc.receive_packet().unwrap(),
                RecvStatus::NeedMore
            ));

            assert_eq!(
                out_pkt.data.data(),
                raw.as_slice(),
                "mismatch for {}",
                codec_id.name()
            );
            assert_eq!(out_pkt.pts, Timestamp::new(12_345));
            assert_eq!(out_pkt.dts, Timestamp::new(12_345));
            assert_eq!(out_pkt.duration, 120);
            assert_eq!(out_pkt.time_base, tb);
            assert!(out_pkt.flags.contains(PacketFlags::KEY));
            assert_eq!(out_pkt.side_data.len(), 1);

            // Drain both
            assert_eq!(dec.send_packet(None).unwrap(), SendStatus::Accepted);
            assert!(matches!(dec.receive().unwrap(), RecvStatus::Eof));
            assert_eq!(
                enc.send_unit(None, EncodeHint::default()).unwrap(),
                SendStatus::Accepted
            );
            assert!(matches!(enc.receive_packet().unwrap(), RecvStatus::Eof));
            assert!(dec.health().is_clean());
        }
    }

    #[test]
    fn planar_input_to_pcm_encoder_interleaves_correctly() {
        let track = make_track(CodecId::PcmS16le, 44_100, ChannelLayout::stereo());
        let mut enc = PcmEncoder::open(CodecId::PcmS16le, &track, CodecContext::default()).unwrap();

        let ch0: [u8; 4] = [0x01, 0x02, 0x05, 0x06];
        let ch1: [u8; 4] = [0x03, 0x04, 0x07, 0x08];
        let mut block = AudioBlock::new(
            vec![Buffer::from_slice(&ch0), Buffer::from_slice(&ch1)],
            2,
            SampleFormat::S16Planar,
            44_100,
            ChannelLayout::stereo(),
        );
        block.pts = Timestamp::new(100);

        enc.send_unit(Some(&MediaUnit::Audio(block)), EncodeHint::default())
            .unwrap();
        let RecvStatus::Ready(pkt) = enc.receive_packet().unwrap() else {
            panic!("expected packet");
        };
        assert_eq!(
            pkt.data.data(),
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
    }

    #[test]
    fn unaligned_packet_resilient_vs_strict_policy() {
        let track = make_track(CodecId::PcmS16le, 44_100, ChannelLayout::stereo());

        // Stereo S16LE block size = 4 bytes. Send 6 bytes (1 valid frame + 2 trailing bytes).
        let pkt = Packet::new(
            Buffer::from_slice(&[1, 0, 2, 0, 99, 99]),
            TrackId(0),
            Rational::new(1, 44_100),
        );

        // Resilient mode truncates trailing partial sample and records HealthIssue::TruncatedPacket.
        let mut dec_res =
            PcmDecoder::open(CodecId::PcmS16le, &track, CodecContext::default()).unwrap();
        dec_res.send_packet(Some(&pkt)).unwrap();
        let RecvStatus::Ready(MediaUnit::Audio(block)) = dec_res.receive().unwrap() else {
            panic!("expected decoded block");
        };
        assert_eq!(block.nb_samples, 1);
        assert_eq!(block.planes[0].data(), &[1, 0, 2, 0]);
        assert_eq!(dec_res.health().corrupt_packets, 1);

        // Strict mode fails immediately with ErrorKind::CorruptStream.
        let strict_ctx = CodecContext {
            error_policy: ErrorPolicy::Strict,
            ..Default::default()
        };
        let mut dec_strict = PcmDecoder::open(CodecId::PcmS16le, &track, strict_ctx).unwrap();
        let err = dec_strict.send_packet(Some(&pkt)).unwrap_err();
        assert_eq!(err.kind, ErrorKind::CorruptStream);
    }
}
