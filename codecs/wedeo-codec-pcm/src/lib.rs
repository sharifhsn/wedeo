use std::collections::VecDeque;

use wedeo_codec::descriptor::{CodecCapabilities, CodecProperties};
use wedeo_codec::encoder::{Encoder, EncoderBuilder, EncoderDescriptor};
use wedeo_codec::registry::EncoderFactory;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::frame::Frame;
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::Packet;
use wedeo_core::sample_format::SampleFormat;

/// Map codec ID to native sample format.
fn codec_id_to_sample_format(id: CodecId) -> Option<SampleFormat> {
    match id {
        CodecId::PcmU8 => Some(SampleFormat::U8),
        CodecId::PcmS16le | CodecId::PcmS16be => Some(SampleFormat::S16),
        // U16 decodes to S16 (unsigned-to-signed conversion)
        CodecId::PcmU16le | CodecId::PcmU16be => Some(SampleFormat::S16),
        CodecId::PcmS32le | CodecId::PcmS32be => Some(SampleFormat::S32),
        // U32 decodes to S32 (unsigned-to-signed conversion)
        CodecId::PcmU32le | CodecId::PcmU32be => Some(SampleFormat::S32),
        // 24-bit formats decode to S32 (3-byte samples expanded to 4 bytes with 8-bit left shift)
        CodecId::PcmS24le | CodecId::PcmS24be => Some(SampleFormat::S32),
        CodecId::PcmU24le | CodecId::PcmU24be => Some(SampleFormat::S32),
        CodecId::PcmF32le | CodecId::PcmF32be => Some(SampleFormat::Flt),
        CodecId::PcmF64le | CodecId::PcmF64be => Some(SampleFormat::Dbl),
        _ => None,
    }
}

/// Check if a PCM codec ID requires byte-swapping on this platform.
fn needs_byte_swap(id: CodecId) -> bool {
    if cfg!(target_endian = "little") {
        matches!(
            id,
            CodecId::PcmS16be
                | CodecId::PcmS32be
                | CodecId::PcmF32be
                | CodecId::PcmF64be
                | CodecId::PcmU16be
                | CodecId::PcmU32be
                | CodecId::PcmS24be
                | CodecId::PcmU24be
        )
    } else {
        matches!(
            id,
            CodecId::PcmS16le
                | CodecId::PcmS32le
                | CodecId::PcmF32le
                | CodecId::PcmF64le
                | CodecId::PcmU16le
                | CodecId::PcmU32le
                | CodecId::PcmS24le
                | CodecId::PcmU24le
        )
    }
}

/// Byte-swap in-place for the given sample size.
fn byte_swap(data: &mut [u8], bytes_per_sample: usize) {
    match bytes_per_sample {
        2 => {
            for chunk in data.chunks_exact_mut(2) {
                chunk.swap(0, 1);
            }
        }
        3 => {
            for chunk in data.chunks_exact_mut(3) {
                chunk.swap(0, 2);
            }
        }
        4 => {
            for chunk in data.chunks_exact_mut(4) {
                chunk.swap(0, 3);
                chunk.swap(1, 2);
            }
        }
        8 => {
            for chunk in data.chunks_exact_mut(8) {
                chunk.swap(0, 7);
                chunk.swap(1, 6);
                chunk.swap(2, 5);
                chunk.swap(3, 4);
            }
        }
        _ => {}
    }
}

// =============================================================================
// PCM Encoder
// =============================================================================

/// PCM encoder — encodes raw PCM audio data.
///
/// PCM encoding is the reverse of decoding: read samples from frames,
/// convert to the coded format (byte-swap, signed-to-unsigned, 32-to-24-bit
/// packing) and write to packets.
struct PcmEncoder {
    codec_id: CodecId,
    // Stored for future use (format validation, muxer queries).
    _sample_format: SampleFormat,
    channel_layout: wedeo_core::channel_layout::ChannelLayout,
    // Stored for future use (muxer queries, bitrate computation).
    _sample_rate: u32,
    pending_frames: VecDeque<Frame>,
    drained: bool,
}

impl PcmEncoder {
    fn new(builder: &EncoderBuilder) -> Result<Self> {
        let sample_format =
            codec_id_to_sample_format(builder.codec_id).unwrap_or(builder.sample_format);

        Ok(Self {
            codec_id: builder.codec_id,
            _sample_format: sample_format,
            channel_layout: builder.channel_layout.clone(),
            _sample_rate: builder.sample_rate,
            pending_frames: VecDeque::new(),
            drained: false,
        })
    }

    fn encode_frame(&self, frame: &Frame) -> Result<Packet> {
        let audio = frame.audio().ok_or(Error::InvalidData)?;
        let nb_channels = self.channel_layout.nb_channels.max(1) as u32;
        let data = audio.planes[0].buffer.data();

        let output = encode_samples(self.codec_id, data, nb_channels);

        let mut pkt = Packet::from_slice(&output);
        pkt.pts = frame.pts;
        pkt.duration = frame.duration;

        Ok(pkt)
    }
}

impl Encoder for PcmEncoder {
    fn send_frame(&mut self, frame: Option<&Frame>) -> Result<()> {
        match frame {
            Some(f) => {
                self.pending_frames.push_back(f.clone());
                Ok(())
            }
            None => {
                self.drained = true;
                Ok(())
            }
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(frame) = self.pending_frames.pop_front() {
            self.encode_frame(&frame)
        } else if self.drained {
            Err(Error::Eof)
        } else {
            Err(Error::Again)
        }
    }

    fn flush(&mut self) {
        self.pending_frames.clear();
        self.drained = false;
    }
}

/// Encode PCM samples — the exact reverse of `decode_samples`.
///
/// Converts native-format audio samples back to the coded PCM format,
/// handling byte-swapping, signed-to-unsigned conversion, and 32-bit-to-24-bit
/// packing as needed.
fn encode_samples(codec_id: CodecId, data: &[u8], _nb_channels: u32) -> Vec<u8> {
    match codec_id {
        // --- 24-bit signed: read 4-byte S32, right-shift 8, write 3 bytes LE ---
        // Reverse of: v = LE24 as u32; sample = v << 8;
        CodecId::PcmS24le => {
            let n = data.len() / 4;
            let mut out = Vec::with_capacity(n * 3);
            for chunk in data.chunks_exact(4) {
                let sample = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let v = sample >> 8;
                out.push(v as u8);
                out.push((v >> 8) as u8);
                out.push((v >> 16) as u8);
            }
            out
        }
        // --- 24-bit signed BE: read 4-byte S32, right-shift 8, write 3 bytes BE ---
        // Reverse of: v = BE24 as u32; sample = v << 8;
        CodecId::PcmS24be => {
            let n = data.len() / 4;
            let mut out = Vec::with_capacity(n * 3);
            for chunk in data.chunks_exact(4) {
                let sample = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let v = sample >> 8;
                out.push((v >> 16) as u8);
                out.push((v >> 8) as u8);
                out.push(v as u8);
            }
            out
        }
        // --- 24-bit unsigned LE: read 4-byte S32, right-shift 8, add 0x800000, write 3 bytes LE ---
        // Reverse of: v = LE24 as u32; sample = (v - 0x800000) << 8;
        CodecId::PcmU24le => {
            let n = data.len() / 4;
            let mut out = Vec::with_capacity(n * 3);
            for chunk in data.chunks_exact(4) {
                let sample = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let v = (sample >> 8).wrapping_add(0x800000);
                out.push(v as u8);
                out.push((v >> 8) as u8);
                out.push((v >> 16) as u8);
            }
            out
        }
        // --- 24-bit unsigned BE: read 4-byte S32, right-shift 8, add 0x800000, write 3 bytes BE ---
        // Reverse of: v = BE24 as u32; sample = (v - 0x800000) << 8;
        CodecId::PcmU24be => {
            let n = data.len() / 4;
            let mut out = Vec::with_capacity(n * 3);
            for chunk in data.chunks_exact(4) {
                let sample = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let v = (sample >> 8).wrapping_add(0x800000);
                out.push((v >> 16) as u8);
                out.push((v >> 8) as u8);
                out.push(v as u8);
            }
            out
        }
        // --- 16-bit unsigned LE: read native S16, add 0x8000, write LE ---
        // Reverse of: v = U16LE; sample = v - 0x8000;
        CodecId::PcmU16le => {
            let n = data.len() / 2;
            let mut out = Vec::with_capacity(n * 2);
            for chunk in data.chunks_exact(2) {
                let sample = u16::from_ne_bytes([chunk[0], chunk[1]]);
                let v = sample.wrapping_add(0x8000);
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        // --- 16-bit unsigned BE: read native S16, add 0x8000, write BE ---
        // Reverse of: v = U16BE; sample = v - 0x8000;
        CodecId::PcmU16be => {
            let n = data.len() / 2;
            let mut out = Vec::with_capacity(n * 2);
            for chunk in data.chunks_exact(2) {
                let sample = u16::from_ne_bytes([chunk[0], chunk[1]]);
                let v = sample.wrapping_add(0x8000);
                out.extend_from_slice(&v.to_be_bytes());
            }
            out
        }
        // --- 32-bit unsigned LE: read native S32, add 0x80000000, write LE ---
        // Reverse of: v = U32LE; sample = v - 0x80000000;
        CodecId::PcmU32le => {
            let n = data.len() / 4;
            let mut out = Vec::with_capacity(n * 4);
            for chunk in data.chunks_exact(4) {
                let sample = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let v = sample.wrapping_add(0x80000000);
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        // --- 32-bit unsigned BE: read native S32, add 0x80000000, write BE ---
        // Reverse of: v = U32BE; sample = v - 0x80000000;
        CodecId::PcmU32be => {
            let n = data.len() / 4;
            let mut out = Vec::with_capacity(n * 4);
            for chunk in data.chunks_exact(4) {
                let sample = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let v = sample.wrapping_add(0x80000000);
                out.extend_from_slice(&v.to_be_bytes());
            }
            out
        }
        // --- All other formats: copy + byte-swap if needed ---
        // This handles S16LE/BE, S32LE/BE, F32LE/BE, F64LE/BE, U8
        _ => {
            let bps = codec_id_to_sample_format(codec_id)
                .map(|sf| sf.bytes_per_sample())
                .unwrap_or(1);
            let mut output = data.to_vec();
            if needs_byte_swap(codec_id) {
                byte_swap(&mut output, bps);
            }
            output
        }
    }
}

// --- Encoder Factory Registration ---

macro_rules! register_pcm_encoder {
    ($name:ident, $codec_id:expr, $factory_name:expr, $sample_fmts:expr) => {
        struct $name;

        impl EncoderFactory for $name {
            fn descriptor(&self) -> &EncoderDescriptor {
                static DESC: EncoderDescriptor = EncoderDescriptor {
                    codec_id: $codec_id,
                    name: $factory_name,
                    long_name: concat!("PCM ", $factory_name, " encoder"),
                    media_type: MediaType::Audio,
                    capabilities: CodecCapabilities::empty(),
                    properties: CodecProperties::LOSSLESS,
                    supported_sample_formats: $sample_fmts,
                    supported_pixel_formats: &[],
                    priority: 100,
                };
                &DESC
            }

            fn create(&self, builder: EncoderBuilder) -> Result<Box<dyn Encoder>> {
                Ok(Box::new(PcmEncoder::new(&builder)?))
            }
        }

        inventory::submit!(&$name as &dyn EncoderFactory);
    };
}

register_pcm_encoder!(
    PcmS16leEncoderFactory,
    CodecId::PcmS16le,
    "pcm_s16le",
    &[SampleFormat::S16]
);
register_pcm_encoder!(
    PcmS16beEncoderFactory,
    CodecId::PcmS16be,
    "pcm_s16be",
    &[SampleFormat::S16]
);
register_pcm_encoder!(
    PcmU8EncoderFactory,
    CodecId::PcmU8,
    "pcm_u8",
    &[SampleFormat::U8]
);
register_pcm_encoder!(
    PcmS24leEncoderFactory,
    CodecId::PcmS24le,
    "pcm_s24le",
    &[SampleFormat::S32]
);
register_pcm_encoder!(
    PcmS24beEncoderFactory,
    CodecId::PcmS24be,
    "pcm_s24be",
    &[SampleFormat::S32]
);
register_pcm_encoder!(
    PcmS32leEncoderFactory,
    CodecId::PcmS32le,
    "pcm_s32le",
    &[SampleFormat::S32]
);
register_pcm_encoder!(
    PcmS32beEncoderFactory,
    CodecId::PcmS32be,
    "pcm_s32be",
    &[SampleFormat::S32]
);
register_pcm_encoder!(
    PcmU16leEncoderFactory,
    CodecId::PcmU16le,
    "pcm_u16le",
    &[SampleFormat::S16]
);
register_pcm_encoder!(
    PcmU16beEncoderFactory,
    CodecId::PcmU16be,
    "pcm_u16be",
    &[SampleFormat::S16]
);
register_pcm_encoder!(
    PcmU24leEncoderFactory,
    CodecId::PcmU24le,
    "pcm_u24le",
    &[SampleFormat::S32]
);
register_pcm_encoder!(
    PcmU24beEncoderFactory,
    CodecId::PcmU24be,
    "pcm_u24be",
    &[SampleFormat::S32]
);
register_pcm_encoder!(
    PcmU32leEncoderFactory,
    CodecId::PcmU32le,
    "pcm_u32le",
    &[SampleFormat::S32]
);
register_pcm_encoder!(
    PcmU32beEncoderFactory,
    CodecId::PcmU32be,
    "pcm_u32be",
    &[SampleFormat::S32]
);
register_pcm_encoder!(
    PcmF32leEncoderFactory,
    CodecId::PcmF32le,
    "pcm_f32le",
    &[SampleFormat::Flt]
);
register_pcm_encoder!(
    PcmF32beEncoderFactory,
    CodecId::PcmF32be,
    "pcm_f32be",
    &[SampleFormat::Flt]
);
register_pcm_encoder!(
    PcmF64leEncoderFactory,
    CodecId::PcmF64le,
    "pcm_f64le",
    &[SampleFormat::Dbl]
);
register_pcm_encoder!(
    PcmF64beEncoderFactory,
    CodecId::PcmF64be,
    "pcm_f64be",
    &[SampleFormat::Dbl]
);
