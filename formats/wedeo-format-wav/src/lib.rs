use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::packet::Packet;
use wedeo_format::demuxer::Stream;
use wedeo_format::io::BufferedIo;
use wedeo_format::muxer::{Muxer, OutputFormatDescriptor, OutputFormatFlags};
use wedeo_format::registry::MuxerFactory;

/// WAVE format codes
const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_ALAW: u16 = 0x0006;
const WAVE_FORMAT_MULAW: u16 = 0x0007;

// --- WAV Muxer ---

/// Reverse mapping from CodecId to WAV format tag and bits per sample.
fn codec_id_to_wav_format(codec_id: CodecId) -> Option<(u16, u16)> {
    match codec_id {
        CodecId::PcmU8 => Some((WAVE_FORMAT_PCM, 8)),
        CodecId::PcmS16le => Some((WAVE_FORMAT_PCM, 16)),
        CodecId::PcmS24le => Some((WAVE_FORMAT_PCM, 24)),
        CodecId::PcmS32le => Some((WAVE_FORMAT_PCM, 32)),
        CodecId::PcmF32le => Some((WAVE_FORMAT_IEEE_FLOAT, 32)),
        CodecId::PcmF64le => Some((WAVE_FORMAT_IEEE_FLOAT, 64)),
        CodecId::PcmAlaw => Some((WAVE_FORMAT_ALAW, 8)),
        CodecId::PcmMulaw => Some((WAVE_FORMAT_MULAW, 8)),
        _ => None,
    }
}

/// WAV muxer state.
struct WavMuxer {
    /// Position of the data chunk payload start.
    data_offset: u64,
    /// Accumulated data size written so far.
    data_size: u64,
    /// Position of the RIFF size field (offset 4 in the file).
    riff_size_pos: u64,
    /// Position of the data chunk size field.
    data_size_pos: u64,
}

impl Muxer for WavMuxer {
    fn write_header(&mut self, io: &mut BufferedIo, streams: &[Stream]) -> Result<()> {
        if streams.is_empty() {
            return Err(Error::StreamNotFound);
        }

        let params = &streams[0].codec_params;
        let (format_tag, bits_per_sample) =
            codec_id_to_wav_format(params.codec_id).ok_or(Error::InvalidArgument)?;

        let nb_channels = params.channel_layout.nb_channels as u16;
        let sample_rate = params.sample_rate;
        let block_align = nb_channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * block_align as u32;

        // RIFF header
        io.write_bytes(b"RIFF")?;
        self.riff_size_pos = 4; // position of the size field
        io.write_u32le(0)?; // placeholder for RIFF size
        io.write_bytes(b"WAVE")?;

        // fmt chunk
        io.write_bytes(b"fmt ")?;
        io.write_u32le(16)?; // fmt chunk size (PCM = 16 bytes)
        io.write_u16le(format_tag)?;
        io.write_u16le(nb_channels)?;
        io.write_u32le(sample_rate)?;
        io.write_u32le(byte_rate)?;
        io.write_u16le(block_align)?;
        io.write_u16le(bits_per_sample)?;

        // data chunk header
        io.write_bytes(b"data")?;
        self.data_size_pos = io.tell()?;
        io.write_u32le(0)?; // placeholder for data size
        self.data_offset = io.tell()?;

        io.flush()?;
        Ok(())
    }

    fn write_packet(&mut self, io: &mut BufferedIo, packet: &Packet) -> Result<()> {
        let data = packet.data.data();
        io.write_bytes(data)?;
        self.data_size += data.len() as u64;
        Ok(())
    }

    fn write_trailer(&mut self, io: &mut BufferedIo) -> Result<()> {
        io.flush()?;

        if io.is_seekable() {
            // Pad to even boundary if data size is odd
            if !self.data_size.is_multiple_of(2) {
                io.write_u8(0)?;
                io.flush()?;
            }

            // Patch RIFF size: data_size + 36 (WAVE tag + fmt chunk + data chunk header)
            // 36 = 4 (WAVE) + 8 (fmt header) + 16 (fmt data) + 8 (data header)
            let riff_size = self.data_size + 36;
            io.seek(self.riff_size_pos)?;
            io.write_u32le(riff_size as u32)?;

            // Patch data chunk size
            io.seek(self.data_size_pos)?;
            io.write_u32le(self.data_size as u32)?;

            io.flush()?;
        }

        Ok(())
    }
}

// --- Muxer Factory Registration ---

struct WavMuxerFactory;

impl MuxerFactory for WavMuxerFactory {
    fn descriptor(&self) -> &OutputFormatDescriptor {
        static DESC: OutputFormatDescriptor = OutputFormatDescriptor {
            name: "wav",
            long_name: "WAV / WAVE (Waveform Audio)",
            extensions: "wav",
            mime_types: "audio/x-wav",
            flags: OutputFormatFlags::empty(),
            audio_codec: CodecId::PcmS16le,
            video_codec: CodecId::None,
        };
        &DESC
    }

    fn create(&self) -> Result<Box<dyn Muxer>> {
        Ok(Box::new(WavMuxer {
            data_offset: 0,
            data_size: 0,
            riff_size_pos: 0,
            data_size_pos: 0,
        }))
    }
}

inventory::submit!(&WavMuxerFactory as &dyn MuxerFactory);
