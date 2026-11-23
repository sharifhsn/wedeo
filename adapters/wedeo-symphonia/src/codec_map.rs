use symphonia::core::codecs::CodecType;
use wedeo_core::CodecId;

/// Map a symphonia `CodecType` to a wedeo `CodecId`.
pub fn symphonia_to_wedeo(ct: CodecType) -> CodecId {
    use symphonia::core::codecs::*;

    match ct {
        // PCM
        CODEC_TYPE_PCM_S8 => CodecId::PcmS8,
        CODEC_TYPE_PCM_S16LE => CodecId::PcmS16le,
        CODEC_TYPE_PCM_S16BE => CodecId::PcmS16be,
        CODEC_TYPE_PCM_S24LE => CodecId::PcmS24le,
        CODEC_TYPE_PCM_S24BE => CodecId::PcmS24be,
        CODEC_TYPE_PCM_S32LE => CodecId::PcmS32le,
        CODEC_TYPE_PCM_S32BE => CodecId::PcmS32be,
        CODEC_TYPE_PCM_U8 => CodecId::PcmU8,
        CODEC_TYPE_PCM_U16LE => CodecId::PcmU16le,
        CODEC_TYPE_PCM_U16BE => CodecId::PcmU16be,
        CODEC_TYPE_PCM_U24LE => CodecId::PcmU24le,
        CODEC_TYPE_PCM_U24BE => CodecId::PcmU24be,
        CODEC_TYPE_PCM_U32LE => CodecId::PcmU32le,
        CODEC_TYPE_PCM_U32BE => CodecId::PcmU32be,
        CODEC_TYPE_PCM_F32LE => CodecId::PcmF32le,
        CODEC_TYPE_PCM_F32BE => CodecId::PcmF32be,
        CODEC_TYPE_PCM_F64LE => CodecId::PcmF64le,
        CODEC_TYPE_PCM_F64BE => CodecId::PcmF64be,
        CODEC_TYPE_PCM_ALAW => CodecId::PcmAlaw,
        CODEC_TYPE_PCM_MULAW => CodecId::PcmMulaw,
        // ADPCM
        CODEC_TYPE_ADPCM_MS => CodecId::AdpcmMs,
        CODEC_TYPE_ADPCM_IMA_WAV => CodecId::AdpcmImaWav,
        // Lossy audio
        CODEC_TYPE_MP1 => CodecId::Mp1,
        CODEC_TYPE_MP2 => CodecId::Mp2,
        CODEC_TYPE_MP3 => CodecId::Mp3,
        CODEC_TYPE_AAC => CodecId::Aac,
        CODEC_TYPE_VORBIS => CodecId::Vorbis,
        CODEC_TYPE_OPUS => CodecId::Opus,
        // Lossless audio
        CODEC_TYPE_FLAC => CodecId::Flac,
        CODEC_TYPE_ALAC => CodecId::Alac,
        CODEC_TYPE_WAVPACK => CodecId::WavPack,
        _ => CodecId::None,
    }
}

/// Map a wedeo `CodecId` to a symphonia `CodecType`.
pub fn wedeo_to_symphonia(id: CodecId) -> CodecType {
    use symphonia::core::codecs::*;

    match id {
        CodecId::PcmS8 => CODEC_TYPE_PCM_S8,
        CodecId::PcmS16le => CODEC_TYPE_PCM_S16LE,
        CodecId::PcmS16be => CODEC_TYPE_PCM_S16BE,
        CodecId::PcmS24le => CODEC_TYPE_PCM_S24LE,
        CodecId::PcmS24be => CODEC_TYPE_PCM_S24BE,
        CodecId::PcmS32le => CODEC_TYPE_PCM_S32LE,
        CodecId::PcmS32be => CODEC_TYPE_PCM_S32BE,
        CodecId::PcmU8 => CODEC_TYPE_PCM_U8,
        CodecId::PcmU16le => CODEC_TYPE_PCM_U16LE,
        CodecId::PcmU16be => CODEC_TYPE_PCM_U16BE,
        CodecId::PcmU24le => CODEC_TYPE_PCM_U24LE,
        CodecId::PcmU24be => CODEC_TYPE_PCM_U24BE,
        CodecId::PcmU32le => CODEC_TYPE_PCM_U32LE,
        CodecId::PcmU32be => CODEC_TYPE_PCM_U32BE,
        CodecId::PcmF32le => CODEC_TYPE_PCM_F32LE,
        CodecId::PcmF32be => CODEC_TYPE_PCM_F32BE,
        CodecId::PcmF64le => CODEC_TYPE_PCM_F64LE,
        CodecId::PcmF64be => CODEC_TYPE_PCM_F64BE,
        CodecId::PcmAlaw => CODEC_TYPE_PCM_ALAW,
        CodecId::PcmMulaw => CODEC_TYPE_PCM_MULAW,
        CodecId::AdpcmMs => CODEC_TYPE_ADPCM_MS,
        CodecId::AdpcmImaWav => CODEC_TYPE_ADPCM_IMA_WAV,
        CodecId::Mp1 => CODEC_TYPE_MP1,
        CodecId::Mp2 => CODEC_TYPE_MP2,
        CodecId::Mp3 => CODEC_TYPE_MP3,
        CodecId::Aac => CODEC_TYPE_AAC,
        CodecId::Vorbis => CODEC_TYPE_VORBIS,
        CodecId::Opus => CODEC_TYPE_OPUS,
        CodecId::Flac => CODEC_TYPE_FLAC,
        CodecId::Alac => CODEC_TYPE_ALAC,
        CodecId::WavPack => CODEC_TYPE_WAVPACK,
        _ => CODEC_TYPE_NULL,
    }
}

/// Infer default bits per coded sample from a `CodecId` when not explicitly set.
pub fn default_bits_for_codec(id: CodecId) -> u32 {
    match id {
        CodecId::AdpcmImaWav | CodecId::AdpcmMs => 4,
        CodecId::PcmS8 | CodecId::PcmU8 | CodecId::PcmAlaw | CodecId::PcmMulaw => 8,
        CodecId::PcmS16le | CodecId::PcmS16be | CodecId::PcmU16le | CodecId::PcmU16be => 16,
        CodecId::PcmS24le | CodecId::PcmS24be | CodecId::PcmU24le | CodecId::PcmU24be => 24,
        CodecId::PcmS32le
        | CodecId::PcmS32be
        | CodecId::PcmU32le
        | CodecId::PcmU32be
        | CodecId::PcmF32le
        | CodecId::PcmF32be => 32,
        CodecId::PcmF64le | CodecId::PcmF64be => 64,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_map_roundtrip_all_supported() {
        let codecs = [
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
            CodecId::AdpcmMs,
            CodecId::AdpcmImaWav,
            CodecId::Mp1,
            CodecId::Mp2,
            CodecId::Mp3,
            CodecId::Aac,
            CodecId::Vorbis,
            CodecId::Opus,
            CodecId::Flac,
            CodecId::Alac,
            CodecId::WavPack,
        ];

        for id in codecs {
            let sym = wedeo_to_symphonia(id);
            assert_ne!(sym, symphonia::core::codecs::CODEC_TYPE_NULL);
            assert_eq!(symphonia_to_wedeo(sym), id);
        }

        assert_eq!(
            wedeo_to_symphonia(CodecId::H264),
            symphonia::core::codecs::CODEC_TYPE_NULL
        );
        assert_eq!(
            symphonia_to_wedeo(symphonia::core::codecs::CODEC_TYPE_NULL),
            CodecId::None
        );
    }
}
