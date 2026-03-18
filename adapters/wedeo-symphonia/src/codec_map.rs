use symphonia::core::codecs::CodecType;
use wedeo_core::codec_id::CodecId;

/// Map a symphonia CodecType to a wedeo CodecId.
pub fn symphonia_to_wedeo(ct: CodecType) -> CodecId {
    // Symphonia codec type constants
    use symphonia::core::codecs::*;

    match ct {
        // PCM
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

/// Map a wedeo CodecId to a symphonia CodecType (for encoder lookup, if needed).
pub fn wedeo_to_symphonia(id: CodecId) -> CodecType {
    use symphonia::core::codecs::*;

    match id {
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
