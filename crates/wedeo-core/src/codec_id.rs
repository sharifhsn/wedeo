/// Codec identifier, matching FFmpeg's AVCodecID discriminants for the
/// codecs we care about. Extend as needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CodecId {
    None = 0,

    // Video codecs
    Mpeg1video = 1,
    Mpeg2video = 2,
    H261 = 3,
    H263 = 4,
    Rv10 = 5,
    Rv20 = 6,
    Mjpeg = 7,
    Mpeg4 = 12,
    Rawvideo = 13,
    H264 = 27,
    Vp8 = 139,  // AV_CODEC_ID_VP8
    Vp9 = 167,  // AV_CODEC_ID_VP9
    Hevc = 173, // AV_CODEC_ID_HEVC
    Av1 = 225,  // AV_CODEC_ID_AV1

    // Audio codecs — start at 0x10000 per FFmpeg
    PcmS16le = 0x10000,
    PcmS16be = 0x10001,
    PcmU16le = 0x10002,
    PcmU16be = 0x10003,
    PcmS8 = 0x10004,
    PcmU8 = 0x10005,
    PcmMulaw = 0x10006,
    PcmAlaw = 0x10007,
    PcmS32le = 0x10008,
    PcmS32be = 0x10009,
    PcmU32le = 0x1000A,
    PcmU32be = 0x1000B,
    PcmS24le = 0x1000C,
    PcmS24be = 0x1000D,
    PcmU24le = 0x1000E,
    PcmU24be = 0x1000F,
    PcmF32be = 0x10014,
    PcmF32le = 0x10015,
    PcmF64be = 0x10016,
    PcmF64le = 0x10017,

    // ADPCM codecs — start at 0x11000 per FFmpeg
    AdpcmImaWav = 0x11001,
    AdpcmMs = 0x11006,

    Mp1 = 0x1502A,
    Mp2 = 0x15000,
    Mp3 = 0x15001,
    Aac = 0x15002,
    Ac3 = 0x15003,
    Vorbis = 0x15005,
    Flac = 0x1500C,
    Alac = 0x15010,
    WavPack = 0x15019,
    Opus = 0x1503C,

    // Subtitle codecs — start at 0x17000
    SubDvdSubtitle = 0x17000,
    SubDvbSubtitle = 0x17001,
    SubText = 0x17002,
    SubXsub = 0x17003,
    SubSsa = 0x17004,
    SubMovText = 0x17005,
    SubSrt = 0x17008,
    SubWebvtt = 0x17012,
}

impl CodecId {
    pub fn name(self) -> &'static str {
        match self {
            CodecId::None => "none",
            CodecId::Mpeg1video => "mpeg1video",
            CodecId::Mpeg2video => "mpeg2video",
            CodecId::H261 => "h261",
            CodecId::H263 => "h263",
            CodecId::Rv10 => "rv10",
            CodecId::Rv20 => "rv20",
            CodecId::Mjpeg => "mjpeg",
            CodecId::Mpeg4 => "mpeg4",
            CodecId::Rawvideo => "rawvideo",
            CodecId::H264 => "h264",
            CodecId::Vp8 => "vp8",
            CodecId::Vp9 => "vp9",
            CodecId::Av1 => "av1",
            CodecId::Hevc => "hevc",
            CodecId::PcmS16le => "pcm_s16le",
            CodecId::PcmS16be => "pcm_s16be",
            CodecId::PcmU16le => "pcm_u16le",
            CodecId::PcmU16be => "pcm_u16be",
            CodecId::PcmS8 => "pcm_s8",
            CodecId::PcmU8 => "pcm_u8",
            CodecId::PcmMulaw => "pcm_mulaw",
            CodecId::PcmAlaw => "pcm_alaw",
            CodecId::PcmS32le => "pcm_s32le",
            CodecId::PcmS32be => "pcm_s32be",
            CodecId::PcmU32le => "pcm_u32le",
            CodecId::PcmU32be => "pcm_u32be",
            CodecId::PcmS24le => "pcm_s24le",
            CodecId::PcmS24be => "pcm_s24be",
            CodecId::PcmU24le => "pcm_u24le",
            CodecId::PcmU24be => "pcm_u24be",
            CodecId::PcmF32le => "pcm_f32le",
            CodecId::PcmF32be => "pcm_f32be",
            CodecId::PcmF64le => "pcm_f64le",
            CodecId::PcmF64be => "pcm_f64be",
            CodecId::AdpcmImaWav => "adpcm_ima_wav",
            CodecId::AdpcmMs => "adpcm_ms",
            CodecId::Mp1 => "mp1",
            CodecId::Mp2 => "mp2",
            CodecId::Mp3 => "mp3",
            CodecId::Aac => "aac",
            CodecId::Ac3 => "ac3",
            CodecId::Vorbis => "vorbis",
            CodecId::Flac => "flac",
            CodecId::Alac => "alac",
            CodecId::WavPack => "wavpack",
            CodecId::Opus => "opus",
            CodecId::SubDvdSubtitle => "dvd_subtitle",
            CodecId::SubDvbSubtitle => "dvb_subtitle",
            CodecId::SubText => "text",
            CodecId::SubXsub => "xsub",
            CodecId::SubSsa => "ssa",
            CodecId::SubMovText => "mov_text",
            CodecId::SubSrt => "srt",
            CodecId::SubWebvtt => "webvtt",
        }
    }
}
