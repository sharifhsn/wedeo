use bitflags::bitflags;

use wedeo_core::{CodecId, MediaType};

bitflags! {
    /// Codec capabilities, matching FFmpeg's AV_CODEC_CAP_*.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CodecCapabilities: u32 {
        const DRAW_HORIZ_BAND     = 1 << 0;
        const DR1                 = 1 << 1;
        const DELAY               = 1 << 5;
        const SMALL_LAST_FRAME    = 1 << 6;
        const SUBFRAMES           = 1 << 8;
        const EXPERIMENTAL        = 1 << 9;
        const CHANNEL_CONF        = 1 << 10;
        const FRAME_THREADS       = 1 << 12;
        const SLICE_THREADS       = 1 << 13;
        const PARAM_CHANGE        = 1 << 14;
        const OTHER_THREADS       = 1 << 15;
        const VARIABLE_FRAME_SIZE = 1 << 16;
        const AVOID_PROBING       = 1 << 17;
        const HARDWARE            = 1 << 18;
        const HYBRID              = 1 << 19;
        const ENCODER_REORDERED_OPAQUE = 1 << 20;
        const ENCODER_FLUSH       = 1 << 21;
        const ENCODER_RECON_FRAME = 1 << 22;
    }
}

bitflags! {
    /// Codec properties, matching FFmpeg's AV_CODEC_PROP_*.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CodecProperties: u32 {
        const INTRA_ONLY  = 1 << 0;
        const LOSSY       = 1 << 1;
        const LOSSLESS    = 1 << 2;
        const REORDER     = 1 << 3;
        const FIELDS      = 1 << 4;
        const BITMAP_SUB  = 1 << 16;
        const TEXT_SUB    = 1 << 17;
    }
}

/// Profile descriptor.
#[derive(Debug, Clone)]
pub struct Profile {
    pub id: i32,
    pub name: &'static str,
}

/// Codec descriptor, matching FFmpeg's AVCodecDescriptor.
#[derive(Debug, Clone)]
pub struct CodecDescriptor {
    pub id: CodecId,
    pub media_type: MediaType,
    pub name: &'static str,
    pub long_name: &'static str,
    pub properties: CodecProperties,
    pub profiles: &'static [Profile],
}
