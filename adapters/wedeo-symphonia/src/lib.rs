//! Segregated Symphonia fallback audio adapter for `wedeo`.
//!
//! Provides a unified `"symphonia"` container demuxer (owning its [`wedeo_core::ByteSource`]
//! cleanly without any borrowed-I/O `DeadIo` hacks) and fallback audio decoders for
//! compressed, lossless, ADPCM, Opus, and PCM audio formats.
//!
//! # Registration Order
//!
//! Call [`register_symphonia`] **after** registering native demuxers and decoders on a
//! [`wedeo_core::Registry`]:
//! - Demuxer resolution picks the highest `probe()` score (`u8`), breaking ties in
//!   registration order. Symphonia caps its `probe()` score at `<= 50` (and returns
//!   exactly `50` for `RIFF`/`WAVE`), ensuring native demuxers (score `100`) always win.
//! - Decoder resolution (`Registry::find_decoder`) returns the first registered factory
//!   for a given [`wedeo_core::CodecId`], so registering native decoders first guarantees
//!   Symphonia acts strictly as a fallback.

#![forbid(unsafe_code)]

pub mod channel_layout;
pub mod codec_map;
pub mod decoder;
pub mod demuxer;
pub mod error;
pub(crate) mod io_bridge;
pub mod metadata;
pub mod opus;
pub mod sample_format;

use wedeo_core::Registry;

/// Register the Symphonia fallback demuxer (`"symphonia"`) and all supported Symphonia /
/// `opus-decoder` audio decoders into `reg`.
///
/// **Important:** Call this function **after** native format and codec registrations so
/// that native implementations take precedence over Symphonia fallbacks.
pub fn register_symphonia(reg: &mut Registry) {
    demuxer::register_demuxer(reg);
    decoder::register_decoders(reg);
    opus::register_opus_decoder(reg);
}
