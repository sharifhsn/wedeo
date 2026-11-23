// VP9 video codec for wedeo.
//
// Pure-Rust translation of FFmpeg's VP9 decoder (libavcodec/vp9*.c).
// No FFI, no bindgen. LGPL-2.1-or-later.

#![forbid(unsafe_code)]

pub mod block;
pub mod bool_decoder;
pub mod context;
pub mod data;
pub mod decoder;
pub mod dsp;
pub mod header;
pub mod idct;
pub mod intra_pred;
pub mod loopfilter;
pub mod mc;
pub mod mvs;
pub mod prob;
pub mod quant;
pub mod recon;
pub mod refs;
pub mod types;

pub use decoder::{Vp9Decoder, Vp9DecoderFactory, register_vp9};

#[inline]
pub(crate) fn invalid_data() -> wedeo_core::Error {
    wedeo_core::Error::invalid_data("vp9", "invalid VP9 bitstream")
}
