//! # wedeo
//!
//! **AI-generated Rust rewrite of FFmpeg** — a pure-Rust multimedia framework
//! for decoding, encoding, demuxing, and muxing audio and video.
//!
//! No bindgen, no c2rust, no FFI. Licensed under LGPL-2.1-or-later (same as
//! FFmpeg). Verification target: bit-for-bit output parity with FFmpeg's FATE
//! test suite.
//!
//! ## Quick start
//!
//! ```no_run
//! use wedeo::{InputContext, CodecParameters, DecoderBuilder, Error};
//!
//! fn main() -> wedeo::Result<()> {
//!     let mut input = InputContext::open("video.mp4")?;
//!     let stream = &input.streams[0];
//!     let mut decoder = DecoderBuilder::new(stream.codec_params.clone()).open()?;
//!
//!     loop {
//!         match input.read_packet() {
//!             Ok(packet) => {
//!                 decoder.send_packet(Some(&packet))?;
//!                 while let Ok(frame) = decoder.receive_frame() {
//!                     // Process decoded frame...
//!                     let _ = frame;
//!                 }
//!             }
//!             Err(Error::Eof) => break,
//!             Err(e) => return Err(e),
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Feature flags
//!
//! Implementation crates are gated behind feature flags so you can slim the
//! binary by disabling unused codecs and formats.
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `h264` | yes | H.264/AVC video decoder |
//! | `pcm` | yes | PCM audio codec |
//! | `wav` | yes | WAV format demuxer/muxer |
//! | `h264-format` | yes | H.264 Annex B raw bitstream demuxer |
//! | `mp4` | yes | MP4/MOV demuxer/muxer |
//! | `symphonia` | yes | Symphonia audio backend (AAC, FLAC, MP3, Vorbis, Opus, etc.) |
//! | `rav1d` | no | rav1d AV1 decoder (requires git dependency) |
//!
//! ## Architecture
//!
//! The crate re-exports the public API from the wedeo workspace:
//!
//! - **wedeo-core** — fundamental types: [`Frame`], [`Packet`], [`Rational`],
//!   [`CodecId`], [`PixelFormat`], [`SampleFormat`], [`Error`]
//! - **wedeo-codec** — codec framework: [`Decoder`], [`Encoder`],
//!   [`DecoderBuilder`], registry functions
//! - **wedeo-format** — format framework: [`Demuxer`], [`Muxer`],
//!   [`InputContext`], [`OutputContext`]
//! - **wedeo-filter** — filter framework: [`Filter`], [`FilterDescriptor`]
//! - **wedeo-resample** — audio resampling: [`Resampler`]
//! - **wedeo-scale** — pixel format conversion: [`Converter`], [`convert_frame`]
//!
//! For the full API of each sub-crate, add the individual crate as a
//! dependency.
//!
//! Source: <https://github.com/sharifhsn/wedeo>

// ---------------------------------------------------------------------------
// Ensure implementation crates are linked for inventory registration.
// ---------------------------------------------------------------------------

#[cfg(feature = "h264")]
use wedeo_codec_h264 as _;
#[cfg(feature = "pcm")]
use wedeo_codec_pcm as _;
#[cfg(feature = "h264-format")]
use wedeo_format_h264 as _;
#[cfg(feature = "mp4")]
use wedeo_format_mp4 as _;
#[cfg(feature = "wav")]
use wedeo_format_wav as _;
#[cfg(feature = "rav1d")]
use wedeo_rav1d as _;
#[cfg(feature = "symphonia")]
use wedeo_symphonia as _;

// ---------------------------------------------------------------------------
// Core types (wedeo-core)
// ---------------------------------------------------------------------------

pub use wedeo_core::{
    Buffer, ChannelLayout, ChromaLocation, CodecId, ColorPrimaries, ColorTransferCharacteristic,
    Error, Frame, FrameSideData, FrameSideDataType, MediaType, Metadata, NOPTS_VALUE, Packet,
    PixelFormat, Rational, Result, SampleFormat, TIME_BASE, TIME_BASE_Q,
};

// Submodule types that users commonly need.
pub use wedeo_core::frame::{
    AUDIO_MAX_PLANES, AudioFrameData, ColorRange, ColorSpace, FrameData, FrameFlags, FramePlane,
    PictureType, VIDEO_MAX_PLANES, VideoFrameData,
};
pub use wedeo_core::packet::{PacketFlags, PacketSideDataType};
pub use wedeo_core::rational::Rounding;

// ---------------------------------------------------------------------------
// Codec framework (wedeo-codec)
// ---------------------------------------------------------------------------

pub use wedeo_codec::CodecOptions;
pub use wedeo_codec::decoder::{CodecParameters, Decoder, DecoderBuilder, DecoderDescriptor};
pub use wedeo_codec::descriptor::{CodecCapabilities, CodecDescriptor, CodecProperties, Profile};
pub use wedeo_codec::encoder::{CodecFlags, Encoder, EncoderBuilder, EncoderDescriptor};
pub use wedeo_codec::registry::{
    DecoderFactory, EncoderFactory, decoders, encoders, find_decoder, find_decoder_by_name,
    find_encoder, find_encoder_by_name,
};

// ---------------------------------------------------------------------------
// Format framework (wedeo-format)
// ---------------------------------------------------------------------------

pub use wedeo_format::context::{InputContext, OutputContext};
pub use wedeo_format::demuxer::{
    Demuxer, DemuxerHeader, Discard, InputFormatDescriptor, InputFormatFlags,
    PROBE_SCORE_EXTENSION, PROBE_SCORE_MAX, ProbeData, SeekFlags, Stream,
};
pub use wedeo_format::muxer::{Muxer, OutputFormatDescriptor, OutputFormatFlags};
pub use wedeo_format::registry::{
    DemuxerFactory, MuxerFactory, demuxers, find_demuxer_by_name, find_muxer_by_name, muxers, probe,
};

// ---------------------------------------------------------------------------
// Filter framework (wedeo-filter)
// ---------------------------------------------------------------------------

pub use wedeo_filter::{
    Filter, FilterDescriptor, FilterFlags, FilterPadDescriptor, FilterPadDirection,
};

// ---------------------------------------------------------------------------
// Audio resampling (wedeo-resample)
// ---------------------------------------------------------------------------

pub use wedeo_resample::{Quality, Resampler};

// ---------------------------------------------------------------------------
// Video scaling (wedeo-scale)
// ---------------------------------------------------------------------------

pub use wedeo_scale::{Converter, convert_frame};
