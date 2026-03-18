use wedeo_core::channel_layout::ChannelLayout;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::Result;
use wedeo_core::frame::Frame;
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::Packet;
use wedeo_core::pixel_format::PixelFormat;
use wedeo_core::rational::Rational;
use wedeo_core::sample_format::SampleFormat;

use crate::descriptor::{CodecCapabilities, CodecDescriptor, CodecProperties};
use crate::options::CodecOptions;

/// Parameters needed to open a decoder, extracted from stream info.
#[derive(Debug, Clone)]
pub struct CodecParameters {
    pub codec_id: CodecId,
    pub media_type: MediaType,
    pub extradata: Vec<u8>,

    // Audio-specific
    pub sample_rate: u32,
    pub sample_format: SampleFormat,
    pub channel_layout: ChannelLayout,

    // Video-specific
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,

    pub bit_rate: i64,
    pub time_base: Rational,
    pub block_align: u32,
    pub bits_per_coded_sample: u32,
    pub bits_per_raw_sample: u32,

    /// Number of threads for decoding. 0 = auto (let the runtime decide).
    pub thread_count: u32,
    /// Codec tag (fourcc) for disambiguation when codec_id alone is ambiguous.
    /// 0 = unset.
    pub codec_tag: u32,
    /// Codec-private options (string key-value pairs).
    pub options: CodecOptions,
}

impl CodecParameters {
    pub fn new(codec_id: CodecId, media_type: MediaType) -> Self {
        Self {
            codec_id,
            media_type,
            extradata: Vec::new(),
            sample_rate: 0,
            sample_format: SampleFormat::None,
            channel_layout: ChannelLayout::unspec(0),
            width: 0,
            height: 0,
            pixel_format: PixelFormat::None,
            bit_rate: 0,
            time_base: Rational::new(0, 1),
            block_align: 0,
            bits_per_coded_sample: 0,
            bits_per_raw_sample: 0,
            thread_count: 0,
            codec_tag: 0,
            options: CodecOptions::new(),
        }
    }
}

/// Decoder trait — the main abstraction for all decoders.
///
/// Follows FFmpeg's send/receive model:
/// 1. `send_packet()` — feed compressed data
/// 2. `receive_frame()` — pull decoded data (may need multiple calls)
/// 3. `flush()` — reset state after seeking
pub trait Decoder: Send {
    /// Send a compressed packet to the decoder.
    /// Pass `None` to signal end-of-stream (drain mode).
    fn send_packet(&mut self, packet: Option<&Packet>) -> Result<()>;

    /// Receive a decoded frame from the decoder.
    /// Returns `Error::Again` when more input is needed.
    /// Returns `Error::Eof` when all data has been drained.
    fn receive_frame(&mut self) -> Result<Frame>;

    /// Flush the decoder (reset internal state for seeking).
    fn flush(&mut self);

    /// Get the codec descriptor.
    fn descriptor(&self) -> &CodecDescriptor;
}

/// Builder for creating decoders.
/// Follows the builder pattern instead of FFmpeg's allocate-then-mutate.
pub struct DecoderBuilder {
    params: CodecParameters,
}

impl DecoderBuilder {
    pub fn new(params: CodecParameters) -> Self {
        Self { params }
    }

    /// Set a codec-private option (string key-value pair).
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.params.options.set(key, value);
        self
    }

    /// Consume the builder and open a decoder.
    /// Uses the codec registry to find the appropriate decoder.
    pub fn open(self) -> Result<Box<dyn Decoder>> {
        crate::registry::find_decoder(self.params.codec_id)
            .ok_or(wedeo_core::Error::DecoderNotFound)?
            .create(self.params)
    }

    pub fn params(&self) -> &CodecParameters {
        &self.params
    }
}

/// Descriptor for a decoder implementation, used in the registry.
#[derive(Debug, Clone)]
pub struct DecoderDescriptor {
    pub codec_id: CodecId,
    pub name: &'static str,
    pub long_name: &'static str,
    pub media_type: MediaType,
    pub capabilities: CodecCapabilities,
    pub properties: CodecProperties,
    /// Priority for registry selection. Higher priority wins when multiple
    /// decoders support the same codec_id. Native implementations use 100,
    /// wrapper/adapter implementations use 50.
    pub priority: i32,
}
