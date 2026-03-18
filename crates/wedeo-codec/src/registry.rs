use wedeo_core::codec_id::CodecId;
use wedeo_core::error::Result;

use crate::decoder::{CodecParameters, Decoder, DecoderDescriptor};
use crate::encoder::{Encoder, EncoderBuilder, EncoderDescriptor};

/// Factory for creating decoder instances. Implementations register via `inventory`.
pub trait DecoderFactory: Send + Sync {
    fn descriptor(&self) -> &DecoderDescriptor;
    fn create(&self, params: CodecParameters) -> Result<Box<dyn Decoder>>;
}

inventory::collect!(&'static dyn DecoderFactory);

/// Factory for creating encoder instances. Implementations register via `inventory`.
pub trait EncoderFactory: Send + Sync {
    fn descriptor(&self) -> &EncoderDescriptor;
    fn create(&self, builder: EncoderBuilder) -> Result<Box<dyn Encoder>>;
}

inventory::collect!(&'static dyn EncoderFactory);

/// Find a decoder factory by codec ID. When multiple decoders support the same
/// codec_id, the one with the highest priority wins.
pub fn find_decoder(codec_id: CodecId) -> Option<&'static dyn DecoderFactory> {
    inventory::iter::<&'static dyn DecoderFactory>()
        .filter(|f| f.descriptor().codec_id == codec_id)
        .max_by_key(|f| f.descriptor().priority)
        .copied()
}

/// Find a decoder factory by name.
pub fn find_decoder_by_name(name: &str) -> Option<&'static dyn DecoderFactory> {
    inventory::iter::<&'static dyn DecoderFactory>()
        .find(|f| f.descriptor().name == name)
        .copied()
}

/// Find an encoder factory by codec ID. When multiple encoders support the same
/// codec_id, the one with the highest priority wins.
pub fn find_encoder(codec_id: CodecId) -> Option<&'static dyn EncoderFactory> {
    inventory::iter::<&'static dyn EncoderFactory>()
        .filter(|f| f.descriptor().codec_id == codec_id)
        .max_by_key(|f| f.descriptor().priority)
        .copied()
}

/// Find an encoder factory by name.
pub fn find_encoder_by_name(name: &str) -> Option<&'static dyn EncoderFactory> {
    inventory::iter::<&'static dyn EncoderFactory>()
        .find(|f| f.descriptor().name == name)
        .copied()
}

/// Iterate over all registered decoder factories.
pub fn decoders() -> impl Iterator<Item = &'static dyn DecoderFactory> {
    inventory::iter::<&'static dyn DecoderFactory>().copied()
}

/// Iterate over all registered encoder factories.
pub fn encoders() -> impl Iterator<Item = &'static dyn EncoderFactory> {
    inventory::iter::<&'static dyn EncoderFactory>().copied()
}
