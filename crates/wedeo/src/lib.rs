//! # wedeo
//!
//! Pure-Rust Universal A/V Adapter — ergonomic facade, deterministic
//! [`Registry`] builders, and typed [`audio::AudioReader`] /
//! [`video::VideoReader`] APIs.
//!
//! Licensed under `LGPL-2.1-or-later`.

#![forbid(unsafe_code)]

pub mod audio;
mod reader;
pub mod video;

pub use wedeo_core as core;

// ---------------------------------------------------------------------------
// Core top-level re-exports (wedeo-core)
// ---------------------------------------------------------------------------

pub use wedeo_core::{
    AudioBlock, AudioStreamParams, BorderPadding, BorderedPlane, Buffer, BufferPool, ByteSink,
    ByteSource, Channel, ChannelLayout, ChromaLocation, CodecContext, CodecId, ColorDesc,
    ColorModel, ColorPrimaries, ColorRange, CompOrder, ContainerHeader, CropRect, Decoder,
    DecoderFactory, Demuxer, DemuxerFactory, EncodeHint, Encoder, EncoderFactory, Endian, Error,
    ErrorKind, ErrorPolicy, FileSink, FileSource, FrameFlags, GpuDeviceId, GpuSurface, HealthEvent,
    HealthIssue, MatrixCoefficients, MediaType, MediaUnit, MemoryDomain, MemorySink, MemorySource,
    Metadata, Muxer, MuxerFactory, Packet, PacketFlags, PassMode, PictureType, PixelLayout,
    PlanePacking, Rational, RecvStatus, Registry, Result, SampleDepth, SampleFormat, SampleTrim,
    SeekTarget, SendStatus, SideData, StreamHealth, StreamParams, Subsampling, SubtitleEvent,
    SubtitlePayload, TIME_BASE, TIME_BASE_Q, TimedData, Timestamp, TrackDescriptor, TrackId,
    TransferCharacteristic, VideoFrame, VideoStorage, VideoStreamParams,
};

// ---------------------------------------------------------------------------
// Feature-gated Layer 1 re-exports
// ---------------------------------------------------------------------------

#[cfg(feature = "audio-ops")]
pub use wedeo_audio_ops as audio_ops;

#[cfg(feature = "video-ops")]
pub use wedeo_video_ops as video_ops;

#[cfg(feature = "syntax")]
pub use wedeo_syntax as syntax;

#[cfg(feature = "graph")]
pub use wedeo_graph as graph;

#[cfg(feature = "gpu-wgpu")]
pub use wedeo_gpu_wgpu as gpu_wgpu;

// ---------------------------------------------------------------------------
// Feature-gated Layer 2 format & codec re-exports
// ---------------------------------------------------------------------------

/// Enabled container format crates.
pub mod formats {
    #[cfg(feature = "ivf")]
    pub use wedeo_format_ivf as ivf;
    #[cfg(feature = "mkv")]
    pub use wedeo_format_mkv as mkv;
    #[cfg(feature = "mp4")]
    pub use wedeo_format_mp4 as mp4;
    #[cfg(feature = "wav")]
    pub use wedeo_format_wav as wav;
}

/// Enabled codec and adapter crates.
pub mod codecs {
    #[cfg(feature = "h264")]
    pub use wedeo_codec_h264 as h264;
    #[cfg(feature = "pcm")]
    pub use wedeo_codec_pcm as pcm;
    #[cfg(feature = "vp9")]
    pub use wedeo_codec_vp9 as vp9;
    #[cfg(feature = "rav1d")]
    pub use wedeo_rav1d as rav1d;
    #[cfg(feature = "rav1e")]
    pub use wedeo_rav1e as rav1e;
    #[cfg(feature = "symphonia")]
    pub use wedeo_symphonia as symphonia;
}

// ---------------------------------------------------------------------------
// Raw elementary stream DemuxerFactory adapter (via wedeo-syntax::raw)
// ---------------------------------------------------------------------------

#[cfg(feature = "syntax")]
mod raw_demuxers {
    use std::marker::PhantomData;
    use std::sync::Arc;

    use wedeo_core::{
        ByteSource, Demuxer, DemuxerDescriptor, DemuxerFactory, ErrorPolicy, Registry, Result,
    };
    use wedeo_syntax::aac::AacAdtsFramer;
    use wedeo_syntax::av1::Av1ObuFramer;
    use wedeo_syntax::h264::H264AnnexBFramer;
    use wedeo_syntax::hevc::HevcAnnexBFramer;
    use wedeo_syntax::raw::{ElementaryFramer, RawStreamDemuxer};

    struct RawDemuxerFactory<F> {
        descriptor: &'static DemuxerDescriptor,
        _marker: PhantomData<fn() -> F>,
    }

    impl<F: ElementaryFramer + Default + 'static> DemuxerFactory for RawDemuxerFactory<F> {
        fn descriptor(&self) -> &DemuxerDescriptor {
            self.descriptor
        }

        fn probe(&self, head: &[u8], ext: Option<&str>) -> u8 {
            let score = F::probe(head);
            if score > 0 {
                return score;
            }
            if let Some(ext) = ext.map(|e| e.trim_start_matches('.'))
                && self
                    .descriptor
                    .extensions
                    .iter()
                    .any(|e| e.eq_ignore_ascii_case(ext))
            {
                return 15;
            }
            0
        }

        fn open(
            &self,
            source: Box<dyn ByteSource>,
            policy: ErrorPolicy,
        ) -> Result<Box<dyn Demuxer>> {
            Ok(Box::new(RawStreamDemuxer::open(
                source,
                F::default(),
                policy,
            )?))
        }
    }

    pub(crate) fn register_raw_demuxers(reg: &mut Registry) {
        static H264_DESC: DemuxerDescriptor = DemuxerDescriptor {
            name: "h264",
            long_name: "Raw H.264 Annex B elementary stream",
            extensions: &["h264", "264"],
            mime_types: &["video/h264"],
        };
        static HEVC_DESC: DemuxerDescriptor = DemuxerDescriptor {
            name: "hevc",
            long_name: "Raw HEVC/H.265 Annex B elementary stream",
            extensions: &["hevc", "h265", "265"],
            mime_types: &["video/hevc"],
        };
        static AV1_DESC: DemuxerDescriptor = DemuxerDescriptor {
            name: "av1",
            long_name: "Raw AV1 OBU elementary stream",
            extensions: &["av1", "obu"],
            mime_types: &["video/av1"],
        };
        static AAC_DESC: DemuxerDescriptor = DemuxerDescriptor {
            name: "aac",
            long_name: "Raw AAC ADTS elementary stream",
            extensions: &["aac", "adts"],
            mime_types: &["audio/aac"],
        };

        reg.register_demuxer(Arc::new(RawDemuxerFactory::<H264AnnexBFramer> {
            descriptor: &H264_DESC,
            _marker: PhantomData,
        }));
        reg.register_demuxer(Arc::new(RawDemuxerFactory::<HevcAnnexBFramer> {
            descriptor: &HEVC_DESC,
            _marker: PhantomData,
        }));
        reg.register_demuxer(Arc::new(RawDemuxerFactory::<Av1ObuFramer> {
            descriptor: &AV1_DESC,
            _marker: PhantomData,
        }));
        reg.register_demuxer(Arc::new(RawDemuxerFactory::<AacAdtsFramer> {
            descriptor: &AAC_DESC,
            _marker: PhantomData,
        }));
    }
}

// ---------------------------------------------------------------------------
// Registry builders
// ---------------------------------------------------------------------------

/// Build a [`Registry`] containing **only** native pure-Rust formats, codecs,
/// and raw elementary stream demuxers (zero external adapters).
pub fn native_registry() -> Registry {
    let mut reg = Registry::new();

    // Native container formats first (highest probe priority).
    #[cfg(feature = "wav")]
    wedeo_format_wav::register_wav(&mut reg);
    #[cfg(feature = "mp4")]
    wedeo_format_mp4::register_mp4(&mut reg);
    #[cfg(feature = "mkv")]
    wedeo_format_mkv::register_mkv(&mut reg);
    #[cfg(feature = "ivf")]
    wedeo_format_ivf::register_ivf(&mut reg);

    // Native codecs.
    #[cfg(feature = "pcm")]
    wedeo_codec_pcm::register_pcm(&mut reg);
    #[cfg(feature = "h264")]
    wedeo_codec_h264::register_h264(&mut reg);
    #[cfg(feature = "vp9")]
    wedeo_codec_vp9::register_vp9(&mut reg);

    // Native raw elementary stream demuxers.
    #[cfg(feature = "syntax")]
    raw_demuxers::register_raw_demuxers(&mut reg);

    reg
}

/// Build the default [`Registry`] with all enabled native formats/codecs
/// registered **first**, followed by raw elementary demuxers, and finally
/// segregated external adapters (`rav1d`, `rav1e`, and `symphonia` last).
pub fn default_registry() -> Registry {
    let mut reg = native_registry();

    #[cfg(feature = "rav1d")]
    wedeo_rav1d::register_rav1d(&mut reg);
    #[cfg(feature = "rav1e")]
    wedeo_rav1e::register_rav1e(&mut reg);
    // Symphonia MUST be registered last so native WAV/PCM and other native
    // formats/codecs always win probe and codec resolution.
    #[cfg(feature = "symphonia")]
    wedeo_symphonia::register_symphonia(&mut reg);

    reg
}
