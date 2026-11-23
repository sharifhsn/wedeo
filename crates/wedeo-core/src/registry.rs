use std::sync::Arc;

use crate::media::CodecId;
use crate::traits::{DecoderFactory, DemuxerFactory, EncoderFactory, MuxerFactory};

/// Plain concrete codec and format registry (zero global `inventory` state).
///
/// Resolution is strictly deterministic: earlier registrations win ties cleanly
/// (register native implementations before fallback adapters).
#[derive(Default)]
pub struct Registry {
    demuxers: Vec<Arc<dyn DemuxerFactory>>,
    muxers: Vec<Arc<dyn MuxerFactory>>,
    decoders: Vec<Arc<dyn DecoderFactory>>,
    encoders: Vec<Arc<dyn EncoderFactory>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_demuxer(&mut self, factory: Arc<dyn DemuxerFactory>) {
        self.demuxers.push(factory);
    }

    pub fn register_muxer(&mut self, factory: Arc<dyn MuxerFactory>) {
        self.muxers.push(factory);
    }

    pub fn register_decoder(&mut self, factory: Arc<dyn DecoderFactory>) {
        self.decoders.push(factory);
    }

    pub fn register_encoder(&mut self, factory: Arc<dyn EncoderFactory>) {
        self.encoders.push(factory);
    }

    /// Probe `head` bytes (and optional file extension without leading dot) across all
    /// registered demuxers. Highest strictly-positive score wins; registration order breaks ties.
    pub fn probe_demuxer(&self, head: &[u8], ext: Option<&str>) -> Option<&dyn DemuxerFactory> {
        let mut best: Option<(&dyn DemuxerFactory, u8)> = None;
        for f in &self.demuxers {
            let score = f.probe(head, ext);
            if score > 0 && best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((f.as_ref(), score));
            }
        }
        best.map(|(f, _)| f)
    }

    pub fn find_demuxer_by_name(&self, name: &str) -> Option<&dyn DemuxerFactory> {
        self.demuxers
            .iter()
            .find(|f| f.descriptor().name == name)
            .map(|f| f.as_ref())
    }

    pub fn find_muxer_by_name(&self, name: &str) -> Option<&dyn MuxerFactory> {
        self.muxers
            .iter()
            .find(|f| f.descriptor().name == name)
            .map(|f| f.as_ref())
    }

    pub fn find_muxer_by_ext(&self, ext: &str) -> Option<&dyn MuxerFactory> {
        let ext = ext.trim_start_matches('.');
        self.muxers
            .iter()
            .find(|f| {
                f.descriptor()
                    .extensions
                    .iter()
                    .any(|e| e.eq_ignore_ascii_case(ext))
            })
            .map(|f| f.as_ref())
    }

    pub fn find_decoder(&self, codec_id: CodecId) -> Option<&dyn DecoderFactory> {
        self.decoders
            .iter()
            .find(|f| f.descriptor().codec_id == codec_id)
            .map(|f| f.as_ref())
    }

    pub fn find_decoder_by_name(&self, name: &str) -> Option<&dyn DecoderFactory> {
        self.decoders
            .iter()
            .find(|f| f.descriptor().name == name)
            .map(|f| f.as_ref())
    }

    pub fn find_encoder(&self, codec_id: CodecId) -> Option<&dyn EncoderFactory> {
        self.encoders
            .iter()
            .find(|f| f.descriptor().codec_id == codec_id)
            .map(|f| f.as_ref())
    }

    pub fn find_encoder_by_name(&self, name: &str) -> Option<&dyn EncoderFactory> {
        self.encoders
            .iter()
            .find(|f| f.descriptor().name == name)
            .map(|f| f.as_ref())
    }

    pub fn demuxers(&self) -> impl Iterator<Item = &dyn DemuxerFactory> {
        self.demuxers.iter().map(|f| f.as_ref())
    }

    pub fn muxers(&self) -> impl Iterator<Item = &dyn MuxerFactory> {
        self.muxers.iter().map(|f| f.as_ref())
    }

    pub fn decoders(&self) -> impl Iterator<Item = &dyn DecoderFactory> {
        self.decoders.iter().map(|f| f.as_ref())
    }

    pub fn encoders(&self) -> impl Iterator<Item = &dyn EncoderFactory> {
        self.encoders.iter().map(|f| f.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{Error, ErrorPolicy, Result};
    use crate::io::ByteSource;
    use crate::media::{MediaType, TrackDescriptor};
    use crate::traits::{
        CodecContext, Decoder, DecoderDescriptor, Demuxer, DemuxerDescriptor, Muxer,
        MuxerDescriptor,
    };

    struct DummyDemux(&'static str, u8);
    impl DemuxerFactory for DummyDemux {
        fn descriptor(&self) -> &DemuxerDescriptor {
            match self.0 {
                "native" => {
                    static D: DemuxerDescriptor = DemuxerDescriptor {
                        name: "native",
                        long_name: "Native",
                        extensions: &["wav"],
                        mime_types: &["audio/wav"],
                    };
                    &D
                }
                _ => {
                    static D: DemuxerDescriptor = DemuxerDescriptor {
                        name: "fallback",
                        long_name: "Fallback",
                        extensions: &["wav"],
                        mime_types: &["audio/wav"],
                    };
                    &D
                }
            }
        }
        fn probe(&self, _head: &[u8], _ext: Option<&str>) -> u8 {
            self.1
        }
        fn open(&self, _s: Box<dyn ByteSource>, _p: ErrorPolicy) -> Result<Box<dyn Demuxer>> {
            Err(Error::unsupported("test", "dummy"))
        }
    }

    struct DummyDec(&'static str);
    impl DecoderFactory for DummyDec {
        fn descriptor(&self) -> &DecoderDescriptor {
            match self.0 {
                "native_pcm" => {
                    static D: DecoderDescriptor = DecoderDescriptor {
                        codec_id: CodecId::PcmS16le,
                        name: "native_pcm",
                        long_name: "Native PCM",
                        media_type: MediaType::Audio,
                    };
                    &D
                }
                _ => {
                    static D: DecoderDescriptor = DecoderDescriptor {
                        codec_id: CodecId::PcmS16le,
                        name: "symphonia_pcm",
                        long_name: "Symphonia PCM",
                        media_type: MediaType::Audio,
                    };
                    &D
                }
            }
        }
        fn open(&self, _t: &TrackDescriptor, _c: CodecContext) -> Result<Box<dyn Decoder>> {
            Err(Error::unsupported("test", "dummy"))
        }
    }

    struct DummyMux;
    impl MuxerFactory for DummyMux {
        fn descriptor(&self) -> &MuxerDescriptor {
            static D: MuxerDescriptor = MuxerDescriptor {
                name: "mp4",
                long_name: "MP4",
                extensions: &["mp4", "m4a"],
                mime_types: &["video/mp4"],
                default_video: CodecId::H264,
                default_audio: CodecId::Aac,
            };
            &D
        }
        fn create(&self, _sink: Box<dyn crate::io::ByteSink>) -> Result<Box<dyn Muxer>> {
            Err(Error::unsupported("test", "dummy"))
        }
    }

    #[test]
    fn registry_deterministic_tie_breaking_and_lookup() {
        let mut reg = Registry::new();
        // Equal probe score: earlier registration wins deterministically
        reg.register_demuxer(Arc::new(DummyDemux("native", 99)));
        reg.register_demuxer(Arc::new(DummyDemux("fallback", 99)));
        assert_eq!(
            reg.probe_demuxer(b"RIFF", Some("wav"))
                .unwrap()
                .descriptor()
                .name,
            "native"
        );

        // Higher score wins regardless of registration order
        let mut reg2 = Registry::new();
        reg2.register_demuxer(Arc::new(DummyDemux("native", 50)));
        reg2.register_demuxer(Arc::new(DummyDemux("fallback", 100)));
        assert_eq!(
            reg2.probe_demuxer(b"RIFF", None).unwrap().descriptor().name,
            "fallback"
        );

        // Decoders & muxer ext lookup
        reg.register_decoder(Arc::new(DummyDec("native_pcm")));
        reg.register_decoder(Arc::new(DummyDec("symphonia_pcm")));
        assert_eq!(
            reg.find_decoder(CodecId::PcmS16le)
                .unwrap()
                .descriptor()
                .name,
            "native_pcm"
        );
        assert_eq!(
            reg.find_decoder_by_name("symphonia_pcm")
                .unwrap()
                .descriptor()
                .name,
            "symphonia_pcm"
        );

        reg.register_muxer(Arc::new(DummyMux));
        assert_eq!(
            reg.find_muxer_by_ext(".M4A").unwrap().descriptor().name,
            "mp4"
        );
    }
}
