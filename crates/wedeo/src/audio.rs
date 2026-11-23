//! Typed, audio-only reader API with zero video or GPU dependencies.

use std::path::Path;

use wedeo_core::{
    AudioBlock, AudioStreamParams, ByteSource, ContainerHeader, Error, ErrorPolicy, FileSource,
    MediaType, MediaUnit, Registry, Result, SeekTarget, StreamHealth, StreamParams,
};

#[cfg(feature = "audio-ops")]
pub use wedeo_audio_ops::{AudioOpList, CompiledAudioOps};

use crate::reader::TrackReader;

/// Ergonomic single-track audio reader returning typed [`AudioBlock`]s directly.
pub struct AudioReader {
    inner: TrackReader,
    params: AudioStreamParams,
}

impl AudioReader {
    /// Open an audio file at `path` using `reg` and [`ErrorPolicy::Resilient`].
    pub fn open(reg: &Registry, path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_policy(reg, path, ErrorPolicy::Resilient)
    }

    /// Open an audio file at `path` using `reg` and an explicit [`ErrorPolicy`].
    pub fn open_with_policy(
        reg: &Registry,
        path: impl AsRef<Path>,
        policy: ErrorPolicy,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        let ext = path_ref.extension().and_then(|e| e.to_str());
        let src = FileSource::open(path_ref)?;
        Self::from_source_with_ext(reg, Box::new(src), ext, policy)
    }

    /// Open an audio stream from any [`ByteSource`].
    pub fn from_source(
        reg: &Registry,
        source: Box<dyn ByteSource>,
        policy: ErrorPolicy,
    ) -> Result<Self> {
        Self::from_source_with_ext(reg, source, None, policy)
    }

    fn from_source_with_ext(
        reg: &Registry,
        source: Box<dyn ByteSource>,
        ext: Option<&str>,
        policy: ErrorPolicy,
    ) -> Result<Self> {
        let (inner, track) = TrackReader::open(reg, source, ext, MediaType::Audio, policy)?;
        let StreamParams::Audio(params) = track.params else {
            return Err(Error::internal(
                "wedeo::audio::AudioReader",
                "selected track did not carry AudioStreamParams",
            ));
        };
        Ok(Self { inner, params })
    }

    /// Audio stream parameters for the selected audio track.
    pub fn params(&self) -> &AudioStreamParams {
        &self.params
    }

    /// Container header metadata and track descriptors.
    pub fn header(&self) -> &ContainerHeader {
        self.inner.header()
    }

    /// Decode and return the next [`AudioBlock`], or `Ok(None)` at end-of-stream.
    pub fn next_block(&mut self) -> Result<Option<AudioBlock>> {
        match self.inner.next_unit()? {
            Some(MediaUnit::Audio(block)) => Ok(Some(block)),
            Some(other) => Err(Error::internal(
                "wedeo::audio::AudioReader",
                format!("audio decoder emitted non-audio unit: {:?}", other),
            )),
            None => Ok(None),
        }
    }

    /// Seek the underlying demuxer and flush the audio decoder.
    pub fn seek(&mut self, target: SeekTarget) -> Result<()> {
        self.inner.seek(target)
    }

    /// Decoder stream health ledger.
    pub fn health(&self) -> &StreamHealth {
        self.inner.health()
    }

    /// Demuxer stream health ledger.
    pub fn demuxer_health(&self) -> &StreamHealth {
        self.inner.demuxer_health()
    }
}
