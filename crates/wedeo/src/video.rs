//! Typed video reader API symmetrical with [`crate::audio::AudioReader`].

use std::path::Path;

use wedeo_core::{
    ByteSource, ContainerHeader, Error, ErrorPolicy, FileSource, MediaType, MediaUnit, Registry,
    Result, SeekTarget, StreamHealth, StreamParams, VideoFrame, VideoStreamParams,
};

#[cfg(feature = "video-ops")]
pub use wedeo_video_ops::{CpuSwsBackend, SwsOpList};

use crate::reader::TrackReader;

/// Ergonomic single-track video reader returning typed [`VideoFrame`]s directly.
pub struct VideoReader {
    inner: TrackReader,
    params: VideoStreamParams,
}

impl VideoReader {
    /// Open a video file at `path` using `reg` and [`ErrorPolicy::Resilient`].
    pub fn open(reg: &Registry, path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_policy(reg, path, ErrorPolicy::Resilient)
    }

    /// Open a video file at `path` using `reg` and an explicit [`ErrorPolicy`].
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

    /// Open a video stream from any [`ByteSource`].
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
        let (inner, track) = TrackReader::open(reg, source, ext, MediaType::Video, policy)?;
        let StreamParams::Video(params) = track.params else {
            return Err(Error::internal(
                "wedeo::video::VideoReader",
                "selected track did not carry VideoStreamParams",
            ));
        };
        Ok(Self { inner, params })
    }

    /// Video stream parameters for the selected video track.
    pub fn params(&self) -> &VideoStreamParams {
        &self.params
    }

    /// Container header metadata and track descriptors.
    pub fn header(&self) -> &ContainerHeader {
        self.inner.header()
    }

    /// Decode and return the next [`VideoFrame`], or `Ok(None)` at end-of-stream.
    pub fn next_frame(&mut self) -> Result<Option<VideoFrame>> {
        match self.inner.next_unit()? {
            Some(MediaUnit::Video(frame)) => Ok(Some(frame)),
            Some(other) => Err(Error::internal(
                "wedeo::video::VideoReader",
                format!("video decoder emitted non-video unit: {:?}", other),
            )),
            None => Ok(None),
        }
    }

    /// Seek the underlying demuxer and flush the video decoder.
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
