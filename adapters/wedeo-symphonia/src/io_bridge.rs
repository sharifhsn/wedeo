use std::io::{Read, Seek, SeekFrom};

use symphonia::core::io::MediaSource;
use wedeo_format::io::IoContext;

/// Wraps a wedeo `Box<dyn IoContext>` as a symphonia `MediaSource`.
///
/// This is the I/O bridge that allows symphonia's FormatReader to read
/// from wedeo's I/O layer.
pub struct WedeoMediaSource {
    inner: Box<dyn IoContext>,
    /// Cached file size (symphonia's `byte_len()` takes `&self`).
    cached_size: Option<u64>,
    seekable: bool,
}

impl WedeoMediaSource {
    pub fn new(mut inner: Box<dyn IoContext>) -> Self {
        let cached_size = inner.size().ok();
        let seekable = inner.is_seekable();
        Self {
            inner,
            cached_size,
            seekable,
        }
    }
}

impl Read for WedeoMediaSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner
            .read(buf)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

impl Seek for WedeoMediaSource {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner
            .seek(pos)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

impl MediaSource for WedeoMediaSource {
    fn is_seekable(&self) -> bool {
        self.seekable
    }

    fn byte_len(&self) -> Option<u64> {
        self.cached_size
    }
}
