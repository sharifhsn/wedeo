use std::io::{Read, Seek, SeekFrom};
use std::sync::Mutex;

use symphonia::core::io::MediaSource;
use wedeo_core::ByteSource;

/// Private I/O bridge wrapping a wedeo `Box<dyn ByteSource>` as a symphonia `MediaSource`.
///
/// `Mutex` satisfies `MediaSource: Send + Sync` under `#![forbid(unsafe_code)]`, while
/// `Read` and `Seek` use `Mutex::get_mut` for zero-lock-overhead exclusive access.
pub(crate) struct WedeoMediaSource {
    inner: Mutex<Box<dyn ByteSource>>,
    cached_size: Option<u64>,
    seekable: bool,
}

impl WedeoMediaSource {
    pub(crate) fn new(inner: Box<dyn ByteSource>) -> Self {
        let cached_size = inner.size_hint();
        let seekable = inner.is_seekable();
        Self {
            inner: Mutex::new(inner),
            cached_size,
            seekable,
        }
    }
}

impl Read for WedeoMediaSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let inner = self
            .inner
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.read(buf)
    }
}

impl Seek for WedeoMediaSource {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        if !self.seekable {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotSeekable,
                "byte source is not seekable",
            ));
        }
        let inner = self
            .inner
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.seek(pos)
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
