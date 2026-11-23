use std::fs::{File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Result};

/// Readable byte source with optional fallible seeking (unifies local files, memory, and pipes).
pub trait ByteSource: Read + Send {
    fn is_seekable(&self) -> bool {
        false
    }

    fn seek(&mut self, _pos: SeekFrom) -> io::Result<u64> {
        Err(io::Error::new(
            io::ErrorKind::NotSeekable,
            "byte source is not seekable",
        ))
    }

    fn size_hint(&self) -> Option<u64> {
        None
    }
}

/// Writable byte sink with optional fallible seeking (unifies local files and `stdout` pipes).
pub trait ByteSink: Write + Send {
    fn is_seekable(&self) -> bool {
        false
    }

    fn seek(&mut self, _pos: SeekFrom) -> io::Result<u64> {
        Err(io::Error::new(
            io::ErrorKind::NotSeekable,
            "byte sink is not seekable",
        ))
    }
}

/// File-backed seekable `ByteSource`.
pub struct FileSource {
    file: File,
    size: Option<u64>,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::io("io/file_source", &e))?;
        let size = file.metadata().ok().map(|m| m.len());
        Ok(Self { file, size })
    }
}

impl Read for FileSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.file.read(buf)
    }
}

impl ByteSource for FileSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }

    fn size_hint(&self) -> Option<u64> {
        self.size
    }
}

/// File-backed seekable `ByteSink`.
pub struct FileSink {
    file: File,
}

impl FileSink {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| Error::io("io/file_sink", &e))?;
        Ok(Self { file })
    }
}

impl Write for FileSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl ByteSink for FileSink {
    fn is_seekable(&self) -> bool {
        true
    }

    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }
}

/// In-memory seekable `ByteSource`.
#[derive(Debug, Clone)]
pub struct MemorySource {
    cursor: Cursor<Vec<u8>>,
}

impl MemorySource {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            cursor: Cursor::new(data),
        }
    }

    pub fn from_slice(data: &[u8]) -> Self {
        Self::new(data.to_vec())
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.cursor.into_inner()
    }
}

impl Read for MemorySource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl ByteSource for MemorySource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }

    fn size_hint(&self) -> Option<u64> {
        Some(self.cursor.get_ref().len() as u64)
    }
}

/// In-memory seekable `ByteSink`.
#[derive(Debug, Default, Clone)]
pub struct MemorySink {
    cursor: Cursor<Vec<u8>>,
}

impl MemorySink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.cursor.get_ref()
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.cursor.into_inner()
    }
}

impl Write for MemorySink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.cursor.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl ByteSink for MemorySink {
    fn is_seekable(&self) -> bool {
        true
    }

    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

/// Buffered reader owning a `Box<dyn ByteSource>` with typed endian helpers and exact position tracking.
pub struct BufByteSource {
    inner: Box<dyn ByteSource>,
    buf: Vec<u8>,
    pos: usize,
    len: usize,
    /// Logical byte offset of `buf[pos]` from stream start.
    stream_pos: u64,
}

impl BufByteSource {
    const DEFAULT_CAP: usize = 32 * 1024;

    pub fn new(inner: Box<dyn ByteSource>) -> Self {
        Self::with_capacity(inner, Self::DEFAULT_CAP)
    }

    pub fn with_capacity(inner: Box<dyn ByteSource>, capacity: usize) -> Self {
        Self {
            inner,
            buf: vec![0u8; capacity.max(4096)],
            pos: 0,
            len: 0,
            stream_pos: 0,
        }
    }

    pub fn into_inner(self) -> Box<dyn ByteSource> {
        self.inner
    }

    #[inline]
    pub fn tell(&self) -> u64 {
        self.stream_pos
    }

    #[inline]
    pub fn is_seekable(&self) -> bool {
        self.inner.is_seekable()
    }

    #[inline]
    pub fn size_hint(&self) -> Option<u64> {
        self.inner.size_hint()
    }

    fn refill(&mut self) -> io::Result<usize> {
        self.pos = 0;
        self.len = self.inner.read(&mut self.buf)?;
        Ok(self.len)
    }

    pub fn read_exact_bytes(&mut self, mut dst: &mut [u8]) -> io::Result<()> {
        while !dst.is_empty() {
            if self.pos >= self.len && self.refill()? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected EOF",
                ));
            }
            let avail = self.len - self.pos;
            let n = avail.min(dst.len());
            dst[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
            self.pos += n;
            self.stream_pos += n as u64;
            dst = &mut dst[n..];
        }
        Ok(())
    }

    pub fn read_u8(&mut self) -> io::Result<u8> {
        let mut b = [0u8; 1];
        self.read_exact_bytes(&mut b)?;
        Ok(b[0])
    }

    pub fn read_u16le(&mut self) -> io::Result<u16> {
        let mut b = [0u8; 2];
        self.read_exact_bytes(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    pub fn read_u16be(&mut self) -> io::Result<u16> {
        let mut b = [0u8; 2];
        self.read_exact_bytes(&mut b)?;
        Ok(u16::from_be_bytes(b))
    }

    pub fn read_u32le(&mut self) -> io::Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact_bytes(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn read_u32be(&mut self) -> io::Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact_bytes(&mut b)?;
        Ok(u32::from_be_bytes(b))
    }

    pub fn read_u64le(&mut self) -> io::Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact_bytes(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    pub fn read_u64be(&mut self) -> io::Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact_bytes(&mut b)?;
        Ok(u64::from_be_bytes(b))
    }

    pub fn read_tag(&mut self) -> io::Result<[u8; 4]> {
        let mut tag = [0u8; 4];
        self.read_exact_bytes(&mut tag)?;
        Ok(tag)
    }

    pub fn skip(&mut self, n: u64) -> io::Result<()> {
        let buffered = (self.len - self.pos) as u64;
        if n <= buffered {
            self.pos += n as usize;
            self.stream_pos += n;
            return Ok(());
        }
        if self.inner.is_seekable() {
            let target = self.stream_pos.saturating_add(n);
            self.seek_abs(target)?;
            return Ok(());
        }
        let mut remaining = n - buffered;
        self.pos = self.len;
        self.stream_pos += buffered;
        while remaining > 0 {
            if self.refill()? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected EOF while skipping",
                ));
            }
            let take = remaining.min(self.len as u64) as usize;
            self.pos = take;
            self.stream_pos += take as u64;
            remaining -= take as u64;
        }
        Ok(())
    }

    pub fn seek_abs(&mut self, pos: u64) -> io::Result<u64> {
        self.pos = 0;
        self.len = 0;
        let actual = self.inner.seek(SeekFrom::Start(pos))?;
        self.stream_pos = actual;
        Ok(actual)
    }
}

impl Read for BufByteSource {
    fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
        if dst.is_empty() {
            return Ok(0);
        }
        if self.pos >= self.len && self.refill()? == 0 {
            return Ok(0);
        }
        let avail = self.len - self.pos;
        let n = avail.min(dst.len());
        dst[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        self.stream_pos += n as u64;
        Ok(n)
    }
}

impl ByteSource for BufByteSource {
    fn is_seekable(&self) -> bool {
        self.inner.is_seekable()
    }

    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let actual = match pos {
            SeekFrom::Start(p) => self.seek_abs(p)?,
            SeekFrom::Current(delta) => {
                let target = self
                    .stream_pos
                    .checked_add_signed(delta)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek underflow"))?;
                self.seek_abs(target)?
            }
            SeekFrom::End(delta) => {
                self.pos = 0;
                self.len = 0;
                let p = self.inner.seek(SeekFrom::End(delta))?;
                self.stream_pos = p;
                p
            }
        };
        Ok(actual)
    }

    fn size_hint(&self) -> Option<u64> {
        self.inner.size_hint()
    }
}

/// Wrapper adding cooperative cancellation to any `ByteSource` without threading a
/// cancellation token through every low-level demuxer read signature.
pub struct InterruptibleSource<R: ByteSource> {
    inner: R,
    cancelled: Arc<AtomicBool>,
}

impl<R: ByteSource> InterruptibleSource<R> {
    pub fn new(inner: R, cancelled: Arc<AtomicBool>) -> Self {
        Self { inner, cancelled }
    }

    #[inline]
    fn check(&self) -> io::Result<()> {
        if self.cancelled.load(Ordering::Relaxed) {
            Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "byte source cancelled",
            ))
        } else {
            Ok(())
        }
    }

    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: ByteSource> Read for InterruptibleSource<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.check()?;
        self.inner.read(buf)
    }
}

impl<R: ByteSource> ByteSource for InterruptibleSource<R> {
    fn is_seekable(&self) -> bool {
        self.inner.is_seekable()
    }

    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.check()?;
        self.inner.seek(pos)
    }

    fn size_hint(&self) -> Option<u64> {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buf_byte_source_read_skip_and_seek() {
        let mut src = BufByteSource::new(Box::new(MemorySource::from_slice(&[
            0x01, 0x02, 0x03, 0x04, b'R', b'I', b'F', b'F',
        ])));
        assert!(src.is_seekable());
        assert_eq!(src.size_hint(), Some(8));
        assert_eq!(src.read_u32be().unwrap(), 0x0102_0304);
        assert_eq!(src.tell(), 4);
        assert_eq!(src.read_tag().unwrap(), *b"RIFF");
        src.seek(SeekFrom::Start(2)).unwrap();
        assert_eq!(src.tell(), 2);
        assert_eq!(src.read_u16le().unwrap(), 0x0403);
    }

    #[test]
    fn interruptible_source_aborts_cleanly() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut src =
            InterruptibleSource::new(MemorySource::from_slice(&[1, 2, 3]), Arc::clone(&flag));
        let mut out = [0u8; 1];
        assert_eq!(src.read(&mut out).unwrap(), 1);
        flag.store(true, Ordering::Relaxed);
        let err = src.read(&mut out).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Interrupted);
    }
}
