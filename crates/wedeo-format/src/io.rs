use std::io::{Read, Seek, SeekFrom, Write};

use wedeo_core::error::{Error, Result};

/// Low-level I/O context trait. Implementations provide raw byte-level access
/// to a data source (file, network, pipe, etc.).
pub trait IoContext: Send + Sync {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize>;
    fn write(&mut self, buf: &[u8]) -> Result<usize>;
    fn seek(&mut self, pos: SeekFrom) -> Result<u64>;
    fn tell(&mut self) -> Result<u64>;
    fn size(&mut self) -> Result<u64>;
    fn is_seekable(&self) -> bool;
}

/// File-based I/O context.
pub struct FileIo {
    file: std::fs::File,
}

impl FileIo {
    pub fn open(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self { file })
    }

    pub fn create(path: &str) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self { file })
    }
}

impl IoContext for FileIo {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        Ok(self.file.read(buf)?)
    }

    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        Ok(self.file.write(buf)?)
    }

    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        Ok(self.file.seek(pos)?)
    }

    fn tell(&mut self) -> Result<u64> {
        Ok(self.file.stream_position()?)
    }

    fn size(&mut self) -> Result<u64> {
        let metadata = self.file.metadata()?;
        Ok(metadata.len())
    }

    fn is_seekable(&self) -> bool {
        true
    }
}

/// Dead I/O stub — all operations return errors.
/// Used after `take_inner()` transfers ownership of the real I/O context.
struct DeadIo;

impl IoContext for DeadIo {
    fn read(&mut self, _buf: &mut [u8]) -> Result<usize> {
        Err(Error::Other("I/O context has been taken".into()))
    }
    fn write(&mut self, _buf: &[u8]) -> Result<usize> {
        Err(Error::Other("I/O context has been taken".into()))
    }
    fn seek(&mut self, _pos: SeekFrom) -> Result<u64> {
        Err(Error::Other("I/O context has been taken".into()))
    }
    fn tell(&mut self) -> Result<u64> {
        Err(Error::Other("I/O context has been taken".into()))
    }
    fn size(&mut self) -> Result<u64> {
        Err(Error::Other("I/O context has been taken".into()))
    }
    fn is_seekable(&self) -> bool {
        false
    }
}

/// Buffered I/O wrapper providing typed read/write operations.
/// Wraps an `IoContext` and adds convenience methods for reading
/// integers in specific byte orders.
pub struct BufferedIo {
    inner: Box<dyn IoContext>,
    read_buf: Vec<u8>,
    read_pos: usize,
    read_len: usize,
    write_buf: Vec<u8>,
    write_pos: usize,
}

impl BufferedIo {
    const DEFAULT_BUF_SIZE: usize = 32768;

    pub fn new(inner: Box<dyn IoContext>) -> Self {
        Self {
            inner,
            read_buf: vec![0u8; Self::DEFAULT_BUF_SIZE],
            read_pos: 0,
            read_len: 0,
            write_buf: vec![0u8; Self::DEFAULT_BUF_SIZE],
            write_pos: 0,
        }
    }

    /// Transfer ownership of the inner IoContext out of this BufferedIo.
    /// After this call, the BufferedIo is backed by a dead stub — all
    /// subsequent I/O operations will return errors.
    pub fn take_inner(&mut self) -> Box<dyn IoContext> {
        self.read_pos = 0;
        self.read_len = 0;
        self.write_pos = 0;
        std::mem::replace(&mut self.inner, Box::new(DeadIo))
    }

    /// Read exactly `buf.len()` bytes into `buf`.
    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        let mut filled = 0;
        while filled < buf.len() {
            if self.read_pos >= self.read_len {
                self.fill_buffer()?;
                if self.read_len == 0 {
                    return Err(Error::Eof);
                }
            }
            let available = self.read_len - self.read_pos;
            let to_copy = available.min(buf.len() - filled);
            buf[filled..filled + to_copy]
                .copy_from_slice(&self.read_buf[self.read_pos..self.read_pos + to_copy]);
            self.read_pos += to_copy;
            filled += to_copy;
        }
        Ok(())
    }

    /// Read a single byte.
    pub fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    /// Read a little-endian u16.
    pub fn read_u16le(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    /// Read a big-endian u16.
    pub fn read_u16be(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_be_bytes(buf))
    }

    /// Read a little-endian u32.
    pub fn read_u32le(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Read a big-endian u32.
    pub fn read_u32be(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_be_bytes(buf))
    }

    /// Read a little-endian i32.
    pub fn read_i32le(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    /// Read a little-endian u64.
    pub fn read_u64le(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Read a big-endian u64.
    pub fn read_u64be(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_be_bytes(buf))
    }

    /// Read raw bytes of specified length.
    pub fn read_bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read up to `len` bytes, returning whatever is available.
    /// Returns empty Vec only at true EOF. Matches FFmpeg's `av_get_packet()`
    /// behavior of returning partial data when less than `len` bytes remain.
    pub fn read_up_to(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(len);
        while buf.len() < len {
            if self.read_pos >= self.read_len {
                self.fill_buffer()?;
                if self.read_len == 0 {
                    break; // EOF
                }
            }
            let available = self.read_len - self.read_pos;
            let to_copy = available.min(len - buf.len());
            buf.extend_from_slice(&self.read_buf[self.read_pos..self.read_pos + to_copy]);
            self.read_pos += to_copy;
        }
        Ok(buf)
    }

    /// Skip `n` bytes forward.
    pub fn skip(&mut self, n: u64) -> Result<()> {
        if self.inner.is_seekable() {
            // Use seek if possible
            let buffered_remaining = (self.read_len - self.read_pos) as u64;
            if n <= buffered_remaining {
                self.read_pos += n as usize;
                return Ok(());
            }
            let skip_from_io = n - buffered_remaining;
            self.read_pos = 0;
            self.read_len = 0;
            self.inner.seek(SeekFrom::Current(skip_from_io as i64))?;
        } else {
            // Read and discard
            let mut remaining = n;
            let buf_len = self.read_buf.len();
            let mut discard = vec![0u8; buf_len];
            while remaining > 0 {
                let to_skip = remaining.min(buf_len as u64) as usize;
                self.read_exact(&mut discard[..to_skip])?;
                remaining -= to_skip as u64;
            }
        }
        Ok(())
    }

    /// Seek to an absolute position.
    pub fn seek(&mut self, pos: u64) -> Result<u64> {
        self.flush_write()?;
        self.read_pos = 0;
        self.read_len = 0;
        self.inner.seek(SeekFrom::Start(pos))
    }

    /// Get current position.
    pub fn tell(&mut self) -> Result<u64> {
        let io_pos = self.inner.tell()?;
        let read_offset = (self.read_len - self.read_pos) as u64;
        let write_offset = self.write_pos as u64;
        Ok(io_pos.saturating_sub(read_offset) + write_offset)
    }

    /// Get the total size of the underlying source.
    pub fn size(&mut self) -> Result<u64> {
        self.inner.size()
    }

    /// Whether the underlying I/O is seekable.
    pub fn is_seekable(&self) -> bool {
        self.inner.is_seekable()
    }

    // --- Write methods ---

    /// Write a single byte.
    pub fn write_u8(&mut self, v: u8) -> Result<()> {
        self.write_all(&[v])
    }

    /// Write a little-endian u16.
    pub fn write_u16le(&mut self, v: u16) -> Result<()> {
        self.write_all(&v.to_le_bytes())
    }

    /// Write a big-endian u16.
    pub fn write_u16be(&mut self, v: u16) -> Result<()> {
        self.write_all(&v.to_be_bytes())
    }

    /// Write a little-endian u32.
    pub fn write_u32le(&mut self, v: u32) -> Result<()> {
        self.write_all(&v.to_le_bytes())
    }

    /// Write a big-endian u32.
    pub fn write_u32be(&mut self, v: u32) -> Result<()> {
        self.write_all(&v.to_be_bytes())
    }

    /// Write a little-endian u64.
    pub fn write_u64le(&mut self, v: u64) -> Result<()> {
        self.write_all(&v.to_le_bytes())
    }

    /// Write raw bytes.
    pub fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.write_all(data)
    }

    /// Write all bytes from the buffer.
    pub fn write_all(&mut self, data: &[u8]) -> Result<()> {
        let mut offset = 0;
        while offset < data.len() {
            let space = self.write_buf.len() - self.write_pos;
            let to_copy = space.min(data.len() - offset);
            self.write_buf[self.write_pos..self.write_pos + to_copy]
                .copy_from_slice(&data[offset..offset + to_copy]);
            self.write_pos += to_copy;
            offset += to_copy;
            if self.write_pos >= self.write_buf.len() {
                self.flush_write()?;
            }
        }
        Ok(())
    }

    /// Flush buffered write data to the underlying IoContext.
    pub fn flush(&mut self) -> Result<()> {
        self.flush_write()
    }

    fn flush_write(&mut self) -> Result<()> {
        if self.write_pos > 0 {
            let mut written = 0;
            while written < self.write_pos {
                let n = self.inner.write(&self.write_buf[written..self.write_pos])?;
                if n == 0 {
                    return Err(Error::Io(std::io::ErrorKind::WriteZero));
                }
                written += n;
            }
            self.write_pos = 0;
        }
        Ok(())
    }

    fn fill_buffer(&mut self) -> Result<()> {
        let n = self.inner.read(&mut self.read_buf)?;
        self.read_pos = 0;
        self.read_len = n;
        Ok(())
    }
}

/// Read a 4-byte ASCII tag (e.g., "RIFF", "WAVE").
pub fn read_tag(io: &mut BufferedIo) -> Result<[u8; 4]> {
    let mut tag = [0u8; 4];
    io.read_exact(&mut tag)?;
    Ok(tag)
}
