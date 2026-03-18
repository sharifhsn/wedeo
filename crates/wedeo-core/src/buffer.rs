use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};

/// SIMD alignment padding size (matches FFmpeg's AV_INPUT_BUFFER_PADDING_SIZE).
pub const INPUT_BUFFER_PADDING_SIZE: usize = 64;

/// Alignment constant for buffer allocations (64 bytes, matching AV_INPUT_BUFFER_PADDING_SIZE).
const BUFFER_ALIGN: usize = 64;

/// Reference-counted byte buffer with copy-on-write semantics.
///
/// Based on FFmpeg's AVBuffer model:
/// - Shared via `Arc` (reference counting)
/// - `make_writable()` copies only if shared (CoW)
/// - All allocations include SIMD padding initialized to zero
/// - Data is 64-byte aligned for SIMD operations
#[derive(Debug, Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

#[derive(Debug)]
struct BufferInner {
    data: AVec<u8, ConstAlign<BUFFER_ALIGN>>,
    /// Logical size (excluding padding).
    size: usize,
}

impl Buffer {
    /// Create a new buffer of the given size, zero-initialized with SIMD padding.
    pub fn new(size: usize) -> Self {
        let data = AVec::from_slice(BUFFER_ALIGN, &vec![0u8; size + INPUT_BUFFER_PADDING_SIZE]);
        Self {
            inner: Arc::new(BufferInner { data, size }),
        }
    }

    /// Create a buffer from existing data. Padding is appended.
    pub fn from_slice(data: &[u8]) -> Self {
        let size = data.len();
        let mut buf = AVec::with_capacity(BUFFER_ALIGN, size + INPUT_BUFFER_PADDING_SIZE);
        buf.extend_from_slice(data);
        buf.resize(size + INPUT_BUFFER_PADDING_SIZE, 0);
        Self {
            inner: Arc::new(BufferInner { data: buf, size }),
        }
    }

    /// Get the logical size of the buffer (excluding padding).
    pub fn size(&self) -> usize {
        self.inner.size
    }

    /// Get an immutable slice of the buffer data (excluding padding).
    pub fn data(&self) -> &[u8] {
        &self.inner.data[..self.inner.size]
    }

    /// Returns true if this is the only reference to the underlying data.
    pub fn is_writable(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Ensure the buffer is writable (CoW). If shared, creates a copy.
    pub fn make_writable(&mut self) {
        if !self.is_writable() {
            let size = self.inner.size;
            let mut data = AVec::with_capacity(BUFFER_ALIGN, size + INPUT_BUFFER_PADDING_SIZE);
            data.extend_from_slice(&self.inner.data[..size]);
            data.resize(size + INPUT_BUFFER_PADDING_SIZE, 0);
            self.inner = Arc::new(BufferInner { data, size });
        }
    }

    /// Get a mutable slice of the buffer data (excluding padding).
    /// Panics if the buffer is shared — call `make_writable()` first.
    pub fn data_mut(&mut self) -> &mut [u8] {
        let size = self.inner.size;
        let inner = Arc::get_mut(&mut self.inner)
            .expect("Buffer::data_mut called on shared buffer; call make_writable() first");
        &mut inner.data[..size]
    }

    /// Resize the buffer. If growing, new bytes are zero-initialized.
    /// Panics if the buffer is shared.
    pub fn resize(&mut self, new_size: usize) {
        let inner = Arc::get_mut(&mut self.inner)
            .expect("Buffer::resize called on shared buffer; call make_writable() first");
        inner.data.resize(new_size + INPUT_BUFFER_PADDING_SIZE, 0);
        // Ensure padding is zeroed
        for byte in &mut inner.data[new_size..] {
            *byte = 0;
        }
        inner.size = new_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_new() {
        let buf = Buffer::new(1024);
        assert_eq!(buf.size(), 1024);
        assert_eq!(buf.data().len(), 1024);
        assert!(buf.data().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_buffer_from_slice() {
        let data = [1u8, 2, 3, 4, 5];
        let buf = Buffer::from_slice(&data);
        assert_eq!(buf.size(), 5);
        assert_eq!(buf.data(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_buffer_cow() {
        let buf1 = Buffer::new(100);
        assert!(buf1.is_writable());

        let buf2 = buf1.clone();
        // Both are shared now
        assert!(!buf1.is_writable());
        assert!(!buf2.is_writable());

        let mut buf3 = buf2.clone();
        buf3.make_writable();
        // buf3 now has its own copy
        assert!(buf3.is_writable());
    }

    #[test]
    fn test_buffer_padding() {
        let buf = Buffer::new(10);
        // Internal storage should be at least 10 + PADDING
        assert!(buf.inner.data.len() >= 10 + INPUT_BUFFER_PADDING_SIZE);
        // Padding should be zeroed
        assert!(buf.inner.data[10..].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_buffer_alignment() {
        let buf = Buffer::new(100);
        let ptr = buf.data().as_ptr() as usize;
        assert_eq!(
            ptr % BUFFER_ALIGN,
            0,
            "Buffer data should be {BUFFER_ALIGN}-byte aligned, but got alignment {}",
            ptr % BUFFER_ALIGN
        );

        let buf2 = Buffer::from_slice(&[1, 2, 3]);
        let ptr2 = buf2.data().as_ptr() as usize;
        assert_eq!(
            ptr2 % BUFFER_ALIGN,
            0,
            "Buffer::from_slice data should be {BUFFER_ALIGN}-byte aligned"
        );
    }
}
