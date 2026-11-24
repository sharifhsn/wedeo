use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::any::Any;
use std::fmt;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use crate::video::PixelLayout;

/// Required byte alignment for all CPU media buffers (AVX-512 / cache-line aligned).
pub const SIMD_ALIGN: usize = 64;

/// Guaranteed readable zeroed tail padding appended after every pooled CPU buffer
/// (matches FFmpeg's `AV_INPUT_BUFFER_PADDING_SIZE`).
pub const SIMD_TAIL_PADDING: usize = 64;

/// Default maximum bytes retained in a `BufferPool` (256 MiB).
const DEFAULT_POOL_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Round `n` up to the next multiple of `align` (power of two).
#[inline]
pub(crate) const fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (n + (align - 1)) & !(align - 1)
}

/// Compute the physical allocation size for `logical_len` bytes plus `SIMD_TAIL_PADDING`,
/// rounded up to a non-zero multiple of `SIMD_ALIGN`.
#[inline]
pub(crate) fn aligned_alloc_size(logical_len: usize) -> usize {
    let with_pad = logical_len
        .checked_add(SIMD_TAIL_PADDING)
        .expect("buffer size overflow");
    align_up(with_pad, SIMD_ALIGN).max(SIMD_ALIGN)
}

/// Low-level uniquely-owned 64-byte aligned heap allocation.
///
/// Shared internally by `Buffer`, `BufferPool`, and `RowProgressPlane` so every
/// CPU allocation goes through a single audited memory primitive.
pub(crate) struct AlignedRaw {
    ptr: NonNull<u8>,
    capacity: usize,
}

// SAFETY: `AlignedRaw` uniquely owns its heap allocation, identical to `Box<[u8]>`.
unsafe impl Send for AlignedRaw {}
// SAFETY: Shared immutable access to initialized bytes is safe across threads.
unsafe impl Sync for AlignedRaw {}

impl AlignedRaw {
    /// Allocate at least `logical_len + SIMD_TAIL_PADDING` zeroed bytes aligned to `SIMD_ALIGN`.
    pub(crate) fn new_zeroed(logical_len: usize) -> Self {
        let capacity = aligned_alloc_size(logical_len);
        let layout = Layout::from_size_align(capacity, SIMD_ALIGN).expect("invalid buffer layout");
        // SAFETY: `capacity >= SIMD_ALIGN > 0` and `SIMD_ALIGN` is a power of two.
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            handle_alloc_error(layout);
        };
        Self { ptr, capacity }
    }

    #[inline]
    pub(crate) fn base_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` points to `self.capacity` initialized bytes for `&self`'s lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.capacity) }
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` guarantees exclusive access to all `self.capacity` initialized bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }
}

impl Drop for AlignedRaw {
    fn drop(&mut self) {
        let layout =
            Layout::from_size_align(self.capacity, SIMD_ALIGN).expect("invalid buffer layout");
        // SAFETY: `self.ptr` was allocated with this exact `layout` via `alloc_zeroed`.
        unsafe { dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// Ref-counted wrapper around `AlignedRaw` that automatically returns the allocation
/// to its originating `BufferPool` when the final `Arc<PooledRaw>` drops.
pub(crate) struct PooledRaw {
    raw: Option<AlignedRaw>,
    pool: Option<Arc<BufferPool>>,
}

impl PooledRaw {
    pub(crate) fn new(raw: AlignedRaw, pool: Option<Arc<BufferPool>>) -> Arc<Self> {
        Arc::new(Self {
            raw: Some(raw),
            pool,
        })
    }

    #[inline]
    pub(crate) fn base_ptr(&self) -> NonNull<u8> {
        self.raw
            .as_ref()
            .expect("PooledRaw accessed after drop")
            .base_ptr()
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.raw
            .as_ref()
            .expect("PooledRaw accessed after drop")
            .capacity()
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        self.raw
            .as_ref()
            .expect("PooledRaw accessed after drop")
            .as_slice()
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        self.raw
            .as_mut()
            .expect("PooledRaw accessed after drop")
            .as_mut_slice()
    }
}

impl Drop for PooledRaw {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take()
            && let Some(pool) = self.pool.take()
        {
            pool.reclaim(raw);
        }
    }
}

/// Opaque identifier for a GPU device (e.g. an external Tessera `wgpu::Device`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuDeviceId(pub u64);

/// Memory residency domain orthogonal to pixel layout and colorimetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryDomain {
    Cpu,
    Gpu(GpuDeviceId),
}

/// Optional OS-level zero-copy GPU surface handle for HW decoder/encoder interop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeGpuHandle {
    DmaBufFd(i32),
    IoSurfaceId(u32),
    VulkanImage(u64),
    D3d11Handle(usize),
}

/// Zero-dependency GPU surface abstraction implemented by `wedeo-gpu-wgpu` or HW adapters.
pub trait GpuSurface: Send + Sync + 'static {
    fn device_id(&self) -> GpuDeviceId;
    fn layout(&self) -> PixelLayout;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    /// Downcast escape hatch so `wedeo-gpu-wgpu` and Tessera extract `wgpu::Texture` zero-copy.
    fn as_any(&self) -> &dyn Any;
    /// Read the surface's pixels on its owning device, in logical plane order.
    fn readback_planes(&self) -> crate::Result<Vec<BorderedPlane>> {
        Err(crate::Error::unsupported(
            "gpu_surface",
            "this surface has no CPU readback provider",
        ))
    }
    /// Optional OS zero-copy interop handle.
    fn export_native(&self) -> Option<NativeGpuHandle> {
        None
    }
}

/// Zero-copy externally owned CPU buffer (e.g. `rav1d` `Dav1dPicture` guard, mmap, NumPy).
pub trait ExternalCpuBuffer: Send + Sync + 'static {
    /// Logical byte slice.
    fn as_slice(&self) -> &[u8];
    /// Optional slice including readable trailing SIMD padding (`>= as_slice().len()`).
    fn as_padded_slice(&self) -> &[u8] {
        self.as_slice()
    }
}

#[derive(Default)]
struct PoolState {
    free: Vec<AlignedRaw>,
    total_bytes: usize,
}

/// Reusable pool of 64-byte aligned CPU slabs with guaranteed 64-byte zeroed tail padding.
///
/// Reuses warm physical memory pages to eliminate steady-state `malloc`/`free` and minor
/// page faults across frames.
pub struct BufferPool {
    state: Mutex<PoolState>,
    max_bytes: usize,
}

impl BufferPool {
    /// Create a pool with the default 256 MiB retention cap.
    pub fn new() -> Arc<Self> {
        Self::with_max_bytes(DEFAULT_POOL_MAX_BYTES)
    }

    /// Create a pool retaining at most `max_bytes` of idle memory.
    pub fn with_max_bytes(max_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(PoolState::default()),
            max_bytes,
        })
    }

    pub(crate) fn acquire_raw(
        self: &Arc<Self>,
        logical_len: usize,
        zero_body: bool,
    ) -> Arc<PooledRaw> {
        let needed = aligned_alloc_size(logical_len);
        let max_acceptable = needed.saturating_mul(2);

        let reused = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let idx = st
                .free
                .iter()
                .position(|b| b.capacity() >= needed && b.capacity() <= max_acceptable);
            if let Some(i) = idx {
                let raw = st.free.swap_remove(i);
                st.total_bytes = st.total_bytes.saturating_sub(raw.capacity());
                Some(raw)
            } else {
                None
            }
        };

        let raw = if let Some(mut raw) = reused {
            let slice = raw.as_mut_slice();
            if zero_body {
                slice[..logical_len + SIMD_TAIL_PADDING].fill(0);
            } else {
                // Body bytes are already initialized from initial alloc_zeroed; zero tail only.
                slice[logical_len..logical_len + SIMD_TAIL_PADDING].fill(0);
            }
            raw
        } else {
            AlignedRaw::new_zeroed(logical_len)
        };

        PooledRaw::new(raw, Some(Arc::clone(self)))
    }

    /// Acquire a zero-initialized pooled `Buffer` of `len` logical bytes.
    pub fn acquire(self: &Arc<Self>, len: usize) -> Buffer {
        let raw = self.acquire_raw(len, true);
        Buffer::from_pooled_raw(raw, len)
    }

    /// Acquire a pooled `Buffer` whose body may contain previously initialized bytes
    /// (zero memset on reuse), while guaranteeing the 64-byte tail padding is zeroed.
    pub fn acquire_reused(self: &Arc<Self>, len: usize) -> Buffer {
        let raw = self.acquire_raw(len, false);
        Buffer::from_pooled_raw(raw, len)
    }

    /// Acquire a pooled `Buffer` initialized by copying `data`.
    pub fn acquire_from_slice(self: &Arc<Self>, data: &[u8]) -> Buffer {
        let mut buf = self.acquire_reused(data.len());
        if let Some(dst) = buf.try_data_mut() {
            dst.copy_from_slice(data);
        }
        buf
    }

    /// Number of idle buffers currently cached in the pool.
    pub fn pooled_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .free
            .len()
    }

    /// Total idle bytes currently cached in the pool.
    pub fn pooled_bytes(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .total_bytes
    }

    /// Drop all cached idle buffers immediately.
    pub fn clear(&self) {
        let drained = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            st.total_bytes = 0;
            std::mem::take(&mut st.free)
        };
        drop(drained);
    }

    fn reclaim(&self, raw: AlignedRaw) {
        let cap = raw.capacity();
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.total_bytes.saturating_add(cap) <= self.max_bytes {
            st.total_bytes += cap;
            st.free.push(raw);
        } else {
            drop(st);
            drop(raw);
        }
    }
}

impl fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferPool")
            .field("pooled_count", &self.pooled_count())
            .field("pooled_bytes", &self.pooled_bytes())
            .field("max_bytes", &self.max_bytes)
            .finish()
    }
}

enum BufferStorage {
    CpuPooled(Arc<PooledRaw>),
    CpuExternal(Box<dyn ExternalCpuBuffer>),
    Gpu(Arc<dyn GpuSurface>),
}

struct BufferInner {
    storage: BufferStorage,
    logical_len: usize,
}

/// Reference-counted, 64-byte aligned, zero-copy-sliceable media buffer supporting
/// pooled CPU memory, zero-copy external CPU memory, and GPU surfaces.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
    offset: usize,
    len: usize,
}

impl Buffer {
    /// Allocate a standalone 64-byte aligned zeroed buffer of `len` bytes (single allocation).
    pub fn new(len: usize) -> Self {
        let raw = PooledRaw::new(AlignedRaw::new_zeroed(len), None);
        Self::from_pooled_raw(raw, len)
    }

    /// Allocate a standalone 64-byte aligned buffer initialized from `data`.
    pub fn from_slice(data: &[u8]) -> Self {
        let mut raw = AlignedRaw::new_zeroed(data.len());
        raw.as_mut_slice()[..data.len()].copy_from_slice(data);
        Self::from_pooled_raw(PooledRaw::new(raw, None), data.len())
    }

    /// Wrap an externally owned CPU buffer with zero copies.
    pub fn from_external(ext: Box<dyn ExternalCpuBuffer>) -> Self {
        let len = ext.as_slice().len();
        Self {
            inner: Arc::new(BufferInner {
                storage: BufferStorage::CpuExternal(ext),
                logical_len: len,
            }),
            offset: 0,
            len,
        }
    }

    /// Wrap a GPU surface with zero copies.
    pub fn from_gpu(surface: Arc<dyn GpuSurface>) -> Self {
        Self {
            inner: Arc::new(BufferInner {
                storage: BufferStorage::Gpu(surface),
                logical_len: 0,
            }),
            offset: 0,
            len: 0,
        }
    }

    pub(crate) fn from_pooled_raw(raw: Arc<PooledRaw>, logical_len: usize) -> Self {
        debug_assert!(raw.capacity() >= logical_len + SIMD_TAIL_PADDING);
        Self {
            inner: Arc::new(BufferInner {
                storage: BufferStorage::CpuPooled(raw),
                logical_len,
            }),
            offset: 0,
            len: logical_len,
        }
    }

    /// Logical byte length (`0` for GPU surfaces).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Memory residency domain (`Cpu` or `Gpu(device_id)`).
    #[inline]
    pub fn domain(&self) -> MemoryDomain {
        match &self.inner.storage {
            BufferStorage::CpuPooled(_) | BufferStorage::CpuExternal(_) => MemoryDomain::Cpu,
            BufferStorage::Gpu(surface) => MemoryDomain::Gpu(surface.device_id()),
        }
    }

    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self.inner.storage, BufferStorage::Gpu(_))
    }

    /// Access the underlying `GpuSurface` if this buffer is GPU-resident.
    #[inline]
    pub fn gpu_surface(&self) -> Option<&Arc<dyn GpuSurface>> {
        match &self.inner.storage {
            BufferStorage::Gpu(surface) => Some(surface),
            _ => None,
        }
    }

    /// Access the logical CPU byte slice, returning `None` for GPU-resident buffers.
    #[inline]
    pub fn as_slice(&self) -> Option<&[u8]> {
        match &self.inner.storage {
            BufferStorage::CpuPooled(raw) => {
                Some(&raw.as_slice()[self.offset..self.offset + self.len])
            }
            BufferStorage::CpuExternal(ext) => {
                Some(&ext.as_slice()[self.offset..self.offset + self.len])
            }
            BufferStorage::Gpu(_) => None,
        }
    }

    /// Access the logical CPU bytes (returns `&[]` for GPU buffers).
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.as_slice().unwrap_or(&[])
    }

    /// Access CPU bytes plus up to `SIMD_TAIL_PADDING` readable trailing bytes.
    /// For `CpuPooled`, always returns `self.len() + SIMD_TAIL_PADDING` bytes;
    /// for `CpuExternal`, returns whatever padding the external buffer guarantees.
    #[inline]
    pub fn data_with_padding(&self) -> &[u8] {
        match &self.inner.storage {
            BufferStorage::CpuPooled(raw) => {
                &raw.as_slice()[self.offset..self.offset + self.len + SIMD_TAIL_PADDING]
            }
            BufferStorage::CpuExternal(ext) => {
                let padded = ext.as_padded_slice();
                let end = (self.offset + self.len + SIMD_TAIL_PADDING).min(padded.len());
                &padded[self.offset..end]
            }
            BufferStorage::Gpu(_) => &[],
        }
    }

    /// Zero-copy checked subslice view sharing the same backing `Arc`.
    /// Returns `None` if `range` is out of bounds or if this is a GPU buffer.
    pub fn subslice(&self, range: Range<usize>) -> Option<Self> {
        if self.is_gpu() || range.start > range.end || range.end > self.len {
            return None;
        }
        Some(Self {
            inner: Arc::clone(&self.inner),
            offset: self.offset + range.start,
            len: range.end - range.start,
        })
    }

    /// Returns `true` if this `Buffer` holds unique mutable ownership over its CPU slab.
    #[inline]
    pub fn is_writable(&self) -> bool {
        if Arc::strong_count(&self.inner) != 1 {
            return false;
        }
        match &self.inner.storage {
            BufferStorage::CpuPooled(raw) => Arc::strong_count(raw) == 1,
            BufferStorage::CpuExternal(_) | BufferStorage::Gpu(_) => false,
        }
    }

    /// Obtain a mutable slice if and only if this buffer is uniquely owned CPU memory.
    /// Never panics on shared, external, or GPU buffers.
    pub fn try_data_mut(&mut self) -> Option<&mut [u8]> {
        let offset = self.offset;
        let len = self.len;
        let inner = Arc::get_mut(&mut self.inner)?;
        match &mut inner.storage {
            BufferStorage::CpuPooled(raw) => {
                let raw = Arc::get_mut(raw)?;
                Some(&mut raw.as_mut_slice()[offset..offset + len])
            }
            BufferStorage::CpuExternal(_) | BufferStorage::Gpu(_) => None,
        }
    }

    /// Ensure this CPU buffer is uniquely writable (Copy-on-Write allocated from `pool`),
    /// returning its mutable slice. Returns `None` for GPU buffers.
    pub fn make_writable(&mut self, pool: &Arc<BufferPool>) -> Option<&mut [u8]> {
        if self.is_gpu() {
            return None;
        }
        if !self.is_writable() {
            let copy = pool.acquire_from_slice(self.data());
            *self = copy;
        }
        self.try_data_mut()
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("domain", &self.domain())
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("logical_backing_len", &self.inner.logical_len)
            .finish()
    }
}

/// 2D border padding around active pixels (in bytes horizontally, rows vertically).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct BorderPadding {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
}

impl BorderPadding {
    pub const NONE: Self = Self {
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
    };

    /// Uniform border of `n` rows top/bottom and `n` bytes left/right.
    #[inline]
    pub const fn uniform(n: usize) -> Self {
        Self {
            top: n,
            bottom: n,
            left: n,
            right: n,
        }
    }
}

/// 2D CPU video plane with explicit border padding, aligned stride, and `origin_offset`.
///
/// Active pixel `(0, 0)` starts at `origin_offset`. When `border` is populated via
/// `extend_edges()`, motion-compensation filters can read negative offsets cleanly
/// within a single safe `&[u8]` slice with zero edge fallbacks or scratch allocations.
#[derive(Debug, Clone)]
pub struct BorderedPlane {
    pub buffer: Buffer,
    pub origin_offset: usize,
    pub stride: isize,
    pub width_bytes: usize,
    pub height: usize,
    pub border: BorderPadding,
}

impl BorderedPlane {
    /// Allocate a zero-initialized bordered plane from `pool` with a 64-byte aligned stride.
    pub fn new(
        pool: &Arc<BufferPool>,
        width_bytes: usize,
        height: usize,
        border: BorderPadding,
    ) -> Self {
        Self::alloc_in(pool, width_bytes, height, border, true)
    }

    /// Allocate a reused (non-zeroed body, zeroed tail) bordered plane from `pool`.
    pub fn new_reused(
        pool: &Arc<BufferPool>,
        width_bytes: usize,
        height: usize,
        border: BorderPadding,
    ) -> Self {
        Self::alloc_in(pool, width_bytes, height, border, false)
    }

    fn alloc_in(
        pool: &Arc<BufferPool>,
        width_bytes: usize,
        height: usize,
        border: BorderPadding,
        zero_body: bool,
    ) -> Self {
        let min_stride = border
            .left
            .checked_add(width_bytes)
            .and_then(|v| v.checked_add(border.right))
            .expect("plane stride overflow");
        let stride = align_up(min_stride, SIMD_ALIGN).max(SIMD_ALIGN);
        let total_rows = border
            .top
            .checked_add(height)
            .and_then(|v| v.checked_add(border.bottom))
            .expect("plane height overflow");
        let total_bytes = stride.checked_mul(total_rows).expect("plane size overflow");
        let origin_offset = border.top * stride + border.left;
        let buffer = if zero_body {
            pool.acquire(total_bytes)
        } else {
            pool.acquire_reused(total_bytes)
        };
        Self {
            buffer,
            origin_offset,
            stride: stride as isize,
            width_bytes,
            height,
            border,
        }
    }

    /// Wrap an existing compact/strided `Buffer` without border padding.
    pub fn from_buffer(
        buffer: Buffer,
        origin_offset: usize,
        stride: isize,
        width_bytes: usize,
        height: usize,
    ) -> Self {
        Self {
            buffer,
            origin_offset,
            stride,
            width_bytes,
            height,
            border: BorderPadding::NONE,
        }
    }

    /// Byte offset of active row `y` inside `self.buffer.data()`.
    #[inline]
    pub fn row_offset(&self, y: usize) -> Option<usize> {
        if y >= self.height {
            return None;
        }
        let off =
            (self.origin_offset as isize).checked_add((y as isize).checked_mul(self.stride)?)?;
        if off < 0 {
            return None;
        }
        let off = off as usize;
        if off.checked_add(self.width_bytes)? <= self.buffer.len() {
            Some(off)
        } else {
            None
        }
    }

    /// Immutable slice of active pixels for row `y` (excludes border padding).
    #[inline]
    pub fn row(&self, y: usize) -> Option<&[u8]> {
        let off = self.row_offset(y)?;
        let data = self.buffer.as_slice()?;
        Some(&data[off..off + self.width_bytes])
    }

    /// Mutable slice of active pixels for row `y` (requires unique CPU ownership).
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> Option<&mut [u8]> {
        let off = self.row_offset(y)?;
        let width_bytes = self.width_bytes;
        let data = self.buffer.try_data_mut()?;
        Some(&mut data[off..off + width_bytes])
    }

    /// Full backing CPU byte slice including all borders.
    #[inline]
    pub fn padded_slice(&self) -> &[u8] {
        self.buffer.data()
    }

    /// Replicate outermost active pixels into the left, right, top, and bottom borders.
    ///
    /// `pixel_bytes` is the byte width of one sample element (`1` for 8-bit, `2` for 10/12/16-bit).
    /// Returns `false` if the buffer is shared/external/GPU or has non-positive stride.
    pub fn extend_edges(&mut self, pixel_bytes: usize) -> bool {
        if self.height == 0 || self.width_bytes == 0 || pixel_bytes == 0 || self.stride <= 0 {
            return true;
        }
        let stride = self.stride as usize;
        let border = self.border;
        let width_bytes = self.width_bytes;
        let height = self.height;
        let Some(data) = self.buffer.try_data_mut() else {
            return false;
        };

        extend_horizontal_rows(
            data,
            border.top,
            height,
            stride,
            border.left,
            border.right,
            width_bytes,
            pixel_bytes,
        );
        extend_vertical_rows(data, border.top, border.bottom, height, stride);
        true
    }
}

/// Extend left and right borders for `row_count` consecutive rows starting at physical row `start_phys_row`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extend_horizontal_rows(
    data: &mut [u8],
    start_phys_row: usize,
    row_count: usize,
    stride: usize,
    pad_left: usize,
    pad_right: usize,
    width_bytes: usize,
    pixel_bytes: usize,
) {
    if width_bytes < pixel_bytes || (pad_left == 0 && pad_right == 0) {
        return;
    }
    let right_start = pad_left + width_bytes;
    let right_end = right_start + pad_right;

    for r in 0..row_count {
        let row_base = (start_phys_row + r) * stride;
        let row = &mut data[row_base..row_base + right_end];
        if pixel_bytes == 1 {
            let left_val = row[pad_left];
            row[..pad_left].fill(left_val);
            let right_val = row[right_start - 1];
            row[right_start..right_end].fill(right_val);
        } else {
            let mut left_px = [0u8; 8];
            let mut right_px = [0u8; 8];
            let pb = pixel_bytes.min(8);
            left_px[..pb].copy_from_slice(&row[pad_left..pad_left + pb]);
            right_px[..pb].copy_from_slice(&row[right_start - pb..right_start]);
            for chunk in row[..pad_left].chunks_exact_mut(pb) {
                chunk.copy_from_slice(&left_px[..pb]);
            }
            for chunk in row[right_start..right_end].chunks_exact_mut(pb) {
                chunk.copy_from_slice(&right_px[..pb]);
            }
        }
    }
}

/// Extend top and bottom borders from the first and last active rows (which must already have
/// horizontal borders filled).
pub(crate) fn extend_vertical_rows(
    data: &mut [u8],
    pad_top: usize,
    pad_bottom: usize,
    height: usize,
    stride: usize,
) {
    if height == 0 {
        return;
    }
    if pad_top > 0 {
        let src_start = pad_top * stride;
        let (top_area, rest) = data.split_at_mut(src_start);
        let first_row = &rest[..stride];
        for dst_row in top_area.chunks_exact_mut(stride) {
            dst_row.copy_from_slice(first_row);
        }
    }
    if pad_bottom > 0 {
        let last_active_start = (pad_top + height - 1) * stride;
        let bottom_start = (pad_top + height) * stride;
        let (before_bottom, bottom_area) = data.split_at_mut(bottom_start);
        let last_row = &before_bottom[last_active_start..bottom_start];
        for dst_row in bottom_area[..pad_bottom * stride].chunks_exact_mut(stride) {
            dst_row.copy_from_slice(last_row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyExternal(Vec<u8>);
    impl ExternalCpuBuffer for DummyExternal {
        fn as_slice(&self) -> &[u8] {
            &self.0
        }
    }

    #[test]
    fn buffer_alignment_padding_and_pool_reuse() {
        let pool = BufferPool::new();
        let mut buf = pool.acquire(100);
        assert_eq!(buf.len(), 100);
        assert_eq!(buf.data().as_ptr() as usize % SIMD_ALIGN, 0);
        assert_eq!(buf.data_with_padding().len(), 100 + SIMD_TAIL_PADDING);
        assert!(buf.data_with_padding()[100..].iter().all(|&b| b == 0));

        let ptr1 = buf.data().as_ptr();
        buf.try_data_mut().unwrap().fill(0xAB);
        drop(buf);

        assert_eq!(pool.pooled_count(), 1);
        let buf2 = pool.acquire_reused(100);
        assert_eq!(buf2.data().as_ptr(), ptr1); // exact slab reused!
        assert_eq!(pool.pooled_count(), 0);
        // Tail padding must still be zeroed after reuse
        assert!(buf2.data_with_padding()[100..].iter().all(|&b| b == 0));
    }

    #[test]
    fn buffer_subslice_and_cow() {
        let pool = BufferPool::new();
        let buf = pool.acquire_from_slice(&[10, 20, 30, 40, 50]);
        let mut sub = buf.subslice(1..4).unwrap();
        assert_eq!(sub.data(), &[20, 30, 40]);
        assert!(sub.data_with_padding().len() >= 3 + SIMD_TAIL_PADDING);
        assert!(!sub.is_writable());
        assert!(sub.try_data_mut().is_none());

        // CoW copies only the subslice into a writable pooled buffer
        let m = sub.make_writable(&pool).unwrap();
        m[0] = 99;
        assert_eq!(sub.data(), &[99, 30, 40]);
        assert_eq!(buf.data(), &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn external_buffer_zero_copy_and_cow() {
        let pool = BufferPool::new();
        let mut buf = Buffer::from_external(Box::new(DummyExternal(vec![1, 2, 3, 4])));
        assert_eq!(buf.data(), &[1, 2, 3, 4]);
        assert!(!buf.is_writable());
        assert!(buf.try_data_mut().is_none());
        let w = buf.make_writable(&pool).unwrap();
        w[3] = 9;
        assert_eq!(buf.data(), &[1, 2, 3, 9]);
    }

    struct DummyGpu {
        id: GpuDeviceId,
        tag: u32,
    }
    impl GpuSurface for DummyGpu {
        fn device_id(&self) -> GpuDeviceId {
            self.id
        }
        fn layout(&self) -> PixelLayout {
            PixelLayout::RGBA8
        }
        fn width(&self) -> u32 {
            1920
        }
        fn height(&self) -> u32 {
            1080
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn gpu_buffer_domain_and_downcast() {
        let surf: Arc<dyn GpuSurface> = Arc::new(DummyGpu {
            id: GpuDeviceId(42),
            tag: 777,
        });
        let mut buf = Buffer::from_gpu(surf);
        assert!(buf.is_gpu());
        assert_eq!(buf.domain(), MemoryDomain::Gpu(GpuDeviceId(42)));
        assert!(buf.as_slice().is_none());
        assert!(buf.subslice(0..0).is_none());
        assert!(buf.try_data_mut().is_none());
        let pool = BufferPool::new();
        assert!(buf.make_writable(&pool).is_none());

        let downcast = buf
            .gpu_surface()
            .unwrap()
            .as_any()
            .downcast_ref::<DummyGpu>()
            .unwrap();
        assert_eq!(downcast.tag, 777);
    }

    #[test]
    fn bordered_plane_extend_edges_8bit_and_16bit() {
        let pool = BufferPool::new();
        let border = BorderPadding::uniform(4);
        let mut plane = BorderedPlane::new(&pool, 4, 2, border);
        plane.row_mut(0).unwrap().copy_from_slice(&[10, 20, 30, 40]);
        plane.row_mut(1).unwrap().copy_from_slice(&[50, 60, 70, 80]);
        assert!(plane.extend_edges(1));

        let stride = plane.stride as usize;
        let raw = plane.padded_slice();
        // Top border row 0 replicates active row 0 left/body/right
        assert_eq!(&raw[0..4], &[10, 10, 10, 10]);
        assert_eq!(&raw[4..8], &[10, 20, 30, 40]);
        assert_eq!(&raw[8..12], &[40, 40, 40, 40]);
        // Bottom border row replicates active row 1
        let bot = (border.top + 2) * stride;
        assert_eq!(&raw[bot..bot + 4], &[50, 50, 50, 50]);
        assert_eq!(&raw[bot + 4..bot + 8], &[50, 60, 70, 80]);
        assert_eq!(&raw[bot + 8..bot + 12], &[80, 80, 80, 80]);

        // 16-bit (2 bytes/pixel) edge replication preserves little-endian pixel pairs
        let mut p16 = BorderedPlane::new(&pool, 4, 1, BorderPadding::uniform(4));
        p16.row_mut(0)
            .unwrap()
            .copy_from_slice(&[0x01, 0x02, 0x03, 0x04]);
        assert!(p16.extend_edges(2));
        let r16 = p16.padded_slice();
        assert_eq!(&r16[0..4], &[0x01, 0x02, 0x01, 0x02]);
        assert_eq!(&r16[8..12], &[0x03, 0x04, 0x03, 0x04]);
    }
}
