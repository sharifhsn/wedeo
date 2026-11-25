//! `wgpu` device/queue context, zero-copy [`WgpuTextureSurface`] implementing
//! [`wedeo_core::GpuSurface`], and CPU ↔ GPU [`VideoFrame`] transfer helpers.

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use wedeo_core::{
    BorderPadding, BorderedPlane, Buffer, BufferPool, CompOrder, Error, GpuDeviceId, GpuSurface,
    MemoryDomain, PixelLayout, PlanePacking, Result, SampleDepth, VideoFrame,
};
use wedeo_video_ops::SwsOpList;

use crate::backend::WgpuSwsBackend;
use crate::wgsl_gen::{CachedPipeline, compile_cached_pipeline, structural_op_list_hash};

static NEXT_DEVICE_ID: AtomicU64 = AtomicU64::new(1);

/// GPU texture surface backed by one or more `wgpu::Texture` planes.
///
/// Implements [`wedeo_core::GpuSurface`] so GPU-resident [`VideoFrame`]s and [`Buffer`]s
/// can travel through the media graph with zero CPU copies and be unwrapped directly
/// by Tessera or `wedeo-gpu-wgpu`.
#[derive(Debug, Clone)]
pub struct WgpuTextureSurface {
    planes: Vec<wgpu::Texture>,
    layout: PixelLayout,
    width: u32,
    height: u32,
    device_id: GpuDeviceId,
    owner: Option<Arc<ReadbackDevice>>,
}

impl WgpuTextureSurface {
    /// Wrap a single-plane `wgpu::Texture` with the given [`PixelLayout`] and [`GpuDeviceId`].
    pub fn new(texture: wgpu::Texture, layout: PixelLayout, device_id: GpuDeviceId) -> Self {
        let width = texture.width();
        let height = texture.height();
        Self {
            planes: vec![texture],
            layout,
            width,
            height,
            device_id,
            owner: None,
        }
    }

    /// Wrap one or more `wgpu::Texture` planes (e.g. multi-plane Y/U/V or Y/UV textures)
    /// into a single [`WgpuTextureSurface`].
    pub fn from_planes(
        planes: Vec<wgpu::Texture>,
        width: u32,
        height: u32,
        layout: PixelLayout,
        device_id: GpuDeviceId,
    ) -> Result<Self> {
        let expected = layout.plane_count();
        if planes.len() != expected {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                format!(
                    "expected {expected} texture plane(s) for layout {layout}, got {}",
                    planes.len()
                ),
            ));
        }
        if width == 0 || height == 0 {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "surface dimensions must be non-zero",
            ));
        }
        Ok(Self {
            planes,
            layout,
            width,
            height,
            device_id,
            owner: None,
        })
    }

    /// Create a descriptor-only virtual surface (without allocated `wgpu::Texture` handles)
    /// for headless domain-placement and downcast testing when no hardware GPU is present.
    pub fn new_virtual(
        width: u32,
        height: u32,
        layout: PixelLayout,
        device_id: GpuDeviceId,
    ) -> Self {
        Self {
            planes: Vec::new(),
            layout,
            width,
            height,
            device_id,
            owner: None,
        }
    }

    /// Primary `wgpu::Texture` (plane 0), if present.
    #[inline]
    pub fn texture(&self) -> Option<&wgpu::Texture> {
        self.planes.first()
    }

    /// All backing `wgpu::Texture` planes (1 for packed RGB/RGBA/Gray, 2 for NV12/P010, 3 for YUV420p/GBRP).
    #[inline]
    pub fn planes(&self) -> &[wgpu::Texture] {
        &self.planes
    }

    /// Access plane `index`'s `wgpu::Texture`.
    #[inline]
    pub fn plane(&self, index: usize) -> Option<&wgpu::Texture> {
        self.planes.get(index)
    }
}

impl GpuSurface for WgpuTextureSurface {
    #[inline]
    fn device_id(&self) -> GpuDeviceId {
        self.device_id
    }

    #[inline]
    fn layout(&self) -> PixelLayout {
        self.layout
    }

    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }

    fn readback_planes(&self) -> Result<Vec<BorderedPlane>> {
        self.owner
            .as_ref()
            .ok_or_else(|| {
                Error::unsupported(
                    "wedeo-gpu-wgpu",
                    "wrap the texture with its WgpuContext to enable readback",
                )
            })?
            .read_planes(self, None)
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct ReadbackDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

/// Shared `wgpu` execution context holding a device, command queue, unique [`GpuDeviceId`],
/// and a thread-safe structural WGSL compute pipeline cache.
#[derive(Debug, Clone)]
pub struct WgpuContext {
    owner: Arc<ReadbackDevice>,
    device_id: GpuDeviceId,
    pipeline_cache: Arc<RwLock<HashMap<u64, CachedPipeline>>>,
}

impl WgpuContext {
    /// Construct a [`WgpuContext`] from an externally managed `wgpu::Device` and `wgpu::Queue`
    /// (e.g. sharing Tessera's render device for zero-copy texture handoff).
    pub fn from_external(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let id = NEXT_DEVICE_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            owner: Arc::new(ReadbackDevice { device, queue }),
            device_id: GpuDeviceId(id),
            pipeline_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a standalone headless [`WgpuContext`] using `pollster::block_on` with automatic
    /// hardware-to-fallback adapter selection for CLI and background transcoding.
    pub fn new_headless() -> Result<Self> {
        pollster::block_on(Self::new_headless_async())
    }

    async fn new_headless_async() -> Result<Self> {
        // Try primary backends (Vulkan / Metal / DX12) first to avoid slow headless GL/EGL probing;
        // fall back to all enabled backends if no primary adapter is found.
        let primary_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let mut adapter = Self::select_adapter(&primary_instance).await;
        if adapter.is_none() {
            let all_instance = wgpu::Instance::default();
            adapter = Self::select_adapter(&all_instance).await;
        }

        let Some(adapter) = adapter else {
            return Err(Error::unsupported(
                "wedeo-gpu-wgpu",
                "no compatible wgpu GPU or software fallback adapter found on this system",
            ));
        };

        let mut required_features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::BGRA8UNORM_STORAGE)
        {
            required_features |= wgpu::Features::BGRA8UNORM_STORAGE;
        }

        let desc = wgpu::DeviceDescriptor {
            label: Some("wedeo-gpu-wgpu-headless"),
            required_features,
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            ..wgpu::DeviceDescriptor::default()
        };

        let (device, queue) = adapter.request_device(&desc).await.map_err(|e| {
            Error::internal(
                "wedeo-gpu-wgpu",
                format!("failed to request wgpu device: {e}"),
            )
        })?;

        Ok(Self::from_external(Arc::new(device), Arc::new(queue)))
    }

    async fn select_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
        for (power_preference, force_fallback_adapter) in [
            (wgpu::PowerPreference::HighPerformance, false),
            (wgpu::PowerPreference::LowPower, false),
            (wgpu::PowerPreference::None, true),
        ] {
            if let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference,
                    compatible_surface: None,
                    force_fallback_adapter,
                })
                .await
            {
                return Some(adapter);
            }
        }
        let mut all = instance.enumerate_adapters(wgpu::Backends::all()).await;
        if !all.is_empty() {
            Some(all.swap_remove(0))
        } else {
            None
        }
    }

    /// Unique opaque identifier for this context's GPU device.
    #[inline]
    pub fn device_id(&self) -> GpuDeviceId {
        self.device_id
    }

    /// Reference to the underlying `Arc<wgpu::Device>`.
    #[inline]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.owner.device
    }

    /// Reference to the underlying `Arc<wgpu::Queue>`.
    #[inline]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.owner.queue
    }

    /// Build a [`WgpuSwsBackend`] sharing this context and its WGSL pipeline cache.
    #[inline]
    pub fn sws_backend(&self) -> WgpuSwsBackend {
        WgpuSwsBackend::new(self.clone())
    }

    /// Access the thread-safe structural WGSL compute pipeline cache.
    #[inline]
    pub fn pipeline_cache(&self) -> &Arc<RwLock<HashMap<u64, CachedPipeline>>> {
        &self.pipeline_cache
    }

    /// Number of compiled compute pipelines currently stored in the cache.
    pub fn cached_pipeline_count(&self) -> usize {
        self.pipeline_cache
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Look up or compile a [`CachedPipeline`] for `ops` keyed by its structural hash.
    pub fn get_or_compile_pipeline(&self, ops: &SwsOpList) -> Result<CachedPipeline> {
        let key = structural_op_list_hash(ops);
        {
            let cache = self
                .pipeline_cache
                .read()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = cache.get(&key) {
                return Ok(hit.clone());
            }
        }

        let compiled = compile_cached_pipeline(&self.owner.device, ops)?;
        let mut cache = self
            .pipeline_cache
            .write()
            .unwrap_or_else(|e| e.into_inner());
        let entry = cache.entry(key).or_insert_with(|| compiled.clone());
        Ok(entry.clone())
    }

    /// Wrap a single `wgpu::Texture` into a [`GpuSurface`] associated with this context's device.
    pub fn wrap_texture(&self, texture: wgpu::Texture, layout: PixelLayout) -> Arc<dyn GpuSurface> {
        let mut surface = WgpuTextureSurface::new(texture, layout, self.device_id);
        surface.owner = Some(self.owner.clone());
        Arc::new(surface)
    }

    /// Wrap multiple planar/semi-planar `wgpu::Texture`s into a [`GpuSurface`] associated with this context's device.
    pub fn wrap_planes(
        &self,
        planes: Vec<wgpu::Texture>,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<Arc<dyn GpuSurface>> {
        let mut surf =
            WgpuTextureSurface::from_planes(planes, width, height, layout, self.device_id)?;
        surf.owner = Some(self.owner.clone());
        Ok(Arc::new(surf))
    }

    /// Downcast a [`GpuSurface`] to extract its primary `&wgpu::Texture` (zero-copy Tessera handoff).
    pub fn unwrap_texture(surface: &dyn GpuSurface) -> Option<&wgpu::Texture> {
        surface
            .as_any()
            .downcast_ref::<WgpuTextureSurface>()?
            .texture()
    }

    /// Downcast a [`GpuSurface`] to the concrete [`WgpuTextureSurface`].
    pub fn unwrap_surface(surface: &dyn GpuSurface) -> Option<&WgpuTextureSurface> {
        surface.as_any().downcast_ref::<WgpuTextureSurface>()
    }

    /// Upload a CPU [`VideoFrame`] into GPU `wgpu::Texture` plane(s) on this device.
    ///
    /// If `frame` already resides on `MemoryDomain::Gpu(self.device_id)`, returns a zero-copy clone.
    pub fn upload_frame(&self, frame: &VideoFrame) -> Result<VideoFrame> {
        match frame.domain() {
            MemoryDomain::Gpu(id) if id == self.device_id => return Ok(frame.clone()),
            MemoryDomain::Gpu(_) => {
                return Err(Error::invalid_argument(
                    "wedeo-gpu-wgpu",
                    "cannot upload a GPU frame belonging to a different GPU device",
                ));
            }
            MemoryDomain::Cpu => {}
        }

        let cpu_planes = frame.planes().ok_or_else(|| {
            Error::invalid_argument("wedeo-gpu-wgpu", "CPU VideoFrame is missing planes")
        })?;
        let expected_planes = frame.layout.plane_count();
        if cpu_planes.len() != expected_planes {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                format!(
                    "expected {expected_planes} CPU plane(s) for {}, got {}",
                    frame.layout,
                    cpu_planes.len()
                ),
            ));
        }

        let mut textures = Vec::with_capacity(expected_planes);
        for (p, plane) in cpu_planes.iter().enumerate() {
            let format = plane_texture_format(frame.layout, p)?;
            let (tex_w, tex_h, bpt) =
                plane_texel_geometry(frame.layout, frame.width, frame.height, p)?;
            if tex_w == 0 || tex_h == 0 {
                return Err(Error::invalid_data(
                    "wedeo-gpu-wgpu",
                    "empty texture extent",
                ));
            }
            let row_bytes = (tex_w as usize) * (bpt as usize);
            if plane.width_bytes < row_bytes || plane.height < tex_h as usize {
                return Err(Error::invalid_data(
                    "wedeo-gpu-wgpu",
                    "CPU plane dimensions are smaller than required texture extent",
                ));
            }

            let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC;
            if supports_storage_binding(format, self.owner.device.features()) {
                usage |= wgpu::TextureUsages::STORAGE_BINDING;
            }

            let texture = self.owner.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("wedeo-video-plane"),
                size: wgpu::Extent3d {
                    width: tex_w,
                    height: tex_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });

            // Queue writes accept a source stride; only inverted or incompatible rows need packing.
            let pitch = u32::try_from(plane.stride)
                .ok()
                .filter(|&n| n as usize >= row_bytes && n % bpt == 0);
            let (bytes, pitch) = if let Some(pitch) = pitch {
                let last = plane.row_offset(tex_h as usize - 1).ok_or_else(|| {
                    Error::invalid_data("wedeo-gpu-wgpu", "CPU plane row out of bounds")
                })?;
                let data = plane
                    .buffer
                    .as_slice()
                    .and_then(|s| s.get(plane.origin_offset..last.checked_add(row_bytes)?))
                    .ok_or_else(|| {
                        Error::invalid_data("wedeo-gpu-wgpu", "CPU plane is inaccessible")
                    })?;
                (std::borrow::Cow::Borrowed(data), pitch)
            } else {
                let mut packed = Vec::with_capacity(row_bytes * tex_h as usize);
                for y in 0..tex_h as usize {
                    let row = plane.row(y).ok_or_else(|| {
                        Error::invalid_data("wedeo-gpu-wgpu", "CPU plane row out of bounds")
                    })?;
                    packed.extend_from_slice(&row[..row_bytes]);
                }
                (std::borrow::Cow::Owned(packed), row_bytes as u32)
            };

            self.owner.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(tex_h),
                },
                wgpu::Extent3d {
                    width: tex_w,
                    height: tex_h,
                    depth_or_array_layers: 1,
                },
            );

            textures.push(texture);
        }

        self.owner.queue.submit([]);

        let surface = self.wrap_planes(textures, frame.width, frame.height, frame.layout)?;
        let mut out = VideoFrame::from_gpu(surface);
        copy_frame_metadata(frame, &mut out);
        Ok(out)
    }

    /// Download a GPU-resident [`VideoFrame`] back into CPU [`BorderedPlane`]s.
    ///
    /// If `frame` already resides in [`MemoryDomain::Cpu`], returns a zero-copy clone.
    pub fn download_frame(&self, frame: &VideoFrame, pool: &BufferPool) -> Result<VideoFrame> {
        let _ = pool.pooled_bytes();
        self.download_frame_impl(frame, None)
    }

    /// Download a GPU-resident [`VideoFrame`] into pooled CPU [`BorderedPlane`]s acquired from `pool`.
    pub fn download_frame_pooled(
        &self,
        frame: &VideoFrame,
        pool: &Arc<BufferPool>,
    ) -> Result<VideoFrame> {
        self.download_frame_impl(frame, Some(pool))
    }

    fn download_frame_impl(
        &self,
        frame: &VideoFrame,
        pool: Option<&Arc<BufferPool>>,
    ) -> Result<VideoFrame> {
        if frame.domain() == MemoryDomain::Cpu {
            return Ok(frame.clone());
        }
        let surf = frame.gpu_surface().ok_or_else(|| {
            Error::invalid_argument("wedeo-gpu-wgpu", "expected GPU-resident VideoFrame")
        })?;
        if surf.device_id() != self.device_id {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "GPU VideoFrame belongs to a different GPU device",
            ));
        }
        let wgpu_surf = Self::unwrap_surface(surf.as_ref()).ok_or_else(|| {
            Error::invalid_argument("wedeo-gpu-wgpu", "GpuSurface is not a WgpuTextureSurface")
        })?;
        let planes = self.owner.read_planes(wgpu_surf, pool)?;
        let mut out = frame.clone();
        out.storage = wedeo_core::VideoStorage::Cpu(planes);
        Ok(out)
    }
}

impl ReadbackDevice {
    fn read_planes(
        &self,
        surface: &WgpuTextureSurface,
        pool: Option<&Arc<BufferPool>>,
    ) -> Result<Vec<BorderedPlane>> {
        if surface.planes().len() != surface.layout.plane_count() {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "missing texture planes for readback",
            ));
        }
        let plane_count = surface.planes().len();
        let mut staging_buffers = Vec::with_capacity(plane_count);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wedeo-download-frame"),
            });

        for (p, tex) in surface.planes().iter().enumerate() {
            let (tex_w, tex_h, bpt) =
                plane_texel_geometry(surface.layout, surface.width, surface.height, p)?;
            if !tex.usage().contains(wgpu::TextureUsages::COPY_SRC)
                || tex.sample_count() != 1
                || tex.width() < tex_w
                || tex.height() < tex_h
                || tex.format() != plane_texture_format(surface.layout, p)?
            {
                return Err(Error::invalid_argument(
                    "wedeo-gpu-wgpu",
                    "texture plane does not support readback with the declared layout",
                ));
            }
            let row_bytes = tex_w * bpt;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_row_bytes = row_bytes.div_ceil(align) * align;
            let buf_size = (padded_row_bytes as u64) * (tex_h as u64);

            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wedeo-download-staging"),
                size: buf_size.max(4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row_bytes),
                        rows_per_image: Some(tex_h),
                    },
                },
                wgpu::Extent3d {
                    width: tex_w,
                    height: tex_h,
                    depth_or_array_layers: 1,
                },
            );

            staging_buffers.push((
                staging,
                row_bytes as usize,
                padded_row_bytes as usize,
                tex_h as usize,
            ));
        }

        self.queue.submit(Some(encoder.finish()));

        // Map all staging buffers and poll device until completion.
        let map_results: Arc<
            std::sync::Mutex<Vec<std::result::Result<(), wgpu::BufferAsyncError>>>,
        > = Arc::new(std::sync::Mutex::new(Vec::with_capacity(plane_count)));
        for (staging, _, _, _) in &staging_buffers {
            let res_clone = Arc::clone(&map_results);
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |res| {
                    res_clone
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(res);
                });
        }

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| {
                Error::internal("wedeo-gpu-wgpu", format!("wgpu device poll error: {e}"))
            })?;

        {
            let locked = map_results.lock().unwrap_or_else(|e| e.into_inner());
            if locked.len() != plane_count || locked.iter().any(|r| r.is_err()) {
                return Err(Error::internal(
                    "wedeo-gpu-wgpu",
                    "failed to map wgpu staging buffer for frame download",
                ));
            }
        }

        let mut cpu_planes = Vec::with_capacity(plane_count);
        for (staging, row_bytes, padded_row_bytes, tex_h) in staging_buffers {
            let view = staging.slice(..).get_mapped_range();
            let mut plane = alloc_cpu_plane(pool, row_bytes, tex_h);
            for y in 0..tex_h {
                let src_off = y * padded_row_bytes;
                let dst_row = plane.row_mut(y).ok_or_else(|| {
                    Error::internal(
                        "wedeo-gpu-wgpu",
                        "failed to access destination CPU plane row",
                    )
                })?;
                dst_row.copy_from_slice(&view[src_off..src_off + row_bytes]);
            }
            drop(view);
            staging.unmap();
            cpu_planes.push(plane);
        }

        Ok(cpu_planes)
    }
}

pub(crate) fn alloc_cpu_plane(
    pool: Option<&Arc<BufferPool>>,
    width_bytes: usize,
    height: usize,
) -> BorderedPlane {
    if let Some(p) = pool {
        BorderedPlane::new_reused(p, width_bytes, height, BorderPadding::NONE)
    } else {
        let stride = width_bytes.div_ceil(wedeo_core::SIMD_ALIGN).max(1) * wedeo_core::SIMD_ALIGN;
        let buf = Buffer::new(stride * height);
        BorderedPlane::from_buffer(buf, 0, stride as isize, width_bytes, height)
    }
}

pub(crate) fn copy_frame_metadata(src: &VideoFrame, dst: &mut VideoFrame) {
    dst.color = src.color;
    dst.crop = src.crop;
    dst.sar = src.sar;
    dst.picture_type = src.picture_type;
    dst.flags = src.flags;
    dst.repeat_pict = src.repeat_pict;
    dst.pts = src.pts;
    dst.pkt_dts = src.pkt_dts;
    dst.duration = src.duration;
    dst.time_base = src.time_base;
    dst.side_data.clone_from(&src.side_data);
}

/// Returns `true` if `format` can be created with `wgpu::TextureUsages::STORAGE_BINDING`
/// under the given `device_features`.
pub fn supports_storage_binding(
    format: wgpu::TextureFormat,
    device_features: wgpu::Features,
) -> bool {
    match format {
        wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8Uint
        | wgpu::TextureFormat::Rgba16Uint
        | wgpu::TextureFormat::Rgba32Float
        | wgpu::TextureFormat::R32Uint
        | wgpu::TextureFormat::R32Float
        | wgpu::TextureFormat::Rg32Uint
        | wgpu::TextureFormat::Rg32Float => true,
        wgpu::TextureFormat::Bgra8Unorm => {
            device_features.contains(wgpu::Features::BGRA8UNORM_STORAGE)
        }
        _ => false,
    }
}

/// Map a `(PixelLayout, plane_idx)` pair to the corresponding `wgpu::TextureFormat`.
pub fn plane_texture_format(layout: PixelLayout, plane_idx: usize) -> Result<wgpu::TextureFormat> {
    if plane_idx >= layout.plane_count() {
        return Err(Error::invalid_argument(
            "wedeo-gpu-wgpu",
            "plane index out of bounds for PixelLayout",
        ));
    }

    match layout.packing {
        PlanePacking::Packed => match (layout.order, layout.depth) {
            (CompOrder::Rgba | CompOrder::Argb | CompOrder::Abgr, SampleDepth::U8) => {
                Ok(wgpu::TextureFormat::Rgba8Unorm)
            }
            (CompOrder::Bgra, SampleDepth::U8) => Ok(wgpu::TextureFormat::Bgra8Unorm),
            (
                CompOrder::Rgba | CompOrder::Bgra | CompOrder::Argb | CompOrder::Abgr,
                SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16,
            ) => Ok(wgpu::TextureFormat::Rgba16Uint),
            (CompOrder::Rgba, SampleDepth::F32) => Ok(wgpu::TextureFormat::Rgba32Float),
            (CompOrder::Gray, SampleDepth::U8) => Ok(wgpu::TextureFormat::R8Unorm),
            (
                CompOrder::Gray,
                SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16,
            ) => Ok(wgpu::TextureFormat::R16Uint),
            (CompOrder::Yuyv | CompOrder::Uyvy, SampleDepth::U8) => {
                Ok(wgpu::TextureFormat::Rgba8Unorm)
            }
            _ => Err(Error::unsupported(
                "wedeo-gpu-wgpu",
                format!("packed layout {layout} has no direct wgpu::TextureFormat"),
            )),
        },
        PlanePacking::SemiPlanar => {
            if plane_idx == 0 || plane_idx == 2 {
                match layout.depth {
                    SampleDepth::U8 => Ok(wgpu::TextureFormat::R8Unorm),
                    SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16 => {
                        Ok(wgpu::TextureFormat::R16Uint)
                    }
                    _ => Err(Error::unsupported(
                        "wedeo-gpu-wgpu",
                        format!("unsupported semi-planar sample depth {:?}", layout.depth),
                    )),
                }
            } else {
                match layout.depth {
                    SampleDepth::U8 => Ok(wgpu::TextureFormat::Rg8Unorm),
                    SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16 => {
                        Ok(wgpu::TextureFormat::Rg16Uint)
                    }
                    _ => Err(Error::unsupported(
                        "wedeo-gpu-wgpu",
                        format!("unsupported semi-planar chroma depth {:?}", layout.depth),
                    )),
                }
            }
        }
        PlanePacking::Planar => match layout.depth {
            SampleDepth::U8 => Ok(wgpu::TextureFormat::R8Unorm),
            SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16 => {
                Ok(wgpu::TextureFormat::R16Uint)
            }
            SampleDepth::F32 => Ok(wgpu::TextureFormat::R32Float),
            SampleDepth::F16 => Err(Error::unsupported(
                "wedeo-gpu-wgpu",
                "F16 planar textures are not yet supported",
            )),
        },
    }
}

/// Compute `(tex_width_texels, tex_height_texels, bytes_per_texel)` for `plane_idx` of `layout`.
pub fn plane_texel_geometry(
    layout: PixelLayout,
    width: u32,
    height: u32,
    plane_idx: usize,
) -> Result<(u32, u32, u32)> {
    let _ = plane_texture_format(layout, plane_idx)?;
    let bps = layout.depth.bytes_per_sample() as u32;
    match layout.packing {
        PlanePacking::Packed => match layout.order {
            CompOrder::Rgba | CompOrder::Bgra | CompOrder::Argb | CompOrder::Abgr => {
                Ok((width, height, 4 * bps))
            }
            CompOrder::Gray => Ok((width, height, bps)),
            CompOrder::Yuyv | CompOrder::Uyvy => Ok((width.div_ceil(2), height, 4 * bps)),
            _ => Err(Error::unsupported(
                "wedeo-gpu-wgpu",
                format!("unsupported packed layout {layout}"),
            )),
        },
        PlanePacking::SemiPlanar => {
            if plane_idx == 0 || plane_idx == 2 {
                Ok((width, height, bps))
            } else {
                let (cw, ch) = layout.subsampling.chroma_dims(width, height);
                Ok((cw, ch, 2 * bps))
            }
        }
        PlanePacking::Planar => {
            if plane_idx == 0 || plane_idx == 3 {
                Ok((width, height, bps))
            } else {
                let (cw, ch) = layout.subsampling.chroma_dims(width, height);
                Ok((cw, ch, bps))
            }
        }
    }
}
