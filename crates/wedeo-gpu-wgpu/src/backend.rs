//! `wgpu` implementation of [`wedeo_video_ops::SwsBackend`] and [`wedeo_video_ops::CompiledSws`].

use std::sync::{Arc, Mutex};

use wedeo_core::{
    BorderPadding, BorderedPlane, BufferPool, CompOrder, CropRect, Endian, Error, MemoryDomain,
    PixelLayout, PlanePacking, PositionedPlane, Rational, Result, SampleDepth, SideData,
    VideoFrame,
};
use wedeo_video_ops::{
    CompiledSws, DisplayTransform, SwsBackend, SwsOp, SwsOpList, ToneMapOp, color, cpu, optimizer,
};

use crate::context::WgpuContext;
use crate::wgsl_gen::{CachedPipeline, SwsUniforms, WORKGROUP_SIZE, WgslOutputBinding};

/// `wgpu` compute shader backend implementing [`wedeo_video_ops::SwsBackend`].
#[derive(Debug, Clone)]
pub struct WgpuSwsBackend {
    ctx: WgpuContext,
}

impl WgpuSwsBackend {
    /// Create a [`WgpuSwsBackend`] backed by `ctx`.
    pub const fn new(ctx: WgpuContext) -> Self {
        Self { ctx }
    }

    /// Reference to the underlying [`WgpuContext`].
    #[inline]
    pub const fn context(&self) -> &WgpuContext {
        &self.ctx
    }
}

impl SwsBackend for WgpuSwsBackend {
    fn compile(&self, ops: &SwsOpList) -> Result<Box<dyn CompiledSws>> {
        let compiled = WgpuCompiledSws::compile(self.ctx.clone(), ops)?;
        Ok(Box::new(compiled))
    }
}

/// Compiled `wgpu` WGSL compute pipeline executing an [`SwsOpList`].
pub struct WgpuCompiledSws {
    ctx: WgpuContext,
    ops: SwsOpList,
    is_noop: bool,
    display_tx: DisplayTransform,
    write_layout: PixelLayout,
    uniforms: SwsUniforms,
    uniform_buffer: wgpu::Buffer,
    cached_pipeline: CachedPipeline,
}

impl WgpuCompiledSws {
    /// Compile an [`SwsOpList`] on `ctx` (reusing any cached pipeline with the same structural hash).
    pub fn compile(ctx: WgpuContext, op_list: &SwsOpList) -> Result<Self> {
        for dom in [op_list.src.domain, op_list.dst.domain] {
            if let MemoryDomain::Gpu(id) = dom
                && id != ctx.device_id()
            {
                return Err(Error::invalid_argument(
                    "wedeo-gpu-wgpu",
                    "VideoContract GPU device_id does not match WgpuContext device_id",
                ));
            }
        }

        let mut ops = op_list.clone();
        optimizer::optimize(&mut ops);
        let is_noop = optimizer::is_noop(&ops);

        let Some(SwsOp::Read { display_matrix, .. }) = ops.ops.first().cloned() else {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "SwsOpList must begin with SwsOp::Read",
            ));
        };
        let Some(SwsOp::Write(write_layout)) = ops.ops.last().cloned() else {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "SwsOpList must end with SwsOp::Write",
            ));
        };

        let display_tx = DisplayTransform::from_matrix(&display_matrix)?;
        let uniforms = SwsUniforms::from_op_list(&ops)?;
        let uniform_bytes = uniforms.to_bytes();

        let uniform_buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wedeo-sws-uniform-buffer"),
            size: uniform_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue().write_buffer(&uniform_buffer, 0, &uniform_bytes);

        let cached_pipeline = ctx.get_or_compile_pipeline(&ops)?;

        Ok(Self {
            ctx,
            ops,
            is_noop,
            display_tx,
            write_layout,
            uniforms,
            uniform_buffer,
            cached_pipeline,
        })
    }

    /// Structural cache key of the compiled WGSL compute pipeline.
    #[inline]
    pub fn structural_hash(&self) -> u64 {
        self.cached_pipeline.structural_hash
    }

    /// Generated WGSL source string for this compiled pipeline.
    #[inline]
    pub fn wgsl_source(&self) -> &str {
        &self.cached_pipeline.wgsl_source
    }

    /// Extracted uniform buffer values for this compiled pipeline.
    #[inline]
    pub const fn uniforms(&self) -> &SwsUniforms {
        &self.uniforms
    }

    fn populate_output_metadata(&self, src: &VideoFrame, out: &mut VideoFrame) {
        out.pts = src.pts;
        out.pkt_dts = src.pkt_dts;
        out.duration = src.duration;
        out.time_base = src.time_base;
        out.picture_type = src.picture_type;
        out.flags = src.flags;
        out.repeat_pict = src.repeat_pict;
        out.sar = if self.display_tx.swaps_axes() && src.sar.num > 0 && src.sar.den > 0 {
            Rational::new(src.sar.den, src.sar.num)
        } else {
            src.sar
        };
        out.crop = CropRect::ZERO;
        out.color = color::sanitize_output_color(self.write_layout, src.color, self.ops.dst.color);

        let tone_mapped = self.ops.src.color.transfer.is_hdr()
            && !out.color.transfer.is_hdr()
            && self.ops.options.tone_map != ToneMapOp::None;
        out.side_data.clear();
        for sd in &src.side_data {
            match sd {
                SideData::DisplayMatrix(_) => {}
                SideData::MasteringDisplay(_)
                | SideData::ContentLightLevel { .. }
                | SideData::Hdr10Plus(_)
                    if tone_mapped => {}
                other => out.side_data.push(other.clone()),
            }
        }
    }

    fn execute_storage_texture(
        &self,
        src_views: &[wgpu::TextureView],
        format: wgpu::TextureFormat,
        src_frame: &VideoFrame,
        dst_pool: &Arc<BufferPool>,
    ) -> Result<VideoFrame> {
        let dst_w = self.ops.dst.width;
        let dst_h = self.ops.dst.height;
        let device = self.ctx.device();

        let dst_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wedeo-sws-dst-texture"),
            size: wgpu::Extent3d {
                width: dst_w,
                height: dst_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let dst_view = dst_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut bg_entries = Vec::with_capacity(2 + src_views.len());
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: self.uniform_buffer.as_entire_binding(),
        });
        for (i, view) in src_views.iter().enumerate() {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        bg_entries.push(wgpu::BindGroupEntry {
            binding: (src_views.len() + 1) as u32,
            resource: wgpu::BindingResource::TextureView(&dst_view),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wedeo-sws-bg"),
            layout: &self.cached_pipeline.bind_group_layout,
            entries: &bg_entries,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wedeo-sws-encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wedeo-sws-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cached_pipeline.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = dst_w.div_ceil(WORKGROUP_SIZE[0]);
            let wg_y = dst_h.div_ceil(WORKGROUP_SIZE[1]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.ctx.queue().submit(Some(encoder.finish()));

        let surface = self.ctx.wrap_texture(dst_texture, self.write_layout);
        let mut gpu_out = VideoFrame::from_gpu(surface);
        self.populate_output_metadata(src_frame, &mut gpu_out);

        if self.ops.dst.domain == MemoryDomain::Cpu {
            self.ctx.download_frame_pooled(&gpu_out, dst_pool)
        } else {
            Ok(gpu_out)
        }
    }

    fn execute_planar_storage(
        &self,
        src_views: &[wgpu::TextureView],
        plane_count: usize,
        src_frame: &VideoFrame,
        dst_pool: &Arc<BufferPool>,
    ) -> Result<VideoFrame> {
        let dst_w = self.ops.dst.width;
        let dst_h = self.ops.dst.height;
        let device = self.ctx.device();

        let mut storage_bufs = Vec::with_capacity(plane_count);
        let mut staging_bufs = Vec::with_capacity(plane_count);
        let mut elem_counts = Vec::with_capacity(plane_count);

        for p in 0..plane_count {
            let elems = plane_u32_element_count(self.write_layout, dst_w, dst_h, p);
            let byte_size = ((elems as u64) * 4).max(4);
            let sbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wedeo-sws-plane-storage"),
                size: byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let rbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wedeo-sws-plane-staging"),
                size: byte_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            storage_bufs.push(sbuf);
            staging_bufs.push(rbuf);
            elem_counts.push(elems);
        }

        let mut bg_entries = Vec::with_capacity(1 + src_views.len() + plane_count);
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: self.uniform_buffer.as_entire_binding(),
        });
        for (i, view) in src_views.iter().enumerate() {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        let dst_base = (src_views.len() + 1) as u32;
        for (p, sbuf) in storage_bufs.iter().enumerate() {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: dst_base + (p as u32),
                resource: sbuf.as_entire_binding(),
            });
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wedeo-sws-planar-bg"),
            layout: &self.cached_pipeline.bind_group_layout,
            entries: &bg_entries,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wedeo-sws-planar-encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wedeo-sws-planar-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cached_pipeline.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = dst_w.div_ceil(WORKGROUP_SIZE[0]);
            let wg_y = dst_h.div_ceil(WORKGROUP_SIZE[1]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        for p in 0..plane_count {
            let byte_size = ((elem_counts[p] as u64) * 4).max(4);
            encoder.copy_buffer_to_buffer(&storage_bufs[p], 0, &staging_bufs[p], 0, byte_size);
        }
        self.ctx.queue().submit(Some(encoder.finish()));

        let map_results: Arc<Mutex<Vec<std::result::Result<(), wgpu::BufferAsyncError>>>> =
            Arc::new(Mutex::new(Vec::with_capacity(plane_count)));
        for rbuf in &staging_bufs {
            let res_clone = Arc::clone(&map_results);
            rbuf.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                res_clone
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(res);
            });
        }
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| {
                Error::internal("wedeo-gpu-wgpu", format!("wgpu device poll error: {e}"))
            })?;

        {
            let locked = map_results.lock().unwrap_or_else(|e| e.into_inner());
            if locked.len() != plane_count || locked.iter().any(|r| r.is_err()) {
                return Err(Error::internal(
                    "wedeo-gpu-wgpu",
                    "failed to map planar storage staging buffers",
                ));
            }
        }

        let mut cpu_planes = Vec::with_capacity(plane_count);
        for (p, rbuf) in staging_bufs.into_iter().enumerate() {
            let (width_bytes, height_rows) = self
                .write_layout
                .plane_dims(dst_w, dst_h, p)
                .ok_or_else(|| Error::internal("wedeo-gpu-wgpu", "invalid output plane index"))?;
            let mut plane =
                BorderedPlane::new_reused(dst_pool, width_bytes, height_rows, BorderPadding::NONE);
            let view = rbuf.slice(..).get_mapped_range();
            unpack_u32_storage_into_plane(&view, self.write_layout, p, &mut plane);
            drop(view);
            rbuf.unmap();
            cpu_planes.push(plane);
        }

        let mut cpu_out = VideoFrame::from_planes(dst_w, dst_h, self.write_layout, cpu_planes);
        self.populate_output_metadata(src_frame, &mut cpu_out);

        if self.ops.dst.domain == MemoryDomain::Cpu {
            Ok(cpu_out)
        } else {
            self.ctx.upload_frame(&cpu_out)
        }
    }
}

impl CompiledSws for WgpuCompiledSws {
    fn execute(&mut self, src: &VideoFrame, dst_pool: &Arc<BufferPool>) -> Result<VideoFrame> {
        if src.width != self.ops.src.width
            || src.height != self.ops.src.height
            || src.layout != self.ops.src.layout
        {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "source VideoFrame dimensions/layout do not match compiled VideoContract",
            ));
        }

        if self.is_noop {
            return match (src.domain(), self.ops.dst.domain) {
                (a, b) if a == b => Ok(src.clone()),
                (MemoryDomain::Cpu, MemoryDomain::Gpu(_)) => self.ctx.upload_frame(src),
                (MemoryDomain::Gpu(_), MemoryDomain::Cpu) => {
                    self.ctx.download_frame_pooled(src, dst_pool)
                }
                _ => Err(Error::invalid_argument(
                    "wedeo-gpu-wgpu",
                    "mismatched GPU device domains in noop transfer",
                )),
            };
        }

        let gpu_src = if src.domain() == MemoryDomain::Cpu {
            self.ctx.upload_frame(src)?
        } else {
            src.clone()
        };

        let surf = gpu_src.gpu_surface().ok_or_else(|| {
            Error::internal("wedeo-gpu-wgpu", "uploaded frame is missing GpuSurface")
        })?;
        let wgpu_surf = WgpuContext::unwrap_surface(surf.as_ref()).ok_or_else(|| {
            Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "source GpuSurface is not WgpuTextureSurface",
            )
        })?;
        if wgpu_surf.planes().len() != self.cached_pipeline.src_plane_count {
            return Err(Error::invalid_argument(
                "wedeo-gpu-wgpu",
                "source WgpuTextureSurface plane count does not match compiled shader",
            ));
        }

        let mut src_views = Vec::with_capacity(wgpu_surf.planes().len());
        for tex in wgpu_surf.planes() {
            src_views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        }

        match self.cached_pipeline.output_binding {
            WgslOutputBinding::StorageTexture(fmt) => {
                self.execute_storage_texture(&src_views, fmt, src, dst_pool)
            }
            WgslOutputBinding::PlanarStorageBuffers { plane_count } => {
                self.execute_planar_storage(&src_views, plane_count, src, dst_pool)
            }
        }
    }

    fn blend_overlay(
        &mut self,
        dst: &mut VideoFrame,
        overlays: &[PositionedPlane],
        dst_pool: &Arc<BufferPool>,
    ) -> Result<()> {
        if overlays.is_empty() {
            return Ok(());
        }
        if dst.domain() == MemoryDomain::Cpu {
            return cpu::blend_overlays_in_place(dst, overlays, dst_pool);
        }
        let mut cpu_frame = self.ctx.download_frame_pooled(dst, dst_pool)?;
        cpu::blend_overlays_in_place(&mut cpu_frame, overlays, dst_pool)?;
        *dst = self.ctx.upload_frame(&cpu_frame)?;
        Ok(())
    }
}

fn plane_u32_element_count(
    layout: PixelLayout,
    width: u32,
    height: u32,
    plane_idx: usize,
) -> usize {
    let w = width as usize;
    let h = height as usize;
    match layout.packing {
        PlanePacking::Packed => match layout.order {
            CompOrder::Yuyv | CompOrder::Uyvy => w.div_ceil(2) * h,
            _ => w * h,
        },
        PlanePacking::SemiPlanar => {
            if plane_idx == 0 || plane_idx == 2 {
                w * h
            } else {
                let (cw, ch) = layout.subsampling.chroma_dims(width, height);
                (cw as usize) * (ch as usize) * 2
            }
        }
        PlanePacking::Planar => {
            if plane_idx == 0 || plane_idx == 3 {
                w * h
            } else {
                let (cw, ch) = layout.subsampling.chroma_dims(width, height);
                (cw as usize) * (ch as usize)
            }
        }
    }
}

fn unpack_u32_storage_into_plane(
    raw_u32_bytes: &[u8],
    layout: PixelLayout,
    plane_idx: usize,
    plane: &mut BorderedPlane,
) {
    let bps = layout.depth.bytes_per_sample();
    let is_packed_4byte = matches!(layout.packing, PlanePacking::Packed)
        && matches!(
            layout.order,
            CompOrder::Bgra
                | CompOrder::Rgba
                | CompOrder::Argb
                | CompOrder::Abgr
                | CompOrder::Yuyv
                | CompOrder::Uyvy
        );

    if is_packed_4byte {
        let words_per_row = plane.width_bytes / 4;
        for y in 0..plane.height {
            let dst_row = plane.row_mut(y).expect("row in bounds");
            let src_start = y * words_per_row * 4;
            dst_row[..words_per_row * 4]
                .copy_from_slice(&raw_u32_bytes[src_start..src_start + words_per_row * 4]);
        }
        return;
    }

    let samples_per_row = plane.width_bytes / bps;
    let _ = plane_idx;
    for y in 0..plane.height {
        let dst_row = plane.row_mut(y).expect("row in bounds");
        let row_word_base = y * samples_per_row;
        for s in 0..samples_per_row {
            let byte_off = (row_word_base + s) * 4;
            let word = u32::from_le_bytes([
                raw_u32_bytes[byte_off],
                raw_u32_bytes[byte_off + 1],
                raw_u32_bytes[byte_off + 2],
                raw_u32_bytes[byte_off + 3],
            ]);
            match layout.depth {
                SampleDepth::U8 => {
                    dst_row[s] = (word.min(255)) as u8;
                }
                SampleDepth::U10 | SampleDepth::U12 | SampleDepth::U14 | SampleDepth::U16 => {
                    let v = (word.min(65535)) as u16;
                    let b = match layout.endian {
                        Endian::Little => v.to_le_bytes(),
                        Endian::Big => v.to_be_bytes(),
                    };
                    dst_row[s * 2] = b[0];
                    dst_row[s * 2 + 1] = b[1];
                }
                SampleDepth::F32 => {
                    let f = f32::from_bits(word);
                    let b = match layout.endian {
                        Endian::Little => f.to_le_bytes(),
                        Endian::Big => f.to_be_bytes(),
                    };
                    dst_row[s * 4..s * 4 + 4].copy_from_slice(&b);
                }
                SampleDepth::F16 => {}
            }
        }
    }
}
