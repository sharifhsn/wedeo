//! First-class ergonomic custom filter framework (`VideoFilter`, `AudioFilter`, `WgslFilter`,
//! `SubtitleRasterizer`) and 11 built-in VFR-native essential filters.

use std::collections::VecDeque;
use std::sync::Arc;

use wedeo_audio_ops::AudioContract;
use wedeo_core::{
    AudioBlock, BorderPadding, BorderedPlane, Buffer, BufferPool, ColorModel, ColorRange,
    CompOrder, Error, FrameFlags, MemoryDomain, PixelLayout, PlanePacking, PositionedPlane,
    Rational, Result, Rounding, SampleDepth, SampleFormat, SubtitleEvent, SubtitlePayload,
    Timestamp, VideoFrame, rescale_q, rescale_q_rnd,
};
use wedeo_video_ops::{CompiledSws, CpuCompiledSws, SwsOpList, SwsOptions, VideoContract};

use crate::timeline::ClipRange;

#[inline]
fn unpack_sample_f64(fmt: SampleFormat, bytes: &[u8]) -> f64 {
    match fmt.packed() {
        SampleFormat::U8 => (bytes[0] as f64 - 128.0) / 128.0,
        SampleFormat::S16 => i16::from_ne_bytes([bytes[0], bytes[1]]) as f64 / 32768.0,
        SampleFormat::S32 => {
            i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64 / 2_147_483_648.0
        }
        SampleFormat::S64 => {
            let v = i64::from_ne_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
            v as f64 / 9_223_372_036_854_775_808.0
        }
        SampleFormat::F32 => f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        SampleFormat::F64 => f64::from_ne_bytes(bytes[..8].try_into().unwrap_or([0; 8])),
        _ => 0.0,
    }
}

#[inline]
fn pack_sample_f64(fmt: SampleFormat, val: f64, dst: &mut [u8]) {
    match fmt.packed() {
        SampleFormat::U8 => {
            let s = (val * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8;
            dst[0] = s;
        }
        SampleFormat::S16 => {
            let s = (val * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
            dst[..2].copy_from_slice(&s.to_ne_bytes());
        }
        SampleFormat::S32 => {
            let s = (val * 2_147_483_648.0)
                .round()
                .clamp(-2_147_483_648.0, 2_147_483_647.0) as i32;
            dst[..4].copy_from_slice(&s.to_ne_bytes());
        }
        SampleFormat::S64 => {
            let s = (val.clamp(-1.0, 1.0) * 9_223_372_036_854_775_807.0).round() as i64;
            dst[..8].copy_from_slice(&s.to_ne_bytes());
        }
        SampleFormat::F32 => {
            dst[..4].copy_from_slice(&(val as f32).to_ne_bytes());
        }
        SampleFormat::F64 => {
            dst[..8].copy_from_slice(&val.to_ne_bytes());
        }
        _ => {}
    }
}

/// First-class ergonomic custom video filter (~15 LOC for a user filter).
pub trait VideoFilter: Send {
    /// Negotiate output contract given input [`VideoContract`]. Defaults to identity pass-through.
    fn negotiate(&mut self, input: &VideoContract) -> Result<VideoContract> {
        Ok(input.clone())
    }

    /// Process a single VFR [`VideoFrame`], returning `Ok(Some(frame))` or `Ok(None)` if dropped/buffered.
    fn process(&mut self, frame: VideoFrame, pool: &BufferPool) -> Result<Option<VideoFrame>>;

    /// Flush any buffered frames at end-of-stream. Called repeatedly until `Ok(None)`.
    fn flush(&mut self, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        Ok(None)
    }
}

/// First-class ergonomic custom audio filter (~15 LOC for a user filter).
pub trait AudioFilter: Send {
    /// Negotiate output contract given input [`AudioContract`]. Defaults to identity pass-through.
    fn negotiate(&mut self, input: &AudioContract) -> Result<AudioContract> {
        Ok(input.clone())
    }

    /// Process a single VFR [`AudioBlock`], returning `Ok(Some(block))` or `Ok(None)` if dropped/buffered.
    fn process(&mut self, block: AudioBlock, pool: &BufferPool) -> Result<Option<AudioBlock>>;

    /// Flush any buffered audio blocks at end-of-stream. Called repeatedly until `Ok(None)`.
    fn flush(&mut self, _pool: &BufferPool) -> Result<Option<AudioBlock>> {
        Ok(None)
    }
}

/// First-class custom GPU WGSL compute shader filter.
pub trait WgslFilter: Send {
    /// WGSL compute shader source text.
    fn wgsl_source(&self) -> &str;

    /// Compute entry point function name (defaults to `"main"`).
    fn entry_point(&self) -> &str {
        "main"
    }

    /// Compute workgroup dimensions `[x, y, z]` (defaults to `[16, 16, 1]`).
    fn workgroup_size(&self) -> [u32; 3] {
        [16, 16, 1]
    }

    /// Negotiate output [`VideoContract`] from input contract (defaults to same contract on GPU).
    fn output_contract(&self, input: &VideoContract) -> Result<VideoContract> {
        Ok(input.clone())
    }

    /// Per-frame uniform buffer bytes updated with exact VFR presentation time in seconds.
    fn uniforms(&self, _pts_sec: f64, _w: u32, _h: u32) -> &[u8] {
        &[]
    }
}

/// Pluggable text/ASS/WebVTT subtitle rasterizer (keeps `wedeo-core` and `wedeo-graph` free of font deps).
pub trait SubtitleRasterizer: Send {
    fn rasterize(
        &mut self,
        event: &SubtitleEvent,
        canvas_w: u32,
        canvas_h: u32,
        pool: &BufferPool,
    ) -> Result<Vec<PositionedPlane>>;
}

/// Allocate a zero-initialized 64-byte-aligned [`BorderedPlane`].
pub(crate) fn alloc_plane(
    width_bytes: usize,
    height: usize,
    border: BorderPadding,
) -> BorderedPlane {
    let min_stride = border
        .left
        .checked_add(width_bytes)
        .and_then(|v| v.checked_add(border.right))
        .expect("plane stride overflow");
    let align = wedeo_core::SIMD_ALIGN;
    let stride = ((min_stride + (align - 1)) & !(align - 1)).max(align);
    let total_rows = border
        .top
        .checked_add(height)
        .and_then(|v| v.checked_add(border.bottom))
        .expect("plane height overflow");
    let total_bytes = stride.checked_mul(total_rows).expect("plane size overflow");
    let origin_offset = border.top * stride + border.left;
    let buffer = Buffer::new(total_bytes);
    BorderedPlane {
        buffer,
        origin_offset,
        stride: stride as isize,
        width_bytes,
        height,
        border,
    }
}

/// Ensure all CPU planes of `frame` are uniquely owned and mutable (Copy-on-Write).
pub(crate) fn ensure_planes_writable(frame: &mut VideoFrame) -> Result<()> {
    let Some(planes) = frame.planes_mut() else {
        return Err(Error::unsupported(
            "wedeo-graph/filter",
            "CPU video filter cannot mutate GPU surface directly",
        ));
    };
    for p in planes.iter_mut() {
        if !p.buffer.is_writable() {
            let mut copy = alloc_plane(p.width_bytes, p.height, p.border);
            for y in 0..p.height {
                if let (Some(src_row), Some(dst_row)) = (p.row(y), copy.row_mut(y)) {
                    dst_row.copy_from_slice(src_row);
                }
            }
            *p = copy;
        }
    }
    Ok(())
}

// ============================================================================
// 1. PadFilter
// ============================================================================

/// Pads a video frame onto a larger `(dst_width, dst_height)` canvas at offset `(x, y)`.
///
/// VFR-native: preserves `pts`, `pkt_dts`, `duration`, `time_base`, `color`, `sar`, and `side_data`.
#[derive(Debug, Clone)]
pub struct PadFilter {
    pub dst_width: u32,
    pub dst_height: u32,
    pub x: u32,
    pub y: u32,
    /// Fill color `[c0, c1, c2, alpha]` in 8-bit units (`[Y, U, V, A]` for YCbCr, `[R, G, B, A]` for RGB).
    pub color: [u8; 4],
}

impl PadFilter {
    pub const fn new(dst_width: u32, dst_height: u32, x: u32, y: u32) -> Self {
        Self {
            dst_width,
            dst_height,
            x,
            y,
            color: [0, 0, 0, 255],
        }
    }

    pub const fn with_color(mut self, color: [u8; 4]) -> Self {
        self.color = color;
        self
    }

    fn effective_fill(&self, layout: PixelLayout, range: ColorRange) -> [u8; 4] {
        if self.color == [0, 0, 0, 255] && matches!(layout.model, ColorModel::Ycbcr) {
            let y_black = if range == ColorRange::Full { 0 } else { 16 };
            [y_black, 128, 128, 255]
        } else {
            self.color
        }
    }
}

impl VideoFilter for PadFilter {
    fn negotiate(&mut self, input: &VideoContract) -> Result<VideoContract> {
        if input.domain != MemoryDomain::Cpu {
            return Err(Error::unsupported(
                "pad",
                "PadFilter requires MemoryDomain::Cpu",
            ));
        }
        if self.x.saturating_add(input.width) > self.dst_width
            || self.y.saturating_add(input.height) > self.dst_height
        {
            return Err(Error::invalid_argument(
                "pad",
                "padded canvas is smaller than offset + input dimensions",
            ));
        }
        let mut out = input.clone();
        out.width = self.dst_width;
        out.height = self.dst_height;
        Ok(out)
    }

    fn process(&mut self, frame: VideoFrame, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        if self.x.saturating_add(frame.width) > self.dst_width
            || self.y.saturating_add(frame.height) > self.dst_height
        {
            return Err(Error::invalid_argument(
                "pad",
                "frame dimensions + offset exceed PadFilter canvas size",
            ));
        }
        if frame.width == self.dst_width && frame.height == self.dst_height {
            return Ok(Some(frame));
        }
        let Some(src_planes) = frame.planes() else {
            return Err(Error::unsupported(
                "pad",
                "GPU frame unsupported in PadFilter",
            ));
        };

        let layout = frame.layout;
        let fill = self.effective_fill(layout, frame.color.range);
        let bps = layout.depth.bytes_per_sample();
        let shift = layout.depth.bits().saturating_sub(8);
        let plane_count = layout.plane_count();
        let mut dst_planes = Vec::with_capacity(plane_count);

        for p_idx in 0..plane_count {
            let (dst_wb, dst_h) = layout
                .plane_dims(self.dst_width, self.dst_height, p_idx)
                .ok_or_else(|| Error::invalid_argument("pad", "invalid plane index"))?;
            let mut dst_p = alloc_plane(dst_wb, dst_h, BorderPadding::NONE);

            let (sub_x, sub_y) = if matches!(
                layout.packing,
                PlanePacking::Planar | PlanePacking::SemiPlanar
            ) && (p_idx == 1 || p_idx == 2)
            {
                let (cw, ch) = layout.subsampling.chroma_dims(2, 2);
                ((2 / cw.max(1)).max(1), (2 / ch.max(1)).max(1))
            } else {
                (1, 1)
            };

            // Fill canvas background
            for row_y in 0..dst_h {
                let row = dst_p.row_mut(row_y).expect("fresh plane is writable");
                match layout.packing {
                    PlanePacking::Planar => {
                        let comp = fill[p_idx.min(3)];
                        if bps == 1 {
                            row.fill(comp);
                        } else if bps == 2 {
                            let val16 = (comp as u16) << shift;
                            let bytes = val16.to_le_bytes();
                            for chunk in row.as_chunks_mut::<2>().0 {
                                chunk.copy_from_slice(&bytes);
                            }
                        }
                    }
                    PlanePacking::SemiPlanar => {
                        if p_idx == 0 {
                            if bps == 1 {
                                row.fill(fill[0]);
                            } else if bps == 2 {
                                let val16 = (fill[0] as u16) << shift;
                                let bytes = val16.to_le_bytes();
                                for chunk in row.as_chunks_mut::<2>().0 {
                                    chunk.copy_from_slice(&bytes);
                                }
                            }
                        } else if bps == 1 {
                            for pair in row.as_chunks_mut::<2>().0 {
                                pair[0] = fill[1];
                                pair[1] = fill[2];
                            }
                        }
                    }
                    PlanePacking::Packed => {
                        let px: &[u8] = match layout.order {
                            CompOrder::Rgba => &[fill[0], fill[1], fill[2], fill[3]],
                            CompOrder::Bgra => &[fill[2], fill[1], fill[0], fill[3]],
                            CompOrder::Argb => &[fill[3], fill[0], fill[1], fill[2]],
                            CompOrder::Abgr => &[fill[3], fill[2], fill[1], fill[0]],
                            CompOrder::Rgb => &[fill[0], fill[1], fill[2]],
                            CompOrder::Bgr => &[fill[2], fill[1], fill[0]],
                            _ => &[fill[0]],
                        };
                        for chunk in row.chunks_exact_mut(px.len()) {
                            chunk.copy_from_slice(px);
                        }
                    }
                }
            }

            // Copy active source rows into offset
            let src_p = &src_planes[p_idx];
            let off_y = (self.y / sub_y) as usize;
            let px_stride = dst_wb / (self.dst_width.div_ceil(sub_x) as usize).max(1);
            let off_x_bytes = (self.x / sub_x) as usize * px_stride;

            for sy in 0..src_p.height {
                let dy = off_y + sy;
                if dy < dst_h
                    && let (Some(s_row), Some(d_row)) = (src_p.row(sy), dst_p.row_mut(dy))
                {
                    let copy_len = s_row.len().min(d_row.len().saturating_sub(off_x_bytes));
                    d_row[off_x_bytes..off_x_bytes + copy_len].copy_from_slice(&s_row[..copy_len]);
                }
            }

            dst_planes.push(dst_p);
        }

        let mut out =
            VideoFrame::from_planes(self.dst_width, self.dst_height, frame.layout, dst_planes);
        out.color = frame.color;
        out.crop = frame.crop;
        out.sar = frame.sar;
        out.picture_type = frame.picture_type;
        out.flags = frame.flags;
        out.repeat_pict = frame.repeat_pict;
        out.pts = frame.pts;
        out.pkt_dts = frame.pkt_dts;
        out.duration = frame.duration;
        out.time_base = frame.time_base;
        out.side_data = frame.side_data;
        Ok(Some(out))
    }
}

// ============================================================================
// 2. OverlayFilter (shared with bitmap PGS/DVD subtitles via wedeo_video_ops)
// ============================================================================

/// Compositing filter shared between static/dynamic image overlays and timed bitmap (`PGS`/`DVD`)
/// or rasterized text [`SubtitleEvent`]s via [`wedeo_video_ops::CompiledSws::blend_overlay`].
pub struct OverlayFilter {
    pub x_offset: u32,
    pub y_offset: u32,
    static_overlays: Vec<PositionedPlane>,
    subtitle_events: Vec<SubtitleEvent>,
    rasterizer: Option<Box<dyn SubtitleRasterizer>>,
    sws_cache: Option<(VideoContract, CpuCompiledSws)>,
    internal_pool: Arc<BufferPool>,
}

impl OverlayFilter {
    pub fn new(x_offset: u32, y_offset: u32) -> Self {
        Self {
            x_offset,
            y_offset,
            static_overlays: Vec::new(),
            subtitle_events: Vec::new(),
            rasterizer: None,
            sws_cache: None,
            internal_pool: BufferPool::new(),
        }
    }

    /// Construct an `OverlayFilter` pre-loaded with static [`PositionedPlane`] overlays.
    pub fn from_planes(overlays: Vec<PositionedPlane>) -> Self {
        let mut f = Self::new(0, 0);
        f.static_overlays = overlays;
        f
    }

    /// Add a static overlay plane that is composited onto every frame.
    pub fn add_overlay(&mut self, mut plane: PositionedPlane) {
        plane.x = plane.x.saturating_add(self.x_offset);
        plane.y = plane.y.saturating_add(self.y_offset);
        self.static_overlays.push(plane);
    }

    /// Queue a timed [`SubtitleEvent`] (`SubtitlePayload::Bitmap` for `PGS`/`DVD` or text for `SubtitleRasterizer`).
    pub fn push_subtitle(&mut self, event: SubtitleEvent) {
        self.subtitle_events.push(event);
    }

    /// Attach a pluggable [`SubtitleRasterizer`] for text/ASS/WebVTT subtitle burn-in.
    pub fn with_rasterizer(mut self, rasterizer: Box<dyn SubtitleRasterizer>) -> Self {
        self.rasterizer = Some(rasterizer);
        self
    }

    fn event_active_at(event: &SubtitleEvent, frame_pts: Timestamp, frame_tb: Rational) -> bool {
        if !event.pts.is_known() || !frame_pts.is_known() {
            return true;
        }
        if !event.time_base.is_valid() || !frame_tb.is_valid() {
            let start = event.pts.raw();
            let end = start.saturating_add(event.duration.max(1));
            let f = frame_pts.raw();
            return f >= start && f < end;
        }
        let start_in_frame_tb = rescale_q(event.pts.raw(), event.time_base, frame_tb);
        let dur_in_frame_tb = rescale_q(event.duration.max(1), event.time_base, frame_tb).max(1);
        let end_in_frame_tb = start_in_frame_tb.saturating_add(dur_in_frame_tb);
        let f = frame_pts.raw();
        f >= start_in_frame_tb && f < end_in_frame_tb
    }
}

impl VideoFilter for OverlayFilter {
    fn negotiate(&mut self, input: &VideoContract) -> Result<VideoContract> {
        if input.domain != MemoryDomain::Cpu {
            return Err(Error::unsupported(
                "overlay",
                "OverlayFilter requires MemoryDomain::Cpu",
            ));
        }
        let ops = SwsOpList::build(input, input, SwsOptions::default())?;
        let compiled = CpuCompiledSws::compile(&ops)?;
        self.sws_cache = Some((input.clone(), compiled));
        Ok(input.clone())
    }

    fn process(&mut self, mut frame: VideoFrame, pool: &BufferPool) -> Result<Option<VideoFrame>> {
        let mut active: Vec<PositionedPlane> = self.static_overlays.clone();

        for ev in &self.subtitle_events {
            if Self::event_active_at(ev, frame.pts, frame.time_base) {
                match &ev.payload {
                    SubtitlePayload::Bitmap(planes) => {
                        for p in planes {
                            let mut shifted = p.clone();
                            shifted.x = shifted.x.saturating_add(self.x_offset);
                            shifted.y = shifted.y.saturating_add(self.y_offset);
                            active.push(shifted);
                        }
                    }
                    SubtitlePayload::Text(_)
                    | SubtitlePayload::Ass(_)
                    | SubtitlePayload::WebVtt(_) => {
                        if let Some(rast) = self.rasterizer.as_mut() {
                            let planes = rast.rasterize(ev, frame.width, frame.height, pool)?;
                            for mut p in planes {
                                p.x = p.x.saturating_add(self.x_offset);
                                p.y = p.y.saturating_add(self.y_offset);
                                active.push(p);
                            }
                        }
                    }
                }
            }
        }

        if active.is_empty() {
            return Ok(Some(frame));
        }

        let contract = VideoContract::from_frame(&frame);
        let needs_rebuild = match &self.sws_cache {
            Some((cached_c, _)) => cached_c != &contract,
            None => true,
        };
        if needs_rebuild {
            let ops = SwsOpList::build(&contract, &contract, SwsOptions::default())?;
            let compiled = CpuCompiledSws::compile(&ops)?;
            self.sws_cache = Some((contract, compiled));
        }

        if let Some((_, compiled)) = self.sws_cache.as_mut() {
            compiled.blend_overlay(&mut frame, &active, &self.internal_pool)?;
        }

        Ok(Some(frame))
    }
}

// ============================================================================
// 3. FpsFilter (explicit VFR -> CFR conversion)
// ============================================================================

/// Explicit VFR-to-CFR conversion filter.
///
/// Rescales incoming VFR presentation timestamps onto an exact CFR grid `time_base = 1 / fps`,
/// dropping sub-frame duplicates and duplicating frames across multi-period VFR gaps.
#[derive(Debug)]
pub struct FpsFilter {
    pub target_fps: Rational,
    out_time_base: Rational,
    next_out_pts: Option<i64>,
    last_frame: Option<VideoFrame>,
    pending_out: VecDeque<VideoFrame>,
}

impl FpsFilter {
    pub fn new(target_fps: Rational) -> Result<Self> {
        if !target_fps.is_valid() || target_fps.num <= 0 {
            return Err(Error::invalid_argument(
                "fps",
                "target_fps must be a positive valid rational",
            ));
        }
        Ok(Self {
            target_fps,
            out_time_base: target_fps.invert(),
            next_out_pts: None,
            last_frame: None,
            pending_out: VecDeque::new(),
        })
    }

    /// Drain any additional duplicated CFR frames produced when a VFR gap spanned multiple CFR ticks.
    pub fn pop_queued(&mut self) -> Option<VideoFrame> {
        self.pending_out.pop_front()
    }

    fn stamp_cfr_frame(&self, mut frame: VideoFrame, cfr_pts: i64) -> VideoFrame {
        frame.pts = Timestamp::new(cfr_pts);
        frame.pkt_dts = Timestamp::new(cfr_pts);
        frame.duration = 1;
        frame.time_base = self.out_time_base;
        frame
    }
}

impl VideoFilter for FpsFilter {
    fn process(&mut self, frame: VideoFrame, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        if !frame.pts.is_known() || !frame.time_base.is_valid() {
            let pts = self.next_out_pts.unwrap_or(0);
            self.next_out_pts = Some(pts + 1);
            let out = self.stamp_cfr_frame(frame, pts);
            return Ok(Some(out));
        }

        let in_pts_cfr = rescale_q_rnd(
            frame.pts.raw(),
            frame.time_base,
            self.out_time_base,
            Rounding::NearInf,
            false,
        );
        let in_dur_cfr = if frame.duration > 0 {
            rescale_q_rnd(
                frame.duration,
                frame.time_base,
                self.out_time_base,
                Rounding::NearInf,
                false,
            )
            .max(1)
        } else {
            1
        };

        let next_pts = match self.next_out_pts {
            None => {
                self.next_out_pts = Some(in_pts_cfr);
                in_pts_cfr
            }
            Some(n) => n,
        };

        // If we had a previous frame and there is a VFR gap before this frame arrives,
        // duplicate the previous frame up to `in_pts_cfr`.
        if let Some(prev) = self.last_frame.as_ref() {
            let mut cur = next_pts;
            while cur < in_pts_cfr {
                let dup = self.stamp_cfr_frame(prev.clone(), cur);
                self.pending_out.push_back(dup);
                cur += 1;
            }
            self.next_out_pts = Some(cur);
        }

        let cur_next = self.next_out_pts.unwrap_or(in_pts_cfr);
        if in_pts_cfr + in_dur_cfr <= cur_next && self.last_frame.is_some() {
            // Incoming VFR frame falls entirely before the next CFR grid point: drop it
            // (or pop a previously queued duplicate if one is waiting).
            self.last_frame = Some(frame);
            return Ok(self.pending_out.pop_front());
        }

        // Emit current frame at `cur_next`, and if its duration spans multiple CFR ticks,
        // queue duplicates for those ticks.
        let mut emit_cursor = cur_next;
        let target_end = (in_pts_cfr + in_dur_cfr).max(cur_next + 1);
        while emit_cursor < target_end {
            let out = self.stamp_cfr_frame(frame.clone(), emit_cursor);
            self.pending_out.push_back(out);
            emit_cursor += 1;
        }
        self.next_out_pts = Some(emit_cursor);
        self.last_frame = Some(frame);

        Ok(self.pending_out.pop_front())
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        Ok(self.pending_out.pop_front())
    }
}

// ============================================================================
// 4. SetPtsFilter
// ============================================================================

/// Timestamp transformation mode for [`SetPtsFilter`].
pub enum SetPtsMode {
    /// Add a constant signed tick offset to every timestamp.
    Offset(i64),
    /// Multiply timestamps and durations by a rational scale factor (e.g. `1/2` for 2× speedup).
    Scale(Rational),
    /// Rescale timestamps and durations into a new `time_base`.
    RescaleTimeBase(Rational),
    /// Shift timestamps so the first seen frame/block begins at `Timestamp::new(0)`.
    ZeroStart,
    /// Custom deterministic callback `(pts, frame_index, time_base) -> Timestamp`.
    Custom(Box<dyn FnMut(Timestamp, usize, Rational) -> Timestamp + Send>),
}

/// Zero-copy VFR timestamp manipulation filter (implements both [`VideoFilter`] and [`AudioFilter`]).
pub struct SetPtsFilter {
    mode: SetPtsMode,
    first_pts: Option<i64>,
    index: usize,
}

impl SetPtsFilter {
    pub fn new(mode: SetPtsMode) -> Self {
        Self {
            mode,
            first_pts: None,
            index: 0,
        }
    }

    pub fn offset(delta_ticks: i64) -> Self {
        Self::new(SetPtsMode::Offset(delta_ticks))
    }

    pub fn scale(factor: Rational) -> Self {
        Self::new(SetPtsMode::Scale(factor))
    }

    pub fn rescale_time_base(new_tb: Rational) -> Self {
        Self::new(SetPtsMode::RescaleTimeBase(new_tb))
    }

    pub fn zero_start() -> Self {
        Self::new(SetPtsMode::ZeroStart)
    }

    fn transform(
        &mut self,
        pts: Timestamp,
        dts: Timestamp,
        duration: i64,
        tb: Rational,
    ) -> (Timestamp, Timestamp, i64, Rational) {
        let idx = self.index;
        self.index += 1;

        match &mut self.mode {
            SetPtsMode::Offset(delta) => (
                pts.checked_add(*delta).unwrap_or(pts),
                dts.checked_add(*delta).unwrap_or(dts),
                duration,
                tb,
            ),
            SetPtsMode::Scale(factor) => {
                if !factor.is_valid() {
                    return (pts, dts, duration, tb);
                }
                let unit = Rational::new(1, 1);
                let new_pts = pts
                    .get()
                    .map(|v| Timestamp::new(rescale_q(v, *factor, unit)))
                    .unwrap_or(Timestamp::NONE);
                let new_dts = dts
                    .get()
                    .map(|v| Timestamp::new(rescale_q(v, *factor, unit)))
                    .unwrap_or(Timestamp::NONE);
                let new_dur = rescale_q(duration, *factor, unit);
                (new_pts, new_dts, new_dur, tb)
            }
            SetPtsMode::RescaleTimeBase(new_tb) => {
                if !tb.is_valid() || !new_tb.is_valid() {
                    return (pts, dts, duration, *new_tb);
                }
                let new_pts = pts.rescale(tb, *new_tb, Rounding::NearInf);
                let new_dts = dts.rescale(tb, *new_tb, Rounding::NearInf);
                let new_dur = rescale_q(duration, tb, *new_tb);
                (new_pts, new_dts, new_dur, *new_tb)
            }
            SetPtsMode::ZeroStart => {
                if let Some(raw) = pts.get() {
                    let base = *self.first_pts.get_or_insert(raw);
                    let new_pts = pts.checked_sub(base).unwrap_or(pts);
                    let new_dts = dts.checked_sub(base).unwrap_or(dts);
                    (new_pts, new_dts, duration, tb)
                } else {
                    (pts, dts, duration, tb)
                }
            }
            SetPtsMode::Custom(f) => {
                let new_pts = f(pts, idx, tb);
                (new_pts, dts, duration, tb)
            }
        }
    }
}

impl VideoFilter for SetPtsFilter {
    fn process(&mut self, mut frame: VideoFrame, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        let (pts, dts, dur, tb) =
            self.transform(frame.pts, frame.pkt_dts, frame.duration, frame.time_base);
        frame.pts = pts;
        frame.pkt_dts = dts;
        frame.duration = dur;
        frame.time_base = tb;
        Ok(Some(frame))
    }
}

impl AudioFilter for SetPtsFilter {
    fn process(&mut self, mut block: AudioBlock, _pool: &BufferPool) -> Result<Option<AudioBlock>> {
        let (pts, _, dur, tb) =
            self.transform(block.pts, Timestamp::NONE, block.duration, block.time_base);
        block.pts = pts;
        block.duration = dur;
        block.time_base = tb;
        Ok(Some(block))
    }
}

// ============================================================================
// 5. TrimFilter (VFR-native video & sample-exact audio range trimmer)
// ============================================================================

/// VFR-native presentation-time range trimming filter implementing both [`VideoFilter`] and [`AudioFilter`].
#[derive(Debug, Clone)]
pub struct TrimFilter {
    pub range: ClipRange,
    /// When `true`, shifts kept timestamps so `range.start` maps to `0`.
    pub zero_based: bool,
}

impl TrimFilter {
    pub const fn new(range: ClipRange, zero_based: bool) -> Self {
        Self { range, zero_based }
    }
}

impl VideoFilter for TrimFilter {
    fn process(&mut self, mut frame: VideoFrame, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        if !frame.pts.is_known() {
            return Ok(None);
        }
        let tb = if frame.time_base.is_valid() {
            frame.time_base
        } else {
            self.range.time_base
        };
        if !self.range.contains(frame.pts, tb) {
            return Ok(None);
        }
        if self.zero_based
            && let Some(start_raw) = self.range.start.get()
        {
            let start_in_tb = if self.range.time_base.is_valid() && tb.is_valid() {
                rescale_q(start_raw, self.range.time_base, tb)
            } else {
                start_raw
            };
            frame.pts = frame.pts.checked_sub(start_in_tb).unwrap_or(frame.pts);
            frame.pkt_dts = frame
                .pkt_dts
                .checked_sub(start_in_tb)
                .unwrap_or(frame.pkt_dts);
        }
        Ok(Some(frame))
    }
}

impl AudioFilter for TrimFilter {
    fn process(&mut self, mut block: AudioBlock, _pool: &BufferPool) -> Result<Option<AudioBlock>> {
        if !block.pts.is_known() {
            return Ok(None);
        }
        let tb = if block.time_base.is_valid() {
            block.time_base
        } else {
            self.range.time_base
        };
        if !self.range.contains(block.pts, tb) {
            return Ok(None);
        }
        if self.zero_based
            && let Some(start_raw) = self.range.start.get()
        {
            let start_in_tb = if self.range.time_base.is_valid() && tb.is_valid() {
                rescale_q(start_raw, self.range.time_base, tb)
            } else {
                start_raw
            };
            block.pts = block.pts.checked_sub(start_in_tb).unwrap_or(block.pts);
        }
        Ok(Some(block))
    }
}

// ============================================================================
// 6. BwdifFilter (Bob Weaver Deinterlacing Filter)
// ============================================================================

/// Field emission mode for [`BwdifFilter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BwdifMode {
    /// Output one deinterlaced frame per input frame (same frame rate).
    #[default]
    SendFrame,
    /// Output one frame per field (bob-weave 2× rate, queueing the second field).
    SendField,
}

/// Field parity selection for [`BwdifFilter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BwdifParity {
    #[default]
    Auto,
    TopFieldFirst,
    BottomFieldFirst,
}

/// Spatial/temporal Bob-Weaver Deinterlacing Filter (`bwdif`).
///
/// Combines vertical edge-directed spatial interpolation with temporal weaving against
/// the previous frame when inter-frame motion is below threshold.
#[derive(Debug, Clone)]
pub struct BwdifFilter {
    pub mode: BwdifMode,
    pub parity: BwdifParity,
    /// When `false`, progressive frames (without `FrameFlags::INTERLACED`) pass through zero-copy.
    pub deint_all: bool,
    prev_frame: Option<VideoFrame>,
    second_field_queue: Option<VideoFrame>,
}

impl BwdifFilter {
    pub const fn new(mode: BwdifMode, parity: BwdifParity, deint_all: bool) -> Self {
        Self {
            mode,
            parity,
            deint_all,
            prev_frame: None,
            second_field_queue: None,
        }
    }

    fn is_tff(&self, flags: FrameFlags) -> bool {
        match self.parity {
            BwdifParity::TopFieldFirst => true,
            BwdifParity::BottomFieldFirst => false,
            BwdifParity::Auto => flags.contains(FrameFlags::TOP_FIELD_FIRST),
        }
    }

    fn deinterlace_field(&self, frame: &VideoFrame, keep_top_field: bool) -> Result<VideoFrame> {
        let Some(src_planes) = frame.planes() else {
            return Err(Error::unsupported(
                "bwdif",
                "BwdifFilter requires MemoryDomain::Cpu",
            ));
        };
        let prev_planes = self.prev_frame.as_ref().and_then(VideoFrame::planes);

        let mut dst_planes = Vec::with_capacity(src_planes.len());
        for (p_idx, sp) in src_planes.iter().enumerate() {
            let mut dp = alloc_plane(sp.width_bytes, sp.height, BorderPadding::NONE);
            let h = sp.height;
            for y in 0..h {
                let keep_row = if keep_top_field {
                    y % 2 == 0
                } else {
                    y % 2 == 1
                };
                let dst_row = dp.row_mut(y).expect("writable");
                if keep_row || h < 2 {
                    if let Some(src_row) = sp.row(y) {
                        dst_row.copy_from_slice(src_row);
                    }
                } else {
                    let y_above = y.saturating_sub(1);
                    let y_below = (y + 1).min(h - 1);
                    let above = sp.row(y_above).unwrap_or(&[]);
                    let below = sp.row(y_below).unwrap_or(&[]);
                    let cur = sp.row(y).unwrap_or(&[]);
                    let prev_row = prev_planes
                        .and_then(|pp| pp.get(p_idx))
                        .and_then(|p| p.row(y));

                    for (i, dst_cell) in dst_row.iter_mut().enumerate() {
                        let a = above.get(i).copied().unwrap_or(0) as i32;
                        let b = below.get(i).copied().unwrap_or(0) as i32;
                        let spatial = (a + b + 1) >> 1;

                        let c = cur.get(i).copied().unwrap_or(spatial as u8) as i32;
                        let val = if let Some(pr) = prev_row {
                            let p = pr.get(i).copied().unwrap_or(c as u8) as i32;
                            let temporal_diff = (c - p).abs();
                            let spatial_diff = (a - b).abs();
                            if temporal_diff <= 8 && temporal_diff <= spatial_diff {
                                // Static region: weave temporal average
                                (c + p + 1) >> 1
                            } else {
                                // Motion region: edge-clamped bwdif spatial filter
                                let hf = (a + b) / 2 + (c - ((a + b) / 2)).clamp(-16, 16);
                                hf.clamp(0, 255)
                            }
                        } else {
                            spatial
                        };
                        *dst_cell = val as u8;
                    }
                }
            }
            dst_planes.push(dp);
        }

        let mut out = VideoFrame::from_planes(frame.width, frame.height, frame.layout, dst_planes);
        out.color = frame.color;
        out.crop = frame.crop;
        out.sar = frame.sar;
        out.picture_type = frame.picture_type;
        out.flags = frame.flags - FrameFlags::INTERLACED - FrameFlags::TOP_FIELD_FIRST;
        out.repeat_pict = 0;
        out.pts = frame.pts;
        out.pkt_dts = frame.pkt_dts;
        out.duration = frame.duration;
        out.time_base = frame.time_base;
        out.side_data = frame.side_data.clone();
        Ok(out)
    }
}

impl Default for BwdifFilter {
    fn default() -> Self {
        Self::new(BwdifMode::SendFrame, BwdifParity::Auto, true)
    }
}

impl VideoFilter for BwdifFilter {
    fn process(&mut self, frame: VideoFrame, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        if !self.deint_all && !frame.flags.contains(FrameFlags::INTERLACED) {
            self.prev_frame = Some(frame.clone());
            return Ok(Some(frame));
        }

        let tff = self.is_tff(frame.flags);
        let mut first = self.deinterlace_field(&frame, tff)?;

        if self.mode == BwdifMode::SendField {
            let half_dur = (frame.duration / 2).max(1);
            first.duration = half_dur;
            let mut second = self.deinterlace_field(&frame, !tff)?;
            second.duration = frame.duration.saturating_sub(half_dur).max(1);
            second.pts = frame.pts.checked_add(half_dur).unwrap_or(frame.pts);
            self.second_field_queue = Some(second);
        }

        self.prev_frame = Some(frame);
        Ok(Some(first))
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        Ok(self.second_field_queue.take())
    }
}

// ============================================================================
// 7. AmixFilter (Multi-stream VFR audio mixer)
// ============================================================================

/// Multi-stream audio mixer (`amix`) with configurable per-stream weights.
///
/// Can mix multiple [`AudioBlock`] slices directly via [`AmixFilter::mix_blocks`] or operate
/// as an [`AudioFilter`] mixing primary blocks with queued secondary blocks.
#[derive(Debug, Clone)]
pub struct AmixFilter {
    pub weights: Vec<f64>,
    pub normalize: bool,
    secondary_queue: VecDeque<AudioBlock>,
}

impl AmixFilter {
    pub fn new(inputs: usize, normalize: bool) -> Self {
        let n = inputs.max(1);
        let w = if normalize { 1.0 / n as f64 } else { 1.0 };
        Self {
            weights: vec![w; n],
            normalize,
            secondary_queue: VecDeque::new(),
        }
    }

    pub fn with_weights(weights: Vec<f64>) -> Self {
        Self {
            weights,
            normalize: false,
            secondary_queue: VecDeque::new(),
        }
    }

    /// Push a secondary stream [`AudioBlock`] to be mixed with the next primary block.
    pub fn push_secondary(&mut self, block: AudioBlock) {
        self.secondary_queue.push_back(block);
    }

    /// Mix a slice of compatible [`AudioBlock`]s sample-by-sample in `f64` precision.
    pub fn mix_blocks(&self, blocks: &[AudioBlock]) -> Result<AudioBlock> {
        let Some(first) = blocks.first() else {
            return Err(Error::invalid_argument(
                "amix",
                "at least one AudioBlock is required",
            ));
        };
        if blocks.len() == 1 && (self.weights.first().copied().unwrap_or(1.0) - 1.0).abs() < 1e-12 {
            return Ok(first.clone());
        }

        let fmt = first.format;
        let channels = first.channel_layout.nb_channels as usize;
        let samples = first.nb_samples as usize;
        let bps = fmt.bytes_per_sample();

        for b in &blocks[1..] {
            if b.format != fmt
                || b.channel_layout.nb_channels != first.channel_layout.nb_channels
                || b.sample_rate != first.sample_rate
            {
                return Err(Error::invalid_argument(
                    "amix",
                    "all mixed AudioBlocks must share format, sample_rate, and channel count",
                ));
            }
        }

        let mut planes = Vec::with_capacity(first.expected_plane_count());
        if fmt.is_planar() {
            for ch in 0..channels {
                let mut out_bytes = vec![0u8; samples * bps];
                for s in 0..samples {
                    let mut acc = 0.0_f64;
                    for (b_idx, blk) in blocks.iter().enumerate() {
                        let w = self.weights.get(b_idx).copied().unwrap_or(1.0);
                        if s < blk.nb_samples as usize {
                            let off = s * bps;
                            let val =
                                unpack_sample_f64(fmt, &blk.planes[ch].data()[off..off + bps]);
                            acc += val * w;
                        }
                    }
                    let off = s * bps;
                    pack_sample_f64(fmt, acc, &mut out_bytes[off..off + bps]);
                }
                planes.push(Buffer::from_slice(&out_bytes));
            }
        } else {
            let total_elems = samples * channels;
            let mut out_bytes = vec![0u8; total_elems * bps];
            for idx in 0..total_elems {
                let s = idx / channels;
                let mut acc = 0.0_f64;
                for (b_idx, blk) in blocks.iter().enumerate() {
                    let w = self.weights.get(b_idx).copied().unwrap_or(1.0);
                    if s < blk.nb_samples as usize {
                        let off = idx * bps;
                        let val = unpack_sample_f64(fmt, &blk.planes[0].data()[off..off + bps]);
                        acc += val * w;
                    }
                }
                let off = idx * bps;
                pack_sample_f64(fmt, acc, &mut out_bytes[off..off + bps]);
            }
            planes.push(Buffer::from_slice(&out_bytes));
        }

        let mut mixed = AudioBlock::new(
            planes,
            first.nb_samples,
            first.format,
            first.sample_rate,
            first.channel_layout.clone(),
        );
        mixed.pts = first.pts;
        mixed.duration = first.duration;
        mixed.time_base = first.time_base;
        mixed.side_data = first.side_data.clone();
        Ok(mixed)
    }
}

impl AudioFilter for AmixFilter {
    fn process(&mut self, block: AudioBlock, _pool: &BufferPool) -> Result<Option<AudioBlock>> {
        if let Some(sec) = self.secondary_queue.pop_front() {
            let mixed = self.mix_blocks(&[block, sec])?;
            Ok(Some(mixed))
        } else {
            let mixed = self.mix_blocks(&[block])?;
            Ok(Some(mixed))
        }
    }
}

// ============================================================================
// 8. FadeFilter (VFR-native video & audio fade-in / fade-out)
// ============================================================================

/// Direction of a [`FadeFilter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeKind {
    In,
    Out,
}

/// VFR-native fade-in / fade-out filter implementing both [`VideoFilter`] and [`AudioFilter`].
#[derive(Debug, Clone)]
pub struct FadeFilter {
    pub kind: FadeKind,
    pub start: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
}

impl FadeFilter {
    pub const fn new(kind: FadeKind, start: Timestamp, duration: i64, time_base: Rational) -> Self {
        Self {
            kind,
            start,
            duration,
            time_base,
        }
    }

    /// Evaluate fade gain factor in `[0.0, 1.0]` at presentation timestamp `pts` in `tb`.
    pub fn gain_at(&self, pts: Timestamp, tb: Rational) -> f64 {
        let (Some(p_raw), Some(s_raw)) = (pts.get(), self.start.get()) else {
            return 1.0;
        };
        if self.duration <= 0 {
            return match self.kind {
                FadeKind::In => 1.0,
                FadeKind::Out => 0.0,
            };
        }
        let p_in_fade_tb = if tb.is_valid() && self.time_base.is_valid() {
            rescale_q(p_raw, tb, self.time_base)
        } else {
            p_raw
        };
        let elapsed = p_in_fade_tb - s_raw;
        let t = (elapsed as f64 / self.duration as f64).clamp(0.0, 1.0);
        match self.kind {
            FadeKind::In => t,
            FadeKind::Out => 1.0 - t,
        }
    }
}

impl VideoFilter for FadeFilter {
    fn process(&mut self, mut frame: VideoFrame, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        let gain = self.gain_at(frame.pts, frame.time_base);
        if (gain - 1.0).abs() < 1e-6 {
            return Ok(Some(frame));
        }
        ensure_planes_writable(&mut frame)?;

        let layout = frame.layout;
        let is_ycbcr = matches!(layout.model, ColorModel::Ycbcr);
        let limited = frame.color.range != ColorRange::Full;
        let y_black = if is_ycbcr && limited { 16.0 } else { 0.0 };
        let uv_neutral = 128.0;

        let Some(planes) = frame.planes_mut() else {
            return Ok(Some(frame));
        };

        if matches!(layout.depth, SampleDepth::U8) {
            for (p_idx, plane) in planes.iter_mut().enumerate() {
                let target = if is_ycbcr && (p_idx == 1 || p_idx == 2) {
                    uv_neutral
                } else {
                    y_black
                };
                for y in 0..plane.height {
                    if let Some(row) = plane.row_mut(y) {
                        if matches!(layout.packing, PlanePacking::Packed)
                            && matches!(layout.order, CompOrder::Rgba | CompOrder::Bgra)
                        {
                            // Preserve alpha channel on packed RGBA/BGRA
                            for px in row.as_chunks_mut::<4>().0 {
                                px[0] = (target + (px[0] as f64 - target) * gain)
                                    .round()
                                    .clamp(0.0, 255.0)
                                    as u8;
                                px[1] = (target + (px[1] as f64 - target) * gain)
                                    .round()
                                    .clamp(0.0, 255.0)
                                    as u8;
                                px[2] = (target + (px[2] as f64 - target) * gain)
                                    .round()
                                    .clamp(0.0, 255.0)
                                    as u8;
                            }
                        } else {
                            for b in row.iter_mut() {
                                *b = (target + (*b as f64 - target) * gain)
                                    .round()
                                    .clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }
            }
        }

        Ok(Some(frame))
    }
}

impl AudioFilter for FadeFilter {
    fn process(&mut self, block: AudioBlock, _pool: &BufferPool) -> Result<Option<AudioBlock>> {
        let fmt = block.format;
        let bps = fmt.bytes_per_sample();
        let channels = block.channel_layout.nb_channels as usize;
        let samples = block.nb_samples as usize;
        let sample_tb = Rational::new(1, block.sample_rate.max(1) as i32);
        let base_pts_in_sample_tb = if block.pts.is_known() && block.time_base.is_valid() {
            Some(rescale_q(block.pts.raw(), block.time_base, sample_tb))
        } else {
            block.pts.get()
        };

        let mut new_planes = Vec::with_capacity(block.planes.len());
        if fmt.is_planar() {
            for plane in block.planes.iter().take(channels) {
                let src = plane.data();
                let mut dst = vec![0u8; samples * bps];
                for s in 0..samples {
                    let s_pts = base_pts_in_sample_tb
                        .map(|b| Timestamp::new(b + s as i64))
                        .unwrap_or(Timestamp::NONE);
                    let g = self.gain_at(s_pts, sample_tb);
                    let off = s * bps;
                    let val = unpack_sample_f64(fmt, &src[off..off + bps]) * g;
                    pack_sample_f64(fmt, val, &mut dst[off..off + bps]);
                }
                new_planes.push(Buffer::from_slice(&dst));
            }
        } else {
            let src = block.planes[0].data();
            let mut dst = vec![0u8; samples * channels * bps];
            for s in 0..samples {
                let s_pts = base_pts_in_sample_tb
                    .map(|b| Timestamp::new(b + s as i64))
                    .unwrap_or(Timestamp::NONE);
                let g = self.gain_at(s_pts, sample_tb);
                for ch in 0..channels {
                    let off = (s * channels + ch) * bps;
                    let val = unpack_sample_f64(fmt, &src[off..off + bps]) * g;
                    pack_sample_f64(fmt, val, &mut dst[off..off + bps]);
                }
            }
            new_planes.push(Buffer::from_slice(&dst));
        }

        let mut out = AudioBlock::new(
            new_planes,
            block.nb_samples,
            block.format,
            block.sample_rate,
            block.channel_layout,
        );
        out.pts = block.pts;
        out.duration = block.duration;
        out.time_base = block.time_base;
        out.side_data = block.side_data;
        Ok(Some(out))
    }
}

// ============================================================================
// 9. SilenceDetectFilter
// ============================================================================

/// One detected silent audio interval with exact VFR timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilenceSpan {
    pub start: Timestamp,
    pub end: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
}

/// VFR-native zero-copy silence detection filter (`silencedetect`).
#[derive(Debug, Clone)]
pub struct SilenceDetectFilter {
    /// Linear amplitude threshold (e.g. `10^(-50/20) ≈ 0.003162` for `-50 dBFS`).
    pub amplitude_threshold: f64,
    /// Minimum silence duration in `time_base` ticks (or seconds converted per stream).
    pub min_duration_sec: f64,
    active_start: Option<(Timestamp, Rational)>,
    last_silent_end: Option<(Timestamp, Rational)>,
    detected: Vec<SilenceSpan>,
}

impl SilenceDetectFilter {
    pub fn new(noise_db: f64, min_duration_sec: f64) -> Self {
        let amplitude_threshold = 10.0_f64.powf(noise_db / 20.0);
        Self {
            amplitude_threshold,
            min_duration_sec: min_duration_sec.max(0.0),
            active_start: None,
            last_silent_end: None,
            detected: Vec::new(),
        }
    }

    pub fn detected_silences(&self) -> &[SilenceSpan] {
        &self.detected
    }

    fn maybe_commit_active(&mut self) {
        if let (Some((start, tb)), Some((end, _))) =
            (self.active_start.take(), self.last_silent_end.take())
            && let Some(dur) = end.diff(start)
        {
            let min_ticks = if tb.is_valid() {
                (self.min_duration_sec / tb.to_f64()).round() as i64
            } else {
                0
            };
            if dur >= min_ticks {
                self.detected.push(SilenceSpan {
                    start,
                    end,
                    duration: dur,
                    time_base: tb,
                });
            }
        }
    }

    fn is_block_silent(&self, block: &AudioBlock) -> bool {
        let fmt = block.format;
        let bps = fmt.bytes_per_sample();
        let samples = block.nb_samples as usize;
        let channels = block.channel_layout.nb_channels as usize;
        if samples == 0 {
            return true;
        }
        if fmt.is_planar() {
            for plane in block.planes.iter().take(channels) {
                let data = plane.data();
                for s in 0..samples {
                    let off = s * bps;
                    let amp = unpack_sample_f64(fmt, &data[off..off + bps]).abs();
                    if amp > self.amplitude_threshold {
                        return false;
                    }
                }
            }
        } else {
            let data = block.planes[0].data();
            let total = samples * channels;
            for idx in 0..total {
                let off = idx * bps;
                let amp = unpack_sample_f64(fmt, &data[off..off + bps]).abs();
                if amp > self.amplitude_threshold {
                    return false;
                }
            }
        }
        true
    }
}

impl AudioFilter for SilenceDetectFilter {
    fn process(&mut self, block: AudioBlock, _pool: &BufferPool) -> Result<Option<AudioBlock>> {
        let silent = self.is_block_silent(&block);
        let tb = block.time_base;
        let pts = block.pts;
        let end_pts = pts.checked_add(block.duration).unwrap_or(pts);

        if silent {
            if self.active_start.is_none() {
                self.active_start = Some((pts, tb));
            }
            self.last_silent_end = Some((end_pts, tb));
        } else {
            self.maybe_commit_active();
        }
        Ok(Some(block))
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Option<AudioBlock>> {
        self.maybe_commit_active();
        Ok(None)
    }
}

// ============================================================================
// 10. BlackDetectFilter
// ============================================================================

/// One detected black video interval with exact VFR timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlackSpan {
    pub start: Timestamp,
    pub end: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
}

/// VFR-native zero-copy black-frame interval detector (`blackdetect`).
#[derive(Debug, Clone)]
pub struct BlackDetectFilter {
    /// Normalized luma threshold in `[0.0, 1.0]` below which a pixel is considered black (default `0.10`).
    pub pixel_black_th: f64,
    /// Minimum fraction of black pixels in `[0.0, 1.0]` for a frame to be classified black (default `0.98`).
    pub picture_black_ratio_th: f64,
    /// Minimum black span duration in seconds.
    pub min_duration_sec: f64,
    active_start: Option<(Timestamp, Rational)>,
    last_black_end: Option<(Timestamp, Rational)>,
    detected: Vec<BlackSpan>,
}

impl BlackDetectFilter {
    pub fn new(pixel_black_th: f64, picture_black_ratio_th: f64, min_duration_sec: f64) -> Self {
        Self {
            pixel_black_th: pixel_black_th.clamp(0.0, 1.0),
            picture_black_ratio_th: picture_black_ratio_th.clamp(0.0, 1.0),
            min_duration_sec: min_duration_sec.max(0.0),
            active_start: None,
            last_black_end: None,
            detected: Vec::new(),
        }
    }

    pub fn detected_black_spans(&self) -> &[BlackSpan] {
        &self.detected
    }

    fn maybe_commit_active(&mut self) {
        if let (Some((start, tb)), Some((end, _))) =
            (self.active_start.take(), self.last_black_end.take())
            && let Some(dur) = end.diff(start)
        {
            let min_ticks = if tb.is_valid() {
                (self.min_duration_sec / tb.to_f64()).round() as i64
            } else {
                0
            };
            if dur >= min_ticks {
                self.detected.push(BlackSpan {
                    start,
                    end,
                    duration: dur,
                    time_base: tb,
                });
            }
        }
    }

    fn is_frame_black(&self, frame: &VideoFrame) -> bool {
        let Some(planes) = frame.planes() else {
            return false;
        };
        let Some(p0) = planes.first() else {
            return false;
        };
        let is_limited = matches!(frame.layout.model, ColorModel::Ycbcr)
            && frame.color.range != ColorRange::Full;
        let (min_v, span) = if is_limited {
            (16.0, 219.0)
        } else {
            (0.0, 255.0)
        };
        let thresh_u8 = (min_v + self.pixel_black_th * span)
            .round()
            .clamp(0.0, 255.0) as u8;

        let mut total_px = 0usize;
        let mut black_px = 0usize;

        if matches!(frame.layout.packing, PlanePacking::Packed)
            && matches!(frame.layout.model, ColorModel::Rgb)
        {
            let step = if matches!(frame.layout.alpha, wedeo_core::AlphaMode::None) {
                3
            } else {
                4
            };
            for y in 0..p0.height {
                if let Some(row) = p0.row(y) {
                    for px in row.chunks_exact(step) {
                        total_px += 1;
                        if px[0] <= thresh_u8 && px[1] <= thresh_u8 && px[2] <= thresh_u8 {
                            black_px += 1;
                        }
                    }
                }
            }
        } else {
            for y in 0..p0.height {
                if let Some(row) = p0.row(y) {
                    total_px += row.len();
                    black_px += row.iter().filter(|&&b| b <= thresh_u8).count();
                }
            }
        }

        if total_px == 0 {
            return false;
        }
        (black_px as f64 / total_px as f64) >= self.picture_black_ratio_th
    }
}

impl VideoFilter for BlackDetectFilter {
    fn process(&mut self, frame: VideoFrame, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        let is_black = self.is_frame_black(&frame);
        let tb = frame.time_base;
        let pts = frame.pts;
        let end_pts = pts.checked_add(frame.duration.max(1)).unwrap_or(pts);

        if is_black {
            if self.active_start.is_none() {
                self.active_start = Some((pts, tb));
            }
            self.last_black_end = Some((end_pts, tb));
        } else {
            self.maybe_commit_active();
        }
        Ok(Some(frame))
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        self.maybe_commit_active();
        Ok(None)
    }
}

// ============================================================================
// 11. DrawBoxFilter
// ============================================================================

/// VFR-native rectangle drawing filter (`drawbox`) supporting outlined and filled boxes with alpha blending.
#[derive(Debug, Clone)]
pub struct DrawBoxFilter {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Color `[c0, c1, c2, alpha]` (`[Y, U, V, A]` for YCbCr, `[R, G, B, A]` for RGB).
    pub color: [u8; 4],
    pub thickness: u32,
    pub fill: bool,
}

impl DrawBoxFilter {
    pub const fn new(
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        color: [u8; 4],
        thickness: u32,
        fill: bool,
    ) -> Self {
        Self {
            x,
            y,
            width,
            height,
            color,
            thickness,
            fill,
        }
    }

    #[inline]
    fn pixel_in_box(&self, px: u32, py: u32) -> bool {
        let x_end = self.x.saturating_add(self.width);
        let y_end = self.y.saturating_add(self.height);
        if px < self.x || px >= x_end || py < self.y || py >= y_end {
            return false;
        }
        if self.fill {
            return true;
        }
        let t = self.thickness.max(1);
        px < self.x.saturating_add(t)
            || px >= x_end.saturating_sub(t)
            || py < self.y.saturating_add(t)
            || py >= y_end.saturating_sub(t)
    }

    #[inline]
    fn blend_u8(dst: u8, src: u8, alpha: u8) -> u8 {
        let a = alpha as u32;
        let inv = 255 - a;
        ((dst as u32 * inv + src as u32 * a + 127) / 255) as u8
    }
}

impl VideoFilter for DrawBoxFilter {
    fn process(&mut self, mut frame: VideoFrame, _pool: &BufferPool) -> Result<Option<VideoFrame>> {
        if self.width == 0 || self.height == 0 || self.color[3] == 0 {
            return Ok(Some(frame));
        }
        ensure_planes_writable(&mut frame)?;

        let layout = frame.layout;
        let fw = frame.width;
        let fh = frame.height;
        let alpha = self.color[3];
        let Some(planes) = frame.planes_mut() else {
            return Ok(Some(frame));
        };

        match layout.packing {
            PlanePacking::Planar if matches!(layout.depth, SampleDepth::U8) => {
                // Plane 0 (Y / G / Gray)
                if let Some(p0) = planes.get_mut(0) {
                    for py in 0..fh.min(p0.height as u32) {
                        if let Some(row) = p0.row_mut(py as usize) {
                            for px in 0..fw.min(row.len() as u32) {
                                if self.pixel_in_box(px, py) {
                                    row[px as usize] =
                                        Self::blend_u8(row[px as usize], self.color[0], alpha);
                                }
                            }
                        }
                    }
                }
                // Chroma planes (if present)
                let (cw, ch) = layout.subsampling.chroma_dims(fw, fh);
                let sub_x = (fw / cw.max(1)).max(1);
                let sub_y = (fh / ch.max(1)).max(1);
                for (c_idx, comp_val) in [(1usize, self.color[1]), (2usize, self.color[2])] {
                    if let Some(pc) = planes.get_mut(c_idx) {
                        for cy in 0..ch.min(pc.height as u32) {
                            if let Some(row) = pc.row_mut(cy as usize) {
                                for cx in 0..cw.min(row.len() as u32) {
                                    if self.pixel_in_box(cx * sub_x, cy * sub_y) {
                                        row[cx as usize] =
                                            Self::blend_u8(row[cx as usize], comp_val, alpha);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            PlanePacking::Packed if matches!(layout.depth, SampleDepth::U8) => {
                let step = if matches!(layout.alpha, wedeo_core::AlphaMode::None) {
                    3
                } else {
                    4
                };
                let (c0, c1, c2) = match layout.order {
                    CompOrder::Bgr | CompOrder::Bgra => {
                        (self.color[2], self.color[1], self.color[0])
                    }
                    _ => (self.color[0], self.color[1], self.color[2]),
                };
                if let Some(p0) = planes.get_mut(0) {
                    for py in 0..fh.min(p0.height as u32) {
                        if let Some(row) = p0.row_mut(py as usize) {
                            for px in 0..fw {
                                let off = px as usize * step;
                                if off + step <= row.len() && self.pixel_in_box(px, py) {
                                    row[off] = Self::blend_u8(row[off], c0, alpha);
                                    row[off + 1] = Self::blend_u8(row[off + 1], c1, alpha);
                                    row[off + 2] = Self::blend_u8(row[off + 2], c2, alpha);
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(Some(frame))
    }
}
