//! Compiled L1-cache-resident audio execution engine (`CompiledAudioOps`).
//!
//! Executes an optimized [`AudioOpList`] using:
//! - Pre-allocated 1024-sample planar `f32`/`f64` L1-resident slabs
//! - Pre-allocated circular ring buffers for `Reframe` and `SampleTrim::skip_end`
//!   (zero `O(N^2)` shifts, zero steady-state heap allocations)
//! - Exact VFR `(pts, duration, time_base)` tracking across trimming, resampling, and reframing

use std::collections::VecDeque;
use std::sync::Arc;

use wedeo_core::{
    AudioBlock, Buffer, BufferPool, Error, Rational, Result, Rounding, SampleFormat, SampleTrim,
    SideData, StreamHealth, Timestamp, rescale, rescale_q,
};

use crate::dither::{
    DitherState, pack_slab_f32_to_buffers, pack_slab_f32_to_vecs, pack_slab_f64_to_buffers,
    pack_slab_f64_to_vecs, record_clipping, unpack_slab_f32, unpack_slab_f64,
};
use crate::ops::{AudioContract, AudioDither, AudioOp, AudioOpList, ChannelMatrix};
use crate::optimizer;
use crate::resample::PolyphaseResampler;

/// L1-cache-resident planar processing slab length in samples per channel.
pub const AUDIO_SLAB_SAMPLES: usize = 1024;

#[derive(Debug, Clone, Copy)]
struct PtsAnchor {
    at_out_sample: i64,
    pts: Timestamp,
    pts_tb: Rational,
}

#[derive(Debug, Clone)]
struct VfrClockTracker {
    in_rate: u32,
    out_rate: u32,
    out_tb: Option<Rational>,
    in_cursor: i64,
    out_cursor: i64,
    active_anchor: Option<PtsAnchor>,
    anchor_in_sample: i64,
    anchors: VecDeque<PtsAnchor>,
}

impl VfrClockTracker {
    fn new(in_rate: u32, out_rate: u32) -> Self {
        Self {
            in_rate,
            out_rate,
            out_tb: None,
            in_cursor: 0,
            out_cursor: 0,
            active_anchor: None,
            anchor_in_sample: 0,
            anchors: VecDeque::with_capacity(64),
        }
    }

    fn reset(&mut self) {
        self.out_tb = None;
        self.in_cursor = 0;
        self.out_cursor = 0;
        self.active_anchor = None;
        self.anchor_in_sample = 0;
        self.anchors.clear();
    }

    fn observe_input(&mut self, in_pts: Timestamp, in_tb: Rational, in_samples: usize) {
        let est_out = rescale(self.in_cursor, self.out_rate as i64, self.in_rate as i64);

        if in_pts.is_known() && in_tb.is_valid() {
            if self.out_tb.is_none() {
                let in_sample_tb = Rational::new(1, self.in_rate as i32);
                self.out_tb = Some(if self.in_rate != self.out_rate && in_tb == in_sample_tb {
                    Rational::new(1, self.out_rate as i32)
                } else {
                    in_tb
                });
            }

            let is_discontinuous = match self.active_anchor.or_else(|| self.anchors.back().copied())
            {
                None => true,
                Some(ref_anchor) => {
                    let expected_in_ref_tb = ref_anchor.pts.raw()
                        + rescale_q(
                            self.in_cursor - self.anchor_in_sample,
                            Rational::new(1, self.in_rate as i32),
                            ref_anchor.pts_tb,
                        );
                    let expected_in_cur_tb =
                        rescale_q(expected_in_ref_tb, ref_anchor.pts_tb, in_tb);
                    (in_pts.raw() - expected_in_cur_tb).abs() > 1
                }
            };

            if is_discontinuous {
                self.anchors.push_back(PtsAnchor {
                    at_out_sample: est_out,
                    pts: in_pts,
                    pts_tb: in_tb,
                });
                self.anchor_in_sample = self.in_cursor;
            }
        }

        self.in_cursor += in_samples as i64;
    }

    fn next_output_timing(&mut self, out_samples: usize) -> (Timestamp, i64, Rational) {
        while let Some(&front) = self.anchors.front() {
            if front.at_out_sample <= self.out_cursor || self.active_anchor.is_none() {
                self.active_anchor = self.anchors.pop_front();
            } else {
                break;
            }
        }

        let out_sample_tb = Rational::new(1, self.out_rate as i32);
        let tb = self.out_tb.unwrap_or(out_sample_tb);

        let (pts, duration) = if let Some(anchor) = self.active_anchor {
            let base_ticks = anchor
                .pts
                .rescale(anchor.pts_tb, tb, Rounding::NearInf)
                .raw();
            let start_off = rescale_q(self.out_cursor - anchor.at_out_sample, out_sample_tb, tb);
            let end_off = rescale_q(
                self.out_cursor + out_samples as i64 - anchor.at_out_sample,
                out_sample_tb,
                tb,
            );
            (Timestamp::new(base_ticks + start_off), end_off - start_off)
        } else {
            (
                Timestamp::NONE,
                rescale_q(out_samples as i64, out_sample_tb, tb),
            )
        };

        self.out_cursor += out_samples as i64;
        (pts, duration, tb)
    }
}

/// Pre-allocated circular sample ring buffer (planar or interleaved raw bytes).
struct RingSampleBuffer {
    format: SampleFormat,
    channels: usize,
    bps: usize,
    cap_samples: usize,
    head: usize,
    len: usize,
    planes: Vec<Vec<u8>>,
}

impl RingSampleBuffer {
    fn new(format: SampleFormat, channels: usize, cap_samples: usize) -> Self {
        let bps = format.bytes_per_sample();
        let cap = cap_samples.max(1);
        let planes = if format.is_planar() {
            (0..channels).map(|_| vec![0u8; cap * bps]).collect()
        } else {
            vec![vec![0u8; cap * bps * channels]]
        };
        Self {
            format,
            channels,
            bps,
            cap_samples: cap,
            head: 0,
            len: 0,
            planes,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    #[inline]
    fn stride_bytes(&self) -> usize {
        if self.format.is_planar() {
            self.bps
        } else {
            self.bps * self.channels
        }
    }

    fn ensure_free(&mut self, needed: usize) {
        if self.len + needed <= self.cap_samples {
            return;
        }
        let new_cap = (self.len + needed)
            .next_power_of_two()
            .max(self.cap_samples * 2);
        let stride = self.stride_bytes();
        for plane in &mut self.planes {
            let mut next = vec![0u8; new_cap * stride];
            if self.len > 0 {
                let first = (self.cap_samples - self.head).min(self.len);
                let second = self.len - first;
                next[..first * stride]
                    .copy_from_slice(&plane[self.head * stride..(self.head + first) * stride]);
                if second > 0 {
                    next[first * stride..(first + second) * stride]
                        .copy_from_slice(&plane[..second * stride]);
                }
            }
            *plane = next;
        }
        self.head = 0;
        self.cap_samples = new_cap;
    }

    fn push_from_vecs(&mut self, src_vecs: &[Vec<u8>], count: usize) {
        if count == 0 {
            return;
        }
        self.ensure_free(count);
        let stride = self.stride_bytes();
        let tail = (self.head + self.len) % self.cap_samples;
        let first = (self.cap_samples - tail).min(count);
        let second = count - first;

        for (dst, src) in self.planes.iter_mut().zip(src_vecs.iter()) {
            dst[tail * stride..(tail + first) * stride].copy_from_slice(&src[..first * stride]);
            if second > 0 {
                dst[..second * stride]
                    .copy_from_slice(&src[first * stride..(first + second) * stride]);
            }
        }
        self.len += count;
    }

    fn push_from_buffers(&mut self, src_bufs: &[Buffer], start_sample: usize, count: usize) {
        if count == 0 {
            return;
        }
        self.ensure_free(count);
        let stride = self.stride_bytes();
        let tail = (self.head + self.len) % self.cap_samples;
        let first = (self.cap_samples - tail).min(count);
        let second = count - first;
        let start_byte = start_sample * stride;

        for (dst, buf) in self.planes.iter_mut().zip(src_bufs.iter()) {
            let src = &buf.data()[start_byte..start_byte + count * stride];
            dst[tail * stride..(tail + first) * stride].copy_from_slice(&src[..first * stride]);
            if second > 0 {
                dst[..second * stride]
                    .copy_from_slice(&src[first * stride..(first + second) * stride]);
            }
        }
        self.len += count;
    }

    fn pop_to_buffers(&mut self, count: usize, pool: &Arc<BufferPool>) -> Vec<Buffer> {
        let stride = self.stride_bytes();
        let first = (self.cap_samples - self.head).min(count);
        let second = count - first;

        let mut out_bufs = Vec::with_capacity(self.planes.len());
        let byte_len = count * stride;
        for plane in &self.planes {
            let mut buf = pool.acquire_reused(byte_len);
            if let Some(dst) = buf.try_data_mut() {
                dst[..first * stride]
                    .copy_from_slice(&plane[self.head * stride..(self.head + first) * stride]);
                if second > 0 {
                    dst[first * stride..byte_len].copy_from_slice(&plane[..second * stride]);
                }
            }
            out_bufs.push(buf);
        }
        self.head = (self.head + count) % self.cap_samples;
        self.len -= count;
        out_bufs
    }

    fn pop_to_vecs(&mut self, count: usize, dst_vecs: &mut [Vec<u8>]) {
        let stride = self.stride_bytes();
        let first = (self.cap_samples - self.head).min(count);
        let second = count - first;
        for (dst, plane) in dst_vecs.iter_mut().zip(self.planes.iter()) {
            dst[..first * stride]
                .copy_from_slice(&plane[self.head * stride..(self.head + first) * stride]);
            if second > 0 {
                dst[first * stride..count * stride].copy_from_slice(&plane[..second * stride]);
            }
        }
        self.head = (self.head + count) % self.cap_samples;
        self.len -= count;
    }
}

#[derive(Debug, Clone, Copy)]
struct TrimSegment {
    count: usize,
    pts: Timestamp,
    time_base: Rational,
}

struct TrimTailState {
    skip_end: usize,
    ring: RingSampleBuffer,
    segments: VecDeque<TrimSegment>,
    scratch_vecs: Vec<Vec<u8>>,
    src_rate: u32,
}

impl TrimTailState {
    fn new(fmt: SampleFormat, channels: usize, src_rate: u32, skip_end: usize) -> Self {
        let cap = (skip_end + AUDIO_SLAB_SAMPLES * 2).max(2048);
        let bps = fmt.bytes_per_sample();
        let scratch_vecs = if fmt.is_planar() {
            (0..channels)
                .map(|_| vec![0u8; AUDIO_SLAB_SAMPLES * bps])
                .collect()
        } else {
            vec![vec![0u8; AUDIO_SLAB_SAMPLES * bps * channels]]
        };
        Self {
            skip_end,
            ring: RingSampleBuffer::new(fmt, channels, cap),
            segments: VecDeque::with_capacity(32),
            scratch_vecs,
            src_rate,
        }
    }

    fn clear(&mut self) {
        self.ring.clear();
        self.segments.clear();
    }

    fn push_segment(&mut self, count: usize, pts: Timestamp, time_base: Rational) {
        if count == 0 {
            return;
        }
        self.segments.push_back(TrimSegment {
            count,
            pts,
            time_base,
        });
    }

    fn pop_timing(&mut self, mut count: usize) -> (Timestamp, Rational) {
        let mut first_pts = Timestamp::NONE;
        let mut first_tb = Rational::new(1, self.src_rate as i32);
        let mut got_first = false;

        while count > 0 {
            let Some(front) = self.segments.front_mut() else {
                break;
            };
            if !got_first {
                first_pts = front.pts;
                first_tb = front.time_base;
                got_first = true;
            }
            if front.count <= count {
                count -= front.count;
                self.segments.pop_front();
            } else {
                let take = count;
                if let Some(p) = front.pts.get()
                    && front.time_base.is_valid()
                {
                    let delta = rescale_q(
                        take as i64,
                        Rational::new(1, self.src_rate as i32),
                        front.time_base,
                    );
                    front.pts = Timestamp::new(p + delta);
                }
                front.count -= take;
                count = 0;
            }
        }
        (first_pts, first_tb)
    }
}

enum SlabStorage {
    F32 { slab_a: Vec<f32>, slab_b: Vec<f32> },
    F64 { slab_a: Vec<f64>, slab_b: Vec<f64> },
}

impl SlabStorage {
    fn new(use_f64: bool, max_channels: usize) -> Self {
        let len = max_channels.max(1) * AUDIO_SLAB_SAMPLES;
        if use_f64 {
            Self::F64 {
                slab_a: vec![0.0_f64; len],
                slab_b: vec![0.0_f64; len],
            }
        } else {
            Self::F32 {
                slab_a: vec![0.0_f32; len],
                slab_b: vec![0.0_f32; len],
            }
        }
    }
}

/// Compiled, L1-cache-resident, steady-state zero-allocation audio execution pipeline.
pub struct CompiledAudioOps {
    src: AudioContract,
    dst: AudioContract,
    is_noop: bool,
    remaining_skip_start: u64,
    initial_skip_start: u32,
    trim_tail: Option<TrimTailState>,
    pre_rematrix: Option<ChannelMatrix>,
    resampler: Option<PolyphaseResampler>,
    post_rematrix: Option<ChannelMatrix>,
    dither: DitherState,
    reframe_size: Option<usize>,
    reframe_ring: Option<RingSampleBuffer>,
    pack_scratch_vecs: Vec<Vec<u8>>,
    slabs: SlabStorage,
    clock: VfrClockTracker,
    health: StreamHealth,
    last_pts: Timestamp,
}

impl CompiledAudioOps {
    /// Compile an [`AudioOpList`] into an executable `CompiledAudioOps` pipeline.
    pub fn compile(op_list: &AudioOpList) -> Result<Self> {
        op_list.src.validate()?;
        op_list.dst.validate()?;

        let mut ops = op_list.ops.clone();
        optimizer::optimize_ops(&mut ops);
        let is_noop = optimizer::is_noop(&op_list.src, &op_list.dst, &ops);

        let src_ch = op_list.src.channel_layout.nb_channels as usize;
        let dst_ch = op_list.dst.channel_layout.nb_channels as usize;
        let mut max_ch = src_ch.max(dst_ch);

        let mut trim = SampleTrim::default();
        let mut pre_rematrix: Option<ChannelMatrix> = None;
        let mut resample_cfg = None;
        let mut post_rematrix: Option<ChannelMatrix> = None;
        let mut dither_mode = AudioDither::None;
        let mut reframe_size: Option<usize> = None;

        for op in &ops {
            match op {
                AudioOp::Read(_) | AudioOp::Write(_) => {}
                AudioOp::Trim(t) => {
                    trim.skip_start = trim.skip_start.saturating_add(t.skip_start);
                    trim.skip_end = trim.skip_end.saturating_add(t.skip_end);
                }
                AudioOp::Rematrix(m) => {
                    max_ch = max_ch.max(m.in_channels).max(m.out_channels);
                    if resample_cfg.is_none() {
                        pre_rematrix = match pre_rematrix {
                            Some(prev) => Some(prev.then(m)?),
                            None => Some(m.clone()),
                        };
                    } else {
                        post_rematrix = match post_rematrix {
                            Some(prev) => Some(prev.then(m)?),
                            None => Some(m.clone()),
                        };
                    }
                }
                AudioOp::Resample(cfg) => {
                    cfg.validate()?;
                    resample_cfg = Some(*cfg);
                }
                AudioOp::Dither(d) => {
                    dither_mode = *d;
                }
                AudioOp::Reframe { frame_size } => {
                    if *frame_size > 0 {
                        reframe_size = Some(*frame_size as usize);
                    }
                }
            }
        }

        let use_f64 = matches!(
            op_list.src.format,
            SampleFormat::F64
                | SampleFormat::F64Planar
                | SampleFormat::S64
                | SampleFormat::S64Planar
        ) || matches!(
            op_list.dst.format,
            SampleFormat::F64
                | SampleFormat::F64Planar
                | SampleFormat::S64
                | SampleFormat::S64Planar
        );

        let resampler = if let Some(cfg) = resample_cfg {
            let resample_ch = pre_rematrix.as_ref().map_or(src_ch, |m| m.out_channels);
            Some(PolyphaseResampler::new(cfg, resample_ch, use_f64)?)
        } else {
            None
        };

        let trim_tail = if trim.skip_end > 0 {
            Some(TrimTailState::new(
                op_list.src.format,
                src_ch,
                op_list.src.sample_rate,
                trim.skip_end as usize,
            ))
        } else {
            None
        };

        let dst_bps = op_list.dst.format.bytes_per_sample();
        let reframe_ring = reframe_size.map(|fs| {
            let cap = (fs + AUDIO_SLAB_SAMPLES * 2).max(4096);
            RingSampleBuffer::new(op_list.dst.format, dst_ch, cap)
        });

        let pack_scratch_vecs = if reframe_size.is_some() {
            if op_list.dst.format.is_planar() {
                (0..dst_ch)
                    .map(|_| vec![0u8; AUDIO_SLAB_SAMPLES * dst_bps])
                    .collect()
            } else {
                vec![vec![0u8; AUDIO_SLAB_SAMPLES * dst_bps * dst_ch]]
            }
        } else {
            Vec::new()
        };

        Ok(Self {
            src: op_list.src.clone(),
            dst: op_list.dst.clone(),
            is_noop,
            remaining_skip_start: trim.skip_start as u64,
            initial_skip_start: trim.skip_start,
            trim_tail,
            pre_rematrix,
            resampler,
            post_rematrix,
            dither: DitherState::new(dither_mode, dst_ch),
            reframe_size,
            reframe_ring,
            pack_scratch_vecs,
            slabs: SlabStorage::new(use_f64, max_ch),
            clock: VfrClockTracker::new(op_list.src.sample_rate, op_list.dst.sample_rate),
            health: StreamHealth::new(),
            last_pts: Timestamp::NONE,
        })
    }

    #[inline]
    pub fn is_noop(&self) -> bool {
        self.is_noop
    }

    #[inline]
    pub fn src(&self) -> &AudioContract {
        &self.src
    }

    #[inline]
    pub fn dst(&self) -> &AudioContract {
        &self.dst
    }

    #[inline]
    pub fn health(&self) -> &StreamHealth {
        &self.health
    }

    #[inline]
    pub fn resampler(&self) -> Option<&PolyphaseResampler> {
        self.resampler.as_ref()
    }

    #[inline]
    pub fn resampler_mut(&mut self) -> Option<&mut PolyphaseResampler> {
        self.resampler.as_mut()
    }

    /// Reset streaming state (trim, resampler, dither PRNG, reframe ring, VFR clock).
    pub fn reset(&mut self) {
        self.remaining_skip_start = self.initial_skip_start as u64;
        if let Some(tail) = &mut self.trim_tail {
            tail.clear();
        }
        if let Some(r) = &mut self.resampler {
            r.reset();
        }
        if let Some(ring) = &mut self.reframe_ring {
            ring.clear();
        }
        self.dither.reset();
        self.clock.reset();
        self.last_pts = Timestamp::NONE;
    }

    fn validate_input(&self, block: &AudioBlock) -> Result<()> {
        if block.format != self.src.format {
            return Err(Error::invalid_argument(
                "audio_chain",
                format!(
                    "expected input format {}, got {}",
                    self.src.format, block.format
                ),
            ));
        }
        if block.sample_rate != self.src.sample_rate {
            return Err(Error::invalid_argument(
                "audio_chain",
                format!(
                    "expected input sample_rate {}, got {}",
                    self.src.sample_rate, block.sample_rate
                ),
            ));
        }
        if block.channel_layout.nb_channels != self.src.channel_layout.nb_channels {
            return Err(Error::invalid_argument(
                "audio_chain",
                format!(
                    "expected {} input channels, got {}",
                    self.src.channel_layout.nb_channels, block.channel_layout.nb_channels
                ),
            ));
        }
        if block.planes.len() != block.expected_plane_count() {
            return Err(Error::invalid_data(
                "audio_chain",
                "AudioBlock plane count does not match format/channels",
            ));
        }
        let min_bytes = block.bytes_per_plane();
        for p in &block.planes {
            if p.len() < min_bytes {
                return Err(Error::invalid_data(
                    "audio_chain",
                    "AudioBlock plane buffer shorter than nb_samples requires",
                ));
            }
        }
        Ok(())
    }

    /// Process one [`AudioBlock`], appending zero or more output [`AudioBlock`]s to `out`.
    pub fn process_into(
        &mut self,
        block: AudioBlock,
        pool: &Arc<BufferPool>,
        out: &mut Vec<AudioBlock>,
    ) -> Result<()> {
        self.validate_input(&block)?;
        if block.nb_samples == 0 {
            return Ok(());
        }
        if self.is_noop {
            out.push(block);
            return Ok(());
        }

        let mut start_sample = 0usize;
        let mut count = block.nb_samples as usize;
        let mut pts = block.pts;
        let tb = block.time_base;

        if self.remaining_skip_start > 0 {
            let drop_n = (count as u64).min(self.remaining_skip_start) as usize;
            self.remaining_skip_start -= drop_n as u64;
            start_sample += drop_n;
            count -= drop_n;
            if let Some(p) = pts.get()
                && tb.is_valid()
            {
                let delta = rescale_q(
                    drop_n as i64,
                    Rational::new(1, self.src.sample_rate as i32),
                    tb,
                );
                pts = Timestamp::new(p + delta);
            }
        }
        if count == 0 {
            return Ok(());
        }

        self.last_pts = pts;

        if self.trim_tail.is_some() {
            self.process_with_tail_trim(&block, start_sample, count, pts, tb, pool, out)
        } else {
            self.process_post_trim(
                &block.planes,
                start_sample,
                count,
                pts,
                tb,
                &block.side_data,
                pool,
                out,
            )
        }
    }

    /// Convenience wrapper around [`Self::process_into`] returning a newly allocated `Vec`.
    pub fn process(
        &mut self,
        block: AudioBlock,
        pool: &Arc<BufferPool>,
    ) -> Result<Vec<AudioBlock>> {
        let mut out = Vec::new();
        self.process_into(block, pool, &mut out)?;
        Ok(out)
    }

    /// Flush all buffered resampler and reframer samples on end-of-stream.
    pub fn flush_into(&mut self, pool: &Arc<BufferPool>, out: &mut Vec<AudioBlock>) -> Result<()> {
        if let Some(tail) = &mut self.trim_tail {
            tail.clear();
        }

        if let Some(resampler) = &mut self.resampler {
            resampler.flush();
        }
        while self.resampler.is_some() {
            let drained = drain_resampler_into_slab(
                self.resampler.as_mut().expect("checked Some"),
                &mut self.slabs,
                AUDIO_SLAB_SAMPLES,
            )?;
            if drained == 0 {
                break;
            }
            self.emit_post_resample_slab(drained, &[], pool, out)?;
        }

        if let Some(fs) = self.reframe_size {
            while let Some(ring) = &self.reframe_ring {
                if ring.len() < fs {
                    break;
                }
                self.pop_reframe_block(fs, &[], pool, out);
            }
            let rem = self.reframe_ring.as_ref().map_or(0, RingSampleBuffer::len);
            if rem > 0 {
                self.pop_reframe_block(rem, &[], pool, out);
            }
        }

        Ok(())
    }

    /// Convenience wrapper around [`Self::flush_into`] returning a newly allocated `Vec`.
    pub fn flush(&mut self, pool: &Arc<BufferPool>) -> Result<Vec<AudioBlock>> {
        let mut out = Vec::new();
        self.flush_into(pool, &mut out)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn process_with_tail_trim(
        &mut self,
        block: &AudioBlock,
        start_sample: usize,
        count: usize,
        pts: Timestamp,
        tb: Rational,
        pool: &Arc<BufferPool>,
        out: &mut Vec<AudioBlock>,
    ) -> Result<()> {
        let mut tail = self.trim_tail.take().expect("trim_tail is Some");
        let res = (|| -> Result<()> {
            let total = tail.ring.len() + count;
            let releasable = total.saturating_sub(tail.skip_end);
            let from_tail = releasable.min(tail.ring.len());
            let direct_count = releasable - from_tail;

            let mut rem_from_tail = from_tail;
            while rem_from_tail > 0 {
                let chunk = rem_from_tail.min(AUDIO_SLAB_SAMPLES);
                tail.ring.pop_to_vecs(chunk, &mut tail.scratch_vecs);
                let (seg_pts, seg_tb) = tail.pop_timing(chunk);
                self.process_post_trim_vecs(
                    &tail.scratch_vecs,
                    chunk,
                    seg_pts,
                    seg_tb,
                    &block.side_data,
                    pool,
                    out,
                )?;
                rem_from_tail -= chunk;
            }

            if direct_count > 0 {
                self.process_post_trim(
                    &block.planes,
                    start_sample,
                    direct_count,
                    pts,
                    tb,
                    &block.side_data,
                    pool,
                    out,
                )?;
            }

            let push_count = count - direct_count;
            if push_count > 0 {
                let push_start = start_sample + direct_count;
                let push_pts = if let Some(p) = pts.get()
                    && tb.is_valid()
                {
                    let delta = rescale_q(
                        direct_count as i64,
                        Rational::new(1, self.src.sample_rate as i32),
                        tb,
                    );
                    Timestamp::new(p + delta)
                } else {
                    pts
                };
                tail.ring
                    .push_from_buffers(&block.planes, push_start, push_count);
                tail.push_segment(push_count, push_pts, tb);
            }
            Ok(())
        })();
        self.trim_tail = Some(tail);
        res
    }

    #[allow(clippy::too_many_arguments)]
    fn process_post_trim_vecs(
        &mut self,
        src_vecs: &[Vec<u8>],
        count: usize,
        pts: Timestamp,
        tb: Rational,
        side_data: &[SideData],
        pool: &Arc<BufferPool>,
        out: &mut Vec<AudioBlock>,
    ) -> Result<()> {
        let src_ch = self.src.channel_layout.nb_channels as usize;
        let src_fmt = self.src.format;
        self.clock.observe_input(pts, tb, count);

        unpack_vecs_into_slab(src_vecs, src_fmt, src_ch, count, &mut self.slabs)?;
        self.run_slabs_on_chunk(count, side_data, pool, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn process_post_trim(
        &mut self,
        planes: &[Buffer],
        start_sample: usize,
        count: usize,
        pts: Timestamp,
        tb: Rational,
        side_data: &[SideData],
        pool: &Arc<BufferPool>,
        out: &mut Vec<AudioBlock>,
    ) -> Result<()> {
        self.clock.observe_input(pts, tb, count);

        // Fast non-resampling, non-reframing path: still executes in L1 1024-sample slabs,
        // writing directly into a single pooled output AudioBlock.
        if self.resampler.is_none() && self.reframe_ring.is_none() {
            let dst_ch = self.dst.channel_layout.nb_channels as usize;
            let dst_fmt = self.dst.format;
            let bps = dst_fmt.bytes_per_sample();
            let mut dst_bufs = if dst_fmt.is_planar() {
                let bytes = count * bps;
                (0..dst_ch).map(|_| pool.acquire_reused(bytes)).collect()
            } else {
                vec![pool.acquire_reused(count * bps * dst_ch)]
            };

            let src_ch = self.src.channel_layout.nb_channels as usize;
            let src_fmt = self.src.format;
            let mut offset = 0usize;
            while offset < count {
                let chunk = (count - offset).min(AUDIO_SLAB_SAMPLES);
                unpack_buffers_into_slab(
                    planes,
                    src_fmt,
                    src_ch,
                    start_sample + offset,
                    chunk,
                    &mut self.slabs,
                )?;
                let rematrix_ref = self.pre_rematrix.as_ref().or(self.post_rematrix.as_ref());
                let out_in_b = apply_rematrix_on_slabs(rematrix_ref, &mut self.slabs, chunk);
                let clipped = pack_slab_into_buffers(
                    &self.slabs,
                    out_in_b,
                    dst_ch,
                    chunk,
                    dst_fmt,
                    &mut self.dither,
                    &mut dst_bufs,
                    offset,
                );
                record_clipping(&mut self.health, self.last_pts, clipped);
                offset += chunk;
            }

            let (out_pts, out_dur, out_tb) = self.clock.next_output_timing(count);
            out.push(AudioBlock {
                planes: dst_bufs,
                nb_samples: count as u32,
                format: self.dst.format,
                sample_rate: self.dst.sample_rate,
                channel_layout: self.dst.channel_layout.clone(),
                pts: out_pts,
                duration: out_dur,
                time_base: out_tb,
                side_data: filter_side_data(side_data),
            });
            return Ok(());
        }

        // General streaming path in 1024-sample L1 slabs (with resampler and/or reframe ring).
        let src_ch = self.src.channel_layout.nb_channels as usize;
        let src_fmt = self.src.format;
        let mut offset = 0usize;
        while offset < count {
            let chunk = (count - offset).min(AUDIO_SLAB_SAMPLES);
            unpack_buffers_into_slab(
                planes,
                src_fmt,
                src_ch,
                start_sample + offset,
                chunk,
                &mut self.slabs,
            )?;
            self.run_slabs_on_chunk(chunk, side_data, pool, out)?;
            offset += chunk;
        }
        Ok(())
    }

    fn run_slabs_on_chunk(
        &mut self,
        chunk: usize,
        side_data: &[SideData],
        pool: &Arc<BufferPool>,
        out: &mut Vec<AudioBlock>,
    ) -> Result<()> {
        let out_in_b = apply_rematrix_on_slabs(self.pre_rematrix.as_ref(), &mut self.slabs, chunk);

        if self.resampler.is_some() {
            push_slab_to_resampler(
                self.resampler.as_mut().expect("checked Some"),
                &self.slabs,
                out_in_b,
                chunk,
            )?;
            loop {
                let drained = drain_resampler_into_slab(
                    self.resampler.as_mut().expect("checked Some"),
                    &mut self.slabs,
                    AUDIO_SLAB_SAMPLES,
                )?;
                if drained == 0 {
                    break;
                }
                self.emit_post_resample_slab(drained, side_data, pool, out)?;
            }
        } else {
            // If pre_rematrix left result in slab_b, move/handle post_rematrix cleanly.
            let final_in_b = if let Some(post) = &self.post_rematrix {
                if out_in_b {
                    apply_rematrix_b_to_a(post, &mut self.slabs, chunk);
                    false
                } else {
                    apply_rematrix_a_to_b(post, &mut self.slabs, chunk);
                    true
                }
            } else {
                out_in_b
            };
            self.write_slab_to_output_or_ring(final_in_b, chunk, side_data, pool, out);
        }

        Ok(())
    }

    fn emit_post_resample_slab(
        &mut self,
        count: usize,
        side_data: &[SideData],
        pool: &Arc<BufferPool>,
        out: &mut Vec<AudioBlock>,
    ) -> Result<()> {
        // Resampler drains into slab_a (`in_b == false`).
        let final_in_b =
            apply_rematrix_on_slabs(self.post_rematrix.as_ref(), &mut self.slabs, count);
        self.write_slab_to_output_or_ring(final_in_b, count, side_data, pool, out);
        Ok(())
    }

    fn write_slab_to_output_or_ring(
        &mut self,
        use_slab_b: bool,
        count: usize,
        side_data: &[SideData],
        pool: &Arc<BufferPool>,
        out: &mut Vec<AudioBlock>,
    ) {
        let dst_ch = self.dst.channel_layout.nb_channels as usize;
        let dst_fmt = self.dst.format;

        if let Some(fs) = self.reframe_size {
            let clipped = pack_slab_into_vecs(
                &self.slabs,
                use_slab_b,
                dst_ch,
                count,
                dst_fmt,
                &mut self.dither,
                &mut self.pack_scratch_vecs,
            );
            record_clipping(&mut self.health, self.last_pts, clipped);

            if let Some(ring) = &mut self.reframe_ring {
                ring.push_from_vecs(&self.pack_scratch_vecs, count);
            }
            while let Some(ring) = &self.reframe_ring {
                if ring.len() < fs {
                    break;
                }
                self.pop_reframe_block(fs, side_data, pool, out);
            }
        } else {
            let bps = dst_fmt.bytes_per_sample();
            let mut dst_bufs = if dst_fmt.is_planar() {
                let bytes = count * bps;
                (0..dst_ch).map(|_| pool.acquire_reused(bytes)).collect()
            } else {
                vec![pool.acquire_reused(count * bps * dst_ch)]
            };
            let clipped = pack_slab_into_buffers(
                &self.slabs,
                use_slab_b,
                dst_ch,
                count,
                dst_fmt,
                &mut self.dither,
                &mut dst_bufs,
                0,
            );
            record_clipping(&mut self.health, self.last_pts, clipped);

            let (out_pts, out_dur, out_tb) = self.clock.next_output_timing(count);
            out.push(AudioBlock {
                planes: dst_bufs,
                nb_samples: count as u32,
                format: self.dst.format,
                sample_rate: self.dst.sample_rate,
                channel_layout: self.dst.channel_layout.clone(),
                pts: out_pts,
                duration: out_dur,
                time_base: out_tb,
                side_data: filter_side_data(side_data),
            });
        }
    }

    fn pop_reframe_block(
        &mut self,
        count: usize,
        side_data: &[SideData],
        pool: &Arc<BufferPool>,
        out: &mut Vec<AudioBlock>,
    ) {
        let Some(ring) = &mut self.reframe_ring else {
            return;
        };
        let planes = ring.pop_to_buffers(count, pool);
        let (out_pts, out_dur, out_tb) = self.clock.next_output_timing(count);
        out.push(AudioBlock {
            planes,
            nb_samples: count as u32,
            format: self.dst.format,
            sample_rate: self.dst.sample_rate,
            channel_layout: self.dst.channel_layout.clone(),
            pts: out_pts,
            duration: out_dur,
            time_base: out_tb,
            side_data: filter_side_data(side_data),
        });
    }
}

fn filter_side_data(side_data: &[SideData]) -> Vec<SideData> {
    if side_data.is_empty() {
        return Vec::new();
    }
    side_data
        .iter()
        .filter(|sd| !matches!(sd, SideData::SampleTrim { .. }))
        .cloned()
        .collect()
}

fn unpack_buffers_into_slab(
    planes: &[Buffer],
    fmt: SampleFormat,
    channels: usize,
    sample_offset: usize,
    count: usize,
    slabs: &mut SlabStorage,
) -> Result<()> {
    match slabs {
        SlabStorage::F32 { slab_a, .. } => unpack_slab_f32(
            planes,
            fmt,
            channels,
            sample_offset,
            count,
            slab_a,
            AUDIO_SLAB_SAMPLES,
        ),
        SlabStorage::F64 { slab_a, .. } => unpack_slab_f64(
            planes,
            fmt,
            channels,
            sample_offset,
            count,
            slab_a,
            AUDIO_SLAB_SAMPLES,
        ),
    }
}

fn unpack_vecs_into_slab(
    src_vecs: &[Vec<u8>],
    fmt: SampleFormat,
    channels: usize,
    count: usize,
    slabs: &mut SlabStorage,
) -> Result<()> {
    let bps = fmt.bytes_per_sample();
    match slabs {
        SlabStorage::F32 { slab_a, .. } => {
            if fmt.is_planar() {
                for (ch, src) in src_vecs.iter().enumerate().take(channels) {
                    let dst = &mut slab_a[ch * AUDIO_SLAB_SAMPLES..ch * AUDIO_SLAB_SAMPLES + count];
                    unpack_bytes_f32(&src[..count * bps], fmt, dst);
                }
            } else {
                let stride = bps * channels;
                let src = &src_vecs[0][..count * stride];
                for i in 0..count {
                    let frame = &src[i * stride..(i + 1) * stride];
                    for ch in 0..channels {
                        let b = &frame[ch * bps..(ch + 1) * bps];
                        slab_a[ch * AUDIO_SLAB_SAMPLES + i] = unpack_one_f64(b, fmt) as f32;
                    }
                }
            }
        }
        SlabStorage::F64 { slab_a, .. } => {
            if fmt.is_planar() {
                for (ch, src) in src_vecs.iter().enumerate().take(channels) {
                    let dst = &mut slab_a[ch * AUDIO_SLAB_SAMPLES..ch * AUDIO_SLAB_SAMPLES + count];
                    unpack_bytes_f64(&src[..count * bps], fmt, dst);
                }
            } else {
                let stride = bps * channels;
                let src = &src_vecs[0][..count * stride];
                for i in 0..count {
                    let frame = &src[i * stride..(i + 1) * stride];
                    for ch in 0..channels {
                        let b = &frame[ch * bps..(ch + 1) * bps];
                        slab_a[ch * AUDIO_SLAB_SAMPLES + i] = unpack_one_f64(b, fmt);
                    }
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn unpack_one_f64(b: &[u8], fmt: SampleFormat) -> f64 {
    match fmt {
        SampleFormat::U8 | SampleFormat::U8Planar => (b[0] as i32 - 0x80) as f64 / 128.0,
        SampleFormat::S16 | SampleFormat::S16Planar => {
            i16::from_ne_bytes([b[0], b[1]]) as f64 / 32768.0
        }
        SampleFormat::S32 | SampleFormat::S32Planar => {
            i32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f64 / 2_147_483_648.0
        }
        SampleFormat::S64 | SampleFormat::S64Planar => {
            i64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f64
                / 9_223_372_036_854_775_808.0
        }
        SampleFormat::F32 | SampleFormat::F32Planar => {
            f32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f64
        }
        SampleFormat::F64 | SampleFormat::F64Planar => {
            f64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        }
    }
}

fn unpack_bytes_f32(src: &[u8], fmt: SampleFormat, dst: &mut [f32]) {
    let bps = fmt.bytes_per_sample();
    for (d, c) in dst.iter_mut().zip(src.chunks_exact(bps)) {
        *d = unpack_one_f64(c, fmt) as f32;
    }
}

fn unpack_bytes_f64(src: &[u8], fmt: SampleFormat, dst: &mut [f64]) {
    let bps = fmt.bytes_per_sample();
    for (d, c) in dst.iter_mut().zip(src.chunks_exact(bps)) {
        *d = unpack_one_f64(c, fmt);
    }
}

fn apply_rematrix_on_slabs(
    matrix: Option<&ChannelMatrix>,
    slabs: &mut SlabStorage,
    count: usize,
) -> bool {
    let Some(m) = matrix else {
        return false;
    };
    apply_rematrix_a_to_b(m, slabs, count);
    true
}

fn apply_rematrix_a_to_b(m: &ChannelMatrix, slabs: &mut SlabStorage, count: usize) {
    match slabs {
        SlabStorage::F32 { slab_a, slab_b } => rematrix_slab_f32(m, slab_a, slab_b, count),
        SlabStorage::F64 { slab_a, slab_b } => rematrix_slab_f64(m, slab_a, slab_b, count),
    }
}

fn apply_rematrix_b_to_a(m: &ChannelMatrix, slabs: &mut SlabStorage, count: usize) {
    match slabs {
        SlabStorage::F32 { slab_a, slab_b } => rematrix_slab_f32(m, slab_b, slab_a, count),
        SlabStorage::F64 { slab_a, slab_b } => rematrix_slab_f64(m, slab_b, slab_a, count),
    }
}

fn rematrix_slab_f32(m: &ChannelMatrix, src: &[f32], dst: &mut [f32], count: usize) {
    for o in 0..m.out_channels {
        let out_slice = &mut dst[o * AUDIO_SLAB_SAMPLES..o * AUDIO_SLAB_SAMPLES + count];
        out_slice.fill(0.0);
        for i in 0..m.in_channels {
            let coeff = m.get(o, i) as f32;
            if coeff == 0.0 {
                continue;
            }
            let in_slice = &src[i * AUDIO_SLAB_SAMPLES..i * AUDIO_SLAB_SAMPLES + count];
            if coeff == 1.0 {
                for (d, &s) in out_slice.iter_mut().zip(in_slice) {
                    *d += s;
                }
            } else {
                for (d, &s) in out_slice.iter_mut().zip(in_slice) {
                    *d += s * coeff;
                }
            }
        }
    }
}

fn rematrix_slab_f64(m: &ChannelMatrix, src: &[f64], dst: &mut [f64], count: usize) {
    for o in 0..m.out_channels {
        let out_slice = &mut dst[o * AUDIO_SLAB_SAMPLES..o * AUDIO_SLAB_SAMPLES + count];
        out_slice.fill(0.0);
        for i in 0..m.in_channels {
            let coeff = m.get(o, i);
            if coeff == 0.0 {
                continue;
            }
            let in_slice = &src[i * AUDIO_SLAB_SAMPLES..i * AUDIO_SLAB_SAMPLES + count];
            if coeff == 1.0 {
                for (d, &s) in out_slice.iter_mut().zip(in_slice) {
                    *d += s;
                }
            } else {
                for (d, &s) in out_slice.iter_mut().zip(in_slice) {
                    *d += s * coeff;
                }
            }
        }
    }
}

fn push_slab_to_resampler(
    resampler: &mut PolyphaseResampler,
    slabs: &SlabStorage,
    use_slab_b: bool,
    count: usize,
) -> Result<()> {
    match slabs {
        SlabStorage::F32 { slab_a, slab_b } => {
            let src = if use_slab_b { slab_b } else { slab_a };
            resampler.push_f32_slab(src, AUDIO_SLAB_SAMPLES, count)
        }
        SlabStorage::F64 { slab_a, slab_b } => {
            let src = if use_slab_b { slab_b } else { slab_a };
            resampler.push_f64_slab(src, AUDIO_SLAB_SAMPLES, count)
        }
    }
}

fn drain_resampler_into_slab(
    resampler: &mut PolyphaseResampler,
    slabs: &mut SlabStorage,
    max_out: usize,
) -> Result<usize> {
    match slabs {
        SlabStorage::F32 { slab_a, .. } => {
            resampler.drain_f32_slab(slab_a, AUDIO_SLAB_SAMPLES, max_out)
        }
        SlabStorage::F64 { slab_a, .. } => {
            resampler.drain_f64_slab(slab_a, AUDIO_SLAB_SAMPLES, max_out)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_slab_into_buffers(
    slabs: &SlabStorage,
    use_slab_b: bool,
    channels: usize,
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_buffers: &mut [Buffer],
    dst_sample_offset: usize,
) -> u64 {
    match slabs {
        SlabStorage::F32 { slab_a, slab_b } => {
            let src = if use_slab_b { slab_b } else { slab_a };
            pack_slab_f32_to_buffers(
                src,
                AUDIO_SLAB_SAMPLES,
                channels,
                count,
                fmt,
                dither,
                dst_buffers,
                dst_sample_offset,
            )
        }
        SlabStorage::F64 { slab_a, slab_b } => {
            let src = if use_slab_b { slab_b } else { slab_a };
            pack_slab_f64_to_buffers(
                src,
                AUDIO_SLAB_SAMPLES,
                channels,
                count,
                fmt,
                dither,
                dst_buffers,
                dst_sample_offset,
            )
        }
    }
}

fn pack_slab_into_vecs(
    slabs: &SlabStorage,
    use_slab_b: bool,
    channels: usize,
    count: usize,
    fmt: SampleFormat,
    dither: &mut DitherState,
    dst_vecs: &mut [Vec<u8>],
) -> u64 {
    match slabs {
        SlabStorage::F32 { slab_a, slab_b } => {
            let src = if use_slab_b { slab_b } else { slab_a };
            pack_slab_f32_to_vecs(
                src,
                AUDIO_SLAB_SAMPLES,
                channels,
                count,
                fmt,
                dither,
                dst_vecs,
            )
        }
        SlabStorage::F64 { slab_a, slab_b } => {
            let src = if use_slab_b { slab_b } else { slab_a };
            pack_slab_f64_to_vecs(
                src,
                AUDIO_SLAB_SAMPLES,
                channels,
                count,
                fmt,
                dither,
                dst_vecs,
            )
        }
    }
}
