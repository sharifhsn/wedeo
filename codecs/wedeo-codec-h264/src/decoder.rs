#![forbid(unsafe_code)]
// H.264/AVC decoder.
//
// Implements NAL unit parsing (SPS, PPS, slice header) and macroblock-level
// decoding for I-frames via CAVLC, intra prediction, dequantization, IDCT,
// and in-loop deblocking.
//
// Reference: FFmpeg libavcodec/h264dec.c, h264_slice.c

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use wedeo_core::{
    BorderPadding, BorderedPlane, CodecContext, CodecId, ColorDesc, CropRect, Decoder,
    DecoderDescriptor, DecoderFactory, Error, ErrorPolicy, FrameFlags, HealthEvent, HealthIssue,
    MediaType, MediaUnit, Packet, PictureType, PixelLayout, Rational, RecvStatus, Registry, Result,
    SendStatus, StreamHealth, Timestamp, TrackDescriptor, TrackId, VideoFrame,
};
use wedeo_syntax::h264::{nal_to_rbsp, parse_avcc};

use crate::bitstream::{BitRead, BitReadBE, get_ue_golomb};
use crate::deblock::{self, PictureBuffer};
use crate::dpb::{Dpb, DpbEntry, RefStatus};
use crate::mb::{self, FrameDecodeContext};
use crate::nal::{NalUnit, NalUnitType, split_annex_b, split_nalff};
use crate::pps::{Pps, parse_pps};
use crate::refs;
use crate::sei;
use crate::shared_picture::{BufferPool, SharedPicture};
use crate::slice::{SliceHeader, SliceType, parse_slice_header};
use crate::sps::{Sps, parse_sps};
use crate::thread_pool::{DecodeThreadPool, FrameWork, InFlightDecode, SliceWorkUnit};

#[inline]
fn invalid_data(msg: &'static str) -> Error {
    Error::invalid_data("h264", msg)
}

// ---------------------------------------------------------------------------
// Threading configuration
// ---------------------------------------------------------------------------

/// Resolve number of frame-decode worker threads from `CodecContext::threads`
/// (`0` = auto, `1` = inline with no worker threads spawned).
fn resolve_thread_count(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 4))
            .unwrap_or(2)
    } else {
        threads.clamp(1, 4)
    }
}

// ---------------------------------------------------------------------------
// Decoder state
// ---------------------------------------------------------------------------

/// H.264/AVC decoder.
///
/// Parses SPS, PPS, SEI, and slice headers from NAL units and decodes
/// progressive/MBAFF 8-bit 4:2:0 streams via CAVLC/CABAC, intra prediction,
/// motion compensation, dequantization, IDCT, and in-loop deblocking.
pub struct H264Decoder {
    /// Stored Sequence Parameter Sets, indexed by sps_id (0..31).
    sps_list: [Option<Sps>; 32],
    /// Stored Picture Parameter Sets, indexed by pps_id (0..255).
    pps_list: Box<[Option<Pps>; 256]>,
    /// Decoded display width (from active SPS, after cropping).
    width: u32,
    /// Decoded display height (from active SPS, after cropping).
    height: u32,
    /// Active coded luma width (macroblock-aligned).
    coded_width: u32,
    /// Active coded luma height (macroblock-aligned).
    coded_height: u32,
    /// Active SPS crop margins inside the coded planes.
    crop: CropRect,
    /// Active VUI color description.
    color: ColorDesc,
    /// Active VUI sample aspect ratio.
    sar: Rational,
    /// Track identifier for diagnostics.
    track_id: TrackId,
    /// Corruption handling policy (`Resilient` vs `Strict`).
    error_policy: ErrorPolicy,
    /// Cumulative stream health ledger.
    health: StreamHealth,
    /// Deferred fatal slice decode error (only raised under `ErrorPolicy::Strict`).
    fatal_error: Option<Error>,
    /// True if a `recovery_point` SEI precedes the next coded picture.
    pending_recovery_point: bool,
    /// Shared `wedeo-core` buffer pool for output `VideoFrame` planes.
    core_pool: Arc<wedeo_core::BufferPool>,
    /// Worker thread budget resolved from `CodecContext::threads`.
    threads: usize,
    /// Running frame counter for debug/tracking.
    frame_num: u64,
    /// Queue of decoded frames awaiting output.
    output_queue: VecDeque<VideoFrame>,
    /// True once send_packet(None) has been called (drain mode).
    draining: bool,
    /// NALFF length size for MP4/avcC-style streams (0 = Annex B).
    nalff_length_size: u8,
    /// In-progress frame decode context for multi-slice frames.
    current_fdc: Option<FrameDecodeContext>,
    /// Last slice header for the current frame (for deblocking parameters).
    current_last_hdr: Option<SliceHeader>,
    /// Buffered slices for the current frame (decoded on worker thread).
    pending_slices: Vec<SliceWorkUnit>,
    /// Queue of in-flight frame metadata (dispatch order).
    /// Non-ref B-frames accumulate here; ref frames trigger join-all.
    in_flight_queue: VecDeque<InFlightFrame>,
    /// Persistent thread pool for frame decode workers.
    pool: DecodeThreadPool,
    /// Monotonic counter for FIFO result ordering.
    next_dispatch_id: u64,
    /// Buffer for out-of-order results from the pool.
    result_buffer: HashMap<u64, Box<InFlightDecode>>,
    /// Pool of reusable PictureBuffers to avoid per-frame malloc/free.
    buffer_pool: Arc<Mutex<BufferPool>>,
    /// Timing of the current frame being decoded.
    current_pts: Timestamp,
    current_dts: Timestamp,
    current_duration: i64,
    current_time_base: Rational,
    /// Decoded Picture Buffer for reference picture management.
    dpb: Dpb,
    /// Reference list 0 (DPB indices) for the current slice.
    ref_list_l0: Vec<usize>,
    /// Reference list 1 (DPB indices) for B-slices.
    ref_list_l1: Vec<usize>,
    /// Whether the current NAL is an IDR.
    current_is_idr: bool,
    /// Whether the current picture is a random-access keyframe (IDR or SEI recovery point).
    current_is_key: bool,
    /// frame_num from the current slice header.
    current_frame_num_h264: u32,
    /// POC type 0 state: previous reference picture's PicOrderCntMsb.
    prev_poc_msb: i32,
    /// POC type 0 state: previous reference picture's pic_order_cnt_lsb.
    prev_poc_lsb: u32,
    /// Computed POC for the current picture.
    current_poc: i32,
    /// nal_ref_idc of the current picture (non-zero = reference).
    current_nal_ref_idc: u8,
    /// POC type 1/2 state: frame_num_offset for wrap detection.
    frame_num_offset: i32,
    /// POC type 1/2 state: previous frame_num_offset.
    prev_frame_num_offset: i32,
    /// POC type 1/2 state: previous frame_num (H.264 level).
    prev_frame_num_h264: u32,
    /// Reorder buffer depth (matching FFmpeg's `has_b_frames`).
    /// Starts at `num_reorder_frames` from VUI, or 0 if not signalled.
    /// Dynamically increased when B-frames or out-of-order POCs are detected.
    reorder_depth: usize,
    /// True if VUI bitstream_restriction is present (num_reorder_frames >= 0).
    /// When set, reorder_depth is NOT dynamically increased beyond the VUI value
    /// (matching FFmpeg h264_slice.c:1328 `!sps->bitstream_restriction_flag`).
    has_bitstream_restriction: bool,
    /// Sorted ascending history of recent POCs for out-of-order detection
    /// (matching FFmpeg's `last_pocs` array in h264_slice.c).
    last_pocs: [i64; 16],
    /// Delayed picture buffer for POC-ordered output (matching FFmpeg's
    /// `delayed_pic`). Frames are inserted here and the minimum-POC frame
    /// is output when `delayed_pics.len() > reorder_depth`.
    /// The bool flag is `mmco_reset` — set when MMCO-5 or out_of_order==16
    /// causes a POC sequence restart. Acts as a barrier in min-POC search.
    delayed_pics: Vec<(i32, VideoFrame, bool)>, // (poc, frame, mmco_reset)
}

// ---------------------------------------------------------------------------
// Frame-level threading types
// ---------------------------------------------------------------------------

/// Tracks an in-flight frame submitted to the thread pool.
/// No longer holds a JoinHandle — the pool manages threads.
struct InFlightFrame {
    /// Monotonic dispatch ID for FIFO result ordering.
    sequence_id: u64,
    /// Pre-computed output ordering data (from pre-dispatch state update).
    frame_mmco_reset: bool,
    out_of_order: usize,
}

// Compile-time check: types must be Send for cross-thread dispatch.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<SliceWorkUnit>();
    assert_send::<FrameDecodeContext>();
};

/// Inline deblock for a completed progressive MB row.
///
/// Deblocks the PREVIOUS row (Y-1), not the just-completed row (Y).
/// This preserves intra prediction correctness: row Y's decode read
/// pre-deblock pixels from row Y-1. It's now safe to deblock Y-1
/// since no future decode will read its pre-deblock state.
///
/// After deblocking row Y-1, row Y-2 is finalized (row Y-1's top edge
/// filter was the last write to row Y-2). So publish row Y-2.
///
/// When deblock is disabled: publish the completed row immediately.
fn inline_deblock_row(
    fdc: &mut FrameDecodeContext,
    completed_row: u32,
    mb_width: u32,
    deblock_enabled: bool,
) {
    // Multi-slice parallel decode: skip inline deblock entirely.
    // A serial deblock pass runs after all slices complete.
    if fdc.postpone_deblock {
        return;
    }
    if deblock_enabled {
        if completed_row > 0 {
            deblock::deblock_row(
                &mut fdc.pic,
                &fdc.mb_info,
                &fdc.slice_table,
                &fdc.slice_deblock_params,
                completed_row - 1,
                mb_width,
            );
            if completed_row > 1 {
                fdc.pic.publish_row((completed_row - 2) as i32);
            }
        }
    } else {
        fdc.pic.publish_row(completed_row as i32);
    }
}

/// Inline deblock for a completed MBAFF pair row.
///
/// Same deferred-by-1 principle: deblocks the PREVIOUS pair row, then
/// publishes the pair row before that.
fn inline_deblock_pair_row(
    fdc: &mut FrameDecodeContext,
    pair_row: u32,
    mb_width: u32,
    deblock_enabled: bool,
) {
    if fdc.postpone_deblock {
        return;
    }
    if deblock_enabled {
        if pair_row > 0 {
            deblock::deblock_row_mbaff(
                &mut fdc.pic,
                &fdc.mb_info,
                &fdc.slice_table,
                &fdc.slice_deblock_params,
                pair_row - 1,
                mb_width,
            );
            if pair_row > 1 {
                fdc.pic.publish_row(((pair_row - 1) * 2 - 1) as i32);
            }
        }
    } else {
        fdc.pic.publish_row((pair_row * 2 + 1) as i32);
    }
}

/// Compute intra4x4 right-column modes as i8 for CABAC neighbor storage.
/// Mirrors the mode array computed in `apply_macroblock` for `NeighborContext`.
fn mb_intra4x4_modes_i8(mb: &crate::cavlc::Macroblock, constrained_intra: bool) -> [i8; 16] {
    if mb.is_intra4x4 {
        let mut modes = [-1i8; 16];
        for (i, mode) in modes.iter_mut().enumerate() {
            *mode = mb.intra4x4_pred_mode[i] as i8;
        }
        modes
    } else if mb.is_intra {
        [2i8; 16] // DC_PRED for I_16x16, I_PCM
    } else if constrained_intra {
        [-1i8; 16] // unavailable for inter when constrained_intra_pred
    } else {
        [2i8; 16] // DC_PRED for inter (matches FFmpeg fill_decode_caches)
    }
}

impl H264Decoder {
    /// Open a new `H264Decoder` from a `TrackDescriptor` and `CodecContext`.
    pub fn open(track: &TrackDescriptor, ctx: CodecContext) -> Result<Self> {
        let video = track.params.video();
        let (init_w, init_h, init_color, init_sar, init_crop, extradata) = match video {
            Some(v) => (
                v.width.max(16),
                v.height.max(16),
                v.color,
                if v.sar.num > 0 && v.sar.den > 0 {
                    v.sar
                } else {
                    Rational::new(1, 1)
                },
                v.crop,
                v.extradata.as_slice(),
            ),
            None => (
                16,
                16,
                ColorDesc::UNSPECIFIED,
                Rational::new(1, 1),
                CropRect::ZERO,
                &[][..],
            ),
        };

        let threads = resolve_thread_count(ctx.threads);
        let mut decoder = Self {
            sps_list: Default::default(),
            pps_list: Box::new(std::array::from_fn(|_| None)),
            width: init_w,
            height: init_h,
            coded_width: init_w.div_ceil(16) * 16,
            coded_height: init_h.div_ceil(16) * 16,
            crop: init_crop,
            color: init_color,
            sar: init_sar,
            track_id: track.id,
            error_policy: ctx.error_policy,
            health: StreamHealth::new(),
            fatal_error: None,
            pending_recovery_point: false,
            core_pool: ctx.pool,
            threads,
            frame_num: 0,
            output_queue: VecDeque::new(),
            draining: false,
            nalff_length_size: 0,
            current_fdc: None,
            current_last_hdr: None,
            pending_slices: Vec::new(),
            in_flight_queue: VecDeque::new(),
            pool: DecodeThreadPool::new(threads),
            next_dispatch_id: 0,
            result_buffer: HashMap::new(),
            buffer_pool: Arc::new(Mutex::new(BufferPool::new())),
            current_pts: Timestamp::NONE,
            current_dts: Timestamp::NONE,
            current_duration: 0,
            current_time_base: track.time_base,
            dpb: Dpb::new(16),
            ref_list_l0: Vec::new(),
            ref_list_l1: Vec::new(),
            current_is_idr: false,
            current_is_key: false,
            current_frame_num_h264: 0,
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            current_poc: 0,
            current_nal_ref_idc: 0,
            frame_num_offset: 0,
            prev_frame_num_offset: 0,
            prev_frame_num_h264: 0,
            reorder_depth: 0,
            has_bitstream_restriction: false,
            last_pocs: [i64::MIN; 16],
            delayed_pics: Vec::new(),
        };

        // Parse avcC extradata if present (MP4/NALFF format), or Annex B parameter sets.
        if extradata.len() >= 7 && extradata[0] == 1 {
            let avcc = parse_avcc(extradata)?;
            decoder.nalff_length_size = avcc.nal_length_size;
            for sps_nal in &avcc.sps {
                if sps_nal.len() > 1
                    && let Ok(sps) = parse_sps_from_nal(sps_nal)
                {
                    let id = sps.sps_id as usize;
                    decoder.apply_sps(&sps);
                    decoder.sps_list[id] = Some(sps);
                }
            }
            for pps_nal in &avcc.pps {
                if pps_nal.len() > 1
                    && let Ok(pps) = parse_pps_from_nal(pps_nal, &decoder.sps_list)
                {
                    let id = pps.pps_id as usize;
                    decoder.pps_list[id] = Some(pps);
                }
            }
        } else if !extradata.is_empty() {
            for nalu in &split_annex_b(extradata) {
                match nalu.nal_type {
                    NalUnitType::Sps => {
                        if let Ok(sps) = parse_sps(&nalu.data) {
                            let id = sps.sps_id as usize;
                            decoder.apply_sps(&sps);
                            decoder.sps_list[id] = Some(sps);
                        }
                    }
                    NalUnitType::Pps => {
                        if let Ok(pps) = parse_pps(&nalu.data, &decoder.sps_list) {
                            let id = pps.pps_id as usize;
                            decoder.pps_list[id] = Some(pps);
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(decoder)
    }

    /// Compute Picture Order Count (type 0) per ITU-T H.264 Section 8.2.1.1.
    ///
    /// For IDR: POC = 0, reset prev_poc_msb/lsb.
    /// For non-IDR: compute poc_msb from wrap-around detection of poc_lsb,
    /// then poc = poc_msb + pic_order_cnt_lsb.
    fn compute_poc_type0(
        &mut self,
        sps: &Sps,
        hdr: &SliceHeader,
        is_idr: bool,
        _nal_ref_idc: u8,
    ) -> i32 {
        if is_idr {
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
            return hdr.pic_order_cnt_lsb as i32;
        }

        let max_poc_lsb = 1u32 << sps.log2_max_poc_lsb;
        let poc_lsb = hdr.pic_order_cnt_lsb;

        // Detect MSB wrap-around (spec 8-3)
        let poc_msb = if poc_lsb < self.prev_poc_lsb
            && (self.prev_poc_lsb.wrapping_sub(poc_lsb)) >= max_poc_lsb / 2
        {
            self.prev_poc_msb + max_poc_lsb as i32
        } else if poc_lsb > self.prev_poc_lsb
            && (poc_lsb.wrapping_sub(self.prev_poc_lsb)) > max_poc_lsb / 2
        {
            self.prev_poc_msb - max_poc_lsb as i32
        } else {
            self.prev_poc_msb
        };

        poc_msb + poc_lsb as i32
    }

    /// Compute frame_num_offset, handling wrap when frame_num < prev_frame_num.
    ///
    /// Reference: FFmpeg h264_parse.c:287-289.
    fn compute_frame_num_offset(&mut self, frame_num: u32, max_frame_num: u32, is_idr: bool) {
        if is_idr {
            self.prev_frame_num_offset = 0;
            self.frame_num_offset = 0;
        } else {
            self.frame_num_offset = self.prev_frame_num_offset;
            if frame_num < self.prev_frame_num_h264 {
                self.frame_num_offset += max_frame_num as i32;
            }
        }
    }

    /// Compute POC for poc_type == 2.
    ///
    /// Reference: FFmpeg h264_parse.c:344-352.
    fn compute_poc_type2(&self, frame_num: u32, nal_ref_idc: u8) -> i32 {
        let mut poc = 2 * (self.frame_num_offset + frame_num as i32);
        if nal_ref_idc == 0 {
            poc -= 1;
        }
        poc
    }

    /// Compute POC for poc_type == 1.
    ///
    /// Reference: FFmpeg h264_parse.c:308-343.
    fn compute_poc_type1(&self, sps: &Sps, hdr: &SliceHeader, nal_ref_idc: u8) -> i32 {
        let abs_frame_num = if sps.num_ref_frames_in_poc_cycle != 0 {
            let mut afn = self.frame_num_offset + hdr.frame_num as i32;
            if nal_ref_idc == 0 && afn > 0 {
                afn -= 1;
            }
            afn
        } else {
            0
        };

        let expectedpoc: i64 = if abs_frame_num > 0 {
            let expected_delta_per_poc_cycle: i64 =
                sps.offset_for_ref_frame.iter().map(|&v| v as i64).sum();
            let poc_cycle_length = sps.num_ref_frames_in_poc_cycle as i32;
            let poc_cycle_cnt = (abs_frame_num - 1) / poc_cycle_length;
            let frame_num_in_poc_cycle = ((abs_frame_num - 1) % poc_cycle_length) as usize;

            let mut ep = poc_cycle_cnt as i64 * expected_delta_per_poc_cycle;
            for &off in sps
                .offset_for_ref_frame
                .iter()
                .take(frame_num_in_poc_cycle + 1)
            {
                ep += off as i64;
            }
            if nal_ref_idc == 0 {
                ep += sps.offset_for_non_ref_pic as i64;
            }
            ep
        } else {
            0
        };

        // field_poc[0] and field_poc[1], take min for frame POC
        let field_poc_0 = expectedpoc + hdr.delta_pic_order_cnt[0] as i64;
        let field_poc_1 = field_poc_0
            + sps.offset_for_top_to_bottom_field as i64
            + hdr.delta_pic_order_cnt[1] as i64;

        // For frame mode, POC is min of both fields
        (field_poc_0.min(field_poc_1)) as i32
    }

    /// Update decoder dimensions from an SPS.
    fn apply_sps(&mut self, sps: &Sps) {
        /// Maximum number of DPB MBs for a given level_idc.
        /// From H.264 Table A-1, matching FFmpeg h264_ps.c level_max_dpb_mbs.
        fn level_max_dpb_mbs(level_idc: u8) -> u32 {
            match level_idc {
                9 => 396, // Level 1b
                10 => 396,
                11 => 900,
                12 | 13 | 20 => 2376,
                21 => 4752,
                22 | 30 => 8100,
                31 => 18000,
                32 => 20480,
                40 | 41 => 32768,
                42 => 34816,
                50..=52 => 184320,
                _ => 184320,
            }
        }

        let w = sps.width();
        let h = sps.height();
        if w > 0 && h > 0 {
            self.width = w;
            self.height = h;
        }
        self.coded_width = sps.coded_width();
        self.coded_height = sps.coded_height();
        self.crop = sps.crop_rect();
        if sps.vui_parameters_present {
            self.color = sps.color;
            if sps.sar.num > 0 && sps.sar.den > 0 {
                self.sar = sps.sar;
            }
        }

        // Pre-set reorder_depth from VUI num_reorder_frames (matching FFmpeg
        // h264_slice.c:1304-1306). This ensures the reorder buffer is active
        // before the first B-frame arrives.
        if sps.num_reorder_frames >= 0 {
            self.has_bitstream_restriction = true;
            let nr = sps.num_reorder_frames as usize;
            if nr > self.reorder_depth {
                self.reorder_depth = nr;
            }
        } else if !self.has_bitstream_restriction && sps.profile_idc != 66 {
            // When VUI bitstream_restriction is absent and the profile
            // supports B-frames (non-Baseline), estimate reorder depth
            // from the level's max DPB capacity (matching FFmpeg
            // h264_ps.c:538-550). FFmpeg normally discovers reorder depth
            // via avformat_find_stream_info probing; since wedeo doesn't
            // probe, we pre-set it from the level to avoid premature
            // output of P-frames before their B-frame successors arrive.
            let mb_count = sps.mb_width * sps.mb_height;
            let level_dpb_frames = level_max_dpb_mbs(sps.level_idc)
                .checked_div(mb_count)
                .map_or(1, |v| v.min(15) as usize);
            // Cap at max_num_ref_frames (like FFmpeg) to avoid excessive
            // buffering for small frames at high levels.
            let estimated = level_dpb_frames.min(sps.max_num_ref_frames as usize);
            if estimated > self.reorder_depth {
                self.reorder_depth = estimated;
            }
        }
    }

    /// Fill frame_num gap by creating dummy DPB entries.
    ///
    /// When frame_num is not consecutive with prev_frame_num, advance
    /// prev_frame_num through each missing value, creating dummy DPB
    /// entries (cloned from the most recent short-term ref) and running
    /// sliding window marking for each.
    ///
    /// Reference: FFmpeg h264_slice.c:1506-1570.
    fn fill_frame_num_gap(&mut self, frame_num: u32, max_frame_num: u32, sps: &Sps) {
        let expected_next = (self.prev_frame_num_h264 + 1) % max_frame_num;
        if frame_num == self.prev_frame_num_h264 || frame_num == expected_next {
            return;
        }

        if !sps.gaps_in_frame_num_allowed {
            self.last_pocs = [i64::MIN; 16];
        }

        let mb_w = sps.mb_width;
        let mb_h = sps.mb_height;
        let pic_w = mb_w * 16;
        let pic_h = mb_h * 16;
        let y_stride = pic_w as usize;
        let uv_stride = (pic_w / 2) as usize;
        let total_4x4 = (mb_w * mb_h * 16) as usize;

        let max_refs = sps.max_num_ref_frames;

        // Find the most recent short-term ref to clone pixel data from
        // (error concealment, matching FFmpeg h264_slice.c:1546-1558).
        let prev_pic = self
            .dpb
            .entries
            .iter()
            .filter_map(|e| e.as_ref())
            .filter(|e| e.status == RefStatus::ShortTerm)
            .max_by_key(|e| e.frame_num)
            .map(|e| (Arc::clone(&e.pic), e.poc));

        // Safety limit: never fill more than max_frame_num steps
        let mut steps = 0u32;
        while self.prev_frame_num_h264 != frame_num
            && (self.prev_frame_num_h264 + 1) % max_frame_num != frame_num
            && steps < max_frame_num
        {
            self.prev_frame_num_h264 = (self.prev_frame_num_h264 + 1) % max_frame_num;
            steps += 1;

            let gap_frame_num = self.prev_frame_num_h264;

            // Clone previous ref picture data, or use mid-grey fallback.
            // Arc::clone is a cheap refcount bump — gap frames share pixel data.
            let (pic, poc) = if let Some((prev_pic, prev_poc)) = &prev_pic {
                (Arc::clone(prev_pic), *prev_poc + 2 * steps as i32)
            } else {
                (
                    SharedPicture::new(PictureBuffer {
                        y: vec![128u8; y_stride * pic_h as usize],
                        u: vec![128u8; uv_stride * (pic_h / 2) as usize],
                        v: vec![128u8; uv_stride * (pic_h / 2) as usize],
                        y_stride,
                        uv_stride,
                        width: pic_w,
                        height: pic_h,
                        mb_width: mb_w,
                        mb_height: mb_h,
                    }),
                    0,
                )
            };

            let entry = DpbEntry {
                pic,
                poc,
                frame_num: gap_frame_num,
                status: RefStatus::Unused,
                long_term_frame_idx: 0,
                mv_info: vec![[0i16; 2]; total_4x4],
                ref_info: vec![-1i8; total_4x4],
                mv_info_l1: vec![[0i16; 2]; total_4x4],
                ref_info_l1: vec![-1i8; total_4x4],
                mb_intra: vec![true; (mb_w * mb_h) as usize],
                needs_output: false,
                ref_poc_l0: Vec::new(),
                ref_poc_l1: Vec::new(),
            };

            // Make room if DPB is full
            if self.dpb.is_full() {
                self.dpb.remove_oldest_short_term();
                if self.dpb.is_full() {
                    for i in 0..self.dpb.entries.len() {
                        if let Some(e) = &self.dpb.entries[i]
                            && e.status == RefStatus::Unused
                        {
                            self.dpb.entries[i] = None;
                            break;
                        }
                    }
                }
            }

            if let Some(dpb_idx) = self.dpb.store(entry) {
                refs::sliding_window_mark_gap(
                    &mut self.dpb,
                    max_refs,
                    dpb_idx,
                    gap_frame_num,
                    max_frame_num,
                );
            }
        }
    }

    /// Process a single NAL unit.
    fn process_nal(
        &mut self,
        nalu: &NalUnit,
        pts: Timestamp,
        dts: Timestamp,
        duration: i64,
        time_base: Rational,
    ) -> Result<()> {
        match nalu.nal_type {
            NalUnitType::Sps => {
                let sps = parse_sps(&nalu.data)?;
                let id = sps.sps_id as usize;
                self.apply_sps(&sps);
                self.sps_list[id] = Some(sps);
            }
            NalUnitType::Pps => {
                let pps = parse_pps(&nalu.data, &self.sps_list)?;
                let id = pps.pps_id as usize;
                self.pps_list[id] = Some(pps);
            }
            NalUnitType::Idr | NalUnitType::Slice => {
                // Parse slice header
                let hdr = parse_slice_header(
                    &nalu.data,
                    nalu.nal_type,
                    nalu.nal_ref_idc,
                    &self.sps_list,
                    &self.pps_list,
                )?;

                // Look up PPS and SPS for this slice (clone to avoid borrow conflicts)
                let pps = self.pps_list[hdr.pps_id as usize]
                    .clone()
                    .ok_or_else(|| invalid_data("referenced PPS not found"))?;
                let sps = self.sps_list[pps.sps_id as usize]
                    .clone()
                    .ok_or_else(|| invalid_data("referenced SPS not found"))?;

                let is_idr = nalu.nal_type == NalUnitType::Idr;

                // Check if this starts a new frame
                if hdr.first_mb_in_slice == 0 {
                    // Flush any in-progress frame and store in DPB
                    self.finish_current_frame();

                    // Refresh active stream metadata from the SPS referenced by this frame.
                    self.apply_sps(&sps);

                    // Start new frame context
                    self.current_fdc =
                        Some(FrameDecodeContext::new(&sps, &pps, Some(&self.buffer_pool)));
                    // Store the packet timing for this frame, matching FFmpeg's
                    // ff_decode_frame_props which copies packet props to frame.
                    self.current_pts = pts;
                    self.current_dts = dts;
                    self.current_duration = duration;
                    self.current_time_base = time_base;
                    self.current_is_idr = is_idr;
                    self.current_is_key =
                        is_idr || std::mem::take(&mut self.pending_recovery_point);
                    self.current_frame_num_h264 = hdr.frame_num;
                    self.current_nal_ref_idc = nalu.nal_ref_idc;

                    // Dynamic reorder depth increase (matching FFmpeg
                    // h264_slice.c:1328-1331). B-frames need at least depth 1.
                    if hdr.slice_type.is_b() && self.reorder_depth < 1 {
                        self.reorder_depth = 1;
                    }

                    // Frame num gap fill (H.264 Section 8.2.5.2).
                    let max_frame_num = 1u32 << sps.log2_max_frame_num;
                    if !is_idr {
                        self.fill_frame_num_gap(hdr.frame_num, max_frame_num, &sps);
                    }

                    // Compute spec-compliant POC
                    self.compute_frame_num_offset(hdr.frame_num, max_frame_num, is_idr);

                    if sps.poc_type == 0 {
                        self.current_poc =
                            self.compute_poc_type0(&sps, &hdr, is_idr, nalu.nal_ref_idc);
                    } else if sps.poc_type == 1 {
                        self.current_poc = self.compute_poc_type1(&sps, &hdr, nalu.nal_ref_idc);
                    } else {
                        // POC type 2
                        self.current_poc = self.compute_poc_type2(hdr.frame_num, nalu.nal_ref_idc);
                    }
                }

                // Build reference lists per-slice. Different slices within
                // a frame can have different types (e.g., CABAST3 has
                // alternating I/P slices). Each P/B slice needs its own
                // reference list; I-slices clear the lists.
                if hdr.slice_type.is_p() {
                    let max_frame_num = 1u32 << sps.log2_max_frame_num;
                    self.ref_list_l0 =
                        refs::build_ref_list_p(&self.dpb, &hdr, hdr.frame_num, max_frame_num);
                    self.ref_list_l1.clear();
                } else if hdr.slice_type.is_b() {
                    let max_frame_num_b = 1u32 << sps.log2_max_frame_num;
                    let (l0, l1) =
                        refs::build_ref_list_b(&self.dpb, &hdr, self.current_poc, max_frame_num_b);
                    self.ref_list_l0 = l0;
                    self.ref_list_l1 = l1;
                } else {
                    self.ref_list_l0.clear();
                    self.ref_list_l1.clear();
                }

                // Buffer this slice for worker thread decode.
                // Take the fdc temporarily to avoid borrow conflicts with self.dpb.
                if let Some(mut fdc) = self.current_fdc.take() {
                    // Track slice boundaries for neighbor availability.
                    // First slice (first_mb==0) starts at 0; continuations increment.
                    if hdr.first_mb_in_slice > 0 {
                        fdc.current_slice += 1;
                    }

                    // Store per-slice deblocking parameters on the FDC.
                    let slice_idx = fdc.current_slice as usize;
                    let slice_chroma_qp_offset = self.pps_list[hdr.pps_id as usize]
                        .as_ref()
                        .map_or([0, 0], |pps| pps.chroma_qp_index_offset);
                    if slice_idx >= fdc.slice_deblock_params.len() {
                        fdc.slice_deblock_params
                            .resize(slice_idx + 1, deblock::SliceDeblockParams::default());
                    }
                    fdc.slice_deblock_params[slice_idx] = deblock::SliceDeblockParams {
                        alpha_c0_offset: hdr.slice_alpha_c0_offset,
                        beta_offset: hdr.slice_beta_offset,
                        disable_deblocking_filter_idc: hdr.disable_deblocking_filter_idc,
                        chroma_qp_index_offset: slice_chroma_qp_offset,
                    };

                    // Build list of SharedPicture references for MC.
                    let ref_pic_list: Vec<Arc<SharedPicture>> = self
                        .ref_list_l0
                        .iter()
                        .filter_map(|&dpb_idx| self.dpb.get(dpb_idx).map(|e| Arc::clone(&e.pic)))
                        .collect();
                    let ref_pic_list_l1: Vec<Arc<SharedPicture>> = self
                        .ref_list_l1
                        .iter()
                        .filter_map(|&dpb_idx| self.dpb.get(dpb_idx).map(|e| Arc::clone(&e.pic)))
                        .collect();

                    if hdr.slice_type.is_p() || hdr.slice_type.is_b() {
                        fdc.cur_l0_ref_poc = self
                            .ref_list_l0
                            .iter()
                            .filter_map(|&i| self.dpb.get(i).map(|e| e.poc))
                            .collect();
                        fdc.cur_l0_ref_dpb = self.ref_list_l0.clone();
                    } else {
                        fdc.cur_l0_ref_poc.clear();
                        fdc.cur_l0_ref_dpb.clear();
                    }
                    if hdr.slice_type.is_b() {
                        fdc.cur_l1_ref_poc = self
                            .ref_list_l1
                            .iter()
                            .filter_map(|&i| self.dpb.get(i).map(|e| e.poc))
                            .collect();
                        fdc.cur_l1_ref_dpb = self.ref_list_l1.clone();
                    } else {
                        fdc.cur_l1_ref_poc.clear();
                        fdc.cur_l1_ref_dpb.clear();
                    }

                    // Populate colocated info from L1[0] for direct mode
                    if hdr.slice_type.is_b() && !self.ref_list_l1.is_empty() {
                        let l1_0_dpb_idx = self.ref_list_l1[0];
                        if let Some(entry) = self.dpb.get(l1_0_dpb_idx) {
                            fdc.col_mv = entry.mv_info.clone();
                            fdc.col_ref = entry.ref_info.clone();
                            fdc.col_mv_l1 = entry.mv_info_l1.clone();
                            fdc.col_ref_l1 = entry.ref_info_l1.clone();
                            fdc.col_mb_intra = entry.mb_intra.clone();
                            fdc.col_poc = entry.poc;
                            fdc.col_ref_poc_l0 = entry.ref_poc_l0.clone();
                            fdc.col_ref_poc_l1 = entry.ref_poc_l1.clone();
                            fdc.col_l1_is_long_term =
                                entry.status == crate::dpb::RefStatus::LongTerm;
                        }
                        fdc.cur_poc = self.current_poc;

                        // Pre-compute implicit weights for weighted_bipred_idc=2
                        if pps.weighted_bipred_idc == 2 {
                            fdc.compute_implicit_weights();
                        }
                    }

                    // Snapshot per-slice state into SliceWorkUnit.
                    let cur_slice_idx = fdc.current_slice;
                    self.pending_slices.push(SliceWorkUnit {
                        hdr: hdr.clone(),
                        rbsp: nalu.data.clone(),
                        sps: sps.clone(),
                        pps: pps.clone(),
                        ref_pics_l0: ref_pic_list,
                        ref_pics_l1: ref_pic_list_l1,
                        slice_idx: cur_slice_idx,
                        cur_l0_ref_poc: std::mem::take(&mut fdc.cur_l0_ref_poc),
                        cur_l0_ref_dpb: std::mem::take(&mut fdc.cur_l0_ref_dpb),
                        cur_l1_ref_poc: std::mem::take(&mut fdc.cur_l1_ref_poc),
                        cur_l1_ref_dpb: std::mem::take(&mut fdc.cur_l1_ref_dpb),
                        col_mv: std::mem::take(&mut fdc.col_mv),
                        col_ref: std::mem::take(&mut fdc.col_ref),
                        col_mv_l1: std::mem::take(&mut fdc.col_mv_l1),
                        col_ref_l1: std::mem::take(&mut fdc.col_ref_l1),
                        col_mb_intra: std::mem::take(&mut fdc.col_mb_intra),
                        col_poc: fdc.col_poc,
                        col_l1_is_long_term: fdc.col_l1_is_long_term,
                        col_ref_poc_l0: std::mem::take(&mut fdc.col_ref_poc_l0),
                        col_ref_poc_l1: std::mem::take(&mut fdc.col_ref_poc_l1),
                        implicit_weight: std::mem::take(&mut fdc.implicit_weight),
                        cur_poc: fdc.cur_poc,
                        next_slice_first_mb: 0, // computed in finish_current_frame
                    });

                    self.current_last_hdr = Some(hdr.clone());
                    self.current_fdc = Some(fdc);
                }
            }
            NalUnitType::Sei => {
                if sei::parse_sei(&nalu.data).is_some() {
                    self.pending_recovery_point = true;
                }
            }
            NalUnitType::Aud
            | NalUnitType::Filler
            | NalUnitType::EndSequence
            | NalUnitType::EndStream
            | NalUnitType::SliceA
            | NalUnitType::SliceB
            | NalUnitType::SliceC => {}
        }
        Ok(())
    }

    /// Flush a completed frame: dispatch all buffered slices to a worker
    /// thread, manage the in-flight queue, apply pre-dispatch state updates.
    ///
    /// Phase 6: the entire slice decode + last-row deblock + mark_complete
    /// runs on a dedicated std::thread. Reference frames join immediately;
    /// non-ref B-frames are deferred in in_flight_queue.
    fn finish_current_frame(&mut self) {
        if let (Some(fdc), Some(last_hdr)) = (self.current_fdc.take(), self.current_last_hdr.take())
        {
            let mut slices = std::mem::take(&mut self.pending_slices);
            if slices.is_empty() {
                // No slices buffered — put FDC back
                self.current_fdc = Some(fdc);
                self.current_last_hdr = Some(last_hdr);
                return;
            }

            // Compute slice boundaries for parallel decode.
            // Sort by first_mb_in_slice (should already be ordered, but enforce).
            slices.sort_by_key(|s| s.hdr.first_mb_in_slice);
            let total_mbs = fdc.mb_width * fdc.mb_height;
            for i in 0..slices.len() {
                slices[i].next_slice_first_mb = if i + 1 < slices.len() {
                    slices[i + 1].hdr.first_mb_in_slice
                } else {
                    total_mbs
                };
            }

            let deblock_enabled = true;
            let poc = self.current_poc;
            let frame_num_h264 = self.current_frame_num_h264;
            let nal_ref_idc = self.current_nal_ref_idc;
            let is_idr = self.current_is_idr;
            let is_key = self.current_is_key;
            let ref_list_l0 = self.ref_list_l0.clone();
            let ref_list_l1 = self.ref_list_l1.clone();
            let is_ref = nal_ref_idc > 0 || is_idr;

            // === Pre-dispatch state updates ===
            // These must happen before the next frame's process_nal to ensure
            // correct POC computation, frame_num gap fill, and ref list build.

            // IDR resets last_pocs
            if is_idr {
                self.last_pocs = [i64::MIN; 16];
            }

            let has_mmco5 = last_hdr.adaptive_ref_pic_marking
                && last_hdr
                    .mmco_ops
                    .iter()
                    .any(|op| matches!(op, crate::slice::MmcoOp::Reset));
            if has_mmco5 {
                self.last_pocs = [i64::MIN; 16];
            }

            // last_pocs update + out_of_order computation
            let cur_poc_i64 = if has_mmco5 { 0i64 } else { poc as i64 };
            let mut insert_pos = 0usize;
            for i in 0..=16 {
                if i == 16 || cur_poc_i64 < self.last_pocs[i] {
                    if i > 0 {
                        self.last_pocs[i - 1] = cur_poc_i64;
                    }
                    insert_pos = i;
                    break;
                } else if i > 0 {
                    self.last_pocs[i - 1] = self.last_pocs[i];
                }
            }

            let mut out_of_order = 16 - insert_pos;
            if last_hdr.slice_type.is_b()
                || (self.last_pocs[14] > i64::MIN && (self.last_pocs[15] - self.last_pocs[14]) > 2)
            {
                out_of_order = out_of_order.max(1);
            }

            if out_of_order == 16 {
                self.last_pocs = [i64::MIN; 16];
                self.last_pocs[0] = cur_poc_i64;
            } else if out_of_order > self.reorder_depth && !self.has_bitstream_restriction {
                self.reorder_depth = out_of_order;
            }

            // === Queue management (before state updates for correct mid_stream_idr) ===
            // Ref frames: join all queued non-ref frames first to ensure
            // correct output ordering in delayed_pics. Must happen before
            // mid_stream_idr check so delayed_pics reflects all prior frames.
            if is_ref {
                self.join_all_in_flight();
            }
            // Enforce max in-flight capacity
            let max_in_flight = self.threads;
            while self.in_flight_queue.len() >= max_in_flight {
                self.join_oldest_in_flight();
            }

            let mid_stream_idr = is_idr && !self.delayed_pics.is_empty();
            let frame_mmco_reset = has_mmco5 || out_of_order == 16 || mid_stream_idr;

            // POC state update (ref frames only — non-ref don't affect POC tracking)
            if nal_ref_idc > 0 {
                let max_poc_lsb = self
                    .sps_list
                    .iter()
                    .find_map(|s| s.as_ref().map(|sps| 1u32 << sps.log2_max_poc_lsb))
                    .unwrap_or(16);
                let poc_lsb = last_hdr.pic_order_cnt_lsb;
                if is_idr {
                    self.prev_poc_msb = 0;
                    self.prev_poc_lsb = 0;
                } else {
                    let poc_msb = if poc_lsb < self.prev_poc_lsb
                        && (self.prev_poc_lsb.wrapping_sub(poc_lsb)) >= max_poc_lsb / 2
                    {
                        self.prev_poc_msb + max_poc_lsb as i32
                    } else if poc_lsb > self.prev_poc_lsb
                        && (poc_lsb.wrapping_sub(self.prev_poc_lsb)) > max_poc_lsb / 2
                    {
                        self.prev_poc_msb - max_poc_lsb as i32
                    } else {
                        self.prev_poc_msb
                    };
                    self.prev_poc_msb = poc_msb;
                    self.prev_poc_lsb = poc_lsb;
                }
            }

            // frame_num state (all frames)
            self.prev_frame_num_offset = self.frame_num_offset;
            self.prev_frame_num_h264 = frame_num_h264;
            self.frame_num += 1;

            // Note: MMCO-5 resets prev_frame_num/poc to 0 in complete_in_flight
            // (post-join). Safe because ref frames are joined before next frame.

            // === Dispatch to thread pool ===
            let seq_id = self.next_dispatch_id;
            self.next_dispatch_id += 1;

            self.pool.submit(Box::new(FrameWork {
                fdc,
                slices,
                deblock_enabled,
                sequence_id: seq_id,
                poc,
                frame_num_h264,
                nal_ref_idc,
                is_idr,
                is_key,
                last_hdr,
                ref_list_l0,
                ref_list_l1,
                pts: self.current_pts,
                dts: self.current_dts,
                duration: self.current_duration,
                time_base: self.current_time_base,
                coded_width: self.coded_width,
                coded_height: self.coded_height,
                color: self.color,
                sar: self.sar,
                crop: self.crop,
            }));

            self.in_flight_queue.push_back(InFlightFrame {
                sequence_id: seq_id,
                frame_mmco_reset,
                out_of_order,
            });

            // Ref frames: join immediately (DPB/MMCO must complete before
            // next frame's ref list build)
            if is_ref {
                self.join_oldest_in_flight();
            }
        }
    }

    /// Wait for the oldest in-flight frame result and run post-join completion.
    /// Uses FIFO ordering: if the pool delivers results out of order, buffers
    /// them in `result_buffer` until the needed `sequence_id` arrives.
    fn join_oldest_in_flight(&mut self) {
        if let Some(frame) = self.in_flight_queue.pop_front() {
            let needed = frame.sequence_id;

            // Check buffer first
            let boxed = if let Some(b) = self.result_buffer.remove(&needed) {
                b
            } else {
                // Recv until we get the one we need, buffering others
                loop {
                    match self.pool.recv() {
                        Some(result) => {
                            if result.sequence_id == needed {
                                break result.decode;
                            }
                            self.result_buffer.insert(result.sequence_id, result.decode);
                        }
                        None => {
                            return;
                        }
                    }
                }
            };

            self.complete_in_flight(boxed, frame.frame_mmco_reset, frame.out_of_order);
        }
    }

    /// Join all in-flight frames in dispatch order.
    fn join_all_in_flight(&mut self) {
        while !self.in_flight_queue.is_empty() {
            self.join_oldest_in_flight();
        }
    }

    /// Complete an in-flight decode: emit output frame, DPB store.
    ///
    /// Post-join work only. Pre-dispatch state updates (last_pocs, reorder_depth,
    /// POC state, frame_num) were already applied in `finish_current_frame`.
    fn complete_in_flight(
        &mut self,
        mut in_flight: Box<InFlightDecode>,
        frame_mmco_reset: bool,
        out_of_order: usize,
    ) {
        let mut is_corrupt = false;
        if let Some(err) = in_flight.decode_error.take() {
            self.health.record(HealthEvent {
                track: self.track_id,
                pts: in_flight.pts,
                byte_offset: None,
                kind: HealthIssue::CorruptSyntax,
                detail: "h264 slice decode error",
            });
            if self.error_policy == ErrorPolicy::Strict {
                if self.fatal_error.is_none() {
                    self.fatal_error = Some(err);
                }
                return;
            }
            is_corrupt = true;
        }

        let frame = self.fdc_to_frame(&in_flight, is_corrupt);

        // Add to delayed_pics and flush output
        self.delayed_pics
            .push((in_flight.poc, frame, frame_mmco_reset));

        let has_barrier = self.delayed_pics.iter().any(|(_, _, reset)| *reset);
        let allow_output = out_of_order == 0 || has_barrier;
        while allow_output && self.delayed_pics.len() > self.reorder_depth {
            let out_idx = if self.delayed_pics[0].2 {
                0
            } else {
                let barrier = self
                    .delayed_pics
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find(|(_, (_, _, reset))| *reset)
                    .map(|(i, _)| i)
                    .unwrap_or(self.delayed_pics.len());
                self.delayed_pics[..barrier]
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, (poc, _, _))| *poc)
                    .map(|(i, _)| i)
                    .unwrap()
            };
            let (_, f, _) = self.delayed_pics.remove(out_idx);
            self.output_queue.push_back(f);
        }

        // Non-reference pictures don't need DPB storage.
        if in_flight.nal_ref_idc == 0 {
            return;
        }

        // Store decoded picture in DPB for reference
        let mb_width = in_flight.fdc.mb_width;
        let mb_height = in_flight.fdc.mb_height;
        let total_blocks = (mb_width * mb_height * 16) as usize;

        let (mv_info, ref_info, mv_info_l1, ref_info_l1, mb_intra) = {
            let fdc = &in_flight.fdc;
            let mv_info = if fdc.mv_ctx.mv.len() == total_blocks {
                fdc.mv_ctx.mv.clone()
            } else {
                vec![[0i16; 2]; total_blocks]
            };
            let ref_info = if fdc.mv_ctx.ref_idx.len() == total_blocks {
                fdc.mv_ctx.ref_idx.clone()
            } else {
                vec![-1i8; total_blocks]
            };
            let mv_info_l1 = if fdc.mv_ctx.mv_l1.len() == total_blocks {
                fdc.mv_ctx.mv_l1.clone()
            } else {
                vec![[0i16; 2]; total_blocks]
            };
            let ref_info_l1 = if fdc.mv_ctx.ref_idx_l1.len() == total_blocks {
                fdc.mv_ctx.ref_idx_l1.clone()
            } else {
                vec![-1i8; total_blocks]
            };

            let mb_intra: Vec<bool> = fdc.mb_info.iter().map(|info| info.is_intra).collect();
            (mv_info, ref_info, mv_info_l1, ref_info_l1, mb_intra)
        };

        let ref_poc_l0: Vec<i32> = in_flight
            .ref_list_l0
            .iter()
            .filter_map(|&dpb_idx| self.dpb.get(dpb_idx).map(|e| e.poc))
            .collect();
        let ref_poc_l1: Vec<i32> = in_flight
            .ref_list_l1
            .iter()
            .filter_map(|&dpb_idx| self.dpb.get(dpb_idx).map(|e| e.poc))
            .collect();

        let entry = DpbEntry {
            pic: in_flight.fdc.pic.into_shared(),
            poc: in_flight.poc,
            frame_num: in_flight.frame_num_h264,
            status: RefStatus::Unused,
            long_term_frame_idx: 0,
            mv_info,
            ref_info,
            mv_info_l1,
            ref_info_l1,
            mb_intra,
            needs_output: false,
            ref_poc_l0,
            ref_poc_l1,
        };

        if self.dpb.is_full() {
            self.dpb.remove_oldest_short_term();
            if self.dpb.is_full() {
                for i in 0..self.dpb.entries.len() {
                    if let Some(e) = &self.dpb.entries[i]
                        && e.status == RefStatus::Unused
                    {
                        self.dpb.entries[i] = None;
                        break;
                    }
                }
            }
        }
        let dpb_idx = self.dpb.store(entry);

        let mut mmco_did_reset = false;
        if let Some(dpb_idx) = dpb_idx {
            let (sps_max_refs, max_frame_num) = self
                .sps_list
                .iter()
                .find_map(|s| {
                    s.as_ref()
                        .map(|sps| (sps.max_num_ref_frames, 1u32 << sps.log2_max_frame_num))
                })
                .unwrap_or((4, 16));
            mmco_did_reset = refs::mark_reference(
                &mut self.dpb,
                &in_flight.last_hdr,
                in_flight.is_idr,
                in_flight.frame_num_h264,
                max_frame_num,
                sps_max_refs,
                Some(dpb_idx),
            );
        }

        // MMCO-5 (Reset) overrides the pre-dispatch state updates.
        // Safe because ref frames (which have MMCO) are joined immediately.
        if mmco_did_reset {
            self.prev_frame_num_h264 = 0;
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
            self.prev_frame_num_offset = 0;
        }
    }

    /// Decode a slice into a FrameDecodeContext.
    ///
    /// Dispatches to CAVLC or CABAC slice decode based on PPS entropy_coding_mode_flag.
    /// Returns the number of MBs decoded in this slice.
    #[allow(clippy::too_many_arguments)] // H.264 slice decode needs all parameters
    pub(crate) fn decode_slice_into(
        rbsp: &[u8],
        hdr: &SliceHeader,
        sps: &Sps,
        pps: &Pps,
        fdc: &mut FrameDecodeContext,
        ref_pics: &[Arc<SharedPicture>],
        ref_pics_l1: &[Arc<SharedPicture>],
    ) -> Result<u32> {
        fdc.qp = hdr.slice_qp as u8;
        fdc.last_qscale_diff = 0; // H.264 spec: prevMbQpDelta = 0 at slice start

        if pps.entropy_coding_mode_flag {
            Self::decode_slice_cabac(rbsp, hdr, sps, pps, fdc, ref_pics, ref_pics_l1)
        } else {
            Self::decode_slice_cavlc(rbsp, hdr, sps, pps, fdc, ref_pics, ref_pics_l1)
        }
    }

    /// Decode a CAVLC-coded slice.
    #[allow(clippy::too_many_arguments)]
    fn decode_slice_cavlc(
        rbsp: &[u8],
        hdr: &SliceHeader,
        sps: &Sps,
        pps: &Pps,
        fdc: &mut FrameDecodeContext,
        ref_pics: &[Arc<SharedPicture>],
        ref_pics_l1: &[Arc<SharedPicture>],
    ) -> Result<u32> {
        let mb_width = sps.mb_width;
        let mb_height = sps.mb_height;
        let total_mbs = mb_width * mb_height;
        let rbsp_bits = rbsp.len() * 8;

        // Create a bitstream reader starting at the macroblock data.
        let mut padded = Vec::with_capacity(rbsp.len() + 8);
        padded.extend_from_slice(rbsp);
        padded.resize(rbsp.len() + 8, 0);
        let mut br = BitReadBE::new(&padded);
        br.skip_bits(hdr.header_bits);

        // Decode macroblocks for this slice
        let first_mb = hdr.first_mb_in_slice;
        let mut mbs_decoded = 0u32;
        let is_inter_slice = hdr.slice_type.is_p() || hdr.slice_type.is_b();
        let is_mbaff = !sps.frame_mbs_only_flag;

        if is_mbaff {
            return Self::decode_slice_cavlc_mbaff(
                &mut br,
                rbsp_bits,
                hdr,
                sps,
                pps,
                fdc,
                ref_pics,
                ref_pics_l1,
                is_inter_slice,
            );
        }

        let deblock_enabled = true;
        let mut mb_addr = first_mb;

        while mb_addr < total_mbs {
            let mb_x = mb_addr % mb_width;
            let mb_y = mb_addr / mb_width;

            // Update neighbor context at the start of each row
            if mb_x == 0 {
                fdc.neighbor_ctx.new_row();
                // Top is available only if it exists AND is in the same slice.
                fdc.neighbor_ctx.top_available =
                    mb_y > 0 && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
            } else if mb_addr == first_mb {
                // First MB of a continuation slice that doesn't start at
                // column 0: the left neighbor is from the previous slice.
                fdc.neighbor_ctx.left_available = false;
                fdc.neighbor_ctx.top_available =
                    mb_y > 0 && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
            }

            if is_inter_slice {
                // For inter slices, mb_skip_run MUST be parsed before any
                // early-exit check.  The skip run can signal that the very last
                // MB in the frame is a P_SKIP; if we broke out before parsing
                // it (because only the run + RBSP trailing bits remain), that
                // MB would stay at the zero-initialised value.
                //
                // However, if parsing fails because we've consumed almost all
                // RBSP data (only trailing bits remain), that's a normal end
                // of slice, not an error.
                let mb_skip_run = match get_ue_golomb(&mut br) {
                    Ok(v) => v,
                    Err(_) if br.consumed() + 8 >= rbsp_bits => break,
                    Err(e) => return Err(e),
                };

                // Process skipped MBs
                for _ in 0..mb_skip_run {
                    if mb_addr >= total_mbs {
                        break;
                    }
                    let skip_x = mb_addr % mb_width;
                    let skip_y = mb_addr / mb_width;
                    if skip_x == 0 && mb_addr != first_mb {
                        fdc.neighbor_ctx.new_row();
                    }
                    // Per-MB top availability (slice-boundary aware)
                    fdc.neighbor_ctx.top_available = skip_y > 0
                        && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
                    if hdr.slice_type.is_b() {
                        mb::decode_b_skip_mb(fdc, hdr, skip_x, skip_y, ref_pics, ref_pics_l1);
                    } else {
                        mb::decode_skip_mb(fdc, hdr, skip_x, skip_y, ref_pics, ref_pics_l1);
                    }
                    mb_addr += 1;
                    mbs_decoded += 1;
                    // Inline deblock + publish with 1-row delay
                    if mb_addr.is_multiple_of(mb_width) {
                        inline_deblock_row(fdc, mb_addr / mb_width - 1, mb_width, deblock_enabled);
                    }
                }

                if mb_addr >= total_mbs {
                    break; // Skip run consumed remaining MBs
                }

                // Check if we've consumed all RBSP data after skip run.
                // Use a tight margin: the stop bit is 1 bit + ≤7 alignment
                // zeros, but after a skip run we may still have a coded MB.
                // A coded MB needs at least a mb_type UE code (1 bit min).
                if br.consumed() + 1 >= rbsp_bits {
                    break;
                }

                // Re-check row boundary after skips
                let mb_x = mb_addr % mb_width;
                let mb_y = mb_addr / mb_width;
                if mb_x == 0 && mb_addr != first_mb {
                    fdc.neighbor_ctx.new_row();
                    fdc.neighbor_ctx.top_available = mb_y > 0
                        && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
                }
            } else {
                // Intra slice: no skip run, but guard against reading past end.
                if br.consumed() + 8 >= rbsp_bits {
                    break;
                }
            }

            // Update per-MB neighbor availability (slice-boundary aware).
            // top_available must be per-MB because the top row may span
            // multiple slices (when first_mb is mid-row).
            let mb_x = mb_addr % mb_width;
            let mb_y = mb_addr / mb_width;
            fdc.neighbor_ctx.top_available =
                mb_y > 0 && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;

            // Decode coded MB
            mb::decode_macroblock(
                fdc,
                &mut br,
                hdr,
                sps,
                pps,
                mb_x,
                mb_y,
                ref_pics,
                ref_pics_l1,
            )?;
            mb_addr += 1;
            mbs_decoded += 1;
            // Inline deblock + publish with 1-row delay
            if mb_addr.is_multiple_of(mb_width) {
                inline_deblock_row(fdc, mb_addr / mb_width - 1, mb_width, deblock_enabled);
            }
        }

        Ok(mbs_decoded)
    }

    /// Decode a CAVLC-coded slice in MBAFF mode (MB pairs).
    ///
    /// Processes MBs in pairs: top (even mb_y) then bottom (odd mb_y).
    /// Reads mb_field_decoding_flag from bitstream at the top MB of each pair.
    /// Skip run persists across pair members.
    ///
    /// Reference: FFmpeg h264_slice.c:2681-2755 (CAVLC MBAFF pair loop),
    ///            h264_cavlc.c:689-701 (MBAFF skip/field flag logic).
    #[allow(clippy::too_many_arguments)]
    fn decode_slice_cavlc_mbaff(
        br: &mut BitReadBE<'_>,
        rbsp_bits: usize,
        hdr: &SliceHeader,
        sps: &Sps,
        pps: &Pps,
        fdc: &mut FrameDecodeContext,
        ref_pics: &[Arc<SharedPicture>],
        ref_pics_l1: &[Arc<SharedPicture>],
        is_inter_slice: bool,
    ) -> Result<u32> {
        let mb_width = sps.mb_width;
        let mb_height = sps.mb_height;
        let total_mbs = mb_width * mb_height;
        let first_mb = hdr.first_mb_in_slice;
        let deblock_enabled = true;

        // For MBAFF, first_mb_in_slice is in pair units.
        let mut mb_x = first_mb % mb_width;
        let mut mb_y = (first_mb / mb_width) * 2;
        let first_mb_flat = mb_x + mb_y * mb_width;

        let mut mbs_decoded = 0u32;
        // mb_skip_run persists across both MBs of a pair and across pairs.
        // -1 means "not yet read for this coded region".
        let mut mb_skip_run: i32 = -1;
        // MBAFF field decoding flag: true when the current pair is field-coded.
        let mut mb_field_decoding_flag = false;

        // Two separate left-side contexts for top and bottom MB rows.
        let mut top_left_nz = [0u8; 8];
        let mut top_left_modes = [-1i8; 4];
        let mut top_left_avail = false;
        let mut bot_left_nz = [0u8; 8];
        let mut bot_left_modes = [-1i8; 4];
        let mut bot_left_avail = false;
        // MBAFF: save top-row context for the bottom MB (same as CABAC path).
        let mut saved_top_modes_cavlc = [-1i8; 4];
        let mut saved_top_nz_luma_cavlc = [0u8; 4];

        'pair_loop: loop {
            if mb_y >= mb_height {
                break;
            }

            for pair_pos in 0..2u32 {
                let cur_y = mb_y + pair_pos;
                let mb_addr = mb_x + cur_y * mb_width;

                if mb_addr >= total_mbs {
                    break 'pair_loop;
                }

                // MBAFF left-side neighbor: load the correct row's context.
                if pair_pos == 0 {
                    fdc.neighbor_ctx.left_nz = top_left_nz;
                    fdc.neighbor_ctx.left_intra4x4_mode = top_left_modes;
                    fdc.neighbor_ctx.left_available = top_left_avail;
                    if mb_x == 0 {
                        fdc.neighbor_ctx.new_row();
                    }
                    // MBAFF: save top-row context before top MB overwrites it.
                    let base = mb_x as usize * 4;
                    saved_top_modes_cavlc
                        .copy_from_slice(&fdc.neighbor_ctx.top_intra4x4_mode[base..base + 4]);
                    saved_top_nz_luma_cavlc
                        .copy_from_slice(&fdc.neighbor_ctx.top_nz_luma[base..base + 4]);
                } else {
                    // Save top's right edge, load bottom's left.
                    top_left_nz = fdc.neighbor_ctx.left_nz;
                    top_left_modes = fdc.neighbor_ctx.left_intra4x4_mode;
                    top_left_avail = fdc.neighbor_ctx.left_available;
                    fdc.neighbor_ctx.left_nz = bot_left_nz;
                    fdc.neighbor_ctx.left_intra4x4_mode = bot_left_modes;
                    fdc.neighbor_ctx.left_available = bot_left_avail;
                    if mb_x == 0 {
                        fdc.neighbor_ctx.new_row();
                    }
                    // MBAFF field-mode: restore pre-top-MB top context.
                    if mb_field_decoding_flag {
                        let base = mb_x as usize * 4;
                        fdc.neighbor_ctx.top_intra4x4_mode[base..base + 4]
                            .copy_from_slice(&saved_top_modes_cavlc);
                        fdc.neighbor_ctx.top_nz_luma[base..base + 4]
                            .copy_from_slice(&saved_top_nz_luma_cavlc);
                    }
                }
                fdc.neighbor_ctx.top_available = cur_y > 0
                    && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
                if mb_addr == first_mb_flat && mb_x == 0 {
                    fdc.neighbor_ctx.left_available = false;
                }

                if is_inter_slice {
                    // Read skip run if we haven't yet
                    if mb_skip_run < 0 {
                        mb_skip_run = match get_ue_golomb(br) {
                            Ok(v) => v as i32,
                            Err(_) if br.consumed() + 8 >= rbsp_bits => break 'pair_loop,
                            Err(e) => return Err(e),
                        };
                    }

                    if mb_skip_run > 0 {
                        mb_skip_run -= 1;

                        // MBAFF: read mb_field_decoding_flag at top MB when this
                        // is the last skip of the run.
                        // Reference: FFmpeg h264_cavlc.c:690-692.
                        if pair_pos == 0 && mb_skip_run == 0 {
                            mb_field_decoding_flag = br.get_bit();
                        }

                        // Set field mode for pixel reconstruction
                        fdc.mb_field = mb_field_decoding_flag;

                        // Decode skip MB
                        if hdr.slice_type.is_b() {
                            mb::decode_b_skip_mb(fdc, hdr, mb_x, cur_y, ref_pics, ref_pics_l1);
                        } else {
                            mb::decode_skip_mb(fdc, hdr, mb_x, cur_y, ref_pics, ref_pics_l1);
                        }
                        mbs_decoded += 1;
                        continue; // next pair_pos
                    }

                    // mb_skip_run == 0: this MB is coded. Reset for next read.
                    mb_skip_run = -1;

                    if br.consumed() + 1 >= rbsp_bits {
                        break 'pair_loop;
                    }
                } else if br.consumed() + 8 >= rbsp_bits {
                    break 'pair_loop;
                }

                // Coded MB: read mb_field_decoding_flag on top MB
                // Reference: FFmpeg h264_cavlc.c:698-701.
                if pair_pos == 0 {
                    mb_field_decoding_flag = br.get_bit();
                }

                // Set field mode for pixel reconstruction
                fdc.mb_field = mb_field_decoding_flag;

                fdc.neighbor_ctx.top_available = cur_y > 0
                    && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;

                mb::decode_macroblock(fdc, br, hdr, sps, pps, mb_x, cur_y, ref_pics, ref_pics_l1)?;
                mbs_decoded += 1;
            }

            // After the pair, save bottom MB's right edge as next pair's bottom-row left.
            bot_left_nz = fdc.neighbor_ctx.left_nz;
            bot_left_modes = fdc.neighbor_ctx.left_intra4x4_mode;
            bot_left_avail = fdc.neighbor_ctx.left_available;

            // Advance to next pair
            mb_x += 1;
            if mb_x >= mb_width {
                // Pair row complete — inline deblock + publish with 1-pair-row delay
                inline_deblock_pair_row(fdc, mb_y / 2, mb_width, deblock_enabled);
                mb_x = 0;
                mb_y += 2;
            }
        }

        Ok(mbs_decoded)
    }

    /// Decode a CABAC-coded slice.
    ///
    /// The CABAC slice loop differs from CAVLC:
    /// - Skip is per-MB (not skip_run)
    /// - Arithmetic coding engine replaces exp-golomb/VLC bitreading
    /// - Terminate symbol signals end of slice
    ///
    /// Reference: FFmpeg h264_cabac.c:1920-2499 (ff_h264_decode_mb_cabac).
    #[allow(clippy::too_many_arguments)]
    fn decode_slice_cabac(
        rbsp: &[u8],
        hdr: &SliceHeader,
        sps: &Sps,
        pps: &Pps,
        fdc: &mut FrameDecodeContext,
        ref_pics: &[Arc<SharedPicture>],
        ref_pics_l1: &[Arc<SharedPicture>],
    ) -> Result<u32> {
        use crate::cabac::{
            CabacDecodeCache, CabacNeighborCtx, CabacReader, decode_cabac_mb_skip, decode_mb_cabac,
        };
        use crate::cabac_tables::init_cabac_states;

        let mb_width = sps.mb_width;
        let mb_height = sps.mb_height;
        let total_mbs = mb_width * mb_height;

        // Byte-align: CABAC data starts at byte boundary after slice header
        let data_start = hdr.header_bits.div_ceil(8);
        if data_start >= rbsp.len() {
            return Err(invalid_data("cabac slice starts past end of rbsp"));
        }

        // Add zero tail padding for safe CABAC renormalization reads
        let mut padded = Vec::with_capacity(rbsp.len() - data_start + 64);
        padded.extend_from_slice(&rbsp[data_start..]);
        padded.resize(padded.len() + 64, 0);

        let mut reader = CabacReader::new(&padded)?;

        // Initialize context states
        let is_intra = hdr.slice_type.is_intra();
        let mut cabac_state = init_cabac_states(hdr.slice_qp, is_intra, hdr.cabac_init_idc);

        // Initialize CABAC neighbor context
        let mut cabac_nb = CabacNeighborCtx::new(mb_width, mb_height);

        // Decode macroblocks
        let first_mb = hdr.first_mb_in_slice;
        let mut mbs_decoded = 0u32;
        let is_inter_slice = hdr.slice_type.is_p() || hdr.slice_type.is_b();
        let is_mbaff = !sps.frame_mbs_only_flag;
        let deblock_enabled = true;

        if is_mbaff {
            mbs_decoded = Self::decode_slice_cabac_mbaff(
                &mut reader,
                &mut cabac_state,
                &mut cabac_nb,
                hdr,
                sps,
                pps,
                fdc,
                ref_pics,
                ref_pics_l1,
                is_inter_slice,
            )?;
        } else {
            let mut mb_addr = first_mb;

            while mb_addr < total_mbs {
                let mb_x = mb_addr % mb_width;
                let mb_y = mb_addr / mb_width;
                let mb_idx = mb_addr as usize;

                // Update neighbor context at the start of each row
                if mb_x == 0 {
                    fdc.neighbor_ctx.new_row();
                    fdc.neighbor_ctx.top_available = mb_y > 0
                        && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
                } else if mb_addr == first_mb {
                    fdc.neighbor_ctx.left_available = false;
                    fdc.neighbor_ctx.top_available = mb_y > 0
                        && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
                }

                // For inter slices: decode skip flag
                if is_inter_slice {
                    let nb = crate::mb::compute_mbaff_neighbors(
                        mb_x,
                        mb_y,
                        mb_width,
                        false,
                        false,
                        &fdc.mb_field_flag,
                        &fdc.slice_table,
                        fdc.current_slice,
                    );
                    let skip = decode_cabac_mb_skip(
                        &mut reader,
                        &mut cabac_state,
                        &cabac_nb,
                        nb.left_idx,
                        nb.top_idx,
                        hdr.slice_type.is_b(),
                    );

                    if skip != 0 {
                        // Handle skip MB
                        if hdr.slice_type.is_b() {
                            mb::decode_b_skip_mb(fdc, hdr, mb_x, mb_y, ref_pics, ref_pics_l1);
                        } else {
                            mb::decode_skip_mb(fdc, hdr, mb_x, mb_y, ref_pics, ref_pics_l1);
                        }
                        fdc.last_qscale_diff = 0;

                        // Update CABAC neighbor context for skip MB
                        cabac_nb.update_after_mb(
                            mb_idx, true, false, false, false, 0, 0, &[0; 24], false,
                        );
                        let skip_modes = if pps.constrained_intra_pred {
                            [-1i8; 16]
                        } else {
                            [2i8; 16]
                        };
                        cabac_nb.store_intra4x4_modes(mb_idx, &skip_modes);
                        cabac_nb.update_mvd_ref_skip(mb_idx);

                        mb_addr += 1;
                        mbs_decoded += 1;
                        // Inline deblock + publish with 1-row delay
                        if mb_addr.is_multiple_of(mb_width) {
                            inline_deblock_row(
                                fdc,
                                mb_addr / mb_width - 1,
                                mb_width,
                                deblock_enabled,
                            );
                        }

                        // Check terminate after skip
                        if reader.get_cabac_terminate() {
                            break;
                        }
                        continue;
                    }
                }

                // Update per-MB neighbor availability (slice-boundary aware)
                fdc.neighbor_ctx.top_available =
                    mb_y > 0 && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;

                // Decode coded MB via CABAC
                let mut cache = CabacDecodeCache::new();
                let nb = crate::mb::compute_mbaff_neighbors(
                    mb_x,
                    mb_y,
                    mb_width,
                    false,
                    false,
                    &fdc.mb_field_flag,
                    &fdc.slice_table,
                    fdc.current_slice,
                );
                let mut mb = decode_mb_cabac(
                    &mut reader,
                    &mut cabac_state,
                    hdr.slice_type,
                    pps,
                    &cabac_nb,
                    &fdc.neighbor_ctx,
                    &fdc.slice_table,
                    fdc.current_slice,
                    mb_x,
                    mb_y,
                    mb_width,
                    hdr.num_ref_idx_l0_active,
                    hdr.num_ref_idx_l1_active,
                    fdc.last_qscale_diff,
                    &mut cache,
                    fdc.direct_8x8_inference_flag,
                    fdc.decode_chroma,
                    false, // non-MBAFF: always frame mode
                    &nb,
                )?;

                // Apply entropy-agnostic processing (dequant, IDCT, pred, MC, neighbor update)
                mb::apply_macroblock(fdc, &mut mb, hdr, pps, mb_x, mb_y, ref_pics, ref_pics_l1)?;

                // Write back MVD/ref from scan8 cache to flat storage, then update scalar fields
                cache.write_back(&mut cabac_nb, mb_idx);
                cabac_nb.update_after_mb(
                    mb_idx,
                    false,
                    mb.is_intra16x16 || mb.is_pcm,
                    mb.is_intra,
                    mb.is_direct,
                    mb.cbp,
                    mb.chroma_pred_mode,
                    &mb.non_zero_count,
                    mb.transform_size_8x8_flag,
                );
                cabac_nb.store_intra4x4_modes(
                    mb_idx,
                    &mb_intra4x4_modes_i8(&mb, pps.constrained_intra_pred),
                );

                mb_addr += 1;
                mbs_decoded += 1;
                // Inline deblock + publish with 1-row delay
                if mb_addr.is_multiple_of(mb_width) {
                    inline_deblock_row(fdc, mb_addr / mb_width - 1, mb_width, deblock_enabled);
                }

                // Check terminate
                if reader.get_cabac_terminate() {
                    break;
                }
            }
        }

        Ok(mbs_decoded)
    }

    /// Decode a CABAC-coded slice in MBAFF mode (MB pairs).
    ///
    /// Processes MBs in pairs: top (even mb_y) then bottom (odd mb_y).
    /// Reads mb_field_decoding_flag from CABAC at the top MB of each pair.
    ///
    /// Reference: FFmpeg h264_slice.c:2612-2680 (CABAC MBAFF pair loop),
    ///            h264_cabac.c:1935-1962 (MBAFF skip/field flag logic).
    #[allow(clippy::too_many_arguments)]
    fn decode_slice_cabac_mbaff(
        reader: &mut crate::cabac::CabacReader<'_>,
        cabac_state: &mut [u8; 1024],
        cabac_nb: &mut crate::cabac::CabacNeighborCtx,
        hdr: &SliceHeader,
        sps: &Sps,
        pps: &Pps,
        fdc: &mut FrameDecodeContext,
        ref_pics: &[Arc<SharedPicture>],
        ref_pics_l1: &[Arc<SharedPicture>],
        is_inter_slice: bool,
    ) -> Result<u32> {
        use crate::cabac::{
            CabacDecodeCache, decode_cabac_field_decoding_flag, decode_cabac_mb_skip,
            decode_mb_cabac,
        };

        let mb_width = sps.mb_width;
        let mb_height = sps.mb_height;
        let first_mb = hdr.first_mb_in_slice;

        // For MBAFF, first_mb_in_slice is in pair units.
        // Convert to (mb_x, pair_row_y):
        let mut mb_x = first_mb % mb_width;
        let mut mb_y = (first_mb / mb_width) * 2; // pair row → even mb_y
        let first_mb_flat = mb_x + mb_y * mb_width;

        let mut mbs_decoded = 0u32;
        let mut mb_field_decoding_flag = false;
        let deblock_enabled = true;

        // Two separate left-side contexts for MBAFF: one for top MBs, one for bottom MBs.
        // After each pair, the top row's left context holds the top MB's right edge,
        // and the bottom row's holds the bottom MB's right edge.
        let mut top_left_nz = [0u8; 8];
        let mut top_left_modes = [-1i8; 4];
        let mut top_left_avail = false;
        let mut bot_left_nz = [0u8; 8];
        let mut bot_left_modes = [-1i8; 4];
        let mut bot_left_avail = false;
        // MBAFF: save top-row context (intra4x4 modes + NNZ) for the bottom MB.
        // When the top MB of a pair runs update_after_mb, it overwrites
        // top_intra4x4_mode at the current column. For field-mode bottom MBs,
        // the actual top neighbor is from the previous pair row, so we must
        // save the pre-overwrite values.
        let mut saved_top_modes = [-1i8; 4];
        let mut saved_top_nz_luma = [0u8; 4];

        loop {
            if mb_y >= mb_height {
                break;
            }

            // --- Process MB pair: top (mb_y) and bottom (mb_y + 1) ---

            // MBAFF skip state for pair interaction (FFmpeg: prev_mb_skipped, next_mb_skipped)
            let mut top_was_skipped = false;
            let mut bot_skip_preread = false;
            let mut bot_skip_value = false;

            for pair_pos in 0..2u32 {
                let cur_y = mb_y + pair_pos;
                let mb_addr = mb_x + cur_y * mb_width;
                let mb_idx = mb_addr as usize;

                // MBAFF left-side neighbor context:
                // The top MB's left neighbor is the TOP MB of the previous pair.
                // The bottom MB's left neighbor is the BOTTOM MB of the previous pair.
                // We maintain separate left contexts for top and bottom rows.
                if pair_pos == 0 {
                    // Top MB: load top-row left context.
                    fdc.neighbor_ctx.left_nz = top_left_nz;
                    fdc.neighbor_ctx.left_intra4x4_mode = top_left_modes;
                    fdc.neighbor_ctx.left_available = top_left_avail;
                    if mb_x == 0 {
                        fdc.neighbor_ctx.new_row();
                    }
                    // MBAFF: save top-row context BEFORE the top MB overwrites it.
                    // The bottom MB needs these values when it's field-mode
                    // (its spatial top neighbor is from the previous pair row).
                    let base = mb_x as usize * 4;
                    saved_top_modes
                        .copy_from_slice(&fdc.neighbor_ctx.top_intra4x4_mode[base..base + 4]);
                    saved_top_nz_luma
                        .copy_from_slice(&fdc.neighbor_ctx.top_nz_luma[base..base + 4]);
                } else {
                    // Bottom MB: save top MB's right edge as the next pair's top-row left.
                    top_left_nz = fdc.neighbor_ctx.left_nz;
                    top_left_modes = fdc.neighbor_ctx.left_intra4x4_mode;
                    top_left_avail = fdc.neighbor_ctx.left_available;
                    // Load bottom-row left context.
                    fdc.neighbor_ctx.left_nz = bot_left_nz;
                    fdc.neighbor_ctx.left_intra4x4_mode = bot_left_modes;
                    fdc.neighbor_ctx.left_available = bot_left_avail;
                    if mb_x == 0 {
                        fdc.neighbor_ctx.new_row();
                    }
                    // MBAFF field-mode: restore pre-top-MB top context.
                    // For field-mode bottom MBs, the spatial above neighbor
                    // is from the previous pair row (mb_y - 2*stride), not
                    // the current pair's top MB. Restore the saved values.
                    if mb_field_decoding_flag {
                        let base = mb_x as usize * 4;
                        fdc.neighbor_ctx.top_intra4x4_mode[base..base + 4]
                            .copy_from_slice(&saved_top_modes);
                        fdc.neighbor_ctx.top_nz_luma[base..base + 4]
                            .copy_from_slice(&saved_top_nz_luma);
                    }
                }
                fdc.neighbor_ctx.top_available = cur_y > 0
                    && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
                // First MB of slice may lack left neighbor
                if mb_addr == first_mb_flat && mb_x == 0 {
                    fdc.neighbor_ctx.left_available = false;
                }

                // For inter slices: decode skip flag
                if is_inter_slice {
                    // Bottom MB of pair: if top was skipped, use pre-decoded skip value
                    let skip = if pair_pos == 1 && top_was_skipped && bot_skip_preread {
                        bot_skip_value
                    } else {
                        {
                            let (skip_left, skip_top) = crate::mb::mbaff_skip_neighbors(
                                mb_x,
                                cur_y,
                                mb_width,
                                true,
                                mb_field_decoding_flag,
                                &fdc.mb_field_flag,
                                &fdc.slice_table,
                                fdc.current_slice,
                            );
                            decode_cabac_mb_skip(
                                reader,
                                cabac_state,
                                cabac_nb,
                                skip_left,
                                skip_top,
                                hdr.slice_type.is_b(),
                            ) != 0
                        }
                    };

                    if skip {
                        if pair_pos == 0 {
                            // Top MB is skip. Also decode bottom MB skip flag.
                            top_was_skipped = true;
                            let bot_skip = {
                                let (skip_left, skip_top) = crate::mb::mbaff_skip_neighbors(
                                    mb_x,
                                    cur_y + 1,
                                    mb_width,
                                    true,
                                    mb_field_decoding_flag,
                                    &fdc.mb_field_flag,
                                    &fdc.slice_table,
                                    fdc.current_slice,
                                );
                                decode_cabac_mb_skip(
                                    reader,
                                    cabac_state,
                                    cabac_nb,
                                    skip_left,
                                    skip_top,
                                    hdr.slice_type.is_b(),
                                ) != 0
                            };
                            bot_skip_preread = true;
                            bot_skip_value = bot_skip;
                            if !bot_skip {
                                // Bottom is coded: read mb_field_decoding_flag
                                let above_pair_field = if mb_y >= 2 {
                                    let above_idx = (mb_x + (mb_y - 2) * mb_width) as usize;
                                    if fdc.slice_table[above_idx] == fdc.current_slice {
                                        cabac_nb.mb_field_flag[above_idx]
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                };
                                mb_field_decoding_flag = decode_cabac_field_decoding_flag(
                                    reader,
                                    cabac_state,
                                    mb_field_decoding_flag,
                                    mb_x,
                                    above_pair_field,
                                );
                            }
                        }

                        // Set field mode for pixel reconstruction
                        fdc.mb_field = mb_field_decoding_flag;

                        // Decode skip MB
                        if hdr.slice_type.is_b() {
                            mb::decode_b_skip_mb(fdc, hdr, mb_x, cur_y, ref_pics, ref_pics_l1);
                        } else {
                            mb::decode_skip_mb(fdc, hdr, mb_x, cur_y, ref_pics, ref_pics_l1);
                        }
                        fdc.last_qscale_diff = 0;

                        cabac_nb.update_after_mb(
                            mb_idx, true, false, false, false, 0, 0, &[0; 24], false,
                        );
                        let skip_modes = if pps.constrained_intra_pred {
                            [-1i8; 16]
                        } else {
                            [2i8; 16]
                        };
                        cabac_nb.store_intra4x4_modes(mb_idx, &skip_modes);
                        cabac_nb.update_mvd_ref_skip(mb_idx);
                        cabac_nb.mb_field_flag[mb_idx] = mb_field_decoding_flag;
                        mbs_decoded += 1;
                        continue; // next pair_pos
                    }
                }

                // Coded MB (or intra slice): read mb_field_decoding_flag on top MB
                if pair_pos == 0 {
                    // Look up above pair's mb_field_decoding_flag for CABAC context.
                    // Above pair is at (mb_x, mb_y-2). Check slice boundary.
                    let above_pair_field = if mb_y >= 2 {
                        let above_idx = (mb_x + (mb_y - 2) * mb_width) as usize;
                        if fdc.slice_table[above_idx] == fdc.current_slice {
                            cabac_nb.mb_field_flag[above_idx]
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    mb_field_decoding_flag = decode_cabac_field_decoding_flag(
                        reader,
                        cabac_state,
                        mb_field_decoding_flag,
                        mb_x,
                        above_pair_field,
                    );
                    top_was_skipped = false;
                }

                // Set field mode for pixel reconstruction
                fdc.mb_field = mb_field_decoding_flag;

                // Update per-MB top availability
                fdc.neighbor_ctx.top_available = cur_y > 0
                    && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;

                // Decode coded MB via CABAC
                let mut cache = CabacDecodeCache::new();
                let nb = crate::mb::compute_mbaff_neighbors(
                    mb_x,
                    cur_y,
                    mb_width,
                    true,
                    mb_field_decoding_flag,
                    &fdc.mb_field_flag,
                    &fdc.slice_table,
                    fdc.current_slice,
                );
                let mut mb = decode_mb_cabac(
                    reader,
                    cabac_state,
                    hdr.slice_type,
                    pps,
                    cabac_nb,
                    &fdc.neighbor_ctx,
                    &fdc.slice_table,
                    fdc.current_slice,
                    mb_x,
                    cur_y,
                    mb_width,
                    hdr.num_ref_idx_l0_active,
                    hdr.num_ref_idx_l1_active,
                    fdc.last_qscale_diff,
                    &mut cache,
                    fdc.direct_8x8_inference_flag,
                    fdc.decode_chroma,
                    mb_field_decoding_flag,
                    &nb,
                )?;

                mb::apply_macroblock(fdc, &mut mb, hdr, pps, mb_x, cur_y, ref_pics, ref_pics_l1)?;

                cache.write_back(cabac_nb, mb_idx);
                cabac_nb.update_after_mb(
                    mb_idx,
                    false,
                    mb.is_intra16x16 || mb.is_pcm,
                    mb.is_intra,
                    mb.is_direct,
                    mb.cbp,
                    mb.chroma_pred_mode,
                    &mb.non_zero_count,
                    mb.transform_size_8x8_flag,
                );
                cabac_nb.store_intra4x4_modes(
                    mb_idx,
                    &mb_intra4x4_modes_i8(&mb, pps.constrained_intra_pred),
                );
                cabac_nb.mb_field_flag[mb_idx] = mb_field_decoding_flag;
                mbs_decoded += 1;
            }

            // After the pair, save bottom MB's right edge as next pair's bottom-row left.
            bot_left_nz = fdc.neighbor_ctx.left_nz;
            bot_left_modes = fdc.neighbor_ctx.left_intra4x4_mode;
            bot_left_avail = fdc.neighbor_ctx.left_available;

            // Advance to next pair
            mb_x += 1;
            if mb_x >= mb_width {
                // Pair row complete — inline deblock + publish with 1-pair-row delay
                inline_deblock_pair_row(fdc, mb_y / 2, mb_width, deblock_enabled);
                mb_x = 0;
                mb_y += 2; // Skip to next pair row
            }

            // A slice that terminates at a pair-row boundary still completes that row.
            if reader.get_cabac_terminate() {
                break;
            }
        }

        Ok(mbs_decoded)
    }

    /// Convert a deblocked `InFlightDecode` to a `VideoFrame`.
    fn fdc_to_frame(&self, in_flight: &InFlightDecode, is_corrupt: bool) -> VideoFrame {
        let fdc = &in_flight.fdc;
        let hdr = &in_flight.last_hdr;

        let coded_w = if in_flight.coded_width > 0 {
            in_flight.coded_width as usize
        } else {
            fdc.pic.width as usize
        };
        let coded_h = if in_flight.coded_height > 0 {
            in_flight.coded_height as usize
        } else {
            fdc.pic.height as usize
        };
        let chroma_w = coded_w / 2;
        let chroma_h = coded_h / 2;

        let mut y_plane =
            BorderedPlane::new(&self.core_pool, coded_w, coded_h, BorderPadding::NONE);
        for row in 0..coded_h {
            let src_start = row * fdc.pic.y_stride;
            if let Some(dst) = y_plane.row_mut(row) {
                dst.copy_from_slice(&fdc.pic.y[src_start..src_start + coded_w]);
            }
        }

        let mut u_plane =
            BorderedPlane::new(&self.core_pool, chroma_w, chroma_h, BorderPadding::NONE);
        for row in 0..chroma_h {
            let src_start = row * fdc.pic.uv_stride;
            if let Some(dst) = u_plane.row_mut(row) {
                dst.copy_from_slice(&fdc.pic.u[src_start..src_start + chroma_w]);
            }
        }

        let mut v_plane =
            BorderedPlane::new(&self.core_pool, chroma_w, chroma_h, BorderPadding::NONE);
        for row in 0..chroma_h {
            let src_start = row * fdc.pic.uv_stride;
            if let Some(dst) = v_plane.row_mut(row) {
                dst.copy_from_slice(&fdc.pic.v[src_start..src_start + chroma_w]);
            }
        }

        let pict_type = match hdr.slice_type {
            SliceType::I | SliceType::SI => PictureType::I,
            SliceType::P | SliceType::SP => PictureType::P,
            SliceType::B => PictureType::B,
        };

        let mut frame = VideoFrame::from_planes(
            coded_w as u32,
            coded_h as u32,
            PixelLayout::YUV420P,
            vec![y_plane, u_plane, v_plane],
        );
        frame.color = in_flight.color;
        frame.crop = in_flight.crop;
        frame.sar = in_flight.sar;
        frame.picture_type = pict_type;
        frame.pts = in_flight.pts;
        frame.pkt_dts = in_flight.dts;
        frame.duration = in_flight.duration;
        frame.time_base = in_flight.time_base;

        if in_flight.is_key {
            frame.flags |= FrameFlags::KEY;
        }
        if is_corrupt {
            frame.flags |= FrameFlags::CORRUPT;
        }

        frame
    }
}

// ---------------------------------------------------------------------------
// Helper functions for parsing NAL units from raw bytes (avcC extradata)
// ---------------------------------------------------------------------------

/// Parse an SPS from a raw NAL unit (header byte + payload).
fn parse_sps_from_nal(nal_bytes: &[u8]) -> Result<Sps> {
    if nal_bytes.is_empty() {
        return Err(invalid_data("empty avcC SPS NAL"));
    }
    let rbsp = nal_to_rbsp(&nal_bytes[1..]);
    parse_sps(&rbsp)
}

/// Parse a PPS from a raw NAL unit (header byte + payload).
fn parse_pps_from_nal(nal_bytes: &[u8], sps_list: &[Option<Sps>; 32]) -> Result<Pps> {
    if nal_bytes.is_empty() {
        return Err(invalid_data("empty avcC PPS NAL"));
    }
    let rbsp = nal_to_rbsp(&nal_bytes[1..]);
    parse_pps(&rbsp, sps_list)
}

// ---------------------------------------------------------------------------
// Decoder trait implementation
// ---------------------------------------------------------------------------

impl Decoder for H264Decoder {
    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<SendStatus> {
        if self.error_policy == ErrorPolicy::Strict
            && let Some(err) = self.fatal_error.take()
        {
            return Err(err);
        }

        match pkt {
            Some(pkt) => {
                let data = pkt.data.data();

                // Split into NAL units using the appropriate framing.
                let nalus = if self.nalff_length_size > 0 {
                    split_nalff(data, self.nalff_length_size)
                } else {
                    split_annex_b(data)
                };

                for nalu in &nalus {
                    if let Err(e) =
                        self.process_nal(nalu, pkt.pts, pkt.dts, pkt.duration, pkt.time_base)
                    {
                        self.health.record(HealthEvent {
                            track: self.track_id,
                            pts: pkt.pts,
                            byte_offset: pkt.pos,
                            kind: HealthIssue::CorruptSyntax,
                            detail: "h264 nal decode error",
                        });
                        if self.error_policy == ErrorPolicy::Strict {
                            return Err(e);
                        }
                    }
                    if self.error_policy == ErrorPolicy::Strict
                        && let Some(err) = self.fatal_error.take()
                    {
                        return Err(err);
                    }
                }

                Ok(SendStatus::Accepted)
            }
            None => {
                // Drain mode: no more packets will be sent.
                self.finish_current_frame();
                self.join_all_in_flight();

                if self.error_policy == ErrorPolicy::Strict
                    && let Some(err) = self.fatal_error.take()
                {
                    return Err(err);
                }

                // Flush all remaining delayed_pics using barrier-aware min-POC search.
                while !self.delayed_pics.is_empty() {
                    let out_idx = if self.delayed_pics[0].2 {
                        0
                    } else {
                        let barrier = self
                            .delayed_pics
                            .iter()
                            .enumerate()
                            .skip(1)
                            .find(|(_, (_, _, reset))| *reset)
                            .map(|(i, _)| i)
                            .unwrap_or(self.delayed_pics.len());
                        self.delayed_pics[..barrier]
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, (poc, _, _))| *poc)
                            .map(|(i, _)| i)
                            .unwrap()
                    };
                    let (_, f, _) = self.delayed_pics.remove(out_idx);
                    self.output_queue.push_back(f);
                }

                self.draining = true;
                Ok(SendStatus::Accepted)
            }
        }
    }

    fn receive(&mut self) -> Result<RecvStatus<MediaUnit>> {
        if self.error_policy == ErrorPolicy::Strict
            && let Some(err) = self.fatal_error.take()
        {
            return Err(err);
        }
        if let Some(frame) = self.output_queue.pop_front() {
            Ok(RecvStatus::Ready(MediaUnit::Video(frame)))
        } else if self.draining {
            Ok(RecvStatus::Eof)
        } else {
            Ok(RecvStatus::NeedMore)
        }
    }

    fn flush(&mut self) {
        self.finish_current_frame();
        self.join_all_in_flight();
        self.pending_slices.clear();
        self.output_queue.clear();
        self.draining = false;
        self.frame_num = 0;
        self.dpb.clear();
        self.ref_list_l0.clear();
        self.ref_list_l1.clear();
        self.current_fdc = None;
        self.current_last_hdr = None;
        self.prev_poc_msb = 0;
        self.prev_poc_lsb = 0;
        self.current_poc = 0;
        self.current_nal_ref_idc = 0;
        self.current_is_idr = false;
        self.current_is_key = false;
        self.pending_recovery_point = false;
        self.fatal_error = None;
        self.current_pts = Timestamp::NONE;
        self.current_dts = Timestamp::NONE;
        self.current_duration = 0;
        self.reorder_depth = 0;
        self.has_bitstream_restriction = false;
        self.last_pocs = [i64::MIN; 16];
        self.delayed_pics.clear();
        // SPS/PPS are retained across flush (matching FFmpeg behavior).
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

// ---------------------------------------------------------------------------
// Factory registration
// ---------------------------------------------------------------------------

/// Decoder factory for `CodecId::H264`.
pub struct H264DecoderFactory;

impl DecoderFactory for H264DecoderFactory {
    fn descriptor(&self) -> &DecoderDescriptor {
        static DESC: DecoderDescriptor = DecoderDescriptor {
            codec_id: CodecId::H264,
            name: "h264",
            long_name: "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
            media_type: MediaType::Video,
        };
        &DESC
    }

    fn open(&self, track: &TrackDescriptor, ctx: CodecContext) -> Result<Box<dyn Decoder>> {
        Ok(Box::new(H264Decoder::open(track, ctx)?))
    }
}

/// Register the H.264 decoder factory on a `wedeo_core::Registry`.
pub fn register_h264(reg: &mut Registry) {
    reg.register_decoder(Arc::new(H264DecoderFactory));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::{StreamParams, VideoStreamParams};

    fn make_track() -> TrackDescriptor {
        TrackDescriptor::new(
            TrackId(0),
            CodecId::H264,
            Rational::new(1, 90_000),
            StreamParams::Video(VideoStreamParams::new(0, 0, PixelLayout::YUV420P)),
        )
    }

    fn make_inline_ctx() -> CodecContext {
        CodecContext {
            threads: 1,
            ..Default::default()
        }
    }

    #[test]
    fn create_decoder() {
        let decoder = H264Decoder::open(&make_track(), make_inline_ctx());
        assert!(decoder.is_ok());
    }

    #[test]
    fn decoder_drain_returns_eof() {
        let mut decoder = H264Decoder::open(&make_track(), make_inline_ctx()).unwrap();

        // No packets sent, receive should return NeedMore
        assert!(matches!(decoder.receive().unwrap(), RecvStatus::NeedMore));

        // Send drain signal
        assert_eq!(decoder.send_packet(None).unwrap(), SendStatus::Accepted);

        // Should return Eof
        assert!(matches!(decoder.receive().unwrap(), RecvStatus::Eof));
    }

    #[test]
    fn decoder_flush_clears_state() {
        let mut decoder = H264Decoder::open(&make_track(), make_inline_ctx()).unwrap();

        decoder.send_packet(None).unwrap();
        assert!(decoder.draining);

        decoder.flush();
        assert!(!decoder.draining);
        assert!(decoder.output_queue.is_empty());
    }

    #[test]
    fn decoder_processes_sps_pps() {
        let mut decoder = H264Decoder::open(&make_track(), make_inline_ctx()).unwrap();

        // Build a minimal Annex B stream: SPS + PPS (no slice data)
        // SPS: Baseline 320x240 (from sps.rs test)
        let sps_rbsp: &[u8] = &[0x42, 0x80, 0x1E, 0xF4, 0x0A, 0x0F, 0xC0];

        // Build PPS RBSP for Baseline (CAVLC, default settings)
        let pps_rbsp = build_test_pps();

        // Assemble Annex B stream (SPS + PPS only, no slice)
        let mut stream = Vec::new();
        // SPS
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67]);
        stream.extend_from_slice(sps_rbsp);
        // PPS
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x68]);
        stream.extend_from_slice(&pps_rbsp);

        let pkt = Packet::from_slice(&stream);
        decoder.send_packet(Some(&pkt)).unwrap();

        // SPS/PPS processing should not produce frames
        assert!(matches!(decoder.receive().unwrap(), RecvStatus::NeedMore));

        // Verify SPS and PPS were stored
        assert!(decoder.sps_list[0].is_some());
        assert!(decoder.pps_list[0].is_some());

        // Verify dimensions updated
        assert_eq!(decoder.width, 320);
        assert_eq!(decoder.height, 240);
    }

    #[test]
    fn decoder_handles_invalid_slice_gracefully() {
        let mut decoder = H264Decoder::open(&make_track(), make_inline_ctx()).unwrap();

        // Build SPS + PPS + IDR with only header (no MB data).
        let sps_rbsp: &[u8] = &[0x42, 0x80, 0x1E, 0xF4, 0x0A, 0x0F, 0xC0];
        let pps_rbsp = build_test_pps();
        let idr_rbsp = build_test_idr_slice();

        let mut stream = Vec::new();
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67]);
        stream.extend_from_slice(sps_rbsp);
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x68]);
        stream.extend_from_slice(&pps_rbsp);
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65]);
        stream.extend_from_slice(&idr_rbsp);

        let pkt = Packet::from_slice(&stream);
        // send_packet should succeed (errors in individual NALs are recorded, not fatal in Resilient mode)
        decoder.send_packet(Some(&pkt)).unwrap();

        // The IDR slice has not been drained yet, so no frame is ready yet.
        assert!(matches!(decoder.receive().unwrap(), RecvStatus::NeedMore));
    }

    #[test]
    fn decoder_health_and_registry() {
        let mut reg = Registry::new();
        register_h264(&mut reg);
        let factory = reg.find_decoder(CodecId::H264).expect("h264 registered");
        let desc = factory.descriptor();
        assert_eq!(desc.codec_id, CodecId::H264);
        assert_eq!(desc.name, "h264");
        assert_eq!(desc.media_type, MediaType::Video);

        let decoder = factory.open(&make_track(), make_inline_ctx()).unwrap();
        assert!(decoder.health().is_clean());
    }

    // --- Test helpers ---

    /// Build a minimal PPS bitstream for Baseline profile.
    fn build_test_pps() -> Vec<u8> {
        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // pps_id = 0
        encode_ue(&mut bits, 0); // sps_id = 0
        bits.push(false); // entropy_coding_mode_flag = 0 (CAVLC)
        bits.push(false); // bottom_field_pic_order_in_frame_present = 0
        encode_ue(&mut bits, 0); // num_slice_groups_minus1 = 0
        encode_ue(&mut bits, 0); // num_ref_idx_l0_default_active_minus1 = 0
        encode_ue(&mut bits, 0); // num_ref_idx_l1_default_active_minus1 = 0
        bits.push(false); // weighted_pred_flag = 0
        bits.push(false); // weighted_bipred_idc = 0 (2 bits)
        bits.push(false);
        encode_se(&mut bits, 0); // pic_init_qp_minus26 = 0
        encode_se(&mut bits, 0); // pic_init_qs_minus26 = 0
        encode_se(&mut bits, 0); // chroma_qp_index_offset = 0
        bits.push(true); // deblocking_filter_parameters_present = 1
        bits.push(false); // constrained_intra_pred = 0
        bits.push(false); // redundant_pic_cnt_present = 0
        bits_to_bytes(&bits)
    }

    /// Build a minimal IDR I-slice header for testing.
    fn build_test_idr_slice() -> Vec<u8> {
        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // first_mb_in_slice = 0
        encode_ue(&mut bits, 7); // slice_type = 7 (I, all same)
        encode_ue(&mut bits, 0); // pps_id = 0
        push_bits(&mut bits, 0, 4); // frame_num = 0 (log2_max_frame_num=4)
        encode_ue(&mut bits, 0); // idr_pic_id = 0
        push_bits(&mut bits, 0, 4); // pic_order_cnt_lsb = 0 (log2_max_poc_lsb=4)
        // dec_ref_pic_marking (IDR, nal_ref_idc=3):
        bits.push(false); // no_output_of_prior_pics = 0
        bits.push(false); // long_term_reference_flag = 0
        encode_se(&mut bits, 0); // slice_qp_delta = 0
        // deblocking:
        encode_ue(&mut bits, 0); // disable_deblocking_filter_idc = 0
        encode_se(&mut bits, 0); // alpha_offset_div2 = 0
        encode_se(&mut bits, 0); // beta_offset_div2 = 0
        bits_to_bytes(&bits)
    }

    fn encode_ue(bits: &mut Vec<bool>, val: u32) {
        let code = val + 1;
        let n = 32 - code.leading_zeros();
        for _ in 0..n - 1 {
            bits.push(false);
        }
        for i in (0..n).rev() {
            bits.push((code >> i) & 1 != 0);
        }
    }

    fn encode_se(bits: &mut Vec<bool>, val: i32) {
        let ue_val = if val <= 0 {
            (-2 * val) as u32
        } else {
            (2 * val - 1) as u32
        };
        encode_ue(bits, ue_val);
    }

    fn push_bits(bits: &mut Vec<bool>, val: u32, n: usize) {
        for i in (0..n).rev() {
            bits.push((val >> i) & 1 != 0);
        }
    }

    fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
        let num_bytes = bits.len().div_ceil(8);
        let mut bytes = vec![0u8; num_bytes];
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        bytes
    }
}
