// H.264/AVC decoder.
//
// Implements NAL unit parsing (SPS, PPS, slice header) and macroblock-level
// decoding for I-frames via CAVLC, intra prediction, dequantization, IDCT,
// and in-loop deblocking.
//
// Reference: FFmpeg libavcodec/h264dec.c, h264_slice.c

use std::collections::VecDeque;

#[cfg(feature = "tracing-detail")]
use tracing::trace;
use tracing::{debug, warn};
use wedeo_codec::bitstream::{BitRead, BitReadBE, get_ue_golomb};
use wedeo_codec::decoder::{CodecParameters, Decoder, DecoderDescriptor};
use wedeo_codec::descriptor::{CodecCapabilities, CodecDescriptor, CodecProperties};
use wedeo_codec::registry::DecoderFactory;
use wedeo_core::buffer::Buffer;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::frame::{Frame, FrameData, FrameFlags, FramePlane, PictureType};
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::Packet;
use wedeo_core::pixel_format::PixelFormat;

use crate::deblock::{self, PictureBuffer};
use crate::dpb::{Dpb, DpbEntry, RefStatus};
use crate::mb::{self, FrameDecodeContext};
use crate::nal::{NalUnit, NalUnitType, split_annex_b, split_nalff};
use crate::pps::{Pps, parse_pps};
use crate::refs;
use crate::slice::{SliceHeader, SliceType, parse_slice_header};
use crate::sps::{Sps, parse_sps};

// ---------------------------------------------------------------------------
// Decoder state
// ---------------------------------------------------------------------------

/// H.264/AVC decoder.
///
/// Parses SPS, PPS, and slice headers from NAL units. Decodes I-frames via
/// CAVLC, intra prediction, dequantization, IDCT, and in-loop deblocking.
pub struct H264Decoder {
    /// Stored Sequence Parameter Sets, indexed by sps_id (0..31).
    sps_list: [Option<Sps>; 32],
    /// Stored Picture Parameter Sets, indexed by pps_id (0..255).
    pps_list: Box<[Option<Pps>; 256]>,
    /// Decoded frame width (from active SPS, after cropping).
    width: u32,
    /// Decoded frame height (from active SPS, after cropping).
    height: u32,
    /// Running frame counter for output PTS assignment.
    frame_num: u64,
    /// Queue of decoded frames awaiting output.
    output_queue: VecDeque<Frame>,
    /// True once send_packet(None) has been called (drain mode).
    draining: bool,
    /// NALFF length size for MP4/avcC-style streams (0 = Annex B).
    nalff_length_size: u8,
    /// Codec descriptor for the Decoder trait.
    codec_descriptor: CodecDescriptor,
    /// In-progress frame decode context for multi-slice frames.
    current_fdc: Option<FrameDecodeContext>,
    /// Total MBs decoded so far for the current frame.
    current_mbs_decoded: u32,
    /// Total MBs expected for the current frame.
    current_total_mbs: u32,
    /// Last slice header for the current frame (for deblocking parameters).
    current_last_hdr: Option<SliceHeader>,
    /// PTS of the current frame being decoded.
    current_pts: i64,
    /// Decoded Picture Buffer for reference picture management.
    dpb: Dpb,
    /// Reference list 0 (DPB indices) for the current slice.
    ref_list_l0: Vec<usize>,
    /// Reference list 1 (DPB indices) for B-slices.
    ref_list_l1: Vec<usize>,
    /// Whether the current NAL is an IDR.
    current_is_idr: bool,
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
    /// Crop offsets in pixels from the active SPS.
    crop_left: u32,
    crop_top: u32,
    /// Output frame counter for sequential PTS assignment during reordering.
    output_frame_counter: i64,
    /// Delayed picture buffer for POC-ordered output (matching FFmpeg's
    /// `delayed_pic`). Frames are inserted here and the minimum-POC frame
    /// is output when `delayed_pics.len() > reorder_depth`.
    /// The bool flag is `mmco_reset` — set when MMCO-5 or out_of_order==16
    /// causes a POC sequence restart. Acts as a barrier in min-POC search.
    delayed_pics: Vec<(i32, Frame, bool)>, // (poc, frame, mmco_reset)
}

impl H264Decoder {
    /// Create a new H264Decoder from codec parameters.
    pub fn new(params: CodecParameters) -> Result<Self> {
        let mut decoder = Self {
            sps_list: Default::default(),
            pps_list: Box::new(std::array::from_fn(|_| None)),
            width: params.width.max(16),
            height: params.height.max(16),
            frame_num: 0,
            output_queue: VecDeque::new(),
            draining: false,
            nalff_length_size: 0,
            codec_descriptor: CodecDescriptor {
                id: CodecId::H264,
                media_type: MediaType::Video,
                name: "h264",
                long_name: "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                properties: CodecProperties::LOSSY.union(CodecProperties::REORDER),
                profiles: &[],
            },
            current_fdc: None,
            current_mbs_decoded: 0,
            current_total_mbs: 0,
            current_last_hdr: None,
            current_pts: 0,
            dpb: Dpb::new(16),
            ref_list_l0: Vec::new(),
            ref_list_l1: Vec::new(),
            current_is_idr: false,
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
            crop_left: 0,
            crop_top: 0,
            output_frame_counter: 0,
            delayed_pics: Vec::new(),
        };

        // Parse avcC extradata if present (MP4/NALFF format).
        // avcC box layout:
        //   byte 0: configurationVersion (1)
        //   byte 1: AVCProfileIndication
        //   byte 2: profile_compatibility
        //   byte 3: AVCLevelIndication
        //   byte 4: 6 reserved bits (111111) + lengthSizeMinusOne (2 bits)
        //   byte 5: 3 reserved bits (111) + numOfSequenceParameterSets (5 bits)
        //   then: { u16 spsLength, spsNALUnit[spsLength] } * numSPS
        //   then: u8 numOfPictureParameterSets
        //   then: { u16 ppsLength, ppsNALUnit[ppsLength] } * numPPS
        if params.extradata.len() >= 7 && params.extradata[0] == 1 {
            decoder.nalff_length_size = (params.extradata[4] & 0x03) + 1;
            decoder.parse_avcc_extradata(&params.extradata)?;
        }

        Ok(decoder)
    }

    /// Parse SPS and PPS NAL units from avcC extradata.
    fn parse_avcc_extradata(&mut self, extradata: &[u8]) -> Result<()> {
        if extradata.len() < 7 {
            return Err(Error::InvalidData);
        }

        let mut pos = 5;

        // Parse SPS entries
        let num_sps = (extradata[pos] & 0x1F) as usize;
        pos += 1;
        for _ in 0..num_sps {
            if pos + 2 > extradata.len() {
                return Err(Error::InvalidData);
            }
            let sps_len = u16::from_be_bytes([extradata[pos], extradata[pos + 1]]) as usize;
            pos += 2;
            if pos + sps_len > extradata.len() {
                return Err(Error::InvalidData);
            }
            // The SPS NAL unit includes the header byte; parse_sps expects RBSP after it.
            if sps_len > 1 {
                let nalus = split_annex_b(
                    // Wrap in a fake start code so split_annex_b can parse it,
                    // or just call parse_nal_unit directly. Since we have raw NAL
                    // bytes (header + RBSP), we can manually handle this.
                    &extradata[pos..pos + sps_len],
                );
                // split_annex_b won't work on raw bytes without start codes.
                // Instead, parse manually: byte 0 is NAL header, rest is raw RBSP.
                if let Ok(sps) = parse_sps_from_nal(&extradata[pos..pos + sps_len]) {
                    let id = sps.sps_id as usize;
                    self.apply_sps(&sps);
                    self.sps_list[id] = Some(sps);
                }
                // Suppress unused variable warning
                let _ = nalus;
            }
            pos += sps_len;
        }

        // Parse PPS entries
        if pos >= extradata.len() {
            return Ok(());
        }
        let num_pps = extradata[pos] as usize;
        pos += 1;
        for _ in 0..num_pps {
            if pos + 2 > extradata.len() {
                return Err(Error::InvalidData);
            }
            let pps_len = u16::from_be_bytes([extradata[pos], extradata[pos + 1]]) as usize;
            pos += 2;
            if pos + pps_len > extradata.len() {
                return Err(Error::InvalidData);
            }
            if pps_len > 1
                && let Ok(pps) = parse_pps_from_nal(&extradata[pos..pos + pps_len], &self.sps_list)
            {
                let id = pps.pps_id as usize;
                self.pps_list[id] = Some(pps);
            }
            pos += pps_len;
        }

        Ok(())
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
            for i in 0..=frame_num_in_poc_cycle {
                ep += sps.offset_for_ref_frame[i] as i64;
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
        let w = sps.width();
        let h = sps.height();
        if w > 0 && h > 0 {
            self.width = w;
            self.height = h;
        }
        self.crop_left = sps.crop_left;
        self.crop_top = sps.crop_top;

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
            // supports B-frames (non-Baseline), use a minimum reorder
            // depth of 1 to prevent premature output of the first frame.
            // Without this, streams with negative POC values (POC < IDR's
            // POC) in the first GOP would output the IDR before frames
            // that should display earlier. The dynamic detection at line
            // 846 will increase this further as needed.
            // Reference: FFmpeg h264_ps.c:538-550 estimates reorder depth
            // from level when VUI is absent.
            if self.reorder_depth < 1 {
                self.reorder_depth = 1;
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

        debug!(
            frame_num,
            prev_frame_num = self.prev_frame_num_h264,
            max_frame_num,
            max_refs = sps.max_num_ref_frames,
            gaps_allowed = sps.gaps_in_frame_num_allowed,
            "frame_num gap detected"
        );

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
            .map(|e| (e.pic.clone(), e.poc));

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
            let (pic, poc) = if let Some((prev_pic, prev_poc)) = &prev_pic {
                (prev_pic.clone(), *prev_poc + 2 * steps as i32)
            } else {
                (
                    PictureBuffer {
                        y: vec![128u8; y_stride * pic_h as usize],
                        u: vec![128u8; uv_stride * (pic_h / 2) as usize],
                        v: vec![128u8; uv_stride * (pic_h / 2) as usize],
                        y_stride,
                        uv_stride,
                        width: pic_w,
                        height: pic_h,
                        mb_width: mb_w,
                        mb_height: mb_h,
                    },
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
    #[cfg_attr(feature = "tracing-detail", tracing::instrument(skip_all, fields(nal_type = ?nalu.nal_type)))]
    fn process_nal(&mut self, nalu: &NalUnit, _pkt_pts: i64) -> Result<()> {
        match nalu.nal_type {
            NalUnitType::Sps => {
                if let Ok(sps) = parse_sps(&nalu.data) {
                    let id = sps.sps_id as usize;
                    debug!(
                        sps_id = id,
                        width = sps.width(),
                        height = sps.height(),
                        "SPS parsed"
                    );
                    self.apply_sps(&sps);
                    self.sps_list[id] = Some(sps);
                }
            }
            NalUnitType::Pps => {
                if let Ok(pps) = parse_pps(&nalu.data, &self.sps_list) {
                    let id = pps.pps_id as usize;
                    debug!(pps_id = id, sps_id = pps.sps_id, "PPS parsed");
                    self.pps_list[id] = Some(pps);
                }
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
                    .ok_or(Error::InvalidData)?;
                let sps = self.sps_list[pps.sps_id as usize]
                    .clone()
                    .ok_or(Error::InvalidData)?;

                let total_mbs = sps.mb_width * sps.mb_height;
                let is_idr = nalu.nal_type == NalUnitType::Idr;

                // Check if this starts a new frame
                if hdr.first_mb_in_slice == 0 {
                    // Flush any in-progress frame and store in DPB
                    self.finish_current_frame();

                    debug!(
                        slice_type = ?hdr.slice_type,
                        frame_num = hdr.frame_num,
                        qp = hdr.slice_qp,
                        is_idr,
                        use_weight = hdr.use_weight,
                        use_weight_chroma = hdr.use_weight_chroma,
                        num_luma_weights = hdr.luma_weight_l0.len(),
                        "slice start"
                    );

                    // Start new frame context
                    self.current_fdc = Some(FrameDecodeContext::new(&sps, &pps));
                    self.current_mbs_decoded = 0;
                    self.current_total_mbs = total_mbs;
                    self.current_pts = self.frame_num as i64;
                    self.current_is_idr = is_idr;
                    self.current_frame_num_h264 = hdr.frame_num;
                    self.current_nal_ref_idc = nalu.nal_ref_idc;

                    // Dynamic reorder depth increase (matching FFmpeg
                    // h264_slice.c:1328-1331). B-frames need at least depth 1.
                    if hdr.slice_type.is_b() && self.reorder_depth < 1 {
                        self.reorder_depth = 1;
                    }

                    // Frame num gap fill (H.264 Section 8.2.5.2).
                    // When frame_num is not consecutive, advance prev_frame_num
                    // through each gap value, creating dummy DPB entries and
                    // running sliding window marking for each.
                    // Reference: FFmpeg h264_slice.c:1506-1570.
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

                    // Build reference lists
                    if hdr.slice_type.is_p() {
                        let max_frame_num = 1u32 << sps.log2_max_frame_num;
                        self.ref_list_l0 =
                            refs::build_ref_list_p(&self.dpb, &hdr, hdr.frame_num, max_frame_num);
                        self.ref_list_l1.clear();
                        debug!(
                            poc = self.current_poc,
                            l0_len = self.ref_list_l0.len(),
                            l0_frame_nums = ?self.ref_list_l0.iter().map(|&i| self.dpb.get(i).map(|e| e.frame_num)).collect::<Vec<_>>(),
                            "P-frame ref list"
                        );
                    } else if hdr.slice_type.is_b() {
                        let max_frame_num_b = 1u32 << sps.log2_max_frame_num;
                        let (l0, l1) = refs::build_ref_list_b(
                            &self.dpb,
                            &hdr,
                            self.current_poc,
                            max_frame_num_b,
                        );
                        debug!(
                            poc = self.current_poc,
                            l0_len = l0.len(),
                            l1_len = l1.len(),
                            l0_pocs = ?l0.iter().map(|&i| self.dpb.get(i).map(|e| e.poc)).collect::<Vec<_>>(),
                            l1_pocs = ?l1.iter().map(|&i| self.dpb.get(i).map(|e| e.poc)).collect::<Vec<_>>(),
                            "B-frame ref lists"
                        );
                        self.ref_list_l0 = l0;
                        self.ref_list_l1 = l1;
                    } else {
                        self.ref_list_l0.clear();
                        self.ref_list_l1.clear();
                    }
                }

                // Decode this slice into the current frame context.
                // Take the fdc temporarily to avoid borrow conflicts.
                if let Some(mut fdc) = self.current_fdc.take() {
                    // Track slice boundaries for neighbor availability.
                    // First slice (first_mb==0) starts at 0; continuations increment.
                    if hdr.first_mb_in_slice > 0 {
                        fdc.current_slice += 1;
                    }

                    // Build list of reference PictureBuffer pointers
                    let ref_pic_list: Vec<&PictureBuffer> = self
                        .ref_list_l0
                        .iter()
                        .filter_map(|&dpb_idx| self.dpb.get(dpb_idx).map(|e| &e.pic))
                        .collect();
                    let ref_pic_list_l1: Vec<&PictureBuffer> = self
                        .ref_list_l1
                        .iter()
                        .filter_map(|&dpb_idx| self.dpb.get(dpb_idx).map(|e| &e.pic))
                        .collect();

                    // Set L0/L1 ref POCs for deblocking ref-identity comparison
                    // and temporal direct dist_scale_factor.
                    if hdr.slice_type.is_p() || hdr.slice_type.is_b() {
                        fdc.cur_l0_ref_poc = self
                            .ref_list_l0
                            .iter()
                            .filter_map(|&i| self.dpb.get(i).map(|e| e.poc))
                            .collect();
                        fdc.cur_l0_ref_dpb = self.ref_list_l0.clone();
                    }
                    if hdr.slice_type.is_b() {
                        fdc.cur_l1_ref_poc = self
                            .ref_list_l1
                            .iter()
                            .filter_map(|&i| self.dpb.get(i).map(|e| e.poc))
                            .collect();
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
                        }
                        fdc.cur_poc = self.current_poc;

                        // Pre-compute implicit weights for weighted_bipred_idc=2
                        if pps.weighted_bipred_idc == 2 {
                            fdc.compute_implicit_weights();
                        }
                    }

                    #[cfg(feature = "tracing-detail")]
                    if hdr.slice_type.is_p() || hdr.slice_type.is_b() {
                        for &dpb_idx in &self.ref_list_l0 {
                            if let Some(e) = self.dpb.get(dpb_idx) {
                                let s = e.pic.y_stride;
                                // Sample pixels at y=128, x=155-165
                                let row128: Vec<u8> = (155..166usize)
                                    .map(|x| {
                                        if 128 < e.pic.height as usize && x < e.pic.width as usize {
                                            e.pic.y[128 * s + x]
                                        } else {
                                            0
                                        }
                                    })
                                    .collect();
                                trace!(
                                    frame_num = self.frame_num,
                                    dpb_idx,
                                    ref_frame_num = e.frame_num,
                                    ref_poc = e.poc,
                                    status = ?e.status,
                                    row128 = ?row128,
                                    "ref_pic_list L0 entry"
                                );
                            }
                        }
                    }

                    match self.decode_slice_into(
                        &nalu.data,
                        &hdr,
                        &sps,
                        &pps,
                        &mut fdc,
                        &ref_pic_list,
                        &ref_pic_list_l1,
                    ) {
                        Ok(mbs) => {
                            self.current_mbs_decoded += mbs;
                            self.current_last_hdr = Some(hdr.clone());
                            self.current_fdc = Some(fdc);
                        }
                        Err(e) => {
                            warn!(
                                first_mb = hdr.first_mb_in_slice,
                                error = ?e,
                                "slice decode failed"
                            );
                            self.current_fdc = Some(fdc);
                        }
                    }
                }

                // Check if the frame is complete
                if self.current_mbs_decoded >= self.current_total_mbs {
                    self.finish_current_frame();
                }
            }
            // SEI, AUD, Filler, and other NAL types are silently ignored.
            NalUnitType::Sei
            | NalUnitType::Aud
            | NalUnitType::Filler
            | NalUnitType::EndSequence
            | NalUnitType::EndStream
            | NalUnitType::SliceA
            | NalUnitType::SliceB
            | NalUnitType::SliceC => {}
        }
        Ok(())
    }

    /// Flush a completed frame: apply deblocking, emit the output frame,
    /// and store the decoded picture in the DPB for reference.
    fn finish_current_frame(&mut self) {
        if let (Some(mut fdc), Some(last_hdr)) =
            (self.current_fdc.take(), self.current_last_hdr.as_ref())
        {
            // Set initial PTS to POC for reordering. Sequential PTS is
            // assigned later when flushing from delayed_pics.
            let frame = self.fdc_to_frame(&mut fdc, last_hdr, self.current_poc as i64);
            debug!(
                frame_num = self.frame_num,
                poc = self.current_poc,
                reorder_depth = self.reorder_depth,
                "frame complete"
            );

            // IDR resets the POC sequence. Clear last_pocs so the gap
            // detection doesn't falsely trigger from the old sequence.
            if self.current_is_idr {
                self.last_pocs = [i64::MIN; 16];
            }

            // Determine if this frame has an MMCO-5 (Reset) operation.
            // Needed early because MMCO-5 resets POC to 0, and we must
            // use the reset POC for last_pocs insertion (matching FFmpeg
            // where MMCO runs before h264_select_output_frame).
            let has_mmco5 = last_hdr.adaptive_ref_pic_marking
                && last_hdr
                    .mmco_ops
                    .iter()
                    .any(|op| matches!(op, crate::slice::MmcoOp::Reset));

            // Dynamically compute reorder depth using FFmpeg's last_pocs
            // algorithm (h264_slice.c:1309-1332). Insert current POC into
            // the sorted ascending last_pocs array and derive out_of_order
            // from the insertion position. The algorithm shifts values left
            // as it scans, then places cur_poc at the insertion point.
            //
            // When MMCO-5 is present, reset last_pocs and use POC 0
            // (matching FFmpeg where MMCO_RESET clears last_pocs in
            // h264_refs.c:724-725 and resets POC before this runs).
            if has_mmco5 {
                self.last_pocs = [i64::MIN; 16];
            }
            let cur_poc = if has_mmco5 {
                0i64
            } else {
                self.current_poc as i64
            };
            let mut insert_pos = 0usize;
            for i in 0..=16 {
                if i == 16 || cur_poc < self.last_pocs[i] {
                    if i > 0 {
                        self.last_pocs[i - 1] = cur_poc;
                    }
                    insert_pos = i;
                    break;
                } else if i > 0 {
                    self.last_pocs[i - 1] = self.last_pocs[i];
                }
            }

            let mut out_of_order = 16 - insert_pos;

            // Also increase for B-frames or POC gap > 2
            if last_hdr.slice_type.is_b()
                || (self.last_pocs[14] > i64::MIN && (self.last_pocs[15] - self.last_pocs[14]) > 2)
            {
                out_of_order = out_of_order.max(1);
            }

            // If all entries are smaller than current (out_of_order == 16),
            // it means POC went backwards (e.g., frame_num gap wrap or
            // MMCO reset). Reset last_pocs.
            if out_of_order == 16 {
                self.last_pocs = [i64::MIN; 16];
                self.last_pocs[0] = cur_poc;
            } else if out_of_order > self.reorder_depth && !self.has_bitstream_restriction {
                self.reorder_depth = out_of_order;
            }

            // The mmco_reset flag acts as a barrier in the delayed_pics
            // buffer, preventing min-POC search from mixing frames across
            // POC sequence boundaries.
            // Reference: FFmpeg h264_slice.c:1301, 1327, 1346-1353.
            let frame_mmco_reset = has_mmco5 || (out_of_order == 16 && !self.current_is_idr);

            // Add current frame to the delayed_pics buffer.
            // Use the original POC (not the MMCO-5 reset value) since
            // min-POC ordering needs the real POC for correct output.
            self.delayed_pics
                .push((self.current_poc, frame, frame_mmco_reset));

            // Output frames when buffer exceeds reorder_depth AND the
            // current frame is in order (out_of_order == 0).
            // Exception: always allow output when there's an mmco_reset
            // barrier in the buffer, to drain old POC sequences. This
            // prevents the out_of_order gate from blocking output when
            // POC cycling (due to MMCO-5) keeps out_of_order > 0.
            let has_barrier = self.delayed_pics.iter().any(|(_, _, reset)| *reset);
            let allow_output = out_of_order == 0 || has_barrier;
            while allow_output && self.delayed_pics.len() > self.reorder_depth {
                let out_idx = if self.delayed_pics[0].2 {
                    // mmco_reset at front: output it directly
                    0
                } else {
                    // Find barrier: first mmco_reset at index >= 1
                    let barrier = self
                        .delayed_pics
                        .iter()
                        .enumerate()
                        .skip(1)
                        .find(|(_, (_, _, reset))| *reset)
                        .map(|(i, _)| i)
                        .unwrap_or(self.delayed_pics.len());

                    // Find min-POC within [0, barrier)
                    self.delayed_pics[..barrier]
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (poc, _, _))| *poc)
                        .map(|(i, _)| i)
                        .unwrap()
                };
                let (_, mut f, _) = self.delayed_pics.remove(out_idx);
                f.pts = self.output_frame_counter;
                self.output_frame_counter += 1;
                self.output_queue.push_back(f);
            }

            // Non-reference pictures (nal_ref_idc == 0, typically B-frames)
            // don't need to be stored in the DPB.
            if self.current_nal_ref_idc == 0 {
                self.frame_num += 1;
                return;
            }

            // Store decoded picture in DPB for reference
            let mb_width = fdc.mb_width;
            let mb_height = fdc.mb_height;
            let total_blocks = (mb_width * mb_height * 16) as usize;

            // Extract MV info from the frame decode context
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

            #[cfg(feature = "tracing-detail")]
            {
                let px = 10 * 16;
                let py = 2 * 16;
                let s = fdc.pic.y_stride;
                let val = fdc.pic.y[py * s + px];
                let val2 = fdc.pic.y[py * s + px + 2];
                tracing::trace!(
                    frame_num = self.frame_num,
                    mb10_2_pixel0 = val,
                    mb10_2_pixel2 = val2,
                    y_ptr = ?fdc.pic.y.as_ptr(),
                    "DPB store check"
                );
            }

            let mb_intra: Vec<bool> = fdc.mb_info.iter().map(|info| info.is_intra).collect();

            // Store L0/L1 ref POCs for temporal direct mode mapping.
            let ref_poc_l0: Vec<i32> = self
                .ref_list_l0
                .iter()
                .filter_map(|&dpb_idx| self.dpb.get(dpb_idx).map(|e| e.poc))
                .collect();
            let ref_poc_l1: Vec<i32> = self
                .ref_list_l1
                .iter()
                .filter_map(|&dpb_idx| self.dpb.get(dpb_idx).map(|e| e.poc))
                .collect();

            let entry = DpbEntry {
                pic: fdc.pic,
                poc: self.current_poc,
                frame_num: self.current_frame_num_h264,
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

            // Try to store in DPB; if full, remove oldest first
            if self.dpb.is_full() {
                self.dpb.remove_oldest_short_term();
                // If still full, remove any unused entry
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

            let mut mmco_did_reset = false;
            if let Some(dpb_idx) = self.dpb.store(entry) {
                #[cfg(feature = "tracing-detail")]
                {
                    let e = self.dpb.get(dpb_idx).unwrap();
                    tracing::trace!(
                        internal_frame = self.frame_num,
                        dpb_idx,
                        h264_frame_num = e.frame_num,
                        y_ptr = ?e.pic.y.as_ptr(),
                        pixel_160_32 = e.pic.y[32 * e.pic.y_stride + 160],
                        "DPB stored entry"
                    );
                }
                // Apply reference picture marking
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
                    last_hdr,
                    self.current_is_idr,
                    self.current_frame_num_h264,
                    max_frame_num,
                    sps_max_refs,
                    Some(dpb_idx),
                );
            }

            // Log DPB summary for debugging tools (dpb_compare.py)
            {
                let st_fns: Vec<u32> = self
                    .dpb
                    .entries
                    .iter()
                    .filter_map(|e| {
                        e.as_ref()
                            .and_then(|e| (e.status == RefStatus::ShortTerm).then_some(e.frame_num))
                    })
                    .collect();
                let lt_fns: Vec<u32> = self
                    .dpb
                    .entries
                    .iter()
                    .filter_map(|e| {
                        e.as_ref().and_then(|e| {
                            (e.status == RefStatus::LongTerm).then_some(e.long_term_frame_idx)
                        })
                    })
                    .collect();
                debug!(
                    h264_fn = self.current_frame_num_h264,
                    poc = self.current_poc,
                    st_count = st_fns.len(),
                    st_frame_nums = ?st_fns,
                    lt_count = lt_fns.len(),
                    lt_indices = ?lt_fns,
                    "DPB state"
                );
            }

            // Update POC type 0 state for reference pictures only.
            // Non-reference pictures (nal_ref_idc == 0, e.g. B-frames)
            // do not update the POC state.
            if self.current_nal_ref_idc > 0 {
                let max_poc_lsb = self
                    .sps_list
                    .iter()
                    .find_map(|s| s.as_ref().map(|sps| 1u32 << sps.log2_max_poc_lsb))
                    .unwrap_or(16);
                let poc_lsb = last_hdr.pic_order_cnt_lsb;
                // Recompute poc_msb using the same logic as compute_poc_type0
                if self.current_is_idr {
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

            // Update POC type 1/2 state: frame_num_offset persists across
            // all frames (not just references).
            self.prev_frame_num_offset = self.frame_num_offset;
            self.prev_frame_num_h264 = self.current_frame_num_h264;

            // MMCO-5 (Reset) resets frame_num to 0 and frame_num_offset.
            // Must happen AFTER the normal prev_frame_num/POC update above,
            // so the reset overrides the normal assignments.
            //
            // Spec 8.2.1.1: after MMCO-5, prevPicOrderCntMsb = 0 and
            // prevPicOrderCntLsb = TopFieldOrderCnt (which is 0 for frames
            // after MMCO-5 resets PicOrderCnt).
            // Spec 8.2.1.2/8.2.1.3: after MMCO-5, prevFrameNumOffset = 0.
            // Reference: FFmpeg h264_slice.c — MMCO-5 resets poc_msb/poc_lsb
            // inside ff_h264_execute_ref_pic_marking BEFORE the prev_poc
            // assignments, so prev_poc_msb and prev_poc_lsb end up as 0.
            if mmco_did_reset {
                debug!(
                    h264_fn = self.current_frame_num_h264,
                    "MMCO-5 reset: prev_frame_num/poc/offset → 0"
                );
                self.prev_frame_num_h264 = 0;
                self.prev_poc_msb = 0;
                self.prev_poc_lsb = 0;
                self.prev_frame_num_offset = 0;
            }

            self.frame_num += 1;
        }
    }

    /// Decode a slice into a FrameDecodeContext.
    ///
    /// Dispatches to CAVLC or CABAC slice decode based on PPS entropy_coding_mode_flag.
    /// Returns the number of MBs decoded in this slice.
    #[allow(clippy::too_many_arguments)] // H.264 slice decode needs all parameters
    #[cfg_attr(feature = "tracing-detail", tracing::instrument(skip_all, fields(poc = self.current_poc, first_mb = hdr.first_mb_in_slice, slice_type = ?hdr.slice_type)))]
    fn decode_slice_into(
        &self,
        rbsp: &[u8],
        hdr: &SliceHeader,
        sps: &Sps,
        pps: &Pps,
        fdc: &mut FrameDecodeContext,
        ref_pics: &[&PictureBuffer],
        ref_pics_l1: &[&PictureBuffer],
    ) -> Result<u32> {
        fdc.qp = hdr.slice_qp as u8;

        if pps.entropy_coding_mode_flag {
            self.decode_slice_cabac(rbsp, hdr, sps, pps, fdc, ref_pics, ref_pics_l1)
        } else {
            self.decode_slice_cavlc(rbsp, hdr, sps, pps, fdc, ref_pics, ref_pics_l1)
        }
    }

    /// Decode a CAVLC-coded slice.
    #[allow(clippy::too_many_arguments)]
    fn decode_slice_cavlc(
        &self,
        rbsp: &[u8],
        hdr: &SliceHeader,
        sps: &Sps,
        pps: &Pps,
        fdc: &mut FrameDecodeContext,
        ref_pics: &[&PictureBuffer],
        ref_pics_l1: &[&PictureBuffer],
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
        tracing::debug!(header_bits = hdr.header_bits, "slice header size");
        br.skip_bits(hdr.header_bits);

        // Decode macroblocks for this slice
        let first_mb = hdr.first_mb_in_slice;
        let mut mbs_decoded = 0u32;
        let is_inter_slice = hdr.slice_type.is_p() || hdr.slice_type.is_b();
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
                #[cfg(feature = "tracing-detail")]
                trace!(mb_addr, mb_skip_run, bits = br.consumed(), "mb_skip_run");

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
        }

        // Validate bitstream position: after decoding all MBs in this slice,
        // the reader should be near the end of the RBSP (within ~16 bits for
        // the trailing RBSP stop bit and alignment padding). Large discrepancies
        // indicate a CAVLC desync.
        let bits_remaining = rbsp_bits.saturating_sub(br.consumed());
        if bits_remaining > 16 {
            warn!(
                first_mb,
                bits_remaining,
                consumed = br.consumed(),
                rbsp_bits,
                mbs_decoded,
                "CAVLC desync: slice ended with excess bits remaining"
            );
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
        &self,
        rbsp: &[u8],
        hdr: &SliceHeader,
        sps: &Sps,
        pps: &Pps,
        fdc: &mut FrameDecodeContext,
        ref_pics: &[&PictureBuffer],
        ref_pics_l1: &[&PictureBuffer],
    ) -> Result<u32> {
        use crate::cabac::{CabacNeighborCtx, CabacReader, decode_cabac_mb_skip, decode_mb_cabac};
        use crate::cabac_tables::init_cabac_states;

        let mb_width = sps.mb_width;
        let mb_height = sps.mb_height;
        let total_mbs = mb_width * mb_height;

        // Byte-align: CABAC data starts at byte boundary after slice header
        let data_start = hdr.header_bits.div_ceil(8);
        if data_start >= rbsp.len() {
            return Err(Error::InvalidData);
        }

        // Add padding for safe CABAC reads
        let mut padded = Vec::with_capacity(rbsp.len() - data_start + 64);
        padded.extend_from_slice(&rbsp[data_start..]);
        padded.resize(padded.len() + 64, 0);

        // Initialize CABAC reader
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
        let mut mb_addr = first_mb;

        while mb_addr < total_mbs {
            let mb_x = mb_addr % mb_width;
            let mb_y = mb_addr / mb_width;
            let mb_idx = mb_addr as usize;

            // Update CAVLC neighbor context at the start of each row
            if mb_x == 0 {
                fdc.neighbor_ctx.new_row();
                fdc.neighbor_ctx.top_available =
                    mb_y > 0 && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
            } else if mb_addr == first_mb {
                fdc.neighbor_ctx.left_available = false;
                fdc.neighbor_ctx.top_available =
                    mb_y > 0 && fdc.slice_table[(mb_addr - mb_width) as usize] == fdc.current_slice;
            }

            // For inter slices: decode skip flag
            if is_inter_slice {
                #[cfg(feature = "cabac-trace")]
                eprintln!(
                    "CABAC_SKIP_DECODE mb_x={} mb_y={} is_b={}",
                    mb_x,
                    mb_y,
                    hdr.slice_type.is_b()
                );
                let skip = decode_cabac_mb_skip(
                    &mut reader,
                    &mut cabac_state,
                    &cabac_nb,
                    &fdc.slice_table,
                    fdc.current_slice,
                    mb_x,
                    mb_y,
                    mb_width,
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
                        mb_idx,
                        true,  // is_skip
                        false, // is_intra16x16_or_pcm
                        false, // is_intra
                        false, // is_direct
                        0,     // cbp
                        0,     // chroma_pred
                        &[0; 24],
                        &[[0; 2]; 16],
                        &[[0; 2]; 16],
                    );

                    mb_addr += 1;
                    mbs_decoded += 1;

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
            )?;

            // Apply entropy-agnostic processing (dequant, IDCT, pred, MC, neighbor update)
            mb::apply_macroblock(fdc, &mut mb, hdr, pps, mb_x, mb_y, ref_pics, ref_pics_l1)?;

            // Build MVD abs values for CABAC neighbor context
            let mut mvd_abs_l0 = [[0u8; 2]; 16];
            let mut mvd_abs_l1 = [[0u8; 2]; 16];
            for i in 0..16 {
                mvd_abs_l0[i] = [
                    (mb.mvd_l0[i][0].unsigned_abs().min(70)) as u8,
                    (mb.mvd_l0[i][1].unsigned_abs().min(70)) as u8,
                ];
                mvd_abs_l1[i] = [
                    (mb.mvd_l1[i][0].unsigned_abs().min(70)) as u8,
                    (mb.mvd_l1[i][1].unsigned_abs().min(70)) as u8,
                ];
            }

            // Update CABAC neighbor context
            cabac_nb.update_after_mb(
                mb_idx,
                false, // is_skip
                mb.is_intra16x16 || mb.is_pcm,
                mb.is_intra,
                mb.is_direct,
                mb.cbp,
                mb.chroma_pred_mode,
                &mb.non_zero_count,
                &mvd_abs_l0,
                &mvd_abs_l1,
            );

            mb_addr += 1;
            mbs_decoded += 1;

            // Check terminate
            if reader.get_cabac_terminate() {
                break;
            }
        }

        debug!(first_mb, mbs_decoded, "CABAC slice decoded");

        Ok(mbs_decoded)
    }

    /// Convert a completed FrameDecodeContext to a Frame.
    fn fdc_to_frame(&self, fdc: &mut FrameDecodeContext, hdr: &SliceHeader, pts: i64) -> Frame {
        if std::env::var("WEDEO_NO_DEBLOCK").is_err() {
            deblock::deblock_frame(
                &mut fdc.pic,
                &fdc.mb_info,
                hdr.disable_deblocking_filter_idc,
                hdr.slice_alpha_c0_offset,
                hdr.slice_beta_offset,
            );
        }

        // Convert PictureBuffer to Frame, applying SPS crop offsets.
        let width = self.width as usize;
        let height = self.height as usize;
        let chroma_width = width / 2;
        let chroma_height = height / 2;
        let crop_x = self.crop_left as usize;
        let crop_y = self.crop_top as usize;
        let chroma_crop_x = crop_x / 2;
        let chroma_crop_y = crop_y / 2;

        let mut y_data = Vec::with_capacity(width * height);
        for row in 0..height {
            let src_start = (crop_y + row) * fdc.pic.y_stride + crop_x;
            y_data.extend_from_slice(&fdc.pic.y[src_start..src_start + width]);
        }
        let y_buf = Buffer::from_slice(&y_data);

        let mut u_data = Vec::with_capacity(chroma_width * chroma_height);
        for row in 0..chroma_height {
            let src_start = (chroma_crop_y + row) * fdc.pic.uv_stride + chroma_crop_x;
            u_data.extend_from_slice(&fdc.pic.u[src_start..src_start + chroma_width]);
        }
        let u_buf = Buffer::from_slice(&u_data);

        let mut v_data = Vec::with_capacity(chroma_width * chroma_height);
        for row in 0..chroma_height {
            let src_start = (chroma_crop_y + row) * fdc.pic.uv_stride + chroma_crop_x;
            v_data.extend_from_slice(&fdc.pic.v[src_start..src_start + chroma_width]);
        }
        let v_buf = Buffer::from_slice(&v_data);

        let y_plane = FramePlane {
            buffer: y_buf,
            offset: 0,
            linesize: width,
        };
        let u_plane = FramePlane {
            buffer: u_buf,
            offset: 0,
            linesize: chroma_width,
        };
        let v_plane = FramePlane {
            buffer: v_buf,
            offset: 0,
            linesize: chroma_width,
        };

        let mut frame = Frame::new_video(self.width, self.height, PixelFormat::Yuv420p);
        frame.pts = pts;

        let pict_type = match hdr.slice_type {
            SliceType::I | SliceType::SI => PictureType::I,
            SliceType::P | SliceType::SP => PictureType::P,
            SliceType::B => PictureType::B,
        };

        if let FrameData::Video(ref mut video) = frame.data {
            video.planes = vec![y_plane, u_plane, v_plane];
            video.picture_type = pict_type;
        }

        if hdr.slice_type.is_intra() {
            frame.flags |= FrameFlags::KEY;
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
        return Err(Error::InvalidData);
    }
    // Skip the NAL header byte, then remove emulation prevention bytes.
    // The NalUnit parser already does EPB removal, but here we have raw bytes
    // from avcC, so we need to do it manually.
    let rbsp = remove_epb(&nal_bytes[1..]);
    parse_sps(&rbsp)
}

/// Parse a PPS from a raw NAL unit (header byte + payload).
fn parse_pps_from_nal(nal_bytes: &[u8], sps_list: &[Option<Sps>; 32]) -> Result<Pps> {
    if nal_bytes.is_empty() {
        return Err(Error::InvalidData);
    }
    let rbsp = remove_epb(&nal_bytes[1..]);
    parse_pps(&rbsp, sps_list)
}

/// Remove emulation prevention bytes (0x00 0x00 0x03 -> 0x00 0x00).
fn remove_epb(data: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x03 {
            rbsp.push(0x00);
            rbsp.push(0x00);
            i += 3;
        } else {
            rbsp.push(data[i]);
            i += 1;
        }
    }
    rbsp
}

// ---------------------------------------------------------------------------
// Decoder trait implementation
// ---------------------------------------------------------------------------

impl Decoder for H264Decoder {
    #[cfg_attr(feature = "tracing-detail", tracing::instrument(skip_all, fields(has_packet = packet.is_some())))]
    fn send_packet(&mut self, packet: Option<&Packet>) -> Result<()> {
        match packet {
            Some(pkt) => {
                let data = pkt.data.data();
                let pts = pkt.pts;

                // Split into NAL units using the appropriate method
                let nalus = if self.nalff_length_size > 0 {
                    split_nalff(data, self.nalff_length_size)
                } else {
                    split_annex_b(data)
                };

                for nalu in &nalus {
                    if let Err(e) = self.process_nal(nalu, pts) {
                        warn!(
                            error = ?e,
                            nal_type = ?nalu.nal_type,
                            "NAL decode error"
                        );
                    }
                }

                Ok(())
            }
            None => {
                // Drain mode: no more packets will be sent.
                // Flush any in-progress frame.
                self.finish_current_frame();

                // Flush all remaining delayed_pics using barrier-aware
                // min-POC search, matching FFmpeg's send_next_delayed_frame()
                // (h264dec.c:987-1001). Unlike a simple sort, this respects
                // mmco_reset barriers to avoid interleaving frames from
                // different POC sequences.
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
                    let (_, mut f, _) = self.delayed_pics.remove(out_idx);
                    f.pts = self.output_frame_counter;
                    self.output_frame_counter += 1;
                    self.output_queue.push_back(f);
                }

                self.draining = true;
                Ok(())
            }
        }
    }

    #[cfg_attr(feature = "tracing-detail", tracing::instrument(skip_all))]
    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(frame) = self.output_queue.pop_front() {
            Ok(frame)
        } else if self.draining {
            Err(Error::Eof)
        } else {
            Err(Error::Again)
        }
    }

    fn flush(&mut self) {
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
        self.reorder_depth = 0;
        self.has_bitstream_restriction = false;
        self.last_pocs = [i64::MIN; 16];
        self.output_frame_counter = 0;
        self.delayed_pics.clear();
        // SPS/PPS are retained across flush (matching FFmpeg behavior).
    }

    fn descriptor(&self) -> &CodecDescriptor {
        &self.codec_descriptor
    }
}

// ---------------------------------------------------------------------------
// Factory registration
// ---------------------------------------------------------------------------

struct H264DecoderFactory;

impl DecoderFactory for H264DecoderFactory {
    fn descriptor(&self) -> &DecoderDescriptor {
        static DESC: DecoderDescriptor = DecoderDescriptor {
            codec_id: CodecId::H264,
            name: "h264",
            long_name: "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
            media_type: MediaType::Video,
            capabilities: CodecCapabilities::DR1,
            properties: CodecProperties::LOSSY.union(CodecProperties::REORDER),
            priority: 100,
        };
        &DESC
    }

    fn create(&self, params: CodecParameters) -> Result<Box<dyn Decoder>> {
        Ok(Box::new(H264Decoder::new(params)?))
    }
}

inventory::submit!(&H264DecoderFactory as &dyn DecoderFactory);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::packet::Packet;

    fn make_params() -> CodecParameters {
        CodecParameters::new(CodecId::H264, MediaType::Video)
    }

    #[test]
    fn create_decoder() {
        let params = make_params();
        let decoder = H264Decoder::new(params);
        assert!(decoder.is_ok());
    }

    #[test]
    fn decoder_drain_returns_eof() {
        let params = make_params();
        let mut decoder = H264Decoder::new(params).unwrap();

        // No packets sent, receive should return Again
        assert_eq!(decoder.receive_frame().unwrap_err(), Error::Again);

        // Send drain signal
        decoder.send_packet(None).unwrap();

        // Should return Eof
        assert_eq!(decoder.receive_frame().unwrap_err(), Error::Eof);
    }

    #[test]
    fn decoder_flush_clears_state() {
        let params = make_params();
        let mut decoder = H264Decoder::new(params).unwrap();

        decoder.send_packet(None).unwrap();
        assert!(decoder.draining);

        decoder.flush();
        assert!(!decoder.draining);
        assert!(decoder.output_queue.is_empty());
    }

    #[test]
    fn decoder_processes_sps_pps() {
        let params = make_params();
        let mut decoder = H264Decoder::new(params).unwrap();

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
        assert_eq!(decoder.receive_frame().unwrap_err(), Error::Again);

        // Verify SPS and PPS were stored
        assert!(decoder.sps_list[0].is_some());
        assert!(decoder.pps_list[0].is_some());

        // Verify dimensions updated
        assert_eq!(decoder.width, 320);
        assert_eq!(decoder.height, 240);
    }

    #[test]
    fn decoder_handles_invalid_slice_gracefully() {
        let params = make_params();
        let mut decoder = H264Decoder::new(params).unwrap();

        // Build SPS + PPS + IDR with only header (no MB data).
        // The decoder should log a warning and skip the NAL, not panic.
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
        // send_packet should succeed (errors in individual NALs are logged, not fatal)
        decoder.send_packet(Some(&pkt)).unwrap();

        // The IDR slice has no MB data, so decode will fail and be skipped.
        // No frame should be produced.
        assert_eq!(decoder.receive_frame().unwrap_err(), Error::Again);
    }

    #[test]
    fn decoder_descriptor() {
        let params = make_params();
        let decoder = H264Decoder::new(params).unwrap();
        let desc = decoder.descriptor();
        assert_eq!(desc.id, CodecId::H264);
        assert_eq!(desc.media_type, MediaType::Video);
        assert_eq!(desc.name, "h264");
    }

    #[test]
    fn factory_descriptor() {
        let factory = H264DecoderFactory;
        let desc = factory.descriptor();
        assert_eq!(desc.codec_id, CodecId::H264);
        assert_eq!(desc.name, "h264");
        assert_eq!(desc.priority, 100);
        assert_eq!(desc.media_type, MediaType::Video);
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
        let num_bytes = (bits.len() + 7) / 8;
        let mut bytes = vec![0u8; num_bytes];
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        bytes
    }
}
