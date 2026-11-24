// VP9 decoder — main entry point.
//
// Implements the `Decoder` trait for VP9 video by orchestrating:
//   1. Frame header parsing   (`header::decode_frame_header`)
//   2. Entropy / block decode (`block::decode_sb`)
//   3. Intra/inter reconstruction
//   4. Loop filtering         (`loopfilter::loop_filter_frame`)
//
// LGPL-2.1-or-later — same licence as FFmpeg.

use std::collections::VecDeque;
use std::sync::Arc;

use wedeo_core::{
    BorderPadding, BorderedPlane, CodecContext, CodecId, ColorDesc, ColorPrimaries, ColorRange,
    Decoder, DecoderDescriptor, DecoderFactory, Error, ErrorPolicy, FrameFlags, HealthEvent,
    HealthIssue, MatrixCoefficients, MediaType, MediaUnit, Packet, PictureType, PixelLayout,
    Rational, RecvStatus, Registry, Result, SendStatus, StreamHealth, StreamParams,
    TransferCharacteristic, VideoFrame,
};

use crate::block::{BlockInfo, TileDecodeContext, decode_sb};
use crate::bool_decoder::BoolDecoder;
use crate::context::{AboveContext, LeftContext};
use crate::data::DEFAULT_COEF_PROBS;
use crate::header::{self, ColorSpace, FrameType, PersistentHeaderState, decode_frame_header};
use crate::loopfilter::loop_filter_frame;
use crate::mc::McScratch;
use crate::prob::{CoefProbArray, CountContext, adapt_probs};
use crate::recon::{FrameBuffer, reconstruct_inter_block, reconstruct_intra_block};
use crate::refs::{MvRefPair, RefFrame, RefStore};
use crate::types::{BlockLevel, CompPredMode, ProbContext};

/// Register the pure-Rust VP9 decoder factory into `reg`.
pub fn register_vp9(reg: &mut Registry) {
    reg.register_decoder(Arc::new(Vp9DecoderFactory));
}

// ---------------------------------------------------------------------------
// Decoder state
// ---------------------------------------------------------------------------

/// Pure-Rust FFmpeg `n9.1` VP9 video decoder.
pub struct Vp9Decoder {
    ctx: CodecContext,
    time_base: Rational,
    sar: Rational,
    fallback_color: ColorDesc,
    output_queue: VecDeque<MediaUnit>,
    draining: bool,
    health: StreamHealth,
    /// 4 probability context slots indexed by `frame_ctx_id`.
    prob_ctx: [ProbContext; 4],
    /// 4 coefficient probability context slots indexed by `frame_ctx_id`.
    coef_ctx: [CoefProbArray; 4],
    /// Whether the previous frame was a keyframe (for adapt_probs update factor).
    last_keyframe: bool,
    /// Reference frame storage for inter prediction.
    ref_store: RefStore,
    /// Previous frame's show_frame flag (FFmpeg: `last_invisible`).
    last_show_frame: bool,
    /// Previous frame's width (for use_last_frame_mvs dimension check).
    last_width: u32,
    /// Previous frame's height.
    last_height: u32,
    /// Active stream bit depth (inherited by inter frames).
    active_bit_depth: u8,
    /// Active stream horizontal subsampling (inherited by inter frames).
    active_ss_h: bool,
    /// Active stream vertical subsampling (inherited by inter frames).
    active_ss_v: bool,
    /// Active stream VP9 color space (inherited by inter frames).
    active_color_space: ColorSpace,
    /// Active stream VP9 color range (inherited by inter frames).
    active_color_range: header::ColorRange,
    /// Persistent cross-frame header state (ref dims, loopfilter deltas, segmentation).
    hdr_state: PersistentHeaderState,
    /// Persistent per-8×8 segment ID map from the most recent map-updating frame.
    prev_seg_map: Vec<u8>,
    prev_seg_cols8: usize,
    prev_seg_rows8: usize,
    /// Reusable motion-compensation scratch buffers.
    mc_scratch: McScratch,
}

impl Vp9Decoder {
    /// Open a new VP9 decoder for `track` with runtime `ctx`.
    pub fn open(track: &TrackDescriptorWrapper<'_>, ctx: CodecContext) -> Result<Self> {
        let (sar, fallback_color) = match &track.0.params {
            StreamParams::Video(v) => (v.sar, v.color),
            _ => (Rational::new(1, 1), ColorDesc::UNSPECIFIED),
        };
        Ok(Self {
            ctx,
            time_base: track.0.time_base,
            sar,
            fallback_color,
            output_queue: VecDeque::new(),
            draining: false,
            health: StreamHealth::new(),
            prob_ctx: Default::default(),
            coef_ctx: [DEFAULT_COEF_PROBS; 4],
            last_keyframe: false,
            ref_store: RefStore::new(),
            last_show_frame: false,
            last_width: 0,
            last_height: 0,
            active_bit_depth: 8,
            active_ss_h: true,
            active_ss_v: true,
            active_color_space: ColorSpace::Unknown,
            active_color_range: header::ColorRange::Limited,
            hdr_state: PersistentHeaderState::default(),
            prev_seg_map: Vec::new(),
            prev_seg_cols8: 0,
            prev_seg_rows8: 0,
            mc_scratch: McScratch::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_video_frame(
        &self,
        fb: &FrameBuffer,
        width: u32,
        height: u32,
        layout: PixelLayout,
        color: ColorDesc,
        picture_type: PictureType,
        is_key: bool,
        pkt: &Packet,
    ) -> Result<VideoFrame> {
        let src_planes = [
            (&fb.y, fb.y_stride),
            (&fb.u, fb.uv_stride),
            (&fb.v, fb.uv_stride),
        ];
        let mut planes = Vec::with_capacity(3);
        for (idx, (src, src_stride)) in src_planes.iter().enumerate() {
            let (w_bytes, h_rows) = layout.plane_dims(width, height, idx).ok_or_else(|| {
                Error::internal("vp9", "invalid plane index for VP9 pixel layout")
            })?;
            let mut plane =
                BorderedPlane::new_reused(&self.ctx.pool, w_bytes, h_rows, BorderPadding::NONE);
            for y in 0..h_rows {
                let dst_row = plane
                    .row_mut(y)
                    .ok_or_else(|| Error::internal("vp9", "pooled plane row out of bounds"))?;
                let src_off = y * *src_stride;
                dst_row.copy_from_slice(&src[src_off..src_off + w_bytes]);
            }
            planes.push(plane);
        }

        let mut frame = VideoFrame::from_planes(width, height, layout, planes);
        frame.color = color;
        frame.sar = self.sar;
        frame.picture_type = picture_type;
        if is_key {
            frame.flags |= FrameFlags::KEY;
        }
        frame.pts = if pkt.pts.is_known() { pkt.pts } else { pkt.dts };
        frame.pkt_dts = pkt.dts;
        frame.duration = pkt.duration;
        frame.time_base = if pkt.time_base.is_valid() {
            pkt.time_base
        } else {
            self.time_base
        };
        frame.side_data.clone_from(&pkt.side_data);
        Ok(frame)
    }

    /// Decode one VP9 frame from `data` and push the result to `output_queue`.
    fn decode_frame(&mut self, data: &[u8], pkt: &Packet) -> Result<()> {
        // Refresh reference slot dimensions snapshot for header tile-bit parsing.
        for (dst, slot) in self
            .hdr_state
            .ref_dims
            .iter_mut()
            .zip(self.ref_store.slots.iter())
        {
            *dst = slot.as_ref().map(|rf| (rf.width, rf.height));
        }

        // --- 1. Parse frame header ---
        let (mut header, _hdr_bytes) =
            match decode_frame_header(data, &self.prob_ctx, &self.coef_ctx, &self.hdr_state) {
                Ok(parsed) => parsed,
                Err(e) => {
                    self.health.record(HealthEvent {
                        track: pkt.track_id,
                        pts: pkt.pts,
                        byte_offset: pkt.pos,
                        kind: HealthIssue::CorruptSyntax,
                        detail: "invalid VP9 frame header",
                    });
                    return Err(e);
                }
            };

        // show_existing_frame: output a previously decoded reference frame.
        if header.show_existing_frame {
            let ref_idx = header.show_existing_frame_ref as usize;
            let Some(rf) = self.ref_store.slots[ref_idx].as_ref() else {
                self.health.record(HealthEvent {
                    track: pkt.track_id,
                    pts: pkt.pts,
                    byte_offset: pkt.pos,
                    kind: HealthIssue::MissingRef,
                    detail: "show_existing_frame references empty slot",
                });
                return Err(Error::invalid_data(
                    "vp9",
                    "show_existing_frame references empty reference slot",
                ));
            };
            let frame = self.build_video_frame(
                &rf.fb,
                rf.width,
                rf.height,
                rf.layout,
                rf.color,
                PictureType::P,
                false,
                pkt,
            )?;
            self.output_queue.push_back(MediaUnit::Video(frame));
            return Ok(());
        }

        // Propagate active stream color/subsampling/bit-depth to inter frames
        // (which do not carry a color_config in the VP9 bitstream).
        let is_kf = header.frame_type == FrameType::KeyFrame || header.intra_only;
        if is_kf {
            self.active_bit_depth = header.bit_depth;
            self.active_ss_h = header.subsampling_x;
            self.active_ss_v = header.subsampling_y;
            self.active_color_space = header.color_space;
            self.active_color_range = header.color_range;
        } else {
            header.bit_depth = self.active_bit_depth;
            header.subsampling_x = self.active_ss_h;
            header.subsampling_y = self.active_ss_v;
            header.color_space = self.active_color_space;
            header.color_range = self.active_color_range;
        }

        // Reject unsupported >8-bit profiles (Profiles 2/3) or 4:4:0 explicitly.
        let layout = derive_pixel_layout(
            header.bit_depth,
            header.color_space,
            header.subsampling_x,
            header.subsampling_y,
            header.profile as u8,
        )?;
        let color = derive_color_desc(
            header.color_space,
            header.color_range,
            header.bit_depth,
            self.fallback_color,
        );

        // Reset probability context slots per VP9 spec (keyframe, error_resilient, or intra_only reset_context).
        if header.frame_type == FrameType::KeyFrame
            || header.error_resilient
            || (header.intra_only && header.reset_context == 3)
        {
            for slot in &mut self.prob_ctx {
                *slot = crate::header::default_prob_context();
            }
            self.coef_ctx = [DEFAULT_COEF_PROBS; 4];
        } else if header.intra_only && header.reset_context == 2 {
            let fctx = header.frame_ctx_id as usize;
            self.prob_ctx[fctx] = crate::header::default_prob_context();
            self.coef_ctx[fctx] = DEFAULT_COEF_PROBS;
        }
        if header.frame_type == FrameType::KeyFrame {
            self.ref_store.clear();
        }

        // Resolve dimensions from reference frame if needed.
        if header.size_from_ref >= 0 {
            let ri = header.ref_idx[header.size_from_ref as usize] as usize;
            if let Some(rf) = &self.ref_store.slots[ri] {
                header.width = rf.width;
                header.height = rf.height;
            } else {
                self.health.record(HealthEvent {
                    track: pkt.track_id,
                    pts: pkt.pts,
                    byte_offset: pkt.pos,
                    kind: HealthIssue::MissingRef,
                    detail: "inter frame references empty slot for dimensions",
                });
                if self.ctx.error_policy == ErrorPolicy::Strict {
                    return Err(Error::corrupt(
                        "vp9",
                        "inter frame references empty slot for dimensions",
                    ));
                }
                return Ok(());
            }
        }

        // FFmpeg: use_last_frame_mvs = !errorres && !last_invisible
        //         use_last_frame_mvs &= (prev_frame exists && same dimensions)
        header.use_last_frame_mvs = !header.error_resilient
            && self.last_show_frame
            && self.last_width == header.width
            && self.last_height == header.height;

        // --- 2. Set up frame buffer and contexts ---
        let width = header.width;
        let height = header.height;
        let mut fb = FrameBuffer::new(width, height, header.subsampling_x, header.subsampling_y);

        // Match FFmpeg's 8x8-aligned block grid (`2 * cols_8x8`, `2 * rows_8x8`).
        let cols_4x4 = (width as usize).div_ceil(8) * 2;
        let rows_4x4 = (height as usize).div_ceil(8) * 2;
        let sb_cols = cols_4x4.div_ceil(16);
        let sb_rows = rows_4x4.div_ceil(16);

        let mut above = AboveContext::new(cols_4x4, sb_cols);
        above.reset(is_kf);

        let mut cur_mv_grid = vec![MvRefPair::default(); rows_4x4 * cols_4x4];
        let cols8 = cols_4x4 >> 1;
        let rows8 = rows_4x4 >> 1;
        if is_kf
            || header.error_resilient
            || cols8 != self.prev_seg_cols8
            || rows8 != self.prev_seg_rows8
        {
            self.prev_seg_map.clear();
            self.prev_seg_cols8 = 0;
            self.prev_seg_rows8 = 0;
        }
        let mut cur_seg_map = if header.segmentation.enabled && header.segmentation.update_map {
            vec![0u8; rows8 * cols8]
        } else {
            Vec::new()
        };
        let prev_seg_opt = if self.prev_seg_map.len() == rows8 * cols8 {
            Some(self.prev_seg_map.as_slice())
        } else {
            None
        };

        // --- 3. Locate and decode tiles deterministically ---
        let tile_data = &data[header.tile_data_offset..];
        let tile_cols = 1usize << (header.tile_cols_log2 as usize);
        let tile_rows = 1usize << (header.tile_rows_log2 as usize);

        let total_tiles = tile_cols * tile_rows;
        let mut tile_offsets: Vec<(usize, usize)> = Vec::with_capacity(total_tiles);
        {
            let mut read_pos = 0usize;
            for t in 0..total_tiles {
                let tile_size = if t == total_tiles - 1 {
                    tile_data.len().saturating_sub(read_pos)
                } else {
                    if read_pos + 4 > tile_data.len() {
                        return Err(Error::invalid_data("vp9", "truncated VP9 tile size header"));
                    }
                    let sz = u32::from_be_bytes([
                        tile_data[read_pos],
                        tile_data[read_pos + 1],
                        tile_data[read_pos + 2],
                        tile_data[read_pos + 3],
                    ]) as usize;
                    read_pos += 4;
                    sz
                };
                if read_pos + tile_size > tile_data.len() {
                    return Err(Error::invalid_data("vp9", "truncated VP9 tile payload"));
                }
                tile_offsets.push((read_pos, tile_size));
                read_pos += tile_size;
            }
        }

        let mut all_blocks = Vec::new();
        let mut merged_counts = CountContext::default();

        for tile_row in 0..tile_rows {
            let tr_start = tile_sb_offset(tile_row, header.tile_rows_log2, sb_rows);
            let tr_end = tile_sb_offset(tile_row + 1, header.tile_rows_log2, sb_rows).min(sb_rows);

            for tile_col in 0..tile_cols {
                let (blocks, counts) = decode_tile_column(
                    tile_data,
                    &tile_offsets,
                    &header,
                    &mut above,
                    tile_row,
                    tile_col,
                    tile_cols,
                    tr_start,
                    tr_end,
                    sb_cols,
                    &mut cur_mv_grid,
                    self.ref_store.prev_frame_mvs.as_deref(),
                    self.ref_store.prev_cols_4x4,
                    &mut cur_seg_map,
                    prev_seg_opt,
                )?;
                all_blocks.extend(blocks);
                merged_counts.merge(&counts);
            }
        }

        // --- 4. Reconstruct each decoded block ---
        for block in &all_blocks {
            if block.is_inter {
                reconstruct_inter_block(
                    &mut fb,
                    block,
                    &header,
                    &self.ref_store.slots,
                    &mut self.mc_scratch,
                )?;
            } else {
                reconstruct_intra_block(&mut fb, block, &header, block.tile_col_start)?;
            }
        }

        // --- 5. Loop filter ---
        loop_filter_frame(&mut fb, &all_blocks, &header);

        // --- 6. Probability context adaptation ---
        let fctx = header.frame_ctx_id as usize;
        if header.refresh_ctx && header.parallel_mode {
            self.prob_ctx[fctx] = header.prob.clone();
            self.coef_ctx[fctx] = header.coef;
        } else if header.refresh_ctx && !header.parallel_mode {
            let comp_pred_mode = CompPredMode::try_from(header.comp_pred_mode).unwrap_or_default();
            adapt_probs(
                &mut self.prob_ctx[fctx],
                &mut self.coef_ctx[fctx],
                &merged_counts,
                is_kf,
                self.last_keyframe,
                header.high_precision_mvs,
                header.filter_mode == 4,
                header.tx_mode == crate::header::TxMode::TxModeSelect,
                comp_pred_mode,
            );
            if is_kf || header.intra_only {
                self.prob_ctx[fctx].skip = header.prob.skip;
                self.prob_ctx[fctx].tx32p = header.prob.tx32p;
                self.prob_ctx[fctx].tx16p = header.prob.tx16p;
                self.prob_ctx[fctx].tx8p = header.prob.tx8p;
            }
        }
        self.last_keyframe = header.frame_type == FrameType::KeyFrame;

        // --- 7. Output frame (if show_frame is set) & update reference slots ---
        if header.show_frame {
            let picture_type = if is_kf {
                PictureType::I
            } else {
                PictureType::P
            };
            let is_key = header.frame_type == FrameType::KeyFrame;
            let frame = self.build_video_frame(
                &fb,
                width,
                height,
                layout,
                color,
                picture_type,
                is_key,
                pkt,
            )?;
            self.output_queue.push_back(MediaUnit::Video(frame));
        }

        let ref_frame = Arc::new(RefFrame {
            fb,
            mv_grid: cur_mv_grid.clone(),
            width,
            height,
            cols_4x4,
            rows_4x4,
            layout,
            color,
            bit_depth: header.bit_depth,
            subsampling_x: header.subsampling_x,
            subsampling_y: header.subsampling_y,
        });
        self.ref_store.refresh(header.refresh_ref_mask, ref_frame);
        self.ref_store.rotate_mvpair(cur_mv_grid, cols_4x4);

        if header.segmentation.enabled && header.segmentation.update_map {
            self.prev_seg_map = cur_seg_map;
            self.prev_seg_cols8 = cols8;
            self.prev_seg_rows8 = rows8;
        }
        self.hdr_state.ref_deltas = header.ref_deltas;
        self.hdr_state.mode_deltas = header.mode_deltas;
        self.hdr_state.seg = header.segmentation.clone();
        self.last_show_frame = header.show_frame;
        self.last_width = width;
        self.last_height = height;

        Ok(())
    }
}

/// Internal newtype to keep `Vp9Decoder::open` signature tidy.
#[doc(hidden)]
pub struct TrackDescriptorWrapper<'a>(pub &'a wedeo_core::TrackDescriptor);

// ---------------------------------------------------------------------------
// Colorimetry & PixelLayout helpers
// ---------------------------------------------------------------------------

fn derive_pixel_layout(
    bit_depth: u8,
    cs: ColorSpace,
    ss_h: bool,
    ss_v: bool,
    profile: u8,
) -> Result<PixelLayout> {
    if bit_depth != 8 {
        return Err(Error::unsupported(
            "vp9",
            format!("VP9 {bit_depth}-bit decoding (profile {profile}) is not supported yet"),
        ));
    }
    if cs == ColorSpace::Rgb {
        return Ok(PixelLayout::GBRP);
    }
    match (ss_h, ss_v) {
        (true, true) => Ok(PixelLayout::YUV420P),
        (true, false) => Ok(PixelLayout::YUV422P),
        (false, false) => Ok(PixelLayout::YUV444P),
        (false, true) => Err(Error::unsupported(
            "vp9",
            "VP9 4:4:0 subsampling is not supported by PixelLayout",
        )),
    }
}

fn derive_color_desc(
    cs: ColorSpace,
    range: header::ColorRange,
    bit_depth: u8,
    fallback: ColorDesc,
) -> ColorDesc {
    let mut color = ColorDesc::UNSPECIFIED;
    match cs {
        ColorSpace::Unknown | ColorSpace::Reserved => {
            color.matrix = fallback.matrix;
            color.primaries = fallback.primaries;
            color.transfer = fallback.transfer;
        }
        ColorSpace::Bt601 => {
            color.matrix = MatrixCoefficients::Bt470bg;
            color.primaries = ColorPrimaries::Bt470bg;
            color.transfer = TransferCharacteristic::Smpte170m;
        }
        ColorSpace::Bt709 => {
            color.matrix = MatrixCoefficients::Bt709;
            color.primaries = ColorPrimaries::Bt709;
            color.transfer = TransferCharacteristic::Bt709;
        }
        ColorSpace::Smpte170 => {
            color.matrix = MatrixCoefficients::Smpte170m;
            color.primaries = ColorPrimaries::Smpte170m;
            color.transfer = TransferCharacteristic::Smpte170m;
        }
        ColorSpace::Smpte240 => {
            color.matrix = MatrixCoefficients::Smpte240m;
            color.primaries = ColorPrimaries::Smpte240m;
            color.transfer = TransferCharacteristic::Smpte240m;
        }
        ColorSpace::Bt2020Ncl => {
            color.matrix = MatrixCoefficients::Bt2020Ncl;
            color.primaries = ColorPrimaries::Bt2020;
            color.transfer = if bit_depth == 12 {
                TransferCharacteristic::Bt2020_12
            } else {
                TransferCharacteristic::Bt2020_10
            };
        }
        ColorSpace::Rgb => {
            color.matrix = MatrixCoefficients::Identity;
            color.primaries = ColorPrimaries::Bt709;
            color.transfer = TransferCharacteristic::Srgb;
        }
    }
    color.range = match range {
        header::ColorRange::Full => ColorRange::Full,
        header::ColorRange::Limited => ColorRange::Limited,
    };
    color.chroma_location = fallback.chroma_location;
    color
}

// ---------------------------------------------------------------------------
// Tile column decoder
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn decode_tile_column(
    tile_data: &[u8],
    tile_offsets: &[(usize, usize)],
    header: &header::FrameHeader,
    above: &mut AboveContext,
    tile_row: usize,
    tile_col: usize,
    tile_cols: usize,
    tr_start: usize,
    tr_end: usize,
    sb_cols: usize,
    cur_mv_grid: &mut [MvRefPair],
    prev_mv_grid: Option<&[MvRefPair]>,
    prev_cols_4x4: usize,
    cur_seg_map: &mut [u8],
    prev_seg_map: Option<&[u8]>,
) -> Result<(Vec<BlockInfo>, CountContext)> {
    let tile_col_start_sb = tile_sb_offset(tile_col, header.tile_cols_log2, sb_cols);
    let tile_col_end_sb = tile_sb_offset(tile_col + 1, header.tile_cols_log2, sb_cols).min(sb_cols);
    let tile_col_start_4x4 = tile_col_start_sb * 16;

    let tile_idx = tile_row * tile_cols + tile_col;
    let (tile_byte_start, tile_byte_len) = tile_offsets[tile_idx];
    let tile_byte_end = tile_byte_start + tile_byte_len;
    if tile_byte_end > tile_data.len() {
        return Err(crate::invalid_data());
    }
    let tile_bytes = &tile_data[tile_byte_start..tile_byte_end];

    let mut bd = BoolDecoder::new(tile_bytes).ok_or_else(crate::invalid_data)?;
    // Marker bit — must be 0 for valid streams.
    if bd.get_prob(128) {
        return Err(crate::invalid_data());
    }

    let mut left = LeftContext::new();
    let mut tile_ctx = TileDecodeContext::new(
        bd,
        header,
        above,
        &mut left,
        tile_col_start_4x4,
        cur_mv_grid,
        prev_mv_grid,
        prev_cols_4x4,
        cur_seg_map,
        prev_seg_map,
    );

    for sb_row in tr_start..tr_end {
        let is_kf = header.frame_type == FrameType::KeyFrame || header.intra_only;
        tile_ctx.left.reset(is_kf);
        for sb_col in tile_col_start_sb..tile_col_end_sb {
            let row = sb_row * 16;
            let col = sb_col * 16;
            decode_sb(&mut tile_ctx, row, col, BlockLevel::Bl64x64)?;
        }
    }

    let counts = std::mem::take(&mut tile_ctx.counts);
    let blocks = std::mem::take(&mut tile_ctx.blocks);
    Ok((blocks, counts))
}

// ---------------------------------------------------------------------------
// Tile layout helpers  (matches set_tile_offset in vp9.c)
// ---------------------------------------------------------------------------

fn tile_sb_offset(tile_num: usize, log2_tiles: u8, n_sbs: usize) -> usize {
    ((tile_num * n_sbs) >> (log2_tiles as usize)).min(n_sbs)
}

// ---------------------------------------------------------------------------
// Decoder trait implementation
// ---------------------------------------------------------------------------

impl Decoder for Vp9Decoder {
    fn send_packet(&mut self, pkt: Option<&Packet>) -> Result<SendStatus> {
        match pkt {
            None => {
                self.draining = true;
                Ok(SendStatus::Accepted)
            }
            Some(pkt) => {
                if self.draining {
                    return Err(Error::invalid_argument(
                        "vp9",
                        "decoder is draining; call flush() before sending new packets",
                    ));
                }
                let data = pkt.data.data();
                let ranges = wedeo_syntax::vp9::parse_superframe(data)?;
                for range in ranges {
                    self.decode_frame(&data[range], pkt)?;
                }
                Ok(SendStatus::Accepted)
            }
        }
    }

    fn receive(&mut self) -> Result<RecvStatus<MediaUnit>> {
        if let Some(unit) = self.output_queue.pop_front() {
            Ok(RecvStatus::Ready(unit))
        } else if self.draining {
            Ok(RecvStatus::Eof)
        } else {
            Ok(RecvStatus::NeedMore)
        }
    }

    fn flush(&mut self) {
        self.output_queue.clear();
        self.draining = false;
        self.ref_store.clear();
        self.last_show_frame = false;
        self.last_keyframe = false;
        self.last_width = 0;
        self.last_height = 0;
        self.hdr_state = PersistentHeaderState::default();
        self.prev_seg_map.clear();
        self.prev_seg_cols8 = 0;
        self.prev_seg_rows8 = 0;
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

// ---------------------------------------------------------------------------
// Factory registration
// ---------------------------------------------------------------------------

/// Factory for opening a pure-Rust [`Vp9Decoder`].
pub struct Vp9DecoderFactory;

impl DecoderFactory for Vp9DecoderFactory {
    fn descriptor(&self) -> &DecoderDescriptor {
        static DESC: DecoderDescriptor = DecoderDescriptor {
            codec_id: CodecId::Vp9,
            name: "vp9",
            long_name: "VP9 Video",
            media_type: MediaType::Video,
        };
        &DESC
    }

    fn open(
        &self,
        track: &wedeo_core::TrackDescriptor,
        ctx: CodecContext,
    ) -> Result<Box<dyn Decoder>> {
        if track.codec_id != CodecId::Vp9 {
            return Err(Error::invalid_argument(
                "vp9",
                format!(
                    "codec id mismatch: expected vp9, got {}",
                    track.codec_id.name()
                ),
            ));
        }
        Ok(Box::new(Vp9Decoder::open(
            &TrackDescriptorWrapper(track),
            ctx,
        )?))
    }
}
