//! Native Matroska (`.mkv`, `.mka`, `.mks`) and WebM (`.webm`) EBML muxer.
//!
//! Supports both seekable outputs (patching `Segment` size, `Info` `Duration`, and `SeekHead`
//! pointing to `Cues`) and non-seekable streaming outputs (indeterminate-size `Segment` with
//! exact-size `Cluster`s).

use std::io::SeekFrom;

use wedeo_core::{
    ByteSink, ChromaLocation, CodecId, ColorDesc, ColorPrimaries, ColorRange, ContainerHeader,
    CropRect, Error, MatrixCoefficients, MediaType, Muxer, MuxerDescriptor, MuxerFactory, Packet,
    PacketFlags, Rational, Result, Rounding, SideData, StreamParams, Timestamp, TrackDisposition,
    TrackId, TransferCharacteristic, rescale_q,
};

use crate::demuxer::CueEntry;
use crate::ebml::{
    ATTACHED_FILE, ATTACHMENTS, AUDIO, BIT_DEPTH, BITS_PER_CHANNEL, BLOCK, BLOCK_DURATION,
    BLOCK_GROUP, CHANNELS, CHAP_LANGUAGE, CHAP_STRING, CHAPTER_ATOM, CHAPTER_DISPLAY,
    CHAPTER_TIME_END, CHAPTER_TIME_START, CHAPTER_UID, CHAPTERS, CHROMA_SITING_HORZ,
    CHROMA_SITING_VERT, CLUSTER, CLUSTER_TIMECODE, CODEC_DELAY, CODEC_ID, CODEC_PRIVATE, COLOUR,
    CUE_CLUSTER_POSITION, CUE_POINT, CUE_TIME, CUE_TRACK, CUE_TRACK_POSITIONS, CUES,
    DEFAULT_DURATION, DISCARD_PADDING, DISPLAY_HEIGHT, DISPLAY_UNIT, DISPLAY_WIDTH, DURATION,
    EBML_DOCTYPE, EBML_DOCTYPE_READ_VERSION, EBML_DOCTYPE_VERSION, EBML_HEADER, EBML_MAX_ID_LENGTH,
    EBML_MAX_SIZE_LENGTH, EBML_READ_VERSION, EBML_VERSION, EBML_VOID, EDITION_ENTRY, FILE_DATA,
    FILE_MIME_TYPE, FILE_NAME, FILE_UID, FLAG_COMMENTARY, FLAG_DEFAULT, FLAG_ENABLED, FLAG_FORCED,
    FLAG_HEARING_IMPAIRED, FLAG_VISUAL_IMPAIRED, INFO, LANGUAGE, MATRIX_COEFFICIENTS, MUXING_APP,
    NAME, PIXEL_CROP_BOTTOM, PIXEL_CROP_LEFT, PIXEL_CROP_RIGHT, PIXEL_CROP_TOP, PIXEL_HEIGHT,
    PIXEL_WIDTH, PRIMARIES, RANGE, REFERENCE_BLOCK, SAMPLING_FREQUENCY, SEEK, SEEK_HEAD, SEEK_ID,
    SEEK_POSITION, SEGMENT, SIMPLE_BLOCK, SIMPLE_TAG, SIZE_UNKNOWN, TAG, TAG_NAME, TAG_STRING,
    TAGS, TARGET_TRACK_UID, TARGETS, TIMECODE_SCALE, TITLE, TRACK_ENTRY, TRACK_NUMBER, TRACK_TYPE,
    TRACK_TYPE_AUDIO, TRACK_TYPE_SUBTITLE, TRACK_TYPE_VIDEO, TRACK_UID, TRACKS,
    TRANSFER_CHARACTERISTICS, VIDEO, WRITING_APP, matroska_from_codec_id, write_bytes_element,
    write_element_id, write_float64_element, write_master_element, write_sint_element,
    write_string_element, write_uint_element, write_vint_size, write_vint_size_fixed8,
};

const DEFAULT_TIMECODE_SCALE_NS: u64 = 1_000_000; // 1 ms ticks
const SEGMENT_TIME_BASE: Rational = Rational::new(1, 1000);
const SEEK_HEAD_RESERVE_SIZE: usize = 160;
const MAX_CLUSTER_BYTES: usize = 2 * 1024 * 1024;
const MAX_CLUSTER_SPAN_TICKS: i64 = 30_000;

struct MuxTrackState {
    track_id: TrackId,
    track_number: u64,
    track_uid: u64,
    codec_id: CodecId,
    media_type: MediaType,
    sample_rate: u32,
    nal_length_size: Option<u8>,
}

struct PendingCluster {
    timecode: i64,
    body: Vec<u8>,
    cue_tracks: Vec<(u64, i64)>,
}

/// Native Matroska and WebM EBML muxer.
pub struct MkvMuxer {
    sink: Box<dyn ByteSink>,
    webm_only: bool,
    sink_pos: u64,
    header_written: bool,
    segment_size_offset: u64,
    segment_data_start: u64,
    seek_head_offset: Option<u64>,
    info_rel_offset: Option<u64>,
    tracks_rel_offset: Option<u64>,
    chapters_rel_offset: Option<u64>,
    attachments_rel_offset: Option<u64>,
    tags_rel_offset: Option<u64>,
    duration_float_offset: Option<u64>,
    tracks: Vec<MuxTrackState>,
    pending_cluster: Option<PendingCluster>,
    cues: Vec<CueEntry>,
    max_end_pts_ticks: i64,
}

impl MkvMuxer {
    /// Create a Matroska (`.mkv`) muxer writing to `sink`.
    pub fn new(sink: Box<dyn ByteSink>) -> Self {
        Self::with_mode(sink, false)
    }

    /// Create a WebM (`.webm`) muxer writing to `sink`.
    pub fn new_webm(sink: Box<dyn ByteSink>) -> Self {
        Self::with_mode(sink, true)
    }

    fn with_mode(sink: Box<dyn ByteSink>, webm_only: bool) -> Self {
        Self {
            sink,
            webm_only,
            sink_pos: 0,
            header_written: false,
            segment_size_offset: 0,
            segment_data_start: 0,
            seek_head_offset: None,
            info_rel_offset: None,
            tracks_rel_offset: None,
            chapters_rel_offset: None,
            attachments_rel_offset: None,
            tags_rel_offset: None,
            duration_float_offset: None,
            tracks: Vec::new(),
            pending_cluster: None,
            cues: Vec::new(),
            max_end_pts_ticks: 0,
        }
    }

    fn write_raw(&mut self, bytes: &[u8]) -> Result<()> {
        self.sink
            .write_all(bytes)
            .map_err(|e| Error::io("mkv/muxer", &e))?;
        self.sink_pos += bytes.len() as u64;
        Ok(())
    }

    fn flush_cluster(&mut self) -> Result<()> {
        let Some(cluster) = self.pending_cluster.take() else {
            return Ok(());
        };
        if cluster.body.is_empty() {
            return Ok(());
        }

        let cluster_rel_offset = self.sink_pos.saturating_sub(self.segment_data_start);
        for (track_number, ts) in cluster.cue_tracks {
            self.cues.push(CueEntry {
                timestamp: ts,
                track_number,
                cluster_rel_offset,
            });
        }

        let mut cluster_payload = Vec::with_capacity(16 + cluster.body.len());
        write_uint_element(
            &mut cluster_payload,
            CLUSTER_TIMECODE,
            cluster.timecode.max(0) as u64,
        );
        cluster_payload.extend_from_slice(&cluster.body);

        let mut out = Vec::with_capacity(12 + cluster_payload.len());
        write_master_element(&mut out, CLUSTER, &cluster_payload);
        self.write_raw(&out)
    }

    fn build_cues_element(&self) -> Vec<u8> {
        let mut cues_body = Vec::new();
        for cue in &self.cues {
            let mut pos_body = Vec::new();
            write_uint_element(&mut pos_body, CUE_TRACK, cue.track_number);
            write_uint_element(&mut pos_body, CUE_CLUSTER_POSITION, cue.cluster_rel_offset);

            let mut point_body = Vec::new();
            write_uint_element(&mut point_body, CUE_TIME, cue.timestamp.max(0) as u64);
            write_master_element(&mut point_body, CUE_TRACK_POSITIONS, &pos_body);

            write_master_element(&mut cues_body, CUE_POINT, &point_body);
        }
        let mut out = Vec::new();
        write_master_element(&mut out, CUES, &cues_body);
        out
    }

    fn build_seek_head_with_void_padding(&self, cues_rel_offset: Option<u64>) -> Vec<u8> {
        let mut entries: Vec<(u32, u64)> = Vec::new();
        if let Some(off) = self.info_rel_offset {
            entries.push((INFO, off));
        }
        if let Some(off) = self.tracks_rel_offset {
            entries.push((TRACKS, off));
        }
        if let Some(off) = self.chapters_rel_offset {
            entries.push((CHAPTERS, off));
        }
        if let Some(off) = self.attachments_rel_offset {
            entries.push((ATTACHMENTS, off));
        }
        if let Some(off) = self.tags_rel_offset {
            entries.push((TAGS, off));
        }
        if let Some(off) = cues_rel_offset {
            entries.push((CUES, off));
        }

        let mut seek_head_body = Vec::new();
        for (id, rel_pos) in entries {
            let mut seek_entry = Vec::new();
            let mut id_bytes = Vec::with_capacity(4);
            write_element_id(&mut id_bytes, id);
            write_bytes_element(&mut seek_entry, SEEK_ID, &id_bytes);
            write_uint_element(&mut seek_entry, SEEK_POSITION, rel_pos);
            write_master_element(&mut seek_head_body, SEEK, &seek_entry);
        }

        let mut out = Vec::with_capacity(SEEK_HEAD_RESERVE_SIZE);
        write_master_element(&mut out, SEEK_HEAD, &seek_head_body);

        // Pad remaining bytes with an EBML Void element to match SEEK_HEAD_RESERVE_SIZE exactly.
        if out.len() + 2 <= SEEK_HEAD_RESERVE_SIZE {
            let rem = SEEK_HEAD_RESERVE_SIZE - out.len();
            out.push(EBML_VOID as u8);
            // Use a fixed VINT width that fills `rem - 1`
            let payload_len = rem - 2;
            out.push(0x80 | (payload_len as u8));
            out.resize(SEEK_HEAD_RESERVE_SIZE, 0);
        }
        out
    }
}

impl Muxer for MkvMuxer {
    fn write_header(&mut self, header: &ContainerHeader) -> Result<()> {
        if self.header_written {
            return Err(Error::invalid_argument(
                "mkv/muxer",
                "write_header called more than once",
            ));
        }
        if header.tracks.is_empty() {
            return Err(Error::invalid_argument(
                "mkv/muxer",
                "ContainerHeader must contain at least one track",
            ));
        }

        // 1. Validate tracks & prepare TrackEntry elements
        let mut tracks_body = Vec::new();
        self.tracks.clear();

        for (idx, track) in header.tracks.iter().enumerate() {
            let track_number = (idx + 1) as u64;
            let track_uid = (idx + 1) as u64;
            let (codec_str, pcm_bits) = matroska_from_codec_id(track.codec_id, self.webm_only)?;
            let media_type = track.params.media_type();
            let track_type = match media_type {
                MediaType::Video => TRACK_TYPE_VIDEO,
                MediaType::Audio => TRACK_TYPE_AUDIO,
                MediaType::Subtitle => TRACK_TYPE_SUBTITLE,
                _ => {
                    return Err(Error::unsupported(
                        "mkv/muxer",
                        "unsupported track media type for Matroska",
                    ));
                }
            };

            let mut entry = Vec::new();
            write_uint_element(&mut entry, TRACK_NUMBER, track_number);
            write_uint_element(&mut entry, TRACK_UID, track_uid);
            write_uint_element(&mut entry, TRACK_TYPE, u64::from(track_type));
            write_uint_element(&mut entry, FLAG_ENABLED, 1);
            let is_default = track.disposition.is_empty()
                || track.disposition.contains(TrackDisposition::DEFAULT);
            write_uint_element(&mut entry, FLAG_DEFAULT, u64::from(is_default));
            if track.disposition.contains(TrackDisposition::FORCED) {
                write_uint_element(&mut entry, FLAG_FORCED, 1);
            }
            if track
                .disposition
                .contains(TrackDisposition::HEARING_IMPAIRED)
            {
                write_uint_element(&mut entry, FLAG_HEARING_IMPAIRED, 1);
            }
            if track
                .disposition
                .contains(TrackDisposition::VISUAL_IMPAIRED)
            {
                write_uint_element(&mut entry, FLAG_VISUAL_IMPAIRED, 1);
            }
            if track.disposition.contains(TrackDisposition::COMMENTARY) {
                write_uint_element(&mut entry, FLAG_COMMENTARY, 1);
            }
            if let Some(lang) = &track.language
                && !lang.is_empty()
            {
                write_string_element(&mut entry, LANGUAGE, lang);
            }
            if let Some(title) = track.metadata.get("title")
                && !title.is_empty()
            {
                write_string_element(&mut entry, NAME, title);
            }
            write_string_element(&mut entry, CODEC_ID, codec_str);

            match &track.params {
                StreamParams::Video(v) => {
                    if v.width == 0 || v.height == 0 {
                        return Err(Error::invalid_argument(
                            "mkv/muxer",
                            "video track width and height must be non-zero",
                        ));
                    }
                    let codec_private = normalize_video_extradata(track.codec_id, &v.extradata)?;
                    if !codec_private.is_empty() {
                        write_bytes_element(&mut entry, CODEC_PRIVATE, &codec_private);
                    }
                    if v.avg_frame_rate.is_valid() && v.avg_frame_rate.num > 0 {
                        let dur_ns = (1_000_000_000u64 * v.avg_frame_rate.den as u64)
                            / (v.avg_frame_rate.num as u64);
                        if dur_ns > 0 {
                            write_uint_element(&mut entry, DEFAULT_DURATION, dur_ns);
                        }
                    }

                    let mut video_el = Vec::new();
                    write_uint_element(&mut video_el, PIXEL_WIDTH, u64::from(v.width));
                    write_uint_element(&mut video_el, PIXEL_HEIGHT, u64::from(v.height));
                    if v.crop != CropRect::ZERO {
                        if v.crop.top > 0 {
                            write_uint_element(
                                &mut video_el,
                                PIXEL_CROP_TOP,
                                u64::from(v.crop.top),
                            );
                        }
                        if v.crop.bottom > 0 {
                            write_uint_element(
                                &mut video_el,
                                PIXEL_CROP_BOTTOM,
                                u64::from(v.crop.bottom),
                            );
                        }
                        if v.crop.left > 0 {
                            write_uint_element(
                                &mut video_el,
                                PIXEL_CROP_LEFT,
                                u64::from(v.crop.left),
                            );
                        }
                        if v.crop.right > 0 {
                            write_uint_element(
                                &mut video_el,
                                PIXEL_CROP_RIGHT,
                                u64::from(v.crop.right),
                            );
                        }
                    }
                    let (crop_w, crop_h) = v.crop.display_size(v.width, v.height);
                    let base_w = if crop_w > 0 { crop_w } else { v.width };
                    let base_h = if crop_h > 0 { crop_h } else { v.height };
                    if v.sar.is_valid() && v.sar != Rational::new(1, 1) {
                        let num = i64::from(v.sar.num).max(1) as u64;
                        let den = i64::from(v.sar.den).max(1) as u64;
                        let mut a = u64::from(base_w);
                        let mut b = den;
                        while b != 0 {
                            let t = a % b;
                            a = b;
                            b = t;
                        }
                        let g = a.max(1);
                        let disp_w = (u64::from(base_w) / g).saturating_mul(num).max(1);
                        let disp_h = u64::from(base_h).saturating_mul(den / g).max(1);
                        write_uint_element(&mut video_el, DISPLAY_WIDTH, disp_w);
                        write_uint_element(&mut video_el, DISPLAY_HEIGHT, disp_h);
                        write_uint_element(&mut video_el, DISPLAY_UNIT, 0);
                    } else if v.crop != CropRect::ZERO {
                        write_uint_element(&mut video_el, DISPLAY_WIDTH, u64::from(base_w));
                        write_uint_element(&mut video_el, DISPLAY_HEIGHT, u64::from(base_h));
                        write_uint_element(&mut video_el, DISPLAY_UNIT, 0);
                    }

                    if v.color != ColorDesc::UNSPECIFIED || v.layout.depth.bits() > 8 {
                        let mut colour_el = Vec::new();
                        if v.color.matrix != MatrixCoefficients::Unspecified {
                            write_uint_element(
                                &mut colour_el,
                                MATRIX_COEFFICIENTS,
                                u64::from(v.color.matrix as u8),
                            );
                        }
                        if v.layout.depth.bits() > 0 {
                            write_uint_element(
                                &mut colour_el,
                                BITS_PER_CHANNEL,
                                u64::from(v.layout.depth.bits()),
                            );
                        }
                        match v.color.range {
                            ColorRange::Limited => write_uint_element(&mut colour_el, RANGE, 1),
                            ColorRange::Full => write_uint_element(&mut colour_el, RANGE, 2),
                            ColorRange::Unspecified => {}
                        }
                        if v.color.transfer != TransferCharacteristic::Unspecified {
                            write_uint_element(
                                &mut colour_el,
                                TRANSFER_CHARACTERISTICS,
                                u64::from(v.color.transfer as u8),
                            );
                        }
                        if v.color.primaries != ColorPrimaries::Unspecified {
                            write_uint_element(
                                &mut colour_el,
                                PRIMARIES,
                                u64::from(v.color.primaries as u8),
                            );
                        }
                        let (sh, sv) = match v.color.chroma_location {
                            ChromaLocation::TopLeft => (1, 1),
                            ChromaLocation::Left => (1, 2),
                            ChromaLocation::Top => (2, 1),
                            ChromaLocation::Center => (2, 2),
                            _ => (0, 0),
                        };
                        if sh > 0 {
                            write_uint_element(&mut colour_el, CHROMA_SITING_HORZ, sh);
                        }
                        if sv > 0 {
                            write_uint_element(&mut colour_el, CHROMA_SITING_VERT, sv);
                        }
                        if !colour_el.is_empty() {
                            write_master_element(&mut video_el, COLOUR, &colour_el);
                        }
                    }

                    write_master_element(&mut entry, VIDEO, &video_el);
                }
                StreamParams::Audio(a) => {
                    if a.sample_rate == 0 || a.channel_layout.nb_channels == 0 {
                        return Err(Error::invalid_argument(
                            "mkv/muxer",
                            "audio track sample_rate and channels must be non-zero",
                        ));
                    }
                    if !a.extradata.is_empty() {
                        write_bytes_element(&mut entry, CODEC_PRIVATE, &a.extradata);
                    }
                    if a.initial_padding > 0 {
                        let delay_ns = (u64::from(a.initial_padding) * 1_000_000_000)
                            / u64::from(a.sample_rate);
                        write_uint_element(&mut entry, CODEC_DELAY, delay_ns);
                    }
                    let mut audio_el = Vec::new();
                    write_float64_element(
                        &mut audio_el,
                        SAMPLING_FREQUENCY,
                        f64::from(a.sample_rate),
                    );
                    write_uint_element(
                        &mut audio_el,
                        CHANNELS,
                        u64::from(a.channel_layout.nb_channels),
                    );
                    let bits = if pcm_bits > 0 {
                        pcm_bits
                    } else {
                        a.bits_per_coded_sample
                    };
                    if bits > 0 {
                        write_uint_element(&mut audio_el, BIT_DEPTH, u64::from(bits));
                    }
                    write_master_element(&mut entry, AUDIO, &audio_el);
                }
                StreamParams::Subtitle(s) => {
                    if !s.header_script.is_empty() {
                        write_bytes_element(&mut entry, CODEC_PRIVATE, &s.header_script);
                    }
                }
                StreamParams::Data(_) => {
                    return Err(Error::unsupported(
                        "mkv/muxer",
                        "timed data tracks not supported in Matroska muxer",
                    ));
                }
            }

            let sample_rate = match &track.params {
                StreamParams::Audio(a) => a.sample_rate,
                _ => 0,
            };
            write_master_element(&mut tracks_body, TRACK_ENTRY, &entry);
            self.tracks.push(MuxTrackState {
                track_id: track.id,
                track_number,
                track_uid,
                codec_id: track.codec_id,
                media_type,
                sample_rate,
                nal_length_size: match &track.params {
                    StreamParams::Video(v) if v.extradata.first() == Some(&1) => {
                        match track.codec_id {
                            CodecId::H264 => v.extradata.get(4),
                            CodecId::Hevc => v.extradata.get(21),
                            _ => None,
                        }
                        .map(|b| (b & 3) + 1)
                    }
                    _ => None,
                },
            });
        }

        // 2. Write EBML Header
        let mut ebml_body = Vec::with_capacity(48);
        write_uint_element(&mut ebml_body, EBML_VERSION, 1);
        write_uint_element(&mut ebml_body, EBML_READ_VERSION, 1);
        write_uint_element(&mut ebml_body, EBML_MAX_ID_LENGTH, 4);
        write_uint_element(&mut ebml_body, EBML_MAX_SIZE_LENGTH, 8);
        let doctype = if self.webm_only || header.format_name == "webm" {
            "webm"
        } else {
            "matroska"
        };
        write_string_element(&mut ebml_body, EBML_DOCTYPE, doctype);
        write_uint_element(&mut ebml_body, EBML_DOCTYPE_VERSION, 4);
        write_uint_element(&mut ebml_body, EBML_DOCTYPE_READ_VERSION, 2);

        let mut header_bytes = Vec::with_capacity(64);
        write_master_element(&mut header_bytes, EBML_HEADER, &ebml_body);

        // 3. Write Segment header (8-byte VINT size placeholder)
        write_element_id(&mut header_bytes, SEGMENT);
        self.write_raw(&header_bytes)?;
        self.segment_size_offset = self.sink_pos;
        let mut seg_size_buf = Vec::with_capacity(8);
        write_vint_size_fixed8(&mut seg_size_buf, SIZE_UNKNOWN);
        self.write_raw(&seg_size_buf)?;
        self.segment_data_start = self.sink_pos;

        // 4. If seekable, reserve a fixed-size Void slot for SeekHead
        if self.sink.is_seekable() {
            self.seek_head_offset = Some(self.sink_pos);
            let mut void_slot = Vec::with_capacity(SEEK_HEAD_RESERVE_SIZE);
            void_slot.push(EBML_VOID as u8);
            void_slot.push(0x80 | ((SEEK_HEAD_RESERVE_SIZE - 2) as u8));
            void_slot.resize(SEEK_HEAD_RESERVE_SIZE, 0);
            self.write_raw(&void_slot)?;
        }

        // 5. Write Info element (reserving 8-byte float64 Duration for seekable patching)
        self.info_rel_offset = Some(self.sink_pos - self.segment_data_start);
        let mut info_body = Vec::with_capacity(96);
        write_uint_element(&mut info_body, TIMECODE_SCALE, DEFAULT_TIMECODE_SCALE_NS);
        write_string_element(&mut info_body, MUXING_APP, "wedeo");
        write_string_element(&mut info_body, WRITING_APP, "wedeo");
        if let Some(title) = header.metadata.get("title")
            && !title.is_empty()
        {
            write_string_element(&mut info_body, TITLE, title);
        }
        let initial_dur = header.duration.map_or(0.0, |d| d as f64 / 1000.0);
        // Record relative position of the 8-byte float inside info_body
        let dur_rel_in_info = info_body.len() + 3; // 2-byte ID (0x4489) + 1-byte size (0x88)
        write_float64_element(&mut info_body, DURATION, initial_dur);

        let mut info_hdr = Vec::with_capacity(12);
        write_element_id(&mut info_hdr, INFO);
        write_vint_size(&mut info_hdr, info_body.len() as u64);
        self.write_raw(&info_hdr)?;
        self.duration_float_offset = Some(self.sink_pos + dur_rel_in_info as u64);
        self.write_raw(&info_body)?;

        // 6. Write Tracks element
        self.tracks_rel_offset = Some(self.sink_pos - self.segment_data_start);
        let mut tracks_el = Vec::with_capacity(12 + tracks_body.len());
        write_master_element(&mut tracks_el, TRACKS, &tracks_body);
        self.write_raw(&tracks_el)?;

        // 7. Write Chapters if present
        if !header.chapters.is_empty() && !self.webm_only {
            self.chapters_rel_offset = Some(self.sink_pos - self.segment_data_start);
            let chapters_el = build_chapters_element(header);
            self.write_raw(&chapters_el)?;
        }

        // 8. Write Attachments if present
        if !header.attachments.is_empty() && !self.webm_only {
            self.attachments_rel_offset = Some(self.sink_pos - self.segment_data_start);
            let attachments_el = build_attachments_element(header);
            self.write_raw(&attachments_el)?;
        }

        // 9. Write Tags if present
        let tags_el = build_tags_element(header, &self.tracks);
        if !tags_el.is_empty() {
            self.tags_rel_offset = Some(self.sink_pos - self.segment_data_start);
            self.write_raw(&tags_el)?;
        }

        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, pkt: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid_argument(
                "mkv/muxer",
                "write_header must be called before write_packet",
            ));
        }

        let Some(track_idx) = self.tracks.iter().position(|t| t.track_id == pkt.track_id) else {
            return Err(Error::not_found(
                "mkv/muxer",
                "packet track_id not found in Matroska muxer",
            ));
        };
        let track_number = self.tracks[track_idx].track_number;
        let codec_id = self.tracks[track_idx].codec_id;
        let media_type = self.tracks[track_idx].media_type;
        let sample_rate = self.tracks[track_idx].sample_rate;

        // Matroska blocks carry PTS. Fall back to DTS or 0 if PTS is unknown.
        let raw_ts = if pkt.pts.is_known() {
            pkt.pts
        } else if pkt.dts.is_known() {
            pkt.dts
        } else {
            Timestamp::new(0)
        };
        let pkt_tb = if pkt.time_base.is_valid() {
            pkt.time_base
        } else {
            SEGMENT_TIME_BASE
        };
        let pts_ticks = raw_ts
            .rescale(pkt_tb, SEGMENT_TIME_BASE, Rounding::NearInf)
            .get()
            .unwrap_or(0);
        let dur_ticks = if pkt.duration > 0 {
            rescale_q(pkt.duration, pkt_tb, SEGMENT_TIME_BASE)
        } else {
            0
        };
        self.max_end_pts_ticks = self.max_end_pts_ticks.max(pts_ticks + dur_ticks.max(1));

        let is_key = pkt.flags.contains(PacketFlags::KEY) || media_type != MediaType::Video;
        let is_video_key = media_type == MediaType::Video && pkt.flags.contains(PacketFlags::KEY);

        // Determine if we need to flush the current cluster and start a new one
        let need_new_cluster = match &self.pending_cluster {
            None => true,
            Some(c) => {
                let delta = pts_ticks - c.timecode;
                is_video_key
                    || !(i64::from(i16::MIN)..=i64::from(i16::MAX)).contains(&delta)
                    || delta.abs() > MAX_CLUSTER_SPAN_TICKS
                    || c.body.len() >= MAX_CLUSTER_BYTES
            }
        };

        if need_new_cluster {
            self.flush_cluster()?;
            self.pending_cluster = Some(PendingCluster {
                timecode: pts_ticks.max(0),
                body: Vec::new(),
                cue_tracks: Vec::new(),
            });
        }

        let payload_cow = normalize_packet_payload(
            codec_id,
            self.tracks[track_idx].nal_length_size,
            pkt.data.as_slice().unwrap_or(&[]),
        )?;
        let payload = payload_cow.as_ref();

        let cluster = self.pending_cluster.as_mut().unwrap();
        let delta =
            (pts_ticks - cluster.timecode).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;

        if is_video_key || (cluster.cue_tracks.is_empty() && is_key) {
            cluster.cue_tracks.push((track_number, pts_ticks.max(0)));
        }

        let discard_padding_ns: Option<i64> = if sample_rate > 0 {
            pkt.side_data.iter().find_map(|sd| match *sd {
                SideData::SampleTrim { skip_end, .. } if skip_end > 0 => {
                    let ns = (u128::from(skip_end) * 1_000_000_000) / u128::from(sample_rate);
                    Some(ns as i64)
                }
                _ => None,
            })
        } else {
            None
        };

        // Subtitles with explicit duration or audio packets carrying DiscardPadding
        // use BlockGroup + Block; ordinary Video/Audio use compact SimpleBlock.
        if (media_type == MediaType::Subtitle && dur_ticks > 0) || discard_padding_ns.is_some() {
            let mut block_data = Vec::with_capacity(4 + payload.len());
            write_vint_size(&mut block_data, track_number);
            block_data.extend_from_slice(&delta.to_be_bytes());
            block_data.push(0x00); // Block flags (no keyframe bit in Block)
            block_data.extend_from_slice(payload);

            let mut group_body = Vec::with_capacity(32 + block_data.len());
            write_bytes_element(&mut group_body, BLOCK, &block_data);
            if media_type == MediaType::Subtitle && dur_ticks > 0 {
                write_uint_element(&mut group_body, BLOCK_DURATION, dur_ticks as u64);
            }
            if !is_key {
                write_sint_element(&mut group_body, REFERENCE_BLOCK, 0);
            }
            if let Some(pad_ns) = discard_padding_ns {
                write_sint_element(&mut group_body, DISCARD_PADDING, pad_ns);
            }
            write_master_element(&mut cluster.body, BLOCK_GROUP, &group_body);
        } else {
            let mut sb = Vec::with_capacity(4 + payload.len());
            write_vint_size(&mut sb, track_number);
            sb.extend_from_slice(&delta.to_be_bytes());
            let mut flags = 0u8;
            if is_key {
                flags |= 0x80;
            }
            if pkt.flags.contains(PacketFlags::DISPOSABLE) {
                flags |= 0x01;
            }
            sb.push(flags);
            sb.extend_from_slice(payload);
            write_bytes_element(&mut cluster.body, SIMPLE_BLOCK, &sb);
        }

        Ok(())
    }

    fn finalize(mut self: Box<Self>) -> Result<()> {
        self.flush_cluster()?;

        let mut cues_rel_offset = None;
        if !self.cues.is_empty() {
            cues_rel_offset = Some(self.sink_pos.saturating_sub(self.segment_data_start));
            let cues_el = self.build_cues_element();
            self.write_raw(&cues_el)?;
        }

        let end_pos = self.sink_pos;
        if self.sink.is_seekable() {
            // 1. Patch Segment 8-byte VINT size
            let seg_size = end_pos.saturating_sub(self.segment_data_start);
            self.sink
                .seek(SeekFrom::Start(self.segment_size_offset))
                .map_err(|e| Error::io("mkv/muxer", &e))?;
            let mut size_buf = Vec::with_capacity(8);
            write_vint_size_fixed8(&mut size_buf, seg_size);
            self.sink
                .write_all(&size_buf)
                .map_err(|e| Error::io("mkv/muxer", &e))?;

            // 2. Patch Info Duration float64
            if let Some(dur_off) = self.duration_float_offset
                && self.max_end_pts_ticks > 0
            {
                self.sink
                    .seek(SeekFrom::Start(dur_off))
                    .map_err(|e| Error::io("mkv/muxer", &e))?;
                let dur_f64 = self.max_end_pts_ticks as f64;
                self.sink
                    .write_all(&dur_f64.to_be_bytes())
                    .map_err(|e| Error::io("mkv/muxer", &e))?;
            }

            // 3. Patch SeekHead into reserved slot
            if let Some(sh_off) = self.seek_head_offset {
                let sh_bytes = self.build_seek_head_with_void_padding(cues_rel_offset);
                if sh_bytes.len() == SEEK_HEAD_RESERVE_SIZE {
                    self.sink
                        .seek(SeekFrom::Start(sh_off))
                        .map_err(|e| Error::io("mkv/muxer", &e))?;
                    self.sink
                        .write_all(&sh_bytes)
                        .map_err(|e| Error::io("mkv/muxer", &e))?;
                }
            }

            self.sink
                .seek(SeekFrom::Start(end_pos))
                .map_err(|e| Error::io("mkv/muxer", &e))?;
        }

        self.sink.flush().map_err(|e| Error::io("mkv/muxer", &e))?;
        Ok(())
    }
}

fn normalize_video_extradata(codec_id: CodecId, extradata: &[u8]) -> Result<Vec<u8>> {
    if extradata.is_empty() {
        return Ok(Vec::new());
    }
    match codec_id {
        CodecId::H264 => {
            if extradata[0] == 1 {
                Ok(extradata.to_vec())
            } else {
                wedeo_syntax::h264::avcc_from_annexb(extradata)
            }
        }
        CodecId::Av1 => {
            if extradata.len() >= 4 && (extradata[0] & 0x80) != 0 {
                Ok(extradata.to_vec())
            } else {
                wedeo_syntax::av1::av1c_from_obus(extradata, true)
            }
        }
        _ => Ok(extradata.to_vec()),
    }
}

fn normalize_packet_payload(
    codec_id: CodecId,
    length_size: Option<u8>,
    data: &[u8],
) -> Result<std::borrow::Cow<'_, [u8]>> {
    if !matches!(codec_id, CodecId::H264 | CodecId::Hevc) || !is_annexb_start_code(data) {
        return Ok(std::borrow::Cow::Borrowed(data));
    }
    // A four-byte length in 256..512 starts with the three-byte Annex B delimiter.
    // Prefer valid packet framing from the track configuration when both are possible.
    if let Some(size) = length_size {
        let mut rest = data;
        while let Some((prefix, tail)) = rest.split_at_checked(usize::from(size)) {
            let len = prefix
                .iter()
                .fold(0usize, |n, &b| (n << 8) | usize::from(b));
            if len == 0 {
                break;
            }
            let Some(next) = tail.get(len..) else { break };
            if next.is_empty() {
                return Ok(std::borrow::Cow::Borrowed(data));
            }
            rest = next;
        }
    }
    let bytes = match codec_id {
        CodecId::H264 => wedeo_syntax::h264::annexb_to_avcc(data, length_size.unwrap_or(4))?,
        CodecId::Hevc => wedeo_syntax::hevc::annexb_to_hvcc(data, length_size.unwrap_or(4))?,
        _ => unreachable!(),
    };
    Ok(std::borrow::Cow::Owned(bytes))
}

fn is_annexb_start_code(data: &[u8]) -> bool {
    (data.len() >= 4 && data[0..4] == [0, 0, 0, 1]) || (data.len() >= 3 && data[0..3] == [0, 0, 1])
}

fn build_chapters_element(header: &ContainerHeader) -> Vec<u8> {
    let ns_tb = Rational::new(1, 1_000_000_000);
    let mut edition_body = Vec::new();
    for (idx, ch) in header.chapters.iter().enumerate() {
        let mut atom_body = Vec::new();
        let uid = if ch.id > 0 {
            ch.id as u64
        } else {
            (idx + 1) as u64
        };
        write_uint_element(&mut atom_body, CHAPTER_UID, uid);
        let start_ns = ch
            .start
            .rescale(ch.time_base, ns_tb, Rounding::NearInf)
            .get()
            .unwrap_or(0)
            .max(0) as u64;
        write_uint_element(&mut atom_body, CHAPTER_TIME_START, start_ns);
        if let Some(end_ns) = ch.end.rescale(ch.time_base, ns_tb, Rounding::NearInf).get()
            && end_ns >= 0
        {
            write_uint_element(&mut atom_body, CHAPTER_TIME_END, end_ns as u64);
        }
        if let Some(title) = ch.metadata.get("title") {
            let mut disp_body = Vec::new();
            write_string_element(&mut disp_body, CHAP_STRING, title);
            let lang = ch.metadata.get("language").unwrap_or("eng");
            write_string_element(&mut disp_body, CHAP_LANGUAGE, lang);
            write_master_element(&mut atom_body, CHAPTER_DISPLAY, &disp_body);
        }
        write_master_element(&mut edition_body, CHAPTER_ATOM, &atom_body);
    }
    let mut chapters_body = Vec::new();
    write_master_element(&mut chapters_body, EDITION_ENTRY, &edition_body);
    let mut out = Vec::new();
    write_master_element(&mut out, CHAPTERS, &chapters_body);
    out
}

fn build_attachments_element(header: &ContainerHeader) -> Vec<u8> {
    let mut att_body = Vec::new();
    for (idx, att) in header.attachments.iter().enumerate() {
        let mut file_body = Vec::new();
        write_string_element(&mut file_body, FILE_NAME, &att.name);
        write_string_element(&mut file_body, FILE_MIME_TYPE, &att.mime_type);
        write_bytes_element(
            &mut file_body,
            FILE_DATA,
            att.data.as_slice().unwrap_or(&[]),
        );
        write_uint_element(&mut file_body, FILE_UID, (idx + 1) as u64);
        write_master_element(&mut att_body, ATTACHED_FILE, &file_body);
    }
    let mut out = Vec::new();
    write_master_element(&mut out, ATTACHMENTS, &att_body);
    out
}

fn build_tags_element(header: &ContainerHeader, tracks: &[MuxTrackState]) -> Vec<u8> {
    let mut tags_body = Vec::new();

    // Global container metadata
    if !header.metadata.is_empty() {
        let mut tag_body = Vec::new();
        write_master_element(&mut tag_body, TARGETS, &[]);
        for (k, v) in header.metadata.iter() {
            if k.eq_ignore_ascii_case("title")
                || k.eq_ignore_ascii_case("muxing_app")
                || k.eq_ignore_ascii_case("encoder")
            {
                continue;
            }
            let mut st = Vec::new();
            write_string_element(&mut st, TAG_NAME, k);
            write_string_element(&mut st, TAG_STRING, v);
            write_master_element(&mut tag_body, SIMPLE_TAG, &st);
        }
        if tag_body.len() > 4 {
            write_master_element(&mut tags_body, TAG, &tag_body);
        }
    }

    // Track-level metadata
    for (track_desc, mts) in header.tracks.iter().zip(tracks.iter()) {
        if track_desc.metadata.is_empty() {
            continue;
        }
        let mut simple_tags = Vec::new();
        for (k, v) in track_desc.metadata.iter() {
            if k.eq_ignore_ascii_case("title") {
                continue;
            }
            let mut st = Vec::new();
            write_string_element(&mut st, TAG_NAME, k);
            write_string_element(&mut st, TAG_STRING, v);
            write_master_element(&mut simple_tags, SIMPLE_TAG, &st);
        }
        if !simple_tags.is_empty() {
            let mut targets = Vec::new();
            write_uint_element(&mut targets, TARGET_TRACK_UID, mts.track_uid);
            let mut tag_body = Vec::new();
            write_master_element(&mut tag_body, TARGETS, &targets);
            tag_body.extend_from_slice(&simple_tags);
            write_master_element(&mut tags_body, TAG, &tag_body);
        }
    }

    if tags_body.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    write_master_element(&mut out, TAGS, &tags_body);
    out
}

// ---------------------------------------------------------------------------
// MuxerFactories (Matroska & WebM)
// ---------------------------------------------------------------------------

pub struct MkvMuxerFactory;

impl MuxerFactory for MkvMuxerFactory {
    fn descriptor(&self) -> &MuxerDescriptor {
        static DESC: MuxerDescriptor = MuxerDescriptor {
            name: "matroska",
            long_name: "Matroska (MKV / MKA / MKS)",
            extensions: &["mkv", "mk3d", "mka", "mks"],
            mime_types: &["video/x-matroska", "audio/x-matroska"],
            default_video: CodecId::H264,
            default_audio: CodecId::Opus,
        };
        &DESC
    }

    fn create(&self, sink: Box<dyn ByteSink>) -> Result<Box<dyn Muxer>> {
        Ok(Box::new(MkvMuxer::new(sink)))
    }
}

pub struct WebmMuxerFactory;

impl MuxerFactory for WebmMuxerFactory {
    fn descriptor(&self) -> &MuxerDescriptor {
        static DESC: MuxerDescriptor = MuxerDescriptor {
            name: "webm",
            long_name: "WebM (VP8/VP9/AV1 + Opus/Vorbis)",
            extensions: &["webm"],
            mime_types: &["video/webm", "audio/webm"],
            default_video: CodecId::Vp9,
            default_audio: CodecId::Opus,
        };
        &DESC
    }

    fn create(&self, sink: Box<dyn ByteSink>) -> Result<Box<dyn Muxer>> {
        Ok(Box::new(MkvMuxer::new_webm(sink)))
    }
}
