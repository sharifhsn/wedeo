//! Complete native Matroska (`.mkv`, `.mka`, `.mks`) and WebM (`.webm`) EBML demuxer.

use std::collections::VecDeque;

use wedeo_core::{
    Attachment, AudioStreamParams, BufByteSource, Buffer, ByteSource, ChannelLayout, Chapter,
    ChromaLocation, CodecId, ColorDesc, ColorPrimaries, ColorRange, ContainerHeader, CropRect,
    Demuxer, DemuxerDescriptor, DemuxerFactory, Error, ErrorPolicy, HealthEvent, HealthIssue,
    MasteringDisplayMeta, MatrixCoefficients, Metadata, Packet, PacketFlags, PixelLayout, Rational,
    RecvStatus, Result, Rounding, SampleFormat, SeekTarget, SideData, StreamHealth, StreamParams,
    SubtitleFormat, SubtitleStreamParams, TIME_BASE_Q, Timestamp, TrackDescriptor,
    TrackDisposition, TrackId, TransferCharacteristic, VideoStreamParams, reduce, rescale_q,
};

use crate::ebml::{
    ATTACHED_FILE, ATTACHMENTS, AUDIO, BIT_DEPTH, BITS_PER_CHANNEL, BLOCK, BLOCK_DURATION,
    BLOCK_GROUP, CHANNELS, CHAP_LANGUAGE, CHAP_STRING, CHAPTER_ATOM, CHAPTER_DISPLAY,
    CHAPTER_TIME_END, CHAPTER_TIME_START, CHAPTER_UID, CHAPTERS, CHROMA_SITING_HORZ,
    CHROMA_SITING_VERT, CHROMA_SUBSAMPLING_HORZ, CHROMA_SUBSAMPLING_VERT, CLUSTER,
    CLUSTER_TIMECODE, CODEC_DELAY, CODEC_ID, CODEC_PRIVATE, COLOUR, CUE_CLUSTER_POSITION,
    CUE_POINT, CUE_TIME, CUE_TRACK, CUE_TRACK_POSITIONS, CUES, DEFAULT_DURATION, DISCARD_PADDING,
    DISPLAY_HEIGHT, DISPLAY_UNIT, DISPLAY_WIDTH, DURATION, EBML_CRC32, EBML_DOCTYPE, EBML_HEADER,
    EBML_VOID, EDITION_ENTRY, FILE_DATA, FILE_MIME_TYPE, FILE_NAME, FLAG_COMMENTARY, FLAG_DEFAULT,
    FLAG_FORCED, FLAG_HEARING_IMPAIRED, FLAG_VISUAL_IMPAIRED, INFO, LANGUAGE, LANGUAGE_BCP47,
    LUMINANCE_MAX, LUMINANCE_MIN, MASTERING_METADATA, MATRIX_COEFFICIENTS, MAX_CLL, MAX_FALL,
    MUXING_APP, NAME, OUTPUT_SAMPLING_FREQUENCY, PIXEL_CROP_BOTTOM, PIXEL_CROP_LEFT,
    PIXEL_CROP_RIGHT, PIXEL_CROP_TOP, PIXEL_HEIGHT, PIXEL_WIDTH, PRIMARIES,
    PRIMARY_B_CHROMATICITY_X, PRIMARY_B_CHROMATICITY_Y, PRIMARY_G_CHROMATICITY_X,
    PRIMARY_G_CHROMATICITY_Y, PRIMARY_R_CHROMATICITY_X, PRIMARY_R_CHROMATICITY_Y, RANGE,
    REFERENCE_BLOCK, SAMPLING_FREQUENCY, SEEK, SEEK_HEAD, SEEK_ID, SEEK_POSITION, SEEK_PRE_ROLL,
    SEGMENT, SIMPLE_BLOCK, SIMPLE_TAG, SIZE_UNKNOWN, TAG, TAG_NAME, TAG_STRING, TAGS,
    TARGET_TRACK_UID, TARGETS, TIMECODE_SCALE, TITLE, TRACK_ENTRY, TRACK_NUMBER, TRACK_TYPE,
    TRACK_TYPE_AUDIO, TRACK_TYPE_SUBTITLE, TRACK_TYPE_VIDEO, TRACK_UID, TRACKS,
    TRANSFER_CHARACTERISTICS, VIDEO, WHITE_POINT_CHROMATICITY_X, WHITE_POINT_CHROMATICITY_Y,
    WRITING_APP, codec_id_from_matroska, is_top_level_segment_element, parse_laced_frames,
    read_bytes, read_element_header, read_float, read_sint, read_string, read_uint, read_vint_data,
    skip_element,
};

/// Internal per-track state tracked by [`MkvDemuxer`].
#[derive(Debug, Clone)]
struct MkvTrackState {
    track_number: u64,
    track_uid: u64,
    track_id: TrackId,
    default_duration_ticks: i64,
    time_base: Rational,
    last_max_pts: Option<i64>,
    saw_reordering: bool,
    pending_side_data: Vec<SideData>,
}

/// One index entry in the Matroska `Cues` table (or dynamically discovered `Cluster`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CueEntry {
    /// Timestamp in Segment timecode units.
    pub timestamp: i64,
    /// Matroska track number (`0` = any track).
    pub track_number: u64,
    /// Byte offset of the `Cluster` relative to `segment_data_start`.
    pub cluster_rel_offset: u64,
}

/// Native Matroska and WebM demuxer.
pub struct MkvDemuxer {
    io: BufByteSource,
    policy: ErrorPolicy,
    header: ContainerHeader,
    health: StreamHealth,
    tracks: Vec<MkvTrackState>,
    timecode_scale_ns: u64,
    segment_time_base: Rational,
    segment_data_start: u64,
    segment_end: Option<u64>,
    first_cluster_offset: Option<u64>,
    in_cluster: bool,
    cluster_end: Option<u64>,
    cluster_timecode: i64,
    cues: Vec<CueEntry>,
    pending_packets: VecDeque<Packet>,
}

impl MkvDemuxer {
    /// Open a Matroska/WebM stream and parse its `EBML` header, `Segment` info, `Tracks`,
    /// `Chapters`, `Attachments`, `Tags`, and `Cues`.
    pub fn open(source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Self> {
        let mut demuxer = Self {
            io: BufByteSource::new(source),
            policy,
            header: ContainerHeader::new("matroska"),
            health: StreamHealth::new(),
            tracks: Vec::new(),
            timecode_scale_ns: 1_000_000,
            segment_time_base: Rational::new(1, 1000),
            segment_data_start: 0,
            segment_end: None,
            first_cluster_offset: None,
            in_cluster: false,
            cluster_end: None,
            cluster_timecode: 0,
            cues: Vec::new(),
            pending_packets: VecDeque::new(),
        };
        demuxer.read_container_header()?;
        Ok(demuxer)
    }

    /// Expose the parsed `Cues` table for inspection and testing.
    pub fn cues(&self) -> &[CueEntry] {
        &self.cues
    }

    fn record_corruption(
        &mut self,
        track: TrackId,
        pts: Timestamp,
        detail: &'static str,
    ) -> Result<()> {
        let pos = Some(self.io.tell());
        self.health.record(HealthEvent {
            track,
            pts,
            byte_offset: pos,
            kind: HealthIssue::CorruptSyntax,
            detail,
        });
        if self.policy == ErrorPolicy::Strict {
            return Err(Error::corrupt("mkv/demuxer", detail));
        }
        Ok(())
    }

    fn read_container_header(&mut self) -> Result<()> {
        // 1. Parse EBML Header
        let Some(ebml_hdr) = read_element_header(&mut self.io)? else {
            return Err(Error::invalid_data("mkv/demuxer", "empty stream"));
        };
        if ebml_hdr.id != EBML_HEADER || ebml_hdr.size == SIZE_UNKNOWN {
            return Err(Error::invalid_data(
                "mkv/demuxer",
                "missing or invalid EBML header",
            ));
        }
        let ebml_end = ebml_hdr.data_offset + ebml_hdr.size;
        let mut doc_type = String::from("matroska");
        while self.io.tell() < ebml_end {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            if child.id == EBML_DOCTYPE {
                doc_type = read_string(&mut self.io, child.size)?;
            } else {
                skip_element(&mut self.io, child.size)?;
            }
        }
        if !doc_type.eq_ignore_ascii_case("matroska") && !doc_type.eq_ignore_ascii_case("webm") {
            return Err(Error::invalid_data(
                "mkv/demuxer",
                "unsupported EBML DocType (expected matroska or webm)",
            ));
        }
        self.header.format_name = if doc_type.eq_ignore_ascii_case("webm") {
            "webm"
        } else {
            "matroska"
        };

        // 2. Locate Segment element (skipping any top-level Void/CRC)
        let seg_hdr = loop {
            let Some(hdr) = read_element_header(&mut self.io)? else {
                return Err(Error::invalid_data(
                    "mkv/demuxer",
                    "missing Segment element",
                ));
            };
            if hdr.id == SEGMENT {
                break hdr;
            }
            skip_element(&mut self.io, hdr.size)?;
        };
        self.segment_data_start = seg_hdr.data_offset;
        self.segment_end = seg_hdr.end_offset();

        // 3. Parse Segment top-level elements up to the first Cluster
        let mut duration_ticks: Option<f64> = None;
        let mut seek_targets: Vec<(u32, u64)> = Vec::new();
        let mut parsed_cues = false;
        let mut parsed_chapters = false;
        let mut parsed_tags = false;
        let mut parsed_attachments = false;

        loop {
            let pos = self.io.tell();
            if self.segment_end.is_some_and(|end| pos >= end) {
                break;
            }
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };

            match child.id {
                SEEK_HEAD => {
                    if child.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data(
                            "mkv/demuxer",
                            "SeekHead with unknown size",
                        ));
                    }
                    self.parse_seek_head(child.data_offset + child.size, &mut seek_targets)?;
                }
                INFO => {
                    if child.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data("mkv/demuxer", "Info with unknown size"));
                    }
                    self.parse_info(child.data_offset + child.size, &mut duration_ticks)?;
                }
                TRACKS => {
                    if child.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data(
                            "mkv/demuxer",
                            "Tracks with unknown size",
                        ));
                    }
                    self.parse_tracks(child.data_offset + child.size)?;
                }
                CHAPTERS => {
                    if child.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data(
                            "mkv/demuxer",
                            "Chapters with unknown size",
                        ));
                    }
                    self.parse_chapters(child.data_offset + child.size)?;
                    parsed_chapters = true;
                }
                ATTACHMENTS => {
                    if child.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data(
                            "mkv/demuxer",
                            "Attachments with unknown size",
                        ));
                    }
                    self.parse_attachments(child.data_offset + child.size)?;
                    parsed_attachments = true;
                }
                TAGS => {
                    if child.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data("mkv/demuxer", "Tags with unknown size"));
                    }
                    self.parse_tags(child.data_offset + child.size)?;
                    parsed_tags = true;
                }
                CUES => {
                    if child.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data("mkv/demuxer", "Cues with unknown size"));
                    }
                    self.parse_cues(child.data_offset + child.size)?;
                    parsed_cues = true;
                }
                CLUSTER => {
                    self.first_cluster_offset = Some(child.header_offset);
                    self.enter_cluster(&child);
                    break;
                }
                _ => {
                    if child.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data(
                            "mkv/demuxer",
                            "unknown Segment element with indeterminate size",
                        ));
                    }
                    skip_element(&mut self.io, child.size)?;
                }
            }
        }

        // 4. Finalize segment time_base now that Info is parsed
        if self.timecode_scale_ns == 0 {
            return Err(Error::invalid_data(
                "mkv/demuxer",
                "TimecodeScale cannot be 0",
            ));
        }
        let (tb_num, tb_den, _) = reduce(
            self.timecode_scale_ns as i64,
            1_000_000_000,
            i32::MAX as i64,
        );
        self.segment_time_base = Rational::new(tb_num, tb_den);

        // Update all tracks and internal track states with the resolved segment_time_base
        for track_desc in &mut self.header.tracks {
            track_desc.time_base = self.segment_time_base;
            if let Some(dur_ticks) = duration_ticks {
                track_desc.duration = dur_ticks.round() as i64;
            }
        }
        for ts in &mut self.tracks {
            ts.time_base = self.segment_time_base;
            if ts.default_duration_ticks > 0 {
                // default_duration_ticks was temporarily stored in nanoseconds; convert to ticks
                ts.default_duration_ticks =
                    (ts.default_duration_ticks as u64 / self.timecode_scale_ns) as i64;
            }
        }

        if let Some(dur_ticks) = duration_ticks {
            let ticks = dur_ticks.round() as i64;
            self.header.duration = Some(rescale_q(ticks, self.segment_time_base, TIME_BASE_Q));
        }
        self.header.start_time = Timestamp::new(0);

        // 5. If seekable and SeekHead points to Cues/Chapters/Tags/Attachments after Clusters,
        //    seek there, parse them, and restore our stream position at the first Cluster.
        if self.io.is_seekable() && !seek_targets.is_empty() {
            let saved_pos = self.io.tell();
            for (seek_id, rel_pos) in seek_targets {
                let target_abs = self.segment_data_start.saturating_add(rel_pos);
                if self.segment_end.is_some_and(|end| target_abs >= end) {
                    continue;
                }
                let need = match seek_id {
                    CUES => !parsed_cues,
                    CHAPTERS => !parsed_chapters,
                    TAGS => !parsed_tags,
                    ATTACHMENTS => !parsed_attachments,
                    _ => false,
                };
                if !need {
                    continue;
                }
                if self.io.seek_abs(target_abs).is_ok()
                    && let Ok(Some(hdr)) = read_element_header(&mut self.io)
                    && hdr.id == seek_id
                    && hdr.size != SIZE_UNKNOWN
                {
                    let end = hdr.data_offset + hdr.size;
                    match seek_id {
                        CUES => {
                            let _ = self.parse_cues(end);
                            parsed_cues = true;
                        }
                        CHAPTERS => {
                            let _ = self.parse_chapters(end);
                            parsed_chapters = true;
                        }
                        TAGS => {
                            let _ = self.parse_tags(end);
                            parsed_tags = true;
                        }
                        ATTACHMENTS => {
                            let _ = self.parse_attachments(end);
                            parsed_attachments = true;
                        }
                        _ => {}
                    }
                }
            }
            self.io
                .seek_abs(saved_pos)
                .map_err(|e| Error::io("mkv/demuxer", &e))?;
        }

        if self.header.tracks.is_empty() {
            return Err(Error::not_found("mkv/demuxer", "no supported tracks found"));
        }

        Ok(())
    }

    fn parse_seek_head(&mut self, end_pos: u64, targets: &mut Vec<(u32, u64)>) -> Result<()> {
        while self.io.tell() < end_pos {
            let Some(hdr) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = hdr.end_offset().unwrap_or(end_pos).min(end_pos);
            if hdr.id == SEEK {
                let mut seek_id = 0u32;
                let mut seek_pos = None;
                while self.io.tell() < child_end {
                    let Some(sub) = read_element_header(&mut self.io)? else {
                        break;
                    };
                    match sub.id {
                        SEEK_ID => {
                            seek_id = read_uint(&mut self.io, sub.size)? as u32;
                        }
                        SEEK_POSITION => {
                            seek_pos = Some(read_uint(&mut self.io, sub.size)?);
                        }
                        _ => skip_element(&mut self.io, sub.size)?,
                    }
                }
                if seek_id != 0
                    && let Some(pos) = seek_pos
                {
                    targets.push((seek_id, pos));
                }
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }
        Ok(())
    }

    fn parse_info(&mut self, end_pos: u64, duration_ticks: &mut Option<f64>) -> Result<()> {
        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            match child.id {
                TIMECODE_SCALE => {
                    self.timecode_scale_ns = read_uint(&mut self.io, child.size)?;
                }
                DURATION => {
                    let dur = read_float(&mut self.io, child.size)?;
                    if dur > 0.0 {
                        *duration_ticks = Some(dur);
                    }
                }
                TITLE => {
                    let title = read_string(&mut self.io, child.size)?;
                    if !title.is_empty() {
                        self.header.metadata.set("title", title);
                    }
                }
                MUXING_APP => {
                    let app = read_string(&mut self.io, child.size)?;
                    if !app.is_empty() {
                        self.header.metadata.set("muxing_app", app);
                    }
                }
                WRITING_APP => {
                    let app = read_string(&mut self.io, child.size)?;
                    if !app.is_empty() {
                        self.header.metadata.set("encoder", app);
                    }
                }
                _ => skip_element(&mut self.io, child.size)?,
            }
        }
        Ok(())
    }

    fn parse_tracks(&mut self, end_pos: u64) -> Result<()> {
        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = child.end_offset().unwrap_or(end_pos).min(end_pos);
            if child.id == TRACK_ENTRY {
                self.parse_track_entry(child_end)?;
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }
        Ok(())
    }

    fn parse_track_entry(&mut self, end_pos: u64) -> Result<()> {
        let mut track_number = 0u64;
        let mut track_uid = 0u64;
        let mut track_type = 0u8;
        let mut flag_default = true;
        let mut flag_forced = false;
        let mut flag_hearing = false;
        let mut flag_visual = false;
        let mut flag_commentary = false;
        let mut default_duration_ns = 0u64;
        let mut name: Option<String> = None;
        let mut language: Option<String> = None;
        let mut language_bcp47: Option<String> = None;
        let mut codec_str = String::new();
        let mut codec_private = Vec::new();
        let mut codec_delay_ns = 0u64;
        let mut seek_preroll_ns = 0u64;

        // Video attributes
        let mut pixel_width = 0u32;
        let mut pixel_height = 0u32;
        let mut display_width = 0u32;
        let mut display_height = 0u32;
        let mut display_unit = 0u64;
        let mut crop = CropRect::ZERO;
        let mut color = ColorDesc::UNSPECIFIED;
        let mut bits_per_channel = 0u32;
        let mut chroma_sub_h = 0u32;
        let mut chroma_sub_v = 0u32;
        let mut pending_side_data = Vec::new();

        // Audio attributes
        let mut sample_rate = 8000.0f64;
        let mut output_sample_rate = 0.0f64;
        let mut channels = 1u32;
        let mut bit_depth = 0u32;

        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = child.end_offset().unwrap_or(end_pos).min(end_pos);
            match child.id {
                TRACK_NUMBER => track_number = read_uint(&mut self.io, child.size)?,
                TRACK_UID => track_uid = read_uint(&mut self.io, child.size)?,
                TRACK_TYPE => track_type = read_uint(&mut self.io, child.size)? as u8,
                FLAG_DEFAULT => flag_default = read_uint(&mut self.io, child.size)? != 0,
                FLAG_FORCED => flag_forced = read_uint(&mut self.io, child.size)? != 0,
                FLAG_HEARING_IMPAIRED => flag_hearing = read_uint(&mut self.io, child.size)? != 0,
                FLAG_VISUAL_IMPAIRED => flag_visual = read_uint(&mut self.io, child.size)? != 0,
                FLAG_COMMENTARY => flag_commentary = read_uint(&mut self.io, child.size)? != 0,
                DEFAULT_DURATION => default_duration_ns = read_uint(&mut self.io, child.size)?,
                NAME => name = Some(read_string(&mut self.io, child.size)?),
                LANGUAGE => language = Some(read_string(&mut self.io, child.size)?),
                LANGUAGE_BCP47 => language_bcp47 = Some(read_string(&mut self.io, child.size)?),
                CODEC_ID => codec_str = read_string(&mut self.io, child.size)?,
                CODEC_PRIVATE => codec_private = read_bytes(&mut self.io, child.size)?,
                CODEC_DELAY => codec_delay_ns = read_uint(&mut self.io, child.size)?,
                SEEK_PRE_ROLL => seek_preroll_ns = read_uint(&mut self.io, child.size)?,
                VIDEO => {
                    self.parse_video_track_element(
                        child_end,
                        &mut pixel_width,
                        &mut pixel_height,
                        &mut display_width,
                        &mut display_height,
                        &mut display_unit,
                        &mut crop,
                        &mut color,
                        &mut bits_per_channel,
                        &mut chroma_sub_h,
                        &mut chroma_sub_v,
                        &mut pending_side_data,
                    )?;
                }
                AUDIO => {
                    self.parse_audio_track_element(
                        child_end,
                        &mut sample_rate,
                        &mut output_sample_rate,
                        &mut channels,
                        &mut bit_depth,
                    )?;
                }
                _ => skip_element(&mut self.io, child.size)?,
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }

        if track_number == 0 {
            return Err(Error::invalid_data(
                "mkv/demuxer",
                "TrackEntry missing TrackNumber",
            ));
        }

        let codec_id = codec_id_from_matroska(&codec_str, bit_depth);
        let track_id = TrackId(self.header.tracks.len() as u32);

        let mut disposition = TrackDisposition::empty();
        if flag_default {
            disposition |= TrackDisposition::DEFAULT;
        }
        if flag_forced {
            disposition |= TrackDisposition::FORCED;
        }
        if flag_hearing {
            disposition |= TrackDisposition::HEARING_IMPAIRED;
        }
        if flag_visual {
            disposition |= TrackDisposition::VISUAL_IMPAIRED;
        }
        if flag_commentary {
            disposition |= TrackDisposition::COMMENTARY;
        }

        let params = match track_type {
            TRACK_TYPE_VIDEO => {
                let mut layout = infer_video_layout(bits_per_channel, chroma_sub_h, chroma_sub_v);
                // Refine from extradata when available (avcC, av1C, vpcC)
                refine_video_from_extradata(
                    codec_id,
                    &codec_private,
                    &mut pixel_width,
                    &mut pixel_height,
                    &mut layout,
                    &mut color,
                );

                let sar = if display_unit == 0
                    && display_width > 0
                    && display_height > 0
                    && pixel_width > 0
                    && pixel_height > 0
                {
                    let (sw, sh) = crop.display_size(pixel_width, pixel_height);
                    let w = if sw > 0 { sw } else { pixel_width };
                    let h = if sh > 0 { sh } else { pixel_height };
                    let (n, d, _) = reduce(
                        i64::from(display_width) * i64::from(h),
                        i64::from(display_height) * i64::from(w),
                        i32::MAX as i64,
                    );
                    if n > 0 && d > 0 {
                        Rational::new(n, d)
                    } else {
                        Rational::new(1, 1)
                    }
                } else {
                    Rational::new(1, 1)
                };

                let avg_frame_rate = if default_duration_ns > 0 {
                    let (n, d, _) =
                        reduce(1_000_000_000, default_duration_ns as i64, i32::MAX as i64);
                    Rational::new(n, d)
                } else {
                    Rational::new(0, 1)
                };

                let mut v = VideoStreamParams::new(pixel_width, pixel_height, layout);
                v.color = color;
                v.crop = crop;
                v.sar = sar;
                v.avg_frame_rate = avg_frame_rate;
                v.extradata = codec_private;
                StreamParams::Video(v)
            }
            TRACK_TYPE_AUDIO => {
                let eff_rate = if output_sample_rate > 0.0 {
                    output_sample_rate.round() as u32
                } else {
                    sample_rate.round() as u32
                };
                let eff_rate = eff_rate.max(1);
                let mut channel_layout = ChannelLayout::default_for_count(channels.max(1));
                let mut final_rate = eff_rate;

                // If AAC CodecPrivate is present, refine from AudioSpecificConfig
                if codec_id == CodecId::Aac
                    && let Ok(asc) = wedeo_syntax::aac::parse_audio_specific_config(&codec_private)
                {
                    if asc.sample_rate > 0 {
                        final_rate = asc.sample_rate;
                    }
                    if asc.channel_config > 0 {
                        channel_layout = asc.channel_layout();
                    }
                }

                let sample_fmt = infer_audio_sample_format(codec_id, bit_depth);
                let mut a = AudioStreamParams::new(final_rate, sample_fmt, channel_layout);
                a.bits_per_coded_sample = bit_depth;
                if codec_delay_ns > 0 {
                    a.initial_padding =
                        ((codec_delay_ns as u128 * u128::from(final_rate)) / 1_000_000_000) as u32;
                }
                let _ = seek_preroll_ns;
                a.extradata = codec_private;
                StreamParams::Audio(a)
            }
            TRACK_TYPE_SUBTITLE => {
                let format = match codec_id {
                    CodecId::SubAss => SubtitleFormat::Ass,
                    CodecId::SubWebvtt => SubtitleFormat::WebVtt,
                    CodecId::SubPgs => SubtitleFormat::Pgs,
                    CodecId::SubDvd => SubtitleFormat::Dvd,
                    CodecId::SubDvb => SubtitleFormat::Dvb,
                    CodecId::SubSrt => SubtitleFormat::Srt,
                    _ => SubtitleFormat::PlainText,
                };
                StreamParams::Subtitle(SubtitleStreamParams {
                    format,
                    canvas_width: 0,
                    canvas_height: 0,
                    header_script: codec_private,
                })
            }
            _ => {
                // Unknown track type (e.g. button/control/complex): skip track cleanly
                return Ok(());
            }
        };

        let mut desc = TrackDescriptor::new(track_id, codec_id, self.segment_time_base, params);
        desc.disposition = disposition;
        desc.language = language_bcp47.or(language);
        if let Some(n) = name
            && !n.is_empty()
        {
            desc.metadata.set("title", n);
        }

        self.header.tracks.push(desc);
        self.tracks.push(MkvTrackState {
            track_number,
            track_uid,
            track_id,
            default_duration_ticks: default_duration_ns as i64,
            time_base: self.segment_time_base,
            last_max_pts: None,
            saw_reordering: false,
            pending_side_data,
        });

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_video_track_element(
        &mut self,
        end_pos: u64,
        pixel_width: &mut u32,
        pixel_height: &mut u32,
        display_width: &mut u32,
        display_height: &mut u32,
        display_unit: &mut u64,
        crop: &mut CropRect,
        color: &mut ColorDesc,
        bits_per_channel: &mut u32,
        chroma_sub_h: &mut u32,
        chroma_sub_v: &mut u32,
        side_data: &mut Vec<SideData>,
    ) -> Result<()> {
        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = child.end_offset().unwrap_or(end_pos).min(end_pos);
            match child.id {
                PIXEL_WIDTH => *pixel_width = read_uint(&mut self.io, child.size)? as u32,
                PIXEL_HEIGHT => *pixel_height = read_uint(&mut self.io, child.size)? as u32,
                DISPLAY_WIDTH => *display_width = read_uint(&mut self.io, child.size)? as u32,
                DISPLAY_HEIGHT => *display_height = read_uint(&mut self.io, child.size)? as u32,
                DISPLAY_UNIT => *display_unit = read_uint(&mut self.io, child.size)?,
                PIXEL_CROP_TOP => crop.top = read_uint(&mut self.io, child.size)? as u32,
                PIXEL_CROP_BOTTOM => crop.bottom = read_uint(&mut self.io, child.size)? as u32,
                PIXEL_CROP_LEFT => crop.left = read_uint(&mut self.io, child.size)? as u32,
                PIXEL_CROP_RIGHT => crop.right = read_uint(&mut self.io, child.size)? as u32,
                COLOUR => {
                    self.parse_colour_element(
                        child_end,
                        color,
                        bits_per_channel,
                        chroma_sub_h,
                        chroma_sub_v,
                        side_data,
                    )?;
                }
                _ => skip_element(&mut self.io, child.size)?,
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }
        Ok(())
    }

    fn parse_colour_element(
        &mut self,
        end_pos: u64,
        color: &mut ColorDesc,
        bits_per_channel: &mut u32,
        chroma_sub_h: &mut u32,
        chroma_sub_v: &mut u32,
        side_data: &mut Vec<SideData>,
    ) -> Result<()> {
        let mut siting_h = 0u8;
        let mut siting_v = 0u8;
        let mut max_cll = 0u32;
        let mut max_fall = 0u32;

        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = child.end_offset().unwrap_or(end_pos).min(end_pos);
            match child.id {
                MATRIX_COEFFICIENTS => {
                    let v = read_uint(&mut self.io, child.size)? as u8;
                    color.matrix = MatrixCoefficients::from_cicp(v);
                }
                BITS_PER_CHANNEL => {
                    *bits_per_channel = read_uint(&mut self.io, child.size)? as u32;
                }
                CHROMA_SUBSAMPLING_HORZ => {
                    *chroma_sub_h = read_uint(&mut self.io, child.size)? as u32;
                }
                CHROMA_SUBSAMPLING_VERT => {
                    *chroma_sub_v = read_uint(&mut self.io, child.size)? as u32;
                }
                CHROMA_SITING_HORZ => {
                    siting_h = read_uint(&mut self.io, child.size)? as u8;
                }
                CHROMA_SITING_VERT => {
                    siting_v = read_uint(&mut self.io, child.size)? as u8;
                }
                RANGE => {
                    color.range = match read_uint(&mut self.io, child.size)? {
                        1 => ColorRange::Limited,
                        2 => ColorRange::Full,
                        _ => ColorRange::Unspecified,
                    };
                }
                TRANSFER_CHARACTERISTICS => {
                    let v = read_uint(&mut self.io, child.size)? as u8;
                    color.transfer = TransferCharacteristic::from_cicp(v);
                }
                PRIMARIES => {
                    let v = read_uint(&mut self.io, child.size)? as u8;
                    color.primaries = ColorPrimaries::from_cicp(v);
                }
                MAX_CLL => {
                    max_cll = read_uint(&mut self.io, child.size)? as u32;
                }
                MAX_FALL => {
                    max_fall = read_uint(&mut self.io, child.size)? as u32;
                }
                MASTERING_METADATA => {
                    if let Some(md) = self.parse_mastering_metadata(child_end)? {
                        side_data.push(SideData::MasteringDisplay(md));
                    }
                }
                _ => skip_element(&mut self.io, child.size)?,
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }

        color.chroma_location = match (siting_h, siting_v) {
            (1, 1) => ChromaLocation::TopLeft,
            (1, 2) | (1, 0) => ChromaLocation::Left,
            (2, 1) => ChromaLocation::Top,
            (2, 2) => ChromaLocation::Center,
            _ => ChromaLocation::Unspecified,
        };

        if max_cll > 0 || max_fall > 0 {
            side_data.push(SideData::ContentLightLevel { max_cll, max_fall });
        }

        Ok(())
    }

    fn parse_mastering_metadata(&mut self, end_pos: u64) -> Result<Option<MasteringDisplayMeta>> {
        let mut rx = 0.0f64;
        let mut ry = 0.0f64;
        let mut gx = 0.0f64;
        let mut gy = 0.0f64;
        let mut bx = 0.0f64;
        let mut by = 0.0f64;
        let mut wx = 0.0f64;
        let mut wy = 0.0f64;
        let mut lmax = 0.0f64;
        let mut lmin = 0.0f64;
        let mut any = false;

        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            match child.id {
                PRIMARY_R_CHROMATICITY_X => {
                    rx = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                PRIMARY_R_CHROMATICITY_Y => {
                    ry = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                PRIMARY_G_CHROMATICITY_X => {
                    gx = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                PRIMARY_G_CHROMATICITY_Y => {
                    gy = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                PRIMARY_B_CHROMATICITY_X => {
                    bx = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                PRIMARY_B_CHROMATICITY_Y => {
                    by = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                WHITE_POINT_CHROMATICITY_X => {
                    wx = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                WHITE_POINT_CHROMATICITY_Y => {
                    wy = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                LUMINANCE_MAX => {
                    lmax = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                LUMINANCE_MIN => {
                    lmin = read_float(&mut self.io, child.size)?;
                    any = true;
                }
                _ => skip_element(&mut self.io, child.size)?,
            }
        }

        if !any {
            return Ok(None);
        }

        let chrom = |v: f64| Rational::new((v * 50_000.0).round() as i32, 50_000);
        let lum = |v: f64| Rational::new((v * 10_000.0).round() as i32, 10_000);
        Ok(Some(MasteringDisplayMeta {
            display_primaries: [
                [chrom(rx), chrom(ry)],
                [chrom(gx), chrom(gy)],
                [chrom(bx), chrom(by)],
            ],
            white_point: [chrom(wx), chrom(wy)],
            min_luminance: lum(lmin),
            max_luminance: lum(lmax),
        }))
    }

    fn parse_audio_track_element(
        &mut self,
        end_pos: u64,
        sample_rate: &mut f64,
        output_sample_rate: &mut f64,
        channels: &mut u32,
        bit_depth: &mut u32,
    ) -> Result<()> {
        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            match child.id {
                SAMPLING_FREQUENCY => *sample_rate = read_float(&mut self.io, child.size)?,
                OUTPUT_SAMPLING_FREQUENCY => {
                    *output_sample_rate = read_float(&mut self.io, child.size)?;
                }
                CHANNELS => *channels = read_uint(&mut self.io, child.size)? as u32,
                BIT_DEPTH => *bit_depth = read_uint(&mut self.io, child.size)? as u32,
                _ => skip_element(&mut self.io, child.size)?,
            }
        }
        Ok(())
    }

    fn parse_chapters(&mut self, end_pos: u64) -> Result<()> {
        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = child.end_offset().unwrap_or(end_pos).min(end_pos);
            if child.id == EDITION_ENTRY {
                while self.io.tell() < child_end {
                    let Some(atom) = read_element_header(&mut self.io)? else {
                        break;
                    };
                    let atom_end = atom.end_offset().unwrap_or(child_end).min(child_end);
                    if atom.id == CHAPTER_ATOM {
                        self.parse_chapter_atom(atom_end)?;
                    }
                    if self.io.tell() < atom_end {
                        let rem = atom_end - self.io.tell();
                        skip_element(&mut self.io, rem)?;
                    }
                }
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }
        Ok(())
    }

    fn parse_chapter_atom(&mut self, end_pos: u64) -> Result<()> {
        let mut uid = self.header.chapters.len() as i64 + 1;
        let mut start_ns = 0u64;
        let mut end_ns: Option<u64> = None;
        let mut metadata = Metadata::new();

        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = child.end_offset().unwrap_or(end_pos).min(end_pos);
            match child.id {
                CHAPTER_UID => uid = read_uint(&mut self.io, child.size)? as i64,
                CHAPTER_TIME_START => start_ns = read_uint(&mut self.io, child.size)?,
                CHAPTER_TIME_END => end_ns = Some(read_uint(&mut self.io, child.size)?),
                CHAPTER_DISPLAY => {
                    while self.io.tell() < child_end {
                        let Some(disp) = read_element_header(&mut self.io)? else {
                            break;
                        };
                        match disp.id {
                            CHAP_STRING => {
                                let title = read_string(&mut self.io, disp.size)?;
                                metadata.set("title", title);
                            }
                            CHAP_LANGUAGE => {
                                let lang = read_string(&mut self.io, disp.size)?;
                                metadata.set("language", lang);
                            }
                            _ => skip_element(&mut self.io, disp.size)?,
                        }
                    }
                }
                _ => skip_element(&mut self.io, child.size)?,
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }

        let ns_tb = Rational::new(1, 1_000_000_000);
        self.header.chapters.push(Chapter {
            id: uid,
            start: Timestamp::new(start_ns as i64),
            end: end_ns.map_or(Timestamp::NONE, |e| Timestamp::new(e as i64)),
            time_base: ns_tb,
            metadata,
        });
        Ok(())
    }

    fn parse_attachments(&mut self, end_pos: u64) -> Result<()> {
        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = child.end_offset().unwrap_or(end_pos).min(end_pos);
            if child.id == ATTACHED_FILE {
                let mut name = String::new();
                let mut mime_type = String::new();
                let mut data = Vec::new();
                while self.io.tell() < child_end {
                    let Some(sub) = read_element_header(&mut self.io)? else {
                        break;
                    };
                    match sub.id {
                        FILE_NAME => name = read_string(&mut self.io, sub.size)?,
                        FILE_MIME_TYPE => mime_type = read_string(&mut self.io, sub.size)?,
                        FILE_DATA => data = read_bytes(&mut self.io, sub.size)?,
                        _ => skip_element(&mut self.io, sub.size)?,
                    }
                }
                if !data.is_empty() || !name.is_empty() {
                    self.header.attachments.push(Attachment {
                        name,
                        mime_type,
                        data: Buffer::from_slice(&data),
                    });
                }
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }
        Ok(())
    }

    fn parse_tags(&mut self, end_pos: u64) -> Result<()> {
        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = child.end_offset().unwrap_or(end_pos).min(end_pos);
            if child.id == TAG {
                let mut target_track_uid = 0u64;
                let mut tags = Vec::new();
                while self.io.tell() < child_end {
                    let Some(sub) = read_element_header(&mut self.io)? else {
                        break;
                    };
                    let sub_end = sub.end_offset().unwrap_or(child_end).min(child_end);
                    match sub.id {
                        TARGETS => {
                            while self.io.tell() < sub_end {
                                let Some(t) = read_element_header(&mut self.io)? else {
                                    break;
                                };
                                if t.id == TARGET_TRACK_UID {
                                    target_track_uid = read_uint(&mut self.io, t.size)?;
                                } else {
                                    skip_element(&mut self.io, t.size)?;
                                }
                            }
                        }
                        SIMPLE_TAG => {
                            let mut k = String::new();
                            let mut v = String::new();
                            while self.io.tell() < sub_end {
                                let Some(st) = read_element_header(&mut self.io)? else {
                                    break;
                                };
                                match st.id {
                                    TAG_NAME => k = read_string(&mut self.io, st.size)?,
                                    TAG_STRING => v = read_string(&mut self.io, st.size)?,
                                    _ => skip_element(&mut self.io, st.size)?,
                                }
                            }
                            if !k.is_empty() {
                                tags.push((k, v));
                            }
                        }
                        _ => skip_element(&mut self.io, sub.size)?,
                    }
                    if self.io.tell() < sub_end {
                        let rem = sub_end - self.io.tell();
                        skip_element(&mut self.io, rem)?;
                    }
                }

                if target_track_uid != 0
                    && let Some(pos) = self
                        .tracks
                        .iter()
                        .position(|t| t.track_uid == target_track_uid)
                {
                    for (k, v) in tags {
                        self.header.tracks[pos].metadata.set(k, v);
                    }
                } else {
                    for (k, v) in tags {
                        self.header.metadata.set(k, v);
                    }
                }
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }
        Ok(())
    }

    fn parse_cues(&mut self, end_pos: u64) -> Result<()> {
        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            let child_end = child.end_offset().unwrap_or(end_pos).min(end_pos);
            if child.id == CUE_POINT {
                let mut cue_time: Option<i64> = None;
                let mut track_positions: Vec<(u64, u64)> = Vec::new();
                while self.io.tell() < child_end {
                    let Some(sub) = read_element_header(&mut self.io)? else {
                        break;
                    };
                    let sub_end = sub.end_offset().unwrap_or(child_end).min(child_end);
                    match sub.id {
                        CUE_TIME => {
                            cue_time = Some(read_uint(&mut self.io, sub.size)? as i64);
                        }
                        CUE_TRACK_POSITIONS => {
                            let mut track_num = 0u64;
                            let mut cluster_pos = None;
                            while self.io.tell() < sub_end {
                                let Some(tp) = read_element_header(&mut self.io)? else {
                                    break;
                                };
                                match tp.id {
                                    CUE_TRACK => track_num = read_uint(&mut self.io, tp.size)?,
                                    CUE_CLUSTER_POSITION => {
                                        cluster_pos = Some(read_uint(&mut self.io, tp.size)?);
                                    }
                                    _ => skip_element(&mut self.io, tp.size)?,
                                }
                            }
                            if let Some(cp) = cluster_pos {
                                track_positions.push((track_num, cp));
                            }
                        }
                        _ => skip_element(&mut self.io, sub.size)?,
                    }
                    if self.io.tell() < sub_end {
                        let rem = sub_end - self.io.tell();
                        skip_element(&mut self.io, rem)?;
                    }
                }
                if let Some(ts) = cue_time {
                    for (track_number, cluster_rel_offset) in track_positions {
                        self.cues.push(CueEntry {
                            timestamp: ts,
                            track_number,
                            cluster_rel_offset,
                        });
                    }
                }
            }
            if self.io.tell() < child_end {
                let rem = child_end - self.io.tell();
                skip_element(&mut self.io, rem)?;
            }
        }
        self.cues.sort_by_key(|c| c.timestamp);
        Ok(())
    }

    fn enter_cluster(&mut self, hdr: &crate::ebml::EbmlHeader) {
        self.in_cluster = true;
        self.cluster_end = hdr.end_offset();
    }

    fn parse_and_enqueue_block(
        &mut self,
        block_pos: u64,
        data: &[u8],
        is_simple_block: bool,
        block_duration: Option<i64>,
        has_reference_block: bool,
        discard_padding_ns: Option<i64>,
    ) -> Result<()> {
        if data.len() < 4 {
            return self.record_corruption(
                TrackId(0),
                Timestamp::NONE,
                "Matroska block shorter than 4 bytes",
            );
        }

        let (track_number, vint_len) = match read_vint_data(data) {
            Ok(v) => v,
            Err(_) => {
                return self.record_corruption(
                    TrackId(0),
                    Timestamp::NONE,
                    "invalid track number VINT in block",
                );
            }
        };
        if data.len() < vint_len + 3 {
            return self.record_corruption(
                TrackId(0),
                Timestamp::NONE,
                "truncated Matroska block header",
            );
        }

        let Some(track_idx) = self
            .tracks
            .iter()
            .position(|t| t.track_number == track_number)
        else {
            // Block belongs to an unknown/ignored track; skip without error.
            return Ok(());
        };

        let timecode_delta = i16::from_be_bytes([data[vint_len], data[vint_len + 1]]);
        let flags_byte = data[vint_len + 2];
        let lacing_mode = (flags_byte >> 1) & 0x03;
        let is_keyframe = if is_simple_block {
            (flags_byte & 0x80) != 0
        } else {
            !has_reference_block
        };
        let is_discardable = is_simple_block && (flags_byte & 0x01) != 0;

        let frame_payload = &data[(vint_len + 3)..];
        let ranges = match parse_laced_frames(frame_payload, lacing_mode) {
            Ok(r) => r,
            Err(_) => {
                let tid = self.tracks[track_idx].track_id;
                return self.record_corruption(tid, Timestamp::NONE, "corrupt block lacing");
            }
        };

        let frame_count = ranges.len() as i64;
        let base_pts = self.cluster_timecode + i64::from(timecode_delta);
        let track = &mut self.tracks[track_idx];
        let per_frame_dur = if let Some(bd) = block_duration {
            if frame_count > 0 {
                bd / frame_count
            } else {
                bd
            }
        } else {
            track.default_duration_ticks
        };

        for (i, range) in ranges.into_iter().enumerate() {
            let pts_ticks = base_pts + (i as i64) * per_frame_dur;
            let pts = Timestamp::new(pts_ticks);

            // VFR + B-frame rule (ARCHITECTURE.md §6.1):
            // SimpleBlock/Block stores PTS only. Never fabricate corrupt non-monotonic DTS!
            let dts = match track.last_max_pts {
                None => {
                    track.last_max_pts = Some(pts_ticks);
                    pts
                }
                Some(max_pts) if pts_ticks >= max_pts => {
                    track.last_max_pts = Some(pts_ticks);
                    if track.saw_reordering {
                        Timestamp::NONE
                    } else {
                        pts
                    }
                }
                Some(_) => {
                    track.saw_reordering = true;
                    Timestamp::NONE
                }
            };

            let mut pkt = Packet::new(
                Buffer::from_slice(&frame_payload[range]),
                track.track_id,
                track.time_base,
            );
            pkt.pts = pts;
            pkt.dts = dts;
            pkt.duration = per_frame_dur;
            pkt.pos = Some(block_pos);
            if is_keyframe && i == 0 {
                pkt.flags |= PacketFlags::KEY;
            }
            if is_discardable {
                pkt.flags |= PacketFlags::DISPOSABLE;
            }
            if !track.pending_side_data.is_empty() {
                pkt.side_data.append(&mut track.pending_side_data);
            }
            if let Some(pad_ns) = discard_padding_ns
                && pad_ns > 0
                && let Some(audio_params) = self.header.tracks[track_idx].params.audio()
            {
                let trim_samples = ((pad_ns as u128 * u128::from(audio_params.sample_rate))
                    / 1_000_000_000) as u32;
                if trim_samples > 0 {
                    pkt.side_data.push(SideData::SampleTrim {
                        skip_start: 0,
                        skip_end: trim_samples,
                    });
                }
            }

            self.pending_packets.push_back(pkt);
        }

        Ok(())
    }

    fn parse_block_group(&mut self, group_pos: u64, end_pos: u64) -> Result<()> {
        let mut block_bytes: Option<Vec<u8>> = None;
        let mut block_duration: Option<i64> = None;
        let mut has_reference_block = false;
        let mut discard_padding_ns: Option<i64> = None;

        while self.io.tell() < end_pos {
            let Some(child) = read_element_header(&mut self.io)? else {
                break;
            };
            match child.id {
                BLOCK => {
                    block_bytes = Some(read_bytes(&mut self.io, child.size)?);
                }
                BLOCK_DURATION => {
                    block_duration = Some(read_uint(&mut self.io, child.size)? as i64);
                }
                REFERENCE_BLOCK => {
                    let _ = read_sint(&mut self.io, child.size)?;
                    has_reference_block = true;
                }
                DISCARD_PADDING => {
                    discard_padding_ns = Some(read_sint(&mut self.io, child.size)?);
                }
                _ => skip_element(&mut self.io, child.size)?,
            }
        }

        if let Some(data) = block_bytes {
            self.parse_and_enqueue_block(
                group_pos,
                &data,
                false,
                block_duration,
                has_reference_block,
                discard_padding_ns,
            )?;
        }
        Ok(())
    }
}

impl Demuxer for MkvDemuxer {
    fn header(&self) -> &ContainerHeader {
        &self.header
    }

    fn read_packet(&mut self) -> Result<RecvStatus<Packet>> {
        loop {
            if let Some(pkt) = self.pending_packets.pop_front() {
                return Ok(RecvStatus::Ready(pkt));
            }

            let pos = self.io.tell();
            if self.in_cluster && self.cluster_end.is_some_and(|end| pos >= end) {
                self.in_cluster = false;
                self.cluster_end = None;
            }
            if self.segment_end.is_some_and(|end| pos >= end) {
                return Ok(RecvStatus::Eof);
            }

            let Some(hdr) = read_element_header(&mut self.io)? else {
                return Ok(RecvStatus::Eof);
            };

            match hdr.id {
                CLUSTER => {
                    if self.first_cluster_offset.is_none() {
                        self.first_cluster_offset = Some(hdr.header_offset);
                    }
                    self.enter_cluster(&hdr);
                }
                CLUSTER_TIMECODE => {
                    self.cluster_timecode = read_uint(&mut self.io, hdr.size)? as i64;
                }
                SIMPLE_BLOCK => {
                    let data = read_bytes(&mut self.io, hdr.size)?;
                    self.parse_and_enqueue_block(
                        hdr.header_offset,
                        &data,
                        true,
                        None,
                        false,
                        None,
                    )?;
                }
                BLOCK_GROUP => {
                    if hdr.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data(
                            "mkv/demuxer",
                            "BlockGroup with unknown size",
                        ));
                    }
                    self.parse_block_group(hdr.header_offset, hdr.data_offset + hdr.size)?;
                }
                EBML_VOID | EBML_CRC32 => {
                    skip_element(&mut self.io, hdr.size)?;
                }
                id if is_top_level_segment_element(id) => {
                    self.in_cluster = false;
                    self.cluster_end = None;
                    if hdr.size == SIZE_UNKNOWN {
                        return Ok(RecvStatus::Eof);
                    }
                    if id == CUES && self.cues.is_empty() {
                        self.parse_cues(hdr.data_offset + hdr.size)?;
                    } else {
                        skip_element(&mut self.io, hdr.size)?;
                    }
                }
                _ => {
                    if hdr.size == SIZE_UNKNOWN {
                        return Err(Error::invalid_data(
                            "mkv/demuxer",
                            "unknown element with indeterminate size",
                        ));
                    }
                    skip_element(&mut self.io, hdr.size)?;
                }
            }
        }
    }

    fn seek(&mut self, target: SeekTarget) -> Result<()> {
        if !self.io.is_seekable() {
            return Err(Error::not_seekable(
                "mkv/demuxer",
                "cannot seek non-seekable Matroska stream",
            ));
        }

        let target_ts = target
            .timestamp
            .rescale(target.time_base, self.segment_time_base, Rounding::NearInf)
            .get()
            .unwrap_or(0);

        let target_track_num = target.track.and_then(|tid| {
            self.tracks
                .iter()
                .find(|t| t.track_id == tid)
                .map(|t| t.track_number)
        });

        let mut chosen_offset: Option<u64> = None;
        if !self.cues.is_empty() {
            let matching = self.cues.iter().filter(|c| {
                target_track_num.is_none_or(|tn| c.track_number == 0 || c.track_number == tn)
            });

            if target.backward {
                let mut best: Option<&CueEntry> = None;
                for c in matching {
                    if c.timestamp <= target_ts {
                        best = Some(c);
                    } else if best.is_some() {
                        break;
                    }
                }
                chosen_offset = best
                    .or_else(|| self.cues.first())
                    .map(|c| self.segment_data_start + c.cluster_rel_offset);
            } else {
                chosen_offset = matching
                    .into_iter()
                    .find(|c| c.timestamp >= target_ts)
                    .or_else(|| self.cues.last())
                    .map(|c| self.segment_data_start + c.cluster_rel_offset);
            }
        }

        let seek_pos = chosen_offset.or(self.first_cluster_offset).ok_or_else(|| {
            Error::not_found("mkv/demuxer", "no cluster offset available for seek")
        })?;

        self.io
            .seek_abs(seek_pos)
            .map_err(|e| Error::io("mkv/demuxer", &e))?;
        self.pending_packets.clear();
        self.in_cluster = false;
        self.cluster_end = None;
        for t in &mut self.tracks {
            t.last_max_pts = None;
            t.saw_reordering = false;
        }
        Ok(())
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

fn infer_video_layout(bits_per_channel: u32, chroma_sub_h: u32, chroma_sub_v: u32) -> PixelLayout {
    match (bits_per_channel, chroma_sub_h, chroma_sub_v) {
        (10, 1, 0) => PixelLayout::YUV422P10LE,
        (10, 0, 0) if chroma_sub_h == 0 && chroma_sub_v == 0 && bits_per_channel == 10 => {
            PixelLayout::YUV420P10LE
        }
        (10, _, _) => PixelLayout::YUV420P10LE,
        (12, _, _) => PixelLayout::YUV420P12LE,
        (_, 1, 0) => PixelLayout::YUV422P,
        _ => PixelLayout::YUV420P,
    }
}

fn refine_video_from_extradata(
    codec_id: CodecId,
    extradata: &[u8],
    width: &mut u32,
    height: &mut u32,
    layout: &mut PixelLayout,
    color: &mut ColorDesc,
) {
    if extradata.is_empty() {
        return;
    }
    match codec_id {
        CodecId::H264 => refine_h264(extradata, width, height, layout, color),
        CodecId::Av1 => {
            if let Ok(av1c) = wedeo_syntax::av1::parse_av1c(extradata) {
                for item in wedeo_syntax::av1::iter_obus(&av1c.config_obus) {
                    if let Ok(obu) = item
                        && obu.header.obu_type == wedeo_syntax::av1::ObuType::SequenceHeader
                        && let Ok(seq) = wedeo_syntax::av1::parse_sequence_header(obu.payload)
                    {
                        if *width == 0 {
                            *width = seq.max_frame_width;
                        }
                        if *height == 0 {
                            *height = seq.max_frame_height;
                        }
                        *layout = seq.pixel_layout();
                        if *color == ColorDesc::UNSPECIFIED {
                            *color = seq.color;
                        }
                    }
                }
            }
        }
        CodecId::Vp9 => {
            if let Ok(vpcc) = wedeo_syntax::vp9::parse_vpcc(extradata)
                && *color == ColorDesc::UNSPECIFIED
            {
                *color = vpcc.color_desc();
            }
        }
        _ => {}
    }
}

fn refine_h264(
    extradata: &[u8],
    width: &mut u32,
    height: &mut u32,
    layout: &mut PixelLayout,
    color: &mut ColorDesc,
) {
    if let Ok(avcc) = wedeo_syntax::h264::parse_avcc(extradata)
        && let Some(first_sps) = avcc.sps.first()
        && let Ok(sps) = wedeo_syntax::h264::Sps::parse_nal(first_sps)
    {
        if *width == 0 {
            *width = sps.display_width();
        }
        if *height == 0 {
            *height = sps.display_height();
        }
        *layout = sps.pixel_layout();
        if *color == ColorDesc::UNSPECIFIED {
            *color = sps.vui.color;
        }
    }
}

fn infer_audio_sample_format(codec_id: CodecId, bit_depth: u32) -> SampleFormat {
    match codec_id {
        CodecId::PcmU8 | CodecId::PcmS8 => SampleFormat::U8,
        CodecId::PcmS16le | CodecId::PcmS16be => SampleFormat::S16,
        CodecId::PcmS24le | CodecId::PcmS24be | CodecId::PcmS32le | CodecId::PcmS32be => {
            SampleFormat::S32
        }
        CodecId::PcmF32le | CodecId::PcmF32be => SampleFormat::F32,
        CodecId::PcmF64le | CodecId::PcmF64be => SampleFormat::F64,
        CodecId::Aac | CodecId::Opus | CodecId::Vorbis | CodecId::Mp3 | CodecId::Mp2 => {
            SampleFormat::F32Planar
        }
        _ => match bit_depth {
            8 => SampleFormat::U8,
            24 | 32 => SampleFormat::S32,
            _ => SampleFormat::S16,
        },
    }
}

// ---------------------------------------------------------------------------
// DemuxerFactory
// ---------------------------------------------------------------------------

pub struct MkvDemuxerFactory;

impl DemuxerFactory for MkvDemuxerFactory {
    fn descriptor(&self) -> &DemuxerDescriptor {
        static DESC: DemuxerDescriptor = DemuxerDescriptor {
            name: "matroska",
            long_name: "Matroska / WebM (EBML)",
            extensions: &["mkv", "mk3d", "mka", "mks", "webm"],
            mime_types: &[
                "video/x-matroska",
                "audio/x-matroska",
                "video/webm",
                "audio/webm",
            ],
        };
        &DESC
    }

    fn probe(&self, head: &[u8], ext: Option<&str>) -> u8 {
        if head.len() >= 4 && head[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            let scan_len = head.len().min(256);
            let prefix = &head[..scan_len];
            if prefix.windows(8).any(|w| w == b"matroska")
                || prefix.windows(4).any(|w| w == b"webm")
            {
                return 100;
            }
            return 60;
        }
        if let Some(e) = ext
            && matches!(
                e.to_ascii_lowercase().as_str(),
                "mkv" | "mk3d" | "mka" | "mks" | "webm"
            )
        {
            return 25;
        }
        0
    }

    fn open(&self, source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Box<dyn Demuxer>> {
        Ok(Box::new(MkvDemuxer::open(source, policy)?))
    }
}
