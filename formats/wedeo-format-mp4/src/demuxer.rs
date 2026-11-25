//! ISOBMFF / MP4 / MOV / M4A demuxer implementation (Wedeo).
//!
//! Supports classic (`moov` + `mdat`) and fragmented (`moov` [`mvex`] + `moof`/`mdat`)
//! MP4 files, VFR `stts` + `ctts` v1 signed composition offsets, `elst` edit lists,
//! `colr` (`nclx` + ICC), `pasp` SAR, `tkhd` display matrix, video (`avc1`/`avc3`/
//! `hvc1`/`hev1`/`av01`/`vp09`), and audio (`mp4a`/`alac`/`Opus`/`fLaC`/`ac-3`/PCM).

use std::io;

use wedeo_core::{
    AudioStreamParams, BufByteSource, Buffer, ByteSource, ChannelLayout, Chapter, CodecId,
    ColorDesc, ContainerHeader, Demuxer, DemuxerDescriptor, DemuxerFactory, Error, ErrorPolicy,
    HealthEvent, HealthIssue, Metadata, Packet, PacketFlags, PixelLayout, Rational, RecvStatus,
    Result, SampleFormat, SeekTarget, SideData, StreamHealth, StreamParams, Timestamp,
    TrackDescriptor, TrackDisposition, TrackId, VideoStreamParams, reduce, rescale_q,
};

use crate::boxes::{self, ElstEntry, StsdEntry, Trex};

// ---------------------------------------------------------------------------
// Sample index
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct SampleEntry {
    offset: u64,
    size: u32,
    dts: i64,
    cts_offset: i32,
    duration: u32,
    is_keyframe: bool,
}

fn build_sample_index(
    stts: &[(u32, u32)],
    stsc: &[(u32, u32, u32)],
    fixed_sample_size: u32,
    sample_sizes: &[u32],
    chunk_offsets: &[u64],
    sync_samples: &[u32],
    ctts: &[(u32, i32)],
) -> Vec<SampleEntry> {
    let total_samples: u64 = stts.iter().map(|&(count, _)| u64::from(count)).sum();
    let total_samples = total_samples as usize;
    if total_samples == 0 {
        return Vec::new();
    }

    let mut samples = Vec::with_capacity(total_samples.min(1_048_576));

    // Step 1: Expand stts -> cumulative DTS and duration per sample
    let mut dts: i64 = 0;
    for &(count, delta) in stts {
        for _ in 0..count {
            samples.push(SampleEntry {
                offset: 0,
                size: 0,
                dts,
                cts_offset: 0,
                duration: delta,
                is_keyframe: false,
            });
            dts = dts.saturating_add(i64::from(delta));
        }
    }

    // Step 2: Expand ctts -> signed composition time offsets (v0 & v1)
    if !ctts.is_empty() {
        let mut sample_idx = 0usize;
        for &(count, offset) in ctts {
            for _ in 0..count {
                if sample_idx < samples.len() {
                    samples[sample_idx].cts_offset = offset;
                    sample_idx += 1;
                } else {
                    break;
                }
            }
        }
    }

    // Step 3: Compute file offsets via stsc + stco/co64 + stsz
    let mut sample_idx: usize = 0;
    let mut stsc_cursor: usize = 0;

    for (chunk_idx, &chunk_offset) in chunk_offsets.iter().enumerate() {
        let chunk_1indexed = chunk_idx as u32 + 1;
        while stsc_cursor + 1 < stsc.len() && stsc[stsc_cursor + 1].0 <= chunk_1indexed {
            stsc_cursor += 1;
        }
        let samples_in_chunk = if stsc_cursor < stsc.len() && stsc[stsc_cursor].0 <= chunk_1indexed
        {
            stsc[stsc_cursor].1
        } else {
            0
        };

        let mut offset_in_chunk: u64 = 0;
        for _ in 0..samples_in_chunk {
            if sample_idx >= samples.len() {
                break;
            }
            let size = if fixed_sample_size > 0 {
                fixed_sample_size
            } else if sample_idx < sample_sizes.len() {
                sample_sizes[sample_idx]
            } else {
                0
            };
            samples[sample_idx].offset = chunk_offset.saturating_add(offset_in_chunk);
            samples[sample_idx].size = size;
            offset_in_chunk = offset_in_chunk.saturating_add(u64::from(size));
            sample_idx += 1;
        }
    }

    // Step 4: Mark keyframes from stss (1-indexed). If absent, all samples are keyframes.
    if sync_samples.is_empty() {
        for s in &mut samples {
            s.is_keyframe = true;
        }
    } else {
        for &sync_idx in sync_samples {
            let idx = sync_idx.saturating_sub(1) as usize;
            if idx < samples.len() {
                samples[idx].is_keyframe = true;
            }
        }
    }

    samples
}

// ---------------------------------------------------------------------------
// Temporary track parsing state during `open()`
// ---------------------------------------------------------------------------

struct TrackParseState {
    track_id: u32,
    handler_type: [u8; 4],
    timescale: u32,
    duration: u64,
    tkhd_duration: u64,
    width: u32,
    height: u32,
    display_matrix: Option<[i32; 9]>,
    language: Option<String>,
    elst: Vec<ElstEntry>,
    stsd_entry: Option<StsdEntry>,
    stts: Vec<(u32, u32)>,
    stsc: Vec<(u32, u32, u32)>,
    fixed_sample_size: u32,
    sample_sizes: Vec<u32>,
    chunk_offsets: Vec<u64>,
    sync_samples: Vec<u32>,
    ctts: Vec<(u32, i32)>,
}

impl TrackParseState {
    fn new() -> Self {
        Self {
            track_id: 0,
            handler_type: [0; 4],
            timescale: 0,
            duration: 0,
            tkhd_duration: 0,
            width: 0,
            height: 0,
            display_matrix: None,
            language: None,
            elst: Vec::new(),
            stsd_entry: None,
            stts: Vec::new(),
            stsc: Vec::new(),
            fixed_sample_size: 0,
            sample_sizes: Vec::new(),
            chunk_offsets: Vec::new(),
            sync_samples: Vec::new(),
            ctts: Vec::new(),
        }
    }

    fn is_video(&self) -> bool {
        &self.handler_type == b"vide"
    }

    fn is_audio(&self) -> bool {
        &self.handler_type == b"soun"
    }
}

// ---------------------------------------------------------------------------
// Demuxer state
// ---------------------------------------------------------------------------

struct Mp4TrackState {
    track_id: TrackId,
    mp4_track_id: u32,
    is_video: bool,
    samples: Vec<SampleEntry>,
    current_sample: usize,
    timescale: u32,
    display_matrix: Option<[i32; 9]>,
    sent_display_matrix: bool,
    extra_side_data: Vec<SideData>,
    sent_extra_side_data: bool,
    last_dts: Option<i64>,
}

/// Native ISOBMFF / MP4 / MOV / M4A demuxer owning a `Box<dyn ByteSource>`.
pub struct Mp4Demuxer {
    io: BufByteSource,
    header: ContainerHeader,
    tracks: Vec<Mp4TrackState>,
    health: StreamHealth,
    policy: ErrorPolicy,
    is_fragmented: bool,
    trex_defaults: Vec<Trex>,
    eof: bool,
}

impl Mp4Demuxer {
    /// Open an MP4/MOV/M4A stream from a `Box<dyn ByteSource>`.
    pub fn open(source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Self> {
        let mut io = BufByteSource::new(source);
        let mut movie_timescale = 1000u32;
        let mut movie_duration = 0u64;
        let mut track_states: Vec<TrackParseState> = Vec::new();
        let mut trex_defaults: Vec<Trex> = Vec::new();
        let mut is_fragmented = false;
        let mut found_moov = false;
        let mut parsed_metadata = Metadata::new();
        let mut parsed_chapters: Vec<Chapter> = Vec::new();

        let file_limit = io.size_hint().unwrap_or(u64::MAX);

        while io.tell() < file_limit {
            let box_start = io.tell();
            let Some(hdr) = boxes::try_read_box_header(&mut io)? else {
                break;
            };
            let box_end = if hdr.size == 0 {
                file_limit
            } else {
                box_start.saturating_add(hdr.size).min(file_limit)
            };

            if &hdr.box_type == b"moov" {
                found_moov = true;
                parse_moov(
                    &mut io,
                    box_end,
                    &mut movie_timescale,
                    &mut movie_duration,
                    &mut track_states,
                    &mut trex_defaults,
                    &mut is_fragmented,
                    &mut parsed_metadata,
                    &mut parsed_chapters,
                )?;
                let cur = io.tell();
                if cur < box_end && box_end != u64::MAX {
                    io.skip(box_end - cur)
                        .map_err(|e| Error::io("mp4/demuxer", &e))?;
                }
                break;
            }

            if box_end == u64::MAX {
                break;
            }
            let cur = io.tell();
            if cur < box_end {
                io.skip(box_end - cur)
                    .map_err(|e| Error::io("mp4/demuxer", &e))?;
            }
        }

        if !found_moov {
            return Err(Error::invalid_data("mp4/demuxer", "missing moov box"));
        }

        let mut header = ContainerHeader::new("mp4");
        let mut tracks = Vec::new();

        for (idx, tps) in track_states.iter().enumerate() {
            if !tps.is_video() && !tps.is_audio() {
                continue;
            }
            let Some(stsd) = &tps.stsd_entry else {
                continue;
            };

            let timescale = tps.timescale.max(1);
            let time_base = Rational::new(1, timescale as i32);
            let samples = build_sample_index(
                &tps.stts,
                &tps.stsc,
                tps.fixed_sample_size,
                &tps.sample_sizes,
                &tps.chunk_offsets,
                &tps.sync_samples,
                &tps.ctts,
            );
            let nb_frames = if samples.is_empty() {
                None
            } else {
                Some(samples.len() as u64)
            };

            let track_id = TrackId(tracks.len() as u32);
            let mp4_track_id = if tps.track_id != 0 {
                tps.track_id
            } else {
                (idx + 1) as u32
            };

            let (codec_id, params) = if tps.is_video() {
                let Some(codec_id) = map_video_codec(&stsd.fourcc) else {
                    continue;
                };
                let mut vp = VideoStreamParams::new(
                    tps.width.max(u32::from(stsd.width)),
                    tps.height.max(u32::from(stsd.height)),
                    PixelLayout::YUV420P,
                );
                vp.color = stsd.color.unwrap_or(ColorDesc::UNSPECIFIED);
                vp.sar = stsd.sar.unwrap_or_else(|| Rational::new(1, 1));
                vp.display_matrix = tps.display_matrix;
                vp.bit_rate = stsd.avg_bitrate;
                vp.extradata = stsd.extradata.clone();
                if let Some(n) = nb_frames
                    && tps.duration > 0
                {
                    let (num, den, _) = reduce(
                        (n as i64).saturating_mul(i64::from(timescale)),
                        tps.duration as i64,
                        i32::MAX as i64,
                    );
                    if den > 0 {
                        vp.avg_frame_rate = Rational::new(num, den);
                    }
                }
                (codec_id, StreamParams::Video(vp))
            } else {
                let Some((codec_id, sample_format)) = map_audio_codec(stsd) else {
                    continue;
                };
                let ch = stsd.channel_count.max(1);
                let layout = match ch {
                    1 => ChannelLayout::mono(),
                    2 => ChannelLayout::stereo(),
                    n => ChannelLayout::unspec(n as u32),
                };
                let mut ap = AudioStreamParams::new(stsd.sample_rate, sample_format, layout);
                ap.bits_per_coded_sample = u32::from(stsd.sample_size);
                ap.bit_rate = stsd.avg_bitrate;
                ap.initial_padding = stsd.initial_padding;
                ap.extradata = stsd.extradata.clone();
                (codec_id, StreamParams::Audio(ap))
            };

            let mut desc = TrackDescriptor::new(track_id, codec_id, time_base, params);
            desc.codec_tag = u32::from_be_bytes(stsd.fourcc);
            desc.duration = if tps.duration > 0 {
                tps.duration as i64
            } else if movie_timescale > 0 && tps.tkhd_duration > 0 {
                rescale_q(
                    tps.tkhd_duration as i64,
                    Rational::new(1, movie_timescale as i32),
                    time_base,
                )
            } else {
                0
            };
            desc.nb_frames = nb_frames;
            desc.language = tps.language.clone();
            if tracks.is_empty() {
                desc.disposition = TrackDisposition::DEFAULT;
            }
            desc.edit_list = tps
                .elst
                .iter()
                .map(|e| e.to_edit_segment(timescale, movie_timescale, tps.duration))
                .collect();

            header.tracks.push(desc);
            tracks.push(Mp4TrackState {
                track_id,
                mp4_track_id,
                is_video: tps.is_video(),
                samples,
                current_sample: 0,
                timescale,
                display_matrix: tps.display_matrix,
                sent_display_matrix: false,
                extra_side_data: stsd.side_data.clone(),
                sent_extra_side_data: false,
                last_dts: None,
            });
        }

        if tracks.is_empty() {
            return Err(Error::not_found(
                "mp4/demuxer",
                "no supported video or audio tracks found",
            ));
        }

        if movie_timescale > 0 && movie_duration > 0 {
            header.duration = Some(rescale_q(
                movie_duration as i64,
                Rational::new(1, movie_timescale as i32),
                header.time_base,
            ));
        }
        header.start_time = Timestamp::new(0);
        header.metadata = parsed_metadata;
        if !parsed_chapters.is_empty() {
            let movie_end_100ns = if movie_timescale > 0 && movie_duration > 0 {
                rescale_q(
                    movie_duration as i64,
                    Rational::new(1, movie_timescale as i32),
                    Rational::new(1, 10_000_000),
                )
            } else {
                0
            };
            for idx in 0..parsed_chapters.len() {
                if idx + 1 < parsed_chapters.len() {
                    parsed_chapters[idx].end = parsed_chapters[idx + 1].start;
                } else if movie_end_100ns >= parsed_chapters[idx].start.get().unwrap_or(0) {
                    parsed_chapters[idx].end = Timestamp::new(movie_end_100ns);
                } else {
                    parsed_chapters[idx].end = parsed_chapters[idx].start;
                }
            }
            header.chapters = parsed_chapters;
        }

        Ok(Self {
            io,
            header,
            tracks,
            health: StreamHealth::new(),
            policy,
            is_fragmented,
            trex_defaults,
            eof: false,
        })
    }

    fn record_or_fail(
        &mut self,
        track: TrackId,
        pts: Timestamp,
        byte_offset: Option<u64>,
        kind: HealthIssue,
        detail: &'static str,
    ) -> Result<()> {
        if self.policy == ErrorPolicy::Strict {
            return Err(Error::corrupt("mp4/demuxer", detail));
        }
        self.health.record(HealthEvent {
            track,
            pts,
            byte_offset,
            kind,
            detail,
        });
        Ok(())
    }

    /// Scan forward for the next `moof` box and append its samples into `self.tracks`.
    /// Returns `Ok(true)` if at least one sample was loaded, `Ok(false)` on clean EOF.
    fn load_next_fragment(&mut self) -> Result<bool> {
        let file_limit = self.io.size_hint().unwrap_or(u64::MAX);
        while self.io.tell() < file_limit {
            let moof_offset = self.io.tell();
            let Some(hdr) = boxes::try_read_box_header(&mut self.io)? else {
                return Ok(false);
            };
            let box_end = if hdr.size == 0 {
                file_limit
            } else {
                moof_offset.saturating_add(hdr.size).min(file_limit)
            };

            if &hdr.box_type == b"moof" {
                let loaded = self.parse_moof(moof_offset, box_end)?;
                let cur = self.io.tell();
                if cur < box_end && box_end != u64::MAX {
                    self.io
                        .skip(box_end - cur)
                        .map_err(|e| Error::io("mp4/moof", &e))?;
                }
                if loaded > 0 {
                    return Ok(true);
                }
                continue;
            }

            if box_end == u64::MAX {
                return Ok(false);
            }
            let cur = self.io.tell();
            if cur < box_end {
                self.io
                    .skip(box_end - cur)
                    .map_err(|e| Error::io("mp4/demuxer", &e))?;
            }
        }
        Ok(false)
    }

    fn parse_moof(&mut self, moof_offset: u64, moof_end: u64) -> Result<usize> {
        let mut total_loaded = 0usize;
        while self.io.tell() + 8 <= moof_end {
            let sub_start = self.io.tell();
            let hdr = boxes::read_box_header(&mut self.io)?;
            let sub_end = if hdr.size == 0 {
                moof_end
            } else {
                sub_start.saturating_add(hdr.size).min(moof_end)
            };

            if &hdr.box_type == b"traf" {
                total_loaded += self.parse_traf(moof_offset, sub_end)?;
            }

            let cur = self.io.tell();
            if cur < sub_end {
                self.io
                    .skip(sub_end - cur)
                    .map_err(|e| Error::io("mp4/moof", &e))?;
            }
        }
        Ok(total_loaded)
    }

    fn parse_traf(&mut self, moof_offset: u64, traf_end: u64) -> Result<usize> {
        let mut tfhd: Option<boxes::Tfhd> = None;
        let mut tfdt_val: Option<u64> = None;
        let mut truns: Vec<boxes::Trun> = Vec::new();

        while self.io.tell() + 8 <= traf_end {
            let sub_start = self.io.tell();
            let hdr = boxes::read_box_header(&mut self.io)?;
            let sub_end = if hdr.size == 0 {
                traf_end
            } else {
                sub_start.saturating_add(hdr.size).min(traf_end)
            };

            match &hdr.box_type {
                b"tfhd" => tfhd = Some(boxes::parse_tfhd(&mut self.io)?),
                b"tfdt" => tfdt_val = Some(boxes::parse_tfdt(&mut self.io)?),
                b"trun" => truns.push(boxes::parse_trun(&mut self.io)?),
                _ => {}
            }

            let cur = self.io.tell();
            if cur < sub_end {
                self.io
                    .skip(sub_end - cur)
                    .map_err(|e| Error::io("mp4/traf", &e))?;
            }
        }

        let Some(tfhd) = tfhd else {
            return Ok(0);
        };
        let Some(track_idx) = self
            .tracks
            .iter()
            .position(|t| t.mp4_track_id == tfhd.track_id)
        else {
            return Ok(0);
        };

        let trex = self
            .trex_defaults
            .iter()
            .find(|t| t.track_id == tfhd.track_id)
            .copied();

        let default_dur = tfhd
            .default_sample_duration
            .or_else(|| trex.map(|t| t.default_sample_duration))
            .unwrap_or(0);
        let default_size = tfhd
            .default_sample_size
            .or_else(|| trex.map(|t| t.default_sample_size))
            .unwrap_or(0);
        let default_flags = tfhd
            .default_sample_flags
            .or_else(|| trex.map(|t| t.default_sample_flags))
            .unwrap_or(0);

        let base_offset = tfhd.base_data_offset.unwrap_or(moof_offset);

        let track = &mut self.tracks[track_idx];
        let mut dts = tfdt_val.map(|v| v as i64).unwrap_or_else(|| {
            track
                .samples
                .last()
                .map_or(0, |s| s.dts.saturating_add(i64::from(s.duration)))
        });

        let mut loaded = 0usize;
        for trun in truns {
            let mut offset = match trun.data_offset {
                Some(doff) => base_offset.saturating_add_signed(i64::from(doff)),
                None => base_offset,
            };
            for (i, sample) in trun.samples.iter().enumerate() {
                let dur = sample.duration.unwrap_or(default_dur);
                let size = sample.size.unwrap_or(default_size);
                let flags = if i == 0 {
                    trun.first_sample_flags
                        .or(sample.flags)
                        .unwrap_or(default_flags)
                } else {
                    sample.flags.unwrap_or(default_flags)
                };
                // ISOBMFF sample_depends_on == 2 or sample_is_non_sync_sample (bit 16) == 0
                let is_non_sync = (flags & 0x0001_0000) != 0;
                let is_keyframe = !is_non_sync || !track.is_video;

                track.samples.push(SampleEntry {
                    offset,
                    size,
                    dts,
                    cts_offset: sample.cts_offset,
                    duration: dur,
                    is_keyframe,
                });
                offset = offset.saturating_add(u64::from(size));
                dts = dts.saturating_add(i64::from(dur));
                loaded += 1;
            }
        }
        Ok(loaded)
    }
}

impl Demuxer for Mp4Demuxer {
    fn header(&self) -> &ContainerHeader {
        &self.header
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }

    fn read_packet(&mut self) -> Result<RecvStatus<Packet>> {
        if self.eof {
            return Ok(RecvStatus::Eof);
        }

        let seekable = self.io.is_seekable();
        loop {
            let mut best_track: Option<usize> = None;
            let mut best_key: i64 = i64::MAX;

            for (i, track) in self.tracks.iter().enumerate() {
                if track.current_sample >= track.samples.len() {
                    continue;
                }
                let sample = &track.samples[track.current_sample];
                let key = if seekable {
                    rescale_q(
                        sample.dts,
                        Rational::new(1, track.timescale.max(1) as i32),
                        wedeo_core::TIME_BASE_Q,
                    )
                } else {
                    sample.offset as i64
                };
                if key < best_key {
                    best_key = key;
                    best_track = Some(i);
                }
            }

            let Some(track_idx) = best_track else {
                if self.is_fragmented && self.load_next_fragment()? {
                    continue;
                }
                self.eof = true;
                return Ok(RecvStatus::Eof);
            };

            let sample = self.tracks[track_idx].samples[self.tracks[track_idx].current_sample];
            let track_id = self.tracks[track_idx].track_id;
            let timescale = self.tracks[track_idx].timescale.max(1);
            let last_dts = self.tracks[track_idx].last_dts;
            let pts_val = sample.dts.saturating_add(i64::from(sample.cts_offset));

            if let Some(prev) = last_dts
                && sample.dts < prev
            {
                self.record_or_fail(
                    track_id,
                    Timestamp::new(pts_val),
                    Some(sample.offset),
                    HealthIssue::NonMonotonicDts,
                    "non-monotonic DTS in MP4 sample table",
                )?;
            }

            // Avoid redundant seek() calls when reading sequential samples.
            if self.io.tell() != sample.offset {
                if !self.io.is_seekable() && sample.offset > self.io.tell() {
                    self.io
                        .skip(sample.offset - self.io.tell())
                        .map_err(|e| Error::io("mp4/demuxer", &e))?;
                } else {
                    self.io
                        .seek_abs(sample.offset)
                        .map_err(|e| Error::io("mp4/demuxer", &e))?;
                }
            }

            let mut data = vec![0u8; sample.size as usize];
            let mut corrupt = false;
            if let Err(e) = self.io.read_exact_bytes(&mut data) {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    self.record_or_fail(
                        track_id,
                        Timestamp::new(pts_val),
                        Some(sample.offset),
                        HealthIssue::TruncatedPacket,
                        "truncated MP4 sample at EOF",
                    )?;
                    corrupt = true;
                    data.clear();
                } else {
                    return Err(Error::io("mp4/demuxer", &e));
                }
            }

            let track = &mut self.tracks[track_idx];
            track.current_sample += 1;
            track.last_dts = Some(sample.dts);

            let mut pkt = Packet::new(
                Buffer::from_slice(&data),
                track.track_id,
                Rational::new(1, timescale as i32),
            );
            pkt.dts = Timestamp::new(sample.dts);
            pkt.pts = Timestamp::new(pts_val);
            pkt.duration = i64::from(sample.duration);
            pkt.pos = Some(sample.offset);
            if sample.is_keyframe {
                pkt.flags |= PacketFlags::KEY;
            }
            if corrupt {
                pkt.flags |= PacketFlags::CORRUPT;
            }

            if !track.sent_display_matrix {
                track.sent_display_matrix = true;
                if let Some(m) = track.display_matrix {
                    pkt.side_data.push(SideData::DisplayMatrix(m));
                }
            }
            if !track.sent_extra_side_data {
                track.sent_extra_side_data = true;
                if !track.extra_side_data.is_empty() {
                    pkt.side_data.extend(track.extra_side_data.iter().cloned());
                }
            }

            return Ok(RecvStatus::Ready(pkt));
        }
    }

    fn seek(&mut self, target: SeekTarget) -> Result<()> {
        if !self.io.is_seekable() {
            return Err(Error::not_seekable(
                "mp4/demuxer",
                "cannot seek a non-seekable byte source",
            ));
        }

        let target_track_idx = match target.track {
            Some(tid) => self
                .tracks
                .iter()
                .position(|t| t.track_id == tid)
                .ok_or_else(|| Error::not_found("mp4/demuxer", "seek target track not found"))?,
            None => 0,
        };

        let target_ts = target.timestamp.get().unwrap_or(0);
        let track = &mut self.tracks[target_track_idx];
        let track_tb = Rational::new(1, track.timescale.max(1) as i32);
        let ts_in_track = if target.time_base.is_valid() {
            rescale_q(target_ts, target.time_base, track_tb)
        } else {
            target_ts
        };

        let keyframe_idx = if track.samples.is_empty() {
            0
        } else {
            let pos = match track.samples.binary_search_by_key(&ts_in_track, |s| s.dts) {
                Ok(i) => i,
                Err(i) => {
                    if target.backward {
                        i.saturating_sub(1)
                    } else {
                        i.min(track.samples.len() - 1)
                    }
                }
            };
            let mut kf = pos;
            if target.backward {
                while kf > 0 && !track.samples[kf].is_keyframe {
                    kf -= 1;
                }
            } else {
                while kf + 1 < track.samples.len() && !track.samples[kf].is_keyframe {
                    kf += 1;
                }
                if !track.samples[kf].is_keyframe {
                    kf = pos;
                    while kf > 0 && !track.samples[kf].is_keyframe {
                        kf -= 1;
                    }
                }
            }
            kf
        };

        track.current_sample = keyframe_idx;
        track.last_dts = None;

        let anchor_dts = track
            .samples
            .get(keyframe_idx)
            .map_or(ts_in_track, |s| s.dts);

        for (i, other) in self.tracks.iter_mut().enumerate() {
            if i == target_track_idx {
                continue;
            }
            let other_tb = Rational::new(1, other.timescale.max(1) as i32);
            let other_dts = rescale_q(anchor_dts, track_tb, other_tb);
            let idx = match other.samples.binary_search_by_key(&other_dts, |s| s.dts) {
                Ok(k) => k,
                Err(k) => k.saturating_sub(1),
            };
            other.current_sample = idx;
            other.last_dts = None;
        }

        self.eof = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Codec mapping helpers
// ---------------------------------------------------------------------------

fn map_video_codec(fourcc: &[u8; 4]) -> Option<CodecId> {
    match fourcc {
        b"avc1" | b"avc3" => Some(CodecId::H264),
        b"hvc1" | b"hev1" => Some(CodecId::Hevc),
        b"av01" => Some(CodecId::Av1),
        b"vp09" => Some(CodecId::Vp9),
        b"vp08" => Some(CodecId::Vp8),
        b"mp4v" => Some(CodecId::Mpeg4),
        b"apch" | b"apcn" | b"apcs" | b"apco" | b"ap4h" => Some(CodecId::Prores),
        _ => None,
    }
}

fn map_audio_codec(stsd: &StsdEntry) -> Option<(CodecId, SampleFormat)> {
    match &stsd.fourcc {
        b"mp4a" => match stsd.esds_object_type {
            Some(0x69 | 0x6B) => Some((CodecId::Mp3, SampleFormat::F32Planar)),
            Some(0xA5) => Some((CodecId::Ac3, SampleFormat::F32Planar)),
            _ => Some((CodecId::Aac, SampleFormat::F32Planar)),
        },
        b"alac" => Some((CodecId::Alac, SampleFormat::S32Planar)),
        b"Opus" => Some((CodecId::Opus, SampleFormat::F32)),
        b"fLaC" => Some((CodecId::Flac, SampleFormat::S32)),
        b"ac-3" => Some((CodecId::Ac3, SampleFormat::F32Planar)),
        b"sowt" => match stsd.sample_size {
            8 => Some((CodecId::PcmS8, SampleFormat::U8)),
            24 => Some((CodecId::PcmS24le, SampleFormat::S32)),
            32 => Some((CodecId::PcmS32le, SampleFormat::S32)),
            _ => Some((CodecId::PcmS16le, SampleFormat::S16)),
        },
        b"twos" => match stsd.sample_size {
            8 => Some((CodecId::PcmS8, SampleFormat::U8)),
            24 => Some((CodecId::PcmS24be, SampleFormat::S32)),
            32 => Some((CodecId::PcmS32be, SampleFormat::S32)),
            _ => Some((CodecId::PcmS16be, SampleFormat::S16)),
        },
        b"in24" => {
            if stsd.pcm_little_endian == Some(true) {
                Some((CodecId::PcmS24le, SampleFormat::S32))
            } else {
                Some((CodecId::PcmS24be, SampleFormat::S32))
            }
        }
        b"in32" => {
            if stsd.pcm_little_endian == Some(true) {
                Some((CodecId::PcmS32le, SampleFormat::S32))
            } else {
                Some((CodecId::PcmS32be, SampleFormat::S32))
            }
        }
        b"fl32" => {
            if stsd.pcm_little_endian == Some(true) {
                Some((CodecId::PcmF32le, SampleFormat::F32))
            } else {
                Some((CodecId::PcmF32be, SampleFormat::F32))
            }
        }
        b"fl64" => {
            if stsd.pcm_little_endian == Some(true) {
                Some((CodecId::PcmF64le, SampleFormat::F64))
            } else {
                Some((CodecId::PcmF64be, SampleFormat::F64))
            }
        }
        b"lpcm" | b"ipcm" => {
            let le = stsd.pcm_little_endian.unwrap_or(true);
            match (stsd.sample_size, le) {
                (8, _) => Some((CodecId::PcmS8, SampleFormat::U8)),
                (24, true) => Some((CodecId::PcmS24le, SampleFormat::S32)),
                (24, false) => Some((CodecId::PcmS24be, SampleFormat::S32)),
                (32, true) => Some((CodecId::PcmS32le, SampleFormat::S32)),
                (32, false) => Some((CodecId::PcmS32be, SampleFormat::S32)),
                (_, true) => Some((CodecId::PcmS16le, SampleFormat::S16)),
                (_, false) => Some((CodecId::PcmS16be, SampleFormat::S16)),
            }
        }
        b"fpcm" => {
            let le = stsd.pcm_little_endian.unwrap_or(true);
            match (stsd.sample_size, le) {
                (64, true) => Some((CodecId::PcmF64le, SampleFormat::F64)),
                (64, false) => Some((CodecId::PcmF64be, SampleFormat::F64)),
                (_, true) => Some((CodecId::PcmF32le, SampleFormat::F32)),
                (_, false) => Some((CodecId::PcmF32be, SampleFormat::F32)),
            }
        }
        b"raw " | b"NONE" => Some((CodecId::PcmU8, SampleFormat::U8)),
        b"ulaw" => Some((CodecId::PcmMulaw, SampleFormat::S16)),
        b"alaw" => Some((CodecId::PcmAlaw, SampleFormat::S16)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// `moov` box recursive parser
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn parse_moov(
    io: &mut BufByteSource,
    end_pos: u64,
    movie_timescale: &mut u32,
    movie_duration: &mut u64,
    tracks: &mut Vec<TrackParseState>,
    trex_defaults: &mut Vec<Trex>,
    is_fragmented: &mut bool,
    metadata: &mut Metadata,
    chapters: &mut Vec<Chapter>,
) -> Result<()> {
    while io.tell() + 8 <= end_pos {
        let box_start = io.tell();
        let header = boxes::read_box_header(io)?;
        let box_end = if header.size == 0 {
            end_pos
        } else {
            box_start.saturating_add(header.size).min(end_pos)
        };

        match &header.box_type {
            b"mvhd" => {
                let mvhd = boxes::parse_mvhd(io)?;
                *movie_timescale = mvhd.timescale;
                *movie_duration = mvhd.duration;
            }
            b"trak" => {
                let mut tps = TrackParseState::new();
                parse_trak(io, box_end, &mut tps)?;
                tracks.push(tps);
            }
            b"mvex" => {
                *is_fragmented = true;
                parse_mvex(io, box_end, trex_defaults)?;
            }
            b"udta" => {
                parse_udta(io, box_end, metadata, chapters)?;
            }
            _ => {}
        }

        let current = io.tell();
        if current < box_end {
            io.skip(box_end - current)
                .map_err(|e| Error::io("mp4/moov", &e))?;
        }
    }
    Ok(())
}

fn parse_udta(
    io: &mut BufByteSource,
    end_pos: u64,
    metadata: &mut Metadata,
    chapters: &mut Vec<Chapter>,
) -> Result<()> {
    while io.tell() + 8 <= end_pos {
        let box_start = io.tell();
        let header = boxes::read_box_header(io)?;
        let box_end = if header.size == 0 {
            end_pos
        } else {
            box_start.saturating_add(header.size).min(end_pos)
        };

        match &header.box_type {
            b"meta" => {
                let payload_len = box_end.saturating_sub(io.tell()) as usize;
                if (5..=4 * 1024 * 1024).contains(&payload_len) {
                    let mut buf = vec![0u8; payload_len];
                    io.read_exact_bytes(&mut buf)
                        .map_err(|e| Error::io("mp4/meta", &e))?;
                    parse_meta_bytes(&buf, metadata);
                }
            }
            b"chpl" => {
                let payload_len = box_end.saturating_sub(io.tell()) as usize;
                if (5..=1024 * 1024).contains(&payload_len) {
                    let mut buf = vec![0u8; payload_len];
                    io.read_exact_bytes(&mut buf)
                        .map_err(|e| Error::io("mp4/chpl", &e))?;
                    parse_chpl_bytes(&buf, chapters);
                }
            }
            _ => {}
        }

        let current = io.tell();
        if current < box_end {
            io.skip(box_end - current)
                .map_err(|e| Error::io("mp4/udta", &e))?;
        }
    }
    Ok(())
}

fn parse_meta_bytes(buf: &[u8], metadata: &mut Metadata) {
    // ISOBMFF `meta` is a full-box (4-byte version+flags); QuickTime omits those 4 bytes.
    let mut off = if buf.len() >= 12 && (&buf[8..12] == b"hdlr" || &buf[8..12] == b"ilst") {
        4
    } else if buf.len() >= 8 && (&buf[4..8] == b"hdlr" || &buf[4..8] == b"ilst") {
        0
    } else {
        4
    };

    while off + 8 <= buf.len() {
        let size =
            u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) as usize;
        let btype = &buf[off + 4..off + 8];
        if size < 8 || off + size > buf.len() {
            break;
        }
        if btype == b"ilst" {
            parse_ilst_bytes(&buf[off + 8..off + size], metadata);
        }
        off += size;
    }
}

fn parse_ilst_bytes(buf: &[u8], metadata: &mut Metadata) {
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let size =
            u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) as usize;
        let btype = &buf[off + 4..off + 8];
        if size < 8 || off + size > buf.len() {
            break;
        }
        let key: Option<&str> = match btype {
            b"\xA9nam" => Some("title"),
            b"\xA9ART" => Some("artist"),
            b"\xA9alb" => Some("album"),
            b"aART" => Some("album_artist"),
            b"\xA9day" => Some("date"),
            b"\xA9gen" => Some("genre"),
            b"\xA9cmt" => Some("comment"),
            b"desc" => Some("description"),
            b"ldes" => Some("synopsis"),
            b"\xA9too" => Some("encoder"),
            b"\xA9wrt" => Some("composer"),
            b"\xA9grp" => Some("grouping"),
            b"cprt" => Some("copyright"),
            b"\xA9lyr" => Some("lyrics"),
            _ => None,
        };
        if let Some(tag_key) = key {
            let item = &buf[off + 8..off + size];
            let mut ioff = 0usize;
            while ioff + 8 <= item.len() {
                let dsize = u32::from_be_bytes([
                    item[ioff],
                    item[ioff + 1],
                    item[ioff + 2],
                    item[ioff + 3],
                ]) as usize;
                let dtype = &item[ioff + 4..ioff + 8];
                if dsize < 8 || ioff + dsize > item.len() {
                    break;
                }
                if dtype == b"data" && dsize >= 16 {
                    let data_type = u32::from_be_bytes([
                        item[ioff + 8],
                        item[ioff + 9],
                        item[ioff + 10],
                        item[ioff + 11],
                    ]);
                    // Type 1 = UTF-8 text.
                    if data_type == 1
                        && let Ok(val) = std::str::from_utf8(&item[ioff + 16..ioff + dsize])
                    {
                        metadata.set(tag_key, val);
                    }
                }
                ioff += dsize;
            }
        }
        off += size;
    }
}

fn parse_chpl_bytes(buf: &[u8], chapters: &mut Vec<Chapter>) {
    if buf.len() < 5 {
        return;
    }
    let version = buf[0];
    let (count, mut off) = if version == 1 {
        if buf.len() < 9 {
            return;
        }
        let n = u32::from_be_bytes([buf[5], buf[6], buf[7], buf[8]]) as usize;
        (n, 9usize)
    } else {
        (buf[4] as usize, 5usize)
    };
    let chpl_tb = Rational::new(1, 10_000_000);
    for idx in 0..count {
        if off + 9 > buf.len() {
            break;
        }
        let start_100ns = u64::from_be_bytes([
            buf[off],
            buf[off + 1],
            buf[off + 2],
            buf[off + 3],
            buf[off + 4],
            buf[off + 5],
            buf[off + 6],
            buf[off + 7],
        ]);
        let title_len = buf[off + 8] as usize;
        off += 9;
        if off + title_len > buf.len() {
            break;
        }
        let title = String::from_utf8_lossy(&buf[off..off + title_len]).into_owned();
        off += title_len;
        let mut meta = Metadata::new();
        if !title.is_empty() {
            meta.set("title", title);
        }
        let ts = Timestamp::new(start_100ns.min(i64::MAX as u64) as i64);
        chapters.push(Chapter {
            id: (idx + 1) as i64,
            time_base: chpl_tb,
            start: ts,
            end: ts,
            metadata: meta,
        });
    }
}

fn parse_mvex(io: &mut BufByteSource, end_pos: u64, trex_defaults: &mut Vec<Trex>) -> Result<()> {
    while io.tell() + 8 <= end_pos {
        let box_start = io.tell();
        let header = boxes::read_box_header(io)?;
        let box_end = if header.size == 0 {
            end_pos
        } else {
            box_start.saturating_add(header.size).min(end_pos)
        };

        if &header.box_type == b"trex" {
            trex_defaults.push(boxes::parse_trex(io)?);
        }

        let current = io.tell();
        if current < box_end {
            io.skip(box_end - current)
                .map_err(|e| Error::io("mp4/mvex", &e))?;
        }
    }
    Ok(())
}

fn parse_trak(io: &mut BufByteSource, end_pos: u64, tps: &mut TrackParseState) -> Result<()> {
    while io.tell() + 8 <= end_pos {
        let box_start = io.tell();
        let header = boxes::read_box_header(io)?;
        let box_end = if header.size == 0 {
            end_pos
        } else {
            box_start.saturating_add(header.size).min(end_pos)
        };

        match &header.box_type {
            b"tkhd" => {
                let tkhd = boxes::parse_tkhd(io)?;
                tps.track_id = tkhd.track_id;
                tps.tkhd_duration = tkhd.duration;
                tps.width = tkhd.width;
                tps.height = tkhd.height;
                tps.display_matrix = tkhd.display_matrix;
            }
            b"edts" => {
                parse_edts(io, box_end, tps)?;
            }
            b"mdia" => {
                parse_mdia(io, box_end, tps)?;
            }
            _ => {}
        }

        let current = io.tell();
        if current < box_end {
            io.skip(box_end - current)
                .map_err(|e| Error::io("mp4/trak", &e))?;
        }
    }
    Ok(())
}

fn parse_edts(io: &mut BufByteSource, end_pos: u64, tps: &mut TrackParseState) -> Result<()> {
    while io.tell() + 8 <= end_pos {
        let box_start = io.tell();
        let header = boxes::read_box_header(io)?;
        let box_end = if header.size == 0 {
            end_pos
        } else {
            box_start.saturating_add(header.size).min(end_pos)
        };

        if &header.box_type == b"elst" {
            tps.elst = boxes::parse_elst(io)?;
        }

        let current = io.tell();
        if current < box_end {
            io.skip(box_end - current)
                .map_err(|e| Error::io("mp4/edts", &e))?;
        }
    }
    Ok(())
}

fn parse_mdia(io: &mut BufByteSource, end_pos: u64, tps: &mut TrackParseState) -> Result<()> {
    while io.tell() + 8 <= end_pos {
        let box_start = io.tell();
        let header = boxes::read_box_header(io)?;
        let box_end = if header.size == 0 {
            end_pos
        } else {
            box_start.saturating_add(header.size).min(end_pos)
        };
        let payload_size = box_end.saturating_sub(io.tell());

        match &header.box_type {
            b"mdhd" => {
                let mdhd = boxes::parse_mdhd(io)?;
                tps.timescale = mdhd.timescale;
                tps.duration = mdhd.duration;
                tps.language = mdhd.language;
            }
            b"hdlr" => {
                let hdlr = boxes::parse_hdlr(io, payload_size)?;
                tps.handler_type = hdlr.handler_type;
            }
            b"minf" => {
                parse_minf(io, box_end, tps)?;
            }
            _ => {}
        }

        let current = io.tell();
        if current < box_end {
            io.skip(box_end - current)
                .map_err(|e| Error::io("mp4/mdia", &e))?;
        }
    }
    Ok(())
}

fn parse_minf(io: &mut BufByteSource, end_pos: u64, tps: &mut TrackParseState) -> Result<()> {
    while io.tell() + 8 <= end_pos {
        let box_start = io.tell();
        let header = boxes::read_box_header(io)?;
        let box_end = if header.size == 0 {
            end_pos
        } else {
            box_start.saturating_add(header.size).min(end_pos)
        };

        if &header.box_type == b"stbl" {
            parse_stbl(io, box_end, tps)?;
        }

        let current = io.tell();
        if current < box_end {
            io.skip(box_end - current)
                .map_err(|e| Error::io("mp4/minf", &e))?;
        }
    }
    Ok(())
}

fn parse_stbl(io: &mut BufByteSource, end_pos: u64, tps: &mut TrackParseState) -> Result<()> {
    while io.tell() + 8 <= end_pos {
        let box_start = io.tell();
        let header = boxes::read_box_header(io)?;
        let box_end = if header.size == 0 {
            end_pos
        } else {
            box_start.saturating_add(header.size).min(end_pos)
        };
        let payload_size = box_end.saturating_sub(io.tell());

        match &header.box_type {
            b"stsd" => tps.stsd_entry = Some(boxes::parse_stsd(io, payload_size)?),
            b"stts" => tps.stts = boxes::parse_stts(io)?,
            b"stsc" => tps.stsc = boxes::parse_stsc(io)?,
            b"stsz" => {
                let (fixed, _count, per_sample) = boxes::parse_stsz(io)?;
                tps.fixed_sample_size = fixed;
                tps.sample_sizes = per_sample;
            }
            b"stco" => tps.chunk_offsets = boxes::parse_stco(io)?,
            b"co64" => tps.chunk_offsets = boxes::parse_co64(io)?,
            b"stss" => tps.sync_samples = boxes::parse_stss(io)?,
            b"ctts" => tps.ctts = boxes::parse_ctts(io)?,
            _ => {}
        }

        let current = io.tell();
        if current < box_end {
            io.skip(box_end - current)
                .map_err(|e| Error::io("mp4/stbl", &e))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Probe & Factory
// ---------------------------------------------------------------------------

/// Probe score for ISOBMFF/MP4/MOV/M4A (`90..=100` on `ftyp`, `0` on non-MP4).
pub fn probe_mp4(head: &[u8], _ext: Option<&str>) -> u8 {
    if head.len() < 8 {
        return 0;
    }
    let size = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
    if size < 8 && size != 0 && size != 1 {
        return 0;
    }
    match &head[4..8] {
        b"ftyp" => {
            if head.len() >= 12 {
                match &head[8..12] {
                    b"isom" | b"iso2" | b"iso5" | b"iso6" | b"mp41" | b"mp42" | b"M4A "
                    | b"M4V " | b"qt  " | b"avc1" | b"hvc1" | b"dash" | b"cmfc" | b"3gp4"
                    | b"3gp5" => 100,
                    _ => 92,
                }
            } else {
                90
            }
        }
        b"moov" | b"mdat" | b"free" | b"wide" | b"styp" | b"moof" => 70,
        _ => 0,
    }
}

pub struct Mp4DemuxerFactory;

impl DemuxerFactory for Mp4DemuxerFactory {
    fn descriptor(&self) -> &DemuxerDescriptor {
        static DESC: DemuxerDescriptor = DemuxerDescriptor {
            name: "mp4",
            long_name: "MP4 / MOV / M4A (ISOBMFF)",
            extensions: &["mp4", "m4a", "m4v", "mov", "3gp"],
            mime_types: &["video/mp4", "audio/mp4", "video/quicktime"],
        };
        &DESC
    }

    fn probe(&self, head: &[u8], ext: Option<&str>) -> u8 {
        probe_mp4(head, ext)
    }

    fn open(&self, source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Box<dyn Demuxer>> {
        Ok(Box::new(Mp4Demuxer::open(source, policy)?))
    }
}
