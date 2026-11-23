//! ISOBMFF / MP4 / M4A muxer implementation (Wedeo v2).
//!
//! Automatically selects output layout based on `sink.is_seekable()`:
//! - **Seekable sinks**: classic faststart `moov`-first MP4 (`ftyp` + `moov` + `mdat`).
//! - **Non-seekable sinks** (pipes/streams): CMAF-ready fragmented MP4 (`ftyp` + `moov` [`mvex`/`trex`] + `moof`/`mdat` fragments).

use wedeo_core::{
    ByteSink, CodecId, ColorDesc, ContainerHeader, EditSegment, Error, MediaType, Muxer,
    MuxerDescriptor, MuxerFactory, Packet, PacketFlags, Rational, Result, StreamParams, TrackId,
    rescale_q,
};

use crate::atoms;

/// Per-track state accumulated during muxing.
pub struct TrackState {
    pub track_id: TrackId,
    pub codec_id: CodecId,
    pub media_type: MediaType,
    pub timescale: u32,
    pub language: Option<String>,
    pub edit_list: Vec<EditSegment>,

    // Video
    pub width: u32,
    pub height: u32,
    pub color: ColorDesc,
    pub sar: Rational,
    pub display_matrix: Option<[i32; 9]>,

    // Audio
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub bit_rate: i64,

    pub extradata: Vec<u8>,

    // Classic sample table data
    pub sample_count: u32,
    pub sample_sizes: Vec<u32>,
    pub sample_durations: Vec<u32>,
    pub chunk_offsets: Vec<u64>,
    pub sync_samples: Vec<u32>,
    pub cts_offsets: Vec<i32>,
    pub has_cts: bool,
    pub duration_ts: u64,

    // Fragmented MP4 tracking
    pub next_dts: i64,
}

/// One sample inside a `moof` fragment.
#[derive(Debug, Clone, Copy)]
pub struct FragmentSample {
    pub duration: u32,
    pub size: u32,
    pub cts_offset: i32,
    pub is_key: bool,
}

/// Per-track fragment buffer for fMP4 `traf` / `trun` generation.
pub struct FragmentTrackData {
    pub track_id: u32,
    pub base_decode_time: u64,
    pub data_offset: i32,
    pub samples: Vec<FragmentSample>,
    pub payload: Vec<u8>,
}

/// Native ISOBMFF / MP4 / M4A muxer owning a `Box<dyn ByteSink>`.
pub struct Mp4Muxer {
    sink: Box<dyn ByteSink>,
    tracks: Vec<TrackState>,
    metadata: Vec<(String, String)>,
    movie_timescale: u32,
    fragmented: bool,
    ftyp_len: u64,
    // Seekable faststart buffer
    mdat_buf: Vec<u8>,
    // Fragmented MP4 state
    frag_seq: u32,
    frag_tracks: Vec<FragmentTrackData>,
    frag_total_bytes: usize,
}

impl Mp4Muxer {
    /// Create a new `Mp4Muxer` owning `sink`.
    pub fn new(sink: Box<dyn ByteSink>) -> Self {
        Self {
            sink,
            tracks: Vec::new(),
            metadata: Vec::new(),
            movie_timescale: 1000,
            fragmented: false,
            ftyp_len: 0,
            mdat_buf: Vec::new(),
            frag_seq: 0,
            frag_tracks: Vec::new(),
            frag_total_bytes: 0,
        }
    }

    fn find_track(&self, track_id: TrackId) -> Option<usize> {
        self.tracks.iter().position(|t| t.track_id == track_id)
    }

    fn flush_fragment(&mut self) -> Result<()> {
        let mut active: Vec<FragmentTrackData> = Vec::new();
        for ft in &mut self.frag_tracks {
            if !ft.samples.is_empty() {
                active.push(FragmentTrackData {
                    track_id: ft.track_id,
                    base_decode_time: ft.base_decode_time,
                    data_offset: 0,
                    samples: std::mem::take(&mut ft.samples),
                    payload: std::mem::take(&mut ft.payload),
                });
            }
        }
        self.frag_total_bytes = 0;

        if active.is_empty() {
            return Ok(());
        }

        self.frag_seq += 1;
        let trial_moof = atoms::write_moof(self.frag_seq, &active);
        let moof_size = trial_moof.len() as i32;

        let mut payload_offset = 0i32;
        for frag in &mut active {
            frag.data_offset = moof_size + 8 + payload_offset;
            payload_offset += frag.payload.len() as i32;
        }

        let moof = atoms::write_moof(self.frag_seq, &active);
        let mdat_size = (8 + payload_offset as usize) as u32;

        self.sink
            .write_all(&moof)
            .map_err(|e| Error::io("mp4/muxer", &e))?;
        self.sink
            .write_all(&mdat_size.to_be_bytes())
            .map_err(|e| Error::io("mp4/muxer", &e))?;
        self.sink
            .write_all(b"mdat")
            .map_err(|e| Error::io("mp4/muxer", &e))?;
        for frag in &active {
            self.sink
                .write_all(&frag.payload)
                .map_err(|e| Error::io("mp4/muxer", &e))?;
        }
        self.sink.flush().map_err(|e| Error::io("mp4/muxer", &e))?;

        Ok(())
    }

    fn build_faststart_moov(&mut self, mdat_header_size: u64) -> Vec<u8> {
        let original_offsets: Vec<Vec<u64>> = self
            .tracks
            .iter()
            .map(|t| t.chunk_offsets.clone())
            .collect();

        let trial_moov =
            atoms::write_moov(&self.tracks, self.movie_timescale, false, &self.metadata);
        let mut moov_size = trial_moov.len() as u64;

        for iter in 0..2 {
            let base = self.ftyp_len + moov_size + mdat_header_size;
            for (track, orig) in self.tracks.iter_mut().zip(original_offsets.iter()) {
                track.chunk_offsets = orig.iter().map(|&o| base + o).collect();
            }

            let moov = atoms::write_moov(&self.tracks, self.movie_timescale, false, &self.metadata);
            let new_size = moov.len() as u64;
            if new_size == moov_size || iter == 1 {
                return moov;
            }
            moov_size = new_size;
        }
        unreachable!()
    }
}

impl Muxer for Mp4Muxer {
    fn write_header(&mut self, header: &ContainerHeader) -> Result<()> {
        if header.tracks.is_empty() {
            return Err(Error::invalid_argument(
                "mp4/muxer",
                "container header has no tracks",
            ));
        }

        self.fragmented = !self.sink.is_seekable();
        self.metadata = header
            .metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let mut has_video = false;
        let mut has_audio = false;

        for (i, desc) in header.tracks.iter().enumerate() {
            if !is_supported_codec(desc.codec_id) {
                return Err(Error::unsupported(
                    "mp4/muxer",
                    format!("unsupported codec: {}", desc.codec_id.name()),
                ));
            }

            let timescale = compute_timescale(desc);
            let (
                media_type,
                width,
                height,
                color,
                sar,
                display_matrix,
                sample_rate,
                channels,
                bits_per_sample,
                bit_rate,
                extradata,
            ) = match &desc.params {
                StreamParams::Video(vp) => {
                    has_video = true;
                    (
                        MediaType::Video,
                        vp.width,
                        vp.height,
                        vp.color,
                        vp.sar,
                        vp.display_matrix,
                        0,
                        0,
                        0,
                        vp.bit_rate,
                        vp.extradata.clone(),
                    )
                }
                StreamParams::Audio(ap) => {
                    has_audio = true;
                    (
                        MediaType::Audio,
                        0,
                        0,
                        ColorDesc::UNSPECIFIED,
                        Rational::new(1, 1),
                        None,
                        ap.sample_rate,
                        ap.channel_layout.nb_channels as u16,
                        (ap.bits_per_coded_sample as u16).max(16),
                        ap.bit_rate,
                        ap.extradata.clone(),
                    )
                }
                _ => {
                    return Err(Error::unsupported(
                        "mp4/muxer",
                        "only video and audio tracks are supported",
                    ));
                }
            };

            validate_extradata(desc.codec_id, &extradata)?;

            self.tracks.push(TrackState {
                track_id: desc.id,
                codec_id: desc.codec_id,
                media_type,
                timescale,
                language: desc.language.clone(),
                edit_list: desc.edit_list.clone(),
                width,
                height,
                color,
                sar,
                display_matrix,
                sample_rate,
                channels,
                bits_per_sample,
                bit_rate,
                extradata,
                sample_count: 0,
                sample_sizes: Vec::new(),
                sample_durations: Vec::new(),
                chunk_offsets: Vec::new(),
                sync_samples: Vec::new(),
                cts_offsets: Vec::new(),
                has_cts: false,
                duration_ts: 0,
                next_dts: 0,
            });

            self.frag_tracks.push(FragmentTrackData {
                track_id: i as u32 + 1,
                base_decode_time: 0,
                data_offset: 0,
                samples: Vec::new(),
                payload: Vec::new(),
            });
        }

        let ftyp = atoms::write_ftyp(has_video, has_audio, self.fragmented);
        self.ftyp_len = ftyp.len() as u64;
        self.sink
            .write_all(&ftyp)
            .map_err(|e| Error::io("mp4/muxer", &e))?;

        if self.fragmented {
            let moov = atoms::write_moov(&self.tracks, self.movie_timescale, true, &self.metadata);
            self.sink
                .write_all(&moov)
                .map_err(|e| Error::io("mp4/muxer", &e))?;
        }

        self.sink.flush().map_err(|e| Error::io("mp4/muxer", &e))?;
        Ok(())
    }

    fn write_packet(&mut self, pkt: &Packet) -> Result<()> {
        let track_idx = self
            .find_track(pkt.track_id)
            .ok_or_else(|| Error::not_found("mp4/muxer", "packet track_id not in header"))?;

        let timescale = self.tracks[track_idx].timescale.max(1);
        let track_tb = Rational::new(1, timescale as i32);

        let rescale_ts = |ts: i64| -> i64 {
            if pkt.time_base.is_valid() && pkt.time_base != track_tb {
                rescale_q(ts, pkt.time_base, track_tb)
            } else {
                ts
            }
        };

        let dts = match pkt.dts.get() {
            Some(d) => rescale_ts(d),
            None => match pkt.pts.get() {
                Some(p) => rescale_ts(p),
                None => self.tracks[track_idx].next_dts,
            },
        };
        let pts = pkt.pts.get().map(rescale_ts).unwrap_or(dts);
        let duration = if pkt.duration > 0 {
            rescale_ts(pkt.duration).max(1) as u32
        } else {
            1
        };
        let cts_offset = (pts - dts) as i32;
        let is_key = pkt.flags.contains(PacketFlags::KEY);
        let data = pkt.data.as_slice().ok_or_else(|| {
            Error::invalid_argument("mp4/muxer", "packet data must be CPU-accessible")
        })?;

        if self.fragmented {
            let is_video_key = self.tracks[track_idx].media_type == MediaType::Video && is_key;
            if (is_video_key && self.frag_total_bytes > 0) || self.frag_total_bytes >= 1_048_576 {
                self.flush_fragment()?;
            }

            let ft = &mut self.frag_tracks[track_idx];
            if ft.samples.is_empty() {
                ft.base_decode_time = dts.max(0) as u64;
            }
            ft.samples.push(FragmentSample {
                duration,
                size: data.len() as u32,
                cts_offset,
                is_key,
            });
            ft.payload.extend_from_slice(data);
            self.frag_total_bytes += data.len();
            self.tracks[track_idx].next_dts = dts + i64::from(duration);
        } else {
            let offset_in_mdat = self.mdat_buf.len() as u64;
            self.mdat_buf.extend_from_slice(data);

            let track = &mut self.tracks[track_idx];
            track.sample_count += 1;
            track.sample_sizes.push(data.len() as u32);
            track.chunk_offsets.push(offset_in_mdat);
            track.sample_durations.push(duration);
            track.duration_ts += u64::from(duration);
            track.next_dts = dts + i64::from(duration);
            if is_key {
                track.sync_samples.push(track.sample_count);
            }
            track.cts_offsets.push(cts_offset);
            if cts_offset != 0 {
                track.has_cts = true;
            }
        }

        Ok(())
    }

    fn finalize(mut self: Box<Self>) -> Result<()> {
        if self.fragmented {
            self.flush_fragment()?;
            self.sink.flush().map_err(|e| Error::io("mp4/muxer", &e))?;
            return Ok(());
        }

        let payload_len = self.mdat_buf.len() as u64;
        let use_largesize = payload_len + 8 > u64::from(u32::MAX);
        let mdat_header_size = if use_largesize { 16u64 } else { 8u64 };

        let moov = self.build_faststart_moov(mdat_header_size);
        self.sink
            .write_all(&moov)
            .map_err(|e| Error::io("mp4/muxer", &e))?;

        if use_largesize {
            let total_size = 16 + payload_len;
            self.sink
                .write_all(&1u32.to_be_bytes())
                .map_err(|e| Error::io("mp4/muxer", &e))?;
            self.sink
                .write_all(b"mdat")
                .map_err(|e| Error::io("mp4/muxer", &e))?;
            self.sink
                .write_all(&total_size.to_be_bytes())
                .map_err(|e| Error::io("mp4/muxer", &e))?;
        } else {
            let total_size = (8 + payload_len) as u32;
            self.sink
                .write_all(&total_size.to_be_bytes())
                .map_err(|e| Error::io("mp4/muxer", &e))?;
            self.sink
                .write_all(b"mdat")
                .map_err(|e| Error::io("mp4/muxer", &e))?;
        }

        self.sink
            .write_all(&self.mdat_buf)
            .map_err(|e| Error::io("mp4/muxer", &e))?;
        self.sink.flush().map_err(|e| Error::io("mp4/muxer", &e))?;
        Ok(())
    }
}

fn validate_extradata(codec_id: CodecId, extradata: &[u8]) -> Result<()> {
    match codec_id {
        CodecId::H264 => {
            if extradata.is_empty() {
                return Err(Error::invalid_argument(
                    "mp4/muxer",
                    "H.264 requires avcC extradata",
                ));
            }
            if extradata[0] != 0x01 {
                return Err(Error::invalid_data(
                    "mp4/muxer",
                    "H.264 extradata must be avcC (starting with version byte 0x01)",
                ));
            }
        }
        CodecId::Hevc | CodecId::Av1 | CodecId::Vp9 | CodecId::Aac if extradata.is_empty() => {
            return Err(Error::invalid_argument(
                "mp4/muxer",
                format!("{} requires extradata", codec_id.name()),
            ));
        }
        _ => {}
    }
    Ok(())
}

fn is_supported_codec(codec_id: CodecId) -> bool {
    matches!(
        codec_id,
        CodecId::H264
            | CodecId::Hevc
            | CodecId::Av1
            | CodecId::Vp9
            | CodecId::Aac
            | CodecId::Alac
            | CodecId::Opus
            | CodecId::Flac
            | CodecId::Ac3
            | CodecId::PcmS8
            | CodecId::PcmU8
            | CodecId::PcmS16be
            | CodecId::PcmS16le
            | CodecId::PcmS24be
            | CodecId::PcmS24le
            | CodecId::PcmS32be
            | CodecId::PcmS32le
            | CodecId::PcmF32be
            | CodecId::PcmF32le
            | CodecId::PcmF64be
            | CodecId::PcmF64le
    )
}

fn compute_timescale(desc: &wedeo_core::TrackDescriptor) -> u32 {
    if desc.time_base.is_valid() && desc.time_base.num == 1 && desc.time_base.den > 0 {
        return desc.time_base.den as u32;
    }
    match &desc.params {
        StreamParams::Audio(ap) if ap.sample_rate > 0 => ap.sample_rate,
        _ => 90_000,
    }
}

pub struct Mp4MuxerFactory;

impl MuxerFactory for Mp4MuxerFactory {
    fn descriptor(&self) -> &MuxerDescriptor {
        static DESC: MuxerDescriptor = MuxerDescriptor {
            name: "mp4",
            long_name: "MP4 / MOV / M4A (ISOBMFF)",
            extensions: &["mp4", "m4a", "m4v", "mov"],
            mime_types: &["video/mp4", "audio/mp4"],
            default_video: CodecId::H264,
            default_audio: CodecId::Aac,
        };
        &DESC
    }

    fn create(&self, sink: Box<dyn ByteSink>) -> Result<Box<dyn Muxer>> {
        Ok(Box::new(Mp4Muxer::new(sink)))
    }
}
