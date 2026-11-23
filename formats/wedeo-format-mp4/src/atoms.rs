//! MP4 / ISOBMFF box (atom) building utilities for classic faststart and fragmented MP4 (fMP4).

use wedeo_core::{CodecId, ColorDesc, ColorRange, EditSegment, MediaType, Rational, rescale_q};

use crate::boxes::{self, IDENTITY_MATRIX_I32};
use crate::muxer::{FragmentTrackData, TrackState};

// ---------------------------------------------------------------------------
// Core box primitives
// ---------------------------------------------------------------------------

/// Build a box: 4-byte size (big-endian) + 4-byte type + content.
pub fn mp4_box(box_type: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let size = (8 + content.len()) as u32;
    let mut buf = Vec::with_capacity(size as usize);
    buf.extend_from_slice(&size.to_be_bytes());
    buf.extend_from_slice(box_type);
    buf.extend_from_slice(content);
    buf
}

/// Build a full box: size + type + version(1) + flags(3) + content.
pub fn full_box(box_type: &[u8; 4], version: u8, flags: u32, content: &[u8]) -> Vec<u8> {
    let mut inner = Vec::with_capacity(4 + content.len());
    let version_flags = (u32::from(version) << 24) | (flags & 0x00FF_FFFF);
    inner.extend_from_slice(&version_flags.to_be_bytes());
    inner.extend_from_slice(content);
    mp4_box(box_type, &inner)
}

fn write_matrix_bytes(matrix: &[i32; 9]) -> [u8; 36] {
    let mut out = [0u8; 36];
    for (i, &v) in matrix.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Top-level boxes
// ---------------------------------------------------------------------------

/// Build `ftyp` box. Audio-only files use `M4A ` branding; fragmented files include `iso5`/`cmfc`.
pub fn write_ftyp(has_video: bool, has_audio: bool, fragmented: bool) -> Vec<u8> {
    let mut content = Vec::with_capacity(32);
    if fragmented {
        content.extend_from_slice(b"iso5");
        content.extend_from_slice(&0x200u32.to_be_bytes());
        content.extend_from_slice(b"isom");
        content.extend_from_slice(b"iso2");
        content.extend_from_slice(b"iso5");
        content.extend_from_slice(b"iso6");
        content.extend_from_slice(b"cmfc");
        if has_video {
            content.extend_from_slice(b"avc1");
        }
        content.extend_from_slice(b"mp41");
    } else {
        if has_audio && !has_video {
            content.extend_from_slice(b"M4A ");
        } else {
            content.extend_from_slice(b"isom");
        }
        content.extend_from_slice(&0x200u32.to_be_bytes());
        content.extend_from_slice(b"isom");
        content.extend_from_slice(b"iso2");
        if has_video {
            content.extend_from_slice(b"avc1");
        }
        if has_audio && !has_video {
            content.extend_from_slice(b"M4A ");
        }
        content.extend_from_slice(b"mp41");
    }
    mp4_box(b"ftyp", &content)
}

fn rescale_duration(duration_ts: u64, media_timescale: u32, movie_timescale: u32) -> u64 {
    if media_timescale == 0 {
        return 0;
    }
    let ts = u64::from(media_timescale);
    (duration_ts.saturating_mul(u64::from(movie_timescale)) + ts / 2) / ts
}

/// Build the complete `moov` box for all tracks (classic or fragmented with `mvex`).
pub fn write_moov(tracks: &[TrackState], movie_timescale: u32, fragmented: bool) -> Vec<u8> {
    let movie_duration: u64 = if fragmented {
        0
    } else {
        tracks
            .iter()
            .map(|t| {
                if t.timescale == 0 || t.sample_count == 0 {
                    0
                } else {
                    rescale_duration(t.duration_ts, t.timescale, movie_timescale)
                }
            })
            .max()
            .unwrap_or(0)
    };

    let next_track_id = tracks.len() as u32 + 1;
    let mvhd = write_mvhd(movie_timescale, movie_duration, next_track_id);

    let mut moov_content = Vec::new();
    moov_content.extend_from_slice(&mvhd);
    for (i, track) in tracks.iter().enumerate() {
        let track_id = i as u32 + 1;
        let track_duration_movie = if fragmented || track.timescale == 0 || track.sample_count == 0
        {
            0
        } else {
            rescale_duration(track.duration_ts, track.timescale, movie_timescale)
        };
        let trak = write_trak(
            track,
            track_id,
            track_duration_movie,
            movie_timescale,
            fragmented,
        );
        moov_content.extend_from_slice(&trak);
    }

    if fragmented {
        moov_content.extend_from_slice(&write_mvex(tracks));
    }

    mp4_box(b"moov", &moov_content)
}

fn write_mvex(tracks: &[TrackState]) -> Vec<u8> {
    let mut content = Vec::with_capacity(tracks.len() * 32);
    for (i, _track) in tracks.iter().enumerate() {
        let track_id = i as u32 + 1;
        let mut c = Vec::with_capacity(20);
        c.extend_from_slice(&track_id.to_be_bytes());
        c.extend_from_slice(&1u32.to_be_bytes()); // default_sample_description_index
        c.extend_from_slice(&0u32.to_be_bytes()); // default_sample_duration
        c.extend_from_slice(&0u32.to_be_bytes()); // default_sample_size
        c.extend_from_slice(&0u32.to_be_bytes()); // default_sample_flags
        content.extend_from_slice(&full_box(b"trex", 0, 0, &c));
    }
    mp4_box(b"mvex", &content)
}

/// Build a `moof` box for a fragment.
pub fn write_moof(sequence_number: u32, frags: &[FragmentTrackData]) -> Vec<u8> {
    let mut moof_content = Vec::new();
    moof_content.extend_from_slice(&full_box(b"mfhd", 0, 0, &sequence_number.to_be_bytes()));
    for frag in frags {
        moof_content.extend_from_slice(&write_traf(frag));
    }
    mp4_box(b"moof", &moof_content)
}

fn write_traf(frag: &FragmentTrackData) -> Vec<u8> {
    let mut content = Vec::new();
    // tfhd: flags = 0x020000 (default-base-is-moof)
    content.extend_from_slice(&full_box(
        b"tfhd",
        0,
        0x020000,
        &frag.track_id.to_be_bytes(),
    ));
    // tfdt: version 1 (64-bit baseMediaDecodeTime)
    content.extend_from_slice(&full_box(
        b"tfdt",
        1,
        0,
        &frag.base_decode_time.to_be_bytes(),
    ));
    // trun: data-offset(0x1) | sample-duration(0x100) | sample-size(0x200) | sample-flags(0x400) | cts(0x800)
    let has_cts = frag.samples.iter().any(|s| s.cts_offset != 0);
    let mut trun_flags: u32 = 0x000001 | 0x000100 | 0x000200 | 0x000400;
    if has_cts {
        trun_flags |= 0x000800;
    }
    let has_neg_cts = frag.samples.iter().any(|s| s.cts_offset < 0);
    let trun_version = u8::from(has_neg_cts);

    let mut trun_c = Vec::with_capacity(8 + frag.samples.len() * 16);
    trun_c.extend_from_slice(&(frag.samples.len() as u32).to_be_bytes());
    trun_c.extend_from_slice(&frag.data_offset.to_be_bytes());
    for s in &frag.samples {
        trun_c.extend_from_slice(&s.duration.to_be_bytes());
        trun_c.extend_from_slice(&s.size.to_be_bytes());
        let sflags: u32 = if s.is_key { 0x0200_0000 } else { 0x0101_0000 };
        trun_c.extend_from_slice(&sflags.to_be_bytes());
        if has_cts {
            trun_c.extend_from_slice(&(s.cts_offset as u32).to_be_bytes());
        }
    }
    content.extend_from_slice(&full_box(b"trun", trun_version, trun_flags, &trun_c));
    mp4_box(b"traf", &content)
}

// ---------------------------------------------------------------------------
// Movie-level boxes
// ---------------------------------------------------------------------------

fn write_mvhd(timescale: u32, duration: u64, next_track_id: u32) -> Vec<u8> {
    let version = u8::from(duration > u64::from(u32::MAX));
    let mut c = Vec::with_capacity(108);
    if version == 1 {
        c.extend_from_slice(&0u64.to_be_bytes());
        c.extend_from_slice(&0u64.to_be_bytes());
        c.extend_from_slice(&timescale.to_be_bytes());
        c.extend_from_slice(&duration.to_be_bytes());
    } else {
        c.extend_from_slice(&0u32.to_be_bytes());
        c.extend_from_slice(&0u32.to_be_bytes());
        c.extend_from_slice(&timescale.to_be_bytes());
        c.extend_from_slice(&(duration as u32).to_be_bytes());
    }
    c.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate = 1.0
    c.extend_from_slice(&0x0100u16.to_be_bytes()); // volume = 1.0
    c.extend_from_slice(&[0u8; 10]); // reserved
    c.extend_from_slice(&boxes::IDENTITY_MATRIX_BYTES);
    c.extend_from_slice(&[0u8; 24]); // pre_defined
    c.extend_from_slice(&next_track_id.to_be_bytes());
    full_box(b"mvhd", version, 0, &c)
}

// ---------------------------------------------------------------------------
// Track boxes
// ---------------------------------------------------------------------------

fn write_trak(
    track: &TrackState,
    track_id: u32,
    duration_movie: u64,
    movie_timescale: u32,
    fragmented: bool,
) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&write_tkhd(track, track_id, duration_movie));
    if !track.edit_list.is_empty() {
        content.extend_from_slice(&write_edts(&track.edit_list, movie_timescale));
    }
    content.extend_from_slice(&write_mdia(track, fragmented));
    mp4_box(b"trak", &content)
}

fn write_tkhd(track: &TrackState, track_id: u32, duration_movie: u64) -> Vec<u8> {
    let version = u8::from(duration_movie > u64::from(u32::MAX));
    let mut c = Vec::with_capacity(92);
    if version == 1 {
        c.extend_from_slice(&0u64.to_be_bytes());
        c.extend_from_slice(&0u64.to_be_bytes());
        c.extend_from_slice(&track_id.to_be_bytes());
        c.extend_from_slice(&0u32.to_be_bytes());
        c.extend_from_slice(&duration_movie.to_be_bytes());
    } else {
        c.extend_from_slice(&0u32.to_be_bytes());
        c.extend_from_slice(&0u32.to_be_bytes());
        c.extend_from_slice(&track_id.to_be_bytes());
        c.extend_from_slice(&0u32.to_be_bytes());
        c.extend_from_slice(&(duration_movie as u32).to_be_bytes());
    }
    c.extend_from_slice(&[0u8; 8]); // reserved
    c.extend_from_slice(&0u16.to_be_bytes()); // layer
    c.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
    let volume: u16 = if track.media_type == MediaType::Audio {
        0x0100
    } else {
        0
    };
    c.extend_from_slice(&volume.to_be_bytes());
    c.extend_from_slice(&0u16.to_be_bytes()); // reserved
    let matrix = track.display_matrix.unwrap_or(IDENTITY_MATRIX_I32);
    c.extend_from_slice(&write_matrix_bytes(&matrix));
    c.extend_from_slice(&(track.width << 16).to_be_bytes());
    c.extend_from_slice(&(track.height << 16).to_be_bytes());
    full_box(b"tkhd", version, 0x03, &c)
}

fn write_edts(edits: &[EditSegment], movie_timescale: u32) -> Vec<u8> {
    let movie_tb = Rational::new(1, movie_timescale.max(1) as i32);
    let mut use_v1 = false;
    let mut entries = Vec::with_capacity(edits.len());

    for seg in edits {
        let dur = if seg.time_base.is_valid() {
            rescale_q(seg.duration, seg.time_base, movie_tb).max(0) as u64
        } else {
            seg.duration.max(0) as u64
        };
        let mt = seg.media_time.get().unwrap_or(-1);
        if dur > u64::from(u32::MAX) || mt > i64::from(i32::MAX) || mt < i64::from(i32::MIN) {
            use_v1 = true;
        }
        let (rate_int, rate_frac) = if seg.rate.is_valid() {
            let fp = ((i64::from(seg.rate.num) << 16) / i64::from(seg.rate.den)) as i32;
            ((fp >> 16) as i16, (fp & 0xFFFF) as u16 as i16)
        } else {
            (1i16, 0i16)
        };
        entries.push((dur, mt, rate_int, rate_frac));
    }

    let version = u8::from(use_v1);
    let mut c = Vec::with_capacity(4 + entries.len() * 20);
    c.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for (dur, mt, ri, rf) in entries {
        if version == 1 {
            c.extend_from_slice(&dur.to_be_bytes());
            c.extend_from_slice(&mt.to_be_bytes());
        } else {
            c.extend_from_slice(&(dur as u32).to_be_bytes());
            c.extend_from_slice(&(mt as i32).to_be_bytes());
        }
        c.extend_from_slice(&ri.to_be_bytes());
        c.extend_from_slice(&rf.to_be_bytes());
    }
    let elst = full_box(b"elst", version, 0, &c);
    mp4_box(b"edts", &elst)
}

// ---------------------------------------------------------------------------
// Media boxes
// ---------------------------------------------------------------------------

fn write_mdia(track: &TrackState, fragmented: bool) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&write_mdhd(track, fragmented));
    content.extend_from_slice(&write_hdlr(track));
    content.extend_from_slice(&write_minf(track, fragmented));
    mp4_box(b"mdia", &content)
}

fn write_mdhd(track: &TrackState, fragmented: bool) -> Vec<u8> {
    let dur = if fragmented { 0 } else { track.duration_ts };
    let version = u8::from(dur > u64::from(u32::MAX));
    let mut c = Vec::with_capacity(32);
    if version == 1 {
        c.extend_from_slice(&0u64.to_be_bytes());
        c.extend_from_slice(&0u64.to_be_bytes());
        c.extend_from_slice(&track.timescale.to_be_bytes());
        c.extend_from_slice(&dur.to_be_bytes());
    } else {
        c.extend_from_slice(&0u32.to_be_bytes());
        c.extend_from_slice(&0u32.to_be_bytes());
        c.extend_from_slice(&track.timescale.to_be_bytes());
        c.extend_from_slice(&(dur as u32).to_be_bytes());
    }
    let lang = boxes::pack_mdhd_language(track.language.as_deref());
    c.extend_from_slice(&lang.to_be_bytes());
    c.extend_from_slice(&0u16.to_be_bytes());
    full_box(b"mdhd", version, 0, &c)
}

fn write_hdlr(track: &TrackState) -> Vec<u8> {
    let (handler_type, name) = match track.media_type {
        MediaType::Video => (b"vide", b"VideoHandler\0" as &[u8]),
        MediaType::Audio => (b"soun", b"SoundHandler\0" as &[u8]),
        _ => (b"data", b"DataHandler\0" as &[u8]),
    };
    let mut c = Vec::with_capacity(24 + name.len());
    c.extend_from_slice(&0u32.to_be_bytes());
    c.extend_from_slice(handler_type);
    c.extend_from_slice(&[0u8; 12]);
    c.extend_from_slice(name);
    full_box(b"hdlr", 0, 0, &c)
}

// ---------------------------------------------------------------------------
// Media information boxes
// ---------------------------------------------------------------------------

fn write_minf(track: &TrackState, fragmented: bool) -> Vec<u8> {
    let mut content = Vec::new();
    match track.media_type {
        MediaType::Video => content.extend_from_slice(&write_vmhd()),
        MediaType::Audio => content.extend_from_slice(&write_smhd()),
        _ => {}
    }
    content.extend_from_slice(&write_dinf());
    content.extend_from_slice(&write_stbl(track, fragmented));
    mp4_box(b"minf", &content)
}

fn write_vmhd() -> Vec<u8> {
    let mut c = Vec::with_capacity(8);
    c.extend_from_slice(&0u16.to_be_bytes());
    c.extend_from_slice(&[0u8; 6]);
    full_box(b"vmhd", 0, 1, &c)
}

fn write_smhd() -> Vec<u8> {
    let mut c = Vec::with_capacity(4);
    c.extend_from_slice(&0u16.to_be_bytes());
    c.extend_from_slice(&0u16.to_be_bytes());
    full_box(b"smhd", 0, 0, &c)
}

fn write_dinf() -> Vec<u8> {
    let url_box = full_box(b"url ", 0, 0x01, &[]);
    let mut dref_content = Vec::with_capacity(4 + url_box.len());
    dref_content.extend_from_slice(&1u32.to_be_bytes());
    dref_content.extend_from_slice(&url_box);
    let dref = full_box(b"dref", 0, 0, &dref_content);
    mp4_box(b"dinf", &dref)
}

// ---------------------------------------------------------------------------
// Sample table boxes
// ---------------------------------------------------------------------------

fn write_stbl(track: &TrackState, fragmented: bool) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&write_stsd(track));
    if fragmented {
        let empty = 0u32.to_be_bytes();
        content.extend_from_slice(&full_box(b"stts", 0, 0, &empty));
        content.extend_from_slice(&full_box(b"stsc", 0, 0, &empty));
        content.extend_from_slice(&full_box(b"stsz", 0, 0, &[0u8; 8]));
        content.extend_from_slice(&full_box(b"stco", 0, 0, &empty));
    } else {
        content.extend_from_slice(&write_stts(track));
        if track.has_cts {
            content.extend_from_slice(&write_ctts(track));
        }
        content.extend_from_slice(&write_stsc());
        content.extend_from_slice(&write_stsz(track));
        content.extend_from_slice(&write_chunk_offsets(track));
        if track.media_type == MediaType::Video
            && !track.sync_samples.is_empty()
            && track.sync_samples.len() != track.sample_count as usize
        {
            content.extend_from_slice(&write_stss(track));
        }
    }
    mp4_box(b"stbl", &content)
}

fn write_stsd(track: &TrackState) -> Vec<u8> {
    let entry = match track.codec_id {
        CodecId::H264 | CodecId::Hevc | CodecId::Av1 | CodecId::Vp9 => write_visual_entry(track),
        CodecId::Aac => write_mp4a_entry(track),
        CodecId::Alac => write_alac_entry(track),
        CodecId::Opus => write_opus_entry(track),
        CodecId::Flac => write_flac_entry(track),
        CodecId::Ac3 => write_ac3_entry(track),
        _ => write_pcm_audio_entry(track),
    };
    let mut c = Vec::with_capacity(4 + entry.len());
    c.extend_from_slice(&1u32.to_be_bytes());
    c.extend_from_slice(&entry);
    full_box(b"stsd", 0, 0, &c)
}

fn write_stts(track: &TrackState) -> Vec<u8> {
    let entries = rle_encode(&track.sample_durations);
    let mut c = Vec::with_capacity(4 + entries.len() * 8);
    c.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for &(count, delta) in &entries {
        c.extend_from_slice(&count.to_be_bytes());
        c.extend_from_slice(&delta.to_be_bytes());
    }
    full_box(b"stts", 0, 0, &c)
}

fn write_ctts(track: &TrackState) -> Vec<u8> {
    let entries = rle_encode_i32(&track.cts_offsets);
    let mut c = Vec::with_capacity(4 + entries.len() * 8);
    c.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for &(count, offset) in &entries {
        c.extend_from_slice(&count.to_be_bytes());
        c.extend_from_slice(&(offset as u32).to_be_bytes());
    }
    let version = u8::from(track.cts_offsets.iter().any(|&o| o < 0));
    full_box(b"ctts", version, 0, &c)
}

fn write_stsc() -> Vec<u8> {
    let mut c = Vec::with_capacity(16);
    c.extend_from_slice(&1u32.to_be_bytes());
    c.extend_from_slice(&1u32.to_be_bytes());
    c.extend_from_slice(&1u32.to_be_bytes());
    c.extend_from_slice(&1u32.to_be_bytes());
    full_box(b"stsc", 0, 0, &c)
}

fn write_stsz(track: &TrackState) -> Vec<u8> {
    let mut c = Vec::with_capacity(8 + track.sample_sizes.len() * 4);
    c.extend_from_slice(&0u32.to_be_bytes());
    c.extend_from_slice(&track.sample_count.to_be_bytes());
    for &size in &track.sample_sizes {
        c.extend_from_slice(&size.to_be_bytes());
    }
    full_box(b"stsz", 0, 0, &c)
}

fn write_chunk_offsets(track: &TrackState) -> Vec<u8> {
    let needs_64bit = track.chunk_offsets.iter().any(|&o| o > u64::from(u32::MAX));
    if needs_64bit {
        let mut c = Vec::with_capacity(4 + track.chunk_offsets.len() * 8);
        c.extend_from_slice(&(track.chunk_offsets.len() as u32).to_be_bytes());
        for &offset in &track.chunk_offsets {
            c.extend_from_slice(&offset.to_be_bytes());
        }
        full_box(b"co64", 0, 0, &c)
    } else {
        let mut c = Vec::with_capacity(4 + track.chunk_offsets.len() * 4);
        c.extend_from_slice(&(track.chunk_offsets.len() as u32).to_be_bytes());
        for &offset in &track.chunk_offsets {
            c.extend_from_slice(&(offset as u32).to_be_bytes());
        }
        full_box(b"stco", 0, 0, &c)
    }
}

fn write_stss(track: &TrackState) -> Vec<u8> {
    let mut c = Vec::with_capacity(4 + track.sync_samples.len() * 4);
    c.extend_from_slice(&(track.sync_samples.len() as u32).to_be_bytes());
    for &sample_num in &track.sync_samples {
        c.extend_from_slice(&sample_num.to_be_bytes());
    }
    full_box(b"stss", 0, 0, &c)
}

// ---------------------------------------------------------------------------
// Sample description entries
// ---------------------------------------------------------------------------

fn write_visual_entry(track: &TrackState) -> Vec<u8> {
    let fourcc = match track.codec_id {
        CodecId::H264 => b"avc1",
        CodecId::Hevc => b"hvc1",
        CodecId::Av1 => b"av01",
        CodecId::Vp9 => b"vp09",
        _ => b"avc1",
    };

    let mut c = Vec::with_capacity(96 + track.extradata.len());
    c.extend_from_slice(&[0u8; 6]);
    c.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    c.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    c.extend_from_slice(&0u16.to_be_bytes()); // reserved
    c.extend_from_slice(&[0u8; 12]); // pre_defined
    c.extend_from_slice(&(track.width as u16).to_be_bytes());
    c.extend_from_slice(&(track.height as u16).to_be_bytes());
    c.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // 72 dpi
    c.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // 72 dpi
    c.extend_from_slice(&0u32.to_be_bytes());
    c.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    c.extend_from_slice(&[0u8; 32]); // compressorname
    c.extend_from_slice(&0x0018u16.to_be_bytes()); // depth = 24
    c.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined

    if !track.extradata.is_empty() {
        match track.codec_id {
            CodecId::H264 => c.extend_from_slice(&mp4_box(b"avcC", &track.extradata)),
            CodecId::Hevc => c.extend_from_slice(&mp4_box(b"hvcC", &track.extradata)),
            CodecId::Vp9 => c.extend_from_slice(&mp4_box(b"vpcC", &track.extradata)),
            CodecId::Av1 => {
                let av1c_content = if track.extradata.len() >= 4 && track.extradata[0] == 0x81 {
                    track.extradata.clone()
                } else {
                    let mut buf = Vec::with_capacity(4 + track.extradata.len());
                    buf.extend_from_slice(&[0x81, 0x04, 0x0C, 0x00]);
                    buf.extend_from_slice(&track.extradata);
                    buf
                };
                c.extend_from_slice(&mp4_box(b"av1C", &av1c_content));
            }
            _ => {}
        }
    }

    if track.color != ColorDesc::UNSPECIFIED {
        c.extend_from_slice(&write_colr(track.color));
    }

    if track.sar.is_valid() && track.sar != Rational::new(1, 1) {
        c.extend_from_slice(&write_pasp(track.sar));
    }

    mp4_box(fourcc, &c)
}

fn write_colr(color: ColorDesc) -> Vec<u8> {
    let mut c = Vec::with_capacity(11);
    c.extend_from_slice(b"nclx");
    c.extend_from_slice(&(color.primaries as u16).to_be_bytes());
    c.extend_from_slice(&(color.transfer as u16).to_be_bytes());
    c.extend_from_slice(&(color.matrix as u16).to_be_bytes());
    let full_range: u8 = if color.range == ColorRange::Full {
        0x80
    } else {
        0x00
    };
    c.push(full_range);
    mp4_box(b"colr", &c)
}

fn write_pasp(sar: Rational) -> Vec<u8> {
    let mut c = Vec::with_capacity(8);
    c.extend_from_slice(&(sar.num.max(1) as u32).to_be_bytes());
    c.extend_from_slice(&(sar.den.max(1) as u32).to_be_bytes());
    mp4_box(b"pasp", &c)
}

fn write_audio_header(track: &TrackState, sample_size: u16) -> Vec<u8> {
    let mut c = Vec::with_capacity(48);
    c.extend_from_slice(&[0u8; 6]);
    c.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    c.extend_from_slice(&[0u8; 8]); // reserved
    c.extend_from_slice(&track.channels.to_be_bytes());
    c.extend_from_slice(&sample_size.to_be_bytes());
    c.extend_from_slice(&0u16.to_be_bytes());
    c.extend_from_slice(&0u16.to_be_bytes());
    c.extend_from_slice(&(track.sample_rate.min(65535) << 16).to_be_bytes());
    c
}

fn write_mp4a_entry(track: &TrackState) -> Vec<u8> {
    let mut c = write_audio_header(track, 16);
    c.extend_from_slice(&write_esds(track));
    mp4_box(b"mp4a", &c)
}

fn write_alac_entry(track: &TrackState) -> Vec<u8> {
    let mut c = write_audio_header(track, track.bits_per_sample.max(16));
    if !track.extradata.is_empty() {
        c.extend_from_slice(&mp4_box(b"alac", &track.extradata));
    }
    mp4_box(b"alac", &c)
}

fn write_opus_entry(track: &TrackState) -> Vec<u8> {
    let mut c = write_audio_header(track, 16);
    if track.extradata.len() >= 19 && &track.extradata[0..8] == b"OpusHead" {
        let mut dops = Vec::with_capacity(11);
        dops.push(0); // Version
        dops.push(track.extradata[9]); // OutputChannelCount
        let pre_skip = u16::from_le_bytes([track.extradata[10], track.extradata[11]]);
        let input_sr = u32::from_le_bytes([
            track.extradata[12],
            track.extradata[13],
            track.extradata[14],
            track.extradata[15],
        ]);
        let out_gain = u16::from_le_bytes([track.extradata[16], track.extradata[17]]);
        dops.extend_from_slice(&pre_skip.to_be_bytes());
        dops.extend_from_slice(&input_sr.to_be_bytes());
        dops.extend_from_slice(&out_gain.to_be_bytes());
        dops.extend_from_slice(&track.extradata[18..]);
        c.extend_from_slice(&mp4_box(b"dOps", &dops));
    } else if !track.extradata.is_empty() {
        c.extend_from_slice(&mp4_box(b"dOps", &track.extradata));
    }
    mp4_box(b"Opus", &c)
}

fn write_flac_entry(track: &TrackState) -> Vec<u8> {
    let mut c = write_audio_header(track, track.bits_per_sample.max(16));
    if !track.extradata.is_empty() {
        c.extend_from_slice(&full_box(b"dfLa", 0, 0, &track.extradata));
    }
    mp4_box(b"fLaC", &c)
}

fn write_ac3_entry(track: &TrackState) -> Vec<u8> {
    let mut c = write_audio_header(track, 16);
    if !track.extradata.is_empty() {
        c.extend_from_slice(&mp4_box(b"dac3", &track.extradata));
    }
    mp4_box(b"ac-3", &c)
}

fn write_pcm_audio_entry(track: &TrackState) -> Vec<u8> {
    let tag = match track.codec_id {
        CodecId::PcmS16be | CodecId::PcmS24be | CodecId::PcmS32be => b"twos",
        CodecId::PcmS16le | CodecId::PcmS24le | CodecId::PcmS32le => b"sowt",
        CodecId::PcmF32le | CodecId::PcmF32be => b"fl32",
        CodecId::PcmF64le | CodecId::PcmF64be => b"fl64",
        CodecId::PcmU8 | CodecId::PcmS8 => b"raw ",
        _ => b"twos",
    };
    let c = write_audio_header(track, track.bits_per_sample.max(16));
    mp4_box(tag, &c)
}

// ---------------------------------------------------------------------------
// MPEG-4 descriptor helpers (for AAC esds)
// ---------------------------------------------------------------------------

fn write_esds(track: &TrackState) -> Vec<u8> {
    let es_descriptor = build_es_descriptor(track);
    full_box(b"esds", 0, 0, &es_descriptor)
}

fn write_descriptor(tag: u8, content: &[u8]) -> Vec<u8> {
    let len = content.len();
    let mut buf = Vec::with_capacity(5 + len);
    buf.push(tag);
    buf.push(((len >> 21) & 0x7F) as u8 | 0x80);
    buf.push(((len >> 14) & 0x7F) as u8 | 0x80);
    buf.push(((len >> 7) & 0x7F) as u8 | 0x80);
    buf.push((len & 0x7F) as u8);
    buf.extend_from_slice(content);
    buf
}

fn build_es_descriptor(track: &TrackState) -> Vec<u8> {
    let dec_specific = write_descriptor(0x05, &track.extradata);

    let mut dec_config_content = Vec::with_capacity(13 + dec_specific.len());
    dec_config_content.push(0x40);
    dec_config_content.push(0x15);
    dec_config_content.extend_from_slice(&[0u8; 3]);
    let max_bitrate = if track.bit_rate > 0 {
        track.bit_rate as u32
    } else {
        0
    };
    dec_config_content.extend_from_slice(&max_bitrate.to_be_bytes());
    dec_config_content.extend_from_slice(&max_bitrate.to_be_bytes());
    dec_config_content.extend_from_slice(&dec_specific);
    let dec_config = write_descriptor(0x04, &dec_config_content);

    let sl_config = write_descriptor(0x06, &[0x02]);

    let mut es_content = Vec::with_capacity(3 + dec_config.len() + sl_config.len());
    es_content.extend_from_slice(&0u16.to_be_bytes());
    es_content.push(0x00);
    es_content.extend_from_slice(&dec_config);
    es_content.extend_from_slice(&sl_config);
    write_descriptor(0x03, &es_content)
}

// ---------------------------------------------------------------------------
// Run-length encoding helpers
// ---------------------------------------------------------------------------

fn rle_encode(values: &[u32]) -> Vec<(u32, u32)> {
    let mut entries = Vec::new();
    if values.is_empty() {
        return entries;
    }
    let mut count = 1u32;
    let mut current = values[0];
    for &v in &values[1..] {
        if v == current {
            count += 1;
        } else {
            entries.push((count, current));
            current = v;
            count = 1;
        }
    }
    entries.push((count, current));
    entries
}

fn rle_encode_i32(values: &[i32]) -> Vec<(u32, i32)> {
    let mut entries = Vec::new();
    if values.is_empty() {
        return entries;
    }
    let mut count = 1u32;
    let mut current = values[0];
    for &v in &values[1..] {
        if v == current {
            count += 1;
        } else {
            entries.push((count, current));
            current = v;
            count = 1;
        }
    }
    entries.push((count, current));
    entries
}
