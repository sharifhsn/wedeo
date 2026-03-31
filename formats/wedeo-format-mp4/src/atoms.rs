// MP4 box (atom) building utilities.
//
// All boxes are built in memory as Vec<u8>, then written to I/O in one shot.
// This is the standard approach for two-pass muxing where moov is written last.

use wedeo_core::codec_id::CodecId;
use wedeo_core::media_type::MediaType;

use crate::muxer::TrackState;

// ---------------------------------------------------------------------------
// Core box primitives
// ---------------------------------------------------------------------------

/// Build a box: 4-byte size (big-endian) + 4-byte type + content.
fn mp4_box(box_type: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let size = (8 + content.len()) as u32;
    let mut buf = Vec::with_capacity(size as usize);
    buf.extend_from_slice(&size.to_be_bytes());
    buf.extend_from_slice(box_type);
    buf.extend_from_slice(content);
    buf
}

/// Build a full box: size + type + version(1) + flags(3) + content.
fn full_box(box_type: &[u8; 4], version: u8, flags: u32, content: &[u8]) -> Vec<u8> {
    let mut inner = Vec::with_capacity(4 + content.len());
    let version_flags = ((version as u32) << 24) | (flags & 0x00FF_FFFF);
    inner.extend_from_slice(&version_flags.to_be_bytes());
    inner.extend_from_slice(content);
    mp4_box(box_type, &inner)
}

// ---------------------------------------------------------------------------
// Identity matrix (used in mvhd and tkhd)
// ---------------------------------------------------------------------------

/// 3x3 identity matrix in 16.16 / 2.30 fixed-point, 36 bytes.
/// [ 0x00010000  0           0
///   0           0x00010000  0
///   0           0           0x40000000 ]
const IDENTITY_MATRIX: [u8; 36] = {
    let mut m = [0u8; 36];
    // a = 1.0 in 16.16
    m[0] = 0x00;
    m[1] = 0x01;
    m[2] = 0x00;
    m[3] = 0x00;
    // e = 1.0 in 16.16
    m[16] = 0x00;
    m[17] = 0x01;
    m[18] = 0x00;
    m[19] = 0x00;
    // i = 1.0 in 2.30
    m[32] = 0x40;
    m[33] = 0x00;
    m[34] = 0x00;
    m[35] = 0x00;
    m
};

// ---------------------------------------------------------------------------
// Top-level boxes
// ---------------------------------------------------------------------------

/// Build ftyp box. Audio-only files use M4A branding.
pub fn write_ftyp(has_video: bool, has_audio: bool) -> Vec<u8> {
    let mut content = Vec::with_capacity(24);
    // major_brand
    if has_audio && !has_video {
        content.extend_from_slice(b"M4A ");
    } else {
        content.extend_from_slice(b"isom");
    }
    // minor_version
    content.extend_from_slice(&0x200u32.to_be_bytes());
    // compatible_brands
    content.extend_from_slice(b"isom");
    content.extend_from_slice(b"iso2");
    if has_video {
        content.extend_from_slice(b"avc1");
    }
    if has_audio && !has_video {
        content.extend_from_slice(b"M4A ");
    }
    content.extend_from_slice(b"mp41");
    mp4_box(b"ftyp", &content)
}

/// Convert a duration from media timescale to movie timescale with rounding.
fn rescale_duration(duration_ts: u64, media_timescale: u32, movie_timescale: u32) -> u64 {
    if media_timescale == 0 {
        return 0;
    }
    let ts = media_timescale as u64;
    (duration_ts * movie_timescale as u64 + ts / 2) / ts
}

/// Build the complete moov box for all tracks.
pub fn write_moov(tracks: &[TrackState], movie_timescale: u32) -> Vec<u8> {
    // Compute movie duration (max of all track durations, scaled to movie timescale)
    let movie_duration: u64 = tracks
        .iter()
        .map(|t| {
            if t.timescale == 0 || t.sample_count == 0 {
                return 0u64;
            }
            rescale_duration(t.duration_ts, t.timescale, movie_timescale)
        })
        .max()
        .unwrap_or(0);

    let next_track_id = tracks.len() as u32 + 1;
    let mvhd = write_mvhd(movie_timescale, movie_duration, next_track_id);

    let mut moov_content = Vec::new();
    moov_content.extend_from_slice(&mvhd);
    for (i, track) in tracks.iter().enumerate() {
        let track_id = i as u32 + 1;
        let track_duration_movie = if track.timescale > 0 && track.sample_count > 0 {
            rescale_duration(track.duration_ts, track.timescale, movie_timescale)
        } else {
            0
        };
        let trak = write_trak(track, track_id, track_duration_movie, movie_timescale);
        moov_content.extend_from_slice(&trak);
    }

    mp4_box(b"moov", &moov_content)
}

// ---------------------------------------------------------------------------
// Movie-level boxes
// ---------------------------------------------------------------------------

fn write_mvhd(timescale: u32, duration: u64, next_track_id: u32) -> Vec<u8> {
    let version = if duration > u32::MAX as u64 { 1 } else { 0 };
    let mut c = Vec::with_capacity(108);
    if version == 1 {
        c.extend_from_slice(&0u64.to_be_bytes()); // creation_time
        c.extend_from_slice(&0u64.to_be_bytes()); // modification_time
        c.extend_from_slice(&timescale.to_be_bytes());
        c.extend_from_slice(&duration.to_be_bytes());
    } else {
        c.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        c.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        c.extend_from_slice(&timescale.to_be_bytes());
        c.extend_from_slice(&(duration as u32).to_be_bytes());
    }
    c.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate = 1.0
    c.extend_from_slice(&0x0100u16.to_be_bytes()); // volume = 1.0
    c.extend_from_slice(&[0u8; 10]); // reserved
    c.extend_from_slice(&IDENTITY_MATRIX);
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
) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&write_tkhd(track, track_id, duration_movie));
    content.extend_from_slice(&write_mdia(track, movie_timescale));
    mp4_box(b"trak", &content)
}

fn write_tkhd(track: &TrackState, track_id: u32, duration_movie: u64) -> Vec<u8> {
    let version = if duration_movie > u32::MAX as u64 {
        1
    } else {
        0
    };
    let mut c = Vec::with_capacity(92);
    if version == 1 {
        c.extend_from_slice(&0u64.to_be_bytes()); // creation_time
        c.extend_from_slice(&0u64.to_be_bytes()); // modification_time
        c.extend_from_slice(&track_id.to_be_bytes());
        c.extend_from_slice(&0u32.to_be_bytes()); // reserved
        c.extend_from_slice(&duration_movie.to_be_bytes());
    } else {
        c.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        c.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        c.extend_from_slice(&track_id.to_be_bytes());
        c.extend_from_slice(&0u32.to_be_bytes()); // reserved
        c.extend_from_slice(&(duration_movie as u32).to_be_bytes());
    }
    c.extend_from_slice(&[0u8; 8]); // reserved
    c.extend_from_slice(&0u16.to_be_bytes()); // layer
    c.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
    // volume: 0x0100 for audio, 0 for video
    let volume: u16 = if track.media_type == MediaType::Audio {
        0x0100
    } else {
        0
    };
    c.extend_from_slice(&volume.to_be_bytes());
    c.extend_from_slice(&0u16.to_be_bytes()); // reserved
    c.extend_from_slice(&IDENTITY_MATRIX);
    // width/height in 16.16 fixed-point
    let width_fp = track.width << 16;
    let height_fp = track.height << 16;
    c.extend_from_slice(&width_fp.to_be_bytes());
    c.extend_from_slice(&height_fp.to_be_bytes());
    // flags = 0x03 (track_enabled | track_in_movie)
    full_box(b"tkhd", version, 0x03, &c)
}

// ---------------------------------------------------------------------------
// Media boxes
// ---------------------------------------------------------------------------

fn write_mdia(track: &TrackState, _movie_timescale: u32) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&write_mdhd(track));
    content.extend_from_slice(&write_hdlr(track));
    content.extend_from_slice(&write_minf(track));
    mp4_box(b"mdia", &content)
}

fn write_mdhd(track: &TrackState) -> Vec<u8> {
    let version = if track.duration_ts > u32::MAX as u64 {
        1
    } else {
        0
    };
    let mut c = Vec::with_capacity(32);
    if version == 1 {
        c.extend_from_slice(&0u64.to_be_bytes()); // creation_time
        c.extend_from_slice(&0u64.to_be_bytes()); // modification_time
        c.extend_from_slice(&track.timescale.to_be_bytes());
        c.extend_from_slice(&track.duration_ts.to_be_bytes());
    } else {
        c.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        c.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        c.extend_from_slice(&track.timescale.to_be_bytes());
        c.extend_from_slice(&(track.duration_ts as u32).to_be_bytes());
    }
    c.extend_from_slice(&0x55C4u16.to_be_bytes()); // language = undetermined
    c.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    full_box(b"mdhd", version, 0, &c)
}

fn write_hdlr(track: &TrackState) -> Vec<u8> {
    let (handler_type, name) = match track.media_type {
        MediaType::Video => (b"vide", b"VideoHandler\0" as &[u8]),
        MediaType::Audio => (b"soun", b"SoundHandler\0" as &[u8]),
        _ => (b"data", b"DataHandler\0" as &[u8]),
    };
    let mut c = Vec::with_capacity(24 + name.len());
    c.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
    c.extend_from_slice(handler_type);
    c.extend_from_slice(&[0u8; 12]); // reserved
    c.extend_from_slice(name);
    full_box(b"hdlr", 0, 0, &c)
}

// ---------------------------------------------------------------------------
// Media information boxes
// ---------------------------------------------------------------------------

fn write_minf(track: &TrackState) -> Vec<u8> {
    let mut content = Vec::new();
    match track.media_type {
        MediaType::Video => content.extend_from_slice(&write_vmhd()),
        MediaType::Audio => content.extend_from_slice(&write_smhd()),
        _ => {}
    }
    content.extend_from_slice(&write_dinf());
    content.extend_from_slice(&write_stbl(track));
    mp4_box(b"minf", &content)
}

fn write_vmhd() -> Vec<u8> {
    let mut c = Vec::with_capacity(8);
    c.extend_from_slice(&0u16.to_be_bytes()); // graphicsmode
    c.extend_from_slice(&[0u8; 6]); // opcolor
    // flags = 1 per spec
    full_box(b"vmhd", 0, 1, &c)
}

fn write_smhd() -> Vec<u8> {
    let mut c = Vec::with_capacity(4);
    c.extend_from_slice(&0u16.to_be_bytes()); // balance
    c.extend_from_slice(&0u16.to_be_bytes()); // reserved
    full_box(b"smhd", 0, 0, &c)
}

fn write_dinf() -> Vec<u8> {
    // dref with one self-contained url entry
    let url_box = full_box(b"url ", 0, 0x01, &[]); // flag 1 = self-contained
    let mut dref_content = Vec::with_capacity(4 + url_box.len());
    dref_content.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    dref_content.extend_from_slice(&url_box);
    let dref = full_box(b"dref", 0, 0, &dref_content);
    mp4_box(b"dinf", &dref)
}

// ---------------------------------------------------------------------------
// Sample table boxes
// ---------------------------------------------------------------------------

fn write_stbl(track: &TrackState) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&write_stsd(track));
    content.extend_from_slice(&write_stts(track));
    if track.has_cts {
        content.extend_from_slice(&write_ctts(track));
    }
    content.extend_from_slice(&write_stsc(track));
    content.extend_from_slice(&write_stsz(track));
    content.extend_from_slice(&write_chunk_offsets(track));
    if track.media_type == MediaType::Video && !track.sync_samples.is_empty() {
        // Only write stss if not all samples are sync (otherwise it's implied)
        if track.sync_samples.len() != track.sample_count as usize {
            content.extend_from_slice(&write_stss(track));
        }
    }
    mp4_box(b"stbl", &content)
}

fn write_stsd(track: &TrackState) -> Vec<u8> {
    let entry = match track.codec_id {
        CodecId::H264 => write_avc1_entry(track),
        CodecId::Av1 => write_av01_entry(track),
        CodecId::Aac => write_mp4a_entry(track),
        _ => write_generic_audio_entry(track),
    };
    let mut c = Vec::with_capacity(4 + entry.len());
    c.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    c.extend_from_slice(&entry);
    full_box(b"stsd", 0, 0, &c)
}

/// stts — decoding time-to-sample (run-length encoded durations).
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

/// ctts — composition time offsets (run-length encoded).
fn write_ctts(track: &TrackState) -> Vec<u8> {
    let entries = rle_encode_i32(&track.cts_offsets);
    let mut c = Vec::with_capacity(4 + entries.len() * 8);
    c.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for &(count, offset) in &entries {
        c.extend_from_slice(&count.to_be_bytes());
        c.extend_from_slice(&(offset as u32).to_be_bytes());
    }
    // version 1 allows negative offsets
    let version = if track.cts_offsets.iter().any(|&o| o < 0) {
        1
    } else {
        0
    };
    full_box(b"ctts", version, 0, &c)
}

/// stsc — sample-to-chunk mapping.
/// We use one sample per chunk, so there's a single entry: (1, 1, 1).
fn write_stsc(_track: &TrackState) -> Vec<u8> {
    let mut c = Vec::with_capacity(16);
    c.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    c.extend_from_slice(&1u32.to_be_bytes()); // first_chunk (1-indexed)
    c.extend_from_slice(&1u32.to_be_bytes()); // samples_per_chunk
    c.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
    full_box(b"stsc", 0, 0, &c)
}

/// stsz — sample sizes.
fn write_stsz(track: &TrackState) -> Vec<u8> {
    let mut c = Vec::with_capacity(8 + track.sample_sizes.len() * 4);
    c.extend_from_slice(&0u32.to_be_bytes()); // sample_size = 0 (variable)
    c.extend_from_slice(&track.sample_count.to_be_bytes());
    for &size in &track.sample_sizes {
        c.extend_from_slice(&size.to_be_bytes());
    }
    full_box(b"stsz", 0, 0, &c)
}

/// stco or co64 — chunk offsets.
fn write_chunk_offsets(track: &TrackState) -> Vec<u8> {
    let needs_64bit = track.chunk_offsets.iter().any(|&o| o > u32::MAX as u64);
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

/// stss — sync sample table (keyframes).
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

/// avc1 sample entry for H.264.
fn write_avc1_entry(track: &TrackState) -> Vec<u8> {
    let mut c = Vec::with_capacity(86 + track.extradata.len());
    c.extend_from_slice(&[0u8; 6]); // reserved
    c.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    c.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    c.extend_from_slice(&0u16.to_be_bytes()); // reserved
    c.extend_from_slice(&[0u8; 12]); // pre_defined
    c.extend_from_slice(&(track.width as u16).to_be_bytes());
    c.extend_from_slice(&(track.height as u16).to_be_bytes());
    c.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horiz resolution 72 dpi
    c.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vert resolution 72 dpi
    c.extend_from_slice(&0u32.to_be_bytes()); // reserved
    c.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    c.extend_from_slice(&[0u8; 32]); // compressorname
    c.extend_from_slice(&0x0018u16.to_be_bytes()); // depth = 24
    c.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined = -1

    // avcC box (raw extradata is already in avcC format)
    if !track.extradata.is_empty() {
        c.extend_from_slice(&mp4_box(b"avcC", &track.extradata));
    }

    mp4_box(b"avc1", &c)
}

/// av01 sample entry for AV1.
fn write_av01_entry(track: &TrackState) -> Vec<u8> {
    let mut c = Vec::with_capacity(86 + 12 + track.extradata.len());
    c.extend_from_slice(&[0u8; 6]); // reserved
    c.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    c.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    c.extend_from_slice(&0u16.to_be_bytes()); // reserved
    c.extend_from_slice(&[0u8; 12]); // pre_defined
    c.extend_from_slice(&(track.width as u16).to_be_bytes());
    c.extend_from_slice(&(track.height as u16).to_be_bytes());
    c.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horiz resolution 72 dpi
    c.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vert resolution 72 dpi
    c.extend_from_slice(&0u32.to_be_bytes()); // reserved
    c.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    c.extend_from_slice(&[0u8; 32]); // compressorname
    c.extend_from_slice(&0x0018u16.to_be_bytes()); // depth = 24
    c.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined = -1

    // av1C box: if extradata starts with the av1C marker byte (0x81),
    // it's a full AV1CodecConfigurationRecord (from rav1e's
    // container_sequence_header). Otherwise it's raw configOBUs
    // (from the demuxer, which strips the 4-byte header), and we
    // prepend default header bytes for 8-bit 420p Profile 0.
    if !track.extradata.is_empty() {
        let av1c_content = if track.extradata.len() >= 4 && track.extradata[0] == 0x81 {
            // Already a full AV1CodecConfigurationRecord
            track.extradata.clone()
        } else {
            // Raw configOBUs — prepend av1C header for 8-bit 420p Profile 0
            let mut buf = Vec::with_capacity(4 + track.extradata.len());
            buf.extend_from_slice(&[0x81, 0x04, 0x0C, 0x00]);
            buf.extend_from_slice(&track.extradata);
            buf
        };
        c.extend_from_slice(&mp4_box(b"av1C", &av1c_content));
    }

    mp4_box(b"av01", &c)
}

/// mp4a sample entry for AAC.
fn write_mp4a_entry(track: &TrackState) -> Vec<u8> {
    let mut c = Vec::with_capacity(36 + 64);
    c.extend_from_slice(&[0u8; 6]); // reserved
    c.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    c.extend_from_slice(&[0u8; 8]); // reserved
    c.extend_from_slice(&track.channels.to_be_bytes());
    c.extend_from_slice(&16u16.to_be_bytes()); // sample_size = 16
    c.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    c.extend_from_slice(&0u16.to_be_bytes()); // reserved
    // sample_rate in 16.16 fixed point (cap at 65535 to avoid overflow)
    c.extend_from_slice(&(track.sample_rate.min(65535) << 16).to_be_bytes());

    // esds box
    c.extend_from_slice(&write_esds(track));

    mp4_box(b"mp4a", &c)
}

/// Generic audio sample entry (for PCM codecs in MP4 — twos/sowt/lpcm).
fn write_generic_audio_entry(track: &TrackState) -> Vec<u8> {
    // Use 'twos' for big-endian PCM, 'sowt' for little-endian, 'fl32'/'fl64' by bit depth
    let tag = match track.codec_id {
        CodecId::PcmS16be | CodecId::PcmS24be | CodecId::PcmS32be => b"twos",
        CodecId::PcmS16le | CodecId::PcmS24le | CodecId::PcmS32le => b"sowt",
        CodecId::PcmF32le | CodecId::PcmF32be => b"fl32",
        CodecId::PcmF64le | CodecId::PcmF64be => b"fl64",
        _ => b"twos", // fallback
    };

    let mut c = Vec::with_capacity(28);
    c.extend_from_slice(&[0u8; 6]); // reserved
    c.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    c.extend_from_slice(&[0u8; 8]); // reserved
    c.extend_from_slice(&track.channels.to_be_bytes());
    c.extend_from_slice(&track.bits_per_sample.to_be_bytes());
    c.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    c.extend_from_slice(&0u16.to_be_bytes()); // reserved
    // sample_rate in 16.16 fixed point (cap at 65535 to avoid overflow)
    c.extend_from_slice(&(track.sample_rate.min(65535) << 16).to_be_bytes());

    mp4_box(tag, &c)
}

// ---------------------------------------------------------------------------
// MPEG-4 descriptor helpers (for AAC esds)
// ---------------------------------------------------------------------------

/// Build an esds full box containing the ES_Descriptor for AAC.
fn write_esds(track: &TrackState) -> Vec<u8> {
    let es_descriptor = build_es_descriptor(track);
    full_box(b"esds", 0, 0, &es_descriptor)
}

/// Write a descriptor tag + 4-byte length prefix (matching FFmpeg's encoding).
fn write_descriptor(tag: u8, content: &[u8]) -> Vec<u8> {
    let len = content.len();
    let mut buf = Vec::with_capacity(5 + len);
    buf.push(tag);
    // 4-byte variable-length encoding (matches FFmpeg)
    buf.push(((len >> 21) & 0x7F) as u8 | 0x80);
    buf.push(((len >> 14) & 0x7F) as u8 | 0x80);
    buf.push(((len >> 7) & 0x7F) as u8 | 0x80);
    buf.push((len & 0x7F) as u8);
    buf.extend_from_slice(content);
    buf
}

fn build_es_descriptor(track: &TrackState) -> Vec<u8> {
    // DecoderSpecificInfo (tag 0x05) — AudioSpecificConfig
    let dec_specific = write_descriptor(0x05, &track.extradata);

    // DecoderConfigDescriptor (tag 0x04)
    let mut dec_config_content = Vec::with_capacity(13 + dec_specific.len());
    dec_config_content.push(0x40); // objectTypeIndication = Audio ISO/IEC 14496-3
    dec_config_content.push(0x15); // streamType = AudioStream(5) << 2 | upstream(0) << 1 | reserved(1)
    dec_config_content.extend_from_slice(&[0u8; 3]); // bufferSizeDB (24-bit)
    let max_bitrate = if track.bit_rate > 0 {
        track.bit_rate as u32
    } else {
        0
    };
    dec_config_content.extend_from_slice(&max_bitrate.to_be_bytes());
    dec_config_content.extend_from_slice(&max_bitrate.to_be_bytes()); // avg = max
    dec_config_content.extend_from_slice(&dec_specific);
    let dec_config = write_descriptor(0x04, &dec_config_content);

    // SLConfigDescriptor (tag 0x06)
    let sl_config = write_descriptor(0x06, &[0x02]); // predefined = 2

    // ES_Descriptor (tag 0x03)
    let mut es_content = Vec::with_capacity(3 + dec_config.len() + sl_config.len());
    es_content.extend_from_slice(&0u16.to_be_bytes()); // ES_ID
    es_content.push(0x00); // flags (no streamDependence, no URL, no OCRstream)
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
