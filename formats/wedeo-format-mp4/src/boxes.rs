// MP4 box (atom) reading utilities.
//
// Read-side counterpart to atoms.rs (write-side). Parses box headers and
// leaf boxes from a BufferedIo source.
//
// Reference: ISO 14496-12 (ISOBMFF), ISO 14496-14 (MP4 file format).

use wedeo_core::error::{Error, Result};
use wedeo_format::io::BufferedIo;

// ---------------------------------------------------------------------------
// Box header
// ---------------------------------------------------------------------------

/// Parsed box header.
#[derive(Debug, Clone)]
pub struct BoxHeader {
    pub box_type: [u8; 4],
    /// Total box size including header. 0 means "extends to EOF".
    pub size: u64,
    /// Header size: 8 for normal boxes, 16 for extended-size boxes.
    pub header_size: u8,
}

impl BoxHeader {
    /// Payload size (total size minus header). Returns None for "extends to EOF" boxes.
    pub fn payload_size(&self) -> Option<u64> {
        if self.size == 0 {
            None
        } else {
            Some(self.size - self.header_size as u64)
        }
    }
}

/// Read a box header from the current position.
pub fn read_box_header(io: &mut BufferedIo) -> Result<BoxHeader> {
    let size32 = io.read_u32be()?;
    let mut box_type = [0u8; 4];
    io.read_exact(&mut box_type)?;

    let (size, header_size) = if size32 == 1 {
        // Extended size: next 8 bytes are the real size.
        let size64 = io.read_u64be()?;
        (size64, 16u8)
    } else if size32 == 0 {
        // Box extends to end of file.
        (0u64, 8u8)
    } else {
        (size32 as u64, 8u8)
    };

    Ok(BoxHeader {
        box_type,
        size,
        header_size,
    })
}

/// Read a full box header (version + flags) and return (version, flags).
pub fn read_full_box_header(io: &mut BufferedIo) -> Result<(u8, u32)> {
    let version_flags = io.read_u32be()?;
    let version = (version_flags >> 24) as u8;
    let flags = version_flags & 0x00FF_FFFF;
    Ok((version, flags))
}

// ---------------------------------------------------------------------------
// Movie header (mvhd)
// ---------------------------------------------------------------------------

pub struct Mvhd {
    pub timescale: u32,
    pub duration: u64,
}

pub fn parse_mvhd(io: &mut BufferedIo) -> Result<Mvhd> {
    let (version, _flags) = read_full_box_header(io)?;
    if version == 1 {
        io.skip(8)?; // creation_time
        io.skip(8)?; // modification_time
        let timescale = io.read_u32be()?;
        let duration = io.read_u64be()?;
        Ok(Mvhd {
            timescale,
            duration,
        })
    } else {
        io.skip(4)?; // creation_time
        io.skip(4)?; // modification_time
        let timescale = io.read_u32be()?;
        let duration = io.read_u32be()? as u64;
        Ok(Mvhd {
            timescale,
            duration,
        })
    }
}

// ---------------------------------------------------------------------------
// Track header (tkhd)
// ---------------------------------------------------------------------------

pub struct Tkhd {
    pub _track_id: u32,
    pub _duration: u64,
    /// Width in 16.16 fixed-point (shifted right by 16 to get pixels).
    pub width: u32,
    /// Height in 16.16 fixed-point (shifted right by 16 to get pixels).
    pub height: u32,
}

pub fn parse_tkhd(io: &mut BufferedIo) -> Result<Tkhd> {
    let (version, _flags) = read_full_box_header(io)?;
    let (track_id, duration) = if version == 1 {
        io.skip(8)?; // creation_time
        io.skip(8)?; // modification_time
        let track_id = io.read_u32be()?;
        io.skip(4)?; // reserved
        let duration = io.read_u64be()?;
        (track_id, duration)
    } else {
        io.skip(4)?; // creation_time
        io.skip(4)?; // modification_time
        let track_id = io.read_u32be()?;
        io.skip(4)?; // reserved
        let duration = io.read_u32be()? as u64;
        (track_id, duration)
    };
    io.skip(8)?; // reserved
    io.skip(2)?; // layer
    io.skip(2)?; // alternate_group
    io.skip(2)?; // volume
    io.skip(2)?; // reserved
    io.skip(36)?; // matrix
    let width_fp = io.read_u32be()?;
    let height_fp = io.read_u32be()?;
    Ok(Tkhd {
        _track_id: track_id,
        _duration: duration,
        width: width_fp >> 16,
        height: height_fp >> 16,
    })
}

// ---------------------------------------------------------------------------
// Media header (mdhd)
// ---------------------------------------------------------------------------

pub struct Mdhd {
    pub timescale: u32,
    pub duration: u64,
}

pub fn parse_mdhd(io: &mut BufferedIo) -> Result<Mdhd> {
    let (version, _flags) = read_full_box_header(io)?;
    if version == 1 {
        io.skip(8)?; // creation_time
        io.skip(8)?; // modification_time
        let timescale = io.read_u32be()?;
        let duration = io.read_u64be()?;
        Ok(Mdhd {
            timescale,
            duration,
        })
    } else {
        io.skip(4)?; // creation_time
        io.skip(4)?; // modification_time
        let timescale = io.read_u32be()?;
        let duration = io.read_u32be()? as u64;
        Ok(Mdhd {
            timescale,
            duration,
        })
    }
}

// ---------------------------------------------------------------------------
// Handler reference (hdlr)
// ---------------------------------------------------------------------------

pub struct Hdlr {
    pub handler_type: [u8; 4],
}

pub fn parse_hdlr(io: &mut BufferedIo, payload_remaining: u64) -> Result<Hdlr> {
    let (_version, _flags) = read_full_box_header(io)?;
    io.skip(4)?; // pre_defined
    let mut handler_type = [0u8; 4];
    io.read_exact(&mut handler_type)?;
    // Skip the rest (reserved + name)
    let consumed = 4 + 4 + 4; // version_flags + pre_defined + handler_type
    if payload_remaining > consumed {
        io.skip(payload_remaining - consumed)?;
    }
    Ok(Hdlr { handler_type })
}

// ---------------------------------------------------------------------------
// Sample description (stsd) — avc1 + avcC, mp4a + esds
// ---------------------------------------------------------------------------

pub struct StsdEntry {
    pub fourcc: [u8; 4],
    /// For video: width, height
    pub width: u16,
    pub height: u16,
    /// For audio: channel_count, sample_rate (integer part)
    pub channel_count: u16,
    pub sample_rate: u32,
    /// Codec-specific extradata (avcC for H.264, AudioSpecificConfig for AAC)
    pub extradata: Vec<u8>,
}

pub fn parse_stsd(io: &mut BufferedIo, _payload_size: u64) -> Result<StsdEntry> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be()?;
    if entry_count == 0 {
        return Err(Error::InvalidData);
    }

    // Read the first sample entry (we only support single sample description)
    let entry_header = read_box_header(io)?;
    let fourcc = entry_header.box_type;
    let entry_payload = entry_header.payload_size().unwrap_or(0);

    match &fourcc {
        b"avc1" | b"avc3" => parse_avc1_entry(io, fourcc, entry_payload),
        b"mp4a" => parse_mp4a_entry(io, fourcc, entry_payload),
        _ => {
            // Unknown codec — skip the entry
            if entry_payload > 0 {
                io.skip(entry_payload)?;
            }
            Ok(StsdEntry {
                fourcc,
                width: 0,
                height: 0,
                channel_count: 0,
                sample_rate: 0,
                extradata: Vec::new(),
            })
        }
    }
}

fn parse_avc1_entry(io: &mut BufferedIo, fourcc: [u8; 4], payload_size: u64) -> Result<StsdEntry> {
    let start_pos = io.tell()?;

    io.skip(6)?; // reserved
    io.skip(2)?; // data_reference_index
    io.skip(2)?; // pre_defined
    io.skip(2)?; // reserved
    io.skip(12)?; // pre_defined
    let width = io.read_u16be()?;
    let height = io.read_u16be()?;
    io.skip(4)?; // horiz_resolution
    io.skip(4)?; // vert_resolution
    io.skip(4)?; // reserved
    io.skip(2)?; // frame_count
    io.skip(32)?; // compressorname
    io.skip(2)?; // depth
    io.skip(2)?; // pre_defined

    // Parse sub-boxes looking for avcC
    let mut extradata = Vec::new();
    let fixed_fields_size = 6 + 2 + 2 + 2 + 12 + 2 + 2 + 4 + 4 + 4 + 2 + 32 + 2 + 2;
    let remaining = payload_size.saturating_sub(fixed_fields_size);
    let end_pos = start_pos + payload_size;

    let mut consumed = 0u64;
    while consumed + 8 <= remaining {
        let current = io.tell()?;
        if current >= end_pos {
            break;
        }
        let sub = read_box_header(io)?;
        let sub_payload = sub.payload_size().unwrap_or(0);
        if &sub.box_type == b"avcC" {
            extradata = io.read_bytes(sub_payload as usize)?;
            consumed += sub.header_size as u64 + sub_payload;
        } else {
            if sub_payload > 0 {
                io.skip(sub_payload)?;
            }
            consumed += sub.header_size as u64 + sub_payload;
        }
    }

    // Skip any remaining bytes
    let actual_end = io.tell()?;
    if actual_end < end_pos {
        io.skip(end_pos - actual_end)?;
    }

    Ok(StsdEntry {
        fourcc,
        width,
        height,
        channel_count: 0,
        sample_rate: 0,
        extradata,
    })
}

fn parse_mp4a_entry(io: &mut BufferedIo, fourcc: [u8; 4], payload_size: u64) -> Result<StsdEntry> {
    let start_pos = io.tell()?;

    io.skip(6)?; // reserved
    io.skip(2)?; // data_reference_index
    io.skip(8)?; // reserved
    let channel_count = io.read_u16be()?;
    io.skip(2)?; // sample_size
    io.skip(2)?; // pre_defined
    io.skip(2)?; // reserved
    let sample_rate_fp = io.read_u32be()?;
    let sample_rate = sample_rate_fp >> 16;

    // Parse sub-boxes looking for esds
    let mut extradata = Vec::new();
    let end_pos = start_pos + payload_size;

    loop {
        let current = io.tell()?;
        if current + 8 > end_pos {
            break;
        }
        let sub = read_box_header(io)?;
        let sub_payload = sub.payload_size().unwrap_or(0);
        if &sub.box_type == b"esds" {
            extradata = parse_esds_extradata(io, sub_payload)?;
        } else if sub_payload > 0 {
            io.skip(sub_payload)?;
        }
    }

    // Skip any remaining bytes
    let actual_end = io.tell()?;
    if actual_end < end_pos {
        io.skip(end_pos - actual_end)?;
    }

    Ok(StsdEntry {
        fourcc,
        width: 0,
        height: 0,
        channel_count,
        sample_rate,
        extradata,
    })
}

/// Parse the esds box to extract the AudioSpecificConfig (DecoderSpecificInfo).
fn parse_esds_extradata(io: &mut BufferedIo, payload_size: u64) -> Result<Vec<u8>> {
    let start_pos = io.tell()?;
    let end_pos = start_pos + payload_size;

    let (_version, _flags) = read_full_box_header(io)?;

    // ES_Descriptor (tag 0x03)
    let tag = io.read_u8()?;
    if tag != 0x03 {
        let current = io.tell()?;
        io.skip(end_pos.saturating_sub(current))?;
        return Ok(Vec::new());
    }
    let _es_len = read_descriptor_length(io)?;
    io.skip(2)?; // ES_ID
    io.skip(1)?; // flags

    // DecoderConfigDescriptor (tag 0x04)
    let tag = io.read_u8()?;
    if tag != 0x04 {
        let current = io.tell()?;
        io.skip(end_pos.saturating_sub(current))?;
        return Ok(Vec::new());
    }
    let _dec_config_len = read_descriptor_length(io)?;
    io.skip(1)?; // objectTypeIndication
    io.skip(1)?; // streamType
    io.skip(3)?; // bufferSizeDB
    io.skip(4)?; // maxBitrate
    io.skip(4)?; // avgBitrate

    // DecoderSpecificInfo (tag 0x05) — this is the AudioSpecificConfig
    let current = io.tell()?;
    if current >= end_pos {
        return Ok(Vec::new());
    }
    let tag = io.read_u8()?;
    if tag != 0x05 {
        let current = io.tell()?;
        io.skip(end_pos.saturating_sub(current))?;
        return Ok(Vec::new());
    }
    let asc_len = read_descriptor_length(io)?;
    let extradata = io.read_bytes(asc_len as usize)?;

    // Skip remaining
    let actual_end = io.tell()?;
    if actual_end < end_pos {
        io.skip(end_pos - actual_end)?;
    }

    Ok(extradata)
}

/// Read MPEG-4 descriptor variable-length encoding.
/// Each byte contributes 7 bits; MSB=1 means more bytes follow.
fn read_descriptor_length(io: &mut BufferedIo) -> Result<u32> {
    let mut len = 0u32;
    for _ in 0..4 {
        let b = io.read_u8()?;
        len = (len << 7) | (b & 0x7F) as u32;
        if (b & 0x80) == 0 {
            break;
        }
    }
    Ok(len)
}

// ---------------------------------------------------------------------------
// Sample table boxes
// ---------------------------------------------------------------------------

/// stts — decoding time-to-sample: Vec<(sample_count, sample_delta)>
pub fn parse_stts(io: &mut BufferedIo) -> Result<Vec<(u32, u32)>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be()? as usize;
    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let count = io.read_u32be()?;
        let delta = io.read_u32be()?;
        entries.push((count, delta));
    }
    Ok(entries)
}

/// stsc — sample-to-chunk: Vec<(first_chunk, samples_per_chunk, sample_desc_index)>
pub fn parse_stsc(io: &mut BufferedIo) -> Result<Vec<(u32, u32, u32)>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be()? as usize;
    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let first_chunk = io.read_u32be()?;
        let samples_per_chunk = io.read_u32be()?;
        let desc_index = io.read_u32be()?;
        entries.push((first_chunk, samples_per_chunk, desc_index));
    }
    Ok(entries)
}

/// stsz — sample sizes. Returns (fixed_size, per_sample_sizes).
/// If fixed_size > 0, all samples have that size and per_sample_sizes is empty.
pub fn parse_stsz(io: &mut BufferedIo) -> Result<(u32, Vec<u32>)> {
    let (_version, _flags) = read_full_box_header(io)?;
    let sample_size = io.read_u32be()?;
    let sample_count = io.read_u32be()? as usize;
    if sample_size != 0 {
        // Fixed-size samples — no per-sample table
        return Ok((sample_size, Vec::new()));
    }
    let mut sizes = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        sizes.push(io.read_u32be()?);
    }
    Ok((0, sizes))
}

/// stco — chunk offsets (32-bit).
pub fn parse_stco(io: &mut BufferedIo) -> Result<Vec<u64>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be()? as usize;
    let mut offsets = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        offsets.push(io.read_u32be()? as u64);
    }
    Ok(offsets)
}

/// co64 — chunk offsets (64-bit).
pub fn parse_co64(io: &mut BufferedIo) -> Result<Vec<u64>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be()? as usize;
    let mut offsets = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        offsets.push(io.read_u64be()?);
    }
    Ok(offsets)
}

/// stss — sync sample table (keyframe indices, 1-indexed).
pub fn parse_stss(io: &mut BufferedIo) -> Result<Vec<u32>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be()? as usize;
    let mut indices = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        indices.push(io.read_u32be()?);
    }
    Ok(indices)
}

/// ctts — composition time offsets: Vec<(sample_count, sample_offset)>.
/// Version 0: offsets are unsigned (stored as u32).
/// Version 1: offsets are signed (stored as i32).
pub fn parse_ctts(io: &mut BufferedIo) -> Result<Vec<(u32, i32)>> {
    let (version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be()? as usize;
    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let count = io.read_u32be()?;
        let offset = if version == 0 {
            // Version 0: unsigned offset
            io.read_u32be()? as i32
        } else {
            // Version 1: signed offset
            let raw = io.read_u32be()?;
            raw as i32
        };
        entries.push((count, offset));
    }
    Ok(entries)
}
