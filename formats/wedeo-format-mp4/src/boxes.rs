//! ISOBMFF / MP4 / MOV box parsing primitives and shared box AST (`Mp4BoxNode`) for `wedeo inspect`.
//!
//! Reference: ISO/IEC 14496-12 (ISOBMFF), ISO/IEC 14496-14 (MP4), ISO/IEC 14496-15 (AVC/HEVC),
//! ISO/IEC 23000-19 (CMAF), and QuickTime File Format Specification.

use std::io;

use wedeo_core::{
    BufByteSource, Buffer, ChromaLocation, ColorDesc, ColorPrimaries, ColorRange, EditSegment,
    Error, MasteringDisplayMeta, MatrixCoefficients, MemorySource, Rational, Result, SideData,
    Timestamp, TransferCharacteristic, reduce, rescale,
};

/// 3x3 identity display matrix in 16.16 / 2.30 fixed-point (`[i32; 9]`).
pub const IDENTITY_MATRIX_I32: [i32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

/// 3x3 identity display matrix serialized as 36 big-endian bytes.
pub const IDENTITY_MATRIX_BYTES: [u8; 36] = {
    let mut m = [0u8; 36];
    m[1] = 0x01;
    m[17] = 0x01;
    m[32] = 0x40;
    m
};

/// Compute clockwise display rotation in degrees (`0`, `90`, `180`, `270`) from a 3x3 `tkhd` matrix.
///
/// Matches FFmpeg's `av_display_rotation_get` convention.
pub fn matrix_rotation_degrees(matrix: &[i32; 9]) -> i32 {
    let a = matrix[0] as f64 / 65536.0;
    let b = matrix[1] as f64 / 65536.0;
    let c = matrix[3] as f64 / 65536.0;
    let d = matrix[4] as f64 / 65536.0;

    let scale0 = (a * a + c * c).sqrt();
    let scale1 = (b * b + d * d).sqrt();
    if scale0 == 0.0 || scale1 == 0.0 {
        return 0;
    }
    let angle = (b / scale1).atan2(a / scale0).to_degrees().round() as i32;
    angle.rem_euclid(360)
}

/// Build a 3x3 `tkhd` fixed-point display matrix for a clockwise rotation (`0`, `90`, `180`, `270`).
pub const fn rotation_degrees_to_matrix(degrees: i32) -> [i32; 9] {
    match degrees.rem_euclid(360) {
        90 => [0, 0x0001_0000, 0, -0x0001_0000, 0, 0, 0, 0, 0x4000_0000],
        180 => [-0x0001_0000, 0, 0, 0, -0x0001_0000, 0, 0, 0, 0x4000_0000],
        270 => [0, -0x0001_0000, 0, 0x0001_0000, 0, 0, 0, 0, 0x4000_0000],
        _ => IDENTITY_MATRIX_I32,
    }
}

/// Parsed ISOBMFF box header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoxHeader {
    pub box_type: [u8; 4],
    /// Total box size in bytes including header (`0` means "extends to EOF").
    pub size: u64,
    /// Header size in bytes (`8` for standard boxes, `16` for 64-bit `largesize` boxes).
    pub header_size: u8,
}

impl BoxHeader {
    /// Payload size in bytes (`size - header_size`), or `None` when `size == 0` (extends to EOF).
    #[inline]
    pub const fn payload_size(&self) -> Option<u64> {
        if self.size == 0 {
            None
        } else {
            Some(self.size.saturating_sub(self.header_size as u64))
        }
    }

    /// Printable 4-character box type name.
    pub fn type_name(&self) -> String {
        fourcc_to_string(&self.box_type)
    }
}

/// Convert a 4-byte FourCC tag to a printable ASCII string.
pub fn fourcc_to_string(tag: &[u8; 4]) -> String {
    let mut s = String::with_capacity(4);
    for &b in tag {
        if (0x20..=0x7E).contains(&b) {
            s.push(b as char);
        } else {
            s.push('?');
        }
    }
    s
}

/// Try reading the next box header, returning `Ok(None)` on clean EOF at box boundary.
pub fn try_read_box_header(io: &mut BufByteSource) -> Result<Option<BoxHeader>> {
    let b0 = match io.read_u8() {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(Error::io("mp4/boxes", &e)),
    };
    let mut rest = [0u8; 7];
    io.read_exact_bytes(&mut rest)
        .map_err(|e| Error::io("mp4/boxes", &e))?;

    let size32 = u32::from_be_bytes([b0, rest[0], rest[1], rest[2]]);
    let box_type = [rest[3], rest[4], rest[5], rest[6]];

    let (size, header_size) = if size32 == 1 {
        let size64 = io.read_u64be().map_err(|e| Error::io("mp4/boxes", &e))?;
        if size64 < 16 {
            return Err(Error::invalid_data(
                "mp4/boxes",
                "extended box size smaller than 16-byte header",
            ));
        }
        (size64, 16u8)
    } else if size32 == 0 {
        (0u64, 8u8)
    } else if size32 < 8 {
        return Err(Error::invalid_data(
            "mp4/boxes",
            "box size smaller than 8-byte header",
        ));
    } else {
        (u64::from(size32), 8u8)
    };

    Ok(Some(BoxHeader {
        box_type,
        size,
        header_size,
    }))
}

/// Read a box header from the current stream position.
pub fn read_box_header(io: &mut BufByteSource) -> Result<BoxHeader> {
    try_read_box_header(io)?
        .ok_or_else(|| Error::io("mp4/boxes", &io::Error::from(io::ErrorKind::UnexpectedEof)))
}

/// Read a FullBox header (`version: u8`, `flags: u24`).
pub fn read_full_box_header(io: &mut BufByteSource) -> Result<(u8, u32)> {
    let vf = io.read_u32be().map_err(|e| Error::io("mp4/boxes", &e))?;
    Ok(((vf >> 24) as u8, vf & 0x00FF_FFFF))
}

/// Read `len` bytes into a `Vec<u8>`.
pub fn read_bytes_vec(io: &mut BufByteSource, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    io.read_exact_bytes(&mut buf)
        .map_err(|e| Error::io("mp4/boxes", &e))?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Shared Box AST for `wedeo inspect`
// ---------------------------------------------------------------------------

/// One node in an inspected ISOBMFF / MP4 box hierarchy (`wedeo inspect`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mp4BoxNode {
    pub box_type: [u8; 4],
    pub offset: u64,
    pub size: u64,
    pub header_size: u8,
    pub summary: Option<String>,
    pub children: Vec<Mp4BoxNode>,
}

impl Mp4BoxNode {
    /// FourCC type as printable string (e.g. `"moov"`, `"trak"`, `"avc1"`).
    pub fn type_name(&self) -> String {
        fourcc_to_string(&self.box_type)
    }

    /// Find the first direct child matching `box_type`.
    pub fn child(&self, box_type: &[u8; 4]) -> Option<&Self> {
        self.children.iter().find(|c| &c.box_type == box_type)
    }
}

fn is_container_box(box_type: &[u8; 4]) -> bool {
    matches!(
        box_type,
        b"moov"
            | b"trak"
            | b"edts"
            | b"mdia"
            | b"minf"
            | b"dinf"
            | b"stbl"
            | b"mvex"
            | b"moof"
            | b"traf"
            | b"mfra"
            | b"udta"
            | b"Wave"
            | b"wave"
    )
}

/// Alias for [`Mp4BoxNode`] exposed for `wedeo inspect`.
pub type BoxNode = Mp4BoxNode;

/// Inspect an MP4/MOV byte slice and return its hierarchical box AST.
pub fn inspect_boxes_from_slice(data: &[u8]) -> Result<Vec<Mp4BoxNode>> {
    let mut io = BufByteSource::new(Box::new(MemorySource::from_slice(data)));
    inspect_boxes(&mut io)
}

/// Inspect an MP4/MOV [`BufByteSource`] and return its hierarchical box AST.
pub fn inspect_boxes(io: &mut BufByteSource) -> Result<Vec<Mp4BoxNode>> {
    let end = io.size_hint().unwrap_or(u64::MAX);
    inspect_box_range(io, end)
}

/// Parse an ISOBMFF/MP4 [`BufByteSource`] into its hierarchical box AST (`BoxNode` tree).
pub fn parse_box_tree(io: &mut BufByteSource) -> Result<Vec<BoxNode>> {
    inspect_boxes(io)
}

/// Parse an ISOBMFF/MP4 byte slice into its hierarchical box AST (`BoxNode` tree).
pub fn parse_box_tree_from_slice(data: &[u8]) -> Result<Vec<BoxNode>> {
    inspect_boxes_from_slice(data)
}

fn inspect_box_range(io: &mut BufByteSource, range_end: u64) -> Result<Vec<Mp4BoxNode>> {
    let mut nodes = Vec::new();
    while io.tell() + 8 <= range_end {
        let offset = io.tell();
        let Some(hdr) = try_read_box_header(io)? else {
            break;
        };
        let box_end = if hdr.size == 0 {
            range_end
        } else {
            offset.saturating_add(hdr.size).min(range_end)
        };
        let payload_len = box_end.saturating_sub(io.tell());

        let mut summary = None;
        let mut children = Vec::new();

        if is_container_box(&hdr.box_type) {
            children = inspect_box_range(io, box_end)?;
        } else if &hdr.box_type == b"ftyp" && payload_len >= 8 {
            let major = io.read_tag().map_err(|e| Error::io("mp4/boxes", &e))?;
            let minor = io.read_u32be().map_err(|e| Error::io("mp4/boxes", &e))?;
            summary = Some(format!(
                "major_brand={} minor_version={minor}",
                fourcc_to_string(&major)
            ));
        } else if &hdr.box_type == b"mvhd" && payload_len >= 20 {
            if let Ok(mvhd) = parse_mvhd(io) {
                summary = Some(format!(
                    "timescale={} duration={}",
                    mvhd.timescale, mvhd.duration
                ));
            }
        } else if &hdr.box_type == b"tkhd" && payload_len >= 24 {
            if let Ok(tkhd) = parse_tkhd(io) {
                let rot = tkhd
                    .display_matrix
                    .as_ref()
                    .map_or(0, matrix_rotation_degrees);
                summary = Some(format!(
                    "track_id={} width={} height={} rotation={rot}",
                    tkhd.track_id, tkhd.width, tkhd.height
                ));
            }
        } else if &hdr.box_type == b"mdhd" && payload_len >= 20 {
            if let Ok(mdhd) = parse_mdhd(io) {
                summary = Some(format!(
                    "timescale={} duration={}",
                    mdhd.timescale, mdhd.duration
                ));
            }
        } else if &hdr.box_type == b"elst" && payload_len >= 8 {
            if let Ok(entries) = parse_elst(io) {
                summary = Some(format!("entries={}", entries.len()));
            }
        } else if &hdr.box_type == b"stsd" && payload_len >= 16 {
            let (_v, _f) = read_full_box_header(io)?;
            let count = io.read_u32be().map_err(|e| Error::io("mp4/boxes", &e))?;
            if count > 0 && io.tell() + 8 <= box_end {
                let entry_offset = io.tell();
                let entry_hdr = read_box_header(io)?;
                summary = Some(format!(
                    "entries={count} codec={}",
                    fourcc_to_string(&entry_hdr.box_type)
                ));
                nodes.push(Mp4BoxNode {
                    box_type: hdr.box_type,
                    offset,
                    size: box_end - offset,
                    header_size: hdr.header_size,
                    summary,
                    children: vec![Mp4BoxNode {
                        box_type: entry_hdr.box_type,
                        offset: entry_offset,
                        size: entry_hdr.size,
                        header_size: entry_hdr.header_size,
                        summary: None,
                        children: Vec::new(),
                    }],
                });
                let cur = io.tell();
                if cur < box_end {
                    io.skip(box_end - cur)
                        .map_err(|e| Error::io("mp4/boxes", &e))?;
                }
                continue;
            }
        }

        let cur = io.tell();
        if cur < box_end {
            if box_end == u64::MAX {
                break;
            }
            io.skip(box_end - cur)
                .map_err(|e| Error::io("mp4/boxes", &e))?;
        }

        nodes.push(Mp4BoxNode {
            box_type: hdr.box_type,
            offset,
            size: box_end.saturating_sub(offset),
            header_size: hdr.header_size,
            summary,
            children,
        });

        if box_end == u64::MAX {
            break;
        }
    }
    Ok(nodes)
}

// ---------------------------------------------------------------------------
// Movie Header (`mvhd`)
// ---------------------------------------------------------------------------

/// Parsed `mvhd` (Movie Header Box).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mvhd {
    pub timescale: u32,
    pub duration: u64,
    pub display_matrix: Option<[i32; 9]>,
}

pub fn parse_mvhd(io: &mut BufByteSource) -> Result<Mvhd> {
    let (version, _flags) = read_full_box_header(io)?;
    let (timescale, duration) = if version == 1 {
        io.skip(16).map_err(|e| Error::io("mp4/mvhd", &e))?;
        let ts = io.read_u32be().map_err(|e| Error::io("mp4/mvhd", &e))?;
        let dur = io.read_u64be().map_err(|e| Error::io("mp4/mvhd", &e))?;
        (ts, dur)
    } else {
        io.skip(8).map_err(|e| Error::io("mp4/mvhd", &e))?;
        let ts = io.read_u32be().map_err(|e| Error::io("mp4/mvhd", &e))?;
        let dur = u64::from(io.read_u32be().map_err(|e| Error::io("mp4/mvhd", &e))?);
        (ts, dur)
    };
    // rate (4) + volume (2) + reserved (10) = 16 bytes, then 36-byte matrix
    if io.skip(16).is_ok() {
        let mut mat = [0i32; 9];
        let mut ok = true;
        for slot in &mut mat {
            if let Ok(v) = io.read_u32be() {
                *slot = v as i32;
            } else {
                ok = false;
                break;
            }
        }
        let display_matrix = if ok && mat != IDENTITY_MATRIX_I32 {
            Some(mat)
        } else {
            None
        };
        Ok(Mvhd {
            timescale,
            duration,
            display_matrix,
        })
    } else {
        Ok(Mvhd {
            timescale,
            duration,
            display_matrix: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Track Header (`tkhd`) — including 3x3 display matrix (rotation)
// ---------------------------------------------------------------------------

/// Parsed `tkhd` (Track Header Box).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tkhd {
    pub track_id: u32,
    pub flags: u32,
    pub duration: u64,
    pub width: u32,
    pub height: u32,
    pub display_matrix: Option<[i32; 9]>,
}

pub fn parse_tkhd(io: &mut BufByteSource) -> Result<Tkhd> {
    let (version, flags) = read_full_box_header(io)?;
    let (track_id, duration) = if version == 1 {
        io.skip(16).map_err(|e| Error::io("mp4/tkhd", &e))?;
        let tid = io.read_u32be().map_err(|e| Error::io("mp4/tkhd", &e))?;
        io.skip(4).map_err(|e| Error::io("mp4/tkhd", &e))?;
        let dur = io.read_u64be().map_err(|e| Error::io("mp4/tkhd", &e))?;
        (tid, dur)
    } else {
        io.skip(8).map_err(|e| Error::io("mp4/tkhd", &e))?;
        let tid = io.read_u32be().map_err(|e| Error::io("mp4/tkhd", &e))?;
        io.skip(4).map_err(|e| Error::io("mp4/tkhd", &e))?;
        let dur = u64::from(io.read_u32be().map_err(|e| Error::io("mp4/tkhd", &e))?);
        (tid, dur)
    };
    // reserved (8) + layer (2) + alternate_group (2) + volume (2) + reserved (2) = 16 bytes
    io.skip(16).map_err(|e| Error::io("mp4/tkhd", &e))?;

    let mut matrix = [0i32; 9];
    for slot in &mut matrix {
        *slot = io.read_u32be().map_err(|e| Error::io("mp4/tkhd", &e))? as i32;
    }
    let display_matrix = (matrix != IDENTITY_MATRIX_I32).then_some(matrix);

    let width_fp = io.read_u32be().map_err(|e| Error::io("mp4/tkhd", &e))?;
    let height_fp = io.read_u32be().map_err(|e| Error::io("mp4/tkhd", &e))?;
    Ok(Tkhd {
        track_id,
        flags,
        duration,
        width: width_fp >> 16,
        height: height_fp >> 16,
        display_matrix,
    })
}

// ---------------------------------------------------------------------------
// Edit List (`elst`)
// ---------------------------------------------------------------------------

/// Raw ISOBMFF `elst` entry (`segment_duration` in movie timescale, `media_time` in media timescale).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElstEntry {
    pub segment_duration: u64,
    /// Media time in track timescale (`-1` indicates an empty dwell edit).
    pub media_time: i64,
    pub media_rate_integer: i16,
    pub media_rate_fraction: i16,
}

impl ElstEntry {
    /// Convert this `elst` entry into a [`wedeo_core::EditSegment`] in the track's timescale.
    pub fn to_edit_segment(
        self,
        track_timescale: u32,
        movie_timescale: u32,
        track_duration: u64,
    ) -> EditSegment {
        let tb = Rational::new(1, track_timescale.max(1) as i32);
        let media_time = if self.media_time < 0 {
            Timestamp::NONE
        } else {
            Timestamp::new(self.media_time)
        };
        let duration = if self.segment_duration == 0 {
            track_duration as i64
        } else if movie_timescale > 0 && track_timescale > 0 && movie_timescale != track_timescale {
            rescale(
                self.segment_duration as i64,
                i64::from(track_timescale),
                i64::from(movie_timescale),
            )
        } else {
            self.segment_duration as i64
        };
        let rate = if self.media_rate_fraction == 0 {
            Rational::new(i32::from(self.media_rate_integer), 1)
        } else {
            let num = (i64::from(self.media_rate_integer) << 16)
                + i64::from(self.media_rate_fraction as u16);
            let (n, d, _) = reduce(num, 65536, i32::MAX as i64);
            Rational::new(n, d)
        };
        EditSegment {
            media_time,
            duration,
            time_base: tb,
            rate,
        }
    }
}

pub fn parse_elst(io: &mut BufByteSource) -> Result<Vec<ElstEntry>> {
    let (version, _flags) = read_full_box_header(io)?;
    let count = io.read_u32be().map_err(|e| Error::io("mp4/elst", &e))? as usize;
    let mut entries = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let (segment_duration, media_time) = if version == 1 {
            let dur = io.read_u64be().map_err(|e| Error::io("mp4/elst", &e))?;
            let mt = io.read_u64be().map_err(|e| Error::io("mp4/elst", &e))? as i64;
            (dur, mt)
        } else {
            let dur = u64::from(io.read_u32be().map_err(|e| Error::io("mp4/elst", &e))?);
            let mt = i64::from(io.read_u32be().map_err(|e| Error::io("mp4/elst", &e))? as i32);
            (dur, mt)
        };
        let rate_int = io.read_u16be().map_err(|e| Error::io("mp4/elst", &e))? as i16;
        let rate_frac = io.read_u16be().map_err(|e| Error::io("mp4/elst", &e))? as i16;
        entries.push(ElstEntry {
            segment_duration,
            media_time,
            media_rate_integer: rate_int,
            media_rate_fraction: rate_frac,
        });
    }
    Ok(entries)
}

// ---------------------------------------------------------------------------
// Media Header (`mdhd`) & Handler (`hdlr`)
// ---------------------------------------------------------------------------

/// Parsed `mdhd` (Media Header Box).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mdhd {
    pub timescale: u32,
    pub duration: u64,
    pub language: Option<String>,
}

/// Unpack a 16-bit ISO-639-2/T language code from `mdhd`.
pub fn unpack_mdhd_language(code: u16) -> Option<String> {
    let c0 = ((code >> 10) & 0x1F) as u8;
    let c1 = ((code >> 5) & 0x1F) as u8;
    let c2 = (code & 0x1F) as u8;
    if (1..=26).contains(&c0) && (1..=26).contains(&c1) && (1..=26).contains(&c2) {
        let s = String::from_utf8(vec![c0 + 0x60, c1 + 0x60, c2 + 0x60]).ok()?;
        if s != "und" {
            return Some(s);
        }
    }
    None
}

/// Pack a 3-letter ASCII language code into a 16-bit `mdhd` language integer.
pub fn pack_mdhd_language(lang: Option<&str>) -> u16 {
    let Some(s) = lang else {
        return 0x55C4; // "und"
    };
    let b = s.as_bytes();
    if b.len() != 3 {
        return 0x55C4;
    }
    let l0 = b[0].to_ascii_lowercase().wrapping_sub(0x60);
    let l1 = b[1].to_ascii_lowercase().wrapping_sub(0x60);
    let l2 = b[2].to_ascii_lowercase().wrapping_sub(0x60);
    if (1..=26).contains(&l0) && (1..=26).contains(&l1) && (1..=26).contains(&l2) {
        (u16::from(l0) << 10) | (u16::from(l1) << 5) | u16::from(l2)
    } else {
        0x55C4
    }
}

pub fn parse_mdhd(io: &mut BufByteSource) -> Result<Mdhd> {
    let (version, _flags) = read_full_box_header(io)?;
    let (timescale, duration) = if version == 1 {
        io.skip(16).map_err(|e| Error::io("mp4/mdhd", &e))?;
        let ts = io.read_u32be().map_err(|e| Error::io("mp4/mdhd", &e))?;
        let dur = io.read_u64be().map_err(|e| Error::io("mp4/mdhd", &e))?;
        (ts, dur)
    } else {
        io.skip(8).map_err(|e| Error::io("mp4/mdhd", &e))?;
        let ts = io.read_u32be().map_err(|e| Error::io("mp4/mdhd", &e))?;
        let dur = u64::from(io.read_u32be().map_err(|e| Error::io("mp4/mdhd", &e))?);
        (ts, dur)
    };
    let lang_raw = io.read_u16be().unwrap_or(0x55C4);
    let _pre_defined = io.read_u16be();
    Ok(Mdhd {
        timescale,
        duration,
        language: unpack_mdhd_language(lang_raw),
    })
}

/// Parsed `hdlr` (Handler Reference Box).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hdlr {
    pub handler_type: [u8; 4],
}

pub fn parse_hdlr(io: &mut BufByteSource, payload_remaining: u64) -> Result<Hdlr> {
    let (_version, _flags) = read_full_box_header(io)?;
    io.skip(4).map_err(|e| Error::io("mp4/hdlr", &e))?; // pre_defined
    let handler_type = io.read_tag().map_err(|e| Error::io("mp4/hdlr", &e))?;
    let consumed = 12u64;
    if payload_remaining > consumed {
        io.skip(payload_remaining - consumed)
            .map_err(|e| Error::io("mp4/hdlr", &e))?;
    }
    Ok(Hdlr { handler_type })
}

// ---------------------------------------------------------------------------
// Colorimetry (`colr`), Pixel Aspect Ratio (`pasp`), HDR (`mdcv`, `clli`)
// ---------------------------------------------------------------------------

/// Parsed `colr` box (`nclx` / `nclc` CICP colorimetry or `prof`/`rICC` ICC profile).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Colr {
    Nclx(ColorDesc),
    IccProfile(Vec<u8>),
}

pub fn parse_colr(io: &mut BufByteSource, payload_size: u64) -> Result<Option<Colr>> {
    if payload_size < 4 {
        io.skip(payload_size)
            .map_err(|e| Error::io("mp4/colr", &e))?;
        return Ok(None);
    }
    let param_type = io.read_tag().map_err(|e| Error::io("mp4/colr", &e))?;
    let rem = payload_size - 4;

    match &param_type {
        b"nclx" | b"nclc" => {
            if rem < 6 {
                io.skip(rem).map_err(|e| Error::io("mp4/colr", &e))?;
                return Ok(None);
            }
            let pri = io.read_u16be().map_err(|e| Error::io("mp4/colr", &e))?;
            let trc = io.read_u16be().map_err(|e| Error::io("mp4/colr", &e))?;
            let mat = io.read_u16be().map_err(|e| Error::io("mp4/colr", &e))?;
            let range = if &param_type == b"nclx" && rem >= 7 {
                let flag = io.read_u8().map_err(|e| Error::io("mp4/colr", &e))?;
                if rem > 7 {
                    io.skip(rem - 7).map_err(|e| Error::io("mp4/colr", &e))?;
                }
                if (flag & 0x80) != 0 {
                    ColorRange::Full
                } else {
                    ColorRange::Limited
                }
            } else {
                if rem > 6 {
                    io.skip(rem - 6).map_err(|e| Error::io("mp4/colr", &e))?;
                }
                ColorRange::Unspecified
            };
            Ok(Some(Colr::Nclx(ColorDesc {
                primaries: ColorPrimaries::from_cicp(pri as u8),
                transfer: TransferCharacteristic::from_cicp(trc as u8),
                matrix: MatrixCoefficients::from_cicp(mat as u8),
                range,
                chroma_location: ChromaLocation::Unspecified,
            })))
        }
        b"prof" | b"rICC" => {
            let icc = read_bytes_vec(io, rem as usize)?;
            Ok(Some(Colr::IccProfile(icc)))
        }
        _ => {
            io.skip(rem).map_err(|e| Error::io("mp4/colr", &e))?;
            Ok(None)
        }
    }
}

/// Parsed `pasp` (Pixel Aspect Ratio Box).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pasp {
    pub h_spacing: u32,
    pub v_spacing: u32,
}

impl Pasp {
    pub fn to_sar(self) -> Option<Rational> {
        if self.h_spacing > 0
            && self.v_spacing > 0
            && self.h_spacing <= i32::MAX as u32
            && self.v_spacing <= i32::MAX as u32
        {
            Some(Rational::new(self.h_spacing as i32, self.v_spacing as i32))
        } else {
            None
        }
    }
}

pub fn parse_pasp(io: &mut BufByteSource) -> Result<Pasp> {
    let h_spacing = io.read_u32be().map_err(|e| Error::io("mp4/pasp", &e))?;
    let v_spacing = io.read_u32be().map_err(|e| Error::io("mp4/pasp", &e))?;
    Ok(Pasp {
        h_spacing,
        v_spacing,
    })
}

fn parse_mdcv(io: &mut BufByteSource) -> Result<MasteringDisplayMeta> {
    // 3 primaries (x, y) in 0.00002 units (u16be), white point (x, y) in 0.00002 units (u16be),
    // max_luminance and min_luminance in 0.0001 cd/m^2 (u32be).
    let mut primaries = [[Rational::new(0, 50_000); 2]; 3];
    for p in &mut primaries {
        let x = io.read_u16be().map_err(|e| Error::io("mp4/mdcv", &e))?;
        let y = io.read_u16be().map_err(|e| Error::io("mp4/mdcv", &e))?;
        p[0] = Rational::new(i32::from(x), 50_000);
        p[1] = Rational::new(i32::from(y), 50_000);
    }
    let wx = io.read_u16be().map_err(|e| Error::io("mp4/mdcv", &e))?;
    let wy = io.read_u16be().map_err(|e| Error::io("mp4/mdcv", &e))?;
    let max_lum = io.read_u32be().map_err(|e| Error::io("mp4/mdcv", &e))?;
    let min_lum = io.read_u32be().map_err(|e| Error::io("mp4/mdcv", &e))?;
    Ok(MasteringDisplayMeta {
        display_primaries: primaries,
        white_point: [
            Rational::new(i32::from(wx), 50_000),
            Rational::new(i32::from(wy), 50_000),
        ],
        min_luminance: Rational::new(min_lum.min(i32::MAX as u32) as i32, 10_000),
        max_luminance: Rational::new(max_lum.min(i32::MAX as u32) as i32, 10_000),
    })
}

fn parse_clli(io: &mut BufByteSource) -> Result<(u32, u32)> {
    let max_cll = u32::from(io.read_u16be().map_err(|e| Error::io("mp4/clli", &e))?);
    let max_fall = u32::from(io.read_u16be().map_err(|e| Error::io("mp4/clli", &e))?);
    Ok((max_cll, max_fall))
}

// ---------------------------------------------------------------------------
// Sample Description (`stsd`) — Video, Audio, and Subtitle sample entries
// ---------------------------------------------------------------------------

/// Parsed sample description entry from `stsd`.
#[derive(Debug, Clone)]
pub struct StsdEntry {
    pub fourcc: [u8; 4],
    pub width: u16,
    pub height: u16,
    pub depth: u16,
    pub channel_count: u16,
    pub sample_size: u16,
    pub sample_rate: u32,
    pub avg_bitrate: i64,
    pub initial_padding: u32,
    pub esds_object_type: Option<u8>,
    pub pcm_little_endian: Option<bool>,
    pub color: Option<ColorDesc>,
    pub sar: Option<Rational>,
    pub side_data: Vec<SideData>,
    pub extradata: Vec<u8>,
}

impl StsdEntry {
    fn empty(fourcc: [u8; 4]) -> Self {
        Self {
            fourcc,
            width: 0,
            height: 0,
            depth: 24,
            channel_count: 0,
            sample_size: 16,
            sample_rate: 0,
            avg_bitrate: 0,
            initial_padding: 0,
            esds_object_type: None,
            pcm_little_endian: None,
            color: None,
            sar: None,
            side_data: Vec::new(),
            extradata: Vec::new(),
        }
    }
}

pub fn parse_stsd(io: &mut BufByteSource, payload_size: u64) -> Result<StsdEntry> {
    let start = io.tell();
    let box_end = start + payload_size;
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be().map_err(|e| Error::io("mp4/stsd", &e))?;
    if entry_count == 0 {
        return Err(Error::invalid_data("mp4/stsd", "stsd has 0 sample entries"));
    }

    let entry_hdr = read_box_header(io)?;
    let fourcc = entry_hdr.box_type;
    let entry_payload = entry_hdr
        .payload_size()
        .unwrap_or_else(|| box_end.saturating_sub(io.tell()));

    let entry = match &fourcc {
        b"avc1" | b"avc3" | b"hvc1" | b"hev1" | b"av01" | b"vp09" | b"vp08" | b"mp4v" | b"apch"
        | b"apcn" | b"apcs" | b"apco" | b"ap4h" => {
            parse_visual_sample_entry(io, fourcc, entry_payload)?
        }
        b"mp4a" | b"alac" | b"Opus" | b"fLaC" | b"ac-3" | b"sowt" | b"twos" | b"in24" | b"in32"
        | b"fl32" | b"fl64" | b"raw " | b"NONE" | b"ulaw" | b"alaw" | b"ipcm" | b"fpcm"
        | b"lpcm" => parse_audio_sample_entry(io, fourcc, entry_payload)?,
        b"tx3g" | b"wvtt" | b"stpp" | b"c608" => {
            let mut e = StsdEntry::empty(fourcc);
            if entry_payload > 0 {
                e.extradata = read_bytes_vec(io, entry_payload as usize)?;
            }
            e
        }
        _ => {
            if entry_payload > 0 {
                io.skip(entry_payload)
                    .map_err(|e| Error::io("mp4/stsd", &e))?;
            }
            StsdEntry::empty(fourcc)
        }
    };

    let cur = io.tell();
    if cur < box_end {
        io.skip(box_end - cur)
            .map_err(|e| Error::io("mp4/stsd", &e))?;
    }
    Ok(entry)
}

fn parse_visual_sample_entry(
    io: &mut BufByteSource,
    fourcc: [u8; 4],
    payload_size: u64,
) -> Result<StsdEntry> {
    let start_pos = io.tell();
    let end_pos = start_pos + payload_size;

    // 78-byte VisualSampleEntry header:
    // reserved(6) + data_ref_idx(2) + pre_defined(2) + reserved(2) + pre_defined(12) = 24 bytes
    io.skip(24).map_err(|e| Error::io("mp4/stsd_video", &e))?;
    let width = io
        .read_u16be()
        .map_err(|e| Error::io("mp4/stsd_video", &e))?;
    let height = io
        .read_u16be()
        .map_err(|e| Error::io("mp4/stsd_video", &e))?;
    // horiz_res(4) + vert_res(4) + reserved(4) + frame_count(2) + compressorname(32) = 46 bytes
    io.skip(46).map_err(|e| Error::io("mp4/stsd_video", &e))?;
    let depth = io
        .read_u16be()
        .map_err(|e| Error::io("mp4/stsd_video", &e))?;
    let _pre_defined = io
        .read_u16be()
        .map_err(|e| Error::io("mp4/stsd_video", &e))?;

    let mut entry = StsdEntry::empty(fourcc);
    entry.width = width;
    entry.height = height;
    entry.depth = depth;

    while io.tell() + 8 <= end_pos {
        let sub_start = io.tell();
        let sub = read_box_header(io)?;
        let sub_end = if sub.size == 0 {
            end_pos
        } else {
            sub_start.saturating_add(sub.size).min(end_pos)
        };
        let sub_payload = sub_end.saturating_sub(io.tell());

        match &sub.box_type {
            b"avcC" | b"hvcC" | b"vpcC" => {
                entry.extradata = read_bytes_vec(io, sub_payload as usize)?;
            }
            b"av1C" => {
                let raw_av1c = read_bytes_vec(io, sub_payload as usize)?;
                // Extract configOBUs if non-empty (so rav1d receives raw OBUs directly),
                // otherwise keep the 4-byte av1C record so remuxing preserves config.
                if raw_av1c.len() > 4 {
                    entry.extradata = raw_av1c[4..].to_vec();
                } else {
                    entry.extradata = raw_av1c;
                }
            }
            b"esds" => {
                let (asc, avg_br, obj_type) = parse_esds_full(io, sub_payload)?;
                entry.extradata = asc;
                if avg_br > 0 {
                    entry.avg_bitrate = i64::from(avg_br);
                }
                entry.esds_object_type = obj_type;
            }
            b"colr" => {
                if let Some(colr) = parse_colr(io, sub_payload)? {
                    match colr {
                        Colr::Nclx(c) => entry.color = Some(c),
                        Colr::IccProfile(icc) => {
                            entry
                                .side_data
                                .push(SideData::IccProfile(Buffer::from_slice(&icc)));
                        }
                    }
                }
            }
            b"pasp" if sub_payload >= 8 => {
                let pasp = parse_pasp(io)?;
                entry.sar = pasp.to_sar();
            }
            b"mdcv" if sub_payload >= 24 => {
                if let Ok(mdcv) = parse_mdcv(io) {
                    entry.side_data.push(SideData::MasteringDisplay(mdcv));
                }
            }
            b"clli" if sub_payload >= 4 => {
                if let Ok((max_cll, max_fall)) = parse_clli(io) {
                    entry
                        .side_data
                        .push(SideData::ContentLightLevel { max_cll, max_fall });
                }
            }
            b"btrt" if sub_payload >= 12 => {
                let _buf_size = io.read_u32be();
                let _max_br = io.read_u32be();
                if let Ok(avg_br) = io.read_u32be()
                    && avg_br > 0
                {
                    entry.avg_bitrate = i64::from(avg_br);
                }
            }
            _ => {}
        }

        let cur = io.tell();
        if cur < sub_end {
            io.skip(sub_end - cur)
                .map_err(|e| Error::io("mp4/stsd_video", &e))?;
        }
    }

    let cur = io.tell();
    if cur < end_pos {
        io.skip(end_pos - cur)
            .map_err(|e| Error::io("mp4/stsd_video", &e))?;
    }
    Ok(entry)
}

fn parse_audio_sample_entry(
    io: &mut BufByteSource,
    fourcc: [u8; 4],
    payload_size: u64,
) -> Result<StsdEntry> {
    let start_pos = io.tell();
    let end_pos = start_pos + payload_size;

    // AudioSampleEntry header (28 bytes minimum):
    // reserved(6) + data_ref_idx(2) = 8 bytes
    io.skip(8).map_err(|e| Error::io("mp4/stsd_audio", &e))?;
    let sound_version = io
        .read_u16be()
        .map_err(|e| Error::io("mp4/stsd_audio", &e))?;
    io.skip(6).map_err(|e| Error::io("mp4/stsd_audio", &e))?; // revision(2) + vendor(4)
    let mut channel_count = io
        .read_u16be()
        .map_err(|e| Error::io("mp4/stsd_audio", &e))?;
    let mut sample_size = io
        .read_u16be()
        .map_err(|e| Error::io("mp4/stsd_audio", &e))?;
    io.skip(4).map_err(|e| Error::io("mp4/stsd_audio", &e))?; // compression_id(2) + packet_size(2)
    let sample_rate_fp = io
        .read_u32be()
        .map_err(|e| Error::io("mp4/stsd_audio", &e))?;
    let mut sample_rate = sample_rate_fp >> 16;

    // QuickTime Sound Sample Description v1 adds 16 bytes; v2 adds 36 bytes
    if sound_version == 1 && io.tell() + 16 <= end_pos {
        let _samples_per_pkt = io.read_u32be();
        let bytes_per_pkt = io.read_u32be().unwrap_or(0);
        let _bytes_per_frame = io.read_u32be();
        let bytes_per_sample = io.read_u32be().unwrap_or(0);
        if sample_size == 16 && bytes_per_sample > 0 && bytes_per_sample <= 8 {
            sample_size = (bytes_per_sample * 8) as u16;
        } else if sample_size == 16 && bytes_per_pkt > 0 && bytes_per_pkt <= 8 {
            sample_size = (bytes_per_pkt * 8) as u16;
        }
    } else if sound_version == 2 && io.tell() + 36 <= end_pos {
        let _size_of_struct = io.read_u32be();
        let sr_f64 = f64::from_bits(io.read_u64be().unwrap_or(0));
        let ch_v2 = io.read_u32be().unwrap_or(0);
        let _const_7f000000 = io.read_u32be();
        let bits_v2 = io.read_u32be().unwrap_or(0);
        let format_flags = io.read_u32be().unwrap_or(0);
        let _bytes_per_pkt = io.read_u32be();
        let _frames_per_pkt = io.read_u32be();
        if sr_f64 > 0.0 && sr_f64 < u32::MAX as f64 {
            sample_rate = sr_f64.round() as u32;
        }
        if ch_v2 > 0 && ch_v2 <= u16::MAX as u32 {
            channel_count = ch_v2 as u16;
        }
        if bits_v2 > 0 && bits_v2 <= 64 {
            sample_size = bits_v2 as u16;
        }
        let _ = format_flags;
    }

    let mut entry = StsdEntry::empty(fourcc);
    entry.channel_count = channel_count;
    entry.sample_size = sample_size;
    entry.sample_rate = sample_rate;

    parse_audio_sub_boxes(io, end_pos, &mut entry)?;

    let cur = io.tell();
    if cur < end_pos {
        io.skip(end_pos - cur)
            .map_err(|e| Error::io("mp4/stsd_audio", &e))?;
    }
    Ok(entry)
}

fn parse_audio_sub_boxes(
    io: &mut BufByteSource,
    end_pos: u64,
    entry: &mut StsdEntry,
) -> Result<()> {
    while io.tell() + 8 <= end_pos {
        let sub_start = io.tell();
        let Ok(sub) = read_box_header(io) else {
            break;
        };
        let sub_end = if sub.size == 0 {
            end_pos
        } else {
            sub_start.saturating_add(sub.size).min(end_pos)
        };
        let sub_payload = sub_end.saturating_sub(io.tell());

        match &sub.box_type {
            b"esds" => {
                let (asc, avg_br, obj_type) = parse_esds_full(io, sub_payload)?;
                entry.extradata = asc;
                if avg_br > 0 {
                    entry.avg_bitrate = i64::from(avg_br);
                }
                entry.esds_object_type = obj_type;
            }
            b"alac" if sub_payload >= 24 => {
                let cookie = read_bytes_vec(io, sub_payload as usize)?;
                // Standard 28-byte alac FullBox payload: 4-byte version/flags + 24-byte ALACSpecificConfig
                // Bytes 4..8: frameLength (u32be), 8: compatibleVersion, 9: bitDepth,
                // 13: numChannels, 24..28: sampleRate (u32be)
                let cfg_offset = if cookie.len() >= 28 && cookie[0..4] == [0, 0, 0, 0] {
                    4
                } else {
                    0
                };
                if cookie.len() >= cfg_offset + 24 {
                    let cfg = &cookie[cfg_offset..cfg_offset + 24];
                    let bit_depth = u16::from(cfg[5]);
                    let channels = u16::from(cfg[9]);
                    let sr = u32::from_be_bytes([cfg[20], cfg[21], cfg[22], cfg[23]]);
                    if bit_depth > 0 {
                        entry.sample_size = bit_depth;
                    }
                    if channels > 0 {
                        entry.channel_count = channels;
                    }
                    if sr > 0 {
                        entry.sample_rate = sr;
                    }
                }
                entry.extradata = cookie;
            }
            b"dOps" if sub_payload >= 11 => {
                let dops = read_bytes_vec(io, sub_payload as usize)?;
                if dops[0] == 0 {
                    let out_channels = u16::from(dops[1]);
                    let pre_skip = u16::from_be_bytes([dops[2], dops[3]]);
                    let input_sr = u32::from_be_bytes([dops[4], dops[5], dops[6], dops[7]]);
                    let out_gain = u16::from_be_bytes([dops[8], dops[9]]);
                    if out_channels > 0 {
                        entry.channel_count = out_channels;
                    }
                    // Opus always decodes at 48 kHz in ISO/IEC 14496-12 Opus encapsulation
                    if entry.sample_rate == 0 {
                        entry.sample_rate = if input_sr > 0 { input_sr } else { 48_000 };
                    }
                    entry.initial_padding = u32::from(pre_skip);

                    // Convert dOps to standard 19+ byte Ogg OpusHead extradata (matches FFmpeg mov_read_dops)
                    let mut opus_head = Vec::with_capacity(8 + dops.len());
                    opus_head.extend_from_slice(b"OpusHead");
                    opus_head.push(1); // version 1
                    opus_head.push(dops[1]); // channels
                    opus_head.extend_from_slice(&pre_skip.to_le_bytes());
                    opus_head.extend_from_slice(&input_sr.to_le_bytes());
                    opus_head.extend_from_slice(&out_gain.to_le_bytes());
                    opus_head.extend_from_slice(&dops[10..]);
                    entry.extradata = opus_head;
                }
            }
            b"dfLa" if sub_payload > 4 => {
                let dfla = read_bytes_vec(io, sub_payload as usize)?;
                // Skip 4-byte FullBox version/flags; if first metadata block is STREAMINFO (34 bytes), keep it
                entry.extradata = dfla[4..].to_vec();
            }
            b"dac3" => {
                entry.extradata = read_bytes_vec(io, sub_payload as usize)?;
            }
            b"enda" if sub_payload >= 2 => {
                let le_flag = io
                    .read_u16be()
                    .map_err(|e| Error::io("mp4/stsd_audio", &e))?;
                entry.pcm_little_endian = Some((le_flag & 0xFF) == 1);
            }
            b"pcmC" if sub_payload >= 6 => {
                let (_version, _flags) = read_full_box_header(io)?;
                let format_flags = io.read_u8().map_err(|e| Error::io("mp4/pcmC", &e))?;
                let pcm_sample_size = io.read_u8().map_err(|e| Error::io("mp4/pcmC", &e))?;
                entry.pcm_little_endian = Some((format_flags & 0x01) != 0);
                if pcm_sample_size > 0 {
                    entry.sample_size = u16::from(pcm_sample_size);
                }
            }
            b"srat" if sub_payload >= 8 => {
                let (_version, _flags) = read_full_box_header(io)?;
                let sr = io.read_u32be().map_err(|e| Error::io("mp4/srat", &e))?;
                if sr > 0 {
                    entry.sample_rate = sr;
                }
            }
            b"btrt" if sub_payload >= 12 => {
                let _buf_size = io.read_u32be();
                let _max_br = io.read_u32be();
                if let Ok(avg_br) = io.read_u32be()
                    && avg_br > 0
                {
                    entry.avg_bitrate = i64::from(avg_br);
                }
            }
            b"wave" | b"Wave" => {
                parse_audio_sub_boxes(io, sub_end, entry)?;
            }
            _ => {}
        }

        let cur = io.tell();
        if cur < sub_end {
            io.skip(sub_end - cur)
                .map_err(|e| Error::io("mp4/stsd_audio", &e))?;
        }
    }
    Ok(())
}

/// Parse an `esds` box and return `(decoder_specific_info, avg_bitrate, object_type_indication)`.
fn parse_esds_full(
    io: &mut BufByteSource,
    payload_size: u64,
) -> Result<(Vec<u8>, u32, Option<u8>)> {
    let start_pos = io.tell();
    let end_pos = start_pos + payload_size;

    let (_version, _flags) = read_full_box_header(io)?;

    let mut extradata = Vec::new();
    let mut avg_bitrate = 0u32;
    let mut object_type = None;

    while io.tell() < end_pos {
        let Ok(tag) = io.read_u8() else {
            break;
        };
        let Ok(len) = read_descriptor_length(io) else {
            break;
        };
        match tag {
            0x03 => {
                // ES_Descriptor: ES_ID (2 bytes) + flags (1 byte)
                if io.tell() + 3 > end_pos {
                    break;
                }
                let _es_id = io.read_u16be().map_err(|e| Error::io("mp4/esds", &e))?;
                let flags = io.read_u8().map_err(|e| Error::io("mp4/esds", &e))?;
                if (flags & 0x80) != 0 {
                    let _ = io.skip(2); // dependsOn_ES_ID
                }
                if (flags & 0x40) != 0
                    && let Ok(url_len) = io.read_u8()
                {
                    let _ = io.skip(u64::from(url_len));
                }
                if (flags & 0x20) != 0 {
                    let _ = io.skip(2); // OCR_ES_Id
                }
            }
            0x04 => {
                // DecoderConfigDescriptor (13 bytes header + optional sub-descriptors)
                if io.tell() + 13 > end_pos {
                    break;
                }
                let oti = io.read_u8().map_err(|e| Error::io("mp4/esds", &e))?;
                object_type = Some(oti);
                let _stream_type = io.read_u8().map_err(|e| Error::io("mp4/esds", &e))?;
                io.skip(3).map_err(|e| Error::io("mp4/esds", &e))?; // bufferSizeDB
                let _max_br = io.read_u32be().map_err(|e| Error::io("mp4/esds", &e))?;
                avg_bitrate = io.read_u32be().map_err(|e| Error::io("mp4/esds", &e))?;
            }
            0x05 => {
                // DecoderSpecificInfo (AudioSpecificConfig)
                let take = u64::from(len).min(end_pos.saturating_sub(io.tell())) as usize;
                extradata = read_bytes_vec(io, take)?;
                break;
            }
            _ => {
                let skip_len = u64::from(len).min(end_pos.saturating_sub(io.tell()));
                io.skip(skip_len).map_err(|e| Error::io("mp4/esds", &e))?;
            }
        }
    }

    let cur = io.tell();
    if cur < end_pos {
        io.skip(end_pos - cur)
            .map_err(|e| Error::io("mp4/esds", &e))?;
    }
    Ok((extradata, avg_bitrate, object_type))
}

fn read_descriptor_length(io: &mut BufByteSource) -> Result<u32> {
    let mut len = 0u32;
    for _ in 0..4 {
        let b = io.read_u8().map_err(|e| Error::io("mp4/esds", &e))?;
        len = (len << 7) | u32::from(b & 0x7F);
        if (b & 0x80) == 0 {
            break;
        }
    }
    Ok(len)
}

// ---------------------------------------------------------------------------
// Classic Sample Table Boxes (`stts`, `stsc`, `stsz`, `stco`, `co64`, `stss`, `ctts`)
// ---------------------------------------------------------------------------

/// `stts` — decoding time-to-sample: `Vec<(sample_count, sample_delta)>`.
pub fn parse_stts(io: &mut BufByteSource) -> Result<Vec<(u32, u32)>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be().map_err(|e| Error::io("mp4/stts", &e))? as usize;
    let mut entries = Vec::with_capacity(entry_count.min(65536));
    for _ in 0..entry_count {
        let count = io.read_u32be().map_err(|e| Error::io("mp4/stts", &e))?;
        let delta = io.read_u32be().map_err(|e| Error::io("mp4/stts", &e))?;
        entries.push((count, delta));
    }
    Ok(entries)
}

/// `stsc` — sample-to-chunk: `Vec<(first_chunk, samples_per_chunk, sample_desc_index)>`.
pub fn parse_stsc(io: &mut BufByteSource) -> Result<Vec<(u32, u32, u32)>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be().map_err(|e| Error::io("mp4/stsc", &e))? as usize;
    let mut entries = Vec::with_capacity(entry_count.min(65536));
    for _ in 0..entry_count {
        let first_chunk = io.read_u32be().map_err(|e| Error::io("mp4/stsc", &e))?;
        let samples_per_chunk = io.read_u32be().map_err(|e| Error::io("mp4/stsc", &e))?;
        let desc_index = io.read_u32be().map_err(|e| Error::io("mp4/stsc", &e))?;
        entries.push((first_chunk, samples_per_chunk, desc_index));
    }
    Ok(entries)
}

/// `stsz` — sample sizes. Returns `(fixed_size, sample_count, per_sample_sizes)`.
pub fn parse_stsz(io: &mut BufByteSource) -> Result<(u32, u32, Vec<u32>)> {
    let (_version, _flags) = read_full_box_header(io)?;
    let sample_size = io.read_u32be().map_err(|e| Error::io("mp4/stsz", &e))?;
    let sample_count = io.read_u32be().map_err(|e| Error::io("mp4/stsz", &e))?;
    if sample_size != 0 {
        return Ok((sample_size, sample_count, Vec::new()));
    }
    let count = sample_count as usize;
    let mut sizes = Vec::with_capacity(count.min(65536));
    for _ in 0..count {
        sizes.push(io.read_u32be().map_err(|e| Error::io("mp4/stsz", &e))?);
    }
    Ok((0, sample_count, sizes))
}

/// `stco` — 32-bit chunk offsets.
pub fn parse_stco(io: &mut BufByteSource) -> Result<Vec<u64>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be().map_err(|e| Error::io("mp4/stco", &e))? as usize;
    let mut offsets = Vec::with_capacity(entry_count.min(65536));
    for _ in 0..entry_count {
        offsets.push(u64::from(
            io.read_u32be().map_err(|e| Error::io("mp4/stco", &e))?,
        ));
    }
    Ok(offsets)
}

/// `co64` — 64-bit chunk offsets.
pub fn parse_co64(io: &mut BufByteSource) -> Result<Vec<u64>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be().map_err(|e| Error::io("mp4/co64", &e))? as usize;
    let mut offsets = Vec::with_capacity(entry_count.min(65536));
    for _ in 0..entry_count {
        offsets.push(io.read_u64be().map_err(|e| Error::io("mp4/co64", &e))?);
    }
    Ok(offsets)
}

/// `stss` — sync sample table (1-indexed keyframe sample numbers).
pub fn parse_stss(io: &mut BufByteSource) -> Result<Vec<u32>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be().map_err(|e| Error::io("mp4/stss", &e))? as usize;
    let mut indices = Vec::with_capacity(entry_count.min(65536));
    for _ in 0..entry_count {
        indices.push(io.read_u32be().map_err(|e| Error::io("mp4/stss", &e))?);
    }
    Ok(indices)
}

/// `ctts` — composition time offsets (version 0 unsigned and version 1 signed `i32`).
pub fn parse_ctts(io: &mut BufByteSource) -> Result<Vec<(u32, i32)>> {
    let (_version, _flags) = read_full_box_header(io)?;
    let entry_count = io.read_u32be().map_err(|e| Error::io("mp4/ctts", &e))? as usize;
    let mut entries = Vec::with_capacity(entry_count.min(65536));
    for _ in 0..entry_count {
        let count = io.read_u32be().map_err(|e| Error::io("mp4/ctts", &e))?;
        // Both v0 (small positive offsets) and v1 (signed i32 offsets) interpret raw u32 as i32
        // so even non-compliant v0 files with negative two's-complement offsets decode properly.
        let raw = io.read_u32be().map_err(|e| Error::io("mp4/ctts", &e))?;
        entries.push((count, raw as i32));
    }
    Ok(entries)
}

// ---------------------------------------------------------------------------
// Fragmented MP4 Boxes (`trex`, `tfhd`, `tfdt`, `trun`)
// ---------------------------------------------------------------------------

/// Parsed `trex` (Track Extends Box) inside `mvex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trex {
    pub track_id: u32,
    pub default_sample_description_index: u32,
    pub default_sample_duration: u32,
    pub default_sample_size: u32,
    pub default_sample_flags: u32,
}

pub fn parse_trex(io: &mut BufByteSource) -> Result<Trex> {
    let (_version, _flags) = read_full_box_header(io)?;
    let track_id = io.read_u32be().map_err(|e| Error::io("mp4/trex", &e))?;
    let desc_idx = io.read_u32be().map_err(|e| Error::io("mp4/trex", &e))?;
    let dur = io.read_u32be().map_err(|e| Error::io("mp4/trex", &e))?;
    let size = io.read_u32be().map_err(|e| Error::io("mp4/trex", &e))?;
    let flags = io.read_u32be().map_err(|e| Error::io("mp4/trex", &e))?;
    Ok(Trex {
        track_id,
        default_sample_description_index: desc_idx,
        default_sample_duration: dur,
        default_sample_size: size,
        default_sample_flags: flags,
    })
}

/// Parsed `tfhd` (Track Fragment Header Box) inside `traf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tfhd {
    pub track_id: u32,
    pub flags: u32,
    pub base_data_offset: Option<u64>,
    pub sample_description_index: Option<u32>,
    pub default_sample_duration: Option<u32>,
    pub default_sample_size: Option<u32>,
    pub default_sample_flags: Option<u32>,
    pub default_base_is_moof: bool,
}

pub fn parse_tfhd(io: &mut BufByteSource) -> Result<Tfhd> {
    let (_version, flags) = read_full_box_header(io)?;
    let track_id = io.read_u32be().map_err(|e| Error::io("mp4/tfhd", &e))?;
    let base_data_offset = if (flags & 0x000001) != 0 {
        Some(io.read_u64be().map_err(|e| Error::io("mp4/tfhd", &e))?)
    } else {
        None
    };
    let sample_description_index = if (flags & 0x000002) != 0 {
        Some(io.read_u32be().map_err(|e| Error::io("mp4/tfhd", &e))?)
    } else {
        None
    };
    let default_sample_duration = if (flags & 0x000008) != 0 {
        Some(io.read_u32be().map_err(|e| Error::io("mp4/tfhd", &e))?)
    } else {
        None
    };
    let default_sample_size = if (flags & 0x000010) != 0 {
        Some(io.read_u32be().map_err(|e| Error::io("mp4/tfhd", &e))?)
    } else {
        None
    };
    let default_sample_flags = if (flags & 0x000020) != 0 {
        Some(io.read_u32be().map_err(|e| Error::io("mp4/tfhd", &e))?)
    } else {
        None
    };
    Ok(Tfhd {
        track_id,
        flags,
        base_data_offset,
        sample_description_index,
        default_sample_duration,
        default_sample_size,
        default_sample_flags,
        default_base_is_moof: (flags & 0x020000) != 0,
    })
}

/// Parse `tfdt` (Track Fragment Base Media Decode Time Box).
pub fn parse_tfdt(io: &mut BufByteSource) -> Result<u64> {
    let (version, _flags) = read_full_box_header(io)?;
    if version == 1 {
        io.read_u64be().map_err(|e| Error::io("mp4/tfdt", &e))
    } else {
        io.read_u32be()
            .map(u64::from)
            .map_err(|e| Error::io("mp4/tfdt", &e))
    }
}

/// One sample record inside a `trun` (Track Run Box).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrunSample {
    pub duration: Option<u32>,
    pub size: Option<u32>,
    pub flags: Option<u32>,
    pub cts_offset: i32,
}

/// Parsed `trun` (Track Run Box) inside `traf`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trun {
    pub version: u8,
    pub flags: u32,
    pub data_offset: Option<i32>,
    pub first_sample_flags: Option<u32>,
    pub samples: Vec<TrunSample>,
}

pub fn parse_trun(io: &mut BufByteSource) -> Result<Trun> {
    let (version, flags) = read_full_box_header(io)?;
    let sample_count = io.read_u32be().map_err(|e| Error::io("mp4/trun", &e))? as usize;
    let data_offset = if (flags & 0x000001) != 0 {
        Some(io.read_u32be().map_err(|e| Error::io("mp4/trun", &e))? as i32)
    } else {
        None
    };
    let first_sample_flags = if (flags & 0x000004) != 0 {
        Some(io.read_u32be().map_err(|e| Error::io("mp4/trun", &e))?)
    } else {
        None
    };
    let has_dur = (flags & 0x000100) != 0;
    let has_size = (flags & 0x000200) != 0;
    let has_flags = (flags & 0x000400) != 0;
    let has_cts = (flags & 0x000800) != 0;

    let mut samples = Vec::with_capacity(sample_count.min(65536));
    for _ in 0..sample_count {
        let duration = if has_dur {
            Some(io.read_u32be().map_err(|e| Error::io("mp4/trun", &e))?)
        } else {
            None
        };
        let size = if has_size {
            Some(io.read_u32be().map_err(|e| Error::io("mp4/trun", &e))?)
        } else {
            None
        };
        let sflags = if has_flags {
            Some(io.read_u32be().map_err(|e| Error::io("mp4/trun", &e))?)
        } else {
            None
        };
        let cts_offset = if has_cts {
            io.read_u32be().map_err(|e| Error::io("mp4/trun", &e))? as i32
        } else {
            0
        };
        samples.push(TrunSample {
            duration,
            size,
            flags: sflags,
            cts_offset,
        });
    }

    Ok(Trun {
        version,
        flags,
        data_offset,
        first_sample_flags,
        samples,
    })
}
