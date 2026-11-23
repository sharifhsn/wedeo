//! EBML (Extensible Binary Meta Language) element IDs, VINT readers/writers,
//! lacing parsers, codec ID mappings, and shared AST inspection for Matroska/WebM.

use std::ops::Range;

use wedeo_core::{BufByteSource, CodecId, Error, MemorySource, Result};

// ---------------------------------------------------------------------------
// EBML Element IDs
// ---------------------------------------------------------------------------

// EBML Header
pub const EBML_HEADER: u32 = 0x1A45_DFA3;
pub const EBML_VERSION: u32 = 0x4286;
pub const EBML_READ_VERSION: u32 = 0x42F7;
pub const EBML_MAX_ID_LENGTH: u32 = 0x42F2;
pub const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
pub const EBML_DOCTYPE: u32 = 0x4282;
pub const EBML_DOCTYPE_VERSION: u32 = 0x4287;
pub const EBML_DOCTYPE_READ_VERSION: u32 = 0x4285;

// Global elements
pub const EBML_VOID: u32 = 0xEC;
pub const EBML_CRC32: u32 = 0xBF;

// Segment
pub const SEGMENT: u32 = 0x1853_8067;

// SeekHead
pub const SEEK_HEAD: u32 = 0x114D_9B74;
pub const SEEK: u32 = 0x4DBB;
pub const SEEK_ID: u32 = 0x53AB;
pub const SEEK_POSITION: u32 = 0x53AC;

// Info
pub const INFO: u32 = 0x1549_A966;
pub const SEGMENT_UID: u32 = 0x73A4;
pub const TIMECODE_SCALE: u32 = 0x002A_D7B1;
pub const DURATION: u32 = 0x4489;
pub const DATE_UTC: u32 = 0x4461;
pub const TITLE: u32 = 0x7BA9;
pub const MUXING_APP: u32 = 0x4D80;
pub const WRITING_APP: u32 = 0x5741;

// Tracks
pub const TRACKS: u32 = 0x1654_AE6B;
pub const TRACK_ENTRY: u32 = 0xAE;
pub const TRACK_NUMBER: u32 = 0xD7;
pub const TRACK_UID: u32 = 0x73C5;
pub const TRACK_TYPE: u32 = 0x83;
pub const FLAG_ENABLED: u32 = 0xB9;
pub const FLAG_DEFAULT: u32 = 0x88;
pub const FLAG_FORCED: u32 = 0x55AA;
pub const FLAG_HEARING_IMPAIRED: u32 = 0x55AB;
pub const FLAG_VISUAL_IMPAIRED: u32 = 0x55AC;
pub const FLAG_COMMENTARY: u32 = 0x55AE;
pub const FLAG_LACING: u32 = 0x9C;
pub const DEFAULT_DURATION: u32 = 0x0023_E383;
pub const TRACK_TIMECODE_SCALE: u32 = 0x0023_314F;
pub const NAME: u32 = 0x536E;
pub const LANGUAGE: u32 = 0x0022_B59C;
pub const LANGUAGE_BCP47: u32 = 0x0022_B59D;
pub const CODEC_ID: u32 = 0x86;
pub const CODEC_PRIVATE: u32 = 0x63A2;
pub const CODEC_NAME: u32 = 0x0025_8688;
pub const CODEC_DELAY: u32 = 0x56AA;
pub const SEEK_PRE_ROLL: u32 = 0x56BB;

// Video
pub const VIDEO: u32 = 0xE0;
pub const FLAG_INTERLACED: u32 = 0x9A;
pub const FIELD_ORDER: u32 = 0x9D;
pub const STEREO_MODE: u32 = 0x53B8;
pub const ALPHA_MODE: u32 = 0x53C0;
pub const PIXEL_WIDTH: u32 = 0xB0;
pub const PIXEL_HEIGHT: u32 = 0xBA;
pub const PIXEL_CROP_BOTTOM: u32 = 0x54AA;
pub const PIXEL_CROP_TOP: u32 = 0x54BB;
pub const PIXEL_CROP_LEFT: u32 = 0x54CC;
pub const PIXEL_CROP_RIGHT: u32 = 0x54DD;
pub const DISPLAY_WIDTH: u32 = 0x54B0;
pub const DISPLAY_HEIGHT: u32 = 0x54BA;
pub const DISPLAY_UNIT: u32 = 0x54B2;

// Colour
pub const COLOUR: u32 = 0x55B0;
pub const MATRIX_COEFFICIENTS: u32 = 0x55B1;
pub const BITS_PER_CHANNEL: u32 = 0x55B2;
pub const CHROMA_SUBSAMPLING_HORZ: u32 = 0x55B3;
pub const CHROMA_SUBSAMPLING_VERT: u32 = 0x55B4;
pub const CB_SUBSAMPLING_HORZ: u32 = 0x55B5;
pub const CB_SUBSAMPLING_VERT: u32 = 0x55B6;
pub const CHROMA_SITING_HORZ: u32 = 0x55B7;
pub const CHROMA_SITING_VERT: u32 = 0x55B8;
pub const RANGE: u32 = 0x55B9;
pub const TRANSFER_CHARACTERISTICS: u32 = 0x55BA;
pub const PRIMARIES: u32 = 0x55BB;
pub const MAX_CLL: u32 = 0x55BC;
pub const MAX_FALL: u32 = 0x55BD;
pub const MASTERING_METADATA: u32 = 0x55D0;
pub const PRIMARY_R_CHROMATICITY_X: u32 = 0x55D1;
pub const PRIMARY_R_CHROMATICITY_Y: u32 = 0x55D2;
pub const PRIMARY_G_CHROMATICITY_X: u32 = 0x55D3;
pub const PRIMARY_G_CHROMATICITY_Y: u32 = 0x55D4;
pub const PRIMARY_B_CHROMATICITY_X: u32 = 0x55D5;
pub const PRIMARY_B_CHROMATICITY_Y: u32 = 0x55D6;
pub const WHITE_POINT_CHROMATICITY_X: u32 = 0x55D7;
pub const WHITE_POINT_CHROMATICITY_Y: u32 = 0x55D8;
pub const LUMINANCE_MAX: u32 = 0x55D9;
pub const LUMINANCE_MIN: u32 = 0x55DA;

// Audio
pub const AUDIO: u32 = 0xE1;
pub const SAMPLING_FREQUENCY: u32 = 0xB5;
pub const OUTPUT_SAMPLING_FREQUENCY: u32 = 0x78B5;
pub const CHANNELS: u32 = 0x9F;
pub const BIT_DEPTH: u32 = 0x6264;

// Cluster & Blocks
pub const CLUSTER: u32 = 0x1F43_B675;
pub const CLUSTER_TIMECODE: u32 = 0xE7;
pub const SIMPLE_BLOCK: u32 = 0xA3;
pub const BLOCK_GROUP: u32 = 0xA0;
pub const BLOCK: u32 = 0xA1;
pub const BLOCK_DURATION: u32 = 0x9B;
pub const REFERENCE_BLOCK: u32 = 0xFB;
pub const DISCARD_PADDING: u32 = 0x75A2;

// Cues
pub const CUES: u32 = 0x1C53_BB6B;
pub const CUE_POINT: u32 = 0xBB;
pub const CUE_TIME: u32 = 0xB3;
pub const CUE_TRACK_POSITIONS: u32 = 0xB7;
pub const CUE_TRACK: u32 = 0xF7;
pub const CUE_CLUSTER_POSITION: u32 = 0xF1;
pub const CUE_RELATIVE_POSITION: u32 = 0xF0;
pub const CUE_DURATION: u32 = 0xB2;

// Chapters
pub const CHAPTERS: u32 = 0x1043_A770;
pub const EDITION_ENTRY: u32 = 0x45B9;
pub const CHAPTER_ATOM: u32 = 0xB6;
pub const CHAPTER_UID: u32 = 0x73C4;
pub const CHAPTER_TIME_START: u32 = 0x91;
pub const CHAPTER_TIME_END: u32 = 0x92;
pub const CHAPTER_DISPLAY: u32 = 0x80;
pub const CHAP_STRING: u32 = 0x85;
pub const CHAP_LANGUAGE: u32 = 0x437C;

// Attachments
pub const ATTACHMENTS: u32 = 0x1941_A469;
pub const ATTACHED_FILE: u32 = 0x61A7;
pub const FILE_DESCRIPTION: u32 = 0x467E;
pub const FILE_NAME: u32 = 0x466E;
pub const FILE_MIME_TYPE: u32 = 0x4660;
pub const FILE_DATA: u32 = 0x465C;
pub const FILE_UID: u32 = 0x46AE;

// Tags
pub const TAGS: u32 = 0x1254_C367;
pub const TAG: u32 = 0x7373;
pub const TARGETS: u32 = 0x63C0;
pub const TARGET_TRACK_UID: u32 = 0x63C5;
pub const SIMPLE_TAG: u32 = 0x67C8;
pub const TAG_NAME: u32 = 0x45A3;
pub const TAG_STRING: u32 = 0x4487;

// Track types
pub const TRACK_TYPE_VIDEO: u8 = 0x01;
pub const TRACK_TYPE_AUDIO: u8 = 0x02;
pub const TRACK_TYPE_SUBTITLE: u8 = 0x11;

/// Sentinel indicating an EBML element of unknown/indeterminate size (`0x01FF_FFFF_FFFF_FFFF`).
pub const SIZE_UNKNOWN: u64 = u64::MAX;

// ---------------------------------------------------------------------------
// EBML Header & AST Nodes for `wedeo inspect`
// ---------------------------------------------------------------------------

/// Parsed EBML element header (ID + size + byte offsets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EbmlHeader {
    pub id: u32,
    /// Payload size in bytes, or [`SIZE_UNKNOWN`] if indeterminate.
    pub size: u64,
    /// Stream byte offset where the element ID begins.
    pub header_offset: u64,
    /// Stream byte offset where the element payload begins.
    pub data_offset: u64,
}

impl EbmlHeader {
    #[inline]
    pub const fn header_len(&self) -> u64 {
        self.data_offset.saturating_sub(self.header_offset)
    }

    #[inline]
    pub const fn end_offset(&self) -> Option<u64> {
        if self.size == SIZE_UNKNOWN {
            None
        } else {
            self.data_offset.checked_add(self.size)
        }
    }

    #[inline]
    pub fn name(&self) -> &'static str {
        element_name(self.id)
    }
}

/// AST node representing an EBML element and its nested children for `wedeo inspect`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EbmlElement {
    pub id: u32,
    pub name: &'static str,
    pub header_offset: u64,
    pub data_offset: u64,
    pub size: u64,
    pub children: Vec<EbmlElement>,
}

/// Return a human-readable Matroska/EBML element name.
pub fn element_name(id: u32) -> &'static str {
    match id {
        EBML_HEADER => "EBML",
        EBML_VERSION => "EBMLVersion",
        EBML_READ_VERSION => "EBMLReadVersion",
        EBML_MAX_ID_LENGTH => "EBMLMaxIDLength",
        EBML_MAX_SIZE_LENGTH => "EBMLMaxSizeLength",
        EBML_DOCTYPE => "DocType",
        EBML_DOCTYPE_VERSION => "DocTypeVersion",
        EBML_DOCTYPE_READ_VERSION => "DocTypeReadVersion",
        EBML_VOID => "Void",
        EBML_CRC32 => "CRC-32",
        SEGMENT => "Segment",
        SEEK_HEAD => "SeekHead",
        SEEK => "Seek",
        SEEK_ID => "SeekID",
        SEEK_POSITION => "SeekPosition",
        INFO => "Info",
        SEGMENT_UID => "SegmentUID",
        TIMECODE_SCALE => "TimestampScale",
        DURATION => "Duration",
        DATE_UTC => "DateUTC",
        TITLE => "Title",
        MUXING_APP => "MuxingApp",
        WRITING_APP => "WritingApp",
        TRACKS => "Tracks",
        TRACK_ENTRY => "TrackEntry",
        TRACK_NUMBER => "TrackNumber",
        TRACK_UID => "TrackUID",
        TRACK_TYPE => "TrackType",
        FLAG_ENABLED => "FlagEnabled",
        FLAG_DEFAULT => "FlagDefault",
        FLAG_FORCED => "FlagForced",
        FLAG_HEARING_IMPAIRED => "FlagHearingImpaired",
        FLAG_VISUAL_IMPAIRED => "FlagVisualImpaired",
        FLAG_COMMENTARY => "FlagCommentary",
        FLAG_LACING => "FlagLacing",
        DEFAULT_DURATION => "DefaultDuration",
        TRACK_TIMECODE_SCALE => "TrackTimestampScale",
        NAME => "Name",
        LANGUAGE => "Language",
        LANGUAGE_BCP47 => "LanguageBCP47",
        CODEC_ID => "CodecID",
        CODEC_PRIVATE => "CodecPrivate",
        CODEC_NAME => "CodecName",
        CODEC_DELAY => "CodecDelay",
        SEEK_PRE_ROLL => "SeekPreRoll",
        VIDEO => "Video",
        FLAG_INTERLACED => "FlagInterlaced",
        FIELD_ORDER => "FieldOrder",
        STEREO_MODE => "StereoMode",
        ALPHA_MODE => "AlphaMode",
        PIXEL_WIDTH => "PixelWidth",
        PIXEL_HEIGHT => "PixelHeight",
        PIXEL_CROP_BOTTOM => "PixelCropBottom",
        PIXEL_CROP_TOP => "PixelCropTop",
        PIXEL_CROP_LEFT => "PixelCropLeft",
        PIXEL_CROP_RIGHT => "PixelCropRight",
        DISPLAY_WIDTH => "DisplayWidth",
        DISPLAY_HEIGHT => "DisplayHeight",
        DISPLAY_UNIT => "DisplayUnit",
        COLOUR => "Colour",
        MATRIX_COEFFICIENTS => "MatrixCoefficients",
        BITS_PER_CHANNEL => "BitsPerChannel",
        CHROMA_SUBSAMPLING_HORZ => "ChromaSubsamplingHorz",
        CHROMA_SUBSAMPLING_VERT => "ChromaSubsamplingVert",
        CB_SUBSAMPLING_HORZ => "CbSubsamplingHorz",
        CB_SUBSAMPLING_VERT => "CbSubsamplingVert",
        CHROMA_SITING_HORZ => "ChromaSitingHorz",
        CHROMA_SITING_VERT => "ChromaSitingVert",
        RANGE => "Range",
        TRANSFER_CHARACTERISTICS => "TransferCharacteristics",
        PRIMARIES => "Primaries",
        MAX_CLL => "MaxCLL",
        MAX_FALL => "MaxFALL",
        MASTERING_METADATA => "MasteringMetadata",
        PRIMARY_R_CHROMATICITY_X => "PrimaryRChromaticityX",
        PRIMARY_R_CHROMATICITY_Y => "PrimaryRChromaticityY",
        PRIMARY_G_CHROMATICITY_X => "PrimaryGChromaticityX",
        PRIMARY_G_CHROMATICITY_Y => "PrimaryGChromaticityY",
        PRIMARY_B_CHROMATICITY_X => "PrimaryBChromaticityX",
        PRIMARY_B_CHROMATICITY_Y => "PrimaryBChromaticityY",
        WHITE_POINT_CHROMATICITY_X => "WhitePointChromaticityX",
        WHITE_POINT_CHROMATICITY_Y => "WhitePointChromaticityY",
        LUMINANCE_MAX => "LuminanceMax",
        LUMINANCE_MIN => "LuminanceMin",
        AUDIO => "Audio",
        SAMPLING_FREQUENCY => "SamplingFrequency",
        OUTPUT_SAMPLING_FREQUENCY => "OutputSamplingFrequency",
        CHANNELS => "Channels",
        BIT_DEPTH => "BitDepth",
        CLUSTER => "Cluster",
        CLUSTER_TIMECODE => "Timestamp",
        SIMPLE_BLOCK => "SimpleBlock",
        BLOCK_GROUP => "BlockGroup",
        BLOCK => "Block",
        BLOCK_DURATION => "BlockDuration",
        REFERENCE_BLOCK => "ReferenceBlock",
        DISCARD_PADDING => "DiscardPadding",
        CUES => "Cues",
        CUE_POINT => "CuePoint",
        CUE_TIME => "CueTime",
        CUE_TRACK_POSITIONS => "CueTrackPositions",
        CUE_TRACK => "CueTrack",
        CUE_CLUSTER_POSITION => "CueClusterPosition",
        CUE_RELATIVE_POSITION => "CueRelativePosition",
        CUE_DURATION => "CueDuration",
        CHAPTERS => "Chapters",
        EDITION_ENTRY => "EditionEntry",
        CHAPTER_ATOM => "ChapterAtom",
        CHAPTER_UID => "ChapterUID",
        CHAPTER_TIME_START => "ChapterTimeStart",
        CHAPTER_TIME_END => "ChapterTimeEnd",
        CHAPTER_DISPLAY => "ChapterDisplay",
        CHAP_STRING => "ChapString",
        CHAP_LANGUAGE => "ChapLanguage",
        ATTACHMENTS => "Attachments",
        ATTACHED_FILE => "AttachedFile",
        FILE_DESCRIPTION => "FileDescription",
        FILE_NAME => "FileName",
        FILE_MIME_TYPE => "FileMimeType",
        FILE_DATA => "FileData",
        FILE_UID => "FileUID",
        TAGS => "Tags",
        TAG => "Tag",
        TARGETS => "Targets",
        TARGET_TRACK_UID => "TargetTrackUID",
        SIMPLE_TAG => "SimpleTag",
        TAG_NAME => "TagName",
        TAG_STRING => "TagString",
        _ => "Unknown",
    }
}

/// Returns `true` if `id` is a known Master (container) element in Matroska/WebM.
pub fn is_master_element(id: u32) -> bool {
    matches!(
        id,
        EBML_HEADER
            | SEGMENT
            | SEEK_HEAD
            | SEEK
            | INFO
            | TRACKS
            | TRACK_ENTRY
            | VIDEO
            | COLOUR
            | MASTERING_METADATA
            | AUDIO
            | CLUSTER
            | BLOCK_GROUP
            | CUES
            | CUE_POINT
            | CUE_TRACK_POSITIONS
            | CHAPTERS
            | EDITION_ENTRY
            | CHAPTER_ATOM
            | CHAPTER_DISPLAY
            | ATTACHMENTS
            | ATTACHED_FILE
            | TAGS
            | TAG
            | TARGETS
            | SIMPLE_TAG
    )
}

/// Returns `true` if `id` is a top-level Segment child (used to detect end of unknown-sized Cluster).
pub fn is_top_level_segment_element(id: u32) -> bool {
    matches!(
        id,
        SEEK_HEAD | INFO | TRACKS | CLUSTER | CUES | CHAPTERS | ATTACHMENTS | TAGS
    )
}

// ---------------------------------------------------------------------------
// EBML Reading Primitives
// ---------------------------------------------------------------------------

/// Read an EBML Element ID (1..=4 bytes). Returns `Ok(None)` on clean EOF at the first byte.
pub fn read_element_id(io: &mut BufByteSource) -> Result<Option<u32>> {
    let b0 = match io.read_u8() {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(Error::io("mkv/ebml", &e)),
    };

    if b0 == 0 {
        return Err(Error::invalid_data(
            "mkv/ebml",
            "invalid zero leading byte in EBML ID",
        ));
    }

    if b0 & 0x80 != 0 {
        Ok(Some(u32::from(b0)))
    } else if b0 & 0x40 != 0 {
        let b1 = io.read_u8().map_err(|e| Error::io("mkv/ebml", &e))?;
        Ok(Some((u32::from(b0) << 8) | u32::from(b1)))
    } else if b0 & 0x20 != 0 {
        let mut buf = [0u8; 2];
        io.read_exact_bytes(&mut buf)
            .map_err(|e| Error::io("mkv/ebml", &e))?;
        Ok(Some(
            (u32::from(b0) << 16) | (u32::from(buf[0]) << 8) | u32::from(buf[1]),
        ))
    } else if b0 & 0x10 != 0 {
        let mut buf = [0u8; 3];
        io.read_exact_bytes(&mut buf)
            .map_err(|e| Error::io("mkv/ebml", &e))?;
        Ok(Some(
            (u32::from(b0) << 24)
                | (u32::from(buf[0]) << 16)
                | (u32::from(buf[1]) << 8)
                | u32::from(buf[2]),
        ))
    } else {
        Err(Error::invalid_data(
            "mkv/ebml",
            "EBML element ID longer than 4 bytes",
        ))
    }
}

/// Read an EBML variable-length size (1..=8 bytes). Returns [`SIZE_UNKNOWN`] if all value bits are 1.
pub fn read_vint_size(io: &mut BufByteSource) -> Result<u64> {
    let b0 = io.read_u8().map_err(|e| Error::io("mkv/ebml", &e))?;
    if b0 == 0 {
        return Err(Error::invalid_data(
            "mkv/ebml",
            "invalid zero leading byte in EBML VINT size",
        ));
    }

    let width = b0.leading_zeros() + 1;
    let mask = 0xFFu8.checked_shr(width).unwrap_or(0);
    let mut value = u64::from(b0 & mask);

    let extra = (width - 1) as usize;
    if extra > 0 {
        let mut buf = [0u8; 7];
        io.read_exact_bytes(&mut buf[..extra])
            .map_err(|e| Error::io("mkv/ebml", &e))?;
        for &b in &buf[..extra] {
            value = (value << 8) | u64::from(b);
        }
    }

    let all_ones = if width == 8 {
        (1u64 << 56) - 1
    } else {
        (1u64 << (7 * width)) - 1
    };
    if value == all_ones {
        return Ok(SIZE_UNKNOWN);
    }

    Ok(value)
}

/// Read an [`EbmlHeader`] (ID + size) from `io`, returning `Ok(None)` on clean EOF.
pub fn read_element_header(io: &mut BufByteSource) -> Result<Option<EbmlHeader>> {
    let header_offset = io.tell();
    let Some(id) = read_element_id(io)? else {
        return Ok(None);
    };
    let size = read_vint_size(io)?;
    let data_offset = io.tell();
    Ok(Some(EbmlHeader {
        id,
        size,
        header_offset,
        data_offset,
    }))
}

/// Read an unsigned VINT from a byte slice (used for track numbers and EBML lacing frame 0).
/// Returns `(value, bytes_consumed)`.
pub fn read_vint_data(data: &[u8]) -> Result<(u64, usize)> {
    let Some(&b0) = data.first() else {
        return Err(Error::invalid_data("mkv/ebml", "empty slice for VINT"));
    };
    if b0 == 0 {
        return Err(Error::invalid_data("mkv/ebml", "zero leading byte in VINT"));
    }
    let width = (b0.leading_zeros() + 1) as usize;
    if width > 8 || width > data.len() {
        return Err(Error::invalid_data("mkv/ebml", "truncated VINT in slice"));
    }
    let mask = 0xFFu8.checked_shr(width as u32).unwrap_or(0);
    let mut value = u64::from(b0 & mask);
    for &b in &data[1..width] {
        value = (value << 8) | u64::from(b);
    }
    Ok((value, width))
}

/// Read a signed VINT from a byte slice (used for EBML lacing frame size deltas).
/// Returns `(signed_delta, bytes_consumed)`.
pub fn read_svint_data(data: &[u8]) -> Result<(i64, usize)> {
    let (raw, width) = read_vint_data(data)?;
    let bits = 7 * (width as u32);
    let bias = (1i64 << (bits - 1)) - 1;
    Ok((raw as i64 - bias, width))
}

/// Read an unsigned integer element body (`0..=8` bytes).
pub fn read_uint(io: &mut BufByteSource, size: u64) -> Result<u64> {
    if size == 0 {
        return Ok(0);
    }
    if size > 8 {
        return Err(Error::invalid_data(
            "mkv/ebml",
            "uint element exceeds 8 bytes",
        ));
    }
    let mut buf = [0u8; 8];
    let n = size as usize;
    io.read_exact_bytes(&mut buf[..n])
        .map_err(|e| Error::io("mkv/ebml", &e))?;
    let mut val = 0u64;
    for &b in &buf[..n] {
        val = (val << 8) | u64::from(b);
    }
    Ok(val)
}

/// Read a signed big-endian integer element body (`0..=8` bytes).
pub fn read_sint(io: &mut BufByteSource, size: u64) -> Result<i64> {
    if size == 0 {
        return Ok(0);
    }
    if size > 8 {
        return Err(Error::invalid_data(
            "mkv/ebml",
            "sint element exceeds 8 bytes",
        ));
    }
    let mut buf = [0u8; 8];
    let n = size as usize;
    io.read_exact_bytes(&mut buf[..n])
        .map_err(|e| Error::io("mkv/ebml", &e))?;
    // Sign-extend from the first byte.
    let mut val = if (buf[0] & 0x80) != 0 { -1i64 } else { 0i64 };
    for &b in &buf[..n] {
        val = (val << 8) | i64::from(b);
    }
    Ok(val)
}

/// Read an EBML float element (`0`, `4`, or `8` bytes).
pub fn read_float(io: &mut BufByteSource, size: u64) -> Result<f64> {
    match size {
        0 => Ok(0.0),
        4 => {
            let mut buf = [0u8; 4];
            io.read_exact_bytes(&mut buf)
                .map_err(|e| Error::io("mkv/ebml", &e))?;
            Ok(f64::from(f32::from_be_bytes(buf)))
        }
        8 => {
            let mut buf = [0u8; 8];
            io.read_exact_bytes(&mut buf)
                .map_err(|e| Error::io("mkv/ebml", &e))?;
            Ok(f64::from_be_bytes(buf))
        }
        _ => Err(Error::invalid_data("mkv/ebml", "invalid EBML float size")),
    }
}

/// Read a UTF-8 or ASCII string element body, stripping trailing NUL padding.
pub fn read_string(io: &mut BufByteSource, size: u64) -> Result<String> {
    if size > 1024 * 1024 {
        return Err(Error::invalid_data(
            "mkv/ebml",
            "string element exceeds 1 MiB limit",
        ));
    }
    let data = read_bytes(io, size)?;
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    Ok(String::from_utf8_lossy(&data[..end]).into_owned())
}

/// Read `size` raw bytes from `io`.
pub fn read_bytes(io: &mut BufByteSource, size: u64) -> Result<Vec<u8>> {
    if size == SIZE_UNKNOWN || size > 64 * 1024 * 1024 {
        return Err(Error::invalid_data(
            "mkv/ebml",
            "binary element size unknown or exceeds 64 MiB limit",
        ));
    }
    let mut buf = vec![0u8; size as usize];
    io.read_exact_bytes(&mut buf)
        .map_err(|e| Error::io("mkv/ebml", &e))?;
    Ok(buf)
}

/// Skip `size` bytes in `io`.
pub fn skip_element(io: &mut BufByteSource, size: u64) -> Result<()> {
    if size == SIZE_UNKNOWN {
        return Err(Error::invalid_data(
            "mkv/ebml",
            "cannot skip element of unknown size",
        ));
    }
    io.skip(size).map_err(|e| Error::io("mkv/ebml", &e))
}

// ---------------------------------------------------------------------------
// EBML Writing Primitives
// ---------------------------------------------------------------------------

/// Write a 1..=4 byte EBML Element ID into `buf`.
pub fn write_element_id(buf: &mut Vec<u8>, id: u32) {
    if id <= 0xFF {
        buf.push(id as u8);
    } else if id <= 0xFFFF {
        buf.extend_from_slice(&(id as u16).to_be_bytes());
    } else if id <= 0xFF_FFFF {
        buf.push(((id >> 16) & 0xFF) as u8);
        buf.push(((id >> 8) & 0xFF) as u8);
        buf.push((id & 0xFF) as u8);
    } else {
        buf.extend_from_slice(&id.to_be_bytes());
    }
}

/// Encode `size` as a minimal-width EBML VINT size (1..=8 bytes), or unknown size if [`SIZE_UNKNOWN`].
pub fn write_vint_size(buf: &mut Vec<u8>, size: u64) {
    if size == SIZE_UNKNOWN {
        buf.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        return;
    }
    // Note: all-ones payload for a given width is reserved for SIZE_UNKNOWN, so the max value
    // for width W is (1 << (7*W)) - 2.
    let width = if size < (1u64 << 7) - 1 {
        1
    } else if size < (1u64 << 14) - 1 {
        2
    } else if size < (1u64 << 21) - 1 {
        3
    } else if size < (1u64 << 28) - 1 {
        4
    } else if size < (1u64 << 35) - 1 {
        5
    } else if size < (1u64 << 42) - 1 {
        6
    } else if size < (1u64 << 49) - 1 {
        7
    } else {
        8
    };

    let marker = 1u64 << (7 * width);
    let encoded = marker | size;
    for i in (0..width).rev() {
        buf.push(((encoded >> (i * 8)) & 0xFF) as u8);
    }
}

/// Encode `size` as a fixed 8-byte EBML VINT size (`0x01` prefix + 7-byte big-endian size).
pub fn write_vint_size_fixed8(buf: &mut Vec<u8>, size: u64) {
    if size == SIZE_UNKNOWN {
        buf.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    } else {
        let encoded = (1u64 << 56) | (size & 0x00FF_FFFF_FFFF_FFFF);
        buf.extend_from_slice(&encoded.to_be_bytes());
    }
}

/// Write an unsigned integer element (minimal 1..=8 bytes).
pub fn write_uint_element(buf: &mut Vec<u8>, id: u32, val: u64) {
    write_element_id(buf, id);
    let bytes = val.to_be_bytes();
    let first_nz = bytes.iter().position(|&b| b != 0).unwrap_or(7);
    let payload = &bytes[first_nz..];
    write_vint_size(buf, payload.len() as u64);
    buf.extend_from_slice(payload);
}

/// Write a signed integer element (minimal 1..=8 bytes).
pub fn write_sint_element(buf: &mut Vec<u8>, id: u32, val: i64) {
    write_element_id(buf, id);
    let bytes = val.to_be_bytes();
    let mut start = 0usize;
    while start < 7 {
        let b0 = bytes[start];
        let b1 = bytes[start + 1];
        if (b0 == 0x00 && (b1 & 0x80) == 0) || (b0 == 0xFF && (b1 & 0x80) != 0) {
            start += 1;
        } else {
            break;
        }
    }
    let payload = &bytes[start..];
    write_vint_size(buf, payload.len() as u64);
    buf.extend_from_slice(payload);
}

/// Write a 64-bit IEEE float element (8 bytes).
pub fn write_float64_element(buf: &mut Vec<u8>, id: u32, val: f64) {
    write_element_id(buf, id);
    write_vint_size(buf, 8);
    buf.extend_from_slice(&val.to_be_bytes());
}

/// Write a UTF-8/ASCII string element.
pub fn write_string_element(buf: &mut Vec<u8>, id: u32, s: &str) {
    write_bytes_element(buf, id, s.as_bytes());
}

/// Write a binary element.
pub fn write_bytes_element(buf: &mut Vec<u8>, id: u32, data: &[u8]) {
    write_element_id(buf, id);
    write_vint_size(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Write a master element wrapping pre-serialized `children`.
pub fn write_master_element(buf: &mut Vec<u8>, id: u32, children: &[u8]) {
    write_element_id(buf, id);
    write_vint_size(buf, children.len() as u64);
    buf.extend_from_slice(children);
}

// ---------------------------------------------------------------------------
// Block Lacing Parser (None, Xiph, Fixed-Size, EBML)
// ---------------------------------------------------------------------------

/// Parse a `SimpleBlock` or `Block` payload (starting right after the 1-byte flags field)
/// into one or more frame byte ranges relative to `frame_payload`.
///
/// `lacing_mode` is `(flags_byte >> 1) & 0x03`:
/// - `0`: no lacing (1 frame)
/// - `1`: Xiph lacing
/// - `2`: fixed-size lacing
/// - `3`: EBML lacing
pub fn parse_laced_frames(frame_payload: &[u8], lacing_mode: u8) -> Result<Vec<Range<usize>>> {
    match lacing_mode {
        0 => {
            let full = 0..frame_payload.len();
            Ok(vec![full])
        }
        1 => {
            // Xiph lacing
            let Some(&num_minus_1) = frame_payload.first() else {
                return Err(Error::invalid_data("mkv/lacing", "empty Xiph-laced block"));
            };
            let count = usize::from(num_minus_1) + 1;
            let mut pos = 1usize;
            let mut sizes = Vec::with_capacity(count);
            for _ in 0..(count - 1) {
                let mut sz = 0usize;
                loop {
                    let Some(&b) = frame_payload.get(pos) else {
                        return Err(Error::invalid_data(
                            "mkv/lacing",
                            "truncated Xiph lacing header",
                        ));
                    };
                    pos += 1;
                    sz += usize::from(b);
                    if b < 255 {
                        break;
                    }
                }
                sizes.push(sz);
            }
            build_laced_ranges(frame_payload.len(), pos, &sizes, count)
        }
        2 => {
            // Fixed-size lacing
            let Some(&num_minus_1) = frame_payload.first() else {
                return Err(Error::invalid_data(
                    "mkv/lacing",
                    "empty fixed-size laced block",
                ));
            };
            let count = usize::from(num_minus_1) + 1;
            let data_start = 1usize;
            let rem = frame_payload.len().saturating_sub(data_start);
            if !rem.is_multiple_of(count) {
                return Err(Error::invalid_data(
                    "mkv/lacing",
                    "fixed-size laced block payload not divisible by frame count",
                ));
            }
            let frame_sz = rem / count;
            let mut ranges = Vec::with_capacity(count);
            for i in 0..count {
                let s = data_start + i * frame_sz;
                ranges.push(s..(s + frame_sz));
            }
            Ok(ranges)
        }
        3 => {
            // EBML lacing
            let Some(&num_minus_1) = frame_payload.first() else {
                return Err(Error::invalid_data("mkv/lacing", "empty EBML-laced block"));
            };
            let count = usize::from(num_minus_1) + 1;
            let mut pos = 1usize;
            let mut sizes = Vec::with_capacity(count);
            if count > 1 {
                let (first_sz, used) = read_vint_data(&frame_payload[pos..])?;
                pos += used;
                let mut prev_sz = first_sz as i64;
                if prev_sz < 0 {
                    return Err(Error::invalid_data(
                        "mkv/lacing",
                        "negative first EBML lace size",
                    ));
                }
                sizes.push(prev_sz as usize);
                for _ in 1..(count - 1) {
                    let (delta, used) = read_svint_data(&frame_payload[pos..])?;
                    pos += used;
                    prev_sz = prev_sz.checked_add(delta).ok_or_else(|| {
                        Error::invalid_data("mkv/lacing", "EBML lace delta overflow")
                    })?;
                    if prev_sz < 0 {
                        return Err(Error::invalid_data(
                            "mkv/lacing",
                            "negative EBML lace frame size",
                        ));
                    }
                    sizes.push(prev_sz as usize);
                }
            }
            build_laced_ranges(frame_payload.len(), pos, &sizes, count)
        }
        _ => Err(Error::invalid_data("mkv/lacing", "invalid lacing mode")),
    }
}

fn build_laced_ranges(
    total_len: usize,
    mut offset: usize,
    explicit_sizes: &[usize],
    count: usize,
) -> Result<Vec<Range<usize>>> {
    let mut ranges = Vec::with_capacity(count);
    for &sz in explicit_sizes {
        let end = offset
            .checked_add(sz)
            .ok_or_else(|| Error::invalid_data("mkv/lacing", "laced frame offset overflow"))?;
        if end > total_len {
            return Err(Error::invalid_data(
                "mkv/lacing",
                "laced frame exceeds block payload",
            ));
        }
        ranges.push(offset..end);
        offset = end;
    }
    ranges.push(offset..total_len);
    Ok(ranges)
}

// ---------------------------------------------------------------------------
// Matroska Codec ID Mapping
// ---------------------------------------------------------------------------

/// Map a Matroska `CodecID` string (and optional `BitDepth` for PCM) to a [`CodecId`].
pub fn codec_id_from_matroska(codec_str: &str, bit_depth: u32) -> CodecId {
    match codec_str {
        // Video
        "V_VP8" => CodecId::Vp8,
        "V_VP9" => CodecId::Vp9,
        "V_AV1" => CodecId::Av1,
        "V_MPEG4/ISO/AVC" => CodecId::H264,
        "V_MPEGH/ISO/HEVC" => CodecId::Hevc,
        "V_MPEG1" => CodecId::Mpeg1video,
        "V_MPEG2" => CodecId::Mpeg2video,
        "V_MPEG4/ISO/SP" | "V_MPEG4/ISO/ASP" | "V_MPEG4/ISO/AP" | "V_MPEG4/MS/V3" => CodecId::Mpeg4,
        "V_MJPEG" => CodecId::Mjpeg,
        "V_FFV1" => CodecId::Ffv1,
        "V_PRORES" => CodecId::Prores,
        "V_UNCOMPRESSED" => CodecId::Rawvideo,
        // Audio
        "A_OPUS" | "A_OPUS/EXPERIMENTAL" => CodecId::Opus,
        "A_VORBIS" => CodecId::Vorbis,
        "A_AAC" | "A_AAC/MPEG2/MAIN" | "A_AAC/MPEG2/LC" | "A_AAC/MPEG2/LC/SBR"
        | "A_AAC/MPEG2/SSR" | "A_AAC/MPEG4/MAIN" | "A_AAC/MPEG4/LC" | "A_AAC/MPEG4/LC/SBR"
        | "A_AAC/MPEG4/SSR" | "A_AAC/MPEG4/LTP" => CodecId::Aac,
        "A_FLAC" => CodecId::Flac,
        "A_ALAC" => CodecId::Alac,
        "A_MPEG/L1" => CodecId::Mp1,
        "A_MPEG/L2" => CodecId::Mp2,
        "A_MPEG/L3" => CodecId::Mp3,
        "A_AC3" => CodecId::Ac3,
        "A_WAVPACK4" => CodecId::WavPack,
        "A_PCM/INT/LIT" => match bit_depth {
            8 => CodecId::PcmU8,
            24 => CodecId::PcmS24le,
            32 => CodecId::PcmS32le,
            _ => CodecId::PcmS16le,
        },
        "A_PCM/INT/BIG" => match bit_depth {
            8 => CodecId::PcmS8,
            24 => CodecId::PcmS24be,
            32 => CodecId::PcmS32be,
            _ => CodecId::PcmS16be,
        },
        "A_PCM/FLOAT/IEEE" => match bit_depth {
            64 => CodecId::PcmF64le,
            _ => CodecId::PcmF32le,
        },
        // Subtitles
        "S_TEXT/UTF8" | "S_TEXT/ASCII" => CodecId::SubSrt,
        "S_TEXT/ASS" | "S_TEXT/SSA" => CodecId::SubAss,
        "S_TEXT/WEBVTT" => CodecId::SubWebvtt,
        "S_HDMV/PGS" => CodecId::SubPgs,
        "S_VOBSUB" => CodecId::SubDvd,
        "S_DVBSUB" => CodecId::SubDvb,
        _ => CodecId::None,
    }
}

/// Map a [`CodecId`] to its canonical Matroska `CodecID` string (and optional PCM bit depth).
/// When `webm_only` is `true`, only WebM-compatible codecs (`VP8`, `VP9`, `AV1`, `Opus`, `Vorbis`, `WebVTT`) are allowed.
pub fn matroska_from_codec_id(codec_id: CodecId, webm_only: bool) -> Result<(&'static str, u32)> {
    if webm_only
        && !matches!(
            codec_id,
            CodecId::Vp8
                | CodecId::Vp9
                | CodecId::Av1
                | CodecId::Opus
                | CodecId::Vorbis
                | CodecId::SubWebvtt
        )
    {
        return Err(Error::unsupported(
            "mkv/webm",
            "WebM container supports only VP8/VP9/AV1 video, Opus/Vorbis audio, and WebVTT subtitles",
        ));
    }

    match codec_id {
        CodecId::Vp8 => Ok(("V_VP8", 0)),
        CodecId::Vp9 => Ok(("V_VP9", 0)),
        CodecId::Av1 => Ok(("V_AV1", 0)),
        CodecId::H264 => Ok(("V_MPEG4/ISO/AVC", 0)),
        CodecId::Hevc => Ok(("V_MPEGH/ISO/HEVC", 0)),
        CodecId::Mpeg1video => Ok(("V_MPEG1", 0)),
        CodecId::Mpeg2video => Ok(("V_MPEG2", 0)),
        CodecId::Mpeg4 => Ok(("V_MPEG4/ISO/ASP", 0)),
        CodecId::Mjpeg => Ok(("V_MJPEG", 0)),
        CodecId::Ffv1 => Ok(("V_FFV1", 0)),
        CodecId::Prores => Ok(("V_PRORES", 0)),
        CodecId::Rawvideo => Ok(("V_UNCOMPRESSED", 0)),
        CodecId::Opus => Ok(("A_OPUS", 0)),
        CodecId::Vorbis => Ok(("A_VORBIS", 0)),
        CodecId::Aac => Ok(("A_AAC", 0)),
        CodecId::Flac => Ok(("A_FLAC", 0)),
        CodecId::Alac => Ok(("A_ALAC", 0)),
        CodecId::Mp1 => Ok(("A_MPEG/L1", 0)),
        CodecId::Mp2 => Ok(("A_MPEG/L2", 0)),
        CodecId::Mp3 => Ok(("A_MPEG/L3", 0)),
        CodecId::Ac3 => Ok(("A_AC3", 0)),
        CodecId::WavPack => Ok(("A_WAVPACK4", 0)),
        CodecId::PcmU8 => Ok(("A_PCM/INT/LIT", 8)),
        CodecId::PcmS8 => Ok(("A_PCM/INT/BIG", 8)),
        CodecId::PcmS16le => Ok(("A_PCM/INT/LIT", 16)),
        CodecId::PcmS16be => Ok(("A_PCM/INT/BIG", 16)),
        CodecId::PcmS24le => Ok(("A_PCM/INT/LIT", 24)),
        CodecId::PcmS24be => Ok(("A_PCM/INT/BIG", 24)),
        CodecId::PcmS32le => Ok(("A_PCM/INT/LIT", 32)),
        CodecId::PcmS32be => Ok(("A_PCM/INT/BIG", 32)),
        CodecId::PcmF32le | CodecId::PcmF32be => Ok(("A_PCM/FLOAT/IEEE", 32)),
        CodecId::PcmF64le | CodecId::PcmF64be => Ok(("A_PCM/FLOAT/IEEE", 64)),
        CodecId::SubSrt | CodecId::SubText | CodecId::SubMovText => Ok(("S_TEXT/UTF8", 0)),
        CodecId::SubAss => Ok(("S_TEXT/ASS", 0)),
        CodecId::SubWebvtt => Ok(("S_TEXT/WEBVTT", 0)),
        CodecId::SubPgs => Ok(("S_HDMV/PGS", 0)),
        CodecId::SubDvd => Ok(("S_VOBSUB", 0)),
        CodecId::SubDvb => Ok(("S_DVBSUB", 0)),
        _ => Err(Error::unsupported(
            "mkv/codec",
            "unsupported codec for Matroska muxing",
        )),
    }
}

// ---------------------------------------------------------------------------
// Shared EBML AST Parser (`wedeo inspect`)
// ---------------------------------------------------------------------------

/// Parse an EBML element hierarchy up to `max_depth` levels deep for `wedeo inspect`.
pub fn parse_ebml_tree(io: &mut BufByteSource, max_depth: usize) -> Result<Vec<EbmlElement>> {
    let limit = io.size_hint().unwrap_or(u64::MAX);
    parse_ebml_subtree(io, limit, 0, max_depth)
}

/// Convenience helper to parse an EBML AST directly from an in-memory byte slice.
pub fn inspect_ebml_bytes(data: &[u8], max_depth: usize) -> Result<Vec<EbmlElement>> {
    let mut io = BufByteSource::new(Box::new(MemorySource::from_slice(data)));
    parse_ebml_tree(&mut io, max_depth)
}

fn parse_ebml_subtree(
    io: &mut BufByteSource,
    end_pos: u64,
    depth: usize,
    max_depth: usize,
) -> Result<Vec<EbmlElement>> {
    let mut elements = Vec::new();
    while io.tell() < end_pos {
        let Some(hdr) = read_element_header(io)? else {
            break;
        };
        let elem_end = hdr.end_offset().map_or(end_pos, |e| e.min(end_pos));
        let children = if is_master_element(hdr.id) && depth < max_depth {
            parse_ebml_subtree(io, elem_end, depth + 1, max_depth)?
        } else {
            Vec::new()
        };
        if io.tell() < elem_end {
            let rem = elem_end - io.tell();
            skip_element(io, rem)?;
        }
        elements.push(EbmlElement {
            id: hdr.id,
            name: hdr.name(),
            header_offset: hdr.header_offset,
            data_offset: hdr.data_offset,
            size: hdr.size,
            children,
        });
    }
    Ok(elements)
}
