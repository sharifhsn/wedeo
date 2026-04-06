// VP9 frame header parsing.
//
// Translated from FFmpeg's libavcodec/vp9.c (`decode_frame_header`).
// Reads the uncompressed header from raw bytes (byte-aligned GetBits-style)
// then switches to the BoolDecoder for the compressed header.
//
// The uncompressed header uses a minimal hand-rolled bit reader (GetBits)
// because it is byte-aligned and does not use the range coder.

use wedeo_core::error::{Error, Result};

use crate::bool_decoder::BoolDecoder;
use crate::data::{DEFAULT_COEF_PROBS, PARETO8};
use crate::prob::CoefProbArray;
use crate::types::{BitstreamProfile, ProbContext};

// ---------------------------------------------------------------------------
// New types
// ---------------------------------------------------------------------------

/// VP9 frame type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FrameType {
    #[default]
    KeyFrame = 0,
    InterFrame = 1,
}

/// VP9 transform mode.
///
/// Mirrors `enum TxfmMode` in vp9.h.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TxMode {
    #[default]
    Only4x4 = 0,
    Allow8x8 = 1,
    Allow16x16 = 2,
    Allow32x32 = 3,
    TxModeSelect = 4,
}

impl TryFrom<u8> for TxMode {
    type Error = u8;
    fn try_from(v: u8) -> std::result::Result<Self, u8> {
        match v {
            0 => Ok(Self::Only4x4),
            1 => Ok(Self::Allow8x8),
            2 => Ok(Self::Allow16x16),
            3 => Ok(Self::Allow32x32),
            4 => Ok(Self::TxModeSelect),
            _ => Err(v),
        }
    }
}

/// VP9 colour space.
///
/// Matches the 3-bit field in the bitstream (Table 9 of the VP9 spec).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ColorSpace {
    #[default]
    Unknown = 0,
    Bt601 = 1,
    Bt709 = 2,
    Smpte170 = 3,
    Smpte240 = 4,
    Bt2020Ncl = 5,
    Reserved = 6,
    Rgb = 7,
}

impl From<u8> for ColorSpace {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Unknown,
            1 => Self::Bt601,
            2 => Self::Bt709,
            3 => Self::Smpte170,
            4 => Self::Smpte240,
            5 => Self::Bt2020Ncl,
            6 => Self::Reserved,
            7 => Self::Rgb,
            _ => Self::Unknown,
        }
    }
}

/// VP9 colour range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ColorRange {
    /// Studio swing (limited range).
    #[default]
    Limited = 0,
    /// Full swing (JPEG range).
    Full = 1,
}

/// Per-segment feature flags and values.
///
/// Mirrors `s->s.h.segmentation.feat[i]` in vp9shared.h.
#[derive(Clone, Copy, Debug, Default)]
pub struct SegmentFeature {
    pub q_enabled: bool,
    pub lf_enabled: bool,
    pub ref_enabled: bool,
    pub skip_enabled: bool,
    /// Quantizer delta / absolute value (signed, 9-bit field).
    pub q_val: i16,
    /// Loop-filter delta / absolute value (signed, 7-bit field).
    pub lf_val: i8,
    /// Reference frame value (2 bits).
    pub ref_val: u8,
    /// Pre-computed loop-filter levels: `lflvl[ref_frame][mode_is_not_zeromv]`.
    /// ref_frame: 0=INTRA, 1=LAST, 2=GOLDEN, 3=ALTREF.
    /// Mirrors `s->s.h.segmentation.feat[i].lflvl` in FFmpeg vp9.c:768–792.
    pub lflvl: [[u8; 2]; 4],
}

/// Segmentation parameters parsed from the uncompressed header.
#[derive(Clone, Debug, Default)]
pub struct SegmentationParams {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal: bool,
    pub absolute_vals: bool,
    /// Per-segment tree probabilities (7 entries).
    pub prob: [u8; 7],
    /// Temporal prediction probabilities (3 entries).
    pub pred_prob: [u8; 3],
    /// Per-segment feature data (8 segments).
    pub feat: [SegmentFeature; 8],
}

/// All parsed fields from one VP9 frame header.
#[derive(Clone, Debug)]
pub struct FrameHeader {
    pub profile: BitstreamProfile,
    pub show_existing_frame: bool,
    /// Index of the reference frame to show (only valid if show_existing_frame).
    pub show_existing_frame_ref: u8,
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub error_resilient: bool,
    pub intra_only: bool,
    /// Context reset mode (0-3) for inter frames.
    pub reset_context: u8,
    /// Refresh flags bitmask (8 bits) for the reference frame slots.
    pub refresh_ref_mask: u8,
    /// Reference frame indices for inter prediction [3].
    pub ref_idx: [u8; 3],
    /// Reference frame sign bias [3].
    pub sign_bias: [bool; 3],
    pub bit_depth: u8,
    pub width: u32,
    pub height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub color_space: ColorSpace,
    pub color_range: ColorRange,
    // Loop filter.
    pub filter_level: u8,
    pub sharpness_level: u8,
    pub mode_ref_delta_enabled: bool,
    pub mode_ref_delta_updated: bool,
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
    // Quantization.
    pub base_q_idx: u8,
    pub y_dc_delta_q: i8,
    pub uv_dc_delta_q: i8,
    pub uv_ac_delta_q: i8,
    pub lossless: bool,
    // Segmentation.
    pub segmentation: SegmentationParams,
    // Tiles.
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    // Compressed header size in bytes (16-bit field).
    pub compressed_header_size: u16,
    /// Byte offset in the original `data` slice where the compressed header begins.
    pub compressed_header_offset: usize,
    /// Byte offset where tile data begins (= compressed_header_offset + compressed_header_size).
    pub tile_data_offset: usize,
    // Probability context after compressed-header updates.
    pub prob: ProbContext,
    /// Coefficient probabilities (updated during compressed header).
    /// Layout: [tx_size(4)][block_type(2)][intra(2)][band(6)][ctx(6)][3].
    #[allow(clippy::type_complexity)]
    pub coef: [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4],
    // Tx mode.
    pub tx_mode: TxMode,
    // Inter-frame specific.
    pub high_precision_mvs: bool,
    pub filter_mode: u8, // 0-3 = fixed, 4 = switchable (FILTER_SWITCHABLE)
    pub allow_comp_inter: bool,
    pub comp_pred_mode: u8, // 0=single, 1=compound, 2=switchable
    pub fix_comp_ref: u8,
    pub var_comp_ref: [u8; 2],
    pub refresh_ctx: bool,
    pub parallel_mode: bool,
    pub frame_ctx_id: u8,
    pub use_last_frame_mvs: bool,
    /// If width/height are 0, this indicates which ref_idx[i] to copy
    /// dimensions from. -1 means explicit dimensions were given.
    pub size_from_ref: i8,
}

// ---------------------------------------------------------------------------
// Minimal byte-aligned bit reader for the uncompressed header.
// ---------------------------------------------------------------------------

/// Byte-aligned bit reader (like FFmpeg's GetBitContext for this purpose).
struct GetBits<'a> {
    data: &'a [u8],
    /// Current byte position (in bits from the start).
    pos: usize,
}

impl<'a> GetBits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Read `n` bits (MSB first), returning them in the low bits of a u32.
    fn get_bits(&mut self, n: usize) -> Result<u32> {
        let mut v: u32 = 0;
        for _ in 0..n {
            let byte_idx = self.pos / 8;
            if byte_idx >= self.data.len() {
                return Err(Error::InvalidData);
            }
            let bit = (self.data[byte_idx] >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | bit as u32;
            self.pos += 1;
        }
        Ok(v)
    }

    /// Read a single bit.
    #[inline]
    fn get_bit(&mut self) -> Result<bool> {
        Ok(self.get_bits(1)? != 0)
    }

    /// Return the current byte offset (rounded up to the next byte boundary).
    fn byte_pos(&self) -> usize {
        self.pos.div_ceil(8)
    }
}

// ---------------------------------------------------------------------------
// Inverse recenter helper (for update_prob)
// ---------------------------------------------------------------------------

fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v & 1 != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

// ---------------------------------------------------------------------------
// Differential probability update (compressed header)
// ---------------------------------------------------------------------------

/// The inv_map_table used in update_prob.
///
/// Verbatim copy from FFmpeg's vp9.c.
const INV_MAP_TABLE: [u8; 255] = [
    7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254, 1,
    2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166,
    167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
    187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248,
    249, 250, 251, 252, 253, 253,
];

/// Differential probability update from BoolDecoder.
///
/// Mirrors `update_prob` in FFmpeg's vp9.c.
fn update_prob(bd: &mut BoolDecoder<'_>, p: u8) -> u8 {
    let d = if !bd.get() {
        bd.get_uint(4)
    } else if !bd.get() {
        bd.get_uint(4) + 16
    } else if !bd.get() {
        bd.get_uint(5) + 32
    } else {
        let mut v = bd.get_uint(7);
        if v >= 65 {
            v = (v << 1) - 65 + u32::from(bd.get());
        }
        v + 64
    } as usize;

    let mapped = INV_MAP_TABLE[d] as i32;
    let pi = p as i32;
    let new_p = if pi <= 128 {
        1 + inv_recenter_nonneg(mapped, pi - 1)
    } else {
        255 - inv_recenter_nonneg(mapped, 255 - pi)
    };
    new_p as u8
}

// ---------------------------------------------------------------------------
// Default probability context
// ---------------------------------------------------------------------------

/// Default ProbContext (all inter-frame probs set to the VP9 spec defaults).
///
/// These values are from FFmpeg's `ff_vp9_default_probs` in vp9data.c.
/// For keyframes the non-coef values are not used during decoding, but we
/// initialise them for completeness.
pub fn default_prob_context() -> ProbContext {
    use crate::types::MvCompProbs;

    // Values from vp9data.c ff_vp9_default_probs.
    ProbContext {
        y_mode: [
            [65, 32, 18, 144, 162, 194, 41, 51, 98],
            [132, 68, 18, 165, 217, 196, 45, 40, 78],
            [173, 80, 19, 176, 240, 193, 64, 35, 46],
            [221, 135, 38, 194, 248, 121, 96, 85, 29],
        ],
        uv_mode: [
            [ 48,  12, 154, 155, 139,  90,  34, 117, 119],
            [ 67,   6,  25, 204, 243, 158,  13,  21,  96],
            [120,   7,  76, 176, 208, 126,  28,  54, 103],
            [ 97,   5,  44, 131, 176, 139,  48,  68,  97],
            [ 83,   5,  42, 156, 111, 152,  26,  49, 152],
            [ 80,   5,  58, 178,  74,  83,  33,  62, 145],
            [ 86,   5,  32, 154, 192, 168,  14,  22, 163],
            [ 77,   7,  64, 116, 132, 122,  37, 126, 120],
            [ 85,   5,  32, 156, 216, 148,  19,  29,  73],
            [101,  21, 107, 181, 192, 103,  19,  67, 125],
        ],
        filter: [[235, 162], [36, 255], [34, 3], [149, 144]],
        mv_mode: [
            [  2, 173,  34],
            [  7, 145,  85],
            [  7, 166,  63],
            [  7,  94,  66],
            [  8,  64,  46],
            [ 17,  81,  31],
            [ 25,  29,  30],
        ],
        intra: [9, 102, 187, 225],
        comp: [239, 183, 119, 96, 41],
        single_ref: [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]],
        comp_ref: [50, 126, 123, 221, 226],
        tx32p: [[3, 136, 37], [5, 52, 13]],
        tx16p: [[20, 152], [15, 101]],
        tx8p: [100, 66],
        skip: [192, 128, 64],
        mv_joint: [32, 64, 96],
        mv_comp: [
            MvCompProbs {
                sign: 128,
                classes: [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
                class0: 216,
                bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
                class0_fp: [[128, 128, 64], [96, 112, 64]],
                fp: [64, 96, 64],
                class0_hp: 160,
                hp: 128,
            },
            MvCompProbs {
                sign: 128,
                classes: [216, 128, 176, 160, 176, 176, 192, 198, 198, 208],
                class0: 208,
                bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
                class0_fp: [[128, 128, 64], [96, 112, 64]],
                fp: [64, 96, 64],
                class0_hp: 160,
                hp: 128,
            },
        ],
        // partition[block_level][above_bit][left_bit][3]
        // Default values from FFmpeg's ff_vp9_default_probs (vp9data.c).
        // Combined context c = above_bit | (left_bit << 1), so:
        //   [0][0] → c=0; [1][0] → c=1; [0][1] → c=2; [1][1] → c=3
        // Unused extra indices (2 and 3 in each dimension) are zero-filled.
        partition: [
            // bl=0: 64x64 → 32x32
            [
                [[222,  34,  30], [ 58,  32,  12], [0, 0, 0], [0, 0, 0]], // a=0: c=0, c=2
                [[ 72,  16,  44], [ 10,   7,   6], [0, 0, 0], [0, 0, 0]], // a=1: c=1, c=3
                [[0, 0, 0]; 4],
                [[0, 0, 0]; 4],
            ],
            // bl=1: 32x32 → 16x16
            [
                [[177,  58,  59], [ 52,  79,  25], [0, 0, 0], [0, 0, 0]],
                [[ 68,  26,  63], [ 17,  14,  12], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0]; 4],
                [[0, 0, 0]; 4],
            ],
            // bl=2: 16x16 → 8x8
            [
                [[174,  73,  87], [ 82,  99,  50], [0, 0, 0], [0, 0, 0]],
                [[ 92,  41,  83], [ 53,  39,  39], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0]; 4],
                [[0, 0, 0]; 4],
            ],
            // bl=3: 8x8 → 4x4
            [
                [[199, 122, 141], [148, 133, 118], [0, 0, 0], [0, 0, 0]],
                [[147,  63, 159], [121, 104, 114], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0]; 4],
                [[0, 0, 0]; 4],
            ],
        ],
    }
}

// ---------------------------------------------------------------------------
// Colour config helpers
// ---------------------------------------------------------------------------

struct ColorConfig {
    bit_depth: u8,
    color_space: ColorSpace,
    color_range: ColorRange,
    subsampling_x: bool,
    subsampling_y: bool,
}

fn read_color_config(gb: &mut GetBits<'_>, profile: u8) -> Result<ColorConfig> {
    // bit_depth: for profiles 0/1 always 8; for profiles 2/3 read 1 bit (0→10, 1→12).
    let bit_depth = if profile <= 1 {
        8u8
    } else {
        if gb.get_bit()? { 12 } else { 10 }
    };

    let cs_raw = gb.get_bits(3)? as u8;
    let color_space = ColorSpace::from(cs_raw);

    let (color_range, subsampling_x, subsampling_y) = if color_space == ColorSpace::Rgb {
        // RGB: always full range, no subsampling.
        if profile & 1 != 0 {
            // Reserved bit must be zero.
            if gb.get_bit()? {
                return Err(Error::InvalidData);
            }
        } else {
            // RGB not supported in profile 0.
            return Err(Error::InvalidData);
        }
        (ColorRange::Full, false, false)
    } else {
        let cr = if gb.get_bit()? {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        let (ss_h, ss_v) = if profile & 1 != 0 {
            let h = gb.get_bit()?;
            let v = gb.get_bit()?;
            // YUV 4:2:0 not supported in odd profiles.
            if h && v {
                return Err(Error::InvalidData);
            }
            // Reserved bit.
            if gb.get_bit()? {
                return Err(Error::InvalidData);
            }
            (h, v)
        } else {
            (true, true) // 4:2:0 for profiles 0/2
        };
        (cr, ss_h, ss_v)
    };

    Ok(ColorConfig {
        bit_depth,
        color_space,
        color_range,
        subsampling_x,
        subsampling_y,
    })
}

// ---------------------------------------------------------------------------
// get_sbits_inv helper (sign bit at the end)
// ---------------------------------------------------------------------------

/// Read `n` magnitude bits then a sign bit; returns `-(v)` if sign=1.
///
/// Mirrors `get_sbits_inv` in FFmpeg's vp9.c.
fn get_sbits_inv(gb: &mut GetBits<'_>, n: usize) -> Result<i8> {
    let v = gb.get_bits(n)? as i8;
    let neg = gb.get_bit()?;
    Ok(if neg { -v } else { v })
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Parse a VP9 frame header from `data`.
///
/// Returns the parsed `FrameHeader` and the byte offset in `data` where the
/// tile bitstream begins (i.e. after the compressed header).
pub fn decode_frame_header(
    data: &[u8],
    prob_ctx: &[ProbContext; 4],
    coef_ctx: &[CoefProbArray; 4],
) -> Result<(FrameHeader, usize)> {
    let mut gb = GetBits::new(data);

    // Frame marker (2 bits, must be 0b10 = 2).
    if gb.get_bits(2)? != 2 {
        return Err(Error::InvalidData);
    }

    // Profile: bit0 then bit1<<1; profile 3 has an additional reserved-zero bit.
    let prof_lo = gb.get_bits(1)? as u8;
    let prof_hi = gb.get_bits(1)? as u8;
    let profile_raw = prof_lo | (prof_hi << 1);
    if profile_raw == 3 {
        // Reserved zero bit.
        if gb.get_bit()? {
            return Err(Error::InvalidData);
        }
    }
    if profile_raw > 3 {
        return Err(Error::InvalidData);
    }
    // Clamp so TryFrom succeeds.
    let profile = BitstreamProfile::try_from(profile_raw).map_err(|_| Error::InvalidData)?;

    // show_existing_frame flag.
    if gb.get_bit()? {
        let ref_idx = gb.get_bits(3)? as u8;
        // Build a minimal header for show_existing_frame.
        let hdr = FrameHeader {
            profile,
            show_existing_frame: true,
            show_existing_frame_ref: ref_idx,
            frame_type: FrameType::KeyFrame,
            show_frame: true,
            error_resilient: false,
            intra_only: false,
            reset_context: 0,
            refresh_ref_mask: 0,
            ref_idx: [0; 3],
            sign_bias: [false; 3],
            bit_depth: 8,
            width: 0,
            height: 0,
            render_width: 0,
            render_height: 0,
            subsampling_x: true,
            subsampling_y: true,
            color_space: ColorSpace::Unknown,
            color_range: ColorRange::Limited,
            filter_level: 0,
            sharpness_level: 0,
            mode_ref_delta_enabled: false,
            mode_ref_delta_updated: false,
            ref_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
            base_q_idx: 0,
            y_dc_delta_q: 0,
            uv_dc_delta_q: 0,
            uv_ac_delta_q: 0,
            lossless: false,
            segmentation: SegmentationParams::default(),
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            compressed_header_size: 0,
            compressed_header_offset: 0,
            tile_data_offset: 0,
            prob: default_prob_context(),
            coef: DEFAULT_COEF_PROBS,
            tx_mode: TxMode::Only4x4,
            high_precision_mvs: false,
            filter_mode: 0,
            allow_comp_inter: false,
            comp_pred_mode: 0,
            fix_comp_ref: 0,
            var_comp_ref: [0; 2],
            refresh_ctx: false,
            parallel_mode: false,
            frame_ctx_id: 0,
            use_last_frame_mvs: false,
            size_from_ref: -1,
        };
        let tile_off = gb.byte_pos();
        return Ok((hdr, tile_off));
    }

    // frame_type: 0=keyframe, 1=inter.
    let is_keyframe = !gb.get_bit()?;
    let frame_type = if is_keyframe {
        FrameType::KeyFrame
    } else {
        FrameType::InterFrame
    };

    let show_frame = gb.get_bit()?;
    let error_resilient = gb.get_bit()?;

    // Default loop-filter delta values (reset on keyframe/errorres).
    let mut ref_deltas: [i8; 4] = [1, 0, -1, -1];
    let mut mode_deltas: [i8; 2] = [0, 0];
    let mut seg = SegmentationParams::default();

    let mut bit_depth = 8u8;
    let mut color_space = ColorSpace::Unknown;
    let mut color_range = ColorRange::Limited;
    let mut subsampling_x = true;
    let mut subsampling_y = true;

    let width: u32;
    let height: u32;
    let render_width: u32;
    let render_height: u32;
    let mut intra_only = false;
    let mut reset_context: u8 = 0;
    let refresh_ref_mask: u8;
    let mut ref_idx = [0u8; 3];
    let mut sign_bias = [false; 3];
    let mut high_precision_mvs = false;
    let mut filter_mode: u8 = 0;
    let mut allow_comp_inter = false;
    let mut comp_pred_mode: u8 = 0;
    let mut size_from_ref: i8 = -1;
    let mut fix_comp_ref: u8 = 0;
    let mut var_comp_ref: [u8; 2] = [0; 2];

    if is_keyframe {
        // Sync code: 0x49, 0x83, 0x42.
        let sync = gb.get_bits(24)?;
        if sync != 0x49_83_42 {
            return Err(Error::InvalidData);
        }
        let cc = read_color_config(&mut gb, profile_raw)?;
        bit_depth = cc.bit_depth;
        color_space = cc.color_space;
        color_range = cc.color_range;
        subsampling_x = cc.subsampling_x;
        subsampling_y = cc.subsampling_y;

        refresh_ref_mask = 0xff;
        width = gb.get_bits(16)? + 1;
        height = gb.get_bits(16)? + 1;
        if gb.get_bit()? {
            // display size — read and store render dimensions.
            render_width = gb.get_bits(16)? + 1;
            render_height = gb.get_bits(16)? + 1;
        } else {
            render_width = width;
            render_height = height;
        }
    } else {
        // Inter / intra-only frame.
        intra_only = if !show_frame { gb.get_bit()? } else { false };
        reset_context = if error_resilient {
            0
        } else {
            gb.get_bits(2)? as u8
        };

        if intra_only {
            let sync = gb.get_bits(24)?;
            if sync != 0x49_83_42 {
                return Err(Error::InvalidData);
            }
            if profile_raw >= 1 {
                let cc = read_color_config(&mut gb, profile_raw)?;
                bit_depth = cc.bit_depth;
                color_space = cc.color_space;
                color_range = cc.color_range;
                subsampling_x = cc.subsampling_x;
                subsampling_y = cc.subsampling_y;
            } else {
                // Profile 0 intra-only: 8-bit YUV 4:2:0.
                bit_depth = 8;
                color_space = ColorSpace::Bt601;
                color_range = ColorRange::Limited;
                subsampling_x = true;
                subsampling_y = true;
            }
            refresh_ref_mask = gb.get_bits(8)? as u8;
            width = gb.get_bits(16)? + 1;
            height = gb.get_bits(16)? + 1;
            if gb.get_bit()? {
                render_width = gb.get_bits(16)? + 1;
                render_height = gb.get_bits(16)? + 1;
            } else {
                render_width = width;
                render_height = height;
            }
        } else {
            // True inter frame.
            refresh_ref_mask = gb.get_bits(8)? as u8;
            ref_idx[0] = gb.get_bits(3)? as u8;
            sign_bias[0] = gb.get_bit()? && !error_resilient;
            ref_idx[1] = gb.get_bits(3)? as u8;
            sign_bias[1] = gb.get_bit()? && !error_resilient;
            ref_idx[2] = gb.get_bits(3)? as u8;
            sign_bias[2] = gb.get_bit()? && !error_resilient;

            // Dimensions: may be inherited from a reference frame or explicit.
            // We cannot look up reference dimensions without a frame buffer store,
            // so we read the "use_ref_frame_size" flags and fall back to explicit
            // dimensions (size_from_ref_flag bits are consumed but the sizes we
            // store will be the ones the caller must resolve from their ref store).
            let use_ref0 = gb.get_bit()?;
            let use_ref1 = if !use_ref0 { gb.get_bit()? } else { false };
            let use_ref2 = if !use_ref0 && !use_ref1 {
                gb.get_bit()?
            } else {
                false
            };
            if use_ref0 || use_ref1 || use_ref2 {
                // Caller must fill in width/height from the referenced frame.
                let which = if use_ref0 {
                    0
                } else if use_ref1 {
                    1
                } else {
                    2
                };
                size_from_ref = which;
                width = 0;
                height = 0;
            } else {
                width = gb.get_bits(16)? + 1;
                height = gb.get_bits(16)? + 1;
            }

            if gb.get_bit()? {
                render_width = gb.get_bits(16)? + 1;
                render_height = gb.get_bits(16)? + 1;
            } else {
                render_width = width;
                render_height = height;
            }

            high_precision_mvs = gb.get_bit()?;
            filter_mode = if gb.get_bit()? {
                4 // FILTER_SWITCHABLE
            } else {
                gb.get_bits(2)? as u8
            };

            allow_comp_inter = (sign_bias[0] != sign_bias[1]) || (sign_bias[0] != sign_bias[2]);
            if allow_comp_inter {
                if sign_bias[0] == sign_bias[1] {
                    fix_comp_ref = 2;
                    var_comp_ref = [0, 1];
                } else if sign_bias[0] == sign_bias[2] {
                    fix_comp_ref = 1;
                    var_comp_ref = [0, 2];
                } else {
                    fix_comp_ref = 0;
                    var_comp_ref = [1, 2];
                }
            }
        }
    }

    let refresh_ctx = if error_resilient {
        false
    } else {
        gb.get_bit()?
    };
    let parallel_mode = if error_resilient { true } else { gb.get_bit()? };
    let frame_ctx_id_raw = gb.get_bits(2)? as u8;
    // BUG: libvpx ignores framectxid for keyframes.
    let frame_ctx_id = if is_keyframe || intra_only {
        0
    } else {
        frame_ctx_id_raw
    };

    // Loop filter.
    if is_keyframe || error_resilient || intra_only {
        // Reset to spec defaults.
        ref_deltas = [1, 0, -1, -1];
        mode_deltas = [0, 0];
        seg = SegmentationParams::default();
    }

    let filter_level = gb.get_bits(6)? as u8;
    let sharpness_level = gb.get_bits(3)? as u8;

    let mode_ref_delta_enabled = gb.get_bit()?;
    let mut mode_ref_delta_updated = false;
    if mode_ref_delta_enabled {
        mode_ref_delta_updated = gb.get_bit()?;
        if mode_ref_delta_updated {
            for rd in ref_deltas.iter_mut() {
                if gb.get_bit()? {
                    *rd = get_sbits_inv(&mut gb, 6)?;
                }
            }
            for md in mode_deltas.iter_mut() {
                if gb.get_bit()? {
                    *md = get_sbits_inv(&mut gb, 6)?;
                }
            }
        }
    }

    // Quantization.
    let base_q_idx = gb.get_bits(8)? as u8;
    let y_dc_delta_q = if gb.get_bit()? {
        get_sbits_inv(&mut gb, 4)?
    } else {
        0
    };
    let uv_dc_delta_q = if gb.get_bit()? {
        get_sbits_inv(&mut gb, 4)?
    } else {
        0
    };
    let uv_ac_delta_q = if gb.get_bit()? {
        get_sbits_inv(&mut gb, 4)?
    } else {
        0
    };
    let lossless = base_q_idx == 0 && y_dc_delta_q == 0 && uv_dc_delta_q == 0 && uv_ac_delta_q == 0;

    // Segmentation.
    seg.enabled = gb.get_bit()?;
    if seg.enabled {
        seg.update_map = gb.get_bit()?;
        if seg.update_map {
            for i in 0..7 {
                seg.prob[i] = if gb.get_bit()? {
                    gb.get_bits(8)? as u8
                } else {
                    255
                };
            }
            seg.temporal = gb.get_bit()?;
            if seg.temporal {
                for i in 0..3 {
                    seg.pred_prob[i] = if gb.get_bit()? {
                        gb.get_bits(8)? as u8
                    } else {
                        255
                    };
                }
            }
        }
        if gb.get_bit()? {
            seg.absolute_vals = gb.get_bit()?;
            for i in 0..8 {
                let q_en = gb.get_bit()?;
                seg.feat[i].q_enabled = q_en;
                if q_en {
                    // 8-bit magnitude + sign bit (get_sbits_inv).
                    let mag = gb.get_bits(8)? as i16;
                    let neg = gb.get_bit()?;
                    seg.feat[i].q_val = if neg { -mag } else { mag };
                }
                let lf_en = gb.get_bit()?;
                seg.feat[i].lf_enabled = lf_en;
                if lf_en {
                    seg.feat[i].lf_val = get_sbits_inv(&mut gb, 6)?;
                }
                let ref_en = gb.get_bit()?;
                seg.feat[i].ref_enabled = ref_en;
                if ref_en {
                    seg.feat[i].ref_val = gb.get_bits(2)? as u8;
                }
                seg.feat[i].skip_enabled = gb.get_bit()?;
            }
        }
    } else {
        seg.temporal = false;
        seg.update_map = false;
    }

    // Pre-compute per-segment loop-filter levels (lflvl).
    // Mirrors FFmpeg vp9.c:768–792.
    {
        let n_seg = if seg.enabled { 8 } else { 1 };
        let sh = u32::from(filter_level >= 32);
        for i in 0..n_seg {
            let base = if seg.enabled && seg.feat[i].lf_enabled {
                if seg.absolute_vals {
                    (seg.feat[i].lf_val as i32).clamp(0, 63)
                } else {
                    (filter_level as i32 + seg.feat[i].lf_val as i32).clamp(0, 63)
                }
            } else {
                filter_level as i32
            };
            if mode_ref_delta_enabled {
                // INTRA_FRAME (ref=0): both mode slots get ref_deltas[0] only.
                let intra_lvl = (base + ((ref_deltas[0] as i32) << sh)).clamp(0, 63) as u8;
                seg.feat[i].lflvl[0][0] = intra_lvl;
                seg.feat[i].lflvl[0][1] = intra_lvl;
                // Inter refs 1..4 (LAST, GOLDEN, ALTREF).
                for (lf, rd) in seg.feat[i].lflvl[1..4]
                    .iter_mut()
                    .zip(ref_deltas[1..4].iter())
                {
                    lf[0] =
                        (base + ((*rd as i32 + mode_deltas[0] as i32) << sh)).clamp(0, 63) as u8;
                    lf[1] =
                        (base + ((*rd as i32 + mode_deltas[1] as i32) << sh)).clamp(0, 63) as u8;
                }
            } else {
                let lvl = base as u8;
                seg.feat[i].lflvl = [[lvl; 2]; 4];
            }
        }
    }

    // Tile info.
    // sb_cols = (width + 63) / 64; min log2_tile_cols so that sb_cols <= 64 << log2.
    let sb_cols = width.saturating_add(63) / 64;
    let mut tile_cols_log2: u8 = 0;
    while sb_cols > (64u32 << tile_cols_log2) {
        tile_cols_log2 += 1;
    }
    // max allowed log2_tile_cols: largest k such that (sb_cols >> k) >= 4.
    let mut max_log2: u8 = 0;
    while (sb_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    max_log2 = max_log2.saturating_sub(1);
    while max_log2 > tile_cols_log2 {
        if gb.get_bit()? {
            tile_cols_log2 += 1;
        } else {
            break;
        }
    }

    // tile_rows_log2 uses decode012: read 0→0, 10→1, 11→2.
    let tile_rows_log2 = if gb.get_bit()? {
        1 + u8::from(gb.get_bit()?)
    } else {
        0
    };

    // Compressed header size (16 bits, byte-aligned).
    let compressed_header_size = gb.get_bits(16)? as u16;

    // Byte offset of the compressed header start (aligned to next byte).
    let compressed_header_offset = gb.byte_pos();

    let tile_data_offset = compressed_header_offset + compressed_header_size as usize;
    if tile_data_offset > data.len() {
        return Err(Error::InvalidData);
    }

    // --- Compressed header (BoolDecoder) ---
    let compressed_data = &data[compressed_header_offset..tile_data_offset];
    let mut bd = BoolDecoder::new(compressed_data).ok_or(Error::InvalidData)?;

    // Marker bit (must be 0).
    if bd.get_prob(128) {
        return Err(Error::InvalidData);
    }

    // Initialize probability context from the selected context slot.
    // For keyframes/intra-only, frame_ctx_id is forced to 0 and probs
    // reset to defaults (matching FFmpeg vp9.c behavior).
    let (mut prob, mut coef) = if is_keyframe || intra_only {
        (default_prob_context(), DEFAULT_COEF_PROBS)
    } else {
        (
            prob_ctx[frame_ctx_id as usize].clone(),
            coef_ctx[frame_ctx_id as usize],
        )
    };

    // Tx mode.
    let tx_mode = if lossless {
        TxMode::Only4x4
    } else {
        let raw = bd.get_uint(2) as u8;
        let raw = if raw == 3 {
            3 + u8::from(bd.get())
        } else {
            raw
        };
        TxMode::try_from(raw).map_err(|_| Error::InvalidData)?
    };

    // Tx prob updates (only when tx_mode == TxModeSelect).
    if tx_mode == TxMode::TxModeSelect {
        for i in 0..2 {
            if bd.get_prob(252) {
                prob.tx8p[i] = update_prob(&mut bd, prob.tx8p[i]);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                if bd.get_prob(252) {
                    prob.tx16p[i][j] = update_prob(&mut bd, prob.tx16p[i][j]);
                }
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                if bd.get_prob(252) {
                    prob.tx32p[i][j] = update_prob(&mut bd, prob.tx32p[i][j]);
                }
            }
        }
    }

    // Coefficient prob updates.
    // Indexed [tx_size][block_type][intra][band][ctx][3].
    // Reference coef probs come from the context selected by frame_ctx_id.
    let tx_mode_max = match tx_mode {
        TxMode::Only4x4 => 0,
        TxMode::Allow8x8 => 1,
        TxMode::Allow16x16 => 2,
        TxMode::Allow32x32 => 3,
        TxMode::TxModeSelect => 3,
    };
    #[allow(clippy::needless_range_loop)] // 5-level nested index into coef[i][j][k][l][m]
    for i in 0..=tx_mode_max {
        if bd.get() {
            // update_flag set: conditionally update each probability.
            for j in 0..2usize {
                for k in 0..2usize {
                    for l in 0..6usize {
                        for m in 0..6usize {
                            if m >= 3 && l == 0 {
                                break;
                            }
                            let r = coef[i][j][k][l][m];
                            let p = &mut coef[i][j][k][l][m];
                            for n in 0..3usize {
                                if bd.get_prob(252) {
                                    p[n] = update_prob(&mut bd, r[n]);
                                }
                                // No else: p[n] already has the correct base value
                                // from the context slot (or defaults for keyframes).
                            }
                            // Extend with model pareto8 for entries [3..11].
                            // We store only [3] here (the full 11-entry array is
                            // used during tile decoding; we keep only 3 entries).
                        }
                    }
                }
            }
        } else {
            // No updates: coef already has the correct base values from
            // the context slot (or defaults for keyframes). Nothing to do.
        }
        if tx_mode as u8 == i as u8 {
            break;
        }
    }

    // Skip prob updates.
    for i in 0..3 {
        if bd.get_prob(252) {
            prob.skip[i] = update_prob(&mut bd, prob.skip[i]);
        }
    }

    // Inter-frame prob updates.
    if !is_keyframe && !intra_only {
        // MV mode.
        for i in 0..7 {
            for j in 0..3 {
                if bd.get_prob(252) {
                    prob.mv_mode[i][j] = update_prob(&mut bd, prob.mv_mode[i][j]);
                }
            }
        }

        // Interpolation filter.
        if filter_mode == 4 {
            // FILTER_SWITCHABLE
            for i in 0..4 {
                for j in 0..2 {
                    if bd.get_prob(252) {
                        prob.filter[i][j] = update_prob(&mut bd, prob.filter[i][j]);
                    }
                }
            }
        }

        // Intra prob.
        for i in 0..4 {
            if bd.get_prob(252) {
                prob.intra[i] = update_prob(&mut bd, prob.intra[i]);
            }
        }

        // Compound prediction mode.
        if allow_comp_inter {
            let cp0 = u8::from(bd.get());
            comp_pred_mode = if cp0 == 1 { 1 + u8::from(bd.get()) } else { 0 };
            if comp_pred_mode == 2 {
                // Switchable: update comp probs.
                for i in 0..5 {
                    if bd.get_prob(252) {
                        prob.comp[i] = update_prob(&mut bd, prob.comp[i]);
                    }
                }
            }
        } else {
            comp_pred_mode = 0; // PRED_SINGLEREF
        }

        // single_ref probs.
        if comp_pred_mode != 1 {
            // not PRED_COMPREF
            for i in 0..5 {
                if bd.get_prob(252) {
                    prob.single_ref[i][0] = update_prob(&mut bd, prob.single_ref[i][0]);
                }
                if bd.get_prob(252) {
                    prob.single_ref[i][1] = update_prob(&mut bd, prob.single_ref[i][1]);
                }
            }
        }

        // comp_ref probs.
        if comp_pred_mode != 0 {
            // not PRED_SINGLEREF
            for i in 0..5 {
                if bd.get_prob(252) {
                    prob.comp_ref[i] = update_prob(&mut bd, prob.comp_ref[i]);
                }
            }
        }

        // Y mode probs.
        for i in 0..4 {
            for j in 0..9 {
                if bd.get_prob(252) {
                    prob.y_mode[i][j] = update_prob(&mut bd, prob.y_mode[i][j]);
                }
            }
        }

        // Partition probs (note: index is 3-i, matching FFmpeg).
        // FFmpeg uses partition[bl][combined_ctx][3] where combined_ctx = above | (left<<1).
        // Rust ProbContext uses partition[bl][above_ctx][left_ctx][3].
        // We iterate over 4 combined contexts and map to [above_bit][left_bit].
        for i in 0..4usize {
            for j in 0..4usize {
                // j = above_bit | (left_bit << 1)
                let above_bit = j & 1;
                let left_bit = j >> 1;
                for k in 0..3usize {
                    if bd.get_prob(252) {
                        prob.partition[3 - i][above_bit][left_bit][k] =
                            update_prob(&mut bd, prob.partition[3 - i][above_bit][left_bit][k]);
                    }
                }
            }
        }

        // MV fields — use raw 7-bit reads (not update_prob subexp model).
        for i in 0..3 {
            if bd.get_prob(252) {
                prob.mv_joint[i] = (bd.get_uint(7) << 1) as u8 | 1;
            }
        }

        for i in 0..2 {
            if bd.get_prob(252) {
                prob.mv_comp[i].sign = (bd.get_uint(7) << 1) as u8 | 1;
            }
            for j in 0..10 {
                if bd.get_prob(252) {
                    prob.mv_comp[i].classes[j] = (bd.get_uint(7) << 1) as u8 | 1;
                }
            }
            if bd.get_prob(252) {
                prob.mv_comp[i].class0 = (bd.get_uint(7) << 1) as u8 | 1;
            }
            for j in 0..10 {
                if bd.get_prob(252) {
                    prob.mv_comp[i].bits[j] = (bd.get_uint(7) << 1) as u8 | 1;
                }
            }
        }

        for i in 0..2 {
            for j in 0..2 {
                for k in 0..3 {
                    if bd.get_prob(252) {
                        prob.mv_comp[i].class0_fp[j][k] = (bd.get_uint(7) << 1) as u8 | 1;
                    }
                }
            }
            for j in 0..3 {
                if bd.get_prob(252) {
                    prob.mv_comp[i].fp[j] = (bd.get_uint(7) << 1) as u8 | 1;
                }
            }
        }

        if high_precision_mvs {
            for i in 0..2 {
                if bd.get_prob(252) {
                    prob.mv_comp[i].class0_hp = (bd.get_uint(7) << 1) as u8 | 1;
                }
                if bd.get_prob(252) {
                    prob.mv_comp[i].hp = (bd.get_uint(7) << 1) as u8 | 1;
                }
            }
        }
    }

    // Populate PARETO8 extensions for coef (only coef[i][j][k][l][m][2] drives
    // the pareto lookup used during tile decoding — we expose raw [3] entries).
    // Suppress unused import warning for PARETO8 since it's referenced in
    // documentation and used in tile decoding (future agents).
    let _ = &PARETO8;

    let hdr = FrameHeader {
        profile,
        show_existing_frame: false,
        show_existing_frame_ref: 0,
        frame_type,
        show_frame,
        error_resilient,
        intra_only,
        reset_context,
        refresh_ref_mask,
        ref_idx,
        sign_bias,
        bit_depth,
        width,
        height,
        render_width,
        render_height,
        subsampling_x,
        subsampling_y,
        color_space,
        color_range,
        filter_level,
        sharpness_level,
        mode_ref_delta_enabled,
        mode_ref_delta_updated,
        ref_deltas,
        mode_deltas,
        base_q_idx,
        y_dc_delta_q,
        uv_dc_delta_q,
        uv_ac_delta_q,
        lossless,
        segmentation: seg,
        tile_cols_log2,
        tile_rows_log2,
        compressed_header_size,
        compressed_header_offset,
        tile_data_offset,
        prob,
        coef,
        tx_mode,
        high_precision_mvs,
        filter_mode,
        allow_comp_inter,
        comp_pred_mode,
        fix_comp_ref,
        var_comp_ref,
        refresh_ctx,
        parallel_mode,
        frame_ctx_id,
        use_last_frame_mvs: !error_resilient && show_frame,
        size_from_ref,
    };

    Ok((hdr, tile_data_offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_frame_marker() {
        // First 2 bits must be 0b10. Provide 0b00 instead.
        let data = [0x00u8; 32];
        let result = decode_frame_header(&data, &Default::default(), &[DEFAULT_COEF_PROBS; 4]);
        assert!(result.is_err(), "Expected error for bad frame marker");
    }

    #[test]
    fn test_empty_data() {
        let result = decode_frame_header(&[], &Default::default(), &[DEFAULT_COEF_PROBS; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tx_mode_values() {
        assert_eq!(TxMode::Only4x4 as u8, 0);
        assert_eq!(TxMode::TxModeSelect as u8, 4);
    }

    #[test]
    fn test_color_space_from() {
        assert_eq!(ColorSpace::from(7u8), ColorSpace::Rgb);
        assert_eq!(ColorSpace::from(0u8), ColorSpace::Unknown);
    }
}
