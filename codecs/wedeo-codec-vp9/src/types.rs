// VP9 type definitions.
//
// Translated from FFmpeg's libavcodec/vp9.h, vp9shared.h, and vp9dec.h.
// Enum discriminants match the integer values used in the bitstream and
// in FFmpeg's C enumerations so that table lookups remain identical.

/// VP9 bitstream profile.
///
/// Profile 0 = 8-bit 4:2:0
/// Profile 1 = 8-bit 4:2:2 / 4:4:4 / 4:4:0
/// Profile 2 = 10/12-bit 4:2:0
/// Profile 3 = 10/12-bit 4:2:2 / 4:4:4 / 4:4:0
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BitstreamProfile {
    #[default]
    Profile0 = 0,
    Profile1 = 1,
    Profile2 = 2,
    Profile3 = 3,
}

impl TryFrom<u8> for BitstreamProfile {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::Profile0),
            1 => Ok(Self::Profile1),
            2 => Ok(Self::Profile2),
            3 => Ok(Self::Profile3),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Block sizes  (enum BlockSize / BS_* in vp9shared.h)
// ---------------------------------------------------------------------------

/// VP9 superblock / block sizes.
///
/// Values mirror `enum BlockSize` (`BS_64x64` … `BS_4x4`, then `N_BS_SIZES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BlockSize {
    #[default]
    Bs64x64 = 0,
    Bs64x32 = 1,
    Bs32x64 = 2,
    Bs32x32 = 3,
    Bs32x16 = 4,
    Bs16x32 = 5,
    Bs16x16 = 6,
    Bs16x8 = 7,
    Bs8x16 = 8,
    Bs8x8 = 9,
    Bs8x4 = 10,
    Bs4x8 = 11,
    Bs4x4 = 12,
}

/// Total number of block sizes.
pub const N_BS_SIZES: usize = 13;

impl TryFrom<u8> for BlockSize {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::Bs64x64),
            1 => Ok(Self::Bs64x32),
            2 => Ok(Self::Bs32x64),
            3 => Ok(Self::Bs32x32),
            4 => Ok(Self::Bs32x16),
            5 => Ok(Self::Bs16x32),
            6 => Ok(Self::Bs16x16),
            7 => Ok(Self::Bs16x8),
            8 => Ok(Self::Bs8x16),
            9 => Ok(Self::Bs8x8),
            10 => Ok(Self::Bs8x4),
            11 => Ok(Self::Bs4x8),
            12 => Ok(Self::Bs4x4),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Block level  (enum BlockLevel / BL_* in vp9shared.h)
// ---------------------------------------------------------------------------

/// Superblock nesting level (64×64 → 8×8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BlockLevel {
    #[default]
    Bl64x64 = 0,
    Bl32x32 = 1,
    Bl16x16 = 2,
    Bl8x8 = 3,
}

impl TryFrom<u8> for BlockLevel {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::Bl64x64),
            1 => Ok(Self::Bl32x32),
            2 => Ok(Self::Bl16x16),
            3 => Ok(Self::Bl8x8),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Block partition  (enum BlockPartition / PARTITION_* in vp9shared.h)
// ---------------------------------------------------------------------------

/// How a superblock is partitioned into sub-blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BlockPartition {
    /// Not split — the entire block is one prediction unit.
    #[default]
    None = 0,
    /// Horizontal split into two halves.
    H = 1,
    /// Vertical split into two halves.
    V = 2,
    /// Four-way split.
    Split = 3,
}

impl TryFrom<u8> for BlockPartition {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::H),
            2 => Ok(Self::V),
            3 => Ok(Self::Split),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Transform size  (enum TxfmMode / TX_* in vp9.h)
// ---------------------------------------------------------------------------

/// Transform block size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum TxSize {
    #[default]
    Tx4x4 = 0,
    Tx8x8 = 1,
    Tx16x16 = 2,
    Tx32x32 = 3,
}

/// Number of non-switchable transform sizes.
pub const N_TX_SIZES: usize = 4;

/// Switchable transform mode sentinel.
pub const TX_SWITCHABLE: u8 = N_TX_SIZES as u8;

impl TryFrom<u8> for TxSize {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::Tx4x4),
            1 => Ok(Self::Tx8x8),
            2 => Ok(Self::Tx16x16),
            3 => Ok(Self::Tx32x32),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Transform type  (enum TxfmType / DCT_DCT … in vp9.h)
// ---------------------------------------------------------------------------

/// 2-D transform type (row × column).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TxType {
    #[default]
    DctDct = 0,
    DctAdst = 1,
    AdstDct = 2,
    AdstAdst = 3,
}

pub const N_TX_TYPES: usize = 4;

impl TryFrom<u8> for TxType {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::DctDct),
            1 => Ok(Self::DctAdst),
            2 => Ok(Self::AdstDct),
            3 => Ok(Self::AdstAdst),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Intra prediction mode  (enum IntraPredMode in vp9.h)
// ---------------------------------------------------------------------------

/// VP9 intra prediction modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum IntraMode {
    #[default]
    VertPred = 0,
    HorPred = 1,
    DcPred = 2,
    DiagDownLeftPred = 3,
    DiagDownRightPred = 4,
    VertRightPred = 5,
    HorDownPred = 6,
    VertLeftPred = 7,
    HorUpPred = 8,
    TmVp8Pred = 9,
    /// Left DC (used internally).
    LeftDcPred = 10,
    /// Top DC (used internally).
    TopDcPred = 11,
    /// 128 DC (used internally).
    Dc128Pred = 12,
    /// 127 DC (used internally).
    Dc127Pred = 13,
    /// 129 DC (used internally).
    Dc129Pred = 14,
}

pub const N_INTRA_PRED_MODES: usize = 15;

impl TryFrom<u8> for IntraMode {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::VertPred),
            1 => Ok(Self::HorPred),
            2 => Ok(Self::DcPred),
            3 => Ok(Self::DiagDownLeftPred),
            4 => Ok(Self::DiagDownRightPred),
            5 => Ok(Self::VertRightPred),
            6 => Ok(Self::HorDownPred),
            7 => Ok(Self::VertLeftPred),
            8 => Ok(Self::HorUpPred),
            9 => Ok(Self::TmVp8Pred),
            10 => Ok(Self::LeftDcPred),
            11 => Ok(Self::TopDcPred),
            12 => Ok(Self::Dc128Pred),
            13 => Ok(Self::Dc127Pred),
            14 => Ok(Self::Dc129Pred),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolation filter  (enum FilterMode / FILTER_* in vp9.h)
// ---------------------------------------------------------------------------

/// VP9 interpolation filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum InterpFilter {
    #[default]
    EightTapSmooth = 0,
    EightTapRegular = 1,
    EightTapSharp = 2,
    Bilinear = 3,
}

pub const N_FILTERS: usize = 4;
/// Switchable filter sentinel.
pub const FILTER_SWITCHABLE: u8 = N_FILTERS as u8;

impl TryFrom<u8> for InterpFilter {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::EightTapSmooth),
            1 => Ok(Self::EightTapRegular),
            2 => Ok(Self::EightTapSharp),
            3 => Ok(Self::Bilinear),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Reference frame  (used in VP9 inter coding)
// ---------------------------------------------------------------------------

/// VP9 reference frame indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(i8)]
pub enum ReferenceFrame {
    #[default]
    IntraFrame = 0,
    LastFrame = 1,
    GoldenFrame = 2,
    AltRefFrame = 3,
}

impl TryFrom<i8> for ReferenceFrame {
    type Error = i8;
    fn try_from(v: i8) -> Result<Self, i8> {
        match v {
            0 => Ok(Self::IntraFrame),
            1 => Ok(Self::LastFrame),
            2 => Ok(Self::GoldenFrame),
            3 => Ok(Self::AltRefFrame),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Inter prediction mode  (enum InterPredMode / NEARESTMV … in vp9shared.h)
// ---------------------------------------------------------------------------

/// VP9 inter prediction modes.
///
/// Note: values start at 10 to match FFmpeg's `enum InterPredMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum InterPredMode {
    NearestMv = 10,
    NearMv = 11,
    ZeroMv = 12,
    NewMv = 13,
}

impl TryFrom<u8> for InterPredMode {
    type Error = u8;
    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            10 => Ok(Self::NearestMv),
            11 => Ok(Self::NearMv),
            12 => Ok(Self::ZeroMv),
            13 => Ok(Self::NewMv),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// Compound prediction mode  (enum CompPredMode in vp9shared.h)
// ---------------------------------------------------------------------------

/// Whether a frame uses single-reference, compound, or switchable inter pred.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CompPredMode {
    #[default]
    SingleRef = 0,
    CompRef = 1,
    Switchable = 2,
}

impl TryFrom<u8> for CompPredMode {
    type Error = u8;
    fn try_from(v: u8) -> std::result::Result<Self, u8> {
        match v {
            0 => Ok(Self::SingleRef),
            1 => Ok(Self::CompRef),
            2 => Ok(Self::Switchable),
            _ => Err(v),
        }
    }
}

// ---------------------------------------------------------------------------
// MV joint  (enum MVJoint in vp9dec.h)
// ---------------------------------------------------------------------------

/// Motion vector joint type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MvJoint {
    #[default]
    Zero = 0,
    H = 1,
    V = 2,
    Hv = 3,
}

// ---------------------------------------------------------------------------
// Probability context  (ProbContext in vp9dec.h)
// ---------------------------------------------------------------------------

/// Full set of frame-level probabilities used by the VP9 entropy coder.
///
/// Mirrors `ProbContext` in vp9dec.h.
/// `partition[block_level][above_ctx][left_ctx]` — 4 block levels,
/// 4 above-context values, 4 left-context values, 3 probability bytes.
#[derive(Clone, Debug, Default)]
pub struct ProbContext {
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub filter: [[u8; 2]; 4],
    pub mv_mode: [[u8; 3]; 7],
    pub intra: [u8; 4],
    pub comp: [u8; 5],
    pub single_ref: [[u8; 2]; 5],
    pub comp_ref: [u8; 5],
    pub tx32p: [[u8; 3]; 2],
    pub tx16p: [[u8; 2]; 2],
    pub tx8p: [u8; 2],
    pub skip: [u8; 3],
    pub mv_joint: [u8; 3],
    pub mv_comp: [MvCompProbs; 2],
    pub partition: [[[[u8; 3]; 4]; 4]; 4],
}

/// Per-component motion vector probability set.
///
/// Mirrors the anonymous struct inside `ProbContext.mv_comp[]` in vp9dec.h.
#[derive(Clone, Copy, Debug, Default)]
pub struct MvCompProbs {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: u8,
    pub bits: [u8; 10],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
    pub class0_hp: u8,
    pub hp: u8,
}
