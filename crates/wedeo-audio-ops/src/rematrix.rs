//! Faithful pure-Rust port of FFmpeg `n9.1` `libswresample/rematrix.c`.
//!
//! Builds exact channel downmix, upmix, and reorder matrices across mono, stereo,
//! 2.1, 3.0, 4.0, quad, 5.0, 5.1, 6.0, 6.1, 7.0, 7.1, and custom layouts, with
//! optional row-sum normalization and Dolby/DPLII matrix encoding.

use std::f64::consts::FRAC_1_SQRT_2;

use wedeo_core::{Channel, ChannelLayout, ChannelOrder, Error, Result};

use crate::ops::ChannelMatrix;

const SQRT1_3: f64 = 0.577_350_269_189_625_8;
const SQRT2_3: f64 = 0.816_496_580_927_726;
const SQRT3_2: f64 = 1.224_744_871_391_589;
const NUM_NAMED_CHANNELS: usize = 41;

const CH_FL: usize = Channel::FrontLeft as usize;
const CH_FR: usize = Channel::FrontRight as usize;
const CH_FC: usize = Channel::FrontCenter as usize;
const CH_LFE: usize = Channel::LowFrequency as usize;
const CH_BL: usize = Channel::BackLeft as usize;
const CH_BR: usize = Channel::BackRight as usize;
const CH_FLC: usize = Channel::FrontLeftOfCenter as usize;
const CH_FRC: usize = Channel::FrontRightOfCenter as usize;
const CH_BC: usize = Channel::BackCenter as usize;
const CH_SL: usize = Channel::SideLeft as usize;
const CH_SR: usize = Channel::SideRight as usize;
const CH_TC: usize = Channel::TopCenter as usize;
const CH_TFL: usize = Channel::TopFrontLeft as usize;
const CH_TFC: usize = Channel::TopFrontCenter as usize;
const CH_TFR: usize = Channel::TopFrontRight as usize;
const CH_TBL: usize = Channel::TopBackLeft as usize;
const CH_TBC: usize = Channel::TopBackCenter as usize;
const CH_TBR: usize = Channel::TopBackRight as usize;
const CH_LFE2: usize = Channel::LowFrequency2 as usize;
const CH_TSL: usize = Channel::TopSideLeft as usize;
const CH_TSR: usize = Channel::TopSideRight as usize;
const CH_BFC: usize = Channel::BottomFrontCenter as usize;
const CH_BFL: usize = Channel::BottomFrontLeft as usize;
const CH_BFR: usize = Channel::BottomFrontRight as usize;

const MASK_FL: u64 = 1 << CH_FL;
const MASK_FR: u64 = 1 << CH_FR;
const MASK_FC: u64 = 1 << CH_FC;
const MASK_LFE: u64 = 1 << CH_LFE;
const MASK_BL: u64 = 1 << CH_BL;
const MASK_BR: u64 = 1 << CH_BR;
const MASK_FLC: u64 = 1 << CH_FLC;
const MASK_FRC: u64 = 1 << CH_FRC;
const MASK_BC: u64 = 1 << CH_BC;
const MASK_SL: u64 = 1 << CH_SL;
const MASK_SR: u64 = 1 << CH_SR;
const MASK_TC: u64 = 1 << CH_TC;
const MASK_TFL: u64 = 1 << CH_TFL;
const MASK_TFC: u64 = 1 << CH_TFC;
const MASK_TFR: u64 = 1 << CH_TFR;
const MASK_TBL: u64 = 1 << CH_TBL;
const MASK_TBC: u64 = 1 << CH_TBC;
const MASK_TBR: u64 = 1 << CH_TBR;
const MASK_SL_DOWNMIX: u64 = 1 << (Channel::StereoLeft as usize);
const MASK_SR_DOWNMIX: u64 = 1 << (Channel::StereoRight as usize);
const MASK_LFE2: u64 = 1 << CH_LFE2;
const MASK_TSL: u64 = 1 << CH_TSL;
const MASK_TSR: u64 = 1 << CH_TSR;
const MASK_BFC: u64 = 1 << CH_BFC;
const MASK_BFL: u64 = 1 << CH_BFL;
const MASK_BFR: u64 = 1 << CH_BFR;

const MASK_STEREO: u64 = MASK_FL | MASK_FR;
const MASK_SURROUND: u64 = MASK_FL | MASK_FR | MASK_FC;
const MASK_STEREO_DOWNMIX: u64 = MASK_SL_DOWNMIX | MASK_SR_DOWNMIX;

/// Stereo surround matrix encoding mode (matching FFmpeg `AVMatrixEncoding`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum MatrixEncoding {
    #[default]
    None,
    Dolby,
    DplII,
}

/// Configuration for automatic channel matrix construction (`libswresample/rematrix.c`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RematrixOptions {
    /// Center channel mix level when downmixing to stereo (default: `SQRT1_2`).
    pub center_mix_level: f64,
    /// Surround channel mix level when downmixing to stereo/mono (default: `SQRT1_2`).
    pub surround_mix_level: f64,
    /// LFE channel mix level when downmixing to front/center (default: `0.0`).
    pub lfe_mix_level: f64,
    /// Optional row-sum upper bound for clipping-safe normalization (e.g. `Some(1.0)`).
    /// When `None`, coefficients are unnormalized unless `rematrix_volume < 0.0`.
    pub maxval: Option<f64>,
    /// Linear output volume multiplier (default: `1.0`). Negative values match FFmpeg's
    /// explicit target divisor `-rematrix_volume`.
    pub rematrix_volume: f64,
    /// Surround matrix encoding mode (default: `MatrixEncoding::None`).
    pub matrix_encoding: MatrixEncoding,
}

impl Default for RematrixOptions {
    fn default() -> Self {
        Self {
            center_mix_level: FRAC_1_SQRT_2,
            surround_mix_level: FRAC_1_SQRT_2,
            lfe_mix_level: 0.0,
            maxval: None,
            rematrix_volume: 1.0,
            matrix_encoding: MatrixEncoding::None,
        }
    }
}

impl RematrixOptions {
    /// Default options with row-sum normalization capped at `1.0` to prevent clipping.
    pub fn normalized() -> Self {
        Self {
            maxval: Some(1.0),
            ..Self::default()
        }
    }
}

#[inline]
fn is_even_pair(mask: u64) -> bool {
    mask == 0 || (mask & (mask - 1)) != 0
}

fn layout_mask(layout: &ChannelLayout) -> u64 {
    let mut mask = 0u64;
    for &ch in &layout.channels {
        let bit = ch as u8;
        if bit < 64 {
            mask |= 1u64 << bit;
        }
    }
    mask
}

fn clean_layout(layout: &ChannelLayout) -> ChannelLayout {
    if layout.nb_channels == 1
        && layout.channels.len() == 1
        && layout.channels[0] != Channel::FrontCenter
    {
        return ChannelLayout::mono();
    }
    layout.clone()
}

fn clean_downmix(layout: &mut ChannelLayout, other: &ChannelLayout) {
    if layout_mask(layout) != MASK_STEREO_DOWNMIX || (layout_mask(other) & MASK_STEREO_DOWNMIX) != 0
    {
        return;
    }
    for ch in &mut layout.channels {
        if *ch == Channel::StereoLeft {
            *ch = Channel::FrontLeft;
        } else if *ch == Channel::StereoRight {
            *ch = Channel::FrontRight;
        }
    }
}

fn check_sane_layout(layout: &ChannelLayout, mixed: bool) -> Result<()> {
    if layout.nb_channels == 0 {
        return Err(Error::invalid_argument(
            "rematrix",
            "channel layout must have at least 1 channel",
        ));
    }
    if layout.order == ChannelOrder::Unspec {
        return Err(Error::unsupported(
            "rematrix",
            "cannot rematrix unspecified channel layout",
        ));
    }
    if layout.channels.len() != layout.nb_channels as usize {
        return Err(Error::invalid_argument(
            "rematrix",
            "channel layout nb_channels does not match channels vector length",
        ));
    }
    if !mixed {
        return Ok(());
    }
    let mask = layout_mask(layout);
    if (mask & MASK_SURROUND) == 0 {
        return Err(Error::unsupported(
            "rematrix",
            "channel layout has no front speaker for downmix/upmix",
        ));
    }
    for pair in [
        MASK_FL | MASK_FR,
        MASK_SL | MASK_SR,
        MASK_BL | MASK_BR,
        MASK_FLC | MASK_FRC,
        MASK_TFL | MASK_TFR,
        MASK_TBL | MASK_TBR,
        MASK_TSL | MASK_TSR,
        MASK_BFL | MASK_BFR,
    ] {
        if !is_even_pair(mask & pair) {
            return Err(Error::unsupported(
                "rematrix",
                "asymmetric left/right channel pair is not supported",
            ));
        }
    }
    Ok(())
}

/// Build an FFmpeg `n9.1` `libswresample/rematrix.c` channel mixing matrix
/// converting from `in_layout` to `out_layout`.
pub fn build_rematrix_matrix(
    in_layout: &ChannelLayout,
    out_layout: &ChannelLayout,
    opts: &RematrixOptions,
) -> Result<ChannelMatrix> {
    if in_layout.nb_channels == 0 || out_layout.nb_channels == 0 {
        return Err(Error::invalid_argument(
            "rematrix",
            "channel count must be non-zero",
        ));
    }

    // Identical or both-unspecified matching channel counts produce a pure identity/copy matrix
    // unless a non-unity rematrix volume was requested.
    if (in_layout == out_layout
        || (in_layout.order == ChannelOrder::Unspec
            && out_layout.order == ChannelOrder::Unspec
            && in_layout.nb_channels == out_layout.nb_channels))
        && opts.rematrix_volume == 1.0
    {
        return Ok(ChannelMatrix::identity(in_layout.nb_channels as usize));
    }

    if (in_layout.order == ChannelOrder::Unspec || out_layout.order == ChannelOrder::Unspec)
        && in_layout.nb_channels == out_layout.nb_channels
        && in_layout.channels.is_empty()
        && out_layout.channels.is_empty()
    {
        return Ok(ChannelMatrix::gain(
            in_layout.nb_channels as usize,
            opts.rematrix_volume,
        ));
    }

    let mut in_ch = clean_layout(in_layout);
    let mut out_ch = clean_layout(out_layout);
    clean_downmix(&mut out_ch, &in_ch);
    clean_downmix(&mut in_ch, &out_ch);

    let in_mask = layout_mask(&in_ch);
    let out_mask = layout_mask(&out_ch);
    let mixed = in_mask != out_mask;

    check_sane_layout(&in_ch, mixed)?;
    check_sane_layout(&out_ch, mixed)?;

    let mut matrix = [[0.0_f64; NUM_NAMED_CHANNELS]; NUM_NAMED_CHANNELS];
    for (i, row) in matrix.iter_mut().enumerate() {
        if (in_mask & out_mask & (1u64 << i)) != 0 {
            row[i] = 1.0;
        }
    }

    let unaccounted = in_mask & !out_mask;
    let clev = opts.center_mix_level;
    let slev = opts.surround_mix_level;
    let lfe_lev = opts.lfe_mix_level;

    if (unaccounted & MASK_FC) != 0 {
        if (out_mask & MASK_STEREO) == MASK_STEREO {
            if (in_mask & MASK_STEREO) != 0 {
                matrix[CH_FL][CH_FC] += clev;
                matrix[CH_FR][CH_FC] += clev;
            } else {
                matrix[CH_FL][CH_FC] += FRAC_1_SQRT_2;
                matrix[CH_FR][CH_FC] += FRAC_1_SQRT_2;
            }
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map front center to output layout",
            ));
        }
    }

    if (unaccounted & MASK_STEREO) != 0 {
        if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_FL] += FRAC_1_SQRT_2;
            matrix[CH_FC][CH_FR] += FRAC_1_SQRT_2;
            if (in_mask & MASK_FC) != 0 {
                matrix[CH_FC][CH_FC] = clev * std::f64::consts::SQRT_2;
            }
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map stereo channels to output layout",
            ));
        }
    }

    if (unaccounted & MASK_BC) != 0 {
        if (out_mask & MASK_BL) != 0 {
            matrix[CH_BL][CH_BC] += FRAC_1_SQRT_2;
            matrix[CH_BR][CH_BC] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_SL) != 0 {
            matrix[CH_SL][CH_BC] += FRAC_1_SQRT_2;
            matrix[CH_SR][CH_BC] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_FL) != 0 {
            match opts.matrix_encoding {
                MatrixEncoding::Dolby | MatrixEncoding::DplII => {
                    if (unaccounted & (MASK_BL | MASK_SL)) != 0 {
                        matrix[CH_FL][CH_BC] -= slev * FRAC_1_SQRT_2;
                        matrix[CH_FR][CH_BC] += slev * FRAC_1_SQRT_2;
                    } else {
                        matrix[CH_FL][CH_BC] -= slev;
                        matrix[CH_FR][CH_BC] += slev;
                    }
                }
                MatrixEncoding::None => {
                    matrix[CH_FL][CH_BC] += slev * FRAC_1_SQRT_2;
                    matrix[CH_FR][CH_BC] += slev * FRAC_1_SQRT_2;
                }
            }
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_BC] += slev * FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map back center to output layout",
            ));
        }
    }

    if (unaccounted & MASK_BL) != 0 {
        if (out_mask & MASK_BC) != 0 {
            matrix[CH_BC][CH_BL] += FRAC_1_SQRT_2;
            matrix[CH_BC][CH_BR] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_SL) != 0 {
            if (in_mask & MASK_SL) != 0 {
                matrix[CH_SL][CH_BL] += FRAC_1_SQRT_2;
                matrix[CH_SR][CH_BR] += FRAC_1_SQRT_2;
            } else {
                matrix[CH_SL][CH_BL] += 1.0;
                matrix[CH_SR][CH_BR] += 1.0;
            }
        } else if (out_mask & MASK_FL) != 0 {
            match opts.matrix_encoding {
                MatrixEncoding::Dolby => {
                    matrix[CH_FL][CH_BL] -= slev * FRAC_1_SQRT_2;
                    matrix[CH_FL][CH_BR] -= slev * FRAC_1_SQRT_2;
                    matrix[CH_FR][CH_BL] += slev * FRAC_1_SQRT_2;
                    matrix[CH_FR][CH_BR] += slev * FRAC_1_SQRT_2;
                }
                MatrixEncoding::DplII => {
                    matrix[CH_FL][CH_BL] -= slev * SQRT3_2;
                    matrix[CH_FL][CH_BR] -= slev * FRAC_1_SQRT_2;
                    matrix[CH_FR][CH_BL] += slev * FRAC_1_SQRT_2;
                    matrix[CH_FR][CH_BR] += slev * SQRT3_2;
                }
                MatrixEncoding::None => {
                    matrix[CH_FL][CH_BL] += slev;
                    matrix[CH_FR][CH_BR] += slev;
                }
            }
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_BL] += slev * FRAC_1_SQRT_2;
            matrix[CH_FC][CH_BR] += slev * FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map back left/right to output layout",
            ));
        }
    }

    if (unaccounted & MASK_SL) != 0 {
        if (out_mask & MASK_BL) != 0 {
            if (in_mask & MASK_BL) != 0 {
                matrix[CH_BL][CH_SL] += FRAC_1_SQRT_2;
                matrix[CH_BR][CH_SR] += FRAC_1_SQRT_2;
            } else {
                matrix[CH_BL][CH_SL] += 1.0;
                matrix[CH_BR][CH_SR] += 1.0;
            }
        } else if (out_mask & MASK_BC) != 0 {
            matrix[CH_BC][CH_SL] += FRAC_1_SQRT_2;
            matrix[CH_BC][CH_SR] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_FL) != 0 {
            match opts.matrix_encoding {
                MatrixEncoding::Dolby => {
                    matrix[CH_FL][CH_SL] -= slev * FRAC_1_SQRT_2;
                    matrix[CH_FL][CH_SR] -= slev * FRAC_1_SQRT_2;
                    matrix[CH_FR][CH_SL] += slev * FRAC_1_SQRT_2;
                    matrix[CH_FR][CH_SR] += slev * FRAC_1_SQRT_2;
                }
                MatrixEncoding::DplII => {
                    matrix[CH_FL][CH_SL] -= slev * SQRT3_2;
                    matrix[CH_FL][CH_SR] -= slev * FRAC_1_SQRT_2;
                    matrix[CH_FR][CH_SL] += slev * FRAC_1_SQRT_2;
                    matrix[CH_FR][CH_SR] += slev * SQRT3_2;
                }
                MatrixEncoding::None => {
                    matrix[CH_FL][CH_SL] += slev;
                    matrix[CH_FR][CH_SR] += slev;
                }
            }
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_SL] += slev * FRAC_1_SQRT_2;
            matrix[CH_FC][CH_SR] += slev * FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map side left/right to output layout",
            ));
        }
    }

    if (unaccounted & MASK_FLC) != 0 {
        if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_FLC] += 1.0;
            matrix[CH_FR][CH_FRC] += 1.0;
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_FLC] += FRAC_1_SQRT_2;
            matrix[CH_FC][CH_FRC] += FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map front left/right of center to output layout",
            ));
        }
    }

    if (unaccounted & MASK_TFL) != 0 {
        if (out_mask & MASK_TFC) != 0 {
            matrix[CH_TFC][CH_TFL] += FRAC_1_SQRT_2;
            matrix[CH_TFC][CH_TFR] += FRAC_1_SQRT_2;
            if (in_mask & MASK_TFC) != 0 {
                matrix[CH_TFC][CH_TFC] = clev * std::f64::consts::SQRT_2;
            }
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_TFL] += 1.0;
            matrix[CH_FR][CH_TFR] += 1.0;
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_TFL] += FRAC_1_SQRT_2;
            matrix[CH_FC][CH_TFR] += FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map top front left/right to output layout",
            ));
        }
    }

    if (unaccounted & MASK_TFC) != 0 {
        if (out_mask & MASK_TFL) != 0 {
            matrix[CH_TFL][CH_TFC] += FRAC_1_SQRT_2;
            matrix[CH_TFR][CH_TFC] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_TFC] += 1.0;
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_TFC] += clev;
            matrix[CH_FR][CH_TFC] += clev;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map top front center to output layout",
            ));
        }
    }

    if (unaccounted & MASK_TBL) != 0 {
        if (out_mask & MASK_TBC) != 0 {
            matrix[CH_TBC][CH_TBL] += FRAC_1_SQRT_2;
            matrix[CH_TBC][CH_TBR] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_TFL) != 0 {
            matrix[CH_TFL][CH_TBL] += FRAC_1_SQRT_2;
            matrix[CH_TFR][CH_TBR] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_BL) != 0 {
            matrix[CH_BL][CH_TBL] += 1.0;
            matrix[CH_BR][CH_TBR] += 1.0;
        } else if (out_mask & MASK_SL) != 0 {
            matrix[CH_SL][CH_TBL] += 1.0;
            matrix[CH_SR][CH_TBR] += 1.0;
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_TBL] += slev;
            matrix[CH_FR][CH_TBR] += slev;
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_TBL] += slev * FRAC_1_SQRT_2;
            matrix[CH_FC][CH_TBR] += slev * FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map top back left/right to output layout",
            ));
        }
    }

    if (unaccounted & MASK_TBC) != 0 {
        if (out_mask & MASK_TBL) != 0 {
            matrix[CH_TBL][CH_TBC] += FRAC_1_SQRT_2;
            matrix[CH_TBR][CH_TBC] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_BL) != 0 {
            matrix[CH_BL][CH_TBC] += FRAC_1_SQRT_2;
            matrix[CH_BR][CH_TBC] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_SL) != 0 {
            matrix[CH_SL][CH_TBC] += FRAC_1_SQRT_2;
            matrix[CH_SR][CH_TBC] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_TBC] += 0.5;
            matrix[CH_FR][CH_TBC] += 0.5;
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_TBC] += 0.5;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map top back center to output layout",
            ));
        }
    }

    if (unaccounted & MASK_TSL) != 0 {
        if (out_mask & (MASK_TFL | MASK_TBC)) == (MASK_TFL | MASK_TBC) {
            matrix[CH_TFL][CH_TSL] += SQRT2_3;
            matrix[CH_TFR][CH_TSR] += SQRT2_3;
            matrix[CH_TBC][CH_TSL] += SQRT1_3;
            matrix[CH_TBC][CH_TSR] += SQRT1_3;
        } else if (out_mask & (MASK_TFL | MASK_TBL)) == (MASK_TFL | MASK_TBL) {
            matrix[CH_TFL][CH_TSL] += FRAC_1_SQRT_2;
            matrix[CH_TFR][CH_TSR] += FRAC_1_SQRT_2;
            matrix[CH_TBL][CH_TSL] += FRAC_1_SQRT_2;
            matrix[CH_TBR][CH_TSR] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_TFL) != 0 && (out_mask & (MASK_BL | MASK_SL)) != 0 {
            matrix[CH_TFL][CH_TSL] += FRAC_1_SQRT_2;
            matrix[CH_TFR][CH_TSR] += FRAC_1_SQRT_2;
            if (out_mask & MASK_BL) != 0 {
                matrix[CH_BL][CH_TSL] += FRAC_1_SQRT_2;
                matrix[CH_BR][CH_TSR] += FRAC_1_SQRT_2;
            } else if (out_mask & MASK_SL) != 0 {
                matrix[CH_SL][CH_TSL] += FRAC_1_SQRT_2;
                matrix[CH_SR][CH_TSR] += FRAC_1_SQRT_2;
            }
        } else if (out_mask & MASK_SL) != 0 {
            matrix[CH_SL][CH_TSL] += 1.0;
            matrix[CH_SR][CH_TSR] += 1.0;
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_TSL] += slev;
            matrix[CH_FR][CH_TSR] += slev;
            if (out_mask & MASK_BL) != 0 {
                matrix[CH_BL][CH_TSL] += FRAC_1_SQRT_2;
                matrix[CH_BR][CH_TSR] += FRAC_1_SQRT_2;
            }
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_TSL] += slev * FRAC_1_SQRT_2;
            matrix[CH_FC][CH_TSR] += slev * FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map top side left/right to output layout",
            ));
        }
    }

    if (unaccounted & MASK_TC) != 0 {
        if (out_mask & (MASK_TFL | MASK_TBL)) == (MASK_TFL | MASK_TBL) {
            matrix[CH_TFL][CH_TC] += 0.5;
            matrix[CH_TFR][CH_TC] += 0.5;
            matrix[CH_TBL][CH_TC] += 0.5;
            matrix[CH_TBR][CH_TC] += 0.5;
        } else if (out_mask & (MASK_TFL | MASK_TBC)) == (MASK_TFL | MASK_TBC) {
            matrix[CH_TFL][CH_TC] += SQRT1_3;
            matrix[CH_TFR][CH_TC] += SQRT1_3;
            matrix[CH_TBC][CH_TC] += SQRT1_3;
        } else if (out_mask & MASK_TFL) != 0 && (out_mask & (MASK_BL | MASK_SL)) != 0 {
            matrix[CH_TFL][CH_TC] += 0.5;
            matrix[CH_TFR][CH_TC] += 0.5;
            if (out_mask & MASK_BL) != 0 {
                matrix[CH_BL][CH_TC] += 0.5;
                matrix[CH_BR][CH_TC] += 0.5;
            } else if (out_mask & MASK_SL) != 0 {
                matrix[CH_SL][CH_TC] += 0.5;
                matrix[CH_SR][CH_TC] += 0.5;
            }
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_TC] += 0.5;
            matrix[CH_FR][CH_TC] += 0.5;
            if (out_mask & MASK_BL) != 0 {
                matrix[CH_BL][CH_TC] += 0.5;
                matrix[CH_BR][CH_TC] += 0.5;
            } else if (out_mask & MASK_SL) != 0 {
                matrix[CH_SL][CH_TC] += 0.5;
                matrix[CH_SR][CH_TC] += 0.5;
            }
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_TC] += 0.5;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map top center to output layout",
            ));
        }
    }

    if (unaccounted & MASK_BFC) != 0 {
        if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_BFC] += 1.0;
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_BFC] += clev;
            matrix[CH_FR][CH_BFC] += clev;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map bottom front center to output layout",
            ));
        }
    }

    if (unaccounted & MASK_BFL) != 0 {
        if (out_mask & MASK_BFC) != 0 {
            matrix[CH_BFC][CH_BFL] += FRAC_1_SQRT_2;
            matrix[CH_BFC][CH_BFR] += FRAC_1_SQRT_2;
            if (in_mask & MASK_BFC) != 0 {
                matrix[CH_BFC][CH_BFC] = clev * std::f64::consts::SQRT_2;
            }
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_BFL] += 1.0;
            matrix[CH_FR][CH_BFR] += 1.0;
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_BFL] += FRAC_1_SQRT_2;
            matrix[CH_FC][CH_BFR] += FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map bottom front left/right to output layout",
            ));
        }
    }

    if (unaccounted & MASK_LFE) != 0 {
        if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_LFE] += lfe_lev;
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_LFE] += lfe_lev * FRAC_1_SQRT_2;
            matrix[CH_FR][CH_LFE] += lfe_lev * FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map LFE to output layout",
            ));
        }
    }

    if (unaccounted & MASK_LFE2) != 0 {
        if (out_mask & MASK_LFE) != 0 {
            matrix[CH_LFE][CH_LFE2] += FRAC_1_SQRT_2;
        } else if (out_mask & MASK_FC) != 0 {
            matrix[CH_FC][CH_LFE2] += lfe_lev;
        } else if (out_mask & MASK_FL) != 0 {
            matrix[CH_FL][CH_LFE2] += lfe_lev * FRAC_1_SQRT_2;
            matrix[CH_FR][CH_LFE2] += lfe_lev * FRAC_1_SQRT_2;
        } else {
            return Err(Error::unsupported(
                "rematrix",
                "cannot map LFE2 to output layout",
            ));
        }
    }

    let nb_in = in_ch.nb_channels as usize;
    let nb_out = out_ch.nb_channels as usize;
    let mut coeffs = vec![0.0_f64; nb_out * nb_in];
    let mut maxcoef = 0.0_f64;

    for (out_i, &out_chan) in out_ch.channels.iter().enumerate() {
        let mut sum = 0.0_f64;
        let i = out_chan as usize;
        for (in_i, &in_chan) in in_ch.channels.iter().enumerate() {
            let j = in_chan as usize;
            let val = if i < NUM_NAMED_CHANNELS && j < NUM_NAMED_CHANNELS {
                matrix[i][j]
            } else if i == j && ((in_mask & out_mask & (1u64 << i)) != 0) {
                1.0
            } else {
                0.0
            };
            coeffs[out_i * nb_in + in_i] = val;
            sum += val.abs();
        }
        if sum > maxcoef {
            maxcoef = sum;
        }
    }

    if opts.rematrix_volume < 0.0 {
        maxcoef = -opts.rematrix_volume;
    }

    if let Some(maxval) = opts.maxval {
        if (maxcoef > maxval || opts.rematrix_volume < 0.0) && maxcoef > 0.0 {
            let divisor = maxcoef / maxval;
            for c in &mut coeffs {
                *c /= divisor;
            }
        }
    } else if opts.rematrix_volume < 0.0 && maxcoef > 0.0 {
        for c in &mut coeffs {
            *c /= maxcoef;
        }
    }

    if opts.rematrix_volume > 0.0 && opts.rematrix_volume != 1.0 {
        for c in &mut coeffs {
            *c *= opts.rematrix_volume;
        }
    }

    ChannelMatrix::new(nb_in, nb_out, coeffs)
}
