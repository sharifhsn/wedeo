// H.264 slice header parsing.
//
// Reference: ITU-T H.264 Section 7.3.3, FFmpeg libavcodec/h264_slice.c
// (`h264_slice_header_parse`), h264_parse.c (`ff_h264_parse_ref_count`),
// h264_refs.c (`ff_h264_decode_ref_pic_list_reordering`),
// h264_refs.c (`ff_h264_decode_ref_pic_marking`).

use wedeo_codec::bitstream::{BitRead, BitReadBE, get_se_golomb, get_ue_golomb};
use wedeo_core::error::{Error, Result};

use crate::nal::NalUnitType;
use crate::pps::Pps;
use crate::sps::Sps;

// ---------------------------------------------------------------------------
// SliceType
// ---------------------------------------------------------------------------

/// H.264 slice type.
///
/// Raw values 0-4 from the bitstream. Values 5-9 are mapped to 0-4
/// (they indicate "all slices in the picture are of this type").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    P = 0,
    B = 1,
    I = 2,
    SP = 3,
    SI = 4,
}

impl SliceType {
    /// Parse a raw slice_type value (0-9) into a `SliceType`.
    ///
    /// Values 5-9 map to 0-4 respectively (they only differ in that they
    /// signal all slices in the picture share the same type).
    pub fn from_raw(raw: u32) -> Result<Self> {
        match raw % 5 {
            0 => Ok(SliceType::P),
            1 => Ok(SliceType::B),
            2 => Ok(SliceType::I),
            3 => Ok(SliceType::SP),
            4 => Ok(SliceType::SI),
            _ => Err(Error::InvalidData),
        }
    }

    /// Returns true if this is an I or SI slice (no inter prediction).
    pub fn is_intra(self) -> bool {
        matches!(self, SliceType::I | SliceType::SI)
    }

    /// Returns true if this is a B slice.
    pub fn is_b(self) -> bool {
        self == SliceType::B
    }

    /// Returns true if this is a P or SP slice.
    pub fn is_p(self) -> bool {
        matches!(self, SliceType::P | SliceType::SP)
    }
}

// ---------------------------------------------------------------------------
// Reference picture list modification
// ---------------------------------------------------------------------------

/// A single ref pic list modification entry (modification_of_pic_nums_idc, value).
///
/// idc 0: abs_diff_pic_num_minus1 (short-term, decrement)
/// idc 1: abs_diff_pic_num_minus1 (short-term, increment)
/// idc 2: long_term_pic_num
/// idc 3: end of loop
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefPicListModification {
    pub idc: u32,
    pub val: u32,
}

// ---------------------------------------------------------------------------
// Decoded reference picture marking (MMCO)
// ---------------------------------------------------------------------------

/// Memory management control operation (MMCO), from dec_ref_pic_marking().
///
/// Reference: ITU-T H.264 Table 7-9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmcoOp {
    /// End of MMCO operations (mmco == 0).
    End,
    /// Mark a short-term picture as "unused for reference" (mmco == 1).
    ShortTermUnused { difference_of_pic_nums_minus1: u32 },
    /// Mark a long-term picture as "unused for reference" (mmco == 2).
    LongTermUnused { long_term_pic_num: u32 },
    /// Assign a long-term frame index to a short-term picture (mmco == 3).
    ShortTermToLongTerm {
        difference_of_pic_nums_minus1: u32,
        long_term_frame_idx: u32,
    },
    /// Set max long-term frame index (mmco == 4).
    MaxLongTermFrameIdx { max_long_term_frame_idx_plus1: u32 },
    /// Mark all reference pictures as "unused for reference" (mmco == 5).
    Reset,
    /// Mark the current picture as long-term (mmco == 6).
    CurrentToLongTerm { long_term_frame_idx: u32 },
}

// ---------------------------------------------------------------------------
// SliceHeader
// ---------------------------------------------------------------------------

/// Parsed H.264 slice header.
///
/// Contains all fields from the slice_header() syntax element, plus
/// computed values derived from PPS/SPS for convenience.
#[derive(Debug, Clone)]
pub struct SliceHeader {
    // --- Core fields ---
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    /// True if the raw slice_type was >= 5 (all slices in picture share type).
    pub slice_type_fixed: bool,
    pub pps_id: u32,
    pub frame_num: u32,

    // --- Field coding ---
    pub field_pic_flag: bool,
    pub bottom_field_flag: bool,

    // --- IDR ---
    pub idr_pic_id: u32,

    // --- POC type 0 ---
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,

    // --- POC type 1 ---
    pub delta_pic_order_cnt: [i32; 2],

    // --- Redundant picture ---
    pub redundant_pic_cnt: u32,

    // --- Inter prediction ---
    pub direct_spatial_mv_pred_flag: bool,
    pub num_ref_idx_l0_active: u32,
    pub num_ref_idx_l1_active: u32,

    // --- Ref pic list modification ---
    pub ref_pic_list_modification_l0: Vec<RefPicListModification>,
    pub ref_pic_list_modification_l1: Vec<RefPicListModification>,

    // --- Dec ref pic marking (IDR) ---
    pub no_output_of_prior_pics: bool,
    pub long_term_reference_flag: bool,

    // --- Dec ref pic marking (non-IDR) ---
    pub adaptive_ref_pic_marking: bool,
    pub mmco_ops: Vec<MmcoOp>,

    // --- Entropy coding ---
    pub cabac_init_idc: u32,

    // --- Quantization ---
    pub slice_qp_delta: i32,
    /// Computed: pps.pic_init_qp + slice_qp_delta
    pub slice_qp: i32,

    // --- SP/SI ---
    pub sp_for_switch_flag: bool,
    pub slice_qs_delta: i32,

    // --- Weighted prediction ---
    pub luma_log2_weight_denom: u32,
    pub chroma_log2_weight_denom: u32,
    /// Per-ref luma weight/offset: [ref_idx] = (weight, offset). List 0.
    pub luma_weight_l0: Vec<(i32, i32)>,
    /// Per-ref chroma weight/offset: [ref_idx][plane] = (weight, offset). List 0.
    pub chroma_weight_l0: Vec<[(i32, i32); 2]>,
    /// Per-ref luma weight/offset for list 1 (B-slices).
    pub luma_weight_l1: Vec<(i32, i32)>,
    /// Per-ref chroma weight/offset for list 1.
    pub chroma_weight_l1: Vec<[(i32, i32); 2]>,
    /// True if any non-default luma weight is present.
    pub use_weight: bool,
    /// True if any non-default chroma weight is present.
    pub use_weight_chroma: bool,

    // --- Deblocking ---
    pub disable_deblocking_filter_idc: u32,
    pub slice_alpha_c0_offset: i32,
    pub slice_beta_offset: i32,

    // --- Derived ---
    /// Number of bits consumed by the slice header in the RBSP.
    /// The macroblock data starts at this bit offset.
    pub header_bits: usize,
}

impl Default for SliceHeader {
    fn default() -> Self {
        Self {
            first_mb_in_slice: 0,
            slice_type: SliceType::I,
            slice_type_fixed: false,
            pps_id: 0,
            frame_num: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            idr_pic_id: 0,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt_bottom: 0,
            delta_pic_order_cnt: [0; 2],
            redundant_pic_cnt: 0,
            direct_spatial_mv_pred_flag: false,
            num_ref_idx_l0_active: 0,
            num_ref_idx_l1_active: 0,
            ref_pic_list_modification_l0: Vec::new(),
            ref_pic_list_modification_l1: Vec::new(),
            no_output_of_prior_pics: false,
            long_term_reference_flag: false,
            adaptive_ref_pic_marking: false,
            mmco_ops: Vec::new(),
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            slice_qp: 0,
            sp_for_switch_flag: false,
            slice_qs_delta: 0,
            luma_log2_weight_denom: 0,
            chroma_log2_weight_denom: 0,
            luma_weight_l0: Vec::new(),
            chroma_weight_l0: Vec::new(),
            luma_weight_l1: Vec::new(),
            chroma_weight_l1: Vec::new(),
            use_weight: false,
            use_weight_chroma: false,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset: 0,
            slice_beta_offset: 0,
            header_bits: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Ref pic list modification parsing
// ---------------------------------------------------------------------------

/// Parse ref_pic_list_modification() for one list.
///
/// Reference: ITU-T H.264 Section 7.3.3.1
fn parse_ref_pic_list_modification(br: &mut BitReadBE<'_>) -> Result<Vec<RefPicListModification>> {
    let mut mods = Vec::new();

    loop {
        let idc = get_ue_golomb(br)?;
        if idc == 3 {
            break;
        }
        if idc > 5 {
            return Err(Error::InvalidData);
        }

        let val = match idc {
            0 | 1 => get_ue_golomb(br)?, // abs_diff_pic_num_minus1
            2 => get_ue_golomb(br)?,     // long_term_pic_num
            4 | 5 => get_ue_golomb(br)?, // abs_diff_view_idx_minus1 (MVC)
            _ => return Err(Error::InvalidData),
        };

        mods.push(RefPicListModification { idc, val });

        // Safety limit: prevent infinite loops on malformed streams
        if mods.len() > 128 {
            return Err(Error::InvalidData);
        }
    }

    Ok(mods)
}

// ---------------------------------------------------------------------------
// Dec ref pic marking parsing
// ---------------------------------------------------------------------------

/// Parse dec_ref_pic_marking() for IDR and non-IDR slices.
///
/// Reference: ITU-T H.264 Section 7.3.3.3
fn parse_dec_ref_pic_marking(
    br: &mut BitReadBE<'_>,
    is_idr: bool,
) -> Result<(bool, bool, bool, Vec<MmcoOp>)> {
    let mut no_output_of_prior_pics = false;
    let mut long_term_reference_flag = false;
    let mut adaptive = false;
    let mut mmco_ops = Vec::new();

    if is_idr {
        no_output_of_prior_pics = br.get_bit();
        long_term_reference_flag = br.get_bit();
    } else {
        adaptive = br.get_bit(); // adaptive_ref_pic_marking_mode_flag
        if adaptive {
            loop {
                let mmco = get_ue_golomb(br)?;
                if mmco == 0 {
                    mmco_ops.push(MmcoOp::End);
                    break;
                }

                let op = match mmco {
                    1 => {
                        let diff = get_ue_golomb(br)?;
                        MmcoOp::ShortTermUnused {
                            difference_of_pic_nums_minus1: diff,
                        }
                    }
                    2 => {
                        let ltp = get_ue_golomb(br)?;
                        MmcoOp::LongTermUnused {
                            long_term_pic_num: ltp,
                        }
                    }
                    3 => {
                        let diff = get_ue_golomb(br)?;
                        let ltfi = get_ue_golomb(br)?;
                        MmcoOp::ShortTermToLongTerm {
                            difference_of_pic_nums_minus1: diff,
                            long_term_frame_idx: ltfi,
                        }
                    }
                    4 => {
                        let max = get_ue_golomb(br)?;
                        MmcoOp::MaxLongTermFrameIdx {
                            max_long_term_frame_idx_plus1: max,
                        }
                    }
                    5 => MmcoOp::Reset,
                    6 => {
                        let ltfi = get_ue_golomb(br)?;
                        MmcoOp::CurrentToLongTerm {
                            long_term_frame_idx: ltfi,
                        }
                    }
                    _ => return Err(Error::InvalidData),
                };

                mmco_ops.push(op);

                // Safety limit
                if mmco_ops.len() > 66 {
                    return Err(Error::InvalidData);
                }
            }
        }
    }

    Ok((
        no_output_of_prior_pics,
        long_term_reference_flag,
        adaptive,
        mmco_ops,
    ))
}

// ---------------------------------------------------------------------------
// Pred weight table parsing
// ---------------------------------------------------------------------------

/// Parse pred_weight_table() from the slice header.
///
/// Reference: ITU-T H.264 Section 7.3.3.2, FFmpeg h264_parse.c:30-120.
fn parse_pred_weight_table(
    br: &mut BitReadBE<'_>,
    hdr: &mut SliceHeader,
    chroma_format_idc: u8,
) -> Result<()> {
    hdr.luma_log2_weight_denom = get_ue_golomb(br)?;
    if hdr.luma_log2_weight_denom > 7 {
        return Err(Error::InvalidData);
    }
    let luma_def = 1i32 << hdr.luma_log2_weight_denom;

    if chroma_format_idc != 0 {
        hdr.chroma_log2_weight_denom = get_ue_golomb(br)?;
        if hdr.chroma_log2_weight_denom > 7 {
            return Err(Error::InvalidData);
        }
    }
    let chroma_def = 1i32 << hdr.chroma_log2_weight_denom;

    let num_lists = if hdr.slice_type.is_b() { 2 } else { 1 };
    let ref_counts = [hdr.num_ref_idx_l0_active, hdr.num_ref_idx_l1_active];

    for (list, &ref_count_u32) in ref_counts.iter().enumerate().take(num_lists) {
        let ref_count = ref_count_u32 as usize;
        let mut luma_weights = Vec::with_capacity(ref_count);
        let mut chroma_weights = Vec::with_capacity(ref_count);

        for _i in 0..ref_count {
            let luma_weight_flag = br.get_bit();
            if luma_weight_flag {
                let w = get_se_golomb(br)?;
                let o = get_se_golomb(br)?;
                if w != luma_def || o != 0 {
                    hdr.use_weight = true;
                }
                luma_weights.push((w, o));
            } else {
                luma_weights.push((luma_def, 0));
            }

            if chroma_format_idc != 0 {
                let chroma_weight_flag = br.get_bit();
                if chroma_weight_flag {
                    let mut cw = [(chroma_def, 0i32); 2];
                    for item in &mut cw {
                        let w = get_se_golomb(br)?;
                        let o = get_se_golomb(br)?;
                        if w != chroma_def || o != 0 {
                            hdr.use_weight_chroma = true;
                        }
                        *item = (w, o);
                    }
                    chroma_weights.push(cw);
                } else {
                    chroma_weights.push([(chroma_def, 0); 2]);
                }
            } else {
                chroma_weights.push([(chroma_def, 0); 2]);
            }
        }

        if list == 0 {
            hdr.luma_weight_l0 = luma_weights;
            hdr.chroma_weight_l0 = chroma_weights;
        } else {
            hdr.luma_weight_l1 = luma_weights;
            hdr.chroma_weight_l1 = chroma_weights;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main slice header parser
// ---------------------------------------------------------------------------

/// Parse an H.264 slice header from RBSP data.
///
/// `data` is the raw NAL unit payload after emulation prevention byte removal
/// (RBSP), starting after the NAL header byte. `nal_type` indicates whether
/// this is an IDR (type 5) or non-IDR (type 1) slice. `nal_ref_idc` is the
/// nal_ref_idc from the NAL header (non-zero means this is a reference picture).
///
/// Reference: FFmpeg `h264_slice_header_parse` in h264_slice.c.
pub fn parse_slice_header(
    data: &[u8],
    nal_type: NalUnitType,
    nal_ref_idc: u8,
    sps_list: &[Option<Sps>; 32],
    pps_list: &[Option<Pps>; 256],
) -> Result<SliceHeader> {
    // Pad data for safe bitstream reading (av-bitstream does 8-byte cache refills).
    let mut padded = Vec::with_capacity(data.len() + 8);
    padded.extend_from_slice(data);
    padded.resize(data.len() + 8, 0);
    let mut br = BitReadBE::new(&padded);

    let mut hdr = SliceHeader::default();
    let is_idr = nal_type == NalUnitType::Idr;

    // 1. first_mb_in_slice (ue)
    hdr.first_mb_in_slice = get_ue_golomb(&mut br)?;

    // 2. slice_type (ue), values 0-9
    let raw_slice_type = get_ue_golomb(&mut br)?;
    if raw_slice_type > 9 {
        return Err(Error::InvalidData);
    }
    hdr.slice_type_fixed = raw_slice_type >= 5;
    hdr.slice_type = SliceType::from_raw(raw_slice_type)?;

    // IDR slices must be I or SI
    if is_idr && !hdr.slice_type.is_intra() {
        return Err(Error::InvalidData);
    }

    // 3. pps_id (ue), look up PPS then SPS
    hdr.pps_id = get_ue_golomb(&mut br)?;
    if hdr.pps_id >= 256 {
        return Err(Error::InvalidData);
    }
    let pps = pps_list[hdr.pps_id as usize]
        .as_ref()
        .ok_or(Error::InvalidData)?;
    let sps = sps_list[pps.sps_id as usize]
        .as_ref()
        .ok_or(Error::InvalidData)?;

    // 4. frame_num (u(log2_max_frame_num) bits)
    hdr.frame_num = br.get_bits_32(sps.log2_max_frame_num as usize);

    // 5. Field coding (only if !frame_mbs_only)
    if !sps.frame_mbs_only_flag {
        hdr.field_pic_flag = br.get_bit();
        if hdr.field_pic_flag {
            hdr.bottom_field_flag = br.get_bit();
        }
    }

    // 6. IDR pic id
    if is_idr {
        hdr.idr_pic_id = get_ue_golomb(&mut br)?;
    }

    // 7. POC type 0
    if sps.poc_type == 0 {
        hdr.pic_order_cnt_lsb = br.get_bits_32(sps.log2_max_poc_lsb as usize);

        if pps.bottom_field_pic_order_in_frame_present && !hdr.field_pic_flag {
            hdr.delta_pic_order_cnt_bottom = get_se_golomb(&mut br)?;
        }
    }

    // 8. POC type 1
    if sps.poc_type == 1 && !sps.delta_pic_order_always_zero_flag {
        hdr.delta_pic_order_cnt[0] = get_se_golomb(&mut br)?;

        if pps.bottom_field_pic_order_in_frame_present && !hdr.field_pic_flag {
            hdr.delta_pic_order_cnt[1] = get_se_golomb(&mut br)?;
        }
    }

    // 9. Redundant pic count
    if pps.redundant_pic_cnt_present {
        hdr.redundant_pic_cnt = get_ue_golomb(&mut br)?;
    }

    // 10. Direct spatial MV prediction (B slices only)
    if hdr.slice_type.is_b() {
        hdr.direct_spatial_mv_pred_flag = br.get_bit();
    }

    // 11. Ref count override
    // Default from PPS
    hdr.num_ref_idx_l0_active = pps.num_ref_idx_l0_default_active;
    hdr.num_ref_idx_l1_active = pps.num_ref_idx_l1_default_active;

    if !hdr.slice_type.is_intra() {
        let num_ref_idx_active_override_flag = br.get_bit();
        if num_ref_idx_active_override_flag {
            hdr.num_ref_idx_l0_active = get_ue_golomb(&mut br)? + 1;
            if hdr.slice_type.is_b() {
                hdr.num_ref_idx_l1_active = get_ue_golomb(&mut br)? + 1;
            }
        }
        // Validate ref counts: max 16 for frames, 32 for fields
        let max_ref = if hdr.field_pic_flag { 32 } else { 16 };
        if hdr.num_ref_idx_l0_active > max_ref {
            return Err(Error::InvalidData);
        }
        if hdr.slice_type.is_b() && hdr.num_ref_idx_l1_active > max_ref {
            return Err(Error::InvalidData);
        }
    }

    // 12. Ref pic list modification (ref_pic_list_modification())
    if !hdr.slice_type.is_intra() {
        let ref_pic_list_modification_flag_l0 = br.get_bit();
        if ref_pic_list_modification_flag_l0 {
            hdr.ref_pic_list_modification_l0 = parse_ref_pic_list_modification(&mut br)?;
        }
    }
    if hdr.slice_type.is_b() {
        let ref_pic_list_modification_flag_l1 = br.get_bit();
        if ref_pic_list_modification_flag_l1 {
            hdr.ref_pic_list_modification_l1 = parse_ref_pic_list_modification(&mut br)?;
        }
    }

    // 13. Weighted prediction
    if (pps.weighted_pred_flag && hdr.slice_type.is_p())
        || (pps.weighted_bipred_idc == 1 && hdr.slice_type.is_b())
    {
        parse_pred_weight_table(&mut br, &mut hdr, sps.chroma_format_idc)?;
    }

    // 14. Dec ref pic marking
    if nal_ref_idc != 0 {
        let (no_output, long_term, adaptive, mmco) = parse_dec_ref_pic_marking(&mut br, is_idr)?;
        hdr.no_output_of_prior_pics = no_output;
        hdr.long_term_reference_flag = long_term;
        hdr.adaptive_ref_pic_marking = adaptive;
        hdr.mmco_ops = mmco;
    }

    // 15. CABAC init idc (only if CABAC and not I/SI)
    if pps.entropy_coding_mode_flag && !hdr.slice_type.is_intra() {
        hdr.cabac_init_idc = get_ue_golomb(&mut br)?;
        if hdr.cabac_init_idc > 2 {
            return Err(Error::InvalidData);
        }
    }

    // 16. Slice QP delta
    hdr.slice_qp_delta = get_se_golomb(&mut br)?;
    hdr.slice_qp = pps.pic_init_qp + hdr.slice_qp_delta;

    // 17. SP/SI specific fields
    if hdr.slice_type == SliceType::SP {
        hdr.sp_for_switch_flag = br.get_bit();
    }
    if hdr.slice_type == SliceType::SP || hdr.slice_type == SliceType::SI {
        hdr.slice_qs_delta = get_se_golomb(&mut br)?;
    }

    // 18. Deblocking filter
    if pps.deblocking_filter_parameters_present {
        hdr.disable_deblocking_filter_idc = get_ue_golomb(&mut br)?;
        if hdr.disable_deblocking_filter_idc > 2 {
            return Err(Error::InvalidData);
        }
        if hdr.disable_deblocking_filter_idc != 1 {
            let alpha_div2 = get_se_golomb(&mut br)?;
            let beta_div2 = get_se_golomb(&mut br)?;
            if !(-6..=6).contains(&alpha_div2) || !(-6..=6).contains(&beta_div2) {
                return Err(Error::InvalidData);
            }
            hdr.slice_alpha_c0_offset = alpha_div2 * 2;
            hdr.slice_beta_offset = beta_div2 * 2;
        }
    }

    // Store the number of header bits consumed so the caller can find
    // the start of macroblock data in the RBSP.
    hdr.header_bits = br.consumed();

    Ok(hdr)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SliceType tests ---

    #[test]
    fn slice_type_from_raw_basic() {
        assert_eq!(SliceType::from_raw(0).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_raw(1).unwrap(), SliceType::B);
        assert_eq!(SliceType::from_raw(2).unwrap(), SliceType::I);
        assert_eq!(SliceType::from_raw(3).unwrap(), SliceType::SP);
        assert_eq!(SliceType::from_raw(4).unwrap(), SliceType::SI);
    }

    #[test]
    fn slice_type_from_raw_high_values() {
        // Values 5-9 map to 0-4
        assert_eq!(SliceType::from_raw(5).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_raw(6).unwrap(), SliceType::B);
        assert_eq!(SliceType::from_raw(7).unwrap(), SliceType::I);
        assert_eq!(SliceType::from_raw(8).unwrap(), SliceType::SP);
        assert_eq!(SliceType::from_raw(9).unwrap(), SliceType::SI);
    }

    #[test]
    fn slice_type_predicates() {
        assert!(SliceType::I.is_intra());
        assert!(SliceType::SI.is_intra());
        assert!(!SliceType::P.is_intra());
        assert!(!SliceType::B.is_intra());
        assert!(!SliceType::SP.is_intra());

        assert!(SliceType::B.is_b());
        assert!(!SliceType::P.is_b());
        assert!(!SliceType::I.is_b());

        assert!(SliceType::P.is_p());
        assert!(SliceType::SP.is_p());
        assert!(!SliceType::B.is_p());
        assert!(!SliceType::I.is_p());
    }

    // --- Helper: bit vector construction ---

    fn encode_ue(bits: &mut Vec<bool>, val: u32) {
        let code = val + 1;
        let n = 32 - code.leading_zeros();
        for _ in 0..n - 1 {
            bits.push(false);
        }
        for i in (0..n).rev() {
            bits.push((code >> i) & 1 != 0);
        }
    }

    fn encode_se(bits: &mut Vec<bool>, val: i32) {
        let ue_val = if val <= 0 {
            (-2 * val) as u32
        } else {
            (2 * val - 1) as u32
        };
        encode_ue(bits, ue_val);
    }

    fn push_bits(bits: &mut Vec<bool>, val: u32, n: usize) {
        for i in (0..n).rev() {
            bits.push((val >> i) & 1 != 0);
        }
    }

    fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
        let num_bytes = (bits.len() + 7) / 8;
        let mut bytes = vec![0u8; num_bytes];
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                bytes[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        bytes
    }

    /// Build minimal test SPS and PPS, returning (sps_list, pps_list).
    fn test_parameter_sets() -> ([Option<Sps>; 32], [Option<Pps>; 256]) {
        let sps = Sps {
            sps_id: 0,
            profile_idc: 66,
            log2_max_frame_num: 4,
            poc_type: 0,
            log2_max_poc_lsb: 4,
            frame_mbs_only_flag: true,
            mb_width: 20,
            mb_height: 15,
            ..Sps::default()
        };

        let pps = Pps {
            pps_id: 0,
            sps_id: 0,
            entropy_coding_mode_flag: false, // CAVLC
            bottom_field_pic_order_in_frame_present: false,
            num_slice_groups: 1,
            num_ref_idx_l0_default_active: 1,
            num_ref_idx_l1_default_active: 1,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp: 26,
            pic_init_qs: 26,
            chroma_qp_index_offset: [0, 0],
            deblocking_filter_parameters_present: true,
            constrained_intra_pred: false,
            redundant_pic_cnt_present: false,
            transform_8x8_mode: false,
            scaling_matrix4: [[16; 16]; 6],
            scaling_matrix8: [[16; 64]; 6],
            scaling_matrix_present: false,
        };

        let mut sps_list: [Option<Sps>; 32] = Default::default();
        sps_list[0] = Some(sps);

        let mut pps_list: [Option<Pps>; 256] = std::array::from_fn(|_| None);
        pps_list[0] = Some(pps);

        (sps_list, pps_list)
    }

    // --- Slice header parsing tests ---

    #[test]
    fn parse_idr_i_slice_basic() {
        let (sps_list, pps_list) = test_parameter_sets();

        // Build a minimal IDR I-slice header bitstream:
        // first_mb_in_slice = 0, slice_type = 7 (I, all-same),
        // pps_id = 0, frame_num = 0, idr_pic_id = 0,
        // pic_order_cnt_lsb = 0,
        // (ref pic marking for IDR: no_output_of_prior=0, long_term_ref=0),
        // slice_qp_delta = 0,
        // deblocking: disable_deblocking_filter_idc = 0,
        //             alpha_offset_div2 = 0, beta_offset_div2 = 0
        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // first_mb_in_slice = 0
        encode_ue(&mut bits, 7); // slice_type = 7 (I, all same)
        encode_ue(&mut bits, 0); // pps_id = 0
        push_bits(&mut bits, 0, 4); // frame_num = 0 (4 bits)
        encode_ue(&mut bits, 0); // idr_pic_id = 0
        push_bits(&mut bits, 0, 4); // pic_order_cnt_lsb = 0 (4 bits)
        // dec_ref_pic_marking (IDR, nal_ref_idc != 0):
        bits.push(false); // no_output_of_prior_pics = 0
        bits.push(false); // long_term_reference_flag = 0
        // slice_qp_delta = 0
        encode_se(&mut bits, 0);
        // deblocking:
        encode_ue(&mut bits, 0); // disable_deblocking_filter_idc = 0
        encode_se(&mut bits, 0); // alpha_offset_div2 = 0
        encode_se(&mut bits, 0); // beta_offset_div2 = 0

        let data = bits_to_bytes(&bits);

        let hdr = parse_slice_header(
            &data,
            NalUnitType::Idr,
            3, // nal_ref_idc = 3
            &sps_list,
            &pps_list,
        )
        .expect("should parse IDR I-slice header");

        assert_eq!(hdr.first_mb_in_slice, 0);
        assert_eq!(hdr.slice_type, SliceType::I);
        assert!(hdr.slice_type_fixed);
        assert_eq!(hdr.pps_id, 0);
        assert_eq!(hdr.frame_num, 0);
        assert_eq!(hdr.idr_pic_id, 0);
        assert_eq!(hdr.pic_order_cnt_lsb, 0);
        assert!(!hdr.no_output_of_prior_pics);
        assert!(!hdr.long_term_reference_flag);
        assert_eq!(hdr.slice_qp, 26);
        assert_eq!(hdr.disable_deblocking_filter_idc, 0);
        assert_eq!(hdr.slice_alpha_c0_offset, 0);
        assert_eq!(hdr.slice_beta_offset, 0);
    }

    #[test]
    fn parse_non_idr_p_slice() {
        let (sps_list, pps_list) = test_parameter_sets();

        // Non-IDR P-slice, frame_num=1, poc_lsb=2
        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // first_mb_in_slice = 0
        encode_ue(&mut bits, 0); // slice_type = 0 (P)
        encode_ue(&mut bits, 0); // pps_id = 0
        push_bits(&mut bits, 1, 4); // frame_num = 1
        push_bits(&mut bits, 2, 4); // pic_order_cnt_lsb = 2
        // ref count override: no
        bits.push(false); // num_ref_idx_active_override_flag = 0
        // ref_pic_list_modification: flag=0
        bits.push(false);
        // dec_ref_pic_marking (non-IDR, nal_ref_idc != 0):
        bits.push(false); // adaptive_ref_pic_marking_mode_flag = 0
        // slice_qp_delta = -2
        encode_se(&mut bits, -2);
        // deblocking: disabled
        encode_ue(&mut bits, 1); // disable_deblocking_filter_idc = 1

        let data = bits_to_bytes(&bits);

        let hdr = parse_slice_header(
            &data,
            NalUnitType::Slice,
            2, // nal_ref_idc = 2
            &sps_list,
            &pps_list,
        )
        .expect("should parse P-slice header");

        assert_eq!(hdr.slice_type, SliceType::P);
        assert!(!hdr.slice_type_fixed);
        assert_eq!(hdr.frame_num, 1);
        assert_eq!(hdr.pic_order_cnt_lsb, 2);
        assert_eq!(hdr.num_ref_idx_l0_active, 1); // default from PPS
        assert!(!hdr.adaptive_ref_pic_marking);
        assert_eq!(hdr.slice_qp, 24); // 26 + (-2)
        assert_eq!(hdr.disable_deblocking_filter_idc, 1);
    }

    #[test]
    fn parse_slice_type_invalid_rejected() {
        let (sps_list, pps_list) = test_parameter_sets();

        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // first_mb_in_slice
        encode_ue(&mut bits, 10); // slice_type = 10 (invalid)

        let data = bits_to_bytes(&bits);
        let result = parse_slice_header(&data, NalUnitType::Slice, 0, &sps_list, &pps_list);
        assert!(result.is_err());
    }

    #[test]
    fn parse_idr_non_intra_rejected() {
        let (sps_list, pps_list) = test_parameter_sets();

        // IDR NAL with P-slice type should be rejected
        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // first_mb_in_slice
        encode_ue(&mut bits, 0); // slice_type = 0 (P)

        let data = bits_to_bytes(&bits);
        let result = parse_slice_header(&data, NalUnitType::Idr, 3, &sps_list, &pps_list);
        assert!(result.is_err());
    }

    #[test]
    fn parse_deblocking_filter_offsets() {
        let (sps_list, pps_list) = test_parameter_sets();

        // IDR I-slice with custom deblocking offsets
        let mut bits = Vec::new();
        encode_ue(&mut bits, 0); // first_mb_in_slice
        encode_ue(&mut bits, 2); // slice_type = I
        encode_ue(&mut bits, 0); // pps_id
        push_bits(&mut bits, 0, 4); // frame_num
        encode_ue(&mut bits, 0); // idr_pic_id
        push_bits(&mut bits, 0, 4); // poc_lsb
        // dec_ref_pic_marking (IDR):
        bits.push(false); // no_output_of_prior_pics
        bits.push(false); // long_term_reference_flag
        // slice_qp_delta = 3
        encode_se(&mut bits, 3);
        // deblocking: enabled with offsets
        encode_ue(&mut bits, 0); // disable_deblocking_filter_idc = 0
        encode_se(&mut bits, 3); // alpha_offset_div2 = 3
        encode_se(&mut bits, -2); // beta_offset_div2 = -2

        let data = bits_to_bytes(&bits);

        let hdr = parse_slice_header(&data, NalUnitType::Idr, 3, &sps_list, &pps_list)
            .expect("should parse slice with deblocking offsets");

        assert_eq!(hdr.slice_qp, 29); // 26 + 3
        assert_eq!(hdr.disable_deblocking_filter_idc, 0);
        assert_eq!(hdr.slice_alpha_c0_offset, 6); // 3 * 2
        assert_eq!(hdr.slice_beta_offset, -4); // -2 * 2
    }

    #[test]
    fn mmco_op_variants() {
        // Verify MmcoOp enum can represent all operations
        let ops = vec![
            MmcoOp::ShortTermUnused {
                difference_of_pic_nums_minus1: 0,
            },
            MmcoOp::LongTermUnused {
                long_term_pic_num: 1,
            },
            MmcoOp::ShortTermToLongTerm {
                difference_of_pic_nums_minus1: 0,
                long_term_frame_idx: 2,
            },
            MmcoOp::MaxLongTermFrameIdx {
                max_long_term_frame_idx_plus1: 3,
            },
            MmcoOp::Reset,
            MmcoOp::CurrentToLongTerm {
                long_term_frame_idx: 0,
            },
            MmcoOp::End,
        ];
        assert_eq!(ops.len(), 7);
    }
}
