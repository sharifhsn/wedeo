// FFI declarations for FFmpeg's hand-written NEON assembly.
//
// These functions are compiled from vendored .S files in asm/aarch64/ by build.rs.
// Only available when cfg(has_asm) is set (aarch64 + feature = "asm").
//
// Calling conventions follow FFmpeg's function pointer typedefs:
// - qpel_mc_func:    fn(dst: *mut u8, src: *const u8, stride: isize)
// - h264_chroma_mc:  fn(dst: *mut u8, src: *const u8, stride: isize, h: i32, x: i32, y: i32)
// - idct_add:        fn(dst: *mut u8, block: *mut i16, stride: i32)
// - loop_filter:     fn(pix: *mut u8, stride: isize, alpha: i32, beta: i32, tc0: *const i8)

// ---------------------------------------------------------------------------
// Quarter-pel luma MC (h264qpel_neon.S)
// ---------------------------------------------------------------------------
//
// Signature: fn(dst: *mut u8, src: *const u8, stride: isize)
// - dst must be aligned to block width (8 or 16)
// - src must be at least 1-byte aligned
// - stride is the row stride in bytes for BOTH dst and src
// - Block height is fixed: 16 for qpel16, 8 for qpel8
//
// Note: mc00 (fullpel copy) is NOT in qpel assembly — it comes from hpeldsp.
// Position 0 is handled in Rust (simple row-by-row memcpy).

macro_rules! declare_qpel {
    ($($name:ident),* $(,)?) => {
        unsafe extern "C" {
            $(pub fn $name(dst: *mut u8, src: *const u8, stride: isize);)*
        }
    };
}

// 8-bit put (write) variants — 16×16 blocks (positions 1-15, no mc00)
declare_qpel! {
    ff_put_h264_qpel16_mc10_neon,
    ff_put_h264_qpel16_mc20_neon,
    ff_put_h264_qpel16_mc30_neon,
    ff_put_h264_qpel16_mc01_neon,
    ff_put_h264_qpel16_mc11_neon,
    ff_put_h264_qpel16_mc21_neon,
    ff_put_h264_qpel16_mc31_neon,
    ff_put_h264_qpel16_mc02_neon,
    ff_put_h264_qpel16_mc12_neon,
    ff_put_h264_qpel16_mc22_neon,
    ff_put_h264_qpel16_mc32_neon,
    ff_put_h264_qpel16_mc03_neon,
    ff_put_h264_qpel16_mc13_neon,
    ff_put_h264_qpel16_mc23_neon,
    ff_put_h264_qpel16_mc33_neon,
}

// 8-bit put (write) variants — 8×8 blocks (positions 1-15, no mc00)
declare_qpel! {
    ff_put_h264_qpel8_mc10_neon,
    ff_put_h264_qpel8_mc20_neon,
    ff_put_h264_qpel8_mc30_neon,
    ff_put_h264_qpel8_mc01_neon,
    ff_put_h264_qpel8_mc11_neon,
    ff_put_h264_qpel8_mc21_neon,
    ff_put_h264_qpel8_mc31_neon,
    ff_put_h264_qpel8_mc02_neon,
    ff_put_h264_qpel8_mc12_neon,
    ff_put_h264_qpel8_mc22_neon,
    ff_put_h264_qpel8_mc32_neon,
    ff_put_h264_qpel8_mc03_neon,
    ff_put_h264_qpel8_mc13_neon,
    ff_put_h264_qpel8_mc23_neon,
    ff_put_h264_qpel8_mc33_neon,
}

// 8-bit avg (read-modify-write) variants — 16×16 blocks (positions 1-15, no mc00)
declare_qpel! {
    ff_avg_h264_qpel16_mc10_neon,
    ff_avg_h264_qpel16_mc20_neon,
    ff_avg_h264_qpel16_mc30_neon,
    ff_avg_h264_qpel16_mc01_neon,
    ff_avg_h264_qpel16_mc11_neon,
    ff_avg_h264_qpel16_mc21_neon,
    ff_avg_h264_qpel16_mc31_neon,
    ff_avg_h264_qpel16_mc02_neon,
    ff_avg_h264_qpel16_mc12_neon,
    ff_avg_h264_qpel16_mc22_neon,
    ff_avg_h264_qpel16_mc32_neon,
    ff_avg_h264_qpel16_mc03_neon,
    ff_avg_h264_qpel16_mc13_neon,
    ff_avg_h264_qpel16_mc23_neon,
    ff_avg_h264_qpel16_mc33_neon,
}

// 8-bit avg (read-modify-write) variants — 8×8 blocks (positions 1-15, no mc00)
declare_qpel! {
    ff_avg_h264_qpel8_mc10_neon,
    ff_avg_h264_qpel8_mc20_neon,
    ff_avg_h264_qpel8_mc30_neon,
    ff_avg_h264_qpel8_mc01_neon,
    ff_avg_h264_qpel8_mc11_neon,
    ff_avg_h264_qpel8_mc21_neon,
    ff_avg_h264_qpel8_mc31_neon,
    ff_avg_h264_qpel8_mc02_neon,
    ff_avg_h264_qpel8_mc12_neon,
    ff_avg_h264_qpel8_mc22_neon,
    ff_avg_h264_qpel8_mc32_neon,
    ff_avg_h264_qpel8_mc03_neon,
    ff_avg_h264_qpel8_mc13_neon,
    ff_avg_h264_qpel8_mc23_neon,
    ff_avg_h264_qpel8_mc33_neon,
}

// ---------------------------------------------------------------------------
// Chroma MC (h264cmc_neon.S)
// ---------------------------------------------------------------------------
//
// Signature: fn(dst: *mut u8, src: *const u8, stride: isize, h: i32, x: i32, y: i32)
// - x, y are 1/8-pel fractional offsets (0..7)
// - h is the block height
// - Block width is fixed by function name: mc8 = 8 wide, mc4 = 4 wide, mc2 = 2 wide

unsafe extern "C" {
    pub fn ff_put_h264_chroma_mc8_neon(
        dst: *mut u8,
        src: *const u8,
        stride: isize,
        h: i32,
        x: i32,
        y: i32,
    );
    pub fn ff_put_h264_chroma_mc4_neon(
        dst: *mut u8,
        src: *const u8,
        stride: isize,
        h: i32,
        x: i32,
        y: i32,
    );
    pub fn ff_put_h264_chroma_mc2_neon(
        dst: *mut u8,
        src: *const u8,
        stride: isize,
        h: i32,
        x: i32,
        y: i32,
    );
    pub fn ff_avg_h264_chroma_mc8_neon(
        dst: *mut u8,
        src: *const u8,
        stride: isize,
        h: i32,
        x: i32,
        y: i32,
    );
    pub fn ff_avg_h264_chroma_mc4_neon(
        dst: *mut u8,
        src: *const u8,
        stride: isize,
        h: i32,
        x: i32,
        y: i32,
    );
    pub fn ff_avg_h264_chroma_mc2_neon(
        dst: *mut u8,
        src: *const u8,
        stride: isize,
        h: i32,
        x: i32,
        y: i32,
    );
}

// ---------------------------------------------------------------------------
// IDCT (h264idct_neon.S)
// ---------------------------------------------------------------------------
//
// idct_add:     fn(dst: *mut u8, block: *mut i16, stride: i32)
// idct_dc_add:  fn(dst: *mut u8, block: *mut i16, stride: i32)
//
// The block pointer is mutable — the function zeroes the coefficients.

unsafe extern "C" {
    pub fn ff_h264_idct_add_neon(dst: *mut u8, block: *mut i16, stride: i32);
    pub fn ff_h264_idct_dc_add_neon(dst: *mut u8, block: *mut i16, stride: i32);
    pub fn ff_h264_idct8_add_neon(dst: *mut u8, block: *mut i16, stride: i32);
    pub fn ff_h264_idct8_dc_add_neon(dst: *mut u8, block: *mut i16, stride: i32);
}

// ---------------------------------------------------------------------------
// Deblocking loop filter (h264dsp_neon.S)
// ---------------------------------------------------------------------------
//
// Normal (bS 1-3): fn(pix: *mut u8, stride: isize, alpha: i32, beta: i32, tc0: *const i8)
// Intra (bS=4):    fn(pix: *mut u8, stride: isize, alpha: i32, beta: i32)
//
// v_ = vertical edge filter (filters along vertical boundary, pixels horizontal)
// h_ = horizontal edge filter (filters along horizontal boundary, pixels vertical)

unsafe extern "C" {
    pub fn ff_h264_v_loop_filter_luma_neon(
        pix: *mut u8,
        stride: isize,
        alpha: i32,
        beta: i32,
        tc0: *const i8,
    );
    pub fn ff_h264_h_loop_filter_luma_neon(
        pix: *mut u8,
        stride: isize,
        alpha: i32,
        beta: i32,
        tc0: *const i8,
    );
    pub fn ff_h264_v_loop_filter_luma_intra_neon(
        pix: *mut u8,
        stride: isize,
        alpha: i32,
        beta: i32,
    );
    pub fn ff_h264_h_loop_filter_luma_intra_neon(
        pix: *mut u8,
        stride: isize,
        alpha: i32,
        beta: i32,
    );
    pub fn ff_h264_v_loop_filter_chroma_neon(
        pix: *mut u8,
        stride: isize,
        alpha: i32,
        beta: i32,
        tc0: *const i8,
    );
    pub fn ff_h264_h_loop_filter_chroma_neon(
        pix: *mut u8,
        stride: isize,
        alpha: i32,
        beta: i32,
        tc0: *const i8,
    );
    pub fn ff_h264_v_loop_filter_chroma_intra_neon(
        pix: *mut u8,
        stride: isize,
        alpha: i32,
        beta: i32,
    );
    pub fn ff_h264_h_loop_filter_chroma_intra_neon(
        pix: *mut u8,
        stride: isize,
        alpha: i32,
        beta: i32,
    );
}

// ---------------------------------------------------------------------------
// Weight / biweight (h264dsp_neon.S)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    pub fn ff_weight_h264_pixels_16_neon(
        block: *mut u8,
        stride: isize,
        height: i32,
        log2_denom: i32,
        weight: i32,
        offset: i32,
    );
    pub fn ff_weight_h264_pixels_8_neon(
        block: *mut u8,
        stride: isize,
        height: i32,
        log2_denom: i32,
        weight: i32,
        offset: i32,
    );
    pub fn ff_weight_h264_pixels_4_neon(
        block: *mut u8,
        stride: isize,
        height: i32,
        log2_denom: i32,
        weight: i32,
        offset: i32,
    );
    pub fn ff_biweight_h264_pixels_16_neon(
        dst: *mut u8,
        src: *mut u8,
        stride: isize,
        height: i32,
        log2_denom: i32,
        weightd: i32,
        weights: i32,
        offset: i32,
    );
    pub fn ff_biweight_h264_pixels_8_neon(
        dst: *mut u8,
        src: *mut u8,
        stride: isize,
        height: i32,
        log2_denom: i32,
        weightd: i32,
        weights: i32,
        offset: i32,
    );
    pub fn ff_biweight_h264_pixels_4_neon(
        dst: *mut u8,
        src: *mut u8,
        stride: isize,
        height: i32,
        log2_denom: i32,
        weightd: i32,
        weights: i32,
        offset: i32,
    );
}

// ---------------------------------------------------------------------------
// Function pointer table types and construction
// ---------------------------------------------------------------------------

/// FFmpeg qpel MC function signature: `fn(dst, src, stride)`.
/// Block height is implicit (16 for qpel16, 8 for qpel8).
pub type QpelFn = unsafe extern "C" fn(*mut u8, *const u8, isize);

/// Qpel put tables: index `dy * 4 + dx`.
/// Position 0 (mc00 = fullpel copy) is `None` — handled by Rust memcpy.
pub static QPEL_PUT_16: [Option<QpelFn>; 16] = [
    None, // mc00: fullpel — not in qpel asm
    Some(ff_put_h264_qpel16_mc10_neon),
    Some(ff_put_h264_qpel16_mc20_neon),
    Some(ff_put_h264_qpel16_mc30_neon),
    Some(ff_put_h264_qpel16_mc01_neon),
    Some(ff_put_h264_qpel16_mc11_neon),
    Some(ff_put_h264_qpel16_mc21_neon),
    Some(ff_put_h264_qpel16_mc31_neon),
    Some(ff_put_h264_qpel16_mc02_neon),
    Some(ff_put_h264_qpel16_mc12_neon),
    Some(ff_put_h264_qpel16_mc22_neon),
    Some(ff_put_h264_qpel16_mc32_neon),
    Some(ff_put_h264_qpel16_mc03_neon),
    Some(ff_put_h264_qpel16_mc13_neon),
    Some(ff_put_h264_qpel16_mc23_neon),
    Some(ff_put_h264_qpel16_mc33_neon),
];

pub static QPEL_PUT_8: [Option<QpelFn>; 16] = [
    None, // mc00: fullpel
    Some(ff_put_h264_qpel8_mc10_neon),
    Some(ff_put_h264_qpel8_mc20_neon),
    Some(ff_put_h264_qpel8_mc30_neon),
    Some(ff_put_h264_qpel8_mc01_neon),
    Some(ff_put_h264_qpel8_mc11_neon),
    Some(ff_put_h264_qpel8_mc21_neon),
    Some(ff_put_h264_qpel8_mc31_neon),
    Some(ff_put_h264_qpel8_mc02_neon),
    Some(ff_put_h264_qpel8_mc12_neon),
    Some(ff_put_h264_qpel8_mc22_neon),
    Some(ff_put_h264_qpel8_mc32_neon),
    Some(ff_put_h264_qpel8_mc03_neon),
    Some(ff_put_h264_qpel8_mc13_neon),
    Some(ff_put_h264_qpel8_mc23_neon),
    Some(ff_put_h264_qpel8_mc33_neon),
];

pub static QPEL_AVG_16: [Option<QpelFn>; 16] = [
    None, // mc00: fullpel avg
    Some(ff_avg_h264_qpel16_mc10_neon),
    Some(ff_avg_h264_qpel16_mc20_neon),
    Some(ff_avg_h264_qpel16_mc30_neon),
    Some(ff_avg_h264_qpel16_mc01_neon),
    Some(ff_avg_h264_qpel16_mc11_neon),
    Some(ff_avg_h264_qpel16_mc21_neon),
    Some(ff_avg_h264_qpel16_mc31_neon),
    Some(ff_avg_h264_qpel16_mc02_neon),
    Some(ff_avg_h264_qpel16_mc12_neon),
    Some(ff_avg_h264_qpel16_mc22_neon),
    Some(ff_avg_h264_qpel16_mc32_neon),
    Some(ff_avg_h264_qpel16_mc03_neon),
    Some(ff_avg_h264_qpel16_mc13_neon),
    Some(ff_avg_h264_qpel16_mc23_neon),
    Some(ff_avg_h264_qpel16_mc33_neon),
];

pub static QPEL_AVG_8: [Option<QpelFn>; 16] = [
    None, // mc00: fullpel avg
    Some(ff_avg_h264_qpel8_mc10_neon),
    Some(ff_avg_h264_qpel8_mc20_neon),
    Some(ff_avg_h264_qpel8_mc30_neon),
    Some(ff_avg_h264_qpel8_mc01_neon),
    Some(ff_avg_h264_qpel8_mc11_neon),
    Some(ff_avg_h264_qpel8_mc21_neon),
    Some(ff_avg_h264_qpel8_mc31_neon),
    Some(ff_avg_h264_qpel8_mc02_neon),
    Some(ff_avg_h264_qpel8_mc12_neon),
    Some(ff_avg_h264_qpel8_mc22_neon),
    Some(ff_avg_h264_qpel8_mc32_neon),
    Some(ff_avg_h264_qpel8_mc03_neon),
    Some(ff_avg_h264_qpel8_mc13_neon),
    Some(ff_avg_h264_qpel8_mc23_neon),
    Some(ff_avg_h264_qpel8_mc33_neon),
];
