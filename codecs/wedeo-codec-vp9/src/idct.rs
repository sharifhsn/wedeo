// VP9 inverse transforms (IDCT, IADST, IWHT).
//
// Translated from FFmpeg's libavcodec/vp9dsp_template.c (Ronald S. Bultje,
// Clément Bœsch). The butterfly constants are verbatim cosine values scaled
// by 2^14. All intermediate arithmetic uses i32 with wrapping operations to
// match FFmpeg's `int` (32-bit) behaviour for 8-bit mode — intermediate
// values can overflow 32 bits and FFmpeg wraps at that width.
//
// LGPL-2.1-or-later — same licence as FFmpeg.

use crate::types::{TxSize, TxType};

/// VP9 rotates pairs using coefficients scaled by 2^14. Keep the full
/// wrapping 32-bit sum until the rounding point specified by each butterfly.
#[inline]
fn weighted_pair(a: i32, a_weight: i32, b: i32, b_weight: i32) -> i32 {
    a.wrapping_mul(a_weight)
        .wrapping_add(b.wrapping_mul(b_weight))
}

#[inline]
fn round_q14(value: i32) -> i32 {
    value.wrapping_add(1 << 13) >> 14
}

#[inline]
fn scale_inv_sqrt2(value: i32) -> i32 {
    round_q14(value.wrapping_mul(11585))
}

// ---------------------------------------------------------------------------
// 4-point IDCT  (idct4_1d in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// 4-point 1-D inverse DCT.
///
/// Input stride is 1 (packed array). Translated directly from `idct4_1d`.
pub fn idct4(input: &[i32; 4], output: &mut [i32; 4]) {
    let (i0, i1, i2, i3) = (input[0], input[1], input[2], input[3]);

    let t0 = scale_inv_sqrt2(i0.wrapping_add(i2));
    let t1 = scale_inv_sqrt2(i0.wrapping_sub(i2));
    let t2 = round_q14(weighted_pair(i1, 6270, i3, -15137));
    let t3 = round_q14(weighted_pair(i1, 15137, i3, 6270));

    output[0] = t0.wrapping_add(t3);
    output[1] = t1.wrapping_add(t2);
    output[2] = t1.wrapping_sub(t2);
    output[3] = t0.wrapping_sub(t3);
}

// ---------------------------------------------------------------------------
// 4-point IADST  (iadst4_1d in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// 4-point 1-D inverse asymmetric DST.
pub fn iadst4(input: &[i32; 4], output: &mut [i32; 4]) {
    let (i0, i1, i2, i3) = (input[0], input[1], input[2], input[3]);

    let t0 = weighted_pair(i0, 5283, i2, 15212).wrapping_add(9929_i32.wrapping_mul(i3));
    let t1 = weighted_pair(i0, 9929, i2, -5283).wrapping_sub(15212_i32.wrapping_mul(i3));
    let t2 = 13377_i32.wrapping_mul(i0.wrapping_sub(i2).wrapping_add(i3));
    let t3 = 13377_i32.wrapping_mul(i1);

    output[0] = round_q14(t0.wrapping_add(t3));
    output[1] = round_q14(t1.wrapping_add(t3));
    output[2] = round_q14(t2);
    output[3] = round_q14(t0.wrapping_add(t1).wrapping_sub(t3));
}

// ---------------------------------------------------------------------------
// 4-point IWHT  (iwht4_1d in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// 4-point 1-D inverse Walsh-Hadamard transform (used for lossless coding).
///
/// `input` is a packed slice indexed by natural order (stride=1 implied).
/// Pass 0 right-shifts inputs by 2; pass 1 uses them as-is.
fn iwht4_1d(input: &[i32; 4], output: &mut [i32; 4], pass: usize) {
    // VP9 IWHT reorders: in[0]→t0, in[3]→t1, in[1]→t2, in[2]→t3.
    let (t0, t1, t2, t3) = if pass == 0 {
        (input[0] >> 2, input[3] >> 2, input[1] >> 2, input[2] >> 2)
    } else {
        (input[0], input[3], input[1], input[2])
    };

    let t0 = t0 + t2;
    let t3 = t3 - t1;
    let t4 = (t0 - t3) >> 1;
    let t1 = t4 - t1;
    let t2 = t4 - t2;
    let t0 = t0 - t1;
    let t3 = t3 + t2;

    output[0] = t0;
    output[1] = t1;
    output[2] = t2;
    output[3] = t3;
}

/// 4-point inverse WHT (public wrapper using packed array).
pub fn iwht4(input: &[i32; 4], output: &mut [i32; 4]) {
    iwht4_1d(input, output, 0);
}

// ---------------------------------------------------------------------------
// 8-point IDCT  (idct8_1d in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// 8-point 1-D inverse DCT.
pub fn idct8(input: &[i32; 8], output: &mut [i32; 8]) {
    let i = |k: usize| input[k];

    let t0a = scale_inv_sqrt2(i(0).wrapping_add(i(4)));
    let t1a = scale_inv_sqrt2(i(0).wrapping_sub(i(4)));
    let t2a = round_q14(weighted_pair(i(2), 6270, i(6), -15137));
    let t3a = round_q14(weighted_pair(i(2), 15137, i(6), 6270));
    let t4a = round_q14(weighted_pair(i(1), 3196, i(7), -16069));
    let t5a = round_q14(weighted_pair(i(5), 13623, i(3), -9102));
    let t6a = round_q14(weighted_pair(i(5), 9102, i(3), 13623));
    let t7a = round_q14(weighted_pair(i(1), 16069, i(7), 3196));

    let t0 = t0a.wrapping_add(t3a);
    let t1 = t1a.wrapping_add(t2a);
    let t2 = t1a.wrapping_sub(t2a);
    let t3 = t0a.wrapping_sub(t3a);
    let t4 = t4a.wrapping_add(t5a);
    let t5a = t4a.wrapping_sub(t5a);
    let t7 = t7a.wrapping_add(t6a);
    let t6a = t7a.wrapping_sub(t6a);

    let t5 = scale_inv_sqrt2(t6a.wrapping_sub(t5a));
    let t6 = scale_inv_sqrt2(t6a.wrapping_add(t5a));

    output[0] = t0.wrapping_add(t7);
    output[1] = t1.wrapping_add(t6);
    output[2] = t2.wrapping_add(t5);
    output[3] = t3.wrapping_add(t4);
    output[4] = t3.wrapping_sub(t4);
    output[5] = t2.wrapping_sub(t5);
    output[6] = t1.wrapping_sub(t6);
    output[7] = t0.wrapping_sub(t7);
}

// ---------------------------------------------------------------------------
// 8-point IADST  (iadst8_1d in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// 8-point 1-D inverse asymmetric DST.
pub fn iadst8(input: &[i32; 8], output: &mut [i32; 8]) {
    let i = |k: usize| input[k];

    let t0a = weighted_pair(i(7), 16305, i(0), 1606);
    let t1a = weighted_pair(i(7), 1606, i(0), -16305);
    let t2a = weighted_pair(i(5), 14449, i(2), 7723);
    let t3a = weighted_pair(i(5), 7723, i(2), -14449);
    let t4a = weighted_pair(i(3), 10394, i(4), 12665);
    let t5a = weighted_pair(i(3), 12665, i(4), -10394);
    let t6a = weighted_pair(i(1), 4756, i(6), 15679);
    let t7a = weighted_pair(i(1), 15679, i(6), -4756);

    let t0 = round_q14(t0a.wrapping_add(t4a));
    let t1 = round_q14(t1a.wrapping_add(t5a));
    let t2 = round_q14(t2a.wrapping_add(t6a));
    let t3 = round_q14(t3a.wrapping_add(t7a));
    let t4 = round_q14(t0a.wrapping_sub(t4a));
    let t5 = round_q14(t1a.wrapping_sub(t5a));
    let t6 = round_q14(t2a.wrapping_sub(t6a));
    let t7 = round_q14(t3a.wrapping_sub(t7a));

    let t4a = weighted_pair(t4, 15137, t5, 6270);
    let t5a = weighted_pair(t4, 6270, t5, -15137);
    let t6a = weighted_pair(t7, 15137, t6, -6270);
    let t7a = weighted_pair(t7, 6270, t6, 15137);

    let out0 = t0.wrapping_add(t2);
    let out7 = t1.wrapping_add(t3).wrapping_neg();
    let t2 = t0.wrapping_sub(t2);
    let t3 = t1.wrapping_sub(t3);

    let out1 = round_q14(t4a.wrapping_add(t6a));
    let out1 = out1.wrapping_neg();
    let out6 = round_q14(t5a.wrapping_add(t7a));
    let t6 = round_q14(t4a.wrapping_sub(t6a));
    let t7 = round_q14(t5a.wrapping_sub(t7a));

    let out3 = scale_inv_sqrt2(t2.wrapping_add(t3));
    let out3 = out3.wrapping_neg();
    let out4 = scale_inv_sqrt2(t2.wrapping_sub(t3));
    let out2 = scale_inv_sqrt2(t6.wrapping_add(t7));
    let out5 = scale_inv_sqrt2(t6.wrapping_sub(t7));
    let out5 = out5.wrapping_neg();

    output[0] = out0;
    output[1] = out1;
    output[2] = out2;
    output[3] = out3;
    output[4] = out4;
    output[5] = out5;
    output[6] = out6;
    output[7] = out7;
}

// ---------------------------------------------------------------------------
// 16-point IDCT  (idct16_1d in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// 16-point 1-D inverse DCT.
pub fn idct16(input: &[i32; 16], output: &mut [i32; 16]) {
    let i = |k: usize| input[k];

    let t0a = scale_inv_sqrt2(i(0).wrapping_add(i(8)));
    let t1a = scale_inv_sqrt2(i(0).wrapping_sub(i(8)));
    let t2a = round_q14(weighted_pair(i(4), 6270, i(12), -15137));
    let t3a = round_q14(weighted_pair(i(4), 15137, i(12), 6270));
    let t4a = round_q14(weighted_pair(i(2), 3196, i(14), -16069));
    let t7a = round_q14(weighted_pair(i(2), 16069, i(14), 3196));
    let t5a = round_q14(weighted_pair(i(10), 13623, i(6), -9102));
    let t6a = round_q14(weighted_pair(i(10), 9102, i(6), 13623));
    let t8a = round_q14(weighted_pair(i(1), 1606, i(15), -16305));
    let t15a = round_q14(weighted_pair(i(1), 16305, i(15), 1606));
    let t9a = round_q14(weighted_pair(i(9), 12665, i(7), -10394));
    let t14a = round_q14(weighted_pair(i(9), 10394, i(7), 12665));
    let t10a = round_q14(weighted_pair(i(5), 7723, i(11), -14449));
    let t13a = round_q14(weighted_pair(i(5), 14449, i(11), 7723));
    let t11a = round_q14(weighted_pair(i(13), 15679, i(3), -4756));
    let t12a = round_q14(weighted_pair(i(13), 4756, i(3), 15679));

    let t0 = t0a.wrapping_add(t3a);
    let t1 = t1a.wrapping_add(t2a);
    let t2 = t1a.wrapping_sub(t2a);
    let t3 = t0a.wrapping_sub(t3a);
    let t4 = t4a.wrapping_add(t5a);
    let t5 = t4a.wrapping_sub(t5a);
    let t6 = t7a.wrapping_sub(t6a);
    let t7 = t7a.wrapping_add(t6a);
    let t8 = t8a.wrapping_add(t9a);
    let t9 = t8a.wrapping_sub(t9a);
    let t10 = t11a.wrapping_sub(t10a);
    let t11 = t11a.wrapping_add(t10a);
    let t12 = t12a.wrapping_add(t13a);
    let t13 = t12a.wrapping_sub(t13a);
    let t14 = t15a.wrapping_sub(t14a);
    let t15 = t15a.wrapping_add(t14a);

    let t5a = scale_inv_sqrt2(t6.wrapping_sub(t5));
    let t6a = scale_inv_sqrt2(t6.wrapping_add(t5));
    let t9a = round_q14(weighted_pair(t14, 6270, t9, -15137));
    let t14a = round_q14(weighted_pair(t14, 15137, t9, 6270));
    let t10a = round_q14(weighted_pair(t13, 15137, t10, 6270).wrapping_neg());
    let t13a = round_q14(weighted_pair(t13, 6270, t10, -15137));

    let t0a = t0.wrapping_add(t7);
    let t1a = t1.wrapping_add(t6a);
    let t2a = t2.wrapping_add(t5a);
    let t3a = t3.wrapping_add(t4);
    let t4 = t3.wrapping_sub(t4);
    let t5 = t2.wrapping_sub(t5a);
    let t6 = t1.wrapping_sub(t6a);
    let t7 = t0.wrapping_sub(t7);
    let t8a = t8.wrapping_add(t11);
    let t9 = t9a.wrapping_add(t10a);
    let t10 = t9a.wrapping_sub(t10a);
    let t11a = t8.wrapping_sub(t11);
    let t12a = t15.wrapping_sub(t12);
    let t13 = t14a.wrapping_sub(t13a);
    let t14 = t14a.wrapping_add(t13a);
    let t15a = t15.wrapping_add(t12);

    let t10a = scale_inv_sqrt2(t13.wrapping_sub(t10));
    let t13a = scale_inv_sqrt2(t13.wrapping_add(t10));
    let t11 = scale_inv_sqrt2(t12a.wrapping_sub(t11a));
    let t12 = scale_inv_sqrt2(t12a.wrapping_add(t11a));

    output[0] = t0a.wrapping_add(t15a);
    output[1] = t1a.wrapping_add(t14);
    output[2] = t2a.wrapping_add(t13a);
    output[3] = t3a.wrapping_add(t12);
    output[4] = t4.wrapping_add(t11);
    output[5] = t5.wrapping_add(t10a);
    output[6] = t6.wrapping_add(t9);
    output[7] = t7.wrapping_add(t8a);
    output[8] = t7.wrapping_sub(t8a);
    output[9] = t6.wrapping_sub(t9);
    output[10] = t5.wrapping_sub(t10a);
    output[11] = t4.wrapping_sub(t11);
    output[12] = t3a.wrapping_sub(t12);
    output[13] = t2a.wrapping_sub(t13a);
    output[14] = t1a.wrapping_sub(t14);
    output[15] = t0a.wrapping_sub(t15a);
}

// ---------------------------------------------------------------------------
// 16-point IADST  (iadst16_1d in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// 16-point 1-D inverse asymmetric DST.
pub fn iadst16(input: &[i32; 16], output: &mut [i32; 16]) {
    let i = |k: usize| input[k];

    let t0 = weighted_pair(i(15), 16364, i(0), 804);
    let t1 = weighted_pair(i(15), 804, i(0), -16364);
    let t2 = weighted_pair(i(13), 15893, i(2), 3981);
    let t3 = weighted_pair(i(13), 3981, i(2), -15893);
    let t4 = weighted_pair(i(11), 14811, i(4), 7005);
    let t5 = weighted_pair(i(11), 7005, i(4), -14811);
    let t6 = weighted_pair(i(9), 13160, i(6), 9760);
    let t7 = weighted_pair(i(9), 9760, i(6), -13160);
    let t8 = weighted_pair(i(7), 11003, i(8), 12140);
    let t9 = weighted_pair(i(7), 12140, i(8), -11003);
    let t10 = weighted_pair(i(5), 8423, i(10), 14053);
    let t11 = weighted_pair(i(5), 14053, i(10), -8423);
    let t12 = weighted_pair(i(3), 5520, i(12), 15426);
    let t13 = weighted_pair(i(3), 15426, i(12), -5520);
    let t14 = weighted_pair(i(1), 2404, i(14), 16207);
    let t15 = weighted_pair(i(1), 16207, i(14), -2404);

    let t0a = round_q14(t0.wrapping_add(t8));
    let t1a = round_q14(t1.wrapping_add(t9));
    let t2a = round_q14(t2.wrapping_add(t10));
    let t3a = round_q14(t3.wrapping_add(t11));
    let t4a = round_q14(t4.wrapping_add(t12));
    let t5a = round_q14(t5.wrapping_add(t13));
    let t6a = round_q14(t6.wrapping_add(t14));
    let t7a = round_q14(t7.wrapping_add(t15));
    let t8a = round_q14(t0.wrapping_sub(t8));
    let t9a = round_q14(t1.wrapping_sub(t9));
    let t10a = round_q14(t2.wrapping_sub(t10));
    let t11a = round_q14(t3.wrapping_sub(t11));
    let t12a = round_q14(t4.wrapping_sub(t12));
    let t13a = round_q14(t5.wrapping_sub(t13));
    let t14a = round_q14(t6.wrapping_sub(t14));
    let t15a = round_q14(t7.wrapping_sub(t15));

    let t8 = weighted_pair(t8a, 16069, t9a, 3196);
    let t9 = weighted_pair(t8a, 3196, t9a, -16069);
    let t10 = weighted_pair(t10a, 9102, t11a, 13623);
    let t11 = weighted_pair(t10a, 13623, t11a, -9102);
    let t12 = weighted_pair(t13a, 16069, t12a, -3196);
    let t13 = weighted_pair(t13a, 3196, t12a, 16069);
    let t14 = weighted_pair(t15a, 9102, t14a, -13623);
    let t15 = weighted_pair(t15a, 13623, t14a, 9102);

    let t0 = t0a.wrapping_add(t4a);
    let t1 = t1a.wrapping_add(t5a);
    let t2 = t2a.wrapping_add(t6a);
    let t3 = t3a.wrapping_add(t7a);
    let t4 = t0a.wrapping_sub(t4a);
    let t5 = t1a.wrapping_sub(t5a);
    let t6 = t2a.wrapping_sub(t6a);
    let t7 = t3a.wrapping_sub(t7a);
    let t8a = round_q14(t8.wrapping_add(t12));
    let t9a = round_q14(t9.wrapping_add(t13));
    let t10a = round_q14(t10.wrapping_add(t14));
    let t11a = round_q14(t11.wrapping_add(t15));
    let t12a = round_q14(t8.wrapping_sub(t12));
    let t13a = round_q14(t9.wrapping_sub(t13));
    let t14a = round_q14(t10.wrapping_sub(t14));
    let t15a = round_q14(t11.wrapping_sub(t15));

    let t4a = weighted_pair(t4, 15137, t5, 6270);
    let t5a = weighted_pair(t4, 6270, t5, -15137);
    let t6a = weighted_pair(t7, 15137, t6, -6270);
    let t7a = weighted_pair(t7, 6270, t6, 15137);
    let t12 = weighted_pair(t12a, 15137, t13a, 6270);
    let t13 = weighted_pair(t12a, 6270, t13a, -15137);
    let t14 = weighted_pair(t15a, 15137, t14a, -6270);
    let t15 = weighted_pair(t15a, 6270, t14a, 15137);

    let out0 = t0.wrapping_add(t2);
    let out15 = t1.wrapping_add(t3).wrapping_neg();
    let t2a = t0.wrapping_sub(t2);
    let t3a = t1.wrapping_sub(t3);

    let out3 = round_q14(t4a.wrapping_add(t6a));
    let out3 = out3.wrapping_neg();
    let out12 = round_q14(t5a.wrapping_add(t7a));
    let t6 = round_q14(t4a.wrapping_sub(t6a));
    let t7 = round_q14(t5a.wrapping_sub(t7a));

    let out1 = t8a.wrapping_add(t10a).wrapping_neg();
    let out14 = t9a.wrapping_add(t11a);
    let t10 = t8a.wrapping_sub(t10a);
    let t11 = t9a.wrapping_sub(t11a);

    let out2 = round_q14(t12.wrapping_add(t14));
    let out13 = round_q14(t13.wrapping_add(t15));
    let out13 = out13.wrapping_neg();
    let t14a = round_q14(t12.wrapping_sub(t14));
    let t15a = round_q14(t13.wrapping_sub(t15));

    let out7 = scale_inv_sqrt2(t2a.wrapping_add(t3a).wrapping_neg());
    let out8 = scale_inv_sqrt2(t2a.wrapping_sub(t3a));
    let out4 = scale_inv_sqrt2(t7.wrapping_add(t6));
    let out11 = scale_inv_sqrt2(t7.wrapping_sub(t6));
    let out6 = scale_inv_sqrt2(t11.wrapping_add(t10));
    let out9 = scale_inv_sqrt2(t11.wrapping_sub(t10));
    let out5 = scale_inv_sqrt2(t14a.wrapping_add(t15a).wrapping_neg());
    let out10 = scale_inv_sqrt2(t14a.wrapping_sub(t15a));

    output[0] = out0;
    output[1] = out1;
    output[2] = out2;
    output[3] = out3;
    output[4] = out4;
    output[5] = out5;
    output[6] = out6;
    output[7] = out7;
    output[8] = out8;
    output[9] = out9;
    output[10] = out10;
    output[11] = out11;
    output[12] = out12;
    output[13] = out13;
    output[14] = out14;
    output[15] = out15;
}

// ---------------------------------------------------------------------------
// 32-point IDCT  (idct32_1d in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// 32-point 1-D inverse DCT.
pub fn idct32(input: &[i32; 32], output: &mut [i32; 32]) {
    let i = |k: usize| input[k];

    // Stage 1 — direct butterfly from frequency-domain inputs.
    let t0a = scale_inv_sqrt2(i(0).wrapping_add(i(16)));
    let t1a = scale_inv_sqrt2(i(0).wrapping_sub(i(16)));
    let t2a = round_q14(weighted_pair(i(8), 6270, i(24), -15137));
    let t3a = round_q14(weighted_pair(i(8), 15137, i(24), 6270));
    let t4a = round_q14(weighted_pair(i(4), 3196, i(28), -16069));
    let t7a = round_q14(weighted_pair(i(4), 16069, i(28), 3196));
    let t5a = round_q14(weighted_pair(i(20), 13623, i(12), -9102));
    let t6a = round_q14(weighted_pair(i(20), 9102, i(12), 13623));
    let t8a = round_q14(weighted_pair(i(2), 1606, i(30), -16305));
    let t15a = round_q14(weighted_pair(i(2), 16305, i(30), 1606));
    let t9a = round_q14(weighted_pair(i(18), 12665, i(14), -10394));
    let t14a = round_q14(weighted_pair(i(18), 10394, i(14), 12665));
    let t10a = round_q14(weighted_pair(i(10), 7723, i(22), -14449));
    let t13a = round_q14(weighted_pair(i(10), 14449, i(22), 7723));
    let t11a = round_q14(weighted_pair(i(26), 15679, i(6), -4756));
    let t12a = round_q14(weighted_pair(i(26), 4756, i(6), 15679));
    let t16a = round_q14(weighted_pair(i(1), 804, i(31), -16364));
    let t31a = round_q14(weighted_pair(i(1), 16364, i(31), 804));
    let t17a = round_q14(weighted_pair(i(17), 12140, i(15), -11003));
    let t30a = round_q14(weighted_pair(i(17), 11003, i(15), 12140));
    let t18a = round_q14(weighted_pair(i(9), 7005, i(23), -14811));
    let t29a = round_q14(weighted_pair(i(9), 14811, i(23), 7005));
    let t19a = round_q14(weighted_pair(i(25), 15426, i(7), -5520));
    let t28a = round_q14(weighted_pair(i(25), 5520, i(7), 15426));
    let t20a = round_q14(weighted_pair(i(5), 3981, i(27), -15893));
    let t27a = round_q14(weighted_pair(i(5), 15893, i(27), 3981));
    let t21a = round_q14(weighted_pair(i(21), 14053, i(11), -8423));
    let t26a = round_q14(weighted_pair(i(21), 8423, i(11), 14053));
    let t22a = round_q14(weighted_pair(i(13), 9760, i(19), -13160));
    let t25a = round_q14(weighted_pair(i(13), 13160, i(19), 9760));
    let t23a = round_q14(weighted_pair(i(29), 16207, i(3), -2404));
    let t24a = round_q14(weighted_pair(i(29), 2404, i(3), 16207));

    // Stage 2
    let t0 = t0a.wrapping_add(t3a);
    let t1 = t1a.wrapping_add(t2a);
    let t2 = t1a.wrapping_sub(t2a);
    let t3 = t0a.wrapping_sub(t3a);
    let t4 = t4a.wrapping_add(t5a);
    let t5 = t4a.wrapping_sub(t5a);
    let t6 = t7a.wrapping_sub(t6a);
    let t7 = t7a.wrapping_add(t6a);
    let t8 = t8a.wrapping_add(t9a);
    let t9 = t8a.wrapping_sub(t9a);
    let t10 = t11a.wrapping_sub(t10a);
    let t11 = t11a.wrapping_add(t10a);
    let t12 = t12a.wrapping_add(t13a);
    let t13 = t12a.wrapping_sub(t13a);
    let t14 = t15a.wrapping_sub(t14a);
    let t15 = t15a.wrapping_add(t14a);
    let t16 = t16a.wrapping_add(t17a);
    let t17 = t16a.wrapping_sub(t17a);
    let t18 = t19a.wrapping_sub(t18a);
    let t19 = t19a.wrapping_add(t18a);
    let t20 = t20a.wrapping_add(t21a);
    let t21 = t20a.wrapping_sub(t21a);
    let t22 = t23a.wrapping_sub(t22a);
    let t23 = t23a.wrapping_add(t22a);
    let t24 = t24a.wrapping_add(t25a);
    let t25 = t24a.wrapping_sub(t25a);
    let t26 = t27a.wrapping_sub(t26a);
    let t27 = t27a.wrapping_add(t26a);
    let t28 = t28a.wrapping_add(t29a);
    let t29 = t28a.wrapping_sub(t29a);
    let t30 = t31a.wrapping_sub(t30a);
    let t31 = t31a.wrapping_add(t30a);

    // Stage 3
    let t5a = scale_inv_sqrt2(t6.wrapping_sub(t5));
    let t6a = scale_inv_sqrt2(t6.wrapping_add(t5));
    let t9a = round_q14(weighted_pair(t14, 6270, t9, -15137));
    let t14a = round_q14(weighted_pair(t14, 15137, t9, 6270));
    let t10a = round_q14(weighted_pair(t13, 15137, t10, 6270).wrapping_neg());
    let t13a = round_q14(weighted_pair(t13, 6270, t10, -15137));
    let t17a = round_q14(weighted_pair(t30, 3196, t17, -16069));
    let t30a = round_q14(weighted_pair(t30, 16069, t17, 3196));
    let t18a = round_q14(weighted_pair(t29, 16069, t18, 3196).wrapping_neg());
    let t29a = round_q14(weighted_pair(t29, 3196, t18, -16069));
    let t21a = round_q14(weighted_pair(t26, 13623, t21, -9102));
    let t26a = round_q14(weighted_pair(t26, 9102, t21, 13623));
    let t22a = round_q14(weighted_pair(t25, 9102, t22, 13623).wrapping_neg());
    let t25a = round_q14(weighted_pair(t25, 13623, t22, -9102));

    // Stage 4
    let t0a = t0.wrapping_add(t7);
    let t1a = t1.wrapping_add(t6a);
    let t2a = t2.wrapping_add(t5a);
    let t3a = t3.wrapping_add(t4);
    let t4a = t3.wrapping_sub(t4);
    let t5 = t2.wrapping_sub(t5a);
    let t6 = t1.wrapping_sub(t6a);
    let t7a = t0.wrapping_sub(t7);
    let t8a = t8.wrapping_add(t11);
    let t9 = t9a.wrapping_add(t10a);
    let t10 = t9a.wrapping_sub(t10a);
    let t11a = t8.wrapping_sub(t11);
    let t12a = t15.wrapping_sub(t12);
    let t13 = t14a.wrapping_sub(t13a);
    let t14 = t14a.wrapping_add(t13a);
    let t15a = t15.wrapping_add(t12);
    let t16a = t16.wrapping_add(t19);
    let t17 = t17a.wrapping_add(t18a);
    let t18 = t17a.wrapping_sub(t18a);
    let t19a = t16.wrapping_sub(t19);
    let t20a = t23.wrapping_sub(t20);
    let t21 = t22a.wrapping_sub(t21a);
    let t22 = t22a.wrapping_add(t21a);
    let t23a = t23.wrapping_add(t20);
    let t24a = t24.wrapping_add(t27);
    let t25 = t25a.wrapping_add(t26a);
    let t26 = t25a.wrapping_sub(t26a);
    let t27a = t24.wrapping_sub(t27);
    let t28a = t31.wrapping_sub(t28);
    let t29 = t30a.wrapping_sub(t29a);
    let t30 = t30a.wrapping_add(t29a);
    let t31a = t31.wrapping_add(t28);

    // Stage 5
    let t10a = scale_inv_sqrt2(t13.wrapping_sub(t10));
    let t13a = scale_inv_sqrt2(t13.wrapping_add(t10));
    let t11 = scale_inv_sqrt2(t12a.wrapping_sub(t11a));
    let t12 = scale_inv_sqrt2(t12a.wrapping_add(t11a));
    let t18a = round_q14(weighted_pair(t29, 6270, t18, -15137));
    let t29a = round_q14(weighted_pair(t29, 15137, t18, 6270));
    let t19 = round_q14(weighted_pair(t28a, 6270, t19a, -15137));
    let t28 = round_q14(weighted_pair(t28a, 15137, t19a, 6270));
    let t20 = round_q14(weighted_pair(t27a, 15137, t20a, 6270).wrapping_neg());
    let t27 = round_q14(weighted_pair(t27a, 6270, t20a, -15137));
    let t21a = round_q14(weighted_pair(t26, 15137, t21, 6270).wrapping_neg());
    let t26a = round_q14(weighted_pair(t26, 6270, t21, -15137));

    // Stage 6
    let t0 = t0a.wrapping_add(t15a);
    let t1 = t1a.wrapping_add(t14);
    let t2 = t2a.wrapping_add(t13a);
    let t3 = t3a.wrapping_add(t12);
    let t4 = t4a.wrapping_add(t11);
    let t5a = t5.wrapping_add(t10a);
    let t6a = t6.wrapping_add(t9);
    let t7 = t7a.wrapping_add(t8a);
    let t8 = t7a.wrapping_sub(t8a);
    let t9a = t6.wrapping_sub(t9);
    let t10 = t5.wrapping_sub(t10a);
    let t11a = t4a.wrapping_sub(t11);
    let t12a = t3a.wrapping_sub(t12);
    let t13 = t2a.wrapping_sub(t13a);
    let t14a = t1a.wrapping_sub(t14);
    let t15 = t0a.wrapping_sub(t15a);
    let t16 = t16a.wrapping_add(t23a);
    let t17a = t17.wrapping_add(t22);
    let t18 = t18a.wrapping_add(t21a);
    let t19a = t19.wrapping_add(t20);
    let t20a = t19.wrapping_sub(t20);
    let t21 = t18a.wrapping_sub(t21a);
    let t22a = t17.wrapping_sub(t22);
    let t23 = t16a.wrapping_sub(t23a);
    let t24 = t31a.wrapping_sub(t24a);
    let t25a = t30.wrapping_sub(t25);
    let t26 = t29a.wrapping_sub(t26a);
    let t27a = t28.wrapping_sub(t27);
    let t28a = t28.wrapping_add(t27);
    let t29 = t29a.wrapping_add(t26a);
    let t30a = t30.wrapping_add(t25);
    let t31 = t31a.wrapping_add(t24a);

    // Stage 7
    let t20 = scale_inv_sqrt2(t27a.wrapping_sub(t20a));
    let t27 = scale_inv_sqrt2(t27a.wrapping_add(t20a));
    let t21a = scale_inv_sqrt2(t26.wrapping_sub(t21));
    let t26a = scale_inv_sqrt2(t26.wrapping_add(t21));
    let t22 = scale_inv_sqrt2(t25a.wrapping_sub(t22a));
    let t25 = scale_inv_sqrt2(t25a.wrapping_add(t22a));
    let t23a = scale_inv_sqrt2(t24.wrapping_sub(t23));
    let t24a = scale_inv_sqrt2(t24.wrapping_add(t23));

    output[0] = t0.wrapping_add(t31);
    output[1] = t1.wrapping_add(t30a);
    output[2] = t2.wrapping_add(t29);
    output[3] = t3.wrapping_add(t28a);
    output[4] = t4.wrapping_add(t27);
    output[5] = t5a.wrapping_add(t26a);
    output[6] = t6a.wrapping_add(t25);
    output[7] = t7.wrapping_add(t24a);
    output[8] = t8.wrapping_add(t23a);
    output[9] = t9a.wrapping_add(t22);
    output[10] = t10.wrapping_add(t21a);
    output[11] = t11a.wrapping_add(t20);
    output[12] = t12a.wrapping_add(t19a);
    output[13] = t13.wrapping_add(t18);
    output[14] = t14a.wrapping_add(t17a);
    output[15] = t15.wrapping_add(t16);
    output[16] = t15.wrapping_sub(t16);
    output[17] = t14a.wrapping_sub(t17a);
    output[18] = t13.wrapping_sub(t18);
    output[19] = t12a.wrapping_sub(t19a);
    output[20] = t11a.wrapping_sub(t20);
    output[21] = t10.wrapping_sub(t21a);
    output[22] = t9a.wrapping_sub(t22);
    output[23] = t8.wrapping_sub(t23a);
    output[24] = t7.wrapping_sub(t24a);
    output[25] = t6a.wrapping_sub(t25);
    output[26] = t5a.wrapping_sub(t26a);
    output[27] = t4.wrapping_sub(t27);
    output[28] = t3.wrapping_sub(t28a);
    output[29] = t2.wrapping_sub(t29);
    output[30] = t1.wrapping_sub(t30a);
    output[31] = t0.wrapping_sub(t31);
}

// ---------------------------------------------------------------------------
// 2-D transform dispatch  (itxfm_wrapper / itxfm_wrap in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// Clamp pixel value to the valid range for the given bit depth.
#[inline(always)]
fn clip_pixel(v: i32, max_val: i32) -> u8 {
    v.clamp(0, max_val) as u8
}

/// Apply the inverse 2-D transform and add to the prediction buffer.
///
/// Translates the `itxfm_wrapper` macro from `vp9dsp_template.c`.
///
/// `dst`     — prediction buffer (modified in place).
/// `stride`  — row stride of `dst` in bytes.
/// `coefs`   — flat row-major array of dequantized coefficients.
/// `tx_size` — transform block size.
/// `tx_type` — combination of row/column transforms.
/// `bit_depth` — output bit depth (8, 10, or 12).
/// `eob`     — last non-zero coefficient index + 1 (0 means all-zero block).
pub fn itxfm_add(
    dst: &mut [u8],
    stride: usize,
    coefs: &[i32],
    tx_size: TxSize,
    tx_type: TxType,
    bit_depth: u8,
    eob: usize,
) {
    // Nothing to do for an empty block.
    if eob == 0 {
        return;
    }

    let max_val = (1_i32 << bit_depth) - 1;

    match tx_size {
        TxSize::Tx4x4 => inverse_block(dst, stride, coefs, tx_type, max_val, eob, 4, idct4, iadst4),
        TxSize::Tx8x8 => inverse_block(dst, stride, coefs, tx_type, max_val, eob, 5, idct8, iadst8),
        TxSize::Tx16x16 => inverse_block(
            dst, stride, coefs, tx_type, max_val, eob, 6, idct16, iadst16,
        ),
        // VP9 only defines DCT_DCT for a 32x32 block.
        TxSize::Tx32x32 => inverse_block(
            dst,
            stride,
            coefs,
            TxType::DctDct,
            max_val,
            eob,
            6,
            idct32,
            idct32,
        ),
    }
}

/// 2-D inverse transform + add for lossless 4×4 (IWHT).
///
/// Matches FFmpeg's `itxfm_wrapper(iwht, iwht, 4, 0, 0)` from
/// `vp9dsp_template.c`.  Both passes read with stride 4 (processing
/// columns), then add+clip to dst.
pub fn itxfm_add_lossless(dst: &mut [u8], stride: usize, coefs: &[i32], bit_depth: u8) {
    let max_val = (1_i32 << bit_depth) - 1;
    let sz = 4_usize;

    // Pass 1 — process each column of the coefficient matrix.
    // FFmpeg: iwht4_1d(block + i, 4, tmp + i * 4, 0)
    //   IN(n) = block[i + n*4]  (column i, stride=4)
    //   OUT → tmp[i*4 .. i*4+4]  (contiguous)
    let mut tmp = [0i32; 16];
    for col in 0..sz {
        let inp: [i32; 4] = core::array::from_fn(|r| coefs[col + r * sz]);
        let mut out = [0i32; 4];
        iwht4_1d(&inp, &mut out, 0);
        tmp[col * sz..col * sz + sz].copy_from_slice(&out);
    }

    // Pass 2 — process each column of tmp, add+clip to dst.
    // FFmpeg: iwht4_1d(tmp + i, 4, out, 1)
    //   IN(n) = tmp[i + n*4]  (column i, stride=4)
    //   then dst[j*stride] = clip(dst[j*stride] + out[j]); dst++
    for col in 0..sz {
        let inp: [i32; 4] = core::array::from_fn(|r| tmp[col + r * sz]);
        let mut out = [0i32; 4];
        iwht4_1d(&inp, &mut out, 1);
        for row in 0..sz {
            let p = dst[row * stride + col] as i32;
            dst[row * stride + col] = clip_pixel(p + out[row], max_val);
        }
    }
}

// ---------------------------------------------------------------------------
// Generic NxN 2-D inverse transform + add
// ---------------------------------------------------------------------------

type Transform1d<const N: usize> = fn(&[i32; N], &mut [i32; N]);

#[inline]
#[allow(clippy::too_many_arguments)]
fn inverse_block<const N: usize>(
    dst: &mut [u8],
    stride: usize,
    coefs: &[i32],
    tx_type: TxType,
    max_val: i32,
    eob: usize,
    bits: u32,
    dct: Transform1d<N>,
    adst: Transform1d<N>,
) {
    let rounded = |value: i32| value.wrapping_add(1 << (bits - 1)) >> bits;
    if eob == 1 && tx_type == TxType::DctDct {
        let dc = rounded(scale_inv_sqrt2(scale_inv_sqrt2(coefs[0])));
        for row in 0..N {
            for px in &mut dst[row * stride..row * stride + N] {
                *px = clip_pixel(i32::from(*px) + dc, max_val);
            }
        }
        return;
    }

    // FFmpeg stores each column contiguously in its transposed dctcoef
    // array. Both passes truncate to i16 at the same points as the 8-bit path.
    let col_transform = if matches!(tx_type, TxType::DctAdst | TxType::AdstAdst) {
        adst
    } else {
        dct
    };
    let row_transform = if matches!(tx_type, TxType::AdstDct | TxType::AdstAdst) {
        adst
    } else {
        dct
    };
    let mut transposed = [[0i16; N]; N];
    let mut out = [0; N];
    for (col, intermediate) in transposed.iter_mut().enumerate() {
        let input = core::array::from_fn(|row| coefs[row * N + col]);
        col_transform(&input, &mut out);
        *intermediate = out.map(|v| v as i16);
    }
    for col in 0..N {
        let input = core::array::from_fn(|row| i32::from(transposed[row][col]));
        row_transform(&input, &mut out);
        for (row, &value) in out.iter().enumerate() {
            let px = &mut dst[row * stride + col];
            *px = clip_pixel(i32::from(*px) + rounded(i32::from(value as i16)), max_val);
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A block of all-zero coefficients should leave the prediction unchanged.
    #[test]
    fn idct_all_zeros_leaves_pred_unchanged() {
        let pred: Vec<u8> = (0..16).map(|i| i as u8 * 8).collect();
        let mut dst = pred.clone();
        let coefs = vec![0i32; 16];
        itxfm_add(&mut dst, 4, &coefs, TxSize::Tx4x4, TxType::DctDct, 8, 0);
        assert_eq!(dst, pred, "zero coefficients must not modify prediction");
    }

    /// DC-only coefficient: all output pixels should be pred + dc_value.
    #[test]
    fn idct4_dc_only() {
        // A DC coefficient of 512 in VP9 dequantized form.
        // After double 11585/16384 scaling: round((512*11585+8192)>>14 = 228.x → 228;
        // then round((228*11585+8192)>>14 = 161.x → 161; with bits=4 shift:
        // (161 + 8) >> 4 = 10.
        let mut dst = vec![100u8; 16]; // 4×4 block of 100
        let mut coefs = vec![0i32; 16];
        coefs[0] = 512;
        itxfm_add(&mut dst, 4, &coefs, TxSize::Tx4x4, TxType::DctDct, 8, 1);
        // All pixels should be 100 + the dc_add value (same for all positions).
        let first = dst[0];
        assert!(dst.iter().all(|&v| v == first), "DC-only must be uniform");
        assert!(
            first > 100,
            "positive DC coefficient should increase pixel values"
        );
    }

    /// idct4 round-trip sanity: known values from the VP9 spec test vector.
    #[test]
    fn idct4_known_values() {
        // Input: [16384, 0, 0, 0] represents a DC-only coefficient equal to 2^14.
        // Expected: all outputs ≈ ((16384*11585+8192)>>14)*11585+8192)>>14 = 11585.
        let input = [16384, 0, 0, 0];
        let mut output = [0i32; 4];
        idct4(&input, &mut output);
        // All outputs of idct4 should equal approximately 11585 (the 1D DC result).
        for v in output {
            assert!((v - 11585).abs() <= 1, "idct4 DC: expected ~11585, got {v}");
        }
    }

    /// iwht4 known values.
    #[test]
    fn iwht4_dc_only() {
        // Input [4, 0, 0, 0] with the IWHT reordering:
        // in[0]→t0=1, in[3]→t1=0, in[1]→t2=0, in[2]→t3=0 (all after >>2).
        // Butterfly: t0=1+0=1, t3=0-0=0, t4=(1-0)/2=0,
        //   t1=0-0=0, t2=0-0=0, t0=1-0=1, t3=0+0=0.
        // Output order: out[0]=t0=1, out[1]=t1=0, out[2]=t2=0, out[3]=t3=0.
        let input = [4, 0, 0, 0];
        let mut output = [0i32; 4];
        iwht4(&input, &mut output);
        assert_eq!(output, [1, 0, 0, 0]);
    }

    /// 8-point IDCT DC.
    #[test]
    fn idct8_dc_only() {
        let input = [16384, 0, 0, 0, 0, 0, 0, 0];
        let mut output = [0i32; 8];
        idct8(&input, &mut output);
        let expected = ((16384_i64 * 11585 + (1 << 13)) >> 14) as i32;
        for v in output {
            assert!((v - expected).abs() <= 1);
        }
    }
}
