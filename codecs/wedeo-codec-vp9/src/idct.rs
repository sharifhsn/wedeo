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

// ---------------------------------------------------------------------------
// 4-point IDCT  (idct4_1d in vp9dsp_template.c)
// ---------------------------------------------------------------------------

/// 4-point 1-D inverse DCT.
///
/// Input stride is 1 (packed array). Translated directly from `idct4_1d`.
pub fn idct4(input: &[i32; 4], output: &mut [i32; 4]) {
    let (i0, i1, i2, i3) = (input[0], input[1], input[2], input[3]);

    let t0 = i0
        .wrapping_add(i2)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t1 = i0
        .wrapping_sub(i2)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t2 = i1
        .wrapping_mul(6270)
        .wrapping_sub(i3.wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t3 = i1
        .wrapping_mul(15137)
        .wrapping_add(i3.wrapping_mul(6270))
        .wrapping_add(1 << 13)
        >> 14;

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

    let t0 = 5283_i32
        .wrapping_mul(i0)
        .wrapping_add(15212_i32.wrapping_mul(i2))
        .wrapping_add(9929_i32.wrapping_mul(i3));
    let t1 = 9929_i32
        .wrapping_mul(i0)
        .wrapping_sub(5283_i32.wrapping_mul(i2))
        .wrapping_sub(15212_i32.wrapping_mul(i3));
    let t2 = 13377_i32.wrapping_mul(i0.wrapping_sub(i2).wrapping_add(i3));
    let t3 = 13377_i32.wrapping_mul(i1);

    output[0] = t0.wrapping_add(t3).wrapping_add(1 << 13) >> 14;
    output[1] = t1.wrapping_add(t3).wrapping_add(1 << 13) >> 14;
    output[2] = t2.wrapping_add(1 << 13) >> 14;
    output[3] = t0.wrapping_add(t1).wrapping_sub(t3).wrapping_add(1 << 13) >> 14;
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

    let t0a = i(0)
        .wrapping_add(i(4))
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t1a = i(0)
        .wrapping_sub(i(4))
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t2a = i(2)
        .wrapping_mul(6270)
        .wrapping_sub(i(6).wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t3a = i(2)
        .wrapping_mul(15137)
        .wrapping_add(i(6).wrapping_mul(6270))
        .wrapping_add(1 << 13)
        >> 14;
    let t4a = i(1)
        .wrapping_mul(3196)
        .wrapping_sub(i(7).wrapping_mul(16069))
        .wrapping_add(1 << 13)
        >> 14;
    let t5a = i(5)
        .wrapping_mul(13623)
        .wrapping_sub(i(3).wrapping_mul(9102))
        .wrapping_add(1 << 13)
        >> 14;
    let t6a = i(5)
        .wrapping_mul(9102)
        .wrapping_add(i(3).wrapping_mul(13623))
        .wrapping_add(1 << 13)
        >> 14;
    let t7a = i(1)
        .wrapping_mul(16069)
        .wrapping_add(i(7).wrapping_mul(3196))
        .wrapping_add(1 << 13)
        >> 14;

    let t0 = t0a.wrapping_add(t3a);
    let t1 = t1a.wrapping_add(t2a);
    let t2 = t1a.wrapping_sub(t2a);
    let t3 = t0a.wrapping_sub(t3a);
    let t4 = t4a.wrapping_add(t5a);
    let t5a = t4a.wrapping_sub(t5a);
    let t7 = t7a.wrapping_add(t6a);
    let t6a = t7a.wrapping_sub(t6a);

    let t5 = t6a
        .wrapping_sub(t5a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t6 = t6a
        .wrapping_add(t5a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;

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

    let t0a = 16305_i32
        .wrapping_mul(i(7))
        .wrapping_add(1606_i32.wrapping_mul(i(0)));
    let t1a = 1606_i32
        .wrapping_mul(i(7))
        .wrapping_sub(16305_i32.wrapping_mul(i(0)));
    let t2a = 14449_i32
        .wrapping_mul(i(5))
        .wrapping_add(7723_i32.wrapping_mul(i(2)));
    let t3a = 7723_i32
        .wrapping_mul(i(5))
        .wrapping_sub(14449_i32.wrapping_mul(i(2)));
    let t4a = 10394_i32
        .wrapping_mul(i(3))
        .wrapping_add(12665_i32.wrapping_mul(i(4)));
    let t5a = 12665_i32
        .wrapping_mul(i(3))
        .wrapping_sub(10394_i32.wrapping_mul(i(4)));
    let t6a = 4756_i32
        .wrapping_mul(i(1))
        .wrapping_add(15679_i32.wrapping_mul(i(6)));
    let t7a = 15679_i32
        .wrapping_mul(i(1))
        .wrapping_sub(4756_i32.wrapping_mul(i(6)));

    let t0 = t0a.wrapping_add(t4a).wrapping_add(1 << 13) >> 14;
    let t1 = t1a.wrapping_add(t5a).wrapping_add(1 << 13) >> 14;
    let t2 = t2a.wrapping_add(t6a).wrapping_add(1 << 13) >> 14;
    let t3 = t3a.wrapping_add(t7a).wrapping_add(1 << 13) >> 14;
    let t4 = t0a.wrapping_sub(t4a).wrapping_add(1 << 13) >> 14;
    let t5 = t1a.wrapping_sub(t5a).wrapping_add(1 << 13) >> 14;
    let t6 = t2a.wrapping_sub(t6a).wrapping_add(1 << 13) >> 14;
    let t7 = t3a.wrapping_sub(t7a).wrapping_add(1 << 13) >> 14;

    let t4a = 15137_i32
        .wrapping_mul(t4)
        .wrapping_add(6270_i32.wrapping_mul(t5));
    let t5a = 6270_i32
        .wrapping_mul(t4)
        .wrapping_sub(15137_i32.wrapping_mul(t5));
    let t6a = 15137_i32
        .wrapping_mul(t7)
        .wrapping_sub(6270_i32.wrapping_mul(t6));
    let t7a = 6270_i32
        .wrapping_mul(t7)
        .wrapping_add(15137_i32.wrapping_mul(t6));

    let out0 = t0.wrapping_add(t2);
    let out7 = t1.wrapping_add(t3).wrapping_neg();
    let t2 = t0.wrapping_sub(t2);
    let t3 = t1.wrapping_sub(t3);

    let out1 = (1_i32 << 13).wrapping_add(t4a).wrapping_add(t6a) >> 14;
    let out1 = out1.wrapping_neg();
    let out6 = (1_i32 << 13).wrapping_add(t5a).wrapping_add(t7a) >> 14;
    let t6 = (1_i32 << 13).wrapping_add(t4a).wrapping_sub(t6a) >> 14;
    let t7 = (1_i32 << 13).wrapping_add(t5a).wrapping_sub(t7a) >> 14;

    let out3 = t2
        .wrapping_add(t3)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out3 = out3.wrapping_neg();
    let out4 = t2
        .wrapping_sub(t3)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out2 = t6
        .wrapping_add(t7)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out5 = t6
        .wrapping_sub(t7)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
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

    let t0a = i(0)
        .wrapping_add(i(8))
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t1a = i(0)
        .wrapping_sub(i(8))
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t2a = i(4)
        .wrapping_mul(6270)
        .wrapping_sub(i(12).wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t3a = i(4)
        .wrapping_mul(15137)
        .wrapping_add(i(12).wrapping_mul(6270))
        .wrapping_add(1 << 13)
        >> 14;
    let t4a = i(2)
        .wrapping_mul(3196)
        .wrapping_sub(i(14).wrapping_mul(16069))
        .wrapping_add(1 << 13)
        >> 14;
    let t7a = i(2)
        .wrapping_mul(16069)
        .wrapping_add(i(14).wrapping_mul(3196))
        .wrapping_add(1 << 13)
        >> 14;
    let t5a = i(10)
        .wrapping_mul(13623)
        .wrapping_sub(i(6).wrapping_mul(9102))
        .wrapping_add(1 << 13)
        >> 14;
    let t6a = i(10)
        .wrapping_mul(9102)
        .wrapping_add(i(6).wrapping_mul(13623))
        .wrapping_add(1 << 13)
        >> 14;
    let t8a = i(1)
        .wrapping_mul(1606)
        .wrapping_sub(i(15).wrapping_mul(16305))
        .wrapping_add(1 << 13)
        >> 14;
    let t15a = i(1)
        .wrapping_mul(16305)
        .wrapping_add(i(15).wrapping_mul(1606))
        .wrapping_add(1 << 13)
        >> 14;
    let t9a = i(9)
        .wrapping_mul(12665)
        .wrapping_sub(i(7).wrapping_mul(10394))
        .wrapping_add(1 << 13)
        >> 14;
    let t14a = i(9)
        .wrapping_mul(10394)
        .wrapping_add(i(7).wrapping_mul(12665))
        .wrapping_add(1 << 13)
        >> 14;
    let t10a = i(5)
        .wrapping_mul(7723)
        .wrapping_sub(i(11).wrapping_mul(14449))
        .wrapping_add(1 << 13)
        >> 14;
    let t13a = i(5)
        .wrapping_mul(14449)
        .wrapping_add(i(11).wrapping_mul(7723))
        .wrapping_add(1 << 13)
        >> 14;
    let t11a = i(13)
        .wrapping_mul(15679)
        .wrapping_sub(i(3).wrapping_mul(4756))
        .wrapping_add(1 << 13)
        >> 14;
    let t12a = i(13)
        .wrapping_mul(4756)
        .wrapping_add(i(3).wrapping_mul(15679))
        .wrapping_add(1 << 13)
        >> 14;

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

    let t5a = t6
        .wrapping_sub(t5)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t6a = t6
        .wrapping_add(t5)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t9a = t14
        .wrapping_mul(6270)
        .wrapping_sub(t9.wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t14a = t14
        .wrapping_mul(15137)
        .wrapping_add(t9.wrapping_mul(6270))
        .wrapping_add(1 << 13)
        >> 14;
    let t10a = t13
        .wrapping_mul(15137)
        .wrapping_add(t10.wrapping_mul(6270))
        .wrapping_neg()
        .wrapping_add(1 << 13)
        >> 14;
    let t13a = t13
        .wrapping_mul(6270)
        .wrapping_sub(t10.wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;

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

    let t10a = t13
        .wrapping_sub(t10)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t13a = t13
        .wrapping_add(t10)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t11 = t12a
        .wrapping_sub(t11a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t12 = t12a
        .wrapping_add(t11a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;

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

    let t0 = i(15)
        .wrapping_mul(16364)
        .wrapping_add(i(0).wrapping_mul(804));
    let t1 = i(15)
        .wrapping_mul(804)
        .wrapping_sub(i(0).wrapping_mul(16364));
    let t2 = i(13)
        .wrapping_mul(15893)
        .wrapping_add(i(2).wrapping_mul(3981));
    let t3 = i(13)
        .wrapping_mul(3981)
        .wrapping_sub(i(2).wrapping_mul(15893));
    let t4 = i(11)
        .wrapping_mul(14811)
        .wrapping_add(i(4).wrapping_mul(7005));
    let t5 = i(11)
        .wrapping_mul(7005)
        .wrapping_sub(i(4).wrapping_mul(14811));
    let t6 = i(9)
        .wrapping_mul(13160)
        .wrapping_add(i(6).wrapping_mul(9760));
    let t7 = i(9)
        .wrapping_mul(9760)
        .wrapping_sub(i(6).wrapping_mul(13160));
    let t8 = i(7)
        .wrapping_mul(11003)
        .wrapping_add(i(8).wrapping_mul(12140));
    let t9 = i(7)
        .wrapping_mul(12140)
        .wrapping_sub(i(8).wrapping_mul(11003));
    let t10 = i(5)
        .wrapping_mul(8423)
        .wrapping_add(i(10).wrapping_mul(14053));
    let t11 = i(5)
        .wrapping_mul(14053)
        .wrapping_sub(i(10).wrapping_mul(8423));
    let t12 = i(3)
        .wrapping_mul(5520)
        .wrapping_add(i(12).wrapping_mul(15426));
    let t13 = i(3)
        .wrapping_mul(15426)
        .wrapping_sub(i(12).wrapping_mul(5520));
    let t14 = i(1)
        .wrapping_mul(2404)
        .wrapping_add(i(14).wrapping_mul(16207));
    let t15 = i(1)
        .wrapping_mul(16207)
        .wrapping_sub(i(14).wrapping_mul(2404));

    let t0a = (1_i32 << 13).wrapping_add(t0).wrapping_add(t8) >> 14;
    let t1a = (1_i32 << 13).wrapping_add(t1).wrapping_add(t9) >> 14;
    let t2a = (1_i32 << 13).wrapping_add(t2).wrapping_add(t10) >> 14;
    let t3a = (1_i32 << 13).wrapping_add(t3).wrapping_add(t11) >> 14;
    let t4a = (1_i32 << 13).wrapping_add(t4).wrapping_add(t12) >> 14;
    let t5a = (1_i32 << 13).wrapping_add(t5).wrapping_add(t13) >> 14;
    let t6a = (1_i32 << 13).wrapping_add(t6).wrapping_add(t14) >> 14;
    let t7a = (1_i32 << 13).wrapping_add(t7).wrapping_add(t15) >> 14;
    let t8a = (1_i32 << 13).wrapping_add(t0).wrapping_sub(t8) >> 14;
    let t9a = (1_i32 << 13).wrapping_add(t1).wrapping_sub(t9) >> 14;
    let t10a = (1_i32 << 13).wrapping_add(t2).wrapping_sub(t10) >> 14;
    let t11a = (1_i32 << 13).wrapping_add(t3).wrapping_sub(t11) >> 14;
    let t12a = (1_i32 << 13).wrapping_add(t4).wrapping_sub(t12) >> 14;
    let t13a = (1_i32 << 13).wrapping_add(t5).wrapping_sub(t13) >> 14;
    let t14a = (1_i32 << 13).wrapping_add(t6).wrapping_sub(t14) >> 14;
    let t15a = (1_i32 << 13).wrapping_add(t7).wrapping_sub(t15) >> 14;

    let t8 = t8a.wrapping_mul(16069).wrapping_add(t9a.wrapping_mul(3196));
    let t9 = t8a.wrapping_mul(3196).wrapping_sub(t9a.wrapping_mul(16069));
    let t10 = t10a
        .wrapping_mul(9102)
        .wrapping_add(t11a.wrapping_mul(13623));
    let t11 = t10a
        .wrapping_mul(13623)
        .wrapping_sub(t11a.wrapping_mul(9102));
    let t12 = t13a
        .wrapping_mul(16069)
        .wrapping_sub(t12a.wrapping_mul(3196));
    let t13 = t13a
        .wrapping_mul(3196)
        .wrapping_add(t12a.wrapping_mul(16069));
    let t14 = t15a
        .wrapping_mul(9102)
        .wrapping_sub(t14a.wrapping_mul(13623));
    let t15 = t15a
        .wrapping_mul(13623)
        .wrapping_add(t14a.wrapping_mul(9102));

    let t0 = t0a.wrapping_add(t4a);
    let t1 = t1a.wrapping_add(t5a);
    let t2 = t2a.wrapping_add(t6a);
    let t3 = t3a.wrapping_add(t7a);
    let t4 = t0a.wrapping_sub(t4a);
    let t5 = t1a.wrapping_sub(t5a);
    let t6 = t2a.wrapping_sub(t6a);
    let t7 = t3a.wrapping_sub(t7a);
    let t8a = (1_i32 << 13).wrapping_add(t8).wrapping_add(t12) >> 14;
    let t9a = (1_i32 << 13).wrapping_add(t9).wrapping_add(t13) >> 14;
    let t10a = (1_i32 << 13).wrapping_add(t10).wrapping_add(t14) >> 14;
    let t11a = (1_i32 << 13).wrapping_add(t11).wrapping_add(t15) >> 14;
    let t12a = (1_i32 << 13).wrapping_add(t8).wrapping_sub(t12) >> 14;
    let t13a = (1_i32 << 13).wrapping_add(t9).wrapping_sub(t13) >> 14;
    let t14a = (1_i32 << 13).wrapping_add(t10).wrapping_sub(t14) >> 14;
    let t15a = (1_i32 << 13).wrapping_add(t11).wrapping_sub(t15) >> 14;

    let t4a = t4.wrapping_mul(15137).wrapping_add(t5.wrapping_mul(6270));
    let t5a = t4.wrapping_mul(6270).wrapping_sub(t5.wrapping_mul(15137));
    let t6a = t7.wrapping_mul(15137).wrapping_sub(t6.wrapping_mul(6270));
    let t7a = t7.wrapping_mul(6270).wrapping_add(t6.wrapping_mul(15137));
    let t12 = t12a
        .wrapping_mul(15137)
        .wrapping_add(t13a.wrapping_mul(6270));
    let t13 = t12a
        .wrapping_mul(6270)
        .wrapping_sub(t13a.wrapping_mul(15137));
    let t14 = t15a
        .wrapping_mul(15137)
        .wrapping_sub(t14a.wrapping_mul(6270));
    let t15 = t15a
        .wrapping_mul(6270)
        .wrapping_add(t14a.wrapping_mul(15137));

    let out0 = t0.wrapping_add(t2);
    let out15 = t1.wrapping_add(t3).wrapping_neg();
    let t2a = t0.wrapping_sub(t2);
    let t3a = t1.wrapping_sub(t3);

    let out3 = (1_i32 << 13).wrapping_add(t4a).wrapping_add(t6a) >> 14;
    let out3 = out3.wrapping_neg();
    let out12 = (1_i32 << 13).wrapping_add(t5a).wrapping_add(t7a) >> 14;
    let t6 = (1_i32 << 13).wrapping_add(t4a).wrapping_sub(t6a) >> 14;
    let t7 = (1_i32 << 13).wrapping_add(t5a).wrapping_sub(t7a) >> 14;

    let out1 = t8a.wrapping_add(t10a).wrapping_neg();
    let out14 = t9a.wrapping_add(t11a);
    let t10 = t8a.wrapping_sub(t10a);
    let t11 = t9a.wrapping_sub(t11a);

    let out2 = (1_i32 << 13).wrapping_add(t12).wrapping_add(t14) >> 14;
    let out13 = (1_i32 << 13).wrapping_add(t13).wrapping_add(t15) >> 14;
    let out13 = out13.wrapping_neg();
    let t14a = (1_i32 << 13).wrapping_add(t12).wrapping_sub(t14) >> 14;
    let t15a = (1_i32 << 13).wrapping_add(t13).wrapping_sub(t15) >> 14;

    let out7 = t2a
        .wrapping_add(t3a)
        .wrapping_neg()
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out8 = t2a
        .wrapping_sub(t3a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out4 = t7
        .wrapping_add(t6)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out11 = t7
        .wrapping_sub(t6)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out6 = t11
        .wrapping_add(t10)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out9 = t11
        .wrapping_sub(t10)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out5 = t14a
        .wrapping_add(t15a)
        .wrapping_neg()
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let out10 = t14a
        .wrapping_sub(t15a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;

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
    let t0a = i(0)
        .wrapping_add(i(16))
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t1a = i(0)
        .wrapping_sub(i(16))
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t2a = i(8)
        .wrapping_mul(6270)
        .wrapping_sub(i(24).wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t3a = i(8)
        .wrapping_mul(15137)
        .wrapping_add(i(24).wrapping_mul(6270))
        .wrapping_add(1 << 13)
        >> 14;
    let t4a = i(4)
        .wrapping_mul(3196)
        .wrapping_sub(i(28).wrapping_mul(16069))
        .wrapping_add(1 << 13)
        >> 14;
    let t7a = i(4)
        .wrapping_mul(16069)
        .wrapping_add(i(28).wrapping_mul(3196))
        .wrapping_add(1 << 13)
        >> 14;
    let t5a = i(20)
        .wrapping_mul(13623)
        .wrapping_sub(i(12).wrapping_mul(9102))
        .wrapping_add(1 << 13)
        >> 14;
    let t6a = i(20)
        .wrapping_mul(9102)
        .wrapping_add(i(12).wrapping_mul(13623))
        .wrapping_add(1 << 13)
        >> 14;
    let t8a = i(2)
        .wrapping_mul(1606)
        .wrapping_sub(i(30).wrapping_mul(16305))
        .wrapping_add(1 << 13)
        >> 14;
    let t15a = i(2)
        .wrapping_mul(16305)
        .wrapping_add(i(30).wrapping_mul(1606))
        .wrapping_add(1 << 13)
        >> 14;
    let t9a = i(18)
        .wrapping_mul(12665)
        .wrapping_sub(i(14).wrapping_mul(10394))
        .wrapping_add(1 << 13)
        >> 14;
    let t14a = i(18)
        .wrapping_mul(10394)
        .wrapping_add(i(14).wrapping_mul(12665))
        .wrapping_add(1 << 13)
        >> 14;
    let t10a = i(10)
        .wrapping_mul(7723)
        .wrapping_sub(i(22).wrapping_mul(14449))
        .wrapping_add(1 << 13)
        >> 14;
    let t13a = i(10)
        .wrapping_mul(14449)
        .wrapping_add(i(22).wrapping_mul(7723))
        .wrapping_add(1 << 13)
        >> 14;
    let t11a = i(26)
        .wrapping_mul(15679)
        .wrapping_sub(i(6).wrapping_mul(4756))
        .wrapping_add(1 << 13)
        >> 14;
    let t12a = i(26)
        .wrapping_mul(4756)
        .wrapping_add(i(6).wrapping_mul(15679))
        .wrapping_add(1 << 13)
        >> 14;
    let t16a = i(1)
        .wrapping_mul(804)
        .wrapping_sub(i(31).wrapping_mul(16364))
        .wrapping_add(1 << 13)
        >> 14;
    let t31a = i(1)
        .wrapping_mul(16364)
        .wrapping_add(i(31).wrapping_mul(804))
        .wrapping_add(1 << 13)
        >> 14;
    let t17a = i(17)
        .wrapping_mul(12140)
        .wrapping_sub(i(15).wrapping_mul(11003))
        .wrapping_add(1 << 13)
        >> 14;
    let t30a = i(17)
        .wrapping_mul(11003)
        .wrapping_add(i(15).wrapping_mul(12140))
        .wrapping_add(1 << 13)
        >> 14;
    let t18a = i(9)
        .wrapping_mul(7005)
        .wrapping_sub(i(23).wrapping_mul(14811))
        .wrapping_add(1 << 13)
        >> 14;
    let t29a = i(9)
        .wrapping_mul(14811)
        .wrapping_add(i(23).wrapping_mul(7005))
        .wrapping_add(1 << 13)
        >> 14;
    let t19a = i(25)
        .wrapping_mul(15426)
        .wrapping_sub(i(7).wrapping_mul(5520))
        .wrapping_add(1 << 13)
        >> 14;
    let t28a = i(25)
        .wrapping_mul(5520)
        .wrapping_add(i(7).wrapping_mul(15426))
        .wrapping_add(1 << 13)
        >> 14;
    let t20a = i(5)
        .wrapping_mul(3981)
        .wrapping_sub(i(27).wrapping_mul(15893))
        .wrapping_add(1 << 13)
        >> 14;
    let t27a = i(5)
        .wrapping_mul(15893)
        .wrapping_add(i(27).wrapping_mul(3981))
        .wrapping_add(1 << 13)
        >> 14;
    let t21a = i(21)
        .wrapping_mul(14053)
        .wrapping_sub(i(11).wrapping_mul(8423))
        .wrapping_add(1 << 13)
        >> 14;
    let t26a = i(21)
        .wrapping_mul(8423)
        .wrapping_add(i(11).wrapping_mul(14053))
        .wrapping_add(1 << 13)
        >> 14;
    let t22a = i(13)
        .wrapping_mul(9760)
        .wrapping_sub(i(19).wrapping_mul(13160))
        .wrapping_add(1 << 13)
        >> 14;
    let t25a = i(13)
        .wrapping_mul(13160)
        .wrapping_add(i(19).wrapping_mul(9760))
        .wrapping_add(1 << 13)
        >> 14;
    let t23a = i(29)
        .wrapping_mul(16207)
        .wrapping_sub(i(3).wrapping_mul(2404))
        .wrapping_add(1 << 13)
        >> 14;
    let t24a = i(29)
        .wrapping_mul(2404)
        .wrapping_add(i(3).wrapping_mul(16207))
        .wrapping_add(1 << 13)
        >> 14;

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
    let t5a = t6
        .wrapping_sub(t5)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t6a = t6
        .wrapping_add(t5)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t9a = t14
        .wrapping_mul(6270)
        .wrapping_sub(t9.wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t14a = t14
        .wrapping_mul(15137)
        .wrapping_add(t9.wrapping_mul(6270))
        .wrapping_add(1 << 13)
        >> 14;
    let t10a = t13
        .wrapping_mul(15137)
        .wrapping_add(t10.wrapping_mul(6270))
        .wrapping_neg()
        .wrapping_add(1 << 13)
        >> 14;
    let t13a = t13
        .wrapping_mul(6270)
        .wrapping_sub(t10.wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t17a = t30
        .wrapping_mul(3196)
        .wrapping_sub(t17.wrapping_mul(16069))
        .wrapping_add(1 << 13)
        >> 14;
    let t30a = t30
        .wrapping_mul(16069)
        .wrapping_add(t17.wrapping_mul(3196))
        .wrapping_add(1 << 13)
        >> 14;
    let t18a = t29
        .wrapping_mul(16069)
        .wrapping_add(t18.wrapping_mul(3196))
        .wrapping_neg()
        .wrapping_add(1 << 13)
        >> 14;
    let t29a = t29
        .wrapping_mul(3196)
        .wrapping_sub(t18.wrapping_mul(16069))
        .wrapping_add(1 << 13)
        >> 14;
    let t21a = t26
        .wrapping_mul(13623)
        .wrapping_sub(t21.wrapping_mul(9102))
        .wrapping_add(1 << 13)
        >> 14;
    let t26a = t26
        .wrapping_mul(9102)
        .wrapping_add(t21.wrapping_mul(13623))
        .wrapping_add(1 << 13)
        >> 14;
    let t22a = t25
        .wrapping_mul(9102)
        .wrapping_add(t22.wrapping_mul(13623))
        .wrapping_neg()
        .wrapping_add(1 << 13)
        >> 14;
    let t25a = t25
        .wrapping_mul(13623)
        .wrapping_sub(t22.wrapping_mul(9102))
        .wrapping_add(1 << 13)
        >> 14;

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
    let t10a = t13
        .wrapping_sub(t10)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t13a = t13
        .wrapping_add(t10)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t11 = t12a
        .wrapping_sub(t11a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t12 = t12a
        .wrapping_add(t11a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t18a = t29
        .wrapping_mul(6270)
        .wrapping_sub(t18.wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t29a = t29
        .wrapping_mul(15137)
        .wrapping_add(t18.wrapping_mul(6270))
        .wrapping_add(1 << 13)
        >> 14;
    let t19 = t28a
        .wrapping_mul(6270)
        .wrapping_sub(t19a.wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t28 = t28a
        .wrapping_mul(15137)
        .wrapping_add(t19a.wrapping_mul(6270))
        .wrapping_add(1 << 13)
        >> 14;
    let t20 = t27a
        .wrapping_mul(15137)
        .wrapping_add(t20a.wrapping_mul(6270))
        .wrapping_neg()
        .wrapping_add(1 << 13)
        >> 14;
    let t27 = t27a
        .wrapping_mul(6270)
        .wrapping_sub(t20a.wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;
    let t21a = t26
        .wrapping_mul(15137)
        .wrapping_add(t21.wrapping_mul(6270))
        .wrapping_neg()
        .wrapping_add(1 << 13)
        >> 14;
    let t26a = t26
        .wrapping_mul(6270)
        .wrapping_sub(t21.wrapping_mul(15137))
        .wrapping_add(1 << 13)
        >> 14;

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
    let t20 = t27a
        .wrapping_sub(t20a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t27 = t27a
        .wrapping_add(t20a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t21a = t26
        .wrapping_sub(t21)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t26a = t26
        .wrapping_add(t21)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t22 = t25a
        .wrapping_sub(t22a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t25 = t25a
        .wrapping_add(t22a)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t23a = t24
        .wrapping_sub(t23)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;
    let t24a = t24
        .wrapping_add(t23)
        .wrapping_mul(11585)
        .wrapping_add(1 << 13)
        >> 14;

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
        TxSize::Tx4x4 => itxfm_add_nxn::<4>(dst, stride, coefs, tx_type, max_val, eob, 4),
        TxSize::Tx8x8 => itxfm_add_nxn::<8>(dst, stride, coefs, tx_type, max_val, eob, 5),
        TxSize::Tx16x16 => itxfm_add_nxn::<16>(dst, stride, coefs, tx_type, max_val, eob, 6),
        TxSize::Tx32x32 => {
            // 32×32 only supports DCT_DCT (ADST is not used at this size).
            itxfm_add_32x32(dst, stride, coefs, max_val, eob)
        }
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

/// Generic 2-D itxfm + add for sizes 4, 8, 16.
///
/// `N`     — size (const generic).
/// `bits`  — right-shift amount after column pass (4→4, 8→5, 16→6).
fn itxfm_add_nxn<const N: usize>(
    dst: &mut [u8],
    stride: usize,
    coefs: &[i32],
    tx_type: TxType,
    max_val: i32,
    eob: usize,
    bits: u32,
) {
    // DC-only fast path: only valid for DCT_DCT (has_dconly=1 in FFmpeg).
    // ADST transforms don't produce uniform output from a single DC coefficient.
    if eob == 1 && tx_type == TxType::DctDct {
        let dc = coefs[0];
        // Two 1-D stages both apply the same 1/sqrt(2) scaling:
        // ((dc * 11585 + (1<<13)) >> 14) * 11585 + (1<<13)) >> 14
        let t = dc.wrapping_mul(11585).wrapping_add(1 << 13) >> 14;
        let t = t.wrapping_mul(11585).wrapping_add(1 << 13) >> 14;
        let add = if bits > 0 {
            t.wrapping_add(1i32 << (bits - 1)) >> bits
        } else {
            t
        };
        for row in 0..N {
            for col in 0..N {
                let p = dst[row * stride + col] as i32;
                dst[row * stride + col] = clip_pixel(p + add, max_val);
            }
        }
        return;
    }

    // Full 2-D path, matching FFmpeg's itxfm_wrapper column-first order.
    // FFmpeg's tmp array is dctcoef (int16_t for 8bpp), so pass-1 outputs are
    // truncated to 16 bits before being fed into pass 2.
    let mut tmp = vec![0i32; N * N];

    // Pass 1 — column transform: for each column i of coefs, apply the
    // column transform (type_a in FFmpeg) and store contiguously in tmp.
    // FFmpeg: type_a##_1d(block + i, sz, tmp + i * sz, 0)
    for col in 0..N {
        let col_in: Vec<i32> = (0..N).map(|row| coefs[row * N + col]).collect();
        let mut col_out = vec![0i32; N];
        apply_col_txfm(N, tx_type, &col_in, &mut col_out);
        // Truncate to 16 bits to match FFmpeg's int16_t intermediate storage.
        for v in &mut col_out {
            *v = *v as i16 as i32;
        }
        // tmp[col * N + row] = col_out[row]
        tmp[col * N..col * N + N].copy_from_slice(&col_out);
    }

    // Pass 2 — row transform: for each column i of tmp (= row of transposed
    // intermediate), apply the row transform (type_b in FFmpeg), add to dst.
    // FFmpeg: type_b##_1d(tmp + i, sz, out, 1); dst[j*stride] += ...
    // FFmpeg's `out` array is also dctcoef (int16_t for 8bpp).
    for col in 0..N {
        let col_in: Vec<i32> = (0..N).map(|row| tmp[row * N + col]).collect();
        let mut col_out = vec![0i32; N];
        apply_row_txfm(N, tx_type, &col_in, &mut col_out);
        for row in 0..N {
            let p = dst[row * stride + col] as i32;
            // Truncate to 16 bits to match FFmpeg's int16_t `out` array,
            // then apply the rounding shift via unsigned addition (matching
            // FFmpeg's `(int)(out[j] + (1U << (bits-1))) >> bits`).
            let v = col_out[row] as i16 as i32;
            let add = if bits > 0 {
                (v.wrapping_add(1i32 << (bits - 1))) >> bits
            } else {
                v
            };
            dst[row * stride + col] = clip_pixel(p + add, max_val);
        }
    }
}

/// Apply the row (first) transform for the given tx_type.
fn apply_row_txfm(n: usize, tx_type: TxType, input: &[i32], output: &mut [i32]) {
    // VP9 TxType naming: {ROW}_{COL}, e.g. ADST_DCT = row=ADST, col=DCT.
    // FFmpeg init_itxfm maps:
    //   DCT_DCT   → idct_idct   (col=idct,  row=idct)
    //   DCT_ADST  → iadst_idct  (col=iadst, row=idct)
    //   ADST_DCT  → idct_iadst  (col=idct,  row=iadst)
    //   ADST_ADST → iadst_iadst (col=iadst, row=iadst)
    // So row uses ADST when the first part of the name is ADST.
    let use_adst = matches!(tx_type, TxType::AdstDct | TxType::AdstAdst);
    apply_1d_txfm(n, use_adst, input, output);
}

/// Apply the column (second) transform for the given tx_type.
fn apply_col_txfm(n: usize, tx_type: TxType, input: &[i32], output: &mut [i32]) {
    // Column uses ADST when the second part of the name is ADST.
    let use_adst = matches!(tx_type, TxType::DctAdst | TxType::AdstAdst);
    apply_1d_txfm(n, use_adst, input, output);
}

/// Dispatch to the appropriate 1-D IDCT or IADST.
fn apply_1d_txfm(n: usize, use_adst: bool, input: &[i32], output: &mut [i32]) {
    match n {
        4 => {
            let inp: &[i32; 4] = input.try_into().expect("slice length 4");
            let out: &mut [i32; 4] = output.try_into().expect("slice length 4");
            if use_adst {
                iadst4(inp, out);
            } else {
                idct4(inp, out);
            }
        }
        8 => {
            let inp: &[i32; 8] = input.try_into().expect("slice length 8");
            let out: &mut [i32; 8] = output.try_into().expect("slice length 8");
            if use_adst {
                iadst8(inp, out);
            } else {
                idct8(inp, out);
            }
        }
        16 => {
            let inp: &[i32; 16] = input.try_into().expect("slice length 16");
            let out: &mut [i32; 16] = output.try_into().expect("slice length 16");
            if use_adst {
                iadst16(inp, out);
            } else {
                idct16(inp, out);
            }
        }
        _ => panic!("unsupported 1-D transform size {n}"),
    }
}

/// 2-D IDCT + add for the 32×32 case (DCT only, bits=6).
fn itxfm_add_32x32(dst: &mut [u8], stride: usize, coefs: &[i32], max_val: i32, eob: usize) {
    const N: usize = 32;
    const BITS: u32 = 6;

    // DC-only fast path.
    if eob == 1 {
        let dc = coefs[0];
        let t = dc.wrapping_mul(11585).wrapping_add(1 << 13) >> 14;
        let t = t.wrapping_mul(11585).wrapping_add(1 << 13) >> 14;
        let add = t.wrapping_add(1i32 << (BITS - 1)) >> BITS;
        for row in 0..N {
            for col in 0..N {
                let p = dst[row * stride + col] as i32;
                dst[row * stride + col] = clip_pixel(p + add, max_val);
            }
        }
        return;
    }

    let mut tmp = vec![0i32; N * N];

    // Pass 1 — column transform (matching FFmpeg's column-first order).
    for col in 0..N {
        let col_in: [i32; 32] = core::array::from_fn(|row| coefs[row * N + col]);
        let mut col_out = [0i32; 32];
        idct32(&col_in, &mut col_out);
        // Truncate to 16 bits to match FFmpeg's int16_t intermediate storage.
        for v in &mut col_out {
            *v = *v as i16 as i32;
        }
        tmp[col * N..col * N + N].copy_from_slice(&col_out);
    }

    // Pass 2 — row transform + add.
    for col in 0..N {
        let col_in: [i32; 32] = core::array::from_fn(|row| tmp[row * N + col]);
        let mut col_out = [0i32; 32];
        idct32(&col_in, &mut col_out);
        for row in 0..N {
            let p = dst[row * stride + col] as i32;
            // Truncate to 16 bits to match FFmpeg's int16_t `out` array.
            let v = col_out[row] as i16 as i32;
            let add = v.wrapping_add(1i32 << (BITS - 1)) >> BITS;
            dst[row * stride + col] = clip_pixel(p + add, max_val);
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
