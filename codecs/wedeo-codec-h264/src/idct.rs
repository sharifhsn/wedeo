// H.264 IDCT (Inverse Discrete Cosine Transform) and related transforms.
//
// Implements the 4x4 integer IDCT, DC-only shortcut, and inverse Hadamard
// transforms for Intra16x16 luma DC and chroma DC coefficients.
//
// Reference: ITU-T H.264 Section 8.5.12, FFmpeg libavcodec/h264idct_template.c

/// 4x4 integer IDCT: transform coefficients and add to destination.
///
/// Applies the H.264 4x4 integer inverse transform to `coeffs` and adds the
/// result to the prediction pixels in `dst`. The output pixels are clamped to
/// [0, 255].
///
/// The transform uses integer-only butterflies matching FFmpeg's
/// `ff_h264_idct_add` exactly:
///   - First pass: column butterflies (iterates over columns, mixes rows)
///   - Second pass: row butterflies, add to dst with `(result + 32) >> 6`
///
/// Coefficients are stored in row-major order: `coeffs[y*4+x]`.
///
/// Reference: FFmpeg `ff_h264_idct_add` in h264idct_template.c
#[allow(unreachable_code)]
pub fn idct4x4_add(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 16]) {
    #[cfg(has_asm)]
    {
        crate::asm_dispatch::idct4x4_add_asm(dst, stride, coeffs);
        return;
    }

    // Add rounding bias to DC coefficient (same as FFmpeg's block[0] += 1 << 5)
    coeffs[0] = coeffs[0].wrapping_add(32);

    // FFmpeg stores coefficients in COLUMN-MAJOR (transposed) order and applies
    // the same loop structure. With our ROW-MAJOR coefficients, we must swap the
    // pass order to match FFmpeg's effective computation:
    //   FFmpeg: row-transform first, column-transform second (on transposed data)
    //   Wedeo:  row-transform first, column-transform second (on row-major data)
    //
    // The >>1 truncation in the butterfly makes pass order significant.
    //
    // Reference: FFmpeg `ff_h264_idct_add` in h264idct_template.c, noting that
    // FFmpeg's block layout is transposed via h264_slice.c:757 TRANSPOSE().

    // First pass: row transform (iterates over rows, mixes columns within each row)
    for i in 0..4 {
        let row = 4 * i;
        let z0 = coeffs[row] as i32 + coeffs[row + 2] as i32;
        let z1 = coeffs[row] as i32 - coeffs[row + 2] as i32;
        let z2 = (coeffs[row + 1] as i32 >> 1) - coeffs[row + 3] as i32;
        let z3 = coeffs[row + 1] as i32 + (coeffs[row + 3] as i32 >> 1);

        coeffs[row] = (z0 + z3) as i16;
        coeffs[row + 1] = (z1 + z2) as i16;
        coeffs[row + 2] = (z1 - z2) as i16;
        coeffs[row + 3] = (z0 - z3) as i16;
    }

    // Second pass: column transform and add to destination
    for i in 0..4 {
        let z0 = coeffs[i] as i32 + coeffs[i + 8] as i32;
        let z1 = coeffs[i] as i32 - coeffs[i + 8] as i32;
        let z2 = (coeffs[i + 4] as i32 >> 1) - coeffs[i + 12] as i32;
        let z3 = coeffs[i + 4] as i32 + (coeffs[i + 12] as i32 >> 1);

        dst[i] = (dst[i] as i32 + ((z0 + z3) >> 6)).clamp(0, 255) as u8;
        dst[stride + i] = (dst[stride + i] as i32 + ((z1 + z2) >> 6)).clamp(0, 255) as u8;
        dst[2 * stride + i] = (dst[2 * stride + i] as i32 + ((z1 - z2) >> 6)).clamp(0, 255) as u8;
        dst[3 * stride + i] = (dst[3 * stride + i] as i32 + ((z0 - z3) >> 6)).clamp(0, 255) as u8;
    }

    // Zero the coefficient block (FFmpeg: memset(block, 0, 16 * sizeof(dctcoef)))
    *coeffs = [0i16; 16];
}

/// DC-only 4x4 IDCT shortcut: add `(dc + 32) >> 6` to all 16 pixels.
///
/// When only the DC coefficient is non-zero, the full IDCT simplifies to
/// adding a constant value to every pixel. This is a common case and avoids
/// the full butterfly computation.
///
/// Reference: FFmpeg `ff_h264_idct_dc_add` in h264idct_template.c
#[allow(unreachable_code)]
pub fn idct4x4_dc_add(dst: &mut [u8], stride: usize, dc: &mut i16) {
    #[cfg(has_asm)]
    {
        crate::asm_dispatch::idct4x4_dc_add_asm(dst, stride, dc);
        return;
    }

    let val = (*dc as i32 + 32) >> 6;
    *dc = 0;

    for j in 0..4 {
        for i in 0..4 {
            dst[j * stride + i] = (dst[j * stride + i] as i32 + val).clamp(0, 255) as u8;
        }
    }
}

/// 8x8 integer IDCT: transform coefficients and add to destination.
///
/// The H.264 8x8 IDCT for High profile. Uses a more complex butterfly than
/// the 4x4 version.
///
/// FFmpeg stores coefficients in COLUMN-MAJOR (transposed) order and its IDCT
/// processes columns first (which are actually rows in transposed data), then
/// rows (which are actually columns). With wedeo's ROW-MAJOR coefficients, we
/// swap pass order: row-first, column-second. The `>>1`/`>>2` truncations in
/// the butterfly make pass order significant for bitexact results, so this
/// swap is required (same approach as the 4x4 IDCT).
///
/// Reference: FFmpeg `ff_h264_idct8_add` in h264idct_template.c
#[allow(unreachable_code)]
pub fn idct8x8_add(dst: &mut [u8], stride: usize, coeffs: &mut [i16; 64]) {
    #[cfg(has_asm)]
    {
        crate::asm_dispatch::idct8x8_add_asm(dst, stride, coeffs);
        return;
    }

    // Add rounding bias
    coeffs[0] = coeffs[0].wrapping_add(32);

    // FFmpeg stores coefficients in COLUMN-MAJOR (transposed) order and applies
    // the same loop structure. With our ROW-MAJOR coefficients, we must swap the
    // pass order to match FFmpeg's effective computation:
    //   FFmpeg: row-transform first, column-transform second (on transposed data)
    //   Wedeo:  row-transform first, column-transform second (on row-major data)
    //
    // The >>1/>>2 truncation in the butterfly makes pass order significant.
    //
    // Reference: FFmpeg `ff_h264_idct8_add` in h264idct_template.c, noting that
    // FFmpeg's block layout is transposed.

    // First pass: row transform (iterates over rows, mixes columns within each row)
    for i in 0..8 {
        let row = i * 8;
        let a0 = coeffs[row] as i32 + coeffs[row + 4] as i32;
        let a2 = coeffs[row] as i32 - coeffs[row + 4] as i32;
        let a4 = (coeffs[row + 2] as i32 >> 1) - coeffs[row + 6] as i32;
        let a6 = (coeffs[row + 6] as i32 >> 1) + coeffs[row + 2] as i32;

        let b0 = a0 + a6;
        let b2 = a2 + a4;
        let b4 = a2 - a4;
        let b6 = a0 - a6;

        let a1 = -(coeffs[row + 3] as i32) + coeffs[row + 5] as i32
            - coeffs[row + 7] as i32
            - (coeffs[row + 7] as i32 >> 1);
        let a3 = coeffs[row + 1] as i32 + coeffs[row + 7] as i32
            - coeffs[row + 3] as i32
            - (coeffs[row + 3] as i32 >> 1);
        let a5 = -(coeffs[row + 1] as i32)
            + coeffs[row + 7] as i32
            + coeffs[row + 5] as i32
            + (coeffs[row + 5] as i32 >> 1);
        let a7 = coeffs[row + 3] as i32
            + coeffs[row + 5] as i32
            + coeffs[row + 1] as i32
            + (coeffs[row + 1] as i32 >> 1);

        let b1 = (a7 >> 2) + a1;
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);

        coeffs[row] = (b0 + b7) as i16;
        coeffs[row + 7] = (b0 - b7) as i16;
        coeffs[row + 1] = (b2 + b5) as i16;
        coeffs[row + 6] = (b2 - b5) as i16;
        coeffs[row + 2] = (b4 + b3) as i16;
        coeffs[row + 5] = (b4 - b3) as i16;
        coeffs[row + 3] = (b6 + b1) as i16;
        coeffs[row + 4] = (b6 - b1) as i16;
    }

    // Second pass: column transform and add to destination
    for i in 0..8 {
        let a0 = coeffs[i] as i32 + coeffs[i + 32] as i32;
        let a2 = coeffs[i] as i32 - coeffs[i + 32] as i32;
        let a4 = (coeffs[i + 16] as i32 >> 1) - coeffs[i + 48] as i32;
        let a6 = (coeffs[i + 48] as i32 >> 1) + coeffs[i + 16] as i32;

        let b0 = a0 + a6;
        let b2 = a2 + a4;
        let b4 = a2 - a4;
        let b6 = a0 - a6;

        let a1 = -(coeffs[i + 24] as i32) + coeffs[i + 40] as i32
            - coeffs[i + 56] as i32
            - (coeffs[i + 56] as i32 >> 1);
        let a3 = coeffs[i + 8] as i32 + coeffs[i + 56] as i32
            - coeffs[i + 24] as i32
            - (coeffs[i + 24] as i32 >> 1);
        let a5 = -(coeffs[i + 8] as i32)
            + coeffs[i + 56] as i32
            + coeffs[i + 40] as i32
            + (coeffs[i + 40] as i32 >> 1);
        let a7 = coeffs[i + 24] as i32
            + coeffs[i + 40] as i32
            + coeffs[i + 8] as i32
            + (coeffs[i + 8] as i32 >> 1);

        let b1 = (a7 >> 2) + a1;
        let b3 = a3 + (a5 >> 2);
        let b5 = (a3 >> 2) - a5;
        let b7 = a7 - (a1 >> 2);

        dst[i] = (dst[i] as i32 + ((b0 + b7) >> 6)).clamp(0, 255) as u8;
        dst[stride + i] = (dst[stride + i] as i32 + ((b2 + b5) >> 6)).clamp(0, 255) as u8;
        dst[2 * stride + i] = (dst[2 * stride + i] as i32 + ((b4 + b3) >> 6)).clamp(0, 255) as u8;
        dst[3 * stride + i] = (dst[3 * stride + i] as i32 + ((b6 + b1) >> 6)).clamp(0, 255) as u8;
        dst[4 * stride + i] = (dst[4 * stride + i] as i32 + ((b6 - b1) >> 6)).clamp(0, 255) as u8;
        dst[5 * stride + i] = (dst[5 * stride + i] as i32 + ((b4 - b3) >> 6)).clamp(0, 255) as u8;
        dst[6 * stride + i] = (dst[6 * stride + i] as i32 + ((b2 - b5) >> 6)).clamp(0, 255) as u8;
        dst[7 * stride + i] = (dst[7 * stride + i] as i32 + ((b0 - b7) >> 6)).clamp(0, 255) as u8;
    }

    *coeffs = [0i16; 64];
}

/// DC-only 8x8 IDCT shortcut: add `(dc + 32) >> 6` to all 64 pixels.
///
/// Reference: FFmpeg `ff_h264_idct8_dc_add` in h264idct_template.c
#[allow(unreachable_code)]
pub fn idct8x8_dc_add(dst: &mut [u8], stride: usize, dc: &mut i16) {
    #[cfg(has_asm)]
    {
        crate::asm_dispatch::idct8x8_dc_add_asm(dst, stride, dc);
        return;
    }

    let val = (*dc as i32 + 32) >> 6;
    *dc = 0;

    for j in 0..8 {
        for i in 0..8 {
            dst[j * stride + i] = (dst[j * stride + i] as i32 + val).clamp(0, 255) as u8;
        }
    }
}

/// 4x4 inverse Hadamard transform with dequantization for Intra16x16 luma DC.
///
/// In Intra16x16 mode, the DC coefficients of all 16 luma 4x4 blocks are
/// coded together. This function applies the inverse Hadamard transform and
/// dequantizes them in one step.
///
/// `output` receives the 16 transformed+dequantized DC values. In FFmpeg these
/// are scattered into the block array at specific offsets; here they are stored
/// contiguously in raster order (block 0..15).
///
/// `input` contains the 16 DC coefficients from the bitstream.
///
/// `qmul` is the combined dequantization scale factor, pre-computed as:
///   `dequant4_coeff[qp][0]` (the DC position scale from the dequant table).
///
/// The transform:
///   1. Horizontal Hadamard on each row of the 4x4 DC block
///   2. Vertical Hadamard on each column
///   3. Scale: `result = (val * qmul + 128) >> 8`
///
/// Reference: FFmpeg `ff_h264_luma_dc_dequant_idct` in h264idct_template.c
pub fn luma_dc_dequant_idct(output: &mut [i32; 16], input: &[i16; 16], qmul: i32) {
    let mut temp = [0i32; 16];

    // Horizontal Hadamard on each row
    for i in 0..4 {
        let base = 4 * i;
        let z0 = input[base] as i32 + input[base + 1] as i32;
        let z1 = input[base] as i32 - input[base + 1] as i32;
        let z2 = input[base + 2] as i32 - input[base + 3] as i32;
        let z3 = input[base + 2] as i32 + input[base + 3] as i32;

        temp[base] = z0 + z3;
        temp[base + 1] = z0 - z3;
        temp[base + 2] = z1 - z2;
        temp[base + 3] = z1 + z2;
    }

    // Vertical Hadamard on each column, then dequantize.
    //
    // FFmpeg scatters results to output[stride*row + x_offset[col]] where
    // x_offset = {0, 2*stride, 8*stride, 10*stride} with stride=16.
    // That places the 16 DC values at the [0] position of each of the 16
    // 4x4 blocks in the macroblock's block array.
    //
    // We store them contiguously in raster scan order (block index 0..15),
    // mapping (row, col) to the block index via the standard 4x4 block layout:
    //   block_idx = block_raster_order[row][col]
    //
    // FFmpeg's x_offset layout maps column j to block columns {0,1,2,3}
    // and stride*row maps to block rows. The output indices are:
    //   (row=0,col=0) -> block 0   (row=0,col=1) -> block 2
    //   (row=0,col=2) -> block 8   (row=0,col=3) -> block 10
    //   etc. (in units of the stride=16 scattered layout)
    //
    // For our contiguous output, we use a mapping table that matches
    // FFmpeg's scatter pattern.
    const BLOCK_MAP: [usize; 16] = [
        // (row, col) -> output index, laid out as [row*4 + col]
        // row 0: cols 0,1,2,3
        0, 1, 4, 5, // row 1: cols 0,1,2,3
        2, 3, 6, 7, // row 2: cols 0,1,2,3
        8, 9, 12, 13, // row 3: cols 0,1,2,3
        10, 11, 14, 15,
    ];

    for j in 0..4 {
        let z0 = temp[j] + temp[8 + j];
        let z1 = temp[j] - temp[8 + j];
        let z2 = temp[4 + j] - temp[12 + j];
        let z3 = temp[4 + j] + temp[12 + j];

        output[BLOCK_MAP[j]] = ((z0 + z3) * qmul + 128) >> 8;
        output[BLOCK_MAP[4 + j]] = ((z1 + z2) * qmul + 128) >> 8;
        output[BLOCK_MAP[8 + j]] = ((z1 - z2) * qmul + 128) >> 8;
        output[BLOCK_MAP[12 + j]] = ((z0 - z3) * qmul + 128) >> 8;
    }
}

/// 2x2 inverse Hadamard transform with dequantization for chroma DC (4:2:0).
///
/// For 4:2:0 chroma, each chroma plane has a 2x2 block of DC coefficients.
/// This function applies the inverse Hadamard and dequantizes.
///
/// `coeffs` contains the 4 DC coefficients and receives the results.
///
/// `qmul` is the combined dequantization scale factor.
///
/// The 2x2 Hadamard:
///   a = c[0] + c[2],  b = c[0] - c[2]
///   c = c[1] + c[3],  d = c[1] - c[3]
///   result[0] = ((a + c) * qmul) >> 7
///   result[1] = ((b + d) * qmul) >> 7
///   result[2] = ((a - c) * qmul) >> 7
///   result[3] = ((b - d) * qmul) >> 7
///
/// Note: FFmpeg's layout uses stride=32, xStride=16 for the 2x2 block in the
/// macroblock block array. Our contiguous layout stores them as [0..3].
///
/// Reference: FFmpeg `ff_h264_chroma_dc_dequant_idct` in h264idct_template.c
pub fn chroma_dc_dequant_idct(output: &mut [i32; 4], coeffs: &[i16; 4], qmul: i32) {
    let a = coeffs[0] as i32;
    let b = coeffs[1] as i32;
    let c = coeffs[2] as i32;
    let d = coeffs[3] as i32;

    let e = a - b;
    let a = a + b;
    let b = c - d;
    let c = c + d;

    output[0] = ((a + c) * qmul) >> 7;
    output[1] = ((e + b) * qmul) >> 7;
    output[2] = ((a - c) * qmul) >> 7;
    output[3] = ((e - b) * qmul) >> 7;
}

/// 2x4 inverse Hadamard transform with dequantization for chroma DC (4:2:2).
///
/// For 4:2:2 chroma, each chroma plane has a 2x4 block of DC coefficients
/// (2 columns, 4 rows).
///
/// `coeffs` contains the 8 DC coefficients and receives the results.
///
/// `qmul` is the combined dequantization scale factor.
///
/// Reference: FFmpeg `ff_h264_chroma422_dc_dequant_idct` in h264idct_template.c
pub fn chroma422_dc_dequant_idct(output: &mut [i32; 8], coeffs: &[i16; 8], qmul: i32) {
    let mut temp = [0i32; 8];

    // Horizontal pass: 2-point Hadamard on each row
    for i in 0..4 {
        let base = 2 * i;
        temp[base] = coeffs[base] as i32 + coeffs[base + 1] as i32;
        temp[base + 1] = coeffs[base] as i32 - coeffs[base + 1] as i32;
    }

    // Vertical pass: 4-point Hadamard on each column, then dequantize
    for i in 0..2 {
        let z0 = temp[i] + temp[4 + i];
        let z1 = temp[i] - temp[4 + i];
        let z2 = temp[2 + i] - temp[6 + i];
        let z3 = temp[2 + i] + temp[6 + i];

        output[i] = ((z0 + z3) * qmul + 128) >> 8;
        output[2 + i] = ((z1 + z2) * qmul + 128) >> 8;
        output[4 + i] = ((z1 - z2) * qmul + 128) >> 8;
        output[6 + i] = ((z0 - z3) * qmul + 128) >> 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct4x4_dc_only_input() {
        // DC-only: coeffs[0] = 128, all others zero.
        // The IDCT should add (128 + 32) >> 6 = 2 to every pixel.
        let mut dst = [100u8; 64];
        let stride = 8;
        let mut coeffs = [0i16; 16];
        coeffs[0] = 128;

        idct4x4_add(&mut dst, stride, &mut coeffs);

        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(dst[j * stride + i], 102, "pixel ({i},{j})");
            }
        }
        // Pixels outside the 4x4 block should be untouched
        assert_eq!(dst[4], 100);
        assert_eq!(dst[stride + 4], 100);
        // Coefficients should be zeroed
        assert_eq!(coeffs, [0i16; 16]);
    }

    #[test]
    fn idct4x4_dc_add_uniform() {
        // DC value = 128 -> (128 + 32) >> 6 = 2
        let mut dst = [50u8; 32];
        let stride = 4;
        let mut dc = 128i16;

        idct4x4_dc_add(&mut dst, stride, &mut dc);

        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(dst[j * stride + i], 52, "pixel ({i},{j})");
            }
        }
        assert_eq!(dc, 0);
    }

    #[test]
    fn idct4x4_dc_add_clamp_high() {
        // DC value that would push pixels above 255
        let mut dst = [254u8; 16];
        let stride = 4;
        let mut dc = 640i16; // (640 + 32) >> 6 = 10

        idct4x4_dc_add(&mut dst, stride, &mut dc);

        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(dst[j * stride + i], 255, "pixel ({i},{j})");
            }
        }
    }

    #[test]
    fn idct4x4_dc_add_clamp_low() {
        // Negative DC that would push pixels below 0
        let mut dst = [2u8; 16];
        let stride = 4;
        let mut dc = -640i16; // (-640 + 32) >> 6 = -9 (truncated toward zero: -608/64 = -9)

        idct4x4_dc_add(&mut dst, stride, &mut dc);

        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(dst[j * stride + i], 0, "pixel ({i},{j})");
            }
        }
    }

    #[test]
    fn idct4x4_add_butterfly_symmetry() {
        // With coeffs[0]=64 and coeffs[1]=64, the butterflies should produce
        // a horizontally varying pattern.
        let mut dst = [128u8; 64];
        let stride = 8;
        let mut coeffs = [0i16; 16];
        coeffs[0] = 64;
        coeffs[1] = 64; // AC coefficient in column 1

        idct4x4_add(&mut dst, stride, &mut coeffs);

        // After the full transform, all rows should be identical (since the
        // column transform only affects column 0 and column 1, and vertical
        // AC coefficients are zero).
        for j in 1..4 {
            for i in 0..4 {
                assert_eq!(
                    dst[j * stride + i],
                    dst[i],
                    "row {j} col {i} differs from row 0"
                );
            }
        }
    }

    #[test]
    fn idct4x4_add_matches_dc_add_for_dc_only() {
        // Verify that the full IDCT and DC shortcut produce the same result
        // for DC-only input.
        let mut dst_full = [100u8; 64];
        let mut dst_dc = [100u8; 64];
        let stride = 8;

        let mut coeffs = [0i16; 16];
        coeffs[0] = 200;
        let mut dc = 200i16;

        idct4x4_add(&mut dst_full, stride, &mut coeffs);
        idct4x4_dc_add(&mut dst_dc, stride, &mut dc);

        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(
                    dst_full[j * stride + i],
                    dst_dc[j * stride + i],
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn chroma_dc_dequant_idct_known() {
        // Input: [4, 0, 0, 0] with qmul = 128
        // a=4, b=0, c=0, d=0
        // e = 4-0 = 4, a = 4+0 = 4
        // b = 0-0 = 0, c = 0+0 = 0
        // result[0] = ((4+0)*128) >> 7 = 512 >> 7 = 4
        // result[1] = ((4+0)*128) >> 7 = 512 >> 7 = 4
        // result[2] = ((4-0)*128) >> 7 = 512 >> 7 = 4
        // result[3] = ((4-0)*128) >> 7 = 512 >> 7 = 4
        let coeffs: [i16; 4] = [4, 0, 0, 0];
        let mut output = [0i32; 4];
        chroma_dc_dequant_idct(&mut output, &coeffs, 128);
        assert_eq!(output, [4, 4, 4, 4]);
    }

    #[test]
    fn chroma_dc_dequant_idct_cross() {
        // Input: [1, 1, 1, 1] with qmul = 128
        // a=1, b=1, c=1, d=1
        // e = 1-1 = 0, a = 1+1 = 2
        // b_new = 1-1 = 0, c_new = 1+1 = 2
        // result[0] = ((2+2)*128) >> 7 = 4
        // result[1] = ((0+0)*128) >> 7 = 0
        // result[2] = ((2-2)*128) >> 7 = 0
        // result[3] = ((0-0)*128) >> 7 = 0
        let coeffs: [i16; 4] = [1, 1, 1, 1];
        let mut output = [0i32; 4];
        chroma_dc_dequant_idct(&mut output, &coeffs, 128);
        assert_eq!(output, [4, 0, 0, 0]);
    }

    #[test]
    fn chroma_dc_dequant_idct_alternating() {
        // Input: [2, -2, 0, 0] with qmul = 128
        // a=2, b=-2, c=0, d=0
        // e = 2-(-2) = 4, a = 2+(-2) = 0
        // b_new = 0-0 = 0, c_new = 0+0 = 0
        // result[0] = ((0+0)*128) >> 7 = 0
        // result[1] = ((4+0)*128) >> 7 = 4
        // result[2] = ((0-0)*128) >> 7 = 0
        // result[3] = ((4-0)*128) >> 7 = 4
        let coeffs: [i16; 4] = [2, -2, 0, 0];
        let mut output = [0i32; 4];
        chroma_dc_dequant_idct(&mut output, &coeffs, 128);
        assert_eq!(output, [0, 4, 0, 4]);
    }

    #[test]
    fn luma_dc_dequant_idct_dc_only() {
        // DC-only: input[0] = 16, rest = 0, qmul = 128
        // Horizontal pass: z0=16, z1=16, z2=0, z3=0
        //   temp = [16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        // Vertical pass col 0: z0=16, z1=16, z2=0, z3=0
        //   results = [16, 16, 16, 16] (before dequant)
        // dequant: (16 * 128 + 128) >> 8 = 2176 >> 8 = 8
        // All 16 outputs should be 8.
        let input = {
            let mut a = [0i16; 16];
            a[0] = 16;
            a
        };
        let mut output = [0i32; 16];
        luma_dc_dequant_idct(&mut output, &input, 128);

        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, 8, "block {i}");
        }
    }

    #[test]
    fn luma_dc_dequant_idct_single_ac() {
        // input[0] = 4, input[1] = 4, rest = 0, qmul = 256
        // Row 0: z0=8, z1=0, z2=4, z3=4
        //   temp[0..3] = [12, 4, -4, 4]   (z0+z3=12, z0-z3=4, z1-z2=-4, z1+z2=4)
        // Rows 1-3: all zero -> temp[4..15] = 0
        //
        // Vertical col 0: z0=12, z1=12, z2=0, z3=0
        //   (12 * 256 + 128) >> 8 = 12 for all outputs in col 0
        // Vertical col 1: z0=4, z1=4, z2=0, z3=0
        //   (4 * 256 + 128) >> 8 = 4
        // Vertical col 2: z0=-4, z1=-4, z2=0, z3=0
        //   (-4 * 256 + 128) >> 8 = (-1024 + 128) >> 8 = -896 >> 8 = -3 (arithmetic shift)
        // Vertical col 3: z0=4, z1=4, z2=0, z3=0
        //   (4 * 256 + 128) >> 8 = 4
        let input = {
            let mut a = [0i16; 16];
            a[0] = 4;
            a[1] = 4;
            a
        };
        let mut output = [0i32; 16];
        luma_dc_dequant_idct(&mut output, &input, 256);

        // Check that not all outputs are equal (the AC coefficient creates variation)
        let all_same = output.iter().all(|&v| v == output[0]);
        assert!(!all_same, "AC coefficient should create variation");
    }

    #[test]
    fn idct8x8_dc_add_uniform() {
        // DC value = 128 -> (128 + 32) >> 6 = 2
        let mut dst = [50u8; 128];
        let stride = 8;
        let mut dc = 128i16;

        idct8x8_dc_add(&mut dst, stride, &mut dc);

        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(dst[j * stride + i], 52, "pixel ({i},{j})");
            }
        }
        assert_eq!(dc, 0);
    }

    #[test]
    fn idct8x8_add_dc_only() {
        // DC-only 8x8: coeffs[0] = 128, rest zero
        // Should add (128+32)>>6 = 2 to every pixel, same as dc_add shortcut
        let mut dst_full = [100u8; 128];
        let mut dst_dc = [100u8; 128];
        let stride = 16;

        let mut coeffs = [0i16; 64];
        coeffs[0] = 128;
        let mut dc = 128i16;

        idct8x8_add(&mut dst_full, stride, &mut coeffs);
        idct8x8_dc_add(&mut dst_dc, stride, &mut dc);

        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(
                    dst_full[j * stride + i],
                    dst_dc[j * stride + i],
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn chroma422_dc_dequant_idct_dc_only() {
        // DC-only: coeffs[0] = 8, rest = 0, qmul = 128
        // Horizontal: temp = [8, 8, 0, 0, 0, 0, 0, 0]
        // Vertical col 0: z0=8, z1=8, z2=0, z3=0 -> all (8*128+128)>>8 = 4
        // Vertical col 1: z0=8, z1=8, z2=0, z3=0 -> all 4
        let mut coeffs = [0i16; 8];
        coeffs[0] = 8;
        let mut output = [0i32; 8];
        chroma422_dc_dequant_idct(&mut output, &coeffs, 128);

        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, 4, "coeff {i}");
        }
    }

    #[test]
    fn idct4x4_add_zeroes_coefficients() {
        let mut dst = [128u8; 16];
        let mut coeffs = [42i16; 16];
        idct4x4_add(&mut dst, 4, &mut coeffs);
        assert_eq!(coeffs, [0i16; 16]);
    }

    #[test]
    fn idct8x8_add_zeroes_coefficients() {
        let mut dst = [128u8; 64];
        let mut coeffs = [42i16; 64];
        idct8x8_add(&mut dst, 8, &mut coeffs);
        assert_eq!(coeffs, [0i16; 64]);
    }
}
