/*
 * Copyright (c) 2008 Mans Rullgard <mans@mansr.com>
 * Copyright (c) 2013 Janne Grunau <janne-libav@jannau.net>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "asm.S"
#include "neon.S"

function ff_h264_idct_add_neon, export=1
.L_ff_h264_idct_add_neon:
        AARCH64_VALID_CALL_TARGET
        ld1             {v0.4h, v1.4h, v2.4h, v3.4h},  [x1]
        sxtw            x2,     w2
        movi            v30.8h, #0

        add             v4.4h,  v0.4h,  v2.4h
        sshr            v16.4h, v1.4h,  #1
        st1             {v30.8h},    [x1], #16
        sshr            v17.4h, v3.4h,  #1
        st1             {v30.8h},    [x1], #16
        sub             v5.4h,  v0.4h,  v2.4h
        sub             v6.4h,  v16.4h, v3.4h
        add             v7.4h,  v1.4h,  v17.4h
        add             v0.4h,  v4.4h,  v7.4h
        add             v1.4h,  v5.4h,  v6.4h
        sub             v2.4h,  v5.4h,  v6.4h
        sub             v3.4h,  v4.4h,  v7.4h

        transpose_4x4H  v0, v1, v2, v3, v4, v5, v6, v7

        add             v4.4h,  v0.4h,  v2.4h
        ld1             {v18.s}[0], [x0], x2
        sshr            v16.4h,  v3.4h,  #1
        sshr            v17.4h,  v1.4h,  #1
        ld1             {v18.s}[1], [x0], x2
        sub             v5.4h,  v0.4h,  v2.4h
        ld1             {v19.s}[1], [x0], x2
        add             v6.4h,  v16.4h, v1.4h
        ins             v4.d[1],  v5.d[0]
        sub             v7.4h,  v17.4h, v3.4h
        ld1             {v19.s}[0], [x0], x2
        ins             v6.d[1],  v7.d[0]
        sub             x0,  x0,  x2, lsl #2
        add             v0.8h,  v4.8h,  v6.8h
        sub             v1.8h,  v4.8h,  v6.8h

        srshr           v0.8h,  v0.8h,  #6
        srshr           v1.8h,  v1.8h,  #6

        uaddw           v0.8h,  v0.8h,  v18.8b
        uaddw           v1.8h,  v1.8h,  v19.8b

        sqxtun          v0.8b, v0.8h
        sqxtun          v1.8b, v1.8h

        st1             {v0.s}[0],  [x0], x2
        st1             {v0.s}[1],  [x0], x2
        st1             {v1.s}[1],  [x0], x2
        st1             {v1.s}[0],  [x0], x2

        sub             x1,  x1,  #32
        ret
endfunc

function ff_h264_idct_dc_add_neon, export=1
.L_ff_h264_idct_dc_add_neon:
        AARCH64_VALID_CALL_TARGET
        sxtw            x2,  w2
        mov             w3,       #0
        ld1r            {v2.8h},  [x1]
        strh            w3,       [x1]
        srshr           v2.8h,  v2.8h,  #6
        ld1             {v0.s}[0],  [x0], x2
        ld1             {v0.s}[1],  [x0], x2
        uaddw           v3.8h,  v2.8h,  v0.8b
        ld1             {v1.s}[0],  [x0], x2
        ld1             {v1.s}[1],  [x0], x2
        uaddw           v4.8h,  v2.8h,  v1.8b
        sqxtun          v0.8b,  v3.8h
        sqxtun          v1.8b,  v4.8h
        sub             x0,  x0,  x2, lsl #2
        st1             {v0.s}[0],  [x0], x2
        st1             {v0.s}[1],  [x0], x2
        st1             {v1.s}[0],  [x0], x2
        st1             {v1.s}[1],  [x0], x2
        ret
endfunc

// Batch dispatch functions (add16, add16intra, add8) removed — they use
// movrel with local labels (scan8, .L_*) which generates text relocations
// on macOS. Wedeo calls individual IDCT functions directly.

.macro  idct8x8_cols    pass
  .if \pass == 0
        va      .req    v18
        vb      .req    v30
        sshr            v18.8h, v26.8h, #1
        add             v16.8h, v24.8h, v28.8h
        ld1             {v30.8h, v31.8h}, [x1]
        st1             {v19.8h}, [x1],  #16
        st1             {v19.8h}, [x1],  #16
        sub             v17.8h,  v24.8h, v28.8h
        sshr            v19.8h,  v30.8h, #1
        sub             v18.8h,  v18.8h,  v30.8h
        add             v19.8h,  v19.8h,  v26.8h
  .else
        va      .req    v30
        vb      .req    v18
        sshr            v30.8h, v26.8h, #1
        sshr            v19.8h, v18.8h, #1
        add             v16.8h, v24.8h, v28.8h
        sub             v17.8h, v24.8h, v28.8h
        sub             v30.8h, v30.8h, v18.8h
        add             v19.8h, v19.8h, v26.8h
  .endif
        add             v26.8h, v17.8h, va.8h
        sub             v28.8h, v17.8h, va.8h
        add             v24.8h, v16.8h, v19.8h
        sub             vb.8h,  v16.8h, v19.8h
        sub             v16.8h, v29.8h, v27.8h
        add             v17.8h, v31.8h, v25.8h
        sub             va.8h,  v31.8h, v25.8h
        add             v19.8h, v29.8h, v27.8h
        sub             v16.8h, v16.8h, v31.8h
        sub             v17.8h, v17.8h, v27.8h
        add             va.8h,  va.8h,  v29.8h
        add             v19.8h, v19.8h, v25.8h
        sshr            v25.8h, v25.8h, #1
        sshr            v27.8h, v27.8h, #1
        sshr            v29.8h, v29.8h, #1
        sshr            v31.8h, v31.8h, #1
        sub             v16.8h, v16.8h, v31.8h
        sub             v17.8h, v17.8h, v27.8h
        add             va.8h,  va.8h,  v29.8h
        add             v19.8h, v19.8h, v25.8h
        sshr            v25.8h, v16.8h, #2
        sshr            v27.8h, v17.8h, #2
        sshr            v29.8h, va.8h,  #2
        sshr            v31.8h, v19.8h, #2
        sub             v19.8h, v19.8h, v25.8h
        sub             va.8h,  v27.8h, va.8h
        add             v17.8h, v17.8h, v29.8h
        add             v16.8h, v16.8h, v31.8h
  .if \pass == 0
        sub             v31.8h, v24.8h, v19.8h
        add             v24.8h, v24.8h, v19.8h
        add             v25.8h, v26.8h, v18.8h
        sub             v18.8h, v26.8h, v18.8h
        add             v26.8h, v28.8h, v17.8h
        add             v27.8h, v30.8h, v16.8h
        sub             v29.8h, v28.8h, v17.8h
        sub             v28.8h, v30.8h, v16.8h
  .else
        sub             v31.8h, v24.8h, v19.8h
        add             v24.8h, v24.8h, v19.8h
        add             v25.8h, v26.8h, v30.8h
        sub             v30.8h, v26.8h, v30.8h
        add             v26.8h, v28.8h, v17.8h
        sub             v29.8h, v28.8h, v17.8h
        add             v27.8h, v18.8h, v16.8h
        sub             v28.8h, v18.8h, v16.8h
  .endif
        .unreq          va
        .unreq          vb
.endm

function ff_h264_idct8_add_neon, export=1
.L_ff_h264_idct8_add_neon:
        AARCH64_VALID_CALL_TARGET
        movi            v19.8h,   #0
        sxtw            x2,       w2
        ld1             {v24.8h, v25.8h}, [x1]
        st1             {v19.8h},  [x1],   #16
        st1             {v19.8h},  [x1],   #16
        ld1             {v26.8h, v27.8h}, [x1]
        st1             {v19.8h},  [x1],   #16
        st1             {v19.8h},  [x1],   #16
        ld1             {v28.8h, v29.8h}, [x1]
        st1             {v19.8h},  [x1],   #16
        st1             {v19.8h},  [x1],   #16

        idct8x8_cols    0
        transpose_8x8H  v24, v25, v26, v27, v28, v29, v18, v31, v6, v7
        idct8x8_cols    1

        mov             x3,  x0
        srshr           v24.8h, v24.8h, #6
        ld1             {v0.8b},     [x0], x2
        srshr           v25.8h, v25.8h, #6
        ld1             {v1.8b},     [x0], x2
        srshr           v26.8h, v26.8h, #6
        ld1             {v2.8b},     [x0], x2
        srshr           v27.8h, v27.8h, #6
        ld1             {v3.8b},     [x0], x2
        srshr           v28.8h, v28.8h, #6
        ld1             {v4.8b},     [x0], x2
        srshr           v29.8h, v29.8h, #6
        ld1             {v5.8b},     [x0], x2
        srshr           v30.8h, v30.8h, #6
        ld1             {v6.8b},     [x0], x2
        srshr           v31.8h, v31.8h, #6
        ld1             {v7.8b},     [x0], x2
        uaddw           v24.8h, v24.8h, v0.8b
        uaddw           v25.8h, v25.8h, v1.8b
        uaddw           v26.8h, v26.8h, v2.8b
        sqxtun          v0.8b,  v24.8h
        uaddw           v27.8h, v27.8h, v3.8b
        sqxtun          v1.8b,  v25.8h
        uaddw           v28.8h, v28.8h, v4.8b
        sqxtun          v2.8b,  v26.8h
        st1             {v0.8b},     [x3], x2
        uaddw           v29.8h, v29.8h, v5.8b
        sqxtun          v3.8b,  v27.8h
        st1             {v1.8b},     [x3], x2
        uaddw           v30.8h, v30.8h, v6.8b
        sqxtun          v4.8b,  v28.8h
        st1             {v2.8b},     [x3], x2
        uaddw           v31.8h, v31.8h, v7.8b
        sqxtun          v5.8b,  v29.8h
        st1             {v3.8b},     [x3], x2
        sqxtun          v6.8b,  v30.8h
        sqxtun          v7.8b,  v31.8h
        st1             {v4.8b},     [x3], x2
        st1             {v5.8b},     [x3], x2
        st1             {v6.8b},     [x3], x2
        st1             {v7.8b},     [x3], x2

        sub             x1,  x1,  #128
        ret
endfunc

function ff_h264_idct8_dc_add_neon, export=1
.L_ff_h264_idct8_dc_add_neon:
        AARCH64_VALID_CALL_TARGET
        mov             w3,       #0
        sxtw            x2,       w2
        ld1r            {v31.8h}, [x1]
        strh            w3,       [x1]
        ld1             {v0.8b},  [x0], x2
        srshr           v31.8h, v31.8h, #6
        ld1             {v1.8b},     [x0], x2
        ld1             {v2.8b},     [x0], x2
        uaddw           v24.8h, v31.8h, v0.8b
        ld1             {v3.8b},     [x0], x2
        uaddw           v25.8h, v31.8h, v1.8b
        ld1             {v4.8b},     [x0], x2
        uaddw           v26.8h, v31.8h, v2.8b
        ld1             {v5.8b},     [x0], x2
        uaddw           v27.8h, v31.8h, v3.8b
        ld1             {v6.8b},     [x0], x2
        uaddw           v28.8h, v31.8h, v4.8b
        ld1             {v7.8b},     [x0], x2
        uaddw           v29.8h, v31.8h, v5.8b
        uaddw           v30.8h, v31.8h, v6.8b
        uaddw           v31.8h, v31.8h, v7.8b
        sqxtun          v0.8b,  v24.8h
        sqxtun          v1.8b,  v25.8h
        sqxtun          v2.8b,  v26.8h
        sqxtun          v3.8b,  v27.8h
        sub             x0,  x0,  x2, lsl #3
        st1             {v0.8b},     [x0], x2
        sqxtun          v4.8b,  v28.8h
        st1             {v1.8b},     [x0], x2
        sqxtun          v5.8b,  v29.8h
        st1             {v2.8b},     [x0], x2
        sqxtun          v6.8b,  v30.8h
        st1             {v3.8b},     [x0], x2
        sqxtun          v7.8b,  v31.8h
        st1             {v4.8b},     [x0], x2
        st1             {v5.8b},     [x0], x2
        st1             {v6.8b},     [x0], x2
        st1             {v7.8b},     [x0], x2
        ret
endfunc

// ff_h264_idct8_add4_neon and scan8 table removed (same text-relocation issue).
