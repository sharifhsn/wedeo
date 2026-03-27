// H.264 decoder micro-benchmarks — establishes scalar baselines for SIMD work.
//
// Run:   cargo bench -p wedeo-codec-h264
// Filter: cargo bench -p wedeo-codec-h264 -- mc
// Test:   cargo bench -p wedeo-codec-h264 -- --test

use divan::AllocProfiler;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

/// Deterministic pseudo-random fill (LCG, no `rand` dependency).
fn seeded_bytes(buf: &mut [u8], seed: u32) {
    let mut s = seed;
    for b in buf.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        *b = (s >> 16) as u8;
    }
}

/// Deterministic pseudo-random i16 values in [-range, range].
fn seeded_i16s(buf: &mut [i16], seed: u32, range: i16) {
    let mut s = seed;
    for v in buf.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let raw = ((s >> 16) as i32 % (range as i32 * 2 + 1)) - range as i32;
        *v = raw as i16;
    }
}

// ---------------------------------------------------------------------------
// Motion Compensation
// ---------------------------------------------------------------------------

mod mc {
    use divan::counter::BytesCount;
    use divan::{Bencher, black_box};
    use wedeo_codec_h264::mc::{self, McScratch};

    use super::seeded_bytes;

    #[derive(Clone, Copy)]
    struct McParam {
        w: usize,
        h: usize,
        dx: u8,
        dy: u8,
    }

    impl std::fmt::Display for McParam {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}x{}_qpel({},{})", self.w, self.h, self.dx, self.dy)
        }
    }

    const MC_PARAMS: &[McParam] = &[
        // 16x16 — full MB, all major code paths
        McParam {
            w: 16,
            h: 16,
            dx: 0,
            dy: 0,
        }, // full-pel copy
        McParam {
            w: 16,
            h: 16,
            dx: 2,
            dy: 0,
        }, // h_lowpass
        McParam {
            w: 16,
            h: 16,
            dx: 0,
            dy: 2,
        }, // v_lowpass
        McParam {
            w: 16,
            h: 16,
            dx: 2,
            dy: 2,
        }, // hv_lowpass (most expensive)
        McParam {
            w: 16,
            h: 16,
            dx: 1,
            dy: 1,
        }, // qpel diagonal
        // 8x8 — sub-MB partitions
        McParam {
            w: 8,
            h: 8,
            dx: 0,
            dy: 0,
        },
        McParam {
            w: 8,
            h: 8,
            dx: 2,
            dy: 2,
        },
        McParam {
            w: 8,
            h: 8,
            dx: 1,
            dy: 1,
        },
        // 4x4 — smallest partition
        McParam {
            w: 4,
            h: 4,
            dx: 0,
            dy: 0,
        },
        McParam {
            w: 4,
            h: 4,
            dx: 2,
            dy: 2,
        },
    ];

    fn make_ref_frame() -> Vec<u8> {
        let mut data = vec![0u8; 64 * 64];
        seeded_bytes(&mut data, 0xDEAD_BEEF);
        data
    }

    #[divan::bench(args = MC_PARAMS)]
    fn mc_luma(bencher: Bencher, p: &McParam) {
        let ref_frame = make_ref_frame();
        bencher
            .counter(BytesCount::new(p.w * p.h))
            .with_inputs(|| (vec![0u8; 16 * p.h], McScratch::new()))
            .bench_local_refs(|(dst, scratch)| {
                mc::mc_luma(
                    black_box(&mut dst[..]),
                    black_box(16),
                    black_box(&ref_frame),
                    black_box(64),
                    black_box(10),
                    black_box(10),
                    black_box(p.dx),
                    black_box(p.dy),
                    black_box(p.w),
                    black_box(p.h),
                    black_box(64),
                    black_box(64),
                    scratch,
                );
            });
    }

    #[derive(Clone, Copy)]
    struct ChromaParam {
        w: usize,
        h: usize,
        dx: u8,
        dy: u8,
    }

    impl std::fmt::Display for ChromaParam {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}x{}_frac({},{})", self.w, self.h, self.dx, self.dy)
        }
    }

    const CHROMA_PARAMS: &[ChromaParam] = &[
        ChromaParam {
            w: 8,
            h: 8,
            dx: 0,
            dy: 0,
        },
        ChromaParam {
            w: 8,
            h: 8,
            dx: 4,
            dy: 4,
        },
        ChromaParam {
            w: 8,
            h: 8,
            dx: 2,
            dy: 3,
        },
        ChromaParam {
            w: 4,
            h: 4,
            dx: 0,
            dy: 0,
        },
        ChromaParam {
            w: 4,
            h: 4,
            dx: 4,
            dy: 4,
        },
    ];

    #[divan::bench(args = CHROMA_PARAMS)]
    fn mc_chroma(bencher: Bencher, p: &ChromaParam) {
        let ref_frame = make_ref_frame();
        bencher
            .counter(BytesCount::new(p.w * p.h))
            .with_inputs(|| vec![0u8; 16 * p.h])
            .bench_local_refs(|dst| {
                mc::mc_chroma(
                    black_box(&mut dst[..]),
                    black_box(16),
                    black_box(&ref_frame),
                    black_box(64),
                    black_box(10),
                    black_box(10),
                    black_box(p.dx),
                    black_box(p.dy),
                    black_box(p.w),
                    black_box(p.h),
                    black_box(32),
                    black_box(32),
                );
            });
    }

    const AVG_SIZES: &[usize] = &[16, 8, 4];

    #[divan::bench(args = AVG_SIZES)]
    fn avg_pixels_inplace(bencher: Bencher, size: usize) {
        bencher
            .counter(BytesCount::new(size * size))
            .with_inputs(|| {
                let mut dst = vec![128u8; size * size];
                let mut src = vec![0u8; size * size];
                seeded_bytes(&mut dst, 0x1234);
                seeded_bytes(&mut src, 0x5678);
                (dst, src)
            })
            .bench_local_refs(|(dst, src)| {
                mc::avg_pixels_inplace(
                    black_box(&mut dst[..]),
                    black_box(size),
                    black_box(&src[..]),
                    black_box(size),
                    black_box(size),
                    black_box(size),
                );
            });
    }
}

// ---------------------------------------------------------------------------
// Inverse DCT
// ---------------------------------------------------------------------------

mod idct {
    use divan::counter::BytesCount;
    use divan::{Bencher, black_box};
    use wedeo_codec_h264::idct;

    use super::seeded_i16s;

    const STRIDES_4: &[usize] = &[4, 32];
    const STRIDES_8: &[usize] = &[8, 32];

    #[divan::bench(args = STRIDES_4)]
    fn idct4x4_add(bencher: Bencher, stride: usize) {
        bencher
            .counter(BytesCount::new(4usize * 4))
            .with_inputs(|| {
                let dst = vec![128u8; stride * 4];
                let mut coeffs = [0i16; 16];
                seeded_i16s(&mut coeffs, 0xCAFE, 200);
                (dst, coeffs)
            })
            .bench_local_refs(|(dst, coeffs)| {
                idct::idct4x4_add(
                    black_box(&mut dst[..]),
                    black_box(stride),
                    black_box(coeffs),
                );
            });
    }

    #[divan::bench(args = STRIDES_4)]
    fn idct4x4_dc_add(bencher: Bencher, stride: usize) {
        bencher
            .counter(BytesCount::new(4usize * 4))
            .with_inputs(|| {
                let dst = vec![128u8; stride * 4];
                let dc = 500i16;
                (dst, dc)
            })
            .bench_local_refs(|(dst, dc)| {
                idct::idct4x4_dc_add(black_box(&mut dst[..]), black_box(stride), black_box(dc));
            });
    }

    #[divan::bench(args = STRIDES_8)]
    fn idct8x8_add(bencher: Bencher, stride: usize) {
        bencher
            .counter(BytesCount::new(8usize * 8))
            .with_inputs(|| {
                let dst = vec![128u8; stride * 8];
                let mut coeffs = [0i16; 64];
                seeded_i16s(&mut coeffs, 0xBEEF, 200);
                (dst, coeffs)
            })
            .bench_local_refs(|(dst, coeffs)| {
                idct::idct8x8_add(
                    black_box(&mut dst[..]),
                    black_box(stride),
                    black_box(coeffs),
                );
            });
    }

    #[divan::bench(args = STRIDES_8)]
    fn idct8x8_dc_add(bencher: Bencher, stride: usize) {
        bencher
            .counter(BytesCount::new(8usize * 8))
            .with_inputs(|| {
                let dst = vec![128u8; stride * 8];
                let dc = 500i16;
                (dst, dc)
            })
            .bench_local_refs(|(dst, dc)| {
                idct::idct8x8_dc_add(black_box(&mut dst[..]), black_box(stride), black_box(dc));
            });
    }

    #[divan::bench]
    fn luma_dc_dequant_idct(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let mut input = [0i16; 16];
                seeded_i16s(&mut input, 0xF00D, 100);
                input
            })
            .bench_local_refs(|input| {
                let mut output = [0i32; 16];
                idct::luma_dc_dequant_idct(black_box(&mut output), black_box(input), black_box(10));
            });
    }

    #[divan::bench]
    fn chroma_dc_dequant_idct(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let mut coeffs = [0i16; 4];
                seeded_i16s(&mut coeffs, 0xBAD0, 100);
                coeffs
            })
            .bench_local_refs(|coeffs| {
                let mut output = [0i32; 4];
                idct::chroma_dc_dequant_idct(
                    black_box(&mut output),
                    black_box(coeffs),
                    black_box(10),
                );
            });
    }

    #[divan::bench]
    fn chroma422_dc_dequant_idct(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let mut coeffs = [0i16; 8];
                seeded_i16s(&mut coeffs, 0xD00D, 100);
                coeffs
            })
            .bench_local_refs(|coeffs| {
                let mut output = [0i32; 8];
                idct::chroma422_dc_dequant_idct(
                    black_box(&mut output),
                    black_box(coeffs),
                    black_box(10),
                );
            });
    }
}

// ---------------------------------------------------------------------------
// Intra Prediction
// ---------------------------------------------------------------------------

mod intra_pred {
    use divan::counter::BytesCount;
    use divan::{Bencher, black_box};
    use wedeo_codec_h264::intra_pred;

    const MODES_4X4: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];
    const MODES_8X8: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];
    const MODES_16X16: &[u8] = &[0, 1, 2, 3];
    const MODES_CHROMA: &[u8] = &[0, 1, 2, 3];

    #[divan::bench(args = MODES_4X4)]
    fn predict_4x4(bencher: Bencher, mode: u8) {
        // top needs 8 bytes (4 pixels + 4 top-right), left needs 4 bytes
        let top: [u8; 8] = [100, 110, 120, 130, 140, 150, 160, 170];
        let left: [u8; 4] = [105, 115, 125, 135];
        let stride = 32usize;
        bencher
            .counter(BytesCount::new(4usize * 4))
            .with_inputs(|| vec![0u8; stride * 4])
            .bench_local_refs(|dst| {
                intra_pred::predict_4x4(
                    black_box(&mut dst[..]),
                    black_box(stride),
                    black_box(mode),
                    black_box(&top),
                    black_box(&left),
                    black_box(95),
                    black_box(true),
                    black_box(true),
                    black_box(true),
                );
            });
    }

    #[divan::bench(args = MODES_8X8)]
    fn predict_8x8l(bencher: Bencher, mode: u8) {
        // top needs 16 bytes (8 pixels + 8 top-right), left needs 8 bytes
        let top: [u8; 16] = [
            100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175,
        ];
        let left: [u8; 8] = [102, 112, 122, 132, 142, 152, 162, 172];
        let stride = 32usize;
        bencher
            .counter(BytesCount::new(8usize * 8))
            .with_inputs(|| vec![0u8; stride * 8])
            .bench_local_refs(|dst| {
                intra_pred::predict_8x8l(
                    black_box(&mut dst[..]),
                    black_box(stride),
                    black_box(mode),
                    black_box(&top),
                    black_box(&left),
                    black_box(98),
                    black_box(true),
                    black_box(true),
                    black_box(true),
                    black_box(true),
                );
            });
    }

    #[divan::bench(args = MODES_16X16)]
    fn predict_16x16(bencher: Bencher, mode: u8) {
        let top: [u8; 16] = [
            100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160,
        ];
        let left: [u8; 16] = [
            102, 106, 110, 114, 118, 122, 126, 130, 134, 138, 142, 146, 150, 154, 158, 162,
        ];
        let stride = 32usize;
        bencher
            .counter(BytesCount::new(16usize * 16))
            .with_inputs(|| vec![0u8; stride * 16])
            .bench_local_refs(|dst| {
                intra_pred::predict_16x16(
                    black_box(&mut dst[..]),
                    black_box(stride),
                    black_box(mode),
                    black_box(&top),
                    black_box(&left),
                    black_box(99),
                    black_box(true),
                    black_box(true),
                );
            });
    }

    #[divan::bench(args = MODES_CHROMA)]
    fn predict_chroma_8x8(bencher: Bencher, mode: u8) {
        let top: [u8; 8] = [100, 110, 120, 130, 140, 150, 160, 170];
        let left: [u8; 8] = [105, 115, 125, 135, 145, 155, 165, 175];
        let stride = 32usize;
        bencher
            .counter(BytesCount::new(8usize * 8))
            .with_inputs(|| vec![0u8; stride * 8])
            .bench_local_refs(|dst| {
                intra_pred::predict_chroma_8x8(
                    black_box(&mut dst[..]),
                    black_box(stride),
                    black_box(mode),
                    black_box(&top),
                    black_box(&left),
                    black_box(98),
                    black_box(true),
                    black_box(true),
                );
            });
    }
}

// ---------------------------------------------------------------------------
// Deblocking Filter
// ---------------------------------------------------------------------------

mod deblock {
    use divan::counter::BytesCount;
    use divan::{Bencher, black_box};
    use wedeo_codec_h264::deblock::{self, MbDeblockInfo, PictureBuffer, SliceDeblockParams};

    use super::seeded_bytes;

    // -- Level 1: compute_bs (pure function) --

    #[derive(Clone, Copy)]
    struct BsScenario {
        name: &'static str,
        is_mb_edge: bool,
        p_intra: bool,
        q_intra: bool,
        p_nnz: u8,
        q_nnz: u8,
        p_ref: i32,
        q_ref: i32,
        p_mv: [i16; 2],
        q_mv: [i16; 2],
        list_count: u8,
    }

    impl std::fmt::Display for BsScenario {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str(self.name)
        }
    }

    const BS_SCENARIOS: &[BsScenario] = &[
        BsScenario {
            name: "both_intra",
            is_mb_edge: true,
            p_intra: true,
            q_intra: true,
            p_nnz: 0,
            q_nnz: 0,
            p_ref: 0,
            q_ref: 0,
            p_mv: [0, 0],
            q_mv: [0, 0],
            list_count: 1,
        },
        BsScenario {
            name: "nnz_diff",
            is_mb_edge: false,
            p_intra: false,
            q_intra: false,
            p_nnz: 3,
            q_nnz: 0,
            p_ref: 0,
            q_ref: 0,
            p_mv: [0, 0],
            q_mv: [0, 0],
            list_count: 1,
        },
        BsScenario {
            name: "mv_diff",
            is_mb_edge: false,
            p_intra: false,
            q_intra: false,
            p_nnz: 0,
            q_nnz: 0,
            p_ref: 0,
            q_ref: 0,
            p_mv: [20, 15],
            q_mv: [0, 0],
            list_count: 1,
        },
        BsScenario {
            name: "same_everything",
            is_mb_edge: false,
            p_intra: false,
            q_intra: false,
            p_nnz: 0,
            q_nnz: 0,
            p_ref: 0,
            q_ref: 0,
            p_mv: [4, 4],
            q_mv: [4, 4],
            list_count: 1,
        },
        BsScenario {
            name: "b_slice_mv_diff",
            is_mb_edge: false,
            p_intra: false,
            q_intra: false,
            p_nnz: 0,
            q_nnz: 0,
            p_ref: 0,
            q_ref: 0,
            p_mv: [10, 5],
            q_mv: [0, 0],
            list_count: 2,
        },
    ];

    #[divan::bench(args = BS_SCENARIOS)]
    fn compute_bs(bencher: Bencher, s: &BsScenario) {
        bencher.bench_local(|| {
            deblock::compute_bs(
                black_box(s.is_mb_edge),
                black_box(s.p_intra),
                black_box(s.q_intra),
                black_box(s.p_nnz),
                black_box(s.q_nnz),
                black_box(s.p_ref),
                black_box(s.q_ref),
                black_box(s.p_mv),
                black_box(s.q_mv),
                black_box(i32::MIN), // p_ref_l1
                black_box(i32::MIN), // q_ref_l1
                black_box([0i16; 2]),
                black_box([0i16; 2]),
                black_box(s.list_count),
                black_box(4),
                black_box(false),
            )
        });
    }

    // -- Level 2: deblock_row (full row) --

    fn make_deblock_fixture(
        mb_width: u32,
    ) -> (
        PictureBuffer,
        Vec<MbDeblockInfo>,
        Vec<u16>,
        Vec<SliceDeblockParams>,
    ) {
        let mb_height = 2u32;
        let width = mb_width * 16;
        let height = mb_height * 16;
        let y_stride = width as usize;
        let uv_stride = (width / 2) as usize;

        let mut y = vec![0u8; y_stride * height as usize];
        let mut u = vec![128u8; uv_stride * (height / 2) as usize];
        let mut v = vec![128u8; uv_stride * (height / 2) as usize];
        seeded_bytes(&mut y, 0xABCD_1234);
        seeded_bytes(&mut u, 0x5678_9ABC);
        seeded_bytes(&mut v, 0xDEF0_1234);

        let pic = PictureBuffer {
            y,
            u,
            v,
            y_stride,
            uv_stride,
            width,
            height,
            mb_width,
            mb_height,
        };

        let total_mbs = (mb_width * mb_height) as usize;
        let mut mb_info = vec![MbDeblockInfo::default(); total_mbs];
        for (i, info) in mb_info.iter_mut().enumerate() {
            info.qp = 28;
            info.is_intra = i % 4 == 0; // every 4th MB is intra
            if !info.is_intra {
                info.non_zero_count[0] = 2;
                info.non_zero_count[1] = 1;
                info.ref_poc[0] = 0;
                info.mv[0] = [8, 4];
            }
        }

        let slice_table = vec![0u16; total_mbs];
        let slice_params = vec![SliceDeblockParams {
            alpha_c0_offset: 0,
            beta_offset: 0,
            disable_deblocking_filter_idc: 0,
            chroma_qp_index_offset: [0, 0],
        }];

        (pic, mb_info, slice_table, slice_params)
    }

    const ROW_WIDTHS: &[u32] = &[4, 8, 16];

    #[divan::bench(args = ROW_WIDTHS)]
    fn deblock_row(bencher: Bencher, mb_width: &u32) {
        let mb_width = *mb_width;
        bencher
            .counter(BytesCount::new(mb_width as usize * 16 * 16 * 3 / 2))
            .with_inputs(|| make_deblock_fixture(mb_width))
            .bench_local_refs(|(pic, mb_info, slice_table, slice_params)| {
                deblock::deblock_row(
                    black_box(pic),
                    black_box(mb_info),
                    black_box(slice_table),
                    black_box(slice_params),
                    black_box(1), // mb_y=1 so top neighbor exists
                    black_box(mb_width),
                );
            });
    }
}

// ---------------------------------------------------------------------------
// CABAC Entropy Decoding
// ---------------------------------------------------------------------------

mod cabac {
    use divan::counter::ItemsCount;
    use divan::{Bencher, black_box};
    use wedeo_codec_h264::cabac::CabacReader;

    const DECODE_COUNT: usize = 1000;

    fn make_cabac_data() -> Vec<u8> {
        let mut data = vec![0u8; 4096];
        super::seeded_bytes(&mut data, 0x12345678);
        data
    }

    const INIT_STATES: &[u8] = &[0, 64, 120];

    #[divan::bench(args = INIT_STATES)]
    fn get_cabac(bencher: Bencher, init_state: u8) {
        let data = make_cabac_data();
        bencher
            .counter(ItemsCount::new(DECODE_COUNT))
            .with_inputs(|| {
                let reader = CabacReader::new(&data).unwrap();
                let state = init_state;
                (reader, state)
            })
            .bench_local_refs(|(reader, state)| {
                for _ in 0..DECODE_COUNT {
                    black_box(reader.get_cabac(state));
                }
            });
    }

    #[divan::bench]
    fn get_cabac_bypass(bencher: Bencher) {
        let data = make_cabac_data();
        bencher
            .counter(ItemsCount::new(DECODE_COUNT))
            .with_inputs(|| CabacReader::new(&data).unwrap())
            .bench_local_refs(|reader| {
                for _ in 0..DECODE_COUNT {
                    black_box(reader.get_cabac_bypass());
                }
            });
    }
}
