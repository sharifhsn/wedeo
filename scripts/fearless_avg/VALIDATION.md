# H.264 square averaging verification

On 2026-11-24, the optimized Wedeo kernel and the fearless_simd 1.2.0 comparison passed against the SSE2 averaging functions in FFmpeg `74e4393595cf2b5857f0da925c4e0dd74ea306e5`. The x86-64 Linux host was a Ryzen 5 5600X. The local `libavcodec/x86/fpel.o` had SHA-256 `012af958216ba4ac638dae699172c4c0e9256f2a8472b60cb42a1bb7a88d8bec`.

- The 13 release checkasm tests passed on Rust 1.100.0. The averaging cases compare public dispatch and explicit Scalar, Sse2 and Avx2 selections against widened integer arithmetic. Widths 1, 4, 7, 8, 9, 15, 16 and 17 exercise both sides of the square specialization. Cases include rectangular blocks, offsets 0–15, unequal and overlapping row strides, zero strides and exact buffer extents. Zero dimensions, a one-row block with maximum strides, short buffers and overflowing extent arithmetic are checked separately. Invalid extents must panic before mutating the destination.
- The two release oracle tests passed on Rust 1.100.0 and 1.98.0. FFmpeg supplied the expected bytes for all 65,536 ordered byte pairs separately at 8×8 and 16×16. Another 160 padded layouts cover source offsets 0–15, both aligned and unaligned 8×8 destinations, aligned 16×16 destinations, and pitches up to 2,048 bytes. Each test compares the entire destination allocation, including padding and guard bytes. FFmpeg receives equal source and destination strides. The tests exercise Wedeo's public and explicit backend entry points, fearless normal dispatch and an explicitly selected `Level::fallback()`.
- The production square helper now rejects widths other than 8 and 16 at compile time. Its fixed-width accesses require that invariant, and its two callers already satisfy it. No output defect was found on the supported cases. The Rust 1.89 MSRV remains unverified: that compiler was absent from the detected project and user-local installations. Rust 1.98 was tested on the standalone oracle crate, which includes the production SIMD module. No complete workspace build on that version is claimed.

## Reproduce on x86-64 Linux

Use the repository-pinned toolchain on PATH, the pinned FFmpeg submodule, and its configured x86 assembly build. From the repository root:

```sh
make -C FFmpeg -j1 libavcodec/x86/fpel.o
git -C FFmpeg rev-parse HEAD
sha256sum FFmpeg/libavcodec/x86/fpel.o
cargo test --locked --offline -j1 --release -p wedeo-codec-h264 --test checkasm_h264
RUSTFLAGS="-A dead_code -C link-arg=$PWD/FFmpeg/libavcodec/x86/fpel.o" \
  cargo test --locked --offline -j1 --release --features fallback \
    --manifest-path scripts/fearless_avg/Cargo.toml
```

To repeat the secondary compiler check, put the installed Rust 1.98 toolchain on PATH and run the last command with a separate target directory. All timings printed by these test commands describe test execution only.

## Recorded test runs

**Rust 1.100.0 checkasm**

```text
Compiling wedeo-codec-h264 v0.3.0 (/home/lain/lain/projects/wedeo/codecs/wedeo-codec-h264)
    Finished `release` profile [optimized] target(s) in 37.10s
     Running tests/checkasm_h264.rs (/tmp/wedeo-cpu-target/release/build/wedeo-codec-h264/a1711a5d05c7185b/out/checkasm_h264-a1711a5d05c7185b)

running 13 tests
test average_zero_and_invalid_extents ... ok
test average_exact_extents_and_independent_strides ... ok
test checkasm_idct_dc_add_4x4 ... ok
test checkasm_reports_backends ... ok
test dispatched_square_average_matches_scalar_with_padding_and_overlap ... ok
test checkasm_idct_add_4x4 ... ok
test checkasm_idct_dc_add_8x8 ... ok
test checkasm_public_dispatch_idct ... ok
test checkasm_public_dispatch_avg_wrappers ... ok
test checkasm_idct_add_8x8 ... ok
test checkasm_put_and_avg_pixels ... ok
test checkasm_chroma_mc ... ok
test checkasm_luma_6tap ... ok

test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s
```

**Rust 1.100.0 backend inventory**

```text
Finished `release` profile [optimized] target(s) in 0.04s
     Running tests/checkasm_h264.rs (/tmp/wedeo-cpu-target/release/build/wedeo-codec-h264/a1711a5d05c7185b/out/checkasm_h264-a1711a5d05c7185b)

running 1 test
checkasm backends: [Scalar, Sse2, Avx2], dispatch = Avx2
test checkasm_reports_backends ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s
```

**Rust 1.100.0 FFmpeg oracle**

```text
Compiling proc-macro2 v1.0.107
   Compiling unicode-ident v1.0.27
   Compiling quote v1.0.47
   Compiling syn v3.0.10
   Compiling fearless_simd_macros v0.1.0
   Compiling fearless_simd v1.2.0
   Compiling wedeo-fearless-eval v0.1.0 (/home/lain/lain/projects/wedeo/scripts/fearless_avg)
    Finished `release` profile [optimized] target(s) in 23.67s
     Running unittests src/main.rs (/tmp/wedeo-cpu-target/release/build/wedeo-fearless-eval/f5cb377ef4a49805/out/wedeo_fearless_eval-f5cb377ef4a49805)

running 2 tests
test tests::padded_and_offset_rows_match_ffmpeg ... ok
test tests::packed_every_input_pair_matches_ffmpeg ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
```

**Rust 1.98.0 FFmpeg oracle**

```text
Compiling proc-macro2 v1.0.107
   Compiling quote v1.0.47
   Compiling unicode-ident v1.0.27
   Compiling syn v3.0.10
   Compiling fearless_simd_macros v0.1.0
   Compiling fearless_simd v1.2.0
   Compiling wedeo-fearless-eval v0.1.0 (/home/lain/lain/projects/wedeo/scripts/fearless_avg)
    Finished `release` profile [optimized] target(s) in 11.34s
     Running unittests src/main.rs (/tmp/wedeo-cpu-edge-198/release/deps/wedeo_fearless_eval-f7cbdce64993d3f1)

running 2 tests
test tests::padded_and_offset_rows_match_ffmpeg ... ok
test tests::packed_every_input_pair_matches_ffmpeg ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
```
