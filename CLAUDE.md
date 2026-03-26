# Wedeo - FFmpeg Rust Rewrite

## Project Overview

Clean-room Rust rewrite of FFmpeg. **No bindgen, no c2rust, no FFI.** Pure Rust.
Verification target: **bit-for-bit output parity** with FFmpeg's FATE test suite.
See `H264.md` for decoder architecture, module map, and conformance status.

## Philosophy

- **"Make it work, make it right, make it fast."** Correctness first.
- Reproduce C bugs if needed for behavioral parity.
- Copy FFmpeg comments verbatim. Don't invent docs where FFmpeg has none.

## Build & Verify

- `cargo clippy` + `cargo fmt` before considering code complete
- `cargo nextest run` for tests (process isolation, leak detection)
- FATE: `FATE_SUITE=./fate-suite cargo nextest run --profile fate -p wedeo-fate`
- Video cross-validate: `cargo run --bin wedeo-framecrc -- <file>` vs `ffmpeg -bitexact -i <file> -f framecrc -`
- Conformance: `python3 scripts/conformance_full.py` (full report), `scripts/regression_check.py` (quick check)

## Debugging H.264 Differences

**Conformance workflow (do this in order):**
1. `scripts/conformance_full.py` — full report. `--save-snapshot` to baseline.
2. `scripts/regression_check.py` — quick check against snapshot. Run after every change.
3. `scripts/framecrc_compare.py --no-deblock --pixel-detail <file>` — triage MC vs deblock.
4. `scripts/mb_compare.py <file> --start-frame N --max-frames 1` — find differing MBs.
5. `scripts/cabac_state_at_mb.py <file> --mb-x X --mb-y Y` — CABAC state comparison via lldb.

**Key rules:**
- **Read the FFmpeg C code FIRST.** Key files: `h264_cavlc.c`, `h264_cabac.c`, `h264idct_template.c`, `h264_mb.c`, `h264_mb_template.c`, `h264_ps.c`, `h264_mvpred.h`.
- **HARD RULE:** After 2 failed hypotheses, **STOP theorizing**. Extract values from BOTH decoders. Find WHERE values diverge before explaining WHY. **5-minute backstop** without ground-truth extraction → use lldb.
- **When formulas look identical, the bug is in the INPUTS.** One lldb extraction > any algebraic analysis.
- **Never infer intermediate values from outputs.** Measure via lldb: `breakpoint set -f file.c -l N` → `expression`.
- **Never manually count entries in C arrays.** Use `scripts/verify_tables.py` or write a parser.
- **Build debug FFmpeg:** `cd FFmpeg && ./configure --disable-optimizations --enable-debug=3 --disable-stripping --disable-asm && make -j$(sysctl -n hw.ncpu) ffmpeg`. `--disable-asm` is critical on ARM64.
- **Tracing is always available.** `RUST_LOG=wedeo_codec_h264::mb=trace`. Never use `eprintln!` — always `tracing` macros.
- **When existing logs don't reveal the divergence, get better logs.** (1) add `trace!()`, (2) FFmpeg `-loglevel debug`, (3) lldb.
- **CABAC cat=5 (8x8 luma) has NO coded_block_flag.** The CBP check is the only gate.
- **CABAC context offsets are 2D for MBAFF** — `significant_coeff_flag_offset[MB_FIELD][cat]` uses field vs frame tables. Use `scripts/verify_cabac_tables.py` to verify.
- **MBAFF field-mode above-neighbor stride** — `top_xy = mb_xy - (mb_stride << MB_FIELD)`. Use `scripts/mbaff_field_map.py` to check field/frame modes.
- **MBAFF left_block_options** — FFmpeg's `fill_decode_neighbors` (h264_mvpred.h:491-538) selects 1 of 4 remapping variants for NNZ/CBP/intra4x4 mode context when there's a field/frame mismatch with the left neighbor. Use `scripts/verify_left_block_tables.py` to verify tables. For CABAC state comparison, breakpoint at h264_cabac.c:1966 (after field flag decode).
- **Pipeline-stage tracing tags:** SLICE→PPS_SCALING→DEQUANT_TABLES→COEFF→DEQUANT→MB_RECON→MB_DEBLOCK.

## Code Quality

- Fix all clippy warnings. No `#[allow(clippy::*)]` without a comment.
- Prefer safe Rust. `unsafe` requires `// SAFETY:` comment.
- `Result<T, E>` + `?` for errors, not panics.
- `wrapping_add`/`wrapping_neg`/`wrapping_mul` for C overflow parity.

## Architecture

- Bottom-up: wedeo-core → wedeo-codec/wedeo-format → implementations → CLI
- Each FFmpeg library → one Rust crate. Codecs/formats registered via `inventory`.
- Traits: `Decoder`, `Demuxer`, `Muxer`, `Filter`. Builder pattern for contexts.
- `Vec<(String, String)>` for metadata (NOT `HashMap` — deterministic ordering for FATE parity).

## Critical Technical Requirements

- SIMD padding: `vec.resize(len + 64, 0)` (matches `AV_INPUT_BUFFER_PADDING_SIZE`)
- Endianness: `u32::from_be_bytes()` / `u32::from_le_bytes()`
- Custom `Rational` with FFmpeg-compatible rounding (not `num-rational`)
- Adler-32 in framecrc: FFmpeg's non-standard init (`s1=0, s2=0`)
- Packet sizes: `bitrate / 8 / 10 / block_align`, rounded down to power of 2

## Reference

- FFmpeg source: `./FFmpeg/` — read C code before writing Rust equivalents
- FATE samples: `./fate-suite/`
- See `H264.md` for decoder status, `CONTRIBUTING.md` for adding codecs/formats
- See `TODO.md` for tasks, `DIVERGENCES.md` for known behavioral differences
