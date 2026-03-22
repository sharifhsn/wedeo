# CABAC Implementation Assessment

## Summary

CABAC (Context-Adaptive Binary Arithmetic Coding) replaces CAVLC as the entropy
decoder for H.264 Main/High profile. Required for playing most real-world MP4
files (phones, cameras, web downloads use High profile with CABAC).

**Estimate: ~4500 lines of Rust, 2-3 focused sessions.**

## Components

| Component | Lines (est.) | Difficulty | Notes |
|-----------|-------------|------------|-------|
| Arithmetic engine (`cabac.rs`) | ~200 | Low | `get_cabac`, `get_cabac_bypass`, `get_cabac_terminate` — 3 functions, pure math on range/offset/state. |
| State tables (`cabac_tables.rs`) | ~2500 | Low (tedious) | LPS range (512B), state transition (128 entries), context init (1024x2 for I, 3x1024x2 for P/B). Use `verify_tables.py`. |
| MB decode (`cabac_decode.rs`) | ~800 | **High** | Every syntax element (mb_type, sub_type, ref_idx, mvd, cbp, residual) has own context model and binarization. Bulk of work. |
| Residual decode | ~400 | Medium | Significance map scanning (different from CAVLC's run/level). DC and non-DC variants. |
| Context init per slice | ~50 | Low | Linear transform of init tables per slice. |
| Integration (decoder.rs) | ~100 | Low | Dispatch CABAC vs CAVLC via `pps.entropy_coding_mode_flag`. Byte-align at slice start. |

## Big Issues

1. **Context state is neighbor-dependent.** Almost every CABAC decode reads
   left/top MB state to select context index (~460 context computations).
   Wrong context index = silent wrong bits, not a crash. This is where most
   bugs will hide.

2. **Binarization is non-trivial.** ~20 different schemes (unary, truncated
   unary, fixed-length, UEGk) across all syntax elements. Each must match
   the spec exactly.

3. **Residual coding is completely different.** CABAC uses significance maps
   (which positions are non-zero) then sign/magnitude, with position-dependent
   contexts. Different from CAVLC's (total_coeff, trailing_ones, levels, runs).

4. **8x8 transform support.** High profile enables `transform_size_8x8_flag`,
   requiring 8x8 IDCT, 8x8 intra prediction, 8x8 dequant scaling. ~500-800
   additional lines on top of CABAC entropy coding.

5. **Testing is harder.** CABAC errors produce subtly wrong values (not desync).
   Debugging requires comparing 1024-entry context state arrays between decoders.

## What Makes It Tractable

- Arithmetic engine is simple and well-specified (~3 core functions).
- MB decode structure mirrors CAVLC closely — same syntax elements, same order,
  same output. Existing mb.rs/mvpred/MC/deblock infrastructure works unchanged.
- FFmpeg's `h264_cabac.c` (2499 lines) is self-contained reference.
- No new `unsafe` needed.
- 49 CABAC conformance files in the FATE suite for verification.

## FFmpeg Reference Files

- `libavcodec/cabac.c` (187 lines) — engine init, table data
- `libavcodec/cabac_functions.h` (221 lines) — `get_cabac`, bypass, terminate
- `libavcodec/h264_cabac.c` (2499 lines) — H.264-specific decode, context init tables
- Compare: wedeo's CAVLC is 2378 lines (cavlc.rs + cavlc_tables.rs)

## FATE Conformance Coverage

- 49 CABAC conformance files (vs 81 CAVLC files, of which 51 are progressive)
- Includes: interlaced (PAFF/MBAFF), High profile, 8x8 transform
- Many share naming convention with CAVLC equivalents (CANL vs NL, CABA vs BA)
