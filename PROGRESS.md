# Wedeo Progress

## Current Status (2026-04-05)

- H.264 CAVLC: **52/52** progressive conformance files BITEXACT (100%)
- H.264 CABAC: **27/27** progressive conformance files BITEXACT (100%)
- H.264 FRext: **23/55 BITEXACT** — all progressive High Profile 4:2:0 8-bit files pass. Remaining are interlaced (MBAFF/PAFF), 10-bit, or 4:2:2.
- H.264 MBAFF: CABAC engine 100% correct, reconstruction partial (deblocking + inter prediction remain)
- VP9 keyframes: **64/64** standard-size keyframes BITEXACT (100%)
- VP9 inter frames: **11/64** quantizer test vectors BITEXACT (both frames); 53 have small pixel diffs (remaining MC/reconstruction bugs, bool decoder is correct)
- WAV/PCM pipeline: byte-identical to FFmpeg 8.0.1 across all FATE suite samples
- Audio via symphonia: 28 decoders, 10 demuxers, SNR-verified lossy codecs
- **Video player (v0.1.1):** GPU-accelerated via wgpu+winit, 24fps 0-drop 1080p playback, ffplay-style A/V sync

## VP9 Decoder (2026-04-05)

- Native Rust, 12 modules, ~6K lines
- IVF + WebM demuxers
- Bool arithmetic decoder, frame header parsing, block-level partition/mode/coefficient decoding
- Full intra prediction (10 modes + 5 fallbacks), IDCT (4/8/16/32 + IADST), loop filter
- Inter prediction: MV parsing/prediction (spatial+temporal), 8-tap subpel MC, compound prediction, reference frame management
- 24 bugs fixed across 4 debugging sessions (2026-04-03 through 2026-04-05)
- Bool decoder symbol counts verified identical to FFmpeg across all 927 inter-frame blocks
- Deferred: probability adaptation (adapt_probs), scaled prediction, odd-size frames, 10/12-bit, superframe parsing, multi-tile parallelism, SIMD

### VP9 Bugs Fixed (2026-04-05 session)

1. **Inter-frame mode context indexing** — y_mode arrays used keyframe 2-per-8×8 indexing for inter frames; FFmpeg uses 1-per-8×8. Fixed reads in decode_inter_mode (sub-8x8, >=8x8, filter ctx) and writes in update_ctx.
2. **end_x_y/end_y_y edge clamping** — coefficient decode edge clamp doubled remaining count; `cols_4x4 - col` is already in NNZ units, no `* 2` needed.
3. **MV clamping bounds** (prior session) — `BWH_TAB[1]` → `BWH_TAB[0]` for bw4/bh4.
4. **MC emu_edge offset** (prior session) — removed erroneous +3 column/row offset from emu buffer src pointer.

## Threading Benchmark (Phase 7 — BBB 1080p 10s, 10 runs)

| Config | Wall | User | System |
|--------|------|------|--------|
| wedeo single-threaded | 18.1s ±1.2s | 16.7s | 0.59s |
| wedeo Phase 7 (auto) | 12.8s ±1.4s | 20.0s | 0.56s |
| FFmpeg -threads 1 | 1.50s ±0.09s | 1.74s | 0.04s |
| FFmpeg -threads 0 | 0.80s ±0.11s | 3.75s | 0.20s |

Threading speedup: wedeo **1.42x**, FFmpeg **1.87x**.
Single-threaded gap: wedeo is **12x slower** than FFmpeg (scalar Rust vs C+NEON).

### Profile Breakdown (active CPU, BBB 1080p)

| Subsystem | % CPU | Notes |
|-----------|-------|-------|
| Motion Compensation | 47.8% | `extract_ref_block` 14.8%, lowpass filters 17% |
| Memory alloc/free | 16.2% | Per-call `vec![]` in mc_luma/mc_chroma |
| MB decode + recon | 11.7% | apply_mc_bi_partition 6.8% |
| Deblocking | 10.8% | filter_mb_edge_luma 3.3% |
| memcpy/memmove | 5.8% | System-optimized |
| CABAC | 1.8% | Inherently serial |

## Next Steps

- VP9 inter frame conformance — remaining pixel diffs likely from MC reconstruction bugs
- VP9 probability adaptation — `adapt_probs` not yet wired up between frames
- VP9 odd-size frame conformance — 39 keyframes with non-8-aligned dimensions mismatch
- Seek reimplementation — ffplay serial mechanism for stale frame discard
- SIMD for MC lowpass filters (NEON) — primary target, 3x slower at qpel(2,2) vs (0,0)
- Interlaced (PAFF) support
- Additional video codecs (HEVC)
