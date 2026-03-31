# Wedeo Progress

## Current Status (2026-03-30)

- H.264 CAVLC: **52/52** progressive conformance files BITEXACT (100%)
- H.264 CABAC: **27/27** progressive conformance files BITEXACT (100%)
- H.264 FRext: **23/55 BITEXACT** — all progressive High Profile 4:2:0 8-bit files pass. Remaining are interlaced (MBAFF/PAFF), 10-bit, or 4:2:2.
- H.264 MBAFF: CABAC engine 100% correct, reconstruction partial (deblocking + inter prediction remain)
- WAV/PCM pipeline: byte-identical to FFmpeg 8.0.1 across all FATE suite samples
- Audio via symphonia: 28 decoders, 10 demuxers, SNR-verified lossy codecs
- **Video player (v0.1.1):** GPU-accelerated via wgpu+winit, 24fps 0-drop 1080p playback, ffplay-style A/V sync

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

- Seek reimplementation — ffplay serial mechanism for stale frame discard
- SIMD for MC lowpass filters (NEON) — primary target, 3x slower at qpel(2,2) vs (0,0)
- SIMD for deblock filter (NEON)
- SIMD for IDCT (NEON) — 8x8 butterfly is 4.7x slower than 4x4
- Interlaced (PAFF) support
- Additional video codecs (HEVC, VP9)
