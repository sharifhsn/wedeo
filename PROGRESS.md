# Wedeo Progress (v2 — FFmpeg `n9.1` "Gauss" Target)

## Architecture & Phase 2 Status (Verified)

- **19-crate workspace (`v0.3.0`, Rust 2024):** 17 product crates + optional `wedeo-gpu-wgpu` + dev-only `wedeo-oracle` (`publish = false`).
- **100% Pure Rust:** Zero `.S` assembly, zero `cc` build scripts, zero `inventory` link-time globals, `#![forbid(unsafe_code)]` across 14 safe crates (with `wedeo-codec-h264` isolating any `core::arch` SIMD in `src/simd.rs`).
- **Strict Audio/Video Separation:** `oracle_arch_lint` verifies audio crates (`wedeo-audio-ops`, `wedeo-codec-pcm`, `wedeo-format-wav`, `wedeo-symphonia`) and audio-only `wedeo` builds pull zero video/GPU crates.
- **L0 `wedeo-core`:** Depends only on `bitflags`; provides VFR `Timestamp`/`Rational`, `BorderedPlane`, `RowProgressPlane`, `BufferPool`, `GpuSurface`, `StreamHealth`, and concrete `Registry`.
- **L1 Ops & Graph:**
  - `wedeo-syntax`: `#![no_std]` CBS/BSF/extradata + `RawStreamDemuxer`.
  - `wedeo-video-ops`: `SwsOpList` optimizer + `256×16` L1/L2 strip-fused `CpuSwsBackend`.
  - `wedeo-audio-ops`: `AudioOpList` + `CompiledAudioOps` (`1024`-sample planar slabs, polyphase resampler, rematrix, dither, circular reframer).
  - `wedeo-graph`: Flat-indexed VFR `CompiledGraph`, `Timeline`, auto-adapter, custom filters.
  - `wedeo-gpu-wgpu`: External `(Arc<Device>, Arc<Queue>)` interop + cached WGSL compute shaders for `SwsOpList` and `WgslFilter`.
- **L2 Codecs, Formats & Adapters:**
  - `wedeo-codec-pcm` + `wedeo-format-wav`: Native PCM (17 formats) and RIFF/RIFX/RF64/BW64 demux + mux.
  - `wedeo-codec-h264`: Pure-Rust CAVLC/CABAC/High-Profile decoder over `BorderedPlane`/`RowProgressPlane`/`wedeo-syntax`.
  - `wedeo-codec-vp9`: Native VP9 decoder (keyframes, inter/altref, odd-dim, Profile 1–3, split-tile parallelism).
  - `wedeo-format-mp4`: Native ISOBMFF/MOV demuxer (`elst`/`colr`/`pasp`/rotation) + muxer (faststart `moov`-first & auto-fMP4 CMAF).
  - `wedeo-format-mkv` + `wedeo-format-ivf`: Complete Matroska/WebM EBML demuxer + muxer, and IVF demuxer + muxer.
  - `wedeo-rav1d`, `wedeo-rav1e`, `wedeo-symphonia`: Segregated AV1 decode/encode and fallback audio decode adapters.
- **L3 Facade & CLI:** `wedeo` (`AudioReader`, `VideoReader`, `default_registry()`) and unified `wedeo` CLI (`inspect`, `transcode`, `cut`, `mux`, `explain`, `wedeo ffmpeg`, `wedeo mcp`).
- **Dev Oracle (`wedeo-oracle`):** `oracle_arch_lint`, `oracle_dod_alloc` (`CountingAlloc` steady-state allocation gates), `framecrc`, `audiogen`, and `checkasm` differential harnesses against pinned FFmpeg `n9.1`.
