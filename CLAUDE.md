# Wedeo — FFmpeg `n9.1` ("Gauss") Pure-Rust Rewrite

## Project Overview

Pure-Rust Universal A/V Adapter verified against **FFmpeg `n9.1` ("Gauss")**.
**100% Pure Rust:** No `.S` files, no `cc` `build.rs`, no `bindgen`, no `c2rust`, no FFI, no `inventory`.
Licensed under **LGPL-2.1-or-later**.
Canonical architecture spec: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Architecture (19 Crates)

- **L0 (`wedeo-core`):** Depends ONLY on `bitflags`. VFR `Timestamp`/`Rational`, `BorderedPlane`, `RowProgressPlane`, `BufferPool`, `GpuSurface`, `StreamHealth`, concrete `Registry`.
- **L1:** `wedeo-syntax` (`#![no_std]`, `#![forbid(unsafe_code)]`), `wedeo-video-ops` (`SwsOpList` + `256×16` `CpuSwsBackend`), `wedeo-audio-ops` (`AudioOpList` + `1024`-sample `CompiledAudioOps`), `wedeo-graph` (`CompiledGraph`, `Timeline`, filters), `wedeo-gpu-wgpu` (optional `wgpu` WGSL compute backend).
- **L2a:** `wedeo-codec-{pcm,h264,vp9}`, `wedeo-format-{wav,mp4,mkv,ivf}`.
- **L2b:** `wedeo-{rav1d,rav1e,symphonia}` (segregated adapters).
- **L3:** `wedeo` (facade with `AudioReader`/`VideoReader`), `wedeo-cli` (`inspect`/`transcode`/`cut`/`mux`/`explain`, `wedeo ffmpeg`, `wedeo mcp`).
- **Dev-Only (`tests/wedeo-oracle`, `publish = false`):** FFmpeg `n9.1` oracle, `framecrc`, `audiogen`, `checkasm`, `CountingAlloc` DOD gates, and DAG lints.

## Hard Architectural Rules

- **Strict Downward DAG & A/V Separation:** Audio crates (`wedeo-audio-ops`, `wedeo-codec-pcm`, `wedeo-format-wav`, `wedeo-symphonia`) must NEVER depend on video or GPU crates. `wedeo-video-ops` must NEVER depend on `wgpu` or `wedeo-gpu-wgpu`.
- **Safety:** `#![forbid(unsafe_code)]` on all safe crates (`wedeo-codec-h264` uses `#![deny(unsafe_code)]` with audited `core::arch` SIMD isolated to `src/simd.rs`).
- **VFR-Native:** Always carry exact rational `(pts, dts, duration, time_base)`; never assume CFR.
- **DOD Steady-State Zero-Alloc:** Reuse `BufferPool`, pre-allocated scratch slabs, and in-place `BorderedPlane::extend_edges`.
- **Zero Dev Slop in Production:** Keep all FFmpeg oracle binaries, `framecrc`, `CountingAlloc`, and stage dumps (`#[cfg(feature = "dev-trace")]`) out of production APIs.

## Build & Verify

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --offline -p wedeo-oracle --test oracle_arch_lint --test oracle_dod_alloc
```

## Debugging & Verification Workflow

- Read FFmpeg `n9.1` C source in `./FFmpeg/` before modifying codec/ops math; use explicit `wrapping_*` arithmetic for C `-fwrapv` parity.
- Never infer intermediate values from outputs — use `tracing` (or `dev-trace` feature) and differential `framecrc` / `audiogen` / `checkasm` comparisons against FFmpeg `n9.1`.
- Preserve deterministic ordering (`Vec<(String, String)>` for metadata, never `HashMap`).
