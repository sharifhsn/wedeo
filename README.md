# wedeo

**Rust Universal A/V Adapter** — a deterministic, VFR-native, data-oriented media compiler and runtime combining exact algorithmic ports of **FFmpeg `n9.1` ("Gauss")** with optional `wgpu` WGSL GPU compute acceleration.

- **Codec implementations:** Wedeo implements H.264 and VP9 in Rust with architecture-specific intrinsics. The optional rav1d AV1 adapter uses a dav1d-compatible FFI boundary and enables rav1d’s assembly optimizations.
- **Strong Audio / Video Separation:** Audio-only builds (`wedeo` with `--no-default-features --features "audio-ops,pcm,wav,symphonia"`) pull zero video decoders, zero `SwsOpList`, and zero `wgpu`.
- **VFR-Native Everywhere:** Exact rational `(pts, dts, duration, time_base)` on every `Packet`, `AudioBlock`, and `VideoFrame`.
- **Data-Oriented Design (DOD):** `256×16` L1/L2 strip-fused `SwsOpList`, `1024`-sample planar `AudioOpList` slabs, warm `BufferPool`, `BorderedPlane` edge extension, and flat-indexed `CompiledGraph`.
- **Unified CLI & MCP:** One `wedeo` binary (`inspect`, `transcode`, `cut`, `mux`, `explain`, `wedeo ffmpeg`, `wedeo mcp`).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the locked v2 architectural specification.

## Workspace Layout (19 Crates)

| Layer | Crates | Role |
|---|---|---|
| **L0** | `crates/wedeo-core` | Lean foundation (`bitflags` only): VFR `Timestamp`/`Rational`, `BorderedPlane`, `RowProgressPlane`, `BufferPool`, `GpuSurface`, `StreamHealth`, traits, concrete `Registry` |
| **L1** | `crates/wedeo-syntax` | `#![no_std]`, `#![forbid(unsafe_code)]` CBS/BSF/extradata + `RawStreamDemuxer` |
| **L1** | `crates/wedeo-video-ops` | Pure-Rust FFmpeg `n9.1` `SwsOpList` optimizer + `256×16` strip-fused `CpuSwsBackend` |
| **L1** | `crates/wedeo-audio-ops` | `AudioOpList` + `CompiledAudioOps` (`libswresample` polyphase/rematrix/dither, `1024`-sample slabs) |
| **L1** | `crates/wedeo-graph` | Flat-indexed VFR-native `CompiledGraph`, `Timeline`, auto-adapter, custom filter traits |
| **L1b** | `crates/wedeo-gpu-wgpu` | Optional `wgpu` backend compiling `SwsOpList` + `WgslFilter` into cached WGSL compute shaders |
| **L2a** | `codecs/wedeo-codec-{pcm,h264,vp9}` | Native pure-Rust PCM (17 formats), H.264/AVC, and VP9 decoders |
| **L2a** | `formats/wedeo-format-{wav,mp4,mkv,ivf}` | Native WAV, ISOBMFF/MP4 (faststart + auto-fMP4), Matroska/WebM EBML, and IVF demuxers + muxers |
| **L2b** | `adapters/wedeo-{rav1d,rav1e,symphonia}` | Segregated AV1 decode/encode and fallback audio decode adapters |
| **L3** | `crates/wedeo`, `bins/wedeo-cli` | Ergonomic library facade (`AudioReader`, `VideoReader`) and unified `wedeo` CLI/MCP binary |
| **Dev** | `tests/wedeo-oracle` | Non-published FFmpeg `n9.1` oracle (`framecrc`, `audiogen`, `checkasm`, `CountingAlloc`, DAG lints) |

## Quick Start

```bash
# Build workspace
cargo build --offline

# Run unit & oracle tests
cargo test --offline -p wedeo-oracle --test oracle_arch_lint --test oracle_dod_alloc

# Unified CLI
cargo run --offline -p wedeo-cli -- inspect input.mp4
cargo run --offline -p wedeo-cli -- transcode input.mp4 output.mkv
```

### Hardware video conversion

Build the CLI with the optional wgpu backend and request it for scaling or color conversion:

```bash
cargo build --release -p wedeo-cli --features gpu-wgpu
target/release/wedeo transcode -i input.ivf --video av1 --size 640x360 --gpu -o output.ivf
```

The conversion backend uploads decoded frames, runs the conversion shader, and reads the
result back for the encoder. The CLI reports the adapter name and device type on stderr.
A CPU adapter indicates software rendering. Decode and encode use their usual codecs.
If no backend is available, the transcode report includes the CPU fallback reason.

### Audio-Only Library Build (Zero Video / Zero GPU)

```toml
[dependencies]
wedeo = { version = "0.3.0", default-features = false, features = ["audio-ops", "pcm", "wav", "symphonia"] }
```

## License

`LGPL-2.1-or-later`. See [LICENSE](LICENSE) and [COPYING.LGPLv2.1](COPYING.LGPLv2.1).
