# Plan: rav1d AV1 Adapter

## Goal
Wrap the `rav1d` crate (pure Rust AV1 decoder, port of dav1d) behind wedeo's
`Decoder` trait, following the same pattern as `adapters/wedeo-symphonia/`.

## Deliverables
1. New crate `adapters/wedeo-rav1d/` with Decoder + DecoderFactory
2. Workspace member added to root `Cargo.toml`
3. `use wedeo_rav1d as _;` in `wedeo-cli` and `wedeo-fate`
4. FATE test comparing AV1 decode vs FFmpeg (framecrc or SNR)
5. Zero clippy warnings, cargo fmt clean

## Architecture

```
adapters/wedeo-rav1d/
  Cargo.toml          # depends on wedeo-core, wedeo-codec, rav1d, inventory
  src/lib.rs          # DecoderFactory + Decoder impl
```

### Key design decisions
- **Priority 50** (adapter, not native) — same as symphonia
- **CodecId::Av1** already exists in wedeo-core
- **Pixel format**: rav1d outputs planar YUV (8/10/12-bit). Map to wedeo's Frame.
  For 8-bit: direct copy to Y/U/V planes. For 10-bit+: store as-is, wedeo-scale
  doesn't handle 10-bit yet so just preserve the data.
- **Extradata**: AV1 in MP4 uses av1C (AV1CodecConfigurationRecord). rav1d handles
  OBU parsing internally, but the demuxer needs to pass sequence header OBUs.

### Decoder trait mapping
```
send_packet(Packet) → rav1d::Decoder::send_data(data)
receive_frame()     → rav1d::Decoder::get_picture() → convert to wedeo Frame
flush()             → rav1d::Decoder::flush()
```

## Reference files to read FIRST
1. `adapters/wedeo-symphonia/src/decoder.rs` — pattern for wrapping external decoder
2. `rav1d` crate docs — `Decoder::send_data()`, `Decoder::get_picture()`, `Picture` type
3. `crates/wedeo-codec/src/decoder.rs` — Decoder trait definition
4. `CONTRIBUTING.md` — "Adding a new codec" section

## FATE test approach
- Use `fate-suite/av1-conformance/` or generate a small AV1 file:
  `ffmpeg -f lavfi -i testsrc=size=176x144:rate=25:duration=1 -c:v libsvtav1 -preset 8 test.mp4`
- Compare: `cargo run --bin wedeo-framecrc -- test.mp4` vs `ffmpeg -bitexact -i test.mp4 -f framecrc -`
- AV1 decode is deterministic (integer-only spec), so bitexact is achievable

## Conflict zones
- **Root Cargo.toml**: Adding workspace member. Do this FIRST as a standalone commit.
- **wedeo-cli/src/main.rs**: Adding `use wedeo_rav1d as _;`
- **wedeo-fate/Cargo.toml**: Adding dependency

## Estimated size
~200-400 lines of Rust. The heavy lifting is in rav1d; this is just the trait bridge.

## Verification
```bash
cargo check --workspace
cargo clippy --workspace
cargo nextest run
# AV1-specific:
cargo run --bin wedeo-framecrc -- test_av1.mp4
ffmpeg -bitexact -i test_av1.mp4 -f framecrc -
diff <(cargo run --bin wedeo-framecrc -- test_av1.mp4) <(ffmpeg -bitexact -i test_av1.mp4 -f framecrc -)
```
