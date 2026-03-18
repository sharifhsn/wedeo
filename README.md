# wedeo

Clean-room Rust rewrite of FFmpeg, built incrementally with AI agent pipelines.

**No bindgen, no c2rust, no FFI.** Pure Rust from scratch.

## Transparency

This codebase was written primarily by AI (Claude) through
[Claude Code](https://claude.ai/claude-code). A human provides architectural
direction, verification targets, and review. The AI writes the implementation
by reading FFmpeg's C source as reference and reimplementing in idiomatic Rust.

The verification target is **bit-for-bit output parity** with FFmpeg's
[FATE](https://fate.ffmpeg.org/) test suite.

## Status

14 workspace crates, 0 clippy warnings.

### What works

| Component | Status | Detail |
|-----------|--------|--------|
| WAV demuxer | Bitexact | RIFF/RIFX/RF64/BW64, 13/13 FATE suite files |
| PCM codec (17 formats) | Bitexact | S16/24/32 LE/BE, U8/16/24/32, F32/F64, alaw, mulaw |
| WAV muxer | Bitexact | Roundtrip verified (demux -> decode -> encode -> mux) |
| FLAC decode | Bitexact | Via symphonia, 16/24-bit |
| WavPack decode | Bitexact | Via symphonia, 16/24-bit |
| Vorbis decode | ~140 dB SNR | Float precision only |
| AAC decode | ~120-134 dB SNR | ADTS + M4A, gapless trim |
| MP3 decode | ~128 dB SNR | Gapless trim via LAME/Xing headers |
| H.264 Baseline I-frame | 1/9 bitexact | BA1_Sony_D bitexact, others in progress |

### Architecture

```
wedeo/
  crates/
    wedeo-core/          # libavutil  — Rational, Buffer, Frame, Packet, error types
    wedeo-codec/         # libavcodec — Decoder/Encoder traits, registry
    wedeo-format/        # libavformat — Demuxer/Muxer traits, BufferedIo, InputContext
    wedeo-filter/        # libavfilter (stub)
    wedeo-resample/      # libswresample (rubato wrapper)
    wedeo-scale/         # libswscale (dcv-color-primitives wrapper)
  adapters/
    wedeo-symphonia/     # Wraps symphonia decoders/demuxers (priority 50)
  codecs/
    wedeo-codec-pcm/     # Native PCM codec (priority 100)
    wedeo-codec-h264/    # Native H.264 decoder (~15,300 lines)
  formats/
    wedeo-format-wav/    # Native WAV demuxer + muxer
    wedeo-format-h264/   # H.264 Annex B demuxer
  bins/
    wedeo-cli/           # CLI: info, decode, codecs, formats
    wedeo-play/          # Simple video player (decode + display)
  tests/
    fate/                # FATE test harness with FFmpeg cross-validation
```

For detailed architecture, conventions, and codec implementation guidelines,
see [CLAUDE.md](CLAUDE.md). For H.264 decoder internals, see [H264.md](H264.md).
For known behavioral differences vs FFmpeg, see [DIVERGENCES.md](DIVERGENCES.md).

## Building

```bash
cargo build
cargo test
cargo clippy
```

### Using nextest (recommended)

[cargo-nextest](https://nexte.st/) provides process-per-test isolation, slow test
detection, and leak detection. Profiles are configured in `.config/nextest.toml`.

```bash
cargo install cargo-nextest --locked
cargo nextest run                              # local dev (fail-fast, full parallelism)
cargo nextest run --profile ci                 # CI mode (collects all failures, JUnit XML)
cargo nextest run -p wedeo-codec-h264          # single crate
FATE_SUITE=./fate-suite cargo nextest run --profile fate -p wedeo-fate
cargo nextest list                             # list all tests
```

## FATE testing

FATE tests require sample files from FFmpeg's FATE server:

```bash
./scripts/fetch-fate-suite.sh        # downloads ~1.2 GB of test samples
FATE_SUITE=./fate-suite cargo nextest run --profile fate -p wedeo-fate
```

Cross-validate against FFmpeg:

```bash
# Audio (packet passthrough)
cargo run --bin wedeo-framecrc -- input.wav
ffmpeg -bitexact -i input.wav -c copy -f framecrc -

# Video (decode mode)
cargo run --bin wedeo-framecrc -- input.264
ffmpeg -bitexact -i input.264 -f framecrc -
```

## FFmpeg reference source

The `FFmpeg/` directory is a git submodule pinned to the `n8.0.1` tag. It is
optional — only needed if you want to read FFmpeg's C source for reference.

```bash
git submodule update --init
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions welcome from both humans
and AI agents.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option.
