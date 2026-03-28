# wedeo

AI-generated Rust rewrite of FFmpeg, verified against FFmpeg's output
bit-for-bit.

**This codebase is AI-generated.** Written by Claude (Anthropic) via
[Claude Code](https://claude.ai/claude-code), directed and reviewed by a
human. The AI reads FFmpeg's C source and reimplements it in Rust. Every
conformance claim below is verified by automated CI on every commit — not
vibes, not benchmarks-on-my-machine, actual bit-for-bit comparison against
FFmpeg's output.

## Status

16 workspace crates. ~47K lines of Rust. 462 tests. 0 clippy warnings.

| Component | Conformance | Notes |
|-----------|-------------|-------|
| **H.264 decode** | **79/79 BITEXACT** | CAVLC + CABAC, Baseline through High profile |
| H.264 FRext | 23/55 bitexact | Progressive 4:2:0 8-bit done; MBAFF/PAFF/10-bit remaining |
| H.264 NEON (aarch64) | 1.75x speedup | MC, IDCT, deblock — FFmpeg's vendored assembly via `cc` |
| WAV demuxer + PCM | bitexact | RIFF/RIFX/RF64/BW64, 17 PCM formats, 13/13 FATE files |
| WAV muxer | bitexact | Roundtrip verified |
| FLAC, WavPack | bitexact | Via symphonia adapter |
| Vorbis, AAC, MP3 | ~120-140 dB SNR | Lossy codecs, float precision only |
| AV1 | bitexact | Via rav1d adapter |
| MP4 demuxer | working | H.264 + AAC tracks |

### H.264 decoder

The H.264 decoder is ~29K lines of Rust across 18 modules. It implements:

- CAVLC and CABAC entropy coding
- All intra prediction modes (4x4, 8x8, 16x16, chroma)
- Quarter-pel motion compensation (6-tap FIR luma, bilinear chroma)
- 4x4 and 8x8 IDCT with Hadamard DC transforms
- In-loop deblocking filter
- MMCO and sliding-window reference management
- Weighted prediction (uni/bi)
- B-frames with direct prediction (spatial + temporal)
- High profile: 8x8 transforms, custom scaling matrices
- MBAFF interlaced (partial — field/frame MB switching, CABAC context adaptation)
- Frame-level threading with wavefront deblocking
- aarch64 NEON assembly for MC, IDCT, and deblocking (feature-gated)

Architecture details: [H264.md](H264.md).
Known FFmpeg behavioral differences: [DIVERGENCES.md](DIVERGENCES.md).

## Quick start

```bash
cargo build
cargo nextest run            # or cargo test
cargo clippy
```

Decode a file and compare against FFmpeg:

```bash
cargo run --release --bin wedeo-framecrc -- input.264
ffmpeg -bitexact -i input.264 -f framecrc -
```

### FATE testing

```bash
./scripts/fetch-fate-suite.sh                    # downloads full suite (~1.2 GB)
FATE_SUITE=./fate-suite cargo nextest run -p wedeo-fate
```

### JVT conformance (ITU test vectors)

204 test vectors from the ITU JVT conformance suite, with MD5 ground truth
from the [Fluster](https://github.com/fluendo/fluster) project. No FFmpeg
required — comparison is against ITU-provided checksums.

```bash
python3 scripts/fetch_jvt.py                     # download vectors (~50 MB)
python3 scripts/suite_runner.py --suite jvt-avc-v1,jvt-fr-ext --format yuv420p
```

The unified suite runner also wraps the FATE suites:

```bash
python3 scripts/suite_runner.py --suite all --format yuv420p
python3 scripts/suite_runner.py --suite fate-cavlc --save-snapshot
python3 scripts/suite_runner.py --suite fate-cavlc --check-snapshot   # regression check
```

## Architecture

```
wedeo/
  crates/
    wedeo-core/          libavutil   — Rational, Buffer, Frame, Packet, errors
    wedeo-codec/         libavcodec  — Decoder/Encoder traits, codec registry
    wedeo-format/        libavformat — Demuxer/Muxer traits, I/O, InputContext
    wedeo-filter/        libavfilter (stub)
    wedeo-resample/      libswresample (rubato)
    wedeo-scale/         libswscale (dcv-color-primitives)
  codecs/
    wedeo-codec-h264/    H.264 decoder — 29K lines, NEON assembly, 55 benchmarks
    wedeo-codec-pcm/     PCM codec — 17 formats
  formats/
    wedeo-format-h264/   H.264 Annex B demuxer
    wedeo-format-wav/    WAV demuxer + muxer
    wedeo-format-mp4/    MP4/MOV demuxer
  adapters/
    wedeo-symphonia/     Wraps symphonia (FLAC, Vorbis, MP3, AAC, WavPack)
    wedeo-rav1d/         Wraps rav1d (AV1)
  bins/
    wedeo-cli/           CLI tool
    wedeo-play/          Video player (decode + display)
  tests/
    fate/                FATE cross-validation harness
  scripts/
    suite_runner.py      Unified conformance runner (FATE + JVT)
    fetch_jvt.py         JVT vector downloader
    conformance_full.py  FATE conformance report
    regression_check.py  Quick regression check
  test_suites/
    h264/                JVT manifest JSONs (tracked)
```

Each FFmpeg library maps to one Rust crate. Codecs and formats register
themselves via `inventory` at link time — no central enum.

## CI

Four parallel jobs run on every PR:

- **Lint** — clippy + rustfmt
- **Test** — 462 unit and integration tests via nextest
- **FATE** — 79 H.264 conformance files, framecrc comparison vs system FFmpeg
- **JVT** — 204 ITU vectors, MD5 comparison (no FFmpeg required)

Pre-commit hook available: `pip install pre-commit && pre-commit install`

## FFmpeg reference

The `FFmpeg/` submodule is pinned to `n8.0.1`. Optional — only needed to read
the C source or build a debug FFmpeg for development:

```bash
git submodule update --init
cd FFmpeg && ./configure --disable-optimizations --enable-debug=3 \
  --disable-stripping --disable-asm && make -j$(nproc) ffmpeg
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

### For AI agents

This repo is designed for AI-assisted development. [CLAUDE.md](CLAUDE.md)
contains the full project context: architecture, conventions, debugging
procedures, and technical requirements. It is the canonical reference for any
AI agent working on this codebase — read it before writing code.

The [llms.txt](llms.txt) file provides a machine-readable project summary
following the [llms.txt convention](https://llmstxt.org/).

## License

LGPL-2.1-or-later. See [LICENSE](LICENSE) and [COPYING.LGPLv2.1](COPYING.LGPLv2.1).
