# Wedeo - FFmpeg Rust Rewrite

## Project Overview

A clean-room Rust rewrite of FFmpeg, built incrementally using AI agent pipelines.
**No bindgen, no c2rust, no FFI-based incremental replacement.** Pure Rust from scratch.

The verification target is **bit-for-bit output parity** with FFmpeg's FATE test suite.
The WAV/PCM pipeline has achieved this: wedeo produces byte-identical framecrc output
to FFmpeg 8.0.1 across all PCM formats and all FATE suite WAV samples.

## Philosophy

- **"Make it work, make it right, make it fast."** Correctness first, performance later.
- Accept that the Rust port may be slower initially. Even reproducing C bugs is acceptable for behavioral parity.
- Copy existing FFmpeg comments verbatim where they exist. Don't invent documentation where FFmpeg has none.
- Accept `unsafe` where genuinely required (SIMD, lock-free concurrency, hardware acceleration) but isolate it.

## Workspace Structure

```
wedeo/
  Cargo.toml                    # workspace root (14 crates)
  FFmpeg/                       # C reference source (read-only, not compiled)
  fate-suite/                   # FATE test samples (synced externally)
  start.json                    # original planning conversation
  H264.md                       # H.264 decoder architecture & status (detailed)

  crates/
    wedeo-core/                 # libavutil — Rational, Buffer, Frame, Packet, Metadata, error types
    wedeo-codec/                # libavcodec — Decoder/Encoder traits, registry (inventory crate)
    wedeo-format/               # libavformat — Demuxer/Muxer traits, BufferedIo, InputContext
    wedeo-filter/               # libavfilter — Filter trait, FilterGraph (stub)
    wedeo-resample/             # libswresample — rubato wrapper, interleaved I/O
    wedeo-scale/                # libswscale — dcv-color-primitives wrapper

  adapters/
    wedeo-symphonia/            # Wraps symphonia decoders/demuxers behind wedeo traits (priority 50)

  codecs/
    wedeo-codec-pcm/            # PCM encoder (17 formats); decoders registered via symphonia at pri 100
    wedeo-codec-h264/           # H.264/AVC Baseline decoder — 19 modules, ~15,300 lines (see H264.md)

  formats/
    wedeo-format-wav/           # WAV demuxer + muxer — RIFF/RIFX/RF64/BW64 probe, WAVEFORMATEXTENSIBLE, streaming
    wedeo-format-h264/          # H.264 Annex B raw bitstream demuxer — probe, AU grouping

  bins/
    wedeo-cli/                  # CLI tools: info (like ffprobe), decode, codecs, formats
    wedeo-play/                 # Simple video player (decode + YUV→RGBA + minifb display)

  tests/
    fate/                       # FATE test harness
      src/audiogen.rs           # Bitexact port of FFmpeg's tests/audiogen.c
      src/framecrc.rs           # framecrc output generator (audio passthrough + video decode modes)
      tests/fate_pcm_wav.rs     # Integration tests with FFmpeg cross-validation
      tests/fate_symphonia.rs   # Symphonia adapter tests (priority, lossless bitexact, lossy SNR)
```

## Rust Development Practices

### Build & Check Commands
- Prefer `cargo nextest run` over `cargo test` — provides process isolation, slow test detection, and leak detection
- Run `cargo clippy` to detect warnings/errors and fix them before considering code complete
- Run `cargo fmt` to format code before considering code complete
- Use `cargo check` for fast compilation checks during development
- Use [Conventional Commits](https://www.conventionalcommits.org/) — see CONTRIBUTING.md for types and scopes

### FATE Verification
- Run FATE tests: `FATE_SUITE=./fate-suite cargo nextest run --profile fate -p wedeo-fate` (or `FATE_SUITE=./fate-suite cargo test -p wedeo-fate`)
- Cross-validate audio with FFmpeg: `cargo run --bin wedeo-framecrc -- <file>` vs `ffmpeg -bitexact -i <file> -c copy -f framecrc -`
- Cross-validate video with FFmpeg: `cargo run --bin wedeo-framecrc -- <file>` vs `ffmpeg -bitexact -i <file> -f framecrc -`
- Generate bitexact test WAVs: `cargo run --bin wedeo-audiogen -- output.wav 44100 2`
- The framecrc tool auto-detects: audio uses packet passthrough (checksums raw packets), video uses decode mode (checksums decoded YUV frames)

### Debugging H.264 Differences
- **Read the FFmpeg C code FIRST** — before investigating any wedeo code path, open the corresponding FFmpeg function in `./FFmpeg/` and compare line by line. Key files: `h264_cavlc.c`, `h264idct_template.c`, `h264_mb.c`, `h264_mb_template.c`, `h264_ps.c`.
- **HARD RULE:** After 2 failed hypotheses about pixel diffs, **STOP theorizing**. Extract actual intermediate values from FFmpeg (via lldb or C program) and wedeo (via `--features tracing`). Find **WHERE** the values first diverge before explaining **WHY**. **Time-based backstop:** if 5 minutes pass on the same pixel diff without extracting ground-truth values from BOTH decoders, STOP immediately and use `scripts/ffmpeg_lldb_chroma_dc.py` or lldb directly. The "almost there" feeling is unreliable — each algebraic proof feels cheap but the chain of proofs burns enormous context without progress.
- **Never infer intermediate values from outputs** — don't compute total_zeros from block positions or levels from dequant values. Measure directly via lldb: `breakpoint set -f file.c -l N` → `frame variable` / `expression`.
- **FFmpeg's block layout is transposed** — `h264_slice.c:757` applies `TRANSPOSE()` to the zigzag scan, storing coefficients in column-major order. When reading block memory via lldb, `block[i]` = position `(i%4, i/4)` NOT `(i/4, i%4)`. The IDCT pass order must account for this.
- **FFmpeg's `gb->index` includes NAL header** — 8-bit offset vs wedeo's `br.consumed()` which starts after the NAL header.
- Use `scripts/mb_compare.py` to find differing MBs (checks luma AND chroma U/V), then `--features tracing` trace output for intermediates.
- Use `scripts/framecrc_compare.py --all --no-deblock --pixel-detail` to get a full per-file, per-plane conformance report. This is the authoritative BITEXACT check (not mb_compare alone, which was luma-only until recently).
- Use `scripts/ffmpeg_lldb_chroma_dc.py` to generate lldb scripts for extracting FFmpeg's chroma DC intermediate values at a specific MB/frame/plane.
- Use `ffmpeg -bsf:v trace_headers -f null -` for exact bit-level slice header layout.
- **Build for tracing:** `cargo build --bin wedeo-framecrc -p wedeo-fate --features tracing` (NOT `--features tracing,tracing-detail` without `-p` — that doesn't init the subscriber). Verify traces captured: `wc -l /tmp/trace.log` should be >0.
- **Slice header overrides PPS defaults** — CAVLC must use slice-level `num_ref_idx_l0_active`, not `pps.num_ref_idx_l0_default_active`. Using PPS default when the slice header overrides to fewer refs causes CAVLC bitstream desync (extra bits consumed for ref_idx).
- **DPB `dpb.clear()` deletes `needs_output=false` entries** — when storing a DPB entry then calling `mark_reference` (which calls `dpb.clear()` for IDR), the entry must be protected or the clear must skip it. Otherwise the entry is removed before it can be marked as ShortTerm, leaving the DPB empty for subsequent P-frames.
- **For CAVLC desync bugs**: add `trace!()` at ALL `InvalidData` return sites in `decode_mb_cavlc`, plus at the `decode_macroblock` call site. Run with `RUST_LOG=wedeo_codec_h264::cavlc=trace,wedeo_codec_h264::mb=trace`. One run reveals WHAT failed and WHERE.
- **Never use `eprintln!` for debug traces** — always use `tracing` crate macros (`trace!`, `debug!`) gated by `#[cfg(feature = "tracing")]`. They compile away in release builds without the feature and don't pollute CI output.
- **When existing logs don't reveal the divergence, get better logs** — don't re-read the same trace output hoping to see something new. Options in order of effort: (1) add `trace!()` calls closer to the suspect operation in wedeo, (2) run FFmpeg with more output (`-loglevel debug`, `trace_headers` BSF, `-report`), (3) use `lldb` on FFmpeg to extract ground-truth values at the exact divergence point. Any of these takes less time than further analysis of insufficient data.

### Code Quality
- Fix all clippy warnings unless there's a documented reason not to
- No `#[allow(clippy::*)]` without a comment explaining why
- Prefer safe Rust. When `unsafe` is required, add a `// SAFETY:` comment explaining the invariants
- Use `Result<T, E>` and the `?` operator for error handling, not panics
- Use `wrapping_add`/`wrapping_neg`/`wrapping_mul` for arithmetic that must match C overflow behavior

### Architecture Principles
- Bottom-up dependency order: wedeo-core -> wedeo-codec/wedeo-format -> wedeo-filter -> codec/format implementations -> CLI
- Each FFmpeg library maps to a Rust crate in a workspace
- Codecs/formats/filters are self-contained crates registered via the `inventory` crate
- Use Rust traits for codec/demuxer/muxer abstractions (`Decoder`, `Demuxer`, `Muxer`, `Filter`)
- Use builder pattern for codec contexts (`DecoderBuilder::new(params).open()`)
- Use `Drop` trait / RAII instead of C's `goto cleanup` patterns
- `Vec<(String, String)>` for metadata (NOT `HashMap` — deterministic ordering required for FATE parity)

### Critical Technical Requirements
- All buffer allocations must include SIMD padding: `vec.resize(len + INPUT_BUFFER_PADDING_SIZE, 0)` (64 bytes, matching `AV_INPUT_BUFFER_PADDING_SIZE`)
- Use `u32::from_be_bytes()` / `u32::from_le_bytes()` for endianness — never hardcode byte order
- Custom `Rational` type with FFmpeg-compatible rounding modes (not `num-rational`)
- `rescale_rnd` uses u128 for overflow-resistant timestamp scaling (produces identical results to FFmpeg's manual long-division)
- `reduce()` uses continued fraction algorithm matching `av_reduce` exactly, including the quality check for semi-convergents
- Adler-32 in framecrc uses FFmpeg's non-standard init (`s1=0, s2=0`, not RFC 1950's `s1=1`)
- Packet sizes must match FFmpeg's `ff_pcm_default_packet_size`: `bitrate / 8 / 10 / block_align`, rounded down to power of 2
- Channel layout Display must use FFmpeg's standard names ("stereo", "5.1", etc.) not raw channel lists

### Adding New Codecs/Formats

See [CONTRIBUTING.md](CONTRIBUTING.md) for step-by-step instructions on adding
new codecs and formats.

### Reference
- FFmpeg source is at `./FFmpeg/` for reference — read C code before writing Rust equivalents
- FATE samples are at `./fate-suite/` — test against real files, not just synthetic data
- The original planning conversation is in `start.json`
- FFmpeg codec_id.h values are sequential enums — count positions carefully, don't guess
- FFmpeg's `tests/audiogen.c` is the canonical synthetic audio generator — our port is bitexact

## Current Status (0 clippy warnings)

### H.264 video decoder (native Rust, in progress)

First native video codec. Decodes H.264 Baseline profile I+P frames to YUV420p.
See `H264.md` for detailed architecture, module map, and known issues.

**Decoder** (`codecs/wedeo-codec-h264/`, ~15,400 lines):
- NAL parsing (Annex B + NALFF/avcC), SPS/PPS, slice header with MMCO
- CAVLC entropy decoding (all VLC tables, level/run decode, mb_type parsing)
- 17 intra prediction modes (9 Intra4x4 + 4 Intra16x16 + 4 chroma)
- 4x4/8x8 integer IDCT, luma/chroma DC Hadamard transforms (i32 precision)
- Flat dequantization (spec-equivalent, avoids i16 overflow in FFmpeg's precomputed tables)
- In-loop deblocking filter (boundary strength, strong/normal filtering, luma+chroma)
- Quarter-pel luma MC (6-tap FIR), eighth-pel chroma bilinear
- MV prediction (median, P_SKIP, 16x8/8x16/8x8 sub-partitions), reference list construction with frame_num wrap-around, MMCO, sliding window DPB
- Cross-MB intra4x4 prediction mode tracking (top/left neighbor modes)
- Multi-slice frame support, avcC extradata parsing

**Demuxer** (`formats/wedeo-format-h264/`, ~480 lines):
- Annex B start code scanning, SPS-based probe (score 100)
- Access unit grouping (AUD, SPS, first_mb_in_slice boundaries)
- File extensions: .264, .h264, .h26l, .avc

**FATE Baseline conformance: 15/17 tests bitexact** (2026-03-18):

| Test | Resolution | Types | Status |
|------|-----------|-------|--------|
| BA1_Sony_D, SVA_BA1_B, SVA_NL1_B | 176x144 | I+P | **BITEXACT** |
| BAMQ1_JVC_C | 176x144 | I | **BITEXACT** (per-MB QP) |
| BA_MW_D, BANM_MW_D, AUD_MW_E | 176x144 | I+P | **BITEXACT** |
| BA2_Sony_F | 176x144 | I+P | **BITEXACT** (300 frames) |
| BAMQ2_JVC_C, SVA_BA2_D, SVA_NL2_E | 176x144 | I+P | **BITEXACT** |
| BASQP1_Sony_C | 176x144 | I | **BITEXACT** (QP=0, multi-slice) |
| SVA_Base_B, SVA_FM1_E, SVA_CL1_E | 176x144 | I+P | **BITEXACT** (multi-slice) |
| BA1_FT_C | 352x288 | I+P | 260/299 frames match, late multi-slice diff |
| BA3_SVA_C | 176x144 | I+P+B | B-frames not implemented |

### FFmpeg audio parity via symphonia

wedeo can decode the most common audio formats by wrapping symphonia 0.5.5 behind
wedeo's trait interfaces. Native Rust implementations (priority 100) win over
symphonia wrappers (priority 50) when both exist (e.g., WAV/PCM always uses native).

**Decode coverage (28 decoders):**

| Codec | Source | Verified | Quality |
|-------|--------|----------|---------|
| PCM (17 variants) | Symphonia (pri 100) | FATE bitexact | Byte-identical to FFmpeg 8.0.1 |
| FLAC 16/24-bit | Symphonia | Bitexact vs FFmpeg | Byte-identical |
| WavPack 16/24-bit | Symphonia | Bitexact vs FFmpeg | Byte-identical |
| Vorbis | Symphonia | SNR verified | ~140 dB SNR (float rounding only) |
| AAC (ADTS + M4A) | Symphonia | SNR verified | ~120-134 dB SNR after alignment |
| MP3 | Symphonia | SNR verified | ~128 dB SNR after alignment (gapless trim differs) |
| MP1/MP2 | Symphonia | Registered | Untested |
| Opus | opus-decoder | SNR verified | CELT ~48 dB, SILK ~11-14 dB (crate quality gap) |
| ALAC | Symphonia | Registered | Untested (no FATE samples) |
| ADPCM IMA WAV | Symphonia | Registered | Untested |
| ADPCM MS | Symphonia | Registered | Untested |

**Encode coverage (17 encoders):** All PCM formats, native (`wedeo-codec-pcm`), bitexact roundtrip verified.

**Demux coverage (10 demuxers):**

| Format | Source | Verified |
|--------|--------|----------|
| WAV (RIFF/RIFX/RF64/BW64) | Native | FATE bitexact, 13/13 FATE suite files |
| H.264 Annex B | Native | Probe + decode verified |
| OGG | Symphonia | Probe + Vorbis decode verified |
| FLAC | Symphonia | Probe + decode bitexact verified |
| MP4/M4A | Symphonia | Probe verified |
| MKV/WebM | Symphonia | Probe verified |
| AIFF/AIFC | Symphonia | Probe verified |
| CAF | Symphonia | Probe verified |
| MP3 (ID3v2/sync) | Symphonia | Probe + decode SNR verified |

**Mux coverage (1 muxer):** WAV — bitexact roundtrip (demux → decode → encode → mux → demux = identical).

**Gapless support:** MP3 (via symphonia's LAME/Xing header parsing), AAC in M4A (via iTunSMPB metadata parsing), OGG/Vorbis (via symphonia's gapless mode). Trim applied through `Packet.trim_start`/`trim_end` fields.

**Audio resampling:** `wedeo-resample` wraps `rubato` with Fast/Normal/High quality modes, interleaved I/O, chunked processing.

### Fully implemented and verified
- **WAV demuxer**: RIFF/RIFX/RF64/BW64 probe, all PCM format tags, streaming, truncated file handling. 13/13 FATE suite files bitexact vs FFmpeg 8.0.1.
- **PCM codec**: 17 decoders (via symphonia adapter at priority 100) + 17 native encoders (`wedeo-codec-pcm`). Byte-swapping, unsigned-to-signed conversion, 24→32-bit expansion matching FFmpeg's pcm.c.
- **WAV muxer**: RIFF/WAVE with fmt + data chunks, PCM/float/alaw/mulaw format tags. Roundtrip bitexact.
- **Symphonia audio backend** (`adapters/wedeo-symphonia/`): 10 non-PCM decoder factories (FLAC, MP1, MP2, MP3, AAC, Vorbis, ALAC, WavPack, ADPCM IMA/MS) + 17 PCM decoder factories + 1 Opus (`opus-decoder` crate) + 8 demuxer factories.
- **Core types**: Rational (`av_rescale_rnd`/`av_reduce`), Buffer (Arc CoW + SIMD padding), Frame (32 side data types, video fields), Packet (41 side data types), Metadata (ordered), 36 channel layout names, all CodecId discriminants.
- **Codec registry**: Priority-based `find_decoder`/`find_encoder`. Demuxer `probe()` uses priority as tie-breaker.
- **FATE harness**: Bitexact audiogen, framecrc with FFmpeg-compatible Adler-32, cross-validation tests.

### Framework ready, awaiting implementation
- **wedeo-filter**: Filter trait, FilterGraph skeleton — needs format negotiation and frame queues before use.

### Known architectural gaps
- **Buffer**: No buffer pool, no custom free callback, no guaranteed SIMD alignment.
- **Frame**: Missing opaque user data.
- **Decoder trait**: No get_buffer callback (needed for hw accel / zero-copy).
- **Demuxer trait**: No read_close, no chapters/programs, no find_stream_info equivalent.
- **Filter graph**: No format negotiation, no frame passing mechanism.
- **Error**: No error context (where/why), no codec-specific error codes.

### In progress
- **H.264 Baseline decoder** — I-frame decode pipeline works end-to-end, P-frame inter prediction implemented but not wired. See `H264.md`.

### Not yet started
- Video codecs (HEVC, VP9, AV1, etc.) — native Rust implementations, no existing crate covers these
- Non-WAV muxers (MP4, MKV, etc.)
- Hardware acceleration

### Available infrastructure crates (added as dependencies)
- `v_frame` 0.5.1 — YUV frame buffers from rav1e (21M downloads), 64-byte aligned. For video Frame expansion.
- `av-bitstream` 0.2.1 — Bitstream reader/writer for video codec parsing (exp-golomb, CABAC).
- `yuvxyb` 0.5.0 — Colorspace conversions (YUV ↔ XYB/RGB). For wedeo-scale.

See `TODO.md` for the full task list and `DIVERGENCES.md` for known behavioral differences vs FFmpeg.
