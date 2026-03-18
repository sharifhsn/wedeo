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
    wedeo-resample/             # libswresample (stub)
    wedeo-scale/                # libswscale (stub)

  adapters/
    wedeo-symphonia/            # Wraps symphonia decoders/demuxers behind wedeo traits (priority 50)

  codecs/
    wedeo-codec-pcm/            # PCM decoder + encoder — S16LE/BE, S24LE/BE, S32LE/BE, U8/16/24/32, F32/F64, alaw, mulaw
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

## Current Status (0 clippy warnings)

### H.264 video decoder (native Rust, in progress)

First native video codec. Decodes H.264 Baseline profile I-frames to YUV420p.
See `H264.md` for detailed architecture, module map, and known issues.

**Decoder** (`codecs/wedeo-codec-h264/`, ~15,300 lines):
- NAL parsing (Annex B + NALFF/avcC), SPS/PPS, slice header with MMCO
- CAVLC entropy decoding (all VLC tables, level/run decode, mb_type parsing)
- 17 intra prediction modes (9 Intra4x4 + 4 Intra16x16 + 4 chroma)
- 4x4/8x8 integer IDCT, luma/chroma DC Hadamard transforms (i32 precision)
- Flat dequantization (spec-equivalent, avoids i16 overflow in FFmpeg's precomputed tables)
- In-loop deblocking filter (boundary strength, strong/normal filtering, luma+chroma)
- Quarter-pel luma MC (6-tap FIR), eighth-pel chroma bilinear (ready, not wired)
- MV prediction, reference list construction, MMCO, DPB (ready, not wired)
- Cross-MB intra4x4 prediction mode tracking (top/left neighbor modes)
- Multi-slice frame support, avcC extradata parsing

**Demuxer** (`formats/wedeo-format-h264/`, ~480 lines):
- Annex B start code scanning, SPS-based probe (score 100)
- Access unit grouping (AUD, SPS, first_mb_in_slice boundaries)
- File extensions: .264, .h264, .h26l, .avc

**FATE Baseline conformance: 1/9 tests bitexact** (2026-03-17):

| Test | Resolution | QP | Types | Status |
|------|-----------|-----|-------|--------|
| BA1_Sony_D | 176x144 | 28 | I | **BITEXACT** (17 frames) |
| BAMQ1_JVC_C | 176x144 | 24 | I | 71 dB (±1-4 diffs, per-MB QP variation) |
| BASQP1_Sony_C | 176x144 | 0 | I | 5.8 dB (QP=0 dequant issue) |
| BA1_FT_C | 352x288 | 30 | I | 3.7 dB (different resolution) |
| BA_MW_D, BA2_Sony_F, BAMQ2, BANM, BA3 | 176x144 | 24-28 | I+P/B | P/B-frame decode not wired |

### FFmpeg audio parity via symphonia

wedeo can decode the most common audio formats by wrapping symphonia 0.5.5 behind
wedeo's trait interfaces. Native Rust implementations (priority 100) win over
symphonia wrappers (priority 50) when both exist (e.g., WAV/PCM always uses native).

**Decode coverage (28 decoders):**

| Codec | Source | Verified | Quality |
|-------|--------|----------|---------|
| PCM (17 variants) | Native | FATE bitexact | Byte-identical to FFmpeg 8.0.1 |
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

**Encode coverage (17 encoders):** All PCM formats, native, bitexact roundtrip verified.

**Demux coverage (9 demuxers):**

| Format | Source | Verified |
|--------|--------|----------|
| WAV (RIFF/RIFX/RF64/BW64) | Native | FATE bitexact, 13/13 FATE suite files |
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
- **WAV demuxer**: All PCM format tags (0x0001 PCM, 0x0003 IEEE float, 0x0006 A-law, 0x0007 mu-law, 0xFFFE extensible). Streaming support (data_size=0xFFFFFFFF). Post-data metadata scanning. Truncated file handling. RIFX/RF64/BW64 probe detection. 13/13 FATE suite PCM WAV files pass bitexact vs FFmpeg 8.0.1.
- **PCM decoder + encoder**: 17 registered decoders + 17 registered encoders covering S16LE/BE, S24LE/BE, S32LE/BE, U8, U16LE/BE, U24LE/BE, U32LE/BE, F32LE/BE, F64LE/BE. Byte-swapping, unsigned-to-signed conversion, 24-bit-to-32-bit expansion all matching FFmpeg's pcm.c DECODE macro behavior. Encoder is the exact reverse (encode_samples).
- **WAV muxer**: Writes standard RIFF/WAVE files with fmt + data chunks. Supports PCM, IEEE float, A-law, mu-law format tags. Patches RIFF/data sizes in trailer on seekable outputs. Roundtrip (demux → decode → encode → mux) is bitexact.
- **Symphonia audio backend** (`adapters/wedeo-symphonia/`): Wraps symphonia 0.5.5 decoders and demuxers behind wedeo's traits. 8 decoder factories (FLAC, MP3, AAC, Vorbis, ALAC, WavPack, ADPCM IMA WAV, ADPCM MS) and 8 demuxer factories (OGG, FLAC, MP4, MKV, AIFF, CAF, MP3, WAV). I/O bridge transfers ownership via `BufferedIo::take_inner()`. Channel layout, metadata, and sample format conversion modules.
- **Core types**: Rational (with FFmpeg-exact `av_rescale_rnd` and `av_reduce`), Buffer (Arc-based CoW with SIMD padding), Frame (with side data matching all 32 `AVFrameSideDataType` values, video fields: ColorPrimaries/ColorTransferCharacteristic/ChromaLocation enums matching FFmpeg pixfmt.h, crop rect, pkt_dts, best_effort_timestamp, repeat_pict), Packet (with 41 side data types), Metadata (ordered, case-insensitive), 36 standard channel layout names, all CodecId discriminants verified against FFmpeg headers.
- **Codec registry**: Priority-based selection — `find_decoder`/`find_encoder` pick the highest-priority factory when multiple implementations exist. Demuxer `probe()` uses priority as tie-breaker.
- **Codec options**: `CodecOptions` type with typed getters (i64/f64/bool), threaded through CodecParameters → DecoderBuilder → factory. CodecParameters also carries `thread_count` and `codec_tag`.
- **OutputContext**: High-level muxer wrapper (create → write_packet → finish) mirroring InputContext.
- **FATE harness**: Bitexact audiogen (byte-identical to FFmpeg's tests/audiogen.c), framecrc generator with FFmpeg-compatible Adler-32, cross-validation integration tests, lossless bitexact tests, lossy SNR tests.

### Framework ready, awaiting implementation
- **wedeo-codec**: Decoder/Encoder traits, DecoderBuilder with codec-private options, inventory-based registry, bitstream utilities (exp-golomb parsing via av-bitstream) — ready for new codecs.
- **wedeo-format**: Demuxer/Muxer traits, BufferedIo (with read/write buffering, typed methods, take_inner), InputContext/OutputContext, protocol layer — ready for new formats.
- **wedeo-filter**: Filter trait, FilterGraph skeleton — needs format negotiation and frame queues before use.
- **wedeo-resample**: Wraps `rubato` for sample rate conversion. Fast (polynomial), Normal (sinc 128pt), High (sinc 256pt) quality modes. Handles deinterleave/reinterleave and chunked processing.
- **wedeo-scale**: Pixel format converter wrapping `dcv-color-primitives`. Supports I420/NV12↔RGB24/BGR24/RGBA/BGRA and RGB↔BGRA conversions. Converter struct preserves frame metadata through conversion.

### Known architectural gaps (from adversarial review)
- **Buffer**: No buffer pool (needed for high-throughput decode), no custom free callback (needed for hw accel / zero-copy). `Vec<u8>` doesn't guarantee SIMD alignment.
- **Frame**: Missing opaque user data. color_primaries, color_trc, chroma_location, crop rect, pkt_dts, best_effort_timestamp, repeat_pict are now implemented.
- **Decoder trait**: No get_buffer callback (needed for hw accel / zero-copy buffer allocation).
- **Demuxer trait**: No read_close, no chapters/programs, no find_stream_info equivalent.
- **Filter graph**: No format negotiation, no frame passing mechanism (queues, push/pull). The stub topology is correct but the data flow is missing.
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
