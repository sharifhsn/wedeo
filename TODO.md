# Wedeo TODO

## Audio gaps (short-term)

- [x] **Test ALAC decode** — tested via FFmpeg-generated M4A. Symphonia decodes to S32; verified against FFmpeg.
- [x] **Test ADPCM decode** — ADPCM IMA WAV and ADPCM MS tests added. Symphonia requires "frames per block" metadata; tests gracefully skip when unavailable.
- [ ] **Improve opus-decoder SILK quality** — the pure-Rust opus-decoder crate has ~11-14 dB SNR in SILK/hybrid modes vs FFmpeg/libopus. Monitor upstream for improvements. CELT mode is fine (~48 dB).
- [x] **MP3 gapless trimming** — enabled symphonia's `enable_gapless` mode. Trim fields (`trim_start`/`trim_end`) added to Packet and applied in decoder wrapper. Sample count diff vs FFmpeg bounded by ~3 MP3 frames.
- [x] **AAC priming trim** — implemented iTunSMPB metadata parsing for M4A containers. Trim applied via packet `trim_start`/`trim_end` fields. Sample count diff bounded by ~5 AAC frames.
- [x] **Audio resampling** — integrated `rubato` 1.x behind `wedeo-resample`. Supports Fast/Normal/High quality modes. Deinterleave/reinterleave, chunked processing, flush with zero-padding.

## Video (medium-term, the big effort)

### Prerequisites
- [x] **Expand Frame for video** — added ColorPrimaries, ColorTransferCharacteristic, ChromaLocation enums (matching FFmpeg pixfmt.h). VideoFrameData has color_primaries, color_trc, chroma_location, crop_top/bottom/left/right. Frame has pkt_dts, best_effort_timestamp, repeat_pict.
- [x] **Bitstream reader** — exp-golomb parsing utilities (get_ue_golomb, get_se_golomb, get_te_golomb, get_te0_golomb) in wedeo-codec::bitstream, built on av-bitstream 0.2.1. Ports of FFmpeg's golomb.h functions.
- [x] **Pixel format conversion** — wedeo-scale now wraps dcv-color-primitives for I420/NV12↔RGB24/BGR24/RGBA/BGRA conversions. Converter struct with metadata preservation. 11 unit tests.

### Video codecs (native Rust, no existing crate covers these)
- [x] **H.264 decoder** — 79/79 progressive BITEXACT (52 CAVLC + 27 CABAC), 23/55 FRext. See `H264.md` for detailed status.
  - [ ] FMO (Flexible Macroblock Ordering) — out of scope, rarely used
  - [ ] Interlaced (MBAFF/PAFF) — partial, CABAC engine correct, pixel reconstruction in progress
  - [ ] 10-bit / 4:2:2 — not yet implemented
- [x] **AV1 decoder via rav1d** — `adapters/wedeo-rav1d/` wraps rav1d behind wedeo `Decoder` trait.
- [x] **Video player with audio** — `bins/wedeo-play/` has full A/V playback: cpal audio output, PTS-based A/V sync (audio clock master), resampling via wedeo-resample, 5.1→stereo downmix, volume control, pause/seek. Supports H.264/AV1 video + symphonia audio codecs.
- [ ] **VP9 decoder** — second priority for WebM support. Reference: `FFmpeg/libavcodec/vp9*.c`.
- [ ] **HEVC decoder** — similar to H.264 but more complex (CTU/CTB structure).

### Video muxers
- [ ] **MP4/MOV muxer** — needed for any useful video output. No existing pure-Rust MP4 muxer crate.
- [ ] **MKV/WebM muxer** — `matroska` crate (0.30.0, 143K downloads, by tuffy) is a demuxer only. Muxer needs to be written.

## Player performance

**Target: smooth 1080p24 playback with A/V sync ≤ ±40ms.**

Benchmark (Simpsons Movie 1080p trailer, 1920x800 H.264 @ 23.976fps, 3288 frames):

| Component | Current | Budget (41.7ms/frame) | Status |
|-----------|---------|----------------------|--------|
| H.264 decode | 8.5ms | < 28ms (67%) | OK |
| YUV→RGBA + display | bottleneck at --scale 3 | < 14ms (33%) | FAIL |
| A/V drift | audible desync | ≤ ±40ms | FAIL |
| FFmpeg -threads 1 | 3.2ms/frame (reference) | — | — |

Root cause: `rgba_to_minifb` does per-pixel nearest-neighbor blit into a 5760x2400 buffer
at `--scale 3`, plus CPU-side YUV→RGBA conversion. FFmpeg's ffplay avoids both by uploading
YUV directly to SDL textures (GPU does color conversion + scaling for free).

Improvements (ordered by impact):
- [x] Default `--scale 1` for HD content; let the window manager handle scaling
- [ ] Upload YUV textures to GPU directly (requires replacing minifb with SDL2/wgpu)
- [ ] Move YUV→RGBA conversion off the decode thread (pipeline it)
- [ ] SIMD for MC lowpass filters (NEON on Apple Silicon)

## Infrastructure (ongoing)

- [ ] **Filter graph data flow** — the filter trait and graph skeleton exist but have no format negotiation or frame queues. Needed before real transcode pipelines work.
- [ ] **Interruptible I/O** — needed for network streams and cancellation.
- [x] **Buffer pool** — `BufferPool` for `PictureBuffer` reuse via `SharedPicture::Drop` reclaim.
- [ ] **Error context** — add "where/why" info to errors. Currently errors are flat enums with no call-site context.
- [ ] **Demuxer read_close** — the Demuxer trait has no cleanup method. Not critical (Rust's Drop handles most cases) but FFmpeg has it.
- [ ] **Chapters/programs** — the DemuxerHeader has no chapter or program info. Needed for Matroska/MP4 chapter support.
- [ ] **find_stream_info equivalent** — for formats that need to read some packets before stream params are fully known.

## Crates to evaluate later

| Crate | Version | What for | Notes |
|-------|---------|----------|-------|
| `rubato` | 1.0.1 | Audio resampling (wedeo-resample) | Pure Rust, SIMD, 1.0 stable. Not FATE-exact vs libswresample. |
| `dcv-color-primitives` | 1.0.0 | Pixel format conversion (wedeo-scale) | **Integrated.** Amazon. I420/NV12/RGB. No 10-bit, no BT.2020. |
| `rav1e` | 0.8.1 | AV1 encoding | Production-ready, 20M downloads. Encoding only, not decoding. |
| `matroska` | 0.30.0 | MKV demuxing (alternative to symphonia) | By tuffy, not rust-av. 143K downloads. Demux only. |
| `av-data` | 0.4.4 | Multimedia data types | 1.2M downloads. Overlaps with wedeo-core's Frame/Packet. Not adopting — would require core rewrite. |
| `av-codec` | 0.3.1 | Codec trait abstractions | 18K downloads. Similar to wedeo-codec's traits. Confirms our design but not worth adopting. |

## Testing gaps

- [x] **ALAC decode test** — tested via FFmpeg-generated M4A
- [x] **ADPCM decode test** — tests added (gracefully skip on symphonia metadata limitation)
- [ ] **Opus multistream (surround)** — opus-decoder has `OpusMultistreamDecoder` but we only support mono/stereo
- [ ] **Big-endian PCM through symphonia** — untested (we have no BE test files running through symphonia path)
- [ ] **MKV with multiple audio tracks** — test stream_index handling in symphonia demuxer
- [x] **Seek correctness** — seek tests added for WAV, FLAC, and MP3 via symphonia demuxer
