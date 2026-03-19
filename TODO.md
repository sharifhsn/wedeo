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
- [~] **H.264 decoder** — 39/57 progressive CAVLC conformance BITEXACT, all 17 Baseline pass. See `H264.md` for detailed status.
  - [x] Wire P-frame inter prediction (mb_skip_run + P_SKIP + coded P-MB types 0-4)
  - [x] Fix demuxer access unit grouping (SPS/AUD/first_mb_in_slice boundaries)
  - [x] Write FATE integration tests (4 bitexact + 4 frame count regression tests)
  - [x] Add per-MB debug infrastructure (tracing in mb.rs, scripts/mb_compare.py)
  - [x] Fix BAMQ1 ±1 pixel diffs — IDCT pass order was row-major column-first instead of row-first
  - [x] Fix multi-slice OOB panic — skip run bounds check prevents mb_y >= mb_height
  - [x] Fix DPB IDR marking — current entry survives dpb.clear() to be marked ShortTerm
  - [x] Fix CAVLC ref_idx desync — use slice header's num_ref_idx_l0_active, not PPS default
  - [x] Fix ref list frame_num wrap-around — pic_num with MaxFrameNum per spec 8.2.4.1
  - [x] Fix MV neighbor C availability — only use C from already-decoded MBs
  - [x] Fix slice loop early-exit — parse mb_skip_run before RBSP exhaustion check
  - [x] Fix intra4x4 neighbor modes from inter MBs — store DC_PRED(2) not -1 (7 files)
  - [x] Fix multi-slice neighbor availability — slice_table + per-MB top_available (BASQP1 + I-frames)
  - [x] Fix slice-aware MV prediction — cross-slice MV neighbors unavailable
  - [x] Fix RBSP exhaustion margin — 8→1 bit, prevents dropping last MB of slice (SVA_Base_B/FM1_E/CL1_E)
  - [x] Fix BA1_FT_C luma — MV neighbor C/D slice check for all blk_y values
  - [x] Fix chroma V ±1 rounding — CHROMA_QP_TABLE had transcription error at index 36 (extra 33)
  - [x] Fix deblocking filter diffs — TC0_TABLE had transcription error at QP 26 (missing [1,1,1] entry)
  - [x] Implement B-frame decode — spatial direct prediction, bi-directional MC, DPB reordering
  - [x] Fix B-frame deblocking — L1 data in MbDeblockInfo, two-permutation BS check, canonical picture IDs
  - [x] Implement I_PCM macroblock decode — byte-align + read 384 raw bytes, QP tracking fix
  - [x] Implement pred_weight_table parsing — consume weight bits for weighted P/B slices
  - [x] Implement direct_8x8_inference_flag=0 — per-4x4 col_zero_flag in spatial direct
  - [x] Fix ref_pic_list_modification wrap — modular arithmetic mod MaxFrameNum
  - [x] Fix frame crop offset — apply SPS crop_left/crop_top in frame output
  - [x] Fix sliding window FrameNumWrap — evict smallest FrameNumWrap, not smallest raw frame_num
  - [ ] Implement weighted prediction application — weights parsed, need MC application
  - [ ] Debug MMCO complex sequences — MR4/MR5 diverge at frame 17 (Op 1+3+4+6 interactions)
  - [ ] Debug HCBP1/HCBP2 15-ref DPB — 17/250 match, sliding window not improving
  - [ ] Fix FMO (Flexible Macroblock Ordering) — FM1_FT_E has num_slice_groups>1
- [ ] **AV1 decoder via rav1d** — wrap Prossimo's rav1d (pure Rust AV1 decoder, port of dav1d) behind wedeo traits, similar to how symphonia is wrapped. Create `adapters/wedeo-rav1d/` with `Decoder` impl. Needs: new `CodecId::Av1` variant, AV1 OBU probe in demuxer, frame type mapping. rav1d handles all decode internally; wedeo just needs the trait bridge and pixel format conversion. Consider using a git worktree (`feat/rav1d-integration`) since it touches workspace Cargo.toml.
- [ ] **Video player with audio** — extend `bins/wedeo-play/` with audio playback (currently video-only via minifb). Needs: audio output backend (cpal or rodio crate), A/V sync (PTS-based with audio clock as master), demuxer that handles both audio+video streams (MP4/MKV via symphonia). Consider using a git worktree (`feat/video-player-audio`) since it may need new Frame fields in wedeo-core.
- [ ] **VP9 decoder** — second priority for WebM support. Reference: `FFmpeg/libavcodec/vp9*.c`.
- [ ] **HEVC decoder** — similar to H.264 but more complex (CTU/CTB structure).

### Video muxers
- [ ] **MP4/MOV muxer** — needed for any useful video output. No existing pure-Rust MP4 muxer crate.
- [ ] **MKV/WebM muxer** — `matroska` crate (0.30.0, 143K downloads, by tuffy) is a demuxer only. Muxer needs to be written.

## Infrastructure (ongoing)

- [ ] **Filter graph data flow** — the filter trait and graph skeleton exist but have no format negotiation or frame queues. Needed before real transcode pipelines work.
- [ ] **Interruptible I/O** — needed for network streams and cancellation.
- [ ] **Buffer pool** — for high-throughput decode. Currently every frame allocates a new buffer.
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
