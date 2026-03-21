# Plan: Video Player with Audio

## Goal
Extend `bins/wedeo-play/` to play audio alongside video from container files
(MP4, MKV). Currently it only displays video frames via minifb.

## Deliverables
1. Audio playback via `cpal` crate (cross-platform audio output)
2. A/V sync using PTS-based timing with audio clock as master
3. Demux both audio + video streams from a single container
4. Zero clippy warnings, cargo fmt clean

## Architecture

The player will have 3 threads:
```
Main thread:   minifb window event loop + video frame display
Audio thread:  cpal callback pulls decoded audio samples from ring buffer
Decode thread: demux → decode audio+video → push to respective queues
```

### A/V sync strategy (audio-master)
FFmpeg and most players use audio as the timing master because:
- Audio glitches (underruns) are immediately audible
- Video can drop/repeat frames without being as noticeable

Implementation:
1. Audio callback reports its current PTS (based on samples consumed)
2. Video display checks audio PTS, shows the frame with nearest PTS ≤ audio_pts
3. If video is behind, drop frames. If ahead, wait.

### Key dependencies to add
- `cpal` (cross-platform audio): stable, widely used
- `ringbuf` or `crossbeam-channel`: for audio sample queue between decode and playback

## Reference files to read FIRST
1. `bins/wedeo-play/src/main.rs` — current video-only player
2. `crates/wedeo-resample/src/lib.rs` — if audio sample rate doesn't match output device
3. `adapters/wedeo-symphonia/src/demuxer.rs` — how demuxing works for containers

## Design notes

### Container demuxing for A+V
Currently wedeo-play uses the format registry to open a file and reads packets.
For A+V, the demuxer returns packets tagged with `stream_index`. Route audio
packets to audio decoder, video packets to video decoder.

### Audio format conversion
The audio decoder produces `Frame` with `SampleFormat` (typically F32 or S16).
The cpal output device expects a specific format. Use wedeo-resample if sample
rates differ. For format conversion (S16→F32), do it inline.

### Frame timing
Each decoded video Frame has `pts` (presentation timestamp) and `time_base`.
Convert to seconds: `pts_sec = pts * time_base.num / time_base.den`.
Compare against audio clock to decide when to display.

## Conflict zones
- **bins/wedeo-play/Cargo.toml**: Adding new dependencies (cpal, ringbuf)
- **bins/wedeo-play/src/main.rs**: Major rewrite of the main loop
- No changes to framework crates needed

## Estimated size
~500-800 lines replacing the current ~200 lines.

## Verification
```bash
cargo check --workspace
cargo clippy --workspace
# Manual test:
cargo run --release --bin wedeo-play -- fate-suite/some-video-with-audio.mp4
# Should display video in window AND play audio in sync
```
