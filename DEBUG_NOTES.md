# H.264 Debug Working Notes

**Purpose**: Persistent scratch pad for debug findings. Read this file at the
start of every context window to avoid re-discovering the same things.

**Current status**: 15/17 BITEXACT. See H264.md for full conformance table.

## WEDEO_NO_DEBLOCK usage

`WEDEO_NO_DEBLOCK=1` disables wedeo's deblocking filter; `mb_compare.py`
also passes `-skip_loop_filter all` to FFmpeg when this env var is set.
This gives clean per-MB pixel isolation for debugging. **Verified**: files
that are BITEXACT without deblocking are also BITEXACT with deblocking
(the deblocking filter is correct). Keep using it for debugging, drop it
for final verification / regression tests.

## How to extract trace data (DON'T LOOP ON THIS)

```bash
# Step 1: Build binary (with or without tracing)
cargo build --bin wedeo-framecrc --features tracing,tracing-detail

# Step 2: Run binary directly (NOT via cargo run), redirect stderr to file
WEDEO_NO_DEBLOCK=1 RUST_LOG=wedeo_codec_h264=trace \
  ./target/debug/wedeo-framecrc fate-suite/h264-conformance/FILE.264 \
  1>/dev/null 2>/tmp/trace.log

# Step 3: Strip ANSI codes and grep
sed 's/\x1b\[[0-9;]*m//g' /tmp/trace.log | grep "pattern"
```

**IMPORTANT**: mb_compare.py uses the DEBUG binary (checks target/debug first).
After code changes, run `cargo build --bin wedeo-framecrc` (no --release) to
update the binary that mb_compare.py uses.

## Prefer lldb over ad-hoc C programs for FFmpeg extraction

See memory/feedback_empirical_debug_workflow.md for details. Use lldb for
one-off questions, not C programs.

## Remaining 2 failures

### BA1_FT_C (352x288, multi-slice) — 260/299 frames match

- First diff at frame 261 MB(19,9)
- 352x288 = 22x18 MBs, many slices per frame with non-uniform first_mb
- Slice boundaries: first_mb = 0,7,15,23,31,45,63,88,119,153,...
- MB(19,9) = mb_addr = 9*22+19 = 217. Need to check which slice this is in.
- Likely the same RBSP exhaustion margin issue or another slice boundary
  edge case. The 1-bit margin fix solved SVA files but may not cover all cases.
- **Next step**: Run mb_compare on just frames 260-265 to isolate the diff.
  Then use the MC ref check trace to see if the reference has valid data.

### BA3_SVA_C — B-frames (slice_type=6) not implemented

- 33 frames: 1 I + alternating P and B frames
- I-frame matches. P-frames wrong because B-frame decode produces garbage
  that goes into DPB and corrupts subsequent P-frame references.
- **Not a bug fix** — requires implementing B-frame decode:
  - Reference list 1 construction (future references)
  - Bidirectional MC (weighted average of L0 and L1 predictions)
  - B-frame specific mb_type parsing
  - Direct mode MV derivation

## Bugs fixed this session (for reference)

1. IDCT pass order (row-first then column) — all I-frames bitexact
2. Inter MB intra4x4 mode context (DC_PRED not -1) — 7 files
3. Ref list frame_num wrap-around (pic_num with MaxFrameNum)
4. MV neighbor C from decoded MBs only
5. Slice boundary tracking (slice_table + per-MB top_available)
6. Slice-aware MV prediction neighbors
7. RBSP exhaustion margin (8→1 bit) — SVA_Base_B/FM1_E/CL1_E

## Key technical notes for future sessions

- `mb_compare.py` checks target/debug FIRST, then target/release
- `cargo run --features tracing` builds a DIFFERENT binary than `cargo build`
  without features — they don't share the cache, so rebuilding one doesn't
  update the other
- FFmpeg `-debug mb_type -threads 1` shows MB types per frame (use -threads 1
  to prevent output interleaving)
- FFmpeg `trace_headers` BSF: `ffmpeg -i file -c copy -bsf:v trace_headers -f null /dev/null 2>&1`
