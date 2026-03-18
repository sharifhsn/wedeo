# H.264 Debug Working Notes

**Purpose**: Persistent scratch pad for debug findings. Read this file at the
start of every context window to avoid re-discovering the same things.

**Current status**: 16/17 BITEXACT (with deblocking). Only BA3_SVA_C (B-frames)
remains. See H264.md for full conformance table.

## WEDEO_NO_DEBLOCK usage

`WEDEO_NO_DEBLOCK=1` disables wedeo's deblocking filter; `mb_compare.py`
also passes `-skip_loop_filter all` to FFmpeg when this env var is set.
This gives clean per-MB pixel isolation for debugging.

**Note**: the deblocking filter has a chroma issue on BA1_FT_C (all frames
differ with deblocking). Files that are BITEXACT without deblocking may NOT
be bitexact with deblocking if they have a chroma V rounding issue.

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

## Remaining issues

### BA3_SVA_C — B-frames (slice_type=6) not implemented

- 33 frames: 1 I + alternating P and B frames
- I-frame matches. P-frames wrong because B-frame decode produces garbage
  that goes into DPB and corrupts subsequent P-frame references.
- **Not a bug fix** — requires implementing B-frame decode:
  - Reference list 1 construction (future references)
  - Bidirectional MC (weighted average of L0 and L1 predictions)
  - B-frame specific mb_type parsing
  - Direct mode MV derivation

### Chroma V (Cr) ±1 rounding — FIXED (2026-03-18)

Root cause: `CHROMA_QP_TABLE[36]` was 33 instead of 34 (transcription error).
Shifted all chroma QPs for luma QP >= 36 by -1, causing wrong dequant scale.
Fixed → 16/17 BITEXACT (no-deblock), up from 11/17.

### Deblocking filter diffs — FIXED (2026-03-18)

Root cause: `TC0_TABLE` had only 3 entries of `[1,1,1]` at QP 23-25
instead of 4 at QP 23-26, shifting all tc0 values from QP 26 onwards.
This caused wrong clipping thresholds in the deblocking filter.
Fixed → 16/17 BITEXACT with full deblocking (up from 10/17).

## Bugs fixed (for reference)

1. IDCT pass order (row-first then column) — all I-frames bitexact
2. Inter MB intra4x4 mode context (DC_PRED not -1) — 7 files
3. Ref list frame_num wrap-around (pic_num with MaxFrameNum)
4. MV neighbor C from decoded MBs only
5. Slice boundary tracking (slice_table + per-MB top_available)
6. Slice-aware MV prediction neighbors
7. RBSP exhaustion margin (8→1 bit) — SVA_Base_B/FM1_E/CL1_E
8. RBSP exhaustion graceful error (mb_skip_run parse at end-of-slice)
9. MV neighbor C/D slice check for blk_y > 0 — BA1_FT_C luma bitexact
10. CHROMA_QP_TABLE transcription error at index 36 — 16/17 BITEXACT (no-deblock)
11. TC0_TABLE transcription error at QP 26 — 16/17 BITEXACT (with deblocking)

## Key technical notes for future sessions

- `mb_compare.py` checks target/debug FIRST, then target/release
- `mb_compare.py --start-frame N` skips to frame N for faster debugging
- **mb_compare.py only checks LUMA** — chroma diffs won't be caught
- `cargo run --features tracing` builds a DIFFERENT binary than `cargo build`
  without features — they don't share the cache, so rebuilding one doesn't
  update the other
- FFmpeg `-debug mb_type -threads 1` shows MB types per frame (use -threads 1
  to prevent output interleaving)
- FFmpeg `trace_headers` BSF: `ffmpeg -i file -c copy -bsf:v trace_headers -f null /dev/null 2>&1`
