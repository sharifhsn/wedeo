# H.264 Debug Working Notes

**Purpose**: Persistent scratch pad for debug findings. Read this file at the
start of every context window to avoid re-discovering the same things.

**Current status**: 16/17 LUMA BITEXACT (no-deblock). Chroma V has ±1-3 diffs
on 4 multi-slice files. See H264.md for full conformance table.

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

### Chroma V (Cr) ±1 rounding issue

- Affects: SVA_BA2_D (1/17), SVA_NL2_E (3/17), SVA_Base_B (8/17),
  SVA_CL1_E (43/50), BA1_FT_C (29/299)
- Y and U planes are perfect. Only V (Cr) has ±1 to ±3 diffs.
- Pattern: uniform ±1 across 4x4 or 8x4 chroma sub-blocks
- Not slice-boundary related (SVA_BA2_D is single-slice)
- Both U and V use identical code paths (same mc_chroma, same IDCT)
- The `second_chroma_qp_index_offset` is same as first for Baseline, so QP is not the issue
- **Next step**: Extract actual chroma coefficients and MC output from both
  decoders for a simple case (SVA_BA2_D frame 16, MB(7,8)) and find WHERE
  the V values first diverge.

### Deblocking filter chroma issue

- BA1_FT_C: all frames differ with deblocking but 270/299 match without
- Other files may also have deblocking chroma issues
- Likely related to or caused by the chroma V rounding issue

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
