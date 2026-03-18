# H.264 Debug Working Notes

**Purpose**: Persistent scratch pad for debug findings. Read this file at the
start of every context window to avoid re-discovering the same things.

## Active Investigation: Frame 31 MB(5,5) intra4x4 bug

### What we know (verified empirically)

**Files affected**: BA_MW_D, BANM_MW_D, AUD_MW_E (first diff at f31 MB(5,5)),
BA2_Sony_F (first diff at f62 MB(5,5)). Same root cause.

**Wedeo trace for BANM_MW_D frame 31 MB(5,5)** (from /tmp/banm_trace.log):
```
blk(0,0): left_mode=-1, top_mode=-1, predicted=2(DC), prev_flag=0, mode=1(HORIZ), bit_pos=1483
blk(1,0): left_mode=1,  top_mode=-1, predicted=2(DC), prev_flag=0, mode=8(HU),    bit_pos=1487
blk(0,1): left_mode=-1, top_mode=1,  predicted=2(DC), prev_flag=1, mode=2(DC),    bit_pos=1491
blk(1,1): left_mode=2,  top_mode=8,  predicted=2(DC), prev_flag=1, mode=2(DC),    bit_pos=1492
blk(2,0): left_mode=8,  top_mode=-1, predicted=2(DC), prev_flag=0, mode=1(HORIZ), bit_pos=1493
blk(3,0): left_mode=1,  top_mode=-1, predicted=2(DC), prev_flag=0, mode=6(HD),    bit_pos=1497
blk(2,1): left_mode=2,  top_mode=1,  predicted=1(HORIZ), prev_flag=1, mode=1(HORIZ), bit_pos=1501
blk(3,1): left_mode=1,  top_mode=6,  predicted=1(HORIZ), prev_flag=1, mode=1(HORIZ), bit_pos=1502
blk(0,2): left_mode=-1, top_mode=2,  predicted=2(DC), prev_flag=0, mode=8(HU),    bit_pos=1503
blk(1,2): left_mode=8,  top_mode=2,  predicted=2(DC), prev_flag=1, mode=2(DC),    bit_pos=1507
blk(0,3): left_mode=-1, top_mode=8,  predicted=2(DC), prev_flag=1, mode=2(DC),    bit_pos=1508
blk(1,3): left_mode=2,  top_mode=2,  predicted=2(DC), prev_flag=1, mode=2(DC),    bit_pos=1509
blk(2,2): left_mode=2,  top_mode=1,  predicted=1(HORIZ), prev_flag=1, mode=1(HORIZ), bit_pos=1510
blk(3,2): left_mode=1,  top_mode=1,  predicted=1(HORIZ), prev_flag=1, mode=1(HORIZ), bit_pos=1511
blk(2,3): left_mode=2,  top_mode=1,  predicted=1(HORIZ), prev_flag=0, mode=8(HU),    bit_pos=1512
blk(3,3): left_mode=8,  top_mode=1,  predicted=1(HORIZ), prev_flag=1, mode=1(HORIZ), bit_pos=1516
```

**Key observation**: left_mode=-1 for all blk_x=0 blocks, top_mode=-1 for all
blk_y=0 blocks. This means `neighbor.left_available` or `neighbor.top_available`
is false, OR the neighbor modes are actually -1 in the cache.

**BUT**: predicted mode is ALWAYS DC_PRED=2 in these cases, because `if left_mode < 0 || top_mode < 0 { DC_PRED }`. So the MODE DECODE should be correct
even with -1 neighbors — the predicted mode is the same either way.

### ROOT CAUSE FOUND (2026-03-18)

Inter MBs store intra4x4 modes as -1 in `left_intra4x4_mode`/`top_intra4x4_mode`.
The H.264 spec says inter neighbors should appear as DC_PRED=2 for mode
prediction context. The predicted mode computation differs:

- **wedeo (WRONG)**: left_mode=-1 → predicted = DC_PRED = 2 (unconditionally)
- **FFmpeg (CORRECT)**: left_mode=2 → predicted = min(2, top_mode) = min(2, 1) = 1

For blk(0,1) of MB(5,5) with prev_flag=1:
- wedeo: mode = predicted = 2 (DC) → flat 82 prediction
- FFmpeg: mode = predicted = 1 (HORIZONTAL) → [74, 53, 77, 97] repeated

The fix: store DC_PRED=2 instead of -1 for inter MB neighbor modes in
`update_after_mb()` at mb.rs:518.

**TODO**:
- [x] Verified: MB(4,5) and MB(5,4) pixel output is correct in both decoders
- [x] Verified: modes DO differ (wedeo DC vs FFmpeg HORIZ for blk(0,1))
- [x] Root cause: -1 vs 2 for inter MB intra4x4 mode context
- [x] Apply fix: change `-1` to `2` for inter MB modes in mb.rs (decode_macroblock + decode_skip_mb)
- [x] Verify fix: BANM_MW_D, BA_MW_D, AUD_MW_E, BA2_Sony_F ALL BITEXACT (100/300 frames)
- [x] Bonus: BAMQ2_JVC_C, SVA_BA2_D, SVA_NL2_E also became BITEXACT (same root cause)

## Findings Log

### 2026-03-18 Finding 1: ANSI codes in trace output
Always strip ANSI codes when grepping trace files:
`sed 's/\x1b\[[0-9;]*m//g' /tmp/trace.log | grep "pattern"`

### 2026-03-18 Finding 2: mb_compare.py shows same first-diff pattern
- BA_MW_D: f31 MB(5,5) max_diff=34
- BANM_MW_D: f31 MB(5,5) max_diff=34
- AUD_MW_E: f31 MB(5,5) max_diff=45
- BA2_Sony_F: f62 MB(5,5) max_diff=40

## Active Investigation: Multi-slice I-frame corruption

### What we know (verified empirically)
- SVA_Base_B: 3 slices per frame: first_mb=0,33,66 (11 MBs per row, 3 rows per slice)
- Frame 0: MBs 0-32 (rows 0-2) are correct, MB 33+ (row 3+) are wrong
- First diff is at MB(0,3) = mb_addr=33 = start of second slice
- max_diff=224, mean_diff=173.8 → completely wrong pixels, not small diffs

### ROOT CAUSE FOUND (2026-03-18)
Cross-slice neighbor availability not tracked. The decoder sets
`top_available = mb_y > 0` unconditionally, but H.264 spec requires
neighbors from different slices to be treated as unavailable.

FFmpeg uses `h->slice_table[top_xy] != sl->slice_num` to check this.
Wedeo has no slice_table.

**FIXED (2026-03-18)**: Added `slice_table: Vec<u16>` and `current_slice: u16`
to FrameDecodeContext. Check slice_table when computing has_top/has_left.
Made BASQP1 BITEXACT and fixed SVA_Base_B/FM1_E/CL1_E frame 0.

## Current status: 12/17 BITEXACT

**FIXED issues:**
1. Intra4x4 neighbor bug — store DC_PRED(2) for inter MBs (7 files)
2. Multi-slice I-frame corruption — slice_table for neighbor availability (BASQP1 + I-frames)
3. Per-MB top_available — check slice_table per-MB not per-row (BA1_FT_C I-frame)

**Remaining 5 files** (all have correct I-frames, P-frame diffs only):
1. SVA_Base_B — I+P, multi-slice (3 slices/frame), f1 first diff MB(2,3)
2. SVA_FM1_E — I+P, multi-slice, similar
3. SVA_CL1_E — I+P, multi-slice, similar
4. BA3_SVA_C — I+P+B, 90/99 MBs wrong in f1 → **has B-frames** (not implemented!)
5. BA1_FT_C — I+P, multi-slice, f2 first diff MB(11,15)

**Likely root causes:**
- BA3_SVA_C: B-frames (slice_type=6). Not yet supported. Need B-frame decode.
- SVA_Base_B/FM1_E/CL1_E: P-frame MV prediction doesn't respect slice boundaries
  for neighbor availability. The `MvContext` doesn't check slice_table.
- BA1_FT_C: Same as SVA files — P-frame MV neighbors from different slices
