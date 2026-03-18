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

## Issue Categories (for tracking)

1. **Intra4x4 neighbor bug** — f31 MB(5,5), affects 4 files
2. **Multi-slice I-frame corruption** — SVA_Base_B, SVA_FM1_E, SVA_CL1_E
3. **BASQP1 CAVLC desync** — QP=0 multi-slice
4. **Early P-frame diffs** — BAMQ2 f2, SVA_BA2_D f2, SVA_NL2_E f2, BA3_SVA_C f1
5. **BA1_FT_C 352x288** — frame 0 broken
