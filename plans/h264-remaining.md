# Plan: H.264 Remaining 3 DIFF Files (47→50 BITEXACT)

## Current state: 47/51 progressive CAVLC BITEXACT (92%)

## Triage results (no-deblock)

| File | With deblock | No deblock | First diff | Max Y diff | Conclusion |
|------|-------------|------------|------------|-----------|------------|
| CVWP3 | 89/90 | 89/90 | frame 69 | 68 | MC bug, not deblock |
| HCMP1 | 87/250 | 87/250 | frame 1 | 110 | MC bug, not deblock |
| CVFC1 | 19/50 | 19/50 | frame 17 | 47 | MC bug, not deblock |

All three are MC issues (same match count with/without deblock).

## Priority 1: CVWP3 (89/90 — single frame diff)

**Profile:** Main, weighted bipred stream, B-frames.
**Diff:** 10/396 MBs differ at frame 69, first at MB(12,14), Y_max=29, chroma V_max=10.

**Approach:**
1. Trace MB(12,14) at frame 69 — check mb_type, ref_idx, MV, weighted pred params
2. If B-frame: check if it's B_Direct, B_L0, B_L1, or B_Bi
3. For B_Bi: check weighted bipred formula (explicit or implicit)
4. Extract same MB from FFmpeg via lldb, compare intermediates
5. The diff is small (max 29) — likely a rounding or weight calculation issue

## Priority 2: HCMP1 (87/250 — hierarchical B with 15 refs)

**Profile:** Main, NO weighted pred (weighted_pred=0, weighted_bipred=0).
**Slice types:** 1 I-slice + 1 P-slice + 8 B-slices per GOP.
**HCBP1 (Baseline, same structure but I+P only) passes BITEXACT.**
**Diff:** 163 frames differ starting at frame 1, Y_max=110+.

The key difference: HCMP1 has B-frames while HCBP1 doesn't. With no weighted pred,
the B-frame MC should be simple (unweighted average). Large diffs from frame 1
suggest a fundamental issue: wrong reference picture, wrong MV, or wrong bi-pred formula.

**Approach:**
1. `mb_compare --start-frame 1 --max-frames 1` — find first differing MB
2. Trace that MB: check B-frame type (B_Direct vs B_L0/L1 vs B_Bi)
3. If B_Bi with unweighted: verify `(L0 + L1 + 1) >> 1` formula
4. Check ref lists for frame 1 via `reflist_compare.py --ffmpeg --frame 1`
5. If ref lists match: extract MVs from both decoders for the first differing MB

## Priority 3: CVFC1 (19/50 — multi-slice + crop)

**Profile:** Baseline (no B-frames), 4 slices/frame, frame crop.
**Diff:** 31 frames differ starting at frame 17, Y_max=47, 26/209 MBs at frame 17.
**Slice boundaries:** first_mb = 0, 99, 198, 297 (non-row-aligned).

All P-frames, no B-frames or weighted pred. The issue is likely cross-slice
MV prediction at non-row-aligned slice boundaries. FFmpeg's fill_decode_caches
handles slice boundaries carefully; wedeo may treat MBs at mid-row slice
boundaries differently.

**Approach:**
1. `mb_compare --start-frame 17 --max-frames 1` — map which MBs differ
2. Check if differing MBs cluster at slice boundary positions (MB 99, 198, 297)
3. If yes: read FFmpeg's fill_decode_caches for slice boundary handling
4. If no: check if it correlates with DPB eviction (frame 17 = frame_num 17,
   max_num_ref_frames=5, so sliding window starts evicting at frame 5)

## Diagnostic protocol (for each file)

1. `framecrc_compare.py --no-deblock --pixel-detail` — done (above)
2. `mb_compare.py --start-frame N --max-frames 1` — find differing MBs
3. `reflist_compare.py --ffmpeg --frame N` — if ref list suspected
4. Trace MB with `--features tracing` — check mb_type, ref_idx, MV, weights
5. lldb on FFmpeg — extract ground truth for the same MB
6. `regression_check.py` — after every fix
