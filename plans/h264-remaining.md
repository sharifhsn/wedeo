# Plan: H.264 Remaining 4 DIFF Files (47→51 BITEXACT)

## Current state: 47/51 progressive CAVLC BITEXACT (92%)

## Completed this session (2026-03-21)

| Fix | Files Fixed | Technique |
|-----|-------------|-----------|
| Reorder depth=1 for non-Baseline without VUI | CVWP1, CVWP2, CVBS3, CVSE3, CVSEFDFT3, MR4, MR5, cvmp | One-line change in apply_sps |
| Ref list modification: pre-size + allow duplicates | CVWP5 (MC part) | Rewrote reorder_list, pre-pad before modification |
| DPB-based deblock ref identity | CVWP5 (deblock part) | Use DPB index instead of ref_idx for P-slice BS |

## Remaining files

| File | Match | Category | Next Step |
|------|-------|----------|-----------|
| CVWP3 | 89/90 | 1 pixel diff at frame 69 | mb_compare at frame 69, check weighted bipred formula |
| HCMP1 | 87/250 | Hierarchical B cascade from frame 1 | mb_compare --no-deblock frame 1, check ref list and MC |
| CVFC1 | 19/50 | Multi-slice + crop | mb_compare frame 17, investigate slice boundary effects |
| FM1_FT_E | 119/305 | FMO | Out of scope (num_slice_groups > 1) |

## Priority order

### 1. CVWP3 (very close — 1 pixel diff)
89/90 match. Single pixel diff at frame 69 in a weighted bipred stream.
Likely a rounding error or edge case in the weighted prediction formula.

**Approach:**
1. `python3 scripts/mb_compare.py fate-suite/h264-conformance/CVWP3_TOSHIBA_E.264 --start-frame 69 --max-frames 1`
2. Find the differing MB(s) and check the mb_type
3. Trace the weighted pred weights and MC values for that specific block
4. Compare with FFmpeg via lldb

### 2. HCMP1 (hierarchical B, large diffs)
87/250, diffs start at frame 1 with Y_max=110+ (no-deblock). This is the
hierarchical B counterpart to HCBP1 (which passes). HCMP1 uses Main profile
with weighted bipred. The large diffs from frame 1 suggest a reference
list or weighted prediction issue specific to hierarchical B structure.

**Approach:**
1. Compare HCMP1 vs HCBP1 stream features (profile, weighted pred, etc.)
2. `python3 scripts/mb_compare.py --start-frame 1 --max-frames 1`
3. Check if weighted bipred is the differentiator

### 3. CVFC1 (multi-slice + crop)
19/50, diffs start at frame 17. Baseline profile, 4 slices/frame at
non-row-aligned boundaries (MB 0, 99, 198, 297). Y_max=47 with 26/209
MBs differing at frame 17. No B-frames (all P after IDR).

**Approach:**
1. Check if diffs correlate with slice boundaries
2. Investigate cross-slice MV prediction at slice boundaries
3. Check if frame crop interacts with slice boundary deblocking

## Anti-patterns
- Don't debug CVWP5 weights (FIXED — was ref list modification bug)
- Don't theorize about cache layout — extract actual values
- Run full conformance after every fix to check for regressions/bonuses
