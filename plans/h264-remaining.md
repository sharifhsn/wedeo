# Plan: H.264 Remaining 3 DIFF Files (47→50 BITEXACT)

## Current state: 47/51 progressive CAVLC BITEXACT (92%)

## Remaining files

| File | Match | Category | Root Cause Hypothesis |
|------|-------|----------|----------------------|
| CVWP3 | 89/90 | 1 pixel diff frame 69 | Weighted bipred rounding or edge case |
| HCMP1 | 87/250 | Hierarchical B, diffs from frame 1 | Weighted bipred (HCBP1 passes, HCMP1 differs — Main vs Baseline) |
| CVFC1 | 19/50 | Multi-slice + crop, diffs from frame 17 | Cross-slice MV pred or deblock at non-aligned slice boundaries |
| FM1_FT_E | 119/305 | FMO | Out of scope (num_slice_groups > 1) |

## Diagnostic protocol (apply for each file)

1. `python3 scripts/framecrc_compare.py --no-deblock --pixel-detail <file>` — triage MC vs deblock
2. `python3 scripts/mb_compare.py <file> --start-frame N --max-frames 1` — find differing MBs
3. If ref list suspected: `python3 scripts/reflist_compare.py --ffmpeg --frame N <file>`
4. If MC suspected: extract MVs from both decoders via lldb
5. `python3 scripts/regression_check.py` — verify no regressions after each fix

## Priority order

### 1. CVWP3 (89/90 — very close)

Single pixel diff at frame 69. Stream uses weighted bipred (weighted_bipred_idc > 0).

**Approach:**
1. `python3 scripts/framecrc_compare.py --no-deblock --pixel-detail CVWP3` — check if deblock-only
2. `python3 scripts/mb_compare.py CVWP3 --start-frame 69 --max-frames 1` — find differing MB(s)
3. Check mb_type: if B-frame with weighted bipred, check the bipred formula
4. Compare with FFmpeg `h264dsp_template.c:63-92` (biweight function)
5. Look for ±1 rounding difference in weighted avg formula

### 2. HCMP1 (87/250 — hierarchical B with 15 refs)

Diffs from frame 1 with Y_max=110+. HCBP1 (Baseline, same structure) passes.
The difference is that HCMP1 uses Main profile — likely weighted_bipred_idc=2
(implicit weighted bipred).

**Approach:**
1. Check `weighted_bipred_idc` in PPS: `ffmpeg -bsf:v trace_headers | grep weighted_bipred`
2. If weighted_bipred_idc=2: implicit weighted bipred is used — check wedeo's implementation
3. Compare implicit weight calculation: `w0 = (64 * tb) / td`, `w1 = 64 - w0`
4. Read FFmpeg `h264_mb.c` implicit weight logic and compare with wedeo's `implicit_weight` table
5. `python3 scripts/mb_compare.py --start-frame 1 --max-frames 1` for first diff details

### 3. CVFC1 (19/50 — multi-slice + crop)

Baseline profile, 4 slices/frame at non-row-aligned boundaries (MB 0, 99, 198, 297).
Diffs start at frame 17 with Y_max=47, 26/209 MBs differ.

**Approach:**
1. Check if diffs correlate with slice boundaries (MB 99/22 ≈ row 4.5, etc.)
2. `python3 scripts/mb_compare.py --start-frame 17 --max-frames 1` — map differing MBs
3. Check cross-slice MV prediction: does wedeo correctly mark neighbors as unavailable
   across non-row-aligned slice boundaries?
4. Read FFmpeg `h264_mb.c` fill_decode_caches for non-row-aligned slice boundary handling
5. The crop itself is working (dimensions match); issue is likely in prediction, not crop

## Key lessons from this session

- **reorder_depth=1** for non-Baseline without VUI prevents premature IDR output
- **ref_pic_list_modification** must pre-size list and allow duplicates
- **DPB index** (not POC or ref_idx) is the correct picture identity for deblocking
- **--no-deblock triage** separates MC from deblock bugs
- **lldb ref list extraction** is the fastest way to verify ref list correctness
- **regression_check.py** catches regressions in ~4s
