# Plan: H.264 Remaining 7 DIFF Files (50→57 BITEXACT)

## Current state: 50/57 progressive CAVLC BITEXACT

## Remaining files

| File | Match | Category | Next Step |
|------|-------|----------|-----------|
| CVWP5 | 7/90 | P-frame MC | Extract MVs from both decoders for MB(9,0) frame 4 |
| CVWP2 | 87/90 | Reorder (3 frames) | Run poc_output_compare.py, check VUI num_reorder_frames |
| CVWP3 | 86/90 | 3 reorder + 1 pixel | Same reorder as CVWP2 + one pixel diff at frame 69 |
| HCMP1 | 87/250 | Hierarchical B cascade | mb_compare on first differing frame, check MC source |
| CVFC1 | 19/50 | Frame crop + multi-slice | Read FFmpeg h264_slice.c crop handling, compare |
| FM1_FT_E | 119/305 | FMO | Out of scope (num_slice_groups > 1) |
| FM1_BT_B | 0/0 | FMO | Out of scope |

## Priority order

### 1. CVWP2/CVWP3 reorder (likely quick win)
3 frames in wrong output order. This is NOT a pixel bug — the frames are correct
but emitted in the wrong sequence. Likely a VUI/num_reorder_frames issue or an
edge case in the delayed output buffer.

**Approach:**
1. `python3 scripts/poc_output_compare.py fate-suite/h264-conformance/CVWP2_TOSHIBA_E.264`
2. Compare POC output order between wedeo and FFmpeg
3. Check if weighted_pred PPS changes affect output timing
4. Read FFmpeg's `h264_field_start` and output logic in `h264dec.c`

### 2. CVWP5 (empirical MV extraction needed)
7/90, huge diffs starting frame 4. Stream has ref_pic_list_modification commands.
Weight table parsing is correct (verified). All MBs use ref_idx=0 with identity
weight. The bug is in MC — either wrong MV or wrong reference picture.

**Approach:**
1. Add trace to predict_mv for P_L0_L0_8x16 (mb_type=2) showing neighbor values
2. Extract FFmpeg's MVs via lldb for MB(9,0) in frame 4 (POC=8)
3. Compare: if MVs match, the ref picture content differs (ref list bug)
4. If MVs differ, find which neighbor is wrong

**Key observation:** Frame has ref_pic_list_modification (bits 27-48 in slice header).
Verify `apply_ref_pic_list_modification()` produces same ref order as FFmpeg.

### 3. HCMP1 (characterize first)
87/250 hierarchical B with 15 refs. Run mb_compare on first few differing frames
to see if they share a pattern (same MB types? same sub-partition types?).

### 4. CVFC1 (code review)
19/50 with frame crop. The frame crop offsets in SPS affect output dimensions.
Read FFmpeg's crop handling and compare with wedeo's implementation.

## Anti-patterns
- Don't debug CVWP5 weights (they're correct — all identity for differing frames)
- Don't theorize about cache layout — extract actual values
- Run `test_file.py --diff` after every fix to check for regressions/bonuses
