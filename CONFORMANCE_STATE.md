# H.264 Conformance State (2026-03-19)

**Status: 43/57 progressive CAVLC BITEXACT** (unchanged across 2 sessions)

## Commits this session
```
36e52c4 fix(h264): pre-mark current picture as ShortTerm before MMCO ops
939f15f docs: update conformance state with B_8x8 MV prediction investigation
```

## Dead ends explored (DO NOT REPEAT)

### MR3 frame_num gap fill (5 attempts, all failed)
- POC-only fix (local variable in compute_frame_num_offset): 284→242
- Gap fill eviction-only (no non-existing refs): no effect
- Gap fill with empty buffer non-existing refs: output truncated to 41 frames
- Gap fill with zeroed buffer refs + prev_fn update: 284→258
- Gap fill without prev_fn update: 284→278 (gap fires repeatedly)
- Gate on `gaps_in_frame_num_allowed` prevents sp1/sp2 regressions ✓
- **Root issue unclear**: gap fill evicts MMCO-managed refs the encoder expects to survive. Need FFmpeg DPB dump at frame 283 WITH gap fill to compare.

### CVBS3 B_8x8 block_width fix (1 attempt, caused regressions)
- Changed `sw` from 2 to 1 for 8x8 sub-partitions: 43→41 BITEXACT
- FFmpeg's `pred_motion(part_width)` and wedeo's `get_neighbors_slice(sw)` have different neighbor selection mechanics — NOT directly comparable

## What IS known about remaining files

### MR3 (284/300)
- POC type 2, gaps_in_frame_num_allowed=1
- 16 PIXEL diffs at end (not ordering), all P-frames
- FFmpeg DPB at frame 283: fn=[203,204,205,206] (recent, sequential)
- Wedeo DPB at frame 283: fn=[67,217,218,221,225] (old MMCO-preserved refs)
- The DPB divergence is the root cause but fixing it is non-trivial

### CVBS3/CVSE3/CVSEFDFT3 (245/224/163 match)
- Diffs are in **B_L0/B_L1/B_Bi sub-partitions** within B_8x8 MBs, NOT B_Direct
- B_Direct blocks produce correct pixels
- Diff is cascade from MV prediction: wrong neighbor context
- CVBS3 frame 7 MB(8,2): block[0]=B_Direct(correct), block[2]=B_L0(WRONG, max_diff=17)
- Sub-types at MB(8,2): [0, 2, 1, 11] = [B_Direct, B_L1, B_L0, B_Bi_4x4]

### Other files (unchanged from previous investigation)
- MR4 (135/300): POC type 0, DPB matches FFmpeg but mixed pixel+ordering diffs
- MR5 (52/300): Complex MMCO + POC type 1
- CVWP2/3 (29/90): Output ordering + weighted bipred
- CVWP5 (7/90): Multi-ref mixed weight flags
- HCMP1 (33/250): Hierarchical B-frames
- CVFC1 (19/50): Multi-slice, fails at frame 17
- cvmp_mot_frm0_full_B (27/30): 3 B-frames, B_8x8 sub-partitions
- FM1_BT_B, FM1_FT_E: FMO (out of scope)

## Priority for next session

1. **STOP investigating MR3** — 5 failed attempts, needs fundamentally different approach
2. **Extract empirical MV prediction values** from BOTH decoders for CVBS3 frame 7 MB(8,2) partition 2 (B_L0). Compare MVP neighbors (A, B, C), MVD, and final MV to find exact divergence point.
3. Consider simpler targets: cvmp_mot_frm0_full_B (27/30, only 3 wrong frames)

## Verify command
```bash
cargo clippy -p wedeo-codec-h264 && cargo test -p wedeo-codec-h264 && \
  cargo build --release -p wedeo-fate && \
  python3 scripts/conformance_report.py --cavlc-only --progressive-only --only-failing 2>/dev/null
```
