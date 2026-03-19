# H.264 Conformance State (2026-03-19)

**Status: 43/57 progressive CAVLC BITEXACT**

## This session's investigation (no new BITEXACT passes)

### Fix applied
- **MMCO pre-mark** (`refs.rs`): Pre-mark current picture as ShortTerm before applying MMCO ops. FFmpeg does this in h264_refs.c. Needed for MMCO op 3 (ShortTermToLongTerm) to find the current pic by PicNum when it's still RefStatus::Unused. No current test exercises this path, but it matches FFmpeg's behavior.

### Root causes identified

1. **MR3 (284/300) — frame_num gap handling**: POC type 2 with gaps_in_frame_num_allowed. The stream has frame_num gaps (69→1 and 242→0). FFmpeg fills these gaps by advancing `prev_frame_num` through each gap (wrapping at max_frame_num=256), then checks for offset wraps. Without gap fill, wedeo detects TWO wraps (offset=512) instead of FFmpeg's one (offset=256). However, the POC error doesn't directly cause pixel diffs for this P-only stream. The 16 failing frames at the end are likely caused by wrong DPB FrameNumWrap values due to the same gap-handling issue. **Full fix requires implementing frame_num gap handling with DPB operations** (creating non-existing reference pictures), not just POC adjustments.

2. **CVBS3/CVSE3/CVSEFDFT3 remaining B-frame diffs**: All diffs are in **B_8x8 MBs with B_Direct_8x8 sub-partitions** (confirmed via mb_types.py). B_Direct_16x16 and B_Skip work correctly. The issue is specifically in temporal direct prediction at the 8x8/4x4 sub-partition level within B_8x8 macroblocks. Non-B_8x8 diffs (max 2-4) are cascade effects from MC using wrong reference pixels.

3. **MR4 (135/300) — DPB state matches but output differs**: POC type 0. DPB state at the first failing frame (output 12) MATCHES FFmpeg. The issue involves mixed pixel and ordering diffs (90 unique CRCs per decoder). Complex MMCO with both ST and LT management. May involve POC computation differences or ref_pic_list_modification.

4. **CVWP5 (7/90) — multi-ref weighted prediction**: Multi-PPS stream where PPS 0 changes weighted_pred_flag between frames. The weight parsing and application code appears correct (frames matching = single-ref weighted pred works). Diffs start at frame 2 which has 4 active refs with mixed weight flags (some refs luma-only, some chroma-only, some none). No CAVLC desync errors. Further investigation needed.

## Remaining 14 DIFF files

### Near-pass
- **MR3** (284/300): frame_num gap handling (POC type 2), 16 diffs at end
- **cvmp_mot_frm0_full_B** (27/30): 3 B-frames, B_8x8 with B_Direct sub-partitions

### B-frame temporal direct (B_8x8 sub-partitions)
- **CVBS3** (245/300), **CVSE3** (224/278), **CVSEFDFT3** (163/200): All diffs in B_8x8 MBs

### Weighted prediction
- **CVWP2** (29/90), **CVWP3** (29/90): Output ordering + weighted bipred
- **CVWP5** (7/90): Multi-ref mixed weight flags, POC type 2

### Multi-slice / cascading
- **HCMP1** (33/250): Hierarchical B-frames
- **CVFC1** (19/50): Multi-slice, fails starting frame 17

### Complex MMCO
- **MR4** (135/300): POC type 0, DPB matches but output differs
- **MR5** (52/300): Complex MMCO + POC type 1

### Out of scope
- **FM1_BT_B**, **FM1_FT_E**: FMO (num_slice_groups > 1)

## Priority next steps

1. **Frame_num gap handling** (MR3, potentially MR4/MR5): Implement FFmpeg's gap fill loop (h264_slice.c:1506-1522) including non-existing reference picture creation and DPB sliding window operations during gaps.
2. **B_Direct_8x8 temporal direct** (CVBS3/CVSE3/CVSEFDFT3/cvmp_mot): Use lldb on FFmpeg to extract colocated MV and ref_idx at a specific differing MB, then compare with wedeo's values to find the exact divergence.
3. **CVWP2/CVWP3 output ordering** — negative POC before IDR.

## Verify command
```bash
cargo clippy -p wedeo-codec-h264 && cargo test -p wedeo-codec-h264 && \
  cargo build --release -p wedeo-fate && \
  python3 scripts/conformance_report.py --cavlc-only --progressive-only --only-failing 2>/dev/null
```
