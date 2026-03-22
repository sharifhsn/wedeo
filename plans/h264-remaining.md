# Plan: H.264 Remaining DIFF Files — COMPLETED

## Final state: 50/51 progressive CAVLC BITEXACT (98%)

Only FM1_FT_E.264 remains (FMO, out of scope).

## Fixes applied (47→50 BITEXACT)

### HCMP1 — already BITEXACT before this session (48/51 baseline)

### CVFC1_Sony_C.jsv — FIXED (19/50 → 50/50)

**Root cause:** Cross-slice C→D MV neighbor fallback missing slice boundary check.

When the C (top-right) MV neighbor was within the current MB but not yet decoded
(PART_NOT_AVAILABLE), the code fell back to D (top-left). The L0 path delegated
to `neighbor_c()` which performed this fallback without checking slice boundaries,
reading MV data from a cross-slice MB.

**Fix:** `mvpred.rs` — inline C→D fallback with slice_table checks (matching L1 path).

### CVWP3_TOSHIBA_E.264 — FIXED (89/90 → 90/90)

**Root cause:** Intra MBs in B-slices left MV context at PART_NOT_AVAILABLE (-2).

Intra MBs (I4x4, I16x16, I_PCM) didn't update the MV context after decoding.
The ref_idx stayed at the initialization value of -2 (PART_NOT_AVAILABLE) instead
of -1 (LIST_NOT_USED). When spatial direct prediction computed the unsigned minimum
of neighbor refs, -2 was treated as 254 (valid ref) instead of 255 (unavailable),
producing a wrong ref_idx.

**Fix:** `mb.rs` — set MV context to ref=-1, mv=[0,0] for all 16 blocks after
intra MB decode.

### Also fixed: spatial direct col_zero_flag L1 fallback

When colocated L0 ref was < 0 (not used), L1 ref wasn't checked as fallback.
Per FFmpeg h264_direct.c lines 443-447: `(l1ref0 == 0 || (l1ref0 < 0 && l1ref1 == 0))`.

**Fix:** `mb.rs` — check L1 colocated ref/MV when L0 ref < 0, for both
direct_8x8_inference paths.
