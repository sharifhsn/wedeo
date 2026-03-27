# Wedeo Progress

## Current Status

See `CLAUDE.md` and `H264.md` for detailed status.

- H.264 CAVLC: **52/52** progressive conformance files BITEXACT (100%) — added CI_MW_D, LS_SVA_D
- H.264 CABAC: 27/27 progressive conformance files BITEXACT (100%)
- H.264 FRext CAVLC: **5/7 BITEXACT** (2 out-of-scope: PAFF)
- H.264 FRext: **23/55 BITEXACT** — all progressive High Profile 4:2:0 8-bit files pass. Remaining are interlaced (MBAFF/PAFF), 10-bit, or 4:2:2.
- H.264 MBAFF: deblocking infrastructure (Phases 1-7), field/frame pixel addressing, CABAC left_block_options remapping, per-MB top-mode storage, field-mode top-right availability. CABAC engine 100% correct. **CAMA1_Sony_C (I-slice) BITEXACT for reconstruction** (no-deblock). Other CAMA files need inter prediction fixes.
- Precommit total: **52 passing, 0 regressed** (snapshot needs update to include new FRext files)
- WAV/PCM pipeline: byte-identical to FFmpeg 8.0.1 across all FATE suite samples
- Audio via symphonia: 28 decoders, 10 demuxers, SNR-verified lossy codecs

## Frame-Level Threading — Infrastructure Complete (2026-03-27)

Steps 1-8 of frame-level threading committed:
- `SharedPicture` with `AtomicI32` row progress + `Condvar` wait
- `PicHandle` Deref wrapper, DPB `Arc<SharedPicture>` migration
- `publish_row()` / `wait_for_row()` for row-level MC dependencies
- `InFlightDecode` decouples frame completion from decode loop
- Deblock offloaded to rayon thread pool (`rayon::spawn` + `mpsc::sync_channel`)
- Non-ref B-frames defer join → deblock overlaps next frame's decode

**Benchmark (BBB 1080p 10s):** 16.28s sync → 15.43s rayon (~5% faster).

**Deblock row wavefront** stays on `std::thread::scope` — rayon deadlocks
with spin-wait dependencies (yield_now creates recursive nested chains).

**New script:** `bench_ab.py` — A/B benchmark current vs previous commit via hyperfine.

**Next:** Pipelined frame decode (multiple frames decoding simultaneously using
the row-level progress infrastructure already in place).

## FRext Conformance — All Progressive Files BITEXACT (2026-03-26)

**Root cause (Group A+B, 17 files):** `CabacNeighborCtx.intra4x4_modes` stored -1
(unavailable) for inter/skip MBs instead of 2 (DC_PRED). This caused wrong CABAC
predicted mode when `min(left_intra_mode, top_mode)` had one inter neighbor and one
intra neighbor with mode < 2. FFmpeg's `fill_decode_caches` stores 2 for non-I4x4
neighbors without `constrained_intra_pred`. Fix: `decoder.rs:mb_intra4x4_modes_i8()`
now returns `[2; 16]` for inter MBs, `[-1; 16]` only with constrained_intra_pred.

**Also fixed:**
- Median overflow in `mvpred.rs:18` (i16→i32 cast). HPCAMAPALQ no longer panics.
- Intra 8x8 top-right extension in `intra_pred.rs:393` — reverted to raw `top[7]` per spec.

**New scripts:** `frame_type_map.py`, `coeff_compare_8x8.py`, `pixel_compare_mb.py`,
`cabac_state_compare.py` (CABAC engine state comparison at MB boundaries).

## MBAFF Deblock Root Cause Found — Vertical Edge at Pair Boundary (2026-03-26)

Traced the CAMA1_Sony_C deblock diffs (Y≤8) to root cause via pixel watchpoint.

**Finding:** MB(17,6) frame-mode (pair 3 top), above pair (17,4-5) field-mode.
The MBAFF horizontal edge 0 handler processes the pair 2/3 boundary with doubled
stride. The q-side pixels (pair 3 row 96) differ because FFmpeg's vertical edge 1
at MB(17,6) modifies reconstruction pixel (276,96) from 64→65, while wedeo's doesn't.
This cascades through the MBAFF horiz edge 0 handler to modify p1 at row 92.

**Next step:** Compare vertical edge 0/1 inputs for frame-mode MB(17,6) to find why
the normal filter produces different output. The MB boundary vertical edge (edge 0)
may modify col 2-3 differently, changing edge 1's threshold result.

New scripts: `deblock_edge_trace.py` (lldb extraction), `deblock_pixel_watch.py`
(finds modifying MB), `deblock_mb_dump.py` (side-by-side MB comparison).

## MBAFF Deblocking Iteration Order + bS=1 Mixed Inter (2026-03-26)

**Fix 1: Iteration order.** FFmpeg's `loop_filter()` (h264_slice.c:2451-2452)
iterates MBAFF deblocking column-first within pair rows: for each mb_x, deblock
both MBs of the pair before moving to mb_x+1. Wedeo used raster order. Fixed
`deblock_frame()` to accept `is_mbaff` and use the correct order. Measurably
better: Y≤8 vs Y≤10 for CAMA1_Sony_C.

**Fix 2: bS=1 for mixed horizontal inter edges.** FFmpeg h264_loopfilter.c:557-559
sets bS=1 and skips MV check for horizontal MB boundary edges between MBs with
different interlace modes. Added in `compute_luma_bs()`. Not triggered by
CAMA1_Sony_C (I-only) but needed for CAMA1_TOSHIBA_B etc.

**Result:**
- CAMA1_Sony_C: Y≤8, U≤8, V≤6 (was Y≤10, U≤10, V≤9 with raster order)
- 52/52 precommit BITEXACT, 0 regressions
- Remaining small diffs (±1 at horiz_e3, cascading to ±8) need lldb investigation

New script: `deblock_ab_compare.py` — pixel-level A/B compare with edge classification.

## MBAFF Intra Prediction Fixes: Per-MB Top Modes + Field Top-Right (2026-03-26)

Two bugs in MBAFF intra 4x4/8x8 prediction, both causing wrong prediction
modes/inputs for field-mode MBs:

**Bug 1: Top neighbor mode overwrite.** The single-row `top_intra4x4_mode`
buffer in `NeighborContext` was overwritten by the top MB of a pair before the
bottom MB could read the previous pair's modes. For field-mode top MBs, the
spatial above neighbor is 2 MB-rows back (previous pair's top), but the buffer
only had the most recent (previous pair's bottom). Fixed by adding per-MB
bottom-row mode storage (`intra4x4_modes_top` in `CabacNeighborCtx`) and
changing the CABAC top-mode lookup to use `nb_top` (MBAFF-adjusted index).

**Bug 2: Field-mode top-right availability.** The top-right availability check
for block column 3 (rightmost) used `(mb_y-1)*mb_width + mb_x+1`, pointing to
the current pair row (not yet decoded). For field-mode MBs, the "above" row is
from the previous pair row (`mb_y-2`), which is always fully decoded. Fixed in
both `decode_intra4x4` and `decode_intra8x8`.

**Result:**
- CAMA1_Sony_C.jsv: **BITEXACT** for reconstruction (0/1350 MB diffs, no-deblock, all 5 frames)
- 102 precommit tests: 0 regressions
- Deblocking still shows diffs (separate known issue)

New scripts:
- `mbaff_pair_diff.py` — field-aware per-pair pixel comparison
- `ffmpeg_recon_extract.py` — lldb-based pixel extraction from FFmpeg at specific MB
- `mbaff_recon_compare.py` — side-by-side wedeo vs FFmpeg reconstruction comparison

## MBAFF left_block_options CABAC Context Remapping (2026-03-25)

Implemented FFmpeg's `left_block_options[4][32]` remapping for CABAC context
derivation in MBAFF frames with field/frame mode mismatch.

**Root cause:** When a field-mode MB has a frame-mode left neighbor (or vice versa),
the CABAC context for NNZ, CBP, and intra4x4 pred modes must use remapped sub-block
indices from the left neighbor. Wedeo always used the default option (0), producing
wrong CABAC context at field/frame boundaries. This didn't cause a CABAC desync
(pos/low/range still matched) but did produce wrong decoded coefficients/modes.

**Fix:**
- Added `left_block_option: u8` (0-3) to `MbaffNeighbors`
- NNZ: remapped left row per `LEFT_LUMA_NZ_ROW` / `LEFT_CHROMA_NZ_ROW` tables
- CBP: apply bit-shift formula from FFmpeg h264_mvpred.h:728-730
- Intra4x4 modes: per-MB storage in `CabacNeighborCtx`, remapped lookups
- LTOP/LBOT selection: top half uses LTOP, bottom half uses LBOT

**Result:**
- CABAC engine verified 100% correct at every MB (pos/low/range/field match FFmpeg)
- CAMA1 frame 0: differing MBs 1136→837 (26% reduction), first diff MB(17,4)→MB(19,4)
- All 102 progressive tests remain BITEXACT
- Remaining MBAFF diffs are pixel reconstruction (apply_macroblock), not CABAC

New scripts:
- `cabac_state_at_mb.py` — side-by-side CABAC state comparison at any MB via lldb
- `verify_left_block_tables.py` — verify wedeo's remapping tables against FFmpeg source

## MBAFF Conformance Plan Triage + Top-Right Fix (2026-03-25)

**Plan triage:** All 5 WP-1 "progressive FRext" files (FREXT01, FREXT02, FRExt2,
FRExt4, Freh7) are MBAFF/PAFF (frame_mbs_only=false), not progressive. WP-1 as
scoped is not actionable without MBAFF/PAFF decode. Only 1 of 8 CAMA* files
(CAMA1_Sony_C) is I-only; rest use I+P+B. All 6 FRext DIFF files are PAFF.

**Fix: Intra 4x4/8x8 top-right availability (mb.rs)**
The rightmost block column's top-right availability only checked bounds
(`mb_x + 1 < mb_width`). In MBAFF, the above-right MB may not be decoded yet.
Now checks `slice_table[tr_idx] == current_slice`. Fixes pre-deblock pixel diffs
for frame-mode pair rows 0-1 of CAMA1. Both 4x4 and 8x8 paths fixed.

**Open:** CABAC state[70] probability diverges at pair (17, 4-5) in CAMA1 pair
row 2, causing wrong field flag decode and cascading errors to all subsequent rows.

New scripts:
- `sps_flags.py` — dump SPS flags (frame_mbs_only, mb_aff) for conformance files
- `cabac_pair_compare.py` — compare CABAC byte positions at pair-row boundaries
- `pair_row_diff.py` — per-pair-row pixel diff summary for MBAFF files

## MBAFF CABAC Field-Mode Context Fix (2026-03-25)

**Root cause found and fixed:** MBAFF CABAC desync was caused by three issues:
1. Field-coded MBs (mb_field_decoding_flag=1) need separate CABAC context offset tables
   for sig/last_coeff flags. FFmpeg uses `significant_coeff_flag_offset[MB_FIELD(sl)][cat]`
   (2D table: frame=105+, field=277+). Wedeo always used frame-mode offsets.
2. Field-coded MBs need field scan tables (column-first) for coefficient placement.
3. `decode_cabac_field_decoding_flag` context depends on above pair's actual field flag,
   not a hardcoded false. This caused cascading wrong field flag values.

CABAC engine (pos/low/range) now matches FFmpeg exactly for all MBAFF pairs.
All 9 MBAFF files produce correct frame counts. Pixel output still differs
(field stride interleaving in reconstruction not yet implemented).

Updated `verify_cabac_tables.py`: now checks 19 tables (was 16), including field-mode variants.

## MBAFF Frame-Mode Implementation (2026-03-25)

Implemented MB pair decode loops for MBAFF (`!frame_mbs_only_flag`):
- `decode_slice_cabac_mbaff()` and `decode_slice_cavlc_mbaff()`: pair iteration
- `decode_cabac_field_decoding_flag()`: reads mb_field_decoding_flag from CABAC context 70
- Dual left-side NeighborContext (top_left/bot_left) for correct pair-interleaved neighbor addressing
- All 9 progressive MBAFF conformance files now produce decoded frames (was 0 before)

**Fixed:** framecrc_compare.py 0-frame false positive bug. Removed FM1_BT_B.h264 (false positive).
Added CI_MW_D.264 and LS_SVA_D.264 to conformance.

New scripts:
- `classify_stream_features.py` — report profile, entropy, MBAFF, resolution, frame count per file
- `audit_conformance_snapshots.py` — detect false positives in conformance snapshots
- `mbaff_cabac_compare.py` — binary search CABAC desync location via lldb + trace comparison

## MBAFF Deblocking Infrastructure (2026-03-25)

Implemented Phases 1-7 of the MBAFF deblocking plan:
- `mb_field: bool` in `MbDeblockInfo` — populated from decode context
- `mvy_limit` (2 for field, 4 for frame) threaded through check_mv/compute_bs
- Field-mode pixel addressing: `deblock_luma_offset`/`deblock_chroma_offset` (doubled stride + field Y offset)
- bS=3 for horizontal interlaced intra MB boundary edges (not vertical — vertical always bS=4 in MBAFF)
- Field-aware above-neighbor: `mb_idx - 2*mb_width` for field MBs (same field of pair above)
- Mixed-interlace first vertical edge: 8 bS values, MBAFF filter (2px/bS luma, 1px/bS chroma)
- Mixed-interlace horizontal edge 0: doubled stride, per-field bS/QP
- CAVLC field scan table selection: `FIELD_SCAN_4X4` for field-mode MBs

102/102 progressive tests remain BITEXACT. CAMA1_Sony_C (MBAFF, mixed field/frame) still DIFF —
remaining diffs come from deblocking iteration order (fixed 2026-03-26) and unknown filter input differences.

New script: `mbaff_field_map.py` — shows per-MB field/frame mode grid for MBAFF files.

## Deferred Automation

- Deblock trace comparison script (compare MB_DEBLOCK sums between wedeo and FFmpeg)

## Conformance Expansion + PPS/Reorder Fixes (2026-03-25)

Added 4 new conformance files:
- **test8b43.264** (Main CAVLC, 960x544, 43 frames): PPS parsing failed because `bit_length` included RBSP stop bit. Added `rbsp_bit_length()` matching FFmpeg's `get_bit_length`.
- **FRExt1_Panasonic.avc** (High CABAC, 8 frames): Output reorder bug — second IDR (poc=0) was output before first-period frames. Fixed by making mid-stream IDRs barriers in delayed_pics.
- **FRExt3_Panasonic.avc** (High CABAC, 11 frames): Already BITEXACT.
- **CI1_FT_B.264** (Constrained Baseline, 291 frames): Already BITEXACT.

Also: SPS/PPS parse failures now log warn!, precommit handles 0-frame files correctly.

New script: `check_dirty_tree.sh` — warns about uncommitted decoder changes at session start.

## Spatial Direct Long-Term Ref Fix (2026-03-25)

**Spatial direct col_zero_flag must skip when L1[0] is long-term.** FFmpeg h264_direct.c lines 374, 405, 443 gate col_zero_flag with `!sl->ref_list[1][0].parent->long_ref`. Without this, the zero-MV suppression is applied incorrectly when the colocated picture is a long-term reference.

Fix: added `col_l1_is_long_term` field to `FrameDecodeContext`, set from DPB entry status, gated col_zero_flag condition. Impact: FRExt_MMCO4_Sony_B.264 59/60 → 60/60 BITEXACT.

New scripts: `check_direct_mode.py`, `ffmpeg_dpb_at_poc.py`, `fix_stale_tracing_features.py`.

Total FRext CABAC: **16/20 BITEXACT**. Remaining: 4 PAFF interlaced.

## CABAC 8x8 + Per-Plane Chroma QP Fix (2026-03-24)

Two bugs fixed:
1. **CABAC CBF skip for cat=5** — For 8x8 luma blocks (cat=5) in non-chroma-4:4:4, there is NO coded_block_flag in the bitstream. We were reading a phantom CBF bin, desyncing the CABAC engine. Fix: skip CBF for cat=5 (FFmpeg h264_cabac.c:1859). Impact: 0/20 → 14/20 FRext CABAC BITEXACT.
2. **Per-plane chroma QP** — `second_chroma_qp_index_offset` (PPS) gives a separate QP offset for Cr. We used offset[0] for both Cb and Cr. Fix: compute `chroma_qp: [u8; 2]` per-plane. Impact: +1 BITEXACT (HPCAQ2LQ_BRCM_B).

Total FRext CABAC: **15/20 BITEXACT** at time of fix. Precommit expanded from 40 → 55 files.

## High Profile 8x8 Transform (2026-03-23)

All structural code is in place (entropy decode, intra pred, dequant/IDCT, deblocking).

**Bugs fixed (2026-03-23):**
1. CAVLC NNZ broadcast — sum written to all 4 sub-blocks → only first (cavlc.rs)
2. pred_8x8l_vert_right — left index `li = y-2x-1` → `y-2x-2` (intra_pred.rs)
3. pred_8x8l_hor_down — top index `ti = x-2y-1` → `x-2y-2` (intra_pred.rs)
4. **dct8x8_allowed guard** — inter `transform_size_8x8_flag` was read unconditionally;
   now gated by sub-partition type check matching FFmpeg's `get_dct8x8_allowed()`
   (P_8x8: all sub_mb_type==0; B_8x8: no sub-8x8 + direct inference check;
   B_Direct_16x16: requires direct_8x8_inference_flag)

**Result:** HPCVNL_BRCM_A.264 **300/300 BITEXACT** (was 21/300 after bugs 1-3).

**New scripts:**
- `scripts/conformance_frext.py` — FRext conformance runner (26 files, CAVLC + CABAC)
- `scripts/probe_h264_features.py` — Probe H.264 files for entropy mode, 8x8, profile, etc.
- `scripts/mb_diff_8x8.py` — Per-MB pixel diff extractor (first differing MB, 16x16 grid dump)
- `scripts/yuv_first_diff.py` — Compare decoded YUV frame 0 between wedeo and FFmpeg
- `scripts/bitpos_compare.py` — Compare per-block CAVLC bit positions between FFmpeg (lldb) and wedeo
- `scripts/transform_flag_compare.py` — Compare per-MB transform_size_8x8_flag between FFmpeg and wedeo
- `scripts/h264_precommit.py` — Combined baseline + CABAC + FRext regression guard (~7s, --quick ~1s)
- `scripts/frext_triage.py` — Categorize FRext DIFF files by failure type (MC/deblock/mixed)

## Dequant Pipeline Fix + Tracing (2026-03-24)

**Bugs fixed:**
1. `dequant4` table built from SPS → PPS (custom scaling matrices now used)
2. `dequant_4x4` now applies `(+32)>>6` matching FFmpeg's STORE_BLOCK macro
3. All `dequant_4x4_flat` replaced with table-based `dequant_4x4` using correct CQM indices
4. Chroma DC dequant now uses per-plane list (Cb vs Cr, intra vs inter)

**Result:** Zero regression (50/50 CAVLC, 27/27 CABAC, HPCVNL_BRCM_A still BITEXACT).
Dequant fix is necessary but not sufficient for Freh1_B (was 0/100).

### Dequant4 Transpose Fix (2026-03-24)

5. `Dequant4Table::new` had spurious `raster_pos = (x>>2)|((x<<2)&0xF)` transpose —
   FFmpeg stores at `x` directly (no transpose). Masked by flat scaling matrices.

**Result:** Freh1_B improved 0/100 → 7/100. Other FRext files unchanged.

### Precommit Infrastructure (2026-03-24)

- `h264_precommit.py` now checks all three suites: baseline CAVLC + CABAC + FRext
- `conformance_full.py --cabac --save-snapshot` now saves to separate `.conformance_cabac_snapshot.json`
  (previously overwrote the CAVLC snapshot)
- **Baseline snapshot is stale** (9/51) — needs re-saving with `conformance_full.py --save-snapshot`

**Pipeline-stage tracing added (7 tags):**
- `SLICE`, `DPB`, `REFLIST` (debug) — decoder.rs
- `PPS_SCALING`, `DEQUANT_TABLES` (debug) — pps.rs, mb.rs
- `COEFF`/`COEFF_CABAC` (trace) — cavlc.rs, cabac.rs
- `DEQUANT` (trace) — mb.rs

**New scripts:**
- `scripts/verify_traces.py` — Verify all pipeline trace tags are reachable
- `scripts/audit_dequant_cqm.py` — Audit dequant CQM indices at all call sites
- `scripts/shodh_watchdog.sh` — Auto-restart shodh server on ONNX re-init panic

## CAVLC 8x8 Deblock NNZ Override Fix (2026-03-24)

**Bug:** CAVLC 8x8 DCT MBs used individual 4x4 sub-block NNZ for deblocking bS,
but FFmpeg's `fill_filter_caches` (h264_slice.c:2396) overrides these with
CBP-based values. If the 8x8 block was coded, ALL sub-blocks are treated as
NNZ>0. Individual sub-block NNZ can be 0 even when the 8x8 block is coded.

**Fix:** Added `deblock_nnz()` helper in `deblock.rs` + `cbp` field in `MbDeblockInfo`.

**Result:** HPCV_BRCM_A 300/300 BITEXACT (was 21/300), Freh1_B 100/100 BITEXACT (was 7/100).
FRext CAVLC: 3/6 BITEXACT. Precommit: 39/39 passing.

**New scripts:**
- `scripts/nnz_compare.py` — Compare stored NNZ arrays between wedeo and FFmpeg via lldb
- `scripts/bs_compare.py` — Compare deblock bS values between wedeo and FFmpeg

## Deferred Automation

- **Per-pixel deblock attribution** — Given a pixel diff, trace back to which specific
  deblock edge/pair/filter caused it (covering both vertical AND horizontal edges).
  Would have saved ~30 min of manual pixel tracing in this session.

## Next Steps

- Expand FRext to CABAC files
- Interlaced (PAFF) support
- Additional video codecs (HEVC, VP9)
