# Wedeo Progress

## Current Status

See `CLAUDE.md` and `H264.md` for detailed status.

- H.264 CAVLC: 50/51 progressive conformance files BITEXACT (98%)
- H.264 CABAC: 27/27 progressive conformance files BITEXACT (100%)
- H.264 FRext CAVLC: **3/6 BITEXACT** (HPCVNL, HPCV_BRCM_A, Freh1_B; 3 out-of-scope: PAFF/monochrome)
- WAV/PCM pipeline: byte-identical to FFmpeg 8.0.1 across all FATE suite samples
- Audio via symphonia: 28 decoders, 10 demuxers, SNR-verified lossy codecs

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

**Result:** Zero regression (50/51 CAVLC, 27/27 CABAC, HPCVNL_BRCM_A still BITEXACT).
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
