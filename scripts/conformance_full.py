#!/usr/bin/env python3
"""Full H.264 progressive CAVLC conformance test across all 50 files.

Tests every progressive CAVLC conformance file against FFmpeg, reports
BITEXACT/DIFF status for each, and provides a summary count.

Usage:
    # Full conformance report
    python3 scripts/conformance_full.py

    # Quick mode: stop on first regression from known-passing set
    python3 scripts/conformance_full.py --quick

    # With no-deblock triage (shows both with/without deblock)
    python3 scripts/conformance_full.py --triage

    # Save a snapshot of current passing files
    python3 scripts/conformance_full.py --save-snapshot

Requires:
    - wedeo-framecrc binary (cargo build)
    - ffmpeg binary in PATH
    - fate-suite/h264-conformance/ directory
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary
from framecrc_compare import compare_one


# All 51 progressive CAVLC conformance files (excludes interlaced, CABAC, FRExt, FMO)
PROGRESSIVE_CAVLC_FILES = [
    # Baseline (17)
    "BA1_Sony_D.jsv", "SVA_BA1_B.264", "SVA_NL1_B.264", "BAMQ1_JVC_C.264",
    "BA_MW_D.264", "BANM_MW_D.264", "AUD_MW_E.264", "BA2_Sony_F.jsv",
    "BAMQ2_JVC_C.264", "SVA_BA2_D.264", "SVA_NL2_E.264", "BASQP1_Sony_C.jsv",
    "SVA_Base_B.264", "SVA_FM1_E.264", "SVA_CL1_E.264", "BA1_FT_C.264",
    "BA3_SVA_C.264",
    # Main/CAVLC (20)
    "NL1_Sony_D.jsv", "NL2_Sony_H.jsv", "NL3_SVA_E.264",
    "NLMQ1_JVC_C.264", "NLMQ2_JVC_C.264",
    "MR1_MW_A.264", "MR2_MW_A.264", "MR2_TANDBERG_E.264", "MR1_BT_A.h264",
    "MIDR_MW_D.264", "MPS_MW_A.264", "NRF_MW_E.264",
    "CVPCMNL1_SVA_C.264", "CVPCMNL2_SVA_C.264",
    "HCBP1_HHI_A.264", "HCBP2_HHI_A.264",
    "SL1_SVA_B.264", "FM2_SVA_B.264", "FM2_SVA_C.264",
    "MR3_TANDBERG_B.264",
    # Weighted pred / complex (8)
    "CVWP1_TOSHIBA_E.264", "CVWP2_TOSHIBA_E.264",
    "CVWP3_TOSHIBA_E.264", "CVWP5_TOSHIBA_E.264",
    "CVBS3_Sony_C.jsv", "CVSE3_Sony_H.jsv", "CVSEFDFT3_Sony_E.jsv",
    "cvmp_mot_frm0_full_B.26l",
    # MMCO / multi-ref (2)
    "MR4_TANDBERG_C.264", "MR5_TANDBERG_C.264",
    # Hierarchical / crop (2)
    "HCMP1_HHI_A.264", "CVFC1_Sony_C.jsv",
    # Constrained intra (1)
    "CI1_FT_B.264",
    # FMO (1) — FM1_BT_B is BITEXACT (both produce 0 frames)
    # FM1_FT_E.264 excluded: FMO is unimplemented in FFmpeg (h264_ps.c:758,
    # "FMO is not implemented"). FFmpeg's own FATE excludes it. Both decoders
    # produce corrupt output for this file.
    "FM1_BT_B.h264",
]

# 27 progressive CABAC conformance files (frame_mbs_only=1)
PROGRESSIVE_CABAC_FILES = [
    # Tier 1 — Simplest
    "CABA2_SVA_B.264",
    "CANL1_SVA_B.264", "CANL2_SVA_B.264", "CANL3_SVA_B.264", "CANL4_SVA_B.264",
    "CABA1_SVA_B.264", "CABA3_SVA_B.264",
    # Tier 2 — Longer sequences
    "CANL1_Sony_E.jsv", "CANL2_Sony_E.jsv", "CANL3_Sony_C.jsv",
    "CABA1_Sony_D.jsv", "CABA2_Sony_E.jsv", "CABA3_Sony_C.jsv",
    "CANL1_TOSHIBA_G.264", "CABA3_TOSHIBA_E.264",
    # Tier 3 — Special features
    "CACQP3_Sony_D.jsv", "CAQP1_Sony_B.jsv",
    "CAPCM1_Sand_E.264", "CAPCMNL1_Sand_E.264",
    "CAWP1_TOSHIBA_E.264", "CAWP5_TOSHIBA_E.264",
    "camp_mot_frm0_full.26l",
    "src19td.IBP.264",
    # Tier 4 — Edge cases
    "CABACI3_Sony_B.jsv",
    "CABAST3_Sony_E.jsv", "CABASTBR3_Sony_B.jsv",
    "CAPM3_Sony_D.jsv",
]

SNAPSHOT_PATH = Path(__file__).resolve().parent / ".conformance_snapshot.json"
CABAC_SNAPSHOT_PATH = Path(__file__).resolve().parent / ".conformance_cabac_snapshot.json"


def load_snapshot(cabac: bool = False) -> set[str]:
    """Load the set of known-passing files from the snapshot."""
    path = CABAC_SNAPSHOT_PATH if cabac else SNAPSHOT_PATH
    if path.exists():
        data = json.loads(path.read_text())
        return set(data.get("passing", []))
    return set()


def save_snapshot(passing: list[str], cabac: bool = False) -> None:
    """Save the current passing files as a snapshot."""
    path = CABAC_SNAPSHOT_PATH if cabac else SNAPSHOT_PATH
    file_list = PROGRESSIVE_CABAC_FILES if cabac else PROGRESSIVE_CAVLC_FILES
    path.write_text(json.dumps({
        "passing": sorted(passing),
        "count": len(passing),
        "total": len(file_list),
    }, indent=2) + "\n")
    print(f"Snapshot saved: {len(passing)}/{len(file_list)} passing")


def run_full(
    fate_dir: str,
    quick: bool = False,
    triage: bool = False,
    no_deblock: bool = False,
    cabac: bool = False,
) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Run conformance on all files. Returns (passing, diffs)."""
    wedeo_bin = find_wedeo_binary()
    known_passing = load_snapshot(cabac=cabac) if quick else set()

    passing = []
    diffs = []
    skipped = 0

    file_list = PROGRESSIVE_CABAC_FILES if cabac else PROGRESSIVE_CAVLC_FILES
    for fname in file_list:
        fpath = Path(fate_dir) / fname
        if not fpath.exists():
            skipped += 1
            continue

        try:
            total, matching, diff_frames, _ = compare_one(
                fpath, wedeo_bin, no_deblock=no_deblock,
            )
        except Exception as e:
            diffs.append((fname, -1, -1))
            print(f"  ERROR     {fname}: {e}", file=sys.stderr)
            continue

        if not diff_frames:
            passing.append(fname)
            print(f"  BITEXACT  {fname} ({total} frames)")
        else:
            match_str = f"{matching}/{total}"
            diffs.append((fname, matching, total))
            print(f"  DIFF      {fname} ({match_str} match, {len(diff_frames)} differ)")

            if quick and fname in known_passing:
                print(f"\n  REGRESSION DETECTED: {fname} was passing!")
                return passing, diffs

        # Triage mode: also test without deblock for DIFF files
        if triage and diff_frames and not no_deblock:
            total_nd, matching_nd, diff_nd, _ = compare_one(
                fpath, wedeo_bin, no_deblock=True,
            )
            if not diff_nd:
                print(f"    (no-deblock: BITEXACT)")
            else:
                print(f"    (no-deblock: {matching_nd}/{total_nd})")

    if skipped:
        print(f"\n  ({skipped} files not found in {fate_dir})")

    return passing, diffs


def main():
    parser = argparse.ArgumentParser(
        description="Full H.264 progressive CAVLC conformance test",
    )
    parser.add_argument("--fate-dir", default="fate-suite/h264-conformance",
                        help="Path to conformance files")
    parser.add_argument("--quick", action="store_true",
                        help="Stop on first regression from snapshot")
    parser.add_argument("--triage", action="store_true",
                        help="Also test without deblock for DIFF files")
    parser.add_argument("--no-deblock", action="store_true",
                        help="Run all tests without deblocking")
    parser.add_argument("--cabac", action="store_true",
                        help="Test CABAC files instead of CAVLC")
    parser.add_argument("--save-snapshot", action="store_true",
                        help="Save current passing files as snapshot")
    args = parser.parse_args()

    t0 = time.monotonic()
    passing, diffs = run_full(
        args.fate_dir,
        quick=args.quick,
        triage=args.triage,
        no_deblock=args.no_deblock,
        cabac=args.cabac,
    )
    elapsed = time.monotonic() - t0

    total = len(passing) + len(diffs)
    if total == 0:
        print(f"\nNo conformance files found in {args.fate_dir}", file=sys.stderr)
        sys.exit(2)
    pct = 100 * len(passing) / total
    print(f"\n{'='*60}")
    print(f"  {len(passing)}/{total} BITEXACT ({pct:.0f}%) in {elapsed:.1f}s")
    if diffs:
        print(f"  DIFF files:")
        for fname, matching, total_frames in diffs:
            print(f"    {fname}: {matching}/{total_frames}")
    print(f"{'='*60}")

    if args.save_snapshot:
        save_snapshot(passing, cabac=args.cabac)

    sys.exit(0 if not diffs else 1)


if __name__ == "__main__":
    main()
