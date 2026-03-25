#!/usr/bin/env python3
"""High Profile FRext conformance test for progressive 8x8 transform files.

Tests progressive High Profile conformance files from fate-suite/h264-conformance/FRext/
against FFmpeg, grouped by entropy mode (CAVLC vs CABAC).

Usage:
    # Full FRext conformance report
    python3 scripts/conformance_frext.py

    # CAVLC-only
    python3 scripts/conformance_frext.py --cavlc

    # CABAC-only
    python3 scripts/conformance_frext.py --cabac

    # Save snapshot for regression checking
    python3 scripts/conformance_frext.py --save-snapshot

Requires:
    - wedeo-framecrc binary (cargo build --release)
    - ffmpeg binary in PATH
    - fate-suite/h264-conformance/FRext/ directory
"""
# /// script
# requires-python = ">=3.10"
# ///

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary
from framecrc_compare import compare_one

# Progressive FRext CAVLC files (4:2:0 8-bit).
# Excludes interlaced, 4:2:2, 10-bit, and High444 files.
FREXT_CAVLC_FILES = [
    "Freh1_B.264",
    "HPCV_BRCM_A.264",
    "HPCVFL_BRCM_A.264",
    "HPCVFLNL_BRCM_A.264",
    "HPCVMOLQ_BRCM_B.264",
    "HPCVNL_BRCM_A.264",
    "test8b43.264",
]

FREXT_CABAC_FILES = [
    "FRExt1_Panasonic.avc",
    "FRExt3_Panasonic.avc",
    "Freh12_B.264",
    "Freh2_B.264",
    "Freh7_B.264",
    "freh3.264",
    "freh8.264",
    "freh9.264",
    "FRExt_MMCO4_Sony_B.264",
    "HCAFF1_HHI.264",
    "HCAFR1_HHI.264",
    "HCAFR2_HHI.264",
    "HCAFR3_HHI.264",
    "HCAFR4_HHI.264",
    "HPCA_BRCM_C.264",
    "HPCADQ_BRCM_B.264",
    "HPCAFL_BRCM_C.264",
    "HPCAFLNL_BRCM_C.264",
    "HPCALQ_BRCM_B.264",
    "HPCAMOLQ_BRCM_B.264",
    "HPCANL_BRCM_C.264",
    "HPCAQ2LQ_BRCM_B.264",
]

SNAPSHOT_PATH = Path(__file__).resolve().parent / ".conformance_frext_snapshot.json"


def run_frext(
    fate_dir: str,
    cavlc_only: bool = False,
    cabac_only: bool = False,
) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Run conformance on FRext files. Returns (passing, diffs)."""
    wedeo_bin = find_wedeo_binary()
    passing = []
    diffs = []

    files = []
    if not cabac_only:
        files.extend(("CAVLC", f) for f in FREXT_CAVLC_FILES)
    if not cavlc_only:
        files.extend(("CABAC", f) for f in FREXT_CABAC_FILES)

    current_group = None
    for group, fname in files:
        if group != current_group:
            current_group = group
            print(f"\n--- {group} ---")

        fpath = Path(fate_dir) / "h264-conformance" / "FRext" / fname
        if not fpath.exists():
            print(f"  SKIP      {fname} (file not found)")
            continue

        result = compare_one(str(fpath), wedeo_bin)
        if result is None:
            print(f"  ERROR     {fname}")
            continue

        total, match_count, _diffs, _plane_info = result
        if match_count == total:
            print(f"  BITEXACT  {fname} ({total} frames)")
            passing.append(fname)
        else:
            print(f"  DIFF      {fname} ({match_count}/{total} match, {total - match_count} differ)")
            diffs.append((fname, match_count, total))

    return passing, diffs


def main():
    parser = argparse.ArgumentParser(description="FRext High Profile conformance test")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cavlc", action="store_true", help="CAVLC files only")
    group.add_argument("--cabac", action="store_true", help="CABAC files only")
    parser.add_argument("--save-snapshot", action="store_true", help="Save passing set")
    args = parser.parse_args()

    fate_dir = Path(os.environ.get("FATE_SUITE", "fate-suite"))

    frext_dir = fate_dir / "h264-conformance" / "FRext"
    if not frext_dir.exists():
        print(f"Error: {frext_dir} not found", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()
    passing, diffs = run_frext(str(fate_dir), args.cavlc, args.cabac)
    elapsed = time.monotonic() - t0

    total_files = 0
    if not args.cabac:
        total_files += len(FREXT_CAVLC_FILES)
    if not args.cavlc:
        total_files += len(FREXT_CABAC_FILES)

    print(f"\n{'=' * 60}")
    print(f"  {len(passing)}/{total_files} BITEXACT in {elapsed:.1f}s")
    if diffs:
        print(f"  DIFF files:")
        for fname, match, total in diffs:
            print(f"    {fname}: {match}/{total}")
    print(f"{'=' * 60}")

    if args.save_snapshot:
        SNAPSHOT_PATH.write_text(json.dumps({
            "passing": sorted(passing),
            "count": len(passing),
        }, indent=2) + "\n")
        print(f"Snapshot saved: {len(passing)} passing")


if __name__ == "__main__":
    main()
