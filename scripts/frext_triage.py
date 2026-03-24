#!/usr/bin/env python3
"""Triage FRext DIFF files by failure category.

For each non-BITEXACT FRext file, determines:
- Entropy mode (CAVLC vs CABAC)
- Frame match ratio
- First differing frame number
- Whether diffs are MC-only or deblock-related (via --no-deblock comparison)

Usage:
    python3 scripts/frext_triage.py                  # full triage
    python3 scripts/frext_triage.py --cavlc           # CAVLC files only
    python3 scripts/frext_triage.py --cabac           # CABAC files only
    python3 scripts/frext_triage.py --skip-nodeblock   # skip slow no-deblock check
"""
# /// script
# requires-python = ">=3.10"
# ///

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformance_frext import FREXT_CABAC_FILES, FREXT_CAVLC_FILES
from ffmpeg_debug import find_wedeo_binary
from framecrc_compare import compare_one

FREXT_DIR_DEFAULT = "fate-suite/h264-conformance/FRext"


def triage_file(
    file_path: str, wedeo_bin: str, check_nodeblock: bool
) -> dict:
    """Triage a single file. Returns a dict with category info."""
    info = {"file": Path(file_path).name}

    try:
        total, matching, diffs, _planes = compare_one(file_path, wedeo_bin)
    except Exception as e:
        info["status"] = "ERROR"
        info["error"] = str(e)
        return info

    if matching == total and total > 0:
        info["status"] = "BITEXACT"
        info["frames"] = total
        return info

    info["status"] = "DIFF"
    info["match"] = matching
    info["total"] = total
    info["first_diff"] = diffs[0] if diffs else None

    if not check_nodeblock:
        return info

    # Check if diffs disappear with deblocking disabled
    try:
        nd_total, nd_matching, nd_diffs, _planes = compare_one(
            file_path, wedeo_bin, no_deblock=True
        )
        if nd_matching == nd_total and nd_total > 0:
            info["category"] = "deblock-only"
        elif nd_matching > matching:
            info["category"] = "mixed (MC+deblock)"
            info["nodeblock_match"] = nd_matching
        else:
            info["category"] = "MC/decode"
            info["nodeblock_match"] = nd_matching
    except Exception:
        info["category"] = "unknown (no-deblock failed)"

    return info


def print_triage(results: list[dict], label: str):
    """Print triage results for a group."""
    if not results:
        return
    print(f"\n--- {label} ---")
    print(f"  {'File':<30} {'Status':<10} {'Frames':<12} {'1st Diff':<10} {'Category'}")
    print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*10} {'-'*20}")
    for r in results:
        name = r["file"]
        status = r["status"]
        if status == "BITEXACT":
            frames = f"{r['frames']}/{r['frames']}"
            print(f"  {name:<30} {status:<10} {frames:<12}")
        elif status == "ERROR":
            print(f"  {name:<30} {status:<10} {'':12} {'':10} {r.get('error', '')}")
        else:
            frames = f"{r['match']}/{r['total']}"
            first = str(r.get("first_diff", "?"))
            cat = r.get("category", "")
            nodeblock = r.get("nodeblock_match")
            extra = f" (nd:{nodeblock}/{r['total']})" if nodeblock is not None else ""
            print(f"  {name:<30} {status:<10} {frames:<12} {first:<10} {cat}{extra}")


def main():
    parser = argparse.ArgumentParser(description="FRext DIFF file triage")
    parser.add_argument("--cavlc", action="store_true", help="CAVLC files only")
    parser.add_argument("--cabac", action="store_true", help="CABAC files only")
    parser.add_argument(
        "--skip-nodeblock",
        action="store_true",
        help="Skip no-deblock check (faster)",
    )
    parser.add_argument("--fate-dir", default=FREXT_DIR_DEFAULT)
    args = parser.parse_args()

    run_cavlc = not args.cabac
    run_cabac = not args.cavlc

    wedeo_bin = find_wedeo_binary()
    if not wedeo_bin:
        print("ERROR: wedeo-framecrc binary not found.", file=sys.stderr)
        sys.exit(2)

    check_nd = not args.skip_nodeblock
    start = time.monotonic()

    if run_cavlc:
        cavlc_results = []
        for name in FREXT_CAVLC_FILES:
            fp = str(Path(args.fate_dir) / name)
            if not Path(fp).exists():
                cavlc_results.append({"file": name, "status": "MISSING"})
                continue
            cavlc_results.append(triage_file(fp, wedeo_bin, check_nd))
        print_triage(cavlc_results, "CAVLC")

    if run_cabac:
        cabac_results = []
        for name in FREXT_CABAC_FILES:
            fp = str(Path(args.fate_dir) / name)
            if not Path(fp).exists():
                cabac_results.append({"file": name, "status": "MISSING"})
                continue
            cabac_results.append(triage_file(fp, wedeo_bin, check_nd))
        print_triage(cabac_results, "CABAC")

    elapsed = time.monotonic() - start
    print(f"\nTriage completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
