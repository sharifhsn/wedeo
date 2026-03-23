#!/usr/bin/env python3
"""Classify conformance errors as reorder-only, deblock-only, or decode errors.

For each DIFF file, runs three comparisons:
  1. Sorted CRC match  -> detects reorder-only issues (pixels correct, output order wrong)
  2. No-deblock compare -> detects deblock-only issues (decode correct, filter wrong)
  3. Pixel-level triage -> first differing frame, per-plane max pixel diff

Usage:
    # Classify all CABAC DIFF files
    python3 scripts/classify_errors.py --cabac

    # Classify all CAVLC DIFF files
    python3 scripts/classify_errors.py

    # Classify specific files
    python3 scripts/classify_errors.py fate-suite/h264-conformance/CABAST3_Sony_E.jsv

Requires:
    - wedeo-framecrc binary (cargo build)
    - ffmpeg binary in PATH
    - fate-suite/h264-conformance/ directory
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformance_full import PROGRESSIVE_CABAC_FILES, PROGRESSIVE_CAVLC_FILES
from ffmpeg_debug import find_wedeo_binary
from framecrc_compare import compare_one


def _run_framecrc_crcs(cmd: list[str], env: dict | None = None) -> list[str]:
    """Run a command and extract CRC hashes from framecrc output."""
    import os
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    result = subprocess.run(cmd, capture_output=True, text=True, env=full_env, timeout=60)
    crcs = []
    for line in result.stdout.splitlines():
        if line.startswith("0,"):
            parts = line.split(",")
            if len(parts) >= 6:
                crcs.append(parts[5].strip())
    return crcs


def classify_file(fpath: Path, wedeo_bin: Path) -> dict:
    """Classify a single file's errors."""
    # Step 1: Normal comparison (with deblock)
    try:
        total, matching, diff_frames, plane_info = compare_one(
            fpath, wedeo_bin, no_deblock=False, pixel_detail=True,
        )
    except Exception as e:
        return {"status": "ERROR", "details": str(e)}

    if not diff_frames:
        return {"status": "BITEXACT", "total": total, "details": f"{total} frames"}

    first_diff = min(diff_frames)

    # Build pixel diff summary from plane_info (tuples: frame_idx, y_max, u_max, v_max)
    max_y = max_u = max_v = 0
    if plane_info:
        for entry in plane_info:
            max_y = max(max_y, entry[1])
            max_u = max(max_u, entry[2])
            max_v = max(max_v, entry[3])
    pixel_str = f"Y<={max_y} U<={max_u} V<={max_v}" if plane_info else "no numpy"

    # Step 2: Sorted CRC comparison -- detect reorder-only
    w_crcs = _run_framecrc_crcs([str(wedeo_bin), str(fpath)])
    f_crcs = _run_framecrc_crcs(
        ["ffmpeg", "-bitexact", "-i", str(fpath), "-f", "framecrc", "-"],
    )
    if sorted(w_crcs) == sorted(f_crcs):
        swapped = sum(1 for a, b in zip(w_crcs, f_crcs) if a != b)
        return {
            "status": "REORDER_ONLY",
            "total": total,
            "matching": matching,
            "first_diff": first_diff,
            "details": f"{swapped} frames swapped (pixels correct)",
        }

    # Step 3: No-deblock comparison using compare_one (handles FFmpeg -skip_loop_filter)
    try:
        nd_total, nd_matching, nd_diffs, _ = compare_one(
            fpath, wedeo_bin, no_deblock=True, pixel_detail=False,
        )
    except Exception:
        nd_matching = 0
        nd_total = total

    if not nd_diffs:
        return {
            "status": "DEBLOCK_ONLY",
            "total": total,
            "matching": matching,
            "first_diff": first_diff,
            "details": f"decode BITEXACT, deblock diffs {total - matching}/{total} [{pixel_str}]",
        }

    return {
        "status": "DECODE_ERROR",
        "total": total,
        "matching": matching,
        "nd_matching": nd_matching,
        "first_diff": first_diff,
        "details": (f"{matching}/{total} deblock, {nd_matching}/{nd_total} no-deblock, "
                    f"first diff frame {first_diff} [{pixel_str}]"),
    }


def main():
    parser = argparse.ArgumentParser(description="Classify conformance errors")
    parser.add_argument("files", nargs="*", help="Specific files to classify")
    parser.add_argument("--cabac", action="store_true", help="Classify CABAC files")
    parser.add_argument("--all", action="store_true", help="All files (CAVLC + CABAC)")
    parser.add_argument("--diff-only", action="store_true",
                        help="Only show non-BITEXACT files")
    parser.add_argument("--fate-dir", default="fate-suite/h264-conformance")
    args = parser.parse_args()

    wedeo_bin = find_wedeo_binary()
    fate_dir = Path(args.fate_dir)

    if args.files:
        file_list = [Path(f).name if Path(f).exists() else f for f in args.files]
        # Resolve full paths
        resolved = []
        for f in args.files:
            p = Path(f)
            resolved.append(p if p.exists() else fate_dir / f)
        file_paths = resolved
    else:
        if args.all:
            file_list = PROGRESSIVE_CAVLC_FILES + PROGRESSIVE_CABAC_FILES
        elif args.cabac:
            file_list = PROGRESSIVE_CABAC_FILES
        else:
            file_list = PROGRESSIVE_CAVLC_FILES
        file_paths = [fate_dir / f for f in file_list]

    icons = {
        "BITEXACT": " ", "REORDER_ONLY": "~", "DEBLOCK_ONLY": "^",
        "DECODE_ERROR": "X", "ERROR": "!",
    }
    counts = {"BITEXACT": 0, "REORDER_ONLY": 0, "DEBLOCK_ONLY": 0, "DECODE_ERROR": 0, "ERROR": 0}

    for fpath in file_paths:
        if not fpath.exists():
            continue

        info = classify_file(fpath, wedeo_bin)
        status = info["status"]
        counts[status] += 1

        if args.diff_only and status == "BITEXACT":
            continue

        print(f"  {icons[status]} {status:14s} {fpath.name}: {info['details']}")

    # Summary
    total_files = sum(counts.values())
    print(f"\n{'=' * 70}")
    print(f"  {counts['BITEXACT']}/{total_files} BITEXACT")
    for cat in ["REORDER_ONLY", "DEBLOCK_ONLY", "DECODE_ERROR", "ERROR"]:
        if counts[cat]:
            label = {
                "REORDER_ONLY": "reorder-only (pixels correct, wrong order)",
                "DEBLOCK_ONLY": "deblock-only (decode correct, filter wrong)",
                "DECODE_ERROR": "decode errors (actual pixel diffs)",
                "ERROR": "errors (couldn't run)",
            }[cat]
            print(f"  {counts[cat]} {label}")
    print(f"{'=' * 70}")

    sys.exit(0 if counts["BITEXACT"] == total_files else 1)


if __name__ == "__main__":
    main()
