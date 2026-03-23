#!/usr/bin/env python3
"""Conformance delta — compare current results against a saved snapshot.

Shows which files changed status (NEW PASS, NEW FAIL, IMPROVED, REGRESSED)
between the snapshot and the current run. Works for both CAVLC and CABAC.

Usage:
    # Save current CABAC state as baseline
    python3 scripts/conformance_delta.py --cabac --save

    # After code changes, see what changed
    python3 scripts/conformance_delta.py --cabac

    # CAVLC mode (default)
    python3 scripts/conformance_delta.py --save
    python3 scripts/conformance_delta.py

    # Just run without comparing (no snapshot needed)
    python3 scripts/conformance_delta.py --cabac --no-compare

Requires:
    - wedeo-framecrc binary (cargo build)
    - ffmpeg binary in PATH
    - fate-suite/h264-conformance/ directory
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformance_full import PROGRESSIVE_CABAC_FILES, PROGRESSIVE_CAVLC_FILES
from ffmpeg_debug import find_wedeo_binary
from framecrc_compare import compare_one

SNAPSHOT_DIR = Path(__file__).resolve().parent

SNAPSHOT_FILES = {
    "cavlc": SNAPSHOT_DIR / ".conformance_delta_cavlc.json",
    "cabac": SNAPSHOT_DIR / ".conformance_delta_cabac.json",
}


def run_conformance(fate_dir: str, cabac: bool) -> dict[str, dict]:
    """Run conformance on all files, returning per-file results.

    Returns:
        {"filename": {"status": "BITEXACT"|"DIFF"|"ERROR", "matching": N, "total": N}}
    """
    wedeo_bin = find_wedeo_binary()
    file_list = PROGRESSIVE_CABAC_FILES if cabac else PROGRESSIVE_CAVLC_FILES
    results = {}

    for fname in file_list:
        fpath = Path(fate_dir) / fname
        if not fpath.exists():
            continue

        try:
            total, matching, diff_frames, _ = compare_one(fpath, wedeo_bin)
        except Exception as e:
            results[fname] = {"status": "ERROR", "matching": 0, "total": 0, "error": str(e)}
            print(f"  ERROR     {fname}: {e}", file=sys.stderr)
            continue

        if not diff_frames:
            results[fname] = {"status": "BITEXACT", "matching": total, "total": total}
            print(f"  BITEXACT  {fname} ({total} frames)")
        else:
            results[fname] = {"status": "DIFF", "matching": matching, "total": total}
            print(f"  DIFF      {fname} ({matching}/{total} match, {len(diff_frames)} differ)")

    return results


def save_snapshot(results: dict[str, dict], cabac: bool) -> None:
    """Atomically save results as a snapshot."""
    mode = "cabac" if cabac else "cavlc"
    path = SNAPSHOT_FILES[mode]
    data = json.dumps({"mode": mode, "results": results}, indent=2) + "\n"

    # Atomic write: write to temp file in same directory, then rename.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    bitexact = sum(1 for r in results.values() if r["status"] == "BITEXACT")
    total = len(results)
    print(f"\nSnapshot saved: {bitexact}/{total} BITEXACT → {path.name}")


def load_snapshot(cabac: bool) -> dict[str, dict] | None:
    """Load a previously saved snapshot, or None if it doesn't exist."""
    mode = "cabac" if cabac else "cavlc"
    path = SNAPSHOT_FILES[mode]
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("results", {})


def compute_delta(
    old: dict[str, dict], new: dict[str, dict]
) -> tuple[list, list, list, list, list]:
    """Compare old and new results.

    Returns:
        (new_pass, new_fail, improved, regressed, unchanged)

    Each entry is (filename, old_info, new_info) where info is the result dict
    or None for files not present in that run.
    """
    all_files = sorted(set(old) | set(new))
    new_pass = []
    new_fail = []
    improved = []
    regressed = []
    unchanged = []

    for fname in all_files:
        o = old.get(fname)
        n = new.get(fname)

        if o is None and n is not None:
            # File not in old snapshot (new test file)
            if n["status"] == "BITEXACT":
                new_pass.append((fname, o, n))
            else:
                new_fail.append((fname, o, n))
            continue

        if n is None and o is not None:
            # File disappeared from new run (removed or not found)
            continue

        if o is None or n is None:
            continue

        old_ok = o["status"] == "BITEXACT"
        new_ok = n["status"] == "BITEXACT"

        if not old_ok and new_ok:
            new_pass.append((fname, o, n))
        elif old_ok and not new_ok:
            regressed.append((fname, o, n))
        elif not old_ok and not new_ok:
            old_match = o.get("matching", 0)
            new_match = n.get("matching", 0)
            if new_match > old_match:
                improved.append((fname, o, n))
            elif new_match < old_match:
                regressed.append((fname, o, n))
            else:
                unchanged.append((fname, o, n))
        else:
            unchanged.append((fname, o, n))

    return new_pass, new_fail, improved, regressed, unchanged


def format_status(info: dict | None) -> str:
    """Format a result dict as a short status string."""
    if info is None:
        return "(new)"
    if info["status"] == "BITEXACT":
        return f"BITEXACT {info['total']}f"
    if info["status"] == "ERROR":
        return "ERROR"
    return f"DIFF {info['matching']}/{info['total']}"


def print_delta(old: dict[str, dict], new: dict[str, dict]) -> bool:
    """Print the delta report. Returns True if there are regressions."""
    new_pass, new_fail, improved, regressed, unchanged = compute_delta(old, new)

    print()
    has_changes = new_pass or new_fail or improved or regressed

    if new_pass:
        for fname, o, n in new_pass:
            print(f"  NEW PASS    {fname}  (was {format_status(o)}, now {format_status(n)})")

    if improved:
        for fname, o, n in improved:
            print(f"  IMPROVED    {fname}  (was {format_status(o)}, now {format_status(n)})")

    if regressed:
        for fname, o, n in regressed:
            print(f"  REGRESSED   {fname}  (was {format_status(o)}, now {format_status(n)})")

    if new_fail:
        for fname, o, n in new_fail:
            print(f"  NEW FAIL    {fname}  ({format_status(n)})")

    if not has_changes:
        print("  No changes from snapshot.")

    # Summary
    old_bitexact = sum(1 for r in old.values() if r["status"] == "BITEXACT")
    new_bitexact = sum(1 for r in new.values() if r["status"] == "BITEXACT")
    old_total = len(old) or 1  # avoid division by zero
    new_total = len(new) or 1
    old_pct = 100 * old_bitexact / old_total
    new_pct = 100 * new_bitexact / new_total
    diff_count = new_bitexact - old_bitexact
    sign = "+" if diff_count >= 0 else ""

    unchanged_pass = sum(1 for _, _, n in unchanged if n["status"] == "BITEXACT")
    unchanged_diff = sum(1 for _, _, n in unchanged if n["status"] != "BITEXACT")

    print(f"\n  {new_bitexact}/{new_total} BITEXACT ({new_pct:.0f}%)"
          f" — was {old_bitexact}/{old_total} ({old_pct:.0f}%), {sign}{diff_count} files")
    if unchanged_pass or unchanged_diff:
        print(f"  Unchanged: {unchanged_pass} BITEXACT, {unchanged_diff} DIFF")

    return bool(regressed)


def main():
    parser = argparse.ArgumentParser(description="Conformance delta reporter")
    parser.add_argument("--fate-dir", default="fate-suite/h264-conformance",
                        help="Path to conformance files")
    parser.add_argument("--cabac", action="store_true",
                        help="Test CABAC files instead of CAVLC")
    parser.add_argument("--save", action="store_true",
                        help="Save current results as the new snapshot")
    parser.add_argument("--no-compare", action="store_true",
                        help="Run without comparing to snapshot")
    args = parser.parse_args()

    t0 = time.monotonic()
    results = run_conformance(args.fate_dir, args.cabac)
    elapsed = time.monotonic() - t0

    if not results:
        print(f"\nNo conformance files found in {args.fate_dir}", file=sys.stderr)
        sys.exit(2)

    mode = "CABAC" if args.cabac else "CAVLC"
    bitexact = sum(1 for r in results.values() if r["status"] == "BITEXACT")
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  {mode}: {bitexact}/{total} BITEXACT ({100*bitexact/total:.0f}%) in {elapsed:.1f}s")

    has_regression = False
    if not args.no_compare:
        old = load_snapshot(args.cabac)
        if old is None:
            print(f"\n  No snapshot found. Run with --save to create one.")
        else:
            has_regression = print_delta(old, results)

    print(f"{'='*60}")

    if args.save:
        save_snapshot(results, args.cabac)

    sys.exit(1 if has_regression else 0)


if __name__ == "__main__":
    main()
