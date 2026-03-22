#!/usr/bin/env python3
"""Quick regression check against known-passing conformance files.

Loads the snapshot from conformance_full.py --save-snapshot and tests
only those files. Exits non-zero on any regression.

Usage:
    # First, create a snapshot of current state
    python3 scripts/conformance_full.py --save-snapshot

    # Then, after code changes, check for regressions
    python3 scripts/regression_check.py

    # Verbose mode shows each file as it's tested
    python3 scripts/regression_check.py -v

Requires:
    - wedeo-framecrc binary (cargo build)
    - ffmpeg binary in PATH
    - scripts/.conformance_snapshot.json (from conformance_full.py --save-snapshot)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conformance_full import SNAPSHOT_PATH, load_snapshot
from ffmpeg_debug import find_wedeo_binary
from framecrc_compare import compare_one


def main():
    parser = argparse.ArgumentParser(
        description="Quick regression check against known-passing files",
    )
    parser.add_argument("--fate-dir", default="fate-suite/h264-conformance",
                        help="Path to conformance files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show each file as it's tested")
    args = parser.parse_args()

    if not SNAPSHOT_PATH.exists():
        print("No snapshot found. Run: python3 scripts/conformance_full.py --save-snapshot",
              file=sys.stderr)
        sys.exit(2)

    known_passing = sorted(load_snapshot())
    if not known_passing:
        print("Snapshot has no passing files.", file=sys.stderr)
        sys.exit(2)

    wedeo_bin = find_wedeo_binary()
    t0 = time.monotonic()
    regressions = []
    tested = 0
    skipped = 0

    for fname in known_passing:
        fpath = Path(args.fate_dir) / fname
        if not fpath.exists():
            skipped += 1
            if args.verbose:
                print(f"  SKIP  {fname} (not found)")
            continue

        tested += 1
        try:
            total, matching, diff_frames, _ = compare_one(fpath, wedeo_bin)
        except Exception as e:
            regressions.append((fname, -1, -1))
            print(f"  ERROR     {fname}: {e}")
            continue

        if diff_frames:
            regressions.append((fname, matching, total))
            print(f"  REGRESSION  {fname} ({matching}/{total})")
        elif args.verbose:
            print(f"  OK    {fname}")

    elapsed = time.monotonic() - t0

    if regressions:
        print(f"\n{len(regressions)} REGRESSION(S) in {elapsed:.1f}s:")
        for fname, matching, total in regressions:
            print(f"  {fname}: {matching}/{total}")
        sys.exit(1)
    else:
        msg = f"All {tested} known-passing files still pass ({elapsed:.1f}s)"
        if skipped:
            msg += f" ({skipped} not found, skipped)"
        print(msg)
        sys.exit(0)


if __name__ == "__main__":
    main()
