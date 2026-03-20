#!/usr/bin/env python3
"""Save and compare H.264 conformance snapshots for regression detection.

Usage:
    # Save current conformance state as baseline
    python3 scripts/conformance_snapshot.py save

    # Save to a specific file
    python3 scripts/conformance_snapshot.py save --file my_baseline.json

    # Compare current state against saved baseline
    python3 scripts/conformance_snapshot.py check

    # Compare against a specific baseline file
    python3 scripts/conformance_snapshot.py check --file my_baseline.json

Output (check mode):
    Shows regressions, improvements, and new BITEXACT files.
    Exits with code 1 if any regressions are detected.

Requires:
    - wedeo-framecrc binary (release build)
    - ffmpeg binary in PATH
    - conformance_report.py in the same directory
"""

import argparse
import json
import os
import sys
from pathlib import Path

from conformance_parse import run_conformance_report

DEFAULT_SNAPSHOT = Path("scripts/.conformance_baseline.json")


def save_snapshot(path: Path):
    """Save current conformance state to a JSON file."""
    _, results = run_conformance_report()
    data = {
        "version": 1,
        "files": {
            name: {"status": s, "match": m, "total": t}
            for name, (s, m, t) in sorted(results.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")

    bitexact = sum(1 for s, _, _ in results.values() if s == "BITEXACT")
    total = len(results)
    print(f"Saved snapshot: {bitexact} BITEXACT / {total} files → {path}")


def check_snapshot(path: Path):
    """Compare current conformance state against saved baseline."""
    if not path.exists():
        print(f"No baseline found at {path}", file=sys.stderr)
        print("Run: python3 scripts/conformance_snapshot.py save", file=sys.stderr)
        sys.exit(2)

    baseline_data = json.loads(path.read_text())
    baseline = {
        name: (f["status"], f["match"], f["total"])
        for name, f in baseline_data["files"].items()
    }
    _, current = run_conformance_report()

    regressions = []
    improvements = []
    new_bitexact = []
    lost_bitexact = []
    disappeared = []

    all_files = sorted(set(baseline) | set(current))

    for name in all_files:
        b = baseline.get(name)
        c = current.get(name)

        # File disappeared from current run — treat as regression
        if b is not None and c is None:
            disappeared.append((name, b[0], b[1], b[2]))
            continue
        # File is new (not in baseline) — informational only
        if b is None:
            continue

        b_status, b_match, b_total = b
        c_status, c_match, c_total = c

        if b_status == "BITEXACT" and c_status != "BITEXACT":
            lost_bitexact.append((name, c_match, c_total))
        elif b_status != "BITEXACT" and c_status == "BITEXACT":
            new_bitexact.append((name, c_total))
        elif c_match < b_match:
            regressions.append((name, b_match, c_match, c_total))
        elif c_match > b_match:
            improvements.append((name, b_match, c_match, c_total))

    # Print results
    b_bitexact = sum(1 for s, _, _ in baseline.values() if s == "BITEXACT")
    c_bitexact = sum(1 for s, _, _ in current.values() if s == "BITEXACT")

    print(f"Conformance: {b_bitexact} → {c_bitexact} BITEXACT")
    print()

    if new_bitexact:
        print(f"NEW BITEXACT ({len(new_bitexact)}):")
        for name, total in new_bitexact:
            print(f"  + {name} ({total} frames)")
        print()

    if lost_bitexact:
        print(f"LOST BITEXACT ({len(lost_bitexact)}):")
        for name, match, total in lost_bitexact:
            print(f"  ! {name} ({match}/{total})")
        print()

    if disappeared:
        print(f"DISAPPEARED ({len(disappeared)}):")
        for name, status, match, total in disappeared:
            print(f"  ! {name}: was {status} {match}/{total}, now missing")
        print()

    if regressions:
        print(f"REGRESSIONS ({len(regressions)}):")
        for name, old, new, total in regressions:
            delta = new - old
            print(f"  - {name}: {old}/{total} → {new}/{total} ({delta:+d})")
        print()

    if improvements:
        total_gain = sum(new - old for _, old, new, _ in improvements)
        print(f"IMPROVEMENTS ({len(improvements)}, +{total_gain} frames):")
        for name, old, new, total in improvements:
            delta = new - old
            print(f"  + {name}: {old}/{total} → {new}/{total} ({delta:+d})")
        print()

    has_regression = bool(lost_bitexact or regressions or disappeared)
    if has_regression:
        print("RESULT: REGRESSIONS DETECTED")
    elif new_bitexact or improvements:
        print("RESULT: improvements, no regressions")
    else:
        print("RESULT: no changes")

    return 1 if has_regression else 0


def main():
    parser = argparse.ArgumentParser(
        description="Save and compare H.264 conformance snapshots."
    )
    parser.add_argument(
        "action", choices=["save", "check"], help="save baseline or check against it"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help=f"snapshot file path (default: {DEFAULT_SNAPSHOT})",
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    if args.action == "save":
        save_snapshot(args.file)
    else:
        sys.exit(check_snapshot(args.file))


if __name__ == "__main__":
    main()
