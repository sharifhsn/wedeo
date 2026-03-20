#!/usr/bin/env python3
"""Compare two conformance report runs and show differences.

Usage:
    # Compare saved baseline against current state
    python3 scripts/conformance_diff.py baseline.txt

    # Compare two report files
    python3 scripts/conformance_diff.py before.txt after.txt

    # Generate a report file: redirect conformance_report.py output
    python3 scripts/conformance_report.py --cavlc-only --progressive-only > before.txt
    # ... make changes, rebuild ...
    python3 scripts/conformance_report.py --cavlc-only --progressive-only > after.txt
    python3 scripts/conformance_diff.py before.txt after.txt

Exit codes:
    0 = no regressions
    1 = regressions detected
"""

import argparse
import os
import sys
from pathlib import Path

from conformance_parse import parse_report, run_conformance_report


def diff_reports(
    before: dict[str, tuple[str, int, int]], after: dict[str, tuple[str, int, int]]
):
    """Compare two reports and print differences. Returns exit code."""
    new_bitexact = []
    lost_bitexact = []
    regressions = []
    improvements = []
    disappeared = []

    all_files = sorted(set(before) | set(after))
    for name in all_files:
        b = before.get(name)
        a = after.get(name)

        # File disappeared from "after" — treat as regression
        if b is not None and a is None:
            disappeared.append((name, b[0], b[1], b[2]))
            continue
        # File is new (not in "before") — informational only
        if b is None:
            continue

        b_status, b_match, b_total = b
        a_status, a_match, a_total = a

        if b_status == "BITEXACT" and a_status != "BITEXACT":
            lost_bitexact.append((name, a_match, a_total))
        elif b_status != "BITEXACT" and a_status == "BITEXACT":
            new_bitexact.append((name, a_total))
        elif a_match < b_match:
            regressions.append((name, b_match, a_match, a_total))
        elif a_match > b_match:
            improvements.append((name, b_match, a_match, a_total))

    b_bitexact = sum(1 for s, _, _ in before.values() if s == "BITEXACT")
    a_bitexact = sum(1 for s, _, _ in after.values() if s == "BITEXACT")
    print(f"BITEXACT: {b_bitexact} → {a_bitexact} ({a_bitexact - b_bitexact:+d})")

    if not (new_bitexact or lost_bitexact or regressions or improvements or disappeared):
        print("No changes.")
        return 0

    print()

    if new_bitexact:
        for name, total in new_bitexact:
            print(f"  + {name}: BITEXACT ({total} frames)")

    if improvements:
        total_gain = sum(new - old for _, old, new, _ in improvements)
        for name, old, new, total in improvements:
            print(f"  + {name}: {old}/{total} → {new}/{total} ({new - old:+d})")
        print(f"  (+{total_gain} frames total)")

    if lost_bitexact:
        for name, match, total in lost_bitexact:
            print(f"  ! {name}: BITEXACT → {match}/{total}")

    if disappeared:
        for name, status, match, total in disappeared:
            print(f"  ! {name}: was {status} {match}/{total}, now missing")

    if regressions:
        for name, old, new, total in regressions:
            print(f"  - {name}: {old}/{total} → {new}/{total} ({new - old:+d})")

    print()
    if lost_bitexact or regressions or disappeared:
        print("REGRESSIONS DETECTED")
        return 1
    else:
        print("No regressions.")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Compare conformance report runs."
    )
    parser.add_argument("before", type=Path, help="baseline report file")
    parser.add_argument(
        "after",
        type=Path,
        nargs="?",
        help="current report file (omit to run live)",
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    if not args.before.exists():
        print(f"File not found: {args.before}", file=sys.stderr)
        sys.exit(2)
    before = parse_report(args.before.read_text())
    if not before:
        print(f"No parseable results in {args.before}", file=sys.stderr)
        sys.exit(2)

    if args.after:
        if not args.after.exists():
            print(f"File not found: {args.after}", file=sys.stderr)
            sys.exit(2)
        after = parse_report(args.after.read_text())
    else:
        _, after = run_conformance_report()

    sys.exit(diff_reports(before, after))


if __name__ == "__main__":
    main()
