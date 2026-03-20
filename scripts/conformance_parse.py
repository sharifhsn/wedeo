#!/usr/bin/env python3
"""Shared parser for conformance_report.py output.

Used by conformance_snapshot.py, conformance_diff.py, and classify_diff_files.py.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def parse_report(text: str) -> dict[str, tuple[str, int, int]]:
    """Parse conformance_report.py output into {name: (status, match, total)}.

    Handles all output categories:
    - BITEXACT: "  filename (N frames)"
    - DIFF:     "  filename: M/N match ..."
    - FAIL:     "  filename: error message" → (FAIL, 0, 0)
    - SKIP:     "  filename: reason (FFmpeg: N frames)" → (SKIP, 0, N)
    """
    results = {}
    current_status = None

    for line in text.splitlines():
        if line.startswith("=== BITEXACT"):
            current_status = "BITEXACT"
        elif line.startswith("=== DIFF"):
            current_status = "DIFF"
        elif line.startswith("=== FAIL"):
            current_status = "FAIL"
        elif line.startswith("=== SKIP"):
            current_status = "SKIP"
        elif current_status and line.startswith("  "):
            line = line.strip()

            if current_status == "BITEXACT":
                # "BA1_FT_C.264 (299 frames)"
                m = re.match(r"(\S+)\s+\((\d+)\s+frames\)", line)
                if m:
                    total = int(m.group(2))
                    results[m.group(1)] = ("BITEXACT", total, total)

            elif current_status in ("DIFF", "FAIL"):
                # DIFF: "CVBS3_Sony_C.jsv: 101/300 match ..."
                m = re.match(r"(\S+):\s+(\d+)/(\d+)\s+match", line)
                if m:
                    results[m.group(1)] = (
                        current_status,
                        int(m.group(2)),
                        int(m.group(3)),
                    )
                else:
                    # FAIL: "filename: error message" or "filename: Wedeo produced 0 frames"
                    m = re.match(r"(\S+):", line)
                    if m:
                        # Try to extract FFmpeg frame count from error msg
                        fc = re.search(r"FFmpeg:\s*(\d+)\s+frames", line)
                        total = int(fc.group(1)) if fc else 0
                        results[m.group(1)] = (current_status, 0, total)

            elif current_status == "SKIP":
                # "filename: reason (FFmpeg: N frames)"
                m = re.match(r"(\S+):", line)
                if m:
                    fc = re.search(r"FFmpeg:\s*(\d+)\s+frames", line)
                    total = int(fc.group(1)) if fc else 0
                    results[m.group(1)] = ("SKIP", 0, total)

    return results


def run_conformance_report() -> tuple[str, dict[str, tuple[str, int, int]]]:
    """Run conformance_report.py and return (raw_text, parsed_results).

    Exits with code 2 if the report script fails.
    """
    os.chdir(Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/conformance_report.py",
            "--cavlc-only",
            "--progressive-only",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"conformance_report.py failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(2)

    parsed = parse_report(result.stdout)
    if not parsed:
        print(
            "conformance_report.py produced no parseable results", file=sys.stderr
        )
        sys.exit(2)

    return result.stdout, parsed
