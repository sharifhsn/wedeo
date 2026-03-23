#!/usr/bin/env python3
"""Check if FFmpeg's CABAC init is stable across runs (ASLR sensitivity).

Runs the patched FFmpeg binary twice on the same file and compares the
initial `low` value at BIN[0]. If they differ, FFmpeg's buffer alignment
changes between runs due to ASLR, making the CABAC init path non-deterministic.

Usage:
    python3 scripts/cabac_aslr_check.py fate-suite/h264-conformance/CANL3_SVA_B.264
    python3 scripts/cabac_aslr_check.py --runs 5 fate-suite/h264-conformance/CANL3_SVA_B.264
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FFMPEG_TRACE_BIN = SCRIPT_DIR / "build" / "ffmpeg_cabac_trace"

RE_BIN = re.compile(
    r"CABAC_BIN\s+0\s+state=(-?\d+)\s+low=(-?\d+)\s+range=(-?\d+)"
    r"\s+->\s+bit=(-?\d+)"
)


def get_bin0_low(input_file: str) -> int | None:
    """Run FFmpeg once and extract BIN[0] low value."""
    if not FFMPEG_TRACE_BIN.exists():
        print(f"Error: {FFMPEG_TRACE_BIN} not found", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["CABAC_MAX_BINS"] = "1"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        with open(log_path, "w") as log_f:
            subprocess.run(
                [str(FFMPEG_TRACE_BIN), "-bitexact", "-i", input_file, "-f", "null", "-"],
                stdout=subprocess.DEVNULL,
                stderr=log_f,
                env=env,
                timeout=30,
            )
        with open(log_path) as f:
            for line in f:
                m = RE_BIN.search(line)
                if m:
                    return int(m.group(2))
        return None
    finally:
        os.unlink(log_path)


def main():
    parser = argparse.ArgumentParser(
        description="Check FFmpeg CABAC init stability across runs (ASLR sensitivity)"
    )
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs (default: 3)"
    )
    args = parser.parse_args()

    print(f"Checking CABAC init stability for {args.input} ({args.runs} runs)...")
    lows = []
    for i in range(args.runs):
        low = get_bin0_low(args.input)
        if low is None:
            print(f"  Run {i+1}: no BIN[0] found (not CABAC?)")
            return
        lows.append(low)
        print(f"  Run {i+1}: BIN[0] low = {low}")

    unique = set(lows)
    if len(unique) == 1:
        print(f"\nSTABLE: all {args.runs} runs produced low={lows[0]}")
        print("FFmpeg's buffer alignment is consistent (no ASLR effect).")
    else:
        print(f"\nUNSTABLE: {len(unique)} distinct low values across {args.runs} runs")
        for v in sorted(unique):
            count = lows.count(v)
            print(f"  low={v} ({count}x)")
        diff = max(unique) - min(unique)
        print(f"  Max difference: {diff}")
        print(
            "\nFFmpeg's init path varies per run due to ASLR. Matching it is unreliable."
        )
        print(
            "Focus on engine arithmetic parity instead of init path matching."
        )


if __name__ == "__main__":
    main()
