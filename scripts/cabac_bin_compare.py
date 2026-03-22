#!/usr/bin/env python3
"""Compare CABAC bin-level traces between FFmpeg and wedeo.

Runs both decoders on the same H.264 input file with CABAC bin tracing enabled,
then parses the trace logs and finds the first diverging bin for each bin type
(regular, bypass, bypass_sign, terminate).

Usage:
    python3 scripts/cabac_bin_compare.py fate-suite/h264-conformance/FI1_Sony_E.264
    python3 scripts/cabac_bin_compare.py file.264 --max-bins 2000
    python3 scripts/cabac_bin_compare.py file.264 --no-deblock
    python3 scripts/cabac_bin_compare.py file.264 --ffmpeg-log /tmp/ff.log --wedeo-log /tmp/we.log
    python3 scripts/cabac_bin_compare.py --parse-only --ffmpeg-log /tmp/ff.log --wedeo-log /tmp/we.log

Prerequisites:
    - Patched FFmpeg binary: scripts/build/ffmpeg_cabac_trace
      (build with: scripts/build_cabac_trace.sh)
    - wedeo-framecrc binary built with cabac-trace feature:
      cargo build --bin wedeo-framecrc -p wedeo-fate --features cabac-trace
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
FFMPEG_TRACE_BIN = SCRIPT_DIR / "build" / "ffmpeg_cabac_trace"
WEDEO_BIN = REPO_DIR / "target" / "debug" / "wedeo-framecrc"


@dataclass
class CabacBin:
    """A single CABAC bin decode event."""

    index: int
    kind: str  # "BIN", "BYPASS", "BYPASS_SIGN", "TERM"
    pre_state: Optional[int]  # only for BIN
    pre_low: int
    pre_range: int
    bit: int
    post_low: int
    post_range: Optional[int]  # not always present
    # extra fields
    val: Optional[int] = None  # for BYPASS_SIGN
    result: Optional[int] = None  # for BYPASS_SIGN


# -- Parsing ------------------------------------------------------------------

# FFmpeg trace line formats (from patched cabac_functions.h):
#   CABAC_BIN <n> state=<s> low=<low> range=<range> -> bit=<b> post_low=<pl> post_range=<pr>
#   CABAC_BYPASS <n> low=<low> range=<range> -> bit=<b> post_low=<pl>
#   CABAC_BYPASS_SIGN <n> low=<low> range=<range> val=<v> -> bit=<b> result=<r> post_low=<pl>
#   CABAC_TERM <n> low=<low> range=<range> -> result=<r> post_low=<pl> post_range=<pr>
#
# Wedeo trace line formats (from cabac.rs with cabac-trace feature):
#   Same format as FFmpeg (designed to match).

RE_BIN = re.compile(
    r"CABAC_BIN\s+(\d+)\s+state=(-?\d+)\s+low=(-?\d+)\s+range=(-?\d+)"
    r"\s+->\s+bit=(-?\d+)\s+post_low=(-?\d+)\s+post_range=(-?\d+)"
)

RE_BYPASS = re.compile(
    r"CABAC_BYPASS\s+(\d+)\s+low=(-?\d+)\s+range=(-?\d+)"
    r"\s+->\s+bit=(-?\d+)\s+post_low=(-?\d+)"
)

RE_BYPASS_SIGN = re.compile(
    r"CABAC_BYPASS_SIGN\s+(\d+)\s+low=(-?\d+)\s+range=(-?\d+)\s+val=(-?\d+)"
    r"\s+->\s+bit=(-?\d+)\s+result=(-?\d+)\s+post_low=(-?\d+)"
)

RE_TERM = re.compile(
    r"CABAC_TERM\s+(\d+)\s+low=(-?\d+)\s+range=(-?\d+)"
    r"\s+->\s+result=(-?\d+)\s+post_low=(-?\d+)\s+post_range=(-?\d+)"
)


def parse_trace_file(path: str) -> dict[str, list[CabacBin]]:
    """Parse a CABAC trace log into categorized bin lists."""
    bins: dict[str, list[CabacBin]] = {
        "BIN": [],
        "BYPASS": [],
        "BYPASS_SIGN": [],
        "TERM": [],
    }

    with open(path) as f:
        for line in f:
            line = line.strip()

            m = RE_BIN.search(line)
            if m:
                bins["BIN"].append(
                    CabacBin(
                        index=int(m.group(1)),
                        kind="BIN",
                        pre_state=int(m.group(2)),
                        pre_low=int(m.group(3)),
                        pre_range=int(m.group(4)),
                        bit=int(m.group(5)),
                        post_low=int(m.group(6)),
                        post_range=int(m.group(7)),
                    )
                )
                continue

            m = RE_BYPASS.search(line)
            if m:
                bins["BYPASS"].append(
                    CabacBin(
                        index=int(m.group(1)),
                        kind="BYPASS",
                        pre_state=None,
                        pre_low=int(m.group(2)),
                        pre_range=int(m.group(3)),
                        bit=int(m.group(4)),
                        post_low=int(m.group(5)),
                        post_range=None,
                    )
                )
                continue

            m = RE_BYPASS_SIGN.search(line)
            if m:
                bins["BYPASS_SIGN"].append(
                    CabacBin(
                        index=int(m.group(1)),
                        kind="BYPASS_SIGN",
                        pre_state=None,
                        pre_low=int(m.group(2)),
                        pre_range=int(m.group(3)),
                        bit=int(m.group(5)),
                        post_low=int(m.group(7)),
                        post_range=None,
                        val=int(m.group(4)),
                        result=int(m.group(6)),
                    )
                )
                continue

            m = RE_TERM.search(line)
            if m:
                bins["TERM"].append(
                    CabacBin(
                        index=int(m.group(1)),
                        kind="TERM",
                        pre_state=None,
                        pre_low=int(m.group(2)),
                        pre_range=int(m.group(3)),
                        bit=int(m.group(4)),  # stored as "result" in trace
                        post_low=int(m.group(5)),
                        post_range=int(m.group(6)),
                    )
                )
                continue

    return bins


# -- Comparison ----------------------------------------------------------------


def compare_bins(
    ff_bins: list[CabacBin],
    we_bins: list[CabacBin],
    kind: str,
    context_lines: int = 3,
) -> Optional[int]:
    """Compare two lists of bins. Returns the index of first divergence, or None."""
    min_len = min(len(ff_bins), len(we_bins))
    diverged_at = None

    for i in range(min_len):
        ff = ff_bins[i]
        we = we_bins[i]
        mismatch = False

        # Compare fields that should match
        if ff.pre_low != we.pre_low:
            mismatch = True
        if ff.pre_range != we.pre_range:
            mismatch = True
        if ff.bit != we.bit:
            mismatch = True
        if ff.post_low != we.post_low:
            mismatch = True
        if kind == "BIN":
            if ff.pre_state != we.pre_state:
                mismatch = True
            if ff.post_range != we.post_range:
                mismatch = True
        if kind == "TERM":
            if ff.post_range != we.post_range:
                mismatch = True

        if mismatch:
            diverged_at = i
            break

    # Print results
    if diverged_at is not None:
        print(f"\n{'='*72}")
        print(f"DIVERGENCE in {kind} bins at index {diverged_at}")
        print(f"{'='*72}")

        # Show context before divergence
        start = max(0, diverged_at - context_lines)
        for j in range(start, min(diverged_at + context_lines + 1, min_len)):
            marker = ">>>" if j == diverged_at else "   "
            ff_b = ff_bins[j]
            we_b = we_bins[j]
            print(f"\n{marker} {kind}[{j}]:")
            if kind == "BIN":
                print(
                    f"    FFmpeg:  state={ff_b.pre_state:3d}  low={ff_b.pre_low:10d}  "
                    f"range={ff_b.pre_range:5d}  -> bit={ff_b.bit}  "
                    f"post_low={ff_b.post_low:10d}  post_range={ff_b.post_range}"
                )
                print(
                    f"    wedeo:   state={we_b.pre_state:3d}  low={we_b.pre_low:10d}  "
                    f"range={we_b.pre_range:5d}  -> bit={we_b.bit}  "
                    f"post_low={we_b.post_low:10d}  post_range={we_b.post_range}"
                )
            elif kind == "BYPASS":
                print(
                    f"    FFmpeg:  low={ff_b.pre_low:10d}  range={ff_b.pre_range:5d}  "
                    f"-> bit={ff_b.bit}  post_low={ff_b.post_low:10d}"
                )
                print(
                    f"    wedeo:   low={we_b.pre_low:10d}  range={we_b.pre_range:5d}  "
                    f"-> bit={we_b.bit}  post_low={we_b.post_low:10d}"
                )
            elif kind == "BYPASS_SIGN":
                print(
                    f"    FFmpeg:  low={ff_b.pre_low:10d}  range={ff_b.pre_range:5d}  "
                    f"val={ff_b.val}  -> bit={ff_b.bit}  result={ff_b.result}  "
                    f"post_low={ff_b.post_low:10d}"
                )
                print(
                    f"    wedeo:   low={we_b.pre_low:10d}  range={we_b.pre_range:5d}  "
                    f"val={we_b.val}  -> bit={we_b.bit}  result={we_b.result}  "
                    f"post_low={we_b.post_low:10d}"
                )
            elif kind == "TERM":
                print(
                    f"    FFmpeg:  low={ff_b.pre_low:10d}  range={ff_b.pre_range:5d}  "
                    f"-> result={ff_b.bit}  post_low={ff_b.post_low:10d}  "
                    f"post_range={ff_b.post_range}"
                )
                print(
                    f"    wedeo:   low={we_b.pre_low:10d}  range={we_b.pre_range:5d}  "
                    f"-> result={we_b.bit}  post_low={we_b.post_low:10d}  "
                    f"post_range={we_b.post_range}"
                )

            # Highlight which fields differ
            if j == diverged_at:
                diffs = []
                if ff_b.pre_low != we_b.pre_low:
                    diffs.append(
                        f"pre_low ({ff_b.pre_low} vs {we_b.pre_low})"
                    )
                if ff_b.pre_range != we_b.pre_range:
                    diffs.append(
                        f"pre_range ({ff_b.pre_range} vs {we_b.pre_range})"
                    )
                if kind == "BIN" and ff_b.pre_state != we_b.pre_state:
                    diffs.append(
                        f"pre_state ({ff_b.pre_state} vs {we_b.pre_state})"
                    )
                if ff_b.bit != we_b.bit:
                    diffs.append(f"bit ({ff_b.bit} vs {we_b.bit})")
                if ff_b.post_low != we_b.post_low:
                    diffs.append(
                        f"post_low ({ff_b.post_low} vs {we_b.post_low})"
                    )
                if (
                    ff_b.post_range is not None
                    and we_b.post_range is not None
                    and ff_b.post_range != we_b.post_range
                ):
                    diffs.append(
                        f"post_range ({ff_b.post_range} vs {we_b.post_range})"
                    )
                print(f"    DIFFERS: {', '.join(diffs)}")

        return diverged_at

    # Check length mismatch
    if len(ff_bins) != len(we_bins):
        print(f"\n{kind}: trace lengths differ: FFmpeg={len(ff_bins)}, wedeo={len(we_bins)}")
        print(f"  (bins matched up to index {min_len - 1})")
        return min_len  # divergence at the point one trace ends

    return None


# -- Running decoders ----------------------------------------------------------


def run_ffmpeg_trace(
    input_file: str, max_bins: int, no_deblock: bool, log_file: str
) -> None:
    """Run the patched FFmpeg binary and capture CABAC trace to log_file."""
    if not FFMPEG_TRACE_BIN.exists():
        print(
            f"Error: patched FFmpeg binary not found at {FFMPEG_TRACE_BIN}",
            file=sys.stderr,
        )
        print(
            "Build it with: scripts/build_cabac_trace.sh",
            file=sys.stderr,
        )
        sys.exit(1)

    env = os.environ.copy()
    env["CABAC_MAX_BINS"] = str(max_bins)

    cmd = [
        str(FFMPEG_TRACE_BIN),
        "-bitexact",
        "-i",
        input_file,
        "-f",
        "null",
        "-",
    ]

    print(f"Running FFmpeg trace: CABAC_MAX_BINS={max_bins} {' '.join(cmd)}")
    print(f"  -> {log_file}")

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=log,
            env=env,
            timeout=60,
        )
        if result.returncode != 0:
            print(
                f"Warning: FFmpeg exited with code {result.returncode}",
                file=sys.stderr,
            )


def run_wedeo_trace(
    input_file: str, max_bins: int, no_deblock: bool, log_file: str
) -> None:
    """Run wedeo-framecrc with cabac-trace and capture trace to log_file."""
    # Build if binary doesn't exist or is older than source
    wedeo_bin = WEDEO_BIN
    if not wedeo_bin.exists():
        print("Building wedeo-framecrc with cabac-trace feature...")
        result = subprocess.run(
            [
                "cargo",
                "build",
                "--bin",
                "wedeo-framecrc",
                "-p",
                "wedeo-fate",
                "--features",
                "cabac-trace",
            ],
            cwd=str(REPO_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Build failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        print("Build OK.")

    env = os.environ.copy()
    env["CABAC_MAX_BINS"] = str(max_bins)
    if no_deblock:
        env["WEDEO_NO_DEBLOCK"] = "1"
    # Enable trace-level logging for CABAC traces (which use tracing::trace!).
    # Only enable the cabac module to avoid noise from other modules.
    env["RUST_LOG"] = "wedeo_codec_h264::cabac=trace"

    cmd = [str(wedeo_bin), input_file]

    print(f"Running wedeo trace: CABAC_MAX_BINS={max_bins} {' '.join(cmd)}")
    print(f"  -> {log_file}")

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=log,
            env=env,
            timeout=60,
        )
        if result.returncode != 0:
            print(
                f"Warning: wedeo exited with code {result.returncode}",
                file=sys.stderr,
            )


# -- Main ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare CABAC bin traces between FFmpeg and wedeo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python3 scripts/cabac_bin_compare.py fate-suite/h264-conformance/FI1_Sony_E.264
  python3 scripts/cabac_bin_compare.py file.264 --max-bins 2000 --no-deblock
  python3 scripts/cabac_bin_compare.py --parse-only --ffmpeg-log /tmp/ff.log --wedeo-log /tmp/we.log
""",
    )
    parser.add_argument("input", nargs="?", help="H.264 input file")
    parser.add_argument(
        "--max-bins",
        type=int,
        default=10000,
        help="Maximum bins to trace per type (default: 10000)",
    )
    parser.add_argument(
        "--no-deblock",
        action="store_true",
        help="Disable deblocking in wedeo (set WEDEO_NO_DEBLOCK=1)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="Context lines around divergence (default: 3)",
    )
    parser.add_argument(
        "--ffmpeg-log",
        help="Path to save/load FFmpeg trace log",
    )
    parser.add_argument(
        "--wedeo-log",
        help="Path to save/load wedeo trace log",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse existing log files, don't run decoders",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Don't rebuild wedeo (use existing binary)",
    )

    args = parser.parse_args()

    if not args.parse_only and not args.input:
        parser.error("input file is required unless --parse-only is used")

    if args.parse_only and (not args.ffmpeg_log or not args.wedeo_log):
        parser.error("--parse-only requires both --ffmpeg-log and --wedeo-log")

    # Determine log file paths
    tmpdir = tempfile.mkdtemp(prefix="cabac_trace_")
    ff_log = args.ffmpeg_log or os.path.join(tmpdir, "ffmpeg_cabac.log")
    we_log = args.wedeo_log or os.path.join(tmpdir, "wedeo_cabac.log")

    # Run decoders (unless parse-only)
    if not args.parse_only:
        run_ffmpeg_trace(args.input, args.max_bins, args.no_deblock, ff_log)
        run_wedeo_trace(args.input, args.max_bins, args.no_deblock, we_log)

    # Parse traces
    print(f"\nParsing FFmpeg trace: {ff_log}")
    ff_bins = parse_trace_file(ff_log)
    print(f"Parsing wedeo trace:  {we_log}")
    we_bins = parse_trace_file(we_log)

    # Summary
    print(f"\n{'='*72}")
    print("CABAC trace summary")
    print(f"{'='*72}")
    for kind in ["BIN", "BYPASS", "BYPASS_SIGN", "TERM"]:
        ff_count = len(ff_bins[kind])
        we_count = len(we_bins[kind])
        status = "OK" if ff_count == we_count else f"MISMATCH ({ff_count} vs {we_count})"
        print(f"  {kind:15s}  FFmpeg={ff_count:6d}  wedeo={we_count:6d}  {status}")

    if all(len(ff_bins[k]) == 0 and len(we_bins[k]) == 0 for k in ff_bins):
        print("\nNo CABAC bins found in either trace.")
        print("Make sure the input file uses CABAC entropy coding (Main/High profile).")
        if not args.parse_only:
            print(f"\nLog files preserved at:\n  {ff_log}\n  {we_log}")
        return

    # Compare each bin type
    any_divergence = False
    first_divergences = {}

    for kind in ["BIN", "BYPASS", "BYPASS_SIGN", "TERM"]:
        if len(ff_bins[kind]) == 0 and len(we_bins[kind]) == 0:
            continue
        idx = compare_bins(ff_bins[kind], we_bins[kind], kind, args.context)
        if idx is not None:
            any_divergence = True
            first_divergences[kind] = idx

    if not any_divergence:
        print(f"\nAll CABAC bins match between FFmpeg and wedeo.")
    else:
        print(f"\n{'='*72}")
        print("DIVERGENCE SUMMARY")
        print(f"{'='*72}")
        for kind, idx in sorted(first_divergences.items(), key=lambda x: x[1]):
            print(f"  First {kind} divergence at index {idx}")

        # Provide debugging hints
        print(f"\nDebugging hints:")
        if "BIN" in first_divergences:
            idx = first_divergences["BIN"]
            ff_b = ff_bins["BIN"][idx] if idx < len(ff_bins["BIN"]) else None
            we_b = we_bins["BIN"][idx] if idx < len(we_bins["BIN"]) else None
            if ff_b and we_b:
                if ff_b.pre_state != we_b.pre_state:
                    print(
                        f"  - State diverged at BIN[{idx}]: the context model was "
                        f"updated differently before this point."
                    )
                    print(
                        f"    This usually means a previous syntax element was "
                        f"decoded incorrectly (wrong bin count, wrong ctx_idx)."
                    )
                elif ff_b.pre_low != we_b.pre_low or ff_b.pre_range != we_b.pre_range:
                    print(
                        f"  - Engine state (low/range) diverged at BIN[{idx}]: "
                        f"the arithmetic engine went out of sync."
                    )
                    print(
                        f"    Check for bugs in refill/refill2 or bit-level init."
                    )

    if not args.parse_only:
        print(f"\nLog files preserved at:\n  {ff_log}\n  {we_log}")


if __name__ == "__main__":
    main()
