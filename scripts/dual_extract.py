#!/usr/bin/env python3
"""Generic dual-decoder value extraction and comparison framework.

Provides a reusable pattern for the core debugging workflow:
  1. Extract value X from FFmpeg (via lldb or debug output)
  2. Extract value X from wedeo (via tracing)
  3. Find and display the first divergence

Usage as a library:
    from dual_extract import DualExtractor, ValueSet

    ext = DualExtractor("file.264", frame=2, mb_x=4, mb_y=0)
    wedeo_vals = ext.wedeo_trace("wedeo_codec_h264::mb=trace", parse_fn)
    ffmpeg_vals = ext.ffmpeg_lldb(func, condition, expressions, parse_fn)
    ext.diff(wedeo_vals, ffmpeg_vals, labels)

Usage as a CLI (quick DPB ref count check):
    python3 scripts/dual_extract.py file.264 --check dpb-refs
    python3 scripts/dual_extract.py file.264 --check framecrc

Available checks:
    dpb-refs   Compare DPB short/long ref counts per frame
    framecrc   Compare framecrc output line by line
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ffmpeg_debug import (
    find_ffmpeg_binary,
    find_wedeo_binary,
    run_wedeo_with_tracing,
)


@dataclass
class ValueSet:
    """A named collection of values extracted from one decoder."""
    decoder: str  # "wedeo" or "ffmpeg"
    label: str    # what the values represent
    values: dict  # key -> value (key is usually frame number or block index)


@dataclass
class DiffResult:
    """Result of comparing two ValueSets."""
    first_diff_key: object | None = None
    total_compared: int = 0
    total_matching: int = 0
    diffs: list = field(default_factory=list)  # list of (key, wedeo_val, ffmpeg_val)


class DualExtractor:
    """Framework for extracting and comparing values from both decoders."""

    def __init__(self, input_path: str | Path):
        self.input_path = Path(input_path).resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(f"{self.input_path} not found")

    def wedeo_framecrc(self, no_deblock: bool = False) -> list[str]:
        """Run wedeo-framecrc and return output lines."""
        wedeo_bin = find_wedeo_binary()
        env = dict(os.environ)
        if no_deblock:
            env["WEDEO_NO_DEBLOCK"] = "1"
        result = subprocess.run(
            [str(wedeo_bin), str(self.input_path)],
            capture_output=True, env=env,
        )
        return result.stdout.decode().splitlines()

    def ffmpeg_framecrc(self, no_deblock: bool = False) -> list[str]:
        """Run FFmpeg framecrc and return output lines."""
        ffmpeg_bin = find_ffmpeg_binary()
        cmd = [str(ffmpeg_bin), "-threads", "1", "-bitexact"]
        if no_deblock:
            cmd += ["-skip_loop_filter", "all"]
        cmd += [
            "-i", str(self.input_path),
            "-flags", "+bitexact",
            "-sws_flags", "+accurate_rnd+bitexact",
            "-fflags", "+bitexact",
            "-an", "-f", "framecrc", "-",
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.stdout.decode().splitlines()

    def wedeo_trace(self, rust_log: str) -> str:
        """Run wedeo with tracing and return cleaned trace text."""
        return run_wedeo_with_tracing(
            str(self.input_path),
            rust_log=rust_log,
            no_deblock=True,
            features=["tracing"],
        )

    @staticmethod
    def diff_values(
        wedeo: ValueSet,
        ffmpeg: ValueSet,
        max_diffs: int = 20,
    ) -> DiffResult:
        """Compare two ValueSets and return the diff."""
        all_keys = sorted(set(wedeo.values.keys()) | set(ffmpeg.values.keys()))
        result = DiffResult()

        for key in all_keys:
            w = wedeo.values.get(key)
            f = ffmpeg.values.get(key)
            result.total_compared += 1

            if w == f:
                result.total_matching += 1
            else:
                if result.first_diff_key is None:
                    result.first_diff_key = key
                if len(result.diffs) < max_diffs:
                    result.diffs.append((key, w, f))

        return result


def check_framecrc(input_path: Path, no_deblock: bool = False) -> None:
    """Quick framecrc comparison between wedeo and FFmpeg."""
    ext = DualExtractor(input_path)

    print("Running wedeo...", file=sys.stderr)
    w_lines = [l for l in ext.wedeo_framecrc(no_deblock) if not l.startswith("#") and l.strip()]

    print("Running FFmpeg...", file=sys.stderr)
    f_lines = [l for l in ext.ffmpeg_framecrc(no_deblock) if not l.startswith("#") and l.strip()]

    w_crcs = {}
    for i, line in enumerate(w_lines):
        parts = line.split(",")
        if len(parts) >= 6:
            w_crcs[i] = parts[5].strip()

    f_crcs = {}
    for i, line in enumerate(f_lines):
        parts = line.split(",")
        if len(parts) >= 6:
            f_crcs[i] = parts[5].strip()

    w_set = ValueSet("wedeo", "framecrc", w_crcs)
    f_set = ValueSet("ffmpeg", "framecrc", f_crcs)

    result = ext.diff_values(w_set, f_set)

    total = result.total_compared
    match = result.total_matching
    print(f"\nFrameCRC: {match}/{total} match", end="")
    if result.first_diff_key is not None:
        print(f", first diff at frame {result.first_diff_key}")
        for key, w, f in result.diffs[:10]:
            print(f"  frame {key}: wedeo={w}  ffmpeg={f}")
    else:
        print(" — BITEXACT")


def check_dpb_refs(input_path: Path) -> None:
    """Quick DPB ref count comparison."""
    ext = DualExtractor(input_path)

    print("Extracting wedeo DPB states...", file=sys.stderr)
    trace = ext.wedeo_trace("wedeo_codec_h264::decoder=debug")

    dpb_re = re.compile(
        r"DPB state\s+h264_fn=(\d+)\s+poc=(-?\d+)"
        r"\s+st_count=(\d+).*lt_count=(\d+)"
    )

    w_refs = {}
    frame_idx = 0
    for line in trace.splitlines():
        m = dpb_re.search(line)
        if m:
            poc = int(m.group(2))
            st, lt = int(m.group(3)), int(m.group(4))
            w_refs[frame_idx] = (poc, st, lt)
            frame_idx += 1

    print(f"  {len(w_refs)} frames", file=sys.stderr)

    print(f"\n{'Frame':>5} {'POC':>5} {'ST refs':>7} {'LT refs':>7}")
    print("-" * 30)
    for i in sorted(w_refs.keys())[:50]:
        poc, st, lt = w_refs[i]
        print(f"{i:5d} {poc:5d} {st:7d} {lt:7d}")


def main():
    parser = argparse.ArgumentParser(
        description="Dual-decoder value extraction and comparison"
    )
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--check", choices=["framecrc", "dpb-refs"],
                        required=True, help="Which check to run")
    parser.add_argument("--no-deblock", action="store_true",
                        help="Disable deblocking for framecrc check")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    if args.check == "framecrc":
        check_framecrc(input_path, args.no_deblock)
    elif args.check == "dpb-refs":
        check_dpb_refs(input_path)


if __name__ == "__main__":
    main()
