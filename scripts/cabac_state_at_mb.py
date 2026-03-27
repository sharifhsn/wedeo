#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Compare CABAC engine state between wedeo and FFmpeg at a specific MB.

Extracts (pos, low, range, mb_field_decoding_flag) from both decoders at the
entry to macroblock decode (after field flag decode) and prints side-by-side.

Usage:
    python3 scripts/cabac_state_at_mb.py <h264_file> --mb-x 17 --mb-y 4
    python3 scripts/cabac_state_at_mb.py <h264_file> --mb-x 17 --mb-y 4 --poc 0
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary


def get_ffmpeg_state(input_path: str, mb_x: int, mb_y: int) -> dict | None:
    """Extract CABAC state from FFmpeg at the given MB via lldb.

    Breakpoint is at h264_cabac.c:1966 (after field flag decode, before
    fill_decode_neighbors), matching wedeo's MBAFF_MB_START trace point.
    """
    ffmpeg_g = Path(__file__).resolve().parent.parent / "FFmpeg" / "ffmpeg_g"
    if not ffmpeg_g.exists():
        print(f"ERROR: {ffmpeg_g} not found. Build with --disable-asm.", file=sys.stderr)
        return None

    abs_input = str(Path(input_path).resolve())
    cmds = [
        f'br s -f h264_cabac.c -l 1966 -c "sl->mb_x == {mb_x} && sl->mb_y == {mb_y}"',
        f"r -bitexact -i {abs_input} -f null /dev/null",
        "expression (int)(sl->cabac.bytestream - sl->cabac.bytestream_start)",
        "expression sl->cabac.low",
        "expression sl->cabac.range",
        "expression sl->mb_field_decoding_flag",
        "kill",
        "q",
    ]

    lldb_args = ["lldb", "-b"]
    for c in cmds:
        lldb_args += ["-o", c]
    lldb_args += ["--", str(ffmpeg_g)]

    try:
        proc = subprocess.run(
            lldb_args,
            capture_output=True,
            timeout=30,
            cwd=str(ffmpeg_g.parent),
        )
    except subprocess.TimeoutExpired:
        print("ERROR: lldb timed out", file=sys.stderr)
        return None

    output = proc.stdout.decode(errors="replace")
    vals = re.findall(r"\(int\) \$\d+ = (-?\d+)", output)
    if len(vals) < 4:
        print(f"ERROR: expected 4 values from lldb, got {len(vals)}", file=sys.stderr)
        return None

    return {
        "pos": int(vals[0]),
        "low": int(vals[1]),
        "range": int(vals[2]),
        "field": int(vals[3]),
    }


def get_wedeo_state(
    input_path: str, mb_x: int, mb_y: int, poc: int | None
) -> dict | None:
    """Extract CABAC state from wedeo trace at the given MB."""
    wedeo_bin = find_wedeo_binary()
    if not wedeo_bin:
        print("ERROR: wedeo-framecrc not found", file=sys.stderr)
        return None

    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::decoder=trace"}
    proc = subprocess.run(
        [wedeo_bin, input_path],
        capture_output=True,
        env=env,
        timeout=60,
    )
    stderr = proc.stderr.decode(errors="replace")
    # Strip ANSI escape codes
    stderr = re.sub(r"\x1b\[[0-9;]*m", "", stderr)

    pattern = re.compile(
        r"MBAFF_MB_START mb_x=(\d+) mb_y=(\d+) mb_field=(true|false)"
        r" pos=(\d+) low=(-?\d+) range=(\d+)"
    )

    for line in stderr.splitlines():
        if poc is not None and f"poc={poc}" not in line:
            continue
        m = pattern.search(line)
        if m and int(m.group(1)) == mb_x and int(m.group(2)) == mb_y:
            return {
                "pos": int(m.group(4)),
                "low": int(m.group(5)),
                "range": int(m.group(6)),
                "field": 1 if m.group(3) == "true" else 0,
            }

    return None


def main():
    parser = argparse.ArgumentParser(description="Compare CABAC state at a specific MB")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--mb-x", type=int, required=True, help="MB x coordinate")
    parser.add_argument("--mb-y", type=int, required=True, help="MB y coordinate")
    parser.add_argument("--poc", type=int, default=None, help="Filter by POC (default: first match)")
    args = parser.parse_args()

    print(f"Comparing CABAC state at MB({args.mb_x},{args.mb_y})...")

    ffmpeg = get_ffmpeg_state(args.input, args.mb_x, args.mb_y)
    wedeo = get_wedeo_state(args.input, args.mb_x, args.mb_y, args.poc)

    if not ffmpeg:
        print("Could not get FFmpeg state.")
        sys.exit(2)
    if not wedeo:
        print("Could not get wedeo state.")
        sys.exit(2)

    fields = ["pos", "low", "range", "field"]
    print(f"\n{'Field':<8} {'FFmpeg':>14} {'Wedeo':>14} {'Match':>6}")
    print("-" * 44)
    all_match = True
    for f in fields:
        match = ffmpeg[f] == wedeo[f]
        marker = "OK" if match else "DIFF"
        if not match:
            all_match = False
        print(f"{f:<8} {ffmpeg[f]:>14} {wedeo[f]:>14} {marker:>6}")

    if all_match:
        print(f"\nMB({args.mb_x},{args.mb_y}): CABAC state matches.")
    else:
        print(f"\nMB({args.mb_x},{args.mb_y}): CABAC state DIVERGES!")
        if ffmpeg["pos"] != wedeo["pos"]:
            print(f"  Byte offset delta: {wedeo['pos'] - ffmpeg['pos']}")

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
