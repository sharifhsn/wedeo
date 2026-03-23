#!/usr/bin/env python3
"""Extract per-MB deblocking parameters from FFmpeg via lldb.

Breaks at ff_h264_filter_mb_fast and dumps QP, bS, is_intra, and
chroma QP for specified MBs. Useful for comparing deblocking behavior
between wedeo and FFmpeg at specific macroblock positions.

Usage:
    # Extract deblock info for MB(1,9) on frame 1
    python3 scripts/ffmpeg_extract_deblock.py CABAST3 --mb 1,9 --frame 1

    # Extract for multiple MBs
    python3 scripts/ffmpeg_extract_deblock.py CACQP3 --mb 0,0 --mb 1,0 --frame 0

Requires:
    - FFmpeg debug build at FFmpeg/ffmpeg (--disable-optimizations --enable-debug=3 --disable-asm)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_ffmpeg_binary, resolve_conformance_file


def extract_deblock_info(
    fpath: Path,
    ffmpeg_bin: Path,
    target_mb_x: int,
    target_mb_y: int,
    target_frame: int,
) -> dict | None:
    """Use lldb to extract deblock info from FFmpeg at a specific MB.

    Returns dict with qp, mb_x, mb_y, or None on failure.
    """
    # The fast path: ff_h264_filter_mb_fast is called for each MB.
    # We break there and check mb_x, mb_y, then extract QP values.
    # Since we need a specific frame, we count calls.

    # Approach: use a conditional breakpoint and let lldb run.
    # The h264_filter_mb_fast_internal function is static inline,
    # so we break at ff_h264_filter_mb_fast which calls it.

    # Build lldb commands
    lldb_commands = [
        f"breakpoint set -n ff_h264_filter_mb_fast -c "
        f"\"mb_x == {target_mb_x} && mb_y == {target_mb_y}\"",
        f"run -bitexact -i {fpath} -f null /dev/null",
    ]

    # Skip to the target frame (each hit is one MB at the target position)
    for _ in range(target_frame):
        lldb_commands.append("continue")

    # At the target frame's MB, extract values
    lldb_commands.extend([
        "expression (int)h->cur_pic.qscale_table[sl->mb_xy]",
        "expression (int)sl->mb_x",
        "expression (int)sl->mb_y",
    ])
    lldb_commands.append("quit")

    lldb_cmd = ["lldb", str(ffmpeg_bin), "-b"]
    for c in lldb_commands:
        lldb_cmd.extend(["-o", c])

    result = subprocess.run(
        lldb_cmd,
        capture_output=True, text=True, timeout=60,
    )

    # Parse output
    values = []
    for line in result.stdout.splitlines():
        m = re.match(r"\(int\) \$\d+ = (-?\d+)", line)
        if m:
            values.append(int(m.group(1)))

    if len(values) >= 3:
        return {
            "qp": values[0],
            "mb_x": values[1],
            "mb_y": values[2],
        }

    print(f"lldb extraction failed. Got {len(values)} values.", file=sys.stderr)
    if result.stderr:
        # Show relevant error lines
        for line in result.stderr.splitlines()[-5:]:
            print(f"  {line}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract FFmpeg deblock info via lldb")
    parser.add_argument("file", help="Conformance file (name or path)")
    parser.add_argument("--mb", action="append", required=True,
                        help="MB position as x,y (can repeat)")
    parser.add_argument("--frame", type=int, default=0, help="Frame number (display order)")
    args = parser.parse_args()

    ffmpeg_bin = find_ffmpeg_binary()
    fpath = resolve_conformance_file(args.file)

    print(f"File: {fpath.name}")
    print(f"Frame: {args.frame}")
    print(f"FFmpeg: {ffmpeg_bin}")
    print()

    for mb_spec in args.mb:
        parts = mb_spec.split(",")
        if len(parts) != 2:
            print(f"Invalid MB spec: {mb_spec} (expected x,y)", file=sys.stderr)
            continue
        mb_x, mb_y = int(parts[0]), int(parts[1])

        info = extract_deblock_info(fpath, ffmpeg_bin, mb_x, mb_y, args.frame)
        if info:
            print(f"  MB({mb_x},{mb_y}): qp={info['qp']} "
                  f"(confirmed at mb_x={info['mb_x']} mb_y={info['mb_y']})")
        else:
            print(f"  MB({mb_x},{mb_y}): extraction failed")


if __name__ == "__main__":
    main()
