#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Show per-MB field/frame mode for MBAFF H.264 files.

Usage:
    python3 scripts/mbaff_field_map.py <h264_file> [--frame N]

Runs wedeo with MBAFF tracing enabled and prints a grid showing which MBs
are field-mode (F) vs frame-mode (.) for each pair. Helps verify that
field-mode deblocking code is actually exercised by a given test file.

Example output:
    Frame 0 (45x30 MBs):
    Row  0-1: . . . . F F . . . .   (2 field pairs at cols 4-5)
    Row  2-3: . . . . . . . . . .   (all frame pairs)
"""

import re
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <h264_file> [--frame N]", file=sys.stderr)
        sys.exit(1)

    h264_file = sys.argv[1]
    target_frame = None
    if "--frame" in sys.argv:
        idx = sys.argv.index("--frame")
        if idx + 1 < len(sys.argv):
            target_frame = int(sys.argv[idx + 1])

    # Run wedeo with MBAFF tracing
    cmd = [
        "cargo", "run", "--release", "--bin", "wedeo-framecrc",
        "-p", "wedeo-fate", "--",
        h264_file,
    ]
    env = {
        "RUST_LOG": "wedeo_codec_h264::decoder=trace",
        "PATH": subprocess.os.environ.get("PATH", ""),
        "HOME": subprocess.os.environ.get("HOME", ""),
        "CARGO_HOME": subprocess.os.environ.get("CARGO_HOME", ""),
        "RUSTUP_HOME": subprocess.os.environ.get("RUSTUP_HOME", ""),
    }

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=120
    )

    # Parse MBAFF_MB_START lines
    # Format: MBAFF_MB_START mb_x=N mb_y=N mb_field=true/false
    pattern = re.compile(
        r"MBAFF_MB_START\s+mb_x=(\d+)\s+mb_y=(\d+)\s+mb_field=(true|false)"
    )

    # Group by frame (delimited by slice POC changes or IDR NALs)
    frames = {}  # frame_idx -> {(mb_x, mb_y): is_field}
    frame_idx = 0
    last_poc = None
    poc_pattern = re.compile(r"poc=(\d+)")

    current_frame = {}
    for line in result.stderr.split("\n"):
        # Track frame boundaries via POC
        poc_match = poc_pattern.search(line)
        if poc_match:
            poc = int(poc_match.group(1))
            if last_poc is not None and poc != last_poc:
                if current_frame:
                    frames[frame_idx] = current_frame
                    current_frame = {}
                    frame_idx += 1
            last_poc = poc

        m = pattern.search(line)
        if m:
            mb_x = int(m.group(1))
            mb_y = int(m.group(2))
            is_field = m.group(3) == "true"
            current_frame[(mb_x, mb_y)] = is_field

    if current_frame:
        frames[frame_idx] = current_frame

    if not frames:
        print("No MBAFF MB data found. Is this an MBAFF file?", file=sys.stderr)
        sys.exit(1)

    # Print results
    for fidx in sorted(frames.keys()):
        if target_frame is not None and fidx != target_frame:
            continue

        data = frames[fidx]
        if not data:
            continue

        max_x = max(x for x, _ in data.keys())
        max_y = max(y for _, y in data.keys())
        mb_width = max_x + 1
        mb_height = max_y + 1

        field_count = sum(1 for v in data.values() if v)
        total = len(data)

        print(f"\nFrame {fidx} ({mb_width}x{mb_height} MBs, "
              f"{field_count}/{total} field-mode):")

        if field_count == 0:
            print("  All MBs are frame-mode (mb_field=false)")
            print("  Field-mode deblocking code is NOT exercised.")
            continue

        if field_count == total:
            print("  All MBs are field-mode (mb_field=true)")
            continue

        # Print pair-by-pair (pairs are mb_y=2k, 2k+1)
        for pair_y in range(0, mb_height, 2):
            row_chars = []
            for x in range(mb_width):
                top_field = data.get((x, pair_y), False)
                bot_field = data.get((x, pair_y + 1), False)
                if top_field or bot_field:
                    row_chars.append("F")
                else:
                    row_chars.append(".")
            line = " ".join(row_chars)
            nfield = sum(1 for c in row_chars if c == "F")
            suffix = f"  ({nfield} field)" if nfield > 0 else ""
            print(f"  Pair {pair_y:2d}-{pair_y+1:2d}: {line}{suffix}")


if __name__ == "__main__":
    main()
