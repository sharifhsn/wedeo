#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""MBAFF field map + MB diff overlay for visual debugging.

Combines per-MB field/frame mode info with mb_compare pixel diff data to show
which field-mode vs frame-mode MBs actually differ, helping prioritize
MBAFF debugging.

Usage:
    python3 scripts/mbaff_diff_map.py CAMA1_Sony_C.jsv --frame 0
    python3 scripts/mbaff_diff_map.py CAMA1_Sony_C.jsv --frame 0 --no-deblock

Output legend:
    .  = frame-mode, pixels match
    F  = field-mode, pixels match
    x  = frame-mode, pixels DIFFER
    X  = field-mode, pixels DIFFER
    ?  = no data (out of bounds or decode error)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import (
    decode_yuv,
    find_wedeo_binary,
    get_video_info,
)


def get_field_flags(
    input_file: Path,
    wedeo_bin: Path,
    target_frame: int,
) -> dict[tuple[int, int], bool]:
    """Extract per-MB field flags from wedeo trace output.

    Returns dict mapping (mb_x, mb_y) -> is_field for the target frame.
    """
    env = {
        **dict(os.environ),
        "RUST_LOG": "wedeo_codec_h264::mb=trace",
    }
    result = subprocess.run(
        [str(wedeo_bin), str(input_file)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    raw = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)

    # INTRA_RECON_START has mb_field. Group by frame.
    pattern = re.compile(
        r"INTRA_RECON_START\s+mb_x=(\d+)\s+mb_y=(\d+)\s+mb_field=(\w+)"
    )
    frames: list[dict[tuple[int, int], bool]] = [{}]
    prev_mb_y = -1

    for line in raw.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        mb_x, mb_y = int(m.group(1)), int(m.group(2))
        is_field = m.group(3) == "true"

        # Detect new frame: mb_y resets to 0 or decreases significantly
        if mb_y < prev_mb_y and mb_x == 0 and mb_y == 0:
            frames.append({})
        frames[-1][(mb_x, mb_y)] = is_field
        prev_mb_y = mb_y

    if target_frame < len(frames):
        return frames[target_frame]
    return {}


def get_mb_diffs(
    input_file: Path,
    wedeo_bin: Path,
    target_frame: int,
    no_deblock: bool,
) -> set[tuple[int, int]]:
    """Get set of (mb_x, mb_y) positions with pixel diffs for the target frame.

    Compares raw YUV from both decoders and checks each 16x16 block.
    """
    wedeo_data = decode_yuv(
        input_file, "wedeo", no_deblock=no_deblock, wedeo_bin=wedeo_bin,
    )
    ffmpeg_data = decode_yuv(input_file, "ffmpeg", no_deblock=no_deblock)

    info = get_video_info(input_file, wedeo_bin=wedeo_bin, no_deblock=no_deblock)
    width, height = info.width, info.height
    mb_w, mb_h = info.mb_w, info.mb_h

    frame_size = width * height * 3 // 2
    y_size = width * height

    w_start = target_frame * frame_size
    f_start = target_frame * frame_size

    if w_start + frame_size > len(wedeo_data) or f_start + frame_size > len(ffmpeg_data):
        print("Error: frame data too short", file=sys.stderr)
        return set()

    w_y = np.frombuffer(wedeo_data[w_start:w_start + y_size], dtype=np.uint8).reshape(height, width)
    f_y = np.frombuffer(ffmpeg_data[f_start:f_start + y_size], dtype=np.uint8).reshape(height, width)

    diffs = set()
    for mby in range(mb_h):
        for mbx in range(mb_w):
            r0, c0 = mby * 16, mbx * 16
            r1, c1 = min(r0 + 16, height), min(c0 + 16, width)
            if np.any(w_y[r0:r1, c0:c1] != f_y[r0:r1, c0:c1]):
                diffs.add((mbx, mby))

    return diffs


def main():
    parser = argparse.ArgumentParser(
        description="MBAFF field map + diff overlay"
    )
    parser.add_argument("file", help="Conformance file (name or path)")
    parser.add_argument("--frame", type=int, default=0, help="Frame number")
    parser.add_argument(
        "--no-deblock", action="store_true",
        help="Compare without deblocking (isolates reconstruction)"
    )
    args = parser.parse_args()

    from ffmpeg_debug import resolve_conformance_file
    input_file = resolve_conformance_file(args.file)
    wedeo_bin = find_wedeo_binary()

    info = get_video_info(input_file, wedeo_bin=wedeo_bin, no_deblock=args.no_deblock)
    mb_w, mb_h = info.mb_w, info.mb_h

    print(f"File: {input_file.name}  ({info.width}x{info.height}, {mb_w}x{mb_h} MBs)")
    print(f"Frame: {args.frame}  deblock={'off' if args.no_deblock else 'on'}")
    print()

    # Collect field flags and pixel diffs
    print("Extracting field flags...", file=sys.stderr)
    field_flags = get_field_flags(input_file, wedeo_bin, args.frame)

    print("Computing pixel diffs...", file=sys.stderr)
    diff_set = get_mb_diffs(input_file, wedeo_bin, args.frame, args.no_deblock)

    # Render map
    n_field = 0
    n_frame = 0
    n_field_diff = 0
    n_frame_diff = 0

    # Print column header (tens digit)
    header1 = "        "
    header2 = "        "
    for x in range(mb_w):
        header1 += str(x // 10) if x >= 10 else " "
        header2 += str(x % 10)
    print(header1)
    print(header2)

    for y in range(mb_h):
        row_chars = []
        for x in range(mb_w):
            is_field = field_flags.get((x, y))
            has_diff = (x, y) in diff_set

            if is_field is None:
                row_chars.append("?")
            elif is_field:
                n_field += 1
                if has_diff:
                    n_field_diff += 1
                    row_chars.append("X")
                else:
                    row_chars.append("F")
            else:
                n_frame += 1
                if has_diff:
                    n_frame_diff += 1
                    row_chars.append("x")
                else:
                    row_chars.append(".")
        print(f"  y={y:3d} {''.join(row_chars)}")

    print()
    print(f"Summary: {mb_w * mb_h} MBs total")
    print(f"  Field: {n_field} ({n_field_diff} differ)")
    print(f"  Frame: {n_frame} ({n_frame_diff} differ)")
    print(f"  Total diffs: {n_field_diff + n_frame_diff}")
    if n_field > 0:
        print(f"  Field diff rate: {n_field_diff}/{n_field} = {n_field_diff / n_field:.1%}")
    if n_frame > 0:
        print(f"  Frame diff rate: {n_frame_diff}/{n_frame} = {n_frame_diff / n_frame:.1%}")


if __name__ == "__main__":
    main()
