#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Compare wedeo vs FFmpeg pixels for a specific MBAFF pair, field-aware.

For a given (mb_x, pair_row), extracts all 32 picture rows covering the pair
from both decoders (no-deblock), then reports which MB in the pair (top field
or bottom field) has the first divergence, and at which block position.

Usage:
    python3 scripts/mbaff_pair_diff.py CAMA1_Sony_C.jsv --pair 19,4
    python3 scripts/mbaff_pair_diff.py CAMA1_Sony_C.jsv --pair 5,8 --frame 0

The pair argument is (mb_x, mb_y_top), where mb_y_top is the even mb_y of the pair.
"""

import argparse
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


def extract_pair_pixels(
    yuv_data: bytes,
    width: int,
    height: int,
    mb_x: int,
    pair_row_top: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the 16x16 pixel blocks for top and bottom field MBs of a pair.

    Returns (top_field_16x16, bottom_field_16x16) arrays.
    For field mode: top = even rows, bottom = odd rows.
    For frame mode: top = rows 0-15, bottom = rows 16-31.
    We extract assuming field interleaving (the common MBAFF case).
    """
    y_size = width * height
    y_plane = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape(height, width)

    col_start = mb_x * 16
    col_end = col_start + 16
    # Pair covers 32 picture rows
    row_start = (pair_row_top // 2) * 32  # pair_row_top is mb_y, pairs are 0/1, 2/3, ...
    row_end = row_start + 32

    if row_end > height or col_end > width:
        print(f"Error: pair ({mb_x},{pair_row_top}) out of bounds "
              f"({width}x{height})", file=sys.stderr)
        sys.exit(1)

    block = y_plane[row_start:row_end, col_start:col_end]

    # Even rows = top field, odd rows = bottom field
    top_field = block[0::2, :]   # rows 0,2,4,...,30 → 16 rows
    bot_field = block[1::2, :]   # rows 1,3,5,...,31 → 16 rows

    return top_field, bot_field


def diff_field(name: str, wedeo: np.ndarray, ffmpeg: np.ndarray) -> int:
    """Compare two 16x16 field blocks. Returns count of differing pixels."""
    diff = np.abs(wedeo.astype(int) - ffmpeg.astype(int))
    n_diff = int(np.count_nonzero(diff))

    if n_diff == 0:
        print(f"  {name}: MATCH (16x16)")
        return 0

    max_diff = int(diff.max())
    mean_diff = float(diff[diff > 0].mean())

    # Find first differing position
    ys, xs = np.where(diff > 0)
    first_y, first_x = int(ys[0]), int(xs[0])
    blk_x, blk_y = first_x // 4, first_y // 4

    print(f"  {name}: {n_diff}/256 pixels differ "
          f"(max={max_diff}, mean={mean_diff:.1f})")
    print(f"    First diff at pixel ({first_x},{first_y}) = "
          f"block ({blk_x},{blk_y})")
    print(f"    wedeo={wedeo[first_y, first_x]} ffmpeg={ffmpeg[first_y, first_x]}")

    # Show first 4 rows of first differing 4x4 block
    by4, bx4 = blk_y * 4, blk_x * 4
    for r in range(4):
        w_row = [int(v) for v in wedeo[by4 + r, bx4:bx4 + 4]]
        f_row = [int(v) for v in ffmpeg[by4 + r, bx4:bx4 + 4]]
        d_row = [abs(a - b) for a, b in zip(w_row, f_row)]
        marker = " <--" if any(d > 0 for d in d_row) else ""
        print(f"    row {r}: w={w_row} f={f_row} d={d_row}{marker}")

    return n_diff


def main():
    parser = argparse.ArgumentParser(
        description="Compare MBAFF pair pixels between wedeo and FFmpeg"
    )
    parser.add_argument("file", help="Conformance file (name or path)")
    parser.add_argument(
        "--pair", required=True,
        help="Pair position as mb_x,mb_y_top (mb_y_top must be even)"
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame number")
    args = parser.parse_args()

    from ffmpeg_debug import resolve_conformance_file
    input_file = resolve_conformance_file(args.file)

    parts = args.pair.split(",")
    mb_x, mb_y_top = int(parts[0]), int(parts[1])
    if mb_y_top % 2 != 0:
        print(f"Error: mb_y_top must be even (got {mb_y_top})", file=sys.stderr)
        sys.exit(1)

    wedeo_bin = find_wedeo_binary()
    info = get_video_info(input_file, wedeo_bin=wedeo_bin, no_deblock=True)
    width, height = info.width, info.height

    print(f"File: {input_file.name}  ({width}x{height})")
    print(f"Pair: MB({mb_x},{mb_y_top}/{mb_y_top + 1})  frame {args.frame}")
    print()

    # Decode YUV from both (no deblock)
    wedeo_data = decode_yuv(input_file, "wedeo", no_deblock=True, wedeo_bin=wedeo_bin)
    ffmpeg_data = decode_yuv(input_file, "ffmpeg", no_deblock=True)

    frame_size = width * height * 3 // 2
    w_frame = wedeo_data[args.frame * frame_size:(args.frame + 1) * frame_size]
    f_frame = ffmpeg_data[args.frame * frame_size:(args.frame + 1) * frame_size]

    if len(w_frame) < frame_size or len(f_frame) < frame_size:
        print("Error: frame data too short", file=sys.stderr)
        sys.exit(1)

    w_top, w_bot = extract_pair_pixels(w_frame, width, height, mb_x, mb_y_top)
    f_top, f_bot = extract_pair_pixels(f_frame, width, height, mb_x, mb_y_top)

    print(f"--- Top field MB({mb_x},{mb_y_top}) ---")
    top_diffs = diff_field("top_field", w_top, f_top)

    print(f"--- Bottom field MB({mb_x},{mb_y_top + 1}) ---")
    bot_diffs = diff_field("bot_field", w_bot, f_bot)

    print()
    if top_diffs == 0 and bot_diffs == 0:
        print("PAIR MATCH — both fields identical")
    elif top_diffs == 0:
        print(f"ROOT CAUSE: bottom field MB({mb_x},{mb_y_top + 1})")
    elif bot_diffs == 0:
        print(f"ROOT CAUSE: top field MB({mb_x},{mb_y_top})")
    else:
        print(f"BOTH FIELDS DIFFER — check top first")


if __name__ == "__main__":
    main()
