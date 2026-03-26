#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Dump and compare deblocked MB pixels between FFmpeg and wedeo.

Extracts final deblocked pixel values for a specific MB from both
decoders and shows diffs. Uses raw YUV output comparison.

Usage:
    python3 scripts/deblock_mb_dump.py <h264_file> --mb 17,5
    python3 scripts/deblock_mb_dump.py <h264_file> --mb 17,5 --frame 0
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import decode_yuv, find_wedeo_binary


def main():
    parser = argparse.ArgumentParser(description="Compare deblocked MB pixels")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--mb", required=True, help="MB coordinates (x,y), e.g., 17,5")
    parser.add_argument("--frame", type=int, default=0, help="Frame number (default: 0)")
    parser.add_argument("--no-deblock", action="store_true", help="Compare without deblocking")
    args = parser.parse_args()

    mb_x, mb_y = map(int, args.mb.split(","))

    # Get dimensions
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(args.input)],
        capture_output=True, text=True, timeout=10,
    )
    dims = probe.stdout.strip().split(",")
    width, height = int(dims[0]), int(dims[1])
    y_size = width * height
    frame_size = y_size * 3 // 2

    wedeo_bin = find_wedeo_binary()
    no_db = args.no_deblock
    w_yuv = decode_yuv(args.input, "wedeo", no_deblock=no_db, wedeo_bin=wedeo_bin)
    f_yuv = decode_yuv(args.input, "ffmpeg", no_deblock=no_db)

    offset = args.frame * frame_size
    ff = np.frombuffer(f_yuv[offset:offset + y_size], dtype=np.uint8).reshape(height, width)
    we = np.frombuffer(w_yuv[offset:offset + y_size], dtype=np.uint8).reshape(height, width)

    # Extract MB region (16x16 pixels in frame coordinates)
    x0, y0 = mb_x * 16, mb_y * 16
    x1, y1 = min(x0 + 16, width), min(y0 + 16, height)

    ff_mb = ff[y0:y1, x0:x1]
    we_mb = we[y0:y1, x0:x1]

    mode = "no-deblock" if no_db else "deblocked"
    print(f"MB({mb_x},{mb_y}) frame {args.frame} [{mode}], picture region ({x0},{y0})-({x1-1},{y1-1})")

    # Show diffs
    diff = we_mb.astype(int) - ff_mb.astype(int)
    nonzero = np.argwhere(diff != 0)

    if len(nonzero) == 0:
        print("MATCH — zero diffs in this MB")
    else:
        print(f"{len(nonzero)} differing pixels (max |delta|={np.abs(diff).max()}):")
        for dy, dx in nonzero:
            py, px = y0 + dy, x0 + dx
            print(f"  ({px},{py}) [mb_col={dx},mb_row={dy}]: ff={ff_mb[dy, dx]:3d} we={we_mb[dy, dx]:3d} d={diff[dy, dx]:+d}")

    # Also show the full 16x16 grid side by side
    print(f"\nFFmpeg {mode}:")
    for r in range(y1 - y0):
        vals = " ".join(f"{ff_mb[r, c]:3d}" for c in range(x1 - x0))
        print(f"  row{r:2d}: {vals}")

    if len(nonzero) > 0:
        print(f"\nWedeo {mode} (diffs marked with *):")
        for r in range(y1 - y0):
            parts = []
            for c in range(x1 - x0):
                v = we_mb[r, c]
                marker = "*" if diff[r, c] != 0 else " "
                parts.append(f"{v:3d}{marker}")
            print(f"  row{r:2d}: {''.join(parts)}")


if __name__ == "__main__":
    main()
