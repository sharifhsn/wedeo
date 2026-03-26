#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Find which MB modifies a specific pixel during deblocking.

Decodes the file with wedeo, compares deblocked vs no-deblock output,
and reports the first differing pixel. Then instruments the deblock
trace to find which MB modifies it.

Usage:
    python3 scripts/deblock_pixel_watch.py <h264_file>
    python3 scripts/deblock_pixel_watch.py <h264_file> --pixel 276,92
    python3 scripts/deblock_pixel_watch.py <h264_file> --frame 0
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import decode_yuv, find_wedeo_binary


def find_first_diff(input_path: str, frame: int = 0) -> tuple[int, int, int, int] | None:
    """Find first differing Y pixel between FFmpeg and wedeo (with deblock).

    Returns (x, y, ffmpeg_val, wedeo_val) or None if BITEXACT.
    """
    wedeo_bin = find_wedeo_binary()
    w_yuv = decode_yuv(input_path, "wedeo", no_deblock=False, wedeo_bin=wedeo_bin)
    f_yuv = decode_yuv(input_path, "ffmpeg", no_deblock=False)

    # Parse dimensions from FFmpeg
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(input_path)],
        capture_output=True, text=True, timeout=10,
    )
    dims = probe.stdout.strip().split(",")
    width, height = int(dims[0]), int(dims[1])
    y_size = width * height
    frame_size = y_size * 3 // 2  # YUV420p

    offset = frame * frame_size
    if offset + y_size > len(w_yuv) or offset + y_size > len(f_yuv):
        print(f"Frame {frame} not available", file=sys.stderr)
        return None

    ff = np.frombuffer(f_yuv[offset:offset + y_size], dtype=np.uint8).reshape(height, width)
    we = np.frombuffer(w_yuv[offset:offset + y_size], dtype=np.uint8).reshape(height, width)

    diff = we.astype(int) - ff.astype(int)
    nonzero = np.argwhere(diff != 0)
    if len(nonzero) == 0:
        return None

    y, x = nonzero[0]
    return (int(x), int(y), int(ff[y, x]), int(we[y, x]))


def main():
    parser = argparse.ArgumentParser(description="Find which MB modifies a pixel during deblocking")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--pixel", help="Specific pixel to watch (x,y), e.g., 276,92")
    parser.add_argument("--frame", type=int, default=0, help="Frame number (default: 0)")
    args = parser.parse_args()

    if args.pixel:
        px, py = map(int, args.pixel.split(","))
        print(f"Watching pixel ({px},{py}) in frame {args.frame}")
    else:
        print(f"Finding first diff in frame {args.frame}...")
        result = find_first_diff(args.input, args.frame)
        if result is None:
            print("BITEXACT — no diffs found!")
            return
        px, py, ff_val, we_val = result
        print(f"First diff at ({px},{py}): FFmpeg={ff_val}, wedeo={we_val}, delta={we_val - ff_val:+d}")

    # Now run wedeo with the pixel watchpoint trace
    # The watchpoint is compiled into deblock.rs — we need to set the pixel address
    # via env var or just grep the trace for PIXEL_WATCHPOINT
    addr = py * 720 + px  # Assumes y_stride=720 for this video
    print(f"\nPixel address in buffer: {addr}")
    print(f"Picture position: ({px},{py})")
    print(f"MB column: {px // 16}, pair row: {py // 32}")

    # Run with deblock trace and grep for watchpoint
    wedeo_bin = find_wedeo_binary()
    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::deblock=trace"}
    proc = subprocess.run(
        [str(wedeo_bin), args.input],
        capture_output=True, env=env, timeout=120,
    )
    # Strip ANSI codes and find watchpoint lines
    output = proc.stderr.decode(errors="replace")
    output = re.sub(r"\x1B\[[0-9;]*m", "", output)

    watchpoints = [l for l in output.split("\n") if "PIXEL_WATCHPOINT" in l]
    if watchpoints:
        # Only show first occurrence (frame 0)
        print(f"\nPixel modified by:")
        print(f"  {watchpoints[0].split('wedeo_codec_h264::deblock: ')[-1]}")
    else:
        print("\nNo PIXEL_WATCHPOINT found — pixel not modified during deblocking")
        print("(Note: the watchpoint address is hardcoded in deblock.rs)")


if __name__ == "__main__":
    main()
