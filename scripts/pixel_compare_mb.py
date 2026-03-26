#!/usr/bin/env python3
# /// script
# dependencies = ["numpy"]
# ///
"""Compare individual pixel values within a specific MB between wedeo and FFmpeg.

Usage:
    python3 scripts/pixel_compare_mb.py <input> --mb-x X --mb-y Y --frame N
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import decode_yuv, find_wedeo_binary, get_video_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--mb-x", type=int, required=True)
    parser.add_argument("--mb-y", type=int, required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--deblock", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    info = get_video_info(str(input_path))
    w, h = info.width, info.height

    no_deblock = not args.deblock
    wedeo_yuv = decode_yuv(str(input_path), "wedeo", no_deblock=no_deblock)
    ffmpeg_yuv = decode_yuv(str(input_path), "ffmpeg", no_deblock=no_deblock)

    frame_size = w * h * 3 // 2  # YUV 4:2:0
    y_size = w * h

    f = args.frame
    if (f + 1) * frame_size > len(wedeo_yuv) or (f + 1) * frame_size > len(ffmpeg_yuv):
        print(f"Frame {f} out of range")
        sys.exit(1)

    w_y = np.frombuffer(wedeo_yuv[f * frame_size:f * frame_size + y_size], dtype=np.uint8).reshape(h, w)
    f_y = np.frombuffer(ffmpeg_yuv[f * frame_size:f * frame_size + y_size], dtype=np.uint8).reshape(h, w)

    mx, my = args.mb_x, args.mb_y
    py, px = my * 16, mx * 16

    w_mb = w_y[py:py + 16, px:px + 16]
    f_mb = f_y[py:py + 16, px:px + 16]
    diff = w_mb.astype(np.int16) - f_mb.astype(np.int16)

    print(f"MB({mx},{my}) frame {f}, deblock={'on' if args.deblock else 'off'}")
    print(f"max_diff={np.abs(diff).max()}, num_diff={np.count_nonzero(diff)}")
    print()

    # Print 4x4 block grid
    for by in range(4):
        for bx in range(4):
            blk_diff = diff[by * 4:(by + 1) * 4, bx * 4:(bx + 1) * 4]
            if np.any(blk_diff != 0):
                print(f"  Block ({bx},{by}) [4x4 at pixel ({px + bx*4},{py + by*4})]:")
                print(f"    wedeo:  {w_mb[by*4:(by+1)*4, bx*4:(bx+1)*4].tolist()}")
                print(f"    ffmpeg: {f_mb[by*4:(by+1)*4, bx*4:(bx+1)*4].tolist()}")
                print(f"    diff:   {blk_diff.tolist()}")
                print()

    # Also show full MB diff heatmap
    print("Full MB diff:")
    for r in range(16):
        row_str = " ".join(f"{d:+3d}" if d != 0 else "  ." for d in diff[r])
        print(f"  row {r:2d}: {row_str}")


if __name__ == "__main__":
    main()
