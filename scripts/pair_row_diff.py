#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Per-pair-row pixel diff summary for MBAFF H.264 files.

Decodes a file with both wedeo and FFmpeg (deblocking disabled), then shows
a per-pair-row summary of pixel differences: first differing MB, max diff,
and number of differing MBs.

Usage:
    python3 scripts/pair_row_diff.py fate-suite/h264-conformance/CAMA1_Sony_C.jsv
    python3 scripts/pair_row_diff.py --deblock <file>    # compare with deblock ON

Requires:
    - wedeo-framecrc binary (cargo build --release)
    - ffmpeg binary in PATH
    - numpy
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary, get_video_info


def decode_raw(filepath: Path, tool: str, no_deblock: bool, wedeo_bin: str) -> bytes:
    """Decode to raw YUV420p."""
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        yuv_path = f.name
    try:
        if tool == "wedeo":
            env = {**os.environ}
            if no_deblock:
                env["WEDEO_NO_DEBLOCK"] = "1"
            subprocess.run(
                [wedeo_bin, str(filepath), "--raw-yuv", yuv_path],
                capture_output=True,
                env=env,
                check=True,
                timeout=30,
            )
        else:
            cmd = ["ffmpeg", "-y", "-bitexact"]
            if no_deblock:
                cmd += ["-skip_loop_filter", "all"]
            cmd += ["-i", str(filepath), "-frames:v", "1", "-f", "rawvideo",
                    "-pix_fmt", "yuv420p", yuv_path]
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        return open(yuv_path, "rb").read()
    finally:
        os.unlink(yuv_path)


def main():
    parser = argparse.ArgumentParser(description="Per-pair-row pixel diff summary")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--deblock", action="store_true", help="Compare with deblocking ON")
    args = parser.parse_args()

    filepath = Path(args.input).resolve()
    wedeo_bin = find_wedeo_binary()
    no_deblock = not args.deblock

    info = get_video_info(filepath, wedeo_bin=wedeo_bin, no_deblock=no_deblock)
    width, height = info.width, info.height
    y_size = width * height
    uv_size = y_size // 4
    uv_w, uv_h = width // 2, height // 2

    w_raw = decode_raw(filepath, "wedeo", no_deblock, wedeo_bin)
    f_raw = decode_raw(filepath, "ffmpeg", no_deblock, wedeo_bin)

    if len(w_raw) < y_size or len(f_raw) < y_size:
        print("Error: decoded YUV too short", file=sys.stderr)
        sys.exit(1)

    w_y = np.frombuffer(w_raw[:y_size], dtype=np.uint8).reshape(height, width)
    f_y = np.frombuffer(f_raw[:y_size], dtype=np.uint8).reshape(height, width)
    w_u = np.frombuffer(w_raw[y_size:y_size + uv_size], dtype=np.uint8).reshape(uv_h, uv_w)
    f_u = np.frombuffer(f_raw[y_size:y_size + uv_size], dtype=np.uint8).reshape(uv_h, uv_w)
    w_v = np.frombuffer(w_raw[y_size + uv_size:y_size + 2 * uv_size], dtype=np.uint8).reshape(uv_h, uv_w)
    f_v = np.frombuffer(f_raw[y_size + uv_size:y_size + 2 * uv_size], dtype=np.uint8).reshape(uv_h, uv_w)

    mb_w = width // 16
    num_pair_rows = height // 32
    mode = "deblock ON" if args.deblock else "deblock OFF"
    print(f"{filepath.name} ({width}x{height}, {mb_w} cols, {num_pair_rows} pair rows, {mode})")
    print()
    print(f"{'Pair':>5} {'mb_y':>6} {'Y_max':>6} {'U_max':>6} {'V_max':>6} {'#MBs':>6} {'First diff':>12}")
    print("-" * 55)

    total_ok = 0
    total_diff = 0
    for pr in range(num_pair_rows):
        yr0, yr1 = pr * 32, pr * 32 + 32
        cr0, cr1 = pr * 16, pr * 16 + 16

        y_diff = np.abs(w_y[yr0:yr1].astype(int) - f_y[yr0:yr1].astype(int))
        u_diff = np.abs(w_u[cr0:cr1].astype(int) - f_u[cr0:cr1].astype(int))
        v_diff = np.abs(w_v[cr0:cr1].astype(int) - f_v[cr0:cr1].astype(int))

        y_max = int(y_diff.max())
        u_max = int(u_diff.max())
        v_max = int(v_diff.max())

        if y_max == 0 and u_max == 0 and v_max == 0:
            print(f"{pr:>5} {pr * 2:>4}-{pr * 2 + 1:<2} {'OK':>6} {'OK':>6} {'OK':>6} {'0':>6} {'—':>12}")
            total_ok += 1
            continue

        # Find first differing MB and count
        first_mb = None
        diff_count = 0
        for mx in range(mb_w):
            for my_off in range(2):
                my = pr * 2 + my_off
                c0, c1 = mx * 16, (mx + 1) * 16
                r0 = my * 16
                yd = np.abs(w_y[r0:r0 + 16, c0:c1].astype(int) - f_y[r0:r0 + 16, c0:c1].astype(int))
                if yd.max() > 0:
                    diff_count += 1
                    if first_mb is None:
                        first_mb = f"MB({mx},{my})"

        total_diff += 1
        print(
            f"{pr:>5} {pr * 2:>4}-{pr * 2 + 1:<2} {y_max:>6} {u_max:>6} {v_max:>6} "
            f"{diff_count:>6} {first_mb or '—':>12}"
        )

    print()
    print(f"{total_ok} OK pair rows, {total_diff} differing pair rows")


if __name__ == "__main__":
    main()
