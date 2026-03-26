# /// script
# /// dependencies = ["numpy"]
# ///
"""Compare deblocking output pixel-by-pixel between two modes.

Usage:
    # Compare wedeo (with deblock) vs FFmpeg (with deblock) at specific MBs:
    uv run scripts/deblock_ab_compare.py fate-suite/h264-conformance/CAMA1_Sony_C.jsv

    # Focus on specific MB:
    uv run scripts/deblock_ab_compare.py CAMA1_Sony_C.jsv --mb 17,5

    # Compare with env var toggle (e.g., iteration order A/B test):
    uv run scripts/deblock_ab_compare.py CAMA1_Sony_C.jsv --env WEDEO_DEBLOCK_RASTER=1
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def find_wedeo_binary():
    for p in [
        "target/release/wedeo-framecrc",
        "target/debug/wedeo-framecrc",
    ]:
        if Path(p).exists():
            return p
    sys.exit("wedeo-framecrc not found; run cargo build --release -p wedeo-fate")


def decode_yuv(input_path, tool, wedeo_bin=None, env_override=None):
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        yuv_path = f.name
    try:
        env = {**os.environ}
        if env_override:
            env.update(env_override)
        if tool == "wedeo":
            subprocess.run(
                [wedeo_bin, str(input_path), "--raw-yuv", yuv_path],
                capture_output=True, env=env, check=True, timeout=120,
            )
        else:
            cmd = ["ffmpeg", "-y", "-bitexact", "-i", str(input_path),
                   "-pix_fmt", "yuv420p", "-f", "rawvideo", yuv_path]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return Path(yuv_path).read_bytes()
    finally:
        Path(yuv_path).unlink(missing_ok=True)


def get_dimensions(input_path, wedeo_bin):
    result = subprocess.run(
        [wedeo_bin, str(input_path)],
        capture_output=True, timeout=60,
    )
    for line in result.stdout.decode().splitlines():
        if "dimensions" in line.lower() or "x" in line:
            pass
    # Fallback: use ffprobe
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(input_path)],
        capture_output=True, timeout=30,
    )
    w, h = r.stdout.decode().strip().split(",")
    return int(w), int(h)


def classify_edge(col, row, mb_w=16, mb_h=16):
    """Classify which deblock edges could have modified this pixel."""
    lx = col % mb_w
    ly = row % mb_h
    edges = []
    # Vertical edges: between columns 4*e-1 and 4*e (e=0..3)
    for e in range(4):
        boundary = e * 4
        # Strong filter (bS=4) modifies p0..p2, q0..q2 = cols boundary-3..boundary+2
        if boundary - 3 <= lx <= boundary + 2:
            side = "p" if lx < boundary else "q"
            dist = boundary - lx if lx < boundary else lx - boundary
            edges.append(f"vert_e{e}_{side}{dist}")
    # Horizontal edges
    for e in range(4):
        boundary = e * 4
        if boundary - 3 <= ly <= boundary + 2:
            side = "p" if ly < boundary else "q"
            dist = boundary - ly if ly < boundary else ly - boundary
            edges.append(f"horiz_e{e}_{side}{dist}")
    return edges


def compare_frames(wy, fy, w, h, frame_idx, mb_filter=None, max_mbs=20):
    """Compare luma planes and report per-MB diffs with edge classification."""
    diff = wy.astype(int) - fy.astype(int)
    abs_diff = np.abs(diff)
    if abs_diff.max() == 0:
        return

    mb_w_count = w // 16
    mb_h_count = h // 16

    print(f"\nFrame {frame_idx}: Y_max={abs_diff.max()}, "
          f"{np.count_nonzero(abs_diff)} differing pixels")

    # Collect per-MB stats
    mb_diffs = []
    for mby in range(mb_h_count):
        for mbx in range(mb_w_count):
            if mb_filter and (mbx, mby) != mb_filter:
                continue
            y0, x0 = mby * 16, mbx * 16
            block = abs_diff[y0:y0+16, x0:x0+16]
            if block.max() > 0:
                mb_diffs.append((mbx, mby, int(block.max()),
                                 int(np.count_nonzero(block))))

    mb_diffs.sort(key=lambda t: -t[2])
    shown = 0
    for mbx, mby, max_d, count in mb_diffs:
        if shown >= max_mbs and not mb_filter:
            print(f"  ... and {len(mb_diffs) - shown} more MBs")
            break
        print(f"  MB({mbx},{mby}): {count}/256 diffs, max_delta={max_d}")
        y0, x0 = mby * 16, mbx * 16
        for dy in range(16):
            row_diffs = []
            for dx in range(16):
                d = int(diff[y0+dy, x0+dx])
                if d != 0:
                    edges = classify_edge(x0+dx, y0+dy)
                    edge_str = ",".join(edges[:2]) if edges else "?"
                    row_diffs.append(
                        f"col{dx}:{wy[y0+dy,x0+dx]}vs{fy[y0+dy,x0+dx]}"
                        f"(d={d:+d} {edge_str})")
            if row_diffs:
                print(f"    row{dy}: {' '.join(row_diffs)}")
        shown += 1


def main():
    parser = argparse.ArgumentParser(
        description="A/B compare deblocking output at pixel level")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--mb", help="Focus on specific MB (e.g., 17,5)")
    parser.add_argument("--env", help="Extra env var for wedeo B mode (e.g., WEDEO_DEBLOCK_RASTER=1)")
    parser.add_argument("--max-frames", type=int, default=1,
                        help="Max frames to compare (default: 1)")
    parser.add_argument("--mode", choices=["wedeo-vs-ffmpeg", "wedeo-ab"],
                        default="wedeo-vs-ffmpeg",
                        help="Comparison mode (default: wedeo-vs-ffmpeg)")
    args = parser.parse_args()

    wb = find_wedeo_binary()
    w, h = get_dimensions(args.input, wb)
    y_size = w * h
    frame_size = y_size + 2 * (w // 2 * h // 2)

    mb_filter = None
    if args.mb:
        parts = args.mb.split(",")
        mb_filter = (int(parts[0]), int(parts[1]))

    env_b = {}
    if args.env:
        k, v = args.env.split("=", 1)
        env_b[k] = v

    if args.mode == "wedeo-vs-ffmpeg":
        print(f"Comparing wedeo vs FFmpeg (with deblock): {args.input}")
        print(f"Dimensions: {w}x{h}")
        a_data = decode_yuv(args.input, "wedeo", wedeo_bin=wb)
        b_data = decode_yuv(args.input, "ffmpeg")
        a_label, b_label = "wedeo", "ffmpeg"
    else:
        print(f"Comparing wedeo (default) vs wedeo ({args.env}): {args.input}")
        print(f"Dimensions: {w}x{h}")
        a_data = decode_yuv(args.input, "wedeo", wedeo_bin=wb)
        b_data = decode_yuv(args.input, "wedeo", wedeo_bin=wb,
                            env_override=env_b)
        a_label, b_label = "default", args.env

    n_frames_a = len(a_data) // frame_size
    n_frames_b = len(b_data) // frame_size
    n_frames = min(n_frames_a, n_frames_b, args.max_frames)

    if n_frames_a != n_frames_b:
        print(f"WARNING: frame count mismatch: {a_label}={n_frames_a}, "
              f"{b_label}={n_frames_b}")

    total_diffs = 0
    for fi in range(n_frames):
        base = fi * frame_size
        ay = np.frombuffer(a_data[base:base+y_size], dtype=np.uint8).reshape(h, w)
        by = np.frombuffer(b_data[base:base+y_size], dtype=np.uint8).reshape(h, w)
        if np.array_equal(ay, by):
            print(f"Frame {fi}: MATCH")
        else:
            total_diffs += 1
            compare_frames(ay, by, w, h, fi, mb_filter)

    if total_diffs == 0:
        print("All frames match!")


if __name__ == "__main__":
    main()
