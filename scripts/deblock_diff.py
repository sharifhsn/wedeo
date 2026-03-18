#!/usr/bin/env python3
"""Analyze deblocking filter differences between wedeo and FFmpeg.

Decodes a H.264 file with and without deblocking from both decoders,
then identifies exactly which chroma/luma edges differ and computes
the applied delta at each differing edge.

Usage:
    python3 scripts/deblock_diff.py fate-suite/h264-conformance/BA_MW_D.264
    python3 scripts/deblock_diff.py BA_MW_D.264 --frame 0
    python3 scripts/deblock_diff.py BA_MW_D.264 --frame 0 --plane U

Requirements:
    - wedeo-framecrc binary (target/debug or target/release)
    - ffmpeg in PATH
    - numpy
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def find_wedeo_bin() -> str:
    for path in ["target/debug/wedeo-framecrc", "target/release/wedeo-framecrc"]:
        if os.path.isfile(path):
            return path
    print("Error: wedeo-framecrc not found. Run: cargo build --bin wedeo-framecrc")
    sys.exit(1)


def get_dimensions(wedeo_bin: str, input_path: str) -> tuple[int, int, int]:
    """Get width, height, frame_count from wedeo framecrc output."""
    result = subprocess.run([wedeo_bin, input_path], capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode().strip()
        print(f"Error: wedeo-framecrc failed: {stderr[:200]}")
        sys.exit(1)
    lines = result.stdout.decode().splitlines()
    width = height = 0
    frame_count = 0
    for line in lines:
        if line.startswith("#dimensions"):
            m = re.search(r'(\d+)x(\d+)', line)
            if m:
                width, height = int(m.group(1)), int(m.group(2))
        elif not line.startswith("#") and line.strip():
            frame_count += 1
    return width, height, frame_count


def decode_yuv(wedeo_bin: str, input_path: str, deblock: bool) -> bytes:
    """Decode to raw YUV420p using wedeo."""
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        yuv_path = f.name
    try:
        env = {**os.environ}
        if not deblock:
            env["WEDEO_NO_DEBLOCK"] = "1"
        subprocess.run(
            [wedeo_bin, input_path, "--raw-yuv", yuv_path],
            capture_output=True, env=env, check=True
        )
        return Path(yuv_path).read_bytes()
    finally:
        if os.path.exists(yuv_path):
            os.unlink(yuv_path)


def decode_yuv_ffmpeg(input_path: str, deblock: bool) -> bytes:
    """Decode to raw YUV420p using FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        yuv_path = f.name
    try:
        cmd = ["ffmpeg", "-y", "-bitexact", "-threads", "1"]
        if not deblock:
            cmd += ["-skip_loop_filter", "all"]
        cmd += ["-i", input_path, "-pix_fmt", "yuv420p", "-f", "rawvideo", yuv_path]
        subprocess.run(cmd, capture_output=True, check=True)
        return Path(yuv_path).read_bytes()
    finally:
        if os.path.exists(yuv_path):
            os.unlink(yuv_path)


def analyze_edge(plane_name: str, pre_w: np.ndarray, post_w: np.ndarray,
                 post_ff: np.ndarray, y_edge: int, x_start: int, x_end: int):
    """Analyze a single horizontal edge and compute applied deltas."""
    for x in range(x_start, x_end):
        p1 = int(pre_w[y_edge - 2, x])
        p0 = int(pre_w[y_edge - 1, x])
        q0 = int(pre_w[y_edge, x])
        q1 = int(pre_w[y_edge + 1, x])

        w_p0 = int(post_w[y_edge - 1, x])
        ff_p0 = int(post_ff[y_edge - 1, x])

        delta_w = w_p0 - p0
        delta_ff = ff_p0 - p0

        if delta_w != delta_ff:
            raw_delta = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3
            clamped = abs(raw_delta) > abs(delta_w) or abs(raw_delta) > abs(delta_ff)
            clamp_note = " (clamped)" if clamped else ""
            print(f"    {plane_name} x={x}: p1={p1} p0={p0} q0={q0} q1={q1} | "
                  f"raw_delta={raw_delta:+d} | "
                  f"wedeo_delta={delta_w:+d} ffmpeg_delta={delta_ff:+d}{clamp_note}")


def main():
    parser = argparse.ArgumentParser(description="Analyze deblocking filter differences")
    parser.add_argument("input", type=Path, help="H.264 input file")
    parser.add_argument("--frame", type=int, default=0, help="Frame to analyze (default: 0)")
    parser.add_argument("--plane", choices=["Y", "U", "V", "all"], default="all",
                        help="Plane to analyze (default: all)")
    parser.add_argument("--max-edges", type=int, default=10,
                        help="Max edges to report per plane (default: 10)")
    args = parser.parse_args()

    wedeo_bin = find_wedeo_bin()
    input_path = str(args.input)

    w, h, n_frames = get_dimensions(wedeo_bin, input_path)
    if w == 0 or h == 0:
        print("Error: could not determine frame dimensions")
        sys.exit(1)
    print(f"Dimensions: {w}x{h}, {n_frames} frames")

    if n_frames == 0:
        print("Error: no frames decoded")
        sys.exit(1)
    if args.frame >= n_frames:
        print(f"Error: frame {args.frame} >= {n_frames} total frames")
        sys.exit(1)

    # Decode all versions
    print("Decoding (4 runs)...")
    wedeo_nd = decode_yuv(wedeo_bin, input_path, deblock=False)
    wedeo_db = decode_yuv(wedeo_bin, input_path, deblock=True)
    ffmpeg_db = decode_yuv_ffmpeg(input_path, deblock=True)
    ffmpeg_nd = decode_yuv_ffmpeg(input_path, deblock=False)

    y_size = w * h
    # YUV420p chroma dimensions round up for odd frame sizes
    chroma_w = (w + 1) // 2
    chroma_h = (h + 1) // 2
    u_size = chroma_w * chroma_h
    frame_size = y_size + 2 * u_size
    frame_off = args.frame * frame_size

    # Validate buffer sizes
    min_needed = frame_off + frame_size
    for label, buf in [("wedeo no-deblock", wedeo_nd), ("wedeo deblock", wedeo_db),
                       ("ffmpeg deblock", ffmpeg_db), ("ffmpeg no-deblock", ffmpeg_nd)]:
        if len(buf) < min_needed:
            print(f"Error: {label} output too short ({len(buf)} bytes, need {min_needed} for frame {args.frame})")
            sys.exit(1)

    planes = []
    if args.plane in ("Y", "all"):
        planes.append(("Y", 0, w, h))
    if args.plane in ("U", "all"):
        planes.append(("U", y_size, chroma_w, chroma_h))
    if args.plane in ("V", "all"):
        planes.append(("V", y_size + u_size, chroma_w, chroma_h))

    # Verify pre-deblock matches
    for name, off, pw, ph in planes:
        nd_w = np.frombuffer(wedeo_nd[frame_off + off:frame_off + off + pw * ph], dtype=np.uint8).reshape(ph, pw)
        nd_ff = np.frombuffer(ffmpeg_nd[frame_off + off:frame_off + off + pw * ph], dtype=np.uint8).reshape(ph, pw)
        nd_diff = np.abs(nd_w.astype(int) - nd_ff.astype(int))
        if nd_diff.max() > 0:
            print(f"  WARNING: {name} pre-deblock NOT bitexact (max_diff={nd_diff.max()})")
            print(f"  Deblock diff analysis may be unreliable")

    print(f"\nFrame {args.frame} deblocking diffs:")
    for name, off, pw, ph in planes:
        pre = np.frombuffer(wedeo_nd[frame_off + off:frame_off + off + pw * ph], dtype=np.uint8).reshape(ph, pw)
        post_w = np.frombuffer(wedeo_db[frame_off + off:frame_off + off + pw * ph], dtype=np.uint8).reshape(ph, pw)
        post_ff = np.frombuffer(ffmpeg_db[frame_off + off:frame_off + off + pw * ph], dtype=np.uint8).reshape(ph, pw)

        diff = np.abs(post_w.astype(int) - post_ff.astype(int))
        if diff.max() == 0:
            print(f"  {name}: MATCH")
            continue

        n_diffs = np.count_nonzero(diff)
        print(f"  {name}: max_diff={diff.max()}, {n_diffs} differing pixels")

        # Find horizontal edges by looking for sign-flip patterns
        mb_size = 16 if name == "Y" else 8
        edge_count = 4 if name == "Y" else 2
        edges_found = 0

        for mb_y in range(ph // mb_size):
            for edge in range(edge_count):
                y_edge = mb_y * mb_size + edge * 4
                if y_edge < 2 or y_edge >= ph - 1:
                    continue

                # Check if this edge row has diffs
                row_above = diff[y_edge - 1, :]
                row_at = diff[y_edge, :]
                if row_above.max() == 0 and row_at.max() == 0:
                    continue

                # Find x ranges with diffs
                diff_mask = (row_above > 0) | (row_at > 0)
                if not diff_mask.any():
                    continue

                xs = np.where(diff_mask)[0]
                x_start, x_end = int(xs[0]), int(xs[-1]) + 1

                print(f"\n  Horizontal edge at y={y_edge} (MB y={mb_y}, edge {edge}), "
                      f"x=[{x_start}..{x_end}]:")
                analyze_edge(name, pre, post_w, post_ff, y_edge, x_start, min(x_end, x_start + 8))

                edges_found += 1
                if edges_found >= args.max_edges:
                    print(f"\n  ... (truncated at {args.max_edges} edges)")
                    break
            if edges_found >= args.max_edges:
                break


if __name__ == "__main__":
    main()
