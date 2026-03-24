#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Compare decoded YUV frames from wedeo and FFmpeg, reporting the first
differing pixel and per-MB diff statistics.

Usage:
    uv run scripts/yuv_first_diff.py <h264_file> [--frame N] [--no-deblock]

Requires: ffmpeg in PATH, wedeo-framecrc built (release or debug).
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def find_wedeo_bin() -> str:
    for p in ["target/release/wedeo-framecrc", "target/debug/wedeo-framecrc"]:
        if Path(p).exists():
            return p
    sys.exit("wedeo-framecrc not found; run: cargo build --bin wedeo-framecrc -p wedeo-fate")


def get_dimensions(path: str) -> tuple[int, int, int]:
    """Return (width, height, frame_count) from wedeo-framecrc header."""
    r = subprocess.run(
        [find_wedeo_bin(), path],
        capture_output=True, text=True, timeout=30,
    )
    w = h = n = 0
    for line in r.stdout.splitlines():
        if line.startswith("#dimensions"):
            parts = line.split()[-1].split("x")
            w, h = int(parts[0]), int(parts[1])
        if not line.startswith("#") and "," in line:
            n += 1
    return w, h, n


def decode_yuv(path: str, tool: str, no_deblock: bool) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        yuv_path = f.name
    try:
        if tool == "wedeo":
            import os
            env = {**os.environ}
            if no_deblock:
                env["WEDEO_NO_DEBLOCK"] = "1"
            subprocess.run(
                [find_wedeo_bin(), path, "--raw-yuv", yuv_path],
                capture_output=True, env=env, check=True, timeout=120,
            )
        else:
            cmd = ["ffmpeg", "-y", "-bitexact"]
            if no_deblock:
                cmd += ["-skip_loop_filter", "all"]
            cmd += ["-i", path, "-pix_fmt", "yuv420p", "-f", "rawvideo", yuv_path]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return Path(yuv_path).read_bytes()
    finally:
        Path(yuv_path).unlink(missing_ok=True)


def compare_frame(
    ffmpeg: bytes, wedeo: bytes, w: int, h: int, frame_idx: int, frame_size: int,
) -> None:
    y_size = w * h
    f_off = frame_idx * frame_size
    w_off = frame_idx * frame_size

    if f_off + frame_size > len(ffmpeg):
        print(f"Frame {frame_idx}: FFmpeg has fewer frames")
        return
    if w_off + frame_size > len(wedeo):
        print(f"Frame {frame_idx}: wedeo has fewer frames")
        return

    fy = ffmpeg[f_off:f_off + y_size]
    wy = wedeo[w_off:w_off + y_size]

    # Y plane comparison
    mb_diffs: dict[tuple[int, int], dict] = {}
    total_y_diffs = 0
    first_diff = None

    for i in range(y_size):
        if fy[i] != wy[i]:
            total_y_diffs += 1
            px, py = i % w, i // w
            mx, my = px // 16, py // 16
            key = (mx, my)
            if key not in mb_diffs:
                mb_diffs[key] = {"count": 0, "max_delta": 0, "first": (px, py)}
            mb_diffs[key]["count"] += 1
            mb_diffs[key]["max_delta"] = max(mb_diffs[key]["max_delta"], abs(fy[i] - wy[i]))
            if first_diff is None:
                first_diff = (px, py, mx, my, fy[i], wy[i])

    # Chroma comparison
    uv_size = (w // 2) * (h // 2)
    u_diffs = sum(1 for i in range(y_size, y_size + uv_size)
                  if ffmpeg[f_off + i] != wedeo[w_off + i])
    v_diffs = sum(1 for i in range(y_size + uv_size, y_size + 2 * uv_size)
                  if ffmpeg[f_off + i] != wedeo[w_off + i])

    if total_y_diffs == 0 and u_diffs == 0 and v_diffs == 0:
        print(f"Frame {frame_idx}: BITEXACT")
        return

    print(f"Frame {frame_idx}: {total_y_diffs} Y diffs, {u_diffs} U diffs, {v_diffs} V diffs")
    if first_diff:
        px, py, mx, my, fv, wv = first_diff
        print(f"  First Y diff: pixel ({px},{py}) MB({mx},{my}) FFmpeg={fv} wedeo={wv} delta={wv-fv:+d}")
    print(f"  Differing MBs: {len(mb_diffs)}/{(w // 16) * (h // 16)}")

    # Show first 10 differing MBs sorted by raster order
    sorted_mbs = sorted(mb_diffs.items(), key=lambda t: (t[0][1], t[0][0]))
    for (mx, my), info in sorted_mbs[:10]:
        print(f"    MB({mx},{my}): {info['count']}/256 diffs, max_delta={info['max_delta']}")
    if len(sorted_mbs) > 10:
        print(f"    ... and {len(sorted_mbs) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare decoded YUV between wedeo and FFmpeg")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to compare (default: 0)")
    parser.add_argument("--no-deblock", action="store_true", help="Disable deblocking in both decoders")
    args = parser.parse_args()

    w, h, n_frames = get_dimensions(args.input)
    print(f"Dimensions: {w}x{h}, {n_frames} frames")

    ffmpeg_data = decode_yuv(args.input, "ffmpeg", args.no_deblock)
    wedeo_data = decode_yuv(args.input, "wedeo", args.no_deblock)

    frame_size = w * h * 3 // 2
    compare_frame(ffmpeg_data, wedeo_data, w, h, args.frame, frame_size)


if __name__ == "__main__":
    main()
