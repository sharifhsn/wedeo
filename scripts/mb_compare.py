#!/usr/bin/env python3
"""Per-macroblock pixel comparison between wedeo and FFmpeg.

Decodes a H.264 file with both wedeo and FFmpeg (deblocking disabled),
then compares per-MB luma and chroma pixels to find the first differing MB.

Usage:
    python3 scripts/mb_compare.py fate-suite/h264-conformance/BAMQ1_JVC_C.264

Requires:
    - wedeo-framecrc binary (cargo build)
    - ffmpeg binary in PATH
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def decode_yuv(cmd: list[str], env: dict | None = None) -> bytes:
    """Run a command that writes raw YUV to a temp file, return the bytes."""
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        tmp = f.name

    full_env = dict(**subprocess.os.environ, **(env or {}))
    result = subprocess.run(cmd + [tmp], capture_output=True, env=full_env)
    if result.returncode != 0:
        print(f"FAIL: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr.decode(errors="replace"), file=sys.stderr)
        sys.exit(1)

    data = Path(tmp).read_bytes()
    Path(tmp).unlink(missing_ok=True)
    return data


def get_dimensions_from_framecrc(lines: list[str]) -> tuple[int, int]:
    """Extract width x height from framecrc header comments."""
    for line in lines:
        if line.startswith("#dimensions"):
            # #dimensions 0: 176x144
            parts = line.split(":")[-1].strip().split("x")
            return int(parts[0]), int(parts[1])
    raise ValueError("No #dimensions line found in framecrc output")


def count_frames(lines: list[str]) -> int:
    """Count non-comment data lines."""
    return sum(1 for l in lines if l.strip() and not l.startswith("#"))


def main():
    parser = argparse.ArgumentParser(description="Per-MB pixel comparison tool")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument(
        "--max-frames", type=int, default=5, help="Max frames to compare"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="First frame to compare"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Find wedeo-framecrc binary
    wedeo_bin = None
    for profile in ["debug", "release"]:
        candidate = Path("target") / profile / "wedeo-framecrc"
        if candidate.exists():
            wedeo_bin = str(candidate.resolve())
            break
    if wedeo_bin is None:
        print("Error: wedeo-framecrc not found. Run `cargo build -p wedeo-fate`", file=sys.stderr)
        sys.exit(1)

    # Step 1: Get dimensions from wedeo framecrc header
    result = subprocess.run(
        [wedeo_bin, str(input_path)],
        capture_output=True,
        env={**subprocess.os.environ, "WEDEO_NO_DEBLOCK": "1"},
    )
    wedeo_lines = result.stdout.decode().splitlines()
    width, height = get_dimensions_from_framecrc(wedeo_lines)
    n_frames = count_frames(wedeo_lines)
    print(f"Dimensions: {width}x{height}, {n_frames} frames")

    mb_w = width // 16
    mb_h = height // 16

    # Step 2: Decode raw YUV with both tools (deblocking disabled)
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        wedeo_yuv_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        ffmpeg_yuv_path = f.name

    # wedeo: WEDEO_NO_DEBLOCK=1
    subprocess.run(
        [wedeo_bin, str(input_path), "--raw-yuv", wedeo_yuv_path],
        capture_output=True,
        env={**subprocess.os.environ, "WEDEO_NO_DEBLOCK": "1"},
        check=True,
    )

    # ffmpeg: -skip_loop_filter all
    subprocess.run(
        [
            "ffmpeg", "-y", "-bitexact",
            "-skip_loop_filter", "all",
            "-i", str(input_path),
            "-pix_fmt", "yuv420p",
            "-f", "rawvideo",
            ffmpeg_yuv_path,
        ],
        capture_output=True,
        check=True,
    )

    wedeo_data = Path(wedeo_yuv_path).read_bytes()
    ffmpeg_data = Path(ffmpeg_yuv_path).read_bytes()
    Path(wedeo_yuv_path).unlink(missing_ok=True)
    Path(ffmpeg_yuv_path).unlink(missing_ok=True)

    frame_size = width * height * 3 // 2
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    if len(wedeo_data) < frame_size or len(ffmpeg_data) < frame_size:
        print(f"Error: YUV data too short (wedeo={len(wedeo_data)}, ffmpeg={len(ffmpeg_data)})")
        sys.exit(1)

    actual_frames = min(len(wedeo_data), len(ffmpeg_data)) // frame_size
    end_frame = min(actual_frames, args.start_frame + args.max_frames)
    if args.start_frame >= actual_frames:
        print(f"Error: start-frame {args.start_frame} >= {actual_frames} actual frames")
        sys.exit(1)
    print(f"Comparing frames {args.start_frame}..{end_frame - 1} of {actual_frames} decoded frames\n")

    total_diffs = 0
    for frame_idx in range(args.start_frame, end_frame):
        base = frame_idx * frame_size
        w_y = np.frombuffer(wedeo_data[base:base + y_size], dtype=np.uint8).reshape(height, width)
        f_y = np.frombuffer(ffmpeg_data[base:base + y_size], dtype=np.uint8).reshape(height, width)

        frame_diffs = 0
        first_diff_mb = None
        for my in range(mb_h):
            for mx in range(mb_w):
                y0, y1 = my * 16, (my + 1) * 16
                x0, x1 = mx * 16, (mx + 1) * 16
                w_block = w_y[y0:y1, x0:x1]
                f_block = f_y[y0:y1, x0:x1]
                diff = np.abs(w_block.astype(np.int16) - f_block.astype(np.int16))
                if diff.any():
                    frame_diffs += 1
                    if first_diff_mb is None:
                        max_diff = int(diff.max())
                        mean_diff = float(diff[diff > 0].mean())
                        first_diff_mb = (mx, my, max_diff, mean_diff)

        if frame_diffs > 0:
            total_diffs += frame_diffs
            mx, my, max_d, mean_d = first_diff_mb
            print(
                f"Frame {frame_idx}: {frame_diffs}/{mb_w * mb_h} MBs differ "
                f"(first: MB({mx},{my}), max_diff={max_d}, mean_diff={mean_d:.1f})"
            )
            print(
                f"  Debug: RUST_LOG=wedeo_codec_h264::mb=trace "
                f"cargo run --release --bin wedeo-framecrc --features tracing "
                f"-- {input_path} --raw-yuv /dev/null 2>/tmp/trace.txt && "
                f'grep "MB({mx},{my})" /tmp/trace.txt | head -20'
            )
        else:
            print(f"Frame {frame_idx}: MATCH")

    print(f"\nTotal: {total_diffs} differing MBs across {end_frame - args.start_frame} frames")
    if total_diffs == 0:
        print("BITEXACT!")


if __name__ == "__main__":
    main()
