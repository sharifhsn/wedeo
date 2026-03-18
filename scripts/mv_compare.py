#!/usr/bin/env python3
"""Compare motion vectors and reference indices for a specific macroblock.

Extracts MV/ref_idx/mb_type from both wedeo (via tracing) and FFmpeg
(via -debug mv) for a specific frame and MB, then shows the differences.

Usage:
    python3 scripts/mv_compare.py input.264 --frame 1 --mb 2,0
    python3 scripts/mv_compare.py input.264 --frame 1   # all differing MBs
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def find_wedeo_bin() -> str:
    """Find the wedeo-framecrc binary (debug preferred for tracing support)."""
    for profile in ["debug", "release"]:
        candidate = Path("target") / profile / "wedeo-framecrc"
        if candidate.exists():
            return str(candidate.resolve())
    print(
        "Error: wedeo-framecrc not found. Run:\n"
        "  cargo build --bin wedeo-framecrc -p wedeo-fate --features tracing",
        file=sys.stderr,
    )
    sys.exit(1)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def get_dimensions(input_path: str) -> tuple[int, int]:
    """Get video dimensions from wedeo-framecrc header output."""
    wedeo_bin = find_wedeo_bin()
    result = subprocess.run(
        [wedeo_bin, input_path],
        capture_output=True,
        env={**os.environ, "WEDEO_NO_DEBLOCK": "1"},
    )
    for line in result.stdout.decode().splitlines():
        if line.startswith("#dimensions"):
            parts = line.split(":")[-1].strip().split("x")
            return int(parts[0]), int(parts[1])
    print("Error: could not determine dimensions", file=sys.stderr)
    sys.exit(1)


def decode_yuv(input_path: str, tool: str, no_deblock: bool = True) -> bytes:
    """Decode to raw YUV420p using the specified tool."""
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        yuv_path = f.name
    try:
        if tool == "wedeo":
            wedeo_bin = find_wedeo_bin()
            env = {**os.environ}
            if no_deblock:
                env["WEDEO_NO_DEBLOCK"] = "1"
            subprocess.run(
                [wedeo_bin, input_path, "--raw-yuv", yuv_path],
                capture_output=True,
                env=env,
                check=True,
            )
        else:
            cmd = ["ffmpeg", "-y", "-bitexact"]
            if no_deblock:
                cmd += ["-skip_loop_filter", "all"]
            cmd += ["-i", input_path, "-pix_fmt", "yuv420p", "-f", "rawvideo", yuv_path]
            subprocess.run(cmd, capture_output=True, check=True)
        return Path(yuv_path).read_bytes()
    finally:
        Path(yuv_path).unlink(missing_ok=True)


def extract_wedeo_mvs(input_path: str, frame_idx: int) -> dict:
    """Extract MV info from wedeo via tracing for a specific frame.

    Returns dict keyed by (mb_x, mb_y) with mb_type, mv, ref_idx info.
    """
    wedeo_bin = find_wedeo_bin()
    env = {
        **os.environ,
        "RUST_LOG": "wedeo_codec_h264=trace",
        "WEDEO_NO_DEBLOCK": "1",
    }

    result = subprocess.run(
        [wedeo_bin, input_path],
        capture_output=True,
        env=env,
    )

    stderr = strip_ansi(result.stderr.decode("utf-8", errors="replace"))

    # Track which internal frame_num corresponds to the requested display frame.
    # The tracing output uses internal decode-order frame numbers.
    # We need to map display frame_idx to decode frame based on POC ordering.

    mbs = {}
    current_decode_frame = -1
    in_target_frame = False
    # Count frames completed to track decode order
    frames_completed = 0

    # Simple approach: collect ALL MB data per decode-frame, then figure out
    # which decode-frame corresponds to display frame_idx.
    all_frames = {}  # decode_frame -> {(mb_x, mb_y): info}
    current_frame_mbs = {}

    for line in stderr.splitlines():
        if "frame complete" in line:
            if current_frame_mbs:
                m = re.search(r"poc=(-?\d+)", line)
                poc = int(m.group(1)) if m else frames_completed * 2
                all_frames[poc] = dict(current_frame_mbs)
            current_frame_mbs = {}
            frames_completed += 1
            continue

        if "decoded MB" in line or "MB type parsed" in line:
            m_x = re.search(r"mb_x=(\d+)", line)
            m_y = re.search(r"mb_y=(\d+)", line)
            if not m_x or not m_y:
                continue
            mx, my = int(m_x.group(1)), int(m_y.group(1))
            key = (mx, my)
            if key not in current_frame_mbs:
                current_frame_mbs[key] = {}
            info = current_frame_mbs[key]

            # Extract available fields
            for field in ["mb_type", "raw_mb_type", "qp", "cbp",
                          "is_intra", "is_intra4x4", "is_intra16x16"]:
                m = re.search(rf"{field}=(\S+)", line)
                if m:
                    info[field] = m.group(1)

        if "MV" in line and ("16x16" in line or "16x8" in line or "8x16" in line or "8x8" in line):
            m_x = re.search(r"mb_x=(\d+)", line)
            m_y = re.search(r"mb_y=(\d+)", line)
            m_mv = re.search(r"mv=\[(-?\d+),\s*(-?\d+)\]", line)
            m_mvp = re.search(r"mvp=\[(-?\d+),\s*(-?\d+)\]", line)
            m_mvd = re.search(r"mvd=\[(-?\d+),\s*(-?\d+)\]", line)
            m_ref = re.search(r"ref_idx=(\d+)", line)
            if m_x and m_y:
                mx, my = int(m_x.group(1)), int(m_y.group(1))
                key = (mx, my)
                if key not in current_frame_mbs:
                    current_frame_mbs[key] = {}
                info = current_frame_mbs[key]
                if m_mv:
                    info["mv"] = (int(m_mv.group(1)), int(m_mv.group(2)))
                if m_mvp:
                    info["mvp"] = (int(m_mvp.group(1)), int(m_mvp.group(2)))
                if m_mvd:
                    info["mvd"] = (int(m_mvd.group(1)), int(m_mvd.group(2)))
                if m_ref:
                    info["ref_idx"] = int(m_ref.group(1))

    # Sort by POC to get display order
    sorted_pocs = sorted(all_frames.keys())
    if frame_idx < len(sorted_pocs):
        target_poc = sorted_pocs[frame_idx]
        mbs = all_frames[target_poc]

    return mbs


def compare_mb_pixels(
    wedeo_data: bytes,
    ffmpeg_data: bytes,
    width: int,
    height: int,
    frame_idx: int,
    mb_x: int,
    mb_y: int,
) -> dict:
    """Compare pixels for a specific MB between wedeo and FFmpeg."""
    frame_size = width * height * 3 // 2
    y_size = width * height

    w_offset = frame_idx * frame_size
    f_offset = frame_idx * frame_size

    if w_offset + frame_size > len(wedeo_data) or f_offset + frame_size > len(ffmpeg_data):
        return {"error": "frame out of range"}

    wy = np.frombuffer(wedeo_data[w_offset:w_offset + y_size], dtype=np.uint8).reshape(height, width)
    fy = np.frombuffer(ffmpeg_data[f_offset:f_offset + y_size], dtype=np.uint8).reshape(height, width)

    y0, y1 = mb_y * 16, (mb_y + 1) * 16
    x0, x1 = mb_x * 16, (mb_x + 1) * 16

    w_block = wy[y0:y1, x0:x1]
    f_block = fy[y0:y1, x0:x1]
    diff = np.abs(w_block.astype(np.int16) - f_block.astype(np.int16))

    return {
        "max_diff": int(diff.max()),
        "mean_diff": float(diff.mean()),
        "num_diff_pixels": int(np.count_nonzero(diff)),
        "wedeo_row0": list(map(int, w_block[0])),
        "ffmpeg_row0": list(map(int, f_block[0])),
        "diff_row0": list(map(int, w_block[0].astype(np.int16) - f_block[0].astype(np.int16))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare motion vectors and pixels for a specific macroblock.",
    )
    parser.add_argument("input", help="Input H.264 file")
    parser.add_argument("--frame", type=int, required=True,
                        help="Display-order frame index (0-based)")
    parser.add_argument("--mb", type=str, default=None,
                        help="Macroblock position as X,Y (e.g., '2,0'). Omit to show all differing MBs.")
    parser.add_argument("--no-pixels", action="store_true",
                        help="Skip pixel comparison (faster, MV-only)")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    width, height = get_dimensions(input_path)
    mb_w, mb_h = width // 16, height // 16
    print(f"Dimensions: {width}x{height} ({mb_w}x{mb_h} MBs)")

    # Extract wedeo MV info
    print(f"\nExtracting wedeo MV info for frame {args.frame}...")
    wedeo_mvs = extract_wedeo_mvs(input_path, args.frame)

    if not wedeo_mvs:
        print(f"Warning: no MV data found for frame {args.frame}", file=sys.stderr)
        print("Make sure wedeo-framecrc was built with tracing:")
        print("  cargo build --bin wedeo-framecrc -p wedeo-fate --features tracing")

    # Pixel comparison
    wedeo_data = b""
    ffmpeg_data = b""
    if not args.no_pixels:
        print("Decoding YUV for pixel comparison...")
        try:
            wedeo_data = decode_yuv(input_path, "wedeo")
            ffmpeg_data = decode_yuv(input_path, "ffmpeg")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: decode failed: {e}", file=sys.stderr)
            wedeo_data = ffmpeg_data = b""

    if args.mb:
        # Single MB mode
        parts = args.mb.split(",")
        mx, my = int(parts[0]), int(parts[1])

        print(f"\n=== MB({mx},{my}) frame {args.frame} ===")
        if (mx, my) in wedeo_mvs:
            info = wedeo_mvs[(mx, my)]
            print(f"  wedeo: {info}")
        else:
            print(f"  wedeo: no MV data")

        if wedeo_data and ffmpeg_data:
            px = compare_mb_pixels(wedeo_data, ffmpeg_data, width, height, args.frame, mx, my)
            if "error" not in px:
                print(f"  pixel max_diff={px['max_diff']}, diff_pixels={px['num_diff_pixels']}/256")
                if px["max_diff"] > 0:
                    print(f"  wedeo  row0: {px['wedeo_row0']}")
                    print(f"  ffmpeg row0: {px['ffmpeg_row0']}")
                    print(f"  diff   row0: {px['diff_row0']}")
    else:
        # All-MB mode: show differing MBs
        if not wedeo_data or not ffmpeg_data:
            print("Cannot compare pixels without YUV data")
            # Just dump all wedeo MV info
            for (mx, my) in sorted(wedeo_mvs.keys()):
                info = wedeo_mvs[(mx, my)]
                print(f"  MB({mx},{my}): {info}")
            return

        print(f"\n=== Differing MBs in frame {args.frame} ===")
        diff_count = 0
        for my in range(mb_h):
            for mx in range(mb_w):
                px = compare_mb_pixels(wedeo_data, ffmpeg_data, width, height, args.frame, mx, my)
                if "error" in px:
                    continue
                if px["max_diff"] > 0:
                    diff_count += 1
                    mv_info = wedeo_mvs.get((mx, my), {})
                    mv_str = ""
                    if "mv" in mv_info:
                        mv_str += f" mv={mv_info['mv']}"
                    if "ref_idx" in mv_info:
                        mv_str += f" ref={mv_info['ref_idx']}"
                    if "raw_mb_type" in mv_info:
                        mv_str += f" type={mv_info['raw_mb_type']}"
                    print(
                        f"  MB({mx},{my}): max_diff={px['max_diff']:3d}"
                        f" diff_px={px['num_diff_pixels']:3d}/256{mv_str}"
                    )

        print(f"\nTotal: {diff_count}/{mb_w * mb_h} MBs differ")


if __name__ == "__main__":
    main()
