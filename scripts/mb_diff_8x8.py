#!/usr/bin/env python3
"""Find the first differing 8x8 MB between wedeo and FFmpeg and dump pixel details.

Decodes a single frame from both decoders to raw YUV, finds the first macroblock
where pixels differ, and reports the MB coordinates, pixel values, and diff magnitude.
Optionally dumps the 16x16 pixel grid for both decoders side-by-side.

Designed for debugging High Profile 8x8 transform pixel differences.

Usage:
    # Find first differing MB on frame 0
    python3 scripts/mb_diff_8x8.py fate-suite/h264-conformance/FRext/HPCVNL_BRCM_A.264

    # Specify a frame number
    python3 scripts/mb_diff_8x8.py file.264 --frame 5

    # Show full 16x16 pixel grid for the first N differing MBs
    python3 scripts/mb_diff_8x8.py file.264 --grid --max-mbs 3

Requires:
    - wedeo-cli binary (cargo build --release --bin wedeo-cli)
    - ffmpeg binary in PATH
"""
# /// script
# requires-python = ">=3.10"
# ///

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary


def find_wedeo_cli() -> Path:
    """Find the wedeo-cli binary (release preferred, then debug)."""
    for profile in ["release", "debug"]:
        p = Path(f"target/{profile}/wedeo-cli")
        if p.exists():
            return p
    print("Error: wedeo-cli binary not found. Run: cargo build --release --bin wedeo-cli",
          file=sys.stderr)
    sys.exit(1)


def decode_frame_wedeo(cli_bin: Path, input_path: str, frame_idx: int,
                       width: int, height: int) -> bytes | None:
    """Decode raw YUV from wedeo-cli, return the requested frame's Y/U/V bytes.

    Pipes output and reads only enough bytes to reach the target frame,
    avoiding loading the entire decode into memory.
    """
    frame_size = width * height * 3 // 2
    need_bytes = (frame_idx + 1) * frame_size
    try:
        proc = subprocess.Popen(
            [str(cli_bin), "decode", input_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        data = proc.stdout.read(need_bytes)
        proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=5)
        if len(data) < need_bytes:
            return None
        return data[frame_idx * frame_size:(frame_idx + 1) * frame_size]
    except (subprocess.TimeoutExpired, OSError):
        return None


def decode_frame_ffmpeg(input_path: str, frame_idx: int,
                        width: int, height: int) -> bytes | None:
    """Decode raw YUV from FFmpeg, return the requested frame's Y/U/V bytes."""
    frame_size = width * height * 3 // 2
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-flags", "+bitexact",
             "-i", input_path, "-vframes", str(frame_idx + 1),
             "-pix_fmt", "yuv420p", "-f", "rawvideo", "-y", tmp_path],
            timeout=30,
        )
        with open(tmp_path, "rb") as f:
            data = f.read()
        if len(data) < (frame_idx + 1) * frame_size:
            return None
        return data[frame_idx * frame_size:(frame_idx + 1) * frame_size]
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(tmp_path)


def get_dimensions(input_path: str) -> tuple[int, int] | None:
    """Get video dimensions via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", input_path],
            capture_output=True, text=True, timeout=5,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


def extract_mb_y(y_plane: bytes, width: int, mb_x: int, mb_y: int) -> list[list[int]]:
    """Extract 16x16 Y pixels for a macroblock."""
    rows = []
    for row in range(16):
        py = mb_y * 16 + row
        px = mb_x * 16
        start = py * width + px
        rows.append(list(y_plane[start:start + 16]))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Find first differing MB between wedeo and FFmpeg")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to compare (default: 0)")
    parser.add_argument("--grid", action="store_true", help="Show full 16x16 pixel grid")
    parser.add_argument("--max-mbs", type=int, default=1, help="Max differing MBs to show (default: 1)")
    args = parser.parse_args()

    dims = get_dimensions(args.input)
    if dims is None:
        print("Error: could not determine video dimensions", file=sys.stderr)
        sys.exit(1)
    width, height = dims
    mb_w = width // 16
    mb_h = height // 16
    y_size = width * height

    print(f"Dimensions: {width}x{height} ({mb_w}x{mb_h} MBs)")
    print(f"Comparing frame {args.frame}...")

    cli_bin = find_wedeo_cli()
    wedeo_frame = decode_frame_wedeo(cli_bin, args.input, args.frame, width, height)
    ffmpeg_frame = decode_frame_ffmpeg(args.input, args.frame, width, height)

    if wedeo_frame is None:
        print("Error: wedeo decode failed or produced insufficient data", file=sys.stderr)
        sys.exit(1)
    if ffmpeg_frame is None:
        print("Error: FFmpeg decode failed or produced insufficient data", file=sys.stderr)
        sys.exit(1)

    wedeo_y = wedeo_frame[:y_size]
    ffmpeg_y = ffmpeg_frame[:y_size]

    # Find differing MBs
    shown = 0
    total_diff_mbs = 0
    for mb_y_idx in range(mb_h):
        for mb_x_idx in range(mb_w):
            # Check if this MB has any Y pixel differences
            has_diff = False
            max_diff = 0
            diff_count = 0
            for row in range(16):
                for col in range(16):
                    py = mb_y_idx * 16 + row
                    px = mb_x_idx * 16 + col
                    idx = py * width + px
                    d = abs(int(wedeo_y[idx]) - int(ffmpeg_y[idx]))
                    if d > 0:
                        has_diff = True
                        diff_count += 1
                        max_diff = max(max_diff, d)

            if not has_diff:
                continue

            total_diff_mbs += 1
            if shown >= args.max_mbs:
                continue
            shown += 1

            # Check if this MB is all-128 in wedeo (uninitialized)
            all_128 = all(wedeo_y[mb_y_idx * 16 * width + row * width + mb_x_idx * 16 + col] == 128
                         for row in range(16) for col in range(16))

            print(f"\nMB({mb_x_idx},{mb_y_idx}): max_diff={max_diff}, diff_pixels={diff_count}/256"
                  f"{' [ALL-128 in wedeo]' if all_128 else ''}")

            if args.grid:
                w_rows = extract_mb_y(wedeo_y, width, mb_x_idx, mb_y_idx)
                f_rows = extract_mb_y(ffmpeg_y, width, mb_x_idx, mb_y_idx)
                print("  Wedeo Y (16x16):")
                for r, row in enumerate(w_rows):
                    print(f"    row{r:2d}: {' '.join(f'{v:3d}' for v in row)}")
                print("  FFmpeg Y (16x16):")
                for r, row in enumerate(f_rows):
                    print(f"    row{r:2d}: {' '.join(f'{v:3d}' for v in row)}")
                print("  Diff (abs):")
                for r in range(16):
                    diffs = [abs(w_rows[r][c] - f_rows[r][c]) for c in range(16)]
                    print(f"    row{r:2d}: {' '.join(f'{v:3d}' for v in diffs)}")

    if total_diff_mbs == 0:
        print("\nBITEXACT: All MBs match!")
    else:
        match_mbs = mb_w * mb_h - total_diff_mbs
        print(f"\nSummary: {total_diff_mbs}/{mb_w * mb_h} MBs differ ({match_mbs} match)")


if __name__ == "__main__":
    main()
