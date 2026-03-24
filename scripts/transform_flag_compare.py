#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Compare per-MB transform_size_8x8_flag between FFmpeg (via lldb) and wedeo (via tracing).

Verifies that wedeo's CAVLC/CABAC parsing of transform_size_8x8_flag matches
FFmpeg's mb_type & MB_TYPE_8x8DCT for every MB in a frame.

Usage:
    python3 scripts/transform_flag_compare.py <h264_file> [--frame 0] [--max-mbs 50]

Requires:
    - Debug FFmpeg built with --disable-asm at ./FFmpeg/
    - wedeo-framecrc built with tracing
    - lldb in PATH
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MB_TYPE_8x8DCT = 0x01000000


def get_dimensions(h264_file: str) -> tuple[int, int]:
    """Get frame dimensions from FFmpeg."""
    r = subprocess.run(
        ["ffmpeg", "-i", h264_file, "-f", "null", "-"],
        capture_output=True, text=True, timeout=10,
    )
    for line in r.stderr.splitlines():
        if "Video:" not in line:
            continue
        m = re.search(r"(\d{2,5})x(\d{2,5})", line)
        if m:
            return int(m.group(1)), int(m.group(2))
    sys.exit(f"Could not determine dimensions from {h264_file}")


def extract_ffmpeg_flags(h264_file: str, num_mbs: int) -> list[tuple[int, int, bool]]:
    """Extract (mb_x, mb_y, is_8x8dct) for each MB from FFmpeg via lldb.

    Breaks at hl_decode_mb_predict_luma (h264_mb.c:629) to read mb_type.
    """
    ffmpeg_g = Path("FFmpeg/ffmpeg_g")
    if not ffmpeg_g.exists():
        sys.exit("FFmpeg debug binary not found at FFmpeg/ffmpeg_g")

    quoted = str(h264_file).replace("\\", "\\\\").replace('"', '\\"')
    lines = [
        "settings set auto-confirm true",
        "b h264_mb.c:629",
        f'r -i "{quoted}" -vframes 1 -f null -',
    ]
    for _ in range(num_mbs):
        lines.append("p (int)sl->mb_x")
        lines.append("p (int)sl->mb_y")
        lines.append("p (int)((mb_type & 0x01000000) != 0)")
        lines.append("c")
    lines.append("q")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lldb", delete=False) as f:
        f.write("\n".join(lines))
        lldb_script = f.name

    try:
        result = subprocess.run(
            ["lldb", "-b", "-s", lldb_script, "--", str(ffmpeg_g)],
            capture_output=True, text=True, timeout=120,
        )
    finally:
        Path(lldb_script).unlink(missing_ok=True)

    int_values = []
    for line in result.stdout.splitlines():
        m = re.search(r"\(int\)(?:\s+\$\d+\s*=)?\s+(-?\d+)", line)
        if m:
            int_values.append(int(m.group(1)))

    mbs = []
    for i in range(0, len(int_values) - 2, 3):
        mb_x = int_values[i]
        mb_y = int_values[i + 1]
        is_8x8 = int_values[i + 2] != 0
        mbs.append((mb_x, mb_y, is_8x8))
    return mbs


def extract_wedeo_flags(h264_file: str) -> list[tuple[int, int, bool]]:
    """Extract (mb_x, mb_y, transform_size_8x8_flag) from wedeo's MB trace.

    Reads the 'decoded MB' trace lines from the first frame.
    """
    # Debug build required for tracing output
    wedeo_bin = None
    for p in ["target/debug/wedeo-framecrc"]:
        if Path(p).exists():
            wedeo_bin = p
            break
    if not wedeo_bin:
        subprocess.run(
            ["cargo", "build", "--bin", "wedeo-framecrc", "-p", "wedeo-fate", "--features", "tracing"],
            capture_output=True, check=True, timeout=120,
        )
        wedeo_bin = "target/debug/wedeo-framecrc"

    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::mb=trace"}
    result = subprocess.run(
        [wedeo_bin, h264_file],
        capture_output=True, text=True, env=env, timeout=120,
    )

    stderr = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)

    mbs = []
    for line in stderr.splitlines():
        if "decoded MB" not in line:
            continue
        mx_m = re.search(r"mb_x=(\d+)", line)
        my_m = re.search(r"mb_y=(\d+)", line)
        if not mx_m or not my_m:
            continue
        mb_x = int(mx_m.group(1))
        mb_y = int(my_m.group(1))

        # Detect second frame start → stop (we only want frame 0)
        if mbs and mb_x == 0 and mb_y == 0:
            break

        t8x8_m = re.search(r"t8x8=(true|false)", line)
        if not t8x8_m:
            sys.exit(
                "Trace missing t8x8 field. Rebuild wedeo with the t8x8 trace:\n"
                "  cargo build --bin wedeo-framecrc -p wedeo-fate --features tracing"
            )
        is_8x8 = t8x8_m.group(1) == "true"
        mbs.append((mb_x, mb_y, is_8x8))

    return mbs


def compare(
    ffmpeg_mbs: list[tuple[int, int, bool]],
    wedeo_mbs: list[tuple[int, int, bool]],
) -> None:
    """Compare transform flags and report mismatches."""
    n = min(len(ffmpeg_mbs), len(wedeo_mbs))
    mismatches = 0
    print(f"Comparing {n} MBs (FFmpeg: {len(ffmpeg_mbs)}, wedeo: {len(wedeo_mbs)})")
    print()
    print(f"{'MB':>10} {'FFmpeg':>8} {'Wedeo':>8} {'Match':>6}")
    print("-" * 36)

    for i in range(n):
        fmx, fmy, f8x8 = ffmpeg_mbs[i]
        wmx, wmy, w8x8 = wedeo_mbs[i]

        if fmx != wmx or fmy != wmy:
            print(f"  MB index mismatch at position {i}: FFmpeg=({fmx},{fmy}) wedeo=({wmx},{wmy})")
            mismatches += 1
            continue

        match = f8x8 == w8x8
        if not match:
            mismatches += 1
        marker = "OK" if match else "MISMATCH"
        # Only print mismatches and first/last few
        if not match or i < 3 or i >= n - 3:
            print(f"({fmx:2d},{fmy:2d}) {str(f8x8):>8} {str(w8x8):>8} {marker:>6}")

    print()
    if mismatches == 0:
        print(f"All {n} MBs match.")
    else:
        print(f"{mismatches}/{n} mismatches found!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare transform_size_8x8_flag between FFmpeg and wedeo",
    )
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument(
        "--max-mbs", type=int, default=50,
        help="Maximum MBs to compare (default: 50, capped by frame size)",
    )
    args = parser.parse_args()

    w, h = get_dimensions(args.input)
    mb_count = (w // 16) * (h // 16)
    num_mbs = min(args.max_mbs, mb_count)
    print(f"Frame: {w}x{h}, {mb_count} MBs, comparing first {num_mbs}")

    print("\nExtracting FFmpeg transform flags via lldb...")
    ffmpeg_mbs = extract_ffmpeg_flags(args.input, num_mbs)

    print("Extracting wedeo transform flags via trace...")
    wedeo_mbs = extract_wedeo_flags(args.input)

    compare(ffmpeg_mbs, wedeo_mbs)


if __name__ == "__main__":
    main()
