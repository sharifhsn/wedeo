#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Compare wedeo vs FFmpeg reconstruction data at a specific MBAFF MB.

Runs wedeo with RUST_LOG tracing to extract INTRA_RECON_START, INTRA_RAW_NEIGHBORS,
and INTRA_FINAL_ROWS for a target MB, then runs ffmpeg_recon_extract.py to get
FFmpeg's values, and compares side-by-side.

Reports the first divergence point in this order:
  1. offset & stride
  2. mb_field
  3. top row pixels
  4. left column pixels
  5. top-left pixel
  6. final pixel rows (after IDCT)

Usage:
    python3 scripts/mbaff_recon_compare.py CAMA1 --mb 19,4 --frame 0
    python3 scripts/mbaff_recon_compare.py fate-suite/h264-conformance/CAMA1_Sony_C.jsv --mb 19,4
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary, resolve_conformance_file


def parse_trace_array(s: str) -> list[int]:
    """Parse a Rust debug array like '[1, 2, 3]' into a list of ints."""
    s = s.strip("[] ")
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",")]


def extract_wedeo_traces(
    input_file: Path,
    mb_x: int,
    mb_y: int,
    frame: int,
) -> dict | None:
    """Run wedeo with tracing and extract reconstruction data for target MB.

    Returns dict with keys: mb_field, has_top, has_left, left_block_option,
    offset, stride, top_row, left_col, top_left, final_rows.
    """
    wedeo_bin = find_wedeo_binary()

    env = {
        **dict(__import__("os").environ),
        "RUST_LOG": "wedeo_codec_h264::mb=trace",
    }

    result = subprocess.run(
        [str(wedeo_bin), str(input_file)],
        capture_output=True, text=True, timeout=120, env=env,
    )

    # Strip ANSI codes from stderr (trace output goes to stderr)
    raw = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)

    # Filter to target MB
    mb_pattern = f"mb_x={mb_x} mb_y={mb_y}"
    mb_lines = [l for l in raw.splitlines() if mb_pattern in l]

    if not mb_lines:
        print(f"No trace output for MB({mb_x},{mb_y})", file=sys.stderr)
        return None

    # Group by frame: each INTRA_RECON_START marks a new frame's MB decode
    frames = []
    current = {}
    for line in mb_lines:
        if "INTRA_RECON_START" in line:
            if current:
                frames.append(current)
            current = {"raw_start": line}
            # Parse fields: mb_field=true has_top=true has_left=true left_block_option=3
            m = re.search(r"mb_field=(\w+)", line)
            if m:
                current["mb_field"] = m.group(1) == "true"
            m = re.search(r"has_top=(\w+)", line)
            if m:
                current["has_top"] = m.group(1) == "true"
            m = re.search(r"has_left=(\w+)", line)
            if m:
                current["has_left"] = m.group(1) == "true"
            m = re.search(r"left_block_option=(\d+)", line)
            if m:
                current["left_block_option"] = int(m.group(1))

        elif "INTRA_RAW_NEIGHBORS" in line and current:
            m = re.search(r"offset=(\d+)", line)
            if m:
                current["offset"] = int(m.group(1))
            m = re.search(r"stride=(\d+)", line)
            if m:
                current["stride"] = int(m.group(1))
            m = re.search(r"top_row=\[([^\]]*)\]", line)
            if m:
                current["top_row"] = parse_trace_array(m.group(1))
            m = re.search(r"left_col=\[([^\]]*)\]", line)
            if m:
                current["left_col"] = parse_trace_array(m.group(1))
            m = re.search(r"top_left=(\d+)", line)
            if m:
                current["top_left"] = int(m.group(1))

        elif "INTRA_FINAL_ROWS" in line and current:
            rows = []
            for i in range(4):
                m = re.search(rf"y_row{i}=\[([^\]]*)\]", line)
                if m:
                    rows.append(parse_trace_array(m.group(1)))
            if rows:
                current["final_rows"] = rows

    if current:
        frames.append(current)

    if frame >= len(frames):
        print(
            f"Only {len(frames)} frames found, requested frame {frame}",
            file=sys.stderr,
        )
        return None

    return frames[frame]


def compare_arrays(name: str, wedeo: list[int], ffmpeg: list[int]) -> bool:
    """Compare two arrays element-by-element. Returns True if they match."""
    if wedeo == ffmpeg:
        print(f"  {name}: MATCH ({len(wedeo)} elements)")
        return True

    diffs = []
    max_len = max(len(wedeo), len(ffmpeg))
    for i in range(max_len):
        w = wedeo[i] if i < len(wedeo) else None
        f = ffmpeg[i] if i < len(ffmpeg) else None
        if w != f:
            diffs.append((i, w, f))

    print(f"  {name}: DIFFER at {len(diffs)}/{max_len} positions")
    for idx, w, f in diffs[:8]:
        print(f"    [{idx}]: wedeo={w} ffmpeg={f}" + (f" (diff={abs(w-f) if w is not None and f is not None else '?'})" if w is not None and f is not None else ""))
    if len(diffs) > 8:
        print(f"    ... and {len(diffs) - 8} more")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare wedeo vs FFmpeg reconstruction at a specific MB"
    )
    parser.add_argument("file", help="Conformance file (name or path)")
    parser.add_argument("--mb", required=True, help="MB position as x,y")
    parser.add_argument("--frame", type=int, default=0, help="Frame number")
    parser.add_argument(
        "--wedeo-only", action="store_true",
        help="Only extract wedeo data (skip FFmpeg)"
    )
    args = parser.parse_args()

    input_file = resolve_conformance_file(args.file)
    parts = args.mb.split(",")
    mb_x, mb_y = int(parts[0]), int(parts[1])

    print(f"=== MBAFF Reconstruction Comparison ===")
    print(f"File: {input_file.name}")
    print(f"Target: MB({mb_x},{mb_y}) frame {args.frame}")
    print()

    # Step 1: Extract wedeo data
    print("--- Extracting wedeo data ---")
    w = extract_wedeo_traces(input_file, mb_x, mb_y, args.frame)
    if w is None:
        print("FAILED: No wedeo data extracted")
        sys.exit(1)

    print(f"  mb_field={w.get('mb_field')}")
    print(f"  has_top={w.get('has_top')} has_left={w.get('has_left')}")
    print(f"  left_block_option={w.get('left_block_option')}")
    print(f"  offset={w.get('offset')} stride={w.get('stride')}")
    print(f"  top_row={w.get('top_row')}")
    print(f"  left_col={w.get('left_col')}")
    print(f"  top_left={w.get('top_left')}")
    if "final_rows" in w:
        for i, row in enumerate(w["final_rows"]):
            print(f"  final_row{i}={row}")
    print()

    if args.wedeo_only:
        return

    # Step 2: Extract FFmpeg data
    print("--- Extracting FFmpeg data ---")
    from ffmpeg_recon_extract import extract_via_subprocess
    from ffmpeg_debug import find_ffmpeg_binary

    ffmpeg_bin = find_ffmpeg_binary()
    f = extract_via_subprocess(ffmpeg_bin, input_file, mb_x, mb_y, args.frame)
    if f is None:
        print("FAILED: No FFmpeg data extracted")
        print("Try running manually: python3 scripts/ffmpeg_recon_extract.py "
              f"{input_file.name} --mb {mb_x},{mb_y} --frame {args.frame}")
        sys.exit(1)

    print(f"  mb_field={f.get('mb_field')}")
    print(f"  mb_type=0x{f.get('mb_type', 0):x}")
    print(f"  offset={f.get('offset')} linesize={f.get('linesize')}")
    print(f"  top_row={f.get('top_row')}")
    print(f"  left_col={f.get('left_col')}")
    print(f"  top_left={f.get('top_left')}")
    if "final_rows" in f:
        for i, row in enumerate(f["final_rows"]):
            print(f"  final_row{i}={row}")
    print()

    # Step 3: Compare (stop at first divergence)
    print("=== Comparison (stop at first divergence) ===")
    all_match = True

    # 1. offset & stride
    w_offset = w.get("offset")
    f_offset = f.get("offset")
    w_stride = w.get("stride")
    f_stride = f.get("linesize")

    if w_offset != f_offset:
        print(f"  DIVERGENCE: offset wedeo={w_offset} ffmpeg={f_offset}")
        print("  → Bug in luma_mb_offset field-mode formula")
        print("  → Compare against FFmpeg h264_mb_template.c:56-73")
        sys.exit(1)
    else:
        print(f"  offset: MATCH ({w_offset})")

    if w_stride != f_stride:
        print(f"  DIVERGENCE: stride wedeo={w_stride} ffmpeg={f_stride}")
        print("  → Bug in stride computation for field mode")
        sys.exit(1)
    else:
        print(f"  stride: MATCH ({w_stride})")

    # 2. mb_field
    w_field = w.get("mb_field")
    f_field = bool(f.get("mb_field"))
    if w_field != f_field:
        print(f"  DIVERGENCE: mb_field wedeo={w_field} ffmpeg={f_field}")
        all_match = False
    else:
        print(f"  mb_field: MATCH ({w_field})")

    # 3. top row
    w_top = w.get("top_row", [])
    f_top = f.get("top_row", [])
    if not compare_arrays("top_row", w_top, f_top):
        all_match = False
        print("  → Neighbor pixels differ: either cascade from earlier MB")
        print("    or wrong stride/offset in neighbor gather")
        if w.get("offset") == f.get("offset"):
            print("  → Since offset matches, this is likely a cascade from earlier MB")
            print("    (an earlier MB wrote wrong pixels to the position above)")

    # 4. left column
    w_left = w.get("left_col", [])
    f_left = f.get("left_col", [])
    if not compare_arrays("left_col", w_left, f_left):
        all_match = False

    # 5. top-left
    w_tl = w.get("top_left")
    f_tl = f.get("top_left")
    if w_tl != f_tl:
        print(f"  top_left: DIFFER wedeo={w_tl} ffmpeg={f_tl}")
        all_match = False
    else:
        print(f"  top_left: MATCH ({w_tl})")

    # 6. final rows
    w_final = w.get("final_rows", [])
    f_final = f.get("final_rows", [])
    for i in range(min(len(w_final), len(f_final))):
        if not compare_arrays(f"final_row{i}", w_final[i], f_final[i]):
            all_match = False

    print()
    if all_match:
        print("ALL MATCH at this MB — first diff may be at a different MB")
    else:
        print("DIVERGENCE FOUND — see details above")
        print()
        # Decision tree hints
        if w_top != f_top or w_left != f_left:
            print("DIAGNOSIS: Neighbor pixels differ.")
            print("  If offset/stride match, this is likely a cascade from an earlier MB.")
            print("  Run: python3 scripts/mb_compare.py <file> --start-frame 0 --max-frames 1")
            print("  to find the first differing MB, then re-run this script on that MB.")
        elif w_final != f_final:
            print("DIAGNOSIS: Final pixels differ but neighbors match.")
            print("  → Bug in prediction or IDCT with field stride")
            print("  → Check predict_4x4/predict_16x16 mode selection")
            print("  → Check that idct4x4_add receives correct field-adjusted stride")


if __name__ == "__main__":
    main()
