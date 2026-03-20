#!/usr/bin/env python3
"""Extract and compare MV neighbor context (A/B/C) between wedeo and FFmpeg.

For a specific MB and partition in a B-frame, extracts the L0 and L1 neighbor
values used for MV prediction from both decoders and shows them side by side.
This directly targets the most common debugging bottleneck: "the prediction
formula is correct but the inputs differ."

Usage:
    python3 scripts/neighbor_debug.py file.264 --frame 2 --mb 4,0 --part 2
    python3 scripts/neighbor_debug.py file.264 --frame 2 --mb 4,0 --list l1

Requires:
    - Debug FFmpeg built with --disable-optimizations --enable-debug=3 --disable-asm
    - wedeo-framecrc binary with tracing feature
    - lldb in PATH

NOTE: L1 neighbor extraction requires trace!() calls in the B_8x8 L1 prediction
path (mb.rs). These are not present in production code — add them temporarily
when debugging. L0 neighbor extraction is not yet supported.
"""

import argparse
import re
import sys
from pathlib import Path

from ffmpeg_debug import (
    SCAN8,
    find_ffmpeg_binary,
    get_frame_order,
    count_slice_type_before,
    run_lldb,
    calibrate_ignore_count,
    run_wedeo_with_tracing,
)


def extract_wedeo_neighbors(
    input_path: Path,
    target_decode_idx: int,
    mb_x: int,
    mb_y: int,
) -> list[dict]:
    """Extract L1 neighbor MV values from wedeo traces.

    Returns a list of dicts with keys: part, j, sub_blk_x, sub_blk_y,
    list, a, b, c (each with mv/ref/avail).

    Requires temporary trace!() calls in the B_8x8 L1 prediction path.
    """
    trace = run_wedeo_with_tracing(
        str(input_path),
        rust_log="wedeo_codec_h264::mb=trace",
        no_deblock=True,
        features=["tracing"],
    )

    # Count frame completions to map decode_idx to trace sections
    frame_start_lines: list[int] = [0]
    lines = trace.splitlines()
    for i, line in enumerate(lines):
        if "frame complete" in line:
            frame_start_lines.append(i + 1)

    if target_decode_idx >= len(frame_start_lines):
        print(f"Error: decode_idx {target_decode_idx} not found "
              f"(only {len(frame_start_lines) - 1} frames)", file=sys.stderr)
        return []

    start = frame_start_lines[target_decode_idx]
    end = (frame_start_lines[target_decode_idx + 1]
           if target_decode_idx + 1 < len(frame_start_lines)
           else len(lines))

    results = []

    # Pattern for L1 neighbor traces (B_8x8 sub-partitions)
    neighbor_re = re.compile(
        r"L1 neighbors\s+mb_x=(\d+)\s+mb_y=(\d+)\s+part=(\d+)\s+j=(\d+)"
        r"\s+sub_blk_x=(\d+)\s+sub_blk_y=(\d+)"
        r"\s+l1_a=\(\[(-?\d+),\s*(-?\d+)\],\s*(-?\d+),\s*(true|false)\)"
        r"\s+l1_b=\(\[(-?\d+),\s*(-?\d+)\],\s*(-?\d+),\s*(true|false)\)"
        r"\s+l1_c=\(\[(-?\d+),\s*(-?\d+)\],\s*(-?\d+),\s*(true|false)\)"
    )

    for i in range(start, end):
        m = neighbor_re.search(lines[i])
        if m and int(m.group(1)) == mb_x and int(m.group(2)) == mb_y:
            results.append({
                "part": int(m.group(3)),
                "j": int(m.group(4)),
                "sub_blk_x": int(m.group(5)),
                "sub_blk_y": int(m.group(6)),
                "list": 1,
                "a": {"mv": (int(m.group(7)), int(m.group(8))),
                      "ref": int(m.group(9)), "avail": m.group(10) == "true"},
                "b": {"mv": (int(m.group(11)), int(m.group(12))),
                      "ref": int(m.group(13)), "avail": m.group(14) == "true"},
                "c": {"mv": (int(m.group(15)), int(m.group(16))),
                      "ref": int(m.group(17)), "avail": m.group(18) == "true"},
            })

    return results


def extract_ffmpeg_neighbors(
    input_path: Path,
    target_poc: int,
    mb_x: int,
    mb_y: int,
    b_frames_before: int,
    list_filter: str,
) -> dict | None:
    """Extract neighbor MV cache values from FFmpeg via lldb.

    Returns dict with mv_cache and ref_cache values for the MB's neighbors.
    """
    ffmpeg_bin = find_ffmpeg_binary()

    func = "ff_h264_hl_decode_mb"
    condition = f"sl->mb_x == {mb_x} && sl->mb_y == {mb_y} && sl->slice_type_nos == 1"

    ignore = calibrate_ignore_count(
        ffmpeg_bin, str(input_path), func, condition,
        target_poc=target_poc, initial_guess=b_frames_before,
    )

    if ignore is None:
        print(f"Warning: could not calibrate lldb ignore count for POC {target_poc}",
              file=sys.stderr)
        return None

    expressions = ["(int)(h->cur_pic_ptr->field_poc[0])"]

    lists_to_extract = []
    if list_filter in ("l0", "both"):
        lists_to_extract.append(0)
    if list_filter in ("l1", "both"):
        lists_to_extract.append(1)

    # MV cache for all 16 blocks
    for list_idx in lists_to_extract:
        for blk in range(16):
            s8 = SCAN8[blk]
            expressions.append(f"sl->mv_cache[{list_idx}][{s8}]")

    # Ref cache for all 16 blocks
    for list_idx in lists_to_extract:
        for blk in range(16):
            s8 = SCAN8[blk]
            expressions.append(f"(int)sl->ref_cache[{list_idx}][{s8}]")

    # Neighbor border positions (A, B for block 0)
    s0 = SCAN8[0]
    for list_idx in lists_to_extract:
        expressions.append(f"sl->mv_cache[{list_idx}][{s0 - 1}]")      # A mv
        expressions.append(f"(int)sl->ref_cache[{list_idx}][{s0 - 1}]") # A ref
        expressions.append(f"sl->mv_cache[{list_idx}][{s0 - 8}]")      # B mv
        expressions.append(f"(int)sl->ref_cache[{list_idx}][{s0 - 8}]") # B ref

    result = run_lldb(
        ffmpeg_bin, str(input_path), expressions, func, condition,
        ignore_count=ignore,
    )

    if not result.values:
        return None

    return {"raw_values": result.values, "poc": result.values[0]}


def main():
    parser = argparse.ArgumentParser(
        description="Compare MV neighbor context between wedeo and FFmpeg"
    )
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--frame", type=int, required=True,
                        help="Output frame number (display order)")
    parser.add_argument("--mb", required=True,
                        help="Macroblock position as X,Y (e.g., 4,0)")
    parser.add_argument("--part", type=int, default=None,
                        help="Filter to specific partition (0-3)")
    parser.add_argument("--list", choices=["l0", "l1", "both"], default="both",
                        help="Which list to show")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    mb_parts = args.mb.split(",")
    if len(mb_parts) != 2:
        print("Error: --mb must be X,Y (e.g., 4,0)", file=sys.stderr)
        sys.exit(1)
    mb_x, mb_y = int(mb_parts[0]), int(mb_parts[1])

    # Get frame order info
    print("Getting frame order...", file=sys.stderr)
    frames = get_frame_order(str(input_path))
    target = next((f for f in frames if f.output_idx == args.frame), None)

    if target is None:
        print(f"Error: output frame {args.frame} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Target: output frame {args.frame}, decode_idx={target.decode_idx}, "
          f"poc={target.poc}, type={target.slice_type}",
          file=sys.stderr)

    b_before = count_slice_type_before(frames, target.decode_idx, "B")

    # Extract wedeo neighbors
    print("Extracting wedeo neighbor values...", file=sys.stderr)
    wedeo_neighbors = extract_wedeo_neighbors(
        input_path, target.decode_idx, mb_x, mb_y
    )

    # Extract FFmpeg cache state
    print("Extracting FFmpeg cache values...", file=sys.stderr)
    ffmpeg_data = extract_ffmpeg_neighbors(
        input_path, target.poc, mb_x, mb_y, b_before, args.list
    )

    # Display results
    print()
    print("=" * 70)
    print(f"Neighbor context for MB({mb_x},{mb_y}) in output frame {args.frame}")
    print("=" * 70)

    if wedeo_neighbors:
        print(f"\nWedeo L1 neighbors ({len(wedeo_neighbors)} entries):")
        for n in wedeo_neighbors:
            if args.part is not None and n["part"] != args.part:
                continue
            print(f"  part={n['part']} j={n['j']} "
                  f"blk=({n['sub_blk_x']},{n['sub_blk_y']}) list={n['list']}")
            for label in ("a", "b", "c"):
                nb = n[label]
                avail = "Y" if nb["avail"] else "N"
                print(f"    {label.upper()}: mv=({nb['mv'][0]:4d},{nb['mv'][1]:4d}) "
                      f"ref={nb['ref']:2d} avail={avail}")
    else:
        print("\nWedeo: no L1 neighbor traces found "
              "(add trace!() to B_8x8 L1 path in mb.rs)")

    if ffmpeg_data:
        print(f"\nFFmpeg POC: {ffmpeg_data['poc']}")
        print(f"  (Raw lldb values: {len(ffmpeg_data['raw_values'])} expressions extracted)")
    else:
        print("\nFFmpeg: could not extract cache values")

    print()


if __name__ == "__main__":
    main()
