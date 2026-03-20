#!/usr/bin/env python3
"""Extract motion vectors from FFmpeg via lldb for specific H.264 MBs.

Uses lldb conditional breakpoints on ff_h264_hl_decode_mb to stop at the
right macroblock in the right B-frame, then reads mv_cache/ref_cache.

Usage:
    # Dump MVs for MB(9,8) in output frame 23
    python3 scripts/ffmpeg_lldb_mv.py fate-suite/h264-conformance/BA3_SVA_C.264 --frame 23 --mb 9,8

    # Dump MVs for MB(3,4) in output frame 5
    python3 scripts/ffmpeg_lldb_mv.py fate-suite/h264-conformance/BA3_SVA_C.264 --frame 5 --mb 3,4

    # Also dump L0 MVs (default: L1 only)
    python3 scripts/ffmpeg_lldb_mv.py file.264 --frame 5 --mb 3,4 --list both

Requires:
    - Debug FFmpeg built with --disable-optimizations --enable-debug=3 --disable-asm
      at FFmpeg/ffmpeg_g (see CLAUDE.md for build instructions)
    - lldb in PATH
    - ffmpeg_debug.py (shared utilities)

Limitations:
    - Only works for B-frames (uses slice_type_nos == 3 condition).
    - POC type 0 only (inherits frame_order limitation).
    - The ignore-count calibration assumes all B-frames decode MB at the
      target position. Multi-slice B-frames where the target MB is in a
      later slice may produce wrong results.
    - No validation of MB coordinates against frame dimensions; out-of-bounds
      values cause the breakpoint to never fire and calibration to time out.
    - Must be run from the project root directory.
"""

import argparse
import sys
from pathlib import Path

from ffmpeg_debug import (
    BLK_XY,
    SCAN8,
    calibrate_ignore_count,
    count_slice_type_before,
    find_ffmpeg_binary,
    get_frame_order,
    h264_mv_preset,
    parse_h264_mv_result,
    run_lldb,
)


def format_output(data: dict, mb_x: int, mb_y: int, lists: list[int]):
    """Print extracted MV data in a readable format."""
    print(f"FFmpeg MVs for MB({mb_x},{mb_y}) poc={data['poc']}")
    print(f"  sub_mb_type: {data['sub_mb_type']}")
    print()

    part_names = ["8x8[0] (0,0)", "8x8[1] (2,0)", "8x8[2] (0,2)", "8x8[3] (2,2)"]

    for list_idx in lists:
        list_name = f"L{list_idx}"
        print(f"  {list_name} motion vectors:")
        refs = data["refs"][list_idx]
        mvs = data["mvs"][list_idx]

        for part in range(4):
            ref = refs[part]
            print(f"    {part_names[part]}  ref={ref}")
            for j in range(4):
                blk = part * 4 + j
                bx, by = BLK_XY[blk]
                mx, my = mvs[blk]
                print(f"      blk({bx},{by}): ({mx:>4},{my:>4})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract FFmpeg motion vectors via lldb"
    )
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument(
        "--frame", type=int, required=True,
        help="Output frame number (display order)"
    )
    parser.add_argument(
        "--mb", required=True,
        help="Macroblock position as X,Y (e.g., 9,8)"
    )
    parser.add_argument(
        "--list", default="l1", choices=["l0", "l1", "both"],
        help="Which reference list to dump (default: l1)"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    mb_parts = args.mb.split(",")
    if len(mb_parts) != 2:
        print("Error: --mb must be X,Y (e.g., 9,8)", file=sys.stderr)
        sys.exit(1)
    mb_x, mb_y = int(mb_parts[0]), int(mb_parts[1])

    if args.list == "both":
        lists = [0, 1]
    elif args.list == "l0":
        lists = [0]
    else:
        lists = [1]

    # Step 1: Get frame order mapping
    frames = get_frame_order(str(input_path))
    if not frames:
        print("Error: no frames found", file=sys.stderr)
        sys.exit(1)

    # Find target frame
    target = None
    for f in frames:
        if f.output_idx == args.frame:
            target = f
            break
    if target is None:
        print(f"Error: output frame {args.frame} not found", file=sys.stderr)
        sys.exit(1)

    if target.slice_type != "B":
        print(f"Error: frame {args.frame} is {target.slice_type}-type, "
              "not B. This tool only works for B-frames.", file=sys.stderr)
        sys.exit(1)

    poc = target.poc
    decode_idx = target.decode_idx

    # Step 2: Count B-frames before target to compute ignore count
    b_before = count_slice_type_before(frames, decode_idx, "B")
    # The breakpoint fires for every B-frame at this MB position.
    # We need to ignore b_before hits, then stop on the (b_before+1)th.
    # But lldb's ignore count may not match exactly due to internal
    # FFmpeg behavior. We calibrate by checking the POC.
    print(f"Target: output frame {args.frame}, decode_idx={decode_idx}, "
          f"poc={poc}, B-frames before={b_before}", file=sys.stderr)

    # Step 3: Find FFmpeg binary
    ffmpeg_bin = find_ffmpeg_binary()

    # Step 4: Calibrate ignore count
    # Start with b_before as initial guess, then search nearby.
    # The exact count may differ from the B-frame count due to FFmpeg internals.
    print("Calibrating lldb ignore count...", file=sys.stderr)

    func, condition, _ = h264_mv_preset(mb_x, mb_y, lists)

    found_ignore = calibrate_ignore_count(
        ffmpeg_bin=ffmpeg_bin,
        input_path=str(input_path),
        breakpoint_func=func,
        breakpoint_condition=condition,
        target_poc=poc,
        initial_guess=b_before,
    )

    if found_ignore is None:
        print(f"Error: could not calibrate ignore count for poc={poc}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Calibrated: ignore_count={found_ignore}", file=sys.stderr)

    # Step 5: Full extraction
    print("Extracting MVs...", file=sys.stderr)

    func, condition, expressions = h264_mv_preset(mb_x, mb_y, lists)

    result = run_lldb(
        ffmpeg_bin=ffmpeg_bin,
        input_path=str(input_path),
        expressions=expressions,
        breakpoint_func=func,
        breakpoint_condition=condition,
        ignore_count=found_ignore,
    )

    if not result.values:
        print("Error: no values extracted from lldb", file=sys.stderr)
        print("lldb stderr:", result.stderr[-500:], file=sys.stderr)
        sys.exit(1)

    expected = 1 + 16 * len(lists) + 4 * len(lists) + 4
    if len(result.values) != expected:
        print(f"Warning: got {len(result.values)} values, expected {expected}",
              file=sys.stderr)

    data = parse_h264_mv_result(result.values, lists)

    if data["poc"] not in {poc, poc | 0x10000}:
        print(f"Warning: extracted poc={data['poc']} != expected poc={poc}",
              file=sys.stderr)

    # Step 6: Display
    format_output(data, mb_x, mb_y, lists)


if __name__ == "__main__":
    main()
