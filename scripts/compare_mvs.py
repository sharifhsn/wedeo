#!/usr/bin/env python3
"""Compare motion vectors between wedeo and FFmpeg for a specific H.264 MB.

Extracts MVs from FFmpeg via lldb (using h264_mv_preset + calibrate_ignore_count)
and from wedeo via tracing, then diffs per-block MVs and reports mismatches.

This replaces the manual 4-step workflow:
  1. mb_compare.py (find differing MBs)
  2. mb_types.py (get MB types)
  3. ffmpeg_lldb_mv.py (extract FFmpeg MVs)
  4. manual comparison

Usage:
    python3 scripts/compare_mvs.py file.264 --frame 23 --mb 9,8
    python3 scripts/compare_mvs.py file.264 --frame 5 --mb 3,4 --list both

Requires:
    - Debug FFmpeg built with --disable-optimizations --enable-debug=3 --disable-asm
    - wedeo-framecrc binary with tracing feature
    - lldb in PATH
"""

import argparse
import re
import sys
from pathlib import Path

from ffmpeg_debug import (
    BLK_XY,
    find_ffmpeg_binary,
    find_wedeo_binary,
    get_frame_order,
    count_slice_type_before,
    h264_mv_preset,
    parse_h264_mv_result,
    run_lldb,
    calibrate_ignore_count,
    run_wedeo_with_tracing,
    strip_ansi,
)


def extract_wedeo_mvs(input_path: str, frame_idx: int, mb_x: int, mb_y: int) -> dict | None:
    """Extract MV info from wedeo via tracing for a specific frame and MB.

    Returns dict with keys: mvs (per-list), refs (per-list), sub_mb_type, poc.
    Returns None if no data found.
    """
    trace = run_wedeo_with_tracing(
        input_path,
        rust_log="wedeo_codec_h264=trace",
        no_deblock=True,
    )

    # Collect per-frame data keyed by POC
    all_frames: dict[int, dict] = {}  # poc -> {(mx,my): info}
    current_frame_mbs: dict[tuple[int, int], dict] = {}
    frames_completed = 0

    for line in trace.splitlines():
        if "frame complete" in line:
            m = re.search(r"poc=(-?\d+)", line)
            poc = int(m.group(1)) if m else frames_completed * 2
            all_frames[poc] = dict(current_frame_mbs)
            current_frame_mbs = {}
            frames_completed += 1
            continue

        # Track MV data for all MBs
        if "MV" in line and any(p in line for p in ("16x16", "16x8", "8x16", "8x8", "4x4", "4x8", "8x4")):
            m_x = re.search(r"mb_x=(\d+)", line)
            m_y = re.search(r"mb_y=(\d+)", line)
            if not m_x or not m_y:
                continue
            mx, my = int(m_x.group(1)), int(m_y.group(1))
            key = (mx, my)
            if key not in current_frame_mbs:
                current_frame_mbs[key] = {"mvs": {}, "refs": {}}
            info = current_frame_mbs[key]

            m_mv = re.search(r"mv=\[(-?\d+),\s*(-?\d+)\]", line)
            m_list = re.search(r"list=(\d+)", line)
            m_blk = re.search(r"blk=(\d+)", line)
            m_ref = re.search(r"ref_idx=(\d+)", line)

            if m_mv and m_list and m_blk:
                list_idx = int(m_list.group(1))
                blk_idx = int(m_blk.group(1))
                if list_idx not in info["mvs"]:
                    info["mvs"][list_idx] = {}
                info["mvs"][list_idx][blk_idx] = (
                    int(m_mv.group(1)), int(m_mv.group(2))
                )
                if m_ref:
                    if list_idx not in info["refs"]:
                        info["refs"][list_idx] = {}
                    info["refs"][list_idx][blk_idx] = int(m_ref.group(1))

        # Track MB type
        if "MB type parsed" in line or "decoded MB" in line:
            m_x = re.search(r"mb_x=(\d+)", line)
            m_y = re.search(r"mb_y=(\d+)", line)
            if m_x and m_y:
                mx, my = int(m_x.group(1)), int(m_y.group(1))
                key = (mx, my)
                if key not in current_frame_mbs:
                    current_frame_mbs[key] = {"mvs": {}, "refs": {}}
                for field in ["raw_mb_type", "mb_type"]:
                    m_f = re.search(rf"{field}=(\S+)", line)
                    if m_f:
                        current_frame_mbs[key][field] = m_f.group(1)

    # Find target frame by output order (sorted by POC)
    sorted_pocs = sorted(all_frames.keys())
    if frame_idx >= len(sorted_pocs):
        return None

    target_poc = sorted_pocs[frame_idx]
    frame_data = all_frames[target_poc]
    mb_data = frame_data.get((mb_x, mb_y))
    if mb_data is None:
        return None

    mb_data["poc"] = target_poc
    return mb_data


def format_comparison(
    ffmpeg_data: dict,
    wedeo_data: dict | None,
    mb_x: int,
    mb_y: int,
    lists: list[int],
):
    """Print side-by-side MV comparison."""
    part_names = ["8x8[0] (0,0)", "8x8[1] (2,0)", "8x8[2] (0,2)", "8x8[3] (2,2)"]

    print(f"\n{'='*70}")
    print(f"MV comparison for MB({mb_x},{mb_y})")
    print(f"  FFmpeg POC: {ffmpeg_data['poc']}")
    if wedeo_data:
        print(f"  Wedeo POC:  {wedeo_data.get('poc', '?')}")
    else:
        print(f"  Wedeo: no MV data found")
    print(f"  FFmpeg sub_mb_type: {ffmpeg_data['sub_mb_type']}")
    print(f"{'='*70}")

    total_blocks = 0
    matching_blocks = 0
    mismatches = []

    for list_idx in lists:
        list_name = f"L{list_idx}"
        print(f"\n  {list_name} motion vectors:")

        ff_mvs = ffmpeg_data["mvs"].get(list_idx, [(0, 0)] * 16)
        ff_refs = ffmpeg_data["refs"].get(list_idx, [-99] * 4)
        w_mvs = wedeo_data.get("mvs", {}).get(list_idx, {}) if wedeo_data else {}
        w_refs = wedeo_data.get("refs", {}).get(list_idx, {}) if wedeo_data else {}

        for part in range(4):
            ff_ref = ff_refs[part]
            print(f"    {part_names[part]}  ref={ff_ref}")
            for j in range(4):
                blk = part * 4 + j
                bx, by = BLK_XY[blk]
                ff_mx, ff_my = ff_mvs[blk] if blk < len(ff_mvs) else (0, 0)

                total_blocks += 1
                w_mv = w_mvs.get(blk)

                if w_mv is not None:
                    w_mx, w_my = w_mv
                    match = ff_mx == w_mx and ff_my == w_my
                    if match:
                        matching_blocks += 1
                        status = "OK"
                    else:
                        status = "DIFF"
                        mismatches.append((list_idx, blk, bx, by, (ff_mx, ff_my), (w_mx, w_my)))
                    print(
                        f"      blk({bx},{by}): ffmpeg=({ff_mx:>4},{ff_my:>4}) "
                        f"wedeo=({w_mx:>4},{w_my:>4})  {status}"
                    )
                else:
                    print(
                        f"      blk({bx},{by}): ffmpeg=({ff_mx:>4},{ff_my:>4}) "
                        f"wedeo=  N/A"
                    )

    print(f"\n{'─'*70}")
    if wedeo_data and w_mvs:
        print(f"Summary: {matching_blocks}/{total_blocks} blocks match")
        if mismatches:
            print(f"Mismatches ({len(mismatches)}):")
            for li, blk, bx, by, ff, w in mismatches:
                print(f"  L{li} blk({bx},{by}): ffmpeg={ff} wedeo={w} delta=({w[0]-ff[0]},{w[1]-ff[1]})")
        else:
            print("ALL MATCH!")
    else:
        print("Cannot compare: wedeo MV data not available")


def main():
    parser = argparse.ArgumentParser(
        description="Compare motion vectors between wedeo and FFmpeg"
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
        "--list", default="both", choices=["l0", "l1", "both"],
        help="Which reference list to compare (default: both)"
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
    print("Getting frame order...", file=sys.stderr)
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
        print(
            f"Warning: frame {args.frame} is {target.slice_type}-type, not B. "
            "LLDB extraction uses B-frame condition by default.",
            file=sys.stderr,
        )

    poc = target.poc
    decode_idx = target.decode_idx
    b_before = count_slice_type_before(frames, decode_idx, "B")
    print(
        f"Target: output frame {args.frame}, decode_idx={decode_idx}, "
        f"poc={poc}, type={target.slice_type}, B-frames before={b_before}",
        file=sys.stderr,
    )

    # Step 2: Extract FFmpeg MVs via lldb
    ffmpeg_bin = find_ffmpeg_binary()
    func, condition, expressions = h264_mv_preset(mb_x, mb_y, lists=lists)

    print("Calibrating lldb ignore count...", file=sys.stderr)
    ignore = calibrate_ignore_count(
        ffmpeg_bin, str(input_path), func, condition, poc,
        initial_guess=b_before,
    )
    if ignore is None:
        print(f"Error: could not calibrate ignore count for poc={poc}", file=sys.stderr)
        sys.exit(1)
    print(f"Calibrated: ignore_count={ignore}", file=sys.stderr)

    print("Extracting FFmpeg MVs...", file=sys.stderr)
    lldb_result = run_lldb(
        ffmpeg_bin, str(input_path), expressions, func,
        breakpoint_condition=condition, ignore_count=ignore,
    )
    if not lldb_result.values:
        print("Error: no values extracted from lldb", file=sys.stderr)
        sys.exit(1)

    ffmpeg_data = parse_h264_mv_result(lldb_result.values, lists)
    if ffmpeg_data["poc"] not in {poc, poc | 0x10000}:
        print(
            f"Warning: extracted poc={ffmpeg_data['poc']} != expected poc={poc}",
            file=sys.stderr,
        )

    # Step 3: Extract wedeo MVs via tracing
    print("Extracting wedeo MVs via tracing...", file=sys.stderr)
    wedeo_data = extract_wedeo_mvs(str(input_path), args.frame, mb_x, mb_y)

    # Step 4: Compare and display
    format_comparison(ffmpeg_data, wedeo_data, mb_x, mb_y, lists)


if __name__ == "__main__":
    main()
