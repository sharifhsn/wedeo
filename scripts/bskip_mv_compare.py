#!/usr/bin/env python3
"""Compare B_Skip/B_Direct motion vectors between wedeo and FFmpeg.

B_Skip MBs use spatial or temporal direct prediction, producing per-4x4 block
MVs that are stored but NOT traced by the standard "B_8x8 sub-partition MV"
path. This script fills that gap by:
  1. Extracting wedeo's B_Skip MVs via the "B_Skip MV" trace message
  2. Extracting FFmpeg's MVs via lldb at ff_h264_hl_decode_mb
  3. Comparing per-block and reporting mismatches

Usage:
    python3 scripts/bskip_mv_compare.py file.264 --frame 2 --mb 3,0
    python3 scripts/bskip_mv_compare.py file.264 --frame 2 --mb 3,0 --list l1

Requires:
    - Debug FFmpeg with --disable-asm
    - wedeo-framecrc with tracing feature (and B_Skip MV trace!() in mb.rs)
    - lldb in PATH
"""

import argparse
import re
import sys
from pathlib import Path

from ffmpeg_debug import (
    BLK_XY,
    find_ffmpeg_binary,
    get_frame_order,
    count_slice_type_before,
    h264_mv_preset,
    parse_h264_mv_result,
    run_lldb,
    calibrate_ignore_count,
    run_wedeo_with_tracing,
)


def extract_wedeo_bskip_mvs(
    input_path: Path,
    target_decode_idx: int,
    mb_x: int,
    mb_y: int,
) -> dict | None:
    """Extract B_Skip MVs from wedeo trace for a specific frame and MB.

    Returns dict mapping i4 -> (mv_l0, ref_l0, mv_l1, ref_l1), or None.
    Requires temporary trace!() calls in decode_b_skip_mb.
    """
    trace = run_wedeo_with_tracing(
        str(input_path),
        rust_log="wedeo_codec_h264::mb=trace",
        no_deblock=True,
        features=["tracing"],
    )

    # Track frame boundaries
    lines = trace.splitlines()
    frame_sections: list[tuple[int, int]] = []
    section_start = 0

    for i, line in enumerate(lines):
        if "frame complete" in line:
            frame_sections.append((section_start, i))
            section_start = i + 1

    # Handle last section
    if section_start < len(lines):
        frame_sections.append((section_start, len(lines)))

    if target_decode_idx >= len(frame_sections):
        print(f"Error: decode_idx {target_decode_idx} not found "
              f"({len(frame_sections)} frames)", file=sys.stderr)
        return None

    start, end = frame_sections[target_decode_idx]

    # Parse B_Skip MV traces
    bskip_re = re.compile(
        r"B_Skip MV\s+mb_x=(\d+)\s+mb_y=(\d+)\s+i4=(\d+)"
        r"\s+mv_l0_x=(-?\d+)\s+mv_l0_y=(-?\d+)\s+ref_l0=(-?\d+)"
        r"\s+mv_l1_x=(-?\d+)\s+mv_l1_y=(-?\d+)\s+ref_l1=(-?\d+)"
    )

    mvs: dict[int, tuple] = {}
    for i in range(start, end):
        m = bskip_re.search(lines[i])
        if m and int(m.group(1)) == mb_x and int(m.group(2)) == mb_y:
            i4 = int(m.group(3))
            mvs[i4] = (
                (int(m.group(4)), int(m.group(5))),  # mv_l0
                int(m.group(6)),                       # ref_l0
                (int(m.group(7)), int(m.group(8))),  # mv_l1
                int(m.group(9)),                       # ref_l1
            )

    return mvs if mvs else None


def main():
    parser = argparse.ArgumentParser(
        description="Compare B_Skip/B_Direct MVs between wedeo and FFmpeg"
    )
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--frame", type=int, required=True, help="Output frame number")
    parser.add_argument("--mb", required=True, help="MB position as X,Y")
    parser.add_argument("--list", choices=["l0", "l1", "both"], default="both",
                        help="Which list to compare")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    mb_parts = args.mb.split(",")
    if len(mb_parts) != 2:
        print("Error: --mb must be X,Y (e.g., 3,0)", file=sys.stderr)
        sys.exit(1)
    mb_x, mb_y = int(mb_parts[0]), int(mb_parts[1])

    # Frame order
    print("Getting frame order...", file=sys.stderr)
    frames = get_frame_order(str(input_path))
    target = next((f for f in frames if f.output_idx == args.frame), None)
    if target is None:
        print(f"Error: output frame {args.frame} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Target: decode_idx={target.decode_idx}, poc={target.poc}, "
          f"type={target.slice_type}", file=sys.stderr)
    b_before = count_slice_type_before(frames, target.decode_idx, "B")

    # Extract wedeo B_Skip MVs
    print("Extracting wedeo B_Skip MVs...", file=sys.stderr)
    wedeo_mvs = extract_wedeo_bskip_mvs(input_path, target.decode_idx, mb_x, mb_y)

    # Extract FFmpeg MVs via lldb
    print("Extracting FFmpeg MVs...", file=sys.stderr)
    ffmpeg_bin = find_ffmpeg_binary()
    lists = [0, 1]
    func, condition, expressions = h264_mv_preset(mb_x, mb_y, lists)

    ignore = calibrate_ignore_count(
        ffmpeg_bin, str(input_path), func, condition,
        target_poc=target.poc, initial_guess=b_before,
    )

    ffmpeg_mvs = None
    if ignore is not None:
        result = run_lldb(
            ffmpeg_bin, str(input_path), expressions, func, condition,
            ignore_count=ignore,
        )
        if result.values:
            ffmpeg_mvs = parse_h264_mv_result(result.values, lists)

    # Display comparison
    print()
    print("=" * 70)
    print(f"B_Skip MV comparison for MB({mb_x},{mb_y}) frame {args.frame}")
    print("=" * 70)

    show_l0 = args.list in ("l0", "both")
    show_l1 = args.list in ("l1", "both")

    # Per-block comparison
    mismatches = 0
    for i4 in range(16):
        bx, by = BLK_XY[i4]
        # Map 4x4 block to 8x8 partition for ref lookup
        part_8x8 = (by // 2) * 2 + (bx // 2)

        w_l0 = w_l1 = w_r0 = w_r1 = None
        if wedeo_mvs and i4 in wedeo_mvs:
            w_l0, w_r0, w_l1, w_r1 = wedeo_mvs[i4]

        f_l0 = f_l1 = None
        f_r0 = f_r1 = None
        if ffmpeg_mvs:
            # mvs[list_idx] is a list of (mx, my) tuples indexed by block
            f_l0 = ffmpeg_mvs["mvs"][0][i4] if 0 in ffmpeg_mvs.get("mvs", {}) else None
            f_l1 = ffmpeg_mvs["mvs"][1][i4] if 1 in ffmpeg_mvs.get("mvs", {}) else None
            # refs[list_idx] is a list of ref values indexed by 8x8 partition
            f_r0 = ffmpeg_mvs["refs"][0][part_8x8] if 0 in ffmpeg_mvs.get("refs", {}) else None
            f_r1 = ffmpeg_mvs["refs"][1][part_8x8] if 1 in ffmpeg_mvs.get("refs", {}) else None

        for list_idx, show, w_mv, w_ref, f_mv, f_ref in [
            (0, show_l0, w_l0, w_r0, f_l0, f_r0),
            (1, show_l1, w_l1, w_r1, f_l1, f_r1),
        ]:
            if not show:
                continue

            match = True
            w_str = (f"({w_mv[0]:4d},{w_mv[1]:4d}) ref={w_ref}"
                     if w_mv is not None else "  N/A")
            f_str = (f"({f_mv[0]:4d},{f_mv[1]:4d}) ref={f_ref}"
                     if f_mv is not None else "  N/A")

            if w_mv is not None and f_mv is not None:
                if w_mv != f_mv or w_ref != f_ref:
                    match = False
                    mismatches += 1

            marker = " " if match else "!"
            print(f"  {marker} blk({bx},{by}) L{list_idx}: wedeo={w_str}  ffmpeg={f_str}")

    print(f"\n{'MATCH' if mismatches == 0 else f'{mismatches} MISMATCHES'}")


if __name__ == "__main__":
    main()
