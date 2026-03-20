#!/usr/bin/env python3
"""Dump per-MB types and MVs for a specific output frame of an H.264 file.

Runs wedeo with tracing enabled and parses the trace output to show
each MB's type, sub_mb_types (for B_8x8), and motion vectors.

Usage:
    python3 scripts/mb_types.py fate-suite/h264-conformance/BA3_SVA_C.264 --frame 3

Limitations:
    - Only supports B-frame analysis. P/I frames exit with a message.
    - Multi-slice B-frames may produce incorrect MB grids (slice boundary
      detection assumes one slice per B-frame).
    - Must be run from the project root directory.
    - Builds the debug binary with tracing (slow for large files).
"""

import argparse
import re
import sys
from pathlib import Path

from ffmpeg_debug import run_wedeo_with_tracing

# B-frame mb_type names (raw_mb_type 0-22, then 23+ = intra offset by 23)
B_MB_TYPE_NAMES = [
    "B_Direct_16x16",
    "B_L0_16x16", "B_L1_16x16", "B_Bi_16x16",
    "B_L0_L0_16x8", "B_L0_L0_8x16",
    "B_L1_L1_16x8", "B_L1_L1_8x16",
    "B_L0_L1_16x8", "B_L0_L1_8x16",
    "B_L1_L0_16x8", "B_L1_L0_8x16",
    "B_L0_Bi_16x8", "B_L0_Bi_8x16",
    "B_L1_Bi_16x8", "B_L1_Bi_8x16",
    "B_Bi_L0_16x8", "B_Bi_L0_8x16",
    "B_Bi_L1_16x8", "B_Bi_L1_8x16",
    "B_Bi_Bi_16x8", "B_Bi_Bi_8x16",
    "B_8x8",
]

B_SUB_TYPE_NAMES = [
    "B_Direct_8x8",
    "B_L0_8x8", "B_L1_8x8", "B_Bi_8x8",
    "B_L0_8x4", "B_L0_4x8",
    "B_L1_8x4", "B_L1_4x8",
    "B_Bi_8x4", "B_Bi_4x8",
    "B_L0_4x4", "B_L1_4x4", "B_Bi_4x4",
]


def main():
    parser = argparse.ArgumentParser(description="Per-MB type dump for H.264 B-frames")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--frame", type=int, required=True, help="Output frame number")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Build and run with tracing for decoder (frame info) and mb (types) and cavlc (mb_type parsing)
    trace = run_wedeo_with_tracing(
        input_path,
        rust_log="wedeo_codec_h264::decoder=trace,wedeo_codec_h264::mb=trace,wedeo_codec_h264::cavlc=trace",
    )

    # Parse frame sequence: find slice starts and completions
    frames = []  # list of (poc, slice_type, decode_idx)
    decode_idx = 0
    current_slice_type = None
    for line in trace.split("\n"):
        m = re.search(r'slice start slice_type=(\w+) frame_num=(\d+)', line)
        if m:
            current_slice_type = m.group(1)

        m = re.search(r'frame complete frame_num=(\d+) poc=(-?\d+)', line)
        if m:
            poc = int(m.group(2))
            frames.append({
                "decode_idx": decode_idx,
                "poc": poc,
                "slice_type": current_slice_type or "?",
            })
            decode_idx += 1

    if not frames:
        print("No frames found in trace", file=sys.stderr)
        sys.exit(1)

    # Compute output order
    sorted_by_poc = sorted(enumerate(frames), key=lambda x: x[1]["poc"])
    for out_idx, (dec_idx, frame) in enumerate(sorted_by_poc):
        frame["output_idx"] = out_idx

    # Find the target frame
    target = None
    for f in frames:
        if f["output_idx"] == args.frame:
            target = f
            break

    if target is None:
        print(f"Output frame {args.frame} not found (max: {len(frames)-1})", file=sys.stderr)
        sys.exit(1)

    print(f"Output frame {args.frame}: decode_idx={target['decode_idx']} "
          f"poc={target['poc']} type={target['slice_type']}")

    if target["slice_type"] not in ("B",):
        print(f"Frame is {target['slice_type']}-type, not B. "
              "This tool only supports B-frame MB type analysis.")
        sys.exit(0)

    # Parse skip runs and MB types for the target frame's B-slice
    # We identify the slice by matching the decode_idx with the sequence of
    # skip_run and mb_type traces.
    # Since traces are interleaved, we track by counting B-slice boundaries.

    # Count which B-slice this is (0-indexed among all B-slices)
    b_slice_count = 0
    for f in frames:
        if f["decode_idx"] < target["decode_idx"] and f["slice_type"] == "B":
            b_slice_count += 1

    target_b_idx = b_slice_count
    print(f"B-slice index: {target_b_idx} (0-indexed among B-slices)")

    # Parse all B-slice skip_run and mb_type entries
    b_slices = []
    current_b_slice = None
    in_b_slice = False

    for line in trace.split("\n"):
        if "slice_type=B" in line and "slice start" in line:
            if current_b_slice is not None:
                b_slices.append(current_b_slice)
            current_b_slice = {"skip_runs": [], "mb_types": [], "sub_types": []}
            in_b_slice = True
        elif "slice start" in line and "slice_type=B" not in line:
            if current_b_slice is not None:
                b_slices.append(current_b_slice)
                current_b_slice = None
            in_b_slice = False

        if in_b_slice and current_b_slice is not None:
            # Skip run
            m = re.search(r'mb_skip_run mb_addr=(\d+) mb_skip_run=(\d+)', line)
            if m:
                current_b_slice["skip_runs"].append((int(m.group(1)), int(m.group(2))))

            # MB type
            m = re.search(r'MB type parsed raw_mb_type=(\d+).*bits_consumed=(\d+)', line)
            if m:
                current_b_slice["mb_types"].append(int(m.group(1)))

            # B_8x8 sub types
            m = re.search(r'B_8x8 sub_mb_types mb_x=(\d+) mb_y=(\d+) sub0=(\d+) sub1=(\d+) sub2=(\d+) sub3=(\d+)', line)
            if m:
                current_b_slice["sub_types"].append({
                    "mb_x": int(m.group(1)), "mb_y": int(m.group(2)),
                    "subs": [int(m.group(i)) for i in range(3, 7)]
                })

    if current_b_slice is not None:
        b_slices.append(current_b_slice)

    if target_b_idx >= len(b_slices):
        print(f"B-slice {target_b_idx} not found (only {len(b_slices)} B-slices)", file=sys.stderr)
        sys.exit(1)

    bs = b_slices[target_b_idx]

    # Determine frame dimensions from trace
    m = re.search(r'SPS parsed sps_id=\d+ width=(\d+) height=(\d+)', trace)
    if m:
        width, height = int(m.group(1)), int(m.group(2))
    else:
        width, height = 176, 144
        print("Warning: SPS not found in trace, assuming 176x144", file=sys.stderr)

    mb_w = width // 16
    mb_h = height // 16

    # Reconstruct MB types from skip_runs and mb_types.
    # Use skip_addr from the trace to sync position (handles any counting bugs).
    mb_grid = {}  # (mx, my) -> type_str
    type_idx = 0
    total_mbs = mb_w * mb_h

    for i, (skip_addr, skip_count) in enumerate(bs["skip_runs"]):
        # Fill skipped MBs from skip_addr
        for j in range(skip_count):
            addr = skip_addr + j
            if addr < total_mbs:
                mx, my = addr % mb_w, addr // mb_w
                mb_grid[(mx, my)] = "B_Skip"

        # Coded MB follows at skip_addr + skip_count
        coded_addr = skip_addr + skip_count
        if coded_addr < total_mbs and type_idx < len(bs["mb_types"]):
            mx, my = coded_addr % mb_w, coded_addr // mb_w
            raw_type = bs["mb_types"][type_idx]
            if raw_type < 23:
                name = B_MB_TYPE_NAMES[raw_type]
            else:
                name = f"Intra({raw_type - 23})"
            mb_grid[(mx, my)] = name
            type_idx += 1

    # Fill remaining as B_Skip (any MB not assigned yet)
    for addr in range(total_mbs):
        mx, my = addr % mb_w, addr // mb_w
        if (mx, my) not in mb_grid:
            mb_grid[(mx, my)] = "B_Skip"

    # Build sub_type lookup
    sub_lookup = {}
    for st in bs["sub_types"]:
        sub_lookup[(st["mb_x"], st["mb_y"])] = st["subs"]

    # Print grid
    print(f"\nMB grid ({mb_w}x{mb_h}):")
    for my in range(mb_h):
        for mx in range(mb_w):
            name = mb_grid.get((mx, my), "???")
            subs = sub_lookup.get((mx, my))
            if subs:
                sub_names = [B_SUB_TYPE_NAMES[s] if s < len(B_SUB_TYPE_NAMES) else f"?{s}"
                             for s in subs]
                short_subs = ",".join(str(s) for s in subs)
                print(f"  MB({mx},{my}): {name} [{short_subs}]")
            elif name != "B_Skip":
                print(f"  MB({mx},{my}): {name}")
            # Skip printing B_Skip MBs unless they're few
        # Print skip summary for this row
        skips = [mx for mx in range(mb_w) if mb_grid.get((mx, my)) == "B_Skip"]
        if skips:
            if len(skips) == mb_w:
                print(f"  Row {my}: all B_Skip")
            else:
                print(f"  Row {my} B_Skip: {skips}")


if __name__ == "__main__":
    main()
