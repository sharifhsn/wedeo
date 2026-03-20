#!/usr/bin/env python3
"""Show decode order, POC, output order, and slice type for an H.264 file.

Runs FFmpeg with trace_headers BSF to extract slice header info, then
computes the output (display) order from POC values.

Usage:
    python3 scripts/frame_order.py fate-suite/h264-conformance/BA3_SVA_C.264

Limitations:
    - Only handles POC type 0 (pic_order_cnt_lsb). POC types 1 and 2 are
      silently ignored, producing wrong output order for those streams.
    - Multi-slice frames: only the first slice (first_mb_in_slice=0) is recorded.
    - Must be run from the project root directory.
"""

import argparse
import sys
from pathlib import Path

from ffmpeg_debug import get_frame_order


def main():
    parser = argparse.ArgumentParser(description="H.264 frame order map")
    parser.add_argument("input", help="H.264 input file")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    frames = get_frame_order(input_path)

    if not frames:
        print("No frames found. Is this a valid H.264 file?", file=sys.stderr)
        sys.exit(1)

    # Print table
    print(f"{'Decode':>6} {'POC':>4} {'Output':>6} {'Type':>5} {'FrmNum':>6} {'Ref':>3}")
    print("-" * 40)
    for f in frames:
        ref = "Y" if f.nal_ref_idc > 0 else "N"
        print(
            f"{f.decode_idx:>6} {f.poc:>4} {f.output_idx:>6} {f.slice_type:>5} "
            f"{f.frame_num:>6} {ref:>3}"
        )

    print(f"\nTotal: {len(frames)} frames")


if __name__ == "__main__":
    main()
