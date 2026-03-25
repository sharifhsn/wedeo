#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Check which direct prediction mode B-frames use in an H.264 file.

Extracts direct_spatial_mv_pred_flag from all B-slice headers via
trace_headers BSF. Essential first step before investigating direct
prediction bugs — avoids fixing the wrong code path.

Usage:
    python3 scripts/check_direct_mode.py file.264
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from ffmpeg_debug import find_ffmpeg_binary, resolve_conformance_file


def main():
    parser = argparse.ArgumentParser(
        description="Check spatial vs temporal direct mode in B-frames",
    )
    parser.add_argument("input", help="H.264 file (path or conformance name)")
    args = parser.parse_args()

    input_path = str(resolve_conformance_file(args.input))
    ffmpeg = find_ffmpeg_binary()

    result = subprocess.run(
        [str(ffmpeg), "-i", input_path, "-c:v", "copy", "-bsf:v",
         "trace_headers", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stderr

    spatial_count = 0
    temporal_count = 0
    b_slices = 0
    current_slice_type = None

    for line in output.splitlines():
        m_type = re.search(r'slice_type\s+\S+\s*=\s*(\d+)', line)
        if m_type:
            current_slice_type = int(m_type.group(1))

        m_direct = re.search(r'direct_spatial_mv_pred_flag\s+\d+\s*=\s*(\d+)', line)
        if m_direct:
            b_slices += 1
            if int(m_direct.group(1)) == 1:
                spatial_count += 1
            else:
                temporal_count += 1

    print(f"File: {Path(input_path).name}")
    print(f"B-slices: {b_slices}")
    if b_slices > 0:
        print(f"  Spatial direct:  {spatial_count} ({100*spatial_count//b_slices}%)")
        print(f"  Temporal direct: {temporal_count} ({100*temporal_count//b_slices}%)")
        mode = "SPATIAL" if spatial_count > temporal_count else "TEMPORAL"
        if spatial_count > 0 and temporal_count > 0:
            mode = "MIXED"
        print(f"  Primary mode: {mode}")
    else:
        print("  No B-slices found (I/P only)")


if __name__ == "__main__":
    main()
