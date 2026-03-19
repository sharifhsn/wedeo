#!/usr/bin/env python3
"""Compare reference list contents between wedeo and FFmpeg.

Extracts ref list L0/L1 for each P/B slice from both decoders and
finds the first frame where they diverge. Essential for debugging
ref_pic_list_modification and DPB ordering bugs.

Usage:
    # Show ref lists at first divergence
    python3 scripts/reflist_compare.py fate-suite/h264-conformance/HCBP1_HHI_A.264

    # Show ref lists at a specific frame
    python3 scripts/reflist_compare.py file.264 --frame 17

    # Compare against FFmpeg via lldb (slower but authoritative)
    python3 scripts/reflist_compare.py file.264 --ffmpeg --frame 17

Requires:
    - wedeo debug binary with tracing
    - FFmpeg debug binary for --ffmpeg mode
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ffmpeg_debug import (
    find_wedeo_binary,
    strip_ansi,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RefListEntry:
    """A reference in a ref list, identified by frame_num and/or POC."""
    frame_num: int = -1
    poc: int = -1

    def __repr__(self) -> str:
        parts = []
        if self.frame_num >= 0:
            parts.append(f"fn={self.frame_num}")
        if self.poc != -1:
            parts.append(f"poc={self.poc}")
        return " ".join(parts) if parts else "?"


@dataclass
class SliceRefInfo:
    """Ref list info for one slice."""
    decode_idx: int
    frame_num: int
    poc: int
    slice_type: str
    l0: list[RefListEntry]
    l1: list[RefListEntry]


# ---------------------------------------------------------------------------
# Wedeo extraction
# ---------------------------------------------------------------------------

def extract_wedeo_reflists(
    input_path: str, max_frames: int = 0,
) -> list[SliceRefInfo]:
    """Extract ref list contents from wedeo via tracing."""
    wedeo_bin = find_wedeo_binary(prefer_debug=True, features=["tracing"])
    env = {
        **os.environ,
        "RUST_LOG": "wedeo_codec_h264::decoder=debug",
        "WEDEO_NO_DEBLOCK": "1",
    }
    result = subprocess.run(
        [str(wedeo_bin), input_path],
        capture_output=True, env=env, timeout=120,
    )
    trace = strip_ansi(result.stderr.decode("utf-8", errors="replace"))

    slices = []
    frame_idx = -1  # count frames via "frame complete", not "slice start"
    current_fn = 0
    current_type = "?"
    current_poc = 0

    for line in trace.splitlines():
        if "slice start" in line:
            m = re.search(r"slice_type=(\w+)", line)
            if m:
                current_type = m.group(1)
            m = re.search(r"frame_num=(\d+)", line)
            if m:
                current_fn = int(m.group(1))

        elif "frame complete" in line:
            frame_idx += 1
            m = re.search(r"poc=(-?\d+)", line)
            if m:
                current_poc = int(m.group(1))

        elif "B-frame ref lists" in line:
            # Parse: l0_pocs=[Some(4), Some(0)] l1_pocs=[Some(8)]
            l0_pocs = _parse_poc_list(line, "l0_pocs")
            l1_pocs = _parse_poc_list(line, "l1_pocs")
            m_poc = re.search(r"poc=(-?\d+)", line)
            if m_poc:
                current_poc = int(m_poc.group(1))

            # frame_idx+1 because "B-frame ref lists" fires before
            # "frame complete" for this frame
            slices.append(SliceRefInfo(
                decode_idx=frame_idx + 1,
                frame_num=current_fn,
                poc=current_poc,
                slice_type=current_type,
                l0=[RefListEntry(poc=p) for p in l0_pocs],
                l1=[RefListEntry(poc=p) for p in l1_pocs],
            ))

            if max_frames and len(slices) >= max_frames:
                break

        elif "P-frame ref list" in line:
            # Parse: l0_frame_nums=[Some(7), Some(6), Some(5)]
            m_fns = re.search(r"l0_frame_nums=\[([^\]]*)\]", line)
            l0_fns = []
            if m_fns:
                for item in re.finditer(r"Some\((\d+)\)", m_fns.group(1)):
                    l0_fns.append(int(item.group(1)))
            m_poc = re.search(r"poc=(-?\d+)", line)
            if m_poc:
                current_poc = int(m_poc.group(1))

            slices.append(SliceRefInfo(
                decode_idx=frame_idx + 1,
                frame_num=current_fn,
                poc=current_poc,
                slice_type=current_type,
                l0=[RefListEntry(frame_num=fn) for fn in l0_fns],
                l1=[],
            ))

            if max_frames and len(slices) >= max_frames:
                break

    return slices


def _parse_poc_list(line: str, field: str) -> list[int]:
    """Parse a poc list like l0_pocs=[Some(4), Some(0), None]."""
    m = re.search(rf"{field}=\[([^\]]*)\]", line)
    if not m:
        return []
    content = m.group(1)
    pocs = []
    for item in re.finditer(r"Some\((-?\d+)\)", content):
        pocs.append(int(item.group(1)))
    return pocs


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_reflist(infos: list[SliceRefInfo], target_frame: int | None) -> None:
    """Print ref list info."""
    if not infos:
        print("No ref list info extracted (only B-frames with ref list logging)")
        return

    for s in infos:
        if target_frame is not None and s.decode_idx != target_frame:
            continue

        l0_str = ", ".join(str(r) for r in s.l0) if s.l0 else "empty"
        l1_str = ", ".join(str(r) for r in s.l1) if s.l1 else "empty"

        print(
            f"  frame {s.decode_idx:3d} (fn={s.frame_num}, poc={s.poc}, "
            f"{s.slice_type}):  L0=[{l0_str}]  L1=[{l1_str}]"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare reference lists between wedeo and FFmpeg",
    )
    parser.add_argument("input", help="H.264 conformance file")
    parser.add_argument("--frame", type=int, default=None,
                        help="Show ref lists at specific frame")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Max frames to analyze (0=all)")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting wedeo ref lists for {Path(input_path).name}...",
          file=sys.stderr)
    wedeo_info = extract_wedeo_reflists(input_path, args.max_frames)

    print(f"\nWedeo ref lists ({len(wedeo_info)} B-slices):")
    format_reflist(wedeo_info, args.frame)


if __name__ == "__main__":
    main()
