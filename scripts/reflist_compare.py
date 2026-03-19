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
    find_ffmpeg_binary,
    find_wedeo_binary,
    run_lldb,
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
    decode_idx = -1
    current_fn = 0
    current_type = "?"
    current_poc = 0

    for line in trace.splitlines():
        if "slice start" in line:
            decode_idx += 1
            m = re.search(r"slice_type=(\w+)", line)
            if m:
                current_type = m.group(1)
            m = re.search(r"frame_num=(\d+)", line)
            if m:
                current_fn = int(m.group(1))
            m = re.search(r"is_idr=(true|false)", line)

        elif "frame complete" in line:
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

            slices.append(SliceRefInfo(
                decode_idx=decode_idx,
                frame_num=current_fn,
                poc=current_poc,
                slice_type=current_type,
                l0=[RefListEntry(poc=p) for p in l0_pocs],
                l1=[RefListEntry(poc=p) for p in l1_pocs],
            ))

            if max_frames and len(slices) >= max_frames:
                break

        # For P-frames, we don't currently log ref lists.
        # TODO: Add ref list logging for P-frames in decoder.rs

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
# FFmpeg extraction (via lldb)
# ---------------------------------------------------------------------------

def extract_ffmpeg_reflist_at_frame(
    input_path: str, frame_idx: int,
) -> SliceRefInfo | None:
    """Extract FFmpeg ref list at a specific frame via lldb.

    Breaks at ff_h264_build_ref_list and reads ref_list[0] and ref_list[1].
    """
    ffmpeg_bin = find_ffmpeg_binary()

    lldb_script = f"""
breakpoint set -n ff_h264_fill_mbaff_ref_list
breakpoint modify 1 -i {frame_idx}
run -bitexact -i {input_path} -f null -

# Read ref counts
expr sl->ref_count[0]
expr sl->ref_count[1]

# Read L0 refs (frame_num and poc for first 4)
expr sl->ref_list[0][0].parent->poc
expr sl->ref_list[0][0].parent->frame_num
expr sl->ref_list[0][1].parent->poc
expr sl->ref_list[0][1].parent->frame_num
expr sl->ref_list[0][2].parent->poc
expr sl->ref_list[0][2].parent->frame_num
expr sl->ref_list[0][3].parent->poc
expr sl->ref_list[0][3].parent->frame_num

# Read L1 refs (first 2)
expr sl->ref_list[1][0].parent->poc
expr sl->ref_list[1][0].parent->frame_num
expr sl->ref_list[1][1].parent->poc
expr sl->ref_list[1][1].parent->frame_num

quit
"""

    try:
        raw = run_lldb(str(ffmpeg_bin), lldb_script, timeout=30)
        # Parsing lldb output is fragile; return None for now if parsing fails
        print(f"FFmpeg lldb raw output:\n{raw}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"FFmpeg lldb failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Comparison
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
    parser.add_argument("--ffmpeg", action="store_true",
                        help="Also extract from FFmpeg via lldb")
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

    if args.ffmpeg and args.frame is not None:
        print(f"\nExtracting FFmpeg ref list at frame {args.frame}...",
              file=sys.stderr)
        ffmpeg_info = extract_ffmpeg_reflist_at_frame(input_path, args.frame)
        if ffmpeg_info:
            print(f"\nFFmpeg ref list at frame {args.frame}:")
            format_reflist([ffmpeg_info], args.frame)


if __name__ == "__main__":
    main()
