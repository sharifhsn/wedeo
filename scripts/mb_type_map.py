#!/usr/bin/env python3
"""Dump a visual grid of macroblock types for a specific frame.

Shows each MB's type (I_PCM, I_4x4, I_16x16, P_SKIP, P_16x16, B_Direct, etc.)
as a compact grid. Helpful for understanding frame structure and identifying
which MBs use which code paths.

Usage:
    # Show MB type grid for frame 0
    python3 scripts/mb_type_map.py fate-suite/h264-conformance/CVPCMNL1_SVA_C.264 --frame 0

    # Show MB types with diff overlay (mark MBs that differ from FFmpeg)
    python3 scripts/mb_type_map.py file.264 --frame 0 --diff

Requires:
    - wedeo debug binary with tracing feature
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from ffmpeg_debug import find_wedeo_binary, strip_ansi

# ---------------------------------------------------------------------------
# MB type abbreviations
# ---------------------------------------------------------------------------

# I-slice mb_type names: 0=I_4x4, 1-24=I_16x16 variants, 25=I_PCM
I_MB_NAMES = {0: "I4", 25: "PCM"}
for i in range(1, 25):
    I_MB_NAMES[i] = "I16"

# P-slice mb_type names
P_MB_NAMES = {
    0: "P16", 1: "P16x8", 2: "P8x16", 3: "P8x8", 4: "P8x8r",
}
# P_SKIP is signalled separately (mb_type u32::MAX in trace)

# B-slice mb_type names
B_MB_NAMES = {
    0: "BDir", 1: "BL0", 2: "BL1", 3: "BBi",
    4: "B0016x8", 5: "B008x16", 6: "B1116x8", 7: "B118x16",
    8: "B0116x8", 9: "B018x16", 10: "B1016x8", 11: "B108x16",
    12: "B0B16x8", 13: "B0B8x16", 14: "B1B16x8", 15: "B1B8x16",
    16: "BB016x8", 17: "BB08x16", 18: "BB116x8", 19: "BB18x16",
    20: "BBB16x8", 21: "BBB8x16", 22: "B8x8",
}


def mb_type_abbrev(raw_type: int, slice_type: str, is_skip: bool = False) -> str:
    """Get a short abbreviation for an MB type."""
    if is_skip:
        return "SKIP" if slice_type in ("P", "SP") else "BSKP"

    if slice_type in ("I", "SI"):
        return I_MB_NAMES.get(raw_type, f"I?{raw_type}")

    if slice_type in ("P", "SP"):
        if raw_type >= 5:
            # Intra in P-slice (offset by 5)
            return I_MB_NAMES.get(raw_type - 5, f"I?{raw_type-5}")
        return P_MB_NAMES.get(raw_type, f"P?{raw_type}")

    if slice_type == "B":
        if raw_type >= 23:
            # Intra in B-slice (offset by 23)
            return I_MB_NAMES.get(raw_type - 23, f"I?{raw_type-23}")
        return B_MB_NAMES.get(raw_type, f"B?{raw_type}")

    return f"?{raw_type}"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_mb_types(
    input_path: str, target_frame: int,
) -> tuple[int, int, str, list[tuple[int, int, str]]]:
    """Extract MB types for a specific frame.

    Returns (mb_width, mb_height, slice_type, [(mb_x, mb_y, type_abbrev), ...])
    """
    wedeo_bin = find_wedeo_binary(prefer_debug=True, features=["tracing"])
    env = {
        **os.environ,
        "RUST_LOG": (
            "wedeo_codec_h264::decoder=debug,"
            "wedeo_codec_h264::mb=trace,"
            "wedeo_codec_h264::cavlc=trace"
        ),
        "WEDEO_NO_DEBLOCK": "1",
    }
    result = subprocess.run(
        [str(wedeo_bin), input_path],
        capture_output=True, env=env, timeout=120,
    )
    trace = strip_ansi(result.stderr.decode("utf-8", errors="replace"))

    # Collect MBs per frame, delimited by "frame complete" lines.
    # Multi-slice frames have multiple "slice start" but only one
    # "frame complete", so this correctly groups them.
    current_type = "?"
    mb_width = 0
    mb_height = 0
    mbs: list[tuple[int, int, str]] = []
    frames: list[tuple[str, list[tuple[int, int, str]]]] = []
    current_mbs: list[tuple[int, int, str]] = []
    # Track mb_width from SPS for SKIP position computation
    sps_mb_width = 0

    for line in trace.splitlines():
        if "slice start" in line:
            m = re.search(r"slice_type=(\w+)", line)
            if m:
                current_type = m.group(1)

        elif "SPS parsed" in line:
            m = re.search(r"width=(\d+)", line)
            if m:
                sps_mb_width = int(m.group(1)) // 16

        elif "frame complete" in line:
            frames.append((current_type, list(current_mbs)))
            current_mbs = []
            if len(frames) > target_frame:
                break

        elif "decoded MB" in line:
            m_x = re.search(r"mb_x=(\d+)", line)
            m_y = re.search(r"mb_y=(\d+)", line)
            m_type = re.search(r"mb_type=(\d+)", line)
            m_pcm = re.search(r"is_pcm=(true)", line)
            if m_x and m_y and m_type:
                x = int(m_x.group(1))
                y = int(m_y.group(1))
                raw = int(m_type.group(1))
                is_pcm = m_pcm is not None
                abbrev = "PCM" if is_pcm else mb_type_abbrev(raw, current_type)
                current_mbs.append((x, y, abbrev))

        elif "mb_skip_run" in line and sps_mb_width > 0:
            # Parse skip runs to show SKIP MBs in the grid.
            # Trace format: mb_addr=N mb_skip_run=M
            m_addr = re.search(r"mb_addr=(\d+)", line)
            m_run = re.search(r"mb_skip_run=(\d+)", line)
            if m_addr and m_run:
                addr = int(m_addr.group(1))
                run = int(m_run.group(1))
                skip_label = "SKIP" if current_type in ("P", "SP") else "BSKP"
                for i in range(run):
                    sx = (addr + i) % sps_mb_width
                    sy = (addr + i) // sps_mb_width
                    current_mbs.append((sx, sy, skip_label))

    if target_frame < len(frames):
        current_type, mbs = frames[target_frame]
        for x, y, _ in mbs:
            mb_width = max(mb_width, x + 1)
            mb_height = max(mb_height, y + 1)
    else:
        print(f"Frame {target_frame} not found (only {len(frames)} frames decoded)",
              file=sys.stderr)

    return mb_width, mb_height, current_type, mbs


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_mb_grid(
    mb_width: int, mb_height: int, slice_type: str,
    mbs: list[tuple[int, int, str]],
    diff_mbs: set[tuple[int, int]] | None = None,
) -> None:
    """Print a compact grid of MB types."""
    # Build grid
    grid = [["    " for _ in range(mb_width)] for _ in range(mb_height)]
    for x, y, abbrev in mbs:
        if y < mb_height and x < mb_width:
            grid[y][x] = abbrev[:4].ljust(4)

    # Print header
    print(f"MB type map ({mb_width}x{mb_height}, {slice_type}-slice):")
    print(f"     {''.join(f'{x:4d}' for x in range(min(mb_width, 30)))}")
    print(f"     {''.join('----' for _ in range(min(mb_width, 30)))}")

    for y in range(mb_height):
        row_str = "".join(grid[y][:30])
        # Highlight differing MBs
        if diff_mbs:
            highlighted = []
            for x in range(min(mb_width, 30)):
                cell = grid[y][x]
                if (x, y) in diff_mbs:
                    highlighted.append(f"\033[31m{cell}\033[0m")  # red
                else:
                    highlighted.append(cell)
            row_str = "".join(highlighted)
        print(f"  {y:2d}: {row_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dump MB type grid for an H.264 frame",
    )
    parser.add_argument("input", help="H.264 file")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to dump (default: 0)")
    parser.add_argument("--diff", action="store_true",
                        help="Overlay diff markers for MBs that differ from FFmpeg")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting MB types for frame {args.frame}...", file=sys.stderr)
    mb_width, mb_height, slice_type, mbs = extract_mb_types(input_path, args.frame)

    if not mbs:
        print("No MBs found for this frame.")
        sys.exit(1)

    # Count types
    type_counts: dict[str, int] = {}
    for _, _, abbrev in mbs:
        type_counts[abbrev] = type_counts.get(abbrev, 0) + 1

    diff_mbs = None
    if args.diff:
        # Run mb_compare to find differing MBs
        try:
            result = subprocess.run(
                [sys.executable, "scripts/mb_compare.py", input_path,
                 "--max-frames", "1", "--start-frame", str(args.frame)],
                capture_output=True, text=True, timeout=60,
            )
            diff_mbs = set()
            for line in result.stdout.splitlines():
                m = re.match(r"\s+MB\((\d+),(\d+)\)", line)
                if m:
                    diff_mbs.add((int(m.group(1)), int(m.group(2))))
        except Exception:
            pass

    print_mb_grid(mb_width, mb_height, slice_type, mbs, diff_mbs)
    print(f"\nType counts: {dict(sorted(type_counts.items()))}")
    if diff_mbs is not None:
        print(f"Differing MBs: {len(diff_mbs)}/{len(mbs)}")


if __name__ == "__main__":
    main()
