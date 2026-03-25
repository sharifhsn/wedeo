#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Extract FFmpeg DPB state at a specific POC using -debug mmco.

Shows short_ref and long_ref entries at the point where a given POC
is being decoded. Much faster than lldb for DPB comparison.

Usage:
    python3 scripts/ffmpeg_dpb_at_poc.py file.264 --poc 21
    python3 scripts/ffmpeg_dpb_at_poc.py file.264 --all   # show all DPB snapshots
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ffmpeg_debug import find_ffmpeg_binary, resolve_conformance_file


@dataclass
class DpbSnapshot:
    """DPB state at a specific decode point."""
    short_refs: list[tuple[int, int]]  # (frame_num, poc)
    long_refs: list[tuple[int, int, int]]  # (lt_idx, frame_num, poc)


def extract_dpb_states(input_path: str) -> list[DpbSnapshot]:
    """Extract all DPB snapshots from FFmpeg -debug mmco output."""
    ffmpeg = find_ffmpeg_binary()
    result = subprocess.run(
        [str(ffmpeg), "-debug", "mmco", "-i", input_path, "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stderr

    snapshots: list[DpbSnapshot] = []
    current_short: list[tuple[int, int]] = []
    current_long: list[tuple[int, int, int]] = []
    in_short = False
    in_long = False

    fn_pat = re.compile(r'(\d+)\s+fn:(\d+)\s+poc:(\d+)')

    for line in output.splitlines():
        if "short term list:" in line:
            # If we were already collecting, save the previous snapshot
            if in_long:
                snapshots.append(DpbSnapshot(
                    short_refs=list(current_short),
                    long_refs=list(current_long),
                ))
                current_short.clear()
                current_long.clear()
            in_short = True
            in_long = False
        elif "long term list:" in line:
            in_short = False
            in_long = True
        elif in_short or in_long:
            m = fn_pat.search(line)
            if m:
                idx = int(m.group(1))
                fn = int(m.group(2))
                poc = int(m.group(3))
                if in_short:
                    current_short.append((fn, poc))
                else:
                    current_long.append((idx, fn, poc))

    # Save last snapshot
    if current_short or current_long:
        snapshots.append(DpbSnapshot(
            short_refs=list(current_short),
            long_refs=list(current_long),
        ))

    return snapshots


def poc_offset(snapshots: list[DpbSnapshot]) -> int:
    """Detect FFmpeg's POC offset (typically 65536 for frame mode)."""
    for s in snapshots:
        for _, poc in s.short_refs:
            if poc > 60000:
                return (poc >> 16) << 16
        for _, _, poc in s.long_refs:
            if poc > 60000:
                return (poc >> 16) << 16
    return 0


def print_snapshot(idx: int, s: DpbSnapshot, offset: int):
    st_str = ", ".join(f"fn={fn} poc={poc - offset}" for fn, poc in s.short_refs) or "none"
    lt_str = ", ".join(
        f"[{li}] fn={fn} poc={poc - offset}" for li, fn, poc in s.long_refs
    ) or "none"
    print(f"  snapshot {idx:3d}: ST=[{st_str}]  LT=[{lt_str}]")


def main():
    parser = argparse.ArgumentParser(
        description="Extract FFmpeg DPB state at specific POC",
    )
    parser.add_argument("input", help="H.264 file (path or conformance name)")
    parser.add_argument("--poc", type=int, default=None, help="Target POC to show")
    parser.add_argument("--all", action="store_true", help="Show all snapshots")
    parser.add_argument("--raw", action="store_true",
                        help="Show raw POC values (no offset subtraction)")
    args = parser.parse_args()

    input_path = str(resolve_conformance_file(args.input))
    print(f"Extracting FFmpeg DPB for {Path(input_path).name}...", file=sys.stderr)

    snapshots = extract_dpb_states(input_path)
    offset = 0 if args.raw else poc_offset(snapshots)

    if offset:
        print(f"POC offset: {offset} (subtract from raw values)\n")

    if args.poc is not None:
        target_poc = args.poc + offset
        # Find snapshots containing or near this POC
        found = False
        for i, s in enumerate(snapshots):
            all_pocs = [poc for _, poc in s.short_refs] + [poc for _, _, poc in s.long_refs]
            if target_poc in all_pocs or any(abs(p - target_poc) <= 2 for p in all_pocs):
                print_snapshot(i, s, offset)
                found = True
        if not found:
            print(f"No snapshots found near POC {args.poc}")
    elif args.all:
        for i, s in enumerate(snapshots):
            print_snapshot(i, s, offset)
    else:
        print(f"{len(snapshots)} DPB snapshots extracted.")
        print("Use --poc N to filter or --all to show all.")


if __name__ == "__main__":
    main()
