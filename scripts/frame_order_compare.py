#!/usr/bin/env python3
"""Compare frame output order between wedeo and FFmpeg.

Checks if two decoders produce the same CRCs in the same order.
Distinguishes ORDERING issues (same CRCs, different order) from
PIXEL issues (different CRCs).

Usage:
    python3 scripts/frame_order_compare.py fate-suite/h264-conformance/CVWP2_TOSHIBA_E.264
"""

import argparse
import subprocess
import sys

WEDEO_BIN = "./target/release/wedeo-framecrc"


def get_crcs(cmd):
    """Run command and extract CRCs from framecrc output."""
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=30,
        )
        crcs = []
        for line in result.stdout.splitlines():
            if line.startswith("0,"):
                parts = line.split(",")
                if len(parts) >= 6:
                    crcs.append(parts[-1].strip())
        return crcs
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def main():
    parser = argparse.ArgumentParser(description="Compare frame output order")
    parser.add_argument("input", help="H.264 input file")
    args = parser.parse_args()

    ff_crcs = get_crcs(["ffmpeg", "-bitexact", "-i", args.input, "-f", "framecrc", "-"])
    we_crcs = get_crcs([WEDEO_BIN, args.input])

    print(f"FFmpeg: {len(ff_crcs)} frames, wedeo: {len(we_crcs)} frames")

    if len(ff_crcs) != len(we_crcs):
        print(f"FRAME_COUNT: different number of frames")

    n = min(len(ff_crcs), len(we_crcs))
    matching = sum(1 for i in range(n) if ff_crcs[i] == we_crcs[i])
    print(f"Position-matched: {matching}/{n}")

    if matching == n and len(ff_crcs) == len(we_crcs):
        print("BITEXACT: all frames match in order")
        return

    # Check if it's just reordering
    ff_sorted = sorted(ff_crcs)
    we_sorted = sorted(we_crcs)
    if ff_sorted == we_sorted:
        print("ORDERING: same CRCs in different order")
        # Find the first mismatch
        for i in range(n):
            if ff_crcs[i] != we_crcs[i]:
                # Find where wedeo's CRC appears in FFmpeg
                try:
                    ff_pos = ff_crcs.index(we_crcs[i])
                    we_pos = we_crcs.index(ff_crcs[i])
                except ValueError:
                    ff_pos = we_pos = -1
                print(f"  First mismatch at frame {i}:")
                print(f"    FFmpeg[{i}] = {ff_crcs[i]} → appears at wedeo[{we_pos}]")
                print(f"    wedeo[{i}] = {we_crcs[i]} → appears at FFmpeg[{ff_pos}]")
                break

        # Show cross-match for first 10 frames
        print("\n  Cross-match (first mismatched frames):")
        shown = 0
        for i in range(n):
            if ff_crcs[i] != we_crcs[i] and shown < 10:
                try:
                    ff_pos = ff_crcs.index(we_crcs[i])
                except ValueError:
                    ff_pos = -1
                print(f"    wedeo[{i}] = FFmpeg[{ff_pos}]")
                shown += 1
    else:
        # Different CRCs exist
        ff_set = set(ff_crcs)
        we_set = set(we_crcs)
        only_ff = ff_set - we_set
        only_we = we_set - ff_set
        print(f"PIXEL: {len(only_ff)} CRCs only in FFmpeg, {len(only_we)} only in wedeo")

        # Check if some are just reordered
        common = ff_set & we_set
        if common:
            ff_common_ordered = [c for c in ff_crcs if c in common]
            we_common_ordered = [c for c in we_crcs if c in common]
            reorder_match = sum(
                1 for a, b in zip(ff_common_ordered, we_common_ordered) if a == b
            )
            print(
                f"  Of {len(common)} shared CRCs: "
                f"{reorder_match}/{len(common)} in same position"
            )


if __name__ == "__main__":
    main()
