#!/usr/bin/env python3
"""Compare POC output order between wedeo and FFmpeg for H.264 files.

Shows side-by-side output positions with CRC matching and cross-mapping
for reordered frames. Useful for debugging output reordering issues.

Usage:
    python3 scripts/poc_output_compare.py fate-suite/h264-conformance/MR4_TANDBERG_C.264
    python3 scripts/poc_output_compare.py MR4  # fuzzy match in fate-suite/h264-conformance/

Requires:
    - wedeo-framecrc binary (release build)
    - ffmpeg binary in PATH
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary, resolve_conformance_file, run_framecrc


def run_comparison(input_path: Path, wedeo_bin: Path, context: int = 3) -> None:
    """Run both decoders and show side-by-side output comparison."""
    wedeo_crcs = run_framecrc([str(wedeo_bin), str(input_path)])
    ffmpeg_cmd = ["ffmpeg", "-bitexact", "-i", str(input_path), "-f", "framecrc", "-"]
    ffmpeg_crcs = run_framecrc(ffmpeg_cmd)

    total = max(len(wedeo_crcs), len(ffmpeg_crcs))
    if total == 0:
        print(f"No frames decoded from {input_path.name}")
        return

    # Build CRC-to-position map for cross-matching reordered frames
    ff_crc_to_pos: dict[str, list[int]] = {}
    for i, crc in enumerate(ffmpeg_crcs):
        ff_crc_to_pos.setdefault(crc, []).append(i)

    # Classify each position
    matches = 0
    reorders = 0
    pixel_diffs = 0
    rows: list[tuple[int, str, str, str]] = []  # (pos, w_crc, ff_crc, status)

    for i in range(total):
        w_crc = wedeo_crcs[i] if i < len(wedeo_crcs) else "---"
        ff_crc = ffmpeg_crcs[i] if i < len(ffmpeg_crcs) else "---"

        if w_crc == ff_crc:
            rows.append((i, w_crc, ff_crc, "MATCH"))
            matches += 1
        elif w_crc in ff_crc_to_pos:
            ff_positions = ff_crc_to_pos[w_crc]
            rows.append((i, w_crc, ff_crc, f"REORDER -> FF[{ff_positions[0]}]"))
            reorders += 1
        else:
            rows.append((i, w_crc, ff_crc, "PIXDIFF"))
            pixel_diffs += 1

    # Print header
    name = input_path.name
    print(f"\n{'='*80}")
    print(f"  {name}: {len(wedeo_crcs)}W / {len(ffmpeg_crcs)}FF frames")
    print(f"  {matches} match, {reorders} reorder, {pixel_diffs} pixdiff")
    print(f"{'='*80}")

    if matches == total:
        print(f"  BITEXACT ({total} frames)")
        return

    # Show diverging positions with context, collapsing long matching runs
    print(f"\n  {'Pos':>4}  {'Wedeo CRC':>12}  {'FFmpeg CRC':>12}  Status")
    print(f"  {'---':>4}  {'----------':>12}  {'----------':>12}  ------")

    diff_positions = {i for i, (_, _, _, s) in enumerate(rows) if s != "MATCH"}
    shown = set()

    for diff_pos in sorted(diff_positions):
        start = max(0, diff_pos - context)
        end = min(total, diff_pos + context + 1)
        for j in range(start, end):
            if j in shown:
                continue
            shown.add(j)
            pos, w_crc, ff_crc, status = rows[j]
            marker = "  " if status == "MATCH" else ">>"
            print(f"{marker}{pos:>4}  {w_crc:>12}  {ff_crc:>12}  {status}")

        # Show ellipsis for long matching gaps
        next_diffs = [d for d in sorted(diff_positions) if d > diff_pos]
        if next_diffs:
            gap = next_diffs[0] - end
            if gap > 2 * context:
                print(f"  ... ({gap} matching frames) ...")

    # Summary of reorder mapping
    if reorders > 0:
        print(f"\nReorder mapping (first 20):")
        count = 0
        for _, (pos, w_crc, _, status) in enumerate(rows):
            if status.startswith("REORDER") and count < 20:
                ff_pos = ff_crc_to_pos[w_crc][0]
                delta = ff_pos - pos
                print(f"  W[{pos}] -> FF[{ff_pos}]  (delta={delta:+d})")
                count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Compare POC output order between wedeo and FFmpeg"
    )
    parser.add_argument("input", help="H.264 file or partial name")
    parser.add_argument(
        "--context", "-C", type=int, default=3,
        help="Lines of context around diffs (default: 3)"
    )
    args = parser.parse_args()

    wedeo_bin = find_wedeo_binary()
    input_path = resolve_conformance_file(args.input)
    run_comparison(input_path, wedeo_bin, context=args.context)


if __name__ == "__main__":
    main()
