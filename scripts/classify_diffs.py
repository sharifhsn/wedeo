#!/usr/bin/env python3
"""Classify H.264 DIFF files by error type: reorder vs pixel-diff.

Cross-matches CRCs between wedeo and FFmpeg to distinguish:
- Same-position matches (frames at the same output index have identical CRCs)
- Reordered frames (CRC exists in the other decoder but at a different index)
- Pixel-diff frames (CRC doesn't exist in the other decoder at all)

This should be the FIRST tool run when investigating conformance gaps.
Different error types need completely different debugging approaches:
- Reorder issues: fix output ordering logic (won't make files BITEXACT alone)
- Pixel diffs: fix MC/prediction/deblock bugs (can make files BITEXACT)

Note: CRC cross-matching uses Adler-32 which has a small collision risk.
A frame classified as "reorder" could theoretically be a pixel-diff whose
CRC happens to match an unrelated FFmpeg frame. This is unlikely but not
impossible for static scenes.

Usage:
    python3 scripts/classify_diffs.py                    # all non-BITEXACT files
    python3 scripts/classify_diffs.py CVBS3_Sony_C.jsv   # specific file
    python3 scripts/classify_diffs.py --show-bitexact     # include BITEXACT in output

Requires:
    - wedeo-framecrc binary (release build)
    - ffmpeg binary in PATH
"""

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import (
    CONFORMANCE_DIR,
    CONFORMANCE_EXTENSIONS,
    find_wedeo_binary,
    run_framecrc,
)


class ClassifyResult(NamedTuple):
    same: int
    reorder: int
    pixel_diff: int
    total: int
    wedeo_frames: int
    ffmpeg_frames: int
    reorder_examples: list  # list of (wedeo_idx, ffmpeg_idx) tuples


def classify_file(input_path, wedeo_bin):
    """Classify a single file's diffs."""
    w_crcs = run_framecrc([str(wedeo_bin), str(input_path)])
    ff_crcs = run_framecrc(
        ["ffmpeg", "-bitexact", "-i", str(input_path), "-f", "framecrc", "-"]
    )

    total = min(len(w_crcs), len(ff_crcs))
    if total == 0:
        return ClassifyResult(0, 0, 0, 0, len(w_crcs), len(ff_crcs), [])

    # Build lookup: FFmpeg CRC → set of unmatched positions.
    # Positions are consumed when matched to avoid double-counting
    # (mitigates the CRC collision false-positive risk).
    ff_map: dict[str, list[int]] = {}
    for i, crc in enumerate(ff_crcs):
        ff_map.setdefault(crc, []).append(i)

    same = 0
    reorder = 0
    pixel_diff = 0
    reorder_examples = []

    for i in range(total):
        w_crc = w_crcs[i]
        if i < len(ff_crcs) and w_crc == ff_crcs[i]:
            same += 1
            # Consume this position from the map
            if w_crc in ff_map and i in ff_map[w_crc]:
                ff_map[w_crc].remove(i)
        elif w_crc in ff_map and ff_map[w_crc]:
            reorder += 1
            matched_pos = ff_map[w_crc].pop(0)
            if len(reorder_examples) < 3:
                reorder_examples.append((i, matched_pos))
        else:
            pixel_diff += 1

    return ClassifyResult(
        same, reorder, pixel_diff, total,
        len(w_crcs), len(ff_crcs), reorder_examples,
    )


def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    parser = argparse.ArgumentParser(description="Classify DIFF files by error type")
    parser.add_argument(
        "files", nargs="*",
        help="Specific files to check (default: all conformance files)"
    )
    parser.add_argument(
        "--show-bitexact", action="store_true",
        help="Include BITEXACT files in output"
    )
    args = parser.parse_args()

    wedeo_bin = find_wedeo_binary(auto_rebuild=True)

    if args.files:
        files = []
        for f in args.files:
            p = Path(f)
            if not p.exists():
                p = CONFORMANCE_DIR / f
            if p.exists():
                files.append(p)
            else:
                print(f"Warning: {f} not found", file=sys.stderr)
    else:
        if not CONFORMANCE_DIR.exists():
            print(f"Error: {CONFORMANCE_DIR} not found", file=sys.stderr)
            sys.exit(1)
        files = sorted(
            p for p in CONFORMANCE_DIR.iterdir()
            if p.suffix in CONFORMANCE_EXTENSIONS
        )

    print(f"{'File':<35} {'Match':>7} {'Reorder':>8} {'PixDiff':>8} {'Total':>6}  Notes")
    print("-" * 90)

    summary = {"bitexact": 0, "reorder_only": 0, "pixel_only": 0, "mixed": 0}

    for f in files:
        r = classify_file(f, wedeo_bin)

        if r.total == 0:
            continue

        name = f.name
        notes = []
        if r.same == r.total and r.reorder == 0 and r.pixel_diff == 0:
            tag = "BITEXACT"
            summary["bitexact"] += 1
        elif r.pixel_diff == 0 and r.reorder > 0:
            tag = "REORDER"
            summary["reorder_only"] += 1
            for wi, fi in r.reorder_examples:
                notes.append(f"W[{wi}]->FF[{fi}]")
        elif r.reorder == 0 and r.pixel_diff > 0:
            tag = "PIXDIFF"
            summary["pixel_only"] += 1
        else:
            tag = "MIXED"
            summary["mixed"] += 1

        if r.wedeo_frames != r.ffmpeg_frames:
            notes.append(f"W={r.wedeo_frames} FF={r.ffmpeg_frames}")

        note_str = " ".join(notes)

        if args.show_bitexact or tag != "BITEXACT":
            print(
                f"{name:<35} {r.same:>5}/{r.total:<3} {r.reorder:>8} {r.pixel_diff:>8} {r.total:>6}  {tag} {note_str}"
            )

    print()
    print(
        f"Summary: {summary['bitexact']} BITEXACT, "
        f"{summary['reorder_only']} reorder-only, "
        f"{summary['pixel_only']} pixel-diff-only, "
        f"{summary['mixed']} mixed"
    )


if __name__ == "__main__":
    main()
