#!/usr/bin/env python3
"""Categorize H.264 conformance DIFF files by failure type.

For each DIFF file, determines whether the issue is:
- ORDERING: same CRCs in different order (output reordering bug)
- PIXEL: different CRCs (decode/MC/residual/deblock bug)
- CHROMA_ONLY: luma matches but chroma differs (chroma-specific bug)
- FRAME_COUNT: different number of frames produced

Usage:
    python3 scripts/categorize_diffs.py [--only FILE_PREFIX]
"""

import argparse
import os
import subprocess
import sys


FATE_DIR = os.environ.get("FATE_SUITE", "./fate-suite")
H264_DIR = os.path.join(FATE_DIR, "h264-conformance")
WEDEO_BIN = "./target/release/wedeo-framecrc"


def get_framecrc(cmd):
    """Run a command and extract (pts, crc) pairs from framecrc output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = []
        for line in result.stdout.splitlines():
            if line.startswith("0,"):
                parts = line.split(",")
                pts = parts[1].strip()
                crc = parts[-1].strip()
                lines.append((pts, crc))
        return lines
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def get_mb_compare(filepath, start_frame=0, max_frames=5):
    """Run mb_compare to check luma vs chroma diffs."""
    try:
        cmd = ["python3", "scripts/mb_compare.py", filepath,
               "--max-frames", str(max_frames)]
        if start_frame > 0:
            cmd.extend(["--start-frame", str(start_frame)])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        has_luma_match_chroma_diff = False
        has_luma_diff = False
        for line in result.stdout.splitlines():
            if "luma MATCH, chroma diff" in line:
                has_luma_match_chroma_diff = True
            elif "MBs differ" in line and "luma MATCH" not in line:
                has_luma_diff = True
        return has_luma_match_chroma_diff, has_luma_diff
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, False


def categorize_file(filepath):
    """Categorize a single file's diff type."""
    basename = os.path.basename(filepath)

    ffmpeg_cmd = [
        "ffmpeg", "-bitexact", "-i", filepath,
        "-f", "framecrc", "-"
    ]
    wedeo_cmd = [WEDEO_BIN, filepath]

    ff_frames = get_framecrc(ffmpeg_cmd)
    we_frames = get_framecrc(wedeo_cmd)

    if not ff_frames and not we_frames:
        return "EMPTY", 0, 0, ""

    if len(ff_frames) != len(we_frames):
        return "FRAME_COUNT", len(ff_frames), len(we_frames), \
            f"FFmpeg={len(ff_frames)} wedeo={len(we_frames)}"

    # Compare CRCs line-by-line
    matching = sum(1 for (_, c1), (_, c2) in zip(ff_frames, we_frames) if c1 == c2)
    total = len(ff_frames)

    if matching == total:
        return "BITEXACT", matching, total, ""

    # Check if all CRCs exist in both (just reordered)
    ff_crcs = sorted(c for _, c in ff_frames)
    we_crcs = sorted(c for _, c in we_frames)
    if ff_crcs == we_crcs:
        return "ORDERING", matching, total, "same CRCs, different order"

    # Find first diverging frame
    first_diff = next(
        (i for i, ((_, c1), (_, c2)) in enumerate(zip(ff_frames, we_frames)) if c1 != c2),
        -1
    )

    # Check luma vs chroma with mb_compare, starting at the first diff
    start = max(0, first_diff)
    chroma_only, has_luma = get_mb_compare(filepath, start_frame=start, max_frames=5)
    if chroma_only and not has_luma:
        return "CHROMA_ONLY", matching, total, \
            f"luma matches, chroma diffs (from frame {first_diff})"

    detail = f"first diff at frame {first_diff}"
    if chroma_only:
        detail += " (chroma-only in some frames)"

    return "PIXEL", matching, total, detail


def main():
    parser = argparse.ArgumentParser(description="Categorize H.264 DIFF files")
    parser.add_argument("--only", help="Only check files matching this prefix")
    args = parser.parse_args()

    if not os.path.exists(WEDEO_BIN):
        print(f"Error: {WEDEO_BIN} not found. Run: cargo build --release -p wedeo-fate",
              file=sys.stderr)
        sys.exit(1)

    # Get list of DIFF files from conformance report
    result = subprocess.run(
        ["python3", "scripts/conformance_report.py",
         "--cavlc-only", "--progressive-only", "--only-failing"],
        capture_output=True, text=True, timeout=300
    )

    diff_files = []
    in_diff = False
    for line in result.stdout.splitlines():
        if "=== DIFF" in line:
            in_diff = True
            continue
        if in_diff and line.startswith("  "):
            name = line.strip().split(":")[0]
            if args.only and not name.startswith(args.only):
                continue
            filepath = os.path.join(H264_DIR, name)
            if os.path.exists(filepath):
                diff_files.append((name, filepath))

    if not diff_files:
        print("No DIFF files found.")
        return

    # Categorize each file
    categories = {}
    for name, filepath in sorted(diff_files):
        cat, matching, total, detail = categorize_file(filepath)
        categories.setdefault(cat, []).append((name, matching, total, detail))
        status = f"{matching}/{total}" if total else "N/A"
        print(f"  {cat:12s}  {status:>8s}  {name}  {detail}")

    # Summary
    print()
    for cat in ["CHROMA_ONLY", "ORDERING", "PIXEL", "FRAME_COUNT", "EMPTY"]:
        if cat in categories:
            names = [n for n, _, _, _ in categories[cat]]
            print(f"{cat} ({len(names)}): {', '.join(names)}")


if __name__ == "__main__":
    main()
