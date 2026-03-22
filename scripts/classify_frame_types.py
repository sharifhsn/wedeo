#!/usr/bin/env python3
"""Classify H.264 conformance files by frame type (I/P/B) distribution.

Annotates conformance results with per-file frame type counts to quickly
identify which decode path is broken (e.g., "all I-only pass, I+P fail").

Usage:
    python3 scripts/classify_frame_types.py                  # CAVLC files
    python3 scripts/classify_frame_types.py --cabac          # CABAC files
    python3 scripts/classify_frame_types.py --diff-only      # only show DIFF files
    python3 scripts/classify_frame_types.py FILE1 FILE2 ...  # specific files
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FATE_DIR_DEFAULT = Path("fate-suite/h264-conformance")

# Import file lists from conformance_full.py.
# Only need the list constants, but conformance_full.py has transitive imports
# that may fail. Extract the lists by parsing the source if import fails.
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from conformance_full import PROGRESSIVE_CABAC_FILES, PROGRESSIVE_CAVLC_FILES
except Exception:
    # Fallback: parse the list constants from the source file directly
    _src = (SCRIPT_DIR / "conformance_full.py").read_text()
    import ast

    def _extract_list(src: str, name: str) -> list[str]:
        for line_idx, line in enumerate(src.splitlines()):
            if line.startswith(f"{name} =") or line.startswith(f"{name}="):
                # Find the matching bracket
                bracket_start = src.index("[", src.index(name))
                depth, i = 0, bracket_start
                while i < len(src):
                    if src[i] == "[":
                        depth += 1
                    elif src[i] == "]":
                        depth -= 1
                        if depth == 0:
                            return ast.literal_eval(src[bracket_start : i + 1])
                    i += 1
        return []

    PROGRESSIVE_CAVLC_FILES = _extract_list(_src, "PROGRESSIVE_CAVLC_FILES")
    PROGRESSIVE_CABAC_FILES = _extract_list(_src, "PROGRESSIVE_CABAC_FILES")


_ffmpeg_warned = False


def get_frame_types(filepath: Path) -> dict[str, int] | None:
    """Get I/P/B frame type counts using FFmpeg showinfo filter.

    Returns None if FFmpeg is not available (prints warning once).
    """
    global _ffmpeg_warned
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(filepath), "-vf", "showinfo", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stderr
    except FileNotFoundError:
        if not _ffmpeg_warned:
            print("WARNING: ffmpeg not found on PATH, frame types unavailable",
                  file=sys.stderr)
            _ffmpeg_warned = True
        return None
    except subprocess.TimeoutExpired:
        return None

    counts: dict[str, int] = {"I": 0, "P": 0, "B": 0}
    for m in re.finditer(r"type:([IPB])", output):
        counts[m.group(1)] += 1
    return counts


def get_conformance_status(filepath: Path) -> str:
    """Run wedeo+ffmpeg framecrc comparison and return status."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "framecrc_compare.py"),
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        line = result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""
        if "BITEXACT" in line:
            return "BITEXACT"
        m = re.search(r"(\d+)/(\d+) match", line)
        if m:
            return f"DIFF {m.group(1)}/{m.group(2)}"
        return "DIFF"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "ERROR"


def main():
    parser = argparse.ArgumentParser(
        description="Classify H.264 conformance files by frame type distribution"
    )
    parser.add_argument("files", nargs="*", help="Specific files to check")
    parser.add_argument(
        "--cabac",
        action="store_true",
        help="Use CABAC file list instead of CAVLC",
    )
    parser.add_argument(
        "--diff-only",
        action="store_true",
        help="Only show DIFF files (skip BITEXACT)",
    )
    parser.add_argument(
        "--fate-dir",
        type=Path,
        default=FATE_DIR_DEFAULT,
        help=f"Path to conformance directory (default: {FATE_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--no-status",
        action="store_true",
        help="Skip conformance check (just show frame types)",
    )
    args = parser.parse_args()

    if args.diff_only and args.no_status:
        print("Warning: --diff-only has no effect with --no-status (status needed to filter)",
              file=sys.stderr)

    if args.files:
        filepaths = [Path(f) for f in args.files]
    else:
        file_list = PROGRESSIVE_CABAC_FILES if args.cabac else PROGRESSIVE_CAVLC_FILES
        if not file_list:
            print("Error: could not import file lists from conformance_full.py")
            sys.exit(1)
        filepaths = [args.fate_dir / f for f in file_list]

    # Header
    print(f"{'File':<40s}  {'Status':<16s}  {'I':>4s} {'P':>4s} {'B':>4s}  {'Pattern'}")
    print("-" * 90)

    for fpath in filepaths:
        if not fpath.exists():
            continue

        fname = fpath.name

        # Get conformance status
        if args.no_status:
            status = "N/A"
        else:
            status = get_conformance_status(fpath)

        if args.diff_only and status == "BITEXACT":
            continue

        # Get frame types
        types = get_frame_types(fpath)
        if types is None:
            i_count = p_count = b_count = -1
        else:
            i_count = types.get("I", 0)
            p_count = types.get("P", 0)
            b_count = types.get("B", 0)

        # Derive pattern
        if i_count < 0:
            pattern = "?"
        elif b_count > 0:
            pattern = "I+P+B"
        elif p_count > 0:
            pattern = "I+P"
        else:
            pattern = "I-only"

        print(
            f"{fname:<40s}  {status:<16s}  {i_count:4d} {p_count:4d} {b_count:4d}  {pattern}"
        )


if __name__ == "__main__":
    main()
