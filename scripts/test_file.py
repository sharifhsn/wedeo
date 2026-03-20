#!/usr/bin/env python3
"""Quick build-and-test for a single H.264 conformance file.

Rebuilds the release binary if source is newer, then runs framecrc_compare.
Shorthand for the repeated cargo build + framecrc_compare pattern.

Usage:
    python3 scripts/test_file.py MR4            # fuzzy match in fate-suite/
    python3 scripts/test_file.py CVBS3 --no-deblock
    python3 scripts/test_file.py CVWP5 --pixel-detail
    python3 scripts/test_file.py --all           # all conformance files
    python3 scripts/test_file.py --diff          # only non-BITEXACT files

Requires:
    - cargo (for rebuilds)
    - ffmpeg binary in PATH
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary, resolve_conformance_file


def main():
    parser = argparse.ArgumentParser(description="Build and test H.264 file")
    parser.add_argument("input", nargs="?", help="File name or partial match")
    parser.add_argument("--all", action="store_true",
                        help="Test all conformance files (via classify_diffs)")
    parser.add_argument("--diff", action="store_true",
                        help="Show only DIFF files (via classify_diffs)")
    parser.add_argument("--no-deblock", action="store_true",
                        help="Disable deblocking in both decoders")
    parser.add_argument("--pixel-detail", action="store_true",
                        help="Show per-plane pixel diffs")
    args = parser.parse_args()

    # Ensure binary is fresh (find_wedeo_binary auto-rebuilds)
    find_wedeo_binary()

    scripts_dir = Path(__file__).resolve().parent

    if args.all or args.diff:
        cmd = [sys.executable, str(scripts_dir / "classify_diffs.py")]
        if not args.diff:
            cmd.append("--show-bitexact")
        sys.exit(subprocess.call(cmd))

    if not args.input:
        parser.print_help()
        sys.exit(1)

    input_path = resolve_conformance_file(args.input)
    cmd = [sys.executable, str(scripts_dir / "framecrc_compare.py"), str(input_path)]
    if args.no_deblock:
        cmd.append("--no-deblock")
    if args.pixel_detail:
        cmd.append("--pixel-detail")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
