#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Dump SPS flags for H.264 conformance files.

Shows frame_mbs_only_flag, mb_aff, profile, and dimensions for one or more
H.264 files. Useful for quickly classifying files as progressive, MBAFF, or
PAFF before starting conformance investigations.

Usage:
    python3 scripts/sps_flags.py fate-suite/h264-conformance/CAMA1_Sony_C.jsv
    python3 scripts/sps_flags.py fate-suite/h264-conformance/FRext/*.264
    python3 scripts/sps_flags.py --fate-dir fate-suite/h264-conformance
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary


def get_sps_flags(filepath: Path, wedeo_bin: str) -> dict | None:
    """Extract SPS flags from a file using wedeo's debug output."""
    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::decoder=debug"}
    try:
        result = subprocess.run(
            [wedeo_bin, str(filepath)],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    combined = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)
    # Parse SPS line: "SPS parsed sps_id=0 width=352 height=288 frame_mbs_only=false mb_aff=true"
    for line in combined.split("\n"):
        if "SPS parsed" not in line:
            continue
        parts = {}
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k] = v
        if "width" in parts:
            return parts
    return None


def classify(flags: dict) -> str:
    """Classify stream as progressive, MBAFF, or PAFF."""
    fmo = flags.get("frame_mbs_only", "true")
    mbaff = flags.get("mb_aff", "false")
    if fmo == "true":
        return "progressive"
    if mbaff == "true":
        return "MBAFF"
    return "PAFF"


def main():
    parser = argparse.ArgumentParser(description="Dump SPS flags for H.264 files")
    parser.add_argument("files", nargs="*", help="H.264 files to check")
    parser.add_argument(
        "--fate-dir",
        help="Check all files in this directory",
    )
    args = parser.parse_args()

    wedeo_bin = find_wedeo_binary()

    files = []
    if args.fate_dir:
        fate = Path(args.fate_dir)
        files = sorted(fate.glob("*"))
    if args.files:
        files = [Path(f) for f in args.files]

    if not files:
        print("No files specified. Use positional args or --fate-dir.", file=sys.stderr)
        sys.exit(1)

    for f in files:
        if not f.is_file():
            continue
        flags = get_sps_flags(f, wedeo_bin)
        if flags is None:
            print(f"  SKIP  {f.name} (no SPS)")
            continue
        cat = classify(flags)
        w = flags.get("width", "?")
        h = flags.get("height", "?")
        print(f"  {cat:12s}  {f.name}  ({w}x{h})")


if __name__ == "__main__":
    main()
