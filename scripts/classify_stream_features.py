#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Classify H.264 conformance files by SPS/PPS features.

Reports: profile, entropy coding, frame_mbs_only, mb_aff, frame count,
and other key features for each file. Useful for triaging which files
need specific decoder features.

Usage:
    python3 scripts/classify_stream_features.py fate-suite/h264-conformance/CAMA1_Sony_C.jsv
    python3 scripts/classify_stream_features.py --all  # all conformance files
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROFILE_NAMES = {
    66: "Baseline", 77: "Main", 88: "Extended",
    100: "High", 110: "High10", 122: "High422", 244: "High444",
}


def classify_file(path: str) -> dict:
    """Extract stream features from an H.264 file using ffmpeg trace_headers."""
    result = {"file": Path(path).name}

    # Get frame count from framecrc
    try:
        proc = subprocess.run(
            ["ffmpeg", "-bitexact", "-i", path, "-f", "framecrc", "-"],
            capture_output=True, timeout=30,
        )
        lines = proc.stdout.decode(errors="replace").splitlines()
        result["frames"] = sum(1 for l in lines if l.strip() and not l.startswith("#"))
    except Exception:
        result["frames"] = -1

    # Get SPS features from trace_headers
    try:
        proc = subprocess.run(
            ["ffmpeg", "-bsf:v", "trace_headers", "-i", path, "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
        trace = proc.stderr.decode(errors="replace")
    except Exception:
        trace = ""

    features = {
        "profile_idc": None,
        "entropy_coding_mode_flag": None,
        "frame_mbs_only_flag": None,
        "mb_adaptive_frame_field_flag": None,
        "transform_8x8_mode_flag": None,
        "pic_width_in_mbs_minus1": None,
        "pic_height_in_map_units_minus1": None,
    }

    for line in trace.splitlines():
        for key in features:
            if key in line and features[key] is None:
                # Extract the last number on the line (after '= ')
                parts = line.strip().split("=")
                if len(parts) >= 2:
                    try:
                        features[key] = int(parts[-1].strip())
                    except ValueError:
                        pass

    profile = features["profile_idc"]
    result["profile"] = PROFILE_NAMES.get(profile, f"Unknown({profile})") if profile else "?"
    result["entropy"] = "CABAC" if features["entropy_coding_mode_flag"] == 1 else "CAVLC" if features["entropy_coding_mode_flag"] == 0 else "?"
    result["mbs_only"] = features["frame_mbs_only_flag"]
    result["mb_aff"] = features["mb_adaptive_frame_field_flag"]
    result["8x8"] = features["transform_8x8_mode_flag"]

    w = (features["pic_width_in_mbs_minus1"] or 0) + 1
    h = (features["pic_height_in_map_units_minus1"] or 0) + 1
    mult = 1 if features["frame_mbs_only_flag"] != 0 else 2
    result["resolution"] = f"{w*16}x{h*mult*16}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Classify H.264 stream features")
    parser.add_argument("input", nargs="?", help="H.264 input file")
    parser.add_argument("--all", action="store_true", help="All conformance files")
    parser.add_argument("--fate-dir", default="fate-suite/h264-conformance",
                        help="Path to conformance files")
    args = parser.parse_args()

    if args.all:
        fate = Path(args.fate_dir)
        files = sorted(fate.glob("*.264")) + sorted(fate.glob("*.jsv")) + \
                sorted(fate.glob("*.26l")) + sorted(fate.glob("*.h264")) + \
                sorted(fate.glob("*.avc"))
    elif args.input:
        files = [Path(args.input)]
    else:
        parser.print_help()
        sys.exit(1)

    fmt = "{:<35s} {:>8s} {:>6s} {:>5s} {:>4s} {:>3s} {:>6s} {:>12s}"
    print(fmt.format("File", "Profile", "Ent", "MBsO", "MAff", "8x8", "Frames", "Resolution"))
    print("-" * 90)

    for f in files:
        r = classify_file(str(f))
        print(fmt.format(
            r["file"][:35],
            r["profile"],
            r["entropy"],
            str(r.get("mbs_only", "?")),
            str(r.get("mb_aff", "?")),
            str(r.get("8x8", "?")),
            str(r["frames"]),
            r.get("resolution", "?"),
        ))


if __name__ == "__main__":
    main()
