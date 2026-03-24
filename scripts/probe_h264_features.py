#!/usr/bin/env python3
"""Probe H.264 files for codec features: entropy mode, 8x8 transform, profile, interlacing.

Scans a directory of H.264 files and reports key codec parameters using FFmpeg's
trace_headers BSF and ffprobe, formatted as a table.

Usage:
    # Probe all FRext files
    python3 scripts/probe_h264_features.py fate-suite/h264-conformance/FRext/

    # Probe a specific file
    python3 scripts/probe_h264_features.py fate-suite/h264-conformance/FRext/HPCV_BRCM_A.264

    # Filter: only show 8x8 transform files
    python3 scripts/probe_h264_features.py fate-suite/h264-conformance/FRext/ --8x8

    # Filter: only progressive
    python3 scripts/probe_h264_features.py fate-suite/h264-conformance/FRext/ --progressive

Requires:
    - ffmpeg and ffprobe in PATH
"""
# /// script
# requires-python = ">=3.10"
# ///

import argparse
import subprocess
import sys
from pathlib import Path


def probe_file(path: Path) -> dict | None:
    """Extract codec features from an H.264 file."""
    info = {
        "file": path.name,
        "profile": "?",
        "entropy": "?",
        "t8x8": False,
        "interlaced": False,
        "width": 0,
        "height": 0,
    }

    # Use trace_headers BSF to get SPS/PPS flags
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(path), "-c:v", "copy", "-bsf:v", "trace_headers",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=10,
        )
        trace = result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    # Parse only the first occurrence of each field (multi-SPS/PPS files repeat them).
    found_profile = False
    found_entropy = False
    found_t8x8 = False
    for line in trace.splitlines():
        if not found_profile and "profile_idc" in line and "=" in line:
            val = line.split("=")[-1].strip()
            try:
                idc = int(val)
                info["profile"] = {66: "Baseline", 77: "Main", 88: "Extended",
                                   100: "High", 110: "High10", 122: "High422",
                                   244: "High444"}.get(idc, f"idc={idc}")
                found_profile = True
            except ValueError:
                pass
        if not found_entropy and "entropy_coding_mode_flag" in line and "=" in line:
            val = line.split("=")[-1].strip()
            info["entropy"] = "CABAC" if val == "1" else "CAVLC"
            found_entropy = True
        if not found_t8x8 and "transform_8x8_mode_flag" in line and "=" in line:
            val = line.split("=")[-1].strip()
            info["t8x8"] = val == "1"
            found_t8x8 = True
        if found_profile and found_entropy and found_t8x8:
            break

    # Use ffprobe for dimensions and interlacing
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) >= 2:
            info["width"] = int(parts[0])
            info["height"] = int(parts[1])
    except (subprocess.TimeoutExpired, ValueError):
        pass

    # Check interlacing from first frame
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_frames", "-read_intervals", "%+#1",
             "-show_entries", "frame=interlaced_frame",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        info["interlaced"] = result.stdout.strip() == "1"
    except subprocess.TimeoutExpired:
        pass

    return info


def main():
    parser = argparse.ArgumentParser(description="Probe H.264 file features")
    parser.add_argument("path", help="File or directory to probe")
    parser.add_argument("--8x8", dest="filter_8x8", action="store_true",
                        help="Only show files with transform_8x8_mode=1")
    parser.add_argument("--progressive", action="store_true",
                        help="Only show progressive (non-interlaced) files")
    ent_group = parser.add_mutually_exclusive_group()
    ent_group.add_argument("--cavlc", action="store_true", help="Only show CAVLC files")
    ent_group.add_argument("--cabac", action="store_true", help="Only show CABAC files")
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("*.264")) + sorted(target.glob("*.jsv")) + \
                sorted(target.glob("*.h264")) + sorted(target.glob("*.26l"))
    else:
        print(f"Error: {target} not found", file=sys.stderr)
        sys.exit(1)

    results = []
    for f in files:
        info = probe_file(f)
        if info is None:
            continue
        if args.filter_8x8 and not info["t8x8"]:
            continue
        if args.progressive and info["interlaced"]:
            continue
        if args.cavlc and info["entropy"] != "CAVLC":
            continue
        if args.cabac and info["entropy"] != "CABAC":
            continue
        results.append(info)

    if not results:
        print("No matching files found.")
        return

    # Print table
    name_w = max(len(r["file"]) for r in results)
    prof_w = max(len(r["profile"]) for r in results)
    print(f"{'File':<{name_w}}  {'Profile':<{prof_w}}  Entropy  8x8  Intl  Size")
    print(f"{'-' * name_w}  {'-' * prof_w}  -------  ---  ----  ----")
    for r in results:
        t8 = "yes" if r["t8x8"] else "no "
        il = "yes" if r["interlaced"] else "no "
        sz = f"{r['width']}x{r['height']}" if r["width"] else "?"
        print(f"{r['file']:<{name_w}}  {r['profile']:<{prof_w}}  {r['entropy']:<7}  {t8}  {il}   {sz}")

    print(f"\n{len(results)} files shown")


if __name__ == "__main__":
    main()
