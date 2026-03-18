#!/usr/bin/env python3
"""Dump decode order, POC, slice type, and nal_ref_idc for every frame.

Runs both wedeo (via tracing) and FFmpeg (via trace_headers BSF) on the
same input file, then prints a side-by-side table showing how each
decoder assigns POC and in what order frames are output.

Usage:
    python3 scripts/poc_dump.py fate-suite/h264-conformance/BA3_SVA_C.264
    python3 scripts/poc_dump.py --wedeo-only input.264
    python3 scripts/poc_dump.py --ffmpeg-only input.264
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def find_wedeo_bin() -> str:
    """Find the wedeo-framecrc binary (debug preferred for tracing support)."""
    for profile in ["debug", "release"]:
        candidate = Path("target") / profile / "wedeo-framecrc"
        if candidate.exists():
            return str(candidate.resolve())
    print(
        "Error: wedeo-framecrc not found. Run:\n"
        "  cargo build --bin wedeo-framecrc -p wedeo-fate --features tracing",
        file=sys.stderr,
    )
    sys.exit(1)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def wedeo_poc_dump(input_path: str) -> list[dict]:
    """Extract POC info from wedeo via tracing.

    Returns list of dicts: {decode_order, poc, slice_type, ref_idc} in decode order.
    """
    wedeo_bin = find_wedeo_bin()
    env = {**os.environ, "RUST_LOG": "debug"}

    result = subprocess.run(
        [wedeo_bin, input_path],
        capture_output=True,
        env=env,
    )

    frames = []
    last_slice_type = "?"
    last_is_idr = "?"
    for line in result.stderr.decode("utf-8", errors="replace").splitlines():
        line = strip_ansi(line)
        if "slice start" in line:
            m_st = re.search(r"slice_type=(\w+)", line)
            m_idr = re.search(r"is_idr=(\w+)", line)
            if m_st:
                last_slice_type = m_st.group(1)
                last_is_idr = m_idr.group(1) if m_idr else "?"
        elif "frame complete" in line:
            m_fn = re.search(r"frame_num=(\d+)", line)
            m_poc = re.search(r"poc=(-?\d+)", line)
            m_hb = re.search(r"has_b=(\w+)", line)
            if m_fn and m_poc:
                frames.append({
                    "decode_order": int(m_fn.group(1)),
                    "poc": int(m_poc.group(1)),
                    "has_b": m_hb.group(1) if m_hb else "?",
                    "slice_type": last_slice_type,
                    "is_idr": last_is_idr,
                })

    return frames


def ffmpeg_poc_dump(input_path: str) -> list[dict]:
    """Extract POC info from FFmpeg via trace_headers BSF.

    Returns list of dicts with poc_lsb, slice_type, nal_ref_idc.
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-bsf:v", "trace_headers",
            "-i", input_path,
            "-f", "null", "-",
        ],
        capture_output=True,
    )

    stderr = result.stderr.decode("utf-8", errors="replace")

    # Parse trace_headers output for slice NALUs
    frames = []
    current = {}

    for line in stderr.splitlines():
        if "slice_type" in line and "=" in line:
            # Start of a new slice
            if current:
                frames.append(current)
            current = {}
            m = re.search(r"slice_type\s+\S+\s*=\s*(\d+)", line)
            if m:
                raw = int(m.group(1))
                types = {0: "P", 1: "B", 2: "I", 3: "SP", 4: "SI",
                         5: "P", 6: "B", 7: "I", 8: "SP", 9: "SI"}
                current["slice_type"] = types.get(raw, f"?({raw})")
                current["slice_type_raw"] = raw

        if "pic_order_cnt_lsb" in line and "log2" not in line:
            m = re.search(r"pic_order_cnt_lsb\s+\S+\s*=\s*(\d+)", line)
            if m:
                current["poc_lsb"] = int(m.group(1))

        if "frame_num " in line and "log2" not in line:
            m = re.search(r"frame_num\s+\S+\s*=\s*(\d+)", line)
            if m:
                current["frame_num"] = int(m.group(1))

    if current:
        frames.append(current)

    # Also get output order from framecrc
    result2 = subprocess.run(
        ["ffmpeg", "-bitexact", "-i", input_path, "-f", "framecrc", "-"],
        capture_output=True,
    )
    output_lines = []
    for line in result2.stdout.decode("utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            output_lines.append(int(parts[1].strip()))

    return frames, output_lines


def main():
    parser = argparse.ArgumentParser(
        description="Dump decode order, POC, and slice type for every frame.",
    )
    parser.add_argument("input", help="Input H.264 file")
    parser.add_argument("--wedeo-only", action="store_true",
                        help="Only show wedeo decode info")
    parser.add_argument("--ffmpeg-only", action="store_true",
                        help="Only show FFmpeg decode info")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    show_wedeo = not args.ffmpeg_only
    show_ffmpeg = not args.wedeo_only

    if show_wedeo:
        wedeo_frames = wedeo_poc_dump(input_path)
        print(f"=== Wedeo decode order ({len(wedeo_frames)} frames) ===")
        print(f"{'#':>3}  {'Type':>5}  {'POC':>5}  {'has_b':>6}")
        print(f"{'---':>3}  {'-----':>5}  {'-----':>5}  {'------':>6}")
        for i, f in enumerate(wedeo_frames):
            print(f"{i:3d}  {f.get('slice_type', '?'):>5}  {f['poc']:5d}  {f.get('has_b', '?'):>6}")
        print()

    if show_ffmpeg:
        try:
            ffmpeg_frames, output_pts = ffmpeg_poc_dump(input_path)
            print(f"=== FFmpeg decode order ({len(ffmpeg_frames)} slices) ===")
            print(f"{'#':>3}  {'Type':>5}  {'poc_lsb':>8}  {'frame_num':>10}")
            print(f"{'---':>3}  {'-----':>5}  {'--------':>8}  {'----------':>10}")
            for i, f in enumerate(ffmpeg_frames):
                print(
                    f"{i:3d}  {f.get('slice_type', '?'):>5}"
                    f"  {f.get('poc_lsb', '?'):>8}"
                    f"  {f.get('frame_num', '?'):>10}"
                )
            print(f"\nFFmpeg output PTS order: {output_pts[:20]}{'...' if len(output_pts) > 20 else ''}")
            print()
        except FileNotFoundError:
            print("SKIP: ffmpeg not found", file=sys.stderr)

    if show_wedeo and show_ffmpeg:
        # Compare output orders
        wedeo_pocs = [f["poc"] for f in wedeo_frames]
        print(f"Wedeo output POC order: {wedeo_pocs[:20]}{'...' if len(wedeo_pocs) > 20 else ''}")


if __name__ == "__main__":
    main()
