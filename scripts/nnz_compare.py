#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Compare NNZ values between wedeo and FFmpeg for a specific MB.

Extracts stored NNZ arrays from both decoders for a given macroblock
in a given decoded frame, identifying mismatches that cause deblock bS
divergences.

Usage:
    python3 scripts/nnz_compare.py <h264_file> --mb 7,0 --frame 1
    python3 scripts/nnz_compare.py HPCV_BRCM_A --mb 8,0 --frame 1

Requires:
    - FFmpeg debug build at ./FFmpeg/ffmpeg (--disable-asm --enable-debug=3)
    - wedeo-framecrc built (cargo build --release -p wedeo-fate)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import (
    find_wedeo_binary,
    resolve_conformance_file,
    run_lldb,
)

FFMPEG_DBG = Path("FFmpeg/ffmpeg")


def extract_ffmpeg_nnz(
    input_path: Path, mb_x: int, mb_y: int, ignore_count: int
) -> list[int] | None:
    """Extract stored NNZ[0..15] from FFmpeg via lldb for a specific MB."""
    if not FFMPEG_DBG.exists():
        print("WARNING: FFmpeg debug build not found at FFmpeg/ffmpeg", file=sys.stderr)
        return None

    exprs = [f"(int)h->non_zero_count[sl->mb_xy][{i}]" for i in range(16)]
    result = run_lldb(
        ffmpeg_bin=str(FFMPEG_DBG),
        input_path=str(input_path),
        expressions=exprs,
        breakpoint_func="ff_h264_filter_mb",
        breakpoint_condition=f"mb_x == {mb_x} && mb_y == {mb_y}",
        ignore_count=ignore_count,
        timeout=60,
        ffmpeg_extra_args=["-threads", "1", "-bitexact"],
    )
    if not result.values or len(result.values) < 16:
        print(f"WARNING: lldb extraction failed (got {len(result.values)} values)", file=sys.stderr)
        return None
    return [v for v in result.values[:16]]


def extract_wedeo_nnz(
    input_path: Path, mb_x: int, mb_y: int, decoded_frame: int
) -> list[int] | None:
    """Extract stored NNZ from wedeo trace for a specific MB."""
    wedeo_bin = find_wedeo_binary()
    if not wedeo_bin:
        print("WARNING: wedeo-framecrc not found", file=sys.stderr)
        return None

    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::mb=trace"}
    result = subprocess.run(
        [wedeo_bin, str(input_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    # Parse MB_RECON traces to find NNZ for the target MB in the target frame.
    # We count deblock_frame spans to identify frames.
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    frame_idx = -1
    target_nnz = None

    for line in result.stderr.splitlines():
        clean = ansi_escape.sub("", line)
        # Track frame boundaries via deblock_frame spans
        if "deblock_frame" in clean and "deblocking frame" in clean:
            frame_idx += 1
            continue
        if frame_idx != decoded_frame:
            continue
        # Look for NNZ_STORE or MB_RECON with the target MB
        if f"mb_x={mb_x}" in clean and f"mb_y={mb_y}" in clean:
            # Try to extract nnz array from NNZ_STORE trace
            m = re.search(r"nnz=\[([^\]]+)\]", clean)
            if m:
                vals = [int(x.strip()) for x in m.group(1).split(",")]
                if len(vals) >= 16:
                    target_nnz = vals[:16]

    return target_nnz


def main():
    parser = argparse.ArgumentParser(description="Compare NNZ between wedeo and FFmpeg")
    parser.add_argument("input", help="H.264 input file (path or conformance name)")
    parser.add_argument("--mb", required=True, help="MB position as x,y (e.g., 7,0)")
    parser.add_argument(
        "--frame",
        type=int,
        default=1,
        help="Decoded frame index (0=IDR, 1=first P, etc.)",
    )
    args = parser.parse_args()

    input_path = resolve_conformance_file(args.input)
    if not input_path or not input_path.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    mb_x, mb_y = (int(x) for x in args.mb.split(","))

    print(f"File: {input_path.name}")
    print(f"MB({mb_x},{mb_y}), decoded frame {args.frame}")
    print()

    # FFmpeg: ignore_count = frame_index (one hit per frame for this MB)
    ffmpeg_nnz = extract_ffmpeg_nnz(input_path, mb_x, mb_y, ignore_count=args.frame)
    # Wedeo: parse traces
    wedeo_nnz = extract_wedeo_nnz(input_path, mb_x, mb_y, args.frame)

    if ffmpeg_nnz is None and wedeo_nnz is None:
        print("ERROR: Could not extract NNZ from either decoder", file=sys.stderr)
        sys.exit(2)

    print("Position  FFmpeg  Wedeo  Match")
    print("--------  ------  -----  -----")
    mismatches = 0
    for i in range(16):
        bx, by = i % 4, i // 4
        fv = ffmpeg_nnz[i] if ffmpeg_nnz else "?"
        wv = wedeo_nnz[i] if wedeo_nnz else "?"
        match = "=" if fv == wv else "DIFF"
        if fv != wv:
            mismatches += 1
        print(f"nnz[{i:2}] ({bx},{by})  {fv:>5}  {wv:>5}  {match}")

    print()
    if mismatches == 0:
        print("NNZ arrays match.")
    else:
        print(f"{mismatches} mismatches found.")


if __name__ == "__main__":
    main()
