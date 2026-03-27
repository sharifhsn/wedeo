#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Compare deblock boundary strength (bS) between wedeo and FFmpeg for a specific MB.

Extracts bS[dir][edge][pair] from both decoders via lldb (FFmpeg) and tracing
(wedeo), identifying mismatches that cause pixel-level deblock divergences.

Usage:
    python3 scripts/bs_compare.py <h264_file> --mb 8,0 --frame 1
    python3 scripts/bs_compare.py HPCV_BRCM_A --mb 8,0 --frame 1

Requires:
    - FFmpeg debug build at ./FFmpeg/ffmpeg (--disable-asm --enable-debug=3)
    - wedeo-framecrc built (cargo build --release -p wedeo-fate)
    - Wedeo deblock tracing (DEBLOCK_BS tag in deblock.rs)
"""
import argparse
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


def extract_ffmpeg_bs(
    input_path: Path, mb_x: int, mb_y: int, ignore_count: int
) -> dict | None:
    """Extract bS values from FFmpeg's filter_mb_dir via lldb.

    Returns dict with 'nnz_left', 'nnz_top', 'nnz_cur', 'mb_type', 'left_type'.
    """
    if not FFMPEG_DBG.exists():
        return None

    # Extract NNZ cache at deblock time — this is what FFmpeg actually uses for bS.
    # Left neighbor: cache positions 11, 19, 27, 35
    # Current MB col 0: cache positions 12, 20, 28, 36
    # Top neighbor: cache positions 4, 5, 6, 7
    # Current MB row 0: cache positions 12, 13, 14, 15
    exprs = [
        # Left neighbor column (for V edge 0)
        "(int)sl->non_zero_count_cache[11]",
        "(int)sl->non_zero_count_cache[19]",
        "(int)sl->non_zero_count_cache[27]",
        "(int)sl->non_zero_count_cache[35]",
        # Current MB col 0 (for V edge 0)
        "(int)sl->non_zero_count_cache[12]",
        "(int)sl->non_zero_count_cache[20]",
        "(int)sl->non_zero_count_cache[28]",
        "(int)sl->non_zero_count_cache[36]",
        # Current MB internal: col 1 (V edge 1), col 2 (V edge 2)
        "(int)sl->non_zero_count_cache[13]",
        "(int)sl->non_zero_count_cache[14]",
        "(int)sl->non_zero_count_cache[15]",
        # MB types
        "(unsigned int)(h->cur_pic.mb_type[sl->mb_xy])",
        "(unsigned int)(sl->left_type[0])",
        "(unsigned int)(sl->top_type)",
    ]
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
    if not result.values or len(result.values) < 14:
        return None

    v = result.values
    return {
        "nnz_left": [v[0], v[1], v[2], v[3]],
        "nnz_cur_col0": [v[4], v[5], v[6], v[7]],
        "nnz_cur_col1": v[8],
        "nnz_cur_col2": v[9],
        "nnz_cur_col3": v[10],
        "mb_type": v[11],
        "left_type": v[12],
        "top_type": v[13],
    }


def extract_wedeo_bs(
    input_path: Path, mb_x: int, mb_y: int, decoded_frame: int
) -> dict | None:
    """Extract bS values from wedeo DEBLOCK_BS trace."""
    wedeo_bin = find_wedeo_binary()
    if not wedeo_bin:
        return None

    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::deblock=trace"}
    result = subprocess.run(
        [wedeo_bin, str(input_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    frame_idx = -1
    v_bs = None
    h_bs = None

    for line in result.stderr.splitlines():
        clean = ansi_escape.sub("", line)
        if "deblocking frame" in clean:
            frame_idx += 1
            continue
        if frame_idx != decoded_frame:
            continue
        if f"mb_x={mb_x}" not in clean or f"mb_y={mb_y}" not in clean:
            continue
        if "DEBLOCK_BS" not in clean:
            continue

        # Parse: dir="V" e0=[a,b,c,d] e1=[...] e2=[...] e3=[...] t8x8=...
        dir_m = re.search(r'dir="([VH])"', clean)
        if not dir_m:
            continue
        direction = dir_m.group(1)

        edges = {}
        for ei in range(4):
            em = re.search(rf"e{ei}=\[([^\]]+)\]", clean)
            if em:
                edges[ei] = [int(x.strip()) for x in em.group(1).split(",")]

        t8x8_m = re.search(r"t8x8=(\w+)", clean)
        t8x8 = t8x8_m.group(1) == "true" if t8x8_m else False

        if direction == "V":
            v_bs = {"edges": edges, "t8x8": t8x8}
        else:
            h_bs = {"edges": edges, "t8x8": t8x8}

    if v_bs is None and h_bs is None:
        return None
    return {"V": v_bs, "H": h_bs}


def main():
    parser = argparse.ArgumentParser(description="Compare deblock bS between wedeo and FFmpeg")
    parser.add_argument("input", help="H.264 input file (path or conformance name)")
    parser.add_argument("--mb", required=True, help="MB position as x,y (e.g., 8,0)")
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

    # Extract FFmpeg NNZ cache (what it actually uses for bS)
    ffmpeg_data = extract_ffmpeg_bs(input_path, mb_x, mb_y, ignore_count=args.frame)
    if ffmpeg_data:
        mb_type = ffmpeg_data["mb_type"]
        is_8x8 = (mb_type >> 24) & 1
        print(f"FFmpeg mb_type=0x{mb_type:08x} IS_8x8DCT={is_8x8}")
        left_type = ffmpeg_data["left_type"]
        left_8x8 = (left_type >> 24) & 1
        print(f"FFmpeg left_type=0x{left_type:08x} IS_8x8DCT={left_8x8}")
        print()
        print("FFmpeg NNZ cache for V edge 0 (left | cur):")
        for pair in range(4):
            nl = ffmpeg_data["nnz_left"][pair]
            nc = ffmpeg_data["nnz_cur_col0"][pair]
            bs = 2 if (nl or nc) else 0
            print(f"  pair {pair}: left_nnz={nl:3}  cur_nnz={nc:3}  → bS≥{bs}")
        print()

    # Extract wedeo bS
    wedeo_data = extract_wedeo_bs(input_path, mb_x, mb_y, args.frame)
    if wedeo_data and wedeo_data.get("V"):
        v = wedeo_data["V"]
        print(f"Wedeo V edges (t8x8={v['t8x8']}):")
        for ei, bs in sorted(v["edges"].items()):
            print(f"  edge {ei}: bS={bs}")
    if wedeo_data and wedeo_data.get("H"):
        h = wedeo_data["H"]
        print(f"Wedeo H edges (t8x8={h['t8x8']}):")
        for ei, bs in sorted(h["edges"].items()):
            print(f"  edge {ei}: bS={bs}")

    # Compare V edge 0 bS from both
    if ffmpeg_data and wedeo_data and wedeo_data.get("V"):
        print()
        print("=== V Edge 0 Comparison ===")
        wedeo_e0 = wedeo_data["V"]["edges"].get(0, [0, 0, 0, 0])
        mismatches = 0
        for pair in range(4):
            nl = ffmpeg_data["nnz_left"][pair]
            nc = ffmpeg_data["nnz_cur_col0"][pair]
            ffmpeg_bs = 2 if (nl or nc) else 0  # simplified, ignores MV check
            wedeo_bs = wedeo_e0[pair] if pair < len(wedeo_e0) else 0
            match = "=" if ffmpeg_bs == wedeo_bs or (ffmpeg_bs > 0) == (wedeo_bs > 0) else "DIFF"
            if match == "DIFF":
                mismatches += 1
            print(f"  pair {pair}: ffmpeg_nnz_bs≥{ffmpeg_bs}  wedeo_bs={wedeo_bs}  {match}")
        if mismatches:
            print(f"\n  {mismatches} potential bS mismatches at V edge 0")
        else:
            print("\n  V edge 0 bS consistent (NNZ-based check)")


if __name__ == "__main__":
    main()
