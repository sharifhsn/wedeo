#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Compare CABAC engine state between wedeo and FFmpeg for MBAFF files.

Runs wedeo with MBAFF_PAIR_DONE traces and FFmpeg under lldb to extract
CABAC (pos, low, range) at pair boundaries. Reports the first diverging pair.

Usage:
    python3 scripts/mbaff_cabac_compare.py fate-suite/h264-conformance/CAMA1_Sony_C.jsv
    python3 scripts/mbaff_cabac_compare.py fate-suite/h264-conformance/CAMA1_Sony_C.jsv --max-pairs 200
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary


def get_wedeo_pairs(input_path: str, wedeo_bin: str, max_pairs: int) -> list[dict]:
    """Run wedeo with MBAFF trace and extract per-pair CABAC state."""
    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::decoder=trace"}
    proc = subprocess.run(
        [wedeo_bin, input_path],
        capture_output=True, env=env, timeout=60,
    )
    stderr = proc.stderr.decode(errors="replace")

    pairs = []
    # Match: MBAFF_PAIR_DONE mb_x=N mb_y=N pos=N low=N range=N
    pattern = re.compile(
        r"MBAFF_PAIR_DONE mb_x=(\d+) mb_y=(\d+) pos=(\d+) low=(-?\d+) range=(\d+)"
    )
    # Only take first slice's pairs (poc=0)
    for line in stderr.splitlines():
        if "poc=0" not in line:
            continue
        m = pattern.search(line)
        if m:
            pairs.append({
                "mb_x": int(m.group(1)),
                "mb_y": int(m.group(2)),
                "pos": int(m.group(3)),
                "low": int(m.group(4)),
                "range": int(m.group(5)),
            })
            if len(pairs) >= max_pairs:
                break
    return pairs


def get_ffmpeg_pair(input_path: str, pair_idx: int) -> dict | None:
    """Use lldb to extract FFmpeg CABAC state at a specific pair index.

    pair_idx is 0-based. We use breakpoint ignore count = pair_idx
    (ignore that many hits, break on the next one).
    """
    ffmpeg_g = Path(__file__).resolve().parent.parent / "FFmpeg" / "ffmpeg_g"
    if not ffmpeg_g.exists():
        return None

    cmds = [
        "br s -f h264_slice.c -l 2637",
        f"br modify -i {pair_idx} 1",
        f"r -bitexact -i {input_path} -f framecrc /dev/null",
        "expression (int)sl->mb_x",
        "expression (int)sl->mb_y",
        "expression (int)(sl->cabac.bytestream - sl->cabac.bytestream_start)",
        "expression (int)sl->cabac.low",
        "expression (int)sl->cabac.range",
        "kill",
    ]

    lldb_args = ["lldb", "-b"]
    for c in cmds:
        lldb_args += ["-o", c]
    lldb_args += ["--", str(ffmpeg_g)]

    try:
        proc = subprocess.run(
            lldb_args, capture_output=True, timeout=30,
            cwd=str(ffmpeg_g.parent),
        )
    except Exception:
        return None

    output = proc.stdout.decode(errors="replace")
    vals = re.findall(r"\(int\) \$\d+ = (-?\d+)", output)
    if len(vals) < 5:
        return None

    return {
        "mb_x": int(vals[0]),
        "mb_y": int(vals[1]),
        "pos": int(vals[2]),
        "low": int(vals[3]),
        "range": int(vals[4]),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare MBAFF CABAC state")
    parser.add_argument("input", help="H.264 MBAFF input file")
    parser.add_argument("--max-pairs", type=int, default=200,
                        help="Max pairs to compare (default 200)")
    args = parser.parse_args()

    wedeo_bin = find_wedeo_binary()
    if not wedeo_bin:
        print("ERROR: wedeo-framecrc not found", file=sys.stderr)
        sys.exit(2)

    print(f"Getting wedeo pairs (max {args.max_pairs})...")
    wedeo_pairs = get_wedeo_pairs(args.input, wedeo_bin, args.max_pairs)
    if not wedeo_pairs:
        print("No MBAFF_PAIR_DONE traces found. Is this an MBAFF file?")
        sys.exit(1)
    print(f"Got {len(wedeo_pairs)} wedeo pairs.")

    # Binary search for first divergence
    lo, hi = 0, len(wedeo_pairs) - 1
    last_match = -1

    # Check first pair
    ffmpeg_first = get_ffmpeg_pair(args.input, 0)
    if not ffmpeg_first:
        print("ERROR: Could not get FFmpeg state (is ffmpeg_g built?)")
        sys.exit(2)

    if ffmpeg_first["pos"] != wedeo_pairs[0]["pos"]:
        print(f"DIVERGED at pair 0!")
        print(f"  FFmpeg: {ffmpeg_first}")
        print(f"  Wedeo:  {wedeo_pairs[0]}")
        sys.exit(0)

    print("Pair 0 matches. Binary searching for first divergence...")

    while lo <= hi:
        mid = (lo + hi) // 2
        ffmpeg_mid = get_ffmpeg_pair(args.input, mid)
        if not ffmpeg_mid:
            hi = mid - 1
            continue

        w = wedeo_pairs[mid]
        match = (ffmpeg_mid["pos"] == w["pos"] and
                 ffmpeg_mid["low"] == w["low"] and
                 ffmpeg_mid["range"] == w["range"])

        if match:
            last_match = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if last_match == len(wedeo_pairs) - 1:
        print(f"All {len(wedeo_pairs)} pairs match!")
    else:
        diverge_idx = last_match + 1
        print(f"\nLast matching pair: {last_match}")
        print(f"First diverging pair: {diverge_idx}")

        ffmpeg_div = get_ffmpeg_pair(args.input, diverge_idx)
        w = wedeo_pairs[diverge_idx]
        print(f"\n  Pair {diverge_idx}: mb_x={w['mb_x']} mb_y={w['mb_y']}")
        if ffmpeg_div:
            print(f"  FFmpeg: pos={ffmpeg_div['pos']} low={ffmpeg_div['low']} range={ffmpeg_div['range']}")
        print(f"  Wedeo:  pos={w['pos']} low={w['low']} range={w['range']}")
        if ffmpeg_div:
            print(f"  Delta:  pos={w['pos'] - ffmpeg_div['pos']} bytes")


if __name__ == "__main__":
    main()
