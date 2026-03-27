#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Compare CABAC engine state (low, range) between wedeo and FFmpeg at MB boundaries.

Extracts CABAC state from both decoders at the start of each non-skip MB
and reports the first divergence point. Useful for isolating whether a CABAC
bug is in the engine (state differs) or in context computation (state matches
but decoded values differ).

Prerequisites:
- FFmpeg built with debug fprintf in ff_h264_decode_mb_cabac (h264_cabac.c)
  after skip flag decode: fprintf(stderr, "FF_CABAC_MB_START mb_x=%d mb_y=%d
  low=%d range=%d frame_num=%d\\n", ...);
- Wedeo built with CABAC_MB_START trace (already present in cabac.rs)

Usage:
    python3 scripts/cabac_state_compare.py <input.264> [--frame-num N]
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def find_wedeo_bin() -> str:
    for profile in ("release", "debug"):
        candidate = Path("target") / profile / "wedeo-framecrc"
        if candidate.exists():
            return str(candidate)
    print("Error: wedeo-framecrc not found. Run cargo build.", file=sys.stderr)
    sys.exit(1)


def find_ffmpeg_bin() -> str:
    local = Path("FFmpeg/ffmpeg")
    if local.exists():
        return str(local)
    print("Error: FFmpeg/ffmpeg not found. Build debug FFmpeg.", file=sys.stderr)
    sys.exit(1)


def extract_wedeo_states(
    wedeo_bin: str, input_path: str, frame_num: int | None
) -> dict[tuple[int, int], tuple[int, int]]:
    """Extract CABAC MB_START states from wedeo."""
    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::cabac=trace"}
    result = subprocess.run(
        [wedeo_bin, input_path],
        capture_output=True,
        text=True,
        env=env,
    )
    combined = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)

    # Find the target slice type context
    pattern = re.compile(
        r"CABAC_MB_START mb_x=(\d+) mb_y=(\d+) slice_type=(\w+)"
    )
    states: dict[tuple[int, int], tuple[int, int]] = {}
    # We need low/range from the trace
    lr_pattern = re.compile(
        r"CABAC_MB_START mb_x=(\d+) mb_y=(\d+) low=(\d+) range=(\d+)"
    )
    for line in combined.split("\n"):
        m = lr_pattern.search(line)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            low, rng = int(m.group(3)), int(m.group(4))
            if key not in states:
                states[key] = (low, rng)
    return states


def extract_ffmpeg_states(
    ffmpeg_bin: str, input_path: str, frame_num: int | None
) -> dict[tuple[int, int], tuple[int, int]]:
    """Extract CABAC MB_START states from FFmpeg."""
    result = subprocess.run(
        [ffmpeg_bin, "-threads", "1", "-i", input_path, "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr

    pattern = re.compile(
        r"FF_CABAC_MB_START mb_x=(\d+) mb_y=(\d+) low=(\d+) range=(\d+)"
        + (rf" frame_num={frame_num}" if frame_num is not None else r" frame_num=(\d+)")
    )
    states: dict[tuple[int, int], tuple[int, int]] = {}
    for line in combined.split("\n"):
        m = pattern.search(line)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            low, rng = int(m.group(3)), int(m.group(4))
            if key not in states:
                states[key] = (low, rng)
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CABAC state between decoders")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--frame-num", type=int, default=None,
                        help="Filter FFmpeg by frame_num (default: first frame)")
    args = parser.parse_args()

    wedeo_bin = find_wedeo_bin()
    ffmpeg_bin = find_ffmpeg_bin()

    print("Extracting wedeo CABAC states...", file=sys.stderr)
    w_states = extract_wedeo_states(wedeo_bin, args.input, args.frame_num)
    print(f"  Got {len(w_states)} MBs", file=sys.stderr)

    print("Extracting FFmpeg CABAC states...", file=sys.stderr)
    f_states = extract_ffmpeg_states(ffmpeg_bin, args.input, args.frame_num)
    print(f"  Got {len(f_states)} MBs", file=sys.stderr)

    # Compare
    matches = 0
    diffs = 0
    first_diff = None
    # Determine grid size from available data
    max_x = max((k[0] for k in {**w_states, **f_states}), default=0)
    max_y = max((k[1] for k in {**w_states, **f_states}), default=0)

    for mb_y in range(max_y + 1):
        for mb_x in range(max_x + 1):
            key = (mb_x, mb_y)
            if key in w_states and key in f_states:
                wl, wr = w_states[key]
                fl, fr = f_states[key]
                if wl == fl and wr == fr:
                    matches += 1
                else:
                    diffs += 1
                    if first_diff is None:
                        first_diff = key
                        print(f"FIRST DIFF at MB({mb_x},{mb_y}):")
                        print(f"  wedeo: low={wl} range={wr}")
                        print(f"  ffmpeg: low={fl} range={fr}")
                    if diffs <= 5:
                        print(f"  DIFF MB({mb_x},{mb_y}): w=({wl},{wr}) f=({fl},{fr})")

    print(f"\nSummary: {matches} match, {diffs} differ")
    if first_diff is None:
        print("All compared MBs have identical CABAC state!")
    else:
        print(f"First divergence: MB{first_diff}")


if __name__ == "__main__":
    main()
