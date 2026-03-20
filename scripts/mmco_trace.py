#!/usr/bin/env python3
"""Compare DPB state between wedeo and FFmpeg after reference picture marking.

Extracts the short-term and long-term reference list contents from both decoders
after each frame's mark_reference call, and reports the first frame where they
diverge. This directly targets MMCO/DPB bugs (MR3/MR4/MR5).

Usage:
    python3 scripts/mmco_trace.py file.264
    python3 scripts/mmco_trace.py file.264 --max-frames 50

Requires:
    - Debug FFmpeg with --disable-asm
    - wedeo-framecrc with tracing feature
    - lldb in PATH
"""

import argparse
import re
import sys
from pathlib import Path

from ffmpeg_debug import (
    find_ffmpeg_binary,
    run_wedeo_with_tracing,
)


def extract_wedeo_dpb_states(input_path: Path, max_frames: int) -> list[dict]:
    """Extract DPB state after each frame from wedeo's debug traces.

    Returns list of dicts: {frame_num, poc, st_frame_nums: [int], lt_indices: [int]}.
    """
    trace = run_wedeo_with_tracing(
        str(input_path),
        rust_log="wedeo_codec_h264::refs=debug,wedeo_codec_h264::decoder=debug",
        no_deblock=True,
        features=["tracing"],
    )

    states = []

    # Parse DPB state log messages
    # Format: "DPB state h264_fn=N poc=M st_count=X st_frame_nums=[...] lt_count=Y lt_indices=[...]"
    dpb_re = re.compile(
        r"DPB state\s+h264_fn=(\d+)\s+poc=(-?\d+)"
        r"\s+st_count=(\d+)\s+st_frame_nums=\[([^\]]*)\]"
        r"\s+lt_count=(\d+)\s+lt_indices=\[([^\]]*)\]"
    )

    for line in trace.splitlines():
        m = dpb_re.search(line)
        if m:
            st_fns = [int(x.strip()) for x in m.group(4).split(",") if x.strip()]
            lt_idxs = [int(x.strip()) for x in m.group(6).split(",") if x.strip()]
            states.append({
                "frame_num": int(m.group(1)),
                "poc": int(m.group(2)),
                "st_count": int(m.group(3)),
                "st_frame_nums": sorted(st_fns),
                "lt_count": int(m.group(5)),
                "lt_indices": sorted(lt_idxs),
            })
            if len(states) >= max_frames:
                break

    return states


def extract_ffmpeg_dpb_states(input_path: Path, max_frames: int) -> list[dict]:
    """Extract DPB state from FFmpeg via -loglevel debug output.

    Returns list of dicts with the same format as extract_wedeo_dpb_states.
    """
    import subprocess

    ffmpeg_bin = find_ffmpeg_binary()
    result = subprocess.run(
        [str(ffmpeg_bin), "-threads", "1", "-bitexact",
         "-loglevel", "debug",
         "-i", str(input_path),
         "-f", "null", "/dev/null"],
        capture_output=True, text=True,
        timeout=300,
    )

    states = []

    # FFmpeg logs short/long ref counts in debug output
    # Pattern: "short ref count: N long ref count: M"
    # We also look for "decode_slice_header" for frame boundaries
    ref_count_re = re.compile(r"short ref count:\s*(\d+)\s*long ref count:\s*(\d+)")
    frame_num_re = re.compile(r"frame_num:\s*(\d+)")
    poc_re = re.compile(r"poc:\s*(-?\d+)")

    current_fn = 0
    current_poc = 0

    for line in result.stderr.splitlines():
        fn_m = frame_num_re.search(line)
        if fn_m:
            current_fn = int(fn_m.group(1))

        poc_m = poc_re.search(line)
        if poc_m:
            current_poc = int(poc_m.group(1))

        rc_m = ref_count_re.search(line)
        if rc_m:
            states.append({
                "frame_num": current_fn,
                "poc": current_poc,
                "st_count": int(rc_m.group(1)),
                "lt_count": int(rc_m.group(2)),
                "st_frame_nums": [],  # Not available from FFmpeg debug log
                "lt_indices": [],     # Not available from FFmpeg debug log
            })
            if len(states) >= max_frames:
                break

    return states


def main():
    parser = argparse.ArgumentParser(
        description="Compare DPB state between wedeo and FFmpeg"
    )
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Maximum number of frames to compare (default: 100)")
    parser.add_argument("--wedeo-only", action="store_true",
                        help="Only show wedeo DPB states (skip FFmpeg)")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Extract wedeo states
    print("Extracting wedeo DPB states...", file=sys.stderr)
    wedeo_states = extract_wedeo_dpb_states(input_path, args.max_frames)
    print(f"  {len(wedeo_states)} frames extracted", file=sys.stderr)

    if args.wedeo_only:
        print(f"\n{'Idx':>5} {'POC':>5} {'FN':>4} {'ST':>3} {'LT':>3} ST_frame_nums")
        print("-" * 60)
        for idx, s in enumerate(wedeo_states):
            fns = ",".join(str(x) for x in s["st_frame_nums"])
            lts = ",".join(str(x) for x in s["lt_indices"])
            lt_str = f" LT=[{lts}]" if lts else ""
            print(f"{idx:5d} {s['poc']:5d} {s['frame_num']:4d} {s['st_count']:3d} {s['lt_count']:3d} [{fns}]{lt_str}")
        return

    # Extract FFmpeg states
    print("Extracting FFmpeg DPB states...", file=sys.stderr)
    ffmpeg_states = extract_ffmpeg_dpb_states(input_path, args.max_frames)
    print(f"  {len(ffmpeg_states)} frames extracted", file=sys.stderr)

    # Compare
    print(f"\n{'Frame':>5} {'POC':>5} {'Wedeo ST':>9} {'FFmpeg ST':>9} {'Wedeo LT':>9} {'FFmpeg LT':>9} {'Status':>8}")
    print("-" * 70)

    n = min(len(wedeo_states), len(ffmpeg_states))
    first_diff = None
    for i in range(n):
        w, f = wedeo_states[i], ffmpeg_states[i]
        match = w["st_count"] == f["st_count"] and w["lt_count"] == f["lt_count"]
        status = "OK" if match else "DIFF"
        if not match and first_diff is None:
            first_diff = i

        print(f"{i:5d} {w['poc']:5d} {w['st_count']:9d} {f['st_count']:9d} "
              f"{w['lt_count']:9d} {f['lt_count']:9d} {status:>8}")

    if first_diff is not None:
        print(f"\nFirst divergence at frame {first_diff}")
        w = wedeo_states[first_diff]
        print(f"  Wedeo: ST={w['st_count']} [{','.join(str(x) for x in w['st_frame_nums'])}] "
              f"LT={w['lt_count']} [{','.join(str(x) for x in w['lt_indices'])}]")
    elif n > 0:
        print(f"\nAll {n} frames match in ref counts")
    else:
        print("\nNo frames to compare")


if __name__ == "__main__":
    main()
