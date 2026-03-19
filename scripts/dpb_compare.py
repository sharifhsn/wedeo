#!/usr/bin/env python3
"""Compare DPB (Decoded Picture Buffer) state between wedeo and FFmpeg.

Extracts DPB contents at each frame from both decoders and shows where
the states first diverge. Critical for debugging MMCO and sliding window
bugs (MR4, MR5, HCBP1, HCBP2).

Usage:
    # Show DPB at first divergence
    python3 scripts/dpb_compare.py fate-suite/h264-conformance/MR4_TANDBERG_C.264

    # Show DPB at a specific frame
    python3 scripts/dpb_compare.py file.264 --frame 17

    # Show DPB for all frames (verbose)
    python3 scripts/dpb_compare.py file.264 --all

Requires:
    - wedeo debug binary with tracing
    - FFmpeg debug binary (FFmpeg/ffmpeg_g) built with --disable-asm
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ffmpeg_debug import (
    find_ffmpeg_binary,
    find_wedeo_binary,
    run_lldb,
    strip_ansi,
)


# ---------------------------------------------------------------------------
# Wedeo DPB extraction (via tracing)
# ---------------------------------------------------------------------------

@dataclass
class DpbRef:
    """A reference picture in the DPB."""
    frame_num: int
    poc: int
    status: str  # "ST" (short-term), "LT" (long-term)
    lt_idx: int = -1  # long-term frame index, -1 if short-term

    def __repr__(self) -> str:
        if self.status == "LT":
            return f"fn={self.frame_num} poc={self.poc} LT[{self.lt_idx}]"
        return f"fn={self.frame_num} poc={self.poc} ST"


@dataclass
class DpbState:
    """DPB state at a specific frame."""
    decode_idx: int
    frame_num: int
    poc: int
    slice_type: str
    refs: list[DpbRef]

    @property
    def short_term_frame_nums(self) -> list[int]:
        return sorted(r.frame_num for r in self.refs if r.status == "ST")

    @property
    def long_term_indices(self) -> list[int]:
        return sorted(r.lt_idx for r in self.refs if r.status == "LT")


def extract_wedeo_dpb(input_path: str, max_frames: int = 0) -> list[DpbState]:
    """Extract per-frame DPB state from wedeo via tracing."""
    wedeo_bin = find_wedeo_binary(prefer_debug=True, features=["tracing"])
    env = {
        "RUST_LOG": "wedeo_codec_h264::decoder=debug,wedeo_codec_h264::refs=debug",
        "WEDEO_NO_DEBLOCK": "1",
    }
    import os
    full_env = {**os.environ, **env}
    result = subprocess.run(
        [str(wedeo_bin), input_path],
        capture_output=True, env=full_env, timeout=120,
    )
    trace = strip_ansi(result.stderr.decode("utf-8", errors="replace"))

    states = []
    decode_idx = -1
    current_frame_num = 0
    current_poc = 0
    current_type = "?"

    # Parse "slice start" / "frame complete" / "DPB state" lines
    pending_refs: list[DpbRef] = []
    for line in trace.splitlines():
        if "slice start" in line:
            m_type = re.search(r"slice_type=(\w+)", line)
            m_fn = re.search(r"frame_num=(\d+)", line)
            if m_type:
                current_type = m_type.group(1)
            if m_fn:
                current_frame_num = int(m_fn.group(1))

        elif "DPB state" in line:
            # Parse: st_frame_nums=[0, 1, 2] lt_indices=[3]
            pending_refs = []
            m_st = re.search(r"st_frame_nums=\[([^\]]*)\]", line)
            if m_st and m_st.group(1).strip():
                for fn_str in m_st.group(1).split(","):
                    fn_str = fn_str.strip()
                    if fn_str:
                        pending_refs.append(DpbRef(
                            frame_num=int(fn_str), poc=-1, status="ST",
                        ))
            m_lt = re.search(r"lt_indices=\[([^\]]*)\]", line)
            if m_lt and m_lt.group(1).strip():
                for idx_str in m_lt.group(1).split(","):
                    idx_str = idx_str.strip()
                    if idx_str:
                        pending_refs.append(DpbRef(
                            frame_num=-1, poc=-1, status="LT",
                            lt_idx=int(idx_str),
                        ))

        elif "frame complete" in line:
            decode_idx += 1
            m_poc = re.search(r"poc=(-?\d+)", line)
            if m_poc:
                current_poc = int(m_poc.group(1))

            states.append(DpbState(
                decode_idx=decode_idx,
                frame_num=current_frame_num,
                poc=current_poc,
                slice_type=current_type,
                refs=list(pending_refs),
            ))
            pending_refs = []

            if max_frames and decode_idx + 1 >= max_frames:
                break

    return states


def extract_wedeo_dpb_detail(input_path: str, max_frames: int = 0) -> list[DpbState]:
    """Extract per-frame DPB state from wedeo via detailed tracing.

    Uses tracing-detail feature for full DPB dump. Falls back to
    basic tracing if tracing-detail is not available.
    """
    wedeo_bin = find_wedeo_binary(
        prefer_debug=True, features=["tracing", "tracing-detail"]
    )
    import os
    env = {
        **os.environ,
        "RUST_LOG": (
            "wedeo_codec_h264::decoder=trace,"
            "wedeo_codec_h264::refs=trace"
        ),
        "WEDEO_NO_DEBLOCK": "1",
    }
    result = subprocess.run(
        [str(wedeo_bin), input_path],
        capture_output=True, env=env, timeout=120,
    )
    trace = strip_ansi(result.stderr.decode("utf-8", errors="replace"))

    states = []
    decode_idx = -1
    current_frame_num = 0
    current_poc = 0
    current_type = "?"

    for line in trace.splitlines():
        if "slice start" in line:
            m = re.search(r"slice_type=(\w+)", line)
            if m:
                current_type = m.group(1)
            m = re.search(r"frame_num=(\d+)", line)
            if m:
                current_frame_num = int(m.group(1))

        elif "frame complete" in line:
            decode_idx += 1
            m = re.search(r"poc=(-?\d+)", line)
            if m:
                current_poc = int(m.group(1))

            states.append(DpbState(
                decode_idx=decode_idx,
                frame_num=current_frame_num,
                poc=current_poc,
                slice_type=current_type,
                refs=[],
            ))

            if max_frames and decode_idx + 1 >= max_frames:
                break

        elif "DPB stored entry" in line:
            # Parse individual DPB entries from tracing-detail
            m_fn = re.search(r"h264_frame_num=(\d+)", line)
            m_poc = re.search(r"poc=(-?\d+)", line)
            m_st = re.search(r"status=(ShortTerm|LongTerm|Unused)", line)
            if m_fn and m_poc and m_st and states:
                status = "ST" if m_st.group(1) == "ShortTerm" else (
                    "LT" if m_st.group(1) == "LongTerm" else "U"
                )
                if status != "U":
                    states[-1].refs.append(DpbRef(
                        frame_num=int(m_fn.group(1)),
                        poc=int(m_poc.group(1)),
                        status=status,
                    ))

    return states


# ---------------------------------------------------------------------------
# FFmpeg DPB extraction (via lldb)
# ---------------------------------------------------------------------------

def extract_ffmpeg_dpb_at_frame(
    input_path: str, frame_idx: int,
) -> DpbState | None:
    """Extract FFmpeg DPB state at a specific frame via lldb.

    Breaks at ff_h264_execute_ref_pic_marking and reads short_ref/long_ref arrays.
    """
    ffmpeg_bin = find_ffmpeg_binary()

    # We break at the start of ref pic marking for the target frame.
    # Each call to ff_h264_execute_ref_pic_marking corresponds to one reference frame.
    ignore_count = frame_idx  # skip this many breakpoint hits

    lldb_commands = f"""
breakpoint set -n ff_h264_execute_ref_pic_marking
breakpoint modify 1 -i {ignore_count}
run -bitexact -i {input_path} -f null -

# Read short_ref array
expr h->short_ref_count
expr (int)h->short_ref[0]->frame_num
expr (int)h->short_ref[0]->poc
expr h->long_ref_count

# Read current frame info
expr sl->slice_type_nos
expr sl->mb_y
expr h->poc.frame_num

quit
"""
    try:
        result = run_lldb(str(ffmpeg_bin), lldb_commands, timeout=30)
        # Parse results - this is fragile and depends on lldb output format
        return None  # TODO: parse lldb output
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_dpb_states(
    wedeo_states: list[DpbState],
    target_frame: int | None = None,
    show_all: bool = False,
) -> None:
    """Print DPB states, highlighting the first divergence point."""
    if not wedeo_states:
        print("No frames decoded by wedeo.")
        return

    if target_frame is not None:
        if target_frame >= len(wedeo_states):
            print(f"Frame {target_frame} not available (only {len(wedeo_states)} frames)")
            return
        s = wedeo_states[target_frame]
        print(f"Frame {s.decode_idx}: fn={s.frame_num} poc={s.poc} type={s.slice_type}")
        print(f"  Short-term frame_nums: {s.short_term_frame_nums}")
        if s.long_term_indices:
            print(f"  Long-term indices: {s.long_term_indices}")
        if s.refs:
            print(f"  All refs ({len(s.refs)}):")
            for r in s.refs:
                print(f"    {r}")
        else:
            print("  (No detailed ref info — use --detail for tracing-detail)")
        return

    # Print all or summary
    for s in wedeo_states:
        st_fns = s.short_term_frame_nums
        lt_fns = s.long_term_indices
        st_str = ",".join(str(f) for f in st_fns) if st_fns else "none"
        lt_str = ",".join(str(f) for f in lt_fns) if lt_fns else ""
        ref_str = f"ST=[{st_str}]"
        if lt_str:
            ref_str += f" LT=[{lt_str}]"

        if show_all or s.refs:
            print(
                f"  frame {s.decode_idx:3d}: fn={s.frame_num:3d} "
                f"poc={s.poc:4d} {s.slice_type:1s}  {ref_str}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare DPB state between wedeo and FFmpeg",
    )
    parser.add_argument("input", help="H.264 conformance file")
    parser.add_argument("--frame", type=int, default=None,
                        help="Show DPB at specific frame index")
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Max frames to decode (0=all)")
    parser.add_argument("--all", action="store_true",
                        help="Show DPB for all frames")
    parser.add_argument("--detail", action="store_true",
                        help="Use tracing-detail for full DPB dump")
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting wedeo DPB state for {Path(input_path).name}...",
          file=sys.stderr)

    if args.detail:
        states = extract_wedeo_dpb_detail(input_path, args.max_frames)
    else:
        states = extract_wedeo_dpb(input_path, args.max_frames)

    print(f"Decoded {len(states)} frames\n")
    compare_dpb_states(states, args.frame, args.all)


if __name__ == "__main__":
    main()
