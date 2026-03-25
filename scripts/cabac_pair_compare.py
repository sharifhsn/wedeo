#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Compare CABAC byte positions at pair-row boundaries between wedeo and FFmpeg.

Extracts CABAC engine state (pos, low, range) at the start of each pair row
(mb_x=0 top MBs) from both wedeo traces and FFmpeg via lldb. Reports the
first divergence point.

Usage:
    python3 scripts/cabac_pair_compare.py fate-suite/h264-conformance/CAMA1_Sony_C.jsv

Requires:
    - wedeo-framecrc binary (cargo build --release)
    - FFmpeg debug build at FFmpeg/ffmpeg_g
    - lldb in PATH
"""

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary


def get_wedeo_pair_states(filepath: Path, wedeo_bin: str) -> list[dict]:
    """Extract MBAFF_MB_START traces for mb_x=0, even mb_y from wedeo."""
    env = {
        **os.environ,
        "RUST_LOG": "wedeo_codec_h264::decoder=trace",
    }
    result = subprocess.run(
        [wedeo_bin, str(filepath)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    # Strip ANSI
    text = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)

    states = []
    for line in text.split("\n"):
        if "MBAFF_MB_START mb_x=0 mb_y=" not in line:
            continue
        m = re.search(
            r"mb_x=(\d+) mb_y=(\d+) mb_field=(\w+) pos=(\d+) low=(\d+) range=(\d+)",
            line,
        )
        if m:
            mb_y = int(m.group(2))
            if mb_y % 2 == 0:  # top MBs only
                states.append(
                    {
                        "mb_y": mb_y,
                        "pos": int(m.group(4)),
                        "low": int(m.group(5)),
                        "range": int(m.group(6)),
                        "field": m.group(3),
                    }
                )
    return states


def get_ffmpeg_state(filepath: Path, mb_y: int) -> dict | None:
    """Extract CABAC state from FFmpeg via lldb at MB(0, mb_y)."""
    ffmpeg_g = Path("FFmpeg/ffmpeg_g")
    if not ffmpeg_g.exists():
        return None

    result = subprocess.run(
        [
            "lldb",
            str(ffmpeg_g),
            "-o",
            f"breakpoint set -n ff_h264_decode_mb_cabac -c 'sl->mb_x == 0 && sl->mb_y == {mb_y}'",
            "-o",
            f"process launch -- -bitexact -i {filepath} -f null -",
            "-o",
            "p (int)(sl->cabac.bytestream - sl->cabac.bytestream_start)",
            "-o",
            "p sl->cabac.range",
            "-o",
            "p sl->cabac.low",
            "-o",
            "kill",
            "-o",
            "quit",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    values = re.findall(r"\(int\)\s+\$\d+\s+=\s+(-?\d+)", result.stdout)
    if len(values) >= 3:
        return {"pos": int(values[0]), "range": int(values[1]), "low": int(values[2])}
    return None


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <h264_file>", file=sys.stderr)
        sys.exit(1)

    filepath = Path(sys.argv[1]).resolve()
    wedeo_bin = find_wedeo_binary()

    print(f"Extracting wedeo pair states...")
    wedeo_states = get_wedeo_pair_states(filepath, wedeo_bin)
    if not wedeo_states:
        print("No MBAFF pair states found. Is this an MBAFF file?")
        sys.exit(1)

    print(f"Found {len(wedeo_states)} pair rows in wedeo output.")
    print()

    # Note: wedeo MBAFF_MB_START is AFTER field flag read.
    # FFmpeg ff_h264_decode_mb_cabac entry is BEFORE field flag read.
    # Only pos is directly comparable (field flag doesn't change pos much).
    print(f"{'mb_y':>5} {'W_pos':>8} {'F_pos':>8} {'pos_Δ':>6} {'W_range':>8} {'F_range':>8} {'Status'}")
    print("-" * 65)

    first_diverge = None
    for ws in wedeo_states:
        mb_y = ws["mb_y"]
        fs = get_ffmpeg_state(filepath, mb_y)
        if fs is None:
            print(f"{mb_y:>5} {ws['pos']:>8} {'?':>8} {'?':>6} {ws['range']:>8} {'?':>8} SKIP")
            continue

        pos_delta = ws["pos"] - fs["pos"]
        # Note: range comparison is approximate due to before/after field flag difference
        status = "OK" if pos_delta == 0 else f"DIVERGE ({pos_delta:+d} bytes)"
        if pos_delta != 0 and first_diverge is None:
            first_diverge = mb_y

        print(
            f"{mb_y:>5} {ws['pos']:>8} {fs['pos']:>8} {pos_delta:>+6d} "
            f"{ws['range']:>8} {fs['range']:>8} {status}"
        )

    print()
    if first_diverge is not None:
        print(f"First pos divergence at mb_y={first_diverge}")
    else:
        print("All pair-row positions match!")


if __name__ == "__main__":
    main()
