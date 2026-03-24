#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Compare per-block CAVLC bit positions between FFmpeg (via lldb) and wedeo (via tracing).

Extracts gb.index at each decode_residual call from FFmpeg using lldb,
and br.consumed() from wedeo's CAVLC trace, then reports the first divergence.

Usage:
    python3 scripts/bitpos_compare.py <h264_file> [--num-blocks 30] [--skip-blocks N]

Requires:
    - Debug FFmpeg built with --disable-asm at ./FFmpeg/
    - wedeo-framecrc built with tracing (cargo build --bin wedeo-framecrc -p wedeo-fate --features tracing)
    - lldb in PATH
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

NAL_HEADER_BITS = 8  # FFmpeg gb.index includes the 1-byte NAL header


def extract_ffmpeg_positions(
    h264_file: str, num_blocks: int, skip_blocks: int = 0,
) -> list[tuple[int, int]]:
    """Extract (gb.index, n) pairs from FFmpeg via lldb.

    Returns a list of (bit_position, block_index) for each decode_residual call.
    """
    ffmpeg_g = Path("FFmpeg/ffmpeg_g")
    if not ffmpeg_g.exists():
        sys.exit("FFmpeg debug binary not found at FFmpeg/ffmpeg_g")

    # Build lldb batch script (quote path for safety)
    quoted = str(h264_file).replace("\\", "\\\\").replace('"', '\\"')
    lines = [
        "settings set auto-confirm true",
        "b decode_residual",
        f'r -i "{quoted}" -vframes 1 -f null -',
    ]
    # Skip blocks
    for _ in range(skip_blocks):
        lines.append("c")
    # Extract positions
    for _ in range(num_blocks):
        lines.append("p (int)gb->index")
        lines.append("p (int)n")
        lines.append("p (int)sl->mb_x")
        lines.append("c")
    lines.append("q")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lldb", delete=False) as f:
        f.write("\n".join(lines))
        lldb_script = f.name

    try:
        result = subprocess.run(
            ["lldb", "-b", "-s", lldb_script, "--", str(ffmpeg_g)],
            capture_output=True, text=True, timeout=120,
        )
    finally:
        Path(lldb_script).unlink(missing_ok=True)

    # Parse output: lines like "(int) 1201" or "(int) $N = 1201"
    int_values = []
    for line in result.stdout.splitlines():
        m = re.search(r"\(int\)(?:\s+\$\d+\s*=)?\s+(-?\d+)", line)
        if m:
            int_values.append(int(m.group(1)))

    # Group into triples (gb.index, n, mb_x)
    blocks = []
    for i in range(0, len(int_values) - 2, 3):
        blocks.append((int_values[i], int_values[i + 1], int_values[i + 2]))
    return [(gb_idx, n) for gb_idx, n, _mb_x in blocks]


def extract_wedeo_positions(h264_file: str) -> list[tuple[int, int]]:
    """Extract (bits_consumed, nc) from wedeo's CAVLC trace for all residual blocks.

    Returns a list of (bit_position, nc) for each decode_residual call in the first frame.
    """
    wedeo_bin = None
    for p in ["target/debug/wedeo-framecrc"]:
        if Path(p).exists():
            wedeo_bin = p
            break
    if not wedeo_bin:
        # Try building
        subprocess.run(
            ["cargo", "build", "--bin", "wedeo-framecrc", "-p", "wedeo-fate", "--features", "tracing"],
            capture_output=True, check=True, timeout=120,
        )
        wedeo_bin = "target/debug/wedeo-framecrc"

    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::cavlc=trace"}
    result = subprocess.run(
        [wedeo_bin, h264_file],
        capture_output=True, text=True, env=env, timeout=120,
    )

    # Strip ANSI codes
    stderr = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)

    # Parse "CAVLC residual nc=X total_coeff=Y" lines from the first I-slice
    blocks = []
    in_first_slice = False
    for line in stderr.splitlines():
        if "slice_type=I" in line and "CAVLC residual" in line:
            in_first_slice = True
            m = re.search(r"nc=(-?\d+)\s+total_coeff=(\d+)", line)
            if m:
                nc = int(m.group(1))
                tc = int(m.group(2))
                blocks.append((nc, tc))
        elif in_first_slice and "slice_type=" in line and "slice_type=I" not in line:
            break  # Left the first I-slice

    return blocks


def compare(
    ffmpeg_blocks: list[tuple[int, int]],
    wedeo_blocks: list[tuple[int, int]],
) -> None:
    """Print FFmpeg bit positions and wedeo nc/total_coeff side by side.

    FFmpeg blocks: (gb.index, block_n)
    Wedeo blocks: (nc, total_coeff)

    Note: wedeo trace only gives nc and total_coeff per block, not exact bit
    positions.  Per-block bit consumption from FFmpeg (delta column) is the
    primary diagnostic — a sudden jump indicates desync.
    """
    print(f"FFmpeg: {len(ffmpeg_blocks)} blocks extracted")
    print(f"Wedeo: {len(wedeo_blocks)} residual blocks extracted")
    print()

    if not ffmpeg_blocks:
        print("No FFmpeg blocks — check that lldb breakpoint hit.")
        return

    n = max(len(ffmpeg_blocks), len(wedeo_blocks))
    hdr = f"{'Blk':>4} {'FFmpeg gb':>10} {'Adj(-8)':>8} {'n':>4} {'Delta':>6}"
    if wedeo_blocks:
        hdr += f" | {'w_nc':>5} {'w_tc':>5}"
    print(hdr)
    print("-" * len(hdr))

    prev = None
    for i in range(min(n, 60)):
        parts: list[str] = []
        if i < len(ffmpeg_blocks):
            gb_idx, blk_n = ffmpeg_blocks[i]
            adj = gb_idx - NAL_HEADER_BITS
            delta = gb_idx - prev if prev is not None else 0
            prev = gb_idx
            parts.append(f"{i:4d} {gb_idx:10d} {adj:8d} {blk_n:4d} {delta:6d}")
        else:
            parts.append(f"{i:4d} {'':>10} {'':>8} {'':>4} {'':>6}")

        if wedeo_blocks:
            if i < len(wedeo_blocks):
                nc, tc = wedeo_blocks[i]
                parts.append(f" | {nc:5d} {tc:5d}")
            else:
                parts.append(f" | {'':>5} {'':>5}")

        print("".join(parts))

    if n > 60:
        print(f"  ... ({n - 60} more blocks)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CAVLC bit positions between FFmpeg and wedeo",
    )
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument(
        "--num-blocks", type=int, default=30,
        help="Number of decode_residual calls to capture from FFmpeg (default: 30)",
    )
    parser.add_argument(
        "--skip-blocks", type=int, default=0,
        help="Skip this many decode_residual calls before capturing (default: 0)",
    )
    args = parser.parse_args()

    print(f"Extracting FFmpeg bit positions ({args.num_blocks} blocks, skip {args.skip_blocks})...")
    ffmpeg_blocks = extract_ffmpeg_positions(args.input, args.num_blocks, args.skip_blocks)

    print("Extracting wedeo residual trace...")
    wedeo_blocks = extract_wedeo_positions(args.input)

    compare(ffmpeg_blocks, wedeo_blocks)


if __name__ == "__main__":
    main()
