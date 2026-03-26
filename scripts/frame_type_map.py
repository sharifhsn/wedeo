#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Show decode order, display order, and frame types for an H.264 stream.

Parses wedeo's decoder log to extract slice types and POC values,
then maps decode order to display order.

Usage:
    python3 scripts/frame_type_map.py <input.264>
"""
import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/frame_type_map.py <input.264>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Find wedeo binary
    for profile in ("release", "debug"):
        candidate = Path("target") / profile / "wedeo-framecrc"
        if candidate.exists():
            wedeo_bin = str(candidate)
            break
    else:
        print("Error: wedeo-framecrc not found. Run cargo build.", file=sys.stderr)
        sys.exit(1)

    # Run with decoder-level debug logging
    env = {"RUST_LOG": "wedeo_codec_h264::decoder=debug", "PATH": "/usr/bin:/bin"}
    import os
    env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")

    result = subprocess.run(
        [wedeo_bin, str(input_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "RUST_LOG": "wedeo_codec_h264::decoder=debug"},
    )

    # Parse SLICE lines for frame_num, poc, slice_type
    # Tracing output has ANSI codes + key=value format
    slice_re = re.compile(
        r"SLICE\s+.*slice_type=(\w+)\s+.*frame_num=(\d+)\s+poc=(-?\d+)"
    )
    frames = []  # list of (decode_idx, slice_type, frame_num, poc)
    seen_pocs = set()

    combined = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
    for line in combined.split("\n"):
        m = slice_re.search(line)
        if m:
            stype = m.group(1)
            fnum = int(m.group(2))
            poc = int(m.group(3))
            if poc not in seen_pocs:
                seen_pocs.add(poc)
                frames.append((len(frames), stype, fnum, poc))

    if not frames:
        print("No frames found in decoder output.", file=sys.stderr)
        sys.exit(1)

    # Sort by POC for display order
    display_order = sorted(frames, key=lambda f: f[3])

    print(f"{'Display':>7} {'Decode':>6} {'Type':>4} {'POC':>5} {'FNum':>5}")
    print("-" * 35)
    for disp_idx, (dec_idx, stype, fnum, poc) in enumerate(display_order):
        print(f"{disp_idx:7d} {dec_idx:6d} {stype:>4} {poc:5d} {fnum:5d}")

    # Summary
    type_counts = {}
    for _, stype, _, _ in frames:
        type_counts[stype] = type_counts.get(stype, 0) + 1
    print(f"\nTotal: {len(frames)} frames ({', '.join(f'{v} {k}' for k, v in sorted(type_counts.items()))})")

    if any(f[3] != f[0] for f in display_order):
        print("NOTE: Display order differs from decode order (B-frames present)")
    else:
        print("NOTE: Display order = decode order (no reordering)")


if __name__ == "__main__":
    main()
