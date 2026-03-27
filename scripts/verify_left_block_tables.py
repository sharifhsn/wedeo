#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Verify wedeo's left_block remapping tables against FFmpeg's left_block_options.

Parses FFmpeg's h264_mvpred.h for the left_block_options[4][32] table and
compares against wedeo's LEFT_LUMA_NZ_ROW, LEFT_CHROMA_NZ_ROW, and CBP_SHIFT
constants in cabac.rs.

Usage:
    python3 scripts/verify_left_block_tables.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_ffmpeg_table() -> list[list[int]]:
    """Parse left_block_options[4][32] from FFmpeg h264_mvpred.h."""
    mvpred = ROOT / "FFmpeg" / "libavcodec" / "h264_mvpred.h"
    if not mvpred.exists():
        print(f"ERROR: {mvpred} not found", file=sys.stderr)
        sys.exit(2)

    text = mvpred.read_text()
    # Find the table declaration
    match = re.search(
        r"left_block_options\[4\]\[32\]\s*=\s*\{(.*?)\};",
        text,
        re.DOTALL,
    )
    if not match:
        print("ERROR: could not find left_block_options table", file=sys.stderr)
        sys.exit(2)

    body = match.group(1)
    # Extract each row: { ... }
    rows = re.findall(r"\{([^}]+)\}", body)
    result = []
    for row in rows:
        # Evaluate C expressions like "3 + 2 * 4"
        vals = []
        for expr in row.split(","):
            expr = expr.strip()
            if expr:
                vals.append(eval(expr))  # noqa: S307 — trusted FFmpeg source
        result.append(vals)
    return result


def parse_rust_2d_array(text: str, name: str) -> list[list[int]]:
    """Parse a Rust const 2D array like `const NAME: [[u8; N]; M] = [...]`."""
    # Match from the `= [` to the closing `];`, skipping the type annotation
    m = re.search(rf"const {name}[^=]*=\s*\[(.*)\];", text, re.DOTALL)
    if not m:
        return []
    body = m.group(1)
    # Strip // comments to avoid matching numbers in comments
    body = re.sub(r"//[^\n]*", "", body)
    # Find row patterns: `[N, N, N, ...]` where N are integers
    rows = re.findall(r"\[(\d[\d\s,]*)\]", body)
    return [
        [int(x.strip()) for x in row.split(",") if x.strip()]
        for row in rows
    ]


def parse_wedeo_tables() -> dict:
    """Parse wedeo's remapping tables from cabac.rs."""
    cabac = ROOT / "codecs" / "wedeo-codec-h264" / "src" / "cabac.rs"
    text = cabac.read_text()

    return {
        "luma_nz": parse_rust_2d_array(text, "LEFT_LUMA_NZ_ROW"),
        "chroma_nz": parse_rust_2d_array(text, "LEFT_CHROMA_NZ_ROW"),
        "cbp_shift": parse_rust_2d_array(text, "CBP_SHIFT"),
    }


def verify():
    ffmpeg = parse_ffmpeg_table()
    wedeo = parse_wedeo_tables()
    errors = 0

    print("=== left_block_options verification ===\n")

    # Verify luma NNZ (entries 8-11)
    print("Luma NNZ row remap (entries 8-11):")
    for opt in range(4):
        ff_nz = [ffmpeg[opt][8 + i] for i in range(4)]
        # Convert from block index to row: block_idx = row * 4 + 3, so row = (idx - 3) / 4
        ff_rows = [(idx - 3) // 4 for idx in ff_nz]
        w_rows = wedeo["luma_nz"][opt]
        match = ff_rows == w_rows
        status = "OK" if match else "MISMATCH"
        if not match:
            errors += 1
        print(f"  opt {opt}: FFmpeg blocks={ff_nz} rows={ff_rows}  wedeo={w_rows}  {status}")

    print()

    # Verify chroma NNZ (entries 12-15)
    print("Chroma NNZ row remap (entries 12-15):")
    for opt in range(4):
        # Entries 12-15: [Cb_top, Cr_top, Cb_bot, Cr_bot]
        ff_chroma = [ffmpeg[opt][12 + i] for i in range(4)]
        # Cb: 4*4+col → row = (idx - 4*4) // 4;  1+4*4=17 → row 0, 1+5*4=21 → row 1
        cb_top_row = (ff_chroma[0] - 1) // 4 - 4  # e.g., 17 → (17-1)/4 - 4 = 0
        cb_bot_row = (ff_chroma[2] - 1) // 4 - 4
        ff_rows = [cb_top_row, cb_bot_row]
        w_rows = wedeo["chroma_nz"][opt]
        match = ff_rows == w_rows
        status = "OK" if match else "MISMATCH"
        if not match:
            errors += 1
        print(f"  opt {opt}: FFmpeg={ff_chroma} rows={ff_rows}  wedeo={w_rows}  {status}")

    print()

    # Verify CBP shifts (left_block[0] & ~1, left_block[2] & ~1)
    print("CBP bit shifts (left_block[0]&~1, left_block[2]&~1):")
    for opt in range(4):
        ff_shifts = [ffmpeg[opt][0] & ~1, ffmpeg[opt][2] & ~1]
        w_shifts = wedeo["cbp_shift"][opt]
        match = ff_shifts == w_shifts
        status = "OK" if match else "MISMATCH"
        if not match:
            errors += 1
        print(f"  opt {opt}: FFmpeg blocks[0,2]={ffmpeg[opt][0]},{ffmpeg[opt][2]} shifts={ff_shifts}  wedeo={w_shifts}  {status}")

    print()

    # Verify intra4x4 mode remap (entries 0-3, using 6-left_block[j] mapping)
    print("Intra4x4 mode row remap (entries 0-3, mode[6-j] → row):")
    for opt in range(4):
        ff_blocks = [ffmpeg[opt][i] for i in range(4)]
        # mode[6-j] gives right col row (6-j maps to stored position)
        # 6-0=6→row0, 6-1=5→row1, 6-2=4→row2, 6-3=3→row3
        ff_rows = [6 - j - 6 + j for j in ff_blocks]  # simplifies to just j...
        # Actually: mode[6-left_block[j]], and mode[6]=row0, mode[5]=row1, etc.
        # So the row is: left_block[j]
        ff_rows = ff_blocks
        w_rows = wedeo["luma_nz"][opt]  # same table used for intra4x4
        match = ff_rows == w_rows
        status = "OK" if match else "MISMATCH"
        if not match:
            errors += 1
        print(f"  opt {opt}: FFmpeg blocks={ff_blocks} rows={ff_rows}  wedeo={w_rows}  {status}")

    print()
    if errors == 0:
        print("All tables verified OK!")
    else:
        print(f"{errors} MISMATCHES found!")
    return errors


if __name__ == "__main__":
    sys.exit(verify())
