#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Extract FFmpeg reconstruction data at a specific MB via lldb.

Extracts dest_y offset, linesize, mb_field_decoding_flag, neighbor pixels,
and final pixel rows for comparison with wedeo's INTRA_RAW_NEIGHBORS /
INTRA_FINAL_ROWS traces.

Usage:
    python3 scripts/ffmpeg_recon_extract.py CAMA1_Sony_C.jsv --mb 19,4 --frame 0
    python3 scripts/ffmpeg_recon_extract.py CAMA1_Sony_C.jsv --mb 19,4 --json
    python3 scripts/ffmpeg_recon_extract.py CAMA1_Sony_C.jsv --mb 19,4 --no-deblock

Requires:
    - FFmpeg debug build at FFmpeg/ffmpeg (--disable-optimizations --enable-debug=3 --disable-asm)
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def _parse_hex_bytes(lines: list[str]) -> list[int]:
    """Parse lldb hex memory read output lines into a flat list of ints."""
    vals = []
    for line in lines:
        m = re.match(r"\s*0x[0-9a-f]+:\s*(.*)", line.strip())
        if m:
            for tok in m.group(1).split():
                if tok.startswith("0x"):
                    vals.append(int(tok, 16))
    return vals


def _parse_expr_values(stdout: str) -> list[str]:
    """Extract all $N = VALUE results from lldb output."""
    vals = []
    for line in stdout.splitlines():
        m = re.search(r"\$\d+\s*=\s*(.+)", line)
        if m:
            vals.append(m.group(1).strip())
    return vals


def _expr_to_int(val: str) -> int:
    """Convert an lldb expression result string to int."""
    if val.startswith("'") and val.endswith("'") and len(val) == 3:
        return ord(val[1])
    m = re.search(r"(-?\d+)", val)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse int from: {val!r}")


def extract_via_subprocess(
    ffmpeg_bin: Path,
    input_file: Path,
    mb_x: int,
    mb_y: int,
    frame: int,
    no_deblock: bool = False,
) -> dict | None:
    """Extract reconstruction data using a single lldb pass.

    Uses hl_decode_mb_complex (MBAFF) or hl_decode_mb_simple_8 (progressive).
    Backtick expressions compute addresses at runtime to handle ASLR.
    """
    for func in ["hl_decode_mb_complex", "hl_decode_mb_simple_8"]:
        result = _try_extract(
            ffmpeg_bin, input_file, mb_x, mb_y, frame, func, no_deblock
        )
        if result is not None:
            return result
    return None


def _try_extract(
    ffmpeg_bin: Path,
    input_file: Path,
    mb_x: int,
    mb_y: int,
    frame: int,
    func_name: str,
    no_deblock: bool,
) -> dict | None:
    deblock_args = "-skip_loop_filter all " if no_deblock else ""

    # Build a single-pass lldb script using backtick expressions for addresses.
    # We assign the computed dest_y to a convenience variable, then use it.
    cmds = [
        f"target create {ffmpeg_bin}",
        f'breakpoint set -n {func_name} -c "sl->mb_x == {mb_x} && sl->mb_y == {mb_y}"',
    ]
    if frame > 0:
        cmds.append(f"breakpoint modify 1 -i {frame}")
    cmds.append("breakpoint modify 1 --one-shot true")
    cmds.append(
        f"run -threads 1 -bitexact {deblock_args}"
        f"-i {input_file} -f null /dev/null"
    )

    # Scalar values: $0..$4
    cmds.append("expression (int)sl->mb_field_decoding_flag")  # $0
    cmds.append("expression (int)sl->linesize")                # $1
    cmds.append("expression (long)h->cur_pic.f->data[0]")      # $2
    cmds.append("expression (int)sl->mb_x")                    # $3
    cmds.append("expression (int)sl->mb_y")                    # $4

    # Compute dest_y as a convenience variable for memory reads.
    # FFmpeg: dest_y = data[0] + (mb_x + mb_y * linesize) * 16
    # Field adjustment: linesize_eff = linesize * 2 if field, else linesize
    #                   dest_y -= linesize * 15 if field && (mb_y & 1)
    # We'll compute the field-adjusted dest_y and linesize using expressions.
    cmds.append(
        "expression unsigned char *$dest = "
        "(unsigned char *)(h->cur_pic.f->data[0] "
        "+ ((long)sl->mb_x + (long)sl->mb_y * sl->linesize) * 16 "
        "+ (sl->mb_field_decoding_flag && (sl->mb_y & 1) ? -sl->linesize * 15 : 0))"
    )  # $5
    cmds.append(
        "expression int $ls = sl->mb_field_decoding_flag ? sl->linesize * 2 : sl->linesize"
    )  # $6

    # Top row: 16 bytes at dest - linesize_eff
    cmds.append("memory read -s1 -c16 -fx `$dest - $ls`")
    # Top-left: 1 byte
    cmds.append("memory read -s1 -c1 -fx `$dest - $ls - 1`")
    # Left column: 16 bytes (each at dest + i*ls - 1)
    for i in range(16):
        cmds.append(f"memory read -s1 -c1 -fx `$dest + {i}*$ls - 1`")

    # Step out to get final pixels after decode
    cmds.append("finish")

    # Final pixel rows: first 4 rows at field stride
    for row in range(4):
        cmds.append(f"memory read -s1 -c16 -fx `$dest + {row}*$ls`")

    cmds.append("quit")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lldb", delete=False, prefix="recon_"
    ) as f:
        f.write("\n".join(cmds) + "\n")
        script_path = f.name

    try:
        r = subprocess.run(
            ["lldb", "--batch", "--source", script_path],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    finally:
        Path(script_path).unlink(missing_ok=True)

    # Parse expression results
    expr_vals = _parse_expr_values(r.stdout)
    if len(expr_vals) < 5:
        return None  # breakpoint not hit

    mb_field = _expr_to_int(expr_vals[0])
    sl_linesize = _expr_to_int(expr_vals[1])
    data0 = _expr_to_int(expr_vals[2])
    mb_x_val = _expr_to_int(expr_vals[3])
    mb_y_val = _expr_to_int(expr_vals[4])

    linesize = sl_linesize * 2 if mb_field else sl_linesize
    dest_y_initial = data0 + (mb_x_val + mb_y_val * sl_linesize) * 16
    dest_y = dest_y_initial
    if mb_field and (mb_y_val & 1):
        dest_y -= sl_linesize * 15
    offset = dest_y - data0

    # Parse memory reads
    mem_lines = [l for l in r.stdout.splitlines() if re.match(r"\s*0x[0-9a-f]+:", l)]

    # Layout: top_row (2 lines), top_left (1), left_col (16), final_rows (8 lines)
    # Total expected: 2 + 1 + 16 + 8 = 27 minimum
    if len(mem_lines) < 20:
        return None

    idx = 0
    # Top row: 16 bytes (typically 2 lines of 8)
    top_row = _parse_hex_bytes(mem_lines[idx:idx + 2])
    idx += 2 if len(top_row) >= 16 else 1
    top_row = top_row[:16]

    # Top-left: 1 byte
    tl = _parse_hex_bytes(mem_lines[idx:idx + 1])
    top_left = tl[0] if tl else 128
    idx += 1

    # Left column: 16 single-byte reads
    left_col = []
    for _ in range(16):
        b = _parse_hex_bytes(mem_lines[idx:idx + 1])
        left_col.append(b[0] if b else 0)
        idx += 1

    # Final rows: 4 rows of 16 bytes each (2 lines per row)
    final_rows = []
    for _ in range(4):
        row = _parse_hex_bytes(mem_lines[idx:idx + 2])
        idx += 2 if len(row) >= 16 else 1
        final_rows.append(row[:16])

    return {
        "mb_x": mb_x_val,
        "mb_y": mb_y_val,
        "mb_field": mb_field,
        "offset": offset,
        "linesize": linesize,
        "sl_linesize": sl_linesize,
        "top_row": top_row,
        "left_col": left_col,
        "top_left": top_left,
        "final_rows": final_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract FFmpeg reconstruction data at a specific MB"
    )
    parser.add_argument("file", help="Conformance file (name or path)")
    parser.add_argument("--mb", required=True, help="MB position as x,y")
    parser.add_argument("--frame", type=int, default=0, help="Frame number")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--no-deblock", action="store_true",
        help="Run FFmpeg with -skip_loop_filter all"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ffmpeg_debug import find_ffmpeg_binary, resolve_conformance_file

    ffmpeg_bin = find_ffmpeg_binary()
    input_file = resolve_conformance_file(args.file)

    parts = args.mb.split(",")
    mb_x, mb_y = int(parts[0]), int(parts[1])

    print(f"Extracting: MB({mb_x},{mb_y}) frame {args.frame}", file=sys.stderr)
    print(f"File: {input_file.name}", file=sys.stderr)
    print(f"FFmpeg: {ffmpeg_bin}", file=sys.stderr)

    data = extract_via_subprocess(
        ffmpeg_bin, input_file, mb_x, mb_y, args.frame, args.no_deblock
    )

    if data is None:
        print("Extraction failed.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"\n=== FFmpeg MB({data['mb_x']},{data['mb_y']}) ===")
        print(f"  mb_field={data['mb_field']}")
        print(f"  offset={data['offset']}  linesize={data['linesize']}  "
              f"sl_linesize={data['sl_linesize']}")
        print(f"  top_row={data['top_row']}")
        print(f"  left_col={data['left_col']}")
        print(f"  top_left={data['top_left']}")
        for i, row in enumerate(data.get("final_rows", [])):
            print(f"  final_row{i}={row}")


if __name__ == "__main__":
    main()
