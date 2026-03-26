#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Extract FFmpeg deblocking filter inputs at a specific MB via lldb.

Breaks at filter_mb_dir for the target MB and direction, then reads
bS, QP, alpha, beta, and pixel values across each edge.

Usage:
    python3 scripts/deblock_edge_trace.py <h264_file> --mb-x 17 --mb-y 5
    python3 scripts/deblock_edge_trace.py <h264_file> --mb-x 17 --mb-y 5 --dir 1
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_lldb(input_path: str, mb_x: int, mb_y: int, direction: int) -> str | None:
    """Run lldb to extract deblock state from FFmpeg at the given MB."""
    ffmpeg_g = Path(__file__).resolve().parent.parent / "FFmpeg" / "ffmpeg_g"
    if not ffmpeg_g.exists():
        print(f"ERROR: {ffmpeg_g} not found. Build with --disable-asm.", file=sys.stderr)
        return None

    abs_input = str(Path(input_path).resolve())

    # We'll use an lldb script that:
    # 1. Breaks at filter_mb_dir entry when mb_x/mb_y/dir match
    # 2. Reads edges, bS, QP, img_y pixels across each horizontal edge
    #
    # For horizontal (dir=1) internal edges (edge 1,2,3), pixels are at:
    #   img_y + 4*edge*linesize + col  (q0)
    #   img_y + (4*edge-1)*linesize + col  (p0)

    # Write a Python lldb script to a temp file
    script = f'''
import lldb
import struct

def extract_deblock(debugger, command, result, internal_dict):
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()
    frame = thread.GetSelectedFrame()

    # Get variables
    mb_x_val = frame.FindVariable("mb_x").GetValueAsSigned()
    mb_y_val = frame.FindVariable("mb_y").GetValueAsSigned()
    dir_val = frame.FindVariable("dir").GetValueAsSigned()

    # Get h context for pixel_shift and mb_stride
    h = frame.FindVariable("h")
    linesize = frame.FindVariable("linesize").GetValueAsUnsigned()

    # Get QP
    mb_xy = frame.FindVariable("mb_xy").GetValueAsSigned()
    qp_expr = frame.EvaluateExpression(f"h->cur_pic.qscale_table[{{mb_xy}}]")
    qp = qp_expr.GetValueAsSigned() if qp_expr.IsValid() else -1

    # Get a, b offsets
    a_val = frame.FindVariable("a").GetValueAsSigned()
    b_val = frame.FindVariable("b").GetValueAsSigned()

    # Get edges count
    edges_val = frame.FindVariable("edges").GetValueAsSigned()

    # Get img_y pointer
    img_y = frame.FindVariable("img_y").GetValueAsUnsigned()

    print(f"\\nFFMPEG_DEBLOCK: mb_x={{mb_x_val}} mb_y={{mb_y_val}} dir={{dir_val}} qp={{qp}} a={{a_val}} b={{b_val}} linesize={{linesize}} edges={{edges_val}} mb_xy={{mb_xy}}")

    # For each edge, extract bS and pixel values
    for edge in range(4):
        # Read bS for this edge by stepping to it
        # Instead, we'll read the pixels at the edge boundary directly
        # For horizontal edges: boundary is between row 4*edge-1 and 4*edge
        pix_row = 4 * edge
        if pix_row == 0:
            # Edge 0 reads from above MB
            continue

        print(f"\\n  Edge {{edge}}: row={{pix_row}}")
        # Read 16 pixels: p2, p1, p0, q0, q1, q2 for columns 0..15
        for col in range(16):
            # q0 = img_y[pix_row * linesize + col]
            # p0 = img_y[(pix_row-1) * linesize + col]
            # etc.
            addrs = []
            labels = ["p2", "p1", "p0", "q0", "q1", "q2"]
            offsets = [pix_row-3, pix_row-2, pix_row-1, pix_row, pix_row+1, pix_row+2]
            vals = []
            for off in offsets:
                addr = img_y + off * linesize + col
                err = lldb.SBError()
                val = process.ReadMemory(addr, 1, err)[0]
                vals.append(val)
            print(f"    col={{col:2d}}: p2={{vals[0]:3d}} p1={{vals[1]:3d}} p0={{vals[2]:3d}} | q0={{vals[3]:3d}} q1={{vals[4]:3d}} q2={{vals[5]:3d}}")

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand("command script add -f deblock_extract.extract_deblock deblock_extract")
'''

    script_path = Path("/tmp/deblock_extract.py")
    script_path.write_text(script)

    # Build lldb commands
    cmds = [
        f'command script import {script_path}',
        # Break at the start of the edge loop (line 621 — "for(edge = start; edge < edges; edge++)")
        # Actually, let's break at the entry of filter_mb_dir with condition
        f'br s -n filter_mb_dir -c "mb_x == {mb_x} && mb_y == {mb_y} && dir == {direction}"',
        f"r -bitexact -i {abs_input} -f null /dev/null",
        # At breakpoint: extract info
        "deblock_extract",
        # Now step to read bS values for each edge
        # Instead, let's just read the non_zero_count_cache and compute bS
        # Read non_zero_count_cache around the luma area
        'expression -f d -- (int)sl->non_zero_count_cache[12]',
        'expression -f d -- (int)sl->non_zero_count_cache[13]',
        'expression -f d -- (int)sl->non_zero_count_cache[14]',
        'expression -f d -- (int)sl->non_zero_count_cache[15]',
        'expression -f d -- (int)sl->non_zero_count_cache[20]',
        'expression -f d -- (int)sl->non_zero_count_cache[21]',
        'expression -f d -- (int)sl->non_zero_count_cache[22]',
        'expression -f d -- (int)sl->non_zero_count_cache[23]',
        'expression -f d -- (int)sl->non_zero_count_cache[28]',
        'expression -f d -- (int)sl->non_zero_count_cache[29]',
        'expression -f d -- (int)sl->non_zero_count_cache[30]',
        'expression -f d -- (int)sl->non_zero_count_cache[31]',
        'expression -f d -- (int)sl->non_zero_count_cache[36]',
        'expression -f d -- (int)sl->non_zero_count_cache[37]',
        'expression -f d -- (int)sl->non_zero_count_cache[38]',
        'expression -f d -- (int)sl->non_zero_count_cache[39]',
        # Also get mask_edge, edges, mbm_type, mb_type
        'expression -f hex -- (int)h->cur_pic.mb_type[mb_xy]',
        'expression -f d -- (int)edges',
        'expression -f d -- (int)mask_edge',
        'expression -f d -- (int)mvy_limit',
        'expression -f d -- (int)IS_INTERLACED(h->cur_pic.mb_type[mb_xy])',
        # Get sl->mb_field_decoding_flag
        'expression -f d -- (int)sl->mb_field_decoding_flag',
        "kill",
        "q",
    ]

    lldb_args = ["lldb", "-b"]
    for c in cmds:
        lldb_args += ["-o", c]
    lldb_args += ["--", str(ffmpeg_g)]

    try:
        proc = subprocess.run(
            lldb_args,
            capture_output=True,
            timeout=60,
            cwd=str(ffmpeg_g.parent),
        )
    except subprocess.TimeoutExpired:
        print("ERROR: lldb timed out", file=sys.stderr)
        return None

    return proc.stdout.decode(errors="replace")


def parse_output(output: str) -> None:
    """Parse and display lldb output."""
    # Print FFMPEG_DEBLOCK lines
    for line in output.splitlines():
        if "FFMPEG_DEBLOCK" in line or "Edge " in line or "col" in line:
            print(line.strip())

    # Print expression results
    print("\n--- Expression results ---")
    # scan8 layout for luma NNZ cache:
    # Row 0: indices 12,13,14,15
    # Row 1: indices 20,21,22,23
    # Row 2: indices 28,29,30,31
    # Row 3: indices 36,37,38,39
    vals = re.findall(r"\$(\d+) = (-?\d+|0x[0-9a-f]+)", output)
    if vals:
        nnz_labels = [
            "nnz[12]", "nnz[13]", "nnz[14]", "nnz[15]",
            "nnz[20]", "nnz[21]", "nnz[22]", "nnz[23]",
            "nnz[28]", "nnz[29]", "nnz[30]", "nnz[31]",
            "nnz[36]", "nnz[37]", "nnz[38]", "nnz[39]",
            "mb_type", "edges", "mask_edge", "mvy_limit",
            "IS_INTERLACED", "mb_field",
        ]
        for i, (var_id, val) in enumerate(vals):
            label = nnz_labels[i] if i < len(nnz_labels) else f"${var_id}"
            print(f"  {label} = {val}")


def main():
    parser = argparse.ArgumentParser(description="Extract FFmpeg deblock state via lldb")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--mb-x", type=int, required=True)
    parser.add_argument("--mb-y", type=int, required=True)
    parser.add_argument("--dir", type=int, default=1, help="0=vertical, 1=horizontal")
    args = parser.parse_args()

    print(f"Extracting FFmpeg deblock state for MB({args.mb_x},{args.mb_y}) dir={args.dir}...")
    output = run_lldb(args.input, args.mb_x, args.mb_y, args.dir)
    if output is None:
        sys.exit(1)

    parse_output(output)

    # Also dump raw output for debugging
    raw_path = Path("/tmp/deblock_lldb_raw.log")
    raw_path.write_text(output)
    print(f"\nRaw lldb output saved to {raw_path}")


if __name__ == "__main__":
    main()
