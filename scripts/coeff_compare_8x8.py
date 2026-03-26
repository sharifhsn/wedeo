#!/usr/bin/env python3
# /// script
# dependencies = ["numpy"]
# ///
"""Compare 8x8 dequantized coefficients between wedeo and FFmpeg for a specific MB.

Extracts coefficients from both decoders for a given (mb_x, mb_y, i8x8) and
compares element-by-element after un-transposing FFmpeg's storage.

Prerequisites:
- FFmpeg built with FFCABAC8x8 fprintf trace in h264_cabac.c
  (see the decode_cabac_residual_nondc IS_8x8DCT block)
- Wedeo built with INTRA8x8_DEQUANT or DEQUANT trace including coeffs

Usage:
    python3 scripts/coeff_compare_8x8.py <ffmpeg_log> <wedeo_log> --mb-x X --mb-y Y --i8x8 N --occurrence K
"""
import argparse
import re
import sys


def parse_ffmpeg_blocks(log_path: str, mb_x: int, mb_y: int, i8x8: int) -> list[list[int]]:
    """Extract coefficient arrays from FFmpeg FFCABAC8x8 log lines."""
    pattern = re.compile(
        rf"FFCABAC8x8 mb_x={mb_x} mb_y={mb_y} i8x8={i8x8} "
        r"cqm=\d+ qp=\d+ sum=\d+ dc=[-\d]+ coeffs=\[([-\d,]+)\]"
    )
    results = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                coeffs = [int(x) for x in m.group(1).split(",")]
                if len(coeffs) == 64:
                    results.append(coeffs)
    return results


def parse_wedeo_blocks(log_path: str, mb_x: int, mb_y: int, i8x8: int) -> list[list[int]]:
    """Extract coefficient arrays from wedeo DEQUANT/INTRA8x8_DEQUANT log lines."""
    pattern = re.compile(
        rf"mb_x={mb_x}.*mb_y={mb_y}.*block_idx={i8x8}.*coeffs=\[([-\d, ]+)\]"
    )
    results = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                coeffs = [int(x.strip()) for x in m.group(1).split(",")]
                if len(coeffs) == 64:
                    results.append(coeffs)
    return results


def untranspose_8x8(transposed: list[int]) -> list[int]:
    """Convert FFmpeg's transposed (column-major) storage to row-major."""
    row_major = [0] * 64
    for x in range(64):
        t = (x >> 3) | ((x & 7) << 3)
        row_major[x] = transposed[t]
    return row_major


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 8x8 coefficients between decoders")
    parser.add_argument("ffmpeg_log", help="FFmpeg log with FFCABAC8x8 traces")
    parser.add_argument("wedeo_log", help="Wedeo log with DEQUANT traces")
    parser.add_argument("--mb-x", type=int, required=True)
    parser.add_argument("--mb-y", type=int, required=True)
    parser.add_argument("--i8x8", type=int, required=True)
    parser.add_argument("--occurrence", type=int, default=1,
                        help="Which occurrence (1=first, 2=P-frame for I+P streams)")
    args = parser.parse_args()

    ff_blocks = parse_ffmpeg_blocks(args.ffmpeg_log, args.mb_x, args.mb_y, args.i8x8)
    w_blocks = parse_wedeo_blocks(args.wedeo_log, args.mb_x, args.mb_y, args.i8x8)

    idx = args.occurrence - 1
    if idx >= len(ff_blocks):
        print(f"Only {len(ff_blocks)} FFmpeg occurrences found (requested {args.occurrence})")
        sys.exit(1)
    if idx >= len(w_blocks):
        print(f"Only {len(w_blocks)} wedeo occurrences found (requested {args.occurrence})")
        sys.exit(1)

    ff_transposed = ff_blocks[idx]
    ff_row_major = untranspose_8x8(ff_transposed)
    w_row_major = w_blocks[idx]

    print(f"Occurrence {args.occurrence} of MB({args.mb_x},{args.mb_y}) i8x8={args.i8x8}")
    print(f"FFmpeg abs_sum={sum(abs(c) for c in ff_row_major)}, wedeo abs_sum={sum(abs(c) for c in w_row_major)}")

    diffs = 0
    for r in range(8):
        ff_row = ff_row_major[r * 8:(r + 1) * 8]
        w_row = w_row_major[r * 8:(r + 1) * 8]
        match = "MATCH" if ff_row == w_row else "DIFF"
        if ff_row != w_row:
            diffs += 1
            print(f"  Row {r}: ff={ff_row} w={w_row} diff={[a-b for a,b in zip(ff_row, w_row)]}")
        else:
            print(f"  Row {r}: {ff_row} {match}")

    if diffs == 0:
        print("\nAll coefficients match.")
    else:
        print(f"\n{diffs} rows differ.")


if __name__ == "__main__":
    main()
