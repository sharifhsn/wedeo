#!/usr/bin/env python3
"""Verify wedeo's H.264 lookup tables against FFmpeg's C source.

Parses FFmpeg's C headers/source files and compares against wedeo's Rust
tables to catch transcription errors (missing/extra/wrong entries).

Usage:
    python3 scripts/verify_tables.py [--ffmpeg-dir FFmpeg]

Checks:
    - TC0_TABLE (deblock.rs vs h264_loopfilter.c tc0_table)
    - ALPHA_TABLE (deblock.rs vs h264_loopfilter.c alpha_table)
    - BETA_TABLE (deblock.rs vs h264_loopfilter.c beta_table)
    - CHROMA_QP_TABLE (tables.rs vs h264data.h ff_h264_chroma_qp)
"""

import argparse
import re
import sys
from pathlib import Path


def parse_c_array_1d(content: str, name: str, expected_len: int | None = None) -> list[int]:
    """Parse a 1D C array like `static const uint8_t name[N] = { ... };`."""
    pattern = rf'{re.escape(name)}\s*\[[^\]]*\]\s*=\s*\{{([^;]+)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array '{name}' in source")
    body = match.group(1)
    values = [int(x.strip()) for x in re.findall(r'-?\d+', body)]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"{name}: expected {expected_len} entries, got {len(values)}")
    return values


def parse_c_array_2d(content: str, name: str) -> list[list[int]]:
    """Parse a 2D C array like `static const uint8_t name[N][M] = { {a,b}, ... };`."""
    pattern = rf'{re.escape(name)}\s*\[[^\]]*\]\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array '{name}' in source")
    body = match.group(1)
    rows = re.findall(r'\{([^}]+)\}', body)
    return [[int(x.strip()) for x in row.split(',')] for row in rows]


def parse_rust_array_1d(content: str, name: str) -> list[int]:
    """Parse a 1D Rust const array like `const NAME: [T; N] = [ ... ];`."""
    pattern = rf'(?:pub\s+)?const\s+{re.escape(name)}\s*:\s*\[[^\]]+\]\s*=\s*\[(.*?)\];'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find const '{name}' in Rust source")
    body = match.group(1)
    return [int(x.strip()) for x in re.findall(r'-?\d+', body)]


def parse_rust_array_2d(content: str, name: str) -> list[list[int]]:
    """Parse a 2D Rust const array like `const NAME: [[T; M]; N] = [ [a,b], ... ];`."""
    pattern = rf'(?:pub\s+)?const\s+{re.escape(name)}\s*:\s*\[\[[^\]]+\];\s*\d+\]\s*=\s*\[(.*?)\];'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find const '{name}' in Rust source")
    body = match.group(1)
    rows = re.findall(r'\[([^\]]+)\]', body)
    return [[int(x.strip()) for x in row.split(',')] for row in rows]


def compare_arrays(name: str, expected: list, actual: list) -> int:
    """Compare two arrays, print mismatches. Returns number of errors."""
    errors = 0
    if len(expected) != len(actual):
        print(f"  LENGTH MISMATCH: FFmpeg has {len(expected)} entries, wedeo has {len(actual)}")
        errors += 1
    element_errors = 0
    for i in range(min(len(expected), len(actual))):
        if expected[i] != actual[i]:
            if element_errors == 0:
                print(f"  First mismatch at index {i}")
            if element_errors < 10:
                print(f"    [{i}]: FFmpeg={expected[i]}, wedeo={actual[i]}")
            element_errors += 1
    if element_errors > 10:
        print(f"    ... and {element_errors - 10} more mismatches")
    return errors + element_errors


def check_tc0_table(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify TC0_TABLE in deblock.rs against FFmpeg tc0_table."""
    print("Checking TC0_TABLE...")

    loopfilter = (ffmpeg_dir / "libavcodec" / "h264_loopfilter.c").read_text()
    deblock = (wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "deblock.rs").read_text()

    # FFmpeg: tc0_table[52*3][4], first 52 entries are padding, next 52 are QP 0-51
    ffmpeg_rows = parse_c_array_2d(loopfilter, "tc0_table")
    # Extract bS=1,2,3 columns (skip bS=0 which is always -1) for QP 0-51
    ffmpeg_tc0 = [row[1:4] for row in ffmpeg_rows[52:104]]

    # Wedeo: TC0_TABLE[52][3], indexed by QP, columns are bS=1,2,3
    wedeo_tc0 = parse_rust_array_2d(deblock, "TC0_TABLE")

    return compare_arrays("TC0_TABLE", ffmpeg_tc0, wedeo_tc0)


def check_alpha_table(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify ALPHA_TABLE in deblock.rs against FFmpeg alpha_table."""
    print("Checking ALPHA_TABLE...")

    loopfilter = (ffmpeg_dir / "libavcodec" / "h264_loopfilter.c").read_text()
    deblock = (wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "deblock.rs").read_text()

    ffmpeg_alpha = parse_c_array_1d(loopfilter, "alpha_table", 52 * 3)
    # QP 0-51 are at indices 52-103
    ffmpeg_alpha_qp = ffmpeg_alpha[52:104]

    wedeo_alpha = parse_rust_array_1d(deblock, "ALPHA_TABLE")

    return compare_arrays("ALPHA_TABLE", ffmpeg_alpha_qp, wedeo_alpha)


def check_beta_table(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify BETA_TABLE in deblock.rs against FFmpeg beta_table."""
    print("Checking BETA_TABLE...")

    loopfilter = (ffmpeg_dir / "libavcodec" / "h264_loopfilter.c").read_text()
    deblock = (wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "deblock.rs").read_text()

    ffmpeg_beta = parse_c_array_1d(loopfilter, "beta_table", 52 * 3)
    ffmpeg_beta_qp = ffmpeg_beta[52:104]

    wedeo_beta = parse_rust_array_1d(deblock, "BETA_TABLE")

    return compare_arrays("BETA_TABLE", ffmpeg_beta_qp, wedeo_beta)


def check_chroma_qp_table(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify CHROMA_QP_TABLE in tables.rs against FFmpeg ff_h264_chroma_qp."""
    print("Checking CHROMA_QP_TABLE...")

    # H.264 spec Table 8-15: QPC as a function of qPI (for 8-bit depth)
    spec_table = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34,
        35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
    ]

    # Try to parse from FFmpeg source to cross-validate.
    # ff_h264_chroma_qp is a 2D array [3][QP_MAX_NUM+1] in h264data.h.
    ffmpeg_qp_8bit = spec_table
    for candidate in ["h264data.h", "h264data.c"]:
        h264data = ffmpeg_dir / "libavcodec" / candidate
        if not h264data.exists():
            continue
        content = h264data.read_text()
        try:
            rows = parse_c_array_2d(content, "ff_h264_chroma_qp")
            ffmpeg_qp_8bit = rows[0][:52]  # first row = 8-bit depth
            break
        except ValueError:
            # May be macro-generated; fall back to spec table
            print("  Could not parse ff_h264_chroma_qp from source, using spec table")
            break

    tables = (wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "tables.rs").read_text()
    wedeo_qp = parse_rust_array_1d(tables, "CHROMA_QP_TABLE")

    return compare_arrays("CHROMA_QP_TABLE", ffmpeg_qp_8bit, wedeo_qp)


def main():
    parser = argparse.ArgumentParser(description="Verify H.264 lookup tables against FFmpeg")
    parser.add_argument("--ffmpeg-dir", type=Path, default=Path("FFmpeg"),
                        help="Path to FFmpeg source (default: FFmpeg)")
    args = parser.parse_args()

    wedeo_dir = Path(".")

    if not args.ffmpeg_dir.exists():
        print(f"Error: FFmpeg directory not found: {args.ffmpeg_dir}")
        sys.exit(1)

    total_errors = 0
    checks = [
        check_tc0_table,
        check_alpha_table,
        check_beta_table,
        check_chroma_qp_table,
    ]

    for check in checks:
        try:
            errors = check(args.ffmpeg_dir, wedeo_dir)
            if errors == 0:
                print("  OK")
            total_errors += errors
        except Exception as e:
            print(f"  ERROR: {e}")
            total_errors += 1
        print()

    if total_errors == 0:
        print(f"All {len(checks)} table checks passed!")
    else:
        print(f"FAILED: {total_errors} error(s) found")
        sys.exit(1)


if __name__ == "__main__":
    main()
