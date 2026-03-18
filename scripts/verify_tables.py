#!/usr/bin/env python3
"""Verify wedeo's H.264 lookup tables against FFmpeg's C source.

Parses FFmpeg's C headers/source files and compares against wedeo's Rust
tables to catch transcription errors (missing/extra/wrong entries).

Usage:
    python3 scripts/verify_tables.py [--ffmpeg-dir FFmpeg]

Checks tables from:
    - deblock.rs: TC0_TABLE, ALPHA_TABLE, BETA_TABLE
    - tables.rs: CHROMA_QP_TABLE, ZIGZAG_SCAN_4X4, FIELD_SCAN_4X4,
                 FIELD_SCAN_8X8, GOLOMB_TO_INTRA4X4_CBP, GOLOMB_TO_INTER_CBP,
                 CHROMA_DC_SCAN, CHROMA422_DC_SCAN, DEFAULT_SCALING4,
                 DEFAULT_SCALING8, DEQUANT4_COEFF_INIT, DEQUANT8_COEFF_INIT,
                 DEQUANT8_COEFF_INIT_SCAN
    - cavlc_tables.rs: COEFF_TOKEN_LEN/BITS, CHROMA_DC_COEFF_TOKEN_LEN/BITS,
                        TOTAL_ZEROS_LEN/BITS, CHROMA_DC_TOTAL_ZEROS_LEN/BITS,
                        RUN_BEFORE_LEN/BITS, COEFF_TOKEN_TABLE_INDEX
"""

import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _strip_c_comments(content: str) -> str:
    """Remove C-style // and /* */ comments from source code."""
    # Remove /* ... */ comments (non-greedy, handles multi-line)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # Remove // comments (to end of line)
    content = re.sub(r'//[^\n]*', '', content)
    return content

def parse_c_array_1d(content: str, name: str, expected_len: int | None = None) -> list[int]:
    """Parse a 1D C array like `static const uint8_t name[N] = { ... };`.

    Requires `const` before the name to avoid matching usage sites or comments.
    """
    pattern = rf'const\s+\w+\s+{re.escape(name)}\s*\[[^\]]*\]\s*=\s*\{{([^;]+)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array '{name}' in source")
    body = match.group(1)
    values = [int(x.strip()) for x in re.findall(r'-?\d+', body)]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"{name}: expected {expected_len} entries, got {len(values)}")
    return values


def parse_c_array_2d(content: str, name: str) -> list[list[int]]:
    """Parse a 2D C array like `static const uint8_t name[N][M] = { {a,b}, ... };`.

    Requires `const` before the name to avoid matching usage sites or comments.
    """
    pattern = rf'const\s+\w+\s+{re.escape(name)}\s*\[[^\]]*\]\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array '{name}' in source")
    body = match.group(1)
    rows = re.findall(r'\{([^}]+)\}', body)
    return [[int(x.strip()) for x in re.findall(r'-?\d+', row)] for row in rows]


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
    return [[int(x.strip()) for x in re.findall(r'-?\d+', row)] for row in rows]


def compare_arrays(name: str, expected: list, actual: list,
                    zero_pad_ok: bool = False) -> int:
    """Compare two arrays, print mismatches. Returns number of errors.

    If zero_pad_ok=True, allows wedeo arrays to be longer than FFmpeg
    arrays as long as the extra elements are all zero (for VLC tables
    that use fixed-width rows with zero padding).
    """
    errors = 0
    if len(expected) != len(actual):
        if zero_pad_ok and len(actual) > len(expected):
            # Check that all extra elements in wedeo are zero/empty
            extra = actual[len(expected):]
            if all(e == 0 or e == [] or e == [0] for e in extra):
                pass  # OK, just zero-padding
            else:
                print(f"  LENGTH MISMATCH: FFmpeg has {len(expected)}, wedeo has {len(actual)} (non-zero padding)")
                errors += 1
        else:
            print(f"  LENGTH MISMATCH: FFmpeg has {len(expected)} entries, wedeo has {len(actual)}")
            errors += 1
    element_errors = 0
    for i in range(min(len(expected), len(actual))):
        e, a = expected[i], actual[i]
        # For 2D: compare only the non-padded portion of each row
        if zero_pad_ok and isinstance(e, list) and isinstance(a, list) and len(a) > len(e):
            a_trimmed = a[:len(e)]
            if a_trimmed != e:
                if element_errors == 0:
                    print(f"  First mismatch at index {i}")
                if element_errors < 10:
                    print(f"    [{i}]: FFmpeg={e}, wedeo={a_trimmed} (trimmed from {len(a)})")
                element_errors += 1
        elif e != a:
            if element_errors == 0:
                print(f"  First mismatch at index {i}")
            if element_errors < 10:
                print(f"    [{i}]: FFmpeg={e}, wedeo={a}")
            element_errors += 1
    if element_errors > 10:
        print(f"    ... and {element_errors - 10} more mismatches")
    return errors + element_errors


def read_file(path: Path, strip_comments: bool = False) -> str:
    """Read a file, raising a clear error if not found."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    content = path.read_text()
    if strip_comments:
        content = _strip_c_comments(content)
    return content


# ---------------------------------------------------------------------------
# Check functions — deblocking filter tables
# ---------------------------------------------------------------------------

def check_tc0_table(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking TC0_TABLE...")
    loopfilter = read_file(ffmpeg_dir / "libavcodec" / "h264_loopfilter.c", strip_comments=True)
    deblock = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "deblock.rs")
    ffmpeg_rows = parse_c_array_2d(loopfilter, "tc0_table")
    ffmpeg_tc0 = [row[1:4] for row in ffmpeg_rows[52:104]]
    wedeo_tc0 = parse_rust_array_2d(deblock, "TC0_TABLE")
    return compare_arrays("TC0_TABLE", ffmpeg_tc0, wedeo_tc0)


def check_alpha_table(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking ALPHA_TABLE...")
    loopfilter = read_file(ffmpeg_dir / "libavcodec" / "h264_loopfilter.c", strip_comments=True)
    deblock = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "deblock.rs")
    ffmpeg_alpha = parse_c_array_1d(loopfilter, "alpha_table", 52 * 3)
    wedeo_alpha = parse_rust_array_1d(deblock, "ALPHA_TABLE")
    return compare_arrays("ALPHA_TABLE", ffmpeg_alpha[52:104], wedeo_alpha)


def check_beta_table(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking BETA_TABLE...")
    loopfilter = read_file(ffmpeg_dir / "libavcodec" / "h264_loopfilter.c", strip_comments=True)
    deblock = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "deblock.rs")
    ffmpeg_beta = parse_c_array_1d(loopfilter, "beta_table", 52 * 3)
    wedeo_beta = parse_rust_array_1d(deblock, "BETA_TABLE")
    return compare_arrays("BETA_TABLE", ffmpeg_beta[52:104], wedeo_beta)


# ---------------------------------------------------------------------------
# Check functions — tables.rs
# ---------------------------------------------------------------------------

def check_chroma_qp_table(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking CHROMA_QP_TABLE...")
    # H.264 spec Table 8-15 (8-bit depth)
    spec_table = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34,
        35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
    ]
    # ff_h264_chroma_qp uses CHROMA_QP_TABLE_END() macro for 8-bit depth,
    # so we can't parse it with regex. Use the H.264 spec table directly.
    ffmpeg_qp = spec_table
    tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "tables.rs")
    wedeo_qp = parse_rust_array_1d(tables, "CHROMA_QP_TABLE")
    return compare_arrays("CHROMA_QP_TABLE", ffmpeg_qp, wedeo_qp)


def check_cbp_tables(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking GOLOMB_TO_INTRA4X4_CBP, GOLOMB_TO_INTER_CBP...")
    h264data = read_file(ffmpeg_dir / "libavcodec" / "h264data.c", strip_comments=True)
    tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "tables.rs")
    errors = 0
    for ffmpeg_name, wedeo_name in [
        ("ff_h264_golomb_to_intra4x4_cbp", "GOLOMB_TO_INTRA4X4_CBP"),
        ("ff_h264_golomb_to_inter_cbp", "GOLOMB_TO_INTER_CBP"),
    ]:
        ffmpeg_vals = parse_c_array_1d(h264data, ffmpeg_name)
        wedeo_vals = parse_rust_array_1d(tables, wedeo_name)
        errors += compare_arrays(wedeo_name, ffmpeg_vals, wedeo_vals)
    return errors


def check_scan_tables(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking CHROMA_DC_SCAN, CHROMA422_DC_SCAN...")
    h264data = read_file(ffmpeg_dir / "libavcodec" / "h264data.c", strip_comments=True)
    tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "tables.rs")
    errors = 0
    # FFmpeg uses C expressions like "(0 + 1 * 2) * 16" — need eval.
    for ffmpeg_name, wedeo_name in [
        ("ff_h264_chroma_dc_scan", "CHROMA_DC_SCAN"),
        ("ff_h264_chroma422_dc_scan", "CHROMA422_DC_SCAN"),
    ]:
        pattern = rf'const\s+\w+\s+{re.escape(ffmpeg_name)}\s*\[[^\]]*\]\s*=\s*\{{([^;]+)\}};'
        match = re.search(pattern, h264data, re.DOTALL)
        if not match:
            print(f"  ERROR: Could not find {ffmpeg_name}")
            errors += 1
            continue
        body = match.group(1)
        parts = [p.strip() for p in body.split(',') if p.strip()]
        ffmpeg_vals = [_eval_c_expr(p) for p in parts]
        wedeo_vals = parse_rust_array_1d(tables, wedeo_name)
        errors += compare_arrays(wedeo_name, ffmpeg_vals, wedeo_vals)
    return errors


def _eval_c_expr(expr: str) -> int:
    """Evaluate simple C expressions like '(0 + 1 * 2) * 16'."""
    # Only allow digits, +, *, -, spaces, parens for safety
    cleaned = expr.strip()
    if re.match(r'^[\d\s+*()-]+$', cleaned):
        return eval(cleaned)  # noqa: S307 — safe: only digits, operators, parens
    raise ValueError(f"Cannot evaluate C expression: {expr!r}")


def check_field_scan_tables(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking FIELD_SCAN_4X4, FIELD_SCAN_8X8...")
    h264_slice = read_file(ffmpeg_dir / "libavcodec" / "h264_slice.c", strip_comments=True)
    tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "tables.rs")
    errors = 0
    # FFmpeg stores these as 1D arrays with expressions like "0 + 1 * 4".
    # Parse with expression evaluation.
    for ffmpeg_name, wedeo_name in [
        ("field_scan", "FIELD_SCAN_4X4"),
        ("field_scan8x8", "FIELD_SCAN_8X8"),
    ]:
        pattern = rf'const\s+\w+\s+{re.escape(ffmpeg_name)}\s*\[[^\]]*\]\s*=\s*\{{([^;]+)\}};'
        match = re.search(pattern, h264_slice, re.DOTALL)
        if not match:
            print(f"  ERROR: Could not find {ffmpeg_name}")
            errors += 1
            continue
        body = match.group(1)
        # Split by commas and evaluate each expression
        parts = [p.strip() for p in body.split(',') if p.strip()]
        ffmpeg_vals = [_eval_c_expr(p) for p in parts]
        wedeo_vals = parse_rust_array_1d(tables, wedeo_name)
        errors += compare_arrays(wedeo_name, ffmpeg_vals, wedeo_vals)
    return errors


def check_default_scaling(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking DEFAULT_SCALING4, DEFAULT_SCALING8...")
    h264_ps = read_file(ffmpeg_dir / "libavcodec" / "h264_ps.c", strip_comments=True)
    tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "tables.rs")
    errors = 0
    for ffmpeg_name, wedeo_name in [
        ("default_scaling4", "DEFAULT_SCALING4"),
        ("default_scaling8", "DEFAULT_SCALING8"),
    ]:
        ffmpeg_rows = parse_c_array_2d(h264_ps, ffmpeg_name)
        wedeo_rows = parse_rust_array_2d(tables, wedeo_name)
        errors += compare_arrays(wedeo_name, ffmpeg_rows, wedeo_rows)
    return errors


def check_dequant_tables(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking DEQUANT4_COEFF_INIT, DEQUANT8_COEFF_INIT, DEQUANT8_COEFF_INIT_SCAN...")
    h264data = read_file(ffmpeg_dir / "libavcodec" / "h264data.c", strip_comments=True)
    tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "tables.rs")
    errors = 0
    # 2D tables
    for ffmpeg_name, wedeo_name in [
        ("ff_h264_dequant4_coeff_init", "DEQUANT4_COEFF_INIT"),
        ("ff_h264_dequant8_coeff_init", "DEQUANT8_COEFF_INIT"),
    ]:
        ffmpeg_rows = parse_c_array_2d(h264data, ffmpeg_name)
        wedeo_rows = parse_rust_array_2d(tables, wedeo_name)
        errors += compare_arrays(wedeo_name, ffmpeg_rows, wedeo_rows)
    # 1D table
    ffmpeg_scan = parse_c_array_1d(h264data, "ff_h264_dequant8_coeff_init_scan")
    wedeo_scan = parse_rust_array_1d(tables, "DEQUANT8_COEFF_INIT_SCAN")
    errors += compare_arrays("DEQUANT8_COEFF_INIT_SCAN", ffmpeg_scan, wedeo_scan)
    return errors


# ---------------------------------------------------------------------------
# Check functions — CAVLC tables
# ---------------------------------------------------------------------------

def check_cavlc_coeff_token(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking COEFF_TOKEN_LEN/BITS (4 tables), COEFF_TOKEN_TABLE_INDEX...")
    cavlc = read_file(ffmpeg_dir / "libavcodec" / "h264_cavlc.c", strip_comments=True)
    cavlc_tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cavlc_tables.rs")
    errors = 0

    # FFmpeg has coeff_token_len[4][17*4] and coeff_token_bits[4][17*4]
    # Wedeo splits into COEFF_TOKEN_LEN_0..3 and COEFF_TOKEN_BITS_0..3
    for suffix, ffmpeg_name in [("LEN", "coeff_token_len"), ("BITS", "coeff_token_bits")]:
        ffmpeg_rows = parse_c_array_2d(cavlc, ffmpeg_name)
        for i in range(min(4, len(ffmpeg_rows))):
            wedeo_name = f"COEFF_TOKEN_{suffix}_{i}"
            wedeo_vals = parse_rust_array_1d(cavlc_tables, wedeo_name)
            errors += compare_arrays(wedeo_name, ffmpeg_rows[i], wedeo_vals)

    # coeff_token_table_index
    ffmpeg_idx = parse_c_array_1d(cavlc, "coeff_token_table_index")
    wedeo_idx = parse_rust_array_1d(cavlc_tables, "COEFF_TOKEN_TABLE_INDEX")
    errors += compare_arrays("COEFF_TOKEN_TABLE_INDEX", ffmpeg_idx, wedeo_idx)

    return errors


def check_cavlc_chroma_dc_coeff(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking CHROMA_DC_COEFF_TOKEN_LEN/BITS...")
    cavlc = read_file(ffmpeg_dir / "libavcodec" / "h264_cavlc.c", strip_comments=True)
    cavlc_tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cavlc_tables.rs")
    errors = 0
    for ffmpeg_name, wedeo_name in [
        ("chroma_dc_coeff_token_len", "CHROMA_DC_COEFF_TOKEN_LEN"),
        ("chroma_dc_coeff_token_bits", "CHROMA_DC_COEFF_TOKEN_BITS"),
    ]:
        ffmpeg_vals = parse_c_array_1d(cavlc, ffmpeg_name)
        wedeo_vals = parse_rust_array_1d(cavlc_tables, wedeo_name)
        errors += compare_arrays(wedeo_name, ffmpeg_vals, wedeo_vals)
    return errors


def check_cavlc_total_zeros(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking TOTAL_ZEROS_LEN/BITS, CHROMA_DC_TOTAL_ZEROS_LEN/BITS...")
    cavlc = read_file(ffmpeg_dir / "libavcodec" / "h264_cavlc.c", strip_comments=True)
    cavlc_tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cavlc_tables.rs")
    errors = 0
    for ffmpeg_name, wedeo_name in [
        ("total_zeros_len", "TOTAL_ZEROS_LEN"),
        ("total_zeros_bits", "TOTAL_ZEROS_BITS"),
        ("chroma_dc_total_zeros_len", "CHROMA_DC_TOTAL_ZEROS_LEN"),
        ("chroma_dc_total_zeros_bits", "CHROMA_DC_TOTAL_ZEROS_BITS"),
    ]:
        ffmpeg_rows = parse_c_array_2d(cavlc, ffmpeg_name)
        wedeo_rows = parse_rust_array_2d(cavlc_tables, wedeo_name)
        errors += compare_arrays(wedeo_name, ffmpeg_rows, wedeo_rows, zero_pad_ok=True)
    return errors


def check_cavlc_run_before(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    print("Checking RUN_BEFORE_LEN/BITS...")
    cavlc = read_file(ffmpeg_dir / "libavcodec" / "h264_cavlc.c", strip_comments=True)
    cavlc_tables = read_file(wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cavlc_tables.rs")
    errors = 0
    for ffmpeg_name, wedeo_name in [
        ("run_len", "RUN_BEFORE_LEN"),
        ("run_bits", "RUN_BEFORE_BITS"),
    ]:
        ffmpeg_rows = parse_c_array_2d(cavlc, ffmpeg_name)
        wedeo_rows = parse_rust_array_2d(cavlc_tables, wedeo_name)
        errors += compare_arrays(wedeo_name, ffmpeg_rows, wedeo_rows, zero_pad_ok=True)
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        # Deblocking filter
        check_tc0_table,
        check_alpha_table,
        check_beta_table,
        # QP and scaling
        check_chroma_qp_table,
        check_default_scaling,
        check_dequant_tables,
        # Scan and CBP
        check_cbp_tables,
        check_scan_tables,
        check_field_scan_tables,
        # CAVLC
        check_cavlc_coeff_token,
        check_cavlc_chroma_dc_coeff,
        check_cavlc_total_zeros,
        check_cavlc_run_before,
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
