#!/usr/bin/env python3
"""Verify wedeo's CABAC tables against FFmpeg's C source.

Parses FFmpeg's C source files and compares against wedeo's Rust CABAC tables
to catch transcription errors (missing/extra/wrong entries).

Usage:
    python3 scripts/verify_cabac_tables.py [--ffmpeg-dir FFmpeg] [--verbose]

Checks tables from:
    - cabac_tables.rs: NORM_SHIFT, LPS_RANGE, MLPS_STATE, LAST_COEFF_FLAG_OFFSET_8X8,
                       CABAC_CONTEXT_INIT_I, CABAC_CONTEXT_INIT_PB0/1/2,
                       SIGNIFICANT_COEFF_FLAG_OFFSET, LAST_COEFF_FLAG_OFFSET,
                       COEFF_ABS_LEVEL_M1_OFFSET, SIGNIFICANT_COEFF_FLAG_OFFSET_8X8,
                       SIG_COEFF_OFFSET_DC, COEFF_ABS_LEVEL1_CTX,
                       COEFF_ABS_LEVELGT1_CTX, COEFF_ABS_LEVEL_TRANSITION,
                       init_cabac_states (functional)
    - cabac.rs: CBF_CTX_BASE
"""

import argparse
import re
import sys
from pathlib import Path

# Import helpers from verify_tables.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_tables import (
    _strip_c_comments,
    compare_arrays,
    parse_rust_array_1d,
    parse_rust_array_2d,
    read_file,
)

VERBOSE = False


def _eval_c_expr(expr: str) -> int:
    """Evaluate simple C expressions like '105+15' or '484+29'."""
    cleaned = expr.strip()
    if re.match(r'^[\d\s+*()-]+$', cleaned):
        return eval(cleaned)  # noqa: S307 — safe: only digits, operators, parens
    raise ValueError(f"Cannot evaluate C expression: {expr!r}")


def _parse_packed_cabac_table(content: str) -> list[int]:
    """Parse ff_h264_cabac_tables from cabac.c.

    This is a single packed array with signed int8 values stored as uint8_t.
    The C source uses negative literals (e.g., -128) which wrap to unsigned.
    We parse all integer values and return them as-is (signed).
    """
    pattern = r'ff_h264_cabac_tables\)\s*\[[^\]]+\]\s*=\s*\{(.*?)\};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError("Could not find ff_h264_cabac_tables in source")
    body = match.group(1)
    return [int(x.strip()) for x in re.findall(r'-?\d+', body)]


def _parse_c_context_init_I(content: str) -> list[list[int]]:
    """Parse cabac_context_init_I[1024][2] from h264_cabac.c."""
    pattern = r'cabac_context_init_I\s*\[\s*1024\s*\]\s*\[\s*2\s*\]\s*=\s*\{(.*?)\};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError("Could not find cabac_context_init_I in source")
    body = match.group(1)
    pairs = re.findall(r'\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}', body)
    return [[int(m), int(n)] for m, n in pairs]


def _parse_c_context_init_PB(content: str) -> list[list[list[int]]]:
    """Parse cabac_context_init_PB[3][1024][2] from h264_cabac.c.

    Returns a list of 3 tables, each containing 1024 [m, n] pairs.
    """
    # Find the entire 3D array
    pattern = r'cabac_context_init_PB\s*\[\s*3\s*\]\s*\[\s*1024\s*\]\s*\[\s*2\s*\]\s*=\s*\{(.*?)\};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError("Could not find cabac_context_init_PB in source")
    body = match.group(1)

    # All {m, n} pairs across the entire 3D array
    all_pairs = re.findall(r'\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}', body)
    if len(all_pairs) != 3 * 1024:
        raise ValueError(
            f"cabac_context_init_PB: expected {3 * 1024} pairs, got {len(all_pairs)}"
        )

    result = []
    for idc in range(3):
        start = idc * 1024
        table = [[int(m), int(n)] for m, n in all_pairs[start : start + 1024]]
        result.append(table)
    return result


def _parse_rust_context_init(content: str, name: str) -> list[list[int]]:
    """Parse a Rust const like `pub const NAME: [[i8; 2]; 1024] = [...]`."""
    pattern = (
        rf'(?:pub\s+)?const\s+{re.escape(name)}\s*:\s*'
        r'\[\[i8;\s*2\];\s*1024\]\s*=\s*\[(.*?)\];'
    )
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find const '{name}' in Rust source")
    body = match.group(1)
    pairs = re.findall(r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', body)
    return [[int(m), int(n)] for m, n in pairs]


def _parse_c_residual_table_2d_row0(content: str, name: str) -> list[int]:
    """Parse a static 2D C array inside a function and return row [0].

    Handles C expressions like '105+15'.
    """
    pattern = rf'static\s+const\s+\w+\s+{re.escape(name)}\s*\[\s*2\s*\]\s*\[\s*\d+\s*\]\s*=\s*\{{(.*?)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find static array '{name}' in source")
    body = match.group(1)
    # Find the two rows: { ... }, { ... }
    rows = re.findall(r'\{([^}]+)\}', body)
    if len(rows) < 1:
        raise ValueError(f"Could not find rows in '{name}'")
    # Parse row 0 with expression evaluation
    parts = [p.strip() for p in rows[0].split(',') if p.strip()]
    return [_eval_c_expr(p) for p in parts]


def _parse_c_residual_table_1d(content: str, name: str) -> list[int]:
    """Parse a static 1D C array inside a function.

    Handles C expressions like '227+10'.
    """
    pattern = rf'static\s+const\s+\w+\s+{re.escape(name)}\s*\[\s*\d+\s*\]\s*=\s*\{{([^;]+)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find static array '{name}' in source")
    body = match.group(1)
    parts = [p.strip() for p in body.split(',') if p.strip()]
    return [_eval_c_expr(p) for p in parts]


def _parse_c_residual_table_2d(content: str, name: str) -> list[list[int]]:
    """Parse a static 2D C array inside a function, returning all rows."""
    pattern = rf'static\s+const\s+\w+\s+{re.escape(name)}\s*\[\s*\d+\s*\]\s*\[\s*\d+\s*\]\s*=\s*\{{(.*?)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find static array '{name}' in source")
    body = match.group(1)
    rows = re.findall(r'\{([^}]+)\}', body)
    return [[int(x.strip()) for x in re.findall(r'-?\d+', row)] for row in rows]


def _parse_c_inline_static_1d(content: str, name: str) -> list[int]:
    """Parse a static const array that may be inside a function body."""
    pattern = rf'static\s+const\s+\w+\s+{re.escape(name)}\s*\[\s*\d+\s*\]\s*=\s*\{{([^;]+)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find static array '{name}' in source")
    body = match.group(1)
    return [int(x.strip()) for x in re.findall(r'-?\d+', body)]


# ---------------------------------------------------------------------------
# Section 1: Engine tables from cabac.c
# ---------------------------------------------------------------------------

def check_norm_shift(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify NORM_SHIFT (512 entries) against ff_h264_cabac_tables[0..511]."""
    print("Checking NORM_SHIFT...")
    cabac_c = read_file(ffmpeg_dir / "libavcodec" / "cabac.c", strip_comments=True)
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    packed = _parse_packed_cabac_table(cabac_c)
    ffmpeg_norm_shift = packed[0:512]
    wedeo_norm_shift = parse_rust_array_1d(cabac_tables_rs, "NORM_SHIFT")

    errors = compare_arrays("NORM_SHIFT", ffmpeg_norm_shift, wedeo_norm_shift)
    if errors == 0:
        print(f"  \u2713 NORM_SHIFT ({len(wedeo_norm_shift)} entries)")
    return errors


def check_lps_range(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify LPS_RANGE (512 entries) against ff_h264_cabac_tables[512..1023].

    C stores as signed int8 (negative values wrap), Rust stores as u8.
    Convert: v if v >= 0 else v + 256.
    """
    print("Checking LPS_RANGE...")
    cabac_c = read_file(ffmpeg_dir / "libavcodec" / "cabac.c", strip_comments=True)
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    packed = _parse_packed_cabac_table(cabac_c)
    ffmpeg_lps_raw = packed[512:1024]
    # Convert signed C values to unsigned u8
    ffmpeg_lps = [v if v >= 0 else v + 256 for v in ffmpeg_lps_raw]
    wedeo_lps = parse_rust_array_1d(cabac_tables_rs, "LPS_RANGE")

    if VERBOSE:
        for i, (e, a) in enumerate(zip(ffmpeg_lps, wedeo_lps)):
            if e != a:
                print(f"  [{i}]: FFmpeg={e} (raw={ffmpeg_lps_raw[i]}), wedeo={a}")

    errors = compare_arrays("LPS_RANGE", ffmpeg_lps, wedeo_lps)
    if errors == 0:
        print(f"  \u2713 LPS_RANGE ({len(wedeo_lps)} entries)")
    return errors


def check_mlps_state(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify MLPS_STATE (256 entries) against ff_h264_cabac_tables[1024..1279]."""
    print("Checking MLPS_STATE...")
    cabac_c = read_file(ffmpeg_dir / "libavcodec" / "cabac.c", strip_comments=True)
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    packed = _parse_packed_cabac_table(cabac_c)
    ffmpeg_mlps = packed[1024:1280]
    wedeo_mlps = parse_rust_array_1d(cabac_tables_rs, "MLPS_STATE")

    errors = compare_arrays("MLPS_STATE", ffmpeg_mlps, wedeo_mlps)
    if errors == 0:
        print(f"  \u2713 MLPS_STATE ({len(wedeo_mlps)} entries)")
    return errors


def check_last_coeff_flag_offset_8x8(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify LAST_COEFF_FLAG_OFFSET_8X8 (63 entries) against ff_h264_cabac_tables[1280..1342]."""
    print("Checking LAST_COEFF_FLAG_OFFSET_8X8...")
    cabac_c = read_file(ffmpeg_dir / "libavcodec" / "cabac.c", strip_comments=True)
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    packed = _parse_packed_cabac_table(cabac_c)
    ffmpeg_last8x8 = packed[1280:1343]
    wedeo_last8x8 = parse_rust_array_1d(cabac_tables_rs, "LAST_COEFF_FLAG_OFFSET_8X8")

    errors = compare_arrays(
        "LAST_COEFF_FLAG_OFFSET_8X8", ffmpeg_last8x8, wedeo_last8x8
    )
    if errors == 0:
        print(f"  \u2713 LAST_COEFF_FLAG_OFFSET_8X8 ({len(wedeo_last8x8)} entries)")
    return errors


# ---------------------------------------------------------------------------
# Section 2: Context init tables
# ---------------------------------------------------------------------------

def check_context_init_I(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify CABAC_CONTEXT_INIT_I (1024 x [i8;2]) against cabac_context_init_I."""
    print("Checking CABAC_CONTEXT_INIT_I...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_pairs = _parse_c_context_init_I(h264_cabac_c)
    wedeo_pairs = _parse_rust_context_init(cabac_tables_rs, "CABAC_CONTEXT_INIT_I")

    if len(ffmpeg_pairs) != 1024:
        print(f"  WARNING: FFmpeg cabac_context_init_I has {len(ffmpeg_pairs)} pairs, expected 1024")

    errors = compare_arrays("CABAC_CONTEXT_INIT_I", ffmpeg_pairs, wedeo_pairs)
    if errors == 0:
        print(f"  \u2713 CABAC_CONTEXT_INIT_I ({len(wedeo_pairs)} entries)")
    return errors


def check_context_init_PB(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify CABAC_CONTEXT_INIT_PB0/1/2 against cabac_context_init_PB[3]."""
    print("Checking CABAC_CONTEXT_INIT_PB0/1/2...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_pb = _parse_c_context_init_PB(h264_cabac_c)
    errors = 0

    for idc in range(3):
        name = f"CABAC_CONTEXT_INIT_PB{idc}"
        wedeo_pairs = _parse_rust_context_init(cabac_tables_rs, name)
        e = compare_arrays(name, ffmpeg_pb[idc], wedeo_pairs)
        if e == 0:
            print(f"  \u2713 {name} ({len(wedeo_pairs)} entries)")
        errors += e

    return errors


# ---------------------------------------------------------------------------
# Section 3: Residual offset tables
# ---------------------------------------------------------------------------

def check_significant_coeff_flag_offset(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify SIGNIFICANT_COEFF_FLAG_OFFSET (14 entries, frame mode = row [0])."""
    print("Checking SIGNIFICANT_COEFF_FLAG_OFFSET...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_vals = _parse_c_residual_table_2d_row0(
        h264_cabac_c, "significant_coeff_flag_offset"
    )
    wedeo_vals = parse_rust_array_1d(
        cabac_tables_rs, "SIGNIFICANT_COEFF_FLAG_OFFSET"
    )

    errors = compare_arrays(
        "SIGNIFICANT_COEFF_FLAG_OFFSET", ffmpeg_vals, wedeo_vals
    )
    if errors == 0:
        print(f"  \u2713 SIGNIFICANT_COEFF_FLAG_OFFSET ({len(wedeo_vals)} entries)")
    return errors


def check_last_coeff_flag_offset(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify LAST_COEFF_FLAG_OFFSET (14 entries, frame mode = row [0])."""
    print("Checking LAST_COEFF_FLAG_OFFSET...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_vals = _parse_c_residual_table_2d_row0(
        h264_cabac_c, "last_coeff_flag_offset"
    )
    wedeo_vals = parse_rust_array_1d(cabac_tables_rs, "LAST_COEFF_FLAG_OFFSET")

    errors = compare_arrays("LAST_COEFF_FLAG_OFFSET", ffmpeg_vals, wedeo_vals)
    if errors == 0:
        print(f"  \u2713 LAST_COEFF_FLAG_OFFSET ({len(wedeo_vals)} entries)")
    return errors


def check_coeff_abs_level_m1_offset(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify COEFF_ABS_LEVEL_M1_OFFSET (14 entries)."""
    print("Checking COEFF_ABS_LEVEL_M1_OFFSET...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_vals = _parse_c_residual_table_1d(h264_cabac_c, "coeff_abs_level_m1_offset")
    wedeo_vals = parse_rust_array_1d(cabac_tables_rs, "COEFF_ABS_LEVEL_M1_OFFSET")

    errors = compare_arrays("COEFF_ABS_LEVEL_M1_OFFSET", ffmpeg_vals, wedeo_vals)
    if errors == 0:
        print(f"  \u2713 COEFF_ABS_LEVEL_M1_OFFSET ({len(wedeo_vals)} entries)")
    return errors


def check_significant_coeff_flag_offset_8x8(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify SIGNIFICANT_COEFF_FLAG_OFFSET_8X8 (63 entries, frame mode = row [0])."""
    print("Checking SIGNIFICANT_COEFF_FLAG_OFFSET_8X8...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_vals = _parse_c_residual_table_2d_row0(
        h264_cabac_c, "significant_coeff_flag_offset_8x8"
    )
    wedeo_vals = parse_rust_array_1d(
        cabac_tables_rs, "SIGNIFICANT_COEFF_FLAG_OFFSET_8X8"
    )

    errors = compare_arrays(
        "SIGNIFICANT_COEFF_FLAG_OFFSET_8X8", ffmpeg_vals, wedeo_vals
    )
    if errors == 0:
        print(
            f"  \u2713 SIGNIFICANT_COEFF_FLAG_OFFSET_8X8 ({len(wedeo_vals)} entries)"
        )
    return errors


def check_sig_coeff_offset_dc(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify SIG_COEFF_OFFSET_DC (7 entries)."""
    print("Checking SIG_COEFF_OFFSET_DC...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_vals = _parse_c_inline_static_1d(h264_cabac_c, "sig_coeff_offset_dc")
    wedeo_vals = parse_rust_array_1d(cabac_tables_rs, "SIG_COEFF_OFFSET_DC")

    errors = compare_arrays("SIG_COEFF_OFFSET_DC", ffmpeg_vals, wedeo_vals)
    if errors == 0:
        print(f"  \u2713 SIG_COEFF_OFFSET_DC ({len(wedeo_vals)} entries)")
    return errors


def check_coeff_abs_level1_ctx(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify COEFF_ABS_LEVEL1_CTX (8 entries)."""
    print("Checking COEFF_ABS_LEVEL1_CTX...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_vals = _parse_c_inline_static_1d(h264_cabac_c, "coeff_abs_level1_ctx")
    wedeo_vals = parse_rust_array_1d(cabac_tables_rs, "COEFF_ABS_LEVEL1_CTX")

    errors = compare_arrays("COEFF_ABS_LEVEL1_CTX", ffmpeg_vals, wedeo_vals)
    if errors == 0:
        print(f"  \u2713 COEFF_ABS_LEVEL1_CTX ({len(wedeo_vals)} entries)")
    return errors


def check_coeff_abs_levelgt1_ctx(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify COEFF_ABS_LEVELGT1_CTX (2x8 entries)."""
    print("Checking COEFF_ABS_LEVELGT1_CTX...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_rows = _parse_c_residual_table_2d(h264_cabac_c, "coeff_abs_levelgt1_ctx")
    wedeo_rows = parse_rust_array_2d(cabac_tables_rs, "COEFF_ABS_LEVELGT1_CTX")

    errors = compare_arrays("COEFF_ABS_LEVELGT1_CTX", ffmpeg_rows, wedeo_rows)
    if errors == 0:
        total = sum(len(r) for r in wedeo_rows)
        print(f"  \u2713 COEFF_ABS_LEVELGT1_CTX ({len(wedeo_rows)}x{len(wedeo_rows[0])} entries)")
    return errors


def check_coeff_abs_level_transition(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify COEFF_ABS_LEVEL_TRANSITION (2x8 entries)."""
    print("Checking COEFF_ABS_LEVEL_TRANSITION...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    ffmpeg_rows = _parse_c_residual_table_2d(
        h264_cabac_c, "coeff_abs_level_transition"
    )
    wedeo_rows = parse_rust_array_2d(cabac_tables_rs, "COEFF_ABS_LEVEL_TRANSITION")

    errors = compare_arrays("COEFF_ABS_LEVEL_TRANSITION", ffmpeg_rows, wedeo_rows)
    if errors == 0:
        print(
            f"  \u2713 COEFF_ABS_LEVEL_TRANSITION "
            f"({len(wedeo_rows)}x{len(wedeo_rows[0])} entries)"
        )
    return errors


# ---------------------------------------------------------------------------
# Section 4: CBF context base table
# ---------------------------------------------------------------------------

def check_cbf_ctx_base(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify CBF_CTX_BASE (14 entries) against base_ctx in h264_cabac.c."""
    print("Checking CBF_CTX_BASE...")
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    cabac_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac.rs"
    )

    # Parse C array (inside function body)
    ffmpeg_vals = _parse_c_inline_static_1d(h264_cabac_c, "base_ctx")

    # Parse Rust const (may not be pub)
    pattern = r'(?:pub\s+)?const\s+CBF_CTX_BASE\s*:\s*\[[^\]]+\]\s*=\s*\[(.*?)\];'
    match = re.search(pattern, cabac_rs, re.DOTALL)
    if not match:
        raise ValueError("Could not find CBF_CTX_BASE in cabac.rs")
    body = match.group(1)
    wedeo_vals = [int(x.strip()) for x in re.findall(r'-?\d+', body)]

    errors = compare_arrays("CBF_CTX_BASE", ffmpeg_vals, wedeo_vals)
    if errors == 0:
        print(f"  \u2713 CBF_CTX_BASE ({len(wedeo_vals)} entries)")
    return errors


# ---------------------------------------------------------------------------
# Section 5: init_cabac_states functional verification
# ---------------------------------------------------------------------------

def _compute_init_states_python(qp: int, tab: list[list[int]]) -> list[int]:
    """Python reference implementation of CABAC state initialization.

    Matches FFmpeg ff_h264_init_cabac_states (h264_cabac.c:1262-1280).
    """
    states = []
    for m, n in tab:
        pre = 2 * (((m * qp) >> 4) + n) - 127
        pre ^= pre >> 31  # abs for negative (Python int, so >> 31 gives -1 or 0)
        # For Python's arbitrary-precision ints, we need to handle this differently:
        # pre ^= pre >> 31 in C with 32-bit int gives -1 for negative, 0 for non-negative.
        # But Python ints are arbitrary width, so >> 31 doesn't give the sign bit.
        # Re-implement using abs:
        pass
    # Redo with correct Python semantics
    states = []
    for m, n in tab:
        pre = 2 * (((m * qp) >> 4) + n) - 127
        # C: pre ^= pre >> 31 — with 32-bit int, this is equivalent to abs(pre)
        # when pre fits in int32. In Python, use explicit abs.
        if pre < 0:
            pre = -pre
        if pre > 124:
            pre = 124 + (pre & 1)
        states.append(pre)
    return states


def check_init_cabac_states(ffmpeg_dir: Path, wedeo_dir: Path) -> int:
    """Verify init_cabac_states formula and output for multiple QP values."""
    print("Checking init_cabac_states (functional)...")
    cabac_tables_rs = read_file(
        wedeo_dir / "codecs" / "wedeo-codec-h264" / "src" / "cabac_tables.rs"
    )

    errors = 0

    # 1. Verify the Rust function formula matches FFmpeg's
    # Extract the Rust function body
    fn_pattern = r'pub fn init_cabac_states\(.*?\)\s*->\s*\[u8;\s*1024\]\s*\{(.*?)\n\}'
    fn_match = re.search(fn_pattern, cabac_tables_rs, re.DOTALL)
    if not fn_match:
        print("  ERROR: Could not find init_cabac_states function")
        return 1
    fn_body = fn_match.group(1)

    # Check for key formula components
    formula_checks = [
        (r'2\s*\*\s*\(\(\(m\s*\*\s*slice_qp\)\s*>>\s*4\)', "pre = 2 * (((m * qp) >> 4)"),
        (r'pre\s*\^=\s*pre\s*>>\s*31', "pre ^= pre >> 31 (abs)"),
        (r'124\s*\+\s*\(pre\s*&\s*1\)', "124 + (pre & 1) (clamp)"),
    ]
    for pattern, desc in formula_checks:
        if not re.search(pattern, fn_body):
            print(f"  WARNING: Could not find formula component: {desc}")
            errors += 1

    # 2. Compute expected states for multiple QP values using Python reference
    h264_cabac_c = read_file(ffmpeg_dir / "libavcodec" / "h264_cabac.c")
    ffmpeg_I = _parse_c_context_init_I(h264_cabac_c)
    ffmpeg_PB = _parse_c_context_init_PB(h264_cabac_c)

    test_qps = [0, 10, 20, 26, 32, 40, 51]

    # Test I-slice init
    for qp in test_qps:
        expected = _compute_init_states_python(qp, ffmpeg_I)
        # Verify against Rust tables (we can't call Rust, but we verify the
        # init tables match, so if the formula matches and tables match,
        # the output must match)
        wedeo_I = _parse_rust_context_init(cabac_tables_rs, "CABAC_CONTEXT_INIT_I")
        actual = _compute_init_states_python(qp, wedeo_I)
        for i in range(1024):
            if expected[i] != actual[i]:
                if errors < 10:
                    print(
                        f"  I-slice QP={qp} ctx[{i}]: "
                        f"FFmpeg={expected[i]}, wedeo={actual[i]} "
                        f"(FFmpeg tab=[{ffmpeg_I[i]}], wedeo tab=[{wedeo_I[i]}])"
                    )
                errors += 1

    # Test PB-slice init for each cabac_init_idc
    for idc in range(3):
        for qp in test_qps:
            expected = _compute_init_states_python(qp, ffmpeg_PB[idc])
            wedeo_PB = _parse_rust_context_init(
                cabac_tables_rs, f"CABAC_CONTEXT_INIT_PB{idc}"
            )
            actual = _compute_init_states_python(qp, wedeo_PB)
            for i in range(1024):
                if expected[i] != actual[i]:
                    if errors < 10:
                        print(
                            f"  PB[{idc}] QP={qp} ctx[{i}]: "
                            f"FFmpeg={expected[i]}, wedeo={actual[i]}"
                        )
                    errors += 1

    if errors == 0:
        print(
            f"  \u2713 init_cabac_states formula verified, "
            f"{len(test_qps)} QP values x 4 tables = "
            f"{len(test_qps) * 4 * 1024} state computations match"
        )
    else:
        if errors > 10:
            print(f"    ... and {errors - 10} more mismatches")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global VERBOSE

    parser = argparse.ArgumentParser(
        description="Verify CABAC lookup tables against FFmpeg"
    )
    parser.add_argument(
        "--ffmpeg-dir",
        type=Path,
        default=Path("FFmpeg"),
        help="Path to FFmpeg source (default: FFmpeg)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show all values for debugging"
    )
    args = parser.parse_args()
    VERBOSE = args.verbose

    wedeo_dir = Path(".")

    if not args.ffmpeg_dir.exists():
        print(f"Error: FFmpeg directory not found: {args.ffmpeg_dir}")
        sys.exit(1)

    total_errors = 0
    checks = [
        # Section 1: Engine tables (cabac.c)
        check_norm_shift,
        check_lps_range,
        check_mlps_state,
        check_last_coeff_flag_offset_8x8,
        # Section 2: Context init tables (h264_cabac.c)
        check_context_init_I,
        check_context_init_PB,
        # Section 3: Residual offset tables (h264_cabac.c)
        check_significant_coeff_flag_offset,
        check_last_coeff_flag_offset,
        check_coeff_abs_level_m1_offset,
        check_significant_coeff_flag_offset_8x8,
        check_sig_coeff_offset_dc,
        check_coeff_abs_level1_ctx,
        check_coeff_abs_levelgt1_ctx,
        check_coeff_abs_level_transition,
        # Section 4: CBF context base
        check_cbf_ctx_base,
        # Section 5: Functional verification
        check_init_cabac_states,
    ]

    for check in checks:
        try:
            errors = check(args.ffmpeg_dir, wedeo_dir)
            if errors > 0:
                print(f"  \u2717 {check.__doc__.split('.')[0].strip() if check.__doc__ else check.__name__}")
            total_errors += errors
        except Exception as e:
            print(f"  ERROR: {e}")
            total_errors += 1
        print()

    if total_errors == 0:
        print(f"All {len(checks)} CABAC table checks passed!")
    else:
        print(f"FAILED: {total_errors} error(s) found")
        sys.exit(1)


if __name__ == "__main__":
    main()
