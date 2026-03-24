#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Verify that all pipeline-stage trace tags appear in decoder output.

Decodes a test file with all trace levels enabled and checks that each
expected tag appears at least once. Useful after adding/modifying traces
to confirm they're reachable.

Usage:
    python3 scripts/verify_traces.py [test_file]

Default test file: fate-suite/h264-conformance/FRext/Freh1_B.264
(uses custom scaling matrices, exercises most code paths)
"""
import subprocess
import sys

DEFAULT_FILE = "fate-suite/h264-conformance/FRext/Freh1_B.264"

# Tags that should appear at debug level (per-frame/slice)
DEBUG_TAGS = [
    "SLICE",
    "DPB",
    "REFLIST",
    "PPS_SCALING",
    "DEQUANT_TABLES",
]

# Tags that should appear at trace level (per-MB/block)
TRACE_TAGS = [
    "COEFF",
    "DEQUANT",
    "MB_RECON",
    "MB_DEBLOCK",
]

def main():
    test_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

    env = {
        "RUST_LOG": "wedeo_codec_h264=trace",
        "PATH": "/usr/bin:/usr/local/bin",
    }

    # Try release first, fall back to debug
    for binary in ["./target/release/wedeo-cli", "./target/debug/wedeo-cli"]:
        try:
            result = subprocess.run(
                [binary, "decode", test_file],
                stdout=subprocess.DEVNULL,  # discard binary YUV output
                stderr=subprocess.PIPE,
                timeout=60,
                env=env,
            )
            break
        except FileNotFoundError:
            continue
    else:
        print("ERROR: wedeo-cli not found. Run: cargo build -p wedeo-cli")
        sys.exit(1)

    stderr = result.stderr.decode("utf-8", errors="replace")
    all_tags = DEBUG_TAGS + TRACE_TAGS
    found = {}
    missing = []

    for tag in all_tags:
        count = stderr.count(f" {tag} ")
        if count == 0:
            # Also try without trailing space (end of line)
            count = stderr.count(f" {tag}\n")
        found[tag] = count
        if count == 0:
            missing.append(tag)

    print(f"Test file: {test_file}")
    print(f"Stderr lines: {len(stderr.splitlines())}")
    print()

    for tag in all_tags:
        status = "OK" if found[tag] > 0 else "MISSING"
        level = "debug" if tag in DEBUG_TAGS else "trace"
        print(f"  [{status:>7}] {tag:<20} ({found[tag]:>6} occurrences, {level})")

    print()
    if missing:
        print(f"FAIL: {len(missing)} tags missing: {', '.join(missing)}")
        sys.exit(1)
    else:
        print(f"PASS: All {len(all_tags)} tags present")


if __name__ == "__main__":
    main()
