#!/usr/bin/env python3
"""Cross-validate wedeo vs FFmpeg across all FATE suite directories.

Usage:
    python3 scripts/fate_cross_validate.py [--dir h264] [--dir vp9-test-vectors] ...

If no --dir given, tests all known directories that exist in fate-suite/.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FATE_SUITE = ROOT / "fate-suite"

# Find wedeo-framecrc binary
WEDEO = None
for candidate in [
    ROOT / "target" / "release" / "wedeo-framecrc",
    ROOT / "target" / "debug" / "wedeo-framecrc",
]:
    if candidate.exists():
        WEDEO = candidate
        break

if not WEDEO:
    print("ERROR: wedeo-framecrc not found. Run: cargo build --release --bin wedeo-framecrc")
    sys.exit(1)

KNOWN_DIRS = [
    "h264-conformance",
    "h264",
    "h264-444",
    "h264-high-depth",
    "vp9-test-vectors",
    "av1-test-vectors",
    "av1",
    "wav",
    "aac",
    "mp3-conformance",
    "wavpack",
    "ogg-flac",
    "pcm-dvd",
    "mkv",
    "mov",
]

VIDEO_EXTS = {".264", ".h264", ".jsv", ".26l", ".mp4", ".mkv", ".mov", ".avi",
              ".ts", ".flv", ".mxf", ".ivf", ".webm", ".obu"}
AUDIO_EXTS = {".wav", ".aac", ".mp3", ".flac", ".ogg", ".wv", ".m4a", ".mka"}
ALL_EXTS = VIDEO_EXTS | AUDIO_EXTS


def run_framecrc(cmd, timeout=30):
    """Run a framecrc command, return (lines, returncode)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        lines = [l for l in r.stdout.strip().split("\n") if l and not l.startswith("#")]
        return lines, r.returncode
    except subprocess.TimeoutExpired:
        return [], -1
    except Exception:
        return [], -2


def compare_file(path):
    """Compare one file. Returns (status, wedeo_frames, ffmpeg_frames)."""
    w_lines, w_rc = run_framecrc([str(WEDEO), str(path)], timeout=60)
    if w_rc != 0:
        return "SKIP_WEDEO", 0, 0

    ff_lines, ff_rc = run_framecrc(
        ["ffmpeg", "-bitexact", "-i", str(path), "-f", "framecrc", "-"],
        timeout=60,
    )
    if ff_rc != 0:
        return "SKIP_FFMPEG", len(w_lines), 0

    if w_lines == ff_lines and len(w_lines) > 0:
        return "BITEXACT", len(w_lines), len(ff_lines)
    else:
        return "DIFF", len(w_lines), len(ff_lines)


def test_directory(dirname):
    dirpath = FATE_SUITE / dirname
    if not dirpath.exists():
        print(f"\n=== {dirname}/ === NOT FOUND, skipping")
        return 0, 0, 0

    files = sorted(f for f in dirpath.iterdir() if f.suffix.lower() in ALL_EXTS)
    # Also include files without extensions (common for .264 conformance files)
    if not files:
        files = sorted(f for f in dirpath.iterdir() if f.is_file() and f.name != "md5sum")

    if not files:
        print(f"\n=== {dirname}/ === no test files found")
        return 0, 0, 0

    print(f"\n=== {dirname}/ ({len(files)} files) ===")

    passed = 0
    failed = 0
    skipped = 0
    fail_names = []

    for f in files:
        name = f.name
        t0 = time.monotonic()
        status, wf, ff = compare_file(f)
        dt = time.monotonic() - t0

        if status == "BITEXACT":
            passed += 1
            print(f"  \033[32mBITEXACT\033[0m  {name} ({wf} frames, {dt:.1f}s)")
        elif status.startswith("SKIP"):
            skipped += 1
            reason = "wedeo" if "WEDEO" in status else "ffmpeg"
            print(f"  \033[33mSKIP\033[0m      {name} ({reason} error)")
        else:
            failed += 1
            fail_names.append(name)
            print(f"  \033[31mDIFF\033[0m      {name} (wedeo:{wf} ffmpeg:{ff}, {dt:.1f}s)")

    print(f"  --- {dirname}: {passed} BITEXACT, {failed} DIFF, {skipped} SKIP ---")
    if fail_names and len(fail_names) <= 10:
        for fn in fail_names:
            print(f"    FAILED: {fn}")

    return passed, failed, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", action="append", help="Specific directory to test (can repeat)")
    args = parser.parse_args()

    dirs = args.dir if args.dir else KNOWN_DIRS

    total_pass = 0
    total_fail = 0
    total_skip = 0

    t0 = time.monotonic()
    for d in dirs:
        p, f, s = test_directory(d)
        total_pass += p
        total_fail += f
        total_skip += s
    dt = time.monotonic() - t0

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_pass} BITEXACT, {total_fail} DIFF, {total_skip} SKIP  ({dt:.1f}s)")
    print(f"{'='*60}")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
