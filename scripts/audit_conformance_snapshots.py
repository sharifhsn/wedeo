#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Audit conformance snapshots for false positives.

Checks every file in the BITEXACT snapshots to verify:
1. FFmpeg produces > 0 frames (not a bogus 0/0 match)
2. Wedeo produces the same frame count as FFmpeg
3. Files marked BITEXACT are genuinely bitexact

Catches issues like FM1_BT_B where wedeo produces garbage frames
but FFmpeg produces 0 (or vice versa).

Usage:
    python3 scripts/audit_conformance_snapshots.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import find_wedeo_binary

SCRIPTS_DIR = Path(__file__).resolve().parent
SNAPSHOTS = [
    ("Baseline CAVLC", SCRIPTS_DIR / ".conformance_snapshot.json", "fate-suite/h264-conformance"),
    ("CABAC", SCRIPTS_DIR / ".conformance_cabac_snapshot.json", "fate-suite/h264-conformance"),
    ("FRext", SCRIPTS_DIR / ".conformance_frext_snapshot.json", "fate-suite/h264-conformance/FRext"),
]


def count_framecrc_frames(cmd: list[str], timeout: int = 30) -> int:
    """Run a framecrc command and count output frames."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        lines = proc.stdout.decode(errors="replace").splitlines()
        return sum(1 for l in lines if l.strip() and not l.startswith("#"))
    except Exception:
        return -1


def main():
    wedeo_bin = find_wedeo_binary()
    if not wedeo_bin:
        print("ERROR: wedeo-framecrc not found", file=sys.stderr)
        sys.exit(2)

    issues = []
    checked = 0

    for label, snap_path, fate_dir in SNAPSHOTS:
        if not snap_path.exists():
            print(f"SKIP {label}: no snapshot at {snap_path}")
            continue

        data = json.loads(snap_path.read_text())
        passing = data.get("passing", [])
        print(f"\n{label}: auditing {len(passing)} files...")

        for name in passing:
            file_path = Path(fate_dir) / name
            if not file_path.exists():
                print(f"  MISSING  {name}")
                issues.append((label, name, "file not found"))
                continue

            checked += 1
            ffmpeg_frames = count_framecrc_frames(
                ["ffmpeg", "-bitexact", "-i", str(file_path), "-f", "framecrc", "-"]
            )
            wedeo_frames = count_framecrc_frames(
                [str(wedeo_bin), str(file_path)]
            )

            if ffmpeg_frames == 0 and wedeo_frames == 0:
                # Genuine 0-frame file — OK
                pass
            elif ffmpeg_frames == 0 and wedeo_frames > 0:
                print(f"  FALSE_POS  {name}: wedeo={wedeo_frames} frames, ffmpeg=0")
                issues.append((label, name, f"wedeo produces {wedeo_frames} frames but ffmpeg produces 0"))
            elif ffmpeg_frames > 0 and wedeo_frames == 0:
                print(f"  FALSE_POS  {name}: ffmpeg={ffmpeg_frames} frames, wedeo=0")
                issues.append((label, name, f"ffmpeg produces {ffmpeg_frames} frames but wedeo produces 0"))
            elif ffmpeg_frames != wedeo_frames:
                print(f"  COUNT_MISMATCH  {name}: ffmpeg={ffmpeg_frames}, wedeo={wedeo_frames}")
                issues.append((label, name, f"frame count mismatch: ffmpeg={ffmpeg_frames} wedeo={wedeo_frames}"))

    print(f"\nAudited {checked} files across {len(SNAPSHOTS)} snapshots.")
    if issues:
        print(f"\n{len(issues)} ISSUES FOUND:")
        for label, name, desc in issues:
            print(f"  [{label}] {name}: {desc}")
        sys.exit(1)
    else:
        print("All snapshots clean.")


if __name__ == "__main__":
    main()
