#!/usr/bin/env python3
"""Compare framecrc output between wedeo and FFmpeg for H.264 files.

Runs both decoders, compares per-frame CRCs, and optionally does per-plane
pixel-level analysis on mismatching frames.

Usage:
    # Compare a single file
    python3 scripts/framecrc_compare.py fate-suite/h264-conformance/BA1_Sony_D.jsv

    # Compare all 17 Baseline conformance files
    python3 scripts/framecrc_compare.py --all

    # Show per-plane pixel diffs for mismatching frames
    python3 scripts/framecrc_compare.py --pixel-detail fate-suite/h264-conformance/SVA_CL1_E.264

Requires:
    - wedeo-framecrc binary (cargo build)
    - ffmpeg binary in PATH
    - numpy (for --pixel-detail mode)
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


CONFORMANCE_FILES = [
    "BA1_Sony_D.jsv", "SVA_BA1_B.264", "SVA_NL1_B.264", "BAMQ1_JVC_C.264",
    "BA_MW_D.264", "BANM_MW_D.264", "AUD_MW_E.264", "BA2_Sony_F.jsv",
    "BAMQ2_JVC_C.264", "SVA_BA2_D.264", "SVA_NL2_E.264", "BASQP1_Sony_C.jsv",
    "SVA_Base_B.264", "SVA_FM1_E.264", "SVA_CL1_E.264", "BA1_FT_C.264",
    "BA3_SVA_C.264",
]


def find_wedeo_bin():
    """Find the wedeo-framecrc binary (debug first, then release)."""
    for profile in ["debug", "release"]:
        candidate = Path("target") / profile / "wedeo-framecrc"
        if candidate.exists():
            return str(candidate.resolve())
    return None


def run_framecrc(cmd, env=None):
    """Run a framecrc command and return parsed lines as (frame_idx, crc) tuples."""
    full_env = {**subprocess.os.environ, **(env or {})}
    result = subprocess.run(cmd, capture_output=True, env=full_env)
    if result.returncode != 0:
        print(f"WARN: {' '.join(cmd[:3])}... exited with {result.returncode}", file=sys.stderr)
    lines = result.stdout.decode().splitlines()
    frames = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            frame_idx = int(parts[1].strip())
            crc = parts[5].strip()
            frames.append((frame_idx, crc))
    return frames


def get_dimensions(wedeo_bin, input_path, env=None):
    """Extract width x height from wedeo framecrc header."""
    full_env = {**subprocess.os.environ, **(env or {})}
    result = subprocess.run(
        [wedeo_bin, str(input_path)], capture_output=True, env=full_env
    )
    for line in result.stdout.decode().splitlines():
        if line.startswith("#dimensions"):
            parts = line.split(":")[-1].strip().split("x")
            return int(parts[0]), int(parts[1])
    return None, None


def compare_one(input_path, wedeo_bin, no_deblock=False, pixel_detail=False):
    """Compare framecrc for a single file. Returns (total, matching, diffs_info)."""
    env = {"WEDEO_NO_DEBLOCK": "1"} if no_deblock else {}

    wedeo_frames = run_framecrc([wedeo_bin, str(input_path)], env=env)

    ffmpeg_cmd = ["ffmpeg", "-bitexact"]
    if no_deblock:
        ffmpeg_cmd += ["-skip_loop_filter", "all"]
    ffmpeg_cmd += ["-i", str(input_path), "-f", "framecrc", "-"]
    ffmpeg_frames = run_framecrc(ffmpeg_cmd)

    total = min(len(wedeo_frames), len(ffmpeg_frames))
    matching = 0
    diffs = []

    for i in range(total):
        w_idx, w_crc = wedeo_frames[i]
        f_idx, f_crc = ffmpeg_frames[i]
        if w_crc == f_crc:
            matching += 1
        else:
            diffs.append(i)

    # Optional pixel-level analysis
    plane_info = None
    if pixel_detail and diffs:
        plane_info = pixel_plane_analysis(
            input_path, wedeo_bin, no_deblock, diffs
        )

    return total, matching, diffs, plane_info


def pixel_plane_analysis(input_path, wedeo_bin, no_deblock, diff_frames):
    """Decode raw YUV and compare Y/U/V planes for differing frames."""
    import numpy as np

    w, h = get_dimensions(wedeo_bin, input_path)
    if w is None:
        return None

    cw, ch = w // 2, h // 2
    y_size = w * h
    uv_size = cw * ch
    frame_size = y_size + 2 * uv_size

    env = {"WEDEO_NO_DEBLOCK": "1"} if no_deblock else {}

    # Decode raw YUV from both
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        wedeo_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        ffmpeg_path = f.name

    subprocess.run(
        [wedeo_bin, str(input_path), "--raw-yuv", wedeo_path],
        capture_output=True,
        env={**subprocess.os.environ, **env},
        check=True,
    )

    ffmpeg_cmd = ["ffmpeg", "-y", "-bitexact"]
    if no_deblock:
        ffmpeg_cmd += ["-skip_loop_filter", "all"]
    ffmpeg_cmd += ["-i", str(input_path), "-pix_fmt", "yuv420p", "-f", "rawvideo", ffmpeg_path]
    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

    wd = Path(wedeo_path).read_bytes()
    fd = Path(ffmpeg_path).read_bytes()
    Path(wedeo_path).unlink(missing_ok=True)
    Path(ffmpeg_path).unlink(missing_ok=True)

    results = []
    for frame_idx in diff_frames:
        if (frame_idx + 1) * frame_size > min(len(wd), len(fd)):
            break
        base = frame_idx * frame_size
        wy = np.frombuffer(wd[base:base + y_size], dtype=np.uint8)
        fy = np.frombuffer(fd[base:base + y_size], dtype=np.uint8)
        wu = np.frombuffer(wd[base + y_size:base + y_size + uv_size], dtype=np.uint8)
        fu = np.frombuffer(fd[base + y_size:base + y_size + uv_size], dtype=np.uint8)
        wv = np.frombuffer(wd[base + y_size + uv_size:base + frame_size], dtype=np.uint8)
        fv = np.frombuffer(fd[base + y_size + uv_size:base + frame_size], dtype=np.uint8)

        y_max = int(np.abs(wy.astype(int) - fy.astype(int)).max())
        u_max = int(np.abs(wu.astype(int) - fu.astype(int)).max())
        v_max = int(np.abs(wv.astype(int) - fv.astype(int)).max())
        results.append((frame_idx, y_max, u_max, v_max))

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare framecrc: wedeo vs FFmpeg")
    parser.add_argument("input", nargs="?", help="H.264 input file")
    parser.add_argument("--all", action="store_true", help="Compare all 17 Baseline files")
    parser.add_argument("--no-deblock", action="store_true", help="Disable deblocking filter")
    parser.add_argument("--pixel-detail", action="store_true", help="Show per-plane pixel diffs")
    parser.add_argument("--fate-dir", default="fate-suite/h264-conformance",
                        help="Path to conformance files (default: fate-suite/h264-conformance)")
    args = parser.parse_args()

    wedeo_bin = find_wedeo_bin()
    if wedeo_bin is None:
        print("Error: wedeo-framecrc not found. Run `cargo build --bin wedeo-framecrc`", file=sys.stderr)
        sys.exit(1)

    if args.all:
        files = [Path(args.fate_dir) / f for f in CONFORMANCE_FILES]
    elif args.input:
        files = [Path(args.input).resolve()]
    else:
        parser.print_help()
        sys.exit(1)

    total_files = 0
    bitexact_files = 0

    for input_path in files:
        if not input_path.exists():
            print(f"SKIP: {input_path.name} (not found)")
            continue

        total_files += 1
        total, matching, diffs, plane_info = compare_one(
            input_path, wedeo_bin, args.no_deblock, args.pixel_detail
        )

        if not diffs:
            bitexact_files += 1
            print(f"BITEXACT  {input_path.name} ({total} frames)")
        else:
            label = f"{matching}/{total}"
            summary = f"DIFF      {input_path.name} ({label} match, {len(diffs)} differ)"

            if plane_info:
                # Summarize plane-level info
                y_any = any(r[1] > 0 for r in plane_info)
                u_any = any(r[2] > 0 for r in plane_info)
                v_any = any(r[3] > 0 for r in plane_info)
                y_max = max(r[1] for r in plane_info)
                u_max = max(r[2] for r in plane_info)
                v_max = max(r[3] for r in plane_info)
                planes = []
                if y_any:
                    planes.append(f"Y≤{y_max}")
                if u_any:
                    planes.append(f"U≤{u_max}")
                if v_any:
                    planes.append(f"V≤{v_max}")
                summary += f"  [{'+'.join(planes) if planes else 'CRC-only'}]"

            print(summary)

            if plane_info and args.pixel_detail:
                for frame_idx, y_m, u_m, v_m in plane_info[:10]:
                    print(f"    frame {frame_idx}: Y_max={y_m} U_max={u_m} V_max={v_m}")
                if len(plane_info) > 10:
                    print(f"    ... and {len(plane_info) - 10} more frames")

    if args.all:
        print(f"\nSummary: {bitexact_files}/{total_files} BITEXACT")


if __name__ == "__main__":
    main()
