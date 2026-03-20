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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import (
    decode_yuv,
    find_wedeo_binary,
    get_video_info,
    run_framecrc,
)


CONFORMANCE_FILES = [
    "BA1_Sony_D.jsv", "SVA_BA1_B.264", "SVA_NL1_B.264", "BAMQ1_JVC_C.264",
    "BA_MW_D.264", "BANM_MW_D.264", "AUD_MW_E.264", "BA2_Sony_F.jsv",
    "BAMQ2_JVC_C.264", "SVA_BA2_D.264", "SVA_NL2_E.264", "BASQP1_Sony_C.jsv",
    "SVA_Base_B.264", "SVA_FM1_E.264", "SVA_CL1_E.264", "BA1_FT_C.264",
    "BA3_SVA_C.264",
]


def compare_one(input_path, wedeo_bin, no_deblock=False, pixel_detail=False):
    """Compare framecrc for a single file. Returns (total, matching, diffs_info)."""
    env = {"WEDEO_NO_DEBLOCK": "1"} if no_deblock else {}

    wedeo_crcs = run_framecrc([str(wedeo_bin), str(input_path)], env=env)

    ffmpeg_cmd = ["ffmpeg", "-bitexact"]
    if no_deblock:
        ffmpeg_cmd += ["-skip_loop_filter", "all"]
    ffmpeg_cmd += ["-i", str(input_path), "-f", "framecrc", "-"]
    ffmpeg_crcs = run_framecrc(ffmpeg_cmd)

    total = min(len(wedeo_crcs), len(ffmpeg_crcs))
    matching = 0
    diffs = []

    for i in range(total):
        if wedeo_crcs[i] == ffmpeg_crcs[i]:
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

    info = get_video_info(input_path, wedeo_bin=wedeo_bin, no_deblock=no_deblock)
    w, h = info.width, info.height
    cw, ch = w // 2, h // 2
    y_size = w * h
    uv_size = cw * ch
    frame_size = y_size + 2 * uv_size

    wd = decode_yuv(input_path, "wedeo", no_deblock=no_deblock, wedeo_bin=wedeo_bin)
    fd = decode_yuv(input_path, "ffmpeg", no_deblock=no_deblock)

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

    wedeo_bin = find_wedeo_binary()

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
