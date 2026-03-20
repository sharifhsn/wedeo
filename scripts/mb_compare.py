#!/usr/bin/env python3
"""Per-macroblock pixel comparison between wedeo and FFmpeg.

Decodes a H.264 file with both wedeo and FFmpeg (deblocking disabled),
then compares per-MB luma and chroma pixels to find the first differing MB.

Usage:
    python3 scripts/mb_compare.py fate-suite/h264-conformance/BAMQ1_JVC_C.264

Requires:
    - wedeo-framecrc binary (cargo build)
    - ffmpeg binary in PATH
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_debug import (
    check_yuv_frame_count,
    decode_yuv,
    find_wedeo_binary,
    get_video_info,
)


def main():
    parser = argparse.ArgumentParser(description="Per-MB pixel comparison tool")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument(
        "--max-frames", type=int, default=5, help="Max frames to compare"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="First frame to compare"
    )
    parser.add_argument(
        "--deblock", action="store_true",
        help="Compare WITH deblocking (default: deblocking disabled)"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    no_deblock = not args.deblock
    wedeo_bin = find_wedeo_binary()

    # Step 1: Get dimensions from wedeo framecrc header
    info = get_video_info(input_path, wedeo_bin=wedeo_bin, no_deblock=no_deblock)
    width, height = info.width, info.height
    mb_w, mb_h = info.mb_w, info.mb_h
    print(f"Dimensions: {width}x{height}, {info.frame_count} frames")

    # Step 2: Decode raw YUV with both tools
    wedeo_data = decode_yuv(
        input_path, "wedeo", no_deblock=no_deblock, wedeo_bin=wedeo_bin,
    )
    ffmpeg_data = decode_yuv(input_path, "ffmpeg", no_deblock=no_deblock)

    frame_size = width * height * 3 // 2
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    if len(wedeo_data) < frame_size or len(ffmpeg_data) < frame_size:
        print(f"Error: YUV data too short (wedeo={len(wedeo_data)}, ffmpeg={len(ffmpeg_data)})")
        sys.exit(1)

    wedeo_frames = check_yuv_frame_count(
        wedeo_data, width, height, info.frame_count, label="wedeo",
    )
    ffmpeg_frames = check_yuv_frame_count(
        ffmpeg_data, width, height, info.frame_count, label="ffmpeg",
    )
    if wedeo_frames != ffmpeg_frames:
        print(
            f"WARNING: Frame count mismatch! wedeo={wedeo_frames}, ffmpeg={ffmpeg_frames}",
            file=sys.stderr,
        )
        print(file=sys.stderr)

    actual_frames = min(wedeo_frames, ffmpeg_frames)
    end_frame = min(actual_frames, args.start_frame + args.max_frames)
    if args.start_frame >= actual_frames:
        print(f"Error: start-frame {args.start_frame} >= {actual_frames} actual frames")
        sys.exit(1)
    print(f"Comparing frames {args.start_frame}..{end_frame - 1} of {actual_frames} decoded frames\n")

    cw = width // 2
    ch = height // 2

    total_diffs = 0
    total_chroma_diffs = 0
    for frame_idx in range(args.start_frame, end_frame):
        base = frame_idx * frame_size
        w_y = np.frombuffer(wedeo_data[base:base + y_size], dtype=np.uint8).reshape(height, width)
        f_y = np.frombuffer(ffmpeg_data[base:base + y_size], dtype=np.uint8).reshape(height, width)
        w_u = np.frombuffer(wedeo_data[base + y_size:base + y_size + uv_size], dtype=np.uint8).reshape(ch, cw)
        f_u = np.frombuffer(ffmpeg_data[base + y_size:base + y_size + uv_size], dtype=np.uint8).reshape(ch, cw)
        w_v = np.frombuffer(wedeo_data[base + y_size + uv_size:base + frame_size], dtype=np.uint8).reshape(ch, cw)
        f_v = np.frombuffer(ffmpeg_data[base + y_size + uv_size:base + frame_size], dtype=np.uint8).reshape(ch, cw)

        # Luma comparison (per-MB)
        frame_diffs = 0
        first_diff_mb = None
        for my in range(mb_h):
            for mx in range(mb_w):
                y0, y1 = my * 16, (my + 1) * 16
                x0, x1 = mx * 16, (mx + 1) * 16
                w_block = w_y[y0:y1, x0:x1]
                f_block = f_y[y0:y1, x0:x1]
                diff = np.abs(w_block.astype(np.int16) - f_block.astype(np.int16))
                if diff.any():
                    frame_diffs += 1
                    if first_diff_mb is None:
                        max_diff = int(diff.max())
                        mean_diff = float(diff[diff > 0].mean())
                        first_diff_mb = (mx, my, max_diff, mean_diff)

        # Chroma comparison (whole-plane)
        u_diff = np.abs(w_u.astype(np.int16) - f_u.astype(np.int16))
        v_diff = np.abs(w_v.astype(np.int16) - f_v.astype(np.int16))
        u_max = int(u_diff.max())
        v_max = int(v_diff.max())
        chroma_ok = u_max == 0 and v_max == 0

        if frame_diffs > 0:
            total_diffs += frame_diffs
            mx, my, max_d, mean_d = first_diff_mb
            chroma_note = "" if chroma_ok else f" [chroma: U_max={u_max} V_max={v_max}]"
            print(
                f"Frame {frame_idx}: {frame_diffs}/{mb_w * mb_h} MBs differ "
                f"(first: MB({mx},{my}), max_diff={max_d}, mean_diff={mean_d:.1f}){chroma_note}"
            )
            print(
                f"  Debug: cargo build --bin wedeo-framecrc -p wedeo-fate --features tracing && "
                f"WEDEO_NO_DEBLOCK=1 RUST_LOG=wedeo_codec_h264::mb=trace "
                f"./target/debug/wedeo-framecrc {input_path} "
                f">/dev/null 2>/tmp/trace.log && "
                f'sed \'s/\\x1b\\[[0-9;]*m//g\' /tmp/trace.log | grep "MB({mx},{my})" | head -20'
            )
        elif not chroma_ok:
            total_chroma_diffs += 1
            print(f"Frame {frame_idx}: luma MATCH, chroma diff U_max={u_max} V_max={v_max}")
        else:
            print(f"Frame {frame_idx}: MATCH")

    print(f"\nTotal: {total_diffs} luma-differing MBs, {total_chroma_diffs} chroma-only diffs across {end_frame - args.start_frame} frames")
    if total_diffs == 0 and total_chroma_diffs == 0:
        print("BITEXACT!")


if __name__ == "__main__":
    main()
