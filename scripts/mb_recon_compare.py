#!/usr/bin/env python3
"""Compare per-MB reconstruction checksums between wedeo and FFmpeg.

Decodes a file with both decoders, computes per-MB pixel sums from the
raw YUV output, and finds the first diverging MB. Reports whether the
divergence is pre-deblock or post-deblock.

Usage:
    python3 scripts/mb_recon_compare.py <file>                    # first diverging MB across all frames
    python3 scripts/mb_recon_compare.py <file> --frame 1          # specific frame only
    python3 scripts/mb_recon_compare.py <file> --no-deblock       # compare without deblock filter
    python3 scripts/mb_recon_compare.py <file> --wedeo-trace      # also show wedeo's MB_RECON trace

Requires:
    - wedeo-framecrc binary (debug build with tracing)
    - ffmpeg binary in PATH
"""
# /// script
# requires-python = ">=3.10"
# ///

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def decode_to_yuv(cmd: list[str], env: dict | None = None) -> bytes:
    """Run a decoder command that outputs raw YUV to stdout."""
    result = subprocess.run(cmd, capture_output=True, timeout=60, env=env)
    if result.returncode != 0:
        print(f"Decode failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr.decode(errors="replace")[-500:], file=sys.stderr)
        sys.exit(1)
    return result.stdout


def get_dimensions(path: str) -> tuple[int, int]:
    """Get video dimensions via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=10,
    )
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def get_frame_count(path: str) -> int:
    """Get frame count via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=30,
    )
    return int(result.stdout.strip())


def mb_sums_from_yuv(yuv: bytes, width: int, height: int, frame_idx: int) -> list[tuple[int, int, int, int, int]]:
    """Extract per-MB pixel sums from a YUV420p frame.
    Returns list of (mb_x, mb_y, y_sum, u_sum, v_sum)."""
    mb_w = (width + 15) // 16
    mb_h = (height + 15) // 16
    y_size = width * height
    uv_w = width // 2
    uv_h = height // 2
    uv_size = uv_w * uv_h
    frame_size = y_size + 2 * uv_size

    offset = frame_idx * frame_size
    if offset + frame_size > len(yuv):
        return []

    y_plane = yuv[offset:offset + y_size]
    u_plane = yuv[offset + y_size:offset + y_size + uv_size]
    v_plane = yuv[offset + y_size + uv_size:offset + frame_size]

    results = []
    for mby in range(mb_h):
        for mbx in range(mb_w):
            # Luma 16x16
            y_sum = 0
            for dy in range(16):
                py = mby * 16 + dy
                if py >= height:
                    break
                for dx in range(16):
                    px = mbx * 16 + dx
                    if px >= width:
                        continue
                    y_sum += y_plane[py * width + px]

            # Chroma 8x8
            u_sum = 0
            v_sum = 0
            for dy in range(8):
                py = mby * 8 + dy
                if py >= uv_h:
                    break
                for dx in range(8):
                    px = mbx * 8 + dx
                    if px >= uv_w:
                        continue
                    u_sum += u_plane[py * uv_w + px]
                    v_sum += v_plane[py * uv_w + px]

            results.append((mbx, mby, y_sum, u_sum, v_sum))

    return results


def main():
    parser = argparse.ArgumentParser(description="Per-MB reconstruction checksum comparison")
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--frame", type=int, default=None, help="Specific frame to compare (default: find first diff)")
    parser.add_argument("--no-deblock", action="store_true", help="Disable deblocking in both decoders")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames to check (default: 10)")
    args = parser.parse_args()

    width, height = get_dimensions(args.input)
    mb_w = (width + 15) // 16
    mb_h = (height + 15) // 16
    print(f"Dimensions: {width}x{height}, MBs: {mb_w}x{mb_h}")

    # Use the debug build which has tracing subscriber initialized.
    # The release build doesn't emit trace-level output.
    debug_bin = Path("target/debug/wedeo-framecrc")
    if not debug_bin.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ffmpeg_debug import find_wedeo_binary
        wedeo_bin = find_wedeo_binary()
    else:
        wedeo_bin = str(debug_bin)
    if not wedeo_bin:
        print("ERROR: wedeo-framecrc binary not found", file=sys.stderr)
        sys.exit(2)

    # Decode with FFmpeg to raw YUV
    ffmpeg_cmd = ["ffmpeg", "-bitexact", "-i", args.input]
    if args.no_deblock:
        ffmpeg_cmd += ["-skip_loop_filter", "all"]
    ffmpeg_cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    print(f"Decoding with FFmpeg...", end=" ", flush=True)
    ffmpeg_yuv = decode_to_yuv(ffmpeg_cmd)
    print(f"{len(ffmpeg_yuv)} bytes")

    # Decode with wedeo to raw YUV (via framecrc, extract from decoded frames)
    # We use ffmpeg to decode wedeo's output... actually, let's decode directly
    # by having wedeo output raw YUV. But wedeo-framecrc outputs framecrc format.
    # Instead, let's use the same approach: decode both to YUV via pipe.
    wedeo_env = {}
    if args.no_deblock:
        wedeo_env["WEDEO_NO_DEBLOCK"] = "1"

    # wedeo-framecrc doesn't output raw YUV. Use ffmpeg to decode the file,
    # and wedeo trace logs to get the pre-deblock checksums.
    # For post-deblock comparison, compare YUV from both decoders.
    # Actually, the simplest approach: pipe both through rawvideo output.
    # wedeo doesn't have a raw YUV output mode, so we compare via framecrc
    # and use the MB_RECON trace for detailed MB-level analysis.

    # Strategy: decode both to YUV via temp files
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as tmp:
        wedeo_yuv_path = tmp.name

    # Use wedeo decode mode to write raw YUV
    # Since wedeo-framecrc only outputs CRC, we need to use ffmpeg for both
    # and compare frame-by-frame. Then use wedeo's trace for MB-level detail.

    # Actually, the best approach is simpler: just compare the FFmpeg YUV
    # against itself-minus-deblock to identify which MBs have deblock-only diffs.
    # For wedeo vs FFmpeg, extract per-MB sums from both YUV streams.

    # Let's decode wedeo via the info/decode pipeline if available, or fall back
    # to comparing framecrc + trace.

    # For now: compare FFmpeg YUV per-MB against wedeo's MB_RECON trace.
    # The trace gives pre-deblock sums; FFmpeg YUV gives post-deblock pixels.
    # To compare apples-to-apples, use --no-deblock on FFmpeg too.

    # For post-deblock comparison (default), use the normal FFmpeg YUV.
    # For pre-deblock, use skip_loop_filter (but note: this also changes
    # reference frames for inter prediction, so only frame 0 is apples-to-apples).
    if args.no_deblock:
        ffmpeg_cmd_nd = ["ffmpeg", "-bitexact", "-skip_loop_filter", "all",
                         "-i", args.input,
                         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
        print("Decoding FFmpeg without deblock...", end=" ", flush=True)
        ffmpeg_yuv_compare = decode_to_yuv(ffmpeg_cmd_nd)
        print(f"{len(ffmpeg_yuv_compare)} bytes")
    else:
        ffmpeg_yuv_compare = ffmpeg_yuv

    # Get wedeo MB_RECON + MB_DEBLOCK traces
    import os
    env = {**os.environ, "RUST_LOG": "wedeo_codec_h264::mb=trace,wedeo_codec_h264::deblock=trace"}
    if args.no_deblock:
        env["WEDEO_NO_DEBLOCK"] = "1"
    print(f"Decoding with wedeo (trace)...", end=" ", flush=True)
    result = subprocess.run(
        [str(wedeo_bin), args.input],
        capture_output=True, text=True, timeout=120, env=env,
    )
    print("done")

    # Strip ANSI escape codes, then parse MB_RECON and MB_DEBLOCK lines.
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    recon_pattern = re.compile(
        r"MB_RECON.*mb_x=(\d+).*mb_y=(\d+).*mb_type=(\d+).*qp=(\d+).*cbp=(\d+)"
        r".*t8x8=(true|false).*is_intra=(true|false).*y_sum=(\d+).*u_sum=(\d+).*v_sum=(\d+)"
    )
    deblock_pattern = re.compile(
        r"MB_DEBLOCK.*mb_x=(\d+).*mb_y=(\d+).*y_sum=(\d+).*u_sum=(\d+).*v_sum=(\d+)"
    )

    # Pre-deblock checksums from MB_RECON
    wedeo_recon = {}  # (frame_idx, mb_x, mb_y) -> (y_sum, u_sum, v_sum, mb_type, qp, cbp, t8x8)
    # Post-deblock checksums from MB_DEBLOCK
    wedeo_deblock = {}  # (frame_idx, mb_x, mb_y) -> (y_sum, u_sum, v_sum)

    recon_frame_idx = -1
    recon_last_mb_y = -1
    deblock_frame_idx = -1
    deblock_last_mb_y = -1

    for line in result.stderr.split("\n"):
        line = ansi_escape.sub("", line)

        m = recon_pattern.search(line)
        if m:
            mbx, mby = int(m.group(1)), int(m.group(2))
            if mbx == 0 and mby == 0 and (recon_last_mb_y > 0 or recon_frame_idx == -1):
                recon_frame_idx += 1
            recon_last_mb_y = mby
            wedeo_recon[(recon_frame_idx, mbx, mby)] = (
                int(m.group(8)), int(m.group(9)), int(m.group(10)),
                int(m.group(3)), int(m.group(4)), int(m.group(5)),
                m.group(6) == "true",
            )
            continue

        m = deblock_pattern.search(line)
        if m:
            mbx, mby = int(m.group(1)), int(m.group(2))
            if mbx == 0 and mby == 0 and (deblock_last_mb_y > 0 or deblock_frame_idx == -1):
                deblock_frame_idx += 1
            deblock_last_mb_y = mby
            wedeo_deblock[(deblock_frame_idx, mbx, mby)] = (
                int(m.group(3)), int(m.group(4)), int(m.group(5)),
            )

    print(f"Parsed {len(wedeo_recon)} MB_RECON + {len(wedeo_deblock)} MB_DEBLOCK entries")

    # Compare per-frame
    start_frame = args.frame if args.frame is not None else 0
    end_frame = (args.frame + 1) if args.frame is not None else min(frame_idx + 1, args.max_frames)

    # Choose which wedeo checksums to compare: post-deblock (default) or pre-deblock
    use_deblock = not args.no_deblock
    wedeo_mbs = wedeo_deblock if use_deblock else wedeo_recon
    stage = "post-deblock" if use_deblock else "pre-deblock"

    first_diff_found = False
    for fi in range(start_frame, end_frame):
        ffmpeg_sums = mb_sums_from_yuv(ffmpeg_yuv_compare, width, height, fi)

        diffs = []
        for (mbx, mby, fy, fu, fv) in ffmpeg_sums:
            key = (fi, mbx, mby)
            if key not in wedeo_mbs:
                # Get info from recon trace if available
                rkey = (fi, mbx, mby)
                info = "MISSING"
                if rkey in wedeo_recon:
                    _, _, _, mt, qp, cbp, t8x8 = wedeo_recon[rkey]
                    info = f"type={mt} qp={qp} cbp={cbp} t8x8={t8x8} (no deblock trace)"
                diffs.append((mbx, mby, info, fy, fu, fv, 0, 0, 0))
                continue

            if use_deblock:
                wy, wu, wv = wedeo_mbs[key]
                # Get parse info from recon trace
                rkey = (fi, mbx, mby)
                mt, qp, cbp, t8x8 = 0, 0, 0, False
                if rkey in wedeo_recon:
                    _, _, _, mt, qp, cbp, t8x8 = wedeo_recon[rkey]
            else:
                wy, wu, wv, mt, qp, cbp, t8x8 = wedeo_mbs[key]

            if wy != fy or wu != fu or wv != fv:
                diffs.append((mbx, mby, f"type={mt} qp={qp} cbp={cbp} t8x8={t8x8}",
                              fy, fu, fv, wy, wu, wv))

        if not diffs:
            print(f"Frame {fi}: MATCH ({len(ffmpeg_sums)} MBs, {stage})")
        else:
            first_diff_found = True
            print(f"\nFrame {fi}: {len(diffs)}/{len(ffmpeg_sums)} MBs differ ({stage})")
            print(f"  {'MB':<10} {'Info':<35} {'FFmpeg Y/U/V':<25} {'Wedeo Y/U/V':<25} {'Delta Y'}")
            print(f"  {'-'*10} {'-'*35} {'-'*25} {'-'*25} {'-'*10}")
            for mbx, mby, info, fy, fu, fv, wy, wu, wv in diffs[:20]:
                ff_str = f"{fy}/{fu}/{fv}"
                we_str = f"{wy}/{wu}/{wv}" if wy else "MISSING"
                dy = wy - fy if wy else 0
                print(f"  ({mbx:2},{mby:2})   {info:<35} {ff_str:<25} {we_str:<25} {dy:+d}")
            if len(diffs) > 20:
                print(f"  ... and {len(diffs) - 20} more")

            if args.frame is None:
                break  # In auto mode, stop at first differing frame

    if not first_diff_found:
        print(f"\nAll {end_frame - start_frame} frames MATCH at MB level ({stage})")

    # Cleanup
    Path(wedeo_yuv_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
