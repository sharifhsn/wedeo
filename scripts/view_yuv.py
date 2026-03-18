#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "Pillow"]
# ///
"""
Side-by-side YUV video viewer: wedeo vs FFmpeg.

Usage:
    uv run scripts/view_yuv.py <input.264>
    uv run scripts/view_yuv.py <input.264> --frame 5
    uv run scripts/view_yuv.py --wedeo /tmp/wedeo.yuv --ffmpeg /tmp/ffmpeg.yuv -W 176 -H 144

Controls (when viewing all frames):
    - Press Enter to see next frame, Ctrl+C to quit
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Install Pillow: pip install Pillow numpy", file=sys.stderr)
    sys.exit(1)


def yuv420p_to_rgb(y_plane, u_plane, v_plane, width, height):
    """Convert YUV420p planes to RGB numpy array."""
    Y = y_plane.reshape(height, width).astype(np.float32)
    U = u_plane.reshape(height // 2, width // 2).astype(np.float32)
    V = v_plane.reshape(height // 2, width // 2).astype(np.float32)

    # Upsample chroma
    U = np.repeat(np.repeat(U, 2, axis=0), 2, axis=1)
    V = np.repeat(np.repeat(V, 2, axis=0), 2, axis=1)

    # BT.601 conversion (TV range)
    R = Y + 1.402 * (V - 128)
    G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
    B = Y + 1.772 * (U - 128)

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def read_yuv_frame(data, frame_idx, width, height):
    """Read one YUV420p frame from raw bytes."""
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size
    offset = frame_idx * frame_size

    if offset + frame_size > len(data):
        return None

    y = np.frombuffer(data, dtype=np.uint8, count=y_size, offset=offset)
    u = np.frombuffer(data, dtype=np.uint8, count=uv_size, offset=offset + y_size)
    v = np.frombuffer(data, dtype=np.uint8, count=uv_size, offset=offset + y_size + uv_size)
    return y, u, v


def decode_with_wedeo(input_path):
    """Decode a video file using wedeo-cli, return (yuv_bytes, width, height)."""
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "wedeo-cli", "--", "decode", input_path],
        capture_output=True,
        cwd=str(Path(__file__).parent.parent),
    )
    if result.returncode != 0:
        print(f"wedeo decode failed:\n{result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    # Parse width/height from stderr
    width = height = 0
    for line in result.stderr.decode().splitlines():
        if line.startswith("WEDEO_VIDEO_WIDTH="):
            width = int(line.split("=")[1])
        elif line.startswith("WEDEO_VIDEO_HEIGHT="):
            height = int(line.split("=")[1])

    return result.stdout, width, height


def decode_with_ffmpeg(input_path, width, height):
    """Decode a video file using ffmpeg, return yuv_bytes."""
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        tmp = f.name

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-f", "rawvideo", "-pix_fmt", "yuv420p", tmp],
        capture_output=True,
    )
    data = Path(tmp).read_bytes()
    Path(tmp).unlink()
    return data


def make_diff_image(rgb_a, rgb_b, gain=4):
    """Create amplified absolute difference image."""
    diff = np.abs(rgb_a.astype(np.int16) - rgb_b.astype(np.int16))
    return np.clip(diff * gain, 0, 255).astype(np.uint8)


def psnr(a, b):
    """Compute PSNR between two images."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def main():
    parser = argparse.ArgumentParser(description="View wedeo vs FFmpeg decoded video")
    parser.add_argument("input", nargs="?", help="Input .264/.jsv file to decode")
    parser.add_argument("--wedeo", help="Pre-decoded wedeo YUV file")
    parser.add_argument("--ffmpeg", help="Pre-decoded FFmpeg YUV file")
    parser.add_argument("-W", "--width", type=int, default=0)
    parser.add_argument("-H", "--height", type=int, default=0)
    parser.add_argument("--frame", type=int, default=None, help="Show specific frame (0-indexed)")
    parser.add_argument("--save", help="Save comparison image to file instead of displaying")
    parser.add_argument("--scale", type=int, default=3, help="Upscale factor for display (default 3)")
    args = parser.parse_args()

    if args.input:
        print(f"Decoding {args.input} with wedeo...", file=sys.stderr)
        wedeo_data, width, height = decode_with_wedeo(args.input)
        print(f"  {width}x{height}, {len(wedeo_data)} bytes", file=sys.stderr)

        print(f"Decoding {args.input} with FFmpeg...", file=sys.stderr)
        ffmpeg_data = decode_with_ffmpeg(args.input, width, height)
        print(f"  {len(ffmpeg_data)} bytes", file=sys.stderr)
    elif args.wedeo and args.ffmpeg:
        wedeo_data = Path(args.wedeo).read_bytes()
        ffmpeg_data = Path(args.ffmpeg).read_bytes()
        width, height = args.width, args.height
        if width == 0 or height == 0:
            print("Must specify --width and --height with pre-decoded files", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    frame_size = width * height * 3 // 2
    n_wedeo = len(wedeo_data) // frame_size
    n_ffmpeg = len(ffmpeg_data) // frame_size
    print(f"Frames: wedeo={n_wedeo}, ffmpeg={n_ffmpeg}", file=sys.stderr)

    frames_to_show = range(min(n_wedeo, n_ffmpeg))
    if args.frame is not None:
        frames_to_show = [args.frame]

    for fi in frames_to_show:
        w_planes = read_yuv_frame(wedeo_data, fi, width, height)
        f_planes = read_yuv_frame(ffmpeg_data, fi, width, height)

        if w_planes is None or f_planes is None:
            break

        w_rgb = yuv420p_to_rgb(*w_planes, width, height)
        f_rgb = yuv420p_to_rgb(*f_planes, width, height)
        d_rgb = make_diff_image(w_rgb, f_rgb)

        p = psnr(w_rgb, f_rgb)

        # Create side-by-side: wedeo | FFmpeg | diff(4x)
        label_h = 20
        canvas = np.ones((height + label_h, width * 3 + 4, 3), dtype=np.uint8) * 32

        # Paste images
        canvas[label_h : label_h + height, 0:width] = w_rgb
        canvas[label_h : label_h + height, width + 2 : 2 * width + 2] = f_rgb
        canvas[label_h : label_h + height, 2 * width + 4 : 3 * width + 4] = d_rgb

        img = Image.fromarray(canvas)

        # Scale up for visibility
        scale = args.scale
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

        status = f"Frame {fi}: PSNR={p:.1f} dB"
        if p == float("inf"):
            status = f"Frame {fi}: IDENTICAL"
        print(status, file=sys.stderr)

        if args.save:
            out_path = args.save if args.frame is not None else f"{args.save}.frame{fi:04d}.png"
            img.save(out_path)
            print(f"  Saved to {out_path}", file=sys.stderr)
        else:
            img.show(title=status)
            if args.frame is None:
                input(f"  [{status}] Press Enter for next frame...")


if __name__ == "__main__":
    main()
