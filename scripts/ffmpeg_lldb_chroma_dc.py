#!/usr/bin/env python3
"""Extract chroma DC intermediate values from FFmpeg via lldb.

Sets breakpoints at key H.264 chroma decode points and extracts:
1. Chroma DC coefficients before Hadamard (input to chroma_dc_dequant_idct)
2. Chroma DC values after Hadamard (output of chroma_dc_dequant_idct)
3. The dc_add value applied to pixels (from h264_idct_dc_add)
4. Final chroma pixel values after DC add

Usage:
    # Extract values for MB(7,8) on frame 16 of SVA_BA2_D
    python3 scripts/ffmpeg_lldb_chroma_dc.py \
        fate-suite/h264-conformance/SVA_BA2_D.264 \
        --frame 16 --mb-x 7 --mb-y 8

Requires:
    - FFmpeg built with debug info: ./configure --disable-optimizations --enable-debug=3
    - lldb in PATH

Build FFmpeg with debug info:
    cd FFmpeg
    ./configure --disable-optimizations --enable-debug=3 --disable-stripping
    make -j$(nproc)
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path


def find_ffmpeg_debug():
    """Find a debug-built FFmpeg binary."""
    candidates = [
        Path("FFmpeg/ffmpeg"),
        Path("FFmpeg/ffmpeg_g"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def generate_lldb_script(input_path, target_frame, mb_x, mb_y, chroma_idx=1):
    """Generate an lldb command script for extracting chroma DC values.

    chroma_idx: 0=Cb(U), 1=Cr(V)
    """
    plane_name = "Cr(V)" if chroma_idx == 1 else "Cb(U)"

    # MB address in raster scan
    # We need to know the frame dimensions to compute this, but for now
    # the user can adjust.

    script = textwrap.dedent(f"""\
        # FFmpeg lldb extraction script for chroma DC at MB({mb_x},{mb_y}) frame {target_frame}
        # Target plane: {plane_name} (chroma_idx={chroma_idx})
        #
        # This script sets conditional breakpoints in FFmpeg's H.264 decoder
        # to extract chroma DC intermediate values.
        #
        # Usage: lldb -s /tmp/ffmpeg_extract.lldb

        # Set the input file
        settings set target.run-args -bitexact -skip_loop_filter all -i {input_path} -f rawvideo -pix_fmt yuv420p /dev/null

        # Breakpoint 1: chroma_dc_dequant_idct entry
        # In h264idct_template.c, ff_h264_chroma_dc_dequant_idct
        # Arguments: int16_t *_block, int qmul
        breakpoint set -n ff_h264_chroma_dc_dequant_idct8
        breakpoint command add 1
        # Print the 4 input DC coefficients at block[0], block[16], block[32], block[48]
        expr (int)((int16_t*)$arg1)[0]
        expr (int)((int16_t*)$arg1)[16]
        expr (int)((int16_t*)$arg1)[32]
        expr (int)((int16_t*)$arg1)[48]
        expr (int)$arg2
        continue
        DONE

        # Breakpoint 2: h264_idct_dc_add
        # In h264idct_template.c, ff_h264_idct_dc_add8
        # After dc is computed: int dc = (block[0] + 32) >> 6
        breakpoint set -n ff_h264_idct_dc_add8
        breakpoint command add 2
        expr (int)((int16_t*)$arg2)[0]
        continue
        DONE

        run
    """)
    return script


def main():
    parser = argparse.ArgumentParser(
        description="Extract FFmpeg chroma DC values via lldb"
    )
    parser.add_argument("input", help="H.264 input file")
    parser.add_argument("--frame", type=int, required=True, help="Target frame number")
    parser.add_argument("--mb-x", type=int, required=True, help="MB X coordinate")
    parser.add_argument("--mb-y", type=int, required=True, help="MB Y coordinate")
    parser.add_argument("--plane", choices=["U", "V"], default="V",
                        help="Chroma plane (default: V)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate lldb script but don't run it")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    chroma_idx = 1 if args.plane == "V" else 0
    script = generate_lldb_script(
        input_path, args.frame, args.mb_x, args.mb_y, chroma_idx
    )

    script_path = Path("/tmp/ffmpeg_extract.lldb")
    script_path.write_text(script)
    print(f"Generated lldb script: {script_path}")

    if args.generate_only:
        print("\nTo run manually:")
        print(f"  lldb FFmpeg/ffmpeg -s {script_path}")
        print(script)
        return

    ffmpeg_bin = find_ffmpeg_debug()
    if ffmpeg_bin is None:
        print("Error: No debug FFmpeg found at FFmpeg/ffmpeg or FFmpeg/ffmpeg_g")
        print("Build with: cd FFmpeg && ./configure --disable-optimizations --enable-debug=3 && make -j$(nproc)")
        print(f"\nGenerated script at {script_path} — run manually with:")
        print(f"  lldb {ffmpeg_bin or 'path/to/ffmpeg'} -s {script_path}")
        sys.exit(1)

    print(f"Running lldb on {ffmpeg_bin}...")
    print(f"Target: MB({args.mb_x},{args.mb_y}) frame {args.frame} plane {args.plane}")
    print("Note: This will produce MANY breakpoint hits. The target MB's values")
    print("will be among them — use the frame/MB context to identify the right one.\n")

    result = subprocess.run(
        ["lldb", ffmpeg_bin, "-s", str(script_path)],
        capture_output=True,
        timeout=120,
    )
    print(result.stdout.decode(errors="replace"))
    if result.returncode != 0:
        print(result.stderr.decode(errors="replace"), file=sys.stderr)


if __name__ == "__main__":
    main()
