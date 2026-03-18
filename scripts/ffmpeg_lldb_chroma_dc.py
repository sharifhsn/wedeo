#!/usr/bin/env python3
"""Extract chroma DC intermediate values from FFmpeg via lldb.

Sets breakpoints at key H.264 chroma decode points and extracts:
1. Chroma DC coefficients (input to chroma_dc_dequant_idct)
2. DC values after Hadamard (output)
3. The dc value passed to h264_idct_dc_add

Usage:
    # Generate lldb script for MB(7,8) on frame 16 of SVA_BA2_D
    python3 scripts/ffmpeg_lldb_chroma_dc.py \
        fate-suite/h264-conformance/SVA_BA2_D.264 \
        --frame 16 --mb-x 7 --mb-y 8

    # Auto-run if debug FFmpeg is available
    python3 scripts/ffmpeg_lldb_chroma_dc.py \
        fate-suite/h264-conformance/SVA_BA2_D.264 \
        --frame 16 --mb-x 7 --mb-y 8 --run

Requires:
    - FFmpeg built with debug info (see below)
    - lldb in PATH

Build FFmpeg with debug info:
    cd FFmpeg
    ./configure --disable-optimizations --enable-debug=3 --disable-stripping
    make -j$(sysctl -n hw.ncpu)
"""

import argparse
import platform
import subprocess
import sys
import textwrap
from pathlib import Path


def find_ffmpeg_debug():
    """Find a debug-built FFmpeg binary."""
    candidates = [
        Path("FFmpeg/ffmpeg_g"),
        Path("FFmpeg/ffmpeg"),
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
    is_arm = platform.machine() == "arm64"

    # On arm64 (Apple Silicon): args in x0, x1, x2, ...
    # On x86_64: args in rdi, rsi, rdx, ...
    if is_arm:
        block_reg = "$x0"  # int16_t *_block
        qmul_reg = "$x1"   # int qmul
        dst_reg = "$x0"    # uint8_t *_dst (for idct_dc_add)
        block2_reg = "$x1" # int16_t *_block (for idct_dc_add)
    else:
        block_reg = "$rdi"
        qmul_reg = "$rsi"
        dst_reg = "$rdi"
        block2_reg = "$rsi"

    script = textwrap.dedent(f"""\
        # FFmpeg lldb extraction script for chroma DC
        # Target: MB({mb_x},{mb_y}) frame {target_frame} plane {plane_name}
        #
        # Note: This dumps ALL chroma_dc_dequant_idct and idct_dc_add calls.
        # To find the target MB, count calls:
        #   - Each frame has 2 chroma_dc_dequant_idct calls (Cb + Cr)
        #   - Frame {target_frame} starts at call #{target_frame * 2 + 1}
        #   - The {plane_name} call is the {'2nd' if chroma_idx == 1 else '1st'} of the pair
        #
        # Platform: {'arm64' if is_arm else 'x86_64'}

        # Breakpoint 1: chroma_dc_dequant_idct
        # Signature: void ff_h264_chroma_dc_dequant_idct8(int16_t *_block, int qmul)
        breakpoint set -n ff_h264_chroma_dc_dequant_idct8
        breakpoint command add 1
        # Input DC coefficients at block[0], block[16], block[32], block[48]
        p (int)((int16_t*){block_reg})[0]
        p (int)((int16_t*){block_reg})[16]
        p (int)((int16_t*){block_reg})[32]
        p (int)((int16_t*){block_reg})[48]
        p (int){qmul_reg}
        continue
        DONE

        # Breakpoint 2: idct_dc_add (called for each 4x4 block with non-zero DC)
        # Signature: void ff_h264_idct_dc_add8(uint8_t *_dst, int16_t *_block, int stride)
        breakpoint set -n ff_h264_idct_dc_add8
        breakpoint command add 2
        # The DC value that will become (block[0] + 32) >> 6
        p (int)((int16_t*){block2_reg})[0]
        continue
        DONE

        process launch -- -bitexact -skip_loop_filter all -threads 1 -i {input_path} -f rawvideo -pix_fmt yuv420p -y /dev/null
        quit
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
    parser.add_argument("--run", action="store_true",
                        help="Auto-run lldb (default: generate script only)")
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

    ffmpeg_bin = find_ffmpeg_debug()

    if not args.run:
        print(f"Generated lldb script: {script_path}")
        print(f"\nTo run:")
        print(f"  lldb {ffmpeg_bin or 'FFmpeg/ffmpeg'} -s {script_path}")
        print(f"\nLook for call #{args.frame * 2 + (1 if chroma_idx == 0 else 2)} of chroma_dc_dequant_idct")
        print(f"(frame {args.frame}, {'Cb' if chroma_idx == 0 else 'Cr'} plane)\n")
        print(script)
        return

    if ffmpeg_bin is None:
        print("Error: No debug FFmpeg found at FFmpeg/ffmpeg_g or FFmpeg/ffmpeg", file=sys.stderr)
        print("Build: cd FFmpeg && ./configure --disable-optimizations --enable-debug=3 && make", file=sys.stderr)
        sys.exit(1)

    print(f"Running: lldb {ffmpeg_bin} -s {script_path}")
    print(f"Target: frame {args.frame}, MB({args.mb_x},{args.mb_y}), plane {args.plane}")
    print(f"Look for chroma_dc_dequant_idct call #{args.frame * 2 + (1 if chroma_idx == 0 else 2)}\n")

    result = subprocess.run(
        ["lldb", ffmpeg_bin, "-s", str(script_path)],
        timeout=300,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
