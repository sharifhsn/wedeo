#!/usr/bin/env python3
"""Trace FFmpeg's get_cabac_bypass_sign via lldb breakpoints.

Since get_cabac_bypass_sign is always_inline, we set a breakpoint on
the source line `c->low += c->low;` (cabac_functions.h:169) which is
the first operation inside the function. At each hit we read the
CABACContext fields (low, range) and the `val` parameter, then continue.

Usage:
    python3 scripts/ffmpeg_lldb_bypass_sign.py fate-suite/h264-conformance/CABA2_SVA_B.264
    python3 scripts/ffmpeg_lldb_bypass_sign.py <file.264> --max-events 500
    python3 scripts/ffmpeg_lldb_bypass_sign.py <file.264> --max-events 500 -o /tmp/ff_bypass_sign.log

Prerequisites:
    - Debug FFmpeg binary at scripts/build/ffmpeg_cabac_trace
      (built with --disable-optimizations --enable-debug=3 --disable-asm)
    - lldb available on PATH (macOS ships it with Xcode)

Output format (one line per event, to match wedeo's CABAC_BYPASS_SIGN trace):
    CABAC_BYPASS_SIGN <n> low=<low> range=<range> val=<val>

Note: this only captures the PRE-state (before the bypass_sign operation).
The post-state would require stepping through the function which is too slow.
Compare pre-states between FFmpeg and wedeo to find the first divergence.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
FFMPEG_BIN = SCRIPT_DIR / "build" / "ffmpeg_cabac_trace"
# The source line where bypass_sign begins: c->low += c->low;
BYPASS_SIGN_FILE = "cabac_functions.h"
BYPASS_SIGN_LINE = 169


def main():
    parser = argparse.ArgumentParser(
        description="Trace FFmpeg get_cabac_bypass_sign via lldb"
    )
    parser.add_argument("input_file", help="H.264 input file")
    parser.add_argument(
        "--max-events",
        type=int,
        default=2000,
        help="Maximum bypass_sign events to capture (default: 2000)",
    )
    parser.add_argument(
        "-o", "--output", help="Output file (default: stdout)", default=None
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default=str(FFMPEG_BIN),
        help=f"Path to debug FFmpeg binary (default: {FFMPEG_BIN})",
    )
    args = parser.parse_args()

    if not Path(args.ffmpeg_bin).exists():
        print(f"Error: FFmpeg binary not found at {args.ffmpeg_bin}", file=sys.stderr)
        print(
            "Build it with: scripts/build_cabac_trace.sh", file=sys.stderr
        )
        sys.exit(1)

    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    max_events = args.max_events

    # Write an lldb command script that sets up the breakpoint with a
    # callback, runs the process, and exits.
    #
    # The breakpoint callback prints the pre-state values and counts events.
    # When max_events is reached, it kills the process.
    #
    # We use lldb's batch mode with a command file rather than the Python API
    # because the Python API requires importing lldb which isn't always on
    # the Python path outside of lldb's embedded interpreter.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lldb", delete=False, prefix="cabac_bp_"
    ) as f:
        lldb_script = f.name
        # Write an lldb Python script that will be sourced
        f.write(f"""\
# Set breakpoint on the first line of get_cabac_bypass_sign (inlined)
breakpoint set -f {BYPASS_SIGN_FILE} -l {BYPASS_SIGN_LINE}

# Add a Python callback for the breakpoint
breakpoint command add 1 -s python
import sys
count = getattr(sys.modules[__name__], '_bp_count', 0)
max_ev = {max_events}
frame = process.GetSelectedThread().GetSelectedFrame()
# Read c->low (before the doubling — this is the pre-state)
c_var = frame.FindVariable("c")
if c_var.IsValid():
    low = c_var.GetChildMemberWithName("low").GetValueAsSigned()
    rng = c_var.GetChildMemberWithName("range").GetValueAsSigned()
else:
    low = frame.EvaluateExpression("c->low").GetValueAsSigned()
    rng = frame.EvaluateExpression("c->range").GetValueAsSigned()
val_var = frame.FindVariable("val")
if val_var.IsValid():
    val = val_var.GetValueAsSigned()
else:
    val = frame.EvaluateExpression("val").GetValueAsSigned()
print(f"CABAC_BYPASS_SIGN {{count}} low={{low}} range={{rng}} val={{val}}", file=sys.stderr)
count += 1
sys.modules[__name__]._bp_count = count
if count >= max_ev:
    process.Kill()
DONE

# Run
run -bitexact -i {os.path.abspath(args.input_file)} -f null -
quit
""")

    out_file = args.output
    try:
        print(
            f"Running lldb on {args.ffmpeg_bin} with bypass_sign breakpoint "
            f"(max {max_events} events)...",
            file=sys.stderr,
        )
        print(
            f"  Breakpoint: {BYPASS_SIGN_FILE}:{BYPASS_SIGN_LINE}",
            file=sys.stderr,
        )

        cmd = ["lldb", "--batch", "--source", lldb_script, args.ffmpeg_bin]

        if out_file:
            with open(out_file, "w") as log:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    timeout=300,
                )
            print(f"Output written to {out_file}", file=sys.stderr)
        else:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                timeout=300,
            )

        if result.returncode not in (0, -9, -15, 1):
            print(
                f"Warning: lldb exited with code {result.returncode}",
                file=sys.stderr,
            )
    finally:
        os.unlink(lldb_script)


if __name__ == "__main__":
    main()
