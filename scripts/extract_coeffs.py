#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Extract dequantized 4x4 coefficients from FFmpeg for a specific MB via lldb.

Usage:
    python3 scripts/extract_coeffs.py <input_file> <mb_x> <mb_y> <frame_num>

Breaks at ff_h264_idct_add to capture the coefficient block before IDCT.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "fate-suite/h264-conformance/FRext/Freh2_B.264"
    target_mb_x = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    target_mb_y = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    target_frame = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    input_path = Path(input_file).resolve()
    ffmpeg = Path("FFmpeg/ffmpeg").resolve()

    lldb_script = f"""
import lldb

target_mb_x = {target_mb_x}
target_mb_y = {target_mb_y}
target_frame = {target_frame}
frame_count = 0
block_count = 0

def stop_handler(frame, bp_loc, dict):
    global frame_count, block_count
    thread = frame.GetThread()
    process = thread.GetProcess()
    target = process.GetTarget()

    # Get sl pointer
    sl_val = frame.FindVariable("sl")
    if not sl_val.IsValid():
        # Try to find via expression
        sl_val = frame.EvaluateExpression("sl")

    mb_x = frame.EvaluateExpression("(int)sl->mb_x").GetValueAsSigned()
    mb_y = frame.EvaluateExpression("(int)sl->mb_y").GetValueAsSigned()
    poc = frame.EvaluateExpression("(int)h->poc.frame_num").GetValueAsSigned()

    if mb_x == target_mb_x and mb_y == target_mb_y and poc == target_frame:
        # Extract the 16 dequantized coefficients from _block
        block_ptr = frame.FindVariable("_block")
        if not block_ptr.IsValid():
            block_ptr = frame.EvaluateExpression("_block")

        coeffs = []
        for i in range(16):
            val = frame.EvaluateExpression(f"(int)((int16_t*)_block)[{{i}}]").GetValueAsSigned()
            coeffs.append(val)

        print(f"FFmpeg MB({{mb_x}},{{mb_y}}) poc={{poc}} block={{block_count}} coeffs={{coeffs}}")
        block_count += 1
    return False  # Don't stop

# Set breakpoint at ff_h264_idct_add (8-bit version: ff_h264_idct_add_8)
bp = lldb.debugger.GetSelectedTarget().BreakpointCreateByName("ff_h264_idct_add_8")
if bp.GetNumLocations() == 0:
    bp = lldb.debugger.GetSelectedTarget().BreakpointCreateByName("ff_h264_idct_add")
bp.SetScriptCallbackFunction("stop_handler")
bp.SetAutoContinue(True)

# Run
lldb.debugger.GetSelectedTarget().GetProcess().Continue()
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(lldb_script)
        script_path = f.name

    # Write lldb commands
    lldb_cmds = f"""
target create {ffmpeg}
settings set -- target.run-args -bitexact -i {input_path} -vframes 5 -f null -
command script import {script_path}
run
quit
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lldb', delete=False) as f:
        f.write(lldb_cmds)
        cmds_path = f.name

    result = subprocess.run(
        ["lldb", "--batch", "-s", cmds_path],
        capture_output=True, text=True, timeout=60,
        cwd=str(Path.cwd())
    )
    print(result.stdout)
    if result.stderr:
        # Filter for our output lines
        for line in result.stderr.split('\n'):
            if 'FFmpeg MB' in line:
                print(line)

if __name__ == "__main__":
    main()
