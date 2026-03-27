#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
Extract FFmpeg CABAC state at key points for MB(17,4) using lldb Python API.

Usage: lldb -b -o "command script import scripts/ffmpeg_cabac_extract.py" FFmpeg/ffmpeg
"""

import lldb
import os


def get_cabac_state(frame):
    """Extract (pos, low, range) from sl->cabac."""
    err = lldb.SBError()
    sl = frame.FindVariable("sl")
    if not sl.IsValid():
        # Try from function args
        sl = frame.EvaluateExpression("sl")
    cabac = sl.GetChildMemberWithName("cabac")
    low = cabac.GetChildMemberWithName("low").GetValueAsSigned()
    rng = cabac.GetChildMemberWithName("range").GetValueAsSigned()
    bs = cabac.GetChildMemberWithName("bytestream").GetValueAsUnsigned()
    bs_start = cabac.GetChildMemberWithName("bytestream_start").GetValueAsUnsigned()
    pos = bs - bs_start
    return pos, low, rng


def __lldb_init_module(debugger, internal_dict):
    target = debugger.GetSelectedTarget()

    # Breakpoint lines in h264_cabac.c and their step names
    steps = [
        (2030, "STEP_1 after_mb_type"),
        (2094, "STEP_2 after_intra4x4"),
        (2110, "STEP_3 after_chroma_pred"),
        (2345, "STEP_4 after_cbp"),
        (2428, "STEP_5 after_qp_delta"),
        (2437, "STEP_6 after_luma_residual"),
        (2495, "STEP_7 after_all_residual"),
    ]

    bp_ids = {}
    for line, name in steps:
        bp = target.BreakpointCreateByLocation("h264_cabac.c", line)
        bp.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")
        bp.SetOneShot(True)
        bp_ids[bp.GetID()] = name

    # Launch with arguments
    launch_info = lldb.SBLaunchInfo([
        "-bitexact", "-i", "fate-suite/h264-conformance/CAMA1_Sony_C.jsv",
        "-f", "framecrc", "-"
    ])
    cwd = os.getcwd()
    launch_info.SetWorkingDirectory(cwd)

    error = lldb.SBError()
    process = target.Launch(launch_info, error)
    if error.Fail():
        print(f"ERROR: Launch failed: {error}")
        return

    results = []
    while process.GetState() != lldb.eStateExited:
        if process.GetState() == lldb.eStateStopped:
            thread = process.GetSelectedThread()
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                bp_id = thread.GetStopReasonDataAtIndex(0)
                step_name = bp_ids.get(bp_id, f"UNKNOWN({bp_id})")
                frame = thread.GetSelectedFrame()
                pos, low, rng = get_cabac_state(frame)

                extra = ""
                if "cbp" in step_name.lower():
                    cbp = frame.EvaluateExpression("cbp").GetValueAsUnsigned()
                    extra = f" cbp=0x{cbp:x}"

                line = f"FFmpeg {step_name} pos={pos} low={low} range={rng}{extra}"
                results.append(line)
                print(line)

            process.Continue()
        else:
            import time
            time.sleep(0.001)
            if process.GetState() not in (lldb.eStateStopped, lldb.eStateRunning):
                break

    print(f"\n--- {len(results)} breakpoint hits ---")
    print(f"Process exited with status {process.GetExitStatus()}")
