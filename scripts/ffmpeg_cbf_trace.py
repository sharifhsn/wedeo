#!/usr/bin/env python3
"""
Extract per-block CBF and NNZ neighbors from FFmpeg for MB(17,4) in first frame.
Sets breakpoint inside decode_cabac_residual_nondc where CBF is checked.

Usage: lldb -b -o "command script import scripts/ffmpeg_cbf_trace.py" FFmpeg/ffmpeg
"""

import lldb
import os


def __lldb_init_module(debugger, internal_dict):
    target = debugger.GetSelectedTarget()

    # Breakpoint at the CBF check in decode_cabac_residual_nondc (line 1859)
    # This is inside the inlined function, so we need to set it by file+line.
    # But since it's always_inline, we need the caller context.
    #
    # Alternative: set breakpoint at decode_cabac_residual_nondc_internal (line 1798)
    # which is the called function AFTER CBF passes.
    # Actually, let's set at line 1906 inside decode_cabac_luma_residual
    # (after decode_cabac_residual_nondc call for each 4x4 block)

    # Set breakpoint at the for loop body for 4x4 blocks: line 1906
    bp = target.BreakpointCreateByLocation("h264_cabac.c", 1906)
    bp.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    # Also set breakpoint at the "fill_rectangle" for uncoded blocks: line 1910
    bp_uncoded = target.BreakpointCreateByLocation("h264_cabac.c", 1910)
    bp_uncoded.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    # Set breakpoint AFTER the luma residual to stop: line 2437
    bp_end = target.BreakpointCreateByLocation("h264_cabac.c", 2437)
    bp_end.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")
    bp_end.SetOneShot(True)

    bp_ids = {
        bp.GetID(): "CODED",
        bp_uncoded.GetID(): "UNCODED_8x8",
        bp_end.GetID(): "END",
    }

    launch_info = lldb.SBLaunchInfo([
        "-bitexact", "-i", "fate-suite/h264-conformance/CAMA1_Sony_C.jsv",
        "-f", "framecrc", "-"
    ])
    launch_info.SetWorkingDirectory(os.getcwd())

    error = lldb.SBError()
    process = target.Launch(launch_info, error)
    if error.Fail():
        print(f"ERROR: {error}")
        return

    count = 0
    while process.GetState() != lldb.eStateExited:
        if process.GetState() == lldb.eStateStopped:
            thread = process.GetSelectedThread()
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                bp_id = thread.GetStopReasonDataAtIndex(0)
                step_name = bp_ids.get(bp_id, f"UNK({bp_id})")
                frame = thread.GetSelectedFrame()

                sl = frame.FindVariable("sl")
                cabac = sl.GetChildMemberWithName("cabac")
                low = cabac.GetChildMemberWithName("low").GetValueAsSigned()
                rng = cabac.GetChildMemberWithName("range").GetValueAsSigned()
                bs = cabac.GetChildMemberWithName("bytestream").GetValueAsUnsigned()
                bs_start = cabac.GetChildMemberWithName("bytestream_start").GetValueAsUnsigned()
                pos = bs - bs_start

                extra = ""
                i8x8_var = frame.FindVariable("i8x8")
                i4x4_var = frame.FindVariable("i4x4")
                if i8x8_var.IsValid():
                    extra += f" i8x8={i8x8_var.GetValueAsSigned()}"
                if i4x4_var.IsValid():
                    extra += f" i4x4={i4x4_var.GetValueAsSigned()}"

                idx_var = frame.FindVariable("index")
                if idx_var.IsValid() and step_name == "CODED":
                    extra += f" idx={frame.EvaluateExpression('16*p + 4*i8x8 + i4x4').GetValueAsSigned()}"

                print(f"FFmpeg {step_name}{extra} pos={pos} low={low} range={rng}")

                if step_name == "END":
                    # Delete all breakpoints and continue
                    for bid in bp_ids:
                        target.BreakpointDelete(bid)
                    process.Continue()
                    continue

                count += 1

            process.Continue()
        else:
            import time
            time.sleep(0.001)
            if process.GetState() not in (lldb.eStateStopped, lldb.eStateRunning):
                break

    print(f"\n--- {count} block hits ---")
