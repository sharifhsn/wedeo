#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
Extract CABAC state from FFmpeg at key syntax element boundaries
inside ff_h264_decode_mb_cabac for MB(17,4) in the first frame.

Usage: lldb -o "command script import scripts/ffmpeg_cabac_trace.py" \
            -o "cabac_trace fate-suite/h264-conformance/CAMA1_Sony_C.jsv" \
            FFmpeg/ffmpeg
"""

import lldb


def cabac_state(frame):
    """Extract (pos, low, range) from sl->cabac."""
    sl = frame.FindVariable("sl")
    cabac = sl.GetChildMemberWithName("cabac")
    low = cabac.GetChildMemberWithName("low").GetValueAsSigned()
    rng = cabac.GetChildMemberWithName("range").GetValueAsSigned()
    bs = cabac.GetChildMemberWithName("bytestream").GetValueAsUnsigned()
    bs_start = cabac.GetChildMemberWithName("bytestream_start").GetValueAsUnsigned()
    pos = bs - bs_start
    return pos, low, rng


def cabac_trace(debugger, command, result, internal_dict):
    """Set breakpoints and run FFmpeg, printing CABAC state at each step."""
    target = debugger.GetSelectedTarget()
    if not target:
        print("No target selected")
        return

    input_file = command.strip() if command.strip() else "fate-suite/h264-conformance/CAMA1_Sony_C.jsv"

    # Key breakpoint lines in h264_cabac.c (FFmpeg n8.1):
    # After mb_type (I-slice path): line 2029
    bp_mb_type = target.BreakpointCreateByLocation("h264_cabac.c", 2030)
    bp_mb_type.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    # After intra4x4 modes: line 2093 (write_back_intra_pred_mode call)
    bp_intra4x4 = target.BreakpointCreateByLocation("h264_cabac.c", 2094)
    bp_intra4x4.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    # After chroma pred mode decode: line 2109
    bp_chroma = target.BreakpointCreateByLocation("h264_cabac.c", 2110)
    bp_chroma.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    # After CBP: line 2345
    bp_cbp = target.BreakpointCreateByLocation("h264_cabac.c", 2345)
    bp_cbp.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    # After qp_delta (both paths converge at IS_INTERLACED check): line 2428
    bp_qp = target.BreakpointCreateByLocation("h264_cabac.c", 2428)
    bp_qp.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    # After luma residual: line 2437 (next line after decode_cabac_luma_residual call)
    bp_luma = target.BreakpointCreateByLocation("h264_cabac.c", 2437)
    bp_luma.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    # After all residual (end of cbp>0 block): line 2495
    bp_end = target.BreakpointCreateByLocation("h264_cabac.c", 2495)
    bp_end.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    # Also set breakpoint at the start for initial state
    bp_start = target.BreakpointCreateByLocation("h264_cabac.c", 1922)
    bp_start.SetCondition("sl->mb_x == 17 && sl->mb_y == 4")

    step_names = {
        bp_start.GetID(): "START",
        bp_mb_type.GetID(): "STEP_1 after_mb_type",
        bp_intra4x4.GetID(): "STEP_2 after_intra4x4_modes",
        bp_chroma.GetID(): "STEP_3 after_chroma_pred",
        bp_cbp.GetID(): "STEP_4 after_cbp",
        bp_qp.GetID(): "STEP_5 after_qp_delta",
        bp_luma.GetID(): "STEP_6 after_luma_residual",
        bp_end.GetID(): "STEP_7 after_all_residual",
    }

    # Launch
    error = lldb.SBError()
    launch_info = lldb.SBLaunchInfo([
        "-bitexact", "-i", input_file,
        "-f", "framecrc", "-"
    ])
    launch_info.SetWorkingDirectory(target.GetDebugger().GetSelectedTarget().GetPlatform().GetWorkingDirectory() or ".")

    process = target.Launch(launch_info, error)
    if error.Fail():
        print(f"Launch failed: {error}")
        return

    frame_count = 0
    max_frames = 1  # Only trace first frame's MB(17,4)
    hit_count = 0

    while True:
        state = process.GetState()
        if state == lldb.eStateExited:
            print(f"Process exited with status {process.GetExitStatus()}")
            break
        if state == lldb.eStateStopped:
            thread = process.GetSelectedThread()
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                bp_id = thread.GetStopReasonDataAtIndex(0)
                frame = thread.GetSelectedFrame()

                step_name = step_names.get(bp_id, f"UNKNOWN(bp={bp_id})")

                # Get mb_x, mb_y
                sl = frame.FindVariable("sl")
                mb_x = sl.GetChildMemberWithName("mb_x").GetValueAsSigned()
                mb_y = sl.GetChildMemberWithName("mb_y").GetValueAsSigned()

                pos, low, rng = cabac_state(frame)

                # For mb_type step, also get the mb_type value
                extra = ""
                if "mb_type" in step_name:
                    mt = frame.FindVariable("mb_type")
                    if mt.IsValid():
                        extra = f" mb_type={mt.GetValueAsSigned()}"
                if "cbp" in step_name.lower():
                    cbp_var = frame.FindVariable("cbp")
                    if cbp_var.IsValid():
                        extra = f" cbp=0x{cbp_var.GetValueAsUnsigned():x}"

                print(f"FFmpeg {step_name} mb_x={mb_x} mb_y={mb_y}{extra} pos={pos} low={low} range={rng}")

                hit_count += 1
                if step_name.startswith("STEP_7") or step_name.startswith("START"):
                    if step_name.startswith("STEP_7"):
                        frame_count += 1
                        if frame_count >= max_frames:
                            print(f"\n--- Traced {frame_count} frame(s), {hit_count} breakpoint hits ---")
                            # Remove breakpoints and continue
                            for bp in [bp_start, bp_mb_type, bp_intra4x4, bp_chroma, bp_cbp, bp_qp, bp_luma, bp_end]:
                                target.BreakpointDelete(bp.GetID())
                            process.Continue()
                            continue

            process.Continue()
        elif state == lldb.eStateCrashed:
            print("Process crashed!")
            break
        else:
            import time
            time.sleep(0.01)

    print("Done.")


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand(
        'command script add -f ffmpeg_cabac_trace.cabac_trace cabac_trace'
    )
    print("Loaded 'cabac_trace' command. Usage: cabac_trace <input_file>")
