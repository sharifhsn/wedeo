#!/usr/bin/env python3
"""
LLDB Python script to extract motion vectors from FFmpeg's H.264 decoder.
Target: BA3_SVA_C.264, ALL B-slices at MB mb_x=7 mb_y=0

We print a summary for every B-slice hit and full detail for all of them.
"""

import lldb

AV_PICTURE_TYPE_I = 1
AV_PICTURE_TYPE_P = 2
AV_PICTURE_TYPE_B = 3

# scan8 table from h264_parse.h
SCAN8 = [
    4 + 1*8, 5 + 1*8, 4 + 2*8, 5 + 2*8,   # 12,13,20,21  (i8=0)
    6 + 1*8, 7 + 1*8, 6 + 2*8, 7 + 2*8,   # 14,15,22,23  (i8=1)
    4 + 3*8, 5 + 3*8, 4 + 4*8, 5 + 4*8,   # 28,29,36,37  (i8=2)
    6 + 3*8, 7 + 3*8, 6 + 4*8, 7 + 4*8,   # 30,31,38,39  (i8=3)
]

b_slice_count = 0
TARGET_MB_X = 7
TARGET_MB_Y = 0
STOP_AFTER = 10  # Print all B-frame hits, stop after this many


def print_mv_detail(frame, b_num):
    """Print full MV detail for a B-slice MB"""
    sl = frame.FindVariable("sl")
    mb_type_var = frame.FindVariable("mb_type")
    mb_type_val = mb_type_var.GetValueAsSigned() if mb_type_var.IsValid() else -1
    mb_xy = sl.GetChildMemberWithName("mb_xy").GetValueAsSigned()

    print(f"\nmb_type = {mb_type_val} (0x{mb_type_val & 0xFFFFFFFF:08x}), mb_xy = {mb_xy}")

    # mb_type flags
    flags = [
        ("INTRA4x4",   0x0001), ("INTRA16x16", 0x0002), ("INTRA_PCM",  0x0004),
        ("16x16",      0x0008), ("16x8",       0x0010), ("8x16",       0x0020),
        ("8x8",        0x0040), ("INTERLACED", 0x0080), ("DIRECT2",    0x0100),
        ("SKIP",       0x0800), ("P0L0",       0x1000), ("P0L1",       0x2000),
        ("P1L0",       0x4000), ("P1L1",       0x8000), ("CBP_LUMA",   0x10000),
    ]
    active = [name for name, mask in flags if mb_type_val & mask]
    print(f"  flags: {' | '.join(active)}")

    # Sub MB types
    print(f"  sub_mb_type: ", end="")
    for i in range(4):
        smt = frame.EvaluateExpression(f"sl->sub_mb_type[{i}]").GetValueAsUnsigned()
        sflags = [name for name, mask in flags if smt & mask]
        print(f"[{i}]=0x{smt:04x}({' | '.join(sflags)}) ", end="")
    print()

    # POC, frame_num, ref counts
    poc = frame.EvaluateExpression("h->cur_pic.poc").GetValueAsSigned()
    frame_num = frame.EvaluateExpression("sl->frame_num").GetValueAsSigned()
    ref_count0 = frame.EvaluateExpression("sl->ref_count[0]").GetValueAsSigned()
    ref_count1 = frame.EvaluateExpression("sl->ref_count[1]").GetValueAsSigned()
    dsp = frame.EvaluateExpression("sl->direct_spatial_mv_pred").GetValueAsSigned()
    print(f"  POC={poc}, frame_num={frame_num}, ref_count=[{ref_count0},{ref_count1}], direct_spatial={dsp}")

    # MVs for all 16 blocks, both lists
    for list_idx in range(2):
        print(f"\n  L{list_idx} mv_cache:")
        for i in range(16):
            s8 = SCAN8[i]
            i8 = i // 4
            mvx = frame.EvaluateExpression(f"sl->mv_cache[{list_idx}][{s8}][0]").GetValueAsSigned()
            mvy = frame.EvaluateExpression(f"sl->mv_cache[{list_idx}][{s8}][1]").GetValueAsSigned()
            ref = frame.EvaluateExpression(f"sl->ref_cache[{list_idx}][{s8}]").GetValueAsSigned()
            print(f"    [{i:2d}] i8={i8} s8={s8:2d}: mv=({mvx:4d},{mvy:4d}) ref={ref}")

    # Neighbor context
    for list_idx in range(2):
        print(f"\n  L{list_idx} cache grid (rows 0-4, cols 3-7):")
        for row in range(5):
            line = f"    r{row}: "
            for col in range(3, 8):
                idx = col + row * 8
                mvx = frame.EvaluateExpression(f"sl->mv_cache[{list_idx}][{idx}][0]").GetValueAsSigned()
                mvy = frame.EvaluateExpression(f"sl->mv_cache[{list_idx}][{idx}][1]").GetValueAsSigned()
                ref = frame.EvaluateExpression(f"sl->ref_cache[{list_idx}][{idx}]").GetValueAsSigned()
                line += f"({mvx:3d},{mvy:3d})r{ref:2d} "
            print(line)


def breakpoint_callback(frame, bp_loc, dict):
    global b_slice_count

    sl = frame.FindVariable("sl")
    mb_x = sl.GetChildMemberWithName("mb_x").GetValueAsSigned()
    mb_y = sl.GetChildMemberWithName("mb_y").GetValueAsSigned()
    slice_type_nos = sl.GetChildMemberWithName("slice_type_nos").GetValueAsSigned()

    if mb_x != TARGET_MB_X or mb_y != TARGET_MB_Y:
        return False

    pict_names = {0: "NONE", 1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP"}

    if slice_type_nos != AV_PICTURE_TYPE_B:
        return False

    b_slice_count += 1

    poc = frame.EvaluateExpression("h->cur_pic.poc").GetValueAsSigned()
    mb_type_var = frame.FindVariable("mb_type")
    mb_type_val = mb_type_var.GetValueAsSigned() if mb_type_var.IsValid() else -1

    print(f"\n{'='*70}")
    print(f"B-slice #{b_slice_count}: mb_x={mb_x}, mb_y={mb_y}, POC={poc}, mb_type=0x{mb_type_val & 0xFFFFFFFF:08x}")
    print(f"{'='*70}")

    print_mv_detail(frame, b_slice_count)

    if b_slice_count >= STOP_AFTER:
        print(f"\nReached {STOP_AFTER} B-slice hits, stopping.")
        return True

    return False


def __lldb_init_module(debugger, internal_dict):
    print("Loading B-frame MV extraction script (all B-slices)...")

    target = debugger.GetSelectedTarget()
    if not target:
        debugger.HandleCommand("target create FFmpeg/ffmpeg_g")
        target = debugger.GetSelectedTarget()

    bp = target.BreakpointCreateByLocation("h264_cavlc.c", 1030)
    bp.SetScriptCallbackFunction("lldb_extract_b_mv.breakpoint_callback")
    print(f"Breakpoint set at h264_cavlc.c:1030: {bp}")

    debugger.HandleCommand("run")
