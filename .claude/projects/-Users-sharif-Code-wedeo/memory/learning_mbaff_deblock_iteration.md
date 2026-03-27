---
name: MBAFF deblock iteration order
description: FFmpeg iterates MBAFF deblocking column-first within pair rows, not raster order
type: project
---

MBAFF deblocking iteration order: FFmpeg's `loop_filter()` (h264_slice.c:2451-2452) iterates column-first within each pair row: for each mb_x, deblock(mb_x, pair_top) then deblock(mb_x, pair_bot) before moving to mb_x+1. Measurably reduces pixel diffs (Y≤8 vs Y≤10 for CAMA1_Sony_C). Implemented in `deblock_frame()` with `is_mbaff` parameter.

**Why:** The order matters because deblocking MB(x,y_bot) modifies MB(x,y_top)'s bottom rows, and the next column MB(x+1,y_top) reads those modified rows for its vertical edge 0.

**How to apply:** When implementing deblocking for any interlaced format, check FFmpeg's iteration order carefully — it's not always raster.
