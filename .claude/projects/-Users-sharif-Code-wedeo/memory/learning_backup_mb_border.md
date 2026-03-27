---
name: backup_mb_border is irrelevant to deblocking
description: FFmpeg's backup_mb_border saves pixels for intra prediction, NOT for the deblock filter
type: feedback
---

FFmpeg's `backup_mb_border` (h264_slice.c:586-681) saves MB border pixels for `xchg_mb_border` during intra prediction — NOT for deblocking. Proof: grep "top_borders" in h264_loopfilter.c returns zero hits. The deblocking filter reads directly from the picture buffer.

**Why:** Investigated this as potential root cause for MBAFF deblock diffs, wasting ~10 minutes. One grep would have ruled it out.

**How to apply:** When debugging deblock diffs, don't investigate backup_mb_border or xchg_mb_border. These only affect reconstruction, not filtering.
