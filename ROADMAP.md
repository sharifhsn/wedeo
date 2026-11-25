# Wedeo roadmap

The next milestone is a faster native GPU scaler, a more usable audio preview for long recordings, and broader H.264 coverage. The starting point is main at `69d7eb3`. Complete the three items in order, with GPU performance first.

## 1. Reuse GPU buffers between video frames

The previous optimization reduced near-1080p upload time by 9.4% on the RTX 2070 SUPER. Planar conversion still allocates a storage buffer and a readback buffer for each output plane on each frame. A recording typically keeps its dimensions and pixel format for many successive frames.

- Measure release-build YUV420P conversion at 1280×720 → 640×360 and 1920×1080 → 960×540. Record cold startup separately from steady-state conversion and readback.
- Reuse scratch buffers across successive frames of a compiled conversion. Keep outstanding GPU work, map failures, and independent conversions isolated.
- **Done when** a runtime check demonstrates zero new planar storage or staging buffer allocations after warm-up at fixed dimensions. Compare every visible output byte with the original implementation on patterned even- and odd-sized frames. Report paired timings and a whole-CLI transcode on the same adapter.

## 2. Audition a selected interval in a long recording

Today the clipper decodes the entire selected audio track into memory. For a short edit in a long recording, a track switch should require only the samples needed for preview.

- Decode the selected interval with the preroll the codec requires. Release superseded audio when the user changes tracks or marks.
- **Done when** the browser can audition either track in a two-minute, two-tone recording at a nonzero in-point. For a ten-second selection, retained decoded PCM must stay bounded by the selection plus documented decoder overhead. Verify the selected tone, compare its timeline with an independent decode, and export a clip with both tracks still synchronized to the video. Preserve the existing open- and closed-GOP checks.

## 3. Expand H.264 conformance beyond the passing baseline

Start with the JVT-AVC field-coded and MBAFF cases `cabac_mot_fld0_full` and `cabac_mot_mbaff0_full`. Both are in the manifest and outside the current passing snapshot. Their current decoder results need to be measured.

- Run both vectors against their published expected output. For a mismatch, locate the first divergent picture and correct the decoder path. Exercise serial and multithread decoding where supported.
- **Done when** at least one previously uncovered field-coded or MBAFF vector matches its published digest and is included in the required regression set. Record the outcome of both named probes, including any remaining mismatch. Keep every existing passing vector in the baseline.

## Completion evidence

Merge the implementation and repeatable checks through the protected main branch. Record the GPU benchmark inputs, adapter, raw samples and pixel comparisons; the browser preview bounds and synchronization result; and the codec vector name and digest. Verify that the deployed clipper runs the accepted preview test. Report each measured gain at the scope actually tested.

[Existing architecture and longer-term phases](docs/ARCHITECTURE.md)
