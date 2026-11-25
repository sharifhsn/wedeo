# Wedeo roadmap

This milestone started at `69d7eb3` with GPU performance first. The results below cover planar scaling, long-recording audio preview, and H.264 conformance.

## Results

| Milestone | Verified outcome |
|---|---|
| [Planar GPU conversion](https://github.com/sharifhsn/wedeo/pull/20) | Six buffers on the first YUV420P conversion, zero new storage or staging buffers on subsequent frames. Two interleaved converters kept their outputs independent. In the final RTX 2070 SUPER release runs, 720p → 360p conversion plus readback fell from 3.00 to 2.39 ms; 1080p → 540p fell from 6.39 to 4.74 ms. Even- and odd-size outputs matched byte for byte. Whole-CLI medians were 1.87 and 1.88 s with identical decoded output. [Raw timings](benchmarks/planar-timings.csv) · [Setup times, inputs and output hashes](benchmarks/planar-results.json) |
| [Interval audition](https://github.com/sharifhsn/wedeo/pull/21) | A 60–70 s preview of either track in a two-minute AAC recording retained 480,000 mono samples (1.92 MB at 48 kHz). Each audition decoded 517 packets including preroll. Both tones aligned with an independent FFmpeg decode, and the exported video and both audio streams preserved packet payloads and relative timestamps. [Runtime regression](tests/e2e/check_web_clip.mjs) |
| [H.264 MBAFF](https://github.com/sharifhsn/wedeo/pull/22) | Wedeo now finishes the deferred deblocking row when a CABAC slice terminates at the right edge. CAMA1_Sony_C matched the published MD5 `c234cc1b862b5b2ef958622fce77b7f5` in serial and multithread decoding. The required JVT-AVC snapshot grew from 78 to 79 vectors. |

**Remaining codec differences.** On `cabac_mot_mbaff0_full`, the first filtered picture went from 11,374 differing samples to an exact match. The next picture still diverges. On `cabac_mot_fld0_full`, Wedeo emits 60 full-height pictures where the reference emits 30 paired frames. Both Motorola streams remain outside the passing snapshot.

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
