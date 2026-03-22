# MP4 Muxer Roadmap

Current state: initial muxer landed on `feat/mp4-muxer` branch. Produces
ffprobe-validated MP4 files with ftyp, mdat (two-pass moov-at-end), moov with
full sample tables, H.264 avc1+avcC, AAC mp4a+esds, generic PCM (twos/sowt),
stts/ctts/stsc/stsz/stco/co64/stss. ~550 lines across `lib.rs`, `muxer.rs`,
`atoms.rs`.

The MP4 **demuxer** is handled by symphonia and is out of scope for this roadmap.

---

## Phase 1: Hardening

Fix bugs and edge cases in the initial implementation. Goal: robust enough to
be the default output format for `wedeo-cli decode -o output.mp4`.

### Bugs to fix

| Issue | Location | Fix |
|-------|----------|-----|
| mdat >4 GB truncates to u32 | `muxer.rs:180` `mdat_size as u32` | Use 64-bit extended size (8-byte `mdat` header with `size=1`, then 8-byte largesize) when `mdat_size > u32::MAX` |
| mvhd/mdhd duration >u32 overflows | `atoms.rs:99,197` cast to `u32` | Use version-1 boxes (64-bit timestamps) when duration exceeds u32 |
| No validation of codec support | `muxer.rs:93` blindly pushes any codec | Return `Error::UnsupportedCodec` for codecs without a known sample entry |
| Empty extradata for H.264/AAC | `atoms.rs:398` only checks `!is_empty()` | Return error in `write_header` if H.264 or AAC track has no extradata |
| Non-seekable output silently produces corrupt file | `muxer.rs:176` skips mdat patch | Return error if output is not seekable (two-pass requires it) |
| sample_rate=0 panic | `atoms.rs:416` `sample_rate << 16` overflows | Validate and default sample_rate before shift |

### Edge cases to handle

- **Single-sample tracks** -- stts/stsc/stsz must be valid with 1 entry
- **Audio-only MP4 (M4A)** -- ftyp should use `M4A ` major brand, no `avc1` compat brand
- **Zero-duration packets** -- clamp to 1 to avoid stts entries with delta=0
- **Interleaved multi-track** -- currently one sample per chunk; group consecutive
  same-track samples into chunks for better stsc compression (reduces file size
  and improves seek performance)

### Tests to add

- Roundtrip: mux H.264+AAC MP4, demux with symphonia, compare packet data
- Large offset: simulate >4 GB mdat, verify co64 is selected
- Audio-only: M4A with AAC, verify ffprobe reads it
- Error cases: no streams, unsupported codec, missing extradata

### Estimated size
~200 lines of fixes + ~150 lines of tests.

### Verification
- `cargo clippy --workspace` clean
- `ffprobe -v error` on all test outputs
- Roundtrip bitexact test for PCM-in-MP4 (mux then demux, compare packets)

### Dependencies
None. Self-contained.

---

## Phase 2: Faststart (moov-before-mdat)

Move moov before mdat so the file is web-playable without downloading the
entire file. This is the `-movflags +faststart` equivalent.

### Approach

Two strategies, implement both:

1. **Post-write relocation** (like FFmpeg's `faststart`): after `write_trailer`
   writes moov at end, read moov back, shift mdat forward by moov size, rewrite
   moov at the original mdat position, patch all chunk offsets by +moov_size.
   Requires the output to be readable+seekable.

2. **Reserved moov space** (optional, for known-size outputs): reserve N bytes
   before mdat with a `free` box. If moov fits, write it there and shrink the
   `free` box. If it doesn't fit, fall back to strategy 1.

### Boxes added/modified
- `free` -- placeholder box before mdat (strategy 2)
- All `stco`/`co64` entries need offset adjustment (+moov_size)

### API surface
- `Mp4MuxerOptions::faststart(bool)` -- opt-in (default false for backward compat)
- Builder pattern: `Mp4Muxer::with_options(options)`

### Estimated size
~250 lines (relocation logic + offset patching + option plumbing).

### Verification
- ffprobe: verify moov appears before mdat in output
- Web playback: serve file over HTTP range requests, verify browser plays
  without full download
- Bitexact: faststart output should decode identically to non-faststart

### Dependencies
Phase 1 (need 64-bit mdat support before relocating large files).

---

## Phase 3: Additional Codecs

Add sample description entries for codecs beyond H.264/AAC/PCM.

### 3a: HEVC (hvc1 + hvcC)

- `hvc1` sample entry in stsd (same visual sample entry layout as avc1)
- `hvcC` box: parse HEVCDecoderConfigurationRecord from extradata
- ftyp: add `hev1` compatible brand when HEVC track present
- Reference: `FFmpeg/libavformat/movenc.c` `mov_write_hvcc_tag()`

Estimated: ~120 lines. Verification: ffprobe + decode roundtrip with FFmpeg.

### 3b: AAC improvements

- SBR/HE-AAC: implicit signaling (no change to esds, but ftyp may need `iso6`)
- ADTS-to-raw conversion: strip ADTS headers if input packets have them
- `mp4a` version 1 fields for backward compatibility with older decoders

Estimated: ~80 lines.

### 3c: AC-3 / E-AC-3

- `ac-3` sample entry + `dac3` box (3 bytes: fscod, bsid, bsmod, acmod, lfeon, bit_rate_code)
- `ec-3` sample entry + `dec3` box (variable length)
- Reference: ETSI TS 102 366, `FFmpeg/libavformat/movenc.c` `mov_write_dac3_tag()`

Estimated: ~150 lines.

### 3d: Opus in MP4

- `Opus` sample entry + `dOps` box (OpusHead structure)
- ctts handling: Opus has a fixed pre-skip that needs an edit list
- Reference: RFC 7845 mapping to ISOBMFF (draft-ietf-codec-opus-in-isobmff)

Estimated: ~100 lines.

### Phase 3 total
~450 lines. Each sub-phase is independent and can land separately.

### Verification
- ffprobe validates each codec type
- `ffmpeg -i output.mp4 -f null -` decodes without errors
- Roundtrip test per codec (mux with wedeo, demux+decode with FFmpeg)

### Dependencies
Phase 1 (robust error handling needed before adding more codecs).

---

## Phase 4: Fragmented MP4 (fMP4)

Support `moof`+`mdat` fragments for DASH and HLS streaming. This is a
fundamentally different write mode from the moov-at-end approach.

### Box structure
```
ftyp (major_brand = iso6 or dash)
moov
  mvhd
  trak (with empty stbl — all sample info is in fragments)
  mvex
    trex  — track extends defaults (default sample duration/size/flags)
moof
  mfhd  — sequence_number
  traf
    tfhd  — track_id, base_data_offset, default flags
    tfdt  — baseMediaDecodeTime (version 1, 64-bit)
    trun  — sample_count, data_offset, per-sample duration/size/flags/cts_offset
mdat
  (packet data for this fragment)
moof + mdat ...  (repeated)
mfra (optional — random access index)
  tfra + mfro
```

### API surface
- `Mp4MuxerOptions::fragmented(true)` -- switches to fMP4 mode
- `Mp4MuxerOptions::fragment_duration(Duration)` -- target fragment length
  (default 2 seconds, matching DASH convention)
- `write_packet` accumulates samples; when fragment duration is reached,
  flushes a moof+mdat pair
- `write_trailer` flushes final fragment + optional mfra

### Key implementation details
- moov contains empty stbl (stts/stsc/stsz/stco with 0 entries) + mvex/trex
- Each trun can use `first_sample_flags` to mark the first sample as a keyframe
- `data_offset` in trun is relative to moof start
- Non-seekable output is supported (fragments are self-contained)
- Fragment boundary alignment: prefer to start fragments at keyframes for video

### Estimated size
~500 lines (new write mode + 6 new box types + fragment buffering logic).

### Verification
- `ffprobe -show_entries format_tags:format=nb_streams output.mp4` validates
- `mp4dump` (Bento4) shows correct fragment structure
- DASH manifest generation (out of scope, but verify fragments are independently decodable)
- `ffmpeg -i frag.mp4 -c copy -f null -` plays without errors

### Dependencies
Phase 1 (error handling). Independent of Phase 2 (faststart is irrelevant for
fMP4) and Phase 3 (codec entries are shared).

---

## Phase 5: Metadata & Polish

Add metadata boxes, edit lists, and display information for spec compliance
and player compatibility.

### 5a: Metadata (udta/meta/ilst)

- `udta` box at moov level
- `meta` full box with `hdlr` type `mdir`
- `ilst` with iTunes-style tags: title, artist, album, date, comment, genre,
  track number, cover art
- Accept metadata from `Stream`/output context metadata fields
- `Xtra` box for Windows Media metadata (optional, low priority)

Estimated: ~200 lines.

### 5b: Edit lists (elst)

- `edts` + `elst` box in each trak
- Initial delay: insert empty edit for tracks that don't start at PTS=0
- Offset compensation: when ctts has negative offsets, use an edit list
  with `media_time` instead of version-1 ctts (wider player compatibility)
- Opus pre-skip via edit list

Estimated: ~100 lines.

### 5c: Chapters

- `chpl` box (Nero-style chapters, simple) or
- Chapter track (`text` handler) with sample entries pointing to chapter names
- Read chapter info from input metadata if available

Estimated: ~80 lines.

### 5d: Display boxes

- `colr` box in visual sample entry: nclx (colour primaries, transfer, matrix)
  or ICC profile
- `pasp` box: pixel aspect ratio
- `clap` box: clean aperture (crop without re-encode)
- Source from SPS VUI parameters for H.264

Estimated: ~80 lines.

### 5e: Creation timestamps

- Set `creation_time` and `modification_time` in mvhd/tkhd/mdhd from
  system clock (currently hardcoded to 0)
- Use version-1 boxes for dates after 2036 (64-bit time since 1904)

Estimated: ~30 lines.

### Phase 5 total
~490 lines.

### Verification
- ffprobe `-show_format -show_tags` shows metadata
- QuickTime Player / VLC reads metadata fields
- `mediainfo` validates colr/pasp

### Dependencies
Phase 1. Independent of Phases 2-4.

---

## Summary

| Phase | What | Est. lines | Depends on |
|-------|------|-----------|------------|
| 1: Hardening | Bug fixes, error handling, edge cases, tests | ~350 | -- |
| 2: Faststart | moov-before-mdat for web playback | ~250 | Phase 1 |
| 3: Codecs | HEVC, AAC improvements, AC-3/E-AC-3, Opus | ~450 | Phase 1 |
| 4: Fragmented | moof+mdat for DASH/HLS | ~500 | Phase 1 |
| 5: Metadata | udta/ilst, edit lists, chapters, colr/pasp | ~490 | Phase 1 |
| **Total** | | **~2040** | |

Phases 2-5 are independent of each other after Phase 1 completes. They can
be implemented in any order based on priority. The recommended order is
1 -> 2 -> 3a -> 4 -> 5a -> rest, which prioritizes web playback and H.265
support.
