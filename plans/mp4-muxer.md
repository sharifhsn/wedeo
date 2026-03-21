# Plan: MP4/MOV Muxer

## Goal
Implement an MP4 (ISO Base Media File Format) muxer so wedeo can produce
actual video files, not just framecrc checksums.

## Deliverables
1. New crate `formats/wedeo-format-mp4/` with Muxer + MuxerFactory
2. Support writing H.264 video + AAC/PCM audio to .mp4
3. Workspace member in root Cargo.toml
4. Roundtrip test: demux → decode → encode → mux → verify playable
5. Zero clippy warnings, cargo fmt clean

## Architecture

```
formats/wedeo-format-mp4/
  Cargo.toml
  src/
    lib.rs          # MuxerFactory registration
    muxer.rs        # Mp4Muxer impl
    atoms.rs        # ftyp, moov, mdat, trak, stbl box writing
```

### MP4 structure (simplified)
```
ftyp            — file type (isom, mp41)
mdat            — raw packet data (written incrementally)
moov            — metadata (written at end or with faststart)
  mvhd          — movie header (duration, timescale)
  trak[0]       — video track
    tkhd        — track header
    mdia
      mdhd      — media header (timescale)
      hdlr      — handler (vide/soun)
      minf
        stbl    — sample table
          stsd  — sample description (avcC for H.264, esds for AAC)
          stts  — sample durations
          stsc  — sample-to-chunk mapping
          stsz  — sample sizes
          stco  — chunk offsets (into mdat)
          stss  — sync sample table (keyframes)
  trak[1]       — audio track (same structure)
```

### Two-pass vs streaming
**Two-pass (moov at end):** Write mdat first, accumulate sample tables in memory,
write moov at end. Simple but not streamable. This is the initial approach.

**Faststart (moov before mdat):** Write to temp, then rearrange. Or reserve space
for moov and patch. This is a future optimization.

## Key decisions

### Write from scratch vs wrap `mp4` crate
The `mp4` crate (0.14.0, by nicox) supports both reading and writing. However:
- It has its own frame/sample types that don't map cleanly to wedeo's Packet
- Writing from scratch gives us control over exact byte layout for FATE testing
- The MP4 box format is relatively simple (big-endian length-prefixed TLV)

**Decision:** Write from scratch. The box writing is mechanical and the `mp4` crate
would add complexity mapping between type systems.

### Extradata handling
- H.264: Packet extradata contains avcC (SPS/PPS). Write into avcC box in stsd.
- AAC: Packet extradata contains AudioSpecificConfig. Write into esds box.
- PCM: No extradata needed.

## Reference files to read FIRST
1. `formats/wedeo-format-wav/src/lib.rs` — existing muxer pattern (write_header/write_packet/write_trailer)
2. `crates/wedeo-format/src/muxer.rs` — Muxer trait + OutputFormatDescriptor
3. `FFmpeg/libavformat/movenc.c` — FFmpeg's MP4 muxer (reference for box layout)
4. ISO 14496-12 (ISOBMFF spec) — box definitions

## Conflict zones
- **Root Cargo.toml**: Adding workspace member
- **wedeo-cli/src/main.rs**: Adding `use wedeo_format_mp4 as _;`
- No framework crate changes needed (Muxer trait is sufficient)

## Estimated size
~800-1200 lines. Most of it is box serialization boilerplate.

## Verification
```bash
cargo check --workspace
cargo clippy --workspace
# Roundtrip test:
# 1. Decode H.264 to raw
# 2. Mux raw packets into MP4
# 3. Verify with ffprobe that the MP4 is valid
# 4. Decode the MP4 with FFmpeg and compare framecrc
ffprobe -v error -show_format -show_streams output.mp4
```
