# Wedeo vs FFmpeg: Behavioral Divergences

This document tracks known differences between wedeo and FFmpeg's behavior.
These are not bugs — they are architectural choices or consequences of using
symphonia as the audio backend instead of reimplementing FFmpeg's code.

## Channel Detection

### AAC channel configuration

**FFmpeg**: The MOV/MP4 demuxer parses the `esds` atom and calls
`avpriv_mpeg4audio_get_config2()` to extract channel count from the
AudioSpecificConfig *before* the decoder opens. Channels are known at
demuxer level.

**Wedeo**: Symphonia's MP4 demuxer sometimes reports `channels: None`.
We use two fallbacks:
1. Parse AudioSpecificConfig bits 9-12 in `guess_channels_from_extradata()`
2. Detect from `AudioBufferRef::spec().channels` after the first decode

**Impact**: The first decoded Frame may have `nb_channels=0` if neither
fallback fires before decode. In practice, the extradata fallback catches
all common cases.

### Opus channel configuration

**FFmpeg**: The Ogg/MKV demuxer reads the Opus ID header (RFC 7845) and
sets channel count at demux time.

**Wedeo**: Symphonia's demuxer may not expose channel count for Opus tracks
in MKV containers. We extract it from the Opus ID header in extradata
(byte 9) or default to stereo.

### Mono representation

**FFmpeg**: Mono is represented as `FrontCenter` (FC) in all contexts.

**Wedeo**: Symphonia reports mono as `FRONT_LEFT` with count=1 for formats
that lack explicit channel mapping (e.g., basic WAV without
WAVEFORMATEXTENSIBLE). We special-case this in `channels_to_layout()` to
return `ChannelLayout::mono()` (FC), matching FFmpeg.

## Packet Boundaries

### PCM packet sizing

**FFmpeg**: Uses `ff_pcm_default_packet_size` which computes
`bitrate / 8 / 10 / block_align`, rounded down to a power of 2, then
multiplied by `block_align`.

**Wedeo**: Symphonia uses its own packet sizing logic. Packet boundaries
differ from FFmpeg, so framecrc output is not line-for-line identical.
However, the total decoded audio data is byte-identical.

### MP3 gapless trimming

**FFmpeg**: Trims encoder delay (priming samples) and padding based on
the LAME/Xing header or iTunes gapless metadata. Output sample count
matches the original audio duration.

**Wedeo**: Symphonia's gapless mode (`enable_gapless: true`) trims
encoder delay via LAME/Xing headers. Sample count difference vs FFmpeg
is now bounded by ~1-3 MP3 frames. SNR exceeds 120 dB.

### AAC priming samples

**FFmpeg**: Trims the standard 2048-sample AAC encoder delay using
`initial_padding` from the container's `edit list` or `iTunSMPB` metadata.

**Wedeo**: Parses `iTunSMPB` metadata from M4A containers to determine
encoder delay and padding, then trims via `trim_start`/`trim_end` on
packets. Symphonia's MP4 demuxer does not natively support gapless, so
this is handled in the wedeo adapter layer. Sample count difference vs
FFmpeg is bounded by ~5 AAC frames. For raw ADTS AAC (no container),
no trimming is applied (same as before).

## Decoder Implementation Differences

### Opus decoder quality

**FFmpeg**: Uses libopus (C reference implementation) via FFI, or its own
internal decoder. Both produce identical output.

**Wedeo**: Uses the `opus-decoder` crate (0.1.1), a pure-Rust
reimplementation. Quality varies by mode:
- CELT mode: ~48 dB SNR vs FFmpeg (good, float precision differences)
- SILK mode: ~11-14 dB SNR vs FFmpeg (poor, real accuracy gaps)
- Hybrid mode: ~13-14 dB SNR vs FFmpeg (same class of issue)

The SILK/hybrid quality gap is a limitation of the opus-decoder crate's
current implementation, not a wedeo adapter issue.

### Vorbis floating-point precision

**FFmpeg**: Internal Vorbis decoder compiled with platform-specific float
instruction scheduling.

**Wedeo**: Symphonia's Vorbis decoder is pure Rust. Produces ~140 dB SNR
vs FFmpeg — only the last bit of IEEE 754 mantissa differs. Functionally
bitexact.

### Lossy codec output format

**FFmpeg**: Lossy codecs (MP3, AAC, Vorbis) typically output `fltp`
(float planar) format.

**Wedeo**: All audio output is interleaved (packed). Symphonia decodes to
planar buffers internally, but `audio_buffer_to_frame()` interleaves them
into a single packed plane. This matches FFmpeg's behavior after
`aresample` or when the output format is explicitly set to packed.

## I/O Architecture

### Demuxer I/O ownership

**FFmpeg**: The demuxer borrows the I/O context (`AVIOContext`) — it never
owns it. The `AVFormatContext` manages the I/O lifecycle.

**Wedeo**: Symphonia's `FormatReader` owns its I/O source. We transfer
ownership via `BufferedIo::take_inner()`, replacing the inner `IoContext`
with a dead stub. After `read_header()`, the `BufferedIo` parameter to
`read_packet()` and `seek()` is ignored — all I/O goes through symphonia's
internal reader.

### Write buffering

**FFmpeg**: `AVIOContext` has unified read/write buffering with separate
read and write positions.

**Wedeo**: `BufferedIo` has independent `read_buf` and `write_buf`. Read
buffering is invalidated on seek/flush. Write buffering flushes on seek
or explicit `flush()` call. The two buffers are never active simultaneously
in practice (a given `BufferedIo` is used for either reading or writing).

## Probe and Registry

### Priority-based codec/format selection

**FFmpeg**: Uses a single flat list of codecs/formats. When multiple
implementations exist (e.g., internal vs external libopus), build-time
configuration controls which is linked.

**Wedeo**: Uses runtime priority-based selection. Multiple implementations
can coexist (e.g., native PCM at priority 100, symphonia PCM at priority
100). `find_decoder()` picks the highest priority; `probe()` uses priority
as a tie-breaker when probe scores are equal.

### Probe score values

**FFmpeg**: WAV probe returns `AVPROBE_SCORE_MAX - 1` (99) for RIFF+WAVE,
`AVPROBE_SCORE_MAX` (100) for RF64/BW64.

**Wedeo**: Same scores. The symphonia WAV demuxer replicates FFmpeg's exact
probe scoring including the ACT demuxer conflict avoidance.

## Timestamp Handling

### Negative timestamps

**FFmpeg**: Timestamps can be negative. `int64_t` is used throughout.
`NOPTS_VALUE` is `INT64_MIN`.

**Wedeo**: Same convention (`i64`, `NOPTS_VALUE = i64::MIN`). However,
when passing timestamps to symphonia, we cast `i64` to `u64` which wraps
negative values. This could cause issues for files with negative DTS, but
is safe for typical audio files.

## Exp-Golomb Overflow Behavior

### Signed exp-Golomb at extreme ue values

**FFmpeg**: `get_se_golomb_long` computes `((buf >> 1) ^ sign) + 1` where
the `+ 1` overflows `int` for ue = 0xFFFFFFFF. C signed overflow is UB,
but FFmpeg relies on wrapping (compiled with `-fwrapv`).

**Wedeo**: Same formula in `get_se_golomb`. In Rust debug mode, the `+ 1`
would panic on i32 overflow. In release mode it wraps. This is unreachable
in practice — H.264 syntax elements constrain ue to ~100,000 max.

**Impact**: None for valid bitstreams. Debug-mode decode of a deliberately
corrupted stream could panic instead of producing garbage.

## H.264 Decoder

### Bitexact status

**BA1_Sony_D.jsv**: Bitexact. All 17 frames (176x144, Baseline, QP=28,
single-slice I-frames) produce byte-identical YUV420p output to
FFmpeg 8.0.1 (`-cpuflags 0`). All framecrc checksums match the FATE
reference file `h264-conformance-ba1_sony_d`.

### CAVLC level decoding

**FFmpeg**: Uses pre-computed `cavlc_level_tab` lookup tables for fast level
decoding. Small levels (fitting in a 16-bit peek) take a direct lookup path;
large levels fall back to prefix/suffix parsing.

**Wedeo**: Uses direct prefix/suffix parsing for all levels (no lookup table
acceleration). Produces identical coefficient values to FFmpeg for all tested
content.

**Impact**: None for single-slice files (bitexact verified). Multi-slice files
may still have a subtle desync — under investigation.

### Dequantization approach

**FFmpeg**: Pre-computes a dequantization table:
`dequant4_coeff[list][qp][pos] = init[qp%6][class(pos)] * scalingMatrix[pos] << (qp/6 + 2)`.
This embeds a 64x normalization factor that is compensated by the IDCT's `>>6`.
The dequantized coefficient stored as `int16_t` overflows for typical QP values.
FFmpeg's C template code (`h264idct_template.c`) has this overflow — verified
by compiling a standalone C test that reproduces wrong pixel values from the
overflow. FFmpeg works in practice because the dequant normalization is fused
inline in `decode_residual` as `(level * qmul + 32) >> 6`, producing smaller
intermediate values that fit in int16.

**Wedeo**: Uses `dequant_4x4_flat()` which applies the spec-equivalent formula
without the intermediate 64x factor:
`level * DEQUANT4_COEFF_INIT[qp%6][pos_class] << (qp/6)`.
For the default flat-16 scaling matrix (Baseline/Main profile), this produces
identical final pixel values because `scalingMatrix=16` cancels with the
normalization: `16 * 2^(-4) = 1`. The DC Hadamard paths output i32 values
to avoid overflow, which are either fed through the IDCT (for blocks with AC)
or applied directly to pixels with `(dc + 32) >> 6` rounding (for DC-only blocks).

**Impact**: Bitexact. Non-default scaling matrices (High profile) will require
a per-position flat dequant variant.

### H.264 raw bitstream demuxer AU grouping

**FFmpeg**: The H.264 raw demuxer (`h264dec.c`) uses a full H.264 parser
(`h264_parser.c`) to detect access unit boundaries, tracking frame_num,
pic_order_cnt, and field_pic_flag changes.

**Wedeo**: Uses a simplified AU boundary detection: AUD NALs, SPS/PPS/SEI NALs
before slices, and `first_mb_in_slice == 0` (detected by checking if the first
exp-golomb bit is '1'). This is correct for conformance test files but may
mis-group NALs in streams with unusual NAL ordering.

**Impact**: Works correctly for FATE test files. May need parser-based grouping
for real-world streams.

### framecrc video decode mode

**FFmpeg**: The framecrc muxer checksums either raw packet data (`-c copy`) or
decoded frame data (when transcoding). For video conformance tests, FFmpeg
decodes the video and checksums the raw YUV output.

**Wedeo**: The framecrc tool auto-detects video streams and uses decode mode
(matching FFmpeg's decode behavior). Audio streams use packet passthrough mode
(matching FFmpeg's `-c copy`). The output format matches FFmpeg's header lines
(`#codec_id 0: rawvideo`, `#dimensions 0: WxH`, `#sar 0: N/D`).

## Adler-32 Checksum

**FFmpeg**: Initializes Adler-32 with `s1=0, s2=0` (non-standard, differs
from RFC 1950's `s1=1`).

**Wedeo**: Matches FFmpeg's non-standard init using
`adler2::Adler32::from_checksum(0)` in the framecrc tool. The standard
Adler-32 (`adler2::Adler32::new()`) is used elsewhere.
