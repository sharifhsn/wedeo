# wedeo

AI-generated Rust rewrite of FFmpeg — a pure-Rust multimedia framework for
decoding, encoding, demuxing, and muxing audio and video.

No bindgen, no c2rust, no FFI. Licensed under LGPL-2.1-or-later (same as
FFmpeg). Verification target: bit-for-bit output parity with FFmpeg's FATE
test suite.

## Quick start

```rust,no_run
use wedeo::{InputContext, CodecParameters, DecoderBuilder, Error};

fn main() -> wedeo::Result<()> {
    let mut input = InputContext::open("video.mp4")?;
    let stream = &input.streams[0];
    let mut decoder = DecoderBuilder::new(stream.codec_params.clone()).open()?;

    loop {
        match input.read_packet() {
            Ok(packet) => {
                decoder.send_packet(Some(&packet))?;
                while let Ok(frame) = decoder.receive_frame() {
                    // Process decoded frame...
                    let _ = frame;
                }
            }
            Err(Error::Eof) => break,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
```

## Status

- **H.264 progressive**: 79/79 FATE vectors, bit-exact
- **H.264 FRext**: 23/55 vectors passing
- **Audio**: PCM native + Symphonia adapter (AAC, FLAC, MP3, Vorbis, Opus)
- **Formats**: MP4/MOV, WAV, H.264 Annex B raw bitstream

See the [repository](https://github.com/sharifhsn/wedeo) for full status
and architecture details.

## License

LGPL-2.1-or-later
