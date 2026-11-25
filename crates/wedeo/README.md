# wedeo

Use Wedeo's registry and typed readers to decode audio blocks or video frames in Rust. This example counts frames in the MP4 bundled with this checkout. Run it from the repository root with the default features enabled.

```rust
use wedeo::{default_registry, video::VideoReader};

fn main() -> wedeo::Result<()> {
    let registry = default_registry();
    let mut reader = VideoReader::open(&registry, "web/samples/h264_aac_320x240.mp4")?;
    let mut frames = 0;
    while reader.next_frame()?.is_some() {
        frames += 1;
    }
    println!("{frames} frames");
    Ok(())
}
```

For an audio-only application, disable default features and enable `audio-ops`, `pcm`, `wav`, and `symphonia`. For GPU conversion, enable `gpu-wgpu`. The optional AV1 decoder uses rav1d through a dav1d-compatible interface.

[CLI and browser studio](https://github.com/sharifhsn/wedeo) · [Feature definitions](Cargo.toml) · [License](../../LICENSE)
