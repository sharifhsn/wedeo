# Wedeo

Wedeo is a Rust media toolkit that lets you look inside a video file, decode its frames, and change how its audio and video are packaged. The same H.264 and VP9 decoders run in the command-line tool and in a browser through WebAssembly.

![Wedeo decoding the bundled MP4 in Firefox, with video, audio, and container information visible](web/studio.png)

[Open the studio](https://wedeo.pages.dev/) · [Download the standalone studio](https://wedeo.pages.dev/downloads/Wedeo.zip)

## Run the studio locally

With Rust installed through rustup, run these commands from the repository root. The build uses the toolchain pinned in `rust-toolchain.toml`. Python 3 serves the demo locally.

```sh
rustup target add wasm32-unknown-unknown
./web/build.sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open [the local studio](http://127.0.0.1:8000/). It starts with a two-second H.264 clip. Click **Step +1** to see the next decoded frame, play its audio, or inspect the MP4 box tree. You can also drop in a file of your own. The browser processes it locally.

## Try the command line

With [uv](https://docs.astral.sh/uv/), run the native CLI from the Wedeo package index:

```sh
uvx --index https://wedeo.pages.dev/simple wedeo studio
uvx --index https://wedeo.pages.dev/simple wedeo ffprobe -v quiet -print_format json -show_streams clip.mp4
```

The prebuilt wheel targets x86-64 Linux with glibc 2.39 or later. On other platforms, uv can build the source distribution using Rust and NASM. Use `wedeo ffmpeg` and `wedeo ffprobe` for the supported FFmpeg-style commands.

The repository includes the same media samples used in the studio. The native build needs Rust and NASM for the AV1 dependencies.

```sh
cargo build --locked -p wedeo-cli
target/debug/wedeo inspect web/samples/h264_aac_320x240.mp4
target/debug/wedeo transcode -i web/samples/chord_48k.wav --sample-rate 16000 --channels 1 -o chord-16k.wav
```

The first command after the build prints the MP4 tracks and box tree. The second writes a 16 kHz mono WAV. Run `target/debug/wedeo --help` for cutting, muxing, and other commands.

## Inside the project

- [Codecs](codecs) and [containers](formats) implement H.264, VP9, PCM, MP4, Matroska, IVF, and WAV. [Adapters](adapters) provide AV1 encoding and decoding and additional audio decoders. The optional rav1d decoder uses a dav1d-compatible FFI boundary and assembly optimizations.
- [The Rust library](crates/wedeo/README.md) exposes typed audio and video readers. Its feature flags support an audio-only build.
- [The wgpu backend](crates/wedeo-gpu-wgpu) can run scaling and color conversion on a GPU. Build the CLI with `--features gpu-wgpu` and pass `--gpu` to a transcode that requests scaling. The CLI reports the selected adapter or why it fell back to the CPU.
- [The test harness](tests/wedeo-oracle) compares selected decoder output with a pinned FFmpeg revision. [Known differences](DIVERGENCES.md) describe where behavior varies.

[Architecture](docs/ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md) · [Historical development notes](docs/history) · [LGPL-2.1-or-later license](LICENSE)
