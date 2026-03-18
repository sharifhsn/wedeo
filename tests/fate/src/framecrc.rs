//! framecrc output generator for wedeo.
//!
//! Produces output in FFmpeg's framecrc format. Operates in two modes:
//!
//! 1. **Packet passthrough** (audio/PCM): checksums raw packet data directly,
//!    matching FFmpeg's `-c copy` framecrc output.
//! 2. **Decode mode** (video): decodes packets through the codec, then checksums
//!    the decoded YUV frame data, matching FFmpeg's framecrc decode output.

use std::env;
use std::io::{self, Write};

// Ensure codec/format registrations are linked.
use wedeo_codec_h264 as _;
use wedeo_codec_pcm as _;
use wedeo_format_h264 as _;
use wedeo_format_wav as _;
use wedeo_symphonia as _;

use wedeo_codec::decoder::{Decoder, DecoderBuilder};
use wedeo_core::error::Error;
use wedeo_core::media_type::MediaType;
use wedeo_core::packet::PacketFlags;
use wedeo_format::context::InputContext;

#[cfg(feature = "tracing")]
fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();
}

/// Adler-32 checksum matching FFmpeg's av_adler32_update(0, ...).
/// FFmpeg passes initial adler=0, giving s1=0, s2=0 (NOT the standard init of s1=1).
fn adler32(data: &[u8]) -> u32 {
    let mut hasher = adler2::Adler32::from_checksum(0);
    hasher.write_slice(data);
    hasher.checksum()
}

/// Check if a stream requires decoding (as opposed to packet passthrough).
/// Video streams always require decoding for framecrc conformance output.
fn needs_decode(media_type: MediaType) -> bool {
    media_type == MediaType::Video
}

fn main() {
    #[cfg(feature = "tracing")]
    init_tracing();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: wedeo-framecrc <input-file>");
        std::process::exit(1);
    }

    // Optional: --raw-yuv <output.yuv> to dump raw decoded frames
    let raw_yuv_path = if args.len() >= 4 && args[2] == "--raw-yuv" {
        Some(args[3].clone())
    } else {
        None
    };
    let mut raw_yuv_file: Option<std::fs::File> = raw_yuv_path.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating {p}: {e}");
            std::process::exit(1);
        })
    });

    let path = &args[1];
    let mut ctx = match InputContext::open(path) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Error opening {path}: {e}");
            std::process::exit(1);
        }
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();

    // Determine which streams need decoding and create decoders for them.
    let mut decoders: Vec<Option<Box<dyn Decoder>>> = Vec::new();
    let mut decode_stream_indices: Vec<bool> = Vec::new();

    for stream in &ctx.streams {
        let cp = &stream.codec_params;
        let i = stream.index;
        let tb = &stream.time_base;

        if needs_decode(cp.media_type) {
            // Video stream: print decode-mode headers
            writeln!(out, "#tb {i}: {}/{}", tb.num, tb.den).unwrap();
            writeln!(out, "#media_type {i}: video").unwrap();
            writeln!(out, "#codec_id {i}: rawvideo").unwrap();
            writeln!(out, "#dimensions {i}: {}x{}", cp.width, cp.height).unwrap();
            // SAR: use 0/1 if not set (matching FFmpeg)
            writeln!(out, "#sar {i}: 0/1").unwrap();

            // Create a decoder for this stream
            match DecoderBuilder::new(cp.clone()).open() {
                Ok(dec) => {
                    decoders.push(Some(dec));
                    decode_stream_indices.push(true);
                }
                Err(e) => {
                    eprintln!("Error creating decoder for stream {i}: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            // Audio / other stream: print passthrough headers
            writeln!(out, "#tb {i}: {}/{}", tb.num, tb.den).unwrap();
            writeln!(out, "#media_type {i}: {}", cp.media_type).unwrap();
            writeln!(out, "#codec_id {i}: {}", cp.codec_id.name()).unwrap();
            if cp.media_type == MediaType::Audio {
                writeln!(out, "#sample_rate {i}: {}", cp.sample_rate).unwrap();
                writeln!(out, "#channel_layout_name {i}: {}", cp.channel_layout).unwrap();
            }
            decoders.push(None);
            decode_stream_indices.push(false);
        }
    }

    // Demux + decode/passthrough loop
    loop {
        match ctx.read_packet() {
            Ok(packet) => {
                let si = packet.stream_index;

                if si < decode_stream_indices.len() && decode_stream_indices[si] {
                    // Decode mode: feed packet to decoder, output decoded frames
                    let decoder = decoders[si].as_mut().unwrap();
                    if let Err(e) = decoder.send_packet(Some(&packet)) {
                        eprintln!("Error decoding packet for stream {si}: {e}");
                        std::process::exit(1);
                    }
                    drain_frames(decoder.as_mut(), si, &mut out, &mut raw_yuv_file);
                } else {
                    // Passthrough mode: checksum raw packet data
                    let data = packet.data.data();
                    let crc = adler32(data);
                    let size = data.len();

                    let mut line = format!(
                        "{si}, {:>10}, {:>10}, {:>8}, {:>8}, 0x{crc:08x}",
                        packet.dts, packet.pts, packet.duration, size,
                    );

                    if !packet.flags.contains(PacketFlags::KEY) {
                        line.push_str(&format!(", F=0x{:04X}", packet.flags.bits()));
                    }

                    writeln!(out, "{line}").unwrap();
                }
            }
            Err(Error::Eof) => {
                // Drain all decoders
                for (si, dec_opt) in decoders.iter_mut().enumerate() {
                    if let Some(decoder) = dec_opt {
                        let _ = decoder.send_packet(None);
                        drain_frames(decoder.as_mut(), si, &mut out, &mut raw_yuv_file);
                    }
                }
                break;
            }
            Err(e) => {
                eprintln!("Error reading packet: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// Drain all available frames from a decoder and write framecrc lines.
fn drain_frames(
    decoder: &mut dyn Decoder,
    si: usize,
    out: &mut impl Write,
    raw_yuv_file: &mut Option<std::fs::File>,
) {
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                if let Some(video) = frame.video() {
                    // Concatenate Y+U+V plane data for checksum.
                    // Each plane is stored row-by-row with linesize that may
                    // differ from the actual width, so we copy row-by-row.
                    let width = video.width as usize;
                    let height = video.height as usize;
                    let chroma_width = width / 2;
                    let chroma_height = height / 2;

                    let mut raw = Vec::with_capacity(width * height * 3 / 2);

                    // Y plane
                    let y_plane = &video.planes[0];
                    let y_data = y_plane.buffer.data();
                    for row in 0..height {
                        let start = y_plane.offset + row * y_plane.linesize;
                        raw.extend_from_slice(&y_data[start..start + width]);
                    }

                    // U plane
                    let u_plane = &video.planes[1];
                    let u_data = u_plane.buffer.data();
                    for row in 0..chroma_height {
                        let start = u_plane.offset + row * u_plane.linesize;
                        raw.extend_from_slice(&u_data[start..start + chroma_width]);
                    }

                    // V plane
                    let v_plane = &video.planes[2];
                    let v_data = v_plane.buffer.data();
                    for row in 0..chroma_height {
                        let start = v_plane.offset + row * v_plane.linesize;
                        raw.extend_from_slice(&v_data[start..start + chroma_width]);
                    }

                    // Optionally dump raw YUV
                    if let Some(f) = raw_yuv_file.as_mut() {
                        use std::io::Write;
                        f.write_all(&raw).unwrap();
                    }

                    let crc = adler32(&raw);
                    let size = raw.len();

                    writeln!(
                        out,
                        "{si}, {:>10}, {:>10}, {:>8}, {:>8}, 0x{crc:08x}",
                        frame.pts, frame.pts, 1, size,
                    )
                    .unwrap();
                }
            }
            Err(Error::Again) => break,
            Err(Error::Eof) => break,
            Err(e) => {
                eprintln!("Error receiving frame from stream {si}: {e}");
                std::process::exit(1);
            }
        }
    }
}
