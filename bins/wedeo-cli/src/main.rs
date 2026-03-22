use std::env;
use std::io::{self, Write};

// These imports ensure the inventory registrations are linked in.
use wedeo_codec_h264 as _;
use wedeo_codec_pcm as _;
use wedeo_format_h264 as _;
use wedeo_format_mp4 as _;
use wedeo_format_wav as _;
use wedeo_rav1d as _;
use wedeo_symphonia as _;

use wedeo_codec::decoder::DecoderBuilder;
use wedeo_codec::registry as codec_registry;
use wedeo_core::error::Error;
use wedeo_format::context::InputContext;
use wedeo_format::registry as format_registry;

fn main() {
    #[cfg(feature = "tracing")]
    init_tracing();

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: wedeo-cli <command> [args...]");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  info <file>     Show stream information (like ffprobe)");
        eprintln!("  decode <file>   Decode and write raw output to stdout");
        eprintln!("  codecs          List registered codecs");
        eprintln!("  formats         List registered formats");
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "info" => cmd_info(&args[2..]),
        "decode" => cmd_decode(&args[2..]),
        "codecs" => cmd_codecs(),
        "formats" => cmd_formats(),
        other => {
            eprintln!("Unknown command: {other}");
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

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

fn cmd_info(args: &[String]) -> wedeo_core::Result<()> {
    if args.is_empty() {
        return Err(Error::Other("Usage: wedeo-cli info <file>".into()));
    }

    let path = &args[0];
    let ctx = InputContext::open(path)?;

    println!("File: {path}");
    println!("Duration: {} us", ctx.duration);
    println!("Start time: {} us", ctx.start_time);
    println!();

    // Print metadata
    for (key, value) in ctx.metadata.iter() {
        println!("  {key}: {value}");
    }
    if !ctx.metadata.is_empty() {
        println!();
    }

    // Print streams
    for stream in &ctx.streams {
        let cp = &stream.codec_params;
        println!(
            "Stream #{}: {} ({})",
            stream.index,
            cp.media_type,
            cp.codec_id.name()
        );
        println!("  Time base: {}", stream.time_base);
        println!("  Duration: {} (in time_base units)", stream.duration);

        match cp.media_type {
            wedeo_core::MediaType::Audio => {
                println!("  Sample rate: {} Hz", cp.sample_rate);
                println!("  Sample format: {}", cp.sample_format);
                println!("  Channels: {}", cp.channel_layout);
                println!("  Bit rate: {} bps", cp.bit_rate);
                println!("  Block align: {}", cp.block_align);
                println!("  Bits per coded sample: {}", cp.bits_per_coded_sample);
            }
            wedeo_core::MediaType::Video => {
                println!("  Resolution: {}x{}", cp.width, cp.height);
                println!("  Pixel format: {}", cp.pixel_format.name());
            }
            _ => {}
        }
        println!();
    }

    Ok(())
}

fn cmd_decode(args: &[String]) -> wedeo_core::Result<()> {
    if args.is_empty() {
        return Err(Error::Other("Usage: wedeo-cli decode <file>".into()));
    }

    let path = &args[0];
    let mut ctx = InputContext::open(path)?;

    let stream = ctx.streams.first().ok_or(Error::StreamNotFound)?;
    let params = stream.codec_params.clone();
    let is_video = params.media_type == wedeo_core::MediaType::Video;
    let mut decoder = DecoderBuilder::new(params).open()?;

    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut frame_count = 0u64;

    let write_frame = |frame: &wedeo_core::frame::Frame,
                       out: &mut io::StdoutLock<'_>,
                       count: &mut u64|
     -> wedeo_core::Result<()> {
        if let Some(video) = frame.video() {
            let w = video.width as usize;
            let h = video.height as usize;
            // Write Y plane row by row (linesize may differ from width)
            let y = &video.planes[0];
            let yd = y.buffer.data();
            for row in 0..h {
                let s = y.offset + row * y.linesize;
                out.write_all(&yd[s..s + w]).map_err(Error::from)?;
            }
            // Write U plane
            let cw = w / 2;
            let ch = h / 2;
            let u = &video.planes[1];
            let ud = u.buffer.data();
            for row in 0..ch {
                let s = u.offset + row * u.linesize;
                out.write_all(&ud[s..s + cw]).map_err(Error::from)?;
            }
            // Write V plane
            let v = &video.planes[2];
            let vd = v.buffer.data();
            for row in 0..ch {
                let s = v.offset + row * v.linesize;
                out.write_all(&vd[s..s + cw]).map_err(Error::from)?;
            }
            *count += 1;
        } else if let Some(audio) = frame.audio() {
            for plane in &audio.planes {
                out.write_all(plane.buffer.data()).map_err(Error::from)?;
            }
            *count += 1;
        }
        Ok(())
    };

    // Print video metadata to stderr so the viewer script can use it
    if is_video {
        let s = &ctx.streams[0].codec_params;
        eprintln!("WEDEO_VIDEO_WIDTH={}", s.width);
        eprintln!("WEDEO_VIDEO_HEIGHT={}", s.height);
        eprintln!("WEDEO_VIDEO_PIX_FMT={}", s.pixel_format.name());
    }

    // Decode loop
    loop {
        match ctx.read_packet() {
            Ok(packet) => {
                decoder.send_packet(Some(&packet))?;
                loop {
                    match decoder.receive_frame() {
                        Ok(frame) => write_frame(&frame, &mut out, &mut frame_count)?,
                        Err(Error::Again) => break,
                        Err(e) => return Err(e),
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(e) => return Err(e),
        }
    }

    // Drain decoder
    decoder.send_packet(None)?;
    loop {
        match decoder.receive_frame() {
            Ok(frame) => write_frame(&frame, &mut out, &mut frame_count)?,
            Err(Error::Eof | Error::Again) => break,
            Err(e) => return Err(e),
        }
    }

    eprintln!("Decoded {frame_count} frames");
    Ok(())
}

fn cmd_codecs() -> wedeo_core::Result<()> {
    println!("Registered decoders:");
    for factory in codec_registry::decoders() {
        let desc = factory.descriptor();
        println!("  {} - {} ({})", desc.name, desc.long_name, desc.media_type);
    }

    println!();
    println!("Registered encoders:");
    for factory in codec_registry::encoders() {
        let desc = factory.descriptor();
        println!("  {} - {} ({})", desc.name, desc.long_name, desc.media_type);
    }

    Ok(())
}

fn cmd_formats() -> wedeo_core::Result<()> {
    println!("Registered demuxers:");
    for factory in format_registry::demuxers() {
        let desc = factory.descriptor();
        println!(
            "  {} - {} (ext: {})",
            desc.name, desc.long_name, desc.extensions
        );
    }

    println!();
    println!("Registered muxers:");
    for factory in format_registry::muxers() {
        let desc = factory.descriptor();
        println!(
            "  {} - {} (ext: {})",
            desc.name, desc.long_name, desc.extensions
        );
    }

    Ok(())
}
