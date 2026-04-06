// wedeo-cli — FFmpeg-style command-line interface.
//
// Supports:
//   wedeo-cli -i input.mp4                       (probe, like ffprobe)
//   wedeo-cli -i input.mp4 output.wav            (transcode)
//   wedeo-cli -i input.mp4 -c:v av1 output.mp4   (transcode with codec selection)
//   wedeo-cli -codecs / -formats                  (list registered codecs/formats)
//
// Reference: FFmpeg ffmpeg.c option parsing, ffprobe.c

use std::env;
use std::io;
use std::path::Path;
use std::process;

// Ensure inventory registrations are linked in.
use wedeo_codec_h264 as _;
use wedeo_codec_pcm as _;
use wedeo_codec_vp9 as _;
use wedeo_format_h264 as _;
use wedeo_format_ivf as _;
use wedeo_format_mp4 as _;
use wedeo_format_wav as _;
use wedeo_rav1d as _;
use wedeo_rav1e as _;
use wedeo_symphonia as _;

use wedeo_codec::decoder::DecoderBuilder;
use wedeo_codec::encoder::{CodecFlags, EncoderBuilder};
use wedeo_codec::registry as codec_registry;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::Error;
use wedeo_core::media_type::MediaType;
use wedeo_format::context::{InputContext, OutputContext};
use wedeo_format::demuxer::Stream;
use wedeo_format::registry as format_registry;

// ---------------------------------------------------------------------------
// Parsed CLI options
// ---------------------------------------------------------------------------

struct CliOptions {
    inputs: Vec<String>,
    output: Option<String>,
    video_codec: Option<String>,
    audio_codec: Option<String>,
    format: Option<String>,
    bitrate_video: Option<i64>,
    bitrate_audio: Option<i64>,
    overwrite: bool,
    frames_video: Option<u64>,
    codec_options: Vec<(String, String)>,
    loglevel: String,
    // Special modes
    list_codecs: bool,
    list_formats: bool,
    list_decoders: bool,
    list_encoders: bool,
    copy_video: bool,
    copy_audio: bool,
    video_disabled: bool,
    audio_disabled: bool,
}

impl CliOptions {
    fn new() -> Self {
        Self {
            inputs: Vec::new(),
            output: None,
            video_codec: None,
            audio_codec: None,
            format: None,
            bitrate_video: None,
            bitrate_audio: None,
            overwrite: false,
            frames_video: None,
            codec_options: Vec::new(),
            loglevel: "warn".to_string(),
            list_codecs: false,
            list_formats: false,
            list_decoders: false,
            list_encoders: false,
            copy_video: false,
            copy_audio: false,
            video_disabled: false,
            audio_disabled: false,
        }
    }
}

fn parse_args() -> CliOptions {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut opts = CliOptions::new();

    if args.is_empty() {
        print_usage();
        process::exit(0);
    }

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-i" => {
                i += 1;
                if i >= args.len() {
                    die("Option -i requires an argument");
                }
                opts.inputs.push(args[i].clone());
            }

            "-c:v" | "-vcodec" => {
                i += 1;
                if i >= args.len() {
                    die("Option -c:v requires an argument");
                }
                if args[i] == "copy" {
                    opts.copy_video = true;
                } else {
                    opts.video_codec = Some(args[i].clone());
                }
            }

            "-c:a" | "-acodec" => {
                i += 1;
                if i >= args.len() {
                    die("Option -c:a requires an argument");
                }
                if args[i] == "copy" {
                    opts.copy_audio = true;
                } else {
                    opts.audio_codec = Some(args[i].clone());
                }
            }

            "-c" => {
                i += 1;
                if i >= args.len() {
                    die("Option -c requires an argument");
                }
                if args[i] == "copy" {
                    opts.copy_video = true;
                    opts.copy_audio = true;
                } else {
                    opts.video_codec = Some(args[i].clone());
                    opts.audio_codec = Some(args[i].clone());
                }
            }

            "-f" => {
                i += 1;
                if i >= args.len() {
                    die("Option -f requires an argument");
                }
                opts.format = Some(args[i].clone());
            }

            "-b:v" => {
                i += 1;
                if i >= args.len() {
                    die("Option -b:v requires an argument");
                }
                opts.bitrate_video = Some(parse_bitrate(&args[i]));
            }

            "-b:a" => {
                i += 1;
                if i >= args.len() {
                    die("Option -b:a requires an argument");
                }
                opts.bitrate_audio = Some(parse_bitrate(&args[i]));
            }

            "-frames:v" | "-vframes" => {
                i += 1;
                if i >= args.len() {
                    die("Option -frames:v requires an argument");
                }
                opts.frames_video = args[i].parse().ok();
            }

            "-y" => opts.overwrite = true,
            "-vn" => opts.video_disabled = true,
            "-an" => opts.audio_disabled = true,

            "-v" | "-loglevel" => {
                i += 1;
                if i >= args.len() {
                    die("Option -loglevel requires an argument");
                }
                opts.loglevel = args[i].clone();
            }

            "-codecs" => opts.list_codecs = true,
            "-formats" => opts.list_formats = true,
            "-decoders" => opts.list_decoders = true,
            "-encoders" => opts.list_encoders = true,

            "-h" | "-help" | "--help" => {
                print_usage();
                process::exit(0);
            }

            // Codec-private options: -key value (anything starting with -)
            s if s.starts_with('-') && s.len() > 1 => {
                let key = s.trim_start_matches('-').to_string();
                i += 1;
                if i >= args.len() {
                    die(&format!("Option {s} requires an argument"));
                }
                opts.codec_options.push((key, args[i].clone()));
            }

            // Positional: output file (first non-flag after inputs)
            _ => {
                if opts.output.is_some() {
                    die(&format!("Unexpected argument: {}", args[i]));
                }
                opts.output = Some(args[i].clone());
            }
        }
        i += 1;
    }

    opts
}

fn print_usage() {
    eprintln!("wedeo-cli — FFmpeg-compatible multimedia tool");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  wedeo-cli [options] [output_file]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -i <input>       Input file (required for most operations)");
    eprintln!("  -c:v <codec>     Video codec (e.g. av1, copy)");
    eprintln!("  -c:a <codec>     Audio codec (e.g. pcm_s16le, copy)");
    eprintln!("  -c <codec>       Set both video and audio codec (or 'copy')");
    eprintln!("  -f <format>      Force output format (e.g. mp4, wav)");
    eprintln!("  -b:v <bitrate>   Video bitrate (e.g. 2M, 500k, 1000000)");
    eprintln!("  -b:a <bitrate>   Audio bitrate");
    eprintln!("  -frames:v <n>    Stop after encoding n video frames");
    eprintln!("  -vn              Disable video");
    eprintln!("  -an              Disable audio");
    eprintln!("  -y               Overwrite output without asking");
    eprintln!("  -v <level>       Log level (error, warn, info, debug, trace)");
    eprintln!("  -codecs          List all registered codecs");
    eprintln!("  -formats         List all registered formats");
    eprintln!("  -decoders        List registered decoders");
    eprintln!("  -encoders        List registered encoders");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  wedeo-cli -i input.mp4                          # probe file info");
    eprintln!("  wedeo-cli -i input.mp4 -c:v av1 output.mp4     # transcode to AV1");
    eprintln!("  wedeo-cli -i input.mp4 -c copy output.mp4      # remux (copy streams)");
    eprintln!("  wedeo-cli -i input.mp4 -an -c:v av1 output.mp4 # video only, encode AV1");
    eprintln!("  wedeo-cli -codecs                               # list codecs");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let opts = parse_args();

    init_tracing(&opts.loglevel);

    // List modes
    if opts.list_codecs {
        cmd_codecs();
        return;
    }
    if opts.list_formats {
        cmd_formats();
        return;
    }
    if opts.list_decoders {
        cmd_decoders();
        return;
    }
    if opts.list_encoders {
        cmd_encoders();
        return;
    }

    if opts.inputs.is_empty() {
        die("At least one input file is required. Use -i <file>.");
    }

    // No output file = probe mode (like ffprobe / ffmpeg -i with no output)
    if opts.output.is_none() {
        for input in &opts.inputs {
            if let Err(e) = cmd_probe(input) {
                eprintln!("Error probing {input}: {e}");
                process::exit(1);
            }
        }
        return;
    }

    // Transcode/remux mode
    if let Err(e) = cmd_transcode(&opts) {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

fn init_tracing(level: &str) {
    let filter = match level {
        "quiet" | "-8" => "off",
        "error" => "error",
        "warning" | "warn" => "warn",
        "info" => "info",
        "verbose" | "debug" => "debug",
        "trace" => "trace",
        other => other,
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_target(true)
        .with_writer(io::stderr)
        .init();
}

// ---------------------------------------------------------------------------
// Probe (ffprobe equivalent)
// ---------------------------------------------------------------------------

fn cmd_probe(path: &str) -> wedeo_core::Result<()> {
    let ctx = InputContext::open(path)?;

    eprintln!("Input: {path}");
    if ctx.duration > 0 {
        let secs = ctx.duration as f64 / 1_000_000.0;
        let h = (secs / 3600.0) as u32;
        let m = ((secs % 3600.0) / 60.0) as u32;
        let s = secs % 60.0;
        eprintln!("  Duration: {h:02}:{m:02}:{s:06.3}");
    }

    for (key, value) in ctx.metadata.iter() {
        eprintln!("  {key}: {value}");
    }

    for stream in &ctx.streams {
        let cp = &stream.codec_params;
        match cp.media_type {
            MediaType::Video => {
                eprintln!(
                    "  Stream #{}: Video: {} {}x{} {}",
                    stream.index,
                    cp.codec_id.name(),
                    cp.width,
                    cp.height,
                    cp.pixel_format.name(),
                );
            }
            MediaType::Audio => {
                eprintln!(
                    "  Stream #{}: Audio: {} {} Hz {} ch {}",
                    stream.index,
                    cp.codec_id.name(),
                    cp.sample_rate,
                    cp.channel_layout.nb_channels,
                    cp.sample_format,
                );
            }
            other => {
                eprintln!(
                    "  Stream #{}: {}: {}",
                    stream.index,
                    other,
                    cp.codec_id.name(),
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Transcode / remux
// ---------------------------------------------------------------------------

fn cmd_transcode(opts: &CliOptions) -> wedeo_core::Result<()> {
    let output_path = opts.output.as_deref().unwrap();

    // Check overwrite
    if !opts.overwrite && Path::new(output_path).exists() {
        die(&format!(
            "File '{output_path}' already exists. Use -y to overwrite."
        ));
    }

    // Open input
    let mut input = InputContext::open(&opts.inputs[0])?;

    // Determine output format
    let format_name = if let Some(ref f) = opts.format {
        f.clone()
    } else {
        guess_format_from_path(output_path).ok_or_else(|| {
            Error::Other(format!(
                "Cannot determine format for '{output_path}'. Use -f."
            ))
        })?
    };

    // Classify input streams and decide what to do with each
    let mut video_stream_idx: Option<usize> = None;
    let mut audio_stream_idx: Option<usize> = None;

    for stream in &input.streams {
        match stream.codec_params.media_type {
            MediaType::Video if !opts.video_disabled && video_stream_idx.is_none() => {
                video_stream_idx = Some(stream.index);
            }
            MediaType::Audio if !opts.audio_disabled && audio_stream_idx.is_none() => {
                audio_stream_idx = Some(stream.index);
            }
            _ => {}
        }
    }

    if video_stream_idx.is_none() && audio_stream_idx.is_none() {
        return Err(Error::Other("No streams selected for output".into()));
    }

    // Build decoders and encoders for selected streams
    let mut video_decoder = None;
    let mut video_encoder = None;
    let mut audio_decoder = None;
    let mut audio_encoder = None;

    // Output stream list for the muxer
    let mut output_streams: Vec<Stream> = Vec::new();

    // --- Video ---
    if let Some(idx) = video_stream_idx {
        let in_stream = &input.streams[idx];
        let in_params = &in_stream.codec_params;

        if opts.copy_video {
            // Stream copy: pass packets through
            let mut out_stream = Stream::new(output_streams.len(), in_params.clone());
            out_stream.time_base = in_stream.time_base;
            output_streams.push(out_stream);
        } else {
            // Decode + encode
            let dec = DecoderBuilder::new(in_params.clone()).open()?;
            video_decoder = Some(dec);

            let out_codec_id = if let Some(ref name) = opts.video_codec {
                CodecId::from_name(name)
                    .ok_or_else(|| Error::Other(format!("Unknown video codec: '{name}'")))?
            } else {
                // Use muxer's default video codec
                let muxer_factory = format_registry::find_muxer_by_name(&format_name)
                    .ok_or(Error::MuxerNotFound)?;
                let default_codec = muxer_factory.descriptor().video_codec;
                if default_codec == CodecId::None {
                    return Err(Error::Other(format!(
                        "Format '{format_name}' has no default video codec. Use -c:v."
                    )));
                }
                default_codec
            };

            let mut enc_builder = EncoderBuilder::new(out_codec_id, MediaType::Video);
            enc_builder.width = in_params.width;
            enc_builder.height = in_params.height;
            enc_builder.pixel_format = in_params.pixel_format;
            enc_builder.time_base = in_stream.time_base;

            if let Some(br) = opts.bitrate_video {
                enc_builder.bit_rate = br;
            }

            // Check if muxer needs GLOBAL_HEADER
            if let Some(muxer_factory) = format_registry::find_muxer_by_name(&format_name) {
                let mflags = muxer_factory.descriptor().flags;
                if mflags.contains(wedeo_format::muxer::OutputFormatFlags::GLOBALHEADER) {
                    enc_builder.flags |= CodecFlags::GLOBAL_HEADER;
                }
            }

            // Apply codec-private options
            for (key, value) in &opts.codec_options {
                enc_builder.options.set(key, value);
            }

            let enc = enc_builder.open()?;
            video_encoder = Some(enc);

            let mut out_params =
                wedeo_codec::decoder::CodecParameters::new(out_codec_id, MediaType::Video);
            out_params.width = in_params.width;
            out_params.height = in_params.height;
            out_params.pixel_format = in_params.pixel_format;
            let mut out_stream = Stream::new(output_streams.len(), out_params);
            out_stream.time_base = in_stream.time_base;
            output_streams.push(out_stream);
        }
    }

    // --- Audio ---
    if let Some(idx) = audio_stream_idx {
        let in_stream = &input.streams[idx];
        let in_params = &in_stream.codec_params;

        if opts.copy_audio {
            let mut out_stream = Stream::new(output_streams.len(), in_params.clone());
            out_stream.time_base = in_stream.time_base;
            output_streams.push(out_stream);
        } else {
            let dec = DecoderBuilder::new(in_params.clone()).open()?;
            audio_decoder = Some(dec);

            let out_codec_id = if let Some(ref name) = opts.audio_codec {
                CodecId::from_name(name)
                    .ok_or_else(|| Error::Other(format!("Unknown audio codec: '{name}'")))?
            } else {
                let muxer_factory = format_registry::find_muxer_by_name(&format_name)
                    .ok_or(Error::MuxerNotFound)?;
                let default_codec = muxer_factory.descriptor().audio_codec;
                if default_codec == CodecId::None {
                    return Err(Error::Other(format!(
                        "Format '{format_name}' has no default audio codec. Use -c:a."
                    )));
                }
                default_codec
            };

            let mut enc_builder = EncoderBuilder::new(out_codec_id, MediaType::Audio);
            enc_builder.sample_rate = in_params.sample_rate;
            enc_builder.sample_format = in_params.sample_format;
            enc_builder.channel_layout = in_params.channel_layout.clone();
            enc_builder.time_base = in_stream.time_base;

            if let Some(br) = opts.bitrate_audio {
                enc_builder.bit_rate = br;
            }

            let enc = enc_builder.open()?;
            audio_encoder = Some(enc);

            let mut out_params =
                wedeo_codec::decoder::CodecParameters::new(out_codec_id, MediaType::Audio);
            out_params.sample_rate = in_params.sample_rate;
            out_params.sample_format = in_params.sample_format;
            out_params.channel_layout = in_params.channel_layout.clone();
            let mut out_stream = Stream::new(output_streams.len(), out_params);
            out_stream.time_base = in_stream.time_base;
            output_streams.push(out_stream);
        }
    }

    // Open output
    let mut output = OutputContext::create(output_path, &format_name, &output_streams)?;

    // --- Main processing loop ---
    let mut video_frames_out = 0u64;
    let frame_limit = opts.frames_video.unwrap_or(u64::MAX);

    // Demux → decode → encode → mux
    loop {
        let packet = match input.read_packet() {
            Ok(pkt) => pkt,
            Err(Error::Eof) => break,
            Err(e) => return Err(e),
        };

        let stream_idx = packet.stream_index;

        // Video stream
        if Some(stream_idx) == video_stream_idx {
            if video_frames_out >= frame_limit {
                continue;
            }

            if opts.copy_video {
                output.write_packet(&packet)?;
                video_frames_out += 1;
            } else if let (Some(dec), Some(enc)) = (&mut video_decoder, &mut video_encoder) {
                dec.send_packet(Some(&packet))?;
                loop {
                    match dec.receive_frame() {
                        Ok(frame) => {
                            if video_frames_out >= frame_limit {
                                break;
                            }
                            enc.send_frame(Some(&frame))?;
                            drain_encoder_to_output(&mut **enc, &mut output)?;
                            video_frames_out += 1;
                        }
                        Err(Error::Again) => break,
                        Err(e) => return Err(e),
                    }
                }
            }
            continue;
        }

        // Audio stream
        if Some(stream_idx) == audio_stream_idx {
            if opts.copy_audio {
                output.write_packet(&packet)?;
            } else if let (Some(dec), Some(enc)) = (&mut audio_decoder, &mut audio_encoder) {
                dec.send_packet(Some(&packet))?;
                loop {
                    match dec.receive_frame() {
                        Ok(frame) => {
                            enc.send_frame(Some(&frame))?;
                            drain_encoder_to_output(&mut **enc, &mut output)?;
                        }
                        Err(Error::Again) => break,
                        Err(e) => return Err(e),
                    }
                }
            }
        }
    }

    // Drain decoders and encoders
    if let (Some(dec), Some(enc)) = (&mut video_decoder, &mut video_encoder) {
        dec.send_packet(None)?;
        loop {
            match dec.receive_frame() {
                Ok(frame) => {
                    if video_frames_out < frame_limit {
                        enc.send_frame(Some(&frame))?;
                        drain_encoder_to_output(&mut **enc, &mut output)?;
                        video_frames_out += 1;
                    }
                }
                Err(Error::Eof | Error::Again) => break,
                Err(e) => return Err(e),
            }
        }
        enc.send_frame(None)?;
        drain_encoder_to_output(&mut **enc, &mut output)?;
    }

    if let (Some(dec), Some(enc)) = (&mut audio_decoder, &mut audio_encoder) {
        dec.send_packet(None)?;
        loop {
            match dec.receive_frame() {
                Ok(frame) => {
                    enc.send_frame(Some(&frame))?;
                    drain_encoder_to_output(&mut **enc, &mut output)?;
                }
                Err(Error::Eof | Error::Again) => break,
                Err(e) => return Err(e),
            }
        }
        enc.send_frame(None)?;
        drain_encoder_to_output(&mut **enc, &mut output)?;
    }

    output.finish()?;

    eprintln!("Output: {output_path}");
    if video_stream_idx.is_some() {
        eprintln!("  Video: {video_frames_out} frames");
    }

    Ok(())
}

fn drain_encoder_to_output(
    enc: &mut dyn wedeo_codec::encoder::Encoder,
    output: &mut OutputContext,
) -> wedeo_core::Result<()> {
    loop {
        match enc.receive_packet() {
            Ok(pkt) => output.write_packet(&pkt)?,
            Err(Error::Again | Error::Eof) => break,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// List commands
// ---------------------------------------------------------------------------

fn cmd_codecs() {
    println!("Codecs:");
    println!(" D..... = Decoding supported");
    println!(" .E.... = Encoding supported");
    println!(" ..V... = Video codec");
    println!(" ..A... = Audio codec");
    println!(" ..S... = Subtitle codec");
    println!(" ------");

    // Collect all known codec IDs from decoders and encoders
    let mut seen = std::collections::BTreeMap::<&str, (bool, bool, &str)>::new();

    for factory in codec_registry::decoders() {
        let desc = factory.descriptor();
        let entry = seen.entry(desc.name).or_insert((false, false, ""));
        entry.0 = true;
        if entry.2.is_empty() {
            entry.2 = desc.long_name;
        }
    }
    for factory in codec_registry::encoders() {
        let desc = factory.descriptor();
        let entry = seen.entry(desc.name).or_insert((false, false, ""));
        entry.1 = true;
        if entry.2.is_empty() {
            entry.2 = desc.long_name;
        }
    }

    for (name, (has_dec, has_enc, long_name)) in &seen {
        let d = if *has_dec { 'D' } else { '.' };
        let e = if *has_enc { 'E' } else { '.' };
        let t = match CodecId::from_name(name).map(media_type_for_codec) {
            Some(MediaType::Video) => 'V',
            Some(MediaType::Audio) => 'A',
            Some(MediaType::Subtitle) => 'S',
            _ => '.',
        };
        println!(" {d}{e}{t}... {name:<20} {long_name}");
    }
}

fn cmd_decoders() {
    println!("Decoders:");
    for factory in codec_registry::decoders() {
        let desc = factory.descriptor();
        println!(
            "  {:<20} {} ({})",
            desc.name, desc.long_name, desc.media_type
        );
    }
}

fn cmd_encoders() {
    println!("Encoders:");
    for factory in codec_registry::encoders() {
        let desc = factory.descriptor();
        println!(
            "  {:<20} {} ({})",
            desc.name, desc.long_name, desc.media_type
        );
    }
}

fn cmd_formats() {
    println!("Formats:");
    println!(" D. = Demuxing supported");
    println!(" .M = Muxing supported");
    println!(" --");

    let mut seen = std::collections::BTreeMap::<&str, (bool, bool, &str)>::new();

    for factory in format_registry::demuxers() {
        let desc = factory.descriptor();
        let entry = seen.entry(desc.name).or_insert((false, false, ""));
        entry.0 = true;
        if entry.2.is_empty() {
            entry.2 = desc.long_name;
        }
    }
    for factory in format_registry::muxers() {
        let desc = factory.descriptor();
        let entry = seen.entry(desc.name).or_insert((false, false, ""));
        entry.1 = true;
        if entry.2.is_empty() {
            entry.2 = desc.long_name;
        }
    }

    for (name, (has_demux, has_mux, long_name)) in &seen {
        let d = if *has_demux { 'D' } else { '.' };
        let m = if *has_mux { 'M' } else { '.' };
        println!(" {d}{m} {name:<20} {long_name}");
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn guess_format_from_path(path: &str) -> Option<String> {
    let ext = Path::new(path).extension()?.to_str()?;
    let factory = format_registry::guess_muxer_by_extension(ext)?;
    Some(factory.descriptor().name.to_string())
}

fn parse_bitrate(s: &str) -> i64 {
    let s = s.trim();
    if let Some(prefix) = s.strip_suffix('k').or_else(|| s.strip_suffix('K')) {
        prefix.parse::<i64>().unwrap_or(0) * 1000
    } else if let Some(prefix) = s.strip_suffix('M') {
        prefix.parse::<i64>().unwrap_or(0) * 1_000_000
    } else {
        s.parse::<i64>().unwrap_or(0)
    }
}

fn media_type_for_codec(id: CodecId) -> MediaType {
    match id {
        CodecId::None => MediaType::Data,
        CodecId::Mpeg1video
        | CodecId::Mpeg2video
        | CodecId::H261
        | CodecId::H263
        | CodecId::Rv10
        | CodecId::Rv20
        | CodecId::Mjpeg
        | CodecId::Mpeg4
        | CodecId::Rawvideo
        | CodecId::H264
        | CodecId::Vp8
        | CodecId::Vp9
        | CodecId::Hevc
        | CodecId::Av1 => MediaType::Video,
        CodecId::SubDvdSubtitle
        | CodecId::SubDvbSubtitle
        | CodecId::SubText
        | CodecId::SubXsub
        | CodecId::SubSsa
        | CodecId::SubMovText
        | CodecId::SubSrt
        | CodecId::SubWebvtt => MediaType::Subtitle,
        _ => MediaType::Audio,
    }
}

fn die(msg: &str) -> ! {
    eprintln!("Error: {msg}");
    process::exit(1);
}
