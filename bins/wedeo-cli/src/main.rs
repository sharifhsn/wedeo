#![forbid(unsafe_code)]
//! Unified `wedeo` CLI (`inspect` / `explain` / `transcode` / `cut` / `mux` /
//! `ffmpeg` / `mcp` / `codecs` / `formats`).
//!
//! Licensed under `LGPL-2.1-or-later`.

mod ffmpeg_compat;
mod json;
mod mcp;
mod ops;

use std::env;
use std::process;

use wedeo::{Error, Registry, Result, default_registry};

use crate::ffmpeg_compat::{FfmpegResult, execute_ffmpeg};
use crate::ops::{
    CutMode, CutReq, ExplainReq, InspectLevel, InspectReq, MuxReq, TranscodeReq,
    format_codecs_table, format_formats_table, parse_pixel_layout, parse_size, parse_timestamp_sec,
    run_cut, run_explain, run_inspect, run_mux, run_transcode,
};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if let Err(err) = dispatch(&args) {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}

fn dispatch(args: &[String]) -> Result<()> {
    if args.is_empty() {
        print!("{}", main_usage());
        return Ok(());
    }

    let reg = default_registry();
    let cmd = args[0].as_str();
    let rest = &args[1..];

    match cmd {
        "-h" | "--help" | "help" => {
            print!("{}", main_usage());
            Ok(())
        }
        "-V" | "--version" | "version" => {
            println!("wedeo {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "inspect" => cmd_inspect(&reg, rest),
        "explain" => cmd_explain(&reg, rest),
        "transcode" => cmd_transcode(&reg, rest),
        "cut" => cmd_cut(&reg, rest),
        "mux" => cmd_mux(&reg, rest),
        "ffmpeg" => cmd_ffmpeg(&reg, rest),
        "mcp" => {
            mcp::serve_stdio(&reg).map_err(|e| Error::io("mcp", &e))?;
            Ok(())
        }
        "codecs" => {
            print!("{}", format_codecs_table(&reg));
            Ok(())
        }
        "formats" => {
            print!("{}", format_formats_table(&reg));
            Ok(())
        }
        other => Err(Error::invalid_argument(
            "wedeo",
            format!("unknown subcommand '{other}' (run `wedeo --help` for usage)"),
        )),
    }
}

fn main_usage() -> &'static str {
    "\
wedeo — Pure-Rust Universal A/V Adapter

Usage:
  wedeo <command> [options]

Commands:
  inspect    Inspect container AST, tracks, colorimetry, packets/GOP/NALs, and StreamHealth
  explain    Compile pipeline via wedeo-graph without running (nodes, bridges, GPU notes)
  transcode  Transcode or remux media preserving tracks, metadata, chapters, and rotation
  cut        Cut time ranges (--mode keyframe|smart|reencode via wedeo_graph::timeline)
  mux        Losslessly mux video/audio/subtitle/chapter/tag streams from multiple inputs
  ffmpeg     Order-sensitive FFmpeg CLI compatibility mode (`wedeo ffmpeg ...`)
  mcp        Start stdio JSON-RPC 2.0 MCP server exposing the 5 typed operations
  codecs     List registered decoders and encoders
  formats    List registered demuxers and muxers
"
}

fn next_val(args: &[String], idx: &mut usize, flag: &str) -> Result<String> {
    *idx += 1;
    args.get(*idx)
        .cloned()
        .ok_or_else(|| Error::invalid_argument("wedeo", format!("option {flag} requires a value")))
}

// ---------------------------------------------------------------------------
// Subcommand Parsers & Runners
// ---------------------------------------------------------------------------

fn cmd_inspect(reg: &Registry, args: &[String]) -> Result<()> {
    let mut input: Option<String> = None;
    let mut level = InspectLevel::Container;
    let mut verify = false;
    let mut max_packets: Option<usize> = None;
    let mut json_out = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: wedeo inspect [--level container|packets|gop|nal] [--verify] [--max-packets N] [--json] <input>"
                );
                return Ok(());
            }
            "-i" | "--input" => input = Some(next_val(args, &mut i, "--input")?),
            "--level" => {
                let v = next_val(args, &mut i, "--level")?;
                level = InspectLevel::parse(&v)?;
            }
            "--verify" => verify = true,
            "--max-packets" => {
                let v = next_val(args, &mut i, "--max-packets")?;
                max_packets = Some(v.parse().map_err(|_| {
                    Error::invalid_argument("inspect", format!("invalid --max-packets '{v}'"))
                })?);
            }
            "--json" => json_out = true,
            flag if flag.starts_with('-') => {
                return Err(Error::invalid_argument(
                    "inspect",
                    format!("unknown option '{flag}'"),
                ));
            }
            pos => {
                if input.is_some() {
                    return Err(Error::invalid_argument(
                        "inspect",
                        format!("unexpected positional argument '{pos}'"),
                    ));
                }
                input = Some(pos.to_string());
            }
        }
        i += 1;
    }

    let path =
        input.ok_or_else(|| Error::invalid_argument("inspect", "missing input file path"))?;
    let req = InspectReq {
        input: path,
        level,
        verify,
        max_packets,
    };
    let report = run_inspect(reg, &req)?;
    if json_out {
        println!("{}", report.to_json());
    } else {
        print!("{}", report.format_human());
    }
    Ok(())
}

fn parse_transcode_args(args: &[String], cmd_name: &'static str) -> Result<(TranscodeReq, bool)> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut format: Option<String> = None;
    let mut video_codec: Option<String> = None;
    let mut audio_codec: Option<String> = None;
    let mut video_bitrate: Option<i64> = None;
    let mut audio_bitrate: Option<i64> = None;
    let mut frames_video: Option<u64> = None;
    let mut no_video = false;
    let mut no_audio = false;
    let mut size = None;
    let mut pixel_layout = None;
    let mut sample_rate = None;
    let mut channels = None;
    let mut start_sec = None;
    let mut duration_sec = None;
    let mut strict = false;
    let mut preserve_all = true;
    let mut overwrite = false;
    let mut prefer_gpu = false;
    let mut json_out = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-i" | "--input" => input = Some(next_val(args, &mut i, "--input")?),
            "-o" | "--output" => output = Some(next_val(args, &mut i, "--output")?),
            "-f" | "--format" => format = Some(next_val(args, &mut i, "--format")?),
            "--video" | "-c:v" => video_codec = Some(next_val(args, &mut i, "--video")?),
            "--audio" | "-c:a" => audio_codec = Some(next_val(args, &mut i, "--audio")?),
            "--bitrate-video" | "-b:v" => {
                let v = next_val(args, &mut i, "--bitrate-video")?;
                video_bitrate = Some(parse_bitrate_value(&v)?);
            }
            "--bitrate-audio" | "-b:a" => {
                let v = next_val(args, &mut i, "--bitrate-audio")?;
                audio_bitrate = Some(parse_bitrate_value(&v)?);
            }
            "--frames" => {
                let v = next_val(args, &mut i, "--frames")?;
                frames_video = Some(v.parse().map_err(|_| {
                    Error::invalid_argument(cmd_name, format!("invalid --frames '{v}'"))
                })?);
            }
            "--no-video" | "-vn" => no_video = true,
            "--no-audio" | "-an" => no_audio = true,
            "--size" => {
                let v = next_val(args, &mut i, "--size")?;
                size = Some(parse_size(&v)?);
            }
            "--pix-fmt" => {
                let v = next_val(args, &mut i, "--pix-fmt")?;
                pixel_layout = Some(parse_pixel_layout(&v).ok_or_else(|| {
                    Error::invalid_argument(cmd_name, format!("unknown pixel layout '{v}'"))
                })?);
            }
            "--sample-rate" => {
                let v = next_val(args, &mut i, "--sample-rate")?;
                sample_rate = Some(v.parse().map_err(|_| {
                    Error::invalid_argument(cmd_name, format!("invalid --sample-rate '{v}'"))
                })?);
            }
            "--channels" => {
                let v = next_val(args, &mut i, "--channels")?;
                channels = Some(v.parse().map_err(|_| {
                    Error::invalid_argument(cmd_name, format!("invalid --channels '{v}'"))
                })?);
            }
            "--from" | "-ss" => {
                let v = next_val(args, &mut i, "--from")?;
                start_sec = Some(parse_timestamp_sec(&v)?);
            }
            "--duration" | "-t" => {
                let v = next_val(args, &mut i, "--duration")?;
                duration_sec = Some(parse_timestamp_sec(&v)?);
            }
            "--strict" => strict = true,
            "--preserve-all" => preserve_all = true,
            "--no-preserve-all" => preserve_all = false,
            "-y" | "--overwrite" => overwrite = true,
            "--gpu" => prefer_gpu = true,
            "--json" => json_out = true,
            flag if flag.starts_with('-') => {
                return Err(Error::invalid_argument(
                    cmd_name,
                    format!("unknown option '{flag}'"),
                ));
            }
            pos => {
                if input.is_none() {
                    input = Some(pos.to_string());
                } else if output.is_none() {
                    output = Some(pos.to_string());
                } else {
                    return Err(Error::invalid_argument(
                        cmd_name,
                        format!("unexpected positional argument '{pos}'"),
                    ));
                }
            }
        }
        i += 1;
    }

    let in_path =
        input.ok_or_else(|| Error::invalid_argument(cmd_name, "missing input (-i <file>)"))?;
    let out_path =
        output.ok_or_else(|| Error::invalid_argument(cmd_name, "missing output (-o <file>)"))?;

    let mut req = TranscodeReq::new(in_path, out_path);
    req.format = format;
    req.video_codec = video_codec;
    req.audio_codec = audio_codec;
    req.video_bitrate = video_bitrate;
    req.audio_bitrate = audio_bitrate;
    req.frames_video = frames_video;
    req.no_video = no_video;
    req.no_audio = no_audio;
    req.size = size;
    req.pixel_layout = pixel_layout;
    req.sample_rate = sample_rate;
    req.channels = channels;
    req.start_sec = start_sec;
    req.duration_sec = duration_sec;
    req.strict = strict;
    req.preserve_all = preserve_all;
    req.overwrite = overwrite;
    req.prefer_gpu = prefer_gpu;
    Ok((req, json_out))
}

fn cmd_explain(reg: &Registry, args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "Usage: wedeo explain -i <input> -o <output> [--video codec] [--audio codec] [--size WxH] [--strict] [--gpu] [--json]"
        );
        return Ok(());
    }
    let (tr, json_out) = parse_transcode_args(args, "explain")?;
    let plan = run_explain(reg, &ExplainReq::new(tr))?;
    if json_out {
        println!("{}", plan.to_json());
    } else {
        print!("{}", plan.explain());
    }
    Ok(())
}

fn cmd_transcode(reg: &Registry, args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "Usage: wedeo transcode -i <input> -o <output> [--video codec] [--audio codec] [--frames N] [--size WxH] [--sample-rate Hz] [--channels N] [--strict] [--no-video] [--no-audio] [-y] [--json]"
        );
        return Ok(());
    }
    let (req, json_out) = parse_transcode_args(args, "transcode")?;
    let report = run_transcode(reg, &req)?;
    if json_out {
        println!("{}", report.to_json());
    } else {
        print!("{}", report.format_human());
    }
    Ok(())
}

fn cmd_cut(reg: &Registry, args: &[String]) -> Result<()> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut from_sec: Option<f64> = None;
    let mut to_sec: Option<f64> = None;
    let mut mode = CutMode::Smart;
    let mut format: Option<String> = None;
    let mut video_codec: Option<String> = None;
    let mut audio_codec: Option<String> = None;
    let mut strict = false;
    let mut preserve_all = true;
    let mut overwrite = false;
    let mut json_out = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: wedeo cut -i <input> --from <time> --to <time> [--mode keyframe|smart|reencode] -o <output> [-y] [--json]"
                );
                return Ok(());
            }
            "-i" | "--input" => input = Some(next_val(args, &mut i, "--input")?),
            "-o" | "--output" => output = Some(next_val(args, &mut i, "--output")?),
            "--from" => {
                let v = next_val(args, &mut i, "--from")?;
                from_sec = Some(parse_timestamp_sec(&v)?);
            }
            "--to" => {
                let v = next_val(args, &mut i, "--to")?;
                to_sec = Some(parse_timestamp_sec(&v)?);
            }
            "--mode" => {
                let v = next_val(args, &mut i, "--mode")?;
                mode = CutMode::parse(&v)?;
            }
            "-f" | "--format" => format = Some(next_val(args, &mut i, "--format")?),
            "--video" => video_codec = Some(next_val(args, &mut i, "--video")?),
            "--audio" => audio_codec = Some(next_val(args, &mut i, "--audio")?),
            "--strict" => strict = true,
            "--preserve-all" => preserve_all = true,
            "--no-preserve-all" => preserve_all = false,
            "-y" | "--overwrite" => overwrite = true,
            "--json" => json_out = true,
            flag if flag.starts_with('-') => {
                return Err(Error::invalid_argument(
                    "cut",
                    format!("unknown option '{flag}'"),
                ));
            }
            pos => {
                if input.is_none() {
                    input = Some(pos.to_string());
                } else if output.is_none() {
                    output = Some(pos.to_string());
                } else {
                    return Err(Error::invalid_argument(
                        "cut",
                        format!("unexpected positional argument '{pos}'"),
                    ));
                }
            }
        }
        i += 1;
    }

    let in_path = input.ok_or_else(|| Error::invalid_argument("cut", "missing -i <input>"))?;
    let out_path = output.ok_or_else(|| Error::invalid_argument("cut", "missing -o <output>"))?;
    let from = from_sec.ok_or_else(|| Error::invalid_argument("cut", "missing --from <time>"))?;
    let to = to_sec.ok_or_else(|| Error::invalid_argument("cut", "missing --to <time>"))?;

    let mut req = CutReq::new(in_path, out_path, from, to);
    req.mode = mode;
    req.format = format;
    req.video_codec = video_codec;
    req.audio_codec = audio_codec;
    req.strict = strict;
    req.preserve_all = preserve_all;
    req.overwrite = overwrite;

    let report = run_cut(reg, &req)?;
    if json_out {
        println!("{}", report.to_json());
    } else {
        print!("{}", report.format_human());
    }
    Ok(())
}

fn cmd_mux(reg: &Registry, args: &[String]) -> Result<()> {
    let mut video_inputs = Vec::new();
    let mut audio_inputs = Vec::new();
    let mut extra_inputs = Vec::new();
    let mut output: Option<String> = None;
    let mut format: Option<String> = None;
    let mut tags = Vec::new();
    let mut strict = false;
    let mut preserve_all = true;
    let mut overwrite = false;
    let mut json_out = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: wedeo mux [--video <v>]... [--audio <a>]... [-i <file>]... --output <out> [--tag KEY=VAL] [-y] [--json]"
                );
                return Ok(());
            }
            "--video" => video_inputs.push(next_val(args, &mut i, "--video")?),
            "--audio" => audio_inputs.push(next_val(args, &mut i, "--audio")?),
            "-i" | "--input" => extra_inputs.push(next_val(args, &mut i, "--input")?),
            "-o" | "--output" => output = Some(next_val(args, &mut i, "--output")?),
            "-f" | "--format" => format = Some(next_val(args, &mut i, "--format")?),
            "--tag" => {
                let kv = next_val(args, &mut i, "--tag")?;
                let Some((k, v)) = kv.split_once('=') else {
                    return Err(Error::invalid_argument(
                        "mux",
                        format!("invalid --tag '{kv}', expected KEY=VALUE"),
                    ));
                };
                tags.push((k.to_string(), v.to_string()));
            }
            "--strict" => strict = true,
            "--preserve-all" => preserve_all = true,
            "--no-preserve-all" => preserve_all = false,
            "-y" | "--overwrite" => overwrite = true,
            "--json" => json_out = true,
            flag if flag.starts_with('-') => {
                return Err(Error::invalid_argument(
                    "mux",
                    format!("unknown option '{flag}'"),
                ));
            }
            pos => {
                if output.is_none() {
                    output = Some(pos.to_string());
                } else {
                    return Err(Error::invalid_argument(
                        "mux",
                        format!("unexpected positional argument '{pos}'"),
                    ));
                }
            }
        }
        i += 1;
    }

    let out_path =
        output.ok_or_else(|| Error::invalid_argument("mux", "missing --output <file>"))?;
    let mut req = MuxReq::new(out_path);
    req.video_inputs = video_inputs;
    req.audio_inputs = audio_inputs;
    req.extra_inputs = extra_inputs;
    req.format = format;
    req.tags = tags;
    req.strict = strict;
    req.preserve_all = preserve_all;
    req.overwrite = overwrite;

    let report = run_mux(reg, &req)?;
    if json_out {
        println!("{}", report.to_json());
    } else {
        print!("{}", report.format_human());
    }
    Ok(())
}

fn cmd_ffmpeg(reg: &Registry, args: &[String]) -> Result<()> {
    match execute_ffmpeg(reg, args)? {
        FfmpegResult::Text(txt) => print!("{txt}"),
        FfmpegResult::Report(rep) => print!("{}", rep.format_human()),
    }
    Ok(())
}

fn parse_bitrate_value(s: &str) -> Result<i64> {
    let s = s.trim();
    if let Some(prefix) = s.strip_suffix('k').or_else(|| s.strip_suffix('K')) {
        let v: i64 = prefix
            .parse()
            .map_err(|_| Error::invalid_argument("wedeo", format!("invalid bitrate '{s}'")))?;
        Ok(v.saturating_mul(1000))
    } else if let Some(prefix) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
        let v: i64 = prefix
            .parse()
            .map_err(|_| Error::invalid_argument("wedeo", format!("invalid bitrate '{s}'")))?;
        Ok(v.saturating_mul(1_000_000))
    } else {
        s.parse()
            .map_err(|_| Error::invalid_argument("wedeo", format!("invalid bitrate '{s}'")))
    }
}

// ---------------------------------------------------------------------------
// Comprehensive Unit & CLI Integration Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::json::JsonValue;
    use wedeo::{
        AudioStreamParams, Buffer, ChannelLayout, CodecId, ContainerHeader, FileSink, Packet,
        PacketFlags, Rational, SampleFormat, StreamParams, Timestamp, TrackDescriptor, TrackId,
    };

    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

    fn temp_path(name: &str) -> PathBuf {
        let id = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = env::temp_dir();
        p.push(format!("wedeo_cli_test_{}_{}_{}", process::id(), id, name));
        p
    }

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    fn vp9_red_ivf() -> PathBuf {
        workspace_root().join("tests/wedeo-oracle/data/vp9_red_64x64.ivf")
    }

    fn vp9_testsrc_ivf() -> PathBuf {
        workspace_root().join("tests/wedeo-oracle/data/vp9_testsrc_128x128.ivf")
    }

    fn create_synthetic_wav(path: &Path, sample_rate: u32, samples_per_ch: usize) {
        let reg = default_registry();
        let muxer_factory = reg.find_muxer_by_name("wav").unwrap();
        let sink = FileSink::create(path).unwrap();
        let mut muxer = muxer_factory.create(Box::new(sink)).unwrap();

        let tb = Rational::new(1, sample_rate as i32);
        let ap = AudioStreamParams::new(sample_rate, SampleFormat::S16, ChannelLayout::stereo());
        let mut hdr = ContainerHeader::new("wav");
        hdr.time_base = tb;
        hdr.tracks.push(TrackDescriptor::new(
            TrackId(0),
            CodecId::PcmS16le,
            tb,
            StreamParams::Audio(ap),
        ));
        muxer.write_header(&hdr).unwrap();

        let mut bytes = vec![0u8; samples_per_ch * 2 * 2];
        for i in 0..samples_per_ch {
            let v = ((i as i32 * 257) % 10000) as i16;
            bytes[i * 4..i * 4 + 2].copy_from_slice(&v.to_le_bytes());
            bytes[i * 4 + 2..i * 4 + 4].copy_from_slice(&(-v).to_le_bytes());
        }
        let mut pkt = Packet::new(Buffer::from_slice(&bytes), TrackId(0), tb);
        pkt.pts = Timestamp::new(0);
        pkt.dts = Timestamp::new(0);
        pkt.duration = samples_per_ch as i64;
        pkt.flags = PacketFlags::KEY;
        muxer.write_packet(&pkt).unwrap();
        muxer.finalize().unwrap();
    }

    #[test]
    fn test_inspect_json_and_verify_on_vp9_red_ivf() {
        let ivf = vp9_red_ivf();
        assert!(
            ivf.exists(),
            "missing bundled test vector {}",
            ivf.display()
        );

        let reg = default_registry();
        let mut req = InspectReq::new(ivf.to_string_lossy().to_string());
        req.level = InspectLevel::Nal;
        req.verify = true;
        let rep = run_inspect(&reg, &req).unwrap();
        assert_eq!(rep.format_name, "ivf");
        assert!(rep.verified);
        assert_eq!(rep.decoded_units, 1);
        assert!(rep.health.is_clean);
        assert_eq!(rep.tracks.len(), 1);
        assert_eq!(rep.tracks[0].codec, "vp9");
        assert_eq!(rep.tracks[0].width, Some(64));
        assert_eq!(rep.tracks[0].height, Some(64));
        assert_eq!(rep.packets.len(), 1);
        assert!(rep.packets[0].is_key);
        assert!(!rep.packets[0].units.is_empty());

        let parsed = JsonValue::parse(&rep.to_json()).unwrap();
        assert_eq!(
            parsed.get("verified").and_then(JsonValue::as_bool),
            Some(true)
        );
        assert_eq!(
            parsed.get("decoded_units").and_then(JsonValue::as_u64),
            Some(1)
        );
        assert_eq!(
            parsed
                .get("health")
                .and_then(|h| h.get("is_clean"))
                .and_then(JsonValue::as_bool),
            Some(true)
        );
    }

    #[test]
    fn test_explain_and_transcode_av1_mp4_in_process() {
        let ivf = vp9_red_ivf();
        let reg = default_registry();
        let out_mp4 = temp_path("inproc_av1.mp4");

        let mut tr = TranscodeReq::new(
            ivf.to_string_lossy().to_string(),
            out_mp4.to_string_lossy().to_string(),
        );
        tr.video_codec = Some("av1".to_string());
        tr.size = Some((32, 32));
        tr.overwrite = true;

        let plan = run_explain(&reg, &ExplainReq::new(tr.clone())).unwrap();
        assert!(!out_mp4.exists(), "explain must not write output file");
        assert!(plan.nodes.iter().any(|n| n.name == "sws_op_list"));
        assert!(!plan.conversion_notes.is_empty());

        let rep = run_transcode(&reg, &tr).unwrap();
        assert!(out_mp4.exists());
        assert_eq!(rep.format, "mp4");
        assert!(rep.health.is_clean);

        let _ = std::fs::remove_file(&out_mp4);
    }

    #[test]
    fn test_mcp_jsonrpc_unit_roundtrip() {
        let reg = default_registry();
        let ivf = vp9_red_ivf();

        let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let r1 = JsonValue::parse(&mcp::handle_jsonrpc_line(&reg, init).unwrap()).unwrap();
        assert_eq!(r1.get("id").and_then(JsonValue::as_i64), Some(1));

        let notif = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        assert!(mcp::handle_jsonrpc_line(&reg, notif).is_none());

        let list = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
        let r2 = JsonValue::parse(&mcp::handle_jsonrpc_line(&reg, list).unwrap()).unwrap();
        assert_eq!(
            r2.get("result")
                .and_then(|r| r.get("tools"))
                .and_then(JsonValue::as_arr)
                .map(<[JsonValue]>::len),
            Some(5)
        );

        let call = JsonValue::obj(vec![
            ("jsonrpc", "2.0".into()),
            ("id", 3i64.into()),
            ("method", "tools/call".into()),
            (
                "params",
                JsonValue::obj(vec![
                    ("name", "inspect".into()),
                    (
                        "arguments",
                        JsonValue::obj(vec![
                            ("input", ivf.to_string_lossy().to_string().into()),
                            ("verify", true.into()),
                        ]),
                    ),
                ]),
            ),
        ])
        .to_compact();
        let r3 = JsonValue::parse(&mcp::handle_jsonrpc_line(&reg, &call).unwrap()).unwrap();
        assert_eq!(
            r3.get("result")
                .and_then(|r| r.get("isError"))
                .and_then(JsonValue::as_bool),
            Some(false)
        );
    }

    #[test]
    fn test_mux_and_ffmpeg_vn_stream_index_remapping() {
        let ivf = vp9_red_ivf();
        let wav = temp_path("audio.wav");
        let muxed_mkv = temp_path("muxed.mkv");
        let audio_only_mkv = temp_path("audio_only.mkv");

        create_synthetic_wav(&wav, 48_000, 4800);

        let reg = default_registry();
        let mut mux_req = MuxReq::new(muxed_mkv.to_string_lossy().to_string());
        mux_req.video_inputs.push(ivf.to_string_lossy().to_string());
        mux_req.audio_inputs.push(wav.to_string_lossy().to_string());
        mux_req
            .tags
            .push(("title".to_string(), "Wedeo Mux Test".to_string()));
        mux_req.overwrite = true;
        let mux_rep = run_mux(&reg, &mux_req).unwrap();
        assert_eq!(mux_rep.tracks, 2);

        // Execute `wedeo ffmpeg` compat translator with `-vn` (remaps track #1 -> track #0)
        let args: Vec<String> = vec![
            "-i",
            muxed_mkv.to_str().unwrap(),
            "-vn",
            "-c:a",
            "copy",
            "-y",
            audio_only_mkv.to_str().unwrap(),
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        let res = execute_ffmpeg(&reg, &args).unwrap();
        assert!(matches!(res, FfmpegResult::Report(_)));

        let inspect_out = run_inspect(
            &reg,
            &InspectReq {
                input: audio_only_mkv.to_string_lossy().to_string(),
                level: InspectLevel::Container,
                verify: true,
                max_packets: None,
            },
        )
        .unwrap();
        assert_eq!(inspect_out.tracks.len(), 1);
        assert_eq!(inspect_out.tracks[0].media_type, "audio");
        assert!(inspect_out.health.is_clean);
        assert!(inspect_out.decoded_units > 0);

        let _ = std::fs::remove_file(&wav);
        let _ = std::fs::remove_file(&muxed_mkv);
        let _ = std::fs::remove_file(&audio_only_mkv);
    }

    #[test]
    fn test_cut_modes_on_multi_frame_ivf() {
        let ivf = vp9_testsrc_ivf();
        assert!(
            ivf.exists(),
            "missing bundled test vector {}",
            ivf.display()
        );
        let reg = default_registry();

        // 1. Keyframe cut on bundled VP9 IVF -> MKV
        let out_kf = temp_path("cut_kf.mkv");
        let mut req_kf = CutReq::new(
            ivf.to_string_lossy().to_string(),
            out_kf.to_string_lossy().to_string(),
            0.0,
            0.04,
        );
        req_kf.mode = CutMode::Keyframe;
        req_kf.overwrite = true;
        let rep_kf = run_cut(&reg, &req_kf).unwrap();
        assert!(rep_kf.packets_processed > 0);
        assert!(!rep_kf.smart_cut_segments.is_empty());

        // 2. Build a 9-frame, 3-GOP synthetic AV1 MKV (keyframes at 0, 300ms, 600ms; 100ms/frame)
        let av1_src = temp_path("cut_src_av1.mkv");
        let pool = wedeo::BufferPool::new();
        let tb = Rational::new(1, 1000);
        let mut vp = wedeo::VideoStreamParams::new(64, 64, wedeo::PixelLayout::YUV420P);
        vp.color = wedeo::ColorDesc::BT709_LIMITED;
        let mut enc_track = TrackDescriptor::new(
            TrackId(0),
            CodecId::Av1,
            tb,
            StreamParams::Video(vp.clone()),
        );
        let mut enc = reg
            .find_encoder(CodecId::Av1)
            .unwrap()
            .open(
                &enc_track,
                wedeo::CodecContext::new(std::sync::Arc::clone(&pool)),
                wedeo::PassMode::Single,
            )
            .unwrap();
        if let StreamParams::Video(ref mut vp_mut) = enc_track.params {
            vp_mut.extradata = enc.extradata().to_vec();
        }
        let mut hdr = ContainerHeader::new("matroska");
        hdr.time_base = tb;
        hdr.tracks.push(enc_track);
        let mut muxer = reg
            .find_muxer_by_name("matroska")
            .unwrap()
            .create(Box::new(FileSink::create(&av1_src).unwrap()))
            .unwrap();
        muxer.write_header(&hdr).unwrap();

        for i in 0..9i64 {
            let mut y = wedeo::BorderedPlane::new(&pool, 64, 64, wedeo::BorderPadding::NONE);
            for r in 0..64 {
                y.row_mut(r).unwrap().fill((20 + i * 20) as u8);
            }
            let mut u = wedeo::BorderedPlane::new(&pool, 32, 32, wedeo::BorderPadding::NONE);
            let mut v = wedeo::BorderedPlane::new(&pool, 32, 32, wedeo::BorderPadding::NONE);
            for r in 0..32 {
                u.row_mut(r).unwrap().fill(128);
                v.row_mut(r).unwrap().fill(128);
            }
            let mut frame =
                wedeo::VideoFrame::from_planes(64, 64, wedeo::PixelLayout::YUV420P, vec![y, u, v]);
            frame.color = wedeo::ColorDesc::BT709_LIMITED;
            frame.pts = Timestamp::new(i * 100);
            frame.duration = 100;
            frame.time_base = tb;
            let hint = wedeo::EncodeHint {
                force_keyframe: i % 3 == 0,
                target_qp: None,
            };
            enc.send_unit(Some(&wedeo::MediaUnit::Video(frame)), hint)
                .unwrap();
            while let wedeo::RecvStatus::Ready(pkt) = enc.receive_packet().unwrap() {
                muxer.write_packet(&pkt).unwrap();
            }
        }
        enc.send_unit(None, wedeo::EncodeHint::default()).unwrap();
        while let wedeo::RecvStatus::Ready(pkt) = enc.receive_packet().unwrap() {
            muxer.write_packet(&pkt).unwrap();
        }
        muxer.finalize().unwrap();

        // Cut [0.15s, 0.75s) -> ReencodeHead [150..300) + CopyInterior [300..600) + ReencodeTail [600..750)
        let out_smart = temp_path("cut_smart_3gop.mkv");
        let mut req_smart = CutReq::new(
            av1_src.to_string_lossy().to_string(),
            out_smart.to_string_lossy().to_string(),
            0.15,
            0.75,
        );
        req_smart.mode = CutMode::Smart;
        req_smart.overwrite = true;
        let rep_smart = run_cut(&reg, &req_smart).unwrap();
        assert_eq!(
            rep_smart.smart_cut_segments.len(),
            3,
            "segments: {:?}",
            rep_smart.smart_cut_segments
        );
        assert!(rep_smart.smart_cut_segments[0].contains("ReencodeHead"));
        assert!(rep_smart.smart_cut_segments[1].contains("CopyInterior"));
        assert!(rep_smart.smart_cut_segments[2].contains("ReencodeTail"));

        let ins = run_inspect(
            &reg,
            &InspectReq {
                input: out_smart.to_string_lossy().to_string(),
                level: InspectLevel::Packets,
                verify: true,
                max_packets: None,
            },
        )
        .unwrap();
        assert!(ins.health.is_clean);
        assert_eq!(ins.decoded_units, 6); // frames at 200, 300, 400, 500, 600, 700 ms

        let _ = std::fs::remove_file(&out_kf);
        let _ = std::fs::remove_file(&av1_src);
        let _ = std::fs::remove_file(&out_smart);
    }
}
