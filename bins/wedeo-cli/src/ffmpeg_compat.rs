//! Order-sensitive `wedeo ffmpeg ...` compatibility translator.
//!
//! Compiles classic FFmpeg CLI flags directly into the unified typed operations
//! in `ops.rs` (`InspectReq`, `TranscodeReq`, `MuxReq`), fixing both the legacy
//! `av1 -> mp4` empty-extradata bug and the `-vn` stream-index remapping bug.

use wedeo::{Error, Registry, Result};

use crate::ops::{
    InspectReq, MuxReq, RunReport, TranscodeReq, format_codecs_table, format_decoders_table,
    format_encoders_table, format_formats_table, parse_pixel_layout, parse_size,
    parse_timestamp_sec, run_inspect, run_mux, run_transcode,
};

#[derive(Debug, Clone)]
pub enum FfmpegAction {
    Usage,
    ListCodecs,
    ListFormats,
    ListDecoders,
    ListEncoders,
    Inspect(Vec<InspectReq>),
    Transcode(TranscodeReq),
    Mux(MuxReq),
}

#[derive(Debug, Clone)]
pub struct FfmpegInputSpec {
    pub path: String,
    pub ss: Option<f64>,
    pub t: Option<f64>,
}

pub fn parse_ffmpeg_args(args: &[String]) -> Result<FfmpegAction> {
    if args.is_empty() {
        return Ok(FfmpegAction::Usage);
    }

    let mut inputs: Vec<FfmpegInputSpec> = Vec::new();
    let mut output: Option<String> = None;

    // Input-scoped pending options (before -i)
    let mut pending_in_ss: Option<f64> = None;
    let mut pending_in_t: Option<f64> = None;

    // Output-scoped options
    let mut video_codec: Option<String> = None;
    let mut audio_codec: Option<String> = None;
    let mut format: Option<String> = None;
    let mut video_bitrate: Option<i64> = None;
    let mut audio_bitrate: Option<i64> = None;
    let mut frames_video: Option<u64> = None;
    let mut video_disabled = false;
    let mut audio_disabled = false;
    let mut overwrite = false;
    let mut strict = false;
    let mut out_ss: Option<f64> = None;
    let mut out_t: Option<f64> = None;
    let mut size: Option<(u32, u32)> = None;
    let mut pixel_layout = None;
    let mut sample_rate: Option<u32> = None;
    let mut channels: Option<u32> = None;
    let mut codec_options: Vec<(String, String)> = Vec::new();

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "-help" | "--help" => return Ok(FfmpegAction::Usage),
            "-codecs" => return Ok(FfmpegAction::ListCodecs),
            "-formats" => return Ok(FfmpegAction::ListFormats),
            "-decoders" => return Ok(FfmpegAction::ListDecoders),
            "-encoders" => return Ok(FfmpegAction::ListEncoders),
            "-y" => overwrite = true,
            "-vn" => video_disabled = true,
            "-an" => audio_disabled = true,
            "-i" => {
                let path = next_arg(args, &mut i, "-i")?;
                inputs.push(FfmpegInputSpec {
                    path,
                    ss: pending_in_ss.take(),
                    t: pending_in_t.take(),
                });
            }
            "-c:v" | "-vcodec" => {
                video_codec = Some(next_arg(args, &mut i, arg)?);
            }
            "-c:a" | "-acodec" => {
                audio_codec = Some(next_arg(args, &mut i, arg)?);
            }
            "-c" => {
                let val = next_arg(args, &mut i, "-c")?;
                video_codec = Some(val.clone());
                audio_codec = Some(val);
            }
            "-f" => {
                format = Some(next_arg(args, &mut i, "-f")?);
            }
            "-b:v" => {
                let val = next_arg(args, &mut i, "-b:v")?;
                video_bitrate = Some(parse_bitrate(&val)?);
            }
            "-b:a" => {
                let val = next_arg(args, &mut i, "-b:a")?;
                audio_bitrate = Some(parse_bitrate(&val)?);
            }
            "-frames:v" | "-vframes" => {
                let val = next_arg(args, &mut i, arg)?;
                let n = val.parse::<u64>().map_err(|_| {
                    Error::invalid_argument("ffmpeg", format!("invalid frame count '{val}'"))
                })?;
                frames_video = Some(n);
            }
            "-ss" => {
                let val = next_arg(args, &mut i, "-ss")?;
                let sec = parse_timestamp_sec(&val)?;
                if inputs.is_empty() {
                    pending_in_ss = Some(sec);
                } else {
                    out_ss = Some(sec);
                }
            }
            "-t" => {
                let val = next_arg(args, &mut i, "-t")?;
                let sec = parse_timestamp_sec(&val)?;
                if inputs.is_empty() {
                    pending_in_t = Some(sec);
                } else {
                    out_t = Some(sec);
                }
            }
            "-s" => {
                let val = next_arg(args, &mut i, "-s")?;
                size = Some(parse_size(&val)?);
            }
            "-pix_fmt" => {
                let val = next_arg(args, &mut i, "-pix_fmt")?;
                pixel_layout = Some(parse_pixel_layout(&val).ok_or_else(|| {
                    Error::invalid_argument("ffmpeg", format!("unknown pixel format '{val}'"))
                })?);
            }
            "-ar" => {
                let val = next_arg(args, &mut i, "-ar")?;
                let sr = val.parse::<u32>().map_err(|_| {
                    Error::invalid_argument("ffmpeg", format!("invalid sample rate '{val}'"))
                })?;
                sample_rate = Some(sr);
            }
            "-ac" => {
                let val = next_arg(args, &mut i, "-ac")?;
                let ch = val.parse::<u32>().map_err(|_| {
                    Error::invalid_argument("ffmpeg", format!("invalid channel count '{val}'"))
                })?;
                channels = Some(ch);
            }
            "-v" | "-loglevel" => {
                let _ = next_arg(args, &mut i, arg)?;
            }
            "-strict" => {
                if let Some(next) = args.get(i + 1)
                    && !next.starts_with('-')
                {
                    i += 1;
                    strict = matches!(next.as_str(), "strict" | "2" | "very" | "true");
                } else {
                    strict = true;
                }
            }
            "-hide_banner" | "-nostdin" => {}
            s if s.starts_with('-') && s.len() > 1 => {
                let key = s.trim_start_matches('-').to_string();
                let val = next_arg(args, &mut i, s)?;
                codec_options.push((key, val));
            }
            pos => {
                if output.is_some() {
                    return Err(Error::invalid_argument(
                        "ffmpeg",
                        format!("unexpected positional argument '{pos}'"),
                    ));
                }
                output = Some(pos.to_string());
            }
        }
        i += 1;
    }

    if inputs.is_empty() {
        return Err(Error::invalid_argument(
            "ffmpeg",
            "at least one input (-i <file>) is required",
        ));
    }

    let Some(out_path) = output else {
        let reqs = inputs
            .into_iter()
            .map(|inp| InspectReq::new(inp.path))
            .collect();
        return Ok(FfmpegAction::Inspect(reqs));
    };

    if inputs.len() > 1 {
        let mut mux = MuxReq::new(out_path);
        mux.extra_inputs = inputs.into_iter().map(|inp| inp.path).collect();
        mux.format = format;
        mux.strict = strict;
        mux.overwrite = overwrite;
        return Ok(FfmpegAction::Mux(mux));
    }

    let inp = inputs.remove(0);
    let mut tr = TranscodeReq::new(inp.path, out_path);
    tr.format = format;
    tr.video_codec = video_codec;
    tr.audio_codec = audio_codec;
    tr.video_bitrate = video_bitrate;
    tr.audio_bitrate = audio_bitrate;
    tr.frames_video = frames_video;
    tr.no_video = video_disabled;
    tr.no_audio = audio_disabled;
    tr.size = size;
    tr.pixel_layout = pixel_layout;
    tr.sample_rate = sample_rate;
    tr.channels = channels;
    tr.start_sec = out_ss.or(inp.ss);
    tr.duration_sec = out_t.or(inp.t);
    tr.strict = strict;
    tr.overwrite = overwrite;
    tr.codec_options = codec_options;

    Ok(FfmpegAction::Transcode(tr))
}

pub fn execute_ffmpeg(reg: &Registry, args: &[String]) -> Result<FfmpegResult> {
    match parse_ffmpeg_args(args)? {
        FfmpegAction::Usage => Ok(FfmpegResult::Text(ffmpeg_usage_text())),
        FfmpegAction::ListCodecs => Ok(FfmpegResult::Text(format_codecs_table(reg))),
        FfmpegAction::ListFormats => Ok(FfmpegResult::Text(format_formats_table(reg))),
        FfmpegAction::ListDecoders => Ok(FfmpegResult::Text(format_decoders_table(reg))),
        FfmpegAction::ListEncoders => Ok(FfmpegResult::Text(format_encoders_table(reg))),
        FfmpegAction::Inspect(reqs) => {
            let mut out = String::new();
            for req in &reqs {
                let rep = run_inspect(reg, req)?;
                out.push_str(&rep.format_human());
            }
            Ok(FfmpegResult::Text(out))
        }
        FfmpegAction::Transcode(req) => {
            Ok(FfmpegResult::Report(Box::new(run_transcode(reg, &req)?)))
        }
        FfmpegAction::Mux(req) => Ok(FfmpegResult::Report(Box::new(run_mux(reg, &req)?))),
    }
}

pub enum FfmpegResult {
    Text(String),
    Report(Box<RunReport>),
}

pub fn ffmpeg_usage_text() -> String {
    "\
wedeo ffmpeg — FFmpeg-compatible order-sensitive flag translator

Usage:
  wedeo ffmpeg [input-options] -i <input> [output-options] <output>

Supported Flags:
  -i <input>        Input file (repeatable for remuxing)
  -c:v <codec>      Video codec (e.g. av1, copy)
  -c:a <codec>      Audio codec (e.g. pcm_s16le, copy)
  -c <codec>        Set both video and audio codec
  -b:v <bitrate>    Video bitrate (e.g. 500k, 2M)
  -b:a <bitrate>    Audio bitrate
  -frames:v <n>     Stop after n video frames
  -vn / -an         Disable video / audio stream
  -ss <time>        Start offset (seconds or HH:MM:SS)
  -t <duration>     Duration limit
  -s <WxH>          Output video dimensions
  -pix_fmt <fmt>    Output pixel layout (e.g. yuv420p)
  -ar <hz>          Output audio sample rate
  -ac <channels>    Output audio channel count
  -f <format>       Force output container format
  -y                Overwrite output without failing
  -codecs/-formats  List registered codecs / formats
"
    .to_string()
}

fn next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String> {
    *idx += 1;
    args.get(*idx).cloned().ok_or_else(|| {
        Error::invalid_argument("ffmpeg", format!("option {flag} requires an argument"))
    })
}

fn parse_bitrate(s: &str) -> Result<i64> {
    let s = s.trim();
    if let Some(prefix) = s.strip_suffix('k').or_else(|| s.strip_suffix('K')) {
        let v = prefix
            .parse::<i64>()
            .map_err(|_| Error::invalid_argument("ffmpeg", format!("invalid bitrate '{s}'")))?;
        Ok(v.saturating_mul(1000))
    } else if let Some(prefix) = s.strip_suffix('M').or_else(|| s.strip_suffix('m')) {
        let v = prefix
            .parse::<i64>()
            .map_err(|_| Error::invalid_argument("ffmpeg", format!("invalid bitrate '{s}'")))?;
        Ok(v.saturating_mul(1_000_000))
    } else {
        s.parse::<i64>()
            .map_err(|_| Error::invalid_argument("ffmpeg", format!("invalid bitrate '{s}'")))
    }
}
