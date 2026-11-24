//! Drop-in `ffmpeg` & `ffprobe` compatibility translator.
//!
//! Works both explicitly (`wedeo ffmpeg ...`, `wedeo ffprobe ...`) and as a
//! BusyBox-style multi-call binary when symlinked as `ffmpeg` / `ffprobe`
//! (e.g. via `wedeo install-shim <dir>` and `yt-dlp --ffmpeg-location <dir>`).
//!
//! Compiles order-sensitive FFmpeg CLI flags into Wedeo's typed operations
//! (`InspectReq`, `TranscodeReq`, `MuxReq`), stripping `file:` URI prefixes
//! and handling common post-processor flags (`-version`, `-map`, `-movflags`,
//! `-bsf:*`, `-metadata`, `-sn`/`-dn`/`-ignore_unknown`, `-to`, etc.).

use std::collections::BTreeMap;

use wedeo::core::Chapter;
use wedeo::{Error, Metadata, Rational, Registry, Result, Timestamp};

use crate::json::JsonValue;
use crate::ops::{
    InspectReport, InspectReq, MuxReq, RunReport, TranscodeReq, format_codecs_table,
    format_decoders_table, format_encoders_table, format_formats_table, parse_pixel_layout,
    parse_size, parse_timestamp_sec, run_inspect, run_mux, run_transcode,
};

#[derive(Debug, Clone)]
pub enum FfmpegAction {
    Usage,
    Version,
    ListBsfs,
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
    pub to: Option<f64>,
}

/// Strip FFmpeg's `file:` protocol prefix (`file:foo.mp4` -> `foo.mp4`) used by
/// tools like `yt-dlp` to prevent colons in filenames from being treated as schemes.
pub fn strip_file_proto(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("file://") {
        rest.to_string()
    } else if let Some(rest) = path.strip_prefix("file:") {
        rest.to_string()
    } else {
        path.to_string()
    }
}

/// Normalize FFmpeg muxer format aliases (e.g. `webm` -> `matroska`, `ipod` -> `mp4`).
fn normalize_format_alias(fmt: &str) -> String {
    match fmt.to_ascii_lowercase().as_str() {
        "webm" | "mkv" => "matroska".to_string(),
        "ipod" | "m4a" | "m4v" | "mov" | "3gp" => "mp4".to_string(),
        other => other.to_string(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MapKind {
    All,
    Video,
    Audio,
    Other,
}

#[derive(Debug, Clone)]
struct MapRule {
    negative: bool,
    input_idx: usize,
    kind: MapKind,
}

fn parse_map_spec(spec: &str) -> Option<MapRule> {
    let (negative, rest) = match spec.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, spec),
    };
    let rest = rest.trim_end_matches('?');
    let mut parts = rest.split(':');
    let input_idx = parts.next()?.parse::<usize>().ok()?;
    let kind = match parts.next() {
        None => MapKind::All,
        Some("v") | Some("V") => MapKind::Video,
        Some("a") => MapKind::Audio,
        Some(_) => MapKind::Other,
    };
    Some(MapRule {
        negative,
        input_idx,
        kind,
    })
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
    let mut pending_in_to: Option<f64> = None;

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
    let mut out_to: Option<f64> = None;
    let mut size: Option<(u32, u32)> = None;
    let mut pixel_layout = None;
    let mut sample_rate: Option<u32> = None;
    let mut channels: Option<u32> = None;
    let mut maps: Vec<MapRule> = Vec::new();
    let mut tags: Vec<(String, String)> = Vec::new();
    let mut chapters: Vec<Chapter> = Vec::new();
    let mut codec_options: Vec<(String, String)> = Vec::new();

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-h" | "-help" | "--help" => return Ok(FfmpegAction::Usage),
            "-version" | "--version" => return Ok(FfmpegAction::Version),
            "-bsfs" => return Ok(FfmpegAction::ListBsfs),
            "-codecs" => return Ok(FfmpegAction::ListCodecs),
            "-formats" => return Ok(FfmpegAction::ListFormats),
            "-decoders" => return Ok(FfmpegAction::ListDecoders),
            "-encoders" => return Ok(FfmpegAction::ListEncoders),
            "-y" => overwrite = true,
            "-n" => overwrite = false,
            "-vn" => video_disabled = true,
            "-an" => audio_disabled = true,
            // Common boolean (0-arg) flags passed by yt-dlp / FFmpeg scripts.
            "-hide_banner" | "-nostdin" | "-sn" | "-dn" | "-ignore_unknown" | "-stats"
            | "-nostats" | "-copyts" | "-start_at_zero" | "-shortest" | "-fix_sub_duration"
            | "-bitexact" | "-accurate_seek" | "-noaccurate_seek" | "-copyinkf" | "-re" => {}
            "-i" => {
                let raw = next_arg(args, &mut i, "-i")?;
                let path = strip_file_proto(&raw);
                if let Some(meta) = try_parse_ffmetadata1(&path) {
                    tags.extend(meta.tags);
                    chapters.extend(meta.chapters);
                } else {
                    inputs.push(FfmpegInputSpec {
                        path,
                        ss: pending_in_ss.take(),
                        t: pending_in_t.take(),
                        to: pending_in_to.take(),
                    });
                }
            }
            "-c:v" | "-vcodec" | "-codec:v" => {
                video_codec = Some(next_arg(args, &mut i, arg)?);
            }
            "-c:a" | "-acodec" | "-codec:a" => {
                audio_codec = Some(next_arg(args, &mut i, arg)?);
            }
            "-c" | "-codec" => {
                let val = next_arg(args, &mut i, arg)?;
                video_codec = Some(val.clone());
                audio_codec = Some(val);
            }
            s if s.starts_with("-c:v:") || s.starts_with("-codec:v:") => {
                video_codec = Some(next_arg(args, &mut i, s)?);
            }
            s if s.starts_with("-c:a:") || s.starts_with("-codec:a:") => {
                audio_codec = Some(next_arg(args, &mut i, s)?);
            }
            s if s.starts_with("-c:s")
                || s.starts_with("-c:d")
                || s.starts_with("-c:t")
                || s == "-scodec" =>
            {
                let _ = next_arg(args, &mut i, s)?;
            }
            "-f" => {
                let val = next_arg(args, &mut i, "-f")?;
                if !inputs.is_empty() {
                    format = Some(normalize_format_alias(&val));
                }
            }
            "-map" => {
                let spec = next_arg(args, &mut i, "-map")?;
                if let Some(rule) = parse_map_spec(&spec) {
                    maps.push(rule);
                }
            }
            "-metadata" => {
                let kv = next_arg(args, &mut i, "-metadata")?;
                if let Some((k, v)) = kv.split_once('=') {
                    tags.push((k.to_string(), v.to_string()));
                }
            }
            s if s.starts_with("-metadata:")
                || s.starts_with("-disposition")
                || s.starts_with("-bsf")
                || s.starts_with("-tag:")
                || s.starts_with("-map_metadata")
                || s.starts_with("-map_chapters") =>
            {
                let _ = next_arg(args, &mut i, s)?;
            }
            "-movflags"
            | "-max_muxing_queue_size"
            | "-threads"
            | "-progress"
            | "-fflags"
            | "-flags"
            | "-avoid_negative_ts"
            | "-muxpreload"
            | "-muxdelay"
            | "-probesize"
            | "-analyzeduration"
            | "-q:a"
            | "-q:v"
            | "-qscale:a"
            | "-qscale:v"
            | "-r" => {
                let _ = next_arg(args, &mut i, arg)?;
            }
            "-b:v" | "-vb" => {
                let val = next_arg(args, &mut i, arg)?;
                video_bitrate = Some(parse_bitrate(&val)?);
            }
            "-b:a" | "-ab" => {
                let val = next_arg(args, &mut i, arg)?;
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
            "-to" => {
                let val = next_arg(args, &mut i, "-to")?;
                let sec = parse_timestamp_sec(&val)?;
                if inputs.is_empty() {
                    pending_in_to = Some(sec);
                } else {
                    out_to = Some(sec);
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
                output = Some(strip_file_proto(pos));
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
        mux.format = format;
        mux.tags = tags;
        mux.chapters = chapters;
        mux.strict = strict;
        mux.overwrite = overwrite;

        let pos_maps: Vec<&MapRule> = maps.iter().filter(|m| !m.negative).collect();
        if pos_maps.is_empty() {
            mux.extra_inputs = inputs.into_iter().map(|inp| inp.path).collect();
        } else {
            // Respect explicit `-map 0:v:0 -map 1:a:0` stream selection (e.g. yt-dlp merging).
            let mut per_input: BTreeMap<usize, (bool, bool, bool)> = BTreeMap::new();
            for m in pos_maps {
                if m.input_idx < inputs.len() {
                    let e = per_input
                        .entry(m.input_idx)
                        .or_insert((false, false, false));
                    match m.kind {
                        MapKind::All | MapKind::Other => e.2 = true,
                        MapKind::Video => e.0 = true,
                        MapKind::Audio => e.1 = true,
                    }
                }
            }
            for (idx, (want_v, want_a, want_all)) in per_input {
                let path = inputs[idx].path.clone();
                if want_all || (want_v && want_a) {
                    mux.extra_inputs.push(path);
                } else if want_v {
                    mux.video_inputs.push(path);
                } else if want_a {
                    mux.audio_inputs.push(path);
                }
            }
            if mux.video_inputs.is_empty()
                && mux.audio_inputs.is_empty()
                && mux.extra_inputs.is_empty()
            {
                mux.extra_inputs = inputs.into_iter().map(|inp| inp.path).collect();
            }
        }
        return Ok(FfmpegAction::Mux(mux));
    }

    // Apply single-input positive `-map` filters (e.g. `-map 0:a:0` disables video).
    let pos_maps: Vec<&MapRule> = maps.iter().filter(|m| !m.negative).collect();
    if !pos_maps.is_empty() && !pos_maps.iter().any(|m| m.kind == MapKind::All) {
        if !pos_maps.iter().any(|m| m.kind == MapKind::Video) {
            video_disabled = true;
        }
        if !pos_maps.iter().any(|m| m.kind == MapKind::Audio) {
            audio_disabled = true;
        }
    }
    for m in maps.iter().filter(|m| m.negative) {
        match m.kind {
            MapKind::Video => video_disabled = true,
            MapKind::Audio => audio_disabled = true,
            _ => {}
        }
    }

    let inp = inputs.remove(0);
    let start_sec = out_ss.or(inp.ss);
    let end_to = out_to.or(inp.to);
    let duration_sec = out_t.or(inp.t).or_else(|| {
        end_to.map(|to| match start_sec {
            Some(s) => (to - s).max(0.0),
            None => to,
        })
    });

    // If user requested `-f wav` or output `.wav` without `-c:a`, default to pcm_s16le
    // (matching FFmpeg's WAV muxer default so `yt-dlp -x --audio-format wav` works directly).
    let is_wav_out =
        format.as_deref() == Some("wav") || out_path.to_ascii_lowercase().ends_with(".wav");
    if is_wav_out {
        video_disabled = true;
        if audio_codec.is_none() {
            audio_codec = Some("pcm_s16le".to_string());
        }
    }

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
    tr.start_sec = start_sec;
    tr.duration_sec = duration_sec;
    tr.strict = strict;
    tr.overwrite = overwrite;
    tr.tags = tags;
    tr.chapters = chapters;
    tr.codec_options = codec_options;

    Ok(FfmpegAction::Transcode(tr))
}

pub fn execute_ffmpeg(reg: &Registry, args: &[String]) -> Result<FfmpegResult> {
    match parse_ffmpeg_args(args)? {
        FfmpegAction::Usage => Ok(FfmpegResult::Text(ffmpeg_usage_text())),
        FfmpegAction::Version => Ok(FfmpegResult::Text(ffmpeg_version_text("ffmpeg"))),
        FfmpegAction::ListBsfs => Ok(FfmpegResult::Text(ffmpeg_bsfs_text("ffmpeg"))),
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

pub fn ffmpeg_version_text(tool: &str) -> String {
    format!(
        "{tool} version 9.1-wedeo-{} Copyright (c) 2026 Wedeo contributors\n\
         built with rustc (pure-Rust FFmpeg n9.1 compatible shim)\n\
         configuration: --enable-pure-rust --enable-wedeo-shim\n\
         libavutil      60. 26.100 / 60. 26.100\n\
         libavcodec     62. 28.100 / 62. 28.100\n\
         libavformat    62. 12.100 / 62. 12.100\n",
        env!("CARGO_PKG_VERSION")
    )
}

pub fn ffmpeg_bsfs_text(tool: &str) -> String {
    format!(
        "{}\
         Bitstream filters:\n\
         aac_adtstoasc\n\
         h264_mp4toannexb\n\
         hevc_mp4toannexb\n\
         vp9_superframe_split\n\
         null\n",
        ffmpeg_version_text(tool)
    )
}

pub fn ffmpeg_usage_text() -> String {
    "\
wedeo ffmpeg — Drop-in FFmpeg-compatible order-sensitive flag translator

Usage:
  wedeo ffmpeg [input-options] -i <input> [output-options] <output>
  (or symlink `ffmpeg` & `ffprobe` via `wedeo install-shim <dir>`)

Supported Flags:
  -i <input>          Input file (repeatable for muxing; strips `file:` prefix)
  -map <spec>         Stream selection (e.g. `-map 0:v:0 -map 1:a:0`)
  -c:v / -c:a / -c    Codec selection (`copy`, `av1`, `pcm_s16le`, ...)
  -vn / -an / -sn     Disable video / audio / subtitle streams
  -ss / -t / -to      Trim start / duration / end time
  -s <WxH>            Scale video via SwsOpList
  -ar <hz> -ac <ch>   Resample / rematrix audio via AudioOpList
  -f <format>         Force output format (`mp4`, `matroska`, `webm`, `wav`, `ivf`)
  -metadata k=v       Attach container metadata tags
  -y / -n             Overwrite control
  -version            Print FFmpeg-compatible version header (for yt-dlp, etc.)
"
    .to_string()
}

// ---------------------------------------------------------------------------
// Drop-in `ffprobe` compatibility mode (JSON & compact output for yt-dlp/tools)
// ---------------------------------------------------------------------------

pub fn execute_ffprobe(reg: &Registry, args: &[String]) -> Result<String> {
    if args.is_empty() {
        return Ok("Usage: ffprobe [options] <input>\n".to_string());
    }

    let mut input: Option<String> = None;
    let mut show_format = false;
    let mut show_streams = false;
    let mut show_chapters = false;
    let mut print_format = "default".to_string();
    let mut select_streams: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "-version" | "--version" => return Ok(ffmpeg_version_text("ffprobe")),
            "-bsfs" => return Ok(ffmpeg_bsfs_text("ffprobe")),
            "-h" | "-help" | "--help" => {
                return Ok(
                    "Usage: ffprobe [-show_format] [-show_streams] [-of json] <input>\n"
                        .to_string(),
                );
            }
            "-show_format" => show_format = true,
            "-show_streams" => show_streams = true,
            "-show_chapters" => show_chapters = true,
            "-hide_banner" | "-show_error" | "-count_frames" | "-count_packets" => {}
            "-v" | "-loglevel" | "-show_entries" | "-probesize" | "-analyzeduration" => {
                let _ = next_arg(args, &mut i, arg)?;
            }
            "-print_format" | "-of" => {
                print_format = next_arg(args, &mut i, arg)?;
            }
            "-select_streams" => {
                select_streams = Some(next_arg(args, &mut i, arg)?);
            }
            "-i" => {
                let raw = next_arg(args, &mut i, "-i")?;
                input = Some(strip_file_proto(&raw));
            }
            s if s.starts_with('-') && s.len() > 1 => {
                // Skip unknown valued flag if followed by a non-flag token.
                if let Some(next) = args.get(i + 1)
                    && !next.starts_with('-')
                    && i + 2 < args.len()
                {
                    i += 1;
                }
            }
            pos => {
                input = Some(strip_file_proto(pos));
            }
        }
        i += 1;
    }

    let path = input.ok_or_else(|| {
        Error::invalid_argument("ffprobe", "an input media file path is required")
    })?;

    let req = InspectReq::new(path);
    let rep = run_inspect(reg, &req)?;

    // Default to showing both format and streams if JSON requested without explicit flags.
    if !show_format && !show_streams && !show_chapters {
        show_format = true;
        show_streams = true;
    }

    if print_format.starts_with("json") {
        let val = format_ffprobe_json(
            &rep,
            show_streams,
            show_format,
            show_chapters,
            select_streams.as_deref(),
        );
        Ok(format!("{}\n", val.to_pretty()))
    } else {
        Ok(format_ffprobe_default(
            &rep,
            show_streams,
            show_format,
            select_streams.as_deref(),
        ))
    }
}

fn format_ffprobe_default(
    rep: &InspectReport,
    show_streams: bool,
    show_format: bool,
    select_streams: Option<&str>,
) -> String {
    let mut out = String::new();
    if show_streams {
        for t in &rep.tracks {
            if let Some(sel) = select_streams {
                let want_video = sel == "v" || sel.starts_with("v:");
                let want_audio = sel == "a" || sel.starts_with("a:");
                if (want_video && t.media_type != "video")
                    || (want_audio && t.media_type != "audio")
                {
                    continue;
                }
            }
            out.push_str("[STREAM]\n");
            out.push_str(&format!("index={}\n", t.id));
            out.push_str(&format!("codec_name={}\n", t.codec));
            out.push_str(&format!("codec_type={}\n", t.media_type));
            out.push_str(&format!("time_base={}\n", t.time_base));
            if let Some(w) = t.width {
                out.push_str(&format!("width={w}\n"));
            }
            if let Some(h) = t.height {
                out.push_str(&format!("height={h}\n"));
            }
            if let Some(ref pix) = t.pixel_layout {
                out.push_str(&format!("pix_fmt={pix}\n"));
            }
            if let Some(sr) = t.sample_rate {
                out.push_str(&format!("sample_rate={sr}\n"));
            }
            if let Some(ch) = t.channels {
                out.push_str(&format!("channels={ch}\n"));
            }
            if let Some(ref sf) = t.sample_format {
                out.push_str(&format!("sample_fmt={sf}\n"));
            }
            out.push_str("[/STREAM]\n");
        }
    }
    if show_format {
        let size_bytes = std::fs::metadata(&rep.input).map(|m| m.len()).unwrap_or(0);
        let fmt_name = match rep.format_name.as_str() {
            "mp4" => "mov,mp4,m4a,3gp,3g2,mj2",
            "matroska" => "matroska,webm",
            other => other,
        };
        out.push_str("[FORMAT]\n");
        out.push_str(&format!("filename={}\n", rep.input));
        out.push_str(&format!("nb_streams={}\n", rep.tracks.len()));
        out.push_str(&format!("format_name={fmt_name}\n"));
        if let Some(d) = rep.duration_sec {
            out.push_str(&format!("duration={d:.6}\n"));
        }
        out.push_str(&format!("size={size_bytes}\n"));
        out.push_str("[/FORMAT]\n");
    }
    out
}

fn format_ffprobe_json(
    rep: &InspectReport,
    show_streams: bool,
    show_format: bool,
    show_chapters: bool,
    select_streams: Option<&str>,
) -> JsonValue {
    let mut root = Vec::new();

    if show_streams {
        let mut streams = Vec::new();
        for t in &rep.tracks {
            if let Some(sel) = select_streams {
                let want_video = sel == "v" || sel.starts_with("v:");
                let want_audio = sel == "a" || sel.starts_with("a:");
                if want_video && t.media_type != "video" {
                    continue;
                }
                if want_audio && t.media_type != "audio" {
                    continue;
                }
            }
            let mut obj = vec![
                ("index".into(), JsonValue::Int(i64::from(t.id))),
                ("codec_name".into(), JsonValue::Str(t.codec.clone())),
                ("codec_type".into(), JsonValue::Str(t.media_type.clone())),
                ("time_base".into(), JsonValue::Str(t.time_base.clone())),
            ];
            if let Some(w) = t.width {
                obj.push(("width".into(), JsonValue::Int(i64::from(w))));
            }
            if let Some(h) = t.height {
                obj.push(("height".into(), JsonValue::Int(i64::from(h))));
            }
            if let Some(ref pix) = t.pixel_layout {
                obj.push(("pix_fmt".into(), JsonValue::Str(pix.clone())));
            }
            if let Some(ref fps) = t.avg_frame_rate {
                obj.push(("r_frame_rate".into(), JsonValue::Str(fps.clone())));
                obj.push(("avg_frame_rate".into(), JsonValue::Str(fps.clone())));
            }
            if let Some(sr) = t.sample_rate {
                obj.push(("sample_rate".into(), JsonValue::Str(sr.to_string())));
            }
            if let Some(ch) = t.channels {
                obj.push(("channels".into(), JsonValue::Int(i64::from(ch))));
            }
            if let Some(ref layout) = t.channel_layout {
                obj.push(("channel_layout".into(), JsonValue::Str(layout.clone())));
            }
            if let Some(ref sf) = t.sample_format {
                obj.push(("sample_fmt".into(), JsonValue::Str(sf.clone())));
            }
            if let Some(d) = rep.duration_sec {
                obj.push(("duration".into(), JsonValue::Str(format!("{d:.6}"))));
            }
            let mut tags = Vec::new();
            if let Some(ref lang) = t.language {
                tags.push(("language".into(), JsonValue::Str(lang.clone())));
            }
            for (k, v) in &t.metadata {
                tags.push((k.clone(), JsonValue::Str(v.clone())));
            }
            if !tags.is_empty() {
                obj.push(("tags".into(), JsonValue::Obj(tags)));
            }
            streams.push(JsonValue::Obj(obj));
        }
        root.push(("streams".into(), JsonValue::Arr(streams)));
    }

    if show_chapters {
        let chaps = rep
            .chapters
            .iter()
            .enumerate()
            .map(|(idx, c)| {
                JsonValue::Obj(vec![
                    ("id".into(), JsonValue::Int(idx as i64)),
                    (
                        "start_time".into(),
                        JsonValue::Str(format!("{:.6}", c.start_sec.unwrap_or(0.0))),
                    ),
                    (
                        "end_time".into(),
                        JsonValue::Str(format!("{:.6}", c.end_sec.unwrap_or(0.0))),
                    ),
                    (
                        "tags".into(),
                        JsonValue::Obj(vec![(
                            "title".into(),
                            JsonValue::Str(c.title.clone().unwrap_or_default()),
                        )]),
                    ),
                ])
            })
            .collect();
        root.push(("chapters".into(), JsonValue::Arr(chaps)));
    }

    if show_format {
        let size_bytes = std::fs::metadata(&rep.input).map(|m| m.len()).unwrap_or(0);
        let fmt_name = match rep.format_name.as_str() {
            "mp4" => "mov,mp4,m4a,3gp,3g2,mj2",
            "matroska" => "matroska,webm",
            other => other,
        };
        let mut fmt_obj = vec![
            ("filename".into(), JsonValue::Str(rep.input.clone())),
            ("nb_streams".into(), JsonValue::Int(rep.tracks.len() as i64)),
            ("format_name".into(), JsonValue::Str(fmt_name.to_string())),
            ("size".into(), JsonValue::Str(size_bytes.to_string())),
        ];
        if let Some(d) = rep.duration_sec {
            fmt_obj.push(("duration".into(), JsonValue::Str(format!("{d:.6}"))));
        }
        if !rep.metadata.is_empty() {
            let tags = rep
                .metadata
                .iter()
                .map(|(k, v)| (k.clone(), JsonValue::Str(v.clone())))
                .collect();
            fmt_obj.push(("tags".into(), JsonValue::Obj(tags)));
        }
        root.push(("format".into(), JsonValue::Obj(fmt_obj)));
    }

    JsonValue::Obj(root)
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

struct FfMetadata {
    tags: Vec<(String, String)>,
    chapters: Vec<Chapter>,
}

/// Parse an FFmpeg `;FFMETADATA1` sidecar file (used by `yt-dlp --embed-chapters`
/// and `--embed-metadata`). Returns `None` if `path` is not an FFMETADATA1 file.
fn try_parse_ffmetadata1(path: &str) -> Option<FfMetadata> {
    let text = std::fs::read_to_string(path).ok()?;
    let first_line = text.lines().next()?.trim();
    if !first_line.starts_with(";FFMETADATA") {
        return None;
    }

    let mut global_tags = Vec::new();
    let mut chapters = Vec::new();
    let mut in_chapter = false;
    let mut tb = Rational::new(1, 1000);
    let mut start: Option<i64> = None;
    let mut end: Option<i64> = None;
    let mut chap_meta = Metadata::new();

    let flush_chapter = |chapters: &mut Vec<Chapter>,
                         tb: Rational,
                         start: &mut Option<i64>,
                         end: &mut Option<i64>,
                         chap_meta: &mut Metadata| {
        if let (Some(s), Some(e)) = (start.take(), end.take()) {
            let id = chapters.len() as i64 + 1;
            chapters.push(Chapter {
                id,
                start: Timestamp::new(s),
                end: Timestamp::new(e),
                time_base: tb,
                metadata: std::mem::take(chap_meta),
            });
        } else {
            *chap_meta = Metadata::new();
        }
    };

    for raw in text.lines().skip(1) {
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if line.eq_ignore_ascii_case("[CHAPTER]") {
            if in_chapter {
                flush_chapter(&mut chapters, tb, &mut start, &mut end, &mut chap_meta);
            }
            in_chapter = true;
            tb = Rational::new(1, 1000);
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            if in_chapter {
                flush_chapter(&mut chapters, tb, &mut start, &mut end, &mut chap_meta);
            }
            in_chapter = false;
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim();
        let val = unescape_ffmetadata(v);
        if in_chapter {
            match key {
                "TIMEBASE" => {
                    if let Some((n, d)) = val.split_once('/')
                        && let (Ok(num), Ok(den)) = (n.parse::<i32>(), d.parse::<i32>())
                        && den > 0
                    {
                        tb = Rational::new(num, den);
                    }
                }
                "START" => start = val.parse::<i64>().ok(),
                "END" => end = val.parse::<i64>().ok(),
                _ => chap_meta.set(key, &val),
            }
        } else {
            global_tags.push((key.to_string(), val));
        }
    }
    if in_chapter {
        flush_chapter(&mut chapters, tb, &mut start, &mut end, &mut chap_meta);
    }

    Some(FfMetadata {
        tags: global_tags,
        chapters,
    })
}

fn unescape_ffmetadata(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                match next {
                    'n' => out.push('\n'),
                    other => out.push(other),
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}
