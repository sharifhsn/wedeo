//! Stdio JSON-RPC 2.0 MCP server (`wedeo mcp`) exposing the 5 typed ops
//! (`inspect`, `explain`, `transcode`, `cut`, `mux`) with zero duplicated logic.

use std::io::{self, BufRead, Write};

use wedeo::Registry;

use crate::json::JsonValue;
use crate::ops::{
    CutReq, ExplainReq, InspectReq, MuxReq, TranscodeReq, run_cut, run_explain, run_inspect,
    run_mux, run_transcode,
};

const PROTOCOL_VERSION: &str = "2024-11-05";

pub fn serve_stdio(reg: &Registry) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line_res in stdin.lock().lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(resp) = handle_jsonrpc_line(reg, trimmed) {
            out.write_all(resp.as_bytes())?;
            out.write_all(b"\n")?;
            out.flush()?;
        }
    }
    Ok(())
}

/// Process a single JSON-RPC 2.0 request/notification line and return an optional
/// compact JSON-RPC 2.0 response string (`None` for notifications without `id`).
pub fn handle_jsonrpc_line(reg: &Registry, line: &str) -> Option<String> {
    let req = match JsonValue::parse(line) {
        Ok(v) => v,
        Err(e) => {
            return Some(jsonrpc_error(
                JsonValue::Null,
                -32700,
                &format!("Parse error: {e}"),
            ));
        }
    };

    let id_opt = req.get("id").cloned();
    let Some(method) = req.get("method").and_then(JsonValue::as_str) else {
        return id_opt.map(|id| jsonrpc_error(id, -32600, "Invalid Request: missing 'method'"));
    };

    // Notifications (no `id`) receive no response per JSON-RPC 2.0
    let id = id_opt?;

    let params = req.get("params");

    let resp = match method {
        "initialize" => jsonrpc_ok(
            id,
            JsonValue::obj(vec![
                ("protocolVersion", PROTOCOL_VERSION.into()),
                (
                    "capabilities",
                    JsonValue::obj(vec![("tools", JsonValue::obj(vec![]))]),
                ),
                (
                    "serverInfo",
                    JsonValue::obj(vec![
                        ("name", "wedeo".into()),
                        ("version", env!("CARGO_PKG_VERSION").into()),
                    ]),
                ),
            ]),
        ),
        "ping" => jsonrpc_ok(id, JsonValue::obj(vec![])),
        "tools/list" => jsonrpc_ok(id, JsonValue::obj(vec![("tools", tool_definitions())])),
        "tools/call" => handle_tool_call(reg, id, params),
        other => jsonrpc_error(id, -32601, &format!("Method not found: {other}")),
    };

    Some(resp)
}

fn handle_tool_call(reg: &Registry, id: JsonValue, params: Option<&JsonValue>) -> String {
    let Some(p) = params else {
        return jsonrpc_error(id, -32602, "Invalid params: expected object");
    };
    let Some(name) = p.get("name").and_then(JsonValue::as_str) else {
        return jsonrpc_error(id, -32602, "Invalid params: missing tool 'name'");
    };
    let empty_obj = JsonValue::obj(vec![]);
    let args = p.get("arguments").unwrap_or(&empty_obj);

    let result = match name {
        "inspect" => InspectReq::from_json(args)
            .and_then(|req| run_inspect(reg, &req))
            .map(|rep| rep.to_json()),
        "explain" => ExplainReq::from_json(args)
            .and_then(|req| run_explain(reg, &req))
            .map(|plan| plan.to_json()),
        "transcode" => TranscodeReq::from_json(args)
            .and_then(|req| run_transcode(reg, &req))
            .map(|rep| rep.to_json()),
        "cut" => CutReq::from_json(args)
            .and_then(|req| run_cut(reg, &req))
            .map(|rep| rep.to_json()),
        "mux" => MuxReq::from_json(args)
            .and_then(|req| run_mux(reg, &req))
            .map(|rep| rep.to_json()),
        unknown => {
            return jsonrpc_error(id, -32602, &format!("Unknown tool '{unknown}'"));
        }
    };

    match result {
        Ok(text) => jsonrpc_ok(
            id,
            JsonValue::obj(vec![
                (
                    "content",
                    JsonValue::Arr(vec![JsonValue::obj(vec![
                        ("type", "text".into()),
                        ("text", text.into()),
                    ])]),
                ),
                ("isError", false.into()),
            ]),
        ),
        Err(err) => jsonrpc_ok(
            id,
            JsonValue::obj(vec![
                (
                    "content",
                    JsonValue::Arr(vec![JsonValue::obj(vec![
                        ("type", "text".into()),
                        ("text", err.to_string().into()),
                    ])]),
                ),
                ("isError", true.into()),
            ]),
        ),
    }
}

fn tool_definitions() -> JsonValue {
    JsonValue::Arr(vec![
        tool_schema(
            "inspect",
            "Inspect container AST (MP4/MKV/WAV/IVF), tracks, colorimetry, chapters, attachments, packets/GOPs/NALs, and optional full-stream decode StreamHealth verification.",
            vec![
                ("input", schema_prop("string", "Input media file path")),
                (
                    "level",
                    schema_enum(
                        "string",
                        "Detail level",
                        &["container", "packets", "gop", "nal"],
                    ),
                ),
                (
                    "verify",
                    schema_prop("boolean", "Decode full stream and verify StreamHealth"),
                ),
                (
                    "max_packets",
                    schema_prop("integer", "Optional cap on listed packets"),
                ),
            ],
            &["input"],
        ),
        tool_schema(
            "explain",
            "Compile a transcode DAG via wedeo-graph without executing, returning all nodes, edges, negotiated formats, auto-inserted bridges, ConversionNotes, and GPU residency notes.",
            vec![
                ("input", schema_prop("string", "Input media file path")),
                ("output", schema_prop("string", "Target output path")),
                ("format", schema_prop("string", "Optional output format")),
                (
                    "video_codec",
                    schema_prop("string", "Video codec or 'copy'"),
                ),
                (
                    "audio_codec",
                    schema_prop("string", "Audio codec or 'copy'"),
                ),
                ("size", schema_prop("string", "Target video size WxH")),
                (
                    "strict",
                    schema_prop("boolean", "Reject implicit lossy conversions"),
                ),
                (
                    "prefer_gpu",
                    schema_prop("boolean", "Prefer GPU domain placement"),
                ),
            ],
            &["input", "output"],
        ),
        tool_schema(
            "transcode",
            "Transcode or remux a media file via wedeo-graph preserving all tracks, chapters, attachments, HDR metadata, and rotation by default.",
            vec![
                ("input", schema_prop("string", "Input media file path")),
                ("output", schema_prop("string", "Output media file path")),
                ("format", schema_prop("string", "Optional output format")),
                (
                    "video_codec",
                    schema_prop("string", "Video codec (e.g. av1, copy)"),
                ),
                (
                    "audio_codec",
                    schema_prop("string", "Audio codec (e.g. pcm_s16le, copy)"),
                ),
                (
                    "frames",
                    schema_prop("integer", "Maximum video frames to output"),
                ),
                ("no_video", schema_prop("boolean", "Disable video streams")),
                ("no_audio", schema_prop("boolean", "Disable audio streams")),
                ("size", schema_prop("string", "Target video dimensions WxH")),
                (
                    "sample_rate",
                    schema_prop("integer", "Target audio sample rate"),
                ),
                (
                    "channels",
                    schema_prop("integer", "Target audio channel count"),
                ),
                (
                    "strict",
                    schema_prop("boolean", "Fail on corruption or implicit lossy conversion"),
                ),
                (
                    "preserve_all",
                    schema_prop("boolean", "Preserve all tracks/chapters/attachments"),
                ),
                (
                    "overwrite",
                    schema_prop("boolean", "Overwrite existing output file"),
                ),
            ],
            &["input", "output"],
        ),
        tool_schema(
            "cut",
            "Cut a time interval [from_sec, to_sec) using keyframe copy, GOP-aware smart-cut, or full re-encode.",
            vec![
                ("input", schema_prop("string", "Input media file path")),
                ("output", schema_prop("string", "Output media file path")),
                ("from_sec", schema_prop("number", "Start time in seconds")),
                ("to_sec", schema_prop("number", "End time in seconds")),
                (
                    "mode",
                    schema_enum("string", "Cut strategy", &["keyframe", "smart", "reencode"]),
                ),
                ("format", schema_prop("string", "Optional output format")),
                (
                    "overwrite",
                    schema_prop("boolean", "Overwrite existing output file"),
                ),
            ],
            &["input", "output", "from_sec", "to_sec"],
        ),
        tool_schema(
            "mux",
            "Losslessly mux multiple video, audio, or container inputs with chapters, attachments, and tags.",
            vec![
                (
                    "video",
                    schema_str_array("Input files to take video tracks from"),
                ),
                (
                    "audio",
                    schema_str_array("Input files to take audio tracks from"),
                ),
                ("inputs", schema_str_array("Additional container inputs")),
                ("output", schema_prop("string", "Output media file path")),
                ("format", schema_prop("string", "Optional output format")),
                (
                    "overwrite",
                    schema_prop("boolean", "Overwrite existing output file"),
                ),
            ],
            &["output"],
        ),
    ])
}

fn tool_schema(
    name: &str,
    description: &str,
    props: Vec<(&str, JsonValue)>,
    required: &[&str],
) -> JsonValue {
    let req_arr: Vec<JsonValue> = required.iter().map(|s| (*s).into()).collect();
    JsonValue::obj(vec![
        ("name", name.into()),
        ("description", description.into()),
        (
            "inputSchema",
            JsonValue::obj(vec![
                ("type", "object".into()),
                ("properties", JsonValue::obj(props)),
                ("required", JsonValue::Arr(req_arr)),
            ]),
        ),
    ])
}

fn schema_prop(ty: &str, desc: &str) -> JsonValue {
    JsonValue::obj(vec![("type", ty.into()), ("description", desc.into())])
}

fn schema_enum(ty: &str, desc: &str, variants: &[&str]) -> JsonValue {
    let vals: Vec<JsonValue> = variants.iter().map(|v| (*v).into()).collect();
    JsonValue::obj(vec![
        ("type", ty.into()),
        ("description", desc.into()),
        ("enum", JsonValue::Arr(vals)),
    ])
}

fn schema_str_array(desc: &str) -> JsonValue {
    JsonValue::obj(vec![
        ("type", "array".into()),
        ("description", desc.into()),
        ("items", JsonValue::obj(vec![("type", "string".into())])),
    ])
}

fn jsonrpc_ok(id: JsonValue, result: JsonValue) -> String {
    JsonValue::obj(vec![
        ("jsonrpc", "2.0".into()),
        ("id", id),
        ("result", result),
    ])
    .to_compact()
}

fn jsonrpc_error(id: JsonValue, code: i64, message: &str) -> String {
    JsonValue::obj(vec![
        ("jsonrpc", "2.0".into()),
        ("id", id),
        (
            "error",
            JsonValue::obj(vec![("code", code.into()), ("message", message.into())]),
        ),
    ])
    .to_compact()
}
