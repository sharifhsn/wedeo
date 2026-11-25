//! Pure-Rust WebAssembly entrypoint for Wedeo.
//!
//! Exposes a zero-dependency C ABI (`extern "C"`) so browsers can drive
//! Wedeo directly via `WebAssembly.instantiateStreaming` with zero Emscripten,
//! zero `wasm-bindgen`, and zero COOP/COEP `SharedArrayBuffer` requirements.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::{self, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use wedeo::audio_ops::{AudioContract, AudioOpList, CompiledAudioOps};
use wedeo::core::io::BufByteSource;
use wedeo::formats::mkv::{EbmlElement, parse_ebml_tree};
use wedeo::formats::mp4::boxes::{Mp4BoxNode, inspect_boxes};
use wedeo::video_ops::{
    CompiledSws, CpuSwsBackend, SwsBackend, SwsOpList, SwsOptions, VideoContract,
};
use wedeo::{
    AudioBlock, AudioStreamParams, BufferPool, ByteSink, ChannelLayout, CodecContext, CodecId,
    ColorDesc, ContainerHeader, Decoder, Demuxer, EncodeHint, Error, ErrorPolicy, MediaType,
    MediaUnit, MemorySink, MemorySource, PassMode, PixelLayout, Rational, RecvStatus, Registry,
    Result, SampleFormat, SendStatus, StreamParams, TrackDescriptor, TrackId, VideoFrame,
};

// ---------------------------------------------------------------------------
// Thread-local WASM state (single-threaded linear memory)
// ---------------------------------------------------------------------------

struct VideoSession {
    demuxer: Box<dyn Demuxer>,
    decoder: Box<dyn Decoder>,
    video_track: TrackId,
    pool: Arc<BufferPool>,
    sws: Option<(VideoContract, Box<dyn CompiledSws>, usize)>,
    pending: VecDeque<VideoFrame>,
    flushed: bool,
    frame_index: u32,
}

#[derive(Default)]
struct WasmState {
    json_out: Vec<u8>,
    bytes_out: Vec<u8>,
    error_out: Vec<u8>,
    video: Option<VideoSession>,
}

thread_local! {
    static STATE: RefCell<WasmState> = RefCell::new(WasmState::default());
}

fn set_error(msg: impl AsRef<str>) {
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        st.error_out.clear();
        st.error_out.extend_from_slice(msg.as_ref().as_bytes());
    });
}

fn set_outputs(json: String, bytes: Vec<u8>) {
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        st.error_out.clear();
        st.json_out = json.into_bytes();
        st.bytes_out = bytes;
    });
}

// ---------------------------------------------------------------------------
// Minimal zero-dependency JSON escaping & builder helpers
// ---------------------------------------------------------------------------

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn ext_from_name(name: &str) -> Option<&str> {
    let dot = name.rfind('.')?;
    let ext = &name[dot + 1..];
    if ext.is_empty() { None } else { Some(ext) }
}

fn wasm_codec_ctx(pool: Arc<BufferPool>) -> CodecContext {
    CodecContext {
        pool,
        // Single-threaded WASM linear memory: run decoders inline without spawning OS threads.
        threads: 1,
        error_policy: ErrorPolicy::Resilient,
    }
}

fn open_demuxer_bytes(
    reg: &Registry,
    data: Vec<u8>,
    ext: Option<&str>,
) -> Result<Box<dyn Demuxer>> {
    let head_len = data.len().min(4096);
    let factory = reg.probe_demuxer(&data[..head_len], ext).ok_or_else(|| {
        Error::unsupported("wasm/probe", "unrecognized media container or stream")
    })?;
    factory.open(Box::new(MemorySource::new(data)), ErrorPolicy::Resilient)
}

// ---------------------------------------------------------------------------
// Shared memory sink for recovering muxed bytes after Muxer::finalize
// ---------------------------------------------------------------------------

struct SharedMemorySink(Arc<Mutex<MemorySink>>);

impl Write for SharedMemorySink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("poisoned sink lock"))?
            .write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("poisoned sink lock"))?
            .flush()
    }
}

impl ByteSink for SharedMemorySink {
    fn is_seekable(&self) -> bool {
        true
    }

    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("poisoned sink lock"))?
            .seek(pos)
    }
}

// ---------------------------------------------------------------------------
// Container AST serialization
// ---------------------------------------------------------------------------

struct AstItem {
    kind: String,
    offset: u64,
    size: u64,
    summary: Option<String>,
    children: Vec<AstItem>,
}

impl AstItem {
    fn from_mp4(b: &Mp4BoxNode) -> Self {
        Self {
            kind: b.type_name(),
            offset: b.offset,
            size: b.size,
            summary: b.summary.clone(),
            children: b.children.iter().map(Self::from_mp4).collect(),
        }
    }

    fn from_ebml(e: &EbmlElement) -> Self {
        let summary = if e.size == u64::MAX {
            Some(format!("id=0x{:X} size=unknown", e.id))
        } else {
            Some(format!("id=0x{:X}", e.id))
        };
        Self {
            kind: e.name.to_string(),
            offset: e.header_offset,
            size: e.size,
            summary,
            children: e.children.iter().map(Self::from_ebml).collect(),
        }
    }

    fn to_json(&self, budget: &mut usize) -> String {
        let mut s = String::new();
        s.push_str("{\"kind\":");
        s.push_str(&json_str(&self.kind));
        s.push_str(&format!(
            ",\"offset\":{},\"size\":{}",
            self.offset, self.size
        ));
        if let Some(ref sum) = self.summary {
            s.push_str(",\"summary\":");
            s.push_str(&json_str(sum));
        }
        s.push_str(",\"children\":[");
        let mut first = true;
        for c in &self.children {
            if *budget == 0 {
                break;
            }
            *budget -= 1;
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str(&c.to_json(budget));
        }
        s.push_str("]}");
        s
    }
}

fn inspect_ast_bytes(data: &[u8], format_name: &str) -> Vec<AstItem> {
    match format_name {
        "mp4" | "mov" => {
            let mut io = BufByteSource::new(Box::new(MemorySource::new(data.to_vec())));
            inspect_boxes(&mut io)
                .map(|b| b.iter().map(AstItem::from_mp4).collect())
                .unwrap_or_default()
        }
        "matroska" | "mkv" | "webm" => {
            let mut io = BufByteSource::new(Box::new(MemorySource::new(data.to_vec())));
            parse_ebml_tree(&mut io, 4)
                .map(|e| e.iter().map(AstItem::from_ebml).collect())
                .unwrap_or_default()
        }
        "wav" => inspect_wav_ast(data),
        "ivf" => inspect_ivf_ast(data),
        _ => Vec::new(),
    }
}

fn inspect_wav_ast(data: &[u8]) -> Vec<AstItem> {
    if data.len() < 12 {
        return Vec::new();
    }
    let magic = &data[0..4];
    if !matches!(magic, b"RIFF" | b"RIFX" | b"RF64" | b"BW64") {
        return Vec::new();
    }
    let be = magic == b"RIFX";
    let read_u32 = |b: &[u8]| {
        let a = [b[0], b[1], b[2], b[3]];
        if be {
            u32::from_be_bytes(a)
        } else {
            u32::from_le_bytes(a)
        }
    };
    let form = String::from_utf8_lossy(&data[8..12]).to_string();
    let mut children = Vec::new();
    let mut pos = 12usize;
    while pos + 8 <= data.len() && children.len() < 64 {
        let id = String::from_utf8_lossy(&data[pos..pos + 4]).to_string();
        let sz = read_u32(&data[pos + 4..pos + 8]) as u64;
        children.push(AstItem {
            kind: id,
            offset: pos as u64,
            size: sz,
            summary: None,
            children: Vec::new(),
        });
        let step = 8usize
            .saturating_add(sz as usize)
            .saturating_add((sz & 1) as usize);
        if step == 0 {
            break;
        }
        pos = pos.saturating_add(step);
    }
    vec![AstItem {
        kind: String::from_utf8_lossy(magic).to_string(),
        offset: 0,
        size: data.len() as u64,
        summary: Some(format!("form={form}")),
        children,
    }]
}

fn inspect_ivf_ast(data: &[u8]) -> Vec<AstItem> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return Vec::new();
    }
    let fourcc = String::from_utf8_lossy(&data[8..12]).to_string();
    let w = u16::from_le_bytes([data[12], data[13]]);
    let h = u16::from_le_bytes([data[14], data[15]]);
    let frames = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    vec![AstItem {
        kind: "DKIF".to_string(),
        offset: 0,
        size: 32,
        summary: Some(format!("codec={fourcc} {w}x{h} frames={frames}")),
        children: Vec::new(),
    }]
}

// ---------------------------------------------------------------------------
// Core Rust implementations (also directly unit-testable on native targets)
// ---------------------------------------------------------------------------

pub fn inspect_bytes(data: Vec<u8>, name: &str) -> Result<String> {
    let reg = wedeo::default_registry();
    let ext = ext_from_name(name);
    let ast_preview_data = data.clone();
    let demuxer = open_demuxer_bytes(&reg, data, ext)?;
    let hdr = demuxer.header();
    let ast = inspect_ast_bytes(&ast_preview_data, hdr.format_name);

    let duration_sec = hdr.duration.map(|d| {
        if hdr.time_base.den > 0 {
            d as f64 * hdr.time_base.num as f64 / hdr.time_base.den as f64
        } else {
            0.0
        }
    });

    let mut out = String::new();
    out.push('{');
    out.push_str(&format!("\"format\":{},", json_str(hdr.format_name)));
    out.push_str(&format!("\"bytes\":{},", ast_preview_data.len()));
    match duration_sec {
        Some(d) => out.push_str(&format!("\"duration_sec\":{d:.4},")),
        None => out.push_str("\"duration_sec\":null,"),
    }
    out.push_str(&format!(
        "\"time_base\":{},",
        json_str(&format!("{}/{}", hdr.time_base.num, hdr.time_base.den))
    ));

    // Tracks
    out.push_str("\"tracks\":[");
    for (i, t) in hdr.tracks.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        out.push_str(&format!("\"id\":{},", t.id.0));
        out.push_str(&format!(
            "\"media_type\":{},",
            json_str(&format!("{:?}", t.params.media_type()).to_lowercase())
        ));
        out.push_str(&format!("\"codec\":{},", json_str(t.codec_id.name())));
        out.push_str(&format!(
            "\"time_base\":{},",
            json_str(&format!("{}/{}", t.time_base.num, t.time_base.den))
        ));
        let extradata_len = match &t.params {
            StreamParams::Video(v) => v.extradata.len(),
            StreamParams::Audio(a) => a.extradata.len(),
            _ => 0,
        };
        out.push_str(&format!("\"extradata_bytes\":{extradata_len},"));
        let decodable = reg.find_decoder(t.codec_id).is_some();
        out.push_str(&format!("\"decodable\":{decodable},"));
        match &t.params {
            StreamParams::Video(v) => {
                out.push_str(&format!("\"width\":{},\"height\":{},", v.width, v.height));
                out.push_str(&format!(
                    "\"pixel_layout\":{},",
                    json_str(&format!("{}", v.layout))
                ));
                out.push_str(&format!(
                    "\"color\":{},",
                    json_str(&format!(
                        "{:?}/{:?}/{:?} ({:?})",
                        v.color.primaries, v.color.transfer, v.color.matrix, v.color.range
                    ))
                ));
                if v.avg_frame_rate.den > 0 && v.avg_frame_rate.num > 0 {
                    let f = v.avg_frame_rate.num as f64 / v.avg_frame_rate.den as f64;
                    out.push_str(&format!("\"fps\":{f:.3},"));
                }
            }
            StreamParams::Audio(a) => {
                out.push_str(&format!(
                    "\"sample_rate\":{},\"channels\":{},\"sample_format\":{},",
                    a.sample_rate,
                    a.channel_layout.nb_channels,
                    json_str(&format!("{}", a.format))
                ));
            }
            _ => {}
        }
        if let Some(lang) = &t.language {
            out.push_str(&format!("\"language\":{},", json_str(lang)));
        }
        if out.ends_with(',') {
            out.pop();
        }
        out.push('}');
    }
    out.push_str("],");

    // Container AST (capped to 200 nodes for crisp UI rendering)
    out.push_str("\"ast\":[");
    let mut budget = 200usize;
    for (i, node) in ast.iter().enumerate() {
        if budget == 0 {
            break;
        }
        budget -= 1;
        if i > 0 {
            out.push(',');
        }
        out.push_str(&node.to_json(&mut budget));
    }
    out.push_str("],");

    // Health
    let h = demuxer.health();
    out.push_str(&format!(
        "\"health\":{{\"is_clean\":{},\"corrupt_packets\":{},\"timestamp_glitches\":{}}}",
        h.is_clean(),
        h.corrupt_packets,
        h.timestamp_glitches
    ));
    out.push('}');
    Ok(out)
}

fn open_video_session(data: Vec<u8>, name: &str) -> Result<(VideoSession, String)> {
    let reg = wedeo::default_registry();
    let ext = ext_from_name(name);
    let demuxer = open_demuxer_bytes(&reg, data, ext)?;
    let hdr = demuxer.header().clone();

    let track = hdr
        .tracks
        .iter()
        .find(|t| t.params.media_type() == MediaType::Video)
        .ok_or_else(|| Error::unsupported("wasm/video", "no video track found in input"))?
        .clone();

    let dec_factory = reg.find_decoder(track.codec_id).ok_or_else(|| {
        Error::unsupported(
            "wasm/video",
            format!(
                "no WASM decoder registered for codec '{}'",
                track.codec_id.name()
            ),
        )
    })?;

    let pool = BufferPool::new();
    let decoder = dec_factory.open(&track, wasm_codec_ctx(Arc::clone(&pool)))?;

    let (w, h, fps, pix, color) = match &track.params {
        StreamParams::Video(v) => {
            let fps = if v.avg_frame_rate.den > 0 && v.avg_frame_rate.num > 0 {
                v.avg_frame_rate.num as f64 / v.avg_frame_rate.den as f64
            } else {
                24.0
            };
            (
                v.width,
                v.height,
                fps,
                format!("{}", v.layout),
                format!(
                    "{:?}/{:?}/{:?}",
                    v.color.primaries, v.color.transfer, v.color.matrix
                ),
            )
        }
        _ => (0, 0, 24.0, "unknown".to_string(), "unknown".to_string()),
    };

    let info = format!(
        "{{\"format\":{},\"codec\":{},\"width\":{w},\"height\":{h},\"fps\":{fps:.3},\"pixel_layout\":{},\"color\":{}}}",
        json_str(hdr.format_name),
        json_str(track.codec_id.name()),
        json_str(&pix),
        json_str(&color),
    );

    let session = VideoSession {
        demuxer,
        decoder,
        video_track: track.id,
        pool,
        sws: None,
        pending: VecDeque::new(),
        flushed: false,
        frame_index: 0,
    };

    Ok((session, info))
}

fn pull_next_video_frame(session: &mut VideoSession) -> Result<Option<(String, Vec<u8>)>> {
    loop {
        if let Some(frame) = session.pending.pop_front() {
            return Ok(Some(convert_frame_to_rgba(session, &frame)?));
        }

        // Drain any already-buffered frames from the decoder.
        loop {
            match session.decoder.receive()? {
                RecvStatus::Ready(MediaUnit::Video(f)) => session.pending.push_back(f),
                RecvStatus::Ready(_) => {}
                RecvStatus::NeedMore | RecvStatus::Eof => break,
            }
        }
        if let Some(frame) = session.pending.pop_front() {
            return Ok(Some(convert_frame_to_rgba(session, &frame)?));
        }

        if session.flushed {
            return Ok(None);
        }

        // Feed packets until the decoder produces at least one frame or hits EOF.
        loop {
            match session.demuxer.read_packet()? {
                RecvStatus::Ready(pkt) => {
                    if pkt.track_id != session.video_track {
                        continue;
                    }
                    let mut pkt_opt = Some(pkt);
                    while let Some(p) = pkt_opt.take() {
                        match session.decoder.send_packet(Some(&p))? {
                            SendStatus::Accepted => {}
                            SendStatus::Full => {
                                while let RecvStatus::Ready(u) = session.decoder.receive()? {
                                    if let MediaUnit::Video(f) = u {
                                        session.pending.push_back(f);
                                    }
                                }
                                pkt_opt = Some(p);
                            }
                        }
                    }
                    break;
                }
                RecvStatus::Eof | RecvStatus::NeedMore => {
                    let _ = session.decoder.send_packet(None)?;
                    session.flushed = true;
                    break;
                }
            }
        }
    }
}

fn convert_frame_to_rgba(
    session: &mut VideoSession,
    frame: &VideoFrame,
) -> Result<(String, Vec<u8>)> {
    let src_contract = VideoContract::new(frame.width, frame.height, frame.layout)
        .with_color(frame.color)
        .with_crop(frame.crop);
    let need_rebuild = session
        .sws
        .as_ref()
        .is_none_or(|(cached, _, _)| *cached != src_contract);

    if need_rebuild {
        let dst_contract = VideoContract::new(frame.width, frame.height, PixelLayout::RGBA8)
            .with_color(ColorDesc::SRGB);
        let ops = SwsOpList::build(&src_contract, &dst_contract, SwsOptions::default())?;
        let op_count = ops.ops.len();
        let compiled = CpuSwsBackend::new().compile(&ops)?;
        session.sws = Some((src_contract, compiled, op_count));
    }

    let (_, compiled, op_count) = session.sws.as_mut().expect("sws initialized");
    let rgba_frame = compiled.execute(frame, &session.pool)?;

    let w = rgba_frame.width as usize;
    let h = rgba_frame.height as usize;
    let mut packed = Vec::with_capacity(w * h * 4);
    let planes = rgba_frame
        .planes()
        .ok_or_else(|| Error::internal("wasm/rgba", "expected CPU planes from Sws"))?;
    let p0 = &planes[0];
    for row in 0..h {
        let slice = p0
            .row(row)
            .ok_or_else(|| Error::internal("wasm/rgba", "RGBA plane row out of bounds"))?;
        packed.extend_from_slice(&slice[..w * 4]);
    }

    let pts_sec = frame.pts.get().map(|v| {
        if frame.time_base.den > 0 {
            v as f64 * frame.time_base.num as f64 / frame.time_base.den as f64
        } else {
            0.0
        }
    });
    let idx = session.frame_index;
    session.frame_index += 1;

    let pts_json = match pts_sec {
        Some(s) => format!("{s:.4}"),
        None => "null".to_string(),
    };
    let meta = format!(
        "{{\"index\":{idx},\"width\":{w},\"height\":{h},\"pts_sec\":{pts_json},\"pict_type\":{},\"src_layout\":{},\"sws_ops\":{op_count}}}",
        json_str(&format!("{:?}", frame.picture_type)),
        json_str(&format!("{}", frame.layout)),
    );

    Ok((meta, packed))
}

fn decode_audio_internal(
    data: Vec<u8>,
    name: &str,
    target_rate: u32,
    target_channels: u16,
    to_wav_s16: bool,
    selected: Option<u32>,
) -> Result<(String, Vec<u8>)> {
    let reg = wedeo::default_registry();
    let ext = ext_from_name(name);
    let mut demuxer = open_demuxer_bytes(&reg, data, ext)?;
    let hdr = demuxer.header().clone();

    let track = hdr
        .tracks
        .iter()
        .find(|t| {
            t.params.media_type() == MediaType::Audio && selected.is_none_or(|id| t.id.0 == id)
        })
        .ok_or_else(|| Error::unsupported("wasm/audio", "no audio track found in input"))?
        .clone();

    let dec_factory = reg.find_decoder(track.codec_id).ok_or_else(|| {
        Error::unsupported(
            "wasm/audio",
            format!("no decoder for audio codec '{}'", track.codec_id.name()),
        )
    })?;

    let pool = BufferPool::new();
    let mut decoder = dec_factory.open(&track, wasm_codec_ctx(Arc::clone(&pool)))?;

    let mut raw_blocks: Vec<AudioBlock> = Vec::new();
    loop {
        match demuxer.read_packet()? {
            RecvStatus::Ready(pkt) => {
                if pkt.track_id != track.id {
                    continue;
                }
                let _ = decoder.send_packet(Some(&pkt))?;
                while let RecvStatus::Ready(unit) = decoder.receive()? {
                    if let MediaUnit::Audio(b) = unit {
                        raw_blocks.push(b);
                    }
                }
            }
            RecvStatus::Eof => {
                let _ = decoder.send_packet(None)?;
                while let RecvStatus::Ready(unit) = decoder.receive()? {
                    if let MediaUnit::Audio(b) = unit {
                        raw_blocks.push(b);
                    }
                }
                break;
            }
            RecvStatus::NeedMore => break,
        }
    }

    if raw_blocks.is_empty() {
        return Err(Error::invalid_data(
            "wasm/audio",
            "audio track decoded zero samples",
        ));
    }

    let first = &raw_blocks[0];
    let start_sec = first.pts.to_seconds(first.time_base).unwrap_or_default()
        + clip_shift(&hdr, track.id).unwrap_or_default();
    let src_rate = first.sample_rate;
    let src_channels = first.channel_layout.nb_channels;
    let out_rate = if target_rate > 0 {
        target_rate
    } else {
        src_rate
    };
    let out_ch: u32 = if target_channels > 0 {
        u32::from(target_channels)
    } else {
        src_channels
    };
    let out_layout = ChannelLayout::default_for_count(out_ch);
    let out_fmt = if to_wav_s16 {
        SampleFormat::S16
    } else {
        SampleFormat::F32Planar
    };

    let dst_contract = AudioContract {
        format: out_fmt,
        sample_rate: out_rate,
        channel_layout: out_layout.clone(),
        frame_size: None,
    };

    let mut current_src = AudioContract {
        format: first.format,
        sample_rate: first.sample_rate,
        channel_layout: first.channel_layout.clone(),
        frame_size: None,
    };
    let mut ops = AudioOpList::build(current_src.clone(), dst_contract.clone())?;
    let mut compiled = CompiledAudioOps::compile(&ops)?;
    let mut converted: Vec<AudioBlock> = Vec::new();

    for b in raw_blocks {
        if b.format != current_src.format
            || b.sample_rate != current_src.sample_rate
            || b.channel_layout != current_src.channel_layout
        {
            current_src = AudioContract {
                format: b.format,
                sample_rate: b.sample_rate,
                channel_layout: b.channel_layout.clone(),
                frame_size: None,
            };
            ops = AudioOpList::build(current_src.clone(), dst_contract.clone())?;
            compiled = CompiledAudioOps::compile(&ops)?;
        }
        converted.extend(compiled.process(b, &pool)?);
    }
    converted.extend(compiled.flush(&pool)?);

    let total_frames: usize = converted.iter().map(|b| b.nb_samples as usize).sum();
    let duration_sec = total_frames as f64 / out_rate.max(1) as f64;

    if to_wav_s16 {
        let enc_factory = reg
            .find_encoder(CodecId::PcmS16le)
            .ok_or_else(|| Error::internal("wasm/wav", "pcm_s16le encoder missing"))?;
        let mut out_track = TrackDescriptor::new(
            TrackId(0),
            CodecId::PcmS16le,
            Rational::new(1, out_rate as i32),
            StreamParams::Audio(AudioStreamParams::new(
                out_rate,
                SampleFormat::S16,
                out_layout.clone(),
            )),
        );
        let mut encoder = enc_factory.open(
            &out_track,
            wasm_codec_ctx(Arc::clone(&pool)),
            PassMode::Single,
        )?;
        let extra = encoder.extradata();
        if !extra.is_empty()
            && let StreamParams::Audio(ref mut a) = out_track.params
        {
            a.extradata = extra.to_vec();
        }

        let mux_factory = reg
            .find_muxer_by_name("wav")
            .ok_or_else(|| Error::internal("wasm/wav", "wav muxer missing"))?;
        let shared = Arc::new(Mutex::new(MemorySink::new()));
        let mut muxer = mux_factory.create(Box::new(SharedMemorySink(Arc::clone(&shared))))?;
        let mut out_hdr = ContainerHeader::new("wav");
        out_hdr.time_base = Rational::new(1, out_rate as i32);
        out_hdr.tracks.push(out_track);
        muxer.write_header(&out_hdr)?;

        let mut packets = 0usize;
        for b in converted {
            let _ = encoder.send_unit(Some(&MediaUnit::Audio(b)), EncodeHint::default())?;
            while let RecvStatus::Ready(mut pkt) = encoder.receive_packet()? {
                pkt.track_id = TrackId(0);
                muxer.write_packet(&pkt)?;
                packets += 1;
            }
        }
        let _ = encoder.send_unit(None, EncodeHint::default())?;
        while let RecvStatus::Ready(mut pkt) = encoder.receive_packet()? {
            pkt.track_id = TrackId(0);
            muxer.write_packet(&pkt)?;
            packets += 1;
        }
        muxer.finalize()?;

        let wav_bytes = Arc::try_unwrap(shared)
            .map_err(|_| Error::internal("wasm/wav", "sink still referenced"))?
            .into_inner()
            .map_err(|_| Error::internal("wasm/wav", "sink lock poisoned"))?
            .into_inner();

        let json = format!(
            "{{\"format\":\"wav\",\"codec\":\"pcm_s16le\",\"src_rate\":{src_rate},\"src_channels\":{src_channels},\"sample_rate\":{out_rate},\"channels\":{out_ch},\"frames_per_channel\":{total_frames},\"duration_sec\":{duration_sec:.4},\"packets\":{packets},\"bytes\":{}}}",
            wav_bytes.len()
        );
        Ok((json, wav_bytes))
    } else {
        // Pack planar f32: [ch0 samples (N * 4 bytes), ch1 samples (N * 4 bytes), ...]
        let ch_count = out_ch as usize;
        let mut channels_out: Vec<Vec<u8>> = (0..ch_count)
            .map(|_| Vec::with_capacity(total_frames * 4))
            .collect();
        for b in &converted {
            let nbytes = b.nb_samples as usize * 4;
            for (c, dst) in channels_out.iter_mut().enumerate() {
                if let Some(plane) = b.planes.get(c).and_then(|p| p.as_slice()) {
                    let take = nbytes.min(plane.len());
                    dst.extend_from_slice(&plane[..take]);
                }
            }
        }
        let mut flat = Vec::with_capacity(total_frames * ch_count * 4);
        for c in channels_out {
            flat.extend_from_slice(&c);
        }

        let json = format!(
            "{{\"start_sec\":{start_sec},\"codec\":{},\"src_rate\":{src_rate},\"src_channels\":{src_channels},\"sample_rate\":{out_rate},\"channels\":{out_ch},\"frames_per_channel\":{total_frames},\"duration_sec\":{duration_sec:.4},\"bytes\":{}}}",
            json_str(track.codec_id.name()),
            flat.len()
        );
        Ok((json, flat))
    }
}

/// Decode a movie-time interval directly into a zero-filled native-rate planar buffer.
/// Gaps and delayed tracks stay silent. Only the current decoder block is unpacked.
fn audition_window(
    data: Vec<u8>,
    name: &str,
    id: u32,
    start: f64,
    end: f64,
) -> Result<(String, Vec<u8>)> {
    if !(start.is_finite() && end.is_finite() && start < end) {
        return Err(Error::invalid_argument(
            "audition",
            "invalid preview interval",
        ));
    }
    let reg = wedeo::default_registry();
    let mut demux = open_demuxer_bytes(&reg, data, ext_from_name(name))?;
    let track = demux
        .header()
        .track(TrackId(id))
        .ok_or_else(|| Error::invalid_argument("audition", "unknown track"))?
        .clone();
    let StreamParams::Audio(params) = &track.params else {
        return Err(Error::invalid_argument(
            "audition",
            "expected an audio track",
        ));
    };
    let rate = params.sample_rate;
    let channels = params.channel_layout.nb_channels as usize;
    if rate == 0 || channels == 0 {
        return Err(Error::unsupported("audition", "missing audio format"));
    }
    let shift = clip_shift(demux.header(), track.id)?;
    let pool = BufferPool::new();
    let mut dec = reg
        .find_decoder(track.codec_id)
        .ok_or_else(|| Error::unsupported("audition", "no audio decoder"))?
        .open(&track, wasm_codec_ctx(pool))?;
    let origin = (start * f64::from(rate)).round() as i64;
    let frames = ((end - start) * f64::from(rate)).round() as usize;
    if frames == 0 {
        return Err(Error::invalid_argument(
            "audition",
            "selection shorter than one sample",
        ));
    }
    let size = frames
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::invalid_argument("audition", "preview too large"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(size)
        .map_err(|_| Error::unsupported("audition", "interval exceeds available memory"))?;
    output.resize(size, 0);

    // One second covers AAC overlap, Opus preroll and the MP3 reservoir at common rates.
    // Containers without a seek index can scan compressed packets to the same point.
    let preroll = (start - shift - 1.0).max(0.0);
    if preroll > 0.0 {
        let target = wedeo::SeekTarget {
            track: Some(track.id),
            timestamp: wedeo::Timestamp::new((preroll * 1e6).floor() as i64),
            time_base: Rational::new(1, 1_000_000),
            backward: true,
        };
        if let Err(e) = demux.seek(target)
            && !matches!(
                e.kind,
                wedeo::ErrorKind::NotSeekable | wedeo::ErrorKind::Unsupported
            )
        {
            return Err(e);
        }
    }
    let mut slab = Vec::new();
    let mut decoded_packets = 0usize;
    loop {
        let packet = match demux.read_packet()? {
            RecvStatus::Ready(pkt) => {
                if pkt.track_id != track.id {
                    continue;
                }
                let ts = pkt
                    .pts
                    .to_seconds(pkt.time_base)
                    .ok_or_else(|| Error::unsupported("audition", "missing packet timestamp"))?;
                if ts + shift >= end {
                    None
                } else if ts
                    + wedeo::Timestamp::new(pkt.duration.max(0))
                        .to_seconds(pkt.time_base)
                        .unwrap_or_default()
                    < preroll
                {
                    continue;
                } else {
                    Some(pkt)
                }
            }
            RecvStatus::NeedMore => {
                return Err(Error::invalid_data("audition", "incomplete input"));
            }
            RecvStatus::Eof => None,
        };
        if dec.send_packet(packet.as_ref())? != SendStatus::Accepted {
            return Err(Error::internal(
                "audition",
                "audio decoder did not accept packet after draining",
            ));
        }
        decoded_packets += usize::from(packet.is_some());
        while let RecvStatus::Ready(unit) = dec.receive()? {
            let MediaUnit::Audio(block) = unit else {
                continue;
            };
            if block.sample_rate != rate || block.channel_layout.nb_channels as usize != channels {
                return Err(Error::unsupported(
                    "audition",
                    "audio format changed within the selection",
                ));
            }
            let ts = block
                .pts
                .to_seconds(block.time_base)
                .ok_or_else(|| Error::unsupported("audition", "missing audio timestamp"))?
                + shift;
            let offset = ((ts * f64::from(rate)).round() as i64).saturating_sub(origin);
            let src = offset.saturating_neg().max(0) as usize;
            let dst = offset.max(0) as usize;
            let take = (block.nb_samples as usize)
                .saturating_sub(src)
                .min(frames.saturating_sub(dst));
            if take == 0 {
                continue;
            }
            slab.resize(take * channels, 0.0);
            wedeo::audio_ops::dither::unpack_slab_f32(
                &block.planes,
                block.format,
                channels,
                src,
                take,
                &mut slab,
                take,
            )?;
            for (channel, samples) in slab.chunks_exact(take).enumerate() {
                let pos = (channel * frames + dst) * 4;
                for (slot, sample) in output[pos..pos + take * 4]
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(samples)
                {
                    *slot = sample.to_le_bytes();
                }
            }
        }
        if packet.is_none() {
            break;
        }
    }
    Ok((
        format!(
            r#"{{"codec":{},"sample_rate":{rate},"channels":{channels},"frames_per_channel":{frames},"start_sec":{start},"duration_sec":{},"decoded_packets":{decoded_packets},"bytes":{size}}}"#,
            json_str(track.codec_id.name()),
            frames as f64 / f64::from(rate)
        ),
        output,
    ))
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_audition_window(
    ptr: *const u8,
    len: usize,
    np: *const u8,
    nl: usize,
    id: u32,
    start: f64,
    end: f64,
) -> i32 {
    wedeo_release_output_bytes();
    clip_result(audition_window(
        read_input_bytes(ptr, len),
        &read_input_str(np, nl),
        id,
        start,
        end,
    ))
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_release_output_bytes() {
    STATE.with(|state| state.borrow_mut().bytes_out = Vec::new());
}

pub fn remux_bytes(data: Vec<u8>, name: &str, target_fmt: &str) -> Result<(String, Vec<u8>)> {
    write_container(data, name, target_fmt, None)
}

fn write_container(
    data: Vec<u8>,
    name: &str,
    target_fmt: &str,
    range: Option<(f64, f64)>,
) -> Result<(String, Vec<u8>)> {
    let reg = wedeo::default_registry();
    let ext = ext_from_name(name);
    let mut demuxer = open_demuxer_bytes(&reg, data, ext)?;
    let hdr = demuxer.header().clone();

    let mux_factory = reg
        .find_muxer_by_name(target_fmt)
        .or_else(|| reg.find_muxer_by_ext(target_fmt))
        .ok_or_else(|| {
            Error::unsupported(
                "wasm/remux",
                format!("unknown output format '{target_fmt}'"),
            )
        })?;

    let shared = Arc::new(Mutex::new(MemorySink::new()));
    let mut muxer = mux_factory.create(Box::new(SharedMemorySink(Arc::clone(&shared))))?;
    let mut out_hdr = hdr.clone();
    out_hdr.format_name = mux_factory.descriptor().name;
    if range.is_some() {
        out_hdr.duration = None;
        out_hdr.chapters.clear();
        out_hdr.start_time = wedeo::Timestamp::new(0);
    }
    muxer.write_header(&out_hdr)?;

    let mut clip_index = range.map(|_| ClipIndex::new(&hdr)).transpose()?;
    let mut packet_count = 0usize;
    while let RecvStatus::Ready(mut pkt) = demuxer.read_packet()? {
        if let (Some((start, end)), Some(index)) = (range, &mut clip_index) {
            let pts = pkt
                .pts
                .to_seconds(pkt.time_base)
                .ok_or_else(|| Error::unsupported("clip", "missing packet timestamp"))?;
            let shift = clip_shift(&hdr, pkt.track_id)?;
            index.observe(&pkt, pts + shift)?;
            if pts + shift < start || pts + shift >= end {
                continue;
            }
            let offset = ((start - shift) * f64::from(pkt.time_base.den)
                / f64::from(pkt.time_base.num))
            .round() as i64;
            pkt.pts = pkt
                .pts
                .checked_sub(offset)
                .unwrap_or(wedeo::Timestamp::NONE);
            pkt.dts = pkt
                .dts
                .checked_sub(offset)
                .unwrap_or(wedeo::Timestamp::NONE);
        }
        muxer.write_packet(&pkt)?;
        packet_count += 1;
    }
    if let (Some((start, end)), Some(index)) = (range, clip_index) {
        let marks = index.finish()?;
        if !marks.contains(&start) || !marks.contains(&end) {
            return Err(Error::unsupported(
                "clip",
                "choose independently decodable clip boundaries",
            ));
        }
    }
    muxer.finalize()?;

    let out_bytes = Arc::try_unwrap(shared)
        .map_err(|_| Error::internal("wasm/remux", "sink still referenced"))?
        .into_inner()
        .map_err(|_| Error::internal("wasm/remux", "sink lock poisoned"))?
        .into_inner();

    let json = format!(
        "{{\"src_format\":{},\"dst_format\":{},\"tracks\":{},\"packets\":{packet_count},\"bytes\":{}}}",
        json_str(hdr.format_name),
        json_str(mux_factory.descriptor().name),
        hdr.tracks.len(),
        out_bytes.len()
    );
    Ok((json, out_bytes))
}

// ---------------------------------------------------------------------------
// C ABI Exports for WebAssembly
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn wedeo_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: Caller passes `(ptr, len)` obtained from `wedeo_alloc(len)`.
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_json_ptr() -> *const u8 {
    STATE.with(|s| s.borrow().json_out.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_json_len() -> usize {
    STATE.with(|s| s.borrow().json_out.len())
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_bytes_ptr() -> *const u8 {
    STATE.with(|s| s.borrow().bytes_out.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_bytes_len() -> usize {
    STATE.with(|s| s.borrow().bytes_out.len())
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_error_ptr() -> *const u8 {
    STATE.with(|s| s.borrow().error_out.as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_error_len() -> usize {
    STATE.with(|s| s.borrow().error_out.len())
}

fn read_input_bytes(ptr: *const u8, len: usize) -> Vec<u8> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    // SAFETY: JS writes `len` bytes into linear memory allocated via `wedeo_alloc`.
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

fn read_input_str(ptr: *const u8, len: usize) -> String {
    String::from_utf8_lossy(&read_input_bytes(ptr, len)).into_owned()
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_inspect(
    ptr: *const u8,
    len: usize,
    name_ptr: *const u8,
    name_len: usize,
) -> i32 {
    let data = read_input_bytes(ptr, len);
    let name = read_input_str(name_ptr, name_len);
    match inspect_bytes(data, &name) {
        Ok(json) => {
            set_outputs(json, Vec::new());
            0
        }
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_video_open(
    ptr: *const u8,
    len: usize,
    name_ptr: *const u8,
    name_len: usize,
) -> i32 {
    let data = read_input_bytes(ptr, len);
    let name = read_input_str(name_ptr, name_len);
    match open_video_session(data, &name) {
        Ok((session, json)) => {
            STATE.with(|s| {
                let mut st = s.borrow_mut();
                st.video = Some(session);
                st.error_out.clear();
                st.json_out = json.into_bytes();
                st.bytes_out.clear();
            });
            0
        }
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

/// Decode next video frame into RGBA bytes (`wedeo_bytes_*`) + JSON metadata (`wedeo_json_*`).
/// Returns `1` if a frame was produced, `0` on EOF, `-1` on error.
#[unsafe(no_mangle)]
pub extern "C" fn wedeo_video_next_frame() -> i32 {
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        let Some(session) = st.video.as_mut() else {
            st.error_out = b"no open video session".to_vec();
            return -1;
        };
        match pull_next_video_frame(session) {
            Ok(Some((meta, rgba))) => {
                st.error_out.clear();
                st.json_out = meta.into_bytes();
                st.bytes_out = rgba;
                1
            }
            Ok(None) => {
                st.error_out.clear();
                st.json_out.clear();
                st.bytes_out.clear();
                0
            }
            Err(e) => {
                st.error_out = e.to_string().into_bytes();
                -1
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_video_close() {
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        st.video = None;
        st.bytes_out.clear();
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_decode_audio(
    ptr: *const u8,
    len: usize,
    name_ptr: *const u8,
    name_len: usize,
    target_rate: u32,
    target_channels: u16,
) -> i32 {
    let data = read_input_bytes(ptr, len);
    let name = read_input_str(name_ptr, name_len);
    match decode_audio_internal(data, &name, target_rate, target_channels, false, None) {
        Ok((json, bytes)) => {
            set_outputs(json, bytes);
            0
        }
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_export_wav(
    ptr: *const u8,
    len: usize,
    name_ptr: *const u8,
    name_len: usize,
    target_rate: u32,
    target_channels: u16,
) -> i32 {
    let data = read_input_bytes(ptr, len);
    let name = read_input_str(name_ptr, name_len);
    match decode_audio_internal(data, &name, target_rate, target_channels, true, None) {
        Ok((json, bytes)) => {
            set_outputs(json, bytes);
            0
        }
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_remux(
    ptr: *const u8,
    len: usize,
    name_ptr: *const u8,
    name_len: usize,
    fmt_ptr: *const u8,
    fmt_len: usize,
) -> i32 {
    let data = read_input_bytes(ptr, len);
    let name = read_input_str(name_ptr, name_len);
    let fmt = read_input_str(fmt_ptr, fmt_len);
    match remux_bytes(data, &name, &fmt) {
        Ok((json, bytes)) => {
            set_outputs(json, bytes);
            0
        }
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

fn clip_result(result: Result<(String, Vec<u8>)>) -> i32 {
    match result {
        Ok((json, bytes)) => {
            set_outputs(json, bytes);
            0
        }
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_audition(
    ptr: *const u8,
    len: usize,
    np: *const u8,
    nl: usize,
    id: u32,
) -> i32 {
    clip_result(decode_audio_internal(
        read_input_bytes(ptr, len),
        &read_input_str(np, nl),
        0,
        0,
        false,
        Some(id),
    ))
}

// Map coded packet timestamps to the movie timeline before comparing tracks.
fn clip_shift(header: &ContainerHeader, id: TrackId) -> Result<f64> {
    let track = header
        .track(id)
        .ok_or_else(|| Error::unsupported("clip", "unknown track"))?;
    let mut shift = 0.0;
    for (index, edit) in track.edit_list.iter().enumerate() {
        if edit.rate != Rational::new(1, 1) {
            return Err(Error::unsupported("clip", "variable-rate edit lists"));
        }
        if let Some(start) = edit.media_time.to_seconds(edit.time_base) {
            if index + 1 != track.edit_list.len() {
                return Err(Error::unsupported("clip", "multiple media edits"));
            }
            shift -= start;
        } else {
            shift += wedeo::Timestamp::new(edit.duration)
                .to_seconds(edit.time_base)
                .unwrap_or_default();
        }
    }
    Ok(shift)
}

struct ClipIndex {
    video: TrackId,
    h264: bool,
    nal_size: Option<u8>,
    marks: Vec<f64>,
    end: f64,
}

impl ClipIndex {
    fn new(header: &ContainerHeader) -> Result<Self> {
        let mut videos = header
            .tracks
            .iter()
            .filter(|t| t.params.media_type() == MediaType::Video);
        let track = videos
            .next()
            .ok_or_else(|| Error::unsupported("clip", "requires a video track"))?;
        if videos.next().is_some() {
            return Err(Error::unsupported(
                "clip",
                "select a recording with one video track",
            ));
        }
        let StreamParams::Video(v) = &track.params else {
            unreachable!()
        };
        let nal_size = if v.extradata.first() == Some(&1) {
            v.extradata.get(4).map(|b| (b & 3) + 1)
        } else {
            None
        };
        Ok(Self {
            video: track.id,
            h264: track.codec_id == CodecId::H264,
            nal_size,
            marks: Vec::new(),
            end: f64::NEG_INFINITY,
        })
    }

    fn observe(&mut self, pkt: &wedeo::Packet, ts: f64) -> Result<()> {
        let duration = wedeo::Timestamp::new(pkt.duration.max(1))
            .to_seconds(pkt.time_base)
            .unwrap_or_default();
        self.end = self.end.max(ts + duration);
        if pkt.track_id != self.video || !pkt.flags.contains(wedeo::core::PacketFlags::KEY) {
            return Ok(());
        }
        if self.h264 {
            let data = pkt.data.as_slice().unwrap_or_default();
            let annexb;
            let payload = if let Some(size) = self.nal_size {
                annexb = wedeo::syntax::h264::avcc_to_annexb(data, size)?;
                &annexb[..]
            } else {
                data
            };
            if !wedeo::syntax::h264::iter_annexb_nals(payload).any(|n| {
                n.is_ok_and(|nal| {
                    nal.header.nal_unit_type == wedeo::syntax::h264::NalUnitType::IdrSlice
                })
            }) {
                return Ok(());
            }
        }
        self.marks.push(ts);
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<f64>> {
        if self.marks.is_empty() {
            return Err(Error::unsupported(
                "clip",
                "no independently decodable video boundary",
            ));
        }
        self.marks.push(self.end);
        if self.marks.windows(2).any(|p| p[0] >= p[1]) {
            return Err(Error::unsupported(
                "clip",
                "non-monotonic keyframe timestamps",
            ));
        }
        Ok(self.marks)
    }
}

fn clip_boundaries(data: Vec<u8>, name: &str) -> Result<Vec<f64>> {
    let mut demux = open_demuxer_bytes(&wedeo::default_registry(), data, ext_from_name(name))?;
    let mut index = ClipIndex::new(demux.header())?;
    while let RecvStatus::Ready(pkt) = demux.read_packet()? {
        let pts = pkt
            .pts
            .to_seconds(pkt.time_base)
            .ok_or_else(|| Error::unsupported("clip", "missing packet timestamp"))?;
        index.observe(&pkt, pts + clip_shift(demux.header(), pkt.track_id)?)?;
    }
    index.finish()
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_clip_marks(ptr: *const u8, len: usize, np: *const u8, nl: usize) -> i32 {
    clip_result(
        clip_boundaries(read_input_bytes(ptr, len), &read_input_str(np, nl))
            .map(|v| (format!("{v:?}"), Vec::new())),
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn wedeo_clip(
    ptr: *const u8,
    len: usize,
    np: *const u8,
    nl: usize,
    start: f64,
    end: f64,
) -> i32 {
    if !(start.is_finite() && end.is_finite() && start < end) {
        set_error("Invalid clip bounds");
        return -1;
    }
    clip_result(write_container(
        read_input_bytes(ptr, len),
        &read_input_str(np, nl),
        "mkv",
        Some((start, end)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_inspect_and_decode_bundled_vp9_ivf() {
        let ivf_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/wedeo-oracle/data/vp9_red_64x64.ivf"
        );
        let bytes = std::fs::read(ivf_path).expect("read bundled vp9 ivf");
        let json = inspect_bytes(bytes.clone(), "vp9_red_64x64.ivf").expect("inspect");
        assert!(json.contains("\"format\":\"ivf\""));
        assert!(json.contains("\"codec\":\"vp9\""));

        let (mut session, open_json) =
            open_video_session(bytes, "vp9_red_64x64.ivf").expect("open");
        assert!(open_json.contains("\"width\":64"));
        let (meta, rgba) = pull_next_video_frame(&mut session)
            .expect("decode")
            .expect("frame 0");
        assert!(meta.contains("\"index\":0"));
        assert_eq!(rgba.len(), 64 * 64 * 4);
    }
}
