//! FFmpeg-compatible `framecrc` generator for Wedeo v2.
//!
//! Uses `wedeo::default_registry()` (zero `inventory` globals) and
//! `wedeo_core::bitstream::adler32_ffmpeg` (init `s1=0, s2=0`):
//! - Video streams are decoded and hashed over packed active display pixels
//!   across all [`PixelLayout`] variants (`YUV420P`, `YUV420P10LE`, `YUV422P`,
//!   `YUV444P`, `NV12`, `P010`, `GBRP`, `RGB24`, `RGBA8`, `BGRA8`, `GRAY8`, ...)
//!   using exact `div_ceil(2)` odd-dimension chroma and [`CropRect`].
//! - Audio/subtitle/data streams hash raw packet payloads in passthrough mode
//!   (matching FFmpeg `-c copy -f framecrc` / PCM WAV `framecrc`).

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, Read, SeekFrom, Write};
use std::path::Path;

use wedeo::core::{
    BufferPool, ByteSource, CodecContext, CompOrder, CropRect, Decoder, Demuxer, ErrorPolicy,
    FileSource, MediaType, MediaUnit, PacketFlags, PixelLayout, PlanePacking, Rational, RecvStatus,
    Registry, SendStatus, StreamParams, Subsampling, Timestamp, TrackId, VideoFrame,
    adler32_ffmpeg,
};

const PROBE_BYTES: usize = 8192;

struct PrefixedSource {
    prefix: Vec<u8>,
    pos: usize,
    inner: Box<dyn ByteSource>,
}

impl Read for PrefixedSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos < self.prefix.len() {
            let avail = self.prefix.len() - self.pos;
            let n = avail.min(buf.len());
            buf[..n].copy_from_slice(&self.prefix[self.pos..self.pos + n]);
            self.pos += n;
            return Ok(n);
        }
        self.inner.read(buf)
    }
}

impl ByteSource for PrefixedSource {
    fn is_seekable(&self) -> bool {
        false
    }

    fn size_hint(&self) -> Option<u64> {
        self.inner.size_hint()
    }
}

fn open_demuxer(
    reg: &Registry,
    path: &Path,
    policy: ErrorPolicy,
) -> Result<Box<dyn Demuxer>, String> {
    let ext = path.extension().and_then(|e| e.to_str());
    let mut source: Box<dyn ByteSource> =
        Box::new(FileSource::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);

    let mut head = Vec::with_capacity(PROBE_BYTES);
    let mut buf = [0u8; 4096];
    while head.len() < PROBE_BYTES {
        let want = (PROBE_BYTES - head.len()).min(buf.len());
        match source.read(&mut buf[..want]) {
            Ok(0) => break,
            Ok(n) => head.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("read {}: {e}", path.display())),
        }
    }

    let factory = reg
        .probe_demuxer(&head, ext)
        .ok_or_else(|| format!("no registered demuxer matched {}", path.display()))?;

    let rewound: Box<dyn ByteSource> = if source.is_seekable() {
        source
            .seek(SeekFrom::Start(0))
            .map_err(|e| format!("seek {}: {e}", path.display()))?;
        source
    } else {
        Box::new(PrefixedSource {
            prefix: head,
            pos: 0,
            inner: source,
        })
    };

    factory
        .open(rewound, policy)
        .map_err(|e| format!("demuxer open {}: {e}", path.display()))
}

fn plane_crop_offset(layout: PixelLayout, crop: CropRect, plane_idx: usize) -> (usize, usize) {
    let bps = layout.depth.bytes_per_sample();
    let (sub_x, sub_y) = match layout.subsampling {
        Subsampling::Cs444 => (1usize, 1usize),
        Subsampling::Cs422 => (2, 1),
        Subsampling::Cs420 => (2, 2),
        Subsampling::Cs411 => (4, 1),
        Subsampling::Cs410 => (4, 4),
        Subsampling::Cs400 => (1, 1),
    };
    match layout.packing {
        PlanePacking::Packed => {
            let x_px = crop.left as usize;
            let x_bytes = match layout.order {
                CompOrder::Yuyv | CompOrder::Uyvy => (x_px / 2) * 4 * bps,
                CompOrder::Rgb | CompOrder::Bgr => x_px * 3 * bps,
                CompOrder::Rgba | CompOrder::Bgra | CompOrder::Argb | CompOrder::Abgr => {
                    x_px * 4 * bps
                }
                CompOrder::Gray => x_px * bps,
                _ => x_px * 3 * bps,
            };
            (x_bytes, crop.top as usize)
        }
        PlanePacking::SemiPlanar => {
            if plane_idx == 0 || plane_idx == 2 {
                (crop.left as usize * bps, crop.top as usize)
            } else {
                (
                    (crop.left as usize / sub_x) * 2 * bps,
                    crop.top as usize / sub_y,
                )
            }
        }
        PlanePacking::Planar => {
            if plane_idx == 0 || plane_idx == 3 {
                (crop.left as usize * bps, crop.top as usize)
            } else {
                (
                    (crop.left as usize / sub_x) * bps,
                    crop.top as usize / sub_y,
                )
            }
        }
    }
}

/// Extract tightly-packed raw pixel bytes for `frame` across all supported
/// [`PixelLayout`]s, honoring `div_ceil` odd chroma and [`CropRect`].
fn pack_video_frame_bytes(frame: &VideoFrame) -> Result<Vec<u8>, String> {
    let planes = frame.planes().ok_or_else(|| {
        "GPU VideoFrame cannot be hashed directly without CPU download".to_string()
    })?;
    let layout = frame.layout;
    let plane_count = layout.plane_count();
    if planes.len() < plane_count {
        return Err(format!(
            "plane count mismatch for {layout}: expected {plane_count}, got {}",
            planes.len()
        ));
    }

    let (disp_w, disp_h) = frame.display_size();
    let mut total_bytes = 0usize;
    for idx in 0..plane_count {
        let (row_bytes, rows) = layout
            .plane_dims(disp_w, disp_h, idx)
            .ok_or_else(|| format!("missing plane_dims({disp_w}, {disp_h}, {idx}) for {layout}"))?;
        total_bytes = total_bytes.saturating_add(row_bytes * rows);
    }

    let mut raw = Vec::with_capacity(total_bytes);
    for (idx, plane) in planes.iter().take(plane_count).enumerate() {
        let (row_bytes, rows) = layout
            .plane_dims(disp_w, disp_h, idx)
            .ok_or_else(|| format!("missing plane_dims({disp_w}, {disp_h}, {idx})"))?;
        let (x_off, y_off) = plane_crop_offset(layout, frame.crop, idx);

        for r in 0..rows {
            let row = plane.row(y_off + r).ok_or_else(|| {
                format!(
                    "plane {idx} row {} out of bounds (height={})",
                    y_off + r,
                    plane.height
                )
            })?;
            let end = x_off + row_bytes;
            if end > row.len() {
                return Err(format!(
                    "plane {idx} slice [{x_off}..{end}] exceeds row width {}",
                    row.len()
                ));
            }
            raw.extend_from_slice(&row[x_off..end]);
        }
    }

    Ok(raw)
}

#[inline]
fn ts_i64(ts: Timestamp) -> i64 {
    ts.get().unwrap_or(i64::MIN)
}

#[derive(Clone)]
struct StreamHeaderInfo {
    tb: Rational,
    media_type: MediaType,
    codec_name: String,
    dimensions: Option<(u32, u32)>,
    sar: Rational,
    sample_rate: u32,
    channel_layout_name: String,
    resolved: bool,
}

fn format_stream_headers(headers: &[StreamHeaderInfo]) -> Vec<String> {
    let mut lines = Vec::new();
    for (i, h) in headers.iter().enumerate() {
        lines.push(format!("#tb {i}: {}/{}", h.tb.num, h.tb.den));
        lines.push(format!("#media_type {i}: {}", h.media_type));
        lines.push(format!("#codec_id {i}: {}", h.codec_name));
        match h.media_type {
            MediaType::Video => {
                let (w, ht) = h.dimensions.unwrap_or((0, 0));
                lines.push(format!("#dimensions {i}: {w}x{ht}"));
                lines.push(format!("#sar {i}: {}/{}", h.sar.num, h.sar.den));
            }
            MediaType::Audio => {
                lines.push(format!("#sample_rate {i}: {}", h.sample_rate));
                lines.push(format!(
                    "#channel_layout_name {i}: {}",
                    h.channel_layout_name
                ));
            }
            _ => {}
        }
    }
    lines
}

fn write_line_or_exit(out: &mut impl Write, line: &str) {
    if let Err(e) = writeln!(out, "{line}") {
        if e.kind() == io::ErrorKind::BrokenPipe {
            std::process::exit(0);
        }
        eprintln!("Error writing output: {e}");
        std::process::exit(1);
    }
}

fn flush_headers_if_ready(
    headers: &[StreamHeaderInfo],
    headers_emitted: &mut bool,
    buffered_lines: &mut Vec<String>,
    out: &mut impl Write,
) {
    if *headers_emitted {
        return;
    }
    if headers.iter().all(|h| h.resolved) {
        for hline in format_stream_headers(headers) {
            write_line_or_exit(out, &hline);
        }
        for line in buffered_lines.drain(..) {
            write_line_or_exit(out, &line);
        }
        *headers_emitted = true;
    }
}

fn emit_line(
    line: String,
    headers_emitted: bool,
    buffered_lines: &mut Vec<String>,
    out: &mut impl Write,
) {
    if headers_emitted {
        write_line_or_exit(out, &line);
    } else {
        buffered_lines.push(line);
    }
}

#[allow(clippy::too_many_arguments)]
fn drain_video_decoder(
    decoder: &mut dyn Decoder,
    si: usize,
    headers: &mut [StreamHeaderInfo],
    headers_emitted: &mut bool,
    buffered_lines: &mut Vec<String>,
    out: &mut impl Write,
    raw_yuv_file: &mut Option<File>,
) -> Result<(), String> {
    loop {
        match decoder
            .receive()
            .map_err(|e| format!("receive stream {si}: {e}"))?
        {
            RecvStatus::Ready(MediaUnit::Video(frame)) => {
                if !headers[si].resolved {
                    let (dw, dh) = frame.display_size();
                    headers[si].dimensions = Some((dw, dh));
                    if frame.sar.num > 0 && frame.sar.den > 0 {
                        headers[si].sar = frame.sar;
                    }
                    if headers[si].tb.den == 0 && frame.time_base.den > 0 {
                        headers[si].tb = frame.time_base;
                    }
                    headers[si].resolved = true;
                    flush_headers_if_ready(headers, headers_emitted, buffered_lines, out);
                }

                let raw = pack_video_frame_bytes(&frame)?;
                if let Some(f) = raw_yuv_file.as_mut() {
                    f.write_all(&raw)
                        .map_err(|e| format!("write raw yuv: {e}"))?;
                }

                let crc = adler32_ffmpeg(0, &raw);
                let size = raw.len();
                let pts = frame.pts.get().unwrap_or(0);
                let dts = frame.pkt_dts.get().unwrap_or(pts);
                let dur = if frame.duration > 0 {
                    frame.duration
                } else {
                    1
                };

                let line = format!("{si}, {dts:>10}, {pts:>10}, {dur:>8}, {size:>8}, 0x{crc:08x}");
                emit_line(line, *headers_emitted, buffered_lines, out);
            }
            RecvStatus::Ready(_) => {}
            RecvStatus::NeedMore | RecvStatus::Eof => break,
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: wedeo-framecrc <input-file> [--raw-yuv <output.yuv>]");
        std::process::exit(1);
    }

    let raw_yuv_path = if args.len() >= 4 && args[2] == "--raw-yuv" {
        Some(args[3].clone())
    } else {
        None
    };
    let mut raw_yuv_file: Option<File> = raw_yuv_path.as_ref().map(|p| {
        File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating {p}: {e}");
            std::process::exit(1);
        })
    });

    let path = Path::new(&args[1]);
    let reg = wedeo::default_registry();
    let mut demuxer = match open_demuxer(&reg, path, ErrorPolicy::Resilient) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error opening {}: {e}", path.display());
            std::process::exit(1);
        }
    };

    let tracks = demuxer.header().tracks.clone();
    let mut track_to_si: HashMap<TrackId, usize> = HashMap::new();
    let mut decoders: Vec<Option<Box<dyn Decoder>>> = Vec::with_capacity(tracks.len());
    let mut headers: Vec<StreamHeaderInfo> = Vec::with_capacity(tracks.len());
    let pool = BufferPool::new();

    for (si, track) in tracks.iter().enumerate() {
        track_to_si.insert(track.id, si);
        match &track.params {
            StreamParams::Video(vp) => {
                let disp = (vp.width, vp.height);
                let resolved = disp.0 > 0 && disp.1 > 0 && track.time_base.den > 0;
                let factory = reg.find_decoder(track.codec_id).unwrap_or_else(|| {
                    eprintln!("Error: no decoder for {:?}", track.codec_id);
                    std::process::exit(1);
                });
                let mut ctx = CodecContext::new(pool.clone());
                ctx.error_policy = ErrorPolicy::Resilient;
                let dec = factory.open(track, ctx).unwrap_or_else(|e| {
                    eprintln!("Error opening decoder for stream {si}: {e}");
                    std::process::exit(1);
                });
                decoders.push(Some(dec));
                headers.push(StreamHeaderInfo {
                    tb: track.time_base,
                    media_type: MediaType::Video,
                    codec_name: "rawvideo".to_string(),
                    dimensions: if disp.0 > 0 && disp.1 > 0 {
                        Some(disp)
                    } else {
                        None
                    },
                    sar: vp.sar,
                    sample_rate: 0,
                    channel_layout_name: String::new(),
                    resolved,
                });
            }
            StreamParams::Audio(ap) => {
                decoders.push(None);
                headers.push(StreamHeaderInfo {
                    tb: track.time_base,
                    media_type: MediaType::Audio,
                    codec_name: track.codec_id.name().to_string(),
                    dimensions: None,
                    sar: Rational::new(0, 1),
                    sample_rate: ap.sample_rate,
                    channel_layout_name: ap.channel_layout.to_string(),
                    resolved: true,
                });
            }
            other => {
                decoders.push(None);
                headers.push(StreamHeaderInfo {
                    tb: track.time_base,
                    media_type: other.media_type(),
                    codec_name: track.codec_id.name().to_string(),
                    dimensions: None,
                    sar: Rational::new(0, 1),
                    sample_rate: 0,
                    channel_layout_name: String::new(),
                    resolved: true,
                });
            }
        }
    }

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let mut headers_emitted = false;
    let mut buffered_lines = Vec::new();
    flush_headers_if_ready(
        &headers,
        &mut headers_emitted,
        &mut buffered_lines,
        &mut out,
    );

    loop {
        match demuxer.read_packet() {
            Ok(RecvStatus::Ready(pkt)) => {
                let Some(&si) = track_to_si.get(&pkt.track_id) else {
                    continue;
                };
                if let Some(dec) = decoders[si].as_mut() {
                    let mut pending = Some(pkt);
                    while let Some(p) = pending.take() {
                        match dec.send_packet(Some(&p)) {
                            Ok(SendStatus::Accepted) => {}
                            Ok(SendStatus::Full) => {
                                pending = Some(p);
                            }
                            Err(e) => {
                                eprintln!("Error decoding packet on stream {si}: {e}");
                                std::process::exit(1);
                            }
                        }
                        if let Err(e) = drain_video_decoder(
                            dec.as_mut(),
                            si,
                            &mut headers,
                            &mut headers_emitted,
                            &mut buffered_lines,
                            &mut out,
                            &mut raw_yuv_file,
                        ) {
                            eprintln!("{e}");
                            std::process::exit(1);
                        }
                    }
                } else {
                    let data = pkt.data.data();
                    let crc = adler32_ffmpeg(0, data);
                    let size = data.len();
                    let dts = if pkt.dts.is_known() {
                        ts_i64(pkt.dts)
                    } else {
                        ts_i64(pkt.pts)
                    };
                    let pts = ts_i64(pkt.pts);
                    let mut line = format!(
                        "{si}, {dts:>10}, {pts:>10}, {:>8}, {size:>8}, 0x{crc:08x}",
                        pkt.duration
                    );
                    if pkt.flags != PacketFlags::KEY {
                        line.push_str(&format!(", F=0x{:X}", pkt.flags.bits()));
                    }
                    emit_line(line, headers_emitted, &mut buffered_lines, &mut out);
                }
            }
            Ok(RecvStatus::NeedMore) => continue,
            Ok(RecvStatus::Eof) => {
                for (si, dec_opt) in decoders.iter_mut().enumerate() {
                    if let Some(dec) = dec_opt {
                        let _ = dec.send_packet(None);
                        if let Err(e) = drain_video_decoder(
                            dec.as_mut(),
                            si,
                            &mut headers,
                            &mut headers_emitted,
                            &mut buffered_lines,
                            &mut out,
                            &mut raw_yuv_file,
                        ) {
                            eprintln!("{e}");
                            std::process::exit(1);
                        }
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

    if !headers_emitted {
        for h in &mut headers {
            h.resolved = true;
        }
        flush_headers_if_ready(
            &headers,
            &mut headers_emitted,
            &mut buffered_lines,
            &mut out,
        );
    }
    if let Err(e) = out.flush()
        && e.kind() != io::ErrorKind::BrokenPipe
    {
        eprintln!("Error flushing output: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo::core::{BorderPadding, BorderedPlane};

    fn make_solid_frame(
        pool: &std::sync::Arc<BufferPool>,
        width: u32,
        height: u32,
        layout: PixelLayout,
        fill_seed: u8,
    ) -> VideoFrame {
        let mut planes = Vec::new();
        for idx in 0..layout.plane_count() {
            let (wb, h) = layout.plane_dims(width, height, idx).unwrap();
            let mut plane = BorderedPlane::new(pool, wb, h, BorderPadding::uniform(8));
            for r in 0..h {
                let val = fill_seed.wrapping_add(idx as u8 * 31).wrapping_add(r as u8);
                plane.row_mut(r).unwrap().fill(val);
            }
            planes.push(plane);
        }
        VideoFrame::from_planes(width, height, layout, planes)
    }

    #[test]
    fn pack_video_frame_supports_all_pixel_layouts_and_odd_dims() {
        let pool = BufferPool::new();
        let layouts = [
            PixelLayout::YUV420P,
            PixelLayout::YUV420P10LE,
            PixelLayout::YUV422P,
            PixelLayout::YUV444P,
            PixelLayout::NV12,
            PixelLayout::P010,
            PixelLayout::GBRP,
            PixelLayout::RGB24,
            PixelLayout::RGBA8,
            PixelLayout::BGRA8,
            PixelLayout::GRAY8,
        ];

        for (i, layout) in layouts.iter().enumerate() {
            let frame = make_solid_frame(&pool, 101, 75, *layout, (i as u8) + 1);
            let bytes = pack_video_frame_bytes(&frame).unwrap();
            let expected_len: usize = (0..layout.plane_count())
                .map(|p| {
                    let (wb, h) = layout.plane_dims(101, 75, p).unwrap();
                    wb * h
                })
                .sum();
            assert_eq!(bytes.len(), expected_len, "layout {layout}");
            assert_ne!(adler32_ffmpeg(0, &bytes), 0);
        }
    }

    #[test]
    fn pack_video_frame_honors_crop_rect() {
        let pool = BufferPool::new();
        // Coded 144x112 cropped to 130x98 (bottom=14, right=14)
        let mut frame = make_solid_frame(&pool, 144, 112, PixelLayout::YUV420P, 7);
        frame.crop = CropRect {
            top: 0,
            bottom: 14,
            left: 0,
            right: 14,
        };
        let bytes = pack_video_frame_bytes(&frame).unwrap();
        assert_eq!(bytes.len(), 130 * 98 + 65 * 49 * 2);
    }
}
