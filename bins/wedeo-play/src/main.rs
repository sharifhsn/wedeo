use std::env;
use std::thread;
use std::time::Duration;

// Ensure inventory registrations are linked in.
use wedeo_codec_h264 as _;
use wedeo_codec_pcm as _;
use wedeo_format_h264 as _;
use wedeo_format_wav as _;
use wedeo_symphonia as _;

use minifb::{Key, Window, WindowOptions};
use wedeo_codec::decoder::DecoderBuilder;
use wedeo_core::error::Error;
use wedeo_core::frame::Frame;
use wedeo_core::pixel_format::PixelFormat;
use wedeo_format::context::InputContext;
use wedeo_scale::Converter;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut scale: usize = 3;
    let mut file = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scale" => {
                i += 1;
                scale = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("--scale requires a positive integer");
                    std::process::exit(1);
                });
                if scale == 0 {
                    scale = 1;
                }
            }
            s if s.starts_with('-') => {
                eprintln!("Unknown option: {s}");
                std::process::exit(1);
            }
            _ => {
                file = Some(args[i].clone());
            }
        }
        i += 1;
    }

    let file = file.unwrap_or_else(|| {
        eprintln!("Usage: wedeo-play <input-file> [--scale N]");
        std::process::exit(1);
    });

    if let Err(e) = run(&file, scale) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run(path: &str, scale: usize) -> wedeo_core::Result<()> {
    let mut ctx = InputContext::open(path)?;

    // Find first video stream.
    let (stream_idx, width, height, time_base) = {
        let stream = ctx
            .streams
            .iter()
            .find(|s| s.codec_params.media_type == wedeo_core::MediaType::Video)
            .ok_or(Error::StreamNotFound)?;
        (
            stream.index,
            stream.codec_params.width as usize,
            stream.codec_params.height as usize,
            stream.time_base,
        )
    };

    let params = ctx.streams[stream_idx].codec_params.clone();
    let mut decoder = DecoderBuilder::new(params).open()?;

    // YUV420p → RGBA converter.
    let converter = Converter::new(
        PixelFormat::Yuv420p,
        PixelFormat::Rgba,
        width as u32,
        height as u32,
    )?;

    let win_w = width * scale;
    let win_h = height * scale;

    let mut window = Window::new(
        &format!("wedeo-play — {path}"),
        win_w,
        win_h,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .map_err(|e| Error::Other(format!("failed to create window: {e}")))?;

    // Default frame duration: 40ms (25fps) if timing info is unavailable.
    let default_duration = Duration::from_millis(40);

    let mut frame_count = 0u64;
    let mut buf = vec![0u32; win_w * win_h];

    let mut display_frame =
        |frame: &Frame, window: &mut Window, buf: &mut Vec<u32>| -> wedeo_core::Result<bool> {
            if !window.is_open() || window.is_key_down(Key::Escape) {
                return Ok(false);
            }

            let rgba_frame = converter.convert(frame)?;
            let video = rgba_frame.video().ok_or(Error::InvalidData)?;
            let rgba_data = video.planes[0].buffer.data();

            // Convert RGBA bytes → 0x00RRGGBB u32 and scale.
            rgba_to_minifb(rgba_data, width, height, buf, scale);

            window
                .update_with_buffer(buf, win_w, win_h)
                .map_err(|e| Error::Other(format!("window update failed: {e}")))?;

            // Frame pacing.
            let duration = if frame.duration > 0 && time_base.num > 0 && time_base.den > 0 {
                let secs = frame.duration as f64 * time_base.num as f64 / time_base.den as f64;
                Duration::from_secs_f64(secs)
            } else {
                default_duration
            };
            thread::sleep(duration);

            frame_count += 1;
            Ok(true)
        };

    // Decode loop.
    'outer: loop {
        match ctx.read_packet() {
            Ok(packet) => {
                decoder.send_packet(Some(&packet))?;
                loop {
                    match decoder.receive_frame() {
                        Ok(frame) => {
                            if !display_frame(&frame, &mut window, &mut buf)? {
                                break 'outer;
                            }
                        }
                        Err(Error::Again) => break,
                        Err(e) => return Err(e),
                    }
                }
            }
            Err(Error::Eof) => break,
            Err(e) => return Err(e),
        }
    }

    // Drain decoder.
    decoder.send_packet(None)?;
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                if !display_frame(&frame, &mut window, &mut buf)? {
                    break;
                }
            }
            Err(Error::Eof | Error::Again) => break,
            Err(e) => return Err(e),
        }
    }

    eprintln!("Played {frame_count} frames");

    // Keep window open until user closes it.
    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update();
        thread::sleep(Duration::from_millis(16));
    }

    Ok(())
}

/// Convert RGBA byte buffer to minifb's `0x00RRGGBB` u32 format with nearest-neighbor scaling.
fn rgba_to_minifb(rgba: &[u8], w: usize, h: usize, buf: &mut [u32], scale: usize) {
    let win_w = w * scale;
    for dy in 0..h * scale {
        let sy = dy / scale;
        for dx in 0..win_w {
            let sx = dx / scale;
            let src_idx = (sy * w + sx) * 4;
            let r = rgba[src_idx] as u32;
            let g = rgba[src_idx + 1] as u32;
            let b = rgba[src_idx + 2] as u32;
            buf[dy * win_w + dx] = (r << 16) | (g << 8) | b;
        }
    }
}
