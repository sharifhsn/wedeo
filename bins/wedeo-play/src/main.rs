use std::collections::VecDeque;
use std::env;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Observer, Producer, Split};

// Ensure inventory registrations are linked in.
use wedeo_codec_h264 as _;
use wedeo_codec_pcm as _;
use wedeo_format_h264 as _;
use wedeo_format_mp4 as _;
use wedeo_format_wav as _;
use wedeo_rav1d as _;
use wedeo_symphonia as _;

use wedeo_codec::decoder::DecoderBuilder;
use wedeo_core::MediaType;
use wedeo_core::error::Error;
use wedeo_core::frame::{Frame, FramePlane};
use wedeo_core::pixel_format::PixelFormat;
use wedeo_core::rational::Rational;
use wedeo_core::sample_format::SampleFormat;
use wedeo_format::context::InputContext;
use wedeo_format::demuxer::SeekFlags;
use wedeo_resample::{Quality, Resampler};
use wedeo_scale::Converter;

/// Decoded video frame ready for display.
struct VideoFrame {
    rgba: Vec<u8>,
    width: usize,
    height: usize,
    /// Presentation time in seconds.
    pts_sec: f64,
}

/// Data sent from the decode thread to the main thread.
enum DecodedData {
    Audio { samples: Vec<f32>, pts_sec: f64 },
    Video(VideoFrame),
    SeekComplete,
    Eof,
}

/// Commands sent from the main thread to the decode thread.
enum Command {
    Seek(f64),
    Quit,
}

/// Arguments for the decode thread, grouped to avoid too_many_arguments.
struct DecodeThreadArgs {
    ctx: InputContext,
    video_idx: Option<usize>,
    audio_idx: Option<usize>,
    video_decoder: Option<Box<dyn wedeo_codec::decoder::Decoder>>,
    audio_decoder: Option<Box<dyn wedeo_codec::decoder::Decoder>>,
    vid_time_base: Rational,
    audio_time_base: Rational,
    src_sample_rate: u32,
    device_sample_rate: u32,
    device_channels: usize,
    data_tx: mpsc::SyncSender<DecodedData>,
    cmd_rx: mpsc::Receiver<Command>,
}

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

fn run(path: &str, initial_scale: usize) -> wedeo_core::Result<()> {
    let ctx = InputContext::open(path)?;

    // Find video and audio streams.
    let video_info: Option<(usize, usize, usize, _, _)> = ctx
        .streams
        .iter()
        .find(|s| s.codec_params.media_type == MediaType::Video)
        .map(|s| {
            (
                s.index,
                s.codec_params.width as usize,
                s.codec_params.height as usize,
                s.time_base,
                s.codec_params.clone(),
            )
        });

    let audio_info: Option<(usize, u32, _, _)> = ctx
        .streams
        .iter()
        .find(|s| s.codec_params.media_type == MediaType::Audio)
        .map(|s| {
            (
                s.index,
                s.codec_params.sample_rate,
                s.time_base,
                s.codec_params.clone(),
            )
        });

    if video_info.is_none() && audio_info.is_none() {
        return Err(Error::StreamNotFound);
    }

    let _has_video = video_info.is_some();
    let has_audio = audio_info.is_some();

    // Video setup.
    let video_idx = video_info.as_ref().map(|v| v.0);
    let (vid_width, vid_height) = video_info
        .as_ref()
        .map(|v| (v.1, v.2))
        .unwrap_or((640, 480));
    let vid_time_base = video_info
        .as_ref()
        .map(|v| v.3)
        .unwrap_or(Rational::new(1, 25));

    let video_decoder = video_info
        .map(|v| DecoderBuilder::new(v.4).open())
        .transpose()?;

    // Audio setup.
    let audio_idx = audio_info.as_ref().map(|a| a.0);
    let src_sample_rate = audio_info.as_ref().map(|a| a.1).unwrap_or(0);
    let audio_time_base = audio_info
        .as_ref()
        .map(|a| a.2)
        .unwrap_or(Rational::new(1, 1));

    let audio_decoder = audio_info
        .map(|a| DecoderBuilder::new(a.3).open())
        .transpose()?;

    // Audio output via cpal.
    let audio_clock = Arc::new(AtomicU64::new(0));
    let volume = Arc::new(AtomicU32::new(f32::to_bits(1.0)));
    let seeking = Arc::new(AtomicBool::new(false));

    let cpal_device = if has_audio {
        let host = cpal::default_host();
        Some(
            host.default_output_device()
                .ok_or(Error::Other("no audio output device".into()))?,
        )
    } else {
        None
    };

    let (device_sample_rate, device_channels) = if has_audio {
        let config = cpal_device
            .as_ref()
            .unwrap()
            .default_output_config()
            .map_err(|e| Error::Other(format!("audio config: {e}")))?;
        let sr = config.sample_rate();
        let ch = config.channels() as usize;
        (sr, ch)
    } else {
        (48000u32, 2usize)
    };

    // Ring buffer: 2 seconds of audio at device rate, or minimal if no audio.
    let rb_capacity = if has_audio {
        device_sample_rate as usize * device_channels * 2
    } else {
        1
    };
    let rb = HeapRb::<f32>::new(rb_capacity);
    let (mut rb_prod, rb_cons) = rb.split();

    // Build cpal output stream with volume and seeking support.
    let audio_stream: Option<cpal::Stream>;
    if has_audio {
        let stream_config = cpal::StreamConfig {
            channels: device_channels as u16,
            sample_rate: device_sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let clock = audio_clock.clone();
        let vol = volume.clone();
        let seek_flag = seeking.clone();
        let mut audio_consumer = rb_cons;

        let stream = cpal_device
            .as_ref()
            .unwrap()
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if seek_flag.load(Ordering::Relaxed) {
                        // Drain ring buffer, output silence, don't advance clock.
                        audio_consumer.pop_slice(data);
                        data.fill(0.0);
                        return;
                    }
                    let gain = f32::from_bits(vol.load(Ordering::Relaxed));
                    let filled = audio_consumer.pop_slice(data);
                    for sample in &mut data[..filled] {
                        *sample *= gain;
                    }
                    for sample in &mut data[filled..] {
                        *sample = 0.0;
                    }
                    clock.fetch_add(filled as u64, Ordering::Relaxed);
                },
                |err| eprintln!("Audio error: {err}"),
                None,
            )
            .map_err(|e| Error::Other(format!("audio stream: {e}")))?;

        audio_stream = Some(stream);
    } else {
        drop(rb_cons);
        audio_stream = None;
    }

    // Window setup — initial size uses --scale, then dynamic.
    let init_win_w = vid_width * initial_scale;
    let init_win_h = vid_height * initial_scale;

    let mut window = Window::new(
        &format!("wedeo-play — {path}"),
        init_win_w,
        init_win_h,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .map_err(|e| Error::Other(format!("failed to create window: {e}")))?;

    let mut win_w = init_win_w;
    let mut win_h = init_win_h;
    let mut buf = vec![0u32; win_w * win_h];

    // Decode thread channels.
    let (data_tx, data_rx) = mpsc::sync_channel::<DecodedData>(32);
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();

    // Spawn decode thread.
    let _decode_handle = thread::spawn(move || {
        decode_thread_fn(DecodeThreadArgs {
            ctx,
            video_idx,
            audio_idx,
            video_decoder,
            audio_decoder,
            vid_time_base,
            audio_time_base,
            src_sample_rate,
            device_sample_rate,
            device_channels,
            data_tx,
            cmd_rx,
        });
    });

    let mut pending_video: VecDeque<VideoFrame> = VecDeque::new();
    let mut pending_audio: Vec<f32> = Vec::new();
    let mut frame_count = 0u64;
    let mut eof = false;
    let mut audio_playing = false;
    let mut first_audio_pts: Option<f64> = None;
    let mut paused = false;

    // Video-only wall clock (deferred start to avoid dropping initial frames).
    let mut playback_start: Option<Instant> = None;
    let mut first_video_pts = 0.0f64;
    let mut pause_offset = Duration::ZERO;
    let mut pause_start: Option<Instant> = None;

    // Volume state.
    let mut current_volume: f32 = 1.0;

    // Main display + input loop.
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Handle window resize.
        let (new_w, new_h) = window.get_size();
        if (new_w != win_w || new_h != win_h) && new_w > 0 && new_h > 0 {
            win_w = new_w;
            win_h = new_h;
            buf.resize(win_w * win_h, 0);
            buf.fill(0);
        }

        // Space: pause/resume.
        if window.is_key_pressed(Key::Space, KeyRepeat::No) {
            paused = !paused;
            if paused {
                if let Some(ref s) = audio_stream {
                    let _ = s.pause();
                }
                pause_start = Some(Instant::now());
            } else {
                if audio_playing && let Some(ref s) = audio_stream {
                    let _ = s.play();
                }
                if let Some(ps) = pause_start.take() {
                    pause_offset += ps.elapsed();
                }
            }
        }

        // Up/Down: volume ±10%.
        if window.is_key_pressed(Key::Up, KeyRepeat::No) {
            current_volume = (current_volume + 0.1).min(2.0);
            volume.store(f32::to_bits(current_volume), Ordering::Relaxed);
        }
        if window.is_key_pressed(Key::Down, KeyRepeat::No) {
            current_volume = (current_volume - 0.1).max(0.0);
            volume.store(f32::to_bits(current_volume), Ordering::Relaxed);
        }

        // Left/Right: seek ±5s.
        let seek_delta = if window.is_key_pressed(Key::Right, KeyRepeat::No) {
            Some(5.0)
        } else if window.is_key_pressed(Key::Left, KeyRepeat::No) {
            Some(-5.0)
        } else {
            None
        };

        if let Some(delta) = seek_delta {
            // Compute current playback time for seek base.
            let current_time = if has_audio && audio_playing {
                let consumed = audio_clock.load(Ordering::Relaxed);
                let clock_secs =
                    consumed as f64 / (device_sample_rate as f64 * device_channels as f64);
                first_audio_pts.unwrap_or(0.0) + clock_secs
            } else if let Some(start) = playback_start {
                let elapsed = start.elapsed().saturating_sub(pause_offset);
                let elapsed = if let Some(ps) = pause_start {
                    elapsed.saturating_sub(ps.elapsed())
                } else {
                    elapsed
                };
                first_video_pts + elapsed.as_secs_f64()
            } else {
                0.0
            };
            let target = (current_time + delta).max(0.0);

            // Signal cpal to output silence and drain ring buffer.
            seeking.store(true, Ordering::Relaxed);

            // Send seek command to decode thread.
            let _ = cmd_tx.send(Command::Seek(target));

            // Drain data channel until SeekComplete.
            loop {
                match data_rx.recv() {
                    Ok(DecodedData::SeekComplete) => break,
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }

            // Reset state.
            pending_video.clear();
            pending_audio.clear();
            audio_clock.store(0, Ordering::Relaxed);
            first_audio_pts = None;
            eof = false;
            audio_playing = false;

            // Reset video-only clock.
            playback_start = None;
            first_video_pts = 0.0;
            pause_offset = Duration::ZERO;
            if paused {
                pause_start = Some(Instant::now());
            }

            seeking.store(false, Ordering::Relaxed);
        }

        // Receive decoded data (skip if paused — backpressures decode thread).
        if !paused {
            // Push pending audio first.
            if !pending_audio.is_empty() {
                let pushed = rb_prod.push_slice(&pending_audio);
                if pushed >= pending_audio.len() {
                    pending_audio.clear();
                } else {
                    pending_audio.drain(..pushed);
                }
            }

            for _ in 0..50 {
                if has_audio && rb_prod.vacant_len() < device_channels * 4096 {
                    break;
                }

                match data_rx.try_recv() {
                    Ok(DecodedData::Audio { samples, pts_sec }) => {
                        if first_audio_pts.is_none() {
                            first_audio_pts = Some(pts_sec);
                        }
                        let pushed = rb_prod.push_slice(&samples);
                        if pushed < samples.len() {
                            pending_audio.extend_from_slice(&samples[pushed..]);
                            break;
                        }
                    }
                    Ok(DecodedData::Video(frame)) => {
                        pending_video.push_back(frame);
                    }
                    Ok(DecodedData::SeekComplete) => {}
                    Ok(DecodedData::Eof) => {
                        eof = true;
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        eof = true;
                        break;
                    }
                }
            }
        }

        // Start audio playback once pre-buffered (~100ms).
        if has_audio && !audio_playing && !paused {
            let buffered = rb_prod.capacity().get() - rb_prod.vacant_len();
            let threshold = (device_sample_rate as usize * device_channels) / 10;
            if buffered >= threshold || eof {
                if let Some(ref s) = audio_stream {
                    s.play()
                        .map_err(|e| Error::Other(format!("audio play: {e}")))?;
                }
                audio_playing = true;
            }
        }

        // A/V sync: determine current playback time.
        let playback_time = if has_audio && audio_playing {
            let consumed = audio_clock.load(Ordering::Relaxed);
            let clock_secs = consumed as f64 / (device_sample_rate as f64 * device_channels as f64);
            first_audio_pts.unwrap_or(0.0) + clock_secs
        } else if has_audio {
            // Pre-buffering: don't advance playback time yet.
            f64::NEG_INFINITY
        } else {
            // Video-only: wall clock with deferred start.
            match playback_start {
                Some(start) => {
                    let elapsed = start.elapsed().saturating_sub(pause_offset);
                    let elapsed = if let Some(ps) = pause_start {
                        elapsed.saturating_sub(ps.elapsed())
                    } else {
                        elapsed
                    };
                    first_video_pts + elapsed.as_secs_f64()
                }
                None => {
                    // Start clock when first frame is available.
                    if let Some(frame) = pending_video.front() {
                        first_video_pts = frame.pts_sec;
                        playback_start = Some(Instant::now());
                        frame.pts_sec // elapsed ≈ 0, displays first frame immediately
                    } else {
                        f64::NEG_INFINITY
                    }
                }
            }
        };

        // Drop late video frames, keeping the latest displayable.
        while pending_video.len() > 1
            && pending_video
                .get(1)
                .is_some_and(|f| f.pts_sec <= playback_time)
        {
            pending_video.pop_front();
        }

        // Display the front frame if its PTS has passed.
        if let Some(frame) = pending_video.front()
            && frame.pts_sec <= playback_time
        {
            rgba_to_minifb(
                &frame.rgba,
                frame.width,
                frame.height,
                &mut buf,
                win_w,
                win_h,
            );
            frame_count += 1;
            pending_video.pop_front();
        }

        window
            .update_with_buffer(&buf, win_w, win_h)
            .map_err(|e| Error::Other(format!("window update: {e}")))?;

        thread::sleep(Duration::from_millis(1));
    }

    let _ = cmd_tx.send(Command::Quit);
    eprintln!("Played {frame_count} video frames");

    Ok(())
}

/// Decode thread: reads packets, decodes audio/video, sends decoded data to main thread.
fn decode_thread_fn(args: DecodeThreadArgs) {
    let DecodeThreadArgs {
        mut ctx,
        video_idx,
        audio_idx,
        mut video_decoder,
        mut audio_decoder,
        vid_time_base,
        audio_time_base,
        src_sample_rate,
        device_sample_rate,
        device_channels,
        data_tx,
        cmd_rx,
    } = args;
    let mut converter: Option<Converter> = None;
    let mut resampler: Option<Resampler> =
        if src_sample_rate > 0 && src_sample_rate != device_sample_rate {
            match Resampler::new(
                src_sample_rate,
                device_sample_rate,
                device_channels,
                Quality::Fast,
            ) {
                Ok(r) => Some(r),
                Err(e) => {
                    eprintln!("Warning: resampler init failed: {e}");
                    None
                }
            }
        } else {
            None
        };
    let mut eof = false;

    loop {
        // Check for commands.
        let cmd = if eof {
            // At EOF, block waiting for seek or quit.
            match cmd_rx.recv() {
                Ok(cmd) => Some(cmd),
                Err(_) => break,
            }
        } else {
            cmd_rx.try_recv().ok()
        };

        if let Some(cmd) = cmd {
            match cmd {
                Command::Quit => break,
                Command::Seek(target_sec) => {
                    let seek_idx = audio_idx.or(video_idx).unwrap_or(0);
                    let seek_tb = if audio_idx.is_some() {
                        audio_time_base
                    } else {
                        vid_time_base
                    };
                    let timestamp = sec_to_pts(target_sec, seek_tb);

                    let _ = ctx.seek(seek_idx, timestamp, SeekFlags::BACKWARD);
                    if let Some(ref mut dec) = video_decoder {
                        dec.flush();
                    }
                    if let Some(ref mut dec) = audio_decoder {
                        dec.flush();
                    }
                    if let Some(ref mut rs) = resampler {
                        rs.reset();
                    }
                    eof = false;

                    if data_tx.send(DecodedData::SeekComplete).is_err() {
                        break;
                    }
                    continue;
                }
            }
        }

        // Read and decode one packet.
        match ctx.read_packet() {
            Ok(packet) => {
                if Some(packet.stream_index) == audio_idx {
                    if let Some(ref mut dec) = audio_decoder
                        && dec.send_packet(Some(&packet)).is_ok()
                    {
                        drain_audio_to_channel(
                            dec.as_mut(),
                            &mut resampler,
                            device_channels,
                            audio_time_base,
                            &data_tx,
                        );
                    }
                } else if Some(packet.stream_index) == video_idx
                    && let Some(ref mut dec) = video_decoder
                    && dec.send_packet(Some(&packet)).is_ok()
                {
                    drain_video_to_channel(dec.as_mut(), &mut converter, vid_time_base, &data_tx);
                }
            }
            Err(Error::Eof) => {
                // Drain decoders.
                if let Some(ref mut dec) = audio_decoder {
                    let _ = dec.send_packet(None);
                    drain_audio_to_channel(
                        dec.as_mut(),
                        &mut resampler,
                        device_channels,
                        audio_time_base,
                        &data_tx,
                    );
                }
                // Flush resampler tail.
                if let Some(ref mut rs) = resampler
                    && let Ok(tail) = rs.flush()
                    && !tail.is_empty()
                {
                    let _ = data_tx.send(DecodedData::Audio {
                        samples: tail,
                        pts_sec: 0.0,
                    });
                }
                if let Some(ref mut dec) = video_decoder {
                    let _ = dec.send_packet(None);
                    drain_video_to_channel(dec.as_mut(), &mut converter, vid_time_base, &data_tx);
                }
                eof = true;
                let _ = data_tx.send(DecodedData::Eof);
            }
            Err(_) => {
                eof = true;
                let _ = data_tx.send(DecodedData::Eof);
            }
        }
    }
}

/// Drain all audio frames from a decoder and send them via channel.
fn drain_audio_to_channel(
    decoder: &mut dyn wedeo_codec::decoder::Decoder,
    resampler: &mut Option<Resampler>,
    dst_channels: usize,
    audio_time_base: Rational,
    data_tx: &mpsc::SyncSender<DecodedData>,
) {
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                let pts_sec = pts_to_sec(frame.pts, audio_time_base);
                match decode_audio_to_f32(&frame, resampler, dst_channels) {
                    Ok(samples) => {
                        if data_tx
                            .send(DecodedData::Audio { samples, pts_sec })
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(_) => break,
                }
            }
            Err(Error::Again | Error::Eof) => break,
            Err(_) => break,
        }
    }
}

/// Drain all video frames from a decoder, convert to RGBA, and send via channel.
fn drain_video_to_channel(
    decoder: &mut dyn wedeo_codec::decoder::Decoder,
    converter: &mut Option<Converter>,
    vid_time_base: Rational,
    data_tx: &mpsc::SyncSender<DecodedData>,
) {
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                let video = match frame.video() {
                    Some(v) => v,
                    None => break,
                };

                let conv = match converter {
                    Some(c) => c,
                    None => {
                        match Converter::new(
                            video.format,
                            PixelFormat::Rgba,
                            video.width,
                            video.height,
                        ) {
                            Ok(c) => {
                                *converter = Some(c);
                                converter.as_mut().unwrap()
                            }
                            Err(_) => break,
                        }
                    }
                };

                let pts_sec = pts_to_sec(frame.pts, vid_time_base);
                match conv.convert(&frame) {
                    Ok(rgba_frame) => {
                        if let Some(rgba_video) = rgba_frame.video() {
                            let rgba_data = rgba_video.planes[0].buffer.data().to_vec();
                            let vf = VideoFrame {
                                rgba: rgba_data,
                                width: video.width as usize,
                                height: video.height as usize,
                                pts_sec,
                            };
                            if data_tx.send(DecodedData::Video(vf)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            Err(Error::Again | Error::Eof) => break,
            Err(_) => break,
        }
    }
}

/// Convert PTS in time_base units to seconds.
fn pts_to_sec(pts: i64, time_base: Rational) -> f64 {
    if pts >= 0 && time_base.num > 0 && time_base.den > 0 {
        pts as f64 * time_base.num as f64 / time_base.den as f64
    } else {
        0.0
    }
}

/// Convert seconds to PTS in time_base units.
fn sec_to_pts(sec: f64, time_base: Rational) -> i64 {
    if time_base.num > 0 && time_base.den > 0 {
        (sec * time_base.den as f64 / time_base.num as f64) as i64
    } else {
        0
    }
}

/// Convert a decoded audio frame to f32 interleaved samples, with channel mapping and resampling.
/// Uses the frame's actual channel count, not the stream header's.
fn decode_audio_to_f32(
    frame: &Frame,
    resampler: &mut Option<Resampler>,
    dst_channels: usize,
) -> wedeo_core::Result<Vec<f32>> {
    let audio = frame.audio().ok_or(Error::InvalidData)?;
    let channels = audio.channel_layout.nb_channels as usize;
    let nb_samples = audio.nb_samples as usize;

    let f32_samples = if audio.format.is_planar() && audio.planes.len() > 1 {
        // True multi-plane planar audio: interleave from separate planes.
        planar_to_f32(&audio.planes, audio.format, nb_samples, channels)
    } else {
        // Packed (interleaved) audio in a single plane.
        let raw = audio.planes.first().map(|p| p.buffer.data()).unwrap_or(&[]);
        samples_to_f32(raw, audio.format, nb_samples, channels)
    };

    let mapped = map_channels(&f32_samples, channels, dst_channels);

    if let Some(rs) = resampler {
        rs.process(&mapped)
    } else {
        Ok(mapped)
    }
}

/// Convert multi-plane planar audio to f32 interleaved samples.
/// Each `FramePlane` contains one channel's worth of samples.
fn planar_to_f32(
    planes: &[FramePlane],
    format: SampleFormat,
    nb_samples: usize,
    channels: usize,
) -> Vec<f32> {
    let bps = format.bytes_per_sample();
    let mut out = Vec::with_capacity(nb_samples * channels);

    for s in 0..nb_samples {
        for ch in 0..channels {
            if ch >= planes.len() {
                out.push(0.0);
                continue;
            }
            let data = planes[ch].buffer.data();
            let offset = s * bps;
            if offset + bps > data.len() {
                out.push(0.0);
                continue;
            }
            out.push(match format {
                SampleFormat::U8p => (data[offset] as f32 - 128.0) / 128.0,
                SampleFormat::S16p => {
                    i16::from_ne_bytes([data[offset], data[offset + 1]]) as f32 / 32768.0
                }
                SampleFormat::S32p => {
                    i32::from_ne_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]) as f32
                        / 2_147_483_648.0
                }
                SampleFormat::Fltp => f32::from_ne_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]),
                SampleFormat::Dblp => f64::from_ne_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]) as f32,
                SampleFormat::S64p => {
                    i64::from_ne_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                        data[offset + 4],
                        data[offset + 5],
                        data[offset + 6],
                        data[offset + 7],
                    ]) as f32
                        / 9_223_372_036_854_775_808.0
                }
                _ => 0.0,
            });
        }
    }
    out
}

/// Convert raw packed (interleaved) audio bytes to f32 samples.
fn samples_to_f32(
    data: &[u8],
    format: SampleFormat,
    nb_samples: usize,
    channels: usize,
) -> Vec<f32> {
    let total = nb_samples * channels;
    let mut out = Vec::with_capacity(total);

    match format {
        SampleFormat::U8 => {
            for &b in data.iter().take(total) {
                out.push((b as f32 - 128.0) / 128.0);
            }
        }
        SampleFormat::S16 => {
            for chunk in data.chunks_exact(2).take(total) {
                let s = i16::from_ne_bytes([chunk[0], chunk[1]]);
                out.push(s as f32 / 32768.0);
            }
        }
        SampleFormat::S32 => {
            for chunk in data.chunks_exact(4).take(total) {
                let s = i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(s as f32 / 2_147_483_648.0);
            }
        }
        SampleFormat::Flt => {
            for chunk in data.chunks_exact(4).take(total) {
                out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        SampleFormat::Dbl => {
            for chunk in data.chunks_exact(8).take(total) {
                let d = f64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(d as f32);
            }
        }
        SampleFormat::S64 => {
            for chunk in data.chunks_exact(8).take(total) {
                let s = i64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(s as f32 / 9_223_372_036_854_775_808.0);
            }
        }
        _ => {
            // Unsupported packed format: output silence.
            out.resize(total, 0.0);
        }
    }

    out
}

/// Map audio from src_channels to dst_channels.
fn map_channels(samples: &[f32], src_ch: usize, dst_ch: usize) -> Vec<f32> {
    if src_ch == dst_ch || src_ch == 0 || dst_ch == 0 {
        return samples.to_vec();
    }

    let frames = samples.len() / src_ch;
    let mut out = Vec::with_capacity(frames * dst_ch);

    // Standard sqrt(2)/2 mixing coefficient (ITU-R BS.775).
    const MIX: f32 = std::f32::consts::FRAC_1_SQRT_2;

    for f in 0..frames {
        let s = f * src_ch;
        if src_ch == 1 && dst_ch == 2 {
            // Mono to stereo: duplicate.
            out.push(samples[s]);
            out.push(samples[s]);
        } else if src_ch == 6 && dst_ch == 2 {
            // 5.1 (FL, FR, C, LFE, SL, SR) → stereo downmix.
            let fl = samples[s];
            let fr = samples[s + 1];
            let c = samples[s + 2];
            // LFE (s+3) excluded from stereo downmix.
            let sl = samples[s + 4];
            let sr = samples[s + 5];
            out.push(fl + MIX * c + MIX * sl);
            out.push(fr + MIX * c + MIX * sr);
        } else if src_ch > dst_ch {
            // Generic downmix: take first N channels.
            for c in 0..dst_ch {
                out.push(samples[s + c]);
            }
        } else {
            // Upmix: copy available, zero-fill rest.
            for c in 0..dst_ch {
                if c < src_ch {
                    out.push(samples[s + c]);
                } else {
                    out.push(0.0);
                }
            }
        }
    }

    out
}

/// Convert RGBA byte buffer to minifb's `0x00RRGGBB` u32 format, scaled to fit the window.
fn rgba_to_minifb(
    rgba: &[u8],
    src_w: usize,
    src_h: usize,
    buf: &mut [u32],
    win_w: usize,
    win_h: usize,
) {
    buf.fill(0);

    if src_w == 0 || src_h == 0 || win_w == 0 || win_h == 0 {
        return;
    }

    // Compute fitted dimensions preserving aspect ratio.
    let (dst_w, dst_h) = if src_w * win_h < src_h * win_w {
        // Height-limited.
        (src_w * win_h / src_h, win_h)
    } else {
        // Width-limited.
        (win_w, src_h * win_w / src_w)
    };

    let offset_x = (win_w - dst_w) / 2;
    let offset_y = (win_h - dst_h) / 2;

    for dy in 0..dst_h {
        let sy = dy * src_h / dst_h;
        for dx in 0..dst_w {
            let sx = dx * src_w / dst_w;
            let src_idx = (sy * src_w + sx) * 4;
            if src_idx + 2 < rgba.len() {
                let r = rgba[src_idx] as u32;
                let g = rgba[src_idx + 1] as u32;
                let b = rgba[src_idx + 2] as u32;
                buf[(dy + offset_y) * win_w + (dx + offset_x)] = (r << 16) | (g << 8) | b;
            }
        }
    }
}
