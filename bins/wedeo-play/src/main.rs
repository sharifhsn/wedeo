use std::collections::VecDeque;
use std::env;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

// Ensure inventory registrations are linked in.
use wedeo_codec_h264 as _;
use wedeo_codec_pcm as _;
use wedeo_format_h264 as _;
use wedeo_format_mp4 as _;
use wedeo_format_wav as _;
use wedeo_rav1d as _;
use wedeo_symphonia as _;

use wedeo_codec::decoder::DecoderBuilder;
use wedeo_core::error::Error;
use wedeo_core::frame::{ColorRange, ColorSpace, Frame, FramePlane};
use wedeo_core::pixel_format::PixelFormat;
use wedeo_core::rational::Rational;
use wedeo_core::sample_format::SampleFormat;
use wedeo_core::{MediaType, Packet};
use wedeo_format::context::InputContext;
use wedeo_resample::{Quality, Resampler};

// ---------------------------------------------------------------------------
// WGSL shader: full-screen triangle + YUV420p → RGB with color space support
// ---------------------------------------------------------------------------
const SHADER: &str = r#"
struct Params {
    color_matrix: u32, // 0 = BT.601, 1 = BT.709
    full_range: u32,   // 0 = MPEG limited [16-235], 1 = JPEG full [0-255]
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var u_tex: texture_2d<f32>;
@group(0) @binding(2) var v_tex: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;
@group(0) @binding(4) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen triangle: 3 vertices cover clip space, no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(idx & 1u)) * 4.0 - 1.0;
    let y = f32(i32(idx >> 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // Map clip space to UV: x [-1,1] → [0,1], y [-1,1] → [1,0] (flip for top-left origin)
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let y_raw = textureSample(y_tex, tex_sampler, in.uv).r;
    let u_raw = textureSample(u_tex, tex_sampler, in.uv).r;
    let v_raw = textureSample(v_tex, tex_sampler, in.uv).r;

    var y: f32;
    var cb: f32;
    var cr: f32;

    if params.full_range == 1u {
        // JPEG / full range: Y in [0,1], Cb/Cr centered at 0.5
        y = y_raw;
        cb = u_raw - 0.5;
        cr = v_raw - 0.5;
    } else {
        // MPEG / limited range: Y in [16,235], Cb/Cr in [16,240] center 128
        y = (y_raw - 0.0627451) * 1.164384;   // (Y - 16/255) * 255/219
        cb = (u_raw - 0.5) * 1.138393;        // (Cb - 128/255) * 255/224
        cr = (v_raw - 0.5) * 1.138393;
    }

    var r: f32;
    var g: f32;
    var b: f32;

    if params.color_matrix == 1u {
        // BT.709 (HD)
        r = y + 1.5748 * cr;
        g = y - 0.1873 * cb - 0.4681 * cr;
        b = y + 1.8556 * cb;
    } else {
        // BT.601 (SD, default)
        r = y + 1.402 * cr;
        g = y - 0.344136 * cb - 0.714136 * cr;
        b = y + 1.772 * cb;
    }

    return vec4<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
}
"#;

// ---------------------------------------------------------------------------
// Audio clock — port of ffplay's Clock struct (lines 138-146)
// ---------------------------------------------------------------------------

/// Audio clock using ffplay's pts_drift model.
///
/// The key insight: `pts_drift = pts - wall_time`. Between updates,
/// `get()` interpolates: `pts_drift + current_wall_time`. During pause,
/// it returns the frozen `pts` directly.
///
/// Reference: ffplay.c lines 138-146 (Clock struct), 1428-1438 (get_clock),
/// 1440-1452 (set_clock_at/set_clock).
struct AudioClock {
    /// `pts - wall_time` when last set. Enables interpolation between updates.
    pts_drift: f64,
    /// Wall-clock time (seconds since process start) when last updated.
    last_updated: f64,
    /// The raw PTS value (returned directly when paused).
    pts: f64,
    /// Whether the clock is frozen (paused).
    paused: bool,
}

impl AudioClock {
    fn new() -> Self {
        Self {
            pts_drift: 0.0,
            last_updated: 0.0,
            pts: 0.0,
            paused: false,
        }
    }

    /// Set the clock to `pts` at wall-clock time `time`.
    /// Matches ffplay's `set_clock_at` (line 1440).
    fn set_at(&mut self, pts: f64, time: f64) {
        self.pts = pts;
        self.last_updated = time;
        self.pts_drift = pts - time;
    }

    /// Get the current clock value.
    /// Matches ffplay's `get_clock` (line 1428).
    fn get(&self, now: f64) -> f64 {
        if self.paused {
            self.pts
        } else {
            self.pts_drift + now
        }
    }
}

/// Get wall-clock time in seconds (monotonic, relative to process start).
fn wall_time() -> f64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_secs_f64()
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Decoded video frame with raw YUV plane data for GPU upload.
struct VideoFrame {
    y_data: Vec<u8>,
    u_data: Vec<u8>,
    v_data: Vec<u8>,
    width: u32,
    height: u32,
    pts_sec: f64,
    color_matrix: u32,
    full_range: u32,
}

/// Commands sent from the main thread to the read thread.
enum Command {
    Quit,
}

/// Arguments for the read thread (reads packets, dispatches to per-stream decode threads).
struct ReadThreadArgs {
    ctx: InputContext,
    video_idx: Option<usize>,
    audio_idx: Option<usize>,
    video_pkt_tx: Option<mpsc::Sender<PacketMsg>>,
    audio_pkt_tx: Option<mpsc::Sender<PacketMsg>>,
    event_tx: mpsc::Sender<DecodeEvent>,
    cmd_rx: mpsc::Receiver<Command>,
    /// Shared atomic counters for self-throttle (matching ffplay's
    /// stream_has_enough_packets check, lines 2854/3131-3141).
    video_pkt_count: Arc<AtomicU64>,
    audio_pkt_count: Arc<AtomicU64>,
}

/// Arguments for the video decode thread.
struct VideoDecodeArgs {
    decoder: Box<dyn wedeo_codec::decoder::Decoder>,
    pkt_rx: mpsc::Receiver<PacketMsg>,
    frame_tx: mpsc::SyncSender<VideoFrame>,
    vid_time_base: Rational,
    pkt_count: Arc<AtomicU64>,
}

/// Arguments for the audio decode thread.
struct AudioDecodeArgs {
    decoder: Box<dyn wedeo_codec::decoder::Decoder>,
    pkt_rx: mpsc::Receiver<PacketMsg>,
    frame_tx: mpsc::SyncSender<AudioData>,
    audio_time_base: Rational,
    src_sample_rate: u32,
    device_sample_rate: u32,
    device_channels: usize,
    pkt_count: Arc<AtomicU64>,
}

/// Audio samples with PTS.
struct AudioData {
    samples: Vec<f32>,
    pts_sec: f64,
}

/// Non-blocking events from decode thread.
enum DecodeEvent {
    Eof,
}

/// Messages sent to per-stream decode threads.
enum PacketMsg {
    Packet(Packet),
}

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    render_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    y_tex: wgpu::Texture,
    u_tex: wgpu::Texture,
    v_tex: wgpu::Texture,
    params_buf: wgpu::Buffer,
    tex_width: u32,
    tex_height: u32,
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct App {
    // Config (set once before event loop)
    path: String,
    vid_width: u32,
    vid_height: u32,
    has_audio: bool,
    device_sample_rate: u32,
    device_channels: usize,
    init_win_size: (u32, u32),

    // Channels (to/from decode thread)
    video_rx: mpsc::Receiver<VideoFrame>,
    audio_rx: mpsc::Receiver<AudioData>,
    event_rx: mpsc::Receiver<DecodeEvent>,
    cmd_tx: mpsc::Sender<Command>,

    // Audio shared state
    audio_clock: Arc<Mutex<AudioClock>>,
    audio_pts_shared: Arc<AtomicU64>,
    audio_samples_written: Arc<AtomicU64>,
    volume: Arc<AtomicU32>,
    rb_prod: <HeapRb<f32> as Split>::Prod,
    audio_stream: Option<cpal::Stream>,

    // GPU (initialized in resumed)
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,

    // Playback state
    pending_video: VecDeque<VideoFrame>,
    pending_audio: Vec<f32>,
    frame_count: u64,
    eof: bool,
    audio_playing: bool,
    first_audio_pts: Option<f64>,
    paused: bool,
    pause_start: Option<Instant>,
    current_volume: f32,
    has_first_frame: bool,

    // Frame pacing (ffplay's frame_timer + compute_target_delay pattern)
    frame_timer: Option<Instant>,
    frame_duration: f64, // nominal frame duration in seconds (e.g. 1/24)

    // FPS stats
    fps_last_print: Option<Instant>,
    fps_frame_count: u64,
    fps_dropped: u64,

    // Error propagation from resumed()
    init_error: Option<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut scale: Option<usize> = None;
    let mut file = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--scale" => {
                i += 1;
                let s = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("--scale requires a positive integer");
                    std::process::exit(1);
                });
                scale = Some(if s == 0 { 1 } else { s });
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

fn run(path: &str, initial_scale: Option<usize>) -> wedeo_core::Result<()> {
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
    let audio_clock = Arc::new(Mutex::new(AudioClock::new()));
    let volume = Arc::new(AtomicU32::new(f32::to_bits(1.0)));
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
    let (rb_prod, rb_cons) = rb.split();

    // Shared audio PTS: audio decode thread writes the PTS of the latest chunk
    // pushed to the ring buffer; cpal callback reads it to update the clock.
    // This tracks the PTS of audio "near the head" of the ring buffer.
    let audio_pts_shared = Arc::new(AtomicU64::new(f64::to_bits(0.0)));
    // Cumulative sample counter for precise PTS interpolation within the ring buffer.
    let audio_samples_written = Arc::new(AtomicU64::new(0));
    let audio_samples_read = Arc::new(AtomicU64::new(0));

    // Build cpal output stream with volume support.
    let audio_stream: Option<cpal::Stream>;
    if has_audio {
        let stream_config = cpal::StreamConfig {
            channels: device_channels as u16,
            sample_rate: device_sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let clock = audio_clock.clone();
        let vol = volume.clone();
        let pts_shared = audio_pts_shared.clone();
        let samples_written = audio_samples_written.clone();
        let samples_read = audio_samples_read.clone();
        let cb_rate = device_sample_rate as f64;
        let cb_ch = device_channels as f64;
        let mut audio_consumer = rb_cons;

        let stream = cpal_device
            .as_ref()
            .unwrap()
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let gain = f32::from_bits(vol.load(Ordering::Relaxed));
                    let filled = audio_consumer.pop_slice(data);
                    for sample in &mut data[..filled] {
                        *sample *= gain;
                    }
                    for sample in &mut data[filled..] {
                        *sample = 0.0;
                    }

                    // Update audio clock (matching ffplay sdl_audio_callback line 2570).
                    // Compute the PTS of the audio currently being played:
                    // base_pts + (samples_read - samples_at_base_pts) / (rate * ch)
                    let total_read =
                        samples_read.fetch_add(filled as u64, Ordering::Relaxed) + filled as u64;
                    let total_written = samples_written.load(Ordering::Relaxed);
                    let base_pts = f64::from_bits(pts_shared.load(Ordering::Relaxed));
                    // Samples still in the ring buffer (not yet played).
                    let buffered = total_written.saturating_sub(total_read);
                    // The PTS of what's actually playing = base_pts minus
                    // the duration of buffered samples.
                    let playing_pts = base_pts - buffered as f64 / (cb_rate * cb_ch);
                    let now = wall_time();
                    if let Ok(mut clk) = clock.try_lock() {
                        clk.set_at(playing_pts, now);
                    }
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

    // Window size — auto-scale so small videos aren't tiny, HD stays 1:1.
    let scale = initial_scale.unwrap_or_else(|| {
        let target = 1280;
        let dim = vid_width.max(vid_height);
        if dim == 0 { 1 } else { (target / dim).max(1) }
    });
    let init_win_w = (vid_width * scale) as u32;
    let init_win_h = (vid_height * scale) as u32;

    // --- Thread architecture (matches ffplay) ---
    // Read thread:        reads packets, dispatches to per-stream packet channels.
    // Video decode thread: decodes video packets → sends VideoFrames (blocks only on own queue).
    // Audio decode thread: decodes audio packets → sends AudioData  (blocks only on own queue).
    // This ensures video decode NEVER stalls audio decode, and vice versa.

    // Output channels (decode threads → display thread).
    let (video_tx, video_rx) = mpsc::sync_channel::<VideoFrame>(8);
    let (audio_tx, audio_rx) = mpsc::sync_channel::<AudioData>(16);
    let (event_tx, event_rx) = mpsc::channel::<DecodeEvent>();
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();

    // Per-stream packet channels (read thread → decode threads).
    // Unbounded channels matching ffplay's auto-growing PacketQueue.
    // The read thread self-throttles (sleep 10ms) when queues have enough
    // packets, matching ffplay lines 3131-3141. Backpressure comes from
    // the bounded frame channels (sync_channel(8) for video), not here.
    let (video_pkt_tx, video_pkt_rx) = if video_decoder.is_some() {
        let (tx, rx) = mpsc::channel::<PacketMsg>();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let (audio_pkt_tx, audio_pkt_rx) = if audio_decoder.is_some() {
        let (tx, rx) = mpsc::channel::<PacketMsg>();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    // Shared atomic counters for read thread self-throttle.
    let video_pkt_count = Arc::new(AtomicU64::new(0));
    let audio_pkt_count = Arc::new(AtomicU64::new(0));

    // Spawn video decode thread.
    if let (Some(dec), Some(pkt_rx)) = (video_decoder, video_pkt_rx) {
        let cnt = Arc::clone(&video_pkt_count);
        thread::spawn(move || {
            video_decode_thread(VideoDecodeArgs {
                decoder: dec,
                pkt_rx,
                frame_tx: video_tx,
                vid_time_base,
                pkt_count: cnt,
            });
        });
    }

    // Spawn audio decode thread.
    if let (Some(dec), Some(pkt_rx)) = (audio_decoder, audio_pkt_rx) {
        let cnt = Arc::clone(&audio_pkt_count);
        thread::spawn(move || {
            audio_decode_thread(AudioDecodeArgs {
                decoder: dec,
                pkt_rx,
                frame_tx: audio_tx,
                audio_time_base,
                src_sample_rate,
                device_sample_rate,
                device_channels,
                pkt_count: cnt,
            });
        });
    }

    // Spawn read thread.
    let _read_handle = thread::spawn(move || {
        read_thread_fn(ReadThreadArgs {
            ctx,
            video_idx,
            audio_idx,
            video_pkt_tx,
            audio_pkt_tx,
            event_tx,
            cmd_rx,
            video_pkt_count,
            audio_pkt_count,
        });
    });

    // Create event loop and application.
    let event_loop = EventLoop::new().map_err(|e| Error::Other(format!("event loop: {e}")))?;

    let mut app = App {
        path: path.to_string(),
        vid_width: vid_width as u32,
        vid_height: vid_height as u32,
        has_audio,
        device_sample_rate,
        device_channels,
        init_win_size: (init_win_w, init_win_h),
        video_rx,
        audio_rx,
        event_rx,
        cmd_tx,
        audio_clock,
        audio_pts_shared,
        audio_samples_written,
        volume,
        rb_prod,
        audio_stream,
        window: None,
        gpu: None,
        pending_video: VecDeque::new(),
        pending_audio: Vec::new(),
        frame_count: 0,
        eof: false,
        audio_playing: false,
        first_audio_pts: None,
        paused: false,
        pause_start: None,
        current_volume: 1.0,
        has_first_frame: false,
        frame_timer: None,
        frame_duration: 1.0 / 24.0, // default; updated from actual PTS gaps
        fps_last_print: None,
        fps_frame_count: 0,
        fps_dropped: 0,
        init_error: None,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| Error::Other(format!("event loop: {e}")))?;

    if let Some(err) = app.init_error {
        return Err(Error::Other(err));
    }

    eprintln!("Played {} video frames", app.frame_count);
    Ok(())
}

// ---------------------------------------------------------------------------
// App helpers
// ---------------------------------------------------------------------------

impl App {
    /// Poll decoded data from separate channels and push audio to the ring buffer.
    fn poll_data(&mut self) {
        if self.paused {
            return;
        }

        // Push any pending audio first.
        if !self.pending_audio.is_empty() {
            let pushed = self.rb_prod.push_slice(&self.pending_audio);
            self.audio_samples_written
                .fetch_add(pushed as u64, Ordering::Relaxed);
            if pushed >= self.pending_audio.len() {
                self.pending_audio.clear();
            } else {
                self.pending_audio.drain(..pushed);
            }
        }

        // Poll audio channel — fill ring buffer.
        for _ in 0..20 {
            if self.has_audio && self.rb_prod.vacant_len() < self.device_channels * 4096 {
                break;
            }
            match self.audio_rx.try_recv() {
                Ok(AudioData { samples, pts_sec }) => {
                    if self.first_audio_pts.is_none() {
                        self.first_audio_pts = Some(pts_sec);
                    }
                    // Update shared PTS for the cpal callback's clock.
                    // PTS represents the END of this chunk (after all samples play).
                    let chunk_duration = samples.len() as f64
                        / (self.device_sample_rate as f64 * self.device_channels as f64);
                    let end_pts = pts_sec + chunk_duration;
                    self.audio_pts_shared
                        .store(f64::to_bits(end_pts), Ordering::Relaxed);
                    let pushed = self.rb_prod.push_slice(&samples);
                    self.audio_samples_written
                        .fetch_add(pushed as u64, Ordering::Relaxed);
                    if pushed < samples.len() {
                        self.pending_audio.extend_from_slice(&samples[pushed..]);
                        break;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }

        // Poll video channel — keep pending_video small (like ffplay's
        // VIDEO_PICTURE_QUEUE_SIZE=3). The decode thread blocks on the
        // frame channel when full, providing natural backpressure.
        while self.pending_video.len() < 3 {
            match self.video_rx.try_recv() {
                Ok(frame) => self.pending_video.push_back(frame),
                Err(_) => break,
            }
        }

        // Poll events.
        while let Ok(evt) = self.event_rx.try_recv() {
            match evt {
                DecodeEvent::Eof => self.eof = true,
            }
        }

        // Start audio playback once pre-buffered (~100ms).
        if self.has_audio && !self.audio_playing && !self.paused {
            let buffered = self.rb_prod.capacity().get() - self.rb_prod.vacant_len();
            let threshold = (self.device_sample_rate as usize * self.device_channels) / 10;
            if buffered >= threshold || self.eof {
                if let Some(ref s) = self.audio_stream {
                    let _ = s.play();
                }
                self.audio_playing = true;
            }
        }
    }

    /// Pause/unpause matching ffplay's stream_toggle_pause (line 1540).
    fn toggle_pause(&mut self) {
        if self.paused {
            // Unpausing — adjust frame_timer for pause duration
            // (matches ffplay stream_toggle_pause line 1543).
            let now_inst = Instant::now();
            if let (Some(ft), Some(ps)) = (self.frame_timer, self.pause_start) {
                self.frame_timer = Some(ft + now_inst.duration_since(ps));
            }
            // Re-anchor audio clock pts_drift to current wall time
            // (matches ffplay line 1547: set_clock(vidclk, get_clock(vidclk))).
            {
                let now = wall_time();
                let mut clk = self.audio_clock.lock().unwrap();
                let pts = clk.get(now);
                clk.paused = false;
                clk.set_at(pts, now);
            }
            // Resume audio stream.
            if self.audio_playing
                && let Some(ref s) = self.audio_stream
            {
                let _ = s.play();
            }
            self.pause_start = None;
        } else {
            // Pausing — freeze audio and record pause start.
            if let Some(ref s) = self.audio_stream {
                let _ = s.pause();
            }
            self.audio_clock.lock().unwrap().paused = true;
            self.pause_start = Some(Instant::now());
        }
        self.paused = !self.paused;
    }

    /// Compute target delay for A/V sync (port of ffplay's compute_target_delay).
    /// `delay` is the nominal frame duration, adjusted based on video-vs-audio drift.
    fn compute_target_delay(&self, delay: f64, video_pts: f64) -> f64 {
        // ffplay sync thresholds
        const SYNC_THRESHOLD_MIN: f64 = 0.04;
        const SYNC_THRESHOLD_MAX: f64 = 0.1;
        const FRAMEDUP_THRESHOLD: f64 = 0.1;

        if !self.has_audio || !self.audio_playing {
            return delay; // no audio master — use nominal delay
        }

        // Audio clock: ffplay's pts_drift model (line 1428/1587).
        // The cpal callback updates the clock with the PTS of the audio
        // currently playing. get() interpolates between updates.
        let master_clock = self.audio_clock.lock().unwrap().get(wall_time());
        let diff = video_pts - master_clock;

        let sync_threshold = delay.clamp(SYNC_THRESHOLD_MIN, SYNC_THRESHOLD_MAX);
        if diff.abs() < 10.0 {
            if diff <= -sync_threshold {
                (delay + diff).max(0.0)
            } else if diff >= sync_threshold && delay > FRAMEDUP_THRESHOLD {
                delay + diff
            } else if diff >= sync_threshold {
                2.0 * delay
            } else {
                delay
            }
        } else {
            delay
        }
    }

    fn render(&mut self) {
        if self.gpu.is_none() {
            return;
        }

        let now = Instant::now();

        // --- Frame pacing (ffplay video_refresh, lines 1628–1745) ---
        // Exact port of ffplay's logic with retry loop for late drops.
        if !self.pending_video.is_empty() {
            if self.frame_timer.is_none() {
                self.frame_timer = Some(now);
            }

            'retry: loop {
                if self.pending_video.is_empty() {
                    break;
                }

                // Skip frame timing when paused (ffplay line 1668: goto display).
                if self.paused {
                    break;
                }

                // vp_duration: PTS gap to next frame (ffplay line 1609-1619).
                let duration = if self.pending_video.len() > 1 {
                    let d = self.pending_video[1].pts_sec - self.pending_video[0].pts_sec;
                    if d > 0.0 && d < 10.0 {
                        d
                    } else {
                        self.frame_duration
                    }
                } else {
                    self.frame_duration
                };

                // compute_target_delay: A/V sync adjustment (ffplay line 1673).
                let delay = self.compute_target_delay(duration, self.pending_video[0].pts_sec);

                let ft = self.frame_timer.unwrap();
                let time = now.duration_since(ft).as_secs_f64();
                if time < delay {
                    break;
                }

                // Advance frame_timer by DELAY (ffplay line 1681).
                self.frame_timer = Some(ft + Duration::from_secs_f64(delay));

                // Snap if way behind (ffplay lines 1682-1683).
                if delay > 0.0 && now.duration_since(self.frame_timer.unwrap()).as_secs_f64() > 0.1
                {
                    self.frame_timer = Some(now);
                }

                // Update cached frame_duration from actual PTS gap.
                if duration > 0.0 && duration < 10.0 {
                    self.frame_duration = duration;
                }

                // Late frame drop with retry (ffplay lines 1690-1697, goto retry).
                // If the next frame is ALSO past its deadline, drop current and retry.
                if self.pending_video.len() > 1 {
                    let next_dur = {
                        let d = self.pending_video[1].pts_sec - self.pending_video[0].pts_sec;
                        if d > 0.0 && d < 10.0 {
                            d
                        } else {
                            self.frame_duration
                        }
                    };
                    let ft = self.frame_timer.unwrap();
                    if now.duration_since(ft).as_secs_f64() > next_dur {
                        self.pending_video.pop_front();
                        self.fps_dropped += 1;
                        continue 'retry; // ffplay: goto retry
                    }
                }

                // Display this frame (ffplay line 1734: frame_queue_next).
                if let Some(frame) = self.pending_video.pop_front() {
                    let gpu = self.gpu.as_mut().unwrap();
                    if frame.width != gpu.tex_width || frame.height != gpu.tex_height {
                        gpu.recreate_textures(frame.width, frame.height);
                    }
                    gpu.upload_frame(&frame);
                    self.has_first_frame = true;
                    self.frame_count += 1;
                    self.fps_frame_count += 1;
                }
                break;
            }
        }

        // Print FPS every 2 seconds.
        let last = *self.fps_last_print.get_or_insert(now);
        let elapsed = now.duration_since(last);
        if elapsed >= Duration::from_secs(2) {
            let fps = self.fps_frame_count as f64 / elapsed.as_secs_f64();
            eprintln!(
                "fps={fps:.1} displayed={} dropped={} pending={}",
                self.fps_frame_count,
                self.fps_dropped,
                self.pending_video.len()
            );
            self.fps_frame_count = 0;
            self.fps_dropped = 0;
            self.fps_last_print = Some(now);
        }

        // Acquire surface texture.
        let result = self.gpu.as_ref().unwrap().surface.get_current_texture();
        let output = match result {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                let gpu = self.gpu.as_mut().unwrap();
                let (w, h) = (gpu.surface_config.width, gpu.surface_config.height);
                gpu.resize(w, h);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("Surface validation error");
                return;
            }
        };
        let gpu = self.gpu.as_ref().unwrap();

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(gpu.render_format),
            ..Default::default()
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("yuv_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            if self.has_first_frame {
                let (vx, vy, vw, vh) = compute_viewport(
                    gpu.tex_width,
                    gpu.tex_height,
                    gpu.surface_config.width,
                    gpu.surface_config.height,
                );
                pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
                pass.set_pipeline(&gpu.pipeline);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

// ---------------------------------------------------------------------------
// ApplicationHandler
// ---------------------------------------------------------------------------

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // Already initialized.
        }

        let attrs = Window::default_attributes()
            .with_title(format!("wedeo-play — {}", self.path))
            .with_inner_size(LogicalSize::new(self.init_win_size.0, self.init_win_size.1))
            .with_resizable(true);

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.init_error = Some(format!("window creation failed: {e}"));
                event_loop.exit();
                return;
            }
        };

        match GpuState::new(window.clone(), self.vid_width, self.vid_height) {
            Ok(gpu) => {
                self.gpu = Some(gpu);
                self.window = Some(window);
            }
            Err(e) => {
                self.init_error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                let _ = self.cmd_tx.send(Command::Quit);
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(ref mut gpu) = self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed && !event.repeat {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            let _ = self.cmd_tx.send(Command::Quit);
                            event_loop.exit();
                        }
                        Key::Named(NamedKey::Space) => self.toggle_pause(),
                        Key::Named(NamedKey::ArrowUp) => {
                            self.current_volume = (self.current_volume + 0.1).min(2.0);
                            self.volume
                                .store(f32::to_bits(self.current_volume), Ordering::Relaxed);
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            self.current_volume = (self.current_volume - 0.1).max(0.0);
                            self.volume
                                .store(f32::to_bits(self.current_volume), Ordering::Relaxed);
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                self.render();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.poll_data();

        // Auto-close when playback is finished: EOF reached and all frames displayed.
        if self.eof && self.pending_video.is_empty() && self.has_first_frame {
            let _ = self.cmd_tx.send(Command::Quit);
            event_loop.exit();
            return;
        }

        // Compute remaining_time until next frame (ffplay's refresh_loop_wait_event
        // pattern, lines 3393-3408). Default to 10ms (REFRESH_RATE).
        let mut remaining = Duration::from_millis(10);
        let mut frame_ready = false;

        if let (Some(ft), Some(front)) = (self.frame_timer, self.pending_video.front()) {
            let duration = if self.pending_video.len() > 1 {
                let d = self.pending_video[1].pts_sec - front.pts_sec;
                if d > 0.0 && d < 10.0 {
                    d
                } else {
                    self.frame_duration
                }
            } else {
                self.frame_duration
            };
            let delay = self.compute_target_delay(duration, front.pts_sec);
            let target = ft + Duration::from_secs_f64(delay);
            let now = Instant::now();
            if target > now {
                remaining = remaining.min(target - now);
            } else {
                frame_ready = true;
                remaining = Duration::ZERO;
            }
        } else if !self.pending_video.is_empty() {
            // First frame, no timer yet — display immediately.
            frame_ready = true;
            remaining = Duration::ZERO;
        }

        // Only request redraw when a frame is actually due or we need to
        // present the first frame. This avoids doing redundant GPU render
        // passes when we're just waiting for the next frame deadline.
        if (frame_ready || (!self.has_first_frame && !self.pending_video.is_empty()))
            && let Some(ref w) = self.window
        {
            w.request_redraw();
        }

        // WaitUntil: wake the event loop when the next frame is due.
        // When frame_ready is true, use Poll so we process immediately.
        if remaining > Duration::ZERO {
            event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + remaining));
        } else {
            event_loop.set_control_flow(ControlFlow::Poll);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU state management
// ---------------------------------------------------------------------------

impl GpuState {
    fn new(window: Arc<Window>, vid_width: u32, vid_height: u32) -> Result<Self, String> {
        pollster::block_on(Self::new_async(window, vid_width, vid_height))
    }

    async fn new_async(
        window: Arc<Window>,
        vid_width: u32,
        vid_height: u32,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create surface: {e}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no compatible GPU adapter: {e}"))?;

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| format!("request device: {e}"))?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats[0];
        // Use non-sRGB view format so the shader outputs gamma-space values directly
        // (video YUV→RGB values are already gamma-encoded).
        let render_format = surface_format.remove_srgb_suffix();

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: caps.alpha_modes[0],
            view_formats: if render_format != surface_format {
                vec![render_format]
            } else {
                vec![]
            },
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Shader module.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yuv_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // Sampler with bilinear filtering for smooth scaling.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // YUV textures.
        let tw = vid_width.max(2);
        let th = vid_height.max(2);
        let (y_tex, u_tex, v_tex) = create_yuv_textures(&device, tw, th);

        // Initialize U/V textures to 128 (neutral chroma = black in YUV).
        let u_init = vec![128u8; (tw / 2 * th / 2) as usize];
        upload_plane(&queue, &u_tex, tw / 2, th / 2, &u_init);
        upload_plane(&queue, &v_tex, tw / 2, th / 2, &u_init);

        // Uniform buffer for shader params (16 bytes for alignment).
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yuv_bgl"),
            entries: &[
                bgl_texture(0),
                bgl_texture(1),
                bgl_texture(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = create_bind_group(
            &device,
            &bind_group_layout,
            &y_tex,
            &u_tex,
            &v_tex,
            &sampler,
            &params_buf,
        );

        // Render pipeline.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yuv_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("yuv_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(GpuState {
            device,
            queue,
            surface,
            surface_config,
            render_format,
            pipeline,
            bind_group_layout,
            bind_group,
            sampler,
            y_tex,
            u_tex,
            v_tex,
            params_buf,
            tex_width: tw,
            tex_height: th,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    fn recreate_textures(&mut self, width: u32, height: u32) {
        let w = width.max(2);
        let h = height.max(2);
        let (y, u, v) = create_yuv_textures(&self.device, w, h);
        self.y_tex = y;
        self.u_tex = u;
        self.v_tex = v;
        self.tex_width = w;
        self.tex_height = h;
        self.bind_group = create_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.y_tex,
            &self.u_tex,
            &self.v_tex,
            &self.sampler,
            &self.params_buf,
        );
    }

    fn upload_frame(&self, frame: &VideoFrame) {
        upload_plane(
            &self.queue,
            &self.y_tex,
            frame.width,
            frame.height,
            &frame.y_data,
        );
        upload_plane(
            &self.queue,
            &self.u_tex,
            frame.width / 2,
            frame.height / 2,
            &frame.u_data,
        );
        upload_plane(
            &self.queue,
            &self.v_tex,
            frame.width / 2,
            frame.height / 2,
            &frame.v_data,
        );

        // Update color space params.
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&frame.color_matrix.to_ne_bytes());
        bytes[4..8].copy_from_slice(&frame.full_range.to_ne_bytes());
        self.queue.write_buffer(&self.params_buf, 0, &bytes);
    }
}

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

fn bgl_texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn create_yuv_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::Texture, wgpu::Texture) {
    let y = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Y"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let cw = width / 2;
    let ch = height / 2;
    let u = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("U"),
        size: wgpu::Extent3d {
            width: cw,
            height: ch,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let v = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("V"),
        size: wgpu::Extent3d {
            width: cw,
            height: ch,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    (y, u, v)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    y_tex: &wgpu::Texture,
    u_tex: &wgpu::Texture,
    v_tex: &wgpu::Texture,
    sampler: &wgpu::Sampler,
    params_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let y_view = y_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let u_view = u_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let v_view = v_tex.create_view(&wgpu::TextureViewDescriptor::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("yuv_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&y_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&u_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&v_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    })
}

fn upload_plane(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    data: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

fn compute_viewport(vid_w: u32, vid_h: u32, win_w: u32, win_h: u32) -> (f32, f32, f32, f32) {
    if vid_w == 0 || vid_h == 0 || win_w == 0 || win_h == 0 {
        return (0.0, 0.0, win_w as f32, win_h as f32);
    }
    let scale = (win_w as f32 / vid_w as f32).min(win_h as f32 / vid_h as f32);
    let dst_w = vid_w as f32 * scale;
    let dst_h = vid_h as f32 * scale;
    let x = (win_w as f32 - dst_w) / 2.0;
    let y = (win_h as f32 - dst_h) / 2.0;
    (x, y, dst_w, dst_h)
}

// ---------------------------------------------------------------------------
// YUV plane extraction
// ---------------------------------------------------------------------------

/// Extract a plane from a decoded frame, stripping stride padding if necessary.
fn extract_plane(plane: &FramePlane, width: usize, height: usize) -> Vec<u8> {
    let data = plane.buffer.data();
    let start = plane.offset;
    let linesize = plane.linesize;

    if linesize == width && start + width * height <= data.len() {
        // Contiguous: slice directly.
        data[start..start + width * height].to_vec()
    } else {
        // Stride padding: copy row by row.
        let mut out = Vec::with_capacity(width * height);
        for row in 0..height {
            let row_start = start + row * linesize;
            let row_end = row_start + width;
            if row_end <= data.len() {
                out.extend_from_slice(&data[row_start..row_end]);
            } else {
                out.resize(out.len() + width, 0);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Read thread — reads packets, dispatches to per-stream decode threads.
// Matches ffplay's read_thread (lines 2878-3224).
// ---------------------------------------------------------------------------

fn read_thread_fn(args: ReadThreadArgs) {
    let ReadThreadArgs {
        mut ctx,
        video_idx,
        audio_idx,
        video_pkt_tx,
        audio_pkt_tx,
        event_tx,
        cmd_rx,
        video_pkt_count,
        audio_pkt_count,
    } = args;
    let mut eof = false;

    /// Minimum buffered packets before read thread considers throttling
    /// (matches ffplay's MIN_FRAMES = 25).
    const MIN_FRAMES: u64 = 25;

    loop {
        // Check for commands.
        let cmd = if eof {
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
            }
        }

        // Self-throttle: if both streams have enough buffered packets,
        // sleep 10ms to avoid unbounded queue growth. Matches ffplay's
        // stream_has_enough_packets check (lines 2854, 3131-3141).
        let video_enough =
            video_idx.is_none() || video_pkt_count.load(Ordering::Relaxed) > MIN_FRAMES;
        let audio_enough =
            audio_idx.is_none() || audio_pkt_count.load(Ordering::Relaxed) > MIN_FRAMES;
        if video_enough && audio_enough {
            thread::sleep(Duration::from_millis(10));
            continue; // re-check commands after sleep
        }

        // Read one packet and dispatch to the correct decode thread.
        match ctx.read_packet() {
            Ok(packet) => {
                let idx = packet.stream_index;
                if Some(idx) == audio_idx {
                    audio_pkt_count.fetch_add(1, Ordering::Relaxed);
                    if let Some(ref tx) = audio_pkt_tx
                        && tx.send(PacketMsg::Packet(packet)).is_err()
                    {
                        break;
                    }
                } else if Some(idx) == video_idx
                    && let Some(ref tx) = video_pkt_tx
                {
                    video_pkt_count.fetch_add(1, Ordering::Relaxed);
                    if tx.send(PacketMsg::Packet(packet)).is_err() {
                        break;
                    }
                }
            }
            Err(Error::Eof) => {
                // Signal EOF to decode threads by dropping the senders.
                // (They'll see Disconnected on recv.)
                eof = true;
                let _ = event_tx.send(DecodeEvent::Eof);
            }
            Err(_) => {
                eof = true;
                let _ = event_tx.send(DecodeEvent::Eof);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Video decode thread — decodes video packets into YUV frames.
// Matches ffplay's video_thread (lines 2216-2317).
// Blocks only on its OWN output queue — never stalls audio.
// ---------------------------------------------------------------------------

fn video_decode_thread(args: VideoDecodeArgs) {
    let VideoDecodeArgs {
        mut decoder,
        pkt_rx,
        frame_tx,
        vid_time_base,
        pkt_count,
    } = args;

    loop {
        match pkt_rx.recv() {
            Ok(PacketMsg::Packet(packet)) => {
                pkt_count.fetch_sub(1, Ordering::Relaxed);
                if decoder.send_packet(Some(&packet)).is_ok() {
                    drain_video_frames(&mut *decoder, vid_time_base, &frame_tx);
                }
            }
            Err(mpsc::RecvError) => {
                // Channel closed (EOF or quit) — flush decoder.
                let _ = decoder.send_packet(None);
                drain_video_frames(&mut *decoder, vid_time_base, &frame_tx);
                break;
            }
        }
    }
}

/// Drain all video frames from a decoder and send via channel.
fn drain_video_frames(
    decoder: &mut dyn wedeo_codec::decoder::Decoder,
    vid_time_base: Rational,
    frame_tx: &mpsc::SyncSender<VideoFrame>,
) {
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                let video = match frame.video() {
                    Some(v) => v,
                    None => break,
                };

                if video.format != PixelFormat::Yuv420p && video.format != PixelFormat::Yuvj420p {
                    continue;
                }

                if video.planes.len() < 3 {
                    continue;
                }

                let w = video.width as usize;
                let h = video.height as usize;
                let y_data = extract_plane(&video.planes[0], w, h);
                let u_data = extract_plane(&video.planes[1], w / 2, h / 2);
                let v_data = extract_plane(&video.planes[2], w / 2, h / 2);

                let color_matrix = if video.color_space == ColorSpace::Bt709 {
                    1
                } else {
                    0
                };
                let full_range = if video.color_range == ColorRange::Jpeg
                    || video.format == PixelFormat::Yuvj420p
                {
                    1
                } else {
                    0
                };

                let pts_sec = pts_to_sec(frame.pts, vid_time_base);
                let vf = VideoFrame {
                    y_data,
                    u_data,
                    v_data,
                    width: video.width,
                    height: video.height,
                    pts_sec,
                    color_matrix,
                    full_range,
                };
                // Blocking send — this thread only handles video, so blocking
                // here is fine. It provides backpressure on decode without
                // affecting audio. (Matches ffplay's frame_queue_peek_writable.)
                if frame_tx.send(vf).is_err() {
                    return;
                }
            }
            Err(Error::Again | Error::Eof) => break,
            Err(_) => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Audio decode thread — decodes audio packets into f32 samples.
// Matches ffplay's audio_thread (lines 2127-2203).
// Blocks only on its OWN output queue — never stalls video.
// ---------------------------------------------------------------------------

fn audio_decode_thread(args: AudioDecodeArgs) {
    let AudioDecodeArgs {
        mut decoder,
        pkt_rx,
        frame_tx,
        audio_time_base,
        src_sample_rate,
        device_sample_rate,
        device_channels,
        pkt_count,
    } = args;

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

    loop {
        match pkt_rx.recv() {
            Ok(PacketMsg::Packet(packet)) => {
                pkt_count.fetch_sub(1, Ordering::Relaxed);
                if decoder.send_packet(Some(&packet)).is_ok() {
                    drain_audio_frames(
                        &mut *decoder,
                        &mut resampler,
                        device_channels,
                        audio_time_base,
                        &frame_tx,
                    );
                }
            }
            Err(mpsc::RecvError) => {
                // Channel closed (EOF or quit) — flush decoder + resampler.
                let _ = decoder.send_packet(None);
                drain_audio_frames(
                    &mut *decoder,
                    &mut resampler,
                    device_channels,
                    audio_time_base,
                    &frame_tx,
                );
                if let Some(ref mut rs) = resampler
                    && let Ok(tail) = rs.flush()
                    && !tail.is_empty()
                {
                    let _ = frame_tx.send(AudioData {
                        samples: tail,
                        pts_sec: 0.0,
                    });
                }
                break;
            }
        }
    }
}

/// Drain all audio frames from a decoder and send via channel.
fn drain_audio_frames(
    decoder: &mut dyn wedeo_codec::decoder::Decoder,
    resampler: &mut Option<Resampler>,
    dst_channels: usize,
    audio_time_base: Rational,
    frame_tx: &mpsc::SyncSender<AudioData>,
) {
    loop {
        match decoder.receive_frame() {
            Ok(frame) => {
                let pts_sec = pts_to_sec(frame.pts, audio_time_base);
                match decode_audio_to_f32(&frame, resampler, dst_channels) {
                    Ok(samples) => {
                        if frame_tx.send(AudioData { samples, pts_sec }).is_err() {
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

// ---------------------------------------------------------------------------
// Time conversion helpers
// ---------------------------------------------------------------------------

fn pts_to_sec(pts: i64, time_base: Rational) -> f64 {
    if pts >= 0 && time_base.num > 0 && time_base.den > 0 {
        pts as f64 * time_base.num as f64 / time_base.den as f64
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Audio conversion helpers (unchanged from original)
// ---------------------------------------------------------------------------

fn decode_audio_to_f32(
    frame: &Frame,
    resampler: &mut Option<Resampler>,
    dst_channels: usize,
) -> wedeo_core::Result<Vec<f32>> {
    let audio = frame.audio().ok_or(Error::InvalidData)?;
    let channels = audio.channel_layout.nb_channels as usize;
    let nb_samples = audio.nb_samples as usize;

    let f32_samples = if audio.format.is_planar() && audio.planes.len() > 1 {
        planar_to_f32(&audio.planes, audio.format, nb_samples, channels)
    } else {
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
            out.push(samples[s]);
            out.push(samples[s]);
        } else if src_ch == 6 && dst_ch == 2 {
            let fl = samples[s];
            let fr = samples[s + 1];
            let c = samples[s + 2];
            let sl = samples[s + 4];
            let sr = samples[s + 5];
            out.push(fl + MIX * c + MIX * sl);
            out.push(fr + MIX * c + MIX * sr);
        } else if src_ch > dst_ch {
            for c in 0..dst_ch {
                out.push(samples[s + c]);
            }
        } else {
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
