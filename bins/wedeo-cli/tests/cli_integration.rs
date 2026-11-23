#![forbid(unsafe_code)]
//! End-to-end CLI & MCP subprocess integration tests for `wedeo-cli`.

use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use wedeo::{
    AudioStreamParams, Buffer, ChannelLayout, CodecId, ContainerHeader, FileSink, Packet,
    PacketFlags, Rational, SampleFormat, StreamParams, Timestamp, TrackDescriptor, TrackId,
    default_registry,
};

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let id = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let mut p = env::temp_dir();
    p.push(format!("wedeo_cli_int_{}_{}_{}", process::id(), id, name));
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

fn ffprobe_bin() -> PathBuf {
    workspace_root().join("target/ffmpeg-oracle/ffprobe")
}

fn wedeo_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_wedeo"))
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
fn cli_inspect_json_and_verify_vp9_red_ivf() {
    let ivf = vp9_red_ivf();
    assert!(ivf.exists(), "missing test vector {}", ivf.display());

    let out = Command::new(wedeo_bin())
        .args(["inspect", "--json", "--verify", "--level", "nal"])
        .arg(&ivf)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "wedeo inspect failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout_str = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout_str.contains("\"format_name\": \"ivf\""),
        "{stdout_str}"
    );
    assert!(stdout_str.contains("\"verified\": true"), "{stdout_str}");
    assert!(stdout_str.contains("\"decoded_units\": 1"), "{stdout_str}");
    assert!(stdout_str.contains("\"is_clean\": true"), "{stdout_str}");
    assert!(stdout_str.contains("\"codec\": \"vp9\""), "{stdout_str}");
}

#[test]
fn cli_transcode_and_ffmpeg_av1_to_mp4_passes_ffprobe() {
    let ivf = vp9_red_ivf();
    let out_transcode = temp_path("cli_transcode_av1.mp4");
    let out_ffmpeg = temp_path("cli_ffmpeg_av1.mp4");

    let res1 = Command::new(wedeo_bin())
        .args(["transcode", "-i"])
        .arg(&ivf)
        .args(["--video", "av1", "-y", "-o"])
        .arg(&out_transcode)
        .output()
        .unwrap();
    assert!(
        res1.status.success(),
        "wedeo transcode failed: {}",
        String::from_utf8_lossy(&res1.stderr)
    );

    let res2 = Command::new(wedeo_bin())
        .args(["ffmpeg", "-i"])
        .arg(&ivf)
        .args(["-c:v", "av1", "-y"])
        .arg(&out_ffmpeg)
        .output()
        .unwrap();
    assert!(
        res2.status.success(),
        "wedeo ffmpeg failed: {}",
        String::from_utf8_lossy(&res2.stderr)
    );

    let ffprobe = ffprobe_bin();
    assert!(
        ffprobe.exists(),
        "expected ffprobe oracle at {}",
        ffprobe.display()
    );
    for mp4 in [&out_transcode, &out_ffmpeg] {
        let probe = Command::new(&ffprobe)
            .args([
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,width,height,pix_fmt",
                "-show_entries",
                "format=format_name",
                "-of",
                "default=noprint_wrappers=1",
            ])
            .arg(mp4)
            .output()
            .unwrap();
        assert!(
            probe.status.success(),
            "ffprobe failed on {}: {}",
            mp4.display(),
            String::from_utf8_lossy(&probe.stderr)
        );
        let txt = String::from_utf8_lossy(&probe.stdout);
        assert!(txt.contains("codec_name=av1"), "ffprobe output:\n{txt}");
        assert!(txt.contains("width=64"), "ffprobe output:\n{txt}");
        assert!(txt.contains("height=64"), "ffprobe output:\n{txt}");
        assert!(
            txt.contains("format_name=mov,mp4"),
            "ffprobe output:\n{txt}"
        );
    }

    let _ = std::fs::remove_file(&out_transcode);
    let _ = std::fs::remove_file(&out_ffmpeg);
}

#[test]
fn cli_explain_json_reports_auto_sws_bridge_without_writing_file() {
    let ivf = vp9_red_ivf();
    let dummy_out = temp_path("explain_never_written.mp4");

    let out = Command::new(wedeo_bin())
        .args(["explain", "--json", "-i"])
        .arg(&ivf)
        .args(["--video", "av1", "--size", "32x32", "-o"])
        .arg(&dummy_out)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "wedeo explain failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!dummy_out.exists());
    let stdout_str = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout_str.contains("\"name\": \"sws_op_list\""),
        "expected sws_op_list in explain output:\n{stdout_str}"
    );
    assert!(
        stdout_str.contains("\"is_lossy\": true"),
        "expected lossy note in explain output:\n{stdout_str}"
    );
}

#[test]
fn cli_mcp_stdio_initialize_tools_list_and_inspect_call() {
    let ivf = vp9_red_ivf();
    let mut child = Command::new(wedeo_bin())
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let mut stdin = child.stdin.take().unwrap();
        writeln!(
            stdin,
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{{}}}}"
        )
        .unwrap();
        writeln!(
            stdin,
            "{{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}}"
        )
        .unwrap();
        writeln!(
            stdin,
            "{{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}}"
        )
        .unwrap();
        let escaped = ivf
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        writeln!(
            stdin,
            "{{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{{\"name\":\"inspect\",\"arguments\":{{\"input\":\"{escaped}\",\"verify\":true}}}}}}"
        )
        .unwrap();
    }

    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let stdout_str = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = stdout_str
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(lines.len(), 3, "got:\n{stdout_str}");
    assert!(lines[0].contains("\"protocolVersion\""), "{}", lines[0]);
    assert!(lines[1].contains("\"inspect\""), "{}", lines[1]);
    assert!(lines[1].contains("\"explain\""), "{}", lines[1]);
    assert!(lines[1].contains("\"transcode\""), "{}", lines[1]);
    assert!(lines[1].contains("\"cut\""), "{}", lines[1]);
    assert!(lines[1].contains("\"mux\""), "{}", lines[1]);
    assert!(lines[2].contains("\"isError\":false"), "{}", lines[2]);
    assert!(lines[2].contains("vp9"), "{}", lines[2]);
}

#[test]
fn cli_mux_and_ffmpeg_vn_stream_index_remapping() {
    let ivf = vp9_red_ivf();
    let wav = temp_path("cli_audio.wav");
    let muxed = temp_path("cli_muxed.mkv");
    let audio_only = temp_path("cli_audio_only.mkv");

    create_synthetic_wav(&wav, 48_000, 4800);

    let res_mux = Command::new(wedeo_bin())
        .args(["mux", "--video"])
        .arg(&ivf)
        .args(["--audio"])
        .arg(&wav)
        .args(["--tag", "title=CliMux", "-y", "-o"])
        .arg(&muxed)
        .output()
        .unwrap();
    assert!(
        res_mux.status.success(),
        "wedeo mux failed: {}",
        String::from_utf8_lossy(&res_mux.stderr)
    );

    let res_vn = Command::new(wedeo_bin())
        .args(["ffmpeg", "-i"])
        .arg(&muxed)
        .args(["-vn", "-c:a", "copy", "-y"])
        .arg(&audio_only)
        .output()
        .unwrap();
    assert!(
        res_vn.status.success(),
        "wedeo ffmpeg -vn failed: {}",
        String::from_utf8_lossy(&res_vn.stderr)
    );

    let res_ins = Command::new(wedeo_bin())
        .args(["inspect", "--json", "--verify"])
        .arg(&audio_only)
        .output()
        .unwrap();
    assert!(res_ins.status.success());
    let txt = String::from_utf8(res_ins.stdout).unwrap();
    assert!(txt.contains("\"media_type\": \"audio\""), "{txt}");
    assert!(!txt.contains("\"media_type\": \"video\""), "{txt}");
    assert!(txt.contains("\"is_clean\": true"), "{txt}");

    let _ = std::fs::remove_file(&wav);
    let _ = std::fs::remove_file(&muxed);
    let _ = std::fs::remove_file(&audio_only);
}
