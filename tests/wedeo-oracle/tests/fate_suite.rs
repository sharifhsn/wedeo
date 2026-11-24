//! Official FFmpeg FATE Conformance Suite Runner & Scoreboard.
//!
//! Run against a synced FFmpeg FATE suite directory (e.g. `target/fate-suite`):
//!
//! ```sh
//! rsync -az rsync://fate-suite.ffmpeg.org/fate-suite/vp9-test-vectors/ target/fate-suite/vp9-test-vectors/
//! rsync -az rsync://fate-suite.ffmpeg.org/fate-suite/h264-conformance/ target/fate-suite/h264-conformance/
//! rsync -az rsync://fate-suite.ffmpeg.org/fate-suite/av1-test-vectors/ target/fate-suite/av1-test-vectors/
//! rsync -az rsync://fate-suite.ffmpeg.org/fate-suite/wav/ target/fate-suite/wav/
//! FATE_SUITE=target/fate-suite cargo test --release -p wedeo-oracle --test fate_suite -- --ignored --nocapture
//! ```
//!
//! All tests are `#[ignore]` so offline `cargo test --workspace` stays fast and self-contained.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use wedeo_oracle::{
    fate_suite_dir, ffmpeg_oracle_bin, parse_framecrc_record, run_ffmpeg_framecrc,
    run_wedeo_framecrc, temp_test_dir, workspace_root,
};

fn resolve_fate_root() -> Option<PathBuf> {
    let primary = fate_suite_dir();
    if primary.is_dir() {
        return Some(primary);
    }
    let fallback = workspace_root().join("target").join("fate-suite");
    if fallback.is_dir() {
        return Some(fallback);
    }
    None
}

fn collect_files(dir: &Path, exts: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&cur) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.is_file() {
                let ext = p
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.to_ascii_lowercase())
                    .unwrap_or_default();
                if exts.iter().any(|wanted| *wanted == ext) {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

/// Extract ordered `(size, crc)` video frame hashes from `framecrc` lines (stream 0).
fn extract_video_crcs(lines: &[String]) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Ok(rec) = parse_framecrc_record(trimmed)
            && rec.stream_index == 0
        {
            out.push((rec.size, rec.crc));
        }
    }
    out
}

fn compare_video_crcs(wedeo: &[(usize, String)], ffmpeg: &[(usize, String)]) -> Result<(), String> {
    if wedeo.len() != ffmpeg.len() {
        return Err(format!(
            "frame count mismatch: wedeo={} vs ffmpeg={}",
            wedeo.len(),
            ffmpeg.len()
        ));
    }
    for (idx, (w, f)) in wedeo.iter().zip(ffmpeg.iter()).enumerate() {
        if w != f {
            return Err(format!(
                "frame #{idx} mismatch: wedeo=(size={}, crc={}) vs ffmpeg=(size={}, crc={})",
                w.0, w.1, f.0, f.1
            ));
        }
    }
    Ok(())
}

fn run_ffmpeg_video_only_framecrc(path: &Path) -> Result<Vec<String>, String> {
    let bin = ffmpeg_oracle_bin();
    let out = Command::new(&bin)
        .args([
            "-v",
            "error",
            "-i",
            path.to_str().ok_or("non-utf8 path")?,
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "framecrc",
            "-",
        ])
        .output()
        .map_err(|e| format!("spawn ffmpeg: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(ToString::to_string)
        .collect())
}

#[derive(Default)]
struct Scoreboard {
    pass: Vec<String>,
    unsupported: Vec<(String, String)>,
    mismatch: Vec<(String, String)>,
    decode_err: Vec<(String, String)>,
    oracle_err: Vec<(String, String)>,
}

impl Scoreboard {
    fn print_summary(&self, title: &str) {
        let total =
            self.pass.len() + self.unsupported.len() + self.mismatch.len() + self.decode_err.len();
        let supported = self.pass.len() + self.mismatch.len() + self.decode_err.len();
        let pct = if supported > 0 {
            100.0 * self.pass.len() as f64 / supported as f64
        } else {
            0.0
        };
        println!("\n==================== {title} ====================");
        println!(
            "Total evaluated: {total} | Supported: {supported} | Bit-Exact PASS: {} ({pct:.1}%)",
            self.pass.len()
        );
        println!(
            "  Unsupported (cleanly rejected): {} | Pixel mismatch: {} | Decode error: {} | Oracle skip: {}",
            self.unsupported.len(),
            self.mismatch.len(),
            self.decode_err.len(),
            self.oracle_err.len()
        );

        let mut err_buckets: BTreeMap<String, usize> = BTreeMap::new();
        for (_, msg) in &self.decode_err {
            let short = msg.lines().next().unwrap_or(msg).chars().take(80).collect();
            *err_buckets.entry(short).or_default() += 1;
        }
        if !err_buckets.is_empty() {
            println!("Top decode error categories:");
            for (k, v) in &err_buckets {
                println!("  [{v:3}x] {k}");
            }
        }
        if !self.mismatch.is_empty() {
            println!("First 20 pixel mismatches:");
            for (name, reason) in self.mismatch.iter().take(20) {
                println!("  MISMATCH {name}: {reason}");
            }
        }
        if !self.decode_err.is_empty() {
            println!("First 15 decode errors:");
            for (name, reason) in self.decode_err.iter().take(15) {
                println!("  ERR      {name}: {reason}");
            }
        }
        if !self.unsupported.is_empty() {
            let mut unsup_buckets: BTreeMap<String, usize> = BTreeMap::new();
            for (_, msg) in &self.unsupported {
                let short = msg.lines().next().unwrap_or(msg).chars().take(80).collect();
                *unsup_buckets.entry(short).or_default() += 1;
            }
            println!("Unsupported buckets:");
            for (k, v) in &unsup_buckets {
                println!("  [{v:3}x] {k}");
            }
        }
    }
}

#[test]
#[ignore]
fn fate_vp9_official_vectors() {
    let Some(root) = resolve_fate_root() else {
        eprintln!("FATE suite not found (set FATE_SUITE or sync to target/fate-suite)");
        return;
    };
    let dir = root.join("vp9-test-vectors");
    let files = collect_files(&dir, &["ivf", "webm"]);
    assert!(
        !files.is_empty(),
        "no VP9 vectors found in {}",
        dir.display()
    );

    let mut sb = Scoreboard::default();
    for path in files {
        let name = path
            .strip_prefix(&dir)
            .unwrap_or(&path)
            .display()
            .to_string();
        let Ok(ff_lines) = run_ffmpeg_video_only_framecrc(&path) else {
            sb.oracle_err.push((name, "ffmpeg oracle failed".into()));
            continue;
        };
        let ff_crcs = extract_video_crcs(&ff_lines);
        if ff_crcs.is_empty() {
            sb.oracle_err
                .push((name, "no video frames from oracle".into()));
            continue;
        }
        match run_wedeo_framecrc(&path) {
            Ok(w_lines) => {
                let w_crcs = extract_video_crcs(&w_lines);
                match compare_video_crcs(&w_crcs, &ff_crcs) {
                    Ok(()) => sb.pass.push(name),
                    Err(e) => sb.mismatch.push((name, e)),
                }
            }
            Err(e) if e.contains("not supported") || e.contains("unsupported") => {
                sb.unsupported.push((name, e));
            }
            Err(e) => sb.decode_err.push((name, e)),
        }
    }
    sb.print_summary("FATE VP9 Official Test Vectors");
}

#[test]
#[ignore]
fn fate_h264_official_conformance() {
    let Some(root) = resolve_fate_root() else {
        eprintln!("FATE suite not found (set FATE_SUITE or sync to target/fate-suite)");
        return;
    };
    let dir = root.join("h264-conformance");
    let files = collect_files(&dir, &["264", "h264", "mp4"]);
    assert!(
        !files.is_empty(),
        "no H.264 vectors found in {}",
        dir.display()
    );

    let tmp = temp_test_dir("fate_h264_mp4");
    let ffbin = ffmpeg_oracle_bin();
    let mut sb = Scoreboard::default();

    for (idx, path) in files.iter().enumerate() {
        let name = path
            .strip_prefix(&dir)
            .unwrap_or(path)
            .display()
            .to_string();
        // Remux raw Annex B `.264` into `.mp4` so Wedeo's MP4 demuxer feeds AVCC packets.
        let test_path = if path.extension().and_then(|e| e.to_str()) == Some("mp4") {
            path.clone()
        } else {
            let mp4 = tmp.join(format!("v_{idx:03}.mp4"));
            let st = Command::new(&ffbin)
                .args([
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    path.to_str().unwrap(),
                    "-c:v",
                    "copy",
                    mp4.to_str().unwrap(),
                ])
                .status();
            if !st.map(|s| s.success()).unwrap_or(false) {
                sb.oracle_err.push((name, "remux to mp4 failed".into()));
                continue;
            }
            mp4
        };

        let Ok(ff_lines) = run_ffmpeg_video_only_framecrc(path) else {
            sb.oracle_err.push((name, "ffmpeg oracle failed".into()));
            continue;
        };
        let ff_crcs = extract_video_crcs(&ff_lines);
        if ff_crcs.is_empty() {
            sb.oracle_err
                .push((name, "no video frames from oracle".into()));
            continue;
        }
        match run_wedeo_framecrc(&test_path) {
            Ok(w_lines) => {
                let w_crcs = extract_video_crcs(&w_lines);
                match compare_video_crcs(&w_crcs, &ff_crcs) {
                    Ok(()) => sb.pass.push(name),
                    Err(e) => sb.mismatch.push((name, e)),
                }
            }
            Err(e) if e.contains("not supported") || e.contains("unsupported") => {
                sb.unsupported.push((name, e));
            }
            Err(e) => sb.decode_err.push((name, e)),
        }
    }
    sb.print_summary("FATE H.264 Official Conformance Suite");
}

#[test]
#[ignore]
fn fate_av1_and_wav_official_vectors() {
    let Some(root) = resolve_fate_root() else {
        eprintln!("FATE suite not found");
        return;
    };

    // WAV
    let wav_dir = root.join("wav");
    if wav_dir.is_dir() {
        let mut sb = Scoreboard::default();
        for path in collect_files(&wav_dir, &["wav"]) {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            match (run_wedeo_framecrc(&path), run_ffmpeg_framecrc(&path)) {
                (Ok(w), Ok(f)) => {
                    let wc = extract_video_crcs(&w);
                    let fc = extract_video_crcs(&f);
                    if wc == fc {
                        sb.pass.push(name);
                    } else {
                        sb.mismatch
                            .push((name, format!("w={} f={}", wc.len(), fc.len())));
                    }
                }
                (Err(e), _) if e.contains("unsupported") => sb.unsupported.push((name, e)),
                (Err(e), _) => sb.decode_err.push((name, e)),
                (_, Err(e)) => sb.oracle_err.push((name, e)),
            }
        }
        sb.print_summary("FATE WAV Vectors");
    }

    // AV1 (compare against system ffmpeg w/ libdav1d since oracle has no dav1d)
    let av1_dir = root.join("av1-test-vectors");
    if av1_dir.is_dir() {
        let mut sb = Scoreboard::default();
        for path in collect_files(&av1_dir, &["ivf", "obu"]) {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let out = Command::new("/usr/bin/ffmpeg")
                .args([
                    "-v",
                    "error",
                    "-i",
                    path.to_str().unwrap(),
                    "-f",
                    "framecrc",
                    "-",
                ])
                .output();
            let Ok(out) = out else { continue };
            if !out.status.success() {
                sb.oracle_err.push((name, "system ffmpeg failed".into()));
                continue;
            }
            let ff_lines: Vec<String> = String::from_utf8_lossy(&out.stdout)
                .lines()
                .map(ToString::to_string)
                .collect();
            let ff_crcs = extract_video_crcs(&ff_lines);
            match run_wedeo_framecrc(&path) {
                Ok(w_lines) => {
                    let w_crcs = extract_video_crcs(&w_lines);
                    match compare_video_crcs(&w_crcs, &ff_crcs) {
                        Ok(()) => sb.pass.push(name),
                        Err(e) => sb.mismatch.push((name, e)),
                    }
                }
                Err(e) if e.contains("unsupported") => sb.unsupported.push((name, e)),
                Err(e) => sb.decode_err.push((name, e)),
            }
        }
        sb.print_summary("FATE AV1 Official Test Vectors");
    }
}

#[test]
#[ignore]
fn fate_live_youtube_shorts_parity() {
    let clips = [
        "/tmp/wedeo-wild/short_vp9.mkv",
        "/tmp/wedeo-wild/short_h264.mp4",
        "/tmp/wedeo-wild/bbb_cut_7m.mp4",
    ];
    let mut sb = Scoreboard::default();
    for clip in clips {
        let path = Path::new(clip);
        if !path.exists() {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let Ok(ff_lines) = run_ffmpeg_video_only_framecrc(path) else {
            sb.oracle_err.push((name, "ffmpeg oracle failed".into()));
            continue;
        };
        let ff_crcs = extract_video_crcs(&ff_lines);
        match run_wedeo_framecrc(path) {
            Ok(w_lines) => {
                let w_crcs = extract_video_crcs(&w_lines);
                match compare_video_crcs(&w_crcs, &ff_crcs) {
                    Ok(()) => {
                        println!("BIT-EXACT YOUTUBE PARITY: {name} ({} frames)", w_crcs.len());
                        sb.pass.push(name);
                    }
                    Err(e) => sb.mismatch.push((name, e)),
                }
            }
            Err(e) => sb.decode_err.push((name, e)),
        }
    }
    sb.print_summary("Live YouTube Shorts & 720p60 Bit-Exact Video Parity");
}
