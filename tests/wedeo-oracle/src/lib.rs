//! Shared FFmpeg `n9.1` verification oracle helpers, `checkasm` differential
//! harness, and `CountingAlloc` steady-state allocation gate for `wedeo-oracle`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default path to the pinned FFmpeg `n9.1` oracle binary built in-tree.
pub const PINNED_FFMPEG_ORACLE: &str = "/home/sharif/wedeo/target/ffmpeg-oracle/ffmpeg";

/// Default path to the pinned FFmpeg `n9.1` `ffprobe` oracle binary built in-tree.
pub const PINNED_FFPROBE_ORACLE: &str = "/home/sharif/wedeo/target/ffmpeg-oracle/ffprobe";

// ---------------------------------------------------------------------------
// CountingAlloc: thread-local + reentrancy-safe DOD allocation tracker
// ---------------------------------------------------------------------------

/// Allocation statistics captured during a `measure_allocs` window on the current thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AllocStats {
    pub allocs: usize,
    pub deallocs: usize,
    pub reallocs: usize,
    pub bytes_allocated: usize,
}

thread_local! {
    static THREAD_ALLOC_STATS: Cell<AllocStats> = const {
        Cell::new(AllocStats {
            allocs: 0,
            deallocs: 0,
            reallocs: 0,
            bytes_allocated: 0,
        })
    };
}

/// Global allocator wrapper over [`System`] tracking per-thread heap allocations
/// with zero allocation inside the hook itself.
pub struct CountingAlloc;

impl CountingAlloc {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for CountingAlloc {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: Delegates all allocations directly to `std::alloc::System` and updates
// a `const`-initialized thread-local `Cell` without allocating.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        let _ = THREAD_ALLOC_STATS.try_with(|c| {
            let mut st = c.get();
            st.allocs = st.allocs.saturating_add(1);
            st.bytes_allocated = st.bytes_allocated.saturating_add(layout.size());
            c.set(st);
        });
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        let _ = THREAD_ALLOC_STATS.try_with(|c| {
            let mut st = c.get();
            st.allocs = st.allocs.saturating_add(1);
            st.bytes_allocated = st.bytes_allocated.saturating_add(layout.size());
            c.set(st);
        });
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = THREAD_ALLOC_STATS.try_with(|c| {
            let mut st = c.get();
            st.deallocs = st.deallocs.saturating_add(1);
            c.set(st);
        });
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        let _ = THREAD_ALLOC_STATS.try_with(|c| {
            let mut st = c.get();
            st.reallocs = st.reallocs.saturating_add(1);
            if new_size > layout.size() {
                st.bytes_allocated = st.bytes_allocated.saturating_add(new_size - layout.size());
            }
            c.set(st);
        });
        new_ptr
    }
}

#[global_allocator]
static GLOBAL_COUNTING_ALLOC: CountingAlloc = CountingAlloc::new();

/// Snapshot the current thread's cumulative allocation counters.
pub fn thread_alloc_stats() -> AllocStats {
    THREAD_ALLOC_STATS.with(Cell::get)
}

/// Measure heap allocations performed on the current thread while running `f`.
pub fn measure_allocs<F: FnOnce() -> R, R>(f: F) -> (R, AllocStats) {
    let before = thread_alloc_stats();
    let result = f();
    let after = thread_alloc_stats();
    let delta = AllocStats {
        allocs: after.allocs.saturating_sub(before.allocs),
        deallocs: after.deallocs.saturating_sub(before.deallocs),
        reallocs: after.reallocs.saturating_sub(before.reallocs),
        bytes_allocated: after.bytes_allocated.saturating_sub(before.bytes_allocated),
    };
    (result, delta)
}

/// Assert that `f` performs zero heap allocations (`allocs == 0 && reallocs == 0`).
pub fn assert_zero_allocs<F: FnOnce() -> R, R>(label: &str, f: F) -> R {
    let (res, stats) = measure_allocs(f);
    assert!(
        stats.allocs == 0 && stats.reallocs == 0,
        "{label}: expected 0 steady-state allocations, observed {stats:?}"
    );
    res
}

// ---------------------------------------------------------------------------
// Pinned FFmpeg n9.1 oracle binary resolution
// ---------------------------------------------------------------------------

/// Resolve the FFmpeg `n9.1` oracle binary:
/// 1. `/home/sharif/wedeo/target/ffmpeg-oracle/ffmpeg`
/// 2. `WEDEO_FFMPEG_BIN` environment variable
/// 3. `"ffmpeg"` on `PATH`
pub fn ffmpeg_oracle_bin() -> PathBuf {
    let pinned = Path::new(PINNED_FFMPEG_ORACLE);
    if pinned.is_file() {
        return pinned.to_path_buf();
    }
    if let Ok(env_bin) = std::env::var("WEDEO_FFMPEG_BIN")
        && !env_bin.trim().is_empty()
    {
        return PathBuf::from(env_bin);
    }
    PathBuf::from("ffmpeg")
}

/// Resolve the FFmpeg `n9.1` `ffprobe` oracle binary:
/// 1. `/home/sharif/wedeo/target/ffmpeg-oracle/ffprobe`
/// 2. `WEDEO_FFPROBE_BIN` environment variable
/// 3. `"ffprobe"` on `PATH`
pub fn ffprobe_oracle_bin() -> PathBuf {
    let pinned = Path::new(PINNED_FFPROBE_ORACLE);
    if pinned.is_file() {
        return pinned.to_path_buf();
    }
    if let Ok(env_bin) = std::env::var("WEDEO_FFPROBE_BIN")
        && !env_bin.trim().is_empty()
    {
        return PathBuf::from(env_bin);
    }
    PathBuf::from("ffprobe")
}

/// Resolve a system FFmpeg binary with external encoders (`libx264`, `libsvtav1`,
/// `libmp3lame`, `libvorbis`) for generating test fixtures.
pub fn system_ffmpeg_bin() -> PathBuf {
    if let Ok(env_bin) = std::env::var("WEDEO_SYSTEM_FFMPEG_BIN")
        && !env_bin.trim().is_empty()
    {
        return PathBuf::from(env_bin);
    }
    let usr_bin = Path::new("/usr/bin/ffmpeg");
    if usr_bin.is_file() {
        return usr_bin.to_path_buf();
    }
    ffmpeg_oracle_bin()
}

// ---------------------------------------------------------------------------
// Workspace / sample / temp helpers
// ---------------------------------------------------------------------------

/// Workspace root (`/home/sharif/wedeo`).
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tests/ dir")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Bundled oracle test data directory (`tests/wedeo-oracle/data`).
pub fn oracle_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// Create and return an isolated temporary directory under `std::env::temp_dir()`.
pub fn temp_test_dir(subdir: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("wedeo-oracle-tests").join(subdir);
    std::fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

/// Location of the optional FATE sample suite (set via `FATE_SUITE` or `./fate-suite`).
pub fn fate_suite_dir() -> PathBuf {
    std::env::var("FATE_SUITE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root().join("fate-suite"))
}

/// Check if a FATE sample file exists.
pub fn fate_sample(relative_path: &str) -> Option<PathBuf> {
    let path = fate_suite_dir().join(relative_path);
    if path.exists() { Some(path) } else { None }
}

/// Compute MD5 hex digest of a file.
pub fn md5_file(path: &Path) -> std::io::Result<String> {
    let data = std::fs::read(path)?;
    Ok(md5_bytes(&data))
}

/// Compute MD5 hex digest of raw bytes.
pub fn md5_bytes(data: &[u8]) -> String {
    use md5::Digest;
    let result = md5::Md5::digest(data);
    result
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

/// FFmpeg-compatible Adler-32 (`s1=0, s2=0` init).
pub fn adler32_ffmpeg(data: &[u8]) -> u32 {
    wedeo::core::adler32_ffmpeg(0, data)
}

/// Standard RFC 1950 Adler-32 (`s1=1, s2=0` init).
pub fn adler32_compute(data: &[u8]) -> u32 {
    let mut hasher = adler2::Adler32::new();
    hasher.write_slice(data);
    hasher.checksum()
}

// ---------------------------------------------------------------------------
// framecrc runners & strict comparator
// ---------------------------------------------------------------------------

/// Find a sibling binary compiled by Cargo in `wedeo-oracle` (e.g. `wedeo-framecrc`).
pub fn find_oracle_bin(name: &str, env_var: Option<&str>) -> Result<PathBuf, String> {
    if let Some(var) = env_var
        && let Ok(val) = std::env::var(var)
    {
        let p = PathBuf::from(&val);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        // Integration test executables live in `<target>/<profile>/deps/`
        let direct = dir.join(name);
        if direct.is_file() {
            return Ok(direct);
        }
        if let Some(profile_dir) = dir.parent() {
            let sibling = profile_dir.join(name);
            if sibling.is_file() {
                return Ok(sibling);
            }
        }
    }
    Err(format!("oracle binary '{name}' not found"))
}

/// Run `wedeo-framecrc` on `path` and return stdout lines.
pub fn run_wedeo_framecrc(path: &Path) -> Result<Vec<String>, String> {
    let bin = find_oracle_bin("wedeo-framecrc", Some("CARGO_BIN_EXE_wedeo-framecrc"))?;
    let output = Command::new(&bin)
        .arg(path)
        .output()
        .map_err(|e| format!("Failed to run {}: {e}", bin.display()))?;

    if !output.status.success() {
        return Err(format!(
            "wedeo-framecrc failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().map(ToOwned::to_owned).collect())
}

/// Run the pinned FFmpeg `n9.1` oracle in `framecrc` mode on `path`.
pub fn run_ffmpeg_framecrc(path: &Path) -> Result<Vec<String>, String> {
    let bin = ffmpeg_oracle_bin();
    let output = Command::new(&bin)
        .args([
            "-v",
            "error",
            "-bitexact",
            "-i",
            path.to_str().ok_or("non-UTF8 path")?,
            "-f",
            "framecrc",
            "-flags",
            "+bitexact",
            "-fflags",
            "+bitexact",
            "-",
        ])
        .output()
        .map_err(|e| format!("Failed to run {}: {e}", bin.display()))?;

    if !output.status.success() {
        return Err(format!(
            "{} framecrc failed for {}: {}",
            bin.display(),
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().map(ToOwned::to_owned).collect())
}

/// Run `ffmpeg` oracle in stream-copy (`-c copy -f framecrc`) mode to inspect demuxer packets.
pub fn run_ffmpeg_packet_framecrc(path: &Path) -> Result<Vec<String>, String> {
    let bin = ffmpeg_oracle_bin();
    let output = Command::new(&bin)
        .args([
            "-v",
            "error",
            "-bitexact",
            "-i",
            path.to_str().ok_or("non-UTF8 path")?,
            "-c",
            "copy",
            "-f",
            "framecrc",
            "-flags",
            "+bitexact",
            "-fflags",
            "+bitexact",
            "-",
        ])
        .output()
        .map_err(|e| format!("Failed to run {}: {e}", bin.display()))?;

    if !output.status.success() {
        return Err(format!(
            "{} -c copy framecrc failed for {}: {}",
            bin.display(),
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().map(ToOwned::to_owned).collect())
}

/// Parsed data line from `framecrc` output (`stream_index, dts, pts, duration, size, crc`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramecrcRecord {
    pub stream_index: usize,
    pub dts: i64,
    pub pts: i64,
    pub duration: i64,
    pub size: usize,
    pub crc: String,
    pub flags: Option<String>,
}

/// Parsed stream header metadata from `framecrc` `#...` comment lines.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FramecrcStreamHeader {
    pub tb: Option<String>,
    pub media_type: Option<String>,
    pub codec_id: Option<String>,
    pub dimensions: Option<String>,
    pub sar: Option<String>,
    pub sample_rate: Option<String>,
    pub channel_layout_name: Option<String>,
}

/// Comparison strictness options for [`compare_framecrc_full`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FramecrcCompareOpts {
    /// Enforce exact `dts`, `pts`, and `duration` on every packet/frame.
    pub check_timestamps: bool,
    /// Enforce `#tb`, `#media_type`, `#codec_id`, `#dimensions`, `#sample_rate`, `#channel_layout_name`.
    pub check_headers: bool,
    /// Enforce `#sar` header line match as well.
    pub check_sar: bool,
}

impl FramecrcCompareOpts {
    /// Full strict parity across all 6 columns and all headers (including `#sar`).
    pub const STRICT_ALL: Self = Self {
        check_timestamps: true,
        check_headers: true,
        check_sar: true,
    };

    /// Full strict parity across all 6 columns and headers, ignoring `#sar` (`0/1` vs `1/1`).
    pub const STRICT_TIMING_AND_HEADERS: Self = Self {
        check_timestamps: true,
        check_headers: true,
        check_sar: false,
    };

    /// Strict 6-column packet/frame parity (`stream, dts, pts, duration, size, crc`), skipping headers.
    pub const STRICT_6COL: Self = Self {
        check_timestamps: true,
        check_headers: false,
        check_sar: false,
    };

    /// Stream index, byte size, and CRC-only comparison (useful when raw streams lack container timing).
    pub const DATA_ONLY: Self = Self {
        check_timestamps: false,
        check_headers: false,
        check_sar: false,
    };
}

/// Parse a single non-comment `framecrc` data line into [`FramecrcRecord`].
pub fn parse_framecrc_record(line: &str) -> Result<FramecrcRecord, String> {
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    if fields.len() < 6 {
        return Err(format!("malformed framecrc line (<6 fields): {line}"));
    }
    let stream_index = fields[0]
        .parse::<usize>()
        .map_err(|e| format!("bad stream_index in '{line}': {e}"))?;
    let dts = fields[1]
        .parse::<i64>()
        .map_err(|e| format!("bad dts in '{line}': {e}"))?;
    let pts = fields[2]
        .parse::<i64>()
        .map_err(|e| format!("bad pts in '{line}': {e}"))?;
    let duration = fields[3]
        .parse::<i64>()
        .map_err(|e| format!("bad duration in '{line}': {e}"))?;
    let size = fields[4]
        .parse::<usize>()
        .map_err(|e| format!("bad size in '{line}': {e}"))?;
    let crc = fields[5].to_string();
    let flags = fields
        .iter()
        .skip(6)
        .find(|f| f.starts_with("F="))
        .map(|s| (*s).to_string());

    Ok(FramecrcRecord {
        stream_index,
        dts,
        pts,
        duration,
        size,
        crc,
        flags,
    })
}

/// Parse `#tb`, `#media_type`, `#codec_id`, `#dimensions`, `#sar`, `#sample_rate`, `#channel_layout_name`.
pub fn parse_framecrc_headers(lines: &[String]) -> Vec<FramecrcStreamHeader> {
    let mut headers: Vec<FramecrcStreamHeader> = Vec::new();

    for line in lines {
        let Some(rest) = line.strip_prefix('#') else {
            continue;
        };
        let Some((key_idx, val)) = rest.split_once(':') else {
            continue;
        };
        let mut parts = key_idx.split_whitespace();
        let Some(key) = parts.next() else {
            continue;
        };
        let Some(idx_str) = parts.next() else {
            continue;
        };
        let Ok(idx) = idx_str.parse::<usize>() else {
            continue;
        };
        if headers.len() <= idx {
            headers.resize_with(idx + 1, FramecrcStreamHeader::default);
        }
        let val = val.trim().to_string();
        match key {
            "tb" => headers[idx].tb = Some(val),
            "media_type" => headers[idx].media_type = Some(val),
            "codec_id" => headers[idx].codec_id = Some(val),
            "dimensions" => headers[idx].dimensions = Some(val),
            "sar" => headers[idx].sar = Some(val),
            "sample_rate" => headers[idx].sample_rate = Some(val),
            "channel_layout_name" => headers[idx].channel_layout_name = Some(val),
            _ => {}
        }
    }

    headers
}

/// Strictly compare `wedeo` and `ffmpeg` `framecrc` outputs under [`FramecrcCompareOpts`].
pub fn compare_framecrc_full(
    wedeo: &[String],
    ffmpeg: &[String],
    opts: FramecrcCompareOpts,
) -> Result<(), String> {
    if opts.check_headers {
        let w_hdrs = parse_framecrc_headers(wedeo);
        let f_hdrs = parse_framecrc_headers(ffmpeg);
        if w_hdrs.len() != f_hdrs.len() {
            return Err(format!(
                "Stream header count differs: wedeo={}, ffmpeg={}",
                w_hdrs.len(),
                f_hdrs.len()
            ));
        }
        for (idx, (wh, fh)) in w_hdrs.iter().zip(f_hdrs.iter()).enumerate() {
            if wh.tb != fh.tb {
                return Err(format!(
                    "Stream {idx} #tb mismatch: wedeo={:?}, ffmpeg={:?}",
                    wh.tb, fh.tb
                ));
            }
            if wh.media_type != fh.media_type {
                return Err(format!(
                    "Stream {idx} #media_type mismatch: wedeo={:?}, ffmpeg={:?}",
                    wh.media_type, fh.media_type
                ));
            }
            if wh.codec_id != fh.codec_id {
                return Err(format!(
                    "Stream {idx} #codec_id mismatch: wedeo={:?}, ffmpeg={:?}",
                    wh.codec_id, fh.codec_id
                ));
            }
            if wh.dimensions != fh.dimensions {
                return Err(format!(
                    "Stream {idx} #dimensions mismatch: wedeo={:?}, ffmpeg={:?}",
                    wh.dimensions, fh.dimensions
                ));
            }
            if wh.sample_rate != fh.sample_rate {
                return Err(format!(
                    "Stream {idx} #sample_rate mismatch: wedeo={:?}, ffmpeg={:?}",
                    wh.sample_rate, fh.sample_rate
                ));
            }
            if wh.channel_layout_name != fh.channel_layout_name {
                return Err(format!(
                    "Stream {idx} #channel_layout_name mismatch: wedeo={:?}, ffmpeg={:?}",
                    wh.channel_layout_name, fh.channel_layout_name
                ));
            }
            if opts.check_sar && wh.sar != fh.sar {
                return Err(format!(
                    "Stream {idx} #sar mismatch: wedeo={:?}, ffmpeg={:?}",
                    wh.sar, fh.sar
                ));
            }
        }
    }

    let w_data: Vec<&str> = wedeo
        .iter()
        .map(String::as_str)
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .collect();
    let f_data: Vec<&str> = ffmpeg
        .iter()
        .map(String::as_str)
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .collect();

    if w_data.len() != f_data.len() {
        return Err(format!(
            "Frame/packet count differs: wedeo={}, ffmpeg={}",
            w_data.len(),
            f_data.len()
        ));
    }

    for (i, (&w_line, &f_line)) in w_data.iter().zip(f_data.iter()).enumerate() {
        let w = parse_framecrc_record(w_line)?;
        let f = parse_framecrc_record(f_line)?;

        if w.stream_index != f.stream_index || w.size != f.size || w.crc != f.crc {
            return Err(format!(
                "Entry {i} data mismatch:\n  wedeo:  {w_line}\n  ffmpeg: {f_line}"
            ));
        }
        if opts.check_timestamps && (w.dts != f.dts || w.pts != f.pts || w.duration != f.duration) {
            return Err(format!(
                "Entry {i} timestamp mismatch:\n  wedeo:  {w_line}\n  ffmpeg: {f_line}"
            ));
        }
    }

    Ok(())
}

/// Compare `framecrc` lines on `stream_index`, `size`, and `crc`.
pub fn compare_framecrc_lines(wedeo: &[String], ffmpeg: &[String]) -> Result<(), String> {
    compare_framecrc_full(wedeo, ffmpeg, FramecrcCompareOpts::DATA_ONLY)
}

// ---------------------------------------------------------------------------
// checkasm: deterministic PRNG + guard-canary buffer verification
// ---------------------------------------------------------------------------

/// Deterministic SplitMix64 PRNG for reproducible `checkasm` kernel fuzzing.
#[derive(Debug, Clone)]
pub struct CheckasmRng {
    state: u64,
}

impl CheckasmRng {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    pub fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }

    /// Fill `buf` with pseudo-random bytes biased toward boundary/overflow values
    /// (`0x00`, `0x01`, `0x7f`, `0x80`, `0xfe`, `0xff`) 25% of the time.
    pub fn fill_edge_bytes(&mut self, buf: &mut [u8]) {
        const EDGES: [u8; 6] = [0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff];
        for b in buf {
            let r = self.next_u64();
            if (r & 3) == 0 {
                *b = EDGES[((r >> 8) as usize) % EDGES.len()];
            } else {
                *b = (r >> 16) as u8;
            }
        }
    }

    /// Fill `buf` with uniformly distributed pseudo-random bytes.
    pub fn fill_bytes(&mut self, buf: &mut [u8]) {
        for b in buf {
            *b = self.next_u8();
        }
    }
}

/// Guard-canary padded output buffer for `checkasm` OOB write detection.
pub struct GuardedBuffer {
    raw: Vec<u8>,
    payload_len: usize,
}

impl GuardedBuffer {
    pub const CANARY_LEN: usize = 64;

    #[inline]
    fn canary_byte(index: usize) -> u8 {
        0xA5u8 ^ ((index as u8).wrapping_mul(37))
    }

    /// Allocate a payload buffer of `payload_len` zeroed bytes surrounded by 64-byte
    /// deterministic canary guards on both ends.
    pub fn new(payload_len: usize) -> Self {
        let total = Self::CANARY_LEN * 2 + payload_len;
        let mut raw = vec![0u8; total];
        for (i, slot) in raw[..Self::CANARY_LEN].iter_mut().enumerate() {
            *slot = Self::canary_byte(i);
        }
        let tail_start = Self::CANARY_LEN + payload_len;
        for (i, slot) in raw[tail_start..].iter_mut().enumerate() {
            *slot = Self::canary_byte(i);
        }
        Self { raw, payload_len }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.raw[Self::CANARY_LEN..Self::CANARY_LEN + self.payload_len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.raw[Self::CANARY_LEN..Self::CANARY_LEN + self.payload_len]
    }

    /// Verify both leading and trailing canary regions are untouched.
    pub fn verify_canaries(&self, context: &str) -> Result<(), String> {
        for (i, &b) in self.raw[..Self::CANARY_LEN].iter().enumerate() {
            let expected = Self::canary_byte(i);
            if b != expected {
                return Err(format!(
                    "{context}: leading guard canary corrupted at offset -{}: got 0x{b:02x}, expected 0x{expected:02x}",
                    Self::CANARY_LEN - i
                ));
            }
        }
        let tail_start = Self::CANARY_LEN + self.payload_len;
        for (i, &b) in self.raw[tail_start..].iter().enumerate() {
            let expected = Self::canary_byte(i);
            if b != expected {
                return Err(format!(
                    "{context}: trailing guard canary corrupted at offset +{i}: got 0x{b:02x}, expected 0x{expected:02x}"
                ));
            }
        }
        Ok(())
    }
}

/// Execute `iterations` deterministic `checkasm` differential trials.
pub fn run_checkasm<F>(name: &str, iterations: usize, seed: u64, mut body: F)
where
    F: FnMut(&mut CheckasmRng, usize) -> Result<(), String>,
{
    let mut rng = CheckasmRng::new(seed);
    for iter in 0..iterations {
        if let Err(msg) = body(&mut rng, iter) {
            panic!("checkasm '{name}' failed at iter={iter} (seed=0x{seed:016x}): {msg}");
        }
    }
}

/// Macro for differential `checkasm` verification of a candidate byte-slice kernel
/// against a pure-Rust scalar reference with guard-canary buffer protection.
#[macro_export]
macro_rules! checkasm_verify_kernel {
    ($name:expr, $iters:expr, $seed:expr, $out_len:expr, |$rng:ident, $ref_buf:ident, $cand_buf:ident| $body:block) => {{
        $crate::run_checkasm($name, $iters, $seed, |$rng, _iter| {
            let mut ref_guard = $crate::GuardedBuffer::new($out_len);
            let mut cand_guard = $crate::GuardedBuffer::new($out_len);
            {
                let $ref_buf = ref_guard.as_mut_slice();
                let $cand_buf = cand_guard.as_mut_slice();
                $body
            }
            ref_guard.verify_canaries(concat!($name, " (reference)"))?;
            cand_guard.verify_canaries(concat!($name, " (candidate)"))?;
            if ref_guard.as_slice() != cand_guard.as_slice() {
                let diff_pos = ref_guard
                    .as_slice()
                    .iter()
                    .zip(cand_guard.as_slice().iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                return Err(format!(
                    "bit-exact mismatch at byte {diff_pos}: ref=0x{:02x} cand=0x{:02x}",
                    ref_guard.as_slice()[diff_pos],
                    cand_guard.as_slice()[diff_pos],
                ));
            }
            Ok(())
        });
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md5_and_adler32_match_vectors() {
        assert_eq!(md5_bytes(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(
            md5_bytes(b"The quick brown fox jumps over the lazy dog"),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
        assert_eq!(adler32_compute(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32_ffmpeg(&[1, 2, 3]), (10 << 16) | 6);
    }

    #[test]
    fn pinned_oracle_paths_resolve() {
        let ffmpeg = ffmpeg_oracle_bin();
        let ffprobe = ffprobe_oracle_bin();
        assert!(
            ffmpeg.is_file(),
            "expected pinned oracle ffmpeg at {}",
            ffmpeg.display()
        );
        assert!(
            ffprobe.is_file(),
            "expected pinned oracle ffprobe at {}",
            ffprobe.display()
        );
    }

    #[test]
    fn counting_alloc_measures_zero_and_nonzero() {
        let mut scratch = vec![0u8; 256];
        let sum = assert_zero_allocs("in-place fill", || {
            scratch.fill(42);
            scratch.iter().map(|&b| b as usize).sum::<usize>()
        });
        assert_eq!(sum, 256 * 42);

        let (_v, stats) = measure_allocs(|| Vec::<u8>::with_capacity(1024));
        assert!(stats.allocs >= 1);
        assert!(stats.bytes_allocated >= 1024);
    }

    #[test]
    fn checkasm_macro_verifies_identical_kernels_and_canaries() {
        checkasm_verify_kernel!(
            "xor_invert",
            128,
            0xDEAD_BEEF_CAFE_BABEu64,
            64,
            |rng, ref_out, cand_out| {
                let mut src = [0u8; 64];
                rng.fill_edge_bytes(&mut src);
                for i in 0..64 {
                    ref_out[i] = src[i] ^ 0x55;
                    cand_out[i] = src[i] ^ 0x55;
                }
            }
        );
    }
}
