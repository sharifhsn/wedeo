//! FATE test support library.
//!
//! Provides utilities for running FATE-style comparison tests between
//! wedeo and FFmpeg outputs.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Location of the FATE sample suite (set via env or default).
pub fn fate_suite_dir() -> PathBuf {
    std::env::var("FATE_SUITE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("fate-suite"))
}

/// Check if a FATE sample file exists.
pub fn fate_sample(relative_path: &str) -> Option<PathBuf> {
    let path = fate_suite_dir().join(relative_path);
    if path.exists() { Some(path) } else { None }
}

/// Compute MD5 of a file.
pub fn md5_file(path: &Path) -> std::io::Result<String> {
    let data = std::fs::read(path)?;
    Ok(md5_bytes(&data))
}

/// Compute MD5 of raw bytes.
pub fn md5_bytes(data: &[u8]) -> String {
    use md5::Digest;
    let result = md5::Md5::digest(data);
    result
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

/// Compare two text outputs line by line, ignoring trailing whitespace.
/// Returns Ok(()) if they match, Err with diff info if not.
pub fn diff_output(expected: &str, actual: &str) -> Result<(), String> {
    let expected_lines: Vec<&str> = expected.lines().collect();
    let actual_lines: Vec<&str> = actual.lines().collect();

    if expected_lines.len() != actual_lines.len() {
        return Err(format!(
            "Line count differs: expected {} lines, got {} lines",
            expected_lines.len(),
            actual_lines.len()
        ));
    }

    for (i, (exp, act)) in expected_lines.iter().zip(actual_lines.iter()).enumerate() {
        let exp_trimmed = exp.trim_end();
        let act_trimmed = act.trim_end();
        if exp_trimmed != act_trimmed {
            return Err(format!(
                "Line {} differs:\n  expected: {}\n  actual:   {}",
                i + 1,
                exp_trimmed,
                act_trimmed
            ));
        }
    }

    Ok(())
}

/// Run wedeo-framecrc on a file and capture stdout.
pub fn run_framecrc(input_path: &Path, binary: &str) -> Result<String, String> {
    let output = Command::new(binary)
        .arg(input_path.to_str().unwrap())
        .output()
        .map_err(|e| format!("Failed to run wedeo-framecrc: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "wedeo-framecrc failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Run wedeo-audiogen to create a test WAV file.
pub fn run_audiogen(
    output_path: &Path,
    sample_rate: u32,
    channels: u32,
    binary: &str,
) -> Result<(), String> {
    let output = Command::new(binary)
        .arg(output_path.to_str().unwrap())
        .arg(sample_rate.to_string())
        .arg(channels.to_string())
        .output()
        .map_err(|e| format!("Failed to run wedeo-audiogen: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "wedeo-audiogen failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(())
}

/// Standard Adler-32 (s1=1 init, per RFC 1950).
pub fn adler32_compute(data: &[u8]) -> u32 {
    let mut hasher = adler2::Adler32::new();
    hasher.write_slice(data);
    hasher.checksum()
}

/// Run wedeo-framecrc binary on a file and return stdout lines.
pub fn run_wedeo_framecrc(path: &Path) -> Result<Vec<String>, String> {
    let bin = std::env::var("CARGO_BIN_EXE_wedeo-framecrc")
        .or_else(|_| {
            // Fallback: search in target directory relative to manifest
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let workspace = manifest_dir.parent().unwrap().parent().unwrap();
            for profile in &["debug", "release"] {
                let bin = workspace
                    .join("target")
                    .join(profile)
                    .join("wedeo-framecrc");
                if bin.exists() {
                    return Ok(bin.to_str().unwrap().to_string());
                }
            }
            Err(std::env::VarError::NotPresent)
        })
        .map_err(|_| "wedeo-framecrc binary not found".to_string())?;

    let output = Command::new(&bin)
        .arg(path.to_str().unwrap())
        .output()
        .map_err(|e| format!("Failed to run wedeo-framecrc: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "wedeo-framecrc failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().map(|l| l.to_string()).collect())
}

/// Run FFmpeg in framecrc decode mode for a video file.
/// Returns None if ffmpeg is not available.
pub fn run_ffmpeg_framecrc_video(path: &Path) -> Option<Vec<String>> {
    let output = Command::new("ffmpeg")
        .args([
            "-bitexact",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "framecrc",
            "-",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(stdout.lines().map(|l| l.to_string()).collect())
}

/// Compare framecrc lines between wedeo and FFmpeg, ignoring comment lines.
/// Returns Ok(()) if all data lines match on size and checksum fields.
pub fn compare_framecrc_lines(wedeo: &[String], ffmpeg: &[String]) -> Result<(), String> {
    let wedeo_data: Vec<&str> = wedeo
        .iter()
        .map(|s| s.as_str())
        .filter(|l| !l.starts_with('#'))
        .collect();
    let ffmpeg_data: Vec<&str> = ffmpeg
        .iter()
        .map(|s| s.as_str())
        .filter(|l| !l.starts_with('#'))
        .collect();

    if wedeo_data.len() != ffmpeg_data.len() {
        return Err(format!(
            "Frame count differs: wedeo={}, ffmpeg={}",
            wedeo_data.len(),
            ffmpeg_data.len()
        ));
    }

    for (i, (w, f)) in wedeo_data.iter().zip(ffmpeg_data.iter()).enumerate() {
        let w_fields: Vec<&str> = w.split(',').map(|s| s.trim()).collect();
        let f_fields: Vec<&str> = f.split(',').map(|s| s.trim()).collect();

        if w_fields.len() < 6 || f_fields.len() < 6 {
            return Err(format!(
                "Frame {}: malformed framecrc line\n  wedeo:  {w}\n  ffmpeg: {f}",
                i
            ));
        }

        // Compare size (field 4) and checksum (field 5)
        if w_fields[4] != f_fields[4] || w_fields[5] != f_fields[5] {
            return Err(format!(
                "Frame {} mismatch:\n  wedeo:  {w}\n  ffmpeg: {f}",
                i
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_md5() {
        assert_eq!(md5_bytes(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(
            md5_bytes(b"The quick brown fox jumps over the lazy dog"),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
    }

    #[test]
    fn test_adler32() {
        // "Wikipedia" -> 0x11E60398
        let checksum = super::adler32_compute(b"Wikipedia");
        assert_eq!(checksum, 0x11E60398);
    }
}
