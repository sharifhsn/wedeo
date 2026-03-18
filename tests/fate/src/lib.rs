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
