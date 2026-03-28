#!/usr/bin/env python3
"""Shared utilities for wedeo comparison and conformance scripts.

Provides binary discovery, YUV decode, framecrc execution, and video info
extraction. Lean subset of the full debug library — only what tracked
scripts need.

Not intended to be run directly — import from other scripts:
    from wedeo_utils import find_wedeo_binary, decode_yuv, run_framecrc
"""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# ── Binary Discovery ─────────────────────────────────────────────────────────

# Directories whose .rs files determine source freshness
_RUST_SRC_DIRS = [
    Path("codecs/wedeo-codec-h264/src"),
    Path("crates/wedeo-core/src"),
    Path("crates/wedeo-codec/src"),
    Path("formats/wedeo-format-h264/src"),
    Path("tests/fate/src"),
]


def newest_source_mtime(extra_dirs: list[Path] | None = None) -> float:
    """Get the newest mtime across all relevant Rust source directories."""
    dirs = list(_RUST_SRC_DIRS)
    if extra_dirs:
        dirs.extend(extra_dirs)
    mtimes = []
    for d in dirs:
        if d.exists():
            mtimes.extend(f.stat().st_mtime for f in d.rglob("*.rs"))
    return max(mtimes, default=0)


def find_ffmpeg_binary() -> Path:
    """Find the debug FFmpeg binary (ffmpeg_g preferred, then ffmpeg in FFmpeg/).

    Exits with an actionable error message if not found.
    """
    candidates = [
        Path("FFmpeg/ffmpeg_g"),
        Path("FFmpeg/ffmpeg"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    print(
        "Error: FFmpeg debug binary not found at FFmpeg/ffmpeg_g\n"
        "Build with: cd FFmpeg && ./configure --disable-optimizations "
        "--enable-debug=3 --disable-stripping --disable-asm && make ffmpeg",
        file=sys.stderr,
    )
    sys.exit(1)


def find_wedeo_binary(
    prefer_debug: bool = False,
    auto_rebuild: bool = True,
    features: list[str] | None = None,
) -> Path:
    """Find the wedeo-framecrc binary, optionally rebuilding if stale.

    Args:
        prefer_debug: If True, check debug before release (for tracing builds).
        auto_rebuild: If True, rebuild when source is newer than binary.
        features: Extra cargo features (e.g., ["tracing"]).

    Returns:
        Resolved Path to the binary.

    Exits with an error message if the binary is not found and cannot be built.
    """
    profiles = ["debug", "release"] if prefer_debug else ["release", "debug"]

    for profile in profiles:
        candidate = Path("target") / profile / "wedeo-framecrc"
        if candidate.exists():
            if auto_rebuild and newest_source_mtime() > candidate.stat().st_mtime:
                print(f"Rebuilding {profile} (source newer)...", file=sys.stderr)
                cmd = ["cargo", "build"]
                if profile == "release":
                    cmd.append("--release")
                cmd += ["--bin", "wedeo-framecrc", "-p", "wedeo-fate"]
                if features:
                    cmd += ["--features", ",".join(features)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Build failed:\n{result.stderr[-500:]}", file=sys.stderr)
                    sys.exit(1)
            return candidate.resolve()

    # No binary exists — try building
    profile = "debug" if prefer_debug else "release"
    print(f"Building {profile} wedeo-framecrc...", file=sys.stderr)
    cmd = ["cargo", "build"]
    if profile == "release":
        cmd.append("--release")
    cmd += ["--bin", "wedeo-framecrc", "-p", "wedeo-fate"]
    if features:
        cmd += ["--features", ",".join(features)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"Error: wedeo-framecrc not found and build failed.\n"
            f"Run: cargo build --bin wedeo-framecrc -p wedeo-fate\n"
            f"{result.stderr[-500:]}",
            file=sys.stderr,
        )
        sys.exit(1)
    candidate = Path("target") / profile / "wedeo-framecrc"
    return candidate.resolve()


# ── YUV Decode & Dimensions ──────────────────────────────────────────────────


@dataclass
class FrameInfo:
    """Video frame metadata extracted from wedeo-framecrc header."""

    width: int
    height: int
    frame_count: int
    mb_w: int = 0
    mb_h: int = 0

    def __post_init__(self):
        if self.mb_w == 0:
            self.mb_w = (self.width + 15) // 16
        if self.mb_h == 0:
            self.mb_h = (self.height + 15) // 16


def get_video_info(
    input_path: str | Path,
    wedeo_bin: str | Path | None = None,
    no_deblock: bool = True,
) -> FrameInfo:
    """Extract video dimensions and frame count from wedeo-framecrc output.

    Args:
        input_path: Path to the H.264 file.
        wedeo_bin: Path to wedeo-framecrc binary (auto-found if None).
        no_deblock: If True, disable deblocking.
    """
    if wedeo_bin is None:
        wedeo_bin = find_wedeo_binary()
    env = {**os.environ}
    if no_deblock:
        env["WEDEO_NO_DEBLOCK"] = "1"

    result = subprocess.run(
        [str(wedeo_bin), str(input_path)],
        capture_output=True,
        env=env,
    )
    lines = result.stdout.decode().splitlines()

    width = height = 0
    frame_count = 0
    for line in lines:
        if line.startswith("#dimensions"):
            parts = line.split(":")[-1].strip().split("x")
            width, height = int(parts[0]), int(parts[1])
        elif not line.startswith("#") and line.strip():
            frame_count += 1

    if width == 0 or height == 0:
        raise ValueError("No #dimensions line found in framecrc output")

    return FrameInfo(width=width, height=height, frame_count=frame_count)


def decode_yuv(
    input_path: str | Path,
    tool: str,
    no_deblock: bool = True,
    wedeo_bin: str | Path | None = None,
    extra_ffmpeg_args: list[str] | None = None,
) -> bytes:
    """Decode to raw YUV420p using the specified tool.

    Args:
        input_path: Path to the H.264 file.
        tool: "wedeo" or "ffmpeg".
        no_deblock: If True, disable deblocking.
        wedeo_bin: Path to wedeo binary (auto-found if None, ignored for ffmpeg).
        extra_ffmpeg_args: Extra args inserted before -i for ffmpeg.

    Returns:
        Raw YUV420p bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        yuv_path = f.name
    try:
        if tool == "wedeo":
            if wedeo_bin is None:
                wedeo_bin = find_wedeo_binary()
            env = {**os.environ}
            if no_deblock:
                env["WEDEO_NO_DEBLOCK"] = "1"
            subprocess.run(
                [str(wedeo_bin), str(input_path), "--raw-yuv", yuv_path],
                capture_output=True,
                env=env,
                check=True,
            )
        elif tool == "ffmpeg":
            cmd = ["ffmpeg", "-y", "-bitexact"]
            if extra_ffmpeg_args:
                cmd += extra_ffmpeg_args
            if no_deblock:
                cmd += ["-skip_loop_filter", "all"]
            cmd += [
                "-i", str(input_path),
                "-pix_fmt", "yuv420p",
                "-f", "rawvideo",
                yuv_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        else:
            raise ValueError(f"Unknown tool: {tool!r} (expected 'wedeo' or 'ffmpeg')")
        return Path(yuv_path).read_bytes()
    finally:
        Path(yuv_path).unlink(missing_ok=True)


# ── Framecrc Execution ───────────────────────────────────────────────────────


def run_framecrc(
    cmd: list[str],
    env: dict[str, str] | None = None,
    timeout: int = 60,
) -> list[str]:
    """Run a framecrc command and return list of CRC strings.

    Handles timeout, non-zero exit, and non-UTF8 output.
    Skips comment lines (starting with #) and empty lines.

    Args:
        cmd: Command to run (e.g., [wedeo_bin, input_path] or ffmpeg args).
        env: Extra environment variables (merged with os.environ).
        timeout: Timeout in seconds.

    Returns:
        List of CRC strings (the 6th comma-separated field from each data line).
    """
    full_env = {**os.environ, **(env or {})}
    try:
        result = subprocess.run(
            cmd, capture_output=True, env=full_env, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(
            f"WARN: {' '.join(str(c) for c in cmd[:3])}... timed out after {timeout}s",
            file=sys.stderr,
        )
        return []
    if result.returncode != 0:
        print(
            f"WARN: {' '.join(str(c) for c in cmd[:3])}... exited with {result.returncode}",
            file=sys.stderr,
        )
    crcs = []
    for line in result.stdout.decode(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            crcs.append(parts[5].strip())
    return crcs
