"""Read-only media corpus paths and cached/temp synthetic fixture generators."""

from __future__ import annotations

import atexit
import math
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Iterable

from .harness import find_system_ffmpeg_bin, repo_root

REPO_ROOT: Path = repo_root()
WEB_SAMPLES_DIR: Path = REPO_ROOT / "web" / "samples"
ORACLE_DATA_DIR: Path = REPO_ROOT / "tests" / "wedeo-oracle" / "data"
FATE_SUITE_DIR: Path = Path(
    os.environ.get("FATE_SUITE", str(REPO_ROOT / "target" / "fate-suite"))
)

_CACHE_DIR: tempfile.TemporaryDirectory[str] | None = None
_SYNTH_CACHE: dict[tuple[Any, ...], Path] = {}


def _get_cache_dir() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = tempfile.TemporaryDirectory(prefix="wedeo_e2e_fixtures_")
        atexit.register(_cleanup_cache)
    return Path(_CACHE_DIR.name)


def _cleanup_cache() -> None:
    global _CACHE_DIR
    _SYNTH_CACHE.clear()
    if _CACHE_DIR is not None:
        _CACHE_DIR.cleanup()
        _CACHE_DIR = None


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} fixture missing: {path}")
    return path


def sample_path(rel: str | Path) -> Path:
    """Return a read-only file under web/samples/."""
    return _require_file(WEB_SAMPLES_DIR / rel, "web/samples")


def oracle_data_path(rel: str | Path) -> Path:
    """Return a read-only file under tests/wedeo-oracle/data/."""
    return _require_file(ORACLE_DATA_DIR / rel, "tests/wedeo-oracle/data")


def fate_sample_path(rel: str | Path) -> Path:
    """Return a read-only file under FATE_SUITE (target/fate-suite/)."""
    return _require_file(FATE_SUITE_DIR / rel, "fate-suite")


def h264_aac_mp4() -> Path:
    return sample_path("h264_aac_320x240.mp4")


def vp9_smpte_mkv() -> Path:
    return sample_path("vp9_smpte_320x240.mkv")


def chord_48k_wav() -> Path:
    return sample_path("chord_48k.wav")


def vp9_red_ivf() -> Path:
    return oracle_data_path("vp9_red_64x64.ivf")


def vp9_keyframe_ivf() -> Path:
    return oracle_data_path("vp9_keyframe_64x64.ivf")


def vp9_blue_ivf() -> Path:
    return oracle_data_path("vp9_blue_100x75.ivf")


def vp9_testsrc_128_ivf() -> Path:
    return oracle_data_path("vp9_testsrc_128x128.ivf")


def vp9_testsrc_320_ivf() -> Path:
    return oracle_data_path("vp9_testsrc_320x240.ivf")


def av1_allintra_ivf() -> Path:
    return fate_sample_path("av1-test-vectors/av1-1-b8-02-allintra.ivf")


def synth_wav(
    out_path: Path | None = None,
    *,
    sample_rate: int = 48000,
    channels: int = 1,
    duration_sec: float = 1.0,
    freq_hz: float = 440.0,
) -> Path:
    """Generate a deterministic PCM S16LE sine-wave WAV file."""
    key = ("wav", sample_rate, channels, duration_sec, freq_hz)
    if out_path is None and key in _SYNTH_CACHE and _SYNTH_CACHE[key].is_file():
        return _SYNTH_CACHE[key]

    target = out_path or (
        _get_cache_dir()
        / f"synth_{sample_rate}_{channels}ch_{duration_sec:.2f}s.wav"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(round(sample_rate * duration_sec))
    amp = 12000.0
    frames = bytearray()
    for i in range(num_frames):
        t = i / sample_rate
        val = int(round(amp * math.sin(2.0 * math.pi * freq_hz * t)))
        val = max(-32768, min(32767, val))
        sample_bytes = struct.pack("<h", val)
        frames.extend(sample_bytes * channels)

    with wave.open(str(target), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))

    if out_path is None:
        _SYNTH_CACHE[key] = target
    return target


def synth_multi_gop_h264_mp4(
    out_path: Path | None = None,
    *,
    width: int = 64,
    height: int = 64,
    fps: int = 30,
    duration: float = 1.8,
    gop: int = 10,
    with_audio: bool = True,
) -> Path:
    """Generate a multi-GOP H.264 (+ optional AAC) MP4 using system ffmpeg."""
    key = ("h264_mp4", width, height, fps, duration, gop, with_audio)
    if out_path is None and key in _SYNTH_CACHE and _SYNTH_CACHE[key].is_file():
        return _SYNTH_CACHE[key]

    target = out_path or (
        _get_cache_dir()
        / f"synth_h264_{width}x{height}_{fps}fps_g{gop}_{int(with_audio)}.mp4"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_system_ffmpeg_bin()
    cmd: list[str] = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s={width}x{height}:r={fps}:d={duration}",
    ]
    if with_audio:
        cmd.extend(["-f", "lavfi", "-i", f"sine=f=440:r=48000:d={duration}"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
        ]
    )
    if with_audio:
        cmd.extend(["-c:a", "aac", "-ac", "1", "-ar", "48000", "-shortest"])
    cmd.append(str(target))

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to synthesize H.264 MP4 ({proc.returncode}): {proc.stderr}"
        )

    if out_path is None:
        _SYNTH_CACHE[key] = target
    return target


def synth_multi_gop_av1_mkv(
    out_path: Path | None = None,
    *,
    width: int = 64,
    height: int = 64,
    fps: int = 10,
    duration: float = 0.9,
    gop: int = 3,
) -> Path:
    """Generate a multi-GOP AV1 MKV fixture using system ffmpeg (libaom-av1)."""
    key = ("av1_mkv", width, height, fps, duration, gop)
    if out_path is None and key in _SYNTH_CACHE and _SYNTH_CACHE[key].is_file():
        return _SYNTH_CACHE[key]

    target = out_path or (
        _get_cache_dir() / f"synth_av1_{width}x{height}_{fps}fps_g{gop}.mkv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_system_ffmpeg_bin()
    cmd = [
        str(ffmpeg),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s={width}x{height}:r={fps}:d={duration}",
        "-c:v",
        "libaom-av1",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-cpu-used",
        "8",
        "-usage",
        "realtime",
        str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to synthesize AV1 MKV ({proc.returncode}): {proc.stderr}"
        )

    if out_path is None:
        _SYNTH_CACHE[key] = target
    return target


def synth_subtitle_vtt(out_path: Path, *, content: str | None = None) -> Path:
    """Write a deterministic 2-cue WebVTT subtitle file to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        content
        if content is not None
        else (
            "WEBVTT\n\n"
            "00:00.500 --> 00:02.000\n"
            "First cue\n\n"
            "00:01:05.250 --> 00:01:07.000\n"
            "Second cue\n"
        )
    )
    out_path.write_text(text, encoding="utf-8")
    return out_path


def synth_subtitle_srt(out_path: Path, *, content: str | None = None) -> Path:
    """Write a deterministic 2-cue SRT subtitle file to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        content
        if content is not None
        else (
            "1\n"
            "00:00:00,500 --> 00:00:02,000\n"
            "Erster Untertitel\n\n"
            "2\n"
            "00:00:03,000 --> 00:00:04,500\n"
            "Zweiter Untertitel\n"
        )
    )
    out_path.write_text(text, encoding="utf-8")
    return out_path


def synth_ffmetadata1(
    out_path: Path,
    *,
    tags: dict[str, str] | None = None,
    chapters: Iterable[tuple[int, int, str]] = (),
    timebase: str = "1/1000",
) -> Path:
    """Write an ;FFMETADATA1 sidecar file with global tags and chapters."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [";FFMETADATA1"]
    if tags:
        for k, v in tags.items():
            lines.append(f"{k}={v}")
    for start, end, title in chapters:
        lines.extend(
            [
                "",
                "[CHAPTER]",
                f"TIMEBASE={timebase}",
                f"START={start}",
                f"END={end}",
                f"title={title}",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def copy_fixture(src: Path, dst: Path) -> Path:
    """Copy a fixture into a test temp directory."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
