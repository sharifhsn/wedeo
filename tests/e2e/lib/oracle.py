"""FFmpeg n9.1 oracle & system libdav1d comparison helpers."""

from __future__ import annotations

import json
import math
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .harness import CmdResult, find_oracle_bins, find_system_ffmpeg_bin


def ffmpeg_oracle_bin() -> Path:
    ffmpeg, _ = find_oracle_bins()
    return ffmpeg


def ffprobe_oracle_bin() -> Path:
    _, ffprobe = find_oracle_bins()
    return ffprobe


def eval_rational(value: str) -> float:
    """Evaluate an FFmpeg rational string like '24/1' or '500000000/20833333'."""
    if "/" in value:
        num_s, den_s = value.split("/", 1)
        den = float(den_s)
        if den == 0.0:
            raise ValueError(f"Zero denominator in rational {value!r}")
        return float(num_s) / den
    return float(value)


def ffprobe_json(
    path: Path | str,
    *,
    ffprobe_bin: Path | None = None,
    show_chapters: bool = False,
    show_format: bool = True,
    show_streams: bool = True,
    select_streams: str | None = None,
) -> dict[str, Any]:
    """Run ffprobe (oracle by default) with JSON output and return parsed dict."""
    exe = ffprobe_bin or ffprobe_oracle_bin()
    cmd: list[str] = [str(exe), "-v", "error", "-print_format", "json"]
    if show_format:
        cmd.append("-show_format")
    if show_streams:
        cmd.append("-show_streams")
    if show_chapters:
        cmd.append("-show_chapters")
    if select_streams is not None:
        cmd.extend(["-select_streams", select_streams])
    cmd.append(str(path))

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
    if proc.returncode != 0:
        raise AssertionError(
            f"ffprobe failed ({proc.returncode}) on {path}: {proc.stderr}"
        )
    return json.loads(proc.stdout)


def assert_ffprobe_stream(
    probe: dict[str, Any],
    index: int,
    *,
    codec_name: str | None = None,
    codec_type: str | None = None,
    width: int | None = None,
    height: int | None = None,
    pix_fmt: str | None = None,
    sample_rate: int | str | None = None,
    channels: int | None = None,
    sample_fmt: str | None = None,
) -> dict[str, Any]:
    """Assert properties of a stream at `index` in an ffprobe JSON dictionary."""
    streams = probe.get("streams", [])
    if index >= len(streams):
        raise AssertionError(
            f"Expected stream index {index}, found only {len(streams)} streams"
        )
    st = streams[index]
    checks: list[tuple[str, Any, Any]] = [
        ("codec_name", codec_name, st.get("codec_name")),
        ("codec_type", codec_type, st.get("codec_type")),
        ("width", width, st.get("width")),
        ("height", height, st.get("height")),
        ("pix_fmt", pix_fmt, st.get("pix_fmt")),
        (
            "sample_rate",
            str(sample_rate) if sample_rate is not None else None,
            str(st.get("sample_rate")) if st.get("sample_rate") is not None else None,
        ),
        ("channels", channels, st.get("channels")),
        ("sample_fmt", sample_fmt, st.get("sample_fmt")),
    ]
    for field, expected, actual in checks:
        if expected is not None and actual != expected:
            raise AssertionError(
                f"Stream {index} {field}: expected {expected!r}, got {actual!r} "
                f"(stream={st})"
            )
    return st


def assert_ffprobe_tags(
    probe: dict[str, Any],
    expected: dict[str, str],
    *,
    stream_index: int | None = None,
) -> None:
    """Assert that expected tags are present in format.tags or streams[i].tags."""
    if stream_index is None:
        tags = probe.get("format", {}).get("tags", {})
    else:
        tags = probe.get("streams", [])[stream_index].get("tags", {})
    lowered = {k.lower(): v for k, v in tags.items()}
    for k, v in expected.items():
        if lowered.get(k.lower()) != v:
            raise AssertionError(
                f"Expected tag {k!r}={v!r}, got tags={tags!r}"
            )


def assert_ffprobe_parity(
    wedeo_probe: dict[str, Any],
    oracle_probe: dict[str, Any],
    *,
    duration_tol: float = 0.05,
    check_chapters: bool = False,
    expected_tags: dict[str, str] | None = None,
) -> None:
    """Compare wedeo ffprobe JSON against oracle ffprobe JSON on parity-safe fields."""
    wf = wedeo_probe.get("format", {})
    of = oracle_probe.get("format", {})

    for key in ("nb_streams", "format_name", "size"):
        if str(wf.get(key)) != str(of.get(key)):
            raise AssertionError(
                f"format.{key} mismatch: wedeo={wf.get(key)!r} vs oracle={of.get(key)!r}"
            )

    w_dur = float(wf["duration"])
    o_dur = float(of["duration"])
    if abs(w_dur - o_dur) > duration_tol:
        raise AssertionError(
            f"format.duration mismatch: wedeo={w_dur} vs oracle={o_dur} "
            f"(tol={duration_tol})"
        )

    if expected_tags:
        assert_ffprobe_tags(wedeo_probe, expected_tags)
        assert_ffprobe_tags(oracle_probe, expected_tags)

    ws = wedeo_probe.get("streams", [])
    os_ = oracle_probe.get("streams", [])
    if len(ws) != len(os_):
        raise AssertionError(
            f"streams length mismatch: wedeo={len(ws)} vs oracle={len(os_)}"
        )

    for idx, (wst, ost) in enumerate(zip(ws, os_)):
        for key in ("index", "codec_name", "codec_type", "time_base"):
            if wst.get(key) != ost.get(key):
                raise AssertionError(
                    f"streams[{idx}].{key} mismatch: wedeo={wst.get(key)!r} "
                    f"vs oracle={ost.get(key)!r}"
                )
        if ost.get("codec_type") == "video":
            for key in ("width", "height", "pix_fmt"):
                if wst.get(key) != ost.get(key):
                    raise AssertionError(
                        f"streams[{idx}].{key} mismatch: wedeo={wst.get(key)!r} "
                        f"vs oracle={ost.get(key)!r}"
                    )
            if "r_frame_rate" in wst and "r_frame_rate" in ost:
                wr = eval_rational(wst["r_frame_rate"])
                orr = eval_rational(ost["r_frame_rate"])
                if abs(wr - orr) > 0.05:
                    raise AssertionError(
                        f"streams[{idx}].r_frame_rate mismatch: {wst['r_frame_rate']} "
                        f"({wr:.3f}) vs {ost['r_frame_rate']} ({orr:.3f})"
                    )
        elif ost.get("codec_type") == "audio":
            if str(wst.get("sample_rate")) != str(ost.get("sample_rate")):
                raise AssertionError(
                    f"streams[{idx}].sample_rate mismatch: wedeo={wst.get('sample_rate')!r} "
                    f"vs oracle={ost.get('sample_rate')!r}"
                )
            for key in ("channels", "sample_fmt"):
                if wst.get(key) != ost.get(key):
                    raise AssertionError(
                        f"streams[{idx}].{key} mismatch: wedeo={wst.get(key)!r} "
                        f"vs oracle={ost.get(key)!r}"
                    )

    if check_chapters:
        wch = wedeo_probe.get("chapters", [])
        och = oracle_probe.get("chapters", [])
        if len(wch) != len(och):
            raise AssertionError(
                f"chapters count mismatch: wedeo={len(wch)} vs oracle={len(och)}"
            )
        for idx, (wc, oc) in enumerate(zip(wch, och)):
            if abs(float(wc["start_time"]) - float(oc["start_time"])) > duration_tol:
                raise AssertionError(
                    f"chapters[{idx}].start_time mismatch: {wc['start_time']} vs {oc['start_time']}"
                )
            if abs(float(wc["end_time"]) - float(oc["end_time"])) > duration_tol:
                raise AssertionError(
                    f"chapters[{idx}].end_time mismatch: {wc['end_time']} vs {oc['end_time']}"
                )
            if wc.get("tags", {}).get("title") != oc.get("tags", {}).get("title"):
                raise AssertionError(
                    f"chapters[{idx}].tags.title mismatch: {wc.get('tags')} vs {oc.get('tags')}"
                )


def _stream0_codec(path: Path | str) -> str | None:
    probe = ffprobe_json(path, show_format=False, show_streams=True)
    streams = probe.get("streams", [])
    if not streams:
        return None
    return streams[0].get("codec_name")


def oracle_ffmpeg_decode_check(path: Path | str) -> None:
    """Verify full stream decoding using oracle ffmpeg (or libdav1d for AV1)."""
    codec = _stream0_codec(path)
    if codec == "av1":
        cmd = [
            str(find_system_ffmpeg_bin()),
            "-v",
            "error",
            "-c:v",
            "libdav1d",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ]
    else:
        cmd = [
            str(ffmpeg_oracle_bin()),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
    if proc.returncode != 0 or proc.stderr.strip() != "":
        raise AssertionError(
            f"Oracle decode check failed on {path} (exit={proc.returncode}): "
            f"{proc.stderr}"
        )


def decode_raw_video(path: Path | str, *, pix_fmt: str = "yuv420p") -> bytes:
    """Decode stream 0 video to raw pixels (using libdav1d for AV1, oracle ffmpeg otherwise)."""
    codec = _stream0_codec(path)
    if codec == "av1":
        cmd = [
            str(find_system_ffmpeg_bin()),
            "-v",
            "error",
            "-c:v",
            "libdav1d",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ]
    else:
        cmd = [
            str(ffmpeg_oracle_bin()),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60.0)
    if proc.returncode != 0:
        raise AssertionError(
            f"Raw video decode failed on {path} (exit={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout


@dataclass(frozen=True)
class FramecrcRecord:
    """Parsed record from ffmpeg -f framecrc."""

    stream_index: int
    dts: int
    pts: int
    duration: int
    size: int
    crc: str


def decode_framecrc(
    path: Path | str, *, stream_copy: bool = False
) -> list[FramecrcRecord]:
    """Decode or stream-copy media through ffmpeg -f framecrc and parse records."""
    codec = _stream0_codec(path)
    if not stream_copy and codec == "av1":
        exe = find_system_ffmpeg_bin()
        pre_args = ["-c:v", "libdav1d"]
    else:
        exe = ffmpeg_oracle_bin()
        pre_args = []

    cmd = [str(exe), "-v", "error", "-bitexact", *pre_args, "-i", str(path)]
    if stream_copy:
        cmd.extend(["-c", "copy"])
    cmd.extend(["-f", "framecrc", "-flags", "+bitexact", "-fflags", "+bitexact", "-"])

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
    if proc.returncode != 0:
        raise AssertionError(
            f"framecrc failed on {path} (exit={proc.returncode}): {proc.stderr}"
        )
    records: list[FramecrcRecord] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) < 6:
            continue
        records.append(
            FramecrcRecord(
                stream_index=int(parts[0]),
                dts=int(parts[1]),
                pts=int(parts[2]),
                duration=int(parts[3]),
                size=int(parts[4]),
                crc=parts[5],
            )
        )
    return records


def decode_pcm_s16le(
    path: Path | str,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> tuple[int, ...]:
    """Decode audio to signed 16-bit little-endian PCM samples using oracle ffmpeg."""
    cmd = [str(ffmpeg_oracle_bin()), "-v", "error", "-i", str(path)]
    if sample_rate is not None:
        cmd.extend(["-ar", str(sample_rate)])
    if channels is not None:
        cmd.extend(["-ac", str(channels)])
    cmd.extend(["-f", "s16le", "-"])

    proc = subprocess.run(cmd, capture_output=True, timeout=60.0)
    if proc.returncode != 0:
        raise AssertionError(
            f"PCM decode failed on {path} (exit={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')}"
        )
    count = len(proc.stdout) // 2
    return struct.unpack(f"<{count}h", proc.stdout[: count * 2])


def compute_psnr(ref_bytes: bytes, got_bytes: bytes) -> float:
    """Compute 8-bit PSNR (dB) between two equal-length raw pixel buffers."""
    if len(ref_bytes) != len(got_bytes):
        raise AssertionError(
            f"Raw video byte length mismatch: ref={len(ref_bytes)} vs got={len(got_bytes)}"
        )
    if not ref_bytes:
        raise AssertionError("Cannot compute PSNR on empty buffer")
    sq_err = sum((a - b) * (a - b) for a, b in zip(ref_bytes, got_bytes))
    if sq_err == 0:
        return float("inf")
    mse = sq_err / len(ref_bytes)
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def assert_video_psnr(
    ref_path: Path | str,
    got_path: Path | str,
    *,
    min_psnr_db: float = 30.0,
    pix_fmt: str = "yuv420p",
) -> float:
    """Assert decoded video PSNR between ref_path and got_path meets min_psnr_db."""
    ref_bytes = decode_raw_video(ref_path, pix_fmt=pix_fmt)
    got_bytes = decode_raw_video(got_path, pix_fmt=pix_fmt)
    psnr = compute_psnr(ref_bytes, got_bytes)
    if psnr < min_psnr_db:
        raise AssertionError(
            f"PSNR {psnr:.2f} dB below threshold {min_psnr_db:.2f} dB "
            f"(ref={ref_path}, got={got_path})"
        )
    return psnr


def assert_audio_close(
    ref_samples: tuple[int, ...] | list[int],
    got_samples: tuple[int, ...] | list[int],
    *,
    max_mae: float = 5.0,
    max_sample_diff: int = 50,
    allow_len_delta: int = 0,
) -> None:
    """Assert two PCM S16LE sample sequences match within error bounds."""
    len_diff = abs(len(ref_samples) - len(got_samples))
    if len_diff > allow_len_delta:
        raise AssertionError(
            f"Sample count mismatch: ref={len(ref_samples)} vs got={len(got_samples)} "
            f"(allowed delta {allow_len_delta})"
        )
    n = min(len(ref_samples), len(got_samples))
    if n == 0:
        raise AssertionError("No audio samples to compare")
    diffs = [abs(a - b) for a, b in zip(ref_samples[:n], got_samples[:n])]
    mae = sum(diffs) / n
    peak = max(diffs)
    if mae > max_mae or peak > max_sample_diff:
        raise AssertionError(
            f"Audio mismatch: mae={mae:.3f} (limit {max_mae}), "
            f"max_diff={peak} (limit {max_sample_diff})"
        )


def assert_wedeo_inspect_clean(
    run_wedeo_fn: Callable[..., CmdResult],
    path: Path | str,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Run `wedeo inspect [--verify] --json <path>` and assert clean health."""
    args: list[str | Path] = ["inspect"]
    if verify:
        args.append("--verify")
    args.extend(["--json", path])
    res = run_wedeo_fn(args)
    if res.returncode != 0:
        raise AssertionError(
            f"wedeo inspect failed ({res.returncode}) on {path}: {res.stderr}"
        )
    data = json.loads(res.stdout)
    if not data.get("health", {}).get("is_clean", False):
        raise AssertionError(f"Expected health.is_clean=True on {path}: {data['health']}")
    if verify:
        if data.get("verified") is not True:
            raise AssertionError(f"Expected verified=True on {path}: {data}")
        if data.get("decoded_units", 0) <= 0:
            raise AssertionError(
                f"Expected decoded_units > 0 on {path}, got {data.get('decoded_units')}"
            )
    return data
