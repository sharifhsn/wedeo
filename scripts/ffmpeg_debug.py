#!/usr/bin/env python3
"""Shared utilities for FFmpeg debug extraction and wedeo comparison scripts.

Provides binary discovery, YUV decode, trace extraction, frame order computation,
LLDB execution, and H.264-specific presets for debug scripts.

Not intended to be run directly — import from other scripts:
    from ffmpeg_debug import find_wedeo_binary, decode_yuv, run_lldb
"""

import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


# ── Section 0: Conformance File Resolution ───────────────────────────────────

CONFORMANCE_DIR = Path("fate-suite/h264-conformance")
CONFORMANCE_EXTENSIONS = (".264", ".jsv", ".26l", ".h264")


def resolve_conformance_file(name: str) -> Path:
    """Resolve a file name or partial match to an H.264 conformance file path.

    Accepts an exact path, a filename, or a partial name for fuzzy matching
    within the conformance directory.
    """
    p = Path(name)
    if p.exists():
        return p
    if CONFORMANCE_DIR.exists():
        for ext in CONFORMANCE_EXTENSIONS:
            matches = list(CONFORMANCE_DIR.glob(f"*{name}*{ext}"))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                print(
                    f"Ambiguous match for '{name}': {[m.name for m in matches]}",
                    file=sys.stderr,
                )
                sys.exit(1)
    print(f"Not found: {name}", file=sys.stderr)
    sys.exit(1)


# ── Section 1: Binary Discovery ──────────────────────────────────────────────

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


# ── Section 2: YUV Decode & Dimensions ───────────────────────────────────────


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
            self.mb_w = self.width // 16
        if self.mb_h == 0:
            self.mb_h = self.height // 16


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


def load_yuv_frame(
    data: bytes,
    frame_idx: int,
    width: int,
    height: int,
) -> tuple | None:
    """Extract Y, U, V planes as numpy arrays for a single frame.

    Returns:
        (y, u, v) tuple of numpy arrays, or None if frame_idx is out of range.
        y shape: (height, width), u/v shape: (height//2, width//2).

    Requires numpy (imported lazily).
    """
    import numpy as np

    cw, ch = width // 2, height // 2
    y_size = width * height
    uv_size = cw * ch
    frame_size = y_size + 2 * uv_size

    base = frame_idx * frame_size
    if base + frame_size > len(data):
        return None

    y = np.frombuffer(data[base : base + y_size], dtype=np.uint8).reshape(
        height, width
    )
    u = np.frombuffer(
        data[base + y_size : base + y_size + uv_size], dtype=np.uint8
    ).reshape(ch, cw)
    v = np.frombuffer(
        data[base + y_size + uv_size : base + frame_size], dtype=np.uint8
    ).reshape(ch, cw)
    return y, u, v


# ── Section 2b: Framecrc & Frame Count ────────────────────────────────────────


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


def check_yuv_frame_count(
    data: bytes,
    width: int,
    height: int,
    expected: int,
    label: str = "",
) -> int:
    """Compute frame count from YUV420p data and warn if it differs from expected.

    Protects against the CVSE3-style mismatch where FFmpeg rawvideo outputs
    a different number of frames than framecrc.

    Returns the actual frame count.
    """
    frame_size = width * height * 3 // 2
    if frame_size == 0:
        return 0
    actual = len(data) // frame_size
    if actual != expected:
        tag = f" ({label})" if label else ""
        print(
            f"WARNING: Frame count mismatch{tag}! "
            f"actual={actual}, expected={expected}. "
            f"Results may be misleading — consider framecrc_compare.py instead.",
            file=sys.stderr,
        )
    return actual


# ── Section 3: Trace Extraction ──────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s: str) -> str:
    """Remove ANSI escape codes from a string."""
    return _ANSI_RE.sub("", s)


def run_wedeo_with_tracing(
    input_path: str | Path,
    rust_log: str = "debug",
    no_deblock: bool = True,
    wedeo_bin: str | Path | None = None,
    features: list[str] | None = None,
) -> str:
    """Build and run wedeo with tracing, returning clean trace text.

    Args:
        input_path: Path to the H.264 file.
        rust_log: RUST_LOG filter string.
        no_deblock: If True, disable deblocking.
        wedeo_bin: Path to wedeo binary (auto-found with tracing feature if None).
        features: Extra cargo features for build (default: ["tracing"]).

    Returns:
        Cleaned trace text (ANSI stripped) from stderr.
    """
    if features is None:
        features = ["tracing"]

    if wedeo_bin is None:
        wedeo_bin = find_wedeo_binary(prefer_debug=True, features=features)

    env = {**os.environ, "RUST_LOG": rust_log}
    if no_deblock:
        env["WEDEO_NO_DEBLOCK"] = "1"

    result = subprocess.run(
        [str(wedeo_bin), str(input_path)],
        capture_output=True,
        env=env,
    )
    return strip_ansi(result.stderr.decode("utf-8", errors="replace"))


# ── Section 4: Frame Order & POC ─────────────────────────────────────────────


@dataclass
class DecodedFrame:
    """A decoded frame with ordering information."""

    decode_idx: int
    poc: int
    output_idx: int = -1
    slice_type: str = "?"
    frame_num: int = 0
    is_idr: bool = False
    nal_ref_idc: int = 0


def get_frame_order(input_path: str | Path) -> list[DecodedFrame]:
    """Compute decode/output order for all frames in an H.264 file.

    Uses FFmpeg's trace_headers BSF to extract slice header info, then
    computes full POC (type 0) and output order.

    Limitations:
        - Only handles POC type 0.
        - Multi-slice frames: only first slice (first_mb_in_slice=0) recorded.
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-c:v",
            "copy",
            "-bsf:v",
            "trace_headers",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    output = result.stderr

    frames_raw = []
    current_nal_type = None
    current_ref_idc = None
    current = None
    max_poc_lsb = None
    slice_type_map = {0: "P", 1: "B", 2: "I", 5: "P", 6: "B", 7: "I"}

    for line in output.split("\n"):
        m = re.search(r"\]\s+\d+\s+(\w+)\s+\S+\s+=\s+(-?\d+)", line)
        if not m:
            continue
        field_name, value = m.group(1), int(m.group(2))

        if field_name == "log2_max_pic_order_cnt_lsb_minus4":
            max_poc_lsb = 1 << (value + 4)
        elif field_name == "nal_ref_idc":
            current_ref_idc = value
        elif field_name == "nal_unit_type":
            current_nal_type = value
        elif field_name == "first_mb_in_slice" and value == 0:
            if current_nal_type in (1, 5):
                current = {
                    "is_idr": current_nal_type == 5,
                    "nal_ref_idc": (
                        current_ref_idc if current_ref_idc is not None else 0
                    ),
                }
        elif field_name == "slice_type" and current is not None:
            current["slice_type"] = slice_type_map.get(value, f"?{value}")
        elif field_name == "frame_num" and current is not None:
            current["frame_num"] = value
        elif field_name == "pic_order_cnt_lsb" and current is not None:
            current["poc_lsb"] = value
            frames_raw.append(current)
            current = None

    # Compute full POC (handle wrap-around for POC type 0)
    prev_poc_msb = 0
    prev_poc_lsb = 0
    decoded_frames = []
    for i, f in enumerate(frames_raw):
        poc_lsb = f.get("poc_lsb", 0)
        if f.get("is_idr"):
            poc_msb = 0
            prev_poc_msb = 0
            prev_poc_lsb = 0
        elif max_poc_lsb is not None:
            if poc_lsb < prev_poc_lsb and (prev_poc_lsb - poc_lsb) >= max_poc_lsb // 2:
                poc_msb = prev_poc_msb + max_poc_lsb
            elif (
                poc_lsb > prev_poc_lsb
                and (poc_lsb - prev_poc_lsb) > max_poc_lsb // 2
            ):
                poc_msb = prev_poc_msb - max_poc_lsb
            else:
                poc_msb = prev_poc_msb
        else:
            poc_msb = 0

        poc = poc_msb + poc_lsb
        if f.get("nal_ref_idc", 0) > 0:
            prev_poc_msb = poc_msb
            prev_poc_lsb = poc_lsb

        decoded_frames.append(
            DecodedFrame(
                decode_idx=i,
                poc=poc,
                slice_type=f.get("slice_type", "?"),
                frame_num=f.get("frame_num", 0),
                is_idr=f.get("is_idr", False),
                nal_ref_idc=f.get("nal_ref_idc", 0),
            )
        )

    # Compute output order
    sorted_indices = sorted(
        range(len(decoded_frames)), key=lambda j: decoded_frames[j].poc
    )
    for out_idx, dec_idx in enumerate(sorted_indices):
        decoded_frames[dec_idx].output_idx = out_idx

    return decoded_frames


def count_slice_type_before(
    frames: list[DecodedFrame],
    target_decode_idx: int,
    slice_type: str,
) -> int:
    """Count how many frames of a given slice type are decoded before the target."""
    return sum(
        1
        for f in frames
        if f.decode_idx < target_decode_idx and f.slice_type == slice_type
    )


# ── Section 5: LLDB Execution Engine ─────────────────────────────────────────


@dataclass
class LldbResult:
    """Result of an lldb extraction run."""

    values: list[str]
    stdout: str
    stderr: str
    returncode: int


@dataclass
class RegisterNames:
    """Platform-specific register names for function arguments."""

    args: list[str]

    @property
    def arg0(self) -> str:
        return self.args[0]

    @property
    def arg1(self) -> str:
        return self.args[1]

    @property
    def arg2(self) -> str:
        return self.args[2]

    @property
    def arg3(self) -> str:
        return self.args[3]


def get_register_names() -> RegisterNames:
    """Get platform-specific register names for function arguments."""
    if platform.machine() == "arm64":
        return RegisterNames(
            args=["$x0", "$x1", "$x2", "$x3", "$x4", "$x5", "$x6", "$x7"]
        )
    else:
        return RegisterNames(args=["$rdi", "$rsi", "$rdx", "$rcx", "$r8", "$r9"])


def parse_lldb_int(value_str: str) -> int:
    """Parse an integer from an lldb expression result like '42' or '-1'."""
    m = re.search(r"(-?\d+)", value_str)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse integer from: {value_str!r}")


def parse_lldb_int_pair(value_str: str) -> tuple[int, int]:
    """Parse a pair of ints from lldb output like '([0] = 3, [1] = -5)'."""
    m = re.search(r"\[0\] = (-?\d+),\s*\[1\] = (-?\d+)", value_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot parse int pair from: {value_str!r}")


def run_lldb(
    ffmpeg_bin: str | Path,
    input_path: str | Path,
    expressions: list[str],
    breakpoint_func: str,
    breakpoint_condition: str | None = None,
    ignore_count: int = 0,
    timeout: int = 180,
    ffmpeg_extra_args: list[str] | None = None,
) -> LldbResult:
    """Run lldb to extract values from FFmpeg at a breakpoint.

    Args:
        ffmpeg_bin: Path to debug FFmpeg binary.
        input_path: Path to input file.
        expressions: List of lldb expressions to evaluate at the breakpoint.
        breakpoint_func: Function name for the breakpoint.
        breakpoint_condition: Optional C condition expression.
        ignore_count: Number of breakpoint hits to skip.
        timeout: Timeout in seconds.
        ffmpeg_extra_args: Extra args for FFmpeg (default: standard debug args).

    Returns:
        LldbResult with parsed values and raw output.
    """
    if ffmpeg_extra_args is None:
        ffmpeg_extra_args = [
            "-threads", "1", "-bitexact", "-skip_loop_filter", "all",
        ]

    cmds = [f"target create {ffmpeg_bin}"]

    bp_cmd = f"breakpoint set -n {breakpoint_func}"
    if breakpoint_condition:
        bp_cmd += f' -c "{breakpoint_condition}"'
    cmds.append(bp_cmd)

    if ignore_count > 0:
        cmds.append(f"breakpoint modify 1 -i {ignore_count}")

    ffmpeg_args = " ".join(
        ffmpeg_extra_args + ["-i", str(input_path), "-f", "null", "/dev/null"]
    )
    cmds.append(f"process launch -- {ffmpeg_args}")

    for expr in expressions:
        cmds.append(f"expression {expr}")

    cmds.append("quit")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".lldb", delete=False) as f:
        f.write("\n".join(cmds) + "\n")
        script_path = f.name

    try:
        result = subprocess.run(
            ["lldb", "--batch", "--source", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return LldbResult(
            values=[], stdout="", stderr=f"Timeout after {timeout}s", returncode=-1
        )
    finally:
        Path(script_path).unlink(missing_ok=True)

    # Parse $N = <value> lines
    values = []
    for line in result.stdout.splitlines():
        m = re.search(r"\$\d+ = (.+)", line)
        if m:
            values.append(m.group(1).strip())

    return LldbResult(
        values=values,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


def calibrate_ignore_count(
    ffmpeg_bin: str | Path,
    input_path: str | Path,
    breakpoint_func: str,
    breakpoint_condition: str | None,
    target_poc: int,
    initial_guess: int = 0,
    search_range: int = 10,
    timeout: int = 120,
    verbose: bool = True,
) -> int | None:
    """Search for the correct lldb ignore count that lands on the target POC.

    Checks POC via ``h->cur_pic_ptr->field_poc[0]``. Accepts both
    ``target_poc`` and ``target_poc | 0x10000`` (frame-mode offset).

    Returns:
        The correct ignore count, or None if not found.
    """
    accept = {target_poc, target_poc | 0x10000}
    seen: set[int] = set()

    for offset in range(search_range):
        for candidate in [initial_guess + offset, initial_guess - offset]:
            if candidate < 0 or candidate in seen:
                continue
            seen.add(candidate)

            if verbose:
                print(
                    f"  trying ignore={candidate}...",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

            result = run_lldb(
                ffmpeg_bin,
                input_path,
                expressions=["(int)(h->cur_pic_ptr->field_poc[0])"],
                breakpoint_func=breakpoint_func,
                breakpoint_condition=breakpoint_condition,
                ignore_count=candidate,
                timeout=timeout,
            )

            if result.returncode == -1:
                if verbose:
                    print(" timeout", file=sys.stderr)
                continue

            if not result.values:
                if verbose:
                    print(" no values", file=sys.stderr)
                continue

            try:
                got_poc = parse_lldb_int(result.values[0])
            except ValueError:
                if verbose:
                    print(
                        f" parse error: {result.values[0]!r}", file=sys.stderr
                    )
                continue

            if got_poc in accept:
                if verbose:
                    print(f" poc={got_poc} OK", file=sys.stderr)
                return candidate
            else:
                if verbose:
                    print(f" poc={got_poc} (want {target_poc})", file=sys.stderr)

    return None


# ── Section 6: H.264 Constants & Presets ─────────────────────────────────────

# FFmpeg's scan8 table for luma 4x4 blocks (first 16 entries).
# Maps block index 0..15 to mv_cache index.
# Block layout within MB:
#   8x8[0]: blk 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)
#   8x8[1]: blk 4=(2,0) 5=(3,0) 6=(2,1) 7=(3,1)
#   8x8[2]: blk 8=(0,2) 9=(1,2) 10=(0,3) 11=(1,3)
#   8x8[3]: blk 12=(2,2) 13=(3,2) 14=(2,3) 15=(3,3)
SCAN8 = [
    4 + 1 * 8, 5 + 1 * 8, 4 + 2 * 8, 5 + 2 * 8,  # 12,13,20,21
    6 + 1 * 8, 7 + 1 * 8, 6 + 2 * 8, 7 + 2 * 8,  # 14,15,22,23
    4 + 3 * 8, 5 + 3 * 8, 4 + 4 * 8, 5 + 4 * 8,  # 28,29,36,37
    6 + 3 * 8, 7 + 3 * 8, 6 + 4 * 8, 7 + 4 * 8,  # 30,31,38,39
]

# Block index to (bx, by) in 4x4 units within MB
BLK_XY = [
    (0, 0), (1, 0), (0, 1), (1, 1),
    (2, 0), (3, 0), (2, 1), (3, 1),
    (0, 2), (1, 2), (0, 3), (1, 3),
    (2, 2), (3, 2), (2, 3), (3, 3),
]

AV_PICTURE_TYPE_I = 1
AV_PICTURE_TYPE_P = 2
AV_PICTURE_TYPE_B = 3


def h264_mv_preset(
    mb_x: int,
    mb_y: int,
    lists: list[int] | None = None,
    slice_type_nos: int = AV_PICTURE_TYPE_B,
) -> tuple[str, str, list[str]]:
    """Generate lldb preset for extracting H.264 motion vectors.

    Returns:
        (breakpoint_func, condition, expressions) tuple for run_lldb().
    """
    if lists is None:
        lists = [0, 1]

    func = "ff_h264_hl_decode_mb"
    condition = (
        f"sl->mb_x == {mb_x} && sl->mb_y == {mb_y} "
        f"&& sl->slice_type_nos == {slice_type_nos}"
    )

    expressions = ["(int)(h->cur_pic_ptr->field_poc[0])"]

    # MVs for requested lists
    for list_idx in lists:
        for blk in range(16):
            s8 = SCAN8[blk]
            expressions.append(f"sl->mv_cache[{list_idx}][{s8}]")

    # Refs (one per 8x8 partition, cast to int for clean output)
    for list_idx in lists:
        for part in range(4):
            s8 = SCAN8[part * 4]
            expressions.append(f"(int)sl->ref_cache[{list_idx}][{s8}]")

    # Sub MB types
    for i in range(4):
        expressions.append(f"(int)sl->sub_mb_type[{i}]")

    return func, condition, expressions


def parse_h264_mv_result(raw_values: list[str], lists: list[int]) -> dict:
    """Parse raw lldb values from h264_mv_preset into structured data.

    Returns dict with keys: poc, mvs, refs, sub_mb_type.
    """
    idx = 0
    result = {}

    # POC
    try:
        result["poc"] = parse_lldb_int(raw_values[idx])
    except (ValueError, IndexError):
        result["poc"] = -1
    idx += 1

    # MVs per list
    result["mvs"] = {}
    for list_idx in lists:
        mvs = []
        for _ in range(16):
            if idx < len(raw_values):
                try:
                    mvs.append(parse_lldb_int_pair(raw_values[idx]))
                except ValueError:
                    mvs.append((0, 0))
            else:
                mvs.append((0, 0))
            idx += 1
        result["mvs"][list_idx] = mvs

    # Refs per list
    result["refs"] = {}
    for list_idx in lists:
        refs = []
        for _ in range(4):
            if idx < len(raw_values):
                try:
                    refs.append(parse_lldb_int(raw_values[idx]))
                except ValueError:
                    refs.append(-99)
            else:
                refs.append(-99)
            idx += 1
        result["refs"][list_idx] = refs

    # Sub MB types
    sub_types = []
    for _ in range(4):
        if idx < len(raw_values):
            try:
                sub_types.append(parse_lldb_int(raw_values[idx]))
            except ValueError:
                sub_types.append(0)
        else:
            sub_types.append(0)
        idx += 1
    result["sub_mb_type"] = sub_types

    return result


def h264_chroma_dc_preset(
    plane: str = "V",
) -> tuple[str, None, list[str]]:
    """Generate lldb preset for extracting H.264 chroma DC coefficients.

    Note: The breakpoint has no condition — it fires for every call.
    Caller must use ignore_count to reach the target frame/plane.

    Returns:
        (breakpoint_func, None, expressions) tuple for run_lldb().
    """
    regs = get_register_names()

    func = "ff_h264_chroma_dc_dequant_idct8"
    expressions = [
        f"(int)((int16_t*){regs.arg0})[0]",
        f"(int)((int16_t*){regs.arg0})[16]",
        f"(int)((int16_t*){regs.arg0})[32]",
        f"(int)((int16_t*){regs.arg0})[48]",
        f"(int){regs.arg1}",
    ]

    return func, None, expressions
