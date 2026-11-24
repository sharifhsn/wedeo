"""Core harness utilities, @e2e_test metadata decorator, binary discovery, and subprocess helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

FEATURES: tuple[str, ...] = tuple(f"F{i}" for i in range(1, 15))
BASELINE_FEATURES: tuple[str, ...] = ("F1", "F2", "F3")
TIERS: tuple[int, ...] = (1, 2, 3, 4)
MILESTONES: tuple[str, ...] = ("baseline", "M2", "M3", "M4", "M5", "M6")

MILESTONE_FEATURES: dict[str, tuple[str, ...]] = {
    "baseline": ("F1", "F2", "F3"),
    "M2": ("F4", "F5", "F6"),
    "M3": ("F7", "F8", "F9"),
    "M4": ("F10",),
    "M5": ("F11", "F12"),
    "M6": ("F13", "F14"),
}


@dataclass(frozen=True)
class TestMeta:
    """Metadata attached to every E2E test method via @e2e_test."""

    feature: str
    tier: int
    milestone: str
    features: tuple[str, ...]


def e2e_test(
    feature: str,
    tier: int,
    milestone: str,
    features: Iterable[str] | None = None,
) -> Callable[[F], F]:
    """Tag a unittest.TestCase method with validated E2E metadata."""
    if feature not in FEATURES:
        raise ValueError(f"Invalid feature {feature!r}; expected one of {FEATURES}")
    if tier not in TIERS:
        raise ValueError(f"Invalid tier {tier!r}; expected one of {TIERS}")
    if milestone not in MILESTONES:
        raise ValueError(
            f"Invalid milestone {milestone!r}; expected one of {MILESTONES}"
        )

    if features is None:
        resolved_features: tuple[str, ...] = (feature,)
    else:
        raw = tuple(features)
        if not raw:
            raise ValueError("features must be non-empty when provided")
        for item in raw:
            if item not in FEATURES:
                raise ValueError(
                    f"Invalid feature {item!r} in features; expected one of {FEATURES}"
                )
        if feature not in raw:
            resolved_features = (feature, *dict.fromkeys(raw))
        else:
            resolved_features = tuple(dict.fromkeys(raw))

    meta = TestMeta(
        feature=feature,
        tier=tier,
        milestone=milestone,
        features=resolved_features,
    )

    def decorator(func: F) -> F:
        setattr(func, "_e2e_meta", meta)
        return func

    return decorator


def repo_root() -> Path:
    """Return the repository root (/home/sharif/wedeo)."""
    return Path(__file__).resolve().parents[3]


def augmented_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of os.environ with ~/.cargo/bin and ~/.local/bin on PATH."""
    env = os.environ.copy()
    home = Path.home()
    candidates = [str(home / ".cargo" / "bin"), str(home / ".local" / "bin")]
    current_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    for cand in reversed(candidates):
        if cand not in current_parts:
            current_parts.insert(0, cand)
    env["PATH"] = os.pathsep.join(current_parts)
    if extra:
        env.update(extra)
    return env


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def build_wedeo_cli(root: Path | None = None, *, release: bool = False) -> Path:
    """Build wedeo-cli via cargo and return the built binary path."""
    base = root or repo_root()
    cmd = ["cargo", "build", "-p", "wedeo-cli"]
    if release:
        cmd.append("--release")
    proc = subprocess.run(
        cmd,
        cwd=str(base),
        env=augmented_env(),
        capture_output=True,
        text=True,
        timeout=600.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cargo build -p wedeo-cli failed (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    profile = "release" if release else "debug"
    built = base / "target" / profile / "wedeo"
    if not _is_executable(built):
        raise FileNotFoundError(f"Built binary missing or not executable: {built}")
    return built


def find_wedeo_bin(
    root: Path | None = None, *, auto_build: bool = False
) -> Path:
    """Resolve WEDEO_BIN or pick the newest existing debug/release wedeo binary."""
    override = os.environ.get("WEDEO_BIN")
    if override:
        p = Path(override)
        if not _is_executable(p):
            raise FileNotFoundError(
                f"WEDEO_BIN={override!r} does not exist or is not executable"
            )
        return p.resolve()

    base = root or repo_root()
    candidates = [
        base / "target" / "debug" / "wedeo",
        base / "target" / "release" / "wedeo",
    ]
    existing = [p for p in candidates if _is_executable(p)]
    if existing:
        newest = max(existing, key=lambda p: p.stat().st_mtime_ns)
        return newest.resolve()

    if auto_build:
        return build_wedeo_cli(base, release=False).resolve()

    raise FileNotFoundError(
        "No wedeo binary found under target/{debug,release}/wedeo; "
        "pass --build or set WEDEO_BIN"
    )


def find_oracle_bins(root: Path | None = None) -> tuple[Path, Path]:
    """Return (ffmpeg, ffprobe) from target/ffmpeg-oracle or env overrides."""
    base = root or repo_root()
    oracle_dir = Path(
        os.environ.get("FFMPEG_ORACLE_DIR", str(base / "target" / "ffmpeg-oracle"))
    )
    ffmpeg_path = Path(
        os.environ.get("WEDEO_FFMPEG_ORACLE", str(oracle_dir / "ffmpeg"))
    )
    ffprobe_path = Path(
        os.environ.get("WEDEO_FFPROBE_ORACLE", str(oracle_dir / "ffprobe"))
    )
    for label, path in (("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)):
        if not _is_executable(path):
            raise FileNotFoundError(
                f"Oracle {label} binary missing or not executable: {path}"
            )
    return ffmpeg_path.resolve(), ffprobe_path.resolve()


def find_system_ffmpeg_bin() -> Path:
    """Locate system ffmpeg (with libdav1d/libx264/libaom-av1) for AV1 decode & fixture synth."""
    override = os.environ.get("WEDEO_SYSTEM_FFMPEG")
    if override:
        p = Path(override)
        if not _is_executable(p):
            raise FileNotFoundError(
                f"WEDEO_SYSTEM_FFMPEG={override!r} is missing or not executable"
            )
        return p.resolve()

    usr_bin = Path("/usr/bin/ffmpeg")
    if _is_executable(usr_bin):
        return usr_bin

    found = shutil.which("ffmpeg", path=augmented_env()["PATH"])
    if found and _is_executable(Path(found)):
        return Path(found).resolve()

    raise FileNotFoundError("System ffmpeg binary not found at /usr/bin/ffmpeg or PATH")


def find_node_bin() -> Path:
    """Locate node executable for WASM/TS tests."""
    found = shutil.which("node", path=augmented_env()["PATH"])
    if not found:
        raise FileNotFoundError("node binary not found on PATH")
    p = Path(found)
    if not _is_executable(p):
        raise FileNotFoundError(f"node binary is not executable: {p}")
    return p.resolve()


@dataclass(frozen=True)
class CmdResult:
    """Captured result from running an external command."""

    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    args: tuple[str, ...]


def run_cmd(
    cmd: Iterable[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
    timeout: float = 120.0,
) -> CmdResult:
    """Run an arbitrary command and return a CmdResult."""
    argv = tuple(str(part) for part in cmd)
    run_env = augmented_env() if env is None else env
    start = time.monotonic()
    proc = subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd is not None else None,
        env=run_env,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start
    return CmdResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_sec=elapsed,
        args=argv,
    )


def run_wedeo(
    args: Iterable[str | Path],
    *,
    wedeo_bin: Path | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
    timeout: float = 120.0,
) -> CmdResult:
    """Invoke the resolved wedeo binary with the given CLI arguments."""
    exe = wedeo_bin or find_wedeo_bin()
    return run_cmd(
        [exe, *args],
        cwd=cwd,
        env=env,
        stdin_text=stdin_text,
        timeout=timeout,
    )


def assert_cli_error(
    res: CmdResult,
    expected_substring: str,
    *,
    expected_code: int = 1,
    allow_stdout: bool = False,
) -> None:
    """Assert that a CLI invocation failed with expected exit code and error message."""
    if res.returncode != expected_code:
        raise AssertionError(
            f"Expected exit code {expected_code}, got {res.returncode} "
            f"for {res.args!r}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    if not allow_stdout and res.stdout != "":
        raise AssertionError(
            f"Expected empty stdout on error for {res.args!r}, got:\n{res.stdout}"
        )
    if not res.stderr.startswith("Error: "):
        raise AssertionError(
            f"Expected stderr to start with 'Error: ' for {res.args!r}, "
            f"got:\n{res.stderr}"
        )
    if expected_substring not in res.stderr:
        raise AssertionError(
            f"Expected {expected_substring!r} in stderr for {res.args!r}, "
            f"got:\n{res.stderr}"
        )
