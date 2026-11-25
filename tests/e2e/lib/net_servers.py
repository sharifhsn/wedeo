"""Local network, TLS, pipe/FD, HTTP(S), and RTMP/SRT loopback servers for E2E tests."""

from __future__ import annotations

import http.server
import os
import socket
import ssl
import struct
import subprocess
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Literal

from .harness import (
    CmdResult,
    augmented_env,
    find_system_ffmpeg_bin,
    find_wedeo_bin,
)

# ---------------------------------------------------------------------------
# Binary pipe / FD helpers (F4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryCmdResult:
    """Captured result from running a command with raw bytes stdout."""

    returncode: int
    stdout: bytes
    stderr: str
    duration_sec: float
    args: tuple[str, ...]


def run_cmd_bytes(
    cmd: Iterable[str | Path],
    *,
    stdin_bytes: bytes | None = None,
    pass_fds: tuple[int, ...] = (),
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> BinaryCmdResult:
    """Run an external command capturing raw stdout bytes and decoded stderr."""
    argv = tuple(str(part) for part in cmd)
    run_env = augmented_env() if env is None else env
    start = time.monotonic()
    proc = subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd is not None else None,
        env=run_env,
        input=stdin_bytes,
        pass_fds=pass_fds,
        capture_output=True,
        text=False,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start
    return BinaryCmdResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr.decode("utf-8", errors="replace"),
        duration_sec=elapsed,
        args=argv,
    )


def run_wedeo_bytes(
    args: Iterable[str | Path],
    *,
    wedeo_bin: Path | None = None,
    stdin_bytes: bytes | None = None,
    pass_fds: tuple[int, ...] = (),
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> BinaryCmdResult:
    """Invoke the resolved wedeo binary capturing raw stdout bytes."""
    exe = wedeo_bin or find_wedeo_bin()
    return run_cmd_bytes(
        [exe, *args],
        stdin_bytes=stdin_bytes,
        pass_fds=pass_fds,
        cwd=cwd,
        env=env,
        timeout=timeout,
    )


def stream_chunks_via_pipe(
    cmd: Iterable[str | Path],
    chunks: Iterable[bytes],
    *,
    use_extra_fd: bool = False,
    delay_sec: float = 0.0,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> BinaryCmdResult:
    """Stream chunks concurrently into stdin or an inherited FD while draining stdout/stderr."""
    run_env = augmented_env() if env is None else env
    argv_raw = [str(part) for part in cmd]

    in_r: int | None = None
    in_w: int | None = None
    pass_fds: tuple[int, ...] = ()
    stdin_opt: int | None = subprocess.PIPE

    if use_extra_fd:
        in_r, in_w = os.pipe()
        argv = tuple(part.replace("{fd}", str(in_r)) for part in argv_raw)
        pass_fds = (in_r,)
        stdin_opt = subprocess.DEVNULL
    else:
        argv = tuple(argv_raw)

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            env=run_env,
            stdin=stdin_opt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=pass_fds,
        )
    except Exception:
        if in_r is not None:
            os.close(in_r)
        if in_w is not None:
            os.close(in_w)
        raise

    if in_r is not None:
        os.close(in_r)

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    def _writer() -> None:
        writer_stream = (
            os.fdopen(in_w, "wb", closefd=True)
            if in_w is not None
            else proc.stdin
        )
        assert writer_stream is not None
        try:
            for chunk in chunks:
                if not chunk:
                    continue
                writer_stream.write(chunk)
                writer_stream.flush()
                if delay_sec > 0.0:
                    time.sleep(delay_sec)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                writer_stream.close()
            except OSError:
                pass

    def _drain(pipe: Any, out: list[bytes]) -> None:
        if pipe is None:
            return
        try:
            while True:
                data = pipe.read(65536)
                if not data:
                    break
                out.append(data)
        except OSError:
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    t_write = threading.Thread(target=_writer, daemon=True)
    t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_chunks), daemon=True)
    t_write.start()
    t_out.start()
    t_err.start()

    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)
        t_write.join(timeout=2.0)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        raise

    t_write.join(timeout=5.0)
    t_out.join(timeout=5.0)
    t_err.join(timeout=5.0)
    elapsed = time.monotonic() - start
    return BinaryCmdResult(
        returncode=rc,
        stdout=b"".join(stdout_chunks),
        stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        duration_sec=elapsed,
        args=argv,
    )


def assert_binary_cli_error(
    res: BinaryCmdResult,
    expected_substring: str | None = None,
    *,
    forbidden_substring: str | None = "No such file or directory",
    expected_code: int = 1,
    allow_stdout: bool = False,
) -> None:
    """Assert a binary CLI invocation failed cleanly without file-source false passes."""
    if res.returncode != expected_code:
        raise AssertionError(
            f"Expected exit code {expected_code}, got {res.returncode} "
            f"for {res.args!r}\nstdout_len={len(res.stdout)}\nstderr:\n{res.stderr}"
        )
    if not allow_stdout and res.stdout != b"":
        raise AssertionError(
            f"Expected empty stdout on error for {res.args!r}, "
            f"got {len(res.stdout)} bytes: {res.stdout[:128]!r}"
        )
    if not res.stderr.startswith("Error: "):
        raise AssertionError(
            f"Expected stderr to start with 'Error: ' for {res.args!r}, "
            f"got:\n{res.stderr}"
        )
    if "panicked" in res.stderr.lower():
        raise AssertionError(
            f"Unexpected panic in stderr for {res.args!r}:\n{res.stderr}"
        )
    if forbidden_substring is not None and forbidden_substring in res.stderr:
        raise AssertionError(
            f"Forbidden baseline fallback error {forbidden_substring!r} found in "
            f"stderr for {res.args!r}:\n{res.stderr}"
        )
    if (
        expected_substring is not None
        and expected_substring.lower() not in res.stderr.lower()
    ):
        raise AssertionError(
            f"Expected {expected_substring!r} (case-insensitive) in stderr for "
            f"{res.args!r}, got:\n{res.stderr}"
        )


def assert_net_protocol_error(
    res: CmdResult | BinaryCmdResult,
    *,
    expected_keywords: tuple[str, ...],
    forbidden_substring: str = "No such file or directory",
    expected_code: int = 1,
) -> None:
    """Assert a network/protocol CLI invocation failed cleanly with domain error."""
    if res.returncode != expected_code:
        raise AssertionError(
            f"Expected exit code {expected_code}, got {res.returncode} "
            f"for {res.args!r}\nstdout:\n{res.stdout!r}\nstderr:\n{res.stderr}"
        )
    empty_stdout = b"" if isinstance(res.stdout, bytes) else ""
    if res.stdout != empty_stdout:
        raise AssertionError(
            f"Expected empty stdout on error for {res.args!r}, got:\n{res.stdout!r}"
        )
    if not res.stderr.startswith("Error: "):
        raise AssertionError(
            f"Expected stderr to start with 'Error: ' for {res.args!r}, "
            f"got:\n{res.stderr}"
        )
    if forbidden_substring and forbidden_substring in res.stderr:
        raise AssertionError(
            f"Forbidden baseline fallback error {forbidden_substring!r} found in "
            f"stderr for {res.args!r}:\n{res.stderr}"
        )
    lowered = res.stderr.lower()
    if "panicked" in lowered:
        raise AssertionError(
            f"Unexpected panic in stderr for {res.args!r}:\n{res.stderr}"
        )
    if not any(kw.lower() in lowered for kw in expected_keywords):
        raise AssertionError(
            f"Expected one of {expected_keywords!r} in stderr for {res.args!r}, "
            f"got:\n{res.stderr}"
        )


def synth_fragmented_mp4(
    out_path: Path,
    *,
    width: int = 64,
    height: int = 64,
    fps: int = 10,
    duration: float = 0.6,
    gop: int = 3,
) -> Path:
    """Synthesize a fragmented MP4 (ftyp+moov+moof+mdat) via system ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-movflags",
        "frag_keyframe+empty_moov",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to synthesize fragmented MP4 ({proc.returncode}): {proc.stderr}"
        )
    return out_path


def synth_nonfaststart_mp4(
    out_path: Path,
    *,
    width: int = 64,
    height: int = 64,
    fps: int = 10,
    duration: float = 0.5,
    gop: int = 15,
    video_bitrate: str | None = None,
    with_audio: bool = False,
) -> Path:
    """Synthesize a non-faststart MP4 with moov box placed after mdat."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    if video_bitrate is not None:
        cmd.extend(["-b:v", video_bitrate, "-maxrate", video_bitrate, "-bufsize", video_bitrate])
    if with_audio:
        cmd.extend(["-c:a", "aac", "-ac", "1", "-ar", "48000", "-shortest"])
    cmd.append(str(out_path))

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to synthesize non-faststart MP4 ({proc.returncode}): {proc.stderr}"
        )
    return out_path


def scan_mp4_top_level_boxes(data: bytes) -> list[tuple[str, int, int]]:
    """Return [(box_type, offset, size), ...] for top-level ISOBMFF boxes."""
    boxes: list[tuple[str, int, int]] = []
    pos = 0
    total = len(data)
    while pos + 8 <= total:
        size32, kind_bytes = struct.unpack(">I4s", data[pos : pos + 8])
        kind = kind_bytes.decode("latin1")
        if size32 == 1:
            if pos + 16 > total:
                break
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
        elif size32 == 0:
            size = total - pos
        else:
            size = size32
        if size < 8:
            break
        boxes.append((kind, pos, size))
        pos += size
    return boxes


# ---------------------------------------------------------------------------
# Port & /proc/net socket helpers
# ---------------------------------------------------------------------------


def find_free_port(kind: int = socket.SOCK_STREAM) -> int:
    """Allocate an ephemeral port on 127.0.0.1 and release it immediately."""
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def find_closed_localhost_port(kind: int = socket.SOCK_STREAM) -> int:
    """Return a localhost port known to have no active listener."""
    return find_free_port(kind=kind)


def _proc_net_has_port(proto: str, port: int) -> bool:
    target_state = "0A" if proto == "tcp" else "07"
    port_suffix = f":{port:04X}"
    paths = [Path(f"/proc/net/{proto}"), Path(f"/proc/net/{proto}6")]
    for p in paths:
        if not p.is_file():
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            local_addr = parts[1].upper()
            state = parts[3].upper()
            if local_addr.endswith(port_suffix) and state == target_state:
                return True
    return False


def wait_proc_net_listening(
    proto: str,
    port: int,
    *,
    timeout: float = 5.0,
    proc: subprocess.Popen[Any] | None = None,
) -> None:
    """Passively wait until `port` is listening in /proc/net/{tcp,udp}."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _proc_net_has_port(proto, port):
            return
        if proc is not None and proc.poll() is not None:
            err_msg = ""
            if proc.stderr is not None:
                try:
                    raw = proc.stderr.read()
                    err_msg = (
                        raw.decode("utf-8", errors="replace")
                        if isinstance(raw, bytes)
                        else str(raw)
                    )
                except OSError:
                    pass
            raise RuntimeError(
                f"Listener process exited early (rc={proc.returncode}) while waiting "
                f"for {proto} port {port}: {err_msg}"
            )
        time.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {proto} port {port} to listen")


# ---------------------------------------------------------------------------
# TLS Certificate Bundle (2-cert CA + SAN leaf)
# ---------------------------------------------------------------------------


class TlsCertBundle:
    """Managed temporary 2-cert CA + localhost/127.0.0.1 SAN certificate bundle."""

    def __init__(
        self,
        tmpdir: tempfile.TemporaryDirectory[str],
        ca_cert_path: Path,
        server_cert_path: Path,
        server_key_path: Path,
    ) -> None:
        self._tmpdir = tmpdir
        self.ca_cert_path = ca_cert_path
        self.server_cert_path = server_cert_path
        self.server_key_path = server_key_path

    @classmethod
    def generate(cls) -> TlsCertBundle:
        tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_tls_")
        root = Path(tmpdir.name)
        try:
            ca_key = root / "ca.key"
            ca_crt = root / "ca.crt"
            ca_cnf = root / "ca.cnf"
            ca_cnf.write_text(
                "[req]\n"
                "distinguished_name = dn\n"
                "x509_extensions = v3_ca\n"
                "prompt = no\n"
                "[dn]\n"
                "CN = wedeo-e2e-ca\n"
                "[v3_ca]\n"
                "subjectKeyIdentifier = hash\n"
                "authorityKeyIdentifier = keyid:always,issuer\n"
                "basicConstraints = critical, CA:TRUE\n"
                "keyUsage = critical, keyCertSign, cRLSign\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-days",
                    "2",
                    "-keyout",
                    str(ca_key),
                    "-out",
                    str(ca_crt),
                    "-config",
                    str(ca_cnf),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30.0,
            )

            srv_key = root / "srv.key"
            srv_csr = root / "srv.csr"
            srv_crt = root / "srv.crt"
            srv_cnf = root / "srv.cnf"
            srv_ext = root / "srv.ext"
            srv_cnf.write_text(
                "[req]\n"
                "distinguished_name = dn\n"
                "prompt = no\n"
                "[dn]\n"
                "CN = localhost\n",
                encoding="utf-8",
            )
            srv_ext.write_text(
                "basicConstraints = critical, CA:FALSE\n"
                "keyUsage = critical, digitalSignature, keyEncipherment\n"
                "extendedKeyUsage = serverAuth\n"
                "subjectAltName = DNS:localhost,IP:127.0.0.1\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-new",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(srv_key),
                    "-out",
                    str(srv_csr),
                    "-config",
                    str(srv_cnf),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            subprocess.run(
                [
                    "openssl",
                    "x509",
                    "-req",
                    "-in",
                    str(srv_csr),
                    "-CA",
                    str(ca_crt),
                    "-CAkey",
                    str(ca_key),
                    "-CAcreateserial",
                    "-out",
                    str(srv_crt),
                    "-days",
                    "2",
                    "-extfile",
                    str(srv_ext),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
        except Exception:
            tmpdir.cleanup()
            raise

        return cls(
            tmpdir=tmpdir,
            ca_cert_path=ca_crt,
            server_cert_path=srv_crt,
            server_key_path=srv_key,
        )

    def close(self) -> None:
        self._tmpdir.cleanup()

    def __enter__(self) -> TlsCertBundle:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# HTTP/HTTPS Media Server (F5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedHttpRequest:
    """Immutable record of one HTTP request received by HttpMediaServer."""

    method: str
    path: str
    headers: dict[str, str]
    range_header: str | None
    status: int
    bytes_sent: int
    was_chunked_upload: bool
    body: bytes


@dataclass
class RouteSpec:
    """Behavior specification for a route served or accepted by HttpMediaServer."""

    data: bytes = b""
    content_type: str = "application/octet-stream"
    accept_ranges: bool = True
    chunked_response: bool = False
    chunk_size: int = 4096
    throttle_ms: float = 0.0
    status_override: int | None = None
    fail_first_n: tuple[int, int] | None = None
    stall_before_headers_sec: float = 0.0
    stall_mid_body_after_bytes: int | None = None
    stall_mid_body_sec: float = 0.0
    truncate_after_bytes: int | None = None
    force_416: bool = False
    allow_upload: bool = False


class _QuietThreadingHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Suppress noisy stack traces on expected client disconnects / untrusted TLS rejections.
        return


def _parse_single_byte_range(
    header_value: str, total: int
) -> tuple[int, int] | None:
    """Parse RFC 9110 single byte range into inclusive (start, end) or None on 416."""
    val = header_value.strip()
    if not val.lower().startswith("bytes="):
        return None
    spec = val[6:].strip()
    if not spec or "," in spec or "-" not in spec:
        return None
    if total <= 0:
        return None
    start_s, end_s = spec.split("-", 1)
    start_s = start_s.strip()
    end_s = end_s.strip()
    try:
        if start_s == "":
            suffix = int(end_s)
            if suffix <= 0:
                return None
            start = max(0, total - suffix)
            end = total - 1
            return (start, end)
        start = int(start_s)
        if start < 0 or start >= total:
            return None
        if end_s == "":
            end = total - 1
        else:
            end = int(end_s)
            if end < start:
                return None
            end = min(end, total - 1)
        return (start, end)
    except ValueError:
        return None


class HttpMediaServer:
    """Local ephemeral-port HTTP/1.1 / HTTPS server supporting Range, chunked, and faults."""

    def __init__(
        self,
        routes: dict[str, RouteSpec] | None = None,
        *,
        tls: bool = False,
        default_upload_prefix: str = "/upload/",
    ) -> None:
        self._routes: dict[str, RouteSpec] = dict(routes or {})
        self._tls = tls
        self._default_upload_prefix = default_upload_prefix
        self._lock = threading.Lock()
        self._requests: list[RecordedHttpRequest] = []
        self._fail_counts: dict[str, int] = {}
        self._stop_event = threading.Event()
        self._tls_bundle: TlsCertBundle | None = None
        self._httpd: _QuietThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    @property
    def scheme(self) -> str:
        return "https" if self._tls else "http"

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://127.0.0.1:{self.port}"

    @property
    def ca_cert_path(self) -> Path | None:
        return self._tls_bundle.ca_cert_path if self._tls_bundle is not None else None

    def url(self, path: str) -> str:
        norm = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{norm}"

    @property
    def requests(self) -> list[RecordedHttpRequest]:
        with self._lock:
            return list(self._requests)

    @property
    def uploads(self) -> list[RecordedHttpRequest]:
        with self._lock:
            return [r for r in self._requests if r.method in ("PUT", "POST")]

    def add_route(self, path: str, spec: RouteSpec) -> str:
        norm = path if path.startswith("/") else f"/{path}"
        with self._lock:
            self._routes[norm] = spec
        return self.url(norm)

    def add_file_route(
        self,
        path: str,
        file_path: Path,
        *,
        mode: Literal["range", "sequential", "chunked"] = "range",
        **route_kwargs: Any,
    ) -> str:
        data = file_path.read_bytes()
        if mode == "range":
            defaults: dict[str, Any] = {
                "accept_ranges": True,
                "chunked_response": False,
            }
        elif mode == "sequential":
            defaults = {"accept_ranges": False, "chunked_response": False}
        elif mode == "chunked":
            defaults = {"accept_ranges": False, "chunked_response": True}
        else:
            raise ValueError(f"Unknown route mode {mode!r}")
        defaults.update(route_kwargs)
        return self.add_route(path, RouteSpec(data=data, **defaults))

    def _record(self, req: RecordedHttpRequest) -> None:
        with self._lock:
            self._requests.append(req)

    def _get_route(self, raw_path: str, *, for_upload: bool) -> tuple[str, RouteSpec | None]:
        path = urllib.parse.urlsplit(raw_path).path
        with self._lock:
            spec = self._routes.get(path)
            if spec is None and for_upload and path.startswith(self._default_upload_prefix):
                spec = RouteSpec(allow_upload=True)
                self._routes[path] = spec
            return path, spec

    def _consume_fail_first_n(self, path: str, spec: RouteSpec) -> int | None:
        if spec.fail_first_n is None:
            return None
        status_code, limit = spec.fail_first_n
        with self._lock:
            seen = self._fail_counts.get(path, 0)
            if seen < limit:
                self._fail_counts[path] = seen + 1
                return status_code
        return None

    def __enter__(self) -> HttpMediaServer:
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

            def _lower_headers(self) -> dict[str, str]:
                return {k.lower(): v for k, v in self.headers.items()}

            def _send_simple(self, status: int, extra_headers: dict[str, str] | None = None) -> None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                if extra_headers:
                    for k, v in extra_headers.items():
                        self.send_header(k, v)
                self.end_headers()
                self.close_connection = True

            def _handle_read(self, *, send_body: bool) -> None:
                headers = self._lower_headers()
                range_hdr = headers.get("range")
                path, spec = parent._get_route(self.path, for_upload=False)
                if spec is None:
                    self._send_simple(404)
                    parent._record(
                        RecordedHttpRequest(
                            self.command, path, headers, range_hdr, 404, 0, False, b""
                        )
                    )
                    return

                if spec.stall_before_headers_sec > 0.0:
                    if parent._stop_event.wait(spec.stall_before_headers_sec):
                        self.close_connection = True
                        return

                fail_status = parent._consume_fail_first_n(path, spec)
                if fail_status is not None:
                    self._send_simple(fail_status, {"Retry-After": "0"})
                    parent._record(
                        RecordedHttpRequest(
                            self.command,
                            path,
                            headers,
                            range_hdr,
                            fail_status,
                            0,
                            False,
                            b"",
                        )
                    )
                    return

                if spec.status_override is not None:
                    st = spec.status_override
                    self._send_simple(st)
                    parent._record(
                        RecordedHttpRequest(
                            self.command, path, headers, range_hdr, st, 0, False, b""
                        )
                    )
                    return

                total = len(spec.data)
                etag = f'"{sha256(spec.data).hexdigest()[:16]}"'

                if spec.force_416 and range_hdr is not None:
                    self._send_simple(416, {"Content-Range": f"bytes */{total}"})
                    parent._record(
                        RecordedHttpRequest(
                            self.command, path, headers, range_hdr, 416, 0, False, b""
                        )
                    )
                    return

                status = 200
                payload = spec.data
                resp_headers: dict[str, str] = {
                    "Content-Type": spec.content_type,
                    "ETag": etag,
                    "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT",
                }

                if spec.accept_ranges:
                    resp_headers["Accept-Ranges"] = "bytes"
                    if range_hdr is not None:
                        parsed = _parse_single_byte_range(range_hdr, total)
                        if parsed is None:
                            self._send_simple(416, {"Content-Range": f"bytes */{total}"})
                            parent._record(
                                RecordedHttpRequest(
                                    self.command,
                                    path,
                                    headers,
                                    range_hdr,
                                    416,
                                    0,
                                    False,
                                    b"",
                                )
                            )
                            return
                        start, end = parsed
                        status = 206
                        payload = spec.data[start : end + 1]
                        resp_headers["Content-Range"] = f"bytes {start}-{end}/{total}"

                use_chunked = (
                    send_body
                    and not spec.accept_ranges
                    and spec.chunked_response
                    and spec.truncate_after_bytes is None
                    and spec.stall_mid_body_after_bytes is None
                )

                self.send_response(status)
                for k, v in resp_headers.items():
                    self.send_header(k, v)
                if use_chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                else:
                    self.send_header("Content-Length", str(len(payload)))
                if (
                    spec.truncate_after_bytes is not None
                    or spec.stall_mid_body_after_bytes is not None
                ):
                    self.send_header("Connection", "close")
                self.end_headers()

                if not send_body:
                    parent._record(
                        RecordedHttpRequest(
                            self.command,
                            path,
                            headers,
                            range_hdr,
                            status,
                            0,
                            False,
                            b"",
                        )
                    )
                    return

                bytes_sent = 0
                try:
                    if use_chunked:
                        step = max(1, spec.chunk_size)
                        for idx in range(0, len(payload), step):
                            if parent._stop_event.is_set():
                                break
                            chunk = payload[idx : idx + step]
                            self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                            bytes_sent += len(chunk)
                            if spec.throttle_ms > 0.0:
                                if parent._stop_event.wait(spec.throttle_ms / 1000.0):
                                    break
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    elif spec.truncate_after_bytes is not None:
                        prefix = payload[: spec.truncate_after_bytes]
                        self.wfile.write(prefix)
                        self.wfile.flush()
                        bytes_sent = len(prefix)
                        self.close_connection = True
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        try:
                            self.connection.close()
                        except OSError:
                            pass
                    elif spec.stall_mid_body_after_bytes is not None:
                        cut = spec.stall_mid_body_after_bytes
                        prefix = payload[:cut]
                        self.wfile.write(prefix)
                        self.wfile.flush()
                        bytes_sent = len(prefix)
                        parent._stop_event.wait(spec.stall_mid_body_sec)
                        self.close_connection = True
                    else:
                        self.wfile.write(payload)
                        self.wfile.flush()
                        bytes_sent = len(payload)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError):
                    self.close_connection = True

                parent._record(
                    RecordedHttpRequest(
                        self.command,
                        path,
                        headers,
                        range_hdr,
                        status,
                        bytes_sent,
                        False,
                        b"",
                    )
                )

            def _handle_upload(self) -> None:
                headers = self._lower_headers()
                path, spec = parent._get_route(self.path, for_upload=True)
                if spec is None or not spec.allow_upload:
                    st = 404 if spec is None else 405
                    self._send_simple(st)
                    parent._record(
                        RecordedHttpRequest(
                            self.command, path, headers, None, st, 0, False, b""
                        )
                    )
                    return

                if spec.stall_before_headers_sec > 0.0:
                    if parent._stop_event.wait(spec.stall_before_headers_sec):
                        self.close_connection = True
                        return

                fail_status = parent._consume_fail_first_n(path, spec)
                if fail_status is not None:
                    self._send_simple(fail_status, {"Retry-After": "0"})
                    parent._record(
                        RecordedHttpRequest(
                            self.command,
                            path,
                            headers,
                            None,
                            fail_status,
                            0,
                            False,
                            b"",
                        )
                    )
                    return

                if spec.status_override is not None:
                    st = spec.status_override
                    self._send_simple(st)
                    parent._record(
                        RecordedHttpRequest(
                            self.command, path, headers, None, st, 0, False, b""
                        )
                    )
                    return

                if headers.get("expect", "").lower() == "100-continue":
                    self.send_response_only(100)
                    self.end_headers()

                te = headers.get("transfer-encoding", "").lower()
                was_chunked = "chunked" in te
                buf = bytearray()
                try:
                    if was_chunked:
                        while True:
                            size_line = self.rfile.readline(65536)
                            if not size_line:
                                break
                            size_token = size_line.split(b";", 1)[0].strip()
                            chunk_len = int(size_token, 16)
                            if chunk_len == 0:
                                while True:
                                    trailer = self.rfile.readline(65536)
                                    if not trailer or trailer in (b"\r\n", b"\n"):
                                        break
                                break
                            buf.extend(self.rfile.read(chunk_len))
                            crlf = self.rfile.read(2)
                            if crlf not in (b"\r\n", b"\n", b""):
                                break
                    elif "content-length" in headers:
                        remaining = int(headers["content-length"])
                        while remaining > 0:
                            chunk = self.rfile.read(min(65536, remaining))
                            if not chunk:
                                break
                            buf.extend(chunk)
                            remaining -= len(chunk)
                    else:
                        while True:
                            chunk = self.rfile.read(65536)
                            if not chunk:
                                break
                            buf.extend(chunk)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError):
                    self.close_connection = True

                body = bytes(buf)
                self._send_simple(200)
                parent._record(
                    RecordedHttpRequest(
                        self.command,
                        path,
                        headers,
                        None,
                        200,
                        0,
                        was_chunked,
                        body,
                    )
                )

            def do_HEAD(self) -> None:
                self._handle_read(send_body=False)

            def do_GET(self) -> None:
                self._handle_read(send_body=True)

            def do_PUT(self) -> None:
                self._handle_upload()

            def do_POST(self) -> None:
                self._handle_upload()

        if self._tls:
            self._tls_bundle = TlsCertBundle.generate()

        httpd = _QuietThreadingHTTPServer(("127.0.0.1", 0), Handler)
        if self._tls and self._tls_bundle is not None:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(
                certfile=str(self._tls_bundle.server_cert_path),
                keyfile=str(self._tls_bundle.server_key_path),
            )
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

        self._httpd = httpd
        self.port = int(httpd.server_address[1])
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop_event.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except OSError:
                pass
            try:
                self._httpd.server_close()
            except OSError:
                pass
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._tls_bundle is not None:
            self._tls_bundle.close()
            self._tls_bundle = None


# ---------------------------------------------------------------------------
# TLS Tunnel & RTMP Truncating Proxy (F6)
# ---------------------------------------------------------------------------


def _pump_bidirectional(
    sock_a: socket.socket,
    sock_b: socket.socket,
    stop_event: threading.Event,
) -> None:
    def _forward(src: socket.socket, dst: socket.socket) -> None:
        src.settimeout(0.2)
        try:
            while not stop_event.is_set():
                try:
                    data = src.recv(65536)
                except socket.timeout:
                    continue
                except (ssl.SSLError, OSError):
                    break
                if not data:
                    break
                try:
                    dst.sendall(data)
                except (ssl.SSLError, OSError):
                    break
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except (ssl.SSLError, OSError):
                pass

    t1 = threading.Thread(target=_forward, args=(sock_a, sock_b), daemon=True)
    t2 = threading.Thread(target=_forward, args=(sock_b, sock_a), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    for s in (sock_a, sock_b):
        try:
            s.close()
        except OSError:
            pass


class TlsTcpTunnel:
    """Local TLS-terminating TCP tunnel forwarding to a plaintext backend port."""

    def __init__(
        self,
        backend_port: int,
        *,
        cert_bundle: TlsCertBundle | None = None,
    ) -> None:
        self._backend_port = backend_port
        self._owns_bundle = cert_bundle is None
        self._bundle = cert_bundle
        self._stop_event = threading.Event()
        self._listen_sock: socket.socket | None = None
        self._active_sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.port: int = 0

    @property
    def ca_cert_path(self) -> Path:
        assert self._bundle is not None
        return self._bundle.ca_cert_path

    def __enter__(self) -> TlsTcpTunnel:
        if self._bundle is None:
            self._bundle = TlsCertBundle.generate()
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(
            certfile=str(self._bundle.server_cert_path),
            keyfile=str(self._bundle.server_key_path),
        )

        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        lsock.settimeout(0.1)
        self._listen_sock = lsock
        self.port = int(lsock.getsockname()[1])

        def _accept_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    raw_client, _ = lsock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                def _handle(conn: socket.socket) -> None:
                    tls_conn: ssl.SSLSocket | None = None
                    backend: socket.socket | None = None
                    try:
                        tls_conn = ssl_ctx.wrap_socket(conn, server_side=True)
                        backend = socket.create_connection(
                            ("127.0.0.1", self._backend_port), timeout=5.0
                        )
                        with self._lock:
                            self._active_sockets.extend([tls_conn, backend])
                        _pump_bidirectional(tls_conn, backend, self._stop_event)
                    except (ssl.SSLError, OSError):
                        for s in (tls_conn, backend, conn):
                            if s is not None:
                                try:
                                    s.close()
                                except OSError:
                                    pass

                threading.Thread(target=_handle, args=(raw_client,), daemon=True).start()

        self._thread = threading.Thread(target=_accept_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop_event.set()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
            self._listen_sock = None
        with self._lock:
            for sock in self._active_sockets:
                try:
                    sock.close()
                except OSError:
                    pass
            self._active_sockets.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._owns_bundle and self._bundle is not None:
            self._bundle.close()
            self._bundle = None


class RtmpTruncatingProxy:
    """TCP proxy that drops the client socket abruptly after N backend->client bytes."""

    def __init__(self, backend_port: int, *, truncate_after_bytes: int) -> None:
        self._backend_port = backend_port
        self._truncate_after = truncate_after_bytes
        self._stop_event = threading.Event()
        self._listen_sock: socket.socket | None = None
        self._active_sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def __enter__(self) -> RtmpTruncatingProxy:
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(4)
        lsock.settimeout(0.1)
        self._listen_sock = lsock
        self.port = int(lsock.getsockname()[1])

        def _accept_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    client, _ = lsock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                def _handle(cli: socket.socket) -> None:
                    backend: socket.socket | None = None
                    try:
                        backend = socket.create_connection(
                            ("127.0.0.1", self._backend_port), timeout=5.0
                        )
                        with self._lock:
                            self._active_sockets.extend([cli, backend])

                        def _c2b() -> None:
                            cli.settimeout(0.2)
                            while not self._stop_event.is_set():
                                try:
                                    data = cli.recv(65536)
                                except socket.timeout:
                                    continue
                                except OSError:
                                    break
                                if not data:
                                    break
                                try:
                                    backend.sendall(data)
                                except OSError:
                                    break

                        def _b2c() -> None:
                            backend.settimeout(0.2)
                            sent = 0
                            while not self._stop_event.is_set():
                                try:
                                    data = backend.recv(4096)
                                except socket.timeout:
                                    continue
                                except OSError:
                                    break
                                if not data:
                                    break
                                remaining = self._truncate_after - sent
                                if remaining <= 0:
                                    break
                                if len(data) >= remaining:
                                    try:
                                        cli.sendall(data[:remaining])
                                        cli.setsockopt(
                                            socket.SOL_SOCKET,
                                            socket.SO_LINGER,
                                            struct.pack("ii", 1, 0),
                                        )
                                    except OSError:
                                        pass
                                    break
                                try:
                                    cli.sendall(data)
                                    sent += len(data)
                                except OSError:
                                    break
                            for s in (cli, backend):
                                try:
                                    s.close()
                                except OSError:
                                    pass

                        t1 = threading.Thread(target=_c2b, daemon=True)
                        t2 = threading.Thread(target=_b2c, daemon=True)
                        t1.start()
                        t2.start()
                        t2.join()
                        t1.join(timeout=1.0)
                    except OSError:
                        for s in (cli, backend):
                            if s is not None:
                                try:
                                    s.close()
                                except OSError:
                                    pass

                threading.Thread(target=_handle, args=(client,), daemon=True).start()

        self._thread = threading.Thread(target=_accept_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop_event.set()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
            self._listen_sock = None
        with self._lock:
            for sock in self._active_sockets:
                try:
                    sock.close()
                except OSError:
                    pass
            self._active_sockets.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None


# ---------------------------------------------------------------------------
# RTMP & SRT Loopback Servers via system ffmpeg (F6)
# ---------------------------------------------------------------------------


class FfmpegRtmpServer:
    """Single-session RTMP/RTMPS loopback server backed by /usr/bin/ffmpeg -listen 1."""

    def __init__(
        self,
        mode: Literal["receive", "serve"],
        *,
        source_path: Path | None = None,
        output_path: Path | None = None,
        app: str = "live",
        stream_key: str = "stream1",
        tls: bool = False,
        truncate_after_bytes: int | None = None,
        duration_sec: float = 1.0,
    ) -> None:
        if mode == "receive" and output_path is None:
            raise ValueError("output_path is required when mode='receive'")
        if tls and truncate_after_bytes is not None:
            raise ValueError("tls and truncate_after_bytes cannot both be set")
        self._mode = mode
        self._source_path = source_path
        self._output_path = output_path
        self._app = app
        self._stream_key = stream_key
        self._tls = tls
        self._truncate_after = truncate_after_bytes
        self._duration_sec = duration_sec
        self._backend_port: int = 0
        self.port: int = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._tunnel: TlsTcpTunnel | None = None
        self._proxy: RtmpTruncatingProxy | None = None

    @property
    def scheme(self) -> str:
        return "rtmps" if self._tls else "rtmp"

    @property
    def url(self) -> str:
        return f"{self.scheme}://127.0.0.1:{self.port}/{self._app}/{self._stream_key}"

    @property
    def ca_cert_path(self) -> Path | None:
        return self._tunnel.ca_cert_path if self._tunnel is not None else None

    def __enter__(self) -> FfmpegRtmpServer:
        ffmpeg = find_system_ffmpeg_bin()
        self._backend_port = find_free_port(socket.SOCK_STREAM)
        backend_url = f"rtmp://127.0.0.1:{self._backend_port}/{self._app}/{self._stream_key}"

        if self._mode == "receive":
            assert self._output_path is not None
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(ffmpeg),
                "-y",
                "-v",
                "error",
                "-listen",
                "1",
                "-f",
                "flv",
                "-i",
                backend_url,
                "-c",
                "copy",
                str(self._output_path),
            ]
        else:
            if self._source_path is not None:
                cmd = [
                    str(ffmpeg),
                    "-y",
                    "-v",
                    "error",
                    "-re",
                    "-i",
                    str(self._source_path),
                    "-c",
                    "copy",
                    "-f",
                    "flv",
                    "-listen",
                    "1",
                    backend_url,
                ]
            else:
                cmd = [
                    str(ffmpeg),
                    "-y",
                    "-v",
                    "error",
                    "-re",
                    "-f",
                    "lavfi",
                    "-i",
                    f"testsrc2=s=160x120:r=15:d={self._duration_sec}",
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=f=440:r=44100:d={self._duration_sec}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    "5",
                    "-keyint_min",
                    "5",
                    "-c:a",
                    "aac",
                    "-ar",
                    "44100",
                    "-shortest",
                    "-f",
                    "flv",
                    "-listen",
                    "1",
                    backend_url,
                ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            wait_proc_net_listening("tcp", self._backend_port, timeout=5.0, proc=self._proc)
            if self._tls:
                self._tunnel = TlsTcpTunnel(self._backend_port).__enter__()
                self.port = self._tunnel.port
            elif self._truncate_after is not None:
                self._proxy = RtmpTruncatingProxy(
                    self._backend_port, truncate_after_bytes=self._truncate_after
                ).__enter__()
                self.port = self._proxy.port
            else:
                self.port = self._backend_port
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def wait(self, timeout: float = 15.0) -> tuple[int, str]:
        if self._proc is None:
            raise RuntimeError("FfmpegRtmpServer not started")
        _, stderr_bytes = self._proc.communicate(timeout=timeout)
        return self._proc.returncode, stderr_bytes.decode("utf-8", errors="replace")

    def __exit__(self, *_: Any) -> None:
        if self._proxy is not None:
            self._proxy.__exit__(None, None, None)
            self._proxy = None
        if self._tunnel is not None:
            self._tunnel.__exit__(None, None, None)
            self._tunnel = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            if self._proc.stderr is not None:
                try:
                    self._proc.stderr.close()
                except OSError:
                    pass
            self._proc = None


def _build_srt_url(
    port: int,
    *,
    mode: str,
    latency_us: int = 50000,
    passphrase: str | None = None,
    pbkeylen: int | None = None,
    streamid: str | None = None,
    linger_sec: int | None = None,
) -> str:
    params: list[tuple[str, str]] = [("mode", mode), ("latency", str(latency_us))]
    if passphrase is not None:
        params.append(("passphrase", passphrase))
    if pbkeylen is not None and passphrase is not None:
        params.append(("pbkeylen", str(pbkeylen)))
    if streamid is not None:
        params.append(("streamid", streamid))
    if linger_sec is not None:
        params.append(("linger", str(linger_sec)))
    return f"srt://127.0.0.1:{port}?{urllib.parse.urlencode(params)}"


class FfmpegSrtEndpoint:
    """SRT listener endpoint backed by /usr/bin/ffmpeg."""

    def __init__(
        self,
        direction: Literal["receive", "serve"],
        *,
        container_format: str = "matroska",
        source_path: Path | None = None,
        output_path: Path | None = None,
        passphrase: str | None = None,
        pbkeylen: int = 16,
        latency_us: int = 50000,
        linger_sec: int = 2,
        kill_after_sec: float | None = None,
    ) -> None:
        if direction == "receive" and output_path is None:
            raise ValueError("output_path is required when direction='receive'")
        if direction == "serve" and source_path is None:
            raise ValueError("source_path is required when direction='serve'")
        self._direction = direction
        self._container_format = container_format
        self._source_path = source_path
        self._output_path = output_path
        self._passphrase = passphrase
        self._pbkeylen = pbkeylen
        self._latency_us = latency_us
        self._linger_sec = linger_sec
        self._kill_after_sec = kill_after_sec
        self.port: int = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._kill_timer: threading.Timer | None = None

    def caller_url(
        self,
        *,
        streamid: str | None = None,
        passphrase: str | None = ...,  # type: ignore[assignment]
        pbkeylen: int | None = ...,  # type: ignore[assignment]
    ) -> str:
        eff_pass = self._passphrase if passphrase is ... else passphrase
        eff_keylen = self._pbkeylen if pbkeylen is ... else pbkeylen
        return _build_srt_url(
            self.port,
            mode="caller",
            latency_us=self._latency_us,
            passphrase=eff_pass,
            pbkeylen=eff_keylen,
            streamid=streamid,
        )

    def __enter__(self) -> FfmpegSrtEndpoint:
        ffmpeg = find_system_ffmpeg_bin()
        self.port = find_free_port(socket.SOCK_DGRAM)
        listener_url = _build_srt_url(
            self.port,
            mode="listener",
            latency_us=self._latency_us,
            passphrase=self._passphrase,
            pbkeylen=self._pbkeylen,
            linger_sec=self._linger_sec if self._direction == "serve" else None,
        )

        if self._direction == "receive":
            assert self._output_path is not None
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(ffmpeg),
                "-y",
                "-v",
                "error",
                "-f",
                self._container_format,
                "-i",
                listener_url,
                "-c",
                "copy",
                str(self._output_path),
            ]
        else:
            assert self._source_path is not None
            cmd = [
                str(ffmpeg),
                "-y",
                "-v",
                "error",
                "-re",
                "-i",
                str(self._source_path),
                "-c",
                "copy",
                "-f",
                self._container_format,
                listener_url,
            ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            wait_proc_net_listening("udp", self.port, timeout=5.0, proc=self._proc)
            if self._kill_after_sec is not None:
                def _kill_proc() -> None:
                    if self._proc is not None and self._proc.poll() is None:
                        try:
                            self._proc.kill()
                        except OSError:
                            pass

                self._kill_timer = threading.Timer(self._kill_after_sec, _kill_proc)
                self._kill_timer.daemon = True
                self._kill_timer.start()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def wait(self, timeout: float = 15.0) -> tuple[int, str]:
        if self._proc is None:
            raise RuntimeError("FfmpegSrtEndpoint not started")
        _, stderr_bytes = self._proc.communicate(timeout=timeout)
        return self._proc.returncode, stderr_bytes.decode("utf-8", errors="replace")

    def __exit__(self, *_: Any) -> None:
        if self._kill_timer is not None:
            self._kill_timer.cancel()
            self._kill_timer = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            if self._proc.stderr is not None:
                try:
                    self._proc.stderr.close()
                except OSError:
                    pass
            self._proc = None


class FfmpegSrtCaller:
    """Helper to push an ffmpeg SRT caller stream to an external SRT listener."""

    def __init__(
        self,
        target_url: str,
        *,
        source_path: Path,
        container_format: str = "matroska",
    ) -> None:
        self._target_url = target_url
        self._source_path = source_path
        self._container_format = container_format
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> FfmpegSrtCaller:
        ffmpeg = find_system_ffmpeg_bin()
        cmd = [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-re",
            "-i",
            str(self._source_path),
            "-c",
            "copy",
            "-f",
            self._container_format,
            self._target_url,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return self

    def wait(self, timeout: float = 15.0) -> tuple[int, str]:
        if self._proc is None:
            raise RuntimeError("FfmpegSrtCaller not started")
        _, stderr_bytes = self._proc.communicate(timeout=timeout)
        return self._proc.returncode, stderr_bytes.decode("utf-8", errors="replace")

    def __exit__(self, *_: Any) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            if self._proc.stderr is not None:
                try:
                    self._proc.stderr.close()
                except OSError:
                    pass
            self._proc = None


# ---------------------------------------------------------------------------
# Pure-Python Fault-Injection RTMP & SRT Servers (F6 Tier 2)
# ---------------------------------------------------------------------------


class FaultyRtmpServer:
    """Minimal pure-Python TCP server injecting RTMP handshake faults."""

    def __init__(
        self,
        fault: Literal["corrupt_s0", "truncated_s1", "stall_handshake"],
        *,
        stall_sec: float = 30.0,
    ) -> None:
        self._fault = fault
        self._stall_sec = stall_sec
        self._stop_event = threading.Event()
        self._listen_sock: socket.socket | None = None
        self._active_sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def url(self, app: str = "live", stream_key: str = "fault") -> str:
        return f"rtmp://127.0.0.1:{self.port}/{app}/{stream_key}"

    def __enter__(self) -> FaultyRtmpServer:
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        lsock.settimeout(0.1)
        self._listen_sock = lsock
        self.port = int(lsock.getsockname()[1])

        def _serve() -> None:
            while not self._stop_event.is_set():
                try:
                    conn, _ = lsock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with self._lock:
                    self._active_sockets.append(conn)
                threading.Thread(
                    target=self._handle_client, args=(conn,), daemon=True
                ).start()

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()
        return self

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(1.0)
            try:
                # Read up to C0+C1 (1537 bytes) without blocking forever.
                received = 0
                while received < 1537 and not self._stop_event.is_set():
                    chunk = conn.recv(1537 - received)
                    if not chunk:
                        break
                    received += len(chunk)
            except (socket.timeout, OSError):
                pass

            if self._fault == "corrupt_s0":
                conn.sendall(b"\x06" + b"\x00" * 3072)
            elif self._fault == "truncated_s1":
                conn.sendall(b"\x03" + b"\x00" * 512)
            elif self._fault == "stall_handshake":
                self._stop_event.wait(self._stall_sec)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def __exit__(self, *_: Any) -> None:
        self._stop_event.set()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
            self._listen_sock = None
        with self._lock:
            for sock in self._active_sockets:
                try:
                    sock.close()
                except OSError:
                    pass
            self._active_sockets.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None


class FaultySrtServer:
    """Minimal pure-Python UDP server injecting SRT handshake stalls/corruption."""

    def __init__(self, fault: Literal["stall", "corrupt_handshake"]) -> None:
        self._fault = fault
        self._stop_event = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def caller_url(self, *, latency_us: int = 50000) -> str:
        return f"srt://127.0.0.1:{self.port}?mode=caller&latency={latency_us}"

    def __enter__(self) -> FaultySrtServer:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(0.1)
        self._sock = sock
        self.port = int(sock.getsockname()[1])

        def _serve() -> None:
            while not self._stop_event.is_set():
                try:
                    _data, addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if self._fault == "corrupt_handshake":
                    try:
                        sock.sendto(b"\x80\x00\x00\x00" + b"\xff" * 12, addr)
                    except OSError:
                        pass

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
