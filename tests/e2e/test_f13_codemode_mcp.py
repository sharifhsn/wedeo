"""E2E tests for F13: Code Mode MCP, Keyframes/Thumbnails & 100-pt Composition Rubric."""

from __future__ import annotations

import http.server
import json
import re
import select
import subprocess
import tempfile
import threading
import time
import tomllib
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib import fixtures, oracle
from lib.harness import (
    assert_cli_error,
    augmented_env,
    e2e_test,
    find_system_ffmpeg_bin,
    find_wedeo_bin,
    repo_root,
    run_wedeo,
)


def workspace_package_version() -> str:
    cargo_toml = repo_root() / "Cargo.toml"
    data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    return str(data["workspace"]["package"]["version"])


def estimate_tokens(text: str) -> int:
    """Conservative stdlib upper-bound token count (word/punctuation + char/4)."""
    regex_tokens = len(re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text))
    return max(len(text) // 4, regex_tokens)


class McpStdioSession:
    """Persistent newline-delimited JSON-RPC 2.0 client for `wedeo mcp`."""

    def __init__(
        self,
        testcase: unittest.TestCase,
        extra_args: list[str] | tuple[str, ...] = (),
        *,
        timeout: float = 30.0,
    ) -> None:
        self.timeout = timeout
        self._next_id = 1
        cmd = [str(find_wedeo_bin()), "mcp", *extra_args]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=augmented_env(),
        )
        testcase.addCleanup(self.close)

    def close(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        if self.proc.poll() is None:
            deadline = time.monotonic() + 2.0
            while self.proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            if self.proc.poll() is None:
                self.proc.terminate()
                term_deadline = time.monotonic() + 2.0
                while self.proc.poll() is None and time.monotonic() < term_deadline:
                    time.sleep(0.02)
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
        if self.proc.stdout is not None and not self.proc.stdout.closed:
            self.proc.stdout.close()
        if self.proc.stderr is not None and not self.proc.stderr.closed:
            self.proc.stderr.close()

    def _read_line(self, timeout: float | None = None) -> str:
        assert self.proc.stdout is not None
        wait_s = self.timeout if timeout is None else timeout
        ready, _, _ = select.select([self.proc.stdout], [], [], wait_s)
        if not ready:
            raise AssertionError(
                f"Timed out after {wait_s:.1f}s waiting for JSON-RPC response from wedeo mcp"
            )
        line = self.proc.stdout.readline()
        if line == "":
            stderr_tail = ""
            if self.proc.stderr is not None:
                stderr_tail = self.proc.stderr.read()
            raise AssertionError(
                f"wedeo mcp exited unexpectedly (rc={self.proc.poll()}): {stderr_tail}"
            )
        return line.rstrip("\r\n")

    def raw_line(self, line: str, *, timeout: float | None = None) -> dict[str, Any]:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return json.loads(self._read_line(timeout=timeout))

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        assert self.proc.stdin is not None
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        resp = self.raw_line(json.dumps(payload), timeout=timeout)
        if resp.get("id") != req_id:
            raise AssertionError(
                f"JSON-RPC id mismatch: expected {req_id}, got {resp!r}"
            )
        return resp

    def initialize(self) -> dict[str, Any]:
        resp = self.request("initialize", {})
        if "error" in resp:
            raise AssertionError(f"initialize failed: {resp}")
        self.notify("notifications/initialized", {})
        return resp["result"]

    def tools_list(self) -> list[dict[str, Any]]:
        resp = self.request("tools/list", {})
        if "error" in resp:
            raise AssertionError(f"tools/list failed: {resp}")
        return list(resp["result"]["tools"])

    def call_tool_raw(
        self, name: str, arguments: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        return self.request(
            "tools/call", {"name": name, "arguments": arguments}, timeout=timeout
        )

    def call_tool_text(
        self, name: str, arguments: dict[str, Any], *, timeout: float | None = None
    ) -> str:
        resp = self.call_tool_raw(name, arguments, timeout=timeout)
        if "error" in resp:
            raise AssertionError(f"tools/call {name} returned JSON-RPC error: {resp}")
        result = resp["result"]
        if result.get("isError", False):
            raise AssertionError(f"tools/call {name} returned isError=true: {result}")
        return str(result["content"][0]["text"])

    def call_tool_json(
        self, name: str, arguments: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        return json.loads(self.call_tool_text(name, arguments, timeout=timeout))


def verify_decoded_thumbnail(
    path: Path, expected_w: int, expected_h: int
) -> dict[str, Any]:
    """Decode PNG/PPM/WebP thumbnail via system ffmpeg (NOT oracle, which lacks png)."""
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Thumbnail missing or empty: {path}")
    ffmpeg = find_system_ffmpeg_bin()
    proc = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        timeout=30.0,
    )
    if proc.returncode != 0 or proc.stderr:
        raise AssertionError(
            f"System ffmpeg thumbnail decode failed on {path}: "
            f"{proc.stderr.decode(errors='replace')}"
        )
    raw = proc.stdout
    expected_bytes = expected_w * expected_h * 3
    if len(raw) != expected_bytes:
        raise AssertionError(
            f"Thumbnail {path} RGB24 byte count {len(raw)} != expected {expected_bytes} "
            f"({expected_w}x{expected_h})"
        )
    min_val = min(raw)
    max_val = max(raw)
    step = max(3, (len(raw) // 3 // 256) * 3)
    triples = {(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), step)}
    if max_val - min_val == 0 or len(triples) <= 1:
        raise AssertionError(
            f"Decoded thumbnail {path} is blank/uniform (range={min_val}..{max_val})"
        )
    return {"bytes": len(raw), "distinct_sampled_colors": len(triples)}


@dataclass(frozen=True)
class RubricScore:
    structural: int
    perceptual: int
    completeness: int
    efficiency: int
    safety: int
    total: int


def evaluate_composition_rubric(
    *,
    output_path: Path,
    thumb_paths: list[tuple[Path, int, int]],
    expected_video_codec: str,
    expected_width: int,
    expected_height: int,
    expected_sample_rate: int,
    expected_min_duration: float,
    expected_max_duration: float,
    stages_verified: int,
    total_stages: int,
    tool_turns: int,
    search_tokens: int,
    tools_list_tokens: int,
    sandbox_clean: bool,
    deterministic_repeat_ok: bool,
) -> RubricScore:
    """Explicit 100-pt Code Mode MCP composition rubric (PROJECT.md & Survey 3)."""
    structural = 0
    insp = oracle.assert_wedeo_inspect_clean(run_wedeo, output_path, verify=True)
    if insp["health"]["is_clean"] and insp["verified"]:
        structural += 10

    probe = oracle.ffprobe_json(output_path)
    streams = probe.get("streams", [])
    if (
        len(streams) >= 2
        and streams[0].get("codec_name") == expected_video_codec
        and streams[0].get("width") == expected_width
        and streams[0].get("height") == expected_height
        and str(streams[1].get("sample_rate")) == str(expected_sample_rate)
    ):
        structural += 10

    dur = float(probe["format"]["duration"])
    if expected_min_duration <= dur <= expected_max_duration:
        structural += 5

    if thumb_paths and all(
        verify_decoded_thumbnail(p, w, h)["distinct_sampled_colors"] > 1
        for p, w, h in thumb_paths
    ):
        structural += 10

    perceptual = 0
    oracle.oracle_ffmpeg_decode_check(output_path)
    perceptual += 10
    records = oracle.decode_framecrc(output_path)
    video_pts = [r.pts for r in records if r.stream_index == 0]
    if video_pts:
        perceptual += 10
    if video_pts == sorted(video_pts):
        perceptual += 5

    completeness = round(15 * min(stages_verified, total_stages) / max(1, total_stages))

    if tool_turns <= 2 and search_tokens < 1200 and tools_list_tokens < 2000:
        efficiency = 15
    elif tool_turns <= 4 and search_tokens < 1200:
        efficiency = 10
    else:
        efficiency = 5

    safety = (5 if sandbox_clean else 0) + (5 if deterministic_repeat_ok else 0)
    total = structural + perceptual + completeness + efficiency + safety
    return RubricScore(
        structural=structural,
        perceptual=perceptual,
        completeness=completeness,
        efficiency=efficiency,
        safety=safety,
        total=total,
    )


class _SilentLogHandler(http.server.BaseHTTPRequestHandler):
    hits: list[str] = []

    def do_GET(self) -> None:
        self.__class__.hits.append(self.path)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: object) -> None:
        return


class TestF13CodeModeMcp(unittest.TestCase):
    """Tier 1 and Tier 2 E2E tests for F13 Code Mode MCP."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_f13_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    @e2e_test(feature="F13", tier=1, milestone="M6")
    def test_t1_code_mode_default_vs_legacy_mode_compatibility(self) -> None:
        expected_ver = workspace_package_version()

        for args in ([], ["--mode", "code"]):
            sess = McpStdioSession(self, args)
            init = sess.initialize()
            self.assertEqual(init["serverInfo"]["name"], "wedeo")
            self.assertEqual(init["serverInfo"]["version"], expected_ver)
            tools = sess.tools_list()
            names = [t["name"] for t in tools]
            self.assertEqual(sorted(names), ["wedeo.execute", "wedeo.search"])

        legacy = McpStdioSession(self, ["--mode", "legacy"])
        legacy.initialize()
        legacy_tools = [t["name"] for t in legacy.tools_list()]
        self.assertEqual(
            legacy_tools, ["inspect", "explain", "transcode", "cut", "mux"]
        )
        insp = legacy.call_tool_json(
            "inspect", {"input": str(fixtures.vp9_red_ivf())}
        )
        self.assertEqual(insp["tracks"][0]["codec"], "vp9")

    @e2e_test(feature="F13", tier=1, milestone="M6")
    def test_t1_search_compact_dts_under_1200_tokens(self) -> None:
        sess = McpStdioSession(self, ["--mode", "code"])
        sess.initialize()
        tools_payload = json.dumps(sess.tools_list())
        self.assertLess(estimate_tokens(tools_payload), 2000)

        for args in (
            {},
            {"category": "keyframes"},
            {"query": "clip concat mux explain export"},
        ):
            text = sess.call_tool_text("wedeo.search", args)
            tokens = estimate_tokens(text)
            self.assertLess(
                tokens,
                1200,
                f"wedeo.search({args}) exceeded 1200 tokens ({tokens}):\n{text}",
            )

        full_slice = sess.call_tool_text("wedeo.search", {})
        for sym in (
            "inspect",
            "keyframes",
            "thumbnail",
            "thumbnails",
            "clip",
            "concat",
            "mux",
            "explain",
            "export",
        ):
            self.assertIn(sym, full_slice)

    @e2e_test(feature="F13", tier=1, milestone="M6")
    def test_t1_execute_keyframes_thumbnail_and_contact_sheet(self) -> None:
        src = fixtures.synth_multi_gop_h264_mp4(
            self.tmp / "multi_gop_src.mp4",
            width=64,
            height=64,
            fps=30,
            duration=1.8,
            gop=10,
            with_audio=False,
        )
        thumb_png = self.tmp / "thumb.png"
        thumb_ppm = self.tmp / "thumb.ppm"
        sheet_png = self.tmp / "sheet.png"

        cli_gop = run_wedeo(["inspect", "--level", "gop", "--json", src])
        self.assertEqual(cli_gop.returncode, 0, cli_gop.stderr)
        gop_pts = [
            g["keyframe_pts"]
            for g in json.loads(cli_gop.stdout)["gops"]
            if g["track_id"] == 0
        ]

        sess = McpStdioSession(self, ["--mode", "code", "--root", str(self.tmp)])
        sess.initialize()
        code = f"""
const kfs = wedeo.keyframes({json.dumps(str(src))});
const p1 = wedeo.thumbnail({json.dumps(str(src))}, {{ at_sec: 0.5, size: "64x64", output: {json.dumps(str(thumb_png))} }});
const p2 = wedeo.thumbnail({json.dumps(str(src))}, {{ at_sec: 0.5, size: "64x64", output: {json.dumps(str(thumb_ppm))} }});
const sheet = wedeo.thumbnails({json.dumps(str(src))}, {{ count: 3, cols: 3, size: "48x48", output: {json.dumps(str(sheet_png))} }});
return {{ kfs, p1, p2, sheet }};
"""
        res = sess.call_tool_json(
            "wedeo.execute", {"code": code, "allow_write": True}
        )
        kfs = res["kfs"] if isinstance(res["kfs"], list) else res["kfs"]["keyframes"]
        self.assertEqual(len(kfs), len(gop_pts))
        verify_decoded_thumbnail(thumb_png, 64, 64)
        verify_decoded_thumbnail(thumb_ppm, 64, 64)
        verify_decoded_thumbnail(sheet_png, 144, 48)

    @e2e_test(feature="F13", tier=1, milestone="M6")
    def test_t1_execute_fluent_clip_concat_mux_explain_export(self) -> None:
        src_v = fixtures.vp9_smpte_mkv()
        src_a = fixtures.synth_wav(
            self.tmp / "bgm_48k.wav", sample_rate=48000, channels=1, duration_sec=1.5
        )
        out_mkv = self.tmp / "composed.mkv"

        sess = McpStdioSession(self, ["--mode", "code", "--root", str(self.tmp)])
        sess.initialize()
        explain_code = f"""
const c1 = wedeo.clip({json.dumps(str(src_v))}).trim(0.0, 0.4).scale(160, 120);
const c2 = wedeo.clip({json.dumps(str(src_v))}).trim(0.5, 0.9).scale(160, 120);
const joined = wedeo.concat([c1, c2]);
const pipe = wedeo.mux({{ video: joined, audio: {json.dumps(str(src_a))} }});
return pipe.explain({{ output: {json.dumps(str(out_mkv))} }});
"""
        plan = sess.call_tool_json(
            "wedeo.execute", {"code": explain_code, "allow_write": False}
        )
        self.assertFalse(out_mkv.exists(), "explain() must not create output file")
        self.assertIn("nodes", json.dumps(plan))

        export_code = f"""
const c1 = wedeo.clip({json.dumps(str(src_v))}).trim(0.0, 0.4).scale(160, 120);
const c2 = wedeo.clip({json.dumps(str(src_v))}).trim(0.5, 0.9).scale(160, 120);
const joined = wedeo.concat([c1, c2]);
const pipe = wedeo.mux({{ video: joined, audio: {json.dumps(str(src_a))} }});
return pipe.export({json.dumps(str(out_mkv))});
"""
        sess.call_tool_json("wedeo.execute", {"code": export_code, "allow_write": True})
        self.assertTrue(out_mkv.is_file())
        oracle.assert_wedeo_inspect_clean(run_wedeo, out_mkv, verify=True)
        probe = oracle.ffprobe_json(out_mkv)
        oracle.assert_ffprobe_stream(probe, 0, codec_type="video", width=160, height=120)
        oracle.assert_ffprobe_stream(probe, 1, codec_type="audio", sample_rate=48000)
        oracle.oracle_ffmpeg_decode_check(out_mkv)

    @e2e_test(feature="F13", tier=1, milestone="M6")
    def test_t1_multistep_composition_100pt_rubric(self) -> None:
        src_v = fixtures.synth_multi_gop_h264_mp4(
            self.tmp / "rubric_src.mp4",
            width=128,
            height=96,
            fps=15,
            duration=1.8,
            gop=5,
            with_audio=False,
        )
        src_a = fixtures.synth_wav(
            self.tmp / "rubric_audio.wav", sample_rate=48000, channels=1, duration_sec=1.5
        )
        thumb_path = self.tmp / "rubric_thumb.png"
        sheet_path = self.tmp / "rubric_sheet.png"
        out_mkv = self.tmp / "rubric_final.mkv"

        sess = McpStdioSession(self, ["--mode", "code", "--root", str(self.tmp)])
        sess.initialize()
        tools_tokens = estimate_tokens(json.dumps(sess.tools_list()))

        # Turn 1: wedeo.search
        search_text = sess.call_tool_text(
            "wedeo.search", {"query": "keyframes thumbnails clip concat mux export"}
        )
        search_tokens = estimate_tokens(search_text)

        # Turn 2: single multi-step wedeo.execute composition
        code = f"""
const info = wedeo.inspect({json.dumps(str(src_v))});
const kfs = wedeo.keyframes({json.dumps(str(src_v))});
const thumb = wedeo.thumbnail({json.dumps(str(src_v))}, {{ at_sec: 0.3, size: "64x48", output: {json.dumps(str(thumb_path))} }});
const sheet = wedeo.thumbnails({json.dumps(str(src_v))}, {{ count: 2, cols: 2, size: "48x36", output: {json.dumps(str(sheet_path))} }});
const c1 = wedeo.clip({json.dumps(str(src_v))}).trim(0.0, 0.5).scale(96, 72);
const c2 = wedeo.clip({json.dumps(str(src_v))}).trim(0.6, 1.1).scale(96, 72);
const joined = wedeo.concat([c1, c2]);
const pipe = wedeo.mux({{ video: joined, audio: {json.dumps(str(src_a))}, preserve_all: true }});
const plan = pipe.explain({{ output: {json.dumps(str(out_mkv))} }});
const exported = pipe.export({json.dumps(str(out_mkv))});
return {{ info, kfs, thumb, sheet, plan, exported }};
"""
        sess.call_tool_json("wedeo.execute", {"code": code, "allow_write": True})

        score = evaluate_composition_rubric(
            output_path=out_mkv,
            thumb_paths=[(thumb_path, 64, 48), (sheet_path, 96, 36)],
            expected_video_codec=oracle.ffprobe_json(out_mkv)["streams"][0]["codec_name"],
            expected_width=96,
            expected_height=72,
            expected_sample_rate=48000,
            expected_min_duration=0.4,
            expected_max_duration=3.0,
            stages_verified=6,
            total_stages=6,
            tool_turns=2,
            search_tokens=search_tokens,
            tools_list_tokens=tools_tokens,
            sandbox_clean=True,
            deterministic_repeat_ok=True,
        )
        self.assertGreaterEqual(score.total, 85, f"Rubric score {score} < 85")
        self.assertGreaterEqual(score.structural, 30)
        self.assertEqual(score.safety, 10)

    @e2e_test(feature="F13", tier=1, milestone="M6")
    def test_t1_execute_root_allowlist_and_deterministic_repeat(self) -> None:
        src = fixtures.vp9_testsrc_128_ivf()
        out1 = self.tmp / "det_1.mkv"
        out2 = self.tmp / "det_2.mkv"

        sess = McpStdioSession(self, ["--mode", "code", "--root", str(self.tmp)])
        sess.initialize()
        for out_path in (out1, out2):
            code = f"""
return wedeo.clip({json.dumps(str(src))}).scale(64, 64).export({json.dumps(str(out_path))});
"""
            sess.call_tool_json("wedeo.execute", {"code": code, "allow_write": True})

        crc1 = [(r.pts, r.crc) for r in oracle.decode_framecrc(out1)]
        crc2 = [(r.pts, r.crc) for r in oracle.decode_framecrc(out2)]
        self.assertTrue(crc1)
        self.assertEqual(crc1, crc2)

    @e2e_test(feature="F13", tier=2, milestone="M6")
    def test_t2_sandbox_blocks_ambient_fs_read_and_write_escape(self) -> None:
        sandbox = self.tmp / "sandbox"
        sandbox.mkdir()
        outside_read = self.tmp / "secret_outside.txt"
        outside_read.write_text("TOP_SECRET_VALUE", encoding="utf-8")
        outside_write = self.tmp / "escaped.mkv"

        sess = McpStdioSession(self, ["--mode", "code", "--root", str(sandbox)])
        sess.initialize()

        escape_scripts = [
            f'return wedeo.inspect("/etc/passwd");',
            f'return wedeo.inspect({json.dumps(str(sandbox / ".." / "secret_outside.txt"))});',
            f'return wedeo.clip({json.dumps(str(fixtures.vp9_red_ivf()))}).export({json.dumps(str(outside_write))});',
            f'return wedeo.clip({json.dumps(str(fixtures.vp9_red_ivf()))}).export({json.dumps(str(sandbox / "denied.mkv"))});',
        ]
        for idx, code in enumerate(escape_scripts):
            allow_write = idx != 3
            resp = sess.call_tool_raw(
                "wedeo.execute", {"code": code, "allow_write": allow_write}
            )
            self.assertNotIn("error", resp, f"Expected tool error, got RPC error: {resp}")
            self.assertTrue(
                resp["result"].get("isError", False),
                f"Sandbox failed to block escape script #{idx}: {resp}",
            )

        self.assertFalse(outside_write.exists())
        self.assertFalse((sandbox / "denied.mkv").exists())
        self.assertEqual(sess.request("ping", {})["result"], {})

    @e2e_test(feature="F13", tier=2, milestone="M6")
    def test_t2_sandbox_blocks_env_process_and_network_access(self) -> None:
        _SilentLogHandler.hits = []
        server = http.server.HTTPServer(("127.0.0.1", 0), _SilentLogHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        sess = McpStdioSession(self, ["--mode", "code", "--root", str(self.tmp)])
        sess.initialize()

        probes = [
            'if (typeof process !== "undefined") throw new Error("process exposed"); return { ok: true };',
            'if (typeof require !== "undefined") throw new Error("require exposed"); return { ok: true };',
            f'if (typeof fetch !== "undefined") return fetch("http://127.0.0.1:{port}/leak"); return {{ blocked: true }};',
        ]
        for code in probes:
            resp = sess.call_tool_raw("wedeo.execute", {"code": code})
            self.assertNotIn("error", resp)
            if not resp["result"].get("isError", False):
                text = resp["result"]["content"][0]["text"]
                self.assertNotIn("exposed", text)

        self.assertEqual(_SilentLogHandler.hits, [], "Sandbox leaked outbound HTTP request!")
        self.assertEqual(sess.request("ping", {})["result"], {})

    @e2e_test(feature="F13", tier=2, milestone="M6")
    def test_t2_execute_timeout_infinite_loop_aborts_cleanly(self) -> None:
        sess = McpStdioSession(self, ["--mode", "code", "--root", str(self.tmp)])
        sess.initialize()
        self.assertEqual(
            sorted(t["name"] for t in sess.tools_list()),
            ["wedeo.execute", "wedeo.search"],
        )

        t0 = time.monotonic()
        resp = sess.call_tool_raw(
            "wedeo.execute",
            {"code": "while (true) {}", "timeout_ms": 250},
            timeout=10.0,
        )
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 5.0)
        self.assertNotIn("error", resp, f"Expected tool timeout error, got RPC error: {resp}")
        self.assertTrue(resp.get("result", {}).get("isError", False))

        bad_timeout = sess.call_tool_raw(
            "wedeo.execute", {"code": "return 1;", "timeout_ms": 0}
        )
        self.assertTrue(
            bad_timeout.get("result", {}).get("isError", False)
            or (
                "error" in bad_timeout
                and "Unknown tool" not in bad_timeout["error"].get("message", "")
            )
        )
        self.assertEqual(sess.request("ping", {})["result"], {})

    @e2e_test(feature="F13", tier=2, milestone="M6")
    def test_t2_execute_invalid_syntax_and_media_error_diagnostics(self) -> None:
        sess = McpStdioSession(self, ["--mode", "code", "--root", str(self.tmp)])
        sess.initialize()
        self.assertEqual(
            sorted(t["name"] for t in sess.tools_list()),
            ["wedeo.execute", "wedeo.search"],
        )

        bad_scripts = [
            "const x = (",
            f'return wedeo.inspect({json.dumps(str(self.tmp / "missing.mp4"))});',
            f'return wedeo.thumbnail({json.dumps(str(fixtures.vp9_red_ivf()))}, {{ at_sec: -1.0, output: {json.dumps(str(self.tmp / "neg.png"))} }});',
            f'return wedeo.thumbnail({json.dumps(str(fixtures.vp9_red_ivf()))}, {{ at_sec: 9999.0, output: {json.dumps(str(self.tmp / "far.png"))} }});',
            f'return wedeo.clip({json.dumps(str(fixtures.vp9_smpte_mkv()))}).scale(160, 120).explain({{ strict: true, output: {json.dumps(str(self.tmp / "strict.mkv"))} }});',
        ]
        for code in bad_scripts:
            resp = sess.call_tool_raw(
                "wedeo.execute", {"code": code, "allow_write": True}
            )
            self.assertNotIn("error", resp, f"Expected tool error, got RPC error: {resp}")
            self.assertTrue(
                resp.get("result", {}).get("isError", False),
                f"Expected error for invalid script: {code}",
            )
        self.assertEqual(sess.request("ping", {})["result"], {})

    @e2e_test(feature="F13", tier=2, milestone="M6")
    def test_t2_cli_invalid_mode_and_root_flags(self) -> None:
        for args in (
            ["mcp", "--mode", "bogus"],
            ["mcp", "--mode"],
            ["mcp", "--root"],
            ["mcp", "--bogus-flag"],
        ):
            res = run_wedeo(args, stdin_text="", timeout=10.0)
            assert_cli_error(res, "[mcp]")

    @e2e_test(feature="F13", tier=2, milestone="M6")
    def test_t2_malformed_jsonrpc_and_invalid_search_params(self) -> None:
        sess = McpStdioSession(self, ["--mode", "code"])
        sess.initialize()
        self.assertEqual(
            sorted(t["name"] for t in sess.tools_list()),
            ["wedeo.execute", "wedeo.search"],
        )

        self.assertEqual(sess.raw_line("{not-json")["error"]["code"], -32700)
        self.assertEqual(sess.raw_line('{"jsonrpc":"2.0","id":99}')["error"]["code"], -32600)
        self.assertEqual(sess.request("nonexistent/method", {})["error"]["code"], -32601)
        self.assertEqual(
            sess.call_tool_raw("unknown_tool", {})["error"]["code"], -32602
        )
        bad_search = sess.call_tool_raw("wedeo.search", {"category": "nonexistent_cat"})
        self.assertTrue(
            bad_search.get("result", {}).get("isError", False)
            or (
                bad_search.get("error", {}).get("code") == -32602
                and "Unknown tool" not in bad_search["error"].get("message", "")
            )
        )
        sess.notify("notifications/cancelled", {"requestId": 1})
        self.assertEqual(sess.request("ping", {})["result"], {})
