"""E2E tests for F11: WASM Chunked Streaming, simd128 & < 985 KiB Size Budget."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from lib import fixtures, oracle
from lib.harness import e2e_test, find_node_bin, repo_root, run_cmd, run_wedeo

WASM_SIZE_LIMIT_BYTES = 985 * 1024  # 1,008,640 bytes
REQUIRED_WASM_EXPORTS = (
    "wedeo_alloc",
    "wedeo_free",
    "wedeo_json_ptr",
    "wedeo_json_len",
    "wedeo_bytes_ptr",
    "wedeo_bytes_len",
    "wedeo_error_ptr",
    "wedeo_error_len",
    "wedeo_stream_open",
    "wedeo_stream_push",
    "wedeo_stream_inspect",
    "wedeo_stream_next_frame",
    "wedeo_stream_next_remux_chunk",
    "wedeo_stream_close",
)


def _wasm_artifact_path() -> Path:
    root = repo_root()
    candidates = (
        root / "web" / "wedeo.wasm",
        root
        / "bins"
        / "wedeo-wasm"
        / "target"
        / "wasm32-unknown-unknown"
        / "wasm-release"
        / "wedeo_wasm.wasm",
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[0]


def _wasm_runner_path() -> Path:
    return repo_root() / "tests" / "e2e" / "lib" / "wasm_runner.mjs"


def run_wasm_runner(
    command: str, args: list[str | Path], *, timeout: float = 120.0
) -> dict[str, Any]:
    node = find_node_bin()
    res = run_cmd([node, _wasm_runner_path(), command, *args], timeout=timeout)
    if res.returncode != 0:
        raise AssertionError(
            f"wasm_runner.mjs {command} failed (exit={res.returncode}):\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    return json.loads(res.stdout)


class TestF11WasmStreamingSize(unittest.TestCase):
    """Tier 1 and Tier 2 E2E tests for F11 WASM streaming ABI, simd128, and size budget."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_f11_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)
        self.wasm_path = _wasm_artifact_path()

    @e2e_test(feature="F11", tier=1, milestone="M5")
    def test_t1_wasm_size_budget_and_zero_host_imports(self) -> None:
        self.assertTrue(
            self.wasm_path.is_file(), f"Missing WASM artifact: {self.wasm_path}"
        )
        probe = run_wasm_runner("probe-wasm", ["--wasm", self.wasm_path])
        self.assertEqual(probe["imports"], [])
        self.assertLess(
            probe["gzip_default_bytes"],
            WASM_SIZE_LIMIT_BYTES,
            f"Gzipped WASM {probe['gzip_default_bytes']} B must be < {WASM_SIZE_LIMIT_BYTES} B (985 KiB)",
        )
        self.assertLess(probe["gzip_max_bytes"], WASM_SIZE_LIMIT_BYTES)

    @e2e_test(feature="F11", tier=1, milestone="M5")
    def test_t1_wasm_streaming_abi_exports_and_simd128(self) -> None:
        probe = run_wasm_runner("probe-wasm", ["--wasm", self.wasm_path])
        export_names = {exp["name"] for exp in probe["exports"]}
        missing = [name for name in REQUIRED_WASM_EXPORTS if name not in export_names]
        self.assertEqual(
            missing, [], f"Missing required WASM streaming exports: {missing}"
        )
        self.assertEqual(probe["desynced_functions"], 0)
        self.assertTrue(
            probe["has_simd128"],
            f"Expected wasm32 simd128 instructions or target_features in {self.wasm_path}",
        )
        self.assertGreater(probe["simd128_op_count"], 0)

    @e2e_test(feature="F11", tier=1, milestone="M5")
    def test_t1_stream_inspect_chunked_containers(self) -> None:
        cases = (
            fixtures.h264_aac_mp4(),
            fixtures.vp9_smpte_mkv(),
            fixtures.vp9_testsrc_128_ivf(),
            fixtures.chord_48k_wav(),
        )
        for sample in cases:
            wasm_res = run_wasm_runner(
                "stream-inspect",
                [
                    "--wasm",
                    self.wasm_path,
                    "--input",
                    sample,
                    "--name",
                    sample.name,
                    "--chunk-size",
                    "512",
                ],
            )
            cli_res = run_wedeo(["inspect", "--json", sample])
            self.assertEqual(cli_res.returncode, 0, cli_res.stderr)
            cli_json = json.loads(cli_res.stdout)
            o_probe = oracle.ffprobe_json(sample)

            insp = wasm_res["inspect"]
            self.assertGreater(wasm_res["chunks_pushed"], 1)
            tracks = insp.get("tracks", [])
            self.assertEqual(len(tracks), len(cli_json["tracks"]))
            self.assertEqual(len(tracks), len(o_probe["streams"]))
            self.assertEqual(tracks[0]["codec"], cli_json["tracks"][0]["codec"])

    @e2e_test(feature="F11", tier=1, milestone="M5")
    def test_t1_stream_decode_vp9_h264_simd128_oracle_parity(self) -> None:
        h264_mp4 = fixtures.synth_multi_gop_h264_mp4(
            self.tmp / "h264_video_only.mp4",
            width=64,
            height=64,
            fps=15,
            duration=0.6,
            gop=5,
            with_audio=False,
        )
        for idx, sample in enumerate((fixtures.vp9_testsrc_128_ivf(), h264_mp4)):
            out_rgba = self.tmp / f"decoded_{idx}.rgba"
            res = run_wasm_runner(
                "stream-decode-video",
                [
                    "--wasm",
                    self.wasm_path,
                    "--input",
                    sample,
                    "--name",
                    sample.name,
                    "--out-rgba",
                    out_rgba,
                    "--chunk-size",
                    "1024",
                ],
            )
            self.assertGreater(len(res["frames"]), 0)
            got_rgba = out_rgba.read_bytes()
            ref_rgba = oracle.decode_raw_video(sample, pix_fmt="rgba")
            self.assertEqual(len(got_rgba), len(ref_rgba))
            psnr = oracle.compute_psnr(ref_rgba, got_rgba)
            self.assertGreaterEqual(
                psnr, 45.0, f"simd128 decode PSNR too low for {sample.name}: {psnr:.2f}"
            )

    @e2e_test(feature="F11", tier=1, milestone="M5")
    def test_t1_stream_remux_chunked_output_oracle(self) -> None:
        remux_cases = (
            (fixtures.h264_aac_mp4(), "mkv", self.tmp / "out_h264.mkv"),
            (fixtures.vp9_smpte_mkv(), "mp4", self.tmp / "out_vp9.mp4"),
        )
        for src, fmt, out_path in remux_cases:
            res = run_wasm_runner(
                "stream-remux",
                [
                    "--wasm",
                    self.wasm_path,
                    "--input",
                    src,
                    "--name",
                    src.name,
                    "--format",
                    fmt,
                    "--out",
                    out_path,
                    "--chunk-size",
                    "2048",
                ],
            )
            self.assertGreater(res["bytes_written"], 0)
            self.assertGreater(res["remux_chunks"], 0)
            self.assertTrue(out_path.is_file())
            src_probe = oracle.ffprobe_json(src)
            out_probe = oracle.ffprobe_json(out_path)
            self.assertEqual(len(out_probe["streams"]), len(src_probe["streams"]))
            self.assertEqual(
                out_probe["streams"][0]["codec_name"],
                src_probe["streams"][0]["codec_name"],
            )
            oracle.oracle_ffmpeg_decode_check(out_path)

    @e2e_test(feature="F11", tier=2, milestone="M5")
    def test_t2_stream_boundary_one_byte_and_single_chunk(self) -> None:
        ivf = fixtures.vp9_red_ivf()
        wav = fixtures.chord_48k_wav()

        for sample in (ivf, wav):
            one_byte = run_wasm_runner(
                "stream-inspect",
                [
                    "--wasm",
                    self.wasm_path,
                    "--input",
                    sample,
                    "--name",
                    sample.name,
                    "--chunk-size",
                    "1",
                ],
            )
            single = run_wasm_runner(
                "stream-inspect",
                [
                    "--wasm",
                    self.wasm_path,
                    "--input",
                    sample,
                    "--name",
                    sample.name,
                    "--chunk-size",
                    str(sample.stat().st_size + 64),
                ],
            )
            self.assertEqual(one_byte["chunks_pushed"], sample.stat().st_size)
            self.assertEqual(single["chunks_pushed"], 1)
            self.assertEqual(
                one_byte["inspect"]["tracks"], single["inspect"]["tracks"]
            )

    @e2e_test(feature="F11", tier=2, milestone="M5")
    def test_t2_stream_corrupt_truncated_and_empty_inputs(self) -> None:
        corrupt = self.tmp / "corrupt.mp4"
        corrupt.write_bytes(b"\xde\xad\xbe\xef_not_a_container_" * 16)
        truncated = self.tmp / "truncated.mp4"
        truncated.write_bytes(fixtures.h264_aac_mp4().read_bytes()[:24])
        empty = self.tmp / "empty.mp4"
        empty.write_bytes(b"")

        for neg_case, path in (
            ("corrupt", corrupt),
            ("truncated", truncated),
            ("empty", empty),
        ):
            res = run_wasm_runner(
                "stream-negative",
                ["--wasm", self.wasm_path, "--case", neg_case, "--input", path],
            )
            self.assertFalse(res["trapped"], f"WASM trapped on {neg_case}: {res}")
            self.assertLess(res["rc"], 0)
            self.assertTrue(res["error_message"].strip())

    @e2e_test(feature="F11", tier=2, milestone="M5")
    def test_t2_stream_invalid_handle_use_after_close_null_ptr(self) -> None:
        ivf = fixtures.vp9_red_ivf()
        for neg_case in ("invalid-handle", "use-after-close", "null-ptr"):
            res = run_wasm_runner(
                "stream-negative",
                ["--wasm", self.wasm_path, "--case", neg_case, "--input", ivf],
            )
            self.assertFalse(res["trapped"], f"WASM trapped on {neg_case}: {res}")
            self.assertLess(res["rc"], 0)
            self.assertTrue(res["error_message"].strip())
            self.assertTrue(res["post_error_session_ok"])

    @e2e_test(feature="F11", tier=2, milestone="M5")
    def test_t2_stream_bad_remux_format_and_audio_only_video_pull(self) -> None:
        wav = fixtures.chord_48k_wav()
        for neg_case in ("bad-format", "audio-next-frame"):
            res = run_wasm_runner(
                "stream-negative",
                ["--wasm", self.wasm_path, "--case", neg_case, "--input", wav],
            )
            self.assertFalse(res["trapped"], f"WASM trapped on {neg_case}: {res}")
            self.assertLess(res["rc"], 0)
            self.assertTrue(res["error_message"].strip())

    @e2e_test(feature="F11", tier=2, milestone="M5")
    def test_t2_wasm_repeated_session_memory_stability(self) -> None:
        res = run_wasm_runner(
            "stream-negative",
            [
                "--wasm",
                self.wasm_path,
                "--case",
                "leak-loop",
                "--input",
                fixtures.vp9_red_ivf(),
            ],
        )
        self.assertFalse(res["trapped"], f"WASM trapped in leak-loop: {res}")
        history = res["memory_bytes_history"]
        self.assertEqual(len(history), 20)
        # Allow warmup on first 3 iterations; memory must stabilize afterward.
        self.assertEqual(
            history[-1],
            history[3],
            f"WASM linear memory grew across repeated open/decode/close cycles: {history}",
        )
