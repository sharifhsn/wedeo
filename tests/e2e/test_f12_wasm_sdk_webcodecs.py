"""E2E tests for F12: @wedeo/wasm TypeScript SDK & WebCodecs Hybrid."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from lib import fixtures, oracle
from lib.harness import e2e_test, find_node_bin, repo_root, run_cmd

REQUIRED_TS_SYMBOLS = (
    "Wedeo",
    "openMediaStream",
    "remuxStream",
    "createVideoDecoder",
    "inspect",
    "extractKeyframes",
    "extractThumbnail",
)


def _pkg_dir() -> Path:
    return repo_root() / "packages" / "wedeo-wasm"


def _wasm_artifact_path() -> Path:
    root = repo_root()
    candidates = (
        root / "web" / "wedeo.wasm",
        root / "packages" / "wedeo-wasm" / "wedeo.wasm",
        root / "packages" / "wedeo-wasm" / "dist" / "wedeo.wasm",
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[0]


def _run_sdk(args: list[str | Path], *, timeout: float = 120.0) -> dict[str, Any]:
    node = find_node_bin()
    runner = repo_root() / "tests" / "e2e" / "lib" / "wasm_runner.mjs"
    res = run_cmd([node, runner, "sdk-test", "--pkg", _pkg_dir(), *args], timeout=timeout)
    if res.returncode != 0:
        raise AssertionError(
            f"wasm_runner.mjs sdk-test failed (exit={res.returncode}):\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    return json.loads(res.stdout)


class TestF12WasmSdkWebcodecs(unittest.TestCase):
    """Tier 1 and Tier 2 E2E tests for F12 @wedeo/wasm TypeScript SDK & WebCodecs hybrid."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_f12_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)
        self.pkg_dir = _pkg_dir()
        self.wasm_path = _wasm_artifact_path()

    @e2e_test(feature="F12", tier=1, milestone="M5")
    def test_t1_package_manifest_dts_and_optional_tsc(self) -> None:
        pkg_json_path = self.pkg_dir / "package.json"
        self.assertTrue(
            pkg_json_path.is_file(), f"Missing @wedeo/wasm manifest: {pkg_json_path}"
        )
        pkg = json.loads(pkg_json_path.read_text(encoding="utf-8"))
        self.assertEqual(pkg.get("name"), "@wedeo/wasm")
        self.assertIn("types", pkg)
        types_path = (self.pkg_dir / pkg["types"]).resolve()
        self.assertTrue(types_path.is_file(), f"Missing types declaration: {types_path}")

        dts_files = list(self.pkg_dir.rglob("*.d.ts"))
        self.assertTrue(dts_files, f"No .d.ts files found under {self.pkg_dir}")
        dts_text = "\n".join(
            p.read_text(encoding="utf-8", errors="replace") for p in dts_files
        )
        for sym in REQUIRED_TS_SYMBOLS:
            self.assertRegex(
                dts_text,
                rf"\b{re.escape(sym)}\b",
                f"Symbol {sym!r} missing from .d.ts declarations",
            )

        tsc_bin = shutil.which("tsc")
        if tsc_bin is not None:
            tsconfig = self.pkg_dir / "tsconfig.json"
            self.assertTrue(tsconfig.is_file(), f"Missing tsconfig.json: {tsconfig}")
            tsc_res = run_cmd([tsc_bin, "--noEmit", "-p", tsconfig], cwd=self.pkg_dir)
            self.assertEqual(tsc_res.returncode, 0, tsc_res.stderr or tsc_res.stdout)

        exported = _run_sdk(["--scenario", "inspect-exports"])
        self.assertEqual(exported["package_name"], "@wedeo/wasm")
        for sym in ("Wedeo", "openMediaStream", "remuxStream", "createVideoDecoder"):
            self.assertIn(sym, exported["exported_keys"])

    @e2e_test(feature="F12", tier=1, milestone="M5")
    def test_t1_readable_stream_inspect_keyframes_thumbnail(self) -> None:
        gop_mp4 = fixtures.synth_multi_gop_h264_mp4(
            self.tmp / "gop_source.mp4",
            width=64,
            height=64,
            fps=30,
            duration=1.8,
            gop=10,
            with_audio=True,
        )
        thumb_rgba = self.tmp / "thumb.rgba"
        res = _run_sdk(
            [
                "--scenario",
                "inspect-stream",
                "--wasm",
                self.wasm_path,
                "--input",
                gop_mp4,
                "--chunk-size",
                "1024",
                "--timestamp-sec",
                "0.5",
                "--out",
                thumb_rgba,
            ]
        )
        tracks = res["inspect"].get("tracks", [])
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0]["codec"], "h264")
        keyframes = res["keyframes"]
        self.assertIsInstance(keyframes, list)
        self.assertGreaterEqual(len(keyframes), 3)
        self.assertIsNotNone(res["thumbnail"])
        self.assertGreater(res["thumbnail"]["width"], 0)
        self.assertGreater(res["thumbnail"]["height"], 0)
        self.assertGreater(res["thumbnail"]["rgbaBytes"], 0)
        raw = thumb_rgba.read_bytes()
        self.assertGreater(len(set(raw)), 1)

    @e2e_test(feature="F12", tier=1, milestone="M5")
    def test_t1_webcodecs_hybrid_config_probing_and_wasm_fallback(self) -> None:
        ivf = fixtures.vp9_testsrc_128_ivf()

        sup_res = _run_sdk(
            [
                "--scenario",
                "decode-hybrid",
                "--wasm",
                self.wasm_path,
                "--input",
                ivf,
                "--codec",
                "vp09.00.10.08",
                "--width",
                "128",
                "--height",
                "128",
                "--webcodecs-mode",
                "supported",
            ]
        )
        self.assertEqual(sup_res["backend"], "webcodecs")
        self.assertGreaterEqual(sup_res["webcodecs"]["isConfigSupportedCalls"], 1)
        self.assertGreaterEqual(sup_res["webcodecs"]["configureCalls"], 1)
        self.assertGreaterEqual(sup_res["webcodecs"]["decodeCalls"], 1)

        for mode in ("unsupported", "native-absent"):
            out_rgba = self.tmp / f"fallback_{mode}.rgba"
            fb_res = _run_sdk(
                [
                    "--scenario",
                    "decode-hybrid",
                    "--wasm",
                    self.wasm_path,
                    "--input",
                    ivf,
                    "--codec",
                    "vp09.00.10.08",
                    "--width",
                    "128",
                    "--height",
                    "128",
                    "--webcodecs-mode",
                    mode,
                    "--out",
                    out_rgba,
                ]
            )
            self.assertEqual(fb_res["backend"], "wasm")
            if mode == "unsupported":
                self.assertGreaterEqual(fb_res["webcodecs"]["isConfigSupportedCalls"], 1)
                self.assertEqual(fb_res["webcodecs"]["configureCalls"], 0)
            self.assertGreater(fb_res["frame_count"], 0)
            ref_rgba = oracle.decode_raw_video(ivf, pix_fmt="rgba")
            got_rgba = out_rgba.read_bytes()
            self.assertEqual(len(got_rgba), len(ref_rgba))
            psnr = oracle.compute_psnr(ref_rgba, got_rgba)
            self.assertGreaterEqual(psnr, 45.0)

    @e2e_test(feature="F12", tier=1, milestone="M5")
    def test_t1_remux_stream_readable_to_writable_oracle(self) -> None:
        src = fixtures.h264_aac_mp4()
        out_mkv = self.tmp / "sdk_remux.mkv"
        res = _run_sdk(
            [
                "--scenario",
                "remux-stream",
                "--wasm",
                self.wasm_path,
                "--input",
                src,
                "--format",
                "mkv",
                "--out",
                out_mkv,
                "--chunk-size",
                "1024",
            ]
        )
        self.assertGreater(res["bytes_written"], 0)
        self.assertTrue(out_mkv.is_file())
        o_probe = oracle.ffprobe_json(out_mkv)
        oracle.assert_ffprobe_stream(o_probe, 0, codec_name="h264", width=320, height=240)
        oracle.assert_ffprobe_stream(o_probe, 1, codec_name="aac", sample_rate=48000)
        oracle.oracle_ffmpeg_decode_check(out_mkv)

    @e2e_test(feature="F12", tier=1, milestone="M5")
    def test_t1_wedeo_class_multi_codec_and_disposal(self) -> None:
        for sample, expected_codec in (
            (fixtures.h264_aac_mp4(), "h264"),
            (fixtures.vp9_smpte_mkv(), "vp9"),
            (fixtures.vp9_red_ivf(), "vp9"),
            (fixtures.chord_48k_wav(), "pcm_s16le"),
        ):
            res = _run_sdk(
                [
                    "--scenario",
                    "wedeo-lifecycle",
                    "--wasm",
                    self.wasm_path,
                    "--input",
                    sample,
                ]
            )
            tracks = res["inspect"].get("tracks", [])
            self.assertTrue(tracks)
            self.assertEqual(tracks[0]["codec"], expected_codec)

    @e2e_test(feature="F12", tier=2, milestone="M5")
    def test_t2_webcodecs_reject_and_runtime_decode_error(self) -> None:
        ivf = fixtures.vp9_red_ivf()
        rej = _run_sdk(
            [
                "--scenario",
                "decode-hybrid",
                "--wasm",
                self.wasm_path,
                "--input",
                ivf,
                "--codec",
                "vp09.00.10.08",
                "--width",
                "64",
                "--height",
                "64",
                "--webcodecs-mode",
                "reject",
            ]
        )
        self.assertEqual(rej["backend"], "wasm")
        self.assertGreater(rej["frame_count"], 0)

        err_mode = _run_sdk(
            [
                "--scenario",
                "decode-hybrid",
                "--wasm",
                self.wasm_path,
                "--input",
                ivf,
                "--codec",
                "vp09.00.10.08",
                "--width",
                "64",
                "--height",
                "64",
                "--webcodecs-mode",
                "decode-error",
            ]
        )
        self.assertGreaterEqual(err_mode["webcodecs"]["decodeCalls"], 1)
        self.assertEqual(err_mode["backend"], "wasm")
        self.assertGreater(err_mode["frame_count"], 0)

    @e2e_test(feature="F12", tier=2, milestone="M5")
    def test_t2_readable_stream_abort_cancel_and_upstream_error(self) -> None:
        src = fixtures.vp9_smpte_mkv()
        for scenario in ("abort-stream", "errored-stream"):
            res = _run_sdk(
                [
                    "--scenario",
                    scenario,
                    "--wasm",
                    self.wasm_path,
                    "--input",
                    src,
                ]
            )
            self.assertTrue(res["rejected"], f"{scenario} should reject cleanly")
            self.assertTrue(res["error_message"].strip())

    @e2e_test(feature="F12", tier=2, milestone="M5")
    def test_t2_empty_and_corrupt_streams_sdk_errors(self) -> None:
        for scenario in ("empty-stream", "corrupt-stream"):
            res = _run_sdk(["--scenario", scenario, "--wasm", self.wasm_path])
            self.assertTrue(res["rejected"], f"{scenario} should reject cleanly")
            self.assertTrue(res["error_message"].strip())

    @e2e_test(feature="F12", tier=2, milestone="M5")
    def test_t2_prime_chunk_splits_and_out_of_range_thumbnail(self) -> None:
        mkv = fixtures.vp9_smpte_mkv()
        for ts in ("-1", "9999"):
            res = _run_sdk(
                [
                    "--scenario",
                    "inspect-stream",
                    "--wasm",
                    self.wasm_path,
                    "--input",
                    mkv,
                    "--chunk-size",
                    "13",
                    "--timestamp-sec",
                    ts,
                ]
            )
            self.assertEqual(res["inspect"]["tracks"][0]["codec"], "vp9")
            self.assertIsNotNone(res["thumbnail"])
            self.assertGreater(res["thumbnail"]["rgbaBytes"], 0)

    @e2e_test(feature="F12", tier=2, milestone="M5")
    def test_t2_bad_wasm_init_and_unsupported_remux_format(self) -> None:
        bad_init = _run_sdk(["--scenario", "bad-init"])
        self.assertTrue(bad_init["rejected"])
        self.assertTrue(bad_init["error_message"].strip())

        bad_fmt = _run_sdk(
            [
                "--scenario",
                "bad-remux-format",
                "--wasm",
                self.wasm_path,
                "--input",
                fixtures.h264_aac_mp4(),
            ]
        )
        self.assertTrue(bad_fmt["rejected"])
        self.assertTrue(bad_fmt["error_message"].strip())
