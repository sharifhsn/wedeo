"""Tier 3 pairwise combinatorial E2E tests for EM4 features (F11, F12, F13, F14, F2, F1)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from lib import fixtures, oracle
from lib.harness import e2e_test, find_wedeo_bin, repo_root, run_cmd, run_wedeo
from test_f11_wasm_streaming_size import _wasm_artifact_path, run_wasm_runner
from test_f12_wasm_sdk_webcodecs import _run_sdk
from test_f13_codemode_mcp import McpStdioSession, workspace_package_version


class TestTier3Em4Pairwise(unittest.TestCase):
    """Pairwise cross-feature E2E tests across F11..F14 and baseline F1/F2."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_t3_em4_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    @e2e_test(feature="F11", tier=3, milestone="M5", features=("F11", "F12", "F1"))
    def test_f11_f12_chunked_abi_vs_sdk_and_native_cli_parity(self) -> None:
        wasm_path = _wasm_artifact_path()
        gop_mp4 = fixtures.synth_multi_gop_h264_mp4(
            self.tmp / "t3_gop.mp4",
            width=64,
            height=64,
            fps=15,
            duration=1.2,
            gop=5,
            with_audio=True,
        )
        for sample in (gop_mp4, fixtures.vp9_smpte_mkv()):
            abi_res = run_wasm_runner(
                "stream-inspect",
                [
                    "--wasm",
                    wasm_path,
                    "--input",
                    sample,
                    "--name",
                    sample.name,
                    "--chunk-size",
                    "512",
                ],
            )
            sdk_res = _run_sdk(
                [
                    "--scenario",
                    "inspect-stream",
                    "--wasm",
                    wasm_path,
                    "--input",
                    sample,
                    "--chunk-size",
                    "512",
                ]
            )
            cli_res = run_wedeo(["inspect", "--json", sample])
            self.assertEqual(cli_res.returncode, 0, cli_res.stderr)
            cli_json = json.loads(cli_res.stdout)

            self.assertEqual(
                len(abi_res["inspect"]["tracks"]), len(cli_json["tracks"])
            )
            self.assertEqual(
                len(sdk_res["inspect"]["tracks"]), len(cli_json["tracks"])
            )
            self.assertEqual(
                abi_res["inspect"]["tracks"][0]["codec"],
                cli_json["tracks"][0]["codec"],
            )
            self.assertEqual(
                sdk_res["inspect"]["tracks"][0]["codec"],
                cli_json["tracks"][0]["codec"],
            )

    @e2e_test(feature="F12", tier=3, milestone="M5", features=("F11", "F12", "F2"))
    def test_f11_f12_f2_sdk_remux_stream_matches_cli_mux_oracle(self) -> None:
        wasm_path = _wasm_artifact_path()
        src = fixtures.h264_aac_mp4()
        sdk_mkv = self.tmp / "sdk_remux.mkv"
        cli_mkv = self.tmp / "cli_mux.mkv"

        _run_sdk(
            [
                "--scenario",
                "remux-stream",
                "--wasm",
                wasm_path,
                "--input",
                src,
                "--format",
                "mkv",
                "--out",
                sdk_mkv,
                "--chunk-size",
                "1024",
            ]
        )
        mux_res = run_wedeo(["mux", "--video", src, "-y", "-o", cli_mkv])
        self.assertEqual(mux_res.returncode, 0, mux_res.stderr)

        oracle.assert_wedeo_inspect_clean(run_wedeo, sdk_mkv, verify=True)
        oracle.oracle_ffmpeg_decode_check(sdk_mkv)
        sdk_probe = oracle.ffprobe_json(sdk_mkv)
        cli_probe = oracle.ffprobe_json(cli_mkv)
        self.assertEqual(
            sdk_probe["streams"][0]["codec_name"],
            cli_probe["streams"][0]["codec_name"],
        )
        self.assertEqual(
            sdk_probe["streams"][0]["width"],
            cli_probe["streams"][0]["width"],
        )
        self.assertEqual(
            sdk_probe["streams"][0]["height"],
            cli_probe["streams"][0]["height"],
        )

    @e2e_test(feature="F13", tier=3, milestone="M6", features=("F13", "F2"))
    def test_f13_f2_codemode_execute_matches_cli_cut_mux_oracle(self) -> None:
        src_v = fixtures.vp9_smpte_mkv()
        src_a = fixtures.synth_wav(
            self.tmp / "t3_audio.wav", sample_rate=48000, channels=1, duration_sec=1.0
        )
        mcp_out = self.tmp / "mcp_out.mkv"
        cli_cut = self.tmp / "cli_cut.mkv"
        cli_tr = self.tmp / "cli_tr.mkv"
        cli_out = self.tmp / "cli_out.mkv"

        sess = McpStdioSession(self, ["--mode", "code", "--root", str(self.tmp)])
        sess.initialize()
        code = f"""
const c = wedeo.clip({json.dumps(str(src_v))}).trim(0.0, 0.5).scale(160, 120);
return wedeo.mux({{ video: c, audio: {json.dumps(str(src_a))} }}).export({json.dumps(str(mcp_out))});
"""
        sess.call_tool_json("wedeo.execute", {"code": code, "allow_write": True})

        self.assertEqual(
            run_wedeo(
                [
                    "cut",
                    "-i",
                    src_v,
                    "--from",
                    "0",
                    "--to",
                    "0.5",
                    "--mode",
                    "keyframe",
                    "-y",
                    "-o",
                    cli_cut,
                ]
            ).returncode,
            0,
        )
        self.assertEqual(
            run_wedeo(
                [
                    "transcode",
                    "-i",
                    cli_cut,
                    "--video",
                    "av1",
                    "--size",
                    "160x120",
                    "-y",
                    "-o",
                    cli_tr,
                ]
            ).returncode,
            0,
        )
        self.assertEqual(
            run_wedeo(
                [
                    "mux",
                    "--video",
                    cli_tr,
                    "--audio",
                    src_a,
                    "-y",
                    "-o",
                    cli_out,
                ]
            ).returncode,
            0,
        )

        oracle.assert_wedeo_inspect_clean(run_wedeo, mcp_out, verify=True)
        mcp_probe = oracle.ffprobe_json(mcp_out)
        cli_probe = oracle.ffprobe_json(cli_out)
        self.assertEqual(len(mcp_probe["streams"]), len(cli_probe["streams"]))
        self.assertEqual(mcp_probe["streams"][0]["width"], 160)
        self.assertEqual(mcp_probe["streams"][0]["height"], 120)
        self.assertEqual(str(mcp_probe["streams"][1]["sample_rate"]), "48000")

    @e2e_test(feature="F14", tier=3, milestone="M6", features=("F13", "F14"))
    def test_f13_f14_mcp_version_and_sdk_docs_match_release_metadata(self) -> None:
        root = repo_root()
        ver = workspace_package_version()

        cli_ver = run_wedeo(["--version"])
        self.assertEqual(cli_ver.returncode, 0, cli_ver.stderr)
        self.assertEqual(cli_ver.stdout.strip(), f"wedeo {ver}")

        sess = McpStdioSession(self, ["--mode", "code"])
        init = sess.initialize()
        self.assertEqual(init["serverInfo"]["version"], ver)
        search_text = sess.call_tool_text("wedeo.search", {})
        for op in ("inspect", "keyframes", "thumbnail", "clip", "concat", "mux"):
            self.assertIn(op, search_text)

        changelog = root / "CHANGELOG.md"
        formula = root / "Formula" / "wedeo.rb"
        self.assertTrue(changelog.is_file(), f"Missing {changelog}")
        self.assertTrue(formula.is_file(), f"Missing {formula}")
        self.assertIn(ver, changelog.read_text(encoding="utf-8"))
        self.assertIn(ver, formula.read_text(encoding="utf-8"))

    @e2e_test(feature="F14", tier=3, milestone="M6", features=("F14", "F2"))
    def test_f14_f2_staged_binary_and_shims_run_transcode_mux_oracle(self) -> None:
        root = repo_root()
        formula = root / "Formula" / "wedeo.rb"
        self.assertTrue(formula.is_file(), f"Missing {formula}")

        stage_bin = self.tmp / "stage" / "bin"
        stage_bin.mkdir(parents=True)
        staged_wedeo = stage_bin / "wedeo"
        shutil.copy2(find_wedeo_bin(), staged_wedeo)
        staged_wedeo.chmod(0o755)
        self.assertEqual(
            run_cmd([staged_wedeo, "install-shim", stage_bin]).returncode, 0
        )
        digest = hashlib.sha256(staged_wedeo.read_bytes()).hexdigest()
        sums_file = self.tmp / "stage" / "SHA256SUMS.txt"
        sums_file.write_text(f"{digest}  bin/wedeo\n", encoding="utf-8")
        self.assertIn(digest, sums_file.read_text(encoding="utf-8"))

        audio = fixtures.synth_wav(
            self.tmp / "t3_stage_audio.wav", sample_rate=48000, channels=1, duration_sec=0.5
        )
        tr_vid = self.tmp / "stage_av1.mkv"
        out_mux = self.tmp / "stage_mux.mkv"

        env = os.environ.copy()
        env["PATH"] = f"{stage_bin}{os.pathsep}{env.get('PATH', '')}"

        r_tr = run_cmd(
            [
                "wedeo",
                "transcode",
                "-i",
                fixtures.vp9_red_ivf(),
                "--video",
                "av1",
                "-y",
                "-o",
                tr_vid,
            ],
            env=env,
        )
        self.assertEqual(r_tr.returncode, 0, r_tr.stderr)

        r_ff = run_cmd(
            [
                "ffmpeg",
                "-i",
                tr_vid,
                "-i",
                audio,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-y",
                out_mux,
            ],
            env=env,
        )
        self.assertEqual(r_ff.returncode, 0, r_ff.stderr)
        oracle.assert_wedeo_inspect_clean(run_wedeo, out_mux, verify=True)
        probe = oracle.ffprobe_json(out_mux)
        oracle.assert_ffprobe_stream(probe, 0, codec_name="av1", width=64, height=64)
        oracle.assert_ffprobe_stream(probe, 1, codec_name="pcm_s16le", sample_rate=48000)
        oracle.oracle_ffmpeg_decode_check(out_mux)
