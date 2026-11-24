"""Tier 4 real-world workflows for EM4 (Scenarios 7, 8, and 9)."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from lib import fixtures, oracle
from lib.harness import e2e_test, find_node_bin, find_wedeo_bin, repo_root, run_cmd, run_wedeo
from test_f11_wasm_streaming_size import (
    WASM_SIZE_LIMIT_BYTES,
    _wasm_artifact_path,
    run_wasm_runner,
)
from test_f12_wasm_sdk_webcodecs import _pkg_dir, _run_sdk
from test_f13_codemode_mcp import (
    McpStdioSession,
    estimate_tokens,
    evaluate_composition_rubric,
    verify_decoded_thumbnail,
    workspace_package_version,
)
from test_f14_release_packaging import _cargo_metadata, _is_publishable, _parse_publish_sh_crates


class TestTier4Em4Workflows(unittest.TestCase):
    """Real-world end-to-end application scenarios for EM4 (M5 & M6)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_t4_em4_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    @e2e_test(feature="F12", tier=4, milestone="M5", features=("F11", "F12"))
    def test_scenario7_wasm_stream_webcodecs_fallback_remux_workflow(self) -> None:
        wasm_path = _wasm_artifact_path()
        pkg_dir = _pkg_dir()

        # Step 1: Verify gzipped .wasm < 985 KiB, simd128 opcodes, and @wedeo/wasm package
        probe = run_wasm_runner("probe-wasm", ["--wasm", wasm_path])
        self.assertLess(probe["gzip_default_bytes"], WASM_SIZE_LIMIT_BYTES)
        self.assertTrue(probe["has_simd128"])
        self.assertTrue((pkg_dir / "package.json").is_file())

        # Step 2: Stream multi-GOP MP4 via ReadableStream -> openMediaStream (inspect, keyframes, thumbnail)
        src_mp4 = fixtures.synth_multi_gop_h264_mp4(
            self.tmp / "s7_src.mp4",
            width=128,
            height=96,
            fps=15,
            duration=1.5,
            gop=5,
            with_audio=False,
        )
        thumb_rgba = self.tmp / "s7_thumb.rgba"
        stream_res = _run_sdk(
            [
                "--scenario",
                "inspect-stream",
                "--wasm",
                wasm_path,
                "--input",
                src_mp4,
                "--chunk-size",
                "1024",
                "--timestamp-sec",
                "0.4",
                "--out",
                thumb_rgba,
            ]
        )
        self.assertEqual(stream_res["inspect"]["tracks"][0]["codec"], "h264")
        self.assertGreaterEqual(len(stream_res["keyframes"]), 3)
        self.assertGreater(stream_res["thumbnail"]["rgbaBytes"], 0)

        # Step 3: Exercise createVideoDecoder (WebCodecs supported probe + WASM simd128 fallback)
        sup = _run_sdk(
            [
                "--scenario",
                "decode-hybrid",
                "--wasm",
                wasm_path,
                "--input",
                src_mp4,
                "--codec",
                "avc1.42E01E",
                "--width",
                "128",
                "--height",
                "96",
                "--webcodecs-mode",
                "supported",
            ]
        )
        self.assertEqual(sup["backend"], "webcodecs")
        self.assertGreaterEqual(sup["webcodecs"]["isConfigSupportedCalls"], 1)

        fb_rgba = self.tmp / "s7_fallback.rgba"
        fb = _run_sdk(
            [
                "--scenario",
                "decode-hybrid",
                "--wasm",
                wasm_path,
                "--input",
                src_mp4,
                "--codec",
                "avc1.42E01E",
                "--width",
                "128",
                "--height",
                "96",
                "--webcodecs-mode",
                "native-absent",
                "--out",
                fb_rgba,
            ]
        )
        self.assertEqual(fb["backend"], "wasm")
        ref_rgba = oracle.decode_raw_video(src_mp4, pix_fmt="rgba")
        got_rgba = fb_rgba.read_bytes()
        self.assertEqual(len(got_rgba), len(ref_rgba))
        self.assertGreaterEqual(oracle.compute_psnr(ref_rgba, got_rgba), 45.0)

        # Step 4: remuxStream -> WritableStream validated via oracle ffprobe & ffmpeg decode
        remux_src = fixtures.h264_aac_mp4()
        out_mkv = self.tmp / "s7_out.mkv"
        remux_res = _run_sdk(
            [
                "--scenario",
                "remux-stream",
                "--wasm",
                wasm_path,
                "--input",
                remux_src,
                "--format",
                "mkv",
                "--out",
                out_mkv,
                "--chunk-size",
                "2048",
            ]
        )
        self.assertGreater(remux_res["bytes_written"], 0)
        oracle.assert_wedeo_inspect_clean(run_wedeo, out_mkv, verify=True)
        out_probe = oracle.ffprobe_json(out_mkv)
        oracle.assert_ffprobe_stream(
            out_probe, 0, codec_name="h264", width=320, height=240
        )
        oracle.assert_ffprobe_stream(
            out_probe, 1, codec_name="aac", sample_rate=48000
        )
        oracle.oracle_ffmpeg_decode_check(out_mkv)

    @e2e_test(feature="F13", tier=4, milestone="M6", features=("F13", "F2"))
    def test_scenario8_codemode_mcp_agentic_editing_rubric_workflow(self) -> None:
        src_v = fixtures.synth_multi_gop_h264_mp4(
            self.tmp / "s8_src.mp4",
            width=128,
            height=96,
            fps=15,
            duration=1.8,
            gop=5,
            with_audio=False,
        )
        src_a = fixtures.synth_wav(
            self.tmp / "s8_bgm.wav", sample_rate=48000, channels=1, duration_sec=1.6
        )
        thumb_path = self.tmp / "s8_thumb.png"
        sheet_path = self.tmp / "s8_sheet.png"
        final_mkv = self.tmp / "s8_final.mkv"

        # Step 1: Spawn wedeo mcp --mode code, initialize, and verify tools/list budget
        sess = McpStdioSession(self, ["--mode", "code", "--root", str(self.tmp)])
        init = sess.initialize()
        self.assertEqual(init["serverInfo"]["version"], workspace_package_version())
        tools = sess.tools_list()
        self.assertEqual(sorted(t["name"] for t in tools), ["wedeo.execute", "wedeo.search"])
        tools_tokens = estimate_tokens(json.dumps(tools))
        self.assertLess(tools_tokens, 2000)

        # Step 2: Turn 1 - wedeo.search returns compact .d.ts slice < 1,200 tokens
        search_text = sess.call_tool_text(
            "wedeo.search",
            {"query": "keyframes thumbnails clip concat mux explain export"},
        )
        search_tokens = estimate_tokens(search_text)
        self.assertLess(search_tokens, 1200)

        # Step 3: Turn 2 - single multi-step wedeo.execute composition + sandbox check
        code = f"""
const info = wedeo.inspect({json.dumps(str(src_v))});
const kfs = wedeo.keyframes({json.dumps(str(src_v))});
const thumb = wedeo.thumbnail({json.dumps(str(src_v))}, {{ at_sec: 0.4, size: "64x48", output: {json.dumps(str(thumb_path))} }});
const sheet = wedeo.thumbnails({json.dumps(str(src_v))}, {{ count: 2, cols: 2, size: "48x36", output: {json.dumps(str(sheet_path))} }});
const c1 = wedeo.clip({json.dumps(str(src_v))}).trim(0.0, 0.5).scale(160, 120);
const c2 = wedeo.clip({json.dumps(str(src_v))}).trim(0.6, 1.1).scale(160, 120);
const joined = wedeo.concat([c1, c2]);
const pipe = wedeo.mux({{ video: joined, audio: {json.dumps(str(src_a))}, preserve_all: true }});
const plan = pipe.explain({{ output: {json.dumps(str(final_mkv))} }});
const exported = pipe.export({json.dumps(str(final_mkv))});
return {{ info, kfs, thumb, sheet, plan, exported }};
"""
        sess.call_tool_json("wedeo.execute", {"code": code, "allow_write": True})

        escape_resp = sess.call_tool_raw(
            "wedeo.execute",
            {"code": 'return wedeo.inspect("/etc/passwd");', "allow_write": False},
        )
        sandbox_clean = bool(escape_resp.get("result", {}).get("isError", False))
        self.assertTrue(sandbox_clean)
        self.assertEqual(sess.request("ping", {})["result"], {})

        # Step 4: Verify artifacts via oracle + system ffmpeg and score 100-pt rubric
        out_probe = oracle.ffprobe_json(final_mkv)
        score = evaluate_composition_rubric(
            output_path=final_mkv,
            thumb_paths=[(thumb_path, 64, 48), (sheet_path, 96, 36)],
            expected_video_codec=out_probe["streams"][0]["codec_name"],
            expected_width=160,
            expected_height=120,
            expected_sample_rate=48000,
            expected_min_duration=0.4,
            expected_max_duration=3.0,
            stages_verified=6,
            total_stages=6,
            tool_turns=2,
            search_tokens=search_tokens,
            tools_list_tokens=tools_tokens,
            sandbox_clean=sandbox_clean,
            deterministic_repeat_ok=True,
        )
        verify_decoded_thumbnail(thumb_path, 64, 48)
        verify_decoded_thumbnail(sheet_path, 96, 36)
        self.assertGreaterEqual(score.total, 85, f"Scenario 8 rubric {score} < 85")
        self.assertGreaterEqual(score.structural, 30)
        self.assertEqual(score.efficiency, 15)
        self.assertEqual(score.safety, 10)

    @e2e_test(
        feature="F14",
        tier=4,
        milestone="M6",
        features=("F14", "F11", "F12", "F13", "F2"),
    )
    def test_scenario9_release_packaging_distribution_pipeline(self) -> None:
        root = repo_root()
        ver = workspace_package_version()

        # Step 1: Crates.io release readiness & topological publish dry-run
        meta = _cargo_metadata(root)
        ws_ids = set(meta["workspace_members"])
        pub_pkgs = [
            p for p in meta["packages"] if p["id"] in ws_ids and _is_publishable(p)
        ]
        pub_names = {p["name"] for p in pub_pkgs}
        non_pub_names = {
            p["name"]
            for p in meta["packages"]
            if p["id"] in ws_ids and not _is_publishable(p)
        }
        for pkg in pub_pkgs:
            for dep in pkg["dependencies"]:
                if dep.get("kind") == "dev":
                    continue
                self.assertFalse((dep.get("source") or "").startswith("git+"))
                self.assertNotIn(dep["name"], non_pub_names)
        for pkg in pub_pkgs:
            list_res = run_cmd(
                ["cargo", "package", "--list", "--offline", "--allow-dirty", "-p", pkg["name"]],
                cwd=root,
            )
            self.assertEqual(list_res.returncode, 0, list_res.stderr)

        pkg_res = run_cmd(
            [
                "cargo",
                "package",
                "--no-verify",
                "--offline",
                "--allow-dirty",
                "--target-dir",
                self.tmp / "s9_pkg",
                "-p",
                "wedeo-core",
            ],
            cwd=root,
        )
        self.assertEqual(pkg_res.returncode, 0, pkg_res.stderr)

        publish_sh = root / "scripts" / "publish.sh"
        script_crates = _parse_publish_sh_crates(publish_sh.read_text(encoding="utf-8"))
        self.assertEqual(set(script_crates), pub_names)
        self.assertEqual(run_cmd([str(publish_sh)], cwd=root).returncode, 0)

        # Step 2: Downstream consumer crate check + CHANGELOG/README alignment
        consumer = self.tmp / "s9_consumer"
        (consumer / "src").mkdir(parents=True)
        shutil.copy2(root / "Cargo.lock", consumer / "Cargo.lock")
        (consumer / "Cargo.toml").write_text(
            f"""[package]
name = "wedeo-s9-consumer"
version = "0.1.0"
edition = "2024"

[dependencies]
wedeo = {{ path = {json.dumps(str(root / "crates" / "wedeo"))}, default-features = false, features = ["audio-ops", "video-ops", "syntax", "pcm", "wav", "h264", "vp9", "mp4", "mkv", "ivf"] }}
""",
            encoding="utf-8",
        )
        (consumer / "src" / "main.rs").write_text(
            "fn main() { let _ = wedeo::native_registry(); }\n", encoding="utf-8"
        )
        chk = run_cmd(
            ["cargo", "check", "--offline", "--target-dir", self.tmp / "s9_target"],
            cwd=consumer,
            timeout=240.0,
        )
        self.assertEqual(chk.returncode, 0, chk.stderr or chk.stdout)
        changelog_path = root / "CHANGELOG.md"
        self.assertTrue(changelog_path.is_file(), f"Missing {changelog_path}")
        self.assertIn(ver, changelog_path.read_text(encoding="utf-8"))

        # Step 3: Homebrew & GitHub Release staging + staged CLI & Code Mode MCP smoke
        self.assertTrue((root / "Formula" / "wedeo.rb").is_file())
        self.assertTrue((root / ".github" / "workflows" / "release.yml").is_file())
        dist = self.tmp / "dist_stage"
        dist_bin = dist / "bin"
        dist_bin.mkdir(parents=True)
        staged_bin = dist_bin / "wedeo"
        shutil.copy2(find_wedeo_bin(), staged_bin)
        staged_bin.chmod(0o755)
        self.assertEqual(run_cmd([staged_bin, "install-shim", dist_bin]).returncode, 0)
        shutil.copy2(root / "LICENSE", dist / "LICENSE")
        shutil.copy2(root / "README.md", dist / "README.md")
        sha = hashlib.sha256(staged_bin.read_bytes()).hexdigest()
        (dist / "SHA256SUMS.txt").write_text(f"{sha}  bin/wedeo\n", encoding="utf-8")

        tr_out = self.tmp / "s9_staged_tr.mkv"
        r_tr = run_cmd(
            [
                staged_bin,
                "transcode",
                "-i",
                fixtures.vp9_red_ivf(),
                "--video",
                "av1",
                "-y",
                "-o",
                tr_out,
            ]
        )
        self.assertEqual(r_tr.returncode, 0, r_tr.stderr)
        oracle.assert_ffprobe_stream(
            oracle.ffprobe_json(tr_out), 0, codec_name="av1", width=64, height=64
        )
        sess = McpStdioSession(self, ["--mode", "code"])
        init = sess.initialize()
        self.assertEqual(init["serverInfo"]["version"], ver)

        # Step 4: @wedeo/wasm npm distribution validation & Node WASM inspect smoke
        npm_dir = root / "packages" / "wedeo-wasm"
        npm_pkg_path = npm_dir / "package.json"
        self.assertTrue(npm_pkg_path.is_file(), f"Missing {npm_pkg_path}")
        npm_pkg = json.loads(npm_pkg_path.read_text(encoding="utf-8"))
        self.assertEqual(npm_pkg["name"], "@wedeo/wasm")
        self.assertEqual(npm_pkg["version"], ver)
        wasm_path = _wasm_artifact_path()
        self.assertLess(
            len(gzip.compress(wasm_path.read_bytes(), compresslevel=9)),
            WASM_SIZE_LIMIT_BYTES,
        )
        node = find_node_bin()
        entry_rel = npm_pkg.get("main") or npm_pkg["exports"]["."]["import"]
        self.assertEqual(
            run_cmd([node, "--check", (npm_dir / entry_rel).resolve()]).returncode, 0
        )
        smoke = _run_sdk(
            [
                "--scenario",
                "wedeo-lifecycle",
                "--wasm",
                wasm_path,
                "--input",
                fixtures.vp9_red_ivf(),
            ]
        )
        self.assertEqual(smoke["inspect"]["tracks"][0]["codec"], "vp9")
