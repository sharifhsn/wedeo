"""Tier 3 pairwise combinatorial E2E tests for EM1 baseline features (F1, F2, F3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lib import fixtures, oracle
from lib.harness import assert_cli_error, e2e_test, run_cmd, run_wedeo


class TestTier3Em1Pairwise(unittest.TestCase):
    """Pairwise cross-feature E2E tests across F1, F2, and F3."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_t3_em1_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    @e2e_test(feature="F1", tier=3, milestone="baseline", features=("F1", "F2"))
    def test_f1_f2_explain_transcode_inspect_verify_roundtrip(self) -> None:
        src = fixtures.vp9_smpte_mkv()
        out = self.tmp / "tr.mkv"

        exp_res = run_wedeo(
            [
                "explain",
                "-i",
                src,
                "-o",
                out,
                "--video",
                "av1",
                "--size",
                "160x120",
                "--frames",
                "4",
                "--json",
            ]
        )
        self.assertEqual(exp_res.returncode, 0, exp_res.stderr)
        self.assertFalse(out.exists())
        exp = json.loads(exp_res.stdout)
        node_names = [n["name"] for n in exp["nodes"]]
        for expected in (
            "demuxer(matroska)",
            "decoder(vp9)",
            "sws_op_list",
            "encoder(av1)",
            "muxer(matroska)",
        ):
            self.assertIn(expected, node_names)
        self.assertTrue(
            any(
                "downscale (320x240 -> 160x120)" in c["note"]
                for c in exp["conversion_notes"]
            )
        )

        tr_res = run_wedeo(
            [
                "transcode",
                "-i",
                src,
                "-o",
                out,
                "--video",
                "av1",
                "--size",
                "160x120",
                "--frames",
                "4",
                "-y",
                "--json",
            ]
        )
        self.assertEqual(tr_res.returncode, 0, tr_res.stderr)
        tr = json.loads(tr_res.stdout)
        self.assertIs(tr["health"]["is_clean"], True)
        self.assertGreater(tr["units_processed"], 0)

        insp = oracle.assert_wedeo_inspect_clean(run_wedeo, out, verify=True)
        self.assertEqual(insp["decoded_units"], 4)
        self.assertEqual(insp["tracks"][0]["codec"], "av1")
        self.assertEqual(insp["tracks"][0]["width"], 160)
        self.assertEqual(insp["tracks"][0]["height"], 120)

    @e2e_test(feature="F2", tier=3, milestone="baseline", features=("F1", "F2"))
    def test_f1_f2_strict_mode_rejection_vs_permissive_transcode(self) -> None:
        src = fixtures.vp9_smpte_mkv()
        strict_out = self.tmp / "strict.mkv"

        exp_strict = run_wedeo(
            [
                "explain",
                "-i",
                src,
                "-o",
                strict_out,
                "--video",
                "av1",
                "--size",
                "160x120",
                "--strict",
                "--json",
            ]
        )
        assert_cli_error(exp_strict, "strict mode rejected implicit lossy video bridge")
        self.assertFalse(strict_out.exists())

        tr_strict = run_wedeo(
            [
                "transcode",
                "-i",
                src,
                "-o",
                strict_out,
                "--video",
                "av1",
                "--size",
                "160x120",
                "--frames",
                "4",
                "--strict",
            ]
        )
        assert_cli_error(tr_strict, "strict mode rejected implicit lossy video bridge")
        if strict_out.exists():
            self.assertEqual(strict_out.stat().st_size, 0)

        perm_out = self.tmp / "permissive.mkv"
        tr_ok = run_wedeo(
            [
                "transcode",
                "-i",
                src,
                "-o",
                perm_out,
                "--video",
                "av1",
                "--size",
                "160x120",
                "--frames",
                "4",
                "--json",
            ]
        )
        self.assertEqual(tr_ok.returncode, 0, tr_ok.stderr)
        self.assertTrue(perm_out.is_file())
        oracle.assert_wedeo_inspect_clean(run_wedeo, perm_out, verify=True)

    @e2e_test(feature="F2", tier=3, milestone="baseline", features=("F2", "F3"))
    def test_f2_f3_transcode_audio_then_shim_map_mux_oracle(self) -> None:
        shim_dir = self.tmp / "shims"
        self.assertEqual(run_wedeo(["install-shim", shim_dir]).returncode, 0)
        ffmpeg_shim = shim_dir / "ffmpeg"
        ffprobe_shim = shim_dir / "ffprobe"

        stereo_24k = self.tmp / "stereo24k.wav"
        tr_res = run_wedeo(
            [
                "transcode",
                "-i",
                fixtures.chord_48k_wav(),
                "--sample-rate",
                "24000",
                "--channels",
                "2",
                "-o",
                stereo_24k,
            ]
        )
        self.assertEqual(tr_res.returncode, 0, tr_res.stderr)

        muxed = self.tmp / "muxed.mkv"
        ff_res = run_cmd(
            [
                ffmpeg_shim,
                "-i",
                fixtures.vp9_testsrc_320_ivf(),
                "-i",
                stereo_24k,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-y",
                muxed,
            ]
        )
        self.assertEqual(ff_res.returncode, 0, ff_res.stderr)

        w_probe = oracle.ffprobe_json(muxed, ffprobe_bin=ffprobe_shim)
        o_probe = oracle.ffprobe_json(muxed)
        oracle.assert_ffprobe_parity(w_probe, o_probe)
        oracle.assert_ffprobe_stream(
            o_probe, 0, codec_name="vp9", width=320, height=240
        )
        oracle.assert_ffprobe_stream(
            o_probe, 1, codec_name="pcm_s16le", sample_rate=24000, channels=2
        )
        oracle.oracle_ffmpeg_decode_check(muxed)

    @e2e_test(feature="F3", tier=3, milestone="baseline", features=("F1", "F3"))
    def test_f1_f3_codecs_formats_cross_check_with_ffmpeg_shim(self) -> None:
        c_core = run_wedeo(["codecs"])
        c_shim = run_wedeo(["ffmpeg", "-codecs"])
        self.assertEqual(c_core.returncode, 0, c_core.stderr)
        self.assertEqual(c_shim.returncode, 0, c_shim.stderr)
        self.assertEqual(c_core.stdout, c_shim.stdout)

        f_core = run_wedeo(["formats"])
        f_shim = run_wedeo(["ffmpeg", "-formats"])
        self.assertEqual(f_core.returncode, 0, f_core.stderr)
        self.assertEqual(f_shim.returncode, 0, f_shim.stderr)
        self.assertEqual(f_core.stdout, f_shim.stdout)

        sample = fixtures.vp9_smpte_mkv()
        insp = json.loads(run_wedeo(["inspect", "--json", sample]).stdout)
        ffp = oracle.ffprobe_json(
            sample, ffprobe_bin=None, show_format=True, show_streams=True
        )
        w_ffp_res = run_wedeo(
            [
                "ffprobe",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                sample,
            ]
        )
        self.assertEqual(w_ffp_res.returncode, 0, w_ffp_res.stderr)
        w_ffp = json.loads(w_ffp_res.stdout)

        self.assertEqual(len(insp["tracks"]), w_ffp["format"]["nb_streams"])
        self.assertEqual(insp["tracks"][0]["codec"], w_ffp["streams"][0]["codec_name"])
        self.assertEqual(insp["tracks"][0]["width"], w_ffp["streams"][0]["width"])
        self.assertEqual(insp["tracks"][0]["height"], w_ffp["streams"][0]["height"])
        self.assertAlmostEqual(
            float(insp["duration_sec"]), float(ffp["format"]["duration"]), delta=0.05
        )

    @e2e_test(
        feature="F3", tier=3, milestone="baseline", features=("F1", "F2", "F3")
    )
    def test_f1_f2_f3_cut_then_shim_wav_extract_and_remux(self) -> None:
        shim_dir = self.tmp / "shims"
        self.assertEqual(run_wedeo(["install-shim", shim_dir]).returncode, 0)
        ffmpeg_shim = shim_dir / "ffmpeg"

        v_cut = self.tmp / "v_cut.mkv"
        r_cut = run_wedeo(
            [
                "cut",
                "-i",
                fixtures.vp9_smpte_mkv(),
                "--from",
                "0",
                "--to",
                "1.0",
                "--mode",
                "keyframe",
                "-y",
                "-o",
                v_cut,
            ]
        )
        self.assertEqual(r_cut.returncode, 0, r_cut.stderr)

        a_trim = self.tmp / "a_trim.wav"
        r_aud = run_cmd(
            [
                ffmpeg_shim,
                "-i",
                fixtures.chord_48k_wav(),
                "-ss",
                "0.25",
                "-t",
                "0.5",
                "-ar",
                "22050",
                "-ac",
                "2",
                "-y",
                a_trim,
            ]
        )
        self.assertEqual(r_aud.returncode, 0, r_aud.stderr)

        merged = self.tmp / "merged.mkv"
        r_mux = run_cmd(
            [
                ffmpeg_shim,
                "-i",
                v_cut,
                "-i",
                a_trim,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-y",
                merged,
            ]
        )
        self.assertEqual(r_mux.returncode, 0, r_mux.stderr)

        oracle.assert_wedeo_inspect_clean(run_wedeo, merged, verify=True)
        o_probe = oracle.ffprobe_json(merged)
        oracle.assert_ffprobe_stream(
            o_probe, 0, codec_name="vp9", width=320, height=240
        )
        oracle.assert_ffprobe_stream(
            o_probe, 1, codec_name="pcm_s16le", sample_rate=22050, channels=2
        )
        oracle.oracle_ffmpeg_decode_check(merged)
