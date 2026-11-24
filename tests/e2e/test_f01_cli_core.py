"""E2E tests for F1: wedeo Core CLI (inspect, explain, codecs, formats, --version/--help, --json)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lib import fixtures
from lib.harness import assert_cli_error, e2e_test, run_wedeo


class TestF01CliCore(unittest.TestCase):
    """Tier 1 and Tier 2 E2E tests for F1 Core CLI."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_f01_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    @e2e_test(feature="F1", tier=1, milestone="baseline")
    def test_t1_version_help_and_empty_args(self) -> None:
        for flag in ("--version", "-V", "version"):
            res = run_wedeo([flag])
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(res.stdout, "wedeo 0.3.0\n")

        subcommands = (
            "inspect",
            "explain",
            "transcode",
            "cut",
            "mux",
            "ffmpeg",
            "ffprobe",
            "install-shim",
            "mcp",
            "codecs",
            "formats",
        )
        for args in ([], ["--help"], ["-h"], ["help"]):
            res = run_wedeo(args)
            self.assertEqual(res.returncode, 0, res.stderr)
            for sub in subcommands:
                self.assertIn(sub, res.stdout)

        for sub in ("inspect", "explain"):
            res = run_wedeo([sub, "--help"])
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("Usage:", res.stdout)

    @e2e_test(feature="F1", tier=1, milestone="baseline")
    def test_t1_codecs_and_formats_tables(self) -> None:
        codecs_res = run_wedeo(["codecs"])
        self.assertEqual(codecs_res.returncode, 0, codecs_res.stderr)
        for token in ("av1", "DEV...", "vp9", "D.V...", "h264", "aac", "D.A...", "pcm_s16le", "DEA..."):
            self.assertIn(token, codecs_res.stdout)

        formats_res = run_wedeo(["formats"])
        self.assertEqual(formats_res.returncode, 0, formats_res.stderr)
        for name, caps in (
            ("ivf", "DM"),
            ("matroska", "DM"),
            ("mp4", "DM"),
            ("wav", "DM"),
            ("webm", ".M"),
        ):
            lines = [ln for ln in formats_res.stdout.splitlines() if name in ln.split()]
            self.assertTrue(lines, f"Missing format {name!r} in:\n{formats_res.stdout}")
            self.assertIn(caps, lines[0])

    @e2e_test(feature="F1", tier=1, milestone="baseline")
    def test_t1_inspect_containers_and_ast(self) -> None:
        cases = [
            (fixtures.vp9_red_ivf(), "ivf", "DKIF", "video", "vp9"),
            (fixtures.h264_aac_mp4(), "mp4", "ftyp", "video", "h264"),
            (fixtures.vp9_smpte_mkv(), "matroska", "EBML", "video", "vp9"),
            (fixtures.chord_48k_wav(), "wav", "RIFF", "audio", "pcm_s16le"),
        ]
        for path, fmt, root_kind, media_type, codec in cases:
            res = run_wedeo(["inspect", "--json", path])
            self.assertEqual(res.returncode, 0, f"{path}: {res.stderr}")
            data = json.loads(res.stdout)
            self.assertEqual(data["format_name"], fmt)
            self.assertIs(data["verified"], False)
            ast_kinds = [node["kind"] for node in data["container_ast"]]
            self.assertIn(root_kind, ast_kinds)
            self.assertEqual(data["tracks"][0]["media_type"], media_type)
            self.assertEqual(data["tracks"][0]["codec"], codec)
            if media_type == "video":
                self.assertGreater(data["tracks"][0]["width"], 0)
                self.assertGreater(data["tracks"][0]["height"], 0)
            else:
                self.assertEqual(data["tracks"][0]["sample_rate"], 48000)
                self.assertEqual(data["tracks"][0]["channels"], 1)

        human = run_wedeo(["inspect", fixtures.vp9_red_ivf()])
        self.assertEqual(human.returncode, 0, human.stderr)
        self.assertIn("ivf", human.stdout)
        self.assertIn("vp9", human.stdout)

    @e2e_test(feature="F1", tier=1, milestone="baseline")
    def test_t1_inspect_levels_and_verify(self) -> None:
        ivf_res = run_wedeo(
            ["inspect", "--verify", "--level", "nal", "--json", fixtures.vp9_red_ivf()]
        )
        self.assertEqual(ivf_res.returncode, 0, ivf_res.stderr)
        ivf_json = json.loads(ivf_res.stdout)
        self.assertIs(ivf_json["verified"], True)
        self.assertEqual(ivf_json["decoded_units"], 1)
        self.assertEqual(ivf_json["packets"][0]["units"][0]["type_name"], "keyframe")
        self.assertIs(ivf_json["health"]["is_clean"], True)

        mp4_gop = run_wedeo(
            ["inspect", "--level", "gop", "--json", fixtures.h264_aac_mp4()]
        )
        self.assertEqual(mp4_gop.returncode, 0, mp4_gop.stderr)
        gop_json = json.loads(mp4_gop.stdout)
        track0_gops = [g for g in gop_json["gops"] if g["track_id"] == 0]
        self.assertTrue(track0_gops)
        self.assertEqual(track0_gops[0]["packet_count"], 48)

        mp4_nal = run_wedeo(
            [
                "inspect",
                "--verify",
                "--level",
                "nal",
                "--max-packets",
                "5",
                "--json",
                fixtures.h264_aac_mp4(),
            ]
        )
        self.assertEqual(mp4_nal.returncode, 0, mp4_nal.stderr)
        nal_json = json.loads(mp4_nal.stdout)
        self.assertEqual(nal_json["decoded_units"], 143)
        self.assertEqual(len(nal_json["packets"]), 5)
        video_pkts = [p for p in nal_json["packets"] if p["track_id"] == 0]
        self.assertTrue(video_pkts and len(video_pkts[0]["units"]) > 0)
        self.assertIs(nal_json["health"]["is_clean"], True)

    @e2e_test(feature="F1", tier=1, milestone="baseline")
    def test_t1_inspect_subtitles_vtt_and_srt(self) -> None:
        vtt_path = fixtures.synth_subtitle_vtt(self.tmp / "sample.en.vtt")
        srt_path = fixtures.synth_subtitle_srt(self.tmp / "sample.de.srt")

        vtt_res = run_wedeo(
            ["inspect", "--level", "nal", "--verify", "--json", vtt_path]
        )
        self.assertEqual(vtt_res.returncode, 0, vtt_res.stderr)
        vtt = json.loads(vtt_res.stdout)
        self.assertEqual(vtt["format_name"], "webvtt")
        self.assertEqual(vtt["tracks"][0]["language"], "eng")
        self.assertEqual(vtt["tracks"][0]["subtitle_format"], "WebVtt")
        self.assertEqual(vtt["duration_sec"], 67)
        self.assertEqual(len(vtt["packets"]), 2)
        self.assertEqual(vtt["packets"][0]["pts"], 500)
        self.assertEqual(vtt["packets"][0]["duration"], 1500)
        self.assertEqual(vtt["packets"][1]["pts"], 65250)
        self.assertEqual(vtt["packets"][1]["duration"], 1750)
        self.assertIs(vtt["verified"], True)
        self.assertEqual(vtt["decoded_units"], 0)
        self.assertIs(vtt["health"]["is_clean"], True)

        srt_res = run_wedeo(["inspect", "--level", "gop", "--json", srt_path])
        self.assertEqual(srt_res.returncode, 0, srt_res.stderr)
        srt = json.loads(srt_res.stdout)
        self.assertEqual(srt["format_name"], "srt")
        self.assertEqual(srt["tracks"][0]["language"], "deu")
        self.assertEqual(len(srt["gops"]), 2)

    @e2e_test(feature="F1", tier=1, milestone="baseline")
    def test_t1_explain_json_and_human_no_output_file(self) -> None:
        out1 = self.tmp / "never1.mp4"
        out2 = self.tmp / "never2.mp4"
        out3 = self.tmp / "never3.mp4"

        res1 = run_wedeo(
            [
                "explain",
                "--json",
                "-i",
                fixtures.vp9_red_ivf(),
                "--video",
                "av1",
                "--size",
                "32x32",
                "--gpu",
                "-o",
                out1,
            ]
        )
        self.assertEqual(res1.returncode, 0, res1.stderr)
        self.assertFalse(out1.exists())
        j1 = json.loads(res1.stdout)
        node_names = [n["name"] for n in j1["nodes"]]
        self.assertIn("sws_op_list", node_names)
        self.assertTrue(any(n.startswith("gpu_upload") for n in node_names))
        self.assertTrue(any(n.startswith("gpu_download") for n in node_names))
        self.assertTrue(j1["edges"])
        self.assertTrue(j1["order"])
        self.assertTrue(any(c["is_lossy"] for c in j1["conversion_notes"]))

        res2 = run_wedeo(
            [
                "explain",
                "--json",
                "-i",
                fixtures.h264_aac_mp4(),
                "-an",
                "--video",
                "av1",
                "--size",
                "160x120",
                "-o",
                out2,
            ]
        )
        self.assertEqual(res2.returncode, 0, res2.stderr)
        self.assertFalse(out2.exists())
        j2 = json.loads(res2.stdout)
        sws_nodes = [n for n in j2["nodes"] if n["name"] == "sws_op_list"]
        self.assertTrue(sws_nodes and sws_nodes[0]["auto_bridge"] is True)

        res3 = run_wedeo(
            [
                "explain",
                "-i",
                fixtures.vp9_red_ivf(),
                "--video",
                "av1",
                "-o",
                out3,
            ]
        )
        self.assertEqual(res3.returncode, 0, res3.stderr)
        self.assertFalse(out3.exists())
        self.assertIn("ExecutionPlan (", res3.stdout)
        self.assertIn("Topological Order:", res3.stdout)

    @e2e_test(feature="F1", tier=2, milestone="baseline")
    def test_t2_unknown_subcommand_flags_and_missing_args(self) -> None:
        assert_cli_error(run_wedeo(["boguscmd"]), "[wedeo]")
        assert_cli_error(run_wedeo(["inspect"]), "[inspect]")
        assert_cli_error(
            run_wedeo(["inspect", "--bogus", fixtures.vp9_red_ivf()]),
            "[inspect]",
        )
        assert_cli_error(
            run_wedeo(
                ["inspect", fixtures.vp9_red_ivf(), fixtures.vp9_blue_ivf()]
            ),
            "[inspect]",
        )
        assert_cli_error(
            run_wedeo(["explain", "-i", fixtures.vp9_red_ivf()]),
            "[explain]",
        )

    @e2e_test(feature="F1", tier=2, milestone="baseline")
    def test_t2_inspect_max_packets_boundaries(self) -> None:
        mkv = fixtures.vp9_smpte_mkv()

        r0 = run_wedeo(
            ["inspect", "--level", "packets", "--max-packets", "0", "--json", mkv]
        )
        self.assertEqual(r0.returncode, 0, r0.stderr)
        j0 = json.loads(r0.stdout)
        self.assertEqual(j0["packets"], [])
        self.assertEqual(j0["decoded_units"], 0)

        r0v = run_wedeo(
            [
                "inspect",
                "--verify",
                "--level",
                "packets",
                "--max-packets",
                "0",
                "--json",
                mkv,
            ]
        )
        self.assertEqual(r0v.returncode, 0, r0v.stderr)
        j0v = json.loads(r0v.stdout)
        self.assertEqual(j0v["packets"], [])
        self.assertEqual(j0v["decoded_units"], 48)

        assert_cli_error(
            run_wedeo(["inspect", "--max-packets", "-1", mkv]),
            "invalid --max-packets",
        )
        assert_cli_error(
            run_wedeo(["inspect", "--max-packets", "abc", mkv]),
            "invalid --max-packets",
        )

    @e2e_test(feature="F1", tier=2, milestone="baseline")
    def test_t2_inspect_invalid_level_missing_and_corrupt(self) -> None:
        assert_cli_error(
            run_wedeo(["inspect", "--level", "bogus", fixtures.vp9_red_ivf()]),
            "unknown inspect level",
        )
        assert_cli_error(
            run_wedeo(["inspect", self.tmp / "does_not_exist.mp4"]),
            "No such file or directory",
        )
        garbage = self.tmp / "garbage.bin"
        garbage.write_bytes(b"\x00\xff\x13\x37not_a_media_container" * 32)
        assert_cli_error(
            run_wedeo(["inspect", garbage]),
            "no registered demuxer matched",
        )

    @e2e_test(feature="F1", tier=2, milestone="baseline")
    def test_t2_inspect_subtitle_edge_cases(self) -> None:
        empty_vtt = fixtures.synth_subtitle_vtt(
            self.tmp / "empty.vtt", content="WEBVTT\n\n"
        )
        bad_srt = fixtures.synth_subtitle_srt(
            self.tmp / "bad_ts.srt",
            content="1\nnot_a_ts --> also_bad\nIgnored cue\n",
        )
        rev_vtt = fixtures.synth_subtitle_vtt(
            self.tmp / "reversed.vtt",
            content="WEBVTT\n\n00:00:05.000 --> 00:00:01.000\nClamped cue\n",
        )
        non_utf8 = self.tmp / "non_utf8.vtt"
        non_utf8.write_bytes(b"WEBVTT\n\n00:00.000 --> 00:01.000\n\xff\xfe\xfd\n")

        r_empty = run_wedeo(["inspect", "--verify", "--json", empty_vtt])
        self.assertEqual(r_empty.returncode, 0, r_empty.stderr)
        j_empty = json.loads(r_empty.stdout)
        self.assertIsNone(j_empty["duration_sec"])
        self.assertEqual(j_empty["packets"], [])

        r_bad = run_wedeo(["inspect", "--json", bad_srt])
        self.assertEqual(r_bad.returncode, 0, r_bad.stderr)
        self.assertEqual(json.loads(r_bad.stdout)["packets"], [])

        r_rev = run_wedeo(["inspect", "--level", "packets", "--json", rev_vtt])
        self.assertEqual(r_rev.returncode, 0, r_rev.stderr)
        j_rev = json.loads(r_rev.stdout)
        self.assertEqual(j_rev["packets"][0]["duration"], 1)

        assert_cli_error(
            run_wedeo(["inspect", non_utf8]),
            "[wedeo-cli/subtitle]",
        )

    @e2e_test(feature="F1", tier=2, milestone="baseline")
    def test_t2_explain_strict_lossy_vs_lossless(self) -> None:
        out_lossy = self.tmp / "strict_lossy.mp4"
        res_lossy = run_wedeo(
            [
                "explain",
                "--strict",
                "--json",
                "-i",
                fixtures.vp9_red_ivf(),
                "--video",
                "av1",
                "--size",
                "32x32",
                "-o",
                out_lossy,
            ]
        )
        assert_cli_error(res_lossy, "strict mode rejected implicit lossy video bridge")
        self.assertFalse(out_lossy.exists())

        out_ok = self.tmp / "strict_ok.mp4"
        res_ok = run_wedeo(
            [
                "explain",
                "--strict",
                "--json",
                "-i",
                fixtures.vp9_red_ivf(),
                "--video",
                "av1",
                "-o",
                out_ok,
            ]
        )
        self.assertEqual(res_ok.returncode, 0, res_ok.stderr)
        self.assertFalse(out_ok.exists())
        j_ok = json.loads(res_ok.stdout)
        self.assertIs(j_ok["strict"], True)
        self.assertEqual(j_ok["conversion_notes"], [])

    @e2e_test(feature="F1", tier=2, milestone="baseline")
    def test_t2_explain_invalid_size_ext_and_missing_encoder(self) -> None:
        out_zero = self.tmp / "out_zero.mp4"
        assert_cli_error(
            run_wedeo(
                [
                    "explain",
                    "-i",
                    fixtures.vp9_red_ivf(),
                    "--size",
                    "0x0",
                    "-o",
                    out_zero,
                ]
            ),
            "must be non-zero",
        )
        self.assertFalse(out_zero.exists())

        out_bad = self.tmp / "out_bad.mp4"
        assert_cli_error(
            run_wedeo(
                [
                    "explain",
                    "-i",
                    fixtures.vp9_red_ivf(),
                    "--size",
                    "bad",
                    "-o",
                    out_bad,
                ]
            ),
            "invalid size",
        )
        self.assertFalse(out_bad.exists())

        out_xyz = self.tmp / "out.xyz"
        assert_cli_error(
            run_wedeo(
                [
                    "explain",
                    "-i",
                    fixtures.vp9_red_ivf(),
                    "-o",
                    out_xyz,
                ]
            ),
            "no muxer registered",
        )
        self.assertFalse(out_xyz.exists())

        out_aac = self.tmp / "out_aac.mp4"
        assert_cli_error(
            run_wedeo(
                [
                    "explain",
                    "-i",
                    fixtures.h264_aac_mp4(),
                    "--video",
                    "av1",
                    "-o",
                    out_aac,
                ]
            ),
            "no encoder for 'aac'",
        )
        self.assertFalse(out_aac.exists())
