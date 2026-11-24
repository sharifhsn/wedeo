"""E2E tests for F3: ffmpeg & ffprobe Multi-Call Shims."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lib import fixtures, oracle
from lib.harness import (
    assert_cli_error,
    augmented_env,
    e2e_test,
    find_wedeo_bin,
    run_cmd,
    run_wedeo,
)


def _parse_ffprobe_ini(text: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    streams: list[dict[str, str]] = []
    fmt: dict[str, str] = {}
    current: dict[str, str] | None = None
    in_stream = False
    in_fmt = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "[STREAM]":
            current = {}
            in_stream = True
            in_fmt = False
        elif line == "[/STREAM]":
            if current is not None:
                streams.append(current)
            current = None
            in_stream = False
        elif line == "[FORMAT]":
            current = {}
            in_fmt = True
            in_stream = False
        elif line == "[/FORMAT]":
            if current is not None:
                fmt = current
            current = None
            in_fmt = False
        elif "=" in line and current is not None and (in_stream or in_fmt):
            k, v = line.split("=", 1)
            current[k] = v
    return streams, fmt


class TestF03Shims(unittest.TestCase):
    """Tier 1 and Tier 2 E2E tests for F3 ffmpeg/ffprobe shims."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_f03_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    def _install_shims(self) -> tuple[Path, Path]:
        shim_dir = self.tmp / "shims"
        res = run_wedeo(["install-shim", shim_dir])
        self.assertEqual(res.returncode, 0, res.stderr)
        ffmpeg_shim = shim_dir / "ffmpeg"
        ffprobe_shim = shim_dir / "ffprobe"
        return ffmpeg_shim, ffprobe_shim

    @e2e_test(feature="F3", tier=1, milestone="baseline")
    def test_t1_install_shim_and_argv0_version(self) -> None:
        ffmpeg_shim, ffprobe_shim = self._install_shims()
        wedeo_bin = find_wedeo_bin()

        for path in (ffmpeg_shim, ffprobe_shim):
            self.assertTrue(path.is_symlink(), f"Expected symlink: {path}")
            self.assertEqual(path.resolve(), wedeo_bin)

        ff_res = run_cmd([ffmpeg_shim, "-version"])
        self.assertEqual(ff_res.returncode, 0, ff_res.stderr)
        self.assertTrue(ff_res.stdout.startswith("ffmpeg version 9.1-wedeo-"))
        for lib in ("libavutil", "libavcodec", "libavformat"):
            self.assertIn(lib, ff_res.stdout)

        fp_res = run_cmd([ffprobe_shim, "-version"])
        self.assertEqual(fp_res.returncode, 0, fp_res.stderr)
        self.assertTrue(fp_res.stdout.startswith("ffprobe version 9.1-wedeo-"))
        for lib in ("libavutil", "libavcodec", "libavformat"):
            self.assertIn(lib, fp_res.stdout)

    @e2e_test(feature="F3", tier=1, milestone="baseline")
    def test_t1_ffmpeg_info_flags_bsfs_codecs_formats(self) -> None:
        bsfs = run_wedeo(["ffmpeg", "-bsfs"])
        self.assertEqual(bsfs.returncode, 0, bsfs.stderr)
        self.assertIn("Bitstream filters:", bsfs.stdout)
        for name in ("h264_mp4toannexb", "aac_adtstoasc", "vp9_superframe_split", "null"):
            self.assertIn(name, bsfs.stdout)

        codecs = run_wedeo(["ffmpeg", "-codecs"])
        self.assertEqual(codecs.returncode, 0, codecs.stderr)
        for token in ("DEV...", "av1", "h264", "vp9", "pcm_s16le", "aac"):
            self.assertIn(token, codecs.stdout)

        formats = run_wedeo(["ffmpeg", "-formats"])
        self.assertEqual(formats.returncode, 0, formats.stderr)
        for name in ("matroska", "mp4", "wav", "ivf"):
            self.assertIn(name, formats.stdout)

        decoders = run_wedeo(["ffmpeg", "-decoders"])
        self.assertEqual(decoders.returncode, 0, decoders.stderr)
        self.assertIn("h264", decoders.stdout)

        encoders = run_wedeo(["ffmpeg", "-encoders"])
        self.assertEqual(encoders.returncode, 0, encoders.stderr)
        self.assertIn("av1", encoders.stdout)

    @e2e_test(feature="F3", tier=1, milestone="baseline")
    def test_t1_ffprobe_json_oracle_parity_mp4_and_mkv(self) -> None:
        _, ffprobe_shim = self._install_shims()
        for sample in (fixtures.h264_aac_mp4(), fixtures.vp9_smpte_mkv()):
            w_res = run_cmd(
                [
                    ffprobe_shim,
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    sample,
                ]
            )
            self.assertEqual(w_res.returncode, 0, w_res.stderr)
            w_json = json.loads(w_res.stdout)
            o_json = oracle.ffprobe_json(sample)
            oracle.assert_ffprobe_parity(w_json, o_json)

    @e2e_test(feature="F3", tier=1, milestone="baseline")
    def test_t1_ffprobe_default_ini_parity(self) -> None:
        _, ffprobe_shim = self._install_shims()
        sample = fixtures.h264_aac_mp4()

        w_res = run_cmd(
            [ffprobe_shim, "-v", "error", "-show_format", "-show_streams", sample]
        )
        self.assertEqual(w_res.returncode, 0, w_res.stderr)
        o_res = run_cmd(
            [
                oracle.ffprobe_oracle_bin(),
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                sample,
            ]
        )
        self.assertEqual(o_res.returncode, 0, o_res.stderr)

        w_streams, w_fmt = _parse_ffprobe_ini(w_res.stdout)
        o_streams, o_fmt = _parse_ffprobe_ini(o_res.stdout)
        self.assertEqual(len(w_streams), len(o_streams))
        for k in ("nb_streams", "format_name", "duration", "size"):
            self.assertEqual(w_fmt[k], o_fmt[k])

        for ws, os_ in zip(w_streams, o_streams):
            for k in ("index", "codec_name", "codec_type", "time_base"):
                self.assertEqual(ws[k], os_[k])
            if os_["codec_type"] == "video":
                for k in ("width", "height", "pix_fmt"):
                    self.assertEqual(ws[k], os_[k])
            elif os_["codec_type"] == "audio":
                for k in ("sample_rate", "channels", "sample_fmt"):
                    self.assertEqual(ws[k], os_[k])

    @e2e_test(feature="F3", tier=1, milestone="baseline")
    def test_t1_ffmpeg_multi_input_map_mux_and_swapped_order(self) -> None:
        ffmpeg_shim, _ = self._install_shims()
        vid = fixtures.vp9_testsrc_320_ivf()
        aud = fixtures.chord_48k_wav()

        out_01 = self.tmp / "mux_01.mkv"
        r01 = run_cmd(
            [
                ffmpeg_shim,
                "-i",
                vid,
                "-i",
                aud,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-y",
                out_01,
            ]
        )
        self.assertEqual(r01.returncode, 0, r01.stderr)

        out_10 = self.tmp / "mux_10.mkv"
        r10 = run_cmd(
            [
                ffmpeg_shim,
                "-i",
                aud,
                "-i",
                vid,
                "-map",
                "1:v:0",
                "-map",
                "0:a:0",
                "-y",
                out_10,
            ]
        )
        self.assertEqual(r10.returncode, 0, r10.stderr)

        for out in (out_01, out_10):
            p = oracle.ffprobe_json(out)
            self.assertEqual(p["format"]["nb_streams"], 2)
            oracle.assert_ffprobe_stream(
                p, 0, codec_name="vp9", codec_type="video", width=320, height=240
            )
            oracle.assert_ffprobe_stream(
                p, 1, codec_name="pcm_s16le", codec_type="audio", sample_rate=48000, channels=1
            )
            oracle.oracle_ffmpeg_decode_check(out)

    @e2e_test(feature="F3", tier=1, milestone="baseline")
    def test_t1_ffmpeg_ffmetadata1_chapters_and_tags_mkv_and_mp4(self) -> None:
        ffmpeg_shim, ffprobe_shim = self._install_shims()
        meta_path = fixtures.synth_ffmetadata1(
            self.tmp / "meta.txt",
            tags={"title": "Sample Podcast", "artist": "Wedeo Team"},
            chapters=[
                (0, 1000, "Intro"),
                (1000, 2000, "Main Topic"),
            ],
        )

        out_mkv = self.tmp / "chap.mkv"
        r_mkv = run_cmd(
            [
                ffmpeg_shim,
                "-i",
                fixtures.vp9_smpte_320x240() if hasattr(fixtures, "vp9_smpte_320x240") else fixtures.vp9_smpte_mkv(),
                "-i",
                meta_path,
                "-c",
                "copy",
                "-y",
                out_mkv,
            ]
        )
        self.assertEqual(r_mkv.returncode, 0, r_mkv.stderr)

        out_mp4 = self.tmp / "chap.mp4"
        r_mp4 = run_cmd(
            [
                ffmpeg_shim,
                "-i",
                fixtures.h264_aac_mp4(),
                "-i",
                meta_path,
                "-c",
                "copy",
                "-y",
                out_mp4,
            ]
        )
        self.assertEqual(r_mp4.returncode, 0, r_mp4.stderr)

        for out in (out_mkv, out_mp4):
            w_probe = oracle.ffprobe_json(
                out, ffprobe_bin=ffprobe_shim, show_chapters=True
            )
            o_probe = oracle.ffprobe_json(out, show_chapters=True)
            oracle.assert_ffprobe_parity(
                w_probe,
                o_probe,
                check_chapters=True,
                expected_tags={"title": "Sample Podcast", "artist": "Wedeo Team"},
            )
            self.assertEqual(len(w_probe["chapters"]), 2)
            self.assertEqual(w_probe["chapters"][0]["tags"]["title"], "Intro")
            self.assertEqual(w_probe["chapters"][1]["tags"]["title"], "Main Topic")
            oracle.oracle_ffmpeg_decode_check(out)

    @e2e_test(feature="F3", tier=2, milestone="baseline")
    def test_t2_file_proto_uri_stripping(self) -> None:
        ffmpeg_shim, ffprobe_shim = self._install_shims()
        sample_mkv = fixtures.vp9_smpte_mkv()

        p_res = run_cmd(
            [
                ffprobe_shim,
                "-print_format",
                "json",
                "-show_format",
                f"file:{sample_mkv}",
            ]
        )
        self.assertEqual(p_res.returncode, 0, p_res.stderr)
        p_json = json.loads(p_res.stdout)
        self.assertEqual(p_json["format"]["nb_streams"], 1)

        uri_out = self.tmp / "uri_out.mkv"
        f_res = run_cmd(
            [
                ffmpeg_shim,
                "-i",
                f"file://{sample_mkv}",
                "-c",
                "copy",
                "-y",
                f"file://{uri_out}",
            ]
        )
        self.assertEqual(f_res.returncode, 0, f_res.stderr)
        self.assertTrue(uri_out.is_file())
        oracle.assert_ffprobe_stream(
            oracle.ffprobe_json(uri_out), 0, codec_name="vp9", width=320, height=240
        )
        oracle.oracle_ffmpeg_decode_check(uri_out)

    @e2e_test(feature="F3", tier=2, milestone="baseline")
    def test_t2_overwrite_guard_n_vs_y(self) -> None:
        ffmpeg_shim, _ = self._install_shims()
        target = self.tmp / "exists.mkv"
        sentinel = b"sentinel_keep_me"
        target.write_bytes(sentinel)

        r_n = run_cmd(
            [ffmpeg_shim, "-i", fixtures.vp9_smpte_mkv(), "-c", "copy", "-n", target]
        )
        assert_cli_error(r_n, "already exists")
        self.assertEqual(target.read_bytes(), sentinel)

        r_default = run_cmd(
            [ffmpeg_shim, "-i", fixtures.vp9_smpte_mkv(), "-c", "copy", target]
        )
        assert_cli_error(r_default, "already exists")
        self.assertEqual(target.read_bytes(), sentinel)

        r_y = run_cmd(
            [ffmpeg_shim, "-i", fixtures.vp9_smpte_mkv(), "-c", "copy", "-y", target]
        )
        self.assertEqual(r_y.returncode, 0, r_y.stderr)
        oracle.assert_ffprobe_stream(oracle.ffprobe_json(target), 0, codec_name="vp9")

    @e2e_test(feature="F3", tier=2, milestone="baseline")
    def test_t2_missing_input_and_missing_required_args_exit_1(self) -> None:
        ffmpeg_shim, ffprobe_shim = self._install_shims()
        missing = self.tmp / "does_not_exist.mkv"

        assert_cli_error(
            run_cmd([ffmpeg_shim, "-i", missing, "-c", "copy", self.tmp / "out.mkv"]),
            "No such file or directory",
        )
        assert_cli_error(
            run_cmd([ffprobe_shim, missing]),
            "No such file or directory",
        )
        assert_cli_error(
            run_cmd([ffprobe_shim, "-print_format", "json"]),
            "an input media file path is required",
        )
        assert_cli_error(
            run_cmd([ffmpeg_shim, "-c", "copy", self.tmp / "out.mkv"]),
            "at least one input (-i <file>) is required",
        )

    @e2e_test(feature="F3", tier=2, milestone="baseline")
    def test_t2_single_input_map_stream_filtering_and_negative_map(self) -> None:
        ffmpeg_shim, _ = self._install_shims()
        mp4 = fixtures.h264_aac_mp4()

        only_a1 = self.tmp / "only_a1.mp4"
        r1 = run_cmd(
            [ffmpeg_shim, "-i", mp4, "-map", "0:a:0", "-c", "copy", "-y", only_a1]
        )
        self.assertEqual(r1.returncode, 0, r1.stderr)
        p1 = oracle.ffprobe_json(only_a1)
        self.assertEqual(p1["format"]["nb_streams"], 1)
        oracle.assert_ffprobe_stream(p1, 0, codec_type="audio", codec_name="aac")
        oracle.oracle_ffmpeg_decode_check(only_a1)

        only_a2 = self.tmp / "only_a2.mp4"
        r2 = run_cmd(
            [ffmpeg_shim, "-i", mp4, "-map", "-0:v", "-c", "copy", "-y", only_a2]
        )
        self.assertEqual(r2.returncode, 0, r2.stderr)
        p2 = oracle.ffprobe_json(only_a2)
        self.assertEqual(p2["format"]["nb_streams"], 1)
        oracle.assert_ffprobe_stream(p2, 0, codec_type="audio", codec_name="aac")
        oracle.oracle_ffmpeg_decode_check(only_a2)

    @e2e_test(feature="F3", tier=2, milestone="baseline")
    def test_t2_ffprobe_select_streams_matching_and_empty(self) -> None:
        _, ffprobe_shim = self._install_shims()
        mp4 = fixtures.h264_aac_mp4()

        pv = oracle.ffprobe_json(
            mp4, ffprobe_bin=ffprobe_shim, show_format=False, select_streams="v"
        )
        self.assertEqual(len(pv["streams"]), 1)
        self.assertEqual(pv["streams"][0]["index"], 0)
        self.assertEqual(pv["streams"][0]["codec_type"], "video")

        pa = oracle.ffprobe_json(
            mp4, ffprobe_bin=ffprobe_shim, show_format=False, select_streams="a"
        )
        self.assertEqual(len(pa["streams"]), 1)
        self.assertEqual(pa["streams"][0]["index"], 1)
        self.assertEqual(pa["streams"][0]["codec_type"], "audio")

        pw = oracle.ffprobe_json(
            fixtures.chord_48k_wav(),
            ffprobe_bin=ffprobe_shim,
            show_format=False,
            select_streams="v",
        )
        self.assertEqual(pw["streams"], [])

    @e2e_test(feature="F3", tier=2, milestone="baseline")
    def test_t2_install_shim_errors_and_invalid_ffmpeg_flag_values(self) -> None:
        ffmpeg_shim, _ = self._install_shims()

        assert_cli_error(run_wedeo(["install-shim"]), "Usage: wedeo install-shim")

        reg_file = self.tmp / "regular_file"
        reg_file.write_text("not a dir", encoding="utf-8")
        assert_cli_error(
            run_wedeo(["install-shim", reg_file]),
            "[install-shim]",
        )

        assert_cli_error(
            run_cmd(
                [
                    ffmpeg_shim,
                    "-i",
                    fixtures.vp9_smpte_mkv(),
                    "-frames:v",
                    "notanumber",
                    "-y",
                    self.tmp / "bad_frames.mkv",
                ]
            ),
            "invalid frame count",
        )
        assert_cli_error(
            run_cmd(
                [
                    ffmpeg_shim,
                    "-i",
                    fixtures.vp9_smpte_mkv(),
                    "-pix_fmt",
                    "bogus_fmt",
                    "-y",
                    self.tmp / "bad_pixfmt.mkv",
                ]
            ),
            "unknown pixel format",
        )
