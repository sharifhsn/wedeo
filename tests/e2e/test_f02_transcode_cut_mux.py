"""E2E tests for F2: wedeo transcode, cut, and mux."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lib import fixtures, oracle
from lib.harness import assert_cli_error, e2e_test, run_wedeo


class TestF02TranscodeCutMux(unittest.TestCase):
    """Tier 1 and Tier 2 E2E tests for F2 transcode, cut, and mux."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_f02_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    @e2e_test(feature="F2", tier=1, milestone="baseline")
    def test_t1_transcode_h264_aac_mp4_to_av1_mp4_and_mkv(self) -> None:
        src = fixtures.h264_aac_mp4()

        out_mp4 = self.tmp / "out.mp4"
        res_mp4 = run_wedeo(
            [
                "transcode",
                "-i",
                src,
                "--video",
                "av1",
                "--audio",
                "pcm_s16le",
                "--frames",
                "4",
                "-o",
                out_mp4,
                "--json",
            ]
        )
        self.assertEqual(res_mp4.returncode, 0, res_mp4.stderr)
        j_mp4 = json.loads(res_mp4.stdout)
        self.assertEqual(j_mp4["format"], "mp4")
        self.assertEqual(j_mp4["tracks"], 2)
        self.assertIs(j_mp4["health"]["is_clean"], True)
        self.assertTrue(any("fltp -> s16" in n for n in j_mp4["conversion_notes"]))
        p_mp4 = oracle.ffprobe_json(out_mp4)
        oracle.assert_ffprobe_stream(
            p_mp4, 0, codec_name="av1", codec_type="video", width=320, height=240
        )
        oracle.assert_ffprobe_stream(
            p_mp4, 1, codec_name="pcm_s16le", codec_type="audio", sample_rate=48000
        )
        oracle.assert_wedeo_inspect_clean(run_wedeo, out_mp4, verify=True)

        out_mkv = self.tmp / "out.mkv"
        res_mkv = run_wedeo(
            [
                "transcode",
                "-i",
                src,
                "--video",
                "av1",
                "--audio",
                "pcm_s16le",
                "--frames",
                "4",
                "-o",
                out_mkv,
                "--json",
            ]
        )
        self.assertEqual(res_mkv.returncode, 0, res_mkv.stderr)
        j_mkv = json.loads(res_mkv.stdout)
        self.assertEqual(j_mkv["format"], "matroska")
        p_mkv = oracle.ffprobe_json(out_mkv)
        oracle.assert_ffprobe_stream(
            p_mkv, 0, codec_name="av1", codec_type="video", width=320, height=240
        )
        oracle.assert_ffprobe_stream(
            p_mkv, 1, codec_name="pcm_s16le", codec_type="audio", sample_rate=48000
        )

        out_copy = self.tmp / "out_copy.mp4"
        res_copy = run_wedeo(
            [
                "transcode",
                "-i",
                src,
                "--video",
                "av1",
                "--audio",
                "copy",
                "--frames",
                "4",
                "--strict",
                "-o",
                out_copy,
                "--json",
            ]
        )
        self.assertEqual(res_copy.returncode, 0, res_copy.stderr)
        j_copy = json.loads(res_copy.stdout)
        self.assertEqual(j_copy["conversion_notes"], [])
        p_copy = oracle.ffprobe_json(out_copy)
        oracle.assert_ffprobe_stream(p_copy, 0, codec_name="av1", codec_type="video")
        oracle.assert_ffprobe_stream(p_copy, 1, codec_name="aac", codec_type="audio")

    @e2e_test(feature="F2", tier=1, milestone="baseline")
    def test_t1_transcode_vp9_ivf_to_av1_ivf_and_scale_pixfmt(self) -> None:
        src = fixtures.vp9_testsrc_128_ivf()

        out_ivf = self.tmp / "out.ivf"
        res = run_wedeo(
            ["transcode", "-i", src, "--video", "av1", "-o", out_ivf, "--json"]
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        j = json.loads(res.stdout)
        self.assertEqual(j["format"], "ivf")
        self.assertEqual(j["tracks"], 1)
        self.assertEqual(j["conversion_notes"], [])
        p = oracle.ffprobe_json(out_ivf)
        oracle.assert_ffprobe_stream(
            p, 0, codec_name="av1", width=128, height=128, pix_fmt="yuv420p"
        )
        oracle.assert_video_psnr(src, out_ivf, min_psnr_db=35.0, pix_fmt="yuv420p")

        out_422 = self.tmp / "out_422.ivf"
        res_422 = run_wedeo(
            [
                "transcode",
                "-i",
                src,
                "--video",
                "av1",
                "--size",
                "64x64",
                "--pix-fmt",
                "yuv422p",
                "-o",
                out_422,
                "--json",
            ]
        )
        self.assertEqual(res_422.returncode, 0, res_422.stderr)
        j_422 = json.loads(res_422.stdout)
        self.assertTrue(
            any("downscale (128x128 -> 64x64)" in n for n in j_422["conversion_notes"])
        )
        oracle.assert_ffprobe_stream(
            oracle.ffprobe_json(out_422),
            0,
            codec_name="av1",
            width=64,
            height=64,
            pix_fmt="yuv422p",
        )

        out_gray = self.tmp / "out_gray.ivf"
        res_gray = run_wedeo(
            [
                "transcode",
                "-i",
                src,
                "--video",
                "av1",
                "--pix-fmt",
                "gray8",
                "-o",
                out_gray,
                "--json",
            ]
        )
        self.assertEqual(res_gray.returncode, 0, res_gray.stderr)
        oracle.assert_ffprobe_stream(
            oracle.ffprobe_json(out_gray),
            0,
            codec_name="av1",
            width=128,
            height=128,
            pix_fmt="gray",
        )

    @e2e_test(feature="F2", tier=1, milestone="baseline")
    def test_t1_transcode_wav_resample_and_channel_conversion(self) -> None:
        src = fixtures.chord_48k_wav()
        out_wav = self.tmp / "chord_16k_stereo.wav"

        res = run_wedeo(
            [
                "transcode",
                "-i",
                src,
                "--sample-rate",
                "16000",
                "--channels",
                "2",
                "-o",
                out_wav,
                "--json",
            ]
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        j = json.loads(res.stdout)
        self.assertEqual(j["format"], "wav")
        self.assertIs(j["health"]["is_clean"], True)
        self.assertTrue(
            any("48000Hz -> 16000Hz" in n for n in j["conversion_notes"])
        )

        probe = oracle.ffprobe_json(out_wav)
        oracle.assert_ffprobe_stream(
            probe, 0, codec_name="pcm_s16le", sample_rate=16000, channels=2
        )
        self.assertEqual(probe["format"]["duration"], "1.500000")
        oracle.assert_wedeo_inspect_clean(run_wedeo, out_wav, verify=True)

        ref_pcm = oracle.decode_pcm_s16le(src, sample_rate=16000, channels=2)
        got_pcm = oracle.decode_pcm_s16le(out_wav)
        oracle.assert_audio_close(
            ref_pcm, got_pcm, max_mae=2.0, max_sample_diff=4, allow_len_delta=0
        )

    @e2e_test(feature="F2", tier=1, milestone="baseline")
    def test_t1_cut_modes_keyframe_smart_reencode(self) -> None:
        av1_src = fixtures.synth_multi_gop_av1_mkv()
        cut_smart = self.tmp / "cut_smart.mkv"
        res_smart = run_wedeo(
            [
                "cut",
                "-i",
                av1_src,
                "--from",
                "0.15",
                "--to",
                "0.75",
                "--mode",
                "smart",
                "-o",
                cut_smart,
                "--json",
            ]
        )
        self.assertEqual(res_smart.returncode, 0, res_smart.stderr)
        j_smart = json.loads(res_smart.stdout)
        self.assertEqual(j_smart["operation"], "cut(smart)")
        segs = " ".join(j_smart["smart_cut_segments"])
        for label in ("ReencodeHead", "CopyInterior", "ReencodeTail"):
            self.assertIn(label, segs)
        insp = oracle.assert_wedeo_inspect_clean(run_wedeo, cut_smart, verify=True)
        self.assertEqual(insp["decoded_units"], 6)

        h264_src = fixtures.synth_multi_gop_h264_mp4()
        cut_kf = self.tmp / "cut_kf.mp4"
        res_kf = run_wedeo(
            [
                "cut",
                "-i",
                h264_src,
                "--from",
                "0.5",
                "--to",
                "1.5",
                "--mode",
                "keyframe",
                "-o",
                cut_kf,
                "--json",
            ]
        )
        self.assertEqual(res_kf.returncode, 0, res_kf.stderr)
        j_kf = json.loads(res_kf.stdout)
        self.assertEqual(j_kf["operation"], "cut(keyframe)")
        self.assertEqual(j_kf["tracks"], 2)
        p_kf = oracle.ffprobe_json(cut_kf)
        oracle.assert_ffprobe_stream(p_kf, 0, codec_name="h264")
        oracle.assert_ffprobe_stream(p_kf, 1, codec_name="aac")
        oracle.assert_wedeo_inspect_clean(run_wedeo, cut_kf, verify=True)

        cut_re = self.tmp / "cut_re.mkv"
        res_re = run_wedeo(
            [
                "cut",
                "-i",
                h264_src,
                "--from",
                "0.5",
                "--to",
                "1.5",
                "--mode",
                "reencode",
                "--video",
                "av1",
                "--audio",
                "pcm_s16le",
                "-o",
                cut_re,
                "--json",
            ]
        )
        self.assertEqual(res_re.returncode, 0, res_re.stderr)
        j_re = json.loads(res_re.stdout)
        self.assertEqual(j_re["operation"], "cut(reencode)")
        self.assertEqual(j_re["format"], "matroska")
        oracle.assert_wedeo_inspect_clean(run_wedeo, cut_re, verify=True)

    @e2e_test(feature="F2", tier=1, milestone="baseline")
    def test_t1_mux_video_audio_tags_and_preserve_all(self) -> None:
        mux_mkv = self.tmp / "muxed.mkv"
        res_mkv = run_wedeo(
            [
                "mux",
                "--video",
                fixtures.vp9_red_ivf(),
                "--audio",
                fixtures.chord_48k_wav(),
                "--tag",
                "title=Probe Title",
                "--tag",
                "artist=Wedeo E2E",
                "-o",
                mux_mkv,
                "--json",
            ]
        )
        self.assertEqual(res_mkv.returncode, 0, res_mkv.stderr)
        self.assertEqual(json.loads(res_mkv.stdout)["tracks"], 2)
        oracle.assert_ffprobe_tags(
            oracle.ffprobe_json(mux_mkv),
            {"title": "Probe Title", "artist": "Wedeo E2E"},
        )
        insp_mkv = oracle.assert_wedeo_inspect_clean(run_wedeo, mux_mkv, verify=True)
        mkv_meta = {(m["key"], m["value"]) for m in insp_mkv["metadata"]}
        self.assertIn(("title", "Probe Title"), mkv_meta)
        self.assertIn(("artist", "Wedeo E2E"), mkv_meta)

        mux_mp4 = self.tmp / "muxed.mp4"
        res_mp4 = run_wedeo(
            [
                "mux",
                "--video",
                fixtures.h264_aac_mp4(),
                "--audio",
                fixtures.h264_aac_mp4(),
                "--tag",
                "title=MP4 Title",
                "--tag",
                "comment=MP4 Comment",
                "-o",
                mux_mp4,
                "--json",
            ]
        )
        self.assertEqual(res_mp4.returncode, 0, res_mp4.stderr)
        self.assertEqual(json.loads(res_mp4.stdout)["tracks"], 2)
        oracle.assert_ffprobe_tags(
            oracle.ffprobe_json(mux_mp4),
            {"title": "MP4 Title", "comment": "MP4 Comment"},
        )
        insp_mp4 = oracle.assert_wedeo_inspect_clean(run_wedeo, mux_mp4, verify=True)
        mp4_meta = {(m["key"], m["value"]) for m in insp_mp4["metadata"]}
        self.assertIn(("title", "MP4 Title"), mp4_meta)
        self.assertIn(("comment", "MP4 Comment"), mp4_meta)

        pres = self.tmp / "pres.mkv"
        nopres = self.tmp / "nopres.mkv"
        for flag, out in (("--preserve-all", pres), ("--no-preserve-all", nopres)):
            r = run_wedeo(
                [
                    "transcode",
                    "-i",
                    fixtures.vp9_smpte_mkv(),
                    "--video",
                    "av1",
                    "--frames",
                    "2",
                    flag,
                    "-o",
                    out,
                ]
            )
            self.assertEqual(r.returncode, 0, r.stderr)
        insp_pres = oracle.assert_wedeo_inspect_clean(run_wedeo, pres, verify=False)
        insp_nopres = oracle.assert_wedeo_inspect_clean(run_wedeo, nopres, verify=False)
        self.assertTrue(len(insp_pres["tracks"][0]["metadata"]) > 0)
        self.assertEqual(insp_nopres["tracks"][0]["metadata"], [])

    @e2e_test(feature="F2", tier=2, milestone="baseline")
    def test_t2_strict_rejects_implicit_lossy_conversions(self) -> None:
        cases = [
            [
                "transcode",
                "-i",
                fixtures.chord_48k_wav(),
                "--sample-rate",
                "16000",
                "--strict",
                "-o",
                self.tmp / "strict_aud_ds.wav",
            ],
            [
                "transcode",
                "-i",
                fixtures.h264_aac_mp4(),
                "--video",
                "av1",
                "--audio",
                "pcm_s16le",
                "--frames",
                "2",
                "--strict",
                "-o",
                self.tmp / "strict_aud_q.mp4",
            ],
            [
                "transcode",
                "-i",
                fixtures.vp9_testsrc_128_ivf(),
                "--video",
                "av1",
                "--size",
                "64x64",
                "--strict",
                "-o",
                self.tmp / "strict_vid_ds.ivf",
            ],
            [
                "transcode",
                "-i",
                fixtures.vp9_testsrc_128_ivf(),
                "--video",
                "av1",
                "--pix-fmt",
                "gray8",
                "--strict",
                "-o",
                self.tmp / "strict_vid_gray.ivf",
            ],
        ]
        for args in cases:
            assert_cli_error(
                run_wedeo(args),
                "strict mode rejected implicit lossy",
            )

    @e2e_test(feature="F2", tier=2, milestone="baseline")
    def test_t2_missing_encoder_defaults_for_mp4_webm_ivf(self) -> None:
        assert_cli_error(
            run_wedeo(
                [
                    "transcode",
                    "-i",
                    fixtures.vp9_red_ivf(),
                    "-o",
                    self.tmp / "no_h264.mp4",
                ]
            ),
            "no encoder for 'h264'",
        )
        for ext in ("webm", "ivf"):
            assert_cli_error(
                run_wedeo(
                    [
                        "transcode",
                        "-i",
                        fixtures.vp9_red_ivf(),
                        "-o",
                        self.tmp / f"no_vp9.{ext}",
                    ]
                ),
                "no encoder for 'vp9'",
            )

    @e2e_test(feature="F2", tier=2, milestone="baseline")
    def test_t2_existing_output_rejected_without_y_and_overwritten_with_y(self) -> None:
        sentinel = b"SENTINEL_DO_NOT_TOUCH"
        existing_wav = self.tmp / "existing.wav"
        existing_mkv = self.tmp / "existing.mkv"
        existing_wav.write_bytes(sentinel)
        existing_mkv.write_bytes(sentinel)

        assert_cli_error(
            run_wedeo(
                ["transcode", "-i", fixtures.chord_48k_wav(), "-o", existing_wav]
            ),
            "already exists (pass -y / --overwrite)",
        )
        self.assertEqual(existing_wav.read_bytes(), sentinel)

        assert_cli_error(
            run_wedeo(
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
                    "-o",
                    existing_mkv,
                ]
            ),
            "already exists (pass -y / --overwrite)",
        )
        self.assertEqual(existing_mkv.read_bytes(), sentinel)

        assert_cli_error(
            run_wedeo(
                [
                    "mux",
                    "--video",
                    fixtures.vp9_red_ivf(),
                    "--audio",
                    fixtures.chord_48k_wav(),
                    "-o",
                    existing_mkv,
                ]
            ),
            "already exists (pass -y / --overwrite)",
        )
        self.assertEqual(existing_mkv.read_bytes(), sentinel)

        over_res = run_wedeo(
            [
                "transcode",
                "-i",
                fixtures.chord_48k_wav(),
                "-y",
                "-o",
                existing_wav,
                "--json",
            ]
        )
        self.assertEqual(over_res.returncode, 0, over_res.stderr)
        oracle.assert_ffprobe_stream(
            oracle.ffprobe_json(existing_wav), 0, codec_name="pcm_s16le"
        )

    @e2e_test(feature="F2", tier=2, milestone="baseline")
    def test_t2_cut_invalid_bounds_past_eof_and_strict_smart_cut(self) -> None:
        mp4 = fixtures.h264_aac_mp4()
        for from_s, to_s in (("2.0", "1.0"), ("1.0", "1.0"), ("-1.0", "1.0")):
            assert_cli_error(
                run_wedeo(
                    [
                        "cut",
                        "-i",
                        mp4,
                        "--from",
                        from_s,
                        "--to",
                        to_s,
                        "-o",
                        self.tmp / f"bad_{from_s}_{to_s}.mp4",
                    ]
                ),
                "cut range requires 0 <= from < to",
            )

        past_eof = self.tmp / "past_eof.mp4"
        res_eof = run_wedeo(
            [
                "cut",
                "-i",
                mp4,
                "--from",
                "10.0",
                "--to",
                "12.0",
                "-o",
                past_eof,
                "--json",
            ]
        )
        self.assertEqual(res_eof.returncode, 0, res_eof.stderr)
        self.assertEqual(json.loads(res_eof.stdout)["smart_cut_segments"], [])
        insp_eof = json.loads(
            run_wedeo(["inspect", "--level", "packets", "--json", past_eof]).stdout
        )
        self.assertEqual(insp_eof["packets"], [])
        self.assertIsNone(insp_eof["duration_sec"])

        h264_src = fixtures.synth_multi_gop_h264_mp4()
        assert_cli_error(
            run_wedeo(
                [
                    "cut",
                    "-i",
                    h264_src,
                    "--from",
                    "0.5",
                    "--to",
                    "1.5",
                    "--mode",
                    "smart",
                    "--strict",
                    "-o",
                    self.tmp / "strict_smart.mp4",
                ]
            ),
            "strict smart cut requires a matching-codec encoder for head/tail GOPs",
        )

    @e2e_test(feature="F2", tier=2, milestone="baseline")
    def test_t2_invalid_cli_arguments_size_pixfmt_sample_rate_mux(self) -> None:
        ivf = fixtures.vp9_red_ivf()
        assert_cli_error(
            run_wedeo(
                ["transcode", "-i", ivf, "--size", "0x64", "-o", self.tmp / "a.ivf"]
            ),
            "non-zero",
        )
        assert_cli_error(
            run_wedeo(
                ["transcode", "-i", ivf, "--size", "abc", "-o", self.tmp / "b.ivf"]
            ),
            "expected WxH",
        )
        assert_cli_error(
            run_wedeo(
                [
                    "transcode",
                    "-i",
                    ivf,
                    "--pix-fmt",
                    "nope",
                    "-o",
                    self.tmp / "c.ivf",
                ]
            ),
            "unknown pixel layout",
        )
        assert_cli_error(
            run_wedeo(
                [
                    "transcode",
                    "-i",
                    fixtures.chord_48k_wav(),
                    "--sample-rate",
                    "bad",
                    "-o",
                    self.tmp / "d.wav",
                ]
            ),
            "invalid --sample-rate",
        )
        assert_cli_error(
            run_wedeo(
                [
                    "mux",
                    "--video",
                    ivf,
                    "--tag",
                    "badtag",
                    "-o",
                    self.tmp / "e.mkv",
                ]
            ),
            "expected KEY=VALUE",
        )
        assert_cli_error(
            run_wedeo(["mux", "-o", self.tmp / "f.mkv"]),
            "at least one input",
        )
