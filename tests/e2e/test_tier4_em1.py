"""Tier 4 real-world baseline workflows for EM1 (F1, F2, F3)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lib import fixtures, oracle
from lib.harness import augmented_env, e2e_test, run_cmd, run_wedeo


class TestTier4Em1Workflows(unittest.TestCase):
    """Real-world end-to-end application scenarios for baseline F1+F2+F3."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_t4_em1_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    @e2e_test(
        feature="F2", tier=4, milestone="baseline", features=("F1", "F2", "F3")
    )
    def test_scenario1_inspect_cut_transcode_mux_oracle_workflow(self) -> None:
        src_video = fixtures.vp9_smpte_mkv()
        src_audio = fixtures.chord_48k_wav()

        # Step 1: Inspect source video
        insp_src = oracle.assert_wedeo_inspect_clean(
            run_wedeo, src_video, verify=True
        )
        self.assertEqual(insp_src["decoded_units"], 48)
        self.assertEqual(insp_src["tracks"][0]["codec"], "vp9")
        self.assertEqual(insp_src["tracks"][0]["width"], 320)
        self.assertEqual(insp_src["tracks"][0]["height"], 240)

        # Step 2: Keyframe cut first 1.0s
        cut_mkv = self.tmp / "cut.mkv"
        r_cut = run_wedeo(
            [
                "cut",
                "-i",
                src_video,
                "--from",
                "0",
                "--to",
                "1.0",
                "--mode",
                "keyframe",
                "-y",
                "-o",
                cut_mkv,
                "--json",
            ]
        )
        self.assertEqual(r_cut.returncode, 0, r_cut.stderr)
        self.assertIs(json.loads(r_cut.stdout)["health"]["is_clean"], True)

        # Step 3: Transcode bounded clip to AV1 160x120
        tr_mkv = self.tmp / "transcoded.mkv"
        r_tr = run_wedeo(
            [
                "transcode",
                "-i",
                cut_mkv,
                "--video",
                "av1",
                "--size",
                "160x120",
                "--frames",
                "6",
                "-y",
                "-o",
                tr_mkv,
                "--json",
            ]
        )
        self.assertEqual(r_tr.returncode, 0, r_tr.stderr)
        j_tr = json.loads(r_tr.stdout)
        self.assertGreater(j_tr["units_processed"], 0)
        self.assertIs(j_tr["health"]["is_clean"], True)

        # Step 4: Audio prep + tagged mux
        audio_24k = self.tmp / "audio_24k.wav"
        r_aud = run_wedeo(
            [
                "transcode",
                "-i",
                src_audio,
                "--sample-rate",
                "24000",
                "--channels",
                "2",
                "-y",
                "-o",
                audio_24k,
            ]
        )
        self.assertEqual(r_aud.returncode, 0, r_aud.stderr)

        final_mkv = self.tmp / "final.mkv"
        r_mux = run_wedeo(
            [
                "mux",
                "--video",
                tr_mkv,
                "--audio",
                audio_24k,
                "--tag",
                "title=Social Clip",
                "-y",
                "-o",
                final_mkv,
                "--json",
            ]
        )
        self.assertEqual(r_mux.returncode, 0, r_mux.stderr)

        # Step 5: Verify via wedeo inspect --verify and wedeo/oracle ffprobe parity
        insp_final = oracle.assert_wedeo_inspect_clean(
            run_wedeo, final_mkv, verify=True
        )
        self.assertEqual(insp_final["tracks"][0]["codec"], "av1")
        self.assertEqual(insp_final["tracks"][0]["width"], 160)
        self.assertEqual(insp_final["tracks"][0]["height"], 120)
        self.assertEqual(insp_final["tracks"][1]["codec"], "pcm_s16le")
        self.assertEqual(insp_final["tracks"][1]["sample_rate"], 24000)
        self.assertEqual(insp_final["tracks"][1]["channels"], 2)

        w_ffp_res = run_wedeo(
            [
                "ffprobe",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                final_mkv,
            ]
        )
        self.assertEqual(w_ffp_res.returncode, 0, w_ffp_res.stderr)
        w_probe = json.loads(w_ffp_res.stdout)
        o_probe = oracle.ffprobe_json(final_mkv)
        oracle.assert_ffprobe_parity(
            w_probe, o_probe, expected_tags={"title": "Social Clip"}
        )

    @e2e_test(
        feature="F3", tier=4, milestone="baseline", features=("F1", "F2", "F3")
    )
    def test_scenario2_install_shim_ffmetadata1_chapters_map_remux_pipeline(
        self,
    ) -> None:
        # Step 1: Install shims and prepend to PATH to simulate drop-in external caller
        shim_dir = self.tmp / "bin"
        self.assertEqual(run_wedeo(["install-shim", shim_dir]).returncode, 0)
        env = augmented_env()
        env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"

        # Step 2: Probe sources via bare `ffprobe` on PATH
        for src in (fixtures.vp9_smpte_mkv(), fixtures.chord_48k_wav()):
            pr = run_cmd(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    src,
                ],
                env=env,
            )
            self.assertEqual(pr.returncode, 0, pr.stderr)
            self.assertGreaterEqual(json.loads(pr.stdout)["format"]["nb_streams"], 1)

        # Step 3: Shim trim & transcode via bare `ffmpeg` on PATH
        v_clip = self.tmp / "v_clip.mkv"
        r_v = run_cmd(
            [
                "ffmpeg",
                "-i",
                fixtures.vp9_smpte_mkv(),
                "-frames:v",
                "6",
                "-c:v",
                "av1",
                "-s",
                "160x120",
                "-y",
                v_clip,
            ],
            env=env,
        )
        self.assertEqual(r_v.returncode, 0, r_v.stderr)

        a_clip = self.tmp / "a_clip.wav"
        r_a = run_cmd(
            [
                "ffmpeg",
                "-i",
                fixtures.chord_48k_wav(),
                "-ss",
                "0",
                "-t",
                "0.5",
                "-ar",
                "22050",
                "-ac",
                "2",
                "-y",
                a_clip,
            ],
            env=env,
        )
        self.assertEqual(r_a.returncode, 0, r_a.stderr)

        # Step 4: ;FFMETADATA1 + -map remux
        ffmeta = fixtures.synth_ffmetadata1(
            self.tmp / "chapters.ffmeta",
            tags={"title": "Drop-In Episode", "artist": "Wedeo"},
            chapters=[
                (0, 125, "Intro"),
                (125, 250, "Body"),
            ],
        )
        episode = self.tmp / "episode.mkv"
        r_ep = run_cmd(
            [
                "ffmpeg",
                "-i",
                v_clip,
                "-i",
                a_clip,
                "-i",
                ffmeta,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-metadata",
                "comment=drop-in",
                "-y",
                episode,
            ],
            env=env,
        )
        self.assertEqual(r_ep.returncode, 0, r_ep.stderr)

        # Step 5: Verify via shim ffprobe vs oracle ffprobe + wedeo inspect --verify
        w_ep_res = run_cmd(
            [
                "ffprobe",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-show_chapters",
                episode,
            ],
            env=env,
        )
        self.assertEqual(w_ep_res.returncode, 0, w_ep_res.stderr)
        w_ep = json.loads(w_ep_res.stdout)
        o_ep = oracle.ffprobe_json(episode, show_chapters=True)

        oracle.assert_ffprobe_parity(
            w_ep,
            o_ep,
            check_chapters=True,
            expected_tags={
                "title": "Drop-In Episode",
                "artist": "Wedeo",
                "comment": "drop-in",
            },
        )
        oracle.assert_ffprobe_stream(
            o_ep, 0, codec_name="av1", width=160, height=120
        )
        oracle.assert_ffprobe_stream(
            o_ep, 1, codec_name="pcm_s16le", sample_rate=22050, channels=2
        )
        self.assertEqual(len(o_ep["chapters"]), 2)
        self.assertEqual(o_ep["chapters"][0]["tags"]["title"], "Intro")
        self.assertEqual(o_ep["chapters"][1]["tags"]["title"], "Body")

        oracle.assert_wedeo_inspect_clean(run_wedeo, episode, verify=True)
