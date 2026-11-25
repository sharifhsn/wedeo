"""Compare default stream selection on a file with two distinct audio tracks."""
import json
import subprocess
import tempfile
from pathlib import Path


def run(*args):
    return subprocess.check_output(args, text=True)


def streams(path):
    data = json.loads(run("ffprobe", "-v", "error", "-show_packets", "-show_streams",
                          "-show_data_hash", "sha256", "-show_entries",
                          "stream=index,codec_name:packet=stream_index,data_hash",
                          "-of", "json", path))
    return [(s["codec_name"], [p["data_hash"] for p in data["packets"]
                               if p["stream_index"] == s["index"]]) for s in data["streams"]]


def main():
    sample = Path(__file__).resolve().parents[1] / "web/samples/h264_aac_320x240.mp4"
    print(run("ffmpeg", "-version").splitlines()[0], run("uvx", "wedeo", "--version").strip(), sep="\n")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input.mp4"
        run("ffmpeg", "-v", "error", "-i", sample, "-f", "lavfi", "-i",
            "sine=frequency=880:duration=2:sample_rate=48000", "-map", "0:v:0",
            "-map", "0:a:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", src)
        original = streams(src)
        assert [s[0] for s in original] == ["h264", "aac", "aac"]
        assert all(packets for _, packets in original) and original[1] != original[2]
        for name, command, expected in [
            ("ffmpeg-default", ["ffmpeg", "-v", "error", "-i", src, "-c", "copy"], original[:2]),
            ("ffmpeg-map-all", ["ffmpeg", "-v", "error", "-i", src, "-map", "0", "-c", "copy"], original),
            ("wedeo-default", ["uvx", "wedeo", "ffmpeg", "-i", src, "-c", "copy"], original),
        ]:
            output = Path(tmp) / (name + ".mkv")
            run(*command, output)
            actual = streams(output)
            assert actual == expected, f"{name}: unexpected streams or packet payloads"
            print(name, [(codec, len(packets)) for codec, packets in actual])


if __name__ == "__main__":
    main()
