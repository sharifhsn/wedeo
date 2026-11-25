// Run after web/build.sh: node tests/e2e/check_web_clip.mjs
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";

const file = "web/samples/multitrack.mp4";
const { instance } = await WebAssembly.instantiate(readFileSync("web/wedeo.wasm"));
const w = instance.exports;
function input(bytes) {
  const ptr = w.wedeo_alloc(bytes.length);
  new Uint8Array(w.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}
function bytes(ptr, len) { return Buffer.from(new Uint8Array(w.memory.buffer, ptr, len)); }
function check(code) {
  if (code !== 0) throw Error(bytes(w.wedeo_error_ptr(), w.wedeo_error_len()).toString());
}
const json = () => JSON.parse(bytes(w.wedeo_json_ptr(), w.wedeo_json_len()));
const args = [...input(readFileSync(file)), ...input(Buffer.from("multitrack.mp4"))];
check(w.wedeo_inspect(...args));
const tracks = json().tracks.filter(t => t.media_type === "audio");
assert.equal(tracks.length, 2);
for (const [index, track] of tracks.entries()) {
  check(w.wedeo_audition(...args, track.id));
  const meta = json();
  const samples = new Float32Array(w.memory.buffer, w.wedeo_bytes_ptr(), meta.frames_per_channel);
  let crossings = 0;
  for (let i = meta.sample_rate; i < 2 * meta.sample_rate; i++) {
    if (samples[i] < 0 && samples[i + 1] >= 0) crossings++;
  }
  assert.ok(Math.abs(crossings - [330, 880][index]) < 3, "audition frequency: " + crossings);
}
check(w.wedeo_clip_marks(...args));
const marks = json();
const start = marks[1], end = marks[6];
assert.ok(Math.abs(end - start - 10) < 0.001);
check(w.wedeo_clip(...args, start, end));
const dir = mkdtempSync(join(tmpdir(), "wedeo-clip-"));
function packets(path) {
  return JSON.parse(execFileSync("ffprobe", ["-v","error","-show_packets","-show_data_hash","sha256",
    "-show_entries","packet=stream_index,pts_time,data_hash","-of","json",path], {maxBuffer:8*1024*1024})).packets;
}
// Compare decoded pictures as well as compressed payloads: open GOPs can
// depend on pictures across a nominal keyframe even when packet hashes match.
function pictureHashes(path) {
  const text = execFileSync("ffmpeg", ["-nostdin", "-v", "error", "-xerror", "-i", path,
    "-map", "0:v:0", "-fps_mode", "passthrough", "-threads", "1", "-f", "framemd5", "-"], {encoding:"utf8"});
  return text.trim().split("\n").filter(line => !line.startsWith("#")).map(line => line.split(",").at(-1).trim());
}
function verifyReorderedVideo(openGop) {
  const source = join(dir, `reordered-${openGop}.mp4`);
  const output = join(dir, `reordered-${openGop}.mkv`);
  execFileSync("ffmpeg", ["-nostdin", "-v","error","-i", file,"-map","0","-c:a","copy",
    "-c:v","libx264","-crf","17","-g","24","-bf","3","-x264-params",
    `open-gop=${openGop}:scenecut=0:b-adapt=0`,"-threads","1",source]);
  const params = [...input(readFileSync(source)), ...input(Buffer.from("source.mp4"))];
  check(w.wedeo_clip_marks(...params));
  const bounds = json();
  let expected = pictureHashes(source);
  if (openGop) {
    assert.equal(bounds.length, 2, "only the initial IDR and end are independent");
    assert.notEqual(w.wedeo_clip(...params, 2, 12), 0, "reject unsafe boundaries from direct callers");
    check(w.wedeo_clip(...params, bounds[0], bounds[1]));
  } else {
    check(w.wedeo_clip(...params, bounds[1], bounds[6]));
    expected = expected.slice(24, 144);
  }
  writeFileSync(output, bytes(w.wedeo_bytes_ptr(), w.wedeo_bytes_len()));
  assert.deepEqual(pictureHashes(output), expected);
  w.wedeo_free(params[0], params[1]);
  w.wedeo_free(params[2], params[3]);
}
function verifyIntervalAudition() {
  const source = join(dir, "two-minutes.mp4");
  execFileSync("ffmpeg", ["-nostdin","-v","error","-f","lavfi","-i","testsrc2=size=128x72:rate=12:duration=120",
    "-f","lavfi","-i","sine=frequency=330:sample_rate=48000:duration=120",
    "-f","lavfi","-i","sine=frequency=880:sample_rate=48000:duration=120",
    "-map","0:v","-map","1:a","-map","2:a","-c:v","libx264","-preset","ultrafast","-threads","1",
    "-g","24","-sc_threshold","0","-bf","0","-c:a","aac","-b:a","48k","-movflags","+faststart",source]);
  const params = [...input(readFileSync(source)), ...input(Buffer.from("source.mp4"))];
  check(w.wedeo_inspect(...params));
  const audio = json().tracks.filter(t => t.media_type === "audio");
  assert.equal(audio.length, 2);
  for (const [index, track] of audio.entries()) {
    check(w.wedeo_audition_window(...params, track.id, 60, 70));
    const meta = json();
    assert.equal(meta.start_sec, 60);
    assert.equal(meta.frames_per_channel, 480000);
    assert.equal(w.wedeo_bytes_len(), 1920000);
    assert.ok(meta.decoded_packets > 0 && meta.decoded_packets < 530, "bounded AAC preroll");
    const ref = execFileSync("ffmpeg", ["-nostdin","-v","error","-i",source,"-map",`0:a:${index}`,
      "-ss","60","-t","10","-c:a","pcm_f32le","-f","f32le","-"], {maxBuffer:4*1024*1024});
    const pcm = new Float32Array(w.memory.buffer, w.wedeo_bytes_ptr(), meta.frames_per_channel);
    assert.equal(ref.length, pcm.byteLength);
    let error = 0, power = 0, crossings = 0;
    for (let i = 0; i < pcm.length; i++) {
      const expected = ref.readFloatLE(i*4);
      error += (expected-pcm[i])**2;
      power += expected**2;
      if (i && pcm[i-1] < 0 && pcm[i] >= 0) crossings++;
    }
    assert.ok(Math.sqrt(error/power) < 0.002, "PCM aligns sample-for-sample with independent decode");
    assert.ok(Math.abs(crossings/10 - [330,880][index]) < 1);
    w.wedeo_release_output_bytes();
    assert.equal(w.wedeo_bytes_len(), 0);
  }
  check(w.wedeo_clip_marks(...params));
  const marks = json();
  check(w.wedeo_clip(...params, marks.find(t => Math.abs(t-60)<1e-6), marks.find(t => Math.abs(t-70)<1e-6)));
  const clipped = join(dir, "minute-clip.mkv");
  writeFileSync(clipped, bytes(w.wedeo_bytes_ptr(), w.wedeo_bytes_len()));
  const original = packets(source), exported = packets(clipped);
  for (const stream of [0,1,2]) {
    const src = original.filter(p => p.stream_index === stream && +p.pts_time >= 60 && +p.pts_time < 70);
    const out = exported.filter(p => p.stream_index === stream);
    assert.ok(src.length > 0);
    assert.equal(out.length, src.length);
    src.forEach((p,i) => {
      assert.equal(out[i].data_hash, p.data_hash);
      assert.ok(Math.abs(+out[i].pts_time - (+p.pts_time-60)) < 0.0011);
    });
  }
  w.wedeo_free(params[0], params[1]);
  w.wedeo_free(params[2], params[3]);
}
try {
  const output = join(dir, "clip.mkv");
  writeFileSync(output, bytes(w.wedeo_bytes_ptr(), w.wedeo_bytes_len()));
  execFileSync("ffmpeg", ["-v","error","-xerror","-i",output,"-map","0","-f","null","-"]);
  const original = packets(file), clip = packets(output);
  for (const track of [0, 1, 2]) {
    const source = original.filter(p => p.stream_index === track);
    const cut = clip.filter(p => p.stream_index === track);
    const offset = source.findIndex(p => p.data_hash === cut[0].data_hash);
    assert.ok(offset >= 0);
    assert.ok(Math.abs(Number(source[offset].pts_time) - start) < 0.11);
    assert.ok(Math.abs(Number(cut.at(-1).pts_time) - 10) < 0.11);
    cut.forEach((packet, i) => {
      assert.equal(packet.data_hash, source[offset+i].data_hash);
      assert.ok(Math.abs(Number(packet.pts_time) - (Number(source[offset+i].pts_time) - start)) <= 0.0011);
    });
  }
  verifyIntervalAudition();
  verifyReorderedVideo(0);
  verifyReorderedVideo(1);
  console.log("Closed/open GOP decoded pictures match. Two audition tones verified; ten-second export independently decoded and packet-checked.");
} finally { rmSync(dir, { recursive: true }); }
