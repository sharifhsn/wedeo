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
    "-show_entries","packet=stream_index,pts_time,data_hash","-of","json",path])).packets;
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
  console.log("Two audition tones verified; ten-second export independently decoded and packet-checked.");
} finally { rmSync(dir, { recursive: true }); }
