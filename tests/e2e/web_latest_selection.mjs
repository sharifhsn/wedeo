#!/usr/bin/env node
// Exercise the actual page, WASM, and sample files in a separate Firefox process.
import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";

const root = resolve(process.argv[2] || "web");
const profile = await mkdtemp(join(tmpdir(), "wedeo-race-firefox-"));
const expected = ["index.html", "app.js", "wedeo.wasm", "samples/h264_aac_320x240.mp4", "samples/chord_48k.wav"];
const hashes = {};
for (const name of expected) hashes[name] = createHash("sha256").update(await readFile(join(root, name))).digest("hex");

let server, firefox, socket;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const timeout = setTimeout(() => { console.error("Timed out"); process.exitCode = 1; socket?.close(); firefox?.kill("SIGTERM"); server?.closeAllConnections(); server?.close(); }, 30000);
try {
  server = createServer(async (req, res) => {
    try {
      let pathname = new URL(req.url, "http://localhost").pathname;
      if (pathname === "/") pathname = "/index.html";
      const filename = resolve(root, "." + pathname);
      if (!filename.startsWith(root + sep)) { res.writeHead(403).end(); return; }
      const bytes = await readFile(filename);
      if (pathname === "/wedeo.wasm") await sleep(1000);
      if (pathname === "/samples/h264_aac_320x240.mp4") await sleep(3000);
      const type = pathname.endsWith(".wasm") ? "application/wasm" : pathname.endsWith(".js") ? "text/javascript" : pathname.endsWith(".html") ? "text/html" : pathname.endsWith(".css") ? "text/css" : pathname.endsWith(".wav") ? "audio/wav" : pathname.endsWith(".mp4") ? "video/mp4" : "application/octet-stream";
      res.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store" }).end(bytes);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise(r => server.listen(0, "127.0.0.1", r));
  firefox = spawn(process.env.FIREFOX || "firefox", ["--headless", "--no-remote", "--profile", profile, "--remote-debugging-port", "0", "about:blank"], { stdio: ["ignore", "pipe", "pipe"] });
  const endpoint = await new Promise((resolve, reject) => {
    let text = "";
    const timer = setTimeout(() => reject(new Error("Firefox BiDi startup timed out")), 10000);
    const handle = data => {
      text += data;
      const match = text.match(/WebDriver BiDi listening on (ws:\/\/127\.0\.0\.1:\d+)/);
      if (match) { clearTimeout(timer); resolve(match[1] + "/session"); }
    };
    firefox.stdout.on("data", handle); firefox.stderr.on("data", handle);
    firefox.once("exit", () => reject(new Error("Firefox exited during startup")));
  });
  socket = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let next = 0;
  const pending = new Map();
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (!message.id || !pending.has(message.id)) return;
    const [resolve, reject] = pending.get(message.id); pending.delete(message.id);
    if (message.type === "error") reject(new Error(message.message)); else resolve(message.result);
  };
  const call = (method, params) => new Promise((resolve, reject) => {
    const id = ++next; pending.set(id, [resolve, reject]); socket.send(JSON.stringify({ id, method, params }));
  });
  await call("session.new", { capabilities: { alwaysMatch: {} } });
  const { context } = await call("browsingContext.create", { type: "tab" });
  const evaluate = async expression => {
    const r = await call("script.evaluate", { expression, target: { context }, awaitPromise: true });
    if (r.type !== "success") throw new Error(JSON.stringify(r));
    return JSON.parse(r.result.value);
  };
  const url = `http://127.0.0.1:${server.address().port}/`;
  const run = async (name, expression, expectedFile) => {
    await call("browsingContext.navigate", { context, url: url + "?" + name, wait: "interactive" });
    const result = await evaluate(expression);
    if (!result.ready || !result.file.startsWith(expectedFile) || result.error || !result.audio) throw new Error(name + " failed: " + JSON.stringify(result));
    console.log(name + " passed");
  };
  const status = `({ready:document.querySelector("#wasm-status").textContent.includes("ready"),file:document.querySelector("#file-label").textContent,error:document.querySelector("#error-banner").textContent,audio:document.body.innerText.includes("pcm_s16le")})`;
  await run("early-sample", `(async()=>{document.querySelector('[data-sample="samples/chord_48k.wav"]').click();await new Promise(r=>setTimeout(r,4000));return JSON.stringify(${status})})()`, "chord_48k.wav");
  await run("late-sample", `(async()=>{while(!document.querySelector("#wasm-status").textContent.includes("ready"))await new Promise(r=>setTimeout(r,20));document.querySelector('[data-sample="samples/chord_48k.wav"]').click();await new Promise(r=>setTimeout(r,4000));return JSON.stringify(${status})})()`, "chord_48k.wav");
  await run("local-file", `(async()=>{const data=await fetch("/samples/chord_48k.wav").then(r=>r.arrayBuffer());const dt=new DataTransfer();dt.items.add(new File([data],"selected.wav",{type:"audio/wav"}));const el=document.querySelector("#file-input");el.files=dt.files;el.dispatchEvent(new Event("change",{bubbles:true}));await new Promise(r=>setTimeout(r,4000));return JSON.stringify(${status})})()`, "selected.wav");
  for (const name of expected) {
    const current = createHash("sha256").update(await readFile(join(root, name))).digest("hex");
    if (current !== hashes[name]) throw new Error("Source changed during test: " + name);
  }
  console.log(JSON.stringify({ status: "PASS", hashes }));
} finally {
  clearTimeout(timeout);
  socket?.close();
  firefox?.kill("SIGTERM");
  if (firefox && firefox.exitCode === null) await Promise.race([new Promise(r => firefox.once("exit", r)), sleep(1500)]);
  server?.closeAllConnections();
  if (server) await new Promise(r => server.close(r));
  await rm(profile, { recursive: true, force: true });
}
