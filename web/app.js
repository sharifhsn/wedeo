(() => {
  const $ = (id) => document.getElementById(id);
  const enc = new TextEncoder();
  const dec = new TextDecoder();

  const clipMode = new URLSearchParams(location.search).has("clip");
  if (clipMode) {
    document.title = "Multitrack clipper | Wedeo";
    for (const selector of [".player-panel > .panel-head", ".stage", "#video-controls", "#video-telemetry"]) {
      document.querySelector(selector).style.display = "none";
    }
  }
  let previewUrl = null;
  let audioStart = 0;
  let wasm = null;
  let wasmReady = null;
  let mediaSelection = 0;
  let currentFile = null; // { name, bytes: Uint8Array, inspect }
  let decodedFrames = []; // [{ imageData, meta, decodeMs }]
  let videoFps = 24;
  let currentFrameIdx = 0;
  let playing = false;
  let playTimer = null;

  let audioCtx = null;
  let audioBuffer = null;
  let audioSource = null;

  function showError(msg) {
    const el = $("error-banner");
    if (!msg) {
      el.hidden = true;
      el.textContent = "";
    } else {
      el.hidden = false;
      el.textContent = msg;
    }
  }

  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MiB`;
  }

  function allocBytes(u8) {
    const ptr = wasm.wedeo_alloc(u8.length);
    new Uint8Array(wasm.memory.buffer, ptr, u8.length).set(u8);
    return [ptr, u8.length];
  }

  function freeBytes(ptr, len) {
    if (ptr && len > 0) wasm.wedeo_free(ptr, len);
  }

  function readJson() {
    const ptr = wasm.wedeo_json_ptr();
    const len = wasm.wedeo_json_len();
    return JSON.parse(dec.decode(new Uint8Array(wasm.memory.buffer, ptr, len)));
  }

  function readBytesCopy() {
    const ptr = wasm.wedeo_bytes_ptr();
    const len = wasm.wedeo_bytes_len();
    return new Uint8Array(wasm.memory.buffer, ptr, len).slice();
  }

  function readError() {
    const ptr = wasm.wedeo_error_ptr();
    const len = wasm.wedeo_error_len();
    return dec.decode(new Uint8Array(wasm.memory.buffer, ptr, len)) || "unknown wasm error";
  }

  async function initWasm() {
    const t0 = performance.now();
    const sampleUrl = clipMode ? "samples/multitrack.mp4" : "samples/h264_aac_320x240.mp4";
    const sampleRequest = fetch(sampleUrl).then(
      response => ({ response }),
      error => ({ error })
    );
    try {
      const resp = await fetch("wedeo.wasm?release=audition1");
      if (!resp.ok) throw new Error(`HTTP ${resp.status} loading wedeo.wasm`);
      let instance;
      if (WebAssembly.instantiateStreaming) {
        try {
          ({ instance } = await WebAssembly.instantiateStreaming(resp.clone(), {}));
        } catch (_) {
          const buf = await resp.arrayBuffer();
          ({ instance } = await WebAssembly.instantiate(buf, {}));
        }
      } else {
        const buf = await resp.arrayBuffer();
        ({ instance } = await WebAssembly.instantiate(buf, {}));
      }
      wasm = instance.exports;
      const ms = Math.round(performance.now() - t0);
      $("wasm-status").textContent = `wasm ready (${ms} ms)`;
      // Auto-load the vivid H.264 + AAC MP4 sample so visitors see it working immediately.
      if (mediaSelection === 0) loadSample(sampleUrl, sampleRequest);
    } catch (e) {
      $("wasm-status").textContent = "wasm failed";
      showError(`Failed to initialize wedeo.wasm: ${e.message}`);
    }
  }

  async function loadSample(url, pendingRequest = null) {
    const selection = ++mediaSelection;
    if (!wasm) await wasmReady;
    if (!wasm || selection !== mediaSelection) return;
    showError(null);
    $("export-status").textContent = "loading sample…";
    try {
      const result = pendingRequest ? await pendingRequest : { response: await fetch(url) };
      if (result.error) throw result.error;
      const resp = result.response;
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = new Uint8Array(await resp.arrayBuffer());
      if (selection !== mediaSelection) return;
      const name = url.split("/").pop();
      await loadMediaBytes(name, buf);
      if (selection === mediaSelection) $("export-status").textContent = "";
    } catch (e) {
      if (selection === mediaSelection) showError(`Failed to fetch sample ${url}: ${e.message}`);
    }
  }

  function stopPlayback() {
    playing = false;
    if (playTimer) {
      clearTimeout(playTimer);
      playTimer = null;
    }
    $("btn-play").textContent = "Play";
  }

  function stopAudio() {
    if (audioSource) {
      try { audioSource.stop(); } catch (_) {}
      audioSource = null;
    }
    $("btn-audio-stop").disabled = true;
  }

  function renderAstNode(node, depth = 0) {
    const label = `<span class="ast-kind">${escapeHtml(node.kind)}</span> <span class="ast-off">@${node.offset} (${formatBytes(node.size)})</span>${node.summary ? ` <span class="ast-sum">${escapeHtml(node.summary)}</span>` : ""}`;
    if (node.children && node.children.length > 0) {
      const openAttr = depth < 2 ? " open" : "";
      return `<details${openAttr}><summary>${label}</summary>${node.children.map((c) => renderAstNode(c, depth + 1)).join("")}</details>`;
    }
    return `<div class="ast-leaf">${label}</div>`;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function renderInspect(info) {
    $("stat-format").textContent = info.format;
    $("stat-size").textContent = formatBytes(info.bytes);
    $("stat-dur").textContent = info.duration_sec != null ? `${info.duration_sec.toFixed(2)}s` : "—";
    $("stat-tracks").textContent = String(info.tracks.length);

    const hb = $("health-badge");
    if (info.health.is_clean) {
      hb.textContent = "StreamHealth: clean";
      hb.className = "badge ok";
    } else {
      hb.textContent = `corrupt=${info.health.corrupt_packets} glitches=${info.health.timestamp_glitches}`;
      hb.className = "badge warn";
    }

    const list = $("tracks-list");
    list.innerHTML = info.tracks.map((t) => {
      const details = t.media_type === "video"
        ? `${t.width}×${t.height} · ${t.pixel_layout} · ${t.fps ? t.fps.toFixed(2) + " fps" : "VFR"} · ${t.color}`
        : t.media_type === "audio"
          ? `${t.sample_rate} Hz · ${t.channels} ch · ${t.sample_format}`
          : `time_base=${t.time_base}`;
      return `<div class="track-card">
        <div class="track-head">
          <strong>#${t.id} ${escapeHtml(t.media_type)}</strong>
          <span class="track-pill">${escapeHtml(t.codec)}${t.decodable ? "" : " (copy only)"}</span>
        </div>
        <div class="track-meta">${escapeHtml(details)}</div>
      </div>`;
    }).join("");

    const astEl = $("ast-tree");
    if (info.ast && info.ast.length > 0) {
      astEl.innerHTML = info.ast.map((n) => renderAstNode(n, 0)).join("");
    } else {
      astEl.textContent = "Raw elementary stream (no container box AST).";
    }
  }

  function drawFrame(idx) {
    if (!decodedFrames.length) return;
    currentFrameIdx = Math.max(0, Math.min(idx, decodedFrames.length - 1));
    const entry = decodedFrames[currentFrameIdx];
    const canvas = $("video-canvas");
    if (canvas.width !== entry.meta.width || canvas.height !== entry.meta.height) {
      canvas.width = entry.meta.width;
      canvas.height = entry.meta.height;
    }
    canvas.style.display = "block";
    $("stage-empty").style.display = "none";
    const ctx = canvas.getContext("2d");
    ctx.putImageData(entry.imageData, 0, 0);

    $("scrubber").value = String(currentFrameIdx);
    $("frame-readout").textContent = `${currentFrameIdx + 1} / ${decodedFrames.length}`;
    const pts = entry.meta.pts_sec != null ? `${entry.meta.pts_sec.toFixed(3)}s` : "n/a";
    $("video-telemetry").innerHTML = [
      `<span>codec: <b>${escapeHtml(entry.codec)}</b></span>`,
      `<span>${entry.meta.width}×${entry.meta.height} (${escapeHtml(entry.meta.src_layout)} → RGBA8)</span>`,
      `<span>SwsOpList: ${entry.meta.sws_ops} ops</span>`,
      `<span>decode: ${entry.decodeMs.toFixed(1)} ms</span>`,
      `<span>pts: ${pts} (${escapeHtml(entry.meta.pict_type)})</span>`,
    ].join("");
  }

  function drawWaveform(f32, channels, frames) {
    const canvas = $("wave-canvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.fillStyle = "#060709";
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(frames / W));
    const mid = H / 2;
    for (let x = 0; x < W; x++) {
      const start = x * step;
      let min = 1, max = -1;
      for (let i = 0; i < step && start + i < frames; i++) {
        const v = f32[start + i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      ctx.moveTo(x, mid + min * (mid - 4));
      ctx.lineTo(x, mid + max * (mid - 4));
    }
    ctx.stroke();
  }

  function decodeSelectedAudio(ptr, len, nptr, nlen) {
    stopAudio();
    audioBuffer = null;
    $("btn-audio-play").disabled = true;
    const at0 = performance.now();
    const id = Number($("audition").value);
    const rc = !$("clip-panel").hidden
      ? wasm.wedeo_audition_window(ptr, len, nptr, nlen, id, Number($("clip-in").value), Number($("clip-out").value))
      : wasm.wedeo_audition(ptr, len, nptr, nlen, id);
    if (rc !== 0) {
      throw new Error(readError());
    }
    const aInfo = readJson();
    audioStart = aInfo.start_sec;
    const raw = readBytesCopy();
    wasm.wedeo_release_output_bytes();
    const f32 = new Float32Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 4));
    const aMs = Math.round(performance.now() - at0);

    drawWaveform(f32, aInfo.channels, aInfo.frames_per_channel);
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioBuffer = audioCtx.createBuffer(aInfo.channels, aInfo.frames_per_channel, aInfo.sample_rate);
    for (let c = 0; c < aInfo.channels; c++) {
      const offset = c * aInfo.frames_per_channel;
      audioBuffer.copyToChannel(f32.subarray(offset, offset + aInfo.frames_per_channel), c);
    }
    $("btn-audio-play").disabled = false;
    $("audio-telemetry").textContent =
      `${aInfo.codec} → F32Planar · ${aInfo.sample_rate} Hz · ${aInfo.channels} ch · ${aInfo.duration_sec.toFixed(2)}s (${aInfo.frames_per_channel.toLocaleString()} samples decoded in ${aMs} ms)`;
  }

  function withFile(operation) {
    if (!currentFile) return;
    const [p, l] = allocBytes(currentFile.bytes);
    const [n, k] = allocBytes(enc.encode(currentFile.name));
    try { operation(p, l, n, k); }
    catch (e) { showError(e.message); }
    finally { freeBytes(n, k); freeBytes(p, l); }
  }

  const monitor = $("clip-monitor");
  function syncMonitorAudio() {
    stopAudio();
    if (!audioBuffer || !audioCtx || monitor.paused || monitor.seeking) return;
    const offset = monitor.currentTime - audioStart;
    if (offset >= audioBuffer.duration) return;
    audioCtx.resume().catch(e => showError(e.message));
    audioSource = audioCtx.createBufferSource();
    audioSource.buffer = audioBuffer;
    audioSource.playbackRate.value = monitor.playbackRate;
    audioSource.connect(audioCtx.destination);
    audioSource.start(audioCtx.currentTime + Math.max(0, -offset) / monitor.playbackRate, Math.max(0, offset));
  }
  monitor.onplaying = monitor.onseeked = monitor.onratechange = syncMonitorAudio;
  monitor.onpause = monitor.onwaiting = monitor.onseeking = stopAudio;
  monitor.onvolumechange = () => { monitor.muted = true; };
  $("audition").onchange = () => {
    withFile(decodeSelectedAudio);
    syncMonitorAudio();
  };
  function markAtPlayhead(start) {
    const selector = $(start ? "clip-in" : "clip-out");
    const options = Array.from(selector.options);
    const position = monitor.currentTime;
    const mark = start ? options.filter(o => Number(o.value) <= position).pop()
                       : options.find(o => !o.disabled && Number(o.value) >= position);
    if (mark) { selector.value = mark.value; changeSelection(); }
  }
  $("mark-in").onclick = () => markAtPlayhead(true);
  $("mark-out").onclick = () => markAtPlayhead(false);
  function updateClip() {
    const start = Number($("clip-in").value);
    for (const option of $("clip-out").options) option.disabled = Number(option.value) <= start;
    if (Number($("clip-out").value) <= start) {
      $("clip-out").value = Array.from($("clip-out").options).find(o => !o.disabled).value;
    }
    $("clip-result").textContent = "";
  }
  function changeSelection() {
    updateClip();
    monitor.pause();
    monitor.currentTime = Number($("clip-in").value);
    if ($("audition").options.length) withFile(decodeSelectedAudio);
  }
  $("clip-in").onchange = $("clip-out").onchange = changeSelection;
  monitor.ontimeupdate = () => {
    if (!monitor.paused && monitor.currentTime >= Number($("clip-out").value)) monitor.pause();
  };
  monitor.onplay = () => {
    const start = Number($("clip-in").value), end = Number($("clip-out").value);
    if (monitor.currentTime < start || monitor.currentTime >= end) monitor.currentTime = start;
  };
  $("clip-save").onclick = () => withFile((p, l, n, k) => {
    const start = Number($("clip-in").value), end = Number($("clip-out").value);
    if (wasm.wedeo_clip(p, l, n, k, start, end) !== 0) throw new Error(readError());
    const bytes = readBytesCopy();
    triggerDownload(bytes, currentFile.name.replace(/\.[^.]+$/, "") + ".clip.mkv", "video/x-matroska");
    $("clip-result").textContent = `Exported ${(end - start).toFixed(2)} seconds · ${formatBytes(bytes.length)}`;
  });

  async function loadMediaBytes(name, bytes) {
    showError(null);
    stopPlayback();
    stopAudio();
    const monitor = $("clip-monitor");
    monitor.pause();
    monitor.removeAttribute("src");
    monitor.load();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    currentFile = null;
    $("clip-panel").hidden = true;
    $("audition").replaceChildren();
    decodedFrames = [];
    audioBuffer = null;
    $("video-canvas").style.display = "none";
    $("stage-empty").style.display = "block";
    $("stage-empty").textContent = "Inspecting & decoding in pure Rust WASM…";
    $("file-label").textContent = `${name} (${formatBytes(bytes.length)})`;

    const [ptr, len] = allocBytes(bytes);
    const [nptr, nlen] = allocBytes(enc.encode(name));
    let info;
    try {
      if (wasm.wedeo_inspect(ptr, len, nptr, nlen) !== 0) {
        throw new Error(readError());
      }
      info = readJson();
      currentFile = { name, bytes, inspect: info };
      renderInspect(info);

      // Enable remux buttons for container inputs
      const canRemux = info.tracks.length > 0;
      $("btn-remux-mkv").disabled = !canRemux;
      $("btn-remux-mp4").disabled = !canRemux;

      const hasVideo = info.tracks.some((t) => t.media_type === "video" && t.decodable);
      const hasAudio = info.tracks.some((t) => t.media_type === "audio" && t.decodable);
      $("btn-export-wav").disabled = !hasAudio;

      const audioTracks = info.tracks.filter(t => t.media_type === "audio" && t.decodable);
      $("audition").replaceChildren(...audioTracks.map((t, i) => new Option(`Audio ${i + 1} · ${t.codec}`, t.id)));
      if (hasVideo && wasm.wedeo_clip_marks(ptr, len, nptr, nlen) === 0) {
        const marks = readJson();
        const options = times => times.map(t => new Option(`${t.toFixed(3)} s`, String(t)));
        $("clip-in").replaceChildren(...options(marks.slice(0, -1)));
        $("clip-out").replaceChildren(...options(marks.slice(1)));
        // Start with a short audition on long recordings; export can still span the whole file.
        const firstWindow = marks.slice(1).findIndex(t => t >= marks[0] + 10);
        $("clip-out").selectedIndex = firstWindow >= 0 ? firstWindow : marks.length - 2;
        updateClip();
        previewUrl = URL.createObjectURL(new Blob([bytes]));
        monitor.src = previewUrl;
        $("clip-panel").hidden = false;
        if (clipMode) $("video-controls").closest(".player-panel").prepend($("clip-panel"));
      }

      // 1. Decode video frames (capped at 120 frames for instant snappy interactive scrubbing)
      if (hasVideo && !clipMode) {
        if (wasm.wedeo_video_open(ptr, len, nptr, nlen) !== 0) {
          throw new Error(readError());
        }
        const openMeta = readJson();
        videoFps = openMeta.fps || 24;
        const maxFrames = 120;
        const totalStart = performance.now();
        while (decodedFrames.length < maxFrames) {
          const ft0 = performance.now();
          const rc = wasm.wedeo_video_next_frame();
          const decodeMs = performance.now() - ft0;
          if (rc === 0) break;
          if (rc < 0) throw new Error(readError());
          const meta = readJson();
          const rgba = readBytesCopy();
          const imageData = new ImageData(new Uint8ClampedArray(rgba.buffer), meta.width, meta.height);
          decodedFrames.push({ imageData, meta, decodeMs, codec: openMeta.codec });
        }
        wasm.wedeo_video_close();

        if (decodedFrames.length > 0) {
          $("scrubber").max = String(decodedFrames.length - 1);
          $("scrubber").disabled = false;
          $("btn-play").disabled = decodedFrames.length < 2;
          $("btn-step").disabled = decodedFrames.length < 2;
          $("btn-rewind").disabled = false;
          drawFrame(0);
          const totalMs = Math.round(performance.now() - totalStart);
          $("export-status").textContent = `decoded ${decodedFrames.length} video frames in ${totalMs} ms`;
        } else {
          $("stage-empty").textContent = "Video track produced 0 frames.";
        }
      } else {
        $("btn-play").disabled = true;
        $("btn-step").disabled = true;
        $("btn-rewind").disabled = true;
        $("scrubber").disabled = true;
        $("frame-readout").textContent = "0 / 0";
        $("video-telemetry").innerHTML = "<span>Audio-only stream (or codec not enabled in WASM)</span>";
        $("stage-empty").textContent = hasAudio
          ? "Audio-only file loaded — see decoded waveform & Web Audio player below."
          : "No decodable video track found.";
      }

      // 2. Decode audio track for waveform + Web Audio playback
      if (hasAudio) {
        decodeSelectedAudio(ptr, len, nptr, nlen);
      } else {
        $("btn-audio-play").disabled = true;
        $("btn-audio-stop").disabled = true;
        const wctx = $("wave-canvas").getContext("2d");
        wctx.fillStyle = "#060709";
        wctx.fillRect(0, 0, 600, 72);
        $("audio-telemetry").textContent = "No audio track in this file.";
      }
    } catch (e) {
      showError(e.message);
      $("stage-empty").textContent = "Error processing file (see banner above).";
    } finally {
      freeBytes(ptr, len);
      freeBytes(nptr, nlen);
    }
  }

  function triggerDownload(bytes, filename, mime) {
    const blob = new Blob([bytes], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }

  function runRemux(targetFmt) {
    if (!wasm || !currentFile) return;
    showError(null);
    const [ptr, len] = allocBytes(currentFile.bytes);
    const [nptr, nlen] = allocBytes(enc.encode(currentFile.name));
    const [fptr, flen] = allocBytes(enc.encode(targetFmt));
    try {
      const t0 = performance.now();
      if (wasm.wedeo_remux(ptr, len, nptr, nlen, fptr, flen) !== 0) {
        throw new Error(readError());
      }
      const meta = readJson();
      const out = readBytesCopy();
      const ms = Math.round(performance.now() - t0);
      const base = currentFile.name.replace(/\.[^.]+$/, "");
      const outName = `${base}.wedeo.${targetFmt}`;
      triggerDownload(out, outName, targetFmt === "mp4" ? "video/mp4" : "video/x-matroska");
      $("export-status").textContent =
        `remuxed ${meta.src_format} → ${meta.dst_format} (${meta.packets} pkts, ${formatBytes(meta.bytes)}) in ${ms} ms`;
    } catch (e) {
      showError(`Remux failed: ${e.message}`);
    } finally {
      freeBytes(ptr, len);
      freeBytes(nptr, nlen);
      freeBytes(fptr, flen);
    }
  }

  function runExportWav() {
    if (!wasm || !currentFile) return;
    showError(null);
    const rate = parseInt($("wav-rate").value, 10) || 0;
    const ch = parseInt($("wav-ch").value, 10) || 0;
    const [ptr, len] = allocBytes(currentFile.bytes);
    const [nptr, nlen] = allocBytes(enc.encode(currentFile.name));
    try {
      const t0 = performance.now();
      if (wasm.wedeo_export_wav(ptr, len, nptr, nlen, rate, ch) !== 0) {
        throw new Error(readError());
      }
      const meta = readJson();
      const out = readBytesCopy();
      const ms = Math.round(performance.now() - t0);
      const base = currentFile.name.replace(/\.[^.]+$/, "");
      triggerDownload(out, `${base}_${meta.sample_rate}hz_${meta.channels}ch.wav`, "audio/wav");
      $("export-status").textContent =
        `exported WAV ${meta.src_rate}Hz/${meta.src_channels}ch → ${meta.sample_rate}Hz/${meta.channels}ch (${formatBytes(meta.bytes)}) in ${ms} ms`;
    } catch (e) {
      showError(`WAV export failed: ${e.message}`);
    } finally {
      freeBytes(ptr, len);
      freeBytes(nptr, nlen);
    }
  }

  function playLoop() {
    if (!playing || !decodedFrames.length) return;
    drawFrame((currentFrameIdx + 1) % decodedFrames.length);
    playTimer = setTimeout(playLoop, Math.max(16, Math.round(1000 / videoFps)));
  }

  // Event bindings
  $("file-input").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const selection = ++mediaSelection;
    const buf = new Uint8Array(await file.arrayBuffer());
    if (!wasm) await wasmReady;
    if (!wasm || selection !== mediaSelection) return;
    await loadMediaBytes(file.name, buf);
  });

  document.querySelectorAll("[data-sample]").forEach((btn) => {
    btn.addEventListener("click", () => loadSample(btn.getAttribute("data-sample")));
  });

  const dz = $("dropzone");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", async (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const selection = ++mediaSelection;
    const buf = new Uint8Array(await file.arrayBuffer());
    if (!wasm) await wasmReady;
    if (!wasm || selection !== mediaSelection) return;
    await loadMediaBytes(file.name, buf);
  });

  $("btn-play").addEventListener("click", () => {
    if (playing) {
      stopPlayback();
    } else {
      playing = true;
      $("btn-play").textContent = "Pause";
      playLoop();
    }
  });
  $("btn-step").addEventListener("click", () => {
    stopPlayback();
    drawFrame((currentFrameIdx + 1) % decodedFrames.length);
  });
  $("btn-rewind").addEventListener("click", () => {
    stopPlayback();
    drawFrame(0);
  });
  $("scrubber").addEventListener("input", (e) => {
    stopPlayback();
    drawFrame(parseInt(e.target.value, 10) || 0);
  });

  $("btn-audio-play").addEventListener("click", async () => {
    if (!audioBuffer || !audioCtx) return;
    monitor.pause();
    stopAudio();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    audioSource = audioCtx.createBufferSource();
    audioSource.buffer = audioBuffer;
    audioSource.connect(audioCtx.destination);
    audioSource.onended = () => { $("btn-audio-stop").disabled = true; };
    audioSource.start();
    $("btn-audio-stop").disabled = false;
  });
  $("btn-audio-stop").addEventListener("click", stopAudio);

  $("btn-remux-mkv").addEventListener("click", () => runRemux("mkv"));
  $("btn-remux-mp4").addEventListener("click", () => runRemux("mp4"));
  $("btn-export-wav").addEventListener("click", runExportWav);

  wasmReady = initWasm();
})();
