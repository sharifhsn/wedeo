#!/usr/bin/env node
/**
 * Zero-dependency Node v20 ESM bridge for wedeo WASM ABI & @wedeo/wasm E2E tests.
 */

import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";

const STREAM_EXPORTS = [
  "wedeo_alloc",
  "wedeo_free",
  "wedeo_json_ptr",
  "wedeo_json_len",
  "wedeo_bytes_ptr",
  "wedeo_bytes_len",
  "wedeo_error_ptr",
  "wedeo_error_len",
  "wedeo_stream_open",
  "wedeo_stream_push",
  "wedeo_stream_inspect",
  "wedeo_stream_next_frame",
  "wedeo_stream_next_remux_chunk",
  "wedeo_stream_close",
];

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function parseCliArgs(argv) {
  if (argv.length === 0) {
    throw new Error("Usage: wasm_runner.mjs <command> [--key value ...]");
  }
  const command = argv[0];
  const flags = {};
  for (let i = 1; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${token}`);
    }
    const eq = token.indexOf("=");
    if (eq !== -1) {
      flags[token.slice(2, eq)] = token.slice(eq + 1);
      continue;
    }
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      flags[key] = true;
    } else {
      flags[key] = next;
      i++;
    }
  }
  return { command, flags };
}

function requireFlag(flags, key) {
  const val = flags[key];
  if (val === undefined || val === true || val === "") {
    throw new Error(`Missing required flag --${key}`);
  }
  return String(val);
}

function intFlag(flags, key, fallback) {
  if (flags[key] === undefined) return fallback;
  const n = Number.parseInt(String(flags[key]), 10);
  if (!Number.isFinite(n)) {
    throw new Error(`Invalid integer for --${key}: ${flags[key]}`);
  }
  return n;
}

function numFlag(flags, key, fallback) {
  if (flags[key] === undefined) return fallback;
  const n = Number.parseFloat(String(flags[key]));
  if (!Number.isFinite(n)) {
    throw new Error(`Invalid number for --${key}: ${flags[key]}`);
  }
  return n;
}

class BinaryReader {
  constructor(buf, start = 0, end = buf.length) {
    this.buf = buf;
    this.pos = start;
    this.end = end;
  }
  hasMore() {
    return this.pos < this.end;
  }
  readByte() {
    if (this.pos >= this.end) {
      throw new Error("Unexpected EOF while reading WASM binary");
    }
    return this.buf[this.pos++];
  }
  readBytes(n) {
    if (this.pos + n > this.end) {
      throw new Error("Unexpected EOF while reading WASM bytes");
    }
    const slice = this.buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return slice;
  }
  readVarU32() {
    let result = 0;
    let shift = 0;
    while (true) {
      const b = this.readByte();
      result |= (b & 0x7f) << shift;
      if ((b & 0x80) === 0) return result >>> 0;
      shift += 7;
      if (shift > 35) throw new Error("LEB128 u32 overflow");
    }
  }
  skipSignedLeb128(maxBits = 64) {
    let shift = 0;
    while (true) {
      const b = this.readByte();
      if ((b & 0x80) === 0) return;
      shift += 7;
      if (shift > maxBits + 7) throw new Error("LEB128 signed overflow");
    }
  }
  readString() {
    const len = this.readVarU32();
    return decoder.decode(this.readBytes(len));
  }
}

function skipMemArg(r) {
  const flags = r.readVarU32();
  if ((flags & 0x40) !== 0) {
    r.readVarU32(); // multi-memory index if enabled
  }
  r.readVarU32(); // offset
}

function inspectWasmBinary(buf) {
  if (
    buf.length < 8 ||
    buf[0] !== 0x00 ||
    buf[1] !== 0x61 ||
    buf[2] !== 0x73 ||
    buf[3] !== 0x6d
  ) {
    throw new Error("Invalid WASM magic header");
  }
  const r = new BinaryReader(buf, 8, buf.length);
  const sections = [];
  const targetFeatures = [];
  let simd128OpCount = 0;
  let codeFunctionsWalked = 0;
  let desyncedFunctions = 0;

  while (r.hasMore()) {
    const id = r.readByte();
    const size = r.readVarU32();
    const sectionEnd = r.pos + size;
    const sec = new BinaryReader(buf, r.pos, sectionEnd);
    let customName = null;

    if (id === 0) {
      customName = sec.readString();
      if (customName === "target_features") {
        const count = sec.readVarU32();
        for (let i = 0; i < count; i++) {
          const prefix = sec.readByte();
          const feat = sec.readString();
          targetFeatures.push({ prefix, name: feat });
        }
      }
    } else if (id === 10) {
      const funcCount = sec.readVarU32();
      for (let f = 0; f < funcCount; f++) {
        const bodySize = sec.readVarU32();
        const bodyEnd = sec.pos + bodySize;
        const body = new BinaryReader(buf, sec.pos, bodyEnd);
        codeFunctionsWalked++;

        const localGroups = body.readVarU32();
        for (let g = 0; g < localGroups; g++) {
          body.readVarU32(); // count
          body.readByte(); // valtype
        }

        let desynced = false;
        while (body.hasMore()) {
          const op = body.readByte();
          if (op === 0x00 || op === 0x01 || op === 0x05 || op === 0x0b || op === 0x0f || op === 0x1a || op === 0x1b || op === 0xd1) {
            continue;
          }
          if (op === 0x02 || op === 0x03 || op === 0x04 || op === 0xd0) {
            body.skipSignedLeb128(33);
            continue;
          }
          if (op === 0x0c || op === 0x0d || op === 0x10 || (op >= 0x20 && op <= 0x26) || op === 0xd2) {
            body.readVarU32();
            continue;
          }
          if (op === 0x0e) {
            const count = body.readVarU32();
            for (let k = 0; k <= count; k++) body.readVarU32();
            continue;
          }
          if (op === 0x11) {
            body.readVarU32();
            body.readVarU32();
            continue;
          }
          if (op === 0x1c) {
            const count = body.readVarU32();
            body.readBytes(count);
            continue;
          }
          if (op >= 0x28 && op <= 0x3e) {
            skipMemArg(body);
            continue;
          }
          if (op === 0x3f || op === 0x40) {
            body.readByte();
            continue;
          }
          if (op === 0x41) {
            body.skipSignedLeb128(32);
            continue;
          }
          if (op === 0x42) {
            body.skipSignedLeb128(64);
            continue;
          }
          if (op === 0x43) {
            body.readBytes(4);
            continue;
          }
          if (op === 0x44) {
            body.readBytes(8);
            continue;
          }
          if (op >= 0x45 && op <= 0xc4) {
            continue;
          }
          if (op === 0xfc) {
            const sub = body.readVarU32();
            if (sub <= 7) {
              continue;
            } else if (sub === 8) {
              body.readVarU32();
              body.readByte();
            } else if (sub === 9 || sub === 13 || (sub >= 15 && sub <= 17)) {
              body.readVarU32();
            } else if (sub === 10) {
              body.readBytes(2);
            } else if (sub === 11) {
              body.readByte();
            } else if (sub === 12 || sub === 14) {
              body.readVarU32();
              body.readVarU32();
            } else {
              desynced = true;
              break;
            }
            continue;
          }
          if (op === 0xfd) {
            simd128OpCount++;
            const sub = body.readVarU32();
            if (sub <= 11) {
              skipMemArg(body);
            } else if (sub === 12 || sub === 13) {
              body.readBytes(16);
            } else if (sub >= 21 && sub <= 34) {
              body.readByte();
            } else if (sub >= 84 && sub <= 91) {
              skipMemArg(body);
              body.readByte();
            } else if (sub === 92 || sub === 93) {
              skipMemArg(body);
            }
            continue;
          }
          desynced = true;
          break;
        }
        if (desynced) desyncedFunctions++;
        sec.pos = bodyEnd;
      }
    }

    sections.push({
      id,
      name: customName,
      bytes: size,
    });
    r.pos = sectionEnd;
  }

  const hasTargetSimd = targetFeatures.some((f) => f.name === "simd128");
  return {
    sections,
    target_features: targetFeatures,
    code_functions_walked: codeFunctionsWalked,
    desynced_functions: desyncedFunctions,
    simd128_op_count: simd128OpCount,
    has_simd128: hasTargetSimd || simd128OpCount > 0,
  };
}

function cmdProbeWasm(flags) {
  const wasmPath = requireFlag(flags, "wasm");
  const buf = fs.readFileSync(wasmPath);
  const mod = new WebAssembly.Module(buf);
  const imports = WebAssembly.Module.imports(mod).map((imp) => ({
    module: imp.module,
    name: imp.name,
    kind: imp.kind,
  }));
  let inst = null;
  if (imports.length === 0) {
    inst = new WebAssembly.Instance(mod, {});
  }
  const exports = WebAssembly.Module.exports(mod).map((exp) => {
    let arity = null;
    if (inst && typeof inst.exports[exp.name] === "function") {
      arity = inst.exports[exp.name].length;
    }
    return { name: exp.name, kind: exp.kind, arity };
  });
  const binInfo = inspectWasmBinary(buf);
  const gzipDefault = zlib.gzipSync(buf);
  const gzipMax = zlib.gzipSync(buf, { level: 9 });

  return {
    wasm_path: path.resolve(wasmPath),
    raw_bytes: buf.length,
    gzip_default_bytes: gzipDefault.length,
    gzip_max_bytes: gzipMax.length,
    imports,
    exports,
    ...binInfo,
  };
}

class WasmAbiBridge {
  constructor(wasmPath) {
    const buf = fs.readFileSync(wasmPath);
    const mod = new WebAssembly.Module(buf);
    const imports = WebAssembly.Module.imports(mod);
    if (imports.length > 0) {
      throw new Error(
        `Unexpected WASM host imports: ${JSON.stringify(imports)}`
      );
    }
    this.instance = new WebAssembly.Instance(mod, {});
    this.exports = this.instance.exports;
    const missing = STREAM_EXPORTS.filter(
      (name) => typeof this.exports[name] !== "function"
    );
    if (missing.length > 0) {
      throw new Error(
        `WASM binary missing required M5 streaming exports: ${missing.join(", ")}`
      );
    }
    if (!(this.exports.memory instanceof WebAssembly.Memory)) {
      throw new Error("WASM binary does not export 'memory'");
    }
  }
  memoryBytes() {
    return this.exports.memory.buffer.byteLength;
  }
  u8() {
    return new Uint8Array(this.exports.memory.buffer);
  }
  allocBytes(bytes) {
    if (bytes.length === 0) return 0;
    const ptr = this.exports.wedeo_alloc(bytes.length);
    this.u8().set(bytes, ptr);
    return ptr;
  }
  allocStr(str) {
    const bytes = encoder.encode(str);
    return { ptr: this.allocBytes(bytes), len: bytes.length };
  }
  free(ptr, len) {
    this.exports.wedeo_free(ptr, len);
  }
  readJson() {
    const ptr = this.exports.wedeo_json_ptr();
    const len = this.exports.wedeo_json_len();
    if (len === 0) return "";
    return decoder.decode(this.u8().subarray(ptr, ptr + len));
  }
  readBytesCopy() {
    const ptr = this.exports.wedeo_bytes_ptr();
    const len = this.exports.wedeo_bytes_len();
    if (len === 0) return Buffer.alloc(0);
    return Buffer.from(this.u8().subarray(ptr, ptr + len));
  }
  readError() {
    const ptr = this.exports.wedeo_error_ptr();
    const len = this.exports.wedeo_error_len();
    if (len === 0) return "";
    return decoder.decode(this.u8().subarray(ptr, ptr + len));
  }
  openStream(name) {
    const { ptr, len } = this.allocStr(name);
    try {
      const handle = this.exports.wedeo_stream_open(ptr, len);
      if (handle < 0) {
        throw new Error(
          `wedeo_stream_open failed (rc=${handle}): ${this.readError()}`
        );
      }
      return handle;
    } finally {
      this.free(ptr, len);
    }
  }
  pushChunk(handle, chunk) {
    if (chunk.length === 0) {
      return this.exports.wedeo_stream_push(handle, 0, 0);
    }
    const ptr = this.allocBytes(chunk);
    try {
      return this.exports.wedeo_stream_push(handle, ptr, chunk.length);
    } finally {
      this.free(ptr, chunk.length);
    }
  }
  closeStream(handle) {
    return this.exports.wedeo_stream_close(handle);
  }
}

function cmdStreamInspect(flags) {
  const wasmPath = requireFlag(flags, "wasm");
  const inputPath = requireFlag(flags, "input");
  const name = flags.name ? String(flags.name) : path.basename(inputPath);
  const chunkSize = intFlag(flags, "chunk-size", 4096);
  const inputBuf = fs.readFileSync(inputPath);

  const bridge = new WasmAbiBridge(wasmPath);
  const memBefore = bridge.memoryBytes();
  const handle = bridge.openStream(name);
  let chunksPushed = 0;

  try {
    for (let offset = 0; offset < inputBuf.length; offset += chunkSize) {
      const slice = inputBuf.subarray(offset, offset + chunkSize);
      const rc = bridge.pushChunk(handle, slice);
      chunksPushed++;
      if (rc < 0) {
        throw new Error(
          `wedeo_stream_push failed at chunk ${chunksPushed} (rc=${rc}): ${bridge.readError()}`
        );
      }
    }
    const eofRc = bridge.pushChunk(handle, Buffer.alloc(0));
    if (eofRc < 0) {
      throw new Error(
        `wedeo_stream_push(EOF) failed (rc=${eofRc}): ${bridge.readError()}`
      );
    }

    const inspectRc = bridge.exports.wedeo_stream_inspect(handle);
    if (inspectRc !== 0) {
      throw new Error(
        `wedeo_stream_inspect returned rc=${inspectRc}: ${bridge.readError()}`
      );
    }
    const jsonText = bridge.readJson();
    const inspect = JSON.parse(jsonText);
    const closeRc = bridge.closeStream(handle);
    if (closeRc < 0) {
      throw new Error(
        `wedeo_stream_close failed (rc=${closeRc}): ${bridge.readError()}`
      );
    }
    const memAfter = bridge.memoryBytes();
    return {
      inspect,
      chunks_pushed: chunksPushed,
      memory_bytes_before: memBefore,
      memory_bytes_after: memAfter,
    };
  } catch (err) {
    try {
      bridge.closeStream(handle);
    } catch {
      // Ignore cleanup error while propagating primary failure.
    }
    throw err;
  }
}

function cmdStreamDecodeVideo(flags) {
  const wasmPath = requireFlag(flags, "wasm");
  const inputPath = requireFlag(flags, "input");
  const name = flags.name ? String(flags.name) : path.basename(inputPath);
  const outRgba = requireFlag(flags, "out-rgba");
  const chunkSize = intFlag(flags, "chunk-size", 4096);
  const maxFrames = intFlag(flags, "max-frames", 120);
  const inputBuf = fs.readFileSync(inputPath);

  const bridge = new WasmAbiBridge(wasmPath);
  const memBefore = bridge.memoryBytes();
  const handle = bridge.openStream(name);
  const frames = [];
  const rgbaChunks = [];
  let totalRgbaBytes = 0;

  const drainReadyFrames = (eofPushed) => {
    while (frames.length < maxFrames) {
      const rc = bridge.exports.wedeo_stream_next_frame(handle);
      if (rc === 1) {
        const meta = JSON.parse(bridge.readJson());
        const rgba = bridge.readBytesCopy();
        frames.push(meta);
        rgbaChunks.push(rgba);
        totalRgbaBytes += rgba.length;
      } else if (rc === 2) {
        if (eofPushed) {
          throw new Error("wedeo_stream_next_frame returned NeedMore after EOF");
        }
        break;
      } else if (rc === 0) {
        return true;
      } else {
        throw new Error(
          `wedeo_stream_next_frame failed (rc=${rc}): ${bridge.readError()}`
        );
      }
    }
    return frames.length >= maxFrames;
  };

  try {
    let done = false;
    for (let offset = 0; offset < inputBuf.length && !done; offset += chunkSize) {
      const slice = inputBuf.subarray(offset, offset + chunkSize);
      const rc = bridge.pushChunk(handle, slice);
      if (rc < 0) {
        throw new Error(
          `wedeo_stream_push failed (rc=${rc}): ${bridge.readError()}`
        );
      }
      done = drainReadyFrames(false);
    }
    if (!done) {
      const eofRc = bridge.pushChunk(handle, Buffer.alloc(0));
      if (eofRc < 0) {
        throw new Error(
          `wedeo_stream_push(EOF) failed (rc=${eofRc}): ${bridge.readError()}`
        );
      }
      drainReadyFrames(true);
    }
    const closeRc = bridge.closeStream(handle);
    if (closeRc < 0) {
      throw new Error(
        `wedeo_stream_close failed (rc=${closeRc}): ${bridge.readError()}`
      );
    }
    fs.mkdirSync(path.dirname(path.resolve(outRgba)), { recursive: true });
    fs.writeFileSync(outRgba, Buffer.concat(rgbaChunks));
    return {
      frames,
      total_rgba_bytes: totalRgbaBytes,
      memory_bytes_before: memBefore,
      memory_bytes_after: bridge.memoryBytes(),
    };
  } catch (err) {
    try {
      bridge.closeStream(handle);
    } catch {
      // Ignore secondary cleanup error.
    }
    throw err;
  }
}

function cmdStreamRemux(flags) {
  const wasmPath = requireFlag(flags, "wasm");
  const inputPath = requireFlag(flags, "input");
  const format = requireFlag(flags, "format");
  const outPath = requireFlag(flags, "out");
  const name = flags.name ? String(flags.name) : path.basename(inputPath);
  const chunkSize = intFlag(flags, "chunk-size", 4096);
  const inputBuf = fs.readFileSync(inputPath);

  const bridge = new WasmAbiBridge(wasmPath);
  const memBefore = bridge.memoryBytes();
  const handle = bridge.openStream(name);
  const fmtAlloc = bridge.allocStr(format);
  const outChunks = [];
  let remuxChunks = 0;

  const drainRemux = (eofPushed) => {
    while (true) {
      const rc = bridge.exports.wedeo_stream_next_remux_chunk(
        handle,
        fmtAlloc.ptr,
        fmtAlloc.len
      );
      if (rc === 1) {
        const chunk = bridge.readBytesCopy();
        outChunks.push(chunk);
        remuxChunks++;
      } else if (rc === 2) {
        if (eofPushed) {
          throw new Error(
            "wedeo_stream_next_remux_chunk returned NeedMore after EOF"
          );
        }
        break;
      } else if (rc === 0) {
        return true;
      } else {
        throw new Error(
          `wedeo_stream_next_remux_chunk failed (rc=${rc}): ${bridge.readError()}`
        );
      }
    }
    return false;
  };

  try {
    let done = false;
    for (let offset = 0; offset < inputBuf.length && !done; offset += chunkSize) {
      const slice = inputBuf.subarray(offset, offset + chunkSize);
      const rc = bridge.pushChunk(handle, slice);
      if (rc < 0) {
        throw new Error(
          `wedeo_stream_push failed (rc=${rc}): ${bridge.readError()}`
        );
      }
      done = drainRemux(false);
    }
    if (!done) {
      const eofRc = bridge.pushChunk(handle, Buffer.alloc(0));
      if (eofRc < 0) {
        throw new Error(
          `wedeo_stream_push(EOF) failed (rc=${eofRc}): ${bridge.readError()}`
        );
      }
      drainRemux(true);
    }
    const closeRc = bridge.closeStream(handle);
    if (closeRc < 0) {
      throw new Error(
        `wedeo_stream_close failed (rc=${closeRc}): ${bridge.readError()}`
      );
    }
    const fullOut = Buffer.concat(outChunks);
    fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
    fs.writeFileSync(outPath, fullOut);
    return {
      bytes_written: fullOut.length,
      remux_chunks: remuxChunks,
      memory_bytes_before: memBefore,
      memory_bytes_after: bridge.memoryBytes(),
    };
  } finally {
    bridge.free(fmtAlloc.ptr, fmtAlloc.len);
    try {
      bridge.closeStream(handle);
    } catch {
      // Ignore already-closed handle errors.
    }
  }
}

function cmdStreamNegative(flags) {
  const wasmPath = requireFlag(flags, "wasm");
  const negCase = requireFlag(flags, "case");
  const bridge = new WasmAbiBridge(wasmPath);

  let trapped = false;
  let trapError = null;
  let rc = 0;
  let errorMessage = "";
  let postErrorSessionOk = false;
  const extra = {};

  try {
    if (negCase === "invalid-handle") {
      const rc1 = bridge.exports.wedeo_stream_inspect(-1);
      const err1 = bridge.readError();
      const rc2 = bridge.exports.wedeo_stream_next_frame(999999);
      const err2 = bridge.readError();
      const rc3 = bridge.exports.wedeo_stream_close(-42);
      const err3 = bridge.readError();
      rc = Math.max(rc1, rc2, rc3);
      errorMessage = [err1, err2, err3].filter(Boolean).join(" | ");
      extra.rcs = [rc1, rc2, rc3];

      if (flags.input) {
        const res = cmdStreamInspect({ wasm: wasmPath, input: flags.input });
        postErrorSessionOk = Boolean(res.inspect);
      } else {
        postErrorSessionOk = true;
      }
    } else if (negCase === "use-after-close") {
      const h = bridge.openStream("test.ivf");
      const closeRc = bridge.closeStream(h);
      if (closeRc < 0) {
        throw new Error(`Initial close failed: ${bridge.readError()}`);
      }
      const rcInspect = bridge.exports.wedeo_stream_inspect(h);
      const errInspect = bridge.readError();
      const rcFrame = bridge.exports.wedeo_stream_next_frame(h);
      const rcCloseAgain = bridge.exports.wedeo_stream_close(h);
      rc = Math.max(rcInspect, rcFrame, rcCloseAgain);
      errorMessage = errInspect || bridge.readError();
      extra.rcs = [rcInspect, rcFrame, rcCloseAgain];
      postErrorSessionOk = true;
    } else if (negCase === "null-ptr") {
      bridge.exports.wedeo_free(0, 0);
      bridge.exports.wedeo_free(0, 16);
      const h = bridge.openStream("null.mp4");
      rc = bridge.exports.wedeo_stream_push(h, 0, 16);
      errorMessage = bridge.readError();
      bridge.closeStream(h);
      postErrorSessionOk = true;
    } else if (
      negCase === "corrupt" ||
      negCase === "truncated" ||
      negCase === "empty"
    ) {
      const inputPath = flags.input ? String(flags.input) : null;
      const inputBuf = inputPath ? fs.readFileSync(inputPath) : Buffer.alloc(0);
      const name = flags.name
        ? String(flags.name)
        : inputPath
          ? path.basename(inputPath)
          : "empty.mp4";
      const h = bridge.openStream(name);
      if (inputBuf.length > 0) {
        const pushRc = bridge.pushChunk(h, inputBuf);
        if (pushRc < 0) {
          rc = pushRc;
          errorMessage = bridge.readError();
        }
      }
      if (rc >= 0) {
        const eofRc = bridge.pushChunk(h, Buffer.alloc(0));
        if (eofRc < 0) {
          rc = eofRc;
          errorMessage = bridge.readError();
        }
      }
      if (rc >= 0) {
        rc = bridge.exports.wedeo_stream_inspect(h);
        errorMessage = bridge.readError();
      }
      bridge.closeStream(h);
      postErrorSessionOk = true;
    } else if (negCase === "bad-format") {
      const inputPath = requireFlag(flags, "input");
      const inputBuf = fs.readFileSync(inputPath);
      const h = bridge.openStream(path.basename(inputPath));
      const fmtAlloc = bridge.allocStr("xyz");
      try {
        bridge.pushChunk(h, inputBuf);
        bridge.pushChunk(h, Buffer.alloc(0));
        rc = bridge.exports.wedeo_stream_next_remux_chunk(
          h,
          fmtAlloc.ptr,
          fmtAlloc.len
        );
        errorMessage = bridge.readError();
      } finally {
        bridge.free(fmtAlloc.ptr, fmtAlloc.len);
        bridge.closeStream(h);
      }
      postErrorSessionOk = true;
    } else if (negCase === "audio-next-frame") {
      const inputPath = requireFlag(flags, "input");
      const inputBuf = fs.readFileSync(inputPath);
      const h = bridge.openStream(path.basename(inputPath));
      bridge.pushChunk(h, inputBuf);
      bridge.pushChunk(h, Buffer.alloc(0));
      rc = bridge.exports.wedeo_stream_next_frame(h);
      errorMessage = bridge.readError();
      bridge.closeStream(h);
      postErrorSessionOk = true;
    } else if (negCase === "leak-loop") {
      const inputPath = requireFlag(flags, "input");
      const inputBuf = fs.readFileSync(inputPath);
      const memHistory = [];
      for (let i = 0; i < 20; i++) {
        const h = bridge.openStream(path.basename(inputPath));
        const pRc = bridge.pushChunk(h, inputBuf);
        if (pRc < 0) throw new Error(`push failed: ${bridge.readError()}`);
        bridge.pushChunk(h, Buffer.alloc(0));
        const iRc = bridge.exports.wedeo_stream_inspect(h);
        if (iRc !== 0) throw new Error(`inspect failed: ${bridge.readError()}`);
        while (true) {
          const fRc = bridge.exports.wedeo_stream_next_frame(h);
          if (fRc === 0) break;
          if (fRc < 0) throw new Error(`next_frame failed: ${bridge.readError()}`);
        }
        bridge.closeStream(h);
        memHistory.push(bridge.memoryBytes());
      }
      extra.memory_bytes_history = memHistory;
      postErrorSessionOk = true;
    } else {
      throw new Error(`Unknown --case ${negCase}`);
    }
  } catch (err) {
    trapped = true;
    trapError = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
  }

  return {
    case: negCase,
    trapped,
    trap_error: trapError,
    rc,
    error_message: errorMessage,
    post_error_session_ok: postErrorSessionOk,
    ...extra,
  };
}

function installWebCodecsMock(mode) {
  const trace = {
    mode,
    isConfigSupportedCalls: 0,
    configureCalls: 0,
    decodeCalls: 0,
    flushCalls: 0,
    closeCalls: 0,
    encoderIsConfigSupportedCalls: 0,
    configs: [],
  };

  if (mode === "native-absent") {
    delete globalThis.VideoDecoder;
    delete globalThis.VideoEncoder;
    delete globalThis.EncodedVideoChunk;
    delete globalThis.VideoFrame;
    return trace;
  }

  class MockEncodedVideoChunk {
    constructor(init) {
      this.type = init?.type ?? "key";
      this.timestamp = init?.timestamp ?? 0;
      this.duration = init?.duration ?? 0;
      this.byteLength = init?.data?.byteLength ?? 0;
      this._data = init?.data;
    }
    copyTo(dest) {
      if (this._data && dest) {
        new Uint8Array(dest.buffer || dest).set(new Uint8Array(this._data));
      }
    }
  }

  class MockVideoFrame {
    constructor(width, height, timestamp = 0) {
      this.codedWidth = width;
      this.codedHeight = height;
      this.displayWidth = width;
      this.displayHeight = height;
      this.timestamp = timestamp;
      this.format = "RGBA";
      this._closed = false;
      this._rgba = new Uint8Array(width * height * 4);
      this._rgba.fill(255);
    }
    allocationSize() {
      return this._rgba.byteLength;
    }
    async copyTo(dest) {
      new Uint8Array(dest.buffer || dest).set(this._rgba);
    }
    close() {
      this._closed = true;
    }
  }

  class MockVideoDecoder {
    constructor(init) {
      this._init = init;
      this.state = "unconfigured";
      this._width = 64;
      this._height = 64;
    }
    static async isConfigSupported(config) {
      trace.isConfigSupportedCalls++;
      trace.configs.push(config);
      if (mode === "reject") {
        throw new Error("Mocked WebCodecs isConfigSupported rejection");
      }
      const supported = mode === "supported" || mode === "decode-error";
      return { supported, config };
    }
    configure(config) {
      trace.configureCalls++;
      this.state = "configured";
      this._width = config?.codedWidth ?? 64;
      this._height = config?.codedHeight ?? 64;
    }
    decode(chunk) {
      trace.decodeCalls++;
      if (mode === "decode-error") {
        if (typeof this._init?.error === "function") {
          this._init.error(new Error("Mocked WebCodecs runtime decode error"));
        }
        return;
      }
      if (typeof this._init?.output === "function") {
        this._init.output(
          new MockVideoFrame(this._width, this._height, chunk?.timestamp ?? 0)
        );
      }
    }
    async flush() {
      trace.flushCalls++;
    }
    close() {
      trace.closeCalls++;
      this.state = "closed";
    }
  }

  class MockVideoEncoder {
    constructor(init) {
      this._init = init;
      this.state = "unconfigured";
    }
    static async isConfigSupported(config) {
      trace.encoderIsConfigSupportedCalls++;
      if (mode === "reject") {
        throw new Error("Mocked VideoEncoder isConfigSupported rejection");
      }
      return { supported: mode === "supported", config };
    }
    configure() {
      this.state = "configured";
    }
    encode() {}
    async flush() {}
    close() {
      this.state = "closed";
    }
  }

  globalThis.EncodedVideoChunk = MockEncodedVideoChunk;
  globalThis.VideoFrame = MockVideoFrame;
  globalThis.VideoDecoder = MockVideoDecoder;
  globalThis.VideoEncoder = MockVideoEncoder;
  return trace;
}

function resolvePackageEntry(pkgDir) {
  const pkgJsonPath = path.join(pkgDir, "package.json");
  if (!fs.existsSync(pkgJsonPath)) {
    throw new Error(`Missing @wedeo/wasm package.json at ${pkgJsonPath}`);
  }
  const pkg = JSON.parse(fs.readFileSync(pkgJsonPath, "utf8"));
  let relEntry = null;
  const dotExport = pkg.exports?.["."];
  if (typeof dotExport === "string") {
    relEntry = dotExport;
  } else if (dotExport && typeof dotExport === "object") {
    relEntry = dotExport.import ?? dotExport.default ?? dotExport.node;
  }
  relEntry = relEntry ?? pkg.module ?? pkg.main;
  if (!relEntry) {
    throw new Error("package.json has no resolvable exports/module/main entrypoint");
  }
  const resolved = path.resolve(pkgDir, relEntry);
  if (!fs.existsSync(resolved)) {
    throw new Error(`Package entrypoint does not exist: ${resolved}`);
  }
  return { pkg, entryPath: resolved };
}

function makeChunkedReadableStream(buf, chunkSize, options = {}) {
  let offset = 0;
  return new ReadableStream({
    pull(controller) {
      if (options.errorAfterFirstChunk && offset > 0) {
        controller.error(new Error("Simulated upstream ReadableStream failure"));
        return;
      }
      if (offset >= buf.length) {
        controller.close();
        return;
      }
      const slice = buf.subarray(offset, offset + chunkSize);
      offset += slice.length;
      controller.enqueue(new Uint8Array(slice));
    },
  });
}

async function cmdSdkTest(flags) {
  const pkgDir = requireFlag(flags, "pkg");
  const scenario = requireFlag(flags, "scenario");
  const webcodecsMode = flags["webcodecs-mode"]
    ? String(flags["webcodecs-mode"])
    : "native-absent";
  const chunkSize = intFlag(flags, "chunk-size", 1024);

  const trace = installWebCodecsMock(webcodecsMode);
  const { pkg, entryPath } = resolvePackageEntry(pkgDir);
  const sdk = await import(pathToFileURL(entryPath).href);

  const wasmPath = flags.wasm ? String(flags.wasm) : null;
  const wasmBytes = wasmPath ? new Uint8Array(fs.readFileSync(wasmPath)) : undefined;
  const wasmUrl = wasmPath ? pathToFileURL(path.resolve(wasmPath)).href : undefined;
  const initOpts = {};
  if (wasmBytes) {
    initOpts.wasmBytes = wasmBytes;
    initOpts.wasm = wasmBytes;
    initOpts.wasmUrl = wasmUrl;
  }

  if (scenario === "inspect-exports") {
    return {
      package_name: pkg.name,
      package_version: pkg.version,
      exported_keys: Object.keys(sdk).sort(),
    };
  }

  if (scenario === "inspect-stream") {
    const inputPath = requireFlag(flags, "input");
    const buf = fs.readFileSync(inputPath);
    const stream = makeChunkedReadableStream(buf, chunkSize);
    const session = await sdk.openMediaStream(stream, {
      ...initOpts,
      filename: path.basename(inputPath),
    });
    try {
      const inspect = await session.inspect();
      const keyframes = await session.extractKeyframes();
      const ts = numFlag(flags, "timestamp-sec", 0.5);
      const thumb = await session.extractThumbnail({ timestampSec: ts });
      if (flags.out && thumb?.rgba) {
        fs.mkdirSync(path.dirname(path.resolve(String(flags.out))), {
          recursive: true,
        });
        fs.writeFileSync(String(flags.out), new Uint8Array(thumb.rgba));
      }
      return {
        inspect,
        keyframes,
        thumbnail: thumb
          ? {
              width: thumb.width,
              height: thumb.height,
              ptsSec: thumb.ptsSec ?? thumb.pts_sec ?? null,
              rgbaBytes: thumb.rgba ? thumb.rgba.byteLength : 0,
            }
          : null,
        webcodecs: trace,
      };
    } finally {
      if (typeof session.close === "function") await session.close();
      else if (typeof session.dispose === "function") await session.dispose();
    }
  }

  if (scenario === "decode-hybrid") {
    const inputPath = requireFlag(flags, "input");
    const buf = fs.readFileSync(inputPath);
    const stream = makeChunkedReadableStream(buf, chunkSize);
    const dec = await sdk.createVideoDecoder({
      ...initOpts,
      filename: path.basename(inputPath),
      codec: flags.codec ? String(flags.codec) : undefined,
      codedWidth: flags.width ? intFlag(flags, "width", 128) : undefined,
      codedHeight: flags.height ? intFlag(flags, "height", 128) : undefined,
    });
    const decodedFrames = [];
    const rgbaChunks = [];
    try {
      if (typeof dec.decodeStream === "function") {
        for await (const frame of dec.decodeStream(stream)) {
          decodedFrames.push({
            width: frame.width ?? frame.codedWidth,
            height: frame.height ?? frame.codedHeight,
            ptsSec: frame.ptsSec ?? frame.pts_sec ?? null,
          });
          if (frame.rgba) rgbaChunks.push(Buffer.from(frame.rgba));
        }
      } else if (typeof dec.decode === "function") {
        const res = await dec.decode(stream);
        const list = Array.isArray(res) ? res : res?.frames ?? [];
        for (const frame of list) {
          decodedFrames.push({
            width: frame.width ?? frame.codedWidth,
            height: frame.height ?? frame.codedHeight,
            ptsSec: frame.ptsSec ?? frame.pts_sec ?? null,
          });
          if (frame.rgba) rgbaChunks.push(Buffer.from(frame.rgba));
        }
      } else {
        throw new Error("createVideoDecoder returned object without decodeStream/decode");
      }
      if (flags.out && rgbaChunks.length > 0) {
        fs.mkdirSync(path.dirname(path.resolve(String(flags.out))), {
          recursive: true,
        });
        fs.writeFileSync(String(flags.out), Buffer.concat(rgbaChunks));
      }
      return {
        backend: dec.backend ?? dec.mode ?? null,
        frame_count: decodedFrames.length,
        frames: decodedFrames,
        total_rgba_bytes: rgbaChunks.reduce((acc, b) => acc + b.length, 0),
        webcodecs: trace,
      };
    } finally {
      if (typeof dec.close === "function") await dec.close();
      else if (typeof dec.dispose === "function") await dec.dispose();
    }
  }

  if (scenario === "remux-stream") {
    const inputPath = requireFlag(flags, "input");
    const format = requireFlag(flags, "format");
    const outPath = requireFlag(flags, "out");
    const buf = fs.readFileSync(inputPath);
    const readable = makeChunkedReadableStream(buf, chunkSize);
    const chunks = [];
    const writable = new WritableStream({
      write(chunk) {
        chunks.push(Buffer.from(chunk));
      },
    });
    const outReadable = await sdk.remuxStream(readable, {
      ...initOpts,
      filename: path.basename(inputPath),
      format,
    });
    await outReadable.pipeTo(writable);
    const full = Buffer.concat(chunks);
    fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
    fs.writeFileSync(outPath, full);
    return {
      bytes_written: full.length,
      chunks_written: chunks.length,
    };
  }

  if (scenario === "wedeo-lifecycle") {
    const inputPath = requireFlag(flags, "input");
    const buf = fs.readFileSync(inputPath);
    const wedeo = await sdk.Wedeo.create({ ...initOpts });
    try {
      const inspect = await wedeo.inspect(new Uint8Array(buf), path.basename(inputPath));
      const keyframes = await wedeo.extractKeyframes(
        new Uint8Array(buf),
        path.basename(inputPath)
      );
      return {
        inspect,
        keyframes,
        t0: performance.now(),
      };
    } finally {
      if (typeof wedeo.close === "function") await wedeo.close();
      else if (typeof wedeo.dispose === "function") await wedeo.dispose();
    }
  }

  if (
    scenario === "abort-stream" ||
    scenario === "errored-stream" ||
    scenario === "empty-stream" ||
    scenario === "corrupt-stream" ||
    scenario === "bad-init" ||
    scenario === "bad-remux-format"
  ) {
    let rejected = false;
    let errorMessage = "";
    try {
      if (scenario === "abort-stream") {
        const inputPath = requireFlag(flags, "input");
        const buf = fs.readFileSync(inputPath);
        const controller = new AbortController();
        const stream = makeChunkedReadableStream(buf, 512);
        controller.abort(new Error("User aborted stream"));
        const session = await sdk.openMediaStream(stream, {
          ...initOpts,
          filename: path.basename(inputPath),
          signal: controller.signal,
        });
        await session.inspect();
      } else if (scenario === "errored-stream") {
        const inputPath = requireFlag(flags, "input");
        const buf = fs.readFileSync(inputPath);
        const stream = makeChunkedReadableStream(buf, 512, {
          errorAfterFirstChunk: true,
        });
        const session = await sdk.openMediaStream(stream, {
          ...initOpts,
          filename: path.basename(inputPath),
        });
        await session.inspect();
      } else if (scenario === "empty-stream") {
        const stream = makeChunkedReadableStream(Buffer.alloc(0), 512);
        const session = await sdk.openMediaStream(stream, {
          ...initOpts,
          filename: "empty.mp4",
        });
        await session.inspect();
      } else if (scenario === "corrupt-stream") {
        const garbage = Buffer.from("not_a_media_container_0123456789".repeat(16));
        const stream = makeChunkedReadableStream(garbage, 128);
        const session = await sdk.openMediaStream(stream, {
          ...initOpts,
          filename: "corrupt.mp4",
        });
        await session.inspect();
      } else if (scenario === "bad-init") {
        await sdk.Wedeo.create({ wasmBytes: new Uint8Array([0, 1, 2, 3]) });
      } else if (scenario === "bad-remux-format") {
        const inputPath = requireFlag(flags, "input");
        const buf = fs.readFileSync(inputPath);
        const stream = makeChunkedReadableStream(buf, 1024);
        const outReadable = await sdk.remuxStream(stream, {
          ...initOpts,
          filename: path.basename(inputPath),
          format: "avi",
        });
        const reader = outReadable.getReader();
        while (!(await reader.read()).done) {
          // drain
        }
      }
    } catch (err) {
      rejected = true;
      errorMessage = err instanceof Error ? err.message : String(err);
    }
    return {
      scenario,
      rejected,
      error_message: errorMessage,
    };
  }

  throw new Error(`Unknown sdk-test scenario: ${scenario}`);
}

async function main() {
  const { command, flags } = parseCliArgs(process.argv.slice(2));
  let result;
  if (command === "probe-wasm") {
    result = cmdProbeWasm(flags);
  } else if (command === "stream-inspect") {
    result = cmdStreamInspect(flags);
  } else if (command === "stream-decode-video") {
    result = cmdStreamDecodeVideo(flags);
  } else if (command === "stream-remux") {
    result = cmdStreamRemux(flags);
  } else if (command === "stream-negative") {
    result = cmdStreamNegative(flags);
  } else if (command === "sdk-test") {
    result = await cmdSdkTest(flags);
  } else {
    throw new Error(`Unknown command: ${command}`);
  }
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

main().catch((err) => {
  const msg = err instanceof Error ? err.stack || err.message : String(err);
  process.stderr.write(`${msg}\n`);
  process.exit(1);
});
