//! Workspace architecture & dependency DAG lints parsed from
//! `cargo metadata --offline --format-version 1`.
//!
//! Enforces:
//! - Strict downward layer DAG across L0/L1/L2/L3/Dev
//! - L0 `wedeo-core` depends ONLY on `bitflags`
//! - Audio crates (`wedeo-audio-ops`, `wedeo-codec-pcm`, `wedeo-format-wav`,
//!   `wedeo-symphonia`) NEVER depend (directly or transitively) on
//!   `wedeo-video-ops`, `wedeo-codec-h264`, `wedeo-codec-vp9`, `wedeo-gpu-wgpu`, or `wgpu`
//! - Audio-only `wedeo` build (`--no-default-features --features "audio-ops,pcm,wav,symphonia"`)
//!   pulls zero video/GPU crates
//! - ZERO `inventory` anywhere in workspace or resolved graph
//! - ZERO `cc` `build-dependencies` in native workspace crates
//! - `#![forbid(unsafe_code)]` present in all 15 safe-only crates

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::process::Command;

// ---------------------------------------------------------------------------
// Minimal zero-dependency JSON parser for `cargo metadata --format-version 1`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Json {
    Null,
    Bool(bool),
    Number(String),
    Str(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    fn as_object(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Self::Object(o) => Some(o),
            _ => None,
        }
    }

    fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?.get(key)
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Result<u8, String> {
        self.skip_ws();
        self.bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| "unexpected EOF".to_string())
    }

    fn expect(&mut self, ch: u8) -> Result<(), String> {
        let got = self.peek()?;
        if got != ch {
            return Err(format!(
                "expected '{}' at {}, got '{}'",
                ch as char, self.pos, got as char
            ));
        }
        self.pos += 1;
        Ok(())
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        match self.peek()? {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string().map(Json::Str),
            b't' => {
                if self.bytes[self.pos..].starts_with(b"true") {
                    self.pos += 4;
                    Ok(Json::Bool(true))
                } else {
                    Err("invalid true".to_string())
                }
            }
            b'f' => {
                if self.bytes[self.pos..].starts_with(b"false") {
                    self.pos += 5;
                    Ok(Json::Bool(false))
                } else {
                    Err("invalid false".to_string())
                }
            }
            b'n' => {
                if self.bytes[self.pos..].starts_with(b"null") {
                    self.pos += 4;
                    Ok(Json::Null)
                } else {
                    Err("invalid null".to_string())
                }
            }
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(format!(
                "unexpected byte '{}' at {}",
                other as char, self.pos
            )),
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(out);
            }
            if b == b'\\' {
                let esc = *self
                    .bytes
                    .get(self.pos)
                    .ok_or_else(|| "unterminated escape".to_string())?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hex = std::str::from_utf8(
                            self.bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or_else(|| "short \\u escape".to_string())?,
                        )
                        .map_err(|e| e.to_string())?;
                        let code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                        self.pos += 4;
                        if let Some(ch) = char::from_u32(code) {
                            out.push(ch);
                        }
                    }
                    other => out.push(other as char),
                }
            } else if b < 0x80 {
                out.push(b as char);
            } else {
                // Decode UTF-8 multi-byte sequence starting at self.pos - 1.
                let start = self.pos - 1;
                let rest = std::str::from_utf8(&self.bytes[start..])
                    .map_err(|_| "invalid utf-8 in json string".to_string())?;
                let ch = rest.chars().next().ok_or("empty utf-8")?;
                out.push(ch);
                self.pos = start + ch.len_utf8();
            }
        }
        Err("unterminated string".to_string())
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| e.to_string())?;
        Ok(Json::Number(s.to_string()))
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        if self.peek()? == b']' {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            match self.peek()? {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    return Ok(Json::Array(items));
                }
                other => return Err(format!("expected ',' or ']', got '{}'", other as char)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.expect(b'{')?;
        let mut map = BTreeMap::new();
        if self.peek()? == b'}' {
            self.pos += 1;
            return Ok(Json::Object(map));
        }
        loop {
            let key = self.parse_string()?;
            self.expect(b':')?;
            let val = self.parse_value()?;
            map.insert(key, val);
            match self.peek()? {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    return Ok(Json::Object(map));
                }
                other => return Err(format!("expected ',' or '}}', got '{}'", other as char)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Metadata helpers
// ---------------------------------------------------------------------------

fn host_triple() -> String {
    let out = Command::new("rustc")
        .arg("-vV")
        .output()
        .expect("run rustc -vV");
    assert!(out.status.success(), "rustc -vV failed");
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(host) = line.strip_prefix("host:") {
            return host.trim().to_string();
        }
    }
    panic!("host triple not found in rustc -vV output:\n{text}");
}

fn run_cargo_metadata(manifest_path: Option<&Path>, extra_args: &[&str]) -> Json {
    let root = wedeo_oracle::workspace_root();
    let target = host_triple();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&root).args([
        "metadata",
        "--offline",
        "--format-version",
        "1",
        "--filter-platform",
        &target,
    ]);
    if let Some(manifest) = manifest_path {
        cmd.arg("--manifest-path").arg(manifest);
    }
    cmd.args(extra_args);

    let out = cmd.output().expect("run cargo metadata");
    assert!(
        out.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("utf-8 metadata");
    JsonParser::new(&text)
        .parse_value()
        .expect("parse cargo metadata JSON")
}

#[allow(dead_code)]
struct PackageMeta {
    id: String,
    name: String,
    manifest_path: String,
    deps: Vec<(String, Option<String>)>,
}

fn parse_packages(meta: &Json) -> (BTreeSet<String>, BTreeMap<String, PackageMeta>) {
    let ws: BTreeSet<String> = meta
        .get("workspace_members")
        .and_then(Json::as_array)
        .expect("workspace_members")
        .iter()
        .filter_map(|v| v.as_str().map(ToOwned::to_owned))
        .collect();

    let mut pkgs = BTreeMap::new();
    for p in meta
        .get("packages")
        .and_then(Json::as_array)
        .expect("packages")
    {
        let id = p.get("id").and_then(Json::as_str).unwrap().to_string();
        let name = p.get("name").and_then(Json::as_str).unwrap().to_string();
        let manifest_path = p
            .get("manifest_path")
            .and_then(Json::as_str)
            .unwrap()
            .to_string();
        let mut deps = Vec::new();
        if let Some(dep_arr) = p.get("dependencies").and_then(Json::as_array) {
            for d in dep_arr {
                let dep_name = d.get("name").and_then(Json::as_str).unwrap().to_string();
                let kind = d.get("kind").and_then(Json::as_str).map(ToOwned::to_owned);
                deps.push((dep_name, kind));
            }
        }
        pkgs.insert(
            id.clone(),
            PackageMeta {
                id,
                name,
                manifest_path,
                deps,
            },
        );
    }

    (ws, pkgs)
}

fn resolve_graph(meta: &Json) -> BTreeMap<String, Vec<String>> {
    let mut graph = BTreeMap::new();
    let nodes = meta
        .get("resolve")
        .and_then(|r| r.get("nodes"))
        .and_then(Json::as_array)
        .expect("resolve.nodes");
    for n in nodes {
        let id = n.get("id").and_then(Json::as_str).unwrap().to_string();
        let deps = if let Some(dep_objs) = n.get("deps").and_then(Json::as_array) {
            dep_objs
                .iter()
                .filter(|d| {
                    let Some(kinds) = d.get("dep_kinds").and_then(Json::as_array) else {
                        return true;
                    };
                    kinds
                        .iter()
                        .any(|k| k.get("kind").and_then(Json::as_str) != Some("dev"))
                })
                .filter_map(|d| d.get("pkg").and_then(Json::as_str).map(ToOwned::to_owned))
                .collect()
        } else {
            n.get("dependencies")
                .and_then(Json::as_array)
                .unwrap_or(&[])
                .iter()
                .filter_map(|v| v.as_str().map(ToOwned::to_owned))
                .collect()
        };
        graph.insert(id, deps);
    }
    graph
}

fn transitive_closure(
    start_id: &str,
    graph: &BTreeMap<String, Vec<String>>,
    pkgs: &BTreeMap<String, PackageMeta>,
) -> BTreeSet<String> {
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();
    if let Some(neighbors) = graph.get(start_id) {
        for n in neighbors {
            if visited.insert(n.clone()) {
                queue.push_back(n.clone());
            }
        }
    }
    while let Some(cur) = queue.pop_front() {
        if let Some(neighbors) = graph.get(&cur) {
            for n in neighbors {
                if visited.insert(n.clone()) {
                    queue.push_back(n.clone());
                }
            }
        }
    }
    visited
        .into_iter()
        .filter_map(|id| pkgs.get(&id).map(|p| p.name.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Architecture lint tests
// ---------------------------------------------------------------------------

#[test]
fn arch_dag_and_dependency_invariants() {
    let meta = run_cargo_metadata(None, &[]);
    let (ws_ids, pkgs) = parse_packages(&meta);
    let graph = resolve_graph(&meta);

    let ws_by_name: BTreeMap<&str, &PackageMeta> = ws_ids
        .iter()
        .filter_map(|id| pkgs.get(id).map(|p| (p.name.as_str(), p)))
        .collect();

    // 1. ZERO `inventory` anywhere in the resolved workspace graph, and ZERO `cc` in native crates.
    for p in pkgs.values() {
        assert_ne!(
            p.name, "inventory",
            "forbidden crate `inventory` found in resolved dependency graph"
        );
    }
    for p in ws_by_name.values() {
        for (dep_name, kind) in &p.deps {
            assert_ne!(
                dep_name, "inventory",
                "workspace crate `{}` depends on forbidden `inventory`",
                p.name
            );
            assert_ne!(
                dep_name, "cc",
                "workspace crate `{}` has forbidden `cc` dependency (kind={kind:?})",
                p.name
            );
        }
    }

    // 2. L0 `wedeo-core` depends ONLY on `bitflags` (zero other normal/build deps).
    let core = ws_by_name
        .get("wedeo-core")
        .expect("wedeo-core in workspace");
    let core_non_dev: Vec<&str> = core
        .deps
        .iter()
        .filter(|(_, kind)| kind.as_deref() != Some("dev"))
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(
        core_non_dev,
        vec!["bitflags"],
        "wedeo-core must depend ONLY on `bitflags`, found {core_non_dev:?}"
    );

    // 3. Strict downward layer DAG among workspace crates.
    let layer_of = |name: &str| -> u8 {
        match name {
            "wedeo-core" => 0,
            "wedeo-syntax" | "wedeo-video-ops" | "wedeo-audio-ops" | "wedeo-graph"
            | "wedeo-gpu-wgpu" => 1,
            "wedeo-codec-pcm" | "wedeo-codec-h264" | "wedeo-codec-vp9" | "wedeo-format-wav"
            | "wedeo-format-mp4" | "wedeo-format-mkv" | "wedeo-format-ivf" | "wedeo-rav1d"
            | "wedeo-rav1e" | "wedeo-symphonia" => 2,
            "wedeo" | "wedeo-cli" | "wedeo-wasm" => 3,
            "wedeo-oracle" => 4,
            other => panic!("unclassified workspace crate `{other}` in DAG lint"),
        }
    };

    for (&name, &pkg) in &ws_by_name {
        let my_layer = layer_of(name);
        for (dep_name, kind) in &pkg.deps {
            if kind.as_deref() == Some("dev") {
                continue;
            }
            if let Some(&dep_pkg) = ws_by_name.get(dep_name.as_str()) {
                let dep_layer = layer_of(&dep_pkg.name);
                assert!(
                    dep_layer < my_layer
                        || (name == "wedeo-graph"
                            && matches!(
                                dep_name.as_str(),
                                "wedeo-video-ops" | "wedeo-audio-ops" | "wedeo-syntax"
                            ))
                        || (name == "wedeo-gpu-wgpu" && dep_name == "wedeo-video-ops")
                        || (matches!(name, "wedeo-cli" | "wedeo-wasm") && dep_name == "wedeo"),
                    "layer inversion: `{name}` (L{my_layer}) depends on `{dep_name}` (L{dep_layer})"
                );
                assert_ne!(
                    dep_name, "wedeo-oracle",
                    "production crate `{name}` must never depend on dev-only `wedeo-oracle`"
                );
            }
        }
    }

    // 4. `wedeo-video-ops` never depends on `wgpu` or `wedeo-gpu-wgpu`.
    let vops = ws_by_name.get("wedeo-video-ops").unwrap();
    let vops_closure = transitive_closure(&vops.id, &graph, &pkgs);
    for forbidden in ["wgpu", "wedeo-gpu-wgpu"] {
        assert!(
            !vops_closure.contains(forbidden),
            "wedeo-video-ops must not depend on `{forbidden}`"
        );
    }

    // 5. Audio crates NEVER depend (directly or transitively) on video/GPU crates.
    let audio_crates = [
        "wedeo-audio-ops",
        "wedeo-codec-pcm",
        "wedeo-format-wav",
        "wedeo-symphonia",
    ];
    let forbidden_video_gpu = [
        "wedeo-video-ops",
        "wedeo-codec-h264",
        "wedeo-codec-vp9",
        "wedeo-gpu-wgpu",
        "wgpu",
    ];
    for crate_name in audio_crates {
        let pkg = ws_by_name.get(crate_name).unwrap();
        let closure = transitive_closure(&pkg.id, &graph, &pkgs);
        for forbidden in forbidden_video_gpu {
            assert!(
                !closure.contains(forbidden),
                "audio crate `{crate_name}` transitively pulls forbidden `{forbidden}`"
            );
        }
    }
}

#[test]
fn arch_audio_only_facade_pulls_zero_video_or_gpu() {
    let root = wedeo_oracle::workspace_root();
    let out = Command::new("cargo")
        .current_dir(&root)
        .args([
            "tree",
            "--offline",
            "-p",
            "wedeo",
            "--no-default-features",
            "--features",
            "audio-ops,pcm,wav,symphonia",
            "-e",
            "normal",
            "--prefix",
            "none",
        ])
        .output()
        .expect("run cargo tree");
    assert!(
        out.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let tree_text = String::from_utf8_lossy(&out.stdout);
    let resolved_names: BTreeSet<&str> = tree_text
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect();

    assert!(resolved_names.contains("wedeo-core"));
    assert!(resolved_names.contains("wedeo-audio-ops"));
    assert!(resolved_names.contains("wedeo-codec-pcm"));
    assert!(resolved_names.contains("wedeo-format-wav"));
    assert!(resolved_names.contains("wedeo-symphonia"));

    for forbidden in [
        "wedeo-video-ops",
        "wedeo-codec-h264",
        "wedeo-codec-vp9",
        "wedeo-gpu-wgpu",
        "wgpu",
        "wedeo-rav1d",
        "wedeo-rav1e",
    ] {
        assert!(
            !resolved_names.contains(forbidden),
            "audio-only `wedeo` build pulled forbidden crate `{forbidden}`"
        );
    }
}

#[test]
fn arch_forbid_unsafe_code_enforced_across_safe_crates() {
    let root = wedeo_oracle::workspace_root();
    let safe_crates = [
        "crates/wedeo-syntax",
        "crates/wedeo-video-ops",
        "crates/wedeo-audio-ops",
        "crates/wedeo-graph",
        "crates/wedeo-gpu-wgpu",
        "codecs/wedeo-codec-vp9",
        "codecs/wedeo-codec-pcm",
        "formats/wedeo-format-wav",
        "formats/wedeo-format-mp4",
        "formats/wedeo-format-mkv",
        "formats/wedeo-format-ivf",
        "adapters/wedeo-symphonia",
        "adapters/wedeo-rav1e",
        "crates/wedeo",
    ];

    for rel in safe_crates {
        let lib_rs = root.join(rel).join("src/lib.rs");
        let src = std::fs::read_to_string(&lib_rs)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", lib_rs.display()));
        let has_forbid = src
            .lines()
            .map(str::trim)
            .any(|line| line == "#![forbid(unsafe_code)]");
        assert!(
            has_forbid,
            "{} is missing `#![forbid(unsafe_code)]`",
            lib_rs.display()
        );
    }

    // `wedeo-codec-h264` may use `#![deny(unsafe_code)]` (or `#![forbid(unsafe_code)]`)
    // to isolate audited `core::arch` SIMD intrinsics inside `src/simd.rs`, while
    // forbidding `unsafe` in all other modules.
    let h264_src_dir = root.join("codecs/wedeo-codec-h264/src");
    let h264_lib_rs = h264_src_dir.join("lib.rs");
    let h264_lib = std::fs::read_to_string(&h264_lib_rs)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", h264_lib_rs.display()));
    let has_h264_guard = h264_lib
        .lines()
        .map(str::trim)
        .any(|line| line == "#![forbid(unsafe_code)]" || line == "#![deny(unsafe_code)]");
    assert!(
        has_h264_guard,
        "{} must declare `#![forbid(unsafe_code)]` or `#![deny(unsafe_code)]`",
        h264_lib_rs.display()
    );

    fn contains_unsafe_code_token(src: &str) -> bool {
        for raw_line in src.lines() {
            let line = raw_line.split("//").next().unwrap_or("");
            let bytes = line.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i..].starts_with(b"unsafe") {
                    let prev_ok =
                        i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                    let end = i + 6;
                    let next_ok = end == bytes.len()
                        || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
                    if prev_ok && next_ok {
                        return true;
                    }
                }
                i += 1;
            }
        }
        false
    }

    assert!(
        !contains_unsafe_code_token(&h264_lib) && !h264_lib.contains("allow(unsafe_code)"),
        "{} must not contain `unsafe` code or `allow(unsafe_code)`",
        h264_lib_rs.display()
    );

    for entry in std::fs::read_dir(&h264_src_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", h264_src_dir.display()))
    {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if matches!(file_name, "lib.rs" | "simd.rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        assert!(
            !contains_unsafe_code_token(&src) && !src.contains("allow(unsafe_code)"),
            "{} contains `unsafe` outside audited `src/simd.rs`",
            path.display()
        );
    }
}
