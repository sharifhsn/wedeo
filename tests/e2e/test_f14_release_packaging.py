"""E2E tests for F14: Release Packaging & Distribution (crates.io, Homebrew, GH Release, npm)."""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import tarfile
import tempfile
import tomllib
import unittest
from pathlib import Path
from typing import Any

from lib.harness import (
    augmented_env,
    e2e_test,
    find_node_bin,
    repo_root,
    run_cmd,
    run_wedeo,
)

WASM_SIZE_LIMIT_BYTES = 985 * 1024
STALE_V1_CRATES = (
    "wedeo-codec",
    "wedeo-format",
    "wedeo-filter",
    "wedeo-resample",
    "wedeo-scale",
    "wedeo-format-h264",
    "wedeo-play",
    "wedeo-fate",
)
STALE_V1_API_SYMBOLS = (
    "InputContext",
    "DecoderBuilder",
    "CodecParameters",
    "send_packet",
    "receive_frame",
)


def _cargo_metadata(root: Path) -> dict[str, Any]:
    res = run_cmd(
        ["cargo", "metadata", "--no-deps", "--offline", "--format-version", "1"],
        cwd=root,
    )
    if res.returncode != 0:
        raise AssertionError(f"cargo metadata failed: {res.stderr}")
    return json.loads(res.stdout)


def _is_publishable(pkg: dict[str, Any]) -> bool:
    pub = pkg.get("publish")
    return pub is None or "crates-io" in pub


def _parse_publish_sh_crates(script_text: str) -> list[str]:
    m = re.search(r"CRATES=\(\s*(.*?)\s*\)", script_text, flags=re.DOTALL)
    if not m:
        raise AssertionError("Could not find CRATES=(...) array in scripts/publish.sh")
    crates: list[str] = []
    for raw_line in m.group(1).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.split():
            crates.append(token.strip("\"'"))
    return crates


class TestF14ReleasePackaging(unittest.TestCase):
    """Tier 1 and Tier 2 E2E tests for F14 Release Packaging & Distribution."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="wedeo_e2e_f14_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)
        self.root = repo_root()
        self.root_cargo = tomllib.loads(
            (self.root / "Cargo.toml").read_text(encoding="utf-8")
        )
        self.ws_pkg = self.root_cargo["workspace"]["package"]

    @e2e_test(feature="F14", tier=1, milestone="M6")
    def test_t1_workspace_publishable_metadata_and_cargo_package(self) -> None:
        meta = _cargo_metadata(self.root)
        ws_ids = set(meta["workspace_members"])
        publishable = [
            p for p in meta["packages"] if p["id"] in ws_ids and _is_publishable(p)
        ]
        self.assertGreaterEqual(len(publishable), 15)

        expected_ver = self.ws_pkg["version"]
        expected_lic = self.ws_pkg["license"]
        expected_repo = self.ws_pkg["repository"]

        for pkg in publishable:
            self.assertEqual(pkg["version"], expected_ver, pkg["name"])
            self.assertEqual(pkg.get("license"), expected_lic, pkg["name"])
            self.assertEqual(pkg.get("repository"), expected_repo, pkg["name"])
            self.assertEqual(pkg.get("edition"), "2024", pkg["name"])
            self.assertTrue((pkg.get("description") or "").strip(), pkg["name"])

            listed = run_cmd(
                [
                    "cargo",
                    "package",
                    "--list",
                    "--offline",
                    "--allow-dirty",
                    "-p",
                    pkg["name"],
                ],
                cwd=self.root,
            )
            self.assertEqual(listed.returncode, 0, f"{pkg['name']}: {listed.stderr}")
            files = set(listed.stdout.splitlines())
            self.assertIn("Cargo.toml", files, pkg["name"])
            self.assertIn("src/lib.rs", files, pkg["name"])

        pkg_target = self.tmp / "pkg_target"
        tar_res = run_cmd(
            [
                "cargo",
                "package",
                "--no-verify",
                "--offline",
                "--allow-dirty",
                "--target-dir",
                pkg_target,
                "-p",
                "wedeo-core",
            ],
            cwd=self.root,
        )
        self.assertEqual(tar_res.returncode, 0, tar_res.stderr)
        crate_file = pkg_target / "package" / f"wedeo-core-{expected_ver}.crate"
        self.assertTrue(crate_file.is_file(), f"Missing .crate archive: {crate_file}")
        with tarfile.open(crate_file, "r:gz") as tf:
            names = set(tf.getnames())
            self.assertIn(f"wedeo-core-{expected_ver}/Cargo.toml", names)
            self.assertIn(f"wedeo-core-{expected_ver}/src/lib.rs", names)

    @e2e_test(feature="F14", tier=1, milestone="M6")
    def test_t1_publish_script_topological_order_and_dry_run(self) -> None:
        script = self.root / "scripts" / "publish.sh"
        self.assertTrue(script.is_file(), f"Missing {script}")
        self.assertTrue(os.access(script, os.X_OK), f"{script} must be executable")

        dry = run_cmd([str(script)], cwd=self.root)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("[dry-run]", dry.stdout)

        crates = _parse_publish_sh_crates(script.read_text(encoding="utf-8"))
        meta = _cargo_metadata(self.root)
        ws_ids = set(meta["workspace_members"])
        pub_pkgs = {
            p["name"]: p
            for p in meta["packages"]
            if p["id"] in ws_ids and _is_publishable(p)
        }
        self.assertEqual(set(crates), set(pub_pkgs.keys()))

        pos = {name: idx for idx, name in enumerate(crates)}
        for name, pkg in pub_pkgs.items():
            for dep in pkg["dependencies"]:
                if dep.get("kind") == "dev":
                    continue
                dep_name = dep["name"]
                if dep_name in pub_pkgs:
                    self.assertLess(
                        pos[dep_name],
                        pos[name],
                        f"Topological order violation: {dep_name} must precede {name}",
                    )

    @e2e_test(feature="F14", tier=1, milestone="M6")
    def test_t1_homebrew_formula_structure_and_install_test_block(self) -> None:
        formula = self.root / "Formula" / "wedeo.rb"
        self.assertTrue(formula.is_file(), f"Missing Homebrew formula: {formula}")
        text = formula.read_text(encoding="utf-8")

        self.assertRegex(text, r"class\s+Wedeo\s+<\s+Formula")
        self.assertRegex(text, re.compile(r'^\s*desc\s+"[^"]+"', re.MULTILINE))
        self.assertIn("https://github.com/sharifhsn/wedeo", text)
        self.assertRegex(text, r'license\s+"LGPL-2\.1-or-later"')
        self.assertRegex(text, r'sha256\s+"[0-9a-f]{64}"')
        self.assertIn("def install", text)
        self.assertIn("test do", text)
        self.assertRegex(text, r"wedeo.*--version")

        ruby_bin = shutil.which("ruby")
        if ruby_bin is not None:
            chk = run_cmd([ruby_bin, "-c", formula])
            self.assertEqual(chk.returncode, 0, chk.stderr)

    @e2e_test(feature="F14", tier=1, milestone="M6")
    def test_t1_github_workflows_ci_and_release(self) -> None:
        wf_dir = self.root / ".github" / "workflows"
        ci_yml = wf_dir / "ci.yml"
        rel_yml = wf_dir / "release.yml"
        self.assertTrue(ci_yml.is_file(), f"Missing {ci_yml}")
        self.assertTrue(rel_yml.is_file(), f"Missing {rel_yml}")

        ci_text = ci_yml.read_text(encoding="utf-8")
        for token in (
            "branches: [main]",
            "clippy",
            "fmt",
            "oracle_arch_lint",
            "oracle_dod_alloc",
            "wasm",
            "miri",
            "tests/e2e/run.py",
        ):
            self.assertIn(token, ci_text, f"ci.yml missing {token!r}")

        rel_text = rel_yml.read_text(encoding="utf-8")
        self.assertRegex(rel_text, r"tags:\s*\n\s*-\s*['\"]?v\*")
        self.assertIn("SHA256SUMS", rel_text)
        self.assertIn("wedeo", rel_text)
        self.assertIn("wasm", rel_text)

    @e2e_test(feature="F14", tier=1, milestone="M6")
    def test_t1_changelog_and_readme_alignment(self) -> None:
        ver = self.ws_pkg["version"]
        changelog = self.root / "CHANGELOG.md"
        self.assertTrue(changelog.is_file(), f"Missing {changelog}")
        cl_text = changelog.read_text(encoding="utf-8")
        self.assertIn(ver, cl_text)
        self.assertGreater(len(cl_text.strip().splitlines()), 3)

        crate_readme = (self.root / "crates" / "wedeo" / "README.md").read_text(
            encoding="utf-8"
        )
        root_readme = (self.root / "README.md").read_text(encoding="utf-8")
        for sym in ("AudioReader", "VideoReader", "LGPL-2.1-or-later"):
            self.assertIn(sym, crate_readme)
            self.assertIn(sym, root_readme)
        self.assertIn("default_registry", crate_readme)

    @e2e_test(feature="F14", tier=1, milestone="M6")
    def test_t1_npm_wedeo_wasm_package_json_and_files(self) -> None:
        pkg_dir = self.root / "packages" / "wedeo-wasm"
        pkg_json = pkg_dir / "package.json"
        self.assertTrue(pkg_json.is_file(), f"Missing {pkg_json}")
        pkg = json.loads(pkg_json.read_text(encoding="utf-8"))

        self.assertEqual(pkg.get("name"), "@wedeo/wasm")
        self.assertEqual(pkg.get("version"), self.ws_pkg["version"])
        self.assertEqual(pkg.get("license"), self.ws_pkg["license"])
        self.assertEqual(pkg.get("type"), "module")
        self.assertIn("sharifhsn/wedeo", json.dumps(pkg.get("repository", "")))

        dot = pkg.get("exports", {}).get(".", {})
        entry_rel = (
            dot
            if isinstance(dot, str)
            else dot.get("import") or dot.get("default") or pkg.get("main")
        )
        self.assertTrue(entry_rel, "package.json missing entrypoint")
        entry_path = (pkg_dir / entry_rel).resolve()
        self.assertTrue(entry_path.is_file(), f"Missing JS entry: {entry_path}")

        types_rel = pkg.get("types") or (
            dot.get("types") if isinstance(dot, dict) else None
        )
        self.assertTrue(types_rel, "package.json missing types field")
        self.assertTrue((pkg_dir / types_rel).is_file())

        node = find_node_bin()
        chk = run_cmd([node, "--check", entry_path])
        self.assertEqual(chk.returncode, 0, chk.stderr)

    @e2e_test(feature="F14", tier=2, milestone="M6")
    def test_t2_publish_graph_decoupled_from_publish_false_and_git_deps(self) -> None:
        meta = _cargo_metadata(self.root)
        ws_ids = set(meta["workspace_members"])
        ws_by_name = {p["name"]: p for p in meta["packages"] if p["id"] in ws_ids}
        publishable = [p for p in ws_by_name.values() if _is_publishable(p)]
        non_publishable_names = {
            p["name"] for p in ws_by_name.values() if not _is_publishable(p)
        }

        for pkg in publishable:
            for dep in pkg["dependencies"]:
                if dep.get("kind") == "dev":
                    continue
                dep_name = dep["name"]
                src = dep.get("source") or ""
                self.assertFalse(
                    src.startswith("git+"),
                    f"Publishable crate {pkg['name']} has git dep {dep_name}: {src}",
                )
                self.assertNotIn(
                    dep_name,
                    non_publishable_names,
                    f"Publishable crate {pkg['name']} depends on publish=false crate {dep_name}",
                )

        wedeo_toml = tomllib.loads(
            (self.root / "crates" / "wedeo" / "Cargo.toml").read_text(encoding="utf-8")
        )
        self.assertNotIn("wedeo-rav1d", wedeo_toml.get("dependencies", {}))
        self.assertNotIn("rav1d", wedeo_toml.get("features", {}))

        tree = run_cmd(
            [
                "cargo",
                "tree",
                "-p",
                "wedeo",
                "--all-features",
                "--offline",
                "--prefix",
                "none",
            ],
            cwd=self.root,
        )
        self.assertEqual(tree.returncode, 0, tree.stderr)
        self.assertNotIn("wedeo-rav1d", tree.stdout)
        self.assertNotIn("rav1d", tree.stdout)

        ws_deps = self.root_cargo.get("workspace", {}).get("dependencies", {})
        for dep_name, spec in ws_deps.items():
            if isinstance(spec, dict) and "path" in spec and dep_name not in non_publishable_names:
                self.assertEqual(
                    spec.get("version"),
                    self.ws_pkg["version"],
                    f"Workspace path dep {dep_name} missing matching explicit version",
                )

    @e2e_test(feature="F14", tier=2, milestone="M6")
    def test_t2_publish_script_rejects_stale_crates_duplicates_and_no_dryrun_exec(
        self,
    ) -> None:
        script = self.root / "scripts" / "publish.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        crates = _parse_publish_sh_crates(text)
        self.assertEqual(len(crates), len(set(crates)), f"Duplicate crates in publish.sh: {crates}")
        for stale in STALE_V1_CRATES:
            self.assertNotIn(stale, crates)
            self.assertNotIn(stale, text)

        shadow_bin = self.tmp / "shadow_bin"
        shadow_bin.mkdir()
        fake_cargo = shadow_bin / "cargo"
        fake_cargo.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
        fake_cargo.chmod(0o755)
        env = augmented_env()
        env["PATH"] = f"{shadow_bin}{os.pathsep}{env['PATH']}"

        dry = run_cmd([str(script)], cwd=self.root, env=env)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("[dry-run]", dry.stdout)

    @e2e_test(feature="F14", tier=2, milestone="M6")
    def test_t2_version_and_license_consistency_across_all_artifacts(self) -> None:
        ver = self.ws_pkg["version"]
        lic = self.ws_pkg["license"]

        cli_ver = run_wedeo(["--version"])
        self.assertEqual(cli_ver.returncode, 0, cli_ver.stderr)
        self.assertEqual(cli_ver.stdout.strip(), f"wedeo {ver}")

        wasm_cargo = tomllib.loads(
            (self.root / "bins" / "wedeo-wasm" / "Cargo.toml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(wasm_cargo["package"]["version"], ver)
        self.assertEqual(wasm_cargo["package"]["license"], lic)

        npm_pkg_path = self.root / "packages" / "wedeo-wasm" / "package.json"
        self.assertTrue(npm_pkg_path.is_file(), f"Missing {npm_pkg_path}")
        npm_pkg = json.loads(npm_pkg_path.read_text(encoding="utf-8"))
        self.assertEqual(npm_pkg["version"], ver)
        self.assertEqual(npm_pkg["license"], lic)

        formula_path = self.root / "Formula" / "wedeo.rb"
        self.assertTrue(formula_path.is_file(), f"Missing {formula_path}")
        formula_text = formula_path.read_text(encoding="utf-8")
        self.assertIn(ver, formula_text)
        self.assertIn(lic, formula_text)

        changelog_path = self.root / "CHANGELOG.md"
        self.assertTrue(changelog_path.is_file(), f"Missing {changelog_path}")
        changelog_text = changelog_path.read_text(encoding="utf-8")
        self.assertIn(ver, changelog_text)

        root_readme = (self.root / "README.md").read_text(encoding="utf-8")
        self.assertIn(ver, root_readme)
        self.assertIn(lic, root_readme)

    @e2e_test(feature="F14", tier=2, milestone="M6")
    def test_t2_readme_no_stale_v1_symbols_and_consumer_crate_compiles(self) -> None:
        crate_readme = (self.root / "crates" / "wedeo" / "README.md").read_text(
            encoding="utf-8"
        )
        for stale in STALE_V1_API_SYMBOLS:
            self.assertNotIn(
                stale, crate_readme, f"Stale v1 symbol {stale!r} in crates/wedeo/README.md"
            )

        consumer = self.tmp / "consumer"
        (consumer / "src").mkdir(parents=True)
        shutil.copy2(self.root / "Cargo.lock", consumer / "Cargo.lock")
        (consumer / "Cargo.toml").write_text(
            f"""[package]
name = "wedeo-e2e-consumer"
version = "0.1.0"
edition = "2024"

[dependencies]
wedeo = {{ path = {json.dumps(str(self.root / "crates" / "wedeo"))}, default-features = false, features = ["audio-ops", "video-ops", "syntax", "pcm", "wav", "h264", "vp9", "mp4", "mkv", "ivf"] }}
""",
            encoding="utf-8",
        )
        (consumer / "src" / "main.rs").write_text(
            """fn main() {
    let _reg = wedeo::native_registry();
    let _policy = wedeo::ErrorPolicy::default();
    let _ = std::any::type_name::<wedeo::audio::AudioReader>();
    let _ = std::any::type_name::<wedeo::video::VideoReader>();
}
""",
            encoding="utf-8",
        )
        chk = run_cmd(
            [
                "cargo",
                "check",
                "--offline",
                "--target-dir",
                self.tmp / "consumer_target",
            ],
            cwd=consumer,
            timeout=240.0,
        )
        self.assertEqual(chk.returncode, 0, chk.stderr or chk.stdout)

    @e2e_test(feature="F14", tier=2, milestone="M6")
    def test_t2_npm_package_negative_ref_dev_specifiers_and_wasm_size_budget(
        self,
    ) -> None:
        pkg_dir = self.root / "packages" / "wedeo-wasm"
        pkg_json = pkg_dir / "package.json"
        self.assertTrue(pkg_json.is_file(), f"Missing {pkg_json}")
        pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
        self.assertIsNot(pkg.get("private"), True)

        for dep_key in ("dependencies", "peerDependencies", "optionalDependencies"):
            deps = pkg.get(dep_key, {})
            for name, spec in deps.items():
                self.assertFalse(
                    str(spec).startswith(("workspace:", "file:", "link:")),
                    f"Forbidden local dep spec {name}={spec!r} in {dep_key}",
                )

        for rel in pkg.get("files", []):
            self.assertNotIn("..", str(rel))
            resolved = (pkg_dir / str(rel)).resolve()
            self.assertTrue(resolved.is_relative_to(pkg_dir.resolve()))

        wasm_cands = list(pkg_dir.rglob("*.wasm")) + [self.root / "web" / "wedeo.wasm"]
        wasm_files = [p for p in wasm_cands if p.is_file()]
        self.assertTrue(wasm_files, "No .wasm artifact found for size check")
        gz_len = len(gzip.compress(wasm_files[0].read_bytes(), compresslevel=9))
        self.assertLess(gz_len, WASM_SIZE_LIMIT_BYTES)

    @e2e_test(feature="F14", tier=2, milestone="M6")
    def test_t2_formula_and_release_workflow_negative_guards(self) -> None:
        formula_path = self.root / "Formula" / "wedeo.rb"
        rel_path = self.root / ".github" / "workflows" / "release.yml"
        self.assertTrue(formula_path.is_file(), f"Missing {formula_path}")
        self.assertTrue(rel_path.is_file(), f"Missing {rel_path}")
        formula = formula_path.read_text(encoding="utf-8")
        rel = rel_path.read_text(encoding="utf-8")
        for label, text in (("Formula/wedeo.rb", formula), ("release.yml", rel)):
            for bad in ("TODO", "FIXME", "REPLACE_ME", "/home/", "/Users/"):
                self.assertNotIn(bad, text, f"Forbidden token {bad!r} in {label}")

        openers = len(re.findall(r"^\s*(class|def)\b|\bdo\b", formula, flags=re.M))
        ends = len(re.findall(r"\bend\b", formula))
        self.assertEqual(openers, ends, "Unbalanced Ruby block/end count in Formula/wedeo.rb")
        self.assertIn("needs:", rel)
