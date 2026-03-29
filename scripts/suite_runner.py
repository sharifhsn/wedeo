#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Unified multi-suite H.264 conformance runner.

Supports built-in FATE suites (framecrc comparison vs FFmpeg) and
JSON-discovered JVT suites (MD5 comparison against ITU ground truth).

Usage:
    python3 scripts/suite_runner.py --suite jvt-avc-v1
    python3 scripts/suite_runner.py --suite jvt-avc-v1,jvt-fr-ext
    python3 scripts/suite_runner.py --suite fate-cavlc
    python3 scripts/suite_runner.py --suite all
    python3 scripts/suite_runner.py --suite jvt-avc-v1 --format yuv420p
    python3 scripts/suite_runner.py --suite jvt-avc-v1 --profile Main,High
    python3 scripts/suite_runner.py --suite jvt-avc-v1 --save-snapshot
    python3 scripts/suite_runner.py --suite jvt-avc-v1 --check-snapshot
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wedeo_utils import find_ffmpeg_binary, find_wedeo_binary, run_framecrc

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "test_suites" / "h264"
VECTORS_DIR = ROOT / "jvt-vectors"
FATE_DIR = ROOT / "fate-suite" / "h264-conformance"
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent


# ── Suite definitions ────────────────────────────────────────────────────────


@dataclass
class Vector:
    name: str
    input_path: Path
    comparison: str  # "framecrc" or "md5"
    expected_md5: str = ""
    output_format: str = "yuv420p"
    profile: str = ""


@dataclass
class Suite:
    name: str
    vectors: list[Vector] = field(default_factory=list)
    comparison: str = "framecrc"


def _discover_json_suites() -> dict[str, Suite]:
    """Load JVT suites from JSON manifests in test_suites/h264/."""
    suites = {}
    if not MANIFEST_DIR.is_dir():
        return suites
    for p in sorted(MANIFEST_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        suite_name = data["name"]
        slug = suite_name.lower().replace("_", "-")
        vectors = []
        for v in data["test_vectors"]:
            # input_file may be nested — we store flattened filename
            input_name = Path(v["input_file"]).name
            input_path = VECTORS_DIR / suite_name / input_name
            vectors.append(Vector(
                name=v["name"],
                input_path=input_path,
                comparison="md5",
                expected_md5=v.get("result", ""),
                output_format=v.get("output_format", "yuv420p"),
                profile=v.get("profile", ""),
            ))
        suites[slug] = Suite(name=suite_name, vectors=vectors, comparison="md5")
    return suites


def _builtin_fate_suites() -> dict[str, Suite]:
    """Built-in FATE suites from hardcoded file lists."""
    # Import the canonical lists from conformance_full
    from conformance_full import PROGRESSIVE_CABAC_FILES, PROGRESSIVE_CAVLC_FILES

    suites = {}
    for slug, file_list in [
        ("fate-cavlc", PROGRESSIVE_CAVLC_FILES),
        ("fate-cabac", PROGRESSIVE_CABAC_FILES),
    ]:
        vectors = [
            Vector(
                name=Path(f).stem,
                input_path=FATE_DIR / f,
                comparison="framecrc",
            )
            for f in file_list
        ]
        suites[slug] = Suite(name=slug, vectors=vectors, comparison="framecrc")
    return suites


def all_suites() -> dict[str, Suite]:
    suites = _builtin_fate_suites()
    suites.update(_discover_json_suites())
    return suites


# ── Comparison functions ─────────────────────────────────────────────────────


def compare_one_framecrc(
    input_path: Path,
    wedeo_bin: Path,
    ffmpeg_bin: Path,
) -> tuple[str, str]:
    """Compare framecrc output. Returns (status, detail)."""
    wedeo_crcs = run_framecrc([str(wedeo_bin), str(input_path)])
    ffmpeg_cmd = [
        str(ffmpeg_bin), "-bitexact",
        "-i", str(input_path),
        "-f", "framecrc", "-",
    ]
    ffmpeg_crcs = run_framecrc(ffmpeg_cmd)

    total = max(len(wedeo_crcs), len(ffmpeg_crcs))
    if total == 0:
        return "ERROR", "no frames from either decoder"

    comparable = min(len(wedeo_crcs), len(ffmpeg_crcs))
    matching = sum(1 for i in range(comparable) if wedeo_crcs[i] == ffmpeg_crcs[i])

    if matching == total:
        return "MATCH", f"framecrc: {total} frames"
    else:
        return "FAIL", f"framecrc: {matching}/{total} match"


def compare_one_md5(
    input_path: Path,
    wedeo_bin: Path,
    expected_md5: str,
) -> tuple[str, str]:
    """Compare MD5 of raw YUV output. Returns (status, detail)."""
    with tempfile.NamedTemporaryFile(suffix=".yuv", delete=False) as f:
        yuv_path = Path(f.name)
    try:
        result = subprocess.run(
            [str(wedeo_bin), str(input_path), "--raw-yuv", str(yuv_path)],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-200:]
            return "ERROR", f"wedeo exited {result.returncode}: {stderr}"
        if not yuv_path.exists() or yuv_path.stat().st_size == 0:
            return "ERROR", "no YUV output"
        actual_md5 = hashlib.md5(yuv_path.read_bytes()).hexdigest()
    except subprocess.TimeoutExpired:
        return "ERROR", "wedeo timed out (120s)"
    finally:
        yuv_path.unlink(missing_ok=True)

    if actual_md5 == expected_md5:
        return "MATCH", f"md5: {actual_md5[:8]}..."
    else:
        return "FAIL", f"expected: {expected_md5[:8]}..., got: {actual_md5[:8]}..."


# ── Runner ───────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    name: str
    status: str  # MATCH, FAIL, SKIP, ERROR
    detail: str = ""


def _snapshot_path(suite_name: str, snapshot_dir: Path) -> Path:
    return snapshot_dir / f".suite_{suite_name.lower()}_snapshot.json"


def run_suite(
    suite: Suite,
    wedeo_bin: Path,
    ffmpeg_bin: Path | None,
    format_filter: set[str] | None = None,
    profile_filter: set[str] | None = None,
    check_snapshot: bool = False,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> list[RunResult]:
    """Run all vectors in a suite and return results."""
    # Load snapshot for regression check
    snapshot_passing: set[str] | None = None
    if check_snapshot:
        snap_path = _snapshot_path(suite.name, snapshot_dir)
        if snap_path.exists():
            snap_data = json.loads(snap_path.read_text())
            snapshot_passing = set(snap_data.get("passing", []))
            if not snapshot_passing:
                print(f"  No baseline tests for {suite.name} (empty snapshot) — skipping")
                return []
        else:
            print(f"  No baseline for {suite.name} (no snapshot at {snap_path.name}) — skipping")
            return []

    results: list[RunResult] = []
    for vec in suite.vectors:
        # Filter by format
        if format_filter and vec.output_format not in format_filter:
            results.append(RunResult(vec.name, "SKIP", f"format: {vec.output_format}, skipped"))
            continue

        # Filter by profile
        if profile_filter and vec.profile not in profile_filter:
            results.append(RunResult(vec.name, "SKIP", f"profile: {vec.profile}, skipped"))
            continue

        # Snapshot regression: only test vectors that were passing
        if snapshot_passing is not None and vec.name not in snapshot_passing:
            results.append(RunResult(vec.name, "SKIP", "not in snapshot"))
            continue

        # Check file exists
        if not vec.input_path.exists():
            results.append(RunResult(vec.name, "SKIP", "file not found"))
            continue

        # Run comparison
        if vec.comparison == "md5":
            status, detail = compare_one_md5(vec.input_path, wedeo_bin, vec.expected_md5)
        elif vec.comparison == "framecrc":
            if ffmpeg_bin is None:
                results.append(RunResult(vec.name, "SKIP", "FFmpeg not available"))
                continue
            status, detail = compare_one_framecrc(vec.input_path, wedeo_bin, ffmpeg_bin)
        else:
            results.append(RunResult(vec.name, "ERROR", f"unknown comparison: {vec.comparison}"))
            continue

        results.append(RunResult(vec.name, status, detail))
    return results


def save_snapshot(suite_name: str, results: list[RunResult], snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR) -> None:
    passing = sorted(r.name for r in results if r.status == "MATCH")
    tested = [r for r in results if r.status != "SKIP"]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snap_path = _snapshot_path(suite_name, snapshot_dir)
    snap_path.write_text(json.dumps({
        "suite": suite_name,
        "passing": passing,
        "count": len(passing),
        "total": len(tested),
    }, indent=2) + "\n")
    print(f"  Snapshot saved: {len(passing)}/{len(tested)} passing -> {snap_path.name}")


def print_results(suite: Suite, results: list[RunResult], verbose: bool) -> tuple[int, int, int, int]:
    """Print results for one suite. Returns (match, fail, error, skip)."""
    tested = [r for r in results if r.status != "SKIP"]
    skipped = [r for r in results if r.status == "SKIP"]
    match_count = sum(1 for r in tested if r.status == "MATCH")
    fail_count = sum(1 for r in tested if r.status == "FAIL")
    error_count = sum(1 for r in tested if r.status == "ERROR")

    total_vecs = len(suite.vectors)
    tested_count = len(tested)
    skip_filter = len(skipped)

    header = f"Suite: {suite.name} ({total_vecs} vectors"
    if skip_filter:
        header += f", {tested_count} after filter"
    header += ")"
    print(f"\n{header}")

    for r in results:
        if r.status == "SKIP" and not verbose:
            continue
        tag = {
            "MATCH": "  MATCH ",
            "FAIL": "  FAIL  ",
            "ERROR": "  ERROR ",
            "SKIP": "  SKIP  ",
        }.get(r.status, "  ???   ")
        print(f"{tag} {r.name} ({r.detail})")

    if tested_count > 0:
        pct = 100 * match_count / tested_count
        parts = []
        if fail_count:
            parts.append(f"{fail_count} FAIL")
        if error_count:
            parts.append(f"{error_count} ERROR")
        summary = f", ".join(parts) if parts else "all pass"
        print(f"  Result: {match_count}/{tested_count} MATCH ({pct:.1f}%) [{summary}]")
    else:
        print("  Result: no vectors tested")

    return match_count, fail_count, error_count, len(skipped)


def main():
    parser = argparse.ArgumentParser(description="Unified H.264 conformance runner")
    parser.add_argument("--suite", required=True,
                        help="Comma-separated suite slugs, or 'all'")
    parser.add_argument("--format", dest="formats",
                        help="Comma-separated output formats to include (e.g. yuv420p)")
    parser.add_argument("--profile", dest="profiles",
                        help="Comma-separated profiles to include (e.g. Main,High)")
    parser.add_argument("--save-snapshot", action="store_true",
                        help="Save passing vectors as snapshot for regression checking")
    parser.add_argument("--check-snapshot", action="store_true",
                        help="Only test vectors from snapshot, exit non-zero on regression")
    parser.add_argument("--snapshot-dir", type=Path, default=None,
                        help="Directory for snapshot files (default: scripts/)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show skipped vectors")
    args = parser.parse_args()

    available = all_suites()

    # Select suites
    if args.suite == "all":
        selected = available
    else:
        slugs = [s.strip() for s in args.suite.split(",")]
        selected = {}
        for slug in slugs:
            if slug not in available:
                print(f"Unknown suite '{slug}'. Available: {', '.join(sorted(available))}",
                      file=sys.stderr)
                sys.exit(1)
            selected[slug] = available[slug]

    # Parse filters
    format_filter = set(args.formats.split(",")) if args.formats else None
    profile_filter = set(args.profiles.split(",")) if args.profiles else None
    snapshot_dir = args.snapshot_dir or DEFAULT_SNAPSHOT_DIR

    # Find binaries
    wedeo_bin = find_wedeo_binary()
    ffmpeg_bin = None
    needs_ffmpeg = any(s.comparison == "framecrc" for s in selected.values())
    if needs_ffmpeg:
        ffmpeg_bin = find_ffmpeg_binary()

    # Run suites
    grand_match = grand_fail = grand_error = grand_skip = 0
    any_regression = False

    for slug, suite in selected.items():
        results = run_suite(
            suite, wedeo_bin, ffmpeg_bin,
            format_filter=format_filter,
            profile_filter=profile_filter,
            check_snapshot=args.check_snapshot,
            snapshot_dir=snapshot_dir,
        )

        m, f, e, s = print_results(suite, results, args.verbose)
        grand_match += m
        grand_fail += f
        grand_error += e
        grand_skip += s

        if args.save_snapshot:
            save_snapshot(suite.name, results, snapshot_dir=snapshot_dir)

        # Regression detection: only compare vectors that were actually tested
        # (not skipped by format/profile filter or missing files)
        if args.check_snapshot:
            snap_path = _snapshot_path(suite.name, snapshot_dir)
            if snap_path.exists():
                snap_data = json.loads(snap_path.read_text())
                snap_passing = set(snap_data.get("passing", []))
                tested_names = {r.name for r in results if r.status != "SKIP"}
                now_passing = {r.name for r in results if r.status == "MATCH"}
                regressions = (snap_passing & tested_names) - now_passing
                if regressions:
                    any_regression = True
                    print(f"\n  REGRESSIONS in {suite.name}:")
                    for name in sorted(regressions):
                        print(f"    - {name}")

    # Grand total
    tested = grand_match + grand_fail + grand_error
    if len(selected) > 1 and tested > 0:
        print(f"\nTOTAL: {grand_match}/{tested} MATCH across {len(selected)} suites")

    # With --check-snapshot, only fail on regressions (not known failures)
    if args.check_snapshot:
        if any_regression:
            print("\nREGRESSION DETECTED", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    sys.exit(0 if grand_fail == 0 and grand_error == 0 else 1)


if __name__ == "__main__":
    main()
