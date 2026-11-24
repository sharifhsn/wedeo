#!/usr/bin/env python3
"""Standalone opaque-box E2E test runner for wedeo v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any, NoReturn

E2E_DIR = Path(__file__).resolve().parent
if str(E2E_DIR) not in sys.path:
    sys.path.insert(0, str(E2E_DIR))

from lib.harness import (  # noqa: E402
    FEATURES,
    MILESTONES,
    MILESTONE_FEATURES,
    TIERS,
    TestMeta,
    build_wedeo_cli,
    find_wedeo_bin,
    repo_root,
)


class E2EArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that exits with code 1 (instead of 2) on CLI usage errors."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        sys.stderr.write(f"error: {message}\n")
        sys.exit(1)


class E2ETestResult(unittest.TestResult):
    """TestResult that tracks metadata, timing, and forbids skipped tests."""

    def __init__(self, *, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.records: list[dict[str, Any]] = []
        self._start_times: dict[str, float] = {}

    def startTest(self, test: unittest.TestCase) -> None:
        super().startTest(test)
        self._start_times[test.id()] = time.monotonic()

    def _record(
        self,
        test: unittest.TestCase,
        status: str,
        message: str | None = None,
    ) -> None:
        meta: TestMeta = getattr(getattr(test, test._testMethodName), "_e2e_meta")
        started = self._start_times.pop(test.id(), time.monotonic())
        duration = round(time.monotonic() - started, 4)
        self.records.append(
            {
                "id": test.id(),
                "feature": meta.feature,
                "features": list(meta.features),
                "tier": meta.tier,
                "milestone": meta.milestone,
                "status": status,
                "duration_sec": duration,
                "message": message,
            }
        )
        if self.verbose:
            feat_str = ",".join(meta.features)
            print(
                f"  [{feat_str} T{meta.tier} {meta.milestone}] "
                f"{test.id()} ... {status} ({duration:.3f}s)",
                flush=True,
            )

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self._record(test, "passed")

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        super().addFailure(test, err)
        msg = "".join(traceback.format_exception(*err))
        self._record(test, "failed", msg)

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        super().addError(test, err)
        msg = "".join(traceback.format_exception(*err))
        self._record(test, "error", msg)

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self._record(
            test,
            "skipped",
            f"Auto-skipping is forbidden in E2E suite: {reason}",
        )

    def addExpectedFailure(self, test: unittest.TestCase, err: Any) -> None:
        super().addExpectedFailure(test, err)
        msg = "".join(traceback.format_exception(*err))
        self._record(test, "failed", f"Unexpected expectedFailure: {msg}")

    def addUnexpectedSuccess(self, test: unittest.TestCase) -> None:
        super().addUnexpectedSuccess(test)
        self._record(test, "failed", "Unexpected success")


def _iter_test_cases(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    cases: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            cases.extend(_iter_test_cases(item))
        elif isinstance(item, unittest.TestCase):
            cases.append(item)
    return cases


def discover_tests() -> list[tuple[unittest.TestCase, TestMeta]]:
    """Discover all test_*.py tests under tests/e2e/ and verify @e2e_test tags."""
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(E2E_DIR),
        pattern="test_*.py",
        top_level_dir=str(E2E_DIR),
    )
    if loader.errors:
        for err in loader.errors:
            sys.stderr.write(f"Test discovery error:\n{err}\n")
        sys.exit(1)

    cases = _iter_test_cases(suite)
    tagged: list[tuple[unittest.TestCase, TestMeta]] = []
    untagged: list[str] = []
    for tc in cases:
        method = getattr(tc, tc._testMethodName, None)
        meta = getattr(method, "_e2e_meta", None) if method is not None else None
        if not isinstance(meta, TestMeta):
            untagged.append(tc.id())
        else:
            tagged.append((tc, meta))

    if untagged:
        sys.stderr.write(
            "Error: found test methods without @e2e_test metadata:\n"
            + "\n".join(f"  - {t}" for t in untagged)
            + "\n"
        )
        sys.exit(1)

    return tagged


def build_parser() -> E2EArgumentParser:
    parser = E2EArgumentParser(
        prog="run.py",
        description="Standalone opaque-box E2E test runner for wedeo v2.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run only tests with milestone == 'baseline'.",
    )
    parser.add_argument(
        "--milestone",
        choices=[*MILESTONES, "all"],
        default=None,
        help="Filter tests by milestone.",
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "all"],
        action="append",
        default=None,
        help="Filter tests by tier (repeatable, or 'all').",
    )
    parser.add_argument(
        "--feature",
        choices=list(FEATURES),
        action="append",
        default=None,
        help="Filter tests by feature F1..F14 (repeatable).",
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Print per-feature x per-tier inventory and enforce thresholds.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="Write structured JSON results or inventory report to PATH.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-test progress.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run `cargo build -p wedeo-cli` before running tests.",
    )
    return parser


def filter_tests(
    discovered: list[tuple[unittest.TestCase, TestMeta]],
    *,
    milestone_filter: str | None,
    tier_filter: set[int] | None,
    feature_filter: set[str] | None,
) -> list[tuple[unittest.TestCase, TestMeta]]:
    selected: list[tuple[unittest.TestCase, TestMeta]] = []
    for tc, meta in discovered:
        if milestone_filter is not None and meta.milestone != milestone_filter:
            continue
        if tier_filter is not None and meta.tier not in tier_filter:
            continue
        if feature_filter is not None:
            if (
                meta.feature not in feature_filter
                and not any(f in feature_filter for f in meta.features)
            ):
                continue
        selected.append((tc, meta))
    return selected


def run_inventory(
    discovered: list[tuple[unittest.TestCase, TestMeta]],
    *,
    baseline: bool,
    milestone: str | None,
    feature_filter: set[str] | None,
    tier_filter: set[int] | None,
    json_path: Path | None,
) -> int:
    counts: dict[str, dict[int, int]] = {
        feat: {t: 0 for t in TIERS} for feat in FEATURES
    }
    milestone_counts: dict[str, int] = {m: 0 for m in MILESTONES}

    for _, meta in discovered:
        if tier_filter is not None and meta.tier not in tier_filter:
            continue
        counts[meta.feature][meta.tier] += 1
        milestone_counts[meta.milestone] += 1

    print("Feature |  T1 |  T2 |  T3 |  T4 | Total")
    print("--------+-----+-----+-----+-----+------")
    t_totals = {t: 0 for t in TIERS}
    grand_total = 0
    for feat in FEATURES:
        row = counts[feat]
        row_total = sum(row.values())
        for t in TIERS:
            t_totals[t] += row[t]
        grand_total += row_total
        print(
            f"{feat:<7} | {row[1]:>3} | {row[2]:>3} | {row[3]:>3} | "
            f"{row[4]:>3} | {row_total:>5}"
        )
    print("--------+-----+-----+-----+-----+------")
    print(
        f"{'TOTAL':<7} | {t_totals[1]:>3} | {t_totals[2]:>3} | "
        f"{t_totals[3]:>3} | {t_totals[4]:>3} | {grand_total:>5}"
    )
    print()
    ms_summary = ", ".join(f"{m}={milestone_counts[m]}" for m in MILESTONES)
    print(f"Milestones: {ms_summary}")

    scoped = baseline or milestone is not None or feature_filter is not None
    if feature_filter is not None:
        scoped_features = tuple(f for f in FEATURES if f in feature_filter)
    elif baseline:
        scoped_features = MILESTONE_FEATURES["baseline"]
    elif milestone is not None and milestone != "all":
        scoped_features = MILESTONE_FEATURES[milestone]
    else:
        scoped_features = FEATURES

    full_mode = not scoped or (milestone == "all" and feature_filter is None)

    shortfalls: list[str] = []
    for feat in scoped_features:
        if counts[feat][1] < 5:
            shortfalls.append(f"{feat} T1: {counts[feat][1]}/5")
        if counts[feat][2] < 5:
            shortfalls.append(f"{feat} T2: {counts[feat][2]}/5")

    if full_mode:
        if t_totals[3] < 14:
            shortfalls.append(f"Tier 3 total: {t_totals[3]}/14")
        if t_totals[4] < 7:
            shortfalls.append(f"Tier 4 total: {t_totals[4]}/7")
        if grand_total < 161:
            shortfalls.append(f"Grand total: {grand_total}/161")

    complete_features = sum(
        1 for f in FEATURES if counts[f][1] >= 5 and counts[f][2] >= 5
    )
    mode_label = "full" if full_mode else f"scoped ({','.join(scoped_features)})"
    passed = len(shortfalls) == 0

    print(
        f"Progress: {complete_features}/{len(FEATURES)} features meet T1>=5 & T2>=5"
    )
    if passed:
        print(f"Inventory check [{mode_label}]: PASS")
    else:
        print(f"Inventory check [{mode_label}]: FAIL")
        print("Shortfalls:")
        for item in shortfalls:
            print(f"  - {item}")

    if json_path is not None:
        payload = {
            "mode": mode_label,
            "passed": passed,
            "features": {
                feat: {f"T{t}": counts[feat][t] for t in TIERS}
                | {"total": sum(counts[feat].values())}
                for feat in FEATURES
            },
            "tiers": {f"T{t}": t_totals[t] for t in TIERS},
            "milestones": milestone_counts,
            "total": grand_total,
            "complete_features": complete_features,
            "shortfalls": shortfalls,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.baseline and args.milestone is not None:
        parser.error("--baseline and --milestone cannot be combined")

    milestone_filter: str | None
    if args.baseline:
        milestone_filter = "baseline"
    elif args.milestone in (None, "all"):
        milestone_filter = None
    else:
        milestone_filter = args.milestone

    if args.tier is None or "all" in args.tier:
        tier_filter: set[int] | None = None
    else:
        tier_filter = {int(t) for t in args.tier}

    feature_filter: set[str] | None = (
        set(args.feature) if args.feature is not None else None
    )

    discovered = discover_tests()

    if args.inventory:
        return run_inventory(
            discovered,
            baseline=args.baseline,
            milestone=args.milestone,
            feature_filter=feature_filter,
            tier_filter=tier_filter,
            json_path=args.json,
        )

    selected = filter_tests(
        discovered,
        milestone_filter=milestone_filter,
        tier_filter=tier_filter,
        feature_filter=feature_filter,
    )
    if not selected:
        sys.stderr.write("Error: no tests matched the specified filters\n")
        return 1

    root = repo_root()
    if args.build:
        wedeo_bin = build_wedeo_cli(root)
    else:
        wedeo_bin = find_wedeo_bin(root, auto_build=True)
    os.environ["WEDEO_BIN"] = str(wedeo_bin)

    if args.verbose:
        print(f"Using wedeo binary: {wedeo_bin}", flush=True)
        print(f"Running {len(selected)} E2E tests...", flush=True)

    suite = unittest.TestSuite([tc for tc, _ in selected])
    result = E2ETestResult(verbose=args.verbose)
    t0 = time.monotonic()
    suite.run(result)
    elapsed = round(time.monotonic() - t0, 4)

    counts = defaultdict(int)
    for rec in result.records:
        counts[rec["status"]] += 1

    passed = (
        len(result.failures) == 0
        and len(result.errors) == 0
        and len(result.skipped) == 0
        and counts["failed"] == 0
        and counts["error"] == 0
        and counts["skipped"] == 0
    )

    if not passed:
        sys.stderr.write("\n=== E2E FAILURES ===\n")
        for rec in result.records:
            if rec["status"] != "passed":
                sys.stderr.write(
                    f"[{rec['status'].upper()}] {rec['id']} "
                    f"({rec['feature']} T{rec['tier']} {rec['milestone']})\n"
                    f"{rec['message'] or ''}\n"
                )

    print(
        f"\nRan {result.testsRun} tests in {elapsed:.2f}s: "
        f"{counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['error']} errors, {counts['skipped']} skipped"
    )

    if args.json is not None:
        payload = {
            "wedeo_bin": str(wedeo_bin),
            "passed": passed,
            "duration_sec": elapsed,
            "summary": {
                "total": result.testsRun,
                "passed": counts["passed"],
                "failed": counts["failed"],
                "errors": counts["error"],
                "skipped": counts["skipped"],
            },
            "tests": result.records,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
