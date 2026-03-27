#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# ///
"""Capture and compare divan benchmark baselines.

Usage:
    # Save current benchmark as baseline:
    python3 scripts/bench_baseline.py save [--name NAME]

    # Compare current benchmarks against saved baseline:
    python3 scripts/bench_baseline.py compare [--name NAME]

    # List saved baselines:
    python3 scripts/bench_baseline.py list

Baselines are stored in .bench_baselines/ (gitignored).
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASELINE_DIR = Path(".bench_baselines")
DEFAULT_NAME = "default"


def run_bench(filter_arg: str | None = None) -> str:
    """Run cargo bench and return the divan output (after 'Timer precision' line)."""
    cmd = ["cargo", "bench", "-p", "wedeo-codec-h264"]
    if filter_arg:
        cmd += ["--", filter_arg]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"cargo bench failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    # Extract divan output (starts after "Timer precision:" line)
    lines = result.stderr.splitlines() + result.stdout.splitlines()
    output_lines = []
    capturing = False
    for line in lines:
        if "Timer precision:" in line:
            capturing = True
        if capturing:
            output_lines.append(line)
    return "\n".join(output_lines)


def parse_bench_results(text: str) -> dict[str, float]:
    """Parse divan output into {bench_name: median_ns} dict."""
    results = {}
    # Match lines like: │  ├─ get_cabac_bypass  1.707 µs  │ ...  │ 1.957 µs  │ ...
    # The columns are: fastest | slowest | median | mean
    # We want the median (3rd numeric column)
    current_path = []
    for line in text.splitlines():
        # Track tree structure for full benchmark names
        stripped = line.strip()
        if not stripped:
            continue

        # Find benchmark name and values
        # Pattern: name followed by numeric value with unit
        match = re.search(
            r"[├╰─│ ]+\s*(\S+)\s+"
            r"([\d.]+)\s*(ns|µs|ms|s)\s*│\s*"
            r"([\d.]+)\s*(ns|µs|ms|s)\s*│\s*"
            r"([\d.]+)\s*(ns|µs|ms|s)\s*│",
            line,
        )
        if match:
            name = match.group(1)
            median_val = float(match.group(6))
            median_unit = match.group(7)
            # Convert to nanoseconds
            multipliers = {"ns": 1, "µs": 1000, "ms": 1_000_000, "s": 1_000_000_000}
            median_ns = median_val * multipliers[median_unit]
            results[name] = median_ns

    return results


def save_baseline(name: str, filter_arg: str | None) -> None:
    """Run benchmarks and save results."""
    BASELINE_DIR.mkdir(exist_ok=True)
    print(f"Running benchmarks...")
    output = run_bench(filter_arg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = BASELINE_DIR / f"{name}.txt"
    meta_path = BASELINE_DIR / f"{name}.meta"

    # Get git commit
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    filepath.write_text(output)
    meta_path.write_text(f"timestamp={timestamp}\ncommit={commit}\n")
    print(f"Baseline saved: {filepath} (commit {commit})")
    print(output)


def compare_baseline(name: str, filter_arg: str | None) -> None:
    """Run benchmarks and compare against saved baseline."""
    filepath = BASELINE_DIR / f"{name}.txt"
    meta_path = BASELINE_DIR / f"{name}.meta"
    if not filepath.exists():
        print(f"No baseline found: {filepath}", file=sys.stderr)
        print(f"Run: python3 scripts/bench_baseline.py save --name {name}", file=sys.stderr)
        sys.exit(1)

    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            k, v = line.split("=", 1)
            meta[k] = v

    print(f"Baseline: {name} (commit {meta.get('commit', '?')}, {meta.get('timestamp', '?')})")
    print(f"Running current benchmarks...")

    old_output = filepath.read_text()
    new_output = run_bench(filter_arg)

    old_results = parse_bench_results(old_output)
    new_results = parse_bench_results(new_output)

    if not old_results:
        print("Warning: could not parse baseline results", file=sys.stderr)
        print("\n--- Current ---")
        print(new_output)
        return

    # Print comparison table
    print(f"\n{'Benchmark':<40} {'Baseline':>10} {'Current':>10} {'Change':>10}")
    print("-" * 75)

    for name in sorted(set(old_results) | set(new_results)):
        old_ns = old_results.get(name)
        new_ns = new_results.get(name)
        if old_ns is None:
            print(f"{name:<40} {'---':>10} {format_ns(new_ns):>10} {'NEW':>10}")
        elif new_ns is None:
            print(f"{name:<40} {format_ns(old_ns):>10} {'---':>10} {'GONE':>10}")
        else:
            ratio = new_ns / old_ns
            if ratio < 0.95:
                change = f"\033[32m{ratio:.2f}x\033[0m"  # green = faster
            elif ratio > 1.05:
                change = f"\033[31m{ratio:.2f}x\033[0m"  # red = slower
            else:
                change = f"{ratio:.2f}x"
            print(f"{name:<40} {format_ns(old_ns):>10} {format_ns(new_ns):>10} {change:>10}")


def format_ns(ns: float) -> str:
    """Format nanoseconds into human-readable string."""
    if ns < 1000:
        return f"{ns:.1f}ns"
    elif ns < 1_000_000:
        return f"{ns / 1000:.1f}µs"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f}ms"
    else:
        return f"{ns / 1_000_000_000:.1f}s"


def list_baselines() -> None:
    """List saved baselines."""
    if not BASELINE_DIR.exists():
        print("No baselines saved yet.")
        return
    for f in sorted(BASELINE_DIR.glob("*.txt")):
        name = f.stem
        meta_path = BASELINE_DIR / f"{name}.meta"
        meta = {}
        if meta_path.exists():
            for line in meta_path.read_text().splitlines():
                k, v = line.split("=", 1)
                meta[k] = v
        print(f"  {name:<20} commit={meta.get('commit', '?'):<10} {meta.get('timestamp', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline capture and comparison")
    sub = parser.add_subparsers(dest="command", required=True)

    save_p = sub.add_parser("save", help="Save current benchmarks as baseline")
    save_p.add_argument("--name", default=DEFAULT_NAME, help="Baseline name")
    save_p.add_argument("--filter", default=None, help="Filter benchmarks (e.g. 'mc')")

    comp_p = sub.add_parser("compare", help="Compare against saved baseline")
    comp_p.add_argument("--name", default=DEFAULT_NAME, help="Baseline name")
    comp_p.add_argument("--filter", default=None, help="Filter benchmarks")

    sub.add_parser("list", help="List saved baselines")

    args = parser.parse_args()

    if args.command == "save":
        save_baseline(args.name, args.filter)
    elif args.command == "compare":
        compare_baseline(args.name, args.filter)
    elif args.command == "list":
        list_baselines()


if __name__ == "__main__":
    main()
