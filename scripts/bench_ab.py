"""A/B benchmark: current working tree vs previous commit.

Usage:
    uv run scripts/bench_ab.py [file] [--runs N] [--warmup N] [--commit REF]

Defaults:
    file    = Big_Buck_Bunny_1080_10s_2MB.mp4
    runs    = 5
    warmup  = 1
    commit  = HEAD~1  (the commit to compare against)

Requires: hyperfine on PATH, cargo in PATH.

The script:
1. Builds release for the current tree
2. Copies the binary to a temp location ("current")
3. Stashes changes, checks out the comparison commit, builds release
4. Copies that binary to a temp location ("baseline")
5. Restores the working tree
6. Runs hyperfine comparing both binaries
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark current vs previous commit")
    parser.add_argument("file", nargs="?", default="Big_Buck_Bunny_1080_10s_2MB.mp4",
                        help="Input file to decode (default: BBB)")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark runs (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (default: 1)")
    parser.add_argument("--commit", default="HEAD~1", help="Baseline commit (default: HEAD~1)")
    args = parser.parse_args()

    input_file = Path(args.file).resolve()
    if not input_file.exists():
        print(f"ERROR: input file not found: {input_file}", file=sys.stderr)
        return 1

    if not shutil.which("hyperfine"):
        print("ERROR: hyperfine not found on PATH", file=sys.stderr)
        return 1

    bin_name = "wedeo-framecrc"
    cargo_build = f"cargo build --release --bin {bin_name}"
    release_bin = Path("target/release") / bin_name

    with tempfile.TemporaryDirectory(prefix="bench_ab_") as tmpdir:
        current_bin = Path(tmpdir) / "current"
        baseline_bin = Path(tmpdir) / "baseline"

        # Build current
        print("=== Building CURRENT tree ===")
        run(cargo_build)
        shutil.copy2(release_bin, current_bin)

        # Check for uncommitted changes
        dirty = subprocess.run("git diff --quiet && git diff --cached --quiet",
                               shell=True).returncode != 0

        try:
            if dirty:
                print("=== Stashing uncommitted changes ===")
                run("git stash --include-untracked -q")

            print(f"=== Checking out {args.commit} ===")
            original_branch = subprocess.run(
                "git rev-parse --abbrev-ref HEAD", shell=True,
                capture_output=True, text=True, check=True
            ).stdout.strip()
            original_sha = subprocess.run(
                "git rev-parse HEAD", shell=True,
                capture_output=True, text=True, check=True
            ).stdout.strip()

            run(f"git checkout -q {args.commit}")
            print("=== Building BASELINE ===")
            run(cargo_build)
            shutil.copy2(release_bin, baseline_bin)

        finally:
            # Restore
            print("=== Restoring working tree ===")
            run(f"git checkout -q {original_sha}")
            if original_branch != "HEAD":
                run(f"git checkout -q {original_branch}")
            if dirty:
                run("git stash pop -q")

        # Run benchmark
        print(f"\n=== Benchmarking ({args.runs} runs, {args.warmup} warmup) ===\n")
        cmd = (
            f"hyperfine --warmup {args.warmup} --runs {args.runs} "
            f"--export-markdown /dev/stderr "
            f"-n baseline '{baseline_bin} {input_file} > /dev/null 2>&1' "
            f"-n current  '{current_bin} {input_file} > /dev/null 2>&1'"
        )
        subprocess.run(cmd, shell=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
