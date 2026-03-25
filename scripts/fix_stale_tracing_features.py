#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""Remove stale features=["tracing"] from all diagnostic scripts.

Tracing is always available (unconditional `tracing` crate dependency).
The `--features tracing` cargo flag no longer exists. Scripts that pass it
cause build failures.

Usage:
    python3 scripts/fix_stale_tracing_features.py          # dry run
    python3 scripts/fix_stale_tracing_features.py --apply   # fix files
"""

import argparse
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path("scripts")

# Pattern 1: features=["tracing"] in find_wedeo_binary() calls
PAT_FEATURES_PARAM = re.compile(
    r',\s*features\s*=\s*\["tracing"\]',
)

# Pattern 2: Default features = ["tracing"] assignment
PAT_FEATURES_DEFAULT = re.compile(
    r'^\s*features\s*=\s*\["tracing"\]\s*$',
    re.MULTILINE,
)

# Pattern 3: "--features", "tracing" in cargo build command arrays
PAT_CARGO_FEATURES = re.compile(
    r',?\s*"--features",\s*"tracing"',
)

# Pattern 4: --features tracing in string literals (error messages, comments)
PAT_STR_FEATURES = re.compile(
    r'--features\s+tracing\b',
)


def scan_file(path: Path, apply: bool) -> list[str]:
    """Scan a file for stale tracing features. Returns list of findings."""
    # Don't scan ourselves
    if path.name == "fix_stale_tracing_features.py":
        return []
    text = path.read_text()
    findings = []

    for i, line in enumerate(text.splitlines(), 1):
        if PAT_FEATURES_PARAM.search(line):
            findings.append(f"  {path}:{i}: features=[\"tracing\"] param")
        if PAT_CARGO_FEATURES.search(line) and '"--features"' in line:
            findings.append(f"  {path}:{i}: --features tracing in cargo cmd")
        if PAT_FEATURES_DEFAULT.match(line.strip() + "\n"):
            findings.append(f"  {path}:{i}: features default assignment")

    if apply and findings:
        new_text = text
        # Remove features=["tracing"] parameter (keep surrounding syntax)
        new_text = PAT_FEATURES_PARAM.sub("", new_text)
        # Remove default assignment (replace with features = None or [])
        new_text = re.sub(
            r'(\s*)features\s*=\s*\["tracing"\]',
            r"\1features = []",
            new_text,
        )
        # Remove "--features", "tracing" from cargo command arrays
        new_text = PAT_CARGO_FEATURES.sub("", new_text)
        if new_text != text:
            path.write_text(new_text)

    return findings


def main():
    parser = argparse.ArgumentParser(description="Fix stale tracing features")
    parser.add_argument("--apply", action="store_true", help="Apply fixes")
    args = parser.parse_args()

    if not SCRIPTS_DIR.exists():
        print(f"Error: {SCRIPTS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    total = []
    for py_file in sorted(SCRIPTS_DIR.glob("*.py")):
        findings = scan_file(py_file, args.apply)
        total.extend(findings)

    if total:
        action = "Fixed" if args.apply else "Found"
        print(f"{action} {len(total)} stale tracing feature references:")
        for f in total:
            print(f)
        if not args.apply:
            print("\nRun with --apply to fix.")
    else:
        print("No stale tracing features found.")


if __name__ == "__main__":
    main()
