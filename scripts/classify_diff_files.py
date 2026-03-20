#!/usr/bin/env python3
"""Classify H.264 conformance DIFF files by stream features.

For each non-BITEXACT file, extracts key H.264 features via FFmpeg's
trace_headers BSF and prints a structured summary to aid debugging triage.

Usage:
    python3 scripts/classify_diff_files.py

    # Include BITEXACT files too
    python3 scripts/classify_diff_files.py --all

    # Only show files matching a prefix
    python3 scripts/classify_diff_files.py --prefix CVWP

Requires:
    - ffmpeg binary in PATH
    - fate-suite/h264-conformance/ directory
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from conformance_parse import run_conformance_report


def detect_features_detailed(input_path: Path) -> dict:
    """Extract detailed H.264 features from a file using trace_headers."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(input_path),
                "-c:v",
                "copy",
                "-bsf:v",
                "trace_headers",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    trace = result.stderr
    features = {}

    def find_first(field_name: str) -> int | None:
        m = re.search(rf"{field_name}\s+\S+\s*=\s*(\d+)", trace)
        return int(m.group(1)) if m else None

    def find_all_values(field_name: str) -> list[int]:
        return [int(m.group(1)) for m in re.finditer(rf"{field_name}\s+\S+\s*=\s*(\d+)", trace)]

    # SPS features
    features["profile_idc"] = find_first("profile_idc") or 0
    features["poc_type"] = find_first("pic_order_cnt_type") or 0
    features["max_ref_frames"] = find_first("max_num_ref_frames") or 0
    features["direct_8x8_inference"] = bool(find_first("direct_8x8_inference_flag"))
    features["frame_mbs_only"] = bool(find_first("frame_mbs_only_flag"))
    features["entropy"] = "cabac" if find_first("entropy_coding_mode_flag") else "cavlc"

    # PPS features
    features["weighted_pred_flag"] = bool(find_first("weighted_pred_flag"))
    features["weighted_bipred_idc"] = find_first("weighted_bipred_idc") or 0

    # FMO
    num_slice_groups = find_first("num_slice_groups_minus1")
    features["fmo"] = (num_slice_groups or 0) > 0

    # Slice types present
    slice_types = find_all_values("slice_type")
    type_names = {0: "P", 1: "B", 2: "I", 5: "P", 6: "B", 7: "I"}
    features["slice_types"] = sorted(set(type_names.get(t, f"?{t}") for t in slice_types))

    # Active ref counts (from slice headers)
    l0_counts = find_all_values("num_ref_idx_l0_active_minus1")
    features["max_l0_active"] = max(l0_counts) + 1 if l0_counts else 1

    # MMCO
    features["has_mmco"] = "memory_management_control_operation" in trace

    # Constrained intra pred
    features["constrained_intra"] = bool(find_first("constrained_intra_pred_flag"))

    # Profile name
    profile_names = {
        66: "Baseline",
        77: "Main",
        88: "Extended",
        100: "High",
    }
    features["profile_name"] = profile_names.get(features["profile_idc"], f"Unknown({features['profile_idc']})")

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Classify H.264 conformance files by stream features."
    )
    parser.add_argument("--all", action="store_true", help="Include BITEXACT files")
    parser.add_argument("--prefix", help="Only show files matching this prefix")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    conf_dir = Path("fate-suite/h264-conformance")
    if not conf_dir.exists():
        print(f"Directory not found: {conf_dir}", file=sys.stderr)
        sys.exit(1)

    # Get conformance status
    _, report = run_conformance_report()

    # Process each file
    for name in sorted(report):
        if args.prefix and not name.upper().startswith(args.prefix.upper()):
            continue

        status, match, total = report[name]
        if not args.all and status == "BITEXACT":
            continue

        # Find the actual file
        filepath = conf_dir / name
        if not filepath.exists():
            continue
        features = detect_features_detailed(filepath)
        if not features:
            continue

        # Format output
        status_str = f"BITEXACT ({total})" if status == "BITEXACT" else f"{match}/{total}"
        print(f"{name}: {status_str}")

        tags = []
        tags.append(features["profile_name"])
        tags.append(features["entropy"].upper())
        tags.append(f"POC type {features['poc_type']}")
        tags.append(f"slices: {'+'.join(features['slice_types'])}")
        if features["max_l0_active"] > 1:
            tags.append(f"L0 refs: {features['max_l0_active']}")
        if features["weighted_pred_flag"]:
            tags.append("weighted_pred")
        if features["weighted_bipred_idc"] > 0:
            tags.append(f"weighted_bipred_idc={features['weighted_bipred_idc']}")
        if not features["direct_8x8_inference"] and "B" in features["slice_types"]:
            tags.append("direct_8x8=0")
        if features["fmo"]:
            tags.append("FMO")
        if features["has_mmco"]:
            tags.append("MMCO")
        if features["constrained_intra"]:
            tags.append("constrained_intra")

        print(f"  {', '.join(tags)}")
        print()


if __name__ == "__main__":
    main()
