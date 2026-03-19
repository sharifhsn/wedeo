#!/usr/bin/env python3
"""Comprehensive H.264 conformance report: wedeo vs FFmpeg.

Scans all H.264 conformance files, detects their features via FFmpeg's
trace_headers BSF, runs both decoders, and produces a structured report
categorizing each file as BITEXACT / DIFF / FAIL / SKIP.

Usage:
    # Full report (all files in fate-suite/h264-conformance/)
    python3 scripts/conformance_report.py

    # Only files matching a prefix
    python3 scripts/conformance_report.py --prefix BA

    # Show feature details for each file
    python3 scripts/conformance_report.py --features

    # Only show failures
    python3 scripts/conformance_report.py --failures

    # Machine-readable JSON output
    python3 scripts/conformance_report.py --json

Requires:
    - wedeo-framecrc binary (auto-rebuilt if stale)
    - ffmpeg binary in PATH
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ffmpeg_debug import find_wedeo_binary


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

@dataclass
class H264Features:
    """H.264 stream features extracted via trace_headers."""
    profile_idc: int = 0
    level_idc: int = 0
    entropy_coding: str = "cavlc"  # "cavlc" or "cabac"
    poc_type: int = 0
    max_num_ref_frames: int = 1
    frame_mbs_only: bool = True
    mb_adaptive: bool = False
    direct_8x8_inference: bool = True
    constrained_intra_pred: bool = False
    weighted_pred: bool = False
    weighted_bipred_idc: int = 0
    num_slice_groups: int = 1
    has_b_slices: bool = False
    has_mmco: bool = False
    mmco_ops_used: list[int] = field(default_factory=list)
    num_slices_per_frame: int = 1  # approximate

    @property
    def profile_name(self) -> str:
        names = {
            66: "Baseline", 77: "Main", 88: "Extended",
            100: "High", 110: "High 10", 122: "High 4:2:2",
            244: "High 4:4:4 Predictive",
        }
        return names.get(self.profile_idc, f"Unknown({self.profile_idc})")

    @property
    def is_progressive(self) -> bool:
        return self.frame_mbs_only

    @property
    def is_cavlc(self) -> bool:
        return self.entropy_coding == "cavlc"

    def unsupported_features(self) -> list[str]:
        """Return list of features not yet supported by wedeo."""
        gaps = []
        if not self.is_cavlc:
            gaps.append("CABAC")
        if not self.is_progressive:
            if self.mb_adaptive:
                gaps.append("MBAFF")
            else:
                gaps.append("PAFF (interlaced)")
        if self.constrained_intra_pred:
            gaps.append("constrained_intra_pred (bug: parsed but not enforced)")
        if self.poc_type == 1:
            gaps.append("POC type 1 (not implemented)")
        if self.poc_type == 2:
            gaps.append("POC type 2 (not implemented)")
        if self.weighted_pred:
            gaps.append("weighted prediction")
        if self.weighted_bipred_idc > 0:
            gaps.append(f"weighted bipred (idc={self.weighted_bipred_idc})")
        if self.num_slice_groups > 1:
            gaps.append(f"FMO ({self.num_slice_groups} slice groups)")
        if not self.direct_8x8_inference and self.has_b_slices:
            gaps.append("direct_8x8_inference=0")
        return gaps


def detect_features(input_path: str | Path) -> H264Features:
    """Detect H.264 features using FFmpeg's trace_headers BSF."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(input_path), "-c:v", "copy",
         "-bsf:v", "trace_headers", "-f", "null", "-"],
        capture_output=True, text=True, timeout=30,
    )
    trace = result.stderr
    feat = H264Features()

    def find_first(field_name: str) -> int | None:
        m = re.search(rf"{field_name}\s+\S+\s*=\s*(\d+)", trace)
        return int(m.group(1)) if m else None

    v = find_first("profile_idc")
    if v is not None:
        feat.profile_idc = v
    v = find_first("level_idc")
    if v is not None:
        feat.level_idc = v
    v = find_first("entropy_coding_mode_flag")
    if v is not None:
        feat.entropy_coding = "cabac" if v == 1 else "cavlc"
    v = find_first("pic_order_cnt_type")
    if v is not None:
        feat.poc_type = v
    v = find_first("max_num_ref_frames")
    if v is not None:
        feat.max_num_ref_frames = v
    v = find_first("frame_mbs_only_flag")
    if v is not None:
        feat.frame_mbs_only = v == 1
    v = find_first("mb_adaptive_frame_field_flag")
    if v is not None:
        feat.mb_adaptive = v == 1
    v = find_first("direct_8x8_inference_flag")
    if v is not None:
        feat.direct_8x8_inference = v == 1
    v = find_first("constrained_intra_pred_flag")
    if v is not None:
        feat.constrained_intra_pred = v == 1
    v = find_first("weighted_pred_flag")
    if v is not None:
        feat.weighted_pred = v == 1
    v = find_first("weighted_bipred_idc")
    if v is not None:
        feat.weighted_bipred_idc = v
    v = find_first("num_slice_groups_minus1")
    if v is not None:
        feat.num_slice_groups = v + 1

    # Check for B-slices (slice_type 1 or 6)
    for m in re.finditer(r"slice_type\s+\S+\s*=\s*(\d+)", trace):
        st = int(m.group(1))
        if st in (1, 6):
            feat.has_b_slices = True
            break

    # Check for MMCO operations
    mmco_ops = set()
    for m in re.finditer(r"memory_management_control_operation\s+\S+\s*=\s*(\d+)", trace):
        op = int(m.group(1))
        if op > 0:
            mmco_ops.add(op)
            feat.has_mmco = True
    feat.mmco_ops_used = sorted(mmco_ops)

    return feat


# ---------------------------------------------------------------------------
# Conformance comparison
# ---------------------------------------------------------------------------

@dataclass
class ConformanceResult:
    """Result of comparing one file between wedeo and FFmpeg."""
    filename: str
    status: str  # BITEXACT, DIFF, FAIL, SKIP
    wedeo_frames: int = 0
    ffmpeg_frames: int = 0
    matching_frames: int = 0
    features: H264Features | None = None
    unsupported: list[str] = field(default_factory=list)
    error: str = ""


def run_framecrc(cmd: list[str], env: dict | None = None) -> list[str]:
    """Run a command and return framecrc CRC values."""
    full_env = {**os.environ, **(env or {})}
    try:
        result = subprocess.run(
            cmd, capture_output=True, env=full_env, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return []
    crcs = []
    for line in result.stdout.decode(errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            crcs.append(parts[5].strip())
    return crcs


def compare_file(
    input_path: Path,
    wedeo_bin: str,
    features: H264Features | None = None,
) -> ConformanceResult:
    """Compare one file between wedeo and FFmpeg."""
    name = input_path.name

    # Detect features if not provided
    if features is None:
        try:
            features = detect_features(input_path)
        except Exception as e:
            return ConformanceResult(
                filename=name, status="FAIL", error=f"Feature detection failed: {e}",
            )

    # Run wedeo
    wedeo_crcs = run_framecrc([wedeo_bin, str(input_path)])

    # Run FFmpeg
    ffmpeg_crcs = run_framecrc([
        "ffmpeg", "-bitexact", "-i", str(input_path), "-f", "framecrc", "-",
    ])

    w_count = len(wedeo_crcs)
    f_count = len(ffmpeg_crcs)

    if w_count == 0 and f_count == 0:
        return ConformanceResult(
            filename=name, status="BITEXACT", features=features,
            ffmpeg_frames=0, wedeo_frames=0,
        )

    if w_count == 0:
        unsupported = features.unsupported_features()
        if unsupported:
            return ConformanceResult(
                filename=name, status="SKIP", features=features,
                ffmpeg_frames=f_count, unsupported=unsupported,
            )
        return ConformanceResult(
            filename=name, status="FAIL", features=features,
            ffmpeg_frames=f_count, error="Wedeo produced 0 frames",
        )

    # Compare CRCs
    match_count = sum(
        1 for w, f in zip(wedeo_crcs, ffmpeg_crcs) if w == f
    )

    if match_count == w_count == f_count:
        status = "BITEXACT"
    else:
        status = "DIFF"

    return ConformanceResult(
        filename=name, status=status, features=features,
        wedeo_frames=w_count, ffmpeg_frames=f_count,
        matching_frames=match_count,
        unsupported=features.unsupported_features(),
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_text_report(results: list[ConformanceResult], show_features: bool) -> str:
    """Format results as a human-readable text report."""
    lines = []

    bitexact = [r for r in results if r.status == "BITEXACT"]
    diff = [r for r in results if r.status == "DIFF"]
    fail = [r for r in results if r.status == "FAIL"]
    skip = [r for r in results if r.status == "SKIP"]

    lines.append(f"H.264 Conformance Report: {len(bitexact)} BITEXACT, "
                 f"{len(diff)} DIFF, {len(fail)} FAIL, {len(skip)} SKIP "
                 f"(of {len(results)} files)")
    lines.append("")

    if bitexact:
        lines.append(f"=== BITEXACT ({len(bitexact)}) ===")
        for r in bitexact:
            feat_str = ""
            if show_features and r.features:
                f = r.features
                parts = [f.profile_name, f.entropy_coding.upper()]
                if f.has_b_slices:
                    parts.append("B")
                if f.max_num_ref_frames > 1:
                    parts.append(f"ref={f.max_num_ref_frames}")
                if f.has_mmco:
                    parts.append(f"MMCO({','.join(map(str, f.mmco_ops_used))})")
                feat_str = f"  [{', '.join(parts)}]"
            lines.append(f"  {r.filename} ({r.wedeo_frames} frames){feat_str}")
        lines.append("")

    if diff:
        lines.append(f"=== DIFF ({len(diff)}) ===")
        for r in diff:
            match_str = f"{r.matching_frames}/{r.ffmpeg_frames}"
            unsup = ""
            if r.unsupported:
                unsup = f"  [{'; '.join(r.unsupported)}]"
            feat_str = ""
            if show_features and r.features:
                f = r.features
                parts = [f.profile_name, f.entropy_coding.upper()]
                if f.poc_type != 0:
                    parts.append(f"POC{f.poc_type}")
                if f.max_num_ref_frames > 1:
                    parts.append(f"ref={f.max_num_ref_frames}")
                if f.has_mmco:
                    parts.append(f"MMCO({','.join(map(str, f.mmco_ops_used))})")
                feat_str = f"  ({', '.join(parts)})"
            lines.append(f"  {r.filename}: {match_str} match{feat_str}{unsup}")
        lines.append("")

    if fail:
        lines.append(f"=== FAIL ({len(fail)}) ===")
        for r in fail:
            lines.append(f"  {r.filename}: {r.error}")
        lines.append("")

    if skip:
        lines.append(f"=== SKIP ({len(skip)}) ===")
        for r in skip:
            unsup = "; ".join(r.unsupported)
            lines.append(f"  {r.filename}: {unsup} (FFmpeg: {r.ffmpeg_frames} frames)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="H.264 conformance report: wedeo vs FFmpeg",
    )
    parser.add_argument("--dir", default="fate-suite/h264-conformance",
                        help="Conformance file directory")
    parser.add_argument("--prefix", default=None,
                        help="Only test files starting with this prefix")
    parser.add_argument("--features", action="store_true",
                        help="Show feature details for each file")
    parser.add_argument("--failures", action="store_true",
                        help="Only show DIFF and FAIL results")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    parser.add_argument("--cavlc-only", action="store_true",
                        help="Only test CAVLC files (skip CABAC)")
    parser.add_argument("--progressive-only", action="store_true",
                        help="Only test progressive files (skip interlaced)")
    parser.add_argument("--only-failing", action="store_true",
                        help="Only test files that were DIFF/FAIL in a previous run (fast mode)")
    parser.add_argument("--cache", default=None,
                        help="Path to cache previous results (default: .conformance-cache.json)")
    args = parser.parse_args()

    conformance_dir = Path(args.dir)
    if not conformance_dir.exists():
        print(f"Error: {conformance_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Find wedeo binary
    wedeo_bin = str(find_wedeo_binary())

    # Collect files
    files = sorted(conformance_dir.iterdir())
    files = [f for f in files if f.is_file() and f.suffix in (
        ".264", ".jsv", ".h264", ".26l", ".avc",
    )]

    if args.prefix:
        prefix_lower = args.prefix.lower()
        files = [f for f in files if f.name.lower().startswith(prefix_lower)]

    if not files:
        print("No conformance files found", file=sys.stderr)
        sys.exit(1)

    # Load cached results for --only-failing mode.
    # Cache is invalidated if the wedeo binary is newer than the cache file.
    cache_path = Path(args.cache) if args.cache else Path(".conformance-cache.json")
    cached_pass: set[str] = set()
    if args.only_failing and cache_path.exists():
        try:
            binary_mtime = Path(wedeo_bin).stat().st_mtime
            cache_mtime = cache_path.stat().st_mtime
            if binary_mtime > cache_mtime:
                print("Cache invalidated (wedeo binary is newer)", file=sys.stderr)
            else:
                cached = json.loads(cache_path.read_text())
                cached_pass = {
                    r["filename"] for r in cached if r.get("status") == "BITEXACT"
                }
                skip_count = len(cached_pass & {f.name for f in files})
                print(f"Loaded {len(cached_pass)} cached BITEXACT, "
                      f"re-testing {len(files) - skip_count} files",
                      file=sys.stderr)
        except Exception:
            pass  # ignore corrupt cache or missing binary

    print(f"Testing {len(files)} files...", file=sys.stderr)

    results = []
    for i, input_path in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {input_path.name}...",
              end="", file=sys.stderr, flush=True)

        # Fast path: skip feature detection and comparison for cached BITEXACT
        if args.only_failing and input_path.name in cached_pass:
            results.append(ConformanceResult(
                filename=input_path.name, status="BITEXACT",
            ))
            print(" BITEXACT (cached)", file=sys.stderr)
            continue

        # Detect features (for filtering and reporting)
        try:
            features = detect_features(input_path)
        except Exception:
            features = H264Features()

        # Apply filters
        if args.cavlc_only and not features.is_cavlc:
            print(" skip (CABAC)", file=sys.stderr)
            continue
        if args.progressive_only and not features.is_progressive:
            print(" skip (interlaced)", file=sys.stderr)
            continue

        result = compare_file(input_path, wedeo_bin, features)
        results.append(result)
        print(f" {result.status}", file=sys.stderr)

    if args.json:
        # JSON output
        json_results = []
        for r in results:
            d = {
                "filename": r.filename,
                "status": r.status,
                "wedeo_frames": r.wedeo_frames,
                "ffmpeg_frames": r.ffmpeg_frames,
                "matching_frames": r.matching_frames,
                "unsupported": r.unsupported,
                "error": r.error,
            }
            if r.features:
                d["features"] = {
                    "profile": r.features.profile_name,
                    "profile_idc": r.features.profile_idc,
                    "entropy": r.features.entropy_coding,
                    "poc_type": r.features.poc_type,
                    "max_refs": r.features.max_num_ref_frames,
                    "progressive": r.features.is_progressive,
                    "b_slices": r.features.has_b_slices,
                    "constrained_intra": r.features.constrained_intra_pred,
                    "weighted_pred": r.features.weighted_pred,
                    "direct_8x8": r.features.direct_8x8_inference,
                    "mmco_ops": r.features.mmco_ops_used,
                    "slice_groups": r.features.num_slice_groups,
                }
            json_results.append(d)
        print(json.dumps(json_results, indent=2))
    else:
        if args.failures:
            results = [r for r in results if r.status in ("DIFF", "FAIL")]
        print(format_text_report(results, args.features))

    # Save cache for --only-failing mode
    cache_data = [
        {"filename": r.filename, "status": r.status,
         "wedeo_frames": r.wedeo_frames, "matching_frames": r.matching_frames}
        for r in results
    ]
    try:
        cache_path.write_text(json.dumps(cache_data, indent=2))
    except Exception:
        pass  # don't fail on cache write errors


if __name__ == "__main__":
    main()
