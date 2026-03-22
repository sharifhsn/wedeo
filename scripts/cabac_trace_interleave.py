#!/usr/bin/env python3
"""Merge CABAC trace events into a single chronological stream.

Parses a wedeo CABAC trace log (produced with --features cabac-trace and
RUST_LOG=wedeo_codec_h264::cabac=trace) and interleaves all event types
(BIN, BYPASS, BYPASS_SIGN, TERM, MB_START, SECTION, SKIP_DECODE, etc.)
into a unified timeline with global sequential indices.

This makes it easy to see exactly what happens between two events of
different types — e.g., which BYPASS_SIGN calls happen between BIN[79]
and BYPASS[0].

IMPORTANT: Each event type (BIN, BYPASS, BYPASS_SIGN) has its own counter
capped by CABAC_MAX_BINS. TERM and metadata events (MB_START, SECTION) are
always printed. When BIN events are capped (e.g., CABAC_MAX_BINS=200) but
BYPASS events are not yet capped, the log ordering becomes misleading —
BYPASS events appear adjacent to SECTION markers even though many
unprintable BIN events happened in between. For accurate interleaving,
set CABAC_MAX_BINS high enough to capture ALL events of ALL types.

Usage:
    # Basic: interleave all events from a wedeo trace log
    python3 scripts/cabac_trace_interleave.py /tmp/wedeo_cabac.log

    # Show only events around the first BYPASS event (5 before/after)
    python3 scripts/cabac_trace_interleave.py /tmp/wedeo_cabac.log \\
        --around BYPASS:0 --context 5

    # Limit total output
    python3 scripts/cabac_trace_interleave.py /tmp/wedeo_cabac.log --max-events 200

    # Generate the trace log first, then interleave:
    RUST_LOG=wedeo_codec_h264::cabac=trace CABAC_MAX_BINS=500 \\
        cargo run --bin wedeo-framecrc -p wedeo-fate --features cabac-trace \\
        -- file.264 2>/tmp/wedeo_cabac.log
    python3 scripts/cabac_trace_interleave.py /tmp/wedeo_cabac.log --around BYPASS:0

Output format:
    [0] BIN 0: state=76 low=29245952 range=510 -> bit=0
    [1] MB_START: mb_x=0 mb_y=0 slice_type=I
    [2] INTRA_MB_TYPE: ctx_base=3 ctx=0 state_idx=3
    [3] SECTION: intra_pred_modes
    ...
    [120] BYPASS_SIGN 0: low=33460224 range=326 val=-1 -> bit=0
    [121] BIN 82: state=46 low=24190976 range=326 -> bit=0
    ...
    [500] TERM 0: low=18141184 range=418 -> result=0
    [501] MB_START: mb_x=1 mb_y=0 slice_type=I
"""

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class Event:
    """A single CABAC trace event."""
    global_idx: int
    kind: str           # BIN, BYPASS, BYPASS_SIGN, TERM, MB_START, SECTION, etc.
    type_idx: Optional[int]  # per-type sequential index (e.g., BIN 42)
    summary: str        # compact one-line summary


# Regex patterns for each event type.
# These use .search() so they work with tracing's timestamp/target prefix.
PATTERNS = [
    ("BIN", re.compile(
        r"CABAC_BIN\s+(\d+)\s+state=(-?\d+)\s+low=(-?\d+)\s+range=(-?\d+)"
        r"\s+->\s+bit=(-?\d+)\s+post_low=(-?\d+)\s+post_range=(-?\d+)"
    )),
    ("BYPASS", re.compile(
        r"CABAC_BYPASS\s+(\d+)\s+low=(-?\d+)\s+range=(-?\d+)"
        r"\s+->\s+bit=(-?\d+)\s+post_low=(-?\d+)"
    )),
    ("BYPASS_SIGN", re.compile(
        r"CABAC_BYPASS_SIGN\s+(\d+)\s+low=(-?\d+)\s+range=(-?\d+)"
        r"\s+val=(-?\d+)\s+->\s+bit=(-?\d+)\s+result=(-?\d+)\s+post_low=(-?\d+)"
    )),
    ("TERM", re.compile(
        r"CABAC_TERM\s+(\d+)\s+low=(-?\d+)\s+range=(-?\d+)"
        r"\s+->\s+result=(-?\d+)\s+post_low=(-?\d+)\s+post_range=(-?\d+)"
    )),
    ("MB_START", re.compile(
        r"CABAC_MB_START\s+(mb_x=\d+\s+mb_y=\d+\s+slice_type=\S+)"
    )),
    ("SKIP_DECODE", re.compile(
        r"CABAC_SKIP_DECODE\s+(mb_x=\d+\s+mb_y=\d+\s+is_b=\S+)"
    )),
    ("SECTION", re.compile(
        r"CABAC_SECTION\s+(.+)"
    )),
    ("INTRA_MB_TYPE", re.compile(
        r"CABAC_INTRA_MB_TYPE\s+(.+)"
    )),
    ("B_MB_TYPE", re.compile(
        r"CABAC_B_MB_TYPE\s+(.+)"
    )),
]


def format_bin(m: re.Match) -> str:
    return (f"state={m.group(2)} low={m.group(3)} range={m.group(4)} "
            f"-> bit={m.group(5)}")


def format_bypass(m: re.Match) -> str:
    return f"low={m.group(2)} range={m.group(3)} -> bit={m.group(4)}"


def format_bypass_sign(m: re.Match) -> str:
    return (f"low={m.group(2)} range={m.group(3)} val={m.group(4)} "
            f"-> bit={m.group(5)}")


def format_term(m: re.Match) -> str:
    return (f"low={m.group(2)} range={m.group(3)} -> result={m.group(4)}")


def format_simple(m: re.Match) -> str:
    return m.group(1).strip()


FORMATTERS = {
    "BIN": format_bin,
    "BYPASS": format_bypass,
    "BYPASS_SIGN": format_bypass_sign,
    "TERM": format_term,
    "MB_START": format_simple,
    "SKIP_DECODE": format_simple,
    "SECTION": format_simple,
    "INTRA_MB_TYPE": format_simple,
    "B_MB_TYPE": format_simple,
}


def parse_events(path: str, max_events: int) -> list[Event]:
    """Parse a trace log into a chronological list of events."""
    events: list[Event] = []
    global_idx = 0

    with open(path) as f:
        for line in f:
            if global_idx >= max_events:
                break

            for kind, pattern in PATTERNS:
                m = pattern.search(line)
                if m:
                    # Extract per-type index (first capture group is the index
                    # for BIN/BYPASS/BYPASS_SIGN/TERM; None for others)
                    type_idx = None
                    if kind in ("BIN", "BYPASS", "BYPASS_SIGN", "TERM"):
                        type_idx = int(m.group(1))

                    summary = FORMATTERS[kind](m)
                    events.append(Event(
                        global_idx=global_idx,
                        kind=kind,
                        type_idx=type_idx,
                        summary=summary,
                    ))
                    global_idx += 1
                    break  # only match first pattern per line

    return events


def format_event(ev: Event) -> str:
    """Format an event as a single output line."""
    idx_str = f"[{ev.global_idx}]"
    if ev.type_idx is not None:
        kind_str = f"{ev.kind} {ev.type_idx}:"
    else:
        kind_str = f"{ev.kind}:"
    return f"{idx_str:>8} {kind_str:<20} {ev.summary}"


def find_event_index(events: list[Event], spec: str) -> Optional[int]:
    """Find the global index of an event specified as TYPE:N (e.g., BYPASS:0)."""
    parts = spec.split(":")
    if len(parts) != 2:
        return None
    kind = parts[0].upper()
    try:
        type_idx = int(parts[1])
    except ValueError:
        return None

    for ev in events:
        if ev.kind == kind and ev.type_idx == type_idx:
            return ev.global_idx
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge CABAC trace events into chronological stream"
    )
    parser.add_argument("trace_log", help="Wedeo CABAC trace log file")
    parser.add_argument(
        "--max-events", type=int, default=100000,
        help="Maximum events to parse (default: 100000)",
    )
    parser.add_argument(
        "--around", metavar="TYPE:N",
        help="Show events around a specific event (e.g., BYPASS:0, TERM:60)",
    )
    parser.add_argument(
        "--context", type=int, default=10,
        help="Number of events before/after --around target (default: 10)",
    )
    args = parser.parse_args()

    events = parse_events(args.trace_log, args.max_events)

    if not events:
        print("No CABAC events found in trace log.", file=sys.stderr)
        sys.exit(1)

    # Count by type
    counts: dict[str, int] = {}
    for ev in events:
        counts[ev.kind] = counts.get(ev.kind, 0) + 1

    print(f"Parsed {len(events)} events:", file=sys.stderr)
    for kind, count in sorted(counts.items()):
        print(f"  {kind}: {count}", file=sys.stderr)
    print(file=sys.stderr)

    # Determine output range
    start = 0
    end = len(events)

    if args.around:
        target_idx = find_event_index(events, args.around)
        if target_idx is None:
            print(
                f"Error: event '{args.around}' not found in trace.",
                file=sys.stderr,
            )
            sys.exit(1)
        # Find position in events list
        target_pos = None
        for i, ev in enumerate(events):
            if ev.global_idx == target_idx:
                target_pos = i
                break
        if target_pos is None:
            print(f"Error: event index {target_idx} not found.", file=sys.stderr)
            sys.exit(1)
        start = max(0, target_pos - args.context)
        end = min(len(events), target_pos + args.context + 1)
        print(
            f"Showing events [{events[start].global_idx}..{events[end-1].global_idx}] "
            f"around {args.around} (global idx {target_idx}):",
            file=sys.stderr,
        )
        print(file=sys.stderr)

    # Output
    for ev in events[start:end]:
        marker = ""
        if args.around:
            target_idx = find_event_index(events, args.around)
            if ev.global_idx == target_idx:
                marker = " <<<<<"
        print(format_event(ev) + marker)


if __name__ == "__main__":
    main()
