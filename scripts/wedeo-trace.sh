#!/usr/bin/env bash
# Run wedeo-framecrc with tracing enabled, cleanly separating outputs.
#
# This script handles the entire build + run + trace workflow:
#   1. Builds wedeo-framecrc with tracing support (if needed)
#   2. Runs it with the specified RUST_LOG level
#   3. Captures framecrc output (stdout) to a file or /dev/null
#   4. Captures trace output (stderr) to a file, ANSI-stripped
#   5. Optionally greps the trace for a pattern
#
# Usage:
#   # Decode and capture trace (default: debug level, trace to /tmp/wedeo_trace.log)
#   scripts/wedeo-trace.sh input.264
#
#   # Custom trace level
#   scripts/wedeo-trace.sh input.264 --level trace
#
#   # Save framecrc output too
#   scripts/wedeo-trace.sh input.264 --framecrc /tmp/framecrc.txt
#
#   # Grep the trace after capture
#   scripts/wedeo-trace.sh input.264 --grep "frame complete"
#
#   # Disable deblocking
#   scripts/wedeo-trace.sh input.264 --no-deblock
#
#   # Custom trace file location
#   scripts/wedeo-trace.sh input.264 --trace /tmp/my_trace.log
#
#   # Skip rebuild (use existing binary)
#   scripts/wedeo-trace.sh input.264 --no-build
#
#   # Just build, don't run
#   scripts/wedeo-trace.sh --build-only
#
# The trace file is always ANSI-stripped for easy grepping.
# Use trace_grep.sh for ad-hoc grepping of existing trace files.

set -eo pipefail

# Defaults
TRACE_FILE="/tmp/wedeo_trace.log"
FRAMECRC_FILE=""
RUST_LOG_LEVEL="debug"
NO_DEBLOCK=""
GREP_PATTERN=""
NO_BUILD=""
BUILD_ONLY=""
INPUT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --level)
            RUST_LOG_LEVEL="$2"
            shift 2
            ;;
        --trace)
            TRACE_FILE="$2"
            shift 2
            ;;
        --framecrc)
            FRAMECRC_FILE="$2"
            shift 2
            ;;
        --grep)
            GREP_PATTERN="$2"
            shift 2
            ;;
        --no-deblock)
            NO_DEBLOCK=1
            shift
            ;;
        --no-build)
            NO_BUILD=1
            shift
            ;;
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --help|-h)
            cat <<'HELP'
Run wedeo-framecrc with tracing, cleanly separating outputs.

Usage: wedeo-trace.sh <input.264> [options]

Options:
  --level LEVEL    RUST_LOG level (default: debug)
  --trace FILE     Trace output file (default: /tmp/wedeo_trace.log)
  --framecrc FILE  Save framecrc output to file (default: /dev/null)
  --grep PATTERN   Grep the trace after capture
  --no-deblock     Set WEDEO_NO_DEBLOCK=1
  --no-build       Skip cargo build step
  --build-only     Build but don't run

The trace file is ANSI-stripped for easy grepping.
HELP
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
        *)
            INPUT_FILE="$1"
            shift
            ;;
    esac
done

WEDEO_BIN="target/debug/wedeo-framecrc"

# Step 1: Build with tracing (unless --no-build)
if [[ -z "$NO_BUILD" ]]; then
    echo "Building wedeo-framecrc with tracing..." >&2
    # Redirect cargo's build output to stderr so it doesn't pollute
    # the caller's stdout, but also capture it in case of errors.
    BUILD_LOG=$(mktemp)
    if ! cargo build --bin wedeo-framecrc -p wedeo-fate --features tracing 2>"$BUILD_LOG"; then
        echo "Build failed:" >&2
        cat "$BUILD_LOG" >&2
        rm -f "$BUILD_LOG"
        exit 1
    fi
    rm -f "$BUILD_LOG"
    echo "Build OK." >&2
fi

if [[ -n "$BUILD_ONLY" ]]; then
    echo "Binary: $WEDEO_BIN" >&2
    exit 0
fi

# Validate input file
if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: no input file specified." >&2
    echo "Usage: $0 <input.264> [options]" >&2
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE not found" >&2
    exit 1
fi

if [[ ! -x "$WEDEO_BIN" ]]; then
    echo "Error: $WEDEO_BIN not found. Run without --no-build." >&2
    exit 1
fi

# Step 2: Set up environment
export RUST_LOG="$RUST_LOG_LEVEL"
if [[ -n "$NO_DEBLOCK" ]]; then
    export WEDEO_NO_DEBLOCK=1
fi

# Step 3: Run with clean output separation
#
# Key insight: cargo run's build output goes to stderr, which mixes with
# tracing output. By building separately (step 1) and running the binary
# directly, stderr contains ONLY trace output.
#
# We pipe stderr through sed to strip ANSI codes in real-time, writing
# the clean trace to TRACE_FILE. Stdout (framecrc) goes to FRAMECRC_FILE
# or /dev/null.

FRAMECRC_DST="${FRAMECRC_FILE:-/dev/null}"

echo "Running: RUST_LOG=$RUST_LOG $WEDEO_BIN $INPUT_FILE" >&2
echo "Trace → $TRACE_FILE" >&2
if [[ -n "$FRAMECRC_FILE" ]]; then
    echo "Framecrc → $FRAMECRC_FILE" >&2
fi

# Run the binary directly (not cargo run) to avoid cargo stderr pollution.
# Strip ANSI from stderr and write to trace file.
"$WEDEO_BIN" "$INPUT_FILE" \
    1>"$FRAMECRC_DST" \
    2> >(sed 's/\x1b\[[0-9;]*m//g' > "$TRACE_FILE")

# Wait for the background sed process to finish
wait

TRACE_LINES=$(wc -l < "$TRACE_FILE")
echo "Trace captured: $TRACE_LINES lines in $TRACE_FILE" >&2

if [[ "$TRACE_LINES" -eq 0 ]]; then
    echo "WARNING: trace is empty. Binary may not have tracing compiled in." >&2
    echo "  Rebuild: cargo build --bin wedeo-framecrc -p wedeo-fate --features tracing" >&2
fi

# Step 4: Optional grep
if [[ -n "$GREP_PATTERN" ]]; then
    echo "--- grep '$GREP_PATTERN' ---" >&2
    grep -- "$GREP_PATTERN" "$TRACE_FILE" || echo "(no matches)" >&2
fi
