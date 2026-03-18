#!/usr/bin/env bash
# Strip ANSI escape codes from a trace log file and grep for a pattern.
#
# Usage:
#   scripts/trace_grep.sh "MB type parsed" /tmp/trace.log
#   scripts/trace_grep.sh "frame complete" /tmp/trace.log | head -10
#   scripts/trace_grep.sh -c "mb_skip_run" /tmp/trace.log   # count matches
#
# If no file is given, reads from stdin:
#   RUST_LOG=trace ./target/debug/wedeo-framecrc input.264 2>&1 | scripts/trace_grep.sh "frame complete"
#
# Passes extra flags through to grep. Common:
#   -c    count matches
#   -i    case insensitive
#   -v    invert match
#   -A N  show N lines after match
#   -B N  show N lines before match

set -eo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 [-grep-flags] <pattern> [trace-file]" >&2
    echo "  Strips ANSI codes then greps. Reads stdin if no file given." >&2
    exit 1
fi

# Parse arguments: last arg may be a file, second-to-last is pattern,
# everything else is grep flags.
args=("$@")
n=${#args[@]}

grep_args=()
pattern=""
file=""

if [ "$n" -eq 1 ]; then
    pattern="${args[0]}"
elif [ -f "${args[$((n-1))]}" ]; then
    file="${args[$((n-1))]}"
    pattern="${args[$((n-2))]}"
    for ((i=0; i<n-2; i++)); do
        grep_args+=("${args[$i]}")
    done
else
    pattern="${args[$((n-1))]}"
    for ((i=0; i<n-1; i++)); do
        grep_args+=("${args[$i]}")
    done
fi

strip_ansi() {
    sed 's/\x1b\[[0-9;]*m//g'
}

if [ -n "$file" ]; then
    if [ ${#grep_args[@]} -gt 0 ]; then
        strip_ansi < "$file" | grep "${grep_args[@]}" -- "$pattern"
    else
        strip_ansi < "$file" | grep -- "$pattern"
    fi
else
    if [ ${#grep_args[@]} -gt 0 ]; then
        strip_ansi | grep "${grep_args[@]}" -- "$pattern"
    else
        strip_ansi | grep -- "$pattern"
    fi
fi
