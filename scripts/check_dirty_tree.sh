#!/usr/bin/env bash
# Check for uncommitted changes in decoder source files at session start.
# Run this before making any code changes to avoid mixing pre-existing
# dirty state with new work.
#
# Usage: bash scripts/check_dirty_tree.sh

set -euo pipefail

DECODER_DIR="codecs/wedeo-codec-h264/src"

dirty_files=$(git diff --name-only -- "$DECODER_DIR" 2>/dev/null || true)

if [ -z "$dirty_files" ]; then
    echo "✓ H.264 decoder source is clean"
    exit 0
fi

echo "⚠ WARNING: H.264 decoder has uncommitted changes:"
echo ""
for f in $dirty_files; do
    lines_changed=$(git diff --stat -- "$f" | tail -1 | grep -oE '[0-9]+ insertion|[0-9]+ deletion' | paste -sd, - || echo "unknown")
    echo "  $f ($lines_changed)"
done
echo ""
echo "Consider: git stash or git diff $DECODER_DIR"
exit 1
