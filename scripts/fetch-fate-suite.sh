#!/bin/bash
# Download FFmpeg FATE test samples.
# Usage: ./scripts/fetch-fate-suite.sh [destination]
set -euo pipefail

DEST="${1:-fate-suite}"
echo "Syncing FATE suite to $DEST/ ..."
rsync -avP rsync://fate-suite.ffmpeg.org/fate-suite/ "$DEST/"
echo "Done. Set FATE_SUITE=$DEST to run FATE tests."
