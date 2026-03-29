#!/usr/bin/env bash
# Publish all wedeo workspace crates to crates.io in topological order.
#
# Usage:
#   ./scripts/publish.sh          # dry-run (default)
#   ./scripts/publish.sh --execute # actually publish
#
# Each crate must wait for the crates.io index to propagate before the next
# dependent crate can be published. The script sleeps between publishes.
#
# Excluded from publishing:
#   - wedeo-rav1d: depends on rav1d via git (unreleased aarch64 NEON fixes)
#   - wedeo-cli, wedeo-play: binary crates
#   - wedeo-fate: test harness
set -euo pipefail

DRY_RUN=true
if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN=false
fi

# Topological publish order — each crate's deps must already be on crates.io.
CRATES=(
    wedeo-core
    wedeo-codec
    wedeo-format
    wedeo-filter
    wedeo-resample
    wedeo-scale
    wedeo-codec-pcm
    wedeo-codec-h264
    wedeo-format-wav
    wedeo-format-h264
    wedeo-format-mp4
    wedeo-symphonia
    wedeo
)

DELAY=30  # seconds between publishes for index propagation

for crate in "${CRATES[@]}"; do
    echo "=== $crate ==="
    if $DRY_RUN; then
        echo "  [dry-run] would publish $crate"
    else
        cargo publish -p "$crate"
        if [[ "$crate" != "wedeo" ]]; then
            echo "  waiting ${DELAY}s for crates.io index..."
            sleep "$DELAY"
        fi
    fi
    echo
done

if $DRY_RUN; then
    echo "Dry run complete. Pass --execute to publish for real."
else
    echo "All crates published!"
fi
