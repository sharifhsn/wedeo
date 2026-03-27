#!/usr/bin/env bash
# Shodh Memory Server Watchdog
#
# Detects when the shodh server enters a broken state (ONNX thread panics
# after user cache eviction) and automatically restarts it.
#
# The bug: shodh v0.1.91 panics on tokio worker threads when re-initializing
# a user's memory system after idle-timeout eviction. The ort crate's dlopen
# fails with bare "libonnxruntime.dylib" on re-init threads. The panics
# don't crash the server but leave it returning 500 errors for all requests.
#
# Install as a cron job:
#   crontab -e
#   */5 * * * * /Users/sharif/Code/wedeo/scripts/shodh_watchdog.sh
#
# Or run manually when Shodh MCP calls start failing.

set -euo pipefail

SHODH_URL="http://127.0.0.1:3030"
API_KEY="${SHODH_API_KEY:-sk-shodh-dev-default}"
ERR_LOG="/tmp/shodh-memory.err"
WATCHDOG_LOG="/tmp/shodh-watchdog.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WATCHDOG_LOG"
}

# Check 1: Is the server running at all?
if ! curl -sf "$SHODH_URL/health" > /dev/null 2>&1; then
    log "Server not responding to health check — launchd should restart it"
    exit 0
fi

# Check 2: Can we actually use the memory system?
# The /api/stats endpoint triggers user initialization, which is where the bug hits.
response=$(curl -sf -H "Authorization: Bearer $API_KEY" \
    "$SHODH_URL/api/stats?user_id=claude-code" 2>&1) || true

if echo "$response" | grep -q "Failed to initialize"; then
    log "BROKEN STATE DETECTED: $response"

    # Count panics in error log
    panic_count=0
    if [ -f "$ERR_LOG" ]; then
        panic_count=$(grep -c "panicked" "$ERR_LOG" 2>/dev/null || echo 0)
        log "Error log has $panic_count panics"
    fi

    # Kill the server — launchd KeepAlive will restart it
    log "Killing shodh server to trigger launchd restart..."
    pkill -f "shodh server" 2>/dev/null || true
    sleep 3

    # Clear the error log so panics don't accumulate across restarts
    > "$ERR_LOG" 2>/dev/null || true

    # Verify the restart worked
    sleep 2
    if curl -sf "$SHODH_URL/health" > /dev/null 2>&1; then
        verify=$(curl -sf -H "Authorization: Bearer $API_KEY" \
            "$SHODH_URL/api/stats?user_id=claude-code" 2>&1) || true
        if echo "$verify" | grep -q "total_memories"; then
            log "RECOVERY SUCCESS: Server restarted and user initialized"
        else
            log "RECOVERY PARTIAL: Server running but user init still failing: $verify"
        fi
    else
        log "RECOVERY PENDING: Waiting for launchd to restart server"
    fi
else
    # Server is healthy — no action needed
    # Only log if there are panics accumulating (early warning)
    if [ -f "$ERR_LOG" ]; then
        panic_count=$(grep -c "panicked" "$ERR_LOG" 2>/dev/null || echo 0)
        if [ "$panic_count" -gt 0 ]; then
            log "WARNING: $panic_count panics in error log but server still responding"
        fi
    fi
fi
