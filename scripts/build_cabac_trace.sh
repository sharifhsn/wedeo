#!/usr/bin/env bash
# Build a patched FFmpeg binary with CABAC bin tracing.
#
# This script patches FFmpeg's cabac_functions.h to add a fprintf trace
# on every get_cabac_inline call, rebuilds the ffmpeg binary, and copies
# it to scripts/build/ffmpeg_cabac_trace. The original file is restored
# after the build.
#
# The patch prints to stderr:
#   CABAC_BIN <n> ctx_state=<s> low=<low> range=<range> -> bit=<bit>
#
# Usage:
#   scripts/build_cabac_trace.sh          # build the patched binary
#   scripts/build_cabac_trace.sh --clean  # remove build artifacts
#
# Prerequisites:
#   FFmpeg must be configured for debug (already done in this repo):
#     cd FFmpeg && ./configure --disable-optimizations --enable-debug=3 \
#       --disable-stripping --disable-asm && make -j$(sysctl -n hw.ncpu) ffmpeg
#
# The patched binary is placed at:
#   scripts/build/ffmpeg_cabac_trace

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FFMPEG_DIR="$REPO_DIR/FFmpeg"
BUILD_DIR="$SCRIPT_DIR/build"
TARGET_BIN="$BUILD_DIR/ffmpeg_cabac_trace"

CABAC_FUNC_H="$FFMPEG_DIR/libavcodec/cabac_functions.h"
CABAC_FUNC_H_ORIG="$CABAC_FUNC_H.orig"

# --- Clean mode ---
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning up..."
    # Restore original if backup exists
    if [[ -f "$CABAC_FUNC_H_ORIG" ]]; then
        mv "$CABAC_FUNC_H_ORIG" "$CABAC_FUNC_H"
        echo "Restored original cabac_functions.h"
    fi
    rm -f "$TARGET_BIN"
    echo "Removed $TARGET_BIN"
    exit 0
fi

# --- Validate ---
if [[ ! -f "$CABAC_FUNC_H" ]]; then
    echo "Error: $CABAC_FUNC_H not found." >&2
    echo "Is FFmpeg checked out at $FFMPEG_DIR?" >&2
    exit 1
fi

if [[ ! -f "$FFMPEG_DIR/ffbuild/config.mak" ]]; then
    echo "Error: FFmpeg is not configured." >&2
    echo "Run: cd $FFMPEG_DIR && ./configure --disable-optimizations --enable-debug=3 --disable-stripping --disable-asm" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

# --- Backup original ---
if [[ ! -f "$CABAC_FUNC_H_ORIG" ]]; then
    cp "$CABAC_FUNC_H" "$CABAC_FUNC_H_ORIG"
    echo "Backed up original cabac_functions.h"
else
    # Restore from backup before patching (in case previous run failed mid-build)
    cp "$CABAC_FUNC_H_ORIG" "$CABAC_FUNC_H"
fi

# --- Apply patch ---
# We replace get_cabac_inline to add trace prints before and after the decode.
# The trace uses a static counter to number bins sequentially.
#
# Strategy: insert a #include <stdio.h> at the top and wrap the existing
# get_cabac_inline function body with trace prints.
#
# We use sed to:
# 1. Add #include <stdio.h> after the first #include
# 2. Add trace variables and pre-print after the LPS_RANGE lookup
# 3. Add post-print before the return

echo "Patching cabac_functions.h for CABAC bin tracing..."

# Create the patched file using Python for reliability (sed varies across platforms)
python3 - "$CABAC_FUNC_H" <<'PATCH_SCRIPT'
import sys

filepath = sys.argv[1]
with open(filepath, 'r') as f:
    content = f.read()

# 1. Add #include <stdio.h> after the first #include <stdint.h>
content = content.replace(
    '#include <stdint.h>',
    '#include <stdint.h>\n#include <stdio.h>',
    1
)

# 2. Replace the get_cabac_inline function with a traced version.
#    We find the #ifndef get_cabac_inline ... function and replace it entirely.
#
#    The original function (non-arch-specific) is:
#      static av_always_inline int get_cabac_inline(CABACContext *c, uint8_t * const state){
#          int s = *state;
#          ...
#          return bit;
#      }
#
#    We wrap it with trace prints.

old_func = '''static av_always_inline int get_cabac_inline(CABACContext *c, uint8_t * const state){
    int s = *state;
    int RangeLPS= ff_h264_lps_range[2*(c->range&0xC0) + s];
    int bit, lps_mask;

    c->range -= RangeLPS;
    lps_mask= ((c->range<<(CABAC_BITS+1)) - c->low)>>31;

    c->low -= (c->range<<(CABAC_BITS+1)) & lps_mask;
    c->range += (RangeLPS - c->range) & lps_mask;

    s^=lps_mask;
    *state= (ff_h264_mlps_state+128)[s];
    bit= s&1;

    lps_mask= ff_h264_norm_shift[c->range];
    c->range<<= lps_mask;
    c->low  <<= lps_mask;
    if(!(c->low & CABAC_MASK))
        refill2(c);
    return bit;
}'''

new_func = '''static av_always_inline int get_cabac_inline(CABACContext *c, uint8_t * const state){
    static int _cabac_bin_count = 0;
    static int _cabac_max_bins = -1;
    if (_cabac_max_bins < 0) {
        const char *env = getenv("CABAC_MAX_BINS");
        _cabac_max_bins = env ? atoi(env) : 10000;
    }
    int _pre_state = *state;
    int _pre_low = c->low;
    int _pre_range = c->range;

    int s = *state;
    int RangeLPS= ff_h264_lps_range[2*(c->range&0xC0) + s];
    int bit, lps_mask;

    c->range -= RangeLPS;
    lps_mask= ((c->range<<(CABAC_BITS+1)) - c->low)>>31;

    c->low -= (c->range<<(CABAC_BITS+1)) & lps_mask;
    c->range += (RangeLPS - c->range) & lps_mask;

    s^=lps_mask;
    *state= (ff_h264_mlps_state+128)[s];
    bit= s&1;

    lps_mask= ff_h264_norm_shift[c->range];
    c->range<<= lps_mask;
    c->low  <<= lps_mask;
    if(!(c->low & CABAC_MASK))
        refill2(c);

    if (_cabac_bin_count < _cabac_max_bins) {
        fprintf(stderr, "CABAC_BIN %d state=%d low=%d range=%d -> bit=%d post_low=%d post_range=%d\\n",
                _cabac_bin_count, _pre_state, _pre_low, _pre_range, bit, c->low, c->range);
    }
    _cabac_bin_count++;
    return bit;
}'''

if old_func not in content:
    print("WARNING: Could not find exact get_cabac_inline function to patch.", file=sys.stderr)
    print("The function may have changed. Check cabac_functions.h manually.", file=sys.stderr)
    sys.exit(1)

content = content.replace(old_func, new_func)

# 3. Also patch get_cabac_bypass to trace bypass bins
old_bypass = '''av_unused static int get_cabac_bypass(CABACContext *c){
    int range;
    c->low += c->low;

    if(!(c->low & CABAC_MASK))
        refill(c);

    range= c->range<<(CABAC_BITS+1);
    if(c->low < range){
        return 0;
    }else{
        c->low -= range;
        return 1;
    }
}'''

new_bypass = '''av_unused static int get_cabac_bypass(CABACContext *c){
    static int _cabac_bypass_count = 0;
    static int _cabac_max_bins_bp = -1;
    if (_cabac_max_bins_bp < 0) {
        const char *env = getenv("CABAC_MAX_BINS");
        _cabac_max_bins_bp = env ? atoi(env) : 10000;
    }
    int _pre_low = c->low;
    int _pre_range = c->range;

    int range;
    c->low += c->low;

    if(!(c->low & CABAC_MASK))
        refill(c);

    range= c->range<<(CABAC_BITS+1);
    int bit;
    if(c->low < range){
        bit = 0;
    }else{
        c->low -= range;
        bit = 1;
    }

    if (_cabac_bypass_count < _cabac_max_bins_bp) {
        fprintf(stderr, "CABAC_BYPASS %d low=%d range=%d -> bit=%d post_low=%d\\n",
                _cabac_bypass_count, _pre_low, _pre_range, bit, c->low);
    }
    _cabac_bypass_count++;
    return bit;
}'''

if old_bypass not in content:
    print("WARNING: Could not find exact get_cabac_bypass function to patch.", file=sys.stderr)
    print("Bypass tracing will not be available.", file=sys.stderr)
else:
    content = content.replace(old_bypass, new_bypass)

# 4. Also patch get_cabac_terminate
old_term = '''av_unused static int get_cabac_terminate(CABACContext *c){
    c->range -= 2;
    if(c->low < c->range<<(CABAC_BITS+1)){
        renorm_cabac_decoder_once(c);
        return 0;
    }else{
        return c->bytestream - c->bytestream_start;
    }
}'''

new_term = '''av_unused static int get_cabac_terminate(CABACContext *c){
    static int _cabac_term_count = 0;
    int _pre_low = c->low;
    int _pre_range = c->range;

    c->range -= 2;
    int result;
    if(c->low < c->range<<(CABAC_BITS+1)){
        renorm_cabac_decoder_once(c);
        result = 0;
    }else{
        result = c->bytestream - c->bytestream_start;
    }

    fprintf(stderr, "CABAC_TERM %d low=%d range=%d -> result=%d post_low=%d post_range=%d\\n",
            _cabac_term_count, _pre_low, _pre_range, result, c->low, c->range);
    _cabac_term_count++;
    return result;
}'''

if old_term not in content:
    print("WARNING: Could not find exact get_cabac_terminate function to patch.", file=sys.stderr)
    print("Terminate tracing will not be available.", file=sys.stderr)
else:
    content = content.replace(old_term, new_term)

with open(filepath, 'w') as f:
    f.write(content)

print("Patch applied successfully.")
PATCH_SCRIPT

# --- Rebuild FFmpeg ---
echo "Rebuilding FFmpeg with CABAC tracing (this may take a moment)..."
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Only rebuild the affected object files + relink ffmpeg binary.
# Touch the patched header to force recompilation of files that include it.
if ! make -C "$FFMPEG_DIR" -j"$NCPU" ffmpeg 2>&1 | tail -5; then
    echo "Error: FFmpeg rebuild failed." >&2
    # Restore original before exiting
    cp "$CABAC_FUNC_H_ORIG" "$CABAC_FUNC_H"
    echo "Restored original cabac_functions.h" >&2
    exit 1
fi

# --- Copy binary ---
cp "$FFMPEG_DIR/ffmpeg" "$TARGET_BIN"
echo "Patched binary: $TARGET_BIN"

# --- Restore original ---
cp "$CABAC_FUNC_H_ORIG" "$CABAC_FUNC_H"
echo "Restored original cabac_functions.h"

echo ""
echo "Usage:"
echo "  $TARGET_BIN -bitexact -i <file.264> -f null - 2>ffmpeg_cabac.log"
echo ""
echo "Control max bins via environment variable:"
echo "  CABAC_MAX_BINS=500 $TARGET_BIN -bitexact -i <file.264> -f null - 2>ffmpeg_cabac.log"
echo ""
echo "Then compare with:"
echo "  python3 scripts/cabac_bin_compare.py <file.264>"
