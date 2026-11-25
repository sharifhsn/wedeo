#!/bin/sh
# Build just the command-line tools and media formats exercised by the native tests.
set -eu
root=$(git rev-parse --show-toplevel)
source_dir="$root/FFmpeg"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$(git -C "$root" rev-parse HEAD:FFmpeg)"
mkdir -p "$root/target/ffmpeg-oracle"
cd "$root/target/ffmpeg-oracle"
"$source_dir/configure" --disable-autodetect --disable-doc --disable-debug --disable-network \
    --disable-everything --enable-ffmpeg --enable-ffprobe --enable-protocol=file,pipe \
    --enable-demuxer=ivf,h264,mov,matroska,wav,pcm_s16le \
    --enable-decoder=h264,vp9,pcm_s16le,pcm_s24le,pcm_s32le,aac,rawvideo \
    --enable-encoder=rawvideo,pcm_s16le,pcm_s32le --enable-muxer=framecrc,framemd5,null,rawvideo,pcm_s16le,wav \
    --enable-filter=scale,format,aresample --enable-parser=h264,vp9,aac,av1
make -j2 ffmpeg ffprobe
