#!/bin/sh
# Build the pure-Rust Wedeo WebAssembly module into web/wedeo.wasm.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
target_dir=${CARGO_TARGET_DIR:-"$root/bins/wedeo-wasm/target"}
cargo build --locked --target-dir "$target_dir" --manifest-path "$root/bins/wedeo-wasm/Cargo.toml" \
  --target wasm32-unknown-unknown \
  --profile wasm-release
cp "$target_dir/wasm32-unknown-unknown/wasm-release/wedeo_wasm.wasm" "$here/wedeo.wasm"
ls -lh "$here/wedeo.wasm"
