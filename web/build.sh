#!/bin/sh
# Build the pure-Rust Wedeo WebAssembly module into web/wedeo.wasm.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
cargo build --manifest-path "$root/Cargo.toml" \
  -p wedeo-wasm \
  --target wasm32-unknown-unknown \
  --profile wasm-release
cp "$root/target/wasm32-unknown-unknown/wasm-release/wedeo_wasm.wasm" "$here/wedeo.wasm"
ls -lh "$here/wedeo.wasm"
