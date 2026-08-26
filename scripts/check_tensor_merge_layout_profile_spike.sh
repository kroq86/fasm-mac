#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'tensor merge layout profile skipped: native arm64 host required'
  exit 0
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-merge-layout.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -arch arm64 -O3 "$ROOT/fasm/spikes/tensor_merge_layout_profile.c" -o "$OUT_DIR/vec"
clang -arch arm64 -O3 -fno-vectorize -fno-slp-vectorize "$ROOT/fasm/spikes/tensor_merge_layout_profile.c" -o "$OUT_DIR/novec"
"$OUT_DIR/vec"
"$OUT_DIR/novec"
