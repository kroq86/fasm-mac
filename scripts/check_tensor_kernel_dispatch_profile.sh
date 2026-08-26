#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'tensor kernel dispatch profile skipped: native arm64 host required'
  exit 0
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-dispatch-profile.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -arch arm64 -O3 -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_kernel_dispatch_profile.c" -o "$OUT_DIR/vec"
clang -arch arm64 -O3 -fno-vectorize -fno-slp-vectorize -DSCALAR_MODE=\"true_scalar\" -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_kernel_dispatch_profile.c" -o "$OUT_DIR/novec"
"$OUT_DIR/vec"
"$OUT_DIR/novec"
