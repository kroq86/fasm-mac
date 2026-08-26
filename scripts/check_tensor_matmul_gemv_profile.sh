#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'tensor matmul gemv profile skipped: native arm64 host required'
  exit 0
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-matmul-gemv-profile.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -arch arm64 -O3 -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_matmul_gemv_profile.c" -o "$OUT_DIR/profile"
"$OUT_DIR/profile"
