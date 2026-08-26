#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'tensor NEON spike skipped: native arm64 host required'
  exit 0
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-neon-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -arch arm64 -O3 "$ROOT/fasm/spikes/tensor_neon_check.c" -o "$OUT_DIR/check"
"$OUT_DIR/check"
clang -arch arm64 -O3 -fno-vectorize -fno-slp-vectorize -DAUTO_VECTORIZE=0 \
  "$ROOT/fasm/spikes/tensor_neon_check.c" -o "$OUT_DIR/check-novec"
"$OUT_DIR/check-novec"
