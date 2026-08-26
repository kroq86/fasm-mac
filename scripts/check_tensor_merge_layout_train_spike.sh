#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'tensor merge layout train spike skipped: native arm64 host required'
  exit 0
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-merge-layout-train.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -arch arm64 -O3 -fno-vectorize -fno-slp-vectorize -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_merge_layout_train_check.c" -o "$OUT_DIR/check"
"$OUT_DIR/check"
