#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAFETENSORS="/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"

if [[ ! -f "$SAFETENSORS" ]]; then
  echo "safetensors_loader fixture (real gpt2 checkpoint) not present at $SAFETENSORS -- skipped"
  exit 0
fi

OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-safetensors-loader.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function \
  -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/tensor_safetensors_loader_differential_check.c" -o "$OUT_DIR/check"
"$OUT_DIR/check" "$SAFETENSORS"
