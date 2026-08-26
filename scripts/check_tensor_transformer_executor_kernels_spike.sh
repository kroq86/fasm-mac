#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-transformer-kernels.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -arch arm64 -O3 -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/tensor_transformer_executor_kernels_check.c" -o "$OUT_DIR/arm64"
"$OUT_DIR/arm64"
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OUT_DIR/executor.o" >/dev/null
clang -arch x86_64 -O3 -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/tensor_transformer_executor_kernels_check.c" "$OUT_DIR/executor.o" -o "$OUT_DIR/x86_64"
arch -x86_64 "$OUT_DIR/x86_64"
