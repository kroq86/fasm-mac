#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-merged-cnn-pool-mnist.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OUT_DIR/executor.o" >/dev/null
clang -arch x86_64 -std=c11 -Wall -Wextra -Werror -Wno-misleading-indentation -O2 \
  "$ROOT/fasm/spikes/tensor_merged_cnn_pool_mnist_check.c" "$OUT_DIR/executor.o" -o "$OUT_DIR/check"
arch -x86_64 "$OUT_DIR/check"
