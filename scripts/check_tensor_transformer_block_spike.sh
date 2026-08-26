#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-transformer-block.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_matmul_f32.asm" "$OUT_DIR/matmul.o" >/dev/null
clang -arch x86_64 -O2 "$ROOT/fasm/spikes/tensor_transformer_block_check.c" "$OUT_DIR/matmul.o" -o "$OUT_DIR/check"
arch -x86_64 "$OUT_DIR/check"
