#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-matmul-spike.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

OBJ="$OUT_DIR/tensor_matmul.o"
BIN="$OUT_DIR/tensor_matmul_check"
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_matmul_f32.asm" "$OBJ" >/dev/null
clang -arch x86_64 -O2 "$ROOT/fasm/spikes/tensor_matmul_check.c" "$OBJ" -o "$BIN"
arch -x86_64 "$BIN"
