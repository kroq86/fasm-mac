#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/fasm/build/out/tensorctl}"
OBJ_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-build.XXXXXX")"
trap 'rm -rf "$OBJ_DIR"' EXIT

mkdir -p "$(dirname "$OUT")"

fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OBJ_DIR/executor.o" >/dev/null
clang -arch x86_64 -O2 -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/examples/tensorctl_main.c" \
  "$ROOT/fasm/examples/tensorctl_mlp.c" \
  "$ROOT/fasm/examples/tensorctl_transformer.c" \
  "$OBJ_DIR/executor.o" \
  -o "$OUT"
chmod +x "$OUT"

printf 'Built %s\n' "$OUT"
