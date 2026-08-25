#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-fusion.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
for src in tensor_matmul_f32 tensor_tape_f32 tensor_ops_f32 tensor_fused_f32 tensor_plan_f32; do
  fasm --emit=macho-obj "$ROOT/fasm/spikes/$src.asm" "$OUT_DIR/$src.o" >/dev/null
done
clang -arch x86_64 -O2 "$ROOT/fasm/spikes/tensor_fusion_check.c" "$OUT_DIR"/*.o -o "$OUT_DIR/check"
arch -x86_64 "$OUT_DIR/check"
