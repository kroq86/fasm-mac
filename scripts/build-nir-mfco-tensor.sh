#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/fasm/build/out/nir-mfco-tensor-train}"
OBJ_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nir-mfco-tensor-build.XXXXXX")"
trap 'rm -rf "$OBJ_DIR"' EXIT
mkdir -p "$(dirname "$OUT")"
for src in tensor_matmul_f32 tensor_tape_f32 tensor_ops_f32 tensor_tape_exec_f32 tensor_sgd_f32 tensor_plan_f32 tensor_fused_f32; do
  fasm --emit=macho-obj "$ROOT/fasm/spikes/$src.asm" "$OBJ_DIR/$src.o" >/dev/null
done
clang -arch x86_64 -O2 -c "$ROOT/fasm/spikes/tensor_plan_fused_compile.c" -o "$OBJ_DIR/tensor_plan_fused_compile.o"
clang -arch x86_64 -O2 -c "$ROOT/fasm/spikes/nir_mfco_tensor_train.c" -o "$OBJ_DIR/nir_mfco_tensor_train.o"
clang -arch x86_64 "$OBJ_DIR/nir_mfco_tensor_train.o" "$OBJ_DIR"/tensor_*.o -o "$OUT"
chmod +x "$OUT"
printf 'Built %s\n' "$OUT"
