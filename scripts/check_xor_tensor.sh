#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/xor-tensor-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

for src in tensor_matmul_f32 tensor_tape_f32 tensor_ops_f32 tensor_tape_exec_f32 tensor_sgd_f32 tensor_plan_f32; do
  fasm --emit=macho-obj "$ROOT/fasm/spikes/$src.asm" "$OUT_DIR/$src.o" >/dev/null
done
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_fused_f32.asm" "$OUT_DIR/tensor_fused_f32.o" >/dev/null
clang -arch x86_64 -O2 -c "$ROOT/fasm/spikes/tensor_plan_fused_compile.c" -o "$OUT_DIR/tensor_plan_fused_compile.o"
fasm --emit=macho-obj "$ROOT/fasm/examples/xor_tensor_train.asm" "$OUT_DIR/xor.o" >/dev/null
clang -arch x86_64 "$OUT_DIR/xor.o" "$OUT_DIR"/tensor_*.o -o "$OUT_DIR/xor"
actual="$(arch -x86_64 "$OUT_DIR/xor")"
expected='xor trained plan_steps=3 fused_contexts=2 scratch_bytes=384 loss_milli=0 predictions_milli=0 999 999 0'
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL fused XOR differs from locked unfused baseline: %s\n' "$actual" >&2
  exit 1
fi
printf '%s\n' "$actual"
