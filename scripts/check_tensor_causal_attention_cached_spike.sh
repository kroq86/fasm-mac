#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-causal-attention-cached.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OUT_DIR/executor.o" >/dev/null
for t in 7 8 9; do
  clang -arch x86_64 -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function \
    -DT="$t" -DH=2 -DD=3 -DM=6 -DQW=18 -DMAXCACHE=9 -I"$ROOT/fasm/spikes" \
    "$ROOT/fasm/spikes/tensor_causal_attention_cached_differential_check.c" "$OUT_DIR/executor.o" -lm -o "$OUT_DIR/check_T$t"
  first="$(arch -x86_64 "$OUT_DIR/check_T$t")"
  [[ "$(arch -x86_64 "$OUT_DIR/check_T$t")" == "$first" ]]
  printf '%s\n' "$first"
done
