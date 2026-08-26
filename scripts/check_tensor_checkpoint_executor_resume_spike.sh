#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-executor-resume.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OUT_DIR/executor.o" >/dev/null
clang -arch x86_64 -std=c11 -Wall -Wextra -Werror -Wno-unused-function -O2 -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_checkpoint_executor_resume_check.c" "$OUT_DIR/executor.o" -o "$OUT_DIR/check"
full="$(arch -x86_64 "$OUT_DIR/check" full "$OUT_DIR/full.ckpt")"
arch -x86_64 "$OUT_DIR/check" split "$OUT_DIR/half.ckpt"
resumed="$(arch -x86_64 "$OUT_DIR/check" resume "$OUT_DIR/half.ckpt" "$OUT_DIR/resumed.ckpt")"
cmp "$OUT_DIR/full.ckpt" "$OUT_DIR/resumed.ckpt"
[[ "$full" == "$resumed" ]]
printf 'tensor checkpoint executor resume passed: processes=3 split=2000+2000 continuous=4000 checkpoint=bit-exact result="%s"\n' "$full"
