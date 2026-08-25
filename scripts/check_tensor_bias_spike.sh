#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-bias-spike.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_tape_f32.asm" "$OUT_DIR/tape.o" >/dev/null
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_ops_f32.asm" "$OUT_DIR/ops.o" >/dev/null
clang -arch x86_64 -O2 "$ROOT/fasm/spikes/tensor_bias_check.c" "$OUT_DIR/tape.o" "$OUT_DIR/ops.o" -o "$OUT_DIR/check"
arch -x86_64 "$OUT_DIR/check"
