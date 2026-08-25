#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-sgd-spike.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_tape_f32.asm" "$OUT_DIR/tape.o" >/dev/null
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_sgd_f32.asm" "$OUT_DIR/sgd.o" >/dev/null
clang -arch x86_64 -O2 "$ROOT/fasm/spikes/tensor_sgd_check.c" "$OUT_DIR/tape.o" "$OUT_DIR/sgd.o" -o "$OUT_DIR/check"
arch -x86_64 "$OUT_DIR/check"
