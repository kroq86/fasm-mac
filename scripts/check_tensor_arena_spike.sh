#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-arena-spike.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_arena_f32.asm" "$OUT_DIR/arena.o" >/dev/null
clang -arch x86_64 -O2 "$ROOT/fasm/spikes/tensor_arena_check.c" "$OUT_DIR/arena.o" -o "$OUT_DIR/check"
arch -x86_64 "$OUT_DIR/check"
