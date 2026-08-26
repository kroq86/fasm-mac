#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/fasm/build/out/tensorctl}"
OBJ_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-build.XXXXXX")"
trap 'rm -rf "$OBJ_DIR"' EXIT

mkdir -p "$(dirname "$OUT")"

fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OBJ_DIR/executor.o" >/dev/null
# Embedded in the layout-decision profile cache key: a cache built by a
# differently-optimized binary (e.g. -O0 debug build) must not be trusted by
# this one, since -O3 vs -fno-vectorize has already been shown to flip
# which variant even wins.
clang -arch x86_64 -O2 -I"$ROOT/fasm/spikes" -DTENSORCTL_BUILD_CONFIG=\"arch=x86_64,opt=O2\" \
  "$ROOT/fasm/examples/tensorctl_main.c" \
  "$ROOT/fasm/examples/tensorctl_mlp.c" \
  "$ROOT/fasm/examples/tensorctl_transformer.c" \
  "$ROOT/fasm/examples/tensorctl_plan_diff.c" \
  "$OBJ_DIR/executor.o" \
  -o "$OUT"
chmod +x "$OUT"

printf 'Built %s\n' "$OUT"
