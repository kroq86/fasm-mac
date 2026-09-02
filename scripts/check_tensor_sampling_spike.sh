#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-sampling.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -arch x86_64 -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function \
  -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/tensor_sampling_differential_check.c" -lm -o "$OUT_DIR/check"
first="$(arch -x86_64 "$OUT_DIR/check")"
[[ "$(arch -x86_64 "$OUT_DIR/check")" == "$first" ]]
printf '%s\n' "$first"
