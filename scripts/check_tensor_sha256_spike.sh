#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-sha256.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function \
  -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/tensor_sha256_differential_check.c" -o "$OUT_DIR/check"
first="$("$OUT_DIR/check")"
[[ "$("$OUT_DIR/check")" == "$first" ]]
printf '%s\n' "$first"
