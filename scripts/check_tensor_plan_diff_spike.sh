#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/plan-diff.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -Wall -Wextra -Werror -Wno-misleading-indentation -O1 -g -fsanitize=address,undefined -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_plan_diff_check.c" -o "$OUT_DIR/check"
"$OUT_DIR/check"
