#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/plan-trace.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -Wall -Wextra -Werror -Wno-misleading-indentation -O2 -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_plan_trace_check.c" -o "$OUT_DIR/check"
"$OUT_DIR/check" "$OUT_DIR/trace.tsv"
[[ "$(wc -l < "$OUT_DIR/trace.tsv" | tr -d ' ')" == 7 ]]
