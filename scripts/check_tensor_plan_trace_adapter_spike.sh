#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/trace-adapter.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -Wall -Wextra -Werror -Wno-unused-function -Wno-missing-field-initializers -Wno-misleading-indentation -O2 -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_plan_trace_adapter_check.c" -o "$OUT_DIR/check"
"$OUT_DIR/check"
