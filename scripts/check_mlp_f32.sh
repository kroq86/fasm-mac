#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlp-f32-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/mlp_forward"
fasm "$ROOT/fasm/examples/mlp_forward.asm" "$BIN" >/dev/null

actual="$(arch -x86_64 "$BIN")"
expected='1750 1250'
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL mlp_forward\nexpected: %q\nactual:   %q\n' "$expected" "$actual" >&2
  exit 1
fi

echo 'mlp f32 checks passed'
