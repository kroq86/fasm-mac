#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/autograd-f32-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/autograd_scalar"
fasm "$ROOT/fasm/examples/autograd_scalar.asm" "$BIN" >/dev/null

actual="$(arch -x86_64 "$BIN")"
expected='4000 -3000 2000 1000'
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL autograd_scalar\nexpected: %q\nactual:   %q\n' "$expected" "$actual" >&2
  exit 1
fi

echo 'autograd f32 checks passed'
