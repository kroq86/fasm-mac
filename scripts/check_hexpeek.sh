#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hexpeek-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/hexpeek"
FILE="$OUT_DIR/sample.bin"

fasm "$ROOT/fasm/apps/hexpeek.asm" "$BIN" >/dev/null

printf 'ABC\nxyz' > "$FILE"

expected='00000000: 41 42 43 0a 78 79 7a                              ABC.xyz'
actual="$(arch -x86_64 "$BIN" -n 7 "$FILE")"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL hexpeek default\nexpected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
  exit 1
fi

expected='00000002: 43 0a 78                                          C.x'
actual="$(arch -x86_64 "$BIN" -s 2 -n 3 "$FILE")"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL hexpeek skip/limit\nexpected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
  exit 1
fi

if arch -x86_64 "$BIN" -n nope "$FILE" >/dev/null 2>"$OUT_DIR/usage"; then
  echo 'FAIL hexpeek bad numeric option should exit 2' >&2
  exit 1
fi
grep -q 'usage: hexpeek' "$OUT_DIR/usage"

if arch -x86_64 "$BIN" "$OUT_DIR/missing.bin" >/dev/null 2>"$OUT_DIR/err"; then
  echo 'FAIL hexpeek missing file should exit 2' >&2
  exit 1
fi
grep -q 'hexpeek: cannot open:' "$OUT_DIR/err"

echo 'hexpeek checks passed'
