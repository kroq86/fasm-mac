#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/core-smoke.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

repl_bin="$OUT_DIR/repl_ping"
fasm "$ROOT/fasm/tests/macos-smoke/repl_ping.asm" "$repl_bin" >/dev/null
actual="$(printf 'PING\n' | arch -x86_64 "$repl_bin")"
if [[ "$actual" != "PONG" ]]; then
  printf 'FAIL repl_ping\nexpected: %q\nactual:   %q\n' "PONG" "$actual" >&2
  exit 1
fi

str_bin="$OUT_DIR/str_hash"
fasm "$ROOT/fasm/tests/macos-smoke/str_hash.asm" "$str_bin" >/dev/null
actual="$(arch -x86_64 "$str_bin")"
expected="58612198064902318"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL str_hash\nexpected: %q\nactual:   %q\n' "$expected" "$actual" >&2
  exit 1
fi

rational_bin="$OUT_DIR/rational"
fasm "$ROOT/fasm/tests/macos-smoke/rational.asm" "$rational_bin" >/dev/null
actual="$(arch -x86_64 "$rational_bin")"
expected="$(cat <<'EXPECTED'
-6/8 -> -3/4
1/3 + 1/6 = 1/2
integral 0..1 x dx = 1/2
integral 0..1 (2x^2 + 3x + 1) dx = 19/6
EXPECTED
)"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL rational\nexpected: %q\nactual:   %q\n' "$expected" "$actual" >&2
  exit 1
fi

polynomial_bin="$OUT_DIR/polynomial"
fasm "$ROOT/fasm/tests/macos-smoke/polynomial.asm" "$polynomial_bin" >/dev/null
actual="$(arch -x86_64 "$polynomial_bin")"
expected=$'p = 2x^2+3x+1\np\' = 4x+3\nintegral p dx = 2/3x^3+3/2x^2+x\np(2) = 15/1\nintegral 0..1 p dx = 19/6\np + (x+1) = 2x^2+4x+2\np * (x+1) = 2x^3+5x^2+4x+1'
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL polynomial\nexpected: %q\nactual:   %q\n' "$expected" "$actual" >&2
  exit 1
fi

taylor_bin="$OUT_DIR/taylor"
fasm "$ROOT/fasm/tests/macos-smoke/taylor.asm" "$taylor_bin" >/dev/null
actual="$(arch -x86_64 "$taylor_bin")"
expected="$(cat <<'EXPECTED'
demidovich-taylor p = x^3-2x+1
taylor at 1 in h = x^3+3x^2+x
EXPECTED
)"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL taylor\nexpected: %q\nactual:   %q\n' "$expected" "$actual" >&2
  exit 1
fi

echo "core smoke checks passed"
