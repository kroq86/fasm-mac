#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fmath-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/fmath"
fasm "$ROOT/fasm/apps/fmath.asm" "$BIN" >/dev/null

assert_eq() {
  local name="$1"
  local expected="$2"
  shift 2
  local actual
  actual="$(arch -x86_64 "$BIN" "$@")"
  if [[ "$actual" != "$expected" ]]; then
    printf 'FAIL %s\nexpected: %q\nactual:   %q\n' "$name" "$expected" "$actual" >&2
    exit 1
  fi
}

assert_eq 'frac normalize/add' '1/2' frac add 1/3 1/6
assert_eq 'frac signed parse' '6/1' frac add +10/2 1
assert_eq 'frac sub' '-5/4' frac sub -3/4 1/2
assert_eq 'frac mul' '-1/4' frac mul -6/8 1/3
assert_eq 'frac div' '3/2' frac div 1/2 1/3

assert_eq 'poly derive' '4x+3' poly-derive 1 3 2
assert_eq 'poly integrate' '2/3x^3+3/2x^2+x' poly-integrate 1 3 2
assert_eq 'poly eval' '15/1' poly-eval 2 1 3 2
assert_eq 'poly rational coefficients' '4/3x+3/2' poly-derive 1/2 3/2 2/3

if arch -x86_64 "$BIN" frac div 1 0 >/dev/null 2>"$OUT_DIR/err"; then
  echo 'FAIL fmath division by zero should exit 2' >&2
  exit 1
fi
grep -q 'fmath: parse/math error' "$OUT_DIR/err"

if arch -x86_64 "$BIN" wat >/dev/null 2>"$OUT_DIR/usage"; then
  echo 'FAIL fmath bad command should exit 2' >&2
  exit 1
fi
grep -q 'usage: fmath' "$OUT_DIR/usage"

too_many_args=(poly-derive)
for _ in {1..33}; do
  too_many_args+=(1)
done
if arch -x86_64 "$BIN" "${too_many_args[@]}" >/dev/null 2>"$OUT_DIR/too_many"; then
  echo 'FAIL fmath too many coefficients should exit 2' >&2
  exit 1
fi
grep -q 'too many polynomial coefficients' "$OUT_DIR/too_many"

echo 'fmath checks passed'
