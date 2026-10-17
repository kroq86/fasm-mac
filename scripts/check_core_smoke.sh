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

hashmap_bin="$OUT_DIR/hashmap_str"
fasm "$ROOT/fasm/tests/macos-smoke/hashmap_str.asm" "$hashmap_bin" >/dev/null
actual="$(arch -x86_64 "$hashmap_bin")"
if [[ "$actual" != "42 hello 1" ]]; then
  printf 'FAIL hashmap_str\nexpected: %q\nactual:   %q\n' "42 hello 1" "$actual" >&2
  exit 1
fi

hashmap_update_bin="$OUT_DIR/hashmap_str_update"
fasm "$ROOT/fasm/tests/macos-smoke/hashmap_str_update.asm" "$hashmap_update_bin" >/dev/null
actual="$(arch -x86_64 "$hashmap_update_bin")"
if [[ "$actual" != "hashmap_str update checks passed" ]]; then
  printf 'FAIL hashmap_str_update\nactual: %q\n' "$actual" >&2
  exit 1
fi

miniredis_script_bin="$OUT_DIR/miniredis_script"
fasm "$ROOT/fasm/tests/macos-smoke/miniredis_script.asm" "$miniredis_script_bin" >/dev/null
actual="$(arch -x86_64 "$miniredis_script_bin")"
expected=$'42\nhello\n2'
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL miniredis_script\nexpected: %q\nactual: %q\n' "$expected" "$actual" >&2
  exit 1
fi

miniredis_bin="$OUT_DIR/miniredis"
fasm "$ROOT/fasm/apps/miniredis.asm" "$miniredis_bin" >/dev/null
actual="$(arch -x86_64 "$miniredis_bin" <<'INPUT'
SET foo 42
SET foo 84
GET foo
SET bff chain
GET foo
SET foo hello
GET foo
SET foo 9
GET foo
INCR foo
DECR foo
GET bff
DBSIZE
QUIT
INPUT
)"
expected=$'miniredis> OK\nminiredis> OK\nminiredis> 84\nminiredis> OK\nminiredis> 84\nminiredis> OK\nminiredis> hello\nminiredis> OK\nminiredis> 9\nminiredis> 10\nminiredis> 9\nminiredis> chain\nminiredis> 2\nminiredis> '
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL miniredis replacement\nexpected: %q\nactual: %q\n' "$expected" "$actual" >&2
  exit 1
fi

echo "core smoke checks passed"
