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

echo "core smoke checks passed"
