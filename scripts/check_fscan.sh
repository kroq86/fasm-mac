#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fscan-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/fscan"
FILE_A="$OUT_DIR/a.txt"
FILE_B="$OUT_DIR/b.txt"

fasm "$ROOT/fasm/apps/fscan.asm" "$BIN" >/dev/null

printf 'alpha\nneedle one\nlast\n' > "$FILE_A"
printf 'needle two\nplain\n' > "$FILE_B"

expected="$FILE_A:2:needle one
$FILE_B:1:needle two"
actual="$(arch -x86_64 "$BIN" needle "$FILE_A" "$FILE_B")"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL fscan matches\nexpected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
  exit 1
fi

actual="$(arch -x86_64 "$BIN" -c needle "$FILE_A" "$FILE_B")"
expected="$FILE_A:1
$FILE_B:1"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL fscan count\nexpected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
  exit 1
fi

actual="$(arch -x86_64 "$BIN" -l needle "$FILE_A" "$FILE_B")"
expected="$FILE_A
$FILE_B"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL fscan files-only\nexpected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
  exit 1
fi

actual="$(arch -x86_64 "$BIN" -i NEEDLE "$FILE_A" "$FILE_B")"
expected="$FILE_A:2:needle one
$FILE_B:1:needle two"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL fscan ignore-case\nexpected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
  exit 1
fi

if arch -x86_64 "$BIN" absent "$FILE_A" >/dev/null; then
  echo 'FAIL fscan absent should exit 1' >&2
  exit 1
fi

if arch -x86_64 "$BIN" needle "$OUT_DIR/missing.txt" >/dev/null 2>"$OUT_DIR/err"; then
  echo 'FAIL fscan missing file should exit 2' >&2
  exit 1
fi
if ! grep -q 'fscan: cannot open:' "$OUT_DIR/err"; then
  echo 'FAIL fscan missing file stderr' >&2
  exit 1
fi

if arch -x86_64 "$BIN" >/dev/null 2>"$OUT_DIR/usage"; then
  echo 'FAIL fscan usage should exit 2' >&2
  exit 1
fi
if ! grep -q 'usage: fscan' "$OUT_DIR/usage"; then
  echo 'FAIL fscan usage stderr' >&2
  exit 1
fi

echo 'fscan checks passed'
