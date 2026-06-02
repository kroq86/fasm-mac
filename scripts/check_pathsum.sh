#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pathsum-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/pathsum"
TREE="$OUT_DIR/tree"

fasm "$ROOT/fasm/apps/pathsum.asm" "$BIN" >/dev/null

mkdir -p "$TREE/sub" "$TREE/empty"
printf 'ab' > "$TREE/a.txt"
printf 'cde' > "$TREE/sub/b.txt"

expected='files 2
dirs 2
bytes 5'
actual="$(arch -x86_64 "$BIN" "$TREE")"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL pathsum tree\nexpected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
  exit 1
fi

if arch -x86_64 "$BIN" "$TREE/a.txt" >/dev/null 2>"$OUT_DIR/file.err"; then
  echo 'FAIL pathsum file input should exit 2' >&2
  exit 1
fi
grep -q 'pathsum: not a directory:' "$OUT_DIR/file.err"

if arch -x86_64 "$BIN" "$OUT_DIR/missing" >/dev/null 2>"$OUT_DIR/missing.err"; then
  echo 'FAIL pathsum missing input should exit 2' >&2
  exit 1
fi
grep -q 'pathsum: not a directory:' "$OUT_DIR/missing.err"

echo 'pathsum checks passed'
