#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/logvec-search-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

PYTHON="${PYTHON:-python3}"
FIXTURE_DIR="$ROOT/fasm/tests/logvec/fixtures"

if ! command -v zig >/dev/null 2>&1; then
    echo 'FAIL zig is required for logvec check' >&2
    exit 1
fi

CORE_OBJ="$OUT_DIR/logvec_core.o"
BIN="$OUT_DIR/logvec"

fasm --emit=macho-obj "$ROOT/fasm/apps/logvec_core.asm" "$CORE_OBJ" >/dev/null
zig build-exe \
    "$ROOT/fasm/apps/logvec.zig" \
    "$CORE_OBJ" \
    -target x86_64-macos \
    -mcpu=baseline \
    -O ReleaseSafe \
    -femit-bin="$BIN"

"$PYTHON" "$ROOT/fasm/tests/logvec/write_search_fixture.py" "$FIXTURE_DIR"

OUT="$(arch -x86_64 "$BIN" search \
    --index "$FIXTURE_DIR/search_smoke.lv" \
    --query "$FIXTURE_DIR/search_query.bin" \
    --top 2)"
printf '%s\n' "$OUT"

diff -u "$FIXTURE_DIR/expected_search.txt" <(printf '%s\n' "$OUT")

echo 'logvec search check passed'
