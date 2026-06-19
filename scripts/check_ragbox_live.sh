#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ragbox-live.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

PYTHON="${PYTHON:-python3}"
TINY_REPO="$ROOT/fasm/tests/ragbox/fixtures/tiny-repo"
INDEX="$OUT_DIR/memory.lv"
MANIFEST="$OUT_DIR/memory.lv.manifest.json"
RAGBOX="$OUT_DIR/ragbox"
CORE_OBJ="$OUT_DIR/logvec_core.o"

OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL="${RAGBOX_MODEL:-nomic-embed-text}"

if ! command -v clang++ >/dev/null 2>&1; then
    echo 'FAIL clang++ is required' >&2
    exit 1
fi

"$ROOT/bin/fasm" --emit=macho-obj "$ROOT/fasm/apps/logvec_core.asm" "$CORE_OBJ" >/dev/null
clang++ -std=c++20 -O2 -arch x86_64 \
    "$ROOT/fasm/apps/ragbox/ragbox.cpp" \
    "$CORE_OBJ" \
    -o "$RAGBOX"

arch -x86_64 "$RAGBOX" doctor --ollama "$OLLAMA_URL" --model "$MODEL"

arch -x86_64 "$RAGBOX" build \
    --root "$TINY_REPO" \
    --out "$INDEX" \
    --manifest "$MANIFEST" \
    --ollama "$OLLAMA_URL" \
    --model "$MODEL" \
    --chunk-size 800 \
    --overlap 100

arch -x86_64 "$RAGBOX" doctor \
    --skip-ollama \
    --index "$INDEX" \
    --manifest "$MANIFEST"

check_query() {
    local query="$1"
    local expect_path="$2"
    local out
    out="$(arch -x86_64 "$RAGBOX" search \
        --index "$INDEX" \
        --manifest "$MANIFEST" \
        --query "$query" \
        --ollama "$OLLAMA_URL" \
        --model "$MODEL" \
        --top 3 \
        --json)"
    if ! "$PYTHON" -c '
import json
import sys

expect_path = sys.argv[1]
hits = json.loads(sys.argv[2])
assert hits, "expected at least one hit"
top = hits[0]
assert top["path"] == expect_path, "top hit path %r != %r" % (top["path"], expect_path)
assert top["score"] > 0.5, "score too low: %s" % top["score"]
print("top hit:", top["path"], top["score"])
' "$expect_path" "$out"
    then
        echo "FAIL live query: $query" >&2
        printf '%s\n' "$out" >&2
        exit 1
    fi
}

check_query "JWT authentication middleware" "docs/auth.md"
check_query "postgres database migrations" "docs/db.md"
check_query "AuthMiddleware HandlerFunc net/http" "src/middleware.go"

echo 'PASS ragbox live check'
