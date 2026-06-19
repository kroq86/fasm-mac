#!/usr/bin/env bash
# Full ragbox E2E: offline checks + brew/source binary + live build/refresh/search.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ragbox-full.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

PYTHON="${PYTHON:-python3}"
FIXTURE_DIR="$ROOT/fasm/tests/ragbox/fixtures"
TINY_REPO="$FIXTURE_DIR/tiny-repo"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL="${RAGBOX_MODEL:-nomic-embed-text}"

RAGBOX="${RAGBOX_BIN:-}"
if [[ -z "$RAGBOX" ]]; then
    if command -v ragbox >/dev/null 2>&1; then
        RAGBOX="$(command -v ragbox)"
    else
        CORE_OBJ="$OUT_DIR/logvec_core.o"
        RAGBOX="$OUT_DIR/ragbox"
        "$ROOT/bin/fasm" --emit=macho-obj "$ROOT/fasm/apps/logvec_core.asm" "$CORE_OBJ" >/dev/null
        clang++ -std=c++20 -O2 -arch x86_64 -pthread \
            "$ROOT/fasm/apps/ragbox/ragbox.cpp" \
            "$CORE_OBJ" \
            -o "$RAGBOX"
    fi
fi

run_ragbox() {
    arch -x86_64 "$RAGBOX" "$@"
}

step() {
    echo "==> $*"
}

fail() {
    echo "FAIL $*" >&2
    exit 1
}

step "offline: check_ragbox.sh"
"$ROOT/scripts/check_ragbox.sh"

step "offline: check_ragbox_release.sh"
"$ROOT/scripts/check_ragbox_release.sh"

step "binary: $RAGBOX ($(file -b "$RAGBOX" | head -1))"

step "doctor --skip-ollama"
run_ragbox doctor --skip-ollama >/dev/null

step "doctor with Ollama"
run_ragbox doctor --ollama "$OLLAMA_URL" --model "$MODEL" >/dev/null

step "build --dry-run"
run_ragbox build \
    --root "$TINY_REPO" \
    --out "$OUT_DIR/dry.lv" \
    --manifest "$OUT_DIR/dry.manifest.json" \
    --dry-run >/dev/null

step "build full (tiny-repo)"
WORK_REPO="$OUT_DIR/work-repo"
cp -R "$TINY_REPO" "$WORK_REPO"
INDEX="$OUT_DIR/memory.lv"
MANIFEST="$OUT_DIR/memory.lv.manifest.json"
run_ragbox build \
    --root "$WORK_REPO" \
    --out "$INDEX" \
    --manifest "$MANIFEST" \
    --ollama "$OLLAMA_URL" \
    --model "$MODEL" \
    --chunk-size 800 \
    --overlap 100

[[ -f "$INDEX" ]] || fail "missing index after build"
[[ -f "$MANIFEST" ]] || fail "missing manifest after build"
[[ -f "$OUT_DIR/memory.lv.state.json" ]] || fail "missing state after build"
[[ ! -f "$OUT_DIR/memory.lv.delta" ]] || fail "delta should not exist after full build"

step "doctor index+manifest+state"
run_ragbox doctor --skip-ollama --index "$INDEX" --manifest "$MANIFEST" >/dev/null

step "search --json (auth)"
OUT="$(run_ragbox search \
    --index "$INDEX" \
    --manifest "$MANIFEST" \
    --query "JWT authentication middleware" \
    --ollama "$OLLAMA_URL" \
    --model "$MODEL" \
    --top 3 \
    --json)"
"$PYTHON" -c '
import json, sys
hits = json.loads(sys.argv[1])
assert hits and hits[0]["path"] == "docs/auth.md", hits
assert hits[0]["score"] > 0.3, hits[0]["score"]
print("  top:", hits[0]["path"], hits[0]["score"])
' "$OUT"

step "search plain text (db)"
PLAIN="$(run_ragbox search \
    --index "$INDEX" \
    --manifest "$MANIFEST" \
    --query "postgres database migrations" \
    --ollama "$OLLAMA_URL" \
    --model "$MODEL" \
    --top 1)"
echo "$PLAIN" | grep -q 'docs/db.md' || fail "plain search db: $PLAIN"

step "search --query-file offline fixture"
run_ragbox search \
    --index "$FIXTURE_DIR/fixture.lv" \
    --manifest "$FIXTURE_DIR/fixture.manifest.json" \
    --query-file "$FIXTURE_DIR/query_auth.bin" \
    --top 1 \
    --json >/dev/null

step "incremental search (fixtures)"
run_ragbox search \
    --index "$FIXTURE_DIR/incremental/base.lv" \
    --manifest "$FIXTURE_DIR/incremental/manifest.json" \
    --query-file "$FIXTURE_DIR/incremental/query_auth.bin" \
    --top 1 \
    --json >/dev/null

step "bench"
run_ragbox bench \
    --index "$FIXTURE_DIR/fixture.lv" \
    --manifest "$FIXTURE_DIR/fixture.manifest.lite.json" \
    --query-file "$FIXTURE_DIR/query_auth.bin" \
    --iters 3 \
    --threads 2 >/dev/null

step "refresh --dry-run (up-to-date)"
OUT="$(run_ragbox refresh \
    --root "$WORK_REPO" \
    --index "$INDEX" \
    --manifest "$MANIFEST" \
    --model "$MODEL" \
    --dry-run)"
echo "$OUT" | grep -q 'up-to-date\|dry-run' || fail "refresh dry-run: $OUT"

step "modify auth.md + refresh"
echo "" >> "$WORK_REPO/docs/auth.md"
echo "OAuth2 PKCE flow for SPA clients." >> "$WORK_REPO/docs/auth.md"
run_ragbox refresh \
    --root "$WORK_REPO" \
    --index "$INDEX" \
    --manifest "$MANIFEST" \
    --ollama "$OLLAMA_URL" \
    --model "$MODEL"

[[ -f "$OUT_DIR/memory.lv.delta" ]] || fail "delta missing after refresh"

step "doctor after refresh (base+delta+state)"
run_ragbox doctor --skip-ollama --index "$INDEX" --manifest "$MANIFEST" >/dev/null

step "search after refresh"
OUT2="$(run_ragbox search \
    --index "$INDEX" \
    --manifest "$MANIFEST" \
    --query "OAuth2 PKCE SPA" \
    --ollama "$OLLAMA_URL" \
    --model "$MODEL" \
    --top 3 \
    --json)"
"$PYTHON" -c '
import json, sys
hits = json.loads(sys.argv[1])
assert hits, "no hits after refresh"
paths = [h["path"] for h in hits]
assert "docs/auth.md" in paths, paths
print("  top after refresh:", hits[0]["path"], hits[0]["score"])
' "$OUT2"

step "delete file + refresh"
rm -f "$WORK_REPO/src/middleware.go"
run_ragbox refresh \
    --root "$WORK_REPO" \
    --index "$INDEX" \
    --manifest "$MANIFEST" \
    --ollama "$OLLAMA_URL" \
    --model "$MODEL" >/dev/null

step "search should not return deleted middleware.go in top-3"
OUT3="$(run_ragbox search \
    --index "$INDEX" \
    --manifest "$MANIFEST" \
    --query "AuthMiddleware HandlerFunc" \
    --ollama "$OLLAMA_URL" \
    --model "$MODEL" \
    --top 3 \
    --json)"
"$PYTHON" -c '
import json, sys
hits = json.loads(sys.argv[1])
paths = [h["path"] for h in hits]
assert "src/middleware.go" not in paths, "deleted file still in results: %r" % hits
print("  ok: middleware.go filtered, hits=", len(hits))
' "$OUT3"

step "build --embed-text (rebuild clean)"
INDEX2="$OUT_DIR/memory2.lv"
run_ragbox build \
    --root "$WORK_REPO" \
    --out "$INDEX2" \
    --manifest "$OUT_DIR/memory2.lv.manifest.json" \
    --ollama "$OLLAMA_URL" \
    --model "$MODEL" \
    --embed-text >/dev/null
"$PYTHON" -c '
import json, sys
m = json.load(open(sys.argv[1]))
assert m["records"][0].get("text"), "embed-text missing"
' "$OUT_DIR/memory2.lv.manifest.json"

step "refresh param mismatch should fail"
if run_ragbox refresh \
    --root "$WORK_REPO" \
    --index "$INDEX2" \
    --manifest "$OUT_DIR/memory2.lv.manifest.json" \
    --model "wrong-model" \
    --dry-run 2>/dev/null; then
    fail "expected ModelMismatch"
fi
echo "  ok: ModelMismatch on wrong model"

echo "PASS ragbox full check (binary=$RAGBOX)"
