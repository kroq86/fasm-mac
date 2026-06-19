#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ragbox-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

PYTHON="${PYTHON:-python3}"
FIXTURE_DIR="$ROOT/fasm/tests/ragbox/fixtures"
TINY_REPO="$FIXTURE_DIR/tiny-repo"

if ! command -v clang++ >/dev/null 2>&1; then
    echo 'FAIL clang++ is required for ragbox check' >&2
    exit 1
fi

CORE_OBJ="$OUT_DIR/logvec_core.o"
RAGBOX="$OUT_DIR/ragbox"

"$ROOT/bin/fasm" --emit=macho-obj "$ROOT/fasm/apps/logvec_core.asm" "$CORE_OBJ" >/dev/null
clang++ -std=c++20 -O2 -arch x86_64 -pthread \
    "$ROOT/fasm/apps/ragbox/ragbox.cpp" \
    "$CORE_OBJ" \
    -o "$RAGBOX"

"$PYTHON" "$ROOT/fasm/tests/ragbox/write_fixture.py" "$FIXTURE_DIR"

for query in auth db middleware; do
    OUT="$(arch -x86_64 "$RAGBOX" search \
        --index "$FIXTURE_DIR/fixture.lv" \
        --manifest "$FIXTURE_DIR/fixture.manifest.json" \
        --query-file "$FIXTURE_DIR/query_${query}.bin" \
        --top 1 \
        --json)"
    if ! "$PYTHON" -c '
import json
import sys

expected = json.load(open(sys.argv[1], encoding="utf-8"))
actual = json.loads(sys.argv[2])
assert actual == expected, f"expected={expected!r} actual={actual!r}"
' "$FIXTURE_DIR/expected_${query}.json" "$OUT"
    then
        echo "FAIL search query: $query" >&2
        printf '%s\n' "$OUT" >&2
        exit 1
    fi
done

arch -x86_64 "$RAGBOX" doctor \
    --skip-ollama \
    --index "$FIXTURE_DIR/fixture.lv" \
    --manifest "$FIXTURE_DIR/fixture.manifest.json"

DRY_MANIFEST="$OUT_DIR/dry.manifest.json"
arch -x86_64 "$RAGBOX" build \
    --root "$TINY_REPO" \
    --out "$OUT_DIR/dry.lv" \
    --manifest "$DRY_MANIFEST" \
    --dry-run >/dev/null

if ! "$PYTHON" - <<'PY' "$DRY_MANIFEST"
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["records"], "dry-run manifest must contain chunks"
assert any(r["path"] == "docs/auth.md" for r in manifest["records"])
print("dry-run chunks:", len(manifest["records"]))
PY
then
    echo 'FAIL dry-run build did not produce expected chunks' >&2
    exit 1
fi

arch -x86_64 "$RAGBOX" bench \
    --index "$FIXTURE_DIR/fixture.lv" \
    --manifest "$FIXTURE_DIR/fixture.manifest.lite.json" \
    --query-file "$FIXTURE_DIR/query_auth.bin" \
    --iters 5 >/dev/null

INC_DIR="$FIXTURE_DIR/incremental"
OUT="$(arch -x86_64 "$RAGBOX" search \
    --index "$INC_DIR/base.lv" \
    --manifest "$INC_DIR/manifest.json" \
    --query-file "$INC_DIR/query_auth.bin" \
    --top 1 \
    --json)"
if ! "$PYTHON" -c '
import json
import sys

expected = json.load(open(sys.argv[1], encoding="utf-8"))
actual = json.loads(sys.argv[2])
assert actual == expected, f"expected={expected!r} actual={actual!r}"
' "$INC_DIR/expected_auth.json" "$OUT"
then
    echo "FAIL incremental search" >&2
    printf '%s\n' "$OUT" >&2
    exit 1
fi

arch -x86_64 "$RAGBOX" doctor \
    --skip-ollama \
    --index "$INC_DIR/base.lv" \
    --manifest "$INC_DIR/manifest.json"

REFRESH_DIR="$OUT_DIR/refresh-smoke"
mkdir -p "$REFRESH_DIR"
cp "$INC_DIR/base.lv" "$REFRESH_DIR/memory.lv"
cp "$INC_DIR/manifest.json" "$REFRESH_DIR/memory.lv.manifest.json"
cp "$INC_DIR/refresh_state.json" "$REFRESH_DIR/memory.lv.state.json"
REFRESH_OUT="$(arch -x86_64 "$RAGBOX" refresh \
    --root "$TINY_REPO" \
    --index "$REFRESH_DIR/memory.lv" \
    --manifest "$REFRESH_DIR/memory.lv.manifest.json" \
    --model "fixture-dim4" \
    --dry-run 2>&1)"
case "$REFRESH_OUT" in
    *"dry-run"*) ;;
    *)
        echo "FAIL refresh dry-run output: $REFRESH_OUT" >&2
        exit 1
        ;;
esac

echo 'PASS ragbox check'
