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
clang++ -std=c++20 -O2 -arch x86_64 \
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

echo 'PASS ragbox check'
