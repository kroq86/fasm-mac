#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${RAGBOX_RELEASE_VERSION:-0.3.0}"
PKG="ragbox-${VERSION}-macos-x86_64"
TARBALL="$ROOT/dist/${PKG}.tar.gz"
FIXTURE_DIR="$ROOT/fasm/tests/ragbox/fixtures"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ragbox-release-check.XXXXXX")"
EXTRACT="$OUT_DIR/extract"
PYTHON="${PYTHON:-python3}"

cleanup() {
  rm -rf "$OUT_DIR"
}
trap cleanup EXIT

run_ragbox() {
  arch -x86_64 "$@"
}

"$ROOT/scripts/build-ragbox-release.sh" "$VERSION" >/dev/null

if [[ ! -f "$TARBALL" ]]; then
  echo "FAIL missing tarball: $TARBALL" >&2
  exit 1
fi

mkdir -p "$EXTRACT"
tar -xzf "$TARBALL" -C "$EXTRACT"
RAGBOX="$EXTRACT/$PKG/ragbox"

if [[ ! -x "$RAGBOX" ]]; then
  echo "FAIL missing ragbox binary in tarball" >&2
  exit 1
fi

if ! file "$RAGBOX" | grep -q 'x86_64'; then
  echo 'FAIL ragbox is not x86_64' >&2
  file "$RAGBOX" >&2
  exit 1
fi

run_ragbox "$RAGBOX" doctor --skip-ollama >/dev/null

DRY_MANIFEST="$OUT_DIR/dry.manifest.json"
run_ragbox "$RAGBOX" build \
  --root "$FIXTURE_DIR/tiny-repo" \
  --out "$OUT_DIR/dry.lv" \
  --manifest "$DRY_MANIFEST" \
  --dry-run >/dev/null

if ! "$PYTHON" - <<'PY' "$DRY_MANIFEST"
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["records"], "dry-run manifest must contain chunks"
assert any(r["path"] == "docs/auth.md" for r in manifest["records"])
PY
then
  echo 'FAIL dry-run build from release binary' >&2
  exit 1
fi

"$PYTHON" "$ROOT/fasm/tests/ragbox/write_fixture.py" "$FIXTURE_DIR" >/dev/null
FIX="$FIXTURE_DIR"

for query in auth db middleware; do
  OUT="$(run_ragbox "$RAGBOX" search \
    --index "$FIX/fixture.lv" \
    --manifest "$FIX/fixture.manifest.json" \
    --query-file "$FIX/query_${query}.bin" \
    --top 1 \
    --json)"
  if ! "$PYTHON" -c '
import json
import sys

expected = json.load(open(sys.argv[1], encoding="utf-8"))
actual = json.loads(sys.argv[2])
assert actual == expected, f"expected={expected!r} actual={actual!r}"
' "$FIX/expected_${query}.json" "$OUT"
  then
    echo "FAIL search query: $query" >&2
    printf '%s\n' "$OUT" >&2
    exit 1
  fi
done

INC_DIR="$FIX/incremental"
REFRESH_DIR="$OUT_DIR/refresh-smoke"
mkdir -p "$REFRESH_DIR"
cp "$INC_DIR/base.lv" "$REFRESH_DIR/memory.lv"
cp "$INC_DIR/manifest.json" "$REFRESH_DIR/memory.lv.manifest.json"
cp "$INC_DIR/refresh_state.json" "$REFRESH_DIR/memory.lv.state.json"
run_ragbox "$RAGBOX" refresh \
  --root "$FIXTURE_DIR/tiny-repo" \
  --index "$REFRESH_DIR/memory.lv" \
  --manifest "$REFRESH_DIR/memory.lv.manifest.json" \
  --model "fixture-dim4" \
  --dry-run >/dev/null

echo 'PASS ragbox release check'
