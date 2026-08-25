#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?usage: $0 <version> (e.g. 0.3.0)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$(mktemp -d)"
PKG="ragbox-${VERSION}-macos-x86_64"
OUT_DIR="$ROOT/dist"
OUT="$OUT_DIR/${PKG}.tar.gz"
FASM="${FASM:-$ROOT/bin/fasm}"
CORE_OBJ="$STAGING/logvec_core.o"

cleanup() {
  rm -rf "$STAGING"
}
trap cleanup EXIT

if ! command -v clang++ >/dev/null 2>&1; then
  echo 'FAIL clang++ is required to build ragbox release' >&2
  exit 1
fi
if [[ ! -x "$FASM" ]]; then
  echo "FAIL fasm not found at $FASM" >&2
  exit 1
fi

DEST="$STAGING/$PKG"
mkdir -p "$DEST"

"$FASM" --emit=macho-obj "$ROOT/fasm/apps/logvec_core.asm" "$CORE_OBJ" >/dev/null
clang++ -std=c++20 -O2 -arch x86_64 -pthread \
  "$ROOT/fasm/apps/ragbox/ragbox.cpp" \
  "$CORE_OBJ" \
  -o "$DEST/ragbox"
chmod +x "$DEST/ragbox"
cp "$ROOT/README.md" "$DEST/README.md"

if ! file "$DEST/ragbox" | grep -q 'Mach-O 64-bit'; then
  echo 'FAIL ragbox is not a Mach-O 64-bit binary' >&2
  file "$DEST/ragbox" >&2
  exit 1
fi

arch -x86_64 "$DEST/ragbox" doctor --skip-ollama >/dev/null

mkdir -p "$OUT_DIR"
if tar --version 2>/dev/null | grep -qi gnu; then
  COPYFILE_DISABLE=1 tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -czf "$OUT" -C "$STAGING" "$PKG"
else
  COPYFILE_DISABLE=1 tar -czf "$OUT" -C "$STAGING" "$PKG"
fi

SHA256="$(if command -v sha256sum >/dev/null 2>&1; then sha256sum "$OUT"; else shasum -a 256 "$OUT"; fi | awk '{print $1}')"
echo "Created $OUT"
echo "sha256: $SHA256"
