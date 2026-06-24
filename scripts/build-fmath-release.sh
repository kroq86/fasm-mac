#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?usage: $0 <version> (e.g. 0.1.0)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$(mktemp -d)"
PKG="fmath-${VERSION}-macos-x86_64"
OUT_DIR="$ROOT/dist"
OUT="$OUT_DIR/${PKG}.tar.gz"

cleanup() {
  rm -rf "$STAGING"
}
trap cleanup EXIT

DEST="$STAGING/$PKG"
mkdir -p "$DEST"

fasm "$ROOT/fasm/apps/fmath.asm" "$DEST/fmath" >/dev/null
chmod +x "$DEST/fmath"
cp "$ROOT/README.md" "$DEST/README.md"

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
