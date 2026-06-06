#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?usage: $0 <version> (e.g. 0.1.0)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$(mktemp -d)"
PKG="macdbg-${VERSION}-macos-x86_64"
OUT_DIR="$ROOT/dist"
OUT="$OUT_DIR/${PKG}.tar.gz"

cleanup() {
  rm -rf "$STAGING"
}
trap cleanup EXIT

raylib_flags() {
  if [[ -n "${RAYLIB_PREFIX:-}" ]]; then
    if [[ -f "$RAYLIB_PREFIX/lib/libraylib.dylib" ]] && lipo -info "$RAYLIB_PREFIX/lib/libraylib.dylib" 2>/dev/null | grep -q 'x86_64'; then
      printf '%s\n' "-I$RAYLIB_PREFIX/include -L$RAYLIB_PREFIX/lib -Wl,-rpath,$RAYLIB_PREFIX/lib -lraylib -framework CoreVideo -framework IOKit -framework Cocoa -framework OpenGL -framework CoreAudio"
      return 0
    fi
  fi
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists raylib; then
    local libdir
    libdir="$(pkg-config --variable=libdir raylib)"
    if [[ -f "$libdir/libraylib.dylib" ]] && lipo -info "$libdir/libraylib.dylib" 2>/dev/null | grep -q 'x86_64'; then
      pkg-config --cflags --libs raylib
      return 0
    fi
  fi
  if command -v brew >/dev/null 2>&1; then
    local prefix
    prefix="$(brew --prefix raylib 2>/dev/null || true)"
    if [[ -n "$prefix" && -f "$prefix/lib/libraylib.dylib" ]] && lipo -info "$prefix/lib/libraylib.dylib" 2>/dev/null | grep -q 'x86_64'; then
      printf '%s\n' "-I$prefix/include -L$prefix/lib -Wl,-rpath,$prefix/lib -lraylib -framework CoreVideo -framework IOKit -framework Cocoa -framework OpenGL -framework CoreAudio"
      return 0
    fi
  fi
  return 1
}

if ! RAYLIB_FLAGS="$(raylib_flags)"; then
  echo 'macdbg release requires an x86_64 raylib; use an Intel/Rosetta Homebrew raylib install' >&2
  exit 1
fi

DEST="$STAGING/$PKG"
mkdir -p "$DEST"

OBJ="$STAGING/macdbg.o"
fasm --emit=macho-obj "$ROOT/fasm/apps/macdbg.asm" "$OBJ" >/dev/null
clang -arch x86_64 "$OBJ" $RAYLIB_FLAGS -o "$DEST/macdbg"
chmod +x "$DEST/macdbg"
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
