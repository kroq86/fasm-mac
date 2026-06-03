#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/raymaze-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

OBJ="$OUT_DIR/raymaze.o"
BIN="$OUT_DIR/raymaze"
PPM="$OUT_DIR/snapshot.ppm"

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

fasm --emit=macho-obj "$ROOT/fasm/apps/raymaze.asm" "$OBJ" >/dev/null
if RAYLIB_FLAGS="$(raylib_flags)"; then
  clang -arch x86_64 "$OBJ" $RAYLIB_FLAGS -o "$BIN"
else
  cat > "$OUT_DIR/raylib_stubs.c" <<'C'
void InitWindow(int width, int height, const char *title) { (void)width; (void)height; (void)title; }
void SetTargetFPS(int fps) { (void)fps; }
int WindowShouldClose(void) { return 1; }
int IsKeyDown(int key) { (void)key; return 0; }
void BeginDrawing(void) {}
void ClearBackground(unsigned int color) { (void)color; }
void DrawRectangle(int x, int y, int width, int height, unsigned int color) { (void)x; (void)y; (void)width; (void)height; (void)color; }
void DrawText(const char *text, int x, int y, int fontSize, unsigned int color) { (void)text; (void)x; (void)y; (void)fontSize; (void)color; }
void EndDrawing(void) {}
void CloseWindow(void) {}
C
  clang -arch x86_64 -c "$OUT_DIR/raylib_stubs.c" -o "$OUT_DIR/raylib_stubs.o"
  clang -arch x86_64 "$OBJ" "$OUT_DIR/raylib_stubs.o" -o "$BIN"
  echo 'raymaze check: x86_64 raylib not found; using local stubs for CLI/snapshot smoke' >&2
fi

if ! arch -x86_64 "$BIN" --help | grep -q 'usage: raymaze'; then
  echo 'FAIL raymaze help output' >&2
  exit 1
fi

if arch -x86_64 "$BIN" --bad-option >/dev/null 2>"$OUT_DIR/bad.err"; then
  echo 'FAIL raymaze bad option should exit nonzero' >&2
  exit 1
fi
grep -q 'usage: raymaze' "$OUT_DIR/bad.err"

arch -x86_64 "$BIN" --snapshot "$PPM"
header_hex="$(od -An -tx1 -N15 "$PPM" | tr -d ' \n')"
if [[ "$header_hex" != "50360a343030203130300a3235350a" ]]; then
  echo 'FAIL raymaze snapshot header' >&2
  exit 1
fi

size="$(wc -c < "$PPM" | tr -d ' ')"
if [[ "$size" != "120015" ]]; then
  printf 'FAIL raymaze snapshot size expected 120015 got %s\n' "$size" >&2
  exit 1
fi

hash="$(shasum -a 256 "$PPM" | awk '{print $1}')"
expected='7ade80f75d02a99e31414cffe51c06286cf6be0f5aed3d454e51c02bf829e244'
if [[ "$hash" != "$expected" ]]; then
  printf 'FAIL raymaze snapshot hash\nexpected: %s\nactual:   %s\n' "$expected" "$hash" >&2
  exit 1
fi

echo 'raymaze checks passed'
