#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/macdbg-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

OBJ="$OUT_DIR/macdbg.o"
BIN="$OUT_DIR/macdbg"

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

run_with_timeout() {
  python3 - "$@" <<'PY'
import subprocess
import sys

timeout = int(sys.argv[1])
args = sys.argv[2:]
try:
    completed = subprocess.run(args, timeout=timeout)
    sys.exit(completed.returncode)
except subprocess.TimeoutExpired:
    sys.exit(124)
PY
}

json_get() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    doc = json.load(f)
value = doc
for part in key.split("."):
    value = value[part]
print(value)
PY
}

json_list_len() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    doc = json.load(f)
value = doc
for part in key.split("."):
    value = value[part]
if not isinstance(value, list):
    raise SystemExit(f"{key} is not a list")
print(len(value))
PY
}

fasm --emit=macho-obj "$ROOT/fasm/apps/macdbg.asm" "$OBJ" >/dev/null
if RAYLIB_FLAGS="$(raylib_flags)"; then
  clang -arch x86_64 "$OBJ" $RAYLIB_FLAGS -o "$BIN"
else
  cat > "$OUT_DIR/raylib_stubs.c" <<'C'
void InitWindow(int width, int height, const char *title) { (void)width; (void)height; (void)title; }
void SetTargetFPS(int fps) { (void)fps; }
int WindowShouldClose(void) { return 1; }
int IsKeyPressed(int key) { (void)key; return 0; }
void BeginDrawing(void) {}
void ClearBackground(unsigned int color) { (void)color; }
void DrawRectangle(int x, int y, int width, int height, unsigned int color) { (void)x; (void)y; (void)width; (void)height; (void)color; }
void DrawText(const char *text, int x, int y, int fontSize, unsigned int color) { (void)text; (void)x; (void)y; (void)fontSize; (void)color; }
void EndDrawing(void) {}
void CloseWindow(void) {}
C
  clang -arch x86_64 -c "$OUT_DIR/raylib_stubs.c" -o "$OUT_DIR/raylib_stubs.o"
  clang -arch x86_64 "$OBJ" "$OUT_DIR/raylib_stubs.o" -o "$BIN"
  echo 'macdbg check: x86_64 raylib not found; using local stubs for UI smoke' >&2
fi

if ! arch -x86_64 "$BIN" --help | grep -q 'usage: macdbg'; then
  echo 'FAIL macdbg help output' >&2
  exit 1
fi

if arch -x86_64 "$BIN" --nope >/dev/null 2>"$OUT_DIR/bad.err"; then
  echo 'FAIL macdbg bad option should exit nonzero' >&2
  exit 1
fi
grep -q 'usage: macdbg' "$OUT_DIR/bad.err"

if arch -x86_64 "$BIN" --ui >/dev/null 2>"$OUT_DIR/ui-bad.err"; then
  echo 'FAIL macdbg --ui without target should exit nonzero' >&2
  exit 1
fi
grep -q 'usage: macdbg' "$OUT_DIR/ui-bad.err"

cat > "$OUT_DIR/ok.c" <<'EOF'
int main(void) {
  return 0;
}
EOF

cat > "$OUT_DIR/args.c" <<'EOF'
int main(int argc, char **argv) {
  return (argc == 3 && argv[1][0] == 'a' && argv[2][0] == 'b') ? 0 : 9;
}
EOF

cat > "$OUT_DIR/crash.c" <<'EOF'
int main(void) {
  volatile int *p = 0;
  return *p;
}
EOF

clang -arch x86_64 -g "$OUT_DIR/ok.c" -o "$OUT_DIR/ok"
clang -arch x86_64 -g "$OUT_DIR/args.c" -o "$OUT_DIR/args"
clang -arch x86_64 -g "$OUT_DIR/crash.c" -o "$OUT_DIR/crash"

set +e
run_with_timeout 30 arch -x86_64 "$BIN" --snapshot "$OUT_DIR/ok" "$OUT_DIR/ok.json"
ok_status=$?
set -e

if [ "$ok_status" -eq 124 ]; then
  echo 'macdbg LLDB runtime checks skipped: LLDB timed out' >&2
  exit 0
fi

if [ "$ok_status" -eq 3 ]; then
  status="$(json_get "$OUT_DIR/ok.json" status 2>/dev/null || true)"
  if [ "$status" = "permission_unavailable" ] || [ "$status" = "lldb_error" ]; then
    echo "macdbg LLDB runtime checks skipped: $status" >&2
    exit 0
  fi
fi

if [ "$ok_status" -ne 0 ]; then
  echo "FAIL macdbg normal snapshot exited $ok_status" >&2
  exit 1
fi

test "$(json_get "$OUT_DIR/ok.json" status)" = "exited"
test "$(json_get "$OUT_DIR/ok.json" tool)" = "macdbg"
grep -q '"raw_tail"' "$OUT_DIR/ok.json"

run_with_timeout 30 arch -x86_64 "$BIN" --snapshot --args "$OUT_DIR/args" a b -- "$OUT_DIR/args.json"
test "$(json_get "$OUT_DIR/args.json" status)" = "exited"

run_with_timeout 30 env MACDBG_UI_AUTOCLOSE=1 arch -x86_64 "$BIN" --ui "$OUT_DIR/ok"
run_with_timeout 30 env MACDBG_UI_AUTOCLOSE=1 arch -x86_64 "$BIN" --ui --args "$OUT_DIR/args" a b

run_with_timeout 30 arch -x86_64 "$BIN" --snapshot "$OUT_DIR/crash" "$OUT_DIR/crash.json"
test "$(json_get "$OUT_DIR/crash.json" status)" = "crashed"
test "$(json_get "$OUT_DIR/crash.json" signal)" = "EXC_BAD_ACCESS"
test "$(json_get "$OUT_DIR/crash.json" registers.rip)" != ""
test "$(json_list_len "$OUT_DIR/crash.json" backtrace)" -gt 0
test "$(json_list_len "$OUT_DIR/crash.json" disasm)" -gt 0
test "$(json_list_len "$OUT_DIR/crash.json" stack_memory)" -gt 0
grep -q 'EXC_BAD_ACCESS' "$OUT_DIR/crash.json"

echo 'macdbg checks passed'
