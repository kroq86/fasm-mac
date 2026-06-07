#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zig-bridge-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

if ! command -v zig >/dev/null 2>&1; then
    echo 'FAIL zig is required for zig bridge check' >&2
    exit 1
fi

CRC_OBJ="$OUT_DIR/crc32c_bridge.o"
CRC_BIN="$OUT_DIR/crc32c_bridge"
VEC_OBJ="$OUT_DIR/vec_bridge.o"
VEC_BIN="$OUT_DIR/vec_bench"
VEC_DYLIB="$OUT_DIR/vec_bridge.dylib"

fasm --emit=macho-obj "$ROOT/fasm/tests/zig-bridge/crc32c_bridge.asm" "$CRC_OBJ" >/dev/null
zig build-exe \
    "$ROOT/fasm/tests/zig-bridge/crc32c_bridge.zig" \
    "$CRC_OBJ" \
    -target x86_64-macos \
    -mcpu=baseline \
    -O ReleaseSafe \
    -femit-bin="$CRC_BIN"
arch -x86_64 "$CRC_BIN"

fasm --emit=macho-obj "$ROOT/fasm/tests/zig-bridge/vec_bridge.asm" "$VEC_OBJ" >/dev/null
zig build-exe \
    "$ROOT/fasm/tests/zig-bridge/vec_bench.zig" \
    "$VEC_OBJ" \
    -target x86_64-macos \
    -mcpu=baseline \
    -O ReleaseFast \
    -femit-bin="$VEC_BIN"
VEC_OUT="$(arch -x86_64 "$VEC_BIN" 2>&1)"
printf '%s\n' "$VEC_OUT"

good_dot_ratio="$(printf '%s\n' "$VEC_OUT" | sed -n 's/^zig-bridge perf fasm_vs_zig_native=[0-9.]* good_dot_ratio=\([0-9.]*\) bad_over_good=.*/\1/p')"
bad_over_good="$(printf '%s\n' "$VEC_OUT" | sed -n 's/^zig-bridge perf fasm_vs_zig_native=[0-9.]* good_dot_ratio=[0-9.]* bad_over_good=\([0-9.]*\).*/\1/p')"

if [[ -z "$good_dot_ratio" || -z "$bad_over_good" ]]; then
    echo 'FAIL could not parse zig vec bench output' >&2
    printf '%s\n' "$VEC_OUT" >&2
    exit 1
fi

if awk -v ratio="$good_dot_ratio" 'BEGIN { exit !(ratio <= 2.5) }'; then
    :
else
    echo "FAIL zig good search too heavy vs one dot: $good_dot_ratio > 2.5" >&2
    exit 1
fi

if awk -v ratio="$bad_over_good" 'BEGIN { exit !(ratio >= 2.0) }'; then
    :
else
    echo "FAIL python-style pattern not slower enough: $bad_over_good < 2.0" >&2
    exit 1
fi

if command -v clang >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    clang -arch x86_64 -dynamiclib "$VEC_OBJ" -o "$VEC_DYLIB" >/dev/null
    set +e
    PY_OUT="$(arch -x86_64 python3 "$ROOT/fasm/tests/zig-bridge/vec_py_style_bench.py" "$VEC_DYLIB" 2>&1)"
    py_status=$?
    set -e
    if [[ "$py_status" -eq 0 ]]; then
        printf '%s (optional diagnostic)\n' "$PY_OUT"
    else
        echo 'zig-bridge python diagnostic skipped (need x86_64 python under Rosetta)'
    fi
fi

echo 'zig bridge checks passed'
