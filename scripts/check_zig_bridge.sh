#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zig-bridge-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

if ! command -v zig >/dev/null 2>&1; then
    echo 'FAIL zig is required for zig bridge check' >&2
    exit 1
fi

OBJ="$OUT_DIR/crc32c_bridge.o"
BIN="$OUT_DIR/crc32c_bridge"

fasm --emit=macho-obj "$ROOT/fasm/tests/zig-bridge/crc32c_bridge.asm" "$OBJ" >/dev/null
zig build-exe \
    "$ROOT/fasm/tests/zig-bridge/crc32c_bridge.zig" \
    "$OBJ" \
    -target x86_64-macos \
    -mcpu=baseline \
    -O ReleaseSafe \
    -femit-bin="$BIN"

arch -x86_64 "$BIN"

echo 'zig bridge checks passed'
