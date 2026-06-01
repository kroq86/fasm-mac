#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${1:-$HOME/.local/bin}"
mkdir -p "$PREFIX"

ln -sf "$ROOT/bin/fasm" "$PREFIX/fasm"

cat <<EOF
Installed fasm shim:
  $PREFIX/fasm -> $ROOT/bin/fasm

Make sure this is in your PATH:
  export PATH="$PREFIX:\$PATH"

Then:
  fasm fasm/basic/fib.asm
  ./fasm/basic/fib
EOF
