#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${FASM_BOOTSTRAP_IMAGE:-debian:bookworm-slim}"
BUILD_DIR="$ROOT/build/out/macos-x64"
ELF_OUT="$BUILD_DIR/fasm-macos-x64.elf"
MACHO_OUT="$BUILD_DIR/fasm-macos-x64"

mkdir -p "$BUILD_DIR"

docker run --rm --platform linux/amd64 \
  -v "$ROOT:/workspace/fasm" \
  -w /workspace/fasm/source/macos/x64 \
  "$IMAGE" \
  bash -lc 'set -e; chmod +x /workspace/fasm/fasm.x64; /workspace/fasm/fasm.x64 fasm.asm /workspace/fasm/build/out/macos-x64/fasm-macos-x64.elf'

python3 "$ROOT/tools/elf64_to_macho64.py" "$ELF_OUT" "$MACHO_OUT"

echo "Built $MACHO_OUT"
file "$MACHO_OUT"
