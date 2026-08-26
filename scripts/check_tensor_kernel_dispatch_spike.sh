#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-dispatch-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -arch arm64 -O3 "$ROOT/fasm/spikes/tensor_kernel_dispatch_check.c" -o "$OUT_DIR/arm64"
"$OUT_DIR/arm64"
clang -arch x86_64 -O3 "$ROOT/fasm/spikes/tensor_kernel_dispatch_check.c" -o "$OUT_DIR/x86_64"
arch -x86_64 "$OUT_DIR/x86_64"
