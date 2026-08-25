#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-compiler-planner.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -arch x86_64 -std=c11 -O2 "$ROOT/fasm/spikes/tensor_compiler_planner_check.c" -o "$OUT_DIR/check"
arch -x86_64 "$OUT_DIR/check"
