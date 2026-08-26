#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-resume.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -Wall -Wextra -Werror -O2 \
  "$ROOT/fasm/spikes/tensor_checkpoint_resume_check.c" -o "$OUT_DIR/check"
"$OUT_DIR/check"
