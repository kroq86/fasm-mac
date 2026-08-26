#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-hostile.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -Wall -Wextra -Werror -O1 -g -fsanitize=address,undefined \
  "$ROOT/fasm/spikes/tensor_checkpoint_hostile_check.c" -o "$OUT_DIR/check"
"$OUT_DIR/check"
