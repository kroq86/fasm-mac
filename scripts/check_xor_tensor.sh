#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/xor-tensor-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

"$ROOT/scripts/build-xor-tensor.sh" "$OUT_DIR/xor" >/dev/null
actual="$(arch -x86_64 "$OUT_DIR/xor")"
expected='xor trained plan_steps=3 fused_contexts=2 matmul_masks=rhs,both scratch_zero_actions=5 remat_actions=1 scratch_bytes=256 loss_milli=0 predictions_milli=0 999 999 0'
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL fused XOR differs from locked unfused baseline: %s\n' "$actual" >&2
  exit 1
fi
printf '%s\n' "$actual"
