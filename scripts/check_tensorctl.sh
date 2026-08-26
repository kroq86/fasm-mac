#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null

mlp_out="$(arch -x86_64 "$OUT_DIR/tensorctl" mlp)"
if ! grep -q "correct=4/4" <<<"$mlp_out"; then
  printf 'FAIL: tensorctl mlp did not converge to the XOR truth table:\n%s\n' "$mlp_out" >&2
  exit 1
fi

# The whole point of the memory report: it must actually be a decision, not a
# fixed string — verify it flips both ways, not just that it prints something.
# --epochs=1 won't converge (exit 1) and that's fine; only the memory report matters here.
generous="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --epochs=1 --memory-budget=8192 || true)"
tight="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --epochs=1 --memory-budget=4928 || true)"
if ! grep -q "decision=save" <<<"$generous"; then
  printf 'FAIL: generous budget did not choose save:\n%s\n' "$generous" >&2
  exit 1
fi
if ! grep -q "decision=rematerialize" <<<"$tight"; then
  printf 'FAIL: tight budget did not choose rematerialize:\n%s\n' "$tight" >&2
  exit 1
fi

train_out="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --epochs=12000)"
if ! grep -q "loss=2.178741->0.000000" <<<"$train_out"; then
  printf 'FAIL: tensorctl transformer training did not converge as expected:\n%s\n' "$train_out" >&2
  exit 1
fi

if arch -x86_64 "$OUT_DIR/tensorctl" >/dev/null 2>&1; then
  printf 'FAIL: tensorctl with no args should exit non-zero\n' >&2
  exit 1
fi

printf 'tensorctl check passed: mlp_converges=yes transformer_converges=yes memory_decision_flips=yes usage_exit_code=nonzero\n'
