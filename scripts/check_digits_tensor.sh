#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/digits-tensor-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
"$ROOT/scripts/build-digits-tensor.sh" "$OUT_DIR/digits" >/dev/null
actual="$(arch -x86_64 "$OUT_DIR/digits")"
case "$actual" in
  'digits trained architecture=7x16x10 plan_steps=6->3 reference=exact accuracy=10/10 loss_milli='*' predictions=0,1,2,3,4,5,6,7,8,9') ;;
  *) printf 'FAIL digits output: %s\n' "$actual" >&2; exit 1;;
esac
printf '%s\n' "$actual"
