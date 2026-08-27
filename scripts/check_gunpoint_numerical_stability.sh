#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${GUNPOINT_DIR:-}" && ( -z "${GUNPOINT_TRAIN:-}" || -z "${GUNPOINT_TEST:-}" ) ]]; then
  printf '%s\n' 'GunPoint numerical-stability check skipped: set GUNPOINT_DIR, or GUNPOINT_TRAIN and GUNPOINT_TEST'
  exit 0
fi

OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gunpoint-stability.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
if [[ -n "${GUNPOINT_DIR:-}" ]]; then
  python3 "$ROOT/fasm/spikes/gunpoint_baseline.py" "$GUNPOINT_DIR" --export-sequence "$OUT_DIR/sequence.bin"
else
  python3 "$ROOT/fasm/spikes/gunpoint_baseline.py" --train "$GUNPOINT_TRAIN" --test "$GUNPOINT_TEST" --export-sequence "$OUT_DIR/sequence.bin"
fi

fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OUT_DIR/executor.o" >/dev/null
clang -arch x86_64 -O2 -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/gunpoint_differential_check.c" "$OUT_DIR/executor.o" -o "$OUT_DIR/check"

probe="$(arch -x86_64 "$OUT_DIR/check" "$OUT_DIR/sequence.bin" --probe)"
trace="$(arch -x86_64 "$OUT_DIR/check" "$OUT_DIR/sequence.bin" --stability-trace "$OUT_DIR/stability.csv")"
plot="$(python3 "$ROOT/fasm/spikes/plot_gunpoint_stability.py" "$OUT_DIR/stability.csv" "$OUT_DIR/stability.svg")"
ordered="$(arch -x86_64 "$OUT_DIR/check" "$OUT_DIR/sequence.bin" --train-fused-ordered)"
shuffled="$(arch -x86_64 "$OUT_DIR/check" "$OUT_DIR/sequence.bin" --train-fused-shuffled)"
printf '%s\n%s\n%s\n%s\n%s\n' "$probe" "$trace" "$plot" "$ordered" "$shuffled"

grep -q 'max_ulp=2' <<<"$probe"
grep -q 'first_numerical_update=2 first_material_update=10874' <<<"$trace"
grep -q 'final=0.827' <<<"$ordered"
grep -q 'final=0.733' <<<"$shuffled"
awk -F, 'NR==1 { next } $7==1 { numerical++ } $8==1 { material++ } END { exit !(numerical==1 && material>=1) }' "$OUT_DIR/stability.csv"
grep -q 'first persistent-state divergence (1 ULP)' "$OUT_DIR/stability.svg"
grep -q 'first material logit divergence' "$OUT_DIR/stability.svg"
printf 'GunPoint numerical-stability evidence passed: trace_rows=%s\n' "$(( $(wc -l < "$OUT_DIR/stability.csv") - 1 ))"
