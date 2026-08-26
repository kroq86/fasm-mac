#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/fasm/spikes/gunpoint_baseline.py" --self-test
if [[ -z "${GUNPOINT_DIR:-}" && ( -z "${GUNPOINT_TRAIN:-}" || -z "${GUNPOINT_TEST:-}" ) ]]; then
  printf '%s\n' 'gunpoint dataset check skipped: set GUNPOINT_DIR, or GUNPOINT_TRAIN and GUNPOINT_TEST'
  exit 0
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gunpoint.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
if [[ -n "${GUNPOINT_DIR:-}" ]]; then
  python3 "$ROOT/fasm/spikes/gunpoint_baseline.py" "$GUNPOINT_DIR" \
    --export-sequence "$OUT_DIR/sequence.bin"
else
  python3 "$ROOT/fasm/spikes/gunpoint_baseline.py" --train "$GUNPOINT_TRAIN" --test "$GUNPOINT_TEST" \
    --export-sequence "$OUT_DIR/sequence.bin"
fi
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OUT_DIR/executor.o" >/dev/null
clang -arch x86_64 -O2 -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/gunpoint_transformer_train.c" "$OUT_DIR/executor.o" -o "$OUT_DIR/train"
python3 "$ROOT/fasm/spikes/gunpoint_sequence_ablation.py" "$OUT_DIR/sequence.bin"
arch -x86_64 "$OUT_DIR/train" "$OUT_DIR/sequence.bin" ordered
arch -x86_64 "$OUT_DIR/train" "$OUT_DIR/sequence.bin" shuffled
