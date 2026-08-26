#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/fasm/spikes/nir_mfco_baseline.py" --self-test
if [[ -z "${NIR_MFCO_ARCHIVE:-}" ]]; then
  printf '%s\n' 'nir-mfco transformer dataset check skipped: set NIR_MFCO_ARCHIVE'
  exit 0
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nir-mfco-transformer.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
python3 "$ROOT/fasm/spikes/nir_mfco_baseline.py" --verify-md5 \
  --export-sequence "$OUT_DIR/sequence.bin" "$NIR_MFCO_ARCHIVE" >/dev/null
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OUT_DIR/executor.o" >/dev/null
clang -arch x86_64 -O2 -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/nir_mfco_transformer_train.c" "$OUT_DIR/executor.o" -o "$OUT_DIR/train"
python3 "$ROOT/fasm/spikes/nir_mfco_sequence_ablation.py" "$OUT_DIR/sequence.bin"
arch -x86_64 "$OUT_DIR/train" "$OUT_DIR/sequence.bin" ordered
arch -x86_64 "$OUT_DIR/train" "$OUT_DIR/sequence.bin" shuffled
