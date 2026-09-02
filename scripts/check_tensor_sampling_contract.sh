#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-sampling-contract.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
COMMON=(-std=c11 -O2 -Wall -Wextra -Werror -I"$ROOT/fasm/spikes")

clang "${COMMON[@]}" "$ROOT/fasm/spikes/tensor_sampling_contract_check.c" -lm -o "$OUT_DIR/check"
"$OUT_DIR/check"

if nm -a "$OUT_DIR/check" | grep -E 'sample_top_k_temperature.*(taken|idx|prob|scaled)' >/dev/null; then
  echo 'sampling_contract shared_mutable_scratch FAIL' >&2
  exit 1
fi
echo 'sampling_contract shared_mutable_scratch PASS'

clang "${COMMON[@]}" -fsanitize=address -fno-omit-frame-pointer \
  "$ROOT/fasm/spikes/tensor_sampling_contract_check.c" -lm -o "$OUT_DIR/hostile"
for case_name in zero-vocab negative-vocab nan-logit inf-logit nan-temperature inf-temperature null-logits null-rng; do
  ASAN_OPTIONS=abort_on_error=1 "$OUT_DIR/hostile" --invalid "$case_name"
done
ASAN_OPTIONS=abort_on_error=1 "$OUT_DIR/hostile" --hostile-large-vocab
