#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/fasm/build/out/tensor-kv-handoff}"
TMP_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/kv-handoff-build.XXXXXX")"
trap 'rm -rf "$TMP_BUILD"' EXIT
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$TMP_BUILD/executor.o" >/dev/null
clang -arch x86_64 -std=c11 -O2 -Wall -Wextra -Werror -DACCELERATE_NEW_LAPACK \
  -c "$ROOT/fasm/examples/tensorctl_gpt2_accelerate.c" -o "$TMP_BUILD/accelerate.o"
clang -arch x86_64 -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-parameter \
  -DM=768 -DH=12 -DD=64 -DQW=2304 -DF=3072 -DMAXCACHE=64 -DGPT2_MAXT=64 \
  -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_kv_handoff.c" "$TMP_BUILD/accelerate.o" \
  "$TMP_BUILD/executor.o" -framework Accelerate -o "$TMP_BUILD/tensor-kv-handoff"
mkdir -p "$(dirname "$OUT")"
mv "$TMP_BUILD/tensor-kv-handoff" "$OUT"
echo "Built $OUT"
