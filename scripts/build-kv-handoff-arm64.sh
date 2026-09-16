#!/usr/bin/env bash
# Experimental arm64-native build of tensor-kv-handoff. Parallel to
# scripts/build-kv-handoff.sh (unmodified, x86_64+FASM, still the primary
# build); this one exists to measure native latency without Rosetta 2
# translation on Apple Silicon hosts. It replaces the linked
# tensor_transformer_executor_f32.asm object (genuine x86-64 assembly,
# not portable) with the portable-C equivalent in
# fasm/spikes/tensor_transformer_executor_f32_native.c, following the same
# pattern already used by the arm64-native killer-workload spikes
# (fasm/spikes/tensor_killer_mnist_native.c and its build in
# scripts/check_tensor_killer_mnist.sh). No FASM invocation and no -arch
# x86_64 flag anywhere in this script.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/fasm/build/out/tensor-kv-handoff-arm64}"
TMP_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/kv-handoff-arm64-build.XXXXXX")"
trap 'rm -rf "$TMP_BUILD"' EXIT
clang -arch arm64 -std=c11 -O2 -Wall -Wextra -Werror -DACCELERATE_NEW_LAPACK \
  -c "$ROOT/fasm/examples/tensorctl_gpt2_accelerate.c" -o "$TMP_BUILD/accelerate.o"
clang -arch arm64 -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-function \
  -c "$ROOT/fasm/spikes/tensor_transformer_executor_f32_native.c" -o "$TMP_BUILD/executor.o"
clang -arch arm64 -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-parameter \
  -DM=768 -DH=12 -DD=64 -DQW=2304 -DF=3072 -DMAXCACHE=64 -DGPT2_MAXT=64 \
  -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_kv_handoff.c" "$TMP_BUILD/accelerate.o" \
  "$TMP_BUILD/executor.o" -framework Accelerate -o "$TMP_BUILD/tensor-kv-handoff-arm64"
mkdir -p "$(dirname "$OUT")"
mv "$TMP_BUILD/tensor-kv-handoff-arm64" "$OUT"
echo "Built $OUT"
