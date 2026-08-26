#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-differential.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OUT_DIR/executor.o" >/dev/null
clang -arch x86_64 -std=c11 -Wall -Wextra -Werror -Wno-misleading-indentation -O2 \
  "$ROOT/fasm/spikes/tensor_differential_mlp_check.c" "$OUT_DIR/executor.o" -o "$OUT_DIR/mlp"
arch -x86_64 "$OUT_DIR/mlp"

clang -arch x86_64 -std=c11 -Wall -Wextra -Werror -Wno-misleading-indentation -Wno-unused-function -Wno-unused-parameter -O2 \
  "$ROOT/fasm/spikes/tensor_differential_transformer_check.c" "$OUT_DIR/executor.o" -o "$OUT_DIR/transformer"
arch -x86_64 "$OUT_DIR/transformer"

# The retained end-to-end programs are independent implementations. Keep
# their public outcomes in the same gate while the detailed Transformer and
# MNIST array snapshots are added without modifying either implementation.
"$ROOT/scripts/check_tensor_merged_transformer_spike.sh"
"$ROOT/scripts/check_tensor_graph_engine_transformer_spike.sh"

if [[ -n "${MNIST_DIR:-}" ]]; then
  clang -arch x86_64 -std=c11 -Wall -Wextra -Werror -Wno-misleading-indentation -O2 \
    "$ROOT/fasm/spikes/tensor_differential_mnist_check.c" "$OUT_DIR/executor.o" -o "$OUT_DIR/mnist-snapshot"
  arch -x86_64 env MNIST_DIR="$MNIST_DIR" "$OUT_DIR/mnist-snapshot"
  canonical="$(MNIST_DIR="$MNIST_DIR" "$ROOT/scripts/check_tensor_merged_mnist_spike.sh")"
  oracle="$(MNIST_DIR="$MNIST_DIR" "$ROOT/scripts/check_tensor_graph_engine_mnist_spike.sh")"
  printf '%s\n%s\n' "$canonical" "$oracle"
  ca="$(sed -nE 's/.*test_accuracy=[^>]*->([0-9.]+)%.*/\1/p' <<<"$canonical")"
  oa="$(sed -nE 's/.*test_accuracy=[^>]*->([0-9.]+)%.*/\1/p' <<<"$oracle")"
  [[ -n "$ca" && "$ca" == "$oa" ]] || { echo "DIFF mnist accuracy: canonical=${ca:-missing} oracle=${oa:-missing}" >&2; exit 1; }
  echo "differential MNIST passed: accuracy=$ca% fusion=MBR+MB+PLAIN steps=25"
else
  echo "differential MNIST skipped: set MNIST_DIR to the shared IDX dataset" >&2
fi
