#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-agent-transaction.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -O2 -Wall -Wextra -Werror \
  "$ROOT/fasm/spikes/tensor_agent_transaction_spike.c" -o "$OUT_DIR/check"
first="$($OUT_DIR/check)"
for _ in 1 2 3 4; do [[ "$($OUT_DIR/check)" == "$first" ]]; done
printf '%s\n' "$first"
