#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE_DIR="${GPT2_FIXTURE_DIR:-$ROOT/scratchpad/gpt2_block_boundary}"
SAFETENSORS="${GPT2_SAFETENSORS:-/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors}"
REQUIRED="${GPT2_CACHED_GEN_REQUIRED:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --required) REQUIRED=1; shift ;;
    --model) [[ $# -ge 2 ]] || { echo "--model requires a path" >&2; exit 2; }; SAFETENSORS="$2"; shift 2 ;;
    --fixtures) [[ $# -ge 2 ]] || { echo "--fixtures requires a path" >&2; exit 2; }; FIXTURE_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

missing() {
  if [[ "$REQUIRED" == 1 ]]; then
    echo "gpt2_cached_generation required gate FAILED: $1" >&2
    exit 1
  fi
  echo "gpt2_cached_generation optional gate skipped: $1"
  exit 0
}

if [[ ! -f "$SAFETENSORS" ]]; then
  missing "real checkpoint not present at $SAFETENSORS"
fi
if [[ ! -f "$FIXTURE_DIR/reference_sha256.txt" ]]; then
  missing "reference_sha256.txt not present under $FIXTURE_DIR"
fi
(cd "$FIXTURE_DIR" && shasum -a 256 -c reference_sha256.txt >/dev/null) || {
  echo "gpt2_cached_generation: reference fixture fingerprint verification FAILED" >&2
  exit 1
}

OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-gpt2-cached-generation.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$OUT_DIR/executor.o" >/dev/null
clang -arch x86_64 -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function \
  -DM=768 -DH=12 -DD=64 -DQW=2304 -DF=3072 -DMAXCACHE=64 -I"$ROOT/fasm/spikes" \
  "$ROOT/fasm/spikes/tensor_gpt2_cached_generation_differential_check.c" "$OUT_DIR/executor.o" -lm -o "$OUT_DIR/check"
GPT2_CACHED_GEN_REQUIRED="$REQUIRED" "$OUT_DIR/check" "$FIXTURE_DIR" "$SAFETENSORS"
