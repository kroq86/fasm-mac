#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE_DIR="${GPT2_FIXTURE_DIR:-$ROOT/scratchpad/gpt2_block_boundary}"
SAFETENSORS="${GPT2_SAFETENSORS:-/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors}"
TOKENIZER_DIR="${GPT2_TOKENIZER_FIXTURE_DIR:-$ROOT/scratchpad/gpt2_tokenizer_fixture}"
REQUIRED="${GPT2_GENERATION_REQUIRED:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --required) REQUIRED=1; shift ;;
    --model) [[ $# -ge 2 ]] || { echo "--model requires a path" >&2; exit 2; }; SAFETENSORS="$2"; shift 2 ;;
    --fixtures) [[ $# -ge 2 ]] || { echo "--fixtures requires a path" >&2; exit 2; }; FIXTURE_DIR="$2"; shift 2 ;;
    --tokenizer) [[ $# -ge 2 ]] || { echo "--tokenizer requires a path" >&2; exit 2; }; TOKENIZER_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

missing() {
  if [[ "$REQUIRED" == 1 ]]; then
    echo "gpt2_generation required gate FAILED: $1" >&2
    exit 1
  fi
  echo "gpt2_generation optional gate skipped: $1"
  exit 0
}

if [[ ! -f "$FIXTURE_DIR/full_logits.f32" ]]; then
  missing "full_logits.f32 not present under $FIXTURE_DIR"
fi
if [[ ! -f "$SAFETENSORS" ]]; then
  missing "real checkpoint not present at $SAFETENSORS"
fi
if [[ ! -f "$FIXTURE_DIR/greedy_reference.txt" ]]; then
  missing "greedy_reference.txt not present under $FIXTURE_DIR"
fi
if [[ ! -f "$TOKENIZER_DIR/vocab.json" || ! -f "$TOKENIZER_DIR/merges.txt" ]]; then
  missing "tokenizer fixture not present under $TOKENIZER_DIR"
fi
if [[ ! -f "$FIXTURE_DIR/reference_sha256.txt" ]]; then
  missing "reference_sha256.txt not present under $FIXTURE_DIR"
fi
(cd "$FIXTURE_DIR" && shasum -a 256 -c reference_sha256.txt >/dev/null) || {
  echo "gpt2_generation: reference fixture fingerprint verification FAILED" >&2
  exit 1
}

OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-gpt2-generation.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function \
  -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/tensor_gpt2_generation_differential_check.c" -lm -o "$OUT_DIR/check"

# Not run-twice like this project's other gates: this check already
# exercises determinism internally (greedy generated twice, temp/top-k
# seed-replayed twice) inside one ~2-2.5min invocation (includes a
# token-for-token check against 4 real PyTorch greedy sequences); a real
# GPT-2 124M forward pass at growing context is expensive enough that
# doubling the whole binary's runtime for an already-internally-covered
# property isn't worth it here.
GPT2_GENERATION_REQUIRED="$REQUIRED" "$OUT_DIR/check" "$FIXTURE_DIR" "$SAFETENSORS" "$TOKENIZER_DIR/vocab.json" "$TOKENIZER_DIR/merges.txt"
