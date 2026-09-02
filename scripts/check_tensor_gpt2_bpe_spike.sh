#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE_DIR="${GPT2_TOKENIZER_FIXTURE_DIR:-$ROOT/scratchpad/gpt2_tokenizer_fixture}"
REQUIRED="${GPT2_BPE_REQUIRED:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --required) REQUIRED=1; shift ;;
    --fixtures) [[ $# -ge 2 ]] || { echo "--fixtures requires a path" >&2; exit 2; }; FIXTURE_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$FIXTURE_DIR/vocab.json" || ! -f "$FIXTURE_DIR/merges.txt" ]]; then
  if [[ "$REQUIRED" == 1 ]]; then
    echo "gpt2_bpe required gate FAILED: tokenizer fixture not present under $FIXTURE_DIR" >&2
    exit 1
  fi
  echo "gpt2_bpe optional gate skipped: tokenizer fixture not present under $FIXTURE_DIR (run scripts/fetch-gpt2-tokenizer.sh first)"
  exit 0
fi

OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-gpt2-bpe.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
clang -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function \
  -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/tensor_gpt2_bpe_differential_check.c" -o "$OUT_DIR/check"

first="$("$OUT_DIR/check" "$FIXTURE_DIR/vocab.json" "$FIXTURE_DIR/merges.txt")"
printf '%s\n' "$first"
second="$("$OUT_DIR/check" "$FIXTURE_DIR/vocab.json" "$FIXTURE_DIR/merges.txt")"
[[ "$second" == "$first" ]]
echo "gpt2_bpe: reran, byte-identical native output confirmed (determinism)"
