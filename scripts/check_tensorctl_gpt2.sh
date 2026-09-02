#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${GPT2_SAFETENSORS:-/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors}"
TOKENIZER="${GPT2_TOKENIZER_FIXTURE_DIR:-$ROOT/scratchpad/gpt2_tokenizer_fixture}"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-gpt2-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

[[ -f "$MODEL" ]] || { echo "tensorctl gpt2 check requires GPT-2 model: $MODEL" >&2; exit 1; }
[[ -f "$TOKENIZER/vocab.json" && -f "$TOKENIZER/merges.txt" ]] || {
  echo "tensorctl gpt2 check requires tokenizer files under: $TOKENIZER" >&2; exit 1;
}

"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null
run=(arch -x86_64 "$OUT_DIR/tensorctl" gpt2 --model "$MODEL" --tokenizer "$TOKENIZER")

actual="$("${run[@]}" --prompt "The quick brown fox" --tokens 12 2>"$OUT_DIR/stderr")"
expected="The quick brown foxes are a great way to get a little bit of a"
[[ "$actual" == "$expected" ]] || { printf 'unexpected GPT-2 output:\n%s\n' "$actual" >&2; exit 1; }
scalar="$("${run[@]}" --backend scalar --prompt "The quick brown fox" --tokens 12 2>/dev/null)"
[[ "$scalar" == "$actual" ]] || { echo "scalar/Accelerate token output mismatch" >&2; exit 1; }
grep -q '^Loading GPT-2 124M' "$OUT_DIR/stderr"

if "${run[@]}" --prompt "" --tokens 1 >/dev/null 2>&1; then echo "empty prompt accepted" >&2; exit 1; fi
if "${run[@]}" --prompt "hello" --tokens 0 >/dev/null 2>&1; then echo "zero token count accepted" >&2; exit 1; fi
if "${run[@]}" --prompt "привет" --tokens 1 >/dev/null 2>&1; then echo "unsupported Unicode prompt accepted" >&2; exit 1; fi
if "${run[@]}" --backend bogus --prompt "hello" --tokens 1 >/dev/null 2>&1; then echo "invalid backend accepted" >&2; exit 1; fi
if arch -x86_64 "$OUT_DIR/tensorctl" gpt2 --model "$OUT_DIR/missing.safetensors" --tokenizer "$TOKENIZER" --prompt "hello" --tokens 1 >/dev/null 2>&1; then
  echo "missing model accepted" >&2; exit 1
fi

echo "tensorctl gpt2 check passed: real GPT-2 output exact, empty/invalid/Unicode/missing-model inputs fail closed"
