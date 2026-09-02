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
grep -q '^GPT-2 runtime metadata' "$OUT_DIR/stderr"
grep -q '^  backend: accelerate$' "$OUT_DIR/stderr"
grep -q '^  prompt_tokens: 4$' "$OUT_DIR/stderr"
grep -q '^  generated_tokens: 12$' "$OUT_DIR/stderr"
grep -q '^  ttft_ms: [0-9]' "$OUT_DIR/stderr"
grep -q '^  decode_tokens_per_sec: [0-9]' "$OUT_DIR/stderr"
grep -q '^  peak_rss_mb: [0-9]' "$OUT_DIR/stderr"
auto="$("${run[@]}" --backend auto --prompt "The quick brown fox" --tokens 1 2>"$OUT_DIR/auto.stderr")"
[[ "$auto" == "The quick brown foxes" ]] || { echo "auto backend output mismatch" >&2; exit 1; }
grep -q '^  backend: auto$' "$OUT_DIR/auto.stderr"
grep -q '^  planner_profile_ms: [0-9]' "$OUT_DIR/auto.stderr"
grep -q '^  kernel_plan\[0\]:' "$OUT_DIR/auto.stderr"
grep -q '^  sampling: greedy_argmax$' "$OUT_DIR/stderr"

sample1="$("${run[@]}" --prompt "how many stars?" --tokens 8 --temperature 0.8 --top-k 40 --seed 42 2>/dev/null)"
sample2="$("${run[@]}" --prompt "how many stars?" --tokens 8 --temperature 0.8 --top-k 40 --seed 42 2>/dev/null)"
[[ "$sample1" == "$sample2" ]] || { echo "seeded sampling replay mismatch" >&2; exit 1; }

if "${run[@]}" --prompt "" --tokens 1 >/dev/null 2>&1; then echo "empty prompt accepted" >&2; exit 1; fi
if "${run[@]}" --prompt "hello" --tokens 0 >/dev/null 2>&1; then echo "zero token count accepted" >&2; exit 1; fi
if "${run[@]}" --prompt "привет" --tokens 1 >/dev/null 2>&1; then echo "unsupported Unicode prompt accepted" >&2; exit 1; fi
if "${run[@]}" --backend bogus --prompt "hello" --tokens 1 >/dev/null 2>&1; then echo "invalid backend accepted" >&2; exit 1; fi
if "${run[@]}" --temperature nan --prompt "hello" --tokens 1 >/dev/null 2>&1; then echo "NaN temperature accepted" >&2; exit 1; fi
if "${run[@]}" --top-k 257 --prompt "hello" --tokens 1 >/dev/null 2>&1; then echo "oversized top-k accepted" >&2; exit 1; fi
if arch -x86_64 "$OUT_DIR/tensorctl" gpt2 --model "$OUT_DIR/missing.safetensors" --tokenizer "$TOKENIZER" --prompt "hello" --tokens 1 >/dev/null 2>&1; then
  echo "missing model accepted" >&2; exit 1
fi

echo "tensorctl gpt2 check passed: real GPT-2 output exact, empty/invalid/Unicode/missing-model inputs fail closed"
