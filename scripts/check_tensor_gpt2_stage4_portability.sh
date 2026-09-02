#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DEFAULT="/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
MODEL="${GPT2_SAFETENSORS:-$MODEL_DEFAULT}"
FIXTURES="${GPT2_FIXTURE_DIR:-$ROOT/scratchpad/gpt2_block_boundary}"
TOKENIZER="${GPT2_TOKENIZER_FIXTURE_DIR:-$ROOT/scratchpad/gpt2_tokenizer_fixture}"

[[ -f "$MODEL" ]] || { echo "stage4 portability test requires explicit/local GPT-2 artifact: $MODEL" >&2; exit 1; }
[[ -f "$FIXTURES/reference_sha256.txt" ]] || { echo "stage4 portability test requires reference fixtures: $FIXTURES" >&2; exit 1; }

tmp="$(mktemp -d "${TMPDIR:-/tmp}/gpt2-stage4-portability.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

copy_fixtures() {
  local destination="$1"
  mkdir -p "$destination"
  while read -r _ file; do cp "$FIXTURES/$file" "$destination/$file"; done < "$FIXTURES/reference_sha256.txt"
}

# check_tensor_gpt2_full_spike.sh: required-missing-model fails,
# required-missing-fixtures fails, optional-missing is an explicit skip,
# corrupt fixture fails.
"$ROOT/scripts/check_tensor_gpt2_full_spike.sh" --required --model "$MODEL" --fixtures "$FIXTURES" >/dev/null
if "$ROOT/scripts/check_tensor_gpt2_full_spike.sh" --required --model "$tmp/missing.safetensors" --fixtures "$FIXTURES" >/dev/null 2>&1; then
  echo "gpt2_full: required mode accepted a missing model" >&2; exit 1
fi
if "$ROOT/scripts/check_tensor_gpt2_full_spike.sh" --required --model "$MODEL" --fixtures "$tmp/missing-fixtures" >/dev/null 2>&1; then
  echo "gpt2_full: required mode accepted missing fixtures" >&2; exit 1
fi
full_optional_out="$("$ROOT/scripts/check_tensor_gpt2_full_spike.sh" --model "$tmp/missing.safetensors" --fixtures "$FIXTURES")"
[[ "$full_optional_out" == *"optional gate skipped"* ]] || { echo "gpt2_full: optional missing-model path was not an explicit skip" >&2; exit 1; }

copy_fixtures "$tmp/corrupt-full"
printf '\001' | dd of="$tmp/corrupt-full/full_logits.f32" bs=1 seek=0 conv=notrunc status=none
if "$ROOT/scripts/check_tensor_gpt2_full_spike.sh" --required --model "$MODEL" --fixtures "$tmp/corrupt-full" >/dev/null 2>&1; then
  echo "gpt2_full: required mode accepted a corrupt reference fixture" >&2; exit 1
fi
echo "gpt2_full portability: required_missing=fail optional_missing=explicit-skip fixture_corruption=fail"

# check_tensor_gpt2_generation_spike.sh: same required/missing/corrupt
# contract, plus a malformed greedy_reference.txt must fail closed, not
# silently misparse.
if "$ROOT/scripts/check_tensor_gpt2_generation_spike.sh" --required --model "$tmp/missing.safetensors" --fixtures "$FIXTURES" --tokenizer "$TOKENIZER" >/dev/null 2>&1; then
  echo "gpt2_generation: required mode accepted a missing model" >&2; exit 1
fi
gen_optional_out="$("$ROOT/scripts/check_tensor_gpt2_generation_spike.sh" --model "$tmp/missing.safetensors" --fixtures "$FIXTURES" --tokenizer "$TOKENIZER")"
[[ "$gen_optional_out" == *"optional gate skipped"* ]] || { echo "gpt2_generation: optional missing-model path was not an explicit skip" >&2; exit 1; }

copy_fixtures "$tmp/malformed-greedy"
echo "999999999 1 2 3" > "$tmp/malformed-greedy/greedy_reference.txt"
# regenerate the checksum manifest so the corruption under test is the
# malformed CONTENT semantics (out-of-bounds length), not merely a
# fingerprint mismatch the shasum -c step would already catch first
(cd "$tmp/malformed-greedy" && shasum -a 256 -- * > reference_sha256.txt 2>/dev/null || true)
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gpt2-gen-malformed.XXXXXX")"
clang -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function \
  -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/tensor_gpt2_generation_differential_check.c" -lm -o "$OUT_DIR/check"
if "$OUT_DIR/check" "$tmp/malformed-greedy" "$MODEL" "$TOKENIZER/vocab.json" "$TOKENIZER/merges.txt" >/dev/null 2>&1; then
  echo "gpt2_generation: accepted a malformed greedy_reference.txt (out-of-bounds prompt length)" >&2; exit 1
fi
rm -rf "$OUT_DIR"
echo "gpt2_generation portability: required_missing=fail optional_missing=explicit-skip malformed_reference=fail-closed"

# check_tensor_gpt2_cached_generation_spike.sh: same contract.
if "$ROOT/scripts/check_tensor_gpt2_cached_generation_spike.sh" --required --model "$tmp/missing.safetensors" --fixtures "$FIXTURES" >/dev/null 2>&1; then
  echo "gpt2_cached_generation: required mode accepted a missing model" >&2; exit 1
fi
cached_optional_out="$("$ROOT/scripts/check_tensor_gpt2_cached_generation_spike.sh" --model "$tmp/missing.safetensors" --fixtures "$FIXTURES")"
[[ "$cached_optional_out" == *"optional gate skipped"* ]] || { echo "gpt2_cached_generation: optional missing-model path was not an explicit skip" >&2; exit 1; }
echo "gpt2_cached_generation portability: required_missing=fail optional_missing=explicit-skip"

# check_tensor_gpt2_bpe_spike.sh: required-missing-tokenizer fails,
# optional-missing is an explicit skip.
if "$ROOT/scripts/check_tensor_gpt2_bpe_spike.sh" --required --fixtures "$tmp/missing-tok" >/dev/null 2>&1; then
  echo "gpt2_bpe: required mode accepted a missing tokenizer fixture" >&2; exit 1
fi
bpe_optional_out="$("$ROOT/scripts/check_tensor_gpt2_bpe_spike.sh" --fixtures "$tmp/missing-tok")"
[[ "$bpe_optional_out" == *"optional gate skipped"* ]] || { echo "gpt2_bpe: optional missing-tokenizer path was not an explicit skip" >&2; exit 1; }
echo "gpt2_bpe portability: required_missing=fail optional_missing=explicit-skip"

echo "gpt2 stage4 portability contract passed: full/generation/cached_generation/bpe gates all fail closed under --required, skip explicitly (not silently) when optional, and reject malformed reference data"
