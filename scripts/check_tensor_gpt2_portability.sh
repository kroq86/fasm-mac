#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DEFAULT="/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
MODEL="${GPT2_SAFETENSORS:-$MODEL_DEFAULT}"
FIXTURES="${GPT2_FIXTURE_DIR:-$ROOT/scratchpad/gpt2_block_boundary}"

[[ -f "$MODEL" ]] || { echo "portability test requires explicit/local GPT-2 artifact: $MODEL" >&2; exit 1; }
[[ -f "$FIXTURES/reference_block0_sha256.txt" ]] || { echo "portability test requires reference fixtures: $FIXTURES" >&2; exit 1; }

tmp="$(mktemp -d "${TMPDIR:-/tmp}/gpt2-stage3-portability.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

"$ROOT/scripts/check_tensor_gpt2_block0_spike.sh" --required --model "$MODEL" --fixtures "$FIXTURES" >/dev/null

if "$ROOT/scripts/check_tensor_gpt2_block0_spike.sh" --required --model "$tmp/missing.safetensors" --fixtures "$FIXTURES" >/dev/null 2>&1; then
  echo "required mode accepted a missing model" >&2; exit 1
fi
if "$ROOT/scripts/check_tensor_gpt2_block0_spike.sh" --required --model "$MODEL" --fixtures "$tmp/missing-fixtures" >/dev/null 2>&1; then
  echo "required mode accepted missing fixtures" >&2; exit 1
fi
optional_out="$("$ROOT/scripts/check_tensor_gpt2_block0_spike.sh" --model "$tmp/missing.safetensors" --fixtures "$FIXTURES")"
[[ "$optional_out" == *"optional gate skipped"* ]] || { echo "optional missing-model path was not an explicit skip" >&2; exit 1; }

copy_stage3_fixtures() {
  local destination="$1"
  mkdir -p "$destination"
  cp "$FIXTURES/gate_config.txt" "$FIXTURES/reference_block0_sha256.txt" "$destination/"
  while read -r _ file; do cp "$FIXTURES/$file" "$destination/$file"; done < "$FIXTURES/reference_block0_sha256.txt"
}

copy_stage3_fixtures "$tmp/corrupt-fixtures"
printf '\001' | dd of="$tmp/corrupt-fixtures/x0_input_hidden.f32" bs=1 seek=0 conv=notrunc status=none
if "$ROOT/scripts/check_tensor_gpt2_block0_spike.sh" --required --model "$MODEL" --fixtures "$tmp/corrupt-fixtures" >/dev/null 2>&1; then
  echo "required mode accepted a corrupt reference fixture" >&2; exit 1
fi

copy_stage3_fixtures "$tmp/appended-fixtures"
printf '\000' >> "$tmp/appended-fixtures/x0_input_hidden.f32"
if "$ROOT/scripts/check_tensor_gpt2_block0_spike.sh" --required --model "$MODEL" --fixtures "$tmp/appended-fixtures" >/dev/null 2>&1; then
  echo "required mode accepted an appended reference fixture" >&2; exit 1
fi

"$ROOT/scripts/fetch-gpt2-124m.sh" "$MODEL" >/dev/null
printf 'wrong' > "$tmp/existing-wrong.safetensors"
wrong_before="$(shasum -a 256 "$tmp/existing-wrong.safetensors" | awk '{print $1}')"
if "$ROOT/scripts/fetch-gpt2-124m.sh" "$tmp/existing-wrong.safetensors" >/dev/null 2>&1; then
  echo "fetch helper accepted an existing wrong artifact" >&2; exit 1
fi
wrong_after="$(shasum -a 256 "$tmp/existing-wrong.safetensors" | awk '{print $1}')"
[[ "$wrong_after" == "$wrong_before" ]] || { echo "fetch helper overwrote an existing wrong artifact" >&2; exit 1; }

if "$ROOT/scripts/prepare-gpt2-block0-oracle.sh" "$ROOT/forbidden-venv" "$tmp/oracle" >/dev/null 2>&1; then
  echo "oracle recipe accepted an in-checkout environment" >&2; exit 1
fi

echo "gpt2 stage3 portability contract passed: required_missing=fail optional_missing=explicit-skip fixture_corruption=fail fixture_append=fail offline_path=verified fetch_existing=non-destructive oracle_env=in-checkout-rejected"
