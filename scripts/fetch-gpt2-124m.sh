#!/usr/bin/env bash
set -euo pipefail

REVISION="607a30d783dfa663caf39e06633721c8d4cfcd7e"
EXPECTED_SIZE="548105171"
EXPECTED_SHA256="248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
URL="https://huggingface.co/gpt2/resolve/${REVISION}/model.safetensors"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_MODEL_SAFETENSORS" >&2
  exit 2
fi

output="$1"
parent="$(dirname "$output")"
mkdir -p "$parent"

verify() {
  local file="$1" size sha
  size="$(wc -c < "$file" | tr -d '[:space:]')"
  [[ "$size" == "$EXPECTED_SIZE" ]] || return 1
  sha="$(shasum -a 256 "$file" | awk '{print $1}')"
  [[ "$sha" == "$EXPECTED_SHA256" ]]
}

if [[ -e "$output" ]]; then
  if verify "$output"; then
    echo "gpt2-124m artifact already verified: $output"
    exit 0
  fi
  echo "refusing to overwrite existing artifact with wrong size or SHA-256: $output" >&2
  exit 1
fi

partial="$(mktemp "${parent}/.gpt2-124m.partial.XXXXXX")"
trap 'rm -f "$partial"' EXIT
curl --fail --location --retry 3 --output "$partial" "$URL"
if ! verify "$partial"; then
  echo "downloaded GPT-2 artifact failed pinned size/SHA-256 verification" >&2
  exit 1
fi
mv "$partial" "$output"
trap - EXIT
echo "gpt2-124m artifact fetched and verified: $output"
