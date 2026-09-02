#!/usr/bin/env bash
set -euo pipefail

REVISION="607a30d783dfa663caf39e06633721c8d4cfcd7e"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi
outdir="$1"
mkdir -p "$outdir"

expected_size() { case "$1" in vocab.json) echo 1042301 ;; merges.txt) echo 456318 ;; esac; }
expected_sha256() { case "$1" in
  vocab.json) echo 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783 ;;
  merges.txt) echo 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5 ;;
esac; }

verify() {
  local file="$1" name="$2" size sha
  size="$(wc -c < "$file" | tr -d '[:space:]')"
  [[ "$size" == "$(expected_size "$name")" ]] || return 1
  sha="$(shasum -a 256 "$file" | awk '{print $1}')"
  [[ "$sha" == "$(expected_sha256 "$name")" ]]
}

for name in vocab.json merges.txt; do
  output="$outdir/$name"
  if [[ -e "$output" ]]; then
    if verify "$output" "$name"; then
      echo "gpt2 tokenizer artifact already verified: $output"
      continue
    fi
    echo "refusing to overwrite existing artifact with wrong size or SHA-256: $output" >&2
    exit 1
  fi
  partial="$(mktemp "${outdir}/.gpt2-tok.partial.XXXXXX")"
  trap 'rm -f "$partial"' EXIT
  curl --fail --location --retry 3 --output "$partial" "https://huggingface.co/gpt2/resolve/${REVISION}/${name}"
  if ! verify "$partial" "$name"; then
    echo "downloaded $name failed pinned size/SHA-256 verification" >&2
    exit 1
  fi
  mv "$partial" "$output"
  trap - EXIT
  echo "gpt2 tokenizer artifact fetched and verified: $output"
done
