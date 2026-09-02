#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 2 ]]; then
  echo "usage: $0 EXTERNAL_VENV_DIR EXTERNAL_OUTPUT_DIR" >&2
  exit 2
fi
venv_dir="$1"
output_dir="$2"
[[ "$venv_dir" == /* && "$output_dir" == /* ]] || {
  echo "oracle environment and output paths must be absolute" >&2
  exit 2
}
case "$venv_dir/" in
  "$ROOT"/*) echo "refusing to create the oracle environment inside the checkout" >&2; exit 1 ;;
esac
case "$output_dir/" in
  "$ROOT"/*) echo "refusing to create oracle output inside the checkout" >&2; exit 1 ;;
esac

python3.11 -m venv "$venv_dir"
"$venv_dir/bin/pip" install \
  'torch==2.13.0' \
  'transformers==5.16.1' \
  'huggingface-hub==1.29.0' \
  'safetensors==0.8.0' \
  'numpy==2.4.6'
GPT2_ORACLE_OUT="$output_dir" "$venv_dir/bin/python" \
  "$ROOT/scratchpad/gpt2_block_boundary/generate_reference.py"
echo "oracle fixtures generated outside checkout: $output_dir"
