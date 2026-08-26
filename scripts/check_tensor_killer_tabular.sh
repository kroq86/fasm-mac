#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WINE_DATA:?set WINE_DATA to the UCI wine.data file (archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/tensor-killer-tabular.XXXXXX")"; trap 'rm -rf "$OUT"' EXIT
if [[ -n "${KILLER_PYTHON:-}" ]]; then
  PYTHON="$KILLER_PYTHON"
else
  command -v uv >/dev/null || { echo "uv is required (or set KILLER_PYTHON)" >&2; exit 2; }
  uv venv --python 3.12 "$OUT/venv" >/dev/null
  uv pip install --python "$OUT/venv/bin/python" numpy==2.5.2 onnx==1.22.0 onnxruntime==1.29.0 tinygrad==0.14.0 >/dev/null
  PYTHON="$OUT/venv/bin/python"
fi
clang -arch arm64 -O3 -Wall -Wextra -Werror "$ROOT/fasm/spikes/tensor_killer_tabular_native.c" -o "$OUT/native"
"$PYTHON" "$ROOT/scripts/tensor_killer_tabular.py" prepare --wine "$WINE_DATA" --out "$OUT/model"
"$PYTHON" "$ROOT/scripts/tensor_killer_compare.py" --native "$OUT/native" --python "$PYTHON" --runner "$ROOT/scripts/tensor_killer_tabular.py" --model "$OUT/model"
