#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/python"
VENV="$ROOT/python/.venv"

if ! command -v python3 >/dev/null 2>&1; then
    echo 'FAIL python3 is required for eml_sr python check' >&2
    exit 1
fi

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q -U pip
python -m pip install -q -e "$PY" pytest

python -m pytest "$PY/tests/test_fit_exp.py" -q -m "not slow"
echo 'eml_sr python checks passed (slow poly: pytest -m slow)'
