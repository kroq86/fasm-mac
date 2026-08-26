#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/fasm/tools/tensorctl_report.py" --self-test
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-report-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null
python3 "$ROOT/fasm/tools/tensorctl_report.py" transformer --tensorctl "$OUT_DIR/tensorctl" --memory-budget=5000 --counterfactual-budget=8192 --human "$OUT_DIR/report.txt" --machine "$OUT_DIR/report.tsv"
grep -q 'decision: attention_scores' "$OUT_DIR/report.txt"
grep -q 'counterfactual: save' "$OUT_DIR/report.txt"
grep -q 'end-to-end effect: unavailable' "$OUT_DIR/report.txt"
grep -q $'^decision\tchosen\tcounterfactual' "$OUT_DIR/report.tsv"
echo 'tensorctl report gate passed: human+tsv counterfactual=yes unavailable-not-invented=yes'
