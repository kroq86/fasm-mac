#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'usage: scripts/tensorctl-bench.sh OUTPUT.tsv [BASELINE.tsv]' >&2
  exit 2
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-bench-run.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
start="$(python3 -c 'import time; print(time.perf_counter_ns())')"
"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null
end="$(python3 -c 'import time; print(time.perf_counter_ns())')"
args=(--tensorctl "$OUT_DIR/tensorctl" --output "$1" --repeats "${TENSORCTL_BENCH_REPEATS:-7}" --compile-us "$(( (end - start) / 1000 ))" --threshold "${TENSORCTL_BENCH_THRESHOLD:-10}")
if [[ $# == 2 ]]; then args+=(--baseline "$2" --diff-output "${1%.tsv}.diff.tsv"); fi
python3 "$ROOT/fasm/tools/tensorctl_bench.py" "${args[@]}"
echo "tensorctl benchmark evidence written: $1"
