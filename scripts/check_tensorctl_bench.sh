#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/fasm/tools/tensorctl_bench.py" --self-test
if [[ "${TENSORCTL_PERF:-0}" != 1 ]]; then
  echo 'tensorctl bench perf skipped: set TENSORCTL_PERF=1'
  exit 0
fi
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-bench.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
start="$(python3 -c 'import time; print(time.perf_counter_ns())')"
"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null
end="$(python3 -c 'import time; print(time.perf_counter_ns())')"
compile_us="$(( (end - start) / 1000 ))"
python3 "$ROOT/fasm/tools/tensorctl_bench.py" --tensorctl "$OUT_DIR/tensorctl" --output "$OUT_DIR/evidence.tsv" --repeats "${TENSORCTL_BENCH_REPEATS:-5}" --compile-us "$compile_us"
echo "tensorctl bench perf passed: evidence=$OUT_DIR/evidence.tsv (temporary)"
