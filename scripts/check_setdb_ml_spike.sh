#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/setdb-ml.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
for src in tensor_matmul_f32 tensor_tape_f32 tensor_ops_f32 tensor_tape_exec_f32 tensor_sgd_f32 tensor_plan_f32 tensor_fused_f32; do
  fasm --emit=macho-obj "$ROOT/fasm/spikes/$src.asm" "$OUT_DIR/$src.o" >/dev/null
done
clang -arch x86_64 -O2 -c "$ROOT/fasm/spikes/tensor_plan_fused_compile.c" -o "$OUT_DIR/tensor_plan_fused_compile.o"
clang -arch x86_64 -O2 "$ROOT/fasm/spikes/setdb_ml_projection.c" "$OUT_DIR"/tensor_*.o -o "$OUT_DIR/project"
actual="$(arch -x86_64 "$OUT_DIR/project" "$ROOT/fasm/spikes/setdb_ml_fixture.setdb")"
expected=$'RADD ml/class/v1 eta safe\nRADD ml/class/v1 theta risky'
[[ "$actual" == "$expected" ]] || { printf 'FAIL setdb ML projection\n%s\n' "$actual" >&2; exit 1; }
printf '%s\n' "$actual" >"$OUT_DIR/predictions.setdb"
FASM_BIN="$ROOT/bin/fasm"; [[ -x "$FASM_BIN" ]] || FASM_BIN="${FASM:-fasm}"
"$FASM_BIN" "$ROOT/fasm/apps/setdb.asm" "$OUT_DIR/setdb" >/dev/null
arch -x86_64 "$OUT_DIR/setdb" new "$OUT_DIR/model.db"
arch -x86_64 "$OUT_DIR/setdb" load "$OUT_DIR/model.db" "$ROOT/fasm/spikes/setdb_ml_fixture.setdb"
arch -x86_64 "$OUT_DIR/setdb" load "$OUT_DIR/model.db" "$OUT_DIR/predictions.setdb"
loaded="$(arch -x86_64 "$OUT_DIR/setdb" pairs "$OUT_DIR/model.db" ml/class/v1)"
[[ "$loaded" == $'(eta,safe)\n(theta,risky)' ]] || { printf 'FAIL materialized predictions\n%s\n' "$loaded" >&2; exit 1; }
printf '%s\n' "$actual"
