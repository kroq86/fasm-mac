#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eml-sr-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

EML_SR="$OUT_DIR/eml_sr"
DERIVED_SMOKE="$OUT_DIR/derived_ops_smoke"
WITNESS_SMOKE="$OUT_DIR/witness_verify_smoke"
COMPILER_SMOKE="$OUT_DIR/eml_compiler_smoke"
ARENA_SMOKE="$OUT_DIR/expr_arena_smoke"
DOT_EXP="$OUT_DIR/tree_exp.dot"
DOT_POLY="$OUT_DIR/tree_poly.dot"
EXPECTED="$ROOT/fasm/tests/eml_sr/expected_verify.txt"
POLY_POINTS="$ROOT/fasm/tests/eml_sr/poly_points.txt"
EXP_POINTS="$ROOT/fasm/tests/eml_sr/exp_points.txt"

if ! command -v clang++ >/dev/null 2>&1; then
    echo 'FAIL clang++ is required for eml_sr check' >&2
    exit 1
fi

if [[ ! -x "$ROOT/bin/fasm" ]]; then
    echo 'FAIL bin/fasm is required for eml_core leaf check' >&2
    exit 1
fi

EML_CORE_OBJ="$OUT_DIR/eml_core.o"
EML_CORE_SMOKE="$OUT_DIR/eml_core_smoke"
"$ROOT/bin/fasm" --emit=macho-obj "$ROOT/fasm/apps/eml_core.asm" "$EML_CORE_OBJ" >/dev/null
clang++ -std=c++20 -O2 -arch x86_64 \
    "$ROOT/fasm/tests/eml_sr/eml_core_smoke.cpp" \
    "$EML_CORE_OBJ" \
    -lm \
    -o "$EML_CORE_SMOKE"
arch -x86_64 "$EML_CORE_SMOKE"

clang++ -std=c++20 -O2 -arch x86_64 "$ROOT/fasm/apps/eml_sr/eml_sr.cpp" -o "$EML_SR"
clang++ -std=c++20 -O2 -arch x86_64 "$ROOT/fasm/tests/eml_sr/derived_ops_smoke.cpp" -o "$DERIVED_SMOKE"
clang++ -std=c++20 -O2 -arch x86_64 "$ROOT/fasm/tests/eml_sr/witness_verify_smoke.cpp" -o "$WITNESS_SMOKE"
clang++ -std=c++20 -O2 -arch x86_64 "$ROOT/fasm/tests/eml_sr/eml_compiler_smoke.cpp" -o "$COMPILER_SMOKE"
clang++ -std=c++20 -O2 -arch x86_64 "$ROOT/fasm/tests/eml_sr/expr_arena_smoke.cpp" -o "$ARENA_SMOKE"
"$DERIVED_SMOKE" >/dev/null
"$WITNESS_SMOKE" >/dev/null
"$COMPILER_SMOKE" >/dev/null
"$ARENA_SMOKE" >/dev/null

verify_out="$("$EML_SR" verify)"
expected="$(tr -d '\r' < "$EXPECTED")"
if [[ "$verify_out" != "$expected" ]]; then
    printf 'FAIL eml_sr verify\nexpected: %q\nactual:   %q\n' "$expected" "$verify_out" >&2
    exit 1
fi

exp_out="$("$EML_SR" recover --target exp --dot "$DOT_EXP")"
if ! grep -q '^mse=0' <<< "$exp_out"; then
    echo "FAIL eml_sr recover --target exp (expected mse=0)" >&2
    echo "$exp_out" >&2
    exit 1
fi
if ! grep -q 'digraph EMLTree' "$DOT_EXP"; then
    echo "FAIL eml_sr recover --target exp --dot (invalid DOT)" >&2
    exit 1
fi

ln_show="$("$EML_SR" show --preset ln --dot-eval 2 --dot "$OUT_DIR/tree_ln.dot")"
if ! grep -q '^eml_nodes=3' <<< "$ln_show"; then
    echo "FAIL eml_sr show --preset ln (expected 3 eml nodes)" >&2
    echo "$ln_show" >&2
    exit 1
fi
if ! grep -q 'digraph EMLTree' "$OUT_DIR/tree_ln.dot"; then
    echo "FAIL eml_sr show --preset ln --dot" >&2
    exit 1
fi

if [[ -n "${CHECK_EML_SR_SKIP_POLY:-}" ]]; then
    echo 'skip recover poly (CHECK_EML_SR_SKIP_POLY set)'
else
poly_out="$("$EML_SR" recover --target poly --max-depth 4 --dot "$DOT_POLY")"
if ! grep -q '^mse=' <<< "$poly_out"; then
    echo "FAIL eml_sr recover --target poly" >&2
    echo "$poly_out" >&2
    exit 1
fi
poly_mse="$(grep '^mse=' <<< "$poly_out" | cut -d= -f2)"
if ! awk -v m="$poly_mse" 'BEGIN { exit !(m <= 0.2) }'; then
    echo "FAIL eml_sr recover --target poly (mse=$poly_mse > 0.2)" >&2
    exit 1
fi
if [[ ! -s "$DOT_POLY" ]] || ! grep -q 'digraph EMLTree' "$DOT_POLY"; then
    echo "FAIL eml_sr recover --target poly --dot (invalid DOT)" >&2
    exit 1
fi
fi

if command -v dot >/dev/null 2>&1; then
    dot -Gdpi=150 -Tpng "$DOT_EXP" -o "$OUT_DIR/tree_exp.png"
    dot -Gdpi=150 -Tpng "$OUT_DIR/tree_ln.dot" -o "$OUT_DIR/tree_ln.png"
    if [[ -z "${CHECK_EML_SR_SKIP_POLY:-}" ]]; then
        dot -Tpng "$DOT_POLY" -o "$OUT_DIR/tree_poly.png"
    fi
else
    echo 'skip graphviz dot (not installed)'
fi

bench_out="$(printf '1000 search 1.5\n10000 search 4.2\n' | "$EML_SR" fit-bench --max-depth 1 --points "$EXP_POINTS" --dot "$OUT_DIR/fit.dot")"
if ! grep -q '^baseline_mse=' <<< "$bench_out"; then
    echo "FAIL eml_sr fit-bench" >&2
    echo "$bench_out" >&2
    exit 1
fi
if ! grep -q '^latency_count=10000 latency_layer=search latency_ms=4.2' <<< "$bench_out"; then
    echo "FAIL eml_sr fit-bench bench_perf pipe" >&2
    echo "$bench_out" >&2
    exit 1
fi

echo 'eml_sr checks passed'
