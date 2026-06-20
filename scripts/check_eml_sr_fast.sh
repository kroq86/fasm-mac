#!/usr/bin/env bash
# Fast eml_sr smoke (~5s). Skips recover poly (depth 4 search ~30s).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eml-sr-fast.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

EML_SR="$OUT_DIR/eml_sr"
DERIVED_SMOKE="$OUT_DIR/derived_ops_smoke"
WITNESS_SMOKE="$OUT_DIR/witness_verify_smoke"
COMPILER_SMOKE="$OUT_DIR/eml_compiler_smoke"
ARENA_SMOKE="$OUT_DIR/expr_arena_smoke"
EXP_POINTS="$ROOT/fasm/tests/eml_sr/exp_points.txt"
EXPECTED="$ROOT/fasm/tests/eml_sr/expected_verify.txt"

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

exp_out="$("$EML_SR" recover --target exp --dot "$OUT_DIR/tree_exp.dot")"
if ! grep -q '^mse=0' <<< "$exp_out"; then
    echo "FAIL eml_sr recover --target exp" >&2
    echo "$exp_out" >&2
    exit 1
fi

adam_out="$("$EML_SR" recover --target exp --method adam --max-depth 1)"
if ! grep -q '^mse=0' <<< "$adam_out"; then
    echo "FAIL eml_sr recover --target exp --method adam" >&2
    echo "$adam_out" >&2
    exit 1
fi

if "$EML_SR" recover --target exp --method nope >/dev/null 2>&1; then
    echo "FAIL eml_sr recover accepted invalid --method" >&2
    exit 1
fi
if "$EML_SR" recover --target exp --domain nope >/dev/null 2>&1; then
    echo "FAIL eml_sr recover accepted invalid --domain" >&2
    exit 1
fi

ln_show="$("$EML_SR" show --preset ln --dot-eval 2 --dot "$OUT_DIR/tree_ln.dot")"
if ! grep -q '^eml_nodes=3' <<< "$ln_show"; then
    echo "FAIL eml_sr show --preset ln" >&2
    echo "$ln_show" >&2
    exit 1
fi

bench_out="$(printf '1000 search 1.5\n' | "$EML_SR" fit-bench --max-depth 1 --profile --points "$EXP_POINTS")"
if ! grep -q '^baseline_mse=' <<< "$bench_out"; then
    echo "FAIL eml_sr fit-bench" >&2
    echo "$bench_out" >&2
    exit 1
fi
if ! grep -q '^forms=' <<< "$bench_out"; then
    echo "FAIL eml_sr fit-bench --profile (missing stats)" >&2
    echo "$bench_out" >&2
    exit 1
fi

echo 'eml_sr fast checks passed (full: scripts/check_eml_sr.sh, includes poly ~30s)'
