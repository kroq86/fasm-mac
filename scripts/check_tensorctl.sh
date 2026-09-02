#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null

capabilities="$(arch -x86_64 "$OUT_DIR/tensorctl" capabilities)"
for row in $'MatMul\tyes\tyes' $'Gemm\tyes\tyes' $'Add\tyes\tyes' $'Relu\tyes\tyes' \
           $'Conv\tyes\tyes' $'MaxPool\tyes\tyes' $'Reshape\tyes\tyes'; do
  if [[ "$(grep -Fxc "$row" <<<"$capabilities")" -ne 1 ]]; then
    printf 'FAIL: capability contract missing or duplicated row %q:\n%s\n' "$row" "$capabilities" >&2
    exit 1
  fi
done
if [[ "$(tail -n +3 <<<"$capabilities" | wc -l | tr -d ' ')" -ne 7 ]]; then
  printf 'FAIL: capability contract contains an unexpected op:\n%s\n' "$capabilities" >&2
  exit 1
fi

mlp_out="$(arch -x86_64 "$OUT_DIR/tensorctl" mlp)"
if ! grep -q "correct=4/4" <<<"$mlp_out"; then
  printf 'FAIL: tensorctl mlp did not converge to the XOR truth table:\n%s\n' "$mlp_out" >&2
  exit 1
fi

# The whole point of the memory report: it must actually be a decision, not a
# fixed string — verify it flips both ways, not just that it prints something.
# --epochs=1 won't converge (exit 1) and that's fine; only the memory report matters here.
generous="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --epochs=1 --memory-budget=8192 || true)"
tight="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --epochs=1 --memory-budget=4928 || true)"
if ! grep -q "decision=save" <<<"$generous"; then
  printf 'FAIL: generous budget did not choose save:\n%s\n' "$generous" >&2
  exit 1
fi
if ! grep -q "decision=rematerialize" <<<"$tight"; then
  printf 'FAIL: tight budget did not choose rematerialize:\n%s\n' "$tight" >&2
  exit 1
fi

train_out="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --epochs=12000)"
if ! grep -q "loss=2.178741->0.000000" <<<"$train_out"; then
  printf 'FAIL: tensorctl transformer training did not converge as expected:\n%s\n' "$train_out" >&2
  exit 1
fi
if ! grep -q "numerical_equivalence=yes merge_never_written=yes" <<<"$train_out"; then
  printf 'FAIL: layout-aware verification did not pass:\n%s\n' "$train_out" >&2
  exit 1
fi
# Whichever path the (data-driven, machine-dependent) layout decision picks,
# training must actually run through that same path — not print one decision
# and silently execute the other.
if grep -q "decision=layout-aware" <<<"$train_out"; then
  if ! grep -q "executed via layout-aware path, 20 backward actions" <<<"$train_out"; then
    printf 'FAIL: decision said layout-aware but training did not execute that path:\n%s\n' "$train_out" >&2
    exit 1
  fi
elif grep -q "decision=standard" <<<"$train_out"; then
  if ! grep -q "executed via standard path, 21 backward actions" <<<"$train_out"; then
    printf 'FAIL: decision said standard but training did not execute that path:\n%s\n' "$train_out" >&2
    exit 1
  fi
else
  printf 'FAIL: no layout decision printed:\n%s\n' "$train_out" >&2
  exit 1
fi

# --plan must exit immediately (no "training:"/"benchmark:" line) and print a
# buffer table with exactly the right row count, not one past the array
# bounds. Counting rows (via awk, LC_ALL=C so a stray non-UTF8 byte in a
# garbage row can't make grep silently treat the line as binary and skip
# it — that false negative is exactly what let this bug through once already).
count_buffer_rows() { LC_ALL=C awk '/^  buffer /{f=1;next} /^layout:/{f=0} f' <<<"$1" | LC_ALL=C wc -l | tr -d ' '; }

plan_remat="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan --memory-budget=4928)"
if grep -q "^training:" <<<"$plan_remat"; then
  printf 'FAIL: --plan ran training instead of exiting early:\n%s\n' "$plan_remat" >&2
  exit 1
fi
remat_rows="$(count_buffer_rows "$plan_remat")"
if [[ "$remat_rows" != 31 ]]; then
  printf 'FAIL: --plan (rematerialize variant) buffer table has %s rows, expected 31 — likely an out-of-bounds read\n' "$remat_rows" >&2
  exit 1
fi
plan_save="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan --memory-budget=8192)"
save_rows="$(count_buffer_rows "$plan_save")"
if [[ "$save_rows" != 30 ]]; then
  printf 'FAIL: --plan (save variant) buffer table has %s rows, expected 30 — likely an out-of-bounds read\n' "$save_rows" >&2
  exit 1
fi

mlp_plan="$(arch -x86_64 "$OUT_DIR/tensorctl" mlp --plan)"
if grep -q "^training:" <<<"$mlp_plan"; then
  printf 'FAIL: mlp --plan ran training instead of exiting early:\n%s\n' "$mlp_plan" >&2
  exit 1
fi

explained="$(XDG_CACHE_HOME="$OUT_DIR/explain-cache" arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan --memory-budget=5000 --counterfactual-budget=8192 --explain-decisions --no-profile-cache)"
if ! grep -q "chosen: rematerialize" <<<"$explained" || ! grep -q "counterfactual budget=8192 -> save" <<<"$explained"; then
  printf 'FAIL: memory explanation was not derived from the requested real/counterfactual budgets:\n%s\n' "$explained" >&2
  exit 1
fi
if grep -q "decision=layout-aware" <<<"$explained"; then
  grep -q "chosen: layout-aware" <<<"$explained" || { printf 'FAIL: layout trace disagrees with planner output\n' >&2; exit 1; }
else
  grep -q "chosen: standard" <<<"$explained" || { printf 'FAIL: layout trace disagrees with planner output\n' >&2; exit 1; }
fi
alternatives="$(XDG_CACHE_HOME="$OUT_DIR/explain-cache" arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan --show-alternatives)"
grep -q "decision: attention_scores" <<<"$alternatives" || { printf 'FAIL: --show-alternatives omitted memory candidates\n' >&2; exit 1; }
grep -q "decision: head_merge_layout" <<<"$alternatives" || { printf 'FAIL: --show-alternatives omitted layout candidates\n' >&2; exit 1; }

XDG_CACHE_HOME="$OUT_DIR/diff-cache" arch -x86_64 "$OUT_DIR/tensorctl" plan transformer --memory-budget=8192 --export "$OUT_DIR/plan-a.tsv" --no-profile-cache >/dev/null
XDG_CACHE_HOME="$OUT_DIR/diff-cache" arch -x86_64 "$OUT_DIR/tensorctl" plan transformer --memory-budget=5000 --export "$OUT_DIR/plan-b.tsv" --no-profile-cache >/dev/null
plan_diff="$(arch -x86_64 "$OUT_DIR/tensorctl" plan diff "$OUT_DIR/plan-a.tsv" "$OUT_DIR/plan-b.tsv")"
grep -q "attention_scores: save -> rematerialize" <<<"$plan_diff" || { printf 'FAIL: plan diff missed save/rematerialize flip\n' >&2; exit 1; }
grep -q "attention_scores.budget: 8192 -> 5000" <<<"$plan_diff" || { printf 'FAIL: plan diff missed budget change\n' >&2; exit 1; }
identical="$(arch -x86_64 "$OUT_DIR/tensorctl" plan diff "$OUT_DIR/plan-a.tsv" "$OUT_DIR/plan-a.tsv")"
[[ "$identical" == "no semantic changes" ]] || { printf 'FAIL: identical plan traces produced a diff\n' >&2; exit 1; }
machine="$(arch -x86_64 "$OUT_DIR/tensorctl" plan diff "$OUT_DIR/plan-a.tsv" "$OUT_DIR/plan-b.tsv" --machine)"
grep -q $'^kind\tsubject\tbefore\tafter$' <<<"$machine" || { printf 'FAIL: machine plan diff is not TSV\n' >&2; exit 1; }
printf 'corrupted trace\n' > "$OUT_DIR/corrupt.tsv"
if arch -x86_64 "$OUT_DIR/tensorctl" plan diff "$OUT_DIR/corrupt.tsv" "$OUT_DIR/plan-a.tsv" >/dev/null 2>&1; then
  printf 'FAIL: corrupted plan trace was accepted\n' >&2
  exit 1
fi

if arch -x86_64 "$OUT_DIR/tensorctl" >/dev/null 2>&1; then
  printf 'FAIL: tensorctl with no args should exit non-zero\n' >&2
  exit 1
fi

# Profile cache: sandboxed via XDG_CACHE_HOME (which the tool already
# prefers) so this never touches the real user cache directory.
export XDG_CACHE_HOME="$OUT_DIR/cache"

first="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan)"
if ! grep -q "source=measured" <<<"$first"; then
  printf 'FAIL: first-ever run should measure (no cache yet):\n%s\n' "$first" >&2
  exit 1
fi
second="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan)"
if ! grep -q "source=cached" <<<"$second"; then
  printf 'FAIL: second run should hit the cache written by the first:\n%s\n' "$second" >&2
  exit 1
fi
no_cache_run="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan --no-profile-cache)"
if ! grep -q "source=measured" <<<"$no_cache_run"; then
  printf 'FAIL: --no-profile-cache should measure even with a valid cache present:\n%s\n' "$no_cache_run" >&2
  exit 1
fi
still_cached="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan)"
if ! grep -q "source=cached" <<<"$still_cached"; then
  printf 'FAIL: --no-profile-cache should not have overwritten or consumed the cache:\n%s\n' "$still_cached" >&2
  exit 1
fi
reprofiled="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan --reprofile)"
if ! grep -q "source=measured" <<<"$reprofiled"; then
  printf 'FAIL: --reprofile should force a fresh measurement:\n%s\n' "$reprofiled" >&2
  exit 1
fi

# Corrupted/truncated cache files must never crash the tool — it's advisory,
# not correctness-bearing. Clobber every cache file found (hash-independent)
# and confirm a normal run still succeeds cleanly.
find "$XDG_CACHE_HOME/tensorctl" -name '*.cache' -exec sh -c 'printf "not a valid cache @#$%%" > "$1"' _ {} \;
if ! arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan >/dev/null 2>&1; then
  printf 'FAIL: a corrupted cache file should be ignored, not crash the tool\n' >&2
  exit 1
fi
truncated_out="$(arch -x86_64 "$OUT_DIR/tensorctl" transformer --plan)"
if ! grep -q "verified: numerical_equivalence=yes merge_never_written=yes" <<<"$truncated_out"; then
  printf 'FAIL: run after corrupted cache did not still verify correctly:\n%s\n' "$truncated_out" >&2
  exit 1
fi

printf 'tensorctl check passed: onnx_capability_contract=machine-readable mlp_converges=yes transformer_converges=yes memory_decision_flips=yes explain_and_alternatives=yes counterfactual=yes trace_execution_consistency=yes plan_diff=human+tsv+safe plan_mode_exits_early=yes plan_buffer_table_in_bounds=yes profile_cache_hit_and_miss=yes corrupted_cache_ignored=yes usage_exit_code=nonzero\n'
