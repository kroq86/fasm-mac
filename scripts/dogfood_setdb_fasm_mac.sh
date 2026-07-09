#!/usr/bin/env bash
# Self-audit dogfood: use setdb to answer "which fasm-mac apps are
# brew-worthy" (have a check script, a release script, a Homebrew formula),
# which core helpers have zero app consumers, which benchmarks have no
# backend tag, and which docs don't name a known app.
#
# Every "X without Y" answer below is computed by setdb itself (`diff`
# against a `store-domain`/`store-range` result), not by the shell — real
# filesystem paths are used as atoms directly, since setdb_name_valid
# allows '/'.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/setdb-dogfood.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/setdb"
DB="$OUT_DIR/fasm-mac.db"
FASM_BIN="$ROOT/bin/fasm"
if [[ ! -x "$FASM_BIN" ]]; then
  FASM_BIN="${FASM:-fasm}"
fi

"$FASM_BIN" "$ROOT/fasm/apps/setdb.asm" "$BIN" >/dev/null

sdb() {
  arch -x86_64 "$BIN" "$@"
}

sdb new "$DB"

# --- discover apps: top-level fasm/apps/*.asm plus product subdirs,
#     excluding FFI-only cores that are never run directly ---
FFI_ONLY=(eml_core logvec_core)
TOP_APPS=()
while IFS= read -r line; do TOP_APPS+=("$line"); done < <(cd "$ROOT" && ls fasm/apps/*.asm | xargs -n1 basename | sed 's/\.asm$//')
SUBDIR_APPS=()
while IFS= read -r line; do SUBDIR_APPS+=("$line"); done < <(cd "$ROOT" && for d in fasm/apps/*/; do basename "$d"; done)

APPS=()
for a in "${TOP_APPS[@]}" "${SUBDIR_APPS[@]}"; do
  skip=0
  for f in "${FFI_ONLY[@]}"; do [[ "$a" == "$f" ]] && skip=1; done
  [[ $skip -eq 0 ]] && APPS+=("$a")
done

sdb add "$DB" apps "${APPS[@]}" >/dev/null

# --- has_check / has_release / has_formula (real paths as atoms) ---
for app in "${APPS[@]}"; do
  check_hit="$(cd "$ROOT" && ls scripts/check_"${app}"*.sh 2>/dev/null | head -1 || true)"
  if [[ -n "$check_hit" ]]; then
    sdb relation "$DB" has_check "$app" "$check_hit" >/dev/null
  fi
  if [[ -f "$ROOT/scripts/build-${app}-release.sh" ]]; then
    sdb relation "$DB" has_release "$app" "scripts/build-${app}-release.sh" >/dev/null
  fi
  if [[ -f "$ROOT/Formula/${app}.rb" ]]; then
    sdb relation "$DB" has_formula "$app" "Formula/${app}.rb" >/dev/null
  fi
done

# --- uses_core: only meaningful for top-level .asm apps ---
ALL_CORE=()
while IFS= read -r line; do ALL_CORE+=("$line"); done < <(cd "$ROOT" && ls fasm/core/*.inc | xargs -n1 basename)
sdb add "$DB" core_incs "${ALL_CORE[@]}" >/dev/null

for app in "${TOP_APPS[@]}"; do
  [[ "$app" == "eml_core" || "$app" == "logvec_core" ]] && continue
  asm="$ROOT/fasm/apps/${app}.asm"
  [[ -f "$asm" ]] || continue
  while read -r inc; do
    [[ -z "$inc" ]] && continue
    sdb relation "$DB" uses_core "$app" "$(basename "$inc")" >/dev/null
  done < <(grep -oE 'fasm/core/[a-zA-Z0-9_]+\.inc' "$asm" | sort -u)
done

# --- benchmarks -> backend (dogfooding tag/files/tags sugar for a
#     non-file entity, to see whether it generalizes) ---
ALL_BENCH=()
while IFS= read -r line; do ALL_BENCH+=("$line"); done < <(cd "$ROOT" && ls scripts/bench*.sh 2>/dev/null | xargs -n1 basename)
if [[ ${#ALL_BENCH[@]} -gt 0 ]]; then
  sdb add "$DB" all_benchmarks "${ALL_BENCH[@]}" >/dev/null
fi
if [[ -f "$ROOT/scripts/bench_logvec.sh" ]]; then
  sdb tag "$DB" bench_logvec.sh avx2 scalar x86_64 >/dev/null
fi
if [[ -f "$ROOT/scripts/bench_perf.sh" ]]; then
  sdb tag "$DB" bench_perf.sh x86_64 >/dev/null
fi
if [[ -f "$ROOT/scripts/bench_eml_sr_compare.sh" ]]; then
  sdb tag "$DB" bench_eml_sr_compare.sh x86_64 >/dev/null
fi

# --- docs -> app, by filename substring match (heuristic, but checkable) ---
DOCS=()
while IFS= read -r line; do DOCS+=("$line"); done < <(cd "$ROOT" && ls docs/*.md 2>/dev/null | xargs -n1 basename)
if [[ ${#DOCS[@]} -gt 0 ]]; then
  sdb add "$DB" all_docs "${DOCS[@]}" >/dev/null
fi
for doc in "${DOCS[@]}"; do
  for app in "${APPS[@]}"; do
    if [[ "$doc" == *"$app"* ]]; then
      sdb relation "$DB" doc_covers "$doc" "$app" >/dev/null
    fi
  done
done

# --- materialize the query results this report needs, entirely in setdb ---
sdb store-domain "$DB" has_check checked_apps >/dev/null
sdb store-domain "$DB" has_release released_apps >/dev/null
sdb store-domain "$DB" has_formula formula_apps >/dev/null
sdb store-range "$DB" uses_core used_core >/dev/null
sdb store-domain "$DB" doc_covers covered_docs >/dev/null

# --- report: every diff below runs inside setdb, not the shell ---
echo "=== fasm-mac product readiness (via setdb) ==="
echo

echo "-- apps without a check script --"
sdb diff "$DB" apps checked_apps
echo
echo "-- apps without a release script --"
sdb diff "$DB" apps released_apps
echo
echo "-- apps without a Homebrew formula --"
sdb diff "$DB" apps formula_apps
echo

echo "-- core helpers (recorded via uses_core) with zero app consumers --"
sdb diff "$DB" core_incs used_core
echo

echo "-- benchmarks without any backend tag --"
if [[ ${#ALL_BENCH[@]} -gt 0 ]]; then
  sdb diff "$DB" all_benchmarks files
fi
echo

echo "-- docs without a matching app name --"
if [[ ${#DOCS[@]} -gt 0 ]]; then
  sdb diff "$DB" all_docs covered_docs
fi
echo

echo "=== raw setdb sets/relations snapshot ==="
sdb sets "$DB"
echo "---"
sdb relations "$DB"
