#!/usr/bin/env bash
# Self-audit dogfood: use setdb against data/fasm_mac_readiness.setdb to
# answer "which fasm-mac apps are brew-worthy" (check/release/formula),
# which core headers have zero app consumers, which benchmarks/docs are
# unlinked. All of it is data/facts -> setdb load -> store-domain/range ->
# diff -> report, no per-fact shell invocations and no shell-level set math.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/setdb-dogfood.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/setdb"
DB="$OUT_DIR/fasm-mac.db"
FACTS="$ROOT/data/fasm_mac_readiness.setdb"
FASM_BIN="$ROOT/bin/fasm"
if [[ ! -x "$FASM_BIN" ]]; then
  FASM_BIN="${FASM:-fasm}"
fi

"$FASM_BIN" "$ROOT/fasm/apps/setdb.asm" "$BIN" >/dev/null

sdb() {
  arch -x86_64 "$BIN" "$@"
}

sdb new "$DB"
sdb load "$DB" "$FACTS"

sdb store-domain "$DB" has_check checked_apps >/dev/null
sdb store-domain "$DB" has_release released_apps >/dev/null
sdb store-domain "$DB" has_formula formula_apps >/dev/null
sdb store-range "$DB" uses_core used_core >/dev/null
sdb store-domain "$DB" doc_covers covered_docs >/dev/null

echo "== Apps without checks =="
sdb diff "$DB" apps checked_apps
echo
echo "== Apps without a release script =="
sdb diff "$DB" apps released_apps
echo
echo "== Apps without a formula =="
sdb diff "$DB" apps formula_apps
echo
echo "== Core files without known consumers =="
sdb diff "$DB" core_incs used_core
echo
echo "== Benchmarks without a backend mapping =="
sdb diff "$DB" all_benchmarks files
echo
echo "== Docs without an app mapping =="
sdb diff "$DB" all_docs covered_docs
