#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/setdb-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/setdb"
DB="$OUT_DIR/universe.db"
FASM_BIN="$ROOT/bin/fasm"
if [[ ! -x "$FASM_BIN" ]]; then
  FASM_BIN="${FASM:-fasm}"
fi

"$FASM_BIN" "$ROOT/fasm/apps/setdb.asm" "$BIN" >/dev/null

run() {
  arch -x86_64 "$BIN" "$@"
}

expect() {
  local label="$1"
  local expected="$2"
  shift 2
  local actual
  actual="$(run "$@")"
  if [[ "$actual" != "$expected" ]]; then
    printf 'FAIL %s\nexpected:\n%s\nactual:\n%s\n' "$label" "$expected" "$actual" >&2
    exit 1
  fi
}

help_out="$(run help)"
for word in new add remove relation unrelation members member union intersect diff subset select join domain range inverse rdiff runion rintersect transitive-closure; do
  if ! grep -q "  $word" <<< "$help_out"; then
    echo "FAIL help missing command: $word" >&2
    exit 1
  fi
done
if [[ "$(run --help)" != "$help_out" ]]; then
  echo 'FAIL --help should match help' >&2
  exit 1
fi
if [[ "$(run -h)" != "$help_out" ]]; then
  echo 'FAIL -h should match help' >&2
  exit 1
fi
if run >/dev/null 2>"$OUT_DIR/noargs.err"; then
  echo 'FAIL no-arg invocation should exit non-zero' >&2
  exit 1
fi
grep -q '^Usage:' "$OUT_DIR/noargs.err"

run new "$DB"
run add "$DB" users carol alice bob alice
run add "$DB" admins alice
run relation "$DB" follows alice bob
run relation "$DB" follows bob carol
run relation "$DB" follows carol dana
run relation "$DB" follows alice bob
run relation "$DB" blocked bob carol

expect members $'alice\nbob\ncarol' members "$DB" users
expect member-yes 'true' member "$DB" users alice
expect member-no 'false' member "$DB" users dave
expect union $'alice\nbob\ncarol' union "$DB" users admins
expect intersect 'alice' intersect "$DB" users admins
expect diff $'bob\ncarol' diff "$DB" users admins
expect subset-yes 'true' subset "$DB" admins users
expect subset-no 'false' subset "$DB" users admins
expect select-first 'bob' select "$DB" follows first alice
expect select-second 'bob' select "$DB" follows second carol
expect join $'(alice,carol)\n(bob,dana)' join "$DB" follows follows
expect domain $'alice\nbob\ncarol' domain "$DB" follows
expect range $'bob\ncarol\ndana' range "$DB" follows
expect inverse $'(bob,alice)\n(carol,bob)\n(dana,carol)' inverse "$DB" follows
expect rdiff $'(alice,bob)\n(carol,dana)' rdiff "$DB" follows blocked
expect runion $'(alice,bob)\n(bob,carol)\n(carol,dana)' runion "$DB" follows blocked
expect rintersect '(bob,carol)' rintersect "$DB" follows blocked
expect transitive-closure $'(alice,bob)\n(alice,carol)\n(alice,dana)\n(bob,carol)\n(bob,dana)\n(carol,dana)' transitive-closure "$DB" follows

run remove "$DB" users carol
expect remove $'alice\nbob' members "$DB" users
run unrelation "$DB" follows bob carol
expect unrelation '' rintersect "$DB" follows blocked

if run add "$DB" 'bad/name' alice >/dev/null 2>"$OUT_DIR/bad.err"; then
  echo 'FAIL invalid set name should exit non-zero' >&2
  exit 1
fi
if run relation "$DB" follows only-one >/dev/null 2>"$OUT_DIR/arity.err"; then
  echo 'FAIL invalid relation arity should exit non-zero' >&2
  exit 1
fi

echo 'setdb checks passed'
