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
for word in new add remove relation unrelation members member union intersect diff subset select join domain range inverse rdiff runion rintersect transitive-closure sets relations contains pairs tag files tags store-domain store-range store-inverse load dump; do
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
expect sets $'admins\nusers' sets "$DB"
expect relations $'blocked\nfollows' relations "$DB"
expect pairs $'(alice,bob)\n(bob,carol)\n(carol,dana)' pairs "$DB" follows
expect pairs-missing-rel '' pairs "$DB" nosuchrel
expect contains-alice $'admins\nusers' contains "$DB" alice
expect contains-bob 'users' contains "$DB" bob
expect contains-missing '' contains "$DB" nobody

run store-domain "$DB" follows follows_domain
expect store-domain $'alice\nbob\ncarol' members "$DB" follows_domain
run store-range "$DB" follows follows_range
expect store-range $'bob\ncarol\ndana' members "$DB" follows_range
run store-inverse "$DB" follows follows_inv
expect store-inverse $'(bob,alice)\n(carol,bob)\n(dana,carol)' pairs "$DB" follows_inv

# The materialize-then-diff workflow this exists for: a plain `diff` against
# a stored query result, entirely inside setdb, no shell-level set math.
run add "$DB" all_people alice bob carol dana eve
expect diff-after-store $'dana\neve' diff "$DB" all_people follows_domain

# load: bulk-apply SADD/SREM/RADD/RREM lines from a facts file in one
# invocation, instead of one setdb process per fact.
FACTS="$OUT_DIR/facts.setdb"
cat >"$FACTS" <<'FACTSEOF'
# comment and blank lines are ignored

SADD load_apps setdb
SADD load_apps logvec
RADD load_has_check setdb scripts/check_setdb.sh
FACTSEOF
LOAD_DB="$OUT_DIR/load.db"
run new "$LOAD_DB"
run load "$LOAD_DB" "$FACTS"
expect load-set $'logvec\nsetdb' members "$LOAD_DB" load_apps
expect load-rel '(setdb,scripts/check_setdb.sh)' pairs "$LOAD_DB" load_has_check

# dump: round trip through dump -> load must reproduce the same state.
DUMPED="$OUT_DIR/dumped.setdb"
run dump "$LOAD_DB" >"$DUMPED"
LOAD_DB2="$OUT_DIR/load2.db"
run new "$LOAD_DB2"
run load "$LOAD_DB2" "$DUMPED"
if [[ "$(run dump "$LOAD_DB" | sort)" != "$(run dump "$LOAD_DB2" | sort)" ]]; then
  echo 'FAIL dump/load round trip mismatch' >&2
  exit 1
fi

# a bad line stops the load but does not roll back lines already applied
BAD_FACTS="$OUT_DIR/bad_facts.setdb"
cat >"$BAD_FACTS" <<'BADEOF'
SADD partial_set atom1
BOGUS not a real opcode
SADD partial_set atom2
BADEOF
BAD_DB="$OUT_DIR/bad.db"
run new "$BAD_DB"
if run load "$BAD_DB" "$BAD_FACTS" >/dev/null 2>"$OUT_DIR/load-bad.err"; then
  echo 'FAIL load with a bad line should exit non-zero' >&2
  exit 1
fi
expect load-partial 'atom1' members "$BAD_DB" partial_set

EMPTY_DB="$OUT_DIR/empty.db"
run new "$EMPTY_DB"
expect sets-empty '' sets "$EMPTY_DB"
expect relations-empty '' relations "$EMPTY_DB"
expect dump-empty '' dump "$EMPTY_DB"

run tag "$DB" song1.mp3 music jazz
run tag "$DB" song2.mp3 music
expect files-music $'song1.mp3\nsong2.mp3' files "$DB" music
expect files-jazz 'song1.mp3' files "$DB" jazz
expect files-missing-tag '' files "$DB" nosuchtag
expect tags-song1 $'jazz\nmusic' tags "$DB" song1.mp3
expect tags-song2 'music' tags "$DB" song2.mp3
expect pairs-has-tag $'(song1.mp3,jazz)\n(song1.mp3,music)\n(song2.mp3,music)' pairs "$DB" has_tag
run tag "$DB" song1.mp3 jazz
expect tag-idempotent 'song1.mp3' files "$DB" jazz

run remove "$DB" users carol
expect remove $'alice\nbob' members "$DB" users
run unrelation "$DB" follows bob carol
expect unrelation '' rintersect "$DB" follows blocked

if run add "$DB" 'bad name' alice >/dev/null 2>"$OUT_DIR/bad.err"; then
  echo 'FAIL invalid set name should exit non-zero' >&2
  exit 1
fi
run add "$DB" 'path/like/name' alice
expect slash-in-name 'alice' members "$DB" 'path/like/name'
if run relation "$DB" follows only-one >/dev/null 2>"$OUT_DIR/arity.err"; then
  echo 'FAIL invalid relation arity should exit non-zero' >&2
  exit 1
fi
if run tag "$DB" song3.mp3 >/dev/null 2>"$OUT_DIR/tag-arity.err"; then
  echo 'FAIL tag with no tags should exit non-zero' >&2
  exit 1
fi
if run store-domain "$DB" follows >/dev/null 2>"$OUT_DIR/store-arity.err"; then
  echo 'FAIL store-domain with no target name should exit non-zero' >&2
  exit 1
fi
if run load "$DB" "$OUT_DIR/no-such-facts-file.setdb" >/dev/null 2>"$OUT_DIR/load-missing.err"; then
  echo 'FAIL load with a missing facts file should exit non-zero' >&2
  exit 1
fi

echo 'setdb checks passed'
