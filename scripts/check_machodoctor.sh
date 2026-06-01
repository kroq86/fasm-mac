#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/machodoctor-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/machodoctor"
TARGET="$OUT_DIR/fscan"
BAD="$OUT_DIR/not-macho.txt"

fasm "$ROOT/fasm/apps/machodoctor.asm" "$BIN" >/dev/null
fasm "$ROOT/fasm/apps/fscan.asm" "$TARGET" >/dev/null
printf 'hello\n' > "$BAD"

human="$(arch -x86_64 "$BIN" "$TARGET")"
grep -q 'format: Mach-O 64-bit' <<<"$human"
grep -q 'arch: x86_64' <<<"$human"
grep -q 'type: executable' <<<"$human"
grep -q 'min macOS: 11.0' <<<"$human"

json="$(arch -x86_64 "$BIN" --json "$TARGET")"
grep -q '"format": "Mach-O 64-bit"' <<<"$json"
grep -q '"arch": "x86_64"' <<<"$json"
grep -q '"type": "executable"' <<<"$json"

deps="$(arch -x86_64 "$BIN" --deps "$TARGET")"
grep -q 'dylibs:' <<<"$deps"
grep -q 'rpaths:' <<<"$deps"

if [[ -x /bin/ls ]]; then
  system_out="$(arch -x86_64 "$BIN" /bin/ls)"
  grep -q 'format: Mach-O 64-bit' <<<"$system_out"
  grep -q 'type: executable' <<<"$system_out"
fi

if arch -x86_64 "$BIN" --check "$TARGET" >/dev/null; then
  echo 'FAIL --check should warn for unsigned fasm binary' >&2
  exit 1
fi

if arch -x86_64 "$BIN" "$BAD" >/dev/null 2>"$OUT_DIR/bad.err"; then
  echo 'FAIL non-Mach-O file should fail' >&2
  exit 1
fi
grep -q 'not a supported Mach-O' "$OUT_DIR/bad.err"

if arch -x86_64 "$BIN" "$OUT_DIR/missing" >/dev/null 2>"$OUT_DIR/missing.err"; then
  echo 'FAIL missing file should fail' >&2
  exit 1
fi
grep -q 'cannot open' "$OUT_DIR/missing.err"

echo 'machodoctor checks passed'
