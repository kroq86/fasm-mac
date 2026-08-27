#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensor-corpus-e2e.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null
output="$(XDG_CACHE_HOME="$OUT_DIR/cache" arch -x86_64 "$OUT_DIR/tensorctl" corpus "$ROOT/fasm/tests/ragbox/fixtures/tiny-repo")"
printf '%s\n' "$output"
accuracy="$(sed -n 's/.* accuracy=[0-9.]*->\([0-9.]*\) .*/\1/p' <<<"$output")"
awk -v value="$accuracy" 'BEGIN { exit !(value >= 0.70) }'
grep -q 'executor=x86_64-assembly' <<<"$output"
printf '%s\n' 'tensor corpus e2e check passed'
