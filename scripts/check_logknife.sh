#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/logknife-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/logknife"
LOG="$OUT_DIR/app.jsonl"
PLAIN="$OUT_DIR/plain.log"

fasm "$ROOT/fasm/apps/logknife.asm" "$BIN" >/dev/null

cat > "$LOG" <<'EOF'
{"ts":"1","level":"info","status":200,"service":"api","msg":"ok"}
{"ts":"2","level":"error","status":500,"service":"api","msg":"timeout"}
{"ts":"3","level":"error","status":404,"service":"web","msg":"missing"}
EOF

cat > "$PLAIN" <<'EOF'
plain info
plain timeout
EOF

expected='{"ts":"2","level":"error","status":500,"service":"api","msg":"timeout"}'
actual="$(arch -x86_64 "$BIN" --jsonl --field status=500 "$LOG")"
if [[ "$actual" != "$expected" ]]; then
  printf 'FAIL logknife field\nexpected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
  exit 1
fi

actual="$(arch -x86_64 "$BIN" --jsonl --level error --count "$LOG")"
if [[ "$actual" != '2' ]]; then
  printf 'FAIL logknife level count expected 2 got %q\n' "$actual" >&2
  exit 1
fi

actual="$(arch -x86_64 "$BIN" --contains timeout "$PLAIN")"
if [[ "$actual" != 'plain timeout' ]]; then
  printf 'FAIL logknife contains expected plain timeout got %q\n' "$actual" >&2
  exit 1
fi

if arch -x86_64 "$BIN" --contains absent "$PLAIN" >/dev/null; then
  echo 'FAIL logknife absent should exit 1' >&2
  exit 1
fi

if arch -x86_64 "$BIN" --contains x "$OUT_DIR/missing.log" >/dev/null 2>"$OUT_DIR/err"; then
  echo 'FAIL logknife missing file should exit 2' >&2
  exit 1
fi
grep -q 'cannot open' "$OUT_DIR/err"

echo 'logknife checks passed'
