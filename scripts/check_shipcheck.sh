#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/shipcheck-check.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

SHIPCHECK="$OUT_DIR/shipcheck"
HEXPEEK_BIN="$OUT_DIR/hexpeek"
PKG="$ROOT/dist/hexpeek-0.1.0-macos-x86_64.tar.gz"
GOOD_FORMULA="$OUT_DIR/hexpeek.rb"
BAD_SHA_FORMULA="$OUT_DIR/hexpeek_bad_sha.rb"
BAD_NAME_BIN="$OUT_DIR/not-hexpeek"
TEXT_BIN="$OUT_DIR/not-macho"

fasm "$ROOT/fasm/apps/shipcheck.asm" "$SHIPCHECK" >/dev/null
fasm "$ROOT/fasm/apps/hexpeek.asm" "$HEXPEEK_BIN" >/dev/null
"$ROOT/scripts/build-hexpeek-release.sh" 0.1.0 >/dev/null

SHA="$(shasum -a 256 "$PKG" | awk '{print $1}')"

cat > "$GOOD_FORMULA" <<EOF
class Hexpeek < Formula
  desc "fixture"
  homepage "https://github.com/kroq86/fasm-mac"
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/hexpeek-0.1.0-macos-x86_64.tar.gz"
  sha256 "$SHA"
  version "0.1.0"

  def install
    bin.install "hexpeek"
  end
end
EOF

cat > "$BAD_SHA_FORMULA" <<EOF
class Hexpeek < Formula
  url "https://github.com/kroq86/fasm-mac/releases/download/v0.1.0/hexpeek-0.1.0-macos-x86_64.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  version "0.1.0"
  def install
    bin.install "hexpeek"
  end
end
EOF

if ! tar -tzf "$PKG" | grep -q '/hexpeek$'; then
  echo 'FAIL shipcheck fixture tarball does not contain hexpeek' >&2
  exit 1
fi

actual="$(arch -x86_64 "$SHIPCHECK" "$GOOD_FORMULA" "$PKG" "$HEXPEEK_BIN")"
if ! grep -q '^status: pass$' <<<"$actual"; then
  printf 'FAIL shipcheck positive\n%s\n' "$actual" >&2
  exit 1
fi

if arch -x86_64 "$SHIPCHECK" "$BAD_SHA_FORMULA" "$PKG" "$HEXPEEK_BIN" >/dev/null 2>"$OUT_DIR/bad_sha.err"; then
  echo 'FAIL shipcheck bad sha should exit 1' >&2
  exit 1
fi
grep -q 'sha256 mismatch' "$OUT_DIR/bad_sha.err"

if arch -x86_64 "$SHIPCHECK" "$GOOD_FORMULA" "$OUT_DIR/missing.tar.gz" "$HEXPEEK_BIN" >/dev/null 2>"$OUT_DIR/missing.err"; then
  echo 'FAIL shipcheck missing package should exit 2' >&2
  exit 1
fi
grep -q 'cannot read package' "$OUT_DIR/missing.err"

printf 'not macho\n' > "$TEXT_BIN"
if arch -x86_64 "$SHIPCHECK" "$GOOD_FORMULA" "$PKG" "$TEXT_BIN" >/dev/null 2>"$OUT_DIR/text.err"; then
  echo 'FAIL shipcheck non-Mach-O should exit 1' >&2
  exit 1
fi
grep -q 'not a supported Mach-O' "$OUT_DIR/text.err"

cp "$HEXPEEK_BIN" "$BAD_NAME_BIN"
if arch -x86_64 "$SHIPCHECK" "$GOOD_FORMULA" "$PKG" "$BAD_NAME_BIN" >/dev/null 2>"$OUT_DIR/name.err"; then
  echo 'FAIL shipcheck mismatched binary name should exit 1' >&2
  exit 1
fi
grep -q 'binary basename does not match' "$OUT_DIR/name.err"

echo 'shipcheck checks passed'
