#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$ROOT/fasm/spikes/nir_mfco_baseline.py" --self-test

if [[ -n "${NIR_MFCO_ARCHIVE:-}" ]]; then
  OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nir-mfco-check.XXXXXX")"
  trap 'rm -rf "$OUT_DIR"' EXIT
  python3 "$ROOT/fasm/spikes/nir_mfco_baseline.py" \
    --verify-md5 --export "$OUT_DIR/fixture.bin" "$NIR_MFCO_ARCHIVE"
  "$ROOT/scripts/build-nir-mfco-tensor.sh" "$OUT_DIR/train" >/dev/null
  arch -x86_64 "$OUT_DIR/train" "$OUT_DIR/fixture.bin"
else
  printf '%s\n' 'nir-mfco dataset check skipped: set NIR_MFCO_ARCHIVE to the external zip'
fi
