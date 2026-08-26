#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-report.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT
"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null
python3 "$ROOT/fasm/tools/tensorctl_report.py" transformer --tensorctl "$OUT_DIR/tensorctl" "$@"
