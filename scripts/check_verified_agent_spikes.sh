#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/check_tensor_agent_transaction_spike.sh"
git -C "$ROOT" diff --check
echo "verified agent spike gate passed"
