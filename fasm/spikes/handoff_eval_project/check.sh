#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cc -std=c11 -Wall -Wextra -Werror "$ROOT/retry_policy.c" "$ROOT/retry_policy_test.c" -o /tmp/retry-policy-test
/tmp/retry-policy-test
