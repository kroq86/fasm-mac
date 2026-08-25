#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cc -std=c11 -Wall -Wextra -Werror "$ROOT/route_result.c" "$ROOT/route_result_test.c" -o /tmp/route-result-test
/tmp/route-result-test
