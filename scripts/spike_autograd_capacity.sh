#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo 'usage: spike_autograd_capacity.sh INPUTS OUTPUTS' >&2
  exit 2
fi

inputs="$1"
outputs="$2"
if [[ ! "$inputs" =~ ^[1-9][0-9]*$ || ! "$outputs" =~ ^[1-9][0-9]*$ ]]; then
  echo 'dimensions must be positive decimal integers' >&2
  exit 2
fi

nodes=$((2 * inputs * outputs + outputs))
bytes=$((nodes * 32))
printf 'dense=%sx%s scalar_op_nodes=%s tape_bytes=%s\n' \
  "$inputs" "$outputs" "$nodes" "$bytes"
