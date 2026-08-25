#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

checks=(
  check_tensor_lifetime_spike.sh
  check_tensor_arena_spike.sh
  check_tensor_tape_spike.sh
  check_tensor_matmul_spike.sh
  check_tensor_bias_spike.sh
  check_tensor_pipeline_spike.sh
  check_tensor_plan_spike.sh
  check_tensor_fusion_spike.sh
  check_tensor_compiler_planner_spike.sh
  check_tensor_grad_prune_spike.sh
  check_tensor_liveness_spike.sh
  check_tensor_sgd_spike.sh
  check_tensor_simd_spike.sh
  check_tensor_volume_stress.sh
  check_autograd_f32.sh
  check_mlp_f32.sh
  check_xor_tensor.sh
  check_digits_tensor.sh
  check_setdb_ml_spike.sh
)

for check in "${checks[@]}"; do
  "$ROOT/scripts/$check"
done

git -C "$ROOT" diff --check
echo 'tensor runtime spike regression gate passed'
