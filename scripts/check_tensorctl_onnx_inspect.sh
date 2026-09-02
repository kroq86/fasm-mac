#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WINE_DATA:?set WINE_DATA to the UCI wine.data file}"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-onnx-inspect.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

# Assertions below check field VALUES (shape/dtype/tolerance/status), not
# exact full-line wording — inspect/build/verify's own prose around those
# values is free to change without silently breaking this gate. This
# script broke twice already on cosmetic rewording of unrelated text next
# to the actual fact being asserted (LC_UUID phrasing, verify's field
# names) while the underlying behavior was already correct; extracting the
# value first is what "match the current public output format" means here.
field() { grep -oE "(^| )$1=[^ ]*" <<<"$2" | head -1 | sed -E "s/^ *$1=//"; }
require() { if ! eval "$1"; then printf 'FAILED: %s\n  context: %s\n' "$1" "$2" >&2; exit 1; fi; }

uv venv --python 3.13 "$OUT_DIR/venv" >/dev/null
uv pip install --python "$OUT_DIR/venv/bin/python" numpy==2.5.2 onnx==1.22.0 onnxruntime==1.29.0 >/dev/null
"$OUT_DIR/venv/bin/python" "$ROOT/scripts/tensor_killer_tabular.py" prepare \
  --wine "$WINE_DATA" --out "$OUT_DIR/model"
"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null

echo "=== tensorctl inspect ==="
report="$(arch -x86_64 "$OUT_DIR/tensorctl" inspect "$OUT_DIR/model/model.onnx")"
printf '%s\n' "$report"

input_line="$(grep '^\[model\] input:' <<<"$report")"
output_line="$(grep '^\[model\] output:' <<<"$report")"
require '[[ "$input_line" == *"[1,13]"* && "$input_line" == *"float32"* ]]' "$input_line"
require '[[ "$output_line" == *"[1,3]"* && "$output_line" == *"float32"* ]]' "$output_line"
require '[[ "$(grep -c "^\[model\] nodes: MatMul Add Relu MatMul Add\$" <<<"$report")" -eq 1 ]]' "node sequence"
require '[[ "$(grep -c "^\[model\] initializers: w1, b1, w2, b2\$" <<<"$report")" -eq 1 ]]' "initializer set"

param_line="$(grep '^\[derived\] total parameters:' <<<"$report")"
total_params="$(grep -oE '[0-9]+' <<<"$param_line" | head -1)"
total_param_bytes="$(grep -oE '[0-9]+' <<<"$param_line" | sed -n 2p)"
require '[[ "$total_params" == "275" ]]' "$param_line"                 # ground truth for the fixed 13->16->3 Wine shape, not a format check
require '[[ "$total_param_bytes" == "1100" ]]' "$param_line"           # 275 float32 elements
require '[[ "$(grep -c "^\[derived\] fusion:" <<<"$report")" -eq 2 ]]' "fusion groups"

macs="$(grep '^\[estimated\] total MACs' <<<"$report" | grep -oE '[0-9]+$')"
elemwise="$(grep '^\[estimated\] elementwise ops' <<<"$report" | grep -oE '[0-9]+$')"
require '[[ "$macs" == "256" ]]' "MACs"                                # 13*16 + 16*3 = 256, ground truth for this shape
require '[[ "$elemwise" == "35" ]]' "elementwise ops"                  # 16+16+3 = 35
require '[[ "$(grep -c "^\[measured\] none " <<<"$report")" -ge 1 ]]' "inspect performs no execution"
require '[[ "$(grep -c "^\[unknown\] lifetime crossover" <<<"$report")" -eq 1 ]]' "unknown section present"
require '[[ "$(grep -c "^\[unsupported\] none " <<<"$report")" -eq 1 ]]' "all nodes in v0 subset"

echo "=== tensorctl build (twice, for reproducibility) ==="
build_report="$(arch -x86_64 "$OUT_DIR/tensorctl" build "$OUT_DIR/model/model.onnx" -o "$OUT_DIR/model-native")"
second_build_report="$(arch -x86_64 "$OUT_DIR/tensorctl" build "$OUT_DIR/model/model.onnx" -o "$OUT_DIR/model-native-second")"
printf '%s\n' "$build_report"
cmp "$OUT_DIR/model-native" "$OUT_DIR/model-native-second"             # byte-identical artifact from the same ONNX input — the actual reproducibility claim, checked directly, not by parsing prose about it
require '[[ "$(grep -c "^\[derived\] native artifact:" <<<"$build_report")" -eq 1 ]]' "$build_report"
require '[[ "$(grep -c "^\[derived\] source: imported ONNX initializers and graph\$" <<<"$build_report")" -eq 1 ]]' "$build_report"
repro_line="$(grep '^\[derived\] reproducibility:' <<<"$build_report")"
require '[[ "$repro_line" == *"UUID"* && "$repro_line" == *"ONNX"* ]]' "$repro_line"   # concept asserted (deterministic Mach-O UUID derived from ONNX bytes), not the exact sentence
# dyld-loadability of the artifact (the actual LC_UUID-missing crash this
# gate exists to catch) is exercised for real by `tensorctl verify` below,
# which invokes it as {model-native, input_file, output_file} — asserting
# a non-empty native raw_fingerprint there already proves it loaded and
# ran; a separate ad-hoc invocation here would just duplicate that with a
# wrong calling convention.

echo "=== tensorctl verify ==="
verify_report="$(TENSORCTL_PYTHON="$OUT_DIR/venv/bin/python" arch -x86_64 "$OUT_DIR/tensorctl" verify "$OUT_DIR/model/model.onnx" "$OUT_DIR/model-native")"
printf '%s\n' "$verify_report"

vin="$(grep '^\[model\] input:' <<<"$verify_report")"
vout="$(grep '^\[model\] output:' <<<"$verify_report")"
require '[[ "$(field shape "$vin")" == "[1,13]" ]]' "$vin"
require '[[ "$(field dtype "$vin")" == "float32" ]]' "$vin"
require '[[ "$(field elements "$vin")" == "13" ]]' "$vin"
require '[[ "$(field bytes "$vin")" == "52" ]]' "$vin"
require '[[ "$(field shape "$vout")" == "[1,3]" ]]' "$vout"
require '[[ "$(field dtype "$vout")" == "float32" ]]' "$vout"
require '[[ "$(field elements "$vout")" == "3" ]]' "$vout"
require '[[ "$(field bytes "$vout")" == "12" ]]' "$vout"

oracle_line="$(grep '^\[measured\] oracle:' <<<"$verify_report")"
native_line="$(grep '^\[measured\] native:' <<<"$verify_report")"
require '[[ "$(field engine "$oracle_line")" == "onnxruntime" ]]' "$oracle_line"        # oracle engine provenance
require '[[ -n "$(field raw_fingerprint "$oracle_line")" ]]' "$oracle_line"             # diagnostic only, per the fixed contract — presence checked, exact value not asserted
require '[[ "$(field nan "$oracle_line")" == "0" && "$(field inf "$oracle_line")" == "0" ]]' "$oracle_line"
require '[[ -n "$(field raw_fingerprint "$native_line")" ]]' "$native_line"             # native output provenance
require '[[ "$(field nan "$native_line")" == "0" && "$(field inf "$native_line")" == "0" ]]' "$native_line"

# The error line is diagnostic detail (not itself a pass/fail gate) — never
# hidden: always printed above as part of $verify_report, and pulled out
# again explicitly here so max_abs/max_relative/max_ulp/bit_mismatches are
# visible in the gate's own summary even if nobody scrolls up to find them.
error_line="$(grep '^\[measured\] error:' <<<"$verify_report")"
max_abs="$(field max_abs "$error_line")"; max_rel="$(field max_relative "$error_line")"
max_ulp="$(field max_ulp "$error_line")"; bit_mismatches="$(field bit_mismatches "$error_line")"
require '[[ -n "$max_abs" && -n "$max_rel" && -n "$max_ulp" && -n "$bit_mismatches" ]]' "$error_line"
printf 'raw float error on this run: max_abs=%s max_relative=%s max_ulp=%s bit_mismatches=%s (informational — semantic_verify below is the gate)\n' \
  "$max_abs" "$max_rel" "$max_ulp" "$bit_mismatches"

sv_line="$(grep '^\[measured\] semantic_verify:' <<<"$verify_report")"
require '[[ "$(field abs_tol "$sv_line")" == "1e-5" && "$(field rel_tol "$sv_line")" == "1e-5" ]]' "$sv_line"
require '[[ "$sv_line" == *"semantic_verify: PASS"* ]]' "$sv_line"

echo "=== tensorctl verify --strict-bits (expected to fail on this exact float error) ==="
if TENSORCTL_PYTHON="$OUT_DIR/venv/bin/python" arch -x86_64 "$OUT_DIR/tensorctl" verify "$OUT_DIR/model/model.onnx" "$OUT_DIR/model-native" --strict-bits >"$OUT_DIR/strict.out"; then
  printf '%s\n' 'strict-bits unexpectedly passed despite the known raw-bit difference (bit_mismatches=1/3 above) — the gate expects it to fail here' >&2
  exit 1
fi
strict_report="$(cat "$OUT_DIR/strict.out")"
printf '%s\n' "$strict_report"
require '[[ "$(grep "^\[measured\] semantic_verify:" <<<"$strict_report")" == *PASS* ]]' "$strict_report"   # same model still semantically equivalent
require '[[ "$(grep -c "^\[measured\] bitwise_verify: FAIL\$" <<<"$strict_report")" -eq 1 ]]' "$strict_report"

echo "=== unsupported graph (Sigmoid, entirely outside the v0 op subset) must not silently succeed or leave a partial artifact ==="
# Conv used to be this fixture's unsupported op, but it is now genuinely
# supported end to end (inspect reports the layout transform, build lowers
# it to native HWC code, verify passes on the real mnist-8.onnx CNN) --
# using it here would assert a premise that is no longer true. Sigmoid is
# not in the v0 subset at all (not even import-supported), so it still
# exercises the same fail-closed contract this test exists to check.
"$OUT_DIR/venv/bin/python" - "$OUT_DIR/unsupported.onnx" <<'PY'
import sys
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
graph = helper.make_graph([helper.make_node("Sigmoid", ["x"], ["y"])], "unsupported", [x], [y], [])
onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10), sys.argv[1])
PY
conv_inspect="$(arch -x86_64 "$OUT_DIR/tensorctl" inspect "$OUT_DIR/unsupported.onnx" || true)"
require '[[ "$conv_inspect" == *"[unsupported]"* && "$conv_inspect" == *"Sigmoid"* && "$conv_inspect" == *"not in the v0 subset"* ]]' "$conv_inspect"
if arch -x86_64 "$OUT_DIR/tensorctl" build "$OUT_DIR/unsupported.onnx" -o "$OUT_DIR/unsupported-native" >"$OUT_DIR/unsupported-build.out" 2>&1; then
  printf '%s\n' 'unsupported graph unexpectedly built' >&2
  exit 1
fi
unsupported_build="$(cat "$OUT_DIR/unsupported-build.out")"
require '[[ "$unsupported_build" == *"error"* || "$unsupported_build" == *"unsupported"* ]]' "$unsupported_build"
require '[[ ! -e "$OUT_DIR/unsupported-native" ]]' "unsupported build must not leave a partial artifact"

echo "=== real foreign CNN (mnist-8.onnx) must build and verify end to end, no manual edits ==="
if [[ -n "${MNIST_ONNX:-}" ]]; then
  mnist_inspect="$(arch -x86_64 "$OUT_DIR/tensorctl" inspect "$MNIST_ONNX")"
  require '[[ "$mnist_inspect" == *"[unsupported] none"* ]]' "$mnist_inspect"
  require '[[ "$mnist_inspect" == *"[build-unsupported] none"* ]]' "$mnist_inspect"
  arch -x86_64 "$OUT_DIR/tensorctl" build "$MNIST_ONNX" -o "$OUT_DIR/mnist-native" >/dev/null
  mnist_verify="$(TENSORCTL_PYTHON="$OUT_DIR/venv/bin/python" arch -x86_64 "$OUT_DIR/tensorctl" verify "$MNIST_ONNX" "$OUT_DIR/mnist-native")"
  printf '%s\n' "$mnist_verify"
  require '[[ "$mnist_verify" == *"semantic_verify: PASS"* ]]' "$mnist_verify"
else
  printf '%s\n' 'mnist-8.onnx CNN build/verify check skipped: set MNIST_ONNX to the downloaded fixture'
fi

echo "=== unsupported Gemm bias broadcast must fail before code generation ==="
"$OUT_DIR/venv/bin/python" - "$OUT_DIR/gemm-bias.onnx" <<'PY'
import sys
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
w = helper.make_tensor("w", TensorProto.FLOAT, [3, 4], [0.25] * 12)
# ONNX can broadcast [M,1] to [M,N], but tensorctl's v0 Gemm emitter only
# implements a column bias.  This model must be rejected, never miscompiled.
c = helper.make_tensor("c", TensorProto.FLOAT, [2, 1], [1.0, 2.0])
node = helper.make_node("Gemm", ["x", "w", "c"], ["y"], name="row_bias")
graph = helper.make_graph([node], "gemm_row_bias", [x], [y], [w, c])
onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10), sys.argv[1])
PY
if arch -x86_64 "$OUT_DIR/tensorctl" inspect "$OUT_DIR/gemm-bias.onnx" >"$OUT_DIR/gemm-bias-inspect.out" 2>&1; then
  printf '%s\n' 'unsupported Gemm bias unexpectedly passed inspect' >&2
  exit 1
fi
gemm_bias_inspect="$(cat "$OUT_DIR/gemm-bias-inspect.out")"
require '[[ "$gemm_bias_inspect" == *"[unsupported]"*"(Gemm)"*"bias C shape"*"expected [4] or [1,4]"* ]]' "$gemm_bias_inspect"
if arch -x86_64 "$OUT_DIR/tensorctl" build "$OUT_DIR/gemm-bias.onnx" -o "$OUT_DIR/gemm-bias-native" >"$OUT_DIR/gemm-bias-build.out" 2>&1; then
  printf '%s\n' 'unsupported Gemm bias unexpectedly built' >&2
  exit 1
fi
gemm_bias_build="$(cat "$OUT_DIR/gemm-bias-build.out")"
require '[[ "$gemm_bias_build" == *"[unsupported]"*"(Gemm)"*"bias C shape"* ]]' "$gemm_bias_build"
require '[[ ! -e "$OUT_DIR/gemm-bias-native" ]]' "unsupported Gemm bias must not leave a partial artifact"

echo "=== grouped Conv must be rejected by the importer contract ==="
"$OUT_DIR/venv/bin/python" - "$OUT_DIR/grouped-conv.onnx" <<'PY'
import sys
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1, 1])
w = helper.make_tensor("w", TensorProto.FLOAT, [2, 1, 3, 3], [1.0] * 18)
node = helper.make_node("Conv", ["x", "w"], ["y"], name="depthwise", group=2)
graph = helper.make_graph([node], "grouped_conv", [x], [y], [w])
onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10), sys.argv[1])
PY
if arch -x86_64 "$OUT_DIR/tensorctl" inspect "$OUT_DIR/grouped-conv.onnx" >"$OUT_DIR/grouped-conv-inspect.out" 2>&1; then
  printf '%s\n' 'grouped Conv unexpectedly passed inspect' >&2
  exit 1
fi
grouped_conv_inspect="$(cat "$OUT_DIR/grouped-conv-inspect.out")"
require '[[ "$grouped_conv_inspect" == *"[unsupported]"*"(Conv)"*"grouped convolution"*"group=2"* ]]' "$grouped_conv_inspect"

printf '%s\n' 'tensorctl foreign ONNX inspect/build/verify passed: Wine 13->16->3'
