#!/usr/bin/env bash
# tensorctl build reproducibility gate.
#
# Build the same supported external ONNX model at least five times from
# distinct working directories and require identical executable bytes and
# identical runtime output. Failed builds must leave no partial artifact.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WINE_DATA:?set WINE_DATA to the UCI wine.data file}"
BUILDS="${TENSORCTL_REPRO_BUILDS:-5}"
if (( BUILDS < 5 )); then
  printf 'FAILED: TENSORCTL_REPRO_BUILDS=%s, gate requires at least 5 builds\n' "$BUILDS" >&2
  exit 1
fi

OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tensorctl-reproducible.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

uv venv --python 3.13 "$OUT_DIR/venv" >/dev/null
uv pip install --python "$OUT_DIR/venv/bin/python" numpy==2.5.2 onnx==1.22.0 >/dev/null
"$OUT_DIR/venv/bin/python" "$ROOT/scripts/tensor_killer_tabular.py" prepare \
  --wine "$WINE_DATA" --out "$OUT_DIR/model" >/dev/null
"$ROOT/scripts/build-tensorctl.sh" "$OUT_DIR/tensorctl" >/dev/null

model="$OUT_DIR/model/model.onnx"
head -c 52 "$OUT_DIR/model/inputs.bin" > "$OUT_DIR/input.f32"
[[ "$(wc -c < "$OUT_DIR/input.f32")" -eq 52 ]] || {
  printf 'FAILED: could not slice a 13-float input row\n' >&2
  exit 1
}

mkdir -p "$OUT_DIR/art"
first_bin_hash=""
first_out_hash=""
for ((i = 1; i <= BUILDS; i++)); do
  wd="$OUT_DIR/wd-$i"
  mkdir -p "$wd"
  art="$OUT_DIR/art/native.$i"
  (cd "$wd" && arch -x86_64 "$OUT_DIR/tensorctl" build "$model" -o "$art") >/dev/null

  [[ -x "$art" ]] || {
    printf 'FAILED: build %s produced no executable at %s\n' "$i" "$art" >&2
    exit 1
  }
  bin_hash="$(shasum -a 256 "$art" | awk '{print $1}')"

  arch -x86_64 "$art" "$OUT_DIR/input.f32" "$OUT_DIR/art/out.$i.f32"
  out_hash="$(shasum -a 256 "$OUT_DIR/art/out.$i.f32" | awk '{print $1}')"

  if [[ -z "$first_bin_hash" ]]; then
    first_bin_hash="$bin_hash"
    first_out_hash="$out_hash"
  else
    if [[ "$bin_hash" != "$first_bin_hash" ]]; then
      printf 'FAILED: build %s Mach-O SHA-256 %s != build 1 %s\n' \
        "$i" "$bin_hash" "$first_bin_hash" >&2
      cmp "$OUT_DIR/art/native.1" "$art" || true
      exit 1
    fi
    if [[ "$out_hash" != "$first_out_hash" ]]; then
      printf 'FAILED: build %s runtime output SHA-256 %s != build 1 %s\n' \
        "$i" "$out_hash" "$first_out_hash" >&2
      exit 1
    fi
  fi
done

# Conv is parsed by the importer but is outside the current build lowering.
"$OUT_DIR/venv/bin/python" - "$OUT_DIR/unsupported.onnx" <<'PY'
import sys
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3, 3])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 1])
w = helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 3, 3], [1.0] * 9)
graph = helper.make_graph(
    [helper.make_node("Conv", ["x", "w"], ["y"])],
    "unsupported",
    [x],
    [y],
    [w],
)
onnx.save(
    helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10),
    sys.argv[1],
)
PY
partial="$OUT_DIR/art/should-not-exist"
if arch -x86_64 "$OUT_DIR/tensorctl" build "$OUT_DIR/unsupported.onnx" -o "$partial" >/dev/null 2>&1; then
  printf 'FAILED: unsupported graph unexpectedly built\n' >&2
  exit 1
fi
[[ ! -e "$partial" ]] || {
  printf 'FAILED: failed build left a partial artifact at %s\n' "$partial" >&2
  exit 1
}

printf 'not an onnx file' > "$OUT_DIR/corrupt.onnx"
partial2="$OUT_DIR/art/should-not-exist-2"
if arch -x86_64 "$OUT_DIR/tensorctl" build "$OUT_DIR/corrupt.onnx" -o "$partial2" >/dev/null 2>&1; then
  printf 'FAILED: corrupt ONNX unexpectedly built\n' >&2
  exit 1
fi
[[ ! -e "$partial2" ]] || {
  printf 'FAILED: corrupt-input build left a partial artifact at %s\n' "$partial2" >&2
  exit 1
}

printf 'tensorctl reproducible check passed: builds=%s mach_o_sha256=%s identical_across_distinct_cwd=yes runtime_output_sha256=%s expected_fail_no_partial_artifact=yes\n' \
  "$BUILDS" "$first_bin_hash" "$first_out_hash"
