#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ $# == 1 ]] || { echo "usage: bash scripts/check_kv_handoff.sh ASSET_DIR (required, never skips)" >&2; exit 2; }
ASSETS="$1"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kv-handoff-check.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT
bash "$ROOT/scripts/build-kv-handoff.sh" "$BUILD_DIR/handoff"
fasm --emit=macho-obj "$ROOT/fasm/spikes/tensor_transformer_executor_f32.asm" "$BUILD_DIR/executor.o" >/dev/null
clang -arch x86_64 -std=c11 -O2 -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-parameter \
  -DM=768 -DH=12 -DD=64 -DQW=2304 -DF=3072 -DMAXCACHE=64 \
  -I"$ROOT/fasm/spikes" "$ROOT/fasm/spikes/tensor_kv_handoff_contract_check.c" "$BUILD_DIR/executor.o" -o "$BUILD_DIR/contract"
"$BUILD_DIR/contract"
python3 "$ROOT/scripts/kv_handoff.py" --assets "$ASSETS" --binary "$BUILD_DIR/handoff" --verify
seq=AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBB
python3 "$ROOT/scripts/kv_handoff.py" --assets "$ASSETS" --binary "$BUILD_DIR/handoff" --sequence "$seq" > "$BUILD_DIR/first"
python3 "$ROOT/scripts/kv_handoff.py" --assets "$ASSETS" --binary "$BUILD_DIR/handoff" --sequence "$seq" > "$BUILD_DIR/second"
cmp "$BUILD_DIR/first" "$BUILD_DIR/second"
if python3 "$ROOT/scripts/kv_handoff.py" --assets "$ASSETS" --binary "$BUILD_DIR/handoff" --sequence ABC >/dev/null 2>&1; then
  echo "invalid sequence accepted" >&2; exit 1
fi
if python3 "$ROOT/scripts/kv_handoff.py" --assets "$BUILD_DIR/missing" --binary "$BUILD_DIR/handoff" --verify >/dev/null 2>&1; then
  echo "missing artifacts accepted" >&2; exit 1
fi
python3 - "$ROOT" "$ASSETS" <<'PY'
import importlib.util,json,pathlib,sys,tempfile
root,assets=map(pathlib.Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('launcher',root/'scripts/kv_handoff.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as td:
    tmp=pathlib.Path(td)
    for name in m.FILES: (tmp/name).symlink_to((assets/name).resolve())
    original=json.loads((assets/'manifest.json').read_text())
    for change in ('bridge_sha','geometry','missing_file'):
        d=json.loads(json.dumps(original))
        if change=='bridge_sha':d['files']['bridge.safetensors']='0'*64
        if change=='geometry':d['layers']=[0,2,4,6,8,10]
        if change=='missing_file':d['files'].pop('reference.safetensors')
        (tmp/'manifest.json').write_text(json.dumps(d))
        try:m.validate_assets(tmp)
        except ValueError:pass
        else:raise AssertionError(f'{change} accepted')
print('bundle negative checks passed')
PY
echo 'native KV handoff gate passed (32 original dev cases, per-boundary differential, replay, fail-closed inputs)'
