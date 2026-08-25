#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/logvec-bench.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

PYTHON="${PYTHON:-python3}"
CORE_OBJ="$OUT_DIR/logvec_core.o"
LOGVEC="$OUT_DIR/logvec_cpp"
QUERY="$OUT_DIR/query.bin"

"$ROOT/bin/fasm" --emit=macho-obj "$ROOT/fasm/apps/logvec_core.asm" "$CORE_OBJ" >/dev/null
clang++ -std=c++20 -O2 -arch x86_64 -pthread \
    "$ROOT/fasm/apps/logvec/logvec.cpp" \
    "$CORE_OBJ" \
    -o "$LOGVEC"

"$PYTHON" - <<'PY' "$OUT_DIR" "$QUERY"
import math, struct, sys, pathlib
out = pathlib.Path(sys.argv[1])
query = pathlib.Path(sys.argv[2])
DIM = 768
MAGIC = b"LOGVEC1\x00"

def unit(vec):
    n = math.sqrt(sum(x*x for x in vec))
    u = [x/n for x in vec]
    return u, math.sqrt(sum(x*x for x in u))

def write_lv(path, count):
    data = bytearray(MAGIC)
    data += struct.pack("<II", 1, DIM)
    data += struct.pack("<Q", count)
    data += struct.pack("<QQ", 0, 0)
    for i in range(count):
        vec = [math.sin(i * 0.01 + j * 0.1) for j in range(DIM)]
        u, un = unit(vec)
        data += struct.pack("<Q", i)
        data += struct.pack("<fI", un, 0)
        data += struct.pack("<" + "f"*DIM, *u)
    path.write_bytes(data)

q = [1.0] + [0.0]*(DIM-1)
query.write_bytes(struct.pack("<" + "f"*DIM, *q))
for count in (1000, 10000, 100000):
    write_lv(out / f"bench_{count}.lv", count)
PY

printf '\nlogvec bench (in-process scan, x86_64)\n'
printf '%-8s %-12s %-12s\n' 'count' 'median_ms' 'subproc_ms'
for count in 1000 10000 100000; do
    INDEX="$OUT_DIR/bench_${count}.lv"
    BENCH_LINE="$(arch -x86_64 "$LOGVEC" bench --index "$INDEX" --query "$QUERY" --top 8 --iters 50)"
    MEDIAN="$(printf '%s\n' "$BENCH_LINE" | sed -n 's/.*median_ms=\([0-9.]*\).*/\1/p')"
    T0=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
    for _ in $(seq 1 10); do
        arch -x86_64 "$LOGVEC" search --index "$INDEX" --query "$QUERY" --top 8 >/dev/null
    done
    T1=$(python3 - <<'PY'
import time
print(time.perf_counter())
PY
)
    SUBPROC="$(python3 - <<PY
print(f"{(float('$T1') - float('$T0')) / 10 * 1000:.2f}")
PY
)"
    printf '%-8s %-12s %-12s\n' "$count" "$MEDIAN" "$SUBPROC"
done

echo
echo "$BENCH_LINE" | sed 's/^/last bench line: /'
