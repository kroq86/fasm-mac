#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/logvec-perf.XXXXXX")"
trap 'rm -rf "$OUT_DIR"' EXIT

PYTHON="${PYTHON:-python3}"
CORE_OBJ="$OUT_DIR/logvec_core.o"
LOGVEC="$OUT_DIR/logvec_cpp"
RAGBOX="$OUT_DIR/ragbox"
QUERY="$OUT_DIR/query.bin"
BASELINE="$ROOT/fasm/tests/logvec/bench_baseline.txt"
FIXTURE_DIR="$ROOT/fasm/tests/ragbox/fixtures"

"$ROOT/bin/fasm" --emit=macho-obj "$ROOT/fasm/apps/logvec_core.asm" "$CORE_OBJ" >/dev/null
clang++ -std=c++20 -O2 -arch x86_64 -pthread \
    "$ROOT/fasm/apps/logvec/logvec.cpp" \
    "$CORE_OBJ" \
    -o "$LOGVEC"
clang++ -std=c++20 -O2 -arch x86_64 -pthread \
    "$ROOT/fasm/apps/ragbox/ragbox.cpp" \
    "$CORE_OBJ" \
    -o "$RAGBOX"

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

parse_median() {
    sed -n 's/.*median_ms=\([0-9.]*\).*/\1/p' | head -1
}

printf '\nlogvec layered bench (x86_64, dim=768)\n'
printf '%-8s %-10s %-12s\n' 'count' 'layer' 'median_ms'

SEARCH_10K=0

for count in 1000 10000 100000; do
    INDEX="$OUT_DIR/bench_${count}.lv"

    DOT_AVX2="$(arch -x86_64 "$LOGVEC" bench --index "$INDEX" --query "$QUERY" \
        --layer dot --simd avx2 --iters 30 | parse_median)"
    DOT_SCALAR="$(arch -x86_64 "$LOGVEC" bench --index "$INDEX" --query "$QUERY" \
        --layer dot --simd scalar --iters 30 | parse_median)"
    RATIO="$(python3 - <<PY
s=float('$DOT_SCALAR')
a=float('$DOT_AVX2')
print(f'{s/a:.2f}' if a else '0')
PY
)"
    printf '%-8s %-10s %-12s  (scalar=%s ratio=%s)\n' "$count" 'dot-avx2' "$DOT_AVX2" "$DOT_SCALAR" "$RATIO"

    SEARCH_MS="$(arch -x86_64 "$LOGVEC" bench --index "$INDEX" --query "$QUERY" \
        --layer search --threads 1 --iters 30 | parse_median)"
    printf '%-8s %-10s %-12s\n' "$count" 'search' "$SEARCH_MS"
    if [[ "$count" == "10000" ]]; then
        SEARCH_10K="$SEARCH_MS"
    fi

    if [[ "$count" != "1000" ]]; then
        SEARCH4="$(arch -x86_64 "$LOGVEC" bench --index "$INDEX" --query "$QUERY" \
            --layer search --threads 4 --iters 30 | parse_median)"
        printf '%-8s %-10s %-12s\n' "$count" 'searchx4' "$SEARCH4"
    fi

    TOPK_MS="$(arch -x86_64 "$LOGVEC" bench --index "$INDEX" --query "$QUERY" \
        --layer topk --iters 30 | parse_median)"
    printf '%-8s %-10s %-12s\n' "$count" 'topk' "$TOPK_MS"

    if [[ "$count" == "100000" ]]; then
        IO_MS="$(arch -x86_64 "$LOGVEC" bench --index "$INDEX" --query "$QUERY" \
            --layer io --iters 10 | parse_median)"
        printf '%-8s %-10s %-12s\n' "$count" 'io-load' "$IO_MS"
    fi
done

printf '\n10k search breakdown:\n'
arch -x86_64 "$LOGVEC" bench --index "$OUT_DIR/bench_10000.lv" --query "$QUERY" \
    --layer search --breakdown --threads 1 --iters 20

"$PYTHON" "$ROOT/fasm/tests/ragbox/write_fixture.py" "$FIXTURE_DIR" >/dev/null
printf '\nragbox bench (lite manifest, offline):\n'
arch -x86_64 "$RAGBOX" bench \
    --index "$FIXTURE_DIR/fixture.lv" \
    --manifest "$FIXTURE_DIR/fixture.manifest.lite.json" \
    --query-file "$FIXTURE_DIR/query_auth.bin" \
    --breakdown --iters 20

if [[ -f "$BASELINE" && -n "$SEARCH_10K" ]]; then
    LIMIT="$(grep '^search_10k=' "$BASELINE" | cut -d= -f2)"
    python3 - <<PY
limit = float('$LIMIT')
actual = float('$SEARCH_10K')
if actual > limit * 1.10:
    raise SystemExit(f'REGRESS search_10k: {actual:.3f} ms > {limit:.3f} ms (+10%)')
print(f'OK regression gate search_10k={actual:.3f} limit={limit:.3f}')
PY
fi

echo
echo 'bench_perf complete'
