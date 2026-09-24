#!/usr/bin/env bash
# Bounded EML search audit. Every case has a hard timeout. Do not raise it to minutes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_ARCH="$(uname -m)"
OUT_DIR="${EML_AUDIT_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/eml-search-audit.XXXXXX")}"
mkdir -p "$OUT_DIR"
INC=(-I "$ROOT/fasm/apps/eml_sr")

run_bounded() {
    local seconds="$1"
    local log="$2"
    shift 2
    "$@" >"$log" 2>&1 &
    local pid=$!
    local i=0
    while [[ "$i" -lt "$seconds" ]]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" || true
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    kill -9 "$pid" 2>/dev/null || true
    echo "TIMEOUT seconds=$seconds" >>"$log"
    return 124
}

build_one() {
    local arch="$1"
    local bin="$OUT_DIR/audit-$arch"
    local reg="$OUT_DIR/regression-$arch"
    local consumer="$OUT_DIR/consumer-$arch"
    clang++ -std=c++20 -O2 -arch "$arch" "${INC[@]}" \
        "$ROOT/fasm/tests/eml_sr/eml_search_audit.cpp" -o "$bin"
    clang++ -std=c++20 -O2 -arch "$arch" "${INC[@]}" \
        "$ROOT/fasm/tests/eml_sr/eml_memo_regression.cpp" -o "$reg"
    clang++ -std=c++20 -O2 -arch "$arch" "${INC[@]}" \
        "$ROOT/fasm/tests/eml_sr/minimal_consumer.cpp" -o "$consumer"
}

launch() {
    local arch="$1"
    shift
    if [[ "$arch" == "$HOST_ARCH" ]]; then
        "$@"
    else
        arch "-$arch" "$@"
    fi
}

run_arch() {
    local arch="$1"
    local bin="$OUT_DIR/audit-$arch"
    local log="$OUT_DIR/$arch.log"
    : >"$log"
    echo "arch=$arch compiler=clang++ flags=-std=c++20 -O2 -arch $arch" | tee -a "$log"

    echo "== regression $arch ==" | tee -a "$log"
    run_bounded 20 "$OUT_DIR/$arch-regression.log" launch "$arch" "$OUT_DIR/regression-$arch" || true
    cat "$OUT_DIR/$arch-regression.log" | tee -a "$log"

    echo "== consumer $arch ==" | tee -a "$log"
    run_bounded 15 "$OUT_DIR/$arch-consumer.log" launch "$arch" "$OUT_DIR/consumer-$arch" || true
    cat "$OUT_DIR/$arch-consumer.log" | tee -a "$log"

    local spec
    for spec in \
        "per exp 1 5 8" \
        "per ln 3 5 8" \
        "per poly 1 3 8" \
        "per poly 2 3 8" \
        "per poly 3 3 8" \
        "per poly 4 3 12" \
        "global exp 1 3 8" \
        "global ln 3 3 8" \
        "global poly 1 1 8" \
        "global poly 2 1 8" \
        "global poly 3 1 8" \
        "global poly 4 1 8"
    do
        read -r memo case depth repeats seconds <<<"$spec"
        local one="$OUT_DIR/$arch-$memo-$case-d$depth.log"
        echo "== $arch memo=$memo case=$case depth=$depth ==" | tee -a "$log"
        if run_bounded "$seconds" "$one" launch "$arch" "$bin" \
            --memo "$memo" --case "$case" --depth "$depth" --repeats "$repeats"; then
            cat "$one" | tee -a "$log"
        else
            echo "TIMEOUT arch=$arch memo=$memo case=$case depth=$depth seconds=$seconds" | tee -a "$log"
            tail -n 5 "$one" | tee -a "$log" || true
        fi
    done

    echo "== depth5 exploratory $arch ==" | tee -a "$log"
    local d5="$OUT_DIR/$arch-per-poly-d5.log"
    if run_bounded 15 "$d5" launch "$arch" "$bin" --memo per --case poly --depth 5 --repeats 1; then
        cat "$d5" | tee -a "$log"
    else
        echo "TIMEOUT arch=$arch memo=per case=poly depth=5 seconds=15" | tee -a "$log"
    fi
}

echo "host=$HOST_ARCH out=$OUT_DIR"
build_one "$HOST_ARCH"
if [[ "$HOST_ARCH" == "arm64" ]]; then
    build_one x86_64
fi

run_arch "$HOST_ARCH"
if [[ "$HOST_ARCH" == "arm64" ]]; then
    run_arch x86_64
fi

echo "audit_log_dir=$OUT_DIR"
