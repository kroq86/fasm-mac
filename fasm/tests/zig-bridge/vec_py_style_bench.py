#!/usr/bin/env python3
"""Optional diagnostic: handbook-style ctypes + Python orchestration overhead."""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import CDLL, c_double, c_ulonglong
from pathlib import Path

DIM = 128
DB_LEN = 256
SEARCH_ITERS = 120


def fill_vectors() -> tuple[list[float], list[list[float]]]:
    query = [(i + 1) * 0.01 for i in range(DIM)]
    database: list[list[float]] = []
    for row in range(DB_LEN):
        database.append([__import__("math").sin((row + 1) * (col + 3) * 0.001) for col in range(DIM)])
    return query, database


def cosine_similarity(lib, query: list[float], vector: list[float]) -> float:
    q_arr = (c_double * len(query))(*query)
    v_arr = (c_double * len(vector))(*vector)
    dot = lib.lb_dot_product(q_arr, v_arr, len(query))
    norm_q = lib.lb_vector_norm(q_arr, len(query))
    norm_v = lib.lb_vector_norm(v_arr, len(vector))
    if norm_q == 0.0 or norm_v == 0.0:
        return 0.0
    return dot / (norm_q * norm_v)


def bench_python_style(lib, query: list[float], database: list[list[float]], iters: int) -> int:
    start = time.perf_counter_ns()
    sink = 0.0
    for _ in range(iters):
        for vector in database:
            sink += cosine_similarity(lib, query, vector)
    if sink == float("inf"):
        print("unexpected", file=sys.stderr)
    elapsed = time.perf_counter_ns() - start
    return elapsed


def main() -> int:
    dylib = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("vec_bridge.dylib")
    lib = CDLL(str(dylib))
    lib.lb_dot_product.argtypes = [ctypes.POINTER(c_double), ctypes.POINTER(c_double), c_ulonglong]
    lib.lb_dot_product.restype = c_double
    lib.lb_vector_norm.argtypes = [ctypes.POINTER(c_double), c_ulonglong]
    lib.lb_vector_norm.restype = c_double

    query, database = fill_vectors()
    ns = bench_python_style(lib, query, database, SEARCH_ITERS)
    per_search = ns / (SEARCH_ITERS * DB_LEN)
    print(f"python-style ns_per_search={per_search:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
