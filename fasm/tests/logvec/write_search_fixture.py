#!/usr/bin/env python3
"""Write PR1 search fixtures: .lv index, raw query, expected stdout."""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

MAGIC = b"LOGVEC1\x00"
VERSION = 1
DIM = 4

DOCS: dict[int, list[float]] = {
    100: [1.0, 0.0, 0.0, 0.0],
    200: [0.0, 1.0, 0.0, 0.0],
    300: [0.9, 0.1, 0.0, 0.0],
}
QUERY = [1.0, 0.0, 0.0, 0.0]


def norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def unit_vec(vec: list[float]) -> tuple[list[float], float]:
    n = norm(vec)
    u = [x / n for x in vec]
    return u, norm(u)


def write_lv(path: Path) -> None:
    items = sorted(DOCS.items(), key=lambda item: item[0])
    out = bytearray()
    out += MAGIC
    out += struct.pack("<II", VERSION, DIM)
    out += struct.pack("<Q", len(items))
    out += struct.pack("<QQ", 0, 0)  # flags, reserved
    for doc_id, vec in items:
        unit, unit_norm = unit_vec(vec)
        out += struct.pack("<Q", doc_id)
        out += struct.pack("<fI", unit_norm, 0)  # norm, record reserved
        out += struct.pack("<" + "f" * DIM, *unit)
    path.write_bytes(out)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} FIXTURE_DIR")

    fixture_dir = Path(sys.argv[1])
    fixture_dir.mkdir(parents=True, exist_ok=True)

    write_lv(fixture_dir / "search_smoke.lv")
    (fixture_dir / "search_query.bin").write_bytes(struct.pack("<" + "f" * DIM, *QUERY))

    scores = {doc_id: cosine(QUERY, vec) for doc_id, vec in DOCS.items()}
    top2 = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:2]
    lines = [f"{doc_id} {score:.6f}" for doc_id, score in top2]
    (fixture_dir / "expected_search.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
