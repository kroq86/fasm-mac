#!/usr/bin/env python3
"""Write offline ragbox fixtures: .lv index, manifest, query bins, expected JSON."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

MAGIC = b"LOGVEC1\x00"
VERSION = 1
DIM = 4
CHUNK_SIZE = 800
OVERLAP = 100
MODEL = "fixture-dim4"

RECORDS = [
    {
        "doc_id": 0,
        "path": "docs/auth.md",
        "offset": 0,
        "text": "JWT authentication and login middleware validate bearer tokens.",
        "vector": [1.0, 0.0, 0.0, 0.0],
    },
    {
        "doc_id": 1,
        "path": "docs/db.md",
        "offset": 0,
        "text": "Postgres migrations and schema_migrations table for database deploys.",
        "vector": [0.0, 1.0, 0.0, 0.0],
    },
    {
        "doc_id": 2,
        "path": "src/middleware.go",
        "offset": 0,
        "text": "HTTP middleware AuthMiddleware wraps handlers for authorization checks.",
        "vector": [0.0, 0.0, 1.0, 0.0],
    },
]

QUERIES = {
    "auth": [1.0, 0.0, 0.0, 0.0],
    "db": [0.0, 1.0, 0.0, 0.0],
    "middleware": [0.0, 0.0, 1.0, 0.0],
}


def norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def write_lv(path: Path) -> None:
    out = bytearray()
    out += MAGIC
    out += struct.pack("<II", VERSION, DIM)
    out += struct.pack("<Q", len(RECORDS))
    out += struct.pack("<QQ", 0, 0)
    for rec in RECORDS:
        vec = rec["vector"]
        n = norm(vec)
        unit = [x / n for x in vec]
        un = norm(unit)
        out += struct.pack("<Q", rec["doc_id"])
        out += struct.pack("<fI", un, 0)
        out += struct.pack("<" + "f" * DIM, *unit)
    path.write_bytes(out)


def write_manifest(path: Path, root: str) -> None:
    manifest = {
        "version": 1,
        "dim": DIM,
        "model": MODEL,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "root": root,
        "records": [
            {
                "doc_id": rec["doc_id"],
                "path": rec["path"],
                "offset": rec["offset"],
                "length": len(rec["text"]),
                "text": rec["text"],
            }
            for rec in RECORDS
        ],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def expected_hit(doc_id: int, query_vec: list[float], snippet_len: int = 200) -> dict:
    rec = RECORDS[doc_id]
    score = cosine(query_vec, rec["vector"])
    text = rec["text"]
    snippet = text if len(text) <= snippet_len else text[:snippet_len]
    return {
        "doc_id": doc_id,
        "score": score,
        "path": rec["path"],
        "offset": rec["offset"],
        "snippet": snippet,
    }


def write_expected_json(path: Path, query_vec: list[float], top_k: int = 1) -> None:
    scores = [(rec["doc_id"], cosine(query_vec, rec["vector"])) for rec in RECORDS]
    scores.sort(key=lambda item: (-item[1], item[0]))
    hits = [expected_hit(doc_id, query_vec) for doc_id, _ in scores[:top_k]]
    path.write_text(json.dumps(hits, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} FIXTURE_DIR")

    fixture_dir = Path(sys.argv[1])
    fixture_dir.mkdir(parents=True, exist_ok=True)

    write_lv(fixture_dir / "fixture.lv")
    write_manifest(fixture_dir / "fixture.manifest.json", str(fixture_dir / "tiny-repo"))

    for name, vec in QUERIES.items():
        (fixture_dir / f"query_{name}.bin").write_bytes(struct.pack("<" + "f" * DIM, *vec))
        write_expected_json(fixture_dir / f"expected_{name}.json", vec, top_k=1)

    print(f"wrote fixtures under {fixture_dir}")


if __name__ == "__main__":
    main()
