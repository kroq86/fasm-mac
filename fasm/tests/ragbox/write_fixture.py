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

INCREMENTAL_DELTA_RECORD = {
    "doc_id": 3,
    "path": "docs/auth.md",
    "offset": 0,
    "text": "Updated JWT authentication middleware with OAuth2 bearer tokens.",
    "vector": [0.9, 0.1, 0.0, 0.0],
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


def write_lv_records(path: Path, records: list[dict]) -> None:
    out = bytearray()
    out += MAGIC
    out += struct.pack("<II", VERSION, DIM)
    out += struct.pack("<Q", len(records))
    out += struct.pack("<QQ", 0, 0)
    for rec in records:
        vec = rec["vector"]
        n = norm(vec)
        unit = [x / n for x in vec]
        un = norm(unit)
        out += struct.pack("<Q", rec["doc_id"])
        out += struct.pack("<fI", un, 0)
        out += struct.pack("<" + "f" * DIM, *unit)
    path.write_bytes(out)


def write_lv(path: Path) -> None:
    write_lv_records(path, RECORDS)


def write_manifest(path: Path, root: str, *, lite: bool = False) -> None:
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
                **({} if lite else {"text": rec["text"]}),
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


def write_incremental_fixtures(fixture_dir: Path) -> None:
    inc_dir = fixture_dir / "incremental"
    inc_dir.mkdir(parents=True, exist_ok=True)
    root = str(fixture_dir / "tiny-repo")

    write_lv_records(inc_dir / "base.lv", RECORDS)
    write_lv_records(inc_dir / "base.lv.delta", [INCREMENTAL_DELTA_RECORD])

    active_records = [RECORDS[1], RECORDS[2], INCREMENTAL_DELTA_RECORD]
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
            }
            for rec in active_records
        ],
    }
    (inc_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    state = {
        "version": 1,
        "next_doc_id": 4,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "model": MODEL,
        "root": root,
        "files": {
            "docs/auth.md": {"hash": "updated-auth-hash", "doc_ids": [3]},
            "docs/db.md": {"hash": "db-hash", "doc_ids": [1]},
            "src/middleware.go": {"hash": "middleware-hash", "doc_ids": [2]},
        },
        "superseded_doc_ids": [0],
    }
    (inc_dir / "base.lv.state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    query_vec = QUERIES["auth"]
    rec = INCREMENTAL_DELTA_RECORD
    score = round(cosine(query_vec, rec["vector"]), 6)
    auth_path = fixture_dir / "tiny-repo/docs/auth.md"
    auth_content = auth_path.read_text(encoding="utf-8")
    snippet = auth_content[rec["offset"] : rec["offset"] + len(rec["text"])]
    expected = [
        {
            "doc_id": rec["doc_id"],
            "score": score,
            "path": rec["path"],
            "offset": rec["offset"],
            "snippet": snippet,
        }
    ]
    (inc_dir / "expected_auth.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    (inc_dir / "query_auth.bin").write_bytes(struct.pack("<" + "f" * DIM, *query_vec))

    # Minimal state for refresh dry-run smoke (auth.md hash differs from tiny-repo content).
    refresh_state = {
        "version": 1,
        "next_doc_id": 3,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "model": MODEL,
        "root": root,
        "files": {
            "docs/auth.md": {"hash": "stale-auth-hash", "doc_ids": [0]},
            "docs/db.md": {"hash": "db-hash", "doc_ids": [1]},
            "src/middleware.go": {"hash": "middleware-hash", "doc_ids": [2]},
        },
        "superseded_doc_ids": [],
    }
    (inc_dir / "refresh_state.json").write_text(json.dumps(refresh_state, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} FIXTURE_DIR")

    fixture_dir = Path(sys.argv[1])
    fixture_dir.mkdir(parents=True, exist_ok=True)

    write_lv(fixture_dir / "fixture.lv")
    write_manifest(fixture_dir / "fixture.manifest.json", str(fixture_dir / "tiny-repo"))
    write_manifest(fixture_dir / "fixture.manifest.lite.json", str(fixture_dir / "tiny-repo"), lite=True)

    for name, vec in QUERIES.items():
        (fixture_dir / f"query_{name}.bin").write_bytes(struct.pack("<" + "f" * DIM, *vec))
        write_expected_json(fixture_dir / f"expected_{name}.json", vec, top_k=1)

    write_incremental_fixtures(fixture_dir)

    print(f"wrote fixtures under {fixture_dir}")


if __name__ == "__main__":
    main()
