# logvec v0

Experimental brew-worthy tool inside `fasm-mac`: batch snapshot consumer that
builds a local `.lv` index from logbus embedding payloads and runs exact cosine
top-k search.

## Architecture (locked)

```text
logbus     dumb durable log — no vector semantics
FASM       f32 dot / norm / exact top-k / in-place `.lv` top-k (`lb_vec_topk_cosine_lv`) only
Zig        payload validate, CRC envelope check, index I/O, FETCH/--dir ingest,
           doc_id mapping, final sort (score desc, doc_id asc)
C++        same host responsibilities as Zig (`fasm/apps/logvec/`) — additive alternative;
           CRC via FASM `lb_crc32c`, index search loads `.lv` via mmap and uses
           in-place `lb_vec_topk_cosine_lv`; `build-index` streams one record at a time
```

`build-index` is a **one-shot snapshot builder**. It does not tail topics or
refresh indexes in real time.

System form (Level 4 — truth, projection, invariants):
[`docs/system_form.md`](system_form.md).

## Metric (v0)

- **Cosine similarity** on raw f32 vectors
- Index stores raw vector + precomputed L2 norm per record
- Query norm is computed at search time
- `score = dot(query, vector) / (norm(query) * norm(vector))`
- Higher score is better

## Embedding payload (logbus record body)

```text
u32 dim
f32 vector[dim]
u64 doc_id   optional; omit => auto-id
```

Auto-id uses the **logical topic record offset** (the offset logbus assigns
within a topic, not a byte offset):

- `build-index --host/--port` — offset from FETCH (`:offset` line)
- `build-index --dir` — same logical offset while replaying topic segment files
- `build-index --payload-dir` — dev-only path; uses sorted file order `0..n-1`

## `.lv` index v1

Header:

```text
u8  magic[8] = "LOGVEC1\0"
u32 version = 1
u32 dim
u64 count
u64 flags = 0
u64 reserved = 0
```

Record (repeated `count` times):

```text
u64 doc_id
f32 norm
u32 reserved = 0
f32 vector[dim]
```

Record size is `16 + dim*4` bytes. The per-record `reserved` field keeps the
vector at a stable offset for FASM readers.

## CLI

```text
logvec search --index PATH --query PATH --top K
logvec build-index --payload-dir DIR --out PATH
logvec build-index --host H --port P --topic TOPIC --out PATH
logvec build-index --dir DATA --topic TOPIC --out PATH
```

`search` writes `doc_id score` lines to stdout (score fixed to 6 decimal places).

```text
logvec bench --index PATH --query PATH [--top K] [--iters N]
  [--layer dot|topk|search|io] [--simd auto|scalar|avx2]
  [--cold] [--threads N] [--breakdown]
```

`bench` loads the index once and times in-process exact top-k (median over
`--iters`, default 50). Layers isolate SIMD dot scan, FASM top-k, full search
(including doc_id resolve), and mmap load. Use for regression checks, not as a
FAISS comparison.

```text
ragbox bench --index PATH --manifest PATH --query-file PATH
  [--top K] [--iters N] [--threads N] [--snippet-len N] [--breakdown]
```

Offline ragbox bench (no Ollama): times search + manifest join + snippet load.

## Performance (v0.2, Apple Silicon via Rosetta x86_64)

Exact brute-force cosine over mmap'd unit vectors. FASM AVX2 dot/norm; parallel
search partitions records across 1–4 threads and merges partial top-k heaps.

| Layer | 10k×768 | 100k×768 | Notes |
|-------|--------:|---------:|-------|
| dot (AVX2) | ~4.6 ms | ~45 ms | dot-only, no heap |
| search (1 thread) | ~4.5 ms | ~46 ms | full path |
| search (4 threads) | ~1.4 ms | ~12 ms | parallel merge top-k |
| top-k kernel | ~4.4 ms | ~44 ms | FASM heap only |

v0 scalar (~44 ms) → v0.1 AVX2 (~4.5 ms) → v0.2 parallel 4t (~1.4 ms at 10k).

Subprocess `search` adds ~25–40 ms process spawn overhead per query — not
representative of in-process use (ragbox embeds search in-process).

```sh
scripts/bench_logvec.sh    # quick end-to-end table
scripts/bench_perf.sh      # layered bench + ragbox breakdown + CI gate
```

This is **not** ANN. At 100k×768 (~300 MB index) scan time scales linearly.
For large corpora use an external ANN index; logvec/ragbox target agent-scale
snapshots (1k–50k chunks), not billion-vector search. Product positioning:
[`docs/system_form.md`](system_form.md).

## CRC32C

Zig verifies logbus record envelopes with the same CRC32C polynomial/format as
logbus (`[u32_len][u32_crc32c][payload]`). The C++ host delegates CRC to FASM
`lb_crc32c` (`fasm/core/crc32c.inc`). Parity is tested against
`scripts/check_logbus.sh` fixtures, not against Python as the spec source.

## Checks

```sh
scripts/check_logvec_search.sh   # PR1: fixture .lv -> search only
scripts/check_logvec.sh          # wrapper; grows with ingest parity tests
scripts/check_logvec_cpp.sh      # same checks for C++ host (clang++)
```
