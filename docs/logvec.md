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
