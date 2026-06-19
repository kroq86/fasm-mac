# ragbox

**SQLite-style local semantic index for agents** — not a vector DB server, a
file you can copy, version, and test.

ragbox wraps the existing logvec `.lv` snapshot format with:

- deterministic chunking over text/code files
- Ollama HTTP embeddings (v0 glue — no embedded model)
- a JSON manifest sidecar for path/offset/snippet metadata
- `build`, `search`, and `doctor` commands in one x86_64 binary

FASM owns dot/norm/top-k; C++ owns chunking, HTTP embed glue, mmap search, and
CLI. Zig logvec remains unchanged.

## Build

```sh
fasm --emit=macho-obj fasm/apps/logvec_core.asm logvec_core.o
clang++ -std=c++20 -O2 -arch x86_64 \
  fasm/apps/ragbox/ragbox.cpp logvec_core.o -o ragbox
```

On Apple Silicon:

```sh
arch -x86_64 ./ragbox doctor
```

## Quick start

Requires [Ollama](https://ollama.com) with an embedding model (default:
`nomic-embed-text`):

```sh
ollama pull nomic-embed-text
arch -x86_64 ./ragbox build --root ./my-repo --out memory.lv
arch -x86_64 ./ragbox search --index memory.lv --query "auth middleware" --json
arch -x86_64 ./ragbox doctor --index memory.lv
```

Outputs:

- `memory.lv` — logvec v1 index (mmap-friendly snapshot)
- `memory.lv.manifest.json` — chunk metadata sidecar (default path)

## Commands

### build

```text
ragbox build --root PATH --out PATH [--manifest PATH]
  [--chunk-size 800] [--overlap 100]
  [--ollama http://127.0.0.1:11434] [--model nomic-embed-text]
  [--dry-run]
```

Walks `--root`, chunks included text/code files, embeds each chunk via Ollama,
streams vectors into a logvec index, and writes the manifest.

`--dry-run` chunks only — writes manifest, skips Ollama and `.lv` build.

Included extensions: `.md`, `.txt`, `.jsonl`, code sources (`.py`, `.go`, `.rs`,
`.js`, `.ts`, `.cpp`, `.hpp`, `.c`, `.h`, `.zig`, `.asm`, `.sh`).

Skipped directories: dot dirs, `node_modules`, `__pycache__`.

### search

```text
ragbox search --index PATH --query TEXT [--top 8] [--json]
  [--manifest PATH] [--ollama URL] [--model MODEL]
  [--query-file PATH] [--snippet-len 200]
```

Embeds the query (or reads raw f32 `--query-file` for offline checks), runs
exact cosine top-k over the mmap index, joins manifest records:

```json
[
  {
    "doc_id": 0,
    "score": 0.912345,
    "path": "docs/auth.md",
    "offset": 0,
    "snippet": "Auth middleware validates JWT..."
  }
]
```

Without `--json`: `path:offset score` lines.

### doctor

```text
ragbox doctor [--index PATH] [--manifest PATH]
  [--ollama URL] [--model MODEL] [--skip-ollama]
```

Checks: x86_64 note, Ollama reachability, model present, index magic/count,
manifest dim/count/doc_id continuity.

Use `--skip-ollama` in CI (offline fixtures).

## Manifest schema v1

Default path: `<index-path>.manifest.json`.

```json
{
  "version": 1,
  "dim": 768,
  "model": "nomic-embed-text",
  "chunk_size": 800,
  "overlap": 100,
  "root": "/abs/path/to/repo",
  "records": [
    {
      "doc_id": 0,
      "path": "docs/auth.md",
      "offset": 0,
      "length": 512,
      "text": "Auth middleware validates JWT..."
    }
  ]
}
```

Payload format in `.lv` is unchanged logvec v0:

```text
u32 dim | f32[dim] | u64 doc_id
```

Metadata lives only in the manifest; `doc_id` is the join key.

## Checks

Offline CI (no Ollama):

```sh
scripts/check_ragbox.sh
```

Optional live smoke (requires Ollama + model):

```sh
scripts/check_ragbox_live.sh
```

Fixtures: `fasm/tests/ragbox/fixtures/` (dim=4 synthetic index + `tiny-repo/`).

## Not in v0

- logbus ingest pipeline
- PDF / HTML parsing
- ANN / HNSW
- MCP server
- Homebrew formula (see `Formula/fscan.rb` as template for v0.1)

## Related

- [`docs/logvec.md`](logvec.md) — `.lv` format and logvec hosts
- [`fasm/apps/logvec/`](../../fasm/apps/logvec/) — C++ logvec host reused by ragbox
