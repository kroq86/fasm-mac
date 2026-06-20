# ragbox

**Local-first codebase memory for AI agents.** Build a searchable semantic
snapshot of your repo, query it from the terminal, and keep it local.
Ollama-compatible, single binary, no vector DB server.

Use ragbox when Codex, Claude, Gemini, or another agent needs fast context from
your repository and plain keyword search is not enough.

```sh
brew tap kroq86/fasm-mac https://github.com/kroq86/fasm-mac
brew install ragbox
brew install ollama
ollama pull nomic-embed-text

arch -x86_64 ragbox build --root . --out memory.lv
arch -x86_64 ragbox search --index memory.lv --query "where is auth handled?" --json
```

On Apple Silicon, ragbox runs through Rosetta (`arch -x86_64`). The generated
index is a set of local files you can copy, version, test, and rebuild.

| Alternative | ragbox difference |
|-------------|-------------------|
| `ripgrep` | semantic search, not lexical search |
| vector DB server | copyable file snapshot, not a running service |
| RAG platform | local CLI for repo memory, not a web platform |

## What it builds

ragbox wraps the logvec `.lv` snapshot format with:

- deterministic chunking over text/code files
- Ollama HTTP embeddings (v0 glue — no embedded model)
- a JSON manifest sidecar for path/offset/snippet metadata
- `build`, `refresh`, `search`, and `doctor` commands in one x86_64 binary

FASM owns dot/norm/top-k; C++ owns chunking, HTTP embed glue, mmap search, and
CLI. Zig logvec remains unchanged.

Why ragbox exists in the stack (Level 4):
[`docs/system_form.md`](system_form.md).

## Install (Homebrew)

```sh
brew tap kroq86/fasm-mac https://github.com/kroq86/fasm-mac
brew install ragbox
arch -x86_64 ragbox doctor --skip-ollama
```

Text `build`/`search` requires [Ollama](https://ollama.com) with an embedding
model:

```sh
brew install ollama
ollama pull nomic-embed-text
arch -x86_64 ragbox build --root ./my-repo --out memory.lv
arch -x86_64 ragbox search --index memory.lv --query "auth middleware" --json
```

Release tarball: `scripts/build-ragbox-release.sh 0.3.0` →
`dist/ragbox-0.3.0-macos-x86_64.tar.gz`. Pre-release check:
`scripts/check_ragbox_release.sh`.

## Build

```sh
fasm --emit=macho-obj fasm/apps/logvec_core.asm logvec_core.o
clang++ -std=c++20 -O2 -arch x86_64 -pthread \
  fasm/apps/ragbox/ragbox.cpp logvec_core.o -o ragbox
```

Contributor build from source (same binary as Homebrew release):

On Apple Silicon:

```sh
arch -x86_64 ./ragbox doctor
```

## Quick start from a source build

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
- `memory.lv.state.json` — file hashes + superseded doc_ids (written by `build`)
- `memory.lv.delta` — append-only incremental vectors (written by `refresh`)

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

Writes `memory.lv`, manifest, and `memory.lv.state.json`. Removes any existing
`.delta` sidecar (clean full snapshot).

### refresh

```text
ragbox refresh --root PATH --index PATH [--manifest PATH]
  [--chunk-size 800] [--overlap 100]
  [--ollama http://127.0.0.1:11434] [--model nomic-embed-text]
  [--dry-run]
```

Incremental update after `build`: compares SHA-256 file hashes in
`memory.lv.state.json`, re-embeds only changed/new files, appends vectors to
`memory.lv.delta`, and updates manifest/state. Unchanged files are skipped.
Deleted files supersede their `doc_id`s (filtered at search). Chunk params and
model must match the state file or refresh errors (run full `build` instead).

`--dry-run` reports deleted/changed/added file counts without Ollama or writes.

Workflow:

```sh
arch -x86_64 ragbox build --root ./my-repo --out memory.lv
# edit files under ./my-repo ...
arch -x86_64 ragbox refresh --root ./my-repo --index memory.lv
arch -x86_64 ragbox search --index memory.lv --query "auth middleware" --json
```

Compaction (merge delta into base): run full `build` again.

### search

```text
ragbox search --index PATH --query TEXT [--top 8] [--json]
  [--manifest PATH] [--ollama URL] [--model MODEL]
  [--query-file PATH] [--snippet-len 200]
```

Embeds the query (or reads raw f32 `--query-file` for offline checks), runs
exact cosine top-k over the mmap index (base + optional `.delta` when state
exists), joins manifest records:

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
manifest validation (strict for full build; merged base+delta when state present),
delta/state sidecars.

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

## Performance

ragbox search is **exact linear scan** over a mmap'd logvec index — same path as
`logvec bench`. Typical agent repo (hundreds to low thousands of chunks at
dim=768) completes in single-digit milliseconds in-process on x86_64 with AVX2;
4 threads reach ~1–2 ms at 10k×768.

Ollama embedding dominates `build` and query latency (network + model), not the
FASM top-k pass. See [`docs/logvec.md`](logvec.md) for bench numbers and limits
(512 MB max index size).

Offline perf path (no Ollama):

```sh
ragbox bench --index memory.lv --manifest memory.lv.manifest.json \
  --query-file query.bin --breakdown
```

Breakdown reports `manifest_load_ms`, `search_ms`, `join_ms`, `snippet_ms`.
Lite manifests (no embedded `text`) load snippets from `--root` on demand;
snippet I/O shows up in `snippet_ms`.

```sh
scripts/bench_perf.sh   # includes ragbox lite-manifest bench on fixtures
```

## Not in v0

- logbus ingest pipeline
- PDF / HTML parsing
- ANN / HNSW
- MCP server

## Related

- [`docs/system_form.md`](system_form.md) — Level 4 architecture (truth, projection, invariants)
- [`docs/logvec.md`](logvec.md) — `.lv` format and logvec hosts
- [`fasm/apps/logvec/`](../../fasm/apps/logvec/) — C++ logvec host reused by ragbox
