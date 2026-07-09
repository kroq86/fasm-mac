# fasm-mac

**ragbox is the flagship tool here:** local-first codebase memory for AI
agents. Build a searchable semantic snapshot of your repo, query it from the
terminal, and keep it local. Ollama-compatible, single binary, no vector DB
server.

This repository is the engineering monorepo behind ragbox and a small set of
macOS developer tools built on a reusable FASM runtime.

| Tool | Promise |
|------|---------|
| [`ragbox`](docs/ragbox.md) | Local codebase memory for Codex, Claude, Gemini, and other AI agents |
| `machodoctor` | Explain why a macOS binary does not run |
| `fasm-mac` | Assembly/runtime foundation for x86_64 FASM tools on macOS |

Product page: <https://kroq86.github.io/fasm-mac/>

## ragbox quick start

Use ragbox when `ripgrep` is too literal, a vector DB is too much machinery,
and you want a local semantic index your AI agent can query.

```sh
brew tap kroq86/fasm-mac https://github.com/kroq86/fasm-mac
brew install ragbox
brew install ollama
ollama pull nomic-embed-text

arch -x86_64 ragbox build --root . --out memory.lv
arch -x86_64 ragbox search --index memory.lv --query "where is auth handled?" --json
```

On Apple Silicon, ragbox runs through Rosetta (`arch -x86_64`). The index stays
in local files: `memory.lv`, `memory.lv.manifest.json`, optional refresh state,
and optional delta sidecar.

Why not the obvious alternatives?

| Alternative | ragbox difference |
|-------------|-------------------|
| `ripgrep` | semantic search, not lexical search |
| vector DB server | copyable file snapshot, not a running service |
| RAG platform | local CLI for repo memory, not a web platform |

More: [`docs/ragbox.md`](docs/ragbox.md). Release check:
`scripts/check_ragbox_release.sh`.

## fasm-mac foundation

fasm-mac is also an experimental macOS bridge for **flat assembler classic
1.73.35**. The goal is practical CLI compatibility for small x86_64 fasm
programs on macOS:

```sh
fasm file.asm
./file
```

On Apple Silicon this runs through Rosetta. This is not a native arm64 rewrite
of fasm classic.

## What This Is

fasm classic does not contain a Mach-O formatter. This project keeps the
upstream fasm compiler core and adds a macOS pipeline around it:

1. fasm emits ELF64 executable or ELF64 relocatable object output.
2. `fasm/tools/elf64_to_macho64.py` converts the supported ELF layout into
   Mach-O64.
3. macOS runs or links the resulting x86_64 Mach-O file.

That means this repository gives you a usable macOS command, not a new
upstream `format Mach-O` directive inside fasm.

## Install

### Homebrew (recommended)

```sh
brew tap kroq86/fasm-mac https://github.com/kroq86/fasm-mac
brew install fasm-mac
```

Then use `fasm` from anywhere:

```sh
fasm hello.asm
arch -x86_64 ./hello
```

On Apple Silicon, output binaries are x86_64 Mach-O and run through Rosetta.

Upgrade or remove:

```sh
brew upgrade fasm-mac
brew uninstall fasm-mac
```

### Manual install

```sh
./install.sh
```

The installer creates:

```text
~/.local/bin/fasm -> <repo>/bin/fasm
```

Make sure `~/.local/bin` is in your `PATH`:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

## CLI

Build a runnable Mach-O executable, using the source basename as output:

```sh
fasm fasm/basic/fib.asm
./fasm/basic/fib
```

Build and run:

```sh
fasm run fasm/basic/fib.asm
```

Explicit modes:

```sh
fasm --emit=macho file.asm [output]       # Mach-O executable
fasm --emit=elf file.asm [output]         # original ELF output
fasm --emit=macho-obj file.asm object.o   # Mach-O object for clang/dylib
```

Directly call the bundled host fasm:

```sh
fasm host <args...>
```

## macOS Includes

Runnable examples should include the platform layer instead of hardcoding Linux
syscall numbers:

```asm
format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable
entry start

start:
    write_file STDOUT, msg, msg_len
    exit 0

segment readable writeable
msg db "hello", 10
msg_len = $ - msg
```

The shim passes:

- `TARGET_OS=macos` for Mach-O output
- `TARGET_OS=linux` for `--emit=elf`

`platform.inc` currently provides `write_file`, `read_file`, `open_file`,
`close_file`, `exit`, syscall helper macros, and the Darwin constants needed by
the reusable file, socket, stat, and directory-walk helpers.

## Standard Library Includes

Platform-agnostic helpers for `format ELF64 executable 3`. All assemble
without data-segment declarations and follow the System V AMD64 ABI.

| File | What it provides |
|------|-----------------|
| [`fasm/core/for.inc`](fasm/core/for.inc) | `for reg, lo, hi` / `endfor reg` — nestable register-based counted loop |
| [`fasm/core/control.inc`](fasm/core/control.inc) | `_if/_else/_endif`, `_while/_endw`, `_repeat/_until` structured control flow |
| [`fasm/core/mem.inc`](fasm/core/mem.inc) | `memcpy`, `memset`, `memcmp`, `memmove`, `memxor` |
| [`fasm/core/base64.inc`](fasm/core/base64.inc) | `base64_encode(rdi,rsi,rdx)→rax`, `base64_decode(rdi,rsi,rdx)→rax` |
| [`fasm/core/math_fp.inc`](fasm/core/math_fp.inc) | `fp_isnan`, `fp_isinfinite`, `fp_isfinite`, `fp_floor`, `fp_ceil`, `fp_fmod`, `fp_frexp` |
| [`fasm/core/eml.inc`](fasm/core/eml.inc) | `lb_eml_f64` — EML operator leaf `exp(x)-log(y)` ([arXiv:2603.21852](https://arxiv.org/abs/2603.21852)); export via [`eml_core.asm`](fasm/apps/eml_core.asm) |

Examples:

```sh
fasm fasm/examples/elfexe/forloop64.asm /tmp/forloop64
fasm fasm/examples/elfexe/control64.asm /tmp/control64
```

## Structures and nested dynamic lists

fasm `struc` defines **memory layout** at assemble time (field offsets and
sizes). Runtime growable vectors live in [`fasm/core/dynvec.inc`](fasm/core/dynvec.inc)
with a central [`fasm/core/heap.inc`](fasm/core/heap.inc) allocator and
mark-and-sweep `gc_collect`.

Runnable demo (prints `[[1, 2], [3]]`):

```sh
fasm fasm/examples/nested_list_demo.asm
arch -x86_64 ./fasm/examples/nested_list_demo
```

Smoke test:

```sh
fasm fasm/tests/macos-smoke/nested_list.asm
arch -x86_64 ./fasm/tests/macos-smoke/nested_list
```

LeetCode-style examples covering arrays, hash maps, linked lists, trees,
graphs, stacks, and dynamic programming. Several examples are intentionally
thin wrappers around reusable helpers in `fasm/core`.

```sh
scripts/check_leetcode_examples.sh
```

| Command | Problem / approach | Output |
|---------|--------------------|--------|
| `best_time_to_buy_sell_stock.asm` | LC 121 via `dp.inc` | `5` |
| `binary_search.asm` | LC 704 binary search | `4` |
| `climbing_stairs.asm` | LC 70 via `dp.inc` | `8` |
| `contains_duplicate.asm` | LC 217 int hash map | `1` |
| `first_unique_character.asm` | LC 387 character counts | `0` |
| `house_robber.asm` | LC 198 via `dp.inc` | `4` |
| `implement_queue_using_stacks.asm` | LC 232 via `stack.inc` | `1 1 0` |
| `intersection_of_two_arrays.asm` | LC 349 int hash map | `2` |
| `invert_binary_tree.asm` | LC 226 via `tree.inc` | `4 7 9 6 2 3 1` |
| `linked_list_cycle.asm` | LC 141 via `listnode.inc` | `1` |
| `majority_element.asm` | LC 169 Boyer-Moore vote | `2` |
| `maximum_depth_binary_tree.asm` | LC 104 via `tree.inc` | `3` |
| `maximum_subarray.asm` | LC 53 via `dp.inc` | `6` |
| `merge_sorted_array.asm` | LC 88 two pointers from end | `1 2 2 3 5 6` |
| `merge_two_sorted_lists.asm` | LC 21 via `listnode.inc` | `1 1 2 3 4 4` |
| `middle_of_linked_list.asm` | LC 876 via `listnode.inc` | `3` |
| `missing_number.asm` | LC 268 xor | `2` |
| `move_zeroes.asm` | LC 283 in-place compaction | `1 3 12 0 0` |
| `nested_list_weight_sum.asm` | [LC 339](https://leetcode.com/problems/nested-list-weight-sum/) | `10` |
| `number_of_islands.asm` | LC 200 via `grid.inc` | `1` |
| `palindrome_linked_list.asm` | LC 234 via `listnode.inc` | `1` |
| `remove_duplicates_sorted_array.asm` | LC 26 in-place unique prefix | `5 0 1 2 3 4` |
| `reverse_linked_list.asm` | [LC 206](https://leetcode.com/problems/reverse-linked-list/) | `5 4 3 2 1` |
| `search_insert_position.asm` | LC 35 lower bound | `2` |
| `single_number.asm` | LC 136 xor | `4` |
| `sort_array.asm` | [LC 912](https://leetcode.com/problems/sort-an-array/) | `-1 0 1 2 3` |
| `two_sum.asm` | [LC 1](https://leetcode.com/problems/two-sum/) brute O(n²) | `0 1` |
| `two_sum_hashmap.asm` | LC 1 hash map O(n) | `0 1` |
| `valid_anagram.asm` | LC 242 character counts | `1` |
| `valid_parentheses.asm` | LC 20 via `stack.inc` | `1` |

Core libraries: [`base64.inc`](fasm/core/base64.inc), [`control.inc`](fasm/core/control.inc), [`dirwalk.inc`](fasm/core/dirwalk.inc), [`dp.inc`](fasm/core/dp.inc), [`file.inc`](fasm/core/file.inc), [`for.inc`](fasm/core/for.inc), [`grid.inc`](fasm/core/grid.inc), [`hashmap.inc`](fasm/core/hashmap.inc), [`hashmap_str.inc`](fasm/core/hashmap_str.inc), [`hex.inc`](fasm/core/hex.inc), [`json.inc`](fasm/core/json.inc), [`listnode.inc`](fasm/core/listnode.inc), [`macho.inc`](fasm/core/macho.inc), [`math_fp.inc`](fasm/core/math_fp.inc), [`mem.inc`](fasm/core/mem.inc), [`scanner.inc`](fasm/core/scanner.inc), [`search.inc`](fasm/core/search.inc), [`stack.inc`](fasm/core/stack.inc), [`tree.inc`](fasm/core/tree.inc), [`sort.inc`](fasm/core/sort.inc), [`str.inc`](fasm/core/str.inc), [`repl.inc`](fasm/core/repl.inc), [`oop.inc`](fasm/core/oop.inc) (vtable + methods).

OOP-style demo (`Playlist` with `append` / `print` / `reverse` via vtable):

```sh
fasm fasm/examples/oop_playlist.asm
arch -x86_64 ./fasm/examples/oop_playlist
```

Expected output: `3 1 4 1 5` then `5 1 4 1 3`.

## Mini-Redis REPL

In-memory string-key store with a Redis CLI–style REPL (stdin/stdout, no TCP):

```sh
fasm fasm/apps/miniredis.asm
arch -x86_64 ./fasm/apps/miniredis
```

Commands (v1.1): `PING`, `SET key value`, `GET key`, `EXISTS key`, `DEL key`, `DBSIZE`, `INCR key`, `DECR key`, `MGET k1 k2 …`, `KEYS`, `SAVE path`, `LOAD path`, `QUIT`, `EXIT`.

`SET` stores a signed int64 when the value parses as an integer; otherwise the raw token is stored as a string. `GET` prints ints, strings, or `(nil)` if missing. `INCR`/`DECR` require an int value (or create `1`/`-1` if the key is absent); strings return `ERR wrongtype`. `MGET` needs at least two key tokens (max 16 tokens per line). `KEYS` lists all keys (no pattern filter in v1). `SAVE`/`LOAD` use a line-oriented text dump (see below).

Limits: 256-byte lines, ASCII keys/values without spaces in v1, up to 16 tokens per REPL line.

Dump format (`# miniredis v1` header):

```text
# miniredis v1
I user 42
S greeting hello
```

`I` = int value, `S` = string value (remainder of line after the key token).

Example session:

```text
miniredis> SET user 42
OK
miniredis> SET greeting hello
OK
miniredis> GET user
42
miniredis> GET greeting
hello
miniredis> INCR user
OK
miniredis> GET user
43
miniredis> SAVE dump.txt
OK
miniredis> LOAD dump.txt
OK
miniredis> DEL user
1
miniredis> GET user
(nil)
```

Pipe a script:

```sh
printf 'SET a 1\nGET a\nDBSIZE\nQUIT\n' | arch -x86_64 ./fasm/apps/miniredis
```

Smoke tests: `fasm/tests/macos-smoke/str_hash.asm`, `hashmap_str.asm`, `repl_ping.asm`, `miniredis_script.asm`, `tcp_echo.asm`.

## fscan

Tiny literal-search CLI inspired by grep/ripgrep. It searches one or more files
for a byte-exact literal and prints matching lines as `path:line:text`.

```sh
fasm fasm/apps/fscan.asm
arch -x86_64 ./fasm/apps/fscan [-c] [-l] [-i] needle file.txt other.txt
```

Homebrew:

```sh
brew install kroq86/fasm-mac/fscan
fscan needle file.txt
```

Smoke test:

```sh
scripts/check_fscan.sh
```

## hexpeek

Tiny native hex dump CLI for peeking at file bytes. It demonstrates the
reusable [`hex.inc`](fasm/core/hex.inc) formatter plus chunked file reads.

```sh
fasm fasm/apps/hexpeek.asm
arch -x86_64 ./fasm/apps/hexpeek [-n bytes] [-s skip] file.bin
```

Homebrew:

```sh
brew install kroq86/fasm-mac/hexpeek
hexpeek -n 64 file.bin
```

Release packaging:

```sh
scripts/build-hexpeek-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_hexpeek.sh
```

## raymaze

Tiny native raylib raycaster maze game. It is an original Doom-inspired mini
game, not a Doom port: it does not ship Doom code, WADs, maps, sprites, sounds,
names, or trademarks. It demonstrates the reusable fixed-point
[`raycast.inc`](fasm/core/raycast.inc) helpers plus the
[`ccall64.inc`](fasm/core/ccall64.inc) C ABI bridge for Mach-O objects.

```sh
brew install raylib
fasm --emit=macho-obj fasm/apps/raymaze.asm /tmp/raymaze.o
clang -arch x86_64 /tmp/raymaze.o $(pkg-config --cflags --libs raylib) -o raymaze
arch -x86_64 ./raymaze
```

Because current fasm-mac output is x86_64-only, the linked raylib must also be
x86_64. On Apple Silicon, the default `/opt/homebrew` raylib bottle is arm64;
use an Intel/Rosetta Homebrew raylib install for a real windowed build.

Snapshot mode for deterministic checks:

```sh
arch -x86_64 ./raymaze --snapshot snapshot.ppm
```

Release packaging:

```sh
scripts/build-raymaze-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_raymaze.sh
```

## httpmini

Single-threaded concurrent static HTTP server for macOS x86_64. It uses the
green-thread scheduler, `kqueue`, and nonblocking sockets: one slow or partial
client does not block other clients. Successful `GET` responses for regular
files use macOS `sendfile`.

```sh
fasm fasm/apps/httpmini.asm httpmini
arch -x86_64 ./httpmini --root ./public --port 8080 --bind 127.0.0.1
```

V1 serves local regular files with `GET` and `HEAD`, closes each connection
after one response, serves `/index.html` for `/`, writes a simple access log to
stderr, and rejects directories, symlinks, `%` escapes, backslashes, and `..`
path components.

Release packaging:

```sh
scripts/build-httpmini-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_httpmini.sh
```

## logbus

Kafka-like local durable append-only message broker for macOS x86_64. It uses
the green-thread scheduler, `kqueue`, nonblocking sockets, and a length-prefixed
record log with a message-offset index and CRC32C per record. Topic data is
stored as rotated base-offset segments plus a global offset index; tune segment
size with `--segment-bytes N`. Accepted `PRODUCE` writes are fsync-backed, and
restart recovery trims segment/index tails back to the committed global offset
index. v1.3 uses a breaking storage/raw fetch format:
`[u32_len][u32_crc32c][payload]`.

```sh
fasm fasm/apps/logbus.asm logbus
arch -x86_64 ./logbus --dir ./data --port 9092 --bind 127.0.0.1
```

V1 commands use a RESP-like protocol: `PING`, `PRODUCE topic payload`,
`FETCH topic offset max_bytes`, `FETCHBATCH topic offset max_bytes`,
`COMMIT group topic offset`, `OFFSET group topic`, and `QUIT`.
`FETCHBATCH` returns raw `[u32_len][u32_crc32c][payload]...` log bytes via macOS
`sendfile`. This is a local single-partition broker, not a distributed Kafka
replacement.

Release packaging:

```sh
scripts/build-logbus-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_logbus.sh
```

## logvec

Experimental brew-worthy tool: **batch snapshot** index builder plus exact
cosine top-k search. logbus stays dumb; FASM owns f32 dot/norm/top-k only;
Zig wires protocol, files, ingest, and doc_id mapping (C++ host available in
`fasm/apps/logvec/`). v0 metric: cosine similarity
(`score = dot / (norm(q)*norm(v))`, higher is better). `build-index`
is one-shot — it does not tail topics. Spec: [`docs/logvec.md`](docs/logvec.md).
System form (Level 4): [`docs/system_form.md`](docs/system_form.md).

```sh
fasm --emit=macho-obj fasm/apps/logvec_core.asm logvec_core.o
zig build-exe fasm/apps/logvec.zig logvec_core.o \
  -target x86_64-macos -mcpu=baseline -O ReleaseSafe -femit-bin=logvec
arch -x86_64 ./logvec search --index index.lv --query query.bin --top 5
```

C++ host (same CLI, binary name `logvec_cpp`):

```sh
fasm --emit=macho-obj fasm/apps/logvec_core.asm logvec_core.o
clang++ -std=c++20 -O2 -arch x86_64 \
  fasm/apps/logvec/logvec.cpp logvec_core.o -o logvec_cpp
arch -x86_64 ./logvec_cpp search --index index.lv --query query.bin --top 5
```

Smoke test:

```sh
scripts/check_logvec.sh
scripts/check_logvec_cpp.sh
scripts/bench_logvec.sh   # in-process top-k regression (1k/10k/100k × dim=768)
scripts/bench_perf.sh     # layered dot/topk/search/io + parallel + ragbox breakdown
```

v0.2 adds layered bench (`--layer dot|topk|search|io`), scalar vs AVX2 dot A/B,
parallel exact search (1–4 threads), and unit-vector top-k fast path. Exact
linear scan — ~4.5 ms for 10k×768 single-thread, ~1.4 ms with 4 threads (see
`docs/logvec.md`). Not ANN; agent-scale snapshots only.

## ragbox

Local-first codebase memory for AI agents: chunk a repo, embed via Ollama,
build a copyable `.lv` index + JSON manifest, and search it from the terminal.
One x86_64 binary — no Python venv, no vector DB server, no web platform. More:
[`docs/ragbox.md`](docs/ragbox.md). System form (Level 4):
[`docs/system_form.md`](docs/system_form.md).

Homebrew:

```sh
brew tap kroq86/fasm-mac https://github.com/kroq86/fasm-mac
brew install ragbox
brew install ollama
ollama pull nomic-embed-text
arch -x86_64 ragbox doctor --skip-ollama
arch -x86_64 ragbox build --root ./repo --out memory.lv
arch -x86_64 ragbox refresh --root ./repo --index memory.lv
arch -x86_64 ragbox search --index memory.lv --query "where is auth handled?" --json
```

Why not the obvious alternatives?

| Alternative | ragbox difference |
|-------------|-------------------|
| `ripgrep` | semantic search, not lexical search |
| vector DB server | copyable file snapshot, not a running service |
| RAG platform | local CLI for repo memory, not a web platform |

Manual build (from source):

```sh
fasm --emit=macho-obj fasm/apps/logvec_core.asm logvec_core.o
clang++ -std=c++20 -O2 -arch x86_64 -pthread \
  fasm/apps/ragbox/ragbox.cpp logvec_core.o -o ragbox
arch -x86_64 ./ragbox build --root ./repo --out memory.lv
arch -x86_64 ./ragbox refresh --root ./repo --index memory.lv
arch -x86_64 ./ragbox search --index memory.lv --query "auth middleware" --json
```

Release packaging:

```sh
scripts/build-ragbox-release.sh 0.3.0
scripts/check_ragbox_release.sh
```

Smoke test:

```sh
scripts/check_ragbox.sh
scripts/check_ragbox_release.sh
```

Optional live check (Ollama required):

```sh
scripts/check_ragbox_live.sh
```

## macdbg

AI-native LLDB snapshot debugger for macOS binaries. Its useful surface is the
CLI report mode: run a target once under LLDB batch mode and write a structured
JSON file with status, exit code, crash signal, registers, backtrace,
disassembly near the program counter, stack memory, Mach-O summary, and an
escaped LLDB output tail.

```sh
fasm --emit=macho-obj fasm/apps/macdbg.asm /tmp/macdbg.o
clang -arch x86_64 /tmp/macdbg.o $(pkg-config --cflags --libs raylib) -o macdbg
arch -x86_64 ./macdbg --snapshot ./program report.json
arch -x86_64 ./macdbg --snapshot --args ./program arg1 arg2 -- report.json
```

The report is intended to be consumed by tools and agents:

```json
{
  "tool": "macdbg",
  "mode": "snapshot",
  "status": "exited",
  "exit_code": 0,
  "signal": null,
  "registers": {"rip": "0x..."},
  "backtrace": [],
  "disasm": [],
  "stack_memory": []
}
```

There is also an experimental raylib snapshot viewer:

```sh
arch -x86_64 ./macdbg --ui ./program
arch -x86_64 ./macdbg --ui --args ./program arg1 arg2
```

The UI is not a live step debugger yet. Press `R` to rerun the LLDB snapshot,
`J` to toggle the raw JSON/LLDB tail view, and `Esc` to quit. Because current
fasm-mac output is x86_64-only, the linked raylib must also be x86_64, just
like `raymaze`.

Release packaging:

```sh
scripts/build-macdbg-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_macdbg.sh
```

## pathsum

Tiny native recursive directory counter. It demonstrates the reusable
[`dirwalk.inc`](fasm/core/dirwalk.inc) API for directory traversal, file type
detection, and `stat64` size reads.

```sh
fasm fasm/apps/pathsum.asm
arch -x86_64 ./fasm/apps/pathsum [dir]
```

Output:

```text
files 2
dirs 1
bytes 1234
```

Homebrew:

```sh
brew install kroq86/fasm-mac/pathsum
pathsum .
```

Release packaging:

```sh
scripts/build-pathsum-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_pathsum.sh
```

## setdb

Tiny pure set-theoretic database CLI. A database is a `universe.db` directory
with an append-only operation log; the model is only sets and binary relations:
no SQL, no NULL, no duplicate rows, and no multisets.
Query results use the reusable [`arena.inc`](fasm/core/arena.inc) region
allocator: a command allocates temporary set/relation results in one arena, then
the process exits and the whole invocation lifetime is reclaimed at once.

```sh
fasm fasm/apps/setdb.asm setdb
arch -x86_64 ./setdb new universe.db
arch -x86_64 ./setdb add universe.db users alice bob carol
arch -x86_64 ./setdb add universe.db admins alice
arch -x86_64 ./setdb relation universe.db follows alice bob
arch -x86_64 ./setdb relation universe.db follows bob carol
arch -x86_64 ./setdb relation universe.db follows carol dana
arch -x86_64 ./setdb diff universe.db users admins
arch -x86_64 ./setdb select universe.db follows first alice
arch -x86_64 ./setdb join universe.db follows follows
arch -x86_64 ./setdb domain universe.db follows
arch -x86_64 ./setdb range universe.db follows
arch -x86_64 ./setdb inverse universe.db follows
arch -x86_64 ./setdb transitive-closure universe.db follows
```

Output examples:

```text
bob
carol
```

```text
(alice,carol)
(bob,dana)
```

```text
(alice,bob)
(alice,carol)
(alice,dana)
(bob,carol)
(bob,dana)
(carol,dana)
```

Homebrew:

```sh
brew install kroq86/fasm-mac/setdb
setdb new universe.db
```

Release packaging:

```sh
scripts/build-setdb-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_setdb.sh
```

## machodoctor

Standalone macOS Mach-O inspector intended to ship as its own Homebrew formula
and ready-to-run binary. It is separate from `fasm-mac`; FASM is only used to
build release artifacts. Universal/fat Mach-O files are supported by inspecting
their x86_64 slice in v1.

```sh
fasm fasm/apps/machodoctor.asm
arch -x86_64 ./fasm/apps/machodoctor ./fasm/apps/machodoctor
arch -x86_64 ./fasm/apps/machodoctor --json ./fasm/apps/machodoctor
arch -x86_64 ./fasm/apps/machodoctor --deps ./fasm/apps/machodoctor
arch -x86_64 ./fasm/apps/machodoctor --check ./fasm/apps/machodoctor
```

Release packaging:

```sh
scripts/build-machodoctor-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_machodoctor.sh
```

## shipcheck

Standalone local release QA checker for Homebrew-style binary products. It
validates one formula, one release tarball, and one built Mach-O binary before
uploading release assets.

```sh
fasm fasm/apps/shipcheck.asm
arch -x86_64 ./fasm/apps/shipcheck Formula/hexpeek.rb dist/hexpeek-0.1.0-macos-x86_64.tar.gz ./hexpeek
```

Checks include formula `url` basename, `version`, `sha256`, `bin.install`, the
tarball filename shape, and whether the binary is an x86_64 Mach-O executable.
Tar archive contents remain a shell-script smoke check in v1.

Homebrew:

```sh
brew install kroq86/fasm-mac/shipcheck
shipcheck Formula/hexpeek.rb dist/hexpeek-0.1.0-macos-x86_64.tar.gz ./hexpeek
```

Release packaging:

```sh
scripts/build-shipcheck-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_shipcheck.sh
```

## logknife

Tiny structured log slicer for plain logs and JSONL. It is the first consumer
of the reusable line scanner in `fasm/core/scanner.inc` and the minimal JSONL
field matcher in `fasm/core/json.inc`.

```sh
fasm fasm/apps/logknife.asm
arch -x86_64 ./fasm/apps/logknife --contains timeout app.log
arch -x86_64 ./fasm/apps/logknife --jsonl --level error --count app.jsonl
arch -x86_64 ./fasm/apps/logknife --jsonl --field status=500 app.jsonl
```

Release packaging:

```sh
scripts/build-logknife-release.sh 0.1.0
```

Smoke test:

```sh
scripts/check_logknife.sh
```

## Mini-Redis TCP server (v0)

RESP/TCP server on port **6379** (macOS x86_64, one client at a time):

```sh
fasm fasm/apps/miniredis_server.asm
arch -x86_64 ./fasm/apps/miniredis_server
```

Commands: `PING`, `SET`, `GET`, `QUIT` (same semantics as REPL for values: int if parseable, else string).

Test with `redis-cli` (x86_64/Rosetta) or `nc`:

```sh
redis-cli -p 6379 PING
redis-cli -p 6379 SET foo bar
redis-cli -p 6379 GET foo
redis-cli -p 6379 QUIT
```

Without redis-cli:

```sh
printf '*1\r\n$4\r\nPING\r\n' | nc localhost 6379
```

TCP echo smoke (port 9999):

```sh
fasm fasm/tests/macos-smoke/tcp_echo.asm
arch -x86_64 ./fasm/tests/macos-smoke/tcp_echo &
printf 'hi' | nc localhost 9999
```

```sh
scripts/check_leetcode_examples.sh
```

## Shared Libraries

Build a Mach-O object and link it into a `.dylib`:

```sh
fasm --emit=macho-obj add.asm add.o
clang -arch x86_64 -dynamiclib wrapper.c add.o -o mylib.dylib
```

On Apple Silicon, Python `ctypes` examples need an x86_64/Rosetta Python to
load that dylib. An arm64 Python cannot load an x86_64 library.

## Current Limits

- Output is x86_64 only.
- Native arm64 fasm classic is out of scope.
- The executable converter supports simple ELF64 executable layouts.
- The object converter supports simple allocatable `.text`, `.data`, `.bss`
  sections and symbols.
- ELF relocations in object files are rejected for now.
- fasm classic still does not understand `format Mach-O`.
- Coroutines and other callback/stack-switching examples need separate ABI
  review before being called supported on macOS.

## Build The Host

The checked-in bridge expects the macOS x64 host binary at:

```text
fasm/build/out/macos-x64/fasm-macos-x64
```

Rebuild it with:

```sh
./fasm/build/macos-x64.sh
```

Verify:

```sh
file fasm/build/out/macos-x64/fasm-macos-x64
arch -x86_64 fasm/build/out/macos-x64/fasm-macos-x64
```

## Smoke Tests

```sh
fasm fasm/basic/fib.asm
arch -x86_64 ./fasm/basic/fib

fasm --emit=elf fasm/basic/fib.asm /tmp/fib.elf
file /tmp/fib.elf

fasm --emit=macho-obj /path/to/add.asm /tmp/add.o
file /tmp/add.o
```

## Upstream

- Original project: <https://flatassembler.net/>
- Upstream archive used here: `fasm-1.73.35.tgz`

The original license is kept at [fasm/license.txt](fasm/license.txt).

## References

- <https://flatassembler.net/> — flat assembler (fasm1) by Tomasz Grysztar
- <https://2ton.com.au/> — HeavyThing x86_64 FASM library by Jeff Marrison (GPLv2+); algorithms in `mem.inc`, `base64.inc`, `math_fp.inc` adapted from here
- <https://www.agner.org/optimize/> — Agner Fog's optimization guides and asmlib; small-copy and memcmp patterns
- <https://board.flatassembler.net/> — FASM community board; macro techniques and fasm1 idioms
- <https://github.com/tgrysztar/fasmg> — fasmg by Tomasz Grysztar; `control.inc` structured-flow macros inspired by `packages/x86/include/macro/if.inc`
