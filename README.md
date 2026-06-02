# fasm-mac

Experimental macOS bridge for **flat assembler classic 1.73.35**.

The goal is practical CLI compatibility for small x86_64 fasm programs on
macOS:

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

Core libraries: [`dirwalk.inc`](fasm/core/dirwalk.inc), [`dp.inc`](fasm/core/dp.inc), [`file.inc`](fasm/core/file.inc), [`grid.inc`](fasm/core/grid.inc), [`hashmap.inc`](fasm/core/hashmap.inc), [`hashmap_str.inc`](fasm/core/hashmap_str.inc), [`hex.inc`](fasm/core/hex.inc), [`json.inc`](fasm/core/json.inc), [`listnode.inc`](fasm/core/listnode.inc), [`macho.inc`](fasm/core/macho.inc), [`scanner.inc`](fasm/core/scanner.inc), [`search.inc`](fasm/core/search.inc), [`stack.inc`](fasm/core/stack.inc), [`tree.inc`](fasm/core/tree.inc), [`sort.inc`](fasm/core/sort.inc), [`str.inc`](fasm/core/str.inc), [`repl.inc`](fasm/core/repl.inc), [`oop.inc`](fasm/core/oop.inc) (vtable + methods).

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
