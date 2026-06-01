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
`close_file`, `exit`, and syscall helper macros for Linux and Darwin x86_64.

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

LeetCode-style examples using the same Vec/Value heap:

| Command | Problem | Output |
|---------|---------|--------|
| `two_sum.asm` | [LC 1](https://leetcode.com/problems/two-sum/) brute O(n²) | `0 1` |
| `two_sum_hashmap.asm` | LC 1 hash map O(n) | `0 1` |
| `nested_list_weight_sum.asm` | [LC 339](https://leetcode.com/problems/nested-list-weight-sum/) | `10` |
| `reverse_linked_list.asm` | [LC 206](https://leetcode.com/problems/reverse-linked-list/) | `5 4 3 2 1` |
| `sort_array.asm` | [LC 912](https://leetcode.com/problems/sort-an-array/) | `-1 0 1 2 3` |

Core libraries: [`hashmap.inc`](fasm/core/hashmap.inc), [`hashmap_str.inc`](fasm/core/hashmap_str.inc), [`listnode.inc`](fasm/core/listnode.inc), [`sort.inc`](fasm/core/sort.inc), [`str.inc`](fasm/core/str.inc), [`repl.inc`](fasm/core/repl.inc), [`oop.inc`](fasm/core/oop.inc) (vtable + methods).

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
fasm fasm/examples/leetcode/two_sum_hashmap.asm
arch -x86_64 ./fasm/examples/leetcode/two_sum_hashmap

fasm fasm/examples/leetcode/reverse_linked_list.asm
arch -x86_64 ./fasm/examples/leetcode/reverse_linked_list

fasm fasm/examples/leetcode/sort_array.asm
arch -x86_64 ./fasm/examples/leetcode/sort_array
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
