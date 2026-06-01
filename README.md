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
brew tap kroq86/fasm-mac
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
