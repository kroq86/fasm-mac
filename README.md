# fasm classic for macOS

Experimental macOS host port of **flat assembler classic 1.73.35**.

The practical goal is simple: run fasm on an Apple Silicon Mac, assemble a
small x86_64 program, wrap it as Mach-O, and execute it locally through
Rosetta.

This repository is intentionally not a rewrite of fasm. The compiler core is
still upstream fasm classic, written in x86/x86_64 assembly. The macOS support
added here is a host/runtime bridge around that core.

## What Works

- Builds a macOS x86_64 fasm host binary:

  ```text
  fasm/build/out/macos-x64/fasm-macos-x64
  ```

- Runs the fasm host on Apple Silicon through Rosetta:

  ```sh
  arch -x86_64 fasm/build/out/macos-x64/fasm-macos-x64
  ```

- Assembles normal fasm outputs on macOS:

  - `format binary`
  - `format ELF64`
  - `format PE64`
  - `format MS64 COFF`

- Wraps suitable ELF64 executable output into static Mach-O64 and runs it:

  ```sh
  ./dotfasm-mac run fasm/basic/fib.asm fib-run
  ```

  Expected output:

  ```text
  flat assembler  version 1.73.35  (16384 kilobytes memory, x64)
  3 passes, 295 bytes.
  1
  1
  2
  3
  5
  8
  13
  21
  34
  55
  ```

## Important Limits

This is a **macOS host port**, not a full Mach-O formatter inside fasm classic.

That means:

- fasm itself runs on macOS.
- fasm can still emit Linux ELF, PE, COFF, and flat binary outputs.
- `./dotfasm-mac macho` and `./dotfasm-mac run` can wrap an ELF64 executable
  layout into Mach-O64.
- Your program code must use **Darwin syscall numbers** if you want it to run
  on macOS.

Linux syscall examples like this will compile, but will not run correctly as
macOS programs:

```asm
SYS_write equ 1
SYS_exit  equ 60
```

For Darwin x86_64, use:

```asm
SYS_write equ 02000004h
SYS_exit  equ 02000001h
```

The included example [fasm/basic/fib.asm](fasm/basic/fib.asm) already uses
Darwin syscall numbers.

## Requirements

- macOS on Apple Silicon
- Rosetta 2
- Docker or Colima with `linux/amd64` support
- Python 3

Check Rosetta:

```sh
arch -x86_64 /usr/bin/true
```

If this fails, install Rosetta:

```sh
softwareupdate --install-rosetta
```

## Build

```sh
./fasm/build/macos-x64.sh
```

The build script:

1. Runs bundled upstream `fasm.x64` inside a Linux amd64 Docker container.
2. Assembles [fasm/source/macos/x64/fasm.asm](fasm/source/macos/x64/fasm.asm)
   into an ELF64 bootstrap image.
3. Converts that image into a static Mach-O64 executable with
   [fasm/tools/elf64_to_macho64.py](fasm/tools/elf64_to_macho64.py).

Output:

```text
fasm/build/out/macos-x64/fasm-macos-x64
```

Verify:

```sh
file fasm/build/out/macos-x64/fasm-macos-x64
arch -x86_64 fasm/build/out/macos-x64/fasm-macos-x64
```

You should see:

```text
Mach-O 64-bit executable x86_64
flat assembler  version 1.73.35
usage: fasm <source> [output]
```

## Usage

Compile only:

```sh
./dotfasm-mac fasm/basic/fib.asm fib-elf
file fib-elf
```

This preserves the source output format. For `fib.asm`, that means the result
is still ELF64:

```text
ELF 64-bit LSB executable, x86-64
```

Compile and wrap as Mach-O:

```sh
./dotfasm-mac macho fasm/basic/fib.asm fib-mac
file fib-mac
arch -x86_64 ./fib-mac
```

Compile, wrap, and run:

```sh
./dotfasm-mac run fasm/basic/fib.asm fib-run
```

## Smoke Tests

```sh
BIN=fasm/build/out/macos-x64/fasm-macos-x64
OUT=fasm/build/out/macos-x64/smoke
mkdir -p "$OUT"

for src in binary elf64 coff pe64; do
  arch -x86_64 "$BIN" "fasm/tests/macos-smoke/$src.asm" "$OUT/$src.out"
done

file "$OUT"/*
```

Known smoke result:

```text
binary.out: ASCII text
coff.out:   data
elf64.out:  ELF 64-bit LSB executable, x86-64
pe64.out:   MS-DOS executable
```

`binary` and `elf64` outputs match upstream Linux fasm byte-for-byte. PE and
COFF outputs are valid but include timestamp fields, so byte-for-byte hashes
can differ from Linux.

## Repository Layout

```text
dotfasm-mac                         wrapper for compile/macho/run modes
fasm/source/macos/x64/              macOS x64 host source
fasm/tools/elf64_to_macho64.py      ELF64 load-segment to Mach-O64 wrapper
fasm/build/macos-x64.sh             reproducible build script
fasm/basic/fib.asm                  runnable Darwin syscall example
fasm/tests/macos-smoke/             smoke test inputs
```

## How The Mach-O Wrapper Works

fasm classic already knows how to produce ELF64 executable images. The macOS
wrapper path uses that existing formatter as a low-level layout engine:

1. Assemble source as ELF64 executable.
2. Read ELF64 `PT_LOAD` segments.
3. Create a static Mach-O64 executable with matching low virtual addresses.
4. Point `LC_UNIXTHREAD` at the ELF entry point.
5. Run it through Rosetta.

This is deliberately narrow, but useful: it gets small syscall-based x86_64
programs running on macOS without adding a full Mach-O formatter to fasm
classic.

## Current Known Issue

PE/COFF smoke inputs produce valid files with exit code `0`, but the macOS host
currently does not print the final `passes/bytes` summary line for those two
formats. Binary and ELF64 print the full summary.

## Upstream

Original flat assembler project:

- <https://flatassembler.net/>
- upstream archive used here: `fasm-1.73.35.tgz`

This repository keeps the original license file at [fasm/license.txt](fasm/license.txt).
