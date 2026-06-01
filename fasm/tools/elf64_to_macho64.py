#!/usr/bin/env python3
"""Wrap a low-address ELF64 executable image as a static Mach-O64 executable.

This is intentionally narrow: it is a bootstrap helper for the fasm macOS/x64
host port. The fasm source is still assembled by upstream fasm as ELF64 to get
the same low virtual addresses the compiler core expects; this script maps the
ELF PT_LOAD payloads into a Mach-O executable and points LC_UNIXTHREAD at the
ELF entry point.
"""

from __future__ import annotations

import argparse
import os
import stat
import struct
from pathlib import Path

PAGE = 0x1000
MH_MAGIC_64 = 0xFEEDFACF
CPU_TYPE_X86_64 = 0x01000007
CPU_SUBTYPE_X86_64_ALL = 3
MH_EXECUTE = 2
MH_NOUNDEFS = 1
LC_SEGMENT_64 = 0x19
LC_UNIXTHREAD = 0x5
X86_THREAD_STATE64 = 4
X86_THREAD_STATE64_COUNT = 42
VM_PROT_READ = 1
VM_PROT_WRITE = 2
VM_PROT_EXECUTE = 4
PT_LOAD = 1


def align(value: int, boundary: int = PAGE) -> int:
    return (value + boundary - 1) & ~(boundary - 1)


def cstr16(name: str) -> bytes:
    raw = name.encode("ascii")
    if len(raw) > 16:
        raise ValueError(f"segment name too long: {name}")
    return raw.ljust(16, b"\0")


def parse_elf64(path: Path) -> tuple[int, list[dict[str, int]], bytes]:
    data = path.read_bytes()
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise ValueError(f"{path} is not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise ValueError("expected little-endian ELF64")

    (
        _ident,
        _etype,
        machine,
        _version,
        entry,
        phoff,
        _shoff,
        _flags,
        _ehsize,
        phentsize,
        phnum,
        *_rest,
    ) = struct.unpack_from("<16sHHIQQQIHHHHHH", data, 0)
    if machine != 0x3E:
        raise ValueError(f"expected x86_64 ELF machine, got {machine:#x}")
    if phentsize != 56:
        raise ValueError(f"unexpected ELF64 program header size: {phentsize}")

    loads: list[dict[str, int]] = []
    for index in range(phnum):
        off = phoff + index * phentsize
        p_type, p_flags, p_offset, p_vaddr, _p_paddr, p_filesz, p_memsz, p_align = struct.unpack_from(
            "<IIQQQQQQ", data, off
        )
        if p_type != PT_LOAD:
            continue
        if p_filesz > p_memsz:
            raise ValueError("PT_LOAD has filesz > memsz")
        if p_offset + p_filesz > len(data):
            raise ValueError("PT_LOAD extends past EOF")
        loads.append(
            {
                "flags": p_flags,
                "offset": p_offset,
                "vaddr": p_vaddr,
                "filesz": p_filesz,
                "memsz": p_memsz,
                "align": p_align,
            }
        )

    if not loads:
        raise ValueError("ELF contains no PT_LOAD segments")
    return entry, loads, data


def prot_from_elf(flags: int) -> int:
    prot = 0
    if flags & 4:
        prot |= VM_PROT_READ
    if flags & 2:
        prot |= VM_PROT_WRITE
    if flags & 1:
        prot |= VM_PROT_EXECUTE
    return prot


def segment_command(name: str, vmaddr: int, vmsize: int, fileoff: int, filesize: int, prot: int) -> bytes:
    return struct.pack(
        "<II16sQQQQIIII",
        LC_SEGMENT_64,
        72,
        cstr16(name),
        vmaddr,
        align(vmsize),
        fileoff,
        filesize,
        prot,
        prot,
        0,
        0,
    )


def unixthread_command(entry: int) -> bytes:
    state = [0] * 21
    state[16] = entry
    state[17] = 0x200
    return struct.pack("<IIII", LC_UNIXTHREAD, 184, X86_THREAD_STATE64, X86_THREAD_STATE64_COUNT) + struct.pack(
        "<21Q", *state
    )


def build_macho(entry: int, loads: list[dict[str, int]], elf_data: bytes) -> bytes:
    commands: list[bytes] = [
        segment_command("__PAGEZERO", 0, PAGE, 0, 0, 0),
        segment_command("__TEXT", PAGE, PAGE, 0, PAGE, VM_PROT_READ | VM_PROT_EXECUTE),
    ]

    file_cursor = PAGE
    payloads: list[tuple[int, bytes]] = []
    for index, load in enumerate(loads):
        file_cursor = align(file_cursor)
        payload = elf_data[load["offset"] : load["offset"] + load["filesz"]]
        prot = prot_from_elf(load["flags"])
        commands.append(
            segment_command(
                f"__FASM{index}",
                load["vaddr"],
                load["memsz"],
                file_cursor,
                len(payload),
                prot,
            )
        )
        payloads.append((file_cursor, payload))
        file_cursor += len(payload)

    commands.append(unixthread_command(entry))
    sizeofcmds = sum(len(command) for command in commands)
    header = struct.pack(
        "<IiiIIII",
        MH_MAGIC_64,
        CPU_TYPE_X86_64,
        CPU_SUBTYPE_X86_64_ALL,
        MH_EXECUTE,
        len(commands),
        sizeofcmds,
        MH_NOUNDEFS,
    ) + struct.pack("<I", 0)

    if len(header) + sizeofcmds > PAGE:
        raise ValueError("Mach-O load commands do not fit in the reserved header page")

    output = bytearray(PAGE)
    output[: len(header)] = header
    pos = len(header)
    for command in commands:
        output[pos : pos + len(command)] = command
        pos += len(command)

    for fileoff, payload in payloads:
        if len(output) < fileoff:
            output.extend(b"\0" * (fileoff - len(output)))
        output[fileoff : fileoff + len(payload)] = payload

    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("macho", type=Path)
    args = parser.parse_args()

    entry, loads, elf_data = parse_elf64(args.elf)
    macho = build_macho(entry, loads, elf_data)
    args.macho.write_bytes(macho)
    mode = args.macho.stat().st_mode
    args.macho.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


if __name__ == "__main__":
    main()
