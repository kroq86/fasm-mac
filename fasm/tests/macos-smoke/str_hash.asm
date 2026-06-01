; Smoke: str_hash("hello") -> fixed value on stdout.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"

entry start

start:
	lea	rdi, [test_str]
	call	str_hash
	call	print_int_nl
	exit EXIT_SUCCESS

segment readable writeable

test_str db 'hello', 0

include "fasm/core/runtime_bss.inc"
runtime_print_bss
