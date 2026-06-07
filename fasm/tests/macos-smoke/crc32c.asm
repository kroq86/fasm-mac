; Smoke: software CRC32C known vectors.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/crc32c.inc"

entry start

start:
	lea	rdi, [empty]
	xor	rsi, rsi
	call	crc32c_compute
	cmp	eax, 000000000h
	jne	fail

	lea	rdi, [hello]
	mov	rsi, hello_len
	call	crc32c_compute
	cmp	eax, 09A71BB4Ch
	jne	fail

	lea	rdi, [digits]
	mov	rsi, digits_len
	call	crc32c_compute
	cmp	eax, 0E3069283h
	jne	fail

	exit	EXIT_SUCCESS

fail:
	exit	EXIT_FAILURE

segment readable writeable

empty db 0
hello db 'hello'
hello_len = $ - hello
digits db '123456789'
digits_len = $ - digits
