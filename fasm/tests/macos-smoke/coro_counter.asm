; Smoke: cooperative green threads yield in round-robin order.

format ELF64 executable 3
include "fasm/core/platform.inc"

CORO_MAX_TASKS equ 4
CORO_STACK_SIZE equ 4096

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/coro.inc"
include "fasm/core/runtime_bss.inc"

entry start

start:
	call	coro_init
	cmp	rax, 0
	jl	fail
	lea	rdi, [task_a]
	xor	rsi, rsi
	call	coro_spawn
	lea	rdi, [task_b]
	xor	rsi, rsi
	call	coro_spawn
	call	coro_run
	mov	al, 10
	call	print_char
	exit	EXIT_SUCCESS

fail:
	exit	EXIT_FAILURE

task_a:
	lea	rdi, [msg_a1]
	mov	rsi, msg_a1_len
	call	io_write
	call	coro_yield
	lea	rdi, [msg_a2]
	mov	rsi, msg_a2_len
	call	io_write
	ret

task_b:
	lea	rdi, [msg_b1]
	mov	rsi, msg_b1_len
	call	io_write
	call	coro_yield
	lea	rdi, [msg_b2]
	mov	rsi, msg_b2_len
	call	io_write
	ret

msg_a1 db 'A1 '
msg_a1_len = $ - msg_a1
msg_a2 db 'A2 '
msg_a2_len = $ - msg_a2
msg_b1 db 'B1 '
msg_b1_len = $ - msg_b1
msg_b2 db 'B2'
msg_b2_len = $ - msg_b2

segment readable writeable

coro_bss
runtime_print_bss
