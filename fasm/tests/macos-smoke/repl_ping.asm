; Smoke: REPL reads lines; PING -> PONG.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/repl.inc"

entry start

start:
.repl_loop:
	lea	rdi, [line_buf]
	mov	rsi, REPL_LINE_MAX
	call	repl_read_line
	cmp	rax, REPL_EOF
	je	.repl_exit
	test	rax, rax
	jz	.repl_loop

	lea	rdi, [line_buf]
	lea	rsi, [token_ptrs]
	mov	rdx, REPL_MAX_TOKENS
	call	repl_tokenize
	test	rax, rax
	jz	.repl_loop

	lea	rdi, [token_ptrs]
	mov	rdi, [rdi]
	lea	rsi, [cmd_ping]
	call	str_eq
	test	rax, rax
	jz	.repl_loop

	call	repl_write_pong
	jmp	.repl_loop

.repl_exit:
	exit EXIT_SUCCESS

segment readable writeable

line_buf rb REPL_LINE_MAX
token_ptrs rq REPL_MAX_TOKENS

cmd_ping db 'PING', 0

include "fasm/core/runtime_bss.inc"
runtime_print_bss
