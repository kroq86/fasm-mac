; hexpeek: tiny hex dump of file bytes.
;
; Usage: hexpeek [-n bytes] [-s skip] <file>

format ELF64 executable 3
include "fasm/core/platform.inc"

SEEK_SET equ 0

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/file.inc"
include "fasm/core/hex.inc"

READ_CHUNK equ 4096
LINE_BUF_SIZE equ 128

entry start

start:
	mov	[argv_base], rsp
	mov	rbx, rsp
	mov	rax, [rbx]
	cmp	rax, 2
	jb	usage
	mov	[argc], rax
	mov	qword [arg_index], 1
	mov	qword [byte_limit], 256
	mov	qword [byte_skip], 0
	call	parse_options
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	mov	rax, [rbx + 8 + rax * 8]
	mov	[file_path], rax
	mov	rdi, rax
	call	file_open_read
	cmp	rax, 0
	jl	hex_open_err
	mov	[file_fd], rax
	mov	rdi, [file_fd]
	mov	rsi, [byte_skip]
	xor	rdx, rdx
	mov	rax, SYS_lseek
	syscall
	jc	hex_open_err
	mov	r15, [byte_limit]
	mov	r14, [byte_skip]
	xor	r13, r13
.read_loop:
	cmp	r13, r15
	jae	.done
	mov	rdi, [file_fd]
	lea	rsi, [read_buf]
	mov	rdx, READ_CHUNK
	call	file_read_chunk
	cmp	rax, 0
	jle	.done
	mov	r12, rax
	xor	rbx, rbx
.line_loop:
	cmp	r13, r15
	jae	.done
	cmp	rbx, r12
	jae	.read_loop
	mov	rax, r15
	sub	rax, r13
	cmp	rax, HEX_LINE_BYTES
	jbe	.hpl_len
	mov	rax, HEX_LINE_BYTES
.hpl_len:
	mov	r11, r12
	sub	r11, rbx
	cmp	rax, r11
	jbe	.hpl_ok
	mov	rax, r11
.hpl_ok:
	test	rax, rax
	jz	.read_loop
	lea	rdi, [line_buf]
	lea	rsi, [read_buf + rbx]
	mov	rdx, rax
	mov	rcx, r14
	add	rcx, r13
	call	hex_write_line
	add	r13, rax
	add	rbx, rax
	jmp	.line_loop
.done:
	mov	rdi, [file_fd]
	call	file_close
	exit EXIT_SUCCESS

parse_options:
.po_loop:
	mov	rbx, [argv_base]
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	.po_done
	mov	rdi, [rbx + 8 + rax * 8]
	cmp	byte [rdi], '-'
	jne	.po_done
	lea	rsi, [opt_n]
	call	str_eq
	test	rax, rax
	jnz	.po_n
	mov	rax, [arg_index]
	mov	rdi, [rbx + 8 + rax * 8]
	lea	rsi, [opt_s]
	call	str_eq
	test	rax, rax
	jnz	.po_s
	jmp	usage
.po_n:
	inc	qword [arg_index]
	mov	rbx, [argv_base]
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	mov	rdi, [rbx + 8 + rax * 8]
	call	str_parse_int64
	test	rbx, rbx
	jnz	usage
	mov	[byte_limit], rax
	jmp	.po_next
.po_s:
	inc	qword [arg_index]
	mov	rbx, [argv_base]
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	mov	rdi, [rbx + 8 + rax * 8]
	call	str_parse_int64
	test	rbx, rbx
	jnz	usage
	mov	[byte_skip], rax
.po_next:
	inc	qword [arg_index]
	jmp	.po_loop
.po_done:
	ret

usage:
	lea	rdi, [usage_msg]
	mov	rsi, usage_msg_len
	call	write_stderr
	exit 2

.open_err:
hex_open_err:
	lea	rdi, [err_open_prefix]
	call	write_cstr_stderr
	mov	rdi, [file_path]
	call	write_cstr_stderr
	mov	al, 10
	call	write_char_stderr
	exit 2

write_stderr:
	mov	rdx, rsi
	mov	rsi, rdi
	mov	rdi, STDERR
	mov	rax, SYS_write
	syscall
	ret

write_char_stderr:
	mov	[stderr_char], al
	lea	rdi, [stderr_char]
	mov	rsi, 1
	jmp	write_stderr

write_cstr_stderr:
	push	rdi
	xor	rax, rax
.wcs_len:
	cmp	byte [rdi + rax], 0
	je	.wcs_out
	inc	rax
	jmp	.wcs_len
.wcs_out:
	mov	rsi, rax
	pop	rdi
	jmp	write_stderr

opt_n db '-n', 0
opt_s db '-s', 0
usage_msg db 'usage: hexpeek [-n bytes] [-s skip] <file>', 10
usage_msg_len = $ - usage_msg
err_open_prefix db 'hexpeek: cannot open: ', 0

segment readable writeable

argc dq ?
argv_base dq ?
arg_index dq ?
byte_limit dq ?
byte_skip dq ?
file_path dq ?
file_fd dq ?
read_buf rb READ_CHUNK
line_buf rb LINE_BUF_SIZE
stderr_char rb 1

include "fasm/core/runtime_bss.inc"
runtime_print_bss
