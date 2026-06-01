; logknife: tiny structured log slicer.
;
; Usage: logknife [--jsonl] [--contains text] [--field key=value] [--level value] [--count] <file>

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/file.inc"
include "fasm/core/search.inc"
include "fasm/core/scanner.inc"
include "fasm/core/json.inc"

READ_BUF_SIZE equ 65536
LINE_BUF_SIZE equ 16384

entry start

start:
	mov	rbx, rsp
	mov	rax, [rbx]
	mov	[argc], rax
	cmp	rax, 2
	jb	usage
	mov	qword [arg_index], 1
	call	parse_args
	cmp	qword [file_path], 0
	je	usage
	mov	rdi, [file_path]
	call	scan_file
	cmp	rax, 2
	je	exit_error
	cmp	qword [flag_count], 0
	je	.exit_by_match
	mov	rax, [match_count]
	call	print_int_nl
.exit_by_match:
	cmp	qword [match_count], 0
	je	exit_nomatch
	exit EXIT_SUCCESS
exit_nomatch:
	exit EXIT_FAILURE
exit_error:
	exit 2

usage:
	lea	rdi, [usage_msg]
	mov	rsi, usage_msg_len
	call	write_stderr
	exit 2

parse_args:
.pa_loop:
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	.pa_done
	mov	rdi, [rbx + 8 + rax * 8]
	cmp	byte [rdi], '-'
	jne	.pa_file
	lea	rsi, [opt_jsonl]
	call	str_eq
	test	rax, rax
	jnz	.pa_jsonl
	mov	rax, [arg_index]
	mov	rdi, [rbx + 8 + rax * 8]
	lea	rsi, [opt_count]
	call	str_eq
	test	rax, rax
	jnz	.pa_count
	mov	rax, [arg_index]
	mov	rdi, [rbx + 8 + rax * 8]
	lea	rsi, [opt_contains]
	call	str_eq
	test	rax, rax
	jnz	.pa_contains
	mov	rax, [arg_index]
	mov	rdi, [rbx + 8 + rax * 8]
	lea	rsi, [opt_field]
	call	str_eq
	test	rax, rax
	jnz	.pa_field
	mov	rax, [arg_index]
	mov	rdi, [rbx + 8 + rax * 8]
	lea	rsi, [opt_level]
	call	str_eq
	test	rax, rax
	jnz	.pa_level
	jmp	usage
.pa_jsonl:
	mov	qword [flag_jsonl], 1
	inc	qword [arg_index]
	jmp	.pa_loop
.pa_count:
	mov	qword [flag_count], 1
	inc	qword [arg_index]
	jmp	.pa_loop
.pa_contains:
	inc	qword [arg_index]
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	mov	rax, [rbx + 8 + rax * 8]
	mov	[contains_ptr], rax
	mov	rdi, rax
	call	str_len
	mov	[contains_len], rax
	inc	qword [arg_index]
	jmp	.pa_loop
.pa_field:
	inc	qword [arg_index]
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	mov	rdi, [rbx + 8 + rax * 8]
	call	parse_field_arg
	inc	qword [arg_index]
	jmp	.pa_loop
.pa_level:
	inc	qword [arg_index]
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	lea	rax, [level_key]
	mov	[field_key_ptr], rax
	mov	qword [field_key_len], level_key_len
	mov	rax, [arg_index]
	mov	rax, [rbx + 8 + rax * 8]
	mov	[field_val_ptr], rax
	mov	rdi, rax
	call	str_len
	mov	[field_val_len], rax
	inc	qword [arg_index]
	jmp	.pa_loop
.pa_file:
	cmp	qword [file_path], 0
	jne	usage
	mov	[file_path], rdi
	inc	qword [arg_index]
	jmp	.pa_loop
.pa_done:
	ret

; rdi = key=value c-string
parse_field_arg:
	mov	[field_key_ptr], rdi
	xor	rax, rax
.pfa_loop:
	mov	cl, [rdi + rax]
	test	cl, cl
	jz	usage
	cmp	cl, '='
	je	.pfa_eq
	inc	rax
	jmp	.pfa_loop
.pfa_eq:
	mov	[field_key_len], rax
	lea	rcx, [rdi + rax + 1]
	mov	[field_val_ptr], rcx
	mov	rdi, rcx
	call	str_len
	mov	[field_val_len], rax
	ret

scan_file:
	lea	rsi, [read_buf]
	mov	rdx, READ_BUF_SIZE
	lea	rcx, [line_buf]
	mov	r8, LINE_BUF_SIZE
	mov	r9, handle_line
	call	scanner_scan_file
	cmp	rax, SCANNER_ERR_OPEN
	je	.sf_open_error
	cmp	rax, SCANNER_ERR_READ
	je	.sf_read_error
	xor	rax, rax
	ret
.sf_open_error:
	lea	rdi, [open_err_msg]
	mov	rsi, open_err_msg_len
	call	write_stderr
	mov	rdi, [file_path]
	call	write_cstr_stderr
	mov	al, 10
	call	write_char_stderr
	mov	rax, 2
	ret
.sf_read_error:
	lea	rdi, [read_err_msg]
	mov	rsi, read_err_msg_len
	call	write_stderr
	mov	rdi, [file_path]
	call	write_cstr_stderr
	mov	al, 10
	call	write_char_stderr
	mov	rax, 2
	ret

handle_line:
	mov	[line_ptr], rdi
	mov	[line_len], rsi
	call	line_matches
	test	rax, rax
	jz	.hl_done
	inc	qword [match_count]
	cmp	qword [flag_count], 0
	jne	.hl_done
	mov	rdi, [line_ptr]
	mov	rsi, [line_len]
	call	io_write
	mov	al, 10
	call	print_char
.hl_done:
	ret

line_matches:
	cmp	qword [contains_ptr], 0
	je	.lm_after_contains
	mov	rdi, [line_ptr]
	mov	rsi, [line_len]
	mov	rdx, [contains_ptr]
	mov	rcx, [contains_len]
	call	search_contains
	test	rax, rax
	jz	.lm_no
.lm_after_contains:
	cmp	qword [field_key_ptr], 0
	je	.lm_yes
	cmp	qword [flag_jsonl], 0
	je	.lm_no
	mov	rdi, [line_ptr]
	mov	rsi, [line_len]
	mov	rdx, [field_key_ptr]
	mov	rcx, [field_key_len]
	mov	r8, [field_val_ptr]
	mov	r9, [field_val_len]
	call	json_field_eq
	test	rax, rax
	jz	.lm_no
.lm_yes:
	mov	rax, 1
	ret
.lm_no:
	xor	rax, rax
	ret

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
	call	str_len
	mov	rsi, rax
	pop	rdi
	jmp	write_stderr

segment readable writeable

argc dq ?
arg_index dq ?
file_path dq 0
flag_jsonl dq 0
flag_count dq 0
contains_ptr dq 0
contains_len dq ?
field_key_ptr dq 0
field_key_len dq ?
field_val_ptr dq ?
field_val_len dq ?
match_count dq 0
line_ptr dq ?
line_len dq ?

opt_jsonl db '--jsonl', 0
opt_contains db '--contains', 0
opt_field db '--field', 0
opt_level db '--level', 0
opt_count db '--count', 0
level_key db 'level'
level_key_len = $ - level_key

usage_msg db 'usage: logknife [--jsonl] [--contains text] [--field key=value] [--level value] [--count] <file>', 10
usage_msg_len = $ - usage_msg
open_err_msg db 'logknife: cannot open: '
open_err_msg_len = $ - open_err_msg
read_err_msg db 'logknife: cannot read: '
read_err_msg_len = $ - read_err_msg

read_buf rb READ_BUF_SIZE
line_buf rb LINE_BUF_SIZE
stderr_char rb 1

include "fasm/core/runtime_bss.inc"
runtime_print_bss
scanner_bss
