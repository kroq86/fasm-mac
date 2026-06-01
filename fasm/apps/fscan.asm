; fscan: tiny literal-search CLI.
;
; Usage: fscan [-c] [-l] [-i] <literal> <file> [file...]
; Default prints: path:line:text

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/search.inc"

READ_BUF_SIZE equ 65536
LINE_BUF_SIZE equ 8192

entry start

start:
	mov	rbx, rsp
	mov	rax, [rbx]
	cmp	rax, 3
	jb	usage
	mov	[argc], rax
	mov	qword [arg_index], 1
	call	parse_options
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	mov	rax, [rbx + 8 + rax * 8]
	mov	[pattern_ptr], rax
	mov	rdi, rax
	call	str_len
	test	rax, rax
	jz	usage
	mov	[pattern_len], rax
	inc	qword [arg_index]
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage

.file_loop:
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	.done
	mov	rcx, [rbx + 8 + rax * 8]
	mov	[current_path], rcx
	mov	rdi, rcx
	call	scan_file
	cmp	rax, 2
	je	.exit_error
	call	print_file_summary
	inc	qword [arg_index]
	jmp	.file_loop

.done:
	cmp	qword [any_found], 0
	je	.exit_nomatch
	exit EXIT_SUCCESS
.exit_nomatch:
	exit EXIT_FAILURE
.exit_error:
	exit 2

usage:
	lea	rdi, [usage_msg]
	mov	rsi, usage_msg_len
	call	write_stderr
	exit 2

parse_options:
.po_loop:
	mov	rax, [arg_index]
	cmp	rax, [argc]
	jae	usage
	mov	rdi, [rbx + 8 + rax * 8]
	cmp	byte [rdi], '-'
	jne	.po_done
	cmp	byte [rdi + 1], 0
	je	.po_done
	cmp	byte [rdi + 1], '-'
	jne	.po_flags
	cmp	byte [rdi + 2], 0
	jne	usage
	inc	qword [arg_index]
	jmp	.po_done
.po_flags:
	inc	rdi
.po_flag_loop:
	mov	al, [rdi]
	test	al, al
	jz	.po_next_arg
	cmp	al, 'c'
	je	.po_count
	cmp	al, 'l'
	je	.po_files
	cmp	al, 'i'
	je	.po_ci
	jmp	usage
.po_count:
	mov	qword [flag_count], 1
	jmp	.po_flag_next
.po_files:
	mov	qword [flag_files_only], 1
	jmp	.po_flag_next
.po_ci:
	mov	qword [flag_ignore_case], 1
.po_flag_next:
	inc	rdi
	jmp	.po_flag_loop
.po_next_arg:
	inc	qword [arg_index]
	jmp	.po_loop
.po_done:
	ret

; rdi = path
; rax = 0/1 success, 2 error
scan_file:
	push	rbx
	mov	[scan_path], rdi
	open_file rdi, O_RDONLY, 0
	jump_if_syscall_error .sf_open_error
	mov	[scan_fd], rax
	mov	qword [line_no], 1
	mov	qword [line_len], 0
	mov	qword [line_overflow], 0
	mov	qword [file_match_count], 0

.read_loop:
	read_file [scan_fd], read_buf, READ_BUF_SIZE
	jump_if_syscall_error .sf_read_error
	test	rax, rax
	jz	.sf_eof
	mov	rbx, rax
	xor	r12, r12
.byte_loop:
	cmp	r12, rbx
	jae	.read_loop
	mov	al, [read_buf + r12]
	cmp	al, 10
	je	.newline
	cmp	qword [line_overflow], 0
	jne	.next_byte
	mov	rcx, [line_len]
	cmp	rcx, LINE_BUF_SIZE
	jae	.mark_overflow
	mov	[line_buf + rcx], al
	inc	qword [line_len]
	jmp	.next_byte
.mark_overflow:
	mov	qword [line_overflow], 1
	jmp	.next_byte
.newline:
	call	scan_current_line
	inc	qword [line_no]
	mov	qword [line_len], 0
	mov	qword [line_overflow], 0
.next_byte:
	inc	r12
	jmp	.byte_loop

.sf_eof:
	cmp	qword [line_len], 0
	jne	.scan_tail
	cmp	qword [line_overflow], 0
	jne	.scan_tail
	jmp	.sf_close_ok
.scan_tail:
	call	scan_current_line
.sf_close_ok:
	close_file [scan_fd]
	xor	rax, rax
	pop	rbx
	ret
.sf_open_error:
	lea	rdi, [open_err_msg]
	mov	rsi, open_err_msg_len
	call	write_stderr
	mov	rdi, [scan_path]
	call	write_cstr_stderr
	mov	al, 10
	call	write_char_stderr
	mov	rax, 2
	pop	rbx
	ret
.sf_read_error:
	close_file [scan_fd]
	lea	rdi, [read_err_msg]
	mov	rsi, read_err_msg_len
	call	write_stderr
	mov	rdi, [scan_path]
	call	write_cstr_stderr
	mov	al, 10
	call	write_char_stderr
	mov	rax, 2
	pop	rbx
	ret

scan_current_line:
	cmp	qword [line_overflow], 0
	jne	.scl_done
	lea	rdi, [line_buf]
	mov	rsi, [line_len]
	mov	rdx, [pattern_ptr]
	mov	rcx, [pattern_len]
	cmp	qword [flag_ignore_case], 0
	jne	.scl_ci
	call	search_contains
	jmp	.scl_after_search
.scl_ci:
	call	search_contains_ascii_ci
.scl_after_search:
	test	rax, rax
	jz	.scl_done
	mov	qword [any_found], 1
	inc	qword [file_match_count]
	cmp	qword [flag_count], 0
	jne	.scl_done
	cmp	qword [flag_files_only], 0
	jne	.scl_done
	mov	rdi, [scan_path]
	call	print_cstr
	mov	al, ':'
	call	print_char
	mov	rax, [line_no]
	call	print_int64
	mov	al, ':'
	call	print_char
	lea	rdi, [line_buf]
	mov	rsi, [line_len]
	call	io_write
	mov	al, 10
	call	print_char
.scl_done:
	ret

print_file_summary:
	cmp	qword [flag_files_only], 0
	jne	.pfs_files_only
	cmp	qword [flag_count], 0
	jne	.pfs_count
	ret
.pfs_files_only:
	cmp	qword [file_match_count], 0
	je	.pfs_done
	mov	rdi, [scan_path]
	call	print_cstr
	mov	al, 10
	call	print_char
	jmp	.pfs_done
.pfs_count:
	mov	rdi, [scan_path]
	call	print_cstr
	mov	al, ':'
	call	print_char
	mov	rax, [file_match_count]
	call	print_int_nl
.pfs_done:
	ret

; rdi = ptr, rsi = len
write_stderr:
	mov	rdx, rsi
	mov	rsi, rdi
	mov	rdi, STDERR
	mov	rax, SYS_write
	syscall
	ret

; al = char
write_char_stderr:
	mov	[stderr_char], al
	lea	rdi, [stderr_char]
	mov	rsi, 1
	jmp	write_stderr

; rdi = null-terminated string
write_cstr_stderr:
	push	rdi
	call	str_len
	mov	rsi, rax
	pop	rdi
	jmp	write_stderr

segment readable writeable

argc dq ?
arg_index dq ?
pattern_ptr dq ?
pattern_len dq ?
current_path dq ?
any_found dq 0

scan_path dq ?
scan_fd dq ?
line_no dq ?
line_len dq ?
line_overflow dq ?

flag_count dq 0
flag_files_only dq 0
flag_ignore_case dq 0
file_match_count dq ?

usage_msg db 'usage: fscan [-c] [-l] [-i] <literal> <file> [file...]', 10
usage_msg_len = $ - usage_msg
open_err_msg db 'fscan: cannot open: '
open_err_msg_len = $ - open_err_msg
read_err_msg db 'fscan: cannot read: '
read_err_msg_len = $ - read_err_msg

read_buf rb READ_BUF_SIZE
line_buf rb LINE_BUF_SIZE

include "fasm/core/runtime_bss.inc"
runtime_print_bss
stderr_char rb 1
