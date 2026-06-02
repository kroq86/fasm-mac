; pathsum: recursive directory file/dir/byte counters.
;
; Usage: pathsum [dir]

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/file.inc"
include "fasm/core/dirwalk.inc"

entry start

start:
	mov	rbx, rsp
	mov	rax, [rbx]
	cmp	rax, 1
	je	.use_dot
	cmp	rax, 2
	jne	usage
	mov	rax, [rbx + 16]
	jmp	.run
.use_dot:
	lea	rax, [dot_path]
.run:
	mov	[root_path], rax
	mov	rdi, rax
	lea	rsi, [pathsum_cb]
	lea	rdx, [file_count]
	call	dirwalk_foreach
	cmp	rax, DIRWALK_OK
	jne	pathsum_bad_path
	lea	rdi, [label_files]
	call	print_cstr
	mov	rax, [file_count]
	call	print_int_nl
	lea	rdi, [label_dirs]
	call	print_cstr
	mov	rax, [dir_count]
	call	print_int_nl
	lea	rdi, [label_bytes]
	call	print_cstr
	mov	rax, [byte_total]
	call	print_int_nl
	exit EXIT_SUCCESS

usage:
	lea	rdi, [usage_msg]
	mov	rsi, usage_msg_len
	call	write_stderr
	exit 2

.bad_path:
pathsum_bad_path:
	lea	rdi, [err_prefix]
	call	write_cstr_stderr
	mov	rdi, [root_path]
	call	write_cstr_stderr
	mov	al, 10
	call	write_char_stderr
	exit 2

; rdi=path, rsi=type, rdx=size, rcx=counters*
pathsum_cb:
	cmp	rsi, DIRWALK_TYPE_FILE
	je	.ps_file
	cmp	rsi, DIRWALK_TYPE_DIR
	je	.ps_dir
	ret
.ps_file:
	inc	qword [rcx]
	add	[rcx + 16], rdx
	ret
.ps_dir:
	inc	qword [rcx + 8]
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

dot_path db '.', 0
label_files db 'files ', 0
label_dirs db 'dirs ', 0
label_bytes db 'bytes ', 0
usage_msg db 'usage: pathsum [dir]', 10
usage_msg_len = $ - usage_msg
err_prefix db 'pathsum: not a directory: ', 0

segment readable writeable

root_path dq ?
file_count dq ?
dir_count dq ?
byte_total dq ?
stderr_char rb 1

include "fasm/core/runtime_bss.inc"
runtime_print_bss
dirwalk_bss
