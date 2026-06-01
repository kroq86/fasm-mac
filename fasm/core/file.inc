; Small file I/O helpers.

FILE_OK equ 0
FILE_ERR_OPEN equ 1
FILE_ERR_READ equ 2
FILE_ERR_TOO_LARGE equ 3

; rdi = path, rsi = destination buffer, rdx = buffer capacity, rcx = size storage*
; rax = FILE_OK / FILE_ERR_*
file_read_all_fixed:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rsi
	mov	r13, rdx
	mov	r14, rcx
	open_file rdi, O_RDONLY, 0
	jump_if_syscall_error .fraf_open_error
	mov	rbx, rax
	mov	qword [r14], 0
.fraf_loop:
	mov	rax, r13
	sub	rax, [r14]
	jz	.fraf_too_large
	mov	rdi, rbx
	mov	rsi, r12
	add	rsi, [r14]
	mov	rdx, rax
	mov	rax, SYS_read
	syscall
	jc	.fraf_read_error
	test	rax, rax
	jz	.fraf_done
	add	[r14], rax
	jmp	.fraf_loop
.fraf_done:
	close_file rbx
	mov	rax, FILE_OK
	jmp	.fraf_out
.fraf_too_large:
	close_file rbx
	mov	rax, FILE_ERR_TOO_LARGE
	jmp	.fraf_out
.fraf_read_error:
	close_file rbx
	mov	rax, FILE_ERR_READ
	jmp	.fraf_out
.fraf_open_error:
	mov	rax, FILE_ERR_OPEN
.fraf_out:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = path
; rax = fd or -1
file_open_read:
	open_file rdi, O_RDONLY, 0
	jump_if_syscall_error .for_err
	ret
.for_err:
	mov	rax, -1
	ret

; rdi = fd, rsi = buffer, rdx = capacity
; rax = bytes read, 0 EOF, -1 error
file_read_chunk:
	mov	rax, SYS_read
	syscall
	jc	.frc_err
	ret
.frc_err:
	mov	rax, -1
	ret

; rdi = fd
file_close:
	close_file rdi
	ret
