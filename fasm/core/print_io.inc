; Low-level stdout helpers (no dynvec dependency).
; Requires runtime_print_bss in the including program.

; rdi = ptr, rsi = len
io_write:
	mov	rdx, rsi
	mov	rsi, rdi
	mov	rdi, STDOUT
	mov	rax, SYS_write
	syscall
	ret

; al = char
print_char:
	mov	[char_buf], al
	lea	rdi, [char_buf]
	mov	rsi, 1
	jmp	io_write

; rdi = null-terminated string
print_cstr:
	push	rdi
	xor	rax, rax
.pc_len:
	cmp	byte [rdi + rax], 0
	je	.pc_out
	inc	rax
	jmp	.pc_len
.pc_out:
	mov	rsi, rax
	pop	rdi
	jmp	io_write

; rax = signed int64
print_int64:
	push	rbx
	push	r12
	mov	r12, rax
	test	r12, r12
	jns	.pi_positive
	neg	r12
	mov	al, '-'
	call	print_char
.pi_positive:
	mov	rbx, r12
	lea	rdi, [int_digits + 20]
.pi_loop:
	xor	rdx, rdx
	mov	rax, rbx
	mov	rcx, 10
	div	rcx
	add	dl, '0'
	dec	rdi
	mov	[rdi], dl
	mov	rbx, rax
	test	rbx, rbx
	jnz	.pi_loop
	cmp	rdi, int_digits + 20
	je	.pi_zero
	lea	rsi, [int_digits + 20]
	sub	rsi, rdi
	jmp	.pi_out
.pi_zero:
	lea	rdi, [int_digits + 19]
	mov	byte [rdi], '0'
	mov	rsi, 1
.pi_out:
	call	io_write
	pop	r12
	pop	rbx
	ret

; rax = int64, print with trailing space
print_int_sp:
	push	rax
	call	print_int64
	mov	al, ' '
	call	print_char
	pop	rax
	ret

; rax = int64; rdi = end of digit buffer (exclusive)
; returns rdi = start, rax = length
print_int64_to_buf:
	push	rbx
	push	r12
	mov	r12, rax
	mov	rbx, rdi
	test	r12, r12
	jns	.pitb_pos
	neg	r12
.pitb_pos:
	mov	rdi, rbx
.pitb_loop:
	xor	rdx, rdx
	mov	rax, r12
	mov	rcx, 10
	div	rcx
	add	dl, '0'
	dec	rdi
	mov	[rdi], dl
	mov	r12, rax
	test	r12, r12
	jnz	.pitb_loop
	cmp	rdi, rbx
	je	.pitb_zero
	mov	rax, rbx
	sub	rax, rdi
	jmp	.pitb_done
.pitb_zero:
	dec	rdi
	mov	byte [rdi], '0'
	mov	rax, 1
.pitb_done:
	pop	r12
	pop	rbx
	ret

; rax = int64, print with trailing newline
print_int_nl:
	call	print_int64
	mov	al, 10
	jmp	print_char
