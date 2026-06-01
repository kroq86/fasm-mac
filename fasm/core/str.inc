; Null-terminated C string helpers.

; rdi = null-terminated string
; rax = length (excluding null)
str_len:
	xor	rax, rax
.sl_loop:
	cmp	byte [rdi + rax], 0
	je	.sl_done
	inc	rax
	jmp	.sl_loop
.sl_done:
	ret

; rdi = a, rsi = b (null-terminated)
; rax = 1 if equal, 0 otherwise
str_eq:
.se_loop:
	mov	al, [rdi]
	mov	cl, [rsi]
	cmp	al, cl
	jne	.se_no
	test	al, al
	jz	.se_yes
	inc	rdi
	inc	rsi
	jmp	.se_loop
.se_yes:
	mov	rax, 1
	ret
.se_no:
	xor	rax, rax
	ret

; rdi = null-terminated string
; rax = hash (xor-shift mix over bytes)
str_hash:
	push	rsi
	xor	rax, rax
.sh_loop:
	movzx	ecx, byte [rdi]
	test	cl, cl
	jz	.sh_done
	xor	rax, rcx
	mov	rcx, rax
	shl	rcx, 13
	xor	rax, rcx
	mov	rcx, rax
	shr	rcx, 7
	xor	rax, rcx
	inc	rdi
	jmp	.sh_loop
.sh_done:
	pop	rsi
	ret

; rdi = null-terminated string
; rax = parsed int64, rbx = 0 on success, 1 on error
str_parse_int64:
	push	r12
	push	r13
	mov	r12, rdi
	xor	rax, rax
	xor	rbx, rbx
	mov	r13, 1
	movzx	ecx, byte [r12]
	cmp	cl, '-'
	jne	.spi_plus
	mov	r13, -1
	inc	r12
	jmp	.spi_digits
.spi_plus:
	cmp	cl, '+'
	jne	.spi_digits
	inc	r12
.spi_digits:
	movzx	ecx, byte [r12]
	cmp	cl, '0'
	jb	.spi_err
	cmp	cl, '9'
	ja	.spi_err
	mov	rbx, 1
	imul	rax, 10
	sub	cl, '0'
	add	rax, rcx
	inc	r12
	jmp	.spi_digits
.spi_err:
	test	rbx, rbx
	jz	.spi_fail
	cmp	byte [r12], 0
	jne	.spi_fail
	imul	rax, r13
	xor	rbx, rbx
	jmp	.spi_done
.spi_fail:
	mov	rbx, 1
	xor	rax, rax
.spi_done:
	pop	r13
	pop	r12
	ret

; rdi = null-terminated string; writes to stdout (requires print_io.inc BSS)
str_write:
	jmp	print_cstr
