; LeetCode helpers: read ints from Vec / Value, build test vectors.

; rdi = Value*
; rax = int64 (0 if not VAL_INT)
value_as_int:
	movzx	eax, byte [rdi + VALUE_TAG_OFF]
	cmp	al, VAL_INT
	jne	.vai_zero
	mov	rax, [rdi + VALUE_DATA_OFF]
	ret
.vai_zero:
	xor	rax, rax
	ret

; rdi = VecHeader*, rsi = index
; rax = int64 at index
vec_get_int:
	push	rdi
	push	rsi
	call	vec_get
	mov	rdi, rax
	call	value_as_int
	pop	rsi
	pop	rdi
	ret

; rdi = Heap*, rsi = VecHeader*, rdx = int array, rcx = count
vec_build_ints:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	rbx, rsi
	mov	r13, rdx
	mov	r14, rcx
	xor	r15, r15
.vbi_loop:
	cmp	r15, r14
	jae	.vbi_done
	mov	rdi, r12
	mov	rsi, rbx
	mov	rdx, [r13 + r15 * 8]
	call	vec_append_int
	inc	r15
	jmp	.vbi_loop
.vbi_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; print_int_sp / print_int_nl live in print_io.inc
