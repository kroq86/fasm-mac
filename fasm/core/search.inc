; Literal byte-string search helpers.

; rdi = haystack ptr, rsi = haystack len, rdx = needle ptr, rcx = needle len
; rax = 1 if haystack contains needle, 0 otherwise
search_contains:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	test	rcx, rcx
	jz	.sc_yes
	cmp	rsi, rcx
	jb	.sc_no
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	xor	rbx, rbx
.sc_outer:
	mov	rax, r13
	sub	rax, r15
	cmp	rbx, rax
	ja	.sc_no
	lea	r11, [r12 + rbx]
	xor	r10, r10
.sc_inner:
	cmp	r10, r15
	jae	.sc_yes
	mov	al, [r11 + r10]
	cmp	al, [r14 + r10]
	jne	.sc_next
	inc	r10
	jmp	.sc_inner
.sc_next:
	inc	rbx
	jmp	.sc_outer
.sc_yes:
	mov	rax, 1
	jmp	.sc_done
.sc_no:
	xor	rax, rax
.sc_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
