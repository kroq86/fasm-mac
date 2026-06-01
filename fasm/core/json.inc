; Minimal JSONL helpers for top-level object fields.

; rdi=line ptr, rsi=line len, rdx=key ptr, rcx=key len, r8=value ptr, r9=value len
; rax = 1 if a top-level string/number/bool-ish field equals the byte value.
json_field_eq:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	xor	rbx, rbx
.jfe_loop:
	cmp	rbx, r13
	jae	.jfe_no
	mov	al, [r12 + rbx]
	cmp	al, '"'
	jne	.jfe_next
	inc	rbx
	mov	rdx, r13
	sub	rdx, rbx
	cmp	rdx, r15
	jb	.jfe_no
	xor	r10, r10
.jfe_key_cmp:
	cmp	r10, r15
	jae	.jfe_key_maybe
	lea	r11, [r12 + rbx]
	mov	al, [r11 + r10]
	cmp	al, [r14 + r10]
	jne	.jfe_skip_string
	inc	r10
	jmp	.jfe_key_cmp
.jfe_key_maybe:
	mov	rax, rbx
	add	rax, r15
	cmp	rax, r13
	jae	.jfe_no
	cmp	byte [r12 + rax], '"'
	jne	.jfe_skip_string
	inc	rax
.jfe_ws_colon:
	cmp	rax, r13
	jae	.jfe_no
	mov	cl, [r12 + rax]
	cmp	cl, ' '
	je	.jfe_ws_colon_next
	cmp	cl, 9
	je	.jfe_ws_colon_next
	cmp	cl, ':'
	je	.jfe_value_start
	jmp	.jfe_skip_string
.jfe_ws_colon_next:
	inc	rax
	jmp	.jfe_ws_colon
.jfe_value_start:
	inc	rax
.jfe_ws_value:
	cmp	rax, r13
	jae	.jfe_no
	mov	cl, [r12 + rax]
	cmp	cl, ' '
	je	.jfe_ws_value_next
	cmp	cl, 9
	je	.jfe_ws_value_next
	jmp	.jfe_compare_value
.jfe_ws_value_next:
	inc	rax
	jmp	.jfe_ws_value
.jfe_compare_value:
	cmp	byte [r12 + rax], '"'
	je	.jfe_string_value
	mov	rdx, rax
	jmp	.jfe_token_value_len
.jfe_string_value:
	inc	rax
	mov	rdx, rax
.jfe_string_value_loop:
	cmp	rax, r13
	jae	.jfe_no
	cmp	byte [r12 + rax], '"'
	je	.jfe_value_len_ready
	inc	rax
	jmp	.jfe_string_value_loop
.jfe_token_value_len:
	cmp	rax, r13
	jae	.jfe_value_len_ready
	mov	cl, [r12 + rax]
	cmp	cl, ','
	je	.jfe_value_len_ready
	cmp	cl, '}'
	je	.jfe_value_len_ready
	cmp	cl, ' '
	je	.jfe_value_len_ready
	inc	rax
	jmp	.jfe_token_value_len
.jfe_value_len_ready:
	mov	rcx, rax
	sub	rcx, rdx
	cmp	rcx, r9
	jne	.jfe_no
	xor	r10, r10
.jfe_value_cmp:
	cmp	r10, r9
	jae	.jfe_yes
	lea	r11, [r12 + rdx]
	mov	al, [r11 + r10]
	cmp	al, [r8 + r10]
	jne	.jfe_no
	inc	r10
	jmp	.jfe_value_cmp
.jfe_skip_string:
	cmp	rbx, r13
	jae	.jfe_no
	cmp	byte [r12 + rbx], '"'
	je	.jfe_after_skip
	inc	rbx
	jmp	.jfe_skip_string
.jfe_after_skip:
.jfe_next:
	inc	rbx
	jmp	.jfe_loop
.jfe_yes:
	mov	rax, 1
	jmp	.jfe_done
.jfe_no:
	xor	rax, rax
.jfe_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
