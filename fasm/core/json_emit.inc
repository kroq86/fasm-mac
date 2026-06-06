; Small JSON emit helpers for caller-owned output buffers.
;
; The including app must define:
;   json_out_ptr dq ?
;   json_out_len dq ?
;   json_out_cap dq ?

; rdi = bytes, rsi = len
json_append:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rax, [json_out_len]
	add	rax, r13
	cmp	rax, [json_out_cap]
	ja	.ja_done
	mov	rbx, [json_out_ptr]
	add	rbx, [json_out_len]
	xor	rcx, rcx
.ja_loop:
	cmp	rcx, r13
	jae	.ja_commit
	mov	al, [r12 + rcx]
	mov	[rbx + rcx], al
	inc	rcx
	jmp	.ja_loop
.ja_commit:
	add	[json_out_len], r13
.ja_done:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = null-terminated string
json_append_cstr:
	push	rdi
	xor	rax, rax
.jac_len:
	cmp	byte [rdi + rax], 0
	je	.jac_go
	inc	rax
	jmp	.jac_len
.jac_go:
	mov	rsi, rax
	pop	rdi
	jmp	json_append

; al = byte
json_append_byte:
	mov	[json_one_byte], al
	lea	rdi, [json_one_byte]
	mov	rsi, 1
	jmp	json_append

; rdi = bytes, rsi = len. Appends a quoted JSON string.
json_append_escaped:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	al, '"'
	call	json_append_byte
	xor	rbx, rbx
.jae_loop:
	cmp	rbx, r13
	jae	.jae_done
	mov	al, [r12 + rbx]
	cmp	al, '"'
	je	.jae_quote
	cmp	al, '\'
	je	.jae_slash
	cmp	al, 10
	je	.jae_nl
	cmp	al, 13
	je	.jae_skip
	cmp	al, 9
	je	.jae_tab
	cmp	al, 32
	jb	.jae_space
	call	json_append_byte
	jmp	.jae_next
.jae_quote:
	lea	rdi, [json_esc_quote]
	mov	rsi, 2
	call	json_append
	jmp	.jae_next
.jae_slash:
	lea	rdi, [json_esc_slash]
	mov	rsi, 2
	call	json_append
	jmp	.jae_next
.jae_nl:
	lea	rdi, [json_esc_nl]
	mov	rsi, 2
	call	json_append
	jmp	.jae_next
.jae_tab:
	lea	rdi, [json_esc_tab]
	mov	rsi, 2
	call	json_append
	jmp	.jae_next
.jae_space:
	mov	al, ' '
	call	json_append_byte
	jmp	.jae_next
.jae_skip:
.jae_next:
	inc	rbx
	jmp	.jae_loop
.jae_done:
	mov	al, '"'
	call	json_append_byte
	pop	r13
	pop	r12
	pop	rbx
	ret
