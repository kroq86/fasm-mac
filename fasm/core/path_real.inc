; Small static-server path helpers.

PATH_REAL_MAX equ 4096

PATH_REAL_OK equ 0
PATH_REAL_NOT_FOUND equ 1
PATH_REAL_ESCAPE equ 2
PATH_REAL_TOO_LONG equ 3

; rdi = root cstr, rsi = HTTP path bytes, rdx = HTTP path len, rcx = output buffer
; rax = output buffer or 0 on overflow
path_join_root_http:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rdi, r12
	call	str_len
	mov	rbx, rax
	cmp	rbx, PATH_REAL_MAX - 2
	jae	.pj_fail
	mov	rdi, r15
	mov	rsi, r12
	mov	rcx, rbx
	rep	movsb
	cmp	r14, 1
	jne	.pj_add_path
	cmp	byte [r13], '/'
	jne	.pj_add_path
	mov	byte [r15 + rbx], 0
	mov	rax, r15
	jmp	.pj_out
.pj_add_path:
	cmp	rbx, 1
	jne	.pj_need_slash
	cmp	byte [r15], '/'
	je	.pj_skip_slash
.pj_need_slash:
	cmp	byte [r15 + rbx - 1], '/'
	je	.pj_skip_slash
	mov	byte [r15 + rbx], '/'
	inc	rbx
.pj_skip_slash:
	cmp	byte [r13], '/'
	jne	.pj_copy
	inc	r13
	dec	r14
.pj_copy:
	mov	rax, rbx
	add	rax, r14
	cmp	rax, PATH_REAL_MAX - 1
	jae	.pj_fail
	lea	rdi, [r15 + rbx]
	mov	rsi, r13
	mov	rcx, r14
	rep	movsb
	lea	rax, [r15 + rbx]
	mov	byte [rax + r14], 0
	mov	rax, r15
	jmp	.pj_out
.pj_fail:
	xor	rax, rax
.pj_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = HTTP path bytes, rsi = HTTP path len
; rax = 1 if unsafe for v1 static serving, else 0
path_http_unsafe:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	xor	rbx, rbx
.phu_loop:
	cmp	rbx, r13
	jae	.phu_safe
	mov	al, [r12 + rbx]
	cmp	al, '\'
	je	.phu_bad
	cmp	al, '%'
	je	.phu_bad
	cmp	al, '.'
	jne	.phu_next
	mov	rax, rbx
	inc	rax
	cmp	rax, r13
	jae	.phu_next
	cmp	byte [r12 + rbx + 1], '.'
	jne	.phu_next
	cmp	rbx, 0
	je	.phu_check_after
	cmp	byte [r12 + rbx - 1], '/'
	jne	.phu_next
.phu_check_after:
	mov	rax, rbx
	add	rax, 2
	cmp	rax, r13
	jae	.phu_bad
	cmp	byte [r12 + rbx + 2], '/'
	je	.phu_bad
.phu_next:
	inc	rbx
	jmp	.phu_loop
.phu_bad:
	mov	rax, 1
	jmp	.phu_out
.phu_safe:
	xor	rax, rax
.phu_out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = canonical root cstr, rsi = canonical target cstr
; rax = 1 if target is root or under root, else 0
path_is_within_root:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, r12
	call	str_len
	mov	rbx, rax
	cmp	rbx, 1
	jne	.pwr_compare
	cmp	byte [r12], '/'
	jne	.pwr_compare
	cmp	byte [r12 + 1], 0
	je	.pwr_yes
.pwr_compare:
	xor	rcx, rcx
.pwr_loop:
	cmp	rcx, rbx
	jae	.pwr_boundary
	mov	al, [r12 + rcx]
	cmp	al, [r13 + rcx]
	jne	.pwr_no
	inc	rcx
	jmp	.pwr_loop
.pwr_boundary:
	mov	al, [r13 + rbx]
	test	al, al
	jz	.pwr_yes
	cmp	al, '/'
	je	.pwr_yes
.pwr_no:
	xor	rax, rax
	jmp	.pwr_out
.pwr_yes:
	mov	rax, 1
.pwr_out:
	pop	r13
	pop	r12
	pop	rbx
	ret
