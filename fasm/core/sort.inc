; In-place quicksort for Vec of VAL_INT.

; rdi = Value* a, rsi = Value* b
; rax = negative if a<b, zero if equal, positive if a>b
value_int_cmp:
	mov	rax, [rdi + VALUE_DATA_OFF]
	mov	rcx, [rsi + VALUE_DATA_OFF]
	sub	rax, rcx
	ret

; rdi = Value* a, rsi = Value* b — swap 16-byte slots
value_swap:
	mov	rax, [rdi]
	mov	rcx, [rdi + 8]
	mov	rdx, [rsi]
	mov	r8, [rsi + 8]
	mov	[rdi], rdx
	mov	[rdi + 8], r8
	mov	[rsi], rax
	mov	[rsi + 8], rcx
	ret

; rdi = VecHeader*, rsi = lo, rdx = hi (inclusive)
vec_quicksort_range:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	cmp	r13, r14
	jge	.qsr_done
	mov	r15, r13
	mov	rbx, r13
.qsr_jloop:
	cmp	rbx, r14
	jge	.qsr_pivot
	mov	rdi, r12
	mov	rsi, r14
	call	vec_get
	mov	r10, rax
	mov	rdi, r12
	mov	rsi, rbx
	call	vec_get
	mov	rdi, rax
	mov	rsi, r10
	call	value_int_cmp
	jg	.qsr_jnext
	mov	rdi, r12
	mov	rsi, r15
	call	vec_get
	mov	r10, rax
	mov	rdi, r12
	mov	rsi, rbx
	call	vec_get
	mov	rdi, r10
	mov	rsi, rax
	call	value_swap
	inc	r15
.qsr_jnext:
	inc	rbx
	jmp	.qsr_jloop
.qsr_pivot:
	mov	rdi, r12
	mov	rsi, r15
	call	vec_get
	mov	r10, rax
	mov	rdi, r12
	mov	rsi, r14
	call	vec_get
	mov	rdi, r10
	mov	rsi, rax
	call	value_swap
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r15
	dec	rdx
	call	vec_quicksort_range
	mov	rdi, r12
	mov	rsi, r15
	inc	rsi
	mov	rdx, r14
	call	vec_quicksort_range
.qsr_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = VecHeader*
vec_sort_int:
	push	rbx
	mov	rbx, rdi
	mov	rsi, 0
	mov	rax, [rbx + VEC_COUNT_OFF]
	test	rax, rax
	jz	.vsi_done
	dec	rax
	mov	rdx, rax
	mov	rdi, rbx
	call	vec_quicksort_range
.vsi_done:
	pop	rbx
	ret

; rdi = VecHeader*
; rax = 1 if sorted ascending, 0 otherwise
vec_is_sorted:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	rbx, [r12 + VEC_COUNT_OFF]
	cmp	rbx, 1
	jbe	.vis_yes
	xor	r13, r13
.vis_loop:
	mov	rax, r13
	inc	rax
	cmp	rax, rbx
	jae	.vis_yes
	mov	rdi, r12
	mov	rsi, r13
	call	vec_get
	mov	r14, rax
	mov	rdi, r12
	mov	rsi, r13
	inc	rsi
	call	vec_get
	mov	rdi, r14
	mov	rsi, rax
	call	value_int_cmp
	jg	.vis_no
	inc	r13
	jmp	.vis_loop
.vis_yes:
	mov	rax, 1
	jmp	.vis_done
.vis_no:
	xor	rax, rax
.vis_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
