; Growable Vec of Value, allocated through Heap.

; rdi = Heap*, rsi = initial capacity
; rax = VecHeader* (0 on failure)
vec_init:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, r12
	mov	rsi, VEC_HEADER_SIZE
	call	heap_alloc
	test	rax, rax
	jz	.init_fail
	mov	rbx, rax
	mov	qword [rbx + VEC_COUNT_OFF], 0
	mov	[rbx + VEC_CAP_OFF], r13
	mov	rdi, r12
	mov	rsi, r13
	imul	rsi, rsi, VALUE_SIZE
	call	heap_alloc
	test	rax, rax
	jz	.init_fail
	mov	[rbx + VEC_DATA_OFF], rax
	mov	rax, rbx
.init_fail:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = Heap*, rsi = VecHeader*
vec_grow:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	rbx, rsi
	mov	r13, [rbx + VEC_CAP_OFF]
	lea	r13, [r13 + r13 + 1]
	mov	rdi, r12
	mov	rsi, r13
	imul	rsi, rsi, VALUE_SIZE
	call	heap_alloc
	test	rax, rax
	jz	.grow_fail
	mov	rcx, rax
	mov	rsi, [rbx + VEC_DATA_OFF]
	mov	rdx, [rbx + VEC_COUNT_OFF]
	imul	rdx, rdx, VALUE_SIZE
	mov	rdi, rcx
	test	rdx, rdx
	jz	.grow_copy_done
.copy_loop:
	mov	al, [rsi]
	mov	[rdi], al
	inc	rsi
	inc	rdi
	dec	rdx
	jnz	.copy_loop
.grow_copy_done:
	mov	[rbx + VEC_DATA_OFF], rcx
	mov	[rbx + VEC_CAP_OFF], r13
.grow_fail:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = VecHeader*, rsi = index
; rax = Value*
vec_get:
	mov	rax, [rdi + VEC_DATA_OFF]
	mov	rcx, rsi
	imul	rcx, rcx, VALUE_SIZE
	add	rax, rcx
	ret

; rdi = Heap*, rsi = VecHeader*, rdx = int64
vec_append_int:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	rbx, rsi
	mov	r13, rdx
	mov	rax, [rbx + VEC_COUNT_OFF]
	cmp	rax, [rbx + VEC_CAP_OFF]
	jb	.int_has_room
	mov	rdi, r12
	mov	rsi, rbx
	call	vec_grow
.int_has_room:
	mov	rcx, [rbx + VEC_COUNT_OFF]
	mov	rdi, rbx
	mov	rsi, rcx
	call	vec_get
	mov	byte [rax + VALUE_TAG_OFF], VAL_INT
	mov	[rax + VALUE_DATA_OFF], r13
	inc	qword [rbx + VEC_COUNT_OFF]
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = Heap*, rsi = VecHeader*, rdx = byte*, rcx = len
vec_append_str:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	rbx, rsi
	mov	r13, rdx
	mov	r14, rcx
	mov	rax, [rbx + VEC_COUNT_OFF]
	cmp	rax, [rbx + VEC_CAP_OFF]
	jb	.str_has_room
	mov	rdi, r12
	mov	rsi, rbx
	call	vec_grow
.str_has_room:
	mov	rdi, r12
	mov	rsi, STRBUF_HEADER_SIZE
	add	rsi, r14
	call	heap_alloc
	test	rax, rax
	jz	.str_fail
	mov	rdx, rax
	mov	[rdx + STRBUF_LEN_OFF], r14
	lea	rdi, [rdx + STRBUF_HEADER_SIZE]
	mov	rsi, r13
	mov	rcx, r14
.str_copy:
	test	rcx, rcx
	jz	.str_copy_done
	mov	al, [rsi]
	mov	[rdi], al
	inc	rsi
	inc	rdi
	dec	rcx
	jnz	.str_copy
.str_copy_done:
	mov	rcx, [rbx + VEC_COUNT_OFF]
	mov	rdi, rbx
	mov	rsi, rcx
	call	vec_get
	mov	byte [rax + VALUE_TAG_OFF], VAL_STR
	mov	[rax + VALUE_DATA_OFF], rdx
	inc	qword [rbx + VEC_COUNT_OFF]
.str_fail:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = Heap*, rsi = VecHeader*, rdx = initial child capacity
; rax = new VecHeader*
vec_append_array:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	rbx, rsi
	mov	r13, rdx
	mov	rax, [rbx + VEC_COUNT_OFF]
	cmp	rax, [rbx + VEC_CAP_OFF]
	jb	.arr_has_room
	mov	rdi, r12
	mov	rsi, rbx
	call	vec_grow
.arr_has_room:
	mov	rdi, r12
	mov	rsi, r13
	call	vec_init
	test	rax, rax
	jz	.arr_fail
	mov	rdx, rax
	mov	rcx, [rbx + VEC_COUNT_OFF]
	mov	rdi, rbx
	mov	rsi, rcx
	call	vec_get
	mov	byte [rax + VALUE_TAG_OFF], VAL_ARRAY
	mov	[rax + VALUE_DATA_OFF], rdx
	inc	qword [rbx + VEC_COUNT_OFF]
	mov	rax, rdx
.arr_fail:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = VecHeader*, rsi = index
vec_truncate:
	mov	rax, [rdi + VEC_COUNT_OFF]
	cmp	rsi, rax
	jae	.trunc_done
	mov	[rdi + VEC_COUNT_OFF], rsi
.trunc_done:
	ret
