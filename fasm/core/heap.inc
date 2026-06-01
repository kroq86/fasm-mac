; Central heap allocator with mark-and-sweep GC.

; rdi = Heap*
heap_init:
	mov	qword [rdi + HEAP_BLOCKS_OFF], 0
	mov	qword [rdi + HEAP_TOTAL_OFF], 0
	ret

; rdi = Heap*, rsi = payload bytes
; rax = payload pointer (0 on failure)
heap_alloc:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rbx, rsi
	lea	rdx, [rbx + BLOCK_HEADER_SIZE + PAGE_SIZE - 1]
	and	rdx, not (PAGE_SIZE - 1)
	call	mmap_alloc_raw
	test	rax, rax
	jz	.alloc_fail
	mov	rbx, rax
	mov	rcx, [r12 + HEAP_BLOCKS_OFF]
	mov	[rbx + BLOCK_NEXT_OFF], rcx
	mov	[r12 + HEAP_BLOCKS_OFF], rbx
	mov	[rbx + BLOCK_SIZE_OFF], rdx
	mov	byte [rbx + BLOCK_MARKED_OFF], 0
	add	qword [r12 + HEAP_TOTAL_OFF], rdx
	lea	rax, [rbx + BLOCK_HEADER_SIZE]
.alloc_fail:
	pop	r12
	pop	rbx
	ret

; rdi = Value*
gc_mark_value:
	push	rbx
	push	r12
	mov	r12, rdi
	test	r12, r12
	jz	.mark_done
	movzx	eax, byte [r12 + VALUE_TAG_OFF]
	cmp	al, VAL_STR
	je	.mark_str
	cmp	al, VAL_ARRAY
	je	.mark_array
.mark_done:
	pop	r12
	pop	rbx
	ret
.mark_str:
	mov	rbx, [r12 + VALUE_DATA_OFF]
	mark_block rbx
	jmp	.mark_done
.mark_array:
	mov	rdi, [r12 + VALUE_DATA_OFF]
	call	gc_mark_vec
	jmp	.mark_done

; rdi = VecHeader*
gc_mark_vec:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	test	r12, r12
	jz	.vec_done
	mark_block r12
	mov	r13, [r12 + VEC_DATA_OFF]
	mark_block r13
	mov	rbx, [r12 + VEC_COUNT_OFF]
	xor	r14, r14
.vec_loop:
	cmp	r14, rbx
	jae	.vec_done
	mov	rdi, r13
	mov	rax, r14
	imul	rax, rax, VALUE_SIZE
	add	rdi, rax
	call	gc_mark_value
	inc	r14
	jmp	.vec_loop
.vec_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = Heap*, rsi = root Value*
gc_collect:
	push	rbx
	push	r12
	push	r14
	mov	r12, rdi
	mov	rcx, [r12 + HEAP_BLOCKS_OFF]
.clear_loop:
	test	rcx, rcx
	jz	.mark_phase
	mov	byte [rcx + BLOCK_MARKED_OFF], 0
	mov	rcx, [rcx + BLOCK_NEXT_OFF]
	jmp	.clear_loop
.mark_phase:
	mov	rdi, rsi
	call	gc_mark_value
	lea	r14, [r12 + HEAP_BLOCKS_OFF]
	mov	rcx, [r12 + HEAP_BLOCKS_OFF]
.sweep_loop:
	test	rcx, rcx
	jz	.collect_done
	mov	rbx, [rcx + BLOCK_NEXT_OFF]
	cmp	byte [rcx + BLOCK_MARKED_OFF], 0
	jne	.keep
	mov	rdi, rcx
	mov	rsi, [rcx + BLOCK_SIZE_OFF]
	mov	rax, SYS_munmap
	syscall
	sub	qword [r12 + HEAP_TOTAL_OFF], rsi
	mov	[r14], rbx
	mov	rcx, rbx
	jmp	.sweep_loop
.keep:
	mov	byte [rcx + BLOCK_MARKED_OFF], 0
	lea	r14, [rcx + BLOCK_NEXT_OFF]
	mov	rcx, rbx
	jmp	.sweep_loop
.collect_done:
	pop	r14
	pop	r12
	pop	rbx
	ret

; rdi = Heap*
heap_destroy:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rcx, [r12 + HEAP_BLOCKS_OFF]
.free_loop:
	test	rcx, rcx
	jz	.destroy_done
	mov	rbx, [rcx + BLOCK_NEXT_OFF]
	mov	rdi, rcx
	mov	rsi, [rcx + BLOCK_SIZE_OFF]
	mov	rax, SYS_munmap
	syscall
	mov	rcx, rbx
	jmp	.free_loop
.destroy_done:
	mov	qword [r12 + HEAP_BLOCKS_OFF], 0
	mov	qword [r12 + HEAP_TOTAL_OFF], 0
	pop	r12
	pop	rbx
	ret

; rax = block header for payload in rdi (0 if null)
payload_to_block:
	test	rdi, rdi
	jz	.ptb_null
	mov	rax, rdi
	sub	rax, BLOCK_HEADER_SIZE
	ret
.ptb_null:
	xor	rax, rax
	ret
