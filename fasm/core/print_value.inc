; Print tagged Value trees to stdout (requires dynvec + runtime_value_bss).

include "fasm/core/print_io.inc"

; rdi = Value*, rsi = depth
print_value:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	cmp	r13, GC_MAX_DEPTH
	jae	.pv_done
	test	r12, r12
	jz	.pv_nil
	movzx	eax, byte [r12 + VALUE_TAG_OFF]
	cmp	al, VAL_INT
	je	.pv_int
	cmp	al, VAL_STR
	je	.pv_str
	cmp	al, VAL_ARRAY
	je	.pv_array
	jmp	.pv_done
.pv_nil:
	lea	rdi, [nil_msg]
	mov	rsi, nil_msg_len
	call	io_write
	jmp	.pv_done
.pv_int:
	mov	rax, [r12 + VALUE_DATA_OFF]
	call	print_int64
	jmp	.pv_done
.pv_str:
	mov	rbx, [r12 + VALUE_DATA_OFF]
	lea	rdi, [rbx + STRBUF_HEADER_SIZE]
	mov	rsi, [rbx + STRBUF_LEN_OFF]
	call	io_write
	jmp	.pv_done
.pv_array:
	mov	rbx, [r12 + VALUE_DATA_OFF]
	call	print_vec
.pv_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rbx = VecHeader*, r13 = depth
print_vec:
	push	r12
	push	r13
	push	r14
	mov	r12, rbx
	mov	r14, r13
	mov	al, '['
	call	print_char
	xor	r13, r13
	mov	rbx, [r12 + VEC_COUNT_OFF]
.pv_loop:
	cmp	r13, rbx
	jae	.pv_end
	cmp	r13, 0
	je	.pv_elem
	mov	al, ','
	call	print_char
	mov	al, ' '
	call	print_char
.pv_elem:
	mov	rdi, r12
	mov	rsi, r13
	call	vec_get
	mov	rdi, rax
	mov	rsi, r14
	inc	rsi
	call	print_value
	inc	r13
	jmp	.pv_loop
.pv_end:
	mov	al, ']'
	call	print_char
	pop	r14
	pop	r13
	pop	r12
	ret
