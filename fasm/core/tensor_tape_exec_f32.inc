; Indexed tape executor. Including object declares external kernels/validators.

NODE_OP equ 0
NODE_LHS equ 4
NODE_RHS equ 8
NODE_TENSOR equ 16
NODE_SIZE equ 32
OP_LEAF equ 0
OP_MATMUL equ 1
OP_RELU equ 2
OP_MSE equ 3
OP_BIAS_ADD equ 4

; int tensor_tape_forward_f32(TapeNode *nodes, uint64_t count,
;                             Arena *const *arenas, uint64_t arena_count)
tensor_tape_forward_f32:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r15, rdx
	mov	rbx, rcx
	call	tensor_tape_validate_f32
	test	eax, eax
	jnz	.ttf_out
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r15
	mov	rcx, rbx
	call	tensor_tape_lifetime_validate_f32
	test	eax, eax
	jnz	.ttf_out
	xor	r14, r14
.ttf_node:
	mov	r10, r14
	shl	r10, 5
	add	r10, r12
	mov	eax, [r10 + NODE_OP]
	test	eax, eax
	jz	.ttf_next
	mov	edx, [r10 + NODE_LHS]
	shl	rdx, 5
	mov	rdi, [r12 + rdx + NODE_TENSOR]
	mov	edx, [r10 + NODE_RHS]
	cmp	eax, OP_RELU
	je	.ttf_relu
	shl	rdx, 5
	mov	rsi, [r12 + rdx + NODE_TENSOR]
	mov	rdx, [r10 + NODE_TENSOR]
	cmp	eax, OP_MATMUL
	je	.ttf_matmul
	cmp	eax, OP_BIAS_ADD
	je	.ttf_bias
	call	tensor_mse_forward_f32
	jmp	.ttf_checked
.ttf_bias:
	call	tensor_bias_add_forward_f32
	jmp	.ttf_checked
.ttf_matmul:
	call	tensor_matmul_forward_f32
	jmp	.ttf_checked
.ttf_relu:
	mov	rsi, [r10 + NODE_TENSOR]
	call	tensor_relu_forward_f32
.ttf_checked:
	test	eax, eax
	jnz	.ttf_out
.ttf_next:
	inc	r14
	cmp	r14, r13
	jb	.ttf_node
	xor	eax, eax
.ttf_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; int tensor_tape_backward_f32(TapeNode *nodes, uint64_t count,
;                              Arena *const *arenas, uint64_t arena_count)
; The caller seeds the output gradient and controls zero_grad boundaries.
tensor_tape_backward_f32:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r15, rdx
	mov	rbx, rcx
	call	tensor_tape_validate_f32
	test	eax, eax
	jnz	.ttb_out
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r15
	mov	rcx, rbx
	call	tensor_tape_lifetime_validate_f32
	test	eax, eax
	jnz	.ttb_out
	sub	rsp, 16
	; Clear every non-parameter gradient except the final output seed. Parameter
	; gradients are optimizer-owned and accumulate until zero_grad/step policy.
	xor	r14, r14
.ttb_clear_node:
	lea	rax, [r14 + 1]
	cmp	rax, r13
	jae	.ttb_clear_done
	mov	r10, r14
	shl	r10, 5
	add	r10, r12
	cmp	dword [r10 + 12], 1
	je	.ttb_clear_next
	mov	rdi, [r10 + NODE_TENSOR]
	cmp	qword [rdi + 8], 0
	je	.ttb_clear_next
	mov	rsi, rsp
	call	tensor_shape_validate_f32
	test	eax, eax
	jnz	.ttb_clear_error
	mov	r10, r14
	shl	r10, 5
	add	r10, r12
	mov	rdi, [r10 + NODE_TENSOR]
	mov	rdi, [rdi + 8]
	mov	rcx, [rsp]
.ttb_zero_loop:
	mov	dword [rdi], 0
	add	rdi, 4
	dec	rcx
	jnz	.ttb_zero_loop
.ttb_clear_next:
	inc	r14
	jmp	.ttb_clear_node
.ttb_clear_error:
	add	rsp, 16
	jmp	.ttb_out
.ttb_clear_done:
	add	rsp, 16
	mov	r14, r13
.ttb_node:
	dec	r14
	mov	r10, r14
	shl	r10, 5
	add	r10, r12
	mov	eax, [r10 + NODE_OP]
	test	eax, eax
	jz	.ttb_next
	mov	edx, [r10 + NODE_LHS]
	shl	rdx, 5
	mov	rdi, [r12 + rdx + NODE_TENSOR]
	mov	edx, [r10 + NODE_RHS]
	cmp	eax, OP_RELU
	je	.ttb_relu
	shl	rdx, 5
	mov	rsi, [r12 + rdx + NODE_TENSOR]
	mov	rdx, [r10 + NODE_TENSOR]
	cmp	eax, OP_MATMUL
	je	.ttb_matmul
	cmp	eax, OP_BIAS_ADD
	je	.ttb_bias
	call	tensor_mse_backward_f32
	jmp	.ttb_checked
.ttb_bias:
	call	tensor_bias_add_backward_f32
	jmp	.ttb_checked
.ttb_matmul:
	call	tensor_matmul_backward_f32
	jmp	.ttb_checked
.ttb_relu:
	mov	rsi, [r10 + NODE_TENSOR]
	call	tensor_relu_backward_f32
.ttb_checked:
	test	eax, eax
	jnz	.ttb_out
.ttb_next:
	test	r14, r14
	jnz	.ttb_node
	xor	eax, eax
.ttb_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
