; Parameter iteration, SGD update, and parameter zero_grad.

NODE_FLAGS equ 12
NODE_TENSOR equ 16
NODE_SIZE equ 32
FLAG_PARAMETER equ 1
TENSOR_DATA equ 0
TENSOR_GRAD equ 8

; int tensor_parameters_sgd_f32(TapeNode *nodes, uint64_t count, float lr)
; xmm0=lr. Preflights every parameter before writing. Returns -2 if the same
; Tensor descriptor appears as more than one parameter node.
tensor_parameters_sgd_f32:
	test	rdi, rdi
	jz	.tps_bad
	test	rsi, rsi
	jz	.tps_bad
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 16
	movss	[rsp], xmm0
	mov	r12, rdi
	mov	r13, rsi
	xor	r14, r14
; Validate storage and reject duplicate parameter descriptors before updates.
.tps_preflight:
	cmp	r14, r13
	jae	.tps_apply_begin
	mov	rax, r14
	shl	rax, 5
	lea	r15, [r12 + rax]
	cmp	dword [r15 + NODE_FLAGS], FLAG_PARAMETER
	jne	.tps_preflight_next
	mov	rdi, [r15 + NODE_TENSOR]
	test	rdi, rdi
	jz	.tps_invalid
	cmp	qword [rdi + TENSOR_GRAD], 0
	je	.tps_invalid
	lea	rsi, [rsp + 8]
	call	tensor_shape_validate_f32
	test	eax, eax
	jnz	.tps_finish
	lea	rbx, [r14 + 1]
.tps_duplicate_scan:
	cmp	rbx, r13
	jae	.tps_preflight_next
	mov	rax, rbx
	shl	rax, 5
	add	rax, r12
	cmp	dword [rax + NODE_FLAGS], FLAG_PARAMETER
	jne	.tps_duplicate_next
	mov	rdx, [rax + NODE_TENSOR]
	cmp	rdx, [r15 + NODE_TENSOR]
	je	.tps_duplicate
.tps_duplicate_next:
	inc	rbx
	jmp	.tps_duplicate_scan
.tps_preflight_next:
	inc	r14
	jmp	.tps_preflight
.tps_duplicate:
	mov	eax, -2
	jmp	.tps_finish
.tps_invalid:
	mov	eax, -1
	jmp	.tps_finish

.tps_apply_begin:
	xor	r14, r14
.tps_apply:
	cmp	r14, r13
	jae	.tps_ok
	mov	rax, r14
	shl	rax, 5
	add	rax, r12
	cmp	dword [rax + NODE_FLAGS], FLAG_PARAMETER
	jne	.tps_apply_next
	mov	rdx, [rax + NODE_TENSOR]
	mov	rdi, [rdx + TENSOR_DATA]
	mov	rsi, [rdx + TENSOR_GRAD]
	mov	rcx, [rdx + 16]
	imul	rcx, [rdx + 24]
	movss	xmm2, [rsp]
.tps_value:
	movss	xmm0, dword [rsi]
	mulss	xmm0, xmm2
	movss	xmm1, dword [rdi]
	subss	xmm1, xmm0
	movss	dword [rdi], xmm1
	add	rdi, 4
	add	rsi, 4
	dec	rcx
	jnz	.tps_value
.tps_apply_next:
	inc	r14
	jmp	.tps_apply
.tps_ok:
	xor	eax, eax
.tps_finish:
	add	rsp, 16
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
.tps_bad:
	mov	eax, -1
	ret

; int tensor_parameters_zero_grad_f32(TapeNode *nodes, uint64_t count)
tensor_parameters_zero_grad_f32:
	test	rdi, rdi
	jz	.tpz_bad
	test	rsi, rsi
	jz	.tpz_bad
	xor	r8, r8
.tpz_node:
	mov	rax, r8
	shl	rax, 5
	add	rax, rdi
	cmp	dword [rax + NODE_FLAGS], FLAG_PARAMETER
	jne	.tpz_next
	mov	rdx, [rax + NODE_TENSOR]
	test	rdx, rdx
	jz	.tpz_bad
	mov	r9, [rdx + TENSOR_GRAD]
	test	r9, r9
	jz	.tpz_bad
	mov	rcx, [rdx + 16]
	imul	rcx, [rdx + 24]
.tpz_value:
	mov	dword [r9], 0
	add	r9, 4
	dec	rcx
	jnz	.tpz_value
.tpz_next:
	inc	r8
	cmp	r8, rsi
	jb	.tpz_node
	xor	eax, eax
	ret
.tpz_bad:
	mov	eax, -1
	ret
