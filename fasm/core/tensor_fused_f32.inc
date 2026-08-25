; Experimental execution-fusion kernels. The semantic graph remains expanded.
; Common six-pointer ABI:
;   rdi=A, rsi=W, rdx=bias, rcx=matmul_out, r8=bias_out, r9=out
; Matmul+bias ignores bias_out and writes out. Matmul+bias+ReLU uses all six.
; Returns eax=0 or the first failing child-kernel status. Callee-saved preserved.
;
; A fused PlanStep stores FusionContext* in lhs and uses these thunks. Context:
;   +0 A, +8 W, +16 bias, +24 matmul_out, +32 bias_out, +40 out.

tensor_plan_matmul_bias_forward_f32:
	mov	r9, [rdi + 40]
	mov	rcx, [rdi + 24]
	mov	rdx, [rdi + 16]
	mov	rsi, [rdi + 8]
	mov	rdi, [rdi]
	jmp	tensor_matmul_bias_forward_f32

tensor_plan_matmul_bias_backward_f32:
	mov	r9, [rdi + 40]
	mov	rcx, [rdi + 24]
	mov	rdx, [rdi + 16]
	mov	rsi, [rdi + 8]
	mov	rdi, [rdi]
	jmp	tensor_matmul_bias_backward_f32

tensor_plan_matmul_bias_relu_forward_f32:
	mov	r9, [rdi + 40]
	mov	r8, [rdi + 32]
	mov	rcx, [rdi + 24]
	mov	rdx, [rdi + 16]
	mov	rsi, [rdi + 8]
	mov	rdi, [rdi]
	jmp	tensor_matmul_bias_relu_forward_f32

tensor_plan_matmul_bias_relu_backward_f32:
	mov	r9, [rdi + 40]
	mov	r8, [rdi + 32]
	mov	rcx, [rdi + 24]
	mov	rdx, [rdi + 16]
	mov	rsi, [rdi + 8]
	mov	rdi, [rdi]
	jmp	tensor_matmul_bias_relu_backward_f32

tensor_matmul_bias_forward_f32:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rbx, r9
	mov	rdx, r15
	call	tensor_matmul_forward_f32
	test	eax, eax
	jnz	.tmbf_out
	mov	rdi, r15
	mov	rsi, r14
	mov	rdx, rbx
	call	tensor_bias_add_forward_f32
.tmbf_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

tensor_matmul_bias_backward_f32:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rbx, r9
	mov	rdi, r15
	mov	rsi, r14
	mov	rdx, rbx
	call	tensor_bias_add_backward_f32
	test	eax, eax
	jnz	.tmbb_out
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r15
	call	tensor_matmul_backward_f32
.tmbb_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

tensor_matmul_bias_relu_forward_f32:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 16
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rbx, r8
	mov	[rsp], r9
	mov	rdx, r15
	call	tensor_matmul_forward_f32
	test	eax, eax
	jnz	.tmbrf_out
	mov	rdi, r15
	mov	rsi, r14
	mov	rdx, rbx
	call	tensor_bias_add_forward_f32
	test	eax, eax
	jnz	.tmbrf_out
	mov	rdi, rbx
	mov	rsi, [rsp]
	call	tensor_relu_forward_f32
.tmbrf_out:
	add	rsp, 16
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

tensor_matmul_bias_relu_backward_f32:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 16
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rbx, r8
	mov	[rsp], r9
	mov	rdi, rbx
	mov	rsi, [rsp]
	call	tensor_relu_backward_f32
	test	eax, eax
	jnz	.tmbrb_out
	mov	rdi, r15
	mov	rsi, r14
	mov	rdx, rbx
	call	tensor_bias_add_backward_f32
	test	eax, eax
	jnz	.tmbrb_out
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r15
	call	tensor_matmul_backward_f32
.tmbrb_out:
	add	rsp, 16
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
