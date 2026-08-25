; Experimental execution-fusion kernels. The semantic graph remains expanded.
; Common six-pointer ABI:
;   rdi=A, rsi=W, rdx=bias, rcx=matmul_out, r8=bias_out, r9=out
; Matmul+bias ignores bias_out and writes out. Matmul+bias+ReLU uses all six.
; Returns eax=0 or the first failing child-kernel status. Callee-saved preserved.
;
; A fused PlanStep stores FusionContext* in lhs and uses these thunks. Context:
;   +0 A, +8 W, +16 bias, +24 matmul_out, +32 bias_out, +40 out,
;   +48 action_flags. Bit 0 requests REMAT by repeating fused forward at the
;   start of its backward thunk. ZERO_GRAD actions are scheduled inside fused
;   backward kernels immediately before each accumulating destination begins.
;   Bits 8-9 select matmul backward edges: 1=lhs, 2=rhs, 3=both.

; MSE's prediction.grad can share the just-dead loss.data slot. Clear it after
; forward and immediately before the accumulating MSE backward kernel.
tensor_plan_mse_backward_zero_lhs_f32:
	push	rdi
	push	rsi
	push	rdx
	push	rcx
	call	tensor_fused_zero_grad_f32
	pop	rcx
	pop	rdx
	pop	rsi
	pop	rdi
	jmp	tensor_mse_backward_f32

tensor_plan_matmul_bias_forward_f32:
	mov	r9, [rdi + 40]
	mov	rcx, [rdi + 24]
	mov	rdx, [rdi + 16]
	mov	rsi, [rdi + 8]
	mov	rdi, [rdi]
	jmp	tensor_matmul_bias_forward_f32

tensor_plan_matmul_bias_backward_f32:
	mov	r10, [rdi + 48]
	shr	r10, 8
	and	r10, 3
	mov	r9, [rdi + 40]
	mov	rcx, [rdi + 24]
	mov	rdx, [rdi + 16]
	mov	rsi, [rdi + 8]
	mov	rdi, [rdi]
	jmp	tensor_matmul_bias_backward_masked_f32

tensor_plan_matmul_bias_relu_forward_f32:
	mov	r9, [rdi + 40]
	mov	r8, [rdi + 32]
	mov	rcx, [rdi + 24]
	mov	rdx, [rdi + 16]
	mov	rsi, [rdi + 8]
	mov	rdi, [rdi]
	jmp	tensor_matmul_bias_relu_forward_f32

tensor_plan_matmul_bias_relu_backward_f32:
	push	rdi
	test	qword [rdi + 48], 1
	jz	.tpmbr_no_remat
	call	tensor_plan_matmul_bias_relu_forward_f32
	test	eax, eax
	jnz	.tpmbr_remat_failed
.tpmbr_no_remat:
	pop	rdi
	mov	r10, [rdi + 48]
	shr	r10, 8
	and	r10, 3
	mov	r9, [rdi + 40]
	mov	r8, [rdi + 32]
	mov	rcx, [rdi + 24]
	mov	rdx, [rdi + 16]
	mov	rsi, [rdi + 8]
	mov	rdi, [rdi]
	jmp	tensor_matmul_bias_relu_backward_masked_f32
.tpmbr_remat_failed:
	add	rsp, 8
	ret

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
	mov	r10, 3
tensor_matmul_bias_backward_masked_f32:
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
	mov	rbx, r9
	mov	[rsp], r10
	mov	rdi, r15
	call	tensor_fused_zero_grad_f32
	mov	rdi, r15
	mov	rsi, r14
	mov	rdx, rbx
	call	tensor_bias_add_backward_f32
	test	eax, eax
	jnz	.tmbb_out
	cmp	qword [rsp], 2
	je	.tmbb_skip_zero_a
	mov	rdi, r12
	call	tensor_fused_zero_grad_f32
.tmbb_skip_zero_a:
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r15
	cmp	qword [rsp], 1
	je	.tmbb_lhs
	cmp	qword [rsp], 2
	je	.tmbb_rhs
	call	tensor_matmul_backward_f32
	jmp	.tmbb_out
.tmbb_lhs:
	call	tensor_matmul_backward_lhs_f32
	jmp	.tmbb_out
.tmbb_rhs:
	call	tensor_matmul_backward_rhs_f32
.tmbb_out:
	add	rsp, 16
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
	mov	r10, 3
tensor_matmul_bias_relu_backward_masked_f32:
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
	mov	[rsp + 8], r10
	mov	rdi, rbx
	call	tensor_fused_zero_grad_f32
	mov	rdi, rbx
	mov	rsi, [rsp]
	call	tensor_relu_backward_f32
	test	eax, eax
	jnz	.tmbrb_out
	mov	rdi, r15
	call	tensor_fused_zero_grad_f32
	mov	rdi, r15
	mov	rsi, r14
	mov	rdx, rbx
	call	tensor_bias_add_backward_f32
	test	eax, eax
	jnz	.tmbrb_out
	cmp	qword [rsp + 8], 2
	je	.tmbrb_skip_zero_a
	mov	rdi, r12
	call	tensor_fused_zero_grad_f32
.tmbrb_skip_zero_a:
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r15
	cmp	qword [rsp + 8], 1
	je	.tmbrb_lhs
	cmp	qword [rsp + 8], 2
	je	.tmbrb_rhs
	call	tensor_matmul_backward_f32
	jmp	.tmbrb_out
.tmbrb_lhs:
	call	tensor_matmul_backward_lhs_f32
	jmp	.tmbrb_out
.tmbrb_rhs:
	call	tensor_matmul_backward_rhs_f32
.tmbrb_out:
	add	rsp, 16
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; Internal action used only for single-consumer fused temporaries.
; rdi=Tensor*. Clears tensor.grad when present, returns eax=0.
tensor_fused_zero_grad_f32:
	mov	rdx, [rdi + 8]
	test	rdx, rdx
	jz	.tfz_done
	mov	rcx, [rdi + 16]
	imul	rcx, [rdi + 24]
	xor	eax, eax
.tfz_loop:
	mov	dword [rdx], eax
	add	rdx, 4
	dec	rcx
	jnz	.tfz_loop
.tfz_done:
	xor	eax, eax
	ret
