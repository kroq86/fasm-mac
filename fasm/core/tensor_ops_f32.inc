; Tensor ReLU, MSE, and bias-add kernels. Requires shape validation.

TENSOR_DATA equ 0
TENSOR_GRAD equ 8
TENSOR_ROWS equ 16
TENSOR_COLS equ 24

; int tensor_relu_forward_f32(const Tensor *input, Tensor *out)
tensor_relu_forward_f32:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	sub	rsp, 8
	call	tensor_pair_same_shape
	add	rsp, 8
	test	eax, eax
	jnz	.trf_out
	mov	rdi, [r12 + TENSOR_DATA]
	mov	rsi, [r13 + TENSOR_DATA]
	mov	rcx, [r12 + TENSOR_ROWS]
	imul	rcx, [r12 + TENSOR_COLS]
	xorps	xmm1, xmm1
.trf_loop:
	movss	xmm0, dword [rdi]
	maxss	xmm0, xmm1
	movss	dword [rsi], xmm0
	add	rdi, 4
	add	rsi, 4
	dec	rcx
	jnz	.trf_loop
	xor	eax, eax
.trf_out:
	pop	r13
	pop	r12
	ret

; int tensor_relu_backward_f32(Tensor *input, const Tensor *out)
tensor_relu_backward_f32:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	sub	rsp, 8
	call	tensor_pair_same_shape
	add	rsp, 8
	test	eax, eax
	jnz	.trb_out
	cmp	qword [r12 + TENSOR_GRAD], 0
	je	.trb_bad
	cmp	qword [r13 + TENSOR_GRAD], 0
	je	.trb_bad
	mov	rdi, [r12 + TENSOR_DATA]
	mov	rsi, [r12 + TENSOR_GRAD]
	mov	rdx, [r13 + TENSOR_GRAD]
	mov	rcx, [r12 + TENSOR_ROWS]
	imul	rcx, [r12 + TENSOR_COLS]
	xorps	xmm1, xmm1
.trb_loop:
	movss	xmm0, dword [rdi]
	ucomiss	xmm0, xmm1
	jbe	.trb_next
	movss	xmm0, dword [rdx]
	addss	xmm0, dword [rsi]
	movss	dword [rsi], xmm0
.trb_next:
	add	rdi, 4
	add	rsi, 4
	add	rdx, 4
	dec	rcx
	jnz	.trb_loop
	xor	eax, eax
	jmp	.trb_out
.trb_bad:
	mov	eax, -1
.trb_out:
	pop	r13
	pop	r12
	ret

; int tensor_mse_forward_f32(const Tensor *prediction, const Tensor *target,
;                            Tensor *out_scalar)
tensor_mse_forward_f32:
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	rsi, 0
	call	tensor_shape_validate_f32
	test	eax, eax
	jnz	.tmf_out
	mov	rdi, r13
	xor	rsi, rsi
	call	tensor_shape_validate_f32
	test	eax, eax
	jnz	.tmf_out
	mov	rdi, r14
	xor	rsi, rsi
	call	tensor_shape_validate_f32
	test	eax, eax
	jnz	.tmf_out
	mov	rax, [r12 + TENSOR_ROWS]
	cmp	rax, [r13 + TENSOR_ROWS]
	jne	.tmf_bad
	mov	rcx, [r12 + TENSOR_COLS]
	cmp	rcx, [r13 + TENSOR_COLS]
	jne	.tmf_bad
	cmp	qword [r14 + TENSOR_ROWS], 1
	jne	.tmf_bad
	cmp	qword [r14 + TENSOR_COLS], 1
	jne	.tmf_bad
	imul	rcx, rax
	mov	rdi, [r12 + TENSOR_DATA]
	mov	rsi, [r13 + TENSOR_DATA]
	xorps	xmm0, xmm0
	mov	r8, rcx
.tmf_loop:
	movss	xmm1, dword [rdi]
	subss	xmm1, dword [rsi]
	mulss	xmm1, xmm1
	addss	xmm0, xmm1
	add	rdi, 4
	add	rsi, 4
	dec	r8
	jnz	.tmf_loop
	cvtsi2ss xmm1, rcx
	divss	xmm0, xmm1
	mov	rdi, [r14 + TENSOR_DATA]
	movss	dword [rdi], xmm0
	xor	eax, eax
	jmp	.tmf_out
.tmf_bad:
	mov	eax, -1
.tmf_out:
	pop	r14
	pop	r13
	pop	r12
	ret

; int tensor_mse_backward_f32(Tensor *prediction, Tensor *target,
;                             const Tensor *out_scalar)
; Target gradients are accumulated when target.grad is non-null.
tensor_mse_backward_f32:
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	cmp	qword [r12 + TENSOR_GRAD], 0
	je	.tmb_bad
	cmp	qword [r14 + TENSOR_GRAD], 0
	je	.tmb_bad
	mov	rcx, [r12 + TENSOR_ROWS]
	imul	rcx, [r12 + TENSOR_COLS]
	test	rcx, rcx
	jz	.tmb_bad
	mov	rdi, [r12 + TENSOR_DATA]
	mov	rsi, [r13 + TENSOR_DATA]
	mov	rdx, [r12 + TENSOR_GRAD]
	mov	r8, [r13 + TENSOR_GRAD]
	mov	r9, [r14 + TENSOR_GRAD]
	movss	xmm2, dword [r9]
	addss	xmm2, xmm2
	cvtsi2ss xmm3, rcx
	divss	xmm2, xmm3
	mov	r9, rcx
.tmb_loop:
	movss	xmm0, dword [rdi]
	subss	xmm0, dword [rsi]
	mulss	xmm0, xmm2
	movss	xmm1, dword [rdx]
	addss	xmm1, xmm0
	movss	dword [rdx], xmm1
	test	r8, r8
	jz	.tmb_next
	movss	xmm1, dword [r8]
	subss	xmm1, xmm0
	movss	dword [r8], xmm1
.tmb_next:
	add	rdi, 4
	add	rsi, 4
	add	rdx, 4
	test	r8, r8
	jz	.tmb_no_target_advance
	add	r8, 4
.tmb_no_target_advance:
	dec	r9
	jnz	.tmb_loop
	xor	eax, eax
	jmp	.tmb_out
.tmb_bad:
	mov	eax, -1
.tmb_out:
	pop	r14
	pop	r13
	pop	r12
	ret

; rdi/rsi Tensor*. Returns 0 for matching valid shapes.
tensor_pair_same_shape:
	push	r12
	push	r13
	sub	rsp, 8
	mov	r12, rdi
	mov	r13, rsi
	xor	rsi, rsi
	call	tensor_shape_validate_f32
	test	eax, eax
	jnz	.tps_out
	mov	rdi, r13
	xor	rsi, rsi
	call	tensor_shape_validate_f32
	test	eax, eax
	jnz	.tps_out
	mov	rax, [r12 + TENSOR_ROWS]
	cmp	rax, [r13 + TENSOR_ROWS]
	jne	.tps_bad
	mov	rax, [r12 + TENSOR_COLS]
	cmp	rax, [r13 + TENSOR_COLS]
	jne	.tps_bad
	xor	eax, eax
	jmp	.tps_out
.tps_bad:
	mov	eax, -1
.tps_out:
	add	rsp, 8
	pop	r13
	pop	r12
	ret

; int tensor_bias_add_forward_f32(const Tensor *input, const Tensor *bias,
;                                 Tensor *out)
; input/out are MxN; bias is 1xN.
tensor_bias_add_forward_f32:
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	rdi, r12
	mov	rsi, r14
	call	tensor_pair_same_shape
	test	eax, eax
	jnz	.tbaf_out
	mov	rdi, r13
	xor	rsi, rsi
	call	tensor_shape_validate_f32
	test	eax, eax
	jnz	.tbaf_out
	cmp	qword [r13 + TENSOR_ROWS], 1
	jne	.tbaf_bad
	mov	rax, [r12 + TENSOR_COLS]
	cmp	rax, [r13 + TENSOR_COLS]
	jne	.tbaf_bad
	mov	r8, [r12 + TENSOR_DATA]
	mov	r9, [r13 + TENSOR_DATA]
	mov	r10, [r14 + TENSOR_DATA]
	xor	rcx, rcx
.tbaf_i:
	xor	rdx, rdx
.tbaf_j:
	mov	rax, rcx
	imul	rax, [r12 + TENSOR_COLS]
	add	rax, rdx
	movss	xmm0, dword [r8 + rax * 4]
	addss	xmm0, dword [r9 + rdx * 4]
	movss	dword [r10 + rax * 4], xmm0
	inc	rdx
	cmp	rdx, [r12 + TENSOR_COLS]
	jb	.tbaf_j
	inc	rcx
	cmp	rcx, [r12 + TENSOR_ROWS]
	jb	.tbaf_i
	xor	eax, eax
	jmp	.tbaf_out
.tbaf_bad:
	mov	eax, -1
.tbaf_out:
	pop	r14
	pop	r13
	pop	r12
	ret

; int tensor_bias_add_backward_f32(Tensor *input, Tensor *bias,
;                                  const Tensor *out)
; Accumulates input.grad elementwise and bias.grad by row reduction.
tensor_bias_add_backward_f32:
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	cmp	qword [r12 + TENSOR_GRAD], 0
	je	.tbab_bad
	cmp	qword [r13 + TENSOR_GRAD], 0
	je	.tbab_bad
	cmp	qword [r14 + TENSOR_GRAD], 0
	je	.tbab_bad
	mov	r8, [r12 + TENSOR_GRAD]
	mov	r9, [r13 + TENSOR_GRAD]
	mov	r10, [r14 + TENSOR_GRAD]
	xor	rcx, rcx
.tbab_i:
	xor	rdx, rdx
.tbab_j:
	mov	rax, rcx
	imul	rax, [r12 + TENSOR_COLS]
	add	rax, rdx
	movss	xmm0, dword [r10 + rax * 4]
	movss	xmm1, dword [r8 + rax * 4]
	addss	xmm1, xmm0
	movss	dword [r8 + rax * 4], xmm1
	movss	xmm1, dword [r9 + rdx * 4]
	addss	xmm1, xmm0
	movss	dword [r9 + rdx * 4], xmm1
	inc	rdx
	cmp	rdx, [r12 + TENSOR_COLS]
	jb	.tbab_j
	inc	rcx
	cmp	rcx, [r12 + TENSOR_ROWS]
	jb	.tbab_i
	xor	eax, eax
	jmp	.tbab_out
.tbab_bad:
	mov	eax, -1
.tbab_out:
	pop	r14
	pop	r13
	pop	r12
	ret
