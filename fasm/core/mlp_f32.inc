; Minimal f32 primitives for multilayer perceptrons.
;
; Dense weights are row-major: weights[output][input]. Activations are kept
; separate from the affine operation so callers can compose layers and, later,
; record the same operations for automatic differentiation.
;
; C ABI:
;   int mlp_dense_f32(const float *input, const float *weights,
;                     const float *bias, float *output,
;                     uint64_t input_count, uint64_t output_count)
;     rdi=input, rsi=weights, rdx=bias (nullable), rcx=output
;     r8=input_count, r9=output_count
;     eax = 0 on success, -1 for null required pointers or zero dimensions
;     preserves rbx, r12-r15; clobbers rax, rdx, r10-r11, xmm0-xmm1
;
;   int mlp_relu_f32_inplace(float *values, uint64_t count)
;     rdi=values, rsi=count
;     eax = 0 on success, -1 for null values or zero count
;     clobbers rcx, xmm0-xmm1

mlp_dense_f32:
	test	rdi, rdi
	jz	.mdf_bad
	test	rsi, rsi
	jz	.mdf_bad
	test	rcx, rcx
	jz	.mdf_bad
	test	r8, r8
	jz	.mdf_bad
	test	r9, r9
	jz	.mdf_bad
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
	mov	rbx, r8
.mdf_output:
	mov	r10, r12
	mov	r11, r13
	mov	rcx, rbx
	xorps	xmm0, xmm0
	test	r14, r14
	jz	.mdf_dot
	movss	xmm0, dword [r14]
.mdf_dot:
	movss	xmm1, dword [r10]
	mulss	xmm1, dword [r11]
	addss	xmm0, xmm1
	add	r10, 4
	add	r11, 4
	dec	rcx
	jnz	.mdf_dot
	movss	dword [r15], xmm0
	lea	r13, [r13 + rbx * 4]
	test	r14, r14
	jz	.mdf_no_bias_advance
	add	r14, 4
.mdf_no_bias_advance:
	add	r15, 4
	dec	r9
	jnz	.mdf_output
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	xor	eax, eax
	ret
.mdf_bad:
	mov	eax, -1
	ret

mlp_relu_f32_inplace:
	test	rdi, rdi
	jz	.mrf_bad
	test	rsi, rsi
	jz	.mrf_bad
	xorps	xmm1, xmm1
	mov	rcx, rsi
.mrf_loop:
	movss	xmm0, dword [rdi]
	maxss	xmm0, xmm1
	movss	dword [rdi], xmm0
	add	rdi, 4
	dec	rcx
	jnz	.mrf_loop
	xor	eax, eax
	ret
.mrf_bad:
	mov	eax, -1
	ret
