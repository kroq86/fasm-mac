; Scalar tensor matmul forward/backward kernels. Include in a text section.

TENSOR_DATA equ 0
TENSOR_GRAD equ 8
TENSOR_ROWS equ 16
TENSOR_COLS equ 24

; int tensor_matmul_forward_f32(const Tensor *a, const Tensor *b, Tensor *out)
; out = a @ b. Returns 0, or -1 for null/empty/mismatched descriptors.
tensor_matmul_forward_f32:
	sub	rsp, 8
	call	tensor_matmul_validate
	add	rsp, 8
	test	eax, eax
	jnz	.tmf_out
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, [rdi + TENSOR_DATA]
	mov	r13, [rsi + TENSOR_DATA]
	mov	r14, [rdx + TENSOR_DATA]
	mov	r15, [rdi + TENSOR_ROWS]
	mov	rbx, [rdi + TENSOR_COLS]
	mov	r10, [rsi + TENSOR_COLS]
	xor	r8, r8
.tmf_i:
	xor	r9, r9
.tmf_j:
	xorps	xmm0, xmm0
	xor	rcx, rcx
.tmf_k:
	mov	rax, r8
	imul	rax, rbx
	add	rax, rcx
	movss	xmm1, dword [r12 + rax * 4]
	mov	rax, rcx
	imul	rax, r10
	add	rax, r9
	mulss	xmm1, dword [r13 + rax * 4]
	addss	xmm0, xmm1
	inc	rcx
	cmp	rcx, rbx
	jb	.tmf_k
	mov	rax, r8
	imul	rax, r10
	add	rax, r9
	movss	dword [r14 + rax * 4], xmm0
	inc	r9
	cmp	r9, r10
	jb	.tmf_j
	inc	r8
	cmp	r8, r15
	jb	.tmf_i
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	xor	eax, eax
.tmf_out:
	ret

; int tensor_matmul_backward_f32(Tensor *a, Tensor *b, const Tensor *out)
; Accumulates: a.grad += out.grad @ transpose(b)
;              b.grad += transpose(a) @ out.grad
tensor_matmul_backward_f32:
	sub	rsp, 8
	call	tensor_matmul_validate
	add	rsp, 8
	test	eax, eax
	jnz	.tmb_out
	cmp	qword [rdi + TENSOR_GRAD], 0
	je	.tmb_bad
	cmp	qword [rsi + TENSOR_GRAD], 0
	je	.tmb_bad
	cmp	qword [rdx + TENSOR_GRAD], 0
	je	.tmb_bad
	push	rbp
	mov	rbp, rsp
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 32
	mov	r12, [rdi + TENSOR_DATA]
	mov	r13, [rdi + TENSOR_GRAD]
	mov	r14, [rsi + TENSOR_DATA]
	mov	r15, [rsi + TENSOR_GRAD]
	mov	rax, [rdx + TENSOR_GRAD]
	mov	[rbp - 48], rax
	mov	rax, [rdi + TENSOR_ROWS]
	mov	[rbp - 56], rax
	mov	rbx, [rdi + TENSOR_COLS]
	mov	rax, [rsi + TENSOR_COLS]
	mov	[rbp - 64], rax

; dA[i,k] += sum_j dOut[i,j] * B[k,j]
	xor	r8, r8
.tmb_ai:
	xor	r9, r9
.tmb_ak:
	xorps	xmm0, xmm0
	xor	rcx, rcx
.tmb_aj:
	mov	r10, r8
	imul	r10, [rbp - 64]
	add	r10, rcx
	mov	r11, [rbp - 48]
	movss	xmm1, dword [r11 + r10 * 4]
	mov	r10, r9
	imul	r10, [rbp - 64]
	add	r10, rcx
	mulss	xmm1, dword [r14 + r10 * 4]
	addss	xmm0, xmm1
	inc	rcx
	cmp	rcx, [rbp - 64]
	jb	.tmb_aj
	mov	r10, r8
	imul	r10, rbx
	add	r10, r9
	addss	xmm0, dword [r13 + r10 * 4]
	movss	dword [r13 + r10 * 4], xmm0
	inc	r9
	cmp	r9, rbx
	jb	.tmb_ak
	inc	r8
	cmp	r8, [rbp - 56]
	jb	.tmb_ai

; dB[k,j] += sum_i A[i,k] * dOut[i,j]
	xor	r8, r8
.tmb_bk:
	xor	r9, r9
.tmb_bj:
	xorps	xmm0, xmm0
	xor	rcx, rcx
.tmb_bi:
	mov	r10, rcx
	imul	r10, rbx
	add	r10, r8
	movss	xmm1, dword [r12 + r10 * 4]
	mov	r10, rcx
	imul	r10, [rbp - 64]
	add	r10, r9
	mov	r11, [rbp - 48]
	mulss	xmm1, dword [r11 + r10 * 4]
	addss	xmm0, xmm1
	inc	rcx
	cmp	rcx, [rbp - 56]
	jb	.tmb_bi
	mov	r10, r8
	imul	r10, [rbp - 64]
	add	r10, r9
	addss	xmm0, dword [r15 + r10 * 4]
	movss	dword [r15 + r10 * 4], xmm0
	inc	r9
	cmp	r9, [rbp - 64]
	jb	.tmb_bj
	inc	r8
	cmp	r8, rbx
	jb	.tmb_bk

	add	rsp, 32
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	pop	rbp
	xor	eax, eax
.tmb_out:
	ret
.tmb_bad:
	mov	eax, -1
	ret

; Shared descriptor validation. Preserves rdi/rsi/rdx; returns eax.
tensor_matmul_validate:
	test	rdi, rdi
	jz	.tv_bad
	test	rsi, rsi
	jz	.tv_bad
	test	rdx, rdx
	jz	.tv_bad
	cmp	qword [rdi + TENSOR_DATA], 0
	je	.tv_bad
	cmp	qword [rsi + TENSOR_DATA], 0
	je	.tv_bad
	cmp	qword [rdx + TENSOR_DATA], 0
	je	.tv_bad
	mov	rax, [rdi + TENSOR_ROWS]
	test	rax, rax
	jz	.tv_bad
	cmp	rax, [rdx + TENSOR_ROWS]
	jne	.tv_bad
	mov	rax, [rdi + TENSOR_COLS]
	test	rax, rax
	jz	.tv_bad
	cmp	rax, [rsi + TENSOR_ROWS]
	jne	.tv_bad
	mov	rax, [rsi + TENSOR_COLS]
	test	rax, rax
	jz	.tv_bad
	cmp	rax, [rdx + TENSOR_COLS]
	jne	.tv_bad
	xor	eax, eax
	ret
.tv_bad:
	mov	eax, -1
	ret
