; Experimental AVX2/FMA matmul and runtime dispatch using the scalar ABI.

format ELF64

section '.text' executable

public tensor_cpu_features_f32
public tensor_matmul_forward_auto_f32
public tensor_matmul_forward_avx2_fma_f32
public tensor_matmul_backward_auto_f32
public tensor_matmul_backward_avx2_fma_f32

extrn tensor_matmul_forward_f32
extrn tensor_matmul_backward_f32
extrn tensor_matmul_validate

TENSOR_DATA equ 0
TENSOR_GRAD equ 8
TENSOR_ROWS equ 16
TENSOR_COLS equ 24
TENSOR_FEATURE_AVX2_FMA equ 1

; uint32_t tensor_cpu_features_f32(void)
; bit 0: AVX2 + FMA available and XMM/YMM state enabled by the OS.
tensor_cpu_features_f32:
	push	rbx
	xor	eax, eax
	cpuid
	cmp	eax, 7
	jb	.tcf_none
	mov	eax, 1
	cpuid
	mov	eax, ecx
	and	eax, (1 shl 12) or (1 shl 27) or (1 shl 28)
	cmp	eax, (1 shl 12) or (1 shl 27) or (1 shl 28)
	jne	.tcf_none
	xor	ecx, ecx
	xgetbv
	and	eax, 6
	cmp	eax, 6
	jne	.tcf_none
	mov	eax, 7
	xor	ecx, ecx
	cpuid
	bt	ebx, 5
	jnc	.tcf_none
	mov	eax, TENSOR_FEATURE_AVX2_FMA
	pop	rbx
	ret
.tcf_none:
	xor	eax, eax
	pop	rbx
	ret

; int tensor_matmul_forward_auto_f32(const Tensor *a, const Tensor *b,
;                                    Tensor *out)
; Same ABI as the scalar kernel; selects AVX2/FMA only when safe.
tensor_matmul_forward_auto_f32:
	push	rdi
	push	rsi
	push	rdx
	call	tensor_cpu_features_f32
	pop	rdx
	pop	rsi
	pop	rdi
	test	eax, TENSOR_FEATURE_AVX2_FMA
	jz	tensor_matmul_forward_f32
	jmp	tensor_matmul_forward_avx2_fma_f32

; int tensor_matmul_forward_avx2_fma_f32(const Tensor *a, const Tensor *b,
;                                        Tensor *out)
; Vectorizes eight adjacent output columns. Any remaining columns use scalar
; FMA, so tensor dimensions never need to be multiples of eight.
tensor_matmul_forward_avx2_fma_f32:
	sub	rsp, 8
	call	tensor_matmul_validate
	add	rsp, 8
	test	eax, eax
	jnz	.tma_out
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
	mov	r11, [rsi + TENSOR_COLS]
	xor	r8, r8
.tma_i:
	; Clear one output row.
	mov	rax, r8
	imul	rax, r11
	lea	rdi, [r14 + rax * 4]
	mov	rcx, r11
	xor	eax, eax
	rep stosd
	xor	r9, r9
.tma_k:
	mov	rax, r8
	imul	rax, rbx
	add	rax, r9
	vbroadcastss ymm0, dword [r12 + rax * 4]
	mov	rax, r9
	imul	rax, r11
	lea	rdx, [r13 + rax * 4]
	mov	rax, r8
	imul	rax, r11
	lea	rdi, [r14 + rax * 4]
	xor	r10, r10
.tma_vec:
	lea	rax, [r10 + 8]
	cmp	rax, r11
	ja	.tma_tail
	vmovups ymm1, yword [rdx + r10 * 4]
	vmovups ymm2, yword [rdi + r10 * 4]
	vfmadd231ps ymm2, ymm0, ymm1
	vmovups yword [rdi + r10 * 4], ymm2
	add	r10, 8
	jmp	.tma_vec
.tma_tail:
	cmp	r10, r11
	jae	.tma_k_next
	vmovss	xmm1, dword [rdx + r10 * 4]
	vmovss	xmm2, dword [rdi + r10 * 4]
	vfmadd231ss xmm2, xmm0, xmm1
	vmovss	dword [rdi + r10 * 4], xmm2
	inc	r10
	jmp	.tma_tail
.tma_k_next:
	inc	r9
	cmp	r9, rbx
	jb	.tma_k
	inc	r8
	cmp	r8, r15
	jb	.tma_i
	vzeroupper
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	xor	eax, eax
.tma_out:
	ret

; int tensor_matmul_backward_auto_f32(Tensor *a, Tensor *b,
;                                     const Tensor *out)
tensor_matmul_backward_auto_f32:
	push	rdi
	push	rsi
	push	rdx
	call	tensor_cpu_features_f32
	pop	rdx
	pop	rsi
	pop	rdi
	test	eax, TENSOR_FEATURE_AVX2_FMA
	jz	tensor_matmul_backward_f32
	jmp	tensor_matmul_backward_avx2_fma_f32

; int tensor_matmul_backward_avx2_fma_f32(Tensor *a, Tensor *b,
;                                         const Tensor *out)
; Accumulates dA and dB. Adjacent output columns are vectorized in both
; gradients; scalar tails handle dimensions that are not multiples of eight.
tensor_matmul_backward_avx2_fma_f32:
	sub	rsp, 8
	call	tensor_matmul_validate
	add	rsp, 8
	test	eax, eax
	jnz	.tmba_out
	cmp	qword [rdi + TENSOR_GRAD], 0
	je	.tmba_bad
	cmp	qword [rsi + TENSOR_GRAD], 0
	je	.tmba_bad
	cmp	qword [rdx + TENSOR_GRAD], 0
	je	.tmba_bad
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

	; dA[i,k] += dot(dOut[i,:], B[k,:])
	xor	r8, r8
.tmba_ai:
	xor	r9, r9
.tmba_ak:
	vxorps	ymm0, ymm0, ymm0
	xor	r10, r10
	mov	rax, r8
	imul	rax, [rbp - 64]
	mov	rdx, [rbp - 48]
	lea	rdx, [rdx + rax * 4]
	mov	rax, r9
	imul	rax, [rbp - 64]
	lea	rdi, [r14 + rax * 4]
.tmba_avec:
	lea	rax, [r10 + 8]
	cmp	rax, [rbp - 64]
	ja	.tmba_areduce
	vmovups ymm1, yword [rdx + r10 * 4]
	vmovups ymm2, yword [rdi + r10 * 4]
	vfmadd231ps ymm0, ymm1, ymm2
	add	r10, 8
	jmp	.tmba_avec
.tmba_areduce:
	vextractf128 xmm1, ymm0, 1
	vaddps	xmm0, xmm0, xmm1
	vhaddps	xmm0, xmm0, xmm0
	vhaddps	xmm0, xmm0, xmm0
.tmba_atail:
	cmp	r10, [rbp - 64]
	jae	.tmba_astore
	vmovss	xmm1, dword [rdx + r10 * 4]
	vmovss	xmm2, dword [rdi + r10 * 4]
	vfmadd231ss xmm0, xmm1, xmm2
	inc	r10
	jmp	.tmba_atail
.tmba_astore:
	mov	rax, r8
	imul	rax, rbx
	add	rax, r9
	vaddss	xmm0, xmm0, dword [r13 + rax * 4]
	vmovss	dword [r13 + rax * 4], xmm0
	inc	r9
	cmp	r9, rbx
	jb	.tmba_ak
	inc	r8
	cmp	r8, [rbp - 56]
	jb	.tmba_ai

	; dB[k,j:j+8] += sum_i A[i,k] * dOut[i,j:j+8]
	xor	r8, r8
.tmba_bk:
	xor	r9, r9
.tmba_bvec:
	lea	rax, [r9 + 8]
	cmp	rax, [rbp - 64]
	ja	.tmba_btail
	mov	rax, r8
	imul	rax, [rbp - 64]
	add	rax, r9
	vmovups ymm0, yword [r15 + rax * 4]
	xor	rcx, rcx
.tmba_bi:
	mov	rax, rcx
	imul	rax, rbx
	add	rax, r8
	vbroadcastss ymm1, dword [r12 + rax * 4]
	mov	rax, rcx
	imul	rax, [rbp - 64]
	add	rax, r9
	mov	rdx, [rbp - 48]
	vmovups ymm2, yword [rdx + rax * 4]
	vfmadd231ps ymm0, ymm1, ymm2
	inc	rcx
	cmp	rcx, [rbp - 56]
	jb	.tmba_bi
	mov	rax, r8
	imul	rax, [rbp - 64]
	add	rax, r9
	vmovups yword [r15 + rax * 4], ymm0
	add	r9, 8
	jmp	.tmba_bvec
.tmba_btail:
	cmp	r9, [rbp - 64]
	jae	.tmba_bnext
	mov	rax, r8
	imul	rax, [rbp - 64]
	add	rax, r9
	vmovss	xmm0, dword [r15 + rax * 4]
	xor	rcx, rcx
.tmba_btail_i:
	mov	rax, rcx
	imul	rax, rbx
	add	rax, r8
	vmovss	xmm1, dword [r12 + rax * 4]
	mov	rax, rcx
	imul	rax, [rbp - 64]
	add	rax, r9
	mov	rdx, [rbp - 48]
	vmovss	xmm2, dword [rdx + rax * 4]
	vfmadd231ss xmm0, xmm1, xmm2
	inc	rcx
	cmp	rcx, [rbp - 56]
	jb	.tmba_btail_i
	mov	rax, r8
	imul	rax, [rbp - 64]
	add	rax, r9
	vmovss	dword [r15 + rax * 4], xmm0
	inc	r9
	jmp	.tmba_btail
.tmba_bnext:
	inc	r8
	cmp	r8, rbx
	jb	.tmba_bk

	vzeroupper
	add	rsp, 32
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	pop	rbp
	xor	eax, eax
.tmba_out:
	ret
.tmba_bad:
	mov	eax, -1
	ret
