; f32 vector math for logvec.
;
; C ABI: float in xmm0, int32 status in eax.

VEC_F32_MAX_TOPK equ 256
VEC_F32_SAVED_REGS equ 40

; float lb_vec_dot_f32(const float *a, const float *b, uint64_t len)
; rdi = a, rsi = b, rdx = len
lb_vec_dot_f32:
	xorps	xmm0, xmm0
	test	rdx, rdx
	jz	.v32d_done
	mov	rcx, rdx
.v32d_loop:
	movss	xmm1, dword [rdi]
	movss	xmm2, dword [rsi]
	mulss	xmm1, xmm2
	addss	xmm0, xmm1
	add	rdi, 4
	add	rsi, 4
	dec	rcx
	jnz	.v32d_loop
.v32d_done:
	ret

; float lb_vec_norm_f32(const float *v, uint64_t len)
; rdi = v, rsi = len
lb_vec_norm_f32:
	xorps	xmm0, xmm0
	test	rsi, rsi
	jz	.v32n_done
	mov	rcx, rsi
.v32n_loop:
	movss	xmm1, dword [rdi]
	mulss	xmm1, xmm1
	addss	xmm0, xmm1
	add	rdi, 4
	dec	rcx
	jnz	.v32n_loop
	sqrtss	xmm0, xmm0
.v32n_done:
	ret

; int lb_vec_topk_cosine_exact(
;   const float *query, const float *vectors, const float *norms,
;   uint64_t count, uint64_t dim, uint64_t k,
;   uint32_t *out_index, float *out_score)
; rdi query, rsi vectors, rdx norms, rcx count, r8 dim, r9 k
; stack: [rbp+16] out_index, [rbp+24] out_score
; eax = 0 ok, -1 bad args
;
; Register map:
;   r10 = query, r11 = norms, r14 = vectors walk ptr
;   r15 = count, rbx = dim, r13 = k_eff, r12 = heap base
;   xmm6 = qnorm (callee-saved), xmm7 = row score, xmm8 = heap min
lb_vec_topk_cosine_exact:
	push	rbp
	mov	rbp, rsp
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	sub	rsp, 8
	mov	r10, rdi
	mov	r14, rsi
	mov	r11, rdx
	mov	r15, rcx
	mov	rbx, r8
	test	r15, r15
	jz	.v32t_bad
	test	rbx, rbx
	jz	.v32t_bad
	test	r9, r9
	jz	.v32t_bad
	cmp	r9, VEC_F32_MAX_TOPK
	ja	.v32t_bad
	mov	r13, r9
	cmp	r13, r15
	jbe	.v32t_k_ok
	mov	r13, r15
.v32t_k_ok:
	mov	rdi, r10
	mov	rsi, rbx
	call	lb_vec_norm_f32
	movaps	xmm6, xmm0
	xorps	xmm1, xmm1
	ucomiss	xmm0, xmm1
	je	.v32t_bad
	mov	rax, r13
	shl	rax, 3
	add	rax, 15
	and	rax, -16
	sub	rsp, rax
	mov	r12, rsp
	xor	rax, rax
.v32t_seed:
	cmp	rax, r13
	jae	.v32t_seed_done
	mov	rdi, r10
	mov	rsi, r14
	mov	rdx, rbx
	call	lb_vec_dot_f32
	movaps	xmm4, xmm6
	movss	xmm5, dword [r11 + rax * 4]
	mulss	xmm4, xmm5
	divss	xmm0, xmm4
	movss	dword [r12 + rax * 8], xmm0
	mov	dword [r12 + rax * 8 + 4], eax
	inc	rax
	mov	rcx, rbx
	shl	rcx, 2
	add	r14, rcx
	jmp	.v32t_seed
.v32t_seed_done:
	mov	rax, r13
.v32t_scan:
	cmp	rax, r15
	jae	.v32t_copy
	mov	rdi, r10
	mov	rsi, r14
	mov	rdx, rbx
	call	lb_vec_dot_f32
	movaps	xmm4, xmm6
	movss	xmm5, dword [r11 + rax * 4]
	mulss	xmm4, xmm5
	divss	xmm0, xmm4
	movaps	xmm7, xmm0
	movss	xmm8, dword [r12]
	xor	r9d, r9d
	mov	ecx, 1
	cmp	r13, 1
	jbe	.v32t_try_replace
.v32t_find_min:
	cmp	rcx, r13
	jae	.v32t_try_replace
	movss	xmm0, dword [r12 + rcx * 8]
	ucomiss	xmm0, xmm8
	jae	.v32t_min_skip
	movaps	xmm8, xmm0
	mov	r9d, ecx
.v32t_min_skip:
	inc	rcx
	jmp	.v32t_find_min
.v32t_try_replace:
	movaps	xmm0, xmm7
	ucomiss	xmm0, xmm8
	jbe	.v32t_scan_next
	movss	dword [r12 + r9 * 8], xmm0
	mov	dword [r12 + r9 * 8 + 4], eax
.v32t_scan_next:
	inc	rax
	mov	rcx, rbx
	shl	rcx, 2
	add	r14, rcx
	jmp	.v32t_scan
.v32t_copy:
	xor	rax, rax
.v32t_copy_loop:
	cmp	rax, r13
	jae	.v32t_ok
	mov	ecx, dword [r12 + rax * 8 + 4]
	mov	rdx, [rbp + 16]
	mov	dword [rdx + rax * 4], ecx
	movss	xmm0, dword [r12 + rax * 8]
	mov	rdx, [rbp + 24]
	movss	dword [rdx + rax * 4], xmm0
	inc	rax
	jmp	.v32t_copy_loop
.v32t_ok:
	xor	eax, eax
	jmp	.v32t_out
.v32t_bad:
	mov	eax, -1
.v32t_out:
	mov	rsp, rbp
	sub	rsp, VEC_F32_SAVED_REGS
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	pop	rbp
	ret
