; f32 vector math for logvec.
;
; C ABI: float in xmm0, int32 status in eax.
;
; CPU flags (lb_vec_cpu_flags): 0 = uninitialized, 1 = scalar, 2 = avx2

VEC_F32_MAX_TOPK equ 256
VEC_F32_SAVED_REGS equ 40
VEC_CPU_UNINIT equ 0
VEC_CPU_SCALAR equ 1
VEC_CPU_AVX2 equ 2

; int lb_vec_has_avx2(void) — eax = 1 on x86_64 mac targets (Rosetta-safe)
lb_vec_has_avx2:
	mov	eax, 1
	ret

; void lb_vec_set_simd_mode(int mode) — rdi: 0=auto, 1=scalar, 2=avx2
lb_vec_set_simd_mode:
	cmp	edi, VEC_CPU_AVX2
	je	.v32set_avx2
	cmp	edi, VEC_CPU_SCALAR
	je	.v32set_scalar
	mov	dword [lb_vec_cpu_flags], VEC_CPU_UNINIT
	ret
.v32set_avx2:
	mov	dword [lb_vec_cpu_flags], VEC_CPU_AVX2
	ret
.v32set_scalar:
	mov	dword [lb_vec_cpu_flags], VEC_CPU_SCALAR
	ret

; void lb_vec_init_cpu_flags(void)
lb_vec_init_cpu_flags:
	push	rbx
	call	lb_vec_has_avx2
	test	eax, eax
	jz	.v32init_scalar
	mov	dword [lb_vec_cpu_flags], VEC_CPU_AVX2
	jmp	.v32init_done
.v32init_scalar:
	mov	dword [lb_vec_cpu_flags], VEC_CPU_SCALAR
.v32init_done:
	pop	rbx
	ret

; float lb_vec_dot_f32_scalar(const float *a, const float *b, uint64_t len)
; rdi = a, rsi = b, rdx = len
lb_vec_dot_f32_scalar:
	xorps	xmm0, xmm0
	test	rdx, rdx
	jz	.v32ds_done
	mov	rcx, rdx
.v32ds_loop:
	movss	xmm1, dword [rdi]
	movss	xmm2, dword [rsi]
	mulss	xmm1, xmm2
	addss	xmm0, xmm1
	add	rdi, 4
	add	rsi, 4
	dec	rcx
	jnz	.v32ds_loop
.v32ds_done:
	ret

; ymm horizontal sum of 8 floats into xmm0[0]
lb_vec_reduce_ymm_ps:
	vextractf128 xmm1, ymm0, 1
	vaddps	xmm0, xmm0, xmm1
	vhaddps	xmm0, xmm0, xmm0
	vhaddps	xmm0, xmm0, xmm0
	ret

; float lb_vec_dot_f32_avx2(const float *a, const float *b, uint64_t len)
; rdi = a, rsi = b, rdx = len
lb_vec_dot_f32_avx2:
	vxorps	ymm0, ymm0, ymm0
	test	rdx, rdx
	jz	.v32da_done
	mov	rcx, rdx
	cmp	rcx, 8
	jb	.v32da_tail
.v32da_loop8:
	vmovups	ymm1, [rdi]
	vmovups	ymm2, [rsi]
	vfmadd231ps ymm0, ymm1, ymm2
	add	rdi, 32
	add	rsi, 32
	sub	rcx, 8
	cmp	rcx, 8
	jae	.v32da_loop8
.v32da_tail:
	test	rcx, rcx
	jz	.v32da_reduce
.v32da_tail_loop:
	movss	xmm1, dword [rdi]
	movss	xmm2, dword [rsi]
	mulss	xmm1, xmm2
	addss	xmm0, xmm1
	add	rdi, 4
	add	rsi, 4
	dec	rcx
	jnz	.v32da_tail_loop
.v32da_reduce:
	test	rdx, rdx
	jbe	.v32da_done
	cmp	rdx, 8
	jb	.v32da_done
	call	lb_vec_reduce_ymm_ps
.v32da_done:
	vzeroupper
	ret

; float lb_vec_dot_f32(const float *a, const float *b, uint64_t len)
; rdi = a, rsi = b, rdx = len
lb_vec_dot_f32:
	cmp	dword [lb_vec_cpu_flags], VEC_CPU_UNINIT
	je	.v32d_init
.v32d_dispatch:
	cmp	dword [lb_vec_cpu_flags], VEC_CPU_AVX2
	je	lb_vec_dot_f32_avx2
	jmp	lb_vec_dot_f32_scalar
.v32d_init:
	call	lb_vec_init_cpu_flags
	jmp	.v32d_dispatch

; float lb_vec_norm_f32_scalar(const float *v, uint64_t len)
; rdi = v, rsi = len
lb_vec_norm_f32_scalar:
	xorps	xmm0, xmm0
	test	rsi, rsi
	jz	.v32ns_done
	mov	rcx, rsi
.v32ns_loop:
	movss	xmm1, dword [rdi]
	mulss	xmm1, xmm1
	addss	xmm0, xmm1
	add	rdi, 4
	dec	rcx
	jnz	.v32ns_loop
	sqrtss	xmm0, xmm0
.v32ns_done:
	ret

; float lb_vec_norm_f32_avx2(const float *v, uint64_t len)
; rdi = v, rsi = len
lb_vec_norm_f32_avx2:
	vxorps	ymm0, ymm0, ymm0
	mov	rcx, rsi
	test	rcx, rcx
	jz	.v32na_done
	cmp	rcx, 8
	jb	.v32na_tail
.v32na_loop8:
	vmovups	ymm1, [rdi]
	vfmadd231ps ymm0, ymm1, ymm1
	add	rdi, 32
	sub	rcx, 8
	cmp	rcx, 8
	jae	.v32na_loop8
.v32na_tail:
	test	rcx, rcx
	jz	.v32na_reduce
.v32na_tail_loop:
	movss	xmm1, dword [rdi]
	mulss	xmm1, xmm1
	addss	xmm0, xmm1
	add	rdi, 4
	dec	rcx
	jnz	.v32na_tail_loop
.v32na_reduce:
	test	rsi, rsi
	jbe	.v32na_done
	cmp	rsi, 8
	jb	.v32na_sqrt
	call	lb_vec_reduce_ymm_ps
.v32na_sqrt:
	sqrtss	xmm0, xmm0
.v32na_done:
	vzeroupper
	ret

; float lb_vec_norm_f32(const float *v, uint64_t len)
; rdi = v, rsi = len
lb_vec_norm_f32:
	cmp	dword [lb_vec_cpu_flags], VEC_CPU_UNINIT
	je	.v32n_init
.v32n_dispatch:
	cmp	dword [lb_vec_cpu_flags], VEC_CPU_AVX2
	je	lb_vec_norm_f32_avx2
	jmp	lb_vec_norm_f32_scalar
.v32n_init:
	call	lb_vec_init_cpu_flags
	jmp	.v32n_dispatch

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
;   xmm6 = inv_qnorm (callee-saved), xmm7 = row score, xmm8 = heap min
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
	xorps	xmm1, xmm1
	ucomiss	xmm0, xmm1
	je	.v32t_bad
	movaps	xmm2, xmm0
	ucomiss	xmm0, xmm2
	jp	.v32t_bad
	movss	xmm6, dword [lb_vec_one_f32]
	divss	xmm6, xmm0
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
	movss	xmm5, dword [r11 + rax * 4]
	xorps	xmm1, xmm1
	ucomiss	xmm5, xmm1
	jbe	.v32t_bad
	movaps	xmm2, xmm5
	ucomiss	xmm5, xmm2
	jp	.v32t_bad
	mulss	xmm0, xmm6
	divss	xmm0, xmm5
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
	movss	xmm5, dword [r11 + rax * 4]
	xorps	xmm1, xmm1
	ucomiss	xmm5, xmm1
	jbe	.v32t_bad
	movaps	xmm2, xmm5
	ucomiss	xmm5, xmm2
	jp	.v32t_bad
	mulss	xmm0, xmm6
	divss	xmm0, xmm5
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

; int lb_vec_topk_cosine_lv(
;   const float *query, const uint8_t *records,
;   uint64_t count, uint64_t dim, uint64_t k, uint64_t record_stride,
;   uint32_t *out_index, float *out_score)
; rdi query, rsi records, rdx count, rcx dim, r8 k, r9 record_stride
; stack: [rbp+16] out_index, [rbp+24] out_score
; eax = 0 ok, -1 bad args/norm
;
; .lv record: u64 doc_id | f32 norm | u32 reserved | f32 vector[dim]
; r14 walks record base; vector at r14+16, norm at r14+8
; xmm6 = inv_qnorm
lb_vec_topk_cosine_lv:
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
	mov	r15, rdx
	mov	rbx, rcx
	mov	r11, r9
	test	r15, r15
	jz	.v32tlv_bad
	test	rbx, rbx
	jz	.v32tlv_bad
	test	r8, r8
	jz	.v32tlv_bad
	cmp	r8, VEC_F32_MAX_TOPK
	ja	.v32tlv_bad
	test	r11, r11
	jz	.v32tlv_bad
	mov	r13, r8
	cmp	r13, r15
	jbe	.v32tlv_k_ok
	mov	r13, r15
.v32tlv_k_ok:
	mov	rdi, r10
	mov	rsi, rbx
	call	lb_vec_norm_f32
	xorps	xmm1, xmm1
	ucomiss	xmm0, xmm1
	je	.v32tlv_bad
	movaps	xmm2, xmm0
	ucomiss	xmm0, xmm2
	jp	.v32tlv_bad
	movss	xmm6, dword [lb_vec_one_f32]
	divss	xmm6, xmm0
	mov	byte [rbp - 9], 0
	movss	xmm5, dword [r14 + 8]
	movss	xmm7, dword [lb_vec_one_m_eps]
	ucomiss	xmm7, xmm5
	ja	.v32tlv_unit_no
	movss	xmm7, dword [lb_vec_one_p_eps]
	ucomiss	xmm5, xmm7
	ja	.v32tlv_unit_no
	mov	byte [rbp - 9], 1
.v32tlv_unit_no:
	mov	rax, r13
	shl	rax, 3
	add	rax, 15
	and	rax, -16
	sub	rsp, rax
	mov	r12, rsp
	xor	rax, rax
.v32tlv_seed:
	cmp	rax, r13
	jae	.v32tlv_seed_done
	mov	rdi, r10
	lea	rsi, [r14 + 16]
	mov	rdx, rbx
	call	lb_vec_dot_f32
	movss	xmm5, dword [r14 + 8]
	xorps	xmm1, xmm1
	ucomiss	xmm5, xmm1
	jbe	.v32tlv_bad
	movaps	xmm2, xmm5
	ucomiss	xmm5, xmm2
	jp	.v32tlv_bad
	call	.v32tlv_score
	movss	dword [r12 + rax * 8], xmm0
	mov	dword [r12 + rax * 8 + 4], eax
	inc	rax
	add	r14, r11
	jmp	.v32tlv_seed
.v32tlv_seed_done:
	mov	rax, r13
.v32tlv_scan:
	cmp	rax, r15
	jae	.v32tlv_copy
	mov	rdi, r10
	lea	rsi, [r14 + 16]
	mov	rdx, rbx
	call	lb_vec_dot_f32
	movss	xmm5, dword [r14 + 8]
	xorps	xmm1, xmm1
	ucomiss	xmm5, xmm1
	jbe	.v32tlv_bad
	movaps	xmm2, xmm5
	ucomiss	xmm5, xmm2
	jp	.v32tlv_bad
	call	.v32tlv_score
	movaps	xmm7, xmm0
	movss	xmm8, dword [r12]
	xor	r9d, r9d
	mov	ecx, 1
	cmp	r13, 1
	jbe	.v32tlv_try_replace
.v32tlv_find_min:
	cmp	rcx, r13
	jae	.v32tlv_try_replace
	movss	xmm0, dword [r12 + rcx * 8]
	ucomiss	xmm0, xmm8
	jae	.v32tlv_min_skip
	movaps	xmm8, xmm0
	mov	r9d, ecx
.v32tlv_min_skip:
	inc	rcx
	jmp	.v32tlv_find_min
.v32tlv_try_replace:
	movaps	xmm0, xmm7
	ucomiss	xmm0, xmm8
	jbe	.v32tlv_scan_next
	movss	dword [r12 + r9 * 8], xmm0
	mov	dword [r12 + r9 * 8 + 4], eax
.v32tlv_scan_next:
	inc	rax
	add	r14, r11
	jmp	.v32tlv_scan
.v32tlv_copy:
	xor	rax, rax
.v32tlv_copy_loop:
	cmp	rax, r13
	jae	.v32tlv_ok
	mov	ecx, dword [r12 + rax * 8 + 4]
	mov	rdx, [rbp + 16]
	mov	dword [rdx + rax * 4], ecx
	movss	xmm0, dword [r12 + rax * 8]
	mov	rdx, [rbp + 24]
	movss	dword [rdx + rax * 4], xmm0
	inc	rax
	jmp	.v32tlv_copy_loop
.v32tlv_ok:
	xor	eax, eax
	jmp	.v32tlv_out
.v32tlv_bad:
	mov	eax, -1
.v32tlv_out:
	mov	rsp, rbp
	sub	rsp, VEC_F32_SAVED_REGS
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	pop	rbp
	ret

; xmm0 = dot, xmm5 = row norm, xmm6 = inv_qnorm; byte [rbp-9] = unit vectors
.v32tlv_score:
	mulss	xmm0, xmm6
	cmp	byte [rbp - 9], 0
	je	.v32tlv_score_div
	ret
.v32tlv_score_div:
	divss	xmm0, xmm5
	ret
