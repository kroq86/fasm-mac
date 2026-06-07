; logvec embedding payload validation (no CRC — envelope is logbus/Zig).
;
; int lb_logvec_payload_validate(
;   const uint8_t *payload, uint64_t len,
;   uint32_t *out_dim, uint64_t *out_doc_id, const float **out_vector)
; rdi payload, rsi len, rdx out_dim, rcx out_doc_id, r8 out_vector ptr slot
; eax = 0 ok, -1 bad args/dim/len

LOGVEC_DIM_MIN equ 1
LOGVEC_DIM_MAX equ 4096

lb_logvec_payload_validate:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rbx, rsi
	test	r12, r12
	jz	.lvp_bad
	cmp	rbx, 4
	jb	.lvp_bad
	mov	eax, dword [r12]
	cmp	eax, LOGVEC_DIM_MIN
	jb	.lvp_bad
	cmp	eax, LOGVEC_DIM_MAX
	ja	.lvp_bad
	mov	r11d, eax
	mov	r10, r11
	shl	r10, 2
	add	r10, 4
	cmp	rbx, r10
	je	.lvp_no_doc
	mov	rax, r10
	add	rax, 8
	cmp	rbx, rax
	jne	.lvp_bad
	mov	rax, [r12 + r10]
	jmp	.lvp_store_doc
.lvp_no_doc:
	mov	rax, 0FFFFFFFFFFFFFFFFh
.lvp_store_doc:
	test	rdx, rdx
	jz	.lvp_skip_dim
	mov	dword [rdx], r11d
.lvp_skip_dim:
	test	rcx, rcx
	jz	.lvp_skip_id
	mov	[rcx], rax
.lvp_skip_id:
	test	r8, r8
	jz	.lvp_ok
	lea	rax, [r12 + 4]
	mov	[r8], rax
.lvp_ok:
	xor	eax, eax
	jmp	.lvp_out
.lvp_bad:
	mov	eax, -1
.lvp_out:
	pop	r12
	pop	rbx
	ret
