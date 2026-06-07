; C ABI vector primitives for zig-bridge perf checks.
; Mirrors fasm-handbook vec/dot_product.asm.

format ELF64

section ".text" executable

public lb_dot_product
public lb_vector_norm

; double lb_dot_product(const double *a, const double *b, uint64_t len)
; rdi = a, rsi = b, rdx = len, xmm0 = dot product
lb_dot_product:
	xorpd	xmm0, xmm0
	test	rdx, rdx
	jz	.dp_done
	mov	rcx, rdx
.dp_loop:
	movsd	xmm1, qword [rdi]
	movsd	xmm2, qword [rsi]
	mulsd	xmm1, xmm2
	addsd	xmm0, xmm1
	add	rdi, 8
	add	rsi, 8
	dec	rcx
	jnz	.dp_loop
.dp_done:
	ret

; double lb_vector_norm(const double *v, uint64_t len)
; rdi = v, rsi = len, xmm0 = euclidean norm
lb_vector_norm:
	xorpd	xmm0, xmm0
	test	rsi, rsi
	jz	.vn_done
	mov	rcx, rsi
.vn_loop:
	movsd	xmm1, qword [rdi]
	mulsd	xmm1, xmm1
	addsd	xmm0, xmm1
	add	rdi, 8
	dec	rcx
	jnz	.vn_loop
	sqrtsd	xmm0, xmm0
.vn_done:
	ret
