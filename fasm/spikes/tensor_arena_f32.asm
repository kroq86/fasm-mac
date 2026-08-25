; Experimental caller-backed arena with stale-view detection.

format ELF64

section '.text' executable

public tensor_arena_init_f32
public tensor_arena_reset_f32
public tensor_arena_alloc_f32
public tensor_owned_validate_f32

ARENA_BASE equ 0
ARENA_CAP equ 8
ARENA_USED equ 16
ARENA_GENERATION equ 24

TENSOR_DATA equ 0
TENSOR_GRAD equ 8
TENSOR_ROWS equ 16
TENSOR_COLS equ 24
TENSOR_OWNER equ 32
TENSOR_GENERATION equ 40

; int tensor_arena_init_f32(Arena *arena, void *buffer, uint64_t capacity)
tensor_arena_init_f32:
	test	rdi, rdi
	jz	.tai_bad
	test	rsi, rsi
	jz	.tai_bad
	test	rdx, rdx
	jz	.tai_bad
	mov	[rdi + ARENA_BASE], rsi
	mov	[rdi + ARENA_CAP], rdx
	mov	qword [rdi + ARENA_USED], 0
	mov	qword [rdi + ARENA_GENERATION], 1
	xor	eax, eax
	ret
.tai_bad:
	mov	eax, -1
	ret

; int tensor_arena_reset_f32(Arena *arena)
; Invalidates every owned tensor view from the previous generation.
tensor_arena_reset_f32:
	test	rdi, rdi
	jz	.tar_bad
	cmp	qword [rdi + ARENA_BASE], 0
	je	.tar_bad
	mov	qword [rdi + ARENA_USED], 0
	inc	qword [rdi + ARENA_GENERATION]
	jnz	.tar_ok
	inc	qword [rdi + ARENA_GENERATION]
.tar_ok:
	xor	eax, eax
	ret
.tar_bad:
	mov	eax, -1
	ret

; int tensor_arena_alloc_f32(Arena *arena, OwnedTensor *out,
;                            uint64_t rows, uint64_t cols,
;                            uint64_t requires_grad)
; Data and optional gradient buffers are 64-byte aligned and zero-filled.
; Returns 0, -1 bad args, -2 size overflow, -3 arena capacity exceeded.
tensor_arena_alloc_f32:
	test	rdi, rdi
	jz	.taa_bad
	test	rsi, rsi
	jz	.taa_bad
	test	rdx, rdx
	jz	.taa_bad
	test	rcx, rcx
	jz	.taa_bad
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
	sub	rsp, 16
	mov	rax, r14
	mul	r15
	test	rdx, rdx
	jnz	.taa_overflow
	mov	rdx, 3fffffffffffffffh
	cmp	rax, rdx
	ja	.taa_overflow
	shl	rax, 2
	mov	[rsp], rax
	mov	rdi, r12
	mov	rsi, rax
	call	tensor_arena_alloc_aligned64
	test	rax, rax
	jz	.taa_oom_pop
	mov	[r13 + TENSOR_DATA], rax
	mov	rdx, [rsp]
	mov	rdi, rax
	mov	rcx, rdx
	xor	eax, eax
	rep stosb
	mov	qword [r13 + TENSOR_GRAD], 0
	test	rbx, rbx
	jz	.taa_buffers_done
	mov	rdi, r12
	mov	rsi, [rsp]
	call	tensor_arena_alloc_aligned64
	test	rax, rax
	jz	.taa_oom_pop
	mov	[r13 + TENSOR_GRAD], rax
	mov	rdi, rax
	mov	rcx, [rsp]
	xor	eax, eax
	rep stosb
.taa_buffers_done:
	mov	[r13 + TENSOR_ROWS], r14
	mov	[r13 + TENSOR_COLS], r15
	mov	[r13 + TENSOR_OWNER], r12
	mov	rax, [r12 + ARENA_GENERATION]
	mov	[r13 + TENSOR_GENERATION], rax
	xor	eax, eax
	jmp	.taa_out
.taa_oom_pop:
	mov	eax, -3
	jmp	.taa_out
.taa_overflow:
	mov	eax, -2
	jmp	.taa_out
.taa_bad:
	mov	eax, -1
	ret
.taa_out:
	add	rsp, 16
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; void *tensor_arena_alloc_aligned64(Arena *arena, uint64_t bytes)
; Internal. Returns 0 on overflow/out-of-space.
tensor_arena_alloc_aligned64:
	mov	rax, [rdi + ARENA_BASE]
	add	rax, [rdi + ARENA_USED]
	jc	.taaa_fail
	add	rax, 63
	jc	.taaa_fail
	and	rax, -64
	mov	rdx, rax
	sub	rdx, [rdi + ARENA_BASE]
	jc	.taaa_fail
	mov	rcx, rdx
	add	rcx, rsi
	jc	.taaa_fail
	cmp	rcx, [rdi + ARENA_CAP]
	ja	.taaa_fail
	mov	[rdi + ARENA_USED], rcx
	ret
.taaa_fail:
	xor	eax, eax
	ret

; int tensor_owned_validate_f32(const OwnedTensor *tensor)
; Returns -4 for a stale arena generation. External views (owner=0) are valid.
tensor_owned_validate_f32:
	test	rdi, rdi
	jz	.tov_bad
	cmp	qword [rdi + TENSOR_DATA], 0
	je	.tov_bad
	mov	rax, [rdi + TENSOR_ROWS]
	test	rax, rax
	jz	.tov_bad
	mov	rcx, [rdi + TENSOR_COLS]
	test	rcx, rcx
	jz	.tov_bad
	mul	rcx
	test	rdx, rdx
	jnz	.tov_overflow
	mov	rdx, 3fffffffffffffffh
	cmp	rax, rdx
	ja	.tov_overflow
	mov	rax, [rdi + TENSOR_OWNER]
	test	rax, rax
	jz	.tov_ok
	mov	rcx, [rdi + TENSOR_GENERATION]
	cmp	rcx, [rax + ARENA_GENERATION]
	jne	.tov_stale
.tov_ok:
	xor	eax, eax
	ret
.tov_bad:
	mov	eax, -1
	ret
.tov_overflow:
	mov	eax, -2
	ret
.tov_stale:
	mov	eax, -4
	ret
