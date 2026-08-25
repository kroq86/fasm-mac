; Bump arena / region allocator.
;
; Arena storage is caller-owned. Mark/reset gives RAII-style scoped lifetime:
; allocate temporary objects, copy out durable results, then reset to mark.

ARENA_OK equ 0
ARENA_ERR equ -1

struc Arena
{
	.base dq ?
	.cap dq ?
	.used dq ?
}

virtual at 0
	_Arena Arena
	ARENA_SIZE = $
end virtual

ARENA_BASE_OFF equ 0
ARENA_CAP_OFF equ 8
ARENA_USED_OFF equ 16

; rdi = Arena*, rsi = storage base, rdx = capacity bytes
arena_init:
	mov	[rdi + ARENA_BASE_OFF], rsi
	mov	[rdi + ARENA_CAP_OFF], rdx
	mov	qword [rdi + ARENA_USED_OFF], 0
	ret

; rdi = Arena*
; rax = mark
arena_mark:
	mov	rax, [rdi + ARENA_USED_OFF]
	ret

; rdi = Arena*, rsi = mark
arena_reset:
	cmp	rsi, [rdi + ARENA_CAP_OFF]
	ja	.err
	mov	[rdi + ARENA_USED_OFF], rsi
	xor	rax, rax
	ret
.err:
	mov	rax, ARENA_ERR
	ret

; rdi = Arena*, rsi = bytes
; rax = allocation ptr, or 0 on overflow
arena_alloc:
	push	rbx
	mov	rbx, [rdi + ARENA_USED_OFF]
	add	rsi, 7
	and	rsi, not 7
	lea	rax, [rbx + rsi]
	cmp	rax, [rdi + ARENA_CAP_OFF]
	ja	.fail
	mov	[rdi + ARENA_USED_OFF], rax
	mov	rax, [rdi + ARENA_BASE_OFF]
	add	rax, rbx
	jmp	.out
.fail:
	xor	rax, rax
.out:
	pop	rbx
	ret
