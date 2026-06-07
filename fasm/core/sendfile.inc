; Coroutine-aware macOS sendfile helper.
;
; Requires platform.inc and coro.inc. The including app must emit
; `sendfile_bss` from a writeable segment.

; rdi = regular file fd, rsi = connected socket fd, rdx = byte offset,
; rcx = byte count
; rax = 0 on success, -1 on error
file_send_range_socket:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	r15, rcx
.fsrs_loop:
	test	r15, r15
	jz	.fsrs_ok
	mov	[sendfile_len], r15
	mov	rax, SYS_sendfile
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r14
	lea	r10, [sendfile_len]
	xor	r8, r8
	xor	r9, r9
	syscall
	jnc	.fsrs_progress
	cmp	rax, EAGAIN
	je	.fsrs_would_block
	cmp	rax, EINTR
	je	.fsrs_would_block
	jmp	.fsrs_err
.fsrs_progress:
	mov	rax, [sendfile_len]
	test	rax, rax
	jz	.fsrs_ok
	add	r14, rax
	sub	r15, rax
	jmp	.fsrs_loop
.fsrs_would_block:
	mov	rax, [sendfile_len]
	test	rax, rax
	jz	.fsrs_wait
	add	r14, rax
	sub	r15, rax
	test	r15, r15
	jz	.fsrs_ok
.fsrs_wait:
	mov	rdi, r13
	call	coro_wait_write
	cmp	rax, 0
	jl	.fsrs_err
	jmp	.fsrs_loop
.fsrs_ok:
	xor	rax, rax
	jmp	.fsrs_out
.fsrs_err:
	mov	rax, -1
.fsrs_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

macro sendfile_bss
{
	sendfile_len dq ?
}

