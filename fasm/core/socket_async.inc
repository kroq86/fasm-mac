; Nonblocking socket helpers for coro.inc.
; Requires platform.inc, socket.inc, and coro.inc.

; rdi = fd; rax = 0 on success, -1 on error
sock_set_nonblock:
	push	rbx
	mov	rbx, rdi
	mov	rax, SYS_fcntl
	mov	rsi, F_GETFL
	xor	rdx, rdx
	syscall
	jc	.ssn_err
	or	rax, O_NONBLOCK
	mov	rdx, rax
	mov	rax, SYS_fcntl
	mov	rdi, rbx
	mov	rsi, F_SETFL
	syscall
	jc	.ssn_err
	xor	rax, rax
	jmp	.ssn_out
.ssn_err:
	mov	rax, -1
.ssn_out:
	pop	rbx
	ret

; rdi = listen fd; rax = client fd, or -1
coro_accept:
	push	rbx
	mov	rbx, rdi
.ca_loop:
	mov	rdi, rbx
	mov	rax, SYS_accept
	xor	rsi, rsi
	xor	rdx, rdx
	syscall
	jnc	.ca_ok
	cmp	rax, EAGAIN
	jne	.ca_err
	mov	rdi, rbx
	call	coro_wait_read
	cmp	rax, 0
	jl	.ca_err
	jmp	.ca_loop
.ca_ok:
	mov	rbx, rax
	mov	rdi, rax
	call	sock_set_nonblock
	mov	rax, rbx
	jmp	.ca_out
.ca_err:
	mov	rax, -1
.ca_out:
	pop	rbx
	ret

; rdi = fd, rsi = buf, rdx = max bytes
; rax = bytes read, 0 on EOF, -1 on error
coro_read:
	push	rbx
	push	r12
	push	r13
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
.crd_loop:
	mov	rdi, rbx
	mov	rsi, r12
	mov	rdx, r13
	mov	rax, SYS_read
	syscall
	jnc	.crd_out
	cmp	rax, EAGAIN
	jne	.crd_err
	mov	rdi, rbx
	call	coro_wait_read
	cmp	rax, 0
	jl	.crd_err
	jmp	.crd_loop
.crd_err:
	mov	rax, -1
.crd_out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = fd, rsi = buf, rdx = len
; rax = bytes written, -1 on error
coro_write:
	push	rbx
	push	r12
	push	r13
	mov	rbx, rdi
	mov	r12, rsi
	mov	r13, rdx
.cw_loop:
	mov	rdi, rbx
	mov	rsi, r12
	mov	rdx, r13
	mov	rax, SYS_write
	syscall
	jnc	.cw_out
	cmp	rax, EAGAIN
	jne	.cw_err
	mov	rdi, rbx
	call	coro_wait_write
	cmp	rax, 0
	jl	.cw_err
	jmp	.cw_loop
.cw_err:
	mov	rax, -1
.cw_out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = fd, rsi = buf, rdx = total len
; rax = 0 on success, -1 on error
coro_write_all:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rbx, rdx
.cwa_loop:
	test	rbx, rbx
	jz	.cwa_ok
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, rbx
	call	coro_write
	cmp	rax, 0
	jle	.cwa_err
	add	r13, rax
	sub	rbx, rax
	jmp	.cwa_loop
.cwa_ok:
	xor	rax, rax
	jmp	.cwa_out
.cwa_err:
	mov	rax, -1
.cwa_out:
	pop	r13
	pop	r12
	pop	rbx
	ret
