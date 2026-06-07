; TCP socket helpers (macOS x86_64 v0).
; Requires platform.inc socket syscalls and TARGET_OS=macos.

; rdi = fd
sock_close:
	syscall1 SYS_close, rdi
	ret

; rdi = fd, rsi = buf, rdx = max bytes
; rax = bytes read, 0 on EOF, -1 on error
sock_read:
	push	rbx
	mov	rax, SYS_read
	syscall
	jc	.sr_err
	pop	rbx
	ret
.sr_err:
	mov	rax, -1
	pop	rbx
	ret

; rdi = fd, rsi = buf, rdx = len
; rax = bytes written, -1 on error
sock_write:
	push	rbx
	mov	rax, SYS_write
	syscall
	jc	.sw_err
	pop	rbx
	ret
.sw_err:
	mov	rax, -1
	pop	rbx
	ret

; rdi = fd, rsi = buf, rdx = total len
; rax = 0 on success, -1 on error
sock_write_all:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rbx, rdx
.swa_loop:
	test	rbx, rbx
	jz	.swa_ok
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, rbx
	call	sock_write
	cmp	rax, 0
	jle	.swa_err
	add	r13, rax
	sub	rbx, rax
	jmp	.swa_loop
.swa_ok:
	xor	rax, rax
	pop	r13
	pop	r12
	pop	rbx
	ret
.swa_err:
	mov	rax, -1
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = port (host order)
; rax = listen fd or -1
tcp_listen:
	xor	rsi, rsi
	jmp	tcp_listen_addr

; rdi = port (host order), rsi = IPv4 address in network byte order, or 0
; rax = listen fd or -1
tcp_listen_addr:
	push	rbp
	mov	rbp, rsp
	push	rbx
	push	r12
	push	r13
	sub	rsp, 20
	mov	r12d, edi
	mov	r13d, esi

	syscall3 SYS_socket, AF_INET, SOCK_STREAM, 0
	jc	.tl_fail
	mov	rbx, rax

	mov	dword [rsp + 16], 1
	mov	rdi, rbx
	mov	rsi, SOL_SOCKET
	mov	rdx, SO_REUSEADDR
	lea	r10, [rsp + 16]
	mov	r8, 4
	mov	rax, SYS_setsockopt
	syscall
	jc	.tl_close_sock

	mov	byte [rsp], SOCKADDR_IN_SIZE
	mov	byte [rsp + 1], AF_INET
	mov	ax, r12w
	xchg	ah, al
	mov	word [rsp + 2], ax
	mov	dword [rsp + 4], r13d
	xor	rax, rax
	mov	[rsp + 8], rax

	mov	rdi, rbx
	mov	rsi, rsp
	mov	rdx, SOCKADDR_IN_SIZE
	mov	rax, SYS_bind
	syscall
	jc	.tl_close_sock

	mov	rdi, rbx
	mov	rsi, 1
	mov	rax, SYS_listen
	syscall
	jc	.tl_close_sock

	mov	rax, rbx
	jmp	.tl_out

.tl_close_sock:
	mov	rdi, rbx
	call	sock_close
.tl_fail:
	mov	rax, -1
.tl_out:
	add	rsp, 20
	pop	r13
	pop	r12
	pop	rbx
	pop	rbp
	ret

; rdi = listen fd
; rax = client fd or -1
tcp_accept:
	push	rbx
	mov	rax, SYS_accept
	xor	rsi, rsi
	xor	rdx, rdx
	syscall
	jc	.ta_err
	pop	rbx
	ret
.ta_err:
	mov	rax, -1
	pop	rbx
	ret
