; Smoke: TCP echo on port 9999.
; Run: arch -x86_64 ./fasm/tests/macos-smoke/tcp_echo &
;      printf 'hi' | nc localhost 9999

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/socket.inc"

ECHO_PORT equ 9999
ECHO_BUF_SIZE equ 256

entry start

start:
	mov	edi, ECHO_PORT
	call	tcp_listen
	cmp	rax, 0
	jl	start_fail
	mov	[listen_fd], rax

.accept_loop:
	mov	rdi, [listen_fd]
	call	tcp_accept
	cmp	rax, 0
	jl	.accept_loop
	mov	[client_fd], rax

.echo_loop:
	mov	rdi, [client_fd]
	lea	rsi, [echo_buf]
	mov	rdx, ECHO_BUF_SIZE
	call	sock_read
	cmp	rax, 0
	jle	.echo_done
	mov	rbx, rax
	mov	rdi, [client_fd]
	lea	rsi, [echo_buf]
	mov	rdx, rbx
	call	sock_write_all
	jmp	.echo_loop

.echo_done:
	mov	rdi, [client_fd]
	call	sock_close
	jmp	.accept_loop

start_fail:
	exit EXIT_FAILURE

segment readable writeable

listen_fd dq ?
client_fd dq ?
echo_buf rb ECHO_BUF_SIZE
