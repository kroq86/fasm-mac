; Minimal RESP encoder/decoder for Redis-style command arrays.
; Requires socket.inc, str.inc, print_io.inc (print_int64_to_buf), runtime_print_bss.
;
; App BSS must provide:
;   resp_buf rb RESP_BUF_MAX
;   resp_buf_len dq ?
;   resp_argv rq RESP_MAX_ARGS

RESP_BUF_MAX equ 4096
RESP_MAX_ARGS equ 16

; rdi = fd, rsi = payload (no prefix), null-terminated
resp_write_simple:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, r12
	lea	rsi, [resp_plus]
	mov	rdx, 1
	call	sock_write_all
	mov	rdi, r13
	call	str_len
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, rax
	call	sock_write_all
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	sock_write_all
	xor	rax, rax
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = fd, rsi = message (without -ERR prefix)
resp_write_error:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, r12
	lea	rsi, [resp_err_prefix]
	mov	rdx, resp_err_prefix_len
	call	sock_write_all
	mov	rdi, r13
	call	str_len
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, rax
	call	sock_write_all
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	sock_write_all
	xor	rax, rax
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = fd, rax = int64
resp_write_int:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rax
	mov	rdi, r12
	mov	al, ':'
	call	resp_write_byte
	mov	rax, r13
	lea	rdi, [int_digits + 20]
	call	print_int64_to_buf
	mov	rsi, rdi
	mov	rdx, rax
	mov	rdi, r12
	call	sock_write_all
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	sock_write_all
	xor	rax, rax
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = fd
resp_write_null:
	push	r12
	mov	r12, rdi
	mov	rdi, r12
	lea	rsi, [resp_null]
	mov	rdx, resp_null_len
	call	sock_write_all
	xor	rax, rax
	pop	r12
	ret

; rdi = fd, rsi = payload ptr, rdx = len
resp_write_bulk:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	rdi, r12
	mov	al, '$'
	call	resp_write_byte
	mov	rax, r14
	push	rcx
	lea	rdi, [int_digits + 20]
	call	print_int64_to_buf
	mov	rsi, rdi
	mov	rdx, rax
	mov	rdi, r12
	call	sock_write_all
	pop	rcx
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	sock_write_all
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r14
	call	sock_write_all
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	sock_write_all
	xor	rax, rax
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

resp_write_byte:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	[char_buf], al
	mov	rdi, r12
	lea	rsi, [char_buf]
	mov	rdx, 1
	call	sock_write_all
	pop	r12
	pop	rbx
	ret

; rdi = fd
; rax = argc (>0), 0 = EOF, -1 = protocol/read error
resp_read_command:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	qword [resp_buf_len], 0
.rrc_loop:
	call	resp_try_parse
	test	rax, rax
	jg	.rrc_done
	cmp	rax, -1
	je	.rrc_err
	mov	rdi, r12
	lea	rsi, [resp_buf]
	mov	rax, [resp_buf_len]
	add	rsi, rax
	mov	rdx, RESP_BUF_MAX
	sub	rdx, rax
	cmp	rdx, 0
	je	.rrc_err
	call	sock_read
	cmp	rax, 0
	jle	.rrc_eof
	add	[resp_buf_len], rax
	jmp	.rrc_loop
.rrc_eof:
	cmp	qword [resp_buf_len], 0
	je	.rrc_zero
.rrc_err:
	mov	rax, -1
	jmp	.rrc_out
.rrc_zero:
	xor	rax, rax
	jmp	.rrc_out
.rrc_done:
	mov	qword [resp_buf_len], 0
.rrc_out:
	pop	r12
	pop	rbx
	ret

; rax = argc, 0 = incomplete, -1 = bad protocol
resp_try_parse:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, [resp_buf_len]
	xor	r13, r13
	cmp	r12, 2
	jb	.rtp_incomplete
	cmp	byte [resp_buf], '*'
	jne	.rtp_bad
	mov	r13, 1
	call	.rtp_parse_int
	jc	.rtp_bad
	mov	r15, rax
	test	r15, r15
	js	.rtp_bad
	cmp	r15, RESP_MAX_ARGS
	ja	.rtp_bad
	xor	r14, r14
.rtp_arg_loop:
	cmp	r14, r15
	jae	.rtp_ok
	cmp	r13, r12
	jae	.rtp_incomplete
	cmp	byte [resp_buf + r13], '$'
	jne	.rtp_bad
	inc	r13
	call	.rtp_parse_int
	jc	.rtp_bad
	mov	rbx, rax
	test	rbx, rbx
	js	.rtp_bad
	mov	r8, r13
	mov	rax, r13
	add	rax, rbx
	add	rax, 2
	cmp	rax, r12
	ja	.rtp_incomplete
	lea	rax, [resp_buf + r8]
	mov	[resp_argv + r14 * 8], rax
	add	r13, rbx
	cmp	r13, r12
	jae	.rtp_incomplete
	cmp	byte [resp_buf + r13], 13
	jne	.rtp_bad
	cmp	byte [resp_buf + r13 + 1], 10
	jne	.rtp_bad
	mov	byte [resp_buf + r13], 0
	add	r13, 2
	inc	r14
	jmp	.rtp_arg_loop
.rtp_ok:
	mov	rax, r15
	jmp	.rtp_out
.rtp_incomplete:
	xor	rax, rax
	jmp	.rtp_out
.rtp_bad:
	mov	rax, -1
.rtp_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; r13 = index; rax = int; jc on bad; r13 advanced past \n
.rtp_parse_int:
	push	rbx
	push	rcx
	xor	rax, rax
.rtp_pi_loop:
	cmp	r13, r12
	jae	.rtp_pi_bad
	movzx	ecx, byte [resp_buf + r13]
	cmp	cl, 13
	je	.rtp_pi_cr
	cmp	cl, '0'
	jb	.rtp_pi_bad
	cmp	cl, '9'
	ja	.rtp_pi_bad
	imul	rax, 10
	sub	cl, '0'
	add	rax, rcx
	inc	r13
	jmp	.rtp_pi_loop
.rtp_pi_cr:
	cmp	r13, r12
	jae	.rtp_pi_bad
	cmp	byte [resp_buf + r13 + 1], 10
	jne	.rtp_pi_bad
	add	r13, 2
	clc
	pop	rcx
	pop	rbx
	ret
.rtp_pi_bad:
	stc
	pop	rcx
	pop	rbx
	ret

resp_plus db '+', 0
resp_crlf db 13, 10
resp_err_prefix db '-ERR ', 0
resp_err_prefix_len = $ - resp_err_prefix - 1
resp_null db '$-1', 13, 10
resp_null_len = $ - resp_null
