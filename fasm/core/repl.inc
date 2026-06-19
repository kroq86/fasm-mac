; REPL line input, tokenization, and Redis-style replies.

REPL_LINE_MAX equ 256
REPL_MAX_TOKENS equ 16
REPL_EOF equ -1

; contract: rdi = buffer, rsi = max bytes (including null room)
; returns: rax = line length, REPL_EOF on EOF, 0 on empty line
; clobbers: rax, rcx, rdx, rsi, rdi, r11
repl_read_line:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	dec	r13
	xor	rbx, rbx
.rrl_loop:
	cmp	rbx, r13
	jae	.rrl_done
	lea	rsi, [r12 + rbx]
	mov	rdi, STDIN
	mov	rdx, 1
	mov	rax, SYS_read
	syscall
	test	rax, rax
	jle	.rrl_eof
	movzx	eax, byte [r12 + rbx]
	cmp	al, 10
	je	.rrl_done
	cmp	al, 13
	je	.rrl_loop
	mov	[r12 + rbx], al
	inc	rbx
	jmp	.rrl_loop
.rrl_eof:
	cmp	rbx, 0
	jne	.rrl_done
	mov	rax, REPL_EOF
	jmp	.rrl_out
.rrl_done:
	mov	byte [r12 + rbx], 0
	mov	rax, rbx
.rrl_out:
	pop	r13
	pop	r12
	pop	rbx
	ret

; contract: rdi = line, rsi = token pointer array, rdx = max tokens
; returns: rax = token count
; clobbers: rax, rcx
repl_tokenize:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	xor	rbx, rbx
.rt_skip:
	movzx	eax, byte [r12]
	cmp	al, ' '
	je	.rt_adv
	cmp	al, 9
	je	.rt_adv
	cmp	al, 0
	je	.rt_done
	cmp	rbx, r14
	jae	.rt_done
	mov	[r13 + rbx * 8], r12
	inc	rbx
.rt_word:
	movzx	eax, byte [r12]
	cmp	al, ' '
	je	.rt_term
	cmp	al, 9
	je	.rt_term
	cmp	al, 0
	je	.rt_done
	inc	r12
	jmp	.rt_word
.rt_term:
	mov	byte [r12], 0
	inc	r12
	jmp	.rt_skip
.rt_adv:
	inc	r12
	jmp	.rt_skip
.rt_done:
	mov	rax, rbx
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; contract: rdi = buffer, rsi = max bytes, rdx = fd
; returns: rax = line length, REPL_EOF on EOF
; clobbers: rax, rcx, rdx, rsi, rdi, r11
repl_read_line_fd:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	dec	r13
	xor	rbx, rbx
.rrlf_loop:
	cmp	rbx, r13
	jae	.rrlf_done
	lea	rsi, [r12 + rbx]
	mov	rdi, r14
	mov	rdx, 1
	mov	rax, SYS_read
	syscall
	test	rax, rax
	jle	.rrlf_eof
	movzx	eax, byte [r12 + rbx]
	cmp	al, 10
	je	.rrlf_done
	cmp	al, 13
	je	.rrlf_loop
	mov	[r12 + rbx], al
	inc	rbx
	jmp	.rrlf_loop
.rrlf_eof:
	cmp	rbx, 0
	jne	.rrlf_done
	mov	rax, REPL_EOF
	jmp	.rrlf_out
.rrlf_done:
	mov	byte [r12 + rbx], 0
	mov	rax, rbx
.rrlf_out:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; contract: none
; returns: writes "OK\n" to stdout
; clobbers: delegated to repl_write_cstr
repl_write_ok:
	lea	rdi, [msg_ok]
	jmp	repl_write_cstr

; contract: none
; returns: writes "PONG\n" to stdout
; clobbers: delegated to repl_write_cstr
repl_write_pong:
	lea	rdi, [msg_pong]
	jmp	repl_write_cstr

; contract: none
; returns: writes "(nil)\n" to stdout
; clobbers: delegated to repl_write_cstr
repl_write_nil:
	lea	rdi, [msg_nil]
	jmp	repl_write_cstr

; contract: none
; returns: writes "ERR syntax\n" to stdout
; clobbers: delegated to repl_write_cstr
repl_write_err:
	lea	rdi, [msg_err]
	jmp	repl_write_cstr

; contract: none
; returns: writes "ERR wrongtype\n" to stdout
; clobbers: delegated to repl_write_cstr
repl_write_wrongtype:
	lea	rdi, [msg_wrongtype]
	jmp	repl_write_cstr

; contract: rdi = null-terminated string
; returns: writes string to stdout
; clobbers: delegated to str_len and io_write
repl_write_cstr:
	call	str_len
	mov	rsi, rax
	jmp	io_write

; contract: none
; returns: writes prompt to stdout
; clobbers: delegated to repl_write_cstr
repl_write_prompt:
	lea	rdi, [msg_prompt]
	jmp	repl_write_cstr

msg_ok db 'OK', 10, 0
msg_pong db 'PONG', 10, 0
msg_nil db '(nil)', 10, 0
msg_err db 'ERR syntax', 10, 0
msg_wrongtype db 'ERR wrongtype', 10, 0
msg_prompt db 'miniredis> ', 0
