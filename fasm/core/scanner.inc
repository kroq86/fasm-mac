; Chunked line scanner.

SCANNER_OK equ 0
SCANNER_ERR_OPEN equ 1
SCANNER_ERR_READ equ 2

; rdi = path, rsi = read buffer, rdx = read buffer size,
; rcx = line buffer, r8 = line buffer size, r9 = callback
; Callback: rdi=line ptr, rsi=line len, rdx=line number. Return ignored.
; rax = SCANNER_OK / SCANNER_ERR_*
scanner_scan_file:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	[scanner_read_buf], rsi
	mov	[scanner_read_cap], rdx
	mov	[scanner_line_buf], rcx
	mov	[scanner_line_cap], r8
	mov	[scanner_callback], r9
	call	file_open_read
	cmp	rax, -1
	je	.ssf_open_error
	mov	rbx, rax
	mov	qword [scanner_line_no], 1
	mov	qword [scanner_line_len], 0
	mov	qword [scanner_overflow], 0
.ssf_read_loop:
	mov	rdi, rbx
	mov	rsi, [scanner_read_buf]
	mov	rdx, [scanner_read_cap]
	call	file_read_chunk
	cmp	rax, -1
	je	.ssf_read_error
	test	rax, rax
	jz	.ssf_eof
	mov	r12, rax
	xor	r13, r13
.ssf_byte_loop:
	cmp	r13, r12
	jae	.ssf_read_loop
	mov	r14, [scanner_read_buf]
	mov	al, [r14 + r13]
	cmp	al, 10
	je	.ssf_newline
	cmp	qword [scanner_overflow], 0
	jne	.ssf_next_byte
	mov	rcx, [scanner_line_len]
	cmp	rcx, [scanner_line_cap]
	jae	.ssf_mark_overflow
	mov	r14, [scanner_line_buf]
	mov	[r14 + rcx], al
	inc	qword [scanner_line_len]
	jmp	.ssf_next_byte
.ssf_mark_overflow:
	mov	qword [scanner_overflow], 1
	jmp	.ssf_next_byte
.ssf_newline:
	call	scanner_emit_line
	inc	qword [scanner_line_no]
	mov	qword [scanner_line_len], 0
	mov	qword [scanner_overflow], 0
.ssf_next_byte:
	inc	r13
	jmp	.ssf_byte_loop
.ssf_eof:
	cmp	qword [scanner_line_len], 0
	jne	.ssf_tail
	cmp	qword [scanner_overflow], 0
	jne	.ssf_tail
	jmp	.ssf_ok
.ssf_tail:
	call	scanner_emit_line
.ssf_ok:
	mov	rdi, rbx
	call	file_close
	mov	rax, SCANNER_OK
	jmp	.ssf_done
.ssf_read_error:
	mov	rdi, rbx
	call	file_close
	mov	rax, SCANNER_ERR_READ
	jmp	.ssf_done
.ssf_open_error:
	mov	rax, SCANNER_ERR_OPEN
.ssf_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

scanner_emit_line:
	cmp	qword [scanner_overflow], 0
	jne	.sel_done
	mov	rdi, [scanner_line_buf]
	mov	rsi, [scanner_line_len]
	mov	rdx, [scanner_line_no]
	call	[scanner_callback]
.sel_done:
	ret

macro scanner_bss
{
	scanner_read_buf dq ?
	scanner_read_cap dq ?
	scanner_line_buf dq ?
	scanner_line_cap dq ?
	scanner_callback dq ?
	scanner_line_no dq ?
	scanner_line_len dq ?
	scanner_overflow dq ?
}
