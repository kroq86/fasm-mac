; Tiny HTTP/1.x request-line parser.
;
; Caller owns buffers. This parser only reads the first request line.

HTTP_PARSE_OK equ 0
HTTP_PARSE_INCOMPLETE equ 1
HTTP_PARSE_BAD equ 2
HTTP_PARSE_UNSUPPORTED equ 3

HTTP_METHOD_GET equ 1
HTTP_METHOD_HEAD equ 2

HTTP_REQ_METHOD_OFF equ 0
HTTP_REQ_PATH_PTR_OFF equ 8
HTTP_REQ_PATH_LEN_OFF equ 16
HTTP_REQ_LINE_LEN_OFF equ 24
HTTP_REQ_SIZE equ 32

; rdi = request bytes, rsi = len, rdx = HTTP_REQ_SIZE storage
; rax = HTTP_PARSE_*
http_parse_request_line:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	xor	rbx, rbx
.hpr_find_eol:
	mov	rax, rbx
	inc	rax
	cmp	rax, r13
	jae	.hpr_incomplete
	cmp	byte [r12 + rbx], 13
	jne	.hpr_next_eol
	cmp	byte [r12 + rbx + 1], 10
	je	.hpr_have_line
.hpr_next_eol:
	inc	rbx
	jmp	.hpr_find_eol
.hpr_have_line:
	mov	rax, rbx
	add	rax, 2
	mov	[r14 + HTTP_REQ_LINE_LEN_OFF], rax
	cmp	rbx, 14
	jb	.hpr_bad
	cmp	byte [r12 + 0], 'G'
	jne	.hpr_try_head
	cmp	byte [r12 + 1], 'E'
	jne	.hpr_bad_method3
	cmp	byte [r12 + 2], 'T'
	jne	.hpr_bad_method3
	cmp	byte [r12 + 3], ' '
	jne	.hpr_bad_method3
	mov	qword [r14 + HTTP_REQ_METHOD_OFF], HTTP_METHOD_GET
	mov	r15, 4
	jmp	.hpr_path
.hpr_try_head:
	cmp	rbx, 15
	jb	.hpr_bad_method3
	cmp	byte [r12 + 0], 'H'
	jne	.hpr_bad_method3
	cmp	byte [r12 + 1], 'E'
	jne	.hpr_bad_method3
	cmp	byte [r12 + 2], 'A'
	jne	.hpr_bad_method3
	cmp	byte [r12 + 3], 'D'
	jne	.hpr_bad_method3
	cmp	byte [r12 + 4], ' '
	jne	.hpr_bad_method3
	mov	qword [r14 + HTTP_REQ_METHOD_OFF], HTTP_METHOD_HEAD
	mov	r15, 5
	jmp	.hpr_path
.hpr_bad_method3:
	xor	r15, r15
.hpr_method_scan:
	cmp	r15, rbx
	jae	.hpr_bad
	cmp	byte [r12 + r15], ' '
	je	.hpr_unsupported
	inc	r15
	jmp	.hpr_method_scan
.hpr_path:
	cmp	r15, rbx
	jae	.hpr_bad
	cmp	byte [r12 + r15], '/'
	jne	.hpr_bad
	lea	rax, [r12 + r15]
	mov	[r14 + HTTP_REQ_PATH_PTR_OFF], rax
	xor	rcx, rcx
.hpr_path_loop:
	mov	rax, r15
	add	rax, rcx
	cmp	rax, rbx
	jae	.hpr_bad
	lea	rax, [r12 + r15]
	mov	al, [rax + rcx]
	cmp	al, ' '
	je	.hpr_path_done
	cmp	al, '?'
	je	.hpr_query
	inc	rcx
	jmp	.hpr_path_loop
.hpr_query:
	mov	[r14 + HTTP_REQ_PATH_LEN_OFF], rcx
	add	r15, rcx
.hpr_query_loop:
	cmp	r15, rbx
	jae	.hpr_bad
	cmp	byte [r12 + r15], ' '
	je	.hpr_version
	inc	r15
	jmp	.hpr_query_loop
.hpr_path_done:
	mov	[r14 + HTTP_REQ_PATH_LEN_OFF], rcx
	add	r15, rcx
.hpr_version:
	cmp	qword [r14 + HTTP_REQ_PATH_LEN_OFF], 0
	je	.hpr_bad
	inc	r15
	mov	rax, r15
	add	rax, 8
	cmp	rax, rbx
	jne	.hpr_bad
	cmp	byte [r12 + r15 + 0], 'H'
	jne	.hpr_bad
	cmp	byte [r12 + r15 + 1], 'T'
	jne	.hpr_bad
	cmp	byte [r12 + r15 + 2], 'T'
	jne	.hpr_bad
	cmp	byte [r12 + r15 + 3], 'P'
	jne	.hpr_bad
	cmp	byte [r12 + r15 + 4], '/'
	jne	.hpr_bad
	cmp	byte [r12 + r15 + 5], '1'
	jne	.hpr_bad
	cmp	byte [r12 + r15 + 6], '.'
	jne	.hpr_bad
	mov	al, [r12 + r15 + 7]
	cmp	al, '0'
	je	.hpr_ok
	cmp	al, '1'
	je	.hpr_ok
	jmp	.hpr_bad
.hpr_ok:
	mov	rax, HTTP_PARSE_OK
	jmp	.hpr_out
.hpr_incomplete:
	mov	rax, HTTP_PARSE_INCOMPLETE
	jmp	.hpr_out
.hpr_bad:
	mov	rax, HTTP_PARSE_BAD
	jmp	.hpr_out
.hpr_unsupported:
	mov	rax, HTTP_PARSE_UNSUPPORTED
.hpr_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret
