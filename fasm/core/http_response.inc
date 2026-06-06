; Small HTTP response helpers.

HTTP_STATUS_200 equ 200
HTTP_STATUS_400 equ 400
HTTP_STATUS_403 equ 403
HTTP_STATUS_404 equ 404
HTTP_STATUS_405 equ 405
HTTP_STATUS_414 equ 414
HTTP_STATUS_500 equ 500

HTTP_RESP_OK equ 0
HTTP_RESP_TOO_SMALL equ -1

macro http_response_bss
{
	http_resp_len dq ?
	http_num_buf rb 32
}

; rdi = extension/name ptr, rsi = len
; rax = content-type cstr
http_content_type:
	test	rsi, rsi
	jz	.hct_bin
	push	rbx
	push	r12
	mov	r12, rdi
	mov	rbx, rsi
.hct_scan:
	test	rbx, rbx
	jz	.hct_bin_pop
	dec	rbx
	cmp	byte [r12 + rbx], '.'
	je	.hct_ext
	jmp	.hct_scan
.hct_ext:
	lea	rdi, [r12 + rbx]
	mov	rsi, r12
	add	rsi, rsi
	cmp	byte [rdi + 1], 'h'
	je	.hct_html
	cmp	byte [rdi + 1], 'c'
	je	.hct_css
	cmp	byte [rdi + 1], 'j'
	je	.hct_j
	cmp	byte [rdi + 1], 't'
	je	.hct_txt
	cmp	byte [rdi + 1], 'p'
	je	.hct_png
	cmp	byte [rdi + 1], 'J'
	je	.hct_jpg
	jmp	.hct_bin_pop
.hct_html:
	lea	rax, [ct_html]
	jmp	.hct_out
.hct_css:
	lea	rax, [ct_css]
	jmp	.hct_out
.hct_j:
	cmp	byte [rdi + 2], 's'
	jne	.hct_json
	cmp	byte [rdi + 3], 'o'
	je	.hct_json
	lea	rax, [ct_js]
	jmp	.hct_out
.hct_json:
	lea	rax, [ct_json]
	jmp	.hct_out
.hct_txt:
	lea	rax, [ct_text]
	jmp	.hct_out
.hct_png:
	lea	rax, [ct_png]
	jmp	.hct_out
.hct_jpg:
	lea	rax, [ct_jpeg]
	jmp	.hct_out
.hct_bin_pop:
	lea	rax, [ct_bin]
.hct_out:
	pop	r12
	pop	rbx
	ret
.hct_bin:
	lea	rax, [ct_bin]
	ret

; rdi = out, rsi = cap, rdx = status, rcx = content length, r8 = content-type cstr
; rax = bytes written, or HTTP_RESP_TOO_SMALL
http_build_header:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rcx
	mov	r15, r8
	mov	qword [http_resp_len], 0
	cmp	rdx, HTTP_STATUS_200
	je	.hbh_200
	cmp	rdx, HTTP_STATUS_400
	je	.hbh_400
	cmp	rdx, HTTP_STATUS_403
	je	.hbh_403
	cmp	rdx, HTTP_STATUS_404
	je	.hbh_404
	cmp	rdx, HTTP_STATUS_405
	je	.hbh_405
	cmp	rdx, HTTP_STATUS_414
	je	.hbh_414
	lea	rdi, [status_500]
	jmp	.hbh_status
.hbh_200:
	lea	rdi, [status_200]
	jmp	.hbh_status
.hbh_400:
	lea	rdi, [status_400]
	jmp	.hbh_status
.hbh_403:
	lea	rdi, [status_403]
	jmp	.hbh_status
.hbh_404:
	lea	rdi, [status_404]
	jmp	.hbh_status
.hbh_405:
	lea	rdi, [status_405]
	jmp	.hbh_status
.hbh_414:
	lea	rdi, [status_414]
.hbh_status:
	call	.hbh_append_cstr
	lea	rdi, [hdr_server]
	call	.hbh_append_cstr
	lea	rdi, [hdr_content_length]
	call	.hbh_append_cstr
	mov	rax, r14
	call	.hbh_append_uint
	lea	rdi, [crlf]
	call	.hbh_append_cstr
	lea	rdi, [hdr_content_type]
	call	.hbh_append_cstr
	mov	rdi, r15
	call	.hbh_append_cstr
	lea	rdi, [hdr_close]
	call	.hbh_append_cstr
	lea	rdi, [crlf]
	call	.hbh_append_cstr
	mov	rax, [http_resp_len]
	jmp	.hbh_out
.hbh_fail:
	mov	rax, HTTP_RESP_TOO_SMALL
	jmp	.hbh_out
.hbh_append_cstr:
	push	rdi
	call	str_len
	mov	rsi, rax
	pop	rdi
	jmp	.hbh_append
.hbh_append:
	mov	rax, [http_resp_len]
	add	rax, rsi
	cmp	rax, r13
	ja	.hbh_fail
	mov	rbx, r12
	add	rbx, [http_resp_len]
	xor	rcx, rcx
.hbh_copy:
	cmp	rcx, rsi
	jae	.hbh_commit
	mov	al, [rdi + rcx]
	mov	[rbx + rcx], al
	inc	rcx
	jmp	.hbh_copy
.hbh_commit:
	add	[http_resp_len], rsi
	ret
.hbh_append_uint:
	push	rbx
	push	r12
	push	r13
	mov	rbx, 10
	lea	r12, [http_num_buf + 31]
	mov	byte [r12], 0
	test	rax, rax
	jnz	.hbh_uint_loop
	dec	r12
	mov	byte [r12], '0'
	jmp	.hbh_uint_emit
.hbh_uint_loop:
	xor	rdx, rdx
	div	rbx
	add	dl, '0'
	dec	r12
	mov	[r12], dl
	test	rax, rax
	jnz	.hbh_uint_loop
.hbh_uint_emit:
	lea	r13, [http_num_buf + 31]
	sub	r13, r12
	mov	rdi, r12
	mov	rsi, r13
	call	.hbh_append
	pop	r13
	pop	r12
	pop	rbx
	ret
.hbh_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

status_200 db 'HTTP/1.1 200 OK', 13, 10, 0
status_400 db 'HTTP/1.1 400 Bad Request', 13, 10, 0
status_403 db 'HTTP/1.1 403 Forbidden', 13, 10, 0
status_404 db 'HTTP/1.1 404 Not Found', 13, 10, 0
status_405 db 'HTTP/1.1 405 Method Not Allowed', 13, 10, 0
status_414 db 'HTTP/1.1 414 URI Too Long', 13, 10, 0
status_500 db 'HTTP/1.1 500 Internal Server Error', 13, 10, 0
hdr_server db 'Server: httpmini', 13, 10, 0
hdr_content_length db 'Content-Length: ', 0
hdr_content_type db 'Content-Type: ', 0
hdr_close db 13, 10, 'Connection: close', 13, 10, 0
crlf db 13, 10, 0

ct_html db 'text/html; charset=utf-8', 0
ct_css db 'text/css; charset=utf-8', 0
ct_js db 'application/javascript; charset=utf-8', 0
ct_json db 'application/json; charset=utf-8', 0
ct_text db 'text/plain; charset=utf-8', 0
ct_png db 'image/png', 0
ct_jpeg db 'image/jpeg', 0
ct_bin db 'application/octet-stream', 0
