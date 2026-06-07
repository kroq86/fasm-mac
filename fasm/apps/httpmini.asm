; httpmini: single-threaded concurrent static HTTP server.
;
; Build:
;   fasm fasm/apps/httpmini.asm /tmp/httpmini
;   arch -x86_64 /tmp/httpmini --root . --port 8080

format ELF64 executable 3
include "fasm/core/platform.inc"

HTTPMINI_MAX_CLIENTS equ 64
CORO_MAX_TASKS equ 65
CORO_STACK_SIZE equ 16384

HTTP_REQ_MAX equ 4096
HTTP_REQ_META_SIZE equ 32
HTTP_HEADER_MAX equ 1024
HTTP_DEFAULT_PORT equ 8080

CONN_FD_OFF equ 0
CONN_REQ_LEN_OFF equ 8
CONN_REQ_META_OFF equ 16
CONN_STATUS_OFF equ CONN_REQ_META_OFF + HTTP_REQ_META_SIZE
CONN_HEADER_LEN_OFF equ CONN_STATUS_OFF + 8
CONN_REQ_BUF_OFF equ CONN_HEADER_LEN_OFF + 8
CONN_HEADER_BUF_OFF equ CONN_REQ_BUF_OFF + HTTP_REQ_MAX
CONN_SIZE equ CONN_HEADER_BUF_OFF + HTTP_HEADER_MAX

segment readable executable

include "fasm/core/ccall64.inc"

; Satisfy str.inc's optional str_write helper without pulling print_io BSS into
; this server.
print_cstr:
	ret

include "fasm/core/str.inc"
include "fasm/core/file.inc"
include "fasm/core/socket.inc"
include "fasm/core/coro.inc"
include "fasm/core/socket_async.inc"
include "fasm/core/sendfile.inc"
include "fasm/core/http.inc"
include "fasm/core/http_response.inc"
include "fasm/core/path_real.inc"

entry start

start:
	mov	rbx, rsp
	mov	rax, [rbx]
	mov	[argc], rax
	lea	rax, [rbx + 8]
	mov	[argv], rax
	lea	rax, [dot_path]
	mov	[root_arg], rax
	mov	qword [listen_port], HTTP_DEFAULT_PORT
	mov	qword [bind_addr], INADDR_ANY
	call	parse_args
	cmp	rax, 0
	jl	start_usage
	mov	rdi, [root_arg]
	lea	rsi, [stat_buf]
	mov	rax, SYS_stat64
	syscall
	jc	start_bad_root
	mov	eax, dword [stat_buf + STAT64_MODE_OFF]
	and	eax, S_IFMT
	cmp	eax, S_IFDIR
	jne	start_bad_root
	call	conn_pool_init
	call	coro_init
	cmp	rax, 0
	jl	start_fail
	lea	rdi, [accept_task]
	xor	rsi, rsi
	call	coro_spawn
	cmp	rax, 0
	jl	start_fail
	call	coro_run
	exit	EXIT_SUCCESS
start_usage:
	lea	rdi, [usage_msg]
	mov	rsi, usage_msg_len
	call	write_stderr
	exit	2
start_bad_root:
	lea	rdi, [bad_root_msg]
	mov	rsi, bad_root_msg_len
	call	write_stderr
	exit	2
start_fail:
	exit	EXIT_FAILURE

parse_args:
	push	rbx
	mov	rbx, 1
.pa_loop:
	cmp	rbx, [argc]
	jae	.pa_ok
	mov	rax, [argv]
	mov	rdi, [rax + rbx * 8]
	lea	rsi, [opt_root]
	call	str_eq
	cmp	rax, 1
	je	.pa_root
	mov	rax, [argv]
	mov	rdi, [rax + rbx * 8]
	lea	rsi, [opt_port]
	call	str_eq
	cmp	rax, 1
	je	.pa_port
	mov	rax, [argv]
	mov	rdi, [rax + rbx * 8]
	lea	rsi, [opt_bind]
	call	str_eq
	cmp	rax, 1
	je	.pa_bind
	mov	rax, -1
	jmp	.pa_out
.pa_root:
	inc	rbx
	cmp	rbx, [argc]
	jae	.pa_bad
	mov	rax, [argv]
	mov	rax, [rax + rbx * 8]
	mov	[root_arg], rax
	inc	rbx
	jmp	.pa_loop
.pa_port:
	inc	rbx
	cmp	rbx, [argc]
	jae	.pa_bad
	mov	rax, [argv]
	mov	rdi, [rax + rbx * 8]
	call	parse_uint
	cmp	rax, 1
	jb	.pa_bad
	cmp	rax, 65535
	ja	.pa_bad
	mov	[listen_port], rax
	inc	rbx
	jmp	.pa_loop
.pa_bind:
	inc	rbx
	cmp	rbx, [argc]
	jae	.pa_bad
	mov	rax, [argv]
	mov	rdi, [rax + rbx * 8]
	lea	rsi, [bind_loopback]
	call	str_eq
	cmp	rax, 1
	je	.pa_bind_loopback
	mov	rax, [argv]
	mov	rdi, [rax + rbx * 8]
	lea	rsi, [bind_any]
	call	str_eq
	cmp	rax, 1
	jne	.pa_bad
	mov	qword [bind_addr], INADDR_ANY
	inc	rbx
	jmp	.pa_loop
.pa_bind_loopback:
	mov	qword [bind_addr], 0100007Fh
	inc	rbx
	jmp	.pa_loop
.pa_ok:
	xor	rax, rax
	jmp	.pa_out
.pa_bad:
	mov	rax, -1
.pa_out:
	pop	rbx
	ret

; rdi = cstr; rax = uint, or 0 on error/zero
parse_uint:
	xor	rax, rax
	xor	rcx, rcx
.pu_loop:
	mov	cl, [rdi]
	test	cl, cl
	jz	.pu_done
	cmp	cl, '0'
	jb	.pu_bad
	cmp	cl, '9'
	ja	.pu_bad
	imul	rax, 10
	sub	cl, '0'
	add	rax, rcx
	inc	rdi
	jmp	.pu_loop
.pu_done:
	ret
.pu_bad:
	xor	rax, rax
	ret

accept_task:
	push	r12
	push	r13
	mov	rdi, [listen_port]
	mov	rsi, [bind_addr]
	call	tcp_listen_addr
	cmp	rax, 0
	jl	.at_done
	mov	[listen_fd], rax
	mov	rdi, rax
	call	sock_set_nonblock
.at_loop:
	mov	rdi, [listen_fd]
	call	coro_accept
	cmp	rax, 0
	jl	.at_loop
	mov	r12, rax
	call	conn_alloc
	test	rax, rax
	jz	.at_reject
	mov	rdi, rax
	mov	rsi, r12
	call	conn_init
	mov	r13, rax
	lea	rdi, [client_task]
	mov	rsi, r13
	call	coro_spawn
	cmp	rax, 0
	jl	.at_reject_conn
	call	coro_yield
	jmp	.at_loop
.at_reject_conn:
	mov	rdi, r12
	call	sock_close
	mov	rdi, r13
	call	conn_release
	jmp	.at_loop
.at_reject:
	mov	rdi, r12
	call	sock_close
	jmp	.at_loop
.at_done:
	pop	r13
	pop	r12
	ret

; rdi = conn*
client_task:
	push	r12
	mov	r12, rdi
	mov	qword [r12 + CONN_STATUS_OFF], 0
	mov	rdi, r12
	call	conn_read_request
	cmp	rax, HTTP_PARSE_OK
	je	.ct_ok
	cmp	rax, HTTP_PARSE_UNSUPPORTED
	je	.ct_405
	cmp	rax, HTTP_PARSE_INCOMPLETE
	je	.ct_414
	mov	rdi, r12
	mov	rsi, HTTP_STATUS_400
	call	send_error
	jmp	.ct_done
.ct_405:
	mov	rdi, r12
	mov	rsi, HTTP_STATUS_405
	call	send_error
	jmp	.ct_done
.ct_414:
	mov	rdi, r12
	mov	rsi, HTTP_STATUS_414
	call	send_error
	jmp	.ct_done
.ct_ok:
	mov	rdi, r12
	call	serve_request
.ct_done:
	mov	rdi, r12
	call	log_access
	mov	rdi, r12
	call	conn_close_release
	pop	r12
	ret

conn_pool_init:
	push	rbx
	xor	rbx, rbx
.cpi_loop:
	cmp	rbx, HTTPMINI_MAX_CLIENTS
	jae	.cpi_done
	mov	rax, rbx
	imul	rax, CONN_SIZE
	lea	rdx, [connections]
	mov	qword [rdx + rax + CONN_FD_OFF], -1
	inc	rbx
	jmp	.cpi_loop
.cpi_done:
	pop	rbx
	ret

conn_alloc:
	push	rbx
	xor	rbx, rbx
.ca_loop:
	cmp	rbx, HTTPMINI_MAX_CLIENTS
	jae	.ca_none
	mov	rax, rbx
	imul	rax, CONN_SIZE
	lea	rdx, [connections]
	cmp	qword [rdx + rax + CONN_FD_OFF], -1
	je	.ca_found
	inc	rbx
	jmp	.ca_loop
.ca_found:
	lea	rdx, [connections]
	add	rax, rdx
	jmp	.ca_out
.ca_none:
	xor	rax, rax
.ca_out:
	pop	rbx
	ret

conn_init:
	mov	[rdi + CONN_FD_OFF], rsi
	mov	qword [rdi + CONN_REQ_LEN_OFF], 0
	mov	qword [rdi + CONN_REQ_META_OFF + HTTP_REQ_METHOD_OFF], 0
	mov	qword [rdi + CONN_REQ_META_OFF + HTTP_REQ_PATH_PTR_OFF], 0
	mov	qword [rdi + CONN_REQ_META_OFF + HTTP_REQ_PATH_LEN_OFF], 0
	mov	qword [rdi + CONN_REQ_META_OFF + HTTP_REQ_LINE_LEN_OFF], 0
	mov	qword [rdi + CONN_HEADER_LEN_OFF], 0
	mov	qword [rdi + CONN_STATUS_OFF], 0
	mov	rax, rdi
	ret

conn_release:
	mov	qword [rdi + CONN_FD_OFF], -1
	mov	qword [rdi + CONN_REQ_LEN_OFF], 0
	mov	qword [rdi + CONN_REQ_META_OFF + HTTP_REQ_METHOD_OFF], 0
	mov	qword [rdi + CONN_REQ_META_OFF + HTTP_REQ_PATH_PTR_OFF], 0
	mov	qword [rdi + CONN_REQ_META_OFF + HTTP_REQ_PATH_LEN_OFF], 0
	mov	qword [rdi + CONN_REQ_META_OFF + HTTP_REQ_LINE_LEN_OFF], 0
	mov	qword [rdi + CONN_HEADER_LEN_OFF], 0
	mov	qword [rdi + CONN_STATUS_OFF], 0
	ret

conn_close_release:
	push	rdi
	mov	rdi, [rdi + CONN_FD_OFF]
	cmp	rdi, 0
	jl	.ccr_release
	call	sock_close
.ccr_release:
	pop	rdi
	jmp	conn_release

; rdi = conn*; rax = HTTP_PARSE_*
conn_read_request:
	push	rbx
	push	r12
	mov	r12, rdi
.crr_parse:
	lea	rdi, [r12 + CONN_REQ_BUF_OFF]
	mov	rsi, [r12 + CONN_REQ_LEN_OFF]
	lea	rdx, [r12 + CONN_REQ_META_OFF]
	call	http_parse_request_line
	cmp	rax, HTTP_PARSE_INCOMPLETE
	jne	.crr_out
	mov	rbx, [r12 + CONN_REQ_LEN_OFF]
	cmp	rbx, HTTP_REQ_MAX
	jae	.crr_too_long
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [r12 + CONN_REQ_BUF_OFF + rbx]
	mov	rdx, HTTP_REQ_MAX
	sub	rdx, rbx
	call	coro_read
	cmp	rax, 0
	jle	.crr_bad
	add	[r12 + CONN_REQ_LEN_OFF], rax
	jmp	.crr_parse
.crr_bad:
	mov	rax, HTTP_PARSE_BAD
	jmp	.crr_out
.crr_too_long:
	mov	rax, HTTP_PARSE_INCOMPLETE
.crr_out:
	pop	r12
	pop	rbx
	ret

; rdi = conn*
serve_request:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r14, [r12 + CONN_REQ_META_OFF + HTTP_REQ_PATH_PTR_OFF]
	mov	rbx, [r12 + CONN_REQ_META_OFF + HTTP_REQ_PATH_LEN_OFF]
	cmp	rbx, 1
	jne	.sr_have_path
	cmp	byte [r14], '/'
	jne	.sr_have_path
	lea	r14, [index_req_path]
	mov	rbx, index_req_path_len
.sr_have_path:
	mov	rdi, r14
	mov	rsi, rbx
	call	path_http_unsafe
	cmp	rax, 1
	je	.sr_403
	mov	rdi, [root_arg]
	mov	rsi, r14
	mov	rdx, rbx
	lea	rcx, [candidate_path]
	call	path_join_root_http
	test	rax, rax
	jz	.sr_414
	lea	rdi, [candidate_path]
	lea	rsi, [stat_buf]
	mov	rax, SYS_lstat64
	syscall
	jc	.sr_404
	mov	eax, dword [stat_buf + STAT64_MODE_OFF]
	and	eax, S_IFMT
	cmp	eax, S_IFLNK
	je	.sr_403
	cmp	eax, S_IFDIR
	je	.sr_403
	cmp	eax, S_IFREG
	jne	.sr_403
	lea	rdi, [candidate_path]
	call	file_open_read
	cmp	rax, 0
	jl	.sr_404
	mov	r13, rax
	mov	rdi, r14
	mov	rsi, rbx
	call	http_content_type
	mov	r8, rax
	mov	rdi, r12
	mov	rsi, HTTP_STATUS_200
	mov	rdx, qword [stat_buf + STAT64_SIZE_OFF]
	call	send_header2
	cmp	rax, 0
	jl	.sr_close_file
	cmp	qword [r12 + CONN_REQ_META_OFF + HTTP_REQ_METHOD_OFF], HTTP_METHOD_HEAD
	je	.sr_close_file
	mov	rdi, r13
	mov	rsi, [r12 + CONN_FD_OFF]
	xor	rdx, rdx
	mov	rcx, qword [stat_buf + STAT64_SIZE_OFF]
	call	file_send_range_socket
.sr_close_file:
	mov	rdi, r13
	call	file_close
	jmp	.sr_done
.sr_414:
	mov	rdi, r12
	mov	rsi, HTTP_STATUS_414
	call	send_error
	jmp	.sr_done
.sr_404:
	mov	rdi, r12
	mov	rsi, HTTP_STATUS_404
	call	send_error
	jmp	.sr_done
.sr_403:
	mov	rdi, r12
	mov	rsi, HTTP_STATUS_403
	call	send_error
	jmp	.sr_done
.sr_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = conn*, rsi = status
send_error:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	r13, rsi
	call	error_body_for_status
	mov	rbx, rax
	mov	r14, rdx
	mov	rdx, r14
	lea	r8, [ct_text]
	mov	rdi, r12
	mov	rsi, r13
	call	send_header2
	cmp	rax, 0
	jl	.se_out
	mov	rdi, [r12 + CONN_FD_OFF]
	mov	rsi, rbx
	mov	rdx, r14
	call	coro_write_all
.se_out:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = conn*, rsi = status, rdx = content len, r8 = content type
send_header2:
	push	r12
	mov	r12, rdi
	mov	[r12 + CONN_STATUS_OFF], rsi
	lea	rdi, [r12 + CONN_HEADER_BUF_OFF]
	mov	rcx, rdx
	mov	rdx, rsi
	mov	rsi, HTTP_HEADER_MAX
	call	http_build_header
	cmp	rax, 0
	jl	.sh2_done
	mov	[r12 + CONN_HEADER_LEN_OFF], rax
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [r12 + CONN_HEADER_BUF_OFF]
	mov	rdx, [r12 + CONN_HEADER_LEN_OFF]
	call	coro_write_all
.sh2_done:
	pop	r12
	ret

error_body_for_status:
	cmp	rsi, HTTP_STATUS_400
	je	.eb_400
	cmp	rsi, HTTP_STATUS_403
	je	.eb_403
	cmp	rsi, HTTP_STATUS_404
	je	.eb_404
	cmp	rsi, HTTP_STATUS_405
	je	.eb_405
	cmp	rsi, HTTP_STATUS_414
	je	.eb_414
	lea	rax, [body_500]
	mov	rdx, body_500_len
	ret
.eb_400:
	lea	rax, [body_400]
	mov	rdx, body_400_len
	ret
.eb_403:
	lea	rax, [body_403]
	mov	rdx, body_403_len
	ret
.eb_404:
	lea	rax, [body_404]
	mov	rdx, body_404_len
	ret
.eb_405:
	lea	rax, [body_405]
	mov	rdx, body_405_len
	ret
.eb_414:
	lea	rax, [body_414]
	mov	rdx, body_414_len
	ret

write_stderr:
	mov	rdx, rsi
	mov	rsi, rdi
	mov	rdi, STDERR
	mov	rax, SYS_write
	syscall
	ret

; rdi = conn*
log_access:
	push	r12
	mov	r12, rdi
	cmp	qword [r12 + CONN_STATUS_OFF], 0
	je	.la_done
	mov	rdi, [r12 + CONN_REQ_META_OFF + HTTP_REQ_PATH_PTR_OFF]
	test	rdi, rdi
	jz	.la_status_only
	mov	rax, [r12 + CONN_REQ_META_OFF + HTTP_REQ_METHOD_OFF]
	cmp	rax, HTTP_METHOD_HEAD
	je	.la_head
	lea	rdi, [log_get]
	mov	rsi, log_get_len
	call	write_stderr
	jmp	.la_path
.la_head:
	lea	rdi, [log_head]
	mov	rsi, log_head_len
	call	write_stderr
.la_path:
	mov	rdi, [r12 + CONN_REQ_META_OFF + HTTP_REQ_PATH_PTR_OFF]
	mov	rsi, [r12 + CONN_REQ_META_OFF + HTTP_REQ_PATH_LEN_OFF]
	call	write_stderr
	lea	rdi, [log_space]
	mov	rsi, 1
	call	write_stderr
	mov	rax, [r12 + CONN_STATUS_OFF]
	call	write_uint_stderr
	mov	al, 10
	call	write_char_stderr
	jmp	.la_done
.la_status_only:
	lea	rdi, [log_dash]
	mov	rsi, log_dash_len
	call	write_stderr
	mov	rax, [r12 + CONN_STATUS_OFF]
	call	write_uint_stderr
	mov	al, 10
	call	write_char_stderr
.la_done:
	pop	r12
	ret

write_char_stderr:
	mov	[stderr_char], al
	lea	rdi, [stderr_char]
	mov	rsi, 1
	jmp	write_stderr

write_uint_stderr:
	push	rbx
	push	r12
	mov	rbx, 10
	lea	r12, [log_num_buf + 31]
	mov	byte [r12], 0
	test	rax, rax
	jnz	.wus_loop
	dec	r12
	mov	byte [r12], '0'
	jmp	.wus_emit
.wus_loop:
	xor	rdx, rdx
	div	rbx
	add	dl, '0'
	dec	r12
	mov	[r12], dl
	test	rax, rax
	jnz	.wus_loop
.wus_emit:
	lea	rsi, [log_num_buf + 31]
	sub	rsi, r12
	mov	rdi, r12
	call	write_stderr
	pop	r12
	pop	rbx
	ret



segment readable writeable

opt_root db '--root', 0
opt_port db '--port', 0
opt_bind db '--bind', 0
bind_loopback db '127.0.0.1', 0
bind_any db '0.0.0.0', 0
dot_path db '.', 0
index_req_path db '/index.html'
index_req_path_len = $ - index_req_path
usage_msg db 'usage: httpmini [--root DIR] [--port PORT] [--bind 127.0.0.1|0.0.0.0]', 10
usage_msg_len = $ - usage_msg
bad_root_msg db 'httpmini: unable to canonicalize root', 10
bad_root_msg_len = $ - bad_root_msg

body_400 db '400 Bad Request', 10
body_400_len = $ - body_400
body_403 db '403 Forbidden', 10
body_403_len = $ - body_403
body_404 db '404 Not Found', 10
body_404_len = $ - body_404
body_405 db '405 Method Not Allowed', 10
body_405_len = $ - body_405
body_414 db '414 URI Too Long', 10
body_414_len = $ - body_414
body_500 db '500 Internal Server Error', 10
body_500_len = $ - body_500
log_get db 'GET '
log_get_len = $ - log_get
log_head db 'HEAD '
log_head_len = $ - log_head
log_dash db '- '
log_dash_len = $ - log_dash
log_space db ' '

argc dq ?
argv dq ?
root_arg dq ?
listen_port dq ?
bind_addr dq ?
listen_fd dq ?
stderr_char rb 1
log_num_buf rb 32

target_real rb PATH_REAL_MAX
candidate_path rb PATH_REAL_MAX
stat_buf rb STAT64_SIZE
connections rb CONN_SIZE * HTTPMINI_MAX_CLIENTS

coro_bss
sendfile_bss
http_response_bss
