; Mini-Redis TCP server v1: cooperative multi-client RESP server.
;
; PING/SET/GET/QUIT on one OS thread using fasm/core/coro.inc + kqueue.
; SET is persisted to miniredis.aof and replayed on restart.

format ELF64 executable 3
include "fasm/core/platform.inc"

match =SERVER_PORT, SERVER_PORT {
	SERVER_PORT equ 6379
}

MINIREDIS_MAX_CLIENTS equ 64
CORO_MAX_TASKS equ MINIREDIS_MAX_CLIENTS + 1
CORO_STACK_SIZE equ 16384

RESP_BUF_MAX equ 4096
RESP_MAX_ARGS equ 16
AOF_LINE_MAX equ 4096

CONN_FD_OFF equ 0
CONN_BUF_LEN_OFF equ 8
CONN_ARGC_OFF equ 16
CONN_QUIT_OFF equ 24
CONN_BUF_OFF equ 32
CONN_ARGV_OFF equ CONN_BUF_OFF + RESP_BUF_MAX
CONN_SIZE equ CONN_ARGV_OFF + RESP_MAX_ARGS * 8

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/hashmap_str.inc"
include "fasm/core/socket.inc"
include "fasm/core/coro.inc"
include "fasm/core/socket_async.inc"
include "fasm/core/runtime_bss.inc"

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init

	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, HASHMAP_DEFAULT_BUCKETS
	call	hashmap_str_init

	call	aof_replay
	call	aof_open_append
	cmp	rax, 0
	jl	start_fail

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

start_fail:
	exit	EXIT_FAILURE

accept_task:
	push	r12
	push	r13
	mov	edi, SERVER_PORT
	call	tcp_listen
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
.ct_loop:
	mov	rdi, r12
	call	resp_conn_read_command
	cmp	rax, 0
	jle	.ct_done
	mov	[r12 + CONN_ARGC_OFF], rax
	mov	rdi, r12
	call	server_dispatch
	cmp	qword [r12 + CONN_QUIT_OFF], 1
	jne	.ct_loop
.ct_done:
	mov	rdi, r12
	call	conn_close_release
	pop	r12
	ret

conn_pool_init:
	push	rbx
	xor	rbx, rbx
.cpi_loop:
	cmp	rbx, MINIREDIS_MAX_CLIENTS
	jae	.cpi_done
	mov	rax, rbx
	imul	rax, CONN_SIZE
	mov	qword [connections + rax + CONN_FD_OFF], -1
	mov	qword [connections + rax + CONN_BUF_LEN_OFF], 0
	mov	qword [connections + rax + CONN_ARGC_OFF], 0
	mov	qword [connections + rax + CONN_QUIT_OFF], 0
	inc	rbx
	jmp	.cpi_loop
.cpi_done:
	pop	rbx
	ret

; rax = conn*, or 0
conn_alloc:
	push	rbx
	xor	rbx, rbx
.ca_loop:
	cmp	rbx, MINIREDIS_MAX_CLIENTS
	jae	.ca_none
	mov	rax, rbx
	imul	rax, CONN_SIZE
	cmp	qword [connections + rax + CONN_FD_OFF], -1
	je	.ca_found
	inc	rbx
	jmp	.ca_loop
.ca_found:
	lea	rax, [connections + rax]
	jmp	.ca_out
.ca_none:
	xor	rax, rax
.ca_out:
	pop	rbx
	ret

; rdi = conn*, rsi = fd; rax = conn*
conn_init:
	mov	[rdi + CONN_FD_OFF], rsi
	mov	qword [rdi + CONN_BUF_LEN_OFF], 0
	mov	qword [rdi + CONN_ARGC_OFF], 0
	mov	qword [rdi + CONN_QUIT_OFF], 0
	mov	rax, rdi
	ret

; rdi = conn*
conn_release:
	mov	qword [rdi + CONN_FD_OFF], -1
	mov	qword [rdi + CONN_BUF_LEN_OFF], 0
	mov	qword [rdi + CONN_ARGC_OFF], 0
	mov	qword [rdi + CONN_QUIT_OFF], 0
	ret

; rdi = conn*
conn_close_release:
	push	rdi
	mov	rdi, [rdi + CONN_FD_OFF]
	cmp	rdi, 0
	jl	.ccr_release
	call	sock_close
.ccr_release:
	pop	rdi
	jmp	conn_release

; rdi = conn*
; rax = argc (>0), 0 = EOF, -1 = protocol/read error
resp_conn_read_command:
	push	rbx
	push	r12
	mov	r12, rdi
	mov	qword [r12 + CONN_BUF_LEN_OFF], 0
.rrc_loop:
	mov	rdi, r12
	call	resp_conn_try_parse
	test	rax, rax
	jg	.rrc_done
	cmp	rax, -1
	je	.rrc_err
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [r12 + CONN_BUF_OFF]
	mov	rax, [r12 + CONN_BUF_LEN_OFF]
	add	rsi, rax
	mov	rdx, RESP_BUF_MAX
	sub	rdx, rax
	cmp	rdx, 0
	je	.rrc_err
	call	coro_read
	cmp	rax, 0
	jl	.rrc_err
	je	.rrc_eof
	add	[r12 + CONN_BUF_LEN_OFF], rax
	jmp	.rrc_loop
.rrc_eof:
	cmp	qword [r12 + CONN_BUF_LEN_OFF], 0
	je	.rrc_zero
.rrc_err:
	mov	rax, -1
	jmp	.rrc_out
.rrc_zero:
	xor	rax, rax
	jmp	.rrc_out
.rrc_done:
	mov	qword [r12 + CONN_BUF_LEN_OFF], 0
.rrc_out:
	pop	r12
	pop	rbx
	ret

; rdi = conn*
; rax = argc, 0 = incomplete, -1 = bad protocol
resp_conn_try_parse:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r15, rdi
	mov	r12, [r15 + CONN_BUF_LEN_OFF]
	xor	r13, r13
	cmp	r12, 2
	jb	.rtp_incomplete
	cmp	byte [r15 + CONN_BUF_OFF], '*'
	jne	.rtp_bad
	mov	r13, 1
	call	.rtp_parse_int
	jc	.rtp_bad
	mov	r14, rax
	test	r14, r14
	js	.rtp_bad
	cmp	r14, RESP_MAX_ARGS
	ja	.rtp_bad
	xor	rbx, rbx
.rtp_arg_loop:
	cmp	rbx, r14
	jae	.rtp_ok
	cmp	r13, r12
	jae	.rtp_incomplete
	cmp	byte [r15 + CONN_BUF_OFF + r13], '$'
	jne	.rtp_bad
	inc	r13
	call	.rtp_parse_int
	jc	.rtp_bad
	mov	r8, rax
	test	r8, r8
	js	.rtp_bad
	mov	r9, r13
	mov	rax, r13
	add	rax, r8
	add	rax, 2
	cmp	rax, r12
	ja	.rtp_incomplete
	lea	rax, [r15 + CONN_BUF_OFF + r9]
	mov	[r15 + CONN_ARGV_OFF + rbx * 8], rax
	add	r13, r8
	cmp	r13, r12
	jae	.rtp_incomplete
	cmp	byte [r15 + CONN_BUF_OFF + r13], 13
	jne	.rtp_bad
	cmp	byte [r15 + CONN_BUF_OFF + r13 + 1], 10
	jne	.rtp_bad
	mov	byte [r15 + CONN_BUF_OFF + r13], 0
	add	r13, 2
	inc	rbx
	jmp	.rtp_arg_loop
.rtp_ok:
	mov	rax, r14
	jmp	.rtp_out
.rtp_incomplete:
	xor	rax, rax
	jmp	.rtp_out
.rtp_bad:
	mov	rax, -1
	jmp	.rtp_out

.rtp_parse_int:
	xor	rax, rax
	xor	r10, r10
.rpi_loop:
	cmp	r13, r12
	jae	.rpi_incomplete
	movzx	ecx, byte [r15 + CONN_BUF_OFF + r13]
	cmp	cl, 13
	je	.rpi_cr
	cmp	cl, '0'
	jb	.rpi_bad
	cmp	cl, '9'
	ja	.rpi_bad
	imul	rax, 10
	sub	cl, '0'
	add	rax, rcx
	inc	r13
	mov	r10, 1
	jmp	.rpi_loop
.rpi_cr:
	cmp	r10, 1
	jne	.rpi_bad
	inc	r13
	cmp	r13, r12
	jae	.rpi_incomplete
	cmp	byte [r15 + CONN_BUF_OFF + r13], 10
	jne	.rpi_bad
	inc	r13
	clc
	ret
.rpi_incomplete:
.rpi_bad:
	stc
	ret
.rtp_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = conn*
server_dispatch:
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r15, rdi
	mov	r12, [r15 + CONN_FD_OFF]
	mov	r13, [r15 + CONN_ARGC_OFF]

	mov	rdi, [r15 + CONN_ARGV_OFF]
	lea	rsi, [cmd_ping]
	call	str_eq
	test	rax, rax
	jnz	.sd_ping

	mov	rdi, [r15 + CONN_ARGV_OFF]
	lea	rsi, [cmd_set]
	call	str_eq
	test	rax, rax
	jnz	.sd_set

	mov	rdi, [r15 + CONN_ARGV_OFF]
	lea	rsi, [cmd_get]
	call	str_eq
	test	rax, rax
	jnz	.sd_get

	mov	rdi, [r15 + CONN_ARGV_OFF]
	lea	rsi, [cmd_quit]
	call	str_eq
	test	rax, rax
	jnz	.sd_quit

	mov	rdi, r12
	lea	rsi, [msg_unknown]
	call	resp_async_write_error
	jmp	.sd_done

.sd_ping:
	cmp	r13, 1
	jne	.sd_arity
	mov	rdi, r12
	lea	rsi, [msg_pong]
	call	resp_async_write_simple
	jmp	.sd_done

.sd_set:
	cmp	r13, 3
	jne	.sd_arity
	mov	rdi, [r15 + CONN_ARGV_OFF + 16]
	call	str_parse_int64
	test	rbx, rbx
	jnz	.sd_set_str
	mov	r14, rax
	mov	rcx, rax
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [r15 + CONN_ARGV_OFF + 8]
	call	hashmap_str_put_int
	mov	rdi, [r15 + CONN_ARGV_OFF + 8]
	mov	rax, r14
	call	aof_append_set_int
	jmp	.sd_ok
.sd_set_str:
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [r15 + CONN_ARGV_OFF + 8]
	mov	rcx, [r15 + CONN_ARGV_OFF + 16]
	call	hashmap_str_put_str
	mov	rdi, [r15 + CONN_ARGV_OFF + 8]
	mov	rsi, [r15 + CONN_ARGV_OFF + 16]
	call	aof_append_set_str
.sd_ok:
	mov	rdi, r12
	lea	rsi, [msg_ok]
	call	resp_async_write_simple
	jmp	.sd_done

.sd_get:
	cmp	r13, 2
	jne	.sd_arity
	lea	rdi, [storage_map]
	mov	rsi, [r15 + CONN_ARGV_OFF + 8]
	call	hashmap_str_get_entry
	test	rax, rax
	jz	.sd_get_null
	cmp	qword [rax + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_INT
	jne	.sd_get_str
	push	rax
	mov	rax, [rax + HASH_ENTRY_STR_VAL_OFF]
	mov	rdi, r12
	call	resp_async_write_int
	pop	rax
	jmp	.sd_done
.sd_get_str:
	mov	rsi, [rax + HASH_ENTRY_STR_VAL_OFF]
	push	rax
	push	rsi
	mov	rdi, rsi
	call	str_len
	mov	rdx, rax
	mov	rdi, r12
	pop	rsi
	call	resp_async_write_bulk
	pop	rax
	jmp	.sd_done
.sd_get_null:
	mov	rdi, r12
	call	resp_async_write_null
	jmp	.sd_done

.sd_quit:
	cmp	r13, 1
	jne	.sd_arity
	mov	rdi, r12
	lea	rsi, [msg_ok]
	call	resp_async_write_simple
	mov	qword [r15 + CONN_QUIT_OFF], 1
	jmp	.sd_done

.sd_arity:
	mov	rdi, r12
	lea	rsi, [msg_arity]
	call	resp_async_write_error

.sd_done:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	ret

aof_replay:
	push	rbx
	push	r12
	push	r13
	lea	rdi, [aof_path]
	open_file rdi, O_RDONLY, 0
	jump_if_syscall_error .ar_done
	mov	rbx, rax
	xor	r12, r12
.ar_loop:
	mov	rdi, rbx
	lea	rsi, [aof_char]
	mov	rdx, 1
	mov	rax, SYS_read
	syscall
	jc	.ar_close
	test	rax, rax
	jz	.ar_eof
	mov	al, [aof_char]
	cmp	al, 10
	je	.ar_line
	cmp	r12, AOF_LINE_MAX - 1
	jae	.ar_reset
	mov	[aof_line_buf + r12], al
	inc	r12
	jmp	.ar_loop
.ar_reset:
	xor	r12, r12
	jmp	.ar_loop
.ar_line:
	mov	byte [aof_line_buf + r12], 0
	test	r12, r12
	jz	.ar_next
	lea	rdi, [aof_line_buf]
	call	aof_replay_line
.ar_next:
	xor	r12, r12
	jmp	.ar_loop
.ar_eof:
	test	r12, r12
	jz	.ar_close
	mov	byte [aof_line_buf + r12], 0
	lea	rdi, [aof_line_buf]
	call	aof_replay_line
.ar_close:
	mov	rdi, rbx
	close_file rdi
.ar_done:
	pop	r13
	pop	r12
	pop	rbx
	ret

; rdi = null-terminated "I key value" or "S key value"
aof_replay_line:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	mov	bl, [r12]
	cmp	bl, '#'
	je	.arl_done
	cmp	bl, 'I'
	je	.arl_type_ok
	cmp	bl, 'S'
	jne	.arl_done
.arl_type_ok:
	cmp	byte [r12 + 1], ' '
	jne	.arl_done
	lea	r13, [r12 + 2]
	mov	r14, r13
.arl_key_loop:
	mov	al, [r14]
	test	al, al
	jz	.arl_done
	cmp	al, ' '
	je	.arl_split
	inc	r14
	jmp	.arl_key_loop
.arl_split:
	mov	byte [r14], 0
	inc	r14
	cmp	byte [r14], 0
	je	.arl_done
	cmp	bl, 'I'
	je	.arl_int
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, r13
	mov	rcx, r14
	call	hashmap_str_put_str
	jmp	.arl_done
.arl_int:
	mov	rdi, r14
	call	str_parse_int64
	test	rbx, rbx
	jnz	.arl_done
	mov	rcx, rax
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, r13
	call	hashmap_str_put_int
.arl_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

; rax = fd or -1
aof_open_append:
	lea	rdi, [aof_path]
	open_file rdi, O_WRONLY or O_CREAT or O_APPEND, 420
	jump_if_syscall_error .aoa_err
	mov	[aof_fd], rax
	ret
.aoa_err:
	mov	qword [aof_fd], -1
	mov	rax, -1
	ret

; rdi = key ptr, rax = int64 value
aof_append_set_int:
	push	r12
	push	r13
	sub	rsp, 32
	mov	r12, rdi
	mov	r13, rax
	lea	rsi, [aof_prefix_i]
	mov	rdx, 2
	call	aof_write
	mov	rdi, r12
	call	aof_write_cstr
	lea	rsi, [aof_space]
	mov	rdx, 1
	call	aof_write
	mov	rax, r13
	lea	rdi, [rsp + 31]
	call	print_int64_to_buf
	mov	rsi, rdi
	mov	rdx, rax
	call	aof_write
	lea	rsi, [aof_nl]
	mov	rdx, 1
	call	aof_write
	add	rsp, 32
	pop	r13
	pop	r12
	ret

; rdi = key ptr, rsi = value ptr
aof_append_set_str:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	lea	rsi, [aof_prefix_s]
	mov	rdx, 2
	call	aof_write
	mov	rdi, r12
	call	aof_write_cstr
	lea	rsi, [aof_space]
	mov	rdx, 1
	call	aof_write
	mov	rdi, r13
	call	aof_write_cstr
	lea	rsi, [aof_nl]
	mov	rdx, 1
	call	aof_write
	pop	r13
	pop	r12
	ret

; rdi = cstr
aof_write_cstr:
	push	rdi
	call	str_len
	pop	rsi
	mov	rdx, rax
	jmp	aof_write

; rsi = buffer, rdx = len
aof_write:
	push	r12
	push	r13
	mov	r12, rsi
	mov	r13, rdx
	cmp	qword [aof_fd], 0
	jl	.aw_done
.aw_loop:
	test	r13, r13
	jz	.aw_done
	mov	rdi, [aof_fd]
	mov	rsi, r12
	mov	rdx, r13
	mov	rax, SYS_write
	syscall
	jc	.aw_done
	test	rax, rax
	jle	.aw_done
	add	r12, rax
	sub	r13, rax
	jmp	.aw_loop
.aw_done:
	pop	r13
	pop	r12
	ret

; rdi = fd, rsi = payload cstr
resp_async_write_simple:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, r12
	lea	rsi, [resp_plus]
	mov	rdx, 1
	call	coro_write_all
	mov	rdi, r13
	call	str_len
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, rax
	call	coro_write_all
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	coro_write_all
	pop	r13
	pop	r12
	ret

; rdi = fd, rsi = message cstr
resp_async_write_error:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, r12
	lea	rsi, [resp_err_prefix]
	mov	rdx, resp_err_prefix_len
	call	coro_write_all
	mov	rdi, r13
	call	str_len
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, rax
	call	coro_write_all
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	coro_write_all
	pop	r13
	pop	r12
	ret

; rdi = fd, rax = int64
resp_async_write_int:
	push	r12
	push	r13
	sub	rsp, 32
	mov	r12, rdi
	mov	r13, rax
	mov	rdi, r12
	lea	rsi, [resp_colon]
	mov	rdx, 1
	call	coro_write_all
	mov	rax, r13
	lea	rdi, [rsp + 31]
	call	print_int64_to_buf
	mov	rsi, rdi
	mov	rdx, rax
	mov	rdi, r12
	call	coro_write_all
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	coro_write_all
	add	rsp, 32
	pop	r13
	pop	r12
	ret

; rdi = fd
resp_async_write_null:
	lea	rsi, [resp_null]
	mov	rdx, resp_null_len
	jmp	coro_write_all

; rdi = fd, rsi = payload ptr, rdx = len
resp_async_write_bulk:
	push	r12
	push	r13
	push	r14
	sub	rsp, 32
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	mov	rdi, r12
	lea	rsi, [resp_dollar]
	mov	rdx, 1
	call	coro_write_all
	mov	rax, r14
	lea	rdi, [rsp + 31]
	call	print_int64_to_buf
	mov	rsi, rdi
	mov	rdx, rax
	mov	rdi, r12
	call	coro_write_all
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	coro_write_all
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r14
	call	coro_write_all
	mov	rdi, r12
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	coro_write_all
	add	rsp, 32
	pop	r14
	pop	r13
	pop	r12
	ret

cmd_ping db 'PING', 0
cmd_set db 'SET', 0
cmd_get db 'GET', 0
cmd_quit db 'QUIT', 0

msg_ok db 'OK', 0
msg_pong db 'PONG', 0
msg_unknown db 'unknown command', 0
msg_arity db 'wrong number of arguments', 0

resp_plus db '+'
resp_colon db ':'
resp_dollar db '$'
resp_crlf db 13, 10
resp_err_prefix db '-ERR '
resp_err_prefix_len = $ - resp_err_prefix
resp_null db '$-1', 13, 10
resp_null_len = $ - resp_null

aof_path db 'miniredis.aof', 0
aof_prefix_i db 'I '
aof_prefix_s db 'S '
aof_space db ' '
aof_nl db 10

segment readable writeable

storage_heap rb HEAP_SIZE
storage_map rb HASHMAP_STR_SIZE

listen_fd dq ?
aof_fd dq ?
aof_char rb 1
aof_line_buf rb AOF_LINE_MAX
connections rb CONN_SIZE * MINIREDIS_MAX_CLIENTS

coro_bss
runtime_print_bss
