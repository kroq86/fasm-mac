; logbus: single-threaded durable append-only message broker.
;
; RESP-like protocol on one OS thread using fasm/core/coro.inc + kqueue.

format ELF64 executable 3
include "fasm/core/platform.inc"

LOGBUS_MAX_CLIENTS equ 64
CORO_MAX_TASKS equ LOGBUS_MAX_CLIENTS + 1
CORO_STACK_SIZE equ 16384

LOGBUS_DEFAULT_PORT equ 9092
LOGBUS_PAYLOAD_MAX equ 65536
LOGBUS_WIRE_PAYLOAD_MAX equ 4096
LOGBUS_FETCHBATCH_MAX equ 1048576
RESP_BUF_MAX equ 8192
RESP_MAX_ARGS equ 16
FETCH_MAX_RECORDS equ 64
FETCH_BUF_MAX equ 8192

CONN_FD_OFF equ 0
CONN_BUF_LEN_OFF equ 8
CONN_ARGC_OFF equ 16
CONN_QUIT_OFF equ 24
CONN_BUF_OFF equ 32
CONN_ARGV_OFF equ CONN_BUF_OFF + RESP_BUF_MAX
CONN_ARGLEN_OFF equ CONN_ARGV_OFF + RESP_MAX_ARGS * 8
CONN_FETCH_OFFSETS_OFF equ CONN_ARGLEN_OFF + RESP_MAX_ARGS * 8
CONN_FETCH_PTRS_OFF equ CONN_FETCH_OFFSETS_OFF + FETCH_MAX_RECORDS * 8
CONN_FETCH_LENS_OFF equ CONN_FETCH_PTRS_OFF + FETCH_MAX_RECORDS * 8
CONN_FETCH_BUF_OFF equ CONN_FETCH_LENS_OFF + FETCH_MAX_RECORDS * 8
CONN_SIZE equ CONN_FETCH_BUF_OFF + FETCH_BUF_MAX

segment readable executable

include "fasm/core/str.inc"
include "fasm/core/print_io.inc"
include "fasm/core/socket.inc"
include "fasm/core/coro.inc"
include "fasm/core/socket_async.inc"
include "fasm/core/sendfile.inc"
include "fasm/core/log_segment.inc"
include "fasm/core/topic_store.inc"
include "fasm/core/runtime_bss.inc"

entry start

start:
	mov	rbx, rsp
	mov	rax, [rbx]
	mov	[argc], rax
	lea	rax, [rbx + 8]
	mov	[argv], rax
	lea	rax, [default_dir]
	mov	[data_dir], rax
	mov	qword [listen_port], LOGBUS_DEFAULT_PORT
	mov	qword [bind_addr], INADDR_ANY
	call	parse_args
	cmp	rax, 0
	jl	start_usage
	mov	rdi, [data_dir]
	lea	rsi, [scratch_path]
	call	topic_store_prepare
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
start_usage:
	lea	rdi, [usage_msg]
	mov	rsi, usage_msg_len
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
	lea	rsi, [opt_dir]
	call	str_eq
	cmp	rax, 1
	je	.pa_dir
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
	jmp	.pa_bad
.pa_dir:
	inc	rbx
	cmp	rbx, [argc]
	jae	.pa_bad
	mov	rax, [argv]
	mov	rax, [rax + rbx * 8]
	mov	[data_dir], rax
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
	push	r12
	push	r13
	lea	r12, [connections]
	mov	r13, LOGBUS_MAX_CLIENTS
.cpi_loop:
	test	r13, r13
	jz	.cpi_done
	mov	qword [r12 + CONN_FD_OFF], -1
	add	r12, CONN_SIZE
	dec	r13
	jmp	.cpi_loop
.cpi_done:
	pop	r13
	pop	r12
	ret

conn_alloc:
	push	r12
	push	r13
	lea	r12, [connections]
	mov	r13, LOGBUS_MAX_CLIENTS
.ca_loop:
	test	r13, r13
	jz	.ca_none
	cmp	qword [r12 + CONN_FD_OFF], -1
	je	.ca_found
	add	r12, CONN_SIZE
	dec	r13
	jmp	.ca_loop
.ca_found:
	mov	rax, r12
	jmp	.ca_out
.ca_none:
	xor	rax, rax
.ca_out:
	pop	r13
	pop	r12
	ret

conn_init:
	mov	[rdi + CONN_FD_OFF], rsi
	mov	qword [rdi + CONN_BUF_LEN_OFF], 0
	mov	qword [rdi + CONN_ARGC_OFF], 0
	mov	qword [rdi + CONN_QUIT_OFF], 0
	mov	rax, rdi
	ret

conn_release:
	mov	qword [rdi + CONN_FD_OFF], -1
	mov	qword [rdi + CONN_BUF_LEN_OFF], 0
	mov	qword [rdi + CONN_ARGC_OFF], 0
	mov	qword [rdi + CONN_QUIT_OFF], 0
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

; rdi = conn*; rax = argc, 0 EOF, -1 error
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
	mov	rbx, [r12 + CONN_BUF_LEN_OFF]
	cmp	rbx, RESP_BUF_MAX
	jae	.rrc_err
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [r12 + CONN_BUF_OFF + rbx]
	mov	rdx, RESP_BUF_MAX
	sub	rdx, rbx
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

; rdi = conn*; rax = argc, 0 incomplete, -1 bad
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
	mov	[r15 + CONN_ARGLEN_OFF + rbx * 8], r8
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

server_dispatch:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, [r12 + CONN_FD_OFF]
	mov	rdi, [r12 + CONN_ARGV_OFF]
	lea	rsi, [cmd_ping]
	call	str_eq
	cmp	rax, 1
	je	.sd_ping
	mov	rdi, [r12 + CONN_ARGV_OFF]
	lea	rsi, [cmd_produce]
	call	str_eq
	cmp	rax, 1
	je	.sd_produce
	mov	rdi, [r12 + CONN_ARGV_OFF]
	lea	rsi, [cmd_fetch]
	call	str_eq
	cmp	rax, 1
	je	.sd_fetch
	mov	rdi, [r12 + CONN_ARGV_OFF]
	lea	rsi, [cmd_fetchbatch]
	call	str_eq
	cmp	rax, 1
	je	.sd_fetchbatch
	mov	rdi, [r12 + CONN_ARGV_OFF]
	lea	rsi, [cmd_commit]
	call	str_eq
	cmp	rax, 1
	je	.sd_commit
	mov	rdi, [r12 + CONN_ARGV_OFF]
	lea	rsi, [cmd_offset]
	call	str_eq
	cmp	rax, 1
	je	.sd_offset
	mov	rdi, [r12 + CONN_ARGV_OFF]
	lea	rsi, [cmd_quit]
	call	str_eq
	cmp	rax, 1
	je	.sd_quit
	mov	rdi, r13
	lea	rsi, [msg_unknown]
	call	resp_async_write_error
	jmp	.sd_done
.sd_ping:
	cmp	qword [r12 + CONN_ARGC_OFF], 1
	jne	.sd_arity
	mov	rdi, r13
	lea	rsi, [msg_pong]
	call	resp_async_write_simple
	jmp	.sd_done
.sd_produce:
	mov	rdi, r12
	call	cmd_do_produce
	jmp	.sd_done
.sd_fetch:
	mov	rdi, r12
	call	cmd_do_fetch
	jmp	.sd_done
.sd_fetchbatch:
	mov	rdi, r12
	call	cmd_do_fetchbatch
	jmp	.sd_done
.sd_commit:
	mov	rdi, r12
	call	cmd_do_commit
	jmp	.sd_done
.sd_offset:
	mov	rdi, r12
	call	cmd_do_offset
	jmp	.sd_done
.sd_quit:
	cmp	qword [r12 + CONN_ARGC_OFF], 1
	jne	.sd_arity
	mov	rdi, r13
	lea	rsi, [msg_ok]
	call	resp_async_write_simple
	mov	qword [r12 + CONN_QUIT_OFF], 1
	jmp	.sd_done
.sd_arity:
	mov	rdi, r13
	lea	rsi, [msg_arity]
	call	resp_async_write_error
.sd_done:
	pop	r13
	pop	r12
	ret

cmd_do_produce:
	push	r12
	mov	r12, rdi
	cmp	qword [r12 + CONN_ARGC_OFF], 3
	jne	.cdp_arity
	mov	rax, [r12 + CONN_ARGLEN_OFF + 16]
	cmp	rax, LOGBUS_WIRE_PAYLOAD_MAX
	ja	.cdp_too_large
	mov	rdi, [data_dir]
	mov	rsi, [r12 + CONN_ARGV_OFF + 8]
	lea	rdx, [log_path]
	lea	rcx, [idx_path]
	lea	r8, [scratch_path]
	call	topic_build_segment_paths
	cmp	rax, 0
	jl	.cdp_bad_name
	lea	rdi, [log_path]
	lea	rsi, [idx_path]
	mov	rdx, [r12 + CONN_ARGV_OFF + 16]
	mov	rcx, [r12 + CONN_ARGLEN_OFF + 16]
	call	log_segment_append
	cmp	rax, 0
	jl	.cdp_ioerr
	mov	rdi, [r12 + CONN_FD_OFF]
	call	resp_async_write_int
	jmp	.cdp_done
.cdp_arity:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_arity]
	call	resp_async_write_error
	jmp	.cdp_done
.cdp_bad_name:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_bad_name]
	call	resp_async_write_error
	jmp	.cdp_done
.cdp_too_large:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_too_large]
	call	resp_async_write_error
	jmp	.cdp_done
.cdp_ioerr:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_io]
	call	resp_async_write_error
.cdp_done:
	pop	r12
	ret

cmd_do_fetch:
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	cmp	qword [r12 + CONN_ARGC_OFF], 4
	jne	.cdf_arity
	mov	rdi, [r12 + CONN_ARGV_OFF + 16]
	call	str_parse_int64
	test	rbx, rbx
	jnz	.cdf_arity
	test	rax, rax
	js	.cdf_arity
	mov	r13, rax
	mov	rdi, [r12 + CONN_ARGV_OFF + 24]
	call	str_parse_int64
	test	rbx, rbx
	jnz	.cdf_arity
	test	rax, rax
	jle	.cdf_arity
	mov	r14, rax
	cmp	r14, FETCH_BUF_MAX
	jbe	.cdf_paths
	mov	r14, FETCH_BUF_MAX
.cdf_paths:
	mov	rdi, [data_dir]
	mov	rsi, [r12 + CONN_ARGV_OFF + 8]
	lea	rdx, [log_path]
	lea	rcx, [idx_path]
	lea	r8, [scratch_path]
	call	topic_build_segment_paths
	cmp	rax, 0
	jl	.cdf_bad_name
	mov	rdi, r12
	mov	rsi, r13
	mov	rdx, r14
	call	fetch_records
	cmp	rax, 0
	jl	.cdf_ioerr
	mov	rdi, r12
	mov	rsi, rax
	call	resp_async_write_fetch
	jmp	.cdf_done
.cdf_arity:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_arity]
	call	resp_async_write_error
	jmp	.cdf_done
.cdf_bad_name:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_bad_name]
	call	resp_async_write_error
	jmp	.cdf_done
.cdf_ioerr:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_io]
	call	resp_async_write_error
.cdf_done:
	pop	r14
	pop	r13
	pop	r12
	ret

; rdi=conn, rsi=start offset, rdx=max payload bytes; rax=count or -1
fetch_records:
	push	rbx
	push	r12
	push	r13
	push	r14
	push	r15
	mov	r12, rdi
	mov	r13, rsi
	mov	r14, rdx
	xor	rbx, rbx
	xor	r15, r15
.fr_loop:
	cmp	rbx, FETCH_MAX_RECORDS
	jae	.fr_done
	cmp	r15, r14
	jae	.fr_done
	lea	rcx, [r12 + CONN_FETCH_BUF_OFF + r15]
	mov	r8, FETCH_BUF_MAX
	sub	r8, r15
	lea	rdi, [log_path]
	lea	rsi, [idx_path]
	mov	rdx, r13
	lea	r9, [fetch_last_len]
	call	log_segment_read
	cmp	rax, LOGSEG_EOF
	je	.fr_done
	cmp	rax, LOGSEG_OK
	jne	.fr_err
	mov	rax, [fetch_last_len]
	mov	r10, r15
	add	r10, rax
	cmp	r10, r14
	ja	.fr_done
	mov	[r12 + CONN_FETCH_OFFSETS_OFF + rbx * 8], r13
	lea	r11, [r12 + CONN_FETCH_BUF_OFF + r15]
	mov	[r12 + CONN_FETCH_PTRS_OFF + rbx * 8], r11
	mov	[r12 + CONN_FETCH_LENS_OFF + rbx * 8], rax
	mov	r15, r10
	inc	r13
	inc	rbx
	jmp	.fr_loop
.fr_done:
	mov	rax, rbx
	jmp	.fr_out
.fr_err:
	mov	rax, -1
.fr_out:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

cmd_do_fetchbatch:
	push	rbx
	push	r12
	push	r13
	push	r14
	mov	r12, rdi
	cmp	qword [r12 + CONN_ARGC_OFF], 4
	jne	.cdfb_arity
	mov	rdi, [r12 + CONN_ARGV_OFF + 16]
	call	str_parse_int64
	test	rbx, rbx
	jnz	.cdfb_arity
	test	rax, rax
	js	.cdfb_arity
	mov	r13, rax
	mov	rdi, [r12 + CONN_ARGV_OFF + 24]
	call	str_parse_int64
	test	rbx, rbx
	jnz	.cdfb_arity
	test	rax, rax
	jle	.cdfb_arity
	mov	r14, rax
	cmp	r14, LOGBUS_FETCHBATCH_MAX
	jbe	.cdfb_paths
	mov	r14, LOGBUS_FETCHBATCH_MAX
.cdfb_paths:
	mov	rdi, [data_dir]
	mov	rsi, [r12 + CONN_ARGV_OFF + 8]
	lea	rdx, [log_path]
	lea	rcx, [idx_path]
	lea	r8, [scratch_path]
	call	topic_build_segment_paths
	cmp	rax, 0
	jl	.cdfb_bad_name
	lea	rdi, [log_path]
	lea	rsi, [idx_path]
	mov	rdx, r13
	mov	rcx, r14
	lea	r8, [fetchbatch_byte_offset]
	lea	r9, [fetchbatch_byte_count]
	call	log_segment_batch_span
	cmp	rax, LOGSEG_OK
	jne	.cdfb_ioerr
	mov	rdi, [r12 + CONN_FD_OFF]
	mov	rax, [fetchbatch_byte_count]
	call	resp_async_write_bulk_header
	cmp	qword [fetchbatch_byte_count], 0
	je	.cdfb_trailer
	lea	rdi, [log_path]
	open_file rdi, O_RDONLY, 0
	jump_if_syscall_error .cdfb_ioerr_after_header
	mov	rbx, rax
	mov	rdi, rbx
	mov	rsi, [r12 + CONN_FD_OFF]
	mov	rdx, [fetchbatch_byte_offset]
	mov	rcx, [fetchbatch_byte_count]
	call	file_send_range_socket
	push	rax
	mov	rdi, rbx
	close_file rdi
	pop	rax
	cmp	rax, 0
	jl	.cdfb_close_conn
.cdfb_trailer:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [resp_crlf]
	mov	rdx, 2
	call	coro_write_all
	jmp	.cdfb_done
.cdfb_arity:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_arity]
	call	resp_async_write_error
	jmp	.cdfb_done
.cdfb_bad_name:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_bad_name]
	call	resp_async_write_error
	jmp	.cdfb_done
.cdfb_ioerr:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_io]
	call	resp_async_write_error
	jmp	.cdfb_done
.cdfb_ioerr_after_header:
.cdfb_close_conn:
	mov	qword [r12 + CONN_QUIT_OFF], 1
.cdfb_done:
	pop	r14
	pop	r13
	pop	r12
	pop	rbx
	ret

cmd_do_commit:
	push	r12
	push	r13
	mov	r12, rdi
	cmp	qword [r12 + CONN_ARGC_OFF], 4
	jne	.cdc_arity
	mov	rdi, [r12 + CONN_ARGV_OFF + 24]
	call	str_parse_int64
	test	rbx, rbx
	jnz	.cdc_arity
	test	rax, rax
	js	.cdc_arity
	mov	r13, rax
	mov	rdi, [data_dir]
	mov	rsi, [r12 + CONN_ARGV_OFF + 8]
	mov	rdx, [r12 + CONN_ARGV_OFF + 16]
	lea	rcx, [offset_path]
	call	topic_build_offset_path
	cmp	rax, 0
	jl	.cdc_bad_name
	lea	rdi, [offset_path]
	mov	rax, r13
	call	write_offset_file
	cmp	rax, 0
	jl	.cdc_ioerr
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_ok]
	call	resp_async_write_simple
	jmp	.cdc_done
.cdc_arity:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_arity]
	call	resp_async_write_error
	jmp	.cdc_done
.cdc_bad_name:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_bad_name]
	call	resp_async_write_error
	jmp	.cdc_done
.cdc_ioerr:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_io]
	call	resp_async_write_error
.cdc_done:
	pop	r13
	pop	r12
	ret

cmd_do_offset:
	push	r12
	mov	r12, rdi
	cmp	qword [r12 + CONN_ARGC_OFF], 3
	jne	.cdo_arity
	mov	rdi, [data_dir]
	mov	rsi, [r12 + CONN_ARGV_OFF + 8]
	mov	rdx, [r12 + CONN_ARGV_OFF + 16]
	lea	rcx, [offset_path]
	call	topic_build_offset_path
	cmp	rax, 0
	jl	.cdo_bad_name
	lea	rdi, [offset_path]
	call	read_offset_file
	mov	rdi, [r12 + CONN_FD_OFF]
	call	resp_async_write_int
	jmp	.cdo_done
.cdo_arity:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_arity]
	call	resp_async_write_error
	jmp	.cdo_done
.cdo_bad_name:
	mov	rdi, [r12 + CONN_FD_OFF]
	lea	rsi, [msg_bad_name]
	call	resp_async_write_error
.cdo_done:
	pop	r12
	ret

; rdi=path, rax=offset; returns 0/-1
write_offset_file:
	push	rbx
	push	r12
	sub	rsp, 40
	mov	r12, rax
	open_file rdi, O_WRONLY or O_CREAT or O_TRUNC, 420
	jump_if_syscall_error .wof_err_stack
	mov	rbx, rax
	mov	rax, r12
	lea	rdi, [rsp + 31]
	call	print_int64_to_buf
	mov	rsi, rdi
	mov	rdx, rax
	mov	rdi, rbx
	call	logseg_write_all_fd
	cmp	rax, 0
	jl	.wof_close_err
	mov	byte [rsp + 32], 10
	mov	rdi, rbx
	lea	rsi, [rsp + 32]
	mov	rdx, 1
	call	logseg_write_all_fd
	cmp	rax, 0
	jl	.wof_close_err
	mov	rdi, rbx
	close_file rdi
	xor	rax, rax
	jmp	.wof_out
.wof_close_err:
	mov	rdi, rbx
	close_file rdi
.wof_err_stack:
	mov	rax, -1
.wof_out:
	add	rsp, 40
	pop	r12
	pop	rbx
	ret

; rdi=path; rax=offset, default 0
read_offset_file:
	push	rbx
	push	r12
	open_file rdi, O_RDONLY, 0
	jump_if_syscall_error .rof_zero
	mov	rbx, rax
	mov	rdi, rbx
	lea	rsi, [offset_read_buf]
	mov	rdx, 63
	mov	rax, SYS_read
	syscall
	jc	.rof_close_zero
	mov	byte [offset_read_buf + rax], 0
	xor	rcx, rcx
.rof_strip:
	cmp	rcx, rax
	jae	.rof_parse
	cmp	byte [offset_read_buf + rcx], 10
	je	.rof_zero_term
	cmp	byte [offset_read_buf + rcx], 13
	je	.rof_zero_term
	inc	rcx
	jmp	.rof_strip
.rof_zero_term:
	mov	byte [offset_read_buf + rcx], 0
.rof_parse:
	mov	rdi, rbx
	close_file rdi
	lea	rdi, [offset_read_buf]
	call	str_parse_int64
	mov	r12, rbx
	test	r12, r12
	jnz	.rof_zero
	jmp	.rof_out
.rof_close_zero:
	mov	rdi, rbx
	close_file rdi
.rof_zero:
	xor	rax, rax
.rof_out:
	pop	r12
	pop	rbx
	ret

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

; rdi = fd, rsi = count
resp_async_write_array_count:
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	lea	rsi, [resp_star]
	mov	rdx, 1
	mov	rdi, r12
	call	coro_write_all
	mov	rdi, r12
	mov	rax, r13
	call	resp_async_write_int_digits
	pop	r13
	pop	r12
	ret

; rdi=fd, rax=int64; writes digits + CRLF without RESP type prefix
resp_async_write_int_digits:
	push	r12
	push	r13
	sub	rsp, 32
	mov	r12, rdi
	mov	r13, rax
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

; rdi = fd, rax = bulk byte length
resp_async_write_bulk_header:
	push	r12
	push	r13
	sub	rsp, 32
	mov	r12, rdi
	mov	r13, rax
	mov	rdi, r12
	lea	rsi, [resp_dollar]
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

; rdi=conn, rsi=count
resp_async_write_fetch:
	push	rbx
	push	r12
	push	r13
	mov	r12, rdi
	mov	r13, rsi
	mov	rdi, [r12 + CONN_FD_OFF]
	mov	rsi, r13
	call	resp_async_write_array_count
	xor	rbx, rbx
.rawf_loop:
	cmp	rbx, r13
	jae	.rawf_done
	mov	rdi, [r12 + CONN_FD_OFF]
	mov	rsi, 2
	call	resp_async_write_array_count
	mov	rax, [r12 + CONN_FETCH_OFFSETS_OFF + rbx * 8]
	mov	rdi, [r12 + CONN_FD_OFF]
	call	resp_async_write_int
	mov	rdi, [r12 + CONN_FD_OFF]
	mov	rsi, [r12 + CONN_FETCH_PTRS_OFF + rbx * 8]
	mov	rdx, [r12 + CONN_FETCH_LENS_OFF + rbx * 8]
	call	resp_async_write_bulk
	inc	rbx
	jmp	.rawf_loop
.rawf_done:
	pop	r13
	pop	r12
	pop	rbx
	ret

write_stderr:
	mov	rdx, rsi
	mov	rsi, rdi
	mov	rdi, STDERR
	mov	rax, SYS_write
	syscall
	ret

cmd_ping db 'PING', 0
cmd_produce db 'PRODUCE', 0
cmd_fetch db 'FETCH', 0
cmd_fetchbatch db 'FETCHBATCH', 0
cmd_commit db 'COMMIT', 0
cmd_offset db 'OFFSET', 0
cmd_quit db 'QUIT', 0

msg_ok db 'OK', 0
msg_pong db 'PONG', 0
msg_unknown db 'unknown command', 0
msg_arity db 'wrong number of arguments', 0
msg_bad_name db 'invalid topic or group name', 0
msg_too_large db 'payload too large', 0
msg_io db 'storage error', 0

resp_plus db '+'
resp_colon db ':'
resp_dollar db '$'
resp_star db '*'
resp_crlf db 13, 10
resp_err_prefix db '-ERR '
resp_err_prefix_len = $ - resp_err_prefix

segment readable writeable

opt_dir db '--dir', 0
opt_port db '--port', 0
opt_bind db '--bind', 0
bind_loopback db '127.0.0.1', 0
bind_any db '0.0.0.0', 0
default_dir db './data', 0
usage_msg db 'usage: logbus [--dir DIR] [--port PORT] [--bind 127.0.0.1|0.0.0.0]', 10
usage_msg_len = $ - usage_msg

argc dq ?
argv dq ?
data_dir dq ?
listen_port dq ?
bind_addr dq ?
listen_fd dq ?
fetch_last_len dq ?
fetchbatch_byte_offset dq ?
fetchbatch_byte_count dq ?

log_path rb TOPIC_STORE_PATH_MAX
idx_path rb TOPIC_STORE_PATH_MAX
offset_path rb TOPIC_STORE_PATH_MAX
scratch_path rb TOPIC_STORE_PATH_MAX
offset_read_buf rb 64
connections rb CONN_SIZE * LOGBUS_MAX_CLIENTS

coro_bss
sendfile_bss
log_segment_bss
topic_store_bss
runtime_print_bss
