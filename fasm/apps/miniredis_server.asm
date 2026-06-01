; Mini-Redis TCP server v0: RESP + PING/SET/GET/QUIT on port 6379.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/hashmap_str.inc"
include "fasm/core/socket.inc"
include "fasm/core/resp.inc"

SERVER_PORT equ 6379

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init

	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, HASHMAP_DEFAULT_BUCKETS
	call	hashmap_str_init

	mov	edi, SERVER_PORT
	call	tcp_listen
	cmp	rax, 0
	jl	start_fail
	mov	[listen_fd], rax

.server_loop:
	mov	rdi, [listen_fd]
	call	tcp_accept
	cmp	rax, 0
	jl	.server_loop
	mov	[client_fd], rax

.client_loop:
	mov	rdi, [client_fd]
	call	resp_read_command
	cmp	rax, 0
	jle	.client_done
	mov	[argc], rax
	call	server_dispatch
	cmp	qword [quit_flag], 1
	je	.client_done
	jmp	.client_loop

.client_done:
	mov	qword [quit_flag], 0
	mov	rdi, [client_fd]
	call	sock_close
	jmp	.server_loop

start_fail:
	exit EXIT_FAILURE

server_dispatch:
	push	r12
	mov	r12, [client_fd]

	mov	rdi, [resp_argv]
	lea	rsi, [cmd_ping]
	call	str_eq
	test	rax, rax
	jnz	.sd_ping

	mov	rdi, [resp_argv]
	lea	rsi, [cmd_set]
	call	str_eq
	test	rax, rax
	jnz	.sd_set

	mov	rdi, [resp_argv]
	lea	rsi, [cmd_get]
	call	str_eq
	test	rax, rax
	jnz	.sd_get

	mov	rdi, [resp_argv]
	lea	rsi, [cmd_quit]
	call	str_eq
	test	rax, rax
	jnz	.sd_quit

	mov	rdi, r12
	lea	rsi, [msg_unknown]
	call	resp_write_error
	jmp	.sd_done

.sd_ping:
	cmp	qword [argc], 1
	jne	.sd_arity
	mov	rdi, r12
	lea	rsi, [msg_pong]
	call	resp_write_simple
	jmp	.sd_done

.sd_set:
	cmp	qword [argc], 3
	jne	.sd_arity
	mov	rdi, [resp_argv + 16]
	call	str_parse_int64
	test	rbx, rbx
	jnz	.sd_set_str
	mov	rcx, rax
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [resp_argv + 8]
	call	hashmap_str_put_int
	jmp	.sd_ok
.sd_set_str:
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [resp_argv + 8]
	mov	rcx, [resp_argv + 16]
	call	hashmap_str_put_str
.sd_ok:
	mov	rdi, r12
	lea	rsi, [msg_ok]
	call	resp_write_simple
	jmp	.sd_done

.sd_get:
	cmp	qword [argc], 2
	jne	.sd_arity
	lea	rdi, [storage_map]
	mov	rsi, [resp_argv + 8]
	call	hashmap_str_get_entry
	test	rax, rax
	jz	.sd_get_null
	push	rax
	cmp	qword [rax + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_INT
	jne	.sd_get_str
	mov	rax, [rax + HASH_ENTRY_STR_VAL_OFF]
	mov	rdi, r12
	call	resp_write_int
	pop	rax
	jmp	.sd_done
.sd_get_str:
	mov	rsi, [rax + HASH_ENTRY_STR_VAL_OFF]
	push	rsi
	mov	rdi, rsi
	call	str_len
	mov	rdx, rax
	mov	rdi, r12
	pop	rsi
	call	resp_write_bulk
	pop	rax
	jmp	.sd_done
.sd_get_null:
	mov	rdi, r12
	call	resp_write_null
	jmp	.sd_done

.sd_quit:
	cmp	qword [argc], 1
	jne	.sd_arity
	mov	rdi, r12
	lea	rsi, [msg_ok]
	call	resp_write_simple
	mov	qword [quit_flag], 1
	jmp	.sd_done

.sd_arity:
	mov	rdi, r12
	lea	rsi, [msg_arity]
	call	resp_write_error

.sd_done:
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

segment readable writeable

storage_heap rb HEAP_SIZE
storage_map rb HASHMAP_STR_SIZE

listen_fd dq ?
client_fd dq ?
argc dq ?
quit_flag dq ?

resp_buf rb RESP_BUF_MAX
resp_buf_len dq ?
resp_argv rq RESP_MAX_ARGS

include "fasm/core/runtime_bss.inc"
runtime_print_bss
