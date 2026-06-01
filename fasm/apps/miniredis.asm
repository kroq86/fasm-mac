; Mini-Redis v1.1: in-memory string-key KV REPL.
;
; PING, SET, GET, EXISTS, DEL, DBSIZE, INCR, DECR, MGET, KEYS, SAVE, LOAD, QUIT, EXIT

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/hashmap_str.inc"
include "fasm/core/repl.inc"

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init

	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, HASHMAP_DEFAULT_BUCKETS
	call	hashmap_str_init

.repl_loop:
	call	repl_write_prompt

	lea	rdi, [line_buf]
	mov	rsi, REPL_LINE_MAX
	call	repl_read_line
	cmp	rax, REPL_EOF
	je	.repl_exit
	test	rax, rax
	jz	.repl_loop

	lea	rdi, [line_buf]
	lea	rsi, [token_ptrs]
	mov	rdx, REPL_MAX_TOKENS
	call	repl_tokenize
	mov	[token_count], rax
	test	rax, rax
	jz	.repl_loop

	lea	rdi, [token_ptrs]
	mov	rdi, [rdi]
	call	dispatch_command
	jmp	.repl_loop

.repl_exit:
	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

; rdi = command name
dispatch_command:
	push	rbx
	mov	rbx, rdi

	lea	rdi, [cmd_ping]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_ping

	lea	rdi, [cmd_set]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_set

	lea	rdi, [cmd_get]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_get

	lea	rdi, [cmd_exists]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_exists

	lea	rdi, [cmd_del]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_del

	lea	rdi, [cmd_dbsize]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_dbsize

	lea	rdi, [cmd_incr]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_incr

	lea	rdi, [cmd_decr]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_decr

	lea	rdi, [cmd_mget]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_mget

	lea	rdi, [cmd_keys]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_keys

	lea	rdi, [cmd_save]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_save

	lea	rdi, [cmd_load]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_load

	lea	rdi, [cmd_quit]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_quit

	lea	rdi, [cmd_exit]
	mov	rsi, rbx
	call	str_eq
	test	rax, rax
	jnz	cmd_do_quit

	jmp	repl_syntax_err

cmd_do_ping:
	call	repl_write_pong
	jmp	dispatch_done

cmd_do_set:
	cmp	qword [token_count], 3
	jne	repl_syntax_err
	mov	rdi, [token_ptrs + 16]
	call	str_parse_int64
	test	rbx, rbx
	jnz	cmd_set_str
	mov	rcx, rax
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [token_ptrs + 8]
	call	hashmap_str_put_int
	call	repl_write_ok
	jmp	dispatch_done
cmd_set_str:
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [token_ptrs + 8]
	mov	rcx, [token_ptrs + 16]
	call	hashmap_str_put_str
	call	repl_write_ok
	jmp	dispatch_done

cmd_do_get:
	cmp	qword [token_count], 2
	jne	repl_syntax_err
	lea	rdi, [storage_map]
	mov	rsi, [token_ptrs + 8]
	call	hashmap_str_get_entry
	test	rax, rax
	jz	cmd_get_nil
	call	print_entry_nl
	jmp	dispatch_done
cmd_get_nil:
	call	repl_write_nil
	jmp	dispatch_done

cmd_do_exists:
	cmp	qword [token_count], 2
	jne	repl_syntax_err
	lea	rdi, [storage_map]
	mov	rsi, [token_ptrs + 8]
	call	hashmap_str_contains
	call	print_int_nl
	jmp	dispatch_done

cmd_do_del:
	cmp	qword [token_count], 2
	jne	repl_syntax_err
	lea	rdi, [storage_map]
	mov	rsi, [token_ptrs + 8]
	call	hashmap_str_del
	call	print_int_nl
	jmp	dispatch_done

cmd_do_dbsize:
	cmp	qword [token_count], 1
	jne	repl_syntax_err
	mov	rax, qword [storage_map + HASHMAP_STR_SIZE_OFF]
	call	print_int_nl
	jmp	dispatch_done

cmd_do_incr:
	cmp	qword [token_count], 2
	jne	repl_syntax_err
	lea	rdi, [storage_map]
	mov	rsi, [token_ptrs + 8]
	call	hashmap_str_get_entry
	test	rax, rax
	jz	cmd_incr_new
	cmp	byte [rax + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_STR
	je	cmd_wrongtype
	mov	rbx, rax
	inc	qword [rbx + HASH_ENTRY_STR_VAL_OFF]
	mov	rax, [rbx + HASH_ENTRY_STR_VAL_OFF]
	call	print_int_nl
	jmp	dispatch_done
cmd_incr_new:
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [token_ptrs + 8]
	mov	rcx, 1
	call	hashmap_str_put_int
	mov	rax, 1
	call	print_int_nl
	jmp	dispatch_done

cmd_do_decr:
	cmp	qword [token_count], 2
	jne	repl_syntax_err
	lea	rdi, [storage_map]
	mov	rsi, [token_ptrs + 8]
	call	hashmap_str_get_entry
	test	rax, rax
	jz	cmd_decr_new
	cmp	byte [rax + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_STR
	je	cmd_wrongtype
	mov	rbx, rax
	dec	qword [rbx + HASH_ENTRY_STR_VAL_OFF]
	mov	rax, [rbx + HASH_ENTRY_STR_VAL_OFF]
	call	print_int_nl
	jmp	dispatch_done
cmd_decr_new:
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [token_ptrs + 8]
	mov	rcx, -1
	call	hashmap_str_put_int
	mov	rax, -1
	call	print_int_nl
	jmp	dispatch_done

cmd_do_mget:
	cmp	qword [token_count], 2
	jb	repl_syntax_err
	push	r12
	mov	r12, 1
.mget_loop:
	cmp	r12, [token_count]
	jae	.mget_done
	lea	rdi, [storage_map]
	mov	rsi, [token_ptrs + r12 * 8]
	call	hashmap_str_get_entry
	test	rax, rax
	jz	.mget_nil
	call	print_entry_nl
	jmp	.mget_next
.mget_nil:
	call	repl_write_nil
.mget_next:
	inc	r12
	jmp	.mget_loop
.mget_done:
	pop	r12
	jmp	dispatch_done

cmd_do_keys:
	cmp	qword [token_count], 1
	jne	repl_syntax_err
	lea	rdi, [storage_map]
	lea	rsi, [keys_print_cb]
	xor	rdx, rdx
	call	hashmap_str_foreach
	jmp	dispatch_done

cmd_do_save:
	cmp	qword [token_count], 2
	jne	repl_syntax_err
	mov	rdi, [token_ptrs + 8]
	open_file rdi, O_WRONLY or O_CREAT or O_TRUNC, 420
	cmp	rax, 0
	jl	repl_syntax_err
	mov	[save_fd], rax
	mov	rdi, [save_fd]
	lea	rsi, [dump_header]
	mov	rdx, dump_header_len
	mov	rax, SYS_write
	syscall
	lea	rdi, [storage_map]
	lea	rsi, [save_entry_cb]
	mov	rdx, [save_fd]
	call	hashmap_str_foreach
	mov	rdi, [save_fd]
	close_file rdi
	call	repl_write_ok
	jmp	dispatch_done

cmd_do_load:
	cmp	qword [token_count], 2
	jne	repl_syntax_err
	lea	rdi, [storage_map]
	call	hashmap_str_clear
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	call	gc_collect_hashmap_str
	mov	rdi, [token_ptrs + 8]
	open_file rdi, O_RDONLY, 0
	cmp	rax, 0
	jl	repl_syntax_err
	mov	[save_fd], rax
.load_loop:
	lea	rdi, [line_buf]
	mov	rsi, REPL_LINE_MAX
	mov	rdx, [save_fd]
	call	repl_read_line_fd
	cmp	rax, REPL_EOF
	je	.load_done
	test	rax, rax
	jz	.load_loop
	cmp	byte [line_buf], '#'
	je	.load_loop
	lea	rdi, [line_buf]
	lea	rsi, [load_tokens]
	mov	rdx, 4
	call	repl_tokenize
	cmp	rax, 3
	jb	.load_loop
	mov	rdi, [load_tokens]
	cmp	byte [rdi], 'I'
	je	.load_int
	cmp	byte [rdi], 'S'
	je	.load_str
	jmp	.load_loop
.load_int:
	mov	rdi, [load_tokens + 16]
	call	str_parse_int64
	test	rbx, rbx
	jnz	.load_loop
	mov	rcx, rax
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [load_tokens + 8]
	call	hashmap_str_put_int
	jmp	.load_loop
.load_str:
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [load_tokens + 8]
	mov	rcx, [load_tokens + 16]
	call	hashmap_str_put_str
	jmp	.load_loop
.load_done:
	mov	rdi, [save_fd]
	close_file rdi
	call	repl_write_ok
	jmp	dispatch_done

cmd_do_quit:
	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

cmd_wrongtype:
	call	repl_write_wrongtype
	jmp	dispatch_done

repl_syntax_err:
	call	repl_write_err

dispatch_done:
	pop	rbx
	ret

; rax = entry*
print_entry_nl:
	push	rbx
	mov	rbx, rax
	cmp	byte [rbx + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_INT
	jne	.pen_str
	mov	rax, [rbx + HASH_ENTRY_STR_VAL_OFF]
	call	print_int64
	jmp	.pen_nl
.pen_str:
	mov	rdi, [rbx + HASH_ENTRY_STR_VAL_OFF]
	call	print_cstr
.pen_nl:
	mov	al, 10
	call	print_char
	pop	rbx
	ret

; foreach cb: print key + newline
keys_print_cb:
	mov	rdi, rdi
	call	print_cstr
	mov	al, 10
	jmp	print_char

save_prefix_i db 'I ', 0
save_prefix_s db 'S ', 0
save_space db ' ', 0
save_nl db 10, 0

; foreach cb: rdx = fd, write dump line
save_entry_cb:
	push	r12
	push	r13
	push	r14
	mov	r12, [save_fd]
	mov	r13, rdi
	mov	r14, rsi
	mov	rdi, r12
	cmp	qword [r14 + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_INT
	je	.sec_int_prefix
	lea	rsi, [save_prefix_s]
	mov	rdx, 2
	jmp	.sec_write_prefix
.sec_int_prefix:
	lea	rsi, [save_prefix_i]
	mov	rdx, 2
.sec_write_prefix:
	mov	rax, SYS_write
	syscall
	mov	rdi, r13
	call	str_len
	mov	rsi, r13
	mov	rdx, rax
	mov	rdi, r12
	mov	rax, SYS_write
	syscall
	lea	rsi, [save_space]
	mov	rdx, 1
	mov	rdi, r12
	mov	rax, SYS_write
	syscall
	cmp	qword [r14 + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_INT
	je	.sec_int_val
.sec_str_val:
	mov	rdi, [r14 + HASH_ENTRY_STR_VAL_OFF]
	call	str_len
	mov	rsi, [r14 + HASH_ENTRY_STR_VAL_OFF]
	mov	rdx, rax
	mov	rdi, r12
	mov	rax, SYS_write
	syscall
	jmp	.sec_nl
.sec_int_val:
	mov	rax, qword [r14 + HASH_ENTRY_STR_VAL_OFF]
	push	rcx
	push	rax
	lea	r8, [int_digits + 20]
	mov	rdi, r8
.siv_loop:
	mov	rax, [rsp]
	xor	rdx, rdx
	mov	rcx, 10
	div	rcx
	mov	[rsp], rax
	add	dl, '0'
	dec	rdi
	mov	byte [rdi], dl
	test	rax, rax
	jnz	.siv_loop
	mov	rsi, rdi
	mov	rdx, r8
	sub	rdx, rdi
	mov	rdi, r12
	mov	rax, SYS_write
	syscall
	add	rsp, 8
	pop	rcx
	jmp	.sec_nl
.sec_nl:
	lea	rsi, [save_nl]
	mov	rdx, 1
	mov	rdi, r12
	mov	rax, SYS_write
	syscall
	mov	rbx, r14
	pop	r14
	pop	r13
	pop	r12
	ret

segment readable writeable

storage_heap rb HEAP_SIZE
storage_map rb HASHMAP_STR_SIZE

line_buf rb REPL_LINE_MAX
save_line rb REPL_LINE_MAX
token_ptrs rq REPL_MAX_TOKENS
load_tokens rq 4
token_count dq ?
save_fd dq ?

cmd_ping db 'PING', 0
cmd_set db 'SET', 0
cmd_get db 'GET', 0
cmd_exists db 'EXISTS', 0
cmd_del db 'DEL', 0
cmd_dbsize db 'DBSIZE', 0
cmd_incr db 'INCR', 0
cmd_decr db 'DECR', 0
cmd_mget db 'MGET', 0
cmd_keys db 'KEYS', 0
cmd_save db 'SAVE', 0
cmd_load db 'LOAD', 0
cmd_quit db 'QUIT', 0
cmd_exit db 'EXIT', 0

dump_header db '# miniredis v1', 10
dump_header_len = $ - dump_header

include "fasm/core/runtime_bss.inc"
runtime_print_bss
