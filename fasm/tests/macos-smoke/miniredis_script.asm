; Smoke: pipe scripted commands into miniredis and check stdout markers.
; Run: printf '...' | arch -x86_64 ./fasm/tests/macos-smoke/miniredis_script

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

	call	script_set_a
	call	script_get_a
	call	script_set_msg
	call	script_get_msg
	call	script_dbsize

	exit EXIT_SUCCESS

script_set_a:
	lea	rdi, [line_set_a]
	mov	[script_line], rdi
	jmp	script_dispatch

script_get_a:
	lea	rdi, [line_get_a]
	mov	[script_line], rdi
	jmp	script_dispatch

script_set_msg:
	lea	rdi, [line_set_msg]
	mov	[script_line], rdi
	jmp	script_dispatch

script_get_msg:
	lea	rdi, [line_get_msg]
	mov	[script_line], rdi
	jmp	script_dispatch

script_dbsize:
	lea	rdi, [line_dbsize]
	mov	[script_line], rdi
	jmp	script_dispatch

script_dispatch:
	mov	rdi, [script_line]
	lea	rsi, [token_ptrs]
	mov	rdx, REPL_MAX_TOKENS
	call	repl_tokenize
	mov	[token_count], rax
	test	rax, rax
	jz	script_dispatch_done

	mov	rdi, [token_ptrs]
	lea	rsi, [cmd_set]
	call	str_eq
	test	rax, rax
	jnz	script_do_set

	mov	rdi, [token_ptrs]
	lea	rsi, [cmd_get]
	call	str_eq
	test	rax, rax
	jnz	script_do_get

	mov	rdi, [token_ptrs]
	lea	rsi, [cmd_dbsize]
	call	str_eq
	test	rax, rax
	jnz	script_do_dbsize

script_dispatch_done:
	ret

script_do_set:
	cmp	qword [token_count], 3
	jne	script_dispatch_done
	mov	rdi, [token_ptrs + 16]
	call	str_parse_int64
	test	rbx, rbx
	jnz	script_set_str
	mov	rcx, rax
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [token_ptrs + 8]
	call	hashmap_str_put_int
	jmp	script_dispatch_done
script_set_str:
	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, [token_ptrs + 8]
	mov	rcx, [token_ptrs + 16]
	call	hashmap_str_put_str
	ret

script_do_get:
	cmp	qword [token_count], 2
	jne	script_dispatch_done
	lea	rdi, [storage_map]
	mov	rsi, [token_ptrs + 8]
	call	hashmap_str_get_entry
	test	rax, rax
	jz	script_get_nil
	cmp	qword [rax + HASH_ENTRY_STR_TYPE_OFF], HM_VAL_INT
	jne	script_get_str
	mov	rax, [rax + HASH_ENTRY_STR_VAL_OFF]
	call	print_int_nl
	ret
script_get_str:
	mov	rdi, [rax + HASH_ENTRY_STR_VAL_OFF]
	call	print_cstr
	mov	al, 10
	call	print_char
	ret
script_get_nil:
	call	repl_write_nil
	ret

script_do_dbsize:
	lea	rdi, [storage_map]
	mov	rax, [rdi + HASHMAP_STR_SIZE_OFF]
	call	print_int_nl
	ret

segment readable writeable

storage_heap rb HEAP_SIZE
storage_map rb HASHMAP_STR_SIZE

token_ptrs rq REPL_MAX_TOKENS
token_count dq ?
script_line dq ?

line_set_a db 'SET a 42', 0
line_get_a db 'GET a', 0
line_set_msg db 'SET msg hello', 0
line_get_msg db 'GET msg', 0
line_dbsize db 'DBSIZE', 0

cmd_set db 'SET', 0
cmd_get db 'GET', 0
cmd_dbsize db 'DBSIZE', 0

include "fasm/core/runtime_bss.inc"
runtime_print_bss
