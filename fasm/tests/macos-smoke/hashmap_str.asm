; Smoke: string hash map int + string values.

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/print_io.inc"
include "fasm/core/str.inc"
include "fasm/core/hashmap_str.inc"

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init

	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	mov	rdx, HASHMAP_DEFAULT_BUCKETS
	call	hashmap_str_init

	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	lea	rdx, [key_foo]
	mov	rcx, 42
	call	hashmap_str_put_int

	lea	rdi, [storage_heap]
	lea	rsi, [storage_map]
	lea	rdx, [key_bar]
	lea	rcx, [val_hello]
	call	hashmap_str_put_str

	lea	rdi, [storage_map]
	lea	rsi, [key_foo]
	call	hashmap_str_get_int
	call	print_int_sp

	lea	rdi, [storage_map]
	lea	rsi, [key_bar]
	call	hashmap_str_get_entry
	test	rax, rax
	jz	.sm_fail
	mov	rdi, [rax + HASH_ENTRY_STR_VAL_OFF]
	call	print_cstr
	mov	al, ' '
	call	print_char

	lea	rdi, [storage_map]
	lea	rsi, [key_foo]
	call	hashmap_str_contains
	call	print_int_nl

	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

.sm_fail:
	exit EXIT_FAILURE

segment readable writeable

storage_heap rb HEAP_SIZE
storage_map rb HASHMAP_STR_SIZE

key_foo db 'foo', 0
key_bar db 'bar', 0
val_hello db 'hello', 0

include "fasm/core/runtime_bss.inc"
runtime_print_bss
