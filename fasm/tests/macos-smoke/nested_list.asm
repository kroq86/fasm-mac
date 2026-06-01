format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/dynvec.inc"
include "fasm/core/print_value.inc"

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init

	lea	rdi, [storage_heap]
	mov	rsi, 2
	call	vec_init
	mov	[storage_outer], rax

	lea	rbx, [storage_root]
	mov	byte [rbx + VALUE_TAG_OFF], VAL_ARRAY
	mov	[rbx + VALUE_DATA_OFF], rax

	lea	rdi, [storage_heap]
	mov	rsi, [storage_outer]
	mov	rdx, 2
	call	vec_append_array
	mov	[storage_inner], rax

	lea	rdi, [storage_heap]
	mov	rsi, [storage_inner]
	mov	rdx, 42
	call	vec_append_int

	lea	rdi, [storage_root]
	xor	rsi, rsi
	call	print_value

	mov	al, 10
	call	print_char

	lea	rdi, [storage_heap]
	call	heap_destroy

	exit EXIT_SUCCESS

segment readable writeable

storage_heap rb HEAP_SIZE
storage_root rb VALUE_SIZE
storage_outer dq ?
storage_inner dq ?

include "fasm/core/runtime_bss.inc"
runtime_value_bss
