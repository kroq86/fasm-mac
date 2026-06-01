; LeetCode 206. Reverse Linked List
; https://leetcode.com/problems/reverse-linked-list/
; Input: 1->2->3->4->5  =>  5 4 3 2 1

format ELF64 executable 3
include "fasm/core/platform.inc"

segment readable executable

include "fasm/core/mmap.inc"
include "fasm/core/value.inc"
include "fasm/core/heap.inc"
include "fasm/core/dynvec.inc"
include "fasm/core/listnode.inc"
include "fasm/core/print_value.inc"
include "fasm/core/leetcode.inc"

entry start

start:
	lea	rdi, [storage_heap]
	call	heap_init

	lea	rdi, [storage_heap]
	mov	rsi, 5
	call	vec_init
	mov	[storage_vec], rax

	lea	rdi, [storage_heap]
	mov	rsi, [storage_vec]
	lea	rdx, [test_vals]
	mov	rcx, test_vals_count
	call	vec_build_ints

	lea	rdi, [storage_heap]
	mov	rsi, [storage_vec]
	call	list_build_from_vec
	mov	[storage_head], rax

	mov	rdi, [storage_head]
	call	list_reverse
	mov	[storage_head], rax

	mov	rdi, [storage_head]
	call	list_print

	mov	al, 10
	call	print_char

	lea	rdi, [storage_heap]
	call	heap_destroy
	exit EXIT_SUCCESS

segment readable writeable

storage_heap rb HEAP_SIZE
storage_vec dq ?
storage_head dq ?

test_vals dq 1, 2, 3, 4, 5
test_vals_count = 5

include "fasm/core/runtime_bss.inc"
runtime_value_bss
